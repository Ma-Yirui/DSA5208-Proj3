// V4: MPI + CUDA. One MPI rank per GPU, using the same 1D row-strip decomposition as V2/V3.
//
// The whole local grid stays on the GPU for the entire run. Per iteration only the two halo
// rows travel, host-staged because Vanda's OpenMPI is not CUDA-aware:
//   device -> pinned host buffer -> MPI_Sendrecv -> pinned host buffer -> device.
// The steps run synchronously, so each phase (copy, MPI, kernel) is timed separately.
#include <mpi.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>
#include "config.hpp"
#include "decomp.hpp"
#include "grid.hpp"
#include "mpi_util.hpp"
#include "report.hpp"

#ifndef HEAT_VERSION
#define HEAT_VERSION "v4_mpi_cuda"
#endif

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            std::fprintf(stderr, "CUDA error \"%s\" at %s:%d\n", cudaGetErrorString(err_), \
                         __FILE__, __LINE__);                                         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// One thread per interior cell of local rows 1..local_ny. threadIdx.x runs along a row, so
// consecutive threads of a warp read consecutive addresses (coalesced global memory access).
// The summation order ((N + S) + W) + E matches the CPU versions, so results are bit-identical.
__global__ void jacobi_step(const double* __restrict__ cur, double* __restrict__ next,
                            long nx, long local_ny) {
    const long j = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    const long i = 1 + blockIdx.y * static_cast<long>(blockDim.y) + threadIdx.y;
    if (i <= local_ny && j >= 1 && j <= nx - 2) {
        const long k = i * nx + j;
        next[k] = 0.25 * (((cur[k - nx] + cur[k + nx]) + cur[k - 1]) + cur[k + 1]);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    const Config c = parse_args(argc, argv);
    if (c.check_every > 0) {
        if (rank == 0) std::fprintf(stderr, "V4 does not implement residual checks\n");
        MPI_Abort(comm, 2);
    }
    if (c.ny - 2 < nprocs) {
        if (rank == 0) std::fprintf(stderr, "need at least one interior row per rank\n");
        MPI_Abort(comm, 2);
    }

    // GPU selection: the k-th rank on a node uses GPU k (modulo the GPUs visible to it).
    MPI_Comm node_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int lrank, lsize, ndev = 0;
    MPI_Comm_rank(node_comm, &lrank);
    MPI_Comm_size(node_comm, &lsize);
    MPI_Comm_free(&node_comm);
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    CUDA_CHECK(cudaSetDevice(lrank % ndev));

    RunResult r;
    r.version = HEAT_VERSION;
    r.ranks = nprocs;
    r.nodes = count_nodes(comm);
    int node_gpus = lrank == 0 ? std::min(lsize, ndev) : 0;
    MPI_Allreduce(MPI_IN_PLACE, &node_gpus, 1, MPI_INT, MPI_SUM, comm);
    r.gpus = node_gpus;

    const long nx = c.nx;
    const Decomp d = make_decomp(c.ny, rank, nprocs, MPI_PROC_NULL);
    const size_t n = static_cast<size_t>(d.local_ny + 2) * nx;
    const size_t row_bytes = nx * sizeof(double);

    // Initialise on the host (local row li = global row row_offset - 1 + li), then copy to
    // BOTH device buffers so the fixed boundary values survive every swap.
    std::vector<double> host(n);
    for (long li = 0; li <= d.local_ny + 1; ++li)
        for (long j = 0; j < nx; ++j) host[li * nx + j] = init_value(d.row_offset - 1 + li, j, c);
    double *d_cur, *d_next;
    CUDA_CHECK(cudaMalloc(&d_cur, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_next, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_cur, host.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_next, host.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // Pinned (page-locked) halo buffers: faster PCIe transfers than pageable memory.
    double *send_up, *send_dn, *recv_up, *recv_dn;
    CUDA_CHECK(cudaMallocHost(&send_up, row_bytes));
    CUDA_CHECK(cudaMallocHost(&send_dn, row_bytes));
    CUDA_CHECK(cudaMallocHost(&recv_up, row_bytes));
    CUDA_CHECK(cudaMallocHost(&recv_dn, row_bytes));

    const dim3 block(c.bx, c.by);
    const dim3 grid((nx + c.bx - 1) / c.bx, (d.local_ny + c.by - 1) / c.by);
    const int count = static_cast<int>(nx);

    // Untimed warm-up launch (loads the kernel). It writes only d_next's interior, which the
    // first real iteration overwrites.
    jacobi_step<<<grid, block>>>(d_cur, d_next, nx, d.local_ny);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double t_copy = 0.0, t_mpi = 0.0, t_kernel = 0.0;
    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();
    for (long it = 0; it < c.iters; ++it) {
        const double ta = MPI_Wtime();
        if (d.up != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpy(send_up, d_cur + nx, row_bytes, cudaMemcpyDeviceToHost));
        if (d.down != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpy(send_dn, d_cur + d.local_ny * nx, row_bytes, cudaMemcpyDeviceToHost));
        const double tb = MPI_Wtime();
        MPI_Sendrecv(send_up, count, MPI_DOUBLE, d.up, 0, recv_dn, count, MPI_DOUBLE, d.down, 0,
                     comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_dn, count, MPI_DOUBLE, d.down, 1, recv_up, count, MPI_DOUBLE, d.up, 1,
                     comm, MPI_STATUS_IGNORE);
        const double tc = MPI_Wtime();
        // At the global edges nothing was received: keep the fixed boundary row on the device.
        if (d.down != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpy(d_cur + (d.local_ny + 1) * nx, recv_dn, row_bytes, cudaMemcpyHostToDevice));
        if (d.up != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpy(d_cur, recv_up, row_bytes, cudaMemcpyHostToDevice));
        const double td = MPI_Wtime();
        jacobi_step<<<grid, block>>>(d_cur, d_next, nx, d.local_ny);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());  // kernel must finish before its halos are copied
        const double te = MPI_Wtime();
        t_copy += (tb - ta) + (td - tc);
        t_mpi += tc - tb;
        t_kernel += te - td;
        std::swap(d_cur, d_next);
    }
    const double mine[4] = {MPI_Wtime() - t0, t_kernel, t_mpi, t_copy};
    double worst[4];
    MPI_Reduce(mine, worst, 4, MPI_DOUBLE, MPI_MAX, 0, comm);  // report the slowest rank

    if (rank == 0) {
        r.t_total = worst[0];
        r.t_compute = worst[1];
        r.t_comm = worst[2];
        r.t_copy = worst[3];
        r.iters_done = c.iters;
        print_result(c, r);
    }
    if (!c.out.empty()) {
        CUDA_CHECK(cudaMemcpy(host.data(), d_cur, n * sizeof(double), cudaMemcpyDeviceToHost));
        gather_and_write(c, d, host.data(), rank, nprocs, comm);
    }

    CUDA_CHECK(cudaFree(d_cur));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFreeHost(send_up));
    CUDA_CHECK(cudaFreeHost(send_dn));
    CUDA_CHECK(cudaFreeHost(recv_up));
    CUDA_CHECK(cudaFreeHost(recv_dn));
    MPI_Finalize();
    return 0;
}
