// V2 (MPI) and V3 (hybrid MPI + OpenMP) are built from this file.
// V2 is compiled without -fopenmp (pragmas ignored, one thread per rank); V3 enables OpenMP for
// the local update. MPI is only ever called outside parallel regions (MPI_THREAD_FUNNELED).
//
// Decomposition: interior rows are split into horizontal strips, one per rank (decomp.hpp).
// Each rank stores local_ny interior rows plus one halo row above and below. At the global top
// and bottom the neighbour is MPI_PROC_NULL, so those halo slots keep the fixed boundary rows.
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <utility>
#include "config.hpp"
#include "decomp.hpp"
#include "grid.hpp"
#include "mpi_util.hpp"
#include "report.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef HEAT_VERSION
#define HEAT_VERSION "v2_mpi"
#endif

// Rows are contiguous in row-major storage, so each halo is sent as-is: no packing needed.
static void halo_exchange(double* g, const Decomp& d, long nx, MPI_Comm comm) {
    const int n = static_cast<int>(nx);
    // First interior row goes up; the bottom halo arrives from below.
    MPI_Sendrecv(g + 1 * nx, n, MPI_DOUBLE, d.up, 0,
                 g + (d.local_ny + 1) * nx, n, MPI_DOUBLE, d.down, 0, comm, MPI_STATUS_IGNORE);
    // Last interior row goes down; the top halo arrives from above.
    MPI_Sendrecv(g + d.local_ny * nx, n, MPI_DOUBLE, d.down, 1,
                 g, n, MPI_DOUBLE, d.up, 1, comm, MPI_STATUS_IGNORE);
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI library lacks MPI_THREAD_FUNNELED support\n");
        MPI_Abort(comm, 1);
    }

    const Config c = parse_args(argc, argv);
    if (c.ny - 2 < nprocs) {
        if (rank == 0) std::fprintf(stderr, "need at least one interior row per rank\n");
        MPI_Abort(comm, 2);
    }
    const long nx = c.nx;
    const Decomp d = make_decomp(c.ny, rank, nprocs, MPI_PROC_NULL);
    const size_t n = static_cast<size_t>(d.local_ny + 2) * nx;

    std::unique_ptr<double[]> a(new double[n]), b(new double[n]);
    double* cur = a.get();
    double* next = b.get();

    // First touch of the interior rows by the threads that update them; halo rows serially.
    // Local row li is global row row_offset - 1 + li.
    #pragma omp parallel for schedule(static)
    for (long li = 1; li <= d.local_ny; ++li)
        for (long j = 0; j < nx; ++j)
            cur[li * nx + j] = next[li * nx + j] = init_value(d.row_offset - 1 + li, j, c);
    for (long li : {0L, d.local_ny + 1})
        for (long j = 0; j < nx; ++j)
            cur[li * nx + j] = next[li * nx + j] = init_value(d.row_offset - 1 + li, j, c);

    RunResult r;
    r.version = HEAT_VERSION;
    r.ranks = nprocs;
    r.nodes = count_nodes(comm);
#ifdef _OPENMP
    r.threads = omp_get_max_threads();
#endif
    double t_comm = 0.0, t_comp = 0.0;

    long it = 0;
    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();
    while (it < c.iters) {
        const double tc0 = MPI_Wtime();
        halo_exchange(cur, d, nx, comm);
        const double tc1 = MPI_Wtime();
        #pragma omp parallel for schedule(static)
        for (long li = 1; li <= d.local_ny; ++li) jacobi_row(cur, next, nx, li);
        const double tc2 = MPI_Wtime();
        t_comm += tc1 - tc0;
        t_comp += tc2 - tc1;
        ++it;

        const bool check = c.check_every > 0 && it % c.check_every == 0;
        if (check) {
            const double tr0 = MPI_Wtime();
            double local = 0.0;
            #pragma omp parallel for schedule(static) reduction(max : local)
            for (long li = 1; li <= d.local_ny; ++li)
                local = std::fmax(local, residual_row(cur, next, nx, li));
            const double tr1 = MPI_Wtime();
            MPI_Allreduce(&local, &r.residual, 1, MPI_DOUBLE, MPI_MAX, comm);
            t_comp += tr1 - tr0;
            t_comm += MPI_Wtime() - tr1;
        }
        std::swap(cur, next);
        if (check && r.residual < c.stop_tol) break;  // same decision on every rank
    }
    const double mine[3] = {MPI_Wtime() - t0, t_comp, t_comm};
    double worst[3];
    MPI_Reduce(mine, worst, 3, MPI_DOUBLE, MPI_MAX, 0, comm);  // report the slowest rank

    if (rank == 0) {
        r.t_total = worst[0];
        r.t_compute = worst[1];
        r.t_comm = worst[2];
        r.iters_done = it;
        print_result(c, r);
    }
    if (!c.out.empty()) gather_and_write(c, d, cur, rank, nprocs, comm);

    MPI_Finalize();
    return 0;
}
