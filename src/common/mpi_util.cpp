#include <climits>
#include <cstdio>
#include <initializer_list>
#include <vector>
#include "grid.hpp"
#include "io.hpp"
#include "mpi_util.hpp"

int count_nodes(MPI_Comm comm) {
    MPI_Comm local;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int lrank, first, nodes;
    MPI_Comm_rank(local, &lrank);
    first = lrank == 0 ? 1 : 0;
    MPI_Allreduce(&first, &nodes, 1, MPI_INT, MPI_SUM, comm);
    MPI_Comm_free(&local);
    return nodes;
}

void gather_and_write(const Config& c, const Decomp& d, const double* local, int rank,
                      int nprocs, MPI_Comm comm) {
    const long nx = c.nx, ny = c.ny;
    if (static_cast<double>(nx) * ny > INT_MAX) {
        if (rank == 0) std::fprintf(stderr, "grid too large to gather; skipping --out\n");
        return;
    }
    std::vector<double> global;
    std::vector<int> counts, displs;
    if (rank == 0) {
        global.resize(static_cast<size_t>(nx) * ny);
        for (int p = 0; p < nprocs; ++p) {
            Decomp dp = make_decomp(ny, p, nprocs, MPI_PROC_NULL);
            counts.push_back(static_cast<int>(dp.local_ny * nx));
            displs.push_back(static_cast<int>(dp.row_offset * nx));
        }
        for (long i : {0L, ny - 1})
            for (long j = 0; j < nx; ++j) global[i * nx + j] = init_value(i, j, c);
    }
    MPI_Gatherv(local + nx, static_cast<int>(d.local_ny * nx), MPI_DOUBLE, global.data(),
                counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    if (rank == 0) write_grid(c.out, global.data(), nx, ny);
}
