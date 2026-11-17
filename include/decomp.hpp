// 1D row-strip decomposition of the interior rows over MPI ranks.
#pragma once
#include <algorithm>

struct Decomp {
    long local_ny;    // interior rows owned by this rank
    long row_offset;  // global row index of local row 1
    int up, down;     // neighbour ranks, or MPI_PROC_NULL at the global edges
};

// Interior rows N = ny - 2 are split over nprocs ranks; the first N % nprocs ranks get one
// extra row. `proc_null` is passed in so this header does not depend on MPI.
inline Decomp make_decomp(long ny, int rank, int nprocs, int proc_null) {
    long n = ny - 2, base = n / nprocs, rem = n % nprocs;
    Decomp d;
    d.local_ny = base + (rank < rem ? 1 : 0);
    d.row_offset = 1 + rank * base + std::min<long>(rank, rem);
    d.up = rank > 0 ? rank - 1 : proc_null;
    d.down = rank < nprocs - 1 ? rank + 1 : proc_null;
    return d;
}
