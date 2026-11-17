// Grid conventions, initial/boundary values and the 5-point update shared by all CPU versions.
//
// Storage is row-major: cell (i, j) of a grid with nx columns lives at i * nx + j.
// Global rows 0 and ny-1 and columns 0 and nx-1 are fixed (Dirichlet) boundary cells.
#pragma once
#include <cmath>
#include "config.hpp"

// The only definition of initial and boundary values; takes GLOBAL coordinates.
// Top row (including its corners) = t_hot, every other cell = t_cold.
inline double init_value(long gi, long /*gj*/, const Config& c) {
    return gi == 0 ? c.t_hot : c.t_cold;
}

// Updates interior columns of local row i. The summation order ((N + S) + W) + E is part of
// the specification: every version uses it, so all versions produce bit-identical results.
inline void jacobi_row(const double* __restrict cur, double* __restrict next, long nx, long i) {
    const double* up = cur + (i - 1) * nx;
    const double* mid = cur + i * nx;
    const double* dn = cur + (i + 1) * nx;
    double* out = next + i * nx;
    for (long j = 1; j < nx - 1; ++j)
        out[j] = 0.25 * (((up[j] + dn[j]) + mid[j - 1]) + mid[j + 1]);
}

// Max |next - cur| over the interior columns of row i.
inline double residual_row(const double* cur, const double* next, long nx, long i) {
    double r = 0.0;
    for (long j = 1; j < nx - 1; ++j)
        r = std::fmax(r, std::fabs(next[i * nx + j] - cur[i * nx + j]));
    return r;
}
