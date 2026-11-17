// V0 (serial) and V1 (OpenMP) are built from this file.
// V0 is compiled without -fopenmp, so every pragma is ignored and the code is plain serial C++;
// V1 is the same code with OpenMP enabled, which guarantees identical arithmetic.
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <utility>
#include "config.hpp"
#include "grid.hpp"
#include "io.hpp"
#include "report.hpp"
#include "timer.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef HEAT_VERSION
#define HEAT_VERSION "v0_serial"
#endif

int main(int argc, char** argv) {
    const Config c = parse_args(argc, argv);
    const long nx = c.nx, ny = c.ny;
    const size_t n = static_cast<size_t>(nx) * ny;

    // new double[] leaves the memory untouched, so the first write below decides page placement.
    std::unique_ptr<double[]> a(new double[n]), b(new double[n]);
    double* cur = a.get();
    double* next = b.get();

    // First touch: each thread initialises the interior rows it will later update (same
    // static schedule and bounds as the update loop). Both buffers hold the boundary values.
    #pragma omp parallel for schedule(static)
    for (long i = 1; i < ny - 1; ++i)
        for (long j = 0; j < nx; ++j) cur[i * nx + j] = next[i * nx + j] = init_value(i, j, c);
    for (long i : {0L, ny - 1})
        for (long j = 0; j < nx; ++j) cur[i * nx + j] = next[i * nx + j] = init_value(i, j, c);

    RunResult r;
    r.version = HEAT_VERSION;
#ifdef _OPENMP
    r.threads = omp_get_max_threads();
#endif

    long it = 0;
    const double t0 = wall_time();
    while (it < c.iters) {
        #pragma omp parallel for schedule(static)
        for (long i = 1; i < ny - 1; ++i) jacobi_row(cur, next, nx, i);
        ++it;

        const bool check = c.check_every > 0 && it % c.check_every == 0;
        if (check) {
            double res = 0.0;
            #pragma omp parallel for schedule(static) reduction(max : res)
            for (long i = 1; i < ny - 1; ++i) res = std::fmax(res, residual_row(cur, next, nx, i));
            r.residual = res;
        }
        std::swap(cur, next);  // cur now holds the newest values
        if (check && r.residual < c.stop_tol) break;
    }
    r.t_total = r.t_compute = wall_time() - t0;
    r.iters_done = it;

    print_result(c, r);
    if (!c.out.empty()) write_grid(c.out, cur, nx, ny);
    return 0;
}
