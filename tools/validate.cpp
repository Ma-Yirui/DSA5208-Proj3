// Compares two binary grids cell by cell and checks the boundary of the second one.
// usage: validate REF TEST [--tol X] [--t-hot X] [--t-cold X]
// Prints one summary line; exit code 0 = PASS, 1 = FAIL, 2 = usage / read error.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "config.hpp"
#include "grid.hpp"
#include "io.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s REF TEST [--tol X] [--t-hot X] [--t-cold X]\n", argv[0]);
        return 2;
    }
    double tol = 0.0;
    Config c;  // only t_hot / t_cold are used, for the boundary check
    for (int a = 3; a + 1 < argc; a += 2) {
        std::string k = argv[a];
        if (k == "--tol") tol = std::stod(argv[a + 1]);
        else if (k == "--t-hot") c.t_hot = std::stod(argv[a + 1]);
        else if (k == "--t-cold") c.t_cold = std::stod(argv[a + 1]);
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }

    std::vector<double> ref, test;
    long rnx, rny, tnx, tny;
    if (!read_grid(argv[1], ref, rnx, rny) || !read_grid(argv[2], test, tnx, tny)) {
        std::fprintf(stderr, "cannot read %s or %s\n", argv[1], argv[2]);
        return 2;
    }
    if (rnx != tnx || rny != tny) {
        std::printf("FAIL dims differ: %ldx%ld vs %ldx%ld\n", rnx, rny, tnx, tny);
        return 1;
    }

    double max_err = 0.0, sq = 0.0;
    long ndiff = 0;
    for (size_t k = 0; k < ref.size(); ++k) {
        double e = std::fabs(ref[k] - test[k]);
        if (ref[k] != test[k]) ++ndiff;
        max_err = std::fmax(max_err, e);
        sq += e * e;
    }
    const double rmse = std::sqrt(sq / ref.size());

    bool boundary_ok = true;
    for (long i = 0; i < tny; ++i)
        for (long j = 0; j < tnx; ++j)
            if ((i == 0 || i == tny - 1 || j == 0 || j == tnx - 1) &&
                test[i * tnx + j] != init_value(i, j, c))
                boundary_ok = false;

    const bool pass = max_err <= tol && boundary_ok;
    std::printf("%s max_abs_err=%.3e rmse=%.3e ndiff=%ld boundary_ok=%s\n",
                pass ? "PASS" : "FAIL", max_err, rmse, ndiff, boundary_ok ? "yes" : "no");
    return pass ? 0 : 1;
}
