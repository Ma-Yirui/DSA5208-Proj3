// Run configuration shared by every version (same CLI everywhere).
#pragma once
#include <string>

struct Config {
    long nx = 1024;          // global columns, including the fixed boundary ring
    long ny = 1024;          // global rows, including the fixed boundary ring
    long iters = 1000;       // maximum number of Jacobi iterations
    long check_every = 0;    // residual check period (0 = never)
    double stop_tol = 0.0;   // stop when residual < stop_tol (needs check_every > 0)
    double t_hot = 100.0;    // top edge temperature
    double t_cold = 0.0;     // other edges and initial interior temperature
    std::string out;         // binary output file ("" = no output)
    bool header = false;     // print the CSV header line before the result line
    int bx = 32, by = 8;     // CUDA thread-block size (V4 only)
};

// Parses --nx --ny --iters --check-every --stop-tol --t-hot --t-cold --out --header --bx --by.
// Prints usage and exits on unknown or malformed arguments.
Config parse_args(int argc, char** argv);
