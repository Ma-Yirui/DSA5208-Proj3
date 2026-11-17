// One CSV result line per run, identical columns for every CPU version:
// version,host,nodes,ranks,threads,gpus,nx,ny,iters,t_total,t_compute,t_comm,residual
// GPU runs (gpus > 0) add a final t_copy column (host <-> device halo transfers); for them
// t_compute is kernel time and t_comm is MPI time only.
// Negative t_comm / residual mean "not applicable" and are printed as empty fields.
#pragma once
#include <string>
#include "config.hpp"

struct RunResult {
    std::string version;
    int nodes = 1, ranks = 1, threads = 1, gpus = 0;
    long iters_done = 0;
    double t_total = 0.0, t_compute = 0.0, t_comm = -1.0, residual = -1.0, t_copy = -1.0;
};

void print_result(const Config& c, const RunResult& r);
