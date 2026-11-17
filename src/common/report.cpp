#include <cstdio>
#include <unistd.h>
#include "report.hpp"

void print_result(const Config& c, const RunResult& r) {
    char host[256] = "unknown";
    gethostname(host, sizeof host - 1);
    const bool gpu = r.gpus > 0;
    if (c.header)
        std::printf("version,host,nodes,ranks,threads,gpus,nx,ny,iters,"
                    "t_total,t_compute,t_comm,residual%s\n", gpu ? ",t_copy" : "");
    std::printf("%s,%s,%d,%d,%d,%d,%ld,%ld,%ld,%.6f,%.6f,",
                r.version.c_str(), host, r.nodes, r.ranks, r.threads, r.gpus,
                c.nx, c.ny, r.iters_done, r.t_total, r.t_compute);
    if (r.t_comm >= 0) std::printf("%.6f", r.t_comm);
    std::printf(",");
    if (r.residual >= 0) std::printf("%.6e", r.residual);
    if (gpu) std::printf(",%.6f", r.t_copy);
    std::printf("\n");
    std::fflush(stdout);
}
