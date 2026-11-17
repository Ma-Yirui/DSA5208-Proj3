#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "config.hpp"

static void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s [--nx N] [--ny N] [--iters N] [--check-every K] [--stop-tol X]\n"
        "          [--t-hot X] [--t-cold X] [--out FILE] [--header] [--bx N] [--by N]\n", prog);
    std::exit(2);
}

Config parse_args(int argc, char** argv) {
    Config c;
    for (int a = 1; a < argc; ++a) {
        std::string k = argv[a];
        if (k == "--header") { c.header = true; continue; }
        if (a + 1 >= argc) usage(argv[0]);
        const char* v = argv[++a];
        try {
            if (k == "--nx") c.nx = std::stol(v);
            else if (k == "--ny") c.ny = std::stol(v);
            else if (k == "--iters") c.iters = std::stol(v);
            else if (k == "--check-every") c.check_every = std::stol(v);
            else if (k == "--stop-tol") c.stop_tol = std::stod(v);
            else if (k == "--t-hot") c.t_hot = std::stod(v);
            else if (k == "--t-cold") c.t_cold = std::stod(v);
            else if (k == "--out") c.out = v;
            else if (k == "--bx") c.bx = std::stoi(v);
            else if (k == "--by") c.by = std::stoi(v);
            else usage(argv[0]);
        } catch (...) {
            usage(argv[0]);
        }
    }
    if (c.nx < 3 || c.ny < 3 || c.iters < 0 || c.check_every < 0) {
        std::fprintf(stderr, "need nx >= 3, ny >= 3, iters >= 0, check-every >= 0\n");
        std::exit(2);
    }
    if (c.bx < 1 || c.by < 1 || c.bx * c.by > 1024) {
        std::fprintf(stderr, "need bx, by >= 1 and bx * by <= 1024\n");
        std::exit(2);
    }
    if (c.stop_tol > 0.0 && c.check_every == 0) {
        std::fprintf(stderr, "--stop-tol needs --check-every > 0\n");
        std::exit(2);
    }
    return c;
}
