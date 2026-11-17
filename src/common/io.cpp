#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "io.hpp"

void write_grid(const std::string& path, const double* data, long nx, long ny) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::perror(path.c_str()); std::exit(1); }
    std::int64_t dims[2] = {nx, ny};
    size_t n = static_cast<size_t>(nx) * ny;
    if (std::fwrite(dims, sizeof dims, 1, f) != 1 || std::fwrite(data, sizeof(double), n, f) != n) {
        std::perror(path.c_str());
        std::exit(1);
    }
    std::fclose(f);
}

bool read_grid(const std::string& path, std::vector<double>& data, long& nx, long& ny) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::int64_t dims[2];
    bool ok = std::fread(dims, sizeof dims, 1, f) == 1 && dims[0] > 0 && dims[1] > 0;
    if (ok) {
        nx = dims[0];
        ny = dims[1];
        data.resize(static_cast<size_t>(nx) * ny);
        ok = std::fread(data.data(), sizeof(double), data.size(), f) == data.size();
    }
    std::fclose(f);
    return ok;
}
