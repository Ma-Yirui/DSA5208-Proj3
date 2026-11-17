// Binary grid files: int64 nx, int64 ny, then nx*ny doubles in row-major order.
#pragma once
#include <string>
#include <vector>

void write_grid(const std::string& path, const double* data, long nx, long ny);
// Returns false if the file cannot be read or is truncated.
bool read_grid(const std::string& path, std::vector<double>& data, long& nx, long& ny);
