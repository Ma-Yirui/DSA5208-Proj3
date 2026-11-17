// Wall-clock time in seconds (monotonic).
#pragma once
#include <chrono>

inline double wall_time() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}
