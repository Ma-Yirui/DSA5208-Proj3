# Builds every version into build/.
#   Laptop:  make
#   Vanda:   make ARCH=sapphirerapids        (after: source scripts/vanda_env.sh)
# V4 (CUDA) is built only when nvcc is found. No -ffast-math anywhere: it reorders
# floating-point sums and breaks bit-exact validation.

CXX    ?= g++
MPICXX ?= mpicxx
NVCC   ?= nvcc
ARCH   ?= native
CUDA_ARCH ?= sm_86
CXXFLAGS  = -O3 -march=$(ARCH) -std=c++17 -Wall -Wextra -Wno-unknown-pragmas -Iinclude
OMPFLAGS  = -fopenmp
# -fmad=false: no fused multiply-adds, so GPU arithmetic matches the CPU versions exactly.
NVCCFLAGS = -O3 -arch=$(CUDA_ARCH) -std=c++17 -fmad=false -Iinclude -ccbin $(MPICXX) \
            -Xcompiler -march=$(ARCH),-Wall

COMMON  = src/common/args.cpp src/common/io.cpp src/common/report.cpp
MPIUTIL = src/common/mpi_util.cpp
HEADERS = $(wildcard include/*.hpp)
BINS    = build/v0_serial build/v1_openmp build/v2_mpi build/v3_hybrid build/validate
ifneq ($(shell command -v $(NVCC) 2> /dev/null),)
BINS   += build/v4_mpi_cuda
endif

all: $(BINS)

build:
	mkdir -p build

build/v0_serial: src/solver_shared.cpp $(COMMON) $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -DHEAT_VERSION='"v0_serial"' -o $@ src/solver_shared.cpp $(COMMON)

build/v1_openmp: src/solver_shared.cpp $(COMMON) $(HEADERS) | build
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DHEAT_VERSION='"v1_openmp"' -o $@ src/solver_shared.cpp $(COMMON)

build/v2_mpi: src/solver_mpi.cpp $(COMMON) $(MPIUTIL) $(HEADERS) | build
	$(MPICXX) $(CXXFLAGS) -DHEAT_VERSION='"v2_mpi"' -o $@ src/solver_mpi.cpp $(COMMON) $(MPIUTIL)

build/v3_hybrid: src/solver_mpi.cpp $(COMMON) $(MPIUTIL) $(HEADERS) | build
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -DHEAT_VERSION='"v3_hybrid"' -o $@ src/solver_mpi.cpp $(COMMON) $(MPIUTIL)

build/v4_mpi_cuda: src/solver_cuda.cu $(COMMON) $(MPIUTIL) $(HEADERS) | build
	$(NVCC) $(NVCCFLAGS) -DHEAT_VERSION='"v4_mpi_cuda"' -o $@ src/solver_cuda.cu $(COMMON) $(MPIUTIL)

build/validate: tools/validate.cpp src/common/io.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -o $@ tools/validate.cpp src/common/io.cpp

clean:
	rm -rf build

.PHONY: all clean
