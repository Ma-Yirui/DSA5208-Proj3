// MPI helpers shared by the distributed versions (V2, V3, V4).
#pragma once
#include <mpi.h>
#include "config.hpp"
#include "decomp.hpp"

// Number of distinct nodes (shared-memory domains) in comm.
int count_nodes(MPI_Comm comm);

// Collects every rank's interior rows (local rows 1..local_ny of `local`, a host array of
// (local_ny + 2) * nx doubles) on rank 0, adds the fixed top/bottom rows and writes c.out.
void gather_and_write(const Config& c, const Decomp& d, const double* local, int rank,
                      int nprocs, MPI_Comm comm);
