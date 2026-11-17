# Vanda environment for building and running. Use: source scripts/vanda_env.sh
# The same modules must be loaded when building and inside every job.
source /etc/profile.d/rec_modules.sh 2>/dev/null
module purge 2>/dev/null
module load gcc/13.1.0 ucx/1.10.1 openmpi/gcc/64/4.1.8 cuda12.4/toolkit/12.4.1
export OMPI_CXX=g++                                    # mpicxx uses the gcc 13.1 just loaded
export OMPI_MCA_mca_base_component_show_load_errors=0  # hide harmless libpbs.so warnings
