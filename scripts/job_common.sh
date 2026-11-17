# Shared setup for the benchmark jobs (E2-E4). Use: source scripts/job_common.sh
# Provides: NCORES, S0/S1 (allocated cores per socket), OMP_PLACES in socket order (for V1),
#           pow2_upto N, run_mpi R T BINARY ARGS..., csv_init FILE, tag (appends alloc_cores).
cd "$PBS_O_WORKDIR"
[ -x build/v3_hybrid ] || { echo "Build first: source scripts/vanda_env.sh && make ARCH=sapphirerapids"; exit 1; }
source scripts/vanda_env.sh

echo "== node: $(hostname)   job: ${PBS_JOBID:-none}"
lscpu | grep -E "Model name|^CPU\(s\)|Socket|NUMA node"

# Cores this job may use (its PBS cpuset), counted per socket, and OpenMP places in socket order.
# On Vanda nodes CPU numbers alternate between sockets (even = socket 0, odd = socket 1).
ALLOWED=$(numactl --show 2>/dev/null | awk '/^physcpubind:/ {for (i = 2; i <= NF; i++) print $i}')
PLACES=""
S0=0
S1=0
for node in 0 1; do
    n=0
    for c in $(numactl -H 2>/dev/null | awk -v k=$node '$1 == "node" && $2 == k && $3 == "cpus:" {for (i = 4; i <= NF; i++) print $i}'); do
        if echo "$ALLOWED" | grep -qx "$c"; then PLACES="$PLACES{$c},"; n=$((n + 1)); fi
    done
    eval "S$node=$n"
done
if [ -n "$PLACES" ]; then
    export OMP_PLACES="${PLACES%,}"
    NCORES=$(echo "$ALLOWED" | wc -l)
else
    export OMP_PLACES=cores   # numactl unavailable: fall back to OS order
    NCORES=$(nproc)
fi
[ -n "${MAXCORES:-}" ] && [ "$MAXCORES" -lt "$NCORES" ] && NCORES=$MAXCORES   # for local testing
echo "== $NCORES cores (socket 0: $S0, socket 1: $S1)"

export OMP_PROC_BIND=close
mkdir -p results/raw

# Powers of two up to N: "1 2 4 ... N".
pow2_upto() {
    local t=1 out=""
    while [ $t -le "$1" ]; do out="$out $t"; t=$((t * 2)); done
    echo $out
}

# run_mpi R T BINARY ARGS...: R ranks x T threads, each rank bound to T cores.
# With cores on both sockets in equal numbers, half the ranks go to each socket; otherwise ranks
# are packed in core order. Inside a rank, threads use only that rank's cores (OMP_PLACES=cores).
# Set MPI_EXTRA (e.g. --report-bindings) to pass extra mpirun options.
run_mpi() {
    local r=$1 t=$2 map
    shift 2
    if [ "$S0" -gt 0 ] && [ "$S0" -eq "$S1" ] && [ "$r" -ge 2 ] && [ $((r % 2)) -eq 0 ]; then
        map="ppr:$((r / 2)):socket:PE=$t"
    else
        map="ppr:$r:node:PE=$t"
    fi
    OMP_NUM_THREADS=$t OMP_PLACES=cores mpirun --host "$(hostname):$NCORES" -np "$r" \
        --map-by "$map" --bind-to core ${MPI_EXTRA:-} \
        -x OMP_NUM_THREADS -x OMP_PLACES -x OMP_PROC_BIND "$@"
}

# CSV header (once) with an extra alloc_cores column; `tag` appends that column to result lines.
csv_init() {
    [ -s "$1" ] || echo "$(./build/v0_serial --nx 3 --ny 3 --iters 0 --header | head -1),alloc_cores" > "$1"
}
tag() { sed "s/\$/,$NCORES/"; }
