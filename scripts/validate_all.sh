#!/bin/bash
# Validation matrix (guidelines §4). Run from the repo root after `make`.
#   Laptop: bash scripts/validate_all.sh
#   Vanda:  submitted by jobs/validate.pbs (sets MPIRUN)
# Writes results/validation/summary.csv and hand_check.txt; exit code 1 if any test fails.
set -u
MPIRUN=${MPIRUN:-"mpirun --oversubscribe --bind-to none"}
# Silence harmless OpenMPI notices (missing InfiniBand libs, no CMA in WSL); no effect on results.
export OMPI_MCA_mca_base_component_show_load_errors=${OMPI_MCA_mca_base_component_show_load_errors:-0}
export OMPI_MCA_btl_vader_single_copy_mechanism=${OMPI_MCA_btl_vader_single_copy_mechanism:-none}
PYTHON=${PYTHON:-python3}
B=build
D=results/validation
T=$D/tmp
mkdir -p "$T"
SUMMARY=$D/summary.csv
echo "test,nx,ny,iters,version,ranks,threads,result,max_abs_err,rmse,ndiff,boundary_ok" > "$SUMMARY"
fails=0

# check TEST NX NY ITERS VERSION RANKS THREADS REF OUT [validate options...]
check() {
    local test=$1 nx=$2 ny=$3 iters=$4 ver=$5 ranks=$6 thr=$7 ref=$8 out=$9
    shift 9
    local line
    line=$("$B/validate" "$ref" "$out" "$@")
    [ $? -eq 0 ] || fails=$((fails + 1))
    echo "$test $ver ${nx}x${ny} P=$ranks T=$thr : $line"
    # "PASS max_abs_err=0.000e+00 rmse=... ndiff=0 boundary_ok=yes" -> CSV fields
    echo "$test,$nx,$ny,$iters,$ver,$ranks,$thr,$(echo "$line" | sed -E 's/ [a-z_]+=/,/g')" >> "$SUMMARY"
}

run_v0() { "$B/v0_serial" "$@" > /dev/null; }
run_v1() { local thr=$1; shift; OMP_NUM_THREADS=$thr "$B/v1_openmp" "$@" > /dev/null; }
run_v2() { local p=$1; shift; $MPIRUN -np "$p" "$B/v2_mpi" "$@" > /dev/null; }
run_v3() {
    local p=$1 thr=$2; shift 2
    OMP_NUM_THREADS=$thr $MPIRUN -x OMP_NUM_THREADS -np "$p" "$B/v3_hybrid" "$@" > /dev/null
}

echo "== Test 1: hand check, V0 vs independent Python reference (must be bit-exact)"
for case in "6 6 1" "6 6 2" "9 7 50"; do
    set -- $case
    $PYTHON scripts/reference.py --nx "$1" --ny "$2" --iters "$3" --out "$T/py.bin"
    run_v0 --nx "$1" --ny "$2" --iters "$3" --out "$T/v0.bin"
    check hand "$1" "$2" "$3" v0_serial 1 1 "$T/py.bin" "$T/v0.bin"
done
$PYTHON scripts/reference.py --nx 6 --ny 6 --iters 2 --out "$T/py.bin" --print > "$D/hand_check.txt"

echo "== Test 2: uniform boundary (all edges 50) must stay exactly 50"
$PYTHON scripts/reference.py --nx 32 --ny 32 --iters 500 --t-hot 50 --t-cold 50 --out "$T/py.bin"
run_v0 --nx 32 --ny 32 --iters 500 --t-hot 50 --t-cold 50 --out "$T/v0.bin"
check uniform 32 32 500 v0_serial 1 1 "$T/py.bin" "$T/v0.bin" --t-hot 50 --t-cold 50

echo "== Test 3: cross-version matrix vs V0 (must be bit-exact)"
for case in "64 64 200" "129 77 300" "1024 1024 50"; do
    set -- $case
    nx=$1 ny=$2 it=$3
    args=(--nx "$nx" --ny "$ny" --iters "$it")
    run_v0 "${args[@]}" --out "$T/ref.bin"
    for thr in 1 2 4 8; do
        run_v1 "$thr" "${args[@]}" --out "$T/t.bin"
        check matrix "$nx" "$ny" "$it" v1_openmp 1 "$thr" "$T/ref.bin" "$T/t.bin"
    done
    for p in 1 2 3 4 7; do
        run_v2 "$p" "${args[@]}" --out "$T/t.bin"
        check matrix "$nx" "$ny" "$it" v2_mpi "$p" 1 "$T/ref.bin" "$T/t.bin"
    done
    for pt in "2 2" "3 4" "4 2" "7 2"; do
        set -- $pt
        run_v3 "$1" "$2" "${args[@]}" --out "$T/t.bin"
        check matrix "$nx" "$ny" "$it" v3_hybrid "$1" "$2" "$T/ref.bin" "$T/t.bin"
    done
done

echo "== Test 4: residual stopping (same iteration count and same grid in every version)"
args=(--nx 129 --ny 77 --iters 100000 --check-every 10 --stop-tol 1e-3)
iters_of() { tail -1 | cut -d, -f9; }
i0=$("$B/v0_serial" "${args[@]}" --out "$T/ref.bin" | iters_of)
i1=$(OMP_NUM_THREADS=4 "$B/v1_openmp" "${args[@]}" --out "$T/t1.bin" | iters_of)
i2=$($MPIRUN -np 3 "$B/v2_mpi" "${args[@]}" --out "$T/t2.bin" | iters_of)
i3=$(OMP_NUM_THREADS=2 $MPIRUN -x OMP_NUM_THREADS -np 3 "$B/v3_hybrid" "${args[@]}" --out "$T/t3.bin" | iters_of)
echo "iterations to residual < 1e-3: v0=$i0 v1=$i1 v2=$i2 v3=$i3"
if [ -n "$i0" ] && [ "$i0" = "$i1" ] && [ "$i0" = "$i2" ] && [ "$i0" = "$i3" ]; then
    echo "residual,129,77,$i0,all,-,-,PASS,,,," >> "$SUMMARY"
else
    echo "residual,129,77,$i0,all,-,-,FAIL,,,," >> "$SUMMARY"
    fails=$((fails + 1))
fi
check residual 129 77 "$i0" v1_openmp 1 4 "$T/ref.bin" "$T/t1.bin"
check residual 129 77 "$i0" v2_mpi 3 1 "$T/ref.bin" "$T/t2.bin"
check residual 129 77 "$i0" v3_hybrid 3 2 "$T/ref.bin" "$T/t3.bin"

echo
if [ "$fails" -eq 0 ]; then echo "ALL TESTS PASSED  (summary: $SUMMARY)"; else echo "$fails TEST(S) FAILED  (summary: $SUMMARY)"; fi
rm -rf "$T"
[ "$fails" -eq 0 ]
