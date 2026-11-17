#!/bin/bash
# V4 (MPI + CUDA) validation against V0: every comparison must be bit-exact (tolerance 0).
# Run from the repo root on a GPU node after `make`. Ranks share GPUs if there are fewer GPUs
# than ranks (fine for correctness). Writes results/validation/summary_gpu.csv; exit 1 on failure.
set -u
MPIRUN=${MPIRUN:-"mpirun --oversubscribe --bind-to none"}
export OMPI_MCA_mca_base_component_show_load_errors=${OMPI_MCA_mca_base_component_show_load_errors:-0}
B=${B:-build}
D=results/validation
T=$D/tmp_gpu
mkdir -p "$T"
SUMMARY=$D/summary_gpu.csv
echo "test,nx,ny,iters,version,ranks,bx,by,result,max_abs_err,rmse,ndiff,boundary_ok" > "$SUMMARY"
fails=0

check() {  # check NX NY ITERS RANKS BX BY
    local line
    line=$("$B/validate" "$T/ref.bin" "$T/gpu.bin")
    [ $? -eq 0 ] || fails=$((fails + 1))
    echo "v4 ${1}x${2} it=$3 P=$4 block=${5}x${6} : $line"
    echo "gpu,$1,$2,$3,v4_mpi_cuda,$4,$5,$6,$(echo "$line" | sed -E 's/ [a-z_]+=/,/g')" >> "$SUMMARY"
}

for case in "64 64 200" "129 77 300" "1000 700 100" "1024 1024 50"; do
    set -- $case
    nx=$1 ny=$2 it=$3
    "$B/v0_serial" --nx $nx --ny $ny --iters $it --out "$T/ref.bin" > /dev/null
    # every block shape on one rank (tests the kernel's bounds checks)
    for blk in "32 8" "16 16" "32 32" "8 32" "256 1"; do
        set -- $blk
        $MPIRUN -np 1 "$B/v4_mpi_cuda" --nx $nx --ny $ny --iters $it --bx $1 --by $2 --out "$T/gpu.bin" > /dev/null
        check $nx $ny $it 1 $1 $2
    done
    # several ranks (tests halo exchange, edges and uneven row splits)
    for p in 2 3 4; do
        $MPIRUN -np $p "$B/v4_mpi_cuda" --nx $nx --ny $ny --iters $it --out "$T/gpu.bin" > /dev/null
        check $nx $ny $it $p 32 8
    done
done

echo
if [ "$fails" -eq 0 ]; then echo "ALL GPU TESTS PASSED  (summary: $SUMMARY)"; else echo "$fails GPU TEST(S) FAILED  (summary: $SUMMARY)"; fi
rm -rf "$T"
[ "$fails" -eq 0 ]
