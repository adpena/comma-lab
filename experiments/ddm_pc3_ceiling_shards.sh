#!/bin/bash
# ddm_pc3 -- the continuous-coefficient ceiling inside the shipped 12-dim span, sharded.
#
# The shards are STRIDED, never contiguous: this field's pose mass sits in the 57-91
# scene block, and a contiguous shard that died would leave a partial biased by exactly
# the amount that matters.  Every shard resumes from its own rows file, so a killed run
# costs only the pair it was on.
#
# Env: OUT, OPERATING_POINT, SHARDS, THREADS.
set -euo pipefail

REPO="/Users/adpena/Projects/pact"
OUT="${OUT:?set OUT}"
OPERATING_POINT="${OPERATING_POINT:?set OPERATING_POINT}"
SHARDS="${SHARDS:-9}"
THREADS="${THREADS:-2}"

mkdir -p "$OUT"
cd "$REPO"

pids=()
for i in $(seq 0 $((SHARDS - 1))); do
    "$REPO/.venv/bin/python" experiments/ddm_pc3_pose_carrier_curve.py ceiling \
        --out-dir "$OUT" --shard-index "$i" --shard-count "$SHARDS" \
        --threads "$THREADS" --operating-point-d-pose "$OPERATING_POINT" \
        --resume --progress \
        > "$OUT/ceiling_shard_${i}.log" 2>&1 &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then
    echo "FAIL: at least one ceiling shard exited nonzero" >&2
    exit 1
fi
echo "ddm_pc3 ceiling complete: $SHARDS shards"
