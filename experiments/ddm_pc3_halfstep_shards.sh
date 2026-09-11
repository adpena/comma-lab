#!/bin/bash
# ddm_pc3 -- the realised per-dimension halving gain, sharded (strided, resumable).
set -euo pipefail
REPO="/Users/adpena/Projects/pact"
OUT="${OUT:?set OUT}"
SHARDS="${SHARDS:-9}"
THREADS="${THREADS:-2}"
mkdir -p "$OUT"
cd "$REPO"
pids=()
for i in $(seq 0 $((SHARDS - 1))); do
    "$REPO/.venv/bin/python" experiments/ddm_pc3_pose_carrier_curve.py halfstep \
        --out-dir "$OUT" --shard-index "$i" --shard-count "$SHARDS" \
        --threads "$THREADS" --resume --progress \
        > "$OUT/halfstep_shard_${i}.log" 2>&1 &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[ "$status" -eq 0 ] || { echo "FAIL: a halfstep shard exited nonzero" >&2; exit 1; }
echo "ddm_pc3 halfstep complete: $SHARDS shards"
