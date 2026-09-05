#!/bin/bash
# ddm_sj1 step 0 -- the n600 baseline seg leg of the cl2 frontier body, sharded.
#
# Runs SHARDS concurrent shards of `ddm_sj1_multipass_token_predistortion.py step0`,
# each pinned to THREADS torch threads, then merges into the n600 leg.  All 600 pairs:
# a sub-n600 seg verdict is a TOY on this axis and the merge refuses one.
set -euo pipefail

REPO="/Users/adpena/Projects/pact"
ROOT="/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/step0"
SHARDS="${SHARDS:-6}"
THREADS="${THREADS:-3}"

mkdir -p "$ROOT"
cd "$REPO"

pids=()
for i in $(seq 0 $((SHARDS - 1))); do
    "$REPO/.venv/bin/python" experiments/ddm_sj1_multipass_token_predistortion.py step0 \
        --shard-index "$i" --shard-count "$SHARDS" --threads "$THREADS" --progress \
        --out "$ROOT/argmax_shard_${i}.npy" \
        --receipt "$ROOT/step0_shard_${i}.json" \
        > "$ROOT/step0_shard_${i}.log" 2>&1 &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then
    echo "FAIL: at least one step0 shard exited nonzero" >&2
    exit 1
fi

"$REPO/.venv/bin/python" experiments/ddm_sj1_multipass_token_predistortion.py step0-merge \
    --shards "$ROOT"/step0_shard_*.json \
    --out-argmax "$ROOT/argmax_n600_base.npy" \
    --out "$ROOT/STEP0_RESULT.json"
