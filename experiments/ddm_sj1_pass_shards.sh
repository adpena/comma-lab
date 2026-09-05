#!/bin/bash
# ddm_sj1 -- one pass of realized token pre-distortion over ALL 600 pairs, sharded.
#
# STRIDED shards (--shard-index i --shard-count N takes pairs i, i+N, ...), never
# contiguous blocks: a contiguous block of this video is a different population and the
# bias is worst on the pose axis, so a shard that dies leaves an unbiased partial.
#
# Env:
#   PASS_INDEX  pass number for the ledger (2, 3, ...)
#   STAGE       gt | rest | full   (which slice of the 36-move family)
#   OUT         output directory for this pass
#   FIELD       optional npz of the prior pass's token planes
#   SHARDS      concurrent shards          (default 9)
#   THREADS     torch threads per shard    (default 2)
#   BATCH       pairs in flight per shard  (default 8; SegNet batches, render stays 1)
set -euo pipefail

REPO="/Users/adpena/Projects/pact"
PASS_INDEX="${PASS_INDEX:?set PASS_INDEX}"
STAGE="${STAGE:?set STAGE}"
OUT="${OUT:?set OUT}"
SHARDS="${SHARDS:-9}"
THREADS="${THREADS:-2}"
BATCH="${BATCH:-8}"
FIELD="${FIELD:-}"

mkdir -p "$OUT"
cd "$REPO"

# macOS ships bash 3.2, where `set -u` treats an EMPTY array expansion as unbound.
# Branching on the flag is clearer than the `${arr[@]+...}` incantation and cannot
# smuggle an empty string into argparse.
pids=()
for i in $(seq 0 $((SHARDS - 1))); do
    if [ -n "$FIELD" ]; then
        "$REPO/.venv/bin/python" experiments/ddm_sj1_multipass_token_predistortion.py pass \
            --pass-index "$PASS_INDEX" --stage "$STAGE" \
            --shard-index "$i" --shard-count "$SHARDS" \
            --threads "$THREADS" --batch "$BATCH" --progress --resume \
            --field "$FIELD" \
            --out-dir "$OUT" > "$OUT/pass_shard_${i}.log" 2>&1 &
    else
        "$REPO/.venv/bin/python" experiments/ddm_sj1_multipass_token_predistortion.py pass \
            --pass-index "$PASS_INDEX" --stage "$STAGE" \
            --shard-index "$i" --shard-count "$SHARDS" \
            --threads "$THREADS" --batch "$BATCH" --progress --resume \
            --out-dir "$OUT" > "$OUT/pass_shard_${i}.log" 2>&1 &
    fi
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then
    echo "FAIL: at least one pass shard exited nonzero" >&2
    exit 1
fi

"$REPO/.venv/bin/python" experiments/ddm_sj1_multipass_token_predistortion.py pass-merge \
    --receipts "$OUT"/PASS_SHARD_*.json \
    --out-field "$OUT/field_after.npz" \
    --out "$OUT/PASS_RESULT.json"
