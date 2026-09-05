#!/bin/bash
# ddm_sj1 influence + yield probe -- SIZING only, never a family verdict.
# Six seeds so the sampled pairs are six disjoint seeded-random draws, never a prefix.
set -euo pipefail
REPO="/Users/adpena/Projects/pact"
ROOT="/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/probe"
SHARDS="${SHARDS:-6}"
THREADS="${THREADS:-3}"
PAIRS="${PAIRS:-3}"
SITES="${SITES:-2}"
mkdir -p "$ROOT"
cd "$REPO"
pids=()
for i in $(seq 0 $((SHARDS - 1))); do
    "$REPO/.venv/bin/python" experiments/ddm_sj1_multipass_token_predistortion.py probe \
        --pairs "$PAIRS" --sites "$SITES" --seed $((20260905 + i)) --threads "$THREADS" --progress \
        --out "$ROOT/probe_${i}.json" > "$ROOT/probe_${i}.log" 2>&1 &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
