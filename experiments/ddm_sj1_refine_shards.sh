#!/bin/bash
# ddm_sj1 -- carrier re-solve on the candidate's OWN renders, sharded.
#
# The re-solve is jg5's `refine_pair` verbatim (br1's damped Gauss-Newton on the shipped
# 12-dim basis plus the +-2 polish, stopping on the pair's own measured decay against a
# DERIVED materiality floor).  What this driver supplies is the two things that make it
# the RIGHT solve for this candidate:
#   * --overlay : the candidate's own odd frames, so the carrier is tuned against the
#     frames this body actually ships and not the base decode's.
#   * the LIVE POINTER's coefficients and lattice, enforced inside the module by
#     `assert_carrier_is_pointer` -- a solve seeded from a superseded body's codes would
#     silently revert the banked carrier move while every code stayed a valid int12.
#
# Env: OVERLAY, FIELD, BASE_POSE, OUT, SHARDS, THREADS.
set -euo pipefail

REPO="/Users/adpena/Projects/pact"
OVERLAY="${OVERLAY:?set OVERLAY}"
FIELD="${FIELD:?set FIELD}"
BASE_POSE="${BASE_POSE:?set BASE_POSE}"
OUT="${OUT:?set OUT}"
SHARDS="${SHARDS:-6}"
THREADS="${THREADS:-2}"

mkdir -p "$OUT"
cd "$REPO"

pids=()
for i in $(seq 0 $((SHARDS - 1))); do
    "$REPO/.venv/bin/python" experiments/ddm_sj1_joint_admission.py refine \
        --overlay "$OVERLAY" --field "$FIELD" --base-pose "$BASE_POSE" \
        --out-dir "$OUT" --shard-index "$i" --shard-count "$SHARDS" \
        --threads "$THREADS" --resume --progress \
        > "$OUT/refine_shard_${i}.log" 2>&1 &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then
    echo "FAIL: at least one refine shard exited nonzero" >&2
    exit 1
fi

"$REPO/.venv/bin/python" experiments/ddm_sj1_joint_admission.py codes \
    --rows "$OUT"/refine_rows_*.jsonl --out "$OUT/codes_resolved.npy"
