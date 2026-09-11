#!/bin/bash
# ddm_pc3 -- the timing-risk diagnostic PAIR, run CONCURRENTLY as a paired design.
#
# The pre-fire risk contract prices a candidate's T4 spend as
# `source_t4_seconds * (1 + max(0, candidate_local_ceiling/base_local - 1))`.  Only the
# RATIO of the two local legs enters; the absolute walls do not.  So the design question
# is not "how do I make each leg fast" but "how do I keep the nuisance load from landing
# on one leg and not the other".
#
# WHY CONCURRENT AND NOT SEQUENTIAL.  This machine cannot be quiesced: sister arms own
# live runs (ddm_obx2 trainers, ddm_ntb2) that this arm must not touch, and they were
# measured at a combined load average above 30.  A sequential pair would measure the DRIFT
# in that sister load between the two legs, and the policy budget here is only 2.238 %
# (1260 / 1232.418725255 - 1) -- far below the drift.  Run concurrently, both legs see the
# same machine at every instant: the shared load cancels in the ratio, and the mutual
# interference between the two legs inflates both walls symmetrically.  A sequential
# attempt was started, abandoned at 100/600 before producing anything, and retained with
# its reason in `diagnostics_sequential_abandoned/WHY_SUPERSEDED.json`.
#
# Neither tree is written to: every byte lands under the per-leg --work-dir, so the base
# leg can run against another arm's sealed tree.
set -euo pipefail

REPO="/Users/adpena/Projects/pact"
W="${W:-/Volumes/VertigoDataTier/pact/ddm_pc3_pose_carrier_curve}"
BASE_RUNTIME="${BASE_RUNTIME:-/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/candidate_runtime}"
CANDIDATE_RUNTIME="${CANDIDATE_RUNTIME:-$W/candidate/candidate_runtime}"
THREADS="${THREADS:-4}"
NOTE="${NOTE:-paired-concurrent on a machine held busy by sister arms; both legs start together and share the nuisance load, so it cancels in the ratio}"

cd "$REPO"

pids=()
"$REPO/.venv/bin/python" experiments/ddm_pc3_public.py diagnostic \
    --runtime "$BASE_RUNTIME" --work-dir "$W/diagnostics/base_move44" \
    --blas-threads "$THREADS" --quiesce-note "$NOTE" \
    > "$W/diagnostics_base.log" 2>&1 &
pids+=($!)

"$REPO/.venv/bin/python" experiments/ddm_pc3_public.py diagnostic \
    --runtime "$CANDIDATE_RUNTIME" --work-dir "$W/diagnostics/candidate" \
    --blas-threads "$THREADS" --quiesce-note "$NOTE" \
    > "$W/diagnostics_candidate.log" 2>&1 &
pids+=($!)

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then
    echo "FAIL: at least one diagnostic leg exited nonzero" >&2
    exit 1
fi
echo "ddm_pc3 diagnostic pair complete"
