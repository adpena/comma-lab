#!/bin/bash
# ddm_sj1 -- price the edited field EXACTLY and measure the pose damage it causes.
#
# Four stages, in the only order that keeps each number honest:
#   1. jg2 --stage control : re-encode the UNEDITED field.  Refuses unless the emitted
#      token stream is byte-identical to the shipped one, which is the proof that this
#      encoder inverts the shipping decoder.  No byte delta is trustworthy before it.
#      Also emits the per-frame bit ledger the admission sweep needs as its BASELINE.
#   2. jg2 --stage encode  : re-encode the EDITED field and build the candidate archive.
#      Per-pair rate cost is (bits_candidate - bits_control)/8 -- a MARGINAL cost from a
#      REAL re-encode, never a `-log2 p` sum and never an average
#      (token_rate_model_direction_dependence_v1; average != marginal by 2.24x, fs3).
#   3. render-edits        : the candidate's own odd frames, at the receiver's batch 1.
#   4. pose base + stale   : d_pose with the SHIPPED carrier on the BASE renders (the
#      base leg) and on the CANDIDATE renders (the damage the edits cause before any
#      re-solve).  Both over all 600 pairs.
#
# Env: FIELD (edited npz), STORE (pricing custody dir), OUT (pose/overlay dir).
set -euo pipefail

REPO="/Users/adpena/Projects/pact"
BODY="/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/rungs/lambda_1p0/retained/receiver_copy_runtime"
TOKENS="/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/rungs/lambda_1p0/retained/decoded_tokens.u8"
FIELD="${FIELD:?set FIELD}"
STORE="${STORE:?set STORE}"
OUT="${OUT:?set OUT}"

mkdir -p "$STORE/retained" "$OUT"
cd "$REPO"

# 1 + 2 concurrently: they are the same length of compute and independent, and the
# encode stage waits for the control's receipt before it reports a trusted delta.
"$REPO/.venv/bin/python" experiments/ddm_jg2_tail_reencode.py --stage control \
    --store "$STORE" --runtime-root "$BODY" --tokens "$TOKENS" \
    --frames 600 --checkpoint-every 25 --resume \
    > "$STORE/control.log" 2>&1 &
control_pid=$!

"$REPO/.venv/bin/python" experiments/ddm_jg2_tail_reencode.py --stage encode \
    --store "$STORE" --runtime-root "$BODY" --tokens "$TOKENS" \
    --edits "$FIELD" --tag sj1_pass2a \
    --frames 600 --checkpoint-every 25 --resume \
    --wait-for-control-seconds 5400 \
    > "$STORE/encode.log" 2>&1 &
encode_pid=$!

status=0
wait "$control_pid" || status=1
wait "$encode_pid" || status=1
if [ "$status" -ne 0 ]; then
    echo "FAIL: jg2 control or encode exited nonzero" >&2
    exit 1
fi

# 3. the candidate's own odd frames
"$REPO/.venv/bin/python" experiments/ddm_sj1_joint_admission.py render-edits \
    --field "$FIELD" --out-dir "$OUT/overlay" --threads 4 --progress \
    > "$OUT/render_edits.log" 2>&1

# 4. the two pose legs that the admission sweep needs before any re-solve
"$REPO/.venv/bin/python" experiments/ddm_sj1_joint_admission.py pose \
    --tag base --out "$OUT/pose_base.npy" --threads 4 \
    > "$OUT/pose_base.log" 2>&1

"$REPO/.venv/bin/python" experiments/ddm_sj1_joint_admission.py pose \
    --tag stale --overlay "$OUT/overlay" --out "$OUT/pose_stale.npy" --threads 4 \
    > "$OUT/pose_stale.log" 2>&1

echo "ddm_sj1 price-and-pose complete"
