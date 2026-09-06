#!/bin/bash
# ddm_sj1 successor pricing -- price a LATER pass's field against the LIVE pointer.
#
# Difference from the first round, and the reason this script exists separately: the
# CONTROL is already done.  The live pointer's token stream IS the previous round's
# encode output, so re-encoding the unedited field would re-derive a number already on
# disk.  The control that matters here is the one that has already passed: the ORIGINAL
# unedited field re-encoded byte-identical to the shipped stream, which is what makes
# this encoder's output the pointer's own bytes at all.
#
# Two independent encodes of the SAME field run concurrently; a byte difference between
# them makes the delta run-to-run variance rather than a measurement, so both are run
# before any byte number is quoted.
#
# Env: FIELD (cumulative edited npz), STORE, TAG, OUT (overlay/pose dir), BASE_RAW.
set -euo pipefail

REPO="/Users/adpena/Projects/pact"
ENCODER="/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/rungs/lambda_1p0/retained/receiver_copy_runtime"
TOKENS="/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/rungs/lambda_1p0/retained/decoded_tokens.u8"
FIELD="${FIELD:?set FIELD}"
STORE="${STORE:?set STORE}"
TAG="${TAG:?set TAG}"
OUT="${OUT:?set OUT}"

mkdir -p "$STORE/retained" "$OUT"
cd "$REPO"

for suffix in "" "_twin"; do
    "$REPO/.venv/bin/python" experiments/ddm_jg2_tail_reencode.py --stage encode \
        --store "$STORE" --runtime-root "$ENCODER" --tokens "$TOKENS" \
        --edits "$FIELD" --tag "${TAG}${suffix}" \
        --frames 600 --checkpoint-every 25 --resume \
        > "$STORE/encode_${TAG}${suffix}.log" 2>&1 &
done

status=0
for pid in $(jobs -p); do wait "$pid" || status=1; done
if [ "$status" -ne 0 ]; then
    echo "FAIL: an encode exited nonzero" >&2
    exit 1
fi

# The candidate's own odd frames.  Every pair is edited, so every odd frame is
# substituted and the decode underneath never shows through on the seg-scored frame --
# but it is pointed at the live pointer's own decode anyway, so the object is unambiguous.
"$REPO/.venv/bin/python" experiments/ddm_sj1_joint_admission.py render-edits \
    --field "$FIELD" --out-dir "$OUT/overlay" --threads 4 --progress \
    > "$OUT/render_edits.log" 2>&1

# The STALE leg: the live pointer's OWN carrier on the NEW renders, i.e. the damage this
# pass's edits do before any re-solve.  The BASE leg is not re-measured -- it is the live
# row's own pose on its own renders, already on disk as pose_resolved.npy.
"$REPO/.venv/bin/python" experiments/ddm_sj1_joint_admission.py pose \
    --tag stale --overlay "$OUT/overlay" --out "$OUT/pose_stale.npy" --threads 6 \
    > "$OUT/pose_stale.log" 2>&1

echo "ddm_sj1 successor pricing complete"
