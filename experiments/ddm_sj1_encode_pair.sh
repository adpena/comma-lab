#!/bin/bash
# ddm_sj1 -- two independent encodes of one field, concurrently.
# A byte difference between them makes any delta run-to-run variance rather than a
# measurement, so the pair is always run before a byte number is quoted.
set -euo pipefail
REPO="/Users/adpena/Projects/pact"
ENCODER="/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/rungs/lambda_1p0/retained/receiver_copy_runtime"
TOKENS="/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/rungs/lambda_1p0/retained/decoded_tokens.u8"
FIELD="${FIELD:?set FIELD}"; STORE="${STORE:?set STORE}"; TAG="${TAG:?set TAG}"
mkdir -p "$STORE/retained"; cd "$REPO"
for suffix in "" "_twin"; do
    "$REPO/.venv/bin/python" experiments/ddm_jg2_tail_reencode.py --stage encode \
        --store "$STORE" --runtime-root "$ENCODER" --tokens "$TOKENS" \
        --edits "$FIELD" --tag "${TAG}${suffix}" --frames 600 --checkpoint-every 25 --resume \
        > "$STORE/encode_${TAG}${suffix}.log" 2>&1 &
done
status=0; for pid in $(jobs -p); do wait "$pid" || status=1; done
[ "$status" -eq 0 ] || { echo "FAIL: an encode exited nonzero" >&2; exit 1; }
echo "encode pair complete: $TAG"
