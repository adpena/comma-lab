#!/bin/bash
# ddm_rp1 -- price a rate-directed field by REAL encode, twice, concurrently.
#
# The ranking that produced the field is a sum of FIRST-ORDER per-position savings.  It
# orders the search; it does not charge it.  sj1's placement law (forced writes cost
# ~1.73x greedy ones) and fs2's direction-dependence law both say the same thing: the
# only honest byte number is the one a real 600-frame encode emits.
#
# Two independent encodes of the SAME field run concurrently.  A byte difference between
# them would make any delta run-to-run variance rather than a measurement, so the pair is
# always run before a byte number is quoted (sj1's discipline, kept).
#
# Env: FIELD (the full 600-plane npz), STORE (custody dir), TAG.
set -euo pipefail

REPO="/Users/adpena/Projects/pact"
ENCODER="/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/rungs/lambda_1p0/retained/receiver_copy_runtime"
TOKENS="/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/rungs/lambda_1p0/retained/decoded_tokens.u8"
FIELD="${FIELD:?set FIELD}"
STORE="${STORE:?set STORE}"
TAG="${TAG:?set TAG}"
TWIN="${TWIN:-1}"

mkdir -p "$STORE/retained"
cd "$REPO"

suffixes=("")
[ "$TWIN" = "1" ] && suffixes=("" "_twin")

for suffix in "${suffixes[@]}"; do
    "$REPO/.venv/bin/python" experiments/ddm_jg2_tail_reencode.py --stage encode \
        --store "$STORE" --runtime-root "$ENCODER" --tokens "$TOKENS" \
        --edits "$FIELD" --tag "${TAG}${suffix}" \
        --frames 600 --checkpoint-every 25 --resume \
        > "$STORE/encode_${TAG}${suffix}.log" 2>&1 &
done

status=0
for pid in $(jobs -p); do wait "$pid" || status=1; done
[ "$status" -eq 0 ] || { echo "FAIL: an encode exited nonzero" >&2; exit 1; }
echo "ddm_rp1 encode complete: $TAG"
