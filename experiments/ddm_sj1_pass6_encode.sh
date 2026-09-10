#!/bin/bash
# ddm_sj1 pass 6 -- price a token field by REAL encode, twice, concurrently.
#
# Same contract as ddm_rp1_price.sh (the shared jg2 loop, the live-pointer coder gate,
# twin encodes before any byte number is quoted).  This arm keeps its OWN driver for two
# reasons, neither cosmetic:
#
#   1. rp1's driver expands `"${CODER_GATE_ARGS[@]}"` unconditionally, and macOS ships
#      bash 3.2, where `set -u` treats an EMPTY array expansion as UNBOUND.  It refuses
#      to start whenever CODER_DIFFERS_BECAUSE is unset -- which is the normal case, and
#      the case this arm needs.  The branch below is the same fix ddm_sj1_pass_shards.sh
#      already documents; it cannot smuggle an empty string into argparse either.
#   2. Editing a sister arm's live script to fix that is a cross-arm edit this charter
#      forbids without telling MAIN.
#
# Env: FIELD (600-plane npz), STORE (custody dir), TAG, TWIN (default 1),
#      EXPECT_POINTER_SHA256 (optional pin on the archive the sections are read from).
set -euo pipefail

REPO="/Users/adpena/Projects/pact"
# The ENCODER tree is cl2's receiver copy: jg2's loader reads each section in the coding
# that copy understands, and the live-pointer coder gate below is what proves the CODER
# itself is still the shipped one.  This is the same pair of objects rp1 round 2 priced
# through, which is why a control encode of the pointer's own field is expected to
# reproduce the pointer's shipped rider body byte-for-byte.
ENCODER="/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/rungs/lambda_1p0/retained/receiver_copy_runtime"
TOKENS="/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/rungs/lambda_1p0/retained/decoded_tokens.u8"
FIELD="${FIELD:?set FIELD}"
STORE="${STORE:?set STORE}"
TAG="${TAG:?set TAG}"
TWIN="${TWIN:-1}"
FRAMES="${FRAMES:-600}"

mkdir -p "$STORE/retained"
cd "$REPO"

suffixes=("")
if [ "$TWIN" = "1" ]; then
    suffixes=("" "_twin")
fi

pids=()
for suffix in "${suffixes[@]}"; do
    if [ -n "${CODER_DIFFERS_BECAUSE:-}" ]; then
        "$REPO/.venv/bin/python" experiments/ddm_jg2_tail_reencode.py --stage encode \
            --store "$STORE" --runtime-root "$ENCODER" --tokens "$TOKENS" \
            --coder-differs-because "$CODER_DIFFERS_BECAUSE" \
            --edits "$FIELD" --tag "${TAG}${suffix}" \
            --frames "$FRAMES" --checkpoint-every 25 --resume \
            > "$STORE/encode_${TAG}${suffix}.log" 2>&1 &
    else
        "$REPO/.venv/bin/python" experiments/ddm_jg2_tail_reencode.py --stage encode \
            --store "$STORE" --runtime-root "$ENCODER" --tokens "$TOKENS" \
            --edits "$FIELD" --tag "${TAG}${suffix}" \
            --frames "$FRAMES" --checkpoint-every 25 --resume \
            > "$STORE/encode_${TAG}${suffix}.log" 2>&1 &
    fi
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then
    echo "FAIL: an encode exited nonzero" >&2
    exit 1
fi
echo "ddm_sj1 pass-6 encode complete: $TAG"
