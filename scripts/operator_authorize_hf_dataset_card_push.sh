#!/bin/bash
# Operator authorize: push the refreshed HF dataset card to adpena/comma_video_compression_challenge_pr_archive.
#
# Per operator F-3 in operator decision dashboard 2026-05-11: this is
# FROZEN-OPERATOR pending "don't submit PR yet". The local card update
# (`docs/comma_pr_archive_dataset_card.md` refreshed per TT autopilot
# end-to-end + HF refresh landing) is staged for operator review before
# remote push.
#
# Per CLAUDE.md "Public Disclosure Hygiene": this script runs the
# check_public_release_hygiene STRICT preflight gate on the card BEFORE
# the network push.
#
# Per CLAUDE.md "Operator gates must be wired and used": canonical wrapper
# for the F-3 transition from FROZEN-OPERATOR to OPERATOR-AUTHORIZED.
#
# Cost: $0 (free HF push).
# Risk: public-facing release. Once pushed, the card is visible at
#       https://huggingface.co/datasets/adpena/comma_video_compression_challenge_pr_archive
#       — public mirrors may cache it.
#
# Usage: bash scripts/operator_authorize_hf_dataset_card_push.sh
#
# Lane: lane_operator_one_command_authorize_scripts L0
# Cross-ref: project_operator_decision_dashboard_20260511.md F-3
#            feedback_autopilot_end_to_end_hf_refresh_phase1_cost_refinement_landed_20260511.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CARD_LOCAL="docs/comma_pr_archive_dataset_card.md"
HF_DATASET="${HF_DATASET:-adpena/comma_video_compression_challenge_pr_archive}"

if [ ! -f "$CARD_LOCAL" ]; then
    echo "[hf-card-push] FATAL: local card missing at ${CARD_LOCAL}" >&2
    exit 1
fi

CARD_HEAD=$(head -1 "$CARD_LOCAL")
CARD_REFRESH_LINE=$(grep "Card last refreshed" "$CARD_LOCAL" | head -1 || echo "<none>")
CARD_SIZE=$(stat -f%z "$CARD_LOCAL" 2>/dev/null || stat -c%s "$CARD_LOCAL" 2>/dev/null || echo "?")

cat <<EOF

=== HF dataset card push operator confirmation ===

local card:              ${CARD_LOCAL}
local card size:         ${CARD_SIZE} bytes
card first line:         ${CARD_HEAD}
card refresh line:       ${CARD_REFRESH_LINE}
target HF dataset:       ${HF_DATASET}

pre-flight: check_public_release_hygiene STRICT preflight will run on the
card before push. If it fails, the push is REFUSED.

risk:                    public-facing release. Visible at
                         https://huggingface.co/datasets/${HF_DATASET}
cost:                    \$0 (free HF push)

EOF
read -r -p "Run check_public_release_hygiene STRICT on the card now? [Y/n] " hygiene
case "$hygiene" in
    [nN]|[nN][oO])
        echo "[hf-card-push] WARN: skipping pre-flight hygiene check at operator's request"
        ;;
    *)
        echo "[hf-card-push] step 1/2: STRICT preflight..."
        if .venv/bin/python -m tac.preflight --release-only --strict 2>&1 | tail -20; then
            echo "[hf-card-push] preflight PASS"
        else
            echo "[hf-card-push] FATAL: preflight FAILED; refusing push" >&2
            exit 2
        fi
        ;;
esac

read -r -p "Proceed with HF push to ${HF_DATASET}? [y/N] " confirm
case "$confirm" in
    [yY]|[yY][eE][sS])
        ;;
    *)
        echo "[hf-card-push] aborted — no remote push"
        exit 0
        ;;
esac

echo "[hf-card-push] step 2/2: pushing card to HF..."

# Try the canonical hf CLI first; fall back to huggingface-cli if needed.
if command -v hf >/dev/null 2>&1; then
    hf upload "$HF_DATASET" "$CARD_LOCAL" README.md --repo-type dataset
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli upload "$HF_DATASET" "$CARD_LOCAL" README.md --repo-type dataset
else
    cat >&2 <<EOM
[hf-card-push] FATAL: neither 'hf' nor 'huggingface-cli' is on PATH.

Install via: pip install huggingface_hub  (or:  uv pip install huggingface_hub)

The card is at ${CARD_LOCAL}; once the CLI is installed, run:
  hf upload ${HF_DATASET} ${CARD_LOCAL} README.md --repo-type dataset
EOM
    exit 1
fi

echo "[hf-card-push] complete — card pushed to https://huggingface.co/datasets/${HF_DATASET}"
