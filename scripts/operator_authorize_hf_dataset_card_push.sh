#!/bin/bash
# Thin shim: delegates to the canonical operator-authorize entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/hf_dataset_card_push.yaml``
#
# Per simplification audit `a2a901c4f43d66a74` (operator approved 2026-05-12)
# + Catalog #162 ``check_operator_authorize_canonical_use``.
#
# Lane: lane_fix_g_t1c_wrapper_unification_20260512
# Cross-ref: feedback_fix_g_t1c_wrapper_unification_landed_20260512.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Phase 1: canonical entry point — recipe banner only. This platform=none
# wrapper owns the real action prompt and must not create phantom claims.
.venv/bin/python tools/operator_authorize.py \
    --recipe hf_dataset_card_push \
    --agent "claude:operator_authorize_hf_dataset_card_push" \
    --no-claim \
    --dry-run \
    "$@" || exit $?

# Phase 2: bespoke action sequence preserved for back-compat.
CARD_LOCAL="docs/comma_pr_archive_dataset_card.md"
HF_DATASET="${HF_DATASET:-adpena/comma_video_compression_challenge_pr_archive}"

if [ ! -f "$CARD_LOCAL" ]; then
    echo "[hf-card-push] FATAL: local card missing at ${CARD_LOCAL}" >&2
    exit 1
fi

read -r -p "Run check_public_release_hygiene STRICT on the card now? [Y/n] " hygiene
case "$hygiene" in
    [nN]|[nN][oO])
        echo "[hf-card-push] WARN: skipping pre-flight hygiene check at operator's request"
        ;;
    *)
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

if command -v hf >/dev/null 2>&1; then
    hf upload "$HF_DATASET" "$CARD_LOCAL" README.md --repo-type dataset
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli upload "$HF_DATASET" "$CARD_LOCAL" README.md --repo-type dataset
else
    echo "[hf-card-push] FATAL: neither 'hf' nor 'huggingface-cli' on PATH" >&2
    echo "  install via: uv pip install huggingface_hub" >&2
    exit 1
fi
echo "[hf-card-push] complete — card pushed to https://huggingface.co/datasets/${HF_DATASET}"
