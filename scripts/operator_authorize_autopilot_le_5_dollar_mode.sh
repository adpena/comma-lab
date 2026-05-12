#!/bin/bash
# Thin shim: delegates to the canonical operator-authorize entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/autopilot_le_5_dollar_mode.yaml``
#
# Per simplification audit `a2a901c4f43d66a74` (operator approved 2026-05-12)
# + Catalog #162 ``check_operator_authorize_canonical_use``.
#
# Env-var overrides honored by the canonical entry point:
#   AUTOPILOT_RANKING_JSON  (default: most recent under experiments/results/cathedral_autopilot_dispatch_ranking_*/)
#   AUTOPILOT_MAX_DISPATCHES (default: 8)
#
# Lane: lane_fix_g_t1c_wrapper_unification_20260512
# Cross-ref: feedback_fix_g_t1c_wrapper_unification_landed_20260512.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Phase 1: canonical entry point — recipe banner only. This platform=none
# wrapper owns the real action prompt and must not create phantom claims.
.venv/bin/python tools/operator_authorize.py \
    --recipe autopilot_le_5_dollar_mode \
    --agent "claude:operator_authorize_autopilot_le_5_dollar_mode" \
    --no-claim \
    --dry-run \
    "$@" || exit $?

# Phase 2: bespoke action sequence preserved for back-compat.
UTC=$(date -u +%Y%m%dT%H%M%SZ)
RANKING_JSON="${AUTOPILOT_RANKING_JSON:-}"
JOURNAL_DIR="experiments/results/autopilot_authorized_journal_${UTC}"
JOURNAL_PATH="${JOURNAL_DIR}/journal.jsonl"
LOOP_REPORT="${JOURNAL_DIR}/autopilot_loop_report.json"
MAX_DISPATCHES="${AUTOPILOT_MAX_DISPATCHES:-8}"

if [ -z "$RANKING_JSON" ]; then
    RANKING_JSON=$(ls -t experiments/results/cathedral_autopilot_dispatch_ranking_*/ranking.json 2>/dev/null | head -1 || true)
fi
if [ -z "$RANKING_JSON" ] || [ ! -f "$RANKING_JSON" ]; then
    echo "[autopilot-le-5] FATAL: no ranking JSON found. Set AUTOPILOT_RANKING_JSON=<path>." >&2
    exit 1
fi

mkdir -p "$JOURNAL_DIR"
echo "[autopilot-le-5] ranking JSON:    ${RANKING_JSON}"
echo "[autopilot-le-5] journal target:  ${JOURNAL_PATH}"
echo "[autopilot-le-5] max dispatches:  ${MAX_DISPATCHES}"

read -r -p "Proceed with autopilot le-\$5/individual + le-\$20-cumulative authorization? [y/N] " confirm
case "$confirm" in
    [yY]|[yY][eE][sS])
        ;;
    *)
        echo "[autopilot-le-5] aborted — no autopilot dispatch fired"
        rmdir "$JOURNAL_DIR" 2>/dev/null || true
        exit 0
        ;;
esac

export CATHEDRAL_AUTOPILOT_OPERATOR_AUTHORIZED_MODE=1

if [ ! -f "tools/claim_lane_dispatch.py" ]; then
    echo "[autopilot-le-5] FATAL: lane-claim helper missing at tools/claim_lane_dispatch.py" >&2
    exit 1
fi

.venv/bin/python tools/cathedral_autopilot_autonomous_loop.py \
    --use-substrate-composition-matrix-ranking "$RANKING_JSON" \
    --operator-authorized-le-5-dollar-mode \
    --journal-path "$JOURNAL_PATH" \
    --output "$LOOP_REPORT" \
    --iterations 1 \
    --max-dispatch-recommendations "$MAX_DISPATCHES"

echo "[autopilot-le-5] complete; review journal=${JOURNAL_PATH} loop=${LOOP_REPORT}"
