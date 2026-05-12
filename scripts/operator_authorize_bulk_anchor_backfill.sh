#!/bin/bash
# Thin shim: delegates to the canonical operator-authorize entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/bulk_anchor_backfill.yaml``
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
    --recipe bulk_anchor_backfill \
    --agent "claude:operator_authorize_bulk_anchor_backfill" \
    --no-claim \
    --dry-run \
    "$@" || exit $?

# Phase 2: bespoke action sequence preserved for back-compat.
UTC=$(date -u +%Y%m%dT%H%M%SZ)
OUTDIR="experiments/results/bulk_backfill_${UTC}"
AUDIT_LOG="${OUTDIR}/audit.jsonl"
SUMMARY_JSON="${OUTDIR}/commit_summary.json"

echo "[bulk-backfill] step 1/3: dry-run discovery..."
.venv/bin/python tools/bulk_backfill_anchors_into_posterior.py \
    --quiet \
    --summary-json "${OUTDIR}/dry_run_summary.json" \
    >/dev/null

if [ ! -f "${OUTDIR}/dry_run_summary.json" ]; then
    echo "[bulk-backfill] FATAL: dry-run summary missing at ${OUTDIR}/dry_run_summary.json" >&2
    exit 1
fi

PROMOTABLE=$(.venv/bin/python -c "import json; p = json.load(open('${OUTDIR}/dry_run_summary.json')); print(p['discovery_summary']['promotable_orphans'])")
echo "[bulk-backfill] promotable orphans: ${PROMOTABLE}"

read -r -p "Proceed with --commit? [y/N] " confirm
case "$confirm" in
    [yY]|[yY][eE][sS])
        ;;
    *)
        echo "[bulk-backfill] aborted — no posterior changes made"
        exit 0
        ;;
esac

echo "[bulk-backfill] step 2/3: commit..."
.venv/bin/python tools/bulk_backfill_anchors_into_posterior.py \
    --commit \
    --audit-log-path "$AUDIT_LOG" \
    --summary-json "$SUMMARY_JSON" \
    --quiet

ACCEPTED=$(.venv/bin/python -c "import json; p = json.load(open('${SUMMARY_JSON}')); print(p['commit_result']['accepted'])")
echo "[bulk-backfill] step 3/3: accepted=${ACCEPTED}"
echo "[bulk-backfill] audit log: ${AUDIT_LOG}"
echo "[bulk-backfill] commit summary: ${SUMMARY_JSON}"
