#!/bin/bash
# Operator authorize: bulk back-fill of 19 (or however many discovered) custody-OK
# orphaned authoritative anchors into the continual-learning posterior.
#
# Per CLAUDE.md "Continual-learning posterior — non-negotiable" + Catalog #128
# (`check_continual_learning_writes_use_lock`) + Catalog #127 (custody validator):
# the back-fill is transactional via `posterior_update_locked`; idempotent re-run
# is safe; custody-refused anchors (e.g. Modal CPU on `[contest-CPU]` short-form
# tag) are correctly REFUSED and surfaced as `skipped_custody_refused` rows.
#
# Per CLAUDE.md FORBIDDEN /tmp paths: audit log lives under
# `experiments/results/bulk_backfill_<UTC>/`.
#
# Cost: $0 GPU. Pure file-update on `.omx/state/continual_learning_posterior.json`.
# Predicted Δ (posterior n_anchors): 6 → 6 + accepted-orphan count (currently
# 20 CUDA orphans per dry-run inventory 2026-05-11; final number depends on
# discovery state at run-time).
#
# Per CLAUDE.md "Operator gates must be wired and used", this script is the
# canonical wrapper for OD-A from
# `project_operator_decision_dashboard_20260511.md`.
#
# Usage: bash scripts/operator_authorize_bulk_anchor_backfill.sh
#
# Lane: lane_bulk_anchor_backfill_tool L0 (per `lane_maturity.py audit`)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

UTC=$(date -u +%Y%m%dT%H%M%SZ)
OUTDIR="experiments/results/bulk_backfill_${UTC}"
AUDIT_LOG="${OUTDIR}/audit.jsonl"
SUMMARY_JSON="${OUTDIR}/commit_summary.json"

# Pre-flight: dry-run to count orphans + verify tool works.
echo "[bulk-backfill] step 1/3: dry-run discovery..."
.venv/bin/python tools/bulk_backfill_anchors_into_posterior.py \
    --quiet \
    --summary-json "${OUTDIR}/dry_run_summary.json" \
    >/dev/null

if [ ! -f "${OUTDIR}/dry_run_summary.json" ]; then
    echo "[bulk-backfill] FATAL: dry-run summary did not land at ${OUTDIR}/dry_run_summary.json" >&2
    exit 1
fi

PROMOTABLE=$(.venv/bin/python -c "import json; p = json.load(open('${OUTDIR}/dry_run_summary.json')); print(p['discovery_summary']['promotable_orphans'])")
TOTAL=$(.venv/bin/python -c "import json; p = json.load(open('${OUTDIR}/dry_run_summary.json')); print(p['discovery_summary']['total_artifacts_discovered'])")
ALREADY=$(.venv/bin/python -c "import json; p = json.load(open('${OUTDIR}/dry_run_summary.json')); print(p['discovery_summary']['already_in_posterior'])")
REFUSED=$(.venv/bin/python -c "import json; p = json.load(open('${OUTDIR}/dry_run_summary.json')); print(p['discovery_summary']['custody_refused'])")

cat <<EOF

=== bulk back-fill operator confirmation ===
discovered total artifacts:   ${TOTAL}
already in posterior:         ${ALREADY}
custody-refused (e.g. Modal CPU): ${REFUSED}
PROMOTABLE ORPHANS (would back-fill): ${PROMOTABLE}

audit log target: ${AUDIT_LOG}

cost:                         \$0 GPU
risk:                         posterior n_anchors shift (currently 6 in posterior;
                              after commit ~ 6 + ${PROMOTABLE} = $(( 6 + PROMOTABLE )))

This will call posterior_update_locked() per orphan inside fcntl LOCK_EX.
Idempotent re-run is safe (duplicate-sha refusal is built in).

EOF
read -r -p "Proceed with --commit? [y/N] " confirm
case "$confirm" in
    [yY]|[yY][eE][sS])
        ;;
    *)
        echo "[bulk-backfill] aborted — no posterior changes made"
        exit 0
        ;;
esac

# Commit step.
echo "[bulk-backfill] step 2/3: commit..."
.venv/bin/python tools/bulk_backfill_anchors_into_posterior.py \
    --commit \
    --audit-log-path "$AUDIT_LOG" \
    --summary-json "$SUMMARY_JSON" \
    --quiet

# Post-flight: verify accepted count.
ACCEPTED=$(.venv/bin/python -c "import json; p = json.load(open('${SUMMARY_JSON}')); print(p['commit_result']['accepted'])")
SKIPPED_DUP=$(.venv/bin/python -c "import json; p = json.load(open('${SUMMARY_JSON}')); print(p['commit_result']['skipped_already_in_posterior'])")

echo "[bulk-backfill] step 3/3: post-flight summary"
cat <<EOF

=== bulk back-fill commit complete ===
accepted:                     ${ACCEPTED}
skipped (already in posterior): ${SKIPPED_DUP}
audit log:                    ${AUDIT_LOG}
commit summary:               ${SUMMARY_JSON}

next: review .omx/state/continual_learning_posterior.json to confirm the new
anchor count, then proceed with the cathedral autopilot dispatch loop with
the updated posterior.
EOF
