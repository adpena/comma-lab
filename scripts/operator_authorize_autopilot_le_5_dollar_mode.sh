#!/bin/bash
# Operator authorize: activate the cathedral autopilot's
# le-$5/individual / le-$20-cumulative mode for ONE loop iteration.
#
# Per CLAUDE.md "Operator gates must be wired and used" + MM autopilot
# activation landing (2026-05-11): the autopilot mode is DUAL-GATED:
#   1. CLI flag `--operator-authorized-le-5-dollar-mode` (default OFF)
#   2. env-var `CATHEDRAL_AUTOPILOT_OPERATOR_AUTHORIZED_MODE=1`
#   3. `--journal-path` (required when authorized mode is on)
#
# This script sets ALL three correctly + invokes the loop with the
# composition-matrix ranking JSON the operator has approved as the dispatch
# source. Per CLAUDE.md cross-agent dispatch coordination, the journal is
# the persistent record of every authorized dispatch.
#
# Cost (envelope): up to $20 cumulative within this run (per-dispatch cap $5).
# KILL events are NEVER auto-authorized — they remain operator-only.
#
# Per CLAUDE.md FORBIDDEN /tmp paths: journal lives under
# `experiments/results/autopilot_authorized_journal_<UTC>/`.
#
# Usage: bash scripts/operator_authorize_autopilot_le_5_dollar_mode.sh
#
# Operator decisions surfaced by this wrapper:
#   - which ranking JSON to consume (default: latest under
#     experiments/results/cathedral_autopilot_dispatch_ranking_*/ranking.json)
#   - whether to include out-of-envelope ranking candidates (default: NO)
#   - max dispatch recommendations per iteration (default: 8)
#
# Lane: lane_operator_one_command_authorize_scripts L0
# Cross-ref: project_operator_decision_dashboard_20260511.md N-3 + OD-3
#            feedback_cathedral_autopilot_activation_phase2_probes_integration_audit_v2_landed_20260511.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

UTC=$(date -u +%Y%m%dT%H%M%SZ)
RANKING_JSON="${AUTOPILOT_RANKING_JSON:-}"
JOURNAL_DIR="experiments/results/autopilot_authorized_journal_${UTC}"
JOURNAL_PATH="${JOURNAL_DIR}/journal.jsonl"
LOOP_REPORT="${JOURNAL_DIR}/autopilot_loop_report.json"
MAX_DISPATCHES="${AUTOPILOT_MAX_DISPATCHES:-8}"

if [ -z "$RANKING_JSON" ]; then
    # Pick the most recent ranking JSON.
    RANKING_JSON=$(ls -t experiments/results/cathedral_autopilot_dispatch_ranking_*/ranking.json 2>/dev/null | head -1 || true)
fi
if [ -z "$RANKING_JSON" ] || [ ! -f "$RANKING_JSON" ]; then
    echo "[autopilot-le-5] FATAL: no ranking JSON found." >&2
    echo "[autopilot-le-5] Generate one with the substrate composition matrix tool, or pass AUTOPILOT_RANKING_JSON=<path>." >&2
    exit 1
fi

mkdir -p "$JOURNAL_DIR"

cat <<EOF

=== cathedral autopilot le-\$5/individual mode operator confirmation ===

ranking JSON:            ${RANKING_JSON}
journal target:          ${JOURNAL_PATH}
loop report target:      ${LOOP_REPORT}
max dispatches:          ${MAX_DISPATCHES}

dual-gate activation:    env-var + CLI flag + journal path

per-dispatch cap:        \$5.00 USD
cumulative cap:          \$20.00 USD (envelope hard cap)
KILL auto-authorized?    NO — operator-only per CLAUDE.md

risk:                    up to \$20 cumulative GPU spend across all
                         approved dispatches in this run; each individual
                         dispatch \$5 max; HALT-and-ASK still fires for
                         any candidate exceeding either cap.

EOF
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

# Pre-flight: lane claim helper must exist (defense-in-depth per CLAUDE.md
# "CROSS-AGENT DISPATCH COORDINATION").
if [ ! -f "tools/claim_lane_dispatch.py" ]; then
    echo "[autopilot-le-5] FATAL: lane-claim helper missing at tools/claim_lane_dispatch.py" >&2
    exit 1
fi

echo "[autopilot-le-5] firing autopilot loop with dual-gate activation..."
.venv/bin/python tools/cathedral_autopilot_autonomous_loop.py \
    --use-substrate-composition-matrix-ranking "$RANKING_JSON" \
    --operator-authorized-le-5-dollar-mode \
    --journal-path "$JOURNAL_PATH" \
    --output "$LOOP_REPORT" \
    --iterations 1 \
    --max-dispatch-recommendations "$MAX_DISPATCHES"

echo "[autopilot-le-5] complete; review:"
echo "  - journal: ${JOURNAL_PATH}"
echo "  - loop report: ${LOOP_REPORT}"
echo "  - lane claims: .omx/state/active_lane_dispatch_claims.md"
