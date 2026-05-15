#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# scripts/cron_harvest_modal.sh — operator-installable cron wrapper for the
# canonical Modal harvest helper.
#
# Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE — NON-NEGOTIABLE, HIGHEST
# EMPHASIS" the FunctionCall result cache has a ~24h TTL; every dispatch via
# `experiments/modal_train_lane.py` MUST be followed by a scheduled harvest
# within 24h. This wrapper is the recommended cadence for active dispatch
# periods (operator-installable; no auto-install).
#
# Usage:
#   ./scripts/cron_harvest_modal.sh              # log to .omx/state/_modal_harvest_cron.log
#   ./scripts/cron_harvest_modal.sh --dry-run    # show command without running
#
# Suggested crontab entry (every 4 hours):
#   0 */4 * * * cd /Users/<you>/Projects/pact && ./scripts/cron_harvest_modal.sh >> /dev/null 2>&1
#
# The wrapper itself logs each invocation (start/end timestamps + rc + summary
# row count) to `.omx/state/_modal_harvest_cron.log`. The harvest tool's own
# JSON summary lives at `experiments/results/_modal_harvest_summary.json`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/.omx/state"
LOG_FILE="$LOG_DIR/_modal_harvest_cron.log"
mkdir -p "$LOG_DIR"

PY="$REPO_ROOT/.venv/bin/python"
HARVEST_TOOL="$REPO_ROOT/tools/harvest_modal_calls.py"
FRESH_TOOL="$REPO_ROOT/tools/check_modal_harvest_freshness.py"

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,32p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if [[ ! -x "$PY" ]]; then
    echo "ERROR: python venv not found at $PY" >&2
    exit 2
fi

if [[ ! -f "$HARVEST_TOOL" ]]; then
    echo "ERROR: harvest tool not found at $HARVEST_TOOL" >&2
    exit 2
fi

START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY-RUN] would invoke: $PY $HARVEST_TOOL --execute"
    echo "[DRY-RUN] would log to:  $LOG_FILE"
    exit 0
fi

{
    echo "---"
    echo "cron_harvest_modal start_utc=$START_UTC"
    "$PY" "$HARVEST_TOOL" --execute || true
    END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "cron_harvest_modal end_utc=$END_UTC"
    if [[ -x "$FRESH_TOOL" || -f "$FRESH_TOOL" ]]; then
        echo "freshness_check_after_harvest:"
        "$PY" "$FRESH_TOOL" || true
    fi
} >> "$LOG_FILE" 2>&1

exit 0
