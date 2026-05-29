#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# tools/schedule_canonical_modal_harvest_cron.sh
#
# CANONICAL RECURRING MODAL HARVEST SCHEDULER (Surface A of the canonical
# 2-landing pattern per CLAUDE.md "Bugs must be permanently fixed AND
# self-protected against" non-negotiable + RECOVERY-AUDIT-V2 op-routable #1
# + STAND-DOWN-REVIEW-AUDIT TOP-2 op-routable).
#
# Wraps the canonical `tools/harvest_modal_calls.py --execute --from-ledger`
# invocation as a recurring scheduled operation per the operator's NON-
# NEGOTIABLE directive ("no ad-hoc, no signal loss, no rediscovery of the
# same thing over and over, no duplicate code, no drift"). Extincts the
# silent-orphan-harvest bug class (`modal_dispatch_succeeded_but_canonical_
# ledger_outcome_never_registered_silent_orphan_harvest_v1`) at the
# OPERATIONAL layer per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE"
# non-negotiable HIGHEST EMPHASIS.
#
# Sister of the STRUCTURAL layer at Catalog #380 STRICT preflight gate
# (`check_dispatch_wrappers_pair_with_harvest_scheduler_invocation`) which
# refuses dispatch wrappers without paired harvest-scheduler invocation.
#
# Auto-detects OS (macOS launchd vs Linux cron) and emits the canonical
# scheduler entry. Idempotent: detects existing entry and updates OR no-op.
# Logs every install / uninstall / run-trigger to the canonical fcntl-
# locked JSONL ledger at `.omx/state/modal_harvest_cron_log.jsonl` per
# Catalog #131 sister discipline.
#
# CLI:
#   --install                Install the canonical scheduler entry (default).
#   --uninstall              Remove the canonical scheduler entry.
#   --status                 Print current install state + last-run timing.
#   --cadence-hours <float>  Recurring cadence (default: 2.0; range 0.5-24.0
#                            per RECOVERY-AUDIT-V2 recommendation).
#   --dry-run                Print what would change without mutating anything.
#   --help                   Print this help.
#
# Exit codes:
#   0  — success (install / uninstall / status)
#   1  — IO error (missing repo, unwritable plist/crontab, etc.)
#   2  — CLI / arg error
#   3  — unsupported OS (not Darwin / Linux)
#
# Canonical reference: Catalog #245 (Modal call_id ledger 4-layer pattern;
# this tool keeps the ledger current via recurring harvest), Catalog #339
# / #360 (silent-no-spawn extinction; this tool extincts the sister POST-
# HARVEST silent-orphan surface), Catalog #380 (sister STRICT preflight
# gate that refuses dispatch wrappers without paired scheduler invocation).

set -euo pipefail

# --- Constants -------------------------------------------------------------

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
HARVEST_TOOL_REL="tools/harvest_modal_calls.py"
HARVEST_TOOL="${REPO_ROOT}/${HARVEST_TOOL_REL}"
VENV_PY="${REPO_ROOT}/.venv/bin/python"
LEDGER_REL=".omx/state/modal_harvest_cron_log.jsonl"
LEDGER="${REPO_ROOT}/${LEDGER_REL}"
LEDGER_LOCK="${LEDGER}.lock"
LAUNCHD_LABEL="com.pact.modal_harvest_canonical"
LAUNCHD_PLIST_USER="${HOME}/Library/LaunchAgents/${LAUNCHD_LABEL}.plist"
CRON_MARKER="# tools/schedule_canonical_modal_harvest_cron.sh::CANONICAL_MODAL_HARVEST"

# --- Args ------------------------------------------------------------------

ACTION="install"
CADENCE_HOURS="2.0"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install) ACTION="install"; shift ;;
        --uninstall) ACTION="uninstall"; shift ;;
        --status) ACTION="status"; shift ;;
        --cadence-hours) CADENCE_HOURS="$2"; shift 2 ;;
        --dry-run) DRY_RUN="1"; shift ;;
        --help|-h)
            sed -n '3,42p' "${BASH_SOURCE[0]}" | sed 's/^# //; s/^#//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

# Validate cadence range.
if ! python3 -c "
import sys
try:
    v = float('${CADENCE_HOURS}')
    if not (0.5 <= v <= 24.0):
        sys.exit(1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    echo "ERROR: --cadence-hours must be a float in [0.5, 24.0]; got: ${CADENCE_HOURS}" >&2
    exit 2
fi

# --- OS detection ----------------------------------------------------------

OS_NAME="$(uname -s)"
case "${OS_NAME}" in
    Darwin)  SCHEDULER="launchd" ;;
    Linux)   SCHEDULER="cron" ;;
    *)
        echo "ERROR: unsupported OS: ${OS_NAME} (expected Darwin or Linux)" >&2
        exit 3
        ;;
esac

# --- Sanity ----------------------------------------------------------------

if [[ ! -f "${HARVEST_TOOL}" ]]; then
    echo "ERROR: canonical harvest tool missing at ${HARVEST_TOOL}" >&2
    exit 1
fi

if [[ ! -x "${VENV_PY}" ]]; then
    echo "ERROR: canonical venv python missing at ${VENV_PY}" >&2
    exit 1
fi

# --- Canonical fcntl-locked ledger append (Catalog #131 sister) -----------

_utc_now() {
    python3 -c "import datetime; print(datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))"
}

_log_event() {
    local event_type="$1"
    local note="$2"
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    mkdir -p "$(dirname -- "${LEDGER}")"
    local utc; utc="$(_utc_now)"
    local pid="$$"
    local host; host="$(hostname -s 2>/dev/null || hostname || echo unknown)"
    # Append via Python helper for fcntl LOCK_EX (matches Catalog #131 pattern).
    "${VENV_PY}" - <<PYEOF
import fcntl
import json
import os
from pathlib import Path

path = Path(${LEDGER@Q})
lock_path = Path(${LEDGER_LOCK@Q})
record = {
    "event_type": ${event_type@Q},
    "scheduler": ${SCHEDULER@Q},
    "action": ${ACTION@Q},
    "cadence_hours": float(${CADENCE_HOURS@Q}),
    "dry_run": ${DRY_RUN@Q} == "1",
    "note": ${note@Q},
    "written_at_utc": ${utc@Q},
    "written_pid": ${pid@Q},
    "written_host": ${host@Q},
}
line = json.dumps(record, sort_keys=True) + "\n"
with open(lock_path, "a") as lock_fh:
    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
    finally:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
PYEOF
}

# --- launchd (macOS) -------------------------------------------------------

_launchd_emit_plist() {
    local interval_seconds
    interval_seconds="$(python3 -c "print(int(float('${CADENCE_HOURS}') * 3600))")"
    cat <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${VENV_PY}</string>
        <string>${HARVEST_TOOL}</string>
        <string>--execute</string>
        <string>--from-ledger</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${REPO_ROOT}</string>
    <key>StartInterval</key>
    <integer>${interval_seconds}</integer>
    <key>RunAtLoad</key>
    <false/>
    <key>StandardOutPath</key>
    <string>${REPO_ROOT}/.omx/state/modal_harvest_cron_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${REPO_ROOT}/.omx/state/modal_harvest_cron_stderr.log</string>
</dict>
</plist>
PLIST
}

_launchd_install() {
    local plist_dir; plist_dir="$(dirname -- "${LAUNCHD_PLIST_USER}")"
    mkdir -p "${plist_dir}"

    # Idempotent: if existing entry present, unload first then reinstall
    # (matches canonical update-or-no-op semantics).
    if [[ -f "${LAUNCHD_PLIST_USER}" ]]; then
        if [[ "${DRY_RUN}" == "1" ]]; then
            echo "[dry-run] would unload + reinstall ${LAUNCHD_PLIST_USER}"
        else
            launchctl unload "${LAUNCHD_PLIST_USER}" 2>/dev/null || true
        fi
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry-run] would write plist to ${LAUNCHD_PLIST_USER}:"
        _launchd_emit_plist | sed 's/^/    /'
        echo "[dry-run] would launchctl load ${LAUNCHD_PLIST_USER}"
        return 0
    fi

    _launchd_emit_plist > "${LAUNCHD_PLIST_USER}"
    launchctl load "${LAUNCHD_PLIST_USER}"
    _log_event "scheduler_installed" "launchd label=${LAUNCHD_LABEL} cadence_hours=${CADENCE_HOURS}"
    echo "INSTALLED: ${LAUNCHD_PLIST_USER}"
    echo "  label:         ${LAUNCHD_LABEL}"
    echo "  cadence_hours: ${CADENCE_HOURS}"
    echo "  verify:        launchctl list | grep ${LAUNCHD_LABEL}"
    echo "  uninstall:     $0 --uninstall"
}

_launchd_uninstall() {
    if [[ ! -f "${LAUNCHD_PLIST_USER}" ]]; then
        echo "NOT INSTALLED: ${LAUNCHD_PLIST_USER} does not exist; no-op"
        return 0
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry-run] would launchctl unload + rm ${LAUNCHD_PLIST_USER}"
        return 0
    fi
    launchctl unload "${LAUNCHD_PLIST_USER}" 2>/dev/null || true
    rm -f "${LAUNCHD_PLIST_USER}"
    _log_event "scheduler_uninstalled" "launchd label=${LAUNCHD_LABEL}"
    echo "UNINSTALLED: ${LAUNCHD_PLIST_USER}"
}

_launchd_status() {
    if [[ -f "${LAUNCHD_PLIST_USER}" ]]; then
        echo "INSTALLED: ${LAUNCHD_PLIST_USER}"
        launchctl list 2>/dev/null | awk -v lbl="${LAUNCHD_LABEL}" '$3 == lbl {print "  launchctl:", $0}' || true
    else
        echo "NOT INSTALLED: ${LAUNCHD_PLIST_USER}"
    fi
}

# --- cron (Linux) ----------------------------------------------------------

_cron_emit_entry() {
    # Cadence translation: float-hours → minute spec. For sub-hour cadence
    # use minute granularity; for hour-aligned use 0 minute + */N hours.
    local cadence_min
    cadence_min="$(python3 -c "print(int(float('${CADENCE_HOURS}') * 60))")"
    if (( cadence_min < 60 )); then
        echo "*/${cadence_min} * * * * cd ${REPO_ROOT} && ${VENV_PY} ${HARVEST_TOOL} --execute --from-ledger >> ${REPO_ROOT}/.omx/state/modal_harvest_cron_stdout.log 2>> ${REPO_ROOT}/.omx/state/modal_harvest_cron_stderr.log ${CRON_MARKER}"
    else
        local cadence_h=$(( cadence_min / 60 ))
        echo "0 */${cadence_h} * * * cd ${REPO_ROOT} && ${VENV_PY} ${HARVEST_TOOL} --execute --from-ledger >> ${REPO_ROOT}/.omx/state/modal_harvest_cron_stdout.log 2>> ${REPO_ROOT}/.omx/state/modal_harvest_cron_stderr.log ${CRON_MARKER}"
    fi
}

_cron_install() {
    local existing new entry
    entry="$(_cron_emit_entry)"
    existing="$(crontab -l 2>/dev/null || true)"
    # Strip any existing canonical-marker entry, then append the new one.
    new="$(printf '%s\n' "${existing}" | grep -vF "${CRON_MARKER}" || true)"
    if [[ -n "${new}" ]]; then
        new="${new}"$'\n'"${entry}"
    else
        new="${entry}"
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry-run] would install crontab entry:"
        echo "    ${entry}"
        return 0
    fi

    echo "${new}" | crontab -
    _log_event "scheduler_installed" "cron marker=${CRON_MARKER} cadence_hours=${CADENCE_HOURS}"
    echo "INSTALLED: crontab entry"
    echo "  marker:        ${CRON_MARKER}"
    echo "  cadence_hours: ${CADENCE_HOURS}"
    echo "  verify:        crontab -l | grep CANONICAL_MODAL_HARVEST"
    echo "  uninstall:     $0 --uninstall"
}

_cron_uninstall() {
    local existing new
    existing="$(crontab -l 2>/dev/null || true)"
    if ! printf '%s\n' "${existing}" | grep -qF "${CRON_MARKER}"; then
        echo "NOT INSTALLED: no canonical-marker entry in crontab; no-op"
        return 0
    fi
    new="$(printf '%s\n' "${existing}" | grep -vF "${CRON_MARKER}" || true)"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry-run] would remove crontab entry marked with: ${CRON_MARKER}"
        return 0
    fi
    if [[ -z "${new}" ]]; then
        crontab -r 2>/dev/null || true
    else
        echo "${new}" | crontab -
    fi
    _log_event "scheduler_uninstalled" "cron marker=${CRON_MARKER}"
    echo "UNINSTALLED: crontab entry marked with ${CRON_MARKER}"
}

_cron_status() {
    local existing
    existing="$(crontab -l 2>/dev/null || true)"
    if printf '%s\n' "${existing}" | grep -qF "${CRON_MARKER}"; then
        echo "INSTALLED: crontab entry marked with ${CRON_MARKER}"
        printf '%s\n' "${existing}" | grep -F "${CRON_MARKER}" | sed 's/^/  /'
    else
        echo "NOT INSTALLED: no canonical-marker entry in crontab"
    fi
}

# --- Dispatch --------------------------------------------------------------

case "${ACTION}_${SCHEDULER}" in
    install_launchd) _launchd_install ;;
    uninstall_launchd) _launchd_uninstall ;;
    status_launchd) _launchd_status ;;
    install_cron) _cron_install ;;
    uninstall_cron) _cron_uninstall ;;
    status_cron) _cron_status ;;
    *)
        echo "ERROR: invalid action/scheduler combination: ${ACTION}/${SCHEDULER}" >&2
        exit 2
        ;;
esac
