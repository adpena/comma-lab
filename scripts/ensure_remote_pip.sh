#!/bin/bash
# Canonical fresh-host pip bootstrap for remote runners.
#
# Bug class permanently extincted (2026-05-01):
#   ".venv/bin/python: No module named pip" on a venv that was created via
#   `uv venv` (or any other path that omits ensurepip). probe_nvdec.sh
#   then fails at `python -m pip install ...` and a $0.30+ Vast.ai dispatch
#   is wasted before the lane script can even start.
#
# Reference: feedback_loop_session_permanent_bug_class_extinction_20260501.md
# (Bug Class #3) + the user's "ensurepip" guidance during the loop session.
#
# Usage:
#   PYBIN=/path/to/python bash scripts/ensure_remote_pip.sh
#   bash scripts/ensure_remote_pip.sh /path/to/python   # positional override
#
# stdout is reserved for the resolved python path so callers can chain:
#   PIPBIN_PY="$(bash scripts/ensure_remote_pip.sh)"
# Logs go to stderr.
#
# Idempotent. Safe to call multiple times. The pip-already-present fast path
# is sub-100ms; the upgrade path costs ~3s on first invocation only.
#
# 2026-05-01: this is the canonical companion to scripts/ensure_remote_uv.sh.
# Any codepath that creates a venv but expects pip-based installers (DALI
# bootstrap, contest scorer pip-install, hash-pinned wheel install) MUST
# call this helper BEFORE the first `python -m pip ...` invocation. The
# matching preflight check is `check_venv_creators_use_ensurepip`.

set -euo pipefail

# Allow positional override; otherwise honor PYBIN env, otherwise system python3.
if [ "$#" -ge 1 ] && [ -n "$1" ]; then
    PYBIN="$1"
elif [ -n "${PYBIN:-}" ]; then
    PYBIN="$PYBIN"
elif command -v python3 >/dev/null 2>&1; then
    PYBIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYBIN="$(command -v python)"
else
    echo "[ensure-pip] FATAL: no python interpreter found on PATH and PYBIN unset" >&2
    exit 9
fi

if [ ! -x "$PYBIN" ]; then
    echo "[ensure-pip] FATAL: PYBIN=$PYBIN is not executable" >&2
    exit 9
fi

# Fast path: pip already importable, nothing to do.
if "$PYBIN" -c "import pip" >/dev/null 2>&1; then
    echo "[ensure-pip] OK: $PYBIN already has pip" >&2
    printf '%s\n' "$PYBIN"
    exit 0
fi

# Slow path: bootstrap pip via ensurepip. We do NOT pass --upgrade here on
# purpose — the PyTorch container's pip-already-present path is handled by
# the fast path above. Without --upgrade the bundled wheel just installs.
log_path="${PIP_BOOTSTRAP_LOG:-/tmp/pip_install.log}"
echo "[ensure-pip] pip missing on $PYBIN; bootstrapping via ensurepip" >&2
if "$PYBIN" -m ensurepip --default-pip > "$log_path" 2>&1; then
    echo "[ensure-pip] ensurepip --default-pip succeeded; verifying import" >&2
elif "$PYBIN" -m ensurepip > "$log_path" 2>&1; then
    echo "[ensure-pip] ensurepip (no --default-pip) succeeded; verifying import" >&2
else
    echo "[ensure-pip] FATAL: ensurepip failed; see $log_path" >&2
    tail -40 "$log_path" >&2 || true
    exit 9
fi

if ! "$PYBIN" -c "import pip" >/dev/null 2>&1; then
    echo "[ensure-pip] FATAL: ensurepip claimed success but pip still not importable" >&2
    exit 9
fi

echo "[ensure-pip] OK: pip now importable on $PYBIN" >&2
printf '%s\n' "$PYBIN"
