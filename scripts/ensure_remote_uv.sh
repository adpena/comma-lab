#!/bin/bash
# Canonical fresh-host uv bootstrap for remote runners.
#
# stdout is reserved for the resolved uv binary path so callers can do:
#   UV_BIN="$(bash scripts/ensure_remote_uv.sh --symlink-system)"
# Logs go to stderr.

set -euo pipefail

SYMLINK_SYSTEM=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --symlink-system)
            SYMLINK_SYSTEM=1
            shift
            ;;
        *)
            echo "[ensure-uv] FATAL: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

candidate_uv() {
    if [ -n "${UV_BIN:-}" ] && [ -x "${UV_BIN:-}" ] && "$UV_BIN" --version >/dev/null 2>&1; then
        printf '%s\n' "$UV_BIN"
        return 0
    fi
    local command_uv
    command_uv="$(command -v uv 2>/dev/null || true)"
    if [ -n "$command_uv" ] && [ -x "$command_uv" ] && "$command_uv" --version >/dev/null 2>&1; then
        printf '%s\n' "$command_uv"
        return 0
    fi
    for candidate in \
        "$HOME/.local/bin/uv" \
        "$HOME/.cargo/bin/uv" \
        "/usr/local/bin/uv" \
        "/opt/conda/bin/uv"
    do
        if [ -x "$candidate" ] && "$candidate" --version >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

install_uv() {
    local log_path="${UV_BOOTSTRAP_LOG:-/tmp/uv_install.log}"
    echo "[ensure-uv] uv missing; installing via official astral bootstrap" >&2
    curl -LsSf https://astral.sh/uv/install.sh | sh > "$log_path" 2>&1 || {
        echo "[ensure-uv] FATAL: uv install failed; see $log_path" >&2
        tail -80 "$log_path" >&2 || true
        exit 7
    }
}

UV_PATH="$(candidate_uv || true)"
if [ -z "$UV_PATH" ]; then
    install_uv
    UV_PATH="$(candidate_uv || true)"
fi

if [ -z "$UV_PATH" ] || [ ! -x "$UV_PATH" ]; then
    echo "[ensure-uv] FATAL: uv missing after bootstrap" >&2
    exit 7
fi

if [ "$SYMLINK_SYSTEM" -eq 1 ] && [ "$(id -u)" = "0" ]; then
    if [ "$UV_PATH" != "/usr/local/bin/uv" ]; then
        if [ -e /usr/local/bin/uv ] && ! /usr/local/bin/uv --version >/dev/null 2>&1; then
            rm -f /usr/local/bin/uv 2>/dev/null || true
        fi
        ln -sf "$UV_PATH" /usr/local/bin/uv 2>/dev/null || true
    fi
fi

"$UV_PATH" --version >&2
printf '%s\n' "$UV_PATH"
