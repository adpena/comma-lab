#!/usr/bin/env bash
# Canonical uv installer for remote runners.
#
# Recovery note: this helper was lost when subagent worktrees were
# auto-cleaned without committing source. Rebuilt 2026-05-04 from the call
# contract in scripts/remote_archive_only_eval.sh:
#
#   bootstrap_runtime_deps() invokes:
#     UV_BIN="$(bash "$WORKSPACE/scripts/ensure_remote_uv.sh" --symlink-system)"
#
#   - Must echo the absolute path to a working `uv` binary on stdout.
#   - Must idempotently install uv if not present (curl https://astral.sh/uv/install.sh | sh).
#   - --symlink-system: also symlink uv into /usr/local/bin (or PATH-visible
#     location) so subprocesses don't need to know the install path.
#
# CLAUDE.md non-negotiable: this is the ONE canonical uv installer. Any new
# remote script MUST source remote_archive_only_eval.sh's bootstrap_runtime_deps
# OR call this helper directly — NEVER copy-paste the curl install line.
#
# Cost of NOT using this canonical path (2026-05-01 loop session):
# 6 sequential bug-class re-discoveries on 4 destroyed Vast.ai instances
# burning ~$1.50 + 30 min wall-clock chasing the same lesson.

set -euo pipefail

SYMLINK_SYSTEM=0
for arg in "$@"; do
    case "$arg" in
        --symlink-system) SYMLINK_SYSTEM=1 ;;
        --help|-h)
            cat >&2 <<'HELP'
usage: ensure_remote_uv.sh [--symlink-system]

Idempotently install uv on the current host. Echoes the absolute path
to the working `uv` binary on stdout. Logs progress to stderr.

  --symlink-system    Symlink uv into /usr/local/bin so subprocesses on
                      $PATH can find it without knowing the install path.
HELP
            exit 0
            ;;
        *)
            echo "ensure_remote_uv.sh: unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

log() { echo "[ensure-remote-uv] $(date -u +%FT%TZ) $*" >&2; }

# 1. If uv is already on PATH, use it
if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    log "uv already installed at $UV_BIN"
    if [ "$SYMLINK_SYSTEM" = 1 ] && [ "$UV_BIN" != "/usr/local/bin/uv" ] && [ -w /usr/local/bin ] 2>/dev/null; then
        ln -sf "$UV_BIN" /usr/local/bin/uv 2>/dev/null || true
    fi
    echo "$UV_BIN"
    exit 0
fi

# 2. Check standard install locations before re-installing
for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" /usr/local/bin/uv /usr/bin/uv; do
    if [ -x "$candidate" ]; then
        log "uv found at $candidate (not on PATH)"
        if [ "$SYMLINK_SYSTEM" = 1 ] && [ "$candidate" != "/usr/local/bin/uv" ] && [ -w /usr/local/bin ] 2>/dev/null; then
            ln -sf "$candidate" /usr/local/bin/uv 2>/dev/null || true
        fi
        echo "$candidate"
        exit 0
    fi
done

# 3. Install uv via the official Astral installer
log "installing uv via https://astral.sh/uv/install.sh"
if ! command -v curl >/dev/null 2>&1; then
    log "FATAL: curl not found; cannot fetch uv installer"
    exit 3
fi

# Run installer; defaults to ~/.local/bin (or honors UV_INSTALL_DIR if set)
if ! curl -LsSf https://astral.sh/uv/install.sh | sh >&2; then
    log "FATAL: uv installer failed"
    exit 4
fi

# 4. Locate the freshly-installed binary
NEW_UV=""
for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" /usr/local/bin/uv; do
    if [ -x "$candidate" ]; then
        NEW_UV="$candidate"
        break
    fi
done

if [ -z "$NEW_UV" ]; then
    log "FATAL: uv installed but binary not found in standard locations"
    exit 5
fi

log "uv installed at $NEW_UV"

if [ "$SYMLINK_SYSTEM" = 1 ] && [ "$NEW_UV" != "/usr/local/bin/uv" ] && [ -w /usr/local/bin ] 2>/dev/null; then
    ln -sf "$NEW_UV" /usr/local/bin/uv 2>/dev/null || log "WARN: could not symlink to /usr/local/bin (continuing)"
fi

echo "$NEW_UV"
