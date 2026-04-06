#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$ROOT/.tooling"
TOOLS_BIN="$TOOLS_DIR/node_modules/.bin"
LOCAL_BIN="$HOME/.local/bin"
VENV_DIR="$ROOT/.venv"
UPSTREAM_ROOT="$ROOT/workspace/upstream/comma_video_compression_challenge"
PROMPT_FILE="$ROOT/PROMPT.md"

log() {
  printf '\n[comma-lab] %s\n' "$*"
}

warn() {
  printf '\n[comma-lab][warn] %s\n' "$*" >&2
}

have() {
  command -v "$1" >/dev/null 2>&1
}

prepend_path() {
  case ":$PATH:" in
    *":$1:"*) ;;
    *) export PATH="$1:$PATH" ;;
  esac
}

ensure_basic_tools() {
  local missing=()
  for tool in git python3 curl; do
    if ! have "$tool"; then
      missing+=("$tool")
    fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    printf 'Missing required tools: %s\n' "${missing[*]}" >&2
    exit 1
  fi
}

ensure_python_env() {
  log "Preparing local Python environment"
  if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
  fi
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip setuptools wheel >/dev/null
  pip install -e "$ROOT" >/dev/null
}

ensure_uv() {
  prepend_path "$LOCAL_BIN"
  if have uv; then
    return 0
  fi

  log "Installing uv into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || {
    warn "Could not install uv automatically. Upstream env setup will be skipped."
    return 1
  }
  prepend_path "$LOCAL_BIN"
  if ! have uv; then
    warn "uv still not visible on PATH after install."
    return 1
  fi
}

ensure_node_lane() {
  prepend_path "$TOOLS_BIN"

  if ! have node || ! have npm; then
    warn "node/npm not found. I will skip the local Codex/OMX/Ralph install."
    return 1
  fi

  mkdir -p "$TOOLS_DIR"
  if [ ! -f "$TOOLS_DIR/package.json" ]; then
    cat > "$TOOLS_DIR/package.json" <<'JSON'
{
  "name": "comma-video-lab-tools",
  "private": true,
  "version": "0.1.0"
}
JSON
  fi

  log "Installing local Codex/OMX/Ralph tooling into .tooling/"
  if ! npm install --prefix "$TOOLS_DIR" @openai/codex oh-my-codex @iannuttall/ralph >/dev/null; then
    warn "Local Codex/OMX/Ralph install failed. You can retry later with npm inside .tooling/."
    return 1
  fi
  prepend_path "$TOOLS_BIN"

  if ! have omx; then
    warn "OMX install completed but 'omx' was not found on PATH."
    return 1
  fi
  return 0
}

detect_uv_group() {
  if [ -n "${CHALLENGE_UV_GROUP:-}" ]; then
    printf '%s' "$CHALLENGE_UV_GROUP"
    return 0
  fi

  case "$(uname -s)" in
    Darwin)
      printf 'mps'
      ;;
    Linux)
      if have nvidia-smi; then
        printf 'cu128'
      else
        printf 'cpu'
      fi
      ;;
    *)
      printf 'cpu'
      ;;
  esac
}

bootstrap_upstream() {
  log "Cloning or refreshing the upstream challenge repo"
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
  comma-lab bootstrap-upstream --dest "$UPSTREAM_ROOT"
  comma-lab install-submission exact_current --upstream-root "$UPSTREAM_ROOT"
  comma-lab install-submission robust_current --upstream-root "$UPSTREAM_ROOT"
}

setup_upstream_env() {
  local uv_group
  uv_group="$(detect_uv_group)"

  if ! have uv; then
    warn "Skipping upstream Python environment setup because uv is unavailable."
    return 0
  fi

  log "Syncing upstream dependencies with uv group '$uv_group'"
  (
    cd "$UPSTREAM_ROOT"
    uv sync --group "$uv_group" || warn "uv sync failed for group '$uv_group'. You can retry manually inside $UPSTREAM_ROOT."
  )
}

seed_state() {
  log "Ensuring durable OMX/Ralph state scaffolding exists"
  mkdir -p \
    "$ROOT/.omx/logs" \
    "$ROOT/.omx/memory" \
    "$ROOT/.omx/plans" \
    "$ROOT/.omx/research" \
    "$ROOT/.omx/sessions" \
    "$ROOT/.omx/state" \
    "$ROOT/.omx/team" \
    "$ROOT/.ralph" \
    "$ROOT/.agents/tasks" \
    "$ROOT/reports" \
    "$ROOT/experiments/runs" \
    "$ROOT/experiments/best"

  [ -f "$ROOT/reports/latest.md" ] || cat > "$ROOT/reports/latest.md" <<'MD'
# latest report

No promoted runs yet.
MD

  [ -f "$ROOT/.omx/state/current_focus.md" ] || cat > "$ROOT/.omx/state/current_focus.md" <<'MD'
# current focus

- Bootstrap complete.
- Verify Track A.
- Verify Track B.
- Queue the next 3 experiments.
MD

  [ -f "$ROOT/.omx/state/next_experiments.md" ] || cat > "$ROOT/.omx/state/next_experiments.md" <<'MD'
# next experiments

1. Exact-current smoke.
2. Robust-current packaging smoke.
3. First x265 floor sweep.
MD

  [ -f "$ROOT/.omx/research/findings.md" ] || cat > "$ROOT/.omx/research/findings.md" <<'MD'
# findings

Record measured findings here.
MD

  [ -f "$ROOT/.ralph/run_log.md" ] || cat > "$ROOT/.ralph/run_log.md" <<'MD'
# Ralph run log

Record each meaningful iteration here.
MD
}

run_omx_setup() {
  if ! have omx; then
    warn "OMX is unavailable, so I cannot auto-launch the Ralph loop."
    return 0
  fi

  log "Running 'omx setup' and 'omx doctor' in this repo"
  (
    cd "$ROOT"
    omx setup || true
    omx doctor || true
  )
}

launch_omx() {
  if ! have omx; then
    printf '\nNext step once Codex + OMX are installed:\n' >&2
    printf '  cd %s\n' "$ROOT" >&2
    printf '  omx --madmax --high\n' >&2
    printf '  paste %s\n' "$PROMPT_FILE" >&2
    return 0
  fi

  log "About to launch OMX"
  printf '\nPaste this file when the Codex/OMX session opens:\n  %s\n' "$PROMPT_FILE"
  printf 'If you prefer a lighter start, exit and relaunch with: omx\n\n'
  (
    cd "$ROOT"
    omx --madmax --high || true
  )
}

main() {
  ensure_basic_tools
  ensure_python_env
  ensure_uv || true
  ensure_node_lane || true
  bootstrap_upstream
  setup_upstream_env
  seed_state
  run_omx_setup

  log "Bootstrap complete"
  printf '\nRepo root: %s\n' "$ROOT"
  printf 'Upstream root: %s\n' "$UPSTREAM_ROOT"
  printf 'Prompt file: %s\n' "$PROMPT_FILE"

  launch_omx
}

main "$@"
