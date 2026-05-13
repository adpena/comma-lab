#!/usr/bin/env bash
# Ensure FFMPEG_BIN points at an ffmpeg build that supports the explicit
# color-contract scale options required by submissions/robust_current/inflate.sh.
#
# This script is deliberately provider-agnostic. It can run on GitHub Actions,
# Modal/Vast bootstrap shells, or local Linux CI. It prefers an already-good
# FFMPEG_BIN/PATH binary, then falls back to the BtbN master build that has
# repeatedly matched the contest inflate color contract.
set -euo pipefail

INSTALL_ROOT="${RUNNER_TEMP:-/tmp}/ffmpeg-btbn"
ENV_FILE="${GITHUB_ENV:-}"
PATH_FILE="${GITHUB_PATH:-}"
ALLOW_MISSING_CONTRACT=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --install-root)
      INSTALL_ROOT="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --path-file)
      PATH_FILE="$2"
      shift 2
      ;;
    --allow-missing-contract)
      ALLOW_MISSING_CONTRACT=1
      shift
      ;;
    -h|--help)
      sed -n '1,80p' "$0"
      exit 0
      ;;
    *)
      echo "usage: $0 [--install-root PATH] [--env-file PATH] [--path-file PATH]" >&2
      exit 2
      ;;
  esac
done

log() {
  printf '[ensure-parity-ffmpeg] %s\n' "$*" >&2
}

resolve_candidate() {
  local candidate="${FFMPEG_BIN:-}"
  if [ -n "$candidate" ]; then
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
    log "FFMPEG_BIN=$candidate is not executable; falling back to PATH lookup"
  fi

  if command -v ffmpeg >/dev/null 2>&1; then
    command -v ffmpeg
    return 0
  fi
  return 1
}

has_scale_contract() {
  local binary="$1"
  local scale_help
  scale_help="$("$binary" -hide_banner -h filter=scale 2>&1 || true)"
  local required_opt
  for required_opt in in_range out_range in_color_matrix in_primaries in_transfer; do
    if ! printf '%s\n' "$scale_help" | grep -q "$required_opt"; then
      return 1
    fi
  done
  return 0
}

install_btbn() {
  local url="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
  local tmp="${TMPDIR:-/tmp}/ffmpeg-btbn.tar.xz"
  local i

  rm -rf "$INSTALL_ROOT"
  mkdir -p "$INSTALL_ROOT"
  for i in 1 2 3; do
    log "downloading BtbN ffmpeg master attempt $i"
    curl -fL --retry 5 --retry-delay 3 -o "$tmp" "$url" 2>&1 | tail -3 >&2
    local actual
    actual="$(stat -c%s "$tmp" 2>/dev/null || wc -c < "$tmp" 2>/dev/null || echo 0)"
    if [ "$actual" -gt 50000000 ]; then
      break
    fi
    log "download looked truncated (${actual} bytes); retrying"
    sleep 3
  done
  tar -xf "$tmp" -C "$INSTALL_ROOT" --strip-components=1
  if [ ! -x "$INSTALL_ROOT/bin/ffmpeg" ]; then
    log "FATAL: BtbN ffmpeg not found after extract at $INSTALL_ROOT/bin/ffmpeg"
    exit 8
  fi
  printf '%s\n' "$INSTALL_ROOT/bin/ffmpeg"
}

resolved=""
if resolved="$(resolve_candidate 2>/dev/null)" && has_scale_contract "$resolved"; then
  log "existing ffmpeg satisfies color contract: $resolved"
else
  if [ -n "$resolved" ]; then
    log "ffmpeg $resolved lacks explicit color-contract scale options"
  else
    log "ffmpeg missing from PATH"
  fi
  resolved="$(install_btbn)"
  if ! has_scale_contract "$resolved"; then
    if [ "$ALLOW_MISSING_CONTRACT" -eq 1 ]; then
      log "WARNING: selected ffmpeg lacks explicit color-contract scale options; downstream inflate.sh will fail closed if the selected runtime mode needs them"
    else
    log "FATAL: installed BtbN ffmpeg still lacks required scale options"
    exit 8
    fi
  fi
fi

export FFMPEG_BIN="$resolved"
log "selected FFMPEG_BIN=$FFMPEG_BIN"
"$FFMPEG_BIN" -version | head -1 >&2

if [ -n "$ENV_FILE" ]; then
  printf 'FFMPEG_BIN=%s\n' "$FFMPEG_BIN" >> "$ENV_FILE"
fi
if [ -n "$PATH_FILE" ]; then
  dirname "$FFMPEG_BIN" >> "$PATH_FILE"
fi
printf '%s\n' "$FFMPEG_BIN"
