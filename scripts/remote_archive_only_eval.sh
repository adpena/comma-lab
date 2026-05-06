#!/bin/bash
# Generic archive-only contest_auth_eval wrapper.
#
# Use case: a pre-built archive.zip exists locally (e.g., from local OWV3 byte-plan
# sweep). Skip all training/build stages and just run contest_auth_eval.py against
# the SCP'd archive on a CUDA host.
#
# Required env:
#   ARCHIVE_PATH: path to the archive.zip on the remote (typically /workspace/pact/iter_0/archive.zip)
#   ARCHIVE_LABEL: human label for log lines
#   PREDICTED_BAND: 2-element list [LOW, HIGH] (recorded in provenance)
#   CONTROLLED_BASELINE: human description of the baseline this is paired against
#
# Pre-flight gates:
#   * CUDA + NVDEC probe
#   * archive.zip exists at ARCHIVE_PATH
#   * upstream/{videos,models} present
#
# Per CLAUDE.md NEVER-INVENT-CLI-FLAGS: every flag passed to contest_auth_eval.py
# was verified by argparse-grep.

set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE:${PYTHONPATH:-}"
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

ARCHIVE_PATH="${ARCHIVE_PATH:-/workspace/pact/iter_0/archive.zip}"
ARCHIVE_LABEL="${ARCHIVE_LABEL:-archive_only_eval}"
INFLATE_SH="${INFLATE_SH:-submissions/robust_current/inflate.sh}"
PREDICTED_LOW="${PREDICTED_LOW:-0.50}"
PREDICTED_HIGH="${PREDICTED_HIGH:-1.05}"
CONTROLLED_BASELINE="${CONTROLLED_BASELINE:-lane_g_v3_pfp16_a_plus_plus_t4 (PFP16 frontier 1.044 [contest-CUDA T4])}"
KEEP_EVAL_WORK="${KEEP_EVAL_WORK:-0}"

LOG_DIR="${LOG_DIR:-$WORKSPACE/${ARCHIVE_LABEL}_results}"
mkdir -p "$LOG_DIR"
PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)

log() { echo "[archive-only-eval] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

resolve_inflate_sh() {
    case "$INFLATE_SH" in
        ""|*..*|*\\*)
            log "FATAL: unsafe INFLATE_SH path: $INFLATE_SH"
            exit 12
            ;;
    esac
    if [[ "$INFLATE_SH" = /* ]]; then
        INFLATE_SH_ABS="$INFLATE_SH"
    else
        INFLATE_SH_ABS="$WORKSPACE/$INFLATE_SH"
    fi
    if [ ! -f "$INFLATE_SH_ABS" ]; then
        log "FATAL: INFLATE_SH missing: $INFLATE_SH_ABS"
        exit 12
    fi
    INFLATE_SH_SHA="$(sha256sum "$INFLATE_SH_ABS" 2>/dev/null | cut -d ' ' -f 1 || shasum -a 256 "$INFLATE_SH_ABS" | cut -d ' ' -f 1)"
    log "inflate_sh: $INFLATE_SH_ABS sha=$INFLATE_SH_SHA"
}

bootstrap_runtime_deps() {
    # Self-bootstrap: install uv + ffmpeg + strip macOS resource forks if missing.
    # Bug-class extincts: uv-not-on-PATH, ffmpeg-missing, ._ resource-fork in upstream.
    # See feedback_uv_not_on_path_vast_instance_20260501.md, feedback_vast_cuda_driver_too_old_silent_cpu_fallback_20260501.md.
    if ! command -v uv >/dev/null 2>&1; then
        log "BOOTSTRAP: uv missing — invoking scripts/ensure_remote_uv.sh"
        if [ ! -f "$WORKSPACE/scripts/ensure_remote_uv.sh" ]; then
            log "FATAL: scripts/ensure_remote_uv.sh missing; remote source bundle incomplete"
            exit 7
        fi
        UV_BIN="$(bash "$WORKSPACE/scripts/ensure_remote_uv.sh" --symlink-system)"
        export UV_BIN
        export PATH="$(dirname "$UV_BIN"):$HOME/.local/bin:$PATH"
    fi
    if ! command -v ffmpeg >/dev/null 2>&1 && [ ! -x /workspace/ffmpeg-btbn/bin/ffmpeg ]; then
        log "BOOTSTRAP: ffmpeg missing — installing via apt"
        DEBIAN_FRONTEND=noninteractive apt-get update -qq > /tmp/apt.log 2>&1
        DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg > /tmp/apt_ffmpeg.log 2>&1 || {
            log "FATAL: apt ffmpeg install failed; see /tmp/apt_ffmpeg.log"
            exit 8
        }
    fi
    # Strip macOS resource forks from upstream/ that break contest_auth_eval validator.
    if [ -d "$WORKSPACE/upstream" ]; then
        find "$WORKSPACE/upstream" -name '._*' -delete 2>/dev/null || true
        find "$WORKSPACE/upstream" -name '.DS_Store' -delete 2>/dev/null || true
    fi
}

require_uv_and_ffmpeg_contract() {
    bootstrap_runtime_deps
    if ! command -v uv >/dev/null 2>&1; then
        log "FATAL: uv missing after bootstrap; robust_current/inflate.sh requires uv"
        exit 7
    fi
    export UV_BIN="${UV_BIN:-$(command -v uv)}"
    export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$LOG_DIR/uv_project_env}"
    export INFLATE_BROTLI_SPEC="${INFLATE_BROTLI_SPEC:-brotli==1.2.0}"
    export INFLATE_AV_SPEC="${INFLATE_AV_SPEC:-av==17.0.1}"
    export INFLATE_NUMPY_SPEC="${INFLATE_NUMPY_SPEC:-numpy==2.4.4}"
    if [ -z "${INFLATE_TORCH_SPEC:-}" ]; then
        local driver_major
        driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || true)"
        if [ -n "$driver_major" ] && [ "$driver_major" -lt 580 ] 2>/dev/null; then
            export INFLATE_TORCH_SPEC="torch==2.5.1+cu124"
            export UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
            export UV_INDEX_STRATEGY="${UV_INDEX_STRATEGY:-unsafe-best-match}"
        else
            export INFLATE_TORCH_SPEC="torch==2.11.0"
        fi
    else
        export INFLATE_TORCH_SPEC
    fi

    local candidate="${FFMPEG_BIN:-}"
    if [ -z "$candidate" ] && [ -x /workspace/ffmpeg-btbn/bin/ffmpeg ]; then
        candidate="/workspace/ffmpeg-btbn/bin/ffmpeg"
    fi
    if [ -z "$candidate" ]; then
        candidate="$(command -v ffmpeg || true)"
    fi
    if [ -z "$candidate" ]; then
        log "FATAL: ffmpeg missing; inflate color-contract cannot be audited"
        exit 8
    fi

    local resolved
    resolved="$(command -v "$candidate" 2>/dev/null || true)"
    if [ -z "$resolved" ] && [ -x "$candidate" ]; then
        resolved="$candidate"
    fi
    if [ -z "$resolved" ]; then
        log "FATAL: FFMPEG_BIN=$candidate is not executable"
        exit 8
    fi

    local scale_help
    scale_help="$("$resolved" -hide_banner -h filter=scale 2>&1 || true)"
    local required_opt
    local needs_btbn=0
    for required_opt in in_range out_range in_color_matrix in_primaries in_transfer; do
        if ! printf '%s\n' "$scale_help" | grep -q "$required_opt"; then
            needs_btbn=1
            break
        fi
    done
    if [ "$needs_btbn" -eq 1 ]; then
        # BOOTSTRAP: system ffmpeg too old; download BtbN static build with retry.
        log "BOOTSTRAP: ffmpeg $resolved lacks scale options; installing BtbN master with retry"
        local btbn_url="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
        local i
        for i in 1 2 3; do
            curl -fL --retry 5 --retry-delay 3 -o /tmp/ffmpeg-btbn.tar.xz "$btbn_url" 2>&1 | tail -3
            local actual
            actual="$(stat -c%s /tmp/ffmpeg-btbn.tar.xz 2>/dev/null || echo 0)"
            if [ "$actual" -gt 50000000 ]; then
                log "BOOTSTRAP: BtbN download OK ($actual bytes) on attempt $i"
                break
            fi
            log "BOOTSTRAP: BtbN attempt $i truncated ($actual bytes), retrying"
            sleep 3
        done
        mkdir -p /workspace/ffmpeg-btbn
        tar -xf /tmp/ffmpeg-btbn.tar.xz -C /workspace/ffmpeg-btbn --strip-components=1 || {
            log "FATAL: BtbN extract failed"
            exit 8
        }
        if [ ! -x /workspace/ffmpeg-btbn/bin/ffmpeg ]; then
            log "FATAL: BtbN ffmpeg not present after extract"
            exit 8
        fi
        resolved="/workspace/ffmpeg-btbn/bin/ffmpeg"
        scale_help="$("$resolved" -hide_banner -h filter=scale 2>&1 || true)"
        for required_opt in in_range out_range in_color_matrix in_primaries in_transfer; do
            if ! printf '%s\n' "$scale_help" | grep -q "$required_opt"; then
                log "FATAL: even BtbN ffmpeg lacks $required_opt; investigate"
                exit 8
            fi
        done
    fi
    export FFMPEG_BIN="$resolved"
    log "tooling: uv=$UV_BIN UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT FFMPEG_BIN=$FFMPEG_BIN INFLATE_TORCH_SPEC=$INFLATE_TORCH_SPEC"
}

ensure_scorer_runtime_deps() {
    # contest_auth_eval.py calls upstream/evaluate.py with PYBIN, not the
    # inflate-side uv environment. Bare PyTorch images often lack scorer deps.
    # Round 2B B2 fix (2026-05-06, 85% confidence): redirect probe + ensurepip
    # logs to $LOG_DIR (durable artifact path) instead of /tmp (transient).
    # CLAUDE.md FORBIDDEN PATTERN: "/tmp paths in any persisted artifact" —
    # /tmp logs vanish on shell exit and are unrecoverable from CI/replays.
    if "$PYBIN" - <<'PY' >"$LOG_DIR/scorer_deps_probe_raw.log" 2>&1
import av, einops, safetensors, segmentation_models_pytorch, timm, tqdm
PY
    then
        log "scorer deps: already importable in $PYBIN"
        return
    fi
    log "BOOTSTRAP: scorer deps missing in $PYBIN; installing runtime scorer deps"
    if ! "$PYBIN" -c "import pip" 2>/dev/null; then
        "$PYBIN" -m ensurepip --upgrade >"$LOG_DIR/ensurepip_scorer.log" 2>&1 || true
    fi
    "$PYBIN" -m pip install -q --root-user-action=ignore \
        "timm>=0.9" \
        "einops>=0.7" \
        "segmentation-models-pytorch>=0.3" \
        "safetensors>=0.4" \
        "av>=10.0" \
        "tqdm>=4.0" \
        > "$LOG_DIR/scorer_deps_install.log" 2>&1 || {
            log "FATAL: scorer dependency install failed; see $LOG_DIR/scorer_deps_install.log"
            cat "$LOG_DIR/scorer_deps_install.log" >> "$LOG_DIR/run.log" 2>/dev/null || true
            exit 9
        }
    "$PYBIN" - <<'PY' > "$LOG_DIR/scorer_deps_probe.json"
import json
mods = {}
for name in ["torch", "timm", "einops", "safetensors", "segmentation_models_pytorch", "av", "tqdm"]:
    module = __import__(name)
    mods[name] = getattr(module, "__version__", "ok")
print(json.dumps(mods, sort_keys=True))
PY
    log "scorer deps: installed and probed"
}

require_declared_source_shas() {
    # Optional fail-closed source/runtime coherence guard.
    #
    # Format:
    #   REQUIRED_SOURCE_SHA256S='src/tac/foo.py=<sha256>
    #   submissions/robust_current/inflate.sh=<sha256>'
    #
    # This prevents exact-eval spend when an archive was built for a newer
    # decoder/runtime than the remote source tree currently contains.
    if [ -z "${REQUIRED_SOURCE_SHA256S:-}" ]; then
        return
    fi
    log "=== Stage 0b: required source SHA preflight ==="
    local line rel expected actual
    while IFS= read -r line; do
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [ -z "$line" ] && continue
        rel="${line%%=*}"
        expected="${line#*=}"
        if [ "$rel" = "$line" ] || [ -z "$rel" ] || [ -z "$expected" ]; then
            log "FATAL: malformed REQUIRED_SOURCE_SHA256S entry: $line"
            exit 11
        fi
        case "$rel" in
            /*|*..*|*\\*)
                log "FATAL: unsafe REQUIRED_SOURCE_SHA256S path: $rel"
                exit 11
                ;;
        esac
        if [ ! -f "$WORKSPACE/$rel" ]; then
            log "FATAL: required source file missing: $rel"
            exit 11
        fi
        actual="$(sha256sum "$WORKSPACE/$rel" | cut -d ' ' -f 1)"
        if [ "$actual" != "$expected" ]; then
            log "FATAL: required source SHA mismatch for $rel expected=$expected actual=$actual"
            exit 11
        fi
        log "source_sha_ok: $rel $actual"
    done <<< "$REQUIRED_SOURCE_SHA256S"
}

# Heartbeat (every 60s) — preflight Check 41
( while true; do echo "$(date -u +%FT%TZ) heartbeat pid=$$" >> "$HEARTBEAT"; sleep 60; done ) &
HB_PID=$!
trap "kill $HB_PID 2>/dev/null || true" EXIT

resolve_inflate_sh

cat > "$PROVENANCE" <<JSON
{
  "schema_version": 1,
  "started_at_utc": "$(date -u +%FT%TZ)",
  "git_hash": "$GIT_HASH",
  "gpu_name": "$GPU_NAME",
  "driver_version": "$DRIVER_VER",
  "lane_script": "scripts/remote_archive_only_eval.sh",
  "tag": "$ARCHIVE_LABEL",
  "archive_path": "$ARCHIVE_PATH",
  "inflate_sh": "$INFLATE_SH_ABS",
  "inflate_sh_sha256": "$INFLATE_SH_SHA",
  "predicted_band": [$PREDICTED_LOW, $PREDICTED_HIGH],
  "controlled_baseline": "$CONTROLLED_BASELINE",
  "no_training": true,
  "no_build": true,
  "strict_scorer_rule_compliant": true,
  "design_memo": "scripts/remote_archive_only_eval.sh"
}
JSON

require_uv_and_ffmpeg_contract
ensure_scorer_runtime_deps
require_declared_source_shas
DALI_BOOTSTRAP_DIR="${DALI_BOOTSTRAP_DIR:-$LOG_DIR}"
export DALI_BOOTSTRAP_DIR
cat > "$LOG_DIR/runtime_tooling.json" <<JSON
{
  "schema_version": 1,
  "recorded_at_utc": "$(date -u +%FT%TZ)",
  "pybin": "$PYBIN",
  "uv_bin": "$UV_BIN",
  "uv_project_environment": "$UV_PROJECT_ENVIRONMENT",
  "uv_cache_dir": "${UV_CACHE_DIR:-}",
  "uv_extra_index_url": "${UV_EXTRA_INDEX_URL:-}",
  "uv_index_strategy": "${UV_INDEX_STRATEGY:-}",
  "ffmpeg_bin": "$FFMPEG_BIN",
  "inflate_sh": "$INFLATE_SH_ABS",
  "inflate_sh_sha256": "$INFLATE_SH_SHA",
  "inflate_brotli_spec": "$INFLATE_BROTLI_SPEC",
  "inflate_av_spec": "$INFLATE_AV_SPEC",
  "inflate_torch_spec": "$INFLATE_TORCH_SPEC",
  "inflate_numpy_spec": "$INFLATE_NUMPY_SPEC",
  "dali_bootstrap_dir": "$DALI_BOOTSTRAP_DIR"
}
JSON

log "=== Stage 0: CUDA/NVDEC preflight ==="
"$PYBIN" - <<'PY'
import sys, torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

if [ "${SKIP_NVDEC_PROBE:-0}" != "1" ] && [ -f "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
        log "FATAL: NVDEC probe failed"; exit 2
    }
fi

[ -f "$ARCHIVE_PATH" ] || { log "FATAL: archive missing at $ARCHIVE_PATH"; exit 1; }
[ -f upstream/videos/0.mkv ] || { log "FATAL: missing upstream/videos/0.mkv"; exit 1; }
[ -f upstream/models/segnet.safetensors ] || { log "FATAL: missing segnet.safetensors"; exit 1; }
[ -f upstream/models/posenet.safetensors ] || { log "FATAL: missing posenet.safetensors"; exit 1; }

ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE_PATH" 2>/dev/null || stat -f '%z' "$ARCHIVE_PATH")
ARCHIVE_SHA=$(sha256sum "$ARCHIVE_PATH" 2>/dev/null | cut -d ' ' -f 1 || shasum -a 256 "$ARCHIVE_PATH" | cut -d ' ' -f 1)
log "archive: $ARCHIVE_PATH (bytes=$ARCHIVE_BYTES sha=$ARCHIVE_SHA label=$ARCHIVE_LABEL)"
CUSTODY_ARCHIVE="$LOG_DIR/archive.zip"
if [ "$ARCHIVE_PATH" != "$CUSTODY_ARCHIVE" ]; then
    cp "$ARCHIVE_PATH" "$CUSTODY_ARCHIVE"
fi
CUSTODY_BYTES=$(stat -c '%s' "$CUSTODY_ARCHIVE" 2>/dev/null || stat -f '%z' "$CUSTODY_ARCHIVE")
CUSTODY_SHA=$(sha256sum "$CUSTODY_ARCHIVE" 2>/dev/null | cut -d ' ' -f 1 || shasum -a 256 "$CUSTODY_ARCHIVE" | cut -d ' ' -f 1)
if [ "$CUSTODY_BYTES" != "$ARCHIVE_BYTES" ] || [ "$CUSTODY_SHA" != "$ARCHIVE_SHA" ]; then
    log "FATAL: archive custody copy drifted: source bytes=$ARCHIVE_BYTES sha=$ARCHIVE_SHA copy bytes=$CUSTODY_BYTES sha=$CUSTODY_SHA"
    exit 10
fi
cat > "$LOG_DIR/archive_custody.json" <<JSON
{
  "schema_version": 1,
  "recorded_at_utc": "$(date -u +%FT%TZ)",
  "archive_label": "$ARCHIVE_LABEL",
  "archive_path": "$ARCHIVE_PATH",
  "custody_archive_path": "$CUSTODY_ARCHIVE",
  "archive_size_bytes": $ARCHIVE_BYTES,
  "archive_sha256": "$ARCHIVE_SHA"
}
JSON

log "=== Stage 4: contest_auth_eval [contest-CUDA] ==="
rm -rf "$LOG_DIR/eval_work"
export INFLATE_REQUIRE_CUDA="${INFLATE_REQUIRE_CUDA:-1}"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE_PATH" \
    --inflate-sh "$INFLATE_SH_ABS" \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -30

PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    log "FATAL: contest_auth_eval failed rc=${PIPE_RC[0]}"
    exit "${PIPE_RC[0]}"
fi

if ! grep -q "RESULT_JSON" "$LOG_DIR/auth_eval.log"; then
    log "FATAL: auth_eval.log missing RESULT_JSON — silent eval crash"
    exit 4
fi
[ -f "$LOG_DIR/eval_work/contest_auth_eval.json" ] || {
    log "FATAL: missing contest_auth_eval.json"
    exit 4
}

cp "$LOG_DIR/eval_work/contest_auth_eval.json" "$LOG_DIR/contest_auth_eval.json"
cp "$LOG_DIR/eval_work/provenance.json" "$LOG_DIR/provenance.contest_auth_eval.json" 2>/dev/null || true
cp "$LOG_DIR/eval_work/report.txt" "$LOG_DIR/report.txt" 2>/dev/null || true
if [ "$KEEP_EVAL_WORK" != "1" ]; then
    rm -rf "$LOG_DIR/eval_work/inflated" "$LOG_DIR/eval_work/extracted" "$LOG_DIR/eval_work/archive.zip"
    if [ -n "${UV_PROJECT_ENVIRONMENT:-}" ]; then
        UV_ENV_REAL="$(realpath -m "$UV_PROJECT_ENVIRONMENT" 2>/dev/null || true)"
        LOG_DIR_REAL="$(realpath -m "$LOG_DIR" 2>/dev/null || true)"
        if [ -n "$UV_ENV_REAL" ] && [ -n "$LOG_DIR_REAL" ] && [[ "$UV_ENV_REAL" == "$LOG_DIR_REAL"/* ]]; then
            rm -rf "$UV_PROJECT_ENVIRONMENT"
        else
            log "SKIP_UV_PROJECT_ENV_CLEANUP path_not_under_log_dir=$UV_PROJECT_ENVIRONMENT"
        fi
    fi
fi
log "=== ARCHIVE_ONLY_EVAL_DONE [contest-CUDA] -- see $LOG_DIR/contest_auth_eval.json ==="
log "  archive_label: $ARCHIVE_LABEL"
log "  archive_sha:   $ARCHIVE_SHA"
log "  archive_bytes: $ARCHIVE_BYTES"
log "  predicted_band: [$PREDICTED_LOW, $PREDICTED_HIGH]"
log "  controlled_baseline: $CONTROLLED_BASELINE"
