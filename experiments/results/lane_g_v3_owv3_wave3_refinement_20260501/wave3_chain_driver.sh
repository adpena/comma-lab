#!/bin/bash
# HISTORICAL_RECIPE_ONLY — frozen wave3 chain driver from 2026-05-01.
# This is a HISTORICAL ARTIFACT preserved for forensic reproduction of the
# wave3_chain_eval sweep that landed in lane_g_v3_owv3_wave3_refinement.
# DO NOT REPLAY VERBATIM on a live Vast.ai instance — the /tmp scratch
# paths and chain-id assumptions are specific to the 2026-05-01 setup.
# For new chain runs, use scripts/remote_archive_only_eval.sh directly.
#
# Wave-3 OWV3 chain-eval driver — runs on Vast.ai instance after SCP'd archives land.
# Orchestrates 6 sequential remote_archive_only_eval.sh invocations under the
# wave3_chain_eval label. Outputs land at /workspace/pact/owv3_wave3_<id>_results/
#
# Pre-conditions:
#   - /workspace/pact/wave3_archives/<candidate>.zip already SCP'd
#   - /workspace/pact (repo) already extracted from anchor tarball
#   - upstream/{videos,models} present in /workspace/pact/upstream
#
# Usage:
#   bash /workspace/pact/experiments/results/lane_g_v3_owv3_wave3_refinement_20260501/wave3_chain_driver.sh
#
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
ARCHIVES_DIR="${ARCHIVES_DIR:-/workspace/pact/wave3_archives}"
cd "$WORKSPACE"

ffmpeg_has_scale_contract() {
    local ffmpeg_bin="$1"
    [ -n "$ffmpeg_bin" ] || return 1
    command -v "$ffmpeg_bin" >/dev/null 2>&1 || [ -x "$ffmpeg_bin" ] || return 1
    local resolved
    resolved="$(command -v "$ffmpeg_bin" 2>/dev/null || true)"
    [ -n "$resolved" ] || resolved="$ffmpeg_bin"
    local scale_help
    scale_help="$("$resolved" -hide_banner -h filter=scale 2>&1 || true)"
    for opt in in_range out_range in_color_matrix in_primaries in_transfer; do
        printf '%s\n' "$scale_help" | grep -q "$opt" || return 1
    done
    return 0
}

ensure_wave3_runtime_tools() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "[wave3-chain] uv missing; invoking scripts/ensure_remote_uv.sh"
        [ -f "$WORKSPACE/scripts/ensure_remote_uv.sh" ] || { echo "[wave3-chain] FATAL: ensure_remote_uv.sh missing"; exit 7; }
        UV_BIN="$(bash "$WORKSPACE/scripts/ensure_remote_uv.sh" --symlink-system)"
        export UV_BIN
        export PATH="$(dirname "$UV_BIN"):$HOME/.local/bin:$PATH"
    fi
    command -v uv >/dev/null 2>&1 || { echo "[wave3-chain] FATAL: uv missing after install"; exit 7; }

    local ffmpeg_root="/workspace/ffmpeg-btbn"
    if ! ffmpeg_has_scale_contract "${FFMPEG_BIN:-}" && ffmpeg_has_scale_contract "$ffmpeg_root/bin/ffmpeg"; then
        export FFMPEG_BIN="$ffmpeg_root/bin/ffmpeg"
    fi

    if ! ffmpeg_has_scale_contract "${FFMPEG_BIN:-}"; then
        local ffmpeg_archive="/tmp/ffmpeg-btbn-master.tar.xz"
        echo "[wave3-chain] installing BtbN ffmpeg master for scale color-contract"
        rm -rf "$ffmpeg_root" /tmp/ffmpeg-master-latest-linux64-gpl
        curl -L --fail \
            -o "$ffmpeg_archive" \
            https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz
        tar -xJf "$ffmpeg_archive" -C /tmp
        mv /tmp/ffmpeg-master-latest-linux64-gpl "$ffmpeg_root"
        rm -f "$ffmpeg_archive"
        export FFMPEG_BIN="$ffmpeg_root/bin/ffmpeg"
    fi
    ffmpeg_has_scale_contract "${FFMPEG_BIN:-}" || { echo "[wave3-chain] FATAL: usable BtbN ffmpeg not found"; exit 8; }
    export FFMPEG_BIN
    echo "[wave3-chain] uv=$(command -v uv)"
    echo "[wave3-chain] FFMPEG_BIN=$FFMPEG_BIN"
    "$FFMPEG_BIN" -version | head -1
}

CANDIDATES=(
    "owv3_0043_bbr0p695_protect0p002_aggr1em05:cons:624419"
    "owv3_0032_bbr0p7_protect0p002_aggr1em05:cons:624996"
    "owv3_0076_bbr0p68_protect0p002_aggr1em05:mid:621914"
    "owv3_0065_bbr0p685_protect0p002_aggr1em05:mid:622407"
    "owv3_0120_bbr0p66_protect0p002_aggr1em05:aggr:617410"
    "owv3_0119_bbr0p66_protect0p0018_aggr1em05:aggr:618443"
)

R7_BASELINE_SCORE=1.0134396099014253
R7_BASELINE_BYTES=631473

ensure_wave3_runtime_tools

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$WORKSPACE/owv3_wave3_uv_project_env}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE/owv3_wave3_uv_cache}"

echo "[wave3-chain] starting at $(date -u +%FT%TZ); $((${#CANDIDATES[@]})) candidates"
echo "[wave3-chain] UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"
echo "[wave3-chain] UV_CACHE_DIR=$UV_CACHE_DIR"
for entry in "${CANDIDATES[@]}"; do
    cid="${entry%%:*}"
    rest="${entry#*:}"
    label="${rest%%:*}"
    bytes="${rest#*:}"
    delta=$((bytes - R7_BASELINE_BYTES))
    archive="$ARCHIVES_DIR/${cid}.zip"
    if [ ! -f "$archive" ]; then
        echo "[wave3-chain] FATAL: missing $archive — skipping"
        continue
    fi
    echo ""
    echo "[wave3-chain] === $cid (label=$label, bytes=$bytes, Δr7=$delta) ==="
    ARCHIVE_PATH="$archive" \
    ARCHIVE_LABEL="owv3_wave3_${cid}" \
    PREDICTED_LOW="0.998" \
    PREDICTED_HIGH="1.013" \
    CONTROLLED_BASELINE="owv3_0001_r7 (1.0134 [contest-CUDA RTX 4090])" \
    LOG_DIR="$WORKSPACE/owv3_wave3_${cid}_results" \
        bash "$WORKSPACE/scripts/remote_archive_only_eval.sh" 2>&1 | tee -a "/tmp/wave3_chain_${cid}.log" || {
        echo "[wave3-chain] ERROR on $cid; continuing"
    }
    if [ -f "$WORKSPACE/owv3_wave3_${cid}_results/contest_auth_eval.json" ]; then
        score=$(python3 -c "import json; d=json.load(open('$WORKSPACE/owv3_wave3_${cid}_results/contest_auth_eval.json')); print(f\"{d.get('score_recomputed_from_components', 0):.4f}\")")
        echo "[wave3-chain] $cid SCORE=$score (vs r7 baseline 1.0134)"
    fi
    # 2026-05-01 (Bug Class #7): per-candidate inflated-frame cleanup so a
    # 6+ candidate chain doesn't pile 6×3.6GB onto the 30/60GB disk and
    # crash mid-chain. Reference:
    # feedback_loop_session_permanent_bug_class_extinction_20260501.md.
    rm -rf "$WORKSPACE/owv3_wave3_${cid}_results/eval_work/inflated" \
           "$WORKSPACE/owv3_wave3_${cid}_results/eval_work/extracted" \
           "$WORKSPACE/owv3_wave3_${cid}_results/eval_work/archive.zip" 2>/dev/null || true
done

echo ""
echo "[wave3-chain] === ALL CHAIN DONE at $(date -u +%FT%TZ) ==="
echo ""
echo "[wave3-chain] Score summary:"
for entry in "${CANDIDATES[@]}"; do
    cid="${entry%%:*}"
    j="$WORKSPACE/owv3_wave3_${cid}_results/contest_auth_eval.json"
    if [ -f "$j" ]; then
        python3 -c "
import json
d = json.load(open('$j'))
print(f'  {\"$cid\":55s} score={d.get(\"score_recomputed_from_components\", 0):.5f} pose={d.get(\"avg_posenet_dist\", 0):.5f} seg={d.get(\"avg_segnet_dist\", 0):.5f} bytes={d.get(\"archive_size_bytes\", 0)}'
"
    else
        printf '  %-55s MISSING JSON\n' "$cid"
    fi
done
