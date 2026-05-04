#!/bin/bash
# Lane LS-C067: PR67-style R(D)-joint coordinate-descent pose refinement on C-067 anchor.
#
# WHAT: greedy coordinate descent on QP1 col0 (length 600) for the C-067 anchor
# archive. For each frame, search radii [1, 2, 3, 5, 8] around current pose-0
# value. For each candidate, forward-pass through (JointFrameGenerator on C-067's
# QZS3 renderer, then PoseNet) and score the joint objective:
#
#   obj = sqrt(10 * pose_dist) + 25 * archive_size / 37545489
#
# Accept the candidate that minimizes joint score (PoseNet distortion + QP1-encoded
# byte cost). 2 passes per radius, then we keep the best archive.
#
# Reference implementation: reports/raw/leaderboard_intel_20260501/pr67_line_search.py
# Local port: experiments/line_search_pose_refinement.py
# Memory: reference_pr67_line_search_R_D_joint_coordinate_descent_20260501.md
#
# C-067 anchor:
#   archive sha256: 226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a
#   archive bytes: 276,214
#   contest-CUDA T4 score: 0.31561703
#   pose_q (brotli) bytes: 677, decoded QP1 bytes: 1140 (length=600)
#
# PREDICTED BAND: [0.310, 0.318] [contest-CUDA].
#   Floor 0.310: -0.005 from full pr67 line-search uplift on Wave-1 anchors
#                (per reference_pr67_line_search_R_D_joint_coordinate_descent_20260501.md)
#   Ceiling 0.318: tied with PFP16-frontier 0.32 region if no improvement is
#                  found (some candidates may net-zero or slightly worsen).
#   Council Quantizr: cheap +0.001 to +0.005 stack-on-top with bounded downside.
#
# CLAUDE.md compliance:
#   * set -euo pipefail (zip_dep_bootstrap_trap memory)
#   * Python `zipfile.ZipFile` (NOT shell `zip`)
#   * --device cuda everywhere (no MPS/CPU fallback)
#   * Stage 0 NVDEC probe (check 33 + feedback_vastai_nvdec_host_variation)
#   * Self-bootstrap via bootstrap_runtime_deps() from
#     scripts/remote_archive_only_eval.sh (uv + ffmpeg + ._ resource forks)
#   * NEVER-INVENT-CLI-FLAGS: every flag verified by argparse-grep against
#       experiments/line_search_pose_refinement.py:
#         --archive-path --metadata-path --output-path --output-metadata
#         --posenet-path --gt-mkv --device --batch-size --candidate-chunk
#         --radii --passes
#       experiments/contest_auth_eval.py:
#         --archive --inflate-sh --upstream-dir --device --keep-work-dir
#         --work-dir
#         (NO --output-json: contest_auth_eval.py:919 auto-writes
#          contest_auth_eval.json to work_dir; we read from
#          $LOG_DIR/eval_work/contest_auth_eval.json post-dispatch.)
#   * predicted_band metadata + [contest-CUDA] tag in completion line
#   * provenance.json + heartbeat.log (canonical_remote_bootstraps memory)
#   * Container Python /opt/conda/bin/python (NOT venv)

set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
LS_RUN_ID="${LS_RUN_ID:-lane_line_search_c067_results}"
LS_RADII="${LS_RADII:-1,2,3,5,8}"
LS_PASSES="${LS_PASSES:-2}"
LS_BATCH_SIZE="${LS_BATCH_SIZE:-16}"
LS_CANDIDATE_CHUNK="${LS_CANDIDATE_CHUNK:-32}"
LS_MAX_CANDIDATE_ITEMS="${LS_MAX_CANDIDATE_ITEMS:-2048}"
LS_BASIS_WINDOW_RADIUS="${LS_BASIS_WINDOW_RADIUS:-0}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE:${PYTHONPATH:-}"
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="${LS_OUTPUT_DIR:-$WORKSPACE/$LS_RUN_ID}"
mkdir -p "$LOG_DIR"

log() { echo "[lane-ls-c067] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat.
PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)

# Self-bootstrap (uv + ffmpeg + AppleDouble) per
# feedback_remote_archive_only_eval_self_bootstraps_all_deps_20260501.
log "=== Stage 0a: bootstrap_runtime_deps (uv + ffmpeg + ._ purge) ==="
if [ -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    # shellcheck source=/dev/null
    source <(grep -A 30 '^bootstrap_runtime_deps()' "$WORKSPACE/scripts/remote_archive_only_eval.sh")
    bootstrap_runtime_deps
else
    log "WARN: scripts/remote_archive_only_eval.sh missing; skipping bootstrap helper"
fi

"$PYBIN" -c "
import json, time, torch
prov = {
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'lane_script': 'scripts/remote_lane_line_search_c067.sh',
    'lane_name': 'lane_line_search_c067',
    'predicted_band': [0.310, 0.318],
    'score_tag': '[contest-CUDA]',
    'baseline_score': 0.31561703,
    'baseline_archive_sha256': '226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a',
    'baseline_archive_bytes': 276214,
    'baseline_anchor_lane': 'C-067 (lane_line_search_pose_refinement codex 2026-05-02 + frontier_public_pr67_fixedslice)',
    'run_id': '$LS_RUN_ID',
    'output_dir': '$LOG_DIR',
    'tool': 'experiments/line_search_pose_refinement.py',
    'reference_tool': 'reports/raw/leaderboard_intel_20260501/pr67_line_search.py',
    'radii': '$LS_RADII',
    'delta_sets': '${LS_DELTA_SETS:-}',
    'gradient_delta_sets': '${LS_GRADIENT_DELTA_SETS:-}',
    'basis_delta_sets': '${LS_BASIS_DELTA_SETS:-}',
    'basis_modes': '${LS_BASIS_MODES:-}',
    'basis_pair_indices': '${LS_BASIS_PAIR_INDICES:-}',
    'basis_window_radius': '${LS_BASIS_WINDOW_RADIUS:-0}',
    'passes': int('$LS_PASSES'),
    'estimated_cost_usd': 1.50,
    'cost_cap_usd': 1.50,
    'note': 'PR67-style R(D)-joint coordinate descent on C-067 anchor pose stream. '
            'Optimizes col0 of QP1 codec under joint sqrt(10*pose) + 25*size/37545489 objective. '
            'Per pr67_line_search.py (194 LOC reference) ~30-60min on RTX 4090.',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=LS-C067 gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0b: NVDEC probe BEFORE any GPU spend.
log "=== Stage 0b: NVDEC probe (--ensure-dali) ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
    log "FATAL: NVDEC probe failed (exit $?). Refusing to spend GPU on a"
    log "       host that cannot run upstream/evaluate.py at the end."
    log "       Destroy this Vast.ai instance and pick a different host."
    exit 2
}

# Stage 1: stage C-067 source archive + metadata (shipped under
# experiments/results/lane_line_search_c067_20260502/).
log "=== Stage 1: stage C-067 source archive + metadata ==="
SOURCE_ARCHIVE="${LS_SOURCE_ARCHIVE:-$WORKSPACE/experiments/results/lane_line_search_c067_20260502/source_archive.zip}"
SOURCE_METADATA="${LS_SOURCE_METADATA:-$WORKSPACE/experiments/results/lane_line_search_c067_20260502/source_metadata.json}"
GT_VIDEO="${LS_GT_VIDEO:-$WORKSPACE/upstream/videos/0.mkv}"
POSENET_WEIGHTS="${LS_POSENET_WEIGHTS:-$WORKSPACE/upstream/models/posenet.safetensors}"

for f in "$SOURCE_ARCHIVE" "$SOURCE_METADATA" "$GT_VIDEO" "$POSENET_WEIGHTS"; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

ANCHOR_SHA="226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a"
ACTUAL_SHA=$(sha256sum "$SOURCE_ARCHIVE" 2>/dev/null | cut -d ' ' -f 1)
if [ "$ACTUAL_SHA" != "$ANCHOR_SHA" ]; then
    log "FATAL: anchor archive SHA mismatch"
    log "  expected: $ANCHOR_SHA"
    log "  actual:   $ACTUAL_SHA"
    exit 2
fi
log "  source_archive.zip SHA verified: $ACTUAL_SHA"

# Re-write source metadata so its archive_path field points to the deployed
# location of source_archive.zip on the remote (the file moved between local
# and remote). build_refined_metadata uses 'source_meta' as a TEMPLATE only;
# the OUTPUT metadata's archive_path/sha/bytes are recomputed from the actual
# refined output file. assert_metadata_matches_archive only checks the OUTPUT.
"$PYBIN" -c "
import json
src = '$SOURCE_METADATA'
with open(src) as f:
    meta = json.load(f)
meta['archive_path'] = '$SOURCE_ARCHIVE'
with open(src, 'w') as f:
    json.dump(meta, f, indent=2)
print('source metadata archive_path rewritten to:', meta['archive_path'])
"

# Stage 2: line-search refinement.
# CLI flags verified against experiments/line_search_pose_refinement.py:1141-1278.
SEARCH_ARGS=()
if [ -n "${LS_BASIS_DELTA_SETS:-}" ]; then
    SEARCH_ARGS+=(--basis-delta-sets "$LS_BASIS_DELTA_SETS")
    [ -n "${LS_BASIS_MODES:-}" ] && SEARCH_ARGS+=(--basis-modes "$LS_BASIS_MODES")
    [ -n "${LS_BASIS_PAIR_INDICES:-}" ] && SEARCH_ARGS+=(--basis-pair-indices "$LS_BASIS_PAIR_INDICES")
    SEARCH_ARGS+=(--basis-window-radius "$LS_BASIS_WINDOW_RADIUS")
elif [ -n "${LS_GRADIENT_DELTA_SETS:-}" ]; then
    SEARCH_ARGS+=(--gradient-delta-sets "$LS_GRADIENT_DELTA_SETS")
    [ -n "${LS_GRADIENT_BACKTRACK_DELTAS:-}" ] && SEARCH_ARGS+=(--gradient-backtrack-deltas "$LS_GRADIENT_BACKTRACK_DELTAS")
elif [ -n "${LS_DELTA_SETS:-}" ]; then
    SEARCH_ARGS+=(--delta-sets "$LS_DELTA_SETS")
else
    SEARCH_ARGS+=(--radii "$LS_RADII")
fi
log "=== Stage 2: PR67-style line-search refinement (${SEARCH_ARGS[*]} passes=$LS_PASSES) ==="
log "  Estimated 102K forward passes on RTX 4090 (~30-60 min)"
REFINED_DIR="$LOG_DIR/refined"
mkdir -p "$REFINED_DIR"
REFINED_ARCHIVE="$REFINED_DIR/refined_archive.zip"
REFINED_METADATA="$REFINED_DIR/refined_metadata.json"

"$PYBIN" -u experiments/line_search_pose_refinement.py \
    --archive-path "$SOURCE_ARCHIVE" \
    --metadata-path "$SOURCE_METADATA" \
    --output-path "$REFINED_ARCHIVE" \
    --output-metadata "$REFINED_METADATA" \
    --posenet-path "$POSENET_WEIGHTS" \
    --gt-mkv "$GT_VIDEO" \
    --device cuda:0 \
    --batch-size "$LS_BATCH_SIZE" \
    --candidate-chunk "$LS_CANDIDATE_CHUNK" \
    --max-candidate-items "$LS_MAX_CANDIDATE_ITEMS" \
    "${SEARCH_ARGS[@]}" \
    --passes "$LS_PASSES" 2>&1 | tee "$LOG_DIR/line_search.log" | tail -60
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: line_search_pose_refinement.py exited rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

if [ ! -f "$REFINED_ARCHIVE" ]; then
    log "FATAL: refined_archive.zip was NOT produced — line search must have failed."
    exit 2
fi
REFINED_BYTES=$(stat -c '%s' "$REFINED_ARCHIVE" 2>/dev/null || stat -f '%z' "$REFINED_ARCHIVE")
REFINED_SHA=$(sha256sum "$REFINED_ARCHIVE" 2>/dev/null | cut -d ' ' -f 1)
cp "$REFINED_ARCHIVE" "$LOG_DIR/archive.zip"
cp "$REFINED_METADATA" "$LOG_DIR/metadata.json"
log "  refined_archive.zip = ${REFINED_BYTES} bytes, sha=${REFINED_SHA}"

# Stage 3: contest_auth_eval on refined archive.
log "=== Stage 3: contest_auth_eval on refined archive [contest-CUDA] ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$REFINED_ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -30
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: contest_auth_eval exited rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

# contest_auth_eval.py:919 auto-writes contest_auth_eval.json to work_dir.
# Read from the auto-written path instead of the (invented) --output-json flag.
EVAL_JSON="$LOG_DIR/eval_work/contest_auth_eval.json"
if [ ! -f "$EVAL_JSON" ]; then
    log "FATAL: $EVAL_JSON was NOT produced — invalid measurement"
    exit 2
fi
# Mirror to a stable convenience path for downstream tooling.
cp "$EVAL_JSON" "$LOG_DIR/contest_auth_eval.json"
if [ -f "$LOG_DIR/eval_work/report.txt" ]; then
    cp "$LOG_DIR/eval_work/report.txt" "$LOG_DIR/report.txt"
fi
if [ -f "$LOG_DIR/eval_work/eval_provenance.json" ]; then
    cp "$LOG_DIR/eval_work/eval_provenance.json" "$LOG_DIR/eval_provenance.json"
fi
if [ "${LS_CLEAN_EVAL_WORK:-1}" = "1" ]; then
    rm -rf "$LOG_DIR/eval_work/inflated" \
           "$LOG_DIR/eval_work/extracted" \
           "$LOG_DIR/eval_work/archive.zip"
fi

REFINED_SCORE=$("$PYBIN" -c "
import json
with open('$EVAL_JSON') as f:
    d = json.load(f)
print(d.get('score_recomputed_from_components', d.get('final_score', 'unknown')))
")
log "=== LANE_LS_C067_DONE [contest-CUDA] — refined score: $REFINED_SCORE (baseline=0.31561703) ==="
log "=== predicted_band=[0.310, 0.318], baseline=0.31561703, see $LOG_DIR/provenance.json ==="
log "=== refined archive: $REFINED_ARCHIVE ==="
log "=== refined sha: $REFINED_SHA bytes: $REFINED_BYTES ==="
