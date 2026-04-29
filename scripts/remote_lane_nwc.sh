#!/bin/bash
# Lane J-NWC: Neural Weight Compression — full end-to-end producer/consumer
# pipeline (codec train → renderer compress → archive build → CUDA auth eval).
#
# Reference: arXiv 2510.11234 ("Neural Weight Compression for Language Models",
# late 2025) — VQ-VAE-style codec encodes every floating-point parameter
# tensor block to (codebook_index + per-block float16 scale). Codec weights
# are bundled INSIDE the NWC1 binary so the inflate-side loader is fully
# self-contained (no external codec asset required at inflate time).
#
# Pipeline:
#   Stage 0: NVDEC probe (memory: feedback_vastai_nvdec_host_variation)
#   Stage 1: anchor + corpus verification (renderer .pt corpus + Lane A
#            renderer.bin anchor)
#   Stage 2: train WeightCodec on the corpus (CPU- or CUDA-trained;
#            ~16K codec params, 2000 steps converges in <2 min on 4090)
#   Stage 3: compress the Lane A anchor renderer through the codec;
#            build full archive (renderer_nwc.bin + masks.mkv + poses.pt)
#   Stage 4: CUDA contest_auth_eval on the EXACT submission archive bytes
#
# Predicted band [0.95, 1.30] [prediction] — the codec trades off some
# weight-precision for ~50% smaller renderer.bin; the SegNet/PoseNet
# distortion is expected to rise modestly relative to Lane A baseline 1.15
# but the rate term should shrink. We do NOT claim a contest-CUDA score
# until Stage 4 produces a RESULT_JSON. The band assumes the codec was
# pretrained for ≥2000 steps on a corpus of ≥10 small-renderer .pt files.
#
# Cost: 4090 @ $0.25/hr × ~1.5h = ~$0.40 (no full retrain — codec is small,
# anchor renderer already trained).
#
# Per CLAUDE.md NEVER-INVENT-CLI-FLAGS: every flag passed to
# train_neural_weight_codec.py / build_baseline_archive.py / contest_auth_eval.py
# was verified by argparse-grep on the target sources (the in-script
# dead-flag scanner re-validates this at launch time).
#
# Per CLAUDE.md MPS-auth-eval-is-NOISE: AUTH_EVAL_DEVICE defaults to "cuda"
# below; the script aborts loudly if cuda is unavailable.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_j_nwc_results}"
mkdir -p "$LOG_DIR"
TAG="lane_j_nwc"
LANE="J-NWC"

log() { echo "[lane-j-nwc] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat (CLAUDE.md canonical pipeline standard +
# memory feedback_canonical_remote_bootstraps + Check L STRICT
# check_remote_scripts_write_provenance).
PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="/tmp/heartbeat_${TAG}.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)
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
    'lane_script': 'scripts/remote_lane_nwc.sh',
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
    'lane': '$LANE',
    'predicted_band': [0.95, 1.30],
    'anchor_score_baseline': 1.15,
    'anchor_lane': 'lane_a (1.15 contest-CUDA)',
    'lane_j_nwc_premise': 'VQ-VAE codec encodes renderer state-dict tensors; codec bundled in NWC1 binary; expected ~50%% byte savings vs FP4',
    'codec_config': {
        'block_size': 16,
        'codebook_size': 64,
        'latent_dim': 16,
        'hidden': 64,
    },
    'cost_estimate_usd': 0.40,
    'wall_clock_estimate_hours': 1.5,
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=$LANE gpu=$GPU stage=$( [ -f "$LOG_DIR/.stage" ] && cat "$LOG_DIR/.stage" || echo unknown)" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend. Reference:
# feedback_vastai_nvdec_host_variation. Check 33 STRICT
# check_remote_scripts_probe_nvdec_early enforces probe at Stage 0.
echo "stage_0_nvdec_probe" > "$LOG_DIR/.stage"
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — destroy this instance and pick a different host."
    exit 2
}

# Stage 1: anchor verification + corpus discovery
echo "stage_1_anchor" > "$LOG_DIR/.stage"
log "=== Stage 1: anchor + corpus verification ==="

ANCHOR_RENDERER="${ANCHOR_RENDERER:-submissions/baseline_dilated_h64_0_90/renderer.bin}"
ANCHOR_POSES="${ANCHOR_POSES:-submissions/baseline_dilated_h64_0_90/optimized_poses.pt}"
CORPUS_DIR="${CORPUS_DIR:-experiments/results}"

for f in upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors \
         "$ANCHOR_RENDERER" \
         "$ANCHOR_POSES"; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

if [ ! -d "$CORPUS_DIR" ]; then
    echo "FATAL: corpus directory $CORPUS_DIR does not exist" >&2
    exit 1
fi

# Pre-flight: dead-flag-wiring guard — every CLI flag in the train_neural_weight_codec
# / build_baseline_archive / contest_auth_eval invocations below must exist
# in the target's argparse. CLAUDE.md non-negotiable: NEVER invent CLI flags.
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
script = open('scripts/remote_lane_nwc.sh').read()
targets = {
    'train_neural_weight_codec.py': 'experiments/train_neural_weight_codec.py',
    'build_baseline_archive.py':    'experiments/build_baseline_archive.py',
    'contest_auth_eval.py':         'experiments/contest_auth_eval.py',
}
for short, path in targets.items():
    src = open(path).read()
    real = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', src))
    # Per-target invocation block: extract a window around \"$short\".
    blocks = re.findall(r'(?:\\\$PYBIN\\b[^\\n]*' + re.escape(short) + r'(?:[^\\n]*\\\\\\n[^\\n]*)*)', script)
    for blk in blocks:
        used = set(re.findall(r'\\B--([a-z][a-z0-9-]+)', blk))
        invented = used - real
        if invented:
            print(f'INVENTED FLAGS in {short}: {sorted(invented)} not in argparse',
                  file=sys.stderr); sys.exit(3)
print('OK: lane_j_nwc dead-flag scan passed')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Stage 2: train the WeightCodec on the corpus.
# uv-managed venv (per CLAUDE.md tooling rule). Codec is small (~16K params)
# so CPU is fine; CUDA is faster but byte-deterministic either way.
echo "stage_2_train_codec" > "$LOG_DIR/.stage"
log "=== Stage 2: train WeightCodec on corpus $CORPUS_DIR ==="

if ! command -v uv >/dev/null 2>&1; then
    log "  uv not on PATH; installing via official curl bootstrap"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "$WORKSPACE/.venv" ]; then
    log "  creating uv venv at $WORKSPACE/.venv"
    uv venv "$WORKSPACE/.venv"
fi
# Activate uv-managed venv
# shellcheck disable=SC1091
source "$WORKSPACE/.venv/bin/activate"
uv pip install --quiet torch numpy pyav 2>&1 | tail -5 | tee -a "$LOG_DIR/run.log" || true

CODEC_PATH="$LOG_DIR/codec.pt"
"$PYBIN" -u experiments/train_neural_weight_codec.py \
    --corpus-dir "$CORPUS_DIR" \
    --output "$CODEC_PATH" \
    --num-steps 2000 \
    --batch-size 256 \
    --lr 1e-3 \
    --device "${CODEC_DEVICE:-cuda}" \
    --block-size 16 \
    --codebook-size 64 \
    --latent-dim 16 \
    --hidden 64 \
    --max-corpus-files 200 \
    --max-blocks-per-ckpt 50000 \
    --seed 1234 \
    --log-interval 200 \
    2>&1 | tee "$LOG_DIR/codec_train.log" | tail -20

[ -f "$CODEC_PATH" ] || { echo "FATAL: codec training did not produce $CODEC_PATH" >&2; exit 2; }
log "  codec checkpoint: $CODEC_PATH ($(stat -c '%s' "$CODEC_PATH" 2>/dev/null || stat -f '%z' "$CODEC_PATH") bytes)"

# Stage 3: compress the Lane A anchor renderer through the codec; build archive.
echo "stage_3_compress_and_archive" > "$LOG_DIR/.stage"
log "=== Stage 3: compress anchor renderer + build full archive ==="

mkdir -p "$LOG_DIR/iter_0"
RENDERER_NWC="$LOG_DIR/iter_0/renderer.bin"
"$PYBIN" -c "
import sys, torch
sys.path.insert(0, 'src')
from tac.renderer_export import (
    export_neural_compressed_checkpoint,
    load_asymmetric_checkpoint,
)

anchor_path = '$ANCHOR_RENDERER'
codec_path = '$CODEC_PATH'
out_bin = '$RENDERER_NWC'

# Anchor is FP4A; we round-trip it back to a float-state model and re-encode
# with NWC1. (load_asymmetric_checkpoint dispatches FP4A internally.)
raw = open(anchor_path, 'rb').read()
model = load_asymmetric_checkpoint(raw, device='cpu')
model.eval()
nbytes = export_neural_compressed_checkpoint(
    model, codec_path=codec_path, output_path=out_bin,
)
orig_bytes = len(raw)
print(f'  Anchor (FP4A): {orig_bytes:,} bytes')
print(f'  NWC1:          {nbytes:,} bytes')
print(f'  Delta:         {orig_bytes - nbytes:+,} bytes ({100.0*(orig_bytes - nbytes)/orig_bytes:+.1f}%) [empirical:$LOG_DIR/codec_train.log]')
" 2>&1 | tee -a "$LOG_DIR/run.log"

[ -f "$RENDERER_NWC" ] || { echo "FATAL: NWC1 export failed" >&2; exit 2; }

# Build the archive: NWC renderer.bin + masks.mkv + optimized_poses.pt.
# Use the canonical build_baseline_archive.py flow to extract masks for
# the same anchor, then copy the NWC renderer.bin into iter_0 and zip.
log "=== Stage 3b: build masks.mkv via build_baseline_archive (CRF 50) ==="
"$PYBIN" experiments/build_baseline_archive.py \
    --renderer "$ANCHOR_RENDERER" \
    --poses "$ANCHOR_POSES" \
    --crf 50 \
    --device cuda \
    --output "$LOG_DIR/anchor_archive_seed.zip" 2>&1 | tee "$LOG_DIR/build.log" | tail -5
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: build_baseline_archive exited rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

mkdir -p "$LOG_DIR/extracted"
cd "$LOG_DIR/extracted" && unzip -o "$LOG_DIR/anchor_archive_seed.zip" 2>&1 | tail -5
cd "$WORKSPACE"
cp "$LOG_DIR/extracted/masks.mkv" "$LOG_DIR/iter_0/masks.mkv"
cp "$LOG_DIR/extracted/optimized_poses.pt" "$LOG_DIR/iter_0/optimized_poses.pt"

ARCHIVE="$LOG_DIR/archive_lane_j_nwc.zip"
# Python zipfile (NOT shell `zip`) — PyTorch container has no `zip` binary
# (memory: feedback_zip_dep_bootstrap_trap). Deterministic timestamps
# (Check #5 R5-r6: check_archive_builders_use_deterministic_zip).
"$PYBIN" -c "
import zipfile, os, time
src = '$LOG_DIR/iter_0'
dst = '$ARCHIVE'
files = ['renderer.bin', 'masks.mkv', 'optimized_poses.pt']
fixed_ts = (1980, 1, 1, 0, 0, 0)
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in files:
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing {p}'
        info = zipfile.ZipInfo(filename=n, date_time=fixed_ts)
        info.compress_type = zipfile.ZIP_DEFLATED
        with open(p, 'rb') as fh:
            z.writestr(info, fh.read())
print(f'archive {dst}: {os.path.getsize(dst)} bytes ({len(files)} files)')
"

# Stage 4: CUDA contest_auth_eval on the EXACT submission archive bytes.
# CLAUDE.md non-negotiable: AUTH EVAL EVERYWHERE. MPS auth eval is NOISE.
echo "stage_4_auth_eval" > "$LOG_DIR/.stage"
log "=== Stage 4: contest_auth_eval on Lane J-NWC archive ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device "${AUTH_EVAL_DEVICE:-cuda}" \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2
    exit "${PIPE_RC[0]}"
fi

# RESULT_JSON guard (LANE-B silent-crash prevention).
if ! grep -q "RESULT_JSON" "$LOG_DIR/auth_eval.log"; then
    log "FATAL: auth_eval.log missing RESULT_JSON — silent eval crash. Investigate before reporting score."
    exit 4
fi

echo "stage_done" > "$LOG_DIR/.stage"
log "=== LANE_J_NWC_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
log "  predicted band: [0.95, 1.30] standalone (vs Lane A 1.15 [contest-CUDA])"
log "  anchor baseline: 1.15 [contest-CUDA] (Lane A frontier)"
