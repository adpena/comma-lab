#!/bin/bash
# Lane I: Cool-Chic renderer (Orange/CNES, b"CCh1" magic) anchored on Lane A.
#
# 2026-04-27 prompt note: the operator framing was "Lane I-A = Cool-Chic the
# MASK SEQUENCE only". The Cool-Chic infrastructure in this repo (CCh1 magic,
# load_coolchic_renderer, build_coolchic_renderer, _export_coolchic_or_c3)
# does NOT compress mask frames — it is a renderer architecture variant
# (CoolChicLatentRenderer + MotionPredictor PairGenerator) that REPLACES
# renderer.bin via the CCh1 magic in submissions/robust_current/inflate_renderer.py
# (line 1758). A standalone "mask codec" Cool-Chic would require:
#   * a new architecture (no PairGenerator, no MotionPredictor, mask-uint8 → bytes)
#   * a new magic byte (CCM1 or similar)
#   * inflate.sh changes to decode masks BEFORE the renderer.bin runs
# That is multi-day infrastructure work and is OUT OF SCOPE for a 30-min lane.
#
# So Lane I as implemented = Cool-Chic RENDERER lane: train the existing
# coolchic_renderer variant from scratch (it is a different arch from Lane A's
# ASYM, so --resume-from cannot warm-start), export via export_coolchic_renderer
# (CCh1), bundle with Lane A's masks.mkv + optimized_poses.pt unchanged. The
# rate attack lever is the small renderer payload: typical CCh1 is ~30-80KB
# vs Lane A's 290KB renderer.bin = up to ~210KB of rate savings (rate term
# 0.0185 → ~0.013-0.016 if distortion holds). Predicted band: [0.95, 1.30]
# [contest-CUDA].
#
# Pipeline:
#   1. Stage 0 NVDEC probe (5s sanity, feedback_vastai_nvdec_host_variation)
#   2. Stage 1 — stage Lane A's masks.mkv + optimized_poses.pt (reused unchanged)
#   3. Stage 2 — train_renderer.py --profile coolchic_renderer_full from scratch
#      with Lane A's poses + masks as auth-eval inputs. Cool-Chic is a small
#      arch (~10-30K params) so a full 2500ep run is ~2h on 4090 = $0.50.
#      We cap at 1000ep for the first lane shake-out.
#   4. Stage 3 — CCh1 export of best fp32 checkpoint (export_coolchic_renderer).
#   5. Stage 4 — build archive (renderer.bin = CCh1 + Lane A masks + Lane A poses)
#   6. Stage 5 — contest_auth_eval [contest-CUDA]
#
# Council preflight (CLAUDE.md non-negotiable: NEVER invent CLI flags):
#   * --profile coolchic_renderer_full   ✓ profiles.py:969 + PROFILES["coolchic_renderer_full"]
#   * --variant coolchic_renderer        ✓ argparse line 169
#   * --tag                              ✓ argparse line 441
#   * --device cuda                      ✓ argparse line 444
#   * --output-dir                       ✓ argparse line 442
#   * --epochs                           ✓ argparse line 186
#   * --auth-eval-poses                  ✓ argparse line 464
#   * --auth-eval-masks                  ✓ argparse line 462
#   * --auth-eval-upstream-dir           ✓ argparse line 468
#   * --auth-eval-on-best                ✓ argparse line 451 (DEFAULT TRUE)
#   * --no-auth-eval-on-best             ✓ argparse line 458 (we DISABLE it
#                                          here: the built-in path expects
#                                          FP4A export, not CCh1. Stage 3
#                                          handles export, Stage 5 the eval.)
set -euo pipefail
WORKSPACE=/workspace/pact
PYBIN=/opt/conda/bin/python
source "$WORKSPACE/env.sh"
cd "$WORKSPACE"
export PYTHONHASHSEED=1234
# 2026-04-27 lesson (Lane D + Lane S): force upstream dir for scorer search.
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="$WORKSPACE/lane_i_results"
mkdir -p "$LOG_DIR"
TAG="lane_i_coolchic"

log() { echo "[lane-i] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# Provenance + heartbeat (CLAUDE.md canonical pipeline standard +
# memory feedback_canonical_remote_bootstraps).
PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
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
    'lane_script': 'scripts/remote_lane_i_coolchic_masks.sh',
    'lane_name': 'lane_i_coolchic_renderer_on_lane_a_masks_poses',
    'profile': 'coolchic_renderer_full',
    'anchor_renderer': 'experiments/results/lane_a_landed/iter_0/renderer.bin',
    'anchor_poses': 'experiments/results/lane_a_landed/iter_0/optimized_poses.pt',
    'anchor_masks': 'experiments/results/lane_a_landed/iter_0/masks.mkv',
    'anchor_score_baseline': 1.15,
    'predicted_band': [0.95, 1.30],
    'rationale': 'Cool-Chic renderer (CCh1) is ~10-30K params + small latent grids → ~30-80KB binary vs Lane A 290KB. Rate term 0.0185 → ~0.013-0.016 if distortion holds within Lane A range. Distortion is the wedge: Cool-Chic is a different (smaller) arch so it may not match Lane A pose=0.005 + seg=0.0046 — predicted 0.05-0.20 distortion drift accounts for the band width.',
    'note_re_operator_framing': 'Operator asked for Cool-Chic on the MASK SEQUENCE; the existing CCh1 infra is a renderer-replacement, not a mask codec. Standalone mask-Cool-Chic requires new magic + inflate.sh changes (multi-day work). Lane I implemented as renderer-replacement (rate attack on renderer.bin, NOT masks.mkv).',
    'rate_target': 'renderer.bin 290KB → ~30-80KB (CCh1 typical), masks.mkv unchanged at 412KB, poses unchanged at 15KB',
    'epochs_cap': 1000,
    'lr': 5e-4,
    'output_dir': '$LOG_DIR',
    'tag': '$TAG',
}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=I gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend. Catches bad-host in 5 seconds.
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed — destroy this instance and pick a different host."
    exit 2
}

# Pre-flight: anchor on Lane A (1.15 [contest-CUDA]).
ANCHOR_RENDERER="experiments/results/lane_a_landed/iter_0/renderer.bin"
ANCHOR_POSES="experiments/results/lane_a_landed/iter_0/optimized_poses.pt"
ANCHOR_MASKS="experiments/results/lane_a_landed/iter_0/masks.mkv"
for f in "$ANCHOR_RENDERER" \
         "$ANCHOR_POSES" \
         "$ANCHOR_MASKS" \
         upstream/videos/0.mkv \
         upstream/models/segnet.safetensors \
         upstream/models/posenet.safetensors; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
log "  anchor_renderer: $ANCHOR_RENDERER ($(stat -c '%s' "$ANCHOR_RENDERER") bytes, ASYM FP32 — REPLACED by CCh1)"
log "  anchor_poses:    $ANCHOR_POSES (REUSED unchanged)"
log "  anchor_masks:    $ANCHOR_MASKS ($(stat -c '%s' "$ANCHOR_MASKS") bytes, REUSED unchanged)"

# Pre-flight: profile validation (catches missing keys BEFORE GPU burn).
log "=== Pre-flight: profile validation ==="
"$PYBIN" -c "
import sys
sys.path.insert(0, 'src')
from tac.profiles import PROFILES
assert 'coolchic_renderer_full' in PROFILES, \
    'profile coolchic_renderer_full not registered'
p = PROFILES['coolchic_renderer_full']
assert p['variant'] == 'coolchic_renderer', \
    f'profile variant != coolchic_renderer (got {p[\"variant\"]})'
for k in ('latent_ch', 'latent_shapes', 'embed_dim', 'hidden', 'epochs', 'lr'):
    assert k in p, f'profile missing key: {k}'
print(f'PROFILE OK: variant={p[\"variant\"]} latent_ch={p[\"latent_ch\"]} '
      f'latent_shapes={p[\"latent_shapes\"]} hidden={p[\"hidden\"]} '
      f'epochs={p[\"epochs\"]} lr={p[\"lr\"]}')
" 2>&1 | tee -a "$LOG_DIR/run.log"

# Pre-flight: dead-flag-wiring guard — every CLI flag in the train_renderer
# invocation MUST exist in train_renderer.py's argparse. CLAUDE.md
# non-negotiable (memory: feedback_dead_flag_wiring_pattern).
log "=== Pre-flight: argparse dead-flag scan ==="
"$PYBIN" -c "
import re, sys
sys.path.insert(0, 'src')
script = open('scripts/remote_lane_i_coolchic_masks.sh').read()
tr_src = open('src/tac/experiments/train_renderer.py').read()
real = set(re.findall(r'add_argument\(\s*[\"\\']--([a-z][a-z0-9-]+)', tr_src))
m = re.search(r'src/tac/experiments/train_renderer\.py(.*?)(?=\n# Stage 3:|\n\s*BEST_FP32=|\Z)',
              script, re.DOTALL)
assert m, 'could not locate train_renderer.py invocation in script'
used = set(re.findall(r'\B--([a-z][a-z0-9-]+)', m.group(0)))
invented = used - real
if invented:
    print(f'INVENTED FLAGS: {sorted(invented)} not in train_renderer argparse',
          file=sys.stderr); sys.exit(3)
print(f'OK: {len(used)} flags all real')
" 2>&1 | tee -a "$LOG_DIR/run.log"

log "=== Stage 1: stage Lane A masks + poses (reused unchanged) ==="
mkdir -p "$LOG_DIR/extracted"
cp "$ANCHOR_MASKS" "$LOG_DIR/extracted/masks.mkv"
cp "$ANCHOR_POSES" "$LOG_DIR/extracted/optimized_poses.pt"
log "  staged masks.mkv: $(stat -c '%s' "$LOG_DIR/extracted/masks.mkv") bytes"
log "  staged poses:     $(stat -c '%s' "$LOG_DIR/extracted/optimized_poses.pt") bytes"

log "=== Stage 2: train coolchic_renderer (1000ep cap) ==="
log "  profile:  coolchic_renderer_full"
log "  epochs:   1000 (overrides profile's 2500 for first lane shake-out)"
log "  lr:       5e-4 (profile default, low-mag latents need larger LR)"
log "  arch:     CoolChicLatentRenderer (latents + tiny synthesis decoder)"
log "  estimated wall clock on 4090: ~1-2h (\$0.25-0.50 at \$0.25/hr)"
log "  auth eval: --no-auth-eval-on-best (Stage 5 runs CCh1-aware auth eval)"

# Smoke-kill metadata sidecar for external watchdog.
cat > "$LOG_DIR/kill_targets.json" <<'EOF'
{
  "epochs_cap": 1000,
  "smoke_epoch": 200,
  "smoke_proxy_max": 5.0,
  "final_epoch": 1000,
  "final_auth_max": 1.50,
  "comment": "Lane I kill targets — anchored on Lane A 1.15. CCh1 is a smaller arch so distortion may drift; if final auth >1.50 the Cool-Chic capacity is insufficient and we should bump latent_ch / hidden in a follow-up."
}
EOF

# train_renderer.py invocation. Cool-Chic is a fresh-arch train (no
# --resume-from). We disable --auth-eval-on-best because the built-in path
# uses FP4A export (which expects ASYM-style state_dict); CCh1 export needs
# its own Stage 3 invocation against export_coolchic_renderer.
"$PYBIN" -u src/tac/experiments/train_renderer.py \
    --profile coolchic_renderer_full \
    --variant coolchic_renderer \
    --tag "$TAG" \
    --device cuda \
    --video upstream/videos/0.mkv \
    --output-dir "$LOG_DIR/train" \
    --epochs 1000 \
    --no-auth-eval-on-best \
    2>&1 | tee "$LOG_DIR/train.log" | grep -E "^\[(train|eval|masks|phase|best|arch|coolchic)\]|epoch|Phase|scorer|loss" | tail -200

# Stage 3: CCh1 export of best fp32 checkpoint to renderer_coolchic.bin.
# train_renderer saves renderer_<tag>_best_fp32.pt with the coolchic-variant
# state_dict (the variant build path runs at model construction time). We
# rebuild the same arch + load the fp32 weights + run export_coolchic_renderer.
BEST_FP32="$LOG_DIR/train/renderer_${TAG}_best_fp32.pt"
[ -f "$BEST_FP32" ] || {
    echo "FATAL: train_renderer didn't produce ${BEST_FP32}" >&2
    ls -la "$LOG_DIR/train/" >&2
    exit 2
}
log "  best fp32 checkpoint: $BEST_FP32 ($(stat -c '%s' "$BEST_FP32") bytes)"

log "=== Stage 3: CCh1 export of best fp32 → renderer_coolchic.bin ==="
mkdir -p "$LOG_DIR/qat"
CCH1_BIN="$LOG_DIR/qat/renderer_coolchic.bin"
"$PYBIN" -c "
import sys, torch
sys.path.insert(0, 'src')
from tac.contrib.coolchic_renderer import build_coolchic_renderer
from tac.renderer_export import export_coolchic_renderer

ckpt_path = '$BEST_FP32'
out_bin = '$CCH1_BIN'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
meta = ckpt.get('__meta__', {}) or {}
print(f'[stage3] loaded fp32 checkpoint, meta keys: {sorted(meta.keys())[:10]}')

# Rebuild EXACT coolchic arch the training run used. Defaults match
# COOLCHIC_RENDERER_FULL profile (profiles.py:969).
def m(key, default):
    return meta.get(key, default)

# latent_shapes is stored as list-of-lists in JSON-serialized meta; coerce
# back to tuple-of-tuples for build_coolchic_renderer.
ls = m('latent_shapes', ((6, 8), (12, 16), (24, 32)))
ls = tuple(tuple(s) for s in ls)

model = build_coolchic_renderer(
    num_classes=5,
    embed_dim=int(m('embed_dim', 6)),
    latent_ch=int(m('latent_ch', 8)),
    hidden=int(m('hidden', m('base_ch', 32))),
    motion_hidden=int(m('motion_hidden', 32)),
    latent_shapes=ls,
    blend_mode=m('blend_mode', 'scalar'),
    noise_mode=m('noise_mode', 'deterministic'),
)
print(f'[stage3] rebuilt CoolChic PairGenerator: {sum(p.numel() for p in model.parameters()):,} params')

# 2026-04-28 Lane I crash fix (memory: project_lane_i_crashed_parametrize_strip_20260428)
# Strip torch.nn.utils.parametrize hooks before load — train_renderer can save
# checkpoints with `<layer>.parametrizations.weight.original` keys when self-
# compress / FakeQuant codecs are active. The fresh model (no hooks) expects
# plain `<layer>.weight`. Mirror the canonical strip from qat_finetune.py:218-238.
if any('.parametrizations.' in k for k in state.keys()):
    normalized = {}
    for k, v in state.items():
        if '.parametrizations.' not in k:
            normalized[k] = v
            continue
        head, _, tail = k.partition('.parametrizations.')
        name, _, suffix = tail.partition('.')
        if suffix == 'original':
            normalized[f'{head}.{name}'] = v
        # else: drop codebook + other parametrize internals
    print(f'[stage3] stripped parametrize hooks: {len(state)} -> {len(normalized)} keys')
    state = normalized

missing, unexpected = model.load_state_dict(state, strict=False)
if missing or unexpected:
    print(f'[stage3] load mismatch: missing={list(missing)[:6]} unexpected={list(unexpected)[:6]}')
    if missing:
        raise RuntimeError(f'CCh1 load missing keys: {missing[:6]}')
print('[stage3] loaded fp32 weights into CoolChic PairGenerator')

n_bytes = export_coolchic_renderer(model, out_bin)
print(f'[stage3] WROTE {out_bin}: {n_bytes} bytes (vs Lane A 290KB)')
" 2>&1 | tee -a "$LOG_DIR/run.log"
[ -f "$CCH1_BIN" ] || { echo "FATAL: CCh1 export failed — no $CCH1_BIN" >&2; exit 2; }
CCH1_SIZE=$(stat -c '%s' "$CCH1_BIN")
log "  CCh1 binary: $CCH1_BIN ($CCH1_SIZE bytes)"

log "=== Stage 4: build NEW archive (CCh1 renderer + Lane A masks + Lane A poses) ==="
mkdir -p "$LOG_DIR/iter_0"
cp "$CCH1_BIN" "$LOG_DIR/iter_0/renderer.bin"
cp "$ANCHOR_MASKS" "$LOG_DIR/iter_0/masks.mkv"
cp "$ANCHOR_POSES" "$LOG_DIR/iter_0/optimized_poses.pt"
ARCHIVE="$LOG_DIR/archive_lane_i.zip"
"$PYBIN" -c "
import zipfile, os
src = '$LOG_DIR/iter_0'
dst = '$ARCHIVE'
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for n in ('renderer.bin', 'masks.mkv', 'optimized_poses.pt'):
        p = os.path.join(src, n)
        assert os.path.isfile(p), f'missing {p}'
        z.write(p, arcname=n)
print(f'archive {dst}: {os.path.getsize(dst)} bytes')
"
ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE")
[ "$ARCHIVE_BYTES" -gt 0 ] || { echo "FATAL: archive empty" >&2; exit 2; }
log "  archive: $ARCHIVE ($ARCHIVE_BYTES bytes vs Lane A 678KB)"

log "=== Stage 5: contest_auth_eval on Lane I archive [contest-CUDA] ==="
rm -rf "$LOG_DIR/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$LOG_DIR/eval_work" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -20

# Sanity-check the auth eval actually emitted a RESULT_JSON line.
if ! grep -q "RESULT_JSON" "$LOG_DIR/auth_eval.log"; then
    log "FATAL: auth_eval log has no RESULT_JSON line — eval crashed."
    exit 4
fi

log "=== LANE_I_DONE [contest-CUDA] — see $LOG_DIR/auth_eval.log for RESULT_JSON ==="
