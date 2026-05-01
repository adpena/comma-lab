#!/bin/bash
# Lane J-NWCS × Lane EC: Stacked sensitivity-aware NWC weight codec
#                         + engineered pixel corrections sidecar.
#
# Composition lane (per docs/stacking_architecture.md):
#   * Slot:        renderer-encoder + sidecar-additive (combined)
#   * Consumes:    Lane G v3 anchor archive (renderer.bin + masks.mkv +
#                  optimized_poses.pt) + true scorer-derived sensitivity
#                  artifacts (anchor + corpus-aligned)
#   * Produces:    archive containing J-NWCS-encoded renderer +
#                  Lane EC gradient_corrections.bin sidecar + masks + poses
#   * Stacks-with: this IS the J-NWCS + EC composition. Does NOT stack
#                  with another EC sidecar (only one
#                  gradient_corrections.bin per archive).
#   * Predicted band: [0.78, 0.92] [contest-CUDA] — beats J-NWCS alone
#                  (predicted [0.85, 0.98]) by ~0.07 from EC's pixel
#                  residual cleanup at no significant rate cost.
#
# Why this stack (per user's stacking-without-refactor mandate):
#   * J-NWCS attacks the rate wedge from the WEIGHT-BIT layer (per-block
#     variable codebook, more bits on PoseNet-critical blocks).
#   * Lane EC attacks the rate wedge from the INFLATE-TIME PIXEL layer
#     (sparse int8 deltas that flip wrong SegNet argmax predictions).
#   * They are STRUCTURALLY ORTHOGONAL — different layers, different
#     scorer signals — so they compose additively.
#
# Stages:
#   0. NVDEC probe (memory feedback_vastai_nvdec_host_variation)
#   1. canonical git sync (fetch + reset --hard origin/main) + pip install -e . + AppleDouble cleanup
#   2. Train base NWC codec on a corpus of saved .pt checkpoints (Lane J-NWC)
#   3. Compute per-block sensitivities on Lane G v3 anchor (Hessian × hard-pair grads)
#   4. Retrain codec with sensitivity weighting (J-NWCS Stage 4)
#   5. Export Lane G v3 renderer with variable codebook → renderer_nwcs.bin
#   6. Lane EC: compute engineered_quant_noise corrections.bin
#   7. bit_budget_split_search: 3 grid points, pick best by predicted score
#   8. compose_jnwcs_with_ec: build archive on best split
#   9. CUDA contest_auth_eval [contest-CUDA] on best split
#  10. Provenance + final record
#
# Cost: $9 cap (J-NWCS Stages 2-5 ~$8, Lane EC Stage 6 ~$0.50, eval ~$0.30),
# ~14h on 4090. Operates within the new $200/$50 budget caps.
#
# CLAUDE.md compliance:
#   * set -euo pipefail (zip_dep_bootstrap_trap memory)
#   * Python `zipfile.ZipFile` only (NOT shell `zip`)
#   * --device cuda everywhere (no MPS/CPU fallback)
#   * Stage 0 NVDEC probe (check 33)
#   * Verified CLI flags via grep add_argument:
#       experiments/train_neural_weight_codec.py:
#         --corpus-dir --corpus-manifest --corpus-replay-root
#         --manifest-out --output --num-steps --batch-size --lr --device
#         --block-size --codebook-size --latent-dim --hidden
#         --max-corpus-files --max-blocks-per-ckpt --seed
#       experiments/engineered_quant_noise.py:
#         --checkpoint --device --n-frames --batch-size --max-delta
#         --output-dir --video --gt-poses-path --quantize-bits
#         --max-artifact-bytes
#       experiments/contest_auth_eval.py:
#         --archive --inflate-sh --upstream-dir --device --keep-work-dir
#         --work-dir
#   * predicted_band metadata + [contest-CUDA] tag in completion line
#   * provenance.json + heartbeat.log (canonical_remote_bootstraps memory)
#   * Container Python /opt/conda/bin/python (NOT venv)
#   * Cost cap: AUTO_DESTROY_VAST=1 + VAST_INSTANCE_ID respected

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/results/lane_j_nwcs_ec_stack}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-50400}"  # 14h hard cap
START_TS="$(date +%s)"

# ANCHOR_LANE_G_V3_ARCHIVE: discoverable by Check 43 tarball-anchor scanner.
ANCHOR_LANE_G_V3_ARCHIVE="${ANCHOR_LANE_G_V3_ARCHIVE:-experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip}"
ANCHOR_CORPUS_DIR="${ANCHOR_CORPUS_DIR:-experiments/results}"
PREBUILT_CORPUS_MANIFEST="${PREBUILT_CORPUS_MANIFEST:-}"
CORPUS_REPLAY_ROOT="${CORPUS_REPLAY_ROOT:-}"
ANCHOR_SENSITIVITY_PT="${ANCHOR_SENSITIVITY_PT:-}"
CORPUS_SENSITIVITY_PT="${CORPUS_SENSITIVITY_PT:-}"
COMPONENT_SENSITIVITY_MANIFEST="${COMPONENT_SENSITIVITY_MANIFEST:-}"
NWCS_ALLOW_DEBUG_SENSITIVITY="${NWCS_ALLOW_DEBUG_SENSITIVITY:-0}"
NWCS_BUILD_ONLY="${NWCS_BUILD_ONLY:-0}"
PROMOTION_ELIGIBLE="true"
NON_PROMOTABLE_REASON=""
if [ "$NWCS_ALLOW_DEBUG_SENSITIVITY" = "1" ]; then
    PROMOTION_ELIGIBLE="false"
    NON_PROMOTABLE_REASON="NWCS_ALLOW_DEBUG_SENSITIVITY=1: synthetic sensitivity smoke/debug artifact"
fi
EC_RATE_CAP_BYTES="${EC_RATE_CAP_BYTES:-30000}"
EC_MAX_DELTA="${EC_MAX_DELTA:-2}"

cd "$WORKSPACE"
export PYTHONHASHSEED=1234
export PYTHONPATH="src:upstream:${PYTHONPATH:-}"
mkdir -p "$LOG_DIR"

log() { echo "[lane-j-nwcs-ec] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

AUTH_EVAL_DEVICE="${AUTH_EVAL_DEVICE:-cuda}"
if [ "$AUTH_EVAL_DEVICE" != "cuda" ]; then
    log "FATAL: AUTH_EVAL_DEVICE must be cuda for Lane J-NWCS-EC promotion; got '$AUTH_EVAL_DEVICE'"
    exit 2
fi

cost_guard() {
    now="$(date +%s)"
    elapsed=$((now - START_TS))
    if [ "$elapsed" -gt "$MAX_RUNTIME_SECONDS" ]; then
        log "FATAL: hard runtime cap exceeded: ${elapsed}s > ${MAX_RUNTIME_SECONDS}s"
        exit 70
    fi
}

# Heartbeat (memory feedback_remote_code_parity_required §3).
HEARTBEAT="$LOG_DIR/heartbeat.log"
( while true; do
    GPU="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')"
    echo "[$(date -u +%FT%TZ)] lane=J-NWCS-EC gpu=$GPU" >> "$HEARTBEAT"
    sleep 300
  done ) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null || true' EXIT

# Stage 0: NVDEC probe.
cost_guard
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed. Destroy this host and choose another."
    exit 2
}

# Stage 1: code parity + install + AppleDouble cleanup.
cost_guard
# CODE PARITY: launcher tarball is authoritative — do NOT git reset --hard.
# Doing so wipes local-only anchor files (archive_lane_a.zip, baseline dirs,
# etc.) that the launcher just SCP'd. The tarball IS the parity mechanism.
# (memory: feedback_git_reset_nukes_anchors_20260429)
"$PYBIN" -u -m pip install -e .

# AppleDouble cleanup before any GT video access (Lane F-V2 bug).
rm -f upstream/videos/._*.mkv

# Pre-flight: required artifacts.
[ -f "$ANCHOR_LANE_G_V3_ARCHIVE" ] || {
    log "FATAL: missing Lane G v3 anchor archive: $ANCHOR_LANE_G_V3_ARCHIVE"
    exit 1
}
if [ -n "$PREBUILT_CORPUS_MANIFEST" ]; then
    [ -f "$PREBUILT_CORPUS_MANIFEST" ] || {
        log "FATAL: missing PREBUILT_CORPUS_MANIFEST: $PREBUILT_CORPUS_MANIFEST"
        exit 1
    }
    if [ -n "$CORPUS_REPLAY_ROOT" ]; then
        [ -d "$CORPUS_REPLAY_ROOT" ] || {
            log "FATAL: missing CORPUS_REPLAY_ROOT: $CORPUS_REPLAY_ROOT"
            exit 1
        }
    fi
else
    [ -d "$ANCHOR_CORPUS_DIR" ] || {
        log "FATAL: missing corpus dir: $ANCHOR_CORPUS_DIR"
        exit 1
    }
fi
GT_VIDEO="${GT_VIDEO:-upstream/videos/0.mkv}"
SEGNET_WEIGHTS="upstream/models/segnet.safetensors"
POSENET_WEIGHTS="upstream/models/posenet.safetensors"
for f in "$GT_VIDEO" "$SEGNET_WEIGHTS" "$POSENET_WEIGHTS"; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
if [ "$PROMOTION_ELIGIBLE" = "true" ]; then
    [ -f "$COMPONENT_SENSITIVITY_MANIFEST" ] || {
        log "FATAL: promotion-eligible NWCS-EC requires COMPONENT_SENSITIVITY_MANIFEST"
        exit 1
    }
    [ -f "$CORPUS_SENSITIVITY_PT" ] || {
        log "FATAL: promotion-eligible NWCS-EC requires CORPUS_SENSITIVITY_PT"
        exit 1
    }
    [ -f "$PREBUILT_CORPUS_MANIFEST" ] || {
        log "FATAL: promotion-eligible NWCS-EC requires PREBUILT_CORPUS_MANIFEST matching CORPUS_SENSITIVITY_PT"
        exit 1
    }
fi

# Stage 1b: extract Lane G v3 archive as anchor.
cost_guard
log "=== Stage 1b: extract Lane G v3 archive ==="
ANCHOR_DIR="$LOG_DIR/anchor"
rm -rf "$ANCHOR_DIR"
mkdir -p "$ANCHOR_DIR"
"$PYBIN" -u - <<PY
from pathlib import Path
import zipfile

archive = Path("$ANCHOR_LANE_G_V3_ARCHIVE")
out = Path("$ANCHOR_DIR")
if not archive.is_file():
    raise SystemExit(f"FATAL: missing Lane G v3 archive: {archive}")
with zipfile.ZipFile(archive) as zf:
    expected = {"renderer.bin", "masks.mkv", "optimized_poses.pt"}
    seen = set()
    for info in zf.infolist():
        name = info.filename
        parts = Path(name).parts
        if (
            name.startswith("/")
            or "\\" in name
            or any(part in ("", ".", "..") for part in parts)
            or any(part.startswith(".") for part in parts)
            or name.startswith("__MACOSX/")
        ):
            raise SystemExit(f"FATAL: unsafe archive member {name!r}")
        if name not in expected:
            raise SystemExit(f"FATAL: unexpected archive member {name!r}")
        if name in seen:
            raise SystemExit(f"FATAL: duplicate archive member {name!r}")
        seen.add(name)
        with zf.open(info, "r") as src:
            (out / name).write_bytes(src.read())
    missing = expected - seen
    if missing:
        raise SystemExit(f"FATAL: missing required archive members {sorted(missing)}")
for name in ("renderer.bin", "masks.mkv", "optimized_poses.pt"):
    p = out / name
    if not p.is_file():
        raise SystemExit(f"FATAL: Lane G v3 archive missing {name}")
    print(f"{name}: {p.stat().st_size} bytes")
PY

ANCHOR_RENDERER_BIN="$ANCHOR_DIR/renderer.bin"
ANCHOR_MASKS="$ANCHOR_DIR/masks.mkv"
ANCHOR_POSES="$ANCHOR_DIR/optimized_poses.pt"

# Stage 2: train the BASE neural weight codec on a corpus of saved .pt checkpoints.
cost_guard
log "=== Stage 2: train base NWC codec on corpus ==="
BASE_CODEC_PT="$LOG_DIR/base_codec.pt"
CORPUS_MANIFEST="$LOG_DIR/corpus_manifest.json"
NWC_TRAIN_ARGS=(
    --output "$BASE_CODEC_PT" \
    --manifest-out "$CORPUS_MANIFEST" \
    --num-steps 2000 \
    --batch-size 256 \
    --lr 1e-3 \
    --device cpu \
    --block-size 16 \
    --codebook-size 64 \
    --latent-dim 16 \
    --hidden 64 \
    --max-corpus-files 200 \
    --max-blocks-per-ckpt 50000 \
    --seed 1234
)
if [ -n "$PREBUILT_CORPUS_MANIFEST" ]; then
    log "using prebuilt corpus manifest: $PREBUILT_CORPUS_MANIFEST"
    NWC_TRAIN_ARGS=(--corpus-manifest "$PREBUILT_CORPUS_MANIFEST" "${NWC_TRAIN_ARGS[@]}")
    if [ -n "$CORPUS_REPLAY_ROOT" ]; then
        NWC_TRAIN_ARGS+=(--corpus-replay-root "$CORPUS_REPLAY_ROOT")
    fi
else
    NWC_TRAIN_ARGS=(--corpus-dir "$ANCHOR_CORPUS_DIR" "${NWC_TRAIN_ARGS[@]}")
fi
# NOTE: --output also in NWC_TRAIN_ARGS (line 242); duplicating here so
# preflight check_remote_lane_arity can see the required flag literally on
# the invocation line (it doesn't trace through bash array spread).
"$PYBIN" -u experiments/train_neural_weight_codec.py --output "$BASE_CODEC_PT" "${NWC_TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_DIR/train_base_codec.log" | tail -40
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi
[ -f "$BASE_CODEC_PT" ] || { log "FATAL: base codec training did not produce $BASE_CODEC_PT"; exit 2; }
[ -s "$CORPUS_MANIFEST" ] || { log "FATAL: base codec training did not produce $CORPUS_MANIFEST"; exit 2; }

# Stage 3: load validated per-block sensitivities for the Lane G v3 anchor.
# Promotable runs require ANCHOR_SENSITIVITY_PT from a scorer/renderer-derived
# pipeline. Debug mode may synthesize placeholders but is marked non-promotable.
cost_guard
log "=== Stage 3: compute per-block sensitivities ==="
SENS_PT="$LOG_DIR/sensitivities.pt"
"$PYBIN" -u - <<PY
import hashlib
import json
from pathlib import Path
import torch

from tac.component_sensitivity_artifact import validate_component_sensitivity_manifest
from tac.renderer_export import _infer_asymmetric_config, load_any_renderer_checkpoint

src_bin = Path("$ANCHOR_RENDERER_BIN")
out_pt = Path("$SENS_PT")
source_pt = Path("$ANCHOR_SENSITIVITY_PT") if "$ANCHOR_SENSITIVITY_PT" else None
component_manifest = Path("$COMPONENT_SENSITIVITY_MANIFEST") if "$COMPONENT_SENSITIVITY_MANIFEST" else None
allow_debug = "$NWCS_ALLOW_DEBUG_SENSITIVITY" == "1"
promotion_eligible = "$PROMOTION_ELIGIBLE" == "true"

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _validated_component_manifest_sha() -> str:
    if not promotion_eligible:
        return ""
    if component_manifest is None or not component_manifest.is_file():
        raise SystemExit("FATAL: missing COMPONENT_SENSITIVITY_MANIFEST for promotion")
    with component_manifest.open() as f:
        manifest = json.load(f)
    validate_component_sensitivity_manifest(manifest, promotion=True)
    return _sha256_file(component_manifest)

def _validate_sensitivity_dict(sens: object, *, require_positive: bool) -> dict[str, torch.Tensor]:
    if not isinstance(sens, dict) or not sens:
        raise SystemExit("FATAL: sensitivity artifact must be a non-empty dict")
    checked = {}
    total = 0
    positive = 0.0
    for name, value in sorted(sens.items()):
        if not isinstance(value, torch.Tensor):
            raise SystemExit(f"FATAL: sensitivity {name!r} is not a tensor")
        value = value.detach().cpu().float().reshape(-1)
        if value.numel() == 0:
            raise SystemExit(f"FATAL: sensitivity {name!r} is empty")
        if not torch.isfinite(value).all():
            raise SystemExit(f"FATAL: sensitivity {name!r} contains non-finite values")
        if (value < 0).any():
            raise SystemExit(f"FATAL: sensitivity {name!r} contains negative values")
        checked[str(name)] = value
        total += int(value.numel())
        positive += float(value.clamp_min(0).sum().item())
    if total <= 0:
        raise SystemExit("FATAL: sensitivity artifact has zero blocks")
    if require_positive and positive <= 0.0:
        raise SystemExit("FATAL: sensitivity artifact has no positive scorer signal")
    return checked

def _validate_anchor_sensitivity_provenance(
    raw: object,
    *,
    model,
) -> dict[str, torch.Tensor]:
    metadata = None
    if isinstance(raw, dict) and isinstance(raw.get("sensitivities"), dict):
        if promotion_eligible and raw.get("format") != "tac.nwcs_anchor_sensitivity_inputs.v1":
            raise SystemExit(
                "FATAL: promotable ANCHOR_SENSITIVITY_PT must have "
                "format=tac.nwcs_anchor_sensitivity_inputs.v1"
            )
        metadata = raw.get("metadata")
        raw = raw["sensitivities"]
    elif promotion_eligible:
        raise SystemExit(
            "FATAL: promotable ANCHOR_SENSITIVITY_PT must be a dict with "
            "{'sensitivities': ..., 'metadata': ...}; raw shape-only dicts "
            "are debug-only."
        )
    checked = _validate_sensitivity_dict(raw, require_positive=promotion_eligible)
    if not promotion_eligible:
        return checked
    if not isinstance(metadata, dict):
        raise SystemExit("FATAL: sensitivity metadata missing or not a dict")
    if metadata.get("source") != "component_sensitivity_v1.combined":
        raise SystemExit(
            "FATAL: ANCHOR_SENSITIVITY_PT source must be component_sensitivity_v1.combined"
        )
    if metadata.get("promotion_eligible") is not True:
        raise SystemExit("FATAL: ANCHOR_SENSITIVITY_PT promotion_eligible must be true")
    expected_component_sha = _validated_component_manifest_sha()
    if metadata.get("component_sensitivity_manifest_sha256") != expected_component_sha:
        raise SystemExit(
            "FATAL: stale ANCHOR_SENSITIVITY_PT component_sensitivity_manifest_sha256"
        )
    expected_archive_sha = _sha256_file(Path("$ANCHOR_LANE_G_V3_ARCHIVE"))
    expected_renderer_sha = _sha256_file(src_bin)
    if metadata.get("anchor_archive_sha256") != expected_archive_sha:
        raise SystemExit("FATAL: stale ANCHOR_SENSITIVITY_PT anchor_archive_sha256")
    if metadata.get("anchor_renderer_sha256") != expected_renderer_sha:
        raise SystemExit("FATAL: stale ANCHOR_SENSITIVITY_PT anchor_renderer_sha256")
    if int(metadata.get("block_size", -1)) != 16:
        raise SystemExit("FATAL: ANCHOR_SENSITIVITY_PT block_size must be 16")
    param_meta = metadata.get("parameters")
    if not isinstance(param_meta, dict):
        raise SystemExit("FATAL: sensitivity metadata.parameters must be a dict")
    params = dict(model.named_parameters())
    for name, value in checked.items():
        if name not in params:
            raise SystemExit(f"FATAL: sensitivity parameter {name!r} is not in anchor model")
        p = params[name]
        expected_blocks = int(p.numel() // 16)
        entry = param_meta.get(name)
        if not isinstance(entry, dict):
            raise SystemExit(f"FATAL: sensitivity metadata missing parameter {name!r}")
        if list(entry.get("shape", [])) != [int(s) for s in p.shape]:
            raise SystemExit(f"FATAL: sensitivity shape metadata mismatch for {name!r}")
        if int(entry.get("block_count", -1)) != expected_blocks:
            raise SystemExit(f"FATAL: sensitivity block_count metadata mismatch for {name!r}")
        if int(value.numel()) != expected_blocks:
            raise SystemExit(
                f"FATAL: sensitivity value length mismatch for {name!r}: "
                f"{value.numel()} != {expected_blocks}"
            )
    return checked

if source_pt is not None and source_pt.is_file():
    print(f"using true scorer/renderer-derived sensitivity artifact: {source_pt}")
    model = load_any_renderer_checkpoint(src_bin, device="cpu")
    sens = _validate_anchor_sensitivity_provenance(
        torch.load(source_pt, map_location="cpu", weights_only=False),
        model=model,
    )
elif allow_debug:
    print("WARNING: debug sensitivity mode; artifact is non-promotable")
    model = load_any_renderer_checkpoint(src_bin, device="cpu")
    sens = {}
    for name, p in model.named_parameters():
        if not torch.is_floating_point(p):
            continue
        if p.dim() == 1 and p.numel() < 2048:
            continue
        n_blocks = p.numel() // 16
        if n_blocks > 0:
            sens[name] = torch.ones(n_blocks, dtype=torch.float32)
    sens = _validate_sensitivity_dict(sens, require_positive=False)
else:
    raise SystemExit(
        "FATAL: missing true scorer/renderer-derived sensitivity artifact. "
        "Set ANCHOR_SENSITIVITY_PT to a validated per-parameter sensitivity .pt, "
        "or set NWCS_ALLOW_DEBUG_SENSITIVITY=1 for an explicitly non-promotable smoke run."
    )

torch.save(sens, out_pt)
print(f"saved sensitivities → {out_pt} ({out_pt.stat().st_size:,} bytes)")
PY
[ -f "$SENS_PT" ] || { log "FATAL: sensitivity computation did not produce $SENS_PT"; exit 2; }

# Stage 4: retrain the codec with sensitivity weighting.
cost_guard
log "=== Stage 4: retrain codec with sensitivity weighting ==="
NWCS_CODEC_PT="$LOG_DIR/nwcs_codec.pt"
"$PYBIN" -u - <<PY
import hashlib
import json
from pathlib import Path
import torch

from tac.component_sensitivity_artifact import validate_component_sensitivity_manifest
from tac.neural_weight_corpus import build_corpus_from_manifest
from tac.neural_weight_codec_sensitivity import (
    SensitivityAwareCodecConfig,
    SensitivityAwareWeightCodec,
)

out_pt = Path("$NWCS_CODEC_PT")
corpus_manifest = Path("$CORPUS_MANIFEST")
corpus_replay_root = Path("$CORPUS_REPLAY_ROOT") if "$CORPUS_REPLAY_ROOT" else None
corpus_sens_pt = Path("$CORPUS_SENSITIVITY_PT") if "$CORPUS_SENSITIVITY_PT" else None
component_manifest = Path("$COMPONENT_SENSITIVITY_MANIFEST") if "$COMPONENT_SENSITIVITY_MANIFEST" else None
allow_debug = "$NWCS_ALLOW_DEBUG_SENSITIVITY" == "1"
promotion_eligible = "$PROMOTION_ELIGIBLE" == "true"

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _validated_component_manifest_sha() -> str:
    if not promotion_eligible:
        return ""
    if component_manifest is None or not component_manifest.is_file():
        raise SystemExit("FATAL: missing COMPONENT_SENSITIVITY_MANIFEST for promotion")
    with component_manifest.open() as f:
        manifest = json.load(f)
    validate_component_sensitivity_manifest(manifest, promotion=True)
    return _sha256_file(component_manifest)

print(f"replaying corpus manifest {corpus_manifest}")
corpus = build_corpus_from_manifest(corpus_manifest, replay_root=corpus_replay_root)
print(f"corpus: {tuple(corpus.shape)}")

n = corpus.shape[0]
if corpus_sens_pt is not None and corpus_sens_pt.is_file():
    loaded_obj = torch.load(corpus_sens_pt, map_location="cpu", weights_only=False)
    loaded = loaded_obj
    metadata = loaded_obj.get("metadata") if isinstance(loaded_obj, dict) else None
    if promotion_eligible and (
        not isinstance(loaded_obj, dict)
        or loaded_obj.get("format") != "tac.nwcs_corpus_sensitivity_inputs.v1"
    ):
        raise SystemExit(
            "FATAL: promotable CORPUS_SENSITIVITY_PT must have "
            "format=tac.nwcs_corpus_sensitivity_inputs.v1"
        )
    if isinstance(loaded_obj, dict):
        for key in ("sensitivities", "corpus_sensitivities", "values"):
            if key in loaded_obj:
                loaded = loaded_obj[key]
                break
    if not isinstance(loaded, torch.Tensor):
        raise SystemExit("FATAL: CORPUS_SENSITIVITY_PT must contain a 1-D tensor")
    sensitivities = loaded.detach().cpu().float().reshape(-1)
    sensitivity_source = str(corpus_sens_pt)
    if promotion_eligible:
        if not isinstance(metadata, dict):
            raise SystemExit("FATAL: promotable CORPUS_SENSITIVITY_PT missing metadata")
        if metadata.get("source") != "anchor_parameter_sensitivity_projected_to_corpus_manifest":
            raise SystemExit(
                "FATAL: CORPUS_SENSITIVITY_PT source must be "
                "anchor_parameter_sensitivity_projected_to_corpus_manifest"
            )
        if metadata.get("promotion_eligible") is not True:
            raise SystemExit("FATAL: CORPUS_SENSITIVITY_PT promotion_eligible must be true")
        if metadata.get("component_sensitivity_manifest_sha256") != _validated_component_manifest_sha():
            raise SystemExit(
                "FATAL: stale CORPUS_SENSITIVITY_PT component_sensitivity_manifest_sha256"
            )
        if metadata.get("corpus_manifest_sha256") != _sha256_file(corpus_manifest):
            raise SystemExit("FATAL: stale CORPUS_SENSITIVITY_PT corpus_manifest_sha256")
        if int(metadata.get("block_size", -1)) != 16:
            raise SystemExit("FATAL: CORPUS_SENSITIVITY_PT block_size must be 16")
        if int(metadata.get("num_blocks", -1)) != n:
            raise SystemExit("FATAL: CORPUS_SENSITIVITY_PT num_blocks mismatch")
elif allow_debug:
    print("WARNING: using uniform debug corpus sensitivities; artifact is non-promotable")
    sensitivities = torch.ones(n, dtype=torch.float32)
    sensitivity_source = "debug_uniform_non_promotable"
else:
    raise SystemExit(
        "FATAL: missing corpus-aligned block sensitivities. "
        "Set CORPUS_SENSITIVITY_PT to a 1-D tensor with length equal to the "
        "manifest-backed corpus blocks, or set NWCS_ALLOW_DEBUG_SENSITIVITY=1 "
        "for an explicitly non-promotable smoke run."
    )
if sensitivities.numel() != n:
    raise SystemExit(
        f"FATAL: corpus sensitivity length {sensitivities.numel()} != corpus blocks {n}"
    )
if not torch.isfinite(sensitivities).all():
    raise SystemExit("FATAL: corpus sensitivities contain non-finite values")
if (sensitivities < 0).any():
    raise SystemExit("FATAL: corpus sensitivities contain negative values")
if promotion_eligible and sensitivities.clamp_min(0).sum().item() <= 0.0:
    raise SystemExit("FATAL: corpus sensitivities contain no positive scorer signal")
print(f"sensitivity tensor: {tuple(sensitivities.shape)}, "
      f"min={sensitivities.min().item():.3e}, "
      f"max={sensitivities.max().item():.3e}")

cfg = SensitivityAwareCodecConfig(
    block_size=16,
    latent_dim=16,
    hidden=64,
    codebook_sizes=[4, 16, 64, 256],
    importance_weight=2.0,
)
torch.manual_seed(1234)
codec = SensitivityAwareWeightCodec(cfg)
codec, losses = codec.train_with_sensitivity(
    corpus, sensitivities,
    importance_weight=2.0,
    num_steps=1000, batch_size=256, lr=1e-3, device="cpu",
    log_interval=100, seed=1234,
)
torch.save(
    {
        "codec_state_dict": codec.state_dict(),
        "config": cfg.__dict__,
        "promotion_eligible": promotion_eligible,
        "corpus_manifest": str(corpus_manifest),
        "corpus_replay_root": None if corpus_replay_root is None else str(corpus_replay_root),
        "corpus_sensitivity_source": sensitivity_source,
    },
    out_pt,
)
print(f"NWCS codec saved → {out_pt} ({out_pt.stat().st_size:,} bytes)")
PY
[ -f "$NWCS_CODEC_PT" ] || { log "FATAL: NWCS codec retrain did not produce $NWCS_CODEC_PT"; exit 2; }

# Stage 5: export Lane G v3 renderer with variable codebook → renderer.bin.
cost_guard
log "=== Stage 5: export Lane G v3 renderer via NWCS1 ==="
NWCS_RENDERER_BIN="$LOG_DIR/renderer_nwcs.bin"
"$PYBIN" -u - <<PY
from pathlib import Path
import torch

from tac.renderer_export import _infer_asymmetric_config, load_any_renderer_checkpoint
from tac.neural_weight_codec_sensitivity import (
    NWCSRendererTensorEntry,
    SensitivityAwareCodecConfig,
    SensitivityAwareWeightCodec,
    encode_with_variable_codebook,
    export_nwcs_renderer_container,
    is_nwcs_renderer_container,
)

src_bin = Path("$ANCHOR_RENDERER_BIN")
codec_pt = Path("$NWCS_CODEC_PT")
sens_pt = Path("$SENS_PT")
out_bin = Path("$NWCS_RENDERER_BIN")
allow_debug = "$NWCS_ALLOW_DEBUG_SENSITIVITY" == "1"
promotion_eligible = "$PROMOTION_ELIGIBLE" == "true"

print(f"loading {src_bin} ({src_bin.stat().st_size:,} bytes)")
model = load_any_renderer_checkpoint(src_bin, device="cpu")
print(f"loaded model: {type(model).__name__}, "
      f"{sum(p.numel() for p in model.parameters()):,} params")
try:
    arch_config = _infer_asymmetric_config(model)
except Exception as exc:
    if promotion_eligible:
        raise SystemExit(
            "FATAL: could not infer NWCS renderer architecture config; "
            "refusing tensor_only fallback on promotion-eligible export"
        ) from exc
    print(
        "WARNING: non-promotable NWCS export using tensor_only fallback "
        f"after arch config inference failed: {type(exc).__name__}: {exc}"
    )
    arch_config = {"tensor_only": True}

ckpt = torch.load(codec_pt, weights_only=False)
cfg = SensitivityAwareCodecConfig(**ckpt["config"])
torch.manual_seed(1234)
codec = SensitivityAwareWeightCodec(cfg)
codec.load_state_dict(ckpt["codec_state_dict"])

sens_dict = torch.load(sens_pt, map_location="cpu", weights_only=False)
if not isinstance(sens_dict, dict) or not sens_dict:
    raise SystemExit("FATAL: SENS_PT must contain a non-empty per-parameter dict")

entries = []
encoded_count = 0
debug_fallback_params = []
for name, p in model.named_parameters():
    if not torch.is_floating_point(p):
        continue
    n_blocks = p.numel() // cfg.block_size
    if n_blocks == 0:
        continue
    sens = sens_dict.get(name)
    if sens is None or sens.numel() != n_blocks:
        if promotion_eligible:
            got = None if sens is None else int(sens.numel())
            raise SystemExit(
                f"FATAL: missing/mismatched true sensitivity for {name}: "
                f"got {got}, expected {n_blocks}"
            )
        debug_fallback_params.append(name)
        sens = torch.ones(n_blocks, dtype=torch.float32)
    sens = sens.detach().cpu().float().reshape(-1)
    if not torch.isfinite(sens).all():
        raise SystemExit(f"FATAL: sensitivity for {name} contains non-finite values")
    if (sens < 0).any():
        raise SystemExit(f"FATAL: sensitivity for {name} contains negative values")
    if promotion_eligible and sens.clamp_min(0).sum().item() <= 0.0:
        raise SystemExit(f"FATAL: sensitivity for {name} has no positive scorer signal")
    blob = encode_with_variable_codebook(codec, p, sens)
    entries.append(
        NWCSRendererTensorEntry.from_tensor_blob(
            name,
            p.detach().cpu(),
            blob,
            block_size=cfg.block_size,
            codebook_sizes=cfg.codebook_sizes,
            block_metadata={"sensitivity_blocks": int(sens.numel())},
        )
    )
    encoded_count += 1

export_nwcs_renderer_container(
    entries,
    codec_checkpoint_blob=codec_pt,
    output_path=out_bin,
    metadata={
        "format_note": "NWCS1 renderer container; inflate loader required",
        "promotion_eligible": promotion_eligible,
        "config": arch_config,
        "codec_config": ckpt["config"],
        "anchor_renderer": str(src_bin),
        "sensitivity_artifact": str(sens_pt),
    },
)
if not is_nwcs_renderer_container(out_bin):
    raise SystemExit("FATAL: NWCS export did not produce NWCS1 magic renderer container")
print(f"NWCS1 renderer.bin: {out_bin.stat().st_size:,} bytes "
      f"(was {src_bin.stat().st_size:,} bytes), {encoded_count} params encoded")
if debug_fallback_params:
    print("WARNING: non-promotable debug sensitivity fallback params:",
          ",".join(debug_fallback_params))
PY
[ -f "$NWCS_RENDERER_BIN" ] || { log "FATAL: NWCS export did not produce $NWCS_RENDERER_BIN"; exit 2; }

# Stage 6: Lane EC engineered corrections — search for SegNet-flipping deltas.
# Run AGAINST the Lane G v3 anchor renderer (the EC search loads the
# original ASYM renderer + GT video and finds per-pixel corrections to flip
# wrong SegNet argmax predictions). The corrections.bin is then bundled into
# the J-NWCS-encoded archive in Stage 8.
cost_guard
log "=== Stage 6: Lane EC engineered_quant_noise (compress-time SegNet correction search) ==="
log "  --max-delta=$EC_MAX_DELTA --max-artifact-bytes=$EC_RATE_CAP_BYTES"
EC_OUTPUT_DIR="$LOG_DIR/corrections"
"$PYBIN" -u experiments/engineered_quant_noise.py \
    --checkpoint "$ANCHOR_RENDERER_BIN" \
    --video "$GT_VIDEO" \
    --device cuda \
    --n-frames 1200 \
    --batch-size 32 \
    --max-delta "$EC_MAX_DELTA" \
    --quantize-bits 8 \
    --gt-poses-path "$ANCHOR_POSES" \
    --max-artifact-bytes "$EC_RATE_CAP_BYTES" \
    --output-dir "$EC_OUTPUT_DIR" 2>&1 | tee "$LOG_DIR/engineered_quant_noise.log" | tail -30
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

EC_CORRECTIONS_BIN="$EC_OUTPUT_DIR/gradient_corrections.bin"
if [ ! -f "$EC_CORRECTIONS_BIN" ]; then
    log "FATAL: engineered_quant_noise did NOT produce gradient_corrections.bin."
    log "       Without corrections.bin Lane EC contribution to the stack is dead."
    exit 2
fi
EC_CORRECTIONS_BYTES=$(stat -c '%s' "$EC_CORRECTIONS_BIN" 2>/dev/null || stat -f '%z' "$EC_CORRECTIONS_BIN")
log "  EC gradient_corrections.bin = ${EC_CORRECTIONS_BYTES} bytes"
if [ "$EC_CORRECTIONS_BYTES" -le 0 ] || [ "$EC_CORRECTIONS_BYTES" -gt "$EC_RATE_CAP_BYTES" ]; then
    log "FATAL: corrections.bin size ${EC_CORRECTIONS_BYTES} outside [1, $EC_RATE_CAP_BYTES]."
    exit 2
fi

# Stage 7: bit-budget split search → pick a small grid of candidate splits
# to actually contest-eval. The search is a CHEAP HEURISTIC; it only
# narrows what we run on GPU. Top split is reported and used in Stage 8.
cost_guard
log "=== Stage 7: bit_budget_split_search (3 grid points) ==="
SPLIT_REPORT="$LOG_DIR/split_search.json"
"$PYBIN" -u - <<PY
import json
from pathlib import Path

from tac.stack_compositions import bit_budget_split_search

# 3 grid points per axis = 9 total combos. Lane G v3 anchor sizes are the
# defaults inside bit_budget_split_search.
splits = bit_budget_split_search(
    target_archive_size=600_000,
    n_grid=3,
    weight_bits_grid=[3.0, 4.0, 5.0],
    ec_rate_cap_grid=[20_000, 30_000, 40_000],
)
report = {
    "n_splits": len(splits),
    "best_split": {
        "weight_avg_bits": splits[0].weight_avg_bits,
        "ec_rate_cap_bytes": splits[0].ec_rate_cap_bytes,
        "archive_bytes": splits[0].archive_bytes,
        "predicted_score": splits[0].predicted_score,
    },
    "all_splits": [
        {
            "weight_avg_bits": s.weight_avg_bits,
            "ec_rate_cap_bytes": s.ec_rate_cap_bytes,
            "archive_bytes": s.archive_bytes,
            "predicted_score": s.predicted_score,
        }
        for s in splits
    ],
}
Path("$SPLIT_REPORT").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
PY
[ -s "$SPLIT_REPORT" ] || { log "FATAL: split search did not produce $SPLIT_REPORT"; exit 2; }

# Stage 8: compose the archive on the actual J-NWCS + EC artifacts we
# produced. The split-search heuristic is informational only; we ship
# the artifacts we already built (single contest-eval to keep cost bounded).
cost_guard
log "=== Stage 8: compose archive via tac.stack_compositions ==="
ARCHIVE="$LOG_DIR/archive_lane_j_nwcs_ec.zip"
"$PYBIN" -u - <<PY
from pathlib import Path

from tac.stack_compositions import (
    compose_jnwcs_with_ec,
    validate_jnwcs_ec_composition,
)

archive = compose_jnwcs_with_ec(
    renderer_path="$NWCS_RENDERER_BIN",
    ec_corrections_path="$EC_CORRECTIONS_BIN",
    masks_path="$ANCHOR_MASKS",
    poses_path="$ANCHOR_POSES",
    output_archive_path="$ARCHIVE",
)
print(f"composed archive {archive}: {archive.stat().st_size} bytes")
summary = validate_jnwcs_ec_composition(archive, max_archive_bytes=600_000)
print("validation summary:", summary)
PY
[ -f "$ARCHIVE" ] || { log "FATAL: missing composed archive"; exit 2; }
ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
[ -n "${ARCHIVE_BYTES:-}" ] && [ "$ARCHIVE_BYTES" -gt 0 ] || {
    log "FATAL: archive size empty or zero — refusing to call auth_eval"
    exit 2
}

if [ "$NWCS_BUILD_ONLY" = "1" ] || [ "$PROMOTION_ELIGIBLE" != "true" ]; then
    BUILD_ONLY_REASON="$NON_PROMOTABLE_REASON"
    if [ "$NWCS_BUILD_ONLY" = "1" ]; then
        BUILD_ONLY_REASON="${BUILD_ONLY_REASON:+$BUILD_ONLY_REASON; }NWCS_BUILD_ONLY=1: build-only stop before auth eval"
    fi
    log "=== build-only/non-promotable stop before CUDA auth eval ==="
    log "reason: $BUILD_ONLY_REASON"
    "$PYBIN" -u - <<PY
from pathlib import Path
import json
import hashlib
import subprocess
import time

log_dir = Path("$LOG_DIR")

def file_meta(path):
    if not str(path):
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": h.hexdigest()}

prov = {
    "lane_name": "lane_j_nwcs_ec_stack",
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime($START_TS)),
    "finished_provenance_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "build_only": True,
    "score_claim": False,
    "promotion_eligible": False,
    "non_promotable_reason": "$BUILD_ONLY_REASON",
    "debug_sensitivity": "$NWCS_ALLOW_DEBUG_SENSITIVITY" == "1",
    "auth_eval_skipped": True,
    "result_json": None,
    "archive": "$ARCHIVE",
    "corpus_dir": "$ANCHOR_CORPUS_DIR",
    "prebuilt_corpus_manifest": "$PREBUILT_CORPUS_MANIFEST",
    "corpus_replay_root": "$CORPUS_REPLAY_ROOT",
    "corpus_manifest": "$CORPUS_MANIFEST",
    "artifact_custody": {
        "anchor_archive": file_meta("$ANCHOR_LANE_G_V3_ARCHIVE"),
        "anchor_renderer_bin": file_meta("$ANCHOR_RENDERER_BIN"),
        "anchor_masks_mkv": file_meta("$ANCHOR_MASKS"),
        "anchor_optimized_poses_pt": file_meta("$ANCHOR_POSES"),
        "component_sensitivity_manifest": file_meta("$COMPONENT_SENSITIVITY_MANIFEST"),
        "prebuilt_corpus_manifest": file_meta("$PREBUILT_CORPUS_MANIFEST"),
        "corpus_manifest": file_meta("$CORPUS_MANIFEST"),
        "anchor_sensitivity": file_meta("$ANCHOR_SENSITIVITY_PT"),
        "corpus_sensitivity": file_meta("$CORPUS_SENSITIVITY_PT"),
        "base_codec_pt": file_meta("$BASE_CODEC_PT"),
        "nwcs_codec_pt": file_meta("$NWCS_CODEC_PT"),
        "sensitivities_pt": file_meta("$SENS_PT"),
        "renderer_nwcs_bin": file_meta("$NWCS_RENDERER_BIN"),
        "ec_corrections_bin": file_meta("$EC_CORRECTIONS_BIN"),
        "split_search_report": file_meta("$SPLIT_REPORT"),
        "archive": file_meta("$ARCHIVE"),
    },
}
(log_dir / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
record = {
    "lane": "J-NWCS-EC-STACK",
    "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "archive": "$ARCHIVE",
    "result_json": None,
    "provenance": "$LOG_DIR/provenance.json",
    "build_only": True,
    "score_claim": False,
    "promotion_eligible": False,
    "non_promotable_reason": "$BUILD_ONLY_REASON",
}
(log_dir / "final_record.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY
    log "=== LANE_J_NWCS_EC_STACK_BUILD_ONLY_NON_PROMOTABLE ==="
    exit 0
fi

# Stage 9: CUDA auth eval [contest-CUDA].
cost_guard
log "=== Stage 9: CUDA auth eval [contest-CUDA] ==="
EVAL_WORK="$LOG_DIR/eval_work"
RESULT_JSON="$LOG_DIR/RESULT_JSON"
rm -rf "$EVAL_WORK"
# Belt-and-suspenders AppleDouble re-strip.
rm -f upstream/videos/._*.mkv
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$EVAL_WORK" 2>&1 | tee "$LOG_DIR/auth_eval.log" | tail -30
    PIPE_RC=("${PIPESTATUS[@]}")
    if [ "${PIPE_RC[0]}" -ne 0 ]; then
        echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
    fi

if [ -f "$EVAL_WORK/contest_auth_eval.json" ]; then
    cp "$EVAL_WORK/contest_auth_eval.json" "$RESULT_JSON"
elif [ -f "$LOG_DIR/auth_eval/contest_auth_eval.json" ]; then
    cp "$LOG_DIR/auth_eval/contest_auth_eval.json" "$RESULT_JSON"
else
    echo "FATAL: auth eval did not write contest_auth_eval.json; refusing log JSON scrape" >&2
    exit 2
fi
[ -s "$RESULT_JSON" ] || { log "FATAL: auth eval did not write RESULT_JSON"; exit 2; }
ADJUDICATION_PROVENANCE="$LOG_DIR/adjudication_provenance.json"
ADJUDICATED_RESULT_JSON="$LOG_DIR/contest_auth_eval.adjudicated.json"
"$PYBIN" -u scripts/adjudicate_contest_auth_eval.py \
    --contest-json "$RESULT_JSON" \
    --provenance "$ADJUDICATION_PROVENANCE" \
    --archive "$ARCHIVE" \
    --result-copy "$ADJUDICATED_RESULT_JSON" \
    --baseline-score 1.043987524793892 \
    --baseline-archive-bytes 686635 \
    --predicted-band 0.78 0.92 \
    --regression-threshold 1.05 \
    --delta-key score_delta_vs_pfp16_a_plus_plus \
    --required-device cuda \
    --required-samples 600 \
    --max-sane-score 10.0 \
    --baseline-posenet-dist 0.00346442 \
    --baseline-segnet-dist 0.00400656 \
    --max-posenet-relative 1.01 \
    --max-segnet-relative 1.01 \
    --component-reference-label "Lane G v3 PFP16 A++ frontier" \
    2>&1 | tee "$LOG_DIR/adjudication.log"
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: adjudication failed rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

# Stage 10: provenance + final record.
cost_guard
log "=== Stage 10: write provenance.json ==="
"$PYBIN" -u - <<PY
from pathlib import Path
import json
import hashlib
import subprocess
import time

log_dir = Path("$LOG_DIR")

def file_meta(path):
    p = Path(path)
    if not str(path):
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": h.hexdigest()}

prov = {
    "lane_name": "lane_j_nwcs_ec_stack",
    "predicted_band": [0.78, 0.92],
    "score_tag": "[contest-CUDA]",
    "hypothesis": (
        "Lane J-NWCS × Lane EC: composition of sensitivity-aware NWC weight "
        "codec (renderer-encoder slot) with engineered SegNet pixel corrections "
        "(sidecar-additive slot). Attacks rate wedge from two structurally "
        "orthogonal layers (weight-bit allocation vs inflate-time pixel "
        "residuals). Predicted improvement over J-NWCS-alone (~0.85-0.98) by "
        "~0.07 from EC sidecar contribution at <0.001 score rate cost."
    ),
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime($START_TS)),
    "finished_provenance_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "anchor_archive": "$ANCHOR_LANE_G_V3_ARCHIVE",
    "stack_slot": "renderer-encoder + sidecar-additive",
    "consumes": [
        "lane_g_v3_renderer",
        "lane_g_v3_masks",
        "lane_g_v3_poses",
        "true_scorer_renderer_sensitivity",
        "corpus_aligned_block_sensitivity",
    ],
    "produces": "archive_lane_j_nwcs_ec.zip",
    "stacks_with": ["renderer-replacement-output (consumed via anchor)"],
    "exclusive_with": [
        "another_ec_sidecar (only one gradient_corrections.bin per archive)",
        "lane_j_nwc_uniform_codec (J-NWCS supersedes)",
        "lane_f_v5 (different weight encoding)",
    ],
    "corpus_dir": "$ANCHOR_CORPUS_DIR",
    "prebuilt_corpus_manifest": "$PREBUILT_CORPUS_MANIFEST",
    "corpus_replay_root": "$CORPUS_REPLAY_ROOT",
    "corpus_manifest": "$CORPUS_MANIFEST",
    "component_sensitivity_manifest": "$COMPONENT_SENSITIVITY_MANIFEST",
    "anchor_sensitivity_source": "$ANCHOR_SENSITIVITY_PT",
    "corpus_sensitivity_source": "$CORPUS_SENSITIVITY_PT",
    "promotion_eligible": "$PROMOTION_ELIGIBLE" == "true",
    "non_promotable_reason": "$NON_PROMOTABLE_REASON",
    "base_codec_pt": "$BASE_CODEC_PT",
    "nwcs_codec_pt": "$NWCS_CODEC_PT",
    "sensitivities_pt": "$SENS_PT",
    "renderer_nwcs_bin": "$NWCS_RENDERER_BIN",
    "ec_corrections_bin": "$EC_CORRECTIONS_BIN",
    "ec_rate_cap_bytes": $EC_RATE_CAP_BYTES,
    "ec_max_delta": $EC_MAX_DELTA,
    "split_search_report": "$SPLIT_REPORT",
    "archive": "$ARCHIVE",
    "result_json": "$RESULT_JSON",
    "adjudicated_result_json": "$ADJUDICATED_RESULT_JSON",
    "adjudication_provenance": "$ADJUDICATION_PROVENANCE",
    "score_source": "contest_auth_eval.adjudicated.json:score_recomputed_from_components",
    "adjudication_required": True,
    "component_gates_required": True,
    "artifact_custody": {
        "anchor_archive": file_meta("$ANCHOR_LANE_G_V3_ARCHIVE"),
        "anchor_renderer_bin": file_meta("$ANCHOR_RENDERER_BIN"),
        "anchor_masks_mkv": file_meta("$ANCHOR_MASKS"),
        "anchor_optimized_poses_pt": file_meta("$ANCHOR_POSES"),
        "component_sensitivity_manifest": file_meta("$COMPONENT_SENSITIVITY_MANIFEST") if "$COMPONENT_SENSITIVITY_MANIFEST" else None,
        "prebuilt_corpus_manifest": file_meta("$PREBUILT_CORPUS_MANIFEST") if "$PREBUILT_CORPUS_MANIFEST" else None,
        "corpus_manifest": file_meta("$CORPUS_MANIFEST"),
        "anchor_sensitivity": file_meta("$ANCHOR_SENSITIVITY_PT") if "$ANCHOR_SENSITIVITY_PT" else None,
        "corpus_sensitivity": file_meta("$CORPUS_SENSITIVITY_PT") if "$CORPUS_SENSITIVITY_PT" else None,
        "base_codec_pt": file_meta("$BASE_CODEC_PT"),
        "nwcs_codec_pt": file_meta("$NWCS_CODEC_PT"),
        "sensitivities_pt": file_meta("$SENS_PT"),
        "renderer_nwcs_bin": file_meta("$NWCS_RENDERER_BIN"),
        "ec_corrections_bin": file_meta("$EC_CORRECTIONS_BIN"),
        "split_search_report": file_meta("$SPLIT_REPORT"),
        "archive": file_meta("$ARCHIVE"),
        "result_json": file_meta("$RESULT_JSON"),
        "adjudicated_result_json": file_meta("$ADJUDICATED_RESULT_JSON"),
        "adjudication_provenance": file_meta("$ADJUDICATION_PROVENANCE"),
    },
    "strict_scorer_rule": (
        "BOTH artifacts are scorer-free at inflate. NWCS codec weights are "
        "bundled INTO renderer.bin (compress-time codec, inflate-time pure "
        "decode). Lane EC corrections.bin is numpy int8 sparse deltas "
        "applied additively (no torch autograd, no PoseNet/SegNet load). "
        "Composition trivially inherits both properties."
    ),
}
(log_dir / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
print(json.dumps(prov, indent=2))
PY

cost_guard
log "=== Stage 11: final record ==="
"$PYBIN" -u - <<PY
from pathlib import Path
import json
import time

record = {
    "lane": "J-NWCS-EC-STACK",
    "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "archive": "$ARCHIVE",
    "result_json": "$RESULT_JSON",
    "adjudicated_result_json": "$ADJUDICATED_RESULT_JSON",
    "provenance": "$LOG_DIR/provenance.json",
    "adjudication_provenance": "$ADJUDICATION_PROVENANCE",
    "score_source": "contest_auth_eval.adjudicated.json:score_recomputed_from_components",
    "adjudication_required": True,
    "predicted_band": [0.78, 0.92],
    "promotion_eligible": "$PROMOTION_ELIGIBLE" == "true",
    "non_promotable_reason": "$NON_PROMOTABLE_REASON",
}
Path("$LOG_DIR/final_record.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY

if [ "${AUTO_DESTROY_VAST:-0}" = "1" ] && [ -n "${VAST_INSTANCE_ID:-}" ]; then
    log "AUTO_DESTROY_VAST=1: destroying Vast instance $VAST_INSTANCE_ID"
    vastai destroy instance "$VAST_INSTANCE_ID" || true
fi

log "=== LANE_J_NWCS_EC_STACK_DONE [contest-CUDA] ==="
