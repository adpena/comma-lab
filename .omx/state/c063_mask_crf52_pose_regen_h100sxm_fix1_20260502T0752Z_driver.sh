#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
cd "$WORKSPACE"
export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE:${PYTHONPATH:-}"
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

OUT="experiments/results/vast_live_harvest/c063_mask_crf52_pose_regen_h100sxm_fix1_20260502T0752Z"
OLD="experiments/results/vast_live_harvest/c063_mask_crf52_pose_regen_h100sxm_20260502"
RUNTIME="$OUT/runtime"
POSE_DIR="$OUT/pose"
mkdir -p "$RUNTIME" "$POSE_DIR"

log() { echo "[c063-crf52-pose-regen-fix1] $(date -u +%FT%TZ) $*" | tee -a "$OUT/driver.log"; }
sha256_file() { sha256sum "$1" | cut -d' ' -f1; }

log "copying runtime renderer/masks from failed pre-evidence run"
cp "$OLD/runtime/renderer.bin" "$RUNTIME/renderer.bin"
cp "$OLD/runtime/masks.mkv" "$RUNTIME/masks.mkv"

log "source sha preflight for patched optimizer/runtime"
cat > "$OUT/source_sha256s.expected" <<'SHA'
experiments/optimize_poses.py=28fbb9d84e23c15afce0a4fd5a6027df8b30adc75975e3d5d6969bfe08144884
src/tac/preflight.py=b04180c8de8e6f9054dd701e2a2cc4d524facf16e6c0619115a65faba400f96b
submissions/robust_current/inflate.sh=86449a1f52ac6b2be120d47287b8410f915dce7e562c69f480103f6e527c6017
submissions/robust_current/inflate_renderer.py=1bf64e9f055c88438c854d1e09f048c07c359177494da8e636079c62706b6472
submissions/robust_current/unpack_renderer_payload.py=cac8cde654f2d875d4567c18b77d573af91c29dbb0b05b7934dc7e019ae66f49
experiments/contest_auth_eval.py=8d9dd3e1e9f97245623c938aa9b28a41fdd3e41752208b16b218131906f7ef53
experiments/build_renderer_packed_payload_archive.py=abe35928dec4643b754b4fd9af022b131926bfc0f0b56eef6eac5fec8e87b691
experiments/repack_single_payload_brotli.py=ba2f262da50cd65649d7cfd783da50a11256ffbdc220baa055b3217137817008
scripts/remote_archive_only_eval.sh=8d1069de4f2426108beb44e519f83428608c1354ea2752b426cddd1f5ab815f8
SHA
"$PYBIN" - <<'PY'
from pathlib import Path
import hashlib

root = Path(".")
bad = []
for line in Path("experiments/results/vast_live_harvest/c063_mask_crf52_pose_regen_h100sxm_fix1_20260502T0752Z/source_sha256s.expected").read_text().splitlines():
    if not line.strip():
        continue
    rel, expected = line.split("=", 1)
    actual = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    if actual != expected:
        bad.append((rel, expected, actual))
if bad:
    for rel, expected, actual in bad:
        print(f"SHA_MISMATCH {rel} expected={expected} actual={actual}")
    raise SystemExit(10)
print("SOURCE_SHA_PREFLIGHT_OK")
PY

log "optimizer loader smoke against actual QZS3 renderer"
"$PYBIN" - <<'PY' | tee "$OUT/qzs3_loader_smoke.json"
from pathlib import Path
import json
import torch
from experiments.optimize_poses import load_renderer
from tac.preflight import preflight_check

renderer = Path("experiments/results/vast_live_harvest/c063_mask_crf52_pose_regen_h100sxm_fix1_20260502T0752Z/runtime/renderer.bin")
preflight_check(renderer_path=renderer, verbose=False)
model = load_renderer(str(renderer), torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print(json.dumps({
    "magic": renderer.read_bytes()[:4].decode("ascii", "replace"),
    "pose_dim": int(getattr(model, "pose_dim", -1)),
    "q_faithful": bool(getattr(model, "q_faithful", False)),
    "n_params": int(sum(p.numel() for p in model.parameters())),
    "torch_cuda_available": bool(torch.cuda.is_available()),
}, sort_keys=True))
PY

MASK_PT="experiments/results/c063_mask_pose_regen_20260502/crf52_masks_repeat2_1200_uint8.pt"
GT_TARGETS="experiments/results/lane_a_landed/gt_pose_targets.pt"
GT_POSES="experiments/results/lane_g_v3_landed/iter_0/optimized_poses.pt"
log "running pose optimization against CRF52 decoded repeat2 masks"
"$PYBIN" -u experiments/optimize_poses.py \
  --checkpoint "$RUNTIME/renderer.bin" \
  --masks "$MASK_PT" \
  --gt-pose-targets "$GT_TARGETS" \
  --gt-poses-path "$GT_POSES" \
  --device cuda \
  --n-frames 1200 \
  --steps 80 \
  --batch-pairs 100 \
  --lr 0.01 \
  --early-stop-patience 30 \
  --output-dir "$POSE_DIR" \
  2>&1 | tee "$OUT/pose_regen.log"

test -f "$POSE_DIR/optimized_poses.bin"
cp "$POSE_DIR/optimized_poses.bin" "$RUNTIME/optimized_poses.bin"

log "building deterministic runtime_members.zip"
"$PYBIN" - <<'PY'
from pathlib import Path
import hashlib
import json
import zipfile

out = Path("experiments/results/vast_live_harvest/c063_mask_crf52_pose_regen_h100sxm_fix1_20260502T0752Z")
runtime = out / "runtime"
members = ("renderer.bin", "masks.mkv", "optimized_poses.bin")
zip_path = out / "runtime_members.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for name in members:
        data = (runtime / name).read_bytes()
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 3
        info.external_attr = 0o644 << 16
        zf.writestr(info, data)
manifest = {
    "members": {
        name: {
            "bytes": (runtime / name).stat().st_size,
            "sha256": hashlib.sha256((runtime / name).read_bytes()).hexdigest(),
        }
        for name in members
    },
    "runtime_zip_bytes": zip_path.stat().st_size,
    "runtime_zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
}
(out / "runtime_members_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, sort_keys=True))
PY

log "building public-pr64-mask-first packed archive with QP1 pose codec"
"$PYBIN" -u experiments/build_renderer_packed_payload_archive.py \
  --source-archive "$OUT/runtime_members.zip" \
  --output-dir "$OUT/build" \
  --pose-codec pose_qp1_v1 \
  --payload-member-name p \
  --payload-format public_pr64_mask_first_len_table \
  --brotli-quality 11 \
  2>&1 | tee "$OUT/build_stdout.json"

log "lossless repack of single payload"
"$PYBIN" -u experiments/repack_single_payload_brotli.py \
  --source-archive "$OUT/build/archive.zip" \
  --output-archive "$OUT/archive.zip" \
  --manifest-json "$OUT/lossless_repack_manifest.json" \
  --member-name p \
  --quality 11 \
  --mode 2 \
  --lgwin 18 \
  --lgblock 0 \
  --allow-non-improvement \
  2>&1 | tee "$OUT/repack_stdout.json"

log "exact CUDA diagnostic eval"
export ARCHIVE_PATH="$WORKSPACE/$OUT/archive.zip"
export ARCHIVE_LABEL="archive_eval_c063_mask_crf52_pose_regen_fix1_h100_20260502"
export LOG_DIR="$WORKSPACE/$OUT/exact_eval"
export PREDICTED_LOW="0.18"
export PREDICTED_HIGH="2.35"
export CONTROLLED_BASELINE="C-063 T4 frontier plus CRF52 mask stream with regenerated optimized_poses.bin; H100 diagnostic, T4 required for promotion"
export REQUIRED_SOURCE_SHA256S="$(cat "$OUT/source_sha256s.expected")"
bash scripts/remote_archive_only_eval.sh

log "done"
