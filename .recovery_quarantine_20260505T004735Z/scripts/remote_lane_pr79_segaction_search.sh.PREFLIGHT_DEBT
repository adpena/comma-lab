#!/usr/bin/env bash
# NO_NVDEC_NEEDED — PR79 SegAction search clones a public repo and runs Python
# search/eval. Video decode (if any) is via the public submission's own
# inflate/eval flow on already-decoded artifacts; this driver does no
# DALI/NVDEC video pipeline work itself.
set -euo pipefail

source ./env.sh 2>/dev/null || true

LABEL="${LANE_LABEL:-pr79_segaction_search_h100_20260503}"
OUT_DIR="${WORKSPACE:-$PWD}/results/${LABEL}"
PUBLIC_REPO="${PR79_PUBLIC_REPO:-https://github.com/EthanYangTW/comma_video_compression_challenge.git}"
PUBLIC_BRANCH="${PR79_PUBLIC_BRANCH:-dev/qpose14-r55-segactions-minp-v2}"
SUBDIR="submissions/qpose14_r55_segactions_minp"
DEVICE="${PR79_ACTION_DEVICE:-cuda}"
BATCH_SIZE="${PR79_ACTION_BATCH_SIZE:-24}"
TOP_TILES="${PR79_ACTION_TOP_TILES:-4}"
PASSES="${PR79_ACTION_PASSES:-3}"
MAX_ACTIONS="${PR79_ACTION_MAX_ACTIONS:-1800}"
PROBE_MIN_GAIN="${PR79_ACTION_PROBE_MIN_GAIN:-0.000002}"
SUBSET_PASSES="${PR79_ACTION_SUBSET_PASSES:-3}"
SUBSET_MIN_GAIN="${PR79_ACTION_SUBSET_MIN_GAIN:-0.0000002}"
TILE="${PR79_ACTION_TILE:-32}"
POSE_FLAGS_RAW="${PR79_ACTION_POSE_FLAGS:---pose-gate --pose-check}"
POSE_ARGS=()
if [ -n "$POSE_FLAGS_RAW" ]; then
  read -r -a POSE_ARGS <<< "$POSE_FLAGS_RAW"
fi

mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/run.log") 2>&1

echo "[pr79-segaction] label=$LABEL"
echo "[pr79-segaction] repo=$PUBLIC_REPO branch=$PUBLIC_BRANCH"
echo "[pr79-segaction] device=$DEVICE batch=$BATCH_SIZE top_tiles=$TOP_TILES passes=$PASSES max_actions=$MAX_ACTIONS"
echo "[pr79-segaction] probe_pose_flags=${POSE_FLAGS_RAW:-<none>}"
date -u +"[pr79-segaction] start_utc=%Y-%m-%dT%H:%M:%SZ"
nvidia-smi || true
python3 - <<'PY'
import importlib.util
import subprocess
import sys

missing = [name for name in ("brotli",) if importlib.util.find_spec(name) is None]
if missing:
    print(f"[pr79-segaction] installing missing python deps: {missing}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
else:
    print("[pr79-segaction] python deps present", flush=True)
PY

WORK="${WORKSPACE:-$PWD}/pr79_action_search_work"
UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-${WORKSPACE:-$PWD}/upstream}"
rm -rf "$WORK"
git clone --depth 1 --branch "$PUBLIC_BRANCH" "$PUBLIC_REPO" "$WORK"
if [ ! -d "$UPSTREAM_DIR/models" ] || [ ! -d "$UPSTREAM_DIR/videos" ]; then
  echo "[pr79-segaction] FATAL: upstream assets missing at $UPSTREAM_DIR" >&2
  exit 31
fi
rm -rf "$WORK/models" "$WORK/videos"
cp -a "$UPSTREAM_DIR/models" "$WORK/models"
cp -a "$UPSTREAM_DIR/videos" "$WORK/videos"
cp -f "$UPSTREAM_DIR/public_test_video_names.txt" "$WORK/public_test_video_names.txt"
cp -f "$UPSTREAM_DIR/public_test_segments.txt" "$WORK/public_test_segments.txt"
cd "$WORK/$SUBDIR"
python3 - <<'PY'
from pathlib import Path

for name in ("probe_more_seg_actions_minp.py", "optimize_action_subset.py"):
    path = Path(name)
    text = path.read_text()
    old = (
        '    ds_cls = DaliVideoDataset if device.type == "cuda" else AVVideoDataset\n'
        '    ds = ds_cls(files, data_dir=ROOT / "videos", batch_size=batch_size, device=device)'
    )
    new = (
        '    ds = AVVideoDataset(files, data_dir=ROOT / "videos", batch_size=batch_size, '
        'device=torch.device("cpu"))  # Modal: CPU/PyAV decode, CUDA model inference'
    )
    if old not in text:
        raise SystemExit(f"expected dataset construction block not found in {name}")
    text = text.replace(old, new)
    unpack_marker = "    raw = brotli.decompress(actions_br_data)\n"
    if name == "optimize_action_subset.py":
        unpack_marker = "    raw = brotli.decompress(actions_br)\n"
    tg1_patch = (
        unpack_marker
        + '    if raw.startswith(b"TG1"):\n'
        + "        raw = raw[5:]\n"
    )
    if unpack_marker in text and tg1_patch not in text:
        text = text.replace(unpack_marker, tg1_patch)
    path.write_text(text)
print("[pr79-segaction] patched public scripts for CPU/PyAV decode with CUDA inference")
PY

bash compress.sh
python3 - <<'PY'
import hashlib
from pathlib import Path
p = Path("archive.zip")
print("[pr79-segaction] base_archive_bytes", p.stat().st_size)
print("[pr79-segaction] base_archive_sha256", hashlib.sha256(p.read_bytes()).hexdigest())
PY

python3 -u probe_more_seg_actions_minp.py \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --tile "$TILE" \
  --top-tiles "$TOP_TILES" \
  --passes "$PASSES" \
  --max-actions "$MAX_ACTIONS" \
  --min-gain "$PROBE_MIN_GAIN" \
  "${POSE_ARGS[@]}" \
  --progress-every 10

test -f seg_tile_actions_probe.br
python3 - "$TILE" <<'PY'
import sys
from pathlib import Path

import brotli

tile = int(sys.argv[1])
path = Path("seg_tile_actions_probe.br")
raw = brotli.decompress(path.read_bytes())
if tile != 32 and not raw.startswith(b"TG1"):
    if tile <= 0 or 384 % tile != 0 or 512 % tile != 0:
        raise SystemExit(f"invalid PR79_ACTION_TILE={tile}")
    path.write_bytes(brotli.compress(b"TG1" + tile.to_bytes(2, "little") + raw, quality=11))
    print(f"[pr79-segaction] wrapped probe actions with charged TG1 tile_size={tile}", flush=True)
PY

python3 - "$OUT_DIR/probe_archive.zip" <<'PY'
import hashlib
import json
import zipfile
from pathlib import Path

MASK_BYTES = 219_472
MODEL_BYTES = 55_756
POSE_BYTES = 898
out = Path(__import__("sys").argv[1])
with zipfile.ZipFile("archive.zip", "r") as zf:
    payload = zf.read("p")
actions = Path("seg_tile_actions_probe.br").read_bytes()
mask = payload[:MASK_BYTES]
model = payload[MASK_BYTES:MASK_BYTES + MODEL_BYTES]
pose = payload[-POSE_BYTES:]
candidate_payload = mask + model + actions + pose
info = zipfile.ZipInfo("p", (1980, 1, 1, 0, 0, 0))
info.compress_type = zipfile.ZIP_STORED
info.external_attr = 0o644 << 16
info.create_system = 3
with zipfile.ZipFile(out, "w") as zf:
    zf.writestr(info, candidate_payload)
manifest = {
    "schema": "pr79_segaction_search_probe_archive_v1",
    "source_archive_bytes": Path("archive.zip").stat().st_size,
    "source_archive_sha256": hashlib.sha256(Path("archive.zip").read_bytes()).hexdigest(),
    "candidate_archive": str(out),
    "candidate_archive_bytes": out.stat().st_size,
    "candidate_archive_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    "payload_bytes": len(candidate_payload),
    "actions_bytes": len(actions),
    "slice_contract": {
        "mask_bytes": MASK_BYTES,
        "model_bytes": MODEL_BYTES,
        "actions_bytes": len(actions),
        "pose_bytes": POSE_BYTES,
    },
    "score_claim": False,
    "evidence_grade": "remote_cuda_proxy_search_only_until_exact_t4_auth_eval",
}
Path("probe_archive_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, sort_keys=True))
PY
cp probe_archive_manifest.json "$OUT_DIR/probe_archive_manifest.json"
cp seg_tile_actions_probe.br "$OUT_DIR/seg_tile_actions_probe.br"

python3 -u optimize_action_subset.py \
  --archive archive.zip \
  --candidate-archive "$OUT_DIR/probe_archive.zip" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --tile "$TILE" \
  --passes "$SUBSET_PASSES" \
  --min-gain "$SUBSET_MIN_GAIN" \
  --out "$OUT_DIR/archive_optimized.zip"

python3 - "$OUT_DIR/archive_optimized.zip" "$TILE" <<'PY'
import hashlib
import sys
import zipfile
from pathlib import Path

import brotli

MASK_BYTES = 219_472
MODEL_BYTES = 55_756
POSE_BYTES = 898
archive = Path(sys.argv[1])
tile = int(sys.argv[2])
if tile == 32:
    raise SystemExit(0)
payload = zipfile.ZipFile(archive, "r").read("p")
mask = payload[:MASK_BYTES]
model = payload[MASK_BYTES:MASK_BYTES + MODEL_BYTES]
actions = payload[MASK_BYTES + MODEL_BYTES:-POSE_BYTES]
pose = payload[-POSE_BYTES:]
raw = brotli.decompress(actions)
if not raw.startswith(b"TG1"):
    actions = brotli.compress(b"TG1" + tile.to_bytes(2, "little") + raw, quality=11)
    payload = mask + model + actions + pose
    info = zipfile.ZipInfo("p", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(info, payload)
    print(
        "[pr79-segaction] wrapped optimized actions with charged TG1 "
        f"tile_size={tile} archive_bytes={archive.stat().st_size} "
        f"sha256={hashlib.sha256(archive.read_bytes()).hexdigest()}",
        flush=True,
    )
PY

python3 - "$OUT_DIR" <<'PY'
import hashlib
import json
from pathlib import Path
out_dir = Path(__import__("sys").argv[1])
summary = {
    "schema": "pr79_segaction_search_h100_summary_v1",
    "score_claim": False,
    "evidence_grade": "remote_cuda_proxy_search_only_until_exact_t4_auth_eval",
    "base_archive": {
        "bytes": Path("archive.zip").stat().st_size,
        "sha256": hashlib.sha256(Path("archive.zip").read_bytes()).hexdigest(),
    },
}
for name in ("probe_archive.zip", "archive_optimized.zip", "seg_tile_actions_probe.br"):
    path = out_dir / name
    if path.exists():
        summary[name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
summary["next_gate"] = (
    "Run robust-current selected/all-pair parity where practical, then claim "
    "a T4 exact eval lane and run archive.zip -> inflate.sh -> upstream/evaluate.py "
    "on the exact archive bytes if projected score can beat 0.31."
)
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY

date -u +"[pr79-segaction] finish_utc=%Y-%m-%dT%H:%M:%SZ"
