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
PUBLIC_COMMIT="${PR79_PUBLIC_COMMIT:-9c93af0a5bf55cc8a03716e0f7b9babf187ad2a1}"
BROTLI_PACKAGE="${PR79_BROTLI_PACKAGE:-brotli==1.1.0}"
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

# ── heartbeat (per check_remote_lane_scripts_have_heartbeat preflight) ──
HEARTBEAT="$OUT_DIR/heartbeat.log"
(
    while true; do
        echo "$(date -u +%FT%TZ) pr79_segaction running" >> "$HEARTBEAT"
        sleep 60
    done
) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

echo "[pr79-segaction] label=$LABEL"
echo "[pr79-segaction] repo=$PUBLIC_REPO branch=$PUBLIC_BRANCH"
echo "[pr79-segaction] device=$DEVICE batch=$BATCH_SIZE top_tiles=$TOP_TILES passes=$PASSES max_actions=$MAX_ACTIONS"
echo "[pr79-segaction] probe_pose_flags=${POSE_FLAGS_RAW:-<none>}"
date -u +"[pr79-segaction] start_utc=%Y-%m-%dT%H:%M:%SZ"
nvidia-smi || true
PACT_REPO="${PACT_REPO:-$PWD}"
if [ ! -d "$PACT_REPO/src/tac" ]; then
  echo "[pr79-segaction] FATAL: PACT_REPO does not point at this checkout: $PACT_REPO" >&2
  exit 33
fi
export PYTHONPATH="$PACT_REPO/src:${PYTHONPATH:-}"

# ── provenance.json (per feedback_canonical_remote_bootstraps + preflight Check L) ──
PR79_GIT_HASH="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
PR79_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
PR79_OUT_PROV="$OUT_DIR/provenance.json" \
PR79_LABEL="$LABEL" PR79_GIT_HASH="$PR79_GIT_HASH" PR79_GPU_NAME="$PR79_GPU_NAME" \
PR79_PUBLIC_REPO="$PUBLIC_REPO" PR79_PUBLIC_BRANCH="$PUBLIC_BRANCH" \
PR79_PUBLIC_COMMIT="$PUBLIC_COMMIT" PR79_BROTLI_PACKAGE="$BROTLI_PACKAGE" \
PR79_DEVICE="$DEVICE" PR79_BATCH_SIZE="$BATCH_SIZE" \
PR79_TOP_TILES="$TOP_TILES" PR79_PASSES="$PASSES" PR79_MAX_ACTIONS="$MAX_ACTIONS" \
PR79_POSE_FLAGS="${POSE_FLAGS_RAW:-}" \
python3 - <<'PY'
import json, os, time
prov = {
    "lane_id": "lane_pr79_segaction_search",
    "label": os.environ["PR79_LABEL"],
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_hash": os.environ["PR79_GIT_HASH"],
    "gpu_name": os.environ["PR79_GPU_NAME"],
    "public_repo": os.environ["PR79_PUBLIC_REPO"],
    "public_branch": os.environ["PR79_PUBLIC_BRANCH"],
    "public_commit": os.environ["PR79_PUBLIC_COMMIT"],
    "python_dependency_package": os.environ["PR79_BROTLI_PACKAGE"],
    "config": {
        "device": os.environ["PR79_DEVICE"],
        "batch_size": int(os.environ["PR79_BATCH_SIZE"]),
        "top_tiles": int(os.environ["PR79_TOP_TILES"]),
        "passes": int(os.environ["PR79_PASSES"]),
        "max_actions": int(os.environ["PR79_MAX_ACTIONS"]),
        "pose_flags": os.environ["PR79_POSE_FLAGS"],
    },
    # PR79 SegAction is a public-frontier replay; no private prediction band.
    "predicted_band_tag": "[predicted-band only — public-frontier replay; final score via upstream/evaluate.py]",
}
with open(os.environ["PR79_OUT_PROV"], "w") as f:
    json.dump(prov, f, indent=2, sort_keys=True)
print(f"[pr79-segaction] provenance written -> {os.environ['PR79_OUT_PROV']}")
PY
python3 - <<'PY'
import importlib.util
import os
import subprocess
import sys

package = os.environ.get("PR79_BROTLI_PACKAGE", "brotli==1.1.0")
missing = [name for name in ("brotli",) if importlib.util.find_spec(name) is None]
if missing:
    print(f"[pr79-segaction] installing pinned python dep: {package}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
else:
    print("[pr79-segaction] python deps present", flush=True)
PY

WORK="${WORKSPACE:-$PWD}/pr79_action_search_work"
UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-${WORKSPACE:-$PWD}/upstream}"
rm -rf "$WORK"
git clone --depth 1 --branch "$PUBLIC_BRANCH" "$PUBLIC_REPO" "$WORK"
cd "$WORK"
git fetch --depth 1 origin "$PUBLIC_COMMIT"
git checkout --detach "$PUBLIC_COMMIT"
CLONED_PUBLIC_COMMIT="$(git rev-parse HEAD)"
if [ "$CLONED_PUBLIC_COMMIT" != "$PUBLIC_COMMIT" ]; then
  echo "[pr79-segaction] FATAL: public commit mismatch expected=$PUBLIC_COMMIT actual=$CLONED_PUBLIC_COMMIT" >&2
  exit 32
fi
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


def replace_required(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"expected patch target not found: {label}")
    return text.replace(old, new, 1)


for name in ("probe_more_seg_actions_minp.py", "optimize_action_subset.py"):
    path = Path(name)
    text = path.read_text()
    import_line = (
        "from tac.pr79_segaction_payload import parse_pr79_archive, "
        "parse_pr79_payload_bytes, write_pr79_single_member_archive  # noqa: E402\n"
    )
    module_line = "from modules import PoseNet, SegNet, posenet_sd_path, segnet_sd_path  # noqa: E402\n"
    if import_line not in text:
        if module_line not in text:
            raise SystemExit(f"expected modules import line not found in {name}")
        text = text.replace(module_line, module_line + import_line)
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
    if name == "probe_more_seg_actions_minp.py":
        start = "def read_packed_archive(submission_dir: Path):\n"
        end = "\n\ndef decode_qp1_pose"
        if start not in text or end not in text:
            raise SystemExit(f"expected read_packed_archive block not found in {name}")
        replacement = (
            "def read_packed_archive(submission_dir: Path):\n"
            "    global ACTION_FREE_ARCHIVE_BYTES\n"
            "    parsed = parse_pr79_archive(submission_dir / \"archive.zip\")\n"
            "    ACTION_FREE_ARCHIVE_BYTES = parsed.action_free_archive_bytes()\n"
            "    return parsed.mask_br, parsed.model_br, parsed.actions_br, parsed.pose_br\n"
        )
        a = text.index(start)
        b = text.index(end, a)
        text = text[:a] + replacement + text[b:]
    if name == "optimize_action_subset.py":
        start = "def split_known_payload(payload: bytes):\n"
        end = "\n\ndef read_uvarint"
        if start not in text or end not in text:
            raise SystemExit(f"expected split_known_payload block not found in {name}")
        replacement = (
            "def split_known_payload(payload: bytes):\n"
            "    parsed = parse_pr79_payload_bytes(payload)\n"
            "    return parsed.mask_br, parsed.model_br, parsed.actions_br, parsed.pose_br\n"
        )
        a = text.index(start)
        b = text.index(end, a)
        text = text[:a] + replacement + text[b:]
        text = replace_required(
            text,
            "    mask_br, model_br, actions_br, pose_br, base1, base2 = load_base(args.archive, device, args.batch_size)\n",
            "    source_payload = parse_pr79_payload_bytes(read_payload(args.archive))\n"
            "    mask_br, model_br, actions_br, pose_br, base1, base2 = load_base(args.archive, device, args.batch_size)\n",
            "optimize source_payload parse",
        )
        text = replace_required(
            text,
            "    payload = mask_br + model_br + pack_best(current) + pose_br\n",
            "    payload = source_payload.replace_actions(pack_best(current))\n",
            "optimize current payload",
        )
        text = replace_required(
            text,
            "            trial_payload = mask_br + model_br + pack_best(trial_records) + pose_br\n",
            "            trial_payload = source_payload.replace_actions(pack_best(trial_records))\n",
            "optimize trial payload",
        )
        text = replace_required(
            text,
            "    final_payload = mask_br + model_br + final_actions + pose_br\n"
            "    with zipfile.ZipFile(args.out, \"w\", compression=zipfile.ZIP_STORED) as zf:\n"
            "        zf.writestr(\"p\", final_payload)\n",
            "    final_payload = source_payload.replace_actions(final_actions)\n"
            "    write_pr79_single_member_archive(args.out, final_payload)\n",
            "optimize final archive write",
        )
    path.write_text(text)
print("[pr79-segaction] patched public scripts for CPU/PyAV decode with CUDA inference")
PY
PATCH_DIFF_SHA256="$(git -C "$WORK" diff -- submissions/qpose14_r55_segactions_minp/probe_more_seg_actions_minp.py submissions/qpose14_r55_segactions_minp/optimize_action_subset.py | shasum -a 256 | cut -d ' ' -f 1)"
cat > "$OUT_DIR/public_repo_lock.json" <<EOF
{
  "public_repo": "$PUBLIC_REPO",
  "public_branch": "$PUBLIC_BRANCH",
  "public_commit": "$PUBLIC_COMMIT",
  "cloned_public_commit": "$CLONED_PUBLIC_COMMIT",
  "patch_diff_sha256": "$PATCH_DIFF_SHA256",
  "brotli_package": "$BROTLI_PACKAGE",
  "score_claim": false
}
EOF
echo "[pr79-segaction] public_repo_lock=$OUT_DIR/public_repo_lock.json patch_diff_sha256=$PATCH_DIFF_SHA256"

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
from pathlib import Path

from tac.pr79_segaction_payload import parse_pr79_archive, write_pr79_single_member_archive

out = Path(__import__("sys").argv[1])
parsed = parse_pr79_archive("archive.zip")
actions = Path("seg_tile_actions_probe.br").read_bytes()
candidate_payload = parsed.replace_actions(actions)
archive_meta = write_pr79_single_member_archive(out, candidate_payload)
manifest = {
    "schema": "pr79_segaction_search_probe_archive_v1",
    "source_archive_bytes": Path("archive.zip").stat().st_size,
    "source_archive_sha256": hashlib.sha256(Path("archive.zip").read_bytes()).hexdigest(),
    "candidate_archive": str(out),
    "candidate_archive_bytes": archive_meta["archive_bytes"],
    "candidate_archive_sha256": archive_meta["archive_sha256"],
    "source_payload": parsed.summary(),
    "candidate_payload": parse_pr79_archive(out).summary(),
    "actions_bytes": len(actions),
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
from pathlib import Path

import brotli

from tac.pr79_segaction_payload import parse_pr79_archive, write_pr79_single_member_archive

archive = Path(sys.argv[1])
tile = int(sys.argv[2])
if tile == 32:
    raise SystemExit(0)
parsed = parse_pr79_archive(archive)
raw = brotli.decompress(parsed.actions_br)
if not raw.startswith(b"TG1"):
    actions_br = brotli.compress(b"TG1" + tile.to_bytes(2, "little") + raw, quality=11)
    payload = parsed.replace_actions(actions_br)
    write_pr79_single_member_archive(archive, payload)
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
