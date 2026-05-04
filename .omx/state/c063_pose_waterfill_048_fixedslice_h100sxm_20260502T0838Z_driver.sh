#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
cd "$WORKSPACE"

export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE:${PYTHONPATH:-}"
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"
PYBIN="${PYBIN:-$WORKSPACE/.venv/bin/python}"
if ! "$PYBIN" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  PYBIN="/opt/conda/bin/python"
fi

OUT_DIR="$WORKSPACE/experiments/results/c063_breakthrough_candidate_matrix_20260502/h100_dispatch_matrix/c063_pose_waterfill_048_pr67_pr65_basis"
SRC_ARCHIVE="$WORKSPACE/experiments/results/c063_breakthrough_candidate_matrix_20260502/line_search_source_c063_fixedslice/archive.zip"
SRC_METADATA="$WORKSPACE/experiments/results/c063_breakthrough_candidate_matrix_20260502/line_search_source_c063_fixedslice/metadata.json"
OUT_ARCHIVE="$OUT_DIR/archive.zip"
OUT_METADATA="$OUT_DIR/metadata.json"
LOG="$OUT_DIR/line_search.log"
DISPATCH_PROVENANCE="$OUT_DIR/dispatch_provenance.json"
mkdir -p "$OUT_DIR"

( while true; do echo "$(date -u +%FT%TZ) heartbeat pid=$$" >> "$OUT_DIR/heartbeat.log"; sleep 60; done ) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null || true' EXIT

"$PYBIN" - <<'PY' "$DISPATCH_PROVENANCE" "$SRC_ARCHIVE" "$SRC_METADATA"
import hashlib, json, pathlib, subprocess, sys, time

out = pathlib.Path(sys.argv[1])
archive = pathlib.Path(sys.argv[2])
metadata = pathlib.Path(sys.argv[3])

def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def cmd(args):
    return subprocess.run(args, check=False, capture_output=True, text=True, timeout=30)

gpu = cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
payload = {
    "schema_version": 1,
    "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "lane_id": "c063_pose_waterfill_048_frontier_fixedslice_h100sxm",
    "instance_job_id": "35995649:c063_pose_waterfill_048_fixedslice_20260502T0838Z",
    "score_claim": False,
    "promotion_eligible": False,
    "purpose": "H100 pose-waterfill line search from a C-063-derived fixed-slice source; exact H100 eval is diagnostic, T4 required for promotion.",
    "source_archive": {"path": str(archive), "bytes": archive.stat().st_size, "sha256": sha(archive)},
    "source_metadata": {"path": str(metadata), "bytes": metadata.stat().st_size, "sha256": sha(metadata)},
    "gpu_probe": gpu.stdout.strip() or gpu.stderr.strip(),
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

echo "[driver] $(date -u +%FT%TZ) CUDA/NVDEC preflight" | tee -a "$LOG"
"$PYBIN" - <<'PY' | tee -a "$LOG"
import json, torch
print(json.dumps({
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, sort_keys=True))
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
PY

if [ -f scripts/probe_nvdec.sh ]; then
    bash scripts/probe_nvdec.sh --ensure-dali 2>&1 | tee -a "$LOG"
fi

echo "[driver] $(date -u +%FT%TZ) line_search start" | tee -a "$LOG"
"$PYBIN" -u experiments/line_search_pose_refinement.py \
  --archive-path "$SRC_ARCHIVE" \
  --metadata-path "$SRC_METADATA" \
  --output-path "$OUT_ARCHIVE" \
  --output-metadata "$OUT_METADATA" \
  --posenet-path upstream/models/posenet.safetensors \
  --gt-mkv upstream/videos/0.mkv \
  --device cuda:0 \
  --batch-size 16 \
  --candidate-chunk 32 \
  --basis-delta-sets 'pair_window:1,2,3;dct:1,2' \
  --basis-pair-indices '128,67,69,64,125,423,97,70,148,121,141,289,435,78,434,150,75,310,138,418,105,159,74,421,290,101,103,191,142,200,430,126,129,95,426,85,144,179,183,133,203,431,439,98,114,96,427,429' \
  --basis-window-radius 1 \
  --passes 2 \
  --progress-every-candidates 64 \
  2>&1 | tee -a "$LOG"

test -f "$OUT_ARCHIVE"
test -f "$OUT_METADATA"

echo "[driver] $(date -u +%FT%TZ) exact H100 eval start" | tee -a "$LOG"
ARCHIVE_PATH="$OUT_ARCHIVE" \
ARCHIVE_LABEL="c063_pose_waterfill_048_fixedslice_h100sxm_20260502T0838Z" \
PREDICTED_LOW="0.31" \
PREDICTED_HIGH="0.316" \
CONTROLLED_BASELINE="C-063_T4_score_0.3156230307844823_bytes_276223" \
LOG_DIR="$OUT_DIR/exact_eval_h100" \
PYBIN="$PYBIN" \
KEEP_EVAL_WORK="0" \
bash scripts/remote_archive_only_eval.sh

echo "[driver] $(date -u +%FT%TZ) done" | tee -a "$LOG"
