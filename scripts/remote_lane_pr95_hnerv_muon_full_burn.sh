#!/usr/bin/env bash
# Owned PR95/HNeRV/Muon full-burn reproduction lane.
#
# This lane copies the public PR95 source intake into the canonical
# submissions/hnerv_muon location, runs its from-scratch curriculum on the
# fastest available CUDA host, and emits deterministic custody metadata around
# whatever checkpoint/archive is produced. It is a training lane, not score
# evidence; exact CUDA auth eval of the final archive remains mandatory.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-python3}"
RUN_ID="${RUN_ID:-owned_pr95_hnerv_muon_full_burn_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-$WORKSPACE/experiments/results/$RUN_ID}"
PR95_SRC="${PR95_SRC:-$WORKSPACE/experiments/results/public_pr95_intake_20260504_codex/pr95_src/submissions/hnerv_muon}"
PUBLIC_ARCHIVE="${PUBLIC_ARCHIVE:-$WORKSPACE/experiments/results/public_pr95_intake_20260504_codex/archive.zip}"
EXPECTED_PUBLIC_ARCHIVE_SHA256="${EXPECTED_PUBLIC_ARCHIVE_SHA256:-e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a}"
COMMA_CHALLENGE_ROOT="${COMMA_CHALLENGE_ROOT:-$WORKSPACE/upstream}"
PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
TORCH_DETERMINISTIC="${TORCH_DETERMINISTIC:-1}"

cd "$WORKSPACE"
if [ -f "$WORKSPACE/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$WORKSPACE/env.sh"
fi

export PYTHONPATH="$WORKSPACE:$WORKSPACE/src:$WORKSPACE/upstream:${PYTHONPATH:-}"
export COMMA_CHALLENGE_ROOT PYTHONHASHSEED TORCH_DETERMINISTIC
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"
HEARTBEAT="$OUT_DIR/heartbeat.log"
PROVENANCE="$OUT_DIR/provenance.json"
SNAPSHOT_INTERVAL_SECONDS="${SNAPSHOT_INTERVAL_SECONDS:-300}"

log() {
  echo "[pr95-hnerv-full-burn] $(date -u +%FT%TZ) $*" | tee -a "$LOG"
}

HB_PID=""
SNAPSHOT_PID=""
cleanup() {
  set +e
  if [ -n "${HB_PID:-}" ]; then
    kill "$HB_PID" 2>/dev/null || true
  fi
  if [ -n "${SNAPSHOT_PID:-}" ]; then
    kill "$SNAPSHOT_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

log "=== Stage 0: NVDEC/CUDA probe ==="
if [ -x "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
  bash "$WORKSPACE/scripts/probe_nvdec.sh" | tee -a "$LOG"
else
  log "WARN: scripts/probe_nvdec.sh missing; continuing because PR95 training uses PyAV decode, but recording this as infrastructure drift"
fi

(
  while true; do
    {
      printf '{"ts":"%s","run_id":"%s"' "$(date -u +%FT%TZ)" "$RUN_ID"
      if command -v nvidia-smi >/dev/null 2>&1; then
        printf ',"gpu":'
        nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
          | "$PYBIN" -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))'
      fi
      printf '}\n'
    } >> "$HEARTBEAT" 2>/dev/null || true
    sleep 60
  done
) &
HB_PID=$!

snapshot_latest_archive() {
  "$PYBIN" - "$WORKSPACE" "$OUT_DIR" <<'PY'
import hashlib
import json
import pathlib
import shutil
import sys
import time
import zipfile

workspace = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
ckpt_root = workspace / "submissions" / "hnerv_muon" / "ckpts"

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "schema_version": 1,
    "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "status": "no_checkpoint_yet",
    "score_claim": False,
    "evidence_grade": "training_snapshot_until_exact_cuda_eval",
    "predicted_band": "training_snapshot_only_no_score_predicted",
}
runs = sorted(
    [p for p in ckpt_root.glob("run_*") if p.is_dir()],
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if runs:
    latest = runs[0]
    manifest["latest_run_dir"] = str(latest)
    candidates = [
        latest / "submission_archive" / "0.bin",
        latest / "stage8" / "best_archive.bin",
        latest / "stage7" / "best_archive.bin",
        latest / "stage6" / "best_archive.bin",
        latest / "stage5" / "best_archive.bin",
        latest / "stage4" / "best_archive.bin",
        latest / "stage3" / "best_archive.bin",
        latest / "stage2" / "best_archive.bin",
        latest / "stage1" / "best_archive.bin",
    ]
    source = next((p for p in candidates if p.is_file()), None)
    if source is not None:
        out_bin = out_dir / "0.latest.bin"
        shutil.copy2(source, out_bin)
        archive_zip = out_dir / "archive.latest.zip"
        info = zipfile.ZipInfo("0.bin")
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.external_attr = 0o100644 << 16
        with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr(info, out_bin.read_bytes())
        manifest.update(
            {
                "status": "snapshot_ready",
                "source_archive_bin": str(source),
                "latest_bin": {
                    "path": str(out_bin),
                    "bytes": out_bin.stat().st_size,
                    "sha256": sha256(out_bin),
                },
                "archive_latest_zip": {
                    "path": str(archive_zip),
                    "bytes": archive_zip.stat().st_size,
                    "sha256": sha256(archive_zip),
                    "member": "0.bin",
                    "zip_compression": "stored",
                    "deterministic_timestamp": "1980-01-01T00:00:00Z",
                },
            }
        )
(out_dir / "snapshot_latest_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps({"snapshot_status": manifest["status"], "recorded_at_utc": manifest["recorded_at_utc"]}, sort_keys=True))
PY
}

(
  while true; do
    snapshot_latest_archive >> "$OUT_DIR/snapshot_loop.log" 2>&1 || true
    sleep "$SNAPSHOT_INTERVAL_SECONDS"
  done
) &
SNAPSHOT_PID=$!

log "=== Stage 1: source and archive custody ==="
test -d "$PR95_SRC"
test -f "$PR95_SRC/src/train.py"
test -f "$PUBLIC_ARCHIVE"
"$PYBIN" - "$PR95_SRC" "$PUBLIC_ARCHIVE" "$EXPECTED_PUBLIC_ARCHIVE_SHA256" "$PROVENANCE" <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

src = pathlib.Path(sys.argv[1])
archive = pathlib.Path(sys.argv[2])
expected_sha = sys.argv[3]
prov = pathlib.Path(sys.argv[4])

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

archive_sha = sha256(archive)
if archive_sha != expected_sha:
    raise SystemExit(f"public archive SHA mismatch: expected={expected_sha} actual={archive_sha}")

files = []
for path in sorted(src.rglob("*")):
    if path.is_file() and "__pycache__" not in path.parts:
        rel = path.relative_to(src).as_posix()
        files.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)})

gpu = None
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True,
        timeout=20,
    )
    gpu = out.strip().splitlines()
except Exception as exc:
    gpu = [f"nvidia-smi unavailable: {exc!r}"]

payload = {
    "schema_version": 1,
    "lane": "owned_pr95_hnerv_muon_full_burn_fastchip",
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "workspace": os.getcwd(),
    "pr95_src": str(src),
    "public_archive": str(archive),
    "public_archive_bytes": archive.stat().st_size,
    "public_archive_sha256": archive_sha,
    "source_files": files,
    "source_file_count": len(files),
    "python": sys.executable,
    "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    "torch_deterministic_env": os.environ.get("TORCH_DETERMINISTIC"),
    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    "comma_challenge_root": os.environ.get("COMMA_CHALLENGE_ROOT"),
    "nvidia_smi": gpu,
    "score_claim": False,
    "evidence_grade": "training_artifact_until_exact_cuda_eval",
}
prov.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({"public_archive_sha256": archive_sha, "source_file_count": len(files)}, sort_keys=True))
PY

log "=== Stage 2: install/copy PR95 runtime source ==="
rm -rf "$WORKSPACE/submissions/hnerv_muon"
mkdir -p "$WORKSPACE/submissions"
cp -a "$PR95_SRC" "$WORKSPACE/submissions/hnerv_muon"
find "$WORKSPACE/submissions/hnerv_muon" -type d -name __pycache__ -prune -exec rm -rf {} +

"$PYBIN" -m py_compile \
  "$WORKSPACE/submissions/hnerv_muon/inflate.py" \
  "$WORKSPACE/submissions/hnerv_muon/src/codec.py" \
  "$WORKSPACE/submissions/hnerv_muon/src/data.py" \
  "$WORKSPACE/submissions/hnerv_muon/src/losses.py" \
  "$WORKSPACE/submissions/hnerv_muon/src/model.py" \
  "$WORKSPACE/submissions/hnerv_muon/src/optim.py" \
  "$WORKSPACE/submissions/hnerv_muon/src/score.py" \
  "$WORKSPACE/submissions/hnerv_muon/src/train.py"

log "=== Stage 3: dependency/runtime preflight ==="
"$PYBIN" - <<'PY' | tee -a "$LOG"
import json
import os
import torch
payload = {
    "torch_version": torch.__version__,
    "torch_cuda": getattr(torch.version, "cuda", None),
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "comma_challenge_root": os.environ.get("COMMA_CHALLENGE_ROOT"),
}
if os.environ.get("TORCH_DETERMINISTIC") == "1":
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        payload["deterministic_algorithms"] = "warn_only"
    except Exception as exc:
        payload["deterministic_algorithms_error"] = repr(exc)
print(json.dumps(payload, sort_keys=True))
PY

log "=== Stage 4: PR95 HNeRV/Muon full curriculum ==="
set +e
(
  export PYTHONPATH="$WORKSPACE/submissions/hnerv_muon/src:$WORKSPACE:$WORKSPACE/src:$WORKSPACE/upstream:${PYTHONPATH:-}"
  cd "$WORKSPACE"
  "$PYBIN" "$WORKSPACE/submissions/hnerv_muon/src/train.py"
) 2>&1 | tee "$OUT_DIR/train.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
log "training_returncode=$TRAIN_RC"

log "=== Stage 5: collect final/partial artifacts ==="
"$PYBIN" - "$WORKSPACE" "$OUT_DIR" "$TRAIN_RC" <<'PY'
import hashlib
import json
import pathlib
import shutil
import sys
import time
import zipfile

workspace = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
train_rc = int(sys.argv[3])
ckpt_root = workspace / "submissions" / "hnerv_muon" / "ckpts"

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

runs = sorted([p for p in ckpt_root.glob("run_*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
manifest = {
    "schema_version": 1,
    "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "training_returncode": train_rc,
    "ckpt_root": str(ckpt_root),
    "run_dirs": [str(p) for p in runs[:5]],
    "score_claim": False,
    "evidence_grade": "training_artifact_until_exact_cuda_eval",
}
if runs:
    latest = runs[0]
    manifest["latest_run_dir"] = str(latest)
    for rel in ("submission_archive/0.bin", "stage8/best_archive.bin", "stage8/decoder_f32.pt", "stage8/latents_f32.pt"):
        src = latest / rel
        if src.is_file():
            dst = out_dir / rel.replace("/", "__")
            shutil.copy2(src, dst)
            manifest[rel] = {"copied_to": str(dst), "bytes": dst.stat().st_size, "sha256": sha256(dst)}
    zero_bin = latest / "submission_archive" / "0.bin"
    if zero_bin.is_file():
        archive_zip = out_dir / "archive.zip"
        info = zipfile.ZipInfo("0.bin")
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.external_attr = 0o100644 << 16
        with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr(info, zero_bin.read_bytes())
        manifest["archive_zip"] = {
            "path": str(archive_zip),
            "bytes": archive_zip.stat().st_size,
            "sha256": sha256(archive_zip),
            "member": "0.bin",
            "zip_compression": "stored",
            "deterministic_timestamp": "1980-01-01T00:00:00Z",
        }
(out_dir / "training_artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, sort_keys=True))
PY

if [ "$TRAIN_RC" -ne 0 ]; then
  log "FATAL: training exited nonzero; artifacts preserved"
  exit "$TRAIN_RC"
fi

log "=== COMPLETE: PR95 HNeRV/Muon training artifact ready for exact CUDA eval ==="
