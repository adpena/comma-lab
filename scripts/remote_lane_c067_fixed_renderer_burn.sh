#!/usr/bin/env bash
# NO_NVDEC_NEEDED — C-067 fixed-renderer burn delegates video decode to canonical
# scripts/remote_archive_only_eval.sh, which already runs scripts/probe_nvdec.sh
# at its own Stage 0. This driver does no DALI/NVDEC video work itself.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-.venv/bin/python}"
RUN_ID="${RUN_ID:-c067_qfaithful_fixedmask_fixedpose_seed20260503_fix2_h200vast}"
RUN_DIR="${RUN_DIR:-experiments/results/c067_fixed_renderer_burn_prep_20260503/${RUN_ID}}"
SCRIPT_PATH="${RUN_DIR}/run_fixed_renderer_burn.sh"
LOG_DIR="${RUN_DIR}/logs"
HEARTBEAT="${LOG_DIR}/vast_heartbeat.log"
PROVENANCE="${LOG_DIR}/vast_dispatch_provenance.json"

cd "$WORKSPACE"
[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
mkdir -p "$LOG_DIR"

# Stage 0: NVDEC probe — required by preflight check_remote_scripts_have_nvdec_probe.
# Defensive even though this script delegates video decode to remote_archive_only_eval.sh,
# which runs its own probe at its Stage 0.
if [ "${SKIP_NVDEC_PROBE:-0}" != "1" ] && [ -f "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
        printf '[c067-fixed-renderer-burn] FATAL: NVDEC/DALI probe failed; exact CUDA eval is not trustworthy on this host.\n' >&2
        exit 2
    }
fi

log() {
  printf '[c067-fixed-renderer-burn] %s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG_DIR/vast_dispatch.log"
}

log "start run_id=${RUN_ID}"
if [ ! -x "$PYBIN" ]; then
  log "FATAL: python not executable at ${PYBIN}"
  exit 2
fi
if [ ! -f "$SCRIPT_PATH" ]; then
  log "FATAL: missing prepared burn script ${SCRIPT_PATH}"
  exit 3
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "FATAL: nvidia-smi missing"
  exit 4
fi

mkdir -p .venv/bin
if [ ! -x .venv/bin/python ]; then
  ln -sf "$PYBIN" .venv/bin/python
fi

"$PYBIN" - "$PROVENANCE" "$RUN_ID" "$RUN_DIR" "$SCRIPT_PATH" <<'PY'
import json
import pathlib
import subprocess
import sys
import time

out = pathlib.Path(sys.argv[1])
payload = {
    "schema": "c067_fixed_renderer_burn_vast_dispatch_v1",
    "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "run_id": sys.argv[2],
    "run_dir": sys.argv[3],
    "script_path": sys.argv[4],
    "score_claim": False,
    "promotion_eligible": False,
    "operator_override": "minimum_wallclock_high_ev_renderer_training_burn",
    "predicted_band": [0.27, 0.33],
}
try:
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    payload["nvidia_smi_returncode"] = probe.returncode
    payload["nvidia_smi_stdout"] = probe.stdout.strip()
    payload["nvidia_smi_stderr"] = probe.stderr.strip()
except Exception as exc:
    payload["nvidia_smi_error"] = repr(exc)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY

( while true; do
    GPU="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>&1 | tr '\n' ' ')"
    printf '[%s] lane=c067-fixed-renderer-burn run_id=%s gpu=%s\n' "$(date -u +%FT%TZ)" "$RUN_ID" "$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null || true' EXIT

log "running ${SCRIPT_PATH}"
bash "$SCRIPT_PATH"
log "completed ${SCRIPT_PATH}"
