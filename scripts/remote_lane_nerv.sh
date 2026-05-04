#!/bin/bash
# Lane 12: NeRV mask codec (Phase 2 ACCELERATE).
# predicted_band=[0.95, 1.30] [contest-CUDA]
# Replaces Lane G v3 masks.mkv (AV1, 421 KB) with masks.nrv (NeRV, ~12-23 KB).
# Renderer + poses unchanged from Lane G v3 anchor.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-python3}"
cd "$WORKSPACE"
if [ -f "$WORKSPACE/env.sh" ]; then
    # shellcheck disable=SC1091
    [ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"
fi

export PYTHONPATH="src:upstream:${PYTHONPATH:-}"
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_12_nerv_results}"
NERV_DIR="$LOG_DIR/nerv"
LANE_G_V3_BASE_ARCHIVE_REL="${LANE_G_V3_BASE_ARCHIVE_REL:-experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip}"
LANE_G_V3_BASE_ARCHIVE="${LANE_G_V3_BASE_ARCHIVE:-$WORKSPACE/$LANE_G_V3_BASE_ARCHIVE_REL}"
BASE_ARCHIVE="${BASE_ARCHIVE:-$LANE_G_V3_BASE_ARCHIVE}"
ARCHIVE="$LOG_DIR/archive_lane_12_nerv.zip"
PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
PROFILE="${PROFILE:-nerv_mask_lane_g_v3}"
GT_MASKS_SOURCE="${GT_MASKS_SOURCE:-decoded-baseline}"
DECODED_BASELINE_PATH="${DECODED_BASELINE_PATH:-$BASE_ARCHIVE}"
DECODED_BASELINE_MEMBER="${DECODED_BASELINE_MEMBER:-masks.mkv}"
RUN_AUTH_EVAL="${RUN_AUTH_EVAL:-0}"
NERV_STEPS="${NERV_STEPS:-}"
NERV_EVAL_EVERY="${NERV_EVAL_EVERY:-}"
NERV_WEIGHT_DTYPE="${NERV_WEIGHT_DTYPE:-}"
POSE_REGEN_PROVENANCE="${POSE_REGEN_PROVENANCE:-}"
ALPHA_GEO_PROVENANCE="${ALPHA_GEO_PROVENANCE:-}"
ALPHA_PRIMITIVE_CONTRACT="${ALPHA_PRIMITIVE_CONTRACT:-}"
L2_CLEARANCE_PATH="${L2_CLEARANCE_PATH:-$WORKSPACE/.omx/state/lane12_nerv_l2_clearance.json}"
ALLOW_RETIRED_SEGNET_TARGET="${ALLOW_RETIRED_SEGNET_TARGET:-0}"
BASELINE_SCORE="${BASELINE_SCORE:-1.05}"
mkdir -p "$LOG_DIR" "$NERV_DIR"

log() { echo "[lane-12-nerv] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

HB_PID=""
cleanup() {
    set +e
    if [ -n "${HB_PID:-}" ]; then
        kill "$HB_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Stage 0: NVDEC probe BEFORE any GPU spend (CLAUDE.md non-negotiable, see
# feedback_vastai_nvdec_host_variation; preflight check_remote_scripts_have_nvdec_probe
# fires if probe runs after first GPU-work marker — torch.cuda.is_available
# below counts as GPU work).
log "=== Stage 0: NVDEC probe ==="
bash "$WORKSPACE/scripts/probe_nvdec.sh" || {
    log "FATAL: NVDEC probe failed"
    exit 2
}

GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "no-git")
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1 || true)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1 || true)
else
    GPU_NAME="nvidia-smi unavailable"
    DRIVER_VER="nvidia-smi unavailable"
fi
export PROVENANCE LOG_DIR GIT_HASH GPU_NAME DRIVER_VER BASE_ARCHIVE NERV_DIR PROFILE GT_MASKS_SOURCE DECODED_BASELINE_PATH DECODED_BASELINE_MEMBER
export NERV_STEPS NERV_EVAL_EVERY NERV_WEIGHT_DTYPE
export RUN_AUTH_EVAL POSE_REGEN_PROVENANCE ALPHA_GEO_PROVENANCE ALPHA_PRIMITIVE_CONTRACT L2_CLEARANCE_PATH ALLOW_RETIRED_SEGNET_TARGET LANE_G_V3_BASE_ARCHIVE
"$PYBIN" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import time

import torch


def optional_json_file_meta(path_str: str) -> dict | None:
    if not path_str:
        return None
    path = Path(path_str)
    meta: dict = {"path": path_str, "exists": path.is_file()}
    if not path.is_file():
        return meta
    raw = path.read_bytes()
    meta.update(
        {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    )
    try:
        payload = json.loads(raw)
    except Exception as exc:
        meta["json_error"] = str(exc)
    else:
        if isinstance(payload, dict):
            meta.update(
                {
                    "diagnostic": payload.get("diagnostic"),
                    "schema_version": payload.get("schema_version"),
                    "promotion_eligible": payload.get("promotion_eligible"),
                    "score_claim_eligible": payload.get("score_claim_eligible"),
                    "exact_eval_claim": payload.get("exact_eval_claim"),
                }
            )
    return meta


prov = {
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_hash": os.environ["GIT_HASH"],
    "gpu_name": os.environ["GPU_NAME"],
    "driver_version": os.environ["DRIVER_VER"],
    "torch_version": torch.__version__,
    "cuda_version": getattr(torch.version, "cuda", None),
    "cuda_available": torch.cuda.is_available(),
    "lane_script": "scripts/remote_lane_nerv.sh",
    "lane_name": "lane_12_nerv_mask_codec",
    "predicted_band": [0.95, 1.30],
    "strict_scorer_rule_compliant": True,
    "base_archive": os.environ["BASE_ARCHIVE"],
    "canonical_lane_g_v3_base_archive": os.environ["LANE_G_V3_BASE_ARCHIVE"],
    "profile": os.environ["PROFILE"],
    "gt_masks_source": os.environ["GT_MASKS_SOURCE"],
    "decoded_baseline_path": os.environ["DECODED_BASELINE_PATH"],
    "decoded_baseline_member": os.environ["DECODED_BASELINE_MEMBER"],
    "nerv_steps_override": os.environ["NERV_STEPS"],
    "nerv_eval_every_override": os.environ["NERV_EVAL_EVERY"],
    "nerv_weight_dtype_override": os.environ["NERV_WEIGHT_DTYPE"],
    "run_auth_eval": os.environ["RUN_AUTH_EVAL"],
    "pose_regen_provenance": os.environ["POSE_REGEN_PROVENANCE"],
    "alpha_geo_provenance": os.environ["ALPHA_GEO_PROVENANCE"],
    "alpha_primitive_contract": optional_json_file_meta(os.environ["ALPHA_PRIMITIVE_CONTRACT"]),
    "l2_clearance_path": os.environ["L2_CLEARANCE_PATH"],
    "allow_retired_segnet_target": os.environ["ALLOW_RETIRED_SEGNET_TARGET"],
    "output_dir": os.environ["LOG_DIR"],
}
with open(os.environ["PROVENANCE"], "w", encoding="utf-8") as f:
    json.dump(prov, f, indent=2)
print("provenance:", json.dumps(prov))
PY

log "=== Stage 0c: Lane 12 L2 retraining clearance ==="
"$PYBIN" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["L2_CLEARANCE_PATH"])
if not path.exists():
    raise SystemExit(
        "FATAL: missing Lane 12 L2 clearance packet: "
        f"{path}. No new NeRV retraining is allowed until this packet is valid."
    )
try:
    payload = json.loads(path.read_text())
except Exception as exc:
    raise SystemExit(f"FATAL: unreadable Lane 12 L2 clearance packet {path}: {exc}") from exc
if not isinstance(payload, dict):
    raise SystemExit(f"FATAL: Lane 12 L2 clearance packet {path} must be a JSON object")

violations = []
if payload.get("lane_id") not in {"lane_12_nerv_mask_codec", "lane_12_nerv"}:
    violations.append("lane_id must be lane_12_nerv_mask_codec or lane_12_nerv")
if payload.get("cleared_for_retraining_unblock") is not True:
    violations.append("cleared_for_retraining_unblock must be true")
if payload.get("lane12_l2") is not True:
    violations.append("lane12_l2 must be true")
if payload.get("geometry_gate_passed") is not True:
    violations.append("geometry_gate_passed must be true")
clean_passes = payload.get("grand_council_clean_passes")
if not isinstance(clean_passes, int) or clean_passes < 3:
    violations.append("grand_council_clean_passes must be an integer >= 3")
evidence = payload.get("evidence")
if isinstance(evidence, str):
    evidence_ok = bool(evidence.strip())
elif isinstance(evidence, list):
    evidence_ok = bool(evidence) and all(isinstance(item, str) and item.strip() for item in evidence)
else:
    evidence_ok = False
if not evidence_ok:
    violations.append("evidence must cite the Lane 12 L2 packet/artifacts")
if violations:
    raise SystemExit("FATAL: invalid Lane 12 L2 clearance packet: " + "; ".join(violations))
print("lane12_l2_clearance:", json.dumps({"path": str(path), "evidence": evidence, "grand_council_clean_passes": clean_passes}))
PY

( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane=12-nerv gpu=$GPU" >> "$HEARTBEAT"
    sleep 300
  done ) &
HB_PID=$!

# NVDEC probe moved earlier (above provenance block) — see Stage 0 above.
"$PYBIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("FATAL: CUDA is required for remote lane 12 (NeRV)")
print("cuda:", torch.cuda.get_device_name(0))
PY

log "=== Stage 0b: install editable package ==="
# CODE PARITY: launcher tarball is authoritative — do NOT git reset --hard.
# (memory: feedback_git_reset_nukes_anchors_20260429)
"$PYBIN" -m pip install -e .

for f in "$BASE_ARCHIVE" "$TAC_UPSTREAM_DIR/videos/0.mkv"; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
for f in experiments/contest_auth_eval.py scripts/adjudicate_contest_auth_eval.py; do
    [ -f "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
if [ "$GT_MASKS_SOURCE" = "segnet" ] && [ "$ALLOW_RETIRED_SEGNET_TARGET" != "1" ]; then
    log "FATAL: GT_MASKS_SOURCE=segnet repeats the retired jsonfix40 target path. Set ALLOW_RETIRED_SEGNET_TARGET=1 only for a documented forensic rerun."
    exit 6
fi
if [ "$GT_MASKS_SOURCE" = "decoded-baseline" ]; then
    [ -f "$DECODED_BASELINE_PATH" ] || { echo "FATAL: missing decoded baseline $DECODED_BASELINE_PATH" >&2; exit 1; }
    if [ -z "$ALPHA_PRIMITIVE_CONTRACT" ]; then
        log "FATAL: GT_MASKS_SOURCE=decoded-baseline requires ALPHA_PRIMITIVE_CONTRACT alpha_geo_primitive_contract_v1 for production dispatch."
        exit 6
    fi
    if [ ! -f "$ALPHA_PRIMITIVE_CONTRACT" ]; then
        log "FATAL: ALPHA_PRIMITIVE_CONTRACT does not exist: $ALPHA_PRIMITIVE_CONTRACT"
        exit 6
    fi
    "$PYBIN" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["ALPHA_PRIMITIVE_CONTRACT"])
try:
    payload = json.loads(path.read_text())
except Exception as exc:
    raise SystemExit(f"FATAL: unreadable ALPHA_PRIMITIVE_CONTRACT {path}: {exc}") from exc
if not isinstance(payload, dict):
    raise SystemExit(f"FATAL: ALPHA_PRIMITIVE_CONTRACT {path} must be a JSON object")
violations = []
if payload.get("diagnostic") != "alpha_geo_primitive_contract_v1":
    violations.append("diagnostic must be alpha_geo_primitive_contract_v1")
if payload.get("promotion_eligible") is not False:
    violations.append("promotion_eligible must be false")
if payload.get("score_claim_eligible") is not False:
    violations.append("score_claim_eligible must be false")
if payload.get("exact_eval_claim") is not False:
    violations.append("exact_eval_claim must be false")
if violations:
    raise SystemExit("FATAL: invalid ALPHA_PRIMITIVE_CONTRACT: " + "; ".join(violations))
print("alpha_primitive_contract:", json.dumps({"path": str(path), "diagnostic": payload.get("diagnostic"), "promotion_eligible": False}))
PY
else
    log "FATAL: GT_MASKS_SOURCE=$GT_MASKS_SOURCE is not Alpha-Geo dispatch-ready. Use decoded-baseline or record a reviewed exception."
    exit 6
fi
if [ "$RUN_AUTH_EVAL" = "1" ]; then
    if [ -z "$POSE_REGEN_PROVENANCE" ]; then
        log "FATAL: RUN_AUTH_EVAL=1 requires POSE_REGEN_PROVENANCE for the candidate mask stream."
        exit 6
    fi
    if [ ! -f "$POSE_REGEN_PROVENANCE" ]; then
        log "FATAL: POSE_REGEN_PROVENANCE does not exist: $POSE_REGEN_PROVENANCE"
        exit 6
    fi
    if [ -z "$ALPHA_GEO_PROVENANCE" ]; then
        log "FATAL: RUN_AUTH_EVAL=1 requires ALPHA_GEO_PROVENANCE with pass_fail.overall_pass=true for this archive."
        exit 6
    fi
    if [ ! -f "$ALPHA_GEO_PROVENANCE" ]; then
        log "FATAL: ALPHA_GEO_PROVENANCE does not exist: $ALPHA_GEO_PROVENANCE"
        exit 6
    fi
else
    log "RUN_AUTH_EVAL=$RUN_AUTH_EVAL; this run will stop after deterministic archive build and will not produce score evidence."
fi
BASELINE_ARCHIVE_BYTES=$(stat -f%z "$BASE_ARCHIVE" 2>/dev/null || stat -c%s "$BASE_ARCHIVE")

log "=== Stage 1: train NeRV mask codec ==="
TRAIN_TARGET_ARGS=()
if [ "$GT_MASKS_SOURCE" = "decoded-baseline" ]; then
    [ -f "$DECODED_BASELINE_PATH" ] || { echo "FATAL: missing $DECODED_BASELINE_PATH" >&2; exit 1; }
	TRAIN_TARGET_ARGS+=(
	    --decoded-baseline-path "$DECODED_BASELINE_PATH"
	    --decoded-baseline-member "$DECODED_BASELINE_MEMBER"
	    --alpha-primitive-contract "$ALPHA_PRIMITIVE_CONTRACT"
	)
fi
NERV_TRAINING_ARGS=()
if [ -n "${NERV_STEPS:-}" ]; then
    NERV_TRAINING_ARGS+=(--steps "$NERV_STEPS")
fi
if [ -n "${NERV_EVAL_EVERY:-}" ]; then
    NERV_TRAINING_ARGS+=(--eval-every "$NERV_EVAL_EVERY")
fi
if [ -n "${NERV_WEIGHT_DTYPE:-}" ]; then
    NERV_TRAINING_ARGS+=(--weight-dtype "$NERV_WEIGHT_DTYPE")
fi
"$PYBIN" -u experiments/train_nerv_mask.py \
    --profile "$PROFILE" \
    --device cuda \
    --upstream "$TAC_UPSTREAM_DIR" \
    --gt-masks-source "$GT_MASKS_SOURCE" \
    --output-dir "$NERV_DIR" \
    "${TRAIN_TARGET_ARGS[@]}" \
    "${NERV_TRAINING_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/train_nerv_mask.log"
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

NERV_PAYLOAD="$NERV_DIR/masks.nrv"
[ -f "$NERV_PAYLOAD" ] || { echo "FATAL: missing $NERV_PAYLOAD" >&2; exit 3; }
NERV_BYTES=$(stat -f%z "$NERV_PAYLOAD" 2>/dev/null || stat -c%s "$NERV_PAYLOAD")
log "NeRV payload: $NERV_PAYLOAD ($NERV_BYTES bytes)"

# Kill criterion: > 100 KB → abandon (per Phase B council)
if [ "$NERV_BYTES" -gt 100000 ]; then
    log "FATAL: NeRV payload $NERV_BYTES B > 100 KB run-abort threshold"
    exit 4
fi

log "=== Stage 2: rebuild archive with masks.nrv replacing masks.mkv ==="
export BASE_ARCHIVE ARCHIVE NERV_PAYLOAD
"$PYBIN" - <<'PY'
import os
from pathlib import PurePosixPath
import zipfile

from tac.submission_archive import detect_pose_manifest, validate_archive

src = os.environ["BASE_ARCHIVE"]
dst = os.environ["ARCHIVE"]
nerv = os.environ["NERV_PAYLOAD"]
# Replace an existing root mask payload with masks.nrv while keeping only the
# canonical renderer and optimized-pose payload from the base archive.
mask_extensions = (".mkv", ".amrc", ".stcb", ".nrv")
allowed_passthrough = {"renderer.bin", "optimized_poses.pt", "optimized_poses.bin"}
payloads = {}
seen = set()
with zipfile.ZipFile(src, "r") as zin:
    for info in zin.infolist():
        member_path = PurePosixPath(info.filename)
        parts = member_path.parts
        if info.filename in seen:
            raise SystemExit(f"FATAL: duplicate BASE_ARCHIVE member: {info.filename!r}")
        seen.add(info.filename)
        if info.is_dir() or member_path.is_absolute() or ".." in parts:
            raise SystemExit(f"FATAL: unsafe BASE_ARCHIVE member path: {info.filename!r}")
        if (
            "__MACOSX" in parts
            or ".DS_Store" in parts
            or "Thumbs.db" in parts
            or any(part.startswith("._") or part.startswith(".") for part in parts)
        ):
            raise SystemExit(f"FATAL: hidden/system BASE_ARCHIVE member: {info.filename!r}")
        if info.filename.startswith("masks.") and info.filename.endswith(mask_extensions):
            continue
        if info.filename not in allowed_passthrough:
            raise SystemExit(f"FATAL: unexpected BASE_ARCHIVE member: {info.filename!r}")
        payloads[info.filename] = zin.read(info.filename)

missing = {"renderer.bin"} - payloads.keys()
pose_members = sorted(set(payloads) & {"optimized_poses.pt", "optimized_poses.bin"})
if missing:
    raise SystemExit(f"FATAL: BASE_ARCHIVE missing required member(s): {sorted(missing)}")
if len(pose_members) != 1:
    raise SystemExit(
        "FATAL: BASE_ARCHIVE must contain exactly one optimized pose artifact; "
        f"found {pose_members}"
    )

with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zout:
    for name in ["renderer.bin", pose_members[0]]:
        out_info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        out_info.compress_type = zipfile.ZIP_DEFLATED
        out_info.external_attr = (0o644 & 0xFFFF) << 16
        zout.writestr(out_info, payloads[name], compresslevel=9)
    out_info = zipfile.ZipInfo("masks.nrv", date_time=(1980, 1, 1, 0, 0, 0))
    out_info.compress_type = zipfile.ZIP_DEFLATED
    out_info.external_attr = (0o644 & 0xFFFF) << 16
    with open(nerv, "rb") as f:
        zout.writestr(out_info, f.read(), compresslevel=9)
result = validate_archive(dst, manifest=detect_pose_manifest(dst), strict=True)
if not result.valid:
    raise SystemExit(f"FATAL: rebuilt Lane 12 archive failed validation:\n{result.summary()}")
print(f"archive {dst}: {os.path.getsize(dst)} bytes (with masks.nrv)")
PY

if [ "$RUN_AUTH_EVAL" = "1" ]; then
    log "=== Stage 2b: verify Alpha-Geo geometry gate for exact archive ==="
    export ALPHA_GEO_PROVENANCE ARCHIVE BASE_ARCHIVE
    "$PYBIN" - <<'PY'
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


geo_path = Path(os.environ["ALPHA_GEO_PROVENANCE"])
archive = Path(os.environ["ARCHIVE"])
base_archive = Path(os.environ["BASE_ARCHIVE"])
try:
    payload = json.loads(geo_path.read_text())
except Exception as exc:
    raise SystemExit(f"FATAL: unreadable ALPHA_GEO_PROVENANCE {geo_path}: {exc}") from exc
if not isinstance(payload, dict):
    raise SystemExit(f"FATAL: ALPHA_GEO_PROVENANCE {geo_path} must be a JSON object")
if payload.get("diagnostic") != "alpha_geo_0_nerv_geometry":
    raise SystemExit("FATAL: ALPHA_GEO_PROVENANCE diagnostic must be alpha_geo_0_nerv_geometry")
if payload.get("pass_fail", {}).get("overall_pass") is not True:
    raise SystemExit("FATAL: Alpha-Geo geometry gate did not pass for RUN_AUTH_EVAL=1")
diagnostic_config = payload.get("diagnostic_config")
if not isinstance(diagnostic_config, dict):
    raise SystemExit("FATAL: Alpha-Geo provenance missing diagnostic_config")
if diagnostic_config.get("threshold_preset") != "promotion":
    raise SystemExit("FATAL: Alpha-Geo threshold_preset must be promotion for RUN_AUTH_EVAL=1")
inputs = payload.get("inputs")
if not isinstance(inputs, dict):
    raise SystemExit("FATAL: Alpha-Geo provenance missing inputs block")
candidate_source = inputs.get("candidate_source")
baseline_source = inputs.get("baseline_source")
if not isinstance(candidate_source, dict) or not isinstance(baseline_source, dict):
    raise SystemExit("FATAL: Alpha-Geo provenance missing source custody blocks")
candidate_sha = candidate_source.get("source_sha256")
baseline_sha = baseline_source.get("source_sha256")
if candidate_sha != sha256_file(archive):
    raise SystemExit(
        "FATAL: Alpha-Geo candidate SHA does not match archive built by this run: "
        f"{candidate_sha} != {sha256_file(archive)}"
    )
if baseline_sha != sha256_file(base_archive):
    raise SystemExit(
        "FATAL: Alpha-Geo baseline SHA does not match BASE_ARCHIVE: "
        f"{baseline_sha} != {sha256_file(base_archive)}"
    )
if inputs.get("candidate_member") != "masks.nrv":
    raise SystemExit("FATAL: Alpha-Geo candidate_member must resolve to masks.nrv")
if inputs.get("baseline_member") != "masks.mkv":
    raise SystemExit("FATAL: Alpha-Geo baseline_member must resolve to masks.mkv")
print("alpha_geo_gate:", json.dumps({"path": str(geo_path), "overall_pass": True}))
PY
fi

if [ "$RUN_AUTH_EVAL" != "1" ]; then
    log "=== LANE_12_NERV_BUILD_ONLY_DONE -- no CUDA auth eval by guardrail ==="
    log "archive candidate written to $ARCHIVE; run Alpha-Geo diagnostics and pose-regeneration review before exact eval."
    exit 0
fi

log "=== Stage 3: CUDA auth eval [contest-CUDA] ==="
EVAL_WORK_DIR="$LOG_DIR/eval_work"
rm -rf "$EVAL_WORK_DIR"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$ARCHIVE" \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir "$EVAL_WORK_DIR" \
    2>&1 | tee "$LOG_DIR/auth_eval.log"
PIPE_RC=("${PIPESTATUS[@]}")
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    echo "FATAL: previous pipeline exited rc=${PIPE_RC[0]}" >&2; exit "${PIPE_RC[0]}"
fi

CONTEST_JSON="$EVAL_WORK_DIR/contest_auth_eval.json"
[ -f "$CONTEST_JSON" ] || {
    log "FATAL: auth eval did not write $CONTEST_JSON; refusing log JSON scrape"
    exit 5
}

RESULT_JSON="$LOG_DIR/contest_auth_eval.json"
ADJUDICATION_LOG="$LOG_DIR/adjudication.log"
"$PYBIN" -u scripts/adjudicate_contest_auth_eval.py \
    --contest-json "$CONTEST_JSON" \
    --provenance "$PROVENANCE" \
    --archive "$ARCHIVE" \
    --result-copy "$RESULT_JSON" \
    --baseline-score "$BASELINE_SCORE" \
    --baseline-archive-bytes "$BASELINE_ARCHIVE_BYTES" \
    --predicted-band 0.95 1.30 \
    --regression-threshold 1.30 \
    --delta-key score_delta_vs_lane_g_v3 \
    --max-sane-score 100.0 | tee "$ADJUDICATION_LOG"
CANONICAL_SCORE=$(grep '^SCORE_RECOMPUTED=' "$ADJUDICATION_LOG" | tail -1 | cut -d= -f2)
[ -n "$CANONICAL_SCORE" ] || { log "FATAL: adjudicator did not emit SCORE_RECOMPUTED"; exit 5; }
cp "$RESULT_JSON" "$LOG_DIR/RESULT_JSON"
log "contest_auth_eval JSON written to $RESULT_JSON"
log "canonical score_recomputed=$CANONICAL_SCORE [contest-CUDA]"
log "=== LANE_12_NERV_DONE [contest-CUDA] -- see $RESULT_JSON ==="
