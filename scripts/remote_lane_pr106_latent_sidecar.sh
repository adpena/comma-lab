#!/bin/bash
# CUDA_REQUIRED — tensor-side codec build plus contest auth eval.
# Lane PR106 + latent sidecar — 28-dim x 600-pair latents corrected via per-pair (dim, delta_q)
#
# Planning target: reproduce the PR100-style sidecar gain on PR106. The default
# path is now score_table: build a CUDA scorer table over latent perturbation
# candidates, reduce measured improvements into charged sidecar bytes, then run
# exact CUDA auth eval on the emitted archive.
#
# Pipeline (4 stages, single Vast.ai 4090 ~$0.30/hr × 30min ≈ $0.30 OR
# Lightning T4 final auth eval ~$0.22/hr × 30min ≈ $0.11):
#
#   Stage 0 (CPU): Provenance + CUDA preflight + heartbeat
#   Stage 1a (CUDA): Build latent candidate score table when mode=score_table
#   Stage 1b (CUDA): Build PR106 + sidecar archive from score table
#   Stage 2 (CPU): Local parser-roundtrip sanity check
#   Stage 3 (CUDA-T4): contest_auth_eval — score must be < 0.20945 (PR106) to ship
#
# Strict-scorer-rule: scorer is loaded only by Stage 1a compress-time table
# generation and Stage 3 contest auth eval. Inflate-time has NO scorer
# dependency. Inflate-time only needs HNeRVDecoder + brotli.
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-/opt/conda/bin/python}"
LANE_ID="lane_pr106_latent_sidecar"
PR106_ARCHIVE="${PR106_ARCHIVE:-experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip}"
SIDECAR_TOP_K="${SIDECAR_TOP_K:-600}"
PR106_LATENT_MODE="${PR106_LATENT_MODE:-score_table}"
PR106_LATENT_DELTA_RADIUS="${PR106_LATENT_DELTA_RADIUS:-2}"
PR106_LATENT_N_PAIRS="${PR106_LATENT_N_PAIRS:-600}"
PR106_LATENT_DIM="${PR106_LATENT_DIM:-28}"
PR106_LATENT_SCORE_TABLE_BATCH_PAIRS="${PR106_LATENT_SCORE_TABLE_BATCH_PAIRS:-2}"
PR106_LATENT_SCORE_TABLE_CANDIDATE_BATCH_SIZE="${PR106_LATENT_SCORE_TABLE_CANDIDATE_BATCH_SIZE:-8}"
PR106_LATENT_SCORE_TABLE_RESUME="${PR106_LATENT_SCORE_TABLE_RESUME:-1}"
PR106_LATENT_SCORE_TABLE_NPY="${PR106_LATENT_SCORE_TABLE_NPY:-}"
PR106_LATENT_SCORE_TABLE_MANIFEST="${PR106_LATENT_SCORE_TABLE_MANIFEST:-}"
PR106_LATENT_ALLOW_PROVIDER_CLAIM_MIRROR="${PR106_LATENT_ALLOW_PROVIDER_CLAIM_MIRROR:-0}"
PR106_LATENT_SCORE_TABLE_LANE_ID="${PR106_LATENT_SCORE_TABLE_LANE_ID:-$LANE_ID}"
PR106_LATENT_SCORE_TABLE_INSTANCE_JOB_ID="${PR106_LATENT_SCORE_TABLE_INSTANCE_JOB_ID:-${INSTANCE_JOB_ID:-}}"
PR106_RUNTIME_DIR="${PR106_RUNTIME_DIR:-submissions/pr106_latent_sidecar_r2_pr101_grammar}"
PR106_ARCHIVE_MEMBER="${PR106_ARCHIVE_MEMBER:-x}"
PR106_EXPECTED_ARCHIVE_SHA256="${PR106_EXPECTED_ARCHIVE_SHA256:-56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7}"
PR106_EXPECTED_ARCHIVE_MEMBER_SHA256="${PR106_EXPECTED_ARCHIVE_MEMBER_SHA256:-852a4cb1231413cf1a8fc867e2a808de9ec78511d2ebf283df2c5b608cb4a749}"
CLOUD_PLATFORM="${CLOUD_PLATFORM:-unknown}"
TAC_UPSTREAM_COMMIT="${TAC_UPSTREAM_COMMIT:-unknown}"

[ -f "$WORKSPACE/env.sh" ] && source "$WORKSPACE/env.sh"

cd "$WORKSPACE"

LOG_DIR="${PR106_LATENT_LOG_DIR:-$WORKSPACE/experiments/results/${LANE_ID}_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$LOG_DIR"
log() { echo "[lane-pr106-sidecar] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

# CPU source/runtime preflight before any CUDA work. This blocks stale PR106
# packets, implicit member inference, and missing runtime trees before spend.
SOURCE_PREFLIGHT_JSON="$LOG_DIR/source_preflight.json"
PR106_ARCHIVE="$PR106_ARCHIVE" \
PR106_ARCHIVE_MEMBER="$PR106_ARCHIVE_MEMBER" \
PR106_EXPECTED_ARCHIVE_SHA256="$PR106_EXPECTED_ARCHIVE_SHA256" \
PR106_EXPECTED_ARCHIVE_MEMBER_SHA256="$PR106_EXPECTED_ARCHIVE_MEMBER_SHA256" \
PR106_RUNTIME_DIR="$PR106_RUNTIME_DIR" \
PR106_SOURCE_PREFLIGHT_JSON="$SOURCE_PREFLIGHT_JSON" \
"$PYBIN" - <<'PY'
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

archive = Path(os.environ["PR106_ARCHIVE"])
archive_member = os.environ["PR106_ARCHIVE_MEMBER"]
expected_archive_sha256 = os.environ["PR106_EXPECTED_ARCHIVE_SHA256"]
expected_archive_member_sha256 = os.environ["PR106_EXPECTED_ARCHIVE_MEMBER_SHA256"]
runtime_dir = Path(os.environ["PR106_RUNTIME_DIR"])
output_json = Path(os.environ["PR106_SOURCE_PREFLIGHT_JSON"])
errors: list[str] = []
record: dict[str, object] = {
    "schema": "pr106_latent_source_preflight_v1",
    "pr106_archive": str(archive),
    "archive_member": archive_member,
    "expected_archive_sha256": expected_archive_sha256,
    "expected_archive_member_sha256": expected_archive_member_sha256,
    "runtime_dir": str(runtime_dir),
}

if not archive.is_file():
    errors.append(f"missing PR106 archive: {archive}")
else:
    record["source_archive_bytes"] = archive.stat().st_size
    record["source_archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    if record["source_archive_sha256"] != expected_archive_sha256:
        errors.append(
            f"PR106 archive SHA mismatch: got {record['source_archive_sha256']} "
            f"expected {expected_archive_sha256}"
        )
    try:
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            record["zip_members"] = names
            if archive_member not in names:
                errors.append(
                    f"archive member {archive_member!r} not found in {archive}; "
                    f"members={names}"
                )
            else:
                payload = zf.read(archive_member)
                record["source_archive_member_bytes"] = len(payload)
                record["source_archive_member_sha256"] = hashlib.sha256(payload).hexdigest()
                if record["source_archive_member_sha256"] != expected_archive_member_sha256:
                    errors.append(
                        "PR106 archive member SHA mismatch: "
                        f"got {record['source_archive_member_sha256']} "
                        f"expected {expected_archive_member_sha256}"
                    )
    except zipfile.BadZipFile as exc:
        errors.append(f"invalid PR106 archive zip {archive}: {exc}")

if not runtime_dir.is_dir():
    errors.append(f"missing PR106 runtime dir: {runtime_dir}")
else:
    missing_runtime_files = [
        str(path)
        for path in (runtime_dir / "inflate.py", runtime_dir / "inflate.sh")
        if not path.is_file()
    ]
    if missing_runtime_files:
        errors.append(f"missing PR106 runtime files: {missing_runtime_files}")
    record["runtime_files_checked"] = ["inflate.py", "inflate.sh"]

record["ok"] = not errors
record["errors"] = errors
output_json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
if errors:
    for error in errors:
        print(f"FATAL: {error}", file=sys.stderr)
    raise SystemExit(3)
print(
    "[stage-0] source preflight OK: "
    f"archive={archive} member={archive_member} runtime={runtime_dir}",
    flush=True,
)
PY

# Stage 0: NVDEC probe — required by preflight check_remote_scripts_have_nvdec_probe.
# probe MUST come before any GPU-work marker including bare `nvidia-smi`.
if [ "${SKIP_NVDEC_PROBE:-0}" != "1" ] && [ -f "$WORKSPACE/scripts/probe_nvdec.sh" ]; then
    bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
        log "FATAL: NVDEC/DALI probe failed; exact CUDA eval is not trustworthy on this host."
        exit 2
    }
fi

# Heartbeat (per CLAUDE.md remote-code-parity rule)
HEARTBEAT="$LOG_DIR/heartbeat.log"
( while true; do echo "$(date -u +%FT%TZ) lane-pr106-sidecar alive" >> "$HEARTBEAT"; sleep 60; done ) &
HB_PID=$!
trap "kill $HB_PID 2>/dev/null || true" EXIT

# ── Stage 0: Provenance + CUDA preflight ──────────────────────────────────
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
GPU_NAME="${GPU_NAME:-no-gpu}"
PR106_STAGE0_LANE_ID="$LANE_ID" \
PR106_STAGE0_GIT_HASH="$GIT_HASH" \
PR106_STAGE0_GPU_NAME="$GPU_NAME" \
PR106_STAGE0_ARCHIVE="$PR106_ARCHIVE" \
PR106_STAGE0_LATENT_MODE="$PR106_LATENT_MODE" \
PR106_STAGE0_DELTA_RADIUS="$PR106_LATENT_DELTA_RADIUS" \
PR106_STAGE0_SCORE_TABLE_LANE_ID="$PR106_LATENT_SCORE_TABLE_LANE_ID" \
PR106_STAGE0_INSTANCE_JOB_ID="$PR106_LATENT_SCORE_TABLE_INSTANCE_JOB_ID" \
PR106_STAGE0_RUNTIME_DIR="$PR106_RUNTIME_DIR" \
PR106_STAGE0_ARCHIVE_MEMBER="$PR106_ARCHIVE_MEMBER" \
PR106_STAGE0_CLOUD_PLATFORM="$CLOUD_PLATFORM" \
PR106_STAGE0_UPSTREAM_COMMIT="$TAC_UPSTREAM_COMMIT" \
PR106_STAGE0_SIDECAR_TOP_K="$SIDECAR_TOP_K" \
PR106_STAGE0_PROVENANCE_JSON="$LOG_DIR/provenance.json" \
"$PYBIN" - <<'PY'
import json
import os
import sys
import time

import torch

if not torch.cuda.is_available():
    sys.exit('FATAL: --device cuda required per CLAUDE.md MPS-auth-eval-is-NOISE')
prov = {
    'lane_id': os.environ['PR106_STAGE0_LANE_ID'],
    'predicted_band': [0.205, 0.208],
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': os.environ['PR106_STAGE0_GIT_HASH'],
    'gpu_name': os.environ['PR106_STAGE0_GPU_NAME'],
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'pr106_archive': os.environ['PR106_STAGE0_ARCHIVE'],
    'latent_mode': os.environ['PR106_STAGE0_LATENT_MODE'],
    'latent_delta_radius': int(os.environ['PR106_STAGE0_DELTA_RADIUS']),
    'latent_score_table_lane_id': os.environ['PR106_STAGE0_SCORE_TABLE_LANE_ID'],
    'latent_score_table_instance_job_id': os.environ['PR106_STAGE0_INSTANCE_JOB_ID'],
    'pr106_runtime_dir': os.environ['PR106_STAGE0_RUNTIME_DIR'],
    'pr106_archive_member': os.environ['PR106_STAGE0_ARCHIVE_MEMBER'],
    'cloud_platform': os.environ['PR106_STAGE0_CLOUD_PLATFORM'],
    'upstream_commit': os.environ['PR106_STAGE0_UPSTREAM_COMMIT'],
    'sidecar_top_k': int(os.environ['PR106_STAGE0_SIDECAR_TOP_K']),
}
with open(os.environ['PR106_STAGE0_PROVENANCE_JSON'], 'w') as f:
    json.dump(prov, f, indent=2)
print(
    f"[stage-0] provenance written; CUDA={torch.cuda.is_available()}; "
    f"top_k={prov['sidecar_top_k']}"
)
PY

# ── Stage 1: Build score table + PR106 sidecar archive ────────────────────
log "=== Stage 1: build PR106 + latent-correction sidecar (mode=$PR106_LATENT_MODE, top_k=$SIDECAR_TOP_K) ==="
BUILD_DIR="$LOG_DIR/build"
mkdir -p "$BUILD_DIR"

if [ "$PR106_LATENT_MODE" = "score_table" ] && [ -z "$PR106_LATENT_SCORE_TABLE_NPY" ]; then
    log "=== Stage 1a: generate CUDA latent score table ==="
    if [ -z "$PR106_LATENT_SCORE_TABLE_INSTANCE_JOB_ID" ]; then
        log "FATAL: PR106_LATENT_SCORE_TABLE_INSTANCE_JOB_ID is required for score_table mode"
        exit 3
    fi
    if [ "$PR106_LATENT_ALLOW_PROVIDER_CLAIM_MIRROR" = "1" ]; then
        PR106_PROVIDER_CLAIM_LANE_ID="$PR106_LATENT_SCORE_TABLE_LANE_ID" \
        PR106_PROVIDER_CLAIM_JOB_ID="$PR106_LATENT_SCORE_TABLE_INSTANCE_JOB_ID" \
        PR106_PROVIDER_CLAIM_PLATFORM="$CLOUD_PLATFORM" \
        "$PYBIN" - <<'PY'
import os
from pathlib import Path

from tac.sidechannel_score_table import mirror_provider_local_active_claim

row = mirror_provider_local_active_claim(
    Path(".omx/state/active_lane_dispatch_claims.md"),
    lane_id=os.environ["PR106_PROVIDER_CLAIM_LANE_ID"],
    instance_job_id=os.environ["PR106_PROVIDER_CLAIM_JOB_ID"],
    platform=os.environ["PR106_PROVIDER_CLAIM_PLATFORM"],
    agent="provider-runtime:pr106-latent-score-table",
    notes="provider-local mirror; local pre-dispatch claim remains source of truth",
)
print(
    "[stage-1a] provider-local claim mirror active: "
    f"lane_id={row['lane_id']} job={row['instance_job_id']} platform={row['platform']}",
    flush=True,
)
PY
    fi
    SCORE_TABLE_DIR="$LOG_DIR/score_table"
    mkdir -p "$SCORE_TABLE_DIR"
    PR106_LATENT_SCORE_TABLE_NPY="$SCORE_TABLE_DIR/score_table.npy"
    PR106_LATENT_SCORE_TABLE_MANIFEST="$SCORE_TABLE_DIR/score_table_manifest.json"
    if [ "$PR106_LATENT_SCORE_TABLE_RESUME" = "1" ] && [ -f "$PR106_LATENT_SCORE_TABLE_NPY" ] && [ -f "$PR106_LATENT_SCORE_TABLE_MANIFEST" ]; then
        log "Stage 1a RESUME: validating completed latent score table at $PR106_LATENT_SCORE_TABLE_NPY"
    fi
    SCORE_TABLE_ARGS=(
        experiments/build_pr106_latent_score_table.py
        --pr106-archive "$PR106_ARCHIVE"
        --out-dir "$SCORE_TABLE_DIR"
        --delta-radius "$PR106_LATENT_DELTA_RADIUS"
        --latent-dim "$PR106_LATENT_DIM"
        --n-pairs "$PR106_LATENT_N_PAIRS"
        --batch-pairs "$PR106_LATENT_SCORE_TABLE_BATCH_PAIRS"
        --candidate-batch-size "$PR106_LATENT_SCORE_TABLE_CANDIDATE_BATCH_SIZE"
        --lane-id "$PR106_LATENT_SCORE_TABLE_LANE_ID"
        --instance-job-id "$PR106_LATENT_SCORE_TABLE_INSTANCE_JOB_ID"
        --runtime-dir "$PR106_RUNTIME_DIR"
    )
    if [ -n "$PR106_ARCHIVE_MEMBER" ]; then
        SCORE_TABLE_ARGS+=(--archive-member "$PR106_ARCHIVE_MEMBER")
    fi
    if [ "$PR106_LATENT_SCORE_TABLE_RESUME" = "1" ]; then
        SCORE_TABLE_ARGS+=(--resume-checkpoint)
    fi
    "$PYBIN" -u "${SCORE_TABLE_ARGS[@]}" 2>&1 | tee -a "$LOG_DIR/run.log"
    if [ ! -f "$PR106_LATENT_SCORE_TABLE_NPY" ] || [ ! -f "$PR106_LATENT_SCORE_TABLE_MANIFEST" ]; then
        log "FATAL: latent score-table generation did not produce table+manifest"
        exit 3
    fi
fi

if [ "$PR106_LATENT_MODE" = "score_table" ]; then
    BUILD_ARGS=(
        tools/materialize_pr106_latent_score_table_candidate.py
        --source-archive "$PR106_ARCHIVE"
        --output-dir "$BUILD_DIR"
        --top-k "$SIDECAR_TOP_K"
        --delta-radius "$PR106_LATENT_DELTA_RADIUS"
        --score-table-npy "$PR106_LATENT_SCORE_TABLE_NPY"
        --python-executable "$PYBIN"
    )
    if [ -n "$PR106_LATENT_SCORE_TABLE_MANIFEST" ]; then
        BUILD_ARGS+=(--score-table-manifest "$PR106_LATENT_SCORE_TABLE_MANIFEST")
    fi
else
    BUILD_ARGS=(
        experiments/build_pr106_latent_sidecar.py
        --source-archive "$PR106_ARCHIVE"
        --output-dir "$BUILD_DIR"
        --top-k "$SIDECAR_TOP_K"
        --device cuda
        --search-mode "$PR106_LATENT_MODE"
        --delta-radius "$PR106_LATENT_DELTA_RADIUS"
    )
fi
"$PYBIN" -u "${BUILD_ARGS[@]}" 2>&1 | tee -a "$LOG_DIR/run.log"
SIDECAR_ARCHIVE="$BUILD_DIR/sidecar_archive.zip"
if [ ! -f "$SIDECAR_ARCHIVE" ]; then
    log "FATAL: stage 1 did not produce $SIDECAR_ARCHIVE"
    exit 3
fi
ARCHIVE_BYTES=$(stat -c '%s' "$SIDECAR_ARCHIVE" 2>/dev/null || stat -f '%z' "$SIDECAR_ARCHIVE")
log "stage 1 OK: archive bytes=$ARCHIVE_BYTES"
EXPECTED_ARCHIVE_SHA=$("$PYBIN" - "$SIDECAR_ARCHIVE" <<'PY'
import hashlib
import sys
from pathlib import Path

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)
EXPECTED_RUNTIME_TREE_SHA=$("$PYBIN" - "$WORKSPACE/src" "$PR106_RUNTIME_DIR" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from tac.packet_compiler.pr106_runtime_consumption import pr106_runtime_source_manifest

print(pr106_runtime_source_manifest(Path(sys.argv[2]))["runtime_source_tree_sha256"])
PY
)
if [ -z "$EXPECTED_ARCHIVE_SHA" ] || [ -z "$EXPECTED_RUNTIME_TREE_SHA" ]; then
    log "FATAL: failed to compute expected archive/runtime SHA custody"
    exit 3
fi

# ── Stage 2: Local parser-roundtrip sanity ────────────────────────────────
log "=== Stage 2: parser-roundtrip sanity (no GPU forward) ==="
"$PYBIN" -u tools/prove_pr106_sidecar_runtime_consumption.py \
    --archive "$SIDECAR_ARCHIVE" \
    --runtime-dir "$PR106_RUNTIME_DIR" \
    --expected-archive-sha256 "$EXPECTED_ARCHIVE_SHA" \
    --expected-runtime-source-tree-sha256 "$EXPECTED_RUNTIME_TREE_SHA" \
    --output-json "$LOG_DIR/runtime_consumption.json" 2>&1 | tee -a "$LOG_DIR/run.log"

# ── Stage 3: Contest auth eval (CUDA T4) ─────────────────────────────────
if [ "$CLOUD_PLATFORM" = "kaggle" ]; then
    AUTH_AXIS_LABEL="[provider-CUDA:kaggle advisory]"
else
    AUTH_AXIS_LABEL="[contest-CUDA]"
fi
log "=== Stage 3: auth eval (CUDA; axis=$AUTH_AXIS_LABEL) ==="
INFLATE_SH="$WORKSPACE/$PR106_RUNTIME_DIR/inflate.sh"
EVAL_DIR="$LOG_DIR/eval"
mkdir -p "$EVAL_DIR"
"$PYBIN" -u experiments/contest_auth_eval.py \
    --archive "$SIDECAR_ARCHIVE" \
    --inflate-sh "$INFLATE_SH" \
    --work-dir "$EVAL_DIR" \
    --keep-work-dir \
    --device cuda 2>&1 | tee -a "$LOG_DIR/run.log"

# ── Final summary ─────────────────────────────────────────────────────────
SCORE_JSON="$EVAL_DIR/contest_auth_eval.json"
if [ -f "$SCORE_JSON" ]; then
    AUTH_SUMMARY=$("$PYBIN" -m tac.auth_eval_schema completion-summary "$SCORE_JSON")
    SCORE=$("$PYBIN" -m tac.auth_eval_schema completion-summary "$SCORE_JSON" --field score 2>/dev/null || echo "PARSE_FAIL")
    log "DONE $AUTH_AXIS_LABEL: lane=$LANE_ID archive_bytes=$ARCHIVE_BYTES auth_eval_summary=$AUTH_SUMMARY"
    if [ "$AUTH_AXIS_LABEL" = "[contest-CUDA]" ]; then
        log "  beats PR106 baseline 0.20946? $("$PYBIN" -c "
s = $SCORE
print('YES — new public-frontier candidate' if isinstance(s, (int, float)) and s < 0.20946 else f'no (score {s} >= 0.20946)')
" 2>/dev/null || echo "?")"
    else
        log "  provider advisory only: no public-frontier or promotion claim until paired contest-axis adjudication"
    fi
else
    log "FATAL: stage 3 did not produce $SCORE_JSON"
    exit 4
fi
