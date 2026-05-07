#!/usr/bin/env bash
# IMP-C067 bounded remote byte-screen driver.
#
# This script is intended to run on an already-claimed remote CUDA host. It
# builds deterministic IMP bridge candidates over the current C067/QZS3 anchor
# and stops at byte-screen evidence. It does not run contest_auth_eval.py and
# does not create a dispatch claim; operators must claim `imp_c067_bridge`
# before launching remote GPU work.

set -euo pipefail

if [ -z "${WORKSPACE:-}" ]; then
    if [ -d /teamspace/studios/this_studio/pact ]; then
        WORKSPACE="/teamspace/studios/this_studio/pact"
    else
        WORKSPACE="/workspace/pact"
    fi
fi
cd "$WORKSPACE"

if [ -f "$WORKSPACE/env.sh" ]; then
    # shellcheck source=/dev/null
    source "$WORKSPACE/env.sh"
fi

PYBIN="${PYBIN:-/opt/conda/bin/python}"
if [ ! -x "$PYBIN" ]; then
    if [ -x "$WORKSPACE/.venv/bin/python" ]; then
        PYBIN="$WORKSPACE/.venv/bin/python"
    else
        PYBIN="$(command -v python3 || true)"
    fi
fi
if [ -z "$PYBIN" ]; then
    echo "FATAL: no Python runtime found; set PYBIN" >&2
    exit 2
fi
export PYBIN

export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE:${PYTHONPATH:-}"
export TAC_UPSTREAM_DIR="${TAC_UPSTREAM_DIR:-$WORKSPACE/upstream}"

RUN_ID="${IMP_C067_RUN_ID:-imp_c067_bridge_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/results/${RUN_ID}}"

abs_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s\n' "$WORKSPACE/$1" ;;
    esac
}

OUT_DIR="$(abs_path "$OUTPUT_DIR")"
mkdir -p "$OUT_DIR"

LOG="$OUT_DIR/remote_lane_imp_c067_bridge.log"
HEARTBEAT="$OUT_DIR/heartbeat.log"
log() { echo "[imp-c067-bridge] $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

require_file() {
    local path="$1"
    if [ ! -f "$path" ]; then
        log "FATAL: missing required file: $path"
        exit 3
    fi
}

sha256_file() {
    sha256sum "$1" 2>/dev/null | cut -d ' ' -f 1 || shasum -a 256 "$1" | cut -d ' ' -f 1
}

size_file() {
    stat -c '%s' "$1" 2>/dev/null || stat -f '%z' "$1"
}

DEFAULT_C067_SOURCE_ARCHIVE="experiments/results/c067_breakthrough_candidate_matrix_20260502T1030Z/line_search_source_c067_fixedslice/archive.zip"
EXPECTED_C067_SOURCE_ARCHIVE_SHA256="${IMP_C067_EXPECTED_SOURCE_ARCHIVE_SHA256:-226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a}"
EXPECTED_C067_SOURCE_ARCHIVE_BYTES="${IMP_C067_EXPECTED_SOURCE_ARCHIVE_BYTES:-276214}"

SOURCE_ARCHIVE="${SOURCE_ARCHIVE:-${IMP_C067_SOURCE_ARCHIVE:-$DEFAULT_C067_SOURCE_ARCHIVE}}"
SOURCE_ARCHIVE_ABS="$(abs_path "$SOURCE_ARCHIVE")"
DEFAULT_C067_SOURCE_ARCHIVE_ABS="$(abs_path "$DEFAULT_C067_SOURCE_ARCHIVE")"

CANDIDATE_DIR="$OUT_DIR/candidates"
SUMMARY_JSON="$CANDIDATE_DIR/imp_c067_bridge_summary.json"
LAUNCHER_PROVENANCE="$OUT_DIR/launcher_provenance.json"
CLASSIFICATION_JSON="$OUT_DIR/byte_screen_classification.json"

log "=== Stage 0: NVDEC and source preflight ==="
require_file "$WORKSPACE/experiments/build_imp_c067_bridge_candidates.py"
require_file "$SOURCE_ARCHIVE_ABS"

SOURCE_LC="$(printf '%s\n' "$SOURCE_ARCHIVE" | tr '[:upper:]' '[:lower:]')"
if [[ "$SOURCE_LC" =~ lane[_-]?g|asym|asymmetric ]]; then
    if [ "${IMP_C067_ALLOW_OLD_ANCHOR:-0}" != "1" ]; then
        log "FATAL: refusing old Lane G/ASYM source anchor for current IMP-C067 bridge: $SOURCE_ARCHIVE"
        exit 4
    fi
fi

if [ "$SOURCE_ARCHIVE_ABS" != "$DEFAULT_C067_SOURCE_ARCHIVE_ABS" ]; then
    if [ "${IMP_C067_ALLOW_SOURCE_OVERRIDE:-0}" != "1" ]; then
        log "FATAL: source override disabled; default current C067 archive is $DEFAULT_C067_SOURCE_ARCHIVE"
        log "       set IMP_C067_ALLOW_SOURCE_OVERRIDE=1 only for an audited C067-compatible anchor"
        exit 4
    fi
fi

if [ "${IMP_C067_ALLOW_SOURCE_OVERRIDE:-0}" != "1" ]; then
    ACTUAL_SOURCE_SHA="$(sha256_file "$SOURCE_ARCHIVE_ABS")"
    ACTUAL_SOURCE_BYTES="$(size_file "$SOURCE_ARCHIVE_ABS")"
    if [ "$ACTUAL_SOURCE_SHA" != "$EXPECTED_C067_SOURCE_ARCHIVE_SHA256" ]; then
        log "FATAL: C067 source SHA mismatch expected=$EXPECTED_C067_SOURCE_ARCHIVE_SHA256 actual=$ACTUAL_SOURCE_SHA"
        exit 4
    fi
    if [ "$ACTUAL_SOURCE_BYTES" != "$EXPECTED_C067_SOURCE_ARCHIVE_BYTES" ]; then
        log "FATAL: C067 source size mismatch expected=$EXPECTED_C067_SOURCE_ARCHIVE_BYTES actual=$ACTUAL_SOURCE_BYTES"
        exit 4
    fi
fi

if [ "${IMP_C067_REQUIRE_NVDEC:-1}" = "1" ]; then
    require_file "$WORKSPACE/scripts/probe_nvdec.sh"
    bash "$WORKSPACE/scripts/probe_nvdec.sh" --ensure-dali || {
        log "FATAL: NVDEC probe failed; refusing remote spend on a host that cannot run later exact CUDA eval"
        exit 5
    }
else
    log "NVDEC probe skipped because IMP_C067_REQUIRE_NVDEC=0; output remains empirical byte-screen only"
fi

(
    while true; do
        if command -v nvidia-smi >/dev/null 2>&1; then
            GPU_STATUS="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\n' ' ')"
        else
            GPU_STATUS="nvidia-smi-unavailable"
        fi
        echo "$(date -u +%FT%TZ) run_id=$RUN_ID stage=byte_screen_non_score gpu=$GPU_STATUS" >> "$HEARTBEAT"
        sleep "${IMP_C067_HEARTBEAT_SECONDS:-60}"
    done
) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null || true' EXIT

log "=== Stage 0b: launcher provenance ==="
IMP_C067_RUN_ID="$RUN_ID" \
IMP_C067_OUT_DIR="$OUT_DIR" \
IMP_C067_SOURCE_ARCHIVE="$SOURCE_ARCHIVE_ABS" \
IMP_C067_DEFAULT_SOURCE_ARCHIVE="$DEFAULT_C067_SOURCE_ARCHIVE_ABS" \
IMP_C067_LAUNCHER_PROVENANCE="$LAUNCHER_PROVENANCE" \
"$PYBIN" - <<'PY'
import hashlib
import json
import os
import platform
import subprocess
import zipfile
from pathlib import Path


def meta(path: str) -> dict[str, object]:
    p = Path(path)
    return {
        "path": str(p),
        "bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }


repo = Path.cwd()
source = Path(os.environ["IMP_C067_SOURCE_ARCHIVE"])
with zipfile.ZipFile(source) as zf:
    zip_members = [
        {
            "name": info.filename,
            "size_bytes": info.file_size,
            "compressed_size_bytes": info.compress_size,
            "crc": f"{info.CRC:08x}",
        }
        for info in zf.infolist()
    ]

tracked = [
    "scripts/remote_lane_imp_c067_bridge.sh",
    "experiments/build_imp_c067_bridge_candidates.py",
    "experiments/build_blockfp_c067_archive.py",
    "src/tac/iterative_magnitude_pruning.py",
    "src/tac/quantizr_qzs3_codec.py",
    "src/tac/quantizr_faithful_renderer.py",
    "scripts/probe_nvdec.sh",
]
payload = {
    "schema": "imp_c067_bridge_remote_launcher_provenance_v1",
    "tool": "scripts/remote_lane_imp_c067_bridge.sh",
    "run_id": os.environ["IMP_C067_RUN_ID"],
    "score_claim": False,
    "promotion_eligible": False,
    "predicted_band": "empirical_byte_screen_only_no_score_predicted",
    "evidence_grade": "empirical_byte_screen_non_score",
    "classification": "empirical byte-screen only; non-score unless exact CUDA auth eval is later run",
    "required_score_truth": "archive.zip -> inflate.sh -> upstream/evaluate.py via experiments/contest_auth_eval.py --device cuda",
    "dispatch_claim_required_before_remote_launch": "tools/claim_lane_dispatch.py claim --lane-id imp_c067_bridge ...",
    "git_head": subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    ).stdout.strip()
    or None,
    "python": {
        "executable": os.environ.get("PYBIN"),
        "version": platform.python_version(),
    },
    "source_archive": {
        **meta(str(source)),
        "default_current_c067_archive": os.environ["IMP_C067_DEFAULT_SOURCE_ARCHIVE"],
        "zip_members": zip_members,
    },
    "source_files": {rel: meta(str(repo / rel)) for rel in tracked if (repo / rel).is_file()},
}
Path(os.environ["IMP_C067_LAUNCHER_PROVENANCE"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps({"launcher_provenance": os.environ["IMP_C067_LAUNCHER_PROVENANCE"], "score_claim": False}, sort_keys=True))
PY

log "=== Stage 1: build IMP-C067 bridge candidates (empirical byte-screen only) ==="
mkdir -p "$CANDIDATE_DIR"
BUILD_CMD=(
    "$PYBIN" -u experiments/build_imp_c067_bridge_candidates.py
    --source-archive "$SOURCE_ARCHIVE_ABS"
    --output-dir "$CANDIDATE_DIR"
    --cycle-counts "${IMP_C067_CYCLE_COUNTS:-1,2,5,10}"
    --sparsity-increment "${IMP_C067_SPARSITY_INCREMENT:-0.20}"
    --qzs3-block-sizes "${IMP_C067_QZS3_BLOCK_SIZES:-16,24,32,48,64,96,128}"
    --payload-member-name "${IMP_C067_PAYLOAD_MEMBER_NAME:-p}"
    --payload-format "${IMP_C067_PAYLOAD_FORMAT:-public_pr64_mask_first_len_table}"
    --brotli-quality "${IMP_C067_BROTLI_QUALITY:-11}"
)
if [ "${IMP_C067_FORCE:-0}" = "1" ]; then
    BUILD_CMD+=(--force)
fi

set +e
if command -v timeout >/dev/null 2>&1; then
    timeout "${IMP_C067_BUILD_TIMEOUT_SECONDS:-7200}" "${BUILD_CMD[@]}" 2>&1 | tee "$OUT_DIR/build_imp_c067_bridge_candidates.log"
else
    "${BUILD_CMD[@]}" 2>&1 | tee "$OUT_DIR/build_imp_c067_bridge_candidates.log"
fi
PIPE_RC=("${PIPESTATUS[@]}")
set -e
if [ "${PIPE_RC[0]}" -ne 0 ]; then
    log "FATAL: build_imp_c067_bridge_candidates.py failed rc=${PIPE_RC[0]}"
    exit "${PIPE_RC[0]}"
fi

require_file "$SUMMARY_JSON"

log "=== Stage 2: classify byte-screen outputs as non-score ==="
IMP_C067_SUMMARY_JSON="$SUMMARY_JSON" \
IMP_C067_CLASSIFICATION_JSON="$CLASSIFICATION_JSON" \
"$PYBIN" - <<'PY'
import json
import os
from pathlib import Path

summary_path = Path(os.environ["IMP_C067_SUMMARY_JSON"])
summary = json.loads(summary_path.read_text())
best = summary.get("best_by_output_archive_bytes", {})
cycles = summary.get("cycles", [])
candidate_count = sum(len(cycle.get("candidates", [])) for cycle in cycles)
payload = {
    "schema": "imp_c067_bridge_remote_byte_screen_classification_v1",
    "summary_json": str(summary_path),
    "score_claim": False,
    "promotion_eligible": False,
    "evidence_grade": "empirical_byte_screen_non_score",
    "classification": "EMPIRICAL BYTE-SCREEN ONLY - NON-SCORE",
    "canonical_score_source_required": "archive.zip -> inflate.sh -> upstream/evaluate.py via experiments/contest_auth_eval.py --device cuda",
    "exact_cuda_eval_later_required_for_any_score_claim": True,
    "safe_to_promote_rank_kill_from_this_run": False,
    "candidate_count": candidate_count,
    "source_archive": summary.get("source_archive"),
    "best_by_output_archive_bytes": {
        "candidate_id": best.get("candidate_id"),
        "archive_path": best.get("archive_path"),
        "archive_bytes": best.get("archive_bytes"),
        "archive_sha256": best.get("archive_sha256"),
        "delta_bytes_vs_source_archive": best.get("delta_bytes_vs_source_archive"),
        "formula_only_rate_delta_vs_source_archive": best.get("formula_only_rate_delta_vs_source_archive"),
    },
    "builder_decision": summary.get("decision"),
}
Path(os.environ["IMP_C067_CLASSIFICATION_JSON"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, sort_keys=True))
PY

log "IMP_C067_BRIDGE_DONE summary=$SUMMARY_JSON classification=$CLASSIFICATION_JSON"
log "EMPIRICAL BYTE-SCREEN ONLY - NON-SCORE; run exact CUDA auth eval later before any score claim"
