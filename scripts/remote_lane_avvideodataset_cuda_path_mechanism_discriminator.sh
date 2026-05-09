#!/usr/bin/env bash
# Dispatch runbook for the AVVideoDataset CUDA-CPU drift mechanism discriminator.
#
# Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first": this
# runbook IS the parallel-dispatch actuator for the discriminator lane. It
# fans out 4 GHA-CPU dispatches (one per variant) and 4 CUDA dispatches.
# Total budget: ~$0 for CPU (free GHA minutes) + ~$0.80-$1.20 for CUDA.
#
# Usage:
#   bash scripts/remote_lane_avvideodataset_cuda_path_mechanism_discriminator.sh
#
# Required env vars:
#   DISCRIMINATOR_TIMESTAMP_SUFFIX  — e.g. 20260509T110211Z (the timestamp
#                                     used when the variants were built)
#   GH_TOKEN                        — GitHub token (auto-discovered from `gh auth status`)
#
# Optional env vars:
#   DRY_RUN=1                       — print commands but don't execute
#   SKIP_CPU=1                      — skip the 4 GHA CPU dispatches
#   SKIP_CUDA=1                     — skip the 4 CUDA dispatches (default: skip
#                                     until operator explicitly enables CUDA)
#   CUDA_PROVIDER=lightning         — one of {lightning, vastai, modal}
#                                     (only used when SKIP_CUDA != 1)
#
# Output:
#   experiments/results/a1_cuda_cpu_drift_discriminator_<variant>_<ts>/
#     gha_dispatch/contest_auth_eval.adjudicated.json   (CPU eval results)
#     cuda_dispatch/contest_auth_eval.json              (CUDA eval results)
#
# Per CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION": this runbook claims the
# lane via tools/claim_lane_dispatch.py BEFORE any dispatch.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
rm -f upstream/videos/._*.mkv
export PYTHON_INFLATE=renderer
if [ "${PYTHON_INFLATE}" != "renderer" ]; then
  echo "FATAL: PYTHON_INFLATE must be renderer for contest auth eval" >&2
  exit 2
fi

VARIANTS=(
  "v_baseline"
  "v_loader_isolated"
  "v_conv_isolated"
  "v_hydra_isolated"
)

if [ -z "${DISCRIMINATOR_TIMESTAMP_SUFFIX:-}" ]; then
  echo "FATAL: DISCRIMINATOR_TIMESTAMP_SUFFIX env var must be set" >&2
  echo "       (this is the timestamp the variants were built with;" >&2
  echo "        e.g. ls experiments/results/ | grep a1_cuda_cpu_drift_discriminator | head -4)" >&2
  exit 2
fi
TS="${DISCRIMINATOR_TIMESTAMP_SUFFIX}"

# Default behavior: CUDA dispatch is OPT-IN to avoid burning GPU budget
# without explicit operator approval. CPU dispatch is opt-out (free).
SKIP_CUDA="${SKIP_CUDA:-1}"
SKIP_CPU="${SKIP_CPU:-0}"
CUDA_PROVIDER="${CUDA_PROVIDER:-lightning}"
DRY_RUN="${DRY_RUN:-0}"
RUN_RECORD_DIR="${RUN_RECORD_DIR:-experiments/results/a1_cuda_cpu_drift_discriminator_run_${TS}}"
PROVENANCE="$RUN_RECORD_DIR/provenance.json"
HEARTBEAT="$RUN_RECORD_DIR/heartbeat.log"
RUN_RECORD="$RUN_RECORD_DIR/run_record.json"
mkdir -p "$RUN_RECORD_DIR"
cat > "$PROVENANCE" <<EOF
{
  "schema_version": "remote_run_provenance.v1",
  "lane_id": "lane_avvideodataset_cuda_path_mechanism_discriminator",
  "timestamp_suffix": "${TS}",
  "script": "scripts/remote_lane_avvideodataset_cuda_path_mechanism_discriminator.sh",
  "git_head": "$(git rev-parse HEAD 2>/dev/null || echo unknown)",
  "skip_cpu": "${SKIP_CPU}",
  "skip_cuda": "${SKIP_CUDA}",
  "cuda_provider": "${CUDA_PROVIDER}",
  "dry_run": "${DRY_RUN}",
  "predicted_band": [0.190, 0.230],
  "prediction_scope": "mechanism-discriminator variants; no promotion without paired exact CPU/CUDA custody",
  "score_claim": false
}
EOF
cat > "$RUN_RECORD" <<EOF
{
  "schema_version": "remote_run_record.v1",
  "status": "started",
  "started_at_utc": "$(date -u +%FT%TZ)",
  "provenance_json": "${PROVENANCE}",
  "heartbeat_log": "${HEARTBEAT}"
}
EOF
echo "$(date -u +%FT%TZ) heartbeat stage=start pid=$$" >> "$HEARTBEAT"

run() {
  echo "+ $*"
  if [ "$DRY_RUN" != "1" ]; then
    "$@"
  fi
}

# Stage 0: NVDEC probe before any opt-in CUDA dispatch path.
#
# MEDIUM 1 FIX (codex round 3, 2026-05-09): the probe was previously
# unconditionally invoked locally whenever SKIP_CUDA=0, including under
# DRY_RUN=1. The operator workstation / CI host typically does NOT have
# CUDA / NVDEC / DALI installed; the probe exited 2 before this script
# could reach the dispatch-decision block. The probe belongs in the
# REMOTE provider bootstrap (where the GPU actually lives), not in the
# local dispatch driver.
#
# Guard semantics:
#   - DRY_RUN=1                       → skip probe (planning, not running)
#   - LOCAL_CUDA_WORKER!=1 (default)  → skip probe (this host is a
#                                       dispatcher, not the GPU worker)
#   - LOCAL_CUDA_WORKER=1 (explicit)  → probe runs (operator opted in)
#   - SKIP_NVDEC_PROBE=1              → skip probe (escape valve)
#
# This keeps the dispatch path runnable on macOS / Linux x86_64 CPU /
# any host without CUDA, while still catching the missing-NVDEC class
# of bug on hosts that legitimately should have it (a single
# LOCAL_CUDA_WORKER=1 invocation reproduces the original behaviour).
#
# Memory: feedback_codex_round3_findings_fix_landed_20260509.md
LOCAL_CUDA_WORKER="${LOCAL_CUDA_WORKER:-0}"
if [ "$SKIP_CUDA" != "1" ] \
   && [ "${SKIP_NVDEC_PROBE:-0}" != "1" ] \
   && [ "$DRY_RUN" != "1" ] \
   && [ "$LOCAL_CUDA_WORKER" = "1" ]; then
  echo "=== Stage 0: NVDEC probe before CUDA dispatch (LOCAL_CUDA_WORKER=1) ==="
  bash "${REPO_ROOT}/scripts/probe_nvdec.sh" || {
    echo "FATAL: NVDEC probe failed; refusing CUDA dispatch on this host" >&2
    exit 2
  }
elif [ "$SKIP_CUDA" != "1" ]; then
  echo "=== Stage 0: NVDEC probe SKIPPED (DRY_RUN=${DRY_RUN}, LOCAL_CUDA_WORKER=${LOCAL_CUDA_WORKER}, SKIP_NVDEC_PROBE=${SKIP_NVDEC_PROBE:-0}) ==="
  echo "    NVDEC probing belongs in the remote provider bootstrap."
  echo "    Set LOCAL_CUDA_WORKER=1 to force-probe the local host."
fi

# 1. Claim the lane (single claim covers all 4 variants since they are the
#    same lane_id, dispatched as one sweep).
echo "=== claiming dispatch lane ==="
run .venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_avvideodataset_cuda_path_mechanism_discriminator \
  --platform "github_actions+${CUDA_PROVIDER}" \
  --instance-job-id "discriminator-sweep-${TS}" \
  --agent "subagent_avvideodataset_cuda_path_mechanism_discriminator" \
  --status "eval" \
  --notes "AVVideoDataset CUDA-CPU drift mechanism discriminator: 4 variants × {CPU GHA, CUDA ${CUDA_PROVIDER}}"

# 2. CPU dispatches via GHA (free).
if [ "$SKIP_CPU" != "1" ]; then
  echo "=== dispatching 4 GHA CPU evals ==="
  for variant in "${VARIANTS[@]}"; do
    VARIANT_DIR="experiments/results/a1_cuda_cpu_drift_discriminator_${variant}_${TS}"
    SUB_DIR="${VARIANT_DIR}/submission_dir"
    if [ ! -d "${SUB_DIR}" ]; then
      echo "FATAL: ${SUB_DIR} not found; build the variants first via" >&2
      echo "  .venv/bin/python tools/build_a1_cuda_cpu_drift_discriminator_variants.py --timestamp ${TS}" >&2
      exit 2
    fi
    ARCHIVE_SHA="$(.venv/bin/python -c "
import hashlib; h=hashlib.sha256()
open('${SUB_DIR}/archive.zip','rb').read() and None
import sys
data=open('${SUB_DIR}/archive.zip','rb').read()
print(hashlib.sha256(data).hexdigest())
")"
    SUBMISSION_NAME="a1_cuda_cpu_drift_discriminator_${variant}_${TS}"
    OUTPUT_DIR="${VARIANT_DIR}/gha_dispatch"
    mkdir -p "${OUTPUT_DIR}"
    echo "--- variant ${variant} (CPU GHA) ---"
    run .venv/bin/python tools/dispatch_cpu_eval_via_github_actions.py \
      --archive-path "${SUB_DIR}/archive.zip" \
      --archive-sha "${ARCHIVE_SHA}" \
      --submission-name "${SUBMISSION_NAME}" \
      --submission-dir "${SUB_DIR}" \
      --auto-create-fork-pr \
      --output-dir "${OUTPUT_DIR}"
  done
fi

# 3. CUDA dispatches (opt-in; defaults to skip).
if [ "$SKIP_CUDA" != "1" ]; then
  echo "=== dispatching 4 CUDA evals via ${CUDA_PROVIDER} ==="
  case "${CUDA_PROVIDER}" in
    lightning)
      for variant in "${VARIANTS[@]}"; do
        VARIANT_DIR="experiments/results/a1_cuda_cpu_drift_discriminator_${variant}_${TS}"
        SUB_DIR="${VARIANT_DIR}/submission_dir"
        echo "--- variant ${variant} (CUDA Lightning) ---"
        # The canonical Lightning T4 dispatch wrapper:
        #   tools/lightning_dispatch_pr106_stack.py
        # adapt for arbitrary submission_dir (Lightning will run inflate.sh +
        # upstream/evaluate.py --device cuda; harvested JSONL contains the
        # auth_eval components). Per CLAUDE.md "NEVER invent CLI flags":
        # operator must verify the wrapper accepts --submission-dir before use.
        echo "OPERATOR DECISION REQUIRED: select Lightning dispatch wrapper"
        echo "  candidate: tools/lightning_dispatch_pr106_stack.py"
        echo "  submission_dir: ${SUB_DIR}"
        echo "  expected cost: ~\$0.20 / variant on T4 g4dn.2xlarge"
        echo "  (subagent context cannot select a wrapper without operator approval"
        echo "   per CLAUDE.md 'NEVER invent CLI flags' rule)"
      done
      ;;
    vastai)
      for variant in "${VARIANTS[@]}"; do
        VARIANT_DIR="experiments/results/a1_cuda_cpu_drift_discriminator_${variant}_${TS}"
        SUB_DIR="${VARIANT_DIR}/submission_dir"
        echo "--- variant ${variant} (CUDA Vast.ai 4090) ---"
        echo "OPERATOR DECISION REQUIRED: select Vast.ai launcher (scripts/launch_lane_on_vastai.py?)"
        echo "  submission_dir: ${SUB_DIR}"
        echo "  expected cost: ~\$0.20 / variant on RTX 4090 (~\$0.25/hr × 0.8 hr)"
      done
      ;;
    modal)
      for variant in "${VARIANTS[@]}"; do
        VARIANT_DIR="experiments/results/a1_cuda_cpu_drift_discriminator_${variant}_${TS}"
        SUB_DIR="${VARIANT_DIR}/submission_dir"
        echo "--- variant ${variant} (CUDA Modal A100) ---"
        echo "OPERATOR DECISION REQUIRED: select Modal app entry"
        echo "  submission_dir: ${SUB_DIR}"
        echo "  expected cost: ~\$0.30 / variant on A100"
      done
      ;;
    *)
      echo "FATAL: unsupported CUDA_PROVIDER=${CUDA_PROVIDER}; expected lightning/vastai/modal" >&2
      exit 2
      ;;
  esac
fi

# 4. After all dispatches return, run the verdict analyzer.
echo ""
echo "=== verdict-analysis recipe (run after all dispatches return) ==="
echo ""
cat <<EOF
.venv/bin/python tools/analyze_a1_cuda_cpu_drift_discriminator_verdict.py \\
EOF
for variant in "${VARIANTS[@]}"; do
  VARIANT_DIR="experiments/results/a1_cuda_cpu_drift_discriminator_${variant}_${TS}"
  cat <<EOF
  --variant-dir "${VARIANT_DIR}" \\
  --cpu-eval "${VARIANT_DIR}/gha_dispatch/contest_auth_eval.adjudicated.json" \\
  --cuda-eval "${VARIANT_DIR}/cuda_dispatch/contest_auth_eval.json" \\
EOF
done
cat <<EOF
  --output-dir experiments/results/a1_cuda_cpu_drift_discriminator_verdict_${TS}/
EOF

echo ""
echo "=== runbook complete ==="
echo "  CPU command emitted: $([ "$SKIP_CPU" != "1" ] && echo "yes" || echo "skipped")"
echo "  CUDA command emitted: $([ "$SKIP_CUDA" != "1" ] && echo "yes (${CUDA_PROVIDER})" || echo "skipped")"
echo "  CUDA result axis tag when exact CUDA artifacts are harvested: [contest-CUDA]"
