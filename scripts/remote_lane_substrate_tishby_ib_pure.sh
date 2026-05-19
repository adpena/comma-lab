#!/bin/bash
# Remote lane script: substrate Tishby IB-pure (PRIMARY variational
# Information Bottleneck) dispatch.
#
# Trainer: experiments/train_substrate_tishby_ib_pure.py
# Lane: lane_tishby_ib_pure_l1_scaffold_substrate_build_plus_probes_20260516
# Design memo: .omx/research/tishby_ib_pure_substrate_asymptotic_pursuit_scoping_design_20260516.md
#
# L1 SCAFFOLD landing: the recipe carries research_only=true +
# dispatch_enabled=false; the trainer's _full_main raises NotImplementedError
# per Catalog #240 cascade. This driver script exists so the operator-authorize
# canonical chain can validate the lane structure but REFUSES TO LAUNCH the
# trainer until Phase 2 council approval lifts the research_only flag.
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" this script
# DELEGATES bootstrap to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() function. Per Catalog #163 the canonical
# REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 sentinel is set before sourcing.
#
# Score-tagging: any score this script would produce (when dispatch is lifted)
# is tagged [contest-CUDA] in the completion-log line (LANE_TISHBY_IB_PURE_DONE
# marker) per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable.
#
# Heartbeat: every 5 min per CLAUDE.md "Remote code parity - non-negotiable".
set -euo pipefail

# === Catalog #244 canonical Modal/CUDA env hygiene (auto-emitted block) ===
# D1 dispatch 2026-05-15 (Modal T4 smoke) crashed at NVML 999 because the lane
# script did not export DALI_DISABLE_NVML=1 before DALI imported NVML. Per
# CLAUDE.md "Forbidden re-implementing remote bootstrap inline" + standing
# directive 2026-05-15 + canonical constants in tac.deploy.modal.runtime.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
LANE_ID="lane_tishby_ib_pure_l1_scaffold_substrate_build_plus_probes_20260516"
TAG="${TAG:-substrate_tishby_ib_pure}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_tishby_ib_pure_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${TISHBY_IB_PURE_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$TISHBY_IB_PURE_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Trainer flags - Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS env-var ladder.
TISHBY_IB_PURE_VIDEO_PATH="${TISHBY_IB_PURE_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
TISHBY_IB_PURE_OUTPUT_DIR="${TISHBY_IB_PURE_OUTPUT_DIR:-$OUTPUT_DIR}"
TISHBY_IB_PURE_EPOCHS="${TISHBY_IB_PURE_EPOCHS:-200}"
TISHBY_IB_PURE_BATCH_SIZE="${TISHBY_IB_PURE_BATCH_SIZE:-4}"
TISHBY_IB_PURE_LR="${TISHBY_IB_PURE_LR:-5e-4}"
TISHBY_IB_PURE_BETA="${TISHBY_IB_PURE_BETA:-0.01}"
TISHBY_IB_PURE_PATH_VARIANT="${TISHBY_IB_PURE_PATH_VARIANT:-VIB}"
TISHBY_IB_PURE_UPSTREAM_DIR="${TISHBY_IB_PURE_UPSTREAM_DIR:-$WORKSPACE/upstream}"
TISHBY_IB_PURE_DEVICE="${TISHBY_IB_PURE_DEVICE:-cuda}"

DISPATCH_INSTANCE_JOB_ID="${TISHBY_IB_PURE_DISPATCH_INSTANCE_JOB_ID:-${DISPATCH_INSTANCE_JOB_ID:-}}"
DISPATCH_PLATFORM="${DISPATCH_PLATFORM:-modal}"

log() { echo "[lane-tishby-ib-pure] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
cd "$WORKSPACE"

# Stage 0a: strip macOS AppleDouble resource forks before any auth eval path.
rm -f upstream/videos/._*.mkv 2>/dev/null || true

# Stage 0: dispatch claim verification (recipe is dispatch_enabled=false at
# landing so this script is structurally a no-op at L1; the gate remains for
# Phase 2 council lift activation).
if [ -z "$DISPATCH_INSTANCE_JOB_ID" ]; then
    log "WARN: TISHBY_IB_PURE_DISPATCH_INSTANCE_JOB_ID not set (recipe is dispatch_enabled=false at L1)"
fi

# Stage 1: bootstrap runtime deps via canonical wrapper (per Catalog #163).
# REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 prevents the sourced script's main
# flow from executing inside this shell.
if [ -f "$WORKSPACE/scripts/remote_archive_only_eval.sh" ]; then
    REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh"
    if declare -f bootstrap_runtime_deps > /dev/null 2>&1; then
        log "Invoking canonical bootstrap_runtime_deps"
        bootstrap_runtime_deps || true
    fi
fi

# Stage 2: research-only refusal (per recipe dispatch_enabled=false + trainer
# _full_main NotImplementedError per Catalog #240 cascade).
log "L1 SCAFFOLD landing: dispatch refused at script level per recipe"
log "  research_only=true + dispatch_enabled=false"
log "  Trainer _full_main raises NotImplementedError per Catalog #240"
log "  Phase 2 council approval required to lift; gated on:"
log "  - D4 probe MEANINGFUL_CONDITIONING (current INDEPENDENT MI=0.0064)"
log "  - VIB tractability TRACTABLE on real-scorer Modal A100 100ep proxy"
log "  - Dykstra-feasibility non-empty intersection"
log "  - Path-VIB vs Path-MINE council adjudication"
log "  See design memo §19.1 V1 lift gate"
log "  Cross-ref:"
log "    .omx/research/tishby_ib_pure_substrate_asymptotic_pursuit_scoping_design_20260516.md"
log "    .omx/state/h_latent_given_scorer_class_tishby_ib_pure.json (D4 verdict)"
log "    .omx/state/variational_ib_tractability_tishby_ib_pure.json (VIB verdict)"

# Stage 3: emit provenance JSON so harvest tooling can see the lane state
cat > "$PROVENANCE" <<EOF
{
  "lane_id": "$LANE_ID",
  "tag": "$TAG",
  "research_only": true,
  "dispatch_enabled": false,
  "completed_at_utc": "$(date -u +%FT%TZ)",
  "rc": 0,
  "design_memo": ".omx/research/tishby_ib_pure_substrate_asymptotic_pursuit_scoping_design_20260516.md",
  "d4_probe_verdict": "INDEPENDENT",
  "d4_probe_mi_bits": 0.0064,
  "vib_tractability_verdict": "TRACTABLE",
  "vib_tractability_snr_mean": 6.75,
  "phase_2_council_required": true,
  "score_claim": false,
  "evidence_grade": "diagnostic_cpu",
  "axis_label": "[diagnostic-CPU; tishby_ib_pure_l1_scaffold_no_dispatch]"
}
EOF

log "LANE_TISHBY_IB_PURE_DONE_RESEARCH_ONLY [diagnostic-CPU; L1 scaffold] provenance=$PROVENANCE"
exit 0
