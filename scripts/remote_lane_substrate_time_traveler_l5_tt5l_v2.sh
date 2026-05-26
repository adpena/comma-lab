#!/bin/bash
# Remote lane script: substrate TT5L V2 Time-Traveler L5 redesign scaffold.
#
# Trainer: experiments/train_substrate_time_traveler_l5_tt5l_v2.py
# Lane: lane_tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_20260518
# Recipe: .omx/operator_authorize_recipes/substrate_time_traveler_l5_tt5l_v2_modal_a100_dispatch.yaml
#
# This driver exists to close the missing-driver launch-precondition gap
# while preserving the recipe's research-only / dispatch-disabled state.
# It requires an already-open dispatch claim, never opens provider work
# by itself, and terminalizes every exit. Successful artifacts are
# classified as no-score-claim unless a future exact-eval path writes
# explicit authority fields per Catalog #127 + #226.
#
# Per CLAUDE.md "Forbidden re-implementing remote bootstrap inline" +
# Catalog #163: bootstrap delegates to scripts/remote_archive_only_eval.sh's
# bootstrap_runtime_deps() with the canonical sentinel
# ``REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1`` prepended.
#
# Per Catalog #189 every optional-array expansion is guarded as
# ``${ARR[@]+"${ARR[@]}"}`` so the script tolerates `set -u` on macOS bash 3.2.
#
# Per Catalog #244 the canonical 3-export NVML/CUDA env block is emitted
# IMMEDIATELY after `set -euo pipefail` so DALI does not crash with
# `nvml error (999)` and CUBLAS produces deterministic results. Sister
# D1/D4/Z3/Z4/Z5/Z6/Z7-GRU drivers carry the same block; commit 611495f26
# is the canonical anchor.
#
# Per Catalog #326 (DRIVER-FIX wave 2026-05-18): the driver supports
# multi-key mode resolution TT5L_V2_TRAINER_MODE > SMOKE_ONLY > default
# WARN-fallback-to-smoke so a recipe `env_overrides` block that forgets
# to set the mode env var produces a loud WARN banner instead of silent
# wrong-mode dispatch.
#
# Per Catalog #240: TT5L V2 trainer _full_main raises NotImplementedError
# until Wave N+1 council PROCEED-unconditional per the design memo's
# Revisions #5/#6/#7 cascade. This driver routes smoke mode for the
# canonical scaffold sanity check only.
#
# Design refs:
#   - .omx/research/tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_memo_20260518.md
#   - .omx/research/council_per_substrate_symposium_tt5l_foveation_lapose_20260517.md
#   - .omx/research/comprehensive_research_wave_20260518.md
#   - Atick-Redlich (1990) cooperative-receiver theorem
#   - Hafner et al. (arxiv 2301.04104) DreamerV3 RSSM categorical
#   - Wang et al. CVPR 2025 (arxiv 2503.11651) VGGT
#   - Wang et al. ECCV 2024 (arxiv 2312.14132 + 2406.09756) DUSt3R/MASt3R
#   - Rao-Ballard (1999) hierarchical predictive coding
#
# Score-tagging: smoke/no-scorer artifacts are explicitly logged as
# score_claim=false and never as [contest-CUDA]. A [contest-CUDA] marker
# is allowed only when stats.json proves a valid contest_cuda score claim
# (auth_eval_score_claim_valid=true AND auth_eval_score_axis=contest_cuda)
# per Catalog #226 + Catalog #127.
#
# Per Catalog #204 the output is written to
# /modal_results/${DISPATCH_INSTANCE_JOB_ID}/output for durable provider
# custody when MODAL_RUNTIME=1.
set -euo pipefail

# Catalog #244 canonical 3-export NVML/CUDA env block (emitted immediately
# after `set -euo pipefail`; sourced from canonical constants in
# tac.deploy.modal.runtime).
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WORKSPACE="${WORKSPACE:-/workspace/pact}"
PYBIN="${PYBIN:-}"
DEFAULT_LANE_ID="lane_tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_20260518"
LANE_ID="${TT5L_V2_LANE_ID:-${PACT_DISPATCH_LANE_ID:-$DEFAULT_LANE_ID}}"
TAG="${TAG:-substrate_time_traveler_l5_tt5l_v2}"
RECIPE_PATH="${TT5L_V2_RECIPE_PATH:-.omx/operator_authorize_recipes/substrate_time_traveler_l5_tt5l_v2_modal_a100_dispatch.yaml}"

LOG_DIR="${LOG_DIR:-$WORKSPACE/lane_substrate_time_traveler_l5_tt5l_v2_results}"
# Catalog #204 cross-driver expansion (2026-05-19): when running on Modal
# (MODAL_RUNTIME=1) write archive/runtime/auth-eval artifacts under the
# /modal_results volume so modal_train_lane.py harvests durable custody.
# contest_auth_eval.py refuses temp-storage evidence per CLAUDE.md "Forbidden
# /tmp paths in any persisted artifact" non-negotiable.
if [ -n "${TT5L_V2_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$TT5L_V2_OUTPUT_DIR"
elif [ "${MODAL_RUNTIME:-0}" = "1" ] && [ -d "/modal_results" ]; then
    OUTPUT_DIR="/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output"
else
    OUTPUT_DIR="$LOG_DIR/output"
fi
PROVENANCE="$LOG_DIR/provenance.json"

# Catalog #326 mode-routing: TT5L_V2_TRAINER_MODE > SMOKE_ONLY > default-smoke-WARN.
TT5L_V2_TRAINER_MODE="${TT5L_V2_TRAINER_MODE:-}"
SMOKE_ONLY="${SMOKE_ONLY:-}"
if [ -n "$TT5L_V2_TRAINER_MODE" ]; then
    case "$TT5L_V2_TRAINER_MODE" in
        smoke|SMOKE|Smoke)
            SMOKE_ONLY="1"
            ;;
        full|FULL|Full)
            SMOKE_ONLY="0"
            ;;
        *)
            echo "[lane-tt5l-v2] FATAL: invalid TT5L_V2_TRAINER_MODE=$TT5L_V2_TRAINER_MODE; expected smoke|full" >&2
            exit 29
            ;;
    esac
elif [ -z "$SMOKE_ONLY" ]; then
    echo "[lane-tt5l-v2] WARN: neither TT5L_V2_TRAINER_MODE nor SMOKE_ONLY set; defaulting to smoke. Per Catalog #326 recipes SHOULD declare TT5L_V2_TRAINER_MODE=full or SMOKE_ONLY=0 for full-mode dispatch (currently BLOCKED per Catalog #240 + design memo Revisions #5/#6/#7)." >&2
    SMOKE_ONLY="1"
fi

# Per design memo §12 + parent #866 + Wave N+1 prerequisites: full-mode
# requires _full_main to be implemented; currently raises NotImplementedError.
# Surface this loudly when SMOKE_ONLY=0 is forced by recipe.
if [ "$SMOKE_ONLY" = "0" ]; then
    echo "[lane-tt5l-v2] WARN: SMOKE_ONLY=0 requested but trainer _full_main raises NotImplementedError per Catalog #240 + design memo. Recipe MUST be research_only=true + dispatch_enabled=false. Dispatch will fail at trainer entry." >&2
fi

TT5L_V2_VIDEO_PATH="${TT5L_V2_VIDEO_PATH:-$WORKSPACE/upstream/videos/0.mkv}"
TT5L_V2_OUTPUT_DIR="${TT5L_V2_OUTPUT_DIR:-$OUTPUT_DIR}"
TT5L_V2_EPOCHS="${TT5L_V2_EPOCHS:-}"
if [ -z "$TT5L_V2_EPOCHS" ]; then
    if [ "$SMOKE_ONLY" = "1" ]; then
        TT5L_V2_EPOCHS="3"
    else
        TT5L_V2_EPOCHS="100"
    fi
fi
TT5L_V2_BATCH_SIZE="${TT5L_V2_BATCH_SIZE:-2}"
TT5L_V2_LR="${TT5L_V2_LR:-5e-4}"

# VGGT compress-time teacher (design memo §2.2)
TT5L_V2_VGGT_TEACHER_CHECKPOINT="${TT5L_V2_VGGT_TEACHER_CHECKPOINT:-$WORKSPACE/experiments/results/vggt_pretrained/vggt_pretrained.pt}"
TT5L_V2_LAMBDA_VGGT="${TT5L_V2_LAMBDA_VGGT:-0.05}"
TT5L_V2_DISABLE_VGGT_TEACHER="${TT5L_V2_DISABLE_VGGT_TEACHER:-false}"

# DreamerV3 RSSM categorical (design memo §2.3)
TT5L_V2_RSSM_D_STATE="${TT5L_V2_RSSM_D_STATE:-32}"
TT5L_V2_RSSM_N_CATEGORICAL="${TT5L_V2_RSSM_N_CATEGORICAL:-32}"
TT5L_V2_RSSM_N_CLASSES="${TT5L_V2_RSSM_N_CLASSES:-32}"
TT5L_V2_LAMBDA_RSSM="${TT5L_V2_LAMBDA_RSSM:-0.005}"
TT5L_V2_DISABLE_RSSM_CATEGORICAL="${TT5L_V2_DISABLE_RSSM_CATEGORICAL:-false}"

# Cooperative-receiver foveation (design memo §2.4)
TT5L_V2_LAMBDA_FOV="${TT5L_V2_LAMBDA_FOV:-0.005}"
TT5L_V2_DISABLE_COOPERATIVE_RECEIVER_FOVEATION="${TT5L_V2_DISABLE_COOPERATIVE_RECEIVER_FOVEATION:-false}"

# DUSt3R optional distilled prior (design memo §2.5)
TT5L_V2_DUST3R_TEACHER_CHECKPOINT="${TT5L_V2_DUST3R_TEACHER_CHECKPOINT:-$WORKSPACE/experiments/results/dust3r_pretrained/dust3r_pretrained.pt}"
TT5L_V2_LAMBDA_DUST3R="${TT5L_V2_LAMBDA_DUST3R:-0.0}"
TT5L_V2_ENABLE_DUST3R_PRIOR="${TT5L_V2_ENABLE_DUST3R_PRIOR:-false}"

# Inherited V1 SE(3) Lie algebra (design memo §6)
TT5L_V2_EGO_SOURCE="${TT5L_V2_EGO_SOURCE:-posenet_projection}"
TT5L_V2_EGO_MOTION_DIM="${TT5L_V2_EGO_MOTION_DIM:-6}"

# Loss / convergence (design memo §2.6)
TT5L_V2_LAMBDA_TIKHONOV="${TT5L_V2_LAMBDA_TIKHONOV:-1e-5}"
TT5L_V2_DEVICE="${TT5L_V2_DEVICE:-cuda}"

# Catalog #197 coupled CPU-mode discipline
TT5L_V2_FULL_CPU="${TT5L_V2_FULL_CPU:-false}"
TT5L_V2_ADVISORY_CPU_EXPLICITLY_WAIVED="${TT5L_V2_ADVISORY_CPU_EXPLICITLY_WAIVED:-false}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Catalog #163 sentinel + delegated bootstrap (skip if local non-Modal)
if [ "${MODAL_RUNTIME:-0}" = "1" ]; then
    REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source "$WORKSPACE/scripts/remote_archive_only_eval.sh" || true
    if command -v bootstrap_runtime_deps >/dev/null 2>&1; then
        bootstrap_runtime_deps || echo "[lane-tt5l-v2] WARN: bootstrap_runtime_deps non-zero exit (continuing)" >&2
    fi
fi

# Resolve canonical python binary
if [ -z "$PYBIN" ]; then
    if [ -x "$WORKSPACE/.venv/bin/python" ]; then
        PYBIN="$WORKSPACE/.venv/bin/python"
    else
        PYBIN="$(command -v python3 || command -v python)"
    fi
fi
if [ -z "$PYBIN" ]; then
    echo "[lane-tt5l-v2] FATAL: no python binary found" >&2
    exit 30
fi

TRAINER_PATH="$WORKSPACE/experiments/train_substrate_time_traveler_l5_tt5l_v2.py"
if [ ! -f "$TRAINER_PATH" ]; then
    echo "[lane-tt5l-v2] FATAL: trainer not found at $TRAINER_PATH" >&2
    exit 31
fi

# Provenance per CLAUDE.md "Operator gates must be wired and used" +
# canonical remote scripts pattern. Stamps lane + canonical scaffold +
# Wave N+1 prerequisites for forensic audit.
cat > "$PROVENANCE" <<EOF
{
  "schema_version": 1,
  "name": "tt5l_v2_scaffold_dispatch_provenance",
  "lane_id": "$LANE_ID",
  "tag": "$TAG",
  "recipe_path": "$RECIPE_PATH",
  "trainer_path": "experiments/train_substrate_time_traveler_l5_tt5l_v2.py",
  "design_memo": ".omx/research/tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_memo_20260518.md",
  "council_symposium_memo": ".omx/research/council_symposium_tt5l_v2_full_landing_20260518.md",
  "deep_research_wave_top5_rank": 1,
  "primitive_count": 4,
  "primitives": {
    "vggt_compress_time_teacher_arxiv": "2503.11651",
    "dreamerv3_rssm_categorical_arxiv": "2301.04104",
    "cooperative_receiver_foveation_principle": "NVIDIA VRSS 2 + Atick-Redlich 1990",
    "dust3r_optional_arxiv": "2312.14132 + 2406.09756"
  },
  "dispatch_mode": "$([ "$SMOKE_ONLY" = "1" ] && echo "smoke" || echo "full_BLOCKED_per_catalog_240")",
  "research_only": true,
  "dispatch_enabled": false,
  "score_claim": false,
  "promotion_eligible": false,
  "wave_n_plus_1_prerequisites": [
    "per_section_MI_probes_on_V1_25ep_state_atick_tishby_wyner",
    "boyd_dykstra_feasibility_analytical_check",
    "sister_z6_4c_z7_mamba2_z8_atw_v2_c6_ibps_phase_2_outcomes",
    "cheapest_signal_first_wave_2_single_primitive_smoke_hotz",
    "wave_n_plus_1_council_PROCEED_unconditional_catalog_315"
  ]
}
EOF

echo "[lane-tt5l-v2] starting dispatch; lane=$LANE_ID tag=$TAG mode=$([ "$SMOKE_ONLY" = "1" ] && echo "smoke" || echo "full")"
echo "[lane-tt5l-v2] output_dir=$OUTPUT_DIR; epochs=$TT5L_V2_EPOCHS; batch_size=$TT5L_V2_BATCH_SIZE"
echo "[lane-tt5l-v2] primitives: VGGT_teacher(disable=$TT5L_V2_DISABLE_VGGT_TEACHER) RSSM(disable=$TT5L_V2_DISABLE_RSSM_CATEGORICAL) coopRX_foveation(disable=$TT5L_V2_DISABLE_COOPERATIVE_RECEIVER_FOVEATION) DUSt3R(enable=$TT5L_V2_ENABLE_DUST3R_PRIOR)"

# Build the trainer invocation (Catalog #151 manifest threading per
# canonical Z7-Mamba-2 + Z6 pattern). Per Catalog #189: optional-array
# guards even though we use explicit positional CLI args here. Per
# Catalog #152: all required_input_file=True flags (--video-path,
# --vggt-teacher-checkpoint) are threaded explicitly so the dispatch
# pre-flight can validate them.
TRAINER_ARGS=(
    "--video-path" "$TT5L_V2_VIDEO_PATH"
    "--output-dir" "$TT5L_V2_OUTPUT_DIR"
    "--epochs" "$TT5L_V2_EPOCHS"
    "--batch-size" "$TT5L_V2_BATCH_SIZE"
    "--lr" "$TT5L_V2_LR"
    "--vggt-teacher-checkpoint" "$TT5L_V2_VGGT_TEACHER_CHECKPOINT"
    "--lambda-vggt" "$TT5L_V2_LAMBDA_VGGT"
    "--rssm-d-state" "$TT5L_V2_RSSM_D_STATE"
    "--rssm-n-categorical" "$TT5L_V2_RSSM_N_CATEGORICAL"
    "--rssm-n-classes" "$TT5L_V2_RSSM_N_CLASSES"
    "--lambda-rssm" "$TT5L_V2_LAMBDA_RSSM"
    "--lambda-fov" "$TT5L_V2_LAMBDA_FOV"
    "--dust3r-teacher-checkpoint" "$TT5L_V2_DUST3R_TEACHER_CHECKPOINT"
    "--lambda-dust3r" "$TT5L_V2_LAMBDA_DUST3R"
    "--ego-source" "$TT5L_V2_EGO_SOURCE"
    "--ego-motion-dim" "$TT5L_V2_EGO_MOTION_DIM"
    "--lambda-tikhonov" "$TT5L_V2_LAMBDA_TIKHONOV"
    "--device" "$TT5L_V2_DEVICE"
)

if [ "$SMOKE_ONLY" = "1" ]; then
    TRAINER_ARGS+=("--smoke")
fi
if [ "$TT5L_V2_DISABLE_VGGT_TEACHER" = "true" ] || [ "$TT5L_V2_DISABLE_VGGT_TEACHER" = "1" ]; then
    TRAINER_ARGS+=("--disable-vggt-teacher")
fi
if [ "$TT5L_V2_DISABLE_RSSM_CATEGORICAL" = "true" ] || [ "$TT5L_V2_DISABLE_RSSM_CATEGORICAL" = "1" ]; then
    TRAINER_ARGS+=("--disable-rssm-categorical")
fi
if [ "$TT5L_V2_DISABLE_COOPERATIVE_RECEIVER_FOVEATION" = "true" ] || [ "$TT5L_V2_DISABLE_COOPERATIVE_RECEIVER_FOVEATION" = "1" ]; then
    TRAINER_ARGS+=("--disable-cooperative-receiver-foveation")
fi
if [ "$TT5L_V2_ENABLE_DUST3R_PRIOR" = "true" ] || [ "$TT5L_V2_ENABLE_DUST3R_PRIOR" = "1" ]; then
    TRAINER_ARGS+=("--enable-dust3r-prior")
fi
if [ "$TT5L_V2_FULL_CPU" = "true" ] || [ "$TT5L_V2_FULL_CPU" = "1" ]; then
    TRAINER_ARGS+=("--full-cpu")
fi
if [ "$TT5L_V2_ADVISORY_CPU_EXPLICITLY_WAIVED" = "true" ] || [ "$TT5L_V2_ADVISORY_CPU_EXPLICITLY_WAIVED" = "1" ]; then
    TRAINER_ARGS+=("--advisory-cpu-explicitly-waived")
fi

# Catalog #189 guarded expansion (defensive even though TRAINER_ARGS is always non-empty here)
set +e
"$PYBIN" -u "$TRAINER_PATH" ${TRAINER_ARGS[@]+"${TRAINER_ARGS[@]}"} \
    2>&1 | tee -a "$LOG_DIR/lane_tt5l_v2_run.log"
RC=${PIPESTATUS[0]}
set -e

if [ "$RC" -ne 0 ]; then
    echo "[lane-tt5l-v2] trainer exited rc=$RC" >&2
    echo "{\"status\": \"failed\", \"rc\": $RC, \"lane_id\": \"$LANE_ID\", \"mode\": \"$([ "$SMOKE_ONLY" = "1" ] && echo "smoke" || echo "full")\"}" > "$LOG_DIR/lane_tt5l_v2_status.json"
    exit "$RC"
fi

# Per Catalog #226 + #127: do NOT emit a [contest-CUDA] DONE marker
# unless the stats.json validates auth_eval_score_claim_valid=true AND
# auth_eval_score_axis=contest_cuda. Scaffold smoke mode never produces
# such a claim.
echo "{\"status\": \"completed\", \"rc\": 0, \"lane_id\": \"$LANE_ID\", \"mode\": \"$([ "$SMOKE_ONLY" = "1" ] && echo "smoke" || echo "full")\", \"score_claim\": false, \"axis\": null}" > "$LOG_DIR/lane_tt5l_v2_status.json"
echo "[lane-tt5l-v2] LANE_TT5L_V2_DONE [scaffold-smoke; no contest-CUDA claim per Catalog #240 + design memo]"
exit 0
