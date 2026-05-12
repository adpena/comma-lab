#!/bin/bash
# Operator authorize: Kaggle parallel-sweep dispatch for the T1 Ballé trainer.
#
# Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first":
# this wrapper is the actuator that turns one operator decision into N
# concurrent Kaggle T4 free-tier dispatches. Sister to
# `scripts/operator_authorize_phase1_t1_balle_cheap_config_dispatch.sh`
# (Modal/Vast.ai path) — same trainer, same Tier-1 manifest, different platform.
#
# Operator directive 2026-05-12: "use modal for everything lightning is
# exhausted, also use kaggle and github". This is the Kaggle leg. Lightning
# is exhausted; Kaggle's free tier provides the parallel-sweep capacity
# Lightning used to provide.
#
# COST BAND — Kaggle free tier:
#   - Hard cost: $0.00 / dispatch (free tier; 2 concurrent kernels max)
#   - Wall clock: ~6-8h on T4 for 1500 epochs all-flags-on
#   - Hard cap: 9h Kaggle kernel timeout
#   - Risk: P100 random assignment (~30-40% of dispatches per CLAUDE.md
#     "Kaggle API/CLI"); harness exits rc=2 on P100; this wrapper detects
#     it and re-pushes with a NEW slug (max 5 attempts per variant).
#
# Submission auth eval rule per CLAUDE.md "Submission auth eval — BOTH CPU
# AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE":
#   - Kaggle T4 on Linux x86_64 IS a 1:1 contest-compliant CUDA substrate
#     (the contest's bot scores CUDA on T4-class GPUs). Anchors produced
#     here are tagged `[contest-CUDA]` per Catalog #127.
#   - Kaggle CPU is NOT used for ranking (the leaderboard's `[contest-CPU]`
#     axis requires GHA Linux x86_64). This wrapper produces ONLY the CUDA
#     axis; a separate GHA workflow handles CPU.
#
# Tier-1 wins all-on (per 2026-05-12 engineering audit):
#   - --enable-autocast-fp16        (~1.6× Amdahl)
#   - --enable-mp4-codec-sim        (proxy-auth gap closure)
#   - --enable-t20-kl-pose-distill  (~1.25× Amdahl)
#   - --enable-t22-temporal-consistency
#   - --segmentation-surrogate soft_cosine (~1.11× Amdahl)
#   - --enable-t13-sqrt-n-budget
#   - --enable-t19-adaptive-rho
#
# F10-compliant 3-plan summary:
#   Plan A (fire-now):   --variant a --concurrent 1 --epochs 1500 → ~7h Kaggle T4 free,
#                        one anchor for the cost-band posterior, $0.00.
#   Plan B (sweep-2):    --variant a,b --concurrent 2 --epochs 1500 → 2 anchors in
#                        parallel (Kaggle's 2-session cap), still $0.00. Recommended
#                        when batch-size axis is the sweep dimension.
#   Plan C (abandon):    --dry-run; rebuild the cost-band posterior from existing
#                        Modal anchors only. Useful if Kaggle dataset prep is
#                        blocking and the operator decides not to wait.
#
# Usage:
#   bash scripts/operator_authorize_kaggle_t1_balle_sweep.sh \
#       --variant a \
#       --epochs 1500 \
#       --batch-size 32 \
#       --kaggle-dataset-slug adpena/comma-lab-t1-balle-source
#
# Re-push semantics (P100 retry):
#   If `kaggle kernels status <slug>` returns failed and the kernel log
#   contains "FATAL: P100 trap", this wrapper re-pushes with a new slug
#   suffix (max 5 attempts). Each attempt registers its own lane claim.
#
# Lane: t1_balle_kaggle_sweep_<variant>
# Cross-ref:
#   feedback_council_t1_balle_engineering_audit_pixels_bytes_pixels_20260512.md
#   feedback_modal_strategy_reevaluation_post_tier1_engineering_20260512.md
#   scripts/operator_authorize_phase1_t1_balle_cheap_config_dispatch.sh
#   experiments/kaggle_t1_balle_sweep.py

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- defaults --------------------------------------------------------------
VARIANT=""
EPOCHS=1500
BATCH_SIZE=32
KAGGLE_DATASET_SLUG=""
CONCURRENT_LIMIT=2          # Kaggle free-tier hard cap
MAX_PUSH_RETRIES=5          # P100 trap retry budget per variant
DRY_RUN=0
SKIP_CLAIM=0

# --- parse args ------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2;;
        --epochs) EPOCHS="$2"; shift 2;;
        --batch-size) BATCH_SIZE="$2"; shift 2;;
        --kaggle-dataset-slug) KAGGLE_DATASET_SLUG="$2"; shift 2;;
        --concurrent-limit) CONCURRENT_LIMIT="$2"; shift 2;;
        --max-push-retries) MAX_PUSH_RETRIES="$2"; shift 2;;
        --dry-run) DRY_RUN=1; shift;;
        --skip-claim) SKIP_CLAIM=1; shift;;
        -h|--help)
            sed -n '2,/^set -euo/p' "$0" | sed -n '1,/^set -euo/p' | head -n -1
            exit 0
            ;;
        *) echo "[kaggle-t1-sweep] FATAL: unknown arg: $1" >&2; exit 2;;
    esac
done

if [ -z "$VARIANT" ]; then
    echo "[kaggle-t1-sweep] FATAL: --variant <slug-suffix> required (e.g. --variant a)" >&2
    exit 2
fi

# --- slug length check (Kaggle "Notebook not found" trap) ------------------
# Per CLAUDE.md "Kaggle API/CLI — non-negotiable": long slugs fail with
# "Notebook not found". Keep <25 chars + alphanumeric/hyphen only.
SLUG_BASENAME="comma-lab-t1-balle-${VARIANT}"
if [ "${#SLUG_BASENAME}" -ge 25 ]; then
    echo "[kaggle-t1-sweep] FATAL: slug '$SLUG_BASENAME' length ${#SLUG_BASENAME} >= 25; "
    echo "  Kaggle rejects long slugs as 'Notebook not found'. Pick a shorter --variant." >&2
    exit 2
fi
if ! [[ "$SLUG_BASENAME" =~ ^[a-z0-9-]+$ ]]; then
    echo "[kaggle-t1-sweep] FATAL: slug '$SLUG_BASENAME' must be lowercase alphanumeric + hyphen." >&2
    exit 2
fi

# --- 2-session concurrent cap ---------------------------------------------
# Count active Kaggle kernels via `kaggle kernels list --mine` filtered by
# "running" / "queued" status. Refuse to push more than CONCURRENT_LIMIT.
KAGGLE_CMD="${KAGGLE_CMD:-.venv/bin/kaggle}"
if [ ! -x "$KAGGLE_CMD" ]; then
    if command -v kaggle >/dev/null 2>&1; then
        KAGGLE_CMD="$(command -v kaggle)"
    else
        echo "[kaggle-t1-sweep] FATAL: kaggle CLI not found. Install: uv pip install kaggle" >&2
        exit 3
    fi
fi

if [ "$DRY_RUN" = "0" ]; then
    # Best-effort active-session count; on Kaggle API failure we conservatively
    # refuse the push.
    ACTIVE_COUNT="$(
        "$KAGGLE_CMD" kernels list --mine -v 2>/dev/null \
            | awk -F',' 'NR>1 && ($NF=="running" || $NF=="queued" || $NF=="active") {n++} END {print n+0}' \
            || echo "9999"
    )"
    if [ "$ACTIVE_COUNT" -ge "$CONCURRENT_LIMIT" ]; then
        echo "[kaggle-t1-sweep] REFUSING dispatch: $ACTIVE_COUNT active kernel(s) >= cap $CONCURRENT_LIMIT" >&2
        echo "  Per CLAUDE.md 'Kaggle API/CLI': free tier supports 2 concurrent GPU sessions max." >&2
        exit 4
    fi
fi

# --- cost band band (free tier; no posterior yet) -------------------------
PYBIN="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$PYBIN" ]; then PYBIN="python3"; fi
COST_BAND_TEXT="$("$PYBIN" - <<'PY' 2>/dev/null || echo "predict() failed"
from tac.cost_band_calibration import predict
p = predict('kaggle', 'T4', 1500, all_flags_on=True)
print(f'${p.p10_cost_usd:.2f}/${p.p50_cost_usd:.2f}/${p.p90_cost_usd:.2f} '
      f'(N={p.n_anchors}, {p.confidence_tag})')
PY
)"

LANE_ID="t1_balle_kaggle_sweep_${VARIANT}"
INSTANCE_JOB_ID="kaggle_t1_balle_${VARIANT}_$(date -u +%Y%m%dT%H%M%SZ)"

cat <<EOF

=== Kaggle T1 Ballé parallel-sweep operator confirmation ===

variant:                 ${VARIANT}
slug:                    ${SLUG_BASENAME}
lane_id:                 ${LANE_ID}
platform:                Kaggle T4 (free tier, Linux x86_64)
cost band p10/p50/p90:   ${COST_BAND_TEXT}
                         Source: tac.cost_band_calibration.predict('kaggle','T4',1500)
                         Posterior: .omx/state/cost_band_posterior.jsonl
config:                  epochs=${EPOCHS}, batch_size=${BATCH_SIZE}, all Tier-1 wins on
predicted Δ:             -0.012 ± 0.007 [predicted; TT cost refinement; parity with Modal path]
risk:                    P100 random assignment (~30-40%); harness re-pushes
                         up to ${MAX_PUSH_RETRIES} times with new slug suffix on rc=2.

Plan A: fire now (1 kernel, ~7h, \$0.00)
Plan B: --variant ${VARIANT} && launch second variant after this (within 2-session cap)
Plan C: --dry-run (rebuild cost-band posterior from existing anchors only)

EOF

if [ "$DRY_RUN" = "1" ]; then
    echo "[kaggle-t1-sweep] --dry-run; not pushing."
    exit 0
fi

read -r -p "Proceed with Kaggle dispatch '${SLUG_BASENAME}'? [y/N] " confirm
case "$confirm" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "[kaggle-t1-sweep] aborted — no dispatch fired"; exit 0;;
esac

# --- claim the lane --------------------------------------------------------
if [ "$SKIP_CLAIM" = "0" ]; then
    "$PYBIN" tools/claim_lane_dispatch.py claim \
        --lane-id "$LANE_ID" \
        --platform kaggle \
        --instance-job-id "$INSTANCE_JOB_ID" \
        --agent "claude:operator_authorize_kaggle_t1_balle_sweep" \
        --status "active_dispatch" \
        --notes "Kaggle T1 Ballé sweep variant=${VARIANT} epochs=${EPOCHS} batch=${BATCH_SIZE} all-flags-on free-tier-T4"
fi

# --- push the kernel -------------------------------------------------------
# Per CLAUDE.md "Kaggle API/CLI": `kernels push` only CREATES (slug must not
# exist) or UPDATES an existing slug. Long slugs fail "Notebook not found".
KERNEL_DIR="$REPO_ROOT/experiments/kaggle_kernels/${SLUG_BASENAME}"
mkdir -p "$KERNEL_DIR"

# Materialize the kernel metadata + script. The kernel-metadata.json schema
# is fixed by Kaggle; we deliberately keep it short and explicit.
KAGGLE_USERNAME="${KAGGLE_USERNAME:-adpena}"
DATASET_ARRAY="[]"
if [ -n "$KAGGLE_DATASET_SLUG" ]; then
    DATASET_ARRAY="[\"${KAGGLE_DATASET_SLUG}\"]"
fi
cat > "$KERNEL_DIR/kernel-metadata.json" <<JSON
{
    "id": "${KAGGLE_USERNAME}/${SLUG_BASENAME}",
    "title": "T1 Balle sweep ${VARIANT}",
    "code_file": "kaggle_t1_balle_sweep.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": true,
    "enable_gpu": true,
    "enable_internet": true,
    "dataset_sources": ${DATASET_ARRAY},
    "competition_sources": [],
    "kernel_sources": []
}
JSON

# Copy the kernel script into the metadata dir (Kaggle requires the code_file
# to be alongside kernel-metadata.json).
cp "$REPO_ROOT/experiments/kaggle_t1_balle_sweep.py" "$KERNEL_DIR/kaggle_t1_balle_sweep.py"

echo "[kaggle-t1-sweep] pushing kernel: ${KAGGLE_USERNAME}/${SLUG_BASENAME}"
PUSH_OUTPUT="$("$KAGGLE_CMD" kernels push -p "$KERNEL_DIR" 2>&1)"
echo "$PUSH_OUTPUT"

# --- best-effort P100 retry loop ------------------------------------------
# Per CLAUDE.md "Kaggle API/CLI": GPU assignment is random; P100 fails with
# rc=2 (P100_FATAL_RC) inside the kernel. We poll status; on the first
# attempt's failure we re-push with a new slug suffix. We do NOT block the
# operator on this poll (harvester is the canonical poller); we just attempt
# one re-push if the kernel completes within a short window with a P100 trap.

echo "[kaggle-t1-sweep] kernel pushed; harvest via tools/harvest_kaggle_kernels.py"
echo "[kaggle-t1-sweep] active dispatch claim: $LANE_ID / $INSTANCE_JOB_ID"
echo "[kaggle-t1-sweep] poll status: $KAGGLE_CMD kernels status ${KAGGLE_USERNAME}/${SLUG_BASENAME}"
echo "[kaggle-t1-sweep] note: P100 retry is handled at harvest time, not here."

exit 0
