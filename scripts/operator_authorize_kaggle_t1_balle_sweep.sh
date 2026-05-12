#!/bin/bash
# Thin shim: delegates to the canonical operator-authorize entry point.
#
# Recipe: ``.omx/operator_authorize_recipes/kaggle_t1_balle_sweep.yaml``
#
# Per simplification audit `a2a901c4f43d66a74` (operator approved 2026-05-12)
# + Catalog #162 ``check_operator_authorize_canonical_use``.
#
# Args (passed through to the legacy logic below for back-compat):
#   --variant <slug-suffix>   (required)
#   --epochs <int>            (default 1500)
#   --batch-size <int>        (default 32)
#   --kaggle-dataset-slug <slug>
#   --concurrent-limit <int>  (default 2)
#   --max-push-retries <int>  (default 5)
#   --dry-run
#   --skip-claim
#
# Lane: lane_fix_g_t1c_wrapper_unification_20260512
# Cross-ref: feedback_fix_g_t1c_wrapper_unification_landed_20260512.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Phase 1: canonical entry point — banner (no lane-claim; this wrapper claims
# its own lane below per --variant).
.venv/bin/python tools/operator_authorize.py \
    --recipe kaggle_t1_balle_sweep \
    --agent "claude:operator_authorize_kaggle_t1_balle_sweep" \
    --no-claim \
    --dry-run

# Phase 2: bespoke argv parsing + kernel push preserved for back-compat.
VARIANT=""
EPOCHS=1500
BATCH_SIZE=32
KAGGLE_DATASET_SLUG=""
CONCURRENT_LIMIT=2
MAX_PUSH_RETRIES=5
DRY_RUN=0
SKIP_CLAIM=0

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
        -h|--help) head -40 "$0"; exit 0;;
        *) echo "[kaggle-t1-sweep] FATAL: unknown arg: $1" >&2; exit 2;;
    esac
done

if [ -z "$VARIANT" ]; then
    echo "[kaggle-t1-sweep] FATAL: --variant <slug-suffix> required (e.g. --variant a)" >&2
    exit 2
fi

SLUG_BASENAME="comma-lab-t1-balle-${VARIANT}"
if [ "${#SLUG_BASENAME}" -ge 25 ]; then
    echo "[kaggle-t1-sweep] FATAL: slug '$SLUG_BASENAME' length ${#SLUG_BASENAME} >= 25" >&2
    exit 2
fi
if ! [[ "$SLUG_BASENAME" =~ ^[a-z0-9-]+$ ]]; then
    echo "[kaggle-t1-sweep] FATAL: slug must be lowercase alphanumeric + hyphen" >&2
    exit 2
fi

KAGGLE_CMD="${KAGGLE_CMD:-.venv/bin/kaggle}"
if [ ! -x "$KAGGLE_CMD" ]; then
    if command -v kaggle >/dev/null 2>&1; then
        KAGGLE_CMD="$(command -v kaggle)"
    else
        echo "[kaggle-t1-sweep] FATAL: kaggle CLI not found; install: uv pip install kaggle" >&2
        exit 3
    fi
fi

if [ "$DRY_RUN" = "0" ]; then
    ACTIVE_COUNT="$(
        "$KAGGLE_CMD" kernels list --mine -v 2>/dev/null \
            | awk -F',' 'NR>1 && ($NF=="running" || $NF=="queued" || $NF=="active") {n++} END {print n+0}' \
            || echo "9999"
    )"
    if [ "$ACTIVE_COUNT" -ge "$CONCURRENT_LIMIT" ]; then
        echo "[kaggle-t1-sweep] REFUSING: $ACTIVE_COUNT active kernel(s) >= cap $CONCURRENT_LIMIT" >&2
        exit 4
    fi
fi

LANE_ID="t1_balle_kaggle_sweep_${VARIANT}"
INSTANCE_JOB_ID="kaggle_t1_balle_${VARIANT}_$(date -u +%Y%m%dT%H%M%SZ)"

echo "[kaggle-t1-sweep] variant=${VARIANT} slug=${SLUG_BASENAME} lane=${LANE_ID}"

if [ "$DRY_RUN" = "1" ]; then
    echo "[kaggle-t1-sweep] --dry-run; not pushing."
    exit 0
fi

read -r -p "Proceed with Kaggle dispatch '${SLUG_BASENAME}'? [y/N] " confirm
case "$confirm" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "[kaggle-t1-sweep] aborted — no dispatch fired"; exit 0;;
esac

PYBIN="$REPO_ROOT/.venv/bin/python"
[ -x "$PYBIN" ] || PYBIN="python3"

if [ "$SKIP_CLAIM" = "0" ]; then
    "$PYBIN" tools/claim_lane_dispatch.py claim \
        --lane-id "$LANE_ID" \
        --platform kaggle \
        --instance-job-id "$INSTANCE_JOB_ID" \
        --agent "claude:operator_authorize_kaggle_t1_balle_sweep" \
        --status "active_dispatch" \
        --notes "Kaggle T1 Balle sweep variant=${VARIANT} epochs=${EPOCHS} batch=${BATCH_SIZE} all-flags-on free-tier-T4"
fi

KERNEL_DIR="$REPO_ROOT/experiments/kaggle_kernels/${SLUG_BASENAME}"
mkdir -p "$KERNEL_DIR"

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

cp "$REPO_ROOT/experiments/kaggle_t1_balle_sweep.py" "$KERNEL_DIR/kaggle_t1_balle_sweep.py"

echo "[kaggle-t1-sweep] pushing: ${KAGGLE_USERNAME}/${SLUG_BASENAME}"
"$KAGGLE_CMD" kernels push -p "$KERNEL_DIR"
echo "[kaggle-t1-sweep] active dispatch claim: $LANE_ID / $INSTANCE_JOB_ID"
echo "[kaggle-t1-sweep] poll: $KAGGLE_CMD kernels status ${KAGGLE_USERNAME}/${SLUG_BASENAME}"
