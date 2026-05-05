#!/usr/bin/env bash
# scripts/wave_deploy_post_apogee_int4_sanity.sh
#
# Pre-staged wave-deploy for the post-apogee_int4-sanity Branch A path.
# Fires 4 PR106-stacking lanes in parallel on Lightning Studio T4.
# Cost: ~$0.80-1.36 total (4 × $0.20-0.34).
#
# DO NOT RUN until apogee_int4_postfix_sanity_20260505T172500Z lands a valid
# score in the [0.155, 0.180] band. That is the gating end-to-end validation
# of the catastrophe-fix dispatcher path.
#
# Reference: ~/.claude/projects/-Users-adpena-Projects-pact/memory/project_post_apogee_int4_sanity_wave_plan_20260505.md
#
# Usage:
#   bash scripts/wave_deploy_post_apogee_int4_sanity.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PYBIN=".venv/bin/python"

DISPATCHER="$PYBIN tools/lightning_dispatch_pr106_stack.py"

# Lane → (archive, predicted_low, predicted_high)
# All archives are CPU-smoke producers; the inflate path applies the per-lane
# correction adapter on the contest scorer side.
LANES=(
    "pr106_latent_sidecar|experiments/results/lane_pr106_latent_sidecar_cpu_smoke_20260505/sidecar_archive.zip|0.205|0.212"
    "pr106_yshift_sidechannel|experiments/results/lane_pr106_yshift_cpu_smoke_20260505T140325Z/pr106_yshift_sidechannel_archive.zip|0.207|0.213"
    "pr106_lrl1_sidechannel|experiments/results/lane_pr106_lrl1_cpu_smoke_20260505T140325Z/pr106_lrl1_sidechannel_archive.zip|0.207|0.213"
    "pr106_stacked|experiments/results/lane_pr106_stacked_3sister_cpu_smoke_20260505T140325Z/pr106_stacked_archive.zip|0.200|0.210"
)

PIDS=()
for spec in "${LANES[@]}"; do
    IFS='|' read -r LANE ARCHIVE PRED_LOW PRED_HIGH <<<"$spec"
    JOB_NAME="${LANE}_wave_${TIMESTAMP}"
    INFLATE_SH="submissions/${LANE}/inflate.sh"

    if [ ! -f "$ARCHIVE" ]; then
        echo "FATAL: missing archive: $ARCHIVE" >&2
        exit 1
    fi
    if [ ! -f "$INFLATE_SH" ]; then
        echo "FATAL: missing inflate.sh: $INFLATE_SH" >&2
        exit 1
    fi

    echo "[wave] dispatching $LANE -> $JOB_NAME (archive=$ARCHIVE)"
    LOG="experiments/results/lightning_batch/${JOB_NAME}.log"
    mkdir -p "$(dirname "$LOG")"
    $DISPATCHER \
        --lane "$LANE" \
        --archive "$ARCHIVE" \
        --inflate-sh "$INFLATE_SH" \
        --predicted-low "$PRED_LOW" \
        --predicted-high "$PRED_HIGH" \
        --job-name "$JOB_NAME" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo "[wave] launched ${#PIDS[@]} dispatches in parallel; waiting for stage+claim+submit phases..."
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "[wave] $FAILED of ${#PIDS[@]} dispatches failed at stage/claim/submit (check experiments/results/lightning_batch/*_wave_${TIMESTAMP}.log)"
    exit 1
fi

echo "[wave] all ${#PIDS[@]} dispatches successfully submitted to Lightning Studio T4 queue"
echo "[wave] poll status: .venv/bin/python -c \"from lightning_sdk import Teamspace; ts = Teamspace(name='comma-lab', user='adpena'); [print(j.name, j.status) for j in ts.jobs if 'wave_${TIMESTAMP}' in j.name.lower()]\""
echo "[wave] timestamp: ${TIMESTAMP}"
