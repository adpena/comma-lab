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

# ── Machine-enforced gates (replace comment-only contracts) ──
# Gate 1: apogee_int4 sanity must have landed in [0.155, 0.180]. It landed 1.4287
#         (8x miss). Hardcode the FALSIFIED status; no future re-arming without
#         operator override.
# Gate 2: the smoke archives this script references have zero-delta corrections
#         (scorer_available: false in build_metadata.json). Without CUDA Stage 3
#         refinement, dispatching them just re-evaluates PR106 baseline +
#         overhead bytes — wasted GPU spend.
# Both gates fail-loud below.

if [ "${WAVE_DEPLOY_OVERRIDE:-0}" != "1" ]; then
    cat >&2 <<'EOF'
[wave-deploy] BLOCKED by post-incident hardening (audit finding C1):

  apogee_int4 sanity dispatch landed score 1.4287 — outside predicted band
  [0.155, 0.180] (job apogee-int4-postfix-sanity-20260505t172500z, 2026-05-05T17:40Z).
  See ~/.claude/projects/.../memory/project_apogee_int4_FALSIFIED_score_1_43_dispatcher_VALIDATED_20260505.md.

  Additionally, the smoke archives this script references have zero-delta
  corrections (scorer_available: false). Dispatching them via exact-eval
  would just re-measure PR106 baseline plus overhead bytes — no Shannon-floor
  signal.

  This script is intentionally fail-closed until at least one of:
    1. A CUDA-refined sidecar archive replaces the zero-delta smoke
       (run experiments/build_pr106_latent_sidecar.py --device cuda on a real GPU host).
    2. Operator passes WAVE_DEPLOY_OVERRIDE=1 with explicit acknowledgment.

  Forensic memo: project_apogee_int4_FALSIFIED_score_1_43_dispatcher_VALIDATED_20260505.md
  Audit memo:    project_full_repo_bug_audit_20260505.md (C1)
EOF
    exit 78  # EX_CONFIG: configuration error
fi

# Override path: verify each archive has non-zero corrections before dispatch.
log_smoke_check() {
    local archive="$1"
    local meta_dir="$(dirname "$archive")"
    local meta="$meta_dir/build_metadata.json"
    if [ ! -f "$meta" ]; then return 0; fi  # missing metadata → don't block
    local delta_min delta_max
    delta_min="$($PYBIN -c "import json; print(json.load(open('$meta')).get('diagnostics',{}).get('delta_q_min',0))" 2>/dev/null || echo "0")"
    delta_max="$($PYBIN -c "import json; print(json.load(open('$meta')).get('diagnostics',{}).get('delta_q_max',0))" 2>/dev/null || echo "0")"
    if [ "$delta_min" = "0" ] && [ "$delta_max" = "0" ]; then
        echo "[wave-deploy] FATAL: $archive has zero-delta corrections (smoke build, no CUDA Stage 3)" >&2
        return 1
    fi
    return 0
}

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
    log_smoke_check "$ARCHIVE" || exit 1  # fail-loud on zero-delta smoke

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
