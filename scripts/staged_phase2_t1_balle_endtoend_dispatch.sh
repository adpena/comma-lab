#!/usr/bin/env bash
# Phase 2 T1 (Ballé 128K end-to-end) dispatch script — PRE-STAGED, operator-approval-gated.
#
# Per CLAUDE.md "Phase 2 GPU dispatch budget" non-negotiable + 2026-05-11
# operator directive ("ready to go in parallel for as soon as we secure
# funding"). This script REFUSES to dispatch without:
#   STAGED_PHASE_DISPATCH_OPERATOR_APPROVED=1  (explicit operator opt-in)
#   STAGED_PHASE_DISPATCH_LANE=t1_balle_128k_endtoend  (lane id match)
#   STAGED_PHASE_DISPATCH_BUDGET_USD=<cap>      (explicit budget cap)
#
# Usage (dry-run; default):
#   bash scripts/staged_phase2_t1_balle_endtoend_dispatch.sh
#
# Usage (dispatch; operator-approved):
#   STAGED_PHASE_DISPATCH_OPERATOR_APPROVED=1 \
#   STAGED_PHASE_DISPATCH_LANE=t1_balle_128k_endtoend \
#   STAGED_PHASE_DISPATCH_BUDGET_USD=80 \
#   bash scripts/staged_phase2_t1_balle_endtoend_dispatch.sh
#
# Per CLAUDE.md non-negotiables:
#   - set -euo pipefail (Catalog #2)
#   - No /tmp paths in persisted artifacts (Catalog "Forbidden /tmp paths")
#   - Does NOT open a lane claim unless a provider job is actually created
#   - Provider: Modal T4 (canonical Phase 2 dispatch substrate; matches
#     A1 retry3 anchor at SHA 87ec7ca5...)
#   - Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

set -euo pipefail

LANE_ID="t1_balle_128k_endtoend"
PROVIDER="modal"
PREDICTED_DELTA_SCORE="-0.012 ± 0.005 [predicted; Phase 2 multi-track]"
COST_BAND_USD="80"
DISPATCH_SCRIPT="$0"

# Repo root detection.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

UTC_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="experiments/results/staged_phase2_${LANE_ID}_${UTC_TIMESTAMP}"

echo "==> Phase 2 T1 dispatch (PRE-STAGED)"
echo "    Lane: ${LANE_ID}"
echo "    Provider: ${PROVIDER}"
echo "    Predicted Δ score: ${PREDICTED_DELTA_SCORE}"
echo "    Cost band: \$${COST_BAND_USD}"
echo "    Output dir: ${OUTPUT_DIR}"
echo

# OPERATOR APPROVAL GATE.
if [[ "${STAGED_PHASE_DISPATCH_OPERATOR_APPROVED:-0}" != "1" ]]; then
  cat <<EOF
[REFUSE] STAGED_PHASE_DISPATCH_OPERATOR_APPROVED != 1

This script is PRE-STAGED only. It REFUSES to dispatch without explicit
operator approval per CLAUDE.md "Phase 2 GPU dispatch budget" non-negotiable
+ 2026-05-11 operator directive.

To dispatch:
  STAGED_PHASE_DISPATCH_OPERATOR_APPROVED=1 \\
  STAGED_PHASE_DISPATCH_LANE=${LANE_ID} \\
  STAGED_PHASE_DISPATCH_BUDGET_USD=${COST_BAND_USD} \\
  bash $0

The dispatch will:
  1. Write provenance to ${OUTPUT_DIR}/provenance.json
  2. Print the canonical Modal actuator command
  3. The actuator creates the provider job id and opens the lane claim
  4. Auth-eval (CUDA + CPU per dual-eval mandate) on landing
  5. Update lane registry via tools/lane_maturity.py mark
EOF
  exit 0
fi

# Lane-id verification.
if [[ "${STAGED_PHASE_DISPATCH_LANE:-}" != "${LANE_ID}" ]]; then
  echo "[REFUSE] STAGED_PHASE_DISPATCH_LANE='${STAGED_PHASE_DISPATCH_LANE:-}' != '${LANE_ID}'" >&2
  exit 2
fi

# Budget cap.
if [[ -z "${STAGED_PHASE_DISPATCH_BUDGET_USD:-}" ]]; then
  echo "[REFUSE] STAGED_PHASE_DISPATCH_BUDGET_USD must be set explicitly" >&2
  exit 2
fi
if [[ "${STAGED_PHASE_DISPATCH_BUDGET_USD}" -gt "${COST_BAND_USD}" ]]; then
  echo "[REFUSE] Budget cap \$${STAGED_PHASE_DISPATCH_BUDGET_USD} exceeds staged band \$${COST_BAND_USD}" >&2
  echo "    Re-stage manifest if a higher cap is required." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

echo "==> Provider job not created by this pre-stage script"
echo "    No lane claim opened. The canonical Modal actuator opens the claim only"
echo "    after a real Modal call id exists, preventing phantom active claims."

# Provenance.
.venv/bin/python -c "
import json, pathlib, datetime
prov = {
    'schema_version': 1,
    'tool': '${DISPATCH_SCRIPT}',
    'lane_id': '${LANE_ID}',
    'provider': '${PROVIDER}',
    'predicted_delta_score': '${PREDICTED_DELTA_SCORE}',
    'cost_band_usd': ${COST_BAND_USD},
    'operator_approved_budget_usd': ${STAGED_PHASE_DISPATCH_BUDGET_USD},
    'dispatch_started_at_utc': '$(date -u +%Y-%m-%dT%H:%M:%SZ)',
    'evidence_grade_target': 'contest_cuda + contest_cpu (dual-eval mandate)',
    'authoritative_axis': 'TBD (post-eval)',
    'archive_grammar': 'Phase1-three-member-x-decoder-bin-balle-bin',
    'parser_section_manifest': ['x','decoder.bin','balle.bin'],
    'inflate_runtime_loc_budget': 'declared in trainer _write_runtime; verified via Phase 1 packet compiler',
    'runtime_dep_closure': 'torch+brotli+compressai',
    'export_format': 'phase1_contest_contract',
    'score_aware_loss': 'alpha*B/N + beta*d_seg + gamma*sqrt(d_pose); Berger-corrected with rho_pose=0.85 fallback',
    'bolt_on_loc_budget': 'substrate_engineering exception',
    'no_op_detector_planned': 'Phase 1 packet compiler byte-mutation smoke (Catalog #139)',
    'monitoring_requirements': ['eval_roundtrip=True','EMA decay 0.997','rgb_to_yuv6 differentiable','real video data (no make_synthetic)'],
}
pathlib.Path('${OUTPUT_DIR}/provenance.json').write_text(json.dumps(prov, indent=2, sort_keys=True))
print(f'Wrote provenance.json')
"

# Dispatch (Modal T4 canonical Phase 2 trainer).
echo "==> Launching Modal T4 (Phase 2 T1 trainer)..."
echo "    Trainer: experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py"
echo "    Output: ${OUTPUT_DIR}"
echo
echo "[NEXT] Operator: run the actual Modal dispatch command. Suggested:"
echo "  PYTHONPATH=src:upstream:\$PWD .venv/bin/modal run --detach \\"
echo "    experiments/modal_t1_balle_endtoend.py --execute \\"
echo "    --epochs 3000 \\"
echo "    --timeout-hours 24 \\"
echo "    --cost-cap-usd ${STAGED_PHASE_DISPATCH_BUDGET_USD}"
echo
echo "[NEXT] On landing, harvest via:"
echo "  .venv/bin/python tools/harvest_modal_calls.py --lane ${LANE_ID}"
echo
echo "[NEXT] Dual-eval per CLAUDE.md mandate:"
echo "  .venv/bin/python tools/plan_dual_device_auth_eval.py --archive ${OUTPUT_DIR}/archive.zip"

# DO NOT auto-execute the Modal dispatch here. The script's job is to prepare
# everything; the operator runs the final dispatch command after reviewing
# the prepared environment.
echo
echo "==> PRE-STAGED dispatch prepared. Operator-approved environment ready."
echo "    No lane claim opened; provenance written; manifest aligned."
echo "    Run the suggested 'experiments/modal_t1_balle_endtoend.py ...' Modal command above to actually dispatch."
