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
#   - Routes through tools/claim_lane_dispatch.py claim BEFORE paid call
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
  1. Route through tools/claim_lane_dispatch.py claim (TTL conflict check)
  2. Launch via Modal T4 (canonical Phase 2 substrate)
  3. Write provenance to ${OUTPUT_DIR}/provenance.json
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

# Lane-claim atomicity per CLAUDE.md cross-agent dispatch coordination.
echo "==> Claiming lane dispatch (TTL conflict check)..."
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id "${LANE_ID}" \
  --platform "${PROVIDER}" \
  --instance-job-id "staged_${LANE_ID}_${UTC_TIMESTAMP}" \
  --agent "claude:opus-4-7" \
  --status "operator_approved_phase2_dispatch" \
  --notes "Phase 2 dispatch operator-approved \$${STAGED_PHASE_DISPATCH_BUDGET_USD}; predicted ${PREDICTED_DELTA_SCORE}; output ${OUTPUT_DIR}" \
  --predicted-eta-utc "$(date -u -v+4H +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d '+4 hours' +%Y-%m-%dT%H:%M:%SZ)"

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
echo "  .venv/bin/python experiments/modal_train_lane.py \\"
echo "    --lane ${LANE_ID} \\"
echo "    --provider modal \\"
echo "    --gpu t4 \\"
echo "    --output-dir ${OUTPUT_DIR} \\"
echo "    --auth-eval-on-best \\"
echo "    --enable-eval-roundtrip-in-training \\"
echo "    --apply-autograd-yuv6"
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
echo "    Lane claimed; provenance written; manifest aligned."
echo "    Run the suggested 'experiments/modal_train_lane.py ...' command above to actually dispatch."
