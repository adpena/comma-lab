#!/usr/bin/env bash
# Operator-authorize wrapper: HDM8 film-grain postfilter / per-pair selector
# exact-CUDA auth-eval dispatch on Modal T4.
#
# Recipe: ``.omx/operator_authorize_recipes/hdm8_film_grain_selector_modal_t4_auth_eval_dispatch.yaml``
#
# This wrapper is AUTH-EVAL-ONLY (NOT a substrate training dispatch). Therefore
# it does NOT route through `tools/run_modal_smoke_before_full.py` (Catalog
# #167 fires on `cost_band.epochs >= 1000`; auth-eval has `epochs=0`) and does
# NOT route through `tools/operator_authorize.py --recipe` (its Modal flow
# hardcodes `experiments/modal_train_lane.py` which is wrong for
# auth-eval-on-existing-archive). Per CLAUDE.md "NEVER invent CLI flags" the
# right surface is `experiments/modal_auth_eval.py` directly, which is what
# `tools/build_hdm8_film_grain_sidecar_packet.py` already emits in the
# canonical `exact_cuda_auth_eval_command_template`.
#
# Per Catalog #199 (`check_operator_authorize_bypass_requires_session_budget`):
# this wrapper does NOT use the `OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE`
# bypass. Instead it surfaces the canonical command + readiness manifest +
# proof status to the operator for explicit review, then exits.
#
# Per Catalog #189 (shell empty arrays guarded under set -u): no array
# expansions in this wrapper — only positional args.
#
# Lane: lane_hdm8_film_grain_selector_dispatch_20260514
# Sister gate: codex Modal sweep hdm8_modal_t4_postfilter_policy_sweep_20260514
# (ETA 14:32Z 2026-05-14) will produce CUDA-in-loop per-pair component arrays
# for credible per-pair selector construction; until that lands, dispatch
# decisions are operator-routable per CLAUDE.md "Design decisions" non-negotiable.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RECIPE_NAME="hdm8_film_grain_selector_modal_t4_auth_eval_dispatch"
RECIPE_PATH=".omx/operator_authorize_recipes/${RECIPE_NAME}.yaml"
LANE_ID="lane_hdm8_film_grain_selector_dispatch_20260514"

# Mode selection: caller may override via HDM8_MODE env var. Default to the
# best 600-pair MPS-positive global mode (smallest signal, safest CUDA bet).
HDM8_MODE="${HDM8_MODE:-even_grain_chroma:1.0}"
HDM8_PACKET_OUTPUT_DIR="${HDM8_PACKET_OUTPUT_DIR:-experiments/results/hdm8_film_grain_selector_dispatch_20260514/packet_$(echo "${HDM8_MODE}" | tr ':,.' '___')}"

# Source archive (the HDM8 base from codex's recode landing).
HDM8_ARCHIVE="${HDM8_ARCHIVE:-experiments/results/pr106_r2_hdm7_hlm2_hdm8_candidate_20260514_codex/exact_eval_static_release_surface/archive.zip}"

# Required-input-file validation BEFORE any Modal dispatch (per Catalog #152).
if [ ! -f "${HDM8_ARCHIVE}" ]; then
    echo "ERROR: HDM8 archive not found: ${HDM8_ARCHIVE}" >&2
    echo "Generate the HDM8 base archive via the codex recode pipeline first." >&2
    exit 2
fi
if [ ! -f ".omx/operator_authorize_recipes/${RECIPE_NAME}.yaml" ]; then
    echo "ERROR: recipe not found: ${RECIPE_PATH}" >&2
    exit 2
fi

cat <<EOF
============================================================
HDM8 film-grain postfilter / per-pair selector exact-CUDA dispatch
============================================================
Lane:        ${LANE_ID}
Recipe:      ${RECIPE_PATH}
Mode:        ${HDM8_MODE}
Archive:     ${HDM8_ARCHIVE}
Output dir:  ${HDM8_PACKET_OUTPUT_DIR}

Strategic gate per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA":
- Sub-0.188 score gate NOT yet cleared (HDM8 base 0.20636 [contest-CUDA T4]).
- This dispatch is RESEED ONLY into continual-learning posterior.
- NO PR submission per operator directive.
- Best-case selector advisory is -0.014 (64-pair MPS) which would land
  ~0.191 — still ABOVE the sub-0.188 gate.

Pre-dispatch proof status (codex section 7 mitigations):
- Frame-parity proof: PASSING for top-6 MPS modes (n_odd_mismatches=0)
  experiments/results/hdm8_film_grain_selector_dispatch_20260514/frame_parity_no_op_proof_top6_64pairs.json
- No-op pixel-delta proof: PASSING (all modes change >87M bytes)
- Selector byte-charging proof: builder supports format_id=0x03

============================================================
EOF

# STAGE 1: Build the candidate packet (no GPU spend; local).
if [ -d "${HDM8_PACKET_OUTPUT_DIR}" ]; then
    echo "[stage 1] candidate packet already exists at ${HDM8_PACKET_OUTPUT_DIR} - reusing"
else
    echo "[stage 1] building candidate packet for mode=${HDM8_MODE}"
    .venv/bin/python tools/build_hdm8_film_grain_sidecar_packet.py \
        --archive "${HDM8_ARCHIVE}" \
        --runtime-template submissions/hdm8_film_grain_sidecar \
        --output-dir "${HDM8_PACKET_OUTPUT_DIR}" \
        --mode "${HDM8_MODE}" \
        --proxy-json experiments/results/hdm8_postfilter_sweep_20260514_codex/proxy_600pairs_even_palette2_mps.json
fi

# STAGE 2: Frame-parity + no-op proof (cheap CPU smoke; ~30s on 16 pairs).
PROOF_JSON="${HDM8_PACKET_OUTPUT_DIR}/frame_parity_no_op_proof.json"
if [ -f "${PROOF_JSON}" ]; then
    echo "[stage 2] proof already exists at ${PROOF_JSON} - reusing"
else
    echo "[stage 2] running frame-parity + no-op proof (16 pairs CPU)"
    .venv/bin/python tools/probe_hdm8_postfilter_frame_parity_and_no_op.py \
        --archive "${HDM8_ARCHIVE}" \
        --runtime-template submissions/hdm8_film_grain_sidecar \
        --modes "none,${HDM8_MODE}" \
        --n-pairs 16 \
        --device cpu \
        --output-json "${PROOF_JSON}"
fi

# STAGE 3: Print the canonical exact-eval command from the packet manifest.
PACKET_MANIFEST="${HDM8_PACKET_OUTPUT_DIR}/packet_manifest.json"
if [ ! -f "${PACKET_MANIFEST}" ]; then
    echo "ERROR: packet manifest not found at ${PACKET_MANIFEST}" >&2
    exit 3
fi

echo
echo "============================================================"
echo "STAGE 3: canonical exact-CUDA auth-eval command (operator-routable)"
echo "============================================================"
.venv/bin/python -c "
import json, shlex
m = json.load(open('${PACKET_MANIFEST}'))
print('archive_sha256:', m['archive']['sha256'])
print('archive_bytes:', m['archive']['bytes'])
print('runtime_tree_sha256:', m['runtime']['runtime_tree_sha256'])
print('modal_uploaded_runtime_tree_sha256:', m['runtime']['modal_uploaded_runtime_tree_sha256'])
print()
print('To dispatch, FIRST claim the lane:')
print('  ' + ' '.join(shlex.quote(x) for x in m['claim_command_template']))
print()
print('Then dispatch:')
print('  ' + ' '.join(shlex.quote(x) for x in m['exact_cuda_auth_eval_command_template']))
print()
print('After dispatch, harvest via tools/harvest_modal_calls.py within 24h')
print('per CLAUDE.md \"Modal .spawn() HARVEST OR LOSE\" non-negotiable.')
print()
print('dispatch_blockers:')
for b in m.get('dispatch_blockers', []):
    print(f'  - {b}')
"

echo
echo "============================================================"
echo "STRATEGIC NEXT-STEP DECISION (operator-routable)"
echo "============================================================"
echo "Per codex's frame-exploit research section 7 + sub-0.188 gate math:"
echo
echo "Option A: dispatch the printed command NOW for global mode '${HDM8_MODE}'"
echo "          (small MPS signal -0.000338 on 600 pairs; CUDA inversion plausible;"
echo "          predicted band 0.180-0.220; ~\$0.30 Modal T4)"
echo
echo "Option B: WAIT for codex's CUDA-in-loop palette sweep"
echo "          (hdm8_modal_t4_postfilter_policy_sweep_20260514, ETA 14:32Z),"
echo "          then build a CUDA-credible per-pair selector via"
echo "          tools/build_hdm8_film_grain_sidecar_packet.py \\"
echo "              --selector-from-proxy-json <modal_t4_cuda_per_pair.json> \\"
echo "              --pack-selector-into-archive (format_id=0x03)"
echo "          (selector advisory -0.014 on 64-pair MPS; CUDA-credible after sweep)"
echo
echo "Option C: skip dispatch entirely; this wave's score-bearing path is"
echo "          NOT THIS — it's the cumulative wave (selector + YUV6 sublattice"
echo "          atoms + motion-aligned translate) per codex Top 10 list to clear"
echo "          sub-0.188 gate. HDM8 single-mode dispatch is reseed only."
echo
echo "Default: Option C (no dispatch from this wrapper)."
echo "============================================================"
