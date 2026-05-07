# Field-Equation Tranche - 2026-05-06

Scope: planning-only cross-paradigm atom ledger and field-equation plan built
from the current exact local HNeRV frontier anchor. No GPU or remote dispatch
was attempted. No score claim is made for any new candidate in this tranche.

## Exact Anchor

- Evidence anchor: `experiments/results/lightning_batch/exact_eval_pr106x_lowlevel_brotli_repack_custody_v2_t4_20260506/contest_auth_eval.adjudicated.json`
- Score: `0.20935073680571203`
- Pose distance: `0.00003351`
- Seg distance: `0.00067142`
- Archive bytes: `186080`
- Archive SHA-256: `b0a12549a39e34a0d7f83ea99e05e55fcd01d795a15db2ffb3d92ccc6267e53f`

## Generated Planning Artifacts

Ignored raw artifacts:

- `.omx/research/artifacts/field_equation_tranche_20260506_codex/cross_paradigm_atom_ledger.json`
- `.omx/research/artifacts/field_equation_tranche_20260506_codex/field_equation_plan.json`

Build commands:

```bash
.venv/bin/python tools/build_cross_paradigm_atom_ledger.py \
  --base-pose-dist 0.00003351 \
  --source exact_frontier_pr106x_lowlevel_brotli_repack_t4_20260506 \
  --hnerv-rate-recode-profile experiments/results/hnerv_decoder_recode_pr106_20260506_codex/profile.json \
  --wr01-wavelet-plan experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/manifest.json \
  --categorical-mask-plan experiments/results/categorical_openpilot_payload_candidate_20260506_codex/construction_plan.json \
  --lapose-plan .omx/research/artifacts/lapose_motion_atoms_20260505_codex/lane_w_component_allocated_lapose_motion_atom_manifest.json \
  --json-out .omx/research/artifacts/field_equation_tranche_20260506_codex/cross_paradigm_atom_ledger.json

.venv/bin/python tools/build_field_equation_plan.py \
  --atom-ledger .omx/research/artifacts/field_equation_tranche_20260506_codex/cross_paradigm_atom_ledger.json \
  --source exact_frontier_pr106x_lowlevel_brotli_repack_t4_20260506 \
  --base-score 0.20935073680571203 \
  --json-out .omx/research/artifacts/field_equation_tranche_20260506_codex/field_equation_plan.json
```

## Result

- Atom rows: `39`
- Pareto-eligible rows: `1`
- KKT-ready rows: `1`
- Exact-eval dispatch readiness: `false`

The only KKT-clean locally descending row is:

| atom | family | byte delta | expected score delta | status |
|---|---:|---:|---:|---|
| `wr01_wavelet_apply:pr106x:latents_and_sidecar_brotli` | `wr01_wavelet_apply_transform` | `-9` | `-0.000005992731` | KKT-clean planning row |

LA-pose motion atoms remain intentionally blocked in the field-equation plan:

- `missing_byte_closed_archive_manifest`
- `pareto_ineligible_atom`
- `seg_violation`

This is the correct mathematical behavior: diagnostic CUDA/global-response
motion atoms can guide the next build, but they are not allowed to dominate the
dispatch queue until they become byte-bearing archive atoms with runtime
consumption and component-collapse controls.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_cross_paradigm_atoms.py \
  src/tac/tests/test_field_equation_planner.py \
  src/tac/tests/test_meta_lagrangian_allocator.py \
  -q
```

Result: `32 passed`.

## Next Action

The immediate score-lowering path is not another abstract sweep. It is to
convert either WR01 or the categorical/openpilot payload into a byte-closed
candidate with:

1. charged archive member manifest,
2. decoded-output equivalence or explicit scorer-changing scope,
3. runtime consumer proof,
4. strict candidate preflight,
5. lane claim before exact CUDA auth eval.
