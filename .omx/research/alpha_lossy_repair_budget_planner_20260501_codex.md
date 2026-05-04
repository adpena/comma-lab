# Alpha Lossy Repair Budget Planner - 2026-05-01

## Scope

Added `experiments/alpha_lossy_repair_budget_planner.py` as an empirical
planning scaffold for Alpha lossy geometry plus charged sparse repair.

The tool ingests:

- `alpha_mask_codec_candidate_matrix_v1` manifests.
- `alpha_mask_primitive_component_response_plan_v1` plans and their archive
  variants manifests.

It emits `alpha_lossy_repair_budget_planner_v1` reports plus
`alpha_lossy_sparse_repair_archive_build_spec_v1` candidate specs. It does not
build archives, run scorer networks, launch remote jobs, or make score claims.

## Evidence Boundary

Evidence grade: `empirical`.

Non-promotable flags are fail-closed at input, report, budget-record, and spec
levels:

- `score_claim=false`
- `promotion_eligible=false`
- `scorer_network_loaded=false`
- `archives_built=false`
- `remote_jobs_launched=false`

The emitted specs require official CUDA component-response before final repair
atom selection and exact CUDA auth eval before any score, rank, promotion,
retirement, or paper empirical claim.

## Verification

Focused checks run locally:

```text
.venv/bin/python -m py_compile experiments/alpha_lossy_repair_budget_planner.py src/tac/tests/test_alpha_lossy_repair_budget_planner.py
.venv/bin/python -m pytest src/tac/tests/test_alpha_lossy_repair_budget_planner.py -q
```

Result: `5 passed`.

Default-manifest smoke run wrote `/tmp/pact_alpha_lossy_repair_budget_planner_verify/alpha_lossy_repair_budget_plan.json`:

```text
records = 39
candidate specs = 12
best exact matrix candidate = coco_rle, 902161 bytes
```

This smoke output remains planning-only and is not a score artifact.
