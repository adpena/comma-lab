# Codex Findings - Z7-Mamba-2 Reference Torch Handoff Blocker Cleanup - 2026-05-19T20:39Z

Author: Codex
Lane: `lane_z7_mamba2_reference_torch_exact_handoff_blocker_cleanup_20260519`
Task: `z7_mamba2_reference_torch_exact_handoff_blocker_cleanup_20260519::RECIPE_BLOCKER_V2`

## Finding

The Z7-Mamba-2 recipe still listed
`z7_mamba2_reference_torch_runtime_exact_handoff_must_validate_before_paid_dispatch`
as an active dispatch blocker after later evidence had already landed:

- `.omx/research/probe_z7_mamba2_temporal_coherence_vs_static_capacity_disambiguator_20260519T155511Z_codex.json`
- `.omx/research/codex_findings_z7_mamba2_score_aware_2pair_handoff_20260519T144801Z_codex.md`

That evidence validates the scorer-free `reference_torch` runtime handoff at
the 600-pair recurrent/static probe surface with paired contest-CUDA and
supplemental contest-CPU exact-eval artifacts. It does not validate the packet
as frontier-competitive.

## Change

Updated
`.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml`
to:

1. remove only the validated reference_torch exact-handoff blocker,
2. add a `dispatch_blockers_cleared` evidence row for that blocker,
3. preserve `research_only: true`,
4. preserve `dispatch_enabled: false`,
5. preserve the remaining Z7-GRU, Wave N+1 council, C6 beta-anchor, and
   identity-disambiguator blockers.

## Authority Boundary

No score claim, promotion claim, rank/kill claim, or dispatch-ready claim.
The current probe reports:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_paid_dispatch=false`

The cleared blocker is narrow: reference_torch runtime handoff evidence is
present. The recipe remains non-dispatchable until the remaining blockers are
cleared through their own authority surfaces.

## Adversarial Review Fixes

Nash xhigh review caught four landing hazards before commit:

1. an existing readiness test still expected the blocker to remain active,
2. the initial canonical task row pointed at an older 2-pair handoff memo,
3. the recipe `exact_eval_path` still mentioned exact-handoff as blocked,
4. unquoted YAML timestamp parsing could coerce `cleared_at_utc` into a
   datetime object.

Fixes landed in this iteration: readiness test updated, initial task cancelled
append-only and replaced with `RECIPE_BLOCKER_V2`, `exact_eval_path` wording
removed the stale exact-handoff blocker, and `cleared_at_utc` is quoted plus
tested as a string.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools:upstream .venv/bin/python -m pytest -q -p no:cacheprovider src/tac/tests/test_z7_mamba2_recipe_blocker_cleanup.py src/tac/tests/test_verify_z7_exact_eval_handoff.py src/tac/tests/test_probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py::test_assess_candidate_z7_mamba2_scaffold_is_visible_with_score_band_axis`
  - `23 passed`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check src/tac/tests/test_z7_mamba2_recipe_blocker_cleanup.py src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py`
  - passed
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/operator_authorize.py --recipe substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch --dry-run`
  - refused as expected: `dispatch_enabled=false` plus five remaining blockers
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/canonical_task_status.py --validate`
  - valid
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/lane_maturity.py validate`
  - valid
- `git diff --check`
  - clean

## Next Z7-Mamba-2 Blockers

The next concrete blocker-burndown candidates are:

1. register or consume the Z7-GRU Wave 2 disambiguator outcome,
2. resolve the C6 IBPS Phase 2 beta-anchor dependency or record a scoped
   operator-frontier-override for stability-smoke only,
3. run the Wave N+1 council only after the above evidence is current,
4. decide whether a new recurrent/static candidate warrants paid dispatch.
