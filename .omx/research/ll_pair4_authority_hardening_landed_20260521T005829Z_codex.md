# LL Pair-4 Authority Hardening Landed

**Author**: codex  
**UTC**: 2026-05-21T00:58:29Z  
**Trigger**: adversarial review of the pair #4 magic-codec seed-boundary and LL planner landing

## Finding

The pair #4 procedural-seed boundary smoke correctly emitted
`score_claim_valid=false`, `promotion_eligible=false`, and
`ready_for_exact_eval_dispatch=false`, but it did not emit the full false
authority surface consumed elsewhere in the apparatus. A malformed downstream
artifact could carry `score_claim=true` while still passing the prior pair #4
LL boundary normalizer.

The real null-byte matrix artifact also predates the stricter authority shape:
it carries `score_claim=false` and `promotable=false`, but omits
`promotion_eligible`, `ready_for_exact_eval_dispatch`, and
`rank_or_kill_eligible`.

## Fix

`tac.optimization.scorer_response_dataset` now requires explicit false
authority fields by default:

- `score_claim`
- `score_claim_valid` for pair #4 boundary smokes
- `promotion_eligible`
- `ready_for_exact_eval_dispatch`
- `rank_or_kill_eligible`
- `promotable`

The null-byte matrix path now has a deliberately named legacy escape hatch:

```bash
--allow-legacy-null-byte-matrix-missing-authority
```

That flag is only for historical matrices that predate the stricter contract.
It still rejects any truthy authority value and records
`legacy_missing_authority_fields_accepted` in the generated plan. Future
matrices must carry explicit false authority bits.

## Regenerated artifacts

- `experiments/results/magic_codec_pair_4_procedural_seed_orthogonality_smoke_20260521T005751Z/smoke_result.json`
- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_scorer_response_next_probe_plan_null_pair4_guarded_20260521_codex.json`
- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_frame_pair_curriculum_pair4_guarded_20260521_codex.json`

The regenerated pair #4 smoke keeps the empirical result unchanged:
`30/30` canonical reversible seed/order rows are raw-seed dominated, and the
best non-raw wrapper remains `+4` bytes. The result now also records the
selected raw-fallback interpretation separately from the best non-raw positive
rate regression.

## Verification

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_magic_codec_pair_4_procedural_seed_orthogonality_smoke.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py
.venv/bin/python tools/run_magic_codec_pair_4_procedural_seed_orthogonality_smoke.py --output-dir experiments/results/magic_codec_pair_4_procedural_seed_orthogonality_smoke_20260521T005751Z
.venv/bin/python tools/plan_ll_scorer_response_next.py --dataset experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/scorer_response_dataset_20260520_codex.json --null-byte-matrix experiments/results/null_byte_probe_matrix_20260520T223742Z/null_byte_matrix.json --allow-legacy-null-byte-matrix-missing-authority --magic-codec-seed-boundary-smoke experiments/results/magic_codec_pair_4_procedural_seed_orthogonality_smoke_20260521T005751Z/smoke_result.json --json-out experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_scorer_response_next_probe_plan_null_pair4_guarded_20260521_codex.json --md-out experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_scorer_response_next_probe_plan_null_pair4_guarded_20260521_codex.md
.venv/bin/python tools/build_ll_frame_pair_curriculum.py --frame-axis-npy experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/master_gradient_frame_axis_l1_20260520_codex.npy --decomposition-json experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/master_gradient_frame_decomposition_20260520_codex.json --response-plan-json experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_scorer_response_next_probe_plan_null_pair4_guarded_20260521_codex.json --json-out experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_frame_pair_curriculum_pair4_guarded_20260521_codex.json --md-out experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_frame_pair_curriculum_pair4_guarded_20260521_codex.md
```

Result:

- focused tests: `24 passed`
- pair #4 boundary unchanged
- LL plan regenerated with explicit legacy-missing authority disclosure
- curriculum regenerated from the hardened plan

