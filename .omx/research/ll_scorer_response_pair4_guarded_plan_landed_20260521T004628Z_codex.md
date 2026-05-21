# LL Scorer Response Pair-4 Guarded Plan Landed

**Author**: codex  
**UTC**: 2026-05-21T00:46:28Z  
**Primary files**:

- `src/tac/optimization/scorer_response_dataset.py`
- `tools/plan_ll_scorer_response_next.py`
- `src/tac/tests/test_scorer_response_dataset.py`
- `src/tac/tests/test_plan_ll_scorer_response_next_cli.py`

## What changed

The LL scorer-response planner now consumes the pair #4 procedural-seed
orthogonality smoke as a first-class boundary input.

New optional CLI input:

```bash
--magic-codec-seed-boundary-smoke <pair4_smoke_result.json>
```

When the pair #4 smoke verdict is
`PAIR_4_BOUNDARY_VALIDATED_RAW_SEED_DOMINATES`, the planner adds the explicit
prohibition:

```text
do_not_wrap_procedural_seed_bytes_with_magic_codec
```

This lets the LL planner still prioritize null-byte / master-gradient training
harvest while refusing the now-closed seed-wrapping path. The correct route is
raw procedural seeds plus residual/runtime streams, not a magic-codec envelope
around the seed itself.

## Boundary evidence consumed

Source smoke:

`experiments/results/magic_codec_pair_4_procedural_seed_orthogonality_smoke_20260521T010000Z/smoke_result.json`

Key fields:

- canonical reversible seed/order rows: `30`
- raw seed dominates rows: `30`
- minimum best-nonraw delta versus raw: `+4` bytes
- score claim: `false`
- promotion eligible: `false`

## Fresh guarded plan

Generated artifact:

`experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_scorer_response_next_probe_plan_null_pair4_guarded_20260521_codex.json`

The guarded plan has two prohibitions:

1. `do_not_widen_coordinate_sparse_residual_sidecar`
2. `do_not_wrap_procedural_seed_bytes_with_magic_codec`

Top null-byte priority rows remain routing priors only:

| Rank | Substrate | Null bytes | LL sampling weight |
|---:|---|---:|---:|
| 1 | `pr106_format0d` | `16909` | `0.208098` |
| 2 | `pr101_fec6_frontier` | `16292` | `0.200498` |
| 3 | `a1_finetuned` | `16037` | `0.197356` |
| 4 | `pr101_lc_v2` | `16033` | `0.197307` |
| 5 | `pr107_apogee` | `15987` | `0.196740` |

All rows remain `score_claim=false`, `promotion_eligible=false`,
`ready_for_exact_eval_dispatch=false`.

## Compatibility fix

The real null-byte matrix artifact predates the stricter authority metadata
shape and omits `promotion_eligible` / `rank_or_kill_eligible`. The consumer now
treats missing authority bits as legacy non-promotional metadata while still
rejecting any explicit truthy value. This preserves fail-closed behavior for
new artifacts and allows the current canonical matrix to be consumed.

## Verification

Commands run:

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py
.venv/bin/python -m py_compile tools/plan_ll_scorer_response_next.py
.venv/bin/python tools/plan_ll_scorer_response_next.py --dataset experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/scorer_response_dataset_20260520_codex.json --null-byte-matrix experiments/results/null_byte_probe_matrix_20260520T223742Z/null_byte_matrix.json --magic-codec-seed-boundary-smoke experiments/results/magic_codec_pair_4_procedural_seed_orthogonality_smoke_20260521T010000Z/smoke_result.json --json-out experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_scorer_response_next_probe_plan_null_pair4_guarded_20260521_codex.json --md-out experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_scorer_response_next_probe_plan_null_pair4_guarded_20260521_codex.md
```

Result:

- tests: `14 passed`
- guarded plan generated
- score claim: `false`
- promotion eligible: `false`

