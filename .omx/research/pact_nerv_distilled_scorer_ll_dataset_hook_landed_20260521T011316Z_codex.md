# PACT-NERV-DistilledScorer LL Dataset Hook Landed

generated_at: 2026-05-21T01:13:16Z
agent: codex
lane: pact_nerv_distilled_scorer_x_codex_ll_dataset_hook
verdict: LANDED
scope: implementation hook + fail-closed authority contract

## Summary

The PACT-NERV-DistilledScorer substrate now exposes an explicit consumer
contract for Codex's LL scorer-response dataset surface:

- `CONSUMES_SCORER_RESPONSE_DATASET = True`
- `SCORER_RESPONSE_DATASET_SCHEMA = "scorer_response_dataset.v1"`
- `load_scorer_response_distill_rows(...)`

This is training-data plumbing only. It does not create a score claim, promotion
claim, or exact-eval dispatch claim.

## Factual Grounding

The sister design memo
`.omx/research/pact_nerv_distilled_scorer_x_codex_ll_integration_design_20260520.md`
identifies the three-surface convergence:

- PACT-NERV-DistilledScorer as the substrate implementation surface.
- Codex LL planner/scorer-response dataset as the observability and data-harvest
  surface.
- Catalog #523 / Hinton KL-T=2.0 as the canonical distillation lane.

The code landing implements only the minimal durable hook needed for those
surfaces to compose.

## Authority Contract

The loader fails closed unless all of the following are explicit `false` on the
dataset, the dataset `authority` block, and every consumed row:

- `score_claim`
- `promotion_eligible`
- `ready_for_exact_eval_dispatch`
- `rank_or_kill_eligible`
- `promotable`

Rows with `authority_source_score_claim = true` are refused. Rows must also
carry finite numeric `advisory_score_report_derived` and
`delta_vs_baseline_score` fields. This keeps advisory response observations
usable for surrogate/curriculum construction while preventing false contest
authority.

## Files

- `src/tac/substrates/pact_nerv_distilled_scorer/score_aware_loss.py`
- `src/tac/substrates/pact_nerv_distilled_scorer/__init__.py`
- `experiments/train_substrate_pact_nerv_distilled_scorer.py`
- `src/tac/optimization/scorer_response_dataset.py`
- `src/tac/substrates/pact_nerv_distilled_scorer/tests/test_pact_nerv_distilled_scorer.py`
- `src/tac/tests/test_scorer_response_dataset.py`
- `reports/latest.md`

## Verification

```text
.venv/bin/python -m pytest -q src/tac/substrates/pact_nerv_distilled_scorer/tests/test_pact_nerv_distilled_scorer.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py
32 passed in 0.84s

.venv/bin/python -m py_compile src/tac/substrates/pact_nerv_distilled_scorer/score_aware_loss.py src/tac/substrates/pact_nerv_distilled_scorer/__init__.py src/tac/optimization/scorer_response_dataset.py experiments/train_substrate_pact_nerv_distilled_scorer.py
passed
```

## Next Action

Use the hook from the future PACT-NERV-DistilledScorer Stage 1 trainer after the
per-substrate symposium gate. Do not dispatch paid training from this landing;
the substrate remains research-only/operator-gated.
