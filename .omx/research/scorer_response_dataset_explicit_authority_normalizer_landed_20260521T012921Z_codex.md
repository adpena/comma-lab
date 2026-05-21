# Scorer-Response Dataset Explicit-Authority Normalizer Landed

generated_at: 2026-05-21T01:29:21Z
agent: codex
lane: ll_scorer_response_dataset_authority_normalizer
verdict: LANDED
scope: fail-closed compatibility tooling

## Summary

Added a reusable normalization path for historical `scorer_response_dataset.v1`
artifacts that predate the later explicit false authority fields:

- `tac.optimization.scorer_response_dataset.normalize_legacy_response_dataset_authority(...)`
- `tools/normalize_scorer_response_dataset_authority.py`

The normalizer only inserts missing historical extended fields as explicit
`false`:

- `rank_or_kill_eligible`
- `promotable`

It refuses to infer or backfill core authority fields:

- `score_claim`
- `promotion_eligible`
- `ready_for_exact_eval_dispatch`

It also refuses missing or non-false row-level `authority_source_score_claim`.
This keeps old advisory datasets usable for PDS/LL training-data plumbing
without weakening the false-authority contract.

## Empirical Artifact

Normalized historical PR110 scorer-response dataset:

`experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/scorer_response_dataset_20260520_codex_explicit_authority_20260521_codex.json`

Normalization summary:

```text
backfilled_missing_false_field_count: 62
score_claim: false
```

The normalized artifact strict-loads through the PACT-NERV-DistilledScorer
loader without `allow_legacy_missing_authority=True`.

## Verification

```text
.venv/bin/python tools/normalize_scorer_response_dataset_authority.py \
  --input experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/scorer_response_dataset_20260520_codex.json \
  --json-out experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/scorer_response_dataset_20260520_codex_explicit_authority_20260521_codex.json \
  --source-label pr110_provisional_hfv1_engineering_20260520_codex/scorer_response_dataset_20260520_codex.json

STRICT_OK 3 62

.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py src/tac/substrates/pact_nerv_distilled_scorer/tests/test_pact_nerv_distilled_scorer.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py
38 passed in 0.78s

.venv/bin/python -m py_compile src/tac/optimization/scorer_response_dataset.py tools/normalize_scorer_response_dataset_authority.py
passed
```

## Next Action

Future PACT-NERV-DistilledScorer Stage 1 work should consume the normalized
explicit-authority artifact, not the legacy artifact plus compatibility flag.
The compatibility flag remains useful only for forensic reads of historical
data.
