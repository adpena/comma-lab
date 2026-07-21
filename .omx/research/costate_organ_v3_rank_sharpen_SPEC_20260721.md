# Costate organ v3 rank-sharpening build specification

Date: 2026-07-21

Lane: `lane_costate_v3_rank_sharpen_20260721`
Authority: delegated prompt SHA-256
`d3ec1543d1c656f506d2c4817771795ca8cb9ac738560d76c7be6f536d6eb2a9`

## Objective

Improve the retrospective ordering of the exact-anchor costate organ without
learning parameters or actuating a run.  Measure each additive v3 stage on the
same frozen 24-row v2 development corpus and preserve the advisory-only,
score-neutral boundary.

## Evidence and equations

- Baseline: `.omx/research/costate_organ_v2_exact_anchor_backtest_20260721.json`.
- Graded survival: the sealed r1b7 498-site histogram in
  `.omx/research/r1b7_uint8_survival_carrier_20260720T224624Z.json`.
- Pool marginal: registered LawRef `witness_measured_reverse_waterfill_v1`.
- EMA lag geometry: executable typed-DSL LawRef
  `ema_decay_run_geometry_v1` with constant-decay response `a_h = 1-d^h`.
  The canonical-equations JSONL row is absent, so the receipt records that
  limitation and does not fabricate registration.

The v3 prediction is evaluated as a staged construction:

1. `lambda_v2 = gap * visibility * realizability_v2 * byte_price`.
2. `lambda_survival = gap * visibility * p_clean(route) * byte_price`, where
   the stage probabilities are the r1b7 empirical histogram with an explicit
   Jeffreys `Beta(1/2,1/2)` finite-sample smoothing rule.
3. `lambda_pool` is the KKT marginal at the current allocation: candidates in
   one opportunity pool consume its measured ceiling in descending raw
   marginal order and may not independently claim the same remaining debt.
4. Realized targets use `DeltaS_live_hat = DeltaS_EMA / (1-d^h)` only for rows
   with custodied constant decay/update horizon.  Their inverse-variance weight
   is `(1-d^h)^2`; n/600 subset rows receive precision weight `n/600`.

No target-derived quantity may enter a prediction.  The target transformation
and precision weights affect only the realized side of the weighted metric.

## Implementation ownership

- `src/tac/witness_control/costate_organ_v3.py`: typed pure functions for
  stage-survival probabilities, pool KKT marginals, EMA target de-lagging,
  weighted rank metrics, bootstrap intervals, and the append-only corpus row.
- `tools/costate_organ_v3_backtest.py`: fixed-n=24 read-only evaluator and
  machine-readable receipt emitter.
- `tools/produce_m1_band_manifest.py`: one fail-closed producer call.  It may
  append only if a future receipt contains a fully realized, byte-closed costate
  row; the current dry manifest must emit `NOT_EMITTED`.
- Focused tests under `src/tac/tests/` and `tools/tests/`.
- Canonical append-only corpus:
  `.omx/research/costate_realized_delta_backtest_corpus.jsonl`.
- Durable JSON, result memo, DAG feed, and equations note under `.omx/research/`.

## Metrics and acceptance

Every stage reports:

- ordinary Spearman;
- weighted Spearman on the target-validity weights;
- top-8 positive-benefit precision;
- NDCG@8 with nonnegative realized benefit as decision gain;
- prediction tie count and top-8 IDs;
- paired deterministic bootstrap 95% confidence interval for each delta from
  the preceding stage and from v2.

Acceptance requires all of the following:

1. exactly the same 24 row IDs as v2, with source hashes unchanged;
2. zero learned parameters and no live-run/process/provider action;
3. r1b7 receipt SHA, registered reverse-waterfill LawRef, and executable
   typed-DSL EMA LawRef verified without inventing a registry row;
4. each stage plus leave-one-factor-out ablations measured;
5. top-8 precision and NDCG@8 reported next to Spearman, with disagreement
   stated explicitly;
6. corpus appender rejects malformed, non-finite, duplicate-conflicting, or
   uncustodied rows and is safe under an exclusive file lock;
7. current M1 manifest does not append a fake realized row;
8. focused tests, Ruff, two review-tracker passes for each Python file, and
   serializer commits using post-edit SHA custody pass.

Exact verification commands:

```text
uv run --with pytest --with scipy pytest -q src/tac/tests/test_costate_organ_v3.py tools/tests/test_costate_organ_v3_backtest.py src/tac/boundary_math/tests/test_m1_band_manifest_producer.py
uv run ruff check src/tac/witness_control/costate_organ_v3.py tools/costate_organ_v3_backtest.py tools/produce_m1_band_manifest.py src/tac/tests/test_costate_organ_v3.py tools/tests/test_costate_organ_v3_backtest.py src/tac/boundary_math/tests/test_m1_band_manifest_producer.py
.venv/bin/python tools/costate_organ_v3_backtest.py --v2-receipt .omx/research/costate_organ_v2_exact_anchor_backtest_20260721.json --r1b7-receipt .omx/research/r1b7_uint8_survival_carrier_20260720T224624Z.json --bootstrap-replicates 10000 --created-utc 2026-07-21T02:25:46Z --seed-corpus --output .omx/research/costate_organ_v3_rank_sharpen_20260721.json
```

## Explicitly out of scope

- No live run writes, signals, dispatch, training, scorer invocation, archive
  mutation, or frontier-pointer mutation.
- No learned ranker and no default-OFF learned component.
- No n600, contest-CPU, or contest-CUDA score claim.
- No claim that byte price is dead while the fixed corpus has no byte-paying
  row.
- No negative verdict beyond the exact formulation and corpus measured here.
- Do not touch main, upstream, witness autoconfig, trainer DSL, or live run dirs.

MAIN must perform landing review before this branch is merged.
