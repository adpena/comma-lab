# Codex Premise Falsification: Sigma=15 SCPP Over-Inclusion

Timestamp: 2026-05-19T21:19:27Z  
Owner: codex  
Scope: CLUSTER_F1 premise verification  
Result: premise partially falsified before dispatch

## Falsified Premise

The phrase "sigma grid x 5 consumers" over-includes SCPP as if all five named surfaces consumed the same grayscale-LUT bandwidth parameter.

That is false.

SCPP's `sigma=15` is a block-FP weight-scale cutoff:

- `SCPP_DEFAULT_SIGMA: int = 15`
- config serializes/deserializes `sigma` as an integer
- pack path uses `scales = sigma * 2**exponents`

This is not the same scalar as `tac.mask_grayscale_lut.create_gaussian_softmax_lut(sigma=...)`.

## Why It Matters

Applying `{0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0}` blindly to SCPP would be false authority:

- `0.5` is not archive-safe through current integer deserialization;
- block-FP sigma changes weight quantization error/rate, not SegNet class-probability softness;
- results would not answer the grayscale-LUT question.

## Corrected Routing

- Grayscale-LUT sweep: Lane MM, Lane AL, SegMap fixed-soft/LCT after parameter injection, Szabolcs after builder/prebuilt-LUT discipline.
- SCPP sweep: separate block-FP integer-cutoff sweep, with payload-byte and reconstruction-error metrics.

This was caught before provider dispatch, scorer invocation, or paid spend.


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:codex-premise-falsification-memo-trigger-tokens-describe-sigma15-SCPP-overinclusion-not-new-equation-falsification-of-single-config -->
