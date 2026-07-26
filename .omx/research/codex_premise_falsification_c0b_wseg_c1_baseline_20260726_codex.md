# Codex premise falsification — W_seg as the compact C1 baseline

UTC: 2026-07-26

Lane: `lane_codex_original_taskspace_inverse_codec_20260725`

Mode: `research_only=true`; score claim `false`; pointer moved `false`.

## Premise tested

The C0B composition route provisionally treated the tiny original W_seg
semantic program as the decoded C1 baseline and the exact 17,926-event S2
packet as its target syndrome.

## Exact test

`tools/measure_c0b_wseg_c1_baseline_join.py` regenerated the live M2 target
labels from custodied raw with the frozen batch-16 CPU scorer, reversed the
geometry-bound S2 events to reconstruct C1, decoded W_seg through its actual
receiver semantic-cell API, and compared all 117,964,800 n600 sites. Thirty-
eight write-once stages and a final 14-input end barrier preserve custody.

Receipt:
`.omx/research/original_taskspace_inverse_witness_codec_20260725/c0b_bj1_wseg_c1_baseline_join.json`

## Verdict

`FALSIFIED: WSEG_SEMANTIC_PROGRAM_DOES_NOT_INITIALIZE_THE_EXACT_C1_BASELINE`.

- total W_seg-to-C1 mismatches: 59,814,423 (`0.5070531463623047`);
- S2 event sites with expected W_seg baseline: 7,039;
- S2 event sites with a different W_seg baseline: 10,887;
- non-event residual: 59,803,536;
- S2 apply-back to its actual C1 baseline: exact;
- live-target/cache mismatches: the expected 3.

## Scope and routing consequence

This falsifies only W_seg-as-C1-initializer. It does not kill W_seg as a
separately identified control, S2 as an exact C1 syndrome/teacher, or V9
semantic factorization. Candidate construction must define one exact decoded
V9 predictor, bind residuals to its program/renderer/semantic identities, and
recompute the syndrome against that predictor. The landed
`tac.witness_dsl.predictor_bound_residual` ABI makes cross-predictor residual
application fail closed.

No archive, official evaluation, promotion, or dispatch was produced.

HISTORICAL_PROVENANCE: append-only premise-falsification receipt.
