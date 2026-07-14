# Task503 v9 admission contract correction

## Objective

Make the default-OFF DecisionCarrierBundle admission satisfiable only by the real Task503 gate:
an alternate receiver changes bytes/behavior, parses back, measures all 600 pairs through R with
frozen scorers, has equal-or-lower `d_seg` (and non-harmful `d_pose` under its declared trust gate),
and reduces exact counted `archive.zip` bytes. Bit-identical output is not required and must not be
conflated with score-equivalent/non-worse behavior.

## Required correction

- Remove the contradiction where the consume receipt requires enabled and disabled output hashes
  to differ while the rate receipt requires baseline and alternate behavior hashes to be equal.
- Require distinct valid baseline/alternate output hashes as genuine actuation evidence.
- Require positive exact archive-byte savings (`baseline_archive_bytes > alternate_archive_bytes`)
  plus exact archive SHA-256 custody; `0.bin` diagnostics alone cannot admit.
- Require measured baseline/alternate full-n600 `d_seg`, with alternate `<=` baseline, and explicit
  full-n600 Pose trust/non-harm status. Preserve `score_claim=false` and pointer unchanged.
- The current Task503 rate receipt must remain non-admissible.
- Preserve canonical metric `argmax_native_vjp_fidelity_v1`, exact pair coverage, parse-back,
  resume, no raw RGB, textured/non-flat palette, uint8, and Pose6 gates.

## Ownership and acceptance

Edit only `src/tac/witness_dsl/decision_carrier_policy.py` and its focused test. Do not touch v9
hot files. Run focused pytest, Ruff check/format, `py_compile`, and `git diff --check`; do not
commit.
