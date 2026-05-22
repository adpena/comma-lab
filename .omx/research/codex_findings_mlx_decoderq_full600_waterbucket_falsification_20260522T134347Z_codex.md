# Codex Findings: MLX Decoder-Q Full600 Waterbucket Falsification

Date: 2026-05-22T13:43:47Z

## Scope

Follow-up to the closed MLX decoder-q parent contract. The goal was to use the
full-600 same-axis MLX/auth response surface to choose a frontier-moving
decoder-q waterbucket candidate, then gate exact auth-eval spend with official
inflate controls and macOS-CPU advisory component response.

## Artifacts

- Pair-0 fixed same-axis dataset:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1113Z/candidate_same_axis_window_response_dataset_pair0_fixed.json`
- Pair-0 fixed family delta:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1113Z/mlx_decoderq_minus_fec6_full600_family_delta_pair0_fixed.json`
- Pair-0 fixed response surface:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1113Z/mlx_decoderq_full600_response_surface_plan_pair0_fixed.json`
- Deduped waterbucket plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1113Z/full600_surface_guided_waterbucket_plan_pair0_fixed_dedup.json`
- Official inflate controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1113Z/full600_surface_guided_inflate_controls.json`
- macOS-CPU advisory batch:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1113Z/full600_surface_guided_advisory_batch.json`

## Findings

1. Pair-0 dataset bug fixed at the producer level.

   The earlier full-600 dataset encoded `source_start_pair: null` for
   `source_pair_window=[0,1]` because a zero `start_pair` fell through an `or`
   expression. The regenerated dataset has 1200 rows, 600/600 family coverage,
   and zero null `source_start_pair` rows. Pair 0 is preserved for both
   `mlx_decoder_q` and `mlx_fec6_auth_parent`.

2. Response surface remains full-600 and unchanged in aggregate.

   Full-600 family delta remains: 600 matched windows, 170 candidate-better,
   430 candidate-worse, min delta `-0.0020326847010743165`, max delta
   `0.00304554049762229`, mean delta `0.0003948113818396977`.

3. Waterbucket materialization now dedupes payloads.

   The original plan had 29 nominal rows, 11 fixed rows, but only 5 unique fixed
   archive payloads. The deduped plan has 11 candidate rows total, 5 fixed
   unique archives, 6 nonfixed rows, and 18 duplicate aliases. Bucket labels are
   now metadata aliases when they collapse to the same `archive_zip_sha256` /
   `mutated_decoder_sha256`.

4. Official inflate controls prove runtime consumption.

   The five unique fixed archives all returned `returncode=0`, all changed 600
   frames versus baseline raw output, and candidate raws were deleted after
   hashing/comparison.

5. Advisory component response falsifies this exact queue for spend.

   Baseline score for gating: `0.1920513168811056`.

   | candidate | bucket class | advisory score | delta vs baseline |
   | --- | --- | ---: | ---: |
   | `a326b4d2e8961b98` | negative_control | `0.19242633847162177` | `+0.0003750215905161669` |
   | `a2d3b72c44e4a1f7` | response_surface_guided-1 | `0.19247233847162176` | `+0.0004210215905161574` |
   | `0a6d176fe5cead2e` | bias_only-1 | `0.1926812523017305` | `+0.0006299354206248942` |
   | `b2d86d2c0af3a8ad` | response_surface_guided-8 | `0.19302381403731766` | `+0.0009724971562120599` |
   | `e97f4613b5c16a74` | bias_only-2 | `0.19335642535772257` | `+0.0013051084766169674` |

   Best candidate is still worse than baseline by `+0.0003750215905161669`.

## Code Hardening

- `tools/plan_decoder_q_signed_waterbucket.py` now emits the complete six-field
  false-authority contract and records duplicate aliases instead of treating
  payload-identical bucket rows as independent candidates.
- `tools/run_decoder_q_candidate_inflate_controls.py` and
  `tools/run_decoder_q_candidate_advisory_batch.py` now share the same six-field
  false-authority contract for future runs.
- Focused tests cover the waterbucket/control false-authority contract. The
  pair-0 preservation regression is already covered in
  `src/tac/tests/test_scorer_response_dataset.py`.

## Verdict

`NO_EXACT_AUTH_DISPATCH`.

The full-600 decoder-q waterbucket queue is operationally real but locally
advisory-negative. This is useful falsification signal, not a promotion path.
Next frontier-moving branch should use the closed MLX parent contract on a
different transform family rather than spending exact eval on these five
archives.

## Verification

```bash
ruff check tools/plan_decoder_q_signed_waterbucket.py \
  tools/run_decoder_q_candidate_inflate_controls.py \
  tools/run_decoder_q_candidate_advisory_batch.py \
  src/tac/tests/test_plan_decoder_q_signed_waterbucket.py \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/optimization/scorer_response_dataset.py
```

Result: `All checks passed!`

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_plan_decoder_q_signed_waterbucket.py -q
```

Result: `96 passed in 1.18s`.
