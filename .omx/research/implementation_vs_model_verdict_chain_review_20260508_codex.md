# Implementation-vs-Model Verdict Chain Review - 2026-05-08

Scope:
`tools/pr101_kalle_fold_8comp_hierarchical_codec.py`,
`tools/pr101_tiny_nn_200_param_faithful.py`,
`tools/pr101_compressai_balle_hyperprior.py`,
`tools/pr101_compressai_balle_hyperprior_full.py`,
`tools/pr101_lossy_int4_roundtrip_test.py`,
`tools/pr101_lossy_int4_per_channel_scales.py`,
`tools/pr101_lossy_int4_qat.py`,
`tools/pr101_lossy_mixed_precision_int4_int8.py`,
`tools/pr101_lossy_coarsening_analytical.py`,
`tools/check_evidence_implementation_matches_model_spec.py`,
`tools/cathedral_autopilot.py`, and
`reports/cathedral_autopilot_evidence.jsonl`.

## Review Verdict

No reviewed row has exact CUDA score evidence. CPU, MPS, byte-anchor,
roundtrip-rel_err, and local build artifacts are useful empirical proxy
signals only. They are not promotion, rank, kill, or paper score evidence.

Current scanner result after this review:

- `kalle_fold_mixture_canonical_shapes`: model-spec mismatch is now closed for
  the scanner by a later faithful NN-weight-shape reactivation row:
  `reports/raw/pr101_kalle_fold_nn_lit_20260508T024328Z/manifest.json`
  (`206,260` B, no decoder/runtime packet, no score). The old generic-shape
  rows remain preserved as superseded diagnosis. Effective grade: empirical
  byte proxy, score grade invalid. Verdict: measured NN-shape config loses to
  PR101 brotli; no family kill.
- `tiny_nn_pmf_predictor`: model-spec mismatch is now closed for the scanner
  by a later faithful 188-param row:
  `reports/raw/pr101_tiny_nn_200param_20260508T022927Z/manifest.json`
  (`206,370` B, no runtime packet, no score). The old rank-K evidence remains
  preserved as superseded diagnosis. Effective grade: empirical CPU byte
  proxy, score grade invalid. Verdict: measured 200-param config loses to
  PR101 brotli; larger LSTM/transformer/cross-tensor sibling configs also lose
  in proxy rows; no family kill.
- `compressai_balle_hyperprior`: the 1.1KB MLP row is capacity-mismatched.
  The full `ScaleHyperprior` artifact exists locally, but no matching evidence
  row is present in `reports/cathedral_autopilot_evidence.jsonl`, and the
  artifact uses a pseudo-image substrate over PR101 weight symbols. Effective
  grade: MPS research signal / empirical proxy, score grade invalid. Verdict:
  partial substrate-adapted negative, canonical family falsification deferred.
- `lossy_int4_quantization`: naive PTQ, QAT/LSQ-style, per-channel, mixed
  precision, GPTQ, and AWQ now have measured proxy rows. The scanner no longer
  reports lossy-int4 variant debt. Effective grade: CPU/MPS empirical proxy,
  score grade invalid. Verdict: measured configs negative or dominated;
  family falsification remains scoped to measured configs only.
- `lossy_coarsening_analytical`: code and local build artifacts support a
  byte-lower proxy candidate, but no harvested `contest_auth_eval.json` exists
  for the exact staged archive under the local Lightning result dirs. Effective
  grade: MPS/CPU empirical proxy and CPU build artifact, score grade invalid.
  Verdict: honest byte-proxy reopening candidate, exact-score falsification or
  promotion deferred.

## Bugs Fixed

1. The implementation-vs-model scanner credited negated variant prose such as
   "QAT/LSQ/GPTQ/AWQ NOT tested" as if those variants were covered. Fixed by
   adding positive variant detection, negation windows, and sibling technique
   aggregation for lossy-int4 rows.

2. The scanner and preflight prose said the bug class was "extincted" while
   live findings remain. Reworded to guarded/advisory language.

3. `tools/pr101_tiny_nn_200_param_faithful.py` overstated 1:1 fidelity. It now
   records that the implementation is capacity/model-overhead faithful but
   distribution-contract partial because it predicts Gaussian mean/log_scale,
   not a full per-tensor PMF.

4. `tools/cathedral_autopilot.py` now marks explicitly negated variant
   inventories as model-spec mismatches so catalog updates stay non-promotable.

5. The scanner now preserves stale mismatch rows while allowing a later,
   explicitly marked 1:1 model-spec re-test to supersede them for current guard
   debt. This closed the kalle and tiny-NN mismatch count without deleting
   historical negative evidence.

## Stale Or Incomplete Artifacts

- Current historical raw manifests for kalle 8-component, full CompressAI,
  old QAT, per-channel, and coarsening were generated before some proxy
  contract fields existed; they should not be used as proof of dispatch
  readiness. Current tool code emits fail-closed fields, but history is
  preserved rather than rewritten.
- `reports/cathedral_autopilot_evidence.jsonl` still contains stale overbroad
  language in the existing coarsening row ("all FAIL", "cannot"). Current
  `tools/pr101_lossy_coarsening_analytical.py` source has narrower language;
  do not cite the stale row as a neural-codec family kill.
- `experiments/results/lightning_batch/lossy-coarsening-cuda-20260508T013829Z/`
  and `...020152Z/` only expose `source_manifest.json` locally, not harvested
  exact auth-eval JSON.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_preflight_implementation_model_match.py src/tac/tests/test_pr101_a1_cpu_anchor_tools.py src/tac/tests/test_pr101_lossy_proxy_guardrails.py src/tac/tests/test_pr101_lossy_int4_qat_dispatch_contract.py src/tac/tests/test_lossy_coarsening_lightning_tools.py`
- `.venv/bin/python -m py_compile tools/check_evidence_implementation_matches_model_spec.py tools/cathedral_autopilot.py tools/pr101_lossy_mixed_precision_int4_int8.py tools/pr101_lossy_coarsening_analytical.py tools/pr101_lossy_int4_roundtrip_test.py tools/pr101_tiny_nn_200_param_faithful.py src/tac/tests/test_preflight_implementation_model_match.py src/tac/tests/test_pr101_a1_cpu_anchor_tools.py`
- `.venv/bin/python tools/check_evidence_implementation_matches_model_spec.py`
- 2026-05-08T03:12Z re-check:
  `.venv/bin/python tools/check_evidence_implementation_matches_model_spec.py --json`
  now reports only `compressai_balle_hyperprior` capacity/variant debt.
- 2026-05-08T03:12Z focused tests:
  `.venv/bin/python -m pytest -q src/tac/tests/test_preflight_implementation_model_match.py src/tac/tests/test_lossy_coarsening_lightning_tools.py`
  - result: 31 passed
- 2026-05-08T03:14Z Omega packet materialization:
  `.venv/bin/python tools/materialize_omega_opt_linear_stack_packet.py --source-plan reports/hstack_vstack_multipass_plan_20260507.json --output reports/omega_opt_linear_stack_packet_20260508.json --score-claim --promotion-eligible --rank-or-kill-eligible --ready-for-exact-eval-dispatch --promotion-allowed --dispatchable`
  - result: wrote `reports/omega_opt_linear_stack_packet_20260508.json`
  - strict checker result:
    `.venv/bin/python tools/check_omega_opt_linear_stack_packet.py --strict --format json reports/omega_opt_linear_stack_packet_20260508.json`
    reports zero findings and keeps all promotion flags false because exact A++
    anchor fields are absent.
- 2026-05-08T03:14Z combined focused tests:
  `.venv/bin/python -m pytest -q src/tac/tests/test_omega_opt_linear_stack_packet.py src/tac/tests/test_omega_opt_claims.py src/tac/tests/test_omega_opt_anchor_discipline_tool.py src/tac/tests/test_codec_stack_planner.py src/tac/tests/test_preflight_implementation_model_match.py src/tac/tests/test_lossy_coarsening_lightning_tools.py`
  - result: 55 passed
