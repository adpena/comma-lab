# Lossy Int4 / Mixed Precision / Coarsening Adversarial Review - 2026-05-08

Scope: `tools/pr101_lossy_int4_roundtrip_test.py`,
`tools/pr101_lossy_int4_per_channel_scales.py`,
`tools/pr101_lossy_mixed_precision_int4_int8.py`,
`tools/pr101_lossy_coarsening_analytical.py`,
`experiments/lossy_coarsening_lightning_cuda_test.py`, cathedral autopilot
evidence handling, and the local lossy-coarsening build artifacts under
`experiments/results/lossy_coarsening_20260508T013829Z/` and
`experiments/results/lossy_coarsening_20260508T020152Z/`.

## Evidence Standard

All CPU/MPS/proxy byte and rel_err rows reviewed here are non-promotable,
non-rankable, and non-falsifying unless an exact CUDA auth eval exists for the
exact archive bytes through `archive.zip -> inflate.sh -> upstream/evaluate.py`.

Current local evidence grades:

- Naive lossy int4 roundtrip: `[CPU-prep precision proxy]`; not score evidence.
- QAT lossy int4: `[MPS-research-signal]`; not score evidence.
- Per-channel int4 scales: `[CPU-prep empirical reactivation]`; not score evidence.
- Mixed precision int4/int6/int8: `[CPU-prep empirical reactivation]`; not score evidence.
- Analytical lossy coarsening: `[MPS-research-signal]` / `[CPU-build]`; not score evidence.

## Findings

1. Mixed-precision byte accounting undercounted 6-bit payloads and per-tensor
   headers. The analytical estimator used `ceil(n * 6 / 8)` for int6 even
   though the packer pads to 4-code groups and emits 3 bytes per group, and it
   omitted the three uint32 per-tensor header fields. Fixed in
   `tools/pr101_lossy_mixed_precision_int4_int8.py`; the tool now asserts that
   estimated raw bytes match actual packed raw bytes.

2. Mixed-precision rel_err aggregation weighted per-tensor means by total
   elements even though the rel_err definition masks `abs(orig) <= 1e-8`.
   Fixed to weight by nontrivial element count and to emit the exact
   `rel_err_definition` in manifests and evidence rows.

3. Mixed precision was being treated as conditionally CUDA-worth-testing from
   rel_err alone. The current measured smoke is byte-dominated by PR101 brotli:
   `/tmp/pr101_mixed_guard_manifest.json` reports 187,494 B, proxy rel_err
   4.865413998382046%, and
   `MEASURED_CONFIG_DOMINATED_BY_PR101_BROTLI_BASELINE`. It is not worth exact
   CUDA spend in that measured configuration.

4. CPU proxy readiness leakage existed or was partially fixed in the lossy-int4
   scripts. The hardened contract is now: `score_claim=false`,
   `promotion_eligible=false`, `rank_or_kill_eligible=false`,
   `ready_for_exact_eval_dispatch=false`, `family_falsified=false`, and explicit
   blockers include missing exact CUDA auth eval plus CPU/proxy semantics.

5. The analytical coarsening writeup and evidence source language used
   over-broad "FAIL/cannot" style wording against neural codecs. It now scopes
   prior results to measured smoke configs and uses byte-proxy language:
   byte-lower than PR101 brotli, not score-better, not a family kill.

6. Local lossy-coarsening build manifests lacked fail-closed score/promotion
   blockers. `experiments/lossy_coarsening_lightning_cuda_test.py` now stamps
   CPU build manifests with proxy-only semantics and exact-CUDA blockers before
   any score/rank/promotion use.

## Remaining Blockers

- No harvested `contest_auth_eval.json` exists under
  `experiments/results/lightning_batch/lossy-coarsening-cuda-20260508T013829Z/`
  or `experiments/results/lightning_batch/lossy-coarsening-cuda-20260508T020152Z/`;
  only `source_manifest.json` is present locally.
- Any lossy-coarsening score claim still requires the Lightning harvest path to
  retrieve and parse a numeric exact CUDA auth-eval JSON whose archive bytes
  match the staged archive.
- Mixed precision is currently dominated by PR101 brotli and should not be
  dispatched unless a new measured config is both byte-lower than 178,144 B and
  still passes the proxy threshold, then produces a byte-closed runtime packet.

## Reactivation Criteria

- Mixed precision: new manifest with actual packed raw byte accounting,
  archive bytes below 178,144 B, rel_err below the stated proxy threshold,
  explicit proxy-only flags, and a byte-closed runtime packet ready for exact
  CUDA auth eval.
- Per-channel/int4 QAT: same proxy-only flags plus a runtime packet; local rel_err
  alone may only mark `cuda_eval_worth_testing`, never promotion/rank/kill.
- Analytical coarsening: harvest `contest_auth_eval.json` for the exact archive
  SHA and bytes, then append a separate `[contest-CUDA]` evidence row. CUDA may
  falsify the measured archive/config, but CPU/MPS rel_err cannot.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_pr101_lossy_proxy_guardrails.py src/tac/tests/test_lossy_coarsening_lightning_tools.py src/tac/tests/test_pr101_lossy_int4_qat_dispatch_contract.py src/tac/tests/test_cathedral_autopilot_proxy_guards.py`
- `.venv/bin/python -m py_compile tools/pr101_lossy_mixed_precision_int4_int8.py tools/pr101_lossy_coarsening_analytical.py tools/pr101_lossy_int4_roundtrip_test.py experiments/lossy_coarsening_lightning_cuda_test.py src/tac/tests/test_pr101_lossy_proxy_guardrails.py`
- `.venv/bin/python tools/pr101_lossy_mixed_precision_int4_int8.py --output-json /tmp/pr101_mixed_guard_manifest.json`
