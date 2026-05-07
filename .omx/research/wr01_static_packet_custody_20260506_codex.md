# WR01 Static Packet Custody - 2026-05-06

## Summary

WR01 `hnerv_wavelet_apply_transform_pr106x_1_2` remains a byte-custody
exact-eval candidate for operator review. This is not a score claim and not a
dispatch approval. The refreshed packet now keeps WR01 submit-blocked behind a
larger HNeRV rate-only byte-custody candidate until that priority decision is
resolved.

The candidate archive remains byte-identical to the prior WR01 apply-transform
candidate:

- archive path:
  `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/release_surface/archive.zip`
- archive bytes: `186222`
- archive sha256:
  `d2208ffa41297c40ce5f0bdbbe4767a9831301e382522afd2f6acf455a6b1628`
- source archive bytes: `186231`
- source archive sha256:
  `d25bca80057e8b533197895b4c56370678feb4e05fea0312c405bd12f29bec8e`
- byte delta: `-9`
- byte-only expected score delta: `-0.0000059927305780995425`

## Custody Artifacts

- packet:
  `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/wr01_exact_eval_packet.json`
  sha256 `bfee5bb53bc95c3e312f47b2f1b1dec2f1169066675ef2db8dd4b9c84f57af8f`
- apply manifest:
  `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/manifest.json`
  sha256 `f106d31d1ffc5f5e991e3e96faef6f849b4020521538c4d819610ab548e48f91`
- runtime decode validation:
  `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/hnerv_wavelet_runtime_decode_validation.json`
  sha256 `2d73c1001e62d44d8524633da5b54ff4ac468a12344b43fc088fca3da2096607`
- compress-time runtime/decode review:
  `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/hnerv_wavelet_compress_time_runtime_decode_review.json`
  sha256 `359e350e9ad0075eb9561f489e71d08bebabbc99ea5224afab1a8c46cda73a50`
- release archive manifest:
  `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/release_surface/archive_manifest.json`
  sha256 `4c93de758ab962e5e9e584bcef85791a74cd8c878aee6e9b1fbcbcbb5cc94453`

## Gate State

- `static_packet_ready=true`
- `candidate_static_preflight_ready=true`
- `byte_custody_exact_eval_candidate_ready=true`
- `runtime_decode_gate_ready=true`
- `static_blockers=[]`
- `runtime_decode_gate_blockers=[]`
- `operator_approved_exact_cuda=true`
- `ready_for_submit=false`
- remaining blockers:
  - `missing_lightning_environment`
  - `missing_active_lane_dispatch_claim`
  - `adversarial_priority_review_prioritizes_rate_only_candidate`

## Adversarial Priority Review

- `priority_decision=defer_wr01_behind_hnerv_rate_only_reference`
- WR01 byte delta: `-9`
- referenced rate-only packet:
  `experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/hnerv_lowlevel_exact_eval_packet.json`
- referenced rate-only byte delta: `-151`
- fastest safe score-lowering path:
  `dispatch_hnerv_rate_only_q10_before_wr01_if_lane_claim_and_env_clear`

Reasoning: WR01 changes a scorer-visible wavelet residual sidecar and has only
a 9-byte rate win before exact CUDA. The q10 HNeRV low-level repack is
rate-only, static-ready, and records a larger 151-byte byte win. Neither packet
may claim score without exact CUDA, but the next scarce exact-eval slot should
prefer the lower-risk, larger byte win unless explicitly superseded.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_wr01_exact_eval_packet.py src/tac/tests/test_cross_paradigm_atoms.py -q`
  - `18 passed`
- `.venv/bin/ruff check tools/build_wr01_exact_eval_packet.py src/tac/tests/test_wr01_exact_eval_packet.py src/tac/optimization/cross_paradigm_atoms.py src/tac/tests/test_cross_paradigm_atoms.py`
  - passed
- `.venv/bin/python tools/all_lanes_preflight.py`
  - all 23 checks passed after this adversarial-priority refresh

## Next Action

Do not dispatch from this ledger alone. The next exact-CUDA action requires, in
order:

1. restore/verify Lightning environment,
2. resolve the adversarial priority blocker or exact-evaluate/supersede the
   referenced HNeRV rate-only candidate,
3. claim lane `wr01_apply_pr106x_half`,
4. refresh packet with explicit operator exact-CUDA approval,
5. assert `ready_for_submit=true`,
6. then submit exact CUDA eval.
