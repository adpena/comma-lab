# WR01 Static Packet Custody - 2026-05-06

## Summary

WR01 `hnerv_wavelet_apply_transform_pr106x_1_2` is now static-packet-ready for
operator review. This is not a score claim and not a dispatch approval.

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
  sha256 `cc1999364d2cd7f53d6608063e91dd87d79993476132a0cbf8822a97874698f8`
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
  sha256 `4553efa43b1e5ec81f18c9a0b6688ab0b7b7677f49255559a7378cedf37e08fd`

## Gate State

- `static_packet_ready=true`
- `candidate_static_preflight_ready=true`
- `runtime_decode_gate_ready=true`
- `static_blockers=[]`
- `runtime_decode_gate_blockers=[]`
- `ready_for_submit=false`
- remaining blockers:
  - `missing_lightning_environment`
  - `missing_active_lane_dispatch_claim`
  - `missing_operator_exact_cuda_approval`

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_wr01_exact_eval_packet.py src/tac/tests/test_hnerv_wavelet_compress_time_harness.py src/tac/tests/test_hnerv_wavelet_apply_transform.py -q`
  - `16 passed`
- `.venv/bin/ruff check src/tac/tests/test_wr01_exact_eval_packet.py tools/build_wr01_exact_eval_packet.py tools/build_hnerv_wavelet_compress_time_harness.py`
  - passed
- `.venv/bin/python tools/all_lanes_preflight.py`
  - all 23 checks passed

## Next Action

Do not dispatch from this ledger alone. The next exact-CUDA action requires, in
order:

1. restore/verify Lightning environment,
2. claim lane `wr01_apply_pr106x_half`,
3. refresh packet with explicit operator exact-CUDA approval,
4. assert `ready_for_submit=true`,
5. then submit exact CUDA eval.

