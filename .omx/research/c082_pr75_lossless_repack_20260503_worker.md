# C082 PR75 Lossless Repack Byte Screen - 2026-05-03

Scope: local archive/packer compression screen for the C082 PR75 actions-only
P3 fixed-slice frontier. No remote dispatch was performed.

Source:

- archive: `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_only_p3_t4_20260503T0401Z/archive.zip`
- bytes: `276460`
- sha256: `851a61ddd88c0ddb4a05762f11e900f132fb771fd9dba743ac2026637ddf15d9`
- payload format: `public_pr75_qzs3_qp1_segactions_p3`

Candidate matrix:

- artifact: `experiments/results/c082_pr75_lossless_repack_20260503_worker/candidate_matrix.json`
- best candidate: `c082_p6_delta_varint_actions_stream_resweep`
- archive: `experiments/results/c082_pr75_lossless_repack_20260503_worker/c082_p6_delta_varint_actions_stream_resweep/archive.zip`
- bytes: `276394`
- sha256: `9b78333dd39c12c986ca7fc02a4bb35a6a61ef5a1cc0c9c9c840820eb840058a`
- delta vs C082: `-66` bytes
- formula-only rate delta vs C082: `-0.000043946690906063314`
- evidence grade: `empirical_lossless_byte_transform`
- score claim: `false`

Decoded-stream parity:

- `masks.mkv`: `223385` decoded bytes, sha256 `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb`
- `renderer.bin`: `59288` decoded bytes, sha256 `5657593aec0bf380f0bd614578cc4a76da8589b6ef8ce0331c28b6a4d6658efb`
- `optimized_poses.bin`: `7200` decoded bytes, sha256 `5236cf75cc95f6a9731b35f4e2fb1211aa99bfcc69e5964869edd19a7af3df9f`
- `seg_tile_actions.bin`: `268` decoded bytes, sha256 `bfd46b2b481a5064cc1f64b7b1288640c51b89ad6aeb5598408150f7945eac15`

Implementation notes:

- Added P6 delta-varint PR75 action decoding in `submissions/robust_current/unpack_renderer_payload.py`.
- Added `experiments/build_pr75_lossless_repack_candidates.py` to build deterministic stored-ZIP candidates with custody manifests.
- `c082_p3_stream_resweep` is the no-runtime-format-change fallback: `276452` bytes, `-8` bytes vs C082, same decoded streams.
- `c082_p6_delta_varint_actions` is the action-codec-only candidate: `276402` bytes, `-58` bytes vs C082, same decoded streams.

Verification:

- Focused tests: `.venv/bin/python -m pytest src/tac/tests/test_unpack_renderer_payload_fixedslice.py src/tac/tests/test_build_pr75_lossless_repack_candidates.py` -> `9 passed`.
- Compile check: `.venv/bin/python -m py_compile experiments/build_pr75_lossless_repack_candidates.py submissions/robust_current/unpack_renderer_payload.py src/tac/tests/test_build_pr75_lossless_repack_candidates.py`.
- Local payload-unpack smoke: `experiments/results/c082_pr75_lossless_repack_20260503_worker/smoke_unpack_best/renderer_payload_unpack_summary.json`.

Promotion boundary:

These candidates are byte/parity evidence only. They require exact CUDA auth eval
on the exact archive bytes and runtime tree before any score or frontier claim.
