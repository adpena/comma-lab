# PR85 STBM1BR Model Recode Fail-Closed Contract - 2026-05-04

## Decision

Grand Council lane B is fail-closed for the current PR85_STBM1BR QFQ4 model
recode screen. Exact tensor parity remains required unless a future
drift-tolerant runtime path first proves local renderer/output parity. The
current QFQ4 opportunity is formula-only and must not emit an archive candidate
or unlock dispatch.

## Evidence

- Source archive:
  `experiments/results/pr85_stbm1br_mask_recode_20260504_worker/pr90_stbm1br_lossless_pr85_mask_recode/archive.zip`
- Source archive bytes: `229756`
- Source archive SHA-256:
  `c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6`
- Model segment bytes: `57074`
- STBM1BR mask segment bytes: `152439`

Local profile artifacts:

- `experiments/results/pr85_stbm1br_model_recode_feasibility_20260504_codex/candidate_summary.json`
- `experiments/results/pr85_stbm1br_model_recode_feasibility_20260504_codex/dispatch_blocker.json`
- `experiments/results/pr85_stbm1br_model_recode_feasibility_20260504_codex/qfq4_serializer/dispatch_blocker.json`

QH0/QM0 result:

- Best row: `qh0_canonical_source_passthrough`
- Decoded tensor parity: `true`
- Model byte delta: `0`
- Decision: reject as byte-neutral.

QFQ4 result:

- Best row: `qfq4_pr85_shifted_int8_rows`
- Formula-only model byte delta: `-659`
- Formula-only projected archive bytes: `229097`
- Decoded tensor parity: `false`
- Mismatch: `frame1_head.block1.film_proj.weight`
- Changed elements: `4726 / 5376`
- Max abs diff: `6.103515625e-05`
- Runtime blockers:
  `public_pr85_replay_missing_QFQ4_model_loader`,
  `robust_current_missing_QFQ4_renderer_loader`,
  `robust_current_missing_pr85_single_x_unpacker`

## Contract

`dispatch=false` is now emitted in structured blocker JSON for both the
standalone QFQ4 screen and the combined PR85_STBM1BR profile.

Required before any future dispatch:

- Build a byte-positive model-recode archive.
- Prove decoded tensor parity, or add a reviewed QFQ4 runtime path and prove
  local source-vs-candidate renderer/output parity.
- Preserve all non-model PR85 segments byte-identically: mask, pose, post,
  shift, frac, frac2, frac3, bias, region, and randmulti.
- Record model member bytes and SHA-256 in any emitted manifest.
- Take a fresh lane claim before any remote exact eval.

No score claim, CUDA eval, lane claim, or remote dispatch was performed.

## Verification

- `.venv/bin/python -m py_compile experiments/analyze_or_build_pr85_qfq4_model_serializer_candidate.py experiments/profile_pr85_stbm1br_model_recode_feasibility.py src/tac/tests/test_pr85_qfq4_model_serializer_probe.py src/tac/tests/test_profile_pr85_stbm1br_model_recode_feasibility.py`
- `.venv/bin/python -m pytest src/tac/tests/test_pr85_qfq4_model_serializer_probe.py src/tac/tests/test_profile_pr85_stbm1br_model_recode_feasibility.py src/tac/tests/test_qh0_record_serializer.py src/tac/tests/test_quantizr_torch_fp4_codec.py -q`

Focused pytest result: `14 passed in 2.98s`.
