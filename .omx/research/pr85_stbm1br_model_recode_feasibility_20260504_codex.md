# PR85 STBM1BR Model Recode Feasibility - 2026-05-04

## Scope

Local-only PR85/JFG model self-compression feasibility on top of the
`PR85_STBM1BR` exact frontier archive. No mask, pose, post, qpost, or
randmulti edits were made. No lane claim, remote dispatch, CUDA eval, or score
claim was performed.

## Implemented Guard

- Added `experiments/profile_pr85_stbm1br_model_recode_feasibility.py`.
- Added `src/tac/tests/test_profile_pr85_stbm1br_model_recode_feasibility.py`.
- Wrote summary artifact:
  `experiments/results/pr85_stbm1br_model_recode_feasibility_20260504_codex/candidate_summary.json`.

The guard reuses the existing QH0/QM0 serializer and QFQ4-style serializer
screens, then audits any emitted archive to ensure only the PR85 `model`
segment changes. Exact eval readiness is false unless a candidate is
byte-positive, decoded-model-parity clean, runtime-compatible, and model-only.

## Source Custody

- Source archive:
  `experiments/results/pr85_stbm1br_mask_recode_20260504_worker/pr90_stbm1br_lossless_pr85_mask_recode/archive.zip`
- Archive bytes: `229756`
- Archive SHA-256:
  `c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6`
- Model segment bytes: `57074`
- Model segment SHA-256:
  `c9194c19d44d3c32c4e588bbe4f9986ed0c562779078f0f4d6b073f0799b5dbc`
- STBM1BR mask segment bytes: `152439`

## Local Screen Result

QH0/QM0 deterministic serializer:

- Best screened candidate: `qh0_canonical_source_passthrough`
- Model delta: `0` bytes
- Decoded tensor parity: `true`
- Blocker: `no_real_byte_win`

QFQ4-style serializer:

- Best screened candidate: `qfq4_pr85_shifted_int8_rows`
- Formula-only model/archive byte delta: `-659`
- Formula-only projected archive bytes if components were identical: `229097`
- Formula-only rate-score delta if components were identical:
  `-0.0004388010501075109`
- Decoded tensor parity: `false`
- Tensor mismatch: `frame1_head.block1.film_proj.weight`
- Changed elements: `4726 / 5376`
- Max abs diff: `6.103515625e-05`
- Runtime blocker: PR85/STBM replay runtime lacks a QFQ4 model loader.

No candidate archive was built. The model-only archive guard passed because
there were no emitted candidates and therefore no forbidden non-model segment
changes.

## Readiness

Exact eval is not justified after lane claim in the current state.

Required before any dispatch:

- Build a byte-positive model recode archive.
- Prove decoded tensor parity.
- Add/reuse a no-sidecar QFQ4 PR85/STBM runtime loader.
- Prove local runtime output parity.
- Pass the model-only guard showing masks, poses, qpost/post, and randmulti
  bytes are unchanged.

Evidence grade: empirical/local fail-closed byte and tensor-parity screen.
