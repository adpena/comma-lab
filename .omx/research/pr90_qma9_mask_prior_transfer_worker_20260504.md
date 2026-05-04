# PR90 QMA9 Mask-Prior Transfer Worker - 2026-05-04

Scope: local-only investigation of PR90 semantic topband/road-boundary
decomposition as a PR85 QMA9 mask-prior transfer opportunity. No remote
dispatch, no training, no scorer run, no lane claim, and no score claim were
performed.

## Artifacts

- Analyzer:
  `experiments/analyze_pr90_qma9_mask_prior_transfer.py`
- Analysis JSON:
  `experiments/results/pr90_qma9_mask_prior_transfer_20260504_worker/analysis.json`
- Ranked candidate policy JSON:
  `experiments/results/pr90_qma9_mask_prior_transfer_20260504_worker/ranked_candidate_policy.json`
- Markdown summary:
  `experiments/results/pr90_qma9_mask_prior_transfer_20260504_worker/summary.md`
- Focused test:
  `src/tac/tests/test_analyze_pr90_qma9_mask_prior_transfer.py`

## Inputs Parsed

- PR90 public intake archive:
  `experiments/results/public_pr90_intake_20260504_worker/archive.zip`
- PR90 compact `p` payload mask body:
  `152431` bytes, interpreted as `STBM1BR\0` + Brotli-compressed QTBM topband
  stream.
- PR85 public intake archive:
  `experiments/results/public_pr85_intake_20260503_codex/archive.zip`
- PR85 QMA9 mask segment:
  `159011` bytes,
  SHA-256 `4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179`
- PR85 decoded token source:
  `experiments/results/public_pr85_intake_20260503_codex/qma9_token_source/pr85_qma9_tokens_u8_storage_order.bin`

## Local Finding

PR90's decoded semantic topband mask tensor is exactly equal to PR85's decoded
QMA9 mask tensor after converting PR85 storage order `[600,512,384]` to render
order `[600,384,512]`.

- PR90 render-order mask SHA-256:
  `0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45`
- PR85 render-order mask SHA-256:
  `0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45`
- Diff pixels: `0`
- Class counts:
  `0=27408427`, `1=690063`, `2=58413695`, `3=1459867`, `4=29992748`

This means the PR90 whole renderer stack remains co-trained, but the mask
stream itself is not too co-trained for a lossless PR85 transfer: it is a
different charged representation of the same PR85 semantic mask tokens.

## PR90 Geometry Profile

Decoded from the actual PR90 QTBM support streams:

- Top support pixels: `51994112` (`0.4407595486111111` of mask pixels)
- Road support pixels: `29992748` (`0.25425167507595486` non-top fraction)
- Top/road overlap pixels: `0`
- Residual-coded pixels after top/road support: `35977940`
  (`0.30498877631293403` of mask pixels)
- Top support SHA-256:
  `ccef7bab5ccaea7ae82260f7fb48c67d09400ac791788ca93a2c0731bb1cb60a`
- Road support SHA-256:
  `68b0f927e8c50bfd9b4a1efe72cc83ef0808973fb9f45523da82af5aea80cf59`

## Ranked Candidate

`pr90_stbm1br_lossless_pr85_mask_recode`

- Target score term: archive bytes/rate only, contingent on decoded-mask and
  runtime output parity.
- No-op status: not a no-op. Charged mask bytes change while decoded semantic
  tokens remain identical.
- PR85 mask segment bytes: `159011`
- Candidate mask segment bytes: `152439`
  (`152431` PR90 mask body bytes + `8` self-describing `STBM1BR\0` magic bytes)
- Delta mask segment bytes: `-6572`
- Estimated archive bytes if only the mask segment changes: `229756`
- Rate score delta if components are unchanged:
  `-0.004376025039918911`

## Builder Requirements

An implementable next archive builder exists, but it is not fixed-runtime PR85.
The builder must:

1. Replace only the PR85 mask segment in the single-member `x` bundle.
2. Charge a self-describing `STBM1BR\0` mask segment; do not rely on PR90's
   compact fixed-offset archive layout inside PR85.
3. Port or reimplement the PR90 QTBM topband/road-boundary decoder in the PR85
   inflate runtime with a distinct magic path.
4. Prove decoded-token SHA parity against the PR85 QMA9 render-order tensor.
5. Prove PR85 inflate/runtime output parity against the baseline archive before
   any exact CUDA eval dispatch.
6. Validate deterministic ZIP structure, member ordering, no sidecars, and the
   updated PR85 bundle length header.
7. Claim a lane before any remote exact eval dispatch.

## Fixed-Runtime Blocker

The current PR85 runtime does not decode `STBM1BR\0`. A geometry-prior reencode
that preserves the fixed QMA9 runtime is blocked because QMA9 has no charged
topband/road-prior side channel. The PR90 prior only becomes archive-changing
through a new reviewed mask decoder path or an equivalent token-parity runtime
extension.

## Classification

- Evidence grade: empirical local exact token parity, no score evidence.
- Transfer decision: implementable local archive builder after runtime port.
- Fixed-runtime PR85 transfer: blocked.
- PR90 representation too co-trained for mask transfer: no, because decoded
  mask parity is exact.
