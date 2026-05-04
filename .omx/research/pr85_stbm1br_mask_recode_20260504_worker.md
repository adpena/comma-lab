# PR85 STBM1BR Mask Recode Candidate - 2026-05-04

## Scope

Worker STBM implemented a local, contest-compliant PR85 mask-recode candidate
for policy `pr90_stbm1br_lossless_pr85_mask_recode`. The change replaces only
the PR85 QMA9 mask segment with `STBM1BR\0` plus the PR90 topband mask body.
No scorer load, exact CUDA eval, lane claim, remote job, or dispatch was
performed.

## Source Custody

- PR85 source archive: `experiments/results/public_pr85_intake_20260503_codex/archive.zip`
- PR85 bytes/SHA-256: `236328`, `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- PR85 member `x` bytes/SHA-256: `236228`, `53bc78effa78cc7850d08a9ddc5488665b93136e9843549d917c17df729a1c50`
- PR85 source mask QMA9 bytes/SHA-256: `159011`, `4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179`
- PR90 source archive: `experiments/results/public_pr90_intake_20260504_worker/archive.zip`
- PR90 bytes/SHA-256: `218080`, `608ea0355e60faad97b046c27644205d05120ac85ab3e8a99543a75a4ab2dd2d`
- PR90 member `p` bytes/SHA-256: `217980`, `b48ba0ea138e4f3b12c02e320528a53ce92ed6540d71b8554249ee7bdcad6d34`

## Candidate Artifact

- Candidate archive: `experiments/results/pr85_stbm1br_mask_recode_20260504_worker/pr90_stbm1br_lossless_pr85_mask_recode/archive.zip`
- Candidate archive bytes/SHA-256: `229756`, `c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6`
- Candidate member `x` bytes/SHA-256: `229656`, `c7586795bb29fb0ef611ad44715aec77e0e815370e19674d4c89ef2a54b417b5`
- Candidate mask segment bytes/SHA-256: `152439`, `1b1ec60b64e284aae11e838dc3d9996bce00125df5712a8ba9c3e8f739c9d313`
- Candidate PR90 mask body bytes/SHA-256: `152431`, `420f74e5a02b7d559954c2920e2617846e52ad9d75d46111a3e224cc7d2c14ee`
- Archive byte delta versus PR85 source: `-6572`

## Parity And Preflight

- Decoded mask equality: `true`
- Diff pixels: `0`
- Render-order shape: `[600, 384, 512]`
- PR85 render-order SHA-256: `0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45`
- Candidate render-order SHA-256: `0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45`
- Runtime support present: `true` for the local mask loader branch in `submissions/robust_current/inflate_renderer.py`
- Fail-closed status: `passed`
- Exact eval status: not run. Main must claim a lane before any exact CUDA eval dispatch.

## Artifact Index

- Manifest: `experiments/results/pr85_stbm1br_mask_recode_20260504_worker/pr90_stbm1br_lossless_pr85_mask_recode/manifest.json`
- Preflight: `experiments/results/pr85_stbm1br_mask_recode_20260504_worker/pr90_stbm1br_lossless_pr85_mask_recode/stbm1br_preflight.json`
- Candidate summary: `experiments/results/pr85_stbm1br_mask_recode_20260504_worker/candidate_summary.json`
- Charged mask segment copy: `experiments/results/pr85_stbm1br_mask_recode_20260504_worker/pr90_stbm1br_lossless_pr85_mask_recode/mask_segment.stbm1br`

Evidence grade: empirical/local byte-and-token parity. This is not score
evidence and cannot be promoted until exact CUDA auth eval runs after lane
claim.

## 2026-05-04T06:05Z Wrong Runtime Exact-Eval Failure

The first two T4 exact-eval attempts used the original PR85 replay runtime:

- `exact_eval_pr85_stbm1br_lossless_mask_recode_t4_20260504T0549Z`
- `exact_eval_pr85_stbm1br_lossless_mask_recode_t4_g4dn2x_20260504T0600Z`

Both failed before scoring with the same inflate error:

```text
brotli.error: brotli: decoder failed
```

Root cause:

- The archive was the intended STBM archive:
  `c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6`,
  `229756` bytes.
- The staged inflate runtime was
  `experiments/results/public_pr85_intake_20260503_codex/replay_submission/inflate.sh`.
- That runtime parses PR85 single-member `x` bundles but assumes any non-QMA
  mask segment is Brotli-compressed AV1. `STBM1BR\0` is a different charged
  mask wire format, so the runtime failed before `contest_auth_eval.json`.

Evidence grade: `invalid` / pre-score runtime-contract failure. This is not a
method result and cannot rank, promote, or retire the STBM mask recode.

Mitigation landed:

- Created dedicated runtime:
  `experiments/results/pr85_stbm1br_mask_recode_20260504_worker/replay_submission_stbm/inflate.py`
- The runtime preserves PR85 single-member `x` bundle parsing and adds a
  narrow `STBM1BR\0` mask route through `tac.stbm1br_mask_codec`.
- Local verification:
  - `py_compile` passed for the dedicated runtime and STBM codec.
  - Focused STBM tests passed: `3 passed`.
  - Direct bundle-mask decode produced shape `(600, 384, 512)` and mask head
    `5354424d31425200`.

Corrected dispatch:

- Lane:
  `pr85_stbm1br_lossless_mask_recode_stbm_runtime_t4`
- Job:
  `exact_eval_pr85_stbm1br_stbm_runtime_t4_20260504T0610Z`
- Manifest:
  `.omx/state/pr85_stbm1br_stbm_runtime_t4_20260504T0610Z_manifest.json`
- Runtime:
  `experiments/results/pr85_stbm1br_mask_recode_20260504_worker/replay_submission_stbm/inflate.sh`
- Status at queue time: submitted, pending.

## 2026-05-04T06:30Z Corrected T4 Exact Result - New Frontier

The corrected g4dn.2xlarge T4 hedge completed and was harvested through the
state-derived SSH path with adjudication.

- Job:
  `exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z`
- Artifact:
  `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/contest_auth_eval.adjudicated.json`
- Adjudication:
  `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/adjudication_provenance.json`
- Score: `0.25369011029397787`
- Score delta versus PR85 exact T4: `-0.004375999999999991`
- Archive bytes/SHA-256:
  `229756`,
  `c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6`
- SegNet: `0.00057185`
- PoseNet: `0.0001894`
- Samples: `600`
- Hardware: Tesla T4, `gpu_t4_match=true`
- Evidence grade: `A++ contest T4`
- Promotion eligible: `true`

Component distances are unchanged at the contest JSON precision relative to
the PR85 exact anchor; the improvement is the measured charged-rate delta from
the STBM1BR lossless mask recode.

## 2026-05-04 Local STBM Inflate-Speed Patch

Priority update: the corrected STBM exact path exposed inflate wall-clock risk.
No GPU dispatch was performed in this local pass.

Patch:

- `src/tac/stbm1br_mask_codec.py` now prefills top-band and road-boundary
  pixels, iterates only residual pixels, carries the previous three decoded
  frame lists forward instead of re-running `numpy.ndarray.tolist()` on prior
  frames, and specializes the observed PR90 sparse feature tuple
  `(DIAG_TLTL, PREV_RIGHT2, PREV_BOTTOM2, X_BIN5_SHIFT, PEEL_DIST42)`.
- `src/tac/tests/test_stbm1br_mask_codec.py` adds a focused regression test
  proving top/road-prefilled pixels do not consume arithmetic symbols and that
  the optimized frame path preserves the materialized frame.

Local timing on the same mask segment:

- Before patch:
  `decode_stbm1br_mask_segment(mask_segment.stbm1br)` = `70.91800000000512s`,
  render-order SHA-256
  `0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45`.
- After active-pixel skip / previous-frame-list reuse:
  `43.176838500017766s`, same SHA-256.
- After sparse-feature specialization:
  `29.136844750028104s`, same SHA-256.

The corrected T4 hedge artifact already present locally records pre-patch
`inflate_elapsed_seconds=253.9567817860002`; the timing above is local mask
decode evidence only, not a new exact score.

Runtime custody note:

- Completed T4 hedge runtime tree:
  `d195f4ecd0743cfd146efafee6729e96ee5428bfb28bbd0ca87cbad055494440`
  with `src/tac/stbm1br_mask_codec.py` SHA-256
  `d52d72f70e653905d393ba2f2b32ec2d2e1d425c3f72b484a0d95bc0dd344eeb`.
- Local speed-patched runtime tree for the same replay runtime:
  `a035b2145c70d3cb768e7ec5d5d65e388967439929d71efcd08abf075220ed4b`
  with `src/tac/stbm1br_mask_codec.py` SHA-256
  `9c596e2869a65d7ed8dbce2a29e5983f5b159e67ecc0ab4b93346a1457f69014`.

Local byte-screen negatives:

- Recompressing the existing QTBM blob did not beat the current Brotli body:
  best observed body bytes remained `152431`.
- ZIP compression of the single `x` member was worse than `ZIP_STORED`:
  deflate added `75` bytes; bzip2/lzma were larger.
- QTBM table-layout variants were worse after outer Brotli even when raw tables
  shrank; the best zlib-sparse-table probe was raw `-3749` bytes but Brotli
  body `+702` bytes.
- Refit QBD2 road-boundary arithmetic grids could shrink the raw road payload
  by up to `147` bytes, but the full STBM Brotli segment grew by at least
  `337` bytes in the screened variants.

Evidence grade: empirical/local runtime and byte-screen evidence. No new
archive bytes supersede
`experiments/results/pr85_stbm1br_mask_recode_20260504_worker/pr90_stbm1br_lossless_pr85_mask_recode/archive.zip`
(`229756` bytes,
`c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6`).

Duplicate handling:

- The primary T4 job
  `exact_eval_pr85_stbm1br_stbm_runtime_t4_20260504T0610Z` was stopped after
  the g4dn.2xlarge packet landed. The SDK stop call timed out, but a subsequent
  refresh showed the job regressed to `Pending` at cost `0.050033335`; the
  next refresh confirmed `Stopped` at cost `0.056366667`.
- The active claim ledger records this as
  `stopped_confirmed_duplicate_after_hedge_success`.

Harness note:

- Earlier state-derived harvest snapshots while jobs were still running wrote
  `ARTIFACT_INFRA_FAILURE` records because `contest_auth_eval.json` was not yet
  present. One such snapshot already showed `inflate returncode=0` and the
  evaluate command, so it was a partial running artifact, not a method failure.
  This is now recorded as a separate harvest-classification hardening target.
