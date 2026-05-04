# Public PR90 qrepro Intake - Worker C - 2026-05-04

Scope: public PR90 intake and contest-faithful reverse engineering. No remote
GPU dispatch, no lane claim, and no exact score claim were performed.

AGENTS contest-compliance rules were read before intake. The required evidence
standard remains exact CUDA auth eval through `archive.zip -> inflate.sh ->
upstream/evaluate.py`; this document only records public PR custody, static
archive/runtime analysis, and a bounded local CPU smoke.

## Artifacts

- Result directory:
  `experiments/results/public_pr90_intake_20260504_worker/`
- PR API snapshot:
  `experiments/results/public_pr90_intake_20260504_worker/pr90_pull.json`
- PR file API snapshot:
  `experiments/results/public_pr90_intake_20260504_worker/pr90_files.json`
- Release API snapshot:
  `experiments/results/public_pr90_intake_20260504_worker/qrepro_release.json`
- Downloaded archive:
  `experiments/results/public_pr90_intake_20260504_worker/archive.zip`
- Payload probe:
  `experiments/results/public_pr90_intake_20260504_worker/payload_probe.json`
- Bounded local smoke log:
  `experiments/results/public_pr90_intake_20260504_worker/smoke_cpu/inflate_stdout_stderr.log`
- Bounded local smoke summary:
  `experiments/results/public_pr90_intake_20260504_worker/smoke_cpu/summary.txt`

## Public PR Metadata

- PR URL: <https://github.com/commaai/comma_video_compression_challenge/pull/90>
- Title: `add qrepro submission(0.28)`
- State: `open`
- Head: `SajayR:qrepro-submission`
- Head SHA: `cce857392701e73861ad513d34906faba523f719`
- Base: `commaai:master`
- Base SHA: `e84851da32108fcadf243c54d091cc71dc150c0e`
- Created: `2026-05-04T04:13:55Z`
- Updated: `2026-05-04T04:16:45Z`
- Changed files: `27`
- Public checks: only `welcome`, completed success. No public eval status was
  present in the statuses API.
- Archive URL from PR body:
  <https://github.com/SajayR/comma_video_compression_challenge/releases/download/qrepro-archive-v1/archive.zip>
- Release tag: `qrepro-archive-v1`
- Release asset: `archive.zip`, `218080` bytes, uploaded
  `2026-05-04T04:10:56Z`

The PR body reports PoseNet `0.00041977`, SegNet `0.00068872`, archive size
`218080`, and final score `0.28`. These are external/unverified values here.
Recomputing the formula from those reported components gives `0.2788721801656914`,
but this is not local exact evidence.

## Archive Custody

- Downloaded archive bytes: `218080`
- Archive SHA-256:
  `608ea0355e60faad97b046c27644205d05120ac85ab3e8a99543a75a4ab2dd2d`
- ZIP member count: `1`
- ZIP member: `p`
- Member method: stored, no compression
- Member bytes: `217980`
- Member SHA-256:
  `b48ba0ea138e4f3b12c02e320528a53ce92ed6540d71b8554249ee7bdcad6d34`
- CRC32: `9da0f139`
- ZIP overhead: `100` bytes
- Local and central directory member names both parse as `p`; no duplicate
  members, absolute paths, parent traversal, comments, or extra fields were
  observed.

This archive is structurally strict enough for study: single stored member,
matching local/central file name, and no ZIP parser-divergence trick observed.

## Runtime And Payload Anatomy

PR90 adds `submissions/qrepro/` with `README.md`, `inflate.sh`, `inflate.py`,
decoder modules, encoder/reference scripts, and visualization assets. The
archive itself is not committed in the PR; replay requires pairing the PR code
with the GitHub release asset.

The runtime path is:

1. `inflate.sh <data_dir> <output_dir> <file_list>` chooses `.venv/bin/python`
   if present, otherwise `python3`.
2. `inflate.py` reads only the unzipped archive data dir, the file-list text,
   and submission-local Python modules for the compact `p` path.
3. Device selection is `cuda` if available, else `cpu`. This machine has no
   CUDA and has MPS, but PR90 does not use MPS.
4. For public `0.mkv`, it decodes 600 semantic masks and renders two RGB frames
   per mask, writing one `0.raw` at `1200 x 874 x 1164 x 3` bytes.

The downloaded `p` payload does not start with a self-describing magic. Because
`len(p) < 260000`, `inflate.py` takes the compact fixed-offset path:

| Slice | Offset | Bytes | Runtime interpretation |
| --- | ---: | ---: | --- |
| mask body | `0` | `152431` | prepended magic `STBM1BR\0`, semantic topband/road/residual decoder |
| model body | `152431` | `56385` | prepended magic `QFQ4\0`, grouped Q-row FP4 model payload |
| pose + QRGB body | `208816` | `9164` | compact pose/bias plus low-frequency residual controls |
| base pose/bias control | `208816` | `5058` | Brotli control stream, raw `8400` bytes |
| QRGB residual controls | `213874` | `4106` | Brotli sparse residual stream, raw `8614` bytes |

Decoded static facts from `payload_probe.json`:

- Masks decode to shape `[600, 384, 512]`, dtype `uint8`.
- Mask class counts:
  `0=27408427`, `1=690063`, `2=58413695`, `3=1459867`, `4=29992748`.
- Pose control decodes to `[600, 6]`.
- Bias/control tensor decodes to `[600, 50]`.
- QRGB residual has `4307` nonzero int8 values, range observed inside the
  combined bias tensor: `[-24, 24]`.
- Model decode produced `111` float tensors, `87836` total elements.

No network fetch, subprocess shell-out, absolute local path, source-video read,
or sidecar outside the archive/submission code was observed on the compact
inflate path. `torch.load()` exists in a legacy fallback branch but was not used
by this `QFQ4` compact payload. The main non-faithful/reproducibility caveats
are:

- the committed PR does not include `archive.zip`; the release asset must be
  paired by URL and SHA;
- the compact `p` layout relies on hard-coded byte offsets rather than a
  manifest inside the payload;
- the PR body says the compression script is not included, although reference
  encoder scripts are present, so full train/compress reproducibility is not
  established;
- exact score cannot be claimed from the PR title, README, or report text.

## Local Smoke

Environment check:

- `.venv/bin/python` has `torch`, `brotli`, `av`, `einops`, `tqdm`, and `numpy`.
- `torch.cuda.is_available() = false`.
- `torch.backends.mps.is_available() = true`, but PR90 ignores MPS and falls
  back to CPU.

Bounded command:

```text
/opt/homebrew/bin/timeout 90s env PYTHON_BIN=/Users/adpena/Projects/pact/.venv/bin/python \
  bash experiments/results/public_pr90_intake_20260504_worker/pr90_src/submissions/qrepro/inflate.sh \
  experiments/results/public_pr90_intake_20260504_worker/archive_extract \
  experiments/results/public_pr90_intake_20260504_worker/smoke_cpu/output \
  experiments/results/public_pr90_intake_20260504_worker/pr90_src/public_test_video_names.txt
```

Result:

- Exit code: `124` from the intentional 90 second timeout.
- The inflater decoded semantic masks and began rendering.
- It wrote `2026533312` bytes to `0.raw` before timeout.
- `1164 * 874 * 3 = 3052008` bytes/frame, so the partial raw equals exactly
  `664` evaluator-shaped frames.
- Expected full raw bytes: `3662409600`.
- The 2 GB partial raw was removed after measurement; the log and summary were
  preserved.

Classification: `empirical` partial local smoke. It proves the public archive
and code can get to evaluator-shaped raw output locally. It is not CUDA eval,
not full output custody, and not score evidence.

## PR85 Comparison

Current exact anchor from local frontier snapshot:

- PR85 adaptive masking joint frame model replay.
- Evidence grade: `A++` T4 exact CUDA auth eval.
- Exact artifact:
  `experiments/results/lightning_batch/exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z/contest_auth_eval.adjudicated.json`
- Score: `0.25806611029397786`
- Archive bytes: `236328`
- SegNet: `0.00057185`
- PoseNet: `0.0001894`
- Archive SHA-256:
  `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`

Against that anchor, PR90's reported external numbers save `18248` bytes
(`-0.012150594176573383` rate contribution), but give up about `+0.011687`
SegNet contribution and `+0.02126954937483038` PoseNet contribution. Net from
reported PR90 values is about `+0.020806` worse than PR85. This comparison uses
PR90 reported values only; PR85 is exact A++.

## Transfer Opportunities Ranked By Wall-Clock Value

1. QRGB-style low-frequency residual controls on PR85 sidechannels.
   Highest immediate value. PR90 spends only `4106` compressed bytes for `4307`
   sparse low-frequency RGB/control edits. PR85 already has small high-value
   correction surfaces (`post`, `bias`, `region`, `randmulti`) and pair-gradient
   planning has identified hard pairs but lacked explicit action/value deltas.
   A PR90-inspired QRGB planner can emit explicit pair-local low-frequency
   brightness/color actions without touching PR85 fixed-runtime or pair-action
   workers. Gate: local non-noop raw-output parity and then exact CUDA only
   after a lane claim.

2. Semantic geometry decomposition as a PR85 QMA9 mask prior.
   Medium-high value. PR90's semantic stream is `152431` bytes versus PR85's
   current QMA9 mask segment at `159011` bytes, but the representations are not
   drop-in compatible. The useful idea is not copying bytes; it is testing
   topband/road-boundary support removal plus residual coding on PR85 decoded
   QMA9 tokens. Existing PR85 QMA9 row-run/alternate screens were negative, so
   this should be a focused PR90-inspired geometry screen, not another generic
   RLE sweep.

3. QFQ4 grouped FP4 model packing audit against PR85 model bytes.
   Medium-low value. PR90's model slice is `56385` bytes and PR85's model
   segment is about `57074` bytes, so the maximum obvious byte target is only
   hundreds of bytes unless the packer also improves runtime structure. Worth a
   static self-compression profile, not a first dispatch candidate.

4. Full qrepro renderer/semantic-program transplant.
   Low immediate value. PR90 is an architecture signal, not a frontier
   replacement: reported PR90 score is worse than the PR85 exact anchor, and the
   renderer, semantic masks, pose controls, and QRGB controls appear co-trained.
   A wholesale transplant would be a new representation lane requiring full
   custody and exact CUDA eval, not a quick PR85 stack.

5. Direct PR90 score promotion.
   Do not do this. Public report/title/README values are external only. The
   local worker proved partial raw-output replayability but did not run full
   CUDA auth eval.

## Next Concrete Local Action

Build a PR90-inspired QRGB planner for PR85 as a read-only profile plus
candidate-spec artifact: input PR85 decoded hard-pair/error plan, output sparse
low-frequency frame-pair edits with charged-byte estimates, non-noop proof, and
explicit target stream/value deltas. Keep it disjoint from the active PR85
fixed-runtime and pair-action workers; do not dispatch without a Level-2 claim.
