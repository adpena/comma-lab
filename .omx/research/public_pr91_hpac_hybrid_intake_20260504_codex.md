# Public PR91 HPAC Hybrid Intake

Status: active exact replay queued; no local score claim yet.

Source:

- PR: https://github.com/commaai/comma_video_compression_challenge/pull/91
- title: `Hpac coder hybrid`
- author: `ottokunkel`
- head sha: `77f958d24e55980d95e01e3e9767b5a94320ed43`
- PR-reported exact score: `0.24879480490416128`
- PR-reported components:
  - PoseNet: `0.00018940`
  - SegNet: `0.00057185`
  - archive bytes: `222404`
- PR comment: borrows PR86 HPAC mask compressor and compresses the same masks
  from PR85, with fallback to the old PR when the compressor fails.

Local custody:

- archive:
  `experiments/results/public_pr91_intake_20260504_codex/archive.zip`
- archive bytes: `222404`
- archive sha256:
  `4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f`
- replay runtime:
  `experiments/results/public_pr91_intake_20260504_codex/replay_submission/hpac_coder_hybrid/`
- runtime file hashes:
  - `inflate.sh`:
    `2c37f19e210f97c8926b70594e4d57fd8b0256dadace0cd55c28bbe6995ff027`
  - `inflate.py`:
    `9665d3832abd8f720617cad4e976536ef9ff4ce9a117623b9a00229c518a22ba`
  - `pr86_hpac.py`:
    `f86f3067386928478d983817c9f9ee095ce6eb02aee8c0fbb7987cd0af1f9b01`
  - `range_mask_codec.cpp`:
    `23a6d9b622b231f61c73cd57eae3af96ca6133d07fab85e384a72e257317fb25`

Local static checks:

- `bash -n inflate.sh`: passed.
- `py_compile inflate.py pr86_hpac.py`: passed.

Archive anatomy:

- ZIP members: one stored member `x`.
- ZIP member uncompressed/compressed bytes: `222304` / `222304`.
- ZIP member sha256:
  `5c213c61cc4d29b62286063bfdcb97e812af6b06c0021aeaecc8bc46644e17bf`
- Header mode: PR91 v5 micro header with fixed `bias=223` and `region=273`
  lengths in runtime code.

Payload slices:

- header: `24`
- mask: `145087`
- model: `57074`
- pose: `1487`
- post: `1400`
- shift: `226`
- frac: `106`
- frac2: `149`
- frac3: `154`
- bias: `223` fixed by runtime
- region: `273` fixed by runtime
- randmulti: `16101`

Mask payload:

- magic: `HPM1`
- mask sha256:
  `a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc`
- HPAC token dimensions: `N=600`, `H=384`, `W=512`
- HPAC params: `P=32`, `delta=2`, `ch=64`, `use_spm=1`,
  `hpac_d_film=8`, `ppmd_order=4`
- token stream bytes: `116796`
- token stream sha256:
  `541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b`
- HPAC model blob bytes: `28243`
- HPAC model blob sha256:
  `de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd`

Exact replay queue:

- T4/equivalent job:
  `exact_eval_public_pr91_hpac_hybrid_t4_20260504T0504Z`
- T4 state:
  `.omx/state/public_pr91_hpac_hybrid_t4_20260504T0504Z_batch_jobs.json`
- T4 staging manifest:
  `.omx/state/public_pr91_hpac_hybrid_t4_20260504T0504Z_manifest.json`
- T4 dispatch claim:
  `public_pr91_hpac_hybrid_t4_replay`
- L40S diagnostic job:
  `exact_eval_public_pr91_hpac_hybrid_l40sdiag_20260504T0506Z`
- L40S state:
  `.omx/state/public_pr91_hpac_hybrid_l40sdiag_20260504T0506Z_batch_jobs.json`
- L40S dispatch claim:
  `public_pr91_hpac_hybrid_l40sdiag_replay`
- T4 hedge job:
  `exact_eval_public_pr91_hpac_hybrid_t4_g4dn2x_20260504T0509Z`
- T4 hedge state:
  `.omx/state/public_pr91_hpac_hybrid_t4_g4dn2x_20260504T0509Z_batch_jobs.json`
- T4 hedge dispatch claim:
  `public_pr91_hpac_hybrid_t4_g4dn2x_replay`

Evidence standard:

- The PR-reported score and local anatomy are external/forensic until our
  queued canonical CUDA evals produce `contest_auth_eval.adjudicated.json`.
- L40S can diagnose runtime/component behavior only.  The T4 replay is the
  promotion-grade confirmation path.

Immediate transfer hypothesis:

- PR91 appears to resolve the practical HPAC block by wrapping PR86-style HPAC
  tokens inside PR85-compatible compact payload grammar and retaining PR85's
  pose/SegNet basin.
- The biggest byte delta versus PR85 is the mask stream: PR85 QMA9 mask was
  about `159011` bytes; PR91 HPM1 mask is `145087` bytes, a `13924` byte
  saving before considering header/runtime side effects.
- The runtime also adds fixed-length micro-header assumptions for `bias` and
  `region`, preserving charged bytes by moving lengths into code.  That is
  contest-visible runtime code, not hidden side data.

Next decision:

- If T4 confirms PR91, promote PR91 as the current exact frontier and compare
  our QRGB atom outputs against PR91, not PR85.
- If T4 fails but L40S succeeds, classify as hardware/runtime portability bug
  and inspect PR91's CPU/GPU-sensitive branches.
- If both fail, preserve as external leaderboard evidence only and keep PR85 as
  the exact local frontier.

## 2026-05-04T05:19Z Exact Replay Verdict

The queued PR91 replays did not produce a contest score under our canonical
replay path.

T4 g4dn.2xlarge hedge:

- job:
  `exact_eval_public_pr91_hpac_hybrid_t4_g4dn2x_20260504T0509Z`
- local mirror:
  `experiments/results/lightning_batch/exact_eval_public_pr91_hpac_hybrid_t4_g4dn2x_20260504T0509Z`
- archive bytes: `222404`
- archive sha256:
  `4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f`
- terminal class: `inflate_returncode_failure`
- failure class: `inflate_failure_before_score`
- evidence grade: `invalid`
- score claim: false
- log failure:
  `AssertionError: Tried to decode from compressed data that is invalid for the employed entropy model.`

L40S diagnostic:

- job:
  `exact_eval_public_pr91_hpac_hybrid_l40sdiag_20260504T0506Z`
- local mirror:
  `experiments/results/lightning_batch/exact_eval_public_pr91_hpac_hybrid_l40sdiag_20260504T0506Z`
- terminal class: `inflate_returncode_failure`
- failure class: `inflate_failure_before_score`
- evidence grade: `invalid`
- score claim: false
- same HPM1/constriction assertion before any `contest_auth_eval.json`.

Primary T4:

- job:
  `exact_eval_public_pr91_hpac_hybrid_t4_20260504T0504Z`
- stopped as redundant after the two HPM1 decode failures.
- refreshed final status: `Stopped`, cost about `0.022694444`, no artifacts
  visible.

Decision:

- PR91's `0.24879480490416128` remains an external public-PR report only, not
  local exact evidence.
- Local exact frontier remains PR85:
  `0.25806611029397786`, bytes `236328`, sha256
  `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`.
- Do not dispatch PR91-derived archives until HPM1 decode/reencode parity is
  fixed or a contest-faithful explanation proves the replay mismatch.
