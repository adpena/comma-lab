# Public PR95/PR96 no-dispatch intake - 2026-05-04

Scope: newest top public leaderboard and open-PR intake with no remote GPU
dispatch. No lane claim was opened and no training/eval job was launched in this
pass. Existing partner edits and artifacts were treated as read-only.

## Live Source Snapshot

Checked live on 2026-05-04 from `/Users/adpena/Projects/pact`.

- Official public leaderboard: https://comma.ai/leaderboard
  - Published video-compression top remains PR85
    `adaptive_masking_joint_frame_model` at display score `0.26`.
  - PR95 and PR96 are not yet official leaderboard rows.
- Open PR ordering from GitHub API:
  `https://api.github.com/repos/commaai/comma_video_compression_challenge/pulls?state=open&sort=created&direction=desc&per_page=20`
  - `#96` open, created `2026-05-04T08:32:02Z`,
    `rem2_HNeRV submission (0.21)`.
  - `#95` open, created `2026-05-04T07:47:15Z`,
    `hnerv_muon submission (0.20)`.
  - No newer open PR than PR96 was present in the API response.
- PR95 public source: https://github.com/commaai/comma_video_compression_challenge/pull/95
- PR96 public source: https://github.com/commaai/comma_video_compression_challenge/pull/96
- Nearby same-day but non-frontier open PRs:
  - PR94 reports MPS score `0.33` and requires GPU inflation.
  - PR93 reports CUDA score around `0.31786`.

## Current Exact/Public/Static Comparison

Lower score is better. Public PR body claims are external/static until exact
CUDA auth replay validates the exact archive bytes.

| Candidate | Status | Evidence | Score | Bytes | PoseNet | SegNet | Artifact |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| PR95 conservative repack | local exact best | T4 CUDA, 600 samples | `0.23091954465634829` | `178321` | `0.00017185` | `0.00070728` | `experiments/results/lightning_batch/exact_eval_pr95_hnerv_muon_repacked_t4_fix2_20260504T0848Z/contest_auth_eval.adjudicated.json` |
| PR95 public archive | exact replay | T4 CUDA, 600 samples | `0.23098329465634826` | `178417` | `0.00017185` | `0.00070728` | `experiments/results/lightning_batch/exact_eval_public_pr95_hnerv_muon_t4_fix2_20260504T0848Z/contest_auth_eval.adjudicated.json` |
| PR85+STBM1BR+PR92/RMB1 | prior exact anchor | T4 CUDA, 600 samples | `0.2535063602939779` | `229480` | `0.00018940` | `0.00057185` | `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_pr92_rmb1_t4_20260504T082220Z/contest_auth_eval.adjudicated.json` |
| PR95 public body | public static | PR body CPU/ubuntu claim | `0.1987048012202245` | `178417` | `0.00003494` | `0.00061212` | `experiments/results/public_pr95_intake_20260504_codex/archive.zip` |
| PR96 public body | public static | PR body CPU claim | `0.20567121179282477` | `186631` | `0.00003675` | `0.00062231` | `experiments/results/leaderboard_intel_20260504_codex/pr96_archive.zip` |
| PR85 official leaderboard | official published | public leaderboard display | `0.26` display | `236328` in PR body/static intake | `0.00018940` | `0.00057185` | `experiments/results/leaderboard_intel_20260504_codex/leaderboard_intel_summary.json` |
| PR91 HPM1 | public static only here | PR body CUDA claim, local HPM1 prefix blocked | `0.24879480490416128` | `222404` | `0.00018940` | `0.00057185` | `experiments/results/public_pr91_intake_20260504_worker/archive.zip` |
| PR92 RMB1 | public static only here | PR body CUDA claim/static intake | `0.2587078229986317` | `236516` | `0.00018963` | `0.00057675` | `experiments/results/public_pr92_intake_20260504_codex/archive.zip` |

Important exact-eval finding: PR95's public CPU/body score did not reproduce on
our exact T4 CUDA path. The exact public archive score is `0.23098329465634826`,
not `0.1987048012202245`. The 96-byte conservative repack is the current exact
best in this checkout because it preserved PR95 components and reduced the rate
term only.

## Archive And Runtime Anatomy

### PR95 `hnerv_muon`

- Public archive:
  `experiments/results/public_pr95_intake_20260504_codex/archive.zip`
  - bytes `178417`
  - SHA-256 `e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a`
  - one stored member `0.bin`, member bytes `178309`, member SHA-256
    `4b8013fb1168e21b12fc7ee6c395e032660d88daded48a0b845085e7d5eb11c4`
- Logical payload from
  `experiments/results/public_pr95_intake_20260504_codex/pr95_blob_anatomy.json`:
  - meta brotli `80` bytes; `latent_dim=28`, `base_channels=36`,
    `eval_size=[384,512]`, `n_pairs=600`
  - decoder brotli `162349` bytes
  - latent brotli `15868` bytes
  - no trailing bytes
- Runtime:
  `experiments/results/public_pr95_intake_20260504_codex/pr95_src/submissions/hnerv_muon/inflate.sh`
  calls a Python module that selects CUDA if available, parses `0.bin`, decodes
  600 two-frame pairs, bicubic-upsamples to `874x1164`, rounds to uint8, and
  writes raw RGB bytes.
- Runtime custody in exact replay:
  `runtime_tree_sha256=a3f8ab2cfbbdfab53a6d437b0c39f525e0adde2d8bd971765de96aeda4da3dc7`.

### PR95 Byte Opportunities

- Conservative repack profile:
  `experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex/profile_pr95_hnerv_muon_packing.json`
  - `archive.pr95_repacked.zip`: `178321` bytes, SHA-256
    `2b9b471358f5bba97ea809dcca544ecca26b504fde770a9002046830f469368b`
  - exact T4 score `0.23091954465634829`
  - pure rate win vs public PR95 exact: `96` bytes, `0.00006375` score
- Conservative stream wins were small but proven:
  - decoder `162349 -> 162232` compressed bytes
  - latents `15868 -> 15857` compressed bytes
  - meta `80 -> 68` compressed bytes
- Aggressive stem/latent permutation candidate:
  `archive.pr95_repacked_stemperm.zip`, `178277` bytes, SHA-256
  `e40c3f2fb3587b12eccb8707e0a1b7831fde149318f3eb212500c674ccbfbf28`.
  It saves `140` bytes but changes floating-point accumulation order, so it is
  not score-preserving evidence without exact CUDA replay.
- PR95 source explicitly notes a dropped hybrid categorical coder for large
  tensors that was about `217` bytes smaller. PR96 independently demonstrates
  this style on four largest tensors, so restoring/porting that path is the
  cleanest next PR95 byte-only target after local tensor equality checks.
- Largest PR95 decoder byte targets from the packing profile:
  `blocks.1.weight`, `blocks.0.weight`, `stem.weight`, `blocks.2.weight`,
  `blocks.3.weight`, `blocks.4.weight`, `blocks.5.weight`. These dominate the
  `162 KB` decoder payload and should be the first per-tensor entropy-coding or
  mixed-precision sensitivity targets.
- Latents are only `15.9 KB`; dimensional recoding can help, but the first-order
  delta/lo-hi brotli path already captured a documented `~240` byte win versus
  plain brotli.

### PR96 `rem2_HNeRV`

- Public archive:
  `experiments/results/leaderboard_intel_20260504_codex/pr96_archive.zip`
  - bytes `186631`
  - SHA-256 `2ecbd2118bebdb5566f719ed538a89c4608ccab19c9edc7ae7a6de778bd42b46`
- Members:
  - `decoder.bin`: ZIP deflate, file bytes `169242`, compressed bytes `169272`,
    SHA-256 `940d2328ad5384c99b51872cc5f87b6153befa20c10c1c2564b08ba469c97868`
  - `latents.bin`: ZIP deflate, file bytes `16920`, compressed bytes `16133`,
    SHA-256 `be4ed381db136baf114701fe6a9bba4f604dac18e79244c9f657383578a6ff71`
  - `p`: stored, `930` zero bytes, SHA-256
    `bf04a4e2dd69ca32e3b1bd1a3c64481d7f6930096b552d49d175eec8768d1c43`
- Runtime:
  `experiments/results/leaderboard_intel_20260504_codex/pr96_runtime/inflate.sh`
  calls `inflate.py`, which reads only `decoder.bin` and `latents.bin`, selects
  CUDA if available, decodes 600 `latent_dim=28` rows, bicubic-upsamples, and
  writes raw RGB.
- Internal decoder split from static parse:
  - header: `br_len=45658`, `hist_len=805`, `meta_len=102`,
    `lengths_len=20`, `comp_id=2` brotli, arithmetic-coded bytes `122640`
  - 24 brotli state records, `52270` quantized elements
  - 4 range-coded state records, `176688` quantized elements:
    `blocks.1.weight`, `blocks.0.weight`, `stem.weight`, `blocks.2.weight`
  - histogram raw length `2048` bytes
  - latents: `n_rows=600`, `n_dim=28`, `16800` q bytes plus fp16 mins/scales
- Immediate PR96 pure-rate opportunities:
  - `p` is not read by visible runtime and costs about `1008` ZIP bytes
    including member/header overhead. Removing it is likely output-preserving
    but must be verified by local raw-byte parity and exact T4 eval before use.
  - `decoder.bin` is deflated despite growing by `30` bytes; store it instead
    if raw-byte parity confirms archive-only change.
  - Combined zip-only opportunity is roughly `1038` bytes, worth about
    `0.000691` score if output is unchanged.

## Contest-Compliance Risks

- PR95 and PR96 are open PR body claims, not official leaderboard entries as of
  this pass. The official public page still shows PR85 as the top published
  row.
- CPU/MPS/body reports are not promotion evidence. PR95 is the concrete warning:
  public static score `0.1987048012202245` became exact T4 score
  `0.23098329465634826` on the same archive bytes.
- PR96 has no exact local/T4 replay artifact in this checkout yet. It imports
  `constriction`, has a conditional `zstandard` path for `comp_id=1`, and relies
  on torch bicubic behavior. Preflight must pin dependencies and runtime tree
  before T4 dispatch.
- PR96's `p` member is charged, not hidden side information, but it appears to
  be unused by visible runtime. Treat it as a no-op/compatibility payload until
  a byte-parity check proves removal is safe.
- PR95 compress/reproducibility code is useful but not equivalent to archive
  custody. The exact scored object is the submitted archive plus runtime tree,
  not the claimed ability to regenerate it from random initialization.
- HNeRV runtimes choose CUDA when available. That is contest-legal only if exact
  eval custody records device, torch version, runtime tree hash, sample count,
  archive SHA, and raw-size success. Cross-device component drift must be
  expected.
- PR91 HPM1 remains design signal only in this checkout because local prefix
  decoding is still blocked; do not dispatch PR91-derived archives without the
  replay contract fixed.

## Exact Next Local Or T4 Recommendations

No dispatch was launched here. Before any T4 dispatch, claim the lane with
`tools/claim_lane_dispatch.py claim ...` and close it with a terminal row.

1. Local PR96 replay preflight:
   - py-compile `pr96_runtime/inflate.py` and `inflate.sh`
   - verify ZIP members, dependency availability, and that `p` is unused by a
     static read-set check
   - run a small local inflate/raw-size smoke if wall-clock is acceptable
2. First T4 exact eval candidate:
   - archive `experiments/results/leaderboard_intel_20260504_codex/pr96_archive.zip`
   - inflate `experiments/results/leaderboard_intel_20260504_codex/pr96_runtime/inflate.sh`
   - lane id suggestion `public_pr96_rem2_hnerv_t4_replay_20260504`
   - purpose: determine whether PR96's CPU `0.20567121179282477` claim survives
     exact CUDA or drifts like PR95.
3. Local PR96 zip-only repack:
   - remove unused-looking `p`
   - store `decoder.bin` instead of deflating it
   - require raw-output byte parity against public PR96 before any exact eval
   - if parity passes, T4 eval after public PR96 replay because the expected
     gain is rate-only and small.
4. Local PR95 entropy-code lowering:
   - restore/port the documented PR95 hybrid categorical coder or PR96-style
     range coding for the four largest PR95 tensors
   - require exact dequantized tensor equality before archive build
   - expected first target: about `217` bytes, rate-only score around
     `0.000144` if components remain fixed.
5. PR95 aggressive `stemperm`:
   - dispatch only after PR96 replay or if T4 capacity is idle
   - reason: it saves only `44` more bytes than the proven conservative repack
     and is already known not to preserve rounded local output exactly.
6. Training/research follow-up:
   - use exact PR95 T4 component trace, not the PR body CPU trace, as the base
     for any hard-pair weighting, residual atoms, or HNeRV retraining.
   - keep PR85-family streams as diagnostic/correction-atom references, not as
     a second full representation; their rate is now dominated by HNeRV-family
     exact replay.

## Verification In This Pass

- Live GitHub API open-PR ordering checked with `curl` and `jq`; PR96 was the
  newest open PR and no PR97/newer PR existed in that response.
- Static PR96 archive parse completed from existing local artifact; no eval or
  remote dispatch was launched.
- SHA-256 checks run for PR96, PR95 public, PR95 conservative repack, PR95
  stemperm, and PR85+STBM/RMB1 exact archive artifacts.
- Focused verification passed:
  `git diff --check -- .omx/research/public_pr95_pr96_no_dispatch_intake_20260504_codex.md`;
  `.venv/bin/python -m py_compile` on the referenced PR95/PR96 Python inflate
  surfaces; and `bash -n` on the referenced PR95/PR96 `inflate.sh` files.
