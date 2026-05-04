# Public Leaderboard / PR Score Trajectory Intake - 2026-05-04 Worker

Fetch time: `2026-05-04T04:11:31Z`.

Scope: fresh read-only public intake for `commaai/comma_video_compression_challenge`,
using primary GitHub APIs/pages plus local cached artifacts. No GPU jobs were
dispatched. No code was edited.

Structured artifact:
`experiments/results/public_leaderboard_score_trajectory_20260504_worker/public_leaderboard_score_trajectory_20260504_worker.json`.

Raw GitHub/API/download custody:
`experiments/results/public_leaderboard_score_trajectory_20260504_worker/`.

## Primary Sources

- Repo API: <https://api.github.com/repos/commaai/comma_video_compression_challenge>
- All PR API snapshot:
  <https://api.github.com/repos/commaai/comma_video_compression_challenge/pulls?state=all&per_page=100&sort=created&direction=desc>
- Current README / leaderboard page:
  <https://github.com/commaai/comma_video_compression_challenge/blob/master/README.md>
- Current master commit:
  <https://github.com/commaai/comma_video_compression_challenge/commit/e84851da32108fcadf243c54d091cc71dc150c0e>
- PR85: <https://github.com/commaai/comma_video_compression_challenge/pull/85>
- PR86: <https://github.com/commaai/comma_video_compression_challenge/pull/86>
- PR87: <https://github.com/commaai/comma_video_compression_challenge/pull/87>
- PR88: <https://github.com/commaai/comma_video_compression_challenge/pull/88>
- PR89: <https://github.com/commaai/comma_video_compression_challenge/pull/89>

Downloaded README custody:

| item | bytes | SHA-256 | Git blob SHA |
|---|---:|---|---|
| `master_README.md` | 15,819 | `010021c244403ed3fc6bc5db5bf907e386d9ce37b074dce0a411a5423e97e176` | `a60f3aad82a2af58293bfb1cbcd796345d9bf456` |

## Current Public Frontier

The latest master commit is `e84851da32108fcadf243c54d091cc71dc150c0e`
(`ci: update leaderboard tables`, committed `2026-05-04T03:38:52Z`). The
README leaderboard now begins:

| rank | score | name | PR | PR state | merged? | source |
|---:|---:|---|---:|---|---|---|
| 1 | `0.26` | `adaptive_masking_joint_frame_model` | #85 | closed | no | README table, official GitHub eval comment |
| 2 | `0.27` | `jas0xf_adversarial_neural_representation` | #86 | closed | yes, `2026-05-04T03:36:55Z` | README table, official GitHub eval comment |
| 3 | `0.28` | `adaptive_range_mask_no_router` | #84 | closed | no | README table |
| 4 | `0.28` | `qzs3_range_mask` | #81 | closed | no | README table |
| 5 | `0.31` | `qpose14_r55_segactions_minp` | #79 | closed | yes | README table |

Interpretation:

- Contest-faithful public leaderboard floor: PR85, display score `0.26`,
  recomputed from stated rounded components as `0.25806622496743437`.
- Latest merged public code frontier: PR86, display score `0.27`, recomputed
  from stated rounded components as `0.2736358503762718`.
- Latest numbered PR observed: PR89, but it was closed unmerged and withdrawn.
- PR90-PR100 do not exist as PRs at the fetch time; GitHub PR API returned 404
  for each `pulls/{90..100}`.

Open PRs at fetch time:

| PR | title | head SHA | score if stated | status |
|---:|---|---|---|---|
| #70 | `Add mask_decoder submission` | `b08810f5c2df491459f2922fa9a7e7adb6d4fc78` | body displays `0.19` on a 57,329-byte archive | open but non-faithful by the author's own note: bytes moved into `inflate.py` |
| #72 | `WIP: neural compressor` | `5191c1ee8966c7b49fb2a11d2ba37c19b2dcbada` | none parsed | open WIP |

## PR85-PR100 Intake

| PR | title | author | state / merge | head SHA | score / bytes if stated | classification |
|---:|---|---|---|---|---|---|
| #85 | `Adaptive masking joint frame model` | `ottokunkel` | closed, unmerged, leaderboard-listed | `515abfbac1dc6ef42499d5b912e424319f982fb2` | official eval: PoseNet `0.00018940`, SegNet `0.00057185`, archive `236,328` bytes, display `0.26`, exact text `0.25806622496743437` | current public leaderboard frontier |
| #86 | `jas0xf_adversarial_neural_representation (0.27)` | `jas0xf` | merged at `2026-05-04T03:36:55Z`; merge commit `14bcede815306415a0005c3cd98804151bce4049` | `0eabe354f09b7490fd1cbb2b05a9102ab528d4d4` | official eval: PoseNet `0.00045701`, SegNet `0.00067815`, archive `207,579` bytes, display `0.27`, recomputed `0.2736358503762718` | latest merged code frontier |
| #87 | `Add 100_bytes submission` | `manthedan` | closed, unmerged | `fde5153dd8472734796a1abc68843c98387328c3` | body: PoseNet `0.00018542`, SegNet `0.00057103`, archive `100` bytes, display `0.10`, recomputed `0.10023000855928522` | invalid/external loophole; author says not official-spirit, maintainer says scoring-script gaming will be fixed |
| #88 | `qzs3 range mask gpu preflight` | `erichasinternet` | closed, unmerged | `b330dd75232f06ac541726fbb29de80074db8eb5` | body CPU report: PoseNet `0.00052047`, SegNet `0.00071882`, archive `215,960` bytes, display `0.29`, recomputed `0.2878245063016197` | wrong-place GPU preflight, closed by author |
| #89 | `henosis_final_bias (0.28)` | `henosis-us` | closed, unmerged, withdrawn | `d970a0eaf5065807f75fc2f386bf5c21fd982ae1` | body CUDA-local report: PoseNet `0.00023557`, SegNet `0.00068982`, archive `236,676` bytes, display `0.28`, exact approx `0.27511`, recomputed `0.27511038756764417` | withdrawn; author says proper public-master CUDA evaluator was significantly worse |
| #90-#100 | n/a | n/a | not found | n/a | n/a | no public PRs exist at fetch time |

Official eval comment sources:

- PR85 eval comment:
  <https://github.com/commaai/comma_video_compression_challenge/pull/85#issuecomment-4367893489>
- PR86 eval comment:
  <https://github.com/commaai/comma_video_compression_challenge/pull/86#issuecomment-4367994628>

Withdrawal / invalidity sources:

- PR87 maintainer comment:
  <https://github.com/commaai/comma_video_compression_challenge/pull/87#issuecomment-4367870339>
- PR88 author close comment:
  <https://github.com/commaai/comma_video_compression_challenge/pull/88#issuecomment-4367560230>
- PR89 author withdrawal comment:
  <https://github.com/commaai/comma_video_compression_challenge/pull/89#issuecomment-4367760073>

## Downloaded Artifact Custody

Full SHA/size manifest:
`experiments/results/public_leaderboard_score_trajectory_20260504_worker/download_sha256.txt`
and
`experiments/results/public_leaderboard_score_trajectory_20260504_worker/download_sizes.txt`.

Archive downloads:

| PR | archive URL | bytes | SHA-256 | ZIP members |
|---:|---|---:|---|---|
| #85 | <https://github.com/user-attachments/files/27326003/archive.zip> | 236,328 | `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e` | `x` 236,228 bytes, SHA-256 `53bc78effa78cc7850d08a9ddc5488665b93136e9843549d917c17df729a1c50` |
| #86 | <https://github.com/jas0xf/comma-anr-supplementary/raw/master/archive.zip> | 207,579 | `e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef` | `master.pt.gz` 31,144; `slave.pt.gz` 32,287; `hpac.pt.ppmd` 28,243; `tokens.bin` 113,900; `meta.pt` 1,499 |
| #87 | <https://github.com/user-attachments/files/27326636/archive.zip> | 100 | `360a2551cfdb609f954f2b4f40b39e5b212356e5886442511a33dcaf728a1cb6` | `a` 0 bytes, empty SHA-256 |
| #88 | <https://github.com/erichasinternet/comma_video_compression_challenge/releases/download/qzs3-range-mask-archive/archive.zip> | 215,960 | `cd01378a52688fe00ee1bfb898c67695aed89a7b3ded602b597eb7fc3031d7fc` | `p` 215,860 bytes, SHA-256 `c59524610474c89e5a41433f47d2bb881f878e694f853db1377272699f9eb3e9` |
| #89 | <https://github.com/henosis-us/comma_video_compression_challenge/releases/download/henosis-final-bias-v1/archive.zip> | 236,676 | `2f4d38f295792bebc3e722abaad7647ecdd647b8a5948285bbe0906a6883ee98` | `x` same as PR85 plus `fb` 300 bytes, SHA-256 `1cac91437fd3ca3523ed6eaca4fc7f57b072fd24684fa2da9f267c42b2d0a5dc` |

Key runtime/report/compress blobs downloaded:

| PR | blob | bytes | SHA-256 | Git blob SHA |
|---:|---|---:|---|---|
| #85 | `README.md` | 2,736 | `e3e8bbcaee44e94c60196e1cc354002ffdcfcf04414107713820990f578ff2f0` | `867e928a9e0b68b52aea49c966e03870cb3d28ab` |
| #85 | `inflate.py` | 85,145 | `ff534ee0ef7a51dd4ec7a24570d812b32ce966e8b98b2dfdbe347ce9fa4ff3e7` | `21b88d4265957c416c6de3f39204dc66245ca7a0` |
| #85 | `inflate.sh` | 228 | `2c37f19e210f97c8926b70594e4d57fd8b0256dadace0cd55c28bbe6995ff027` | `f1fed20f30d0e77b7d9c44d161333a046b328d24` |
| #85 | `range_mask_codec.cpp` | 53,975 | `94cd1a86111fb6d34b6e12d37c624bd5938df0fbc6c4c24c8d40c5a83fcb016b` | `5a8f7a11d63e187accdb5fa204542645b058686f` |
| #86 | `compress.py` | 5,215 | `fda16b6e3d32bad77bf371f198e2253678e9f91470ee191cf7a9eef6bea196eb` | `13f3967a6ece5f59b61cd6bc3ceeeab56fb84ba9` |
| #86 | `compress.sh` | 297 | `b562a4246d527880a4ae21ec43317eabd4ea8d96ae75ac229320595bd293d5b9` | `7d5d48526c484c03dc1c14ee721d02a10cc0b2f8` |
| #86 | `inflate.py` | 19,657 | `f86f3067386928478d983817c9f9ee095ce6eb02aee8c0fbb7987cd0af1f9b01` | `3887d2ba511a163a15813844c39c57a698e32974` |
| #86 | `inflate.sh` | 396 | `8c43acac3e0959161b612d9bacd66d7e70070f412aefbed74f7b5d65e60e94ce` | `062b3cda3dfcd38dc0b5c3c7b244ff87dbde6614` |
| #86 | `training/hpac.py` | 26,408 | `2d80be4ba7df9e45e9824f0acfa489f8678aadb3014d7938ca6691449c3d0a07` | `24e42a90b2844b4ae96674e227dac4cc558d398f` |
| #86 | `training/master.py` | 23,302 | `289969bbad32bcb791f6548d034aa2beee83319b184ed4cb5ef51f00d279118c` | `a849b8fc618f5aac700fbcdef24580340fd5a584` |
| #86 | `training/slave.py` | 52,351 | `04d26c6cddeb9d36d4e95d8733ad826acb28c5a88882865bd21b07d15d2d80c0` | `acdfca4bd7437f0b6cebe07795bac3b1ecac270a` |
| #87 | `inflate.py` | 767,055 | `7fb6266b448b2a1f51f7b809e7fdf6ba4d37ec7e5f30fc0affe0d2ff6d9f267f` | `4c51d74abefc5e8a90df8e325ee487b7de00ef84` |
| #87 | `inflate.sh` | 418 | `6c75c76c628c1aa299d5f246c6b6b42129a6d934e7e7fd37bc5495cf9b5bad22` | `217b0c17369c75789f59ac5ca2d9a19044934759d` |
| #88 | `inflate.py` | 28,275 | `429f6580ce409ccf01e2e46f72149181c0e40cc889a8ebd6d0bcefbddcebdaee` | `4da47c52e90dc39d784d34ed6052d9e946994c9c` |
| #88 | `inflate.sh` | 524 | `7824116e0d88c290e27b782d314de1d88611d5f99379521256af5369bb3ae74f` | `cc70109e5bf4e211b08676a5beb23b3f46fd39c4` |
| #89 | `PR_BODY.md` | 1,375 | `3bb231f62cfdbcd63bde4fb650947427361d65d5a8e233a4484305e3ca02e568` | `e3d5fd2a6e001f71496b19ead06e76c46c6c8ef6` |
| #89 | `inflate.py` | 86,351 | `c1116e9b00fae4e8ba0c21890f46fa467749623b8d887d89839519adc233995a` | `dbeb873052ab288b71d5abda3da50a855d4c5613` |
| #89 | `inflate.sh` | 228 | `2c37f19e210f97c8926b70594e4d57fd8b0256dadace0cd55c28bbe6995ff027` | `f1fed20f30d0e77b7d9c44d161333a046b328d24` |
| #89 | `report.txt` | 625 | `ae8acf9005c8b3a871cfab41f39f07c6f46ae571307e1b05ad6682e2dd1b3702` | `94047116e30a91dfc68be8ab883018db686a5dc6` |

## Local Cached Evidence Cross-Check

Local exact PR85 T4 replay:
`experiments/results/lightning_batch/exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z/contest_auth_eval.adjudicated.json`.

- Archive SHA-256: `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- Archive bytes: `236328`
- Device: `cuda`
- Samples: `600`
- PoseNet: `0.0001894`
- SegNet: `0.00057185`
- Recomputed score: `0.25806611029397786`
- Runtime tree SHA-256: `d008db50f8a8165a9c7a5954e8eeec443878dbeb7a892e1d455515b4586b1a73`

The local exact PR85 replay confirms the public eval components for the exact
downloaded archive bytes. The tiny difference between `0.25806611029397786`
and the PR-body exact text `0.25806622496743437` is from component rounding in
the public report.

Local PR86 replay attempts existed but are invalid score evidence:
`experiments/results/lightning_batch/exact_eval_public_pr86_hpac_t4_hedge_20260504T0152Z/`
and
`experiments/results/lightning_batch/exact_eval_public_pr86_hpac_t4_retry1_20260504T0213Z/`
contain partial artifacts and no `contest_auth_eval.json`. They are classified
as `runtime_or_harness_failure_before_score_json`; do not rank from them. PR86
score custody here comes from the official GitHub eval comment and merged
leaderboard update.

PR85 static local bundle profile:
`experiments/results/public_pr85_intake_20260503_codex/profile_pr85_bundle.json`.

Important PR85 internal segments:

| segment | bytes | decoded bytes | note |
|---|---:|---:|---|
| `mask` | 159,011 | n/a | `QMA9X` semantic mask stream |
| `model` | 57,074 | 61,590 | Brotli-compressed `QH0` renderer/model payload |
| `pose` | 1,487 | 1,806 | Brotli-compressed `P1D1` pose stream |
| `post` | 1,400 | 2,400 | correction stream |
| `shift` | 226 | 603 | `SD4` stream |
| `frac` | 106 | 179 | `FV1` stream |
| `frac2` | 149 | 603 | `FH2` stream |
| `frac3` | 154 | 603 | `FD3` stream |
| `bias` | 223 | 603 | `BD1` stream |
| `region` | 273 | 603 | `RH1` stream |
| `randmulti` | 16,101 | 27,105 | sparse random/multi-action stream |

## Score-Trajectory Reverse Engineering

The public trajectory has moved through four regimes:

1. Conventional video-codec tuning and ROI preprocessing produced scores in the
   `~2-4` range.
2. Quantizr / evaluator-latent submissions broke through to `~0.33`, shifting
   the game from perceptual compression to charged sufficient statistics for
   SegNet/PoseNet.
3. qpose14 / QZS3 / segaction variants clustered around `0.31-0.32`, showing
   that exact semantic masks plus tiny pose/action streams were the main
   frontier representation.
4. Range-coded exact semantic masks and compact neural renderers pushed the
   frontier to `0.28`, then PR85 reached `0.258` by accepting a larger
   236 KB archive in exchange for materially lower PoseNet and SegNet terms.

PR85 vs PR86 decomposes the current frontier:

| PR | archive bytes | Seg term | Pose term | Rate term | recomputed score |
|---:|---:|---:|---:|---:|---:|
| #85 | 236,328 | `0.057185` | `0.043520` | `0.157361` | `0.258066` |
| #86 | 207,579 | `0.067815` | `0.067603` | `0.138218` | `0.273636` |

PR86 buys `28,749` fewer bytes and `~0.01914` lower rate term, but loses
`~0.01063` SegNet and `~0.02408` PoseNet contribution. Net: PR85 remains the
public leaderboard floor by `~0.01557`.

PR89's stated local score is numerically close to PR84, but its author withdrew
after public-master CUDA mismatch. Treat `fb` as a cheap interaction hypothesis,
not as score evidence.

## Innovations Observed In PR85-PR100

PR85 introduced the currently dominant practical bundle:

- Stored evaluator-relevant 5-class SegNet mask IDs directly instead of a
  mask video.
- Used a custom adaptive range coder with spatial/temporal context.
- Packed the whole payload into a single `x` ZIP member with a v5 micro-bundle.
- Coupled `QMA9X` masks to a compact QZS3/QH0 neural renderer.
- Added tiny pose and correction side channels (`P1D1`, `post`, `shift`,
  `frac*`, `bias`, `region`, `randmulti`).

PR86 introduced the strongest distinct family:

- Learned master/slave neural representation instead of direct QZS3 clone.
- Archive members are actual learned/checkpoint/token payloads:
  `master.pt.gz`, `slave.pt.gz`, `hpac.pt.ppmd`, `tokens.bin`, `meta.pt`.
- Master: token-to-RGB CNN with FiLM frame modulation.
- Slave: per-frame NeRV-style decoder for first frames.
- HPAC: patch/group autoregressive token entropy model with arithmetic coding
  and PPMd-compressed model payload.
- Reproducibility signals: DALI version sensitivity, CPU FP32 FiLM table at
  inflate time, and quantized probability grid for cross-hardware portability.

PR87 exposed a non-faithful scoring loophole:

- The charged archive is a 100-byte dummy ZIP with empty member `a`.
- `inflate.py` is 767 KB and embeds large base85 payload literals and PR82-like
  correction streams.
- Useful only as a harness-forensics guardrail; not a model to copy into
  contest-faithful archive work.

PR88 was not a new frontier candidate:

- It copied/packaged the QZS3 range-mask runtime as a GPU preflight under a new
  submission name.
- The body report was CPU, not official CUDA score truth.
- The author closed it as opened in the wrong place.

PR89 added a cheap final-bias interaction idea:

- Archive is PR85's `x` plus a 300-byte `fb` member.
- `fb` packs 600 4-bit choices selecting one of nine tiny RGB biases per frame
  pair.
- The side channel is attractive because it is tiny and orthogonal to the mask
  stream, but the submitted score claim is withdrawn.

PR90-PR100: no public PRs exist at this fetch time.

## Implementation Recommendations

1. Treat PR85 as the score floor to beat and as the fixed-runtime integration
   target. The exact T4 replay confirms the public components, but the PR is
   closed unmerged, so custody must preserve external runtime SHA and archive
   SHA separately.
2. Finish a fail-closed PR85 v5 bundle parser/consumer in `robust_current`
   before new exact evals. The runtime must explicitly understand `QMA9X`,
   `QH0`, `P1D1`, `post`, `shift`, `frac`, `frac2`, `frac3`, `bias`,
   `region`, and `randmulti`.
3. Do not pursue coarse randmulti deletion as a promotion path. Existing local
   exact probes showed rate savings do not pay for PoseNet damage; use finer
   atom-level edits and side-channel preservation instead.
4. Mine PR86 for learned entropy coding, not direct byte transplant. First
   reproduce token decode/re-encode parity and version constraints; then test
   an Apogee-owned HPAC over PR85/QMA9 tokens or residual token maps.
5. Test PR89-style final micro-bias as a controlled, archive-charged atom only
   after it passes public-master runtime parity. It is exactly 300 payload
   bytes over PR85's `x`, so the break-even component improvement is small,
   but the withdrawn PR proves local harness mismatch risk is real.
6. Keep PR87/PR70 quarantined as guardrail cases. Any source-embedded payload
   or tiny dummy archive should fail strict public-replay dispatch unless an
   explicit forensic override is used.

