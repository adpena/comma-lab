# Public Frontier Watch Postdeadline - 2026-05-14

Generated UTC: 2026-05-14T16:46:59Z

Scope: PUBLIC-FRONTIER-WATCH-POSTDEADLINE. Read-only live check of
`commaai/comma_video_compression_challenge` since the 2026-05-05T22:47:20Z
leaderboard baseline. No dispatch was attempted. No `gh pr checkout` was used.

Evidence grade: external GitHub API, official leaderboard/README refresh,
public PR comment/file inspection, and local static hashing of small downloaded
artifacts in ignored custody under
`experiments/results/public_frontier_watch_postdeadline_20260514_codex/`.
This ledger is not a score claim.

## Verdict

No actual leaderboard movement was found after 2026-05-05T22:47:20Z.

The official public frontier still routes through the same HNeRV family rows:

| Rank surface | Submission | Public rounded score |
| --- | --- | ---: |
| Official leaderboard / current README | #101 `hnerv_ft_microcodec` | 0.193 |
| Official leaderboard / current README | #103 `hnerv_lc_ac` | 0.195 |
| Official leaderboard / current README | #102 `hnerv_lc_v2_scale095_rplus1` | 0.195 |
| Official leaderboard / current README | #100 `hnerv_lc_v2` | 0.195 |

Post-baseline PR activity was limited to #108, #105, #95, and #71. #108 was
closed as non-novel/non-frontier under the newer submission guidance. #105,
#95, and #71 had prize, write-up, CPU/CUDA rerun, or logistics comments, but
no new leaderboard-better archive or code path.

Routing decision: no next dispatch/build changes. Keep PR101/PR95/HNeRV-family
exact-eval and byte-anatomy work as the active public-frontier control surface.
Treat PR105's write-up/storybook material as research signal for semantic-codec
and selector-packet design, not as a new exact-eval target.

## Live Sources

- Official leaderboard: <https://comma.ai/leaderboard>
- Current upstream README: <https://github.com/commaai/comma_video_compression_challenge/blob/master/README.md>
- Current upstream `master`: `d0013db5a97066414217667e82673254eff2347d`
- Current README blob SHA: `e6657cf4cbf9a98c5144d6dd3a3866f8efcc5b89`
- Post-baseline upstream commits:
  - `f56b918579864ebed40f96bcfe4e0b9192edf8e9` - `prizes claimed`
  - `73cf750131c0bbf6e4bc71ea575070fadcdb960a` - `update readme + pr template`
  - `d0013db5a97066414217667e82673254eff2347d` - `update pr template`

README/pr-template movement is process-only: the prize header now says prizes
are claimed, README links community write-ups, and the PR template asks whether
a new PR is "competitive" or "innovative" with guidance that competitive means
better than the top #1 submission.

## Updated PRs After Baseline

GitHub API search for PRs updated after `2026-05-05T22:47:20Z` returned only
these four PRs.

| PR | Author | State | Updated UTC | Head SHA | Finding |
| --- | --- | --- | --- | --- | --- |
| #108 <https://github.com/commaai/comma_video_compression_challenge/pull/108> | `andrei-minca` | closed | 2026-05-11T19:19:58Z | `59c1bbd544bb2aa166656d24d7de117ad3e3e62e` | Non-frontier AV1/ROI idea; closed by maintainer as already-established tricks under new guidance. |
| #105 <https://github.com/commaai/comma_video_compression_challenge/pull/105> | `valtterivalo` | merged | 2026-05-06T08:01:30Z | `9376a6f86c76cb576b5f25afd8c789a8a727077f` | New comments only: CUDA/CPU eval comments, failed/cancelled reruns, prize/write-up logistics. No new archive. |
| #95 <https://github.com/commaai/comma_video_compression_challenge/pull/95> | `AaronLeslie138` | merged | 2026-05-06T01:44:12Z | `9bdce26f2a4f996828c4e3fa2b87c454a0e8fcc9` | New comments only: CUDA/CPU axis clarification and prize/write-up logistics. No new archive. |
| #71 <https://github.com/commaai/comma_video_compression_challenge/pull/71> | `TomDousek` | merged | 2026-05-06T12:13:08Z | `6aec739a588342302330812c30a972736ff8c45b` | New comments only: write-up/job logistics. Non-frontier score band. |

## Score/Artifact Custody

### PR108 `andimin01`

- URL: <https://github.com/commaai/comma_video_compression_challenge/pull/108>
- Close event/comment: <https://github.com/commaai/comma_video_compression_challenge/pull/108#issuecomment-4424101686>
- Claimed/body score: `3.593627238977`
- Archive URL: <https://github.com/user-attachments/files/27408563/archive.zip>
- Archive bytes/SHA-256: `442979` /
  `127b0b318ba2355cdac0d513f4027f0ca3297be4cba0f44e1ddb25cc70586804`
- ZIP member: `0.mkv`, SHA-256
  `3541f5031914a76d8632e094703ec1f96e59c7fb07942963379fc3d82bbe3035`
- PR file blobs:
  - `submissions/andimin01/compress.sh`:
    `e5e967dc6c00dad46599c75138947530ea00beef`
  - `submissions/andimin01/inflate.py`:
    `4d7d21796c7f89892ef88ae69c501535954d7294`
  - `submissions/andimin01/inflate.sh`:
    `28fbf7658d30f8bce3fd15dda198346746b665ad`
  - `submissions/andimin01/roi_preprocess.py`:
    `40ca1f90b9938f9677b23513e9400ab87a72ff82`
- Classification: non-frontier and already covered by known ROI/preprocess +
  AV1/video-codec family ideas. No intake action beyond the existing
  PR102/PR108 manifest.

### PR105 `kitchen_sink`

- URL: <https://github.com/commaai/comma_video_compression_challenge/pull/105>
- CUDA eval comment:
  <https://github.com/commaai/comma_video_compression_challenge/pull/105#issuecomment-4372737788>
- CPU eval comment:
  <https://github.com/commaai/comma_video_compression_challenge/pull/105#issuecomment-4381357121>
- CUDA comment fields: seg `0.00070456`, pose `0.00017267`, bytes `177857`,
  recomputed score `0.230437255695`
- CPU comment fields: seg `0.00060913`, pose `0.00003472`, bytes `177857`,
  recomputed score `0.197973979344`
- Archive bytes/SHA-256 from local public intake:
  `177857` /
  `597ba0732810eba08cdae619b679d211d398bc0249b8831898f7096d5beece1d`
- Member `0.bin` SHA-256:
  `5f88fa8ed26816cf6e86a00cf3c88915963c2a84f899173fd6820359df29aad0`
- Failed/cancelled post-baseline action runs:
  - <https://github.com/commaai/comma_video_compression_challenge/actions/runs/25387480323>
  - <https://github.com/commaai/comma_video_compression_challenge/actions/runs/25388135899>
- Classification: exact public PR105 remains non-frontier on CUDA/T4 and
  already known as CPU-better than CUDA. No new score movement.

### PR95 `hnerv_muon`

- URL: <https://github.com/commaai/comma_video_compression_challenge/pull/95>
- CUDA eval comment:
  <https://github.com/commaai/comma_video_compression_challenge/pull/95#issuecomment-4372709477>
- CPU eval comment:
  <https://github.com/commaai/comma_video_compression_challenge/pull/95#issuecomment-4381186907>
- CUDA comment fields: seg `0.00070728`, pose `0.00017185`, bytes `178417`,
  recomputed score `0.230983351496`
- CPU comment fields: seg `0.00061212`, pose `0.00003494`, bytes `178417`,
  recomputed score `0.198704801220`
- Archive bytes/SHA-256 from local public intake:
  `178417` /
  `e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a`
- Member `0.bin` SHA-256:
  `4b8013fb1168e21b12fc7ee6c395e032660d88daded48a0b845085e7d5eb11c4`
- Classification: still useful as HNeRV control-arm source. No new archive or
  code material after the baseline.

### PR71 `ditcher`

- URL: <https://github.com/commaai/comma_video_compression_challenge/pull/71>
- CUDA eval comment:
  <https://github.com/commaai/comma_video_compression_challenge/pull/71#issuecomment-4365172070>
- Body CPU-style fields: seg `0.00328412`, pose `0.00358563`, bytes `290747`,
  recomputed score `0.711366091254`
- CUDA comment fields: seg `0.00332314`, pose `0.00366678`, bytes `290732`,
  recomputed score `0.717388886008`
- Downloaded archive bytes/SHA-256:
  `290732` /
  `2849248c0d391b1be5b586b8d2a4d66140013e6393f73d64d4be76d730d3ac12`
- ZIP members:
  - `pose.br`: `ab808624f2912bcd7307502968f0bc4e4e49b34aba100c957ba8ef5155922962`
  - `mask.br`: `e40059804cb5c1284be8e2f176a982673f517bcb128e981aa4ea2811391f7cea`
  - `obrdo.br`: `9f28e16d90e2934bb666eadfaf94c72712a0f9733b6612a81c706e31401159e5`
  - `meta.pkl`: `d534b5d728c5cbd3c22f7730b36ce2b85f13bae983c902b47f12ffed51b1a4de`
  - `inter_pose.br`: `7792a09dad39c622548ab74ce3ebc66967e8e1278864234454f3fc32b77c6b1c`
- Classification: non-frontier but conceptually useful for uncertainty-map and
  segmentation/pose split thinking. No immediate build change.

## Author-Linked Sources And Post-Deadline Learning

Downloaded source copies are ignored custody artifacts only. Durable metadata is
also recorded in
`reverse_engineering/public_frontier/postdeadline_watch_20260514_manifest.json`.

| Source | URL | Local SHA-256 | Learning |
| --- | --- | --- | --- |
| Valtteri kitchen-sink storybook | <https://github.com/user-attachments/files/27357161/kitchen-sink.html> | `d8f5e279a1bbcde6b83d6c2c1a016a8e4bf150d9aabf71b690b49e975a4d0c78` | The non-final semantic-codec branch had a 231,891-byte archive with mask-cache dominance: mask stream `169686` bytes, pose stream `3133` bytes, selector tails `691` bytes, and a T4 inflate-time optimizer at `1709.9s`. This is research-only design signal for mask-cache/selector/range-coder lanes, not a new frontier archive. |
| Aaron blog shell | <https://aaronleslie.dev/blog/comma-compression> | `a33978cb22cc8a863a39ad5980dafab36ee80c2bf855106212d889839efd9f11` | Static fetch was a JS shell; PR95 source remains the useful control artifact. |
| Tom visualization page | <https://tomdousek.github.io/> | `20f836a256f8ad4e615472245414e7a27df213fcb4b1a48dd5701454e3dbd463` | Uncertainty visualization highlights boundary/distant-object SegNet error; author notes redundant learned kernels and suggests Ghost Convolutions. |
| Brady write-up shell | <https://comma-writeup.pages.dev/> | `8349eca1dae98a64b65880b631c87e646b7f5667a3ea1d6fb2ca2ded18dfcf73` | Static fetch was a JS shell in this pass. No routing change. |
| Sajay QRepro README | <https://github.com/SajayR/comma_video_compression_challenge/blob/cce857392701e73861ad513d34906faba523f719/submissions/qrepro/README.md> | `d20d37f20041f18f66e3a8ca2658d074cd274cb5d42c80ccc5c0c86589304832` | Non-frontier score `0.278872`; useful composition note: semantic mask stream + FP4 renderer + sparse QRGB residuals. |

## No New Dispatch

This sweep changes no exact-eval dispatch queue. The only actionable carryover is
to ensure future public-frontier HNeRV work keeps the PR105 semantic-codec
lessons separate from the exact PR105 final archive result:

- semantic mask-cache coding and pose delta range coding are design priors;
- selector-tail packing is a small byte-budget idea;
- T4 inflate-time optimization remains budget-sensitive and must be measured
  against the contest runtime;
- none of the above supersedes the current exact-CUDA public-frontier control
  evidence.
