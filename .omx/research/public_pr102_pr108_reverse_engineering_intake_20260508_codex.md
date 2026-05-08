# Public PR102/PR108 reverse-engineering intake

Created UTC: `2026-05-08T09:33:23Z`
Updated UTC: `2026-05-08T10:55:00Z`

Evidence grade: PR102 now has local `A++ contest T4` replay evidence for the
corrected archive/runtime pair; PR108 remains `[external+empirical-custody]`.

## Scope

Bounded lane: public contest artifact reverse-engineering intake and
deconstruction. Main branch only. This pass refreshed live upstream GitHub PRs,
the upstream README leaderboard, and local custody artifacts, then wrote a
small curated manifest:

- `reverse_engineering/public_pr102_pr108_intake_20260508/manifest.json`
- `reverse_engineering/public_pr102_pr108_intake_20260508/README.md`

Raw archives remain outside `reverse_engineering/`.

## Live upstream refresh

- Upstream repo: `https://github.com/commaai/comma_video_compression_challenge`
- README leaderboard refresh: PR101 first at `0.193`, PR103 second at `0.195`,
  PR102 third at `0.195`; PR108 was not listed.
- Latest public PRs from the GitHub pulls API included PR108, then PR107 down
  through PR100. PR108 is open; PR102 is merged.

## PR102 custody gap

- PR: `https://github.com/commaai/comma_video_compression_challenge/pull/102`
- Title: `hnerv_lc_v2_scale095_rplus1 submission (0.19538 CPU)`
- Author: `EthanYangTW`
- Head SHA: `1e330ec5633539c48278ce3cc96d2b15ea7a9eac`
- Merge commit: `0ac4e56c227bc79e94756217374490dea92a97a3`
- Canonical archive URL:
  `https://github.com/user-attachments/files/27369164/archive.zip`
- Local archive:
  `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/archive.zip`
- Archive bytes: `178981`
- Archive SHA-256:
  `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
- ZIP member: `0.bin`, stored, `178873` bytes, SHA-256
  `3234f0689164cfc95b7ee9f9cdf38ecf4d082cfb7048058e2b3ff0f54f864e43`

Classification:

- Corrected custody remains unblocked: the canonical PR102 archive is the
  maintainer-comment attachment and is byte-identical to the PR100 release
  archive referenced by PR102 `compress.sh`.
- The stale local auto intake
  `experiments/results/public_pr_archive_release_view/public_pr102_intake_20260505_auto/archive.zip`
  is the wrong qpose asset (`276481` bytes, member `p`) and must not be used for
  PR102 replay.
- Public upstream drift is real and now locally reproduced on contest-equivalent
  CUDA: PR102 has a CPU report around `0.19538`, a GitHub Actions CUDA eval
  comment recomputing to about `0.22839` from rounded fields, and a later CPU
  eval comment recomputing to about `0.195376`. The hardened local Lightning T4
  replay of the corrected archive/runtime landed at `0.22839372989108092`, with
  600 samples, archive SHA
  `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`, and
  runtime tree SHA
  `62560a0411dc341286eebfaf6e8ed79564efeb14fb8da5e3e1be026611e7aba1`.
  Classification: local exact CUDA confirms the public CUDA comment band and
  mismatches the public CPU/leaderboard band.

Runtime/compliance notes:

- `compress.sh` downloads and verifies the unchanged PR100 release archive.
- `inflate.sh` may install `brotli` at inflate time if missing. A local exact
  replay adapter should use the repo-managed environment and fail closed on
  missing dependencies rather than depending on network installs.
- `inflate.py` decodes HNeRV, applies the latent correction sidecar with
  `DELTA_SCALE=0.0095`, performs bicubic upsample, and adds `+1.0` to frame-0
  red channel before clamping/rounding.

Exact replay result:

- Job: `pr102-public-exact-replay-hardened-g4dn2-20260508T103725Z`
- Artifact dir:
  `experiments/results/lightning_batch/pr102-public-exact-replay-hardened-g4dn2-20260508T103725Z`
- Result summary:
  `.omx/research/pr102_hardened_exact_replay_result_20260508_codex.json`
- Score: `0.22839372989108092` `[contest-CUDA]`
- Remaining PR102 work: byte-level decode/re-encode parity and
  compress-to-archive reproduction; do not rerun the same archive/runtime unless
  custody or upstream evaluator state changes.

## PR108 non-frontier intake

- PR: `https://github.com/commaai/comma_video_compression_challenge/pull/108`
- Title: `andimin01`
- Author: `andrei-minca`
- State: open
- Head SHA: `59c1bbd544bb2aa166656d24d7de117ad3e3e62e`
- Archive URL:
  `https://github.com/user-attachments/files/27408563/archive.zip`
- Local archive:
  `experiments/results/public_pr108_andimin01_intake_20260508_codex/archive.zip`
- Archive bytes: `442979`
- Archive SHA-256:
  `127b0b318ba2355cdac0d513f4027f0ca3297be4cba0f44e1ddb25cc70586804`
- ZIP member: `0.mkv`, deflated, `448786` uncompressed bytes, `442819`
  compressed bytes, SHA-256
  `3541f5031914a76d8632e094703ec1f96e59c7fb07942963379fc3d82bbe3035`

Classification:

- PR108 is non-frontier by its own CPU report: pose `0.65178943`, seg
  `0.00745649`, archive `442979` bytes, recomputed score
  `3.593627238977` from rounded fields.
- No official GitHub eval result comment was present at refresh.
- It is not a HNeRV payload grammar target. It is an AV1/ROI/sharpening
  reference with a single MKV member.

Runtime/compliance notes:

- `compress.sh` runs `roi_preprocess.py`, downscales to `512x384`, encodes with
  `libsvtav1` preset 6 / CRF 40, and zips `archive/*.mkv`.
- `inflate.sh` expects extracted `${BASE}.mkv` and calls `python3 inflate.py`.
- `inflate.py` uses PyAV and torch, bilinear-upscales to `1164x874`, applies a
  sharpening convolution per channel, and writes uint8 raw RGB.
- Replay should capture PyAV, ffmpeg, torch, and runtime tree details before
  any comparison to the submitter's CPU report.

Fastest exact replay path:

1. Use PR108 source files at head SHA
   `59c1bbd544bb2aa166656d24d7de117ad3e3e62e` and the local archive path above.
2. Run `.venv/bin/python experiments/contest_auth_eval.py --archive ... --inflate-sh ... --upstream-dir upstream --device cuda` only if a drift explanation requires it.
3. Treat any result as classification evidence unless exact CUDA unexpectedly
   beats the validated frontier.

## Next actions

- PR102: exact CUDA replay is complete. Next highest-value work is deconstruction
  of the monolithic `0.bin`: decode/re-encode parity, compress-to-archive
  reproduction, and wire-grammar proof.
- PR108: keep the custody record; do not spend exact CUDA unless needed for a
  public drift audit or future leaderboard movement.
