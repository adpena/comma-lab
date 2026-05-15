# Public frontier refresh - 2026-05-15

## Source Of Truth

- Official leaderboard checked 2026-05-15: `https://comma.ai/leaderboard`
- Video compression challenge status: `ACTIVE`
- Current listed frontier: PR #101 `hnerv_ft_microcodec` at `0.193`
- No public leaderboard row below `0.192` observed.

## Post-Deadline PR Signal

- PR #101: `https://github.com/commaai/comma_video_compression_challenge/pull/101`
  - Public CPU-axis bot rescore at 2026-05-05 reports approximately `0.192845`, with pose `0.00003286`, seg `0.00056023`, archive bytes `178258`.
  - Initial CUDA-axis bot score at 2026-05-04 reports approximately `0.226354`.
  - Actionable: CPU and CUDA axes remain separate promotion surfaces; paired eval is mandatory.
- PR #95: `https://github.com/commaai/comma_video_compression_challenge/pull/95`
  - HNeRV source/root submission. Body documents 229K-parameter HNeRV decoder, 28-d per frame-pair latents, 8-stage curriculum, Muon, and C1a entropy-shaping.
  - Maintainer comment at 2026-05-04 17:43 UTC says there is a small hardware difference in decode and submissions were run on T4 for fair comparison.
  - Actionable: use PR95 as source-faithful HNeRV control, but never convert CPU score into T4 score.
- PR #106: `https://github.com/commaai/comma_video_compression_challenge/pull/106`
  - `belt_and_suspenders`; CUDA report score `0.20946`, archive bytes `186239`, GPU required for evaluation.
  - Actionable: PR106 remains a useful sidecar/format laboratory, not a standalone frontier.
- PR #103: `https://github.com/commaai/comma_video_compression_challenge/pull/103`
  - Public CPU-axis score approximately `0.194881`, archive bytes `178223`.
  - Body/comment signal: range-coded the eight largest weight tensors plus latent-hi stream, saving about `290 B` over Brotli.
  - Actionable: PR103 is the highest-signal arithmetic/range-coding public lesson for PR101/PR106 export discipline.
- PR #107: `https://github.com/commaai/comma_video_compression_challenge/pull/107`
  - Our apogee PR received official eval at `0.23`.
  - 2026-05-05 maintainer comment: email comma with a link to the PR for job/internship follow-up.
  - Actionable: future submission packets should include repo links, exact archive/runtime custody, and OSS extraction links by default.
- PR #108: `https://github.com/commaai/comma_video_compression_challenge/pull/108`
  - Closed 2026-05-11 under new submission guidance.
  - Maintainer quoted required framing: competitive means better than top #1; innovative means novel idea not on leaderboard yet with potential.
  - Actionable: do not submit incremental AV1/ROI/sharpening-style work unless it is either < current top or paired with a genuinely new mechanism.

## Integration Decisions

1. Keep hard threshold: only submit if exact, contest-compliant score is `<0.192`.
2. Record `[contest-CUDA]`, `[contest-CPU]`, and provider/advisory axes separately in every artifact.
3. Favor byte-closed PR95/PR101-style HNeRV-family exact replay and PR106 format0C sidecar deconstruction for short-term moves.
4. Promote PR101/PR103 microcodec/range-coded payload sections to first-class TAC packet compiler surfaces rather than another ad hoc sidecar.
5. Treat generic video-codec tweaks as low priority unless xray/scorer evidence proves orthogonality to the HNeRV basin.
