# Public frontier refresh - 2026-05-15

## Source checks

- Repository: <https://github.com/commaai/comma_video_compression_challenge>
- README/leaderboard fetched from `master` on 2026-05-15 UTC.
- Pull requests checked with `gh pr list` and `gh pr view` for PRs 95-108.

## Current public leaderboard state

The README still names the prize winners and top public leaderboard entries:

- #1: PR #101 `hnerv_ft_microcodec`, displayed score `0.193`.
- #2: PR #103 `hnerv_lc_ac`, displayed score `0.195`.
- #3: PR #102 `hnerv_lc_v2_scale095_rplus1`, displayed score `0.195`.

README now also states that the challenge remains open for submissions after
the prize deadline, but the prize winners are fixed.

## Post-deadline PR/comment signal

- PR #108 is the only newer PR after the main prize wave. It was closed on
  2026-05-11 by YassineYousfi under new submission-guideline language:
  a new submission should be either better than current #1 or innovative with
  an idea not already on the leaderboard.
- PR #107 has no new technical comment after YassineYousfi's 2026-05-05
  leaderboard/job note.
- No `geohot`/George Hotz-authored PR comment was found in the checked PR
  comment range. The visible maintainer guidance is from YassineYousfi.
- The HW-axis discussion on PR #103 is still important: YassineYousfi stated
  that running all submissions on the same hardware was chosen for comparable
  scores, while the public README/ranking displays CPU leaderboard results.
  Keep `[contest-CPU]` and `[contest-CUDA]` separate.

## Implications for our queue

- The `0.192` local result remains a CPU-axis reproduction/improvement signal,
  not a CUDA frontier or submission-ready claim.
- The leaderboard source of truth did not move below PR #101 during this
  refresh.
- New post-deadline submissions must clear one of two bars: beat PR #101 on the
  public leaderboard axis, or introduce a clearly novel technical idea. Small
  AV1/ROI/sharpening variants are explicitly considered established.

