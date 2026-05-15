# Public Frontier Live Refresh - 2026-05-15

score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
evidence_grade: external_public_refresh

## Sources Checked

Commands run 2026-05-15 UTC:

```bash
.venv/bin/python tools/leaderboard_poll.py --initialize-baseline --source official \
  --state-path experiments/results/public_frontier_postdeadline_20260515_codex/live_official_leaderboard_state_20260515.json

.venv/bin/python tools/leaderboard_poll.py --initialize-baseline --source github-readme \
  --state-path experiments/results/public_frontier_postdeadline_20260515_codex/live_github_readme_leaderboard_state_20260515.json

gh pr list --repo commaai/comma_video_compression_challenge --state all --limit 120 \
  --json number,title,state,author,createdAt,updatedAt,closedAt,mergedAt,url,headRefName,headRefOid

gh pr view 107 --repo commaai/comma_video_compression_challenge \
  --json number,title,state,author,createdAt,updatedAt,closedAt,mergedAt,comments,reviews,url,headRefName,headRepositoryOwner,headRefOid,files

gh api 'repos/commaai/comma_video_compression_challenge/issues/comments?since=2026-05-14T00:00:00Z&per_page=100'
gh api 'repos/commaai/comma_video_compression_challenge/pulls/comments?since=2026-05-14T00:00:00Z&per_page=100'
gh api 'search/issues?q=repo:commaai/comma_video_compression_challenge+george+OR+geohot+OR+hotz'
```

Generated ignored evidence bundle:

- `experiments/results/public_frontier_postdeadline_20260515_codex/live_official_leaderboard_state_20260515.json`
- `experiments/results/public_frontier_postdeadline_20260515_codex/live_github_readme_leaderboard_state_20260515.json`
- `experiments/results/public_frontier_postdeadline_20260515_codex/pr_list_all_20260515.json`
- `experiments/results/public_frontier_postdeadline_20260515_codex/pr101.json`
- `experiments/results/public_frontier_postdeadline_20260515_codex/pr103.json`
- `experiments/results/public_frontier_postdeadline_20260515_codex/pr107.json`
- `experiments/results/public_frontier_postdeadline_20260515_codex/pr108.json`
- `experiments/results/public_frontier_postdeadline_20260515_codex/postdeadline_issue_comments_since_20260514.tsv`
- `experiments/results/public_frontier_postdeadline_20260515_codex/postdeadline_review_comments_since_20260514.tsv`
- `experiments/results/public_frontier_postdeadline_20260515_codex/george_geohot_hotz_issue_search_20260515.json`

## Leaderboard State

The official comma.ai leaderboard and GitHub README mirror agree:

- entries: `56`
- score column hash: `cfd9fe0f97383c6165f5536ce065abd8c22e253c868d9a41d4e042e51a220c3c`
- frontier identity hash: `1592f97f7d62c55907dc77e2762eeaace6e170ca574fd52951a89f103942f895`

Top rows:

1. `0.193` - `hnerv_ft_microcodec` - PR101
2. `0.195` - `hnerv_lc_ac` - PR103
3. `0.195` - `hnerv_lc_v2_scale095_rplus1` - PR102

No live public leaderboard movement below PR101 was observed.

## PR / Comment Movement

No PR in the repository has `updatedAt >= 2026-05-14T00:00:00Z`.

Post-2026-05-14 comment searches returned zero rows:

- issue comments since 2026-05-14: `0`
- pull review comments since 2026-05-14: `0`

PR107 remains unchanged:

- PR: `https://github.com/commaai/comma_video_compression_challenge/pull/107`
- title: `apogee submission (0.2293)`
- state: `closed`
- updated: `2026-05-05T22:39:09Z`
- public eval comment: `[contest-CUDA]`, rounded `0.23`, bytes `178392`
- maintainer comment: 2026-05-05 congratulatory / job-internship contact note

PR108 remains the latest post-deadline governance signal:

- PR: `https://github.com/commaai/comma_video_compression_challenge/pull/108`
- title: `andimin01`
- closed: `2026-05-11T19:19:57Z`
- no leaderboard row

Search for George / geohot / Hotz comments in the challenge repository returned
`total_count=0` for this refresh.

## Consequence

No new public archive or public PR comment changes the score-lowering queue.
The right next work remains internal:

1. use PR101/PR103 as CPU-axis HNeRV control arms;
2. use PR106/R2 PacketIR as the strongest CUDA PacketIR byte/compiler control;
3. avoid rate-only polishing where the paired-axis xray shows component drift
   dominates;
4. reopen film-grain / selector / water-fill only with CUDA-in-loop component
   rows or a charged scorer objective;
5. keep future public packets explicitly competitive or innovative, with
   public repo links, archive SHA, runtime SHA, exact eval artifacts, and
   deterministic reproduction commands.
