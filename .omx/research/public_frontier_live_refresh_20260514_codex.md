# Public Frontier Live Refresh - 2026-05-14

## Sources Checked

- official leaderboard: `https://comma.ai/leaderboard`
- challenge PRs: `https://github.com/commaai/comma_video_compression_challenge/pulls`
- PR107: `https://github.com/commaai/comma_video_compression_challenge/pull/107`
- PR108: `https://github.com/commaai/comma_video_compression_challenge/pull/108`

## Current Official Top Rows

Live official leaderboard and GitHub README mirror agree:

1. `0.193` - `hnerv_ft_microcodec` - PR101
2. `0.195` - `hnerv_lc_ac` - PR103
3. `0.195` - `hnerv_lc_v2_scale095_rplus1` - PR102
4. `0.195` - `hnerv_lc_v2` - PR100
5. `0.197` - `hnerv_muon_finetuned_from_pr95` - PR98
6. `0.198` - `kitchen_sink` - PR105
7. `0.199` - `hnerv_muon` - PR95
8. `0.206` - `rem2_HNeRV` - PR96
9. `0.209` - `belt_and_suspenders` - PR106
10. `0.229` - `vibe_coder_final_boss` - PR97
11. `0.229` - `apogee` - PR107

Artifact:
`experiments/results/public_frontier_postdeadline_20260514_codex/live_official_leaderboard_state_20260514.json`.

## PR107

No new PR107 comment after the 2026-05-05 maintainer note was found in the live
GitHub refresh. PR107 remains leaderboard row `apogee` at `0.229`.

The PR107 body already links the release asset, PR branch, `tac`, and
`comma-lab`, and states exact archive/runtime custody. That satisfies the
operator requirement that submission packets expose repository links and
reproducibility context.

## PR108 / Post-Deadline Policy

PR108 was closed 2026-05-11 with maintainer guidance that post-deadline
submissions need to be competitive or innovative:

- competitive: better than the top #1 submission;
- innovative: a novel idea not on the leaderboard yet, even if not competitive.

This policy is already enforced by `scripts/pre_submission_compliance_check.py`
through the `competitive_or_innovative` statement checks. The live refresh
found the same policy signal and hardened `tools/leaderboard_poll.py` so the
default poll target is now the official comma.ai leaderboard rather than only
the GitHub README mirror.

## Actionable Consequence

- No new public archive supersedes PR101.
- PR108 does not add a new score-lowering implementation to intake.
- Continue score-lowering against PR101/PR103/PR106/PR107-family bytes and
  exact CPU/CUDA dual-axis custody.
- Any new public/release packet must include a competitive-or-innovative
  statement plus public repo/reproducibility links before contest-final
  compliance can pass.
