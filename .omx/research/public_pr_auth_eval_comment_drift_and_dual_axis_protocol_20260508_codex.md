# Public PR auth-eval comments and dual-axis protocol - 2026-05-08

Evidence grade: `external_github_pr_comment` plus `tooling_protocol`.
Score claim: false. Promotion/rank/kill claim: false.

## Finding

Live GitHub PR comment refresh confirms that several PR100+ submissions have
both CUDA and CPU host eval comments. The apparent drift between local T4 CUDA
replays and public medal-band scores is not explained by a broken local CUDA
replay for PR102/PR104; local CUDA matches the public CUDA comments within
rounded-component noise. The large PR102 drift is device-axis drift: CPU comment
band is lower than CUDA comment band on the same byte count.

## Comment scorecard

Scores below are recomputed from the rounded PoseNet, SegNet, and byte fields
printed in the host `github-actions` PR comments.

| PR | Public title | CUDA comment | CPU comment | Notes |
| ---: | --- | ---: | ---: | --- |
| 100 | hnerv_lc_v2 submission (0.1954) | `0.228269572711` | `0.195385423975` | CPU comment aligns with public title band. |
| 101 | add hnerv ft microcodec submission | `0.226354458744` | `0.192845012702` | CPU comment aligns with public first-place band. |
| 102 | hnerv_lc_v2_scale095_rplus1 submission (0.19538 CPU) | `0.228390831180` | `0.195376176526` | Local T4 replay `0.22839372989108092` matches CUDA comment. |
| 103 | hnerv_lc_ac submission (0.19) | `0.227764851625` | `0.194880702889` | CPU comment aligns with public title band. |
| 104 | qhnerv_ft_best | `0.231145103318` | none observed | Local T4 replay `0.23113446620399658` matches CUDA comment. |
| 105 | kitchen_sink (0.19797) | `0.230437255695` | `0.197973979344` | CPU comment aligns with public title band. |
| 106 | belt_and_suspenders (0.20946) | `0.209456642376` | none observed | Local PR106 adapter replay `0.20945673680571203` matches CUDA comment. |
| 107 | apogee submission (0.2293) | `0.229331025025` | none observed | No public CPU comment observed for our PR107 packet. |
| 108 | andimin01 | none observed | none observed | No host eval comment in this refresh. |

## Engineering conclusion

1. CUDA exact replay is not currently broken for the inspected PR102/PR104
   cases. CUDA-vs-CUDA matches.
2. Public PR/leaderboard comparisons need a second axis: exact CPU auth eval.
   This must be measured, not inferred from CUDA.
3. Local macOS CPU is not contest-compliant CPU evidence. `[contest-CPU]`
   requires Linux x86_64 custody; local macOS CPU can only be advisory.
4. CUDA remains the internal promotion/ranking/kill/paper-score truth. CPU is
   a public leaderboard/PR-comment reproduction axis.

## Tooling landed

- `tools/public_pr_eval_comment_scorecard.py`: fetches PR comments with `gh pr
  view`, parses host eval rows, and recomputes score from rounded components.
- `tools/plan_public_pr_cpu_auth_eval.py`: plans or executes public-PR CPU
  replay from the PR100-107 reproduction ledger.
- `tools/plan_dual_device_auth_eval.py`: emits paired CPU and CUDA auth-eval
  commands for the exact same archive/runtime.
- `experiments/contest_auth_eval.py`: stamps full-sample Linux x86_64 CPU
  output as `evidence_grade="contest-CPU"` and full-sample macOS CPU output as
  `evidence_grade="macOS-CPU advisory"`. CPU artifacts always set
  `promotion_eligible=false`, `score_claim_valid=false`, and
  `rank_or_kill_eligible=false`.

## Commands used

```bash
.venv/bin/python tools/public_pr_eval_comment_scorecard.py \
  --pr-range 100 108 \
  --json-out reports/public_pr100_108_eval_comment_scorecard_20260508.json

.venv/bin/python tools/plan_dual_device_auth_eval.py \
  --public-pr 102 \
  --json-out reports/pr102_dual_device_auth_eval_plan_20260508.json
```

## Source URLs

- PR100: https://github.com/commaai/comma_video_compression_challenge/pull/100
- PR101: https://github.com/commaai/comma_video_compression_challenge/pull/101
- PR102: https://github.com/commaai/comma_video_compression_challenge/pull/102
- PR103: https://github.com/commaai/comma_video_compression_challenge/pull/103
- PR104: https://github.com/commaai/comma_video_compression_challenge/pull/104
- PR105: https://github.com/commaai/comma_video_compression_challenge/pull/105
- PR106: https://github.com/commaai/comma_video_compression_challenge/pull/106
- PR107: https://github.com/commaai/comma_video_compression_challenge/pull/107
- PR108: https://github.com/commaai/comma_video_compression_challenge/pull/108
