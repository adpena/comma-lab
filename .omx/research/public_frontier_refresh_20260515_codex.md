# Public Frontier Refresh - 2026-05-15

Operator request: refresh the official leaderboard, post-deadline PR updates,
PR #107 comments, and George/geohot/Yousfi signal before choosing the next
score-lowering path. This is external-signal intake only; exact local
promotion still requires archive/runtime custody and paired axis labels.

## Commands

```bash
gh pr list -R commaai/comma_video_compression_challenge --state all --limit 40 \
  --json number,title,state,author,createdAt,updatedAt,headRefOid,url

gh api repos/commaai/comma_video_compression_challenge/readme

gh api repos/commaai/comma_video_compression_challenge/pulls/107

gh api repos/commaai/comma_video_compression_challenge/issues/107/comments --paginate

for n in 101 102 103 105 107 108; do
  gh api repos/commaai/comma_video_compression_challenge/issues/$n/comments --paginate
done

gh search issues --repo commaai/comma_video_compression_challenge \
  --commenter geohot --limit 20
```

## Official README / Leaderboard Snapshot

Live upstream README as of this pass still says the challenge is open for
submissions after the original May 3 deadline, and final ranking is based on
the public leaderboard with no private testing.

Top official README leaderboard entries:

| Rank | PR | Name | README score |
|---:|---:|---|---:|
| 1 | #101 | `hnerv_ft_microcodec` | 0.193 |
| 2 | #103 | `hnerv_lc_ac` | 0.195 |
| 3 | #102 | `hnerv_lc_v2_scale095_rplus1` | 0.195 |
| 4 | #100 | `hnerv_lc_v2` | 0.195 |
| 5 | #98 | `hnerv_muon_finetuned_from_pr95` | 0.197 |

No newer PR currently displaces PR #101 on the official README leaderboard.

## Latest PR Movement

Latest 40 PR query showed #108 as the newest post-deadline submission:

- #108 `andimin01`, closed, updated `2026-05-11T19:19:58Z`.
- Yassine comment on #108: closing per new submission guidelines; established
  tricks are insufficient unless the submission is competitive against #1 or
  innovative with a novel idea not already on the leaderboard.

Actionable implication: further public submissions should not be incremental
HNeRV/selector churn unless they either beat PR #101 or demonstrate a genuinely
new mechanism. This supports the current gate: do not submit above `<0.192`.

## PR #107 Status

PR #107 `apogee submission (0.2293)` has no new technical comment after the
May 5 closeout. Comments:

- `2026-05-04T16:38:25Z`: GitHub Actions CUDA eval result, score rounded to
  `0.23`.
- `2026-05-05T22:06:03Z`: Yassine congratulated leaderboard placement and
  suggested emailing `givemeajob@comma.ai` with a PR link.

No new PR #107 instruction changes the technical path.

## PR #101 / #102 / #103 / #105 Axis Signal

GitHub Actions comments confirm the critical CPU/CUDA split:

| PR | CUDA rounded score | CPU rounded score | Key implication |
|---:|---:|---:|---|
| #101 | 0.23 | 0.19 | Official winner depends on CPU-axis behavior; CUDA is much worse. |
| #103 | 0.23 | 0.19 | CPU request/dispute documents hardware sensitivity. |
| #102 | 0.23 | 0.20 | Same HNeRV family, same CPU/CUDA cliff. |
| #105 | 0.23 | 0.20 | Kitchen-sink writeup is useful, but not score-frontier. |

The PR #103 discussion is the clearest public evidence that small hardware
differences become large at the optimized HNeRV frontier. Yassine's stated
preference was same-hardware comparability for prize decisions, while the
README now says evaluation can be CPU or CUDA depending on submission needs.
For our repo this means paired evidence is mandatory: `[contest-CPU]` can be
a legitimate public-axis result, but it must never be silently promoted across
to `[contest-CUDA]`.

Local exact paired PR101/FEC6 evidence remains:

- `[contest-CPU]`: `0.1920513168811056`
- `[contest-CUDA/T4]`: `0.22621002169349796`
- Review packets:
  `.omx/research/pr101_fec6_fixed_huffman_k16_cpu_result_review_20260515_codex.json`
  and
  `.omx/research/pr101_fec6_fixed_huffman_k16_cuda_result_review_20260515_codex.json`

This is real CPU-axis signal, but it does not satisfy the operator's strict
`<0.192` submission gate and it is not a CUDA promotion result.

## George / geohot Signal

`gh search issues --repo commaai/comma_video_compression_challenge --commenter geohot`
returned no results in this repository. I found Yassine comments on PRs and
issues, but no direct George/geohot technical comment surface to integrate in
this pass.

## Next Build Decision

Do not spend another turn on byte-only PR101 selector shrink unless it crosses
`<0.192` CPU or materially changes CUDA. Current external signal supports:

1. PR106/format0C-native score-table materializer, because it makes the
   existing scorer table byte-closed and interpretable instead of routing
   through stale `0.bin` materialization.
2. Cross-family/time-traveler/Z4/Z5/C1 substrate work, because #108 confirms
   incremental established tricks are no longer enough for post-deadline PRs.
3. Paired CPU/CUDA measurement by default for any candidate, with runtime
   custody including `inflate.sh`.

