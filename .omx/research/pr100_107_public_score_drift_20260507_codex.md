# PR100-PR107 public score drift audit - 2026-05-07

Evidence grade: `[external:github-pr-comments]` plus `[empirical:local-custody-inventory]`.
This note records a scoreboard-custody issue only. It is not a score claim for
any internal lane.

## Finding

The apparent drift between our local auth-eval scores and the upstream
leaderboard is real for several PR100+ public HNeRV submissions, but it is not
caused by a different upstream `evaluate.py` in this checkout. The local
`upstream/evaluate.py` SHA-256 matches GitHub `master`:

- `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`

The drift comes from replay mode. GitHub PR comments show those submissions
were evaluated once with `device: cuda` and later with `device: cpu`; the lower
leaderboard values line up with the CPU reruns, while our local public replay
artifacts for PR100/101/103/105 are CUDA/T4 artifacts.

## Public PR comment evidence

| PR | upstream row | CUDA PR-comment score | CPU PR-comment score | local same-archive replay in ledger | classification |
|---:|---:|---:|---:|---:|---|
| 100 | 0.195 | 0.23 | 0.20 | cuda:0.228269 | leaderboard follows lower CPU path, local replay is CUDA |
| 101 | 0.193 | 0.23 | 0.19 | cuda:0.226353 | leaderboard follows lower CPU path, local replay is CUDA |
| 102 | 0.195 | 0.23 | 0.20 | none | corrected archive needs same-archive local replay |
| 103 | 0.195 | 0.23 | 0.19 | cuda:0.227765 | leaderboard follows lower CPU path, local replay is CUDA |
| 105 | 0.198 | 0.23 | 0.20 | cuda:0.230437 | leaderboard follows lower CPU path, local replay is CUDA |
| 106 | 0.209 | 0.21 | not observed in this pass | cuda:0.209457 | no material drift at rounded leaderboard precision |
| 107 | 0.229 | 0.23 | not observed in this pass | cuda:0.229331 | no material drift at rounded leaderboard precision |

The source comments were read live with:

```bash
gh pr view <PR> --repo commaai/comma_video_compression_challenge \
  --json number,title,author,createdAt,updatedAt,closedAt,mergedAt,headRefOid,url,comments
```

Key timestamps:

- PR100 CUDA comment: `2026-05-04T16:35:49Z`; CPU comment:
  `2026-05-05T17:47:33Z`.
- PR101 CUDA comment: `2026-05-04T16:36:14Z`; CPU comment:
  `2026-05-05T16:58:50Z`.
- PR102 CUDA comment: `2026-05-04T16:58:03Z`; CPU comment:
  `2026-05-05T17:05:18Z`.
- PR103 CUDA comment: `2026-05-04T16:36:21Z`; CPU comment:
  `2026-05-05T16:37:33Z`.
- PR105 CUDA comment: `2026-05-04T16:37:57Z`; CPU comment:
  `2026-05-05T17:01:12Z`.
- PR106 CUDA comment: `2026-05-04T16:37:44Z`.
- PR107 CUDA comment: `2026-05-04T16:38:25Z`.

## Impact on our ledgers

`tools/build_pr100_107_reproduction_ledger.py` now separates:

- `leaderboard_score`: the upstream leaderboard/metadata score.
- `exact_eval_artifacts[*].score`: local same-archive replay score, with
  device, component distances, score basis, and runtime-tree SHA when present.
- `exact_eval_summary`: same-archive counts separated from broad PR-glob eval
  hits, with replay identity keyed by archive SHA, device, and runtime-tree SHA.
- `same_archive_structured_exact_eval_json_missing`: a fail-closed proof
  blocker when the only structured JSON belongs to a different repack archive
  or when a same-archive score was recovered only from a legacy log line.
- `leaderboard_replay_drift.status`: whether same-archive local replay matches
  the leaderboard or mismatches by replay mode.

Current regenerated summary:

- `leaderboard_replay_drift_count`: `4`
- `missing_same_archive_structured_json_count`: `6`
- Drift rows: PR100, PR101, PR103, PR105.
- No same-archive local replay yet: PR102, PR104.
- Matches at current precision: PR106, PR107.

## Operating rule

Do not compare a local CUDA replay directly against a public CPU leaderboard row
without labeling the device and runtime-tree SHA. For our own promoted internal
lanes, continue requiring exact archive custody and CUDA auth eval unless the
operator explicitly starts a separate CPU-public-leaderboard reproduction lane.

For public-frontier reverse engineering, record both:

- `public_leaderboard_mode`: CPU/CUDA/unknown, from PR comments or official
  leaderboard metadata.
- `local_replay_mode`: CPU/CUDA/unknown, from the exact local artifact.

Any score rollup that has only one of these is incomplete and must not be used
to rank, kill, or promote a stack atom.
