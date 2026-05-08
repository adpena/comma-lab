# PR102 Linux CPU Auth Eval via GitHub Actions - 2026-05-08

## Artifact

- Evidence grade: `contest-CPU-1to1`
- Workflow run: `25571618194`
- Fork PR/runtime adapter: `adpena/comma_video_compression_challenge#2`
- Release tag: `cpu-eval-pr102_hnerv_lc_v2_scale095_rplus1_cpu_20260508T1815Z-20260508T180940Z`
- Local adjudicated artifact: `.omx/research/artifacts/pr102_linux_cpu_auth_eval_gha_20260508_codex.json`
- Ignored custody artifact: `experiments/results/public_pr102_cpu_auth_eval_gha_20260508T1815Z/contest_auth_eval.adjudicated.json`
- Report: `experiments/results/public_pr102_cpu_auth_eval_gha_20260508T1815Z/report.txt`

## Archive Custody

- Archive path: `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/archive.zip`
- Archive bytes: `178981`
- Archive SHA-256: `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
- Archive member: `0.bin`
- Archive member SHA-256: `3234f0689164cfc95b7ee9f9cdf38ecf4d082cfb7048058e2b3ff0f54f864e43`

## Result

- Device: `cpu`
- Hardware: `github-actions-ubuntu-latest-x86_64`
- Samples: `600`
- Average PoseNet distortion: `0.00003460`
- Average SegNet distortion: `0.00057601`
- Compression rate: `0.00476704`
- Reported display score: `0.20`
- Recomputed canonical score: `0.19537807523773826`

The upstream report rounds the displayed final score to two decimals, so the
canonical value is recomputed from components:

```text
100 * 0.00057601 + sqrt(10 * 0.00003460) + 25 * 0.00476704
= 0.19537807523773826
```

## Harvester Fix

The first harvest pass wrote `canonical_score=null` and a bogus recomputed
score of `0.0` because `tools/harvest_gha_runs.py` only parsed `key=value`
rows and used `+ rate` instead of `+ 25 * rate`. This was a custody bug in the
harvester, not a score result. The parser now fails closed on missing upstream
report fields and records the unrounded recomputed score while retaining the
rounded display score.

## Dispatch Claim

The lane `public_pr102_cpu_auth_eval_gha` was closed with terminal status
`completed_score_0.19537807523773826` for job
`gha-pr102-cpu-20260508T1815Z-pr2`.
