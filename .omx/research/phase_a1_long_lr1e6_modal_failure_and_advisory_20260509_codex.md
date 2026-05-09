# Phase A1 Long LR1e-6 Modal Failure And Advisory - 2026-05-09

## Verdict

The Modal long A1 run trained and built a valid archive, but remote CUDA auth
eval failed before scoring because `contest_auth_eval.py` rejected the Modal
runner's temp evidence path:

```text
contest_auth_eval evidence path is under temp storage: /tmp/modal_phase_a1/.../eval_work.
Choose a durable repo/provider work dir or pass --allow-temp-work-dir for diagnostic scratch only.
```

Classification: infrastructure / custody bug, not score-collapse. The bug is
fixed in `experiments/modal_phase_a1_score_gradient_pr101.py` by moving
`REMOTE_OUT_ROOT` to
`/workspace/pact/experiments/results/modal_phase_a1_remote/...` and adding a
regression test that forbids temp score-evidence output and forbids
`--allow-temp-work-dir` in this score-bearing dispatcher path.

The harvested archive was screened locally on macOS CPU advisory:

- canonical score: `0.19359165212458496`
- PoseNet: `0.00003300`
- SegNet: `0.00056719`
- archive bytes: `178,276`
- archive SHA-256:
  `55d4a4a0d0ad9915e9b74c679ad8ea31e81f4383f60c132bf017e6df40301111`

This is a measured-config regression versus the current A1 contest-CPU anchor
`0.19284757743677347` (`87ec7ca5...492b5`). It is close enough to preserve as
learning signal, but it is not a dispatch-promotion candidate.

## Artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `experiments/results/track1_phase_a1_score_gradient_long_lr1e6_20260509T030424Z_modal/harvest_summary.json` | `521` | `c34d561ae1840bd9a8d376f3e251c8f43201b0156d3eb5bd2eb8de6d7a260f3e` |
| `experiments/results/track1_phase_a1_score_gradient_long_lr1e6_20260509T030424Z_modal/harvested_artifacts/phase_a1_summary.json` | `9,203` | `aaaa9f2d82cba6f60d7e74820e438c28e85e48e679ae84838c1dcd823344e2cb` |
| `experiments/results/track1_phase_a1_score_gradient_long_lr1e6_20260509T030424Z_modal/harvested_artifacts/finetuned_archive/build_manifest.json` | `1,720` | `462bacc52e54d91bb28a7576e46b91de33a891444d4d2768964d549074e778c0` |
| `experiments/results/track1_phase_a1_score_gradient_long_lr1e6_20260509T030424Z_modal/contest_auth_eval.macos_cpu_advisory.json` | `7,709` | `fb59e6140b583a3d15f256bc38809005aa220c46124576de1367ecf5464701d4` |

## Candidate Archive

- Archive:
  `experiments/results/track1_phase_a1_score_gradient_long_lr1e6_20260509T030424Z_modal/harvested_artifacts/finetuned_archive/archive.zip`
- Archive bytes: `178,276`
- Archive SHA-256:
  `55d4a4a0d0ad9915e9b74c679ad8ea31e81f4383f60c132bf017e6df40301111`
- Runtime-tree SHA-256 from local advisory eval:
  `a31392a194a77df4c15ec4b8c39803444b03381be9af781fd6537bdcdb3c2401`
- Checkpoint SHA-256:
  `9baf0724dd502b411666fba2ca5373c42e8c4c1b1677dc57dab9fccef9c06edd`
- Build smoke: `smoke_ok=true`, `max_rel_err=0.0019097625045105815`,
  `mean_rel_err=0.0008886701128046427`, `n_tensors=28`

## Commands

Harvest:

```bash
.venv/bin/python experiments/modal_phase_a1_score_gradient_pr101.py recover \
  --label track1_phase_a1_score_gradient_long_lr1e6_20260509T030424Z_modal
```

Local advisory claim:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id a1_long_lr1e6_macos_cpu_advisory \
  --platform local_macos_cpu \
  --instance-job-id local:a1-long-lr1e6-macos-cpu-20260509T0510Z \
  --agent codex:gpt-5.5 \
  --predicted-eta-utc 2026-05-09T05:23Z \
  --status active_eval \
  --notes "Local macOS CPU advisory eval for harvested A1 long lr1e6 Modal archive after remote CUDA eval failed on temp work-dir guard; non-promotable diagnostic; archive_sha=55d4a4a0d0ad9915e9b74c679ad8ea31e81f4383f60c132bf017e6df40301111"
```

Local advisory eval:

```bash
PYTHON=.venv/bin/python .venv/bin/python -u experiments/contest_auth_eval.py \
  --archive experiments/results/track1_phase_a1_score_gradient_long_lr1e6_20260509T030424Z_modal/harvested_artifacts/finetuned_archive/archive.zip \
  --inflate-sh experiments/results/track1_phase_a1_score_gradient_long_lr1e6_20260509T030424Z_modal/harvested_artifacts/finetuned_archive/submission_dir/inflate.sh \
  --upstream-dir upstream \
  --device cpu \
  --work-dir experiments/results/track1_phase_a1_score_gradient_long_lr1e6_20260509T030424Z_modal/macos_cpu_advisory_work \
  --json-out experiments/results/track1_phase_a1_score_gradient_long_lr1e6_20260509T030424Z_modal/contest_auth_eval.macos_cpu_advisory.json \
  --inflate-timeout 1800 \
  --evaluate-timeout 5400 \
  --keep-work-dir
```

Terminal advisory claim status:
`completed_macos_cpu_advisory_regression`.

## Interpretation

This run improved PoseNet versus the current A1 anchor but lost enough SegNet
and bytes to regress overall. The training log oscillated after roughly epoch
80 and the final archive is `14 B` larger than the current A1 archive. The
next A1 run should not simply be longer at this learning rate. Use the current
A1 anchor unless a new schedule includes stricter early stopping, a lower
SegNet guard, or explicit selection by local advisory score before remote
CUDA/contest-CPU exact eval.

## Reactivation Criteria

- Relaunch A1 only after the Modal temp-workdir fix is committed and pushed.
- Prefer an early-stopped or lower-SegNet-guard schedule over another blind
  long `lr=1e-6` run.
- Promote only with paired exact `[contest-CUDA]` and `[contest-CPU]` evidence
  on the same archive/runtime SHA.
