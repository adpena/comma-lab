# Codex Findings: HiNeRV 32/128-Pair Scale-Up

UTC: 2026-06-01T21:54:00Z
Author: Codex
Axis: `[macOS-MLX research-signal]`
Score authority: false

## Finding

HiNeRV now clears the real-teacher and PR95-curriculum gates at 32 and 128
pairs with byte-closed archive export plus receiver proof. The campaign is not
promotion-ready: full-video MLX scorer replay, MLX winner filtering, full
coverage, local CPU replay, and exact CPU/CUDA auth remain required.

The local CPU replay policy was tightened after review: full 600-pair coverage
does not automatically trigger local CPU replay unless a full-video MLX prefilter
profile is attached, or the operator explicitly passes `--run-local-cpu-replay`.
This preserves the rule "CPU only for MLX-filtered candidates" and avoids
expensive scorer replay on candidates with no local winner signal.

## Artifacts

32-pair real-teacher/PR95-curriculum run:

```text
/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_32pair_real_teacher_pr95curriculum_20260601T215152Z/compact_renderer_mlx_spine_runner_report.json
```

Summary:

- `archive_bytes`: 41,262
- `archive_sha256`: `e4f0285709ad5cc3691ca1fa69be264827e58cf6a3c6668919b26c4c80fe9d02`
- receiver proof: passed
- raw proof output: 195,328,512 bytes, certified and cleaned
- retained artifact directory size: about 956 KiB

128-pair real-teacher/PR95-curriculum run:

```text
/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_128pair_real_teacher_pr95curriculum_20260601T215234Z/compact_renderer_mlx_spine_runner_report.json
```

Summary:

- `archive_bytes`: 46,556
- `archive_sha256`: `5809b996cf800eefb22a2e623657238c0ba843ac8e5e494af3dd37831830a810`
- receiver proof: passed
- raw proof output: 781,314,048 bytes, certified and cleaned
- retained artifact directory size: about 1.0 MiB

Both runs are still blocked by:

- `local_cpu_replay_not_run_partial_pair_coverage`
- `full_video_mlx_scorer_replay_not_attached`
- `no_full_coverage_compact_base_candidate`
- `no_full_coverage_candidate_under_any_hard_ceiling`
- `contest_cpu_cuda_exact_eval_not_executed`

## Next

The next executable tranche is a 600-pair HiNeRV campaign with a full-video MLX
scorer replay/prefilter profile attached. Only if that profile indicates a true
local winner should the local CPU replay gate run, followed by exact contest CPU
and CUDA auth eval.
