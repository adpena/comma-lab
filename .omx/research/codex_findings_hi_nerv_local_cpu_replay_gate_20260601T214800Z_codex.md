# Codex Findings: HiNeRV Local CPU Replay Gate

UTC: 2026-06-01T21:48:00Z
Author: Codex
Axis: `[macOS-MLX research-signal]` plus `[macOS-CPU advisory]`
Score authority: false

## Finding

The compact renderer spine runner now has a reusable local CPU replay gate for
HiNeRV byte-closed archive outputs. It reuses `comma_lab.local_submission_replay`
instead of introducing a second evaluator path:

- stages the generated runtime submission plus `archive.zip`;
- runs `upstream/evaluate.sh --device cpu` only for full 600-pair coverage by
  default;
- writes `local_cpu_replay/local_submission_replay_summary.json`;
- preserves false-authority fields;
- certifies and deletes rebuildable inflated/raw scratch through the existing
  cleanup manifest unless explicitly retained.

Partial 1/32/128-pair smokes now refuse replay with the typed blocker
`local_cpu_replay_not_run_partial_pair_coverage`. That is intentional: receiver
proof can validate decode/runtime consumption on partial archives, but upstream
local replay is not promotion-comparable until the archive covers the full
contest video.

## Proof Artifacts

Focused test suite:

```bash
/Users/adpena/Projects/pact/.venv/bin/python -m pytest -q \
  src/tac/substrates/hi_nerv/tests/test_hi_nerv_roundtrip.py \
  src/tac/substrates/hi_nerv/tests/test_hi_nerv_mlx_renderer_and_archive_candidate.py \
  src/tac/tests/test_compact_renderer_mlx_spine_runner.py
```

Result: `30 passed in 3.18s`.

Lint:

```bash
/Users/adpena/Projects/pact/.venv/bin/python -m ruff check \
  tools/run_compact_renderer_mlx_spine_runner.py \
  src/tac/tests/test_compact_renderer_mlx_spine_runner.py
```

Result: `All checks passed!`.

Real one-pair HiNeRV train/export/receiver smoke:

```text
/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_replay_gate_partial_smoke_20260601T214703Z/compact_renderer_mlx_spine_runner_report.json
```

Result: byte-closed archive exported, receiver proof emitted, false-authority,
blocked by `local_cpu_replay_not_run_partial_pair_coverage`,
`hi_nerv_pr95_faithful_curriculum_requires_min_8_epochs`,
`hi_nerv_real_segnet_posenet_teachers_not_both_attached`,
`full_video_mlx_scorer_replay_not_attached`, and exact auth blockers.

Real one-pair/eight-epoch HiNeRV PR95-curriculum smoke:

```text
/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_replay_gate_pr95curriculum_smoke_20260601T214716Z/compact_renderer_mlx_spine_runner_report.json
```

Result: byte-closed archive exported, receiver proof emitted, false-authority,
and the PR95-curriculum blocker is cleared. Remaining blockers are real:
partial replay coverage, missing real SegNet/PoseNet teachers, missing
full-video MLX scorer replay, no full-coverage candidate, and no exact CPU/CUDA
auth eval.

## Next Engineering Blockers

1. Attach real SegNet and PoseNet teachers to the HiNeRV full-main path for a
   two-pair smoke, then bounded 32/128/600-pair campaigns.
2. Run a full 600-pair HiNeRV archive through the new default local CPU replay
   gate after MLX prefiltering clears.
3. Promote only local replay winners to exact contest CPU, then CUDA if CPU
   clears.
4. Implement SNeRV MLX train/export/archive adapter under the same gate rather
   than keeping it as a planner-only carrier.
