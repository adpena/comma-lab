# Codex Findings: HiNeRV Real Teacher Upstream Custody

UTC: 2026-06-01T21:51:00Z
Author: Codex
Axis: `[macOS-MLX research-signal]`
Score authority: false

## Finding

The HiNeRV execute path accepted `--upstream-dir` but the internal teacher
builder path was still effectively tied to `repo_root / "upstream"`. That broke
clean SSD worktrees where source lives on the SSD and the pinned contest
upstream snapshot lives at `/Users/adpena/Projects/pact/upstream`.

The runner now resolves the scorer upstream directory once, passes that exact
path into both real teacher builders, and records the upstream snapshot hashes
in both failure reports and successful score-aware training metadata.

## Proof Artifacts

Real one-pair HiNeRV smoke with both real teachers:

```text
/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_real_teacher_smoke_20260601T215014Z/compact_renderer_mlx_spine_runner_report.json
```

Result: byte-closed archive exported, receiver proof emitted, SegNet/PoseNet
teacher blocker cleared. Remaining blockers were partial local replay coverage,
short PR95 curriculum, missing full-video MLX scorer replay, no full-coverage
candidate, and no exact CPU/CUDA auth eval.

Real one-pair/eight-epoch HiNeRV smoke with both real teachers and PR95
curriculum gate:

```text
/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_real_teacher_pr95curriculum_smoke_20260601T215029Z/compact_renderer_mlx_spine_runner_report.json
```

Result: real teacher blocker and PR95-curriculum blocker both cleared. Remaining
blockers are now the right next score-lowering blockers: partial local replay
coverage, missing full-video MLX scorer replay, no full-coverage candidate, no
candidate under hard ceiling, and no exact CPU/CUDA auth eval.

## Guard

`src/tac/tests/test_compact_renderer_mlx_spine_runner.py` now asserts the
HiNeRV execute path passes the resolved `scorer_upstream_dir` through to the
trainer, preventing future SSD worktree regressions back to `repo_root/upstream`.
