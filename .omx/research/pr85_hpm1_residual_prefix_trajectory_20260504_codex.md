# PR85 -> HPM1 Residual Prefix Trajectory Profiler

Date: 2026-05-04
Owner: codex
Evidence grade: empirical/local planning smoke

## Artifact

- Tool: `experiments/profile_pr85_hpm1_residual_prefix_trajectory.py`
- Tests: `src/tac/tests/test_profile_pr85_hpm1_residual_prefix_trajectory.py`
- Purpose: automate small-prefix PR85 QMA9 decoded-token to PR91 HPM1 residual re-encode byte trajectories without repeated manual CLI calls.
- Safety contract: `planning_only=true`, `score_claim=false`, `dispatch_unlocked=false`, `dispatch_performed=false`, no scorer load, no exact eval, no remote dispatch.

## Local Smoke

Command:

```bash
.venv/bin/python experiments/profile_pr85_hpm1_residual_prefix_trajectory.py --frame-counts 1 --json-out /tmp/pr85_hpm1_prefix_profile.json --md-out /tmp/pr85_hpm1_prefix_profile.md
```

Result:

- Status: passed local smoke.
- Normalized token shape: `N,H,W = 600,384,512`.
- Prefix frames: `1`.
- Raw token prefix bytes: `196608`.
- Candidate HPM1 residual segment bytes: `28695`.
- Elapsed: `61.863s`.
- Flags preserved: `score_claim=false`, `dispatch_unlocked=false`, `dispatch_performed=false`.

This is not score evidence and does not unlock dispatch. The only supported use is local rate-trajectory planning before any archive/runtime parity or exact CUDA gate.
