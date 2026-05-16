# L5 v2 Exact-Eval Custody Auto-Inflate Hardening - 2026-05-16

## Summary

L5 v2 probe intake was carrying a false CUDA custody blocker:
`l5_v2_probe_axis_inflate_device_missing:contest_cuda`. The TT5L recovered
CUDA exact-eval artifact records `--inflate-device auto` in the auth-eval
command, but the intake only read structured `inflate_device` fields and the
shared exact-eval validator only accepted literal CUDA/GPU tokens for CUDA
inflate policy.

This was an engineering/custody bug, not a substrate result. It could make the
L5 staircase look less complete than the evidence actually supports, while also
muddying the true blocker set.

## Fix

- `src/tac/optimization/l5_v2_probe_intake.py` now recovers `--inflate-device`
  and `--device` from the canonical auth-eval command when structured fields are
  absent.
- `src/tac/exact_eval_custody.py` now accepts `inflate_device=auto` only for
  contest-CUDA inflate policy. CPU-axis evidence still rejects `auto`; eval
  device evidence still requires a real CPU/CUDA token.
- The L5 v2 probe intake, gate artifact, and Markdown report were regenerated.

## Evidence

Regenerated artifact hashes:

- `.omx/research/l5_v2_probe_observation_intake_20260516_codex.json`
  `87a39f6e086a455caad91c2e4b89e1d16bf15abdd2de34ac3683e64c6ac3ada5`
- `.omx/research/l5_v2_probe_observation_intake_20260516_codex.md`
  `c0f5ec3888c5f36d8f27473c2efffa9e0b2ad9e2b3d8359daf2b885df628ab47`
- `.omx/research/l5_v2_probe_gate_artifact_20260516_codex.json`
  `c2225dc6a62e841ef1785bdecc2358daeb8c14bf176a4fd6563aedc9ad2a35e0`

Focused verification:

- `.venv/bin/python -m ruff check src/tac/exact_eval_custody.py src/tac/optimization/l5_v2_probe_intake.py src/tac/tests/test_exact_eval_custody.py src/tac/tests/test_l5_v2_probe_intake.py`
- `.venv/bin/python -m pytest src/tac/tests/test_exact_eval_custody.py src/tac/tests/test_l5_v2_probe_intake.py -q`
  - result: `21 passed`

## Current L5 v2 Status After Fix

The TT5L CUDA axis now carries:

- `inflate_device=auto`
- `eval_device=cuda`
- no `l5_v2_probe_axis_inflate_device_missing:contest_cuda`
- no `l5_v2_probe_axis_inflate_device_not_cuda:contest_cuda`

Architecture lock remains correctly blocked. Remaining TT5L blockers:

- `l5_v2_probe_predicate_failed`
- `l5_v2_probe_paired_exact_axes_missing`
- `l5_v2_probe_sideinfo_consumption_missing`
- `l5_v2_probe_runtime_tree_sha_by_axis_invalid:contest_cpu`
- `l5_v2_probe_contest_evidence_grade_missing`
- `l5_v2_probe_axis_evidence_missing:contest_cpu`
- `l5_v2_probe_axis_log_path_missing:contest_cuda`
- `l5_v2_probe_axis_score_delta_missing:contest_cuda`

This means the next material L5-v2 work should bind the paired CPU axis and
durable CUDA log/score-delta custody, not chase the now-cleared inflate-device
false blocker.

## Authority

This memo is not a score claim, promotion claim, rank/kill claim, or dispatch
authorization. It is an evidence-path hardening note for the L5 v2 staircase.
