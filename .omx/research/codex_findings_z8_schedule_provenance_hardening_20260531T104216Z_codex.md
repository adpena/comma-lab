# Z8 Schedule Provenance Hardening - Codex Findings

Timestamp: 2026-05-31T10:42:16Z

## Verdict

Read-only partner review found two P1 issues in the dirty Z8 MLX route: invalid Gumbel-Softmax tau schedules could fall through to a best-effort epoch hook, and the new tau/cosine knobs were not replay-captured in the training artifact metadata.

This slice hardens both without promoting MLX output beyond `[macOS-MLX research-signal]`.

## Fix

- `_full_main` validates `0 < gumbel_tau_min <= tau_start` before importing MLX.
- `_full_main` validates cosine LR decay with finite `0 < ratio <= 1` and `0 < warmup_epochs < epochs`.
- `Z8HierarchicalPredictiveCoderMLX.set_anneal_schedule` now rejects invalid tau floors and nonpositive epoch budgets immediately.
- `substrate_artifact_metadata` now carries `z8_mlx_schedule_provenance.v1`, including tau enable/start/min/total/expected-final/current-at-bundle-build and cosine LR enable/start/min-ratio/total/expected-final.
- Tests cover invalid tau/cosine fail-closed behavior and schedule metadata presence.

## Verification

- Ruff on the touched Z8 trainer, renderer, and tests: passed.
- Focused pytest for Z8 M12a route and tau anneal: 15 passed.

## Remaining Boundary

The Z8 smoke remains local MLX research signal only. Archive export, inflate consumption, and exact CPU/CUDA replay remain required before score or promotion authority.
