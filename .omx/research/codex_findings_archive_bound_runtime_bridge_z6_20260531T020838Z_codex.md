# Codex Findings: Shared Archive-Bound Runtime Bridge + Z6 MLX Wiring

UTC: 2026-05-31T02:08:38Z
Agent: Codex
Scope:
- `src/tac/optimization/archive_bound_candidate_runtime_bridge.py`
- `src/tac/substrates/time_traveler_l5_z7_mamba2/archive_candidate.py`
- `src/tac/substrates/z6_v2_cargo_cult_unwind/archive_candidate.py`
- archive-bound candidate contract tag classification and tests

## Verdict

PROCEED. The Z7 MLX archive-bound package path is no longer a one-off
substrate helper, and Z6-v2 MLX now emits the same shared archive-bound
candidate package by default.

## What Landed

- Added a reusable TAC bridge for exporters that already produce
  `archive.zip` and `submission/inflate.sh`.
- Refactored Z7 Mamba2 MLX archive export to use the shared bridge without
  changing the false-authority contract.
- Wired Z6-v2 MLX archive export to emit:
  - byte-closed `archive.zip`;
  - generated-runtime receiver proof;
  - shared archive-bound candidate adapter package;
  - exact-axis blockers;
  - MLX advisory metadata without score authority.
- Fixed a Z6-v2 runtime packaging bug by vendoring the shared inflate runtime
  into generated submissions.
- Split predictive-coding substrate tags so Z6/Z5/Z7 no longer collapse into
  the `z7_mamba2` tag.

## Important Negative

The current tiny Z6-v2 MLX smoke fixture uses `num_pairs=2`. The generated
contest runtime correctly refuses that archive because the receiver contract
requires 600 pairs. The new test preserves this as a fail-closed package:
archive bytes exist, but receiver proof, exact handoff, score claim,
promotion, and rank/kill authority all stay false with explicit blockers.

## Verification

- `git diff --check` on touched scope: pass.
- `ruff` on touched Python files: pass.
- Focused pytest:
  `src/tac/tests/test_archive_bound_candidate_adapter_spine.py`
  `src/tac/tests/test_z7_mamba2_mlx_module_smoke.py::test_z7_mamba2_mlx_canonical_ssd_backend_uses_helper_and_exports_bridge`
  `src/tac/substrates/z6_v2_cargo_cult_unwind/tests/test_z6_v2_mlx_renderer_and_bridge.py`
  result: 18 passed.
- `tools/lane_maturity.py validate`: 1554 lanes validated cleanly.
- `tools/review_gate_hook.py`: pass.
- `tools/review_tracker.py selftest`: pass, with the pre-existing duplicate
  qualified-name warning in `tac.tests.test_dispatch_advisor::_load_advisor`.
- Recursive adversarial review bundle `a7db0db2eff8a58a`: three clean passes,
  sealed on round 3.

## Remaining Work

The same bridge should now absorb the remaining MLX/substrate archive
exporters: Z5, PACT-NeRV selector v2/v3/v4, VQ, public-frontier replay
emitters, PR103/DQS1 byte-shaving outputs, and entropy/archive family
materializers. The invariant should be one candidate contract everywhere:
archive bytes, runtime proof, replay bundle, exact blockers, posterior hooks,
and no duplicated readiness readers.
