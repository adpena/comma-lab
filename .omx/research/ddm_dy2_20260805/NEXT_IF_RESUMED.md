# ddm_dy2 NEXT_IF_RESUMED

State as of 2026-08-05T22:16:26Z:

- Plateau-tail JD1 EMA mode is implemented in the trainer.
- DSL lever `lever_jd1_plateau_tail_average_ema(anchor_epoch=...)` is implemented.
- Focused tests pass: 18 passed.
- Full boundary-reset file has 6 Metal-device failures unrelated to dy2 logic in this sandbox.

If resuming before commit:

1. Re-run focused verification:
   - `.venv/bin/python -m py_compile experiments/train_tr1_partition_renderer_mlx.py src/tac/witness_dsl/spec_tr1_renderer_20260728.py src/tac/tests/test_ddm_dy2_jd1_tail_average_ema.py src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py src/tac/tests/test_ddm_bp1_boundary_reset_race.py`
   - `.venv/bin/python -m pytest src/tac/tests/test_ddm_dy2_jd1_tail_average_ema.py src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py src/tac/tests/test_ddm_bp1_boundary_reset_race.py::test_gate_basis_differs_between_fresh_and_resumed_runs`
2. Run two review-tracker passes on each changed Python file.
3. Recompute SHA-256 for the commit files.
4. Commit via `tools/subagent_commit_serializer.py` with explicit `--files` and post-edit `--expected-content-sha256`.

If resuming after commit:

1. Do not launch a scorer from dy2.
2. MAIN should select a concrete anchor epoch from the tp1 Case-0 plateau detector at the jd4 boundary.
3. Compile the A/B through DSL:
   - `lever_jd1_joint_pose_finish(...)`
   - `lever_jd1_plateau_tail_average_ema(anchor_epoch=<tp1_case0_epoch>)`
4. On a Metal-capable host, run the true JD1 tail-state `save_checkpoint` -> `load_checkpoint` resume round-trip before launch.
5. After dy1 scope-law merge, register the tail-average law under `T3_LIVE_ADAPTED`.

Frontier line: S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
