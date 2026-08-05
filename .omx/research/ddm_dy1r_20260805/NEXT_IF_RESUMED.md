# ddm_dy1r NEXT_IF_RESUMED

State:

- dy1 scope-law resolver has been reconciled onto current main/dy2.
- `jd1_plateau_tail_average_ema_v1` is registered in the canonical equation registry.
- dy2's tail-law pending waiver is resolved in the dy2 receipt and resume note.
- Focused verification passed: 31 tests.
- No scorer, MLX training, launch, or archive work was run.

Before any launch:

1. Re-run the focused tests if these files moved:
   - `src/tac/witness_dsl/tests/test_scope_laws.py`
   - `src/tac/tests/test_ddm_dy2_jd1_tail_average_ema.py`
   - `src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py`
   - `src/tac/tests/test_ddm_jd1_ticket_regenerate.py`
2. At the jd4/tp1 boundary, MAIN selects the explicit Case-0 plateau anchor epoch.
3. Compile through DSL, not hand flags:
   - `lever_jd1_joint_pose_finish(...)`
   - `lever_jd1_plateau_tail_average_ema(anchor_epoch=<case0_epoch>)`
4. On a Metal-capable host, run a true JD1 tail-state
   `save_checkpoint` -> `load_checkpoint` resume round-trip before launch.
5. Treat any declared scope law without a matching runtime `scope_law_resolution`
   row as an inertness failure.

Own-vehicle frontier line: S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
