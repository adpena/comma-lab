# ddm_cb2 Next If Resumed

1. On a MAIN host with Metal, run:

```bash
.venv/bin/python tools/repro_cb2_pr130_lift_pose_ci_blind_order.py --run-current
```

Expected after this patch: `current_count=0`, no pytest targets, rc 0.

2. To reproduce the pre-fix broad order without reverting the hook, run:

```bash
.venv/bin/python tools/repro_cb2_pr130_lift_pose_ci_blind_order.py --run-legacy
```

This executes the preserved legacy bare-`pose` selection. If it SIGBUSes, keep the
full fatal traceback and the first target that entered
`adapter.py::_score_aware_loss_part_metrics`.

3. To test the smaller ordered pair first, run:

```bash
.venv/bin/python tools/repro_cb2_pr130_lift_pose_ci_blind_order.py --run-pair
```

4. Before committing, use patch-file serializer mode so unrelated pre-existing
dirty hunks in `tools/preflight_hook.py` are not absorbed into this landing.

5. Re-run:

```bash
.venv/bin/python -m pytest src/tac/tests/test_preflight_hook.py::test_nested_package_init_drops_generic_leaf_token src/tac/tests/test_preflight_hook.py::test_nested_pose_package_init_does_not_select_pose_word_mlx_modules -q
.venv/bin/python -m pytest src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_execute_refuses_bad_archive_section_telemetry_before_training -q
```

No scorer, no n600, no exact eval is owed by this arm.

Own-vehicle frontier line remains:
`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
