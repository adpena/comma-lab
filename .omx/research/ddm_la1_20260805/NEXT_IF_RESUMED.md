# ddm_la1 Next If Resumed

## Status

Build complete. No launch or scorer was run.

Current files for this arm:

- `experiments/train_tr1_partition_renderer_mlx.py`
- `src/tac/witness_dsl/spec_tr1_renderer_20260728.py`
- `src/tac/tests/test_ddm_bp1_boundary_reset_race.py`
- `src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py`
- `.omx/research/ddm_la1_20260805/RECEIPT.md`

## Do Next

1. Re-run the focused verification if code changed:
   - `.venv/bin/python -m py_compile experiments/train_tr1_partition_renderer_mlx.py src/tac/witness_dsl/spec_tr1_renderer_20260728.py src/tac/tests/test_ddm_bp1_boundary_reset_race.py src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py`
   - `.venv/bin/python -m pytest -q src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py::test_jd1_lr_anneal_lever_composes_with_joint_pose_finish_and_defaults_derive src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py::test_jd1_lr_anneal_lever_flags_are_declared_by_the_live_trainer src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py::test_jd1_lr_anneal_factory_refuses_inert_or_invalid_shapes src/tac/tests/test_ddm_bp1_boundary_reset_race.py::test_trainer_actually_wires_the_arm_into_its_optimizer src/tac/tests/test_ddm_bp1_boundary_reset_race.py::test_la1_jd1_lr_anneal_flags_fail_closed_when_inert_or_unresumable src/tac/tests/test_ddm_bp1_boundary_reset_race.py::test_la1_jd1_lr_anneal_derives_tail_from_parent_telemetry`
2. Run the two-pass review gate again if any `.py` file changes.
3. Serializer-commit with post-edit SHA256s and tags `[no-triality] [p0-ledger-ok]`.
4. Do not launch an A/B from this handoff unless a lane/slot is explicitly available and claimed.

## Boundary A/B Ticket Shape

At the jd7-or-Case-B boundary, compile two matched TR1 resumes from the same checkpoint:

- OFF arm: omit `--jd1-lr-anneal`.
- ON arm: add `--jd1-lr-anneal derived_tail`; omit `--jd1-lr-final-frac` so the trainer derives
  from the parent telemetry tail.

Keep all other knobs identical, including seed, `batch_pairs`, EMA mode, epoch window, and
scorer endpoint protocol. Measure n600 live and EMA basis endpoint deltas.

Expected diagnostic:

- If LR oscillation is causal, ON reduces live/EMA divergence.
- If ON endpoint EMA is no better and live divergence is unchanged, classify the mechanism as
  not LR-driven and do not continue tuning LR constants.

## Current Derived Values From jd5 Parent

Using parent telemetry
`/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/tr1_jd4_cont_ep1526/telemetry.jsonl`:

- `tail_epochs=60`
- `onset_epoch=1586` for the 1646 boundary
- `final_frac=0.2550714251294281`
- `final_lr=0.0005101428502588562` at base LR `0.002`
- signal source: `epoch.ep_loss[jd1_pose_finish_active]`

The jd6 live directory remains read-only for this arm.
