# JD1 Receipt - TR1 Joint Pose-Finish Build

JD1 built the missing TR1 joint pose-finish surface and queued a recursive
pass ticket. It did not launch training, did not run the scorer, did not create
an archive, and does not move any pointer.

## Build Inventory

- `experiments/train_tr1_partition_renderer_mlx.py`
  - Added default-off JD1 args for joint pose-finish, engagement timing, pose
    weight, pose epsilon, and seg-hold floor/hinge control.
  - Added fail-closed validation so JD1 value flags cannot be declared but
    unread while the mode is off.
  - Loads `gt_poses` only when JD1 is armed, passes the real PoseNet loss path
    through `make_loss_fn(..., compute_pose=True)`, and keeps the default TR1
    path byte/config compatible with `compute_pose=False`.
  - Latches a seg-hold floor at the pose boundary, saves
    `checkpoints/stage_joint_pose_finish_entry.npz` before pose-updated steps,
    carries JD1 runtime metadata in checkpoints, and refuses unsafe resumes.
- `src/tac/witness_dsl/spec_tr1_renderer_20260728.py`
  - Added `lever_jd1_joint_pose_finish(...)` so tickets can emit the live JD1
    trainer flags through the DSL instead of inventing out-of-band launch
    arguments.
- `src/tac/tests/test_ddm_tb1_tr1_renderer.py`
  - Added focused JD1 trainer tests for default-off behavior, flag refusal,
    engagement predicate, checkpoint-tail seg floor resolution, and checkpoint
    metadata round-trip.
- `src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py`
  - Added focused DSL tests that compile the JD1 lever, validate it against the
    live trainer parser, and reject inert or malformed lever shapes.
- `.omx/research/ddm_jd1_20260805/JD1_TICKET.json`
  - Queued the child pass from the TP1 `full_birth_lane_on` final checkpoint.
  - `launch_now=false` because TP1 owns the machine and JD1 owns no scorer slot.

## Ticket

- Parent ticket: `.omx/research/ddm_tp1_20260805/tickets/full_birth_lane_on.json`
- Parent sealed ticket hash:
  `028a11c25ad3423c8cc167209efe01116b0d5fed796dccbd42995b65bbf0df5a`
- Parent file SHA-256:
  `b38c795434b70420d9b92d920702f2eb3b566f9cd8c1aa7ecb7a43c40625e962`
- JD1 ticket hash:
  `6564914a7d090bcde2b46a07ebf8aea529725ecb742d34bbd4a1cf48903a865b`
- JD1 ticket file SHA-256:
  `bbc4a570e0890ba385fd6f390cabf15872a547e1770c373cb9ced46b5efeba16`
- Child output dir:
  `/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/tr1_joint_pose_finish_after_tp1_lane_on`
- Child resume source:
  `/Volumes/VertigoDataTier/pact/ddm_tp1_20260805/full_birth_lane_on/checkpoints/stage_seg_trunk_tau_final.npz`

## Memory Preflight

Axis: DERIVED projection, not measured current JD1 RSS.

- Inherited `batch_pairs=8`: REFUSED. Projected peak 108.07 GiB exceeds the
  98.6 GiB safe ceiling, rc=3.
- Accepted `batch_pairs=4`: SAFE. Projected peak 84.95 GiB is within the
  98.6 GiB safe ceiling, rc=0.
- The child ticket pins `--ema-decay 0.9999436222692036`, derived from the TP1
  parent run geometry, so the smaller safe batch size does not silently change
  the EMA controller at the resume boundary.

## Verification

- `py_compile` passed for the changed Python files.
- `.venv/bin/python -m pytest src/tac/tests/test_ddm_tb1_tr1_renderer.py -k jd1 -q`
  passed: 5 passed, 36 deselected. The process emitted the known no-Metal
  atexit warning after the test result.
- `.venv/bin/python -m pytest src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py -q`
  passed: 3 passed.
- JD1 ticket parse and `validate_jd1_pose_finish_args(...)` passed; the sealed
  hash recalculated to the stored ticket hash.
- Full `src/tac/tests/test_ddm_tb1_tr1_renderer.py -q` was attempted and is not
  usable in this sandbox as a full signal: the existing MLX-heavy tests fail on
  `RuntimeError: [metal::load_device] No Metal device available`.
- `review_tracker.py scan` ran, followed by two explicit `mark-file` passes for
  the changed Python surfaces.

## Recall Evidence

- Read the JD1 charter and common contract, plus `PROGRAM.md`, `CLAUDE.md`,
  `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, and
  `.omx/state/main_hot_state.md`.
- Recalled `.omx/research/ddm_seg_bank_routing_20260805.md` for the TR1
  SEG-SOLVE/TRAIN -> CONSTRAIN -> JOINT DESCENT window and existing recursive
  pass-loop precedent.
- Recalled the TP1 parent ticket as the producer JD1 must wait on.
- Searched memory for JD1/common-contract/#899/#904 context; no direct JD1
  prior was found. The relevant reusable memory was the Pact commit/review
  discipline and double review-tracker practice.

## NEXT_IF_RESUMED

1. Wait for TP1 `full_birth_lane_on` to produce
   `/Volumes/VertigoDataTier/pact/ddm_tp1_20260805/full_birth_lane_on/checkpoints/stage_seg_trunk_tau_final.npz`.
2. Launch `.omx/research/ddm_jd1_20260805/JD1_TICKET.json` only after TP1 is
   done and the machine/scorer ownership rules allow it.
3. The launch must resume the TP1 checkpoint, latch the parent checkpoint-tail
   `ep_loss` as the seg-hold floor, save
   `stage_joint_pose_finish_entry.npz`, then run the JD1 joint pose-finish pass.
4. Queue scorer/exact replay outside JD1 after the pass. Recurse only on a
   typed positive continuation reason and a non-violated seg-hold floor.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
