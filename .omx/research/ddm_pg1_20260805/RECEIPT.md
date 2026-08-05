# ddm_pg1 Q3 Seg-Gradient Projector Receipt

Date: 2026-08-05

Axis: build-only, scorer-free. No scorer run, no training launch, no live run dir touched. Own-vehicle frontier remains:

`S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`

Contest pointer remains borrowed/unmoved: `0.1910828242 [contest-CPU]`.

## Build

Built default-off `--seg-grad-q3-project {off,on}` in `experiments/train_tr1_partition_renderer_mlx.py`.

When `on`, frame_1 forward pixels are unchanged. The backward cotangent from the SEG loss is projected blockwise through sq1's canonical frame_1 yuv6 pose-null projector P before reaching the renderer. JD1 pose-finish uses an unwrapped loss/render path when pose is active, so the pose gradient path is not projected.

Added DSL Lever factory `lever_seg_grad_q3_project()` in `src/tac/witness_dsl/spec_tr1_renderer_20260728.py`. The control arm is omission of the lever; explicit inert `off` is refused by the factory.

Resumable state: none beyond the args-only flag. The flag is not part of `TR1Config`, config hash, checkpoint state, or archive grammar. Telemetry row `seg_grad_q3_project_config` records mode, projector SHA/rank, `resumable_state=none_args_only`, `score_claim=False`, and scope-law status.

Scope-law status: `FORMALIZATION_PENDING_NOT_APPLICABLE_BINARY_FLAG`. dy1r's `T3_LIVE_ADAPTED` surface is for dynamic runtime values; pg1 is a binary structural switch. Fire order: if a future dynamic `q3_first` schedule or live scalar threshold is introduced, register a `T3_LIVE_ADAPTED` law before launch.

## Projector Parity

Canonical source: `experiments/ddm_sq1_pose_null_constrained_paint.py::pose_null_projector`.

Measured on synthetic/projector-only fixtures:

- Projector SHA-256: `4fd2121c5a91ee09798aa54bc32fd2425302c59c107efc4bbf143c95d15185fc`
- Max abs difference vs sq1 float32-returned P: `1.2885450928479258e-08`
- Rank: `6`
- Idempotence max abs `|P@P-P|`: `4.163336342344337e-16`
- Kernel residual max abs `|A@P|`: `2.393918396847994e-16`

The float-null property is exact at the training-gradient surface. The m85 integer caveat remains binding: uint8 realization residual is a later measurement, not assumed zero.

## Byte-Identity Proof Method

Off path method implemented in tests: `apply_seg_grad_q3_project(frame, "off")` returns the same object path without wrapping; synthetic forward/backward fixture compares plain `sum(frame*w)` to `sum(apply_seg_grad_q3_project(frame,"off")*w)` for exact equality when MLX arrays are available.

On this host, MLX imports but array allocation attempts Metal and raises `No Metal device available`, so the MLX forward/backward fixture tests are present but skipped. Pure numpy/sq1 projector tests and DSL/argparse tests passed here. The skipped tests must be rerun on an MLX-capable host before any launch.

## Recall Evidence

Commands/scopes searched beyond the charter seeds:

- `rg --files .omx/research | rg 'ddm_bo1|ddm_sq1|ddm_jd1|q3|pose_null|dy1r'`
- `rg --files experiments src/tac tests | rg 'sq1|pose_null|q3|tr1_partition|spec_tr1'`
- `.venv/bin/python tools/list_canonical_equations.py --json | rg 'pose_null|jd1|T3_LIVE_ADAPTED|q3'`
- Governing reads: `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, the pg1 charter, and common contract.

Findings beyond the seed list that changed the build:

- `ddm_dy1r_20260805/RECEIPT.md`: scope-law registration is not appropriate for this static binary flag; recorded a formalization-pending waiver with the future dynamic-fire order instead of inventing a law.
- `ddm_jd1_20260805/JD1_RECEIPT.md`: JD1 pose-finish exists in this trainer, so pg1 must split the active pose case and keep pose gradients unwrapped.
- `ddm_q31_20260804` and `ddm_se2_20260804`: prior Q3 negatives are formulation-scoped solve/prototype measurements, not a kill of this trainer-gradient structural build.
- `pose_null_subspace_is_ac_only_v1`: DC is not in the pose-null subspace; the build uses the sq1 projector and does not assert an integer/uint8 null.

## Tests

Commands run:

- `.venv/bin/python -m py_compile experiments/train_tr1_partition_renderer_mlx.py src/tac/witness_dsl/spec_tr1_renderer_20260728.py src/tac/tests/test_ddm_tb1_tr1_renderer.py src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py`
- `.venv/bin/python -m pytest src/tac/tests/test_ddm_tb1_tr1_renderer.py -k pg1 -q`
- `.venv/bin/python -m pytest src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py -k pg1 -q`
- `git diff --check -- experiments/train_tr1_partition_renderer_mlx.py src/tac/witness_dsl/spec_tr1_renderer_20260728.py src/tac/tests/test_ddm_tb1_tr1_renderer.py src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py`

Results:

- Py compile: pass.
- Trainer pg1 focused tests: `9 passed, 5 skipped, 41 deselected`. The 5 skipped tests are MLX custom-VJP/array-runtime checks skipped because this host cannot allocate MLX arrays without Metal.
- DSL pg1 focused tests: `3 passed, 6 deselected`.
- Total focused pg1 active pass count: 12.
- `git diff --check`: pass.

Clean patch verification:

- Final serializer intent patch applies to a clean `HEAD` archive.
- The same py-compile and focused pg1 tests pass from that clean patched tree when the real checkout's `upstream/` is supplied on `PYTHONPATH` for sq1's `frame_utils` import.

## Serializer Status

Commit was attempted through `tools/subagent_commit_serializer.py` with `--patch-file`, `--no-co-author`, required tags, and post-edit `--expected-content-sha256` values computed from the clean patched tree.

Attempt 1 refused with rc=8 because fx1 had a recent in-flight checkpoint overlapping `experiments/train_tr1_partition_renderer_mlx.py`.

Attempt 2 used the paired-env checkpoint override with rationale: patch-file intent manifest against `HEAD`, expected shas, and no whole-file staging. It then failed before commit with rc=128:

`git apply --cached failed ... error: unable to create temporary file: Operation not permitted ... unable to create backing store for newly created file experiments/train_tr1_partition_renderer_mlx.py`

Real staged index after the failed attempt: empty (`git diff --cached --stat` printed no entries). This is a sandbox/Git-object-write blocker, not a code/test failure.

Exact intended commit patch is persisted at `.omx/research/ddm_pg1_20260805/COMMIT_INTENT.patch`.

## Measurement Not Run

Pre-registered next measurement name: Q3-constrained window A/B at a boundary slot.

Falsifier pair:

- If Q3-constrained seg descent is materially slower than unconstrained under matched boundary conditions, Q4 spend was load-bearing.
- If descent is comparable while pose damage stays at zero/within measured pose-null residual, pg1 is the win path.

Disposition: `QUEUED-WITH-A-FIRE-ORDER`. Fire only from a scorer-owner boundary slot, after rerunning the skipped MLX VJP tests on an MLX-capable host.
