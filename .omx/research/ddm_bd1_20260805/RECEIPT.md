# ddm_bd1 Receipt - 2026-08-06

## Status

BD1 charter executed as two scorer-free code units plus this receipt.

- Unit 1 landed in `c478dd1712`: task #970 adds default-on `a1_gate` realized POSE telemetry to `experiments/train_tr1_partition_renderer_mlx.py`.
- Unit 2 landed in `b7056a8ca7`: `--emit-error-atlas` default-off sidecars for `experiments/ddm_jd4_endpoint_n600_both_bases.py`.
- No n600 sweep, no training launch, no archive construction, no upstream exact eval, and no pointer movement were performed.
- Run directories were read-only. No `ddm_tq1` files were edited. No `REVIEW_GATE_OVERRIDE=1` was used.

Own-vehicle frontier line remains:
`S = 0.7537933983374265 @ 357,837 B [macOS-CPU advisory]` from tq1b `move_0012_snap_r00_c02_L13`.
Borrowed contest pointer remains `0.19108 [contest-CPU]` unmoved.

## Recall Evidence

| Evidence | Scope | Impact on BD1 |
| --- | --- | --- |
| `.omx/state/main_hot_state.md` | Live authority before edits | Confirmed #970 and endpoint error-atlas work were queued, CPU/scorer-free trainer edits were in scope, and active run dirs must remain untouched. |
| `.omx/research/ddm_rr1_20260805/RECEIPT.md` | Endpoint vs trainer calibration precedent | Kept Unit 1 labeled advisory trend only: the trainer gate channel uses the MLX frozen adapter path and n600 endpoint probe remains authority. |
| `.omx/research/ddm_mp1_lsb_misplacement_margin_join_20260802.md` | Per-pixel field persistence precedent | Shaped Unit 2 as deterministic packed per-pair fields with manifest shas, so atlas consumers can audit exact sidecars without rerunning the endpoint probe. |
| `.omx/research/arm_final_messages/wp1_20260806T023856Z.md` | Downstream consumer gap | Confirmed the endpoint atlas answers a real queued gap: no TR1 residual/argmax/error atlas was found in that arm's scope. |

## Unit 1 - A1 Gate Realized POSE Channel

Implemented in `experiments/train_tr1_partition_renderer_mlx.py`.

- Added `realized_gate_dpose_fields`, `realized_gate_pose_yuv12`, and `realized_gate_dposes`.
- The pass uses the same `a1_gate` pair set, the same rendered basis already selected by the gate, `_apply_R(render(max(idx-1,0)))`, `_apply_R(render(idx))`, endpoint-style yuv12 packing, and the already-loaded frozen MLX scorer adapter's PoseNet.
- The target is `gt_poses[idx][:6]`; the emitted value is first-6 PoseNet MSE per pair.
- Telemetry keys emitted: `realized_gate_dpose_per_pair`, `realized_gate_dpose_mean`, `realized_gate_dpose_per_pair_max`, `realized_gate_dpose_per_pair_sd`, `realized_gate_dpose_per_pair_q50`, `realized_gate_dpose_per_pair_q90`, `realized_gate_dpose_per_pair_q95`, `realized_gate_dpose_wall_seconds`, `realized_gate_dpose_axis`, `realized_gate_dpose_label`, `realized_gate_dpose_semantics`, and `realized_gate_dpose_gate36_n600_calibration`.
- All new dpose keys are included in `BS3_TELEMETRY_ONLY_KEYS`, so the channel is stripped from checkpoint `telemetry_tail` and remains score-neutral metadata.
- Default-on wiring is the normal `a1_gate` path only. No extra live-basis gate pass was added.
- The channel emits a config row with `score_claim: false` and label `advisory trend channel; n600 endpoint probe remains boundary authority`.

Wall-clock measurement status:

- The code records `realized_gate_dpose_wall_seconds` on every future live `a1_gate`.
- A bounded attempt to measure the real ep1766 gate36 delta in this sandbox failed before model build with `RuntimeError: [metal::load_device] No Metal device available`. This is an environment blocker, not a negative timing result.

Byte-identity proof status:

- A full post-step state hash run would require a live MLX model path blocked here by the same Metal-device error.
- The landed test `test_bd1_realized_gate_dpose_pass_is_structurally_read_only` pins the dpose pass as render plus adapter evaluation only: no RNG, no `value_and_grad`, no optimizer, and no `.update(` mutation in the helper source.

Post-edit sha256:

- `experiments/train_tr1_partition_renderer_mlx.py`: `d0c9b5ef94ac3ddcc315345696114f312969808c53acf837a76e16a74ae50899`
- `src/tac/tests/test_ddm_bs3_gate_projection_kernel.py`: `41dbeaed90959430c2eb57da261d86b4562c505dd17baf851c063109ef039f50`

## Unit 2 - Endpoint Error Atlas Sidecars

Implemented in `experiments/ddm_jd4_endpoint_n600_both_bases.py` and `src/tac/tests/test_ddm_bd1_endpoint_error_atlas.py`.

- Added `--emit-error-atlas`, default off.
- When enabled, the endpoint probe writes one deterministic NPZ sidecar per basis next to `--out`: `<out_stem>.error_atlas.ema.npz` and `<out_stem>.error_atlas.live.npz`.
- Each sidecar stores `error_atlas_packbits`, `pair_ids`, and `field_shape`.
- `error_atlas_packbits` is `np.packbits(realized != lstar, bitorder="big")` over raster order, one row per pair.
- A deterministic JSON manifest `<out_stem>.error_atlas_manifest.json` records path, bytes, sha256, prewrite blob sha256, shape, field shape, bit order, and semantics for each basis.
- The receipt JSON gets `error_atlas_manifest` only when the flag is present. The absent-flag schema is unchanged.
- The output is a diagnostic atlas only and carries `score_claim: false`.

Post-edit sha256:

- `experiments/ddm_jd4_endpoint_n600_both_bases.py`: `adfcf5bf773afc0a1efcb95d20ff082de53c4c73f34dad0f7f6a7ece045c645e`
- `src/tac/tests/test_ddm_bd1_endpoint_error_atlas.py`: `91f3ae7a21c28a9a2692bd398cf8bc7b24e41b516b004b3c81aca78db0d2f6ad`

## Verification

Passed:

- `.venv/bin/python -m py_compile experiments/train_tr1_partition_renderer_mlx.py experiments/ddm_jd4_endpoint_n600_both_bases.py src/tac/tests/test_ddm_bs3_gate_projection_kernel.py src/tac/tests/test_ddm_bd1_endpoint_error_atlas.py`
- `.venv/bin/python -m pytest src/tac/tests/test_ddm_bs3_gate_projection_kernel.py src/tac/tests/test_ddm_bd1_endpoint_error_atlas.py -q`
  - `29 passed`
- `.venv/bin/python -m ruff check experiments/ddm_jd4_endpoint_n600_both_bases.py src/tac/tests/test_ddm_bd1_endpoint_error_atlas.py`
- `git diff --check -- experiments/train_tr1_partition_renderer_mlx.py experiments/ddm_jd4_endpoint_n600_both_bases.py src/tac/tests/test_ddm_bs3_gate_projection_kernel.py src/tac/tests/test_ddm_bd1_endpoint_error_atlas.py`
- `.venv/bin/python tools/review_tracker.py scan`
- `.venv/bin/python tools/review_tracker.py policy-check --file ...` after two explicit review passes for all edited Python files:
  - `experiments/train_tr1_partition_renderer_mlx.py`: 126 compliant, 0 violations
  - `experiments/ddm_jd4_endpoint_n600_both_bases.py`: 10 compliant, 0 violations
  - `src/tac/tests/test_ddm_bs3_gate_projection_kernel.py`: 28 compliant, 0 violations
  - `src/tac/tests/test_ddm_bd1_endpoint_error_atlas.py`: 3 compliant, 0 violations
- Serializer commits:
  - `c478dd1712` with expected-content sha checks for Unit 1 files.
  - `b7056a8ca7` with expected-content sha checks for Unit 2 files.

Blocked or intentionally not run:

- Real gate36 wall-clock delta: blocked by unavailable Metal device before model build in this sandbox.
- Full trainer lint on `experiments/train_tr1_partition_renderer_mlx.py`: existing unrelated style debt remains outside this unit; endpoint and new test lint clean.
- Any n600 endpoint probe, long launch, training run, archive build, or exact eval: out of scope by charter's scorer-free constraint.

## Verdict

BD1 is an instrumentation landing, not goal progress. It creates no exact row and moves no frontier pointer.

- Unit 1 is ready for the next live `a1_gate` to emit advisory d_pose trend fields and wall-clock cost.
- Unit 2 is ready for the next governed endpoint probe to add deterministic error-atlas sidecars by passing `--emit-error-atlas`.
