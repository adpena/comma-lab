# LA-POSE Foveation Transport Atom Planner - 2026-05-06

## Primary Sources Checked

- LA-Pose, arXiv:2604.27448, submitted 2026-04-30:
  latent-action pretraining for camera pose estimation from driving video.
- Telescope, arXiv:2604.06332, submitted 2026-04-07:
  learnable hyperbolic foveation for long-range autonomous-driving detection.
- Foveated Diffusion, arXiv:2603.23491, submitted 2026-03-24:
  mixed-resolution foveated token allocation for image/video generation.
- Foveated Compression for Immersive Telepresence Visualization,
  arXiv:2510.19848, submitted 2025-10-21:
  spatial QP modulation and foveated bandwidth allocation.
- Geometric Visual Servo Via Optimal Transport, arXiv:2506.02768v2,
  updated 2026-04-01:
  Wasserstein/geodesic transport over SE(3)-like visual-servo pose measures.

## Implementation

Added `tac.analysis.lapose_foveation_atoms` plus
`tools/build_lapose_foveation_atom_manifest.py`.

The planner maps existing LA-POSE-lite/motion records into per-pair
`lapose_foveation_transport_atom` rows. Each row carries an estimated charged
byte budget for a deterministic foveation tuple:

```text
opcode + pair_index + 5 quantized scalars
```

with the default model:

```text
1 byte opcode + 2 byte pair index + 5 * 2 byte scalars = 13 bytes
```

The five scalars are a Telescope-style planning tuple:

```text
alpha, radius, power, origin_x, origin_y
```

The tuple is derived deterministically from the LA-POSE-lite latent proxy and
hard-pair metadata. This is not a runtime warp, archive builder, or score
claim.

## Hardening Status

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- atom rows set `proxy_row=true`
- atom and manifest blockers include:
  - `byte_delta_is_estimated_not_measured_archive_bytes`
  - `requires_foveation_archive_builder`
  - `requires_runtime_consumption_proof`
  - `requires_noop_controls`
  - `requires_exact_cuda_auth_eval`

## Remaining Blockers

- Build a byte-closed archive payload that actually contains the foveation
  tuple bytes.
- Prove `inflate.sh` consumes the payload and changes runtime output in the
  intended path.
- Add no-op controls proving the planner did not reuse stale candidate bytes.
- Run exact CUDA auth eval on the exact archive before any score/rank claim.

## Addendum - LFV1 Local Payload Custody

Added `tac.analysis.lapose_foveation_payload` plus
`tools/build_lapose_foveation_tuple_payload.py`.

The helper lowers selected `lapose_foveation_transport_atom` rows into a
deterministic local `LFV1` binary payload:

```text
header = magic, schema_version, row_count, frame_width, frame_height
row    = opcode, pair_index, q(alpha), q(radius), q(power), q(origin_x), q(origin_y)
```

The row body preserves the prior 13-byte tuple model:

```text
1 byte opcode + 2 byte pair index + 5 * 2 byte quantized scalars
```

The readiness JSON records exact local payload bytes and SHA-256 and sets:

- `score_claim=false`
- `dispatch_attempted=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

This is still planning/local artifact custody, not archive evidence. The
readiness blockers explicitly include:

- `not_archive_consumed_payload`
- `no_runtime_consumer`
- `no_noop_controls`
- `no_exact_cuda_eval`
- `exact_cuda_auth_eval_required_before_score_claim`

Focused verification:

```text
.venv/bin/python -m pytest src/tac/tests/test_lapose_foveation_atoms.py src/tac/tests/test_lapose_foveation_payload.py src/tac/tests/test_cross_paradigm_atoms.py -q
.venv/bin/python -m ruff check src/tac/analysis/lapose_foveation_payload.py tools/build_lapose_foveation_tuple_payload.py src/tac/tests/test_lapose_foveation_payload.py
```
