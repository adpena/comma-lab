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
