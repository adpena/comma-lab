# Codex Findings: F1-as-A2 RGB Invariance Probe

Timestamp: 2026-05-19T02:06:39Z
Actor: codex
Canonical task: `codex_routing_directive_rate_attack_vector_1_f1_hydra_dims_7_12_20260518::PHASE_1_PROBE`
Lane: `lane_rate_attack_f1_as_a2_posenet_rgb_invariance_20260519`

## Verdict

`PROCEED` for the narrow local mechanism claim only:

> small physical RGB perturbations can exist while PoseNet scored dims 0:6 and
> SegNet argmax remain inside the tested trust region.

This is not score authority, not dispatch readiness, and not proof that dims
7:12 are an archive-visible channel.

## Evidence

- Probe tool: `tools/probe_f1_as_a2_posenet_rgb_invariance.py`
- Compatibility wrapper: `tools/probe_hydra_dim_7_12_score_invariance.py`
- Real local evidence: `experiments/results/f1_as_a2_local_probe_20260519T020441Z/real_cpu_pair1_stride256_amp1_corrected_v3.json`
- Real local evidence SHA-256: `6b6a57d98f4c5da6e0824c64ca1c4667c2aa1a530110aaaf03afff767b33b05e`
- Synthetic smoke evidence: `experiments/results/f1_as_a2_local_probe_20260519T020441Z/synthetic.json`
- Synthetic smoke evidence SHA-256: `f05be6716b23d76ad0fa8c47fff65d80837b9f188f596ba1f4c0112ff2f26cb5`
- Superseded probe outcome: `f1_as_a2_rgb_invariance_20260519T020507Z`
- Blocking direct-Hydra outcome: `f1_direct_hydra_dim_channel_blocked_20260519T021330Z`
- Superseded corrected-row terminology outcome:
  `f1_corrected_a2_rgb_invariance_20260519T021330Z`
- Current corrected F1-as-A2 outcome: `f1_as_a2_rgb_invariance_20260519T021450Z`
- Predecessor gate check:
  `tools/check_predecessor_probe_outcome.py --substrate rate_attack_f1_hydra_dims_7_12 --recipe substrate_rate_attack_f1_hydra_dims_7_12_modal_t4_dispatch.yaml`
  now refuses dispatch with the direct-Hydra blocking row above.

Real local CPU probe metrics on `upstream/videos/0.mkv` pair 0, stride 256,
amplitude 1:

- changed RGB values: `110`
- changed RGB values per pair: `110.0`
- recovered payload bits per pair: `0.0`
- `pose_0_5_rmse`: `4.727051377700551e-06`
- `pose_0_5_max_abs`: `1.1444091796875e-05`
- `pose_6_11_rmse`: `5.30738000536199e-06`
- `seg_delta_fraction`: `0.0`
- measurement axis: `[macOS-CPU advisory]`
- hardware substrate: `macos_arm64_cpu`
- baseline RGB SHA-256: `ec1323fb5c6aeb93f46217725a09f118681be0035aebb408fc3b9847cddb4ae2`
- perturbed RGB SHA-256: `570ba85c08b3bdca1bf051c92758da5f0c96b44cdc4d8e1547d3fd5c71098260`
- video SHA-256: `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`

## Authority Boundary

The original F1 claim conflated two facts:

1. PoseNet outputs 12 dims and the scorer uses only dims 0:6. This is a real
   source fact.
2. PoseNet dims 6:12 are therefore free archive bytes. This is false as
   stated because those dims are internal scorer outputs, not bytes emitted by
   `inflate.py`.

The corrected candidate physical channel is RGB perturbation in eventual archive
output. This Phase 1 probe only proves an in-memory RGB tensor perturbation
signal, so the report emits `channel_realization_surface=in_memory_rgb_tensor_probe`
and `candidate_channel_realization_surface=rgb_archive_output_pending_archive_proof`.
The compatibility wrapper keeps the routed filename alive but emits
`f1_framing_version=corrected_A2` and
registers a blocking direct-Hydra `DEFER`. The corrected-A2 local mechanism
signal uses `probe_kind=in_memory_rgb_tensor_invariance_signal`; the previous
`rgb_archive_output_invariance_capacity` wording was superseded before commit.

All reports carry:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `contest_cuda_auth_eval=false`
- `contest_cpu_auth_eval=false`
- `direct_hydra_dim_channel_verdict=DEFER`
- `payload_recovery_authority=false`
- `physical_rgb_payload_capacity=false`

## Blockers Before Any Build Or Dispatch

- no archive bytes rewritten
- no runtime consumption proof
- no full-frame inflate parity
- no exact CUDA auth eval
- no paired Linux x86_64 CPU replay
- no capacity sweep beyond one local pair
- no charged-byte packing grammar proving net rate savings
- no payload decoder or recovery proof; changed RGB values are not payload bits

## Next Engineering Step

Build an archive-contained F1-as-A2 capacity builder only if it produces:

1. a charged-byte perturbation grammar;
2. byte-consumption/no-op proof;
3. full-frame inflate parity against its own runtime;
4. paired CPU/CUDA exact-eval custody;
5. explicit net rate-vs-distortion accounting.

The direct Hydra dim 7:12 side-channel substrate should remain blocked unless
it is re-expressed through a self-contained deterministic packet compiler with
charged bytes and exact eval.
