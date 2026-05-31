# Codex Findings: Z8 Entropy-Delta Work Order + P19 Authority Gate

UTC: 2026-05-31T20:58:51Z
Agent: Codex
Axis: `[macOS-CPU advisory]`, non-promotional

## Verdict

The live Z8 rate-attack path is not yet "bulk filling by SegNet class regions" as the byte-spend authority. It is currently an executable P18/P19 coefficient-codec path:

- P18 protects SegNet boundary/argmax-margin sensitivity.
- P19 protects PoseNet-sensitive atoms.
- The RD-waterfill schedule spends codec precision per Z8 detail subband.
- The materializer emits byte-closed Z8HPC1 archive/runtime candidates.

That is a real rate-axis path, but class-region filling remains an upstream/adjacent repair signal until a class-region surface is reduced into the same archive-bound Z8 coefficient allocation contract.

## Changes Landed In This Slice

1. Added `z8_entropy_delta_materializer_work_order.v1`.
   - Consumes a ready entropy-detail RD-waterfill schedule.
   - Requires an existing source `0.bin` by default.
   - Emits the exact materializer command with `--no-mutate-coefficients` and `--entropy-code-quantized-details`.
   - Keeps score authority false and exact dispatch blocked.

2. Wired Z8HPC1 into the shared materializer registry.
   - Unit kind: `z8_hpc1_archive`.
   - Operation family: `z8_detail_entropy_delta`.
   - Target kind: `z8_hpc1_detail_entropy_delta_v1`.
   - Registered as a byte-closed packet-IR unit and quantization-stage operation.

3. Hardened P19 authority.
   - True P19 budget spend now requires six pose axes plus six inverse-variance weights.
   - Three-axis or scalar PoseNet gradients remain proposal/ranking probes only.
   - This prevents incomplete PoseNet proxies from receiving codec budget authority.

4. Fixed exact-axis blocker semantics for receiver-proven Z8 candidates.
   - Receiver proof no longer clears the final blocker.
   - Candidate rows remain blocked on `contest_cpu_cuda_eval_not_executed`.

5. Reduced materializer stdout noise.
   - Default CLI output is now a compact manifest summary.
   - Full manifest output is still available with `--print-full-manifest`.

## Live Artifact Read

Existing durable materialized RD-waterfill candidate:

`.omx/research/z8_full_video_mlx_vjp_live_20260531T181115Z/per_subband_delta_schedule_codex/materialized_rd_waterfill_full600_max_weighted_mse_5e-5/z8_joint_p18_p19_deadzone_manifest.json`

Observed state:

- `candidate_bin_bytes`: 24,475,266
- `archive_zip_bytes`: 24,573,973
- `archive_byte_delta`: -3,930,989
- `archive_rate_ratio`: 0.8616153730930036
- `receiver_proof_executed`: true
- `ready_for_exact_eval_dispatch`: false
- stale generated `exact_axis_blocker`: null

The stale null blocker is fixed for future generated manifests; the existing artifact remains historical and must not be promoted from that field.

Existing local prefilter gate says do not exact-dispatch that candidate:

- `local_replay_required_for_exact_auth`
- `mlx_prefilter_action_not_below_target`
- `exact_cpu_dispatch_recommended`: false
- `exact_cuda_dispatch_recommended`: false

## Focused Verification

```text
RUFF_CACHE_DIR=/tmp/ruff_cache_codex .venv/bin/python -m ruff check \
  src/tac/optimization/byte_shaving_campaign.py \
  src/comma_lab/scheduler/byte_shaving_materializer_registry.py \
  src/tac/tests/test_byte_shaving_campaign_queue.py \
  src/tac/substrates/z8_hierarchical_predictive_coding/full_video_vjp_acquisition.py \
  src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_full_video_vjp_acquisition.py \
  src/tac/substrates/z8_hierarchical_predictive_coding/entropy_delta_schedule.py \
  src/tac/substrates/z8_hierarchical_predictive_coding/joint_coefficient_waterfill.py \
  src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_entropy_delta_schedule.py \
  tools/build_z8_entropy_delta_schedule.py \
  tools/materialize_z8_joint_p18_p19_deadzone_candidate.py

All checks passed.
```

```text
TMPDIR=/tmp .venv/bin/python -m pytest \
  src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_full_video_vjp_acquisition.py \
  src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_entropy_delta_schedule.py \
  src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_joint_coefficient_waterfill.py \
  src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_per_subband_rd_waterfill_solver.py \
  src/tac/tests/test_byte_shaving_campaign_queue.py::test_byte_shaving_materializer_registry_registers_z8_hpc1_entropy_delta \
  src/tac/tests/test_byte_shaving_campaign_queue.py::test_byte_shaving_materializer_registry_exposes_dqs1_and_byte_range_contracts \
  -q

78 passed in 1.83s.
```

## Next Correct Build Step

Make SegNet class-region and boundary filling an authority-bearing allocator input only after it is reduced into the same full-video, archive-fresh Z8 coefficient surface:

1. Emit class/boundary surfaces as typed P18 channels keyed by frame, class, boundary band, region, and pair.
2. Project those channels through the wavelet adjoint / coefficient support map, not only area pooling.
3. Couple them with true six-axis P19 Mahalanobis null subsets.
4. Let the KKT/Dykstra RD-waterfill solver spend bytes over the joint class-boundary-pose-rate surface.
5. Materialize only through the Z8HPC1 archive-bound work order and promote only after receiver proof plus exact CPU/CUDA authority.

