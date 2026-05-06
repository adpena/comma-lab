# Cross-Paradigm Frontier Action Queue - 2026-05-06

This tranche converts the cross-paradigm inventory from a passive checklist
into a deterministic, non-dispatching action queue. It is an orchestration
artifact only: `score_claim=false`, `dispatch_attempted=false`, and
`ready_for_exact_eval_dispatch=false` remain top-level invariants.

## What Changed

- `tools/build_cross_paradigm_frontier_inventory.py` now emits
  `frontier_action_queue`, `frontier_action_queue_count`,
  `action_class_counts`, and per-row `action_class` / `priority_tier`.
- `tools/all_lanes_preflight.py` verifies that the action queue covers every
  inventory row, keeps all rows non-dispatching, and preserves the current
  first-tranche routing.
- `tools/all_lanes_preflight.py -v` no longer forwards `--verbose` into lane
  tools that do not expose that flag; this fixed the PR106 sidechannel dry-run
  regression in verbose preflight mode.

## Current Queue

1. `hnerv_wavelet_wr01_apply` -> `claim_exact_eval_packet_after_static_gate`
2. `hnerv_lowlevel_brotli_repack` -> `exact_eval_or_promote_measured_rate_candidate`
3. `categorical_qma9_clade_spade_openpilot` -> `build_byte_closed_categorical_candidate`
4. `joint_admm_balle_arithmetic_stack` -> `build_end_to_end_noop_stack_fixture`
5. `hnerv_per_tensor_context_entropy` -> `reduce_entropy_model_overhead`
6. `sensitivity_omega_w_v3` -> `replace_stub_sensitivity_with_certified_cuda_artifact`
7. `telescopic_foveation_field` -> `charge_runtime_geometry_consumer_contract`
8. `lapose_motion_atom_allocator` -> `calibrate_planning_signal_and_attach_archive_consumer`
9. `raft_radial_openpilot_pose` -> `emit_pose_disagreement_readiness_artifact`
10. `cmg3_predictive_mask_grammar` -> `close_runtime_decoder_fixture`
11. `meta_lagrangian_cross_paradigm_allocator` -> `attach_byte_closed_manifest_gate`
12. `selfcompress_mdl_fp4_tto` -> `prove_deterministic_export_and_inflate_closure`

This queue intentionally separates score-lowering direction from dispatch
authorization. The first two rows are closest to existing exact-custody
surfaces. The categorical/openpilot, joint stack, AQ/context entropy,
telescopic foveation, and LA-Pose rows are routed as concrete local-patch
classes, not as score claims.

## Evidence

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_build_cross_paradigm_frontier_inventory.py \
  src/tac/tests/test_all_lanes_pr91_gate.py
```

Result: `12 passed`.

```bash
.venv/bin/python tools/all_lanes_preflight.py --timings -v
```

Result: `ALL 23 PREFLIGHT CHECKS PASSED`.

The preflight timing run also verifies:

- cross-paradigm inventory: `12 rows`, `0 missing code/evidence paths`,
  action queue inventory-only
- PR91/HPM1 remains fail-closed with static custody visible and runtime
  contract blocked
- PR106 sidechannels pass local readiness surfaces under verbose preflight
  without false `--verbose` propagation

## Next

Work the queue in order unless fresh exact evidence changes priority:

1. Claim/submit/harvest the WR01 exact-eval packet only after the dispatch
   claim is made and packet env is complete.
2. Promote the measured PR106x low-level brotli recode only by exact archive
   SHA, and exact-eval any PR106 rebuild before claiming score.
3. Build the first byte-closed categorical/openpilot candidate or finish PR91
   HPM1 full decode/reencode parity; keep CLADE/SPADE/openpilot labels charged
   inside the archive before dispatch.
4. Build a no-op typed joint stack fixture before attempting ADMM/Balle/AQ
   optimization.
