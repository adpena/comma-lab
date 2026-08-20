# ddm_oc2 — the 77 src/tac sources the review gate still blocks

`date_utc: 2026-08-20` · `owner: ddm_oc2`

These authored modules sit UNCOMMITTED in the main working tree. They are the single largest
remaining custody gap in the repo, and they are blocked for a legitimate reason, not an
oversight.

**Why they did not land.** The review gate scores them at STANDARD policy (the `src/tac/`
prefix) and reports **2,773 UNREVIEWED entities across 121 files** in a single staging attempt.
CLAUDE.md forbids `REVIEW_GATE_OVERRIDE=1` for `.py`, and marking 2,773 entities `reviewed`
without reviewing them would be a fake review claim (NO-FAKE forbidden class 2). The 44 files
under `tools/` and `.omx/research/` from the same batch scored NORMAL and **did** land
(commit `d69d59daaa`); only the `src/tac/` remainder is blocked.

**Evidence that the gate is right to block:** a real review of just 3 comparable files from the
sh1 merge found 1 critical and 5 important correctness bugs, two of them the
tests-verify-constants-not-behaviour pattern. See `REVIEW_FINDINGS_OWED.md`. There is no reason
to expect these 77 to be cleaner.

**Owed to MAIN:** a genuine review pass, in batches, then `review_tracker.py mark-file` and
commit. Do not shortcut it. Until then these files exist only on this machine — they are NOT
on GitHub and NOT on the SSD tier, so a disk loss takes them.

## The blocked set (      77 files)

```
src/tac/boundary_math/dense_raster_lzma_baseline.py
src/tac/optimization/stage_transition_soft_velocity_blend.py
src/tac/optimization/tests/test_stage_transition_soft_velocity_blend.py
src/tac/tests/test_ddm_vh2_vehicle_routing_ledger.py
src/tac/tests/test_materialize_g2_a2_composition_receipt.py
src/tac/tests/test_measurement_integrity.py
src/tac/witness_control/taskspace_candidate_batch_replay_v1.py
src/tac/witness_control/taskspace_feedback_costate_materializer_v1.py
src/tac/witness_control/taskspace_g23_g29_g33_endpoint_adapter.py
src/tac/witness_control/taskspace_interaction_costate_bridge_v1.py
src/tac/witness_control/taskspace_receding_horizon_controller_v1.py
src/tac/witness_control/taskspace_single_stage_score_attempt_v1.py
src/tac/witness_control/tests/test_taskspace_candidate_batch_replay_v1.py
src/tac/witness_control/tests/test_taskspace_feedback_costate_materializer_v1.py
src/tac/witness_control/tests/test_taskspace_g23_g29_g33_endpoint_adapter.py
src/tac/witness_control/tests/test_taskspace_interaction_costate_bridge_v1.py
src/tac/witness_control/tests/test_taskspace_receding_horizon_controller_v1.py
src/tac/witness_control/tests/test_taskspace_single_stage_score_attempt_v1.py
src/tac/witness_control/tests/test_verified_continuation_certificate_v1.py
src/tac/witness_control/verified_continuation_certificate_v1.py
src/tac/witness_dsl/bounded_target_g_encoder.py
src/tac/witness_dsl/coupled_preimage_pair_adapter.py
src/tac/witness_dsl/ep725_levelset_predictor_adapter.py
src/tac/witness_dsl/generative_taskspace_receiver.py
src/tac/witness_dsl/predictor_preserving_coupled_preimage.py
src/tac/witness_dsl/predictor_preserving_taskspace_overlay.py
src/tac/witness_dsl/taskspace_chronological_a3_encoder.py
src/tac/witness_dsl/taskspace_counted_xip2_chronological_a3.py
src/tac/witness_dsl/taskspace_ep725_bridge_eval.py
src/tac/witness_dsl/taskspace_ep725_label_local_g_stream.py
src/tac/witness_dsl/taskspace_g107_g94v2_conditional_y0_final_semanticroot_y1_v1.py
src/tac/witness_dsl/taskspace_g8_a3_interaction_feedback.py
src/tac/witness_dsl/taskspace_lvpg2_public_inverse.py
src/tac/witness_dsl/taskspace_monolithic_pga_receiver.py
src/tac/witness_dsl/taskspace_outer_archive_codec.py
src/tac/witness_dsl/taskspace_pair_fragment_receiver.py
src/tac/witness_dsl/taskspace_pass_conditional_a.py
src/tac/witness_dsl/taskspace_pass_semantic_g.py
src/tac/witness_dsl/taskspace_post_g8_conditional_a.py
src/tac/witness_dsl/taskspace_predictor_state_v2.py
src/tac/witness_dsl/taskspace_predictor_v2_consumer_seam.py
src/tac/witness_dsl/taskspace_quantized_xip2_inverse_solver.py
src/tac/witness_dsl/taskspace_r10_feature_texture_relay.py
src/tac/witness_dsl/taskspace_r10_n600_maximum_inverse_fitter.py
src/tac/witness_dsl/taskspace_same_class_realization_encoder.py
src/tac/witness_dsl/taskspace_same_class_realization_repair.py
src/tac/witness_dsl/taskspace_selective_topology_acquisition.py
src/tac/witness_dsl/taskspace_whole_archive_allocator.py
src/tac/witness_dsl/tests/test_bounded_target_g_encoder.py
src/tac/witness_dsl/tests/test_coupled_preimage_pair_adapter.py
src/tac/witness_dsl/tests/test_ep725_levelset_predictor_adapter.py
src/tac/witness_dsl/tests/test_generative_taskspace_receiver.py
src/tac/witness_dsl/tests/test_predictor_preserving_coupled_preimage.py
src/tac/witness_dsl/tests/test_predictor_preserving_taskspace_overlay.py
src/tac/witness_dsl/tests/test_taskspace_chronological_a3_encoder.py
src/tac/witness_dsl/tests/test_taskspace_counted_xip2_chronological_a3.py
src/tac/witness_dsl/tests/test_taskspace_ep725_bridge_eval.py
src/tac/witness_dsl/tests/test_taskspace_ep725_label_local_g_stream.py
src/tac/witness_dsl/tests/test_taskspace_g107_g94v2_conditional_y0_final_semanticroot_y1_v1.py
src/tac/witness_dsl/tests/test_taskspace_g8_a3_interaction_feedback.py
src/tac/witness_dsl/tests/test_taskspace_monolithic_pga_receiver.py
src/tac/witness_dsl/tests/test_taskspace_outer_archive_codec.py
src/tac/witness_dsl/tests/test_taskspace_pair_fragment_receiver.py
src/tac/witness_dsl/tests/test_taskspace_pass_conditional_a.py
src/tac/witness_dsl/tests/test_taskspace_pass_semantic_g.py
src/tac/witness_dsl/tests/test_taskspace_post_g8_conditional_a.py
src/tac/witness_dsl/tests/test_taskspace_predictor_state_v2.py
src/tac/witness_dsl/tests/test_taskspace_predictor_v2_consumer_seam.py
src/tac/witness_dsl/tests/test_taskspace_quantized_xip2_inverse_solver.py
src/tac/witness_dsl/tests/test_taskspace_r10_feature_texture_relay.py
src/tac/witness_dsl/tests/test_taskspace_r10_n600_maximum_inverse_fitter.py
src/tac/witness_dsl/tests/test_taskspace_same_class_realization_encoder.py
src/tac/witness_dsl/tests/test_taskspace_same_class_realization_repair.py
src/tac/witness_dsl/tests/test_taskspace_selective_topology_acquisition.py
src/tac/witness_dsl/tests/test_taskspace_whole_archive_allocator.py
src/tac/witness_dsl/tests/test_v15_ms1_coordinate_compatibility.py
src/tac/witness_dsl/v15_ms1_coordinate_compatibility.py
```
