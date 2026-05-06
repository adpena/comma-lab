# Frontier Roadmap Status

Live-safe operator roadmap. It does not claim scores or dispatch work.

- row_count: `12`
- dirty_path_count: `43`
- dirty_blocked_row_count: `6`
- next_unblocked_keys: `hnerv_wavelet_wr01_apply, hnerv_lowlevel_brotli_repack, telescopic_foveation_field, lapose_motion_atom_allocator, raft_radial_openpilot_pose`

| key | tier | stage | safe | action | dirty blockers | next patch |
|---|---:|---|---|---|---|---|
| `hnerv_wavelet_wr01_apply` | 10 | `needs_lane_claim_and_exact_cuda` | `yes` | `claim_exact_eval_packet_after_static_gate` |  | Harvest/finalize incoming custody hardening, then exact CUDA only after lane claim. |
| `hnerv_lowlevel_brotli_repack` | 20 | `exact_evidence_present_review_before_promotion` | `yes` | `exact_eval_or_promote_measured_rate_candidate` |  | Promote only exact-evaluated archive SHAs; keep PR106 q10 as archive-preflight-ready until lane claim and exact CUDA auth eval. |
| `categorical_qma9_clade_spade_openpilot` | 30 | `blocked_by_dirty_worktree` | `no` | `build_byte_closed_categorical_candidate` | `src/tac/categorical_candidate_readiness.py`<br>`tools/build_categorical_candidate_fixture.py` | Recover PR91/HPM1 full decode/reencode parity or replace the deterministic fixture with the first real byte-closed categorical candidate; resolve the HPAC CPU/CUDA runtime contract and pass the matching readiness audit before any lane claim or exact eval. |
| `joint_admm_balle_arithmetic_stack` | 40 | `blocked_by_dirty_worktree` | `no` | `build_end_to_end_noop_stack_fixture` | `src/tac/arithmetic_qint_codec.py`<br>`src/tac/joint_codec_stack_orchestrator.py` | Replace the fixture-only contract with a byte-closed JCSP archive member and runtime loader parity, then claim a lane before exact CUDA auth eval. |
| `hnerv_per_tensor_context_entropy` | 50 | `blocked_by_dirty_worktree` | `no` | `reduce_entropy_model_overhead` | `src/tac/arithmetic_qint_codec.py`<br>`src/tac/optimization/entropy_codec_gap_audit.py`<br>`tools/audit_entropy_codec_gap.py` | Cluster or codebook-share HDC2 context tables; HDC2 cut PR106x penalty from +96,671B to +51,103B but remains byte-negative. |
| `sensitivity_omega_w_v3` | 60 | `blocked_by_dirty_worktree` | `no` | `replace_stub_sensitivity_with_certified_cuda_artifact` | `src/tac/component_sensitivity_artifact.py`<br>`src/tac/sensitivity_map.py` | Replace all-ones/stub sensitivity producers with certified CUDA/component artifacts. |
| `telescopic_foveation_field` | 70 | `needs_research_or_contract_hardening` | `yes` | `charge_runtime_geometry_consumer_contract` |  | Run charged foveation-params readiness audit, then keep foveation as ranking feedback until a runtime consumer passes geometry preflight and exact component gates. |
| `lapose_motion_atom_allocator` | 80 | `needs_research_or_contract_hardening` | `yes` | `calibrate_planning_signal_and_attach_archive_consumer` |  | Keep labeled as LA-Pose-inspired until a paper-faithful inverse-dynamics encoder and pose head exist; add class/openpilot manifests, calibrate confidence, and require a charged archive consumer before dispatch. |
| `raft_radial_openpilot_pose` | 90 | `needs_research_or_contract_hardening` | `yes` | `emit_pose_disagreement_readiness_artifact` |  | Emit deterministic pose-disagreement and runtime-consumption readiness artifacts. |
| `cmg3_predictive_mask_grammar` | 100 | `needs_research_or_contract_hardening` | `yes` | `close_runtime_decoder_fixture` |  | Close a deterministic runtime decoder and exact archive fixture before ranking. |
| `meta_lagrangian_cross_paradigm_allocator` | 110 | `blocked_by_dirty_worktree` | `no` | `attach_byte_closed_manifest_gate` | `src/tac/optimization/meta_lagrangian_allocator.py` | Add cross-paradigm family/conflict fields and refuse dispatchable rows unless a byte-closed archive manifest is attached. |
| `selfcompress_mdl_fp4_tto` | 120 | `blocked_by_dirty_worktree` | `no` | `prove_deterministic_export_and_inflate_closure` | `src/tac/mdl_bayesian_codec.py`<br>`src/tac/self_compressing_nn.py`<br>`src/tac/tto.py` | Require deterministic export manifest, inflate budget proof, and no scorer load at inflate. |
