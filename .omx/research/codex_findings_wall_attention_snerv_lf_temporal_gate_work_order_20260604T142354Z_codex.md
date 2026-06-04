# Codex Findings: Wall Attention SNeRV LF Temporal Gate Work Order

written_at_utc: 2026-06-04T14:23:54Z
lane_id: lane_tilde_research_parallax_nerv_intake_20260603
agent: codex
branch_observed: main
primary_scope: executable false-authority work-order rows for future SNeRV LF/TUB temporal-gate work
write_scope: .omx/research/codex_findings_wall_attention_snerv_lf_temporal_gate_work_order_20260604T142354Z_codex.md
implementation_files_touched: false
large_artifacts_created: false
score_claim: false
frontier_score_claim: false
rank_or_kill_eligible: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
research_only: true

## Preflight Status

- Read `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, the prior Tilde/Parallax intake memo, recent SNeRV source-parity and LF-recode memos, `reports/latest.md`, `.omx/state/subagent_progress.jsonl`, and the relevant rate-allocator/LF-recode source surfaces.
- Confirmed worktree is already dirty in shared MLX/SNeRV implementation files and lane registry state; left those files untouched.
- Did not implement Wall Attention, Parallax, LF/TUB gate kernels, trainer changes, queue mutations, dispatch claims, or exact eval.
- This memo is the executable handoff. It is intentionally false-authority and is not a score, promotion, dispatch, or rank/kill surface.

## Current Authority Boundary

- SNeRV official MFU/HFR/TUB numeric/primitive evidence is useful, but full source-forward parity remains false. `src/tac/substrates/snerv_inverse_steg_carrier/official_tub.py` is currently a source-faithful TUB graph-input primitive, not full TUB/source-forward authority.
- SNeRV LF payload remains the rate bottleneck. Existing LF recode paths prove packet/receiver-contract facts, but they still need full archive replay, full-video scorer/component deltas, and paired contest CPU/CUDA evidence before any authority.
- Wall Attention is only a conceptual source for a small Pact-native per-channel temporal gate. Direct Wall/Parallax LLM attention kernel import is rejected.

## Machine-Readable Work Orders

```json
{
  "schema": "codex_wall_attention_snerv_lf_temporal_gate_work_orders.v1",
  "allowed_use": "local_planning_and_future_queue_ingest_only",
  "forbidden_use": "score_claim_rank_or_kill_promotion_or_exact_eval_dispatch",
  "axis_tag": "[planning/control:false-authority]",
  "source_memo": ".omx/research/codex_findings_tilde_research_parallax_nerv_intake_20260603T174229Z_codex.md",
  "work_orders": [
    {
      "work_order_id": "snerv_wall_attention_lf_tub_temporal_gate_receiver_visible_followup",
      "work_order_type": "snerv_lf_tub_temporal_gate_receiver_visible_work_order",
      "target_family": "snerv",
      "priority": 9,
      "classification": "blocked_until_official_parity_or_explicit_side_smoke",
      "planner_action": "build_pact_native_byte_charged_lf_tub_temporal_gate",
      "target_consumers": [
        "snerv_lf_payload_recode",
        "nerv_rate_allocator_bridge",
        "nerv_long_training_campaign_plan",
        "bit_allocator",
        "cathedral_autopilot",
        "continual_learning_posterior"
      ],
      "future_target_files": [
        "src/tac/substrates/snerv_inverse_steg_carrier/official_tub.py",
        "src/tac/substrates/snerv_inverse_steg_carrier/archive.py",
        "src/tac/analysis/snerv_lf_payload_archive_recode.py",
        "src/tac/analysis/nerv_rate_allocator_bridge.py",
        "src/tac/analysis/nerv_long_training_campaign_plan.py",
        "src/tac/analysis/nerv_control_inventory.py",
        "tools/recode_snerv_lf_payload_archive.py"
      ],
      "future_target_tests": [
        "src/tac/substrates/snerv_inverse_steg_carrier/tests/test_official_tub.py",
        "src/tac/tests/test_snerv_lf_payload_archive_recode.py",
        "src/tac/tests/test_nerv_long_training_campaign_plan.py"
      ],
      "source_idea": {
        "wall_attention_use": "per_channel_per_timestep_multiplicative_decay_as_concept_only",
        "direct_wall_kernel_import_allowed": false,
        "direct_parallax_kernel_import_allowed": false,
        "runtime_dependency_import_allowed": false,
        "forbidden_runtime_dependencies": [
          "torch_triton_wall_attention_kernel",
          "flash_linear_attention",
          "parallax_cuda_triton_kernel",
          "cutlass_dsl_decode_stack"
        ]
      },
      "required_entry_gate": {
        "official_parity_path": {
          "accepted_when": "official MFU/HFR/TUB source-forward parity is proven by current audit artifact",
          "required_symbol_or_artifact": "SNERV_OFFICIAL_MFU_HFR_TUB_PARITY_PROOF or equivalent current official-forward-parity artifact",
          "must_remain_false_authority_until_exact_eval": true
        },
        "side_smoke_path": {
          "accepted_when": "explicitly labeled non-source-faithful side smoke",
          "required_label": "non_source_faithful_side_smoke_wall_attention_lf_tub_temporal_gate",
          "allowed_authority": "timing_fit_and_receiver_grammar_research_signal_only",
          "dispatch_allowed": false
        }
      },
      "implementation_contract": {
        "pact_native_only": true,
        "receiver_visible": true,
        "learned_gate_parameters_must_be_serialized": true,
        "learned_gate_bytes_must_be_charged": true,
        "gate_payload_sha256_required": true,
        "no_unpriced_runtime_sidecars": true,
        "no_direct_wall_or_parallax_kernel_import": true,
        "lf_prediction_target": "reduce explicit LF payload bytes by predicting LF/TUB structure from neighboring official TUB lowpass inputs",
        "candidate_packet_must_remain_snar1_receiver_decodable": true
      },
      "required_measurements": {
        "gate_payload_bytes": "required",
        "gate_payload_sha256": "required",
        "source_packet_bytes": "required",
        "candidate_packet_bytes": "required",
        "packet_byte_delta": "required",
        "lf_source_bytes": "required",
        "lf_candidate_bytes": "required",
        "lf_payload_byte_delta": "required",
        "learned_gate_rate_score_delta": "required",
        "total_rate_score_delta": "required",
        "receiver_replay_proof": "required",
        "runtime_consumption_proof": "required",
        "unchanged_non_gate_sections_exact": "required",
        "segnet_delta": "required_with_axis_label",
        "posenet_delta": "required_with_axis_label",
        "lf_reconstruction_error": "required",
        "full_video_coverage": "required_before_queue_admission",
        "paired_contest_cpu_cuda_auth_eval": "required_before_any_promotion"
      },
      "minimum_json_row_fields_for_future_ingest": [
        "schema",
        "work_order_id",
        "axis_tag",
        "candidate_id",
        "source_packet_sha256",
        "candidate_packet_sha256",
        "gate_payload_bytes",
        "gate_payload_sha256",
        "lf_payload_byte_delta",
        "packet_byte_delta",
        "segnet_delta",
        "posenet_delta",
        "receiver_contract_satisfied",
        "runtime_consumption_proof_ready",
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "blockers"
      ],
      "blockers": [
        "snerv_official_mfu_hfr_tub_parity_missing_or_explicit_side_smoke_label_required",
        "snerv_lf_tub_temporal_gate_not_implemented",
        "snerv_lf_tub_temporal_gate_learned_bytes_not_charged",
        "snerv_lf_tub_temporal_gate_receiver_replay_proof_missing",
        "snerv_lf_tub_temporal_gate_lf_byte_delta_missing",
        "snerv_lf_tub_temporal_gate_segnet_posenet_deltas_missing",
        "snerv_wall_attention_direct_kernel_import_forbidden",
        "not_packaged_as_contest_archive_zip",
        "full_video_scorer_replay_missing",
        "paired_contest_cpu_cuda_auth_eval_missing",
        "false_authority_no_score_or_dispatch_authority"
      ],
      "score_claim": false,
      "score_claim_valid": false,
      "frontier_score_claim": false,
      "promotion_eligible": false,
      "rank_or_kill_eligible": false,
      "ready_for_exact_eval_dispatch": false,
      "predicted_delta_adjustment": 0.0
    },
    {
      "work_order_id": "snerv_wall_attention_direct_import_refusal_gate",
      "work_order_type": "direct_kernel_import_refusal_gate",
      "target_family": "snerv",
      "priority": 10,
      "classification": "hard_blocker_for_direct_runtime_import",
      "planner_action": "reject_direct_wall_or_parallax_kernel_import",
      "target_consumers": [
        "nerv_source_parity_contract",
        "nerv_control_inventory",
        "nerv_rate_allocator_bridge"
      ],
      "rationale": "Wall and Parallax public kernels are LLM attention/runtime code, not SNeRV archive grammar, inflate runtime, or video decoder source parity.",
      "forbidden_imports": [
        "tilde-research/wall-attention-release runtime kernels",
        "tilde-research/wall-flash-linear-attention runtime kernels",
        "Yifei-Zuo/Parallax runtime kernels",
        "Yifei-Zuo/FlashLLA runtime kernels"
      ],
      "required_resolution": "translate only the tiny temporal-decay idea into Pact-native byte-charged receiver-visible code after the official-or-side-smoke gate",
      "blockers": [
        "torch_triton_attention_runtime_not_pact_decoder_grammar",
        "no_byte_closed_archive_path",
        "no_receiver_replay_path",
        "no_lf_byte_delta_or_component_delta_measurements",
        "direct_wall_parallax_kernel_import_forbidden"
      ],
      "score_claim": false,
      "score_claim_valid": false,
      "frontier_score_claim": false,
      "promotion_eligible": false,
      "rank_or_kill_eligible": false,
      "ready_for_exact_eval_dispatch": false,
      "predicted_delta_adjustment": 0.0
    },
    {
      "work_order_id": "wire_snerv_wall_attention_lf_tub_gate_into_planner_after_first_proof",
      "work_order_type": "planner_wiring_followup",
      "target_family": "snerv",
      "priority": 7,
      "classification": "blocked_until_first_receiver_visible_gate_artifact",
      "planner_action": "teach_existing_snerv_lf_recode_and_rate_allocator_surfaces_to_ingest_gate_rows",
      "target_consumers": [
        "snerv_lf_payload_recode",
        "nerv_rate_allocator_bridge",
        "nerv_long_training_campaign_plan",
        "nerv_control_inventory"
      ],
      "future_target_files": [
        "src/tac/analysis/snerv_lf_payload_archive_recode.py",
        "src/tac/analysis/nerv_rate_allocator_bridge.py",
        "src/tac/analysis/nerv_long_training_campaign_plan.py",
        "src/tac/analysis/nerv_control_inventory.py"
      ],
      "required_first_artifact_schema": "snerv_lf_tub_temporal_gate_receiver_proof.v1",
      "required_planner_behavior": [
        "emit section-value rows with gate bytes charged separately from LF byte savings",
        "block rows without receiver replay proof",
        "block rows without LF-byte delta",
        "block rows without SegNet and PoseNet deltas",
        "preserve score_claim false and ready_for_exact_eval_dispatch false",
        "route only local planner follow-up until paired contest CPU/CUDA evidence exists"
      ],
      "blockers": [
        "snerv_lf_tub_temporal_gate_first_receiver_visible_artifact_missing",
        "snerv_lf_tub_temporal_gate_planner_schema_missing",
        "snerv_lf_tub_temporal_gate_section_value_rows_missing",
        "paired_contest_cpu_cuda_auth_eval_missing",
        "false_authority_no_score_or_dispatch_authority"
      ],
      "score_claim": false,
      "score_claim_valid": false,
      "frontier_score_claim": false,
      "promotion_eligible": false,
      "rank_or_kill_eligible": false,
      "ready_for_exact_eval_dispatch": false,
      "predicted_delta_adjustment": 0.0
    }
  ]
}
```

## Why This Is Not A Kernel Task

The next implementation should not port Wall Attention or Parallax. The useful idea is only a tiny temporal-retention gate over SNeRV LF/TUB inputs. A correct Pact version must be receiver-visible, byte-charged, and replay-proven; otherwise it repeats the old local-advisory/no-op failure class.

## Top 3 Future Commands And Tests

1. Recheck the official-or-side-smoke entry gate:
   `uv run python tools/audit_snerv_official_source_parity.py --official-repo-dir /Volumes/VertigoDataTier/pact/experiments/results/oss_nerv_source_audit_20260602T113720Z/repos/SNeRV --output-forward-parity-artifact --output-json .omx/research/snerv_official_source_parity_audit_<utc>_wall_gate_entry.json --output-md .omx/research/snerv_official_source_parity_audit_<utc>_wall_gate_entry.md`

2. After a receiver-visible gate artifact exists, price LF and gate bytes through the existing LF recode surface:
   `uv run python tools/recode_snerv_lf_payload_archive.py --packet <source.snar> --mode <lf_or_gate_mode> --output-packet /Volumes/VertigoDataTier/pact/snerv_wall_gate/<utc>/candidate.snar --output-json .omx/research/snerv_lf_tub_temporal_gate_recode_<utc>.json --output-md .omx/research/snerv_lf_tub_temporal_gate_recode_<utc>.md --force-frame-proof`

3. Focused tests for the first planner/code landing:
   `uv run python -m pytest src/tac/substrates/snerv_inverse_steg_carrier/tests/test_official_tub.py src/tac/tests/test_snerv_lf_payload_archive_recode.py src/tac/tests/test_nerv_long_training_campaign_plan.py`

## Bottom Line

The Wall Attention follow-up is now an executable false-authority work order, not an implementation claim. The next worker has a concrete gate: either close official MFU/HFR/TUB parity or label a non-source-faithful side smoke, then charge every learned gate byte and prove receiver replay, LF-byte delta, and SegNet/PoseNet deltas before planner admission.
