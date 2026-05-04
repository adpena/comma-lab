# PR85 Full-Stack Opportunity Matrix

- planning_only: true
- score_claim: false
- dispatch_performed: false

## Baseline
- PR85 baseline: score 0.25806611029397786 at 236328 bytes from `experiments/results/lightning_batch/exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z/contest_auth_eval.json`.

## Top Stack Plans

| Rank | Family | Bytes at stake | Rate-only bound | Blocked | First gate |
|---:|---|---:|---:|---|---|
| 1 | `qh0_record_level_model_repack` | 57074 | 0.03800323389 | false | reviewed deterministic QH0 serializer |
| 2 | `qma9_native_run_grammar_or_table_reduction` | 159011 | 0.105878897995 | false | full-stream deterministic QMA9-compatible encoder |
| 3 | `pr86_hpac_pr85_mask_contract_port` | 15369 | 0.010233586251 | true | PR86 full decode/reencode byte parity |
| 4 | `protected_randmulti_group_waterfill` | 14592 | 0.009716213844 | false | preserve exact candidate archive custody |
| 5 | `protected_post_motion_group_policy` | 1387 | 0.000923546368 | false | semantic group validation remains passed |

## Opportunity Records

| Family | Surface | Evidence | Stackability | Recommendation |
|---|---|---|---|---|
| `qh0_record_level_model_repack` | PR85 QH0 joint-frame model payload | planning_profile_only_no_runtime_parity | high: changes model segment only and should compose with mask/post/randmulti work after runtime parity | Top byte-saving route when lossless Brotli recode is non-improving: build a deterministic QH0 record-level serializer/repacker and target low-entropy decoded model records under runtime parity. |
| `qma9_native_run_grammar_or_table_reduction` | PR85 QMA9 mask payload | planning_profile_only_mask_bytes_no_candidate_archive | medium: mask-only if decoded-token and runtime parity are preserved | Target native run grammar/table-overhead reductions, not generic symbol/context entropy replacement. |
| `pr86_hpac_pr85_mask_contract_port` | PR85 mask payload via PR86 HPAC contract | fail_closed_planning_only_no_pr86_exact_score_or_pr85_hpac_parity | medium after parity: mask-contract replacement should compose with model and protected sidechannels | Do not dispatch; unblock local HPAC parity gates first. |
| `protected_randmulti_group_waterfill` | PR85 randmulti sidechannel groups | candidate_archive_empirical_plus_whole_stream_exact_negative_guardrail | high with model/mask work; medium with post/motion because scorer interactions are not proven additive | Best current policy is waterfill_top001 with rate-only byte saving; keep as protected group-level candidate, not whole deletion. |
| `protected_post_motion_group_policy` | PR85 post/motion micro sidechannels | candidate_archive_empirical_plus_whole_stream_exact_negative_guardrail | medium: small bytes, high scorer sensitivity, likely stackable only after exact component trace | Best current protected policy is preserve_motion_only; treat as a narrow exact-eval candidate only. |
| `pr89_final_bias_stack_on_pr85` | PR85 plus final-bias sidecar member | empirical_stack_candidate_no_exact_component_benefit | medium: explicit sidecar-style stack, but requires PR89 inflate family and component benefit proof | Do not rank by rate; keep only if exact eval can prove final-bias component gain. |
| `qma9_simple_context_entropy_replacement` | PR85 QMA9 mask payload | empirical_static_profile_refutes_byte_economics | low: refuted as a replacement family unless it becomes a parity control | Already refuted for dispatch: do not build a simple entropy-context replacement as a score candidate. |
| `qma9_block_copy_escape_screens` | PR85 QMA9 block/copy escape variants | empirical_prefix_screen_negative | low until a full-stream screen beats QMA9 and preserves decode parity | Already refuted for the screened variants; revisit only with a structurally different native run grammar. |
| `fixed_runtime_bridge_sparse_action_deletions` | PR85 fixed-runtime bridge qpost/randmulti sparse actions | preflight_blocked_planning_only | low until protected qpost/randmulti deletion blockers have exact evidence override | Already blocked by preflight; do not dispatch sparse-action bridge deletions without exact blocker coverage. |
| `whole_sidechannel_deletion_routes` | PR85 whole post/motion/randmulti sidechannel deletion | exact_cuda_negative_full_600_samples | none: deletion routes are guardrails, not candidate families | Already refuted: do not pursue whole sidechannel deletion as a stack plan. |

## Refuted Or Blocked Routes

- `pr86_hpac_pr85_mask_contract_port`: Do not dispatch; unblock local HPAC parity gates first.
- `pr89_final_bias_stack_on_pr85`: Do not rank by rate; keep only if exact eval can prove final-bias component gain.
- `qma9_simple_context_entropy_replacement`: Already refuted for dispatch: do not build a simple entropy-context replacement as a score candidate.
- `qma9_block_copy_escape_screens`: Already refuted for the screened variants; revisit only with a structurally different native run grammar.
- `fixed_runtime_bridge_sparse_action_deletions`: Already blocked by preflight; do not dispatch sparse-action bridge deletions without exact blocker coverage.
- `whole_sidechannel_deletion_routes`: Already refuted: do not pursue whole sidechannel deletion as a stack plan.
