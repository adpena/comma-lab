# PR85 Full-Stack Opportunity Matrix

- planning_only: true
- score_claim: false
- dispatch_performed: false

## Baseline
- PR85 baseline: score 0.25806611029397786 at 236328 bytes from `experiments/results/lightning_batch/exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z/contest_auth_eval.json`.

## Top Stack Plans

| Rank | Family | Bytes at stake | Rate-only bound | Blocked | First gate |
|---:|---|---:|---:|---|---|
| 1 | `qma9_native_run_grammar_or_table_reduction` | 159011 | 0.105878897995 | false | full-stream deterministic QMA9-compatible encoder |
| 2 | `pr86_hpac_pr85_mask_contract_port` | 15369 | 0.010233586251 | true | PR86 full decode/reencode byte parity |
| 3 | `scorer_gradient_pair_atom_policy` | 517 | 0.000344249079 | true | explicit pair-action spec |
| 4 | `pr89_final_bias_stack_on_pr85` | 380 | 0.000253026402 | true | public inflate runtime compatibility |

## Opportunity Records

| Family | Surface | Evidence | Stackability | Recommendation |
|---|---|---|---|---|
| `qma9_native_run_grammar_or_table_reduction` | PR85 QMA9 mask payload | planning_profile_only_mask_bytes_no_candidate_archive | medium: mask-only if decoded-token and runtime parity are preserved | Target native run grammar/table-overhead reductions, not generic symbol/context entropy replacement. |
| `pr86_hpac_pr85_mask_contract_port` | PR85 mask payload via PR86 HPAC contract | fail_closed_planning_only_no_pr86_exact_score_or_pr85_hpac_parity | medium after parity: mask-contract replacement should compose with model and protected sidechannels | Do not dispatch; unblock local HPAC parity gates first. |
| `scorer_gradient_pair_atom_policy` | PR85 pair-index scorer-gradient correction atoms | fail_closed_pair_action_lowered_missing_grounded_action_evidence | medium after contract: pair-local corrections should compose only after exact stacked eval | Pair-action lowering exists but remains blocked; generate grounded stream/value action evidence with a non-noop archive-changing path before dispatch. |
| `pr89_final_bias_stack_on_pr85` | PR85 plus final-bias sidecar member | empirical_stack_candidate_no_exact_component_benefit | medium: explicit sidecar-style stack, but requires PR89 inflate family and component benefit proof | Do not rank by rate; keep only if exact eval can prove final-bias component gain. |
| `qh0_record_level_model_repack` | PR85 QH0 joint-frame model payload | empirical_serializer_screen_no_real_byte_win | high: changes model segment only and should compose with mask/post/randmulti work after runtime parity | Measured deterministic QH0/QM0 serializer recodes are neutral or byte-negative; pursue only representation-changing model compression, not wrapper recode. |
| `protected_randmulti_group_waterfill` | PR85 randmulti sidechannel groups | exact_cuda_negative_full_600_samples | high with model/mask work; medium with post/motion because scorer interactions are not proven additive | Measured randmulti group-deletion/waterfill configs are exact negatives; pursue only decoded-output-preserving recode or new component-response microatoms. |
| `protected_post_motion_group_policy` | PR85 post/motion micro sidechannels | exact_cuda_negative_full_600_samples | medium: small bytes, high scorer sensitivity, likely stackable only after exact component trace | Current best post/motion policy is covered by exact-negative deletion evidence; do not dispatch this candidate family until a new policy preserves the measured sensitive groups. |
| `qma9_simple_context_entropy_replacement` | PR85 QMA9 mask payload | empirical_static_profile_refutes_byte_economics | low: refuted as a replacement family unless it becomes a parity control | Already refuted for dispatch: do not build a simple entropy-context replacement as a score candidate. |
| `qma9_native_runtime_supported_grammar_screen` | PR85 QMA9 runtime-supported grammar reductions | empirical_runtime_supported_screen_no_byte_win | medium only after decoded-token parity and exact stacked eval | Measured runtime-supported QMA9 grammar trims are closed; next QMA9 work must implement an alternate grammar/table reduction, not a suffix trim. |
| `qma9_alternate_neighbor_table_grammar_screen` | PR85 QMA9 alternate neighbor/table grammars | empirical_alt_grammar_full_stream_no_byte_win | medium only if a future byte-positive grammar has token parity and an explicit charged runtime mode | Do not redispatch screened alternate neighbor/table modes; push only a structurally different run grammar or HPAC-style model. |
| `decoded_parity_correction_stream_recode` | PR85 post/motion/randmulti/bias/region decoded-parity recodes | empirical_decoded_parity_recode_no_byte_win | high if byte-positive because decoded semantics are preserved | Measured runtime-supported decoded-parity recodes are byte-neutral; pursue grammar-changing sidechannel compression or component-benefit atoms. |
| `qma9_qrg1_row_run_grammar_screen` | PR85 QMA9 row-run grammar replacement | empirical_qrg1_full_stream_no_byte_win | low after current screen; only future model-based or radically different grammars should revisit this surface | Do not pursue raw row-run grammar as a score lane; PR85 QMA9 is already far smaller. Shift QMA9 work to learned/HPAC-style entropy models. |
| `pr86_hpac_probability_contract_variants` | PR86 HPAC submitted-token probability model | fail_closed_probability_variants_no_full_decode | medium after full PR86 decode/reencode parity; currently blocks HPAC transfer | Do not dispatch or transfer HPAC from probability variants; next HPAC work must recover the missing full-stream contract, not tune scalar dtype/perfect flags. |
| `qma9_block_copy_escape_screens` | PR85 QMA9 block/copy escape variants | empirical_prefix_screen_negative | low until a full-stream screen beats QMA9 and preserves decode parity | Already refuted for the screened variants; revisit only with a structurally different native run grammar. |
| `fixed_runtime_bridge_sparse_action_deletions` | PR85 fixed-runtime bridge qpost/randmulti sparse actions | preflight_blocked_planning_only | low until protected qpost/randmulti deletion blockers have exact evidence override | Already blocked by preflight; do not dispatch sparse-action bridge deletions without exact blocker coverage. |
| `whole_sidechannel_deletion_routes` | PR85 whole post/motion/randmulti sidechannel deletion | exact_cuda_negative_full_600_samples | none: deletion routes are guardrails, not candidate families | Already refuted: do not pursue whole sidechannel deletion as a stack plan. |

## Refuted Or Blocked Routes

- `pr86_hpac_pr85_mask_contract_port`: Do not dispatch; unblock local HPAC parity gates first.
- `scorer_gradient_pair_atom_policy`: Pair-action lowering exists but remains blocked; generate grounded stream/value action evidence with a non-noop archive-changing path before dispatch.
- `pr89_final_bias_stack_on_pr85`: Do not rank by rate; keep only if exact eval can prove final-bias component gain.
- `qh0_record_level_model_repack`: Measured deterministic QH0/QM0 serializer recodes are neutral or byte-negative; pursue only representation-changing model compression, not wrapper recode.
- `protected_randmulti_group_waterfill`: Measured randmulti group-deletion/waterfill configs are exact negatives; pursue only decoded-output-preserving recode or new component-response microatoms.
- `protected_post_motion_group_policy`: Current best post/motion policy is covered by exact-negative deletion evidence; do not dispatch this candidate family until a new policy preserves the measured sensitive groups.
- `qma9_simple_context_entropy_replacement`: Already refuted for dispatch: do not build a simple entropy-context replacement as a score candidate.
- `qma9_native_runtime_supported_grammar_screen`: Measured runtime-supported QMA9 grammar trims are closed; next QMA9 work must implement an alternate grammar/table reduction, not a suffix trim.
- `qma9_alternate_neighbor_table_grammar_screen`: Do not redispatch screened alternate neighbor/table modes; push only a structurally different run grammar or HPAC-style model.
- `decoded_parity_correction_stream_recode`: Measured runtime-supported decoded-parity recodes are byte-neutral; pursue grammar-changing sidechannel compression or component-benefit atoms.
- `qma9_qrg1_row_run_grammar_screen`: Do not pursue raw row-run grammar as a score lane; PR85 QMA9 is already far smaller. Shift QMA9 work to learned/HPAC-style entropy models.
- `pr86_hpac_probability_contract_variants`: Do not dispatch or transfer HPAC from probability variants; next HPAC work must recover the missing full-stream contract, not tune scalar dtype/perfect flags.
- `qma9_block_copy_escape_screens`: Already refuted for the screened variants; revisit only with a structurally different native run grammar.
- `fixed_runtime_bridge_sparse_action_deletions`: Already blocked by preflight; do not dispatch sparse-action bridge deletions without exact blocker coverage.
- `whole_sidechannel_deletion_routes`: Already refuted: do not pursue whole sidechannel deletion as a stack plan.
