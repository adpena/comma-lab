# Frontier Stack Reconstruction Plan

- planning_only: true
- score_claim: false
- dispatch_performed: false

## Exact Anchor
- PR85 exact anchor: score 0.25806611029397786 at 236328 bytes from `experiments/results/lightning_batch/exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z/contest_auth_eval.adjudicated.json`.

## Ranked Opportunities

| Rank | Opportunity | Bytes saved vs PR85 | Rate delta if neutral | Evidence | Gate |
|---:|---|---:|---:|---|---|
| 1 | `recover_pr91_hpm1_mask_contract_on_pr85_runtime` | 13924 | -0.009271420063 | external_frontier_signal_plus_local_decode_failure | full local HPM1 decode of 600x384x512 tokens |
| 2 | `exact_eval_pr85_stbm1br_lossless_mask_recode` | 6572 | -0.00437602504 | local_candidate_ready_unscored | review stbm1br_preflight.json and manifest SHA against exact archive |
| 3 | `pr90_topband_geometry_mask_prior_for_pr85` | 6580 | -0.004381351912 | local_static_anatomy_plus_external_pr_text | derive PR85 decoded-token geometry policy from PR90, not PR90 byte transplant |
| 4 | `pr91_hpm1_qrgb_atoms_after_hpm1_replay` | None | None | empirical_local_build_blocked_by_pr91_base_decode | PR91 HPM1 base exact replay fixed |
| 5 | `qfq4_model_payload_serializer_probe` | 689 | -0.000458776819 | static_anatomy_low_ceiling | tensor-equivalent PR85 model serialization proof |
| 6 | `pr85_stbm1br_plus_qrgb_randmulti_pair_0192_stack` | 6290 | -0.004188252815 | local_stack_candidate_blocked_by_standalone_positive_gates | STBM standalone exact CUDA positive for exact source SHA |
| 7 | `pr85_qrgb_pair_atoms_negative_guardrail` | None | None | exact_cuda_negative_or_non_improving | new action family outside measured configs |
| 8 | `randmulti_deletion_waterfill_negative_guardrail` | None | None | exact_cuda_negative | new protected atom outside measured deletion basin |
| 9 | `line_search_pose_stack_negative_guardrail` | None | None | exact_cuda_negative_vs_pr85_anchor | only reuse as pair/hard-case profile signal |

## Archive Byte Accounting

| Artifact | Family | Bytes | Member | Member bytes | ZIP overhead | Strict ZIP |
|---|---|---:|---|---:|---:|---|
| `archive_eval_line_search_qzs3_qp1_basis_r13_fix1_20260502T0156Z` | line_search | 276423 | `p` | 276323 | 100 | true |
| `archive_eval_line_search_qzs3_qp1_fixedslice_20260502T0057Z` | line_search | 276427 | `p` | 276327 | 100 | true |
| `archive_eval_line_search_qzs3_qp1_fixedslice_continue_r13_20260502T0123Z` | line_search | 276423 | `p` | 276323 | 100 | true |
| `archive_eval_line_search_qzs3_qp1_fixedslice_continue_r13_20260502T0124Z` | line_search | 276423 | `p` | 276323 | 100 | true |
| `archive_eval_line_search_qzs3_qp1_fixedslice_continue_r3_20260502T0108Z` | line_search | 276426 | `p` | 276326 | 100 | true |
| `archive_eval_line_search_qzs3_qp1_gradient_r13_fix1_20260502T0141Z` | line_search | 276424 | `p` | 276324 | 100 | true |
| `neighbor_combined_top20_a020` | pose_active_subspace | 276383 | `p` | 276283 | 100 | true |
| `neighbor_pose_top12_a025` | pose_active_subspace | 276377 | `p` | 276277 | 100 | true |
| `neighbor_pose_top32_a0125` | pose_active_subspace | 276397 | `p` | 276297 | 100 | true |
| `ref_active_combined_top32_s0125` | pose_active_subspace | 276398 | `p` | 276298 | 100 | true |
| `ref_active_pose_top16_s025` | pose_active_subspace | 276376 | `p` | 276276 | 100 | true |
| `pr85_qrgb_combo_f1_bias_pair_0060__f1_bias_pair_0164` | pr85_qrgb | 236336 | `x` | 236236 | 100 | true |
| `pr85_qrgb_combo_f1_bias_pair_0060__f1_bias_pair_0164__f1_region_pair_0197` | pr85_qrgb | 236337 | `x` | 236237 | 100 | true |
| `pr85_qrgb_combo_f1_bias_pair_0060__f1_region_pair_0197` | pr85_qrgb | 236337 | `x` | 236237 | 100 | true |
| `pr85_qrgb_combo_f1_bias_pair_0164__f1_region_pair_0197` | pr85_qrgb | 236336 | `x` | 236236 | 100 | true |
| `pr85_qrgb_f1_bias_pair_0060` | pr85_qrgb | 236336 | `x` | 236236 | 100 | true |
| `pr85_qrgb_f1_bias_pair_0164` | pr85_qrgb | 236335 | `x` | 236235 | 100 | true |
| `pr85_qrgb_f1_region_pair_0197` | pr85_qrgb | 236335 | `x` | 236235 | 100 | true |
| `pr85_qrgb_f2_randglobal_pair_0192` | pr85_qrgb | 236616 | `x` | 236516 | 100 | true |
| `pr85_stbm1br_mask_recode` | pr85_stbm1br | 229756 | `x` | 229656 | 100 | true |
| `pr85_qrgb_f2_randglobal_pair_0192` | pr85_stbm1br_plus_qrgb | 230038 | `x` | 229938 | 100 | true |
| `pr85_stbm1br_plus_qrgb_f2_randglobal_pair_0192` | pr85_stbm1br_plus_qrgb | 230038 | `x` | 229938 | 100 | true |
| `pr91_hpm1_qrgb_f1_bias_pair_0060` | pr91_qrgb | 222412 | `x` | 222312 | 100 | true |
| `pr91_hpm1_qrgb_f1_bias_pair_0164` | pr91_qrgb | 222411 | `x` | 222311 | 100 | true |
| `pr91_hpm1_qrgb_f1_region_pair_0197` | pr91_qrgb | 222411 | `x` | 222311 | 100 | true |
| `public_pr85_exact_anchor` | public_pr85 | 236328 | `x` | 236228 | 100 | true |
| `public_pr90_qrepro_external` | public_pr90 | 218080 | `p` | 217980 | 100 | true |
| `public_pr91_hpm1_external` | public_pr91 | 222404 | `x` | 222304 | 100 | true |
| `public_pr91_hpm1_external` | public_pr91 | 222404 | `x` | 222304 | 100 | true |
| `c101_renderer_x_top160` | renderer_pose_stack | 275686 | `p` | 275586 | 100 | true |
| `c101_renderer_x_top192` | renderer_pose_stack | 275683 | `p` | 275583 | 100 | true |

## Failure Mode Review

- `recover_pr91_hpm1_mask_contract_on_pr85_runtime`: Highest EV, but only as a local parity/replay repair lane. No dispatch of derivatives until the decode contract is fixed.
- `exact_eval_pr85_stbm1br_lossless_mask_recode`: Promotable next only as standalone exact CUDA eval with lane claim; do not stack until standalone result is positive.
- `pr90_topband_geometry_mask_prior_for_pr85`: Implement only a PR85-token geometry profiler/builder; do not transplant PR90 runtime wholesale.
- `pr91_hpm1_qrgb_atoms_after_hpm1_replay`: Keep built archives as post-HPM1 repair probes, not dispatch candidates today.
- `qfq4_model_payload_serializer_probe`: Low priority: keep as a serializer probe, not a remote eval lane.
- `pr85_stbm1br_plus_qrgb_randmulti_pair_0192_stack`: Hold as a blocked reconstruction candidate. Do not dispatch while QRGB standalone exact evidence is negative.
- `pr85_qrgb_pair_atoms_negative_guardrail`: Retire measured QRGB configs narrowly; keep the action compiler as a negative-signal profiler.
- `randmulti_deletion_waterfill_negative_guardrail`: Do not dispatch measured randmulti deletion/waterfill candidates.
- `line_search_pose_stack_negative_guardrail`: Use only as profile feedback for new pair atoms.
