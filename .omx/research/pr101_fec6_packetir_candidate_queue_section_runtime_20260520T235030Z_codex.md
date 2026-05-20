# PR101/FEC6 PacketIR Candidate Queue

- Schema: `pr101_fec6_packetir_candidate_queue_v2`
- Archive: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip`
- Archive SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- Candidate count: `35`
- Operator candidate count: `29`
- Materialized new archives: `0`
- Score claim: `False`
- Promotion eligible: `False`
- Ready for exact eval dispatch: `False`

## Consumer Surfaces

- `tac.packet_compiler.pr101_frontier_packetir_matrix._candidate_queue_row`
- `tools/build_pr101_frontier_packetir_matrix.py`
- `tac.cathedral_consumers.packetir_candidate_queue_consumer.consume_candidate`

## Candidates

| id | kind | status | dispatch |
|---|---|---|---|
| `fec6_identity_current_archive` | `identity_reference` | `materialized_existing_archive_only` | `False` |
| `fec6_selector_entropy_recode_probe` | `selector_entropy_recode_probe` | `queue_only_not_materialized` | `False` |
| `fec6_wrapper_length_field_elision_probe` | `wrapper_metadata_recode_probe` | `queue_only_not_materialized` | `False` |
| `fec6_source_payload_packetir_recode_probe` | `source_payload_recode_probe` | `queue_only_not_materialized` | `False` |
| `pr101_sidecar_only_runtime_probe` | `section_runtime_visibility_probe` | `queue_only_not_materialized` | `False` |
| `pr101_latent_plus_sidecar_runtime_adapter_probe` | `section_runtime_adapter_probe` | `queue_only_not_materialized` | `False` |
| `fec6_selector_pair_410_frame0_luma_bias_-1_to_none` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_409_frame0_luma_bias_-1_to_none` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_410_frame0_luma_bias_-1_to_frame0_blue_chroma_amp_3` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_409_frame0_luma_bias_-1_to_frame0_blue_chroma_amp_3` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_379_frame0_luma_bias_-1_to_none` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_379_frame0_luma_bias_-1_to_frame0_blue_chroma_amp_3` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_546_frame0_luma_bias_-1_to_none` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_546_frame0_luma_bias_-1_to_frame0_blue_chroma_amp_3` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_502_frame0_luma_bias_-1_to_none` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_502_frame0_luma_bias_-1_to_frame0_blue_chroma_amp_3` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_410_frame0_luma_bias_-1_to_frame0_rgb_bias_p2_m1_m1` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_546_frame0_luma_bias_-1_to_frame0_rgb_bias_p2_m1_m1` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_409_frame0_luma_bias_-1_to_frame0_rgb_bias_p2_m1_m1` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_379_frame0_luma_bias_-1_to_frame0_rgb_bias_p2_m1_m1` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_502_frame0_luma_bias_-1_to_frame0_rgb_bias_p2_m1_m1` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_409_frame0_luma_bias_-1_to_frame0_rgb_bias_m2_p1_p1` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_410_frame0_luma_bias_-1_to_frame0_rgb_bias_m2_p1_p1` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_379_frame0_luma_bias_-1_to_frame0_rgb_bias_m2_p1_p1` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_546_frame0_luma_bias_-1_to_frame0_rgb_bias_m2_p1_p1` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_502_frame0_luma_bias_-1_to_frame0_rgb_bias_m2_p1_p1` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_507_frame0_rgb_bias_p2_m1_m1_to_frame0_luma_bias_-1` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_540_frame0_rgb_bias_p2_m1_m1_to_frame0_luma_bias_-1` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_558_frame0_rgb_bias_p2_m1_m1_to_frame0_luma_bias_-1` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_504_frame0_rgb_bias_p2_m1_m1_to_frame0_luma_bias_-1` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_540_frame0_rgb_bias_p2_m1_m1_to_none` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_504_frame0_rgb_bias_p2_m1_m1_to_none` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_507_frame0_rgb_bias_p2_m1_m1_to_none` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_558_frame0_rgb_bias_p2_m1_m1_to_none` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
| `fec6_selector_pair_504_frame0_rgb_bias_p2_m1_m1_to_frame0_blue_chroma_amp_3` | `grammar_aware_selector_symbol_substitution` | `blocked_until_materialized_runtime_proven_and_paired_exact_eval` | `False` |
