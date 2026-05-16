# PR106 PacketIR candidate matrix

Schema: `pr106_packetir_candidate_matrix_v1`

This is an audit artifact only: `score_claim=false`, `promotion_eligible=false`, and CPU/CUDA axes are not converted.

| candidate | format | status | archive bytes | exact axes | blockers |
|---|---:|---|---:|---|---|
| format_0x01_r2_release | `0x01` | `runtime_consumed_needs_paired_exact_eval` | 186822 | contest_cpu:blocked, contest_cuda:blocked | paired_exact_eval_missing:contest_cpu,contest_cuda, contest_cpu:exact_eval_score_formula_mismatch, contest_cuda:exact_eval_score_formula_mismatch |
| format_0x02_pr101_grammar | `0x02` | `runtime_consumed_needs_paired_exact_eval` | 186780 | contest_cpu:blocked, contest_cuda:blocked | paired_exact_eval_missing:contest_cpu,contest_cuda, contest_cpu:exact_eval_score_formula_mismatch, contest_cuda:exact_eval_score_formula_mismatch |
| format_0x04_rank_elided | `0x04` | `single_axis_exact_measured_needs_pair` | 186776 | contest_cuda:valid | paired_exact_eval_missing:contest_cpu |
| format_0x05_fixed_meta | `0x05` | `single_axis_exact_measured_needs_pair` | 186771 | contest_cuda:valid | paired_exact_eval_missing:contest_cpu |
| format_0x05_hdm8_fixed_meta | `0x05` | `single_axis_exact_measured_needs_pair` | 186386 | contest_cuda:valid | paired_exact_eval_missing:contest_cpu |
| format_0x06_implicit_len | `0x06` | `single_axis_exact_measured_needs_pair` | 186382 | contest_cuda:valid | paired_exact_eval_missing:contest_cpu |
| format_0x07_headerless | `0x07` | `single_axis_exact_measured_needs_pair` | 186380 | contest_cuda:valid | paired_exact_eval_missing:contest_cpu |
| format_0x08_inner_headerless | `0x08` | `single_axis_exact_measured_needs_pair` | 186376 | contest_cuda:valid | paired_exact_eval_missing:contest_cpu |
| format_0x09_hdm9 | `0x09` | `single_axis_exact_measured_needs_pair` | 186352 | contest_cuda:valid | paired_exact_eval_missing:contest_cpu |
| format_0x0a_hlm3 | `0x0A` | `single_axis_exact_measured_needs_pair` | 186349 | contest_cuda:valid | paired_exact_eval_missing:contest_cpu |
| format_0x0b_magicless | `0x0B` | `single_axis_exact_measured_needs_pair` | 186341 | contest_cuda:valid | paired_exact_eval_missing:contest_cpu |
| format_0x0c_exact_radix | `0x0C` | `paired_exact_blocked` | 186327 | contest_cpu:valid, contest_cuda:valid | paired_exact_eval_runtime_consumption_content_tree_sha_missing |
| format_0x0d_latent_score_table | `0x0D` | `paired_exact_blocked` | 186876 | contest_cpu:valid, contest_cuda:valid | paired_exact_eval_runtime_content_tree_sha_mismatch_with_consumption |
| prefix_top_1_pr101grammar | `0x02` | `runtime_consumed_needs_paired_exact_eval` | 186258 | - | paired_exact_eval_missing:contest_cpu,contest_cuda |
| prefix_top_4_pr101grammar | `0x02` | `runtime_consumed_needs_paired_exact_eval` | 186263 | - | paired_exact_eval_missing:contest_cpu,contest_cuda |
| prefix_top_16_pr101grammar | `0x02` | `paired_exact_blocked` | 186278 | contest_cpu:valid, contest_cuda:valid | paired_exact_eval_runtime_consumption_content_tree_sha_missing |

## Next exact eval targets

These are fail-fast dispatch targets only. They still require a `tools/claim_lane_dispatch.py` claim and Modal recovery before any score or promotion claim.

| candidate | missing axis | provider | lane | dispatch status | archive | axis blockers |
|---|---|---|---|---|---|---|
| format_0x01_r2_release | `contest_cpu,contest_cuda` | `modal_paired_cpu_cuda` | `pr106_packetir_format_0x01_r2_release` | `requires_claim_lane_dispatch_before_provider_launch` | `submissions/pr106_latent_sidecar_r2/archive.zip` | contest_cpu:exact_eval_score_formula_mismatch, contest_cuda:exact_eval_score_formula_mismatch |
| format_0x02_pr101_grammar | `contest_cpu,contest_cuda` | `modal_paired_cpu_cuda` | `pr106_packetir_format_0x02_pr101_grammar` | `requires_claim_lane_dispatch_before_provider_launch` | `submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip` | contest_cpu:exact_eval_score_formula_mismatch, contest_cuda:exact_eval_score_formula_mismatch |
| format_0x04_rank_elided | `contest_cpu` | `modal_paired_cpu_cuda` | `pr106_packetir_format_0x04_rank_elided` | `requires_claim_lane_dispatch_before_provider_launch` | `experiments/results/pr106_r2_rank_elided_format04_candidate_20260514_codex/pr106_sidecar_rank_elided_format04_candidate.zip` | contest_cpu:exact_eval_artifact_not_listed |
| format_0x05_fixed_meta | `contest_cpu` | `modal_paired_cpu_cuda` | `pr106_packetir_format_0x05_fixed_meta` | `requires_claim_lane_dispatch_before_provider_launch` | `experiments/results/pr106_fixed_meta_rank_elided_20260514_codex/archive.zip` | contest_cpu:exact_eval_artifact_not_listed |
| format_0x05_hdm8_fixed_meta | `contest_cpu` | `modal_paired_cpu_cuda` | `pr106_packetir_format_0x05_hdm8_fixed_meta` | `requires_claim_lane_dispatch_before_provider_launch` | `experiments/results/pr106_hdm8_fixed_meta_rank_elided_20260514_codex/archive.zip` | contest_cpu:exact_eval_artifact_not_listed |
| format_0x06_implicit_len | `contest_cpu` | `modal_paired_cpu_cuda` | `pr106_packetir_format_0x06_implicit_len` | `requires_claim_lane_dispatch_before_provider_launch` | `experiments/results/pr106_hdm8_implicit_len_fixed_meta_20260515_codex/emitted_candidates/pr101_implicit_len_fixed_meta_rank_elided_sidecar_format_0x06.archive.zip` | contest_cpu:exact_eval_artifact_not_listed |
| format_0x07_headerless | `contest_cpu` | `modal_paired_cpu_cuda` | `pr106_packetir_format_0x07_headerless` | `requires_claim_lane_dispatch_before_provider_launch` | `experiments/results/pr106_hdm8_headerless_fmt07_20260515_codex/emitted_candidates/pr101_headerless_implicit_len_fixed_meta_rank_elided_sidecar_format_0x07.archive.zip` | contest_cpu:exact_eval_artifact_not_listed |
| format_0x08_inner_headerless | `contest_cpu` | `modal_paired_cpu_cuda` | `pr106_packetir_format_0x08_inner_headerless` | `requires_claim_lane_dispatch_before_provider_launch` | `experiments/results/pr106_hdm8_inner_headerless_fmt08_20260515_codex/emitted_candidates/pr101_hdm8_hlm2_inner_headerless_fixed_meta_rank_elided_sidecar_format_0x08.archive.zip` | contest_cpu:exact_eval_artifact_not_listed |
| format_0x09_hdm9 | `contest_cpu` | `modal_paired_cpu_cuda` | `pr106_packetir_format_0x09_hdm9` | `requires_claim_lane_dispatch_before_provider_launch` | `experiments/results/pr106_hdm9_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm2_inner_headerless_fixed_meta_rank_elided_sidecar_format_0x09.archive.zip` | contest_cpu:exact_eval_artifact_not_listed |
| format_0x0a_hlm3 | `contest_cpu` | `modal_paired_cpu_cuda` | `pr106_packetir_format_0x0a_hlm3` | `requires_claim_lane_dispatch_before_provider_launch` | `experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm3_inner_headerless_fixed_meta_noop_rank_elided_sidecar_format_0x0a.archive.zip` | contest_cpu:exact_eval_artifact_not_listed |
| format_0x0b_magicless | `contest_cpu` | `modal_paired_cpu_cuda` | `pr106_packetir_format_0x0b_magicless` | `requires_claim_lane_dispatch_before_provider_launch` | `experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_fixed_meta_noop_rank_elided_sidecar_format_0x0b.archive.zip` | contest_cpu:exact_eval_artifact_not_listed |
| prefix_top_1_pr101grammar | `contest_cpu,contest_cuda` | `modal_paired_cpu_cuda` | `pr106_packetir_prefix_top_1_pr101grammar` | `requires_claim_lane_dispatch_before_provider_launch` | `experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_1/archive.zip` | contest_cpu:exact_eval_artifact_not_listed, contest_cuda:exact_eval_artifact_not_listed |
| prefix_top_4_pr101grammar | `contest_cpu,contest_cuda` | `modal_paired_cpu_cuda` | `pr106_packetir_prefix_top_4_pr101grammar` | `requires_claim_lane_dispatch_before_provider_launch` | `experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_4/archive.zip` | contest_cpu:exact_eval_artifact_not_listed, contest_cuda:exact_eval_artifact_not_listed |
