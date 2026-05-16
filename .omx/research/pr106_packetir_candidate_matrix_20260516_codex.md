# PR106 PacketIR candidate matrix

Schema: `pr106_packetir_candidate_matrix_v1`

This is an audit artifact only: `score_claim=false`, `promotion_eligible=false`, and CPU/CUDA axes are not converted.

| candidate | format | status | archive bytes | exact axes | blockers |
|---|---:|---|---:|---|---|
| format_0x01_r2_release | `0x01` | `paired_exact_measured` | 186822 | contest_cpu:valid, contest_cuda:valid | - |
| format_0x02_pr101_grammar | `0x02` | `paired_exact_measured` | 186780 | contest_cpu:valid, contest_cuda:valid | - |
| format_0x04_rank_elided | `0x04` | `single_axis_exact_measured_needs_pair` | 186776 | contest_cuda:valid | - |
| format_0x05_fixed_meta | `0x05` | `single_axis_exact_measured_needs_pair` | 186771 | contest_cuda:valid | - |
| format_0x05_hdm8_fixed_meta | `0x05` | `single_axis_exact_measured_needs_pair` | 186386 | contest_cuda:valid | - |
| format_0x06_implicit_len | `0x06` | `single_axis_exact_measured_needs_pair` | 186382 | contest_cuda:valid | - |
| format_0x07_headerless | `0x07` | `single_axis_exact_measured_needs_pair` | 186380 | contest_cuda:valid | - |
| format_0x08_inner_headerless | `0x08` | `single_axis_exact_measured_needs_pair` | 186376 | contest_cuda:valid | - |
| format_0x09_hdm9 | `0x09` | `single_axis_exact_measured_needs_pair` | 186352 | contest_cuda:valid | - |
| format_0x0a_hlm3 | `0x0A` | `single_axis_exact_measured_needs_pair` | 186349 | contest_cuda:valid | - |
| format_0x0b_magicless | `0x0B` | `single_axis_exact_measured_needs_pair` | 186341 | contest_cuda:valid | - |
| format_0x0c_exact_radix | `0x0C` | `paired_exact_measured` | 186327 | contest_cpu:valid, contest_cuda:valid | - |
| format_0x0d_latent_score_table | `0x0D` | `paired_exact_measured` | 186876 | contest_cpu:valid, contest_cuda:valid | - |
| prefix_top_1_pr101grammar | `0x02` | `runtime_consumed_needs_paired_exact_eval` | 186258 | - | - |
| prefix_top_4_pr101grammar | `0x02` | `runtime_consumed_needs_paired_exact_eval` | 186263 | - | - |
| prefix_top_16_pr101grammar | `0x02` | `paired_exact_measured` | 186278 | contest_cpu:valid, contest_cuda:valid | - |
