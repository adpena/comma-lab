# PR101/FEC6 frontier PacketIR authority matrix

Schema: `pr101_fec6_frontier_packetir_matrix_v1`

This is an audit artifact only: `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`, and no dispatch commands are emitted.

## Authority

- PR101 PacketIR primitives general: `True`
- FEC6 frontier archive present: `True`
- contest_cpu evidence: `True`
- contest_cuda evidence: `True`
- parser/profile evidence: `True`
- deterministic compiler identity evidence: `False`
- PR106-style PacketIR candidate queue: `False`

## Archive

| path | bytes | sha256 | manifest match |
|---|---:|---|---|
| `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip` | 178517 | `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` | `True` |

## Exact Eval Evidence

| axis | valid | score | archive bytes | artifact | artifact sha | runtime content sha | blockers |
|---|---|---:|---:|---|---|---|---|
| `contest_cpu` | `True` | 0.1920513168811056 | 178517 | `experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/contest_auth_eval.json` | `e82e1b46c61f72e17366e52e64bfa7ccdbbf8058f2363e80d4eb914a9485b6bd` | `6811f28c2116757851b4a6e68a5bdefd7866b4da1867eb13b3c62405de8834df` | - |
| `contest_cuda` | `True` | 0.22621002169349796 | 178517 | `experiments/results/modal_auth_eval/archive_6bae0201fb08/contest_auth_eval.json` | `e7b64d010ad1b68a07d18304bec32869f156ed1f7da105efd25876a969e4a9b8` | `6811f28c2116757851b4a6e68a5bdefd7866b4da1867eb13b3c62405de8834df` | - |

## Parser/Profile Artifacts

| artifact | bytes | sha256 |
|---|---:|---|
| `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/fec3_k16_fec6_k16_full_streaming_parity.json` | 2639 | `67c3f993f62488ed1351a9b437fe9c7d1828d3321827af5ab19cda97d5488e86` |
| `experiments/results/pr101_fec6_byte_escape_profile_20260515_codex/profile.json` | 10916 | `ad458d9f4a50dc498e82021abd664a3ef3af602487beb7f727c0f3cb840618ff` |
| `experiments/results/pr101_fec6_byte_escape_profile_20260515_codex/profile.md` | 1047 | `bf93e6f265b8c4b794ffda535178f3b414f8d89584d055f37f11bbab7bf57a76` |
| `.omx/research/pr101_fec6_wrapper_profile_20260515_codex.md` | 2145 | `1ba39d143681a1ab655fa212877b08f4fb74a6a7b9ed90e0c007e637ce4ea03b` |
| `.omx/research/pr101_fec6_parser_wire_in_20260515_codex.md` | 2978 | `ceb35899f940692ae6067a7530bcfb87ab780938c2e27a3f806692fcf333b395` |
| `.omx/research/pr101_fec6_paired_cpu_cuda_axis_xray_20260515_codex.md` | 4309 | `d475970747a80378b04bbad6ba8aef927116f2a1468bea524807abc685f14342` |
| `experiments/results/xray_paired_cpu_cuda_axis_delta_pr101_fec6_20260515_codex/paired_axis_delta.json` | 6153 | `fbc2a95fcd9611eb17359a7497df2b87e51a0ea0ef889376e8ab8afa3eb6338c` |
| `experiments/results/xray_entropy_pr101_fec6_vs_pr106_packetir_20260515_codex/heatmap.json` | 2079 | `caeeeed7e21e535813359e2068a0f292912f6743101b539aa7996eed5cbebb86` |
| `experiments/results/fec6_selector_operator_space_20260517_codex/operator_space_manifest.json` | 64010 | `af36eee0ebe7d27bb9b8a120c3a901e4800624a0f150d8122aedd7768281af2b` |

## Next Actions

| id | status | non-promotional flags |
|---|---|---|
| `run_compile_packet_identity_closure` | `pending` | score_claim=false, promotion_eligible=false, ready_for_exact_eval_dispatch=false |
| `generate_fec6_packetir_candidate_queue` | `pending` | score_claim=false, promotion_eligible=false, ready_for_exact_eval_dispatch=false |
| `prove_parser_consumption_and_byte_accounting` | `pending` | score_claim=false, promotion_eligible=false, ready_for_exact_eval_dispatch=false |
| `local_identity_profile_smoke` | `pending` | score_claim=false, promotion_eligible=false, ready_for_exact_eval_dispatch=false |
| `paired_exact_eval_after_candidate_queue` | `blocked_until_candidate_queue_and_operator_authorization` | score_claim=false, promotion_eligible=false, ready_for_exact_eval_dispatch=false |
