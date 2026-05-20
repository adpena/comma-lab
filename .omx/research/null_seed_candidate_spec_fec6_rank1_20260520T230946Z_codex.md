# Null-Seed Candidate Spec

- Schema: `null_seed_candidate_spec_v1`
- Spec: `null_seed_candidate::null-seed-contiguous_null_run-source_payload_selector_len_hdr_selector_payload-162171-178417`
- Verdict: `blocked_until_runtime_adapter_and_exact_eval`
- Score claim: `false`
- Promotion eligible: `false`
- Archive: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip`
- Archive SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- Member: `x`
- Member SHA-256: `f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd`
- Span: `[162171, 178417]` in `source_payload+selector_len_hdr+selector_payload`
- Original bytes: `16246`
- Original span SHA-256: `473a8343306b5d7d2d2279dd947856328c4038ae28f1bb60c47cca0ac971a85d`
- Seed bytes: `8`
- Generator: `pcg64`
- Seed reconstructs original payload: `false`
- Direct replacement ready: `false`
- Runtime adapter required: `true`

## Blockers

- `full_frame_inflate_output_parity_missing`
- `contest_cpu_exact_eval_missing`
- `contest_cuda_exact_eval_missing`
- `seed_reconstruction_mismatch`
- `runtime_adapter_not_materialized`
- `seed_mutation_frame_delta_proof_missing`
- `source_payload_seed_substitution_parse_risk`
- `selector_seed_adapter_required`

This spec is a lowering target, not a score artifact. It proves
archive/member custody and seed reconstruction status only.
