# Null-Seed Candidate Spec

- Schema: `null_seed_candidate_spec_v1`
- Spec: `null_seed_candidate::null-seed-whole_null_section-selector_payload-178168-178417`
- Verdict: `blocked_until_runtime_adapter_and_exact_eval`
- Score claim: `false`
- Promotion eligible: `false`
- Archive: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip`
- Archive SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- Member: `x`
- Member SHA-256: `f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd`
- Span: `[178168, 178417]` in `selector_payload`
- Original bytes: `249`
- Original span SHA-256: `fc5c431b5d793c33e2f320076fe6f0dd76c2d91e3826ae4b05abfb4f86f453ca`
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
- `selector_seed_adapter_required`

This spec is a lowering target, not a score artifact. It proves
archive/member custody and seed reconstruction status only.
