# PR103 Arithmetic Histogram Global Combo Probe

- planning_only: `true`
- score_claim: `false`
- dispatch_attempted: `false`
- gpu_required: `false`
- ready_for_archive_preflight: `false`
- ready_for_exact_eval_dispatch: `false`

## Search

- stream_count: `5`
- top_per_stream: `20`
- beam_width: `256`
- evaluated_state_count: `16590`
- objective: `minimize_exact_merged_ac_delta_plus_exact_ac_histograms_brotli_delta`
- mathematical_guard: `selected per-stream proposal deltas are retained for audit, but ranking uses exact recomputation of the full merged AC stream plus the full Brotli-compressed histogram sideband`
- known_scope_limit: `q8 AC histogram rows only; latent_hi_histogram_brotli remains a separate optimization surface`

## Best Candidate

- merged_ac_delta: `0`
- histogram_brotli_delta: `-12`
- estimated_member_delta_if_runtime_adapter_supported: `-12`
- source_probe_delta_sum: `-13`
- non_additivity_delta: `1`
- selected_options: `['blocks.0.weight:candidate13', 'blocks.1.weight:candidate14', 'blocks.2.weight:source', 'blocks.3.weight:candidate4', 'stem.weight:source']`

## Frontier

- `blocks.0.weight`: options `21`, expanded `21`, kept `21`, best `-9`
- `blocks.1.weight`: options `21`, expanded `441`, kept `256`, best `-11`
- `blocks.2.weight`: options `21`, expanded `5376`, kept `256`, best `-11`
- `blocks.3.weight`: options `21`, expanded `5376`, kept `256`, best `-12`
- `stem.weight`: options `21`, expanded `5376`, kept `256`, best `-12`

## Blockers

- `candidate_archive_missing`
- `candidate_runtime_adapter_missing`
- `candidate_inflate_output_parity_missing`
- `strict_pre_submission_compliance_json_missing`
- `lane_dispatch_claim_missing`
- `exact_cuda_auth_eval_missing`

## Dispatch Blockers

- `pr103_arithmetic_histogram_global_combo_probe_is_not_dispatch_authorization`
- `candidate_archive_missing`
- `candidate_runtime_adapter_missing`
- `candidate_inflate_output_parity_missing`
- `strict_pre_submission_compliance_json_missing`
- `lane_dispatch_claim_missing`
- `exact_cuda_auth_eval_missing`
