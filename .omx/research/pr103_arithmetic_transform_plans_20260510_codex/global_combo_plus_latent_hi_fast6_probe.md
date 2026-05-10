# PR103 Arithmetic Histogram Global Combo Probe

- planning_only: `true`
- score_claim: `false`
- dispatch_attempted: `false`
- gpu_required: `false`
- ready_for_archive_preflight: `false`
- ready_for_exact_eval_dispatch: `false`

## Search

- stream_count: `6`
- top_per_stream: `6`
- beam_width: `64`
- evaluated_state_count: `1743`
- objective: `minimize_exact_merged_ac_delta_plus_exact_ac_histograms_brotli_delta`
- mathematical_guard: `selected per-stream proposal deltas are retained for audit, but ranking uses exact recomputation of the full merged AC stream plus the full Brotli-compressed histogram sideband`
- known_scope_limit: `supports q8 AC histogram rows and latent_hi_histogram_brotli when the input frontier contains a latent_hi_bytes target`

## Best Candidate

- merged_ac_delta: `0`
- histogram_brotli_delta: `-10`
- estimated_member_delta_if_runtime_adapter_supported: `-11`
- source_probe_delta_sum: `-18`
- non_additivity_delta: `7`
- selected_options: `['blocks.0.weight:candidate1', 'blocks.1.weight:candidate4', 'blocks.2.weight:source', 'blocks.3.weight:candidate4', 'stem.weight:source', 'latent_hi_bytes:candidate1']`

## Frontier

- `blocks.0.weight`: options `7`, expanded `7`, kept `7`, best `-9`
- `blocks.1.weight`: options `7`, expanded `49`, kept `49`, best `-9`
- `blocks.2.weight`: options `7`, expanded `343`, kept `64`, best `-9`
- `blocks.3.weight`: options `7`, expanded `448`, kept `64`, best `-10`
- `stem.weight`: options `7`, expanded `448`, kept `64`, best `-10`
- `latent_hi_bytes`: options `7`, expanded `448`, kept `64`, best `-11`

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
