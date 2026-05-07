# HDC2 Combined Entropy Reduction Manifest

- planning_only: `true`
- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_exact_eval_dispatch: `false`

## Target

- label: `public_pr106_belt_and_suspenders:hdc2_global_prev_symbol_contexts`
- section: `decoder_packed_brotli`
- frontier_archive_sha256: `b0a12549a39e34a0d7f83ea99e05e55fcd01d795a15db2ffb3d92ccc6267e53f`
- frontier_section_bytes: `170127`

## Byte Accounting

- actual_hdc2_replacement_bytes: `221381`
- net_byte_delta_now: `51254`
- model_overhead_target_bytes: `40840`
- payload_entropy_gap_target_bytes: `23979`
- net_byte_delta_after_model_overhead_only: `10414`
- net_byte_delta_after_combined_targets: `-13565`
- projected_rate_score_delta_after_combined_targets: `-0.009032376699102255`

## Bounded Candidate

- variant: `mixed_range_raw_global_prev_symbol_schema_indexed_q_streams_plus_raw_scales`
- bytes: `208821`
- byte_reduction_vs_hdc2: `12560`
- net_byte_delta_vs_frontier_section: `38694`
- static_context_header_reduction_vs_hdc2: `21511`
- payload_delta_vs_hdc2: `8951`
- ready_for_exact_eval_dispatch: `false`

## Break-Even Gate

- remaining_reduction_to_beat_frontier_section_bytes: `38695`
- archive_build_gate: `candidate_stream_bytes_must_be_less_than_current_frontier_section_bytes`

## Stream Evidence

- roundtrip_valid: `true`
- raw_equal: `true`
- decoded_output_equal: `true`

Interpretation: HDC2 model overhead alone is still byte-negative.
The next implementation must reduce both static context overhead and
range-payload entropy gap before archive construction is rational.
