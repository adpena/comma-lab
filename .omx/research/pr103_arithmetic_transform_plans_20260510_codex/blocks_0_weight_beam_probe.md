# PR103 Arithmetic Histogram Beam Probe

- proposal_id: `pr103_ac_plan_f2e951cdd6b1a8a8`
- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_archive_preflight: `false`
- ready_for_exact_eval_dispatch: `false`

## Target

- label: `blocks.0.weight`
- stream_index: `1`
- symbols: `46656`

## Search

- evaluated_candidate_count: `968`
- candidate_symbol_count: `16`
- top_symbols: `16`
- rounds: `3`
- beam_width: `8`
- deltas: `[-2, -1, 1, 2]`

## Best Candidate

- change_count: `3`
- merged_ac_delta: `0`
- histogram_brotli_delta: `-9`
- estimated_member_delta_if_runtime_adapter_supported: `-9`

## Moves

- round `1` symbol `118` delta `1` old `180` new `181` count `1538`
- round `2` symbol `147` delta `-2` old `179` new `177` count `1536`
- round `3` symbol `115` delta `2` old `228` new `230` count `1957`

## Blockers

- `candidate_archive_missing`
- `candidate_runtime_adapter_missing`
- `candidate_inflate_output_parity_missing`
- `strict_pre_submission_compliance_json_missing`
- `lane_dispatch_claim_missing`
- `exact_cuda_auth_eval_missing`
