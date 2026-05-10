# PR103 Arithmetic Histogram Beam Probe

- proposal_id: `pr103_ac_plan_ca7c48857ebf6c8e`
- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_archive_preflight: `false`
- ready_for_exact_eval_dispatch: `false`

## Target

- label: `blocks.3.weight`
- stream_index: `4`
- symbols: `19440`

## Search

- evaluated_candidate_count: `979`
- candidate_symbol_count: `16`
- top_symbols: `16`
- rounds: `3`
- beam_width: `8`
- deltas: `[-2, -1, 1, 2]`

## Best Candidate

- change_count: `3`
- merged_ac_delta: `0`
- histogram_brotli_delta: `-6`
- estimated_member_delta_if_runtime_adapter_supported: `-6`

## Moves

- round `1` symbol `132` delta `-2` old `255` new `253` count `769`
- round `2` symbol `123` delta `-2` old `194` new `192` count `584`
- round `3` symbol `120` delta `2` old `248` new `250` count `749`

## Blockers

- `candidate_archive_missing`
- `candidate_runtime_adapter_missing`
- `candidate_inflate_output_parity_missing`
- `strict_pre_submission_compliance_json_missing`
- `lane_dispatch_claim_missing`
- `exact_cuda_auth_eval_missing`
