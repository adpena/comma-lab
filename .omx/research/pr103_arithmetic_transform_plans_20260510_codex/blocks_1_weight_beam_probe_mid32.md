# PR103 Arithmetic Histogram Beam Probe

- proposal_id: `pr103_ac_plan_f99018242b93f2a0`
- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_archive_preflight: `false`
- ready_for_exact_eval_dispatch: `false`

## Target

- label: `blocks.1.weight`
- stream_index: `2`
- symbols: `46656`

## Search

- evaluated_candidate_count: `11414`
- candidate_symbol_count: `32`
- top_symbols: `32`
- rounds: `3`
- beam_width: `32`
- deltas: `[-4, -2, -1, 1, 2, 4]`

## Best Candidate

- change_count: `3`
- merged_ac_delta: `0`
- histogram_brotli_delta: `-5`
- estimated_member_delta_if_runtime_adapter_supported: `-5`

## Moves

- round `1` symbol `126` delta `4` old `241` new `245` count `2088`
- round `2` symbol `114` delta `4` old `206` new `210` count `1787`
- round `3` symbol `117` delta `-1` old `231` new `230` count `2003`

## Blockers

- `candidate_archive_missing`
- `candidate_runtime_adapter_missing`
- `candidate_inflate_output_parity_missing`
- `strict_pre_submission_compliance_json_missing`
- `lane_dispatch_claim_missing`
- `exact_cuda_auth_eval_missing`
