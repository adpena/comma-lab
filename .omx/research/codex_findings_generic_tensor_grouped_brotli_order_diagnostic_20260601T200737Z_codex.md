# Generic Tensor Grouped Brotli Order Diagnostic Landed

Date: 2026-06-01T20:07:37Z
Author: Codex

## Verdict

The generic `tensor_payload_grammar_optimizer.v1` now carries a grouped Brotli
order diagnostic over the selected transformed tensor payloads. The diagnostic
tests deterministic section orders (`identity`, `size_desc`,
`histogram_greedy`) and records both:

- `grouped_saved_bytes_vs_identity`: useful statistical-order learning signal.
- `grouped_saved_bytes_vs_selected_isolated`: the stricter action signal used
  for queue/receiver work.

This distinction is load-bearing. A grouped stream can improve over identity
order while still losing to the already-selected isolated per-tensor grammar.
Only the stricter selected-isolated comparison is allowed to create a grouped
receiver-binding queue candidate.

## Live PR101 Proof

Source:
`/Volumes/VertigoDataTier/pact/pr101_real_grouped_runtime_campaign_20260601T172357Z/source_decoder_state_dict.pt`

Artifacts:

- `/Volumes/VertigoDataTier/pact/tensor_payload_grammar_grouped_diag_20260601T200522Z/pr101_tensor_payload_grouped_diag_report.json`
- `/Volumes/VertigoDataTier/pact/tensor_payload_grammar_grouped_diag_20260601T200522Z/pr101_tensor_payload_grouped_diag_queue.json`
- `/Volumes/VertigoDataTier/pact/tensor_payload_grammar_grouped_diag_20260601T200522Z/pr101_tensor_payload_grouped_diag_report_consumer_result.json`

Measured result:

- tensors: 28
- selected isolated tensor bytes: 162,223
- baseline isolated tensor bytes: 162,273
- isolated selected savings: 50 bytes
- selected over empirical floor: 1.0146715678015736
- saturation status: `entropy_saturated`
- histogram-greedy grouped Brotli bytes: 162,315
- identity grouped Brotli bytes: 162,694
- grouped saved versus identity: 379 bytes
- grouped delta versus selected isolated: +92 bytes
- grouped saved versus selected isolated: 0 bytes

Consumer verdict:

- `planner_action`: `record_tensor_payload_saturation_and_demote_format_churn`
- `receiver_work_justified`: false
- `demotion_recommended`: true
- `score_claim`: false
- `ready_for_exact_eval_dispatch`: false

## System Wiring

The grouped diagnostic is now part of the generic optimizer report and planner
feedback. If a future substrate has positive
`grouped_saved_bytes_vs_selected_isolated`, `build_tensor_payload_optimizer_queue`
emits a `tensor_payload_grouped_brotli_order` queue row with normal false
authority blockers. The cathedral consumer and operator briefing use the same
stricter selected-isolated metric for receiver work, while preserving identity
order savings as posterior signal.

## Consequence

For PR101/fec6-class decoder weights, generic tensor grammar is saturated at the
format layer. The remaining score movement is not another same-substrate
container/order pass; it requires either a substrate/runtime-specific receiver
adapter with a measured archive win or a representation/training change that
alters the tensor payload distribution itself.
