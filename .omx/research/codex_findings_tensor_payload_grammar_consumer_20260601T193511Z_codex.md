# Generic tensor payload grammar consumer landed

## Context

The substrate-agnostic tensor payload grammar optimizer now emits
`tensor_payload_grammar_optimizer.v1` reports and optimizer queues. The next
integration gap was autopilot consumption: without a consumer, saturation and
receiver-binding verdicts would remain standalone tool output.

## Landed surfaces

- `src/tac/cathedral_consumers/tensor_payload_grammar_consumer/__init__.py`
  - consumes `tensor_payload_grammar_optimizer.v1`
  - classifies saturated/weak-gap reports as negative rate posterior
  - routes unsaturated positive reports to receiver/archive binding
  - preserves Tier-A false-authority markers
- `src/tac/tests/test_tensor_payload_grammar_consumer.py`
  - canonical cathedral consumer contract validation
  - saturated demotion routing
  - unsaturated receiver-binding routing
  - schema and false-authority blocker preservation

## Real artifact consumption

Artifact root:

`/Volumes/VertigoDataTier/pact/tensor_payload_grammar_real_artifacts_20260601T192954Z`

Consumed reports:

- `pr101_generic_tensor_payload_report.json`
  - consumer result: `pr101_generic_tensor_payload_report_consumer_result.json`
  - saved bytes: `39`
  - saturation: `entropy_saturated`
  - planner action: `record_tensor_payload_saturation_and_demote_format_churn`
- `pact_nerv_generic_tensor_payload_report.json`
  - consumer result: `pact_nerv_generic_tensor_payload_report_consumer_result.json`
  - saved bytes: `44`
  - saturation: `entropy_saturated`
  - planner action: `record_tensor_payload_saturation_and_demote_format_churn`
- `pact_nerv_generic_tensor_payload_exhaustive4_report.json`
  - consumer result: `pact_nerv_generic_tensor_payload_exhaustive4_report_consumer_result.json`
  - saved bytes: `50`
  - saturation: `entropy_saturated`
  - planner action: `record_tensor_payload_saturation_and_demote_format_churn`

## Verdict

The generic tensor grammar lane is now system intelligence, not an orphan tool:
future MLX/HPRC/HNeRV/NeRV/non-NeRV tensor exports can be routed by the
cathedral/autopilot layer as saturated, receiver-worthy, or incomplete. The
current PR101 and PACT-NeRV tested tensor payloads are correctly demoted as
within-substrate format churn rather than queued for receiver work.

This reinforces the main optimal-grammar result: competitive integer payloads
have negligible grammar headroom; future score movement must come from better
trained substrates, better latent/residual grammars, or payload families that
are structurally unsaturated before entropy coding.

## Verification

- `uv run ruff check src/tac/cathedral_consumers/tensor_payload_grammar_consumer/__init__.py src/tac/tests/test_tensor_payload_grammar_consumer.py src/tac/packet_compiler/tensor_payload_grammar_optimizer.py tools/tensor_payload_grammar_optimizer.py src/tac/tests/test_tensor_payload_grammar_optimizer.py`
- `uv run pytest src/tac/tests/test_tensor_payload_grammar_consumer.py src/tac/tests/test_tensor_payload_grammar_optimizer.py src/tac/tests/test_pr101_optimal_grammar_campaign_consumer.py src/tac/tests/test_check_335_cathedral_consumer_directory_contract.py -q`

