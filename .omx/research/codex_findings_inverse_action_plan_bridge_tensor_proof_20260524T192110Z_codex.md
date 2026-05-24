# Codex Findings: Inverse Action Plan Bridge And Tensor Proof

generated_at_utc: 2026-05-24T19:21:10Z
agent: codex
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false

## Verdict

The inverse-steganalysis stack is real, but the automation gap is still the
materialization layer. This landing tightens two boundaries that move selected
water buckets closer to queue-owned deterministic candidate generation without
relaxing exact-auth authority:

- `byte_shaving_campaign_plan.v1` now embeds its inverse-action
  materialization bridge when inverse-action water-bucket portfolios are
  present.
- `tensor_factorize_v1` can emit a file-backed
  `family_agnostic_runtime_consumption_proof_v1` reconstruction proof.

## XHigh Audit Signal

The Lagrange read-only adversarial pass found the authority layer and planner
layer broad enough, but the materializer layer too thin. Its shortest concrete
patch recommendation was to make strict MLX/inverse-scorer water-bucket cells
with operation-set compiler hints become family-agnostic materializer rows
instead of falling back to `high_level_operation_compiler_required`.

This landing addresses the first no-signal-loss edge: the campaign plan itself
now carries `materialization_bridge`, so PacketIR/materializer readiness counts
travel with the plan artifact and operator/queue consumers do not need to
reconstruct the bridge out of band.

## Tensor Proof Boundary

`tensor_factorize_v1` now writes a deterministic runtime-consumption proof when
`runtime_consumption_proof_out` is requested. The proof records candidate/source
archive and member SHA-256s, factor rank, source shape/dtype, reconstruction
error, tolerance checks, and declared cooperative receiver identity.

The proof only satisfies the receiver contract when reconstruction is finite,
within declared tolerance, and both cooperative receiver id and adapter kind are
declared. Otherwise tensor candidates remain blocked by
`tensor_factorized_payload_requires_cooperative_receiver`.

## Safeguards

- Generated bridge/proof artifacts keep score, promotion, rank/kill, dispatch,
  GPU, and exact-CUDA authority false.
- Materializer work rows remain local proof-chain work and still require
  exact auth eval before score or promotion.
- Tensor proof support is file-backed and member-bound through generic
  runtime-consumption verification.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_signal_surface_builder.py -q`
  - `115 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_operator_briefing.py -q`
  - `31 passed`
- `.venv/bin/python -m ruff check src/tac/optimization/family_agnostic_materializers.py tools/run_family_agnostic_materializer.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/optimization/byte_shaving_campaign.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_byte_shaving_campaign.py`
  - `All checks passed`

## Review Gate Greenup

- src/tac/optimization/byte_shaving_campaign.py -- CLEAN
- src/tac/tests/test_byte_shaving_campaign.py -- CLEAN

## Remaining Work

The next patch should extend compiler defaults and fixtures so at least one
strict MLX or inverse-scorer water-bucket cell with concrete target coordinates
compiles through action functional -> campaign plan -> PacketIR -> materializer
work queue as an executable family-agnostic archive/tensor/member job. After
that, the long-burn compression portfolio should become a queue-owned
materializer target instead of ad hoc quality/level loops.
