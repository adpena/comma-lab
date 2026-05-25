# Codex Findings: Many-Materializer Autonomous Portfolio Wiring

- utc: 2026-05-25T22:51:17Z
- lane_id: codex_many_materializer_autonomous_portfolio
- status: landed-local-artifacts-awaiting-gates
- authority: planning-only; no score, promotion, rank/kill, dispatch, or exact-eval authority

## Why

The prior local work was too leaf-oriented: single receiver/materializer slices were getting hardened, but the optimizer still lacked a registry-wide way to ask "which materializer families should be composed next?" This patch makes the materializer registry itself a first-class byte-shaving signal source so archive-section, packet-member, tensor, inverse-scorer, DQS1, and receiver-native contracts can be ranked, combined, lowered, and queued together.

## Implemented

- `tools/build_byte_shaving_signal_surface.py` can now seed a surface with `--include-materializer-registry-portfolio`.
- `tac.optimization.byte_shaving_signal_surface_builder` accepts scheduler-built materializer registry units without importing scheduler policy into `tac`.
- `tac.optimization.byte_shaving_campaign` preserves `materializer_registry_signal` through ranking, combinations, operation sets, PacketIR lowering, and plan refs.
- Regression coverage now drives registry portfolio signal through surface -> plan -> campaign queue/backlog/work-queue compilation.

## Concrete Artifacts

- `.omx/research/codex_many_materializer_autonomous_portfolio_20260525T225117Z/byte_shaving_signal_surface.json`
- `.omx/research/codex_many_materializer_autonomous_portfolio_20260525T225117Z/byte_shaving_campaign_plan.json`
- `.omx/research/codex_many_materializer_autonomous_portfolio_20260525T225117Z/materializer_backlog.json`
- `.omx/research/codex_many_materializer_autonomous_portfolio_20260525T225117Z/materializer_contexts.json`
- `.omx/research/codex_many_materializer_autonomous_portfolio_20260525T225117Z/materializer_work_queue.json`

## Artifact Summary

- Surface: 20 units total.
- Registry portfolio: 18 registered adapters, 18 target kinds, 9 executable adapters, 16 candidate-archive adapters, 10 cooperative receiver grammar hooks.
- Plan: 20 ranked units, 32 combinations, 32 PacketIR operation sets.
- Queue compile: 43 selected materialization rows collapsed into 19 typed backlog/work-queue rows.
- Top backlog families: archive-section entropy recode, packet-member merge, packet-member recompress, renderer-payload DFL1, DQS1 drop-pair, ZIP-header elide, inverse-scorer cell, tensor factorize.

## Guardrails

- Every registry portfolio unit carries `materializer_registry_portfolio_requires_concrete_artifact_context`.
- Required context fields and receiver runtime proof remain explicit blockers.
- Local CPU advisory DQS1 observations remain planner signal only.
- Canonical MLX/PyTorch downstream drift equation is attached as macOS-MLX research signal only.
- All emitted artifacts keep score/promotion/rank/dispatch authority false.

## Next Gate

Bind concrete artifact contexts for the top materializer rows, especially archive-section entropy recode, packet-member merge/recompress, renderer-payload DFL1, and tensor factorize. Then run the materializer work queue locally, harvest observations, and feed the resulting exact-readiness failures/successes back into this same many-op surface rather than selecting a single leaf by hand.
