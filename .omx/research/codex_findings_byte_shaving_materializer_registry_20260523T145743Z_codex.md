# Codex Findings: Byte-Shaving Materializer Registry

UTC: 2026-05-23T14:57:43Z

## Scope

Continuation of the local-first byte-shaving automation bridge. The pass
reviewed the DQS1-specific campaign queue compiler and hardened it into a
registry-owned materialization boundary while preserving false-authority score
semantics.

## Landed Changes

- Added `src/comma_lab/scheduler/byte_shaving_materializer_registry.py` as the
  typed fail-closed boundary between `byte_shaving_campaign_plan.v1` rows and
  executable local-first queue rows.
- Registered exactly one executable adapter:
  `dqs1_pairset_drop_pair_adapter`, for `unit_kind=pair` and
  `operation_family=drop_pair`, targeting DQS1 pairset local-first queues.
- Refactored `byte_shaving_campaign_queue.py` to resolve every selected
  operation through the registry and emit per-operation
  `materializer_resolutions`.
- Replaced hard-coded DQS1 unsupported-family checks with typed blockers such
  as `materializer_not_registered:byte_range:entropy_recode` and
  `materializer_not_registered:byte_range:null_remove_or_seed`.
- Preserved custody and false-authority metadata on materialization rows,
  portfolio rows, and action summaries. Queueable rows remain local-first
  materialization inputs only; they do not become exact-eval, promotion, rank,
  or score authority.
- Avoided per-row registry-manifest duplication after review; rows carry the
  registry schema pointer and top-level surfaces carry the full registry
  manifest once.
- Hardened the registry after adversarial review: `pair/drop_pair` no longer
  auto-resolves to DQS1. DQS1 materialization now requires either the explicit
  `dqs1_pairset_drop_pair_adapter` materializer or an explicit
  `target_kind=dqs1_pairset_drop_pair`.
- Propagated selected operation/unit blockers into materialization blockers so
  source rows carrying parity, locality, trust-region, or planning-coordinate
  caveats cannot become queueable by omission.
- Made partial queue generation fail closed by default when a campaign plan has
  both executable and blocked rows. Operators must pass an explicit partial
  materialization flag and rationale to build a queue from the executable
  subset.
- Wrapped portfolio/action-summary/materialization outputs in the canonical
  proxy evidence boundary while preserving `local_materialization_ready` only as
  a non-score-authority local queue signal.
- Preserved materializer/source custody into DQS1 queue experiment metadata and
  harvest records by extending the generic experiment queue normalizer to retain
  experiment-level metadata.
- Switched `tools/build_byte_shaving_campaign_queue.py` to guarded artifact
  writes with no silent overwrite; replacement requires `--overwrite-output`
  plus the expected existing SHA for each output.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_campaign_queue.py`
  -> 10 passed.
- `.venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_optimizer_candidate_queue.py src/tac/tests/test_dqs1_local_first_queue_builder.py`
  -> 68 passed.
- `.venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_signal_surface_builder.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_optimizer_candidate_queue.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_dqs1_local_first_autopilot.py src/tac/tests/test_experiment_queue.py`
  -> 109 passed.
- `.venv/bin/ruff check src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/comma_lab/scheduler/byte_shaving_materializer_registry.py src/tac/tests/test_byte_shaving_campaign_queue.py tools/build_byte_shaving_campaign_queue.py`
  -> passed.
- `tools/build_byte_shaving_campaign_queue.py` smoke on
  `.omx/research/byte_shaving_campaign_master_gradient_plan_20260523T144718Z.json`
  produced 36 blocked rows and 0 executable rows, preserving fail-closed
  behavior for master-gradient byte-range rows until byte-range materializers
  exist. The hardened smoke additionally shows selected unit blockers preserved
  in every blocked row.

## Artifacts

- `.omx/research/byte_shaving_campaign_master_gradient_materializer_registry_smoke_20260523T150754Z.json`
- `.omx/research/byte_shaving_campaign_master_gradient_materializer_registry_portfolio_20260523T150754Z.json`
- `.omx/research/byte_shaving_campaign_master_gradient_materializer_registry_action_summary_20260523T150754Z.json`

## Remaining Work

- Add first non-DQS1 materializer adapter, likely byte-range
  `null_remove_or_seed` or `entropy_recode`, with archive/runtime custody and
  locality proof.
- Generalize queue emission beyond DQS1 pairsets so NeRV, HNeRV, residual,
  PacketIR, scorer-response, tensor, and archive-section materializers can
  produce executable DAG nodes without implying score authority.
- Teach the DAG scheduler to use the registry manifest for resource routing,
  artifact-retention policy, and storage-tier planning before launching large
  materialization batches.
