# Codex Findings: Byte-Shaving Campaign Queue Bridge

Date: 2026-05-23T14:48:00Z
Author: Codex
Status: landed candidate for commit

## Scope

Added the first executable bridge from `byte_shaving_campaign_plan.v1` into
the existing experiment queue system.

The bridge is deliberately narrow:

- DQS1 pairset `drop_pair` selections can compile into ordinary DQS1
  local-first queue actions.
- Unsupported operation families such as byte ranges, tensors, archive
  sections, packet members, frames, and scorer-response rows remain blocked
  until their archive/runtime materializers exist.
- All emitted rows remain proxy/local planning evidence with false authority.

## Landed Surfaces

- `src/comma_lab/scheduler/byte_shaving_campaign_queue.py`
  - validates byte-shaving campaign plans;
  - joins selected operations back to DQS1 pair indices;
  - computes selected pairsets from a base DQS1 pairset;
  - emits a DQS1-compatible portfolio/action-summary pair only for executable
    `drop_pair` rows;
  - records blocked rows and blocker reasons for everything else.

- `tools/build_byte_shaving_campaign_queue.py`
  - writes materialization, portfolio, and action-summary JSON artifacts;
  - optionally writes a DQS1 local-first `experiment_queue.v1`;
  - refuses queue creation when there are no executable DQS1 rows.

- `src/tac/tests/test_byte_shaving_campaign_queue.py`
  - covers executable DQS1 drop-pair compilation;
  - covers mixed unsupported byte-range blocking;
  - covers CLI emission of a validating DQS1 queue.

## Smoke

Durable fail-closed smoke artifacts:

- `.omx/research/byte_shaving_campaign_master_gradient_plan_20260523T144718Z.json`
- `.omx/research/byte_shaving_campaign_master_gradient_materialization_20260523T144718Z.json`
- `.omx/research/byte_shaving_campaign_master_gradient_action_summary_20260523T144718Z.json`
- `.omx/research/byte_shaving_campaign_master_gradient_portfolio_20260523T144718Z.json`

The master-gradient byte-range plan produced `0` executable rows and `36`
blocked rows. First blockers included unsupported `entropy_recode`, missing
DQS1 base-pair context, and missing pair indices. This is the intended
fail-closed behavior.

## Verification

- `py_compile` passed for the new module and CLI.
- `ruff check` passed for the new module, CLI, and related planner surfaces.
- Focused pytest bundle passed: `73 passed`.
- `git diff --check` clean.

## Next Integration

The next bridge should add an explicit materializer registry so non-DQS1
operation families can become executable only after they provide:

- archive/runtime write contract;
- input unit custody schema;
- deterministic no-op/runtime-consumption proof;
- locality or inflate parity controls;
- retention policy for generated raw/cache bulk;
- local CPU/MLX advisory and exact-auth eureka hooks.

Priority adapters: `byte_range:null_remove_or_seed`, `byte_range:entropy_recode`,
and `archive_section:section_entropy_recode`, because master-gradient and X-ray
signals are already producing those opportunities.
