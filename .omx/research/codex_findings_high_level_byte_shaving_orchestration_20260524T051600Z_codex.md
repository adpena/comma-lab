---
schema: codex_findings_v1
created_at_utc: 2026-05-24T05:16:00Z
agent: codex
topic: high_level_byte_shaving_orchestration_and_ssh_custody
score_authority: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
---

# High-Level Byte-Shaving Orchestration + SSH Custody Hardening

## Summary

The inverse-cell materializer is now treated as a leaf actuator under the
planner, not the primary planning surface. The canonical high-level flow is:

`signal surface -> inverse-steganalysis action functional -> byte-shaving campaign plan -> materializer work queue -> experiment_queue.v1 -> staircase DAG / SSH executor`

This closes two bug classes:

1. Recursive SSH output pullbacks could copy directories into stale nested
   local trees and had no post-pull recursive manifest/cap failure.
2. Inverse action-functional results were plan-able only through a separate
   path, losing composition signal with candidate queues, master-gradient,
   X-ray, engineered corrections, and non-HNeRV units.

## Landed Engineering

- `src/comma_lab/scheduler/ssh_experiment_queue_executor.py`
  - recursive output pullbacks now use content-sync semantics with
    `rsync --delete` and trailing slash source/destination handling;
  - output pullbacks emit pre/post recursive manifests and SHA-256s;
  - recursive manifest truncation fails closed with return code `75`.
- `src/comma_lab/scheduler/byte_shaving_campaign_queue.py`
  - inverse-cell chain postconditions now require `inflate_parity_satisfied`
    when parity context exists or `fail_if_inflate_parity_blocked=true`.
- `src/tac/optimization/byte_shaving_signal_surface_builder.py`
  - `inverse_steganalysis_discrete_action_functional.v1` artifacts can now be
    mixed into the canonical byte-shaving signal surface through
    `inverse_action_functional_paths`;
  - mixed surfaces preserve false authority and retain
    `inverse_action_functional_refs` for source custody.
- `tools/build_byte_shaving_signal_surface.py`
  - added `--inverse-action-functional`.
- `tools/run_byte_shaving_materializer_campaign.py`
  - can now start without `--plan` from high-level sources such as
    `--scorer-response`, `--inverse-scorer-surface`,
    `--mlx-effective-spend-triage-selection`, or `--atom`;
  - it builds an action functional, campaign plan, materializer queue, and
    staircase/DAG artifacts in one queue-owned control surface.
- `.gitignore`
  - ignores `.omx/research/**/queue_state*.sqlite|db` and sidecars.
- `docs/experiment_scheduler_design.md`
  - documents recursive SSH custody and the high-level runner path.

## Evidence

Pre-fix live SSH evidence:

- `experiments/results/inverse_cell_chain_ssh_input_manifest_smoke_20260524T045043Z/campaign/staircase_ssh_executor_execute_ssh_mlx.json`
- SHA-256: `13dcc852bf8d5d75b57e1f55041c2460d70580443b7789208a5bc0bd7dec70a9`
- Result: remote command succeeded and input directory custody used recursive
  manifests, but local recursive pullback placed `chain_output` under
  `chain_output/chain_output`, causing postcondition failure.

No-network recursive custody smoke:

- `.omx/research/staircase_ssh_input_custody_smoke_20260524T000000Z/smoke_report.json`
- Report SHA-256: `913025f16a8294ff7ebaebf91a3ecb56541870f1e259263e91fc248c0bf969c5`
- Result: `success_count=1`, `failure_count=0`,
  `directory_push_used_delete=true`, `output_artifact_exists=true`,
  `network_attempted=false`, `paid_dispatch_attempted=false`.

High-level orchestration smoke:

- `.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/campaign3/materializer_campaign_run.json`
- Run SHA-256: `42aefbc2d9f1665b7f8783ecabe780239b55c45c355a349fe28a580c43370cd0`
- Campaign plan SHA-256:
  `f9be792094dca14ed695a0d14dfcb5f9497b7d7f5aafee895a103ec404d3cee0`
- Action functional SHA-256:
  `788ba5ccd1043f009d90fb672f1a9c357c0d75bfd8f3a2a24fe95ed764e0353a`
- Result: one high-level scorer-response source generated an action
  functional, byte-shaving plan, materializer execution queue, and staircase
  dispatch plan. The worker stayed dry-run (`stop_reason=dry_run`), selected
  one `local_mlx` materializer step, and preserved
  `score_claim=false`, `promotion_eligible=false`,
  `rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_ssh_experiment_queue_executor.py \
  src/tac/tests/test_ssh_input_custody_smoke.py -q

.venv/bin/python -m pytest \
  src/tac/tests/test_byte_shaving_campaign_queue.py \
  src/tac/tests/test_byte_shaving_signal_surface_builder.py \
  src/tac/tests/test_byte_shaving_campaign.py \
  src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q

.venv/bin/ruff check \
  src/comma_lab/scheduler/ssh_experiment_queue_executor.py \
  src/comma_lab/scheduler/byte_shaving_campaign_queue.py \
  src/tac/optimization/byte_shaving_signal_surface_builder.py \
  tools/build_byte_shaving_signal_surface.py \
  tools/run_byte_shaving_materializer_campaign.py \
  src/tac/tests/test_ssh_experiment_queue_executor.py \
  src/tac/tests/test_byte_shaving_campaign_queue.py \
  src/tac/tests/test_byte_shaving_signal_surface_builder.py \
  src/tac/tests/test_byte_shaving_campaign.py \
  src/tac/tests/test_byte_shaving_materializer_campaign_runner.py
```

All passed locally.

## Remaining Work

- After this patch is committed and pulled on `tertiary`, rerun a live SSH
  execute smoke against patched pullback code.
- Wire latest mixed byte-shaving plan discovery into `tools/operator_briefing.py`
  so the operator sees the current best high-level queue without spelunking
  research directories.
- Expand the high-level runner source set with master-gradient, X-ray,
  canonical-equation, atom, PacketIR, and cross-family candidate portfolio
  inputs so HNeRV, NeRV-family, and non-NeRV candidates share one acquisition
  surface.
