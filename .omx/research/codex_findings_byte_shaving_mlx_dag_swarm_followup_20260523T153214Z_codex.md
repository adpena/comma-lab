# Codex Findings - Byte-Shaving MLX DAG Swarm Follow-Up

- timestamp_utc: `2026-05-23T15:32:14Z`
- lane_id: `lane_codex_byte_shaving_mlx_dag_swarm_hardening_20260523`
- evidence_axis: `[planning-only]`, `[macOS-CPU advisory only]`
- score_claim: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Swarm Inputs

Three read-only explorer agents inspected the current landing:

- Queue/DAG/MLX roadmap: confirmed queue validates but only uses `local_cpu`;
  `local_mlx=1` is declared and idle; state reconciliation must precede
  workers.
- Byte-shaving adversarial review: found a red focused test, target-kind
  metadata loss through the planner, and optimizer-candidate queue intake
  dropping materializer-ready operation metadata.
- Observability review: identified materializer backlog artifacts, compact
  queue-observe summaries, idle-resource explanations, and cross-axis candidate
  cards as the next interpretability surfaces.

## Landed Hardening

- Preserved operation `target_kind` / `materializer_target_kind` through
  `tac.optimization.byte_shaving_campaign` operation candidates, ranked units,
  prefix rows, and combination rows.
- Preserved row-level and nested selected-operation materializer metadata when
  converting `optimizer_candidate_queue_v1` rows into byte-shaving signal
  surfaces.
- Added regression coverage proving an explicit DQS1 target kind, without an
  explicit materializer id, compiles through the DQS1 drop-pair registry
  adapter.
- Split materializer backlog metrics so `blocked_row_count` is unique source
  selections while `blocked_resolution_count` / `selected_operation_count`
  preserve repeated operation evidence.
- Reconciled the active DQS1 queue state before any worker execution. Blocking
  orphan count changed from `7` to `0`; definition drift is now `0`.
- Executed a bounded local-only worker pass with `--max-steps 2 --max-parallel
  2`. The storage-tier planning step and raw-artifact retention-planning step
  both succeeded; no cloud, GPU, exact-eval, score, rank, or promotion authority
  was attempted.

## Durable Artifacts

- `.omx/research/byte_shaving_campaign_master_gradient_swarm_followup_smoke_20260523T153126Z.json`
- `.omx/research/byte_shaving_campaign_master_gradient_swarm_followup_materializer_backlog_20260523T153126Z.json`
- `.omx/research/byte_shaving_campaign_master_gradient_swarm_followup_portfolio_20260523T153126Z.json`
- `.omx/research/byte_shaving_campaign_master_gradient_swarm_followup_action_summary_20260523T153126Z.json`
- `.omx/research/dqs1_queue_state_reconciliation_swarm_hardening_20260523T153048Z_codex.json`
- `.omx/research/dqs1_local_first_storage_plan_20260523.json`

## Current Backlog Signal

The regenerated master-gradient byte-shaving smoke remains fail-closed:

- executable rows: `0`
- blocked rows: `36`
- queueable rows: `0`
- materializer backlog rows: `3`

Top adapter work orders:

1. `byte_range/entropy_recode`: `33` unique blocked selections, `66` blocked
   resolution occurrences, `4` affected units, `268542` candidate saved bytes,
   receiver status `receiver_target_contract_required`.
2. `byte_range/null_remove_or_seed`: `29` unique blocked selections, `56`
   blocked resolution occurrences, `4` affected units, `227720` candidate
   saved bytes, receiver status `receiver_target_contract_required`.
3. `byte_range/delta_encode`: `16` blocked selections, `16` blocked resolution
   occurrences, `2` affected units, `64432` candidate saved bytes, receiver
   status `receiver_target_contract_required`.

## Queue State

Post-reconciliation observe:

- queue id: `dqs1_pairset_local_first`
- mode: `running`
- definition drift: `0`
- failed steps: `0`
- status counts after bounded worker: `queued=6`, `skipped=1`, `succeeded=9`
- blocking orphan count: `0`
- historical orphan rows: `188`, nonblocking provenance
- completed worker steps: `storage_tier_plan` and `plan_raw_artifact_retention`
- next ready steps: `proactive_cleanup` and
  `local_cpu_contest_drift_eureka` for `pairset_drop_one_rank024_pair0112`
- idle declared resource: `local_mlx=1` because the active queue currently has
  only `local_cpu` resource steps

## Remaining Gaps

- Build the first non-DQS1 byte-range receiver/materializer contract, starting
  with `byte_range/entropy_recode`, including archive grammar mapping,
  no-op/runtime-consumption proof, locality controls, and false-authority
  output.
- Add compact queue-observe summaries for grouped orphans, idle-resource causes,
  missing artifacts, and storage readiness.
- Promote batch harvest/portfolio rebuild/queue regeneration into scheduler DAG
  steps instead of leaving that loop in the DQS1 autopilot stop gap.
- Add a gated `[macOS-MLX research-signal]` branch after CPU advisory only when
  batch-shape invariance and calibration gates pass; keep MLX separate from
  contest CPU/CUDA authority.
- Add cross-axis candidate cards keyed by candidate id so `[macOS-CPU advisory
  only]`, `[macOS-MLX research-signal]`, `[contest-CPU]`, and `[contest-CUDA]`
  records are visible without conflating authority.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_signal_surface_builder.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_optimizer_candidate_queue.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_dqs1_local_first_autopilot.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_cooperative_receiver_packet_grammars.py`
  -> `115 passed in 8.74s`
- `.venv/bin/ruff check src/comma_lab/scheduler/__init__.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/comma_lab/scheduler/byte_shaving_materializer_registry.py src/tac/optimization/byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py tools/build_byte_shaving_campaign_queue.py`
  -> passed
- `.venv/bin/python -m py_compile src/comma_lab/scheduler/__init__.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/comma_lab/scheduler/byte_shaving_materializer_registry.py src/tac/optimization/byte_shaving_campaign.py tools/build_byte_shaving_campaign_queue.py`
  -> passed
- `git diff --check` -> passed
- `.venv/bin/python tools/lane_maturity.py validate` -> `1180 lane(s) validated cleanly`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
  -> valid, `experiment_count=3`, `step_count=16`, `local_cpu=2`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --execute --max-steps 2 --max-parallel 2`
  -> `success_count=2`, `failure_count=0`, `stop_reason=max_steps_reached`
