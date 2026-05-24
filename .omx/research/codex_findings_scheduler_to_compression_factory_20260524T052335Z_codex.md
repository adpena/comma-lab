# Codex Findings: Scheduler To Compression Factory Bridge

UTC: 2026-05-24T05:23:35Z
Role: xhigh bridge reviewer
Scope: read-only assessment of queue/DAG/scheduler/executor/harvest/autopilot authority. No code changed.

## Authority Boundary

The implemented system is a local proof-chain factory plus fail-closed exact-eval handoff, not a score authority. Planning rows, signal surfaces, staircase DAGs, MLX/MPS/local queue telemetry, exact-ready consumers, and dispatch plans keep score/promotion/rank/dispatch authority false until a contest CPU/CUDA auth result and claim lifecycle exist.

Primary references:

- `src/comma_lab/scheduler/experiment_queue.py`: `experiment_queue.v1`, resource kinds, SQLite claim/worker/observe/performance.
- `src/comma_lab/scheduler/staircase_dag.py`: planning-only DAG built from experiment queue.
- `src/comma_lab/scheduler/ssh_experiment_queue_executor.py`: queue-owned SSH executor with artifact mobility and local postcondition authority.
- `tools/run_byte_shaving_materializer_campaign.py`: one-command local materializer loop.
- `src/comma_lab/scheduler/byte_shaving_campaign_queue.py`: campaign plan to materializer work queue to `experiment_queue.v1`.
- `tools/harvest_materializer_chain_candidates.py`: materializer chain harvest and optional exact-readiness bridge.
- `src/comma_lab/scheduler/materializer_exact_eval_consumer.py` and `materializer_exact_eval_dispatch_plan.py`: exact-ready to paused/dry-run dispatch queue.
- `src/tac/optimizer/exact_dispatch_authority.py`: dispatch authorization gate; `ready_for_exact_eval_dispatch` is only an input fact.
- `tools/parallel_dispatch_top_k.py` and `tools/harvest_and_reseed.py`: exact-CUDA actuator/harvest/reseed loop after candidates are already exact-ready.

## What Is Implemented Today

1. Durable queue: `experiment_queue.v1` owns experiment steps, dependencies, resources (`local_cpu`, `local_io_heavy`, `local_mlx`, cloud kinds), SQLite state, claim/refusal, dry-run/execute workers, observation, and telemetry-only performance summaries.
2. DAG layer: `staircase_dag.v1` and `staircase_dispatch_plan.v1` are derived planning views over `experiment_queue.v1`; they do not own score, promotion, or step truth.
3. SSH executor: `tools/run_staircase_ssh_executor.py` consumes the staircase plan plus source queue, rechecks source hash, claims locally, pushes declared inputs, runs remote command, pulls artifacts back, and evaluates terminal postconditions locally.
4. Byte-shaving signal surface: `src/tac/optimization/byte_shaving_signal_surface_builder.py` aggregates candidate queues, scorer-response rows, inverse scorer surfaces, auth references, MLX calibration, atoms, and equations into planning units. It is planning-only.
5. Materializer campaign runner: `tools/run_byte_shaving_materializer_campaign.py` can start from high-level sources or an existing campaign plan, build an inverse action functional, build a campaign plan, compile a materializer work queue, emit an `experiment_queue.v1`, optionally emit staircase artifacts, initialize state, run a bounded worker, and write observation/performance.
6. Candidate-producing local chains: DQS1 pairset drop rows and inverse-scorer cell candidate chains can emit byte-closed candidate archives when required contexts/postconditions are satisfied. The DQS1 parent lane is already running; do not duplicate it.
7. Harvest to exact-ready: materializer chain manifests can be harvested into an optimizer candidate queue, then exact-readiness bridge artifacts.
8. Exact-eval handoff: exact-ready rows can be deduped and converted into paused/dry-run dispatch queues; real paid dispatch requires explicit execute flags, active claim policy, exact dispatch authority, and later contest auth results.
9. Operator briefing currently discovers materializer exact-ready handoffs and dispatch plans, but not full materializer campaign run summaries.

## Apparatus-Only Versus Candidate/Training Signal

Apparatus-only today:

- `docs/experiment_scheduler_design.md` broad `pact-sched` design text.
- Byte-shaving signal surfaces, inverse action functionals, campaign plans, materialization summaries, portfolios, backlog artifacts.
- Staircase DAGs and dispatch plans.
- Queue performance summaries and MLX/local advisory rows.
- Exact-ready consumers and dispatch plans until a real claim/dispatch/auth result lands.

Directly candidate-producing today:

- DQS1 pairset drop-pair materializer rows, but that tranche is parent-owned right now.
- Inverse-scorer cell candidate chain via `tools/run_inverse_scorer_cell_candidate_chain.py` when `materializer_contexts` supplies the candidate archive template, action functional, digest, output dir, and parity/runtime context.
- Byte-range entropy-recode chain infrastructure exists, but it is only useful when explicit context/proofs provide a byte range, runtime consumption proof, and output chain directory.

Direct scorer-response/training signal today:

- Existing scorer-response dataset builders normalize advisory rows into fail-closed scorer-response tables.
- Exact CUDA dispatch harvest through `tools/parallel_dispatch_top_k.py` plus `tools/harvest_and_reseed.py` creates authoritative calibration anchors only after `contest_auth_eval.json` validates as contest CUDA. Local materializer telemetry is not scorer-response training signal by itself.

## Minimal Local Production Loop

The production loop should live as:

- Reusable math, parsers, materializers, byte grammars, and scorer-response normalization in `src/tac/optimization/` or related `tac` modules.
- Queue/DAG/executor/harvest/briefing/dispatch orchestration in `src/comma_lab/scheduler/`.
- Thin operator entry points in `tools/`.
- Run artifacts and control ledgers in `.omx/research/byte_shaving_materializer_campaign_<UTC>/`; queue state in `.omx/state/experiment_queue_<queue_id>.sqlite`.

Local loop command shape:

```bash
UTC=$(date -u +%Y%m%dT%H%M%SZ)
.venv/bin/python tools/run_byte_shaving_materializer_campaign.py \
  --campaign-id byte_factory_${UTC} \
  --queue-id byte_shaving_materializer_factory_${UTC} \
  --run-dir .omx/research/byte_shaving_materializer_campaign_${UTC} \
  --scorer-response <scorer_response_dataset_or_response_json> \
  --inverse-scorer-surface <inverse_scorer_surface_json> \
  --mlx-effective-spend-triage-selection <mlx_triage_selection_json> \
  --materializer-contexts <materializer_contexts_json> \
  --candidate-limit 64 \
  --materializer-execution-limit 64 \
  --local-cpu-concurrency auto \
  --materializer-resource-concurrency local_mlx=1 \
  --include-storage-preflight \
  --storage-expected-workload-root /Users/adpena/Projects/pact/experiments/results \
  --proactive-cleanup-cold-store-root <VertigoDataTier_or_other_cold_store_root> \
  --exact-eval-dispatch-provider lightning \
  --exact-eval-dispatch-max-total-cost 5.0 \
  --emit-staircase-plan \
  --execute \
  --max-steps 64 \
  --max-parallel 18
```

Notes:

- `--include-storage-preflight` executes the cleanup preflight; with move cleanup it needs a cold-store root.
- Do not add `--exact-readiness-require-ready` on first factory runs unless the goal is to fail hard when no exact-ready rows are emitted.
- Paid exact eval remains a separate claim-gated action through the generated exact-ready bridge/dispatch queue and `parallel_dispatch_top_k.py`.

## Top Integration Gaps And Fastest Fixes

1. Full campaign run summaries are not first-class operator briefing inputs. Fast patch: add scan/summary for `byte_shaving_materializer_campaign_run.v1` and `materializer_execution_queue.json` next commands in `tools/operator_briefing.py`, analogous to the existing materializer exact-ready handoff section.
2. The factory is not yet an always-on loop. Fast command: run the local loop above with `--execute --max-steps N`, then rerun against the same queue with `tools/experiment_queue.py --queue <queue> run-worker --execute --max-steps N --max-parallel <n>` until no ready steps remain.
3. Candidate-producing adapters are still narrow. Fast patch: take the highest-ranked blocked materializer backlog row and add the missing `materializer_contexts` entry, rather than broadening scheduler abstractions. For byte-range entropy recode this means source archive/runtime proof/byte range/output dir; for inverse-scorer cell this means template/action functional/digest/output dir plus parity inputs.
4. Local queue telemetry is not automatically folded back into acquisition. Fast patch: feed `tools/experiment_queue.py --queue <queue> performance` into the next runner with `--queue-performance-summary` and stable runtime/cache identities, then add only a small briefing/preflight check that every executed factory run has a follow-on performance summary.
5. Exact-ready dispatch is correctly strict but still requires a separate human-visible bridge step. Fast command after local chains emit accepted exact-ready rows: `tools/build_materializer_exact_eval_dispatch_plan.py --bridge-report <exact_readiness_bridge_report.json> --dispatch-plan-out <dispatch_plan.json> --experiment-queue-out <dispatch_queue.json> --provider lightning`; keep it dry-run until claim/paid-dispatch authorization is explicit.

## Scheduler Work To Cap

- Cap broad `pact-sched` platform registry/dashboard/promote work from `docs/experiment_scheduler_design.md`; the real queue and runner already exist.
- Cap additional staircase/DAG polish unless it changes queue-owned execution or artifact pullback reliability.
- Cap new proxy authority fields and advisory score routing; exact-dispatch authority already fails closed.
- Cap DQS1 local-first scheduler tweaks while the parent DQS1 tranche is running.
- Prioritize frontier artifacts now: run materializer chains that emit byte-closed archives, harvest them, and only then use exact-ready dispatch handoff if custody gates pass.

## Codex Integration Addendum 2026-05-24T06Z

- Gap 1 was closed by `e50cf9392` (`Surface high-level byte shaving queues in briefing`): operator briefing now surfaces `byte_shaving_materializer_campaign_run.v1`, queue paths, run commands, storage plans, cleanup plans, exact-ready paths, and recent errors.
- Follow-on hardening in progress: queue-owned storage preflight artifacts must use current run IDs instead of stale action-summary dates, existing storage-plan overwrites must carry expected SHA-256 guards, generated artifact-retention JSON must stay ignored, and paid exact-dispatch consumers must require explicit `contest_cuda` score-axis metadata.
