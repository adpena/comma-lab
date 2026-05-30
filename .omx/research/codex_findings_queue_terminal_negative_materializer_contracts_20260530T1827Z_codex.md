# Codex Findings: Queue Terminal-Negative Materializer Contracts

UTC: 2026-05-30T18:27Z
Agent: codex
Scope: experiment_queue.v1, FECA final-rate materializer queues, queue fleet supervision

## Finding

Terminal-negative materializer outcomes were being treated as broken automation.
The failure class showed up when a sweep validly harvested zero accepted
candidates, or when a materializer candidate was receiver-refused / runtime-
deferred before exact readiness. Those outcomes are signal, not queue failure.

## Landed Fix

- Family-agnostic empirical sweep completion contracts are now classified as
  planning-only unless they explicitly require archive/runtime/receiver custody.
- Optimizer candidate queues now accept typed receiver-blocked and runtime-
  deferred rows as terminal signal when authority fields remain false.
- Materializer submission closure now emits schema-valid refusal artifacts for
  empty source queues and receiver-contract refusals instead of crashing.
- The queue reconciler gained opt-in `--include-failed` support for historical
  failed steps whose declared postconditions already pass.
- The FECA campaign queue no longer requires accepted candidates when harvesting
  sweep manifests.
- Queue fleet status now classifies paused exact-eval dispatch queues as
  `PAUSED_EXACT_DISPATCH_GATE`, with `mlx_first_no_auto_cloud_dispatch`, instead
  of exposing manual resume commands.

## Live Evidence

Bounded fleet scan:

- `queue_count=160`
- `actionable_count=0`
- `invalid_queue_count=0`
- `needs_recovery_count=0`
- `ready_to_supervise_count=0`
- `paused_exact_dispatch_gate_count=4`
- `next_resume_commands=[]`
- `next_supervise_commands=[]`

Recovered FECA queues:

- `frontier_final_rate_attack_feca_default_exec2_20260528Tlocal`: terminal,
  `succeeded=20`
- `frontier_final_rate_attack_feca_default_exec3_20260528Tlocal`: terminal,
  `succeeded=20`
- `frontier_final_rate_attack_feca_default_exec4_20260528Tlocal`: terminal,
  `succeeded=20`

Authority posture:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

## System Intelligence

This closes a queue-custody bug class: terminal negative/refusal outcomes must
stay in the queue and artifact graph, not in operator memory. Exact-dispatch
queues remain paused unless the exact-authority path explicitly resumes them;
local MLX / local CPU queue execution remains the default campaign substrate.

## Frontier

No frontier score changed in this slice. Current known exact frontier remains:

- `[contest-CPU Linux x86_64]` `0.19198533626623068`
- `[contest-CUDA T4]` `0.20533002902019143`
