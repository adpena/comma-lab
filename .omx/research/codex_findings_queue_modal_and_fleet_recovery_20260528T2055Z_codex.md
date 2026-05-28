# Codex Findings - Queue-Owned Modal Dispatch And Fleet Recovery

## Summary

2026-05-28 Codex pass converted the manually launched FP11 brotli survivor
exact eval into a queue-owned Modal dispatch surface, recovered the exact CUDA
negative, and drained the remaining unhealthy semantic-mutation materialization
queue to terminal local evidence.

## Landed Code

- `a4d88e89e` - Queue Modal materializer exact dispatch.
  - `materializer_exact_eval_dispatch_plan` now supports `provider=modal`.
  - Modal dispatch commands run provider-detached and require a pre-existing
    active queue-owned lane claim via `--claim-policy require_active`.
  - Provider preclaim checks now cover Modal CLI/auth config without leaking
    token values.
- `92f489b03` - Expose paused queue resume commands.
  - Queue-fleet status now emits `paused_with_queued_work_count` and
    `next_resume_commands`.
  - This closes the manual loop where paused queues were counted as actionable
    but did not expose the next executable command.

## Empirical Anchors

- FP11 brotli exact CUDA replay:
  - archive SHA-256:
    `b7106c9bdbb8a2df18af622636ca79a11fa0c771a09c75219474d980b8997c8c`
  - archive bytes: `178493`
  - Modal call id: `fc-01KSR4YNDEJW45CM6VJC8FTC31`
  - recovered output:
    `experiments/results/modal_auth_eval/fp11_brotli_recode_b7106_cuda_t4_20260528T203810Z/modal_cuda_auth_eval_result.json`
  - `[contest-CUDA T4]` score: `0.226147822852098`
  - verdict: exact negative for CUDA frontier promotion.
- Follow-up queue-owned Modal dispatch plan correctly refused duplicate spend
  after the same-archive CUDA negative:
  - blockers included same-lane terminal score not below active CUDA floor,
    stale exact-ready rows, and missing runtime-consumption proof.
- Queue fleet scan after local recovery:
  - `invalid_queue_count=0`
  - `needs_recovery_count=0`
  - `ready_to_supervise_count=0`
  - `paused_with_queued_work_count=4`
  - `next_resume_commands` now lists the four paused queues explicitly.

## Local Queue Recovery

Drained
`.omx/research/repair_multi_archive_autonomous_semantic_mutation_psv3_fec6_20260528T0640Z/repair_materialization_queue.json`
with state
`.omx/state/experiment_queue_repair_multi_archive_semantic_mutation_materialization.sqlite`.

The queue moved from `NEEDS_RECOVERY` with
`experiment_queue_observation_artifact_postcondition_failures:15` to terminal
healthy evidence. The run executed `90/90` local-only steps with no failures.
All ten generated repair gates remain fail-closed:

- candidate archives not materialized;
- runtime-consumption proof missing;
- exact auth eval still required before score or promotion claim.

That is a useful negative/blocked posterior, not a score claim.

## Frontier

Unchanged after this pass:

- `[contest-CPU Linux x86_64]` `0.1919853363`, archive `b7106c9bdbb8...`
- `[contest-CUDA T4]` `0.2053300290`, archive `9cb989cef519...`

## Next Actions

1. Use the new Modal provider path for the next exact-ready survivor instead of
   hand-building Modal commands.
2. Resume only paused queues whose exact-readiness and spend posture still pass
   current authority gates.
3. Promote the semantic-mutation repair negatives into the acquisition posterior
   as blocked materialization families, not as frontier failures.
4. Continue PR95/HNeRV MLX and P18/P19/P11 cascade work with queue-owned local
   execution first and exact auth only at the eureka boundary.
