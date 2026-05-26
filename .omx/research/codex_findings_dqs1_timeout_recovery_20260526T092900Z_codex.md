# Codex Findings - DQS1 Timeout Recovery

## Scope

Codex recovered the targeted drop-many DQS1 follow-up queue after locality
controls failed at the 540 second inflate boundary. This is a no-score-authority
local-first queue artifact and does not create promotion, rank, kill, or exact
eval authority.

## Findings

- Root cause: under-budgeted locality inflates plus too much concurrent external
  storage pressure, not a DQS1 method negative.
- Retry bug class: unmanifested non-empty raw files were previously accepted as
  legacy reusable outputs. This could preserve timed-out partials as if they
  were valid locality evidence.
- Fixed behavior: locality reuse is manifest-only; unmanifested partial raw
  outputs are cleared and rerun.
- Queue defaults: locality controls now use step timeout 2400 seconds,
  per-inflate timeout 1200 seconds, global-mutated timeout 1800 seconds, and
  max inflate parallelism 2.
- Scheduler resource policy: generated targeted queue uses `local_io_heavy: 1`.

## Executed Evidence

- Queue:
  `.omx/research/frontier_rate_attack_feedback_refresh_20260526T085400Z_targeted_dqs1_timeout_recovery/targeted_drop_many_dqs1_followup_queue.json`
- First recovery worker:
  `success_count=2`, `failure_count=0`, both locality controls passed.
- Second recovery worker:
  `success_count=2`, `failure_count=0`, both local CPU advisory evals passed.
- Third recovery worker:
  `success_count=4`, `failure_count=0`, both drift/eureka rows and raw-retention
  plans passed.

## Advisory Scores

- `pairset_drop_many_k012_h1ecc99d178`: local CPU advisory score
  `0.1920439585086524`, projected contest score `0.1920334585086524`,
  eureka trigger `false`.
- `pairset_drop_many_k006_hd0960b13f2`: local CPU advisory score
  `0.19204195366237115`, projected contest score `0.19203145366237112`,
  eureka trigger `false`.

Both advisories are above the current contest-CPU frontier pointer and should be
used as calibration and acquisition signal only.

## Verification

- `ruff check` on touched Python files passed.
- Static queue fixture validation passed.
- Timeout-recovery queue validation passed.
- Focused pytest passed:
  manifest reuse, unmanifested partial rerun, DQS1 queue defaults, checked-in
  DQS1 queue fixture, and frontier-cycle local I/O cap.

## Next Work

- Continue the timeout-recovery queue from 14 succeeded / 14 queued.
- Run the next two build/materialize/locality candidates with `local_io_heavy: 1`.
- Promote mixed I/O scorer evals into a more precise scheduler resource class if
  further profiling shows local CPU advisory inflates are the bottleneck.
