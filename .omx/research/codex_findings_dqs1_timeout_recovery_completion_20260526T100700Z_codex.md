# Codex Findings - DQS1 Timeout Recovery Completion

## Scope

Codex completed the targeted drop-many DQS1 timeout-recovery queue after landing
the manifest-only locality reuse hardening. This memo records queue-owned local
evidence only. It does not create score authority, promotion authority,
rank-or-kill authority, or exact-eval dispatch authority.

## Queue Result

- Queue:
  `.omx/research/frontier_rate_attack_feedback_refresh_20260526T085400Z_targeted_dqs1_timeout_recovery/targeted_drop_many_dqs1_followup_queue.json`
- Final queue status: `28/28` steps succeeded.
- Ready steps: none.
- Orphaned steps: `0`.
- Resource limits used by the recovery queue: `local_cpu: 2`,
  `local_io_heavy: 1`.

## Additional Candidates Completed

- `pairset_drop_many_k012_hc4a6f54207`
  - Locality controls: passed in `337.25697125005536` seconds.
  - Local CPU advisory: passed in `573.5224706670269` seconds.
  - Local CPU advisory score: `0.1920419585086524`.
  - Projected contest-CPU score: `0.1920314585086524`.
  - Conservative projected contest-CPU score: `0.1920344585086524`.
  - Eureka trigger: `false`.
  - Archive bytes: `178548`.
  - Archive SHA-256:
    `3f8844d3166c4637154b337b3a4b86de23ed28b64534dd9d9491ee673b9fb34b`.
- `pairset_drop_many_k016_h1759043cc6`
  - Locality controls: passed in `330.69871270807926` seconds.
  - Local CPU advisory: passed in `558.2561920000007` seconds.
  - Local CPU advisory score: `0.19204696093179302`.
  - Projected contest-CPU score: `0.192036460931793`.
  - Conservative projected contest-CPU score: `0.192039460931793`.
  - Eureka trigger: `false`.
  - Archive bytes: `178545`.
  - Archive SHA-256:
    `862b5e2c216b1694d4b95ab22e8d03556fc0cc69b14e2e7684c258f85606da99`.

## Interpretation

The longer locality budgets and single heavy-I/O queue limit converted the
remaining two candidates into successful locality evidence rather than timeout
artifacts. None of the four recovered candidates triggered the local-CPU drift
eureka gate against the current contest-CPU frontier pointer
`0.19202828295713675`.

The strongest recovered local advisory remains
`pairset_drop_many_k006_hd0960b13f2` from the prior memo, with local CPU
advisory score `0.19204195366237115` and projected contest-CPU score
`0.19203145366237112`. That is still above the current frontier pointer after
the guard band, so this queue is calibration/acquisition signal, not dispatch
signal.

## Follow-Up

- Add an I/O-aware scorer-eval resource kind so local CPU advisory jobs that
  inflate large raw outputs do not compete as pure CPU work.
- Feed the recovered successful negatives into grouped DQS1 acquisition so
  future drop-many proposals learn from the no-eureka band rather than repeating
  nearby combinations blindly.
- Keep manifest-only raw reuse as the permanent locality-controls contract.
