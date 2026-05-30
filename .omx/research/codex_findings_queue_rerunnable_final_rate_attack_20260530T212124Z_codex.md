# Codex Findings: Queue-Owned Final Rate Attack Rerunnability

Date: 2026-05-30T21:21:24Z
Repo: `/Users/adpena/Projects/pact`
Axis authority: no score claim; local queue execution and false-authority readiness only.

## Findings

The bounded queue fleet drain exposed several automation blockers that were not
method negatives:

- Existing materializer sweep queues failed on prior artifacts because
  `tools/run_family_agnostic_materializer_sweep.py` commands lacked
  `--allow-overwrite`.
- Selector/source recode steps failed on existing output directories or
  non-positive candidates instead of preserving a typed zero-delta/negative
  signal, because stale queues lacked `--allow-nonpositive-candidate` and
  `--overwrite`.
- Submission-closure native consumers failed on missing candidate archives
  instead of emitting per-candidate blocked refusals.
- Submission-closure native consumers were not rerunnable against existing
  closure directories because queue-fleet did not pass `--overwrite`.
- Loading older queue definitions could strand the fleet in this stale-command
  basin even after new builders were corrected.

## Landed Fixes

- Materializer execution queues now make family sweep commands rerunnable while
  preserving expected-existing-hash guards for single-candidate materializers.
- FECa selector and FP11 source brotli recode queue commands now preserve
  non-positive candidates as signal and are rerunnable.
- `experiment_queue.v1` normalization now canonicalizes known stale
  materializer commands at load time. This lets normal hash-drift/requeue
  machinery recover old queues without hand-editing generated queue JSON.
- Queue-fleet submission closure native consumers now pass `--overwrite`.
- Submission closure now blocks missing candidate archives as typed
  `candidate_blocked_refusal` rows instead of crashing the consumer path.

## Proof

- `ruff` passed on touched queue/materializer files.
- Focused regression:
  `128 passed in 46.23s`
  for `test_experiment_queue.py`, `test_queue_fleet_tool.py`,
  `test_materializer_submission_closure.py`, targeted
  `test_byte_shaving_campaign_queue.py`, and
  `test_family_agnostic_materializer_sweep.py`.
- Bounded local drain artifact:
  `.omx/research/queue_fleet_local_drain_exec_20260530T211206Z/fleet_local_drain_result.json`
  advanced native consumers and exposed the stale FECa supervisor blocker.
- Recovered FECa parent queue:
  `.omx/research/queue_fleet_local_drain_recovery_20260530T211857Z/frontier_final_rate_attack_feca_default_20260528Tlocal/supervisor_result.json`
  finished with `final_reason=terminal_queue_state` and
  `status_counts={"succeeded": 20}`.

## Frontier

No frontier score changed in this slice. The latest reported canonical frontier
remains:

- `[contest-CPU Linux x86_64]` 0.1919853363
- `[contest-CUDA T4]` 0.2053300290

## Next

Continue draining the queue fleet after this recovery pass. Remaining
`NEEDS_RECOVERY` rows are now mostly stale hash-drift rows caused by the new
canonical command migration; they should be retired by queue recovery or
supervision, not by manual artifact edits. The next frontier-relevant work is
to harvest exact-ready survivors from the rate-only passes, then feed any safe
rate budget into the P19/P18/P11 distortion-budget cascade.
