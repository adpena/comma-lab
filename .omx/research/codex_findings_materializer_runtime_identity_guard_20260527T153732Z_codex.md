# Codex findings: materializer runtime identity guard

UTC: 2026-05-27T15:37:32Z

Authority: research_only_false_authority

## Finding

`materializer_chain_complete` still allowed a chain manifest that claimed
`runtime_adapter_ready=true` and `candidate_runtime_adapter_blocker_cleared=true`
to pass without a live runtime tree, unless the queue author remembered to set
`required_runtime_adapter_identity=true`.

That made runtime identity an optional postcondition detail instead of a default
chain-completion invariant. It was a false-authority surface because a chain
could look complete to the queue while lacking proof that the runtime tree being
harvested was the runtime tree that would inflate the candidate.

## Fix

- `src/comma_lab/scheduler/experiment_queue.py` now requires runtime-adapter
  identity for `materializer_chain_complete` by default. A caller must
  explicitly set `required_runtime_adapter_identity=false` to opt out.
- `src/tac/optimizer/materializer_chain_harvest.py` now validates every claimed
  chain runtime directory. A valid `runtime_dir` alias can no longer mask a
  stale or missing `candidate_runtime_dir`.
- Tests now pin both invariants.

## Queue-owned smoke

Ran a bounded final-rate campaign smoke:

```bash
.venv/bin/python tools/build_frontier_final_rate_attack_queue.py \
  --queue-id frontier_final_rate_attack_identity_guard_20260527Tsmoke \
  --output-dir .omx/research/frontier_final_rate_attack_identity_guard_20260527Tsmoke \
  --results-root /Volumes/VertigoDataTier/experiments/results/frontier_final_rate_attack/identity_guard_20260527Tsmoke \
  --target-kind packet_member_recompress_v1 \
  --target-kind packet_member_zip_header_elide_v1 \
  --allow-materializer-overwrite \
  --include-exact-readiness-followup \
  --local-cpu-concurrency 2 \
  --max-steps 4 \
  --max-parallel 2 \
  --execute
```

Then resumed the same queue for the remaining local follow-up steps:

```bash
.venv/bin/python tools/experiment_queue.py \
  --queue .omx/research/frontier_final_rate_attack_identity_guard_20260527Tsmoke/experiment_queue.json \
  --state .omx/state/experiment_queue_frontier_final_rate_attack_identity_guard_20260527Tsmoke.sqlite \
  run-worker --execute --max-steps 6 --max-parallel 2 \
  --poll-interval-seconds 0.05 --idle-sleep-seconds 0.0 --max-idle-cycles 1
```

Evidence:

- all 10 queue steps succeeded;
- final observer health: `healthy=true`, `blockers=[]`;
- materializer harvest accepted one source row for each tested executable target;
- exact-readiness bridge skipped both rows as non-rate-positive/zero-delta;
- dispatch plans remained false-authority with `ready_for_exact_eval_dispatch=false`;
- no auth eval dispatch was attempted.

Compact local evidence lives at
`.omx/research/frontier_final_rate_attack_identity_guard_20260527Tsmoke/`.
Bulky generated candidate/closure artifacts live on VertigoDataTier under
`/Volumes/VertigoDataTier/experiments/results/frontier_final_rate_attack/identity_guard_20260527Tsmoke/`.

## Validation

```bash
.venv/bin/ruff check \
  src/comma_lab/scheduler/experiment_queue.py \
  src/tac/optimizer/materializer_chain_harvest.py \
  src/tac/tests/test_experiment_queue.py \
  src/tac/tests/test_materializer_chain_harvest_scheduler.py \
  src/tac/tests/test_experiment_queue_observer.py \
  src/tac/tests/test_byte_shaving_campaign_queue.py

.venv/bin/python -m pytest \
  src/tac/tests/test_experiment_queue.py \
  src/tac/tests/test_materializer_chain_harvest_scheduler.py \
  src/tac/tests/test_experiment_queue_observer.py \
  src/tac/tests/test_byte_shaving_campaign_queue.py -q
```

Result: `244 passed in 23.71s`.

## Next binding

The false-authority guard is now stronger. The next score-moving step is not more
guard-only polish; it is binding a non-zero-saving materializer context:
FEC10/DQS1 payload grammar, byte-range entropy/ANS target, or a grouped
PoseNet-null + SegNet-region waterfill cascade that emits receiver-closed
materializer rows for the same queue-owned final-rate path.
