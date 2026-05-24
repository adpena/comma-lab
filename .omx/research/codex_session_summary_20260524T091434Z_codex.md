# Codex Session Summary

UTC: 2026-05-24T09:14:34Z
Agent: Codex

## Landed This Session

- Hardened PR95 MLX timing-smoke identity so plan and runtime sidecars share the
  same candidate key: stage, optimizer descriptor, seed, steps, and channel
  count.
- Wired PR95 public archive export and public-runtime consumption proof through
  plan generation, optimizer matrix queue generation, queue postconditions,
  timing execution, and representation training sidecars.
- Added a regression that builds a one-cell PR95 MLX optimizer matrix, executes
  it through `experiment_queue.v1`, and harvests it into
  `optimizer_candidate_queue_v1` while preserving false authority.
- Executed the corrected three-cell PR95 MLX matrix queue and harvested timing
  candidates for stages 1, 5, and 8.

## Authority Boundary

All PR95 MLX matrix timing outputs remain `[macOS-MLX research-signal]`.
They are useful for local training throughput, optimizer selection, and queue
plumbing, but they are not score claims, promotion authority, rank/kill
authority, or exact-eval dispatch authority.

## Current Frontier Context

No exact auth-eval frontier movement happened in this session. The current
scanner-derived snapshot remains:

- `[contest-CPU Linux x86_64]` best: `0.1920282830`
- `[contest-CUDA T4]` best: `0.2053300290`

## Next Claude/Codex Hooks

- Consume PR95 timing harvests as neutral cost/throughput intelligence via
  optimizer atoms and scheduler telemetry.
- Teach learned sweep to distinguish timing-only rows from calibrated
  quality/scorer rows by requiring an explicit adapter.
- Generalize the runtime-consumption proof into a neutral materializer receiver
  proof contract for archive-section, packet-member, tensor-factorization, and
  future family-agnostic materializers.
