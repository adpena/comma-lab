# Codex Findings - HPRC Pair-Scoped Bounded Runner

UTC: 2026-06-01T01:55:42Z
Author: Codex
Authority: runner-plan and MLX-local advisory plumbing; no contest CPU/CUDA score claim

## Finding

Pair-scoped HPRC residual candidates are no longer report-only rows.  This
landing adds a substrate-local bounded-runner plan API and a thin operator CLI
that converts measured pair-scoped residual candidates into executable profile
rows.

Every emitted row:

- carries the measured `threshold_abs_le_pairs=...` transform;
- requires `--reuse-baseline-profile`;
- writes into a deterministic candidate output directory;
- preserves receiver-proof follow-up requirements;
- keeps `score_claim=false`, `promotion_eligible=false`, and
  `ready_for_exact_eval_dispatch=false`.

## Live Plan

Artifact:

`.omx/research/hprc_pair_scoped_bounded_runner_plan_20260601T015542Z_codex/hprc_pair_scoped_residual_bounded_runner_plan.json`

Rows:

1. `hprc-threshold-abs-le-pairs-984a5d110c0f3a66`
   - estimated bytes removed: `335775`
   - selected pairs: `284`
   - protected pairs: `316`
2. `hprc-threshold-abs-le-pairs-3317d5db885b6839`
   - estimated bytes removed: `239877`
   - selected pairs: `270`
   - protected pairs: `330`
3. `hprc-threshold-abs-le-pairs-58b03d146e87e401`
   - estimated bytes removed: `116618`
   - selected pairs: `262`
   - protected pairs: `338`

## Why This Matters

The previous full-video run proved the top row is an MLX-local advisory win and
receiver-proven candidate.  The missing automation was not another manual
rerun; it was making the same measured candidates executable by the bounded
runner with baseline reuse by default.  This closes that loop and makes the next
allocator iteration a queue action rather than an operator copy-paste.

## Next Action

Execute the runner rows with baseline reuse, receiver-proof the best survivor,
then exact-gate only candidates whose MLX-local component deltas and archive
bytes justify contest CPU/CUDA spend.  The remaining speed target is variant
scorer response; baseline recomputation is now removed from the repeated-sweep
path.
