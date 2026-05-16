# L5 v2 TT5L side-info effect curve tracked artifact hardening

Date: 2026-05-16
Agent: Codex
Scope: L5 v2 staircase / TT5L side-info effect curve

## Problem

The canonical side-info effect-curve producer existed, but its default output
path was under ignored `experiments/results/`:

```text
experiments/results/l5_v2_probe/tt5l_sideinfo_effect_curve.jsonl
```

That is the wrong durability class for the effect curve. The curve is a small
structured planning/custody artifact consumed by the L5 v2 readiness surface,
not a raw provider transcript. Leaving it ignored made the architecture-lock
gate vulnerable to no-signal-loss failure: the local artifact could exist while
`main` carried no durable evidence.

## Fix

Moved the canonical default to the tracked research plane:

```text
.omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json
```

Also added a regression test requiring the default side-info effect curve path
to stay under `.omx/research/` rather than `experiments/results/`.

## Seeded evidence

Added a first seed-cell file:

```text
.omx/research/l5_v2_tt5l_sideinfo_effect_curve_seed_cells_20260516_codex.json
```

It contains exactly one real cell: TT5L `trained` on `contest_cuda`, sourced from
the reviewed recovered exact CUDA artifact:

```text
archive_sha256: 2b05b7351b690b0b2251ddc620d80dd9a1833051cfa07e679106d00fbc70024a
runtime_tree_sha256: ed41e941b624b00412c680c56dc5b9b23db32c70ce1008d03f5bca939917b6cd
score: 3.9007398365396795
seg_dist: 0.02515214
pose_dist: 0.18563657
archive_bytes: 34603
axis: contest_cuda
variant: trained
```

This is not a score claim and not architecture-lock evidence by itself. It
preserves the available cell so the missing work is explicit: `contest_cpu` for
trained plus `zero`, `random_lsb`, `shuffled`, and `ablated` controls on both
axes.

## Expected gate state

The generated effect-curve artifact should remain fail-closed until all required
cells are present:

```text
score_claim=false
promotion_eligible=false
ready_for_exact_eval_dispatch=false
predicate_passed=false
```

This turns the old ambiguous blocker `tt5l_sideinfo_effect_curve_artifact_missing`
into concrete missing-cell and missing-control blockers while preserving all
available exact CUDA signal.
