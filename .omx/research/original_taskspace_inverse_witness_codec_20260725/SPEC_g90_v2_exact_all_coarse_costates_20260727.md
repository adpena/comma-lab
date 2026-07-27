# G90 V2 — Authority-Separated Exact-All Coarse Costates

## Purpose

G90 V1 proved that finite realized response reverses first-order Pose and Seg
directions, then terminated on a one-cell inference-versus-autograd boundary
tie. V2 retains the useful family atlas while removing both unsound gates:

1. scorer inference owns cells, Pose targets, base Pose MSE, and exact replay;
2. the autograd surface supplies derivatives only and records its numerical
   drift from authority;
3. all deterministic physical groups produced by
   `role × direction × amplitude × collision partition` are replayed exactly;
   and
4. Pareto pruning, local admission, member-byte rate, and candidate claims are
   forbidden.

This remains a coarse family atlas. It is not an atom selector. There are eight
semantic family coordinates before collision partitioning, but the exact
physical-group count is variable and at least eight. The retained n600 geometry
has a known 12-group batch at `[288,304)`. Exact-all replay closes false
negatives across the complete deterministic physical-group set; hierarchical
refinement is still required inside a group before program induction.

## Frozen state and authority

The base is exact G85 `P + incumbent G74 A`, archive SHA
`b9c8ab2af8886c5b26bba63e02b7c5fe9951bb42a871c5e8472483977788d9fd`
and raw SHA
`436ce2b6965c859556a217df9b1cc17784d988f2af900c35201d3e3c7f372782`.
G46 owns target cells; G78/G87 own historical semantic fields and fresh G72
proposal custody. Current cells are separately recomputed through frozen
SegNet inference. Target cells are also independently replayed through
inference and must equal G46.

The differentiable forward may choose a different argmax at a numerical tie.
That does not alter authority. Each batch records current/target:

- expected, inference-authority, and differentiable cell SHAs;
- drift cell count and pair IDs; and
- minimum top-two margin on drift cells.

Pose receives the same separation. Inference-mode PoseNet supplies the target
Pose6 and base per-pair MSE used by exact replay and aggregate closure. The
autograd candidate/target Pose6 SHAs and maximum absolute deltas are telemetry.
The gradient objective differentiates the candidate output against the frozen
inference-authority target.

## Exact-all replay and state custody

Every batch binds its ordered expected physical-group IDs and count, and every
one of those coordinates must have non-null exact Seg and Pose fields. A batch
with fewer than eight groups is invalid, but a collision-split batch may have
more; fixed-count formulas are forbidden. No linearized Pareto filter runs.
Each exact row binds:

- `pose_conditioning_y0_sha256`;
- `pose_conditioning_y1_sha256`;
- `seg_base_y1_sha256`;
- `seg_candidate_y1_sha256`; and
- proof that candidate Y0 is unchanged.

Pose fields expire on any Y0 or Y1 state change. Seg fields expire if either
base or candidate Y1 changes. G91 remains an initializer/factorability witness;
the final order is semantic Y1 first, then conditional `Y0 | final Y1`, then
G94 sequential serialization and actual ZIP pricing.

## Resumability and schemas

V2 uses a new SSD root:

`/Volumes/VertigoDataTier/pact/taskspace_exact_coarse_costates_v2_n600_20260727_r5`

Schemas:

- `tac.taskspace_exact_coarse_costate_config.v2`
- `tac.taskspace_exact_coarse_costate_preflight.v2`
- `tac.taskspace_exact_coarse_costate_batch.v2`
- `tac.taskspace_exact_coarse_costate_stage.v2`
- `tac.taskspace_exact_coarse_costate_aggregate.v2`
- `tac.taskspace_exact_coarse_costate_blocker.v2`

Batch checkpoints are immutable and atomic every at most 16 pairs. Stage
checkpoints are immutable every 120 pairs. The aggregate exists only after five
complete stages and must reproduce the exact G85 n600 base components to
reported precision. Dense costates are batch-ephemeral and never persisted.
The 50-GiB SSD reserve is rechecked on every invocation.

An internally valid checkpoint self-hash is necessary but not sufficient for
resume. Before any existing batch is skipped, V2 freshly reopens the source,
receiver, G78/G87 state, and proposals; rederives the ordered deterministic
physical groups for that exact pair range; and requires exact agreement with
the checkpoint's expected IDs and count, projection-row IDs/order/count,
exact-replay policy and coverage, basis IDs, and replay-state-custody IDs.
The immutable blocker frontier uses this same validator. A regression seals a
self-consistent seven-of-eight forged checkpoint and proves that resume rejects
it against the freshly expected eight groups.

## Review-closed preflight

The current review preflight is:

- path:
  `/Volumes/VertigoDataTier/pact/taskspace_exact_coarse_costates_v2_n600_20260727_r5/00_preflight_receipt.json`
- file SHA:
  `f03135a848a5b7ddeb295544704210acb38da54de8258eb8764d7e6ab80ca1b1`;
- canonical preflight SHA:
  `bcdb09fe320c96d55939cad6af656a96a9fa0443b9e7269a66e5ca64b3ae49f8`;
- observed free SSD bytes: `349042044928`
- peak preflight RSS: `853 MiB`
- status: `preflight_complete_launch_not_executed`

Earlier r1/r2/r3/r4 preflights are preserved but superseded. R2 added
inference-authority Pose targets/base MSE, full base aggregate closure, and a
V2 resume frontier; r3 removes the false fixed-eight assumption and binds the
variable deterministic physical-group set and actual checkpoint-derived counts;
r4 makes source-rederived physical-group identity, ordering, and coverage
mandatory before any checkpoint can be skipped; r5 binds the same contract to
the final Ruff-formatted V1/V2 source set after 17 focused tests, lint,
format-check, compile, and two clean review-tracker passes.

## Command awaiting review

This command has **not** been executed:

```bash
.venv/bin/python tools/safe_run.py \
  --rss-mb 49152 \
  --projected-gib 40 \
  --timeout 7200 \
  --label g90_v2_exact_all_physical_groups_stage0_r1 \
  -- \
  .venv/bin/python tools/materialize_taskspace_exact_coarse_costates_v2.py \
  .omx/research/configs/taskspace_exact_coarse_costates_v2_n600_20260727_r5.json \
  --materialize-next-stage
```

It materializes or resumes exactly one 120-pair stage. It does not launch
stages 1–4 automatically.

## Pointer truth and next gate

The effective frontier remains `0.172`. V2 preflight is not score progress and
no archive exists. After review, stage 0 may map the exact-all coarse surface.
The next admission-relevant object is not a V2 row: it is a hierarchical
refinement that reproduces a chosen exact Y1, a conditional Pose solve at that
final pair state, a G94 receiver-closed sequential product, an actual ZIP byte
delta, and full n600 upstream replay.
