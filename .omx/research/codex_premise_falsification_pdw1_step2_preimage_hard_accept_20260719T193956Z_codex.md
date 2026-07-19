# Codex Premise Falsification — PDW1 STEP-2 exact-target preimage hard accept

Date: 2026-07-19T19:39:56Z  
Lane: `pdw1_dB_attack_step2_20260719T192712Z`  
Axis: `[macOS-CPU advisory]`  
Authority: `research_only=true`, `score_claim=false`, pointer `0.1910828242 [contest-CPU Linux x86_64] UNMOVED`

## Premise checked before implementation

The delegated attack describes
`DisjointResizeOperator.solve_uint8` as choosing a bounded integer preimage that
satisfies both `c^T z = T` and a frozen-SegNet hard-accept predicate. Source
inspection falsifies that description:

- `solve_uint8` and `solve_bounded_integer_block` certify only the affine
  integer equation and uint8 bounds. Neither accepts nor invokes a SegNet
  oracle.
- `bounded_continuous_preimage(target, reference=...)` can aim the real-valued
  affine projection at an unrounded reference, but this preference is not part
  of the counted PDW1P receiver grammar.
- `tac.optimization.tie_aware_preimage` already proves that the canonical
  support-fill reproduces the pre-rounded uint8 scorer plane exactly. Changing
  the scorer target toward the unrounded reference is therefore a
  target-selection payload change, not an exact-target preimage choice.

## Fresh measured canary

On real pair 0, through the frozen CPU-Torch SegNet and the production
factor-2 resize geometry:

- canonical PDW1P plane: 3,631 wrong argmax pixels;
- exact alternative constructed by
  `bounded_continuous_preimage(plane, reference=gt_f1)` plus
  `solve_bounded_integer_block` on all 3,631 flip cells: 3,631 wrong pixels;
- 10,893/10,893 channel blocks were `FEASIBLE_EXACT`;
- 40,203 camera values changed;
- the exact rational resize numerators remained bit-identical to the target.

This is a one-pair canary, not the delegated authority row. The landing must
measure n24 and a larger subset, count peak RSS, and distinguish:

1. affine Diophantine infeasibility (`c^T z = T` has no uint8 solution), from
2. hard-oracle incompatibility under the specified exact-target,
   source-reference-aimed policy, from
3. an unmeasured exhaustive claim over every exact preimage.

## Verdict scope and next implementation

`FORMULATION` only. The source inspection falsifies the claim that the existing
solver already composes affine feasibility with SegNet hard acceptance. It does
not kill target selection, sparse repair payloads, or the realization family.

The bounded next action is a Phase C2 diagnostic in the recovered driver. It
will compose the existing affine primitives exactly, re-score both canonical
and source-reference-aimed frames, record affine proof counts and the remaining
hard-oracle cells, and refuse an in-box/rate claim unless the counted receiver
bytes and parse-back actually close.

