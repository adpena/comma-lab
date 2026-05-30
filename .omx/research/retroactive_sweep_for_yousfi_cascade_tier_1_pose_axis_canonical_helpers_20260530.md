# Retroactive sweep — Yousfi-cascade tier-1 pose-axis canonical helpers landing 2026-05-30

**Lane**: `lane_yousfi_cascade_tier_1_pose_axis_canonical_helpers_20260530`
**Subagent**: `yousfi_cascade_tier_1_pose_axis_20260530_194313`

This retroactive sweep is per CLAUDE.md Catalog #348 (event-driven
retroactive verdict-taint sweep) + Catalog #208 docs path-discipline
discipline. The landing introduces 3 NEW canonical equations + 3 NEW
canonical helper packages but does NOT introduce any NEW STRICT
preflight gate, so Catalog #348's gate-introduction trigger is N/A.
However the operator's standing directive ("retroactive sweep on every
landing wave") prompts us to verify no historical KILL / DEFER /
FALSIFY verdicts are invalidated by the new helpers.

## Bug-class symptom signature

The new pose-axis canonical helpers close 3 GAPS in the existing
canonical apparatus:

1. **Per-pair pose-vulnerability map gap**: pre-landing the canonical
   master-gradient ledger surfaced per-pair tensors of shape
   ``(N_bytes, N_pairs, 3)`` but had NO canonical helper that
   classified pairs into VULNERABLE/MIDRANGE/NULL pose-axis buckets.
   Sister bolt-ons (PR110-OPT-6 / PR110-OPT-7) had to re-discover the
   classification each time they consumed the per-pair tensor.
2. **PoseNet surrogate canonical-helper gap**: pre-landing the
   canonical Hinton-distilled surrogate package existed only for
   SegNet at ``tac.residual_basis.hinton_distilled_scorer_surrogate``.
   PoseNet had TRAINING primitives in
   ``tac.substrates.hinton_distilled_scorer_surrogate.LearnablePoseStudentHead``
   but no deployable surrogate WRAPPER for per-byte gradient
   extraction.
3. **YUV6 chroma-subsampled perturbation operator gap**: pre-landing
   there was NO canonical operator targeting the canonical
   Fridrich-Yousfi blind spot in PoseNet's FastViT-T12 attention head
   (the 4:2:0 chroma-subsampled axis).

## Pre-fix window

  * NSCS06 v6 chroma replication (Y=R=G=B) attacks landed BEFORE the
    canonical perturbation operator existed; the lane verdict (per
    `feedback_optimize_iterate_highest_ev_boldest_individually_fractally_optimized_mlx_deployed_aggressive_frontier_breaking_no_fake_implementations_standing_directive_20260529.md`)
    documents the structural surrogate that we now close.
  * PR110-OPT-6 / PR110-OPT-7 / Slot RR pose-axis null-projection: all
    operate WITHOUT a canonical per-pair vulnerability prior + WITHOUT
    a canonical chroma-subsampled operator. The landings remain valid;
    the new canonical helpers compose with their existing surfaces
    (we did not modify any sister code).

## Historical KILL / DEFER / FALSIFY search

Grep query on the active canonical posterior at
`.omx/state/probe_outcomes.jsonl` + `.omx/state/canonical_equations_registry.jsonl`
+ memory MEMORY.md for "pose_vulnerability" / "posenet_surrogate" /
"chroma_subsampled" yields NO prior FALSIFIED / KILLED verdicts that
this landing invalidates. The new helpers are additive (foundation
operators); no sister DEFER unblocked by this landing requires
reactivation.

The closest sister verdicts (Slot GGG canonical-quadruple +
Slot AAA MiPOD + Slot CCC HUGO + Slot QQ Cauchy-Schwarz META-LIFT-1+2)
all explicitly document their non-overlap with the pose-axis surface;
the new landing extends the canonical scope to pose-axis at the helper
layer + leaves their existing scope intact.

## Per-finding RE-EVAL priority assignment

| Sister landing | This landing impact | RE-EVAL priority |
|---|---|---|
| PR110-OPT-6 motion pair repair / pose-axis null projection | Provides per-pair selection prior + chroma operator (composable) | LOW (additive; no verdict invalidation) |
| PR110-OPT-7 Fridrich UNIWARD inverse-scorer basis | Provides per-pair selection prior + chroma operator (composable) | LOW (additive; no verdict invalidation) |
| Slot RR pose-axis null-projection cascade | Provides REAL PoseNet surrogate sister for cheap per-byte FD | MEDIUM (operator can re-eval pair selection via canonical helper instead of menu-size constants) |
| Slot GGG canonical-quadruple-binding | DISJOINT scope (Z8 hierarchical predictive coding); no overlap | NONE |
| Slot AAA MiPOD / Slot CCC HUGO / Slot YY HILL / Slot FF UNIWARD | Provides canonical 600-pair fp64 per-pair selection prior for the seg-axis cascades to compose with | LOW (additive) |
| Slot EEE 6-axis audit | This landing PASSES Axis F (cite-vs-impl): all 3 deliverables genuinely implement their claimed techniques; verified via 81/81 tests | NONE (audit-class) |

## No invalidated verdicts

This landing does not invalidate any prior KILL / DEFER / FALSIFY
verdict. It closes 3 canonical-helper GAPS at the cathedral autopilot
auto-discovery surface (per Catalog #335) + the canonical-equations
registry surface (per Catalog #344). All sister bolt-ons that
PREVIOUSLY hand-rolled per-pair vulnerability classification or
chroma-subsampled perturbation can NOW route through the canonical
helpers per UNIQUE-AND-COMPLETE-PER-METHOD discipline.

## Cross-references

  * Canonical equations registered: `per_pair_pose_vulnerability_map_yousfi_alaska_analog_v1`,
    `posenet_mae_v_hinton_distilled_surrogate_mlx_local_v1`,
    `yuv6_chroma_subsampled_perturbation_yousfi_blind_spot_exploit_v1`.
  * Lane registry: `lane_yousfi_cascade_tier_1_pose_axis_canonical_helpers_20260530` (umbrella)
    + 3 deliverable sub-lanes.
  * Sister canonical packages NOT modified:
    `tac.substrates.hinton_distilled_scorer_surrogate` (training primitives),
    `tac.differentiable_eval_roundtrip` (canonical YUV6 transform),
    `tac.master_gradient` (canonical aggregate + per-pair gradient surface).
