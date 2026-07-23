---
title: Codex findings - DDM v19b joint remeasurement stack
date_utc: 2026-07-23
lane_id: ddm_v19b_joint_remeasure_stack
research_only: true
execution_allowed: false
score_claim: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
verdict: MULTI_MOVE_JOINT_STACK_ADMITTED_N600_ADVISORY
verdict_scope: "INSTANCE:V19B only; no contest-axis, family, score, or promotion verdict"
pointer_moved: false
main_landing_review_required: true
---

# Finding

The v19 non-additivity guard was necessary but conservative: all ten admitted
single-step moves remain strictly improving under greedy common-master
remeasurement. The measured n600 stack is 137,825 B with
`d_seg=0.026594424778`, `d_pose=163.061176604795`, and
`delta_S=-0.08501960537266746` versus the exact v15 control.

The result is not Lane+Movable-only. Of 103,322 net Seg flips, 29,377 are in
Lane+Movable and 73,945 are in Road+Undrivable+MyCar. c1 and #366 must consume
this exact residual state rather than assigning the correction line only to the
role bucket.

# Bugs found and extincted

1. A first draft reused v19's track-bound helper on the direct G1 lift even
   though that helper expects the higher joint-lift wrapper. The runner now has
   a direct-G1 bounds function over the same polygon records.
2. A first draft represented each grammar candidate as a global track shift.
   Source v19 actually shifts only tracks feasible for the candidate sign.
   The final compiler preserves those feasible subsets per move and adds
   translations per track, refusing cumulative bounds escape.
3. The v15 ladder stores class counts as `per_stratum`, while later v16/v19
   receipts use `per_role`. The control binder now explicitly maps sealed
   `per_stratum.{Lane,Movable}` rows instead of silently assuming schema
   continuity.

All three failures occurred before a false verdict could be written. Completed
candidate/scorer checkpoints were reused; no measurement was fabricated or
lost.

# Non-additivity disposition

The nine conditional moves preserve 95.9616459% of their original single-step
gain before amplification. Measured amplification contributes an additional
0.0804967212 score units. The final DEV cumulative delta is
`-0.12117433392376804`; n64 is `-0.15747359282201992`; n600 is
`-0.08501960537266746`.

# c1 division of labor

v19b owns the ten-move REALIZE correction line. Its exact n600 output costs
3,884 B over v15 and realizes 26.6019567456 net flips/B. It leaves 3,000,367
errors above the c1 target. v18b and #366/J3 own the residual finish under their
two 16,384-byte budgets, with sequential exact replay and no additive credit.

The operator's atom-order gauge was measured at the present coding surface:
140 B as emitted and 140 B under canonical ordering. The current fixed-width
ZIP-stored payload has no order-sensitive rate actuator. c1 CODE may reopen the
lever only with an actual delta/entropy coder, placement-index remap, and exact
camera identity proof. No scorer permutation or training-landscape claim was
imported.

# Custody

- final receipt SHA:
  `4bb5d6b4b793b667c7cbe15e37cbf9a27f6c0e75451374839fb5df8ca1c1b8e8`
- final n600 archive SHA:
  `74ede4194ed0eb6c2716f1033a47569c87a7afdf0c21fb1ee23ac059beff3aae`
- n600 batch chain:
  `49248868df850d0ce55eb17f0ff0fd386798906db433a3b02ef4d7605eadeb24`
- axis: `[macOS-CPU frozen-scorer advisory]`
- `score_claim=false`
- pointer: `0.1910828242 [contest-CPU]`, unmoved

# MAIN review required

Do not land from the headline alone. MAIN must re-derive source custody,
common-master merge semantics, all ten strict incremental decisions, n64/n600
batch coverage, role/residual counts, c1 budget arithmetic, false-authority
labels, and pointer immobility.

The v19b lane is registered at L2 with `impl_complete` and
`real_archive_empirical`. Global `lane_maturity validate` remains blocked by
110 pre-existing missing legacy evidence paths; none names this lane. That
repository-wide debt is separately scoped and was not rewritten here.
