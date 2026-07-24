---
research_only: true
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
score_claim: false
pointer: "0.1910828242 [contest-CPU]"
pointer_moved: false
main_review_required: true
---

# DDM WS3 canonical-equation consumption note

No new equation ID is introduced. This lane consumes the existing contest
action and registered warm-start falsifier.

## Exact realized action

For exact archive bytes `B` and frozen-scorer, realized-through-R components:

`S = 100*d_seg + sqrt(10*d_pose) + 25*B/37_545_489`.

Every reported proposal delta is decomposed as:

`delta S = 100*delta d_seg
         + [sqrt(10*d_pose_new) - sqrt(10*d_pose_ref)]
         + 25*delta B/37_545_489`.

The opening campaign policy admits a proposal only when all of the following
are true on exact n600 receiver replay:

1. strict `delta S < 0`;
2. the component classifier is one of `REALIZED_STAGE_TARGET_MET`,
   `REALIZED_STAGE_DESCENT_CONTINUE`, or
   `REALIZED_STAGE_SEG_FLAT_POSE_DESCENT_CONTINUE`;
3. the cumulative fire gate versus stage 0 is green, including the typed
   residual-bucket descent requirement when enabled.

Proxy Seg ordering chooses which candidate is measured first. It never replaces
those exact acceptance conditions.

## Registered slope falsifier

The unchanged registered critical ratio is:

`R* = 4.1215446777965665`.

For the exact terminal W_seg proposal:

- pose-term progress per step:
  `P_seg = 0.0028920699387597892`;
- Seg regression per step:
  `G_seg = 0.002464294433594194`;
- observed ratio:
  `P_seg/G_seg = 1.1735894458608507 < R*`;
- predicted pose repayment:
  `6611.801236822376` steps;
- predicted Seg-advantage exhaustion:
  `1882.677674578264` steps.

The registered callable therefore returns `KEEP_WJOINT` with reason
`SEG_REGRESSION`. Its scope is the W_seg reformed-opening FORMULATION versus
this exact W_joint INSTANCE, not a family or paradigm verdict.

## W_joint exact window

Exact history is `[0,1,2,3,4]`. Across that window:

- Seg-term delta per step: `-0.007588916354709152`;
- pose-term progress per step: `0.015854339538902806`;
- total distortion-term delta: `-0.09377302357444606`.

Rate is held separately in the exact archive/action receipts. No macOS advisory
row moves `0.1910828242 [contest-CPU]`, and no row is a score claim.

## Late scorer-recursion directive

The operator directive at `2026-07-24T14:45:16Z` changes the authority of
proposal construction, not these measured equations: a generic proposal menu
is a `[naive-menu upper bound]` unless it is derived from recursive scorer
factorization. Consequently an exact negative delta from the selected generic
menu is necessary evidence but no longer sufficient evidence for a fire-ready
verdict.
