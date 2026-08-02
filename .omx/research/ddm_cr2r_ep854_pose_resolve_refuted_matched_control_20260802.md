# ddm_cr2r — the ep854 pose re-solve is REFUTED, and the control proves it is the BASE

**Date:** 2026-08-02 · **Scope:** FORMULATION (post-hoc pose solve on a seg-only-burned base)
· **Cost:** $0 — the control was already on disk · **Tasks:** #827 (closed), #881, #889, #890

## The claim under test

Task #881/#827 named one blocker between us and a measured **−0.0866789 S** seg+rate win:
the cr2 composition (ep854 seg base × cell_drop50 rate) shipped a *transplanted* pose field that
measured `d_pose 37.877` against a `break_even_d_pose 0.0131903119638695` — 2,871× over, so cr2's
own falsifier fired. Its recorded next action was *"the pose must be re-solved against ep854 first."*

That re-solve was fired (`ddm_v4c_resolve.py --mode solve --base ep854 --n-pairs 600 --resume`,
ep854 added to `BASES`, base sha256 `fd50925899b2…4904`). It ran 77 of 600 pairs.

## The control (this is the whole finding)

`ddm_v4c_resolve.py --mode solve` had already been run on the **celldrop50** base by a prior arm.
Same tool, same code path, same fields, 250 rows on disk. The only variable is which base archive
the pose is solved *against*. On the **74 pairs both runs cover**:

| same solver, 74 matched pairs | mean `d_best_static` | median | 
|---|---:|---:|
| celldrop50 base (control) | **0.0778** | 0.0029 |
| ep854 base (live) | **11.5904** | 2.5308 |
| ep854 better on | **1 / 74 (1.4%)** | |

Full-population context: celldrop50 solve-stage mean **0.0273** over 250 rows, and its photo stage
refines all 600 to `d_rungAB` mean **0.00953**. **The solver is healthy.** The defect is the base.

## The arithmetic that closes it

- break-even `d_pose = 0.0132` → contribution `√(10·d) = 0.363`
- **Floor**, assuming the 526 unsolved pairs return *exactly zero*: mean₆₀₀ ≥ 1.6313 →
  contribution ≥ **4.0389 S**, against **−0.0867 S** to defend → **46× over at the floor.**
- The hardest-first ordering (by Knee-A ship `d`) measured **Pearson r = −0.016** against ep854
  solve difficulty — *uninformative for this base*. The unsolved remainder is therefore ~random,
  giving an expected mean ≈ 12.8 → contribution ≈ **11.3 S**.
- Even a hypothetical **100×** photo-stage improvement (celldrop50's measured photo gain is 2.8×)
  leaves 11.6 → 0.116 → **1.08 S**, still 12× over.

No completion of the run changes the verdict. Stopping it is the correct call; the remaining
~3.5 h buys only wall-characterization, which the GOAL section explicitly forbids as means-hoarding.

## What is and is not falsified

- **REFUTED (INSTANCE):** the v4c static two-plane pose re-solve rescues the cr2 composition.
- **REFUTED (FORMULATION):** post-hoc pose solving on a seg-only-burned base. A *fresh full
  re-solve* improves on the transplant 37.877 → 11.59 (3.3×) and is still ~880× above break-even.
- **NOT refuted (FAMILY):** pose on ep854 as such. Joint/in-loop pose descent during the burn is
  untested on this base and is the named alternative — consistent with the standing clarification
  that on this vehicle only JOINT descent crosses the photometric wall.
- **The −0.0867 S seg+rate half remains real.** It needs a *pose-carrying* base, not a better solver.

## The law this re-confirms, on a new vehicle

**Seg-only training spends pose legibility** (#889) was measured on the v9/v10 witness. It now holds
on the TR1 burn base. This is a re-anchor on a new vehicle, not a new discovery — but the matched
control is new, and it converts the law from an inference into a base-level A/B.

## The cheap guard this buys (free, mandatory going forward)

Before composing any seg base with any pose stream: run the matched-control solve on ≥32 pairs of
**both** bases and refuse the composition if the ratio exceeds ~10×. Cost is minutes; it would have
retired this whole 3.7 h run at the 32-pair mark. Sister of the existing degenerate-baseline control
debt (#833) — the same defect class (no control ⇒ a base property reads as a solver property).

## Method note

The control cost $0 because a sister arm had already run the identical stage on a different base and
left the JSONL on the SSD tier. Checking for an existing matched run *before* interpreting a partial
result is the reusable move here — not a new measurement, a join.

**Evidence:** `/Volumes/VertigoDataTier/pact/ddm_v4c_20260730/solve_ep854.partial.jsonl` ·
`solve_celldrop50.partial.jsonl` · `photo_celldrop50_resolve.partial.jsonl` ·
`/Volumes/VertigoDataTier/pact/ddm_cr2_20260801/ddm_cr2_receipt.json`
