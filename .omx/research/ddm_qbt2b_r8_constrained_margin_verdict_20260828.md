# ddm_qbt2b_r8 — 15,020-step constrained margin ENDPOINT VERDICT (counter 696)

Date: 2026-08-28. Owner: MAIN. Axis: **[macOS-MPS governed n32 frozen-scorer advisory]**,
score_claim=false everywhere in this memo. Continuation of
`ddm_qbt2b_r7_constrained_margin_verdict_20260828.md` (same basis, same constraint tuple);
config `AUTHORIZED_N32_R8_15020_20260828.json` (canonical 50ab690a…, identity 450f61d2…),
init = r7 stage-03 EMA endpoint (sha 9d28c75d…, 398,751 B). Run: counter 696, pid
82816/82826, rc=0, elapsed 30,594 s, all_payloads_retained=true, 15,000 constrained margin
rows retained in checkpoint-embedded history.

## 1. Headline

**The constrained-margin law scales: the half-r6 milestone is HIT, the descent law
STEEPENED, and the advisory gate gap collapsed 2.39× — no wall detected at 20k cumulative
steps.** The r7→r8 single-variable continuation (margin_steps 5,000→15,000 from the r7 EMA
endpoint; every mechanism pin identical) delivered:

- **Flip milestone HIT**: seg_expected_flip_realized tail-25 **0.004680** (tail-100
  0.004671, final 0.004685) vs the pre-registered 0.005 half-r6 target. The r7-law
  projection said 16,702 cumulative steps; observed crossing ≈19.7k — the r7 fit was
  predictive within ~18% on step count.
- **The power law STEEPENED, not flattened**: refit over the r8 window (250-step window
  means, cumulative 5,020→20,020) gives exponent **−0.9632** (coeff 68.417) vs r7's
  −0.7811. Box-class flip (0.00116) now projects at **~89,761 cumulative steps** (was
  ~108,412 under the r7 law). Windows/methods differ between the two fits, so the
  steepening magnitude is indicative; the SIGN (not flattening) is the finding.
- **Constraints held (tail-mean form, per the r7 letter-vs-substance lesson)**: Lane
  tail-25 0.119529 at the 0.12 bound (final 0.102346, max 0.4024 early transient, ZERO
  0.50-existence breaches in 15,000 steps); Movable tail-25 0.009225 chattering at the
  0.009 bound (final 0.005195). λ ceiling contact **0/15,000 both classes** (max λ_Lane
  0.1355, λ_Movable 0.1070 — LOWER than r7's maxima; the constraint is getting cheaper to
  hold as the field improves). Pre-registered falsifier NOT fired.
- **Pose improved 1.6× in-window**: pose_mse_realized tail-25 7.472e-4 (final 5.270e-4)
  vs r7's 1.188e-3. Still ~6× above the m110 shipped-d_pose budget 1.25e-4; descending
  jointly with seg — no separate pose leg fires (m110 ratio discipline unchanged).

## 2. Stage-05 admission gate (honest refusal, gap collapsed)

`GATE.json`: admitted=false, correctly. B_hat **122,325 B** (rate-feasible, 15,661 B
under the 137,986 B sub-0.12 demand). S_hat **0.6406** vs r7's 1.5304 (2.39×):
d_seg_hat 0.0045929 (r7 0.013135, 2.86× — the seg axis is the mover; now 3.96× from the
~0.00116 box need, was 11×) · d_pose_hat 9.965e-4 (r7 1.829e-3, 1.84×). Failed gates:
`s_hat`, `d_pose_hat`, `same_budget_qbw1_control` (REFUSED_MISSING).

**QBW1-control clause (recorded per the r7 routing, discharging the standing owed
control):** the stage-05 gate hard-requires a same-budget QBW1 control row, and none
exists — CORRECTLY, because the sealed QBW1 fire order
(`sealed_main_fire_order/FIRE_ORDER.json`, sha 5fd3e62e…) adjudicated its OWN launch as
FOLDED_BY_SCORER_FREE_GATE: QBW1's rate estimate 336,286 B misses the 137,986 B cap by
251,376 B (rate term alone 0.2239 > the whole 0.12 target), arm_must_not_launch=true.
The control is **discharged by refutation** — the gate's REFUSED_MISSING is the fire
order's refusal outcome, not an orphaned pending obligation. No Modal spend; no launch.

## 3. Apparatus results this unit (both structural, both landed)

1. **Checkpoint-cadence law (84514376ee).** The r8 compile was first refused honestly:
   at r7's every-5 cadence, 15,020 steps project 146.4 GB of retained checkpoints
   (3,004 copies × worst-case final size — history rides inside every checkpoint at
   ~2,925 B/step) vs 50.0 GB free. Cure = a DERIVED retention law, not a constant swap:
   validator bound `checkpoint_every_steps ≤ max(5, steps // 300)` (≥300 periodic saves
   per run ⇒ worst-case crash loss ≤ ~0.33% of the run; ~102 s at the measured r8
   2.04 s/step). Retention-only by construction (saves gate save_checkpoint +
   reencode_inference_state exclusively; verdict cadence is a separate key; all stage-end
   saves unconditional). r8 ran at cadence 50 → 301 periodics → projection 15.67 GB,
   PASSED with 24.2 GB headroom. Side effect: the run came in at 2.04 s/step vs r7's
   2.51 (fewer save+re-encode events), 30,594 s total vs the 37,700 s projection.
2. **O(steps²) retention wall NAMED for the box chase.** Because every periodic
   checkpoint embeds the full history, retained bytes grow quadratically in steps across
   the set. At the ~90k-step box projection, final checkpoints are ~265 MB and even the
   cadence-law ceiling (steps//300) projects ~80+ GB worst-case — over AP's free space.
   The named engineering for r10+: periodic checkpoints drop embedded history (a separate
   append-only history sidecar becomes the SoT; stage-end checkpoints keep full history;
   resume reads the sidecar). Until that lands, continuations are budgeted to fit the
   conservative projection.

Instrument note (honest): the MAIN-armed Monitor emitted a FALSE "child dead without
done-receipt" at completion — it watched for a literal `QBT2B_R8_DONE` file in the run
dir while the launcher's receipt fired through the keeper watcher (ARM … FINISHED rc=0);
safe_run status shows child_exit_nonzero=false. Sibling of the #1306
instrument-blind genus (receipt-name/path mismatch between watcher layers); the keeper
watcher is the authoritative completion signal.

## 4. What r8 settles and what it opens

SETTLED (n32 seeded-stratified cohort, this basis):
- The constrained-margin law is stable across a 3× budget extension: constraint cost
  FALLS as the field improves (λ maxima shrank r7→r8), no infeasibility signature.
- The 0.005 half-r6 milestone is reached un-plateaued; the r7 trajectory fit was
  predictive (~18% step error) and the refit steepened.
- The vehicle remains rate-feasible for the sub-0.12 box (122,325 B, 15,661 B spare).

OPEN:
- Exponent-stability evidence stands at 2 of the 3 pre-registered doublings (5k→10k→20k;
  the third is 40k cumulative). The box-class need is ~4× below the current flip.
- d_pose_hat 9.97e-4 vs the 1.25e-4 budget — 8×, riding down jointly.
- The O(steps²) retention wall (item 3.2) gates any single run past ~30k cumulative.

ROUTING:
1. **r9 continuation** from the r8 stage-03 EMA endpoint, margin_steps 10,000 (→30,020
   cumulative, ~5.7 h at measured pace, storage-safe under the conservative projection):
   buys the next half-doubling of exponent evidence + projected flip ~0.0033.
2. **History-slimming engineering** (item 3.2) built during/after the r9 window — the
   prerequisite for the ~90k box chase or any n600-scale decision run.
3. n600-scale decision at the r9 endpoint with the accumulated curvature evidence.

## 5. Lane closure

Terminal rows appended for `ddm_qbt2b_r8_scorer_20260828` + `ddm_qbt2b_r8_metal_20260828`
(completed). Own-vehicle frontier UNMOVED this unit: the exact pointer stands at gb1
S 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600] — this unit produced a scaling
confirmation of the constrained-training law + an honest advisory-gate refusal, not an
exact row.

---

## ADDENDUM (ddm_eq1, 2026-09-04) — the equations leg

**Law:** `trajectory_derived_stopping_law_v1` — `tac.canonical_equations.trajectory_derived_stopping_20260805` (`tac.canonical_equations`). **Relation:** IN-DOMAIN (trajectory segment, first doubling: 15,020 steps).

Continuation of r7 on the same basis and constraint tuple; its endpoint is the second point the segment exponent is fitted through.

This memo's Catalog #344 trigger was the word **stratified** — `"ratified"` is a substring of it, and the gate matched plainly. MEASURED by this arm: 16 of the 29 live memos (55.2%) tripped the gate ONLY that way, i.e. the gate was flagging the memos that did their sampling right. Fixed in the same batch (`(?<!st)ratified`); the disposition above stands on its own merit, not on the misfire.
