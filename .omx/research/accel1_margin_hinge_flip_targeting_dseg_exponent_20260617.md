# ACCELERATOR PROBE 1 — margin-hinge flip-targeting seg loss: does it BEND the d_seg power-law exponent? (2026-06-17)

**Authority:** `[contest-CPU advisory]` NON-PROMOTABLE. All d_seg numbers are the
EXACT contest argmax-flip rate (`(SegNet(decoded).argmax != SegNet(GT).argmax).mean()`
on the last frame, via the real frozen `RealScorerContext` on CPU — NO MPS). This is a
$0 small-n overfit *mechanism* diagnostic, not a score claim. READ-ONLY on the live
5-day basin (deepcopy-per-arm init; nothing perturbed).

**Lane:** `lane_accel1_margin_hinge_flip_targeting_20260617`.

**Question (the top lever).** Probe C proved the d_seg "wall" is REAL (not an
EMA-shadow artifact) and that an epochs-only descent projects ~2.5–3.7× ABOVE the
sub-0.15 d_seg target 0.000322, because the CE / soft-cosine seg-loss gradient VANISHES
exactly at the residual argmax-flips. Does a FLIP-TARGETING margin-hinge — which puts ALL
gradient on the flip set and ZERO on correct-with-margin interior pixels — BEND the
d_seg-vs-step power-law exponent enough to overturn that projection?

---

## The loss (built + wired, reusable)

`tac.losses.core.segnet_margin_hinge_per_pixel(pred_logits, gt_hard, margin_target)`:

    L(pixel) = max(0, margin_target − (logit[GT] − max_{c≠GT} logit[c]))

A per-pixel multiclass hinge restricted to "GT vs its strongest competitor" — the exact
binary margin the contest d_seg argmax flips on. It is defined on the RAW logits (the
contest d_seg is a hard argmax; no softmax/temperature to tune). Geometry:

- `g = logit[GT] − max_{c≠GT} logit[c]` is the signed margin: `g>0` ⇔ correct, `g<0` ⇔ FLIPPED.
- `L = relu(margin_target − g)`: **ZERO** when `g ≥ margin_target` (correct WITH margin →
  no wasted gradient on safe interior pixels); **linear slope −1** below the target → the
  SAME, MAXIMAL constant-magnitude pull on every flip / near-flip.

Wiring: `seg_surrogate="margin_hinge"` in `StageSpec` (+ `seg_margin_hinge_target`,
`road_lane_emphasis`) routes through the SAME driver `_seg_loss_for_spec` the real trainer
runs — so a winning verdict drops straight into the curriculum. Probe E found 64% of flips
are road↔lane (classes 0↔1); `road_lane_emphasis>1` (via
`road_lane_emphasis_class_weights`) concentrates the hinge there. Default path is
byte-identical (verified: `test_seg_surrogate_lever.py` 12/12 still green).

### The decisive gradient property (unit-tested, deterministic)

On a **confident flip** (the residual d_seg pixels Probe C said CE/soft-cosine cannot
fix), with the GT-class logit getting the gradient:

| loss | gradient magnitude on a confident flip |
|------|----------------------------------------:|
| margin-hinge | **1.0** (constant — full pull, independent of flip depth) |
| soft-cosine  | **~1.9e-22** (VANISHED — Probe C's root cause) |
| CE | ~1.0 on the flip, but ALSO nonzero on confident-correct interior (wasted) |

This is the mechanism the probe tests empirically: does that constant flip-pull translate
into a steeper d_seg descent? (`src/tac/tests/test_segnet_margin_hinge_loss.py`, 15 tests:
zero-on-correct, value=target−margin on flips, correct grad sign, constant-magnitude
across flip depths, non-vanish vs soft-cosine, dispatcher routing, no pred mutation.)

---

## Method (apples-to-apples, $0 CPU)

From the base_ch20 basin forkpoint (`basin_bc20_20260612T121523Z`, the EMA shadow =
inference/score state). Per arm: deepcopy the identical decoder+latent init, fresh AdamW,
same LR / seg_weight=100 / pose isolated (pose_weight=0) / same seed / same step count.
The seg-loss FUNCTION (+ config) is the ONLY variable. Record the full exact-d_seg(step)
trajectory; fit `d_seg = A·step^(−p)` (full + late-half). Higher `p` = faster descent =
the wall BENDS. Primary regime = `random` init (high-flip start ≈ the from-0 dynamic of
driving d_seg DOWN; genuine flips to fix). 4 arms: A=CE, B=soft_cosine(T=0.3),
C=margin_hinge, C'=margin_hinge+road↔lane.

---

## RESULT (random regime, REPAIR; 6 pairs, 120 steps, seed 0; CPU authority)

`.omx/research/accel1_margin_hinge_exponent_random_20260617.json`. Every arm from the
SAME init (d_seg start 0.04293), same LR/seg_weight/steps/seed; loss is the only variable.

| arm | d_seg_after | p(full) | **p(late)** | d_seg(50k)* | grad-norm@step1 | fit r² |
|-----|------------:|--------:|------------:|-----------:|----------------:|-------:|
| A · CE | 0.00142 | 0.934 | **0.608** | 4.5e-6 | 111 | 0.978 |
| B · soft_cosine (T=0.3) | 0.00407 | 1.034 | **0.646** | 7.2e-6 | 23 | 0.891 |
| **C · margin_hinge** | **0.00120** | 1.052 | **0.787** | **1.9e-6** | 125 | **0.992** |
| C' · margin_hinge + road↔lane | 0.00141 | 0.998 | **0.687** | 3.1e-6 | 149 | 0.987 |

\* d_seg(50k) is the power-law extrapolation; see the caveat below — the ABSOLUTE
projection is NOT trustworthy off a small-n overfit, the RELATIVE exponent ordering is.

**The exponent BENDS.** The trustworthy signal is the **late-window exponent** (the
steady-state descent rate; the full-window p is inflated by the steep early-overfit phase
the converged 5-day run does NOT have). The margin-hinge late exponent **0.787** is a clear
bend above CE **0.608** (+0.18) and soft_cosine **0.646** (+0.14). The bare hinge also:
- reached the LOWEST d_seg (0.00120 < CE 0.00142 < soft_cosine 0.00407) in identical compute;
- has the CLEANEST power-law fit (r² 0.992) — a steeper, straighter descent, not noise;
- preserved a HIGH gradient norm (125, ≈CE's 111; soft_cosine collapsed to 23) — the
  mechanism is exactly the non-vanishing flip-pull (soft_cosine's low grad-norm IS its
  gradient-vanish; it removed the FEWEST flips: 0.00407, ~3× CE's residual).

**soft_cosine is the WORST arm** — confirming Probe C's diagnosis from the opposite side:
the current oomph lever's gradient vanishes on the flips, so even at its alive sweet spot
(T=0.3) it leaves ~3× more residual flips than CE and bends the exponent LEAST.

**road↔lane emphasis HURT the bare hinge here** (p_late 0.687 < 0.787; d_seg 0.00141 >
0.00120). On this small slice over-weighting road/lane (64% of flips, but the other 36% are
real too) traded away the broad-flip gradient. The class emphasis is a knob to sweep at
scale, not an automatic win; the bare hinge is the clean lever.

---

## VERDICT

**The flip-targeting margin-hinge BENDS the d_seg power-law exponent — the prize the probe
was after. PROCEED to wire it into the curriculum seg loss as the d_seg lever.** It is the
only arm that both (a) lowered the steady-state exponent's deficit (p_late 0.787 vs the
soft_cosine lever's 0.646 and CE's 0.608) AND (b) reached the lowest residual d_seg in
matched compute, with the cleanest fit and a non-vanishing gradient. The mechanism is the
one Probe C named: constant pull on the flip set (grad ≈ −1 even on confident flips) vs
soft_cosine's vanishing pull (grad ≈ 1e-22) — measured directly in the unit tests.

**Does it overturn Probe C's "sub-0.15 is d_seg-infeasible under epochs alone"? PARTIALLY,
and HONESTLY not on the absolute number.** The probe's auto-label
"OVERTURNS…SOME_ARM_REACHES_SUB015_DSEG" is OVER-STRONG and must NOT be quoted as a score
claim: a 6-pair / 120-step OVERFIT produces inflated absolute exponents (0.6–1.0 here vs the
canonical 5-day basin's 0.19–0.35 in Probe C), so EVERY arm's d_seg(50k) extrapolation lands
~1e-6 — an artifact of small-n memorization, not a 600-pair contest projection. What IS
trustworthy is the **relative, apples-to-apples ordering under identical compute**: the
hinge's steady-state exponent is ~1.2–1.3× CE's and ~1.2× soft_cosine's. Carrying that bend
factor onto Probe C's canonical late exponent (basin p≈0.19) gives p≈0.24–0.25 — which moves
the d_seg(50k) projection toward, but does not by itself clear, the sub-0.15 target. **The
honest claim: the hinge is a real, measured accelerant of the d_seg descent that soft_cosine
(the current lever) is not; it is NOT a proven standalone path to sub-0.15.** The decisive
test is now a 600-pair from-basin A/B (hinge vs the live soft_cosine lever) measuring the
late exponent on the REAL operating point — exactly the next byte-closed campaign gate.

**NEXT (the campaign gate this unlocks):** run the live curriculum with
`seg_surrogate="margin_hinge"` (it is wired; default path byte-identical) as an A/B against
the current soft_cosine stage at 600 pairs from the basin, comparing the late-window d_seg
exponent + residual d_seg at matched epochs. If the bend survives at 600 pairs, the hinge
replaces soft_cosine as the d_seg lever and re-opens epochs as a sub-0.15 path; if it
collapses to the soft_cosine floor, the wall is capacity (per the symposium) and the
byte-neutral structural levers (taper realloc / KD warm-start) remain primary.

## Wire-in (6-hook)
1. sensitivity-map: N/A (the loss IS a gradient-concentration map; no per-byte map).
2. Pareto: informs the d_seg-vs-epochs constraint (if the hinge bends p, epochs become a
   live lever again vs the byte-realloc-dominates conclusion).
3. bit-allocator: N/A.
4. cathedral autopilot: N/A (research diagnostic + a curriculum-ready loss lever).
5. continual-learning posterior: this memo + JSON; updates the symposium
   "d_seg-infeasible-under-epochs" prior conditional on the loss used.
6. probe-disambiguator: this IS the disambiguator (loss-bound floor vs capacity floor;
   flip-targeting vs gradient-vanishing loss).
