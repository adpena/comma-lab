# Closure re-audit ROUND 2 — synthesis + MY adversarial audit of both re-auditors

**Operator: "adversarial audit all results coming back + review all negative/deferred/killed + another
math/algebra/geometry/calculus pass."** Two re-audit adversaries returned (RA-1 session negatives,
RA-2 historical ledger). Per the standing discipline I adversarially AUDITED each before acting — and
**rejected two of their headline claims.** All `[advisory]`; exact pointer UNMOVED at 0.19110.

## The trigger + the meta-pattern (RA-1's gold insight, AUDITED-SOUND)
The #1 pose-low-rank canary (my falsification at the wrong fidelity) generalizes: **closures grounded in
operating-point-INVARIANT quantities held (entropy floor, geometric smear-wall); closures grounded in a
CHOICE of region / model-class / fidelity / config are where the errors concentrate.** RA-2's structural
finding: the apparatus already KNOWS it over-killed (28-32/34 verdicts fail today's rigor) and the registry
is CLEAN of mislabeled paradigm-kills — so the live fallibility is **un-executed reactivations**, not
rotting labels. The canary sat at "Theory / never byte-closed" 6 weeks.

## AUDITED reopens (ranked by MY audited score-EV)

### 1. FP-shrink QAT rate lever — TOP (convergent RA-1#5 + RA-2 R-2; AUDITED-STRONG)
Both auditors independently land here. apogee_int4/FP4 was killed on NAIVE-PTQ (a memorized-frontier
operating point), QAT/LSQ never tried; the frontier-rate-cut closure (SOUND on the lossless floor)
explicitly surfaced FP-shrink as the real lever. Napkins: FP11→FP8 holding d_seg → ~133KB → S≈0.162
(−0.029); the bc20 thesis names FP4 as −0.022 rate headroom. **This is the single biggest reopened
score-EV — larger than everything else combined.** $0 next: LSQ/QAT-fp4(or fp8) smoke on the bc20 basin
weights, measure whether d_seg/d_pose hold under the shrink. Task #136 (+ bc20 surface).

### 2. Power-law was the WRONG MODEL CLASS → the long-train sub-0.15 thesis is REOPENED (RA-1#2; AUDITED-STRONG)
A stretched-exponential `d=0.00566·exp(−(ep/4263)^0.860)` (the physically-correct annealing/glassy family)
fits the CE d_seg trajectory **16× better** (SSE 2.7e-8 vs power-law 4.3e-7). Projection diverges
catastrophically: power-law → sub-0.15 d_seg at ~999k ep (infeasible); stretched-exp → **~14.5k ep
(feasible, inside the 50k run's budget).** The symposium's "capacity/epochs-infeasible → don't launch the
long train" verdict (and R3's L1) RESTS ON THE WORSE MODEL. **This VINDICATES keeping the running
margin-hinge 50k run going** — and supersedes my own earlier "epochs-only won't reach sub-0.15" framing:
the better-fit model says it might, at ~14.5k ep. Honest caveat: still an extrapolation; defensible range
[14.5k, 999k] ep; the live run + the registered CE-control trigger is the real test.

### 3. Pose is ~1-DOF radial-zoom (RA-2 R-1 + #140; AUDITED-STRONG, deepens #140)
PoseNet Jacobian rank ≈1.008 (dim-0 = 99.8% var) → pose lives on a 1-D radial-zoom manifold from the
focus-of-expansion (ego drives forward → radial optical-flow). Consistent with my stored-pose SVD (rank-2 =
99.97%). The geometry-optimal pose codec = 1 scalar/frame (zoom rate) — deeper than my rank-2 SVD. Folds
into #140; $0 test: render the radial-zoom warp, measure d_pose vs the dense carrier on the frozen basin.

### 4. Native-grid in-cell repair byte-closes + SURVIVES (RA-1#3 + #137; AUDITED-STRONG, distinct from the survival wall)
RA-1 traced the probe scripts: blindspot-B's −32% in-cell repair optimizes on the **storable native 384×512
grid** and scores via the FULL roundtrip → the −32% is roundtrip-REAL, UNLIKE the compress-time solve
(which failed because it perturbed the camera-res 874×1164 grid that decorrelates under downsample). Residual
flips code at 0.78-0.83 B/flip < 1.273 waterline → byte-closes. This STRENGTHENS #137 (the boundary-flip
sidecar). Caveat: the −32% is a 3-pair/50-step smoke → needs a full-600 roundtrip-verified byte-closed A/B
before banking (~−0.033 S generous estimate).

### 5. Ego-hood (#139) — I measured the WRONG region (RA-1#1; AUDITED-PARTIAL, REOPEN-for-re-measurement)
RA-1 is right: the probe applied the proposal's own 3% threshold to the all-frame STATIC-CORE (0.038%) when
SegNet reads one frame → the mechanism's region is the PER-FRAME class-4 mask, proxied by the ever-hood band
= **7.36% of flips (ABOVE 3%)**, a −0.0193 S d_seg-if-clamped. **My "falsified" was region-wrong.** BUT MY
AUDIT of RA-1's net-win: the −0.0167 S net assumes the per-frame hood-edge clamp (a) byte-closes (~3900 B)
AND (b) SURVIVES the roundtrip — and (b) is the same survival wall that killed the RGB seg-correction sidecar
(R-11: 36.9% survival, ΔS +0.152 WORSE). The hood top-edge is the boundary residual the boundary levers
(margin-hinge + #137) already attack. **Disposition: REOPEN #139 for a per-frame-mask re-measurement WITH a
survival check — NOT a confirmed −0.0167 win.** Folds into #137 if it survives.

## REJECTED by my audit (do NOT act on these as claimed)
- **RA-2 R-3 "R1⊕R2⊕R3 lossless recode beats frontier −0.00092" — FALSE vs the current pointer.** Recompute:
  S≈0.191117 vs pointer 0.19110 = **+0.000017 (WORSE/NEUTRAL)**. The −0.00092 is vs a STALE ~0.192 baseline;
  this stack was already banked into the 0.19110 pointer (task #64). NOT a new pointer-mover. (Apples-to-apples
  + "scores are pointer-only" caught it.)
- **RA-2 R-5 "store the SegNet partition = d_seg=0 by construction" — survival-wall-suspect.** You can't impose
  argmax(SegNet(recon)); you can only impose RGB, which must survive the roundtrip (the 36.9% survival wall,
  R-11). Superseded by reopen #4 (native-grid in-cell repair is the better-posed, roundtrip-real version).

## NO-FAKE fix landed
- Lane-geometric-solve JSON labels LIED (`EUREKA_HOLDS_pure_pose_warps_contour` contradicted its own rows:
  identity 0.33 < pose_init 0.53 → pose-warp FALSIFIED, quasi-stationary). Corrected + original preserved in
  `_label_correction_note` (commit 04f60aef7). A continual-learning consumer would have mislearned the opposite.

## Mission status (GOAL firewall)
Pointer UNMOVED at 0.19110. The biggest reopened lever is **FP-shrink QAT (−0.022 to −0.029 S)** — and the
power-law correction means the **running long-train is now well-motivated, not betting against the symposium.**
G3 (the first byte-closed exact row, dispatching) is the calibration that grounds all of this.
