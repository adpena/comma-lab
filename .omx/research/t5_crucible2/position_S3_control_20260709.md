# P1 SEAT S3 — TRAINING-DYNAMICS / CONTROL / SCHEDULE (independent position)

T5 CRUCIBLE-2 (task #379, v7.5.2 optimal single trunk). Seat charter: costate #247 / NCDE #344 /
schedule — **OPEN Q4 (event-vs-epoch closure + train-verdict DECOUPLING as a first-class stage-exit
event, the R-4 operator catch)** primarily, **Q2 (warm-start-vs-fresh from the CONTROL view)**, **Q5
(terminal solve + pose-finish ENGAGE — event conditions, now under the operator-binding d_seg-
conditioning frame)**. Independent — no cross-read of sibling seats. Cites
`docs/operating_manual_craft_handoff.md` (§5 label MEASURED/DERIVED/INFERRED/ASSUMED; §7 answer-first-
then-risk; §6 attack-own-conclusion). Works from DELTA_GROUNDING only. I did NOT edit any
`witness_control/` file — I design the control LAWS *on top of* the parallel verdict-trend alarm's
classes {RISING-VERDICT, DECOUPLING, per-class} and the existing sensor surfaces. Pointer **0.19110
UNMOVED** — everything here is [macOS-MLX/CPU advisory] MEANS; only a byte-closed n600 row moves it.
`$0` — no compute; #205 untouched.

---

## ANSWER FIRST (the three positions in one paragraph each)

**Q4 — event-vs-epoch + decoupling.** Close the schedule as **event-first with epoch FAIL-SAFE CAPS,
where the min-stage floor is DERIVED into the critical-slowing relaxation time τ_relax and that ONE
floor simultaneously guards BOTH confounds** (the critical-slowing transient *and* the EMA-shadow lag).
**Make train-verdict DECOUPLING a first-class stage-exit event, DISTINCT from plateau** — a stage whose
train loss descends while the (live-confirmed) verdict d_seg rises is a term-exhausted stage: hand off /
change the term, never "more of the same gradient." The classifier already carries the signature
(`DIVERGING_ERASING` / `TRANSITION_TRANSIENT`, `tools/witness_control_monitor.py:130`); R-4's false-
green was under-firing from **sparse verdict cadence starving the persistence test + EMA-lag corrupting
the d_seg it reads** — not a missing class. My disambiguation rule converts the parallel alarm's
RISING-VERDICT/DECOUPLING into a **CONFIRMED-vs-EMA-LAG-SUSPECT disposition via one live-weight spot
verdict + a settle-window gate.** Every event emits an **advisory-only ranked costate row** (never
auto-fires heavy — CONTAINMENT).

**Q2 — warm-start vs fresh (control view).** **FRESH from ep0 with counter-force + collapse-fix is the
control-correct DEFAULT.** run-1 is PRE-actuation: its Road-floor (~0.40, R-3) is a CONVERGED WRONG-MASS
attractor (birth over-paints Lane 13.8× / Movable 4.6× into Road, mass-conserved). The over-paint
happens DURING CE birth, so even the preserved ep257 CE→tau ckpt inherits the over-painted mass. The
v7.5.2 loss has a DIFFERENT minimum; warm-starting drops you in the OLD one and asks the area constraint
to SHRINK already-over-grown islands (fight born-island momentum, climb OUT of a local min) instead of
NUCLEATING the right mass from birth. Because d_seg is separatrix-bound/stage-sensitive (P-3), the basin
is decisive — and run-1's is the wrong basin. Warm-start-ep257 is at most a cheap CONFIRM-ARM judged by
the **stage-1 island-birth MASS verdict** (not train loss — it decoupled), never the default launch.

**Q5 — terminal solve + pose-finish ENGAGE (operator-binding conditioning frame).** Pose is fired for
joint descent **only after d_seg is sufficiently conditioned** — the entry is a **d_seg-CONDITIONING
event, never an epoch**, built from the `jacobian_basin` sensor: `median_sigma_min ≥ 0.9·σ_min^plateau
AND basin_frac ≥ 0.8`, held with hysteresis (≥3 consecutive basin-cadence verdicts, σ_min non-
decreasing), AND gated by the d_seg-BASIN precondition (σ_min↑ is NECESSARY-not-SUFFICIENT, P-6). The
engagement is an **EVENT CHAIN**: d_seg-BASIN → #341 head GN/CG solve (IFF LM ρ re-verify ∈[0.8,1.2];
the seg-head solve SHARPENS the boundary → richer normals → σ_min↑, so it is a *conditioning amplifier*
for pose) → re-check conditioning → pose-finish. **Fallback if conditioning is never reached within
budget: pose-finish does NOT fire; SHIP the already-banked R1 dxi (0.127, 7.2 KB, n600 authority) as
the store-nothing carrier.** Because R1 is banked, a never-fired in-basin finish costs ZERO — the in-
basin finish is an OPTIONAL improvement over a banked floor, never a launch dependency (this resolves
P-5's HONEST FLAG).

---

## Q4 — EVENT-vs-EPOCH CLOSURE + TRAIN-VERDICT DECOUPLING AS A STAGE-EXIT EVENT

### Q4.1 The τ_relax closure — ONE floor guards TWO confounds (the key unification)

S-4 owes: derive `curriculum_min_stage_epochs` (250, HARDCODED-WITH-WAIVER) into the **critical-slowing
relaxation time τ_relax**. My contribution is that **min_stage and the decoupling detector are the SAME
physics**, so one derived floor closes both.

**Why (DERIVED).** Near a curriculum stage transition the system relaxes to the new stage's attractor
with a time-constant τ_relax that *diverges* approaching criticality (critical slowing down). During
that transient, ep_loss goes transiently flat *and the verdict can flatten or rise while train loss
still descends* — which is **exactly the decoupling signature**. So a decoupling (or plateau) read
INSIDE τ_relax is indistinguishable from the post-transition transient. The classifier already encodes
this as `TRANSITION_TRANSIENT` ("d_seg rising while loss falls, but RECENT — WATCH, do NOT act",
`witness_control_monitor.py:142`).

**The derivation (DERIVED-AT-CONFIG).** After the FIRST fired transition, fit the post-transition
ep_loss: `ep_loss(t) − c ≈ A·exp(−t/τ_relax)` over the within-stage window (closed-form log-linear
fit; the same fit machinery the NCDE `dynamics_analyzer` already runs). Set
`min_stage_epochs = ⌈k·τ_relax⌉` with `k=3` so the transient decays to `exp(−3)=0.050` — **5%, the SAME
threshold as the NCDE `remaining_descent_frac=0.05` basin** (`ncde_trajectory.py:331`), a satisfying
cross-consistency. The current 250 is consistent with τ_relax≈80 ep. **Bracket with an absolute fail-
safe `[150, 400]`** so a pathological fit cannot under-floor (stall-risk) or over-floor (waste).

**The EMA settle window rolls into the SAME floor (DERIVED).** EMA decay = 0.997 (CLAUDE.md EMA non-
negotiable) → time-constant τ_EMA = 1/(1−0.997) = 333 steps; at n600 per-pair batch=1 = 600 steps/ep →
τ_EMA ≈ 0.56 ep; 99% settle at 4.6·τ_EMA ≈ **W_settle = 2.6 ep** (label DERIVED from two MEASURED
constants). Because 3·τ_relax (≈240) ≫ W_settle (≈3), the τ_relax floor **already covers** the EMA
settle. Net:

```
min_stage_epochs = max( ceil(3 * tau_relax_fit),   # DERIVED-AT-CONFIG (critical slowing)
                        ceil(2.6),                  # W_settle (EMA lag), DERIVED
                        150 )                        # absolute fail-safe floor
                   clamped to [150, 400]             # fail-safe bracket
```

**No stage-exit event (plateau / decoupling / erasing) may fire before this floor.** That is the whole
point: the floor is not decoration, it is the guard that makes decoupling-as-exit SOUND. LABEL: the
*procedure* is DERIVED-AT-CONFIG; the specific fit value is produced live; 250 is the current placeholder
(HARDCODED-WITH-WAIVER) pending the first-transition fit.

### Q4.2 The EMA-lag disambiguation rule (the R-4 core, design ON TOP of the parallel alarm)

The parallel alarm (defaults-ON, being built by a sibling — I do NOT edit it) surfaces classes
{RISING-VERDICT (EMA-verdict slope>0), DECOUPLING (train↓ AND verdict↑), per-class (which class rises)}.
R-4's failure was that at ep50→ep100 the shadow controller read **CONVERGING** while Lane rose
0.349→0.381. **Root cause (INFERRED from the code + R-2):** (a) the scalar classifier reads d_seg ALONE
on the EMA shadow (`shadow_controller.py:314` comment: "the scalar classifier reads d_seg ALONE"), and
the ep100 rise is **partly EMA-shadow lag** (R-2 — the shadow, integrating ~0.56 ep of recent weights,
captured the CE-birth over-paint transient the LIVE weights had already recovered from); (b) the
`DIVERGING_ERASING` class requires `min_sustained_windows=3` *same-stage* verdicts, and the coarse
verdict cadence starved it → it fell through to CONVERGING on a noisy 2-point slope.

**The rule** (a disposition layer over the alarm; advisory-only):

A candidate decoupling (train-loss slope < 0 AND EMA-verdict d_seg slope > `creep_eps`) is:

- **EMA-LAG-SUSPECT → WATCH, do NOT hand off** if EITHER
  1. the rise onset is within **W_settle ≈ 3 epochs** of a moment-injecting event (stage boundary,
     Muon nucleation, birth-completion, moment reset) — the shadow has not settled; OR
  2. a **live-weight SPOT VERDICT** (one non-EMA R→SegNet argmax pass on the CURRENT live weights, the
     cheap disambiguator) reads `d_seg_live ≤ d_seg_EMA − margin` — the live is BELOW the shadow ⇒ the
     shadow is LAGGING (smoothing a spike-then-recovery), not decoupling.

- **DECOUPLING-CONFIRMED → hand-off advisory** iff the rise PERSISTS ≥ `max(min_sustained_windows
  verdicts, W_settle epochs)` AND the live-weight spot verdict CONFIRMS (`d_seg_live` slope also > 0)
  AND the net-stage slope > 0 (the classifier's existing `full_slope>0` persistence test).

The **live-weight spot verdict is the decisive disambiguator**: EMA-lag makes the shadow a lagging,
smoothed copy of live — during a spike-then-recover it rises AFTER the live has fallen, so
`live < shadow`; genuine decoupling has `live` rising TOO. This is one extra cheap R→SegNet pass fired
ONLY when the alarm raises RISING-VERDICT/DECOUPLING — not every verdict.

### Q4.3 Which epoch caps become events, which stay fail-safes (the closure table)

| control point | current | disposition | fail-safe cap (kept) |
|---|---|---|---|
| **min-stage 250** | hardcoded | → **DERIVED floor `3·τ_relax`** (DERIVED-AT-CONFIG); gates ALL exit events | absolute bracket `[150, 400]` |
| **CE→tau** | #315 plateau-slope event | KEEP event; **ADD decoupling-confirmed as a co-trigger** (R-4: CE decoupling IS the CE→tau signal) | ep_CE_max backstop |
| **Muon nucleation** | event | KEEP; treat boundary as stage transition (#270, Q2.3) | **muon cap 726** = "nucleate anyway" |
| **birth-completion** | Morse-Smale persistence event | KEEP | per-class part_frac backstop |
| **temporal-screw** | annulus_plateau FORMED-boundary event | KEEP (event-governed, B.4) | — |
| **decoupling-confirmed** | *(none — NEW)* | **NEW first-class stage-exit event** (Q4.2), floor-gated | subsumed by stage cap |
| **terminal solve / pose-finish** | epoch 2546 / 726 | → **EVENTS** (Q5: NCDE BASIN + conditioning gate) | **Polyak 2546** = "finish anyway" |
| **hosc β 3.177** | frozen constant | **anneal β 1→4** (fixed-β DIVERGES — CLAUDE.md launch caveat, tanh(β·sin) saturation → vanishing grad) OR `step_basis`; NOT a schedule event | frozen-hold value = fail-safe |

**Control-law contract compliance (§D):** every anneal provably completes before its consumer fires,
OR truncation-at-cap is event-safe. Each backstop cap is set > the expected event epoch (event normally
fires first) and truncation leaves a valid EMA (event-safe). The τ_relax floor + W_settle guarantee no
exit fires into a transient.

### Q4.4 What the controller DOES on each event within CONTAINMENT (advisory rec shapes)

The costate #247 ranker (never-regress guard: refuses candidates with predicted ΔS > 0; ranks survivors
by ΔS-PER-COST, `shadow_controller.py:_recommendations`) emits a **ranked advisory row** per event,
surfaced to the P8 operator-GO gate. **NOTHING auto-fires heavy** (autonomous = advisory-only; heavy/
stop/config = operator-GO).

| event | advisory shape (rec) | candidate(s) ranked by ΔS/cost |
|---|---|---|
| **PLATEAU** (\|dV/dt\|~0, after floor) | ADVANCE-STAGE or EARLY-STOP | next curriculum stage (cost≈0) |
| **DECOUPLING-CONFIRMED** (train↓ live-verdict↑, per-class) | **HAND-OFF / CHANGE-TERM** — "current gradient exhausted for the residual; NOT more of the same." Route the **class-specific** lever to the eroding class | eroding **Lane** → {#169 horizon-margin, #276 chroma, area-constraint}; **Undriv** → #360 temporal-screw sky-rotation; **Road** → Chan-Vese area constraint — **ONE per §9 increment** |
| **DIVERGING_ERASING** (sustained tau-creep, #205 signature) | **ROLLBACK-to-best + STOP** (tau/MCF erosion guard) | rollback_to_best costate |
| **BINDING-TERM-STALL** (#315: d_seg frozen while ep_loss/implied_S still move) | do NOT early-stop — deadlocked on the binding term | (overlay already live, `shadow_controller.py:315`) |
| **d_seg BASIN** (NCDE remaining descent <5%) | TERMINAL-SOLVE ADMISSIBLE | #341 head GN/CG (IFF ρ re-verify) |
| **POSE-CONDITIONING fired** | POSE-FINISH ADMISSIBLE | pose-finish (Q5) |

The DECOUPLING advisory is the direct operationalization of the R-4 operator read ("more CE isn't going
to help"): a confirmed decoupling means the costate `dS_depoch[stage]` has gone non-negative → the
never-regress guard REFUSES "continue" → the ranker surfaces the next-best move (a class-routed lever),
and the operator GO-gate decides.

---

## Q2 — WARM-START vs FRESH (control view) + RUN-1 LINEAGE RECONCILE

### Q2.1 Lineage reconcile (FEED-205stop ep325 vs the live 20260709T105312Z arm)

MEASURED (FEED-205stop + `experiments/results/` listing): run-1 (#205 birth-arm) STOPPED at **ep325**,
best `levelset_witness_ema_BEST.npz` d_seg **0.115102@ep325** (improving 0.1198@275→0.1151), preserved:
`levelset_resume_state.npz` (w/ optimizer moments) + `levelset_witness_ema_mlx.npz`@ep325 + per-stage
`levelset_resume_stageTau_ep257.npz` (**CE→tau boundary**). The `20260709T105312Z` (and sibling
`102011Z`/`101602Z`) are SHORT 07-09 v7.5-actuation smoke arms, NOT the #205 long lineage — separate
short runs, not a competing warm-start basin. **Reconcile with the operator recommendation:** "re-resume
to the stage-1 boundary ~ep250" is ALREADY SATISFIED by the preserved `ep257` CE→tau ckpt (stage-1 = CE
birth, ends at the tau boundary ep257). So the live disposition is **{fresh} vs {warm-start-from-ep257
or ep325}**, not a re-resume-to-produce-a-ckpt.

### Q2.2 The control-view verdict — FRESH default

**What transfers on warm-start:** decoder θ, born islands (part_frac lane 0.014/movable 0.022 — but
OVER-PAINTED, R-3), EMA shadow, optimizer moments (`resume_state.npz`).

**The basin risk (DERIVED from R-3's mass-conservation mechanism).** run-1 is PRE-actuation (no Chan-
Vese counter-force). Its Road-floor ~0.40 is a **converged WRONG-MASS attractor**: the birth stack
recalls-without-precision, over-painting Lane 13.8×GT / Movable 4.6×GT INTO GT-Road, mass-conserved with
the Road+Undriv deficit (0.1191≈0.1189). This over-paint is a **CE-birth** effect, so **even the ep257
CE→tau ckpt carries the over-painted mass**. The v7.5.2 loss (area constraint λ_lane 683.8 / λ_movable
322.6, equilibrium 1.25×GT) has a DIFFERENT minimum. Warm-starting drops the optimizer in the OLD
minimum's neighborhood and asks the constraint to **SHRINK already-over-grown islands** — climbing OUT
of a local min against born-island momentum — instead of **NUCLEATING the right mass from birth**
(nucleate, don't un-paint). Because d_seg is separatrix-bound / stage-sensitive / HARD (P-3), the basin
is decisive; pose is benign/monotone (P-3) so warm-start is irrelevant TO pose — **the warm-start
decision is PURELY a d_seg-basin decision, and run-1's is the wrong basin.**

**Red herrings ruled out:** warm-start-from-mod32cap (0.003366) — mod32cap is the mod-cap CONTROL with
islands DELIBERATELY UNBORN → a no-island basin, WORSE than fresh for a birth vehicle (S2/S1 own the
composition; flagged here as a control-basin non-starter).

**The A/B framing (honoring the operator).** The cheap disambiguating arm is **fresh-with-counter-force
vs warm-start-ep257-with-counter-force, judged by the STAGE-1 ISLAND-BIRTH MASS verdict** (part_frac[c]
vs GT — NOT train loss, which R-4 proved decouples; the mass verdict is the right arbiter). My prior is
STRONG for fresh (the mass-conservation mechanism is a converged-attractor argument, not a hunch), so
**fresh is the default launch; warm-start-ep257 is an optional confirm-arm ONLY if box-time allows** —
never the primary, so the launch is not spent on the inherited-floor risk.

### Q2.3 #270 Muon warm-start is ORTHOGONAL — KEEP unconditionally

#270 (S-2) is a DIFFERENT warm-start: the Muon-BOUNDARY treatment *within* a run — warm-start-momentum
+ LR re-warmup (cosine ~20 ep) + lr-final-frac 0.1, treating the Muon boundary as a stage transition.
Muon = −32% d_seg vs AdamW (MEASURED, L78). **KEEP unconditionally**, independent of the run-disposition
decision. Control-law: at the Muon boundary apply the stage-transition rewarmup AND re-apply the
τ_relax floor (a post-Muon transient must not be misread as decoupling/plateau — same critical-slowing
physics as any stage transition). This is why "Muon nucleation" appears as a stage boundary in the graph
below.

---

## Q5 — TERMINAL SOLVE + POSE-FINISH ENGAGE (operator-binding d_seg-conditioning frame)

**OPERATOR BINDING (2026-07-09, verbatim):** *"pose must not be fired for joint descent until optimal —
it needs d_seg to be sufficiently conditioned first."* ⇒ pose-finish ENTRY = a d_seg-CONDITIONING event,
NEVER an epoch. Built from the existing `jacobian_basin` sensor surface (MEASURED-BUILT, observer-only).

### Q5.1 The conditioning event {quantity, threshold-with-provenance, hysteresis}

**Quantity (BUILT).** The `jacobian_basin` aggregate — `median_sigma_min` + `basin_frac` — from the SVD
conditioning of `J_ξ = ∂(PoseNet∘R)/∂ξ ∈ ℝ^{6×6}` (`jacobian_basin.conditioning` /
`aggregate_conditioning`). P-6 DERIVED the coherence↔conditioning link: converging d_seg → richer
boundary normals → σ_min↑; this is WHY σ_min is a valid **d_seg-conditioning** proxy (it is the
Jacobian reason behind the flat 1.2–1.8 pose floor vs R1-from-converged 0.0011).

**Threshold-with-provenance.** The BUILT predicate `jacobian_basin.would_have_fired`:
```
median_sigma_min >= f_basin * sigma_min_plateau   AND   basin_frac >= quorum_q
```
Provenance (`JacobianBasinConfig`, MEASURED-tunable defaults): `f_basin=1.0` (default = reproduces the
current TERMINAL policy EXACTLY), `quorum_q=0.8`, `sigma_floor=1e-4`, `k_pairs=32`, cadence `every=4`.
`sigma_min_plateau` = the running MAX of median σ_min observed so far (the true plateau is only known
offline; live it is PROVISIONAL, labeled so in the row).

**My ENTRY-value correction (DERIVED):** for a LIVE entry gate, `f_basin=1.0` is unreachable — requiring
median σ_min to EQUAL its own provisional running-max never fires (a new max always resets the target).
Set **`f_basin=0.9`** (within 10% of best-observed conditioning) so the gate fires when σ_min has
essentially plateaued. LABEL: the predicate is BUILT; `f_basin=0.9` is DERIVED (provisional-plateau
correction); the exact 0.9 is **ASSUMED_AWAITING_VERIFICATION** (owed a sensor-trust A/B per B5, "the
sensor earns trust first"; dashboard currently reads would-fire=no under f_basin=1.0).

**NECESSARY-not-SUFFICIENT precondition (P-6).** σ_min↑ says the pose basin is CONDITIONED, but not that
the seg render has stopped moving. **AND the d_seg-BASIN precondition** (NCDE BASIN, remaining within-
stage descent < 5%) so pose descent does not chase a still-moving separatrix. The operator's "d_seg
sufficiently conditioned first" = **(seg-converged BASIN) AND (pose-well-conditioned σ_min gate)**, both.

**Hysteresis (transient basin must not fire it).** Require the conditioning predicate TRUE for
**≥ H_cond = 3 consecutive basin-cadence verdicts** (cadence `every=4` → ~12 verdict-intervals of
persistence) AND `median_sigma_min` **non-decreasing** across that window (a transient σ_min spike that
reverts fails the non-decreasing test). Same persistence philosophy as the classifier's
`min_sustained_windows=3`.

### Q5.2 Fallback if conditioning is NEVER reached within budget

Pose-finish **does NOT fire** (stays pose-blind → byte-identical incumbent). At export, **SHIP the R1-
banked dxi** (d_pose 0.001610 → contribution 0.127, ξ_eff 7.2 KB, banked at n600 AUTHORITY, P-1) as the
store-nothing carrier attached to the seg-converged EMA ckpt. Because R1 is ALREADY banked, a never-
fired in-basin finish costs **ZERO** — the in-basin finish is an OPTIONAL improvement over a banked
floor, never a launch dependency. **This resolves P-5's HONEST FLAG** (in-basin efficacy UNVALIDATED):
we never bet the run on it. The budget cap = the Polyak backstop epoch; if the conditioning gate has not
fired by then, ship banked.

**Regression guard (if pose-finish DOES fire).** If d_pose does not beat the banked 0.001610 within the
finish window → ROLL BACK to the pre-finish EMA + ship the banked R1 dxi. A failed finish must never
corrupt the seg-converged ckpt nor ship a WORSE dxi. (Advisory rec: `pose_finish_regression_rollback`.)

### Q5.3 Composition with the terminal head solve — an EVENT CHAIN (solve THEN pose)

NOT two epoch stamps and NOT joint-before-conditioning (joint before conditioning violates the operator
constraint). The chain:

1. **d_seg BASIN** (NCDE remaining descent < 5%, `ncde_trajectory` advisory) →
2. **#341 head GN/CG solve** — ADMISSIBLE IFF **LM ρ re-verify ∈ [0.8, 1.2]** on the CURRENT ckpt
   (full-P P=600 ONLY — K=8 subset OVERFITS +5.1%, N-3/L-8 MEASURED; exact tau-stage loss;
   `--fused-r-kernel` bit-identity; verdict through R + frozen CPU SegNet). Solve the SEG head first
   (out_sdf.{w,b} + out_tex.{w,b} + palette; ~791 affine params; FiLM EXCLUDED non-affine). **The head
   solve is a d_seg move that SHARPENS the boundary → richer normals → σ_min↑ → it is a CONDITIONING
   AMPLIFIER for pose.** Cost ~3 h GPU (operator-GO). Fallback: ρ out of range → SKIP the solve, run the
   terminal train stage instead (the conditioning gate still governs pose either way). →
3. **RE-CHECK the pose-conditioning gate** (Q5.1) after the head solve (σ_min should have risen) →
4. IFF conditioning fired (+ hysteresis) → **pose-finish** engages (w_pose on, terminal joint pose-
   descent) → serialize dxi at export; else → **ship banked R1 dxi** (Q5.2).

Ordering rationale (DERIVED): the seg-head solve strictly PRECEDES pose because it improves the very
conditioning the pose gate requires. σ_min-basin "earlier-engage" (P-6) does NOT win over this ordering
— σ_min is observability, the head solve is the ACT that raises it.

---

## THE STAGE GRAPH — CONFIG-SHAPED (the deliverable)

`{stage, entry-event, exit-events (incl. decoupling), floor, controller advisories}`. All events advisory
→ operator-GO wall (P8). LABEL per row: floors DERIVED-AT-CONFIG unless noted.

```
FLOORS (global, DERIVED):
  min_stage_epochs = clamp( max(ceil(3*tau_relax_fit), ceil(2.6_W_settle), 150), 150, 400 )
      tau_relax_fit : log-linear fit of post-transition ep_loss  (DERIVED-AT-CONFIG; 250 = placeholder)
      W_settle      : 4.6 * tau_EMA = 2.6 ep  (tau_EMA = 1/(1-0.997)/600 ep)  (DERIVED)
  DISAMBIGUATION (on any RISING-VERDICT/DECOUPLING alarm):
      EMA-LAG-SUSPECT  if onset < W_settle of a moment-inject event OR live_spot_verdict <= EMA - margin
      CONFIRMED        iff persists >= max(3 verdicts, W_settle) AND live_spot confirms AND net-slope>0

STAGE 0  ADMISSION (ep0)
  entry   : launch
  exit    : part_frac[lane]>0 AND part_frac[movable]>0  ->  CE      (paint-not-replace admission gate)
  floor   : none
  advisory: REFUSE-LAUNCH if any birth class zero-mass (structurally unrecoverable)
  precond : collapse-fix amber ON  (grad-clip + pose-eps-floor eps=(5/C)^2 + per-param grad-normalize)

STAGE 1  CE (birth)   [Chan-Vese counter-force ON: lambda_lane 683.8 / lambda_movable 322.6; birth stack]
  entry   : admission pass
  exit    : (a) PLATEAU (|dV/dt|~0, after floor)                 -> tau
            (b) DECOUPLING-CONFIRMED (per-class, live-confirmed)  -> tau   [R-4: CE decoupling IS the
                                                                            CE->tau signal; "more CE won't help"]
            (c) #315 plateau-slope event                          -> tau
  floor   : min_stage_epochs   (gates ALL of a/b/c)
  cap     : ep_CE_max backstop (fail-safe: advance anyway)
  advisory: on (b) note eroding class -> route the tau-stage lever; per-class costate row

STAGE 2  tau_softplus (unify)   [top-2 argmax term; eikonal step-up ~0.10 at onset; hosc beta ANNEAL 1->4
                                 NOT fixed 3.177; l7 DEMOTED from default (measured DEFECT, decoupling)]
  entry   : CE exit-event fired AND CE counter-force ramp complete (anneal-complete precond)
  exit    : (a) PLATEAU                                           -> Muon
            (b) DECOUPLING-CONFIRMED (per-class)                  -> fire ONE queued class-lever
                                                                     (§9 one-per-increment: #121 taper /
                                                                      #169 horizon-margin / #276 chroma)
            (c) DIVERGING_ERASING (sustained tau-creep, #205 sig) -> ROLLBACK-to-best + STOP advisory
  floor   : min_stage_epochs
  advisory: (b) route class-lever by eroding class; (c) tau/MCF erosion guard (rollback+stop)

STAGE 3  Muon finisher   [nucleation-fired; -32% d_seg vs AdamW]
  entry   : tau nucleation/PLATEAU event
  boundary: TREAT AS STAGE TRANSITION -> #270 warm-start-momentum + LR re-warmup (cosine ~20ep) +
            lr-final-frac 0.1 ; tau_relax floor RE-APPLIES (post-Muon transient not misread)
  exit    : PLATEAU -> TERMINAL-BAND ; DECOUPLING -> route lever
  floor   : min_stage_epochs
  cap     : muon cap 726 (fail-safe: nucleate anyway)

STAGE 4  TERMINAL-BAND   (EVENT CHAIN, not epochs)
  4a HEAD-SOLVE (#341)
     entry : d_seg BASIN (NCDE remaining descent < 5%)  AND  LM rho re-verify in [0.8,1.2] (full-P,
             exact tau loss, --fused-r-kernel, verdict through R + frozen CPU SegNet)
     act   : GN/CG solve of the ~791-param AFFINE seg head  (sharpens boundary -> sigma_min UP =
             conditioning amplifier)   cost ~3h GPU (operator-GO)
     fallbk: rho out of range -> SKIP solve, run terminal train stage
     advis : TERMINAL-SOLVE-ADMISSIBLE
  4b POSE-CONDITIONING GATE   (operator-binding; NEVER an epoch)
     entry : (after 4a) jacobian_basin fired [median_sigma_min >= 0.9*plateau_est AND basin_frac >= 0.8]
             WITH hysteresis [>=3 consecutive basin-cadence verdicts, sigma_min non-decreasing]
             AND d_seg BASIN                          (sigma_min NECESSARY-not-SUFFICIENT, P-6)
     fallbk: conditioning NEVER reached by budget -> pose-finish does NOT fire -> SHIP banked R1 dxi
             (0.127 / 7.2KB, n600 authority)          (in-basin finish is OPTIONAL, never a dependency)
     advis : POSE-FINISH-ADMISSIBLE | POSE-BLIND-SHIP-BANKED
  4c POSE-FINISH (D.9)
     entry : 4b fired
     act   : w_pose engages -> terminal joint pose-descent -> serialize dxi at export
     guard : d_pose fails to beat banked 0.001610 in window -> ROLLBACK to pre-finish EMA + ship banked
     advis : pose_finish_regression_rollback

STAGE 5  POLYAK FINISHER (backstop)
  entry   : fail-safe cap 2546 (no BASIN event fired by here -> finish anyway) OR finisher-regression-guard
  advisory: finisher regression guard
```

---

## §6 ATTACK MY OWN CONCLUSION (per the operating manual §6; risk always, §7)

1. **τ_relax fit fragility.** A single-transition log-linear fit can be noisy; a bad τ_relax could over-
   or under-floor. MITIGATION: the `[150,400]` fail-safe bracket + re-fit at each transition + the
   absolute 150 floor. RISK: if the fit systematically underestimates (fast early transient hiding a
   slow tail), decoupling could fire early. The live-spot-verdict confirmation is the second line of
   defense. LABEL: τ_relax procedure DERIVED; robustness ASSUMED_AWAITING_VERIFICATION (owed a backtest
   on the #205 CE→tau ep257 transition log).
2. **live-weight spot verdict cost/parity.** One extra R→SegNet pass per alarm — cheap, but it must use
   the SAME frozen CPU SegNet + through-R path as the verdict authority (else it introduces a NEW
   parity gap). MITIGATION: reuse the exact verdict codepath, live weights instead of EMA shadow. If
   MLX-GPU non-bit-identity bites (mlx_gpu_crossprocess law), gate on `--fused-r-kernel` or CPU. LABEL:
   INFERRED it is cheap; the parity must be VERIFIED before trusting the disambiguation.
3. **f_basin=0.9 is un-A/B'd.** The pose-conditioning gate could fire early (transient σ_min plateau) or
   never (σ_min never plateaus because d_seg never converges). MITIGATION: hysteresis + the never-fire
   fallback (ship banked R1) makes a never-fire HARMLESS; an early-fire is caught by the pose regression
   guard. LABEL: 0.9 ASSUMED_AWAITING_VERIFICATION (B5 sensor-trust A/B).
4. **decoupling-as-exit could over-fire and thrash the curriculum.** If every noisy verdict rise triggers
   a hand-off, the run never settles. MITIGATION: the τ_relax floor + persistence + live-confirmation
   gate; and the advisory is operator-GO, never autonomous — a human sees the ranked row. RISK: advisory
   fatigue if it fires often; the never-regress guard + ΔS-per-cost ranking keeps only frontier-improving
   rows.
5. **fresh-default forfeits run-1's 325 epochs of compute.** MITIGATION: this is intentional — the basin
   is wrong (R-3 converged-attractor); the warm-start-ep257 confirm-arm is available if the operator
   wants the evidence. The mass-conservation argument is the load-bearing DERIVED claim; if S1/S2's
   composition analysis contradicts it, defer to the synthesis.
6. **I am the CONTROL seat, not the STRUCTURE-BLIND seat (S6).** My stage skeleton (CE→tau→Muon→terminal)
   references the incumbent — it is NOT a blind derivation and could carry a residual PR95 echo (the
   elementwise-audits-launder-structural-cargocult LAW). S6 derives the skeleton blind; if S6's blind
   skeleton DIVERGES from mine, that divergence is the cargo-cult catch and S6 wins on the skeleton. My
   contribution is the CONTROL LAWS on top of whatever skeleton the synthesis adopts — the floors,
   events, disambiguation, and conditioning gates transfer to any stage sequence.

---

## PROVENANCE (per manual §5)

- **MEASURED (artifact):** run-1 d_seg 0.115102@ep325 + ep257 CE→tau ckpt (FEED-205stop); R-3 mass-
  conservation 0.1191≈0.1189 (FEED-roadfloor); LM ρ 0.847/0.868 (FEED-#342); K=8 subset +5.1% (N-3);
  R1 dxi 0.001610/0.127/7.2KB (FEED-238resolved, P-1); classifier `DIVERGING_ERASING`/`TRANSITION_
  TRANSIENT` (`witness_control_monitor.py:130-148`); `jacobian_basin.would_have_fired` predicate +
  defaults f_basin=1.0/quorum_q=0.8/sigma_floor=1e-4 (`jacobian_basin.py:270-280,45-60`); NCDE
  remaining_descent_frac=0.05 (`ncde_trajectory.py:331`); EMA decay 0.997 (CLAUDE.md); l7 DEFECT +
  fixed-β divergence (CLAUDE.md launch caveat).
- **DERIVED:** τ_EMA / W_settle=2.6 ep (from 0.997 + 600 steps/ep); min_stage=3·τ_relax with k=3↔5%↔NCDE
  basin; f_basin=0.9 entry correction (provisional-plateau); the EMA-lag disambiguation rule; the head-
  solve-as-conditioning-amplifier ordering; the warm-start-inherits-floored-basin argument (from R-3's
  converged-attractor mechanism).
- **INFERRED:** R-4 root cause (sparse cadence starving persistence + EMA-lag), from the code + R-2; the
  live-spot-verdict cheapness.
- **ASSUMED_AWAITING_VERIFICATION:** exact f_basin=0.9, the τ_relax-fit robustness, the H_cond=3 /
  H_settle persistence counts (all owed sensor-trust / backtest A/Bs; B5 "sensor earns trust first").

**Pointer 0.19110 UNMOVED. This is MEANS — a control design, not a score. Only a byte-closed n600
`upstream/evaluate.py` row moves the pointer.**
