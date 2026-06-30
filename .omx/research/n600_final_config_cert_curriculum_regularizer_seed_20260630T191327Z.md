# n600 v2 witness — FINAL CONFIG CERTIFICATION: CURRICULUM × REGULARIZER × SEED axis

**UTC** 20260630T191327Z · **tag** `[macOS-MLX/numpy advisory · certification artifact · NON-PROMOTABLE]` · **pointer 0.19110 UNMOVED.**
Scope: a first-principles certification of the **curriculum + stage-transition-treatment + variational-regularizer + openpilot-seed** knobs of the from-scratch n600 v2 witness, BEFORE a 1000-epoch one-GPU burn. CPU-only, NO GPU, NO launch.
**means≠ends:** this certification is a MEANS. The only END is a byte-closed n600 exact row < 0.19110 (CPU/CUDA, never MPS). Certifying a config does not move the pointer.
**Out of scope (certified elsewhere):** arch (mod-dim/hidden) — see `n600_v2_recursive_review_20260630T182612Z.md` revisions #3 (26/96 rate-positive); surgical θ* levers (margin-saliency/lane-thin/hardness/DM1) — attribution-clean-first per review #2. This doc certifies ONLY the 3 named axes.

---

## 0. The single strongest grounding (state it first)

Every value in the axis I certify is **the live n200 proven config**, not an invention. The proven Muon arm command
(`experiments/results/levelset_thetastar_muon_arm/run_muon.log` launch line) ran EXACTLY:

```
--curriculum --tau-softplus-start-epoch 300 --l7-start-epoch 600 --muon-start-epoch 726 --muon-lr 0.002
--epochs 1000 --softmax-temp-start 1.0 --softmax-temp-end 0.05 --tau-softplus-tau 0.3
--stage-transition-rewarmup-epochs 8 --stage-transition-rewarmup-floor 0.1 --stage-transition-rewarmup-shape linear
--stage-transition-reset-moments --eikonal-weight 0.01 --length-weight 0.001 --ema-decay 0.997
--structured-init --structured-init-include-lane --lane-prior-phi1 --lane-prior-phi1-mode replace --lane-prior-phi1-dash-gate
```

and reached the **best measured realized-through-R d_seg 0.0036976 @ ep1000** (`levelset_best.json`, n200, `[macOS-CPU advisory]`).
**That is the existence proof.** The curriculum/regularizer/seed axis is therefore certified primarily by *reproduction*, with deep-math
secondarily to (a) justify WHY it works, (b) locate the defensible refinements, and (c) decide the scheduler build.

**Regime-faithfulness (load-bearing, the one thing the from-scratch run must get right):** the proven 0.003698 came from a CHAIN of
warm-started arms (CE→Tau→L7→Muon), each `--resume-from` the prior, all with `--epochs 1000` and `--anneal-epochs` UNSET (→ default = epochs
= 1000). The from-scratch n600 collapses that chain into ONE continuous run. It is regime-faithful **iff** `--anneal-epochs` is again unset
(→ 1000) so the softmax-temp + LR cosine denominators equal the run length, reproducing the exact (temp, LR) trajectory at every epoch.
The from-scratch command does omit `--anneal-epochs` → **regime-faithful ✓.** (For from-scratch this is the simple/correct case; the
`--anneal-epochs` warm-start subtlety in the help — set it to the ORIGINAL schedule on a partial-window resume — does NOT apply here.)

**The unifying frame (variational level-set PDE).** The witness is the viscosity solution of `S[φ] = 100·∫disagree + √(10·∫pose) + 25·bytes[φ]`;
training is gradient flow; the curriculum is a **homotopy of relaxations** of that one functional (CE convex surrogate → τ-softplus temperature
homotopy → l7 high-p soft-L∞ → Muon orthogonalized geometric finisher); eikonal (|∇φ|→1) and length (∫ds) are LIVE derivative/integral
regularizers; the seed places φ's zero-level-set at the known separatrix from ep0. The certification reads each knob against THIS PDE.
(`theta_star_witness_lever_stack_and_variational_levelset_frame_20260627`.)

---

## 1. MEASURED anneal trajectory (computed from the trainer's own formulas)

`_softmax_temp_for_epoch` (line 731) = `0.05 + 0.475·(1+cos(π·ep/1000))`; `_lr` cosine (line ~1765) = `1e-4 + 4.5e-4·(1+cos(π·ep/1000))`.
**Both FREEZE at the Muon boundary** — code line 1905: `_anneal_ep = muon_start_epoch if muon_switched else ep` → during the Muon finisher
(726–1000) softmax-temp and base-LR are pinned to their ep726 values; Muon runs at its own `--muon-lr 0.002` (log: "LR schedule frozen for the finisher").

| epoch | stage boundary | softmax-temp | base-LR | note |
|---:|---|---:|---:|---|
| 0 | CE start | 1.000 | 1.00e-3 | soft, gradients flow |
| 275 | (CE knee, measured) | 0.834 | 8.42e-4 | CE saturated ~here |
| **300** | **τ-softplus start** | **0.804** | 8.15e-4 | seg-form ce→tau_softplus |
| 450 | (τ knee, measured) | 0.599 | 6.20e-4 | τ best ~0.00431, then drifts |
| **600** | **l7-softplus start** | **0.378** | 4.11e-4 | seg-form tau→l7_softplus |
| 650 | (l7 mid, measured) | 0.309 | 3.46e-4 | l7 0.00426, descending |
| **726** | **Muon start** | **0.215** | 2.57e-4 | temp+LR FREEZE here; Muon @ 2e-3 takes over |
| 1000 | end (nominal) | 0.050 | 1.00e-4 | **never reached** — frozen at 0.215/2.57e-4 |

Two consequences this table makes concrete and that the certification depends on: (i) **`--softmax-temp-end 0.05` is never reached** (Muon
freezes temp at ~0.215); the proven snapshot temp 0.2157 confirms it. So temp-end is a *nominal* knob, weakly load-bearing. (ii) **`--muon-lr
0.002` is 7.8× the frozen base-LR (2.57e-4)** — Muon's spectral-normalized step at 2e-3 is structurally "the drop"; the default (omit → 0.1·lr
= 1e-4) is 20× lower and would not reproduce the descent (the NO-FAKE catch, below).

---

## 2. PER-KNOB CERTIFICATION

Verdict legend: **OPTIMUM-CONFIRMED** (proven + first-principles sound; ship) · **REFINE-TO-x** (defensible improvement, not launch-blocking) ·
**NEEDS-θ*-A/B** (genuinely open; ship the proven value, A/B as a warm-start arm).

### A. CURRICULUM

**A1 `--curriculum` (ON)** — value: ON. **5-lens:** the seg target is a piecewise-constant argmax (a free-discontinuity functional); a single
hard loss from ep0 is non-convex with bad basins. Curriculum = **deterministic annealing** (Rose 1998, F=D−T·H): solve a sequence of
progressively-sharper relaxations, each warm-starting the next — the convex CE basin → soft-max-margin → soft-L∞ → geometric finisher tracks the
global solution branch instead of quenching into a local one. **Measured:** CE+softplus LOWER d_seg, the smooth stage RAISES it (dropped from this
curriculum), Muon is the drop (`feedback_different_stages...`). **Verdict: OPTIMUM-CONFIRMED.**

**A2 `--epochs 1000`** — value: 1000. **Justification:** the proven TOTAL budget (CE 0–300 / τ 300–600 / l7 600–726 / Muon 726–1000). **Open
on the Muon tail:** the n200 Muon was STILL DESCENDING at ep1000 (0.003988@900 → 0.003805@950 → 0.003718@975 → 0.003698@1000, decelerating) =
critical slowing near a rate-distortion topological transition (Agmon–Tishby ISIT 2021 arXiv:2103.02646; PROVEN for the Arimoto–Blahut class our
annealing descends). **Verdict: OPTIMUM-CONFIRMED for the first launch; REFINE-TO-extend-Muon (epochs 1000→~1200) as a config-only follow-on** —
the tail is real descent, not a plateau, and extending the *finisher* is where the expensive tail is worth paying (the pivot memo's "Muon
long-grind = the n600 TRUE finisher"). This is a free config knob, not a build.

**A3 `--anneal-epochs` (unset → 1000)** — **Verdict: OPTIMUM-CONFIRMED + a binding pre-flight assertion.** For a from-scratch run it MUST stay
unset (denominator = epochs = run length = regime-faithful to the proven chain, §0). If A2's extend-Muon refinement is taken (epochs→1200), set
`--anneal-epochs 1000` so CE/τ/l7 keep their proven (temp,LR) trajectory and ONLY the Muon tail lengthens — otherwise the whole anneal dilates and
the stage boundaries land at the wrong temperatures.

**A4 `--tau-softplus-start-epoch 300`** — value: 300. **5-lens (deterministic annealing + measured saturation):** the τ stage should begin once
the *previous* relaxation (CE) has saturated; CE saturates ~ep275 (n200 DOE). Saturation epoch is an optimization-dynamics quantity and is
**~pairs-independent** (memo: "saturation EPOCH ~pairs-independent; FLOOR is capacity+data-bound") → it transfers n200→n600 at the SAME epoch
budget (1000). 300 sits just past the CE knee (temp 0.804, still soft — gradients flow into the sharpening). **Verdict: OPTIMUM-CONFIRMED** (a ~ep280
start is a marginal, non-decisive tightening).

**A5 `--tau-softplus-tau 0.3`** — value: 0.3. The softplus margin temperature of the τ relaxation (distinct from the softmax render temp). Proven
default; sets the sharpness of the soft-max-margin surrogate. Unswept independently but inside the proven config that reached 0.003698. **Verdict:
OPTIMUM-CONFIRMED** (low-priority A/B; the dominant temperature lever is the softmax-temp anneal, A8).

**A6 `--l7-start-epoch 600`** — value: 600. **THE one genuinely-open curriculum boundary.** Measured: τ saturates ~ep450 (best 0.00431) then
**over-trains/drifts**; the proven config nonetheless runs τ to ep600 (150 ep past its knee) before l7. The DOE memo explicitly recommends "n600:
shorter CE + τ ~150ep with EARLY-STOP at the knee." So there are two readings: (i) the τ over-run is wasted/harmful (start l7 ~ep450–500, hand l7
the freed budget); (ii) the over-trained τ *primes* the partition for l7/Muon (the primes-vs-corrects attribution frame) and the proven best came
*with* the over-run. The proven existence proof breaks the tie toward shipping 600, but the EV of an earlier l7-start is real. **Verdict:
NEEDS-θ*-A/B — ship 600 for the certified launch; A/B `--l7-start-epoch 480` as a warm-start arm from the τ-knee ckpt.** (This is the highest-EV
single curriculum refinement and the natural first job of a plateau-detector / root-tracking scheduler — see §3.)

**A7 `--muon-start-epoch 726`** — value: 726. **Justification:** ≥ l7-start (validator line 2636: 726≥600, no WARN), PR95 stage-8 placement (the
finisher polishes a FORMED partition), and the measured "Muon is THE drop." The proven `stage_checkpoints[0].epoch=726`. **5-lens:** Muon =
steepest descent under the spectral (operator) norm (Bernstein–Newhouse 2409.20325; Keller-Jordan Newton–Schulz) → it spreads the update across ALL
singular directions = an anti-collapse, full-rank finishing step on the 2-D hidden matrices, exactly when the partition is formed and the remaining
work is conditioning. **Verdict: OPTIMUM-CONFIRMED.**

**A8 `--muon-lr 0.002` [CRITICAL]** — value: 0.002. **The NO-FAKE catch (caught in `n600_v2_recursive_review`, re-verified here from the live
log + `muon_finisher_switch` JSON: `muon_lr 0.002, muon_adamw_lr 0.0001`).** Muon normalizes its update to ~unit spectral norm, so muon-lr is a
spectral-norm step size; at 2e-3 it is **7.8× the frozen base-LR (2.57e-4) at ep726** — that ratio IS why Muon is the drop. Omitting the flag →
trainer default `0.1·lr = 1e-4` = **20× too low** = the finisher crawls = the 0.003698 descent is NOT reproduced. **Verdict: OPTIMUM-CONFIRMED and
MANDATORY — the GO-ready command MUST carry `--muon-lr 0.002`.** (Also inside the help's documented optimal-form band 1e-3–5e-3; tuning up is a
later optimal-form lever, not a launch change.)

**A9 `--softmax-temp-start 1.0 / --softmax-temp-end 0.05`** — values: 1.0 / 0.05. **5-lens:** the render softmax temperature is the *continuous*
temperature homotopy that sharpens softmax(φ/T) toward the hard argmax as T→0 (config-review #4: anneal hi→lo, NOT a fixed 0.1 which reintroduces
RGB-Gibbs). Start 1.0 = fully soft (gradients flow through all classes early); end 0.05 = sharp. **CAVEAT (measured, §1):** the end is **never
reached** — Muon freezes temp at ~0.215 at ep726, so 0.05 is nominal. This is fine (the proven arm pinned at 0.215 and won), and it means temp-end
is weakly load-bearing. **Verdict: OPTIMUM-CONFIRMED-with-caveat** (start=1.0 is the real lever; end=0.05 is cosmetic under the Muon freeze; if
A2's extend-Muon is taken, the freeze just holds longer at 0.215 — no action).

### B. STAGE-TRANSITION TREATMENT ("different stages need different treatment" — the non-negotiable, operationalized)

**B1 `--stage-transition-rewarmup-epochs 8` · B2 `--…-floor 0.1` · B3 `--…-shape linear`** — values: 8 / 0.1 / linear. **5-lens (RAdam/warmup
variance + landscape-change first-step):** a stage boundary (CE→τ, τ→l7) is a discrete change of the loss landscape; the scheduled LR there
(8.1e-4 at ep300, 4.1e-4 at ep600) applied as the *first* step in a new landscape is a destabilizing large step (the margin-engage spike-skip
incident: stale regime → gnorm explosion → all-batches-skipped → stall). Ramping LR from 10% back to scheduled over 8 ep takes small first steps
while the new-landscape gradient statistics settle. 8 ep is short but adequate (the variance settles far faster than AdamW's β2=0.999 ~1000-step
memory; the proven arm used exactly 8/0.1/linear). **Verdict: OPTIMUM-CONFIRMED** (cosine shape is a marginal A/B; the non-negotiable is satisfied).

**B4 `--stage-transition-reset-moments` (ON)** — **5-lens:** AdamW m/v moments accumulated under the OLD loss are stale through a landscape change
(the FEED-ft#3 tau-jump root cause); the first post-boundary update divides a new-landscape gradient by old-landscape second-moment statistics →
mis-scaled. Reset zeroes m/v so the new stage re-accumulates correct statistics. This is the structural core of the non-negotiable, and it composes
with B1–B3 (reset sets direction-statistics; rewarmup sets step-size) without double-counting. **Verdict: OPTIMUM-CONFIRMED.**

**B5 [GAP NOTE] the Muon boundary (l7→Muon @ ep726) is the one transition NOT covered by B1–B4** (help: "no effect during the Muon finisher").
The log shows it: d_seg ROSE +8% at the switch (ep725 0.004316 → ep750 0.004674) then took ~75 ep to recover below the pre-switch value. The Muon
optimizer self-inits a fresh optimizer (no stale momentum) and spectral-normalizes its step (bounded first-step magnitude ≈ muon-lr × unit-norm
regardless of momentum), which is the *structural substitute* for a rewarmup — so the proven arm reached 0.003698 despite the transient.
**Verdict: OPTIMUM-CONFIRMED-for-launch, with a documented REFINE-candidate** — a Muon-boundary LR rewarmup (ramp muon-lr from a floor over a few
ep) might shave the +8% transient; this is a small BUILD (the rewarmup path currently excludes the Muon stage), best folded into the root-tracking
scheduler (§3), NOT a launch blocker.

### C. VARIATIONAL REGULARIZERS

**C1 `--eikonal-weight 0.01`** — value: 0.01. **5-lens (viscosity-solution PDE):** the eikonal term penalizes (|∇φ|−1)² → pushes φ toward a true
signed-distance function, so the margin φ₍₁₎−φ₍₂₎ = signed distance to the boundary = calibrated "pixels-from-flipping." This is what makes the
level-set OBSERVABLE/targetable (operator memo: margin = distance under eikonal) AND well-conditions the boundary. Weight 0.01 is small vs the data
term w_seg=100 (ratio 1e-4) — a gentle topology bias that shapes without fighting the data. **Risk if too large:** over-regularizing to a global SDF
fights the locally-sharp argmax. 0.01 is conservative-correct. **UNSWEPT** (no isolated A/B). **Verdict: OPTIMUM-CONFIRMED-conservative; REFINE via
a low-priority A/B (0.003 / 0.03)** — small expected effect; the data term dominates.

**C2 `--length-weight 0.001` [FLAG a principled tension]** — value: 0.001. **5-lens (Chan-Vese / geodesic active contour):** penalizes boundary
length ∫ds → short, smooth boundaries; classic denoiser of jagged spurious boundaries. **The tension (deep-math, must be on the record):** the
binding residual is the **lane dashes** — many SHORT, high-curvature, short-persistence boundary features (birth-death: lane <3px flip 92%; lane
PH⁰-dim 0.83 = multi-scale). A boundary-length penalty *intrinsically shortens boundaries* → it could SUPPRESS exactly the dash structure we need to
synthesize. At 0.001 (ratio 1e-5 vs w_seg) it is small enough that the proven arm reached 0.003698 with the lane residual still alive, so it is not
actively harmful at this weight. But it is the one regularizer whose gradient points AGAINST the residual. **Verdict: OPTIMUM-CONFIRMED-conservative
WITH a monitoring flag — keep 0.001; if a lane-targeted warm-start arm under-fits the dashes, the first thing to A/B-lower is length-weight (→0.0003
or 0).** Do not raise it.

**C3 `--ema-decay 0.997` [REFINE: stage-dependent]** — value: 0.997. **Justification:** the EMA non-negotiable value (Quantizr 0.997); the deployed
weights ARE the EMA shadow and the verdict authority is the shadow, so the shipped checkpoint is what 0.997 produces. **5-lens tension (the
"different stages" law applied to EMA):** 0.997 has effective memory ~1/(1−0.997)=333 steps. In the EARLY fast-descent stage this LAGS the live
weights badly (measured up to 78× — the EMA-shadow-lag artifact); in the converged Muon tail (slow, oscillating around a fixed point) 0.997 is
near-ideal (Polyak–Ruppert averaging of a settled iterate). So a single global decay is in genuine tension with the per-stage principle: the early
stage wants a *faster* (smaller) decay to track, the finisher wants a *slow* (large) decay to average. **Why ship 0.997 anyway:** (a) the lag hurts
mostly the EARLY *telemetry*, not the FINAL converged checkpoint (the only thing scored); (b) it is the proven value + the EMA non-negotiable + the
verdict-authority definition; (c) there is no stage-dependent-EMA flag — it would be a BUILD. **Verdict: OPTIMUM-CONFIRMED-for-the-final-checkpoint;
REFINE = stage-dependent decay (e.g. 0.99 in CE→τ, 0.997 in l7→Muon), a future build, NOT a launch change.** Telemetry note (telemetry-accuracy
discipline): treat early-stage realized-d_seg as lagged; trust the converged-stage verdicts.

### D. OPENPILOT SEED

**D1 `--structured-init` (ON)** — value: ON. **5-lens:** initialize φ so argmax(φ) ≈ the validated self-detected static-core partition
(road/sky/hood deep SDFs) → the run starts near the structured floor and learns only the residual (lane + movables). Geometric prior = a better
basin. **MEASURED CAVEAT (help + FEED-ef):** NO epoch-0 realized win (the render is texture-dominated at init, so SegNet reads texture not the
partition); the benefit is purely a *training-trajectory* A/B (does a correct partition init converge faster?), UNPROVEN, and the pretrain path is
hosc/SIREN-init-FRAGILE (loud WARN if it stalls). **De-risked:** the proven arm used it + hosc + siren-init and did NOT stall (pretrain
direct-argmax-disagree 0.00313 — converged cleanly). **Verdict: OPTIMUM-CONFIRMED-as-proven (existence proof it does not hurt); benefit UNPROVEN →
it is the cheapest trajectory-A/B to run later (`--no-structured-init` arm), but ship ON for proven-config fidelity.**

**D2 `--structured-init-include-lane` (ON) [measured NO-OP]** — value: ON. **Measured (proven log):** `lane_static_mask_px=0, lane_px=0,
lane_mean_iou=0.0`, part_frac class-1=0.0 → the static lane *band* path did NOT fire (no static lane mask was injected); the structured init seeded
road (0.248) / sky (0.498) / hood (0.254) only. So this flag was INERT in the proven run; the lane geometry came entirely from D4 (lane-prior-phi1).
**Verdict: OPTIMUM-CONFIRMED-harmless (no-op in this config); the lane seed is D4, not this flag.** (Keep ON for proven fidelity; it costs nothing.)

**D3 `--structured-init-steps 600`** — value: 600 (= trainer default; the proven arm did not pass it, so it used 600). The subsampled-Adam pretrain
budget that fits φ to the structured target; the proven pretrain converged to 0.00313 direct-disagree at 600 steps / lr 5e-3. **Verdict:
OPTIMUM-CONFIRMED** (note: help warns lr 8e-3 stalls; 5e-3 default is the converging value — do not raise).

**D4 `--lane-prior-phi1 --lane-prior-phi1-mode replace --lane-prior-phi1-dash-gate` (ON/replace/ON)** — **the strongest-grounded seed knob.**
**5-lens (geometry):** inject the openpilot deg-3 centerline signed-distance into the lane (φ1) channel of the structured-init target. FEED-fs
MEASURED that this centerline IS the Road↔Lane separatrix (fit residual 1.9e-5). It is a GENERIC same-rig camera-geometry prior (ground-plane
homography K → image-space deg-3 curve → per-pixel SDF), rule-118 FREE as a train-time init (ships 0 archive bytes; only SHIPPING the coords would
count). Proven log: `lane_n_lines:5, lane_total_floats:45, lane_band_px:1261, dash_gate:true`. mode=replace (set the lane channel TO the openpilot
fit, vs bias=add) is the proven choice and the correct one (the centerline IS the separatrix, not a perturbation of a learned one). dash-gate models
the dash period (the lane is dashed, not solid). **Verdict: OPTIMUM-CONFIRMED** (strong measured geometric grounding + proven + the ACTIVE lane
seed). Like D1 its epoch-0 *realized* benefit is texture-gated/trajectory-only, but it places φ's zero-level-set at the right manifold location from
ep0 — the seed analog of the −48% directional-basis lever, and the cheapest way to give the lane residual a correct starting boundary.

---

## 3. DECISIVE SUB-QUESTIONS (answered honestly)

**Q1 — Is the 300/600/726 schedule optimal at n600, or should boundaries scale with the epoch budget / be set by a plateau-detector?**
**Answer: the schedule transfers DIRECTLY to n600 and needs NO scaling, because n600 keeps the SAME epoch budget (1000) as the proven n200, and the
stage saturation epochs are ~pairs-independent (optimization dynamics) while only the FLOOR is pairs-dependent.** There is no scaling question
between n200@1000 and n600@1000 — the boundaries land at the same epochs. **Scaling only becomes necessary if `--epochs` changes** (e.g. the A2
extend-Muon refinement): then boundaries must scale with `--anneal-epochs` (set it to 1000 so CE/τ/l7 stay put and only the Muon tail lengthens —
the A3 assertion). **A plateau-detector is the principled generalization** (fire l7 at the τ-knee instead of a fixed 600 — directly the A6 open
question) and IS worth having, but it is NOT needed for THIS launch because the proven fixed boundaries have an existence proof at this exact budget.
The one boundary with real refinement EV is **l7-start (A6: 600 over-runs the τ-knee ~450 by ~150 ep)** — ship 600, A/B 480 as a warm-start arm.
The deterministic-annealing nuance: passing too FAST through a critical temperature can quench into a worse fixed point — but the proven arm's
SMOOTH monotone descent (no quench pathology, just the expected critical *slowing* of a proper anneal) is evidence the static cosine is annealing
acceptably, not quenching. (If a $0 anneal-shape A/B later shows quenching, that REOPENS scaling — see Q2.)

**Q2 — Is the ROOT-TRACKING ANNEAL SCHEDULER worth BUILDING before the burn? BUILD/SKIP + EV.**
**Answer: SKIP-BUILD-BEFORE-BURN. Build it as a PARALLEL follow-on, gated behind a $0 anneal-shape A/B.** See §4 — this is the one decision that
changes whether we burn before the scheduler lands, so it gets its own section.

**Q3 — Are eikonal/length weights at their optimum?**
**Answer: both are at conservative, proven, defensible values; neither is launch-blocking; one carries a flagged tension.** `--eikonal-weight
0.01` is sound (makes margin=distance; gentle vs w_seg=100) and UNSWEPT → OPTIMUM-CONFIRMED-conservative, low-priority A/B. `--length-weight 0.001`
is sound at this small magnitude BUT its gradient is the only regularizer that points AGAINST the lane-dash residual (Chan-Vese shortens boundaries;
lanes ARE short boundaries) → OPTIMUM-CONFIRMED-conservative WITH a monitoring flag: if a lane-targeted arm under-fits dashes, lower length-weight
first (→0.0003/0); never raise it. Both A/Bs are warm-start refinements, not launch changes.

---

## 4. THE BUILD/SKIP DECISION — root-tracking anneal scheduler

**RECOMMENDATION: SKIP building it before the n600 burn. Launch the proven static schedule now; build the scheduler in PARALLEL (CPU design + the
$0 falsification smoke) for the warm-start re-treatment / v2, gated behind a measurement.**

What the scheduler is (`per_stage_fractal_optimizer_priming_reheat_anneal_design_20260629` + the root-tracking anneal arXiv:2306.09790 referenced
in `post_muon_application_plan`): a `--optimizer custom` per-stage thermal controller — Stiefel-W + code-spectral-entropy (the "root cure" half) +
Muon-prime/moment-reset at transitions + PR-triggered/SGDR reheat + a root-tracking anneal that slows the softmax-temp through the critical
temperature and speeds between (the wall-clock-optimal "same d_seg in fewer epochs" lever).

**The EV ledger (why SKIP-before-burn wins):**

1. **The two HIGHEST-EV pieces are already shipped as flags.** The design's HIGH-EV items #2 ("per-stage moment-reset + prime at transitions" — the
   binding non-negotiable) are ALREADY satisfied by `--stage-transition-reset-moments` + `--stage-transition-rewarmup-epochs 8` (B1–B4, proven). No
   build buys what config already gives.
2. **The design's #1-HIGHEST piece (Stiefel-W + code-entropy "root cure") is MEASURED-likely-a-non-problem.** The recursive review ran TwoNN+MLE
   nonlinear-ID and found the per-pair code spans ~9 dims and the GT manifold's intrinsic dim is ~9 — so the FiLM PR-collapse (DM1, the disease the
   root-cure targets) is NOT the binding d_seg cause (the proven arm hit 0.003698 with linear-PR-1.32 code = the existence proof). Building the
   root-cure has low expected d_seg value; the review explicitly says DROP `--film-stiefel --code-spectral-entropy-weight` from the first launch.
3. **The root-tracking anneal itself is mostly WALL-CLOCK, not FLOOR.** Its proven benefit is "same d_seg in fewer epochs." The n600 burn exists to
   measure the FLOOR (the potential pointer-mover); wall-clock is secondary. The config-only wall-clock response to the still-descending Muon tail is
   simply **extend Muon** (A2: epochs→1200 + anneal-epochs 1000) — free, no build.
4. **The proven static schedule has an existence proof (0.003698, monotone, no quench).** Building an unbuilt/untested 24-knob scheduler before the
   burn trades a guaranteed-launchable proven config for build-risk + test-delay + the kitchen-sink over-engineering trap the design memo itself
   warns against (§7: PR105 1776-LOC LOST to 241-LOC). means≠ends + race-mode: ship the measured row, don't build infrastructure first.
5. **The one real upside (floor-quench risk) is unmeasured and cheaply gateable.** IF the static cosine quenches through the critical-τ (cools too
   fast at the phase transition), root-tracking could buy FLOOR, not just wall-clock. But (a) the proven arm shows no quench signature, and (b) the
   decisive test is a **$0 anneal-shape A/B** (warm-start the τ/l7 stages with a longer `--anneal-epochs` and check if the floor drops), NOT a full
   scheduler build. **Gate the build behind that A/B** (the design memo's own §6 firewall discipline: if the cheap lever doesn't move the END, don't
   build the expensive one).

**Net:** building first delays the only thing that matters (a measured n600 floor row) for an unbuilt/untested, wall-clock-mostly optimizer whose
highest-EV structural half is measured-likely-a-non-problem and whose transition-treatment half is already shipped as config. **Burn on the proven
static schedule. Build the scheduler in parallel as a follow-on for the warm-start re-treatment / v2, after the $0 anneal-shape + l7-start A/B's say
whether the static anneal is leaving floor on the table.**

---

## 5. CERTIFIED config block (curriculum × regularizer × seed only)

```
--curriculum --tau-softplus-start-epoch 300 --tau-softplus-tau 0.3 --l7-start-epoch 600 \
--muon-start-epoch 726 --muon-lr 0.002 --muon-momentum 0.95 --muon-ns-steps 5 \
--epochs 1000 --softmax-temp-start 1.0 --softmax-temp-end 0.05  (--anneal-epochs UNSET → 1000) \
--stage-transition-rewarmup-epochs 8 --stage-transition-rewarmup-floor 0.1 \
--stage-transition-rewarmup-shape linear --stage-transition-reset-moments \
--eikonal-weight 0.01 --length-weight 0.001 --ema-decay 0.997 \
--structured-init --structured-init-include-lane --structured-init-steps 600 \
--lane-prior-phi1 --lane-prior-phi1-mode replace --lane-prior-phi1-dash-gate
```

**All 20 knobs CERTIFIED. 1 MANDATORY addition vs a naive command (`--muon-lr 0.002`, A8 — NO-FAKE). 0 launch-blocking opens.** Defensible
warm-start-arm refinements (none block the burn): A6 `--l7-start-epoch 480`; A2 extend-Muon (+A3 `--anneal-epochs 1000`); B5 Muon-boundary
rewarmup; C2 lower length-weight only if lanes under-fit; C3 stage-dependent EMA; D1/D4 trajectory A/B. **BUILD/SKIP: SKIP the root-tracking
scheduler before the burn; parallel follow-on gated on a $0 anneal-shape A/B.**

## 6. Honest provenance
- **MEASURED (proven, real artifacts):** the entire axis recalled from the live n200 Muon arm (`run_muon.log` launch line + `muon_finisher_switch`
  JSON + `levelset_train_result.json` + `levelset_best.json` 0.0036976@ep1000) `[macOS-CPU advisory]`; CE/τ/l7 saturation epochs (DOE memo); the
  +8% Muon-boundary transient + monotone recovery (result-JSON history); the structured-init log (lane static band px=0; lane-prior-phi1 active,
  residual 1.9e-5); all flag names/defaults grepped against the real argparse (`train_levelset_witness_realized_through_R_mlx.py`, no flag invented);
  the anneal temp/LR table computed from the trainer's own `_softmax_temp_for_epoch`/`_lr` + the line-1905 Muon freeze.
- **DEEP-MATH (sound, secondary to the existence proof):** deterministic annealing (Rose 1998), critical slowing near an RD topological transition
  (Agmon–Tishby 2103.02646), Muon=spectral-norm steepest descent (2409.20325 / Keller-Jordan), eikonal viscosity-solution SDF, Chan-Vese length,
  Polyak–Ruppert EMA averaging, RAdam/warmup variance. The nonlinear-ID ~9 (recursive review) that de-prioritizes the root-cure.
- **NOT a score:** `[macOS-MLX/numpy advisory · NON-PROMOTABLE]`. The realized d_seg 0.003698 is the n200 surrogate; the **pointer 0.19110 is
  UNMOVED** and stays UNMOVED until a byte-closed n600 archive is scored by `upstream/evaluate.py`. This certification is a MEANS.
