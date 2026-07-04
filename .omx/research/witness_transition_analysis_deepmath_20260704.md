# Witness training TRANSITION-ANALYSIS — the micro deep-math of the curriculum regime changes (task #292-A)

**2026-07-04. Operator: "micro deep math from pre train to post train and interactions and transitions
and priming and all; this is an algorithm and deserves control systems and observability and telemetry."**
$0 analysis + measurement on #205's REAL trace (dir `experiments/results/levelset_n600_witness_20260703T120444Z`,
pid 29129, SACRED READ-ONLY). NO heavy/paid/GPU. **Pointer contest-CPU 0.19110 UNMOVED; ALL MEANS** — this
memo moves nothing; it is the control-theoretic instrument spec for the fresh seeded run. Every number is
tagged MEASURED (from `run.log` / `launch.sh`), DERIVED (from the registered laws), or A/B-OWED.

The witness training is a **controlled gradient flow** `θ̇ = −∇_g S_τ(θ)` on the Fisher manifold, run
through a **curriculum of regime changes** (priming → CE → tau_softplus → Muon → l7 → byte-close). A regime
change is a discontinuity in ONE of three operators: the **loss geometry** (Bregman/mirror map), the
**preconditioner** (metric `g`), or the **initial condition / boundary forcing** (seed, band, schedule). This
analysis is the per-transition micro-dynamics + the MEASURED signature on #205 + the CONTROL and TELEMETRY
implication of each. Substrate: `deepmath_amortizing_argmax_maslov_caustic_tau_eps_hbar_20260704` (8 laws),
facets 4 (`scaling_law_facet4_adiabatic_schedule_nucleation`) + 5 (`..._facet5_dynamic_control`), the
nucleation memory, and the master lever ledger.

---

## 0. The #205 configuration under analysis (from `launch.sh`, grep-verified)

`--curriculum` with: CE (ep0–299) → `--tau-softplus-start-epoch 300 --tau-softplus-tau 0.3` (ep300–725) →
`--muon-start-epoch 726` (ep726–999) → `--l7-start-epoch 1000` (NEVER fires; = `--epochs 1000`). Render
softmax temperature `--softmax-temp-start 1.0 --softmax-temp-end 0.05 --tau-anneal-shape cosine`. Seed:
`--structured-init --structured-init-include-lane --lane-prior-phi1 --lane-prior-phi1-mode replace`. MCF/SDF
controls: `--eikonal-weight 0.01 --length-weight 0.001`. Geometry: `--self-orient --n-dir-freqs 2`, `--hosc`
β1→4 annealed, `--mod-dim 32`. Optimizer: AdamW `--lr 1e-3 --lr-end 1e-4 --adam-beta2 0.999 --ema-decay 0.997
--accum-pairs 8 --grad-clip 1.0`. Transition treatment: `--stage-transition-reset-moments
--stage-transition-rewarmup-epochs 8 --stage-transition-rewarmup-shape linear`.

**The run reached ep450 (last verdict 2026-07-04T13:25Z), still in the tau_softplus stage.** The Muon (ep726)
and l7 (ep1000) transitions have NOT occurred → analyzed theoretically; the CE→tau transition (ep300) HAS
occurred and is the central measured finding.

**The full measured d_seg trace** (verdict stream, `[macOS-CPU advisory] NON-PROMOTABLE`):

| ep | 0 | 25 | 50 | 100 | 150 | 200 | 250 | 275 | **300** | 325 | 350 | 400 | **450** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d_seg | 0.7457 | .01030 | .00783 | .00615 | .00547 | .00509 | .00484 | .00476 | **.004752** | .005923 | .006267 | .006568 | **.006674** |
| d_pose | 1.887 | .0387 | .0173 | .00878 | .00423 | .00308 | .00276 | .00245 | .00210 | .00383 | .00253 | .00228 | .00195 |
| ep_loss | — | 735.0 | 633.3 | 563.4 | 536.7 | 521.2 | 514.7 | 513.5 | **148.6** | 147.5 | 141.3 | 134.1 | 129.8 |

The `levelset_best.json` freezes the global best at **ep300, d_seg 0.004752** — the run has not improved d_seg
in 150 epochs and is monotonically eroding.

---

## 1. PRIMING (pre-train → train, ep0): initial-condition selection on the manifold

### The micro-math
Priming sets `θ₀` = the point the flow starts from, which **selects the Morse–Smale basin** the descent
lands in (facet-4 §0: "facet 3 sets the initial field; facet 4 sets which basin the descent lands in"). Two
priming operators act before ep1:
1. **`structured-init`** — 600 pretrain steps fit the coordinate-INR `φ_θ` to a *painted partition target*
   (road/sky/hood + the openpilot deg-3 lane SDF). This is a **projection onto the seed manifold**: minimize
   `‖argmax φ_θ − Π_paint‖` where `Π_paint` is the hand-built partition. MEASURED convergence:
   `pretrain_direct_argmax_disagree_vs_part = 0.00035` — the witness fits the paint target to 3.5e-4.
2. **FiLM code differentiation** — the per-pair modulation codes start at **all-zero** (task-stated; the
   `mod-dim 32` code vector is `0` at init). At ep0 the render is **pair-invariant** (one code ⇒ one frame for
   all 600 pairs), so it can only match a single canonical layout.

### The MEASURED signature (does #205 show a code-unsharing phase? — YES)
- **ep0 verdict d_seg = 0.745683** despite the pretrain hitting 3.5e-4 vs its *own* paint target. The gap
  (0.746 vs 0.00035) is the priming diagnosis in one number: **the witness perfectly fits a laneless,
  pair-invariant target that is 74.6% wrong vs the per-pair GT argmax.** MEASURED `part_frac` @ep0 =
  `{road 0.248, lane 0.0, sky/undrivable 0.498, movable 0.0, hood 0.254}` — the seed is a single static
  layout with **zero lane and zero movable mass** (the nucleation failure, §2-hazard).
- **ep0→ep25: 0.745683 → 0.010299 = a 72× collapse.** This IS the code-unsharing (per-pair differentiation)
  transient: as the FiLM codes spread from `0`, each pair acquires its own render and the shared-frame
  disagreement evaporates. d_pose shows the same transient (1.887 → 0.0387, 49×). The transient is essentially
  complete by ep25 (the ep25→50 rate is already −9.9e-5/ep, an ordinary descent, not a 72× jump).

### CONTROL implication
The priming operator is a **hard admission gate**, not a soft knob. The controller MUST verify at ep0, BEFORE
committing GPU, that `part_frac[lane] > 0` (ledger acceptance gate). A zero-mass class at ep0 is
**structurally unrecoverable** (§2) — abort-and-reseed is the only correct action; no downstream lever can
birth it. The `structured-init --mode replace` seed is a **MEASURED no-op for the lane** (nucleation memo:
shallow lane SDF loses the argmax to the deep road static-core; lane_px stays 0) — the fresh run must use
`--lane-prior-phi1-mode paint` (#291, built `4f1580d0c`, measured part_frac[lane] 0→0.0064).

### TELEMETRY implication
Stream at ep0 + each early verdict: (a) **per-class `part_frac`** (the admission gate); (b) the
**code-differentiation scalar** `‖code_i − mean_j code_j‖₂` mean over pairs — a rising-then-plateau curve
whose knee marks the end of the unsharing transient; (c) `pretrain_direct_argmax_disagree_vs_part` vs the ep0
verdict — the **seed-fidelity-vs-GT gap** that exposes a well-fit-but-wrong target.

---

## 2. CE → tau_softplus (ep300): mirror-descent → mean-curvature-flow — THE erosion onset

### The micro-math (two registered laws collide here)
- **CE stage = mirror descent = natural gradient** (law `ce_softmax_mirror_descent_natural_gradient_v1`): CE
  is the Bregman divergence of the categorical entropy; `∇CE` in the softmax mirror map IS the Fisher natural
  gradient. It descends the **per-pixel margin** directly and is *class-mass agnostic* — a thin lane and a fat
  road get equal per-pixel pull. This is why CE cleanly drives d_seg to its floor.
- **tau_softplus stage = sharp-limit mean-curvature flow** (laws `tau_eps_hbar` + `mcf_minority_erasure_
  inevitability_v1`): the softplus-with-tau seg loss + the `length_weight` coarea term = a **perimeter
  gradient flow** = motion by mean curvature. MCF moves each interface at velocity `∝ −κ` (curvature). **Thin,
  high-curvature structures (the 6px lane, small movables) have the largest `κ` ⇒ erase FIRST** (Bronsard–Kohn;
  Allen–Cahn critical-nucleus theorem: a phase below the critical size shrinks to zero). The switch at ep300
  changes the **loss geometry** from margin-descent to perimeter-descent.

**The collision (Ch.6, MEASURED root cause).** #205 sets `--tau-softplus-start-epoch 300` AND
`--lane-band-start-epoch 300` to the SAME epoch. The log confirms both fire at ep300
(`curriculum_transition` + `lane_render_band_engage`). Three shocks superpose at one epoch: (i) loss-geometry
change (CE→MCF); (ii) lane-band render-authority engages; (iii) `--stage-transition-reset-moments` zeros the
AdamW `m/v` (fresh optimizer) with only an 8-epoch **linear** rewarmup. The keystone measured this collision
at **3.4× d_seg harm** in the ancestor probe.

### The MEASURED signature (the cleanest τ-creep in the trace)
- **d_seg REVERSES SIGN at exactly ep300.** MEASURED descent-rate `d(d_seg)/dep`:
  - CE terminal (ep275→300): **−4.4e-7/ep** (converged — the CE stage hit its floor ≈ the popout/flicker
    floor 0.0048–0.0052).
  - tau onset (ep300→325): **+4.68e-5/ep** — a **sign flip of ~106× the terminal CE magnitude**. The erosion
    shock.
  - tau steady (ep325→450): **+6.0e-6/ep**, decaying (the creep slows as the lane erodes toward its
    zero-mass fixed point).
- **Net creep ep300→450 = +0.001922 = +40.4% over the CE-best**, and still rising at ep450.
- **The surrogate↔verdict DECOUPLING** (the diagnostic fingerprint): while d_seg RISES, `ep_loss` FALLS
  monotonically (148.6 → 129.8, ep300→450). The MCF loss is being minimized *correctly*; it is simply
  **anti-aligned with the hard argmax verdict** for the un-nucleated thin lane. (The ep275→ep300 `ep_loss`
  drop 513.5→148.6 is a **loss-FORM rescaling**, NOT improvement — CE nats vs softplus-score units are not
  comparable; only within-stage slopes are meaningful.)
- **Not aliasing (yet).** DERIVED: at ep300–450 the render softmax-temp `τ_render` (cosine 1.0→0.05) is still
  **0.80 → 0.60** (interface half-width `τ/2` = 0.40 → 0.30), far above the sub-pixel aliasing floor
  (`τ_render→0.05` ≈ ep900). ⇒ **the ep300 erosion is the tau_softplus SEG-LOSS MCF acting on a zero-mass
  lane, not render aliasing.** (Aliasing is a distinct, LATER hazard, §hazard-3.)
- **Pose is decoupled (seg ⊥ pose, MEASURED FREE):** d_pose keeps improving straight through the seg erosion
  (0.00210@ep300 → 0.00195@ep450) — the pose store-nothing carrier is unaffected by the seg regime change.

### CONTROL implication (the τ-creep detector must fire at the ep300→325 verdict)
This is the exact **DIVERGING/ERASING** class of facet-5's controller: `r̂ ≥ +δ_creep` ∧ `net_Δd_seg > 0` ∧
`ep_loss ↓`. The monitor (`tools/witness_control_monitor.py`, BUILT `3db114735`) would flag it at ep325
(instead of the human noticing at ep425). The correct closed-loop actions, in priority: **(1)** the ep0 gate
should have prevented it (zero-mass lane ⇒ MCF can only erase); **(2)** given a nucleated lane, engage the
**per-class AREA constraint** (auction-MBO) to pin lane mass ≠ 0 — the principled cure for MCF minority
erasure; **(3)** ramp the **eikonal** at the MCF onset (§eikonal-ramp) to hold the interface sharp; **(4)**
**deconflict the ep300 collision** — move `--lane-band-start-epoch` to 350 and widen the rewarmup to 20-epoch
cosine (keystone Ch.6 L1+L2). The controller emits config-diffs only; it never launches (CONTAINMENT + P0
governor).

### TELEMETRY implication
Stream the **3-signal τ-creep tuple** every verdict: `r̂` (EWMA descent rate), `net_Δd_seg` since stage
start, and `sign(ep_loss slope)`. The DIVERGING classifier = all three adverse. Additionally stream
**per-class d_seg attribution** (which class is eroding) so the re-steer targets the right class — MEASURED
expectation: the creep is lane-dominated (nucleation memo; confirm at n600 when #205 frees, memory-gated to
avoid the >128 GB P0 crash).

---

## 3. tau_softplus → Muon (ep726): metric / preconditioner change (theoretical — not yet reached)

### The micro-math
Muon changes the **preconditioner `g`**, not the loss. AdamW is a **diagonal** metric (per-coordinate
`1/√v̂`). Muon is a **spectral** preconditioner: it orthogonalizes each 2D weight-matrix gradient via
Newton–Schulz (`--muon-ns-steps 5`) ⇒ a steepest-descent step in the **spectral norm** in weight space
(Bernstein–Newhouse). **NO-FAKE (keystone catch):** Muon is a FALSE FRIEND for "natural gradient" — it is
weight-space spectral steepest descent, NOT output-space categorical Fisher–Rao NG. Its measured −32% d_seg
(ancestor) is REAL but its attribution is CONJECTURAL (equally plausibly κ≈19 boundary-Hessian busting). The
switch rebuilds the optimizer to `MultiOptimizer([Muon (2D hidden weights), AdamW (biases/code/heads)])`,
re-inits state, and clears the spike-guard (the orthogonalized lower-LR step has a different loss scale).

**The cold-start spike (#269).** The fresh Muon momentum buffer `v = 0`. The first orthogonalized steps use
the *instantaneous* orthogonalized gradient (no momentum smoothing) ⇒ noisier, larger first steps ⇒ a
transient d_seg/loss spike right after the switch. This is the direct analogue of the ep300 moment-reset
shock, at the ep726 boundary.

**The launch-config predicts a COLD Muon at ep726** (see §muon-flag-verdict): `--muon-warm-start-momentum`
and `--muon-lr-final-frac` are ABSENT from this launch.sh ⇒ `warm_start=False` (cold zeros) and
`lr_final_frac=1.0` (scalar, non-decaying LR). So #205-as-launched will take the cold-start spike AND run a
flat Muon LR to ep1000.

**The nucleation over-ride (decisive).** Per the nucleation memo + law #8: **Muon sharpens existing
boundaries but CANNOT nucleate a zero-mass class.** #205's lane is at ~0 mass by ep726 (426 epochs of MCF
erosion). ⇒ the Muon stage is **predicted low-EV for d_seg** on #205: it will sharpen the already-excellent
smooth-class boundaries (road/sky/hood, IoU 0.95–0.99) but cannot rebuild the lane. This is why the
recommended path is a fresh SEEDED run, not resuming #205 into Muon.

### CONTROL / TELEMETRY implication
The controller must treat the switch epoch as a **fresh spike-guard window**: clear `recent_losses` (the code
does), and expect `r̂ ≥ 0` for a few verdicts WITHOUT classifying DIVERGING (a post-switch transient ≠
erosion). Distinguish the two by duration + `ep_loss` sign: a spike that recovers (`r̂→−` within `K` ticks)
is the cold-start transient; a sustained `r̂≥+δ ∧ net_Δ>0 ∧ ep_loss↓` is erosion. Stream: `warm_seeded_leaves`
(how many Muon `v` leaves were seeded from AdamW `m`; 0 = cold), `n_muon/n_adamw` param split, and the
post-switch spike magnitude vs the pre-switch median.

---

## 4. l7 (L∞ sharpening) — demoted defect (one line, per the DAG)

l7 penalizes the single **worst-margin pixel** (`L^p→L^∞` as `p→∞`), which in a piecewise-constant argmax
target is dominated by the **thinnest, highest-curvature boundary's jitter** (the lane edge) ⇒ it **amplifies
flicker rather than reducing MEAN argmax disagreement**, a deformation DISTINCT from and anti-aligned with the
tau viscosity flow (keystone: "l7 p→∞ remains a distinct deformation"; measured DEFECT: l7 RAISES d_seg). #205
correctly sets `--l7-start-epoch 1000 = --epochs` ⇒ **l7 never fires** (demoted from the default curriculum).

---

## 5. train → post-train (byte-close): EMA θ* → discretized codes → decode through R → argmax

### The micro-math
The training authority is the **continuous EMA shadow** `θ*` (decay 0.997). Byte-close applies three
non-continuous operators the training loop only STE-approximates:
1. **Discretization** — the FiLM codes + weights → quantized archive blob (MEASURED ~100 KB in the verdict
   stream; the rate term is `25·|archive|/37.5M`).
2. **The R round-trip** — `render(384×512) → bicubic↑ 874×1164 → uint8 → bilinear↓ 512×384 → SegNet argmax`.
   The uint8 quantization is a **set-valued (Clarke) subgradient through the argmax**; training STE's it
   (straight-through), byte-close realizes it exactly ⇒ a **surrogate↔exact gap** = the **STE-flicker floor**
   (keystone: the flicker floor is the SDE irreducible-noise floor, NOT removable by transition-easing).
3. **The frozen SegNet argmax** — the only authority; d_seg = HAMMING mismatch of two Laguerre labelings.

**Why CPU-bit-exact is the ONLY authority.** MLX-GPU is NOT bit-identical cross-process (MEASURED 28/28
tensors diverge). ⇒ the byte-close reproduction + `upstream/evaluate.py` MUST run **CPU-locked** for a
bit-exact, deterministically-reproducible verdict. The GPU verdicts in `run.log` are `[macOS-CPU advisory]
NON-PROMOTABLE` training-signal only; **no row here has moved the 0.19110 pointer.**

### CONTROL / TELEMETRY implication
The train↔byte-close gap is itself the highest-value telemetry: stream the **surrogate d_seg (MLX render
argmax) vs the exact CPU byte-close SegNet argmax** at each checkpoint. A widening gap = the STE-flicker floor
is being hit (further training is refining a boundary the quantize+R round-trip erases). The convergence
milestone is NOT another advisory verdict; it is **one byte-closed CPU exact row** (the fresh run's first
post-convergence deliverable, ledger §8).

---

## 6. LEVER INTERACTIONS — the cross-terms (synergistic vs antagonistic)

| interaction | class | evidence | mechanism |
|---|---|---|---|
| **seed × eikonal** | **SYNERGISTIC (survival partner)** | MEASURED (ledger §5): paint-seed retains **93% of the nucleated lane at σ0.8 (eikonal-0.05 regime)** vs **52% at σ1.5 (raw MCF, no eikonal)** | the eikonal pins `|∇φ|=1` so the seeded lane keeps a sharp `τ/2` interface and stays in the high-survival low-σ regime under MCF; without eikonal the seed smears then collapses |
| **τ_seg × eikonal** | **COUPLED (interface-width control)** | DERIVED (facet-4 §2.1, law `tau_eps_hbar`) | interface half-width = `τ/2`; as `τ` narrows the interface, eikonal must strengthen to keep it a valid unit-gradient SDF (§eikonal-ramp) |
| **amplify × seed-islands** | **SYNERGISTIC (multiplicative gate)** | MEASURED (ledger sweep-A #1): "amplify was a NO-OP without seed-islands" | `island_amplify` rides the signed margin of an existing island; a zero-mass lane has no island to amplify ⇒ `amplify_effect = f(seed_mass)·w`, and `f(0)=0`. #205 (seed-islands OFF, lane mass 0) ⇒ amplify inert |
| **Muon × warm-start** | **SYNERGISTIC** | DERIVED (#270/facet-1; code lines 3502–3556 levelset) | GAP2 (v←AdamW m) removes the cold-start spike; GAP1 (cosine LR→0.1×) escapes the flat-LR plateau; safe because at ep726 the AdamW `m` is well-adapted to the 426-epoch tau landscape |
| **length_weight × everything (MCF)** | **ANTAGONISTIC if raised** | MEASURED (ledger, 3 sweeps converged): `--length-weight` IS the MCF surface-tension erosion term | keep at 0.001; raising it accelerates the minority erasure (§2). The one lever that must stay SMALL |
| **tau-start × lane-band-start (ep300 collision)** | **ANTAGONISTIC (co-timed shocks)** | MEASURED (keystone Ch.6): 3.4× d_seg harm | deconflict: band→350, rewarmup→20-epoch cosine |

---

## THE EIKONAL-RAMP SCHEDULE (measured-justified; coupled to geometric-τ)

**Grounding (DERIVED, facet-4 §2.1 + law `tau_eps_hbar` `tau_interface_halfwidth`):** the SDF interface
half-width is exactly **`τ/2`**; the eikonal `λ_eik·(|∇φ|−1)²` pins `|∇φ|=1` so that half-width is a *real*
`τ/2` and does not "smear wider then collapse." **MEASURED anchor (ledger §5):** eikonal-**0.05 ↔ σ0.8 (94%
survival)**; the raw tau/MCF regime is σ1.5 (49% survival). The eikonal is the **critical survival partner** of
the seed (§6). The recommendation:

1. **CE stage (ep0 → tau_start): `λ_eik = 0.05` CONSTANT.** Hold the seeded-lane SDF sharp from init (facet-4;
   #205's 0.01 is 5× under-powered — it lets the seeded interface smear before the flow even engages).
2. **At tau_softplus engage (the MCF onset), eased over `--stage-transition-rewarmup-epochs 20` cosine:
   step `λ_eik` UP to hold the interface in the σ≤0.8 (94%-survival) regime as the perimeter flow turns on.**
   The MCF pushes the effective regime toward σ1.5 (49%); to STAY at σ0.8 the eikonal must at least ~double.
   **A/B bracket `{0.05, 0.10, 0.15}`** for the tau-stage plateau; measured survival predicts ~0.10 is the
   knee. Cap with `--eikonal-junction-relax` so triple junctions (Herring 120° angles) are not over-penalized.
3. **Geometric-τ coupling (only if a `τ_end < 1.0` A/B is run):** with a geometric render-τ anneal
   `τ_render: 1.0 → τ_end`, add the inverse coupling `λ_eik(t) = λ_eik,0 · (τ_start / τ_render(t))` so the
   eikonal tracks the narrowing interface (half-width `τ/2`) and holds `|∇φ|=1` across the anneal. At `τ_end`
   the eikonal reaches `0.05·(1.0/τ_end)` (e.g. `τ_end=0.25 ⇒ 0.20`).
4. **HONEST caveat (collapse to constant):** the fresh-run choice is `τ_end = 1.0` (MEASURED resolution floor;
   `τ=0.05` = 20× sub-pixel aliasing). With `τ_render ≈ 1.0` held, the render interface barely narrows ⇒ the
   §3 inverse ramp collapses to the **constant 0.05** the ledger already recommends. The RAMP is load-bearing
   only for (a) the tau_softplus-engage step-up (item 2, the MCF onset — this always applies) and (b) any
   `τ_end < 1.0` A/B (item 3). KEEP `--length-weight 0.001` throughout (the MCF driver; §6).

**One-line schedule:** `λ_eik(t) = 0.05` for CE; `→ {0.05,0.10,0.15}` A/B cosine-eased at tau-engage; `×
(τ_start/τ_render(t))` iff `τ_end<1.0`. Direction DERIVED + survival-MEASURED; plateau magnitude A/B-owed.

---

## THE MUON-FLAG VERDICT (grep of both trainers) — #272 is REAL, not fake-completed

**Both flags EXIST and are fully wired in BOTH trainers.** Grep receipts:
- **Levelset entry point** (`experiments/train_levelset_witness_realized_through_R_mlx.py`): `--muon-lr-final-frac`
  declared L4647 (default 1.0), `--muon-warm-start-momentum` declared L4658 (`BooleanOptionalAction`, default
  False). Consumed at the switch block L3502–3556 (GAP2: capture outgoing AdamW `m`, seed Muon `v` via
  `_seed_muon_momentum_from_adam`, cold-fallback on mismatch; GAP1: cosine-decay muon_lr→`muon_lr·final_frac`
  anchored on `muon_start_epoch`). Also handled on the resume-into-finisher path L3243–3258.
- **Base trainer** (`experiments/train_witness_realized_through_R_mlx.py`): declared L2861/L2871, consumed
  L2108–2171 (identical GAP1/GAP2 logic).

**Verdict: #272 is NOT fake-completed.** The flags are present, wired, tested-by-construction (both default to
the byte-identical off state — `final_frac≥1.0 ⇒ scalar LR`, `warm_start=False ⇒ cold zeros`), and log a
`muon_finisher_switch` record with `muon_warm_start_momentum` / `muon_warm_seeded_leaves` /
`muon_lr_decay_active`. **What is absent is their USE in THIS run's `launch.sh`** — #205-as-launched passes
neither flag ⇒ it will take a COLD Muon start with a flat LR at ep726. The #270 "improved-Muon RESTART" is a
SEPARATE planned intervention (resume from a pre-726 checkpoint WITH the flags on), not a property of the
currently-running config. **Recommendation:** since #205's lane is zero-mass (§3, Muon can't nucleate it), the
higher-EV path is the fresh SEEDED run carrying `--muon-warm-start-momentum --muon-lr-final-frac 0.1` from the
start, not a #205 Muon resume. If #205 IS resumed into Muon for the warm-start A/B (a clean L2/L5-principle
test per keystone), add both flags at the resume.

---

## TOP-3 TRANSITION HAZARDS the control system must guard

1. **CE→tau erosion / MCF minority erasure (ep300) — the hazard that ACTUALLY fired.** MEASURED: d_seg
   sign-flips +4.68e-5/ep at ep300, nets +40.4% by ep450, while ep_loss falls (surrogate↔verdict decoupling).
   Guard = the τ-creep detector (`r̂≥+δ ∧ net_Δd_seg>0 ∧ ep_loss↓`) firing at the FIRST post-transition verdict
   (ep325), + the pre-conditions (area-constraint, eikonal-ramp, ep300 collision deconflict). This is hazard #1
   because it cost #205 its d_seg goal and is invisible to the training loss.
2. **Zero-mass class at ep0 (nucleation) — the structurally-unrecoverable pre-condition.** MEASURED:
   `part_frac[lane]=0, part_frac[movable]=0`. A class born at 0 mass CANNOT be nucleated by any downstream
   lever (Allen–Cahn critical nucleus; Muon/amplify/persistence all no-op on zero mass). Guard = a HARD ep0
   admission gate (`part_frac[lane] > 0`, abort-and-reseed) — check the MEASURED seed log, never infer from
   flag presence (`--mode replace` is a measured no-op; use `--mode paint`).
3. **Optimizer/loss discontinuity shocks at EVERY stage boundary (ep300 moment-reset; ep726 Muon cold-start;
   ep900+ softmax-temp sub-pixel aliasing).** MEASURED: the ep_loss form-rescaling 513→148 at ep300 makes
   cross-stage loss comparison meaningless; the moment-reset + 8-epoch linear rewarmup is the collision shock;
   ep726 will add the cold Muon spike; DERIVED: `τ_render` reaches the aliasing floor (half-width 0.025) near
   ep900. Guard = per-boundary spike-guard RE-TREAT (clear `recent_losses` — the code does this), a 20-epoch
   cosine rewarmup (built, default-off — turn ON), warm-start the Muon, and floor `--softmax-temp-end` at the
   resolution scale (`~1.0`, NOT 0.05). The controller must distinguish a **recoverable boundary transient**
   (`r̂≥0` briefly then `r̂→−`) from **erosion** (`r̂≥+δ` sustained ∧ `ep_loss↓`) — same signal, opposite action.

---

## Canonical-equation registration (NO-FAKE honesty)

This transition analysis produces **no genuinely new generalizable LAW** — it produces (a) a MEASURED
real-run confirmation-anchor for the already-registered `mcf_minority_erasure_inevitability_v1` (the #205
ep300→450 creep: `dd_seg/dep` +4.68e-5→+6.0e-6, net +40.4%, with `ep_loss↓` — the strongest real-run
instantiation of MCF minority erasure to date), and (b) a DERIVED-but-unmeasured **eikonal-ramp coupling**
`λ_eik(t) = λ_eik,0·(τ_start/τ_render(t))` that is a COROLLARY of the registered `tau_interface_halfwidth`
(`half-width = τ/2`), NOT a new independent law. Both are recorded here **FORMALIZATION_PENDING** (the exact
forms + the #205 empirical anchor above are ready to flush) and should be added in the apparatus-hygiene pass
(master ledger §5.I: the 6 code-live lever laws + these two anchors are all un-flushed; JSONL registry
currently = 0). Registering a lone new entry into an empty JSONL while 14 code-defined laws sit un-flushed
would create exactly the DAG↔DSL↔equations drift the triality discipline forbids — so the honest action is the
memo anchor now + a batched flush later, NOT a risky isolated registry mutation in this $0 pass.

Sisters: `deepmath_amortizing_argmax_maslov_caustic_tau_eps_hbar_20260704` (laws #3/#7/`tau_eps_hbar`) ·
`lane_nucleation_failure_seed_above_critical_nucleus_20260704` · `scaling_law_facet4_adiabatic_schedule_nucleation_20260704`
(eikonal/τ) · `scaling_law_facet5_dynamic_control_self_convergence_20260704` (the controller) ·
`fresh_run_master_lever_ledger_20260704` (the launch gate) · DAG FEED-04a..04h. Task #292-A.
**ALL MEANS; pointer contest-CPU 0.19110 UNMOVED.**
