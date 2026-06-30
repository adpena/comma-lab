# Openpilot-seeded from-scratch witness CURRICULUM — complementary deep-math (5-lens) + the DSL WitnessProgram (adaptive)

**Date:** 2026-06-29T03:40:00Z · **Authority:** `[macOS-CPU advisory / design]` — this is a DESIGN + PROGRAM
(a MEANS), $0 CPU-only, NO training launch, NO MPS. **Pointer UNMOVED contest-CPU 0.19110** — only a
byte-closed n600 `upstream/evaluate.py` row from the witness moves it. NO-FAKE: every grounded claim cites a
MEASURED FEED; every estimate/hypothesis is labelled. **Cross-refs:** FEED-bu/-bv (measured per-stage d_seg
dirs + the "different stages need different treatment" incident), FEED-fz BUILD 1 (the wired reheat), FEED-fi/-fk/-fm
(Muon finisher = spectral conditioner + the freeze), FEED-fs (separatrix seed), anneal-memo
(`anneal_optimal_math_geometry_calculus_20260613.md`, the margin-resonance), `tac.witness_dsl`
(curriculum_dsl + campaign), CLAUDE.md "different stages need different treatment" + "never launch
non-resumable / per-stage checkpoints" + "deterministic reproducibility" + the WITNESS CAPSTONE frontier.

---

## 0. The proposed curriculum (recalled), verified against the MEASURED FEEDs

Recalled design: **S0 structured-init+lane-prior-phi1 (seed) → S1 short CE uniform → S2 tau_softplus(τ=0.3)
[REHEAT] → S3 l7_softplus + margin-weight engage [REHEAT] → S4 Muon finish (muon-lr 0.03) [REHEAT]**, SKIP
smooth + λ/σ; softmax-temp 1.0→0.05 throughout; MD-Decoupling; NCA stabilizer; pose on stored Quantizr sidecar.

**Per-stage d_seg directions — GROUNDED (MEASURED, MLX port n600, FEED-bv `:532/:538/:637`):**
CE 0.01045→0.00643 (↓) · **tau_softplus →0.00396 (THE primary drop, the min)** · smooth →0.00423 (↑ +6.8%,
DROP) · l7 →0.00369 (slow new min) · Muon = the conditioning finisher (PR95 L15 "THE drop"; witness value
PREDICTED ~6–9e-4, FEED-fi — NOT yet measured on the witness). The recalled "CE↓, tau↓, smooth↑, λ/σ↑,
Muon=THE drop" is **CONFIRMED measured.** The skip of smooth + λ/σ is correct AND structural: the level-set
trainer's `--curriculum` has only ce→tau→l7 stages (no smooth, no λ/σ) — the skip is not even a flag.

---

## 1. The 5-lens complementary deep-math pass (joint across the full space)

### Lens A — math / algebra (CONDITIONING). VERDICT: CONFIRM (Muon) + REFINE (muon-lr) + CONTRADICT (MD)
- **Muon = the clean finishing drop — CONFIRM.** Newton–Schulz orthogonalization drives every singular value
  of the momentum step →1 (unit spectral norm). At the end of l7 the decoder sits in an ILL-CONDITIONED valley
  (the boundary-annulus directions have tiny curvature, the bulk large); AdamW's *diagonal* preconditioner
  cannot fix the *rotational/off-diagonal* ill-conditioning, so it crawls the valley floor. Muon equalizes the
  spectrum → equal motion in all directions → escapes the elongated floor. This IS FEED-fk ("Muon = spectral
  conditioner of the low-energy boundary-annulus") and CLAUDE.md L15 "Muon is THE drop". GROUNDED.
- **muon-lr — REFINE (the recalled 0.03 is UNGROUNDED for this trainer).** FEED-fi/-fl MEASURED-recommend
  **muon-lr 1e-3…2e-3** (sweep ceiling 5e-3) because the finisher runs FLAT (the cosine is frozen, FEED-fm
  FIX-2) and a unit-spectral-norm step moves the boundary meaningfully even at a modest LR; the finisher's job
  is REFINEMENT (placement), not bulk re-fit. The recalled **0.03 is 6× above that measured ceiling** and would
  over-move the boundary on a near-converged partition (likely a confusion with a from-scratch Muon convention,
  e.g. nanogpt's ~0.02–0.05 on randomly-initialized matrices — a DIFFERENT regime). **DSL sets muon-lr = 2e-3**
  (the conservative measured band), exposed as a sweepable param. CALIBRATION: 0.03 = UNGROUNDED-hypothesis;
  2e-3 = GROUNDED (FEED-fi measured band).
- **MD-Decoupling "stable-by-construction" — CONTRADICT (a TRAINER-GAP, not a design choice).** The level-set
  trainer `experiments/train_levelset_witness_realized_through_R_mlx.py` has **NO `--optimizer` / `--md` /
  `--md-base` flag.** `md_decoupling.newton_schulz5` exists only as the Newton–Schulz routine *inside* Muon
  (FEED-fi). So "MD-Decoupling replaces manual reheat" is **MOOT** — MD cannot be selected. The WIRED
  stable-transition mechanism is the FEED-fz reheat (below), and it is GROUNDED. (MD-as-standalone-optimizer is
  a future trainer-gap item, §3.)
- **DM1 (Stiefel-W + code-spectral-entropy) — secondary, byte-free CONDITIONING lever, OFF the critical path.**
  Per the GR memo DM1 is DEMOTED to second-order (PR collapses while d_seg improves). It COMPOSES with the
  curriculum (`BASELINE.with_lever(*DM1Minimal())`) but is NOT in the opening; engage adaptively only if the
  FiLM rank-collapse becomes the binding limit. CALIBRATION: GROUNDED-but-demoted.

### Lens B — geometry (LEVEL-SET / SDF HOMOTOPY). VERDICT: CONFIRM (seed in-basin) + REFINE (CE shortened, not deleted)
- **The seed starts the homotopy IN-BASIN — CONFIRM.** `--structured-init` + `--lane-prior-phi1` inject the
  openpilot deg-3 centerline SIGNED-DISTANCE field into the φ1 (lane) channel of the structured-init pretrain
  target. FEED-fs MEASURED that centerline IS the Road↔Lane separatrix (residual **1.9e-5**) → the initial
  zero-level-set sits ON the true class boundary → the τ→0 continuation begins inside the correct basin instead
  of a random SDF. GROUNDED.
- **CE is SHORTENED, NOT eliminated — REFINE.** The seed supplies the SDF *shape* (boundary placement); it does
  NOT calibrate the per-pixel argmax *confidence* (the logit scale that couples to the render softmax-temp and
  to R-survival). A SHORT CE pass calibrates that confidence. So S1 "short CE" is correct — but the design
  should keep it short-but-nonzero (the opening uses ce_to=300, parameterized). CALIBRATION: ESTIMATE (the
  "short" length is a knob the adaptive policy can also tune; FEED-bv's CE-floor ~0.0064 is the descent target).

### Lens C — calculus / optimization (GRADUATED NON-CONVEXITY / NUMERICAL CONTINUATION). VERDICT: CONFIRM (homotopy + reheat) + REFINE (τ vs render-temp distinction) + new (cyclical restart hypothesis)
- **CE→tau→l7 IS graduated non-convexity — CONFIRM.** Start with a smooth, near-convex surrogate (CE), then
  progressively sharpen toward the true piecewise-constant argmax objective (tau_softplus, then l7's hard-pixel
  weighting). Each stage is a continuation step in the homotopy parameter. GROUNDED (this is the textbook GNC
  schedule; matches the measured monotone improvement CE→tau→l7).
- **Reheat = continuation step-size control — CONFIRM + GROUNDED magnitude.** At each homotopy step the
  objective changes; the optimizer's accumulated momentum is calibrated to the OLD objective. The FEED-fz wired
  levers — `--stage-transition-rewarmup-epochs` (LR floor→1× over N ep) + `--stage-transition-reset-moments`
  (zero stale AdamW 2nd-moments) — take a small, fresh continuation step into the new objective (predictor–
  corrector near a fold). FEED-bu MEASURED the magnitude: **floor 0.1× over ~8 ep** holds (n_skips=0, stable),
  where a FULL restart (1.0×) reproduced the v3 destabilization. So this is a PARTIAL restart, NOT full SGDR —
  GROUNDED. **DSL bakes rewarmup-epochs=8, floor=0.1, reset-moments=ON into the opening base.**
- **THE τ-ANNEAL — the biggest precision refinement (REFINE, avoids a FALSE contradiction).** The recalled
  "softmax-temp 1.0→0.05" and the anneal-memo's "miscalibrated cosine" are **DIFFERENT temperatures**:
  - `--softmax-temp` = the witness's **RENDER-partition** sharpness (SDF→argmax of the witness's OWN output).
    1.0→0.05 trains soft early (gradient flows through the partition) and sharp late (the partition is pinned so
    the through-R uint8 argmax is stable). FEED-fm FIX-2 FREEZES it at 0.05 during the Muon finisher (placement
    conditioned against a STATIONARY partition). This is CORRECT and is NOT what the anneal-memo falsified.
  - The anneal-memo's margin-resonance (`grad ∝ (1/T)e^{−Δ/T}`, peak at **T\*=Δ**, optimal global T = the
    e^{−Δ/T}-WEIGHTED-MEAN of live flip margins, clamped [0.3,0.6]) is about the **SEG-SURROGATE** temperature
    — here that is **`--tau-softplus-tau`**. The design's **τ=0.3 == the memo's reachability FLOOR Δ_min≈0.3**
    (the resonance for the *fixable* boundary flips). So the design's τ choice is **CONFIRMED by the memo**, not
    contradicted. The memo's open refinement (a margin-adaptive / weighted-mean τ schedule that beats the fixed
    0.3) is a **trainer-gap** (the level-set trainer uses a fixed `--tau-softplus-tau`; no adaptive-τ form
    wired) → §3, a future lever, NOT in the immediate design. CALIBRATION: τ=0.3 GROUNDED; adaptive-τ
    UNGROUNDED-future.
- **Within-stage cyclical restart (SGDR) to escape the long-900 plateau — NEW HYPOTHESIS.** The measured
  long900 result (tau running-min through ~900, l7 a *slow* new min) is a plateau; SGDR theory says a cyclical
  LR restart *within* a stage can jump to a new basin. The campaign engine's `Cycle`/`expand_cycles` already
  expresses this as a warm-start chain. CALIBRATION: UNGROUNDED-hypothesis (not measured on the witness) → the
  ADAPTIVE policy's EXTEND-vs-ADVANCE branch is the de-risked, measured alternative (extend only while the slope
  is still steep; advance on plateau) — preferred over a blind cyclical restart.

### Lens D — physics (SIMULATED ANNEALING + WARM RESTARTS). VERDICT: CONFIRM
- Both temperatures (seg-surrogate τ and render softmax-temp) are a simulated-annealing schedule: hot early
  (explore a smooth landscape), cold late (commit to a partition). Reheat = a warm restart (Loshchilov–Hutter
  SGDR), but PARTIAL (0.1× floor) — the physically-grounded "small perturbation to cross a saddle without
  melting the solution." GROUNDED. The anneal-memo §4 geometry (saturated softmax near a vertex → vanishing
  gradient) is the physics reason the render-temp must NOT sit at 0.05 during the gradient-bearing stages — and
  FEED-fm only freezes it at 0.05 for the Muon finisher (which is refinement, not bulk gradient). Coherent.

### Lens E — info / signal (NTK FREQUENCY VIEW). VERDICT: CONFIRM
- A Fourier-feature coordinate-INR has a frequency-dependent NTK learning rate (spectral bias: low-freq first).
  The seed supplies the LOW-FREQUENCY lane structure (the smooth centerline SDF) FREE → the curriculum jumps
  straight to the HIGH-FREQUENCY boundary annulus (FEED-fs measured the residual as a *high-frequency per-row*
  boundary-placement problem, not a smooth-width one). This is WHY the seed shortens CE (CE-from-scratch must
  learn the slow low-freq structure; the seed gives it). The base's anisotropic directional basis (freq-across
  32, freq-along 4) is the HIGH-across/LOW-along match to the boundary tangent field (the measured −48% lever) —
  SYNERGISTIC with the curriculum (basis-match PRIOR to capacity, per the frontier section). GROUNDED.

---

## 2. The biggest refinement (lead)

**τ is two temperatures, and the design already gets the load-bearing one right.** The anneal-memo's
margin-resonance does NOT condemn the design's schedule — it CONFIRMS it: the seg-surrogate τ=0.3 sits exactly
at the measured reachability floor Δ_min≈0.3 (the resonance for fixable flips). The `--softmax-temp` 1.0→0.05 is
a *separate* render-partition anneal (frozen at 0.05 only for the Muon finisher, FEED-fm) and is correct.
**Second:** muon-lr is **2e-3** (FEED-fi measured band), NOT the recalled 0.03 (6× too hot for a flat refinement
finisher). **Third:** MD-Decoupling is a trainer-gap (unwired) — the GROUNDED reheat (rewarmup 0.1×/8ep +
reset-moments, FEED-fz/-bu) IS the stable-transition mechanism, so "MD-replaces-reheat" is moot. **Fourth, and
the operator's riff:** the per-stage *budget* should not be fixed — the ADAPTIVE policy (extend while
descending, advance on plateau, rollback if a stage raises d_seg) is the measured-better answer to
"longer-per-stage pays."

---

## 3. Trainer-gap blockers (flags the recall named that DO NOT exist — NOT invented; recorded here)

| Recalled name | Status in the level-set trainer | The REAL mechanism (what the DSL emits instead) |
|---|---|---|
| `--margin-weighted-loss` / `--margin-weight-start-epoch` | **DO NOT EXIST** (they are OLD-trainer `train_witness_realized_through_R_mlx.py` flags) | the margin-weight is INTRINSIC to `l7_softplus` (5× weight on margin<`--l7-threshold` via `--l7-mult`); an additional finetune-only margin lever = `--margin-saliency-start-epoch` or `--lane-thin-start-epoch` set to the l7 boundary |
| `--optimizer md` / `--md-base` (MD-Decoupling) | **DO NOT EXIST** — no `--optimizer` flag at all | optimizer curriculum = AdamW (ce/tau/l7) → optional Muon finisher (`--muon-start-epoch`); stable transitions via `--stage-transition-rewarmup-epochs` + `--stage-transition-reset-moments` |
| "NCA stabilizer" (n_restarts≥2, keep-best) | **NOT a flag** | the wired stabilizers are `--grad-clip` (base 1.0) + `--spike-factor` (5.0) + the per-boundary spike-guard reset; multi-restart keep-best is expressed at the CAMPAIGN level (the adaptive ROLLBACK_BRANCH to the best ckpt), not a trainer flag |
| adaptive/weighted-mean τ schedule (anneal-memo crux) | **NOT wired** — `--tau-softplus-tau` is a FIXED scalar | use the fixed τ=0.3 (== Δ_min floor) now; a margin-adaptive `--tau-softplus-tau` form is a future trainer add |

These are the items to flag to the operator + the sister trainer-fixer (a6ccf5cd) as launch-config gaps. The DSL
**does not invent any of them** — `validate()` structurally refuses any flag not in the real argparse (every
emitted flag was checked: all 57 tests green, including `real_trainer_flags` membership).

---

## 4. The DSL deliverable (the triality program-view) — BOTH the fixed opening AND the adaptive policy

### 4a. The FIXED opening (`curriculum_dsl.openpilot_seeded_opening`)
A `WitnessProgram` for **S0 seed → S1 short-CE → S2 tau_softplus**, FROM SCRATCH (`resume_from=None`; the
structured-init IS the seed): `--structured-init` + `--lane-prior-phi1`(replace) + `--curriculum` +
`--tau-softplus-tau 0.3` + `--w-pose 0` (pose on the stored sidecar) + reheat (rewarmup 8 / floor 0.1 /
reset-moments) + temp 1.0→0.05 + eikonal 0.01 / length 0.001. **l7 PARKED at `epochs`** (no-op tail) so the
opening is exactly ce→tau and the trainer's `tau_start < l7_start <= epochs` assert holds; l7 + Muon are stacked
adaptively. `validate()` == [] (green). DETERMINISTIC: single recorded `--seed`; PRESERVE clause (per-stage +
≤25-ep ckpts, EMA-shadow); `--resume-from`-compatible.

### 4b. The ADAPTIVE stacking policy (`campaign`, the REACTIVE curriculum)
The curriculum is NOT fully fixed up front — the NEXT stage is STACKED from the prior stage's MEASURED d_seg
trajectory off the per-stage checkpoints:
- `StagePolicy` — deterministic thresholds (trailing-window least-squares slope of the realized-through-R d_seg
  verdicts).
- `decide_next_stage(trajectory, policy, final_ckpt, best_ckpt)` — **PURE** function → `StageDecision`:
  **EXTEND** (slope ≤ descend_slope, still descending → longer-per-stage, resume from final) · **ADVANCE**
  (|slope| < plateau → stack the next stage + reheat, resume from final) · **ROLLBACK_BRANCH** (final rose >
  rise_tol above best → a d_seg-raising stage → resume from the BEST preserved ckpt and skip).
- program transforms: `extend_stage` (push l7 boundary out, more tau) · `advance_to_l7` (l7 fires at the resume
  epoch, reheat at the tau→l7 boundary) · `advance_to_muon` (Muon finisher, **muon-lr 2e-3** FEED-fi, tau +
  render-temp frozen at 0.05, reset-moments) · `stack_next_program` (dispatch on the decision).
- `plan_adaptive_step(prev, prev_out_dir, log_path, advance_to, out_dir, next_log, policy)` — the closed loop:
  harvest the measured trajectory → decide (pure) → build+validate the warm-started continuation → **EMIT** its
  daemon launch command + the recorded decision. **CONTAINMENT: emit-only, never auto-fires.**

**DETERMINISTIC-REPRODUCIBLE BY CONSTRUCTION:** `decide_next_stage` is a pure function of (the on-disk
trajectory rows, the policy thresholds) — no RNG, no wall-clock, no env. Same `(seed, measured-trajectory)` →
same stacked curriculum; the `StageDecision` is returned + recorded (`to_record()`). The per-stage + EMA-shadow
+ `--resume-from` checkpoints are the substrate (the never-launch-non-resumable / per-stage-checkpoint
non-negotiable). 16 new tests (57 total green) lock: opening validates / from-scratch / d_seg-only / τ=0.3 /
reheat / l7-parked / single-seed-determinism / smooth-skip-structural / curriculum-ordering-refused;
decision EXTEND/ADVANCE/ROLLBACK/empty/determinism; transforms validate + launch-valid + muon-lr=2e-3-not-0.03;
plan_adaptive_step emit-only round-trip + rollback-to-best-preserved-ckpt; stage_trajectory parse.

---

## 5. What's needed before the from-scratch GPU launch
1. **Operator bless of the final opening stage design** (esp. w-pose=0 d_seg-only + ce_to/tau_window budgets +
   muon-lr 2e-3) — these are design tradeoffs, council/operator-gated, not unilateral.
2. **GPU steer** (one GPU; no autonomous heavy-GPU launch per CONTAINMENT) — the opening is `resume_from=None`
   FROM SCRATCH; the adaptive continuations warm-start from the preserved tau/l7 ckpts.
3. **Trainer-gap resolution (optional, with sister a6ccf5cd):** if an additional explicit margin-weight lever is
   wanted at l7, wire `--margin-saliency-start-epoch`=l7_start (the flag EXISTS); MD-Decoupling-as-optimizer and
   adaptive-τ remain future trainer adds (not blockers — the GROUNDED reheat + fixed τ=0.3 suffice).
4. **The verdict is the byte-closed n600 `upstream/evaluate.py` row** at the composed θ\* (opening + the
   adaptively-stacked l7 + Muon) — pointer UNMOVED 0.19110 until then. MEANS≠ENDS.
