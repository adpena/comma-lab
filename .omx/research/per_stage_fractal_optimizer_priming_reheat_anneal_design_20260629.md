# Per-Stage Fractal-Optimal Optimizer Family — PRIMING / REHEAT / ANNEAL — DESIGN

Tag: **[research / advisory / design]** · 2026-06-29 · NON-COMMITTED design memo + MLX pseudocode sketch.
NO GPU run, NO score claim. Means≠ends firewall: the optimizer is a MEANS; the END is realized
through-R `d_seg` descent. Every component below is justified by its effect on d_seg-descent or
conditioning-health, never by optimizer-elegance.

> Mission framing: this is a witness-capstone d_seg lever (CLAUDE.md §"THE CURRENT FRONTIER … WITNESS
> CAPSTONE"). It does NOT itself move the exact pointer; it is infrastructure in service of an imminent
> d_seg-improving witness row. Build the MINIMAL version first; gate every increment behind its own A/B.

---

## 0. The deep-math spine (state this first — everything else serves it)

Witness: coordinate-INR → K=5 SDFs `φ_k(x; θ, code)`; per-pair FiLM `M = code @ Wᵀ` modulates the SDFs;
argmax_k φ_k = 5-class partition; frozen SegNet argmax scores d_seg THROUGH R.

**THE DISEASE (DM1, measured):** `M`'s participation ratio collapses 3.34→1.19 across the curriculum;
`cov(code)` and `WᵀW` commute 96% (a *multiplicative resonance* — they share an eigenbasis, so the
modulation energy concentrates into a shrinking subspace). Measured correlation; elevated to the binding
d_seg-plateau hypothesis. **Causation is UNPROVEN until the §4 smoke** (does fixing PR move d_seg?).

**The root-cause closed form (MY-DESIGN, exact):** participation ratio
`PR(A) = (tr A)² / ‖A‖_F²` (Rényi-2). For centered codes, `E[M Mᵀ] = W·cov(code)·Wᵀ`. If `W` is
**orthonormal (Stiefel, WᵀW=I)** then `W` is an isometry on its row space, so the nonzero spectrum of
`E[M Mᵀ]` EQUALS the spectrum of `cov(code)`:

```
WᵀW = I   ⟹   PR(M) = PR(cov(code))          [the resonance cannot concentrate through W]
```

So the disease is cured **at the root, byte-free**, by TWO structural moves, not by the optimizer:
1. **Stiefel-orthonormal `W`** — removes W's eigenbasis-shaping contribution to the resonance.
2. **Spectral-entropy penalty on `cov(code)`** — keeps the code spectrum spread → `PR(cov(code))` high →
   `PR(M)` high by the identity above.

**Consequence for the optimizer (the prioritization):** with the structural cure in place, the optimizer
only has to (a) KEEP `W` on the manifold and (b) MAINTAIN the spread during the collapse-prone stages.
The bespoke optimizer is therefore a *maintenance/transient* device, NOT the cure. This is exactly why
the §2 minimal version is so small.

**Cheap differentiable form (implementable, no eigendecomp in the hot loop):**
`PR(cov(code)) = (tr C)² / ‖C‖_F²`, `C = cov(code)` over the batch (22×22). Penalty
`L_spec = -β · log[(tr C)² / ‖C‖_F²]` (maximize PR). This is O(22²), fully differentiable, and the
penalty quantity IS the telemetry we monitor. (True Shannon spectral entropy needs an eigendecomp; the
Rényi-2 / PR surrogate is cheaper AND is literally the metric we want to raise.)

**Nuance (coordinator fold-in 2026-06-29; INSPIRE, evaluated):** spectral-entropy is a **CAPACITY** lever,
NOT an identifiability/disentanglement lever. A uniform code spectrum = maximally NON-identifiable (the
latent is only defined up to permutation+scaling, which FiLM's `W` absorbs exactly: `ĉ=ΠΛc ⟹ W'=WΛ⁻¹Π⁻¹`
gives identical output) — but for pure d_seg capacity that is *exactly what we want*: spread the spectrum
so all ~22 directions are live and the partition can use full capacity. So design intent everywhere below
is "**keep all directions live / preserve rank**," never "disentangle." (Identifiability ≠ conditioning.)

---

## 1. PROVEN frame: optimizer = steepest descent under a norm (per-stage norm choice)

Bernstein & Newhouse, *Old Optimizer, New Norm: An Anthology* (arXiv **2409.20325**, PROVEN): after
switching off EMA, each first-order optimizer = steepest descent under a particular norm, and the
**modular norm** says *different tensors should get different norms based on their role in the network.*
This is the theoretical license for per-group AND per-stage norm switching.

**GR reframe (coordinator fold-in 2026-06-29; INSPIRE):** the frozen SegNet scorer = a **FIXED Fisher
background metric `G`** (we are matter on a fixed background; the metric does not co-evolve). Choosing the
per-stage/per-group NORM is choosing the **descent geometry in that fixed `G`**; the byte-free
Sinkhorn/Muon conditioning is the **vielbein gauge that locally flattens `G`** so the steepest-descent
step is taken in a well-conditioned (≈ identity-metric) frame. This is a useful conceptual lens, not a new
mechanism — it re-says "match the optimizer norm to the local curvature of the fixed scorer metric."

| Optimizer | Norm geometry | What it does to the update | Matched stage/group role |
|---|---|---|---|
| **Adam/AdamW** (1711.05101) | diagonal / per-coord (≈ max-norm whitened) | fast independent per-coordinate progress | EXPLORE (rank already high); small readout/latent groups |
| **Muon** (Keller Jordan; Newton–Schulz orthogonalize momentum) | **spectral** (RMS→RMS operator norm) | spreads update across ALL singular directions → **anti-collapse, full-rank step** | rank-collapse zone + EQUILIBRATE; the big trunk matrix |
| **SinkGD** (2502.06742; Sinkhorn row/col L2) | **doubly-stochastic** | equalizes per-row (channel) & per-column (code-dim) gradient energy → **conditioning-normalizing** | the FiLM-W conditioning matrix (rows=channels, cols=code-dims) — the resonance locus |
| (SWAN 2412.13148) | normalize+whiten | stateless precond reference | prior art that motivates SinkGD |

**Why SinkGD specifically matches FiLM-W (MY-DESIGN, SPECULATION until measured):** the disease is
energy concentrating into a subspace shared by code-dims and channels. Doubly-stochastic (Sinkhorn)
balancing forces *every* row and *every* column of W's update to carry equal L2 energy → no single
channel/code-dim can dominate the update → it directly fights the multiplicative resonance on the
transient. Muon (spectral) spreads across singular *directions* but does not equalize the row/col
*assignment*; for a conditioning matrix the row/col balance is the more targeted invariant. This is a
testable claim, not a proven one.

### Per-stage norm assignment (the curriculum is 4 different jobs)

| Stage | Job (temperature) | Rank regime | Trunk norm | FiLM-W norm | Why (deep-math) |
|---|---|---|---|---|---|
| **CE** | EXPLORE (τ=1.0) | rank HIGH (~3.34); implicit τ-noise keeps it up | **Adam** | Adam | coords ~independent; cheap broad moves; collapse not yet active — no need to pay for orthogonalization |
| **tau_softplus** | SHARPEN (τ:1.0→0.05) | **collapse ONSET** (grad concentrates on boundary px as τ→0) | **Muon** | **SinkGD** + Stiefel | sharpening concentrates the update → spectral spread (trunk) + doubly-stochastic balance (W) resist concentration exactly where it starts |
| **l7_softplus** | MARGIN REFINE (p=7 ≈ soft-L∞) | **deep collapse** (hyper-local grad; near-stationarity → collapse peaks) | **Muon** | **SinkGD** + Stiefel | only the hardest boundary px gradient; full-rank step is essential to not freeze into the collapsed subspace; **REHEAT active here** |
| **muon_tail** | EQUILIBRATE | spread to max rank, settle | **Muon** | Muon/Stiefel | spectral norm → well-conditioned fixed point; then ANNEAL into EMA shadow + Fisher-Rao re-orthonormalize |

The two stages where the bespoke geometry earns its keep are **tau** and **l7** — the rank-collapse zone,
exactly where Hebbian theory predicts collapse peaks (next section).

---

## 2. THERMAL SCHEDULE: PRIMING / REHEAT / ANNEAL

**Thermal / latent-SDE view (MY-DESIGN interpolation across two real literatures):** each stage = an
environment with a diffusion (noise) covariance; the curriculum is a thermal anneal with τ as temperature.

Grounding split (NO-FAKE attribution fix):
- **WD → low-rank bias** is PROVEN in *Neural Rank Collapse* (arXiv **2402.03991**) and *WD induces
  low-rank attention layers* (arXiv **2410.23819**): distance of W to rank-K is bounded ∝ 1/λ_WD; the bias
  sharpens near critical points. (NOT the paper the grounding attributed it to.)
- **WD → Hebbian alignment near stationarity, and injected NOISE → anti-Hebbian** is PROVEN in
  *Homeostatic Ubiquity of Hebbian Dynamics in Regularized Learning Rules* (arXiv **2505.18069**,
  Koplow–Poggio–Ziyin).
- **The synthesis** — "the curriculum's implicit τ-noise is the anti-collapse force; it *vanishes* as τ
  anneals; so collapse peaks late-in-stage; so REHEAT must re-inject that vanishing noise" — is OUR
  interpolation across the two, **MY-DESIGN**, not a single theorem.

### PRIMING (Muon-primed warm-up at every stage transition)

Operationalizes the binding non-negotiable *"different stages need different treatment + RE-TREAT at
transitions; stage transitions must RESET moments, not inherit"* (MEMORY: different_stages_need_different_treatment).

At each transition: **reset/decay optimizer moment buffers** (do NOT inherit), then run a short
(~100–300 step) **Muon-primed warm-up** with a linear LR ramp, regardless of the incoming stage's main
geometry. Why Muon for priming specifically: it is the *anti-collapse* geometry, so each stage begins from
a spectrally-spread, full-rank update basis and **cannot inherit the prior stage's collapsed subspace**.
(General warmup is known to stabilize by avoiding early high-curvature steps; Muon-priming additionally
*guarantees* the early steps are full-rank.)

### REHEAT (controlled re-injection of a stage-noise-COVARIANCE shift — SINGLE duty: anti-collapse)

**Refined per coordinator fold-in 2026-06-29 (INSPIRE, evaluated; high-value to test, not a mandate).**
Three independent lines converge on "stage noise shifts": (1) Hebbian (arXiv **2505.18069**) — noise is the
anti-collapse / rank-preserving force that VANISHES as τ-relaxation noise anneals (→ collapse peaks late
in stage); (2) our OWN measured phenomena — **ringing**, the **stage-transition perturbation spike** (the
margin-engage spike), and warmup; (3) a latent-SDE line — **DROPPED as a false friend** (see firewall §7).

So REHEAT is the **controlled re-injection of a stage-noise-COVARIANCE shift** (NOT a generic LR bump),
**SINGLE duty = anti-collapse** (Hebbian). Within tau/l7, monitor `PR(M)`, `PR(cov(code))` every N steps;
when PR drops below threshold (collapse detected):
- (a) **Covariance-shifted noise** — re-inject anti-collapse noise concentrated on the *collapsing*
  directions: `ε·N(0,Σ_reheat)` added to the codes, with `Σ_reheat` aligned to `cov(code)`'s SMALL
  eigen-directions (and/or W's small singular directions). This re-injects exactly the rank-preserving force
  that τ-annealing removed. Intent = "keep all directions live late," NOT "disentangle."
- (b) **LR warm restart** — SGDR cosine restart (arXiv **1608.03983**) as the accompanying step-size cycle.

**Empirical calibration (key — not arbitrary):** set the reheat magnitude + `Σ_reheat` against our
MEASURED signatures — the ringing amplitude and the stage-transition (margin-engage) spike. The reheat
covariance should restore the noise floor to roughly the early-stage (pre-anneal) level that kept PR high,
read off from the same telemetry. This makes reheat a measured controller, not a free hyperparameter.

**Novel vs prior art:** SGDR restarts on a FIXED schedule with isotropic/no noise; here the restart is
**PR-triggered** AND carries a **calibrated covariance-shifted noise** matched to the collapsing subspace —
a closed-loop thermal controller. **EV stays MED-LOW:** evaluate only if fixed SGDR-at-boundaries +
the §0 structural cure prove insufficient — test in that order; it is a SECOND byte-free anti-collapse
route, redundant-by-design with Stiefel/spectral-entropy (so it must show marginal PR-or-d_seg lift over
the structural cure to earn inclusion, else it is double-counting the same anti-collapse force).

### ANNEAL (to the EMA fixed point — the deploy point)

The deployed weights are the **EMA(0.997) shadow** = an iterate average = Polyak-Ruppert / "slow weights"
(connects to Lookahead [Zhang et al., id ~1907.08610, unverified-this-session] and Schedule-Free's
iterate averaging [Defazio, facebookresearch/schedule_free]). In muon_tail and at each stage end,
cosine-anneal LR to a small floor and let noise vanish, so the live iterate oscillates *tightly* around
the average → the average is a good fixed point.

**Fisher-Rao fold-in (NO-FAKE: claim true, original citation 2307.10644 WRONG):** the arithmetic EMA of an
orthonormal `W` leaves the Stiefel manifold (Euclidean average of manifold points ∉ manifold). This is
STANDARD geometry (Karcher/Fréchet mean; Stiefel retraction — Cayley arXiv **2002.01113**; Momentum
Stiefel arXiv **2205.14173**), NOT what Nielsen's 2307.10644 (Fisher-Rao distance between Gaussians) says.
**Fix:** re-orthonormalize the EMA shadow's `W` before every deploy/eval read via polar projection
`W ← W (WᵀW)^{-1/2}` (one Newton–Schulz pass on the WEIGHT). Cheap, reuses the Muon kernel.

---

## 3. FRACTAL per-group treatment (prime→reheat→anneal × norm, recursively)

| Group | Norm | WD policy | Noise / reheat | LR scale (modular-norm) | Why |
|---|---|---|---|---|---|
| **INR trunk** | Adam(CE)→**Muon** | decoupled WD on **WD↓ schedule** (1e-2→1e-4) | SGDR restart | RMS fan-in/out | largest matrix; spectral step keeps representation full-rank; WD↓ defers the WD→low-rank bias to never bite the trunk |
| **SDF heads (5)** | **Adam** | light WD 1e-4 | none | readout norm | small linear readouts; per-coord is adequate (modular norm: readout ≠ hidden) |
| **FiLM-W [Stiefel]** | **SinkGD** + polar retraction | **WD = 0** (Hebbian non-negotiable; WD on a manifold param is harmful/meaningless) | PR-triggered noise on small σ-dirs; **EMA re-orthonormalize** | per-tensor | the disease locus: doubly-stochastic balance + manifold constraint + no-WD + anti-collapse noise all target HERE |
| **code manifold (600×~22)** | **Adam** + `L_spec` penalty | **WD ≈ 0** (WD shrinks latents → collapse) | PR-triggered isotropic code noise | per-tensor | the OTHER half of M; spreading `cov(code)` is half the cure; per-coord geometry + structured anti-collapse penalty |
| **palette (chroma/value)** | Adam | light WD 1e-4 | none | LR ×0.3 | low-dim nuisance; cheap geometry suffices |
| **pose-FiLM** | Adam | — | none | LR ×0.1, **freeze after CE** | pose is SOLVED via stored-target sidecar (means≠ends) — do NOT spend optimizer complexity here |

**Composition without double-counting (the locked fold-ins):**
- *WD↓ (trunk) vs noise↑ (W, code)* are the SAME thermal control on two knobs, but ROUTED to DIFFERENT
  params (WD-control → trunk; noise-control → conditioning). Both are anti-collapse, so applying both to
  the *same* param would double-count — the routing prevents it. Clean separation.
- *SSinkGD direction vs modular-norm magnitude*: Sinkhorn sets the update DIRECTION (balanced), modular
  norm sets the per-group STEP SIZE. Direction ⊥ magnitude → compose, no double-count.
- *`L_spec` penalty vs code-noise-reheat*: both raise `PR(cov(code))`, so to avoid double-counting they
  are active at DIFFERENT times — `L_spec` is the always-on STEADY-STATE force; reheat-noise is the
  TRANSIENT, PR-trigger-gated force.
- *EMA re-orthonormalization* acts on the SHADOW (deploy read), orthogonal to SinkGD/Muon on the LIVE
  weights → no interaction.

---

## 4. The MINIMAL version (the 80/20) — build THIS first

Because §0 proves `PR(M) = PR(cov(code))` is fixed **by construction** by Stiefel-W + code-entropy, the
structural cure dominates the optimizer-geometry refinements. The minimal optimizer:

1. **Stiefel-W (no-WD):** periodically polar-project `W ← W(WᵀW)^{-1/2}` (one Newton–Schulz pass on the
   weight — kernel already exists for Muon) and set `WD(W)=0`. [root cure half 1]
2. **Code spectral-entropy penalty:** add `L_spec = -β·log[(tr C)²/‖C‖_F²]`, `C=cov(code)`. [root cure half 2]
3. **Per-stage moment-reset + Muon-prime:** at each transition, reset moments + short Muon warm-up; SGDR
   cosine restart at stage boundaries (NOT closed-loop). [the "re-treat at transitions" non-negotiable]
4. Keep AdamW everywhere else; keep the existing Muon tail.

**NOT in minimal (the incremental A/B ladder):** SinkGD-on-W, closed-loop PR-triggered reheat, full
per-group fractal (palette/pose-film custom norms), Fisher-Rao EMA re-orthonormalization (add as a
correctness fix early — it is cheap), modular-norm per-group LR.

Minimal cost ≈ near-zero: Newton–Schulz already in toolbox; `L_spec` is O(22²); moment-reset is a buffer
clear. It captures the ROOT cure + the transition non-negotiable. Everything else is gated incrementally.

---

## 5. Fold into the v2 build `--optimizer {adamw,sinkgd,muon,custom}` + DSL lever shape

- `--optimizer adamw` — baseline arm.
- `--optimizer muon` — Muon-everywhere (existing).
- `--optimizer sinkgd` — SinkGD-everywhere (the FiLM-W geometry ablation arm).
- `--optimizer custom` — the per-group/per-stage scheduler, configured by a DSL block; each row is an
  independently A/B-able lever (matches "turn any knob anywhere, map the curves"):

```yaml
optimizer.custom:
  groups:
    film_w:    {norm: sinkgd, stiefel: true,  wd: 0.0, ema_reorthonormalize: true, reheat: pr_triggered}
    code:      {norm: adam,                    wd: 0.0, spectral_entropy_beta: 0.01, reheat: pr_triggered}
    trunk:     {norm: muon,                    wd_schedule: [1e-2, 1e-4],            reheat: sgdr}
    heads:     {norm: adam,                    wd: 1e-4}
    palette:   {norm: adam,                    wd: 1e-4, lr_scale: 0.3}
    pose_film: {norm: adam,                    lr_scale: 0.1, frozen_after: ce}
  stages:
    ce:           {trunk: adam, prime: none}
    tau_softplus: {trunk: muon, prime: muon, reheat: on}
    l7_softplus:  {trunk: muon, prime: muon, reheat: on}
    muon_tail:    {trunk: muon, prime: muon, anneal: cosine_to_floor}
  transition:  {reset_moments: true, prime_steps: 200, prime_lr_ramp: linear}
  reheat:      {monitor: pr_film, threshold: 2.0, restart: sgdr_cosine,
                noise_eps: 1e-3, noise_cov: small_eig_aligned,   # covariance-shift, not isotropic
                noise_calibrate_to: [ringing_amp, margin_engage_spike]}  # measured, not arbitrary
```

Minimal version = set `film_w.norm: muon` (instead of sinkgd), `*.reheat: sgdr` (no pr_triggered),
`film_w.ema_reorthonormalize: true`, keep stiefel+wd0+spectral_entropy. One YAML diff = the first A/B.

---

## 6. Cheapest $0 validation smoke (the means/ends firewall, built in)

Existing MLX trainer, MPS-as-TRAINING-gradient (allowed; NEVER as authority) or local CPU; n=24–48 pairs;
run a few hundred steps INTO the tau_softplus stage (where collapse bites); same seed across arms.

Arms: **A0** baseline AdamW · **A1** +Stiefel-W(no-WD) · **A2** +code-spectral-entropy · **A3=A1+A2**
(the minimal version).

Measure per step: `PR(M)`, `PR(cov(code))`, and realized-through-R `d_seg` on the n-subset (advisory,
`[macOS-MLX research-signal]`, NOT a score).

**Falsification thresholds (decisive either way, $0):**
- A3 must HOLD `PR(M) ≥ ~3.0` through the tau anneal (baseline collapses to ~1.2). If not → Stiefel+entropy
  theory FALSIFIED at implementation; re-open.
- A3 must LOWER advisory d_seg ≥ ~10% rel. vs A0 at equal step budget, monotone (no chaos). 
- **The firewall:** if PR holds (means fixed) but d_seg does NOT improve (end unmoved) → **DM1 is NOT the
  binding d_seg cause** — a major finding (we were treating a symptom), and the whole optimizer program is
  de-prioritized in favor of finding the real d_seg lever. This is the means≠ends test, made structural.

---

## 7. Honesty firewall

**PROVEN (real, verified this session):** norm=steepest-descent + modular norm (2409.20325); Muon=spectral
(Keller Jordan); SinkGD=doubly-stochastic Sinkhorn (2502.06742); SWAN (2412.13148); WD→low-rank
(2402.03991, 2410.23819); WD→Hebbian-near-stationarity & noise→anti-Hebbian (2505.18069); SGDR (1608.03983);
AdamW (1711.05101); Schedule-Free iterate averaging (Defazio); Cayley/Stiefel (2002.01113); Momentum
Stiefel (2205.14173); Sophia (2305.14342); Lion (2302.06675). The identity `WᵀW=I ⟹ PR(M)=PR(cov code)`
is elementary linear algebra (PROVEN).

**NO-FAKE flags (citation errors in the grounding, corrected here):**
- arXiv **2307.10644** is Nielsen, *Fisher-Rao distance … between multivariate normals* — it does NOT
  support "EMA of orthonormal W drifts off Stiefel → re-orthonormalize." Claim is TRUE (standard manifold
  geometry) but the id is WRONG; use Cayley/Karcher grounding (2002.01113) instead, or drop the id.
- The "decay-induced low-rank collapse monotone in WD, peaks near stationarity" result belongs to the
  rank-collapse literature (**2402.03991** / **2410.23819**), NOT to 2505.18069 (which is the Hebbian/noise
  paper). Both real; the attribution must be SPLIT.

**MY-DESIGN (original, unproven):** per-stage norm assignment (Adam→SinkGD-on-W→Muon→Muon); Muon-priming +
moment-reset at transitions; closed-loop PR-triggered reheat (vs fixed SGDR); the per-group fractal table;
WD-to-trunk / noise-to-conditioning routing to avoid double-counting; the §0 root-cure argument; the DSL
lever shape; the 4-arm dual-PR/d_seg falsification smoke.

**SPECULATION (plausible, unmeasured):** SinkGD > Muon for FiLM-W specifically; closed-loop reheat > fixed
SGDR; reheat covariance-shift adds lift over the structural cure; exact β, PR threshold, prime_steps; and
— most importantly — **that the resonance is causal for d_seg at all** (the §6 smoke tests this; until
then DM1-as-binding-cause is a correlation-elevated hypothesis, not proven causation).

**DROPPED as a FALSE FRIEND (coordinator correction 2026-06-29):** the "reheat also buys code
identifiability/disentanglement (latent-SDE)" justification is REMOVED. Reasons: (i) the latent-SDE
identifiability theorem identifies the latent only up to permutation+scaling, which FiLM's `W` already
absorbs (`ĉ=ΠΛc ⟹ W'=WΛ⁻¹Π⁻¹`, identical output); (ii) it cures rotational non-identifiability, NOT our
rank-collapse — identifiability ≠ conditioning. Reheat is therefore SINGLE-duty anti-collapse on solid
Hebbian ground. (The latent-SDE arXiv id relayed by the coordinator is unverified-this-session and not
relied upon, since the line is dropped.) Spectral-entropy is reclassified as a CAPACITY lever (uniform
spectrum = all directions live for d_seg), not an interpretability lever.

**INSPIRE folded in (DESIGN, not proven):** GR "matter on a fixed Fisher background" reframe (frozen
scorer = fixed metric `G`; per-stage norm = descent geometry in `G`; Sinkhorn/Muon = vielbein gauge
flattening `G`) — a conceptual lens, no new mechanism. Reheat-as-stage-noise-covariance-shift calibrated
to measured ringing + margin-engage spike — a measured controller, MED-LOW EV, evaluate-don't-mandate.

**EV ranking (Δd_seg-descent or conditioning-health per implementation cost):**
1. Stiefel-W(no-WD) + code-spectral-entropy — root cure, near-zero cost — **HIGHEST**.
2. Per-stage moment-reset + Muon-priming at transitions — cheap; the binding "re-treat" non-negotiable — **HIGH**.
3. Fisher-Rao EMA re-orthonormalization — cheap correctness fix for the deployed shadow — **MED-HIGH**.
4. SinkGD-on-FiLM-W (transient conditioning) — theory-motivated, second-order to the cure — **MED**.
5. Closed-loop PR-triggered reheat — control-theory nicety; only if fixed SGDR insufficient — **MED-LOW**.
6. Full per-group fractal (palette/pose-film custom geometry) — small params, pose solved — **LOW**.

**Over-engineering risk (kitchen-sink trap; CLAUDE.md PR105=1776 LOC LOST to 241 LOC):** the full fractal
is 6 groups × 4 stages ≈ 24 knobs — jointly tuning them is its own optimization problem risking
(a) un-attributable results, (b) chaos (Muon+pose divergence anchor), (c) means-elegance over d_seg-ends.
**Mitigation:** ship §4 minimal first; gate each incremental lever behind its own A/B that must show a
measured PR-or-d_seg win to stay; the DSL makes every lever independently toggleable so we never tune 24
knobs at once; the §6 firewall stops us polishing conditioning-health if it doesn't move the END.

## Observability surface
- Per-step telemetry: `PR(M)`, `PR(cov(code))`, per-group grad-norm, per-group update-norm (post-geometry),
  Stiefel residual `‖WᵀW − I‖_F`, EMA-vs-live divergence, advisory d_seg(n) — all JSONL, diff-able per arm.
- Cite-chain: every smoke row tagged (git sha, seed, n, stage, arm) `[macOS-MLX research-signal]`,
  `score_claim=false`, `promotable=false`.
- Counterfactual: each DSL lever is a single toggle → A/B is one YAML diff.
