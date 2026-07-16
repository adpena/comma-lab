# Optimizer-dynamics follow-up: weight-norm trajectories MEASURED, the INR radial-norm law DERIVED, norm↔rate coupling MEASURED, AutoClip reversal mechanism RESOLVED (2026-07-15)

**Operator directive:** "there is probably follow up research to do as well based on what that
optimizer research and the deep math and dynamics have revealed." Predecessor:
`.omx/research/adamc_muonc_optimizer_research_20260715.md`.

**STORES CONSULTED:** predecessor memo (P1–P3 predictions) · memory
`perparam_normalize_masks_all_norm_clipping_c0_confound_20260715` · maglaw law commit `9fefd157f7`
(anchor `autoclip_descent_speed_effect_n24_ab_measured_20260715`) ·
`src/tac/optimization/muon_finisher_mlx.py` + installed `mlx.optimizers.Muon.apply_single` ·
stage checkpoints on disk (mod32cap n600 20260706, l7 n200 20260628, maglaw n24 arms 20260715) ·
`tac.witness_dsl.adaptivization_tickets_20260715` · equation `adamc_wd_lr_equilibrium_v1`.

means != ends: everything below is MEANS (apparatus/law). **Pointer 0.19108 UNMOVED.**
All rows `[macOS-MLX research-signal]` / `[measured: existing checkpoints]`, NON-PROMOTABLE,
verdict_scope: FORMULATION (v9/witness lineages named per row). $0 — no run was launched;
every number below is computed from bytes that already existed on disk.

Artifacts (scripts + JSON):
`experiments/results/optdyn_followup_20260715/{t1_norm_trajectory.py, t1_norm_trajectories.json,
t1b_effective_step.py, t3_norm_rate_cost.py, t3_norm_rate_cost.json, t4_autoclip_mechanism.py,
t4_arm_telemetry.json}`.

---

## T1 — Weight-norm trajectory decomposition (MEASURED; the $0 P3 probe, and it FALSIFIES P3's sign)

Muon group per `muon_finisher_param_filter`: {in_proj, film, hidden.0–3}.weight; AdamW = rest.
Steps/ep: n600@accum8 = 75; n24@accum8 = 3; n200@accum8 = 25.

### A. mod32cap n600 (20260706T115554Z; Muon starts ep726, FLAT muon_lr=2e-3, trunk lr=1e-3, wd=1e-4, grad-normalize=none) — live weights

| tensor (MUON grp) | ‖W‖ ep299 | ep726 (MuonStart) | ep1000 | finisher Δ% | u=γ√(n/m)√min | rel step ep726 | ep1000 | hidden-LR × |
|---|---|---|---|---|---|---|---|---|
| in_proj.weight | 18.61 | 18.72 | 18.56 | −0.9% | 0.0196 | 0.105% | 0.106% | ×1.009 |
| film.weight | 27.17 | 28.58 | 20.38 | **−28.7%** | 0.0554 | 0.194% | 0.272% | **×1.402** |
| hidden.0.weight | 21.49 | 21.80 | 19.93 | −8.6% | 0.0196 | 0.090% | 0.098% | ×1.094 |
| hidden.1.weight | 21.49 | 21.66 | 19.34 | −10.7% | 0.0196 | 0.090% | 0.101% | ×1.120 |
| hidden.2.weight | 20.24 | 20.71 | 17.55 | −15.3% | 0.0196 | 0.095% | 0.112% | ×1.180 |
| hidden.3.weight | 19.56 | 20.11 | 16.45 | **−18.2%** | 0.0196 | 0.097% | 0.119% | ×1.222 |

Group aggregates: MUON 52.92→54.27→45.93 (pre-Muon +2.5%, finisher **−15.4%**); AdamW-rest
(incl. code) 111.7→112.6→116.0 (drifts UP slowly). Head out_sdf.weight 17.32→16.70→16.64 (flat).

**P3 FALSIFIED (sign inverted):** the predecessor predicted undamped random-walk ‖W‖ GROWTH in the
finisher (inert coupled wd → no damping). Measured: Muon-group norms SHRANK, 0.9–28.7% per tensor.
Decomposition (film.weight, 20,550 finisher steps): Δ‖W‖² = −401.4; diffusion Σ‖u‖² = +63.1
(u = 0.0554/step, weight-independent by NS construction); ⇒ radial work Σ⟨u,W⟩ = +232.3 ⇒ mean
inward cosine ⟨û,Ŵ⟩ ≈ **+0.0085**. A 0.85% persistent radial alignment beats the diffusion 7:1.
wd channel: exp(−λΣγ) = 0.9959 (−0.41%) — 70× too small for the observed −28.7%. **The drift is
GRADIENT-driven (landscape radial force), not wd, not a random walk.**

**Hidden-LR verdict:** Muon's NS update norm is weight-independent, so effective relative step
η_rel = γ·√max(1,n/m)·√min(n,m)/‖W(t)‖. With flat γ and shrinking ‖W‖ we ran a **hidden per-layer
LR INCREASE** during the finisher: ×1.40 on film, ×1.09–1.22 on hidden — self-accelerating
(shrink → larger relative step → more shrink per step while the radial force persists). This
COMPOUNDS with any `--muon-lr-final-frac` anneal (0.1 anneal × 1.4 drift = net ×0.14, not ×0.10).
Also NOTE the static disparity: the √max(1,n/m) RMS-matching rule gives film.weight 2.0–2.8× the
relative step of the square hidden layers — a per-layer LR ratio nobody chose.

### B. l7 lineage n200 (20260628; AdamW-only, hosc, CE ep1-299 → Tau ep300-899 → L7 ep900-1500) — the logit-scale drift

| tensor | ep299 | ep899 (Tau end) | ep1500 | total × |
|---|---|---|---|---|
| **out_sdf.weight (head)** | 16.24 | 90.30 | 98.58 | **×6.07** |
| out_sdf.bias | 2.83 | 10.34 | 10.90 | ×3.85 |
| hidden.3.weight | 14.76 | 18.57 | 19.31 | ×1.31 |
| film.weight | 28.37 | 33.77 | 34.71 | ×1.22 |
| code | 12.31 | 15.98 | 16.38 | ×1.33 |

Contrast mod32cap: head ×0.96 (flat) under its explicit `--softmax-temp-start 1.0 --temp-end 0.05`
schedule (measured softmax_temp 0.806@ep299 → 0.216@ep1000, ×3.74 scheduled sharpening).

**Measured law instance:** softmax CE has the exact gauge (τ, W_head, b_head) → (cτ, cW_head,
cb_head). The physical sharpness is κ = ‖W_head‖/τ. In the l7-lineage formulation the schedule
under-delivered sharpening and the optimizer filled it in via head-norm growth: **×6.07 ≈ +2.6
octaves of UNSCHEDULED temperature anneal** — more than a τ-ladder rung, delivered by norm drift.
In mod32cap the schedule delivered it and the head stayed flat. So yes — **past τ-anneal behavior
was partially norm-drift in disguise, and the split is formulation-dependent**: only κ is physical;
reading τ alone under- or over-states the anneal by the head-norm factor. Dual consequence: Adam's
per-param step is ~lr regardless of ‖W‖, so the head's relative step fell ×6 during the l7 Tau
stage — a hidden per-layer LR DECAY on the head, exactly during the stage that tunes sharpness.

### C. Defazio-equilibrium check from the Adam v-buffers (mod32cap, √Σv ≈ ‖g‖_RMS)

Measured ‖g‖/‖W‖ at CE end (ep299): code 0.009, hidden weights 0.017–0.030, film 0.018, head
0.008, biases 0.02–0.14; out_tex.weight 0.82 (sole outlier ABOVE). Defazio target √(2λ/γ) =
**0.447**. We sit **4–50× BELOW the wd-lr equilibrium**, and the chase time-constant 1/(2λγ) ≈
5e6 steps ≫ the 7.5e4-step run. **MEASURED confirmation of the predicted-null:** the wd×schedule
equilibrium is never approached on our runs; weight-norm stationarity (where it exists) is
LANDSCAPE-set, not wd-set. (Also confirms `--stage-transition-reset-moments` fired: v≡0 at ep726.)

## T2 — The derived law for OUR architecture (coord-INR, sin/hosc, FiLM, no norm layers, AdamW trunk + Muon finisher, EMA 0.997)

Per-tensor norm ODE (exact expansion of the update recursion, continuous limit):

    d‖W‖²/dt = −2γλ‖W‖²  −  2γ⟨∇_W L, W⟩  +  γ²·E‖u‖²
                 (wd)        (radial force R)     (diffusion)

Defazio/Chou's setting has R ≡ 0 (scale invariance from norm layers) → equilibrium between wd and
diffusion → ‖g‖/‖W‖ = √(2λ/γ). **Our witness has NO norm layers and non-homogeneous activations
(sin/hosc), so R ≢ 0 and everything changes:**

1. **Channel magnitudes (MEASURED, film.weight finisher):** radial −464 vs diffusion +63 vs wd −3.3
   (units of ‖W‖²): the radial gradient channel dominates 7:1:0.05. Stationarity condition is
   ⟨∇L,W⟩ = γE‖u‖²/2 − λ‖W‖² ≈ γE‖u‖²/2 — norms equilibrate where the landscape's radial slope
   matches half the diffusion, i.e. **the landscape, through γ, sets the norm — not wd**.

2. **The head is a GAUGE direction (exact):** logits ℓ = W_h·h + b with softmax at temperature τ
   are invariant under (τ,W_h,b) → (cτ,cW_h,cb). κ = ‖W_head‖/τ is the physical inverse
   temperature; the τ octave ladder controls the WITNESS only through κ. The CE radial force on
   the head is outward (sharpening lowers CE once margins are majority-correct) and is exactly the
   force that fills in whatever the τ schedule does not deliver (measured: ×6.07 l7 vs ×0.96
   mod32cap). This is the #464 Weyl/gauge zero-mode shape at the optimizer surface: a flat
   direction of the loss that the SCHEDULE must gauge-fix, or the optimizer random-walks/drifts it.

3. **Hidden weights are SPECTRAL coordinates, not nuisance scales:** for a row w of a sin/hosc
   layer, the feature sin(ω⟨w,x⟩) has spatial frequency ω‖w‖ along ŵ — ‖W‖ drift IS an NTK
   band-pass shift (#204/#309). The measured finisher shrink (film −28.7%) is a DOWNWARD spectral
   drift during polish (consistent with de-ringing/Gibbs suppression, the L85 flicker axis), and it
   doubles as the hidden LR increase above. Norm control on these tensors is a spectral lever
   first, a regularizer second.

4. **Muon-group effective dynamics:** u is weight-independent (NS: ‖NS(m)‖_F ≈ √min(n,m), lr
   scaled by √max(1,n/m)), so d‖W‖/dt = −γ_rad(t) + u²/(2‖W‖) with η_rel = u/‖W‖ a state variable.
   No wd term exists (MLX coupled-through-NS wd ≈ inert, predecessor §2). The system
   self-accelerates whenever γ_rad > 0.

**The right invariant to hold stationary** (the deliverable): NOT Defazio's ‖g‖/‖x‖ (measured
4–50× away, never approached). For a spectral-content-sensitive INR with a softmax head, the two
gauge-fixed quantities are:

- **κ = ‖W_head‖/τ** — schedule κ, not τ: either project ‖W_head‖ back to its stage-start value at
  checkpoint cadence, or fold the measured ‖W_head‖ drift into the τ-ladder rung accounting (a
  telemetry read + a bookkeeping change, byte-identical to training).
- **η_rel(t) = u/‖W(t)‖ per Muon tensor** — pin the per-tensor relative step (γ_muon ∝ ‖W(t)‖
  tracking, or stage-cadence norm re-projection). Per-param normalize is the AdamW-stage analog of
  exactly this pin — **which is WHY the incumbent is durable** (magnitude-stationary applied law;
  the predecessor's frame, now with the mechanism named for our architecture).

#318 DE-framework slot: the norm ODE joins the coupled system {τ ladder, β anneal, lr cosine} with
κ as a first integral to gauge-fix; the l7 head blow-up is what an un-gauge-fixed zero-mode does.

## T3 — Norm ↔ counted-bytes coupling (MEASURED; honest: real but second-order at our drifts)

Method (`t3_norm_rate_cost.py`, EMA-shadow tensors = the shipped artifact): (i) per-tensor-scale
int8 (the L29 grammar: scale = linf/127 per stage) → entropy + zlib; (ii) FIXED absolute step δ
frozen from the ep299 scale (constant functional fidelity) → entropy + zlib.

| run | stage | bits/w (i) scaled | bits/w (ii) fixed-δ | zlib (i) B | zlib (ii) B |
|---|---|---|---|---|---|
| mod32cap | 299 | 6.671 | 6.671 | 85,374 | 116,175 |
| mod32cap | 1000 | 6.739 | 6.552 | 86,619 | 114,636 |
| l7 | 299 | 7.054 | 7.054 | 76,192 | 106,016 |
| l7 | 1500 | 6.854 | **7.298** | 73,900 | **108,993** |

Cross-tensor correlation Δlog2(linf) vs Δbits@fixed-δ: **r = 0.52/0.54, slope 0.17/0.23
bits/octave** (naive theory 1.0 — the shortfall is distribution shape: linf grows via tails while
the bulk stays put; per-tensor-scale entropy actually FELL on l7 as kurtosis rose). Biggest single
mover: out_sdf.weight +2.47 octaves → +0.83 bits/w (480 params → ~50 B; small).

**Verdict:** at our measured drift magnitudes the norm→BYTES channel is real but second-order
(+3.5% bits/w on the l7 lineage at fixed fidelity; ~0 on mod32cap). The FIRST-ORDER cost is the
**fidelity channel**: the per-tensor-scale grammar absorbs norm growth into δ = linf/127, so the l7
head's ×5.5 linf growth coarsened its absolute quantization step ×5.5 — harmless for the head
(margins co-grew, gauge again) but REAL for frequency-carrying hidden tensors, where δ·‖x‖·ω is an
absolute PHASE error (hidden.3: +1.29 octaves → ×2.45 coarser phase resolution at fixed 8 bits =
quantization-induced spectral jitter, a d_seg risk not a rate line-item). **Norm control enters the
sealed config as a QUANT-FIDELITY argument (phase-error cap on spectral tensors), not as a
byte-count argument** — different from the paper's justification, per the charter's hope, but with
the honest measured magnitude: modest at today's drifts.

## T2′ — PRESCRIPTIVE (charter upgrade, operator 2026-07-15: "We can control and engineer the gradient and weights and classes and carriers and all however is optimal across the full pipeline")

The invariant is a DESIGN CHOICE we own, not a property to discover. Deriving the jointly-optimal
stationarity target for spectral content (d_seg) AND grid entropy (rate) together:

**Per-tensor design variables:** row-norm profile {‖w_i‖} (spectral band placement), grid step δ
(rate/fidelity lattice), relative step η_rel (descent dynamics). All three are ours.

**The joint derivation.** (a) Spectral: a sin/hosc row w contributes frequency ω‖w‖ along ŵ. The
scored signal passes through R (bicubic↑874 → uint8 → bilinear↓384), a low-pass with cutoff k_R;
frequencies above k_R are invisible-or-aliased (Gibbs — the measured flicker/ringing axis).
d_seg needs band coverage up to the separatrix's finest persistent scale k_need (lane dashes,
L65 crossover) — so the OPTIMAL band edge is ω‖w‖_max = min(k_need, k_R), and the optimal
row-norm DISTRIBUTION is the designed octave ladder across rows (the weight-space analog of the
τ ladder). (b) Rate: at grid step δ, bits/weight ≈ h(W/δ); phase fidelity bounds
δ ≤ ε_φ/(ω‖x‖_rms) (T3's measured fidelity channel). Bits = log₂(2‖w‖_∞ω‖x‖/ε_φ) — grows with
log norm. **So any norm above the R-cutoff band edge is STRICTLY DOMINATED: it pays bits AND
aliases through R.** The jointly-optimal target:

    ‖w‖* = min(k_need, k_R)/ω   (per row, ladder-profiled)
    δ*   = ε_φ/(ω·‖x‖_rms)      (per tensor, phase-budget law)
    κ*   = ‖W_head‖/τ           (head: scheduled, gauge-fixed — not spectral, temperature)

**Optimizer engineering candidates to HOLD it** (pick by derivation + staged n24 measurement;
NOT restricted to published normalized-transformer corrections):

1. **Row-norm projection at stage/ckpt cadence** onto the designed spectral profile — exact,
   deterministic, resumable, zero new hyper-parameters beyond the profile itself.
2. **η_rel pin for Muon** (γ_muon ∝ ‖W(t)‖) — holds dynamics stationary; does not pin the norm.
3. **Restoring decay** −λ(‖W‖−‖W‖*)·Ŵ — decay toward the DESIGN target (the honest "MuonC for
   us": a target-seeking force, trivially schedule-stable since the target is static). Plain
   shrink-to-zero wd is the wrong force everywhere we measured.
4. **Grid-native training** (#496 M+Adam / Pluralis int8-grid sparsity family): master weights
   live ON the shipping lattice δ* (fake-quant STE forward — `tac.quantization.FakeQuantSTE`
   already exists; EMA shadow on-grid) — norm drift beyond grid range becomes impossible BY
   CONSTRUCTION and export is the identity. Strongest form; see T3′.

## T3′ — PRESCRIPTIVE: the co-designed control (optimizer and codec share ONE lattice)

The measured coupling (T3: r≈0.52, slope 0.17–0.23 bits/octave; fidelity channel first-order)
justifies co-design at the FIDELITY surface, and the loss stack ALREADY carries a
`weight_entropy` term (measured live in the maglaw arms: 0.508–0.540) — the rate side of the
Lagrangian is already in training; what's missing is grid alignment. **Proposal
(`GridNativeWitness`, design-note scope):** per-tensor lattice δ_t from the phase-budget law
above; train with fake-quant STE on that exact lattice; EMA shadow accumulated on-grid;
`weight_entropy` evaluated on the ACTUAL shipped codes (int codes at δ_t) instead of a proxy;
byte-close export = identity (no post-training quantization gap — the train-big-compress-small
endgame in its strongest form). Deterministic + resumable by construction (grid states are exact
in npz). Duty-queue routed; NOT built here. Unlock: the weight-norm/grid telemetry stream +
one n24 A/B (grid-native vs post-hoc int8) measuring d_seg-at-equal-bytes through the real
byte-close.

## T5 — DESIGN NOTE (scope-bounded): classes as a TRAINING control dimension × v8 per-class carriers

Classes are already a carrier dimension (v8 #359/#380/#386 per-class edge carriers) and a loss
dimension (per-class λ organ, #433: prior-mean −18%, direction-neutral n=9). The upgrade: they
are also a GRADIENT dimension — per-class gradient treatment (per-class clip / normalize /
precondition gain g_c on the class-c component of the seg loss, extending the per-class-λ
machinery which already exposes the class split point).

**The joint allocation problem (sketch):** choose per-class training gains g_c AND per-class
carrier bytes b_c to minimize Σ_c w_c·d_seg_c(g_c, b_c) + (25/37.5M)·Σ_c b_c, subject to the
training-compute and byte budgets. KKT: equalize ∂d_c/∂g_c across classes (training side — g_c
is byte-FREE, so its optimum is where per-class marginal d_seg per unit compute equalizes) and
∂d_c/∂b_c = −(25/37.5M)/w_c (carrier side — waterfill, `boundary_routing.py` primitives exist).
The COUPLING makes it one problem, not two: a class trained sharper needs fewer carrier bytes
(g_c and b_c are substitutes), so the allocator must iterate the two KKT systems jointly —
2-resource waterfilling over the 5 classes, with the measured per-class flip shares
(50% road / 19% lane / 13% undrivable) as the initial marginals. Buildable pieces (duty-queue,
NOT built): (i) per-class gradient-gain lever g_c (extends the per-class-λ organ; DSL Lever,
default-OFF); (ii) per-class marginal telemetry — `d_seg_by_class` ALREADY exists in verdict
rows, add ∂d_c/∂ep finite-difference reading (sensor READS the stream, no recompute);
(iii) the joint 2-resource KKT allocator consuming (ii) + the v8 carrier byte table.

## T4 — AutoClip ep25+ reversal: mechanism RESOLVED from existing telemetry; discriminators pre-registered

Existing armA/B telemetry (`t4_arm_telemetry.json`; 3 opt steps/ep) already adjudicates the three
pre-registration candidates:

- **(b) spike-guard interaction: FALSIFIED.** spike_skipped=0 on all 79/80 epochs both arms;
  accepted_frac ≡ 1.0.
- **(c) τ handoff at ~ep25: FALSIFIED.** softmax_temp ≡ 1.0 and hosc_beta ≡ 1.0 through ep80
  (3000-ep anneal barely moved in-window); no stage boundary near ep25.
- **(a) percentile-window nonstationarity: CONFIRMED — with the SIGN INVERTED from the draft
  hypothesis.** Raw gnorm DECAYED 217.9 (ep1) → ~3 (ep30+, plateau 2–4); it never rose. What
  actually happened: AutoClip p10/w1000 under a decaying gnorm is a **LAGGED NORM-TARGET, not an
  outlier clipper** — frac_clipped ≈ 1.0 (p10 threshold ⇒ ~90–100% of steps clip by construction),
  so the applied global step ≈ clip_t. Trajectory: warmup fallback 0.5 (ep1–4) → jumps to 8.99
  (ep5, p10 of a tiny history full of giant early gnorms) → decays with the gnorm to ~2.55 and
  then FLOORS there (w=1000 @ 3 steps/ep = **333 epochs of memory**; history_filled=237 < w at
  ep79 ⇒ the percentile never forgets the early epochs). Meanwhile armC pinned 0.5 and armA pinned
  per-tensor unit norms.

  So armB ran an applied step **5–18× larger than the stationary arms from ep5 onward** (2.55–8.99
  vs 0.5), which explains BOTH phases with one mechanism: the early win (ep25 d_seg 0.015902 vs A
  0.018045 — bigger steps descend faster far from the minimum) AND the reversal (ep50 0.018215,
  ep75 0.018644 while A descends 0.016966→0.015325 — the same large steps near the sharpening
  overfit minimum are super-critical → overshoot/EoS regime; the gnorm plateau at 2–4 is the
  step-size-limited noise floor, consistent with curvature-limited oscillation, sister of the
  curriculum-as-continuation bifurcation lens `curriculum_is_continuation_instabilities_are_bifurcations_20260714`).

**Pre-registered discriminating signatures for the ≥150-ep durability A/B** (fire only when the
GPU frees; admission per the governed launcher):

- **S1 (window):** arms w∈{100, 1000, 3000} at fixed p10. Prediction: clip_t floor ≈ trailing-decile
  of the last w steps ⇒ shorter w tracks the gnorm decay down ⇒ reversal DELAYED/ABSENT at w=100,
  EARLIER/STRONGER at w=3000. If reversal epoch does NOT move with w, the window-memory mechanism
  is falsified.
- **S2 (percentile):** p10→p2 at fixed w=1000 lowers the late floor toward armC behavior ⇒ reversal
  removed AND early win shrinks — the one-knob-two-phases signature. If early win survives p2 with
  no reversal, AutoClip gets a redemption path (tight-percentile variant).
- **S3 (within-run EoS):** post-reversal, armB verdict-to-verdict d_seg wobble variance > armA at
  matched epochs; gnorm plateau level tracks the clip_t floor. (Baseline from existing run:
  B 0.0159→0.0182→0.0186 non-monotone vs A monotone.)
- **S4 (causal rebase — cheapest decisive, ~$0):** RESUME the existing armB ep75 checkpoint
  (`levelset_n24_maglaw_armB_20260715/levelset_resume_state.npz`) with the clip law switched to
  fixed-0.5 (or per-param normalize), same seed/config. Prediction: d_seg resumes descending
  within ~10 ep ⇒ the reversal is ONGOING-step-size-driven (mechanism (a) causal). If it does NOT
  descend, the large-step phase already poisoned the state (off-manifold weights) ⇒ rewind-to-ep25
  is the fix and the mechanism is path-dependent, not instantaneous.

Ranking: (a-modified) is the surviving mechanism with 4 orthogonal falsifiers pre-registered;
(b),(c) are measured-dead. The AutoClip law's sealed constants (p=10, w=1000) are the
constants-are-poison shape — ticketed (see routing).

## Routing (duty-queue; config-orphan rule binds — nothing armed)

1. **Equation registered:** `inr_weight_norm_radial_ode_v1`
   (`src/tac/canonical_equations/inr_weight_norm_radial_ode_20260715.py`) — the T2 norm ODE + κ
   gauge + η_rel pin law, with three MEASURED anchors (Muon-shrink T1-A, head-gauge T1-B, norm↔rate
   T3) and the S1–S4 pre-registration carried in the T4 anchor.
2. **Tickets** (`tac.witness_dsl.adaptivization_tickets_20260715`):
   - `--muon-weight-decay` ticket UPDATED with the P3 measurement: growth prediction falsified
     (norms SHRANK, gradient-driven); decoupled-wd promotion is now the WRONG control (adds
     pressure in the measured drift direction); superseded by the η_rel pin ticket.
   - NEW `--muon-lr` ticket: relative-step pin law γ_muon(t) ∝ ‖W(t)‖/‖W(t₀)‖ per Muon tensor
     (equivalently stage-cadence norm re-projection); unlock = the norm-telemetry stream + a
     bounded n24 A/B.
   - NEW `--grad-autoclip-percentile/window` ticket: w=1000@3steps/ep = 333-ep memory (measured
     floor mechanism); law = window in EPOCH units w_steps = w_ep·steps_per_ep, percentile from
     the S2 sweep; unlock = the S1/S2 arms.
3. **Owed ONE logging change** (score-neutral observability → defaults ON per the default-off
   doctrine; also satisfies FORE/HCM per `hcm_causal_attribution_dig`): per-tensor ‖W‖ (live+EMA)
   emitted at verdict cadence as a `weight_norm_telemetry` row — the sensor stream the κ gauge-fix
   and η_rel pin READ (poison-taxonomy: sensors read trainer streams, no recompute). Not landed
   here (trainer is the hot #507 surface); named as the unlock on both new tickets.
4. **S4 rebase arm** is staged in this memo (resume-from-existing-checkpoint, n24, ~minutes of
   GPU) — the first thing to fire when the 507/r6 chain frees the GPU, BEFORE the full ≥150-ep A/B
   (it can kill or confirm mechanism (a) for the price of ~10 epochs).

**verdict_scope: FORMULATION (mod32cap + l7 + v9_cgauge-maglaw lineages; n24/n200/n600 MLX
advisory). No score claim. Pointer 0.19108 UNMOVED.**
