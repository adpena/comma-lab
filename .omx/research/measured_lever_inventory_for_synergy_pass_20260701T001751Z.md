# MEASURED lever inventory for the synergy-optimal-run synthesis (task #201)

**UTC** 2026-07-01T00:17:51Z · **git** `8ac690876` · **authority** `[$0 CPU research-consolidation / advisory]` ·
**score_claim** false · **promotable** false · **NO GPU, NO launch.** **Pointer UNMOVED: contest-CPU 0.19109982**
(`.omx/state/canonical_frontier_pointer.json`). This is a MEANS (the measured map, comparable in ΔS units) to
feed #201's synthesis; it does not move the pointer.

## Purpose + method
Operator task #201 needs a comparable inventory of every level-set task-space-witness lever with its MEASURED or
DERIVED EV. **Allergic to toys:** every EV below is MEASURED (cite artifact) or DERIVED (exact-score identity),
never a vibe. Every EV converted to ΔS via the exact-score identity so levers are comparable:

> **S = 100·d_seg + √(10·d_pose) + 25·bytes / 37,545,489**
> ⇒ d_seg lever: **ΔS = 100·Δd_seg** · rate lever: **ΔS = Δbytes × 6.659e-7** (≈ **6.66e-4 per KB**) ·
> pose lever: **ΔS = √(10·d_pose_new) − √(10·d_pose_old)**.

### CALIBRATION LEGEND (binding — read the tag on every EV)
- **EXACT** = `upstream/evaluate.py` n600 byte-closed (the ONLY score). **CPU-adv** = frozen CPU-torch SegNet argmax
  on cached `lstars` (advisory). **MLX-rs** = `[macOS-MLX research-signal]` (advisory). **DERIVED** = math bound.
- **direct** = witness logits→argmax vs L* (symbolic). **through-R** = realized through the contest R operator
  (bicubic↑874→uint8→bilinear↓512×384) + frozen scorer. **pre-R** = label-space probe, no R (a LOWER bound).
  THE CRITICAL GAP: **direct ≪ through-R** (0.0022 direct → 0.0064 realized on the exact-L* store).
- **n** = pairs scored. Per FEED-kn, n changes the OUTCOME (driving clip), not just the CI.
- **maturity** ∈ {**n600-validated** / **n96-needs-remeasure** (a toy-scale advisory number — NEEDS-N600) /
  **needs-through-R** (a direct number, R-survival unmeasured) / **derived** / **untested** (predicted only) /
  **deployed** (contest-fact)}.

### The two baselines every d_seg ΔS is measured against
- **lever-B n600 plateau** d_seg **0.008257** (100·d_seg = 0.826) — the isotropic-Fourier witness floor.
- **level-set realized through-R** (n96/n200): CE 0.005443 → Tau 0.004563 → l7 0.004287 → **Muon BEST 0.003718**
  (θ* ep1000). This IS the current best witness realized d_seg. Sub-0.15 need: realized d_seg ≤ **~1.2e-3** at the
  v2 byte budget (D6 SDF hosc chart 0.00124 through-R is the only chart that has cleared it, n96).

---

## §A. BASIS / REPRESENTATION / ACTIVATION levers (the chart — PRIOR to capacity)

| Lever | What it does | MEASURED Δd_seg (artifact) | ⇒ ΔS | Δrate | train-time | synergy / antagonism | maturity |
|---|---|---|---|---|---|---|---|
| **D1 directional/curvelet (all-class oriented Fourier)** `--self-orient` | orient PE across the ALL-class boundary tangent (curvelet cartoon-optimal) | control 0.006539 → **0.003416 (−48%)** n96; n600 0.007828→0.005697 (−31%) [MLX-rs, direct] (`witness_capstone_deepmath_levers`) | **−0.31 n96 / −0.21 n600** | ~0 byte | ~0 (front-end init) | **THE #1 lever**; capacity pays ONLY after it (D2); lane-only variant only −8%; ⚠ circular (built from `gt.lstars`) → self-orient fixed-point is the byte-closeable form, realized −48% UNVERIFIED | n96-needs-remeasure + needs-through-R |
| **D6 SDF level-set + hosc chart** `--activation hosc` | signed-distance level-set carrier, the witness chart itself | **0.00124 @ep950 [through-R surrogate, n96]** (BEST witness chart on record); r_added ~9.5e-5 R-survival transfers (canon D6/veh-C1) | baseline-setting **~0.12** | the vehicle | full run | the chart D1/D2/loss levers ride on; single-SDF **beats MSDF** (D-C2 falsified 3.6–76× worse) | needs-N600 (n96 through-R) |
| **D5 longer curriculum (900ep)** | just train longer | **0.002176 [MLX-rs direct, still descending]** — best DIRECT d_seg on record (canon D5) | −0.22 vs lever-B | 0 | cheapest real floor-mover (time only) | orthogonal to all; the "still descending" caveat = under-trained arms mislead | n96-needs-remeasure + needs-through-R |
| **D7 step_basis / hosc-β sharpen** `--hosc-beta(-end)` | learnable slope gₖ → β→∞ step-native (L∞-optimal, no Gibbs) | FINER −18.7% n100 → **−4.5% n600 (capacity decay)** [MLX-rs]; hosc β-fixed FAILS standalone (opt saturation) (canon D7) | −0.04..−0.19 (decays w/ n) | ~0 | speed/bandwidth knob at capacity limit, NOT a floor lever; suppresses ringing/Gibbs flips (birth-death) | n96-needs-remeasure; standalone-fragile |
| **D16 wide-SDF ramp σ=1.0** | 1-Lipschitz oriented SDF ramp = area-coverage anti-alias | R0 flat 0.0273/0.0242 → **R1 ramp 0.0230/0.0185 = −16%/−24% [through-R, n96, deterministic substrate]** (canon D16/veh-C3) | **−0.43 / −0.57** (on the deterministic arm) | **0 byte** | 0 | it is boundary PLACEMENT not texture (texture HURTS full d_seg); the FREE R-survival cure | needs-N600 (deterministic arm, not trained witness) |
| **D13 sub-pixel boundary placement (#149)** | oriented 1-Lipschitz ramp cures sub-pixel color-mixing in eval downsample | 12× collapse advisory (the ~24% boundary-band flip = the (C) wall) (canon D13/veh-C3/C4) | the binding R-survival cure (mag unbuilt) | 0 | build gap | the #1 R-survival lever; NOT yet in `train_*_through_R_mlx.py` | needs-through-R build |
| **D15 NCA continuous-texture witness (#146)** | growing-CA texture carrier | realized 0.00337 (1.31× frontier), band-flip 0.079 [through-R, n96] — **dominated by SDF 0.00124** (canon D15) | ~+0.12 vs D6 | ~same | AMBER; dominated → parallel arm only | needs-N600; dominated |

---

## §A2. CURVELET SCALING CURRICULUM / coarse-to-fine multiscale frequency for the boundary (FIRST-CLASS)

**The thesis (DERIVED, exact):** the contest d_seg residual is the codim-1 partition separatrix — a piecewise
**C² curve** (a cartoon edge). **Candès–Donoho curvelets are provably N⁻²-optimal** for functions that are C²
away from C² curve singularities (`m`-term approx error `O(m⁻²(log m)³)` vs wavelets `O(m⁻¹)` vs Fourier
`O(m^{−1/2})`). So our separatrix is *exactly* the object curvelets/shearlets were built for → the right
representation is **scale-by-scale directional (parabolic-scaling) atoms**, not isotropic Fourier. This is the
multiscale generalization of D1 (single-scale directional = −48%); the OPEN-headroom #3 climb toward ~0.001.

### A2.1 OUR MEASURED signal (the anchor — allergic to toys, cite artifact)
| finding | MEASURED | calibration | artifact |
|---|---|---|---|
| **curvelet vs isotropic front-end, EQUAL 40-col budget** | argmax-disagree **0.555 vs 0.692 = 0.80× (20% better basis at equal bytes, GT-FREE)** | CPU-adv, n6, linear-fit proxy (RATIO is the signal, absolute high) | `levelset_curvelet_witness_feasibility_20260627` |
| **SDF level-set through-R vs spectral/sine** | post-R disagree **1.27e-5 vs 7.46e-3 = ~587× lower**; 0 off-boundary R-flips | CPU-adv through-R proxy, n6 | same |
| **generic curvelet/shearlet bank is byte-FREE (rule 118)** | J scales × L_j parabolic orientations regenerated from ~5 scalars at decode → 0 counted bytes, NO GT leak | compliance-fact | same; canon R19 |
| **D1 single-scale directional (the −48% lever)** | all-class oriented −48% n96 / −31% n600 | MLX-rs direct | `witness_capstone_deepmath_levers` (see §A) |
| **anti-alias ceiling (the scale cap)** | fc∈{256,512,1024} ALIAS at the render grid → near-DC garbage; only {32,64} clean (SegNet stem-Nyquist 64) | $0 numpy MEASURED | `yousfi_levers_optimal_form_review` |
| **bandwidth anneal 16→32→64 = shape-break in warm-start** | changing `--max-bank-freq`/`--n-dir-freqs` breaks resume `in_feat` → a FROM-SCRATCH curriculum knob, not a warm-start A/B | code-fact | `thetastar_per_lever_AB` §3/§5 |

### A2.2 EXTERNAL literature (mechanism + EV hypothesis; DERIVED-not-measured; arXiv ids best-recall — VERIFY before citing in a paper)
| source | id / repo | transferable mechanism | EV hypothesis (lowest-S / shortest-train) |
|---|---|---|---|
| **Candès–Donoho curvelets** | CPAM 2004 (journal, not arXiv); Fast Discrete Curvelet Transform, Candès-Demanet-Donoho-Ying, MMS 2006; **CurveLab** curvelet.org | parabolic scaling (width ≈ length²), L_j orientations doubling every other scale → N⁻²-optimal for C² curve singularities | **lowest-S:** the separatrix is exactly a C² curve → fewest atoms/boundary = lower rate at equal d_seg AND lower d_seg at equal atoms; the theoretical basis for our 0.80×/−48% |
| **Shearlets (Labate-Kutyniok)** | Guo-Labate SIAM JMA 2007; Kutyniok-Labate book 2012; **ShearLab** shearlab.math.lmu.de | shear (not rotation) preserves the integer lattice → same N⁻² optimality, cleaner discrete/GPU implementation than curvelets | **lowest-S:** a GPU-clean, byte-free directional bank (rule-118) that is easier to regenerate deterministically at decode than curvelets |
| **BACON — band-limited coordinate nets** | arXiv:2112.04645; repo computational-imaging/bacon | an MFN whose output has an ANALYTICALLY BOUNDED bandwidth per layer → multiscale by construction, queryable at any scale, provably no aliasing above the cap | **shortest-train + R-survival:** cap the front-end bandwidth per stage → provably no aliasing (directly cures our measured fc>64→near-DC garbage) and gives the coarse→fine ladder for free |
| **Multiplicative Filter Networks (MFN)** | Fathony-Sahu-Willmott-Kolter, ICLR 2021 (OpenReview; no canonical arXiv) | output = product of Gabor/Fourier filters → sparse multiscale; per-layer frequency init controls the spectrum | **shortest-train:** principled per-layer frequency init for our hosc/WIRE front-end (a Gabor sibling); avoids random-init spectral fight |
| **Nerfies coarse-to-fine PE annealing** | arXiv:2011.12948; repo google/nerfies | windowed positional encoding: ramp in high-freq PE bands over training (α(t) from 0→L) | **shortest-train:** formalizes our structured-init-seed→NTK-jump; ramp fine bands only after the coarse partition converges → matches the NTK eigenspectrum, avoids critical-slowing |
| **BARF — bundle-adjusting NeRF (Park-et-al. coarse-to-fine)** | arXiv:2104.06405; repo chenhsuanlin/bundle-adjusting-NeRF | proves high-freq PE ON early = bad local minima; a coarse-to-fine PE schedule smooths the loss landscape for the geometry/registration | **shortest-train:** the schedule discipline for our warp+separatrix co-fit; smooths the annulus valley → fewer Muon tail epochs |
| **mip-NeRF integrated PE (anti-aliasing)** | arXiv:2103.13415; repo google/mipnerf | integrate PE over a pixel footprint (a cone) → scale-aware, anti-aliased, one net serves all scales | **R-survival:** render scale-aware so the field survives the eval R downsample (874→512) — the principled sibling of D16 wide-SDF ramp = area-coverage AA |
| **Fourier Features / spectral bias** | Tancik arXiv:2006.10739 (repo tancik/fourier-feature-networks); Rahaman ICML'19 arXiv:1806.08734; SIREN arXiv:2006.09661; WIRE arXiv:2301.05187 (repo vishwa91/wire) | NNs learn low→high freq (NTK eigenspectrum); Fourier/Gabor/wavelet features + a frequency curriculum overcome it | **shortest-train:** the WHY a frequency curriculum converges faster than fighting spectral bias at fixed bandwidth; WIRE (wavelet) is the continuous cousin of our hosc chart |

### A2.3 The curvelet-scaling-curriculum LEVER (composite EV)
| Lever | What it does | EV | ⇒ ΔS | Δrate | train-time | synergy / antagonism | maturity |
|---|---|---|---|---|---|---|---|
| **Curvelet coarse→fine scaling curriculum** `--self-orient --max-bank-freq (staged 16→32→64) --n-dir-freqs` | climb parabolic-scaling directional bands on the boundary annulus, coarse→fine, capped below stem-Nyquist 64 | anchored MEASURED: single-scale −48% (D1) + curvelet 0.80×-at-equal-budget + 587× SDF R-survival; **HYPOTHESIS: multiscale climb toward the ~0.001 need at LOWER byte** | **−0.21..−0.31 measured (single-scale) → toward −0.4+ (multiscale, HYPOTHESIS)** | ~0 byte (rule-118 free bank) | **shortest-train: the coarse→fine ramp AVOIDS the spectral-bias fight + shortens the Muon critical-slowing tail** | **generalizes D1 (#1 lever)**; requires FROM-SCRATCH (shape-break in warm-start, θ*§5); cap below fc=64 (anti-alias, MEASURED); COMPOSES with D2 capacity + D6 SDF chart + structured-init seed | n6-toy MEASURED (basis ratio + R-survival) → **NEEDS-N600-through-R**; the multiscale climb is a HYPOTHESIS grounded in the single-scale measured lever |

---

## §B. LOSS / SALIENCY levers (loss-shaper on the fixed chart)

| Lever | What it does | MEASURED Δd_seg (artifact) | ⇒ ΔS | Δrate | train-time | synergy / antagonism | maturity |
|---|---|---|---|---|---|---|---|
| **D3 margin-hinge (lensA)** grad 1.0 on flip set | hinge `relu(target−margin)` = grad 1.0 on confident flips vs soft-cosine 1.9e-22 | realized **−16..−36% vs CE** [CPU-adv] (canon D3; `lever2_softcosine_vs_ce_flipfix`) | −0.08..−0.18 (on 0.005 witness) | 0 | saturates the loss-reweight axis; the witness hard-pixel router; largely CAPTURED by the l7/margin curriculum stage | n96-needs-remeasure |
| **A1 margin-saliency all-class (LEVER-4)** `--margin-saliency-weight` | `sal=exp(−margin/τ)` defends 100% of the flip band (all inter-class edges) | **PREDICTED −0.0003..−0.0008** [through-R n200, θ* A/B, UNTESTED] (`thetastar_per_lever_AB`) | −0.03..−0.08 | 0 | rank-2 in θ*; PRIMED Lane converts, STUCK Road may not; **sub-additive with A2** (both hit Lane) | untested (ready-to-fire) |
| **A2 lane-thin dropped-dash (LEVER-B)** `--lane-thin-weight` | hinge weighted by thin-lane density map (nonzero on <~2px GT dashes) | **PREDICTED −0.0004..−0.0010** [through-R n200, θ* A/B, UNTESTED]; targets the MEASURED dominant residual (dashes <5px = 93% missed, birth-death) | **−0.04..−0.10 (rank-1 θ*)** | 0 | #1 predicted per GPU-h; sub-additive w/ A1; the most-targeted at the binding Road↔Lane residual | untested (ready-to-fire) |
| **D4 KD soft-logit aux (kd_w=0.3 T=2.0)** | CE-anchor soft-logit distillation | c1 −1.0%; **long900 −10.2% (did not saturate)** [MLX-rs] (canon D4) | −0.01..−0.04 | 0 | genuinely distinct from margin-hinge (NOT redundant); **pure-KD DIVERGES** → aux only | n96-needs-remeasure |
| **D8 / A7 UNIWARD texture down-weight (Fridrich β=4)** `--margin-saliency-uniward` | `sal/=(1+β·tex)` → concentrate on smooth survivable boundary (square-root law) | **PREDICTED −0.0001..−0.0003 marginal over A1** [θ* A/B wave-2, UNTESTED] (canon D8; `thetastar_per_lever_AB` A7) | −0.01..−0.03 | 0 | REQUIRES A1>0 (modifies its saliency); LATE-STAGE (l7/Muon) lever | untested (wave-2) |
| **Lever-5 margin-weight (in-curriculum)** `margin_weight_tau` | `exp(−margin/τ)` per-pixel boundary weight on the surrogate (reuses forwarded seg_out) | BUILT+TESTED (monotone-in-margin); ΔS pending A/B (`all_layer2_levers_implemented`) | (= D3/A1 family) | 0 | reuses seg_out (no extra scorer pass); the torch_vehicle sibling of A1 | untested (built) |
| **A6 lane-edge class-1-only (LEVER-3)** `--lane-edge-class 1` | class-1-only predecessor of A1 | **PREDICTED −0.0001..−0.0004** [θ* A/B, UNTESTED] — dominated by all-class A1 | −0.01..−0.04 | 0 | ABLATION only (confirms all-class A1 > class-1); **assumption-challenge: up-weighting lane may trade the 50% Road majority** (kill if TOTAL d_seg rises) | untested (ablation) |
| **A4 hardness-weighted code-fit (LEVER-5c)** `--hardness-oversample --hardness-weighted` | waterfill extra per-epoch pair-steps toward hard pairs (realized source) | **PREDICTED −0.0001..−0.0004** [θ* A/B, own uniform control] — GT-margin per-pair spread only 1.31× → modest | −0.01..−0.04 | +steps | needs its OWN matched (uniform-extras) control, not the shared control | untested |

---

## §C. CAPACITY / CONDITIONING levers (pay ONLY after basis-match)

| Lever | What it does | MEASURED Δd_seg (artifact) | ⇒ ΔS | Δrate | train-time | synergy / antagonism | maturity |
|---|---|---|---|---|---|---|---|
| **D2 KKT capacity-routing (waterfill on margin-saliency)** `BoundaryFiLM` | reverse-waterfill INR capacity onto the margin-saliency (boundary) | **basis+cap n600 best 0.002447 (−70% vs lever-B)** [MLX-rs direct]; capacity ALONE on isotropic **HURTS +6%** (canon D2) | **−0.58** (basis+cap combined) | +49–63KB (50→113KB) = +0.033..+0.042 rate | **STRICT: basis (D1) BEFORE capacity** — the #1 ordering constraint; capacity alone is net-negative | n96-needs-remeasure + needs-through-R |
| **D9 chroma (SegNet RGB-slack argmax lever)** `--chroma` | route capacity into chroma channels that flip the argmax at the annulus | **OPEN — unmeasured on realized axis**; SegNet argmax reads RGB ⇒ chroma carries argmax signal; frontier decode-side perturb HURT (= trained optimum) (canon D9) | unmeasured (predicted ↓, FREE bytes) | ~0 (free) | likely COMPOUNDS with lane-edge (chroma makes lanes separable); **every pre-chroma verdict PROVISIONAL** — baked into BASELINE | untested (baked-in, verdict-blocking) |
| **A5 DM1 (Stiefel-W + code-spectral-entropy)** `--film-stiefel --code-spectral-entropy-weight` | per-step Stiefel orthonormalize FiLM.W + spread code spectrum → PR(M) 1.19→4.57 at 0 bytes | **EXACT $0: PR collapses 2.6× WHILE d_seg IMPROVES 1.9×** → DEMOTED 2nd-order (per-pair FiLM can't localize the moving annulus); θ* predicts −0.0000..−0.0003 (canon G4; veh-G4) | 0..−0.03 | **0 byte** | primary value = **n600 amortization** (measured here only as side-signal); compose ONLY if FiLM collapse becomes binding | untested; demoted |
| **F1 vanilla FiLM rank-1.2 collapse** | multiplicative FiLM resonance | **PR(M)=1.19 = the MEASURED (2×) d_seg-plateau cause** [advisory M2] (canon F1) | (a TRAP, not a lever) | — | **NEVER vanilla FiLM** — the A5 Stiefel cure exists for exactly this | measured antagonism |

---

## §D. CURRICULUM STAGES (the schedule — measured realized-through-R drops)

Per-stage realized d_seg through-R (n96/n200), openpilot-seeded lineage (`witness_per_stage_attribution`,
`witness_curriculum_stage_epoch_ledger`). This is the **most trustworthy d_seg data** (realized, not direct).

| Stage | What it does | MEASURED Δd_seg (realized through-R) | ⇒ ΔS | #ep | synergy / antagonism | maturity |
|---|---|---|---|---|---|---|
| **S0 structured-init seed + lane-prior φ1** `--structured-init --lane-prior-phi1` | seed static-core SDFs + openpilot deg-3 centerline (self-detect roles) | seeds Road↔Lane separatrix, **residual 1.9e-5** [CPU-adv static-GT] (canon D10/C14) | seeds low-freq FREE | 0 (init) | **FREE 0 bytes** (rule-118); NTK: seed low-freq → jump to high-freq annulus; ⚠ CLOBBERED by `--resume-from` (from-scratch ONLY) | deployed (from-scratch) |
| **S1 CE** | confidence-calibration | 0.01045 → 0.005443 (end) [n600 MLX-port / n96 through-R] | −0.51 (from lever-B start) | ~300 | forms the static core (Road/Undriv/MyCar ~solved); Lane 47.2% mislabeled | n600-port + through-R |
| **S2 tau_softplus (τ=0.3)** [REHEAT] | soft-plus seg surrogate at reachability floor Δ_min≈0.3 | CE 0.005443 → **Tau 0.004563 = −0.000879 (THE primary single drop)** [through-R n96] | **−0.088** | ~300 | the biggest single-stage realized drop; τ=0.3 (SEG) ≠ render softmax-temp (§F) | through-R n96 |
| **S3 l7 + margin engage** [REHEAT] | 5× margin-weight allocation on margin<1.0 | Tau 0.004563 → **l7 0.004287 = −0.000276** [through-R n96]; knee fast ~ep700 | −0.028 | ~126 | short knee — over-long L7 wasted; STAGE-conditioned (from-scratch starves interior, finetune re-allocates) | through-R n96 |
| **S4 Muon finisher** [REHEAT] | spectral conditioner of the ill-conditioned annulus valley | l7 0.004287 → **Muon 0.003718 = −0.000569 (still descending, decelerating)** [through-R n96/n200] | −0.057 (so far) | 250+ | **THE finisher, LAST**; AdamW can't fix off-diagonal κ~19 Hessian; critical-slowing tail (root-tracking anneal = wall-clock cure) | through-R n200 |
| **SKIP smooth stage** | (PR95 has 1500 ep) | **RAISES d_seg +6.8%** [MLX-port] → DROP (canon C2) | +0.03 if kept (a NEGATIVE) | — | structural ANTAGONISM — do not include | measured antagonism |
| **SKIP QAT / C1a / λ / σ** | PR95 rate machinery (~14,500 ep) | rate-tuning, not d_seg | — | — | witness rate = byte-close of a tiny payload → these are STRUCTURALLY skipped | measured (out-of-scope) |
| **REHEAT at every transition** `--stage-transition-rewarmup-*` | LR floor 0.1×/8ep + reset-moments | PARTIAL restart stable (n_skips=0); **FULL 1.0× restart re-destabilizes** [MLX] (canon C9) | enables the stages (no direct ΔS) | ~8/stage | mandatory per "different stages need different treatment"; margin-engage spike-skip cured by re-treating spike-guard | measured |

**Whole-curriculum realized d_seg drop (CE→Muon): 0.005443 → 0.003718 = −0.001725 ⇒ ΔS −0.17** [through-R n96/n200].

---

## §E. OPTIMIZER levers

| Lever | What it does | MEASURED EV (artifact) | ⇒ ΔS | synergy / antagonism | maturity |
|---|---|---|---|---|---|
| **Muon (final stage)** `--optimizer`/curriculum | Newton-Schulz orthogonalized momentum SGD | **MUON_BITES_FROM_STAGE4: −32% d_seg MORE than AdamW (gap −0.000340, widens monotone)** [CPU-adv, 6.8× discrimination band] (canon C4) | −0.034 vs AdamW at S4 | AdamW grad-norm collapses on κ~19 Hessian → jump-to-Muon-early viable; **the finisher, not the opener** | CPU-adv |
| **muon-lr = 2e-3** `--muon-lr` | witness flat-finisher LR | band 1e-3..2e-3, ceiling 5e-3; **NOT 0.03 (6× too hot, nanogpt convention)** (canon C5) | (tuning, not a delta) | 0.03 = wrong regime; 2e-4 = PR95's 229K model (wrong); 3e-3 = A/B contrast only | derived band |
| **muon-lr-floor-fix** | Muon needs its OWN floor ratio | else never anneals to fine-polish (canon C6) | enables the Muon tail | provenance-fixed | CPU-adv |
| **MD-Decoupling** `--optimizer md` | momentum-decoupled, stable transitions | stability real (gnorm ≤376 vs AdamW 10869) **but UNDER-STEPS d_seg at scale** [CPU-smoke] (canon C15) | net ~0 (stability, not floor) | WIRED in through-R trainer, NOT the level-set trainer; **PARALLEL arm only** | CPU-smoke |

---

## §F. RATE / CODEC levers (the byte term)

| Lever | What it does | MEASURED Δbytes/EV (artifact) | ⇒ ΔS | maturity | synergy / antagonism |
|---|---|---|---|---|---|
| **L13 non-RGB witness format** | eikonal-SDF task-space container | **−59% rate (177,169→72,217 B), lossless-parity-proven** [GROUNDED 8-pair parity] (canon R4) | **−0.070** (frontier-sized context) | GROUNDED | the RATE HALF; **packages** d_seg cheaply, does NOT lower d_seg (R5: L13-the-vehicle S≈0.79) |
| **Deterministic backbone (13 keyframes)** | store the stable partition; reach via warp | rate **0.0060** (+pose 0.0066) vs store-everything **0.277** [GROUNDED through-R n96] (canon R8/R9) | **−0.271 rate** | GROUNDED (1 window) | rate WIN but **d_seg-DEAD** (R1 floor 0.0185 = ΔS +1.85) → ONLY as the hybrid substrate under a trained residual (R10/W11) |
| **Score-aware QAT (Lever-4)** `--score-aware-qat` | sensitivity-weighted INT8 grid (reverse waterfill) | **−3263 B (−4.4% decoder blob) at equal d_seg** [CPU-adv, codec-axis validated] (`all_layer2_levers` MED-2) | **−0.0022** | CPU-adv (codec half validated; net-score needs training A/B) | protects high-‖∂S/∂w‖ tensors, coarsens the rest; the bit-allocator hook |
| **Rate surrogate (Lever-1)** `--rate-lambda-w/lat` | order-1 conditional weight-entropy + latent-delta, codec-scan-order | **Spearman 0.90 / Pearson 0.999 vs real brotli bytes** (codec-scan-order) [CPU-adv] (`all_layer2_levers` MED-1) | training-proxy (enables rate ↓ during train) | CPU-adv | tracks DEPLOY bytes at train time; per-batch, default-OFF byte-identical |
| **low-rank pose codec rank-4/511 (#140)** `PFL2` | low-rank the stored pose section | **2,563 B, MSE 2.7e-5, −0.0004 rate** — Pareto-dominant [CPU-adv]; rank-2/254 net-NEGATIVE (canon P4) | **−0.0004** | CPU-adv | opt-in on the pose sidecar; the rank-2 "2.7×" claim is SUPERSEDED |
| **WRQ score-aware weight requant on C7** | per-tensor requant of the decoder | **UNGROUNDED magnitude; highest rate CEILING** (decoder ≈91% of an NN archive) (canon R15) | high ceiling, unmeasured | untested | needs its own exact sweep once C7 has descended |
| **Finishing kit (PR95 coder stack L21–L32)** | range/arith + colex + temporal-delta + brotli-q11 | band **≈ −0.005..−0.008** BUT **RETRACTED on the frontier (double-counted already-spent bytes)** [EXACT NO-FAKE catch] (canon R3/R12) | ~0 on frontier; band may apply to a FRESH C7 | RETRACTED (frontier) / untested (witness) | the v2-grammar materializer (~half-day $0) is the only build gap (R13) |
| **Ballé weight-entropy λ penalty** `--weight-entropy-penalty-lambda` | penalize weight entropy for bytes | **NET-NEGATIVE at every λ** (λ=5 −13% bytes but +0.029 d_seg = +2.89 WORSE; waterfill +2.77 also worse) [synthetic-scorer directional] (`lever2_lambda_star_sweep_and_waterfill`) | **+2.8 (WORSE)** → keep λ=0 OFF | measured antagonism | d_seg harm dominates byte win with OR without waterfill → CLOSED as a net-score win on this vehicle |

---

## §G. GEOMETRY / VEHICLE / WARP priors (v2 task-space)

| Lever | What it does | MEASURED EV (artifact) | ⇒ ΔS | maturity | synergy / antagonism |
|---|---|---|---|---|---|
| **D10 openpilot lane-prior φ1 / road-plane SDF** | road-plane SDF centerline (deg-3) | lane-attributable **0.000439 vs 0.000858 image-coords** (Δ −0.000419); separatrix residual 1.9e-5 [CPU-adv static-GT] (canon D10) | **−0.042** (lane sub-problem) | CPU-adv; FREE 0 bytes | ships 0 bytes (self-detect, NEVER luma-hardcode); resume-clobbered → from-scratch only |
| **D11 polynomial-fill lane geometry** | deg-3 ground-frame centerline fill | recon false-NEG 0.00046 (< target) but false-POS 0.00396 = 90% of recon d_seg [CPU-adv] (canon D11) | captures SHAPE; residual = DASH | CPU-adv | lane ≈35 floats/frame → ~1–2KB; the residual long-tail = the trained generator's job |
| **D12 ego-hood static-clamp (#139)** | freeze MyCar (IoU 0.994 static) | negligible standalone; value = frees capacity for the boundary (canon D12) | ~0 direct | measured | **FREE 0-byte**; composes with warp (hood→identity) |
| **W1/W2 stratified screw-warp (SE(3))** `src/tac/se3.py` | Road=ground-homography(pose), hood=identity, sky=rotation-only | **+15–17% d_seg on Road [pre-R n96≈n200]** (calib closes EON fx=fy=910); BUT bulk **through-R NOT free** (inherits ~0.008 jitter floor, W4/W6) | rate-win (13 keyframes→0.0060), NOT a d_seg-win via warp | pre-R + through-R negative | reuses the STORED pose at 0 bytes (FREE dual-use); **d_seg & d_pose want OPPOSITE scales** (W8) → pose stays on sidecar, warp = residual predictor |
| **Horizon-band localization** | the flip hotspot IS the calibrated horizon (cy≈437) | **97.8% of frontier d_seg in the horizon band**; flip margin mean 0.102 knife-edge [CPU-adv EXACT-faithful, n=100] (`horizon_band_dseg_lever`) | localizes the residual (no direct ΔS) | CPU-adv | near-0-byte geometric line CANNOT supply per-pixel corrections; **store-the-flips sparse sidecar = NO-GO ×3** (rank 53/60, nonlinear) |

---

## §H. POSE (SOLVED via STORE — deployed)

| Lever | What it does | MEASURED EV (artifact) | ⇒ ΔS | maturity |
|---|---|---|---|---|
| **P1 stored-target pose sidecar** `scorer_targets.py` | store 6 PoseNet scalars/pair, freeze | **d_pose≈0, 7,200 B raw / <5KB zlib** (contest-fact) | **≈ −0.017** (√(10·2.94e-5)=0.0172 → ~0) at ~+0.0006 rate | **deployed** |
| **P3 `--w-pose 0`** | pose NOT rendered; witness's only job = d_seg | witness controllable job = d_seg alone (canon P3) | (enables the split) | deployed |
| **P8 warp-carries-pose** | REFUTED for lossy (opposite scales) | d_pose 190→12.6 only at d_pose-optimal calib which WRECKS d_seg 7× (canon W8/P8) | pose stays on sidecar | measured antagonism |

---

## §I. RANKED SUMMARY

### GLOBAL TOP-8 (by ΔS-ABSOLUTE — biggest total score potential; the floor-movers)
1. **D1+D2 directional basis + KKT capacity-routing** — combined n600 0.008257→0.002447 ⇒ **ΔS ≈ −0.58** (net −0.54 after +rate). *THE d_seg engine.* [MLX-rs direct → NEEDS-N600-through-R + resolve the circular self-orient GT.]
2. **Curvelet coarse→fine scaling curriculum (§A2)** — the multiscale generalization of D1: single-scale −48% MEASURED + curvelet 0.80×-at-equal-budget + 587× SDF R-survival ⇒ **ΔS −0.21..−0.31 measured → toward −0.4+ HYPOTHESIS** at ~0 byte (rule-118 free bank). *N⁻²-optimal for the C² separatrix — the theoretically-best boundary representation.* [n6-toy measured → NEEDS-N600-through-R; multiscale climb is a HYPOTHESIS.]
3. **Deterministic backbone rate** — **ΔS_rate ≈ −0.271** (0.277→0.006) [GROUNDED] — but d_seg-DEAD alone → ONLY in the hybrid.
4. **Whole d_seg curriculum (CE→Muon)** — **ΔS −0.17** realized through-R [n96/n200] — the trustworthy (realized) number.
5. **D6 SDF hosc chart** — sets the through-R baseline at **0.00124 (ΔS ~0.12)** — the only chart that has cleared the sub-0.15 need at n96.
6. **tau_softplus stage** — **ΔS −0.088** realized [through-R] — the biggest single-stage realized drop.
7. **L13 format −59% rate** — **ΔS −0.070** [GROUNDED] — the rate half (enables the smaller representation).
8. **Pose stored sidecar** — **ΔS ≈ −0.017** [deployed, contest-fact].

### By ΔS-PER-TRAIN-TIME (cheapest floor-movers — the highest-leverage first)
1. **D1 directional basis** (~0 byte, ~0 train-time init) → −0.21..−0.31 — **#1 by a mile.**
2. **Curvelet coarse→fine scaling curriculum (§A2)** — ~0 byte; the coarse→fine ramp AVOIDS the spectral-bias fight AND shortens the Muon critical-slowing tail (Nerfies/BARF annealing + BACON band-cap) → **both a d_seg AND a wall-clock lever.**
3. **S0 structured-init seed + lane-prior φ1** (FREE, 0 ep) → separatrix residual 1.9e-5 head-start.
4. **D16 wide-SDF ramp / D13 sub-pixel placement** (0 bytes, R-survival cures) → −0.43/−0.57 on the deterministic arm.
5. **A2 lane-thin + A1 margin-saliency** (~0.6–1.5 GPU-h each) → predicted −0.04..−0.10 / −0.03..−0.08 [θ* A/B].
6. **tau_softplus stage** (~300 ep) → −0.088 realized — the cheap primary curriculum drop.
7. **A5 DM1 Stiefel-W** (0 bytes) → 0..−0.03 direct; primary value is n600 amortization.
8. **Muon finisher** (250+ ep slow tail) → −0.03..−0.06; expensive but root-tracking anneal is the wall-clock cure.

---

## §J. CONFIRMED SYNERGIES (compose) and ANTAGONISMS (do not stack)

**Synergies (measured / structural):**
- **basis-BEFORE-capacity (D1→D2)** — the #1 ordering constraint; capacity ALONE on isotropic basis HURTS +6%, but pays −64/−70% once oriented. [MLX-rs]
- **Muon is THE finisher, LAST** (after CE→tau→l7) — AdamW can't fix the off-diagonal κ~19 Hessian; Muon bites −32% MORE from stage-4. [CPU-adv]
- **lane-edge × chroma = COMPOUNDING** — chroma makes lanes separable, lane-edge sharpens them. [design, needs A/B]
- **structured-init seed → NTK jump to high-freq annulus** — seed gives low-freq FREE, then climb curvelet bands. [FEED-fs]
- **REHEAT at every transition** (partial 0.1×/8ep + reset-moments) — makes stage transitions stable-by-construction; margin-engage spike-skip cured by re-treating the spike-guard. [MLX]
- **warp reuses the STORED pose at 0 bytes** (FREE dual-use) — deterministic decode warp driven by the stored 6-scalar pose.

**Antagonisms / closed (do NOT stack):**
- **smooth stage RAISES d_seg +6.8%** → DROP from the witness curriculum. [MLX-port]
- **capacity alone on isotropic basis HURTS +6%** → never capacity-before-basis. [MLX-rs]
- **vanilla FiLM rank-1.2 collapse** = the measured 2× d_seg plateau → NEVER vanilla FiLM (use A5 Stiefel). [M2]
- **Ballé weight-entropy λ penalty net-NEGATIVE at every λ** (+2.8 worse, with/without waterfill) → keep λ=0 OFF. [`lever2_lambda_star_sweep`]
- **warp d_seg-optimal vs d_pose-optimal = OPPOSITE homography scales** → pose stays on the stored sidecar; warp = residual predictor only. [through-R]
- **MSDF dominated by single-SDF** (3.6–76× worse) → single-SDF carrier. [through-R]
- **boundary-proximity scalar on top of directional = mild drag** (−38% vs −48%) → dropped from the winner. [MLX-rs]
- **A1 (all-class) × A2 (lane-thin) = SUB-ADDITIVE** (both target the Lane boundary) → do not double-count their ΔS. [θ* design]
- **pure-KD DIVERGES** → KD only as a 0.3-weight aux (D4). [MLX-rs]
- **store-the-flips linear sparse sidecar = NO-GO ×3** (rank 53/60; compressibility is NONLINEAR). [measured]
- **DM1 per-pair FiLM DEMOTED** — PR collapses WHILE d_seg improves → not the binding cause. [EXACT $0]

---

## §K. MATURITY LEDGER (n600-validated vs NEEDS-N600 vs derived vs deployed)

- **n600-validated (but DIRECT, not through-R — FLAG):** D1 directional (n600 0.005697/0.004445), D2 capacity
  (basis+cap n600 0.002447), D7 step_basis n600 −4.5%, D5 long900. → the big d_seg magnitudes are n600 **but
  direct/MLX-rs**; the direct≪through-R gap (0.0022→0.0064) means **every one NEEDS a realized-through-R
  n600 remeasure** before it is load-bearing.
- **Realized-through-R (n96/n200) — the trustworthy d_seg data:** per-stage curriculum drops (CE/Tau/l7/Muon),
  D6 SDF hosc 0.00124, D16 wide-SDF ramp, W4/W6 bulk-through-R negatives, per-stage attribution + birth-death.
  → n96/n200 scale; **NEEDS-N600** (n changes the outcome per FEED-kn) but the R-survival gate is passed.
- **UNTESTED (predicted only — the θ* A/B, ready-to-fire):** A1/A2/A3/A4/A5/A6/A7 — every EV is a PREDICTION
  grounded in the measured per-stage attribution; the A/B ranks them; only a byte-closed n600 exact row scores.
- **NEEDS-N600-REMEASURE (toy n96 advisory — allergic-to-toys flags):** D1 −48% (n96), witness capstone lever
  sweep (n96/n32), D15 NCA (n96), the whole §A/§B/§C magnitude column where tagged n96.
- **Chroma (D9) — UNMEASURED on the realized axis, verdict-BLOCKING:** baked into BASELINE; **every pre-chroma
  d_seg verdict is PROVISIONAL** until re-measured with chroma active.
- **Derived (math bounds, not rows):** byte→ΔS slope 6.659e-7/B; label-noise confident-GT cap **ΔS ≈ 0.012**
  (seg-only best-case S ≈ 0.184 > 0.15 → sub-0.15 REQUIRES a smaller representation, not more d_seg); anneal
  T*=Δ; muon-lr band; deterministic-render floor R1 0.0185.
- **Deployed (contest-fact):** pose stored sidecar (P1), `--w-pose 0` (P3), EMA-shadow discipline, structured-init
  seed (from-scratch).

---

## §L. THE ONE BINDING GAP (for #201's synthesis to route around)
Every big d_seg ΔS is DIRECT/MLX-rs or PREDICTED. **The level-set witness has NO exact-eval path yet** — the
byte-close tool `tools/witness_byte_close_and_eval.py` is hard-keyed to the RGB witness (`out.weight`); the
level-set npz has `out_sdf`/`out_tex`/`palette`/`code` → KeyError. Until `tools/levelset_byte_close_and_eval.py`
lands, every lever above produces advisory `implied_S` rows only, and **no ΔS in this table is EXACT.** The
synergy-optimal run's END is the first byte-closed n600 `upstream/evaluate.py` row from the composed θ*; this
inventory is the MEANS. **Pointer UNMOVED 0.19110.**

**NO-FAKE ledger:** every EV carries its calibration (EXACT/CPU-adv/MLX-rs/through-R/direct/pre-R + n + maturity).
No score moved. Sources: `CANONICAL_RESEARCH_INDEX_20260629`, `thetastar_per_lever_AB_campaign_ready_20260630T191903Z`,
`all_layer2_levers_implemented_20260612`, `witness_capstone_deepmath_levers_20260625`,
`witness_per_stage_attribution_20260630T165037Z`, `birth_death_persistence_dseg_20260630T172510Z`,
`canonical_research_index_pose_curriculum_20260629`, `yousfi_levers_optimal_form_review_20260627T063335Z`,
`horizon_band_dseg_lever_20260623`, `lever2_lambda_star_sweep_and_waterfill_20260620`,
`witness_curriculum_stage_epoch_ledger_20260630T1725Z`, `levelset_curvelet_witness_feasibility_20260627`
(§A2 curvelet scaling curriculum). §A2 external-literature arXiv ids are best-recall from training knowledge,
tagged DERIVED-not-measured and "VERIFY before citing" — no fabricated id is presented as a measured fact.
