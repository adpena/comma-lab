---
title: VCM / Task-Aware / Coding-for-Machines SOTA survey — mapped to OUR exact S, ranked by $0-probe EV
authority: "[research/advisory] — pointer UNMOVED 0.19110; no paid GPU, no PR; full online-research grant 2026-06-19"
score_claim: false
promotable: false
date: 2026-06-19
provenance:
  - 6 parallel deep-research subagents (web: papers + OSS + benchmarks), 2026-06-19
  - OUR problem grounded in: CLAUDE.md (the GOAL), eval_roundtrip_deep_math_pr95_handling_and_exploits_20260619.md, SESSION_SYNTHESIS_SoT_20260617_20260618.md
  - .omx/state/canonical_frontier_pointer.json (frontier 0.19110, archive 177,169 B)
cross_refs:
  - eval_roundtrip_deep_math_pr95_handling_and_exploits_20260619.md (the exact scorer + the margin-polytope lever §5.2)
  - SESSION_SYNTHESIS_SoT_20260617_20260618.md (campaign state: RATE is binding 62%; the pincer; what's tried)
  - tasks #146 (NCA continuous-texture), #147 (best-shot int5 re-test)
---

# Coding-for-Machines / task-aware compression SOTA — for S = 100·d_seg + √(10·d_pose) + 25·B/B₀

**The mission frame (per the GOAL firewall).** Research is a MEANS. This survey's deliverable is a ranked list
of techniques each convertible into a $0 byte-closed exact-row probe toward a LOWER exact S. Pointer UNMOVED at
**0.19110** ([contest-CPU]); nothing here is a score claim. The contest's exact problem class is **Video/Image
Coding for Machines (VCM/ICM)** — compress to serve a FROZEN downstream task model, not human eyes. The field's
entire thesis is on our side: **human visual fidelity is wasted rate, and our scorer already discards it.** So
the operative question is *measurable and mostly $0*: "where does our per-video INR currently spend bytes that
neither argmax-SegNet nor the 6-dim-PoseNet reads?"

**OUR objective recap (so each finding maps precisely):** one driving video → a per-video INR (decoder+latents);
frontier = HNeRV-class 177,169 B archive (decoder ≈ 91% of bytes). **RATE is the BINDING term: 25·B/B₀ ≈ 0.118
= ~62% of the 0.191 score.** We have full gradient access to the two frozen models at compress time. The recon
MUST inflate to raw uint8 camera-res frames the official models ingest (no shipping latents the models don't
accept; no modifying the frozen models).

**The two big "we may have never tried this CLASS" finds, up front:**
1. **The model-aware rate-distortion limit (theory): bits ∝ frozen-model gradient sensitivity.** Re-allocate /
   starve our INR's bytes by the two frozen readers' per-coordinate sensitivity. Attacks RATE. (§A)
2. **Level-set / argmax-polytope coding (the genuine literature GAP): code in the quotient pixel-space modulo the
   frozen models' invariances** — perturb the recon freely INSIDE the per-pixel SegNet argmax polytope + the
   PoseNet 6-dim null subspace, spending bytes only on the thin boundary. Nobody has assembled this into a codec;
   we have every component. Attacks RATE *and* d_seg. (§F)

---

## RANKED TECHNIQUES (by EV for lowering OUR exact S, with the $0 probe each implies)

Each row flags the **term attacked**: **[RATE]** = binding (62%); **[d_seg]** / **[d_pose]** = task-distortion.
"$0 probe" = no-GPU / measurement-or-tiny-compute on the EXISTING frontier archive unless noted.

### Rank 1 — Decoder-weight entropy headroom measurement (NVRC dual-axis Gaussian, offline) **[RATE]**
**Why #1:** the decoder weights are ~91% of our 177KB. Almost all post-hoc coders (FP4+brotli) sit above the
Shannon bound of a *fitted parametric prior over the actual weights*. This is a pure, deterministic byte cut with
**zero scorer risk** — every byte saved is ΔS = −25·Δbytes/37,545,489 (1% of B₀ ≈ 0.25 points).
**Mechanism (NVRC, Kwan et al. 2024, arXiv:2409.07414):** reshape each weight tensor to 2D (out×in); fit a
**dual-axis conditional Gaussian** σ[i,j]=σ_out[i]·σ_in[j] (μ likewise) — near-zero side-info models the full
tensor; arithmetic-code under it. NVRC got −50.7% BD-rate vs HiNeRV and was the **first INR codec to beat VVC**,
largely from weight coding.
**Untried?** We tried partition stores / VQ-NeRV but NOT a learned per-tensor-Gaussian arithmetic coder fit to
the trained weights.
**$0 probe (P-R1):** load the frontier decoder weights; fit dual-axis μ/σ per tensor; compute Σ −log₂ p(ŵ) in
bytes; compare to current FP4+brotli bytes. If the Gaussian cross-entropy is materially below, build the coder.
**Expected ΔS:** RATE; NVRC's headroom is large — even a fraction of −50% BD-rate on the decoder term is
multiple-hundredths of S. Decisive, free, do first.

### Rank 2 — Best-shot int5/int4 decoder QAT (per-channel + LSQ + mixed-precision) — the #147 re-test **[RATE]**
**Why #2:** int8→int5 is ~37% fewer decoder bytes (→ archive ~177KB→~117KB → rate term drops ~0.04); int4 is
larger. Our prior int5 cap (S=0.483, d_seg walled ~0.0035) was **under-powered**: per-tensor abs-max, NO
per-channel, NO LSQ, NO outlier handling — the three worst defaults for a small overfit net. The d_seg wall is
the *predictable* failure of per-tensor quant (one fat channel sets the scale → most channels waste their int5
range → argmax flips at boundaries while PSNR barely moves), NOT a fundamental int5 limit.
**The best-shot recipe (ranked combination):**
- (1) **Per-channel symmetric** quant (one fp16/fp8 scale per output channel, no zero-point) — the single biggest
  fix; likely closes most of the 0.483 wall *before any QAT*.
- (2) **LSQ** learnable step-size QAT (Esser et al. 2020, arXiv:1902.08153; OSS `zhutmost/lsq-net`) with **LSQ+**
  MSE init (Bhalgat 2020, arXiv:2004.09576), trained against our score-aware loss (100·d_seg+√(10·d_pose)) so the
  boundary-flip gradient enters the step size. LSQ reaches FP baseline at 3-bit on ResNet → int5 weights-only
  near-lossless with a short fine-tune.
- (3) **Mixed-precision** keyed to d_seg-sensitivity (Hessian/Fisher of the contest loss; HAWQ arXiv:1905.03696):
  keep the few d_seg-sensitive layers/channels at int6–8, push the rest to int4. The argmax wall is a *few
  sensitive parameters* problem.
- (4, int4 stretch only) **Hadamard incoherence rotation** (QuIP# arXiv:2402.04396) — bijective, byte-clean,
  suppresses outlier channels.
**The near-exact-match paper:** **NeuroQuant (ICLR 2025 Spotlight, arXiv:2502.11729; OSS
https://github.com/Eric-qi/NeuroQuant)** — PTQ for *non-generalized (overfit) INR video coding*, the exact class.
It diagnoses that overfit INRs have strong **inter-layer dependency** so layer-wise PTQ (AdaRound/BRECQ/GPTQ) is
"ineffective", and uses **network-wise calibration** + the Hessian sensitivity Ω=Δwᵀ·H·Δw for mixed-precision +
a Hadamard option. It quantizes overfit INRs to INT2 (int3 is the safe aggressive floor). HiNeRV itself operates
at [3,8] bits → **our int5 target is conservative and should hold once per-channel + QAT are in place.**
**$0/cheap probe (P-R2):** clone NeuroQuant; run its PTQ on our already-trained decoder; sweep int8→int6→int5→int4
mixed; re-measure d_seg/d_pose at each bit-width (gate on the contest scorer, NOT PSNR — PSNR hides the flip).
**Expected ΔS:** RATE −0.03 to −0.05 at int5-iso-distortion; larger at int4 if d_seg holds. **This is the
strongest near-term sub-0.15 lever** (rate is binding; the review already flagged int5→S~0.14 as plausible).

### Rank 3 — Per-channel/per-coordinate task-ablation: delete the score-invisible bytes **[RATE]**
**Why #3:** the cleanest realization of "compress only what the model reads." The scalable-coding-for-machines
line (Choi & Bajić, IEEE TIP 2022, arXiv:2107.08373) gets **37–80% machine-task bitrate savings** by *deleting
the human-enhancement layer*. We have NO human layer — so any INR capacity/latent that exists only for
human-perceptual detail the two frozen models ignore is pure deletable rate.
**Mechanism:** for each latent channel / decoder tensor group, zero it and measure Δd_seg, Δd_pose. Channels with
~0 task-effect across BOTH models = pure human-enhancement bytes → delete or heavily quantize.
**Untried?** We build HNeRV-class INRs whose architecture bakes in a human-fidelity (frame-reconstruction)
objective. A task-ablation byte audit is plausibly not done as a *rate* mechanism.
**$0 probe (P-R3):** N forwards (no training, no GPU at INR scale on M5 Max/MLX) — per-channel/per-tensor ablation
→ Δd_seg, Δd_pose table → byte mass with ~0 task-effect. **Expected ΔS:** RATE; even 20% iso-task byte deletion
≈ −0.13 on the rate term. (Bounds Rank 1/2; tells you which bytes to attack.)

### Rank 4 — Autoregressive / context entropy model over the latent grid (Cool-Chic / C3 / Minnen-2018) **[RATE]**
**Why #4:** our overfit-one-video + small-fixed-latent setup is the EXACT regime Cool-Chic/C3 were built for.
A factorized prior (what we likely use) is dominated by a **causal-conv spatial-AR Laplace model** + channel-AR
context (Minnen 2018 joint AR+hierarchical, +15.8% rate; Cool-Chic 5.0 arXiv:2605.02726 beats VVC by 11% at
1–3k MAC/pixel — within our 30-min inflate budget). Checkerboard context (He 2021, arXiv:2103.15306) gets ~most
of full-AR's gain at 2 passes (40× faster) if decode time matters. **Realizer:** `constriction` Range coder
(<0.1% above the modeled bits) — strictly dominates the static per-tensor Categorical histogram we already use
(a 0-context model).
**$0 probe (P-R4):** (a) fit a per-channel Gaussian/Laplace to the frontier's actual latent histogram, compute
Σ −log₂ p vs current latent bytes (minutes, numpy) — the headroom floor; (b) train a tiny mask-conv Laplace AR
model on the *fixed* frontier latents (no decoder retrain) to minimize Σ −log₂ p; compare bits → the spatial-AR
gain. **Expected ΔS:** RATE; latents are a smaller share than weights so the win is smaller than Rank 1, but it's
cheap and stacks.

### Rank 5 — Entropy-penalized TRAINING (rate term in the R-D Lagrangian) — CEM **[RATE]**
**Why #5:** post-hoc coding only *measures* compressibility; putting −log₂ p(latent)+−log₂ p(weight) IN the
training loss *produces* compressible weights/latents. **Consistent Entropy Minimization** (Boosting-NeRV, CVPR
2024, arXiv:2402.18152; OSS `Xinjie-Q/Boosting-NeRV`): add κ·ReLU(R−R_target) with a network-free per-group
Gaussian entropy estimate (mean+var per weight group, negligible metadata). Gomes et al. (Disney, CVPR 2023) and
NVRC use the same idea with alternating R/D opt (minimize D for K steps, R on step K+1). **This is the highest
*eventual* ROI** and composes with Ranks 1/2/4, but needs a (short) retrain on the contest video — gate behind
the Rank-1/4 headroom reads.
**$0-ish probe (P-R5):** add the κ·ReLU(R−R_target) per-group-Gaussian term to our existing INR trainer; sweep κ;
re-measure B + d_seg/d_pose. **Expected ΔS:** RATE, modest-to-large; cheapest training-loop change (few lines).

### Rank 6 — Jacobian-sensitivity-weighted bit allocation (the model-aware RD limit, made operational) **[RATE]**
**Why #6:** the theory that grounds Ranks 1–3. **"Feature-Preserving RDO in Image Coding for Machines"**
(Menduiña, Pavez, Ortega 2024, arXiv:2408.07028) Taylor-expands the frozen extractor → task distortion ≈
(x−x̂)ᵀJᵀJ(x−x̂), so **bits ∝ local Jacobian energy** (and ZERO bits where JᵀJ≈0). **"Model-Aware
Rate-Distortion Limits"** (Enttsel & Corlay, arXiv:2602.12866, Feb 2026 preprint — unverified peer review) proves
the rate-optimal allocation puts bits ∝ task-model gradient sensitivity — our exact regime (frozen model +
gradient access + rate-binding). The classical closed form this generalizes is **reverse water-filling /
Shoham-Gersho weighted-MSE** — so the byte you can shed from low-sensitivity regions is closed-form, not a sweep.
**Untried as a clean closed-form?** We have master-gradient/saliency machinery (Catalog #121 d_seg-aware taper,
Gini≈0.60) but a "bits ∝ sensitivity" realloc grounded in this RD-limit, folding the √(10·d_pose) nonlinearity
(∂S/∂d_pose=85.8) into the weight, may not be assembled.
**$0 probe (P-R6):** on each frontier frame compute the combined sensitivity map
M(p)=100·(seg-flip-risk) + (5/√(10·d_pose))·|∂pose/∂p| (finite-diff on CPU works); histogram M; the byte mass at
low M = free rate. **Expected ΔS:** RATE; this is the *map* that directs Ranks 1–3.

### Rank 7 — Margin / argmax-polytope coding (inverse-steganalysis; the d_seg lever) **[d_seg + RATE]**
**Why here (but high strategic value):** d_seg = argmax-FLIP rate ⇒ each pixel has a per-pixel **argmax polytope**
(logit perturbations keeping top-1). The recon may roam inside it at ZERO d_seg cost. Spend 0 bytes where the
top-2 margin is large (huge polytope), bytes only on thin-margin boundary pixels (~1.3%). This is **UNIWARD**
(Holub-Fridrich-Denemark 2014, EURASIP; DDE-lab reference code public) with the cost map replaced by the frozen
scorer's sensitivity, and the **dual of invariance-based adversarial examples** (Tramèr et al. ICML 2020,
arXiv:2002.04599, OSS `ftramer/Excessive-Invariance` — they find directions that DON'T change the output; we
exploit them to shed rate). The eval-roundtrip memo §5.2 already named this lever ("margin-polytope free budget,
$0 deterministic").
**$0 probe (P-R7):** on frontier frames, SegNet per-pixel top-2 logit margin → count large-margin pixels
(coarse-codeable / shareable) → bytes saved if those pixels quantize to 1–2 bits vs current; cross-check actual
d_seg flip count (we have `scorer_sensitivity_map` + `compute_uniward_cost_map` + the joint-P18/P19 dead-zone
tooling). **Expected ΔS:** primarily RATE (coarsen the safe majority), with a d_seg residual-coder on the boundary.

### Rank 8 — Level-set / fiber (quotient) coding — the GENUINE GAP, "never tried this CLASS" **[RATE, fundamentally different paradigm]**
**Why flagged separately:** searches confirm the differential-topology framing — "quantize the recon within the
preimage/fiber of the frozen model's output; treat the level set as the coding-lattice cell" — **does not exist in
the compression literature.** Everyone trains a new codec; nobody codes the quotient pixel-space ℝᴺ /
(SegNet-argmax-polytope × PoseNet-6dim-null). The information-theoretic backing is **"Lossy Compression for
Lossless Prediction"** (Dubois et al., NeurIPS 2021, arXiv:2106.10800, OSS `YannDubs/lossyless`): you only pay to
encode the **equivalence class (orbit)** under the task's invariances (>1000× savings claimed, no accuracy loss) —
the rigorous statement of our "scorer-quotient / null-space compiler" idea (tasks #47/#49). This is the
highest-ceiling, highest-risk class; the admissible realization is forcing our INR to be the cheapest frame-tuple
whose *orbit* matches (= Rank 9's objective). **Caveat:** the Dubois codec outputs latents (inadmissible as-is);
we use the THEOREM, not the codec.
**$0 probe (P-R8):** estimate the dimension of SegNet's argmax-invariant pixel subspace per frame =
(#pixels)·log₂(levels) − H(argmax-map) − (pose-relevant bits); the gap is an upper bound on free rate vs B.

### Rank 9 — Distillation-to-the-codec: drop the pixel/PSNR loss term ENTIRELY **[d_seg + d_pose; objective change, low risk]**
**Why:** TACTIC (Kubiak-Hadfield 2021, arXiv:2109.10658) and Observer-Dependent compression (arXiv:1910.03472)
show the SOTA move when only the frozen model matters is to make the reconstruction loss BE the task loss through
the frozen net, with **zero** pixel weight (not down-weighted). We already mandate score-aware loss; the research
refinement is **boundary-weighted target-class KD / DKD** (Zhao et al. CVPR 2022, arXiv:2203.08679) — since d_seg
cares only about argmax, full-logit KL (our Hinton T=2.0) is HARD-EARNED as a temperature but CARGO-CULTED as the
seg functional; DKD's decoupled/target-class form + a boundary edge-loss (BPKD) is the d_seg-optimal distill.
**$0-ish probe (P-R9):** swap the seg-distill loss to boundary-weighted TCKD in our trainer; A/B vs current
KL-T2 on a short run; falsifiable if ≥8% d_seg reduction at matched bytes. **Same vehicle, different loss — not a
paradigm shift; lowest-risk, already-aligned.**

### Rank 10 — Continuous-texture generator for the d_seg wall (NCA, best-shot) — the #146 re-test **[d_seg]**
**Why:** the eval-roundtrip memo §3 proved the d_seg wall is SegNet's TEXTURE-dependence (flat per-class colour
lands outside the argmax polytope at boundaries). A few-param iterated texture generator could paint "enough"
continuous texture byte-cheaply. Our prior NCA probe was a fragile spark (2/8 converge) because it **omitted the
canonical Mordvintsev stabilizers**. This is a d_seg lever (NOT rate-binding) — important but secondary to the
rate attacks above unless it unlocks a byte-cheap-texture decoder.
**The named fixes (Growing-NCA, Distill 2020; Self-Organising Textures, Distill 2021):** (1) **POOL/sample-replay**
(pool=1024, batch=8–32) — turns the target into a stable attractor; (2) **reseed the highest-loss sample**;
(3) **per-variable L2 gradient normalization** (the explicit anti-explosion fix); (4) **stochastic update mask
p=0.5**; (5) overflow/clip to [-1,1]. Each independently targets one named failure mode. **Cross-frame
amortization (the heart of #146): ONE 1,500-param signal-conditioned rule generates 8 distinct textures** via a
1–3-channel binary "genomic signal" (Catrina-Plajer-Băicoianu, Sci.Reports 2025, arXiv:2407.05991) — the
existence proof that shared-rule + tiny-per-frame-signal converges at ~1.5K params. **DyNCA** (CVPR 2023,
arXiv:2211.11417, OSS `IVRL/DyNCA`) is video-native (temporal coherence + motion conditioning) if the static rule
underfits. Replace the VGG-Gram loss with our SegNet-margin + reconstruction loss; adopt the **Laplacian
perception channel** (2nd-order spatial structure conv stacks respond to).
**$0/cheap probe (P-R10):** re-run #146 with POOL replay + per-var grad-norm + reseed-highest-loss + stochastic
mask + a 2–4-dim per-frame latent; train ONE rule across n=1→16→48 frames; **average d_seg, report true amortized
rate** (rule bytes/N + latent bytes×N); measure convergence RATE over many seeds (CAX/JAX harness). **Expected
ΔS:** d_seg, gated on whether amortized rate beats the frontier decoder.

---

## BINDING-TERM MAP (which rank attacks WHAT)

| Rank | Technique | Term | Paradigm | Untried? | $0 probe |
|---|---|---|---|---|---|
| 1 | NVRC dual-axis Gaussian weight coding | **RATE** | entropy-coder bolt-on | likely | P-R1 weight cross-entropy vs current bytes |
| 2 | Best-shot int5/int4 QAT (per-ch+LSQ+mixed; NeuroQuant) | **RATE** | quant of same vehicle | yes (under-powered before) | P-R2 NeuroQuant PTQ sweep |
| 3 | Per-channel task-ablation byte deletion | **RATE** | scalable/ROI realization | likely | P-R3 ablation Δd_seg/Δd_pose table |
| 4 | AR/context latent entropy model (Cool-Chic/C3) | **RATE** | entropy-coder upgrade | likely | P-R4 fitted-prior + tiny AR bits |
| 5 | Entropy-penalized training (CEM) | **RATE** | loss-term add (retrain) | yes | P-R5 κ·rate term sweep |
| 6 | Jacobian-sensitivity bit allocation | **RATE** | RDO objective reframe | partly | P-R6 sensitivity histogram |
| 7 | Margin/argmax-polytope coding (UNIWARD-style) | **d_seg+RATE** | constrained coding | partly (named, unbuilt) | P-R7 margin map + coarsen-safe bytes |
| 8 | **Level-set/fiber quotient coding** | **RATE** | **fundamentally different** | **never (literature gap)** | P-R8 invariant-subspace dimension |
| 9 | Distill-to-codec, drop pixel loss / DKD | d_seg+d_pose | same vehicle, loss change | partly | P-R9 boundary-TCKD A/B |
| 10 | Continuous-texture NCA best-shot (#146) | d_seg | generative axis | yes (stabilizers omitted) | P-R10 POOL-replay shared-rule |

**Strategic read:** the BINDING term is RATE, and **6 of the top-8 ranks attack RATE** — the campaign's d_seg
work (pincer, NCA) is real but secondary while rate sits at 62% of S. The two cheapest, highest-confidence,
zero-scorer-risk moves are **P-R1 (weight entropy headroom)** and **P-R3 (task-ablation byte deletion)** — both
pure measurement on the existing frontier, both directly bounding how much rate is deletable. **P-R2 (best-shot
int5, #147)** is the strongest *near-term sub-0.15* lever (the review already projected int5→S~0.14). The
highest-ceiling "never tried this CLASS" is **Rank 8 level-set/fiber coding**, with **Rank 6** the operational
bridge to it.

---

## SOTA-but-INADMISSIBLE (honest flags — do NOT pursue)
- **Compressed-domain / latent task inference** (Torfason et al. ICLR 2018 arXiv:1803.06131; semantic-seg-in-
  compressed-domain arXiv:2209.01355): runs the task model on the latent, never decodes to frames. Our official
  frozen models ingest uint8 camera-res frames — **flatly inadmissible; no variant produces frames.** This is the
  trap our constraint forbids.
- **Embedding-Compression-Distortion VCM** (arXiv:2503.21469), JPEG-AI machine track, anything that **modifies/
  augments the downstream model** — our scorers are frozen and official. Inadmissible.
- **COIN++/meta-learned shared base** (arXiv:2201.12904) and **SR-NeRV** (arXiv:2505.00046): only win if a fixed,
  video-independent base decoder can ship as UNCHARGED runtime in inflate.sh with only tiny per-video modulations
  charged. **Gated entirely on a contest-compliance ruling** (operator-routable). If the base must live inside the
  charged archive, it gives nothing.
- **Full feature-space coding** (Choi-Bajić scalable, FCM, Dubois): the *theory* (orbit/quotient coding, 37–80%
  machine savings) is gold and drives Ranks 3/8; the *codecs* output latents → use the principle, not the codec.

## Honesty / verification notes
- All BD-rate / accuracy-saving numbers (−50% NVRC, 37–80% scalable, +15.8% Minnen, LSQ 3-bit≈FP) are
  **PSNR/natural-task-domain** results = directional priors, NOT predictions for our argmax-d_seg/pose-MSE
  objective. The named $0 probes convert each into a measured byte/ΔS number on OUR archive before any spend.
- **OSS verified public:** NeuroQuant (`Eric-qi/NeuroQuant`), CompressAI (`InterDigitalInc/CompressAI`),
  constriction (`bamler-lab/constriction`), Boosting-NeRV (`Xinjie-Q/Boosting-NeRV`), lsq-net (`zhutmost/lsq-net`),
  Cool-Chic, DyNCA (`IVRL/DyNCA`), lossyless (`YannDubs/lossyless`), UNIWARD (DDE-lab), Excessive-Invariance.
  **No public OSS found** for NVRC (project page only — budget re-impl) and Menduiña-Pavez-Ortega 2024 (self-impl
  the Jacobian sketch).
- **Recent unverified-peer-review preprints** (treat as strong priors only): Model-Aware RD Limits
  (arXiv:2602.12866), Unified ROI-GGM (arXiv:2602.01325), TeCoNeRV (arXiv:2602.16711).

## Wire-in (per "Results must become system intelligence")
This is a research/advisory survey (`research_only=true` w.r.t. the 6 solver hooks). The actionable wire-ins are
the 10 $0 probes above — each is a measurement-first step that produces a byte/ΔS row on the EXISTING frontier
archive (no GPU, no PR). Suggested execution order by EV×cheapness: **P-R1 → P-R3 → P-R6 → P-R4 → P-R2 (#147) →
P-R7**, then the retrain-gated P-R5 / P-R9, with P-R10 (#146) on the d_seg axis in parallel. Pointer UNMOVED
0.19110; nothing here is a score claim.
