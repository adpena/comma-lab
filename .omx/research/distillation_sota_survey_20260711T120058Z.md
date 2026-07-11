# Knowledge Distillation — SOTA survey (2024–2026 frontier), ranked by EV for the costate-organ scorer-surrogate + the witness

**Agent:** research surveyor (Fable), 2026-07-11 · **Cost:** $0 (web research only; no scorer forwards; pid 88030 untouched).
**Pointer 0.19108282 [contest-CPU] UNMOVED** — this survey is MEANS (campaign accelerator). Every applicability label is honest:
**MEASURED-elsewhere** (paper's own benchmark) / **CLAIMED** (paper claim, not independently reproduced) / **SPECULATIVE-for-us**
(no measurement on our vehicle; earns adoption ONLY via the organ's backtest / a real n600 row).
**Primary consumer:** `.omx/research/amortized_operator_pontryagin_loop_cluster_20260711.md` (#426 λ-organ: λ=∂S/∂x over frozen
SegNet EfficientNet-B2 5-class-argmax + PoseNet FastViT-T12). **Secondary:** witness KD warm-start / distill-to-smaller-basis.
**Sister ledger:** `papers_checked_knowledge_distillation_sota_20260711.md` (per-paper rows, anti-re-research).

---

## 0. THE FRAMING SHARPENING (read first — it reorders the whole survey)

Our teachers are **white-box, frozen, and differentiable almost everywhere**. SegNet/PoseNet forward passes are ordinary
differentiable networks; the non-differentiability of S lives in the **METRIC**, not the teacher: `d_seg = per-pixel 0-1
argmax-disagreement` (piecewise-constant in logits), and `d_pose = MSE` (already smooth). So "distill the frozen teacher into a
differentiable surrogate" decomposes into TWO distinct problems with different SOTA:

- **(A) Metric relaxation** — smooth the argmax-metric of the *existing* frozen teacher. Needs NO learned surrogate at all:
  temperature-softmax / margin fields / Lovász extension / perturbed-optimizer smoothing. Cheapest, exactest, already
  half-built in this repo (margin field ↔ Fisher 0.978; τ=ε=ħ Maslov result).
- **(B) Surrogate distillation** — train a smooth network S̃(x) ≈ S(x) (or λ̃(x) ≈ ∂S/∂x directly) when you need λ *cheap*
  (no full scorer forward), *smooth across flip discontinuities*, or *defined on campaign-state space* rather than pixel space
  (the #426 λ-field case). This is where the KD literature proper pays.

The TOP-5 below interleave both. Everything is SPECULATIVE-for-us until backtested.

---

## TOP-5 HIGHEST-EV — distilling frozen SegNet/PoseNet into a differentiable λ-surrogate

**#1 — Perturbed-optimizer / temperature relaxation of the argmax metric itself (route A; no training).**
Berthet et al., *Learning with Differentiable Perturbed Optimizers*, NeurIPS 2020, **arXiv 2002.08676** (+ Vlastelica et al.,
*Differentiation of Blackbox Combinatorial Solvers*, ICLR 2020, arXiv 1912.02175). Replace `argmax` with the expectation of
argmax under input-noise ε → gradients are exact Monte-Carlo estimates of a *smoothed* d_seg; ε controls the
smoothing-vs-fidelity trade **and is mathematically the same knob as our measured τ=ε=ħ (Maslov err ≤ τ·ln5, L75)** — the
Fenchel-Young / Gumbel-softmax temperature family is the analytic special case (Jang/Maddison 2016, arXiv 1611.01144).
*Applies:* λ on pixel/logit space = ∂(smoothed d_seg)/∂x computed THROUGH the real frozen SegNet — zero model error, no
surrogate-vs-teacher gap to audit. MEASURED-elsewhere (combinatorial benchmarks); the margin-surrogate half is effectively
MEASURED-in-repo (Fisher↔margin 0.978). **EV: highest — do this before any learned surrogate.**

**#2 — Jacobian/Sobolev-matching distillation (route B; THE λ-fidelity objective).**
Srinivas & Fleuret, *Knowledge Transfer with Jacobian Matching*, ICML 2018, **arXiv 1803.00443**; Czarnecki et al., *Sobolev
Training*, NeurIPS 2017 (arXiv 1706.04859). If we train a surrogate S̃, matching outputs alone leaves ∂S̃/∂x unconstrained
exactly where the organ reads it. Match Jacobians too; two decisive tricks: (i) Jacobian-matching ≡ soft-label distillation
with input noise (so cheap noise-augmented KD buys gradient fidelity), (ii) random-projection Sobolev matching (match Jᵀv for
random unit v) kills the huge-Jacobian cost — for a (H×W×5) SegNet Jacobian this is the only tractable form. 2024 follow-on:
backward-pass knowledge distillation (arXiv 2301.12006). *Applies:* the training objective for ANY learned λ-surrogate — both
a pixel-space S̃ and the #426 campaign-state λ-net (whose self-supervised-rollout loss is already an implicit Sobolev form).
MEASURED-elsewhere (classification/regression transfer). **EV: very high, and it composes with #1 (distill the ε-smoothed
teacher, not the raw argmax).**

**#3 — Boundary/annulus-weighted canonical KD for dense prediction (route B; segmentation-specific).**
The 2026 headline: *The Surprising Effectiveness of Canonical Knowledge Distillation for Semantic Segmentation*,
**arXiv 2604.25530** — at MATCHED wall-clock compute, plain logit+feature KD beats the specialized seg-KD zoo (CWD arXiv
2011.13256, CIRKD arXiv 2204.06986, BPKD arXiv 2306.08075, ACAM-KD ICCV 2025); ResNet-18 student hits 99% of teacher mIoU on
Cityscapes. The specialized methods' wins were an unequal-compute artifact. BUT boundary-aware *sampling* remains free: BPKD's
edge/body loss split (CLAIMED +up-to-3.87% over CWD at fixed iters) rhymes exactly with our MEASURED annulus fact (~97% of
d_seg in ~4.7% area, L66). *Applies:* if we distill SegNet into a smaller/smoother student (witness-side or surrogate-side),
use canonical logit-KD (temperature soft labels) + longer training, with sample/pixel weighting concentrated on the boundary
annulus — not a bespoke seg-KD objective. **EV: high; also the recipe for distill-to-smaller-basis.**

**#4 — Decision-boundary distillation via adversarial/near-boundary samples (route B; where to QUERY the teacher).**
Seminal: Heo et al., *KD with Adversarial Samples Supporting Decision Boundary*, AAAI 2019, **arXiv 1805.05532** (boundary-
supporting samples = adversarial points ON the teacher's boundary carry the discriminative geometry). Robust-distillation
line: RSLAD (arXiv 2108.07969), AdaAD (2024), ECCV 2024 variance-gap, ICLR 2025 medium-difficulty-samples (boundary-drift
control), *Why Robust Teachers Fail* (arXiv 2605.21999). Data-free KD supplies the generator machinery when you must
SYNTHESIZE boundary queries (DiffDFKD arXiv 2504.00870 = diffusion-guided synthesis, 2025 SOTA). *Applies:* a surrogate is
only trusted where it was fit — fit it ON the flip-set: sample perturbations of witness renders that cross SegNet's argmax
boundary (our margin field already locates them for free) and weight KD there. This is ALSO literally the contest's own game:
the adversarial-embedding steganography lineage (ADV-EMB arXiv 1803.09043; game-theoretic framework arXiv 1906.00697) attacks
CNN steganalyzers through exactly such surrogate gradients — our inverse-steg frame ratified by the attack literature.
**EV: high; pairs with #1 (margin field = free boundary sampler).**

**#5 — On-policy distillation: fit the surrogate on the STUDENT'S own trajectory distribution (route B; distribution-shift
control).** Agarwal et al., *GKD: On-Policy Distillation of LMs*, ICLR 2024, **arXiv 2306.13649** (reverse-KL + student-
generated inputs); *A Survey of On-Policy Distillation for LLMs*, arXiv 2604.00626; industrial default per the lab-frontier
sweep (DeepSeek-V4 reverse-KL on-policy panel consolidation). The transplant: a λ-surrogate trained on a frozen corpus of
renders goes stale as the witness descends — the input distribution IS the training trajectory. Refit the surrogate on-policy
(on the live run's own renders/states), reverse-KL-style mode-seeking so it does not overestimate never-visited regions.
The #426 organ already does this in embryo (λ-net trained on #205's own trajectory; SAO async-refit + trust-region addendum
§5c). **EV: high as a DISCIPLINE — it is the correctness condition for any learned surrogate we deploy in the loop.**

---

## THE AREAS — SOTA + seminal + OSS + applicability + EV

### 1. Frozen / non-differentiable teacher → differentiable surrogate ⭐ (the money area)
- **SOTA (metric-relaxation):** perturbed optimizers (2002.08676), blackbox-solver differentiation (1912.02175), Fenchel-Young
  losses; Lovász-softmax (Berman CVPR 2018, arXiv 1705.08790) = the convex surrogate of set-IoU (our d_seg is per-pixel 0-1,
  so softmax-margin/temperature is the tighter fit; Lovász if we ever score IoU-like quantities). Learning-to-surrogate line:
  *Learning Surrogate Losses* (1905.10108), *Relational Surrogate Loss Learning* (ICLR 2022, 2202.13197 — ranking-correlation
  instead of value-matching: fit the surrogate to preserve ORDERINGS of S, which is all a duty-queue ranker needs).
- **SOTA (query-synthesis for black-box boundary capture):** DFKD with diffusion (2504.00870); OOD-trap escape (2507.04119).
- **Seminal:** Hinton et al. 2015 (1503.02531 — soft labels ARE the original differentiable surrogate of a hard teacher);
  Gumbel-softmax/Concrete (1611.01144/1611.00712); straight-through (1308.3432).
- **OSS:** `google-research/perturbations` + `tuero/perturbations-differ` (perturbed optimizers); `bermanmaxim/LovaszSoftmax`.
- **For us:** #1/#2/#4 above. λ smooth everywhere at controllable ε, exact through the real teacher. SPECULATIVE-for-us.
- **EV: 1st.**

### 2. Segmentation / dense-prediction / pose KD (directly our scorers)
- **SOTA:** canonical-KD-at-matched-compute (2604.25530) DETHRONES the specialized zoo — the single most decision-relevant
  2026 seg-KD result. Zoo for reference: CWD (2011.13256), CIRKD (2204.06986), MasKD, FAKD (2208.14143), BPKD (2306.08075,
  boundary-privileged), ACAM-KD (ICCV 2025, adaptive student-teacher cooperative masking). Pose: uncertainty-aware 6DoF KD
  (2503.13053), SCJD sparse-correlation joint distillation (ICME 2025, 2503.14097), SDPose self-distillation (2404.03518);
  heatmap→regression KD hybrids. PoseNet's 6-dim MSE head is already smooth — pose-KD matters only for witness-side
  compression, not for λ smoothing.
- **Seminal:** structured KD for semantic segmentation (Liu et al. CVPR 2019, 1903.04197 — pairwise + holistic/adversarial).
- **OSS:** `winycg/CIRKD` (bundles CWD/CIRKD baselines for Cityscapes); `megvii-research/mdistiller`; MMRazor seg configs.
- **For us:** the distill-to-smaller-basis recipe (TOP-3). Boundary-weighted canonical KD. SPECULATIVE-for-us.
- **EV: 2nd (as the witness-side recipe).**

### 3. Feature / representation / relational / manifold KD
- **SOTA 2024-2026:** logit standardization (CVPR 2024, arXiv 2403.01427 — z-score logits before KD; fixes temperature-sharing
  fallacy, plug-in gain over DKD/vanilla); Scale-Decoupled Distillation (CVPR 2024); DiffKD (2305.15712 — denoise student
  features with a diffusion model on teacher features); AttnFD (2024); CanKD (2511.21503, cross-attention non-local KD);
  Local Dense Logit Relations (2507.15911); iCD implicit clustering distillation (2509.12553); *A Functional Perspective on
  KD* (2510.12615). Transformed Teacher Matching (2402.11148 — drop temperature on student side ⇒ Rényi-regularized
  matching).
- **Seminal:** FitNets (1412.6550); attention transfer (1612.03928); RKD (1904.05068 — distance/angle relations = manifold
  geometry); CRD (1910.10699); DKD (CVPR 2022, 2203.08679 — target/non-target decoupling).
- **OSS:** `megvii-research/mdistiller` (DKD + logit-std + zoo, maintained); torchdistill (26 methods).
- **For us:** RKD-style *relational* matching is the right shape for distilling SUBSPACE geometry (our BSF block-subspaces);
  logit standardization is a free hygiene trick for any KD we run. DKD's insight (non-target-class mass carries the boundary
  info) is exactly why soft labels capture the argmax boundary. SPECULATIVE-for-us.
- **EV: 3rd.**

### 4. Adversarial / robust / boundary distillation
- **SOTA:** AdaAD (adaptive inner-max alignment); ProARD (2506.07666, progressive robust students); DGAD (2409.01627);
  heterogeneous-teacher robustness (2402.15586); variance-gap reduction (ECCV 2024); *why robust teachers fail* (2605.21999);
  faithful KD (2306.04431 — student agrees with teacher UNDER perturbation balls, i.e. certified-agreement, the strongest
  formal notion of "captured the decision boundary").
- **Seminal:** BSS boundary-supporting samples (1805.05532); ARD (1905.09747); RSLAD (2108.07969).
- **For us:** TOP-4 above — boundary-query curriculum for the surrogate; faithful-KD's ball-agreement is the right acceptance
  test for "surrogate captured SegNet's separatrix" (vs argmax-agreement rate, which is blind off-boundary). Also ratifies our
  inverse-steg arm: the whole adversarial-embedding literature (ADV-EMB 1803.09043, 1906.00697, MDPI 2025 attack comparison)
  is gradient-through-detector attack — what we do to the frozen scorers, formalized. SPECULATIVE-for-us.
- **EV: 4th.**

### 5. Self-distillation / self-supervised
- **SOTA:** DINOv3 (Meta, Aug 2025, **arXiv 2508.10104**, 7B SSL ViT) — the headline mechanism for US is **Gram anchoring**:
  constrain the Gram matrix of current dense features to an earlier stable checkpoint to stop dense-feature degradation over
  long training. iBOT (2111.07832), DINOv2 (2304.07193); born-again networks (1805.04770 — sequential self-KD beats teacher);
  self-distillation-as-regularizer theory (Mobahi et al. 2002.05715 — amplifies then shrinks; few rounds help, many collapse).
- **For us:** Gram anchoring is a drop-in idea for OUR long witness runs (dense-feature drift over 10k+ epochs — same disease,
  MEASURED-elsewhere at 7B scale); born-again = a $0 witness experiment shape (retrain the witness against its own EMA soft
  output). SPECULATIVE-for-us.
- **EV: 5th (Gram anchoring), rest lower.**

### 6. Dataset distillation
- **SOTA:** SRe2L (NeurIPS 2023, 2306.13092) → RDED (CVPR 2024) → generative/latent (GLaD CVPR 2023; LD3M; diffusion-based
  2403.03881; OT-geometry 2512.00308); DD-Ranking (2505.13300) exposes evaluation inflation; **the sobering 2026 result:
  arXiv 2604.18811 + 2606.18209 — random-image baselines match SOTA DD once SOFT LABELS are used; the knowledge lives in the
  labels/relabeling, not the synthesized pixels**; survey 2502.05673.
- **Seminal:** Wang et al. 2018 (1811.10959); trajectory matching MTT (2203.11932).
- **For us:** if we ever distill comma10k → compact set for scorer-surrogate pretraining, the evidence says: spend on SOFT
  RELABELING with the frozen teachers over synthetic-pixel optimization. Low priority — we have the actual contest video and
  free teacher queries. SPECULATIVE-for-us. **EV: 8th.**

### 7. Diffusion / generative distillation
- **SOTA:** DMD (2311.18828) → **DMD2** (NeurIPS 2024, 2405.14867, GAN-augmented, beats teacher one-step) → sCM/TrigFlow
  (OpenAI 2410.11081, continuous-time consistency at 1.5B+) → **rCM score-regularized continuous-time consistency
  (2510.08431, matches DMD2 without GAN tuning, better diversity)** → 2026: continuous-time distribution matching (2605.06376),
  AdvDMD (2604.28126), TwinFlow, MeanFlow distillation (2606.11155). Seminal: progressive distillation (Salimans & Ho
  2202.00512); consistency models (2303.01469); SDS/DreamFusion (2209.14988).
- **For us:** mechanism-mining only — DMD's trick (match DISTRIBUTIONS via two score fields, not trajectories) is the same
  move as our distribution-level λ; score-distillation = using a frozen model's gradient field as a loss is structurally our
  "frozen scorer as loss" pattern, at industrial maturity. No direct adoption path. Fast-moving = hype-rich; load-bearing
  pieces are DMD2/rCM. **EV: 7th (idea-transfer only).**

### 8. Distillation scaling laws + when-it-helps
- **SOTA:** Busbridge et al. (Apple), *Distillation Scaling Laws*, **arXiv 2502.08606** (ICLR 2025): student performance as a
  function of compute split; distillation beats supervised ONLY when the teacher is amortized (exists already / many
  students); capacity-gap is real and predictable — stronger teacher ≠ better student past a crossing. Law of capacity gap
  (2311.07052: optimal teacher scales ~linearly with student); functional perspective (2510.12615); *What mechanisms does KD
  distill?* (Wu et al. 2024); fidelity paradox (2505.15442 — students generalize while disagreeing with teachers; agreement ≠
  the objective). Seminal: *Does KD really work?* (Stanton et al. NeurIPS 2021, 2106.05237's sibling); *Patient & Consistent*
  (Beyer et al. CVPR 2022, 2106.05237 — function matching: SAME augmented views to both + long training = the compute-matched
  lesson that 2604.25530 rediscovered for segmentation).
- **For us:** our teachers are FROZEN and FREE-to-query = precisely the "teacher exists" regime where distillation dominates.
  Sizing guidance for a surrogate: match capacity to the student's (organ's) needs, not the teacher's; patient-and-consistent
  says long+consistent-views beats clever objectives. DERIVED-guidance from MEASURED-elsewhere laws. **EV: 6th (design
  discipline for everything above).**

### 9. OSS libraries (maintained + usable, 2026)
- **`yoshitomo-matsubara/torchdistill`** — PyTorch Ecosystem member, YAML-config, 26 methods, reproducible logs. The
  best-maintained pure-KD framework. **Adopt-first for any KD experiment.**
- **`megvii-research/mdistiller`** — DKD/DOT/logit-std reference impls; research-grade, canonical for logit-KD baselines.
- **`open-mmlab/mmrazor`** — KD+NAS+pruning+quant unified; heavyweight OpenMMLab dependency; low churn since 2023 — use for
  its seg-KD configs (CWD) as reference, not as a dependency.
- **`winycg/CIRKD`** — the segmentation-KD zoo in one repo (Cityscapes/ADE20K).
- **KD_Lib** — stale; skip. HF ecosystem: distillation recipes live in `transformers` examples + TRL (GKD trainer — the
  on-policy reference impl, `trl.GKDTrainer`).
- **For us:** we build MLX-first with numpy-fp32 authority; these are REFERENCE surfaces, not dependencies. Repo-native
  reimplementation per UNIQUE-AND-COMPLETE-PER-METHOD.

---

## HYPE vs LOAD-BEARING (honest flags)
- **Load-bearing:** perturbed-optimizer smoothing (math is exact); Jacobian≡noise-KD equivalence; canonical-KD-at-matched-
  compute (2604.25530 — an equal-compute audit, the kind of result that survives); distillation scaling laws (dense
  empirical grid); DMD2/rCM (deployed at scale); soft-labels-do-the-work in dataset distillation (a debunk, so robust).
- **Hype-rich / handle with tongs:** the long tail of per-venue seg-KD objectives (compute-confounded per 2604.25530); most
  "novel KD loss" papers (+0.3 mIoU class); dataset-distillation SOTA claims pre-DD-Ranking; "student surpasses teacher"
  claims without the capacity-gap control.
- **The one result that reframes our defaults:** 2604.25530 + Beyer-patient-consistent — clever KD objectives are mostly
  compute reallocation; spend the cleverness on WHERE to query the teacher (boundary annulus, on-policy renders) and keep the
  loss canonical.

## NEXT ACTIONS (advisory; each earns adoption only via backtest / measured row)
1. $0: wire an ε-perturbed smoothed-d_seg readout next to the existing margin surrogate; A/B their λ-fields on banked #205
   telemetry (organ backtest harness exists: `tools/lambda_net_backtest.py`).
2. $0: add random-projection Sobolev term to any future λ-net refit (one extra VJP per sample through the frozen scorer).
3. Witness-side: distill-to-smaller-basis experiment shaped as canonical logit-KD + annulus-weighted sampling + patient
   schedule (NOT a bespoke seg-KD loss).
4. Acceptance test for any surrogate: faithful-KD-style ball-agreement on the boundary annulus, not global argmax agreement.

**Pointer 0.19108282 [contest-CPU] UNMOVED.**
