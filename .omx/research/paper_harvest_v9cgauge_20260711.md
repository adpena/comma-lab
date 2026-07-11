# Paper harvest → V9·CGauge triage (2026-07-11) — 13 papers + The Well corpus

**Scope:** $0 online-read harvest (abstracts + method sections via arXiv/site fetch), triaged
against the V9·CGauge witness stack (single covariant trunk · level-set flow in frozen-scorer
Fisher metric · se(3) ξ · costate #247 · gauge-zero-mode carriers). Pointer **0.19108282
[contest-CPU] UNMOVED** — this is a harvest, not a score; every claim below is READ-derived
(labels: MEASURED-by-paper / DERIVED / INFERRED / SPECULATIVE-UNTIL-PROBED).

**STORES CONSULTED:** `tools/graph_memory_recall.py` ("adjoint costate optimal control…",
"INR video compression training recipe… operator learning FNO") · papers-checked ledger
`reference_papers_checked_not_relevant_or_watch_item_ledger_20260701.md` (grep hit: 2602.12866
already FRAMING; 2506.24127 already WATCH/#211-skeleton; SMC++ already in the v2 originality
neighbor map) · `vehicle_v9_cgauge_naming_20260711.md` · MEMORY.md L55/L86/L87/L-v8.
Prior verdicts NOT re-derived; deepened only where this harvest's charter asked.

---

## Ranked triage table

| # | Paper | Verdict | Surface | EV × cost |
|---|-------|---------|---------|-----------|
| 7 | EKI gradient-free neural-ODE training (2307.07882) | **LEVER** (→ #396 MC-finisher: EKI over gauge-zero-mode constants vs EXACT argmax d_seg) | d_seg, surrogate↔exact gap | **HIGH-EV × small tool** |
| 1 | Adjoint Neural Regulator (2606.16303) | **FRAMING** (costate-leg input for #247/Einstein; amortized-costate lever SPECULATIVE-UNTIL-PROBED) | costate/curriculum | HIGH-framing × $0 note |
| 5 | Self-supervised amortized neural operators, scaling laws (2512.24897) | **FRAMING** (quantitative: amortization viability ∝ intrinsic dim; ours = rank-8 → FAVORABLE regime) | costate + #211 | MED × $0 note |
| 3 | Model-aware task-RD limits (2602.12866) | **FLOOR-CONFIRM** (prior ledger verdict stands; NEW: "transmitter-side complexity is the bottleneck" ⇒ our unlimited compress-time compute is exactly the gap-closer) | floor | MED × $0 |
| 6 | Constrained neural parameterization (2606.00855) | **FRAMING** (constraint-by-architecture ≻ penalty; SPECULATIVE: eikonal-by-construction parametrization) | parametrization | LOW-MED |
| 2 | INR design/train for video compression (2506.24127) | **CONFIRM-NOT-LEVER** (recipe half) + **WATCH** (hypernet half = #211 skeleton) — prior ledger verdict UNCHANGED | — | LOW (already ledgered) |
| 8 | Principled neural operators (2506.10973 / NMI s42256-026-01267-z) | **FRAMING** (weak): resolution-invariance recipe; our coord-INR already discretization-free by construction | #211 | LOW |
| 4 | The Well (Polymathic, NeurIPS 2024) | **FRAMING** (weak, corpus-gated with #211): existence proof that operators learn advection/reaction-diffusion; NOT-APPLICABLE as data for the single-clip contest objective | #211 | LOW |
| 13 | 3D Gaussian volume compression (2607.01164) | **CONFIRM-NOT-LEVER** (generators-not-boundaries = our MEASURED Laguerre result L-v8/#284; densification ≈ margin-saliency routing #141) | — | — |
| 11 | Task-aware encoder control (2404.04848) | **CONFIRM-NOT-LEVER** (encoder-side freedom is where task gains live = our compress-time asymmetry, taken further) | — | — |
| 12 | Scalable human+machine coding (2307.08978) | **CONFIRM-NOT-LEVER** (machine layer needs fewer bits; we take that to the limit — no human layer at all) | — | — |
| 10 | SMC++ (2406.04765) | **CONFIRM-NOT-LEVER** (generic-semantic objective; already the v2-originality second flank — cite in writeup, ledgered) | — | — |
| 9 | CDRE embedding compression distortion (2503.21469) | **NOT-APPLICABLE** (mechanism = modify the downstream task model; our scorer is FROZEN/untouchable) | — | — |

---

## Per-paper signal (2–3 sentences each)

### 1 · Adjoint-based Neural Regulator for Real-Time Optimal Control (arXiv 2606.16303) — FRAMING → costate leg
MEASURED-by-paper mechanics (full-text): a **CoNN maps state → a HORIZON of costates**
(n_p=30 steps), trained **self-supervised by backpropagating the ORIGINAL objective through the
rollout** (stage cost + terminal cost + L1 costate regularizer — NOT an adjoint-ODE residual, NOT
supervised on solved trajectories); control recovered **closed-form** per step by pointwise
Hamiltonian minimization (û = −½R⁻¹gᵀλ̂); constraints handled **only at act-time** by a convex
QP projection (input bounds + CBF), unconstrained during training. Claims NMPC-parity at >100×
lower cost and near-invariant OOD behavior vs RL.
**Transfer to #247 (route BY NOTE — Einstein's files untouched):** three structural datums for
the costate crown: (a) predict a **horizon** of λ, not an instantaneous λ — schedule foresight
(our curriculum/τ-ladder decisions are horizon decisions); (b) actuation = **closed-form pointwise
minimization given λ** — DECIDE stays trivially cheap if λ is what's modeled; (c) **feasibility
(sealed-config invariants, budget, containment) belongs in an act-time projection, NOT inside
λ-learning** — matches our governor/actuation-boundary design, now with a literature anchor.
SPECULATIVE-UNTIL-PROBED lever: an amortized λ-net over run-telemetry state (their rollouts are
cheap; ours are epochs — the amortization economics differ; probe only if #247's heuristic
marginal-ΔS/byte proves insufficient).

### 2 · How to Design and Train Your INR for Video Compression (arXiv 2506.24127) — CONFIRM-NOT-LEVER / WATCH (ledger verdict unchanged)
Deepened read: RNeRV = +1.27% PSNR recipe tuning at equal 300-epoch budget; hypernet
weight-masking gives variable-rate (+1.7% PSNR/MS-SSIM @0.037bpp) and +0.4% params → +2.5/2.7%.
All PSNR-objective, NeRV-family = ancestor-vehicle class (L18: lessons not numbers). Prior ledger
verdict STANDS: recipe half = ancestor-lessons only; **hypernet-predicts-INR-weights half =
closest published skeleton for #211 amortized meta-init, WATCH, corpus-gated**. No new lever.

### 3 · Model-Aware Rate–Distortion Limits for Task-Oriented Source Coding (arXiv 2602.12866) — FLOOR-CONFIRM
Prior ledger verdict (FRAMING-that-proves-the-direction, task-RD ≺ reconstruction-RD) stands; no
computable BA-variant procedure surfaced (abstract-level; INFERRED). NEW harvest datum: their
model-aware bounds account for **deployed-model imperfection** — our S_floor 0.118 is computed
against the ACTUAL frozen scorer, i.e. already the "deployed model" regime their critique targets;
our floor methodology is model-aware by construction. Their empirical finding "**current learned
TOSC schemes operate far from these limits — transmitter-side complexity is the key bottleneck**"
is direct evidence FOR our regime: we have unbounded compress-time compute (rule-118 asymmetry),
so the classical reason for a large gap-to-floor does not bind us. Cite in any floor/gap analysis.

### 4 · The Well (Polymathic AI, NeurIPS 2024) — FRAMING (weak), corpus-gated
16 datasets/15TB incl. Gray-Scott reaction-diffusion, planetary shallow-water, advection-dominated
flows; FNO-class baselines. NOT-APPLICABLE as data (we train per-clip against a frozen scorer, not
against PDE-solution supervision). The operator-learning framing (clip → witness-INR weights as an
operator) is an existence proof FOR #211's amortized meta-init — but #211 is corpus-gated and the
contest objective is single-clip; nothing here un-gates it. The reaction-diffusion ↔ island-birth
analogy supplies no technique beyond what our ladder-homotopy already does (LIVE-confirmed working,
FEED-v9-harvest-1: Movable islands born 0.9998→0.0073). SPECULATIVE beyond that.

### 5 · Self-Supervised Amortized Neural Operators for Optimal Control: Scaling Laws (arXiv 2512.24897) — FRAMING (quantitative)
Amortizes condition→optimal-control maps with NO precomputed optimal trajectories (trains through
the objective — same trick as paper 1); derives generalization-error scaling in the problem's
**intrinsic dimension** + control-function regularity; works well ONLY in low-intrinsic-dim
regimes. Transfer: our d_seg manifold is doubly-measured **rank-8** (mod-17-19 Whitney) ⇒ both the
#211 amortization and any amortized-costate idea sit in the paper's FAVORABLE regime — a
literature-side viability check we didn't have. Their operator-in-MPC-loop composition mirrors our
costate-in-training-loop (#247) shape.

### 6 · Constrained Neural Parameterization for Optimization in Function Spaces (arXiv 2606.00855) — FRAMING
Enforces constraints **by architecture** (parameterization whose image lies in the admissible set,
asymptotically dense) instead of penalties/multipliers — for polyhedral, pointwise, and separable
PDE constraints. SPECULATIVE-UNTIL-PROBED transfer: an eikonal-by-construction level-set
parametrization would delete our eikonal penalty (0.01) + its tuning; but our penalty is small,
already stable, and the FALSIFIED_MECHANISM edge on eikonal-CFL (FEED-06g) says this axis is not
currently binding — LOW EV, note only. The philosophical rhyme: our gauge-zero-mode carriers
(per-boundary integration constants) are ALREADY constraint-by-parametrization.

### 7 · Gradient-free training of neural ODEs via Ensemble Kalman Inversion (arXiv 2307.07882) — **LEVER → #396**
EKI trains by forward-passes only (ensemble + Kalman update, Tikhonov-regularized inverse-problem
form); competitive with gradient training on low-dim problems; **known limitation: ensemble size
scales with parameter dimension** (workshop paper, ICML'23 NF-LCD). This limitation is exactly our
opportunity: the V9 **gauge-zero-mode constants** (per-boundary integration constants, ~10–10²
scalars — L87: 63–76% of the pose-HF band was a ~24-byte-correctable constant bias; d_gauge is the
whole residual at convergence) are a LOW-DIM, **non-differentiable-objective** DOF. Gradient
descent only ever sees the τ-relaxed surrogate; EKI can fit these constants against the **TRUE
argmax flip-count through R** (the exact per-pair d_seg), directly closing the surrogate↔exact gap
on the cheapest DOF — and Maslov err≤τ·ln5 (L75) says exactly where the surrogate is loose.
**Concrete $0 probe (routes into #396 MC-finisher, informs its algorithm choice):** take any
trained checkpoint → expose the phase/gauge constants (the #425 phase-carrier residual values are
the natural first target) → EKI ensemble ~32–64, objective = exact n96 flip-count through frozen
CPU-torch R (chunked per #240; no SegNet-forward concurrency with pid 88030 — run AFTER the live
run or on cached GT margins) → accept if exact d_seg drops below the gradient-converged value at
equal bytes; scale n600 only on a positive. Ensemble-Kalman ≻ naive MC on sample-efficiency at
this dimension — that is the datum #396 gains.

### 8 · Principled Approaches for Extending Neural Architectures to Function Spaces (arXiv 2506.10973 / Nat. Mach. Intell.) — FRAMING (weak)
NVIDIA/Caltech recipe for lifting transformers/CNNs to discretization-invariant operators
(positional encodings + normalization must be resolution-consistent). Our witness is a coordinate
INR — resolution-free by construction; R is fixed. Relevant only if #211 un-gates and we build a
clip→weights hypernet (then: build it as a neural operator per this recipe, not a fixed-grid CNN).
Ledger note only.

### 9 · CDRE — Embedding Compression Distortion in VCM (arXiv 2503.21469) — NOT-APPLICABLE
Extracts compression-distortion features and **progressively embeds them into the downstream
model** so the task net compensates for codec damage. Our downstream model is the FROZEN contest
scorer — modifying/augmenting it is forbidden (strict scorer rule). The dual insight (distortion
measured in the task's feature domain) is already our Fisher-metric/margin machinery, measured at
0.978. No reading survives for us.

### 10 · SMC++ (arXiv 2406.04765) — CONFIRM-NOT-LEVER (ledgered)
Masked-video-modeling semantic objective + non-semantic-entropy regularization in MVM token space;
generic-semantics, no fixed task. Already the second flank of the v2 originality neighbor map
(objective axis; MUST-CITE in the v2 writeup — prior ledger row). Our setting has the EXACT
deployed task; generic-semantic preservation is strictly weaker signal. Nothing new.

### 11 · Task-Aware Encoder Control for Deep Video Compression (arXiv 2404.04848) — CONFIRM-NOT-LEVER
Steers ONE pretrained codec per-task purely encoder-side (mode-prediction + GoP selection; ~25%
bitrate saved) with an unchanged decoder. Confirms the asymmetry our whole design exploits —
task-adaptation lives at the encoder/compress side; we hold that asymmetry in its limit form
(per-clip overfit, free generic decoder, counted statistic). No mechanism transfers (their control
DOF are codec modes we don't have).

### 12 · Learned Scalable Video Coding for Humans and Machines (arXiv 2307.08978) — CONFIRM-NOT-LEVER
Conditional-coding base layer for the machine task + enhancement layer for humans; base layer
beats SOTA codecs at the machine task. Confirms machine-task bits ≪ human-fidelity bits — our
witness is the degenerate optimum of their hierarchy (zero human layer). No lever.

### 13 · Efficient Compression of Volumes via Learned 3D Gaussian Representation (arXiv 2607.01164) — CONFIRM-NOT-LEVER
Explicit 3D-Gaussian primitives as scalar-field codec + **sampling-error-based densification** +
CUDA sampling; beats INR baselines on unstructured volumes by dropping mesh storage. This is the
soft/volumetric cousin of our MEASURED L-v8 result (argmax = Laguerre power diagram → store
GENERATORS not boundaries, #284), and their error-driven densification ≈ our margin-saliency
KKT waterfill (#141). We already hold the sharper (task-exact, tropical) version; GaussianQuant
is already a ledger WATCH. Nothing new transfers.

---

## Routing (tracked, not orphaned)
- **LEVER #7 → task #396 (MC-finisher):** note appended via DAG FEED-paper-harvest (below) —
  EKI-over-gauge-constants against exact argmax d_seg, probe recipe above. #396 remains the owner;
  no new task created (anti-duplication).
- **FRAMING #1 + #5 → costate leg (#247 / Einstein):** routed BY NOTE in this memo + the DAG FEED
  (horizon-λ · closed-form pointwise actuation · act-time constraint projection · low-intrinsic-dim
  viability). Einstein's files (`einstein_*`, master-system module, #223/#299) untouched.
- **FLOOR #3:** ledger row updated (transmitter-side-bottleneck datum) — cite in gap-to-floor
  analyses.
- **Papers-checked ledger:** all 13 appended/refreshed in
  `reference_papers_checked_not_relevant_or_watch_item_ledger_20260701.md` (anti-re-research).

## Triality legs (honest)
- **DAG** = FEED-paper-harvest (appended this pass).
- **DSL** = N/A — no trainer-flag lever emerged (the one LEVER is a post-training finisher
  algorithm choice inside #396's existing scope, not a curriculum/loss lever).
- **equations** = N/A — no paper supplies a law groundable on our measurements (paper 5's
  intrinsic-dim scaling law is literature-side; it would enter only with our own measured anchor).

**Pointer 0.19108282 [contest-CPU] UNMOVED.**
