# Fast-witness-training — external OSS/paper survey (MIN-WALL-CLOCK to a converged witness)

- **UTC:** 20260712T135024Z · **git HEAD:** 67fa2c9512
- **Axis / authority:** `[external paper claim]` / `[advisory only]` — NO score claim; **pointer 0.19108282
  [contest-CPU] UNMOVED** (SoT `.omx/state/canonical_frontier_pointer.json`). This is a MEANS (a
  campaign accelerator). Every external number is the paper's own benchmark; nothing transfers to our
  score without a through-R n600 measured row on OUR vehicle (ancestor-vehicle discipline).
- **Scope (operator, 2026-07-12):** external papers + OSS on MINIMUM-WALL-CLOCK training/solving of the
  task-space level-set / SDF / SegNet-argmax witness — the SPEED / amortization / SOLVE angle, **not**
  representation math.
- **NO-FAKE:** every arXiv id + repo below was returned by a live web search this session; the ELM-INR
  paper (2602.07603) was additionally fetched and its method read. Uncertainty flagged inline. Where a
  technique is already covered by #305 / #443 / the distillation survey, it is marked **ALREADY KNOWN —
  SKIP** and not re-reported.

---

## 0. THE REFRAME THAT REORDERS THE WHOLE SURVEY (read first)

**MEASURED (per `compute_facet_mlx_metal_profile_…_20260702` + #443 kernel sweep):**
~**95% of the #205 per-epoch wall-clock is the frozen SegNet + trunk forward+backward**; the SegNet
backward is **already 16.9×** via the built, default-ON Metal grouped-conv kernel
(`local_acceleration/metal_grouped_conv_backward.py` — correctness-critical, native MLX grouped-conv
backward is numerically WRONG at cos ~0.025). fp32 is both the authority AND the fast path on Apple GPU
(fp16/bf16 measured SLOWER + worse-gradient). The coordinate-INR/trunk render is only ~**26%** of the
step and its basis is *deliberately* the directional-Fourier `--self-orient` basis (the measured −48%
d_seg lever), not a generic MLP.

**Consequence — the honest fit test every candidate must pass:** a speed technique helps OUR wall-clock
**only** if it (a) **cuts the number of scorer-gradient calls to converge** (fewer epochs / warm-start /
meta-init / better optimizer), (b) **makes each scorer fwd+bwd cheaper** (distilled surrogate; already a
line — see §D), or (c) **replaces a trained block with a closed-form/iterative SOLVE** (#341/#342/#396).

**A technique that only speeds the coordinate-field forward (Instant-NGP, hash grids, tiny-cuda-nn,
TensoRF, K-planes) attacks the ~26% slice that is already fast and already directional** — it is capped
at a fraction of a 26% slice, needs an MLX port (all are CUDA/tiny-cuda-nn), and MLX has no hash-grid
primitive. **This is a structural MISFIT for our bottleneck** and is ranked accordingly (§C/§D), not
because the papers are weak but because they accelerate the wrong part of *our* loop. The lever that
would actually move our wall-clock is **fewer epochs** (each epoch = 600 scorer fwd+bwd) — so the
epochs-reducing tricks (§A) dominate.

---

## RANKED FINDINGS

Ranking key = (wall-clock win on OUR bottleneck) × (ease-of-adoption) × (fit to through-R + frozen
dual-scorer + argmax target).

### TIER A — genuinely new, adoptable, and hit the dominant cost (fewer scorer calls / a real solve)

#### A1. FreSh — Frequency-Shifting init for accelerated INR learning  — **RANK #1**
- **Paper/repo:** arXiv **2410.05050** (OpenReview `zMjjzXxS64`); code released (github, "FreSh").
- **Mechanism (one line):** the output spectrum of an *untrained* INR predicts what it will learn well;
  FreSh does a one-time, backprop-free **init-time embedding/frequency selection** so the model's
  init spectrum matches the target signal's spectrum → escapes spectral bias faster.
- **Wall-clock win class:** **fewer epochs to a fixed quality** (init-only, cost ≈ one forward pass +
  an FFT). Because each epoch = 600 scorer fwd+bwd, *any* epochs-cut is a direct proportional
  wall-clock cut on the 95% slice — this is the rare INR trick that helps US.
- **Quality tradeoff:** none claimed (it's an init, not a constraint); it reaches the same/better
  quality sooner. Risk = it targets the trunk's spectral init, which interacts with our
  directional-Fourier basis — must be applied to the *residual* frequency content the basis doesn't
  already cover (measured along-tangent 3.2× deficit is the natural target).
- **MLX-portable?** YES — init-only, an FFT of the target argmax/margin field + a bias-init pick; no
  hot-path kernel. Trivial port.
- **Maps onto us:** compute the target-spectrum of the frozen-SegNet margin field once (n600), pick the
  step/hosc bias-init that aligns to it, launch. $0 probe: measure epochs-to-CE-floor with/without.
  Companion/near-duplicate: **Fourier Reparameterized Training (FR, CVPR 2024, 2401.07402, code)** —
  reparameterize the first layer over a learnable Fourier basis so low- and high-freq error decay at
  matched speed; overlaps our fixed Fourier features but the *reparam-during-training* variant is
  untried. Treat FR as the B-tier sibling arm of the same A/B.

#### A2. Meta-learned init (MetaSDF + hypernet-INR) → OSS makes #211 cheap  — **RANK #2**
- **Papers/repos:** **MetaSDF** (Sitzmann et al. 2020, `vsitzmann/awesome-implicit-representations`) —
  gradient-based meta-learning of an SDF-INR init, **order-of-magnitude faster test-time fitting** vs
  auto-decoder; **"Fast Medical Shape Reconstruction via Meta-learned INR"** (2409.07100) — SDF-target,
  same few-step-fit claim; **hypernet-predicts-INR-weights for video** (2506.24127, code) — the closest
  shipping skeleton for `H_ψ(clip)→θ₀`. (2506.24127 is **ALREADY LEDGERED** as the #211 skeleton in
  the paper-harvest ledger — cited here only to complete the amortized-init map.)
- **Mechanism (one line):** learn a weight **initialization** (MAML/Reptile inner-outer, or a
  hypernetwork) from a corpus of (signal, θ*) pairs so a NEW signal converges in a handful of inner
  steps instead of thousands.
- **Wall-clock win class:** **largest possible on our bottleneck** — cuts epochs-to-converge by the
  meta-init factor (order-of-magnitude in the SDF papers), i.e. order-of-magnitude fewer scorer calls.
- **Quality tradeoff:** meta-init trades some peak quality for speed *unless* refined; we would meta-init
  then finish (the #211 "amortized-init + adaptive refine" plan). Corpus-GATED — but **we already
  generate (clip-slice, θ*) pairs every witness run**; the corpus is a byproduct, not a new cost.
- **MLX-portable?** MAML/Reptile inner-loop is plain autodiff (portable); a hypernet is a small MLX
  module. No exotic kernel.
- **Maps onto us:** this IS open task **#211**. External status: MetaSDF proves the SDF-target case;
  2506.24127 ships the video-INR hypernet code to adapt. **Actionable:** stand up the (clip, θ*) corpus
  from existing run checkpoints, meta-init the trunk, measure epochs-to-CE-floor. Highest ceiling of any
  item here; higher build cost than A1.

#### A3. ELM-INR — backprop-free closed-form INR fit (solve-don't-train)  — **RANK #3**
- **Paper/repo:** arXiv **2602.07603** ("Escaping Spectral Bias without Backpropagation: Fast INR with
  Extreme Learning Machines"), *fetched*. **No code released** (as of fetch) → needs a port.
- **Mechanism (one line):** freeze hidden weights (random Fourier features), solve **only the output
  layer in closed form** per subdomain — `α* = (HᵀH)⁻¹ Hᵀy` (linear least-squares) — then blend
  subdomains by partition-of-unity; a Barron-enhanced adaptive mesh equalizes spectral complexity.
  Complexity `O(Sm² + Nm³)` vs `O(T·L·S·m²)` for backprop — the ×(thousands of iterations) factor is
  gone; "high-quality reconstruction in a few seconds," ~2× PSNR vs iterative INRs `[external, PSNR]`.
- **Wall-clock win class:** **replaces the terminal head-fit leg with a one-shot solve.** Directly
  extends our **#341 terminal-head Gauss-Newton / damped-Newton-CG** (GO-issued) and the **#342
  solve-don't-train inventory** (row 1). The exact-argmax d_seg is non-differentiable, so this can't fit
  the *whole* witness against the true metric — but on a *smoothed/CE-surrogate* target it can solve the
  head (and, subdomain-wise, the per-boundary charts) instead of descending them.
- **Quality tradeoff:** PSNR-objective paper (ancestor-discipline: numbers don't transfer); the transfer
  is the MECHANISM (random-features + LS-head + partition-of-unity), which is exactly our
  affine-head-is-solvable finding (#341 head chart near-quadratic, LM ρ 0.847/0.868). Our FiLM path is
  non-affine (stays trained) — so this solves the *head*, not the trunk.
- **MLX-portable?** Linear solve is `mx.linalg`/CPU-lstsq; partition-of-unity is elementwise. Portable;
  needs a from-scratch build (no repo).
- **Maps onto us:** fold as the closed-form seed for the #341 head finisher — solve the affine head by
  random-feature LS (seconds), then optionally 1–2 GN polish steps, verified through R. Composes with
  the existing #342 row-1 GO. Modest total-wall-clock win (the head is a small slice of the run) but it
  **removes an entire terminal training leg** and is the cleanest external match for our solve line.

### TIER B — new + adoptable, but attack the smaller ~26% trunk slice (bank as arms, not headline)

#### B1. Factorized grids as fast-trainable carriers (K-planes / TensoRF / DeepSDF)
- **ALREADY SURVEYED** for the RATE/carrier axis in `factorized_4d_kplanes_observability_20260630`. The
  SPEED note not yet A/B'd: per **2506.11139 (grids-beat-INRs on dense signals, NeurIPS 2025, already in
  #305)**, a regularized grid **trains faster** than an INR at matched params for the *smooth-bulk*
  classes (Road/Undrivable/hood interiors) — while our binary-contour target keeps the INR only on the
  boundary annulus. **Actionable = #305 probe P1** (hybrid: grid arm for smooth bulk, coord-INR for the
  annulus/lane manifold), which cuts trunk epochs on the bulk. Cross-ref, do not re-survey. Fit: MED
  (bulk is deep-margin/cheap anyway; the grid still needs the step/curvelet decoder at the boundary).

#### B2. Dynamical INRs / gradient-diversity accelerators (DINR 2511.21787; NOWS 2511.02481)
- DINR claims "convergence in fewer epochs" via enhanced gradient diversity; NOWS = neural-operator
  warm-starts that cut iterative-solver iterations. Both are epochs-reducers in principle, but DINR is a
  representation change (2511.21787 already flagged WATCH in #305) and NOWS targets PDE solvers, not our
  scorer-in-the-loop. Fit: LOW-MED. WATCH; no build until A1/A2 measured.

### TIER C — fast level-set / active-contour SOLVERS for the geometry TERMS (misfit for our objective)

#### C1. AOS / narrow-band / multigrid Chan-Vese + GPU RSF level-set (2404.02813)
- **Papers:** AOS (additive operator splitting — splits 2D into two tridiagonal 1D solves, Thomas
  algorithm); multigrid Chan-Vese ("many magnitudes faster than AOS"); **GPU-Accelerated RSF Level-Set
  Evolution** (arXiv 2404.02813, 2024). Classical + modern fast solvers for level-set *evolution*.
- **Honest verdict — LOW fit:** our **binding objective is the through-R frozen-SegNet argmax loss**, not
  a Chan-Vese region-energy; these solvers evolve a level set toward an *intensity*-defined contour,
  which is a DIFFERENT functional. Our eikonal (0.01) + length (0.001) terms — the only parts these could
  replace — are a **tiny fraction of the loss and NOT the bottleneck** (95% is the scorer). Replacing
  gradient-descent on the geometry terms with a solver saves ≪1% of wall-clock. **Related and already
  covered:** HotSpot screened-Poisson/heat-method SDF solve + StEik/ViscoReg eikonal-stability cures are
  in `litsweep_training_dynamics_control_20260705` (§D). Cite as spare-tire for eikonal STABILITY, not as
  a speed lever. WATCH only.

### TIER D — ALREADY KNOWN — SKIP (verified against our own memos; listed so a future agent doesn't re-add)

- **Instant-NGP / multiresolution hash grids / tiny-cuda-nn / factorized feature encodings** — accelerate
  the ~26% trunk forward; **CUDA/tiny-cuda-nn only** (no MLX hash-grid primitive → full port); our trunk
  is already fast and its basis is the deliberate directional-Fourier −48% lever. **Structural MISFIT
  per §0.** (K-planes/TensoRF captured as B1 for their carrier value.)
- **mx.compile / mx.fast fused kernels / Metal custom kernels** — `#443` kernel sweep is the SoT: the
  grouped-conv backward (16.9×) is banked + default-ON; `mx.compile` on the seg-only closure is a **~5%
  un-wired fruit (F1)**; MLX ecosystem web-scan (v0.31.x, WWDC25) returned **nothing past #443**. SKIP —
  the one real fruit (F1) is already named in #443.
- **Distilled surrogate scorer for the training gradient / feature-space perceptual distillation /
  cheaper-Jacobian** — covered by `distillation_sota_survey_20260711` + the **#426 costate λ-organ** +
  **#247**. The framing there is sharper than anything the web returned: our teacher is white-box, frozen,
  differentiable-a.e.; the non-differentiability is in the METRIC (argmax), fixed by margin/τ-softmax
  relaxation (margin↔Fisher 0.978), not by a learned surrogate. SKIP.
- **EKI / gradient-free terminal solve over gauge constants** — `#396` (2307.07882), already a LEVER in
  the paper-harvest triage. SKIP.
- **StEik / ViscoReg / HotSpot / Edge-of-Stability / PINN-balancing / optimal-control-of-training / GNC**
  — `litsweep_training_dynamics_control_20260705` (the #305 dynamics sibling). SKIP.
- **Gradient checkpointing** — irrelevant: the frozen scorer isn't activation-memory-bound at our
  per-pair batch (the memory constraint is the 41 GB `cf_mx_cache`, addressed by in-place rebuild, #443
  F4). No wall-clock fruit. SKIP.

---

## THE ONE-PARAGRAPH ANSWER (what to actually do)

Our wall-clock is scorer-fwd+bwd-bound and that kernel is already banked at 16.9×, so **the only large
lever left is FEWER EPOCHS**, and the two cheapest external ways to buy them are **(A1) FreSh
spectral-init** ($0, init-only, MLX-trivial — measure epochs-to-CE-floor with/without on the next run)
and **(A2) meta-learned init = task #211**, for which MetaSDF (SDF-target, order-of-magnitude test-time
speedup) and the 2506.24127 video-INR hypernet code de-risk the build against a corpus we already
generate. **(A3) ELM-INR closed-form head solve** is the best external match for the solve line — fold it
as the LS seed for the already-GO'd #341 head finisher (removes the terminal training leg). Everything
else (hash grids, K-planes, fast Chan-Vese, distilled surrogates, mx.compile) either targets the fast
26% slice, needs a port for a fraction-of-26% win, or is already inventoried in #443 / #305 / the
distillation survey / #396. **Rank to fire:** A1 (this run) → A3 (fold into #341) → A2 (stand up the
corpus). All MEANS; the pointer moves only through a byte-closed n600 `upstream/evaluate.py` row.

---

*Sources (live web, this session):* FreSh arXiv:2410.05050 (OpenReview zMjjzXxS64) · Fourier
Reparameterized Training arXiv:2401.07402 (CVPR 2024) · MetaSDF (Sitzmann 2020, vsitzmann/
awesome-implicit-representations) · Fast Medical Shape Reconstruction via Meta-learned INR
arXiv:2409.07100 · hypernet-INR-for-video arXiv:2506.24127 · ELM-INR arXiv:2602.07603 (fetched) ·
DINR arXiv:2511.21787 · NOWS arXiv:2511.02481 · GPU RSF level-set arXiv:2404.02813 · grids-beat-INRs
arXiv:2506.11139 · MLX WWDC25/v0.31.x (developer.apple.com/videos/play/wwdc2025/315). Internal anchors:
`compute_facet_mlx_metal_profile_…_20260702`, `kernel_stack_sweep_443_20260711`,
`solve_dont_train_inventory_20260709`, `litsweep_training_dynamics_control_20260705`,
`litsweep_representation_taskspace_20260705`, `distillation_sota_survey_20260711`,
`factorized_4d_kplanes_observability_20260630`, `amortized_operator_pontryagin_loop_cluster_20260711`.
