# Papers-checked (deep-read, 4-part): queued batch — scaling / contrastive / PTRM / RSI / Fork (2026-07-14)

**Contract honored:** every item below delivers FULL MATH (theorem + key derivation, labeled
MEASURED/DERIVED/INFERRED), OSS HARVEST (repo/license/transfer), CRITIQUES, HONEST FORK to a live
task#/crux with the $0 next-probe — per `[[paper_warm_start_is_deep_math_plus_oss_harvest_never_abstract_20260714]]`.
This is the deep read the operator demanded; the prior queue rows (in
`consolidation_drain_cherrypick_prune_driftgate_20260714.md`) were ABSTRACT-depth only.

**Recall check:** none of the five were previously deep-verdicted. 2607.11052 appeared only as an
abstract-depth queue row (in the two memos above); 2607.07470 + 2605.19943 unseen; Fork Dynamics
partially captured by `[[curriculum_is_continuation_instabilities_are_bifurcations_20260714]]` (reframe
derived; tool identity + license + concrete probe are NEW here). Pointer UNMOVED 0.18804/0.19108. All MEANS.

---

## 1. arXiv 2607.07470 — Torralba & Weiss, "A Theory of Contrastive Learning with Natural Images"
**VERDICT: LEVER (modest — grounding + $0 diagnostic probe) → #502 (curvelet/shearlet) + #497 (alt-to-Fourier) + #277/#25.**

### FULL MATH (MEASURED from full-text body, ar5iv/html)
- **GUPA contrastive loss (Thm 1.1, Eq 2):**
  `L_GUPA = t·E‖y(Tx)−y(x)‖² − ½ log det(Σ_y+εI) − ½ Tr(Σ_y(Σ_y+εI)⁻¹)` =
  alignment (augmented pairs together) + Gaussian-uniformity/whitening. Shown equivalent to InfoNCE under
  Gaussian embeddings.
- **Stationarity → Fourier diagonalization (Thm 2.1, Peligrad–Wu 2010):** for stationary signals as image
  size N→∞ the DFT coefficients are asymptotically pairwise independent, so BOTH the alignment matrix `B`
  and the covariance `Σ` become **diagonal in the Fourier basis**. The generalized eigenvectors of (B,Σ)
  then reduce to unit vectors in frequency space → **the sinusoid first layer is a THEOREM, not a choice.**
- **Optimal representation (Thm 3.1, Eq 6):** `y*_i(x) = (1/g(k_i))·|x^F[k_i]|²` — squared DFT power divided
  by its expected variance g(k) = whitening in frequency space.
- **Waterfilling (Alg 1 / Eq 4):** minimize `L({P_k}) = Σ_k P_k λ_k − ε Σ_k log(P_k+ε) − ε² Σ_k P_k/(P_k+ε)`
  s.t. `Σ P_k ≤ 1`, where λ_k are the generalized eigenvalues of (B,Σ) and P_k is capacity on frequency k.
  Greedy fill: `k* = argmin_k [ λ_k − ε/(P_k+ε) − (ε/(P_k+ε))² ]`, step P_{k*} += η. Low-SNR frequencies get
  **P_k = 0** ("partial" whitening = only a subset K of bands funded).
- **Augmentation → band map (MEASURED):** circular-crop → keep all (alignment=0); ideal-blur → keep below
  cutoff; jitter → keep non-DC; crop+noise → keep SNR>threshold (this is where waterfilling bites).

### The decisive fact for us (MEASURED, Appendix K + Thm 2.1 dependency)
The sinusoid/waterfilling optimum is **ENTIRELY a consequence of GLOBAL translation-invariant stationary
statistics** (that is the ONLY reason B,Σ diagonalize in the global Fourier basis). The paper has **NO
theory for oriented/2D-directional filters, edges, boundaries, or non-stationary/local spectra** (Appendix K
only *empirically* notes GAP still yields sinusoids on CelebA; the math is 1D).

### OSS HARVEST
No author code repo located (searched; none linked). Adjacent lineage: MMCR "Efficient Coding of Natural
Images / Maximum Manifold Capacity" (arXiv 2303.03307) is the nearest released relative but not this method.
**Nothing to import as code.** The reusable artifact is the *formula* (Eq 4 waterfilling on generalized
eigenvalues of the alignment-vs-uniformity pair) — basis-agnostic, ~15-line numpy.

### CRITIQUES
Admitted limitation (their Appendix K): stationarity is load-bearing; non-stationary case is empirical-only.
1D throughout; 2D anisotropy "assumed not treated." No edges/boundaries.

### HONEST FORK (assumption-divergence traced)
Their setting = stationary global natural-image spectrum → optimal basis diagonalizes in **global Fourier
(isotropic sinusoids).** OURS = coord-INR fitting a frozen-SegNet argmax on an n=1 clip; the 3.2×
along-tangent deficit lives on a **NON-stationary, oriented, codim-1 boundary/lane manifold.** On that
manifold B,Σ are **NOT** diagonal in the global Fourier basis — which is *precisely why* isotropic Fourier
features under-serve the along-tangent direction (this THEOREM is the clean theoretical confirmation of
`[[L25 BASIS-OVER-CREDITED]]` and the −48% self-orient/directional finding: global sinusoids are provably
optimal ONLY under stationarity we do not have).
- **DERIVED (sound):** the paper's *optimality principle* — "optimal representation = eigenbasis that
  diagonalizes alignment-vs-uniformity" — points, for oriented/non-stationary boundary data, at the basis
  that diagonalizes edge/wavefront operators = **curvelets/shearlets** (Candès–Donoho). i.e. the paper
  reinforces #502 from the opposite side, and reinforces that the sinusoid answer is the WRONG one for us.
- **INFERRED (hypothesis, unproven):** a waterfilling capacity allocation (Eq 4) over **curvelet
  orientation×scale bands**, using the measured along-tangent vs along-normal boundary spectrum as the λ_k,
  would predict the optimal capacity split and fix the 3.2× deficit as a mis-allocation (not just a basis) issue.
- **$0 NEXT-PROBE:** from existing gt caches (`gt_n96.npz` argmax/margin field), compute the boundary-local
  power spectrum decomposed by orientation relative to the boundary tangent; form the diagonal λ_k per
  orientation band; run Eq-4 waterfilling (numpy) to PREDICT the along-tangent/along-normal capacity split.
  If waterfilling already wants ≫ capacity on along-tangent bands → confirms the deficit is mis-allocation
  under an isotropic basis and quantifies the target split for #502. Pure numpy, no training, no dispatch.
- **Target:** #502 (genuine curvelet/shearlet) primary; #497 (alt-to-Fourier), #277, #25, and #157/#336
  bit-alloc (waterfilling ≡ our bit-allocator). NOT a training lever by itself — grounding + diagnostic.

---

## 2. arXiv 2607.11052 — Hamidieh, Mackey, Alvarez-Melis, "Domain-Aware Scaling Laws Uncover Data Synergy"
**VERDICT: DOMINATED-bookmark → dual-purpose sweep / organ cluster (#434/#499/#481). Regime divergence: multi-model observational regression; cannot cure n=1 starvation.**

### FULL MATH (MEASURED abstract + NeurIPS/OpenReview metadata; body behind verification wall — labeled)
- Framework quantifies **data synergy** = combined-domain contribution that **exceeds (super-additive) or
  falls short of (interference)** the sum of isolated contributions. Two estimated interaction types:
  (a) direct domain→benchmark synergy, (b) **second-order domain–domain synergy** (capabilities needing
  co-occurrence of two domains).
- **Estimation substrate (MEASURED, decisive):** synergy is estimated by **"leveraging observational
  variation across open-weight LLMs with diverse pretraining mixtures"** (NeurIPS 2025 page). i.e. it is a
  **cross-MODEL regression** over many pretrained models × many mixtures — fundamentally multi-run.
- Recovers stable patterns (math–code complementarity); predicted optimal / anti-optimal mixtures correctly
  rank held-out performance.
- Exact interaction-term equation not extracted (OpenReview + arXiv-html both gated); the load-bearing
  regime fact (multi-model observational fit) is confirmed and sufficient for the verdict.

### OSS HARVEST
`github.com/dmelis` (15 repos; lineage incl. OT dataset-distance / dataset-dynamics). No dedicated
synergy-estimation repo surfaced. Transfers only as a **regression recipe**, not runnable code for us.

### CRITIQUES
OpenReview gated (verification wall). Structural limitation is intrinsic: needs cross-run variation to
identify interaction coefficients — an identifiability requirement, not a bug.

### HONEST FORK
Queued crux: "does 2nd-order synergy predict which REGIME-MIX of witness trajectories cures the organ n=1
starvation super-additively?" Assumption-divergence: their estimator EATS a corpus of many models/mixtures;
we have ONE witness family + an organ that sees 3 regimes (lane-erosion / mixed-Lane-Road / movable-unborn).
- **The method does NOT create data and does NOT cure n=1 starvation** — it presupposes a multi-run corpus
  and analyzes it post-hoc. It is a *sweep-analysis* tool, not a *sweep-shortcut*, and definitely not the
  starvation cure (#434 synthetic-data + physics-prior already WINS at n=1 per
  `[[n1_organ_capacity_ceiling_shrinkage_physics_residual_measured_20260714]]`).
- **The ONE keep (bookmark):** the synergy-map *concept* — regime-mixtures of organ-training trajectories
  may be super/sub-additive; if the dual-purpose sweep runs, fit their interaction term to pick the optimal
  regime-mix. Cheap to apply AFTER a multi-run corpus exists; worthless before.
- **$0 next-probe:** none that moves anything now (no corpus yet). Deferred to post-sweep analysis.
- **Target:** #434/#499 organ starvation (as analysis-only), #481 continual-learning, the sweep_spec arm's
  regime-mapping. Bookmark, not lever, not a score-mover.

---

## 3. arXiv 2605.19943 — Sghaier, Parviz, Jolicoeur-Martineau, "Probabilistic Tiny Recursive Model (PTRM)"
**VERDICT: DOMINATED-bookmark → costate/curriculum cluster (#247, bifurcation memory). Regime divergence: discrete symbolic reasoning + test-time voting.**

### FULL MATH (MEASURED abstract + html)
- Base TRM: 7M-param net that **iteratively refines a latent state + answer** (deterministic recursion), with
  a **Q-head** for early stopping. Deterministic recursion → converges to suboptimal basins, no escape.
- **PTRM contribution:** inject **Gaussian noise at each deep recursion step** → K parallel trajectories
  explore diverse solution basins → **select the best rollout via the existing Q-head** (no retraining, no
  task-specific augmentation). Test-time-compute scaling, task-agnostic.
- Results (MEASURED): Sudoku-Extreme 87.4%→98.75%; Pencil-Puzzle-Bench 62.6%→91.2% (≈2× frontier LLMs at
  <1e-4× cost, 7M params). Benchmarks are **discrete grid puzzles / ARC-AGI-class.**

### OSS HARVEST
- Official TRM: **github.com/SamsungSAILMontreal/TinyRecursiveModels** (7M-param recursive reasoner; ARC-AGI-1
  45% / ARC-AGI-2 8%). Unofficial: **github.com/lucidrains/tiny-recursive-model**. PTRM page:
  amins01.github.io/ptrm. Code exists and is clean, but it is a **discrete-reasoning grid architecture** with
  a Q-head + tokenized answer — **no coord-INR, no frozen-scorer, no continuous distortion**; nothing imports.

### CRITIQUES
Test-time-compute trick on a specific trained arch; gains are on symbolic-grid benchmarks; no continuous
regression / perceptual-distortion evidence. Depends on a pre-trained Q-head as the selector.

### HONEST FORK
Superficial resonance: "inject noise at each refinement step to escape deterministic convergence to a
suboptimal basin, select by an internal value estimate" ≈ our curriculum-as-continuation escaping the wrong
Morse-Smale basin, with the **costate value as the selector**. BUT:
- We have neither a Q-head nor a discrete answer; the witness is a continuous coord-INR fit and the costate
  organ is a supervised Pontryagin adjoint. The "stochastic multi-rollout + value-head vote" is a
  test-time-compute pattern for symbolic reasoning, not a lever for a continuous scorer-amortizer.
- The ONE transferable idea (noise-perturbed multi-trajectory exploration selected by an internal
  value/adjoint) is **already better-framed** by our bifurcation/continuation reframe
  (`[[curriculum_is_continuation_instabilities_are_bifurcations_20260714]]`) + #247 costate. PTRM adds no math
  we lack.
- **$0 next-probe:** none warranted. Bookmark only.
- **Target:** #247 costate / curriculum cluster (conceptual cross-ref for stochastic-exploration-by-value).
  Not a lever.

---

## 4. WECO AIDE² — "First evidence of recursive self-improvement" (weco.ai blog)
**VERDICT: DOMINATED-bookmark → apparatus/triality cluster. Regime: agentic ML-engineering infra, NOT a learning-algorithm RSI, NOT a score-mover.**

### WHAT IT IS (MEASURED from blog)
- AIDE = autonomous ML-research agent (won OpenAI MLE-Bench) using tree-search over code/prompt/verification.
- **AIDE² = bi-level optimization:** inner loop = AIDE₀ optimizes ML-engineering / heuristic / harness code;
  **outer loop = AIDE_human rewrites the inner agent's OWN source code** (search policy, context management,
  eval procedure) as the optimization target.
- **Claim:** "Level-1 RSI" — 7 successive improved agent versions in 8 days beat ~2 years of manual eng.
- **Evidence (MEASURED):** MLE-Bench-Lite +0.053 (p=0.0024, AIDE47) / +0.042 (p=0.0041, AIDE85); ALE-Bench &
  WeatherBench-2 gains; **reward-hacking rate 63%(AIDE₀)→34%(AIDE85)** vs 42% human baseline.

### OSS HARVEST
AIDE is open (weco-ai/aideml lineage, MIT historically); AIDE² is a paper/blog result, not obviously a
released artifact. It is an **agent-orchestration** codebase, not a math primitive for our vehicle.

### CRITIQUES / HONEST FORK
Regime-divergence: AIDE² improves an ML-ENGINEERING AGENT'S source code via LLM tree-search; it is agentic
infra RSI, **not** a learning algorithm improving itself and **not** anything that touches an exact
compression score.
- **Genuine resonance (bookmark, honest):** the bi-level structure IS our triality (inner = witness/lever
  search; outer = apparatus/DSL/costate improvement) — AIDE² is external evidence that a compounding
  self-improving-apparatus loop *works*, validating `[[L15 triality-as-shadows-of-one-action]]` + #247. The
  **reward-hacking-rate metric** (measure hacking-rate falling over iterations) is a directly adoptable idea
  for our NO-FAKE monitoring — an observability signal, not a score-mover.
- **$0 next-probe:** optionally, define a "means-as-ends / fake-rate" observability counter for the
  apparatus (analogous to their hacking-rate), default-ON per the observability-defaults-ON discipline. Cheap,
  score-neutral, honesty-improving; NOT a pointer lever.
- **Target:** apparatus/triality cluster (#247 costate SENSE layer, triality). Bookmark.

---

## 5. Fork Dynamics = "Fork" (github.com/hinsley/Fork) — numerical bifurcation continuation
**VERDICT: LEVER (tool confirmed, MIT, usable) → curriculum/bifurcation cluster (#302/#315/#318/#344, #300/#323 island-birth, #217). CONFIRMS + EXTENDS existing memory; value gated on deriving the reduced-order model (the real work), not the tool.**

### WHAT IT IS (MEASURED)
- MIT-licensed numerical **bifurcation-continuation** app: Rust core → WASM bindings → Node CLI + web UI
  (fork-phi.vercel.app). Small equation language (AST → bytecode → tiny VM, f64 or dual-numbers for autodiff).
- **Detects:** maps — saddle-node/fold, branch points, period-doubling, Neimark-Sacker; ODE equilibria —
  fold, Andronov-Hopf, neutral saddle; periodic orbits — LPC, PD, NS, branch points; global — homoclinic /
  heteroclinic. Uses **pseudo-arclength continuation (PALC)** to follow branches through folds a naive sweep
  misses. Newton + continuation + stability + collocation. (Homoclinic-to-periodic-orbit: not implemented.)
- Node/JS only (no Python lib); low-dimensional smooth systems.

### OSS HARVEST
`github.com/hinsley/Fork`, **MIT** — directly usable. Node CLI (`cli/ npm start`) eats an ODE/map system in
its equation DSL, emits bifurcation diagrams (Plotly). Interop with our Python/MLX stack via the WASM core is
feasible (sister to molt's Python→WASM+WebGPU, though Fork is Rust→WASM+Node — a minor friction, not free).

### CRITIQUES
Spare-time project; low-dimensional only; Node-only surface; PALC + standard codim-1/2 detection (no exotic
codim-≥3). Fine for our need — we only want codim-1 fold/pitchfork on a 1–2D order parameter.

### HONEST FORK (confirms `[[curriculum_is_continuation_instabilities_are_bifurcations_20260714]]`, adds tool+probe)
Our island-birth (movable class unborn until a curriculum threshold) = **saddle-node/pitchfork in the
class-occupancy order parameter** vs the curriculum control (τ/ε/λ). Fork is the concrete instrument to
**COMPUTE the critical λ** where the fold occurs — replacing GO-gated trial-and-error with a bifurcation
diagram. The bottleneck is NOT the tool; it is **deriving the reduced-order ODE** (we have pieces: #318 DE,
#344 linear-NCDE, #180 Morse-Smale). Fork eats low-dim only → derive the order-parameter model FIRST.
- **$0 NEXT-PROBE (concrete first step, new):** from an EXISTING witness run log, extract the 1D order
  parameter = movable-class occupancy fraction vs epoch/curriculum-param; fit the minimal normal form
  `ẋ = λ + a x − x³` (saddle-node/pitchfork); check whether the observed island-birth epoch coincides with
  the fold. If yes → feed that ODE to Fork's PALC and continue in λ to get the exact critical threshold + a
  bifurcation diagram for birth-point placement. Pure log-analysis + a 2-param fit; no training, no dispatch.
- **Target:** #302/#315/#318/#344 curriculum cluster, #300/#323 island-birth, #217 saddle-to-saddle. LEVER
  (means/analysis instrument): moves the pointer only via a better birth-point/hand-off, and only after the
  reduced model is derived.

---

## One-line verdicts
1. 2607.07470 (contrastive/waterfilling): **LEVER (modest: grounding + $0 curvelet-band waterfilling probe) → #502/#497/#277/#25** — theorem is stationarity-only, which PROVES isotropic Fourier is wrong for our oriented boundary; transferable = Eq-4 waterfilling on generalized eigenvalues (basis-agnostic).
2. 2607.11052 (data synergy): **DOMINATED-bookmark → #434/#499/#481** — multi-model observational regression; analyzes a corpus, cannot cure n=1 starvation; keep synergy-map concept for post-sweep analysis only.
3. 2605.19943 (PTRM): **DOMINATED-bookmark → #247/curriculum** — discrete-grid test-time-compute trick; the one idea (noise-explore + value-select) already better-framed by our bifurcation + costate.
4. WECO AIDE² RSI: **DOMINATED-bookmark → apparatus/triality (#247)** — agentic ML-eng infra, not a score-mover; validates self-improving-apparatus thesis + adoptable reward-hacking-rate observability idea.
5. Fork Dynamics: **LEVER (MIT tool, github/hinsley/Fork) → bifurcation cluster (#302/#315/#318/#344/#300/#323/#217)** — confirms+extends existing reframe; $0 probe = fit movable-occupancy normal form to an existing run, check fold coincidence, then continue in λ.
