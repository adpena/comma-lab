# Deterministic optimizer design-constraint directive: problem + domain + performance + signal + elegant + beautiful + clean
# Date: 2026-05-18
# Audience: in-flight subagent acb41f8d3f7f0a3ea (deterministic score optimizer + Wyner-Ziv Q4 anchor + 12 alternative frameworks)
# SUPPLEMENTS the prior alternative-frameworks directive (commit 1694726b4)
# Per CLAUDE.md "Subagent coherence-by-default" + "Beauty, simplicity, and developer experience" non-negotiables

## Operator directive (verbatim 2026-05-18)

> *"must ensure solver is problem and domain space optimized and performance and signal optimized, elegant and beautiful and clean"*

This is a BINDING DESIGN CONSTRAINT that filters the 12-framework comparative analysis from the prior directive. NOT just a preference — a hard filter.

## What "problem-space optimized" means here

The solver MUST be designed specifically for the contest scorer's exact mathematical structure:
```
score = 100·d_seg(theta) + sqrt(10·d_pose(theta)) + 25·archive_bytes(theta)/37_545_489
```

Where each term has DIFFERENT mathematical structure:
- `d_seg(theta)` = average per-pixel class disagreement (5-class argmax of UNet output) — **piecewise-constant; tropical/max-plus algebra**
- `d_pose(theta)` = MSE of first 6 pose dims (FastViT regression) — **locally quadratic; Newton's method**
- `archive_bytes(theta)` = size of archive.zip after codec encoding — **linear in byte count; integer programming**

Generic-purpose solvers (L-BFGS, SQP, generic ADMM) are DISFAVORED by this constraint. Domain-specialized solvers (tropical for d_seg + quadratic Newton for d_pose + LP for rate) are FAVORED.

## What "domain-space optimized" means here

The solver MUST exploit known domain structure:
- **Video compression** = temporal redundancy → predictive coding / Wyner-Ziv side-info
- **600 video pairs** = finite-sum objective → SVRG / SAGA variance reduction (NOT vanilla SGD)
- **Per-pair master gradient** = cos(seg, pose) = 0.8973 → 1D null-space exploit
- **Archive.zip** = constraint set with linear minimization oracle → Frank-Wolfe-friendly
- **Per-pixel SegNet UNet** = bilinear upsample structure → wavelet-multi-scale priors (Daubechies)
- **Per-pose FastViT** = RepMixer + MSE → smooth-locally-quadratic Newton ideal
- **Contest-compliance** = hard yes/no constraint per maintainer precedent → active-set methods

## What "performance-optimized" means here

The solver MUST:
- Run on M5 Max in seconds-to-minutes (NOT hours; operator-attention budget)
- Leverage existing canonical helpers (master_gradient + sensitivity_map + xray + cathedral autopilot) rather than re-implement
- Exploit FP64 precision per-pair master gradient (NOT redo gradient extraction inside the solver)
- Compose orthogonally with existing reward factors in cathedral autopilot v2 cascade

## What "signal-optimized" means here

Per the 70-surface synthesis + 18-granularity expansion: the solver MUST consume EVERY analytical surface that moves the score, and IGNORE the ones that don't:
- CONSUME: per-pair master gradient (fp64); per-byte sensitivity_mask_aware_quantizr_v1 tiers; per-class chroma anchors; per-region SegNet softmax histograms; per-pixel UNIWARD weights; null-space basis cos(seg, pose); xray 13 canonical primitives
- IGNORE: surfaces that don't empirically contribute to score (per Catalog #287 [empirical:<path>] tag discipline)

The solver's signal-to-noise ratio = (score-impact-per-compute-cycle). Maximize this ratio.

## What "elegant + beautiful + clean" means here

Per CLAUDE.md "Beauty, simplicity, and developer experience — non-negotiable" + HNeRV parity discipline L4 (reviewability in 30 seconds):
- **Small typed abstractions** — frozen dataclasses + protocols, NOT inheritance hierarchies
- **Clear names** — `OptimalThetaUpdate` NOT `OpaqueResultObject`
- **Deterministic schemas** — fp64 throughout (NOT fp32 for speed); reproducible across CPU/CUDA
- **Composable contracts** — solver composes with bit_allocator + cathedral autopilot + Pareto solver via canonical interfaces
- **30-second reviewability** — each solver phase + each analytical-surface integration must be reviewable in 30 sec
- **Mathematically pure** — first-principles derivation (tropical algebra is canonical algebra of argmax; proximal splitting is canonical for additive nonsmooth+smooth+linear) NOT ad-hoc heuristics

## REVISED FRAMEWORK RECOMMENDATION (filtered by binding design constraints)

Out of the 12 frameworks from the prior directive, the following are MOST ALIGNED with the design constraints:

### TIER 1 — Foundational (must include)
1. **Tropical/max-plus algebra for d_seg** — canonical algebra of argmax; mathematically pure; problem-space optimized
2. **Newton's method (locally quadratic) for d_pose** — MSE structure → analytically derivable Hessian; problem-space optimized
3. **Linear programming for rate** — archive_bytes is integer-linear; problem-space optimized
4. **Proximal splitting (Douglas-Rachford) for term composition** — separates the 3 different mathematical structures cleanly; elegant; clean

### TIER 2 — Domain-specialized (highly recommended)
5. **Wyner-Ziv source coding with decoder side-info** — exact match for "ship seed; reconstruct codebook at inflate"; already canonical in repo via Catalog #319 Q1-Q5; domain-space optimized
6. **Tishby Information Bottleneck** — `min I(theta; archive_bytes) subject to I(archive_bytes; frames) ≥ k_seg, I(archive_bytes; poses) ≥ k_pose` — IS the contest scorer in IB formulation; mathematically elegant
7. **Daubechies wavelet-multi-scale for per-pixel sensitivity** — matches bilinear-upsample structure of SegNet UNet; domain-space optimized; performance-optimized via FFT
8. **Mirror descent on Pareto simplex (alpha+beta+gamma=1)** — Bregman-divergence-aware; faster convergence than projected gradient; clean implementation

### TIER 3 — Supporting (use when applicable)
9. **SVRG/SAGA variance reduction over 600 pairs** — finite-sum objective; performance-optimized
10. **Frank-Wolfe for archive.zip constraint set** — linear minimization oracle; avoids projection; clean

### DISFAVORED (filtered out by design constraints)
11. **Algebraic geometry / Gröbner basis** — too computationally expensive (exponential worst case); fails performance constraint
12. **Game-theoretic minimax** — overcomplicated; fails elegance constraint unless we're training adversarial codec

## CANONICAL HYBRID COMPOSITION

The architecturally clean solver design:

```python
# src/tac/deterministic_score_optimizer/canonical_hybrid_solver.py

@dataclass(frozen=True, slots=True)
class HybridDeterministicSolver:
    """Problem-space + domain-space + performance + signal + elegant + beautiful + clean.

    Composition rationale per CLAUDE.md "Meta-Lagrangian/Pareto solver" GR-style variational principle:
    - d_seg term: tropical/max-plus algebra (argmax-piecewise-constant)
    - d_pose term: Newton's method (locally quadratic MSE)
    - rate term: linear programming (integer-linear)
    - composition: Douglas-Rachford proximal splitting (handles mixed smoothness)
    - codebook layer: Wyner-Ziv source coding (Catalog #319 Q1-Q5 canonical)
    - per-pixel sensitivity: Daubechies wavelet-multi-scale (matches UNet bilinear)
    - Pareto simplex: mirror descent (KL-divergence aware)
    - finite-sum acceleration: SVRG (600-pair variance reduction)
    - constraint set: Frank-Wolfe (linear minimization oracle on archive.zip)
    """
    tropical_d_seg_solver: TropicalDSegSolver
    newton_d_pose_solver: NewtonDPoseSolver
    linear_rate_solver: LinearRateSolver
    proximal_splitter: DouglasRachfordSplitter
    wyner_ziv_codebook_layer: WynerZivCodebookLayer  # delegates to tac.wyner_ziv_deliverability
    daubechies_per_pixel: DaubechiesWaveletMultiScale
    mirror_descent_pareto: MirrorDescentPareto
    svrg_finite_sum: SVRGVarianceReduction
    frank_wolfe_constraint: FrankWolfeArchiveConstraint

    # Inputs (consumed from existing canonical helpers):
    master_gradient: PerPairMasterGradient  # tac.master_gradient (fp64)
    sensitivity_map: SensitivityMap  # tac.sensitivity_map
    venn_classification: VennClassification  # Catalog #319 v2 + 18-granularity
    xray_primitives: dict[str, XrayPrimitiveOutput]  # tac.xray.registry (13 canonical)

    def derive_optimal_update(self, current_theta, current_archive_bytes, pareto_alpha_beta_gamma) -> OptimalUpdate:
        """One-shot deterministic solve. Returns analytically-derived optimal theta update + predicted ΔS per axis."""
        ...
```

EACH sub-solver is small (~100-200 LOC), pure (no global state), tested (~20-30 dedicated tests), reviewable in 30 sec.

The HYBRID composition replaces the current 6-reward-factor heuristic cathedral autopilot ranker with an analytically-derived second-order Newton step that consumes ALL analytical surfaces (per the synthesis 70-surface inventory + 18-granularity expansion).

## INSTRUCTION TO acb41f8d3f7f0a3ea

INCORPORATE this design constraint analysis as the FINAL section of your DELIVERABLE 1 (`deterministic_score_optimizer_design_memo_*`). The hybrid composition above is the LEADING DESIGN RECOMMENDATION — but feel free to refine based on your deeper analysis of each framework's strengths.

The CRITICAL THESIS the operator needs from your deliverable: **the canonical hybrid solver design above (tropical + Newton + linear + Douglas-Rachford + Wyner-Ziv + Daubechies + mirror + SVRG + Frank-Wolfe composition) is problem-space + domain-space + performance + signal + elegant + beautiful + clean — and REPLACES the current 6-reward-factor heuristic cathedral autopilot ranker with second-order Newton's method on the contest score function consuming ALL canonical analytical surfaces.**

Acknowledge this directive in your next checkpoint via `tools/subagent_checkpoint.py --notes "incorporated design_constraint_directive_problem_domain_performance_signal_elegant_20260518"`.

— Main-Claude (relayed on behalf of operator 2026-05-18)
