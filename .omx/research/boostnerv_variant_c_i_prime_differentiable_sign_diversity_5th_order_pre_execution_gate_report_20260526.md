<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — BoostNeRV Variant C-i' pre-execution gate report. DO NOT mutate after landing. -->
<!-- # FORMALIZATION_PENDING:variant_c_i_prime_5th_order_diagnosed_mechanism_fix_differentiable_tanh_sign_proxy_replaces_non_differentiable_indicator_pre_execution_gate_predicates_locked_before_empirical_run_per_catalog_229_PV_discipline_20260526 -->

# BoostNeRV-PR110 Variant C-i' `differentiable_sign_diversity_via_tanh_proxy` — Pre-execution gate report (5TH-ORDER)

**Lane**: `lane_boostnerv_variant_c_i_prime_differentiable_sign_diversity_5th_order_20260526`
**Subagent**: `boostnerv-variant-c-i-prime-differentiable-sign-diversity-via-tanh-proxy-5th-order-recursive-doctrine-20260526`
**Predecessor**: Variant C-i 4TH-ORDER REFUTATION (commit `1075a2f30`; subagent `a760c7d0720813162`)
**Authority**: Operator 2026-05-26 verbatim "all are approved"
**Fixture**: M5 Max MLX-local 50 pairs × 96×128 × 30ep × Adam (sister-identical to #1337/#1342/#1345/#1349/#1351)
**Cost**: $0 (MLX-local-only per "Remember all on MLX")

## Entropy-position declaration

**Entropy position**: P2 = loss-shape (TRAIN phase). Intervention REPLACES the non-differentiable `(residual > 0).astype(float32)` indicator with the differentiable soft proxy `(tanh(k * residual) + 1) / 2`. Both operate at the same entropy-position layer (UPSTREAM residual sign distribution) — but Variant C-i' provides nonzero chain-rule gradient back to residual parameters, whereas Variant C-i had ZERO chain-rule gradient (empirically refuted at #1351). Attacks the SAME entropy distribution as #1351 via DIFFERENTIABLE proxy.

## Full-stack fractal optimization decomposition (5TH-ORDER node)

```
BoostNeRV-PR110 substrate
├── ingredient #6 curriculum / loss-shape
│   └── sub-ingredient L2 loss MSE
│       └── sub-sub-ingredient sign-diversity-regularizer (Variant C-i) ← 4TH-ORDER REFUTED via non-differentiable indicator
│           └── sub-sub-sub-ingredient differentiable sign-proxy (Variant C-i' = tanh) ← 5TH-ORDER NODE THIS WORK
│               ├── sub-sub-sub-sub-ingredient sharpness parameter k (controls hardness of soft indicator)
│               ├── sub-sub-sub-sub-ingredient gain_clamp (Carmack-best from #1342)
│               └── sub-sub-sub-sub-ingredient λ_sign_diversity (penalty magnitude)
```

**Recursive doctrine cumulative trajectory** (THIS = 6th landing):

| # | Landing | Decomposition node | Outcome |
|---|---|---|---|
| 1 | #1337 L1 EMPIRICAL | training schedule | WIN |
| 2 | #1342 gain_clamp sweep | codec hyperparameter | WIN |
| 3 | #1345 Variant B-d codec | codec design | REFUTED→2nd-order |
| 4 | #1349 Variant C-ii centering | base-output centering | REFUTED→3rd-order |
| 5 | #1351 Variant C-i sign-diversity | sign-diversity-regularizer | REFUTED→4th-order (non-differentiable indicator diagnosed) |
| 6 | **#1352 (THIS) Variant C-i' tanh-proxy** | **5TH-ORDER differentiable sign-proxy via tanh** | **PENDING** |

## Premise verification (Catalog #229)

**4th-order DIAGNOSED MECHANISM** = HARD-EARNED operating-within assumption per Catalog #292:
- **Evidence**: `1075a2f30` empirical anchor: positive_fraction = 0.0000 + sign_entropy = 0.0000 + mse trajectory IDENTICAL across all 6 cells + penalty FINAL = λ × 0.5 EXACTLY (confirming `positive_fraction = 0.0` at convergence, penalty = constant cost without gradient pressure).
- **Mathematical mechanism**: `(residual > 0).astype(float32)` indicator has gradient ZERO almost everywhere; chain-rule gradient from `|positive_fraction - 0.5|` back to residual parameters is ZERO.
- **The fix's mathematical justification**: `tanh(k * residual)` has gradient `k * (1 - tanh²(k * residual))` w.r.t. residual, which is NONZERO almost everywhere (zero only at saturation extremes). As k → ∞, `(tanh(k*r) + 1) / 2` recovers the indicator at the limit. For finite k, the proxy is differentiable everywhere and provides chain-rule gradient back to residual parameters.

## Sweep grid (6 cells)

- **Sharpness k ∈ {1, 5, 20}**: low k = smooth proxy with strong gradient signal but poor sign-resolution; high k = sharp proxy approaching indicator but with vanishing gradient at saturation. Sweep finds the sweet spot.
- **gain_clamp ∈ {0.05, 0.20}**: preserves Carmack-best from #1342; allows apples-to-apples comparison with sister sweeps.
- **λ_sign_diversity = 1.0 FIXED**: at #1351 max penalty was 0.5 absolute (λ=1.0); we hold λ=1.0 to provide maximum gradient pressure for the new differentiable formulation.

## Pre-declared verdict criteria (LOCKED BEFORE EMPIRICAL RUN)

**VALIDATED branch** (4th-order diagnosis CORRECT; canonical-fix-CONFIRMED):
- (V1) **avg positive_fraction_soft → 0.5** (within ±0.10) across ≥4 of 6 cells AT AT LEAST ONE k value
- (V2) **avg sign-bitmap entropy > 0.2 bits** at ≥4 of 6 cells (sign distribution actually diversifies)
- (V3) **Variant B-d sidecar bytes shrink** below 149B baseline at ≥1 cell (downstream byte-axis confirmation)
- ALL THREE must hold for VALIDATED verdict.

**REFUTED branch** (6th-order recursive iteration required):
- (R1) avg positive_fraction_soft < 0.10 OR > 0.90 at ALL 6 cells (still sign-degenerate)
- (R2) avg sign-bitmap entropy < 0.1 bits at ALL 6 cells (no real diversification)
- (R3) Variant B-d sidecar bytes stays ≥ 149B at ALL cells (no byte-axis improvement)
- ANY of R1/R2/R3 triggers REFUTED → 6th-order branches.

**SPLIT branch** (partial signal): only some criteria met → CONTINGENT-promote to nuanced verdict.

## 6th-order candidates if REFUTED

Per Catalog #308 alternative-probe-methodology enumeration:
- **C-iii architectural**: paired +/- residual heads (forces sign decomposition by structure)
- **C-v asymmetric activation**: replace tanh with shifted sigmoid (breaks tanh symmetry)
- **C-vi Huber loss**: replace L2 with Huber (softer gradient field; admits sign mixtures)
- **C-vii per-channel sign-diversity**: per-element penalty (provides differential signal global cannot)
- **C-viii initialization-driven**: positive bias init on conv2 (breaks all-negative attractor at start)

## Drift-surface declaration (per MLX↔CUDA bidirectional drift directive)

All results stamped `[macOS-MLX research-signal]` per Catalog #192/#317/#341. Sign-distribution behavior + sidecar-byte arithmetic should be ARCHITECTURE-CLASS-INVARIANT (deterministic + non-stochastic operations); MLX-CUDA drift expected ≤1e-4. NO CUDA dispatch in this work; predicted drift band based on Catalog #344 equation `mps_drift_architecture_class_dependent_v1`.

## Canonical-vs-frontier-push decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Fixture (50 pairs × 96×128 × 30ep × Adam × seed 42) | CANONICAL-SISTER | Apples-to-apples vs 5 sister sweeps |
| ResidualHeadModule | CANONICAL-SISTER | Identical to #1337/#1342/#1345/#1349/#1351 |
| Variant A int8 codec | CANONICAL-SISTER | #1337 baseline; 42B sister |
| Variant B-d sign-bitmap codec | CANONICAL-SISTER | #1345 baseline; 149B sister |
| Penalty formulation | FRONTIER-PUSH | NEW differentiable tanh-proxy replaces non-differentiable indicator |
| Sweep grid axes | FRONTIER-PUSH | k × gain_clamp at fixed λ=1.0 (max-pressure) |

## 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map: ACTIVE (chain-rule sensitivity from penalty to residual params IS the diagnosed mechanism)
- hook #2 Pareto constraint: N/A (single-objective)
- hook #3 bit-allocator: N/A (no allocator change)
- hook #4 cathedral autopilot dispatch: ACTIVE (per Catalog #344 equation #347 anchor)
- hook #5 continual-learning posterior: ACTIVE (canonical equation #347 anchor will land)
- hook #6 probe-disambiguator: ACTIVE (this empirical anchor IS the disambiguator between VALIDATED vs 6th-order branches)

## Operator-routable next step

Authorize: `.venv/bin/python .omx/tmp/boostnerv_variant_c_i_prime_differentiable_sign_diversity_sweep.py` (6-cell sweep; ~3-5s wallclock; $0 MLX-local).
