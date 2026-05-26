<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — BoostNeRV-PR110 Variant C-i sign-diversity penalty PRE-EXECUTION GATE REPORT. DO NOT mutate after landing. -->
<!-- Catalog #229 PV: read 4 sister landing memos (#1337 + #1342 + #1345 + #1349) + 3 sister sweep harnesses + 2 canonical doctrine memos before drafting. -->
<!-- # FORMALIZATION_PENDING:variant_c_i_sign_diversity_4th_order_recursive_doctrine_directly_attacks_L2_tanh_clip_attractor_topology_per_just_elevated_GUIDING_PRINCIPLE_2026_05_26 -->

# BoostNeRV-PR110 Variant C-i `residual_loss_with_sign_diversity_term` — PRE-EXECUTION GATE REPORT 2026-05-26

**Lane**: `lane_boostnerv_variant_c_i_residual_loss_sign_diversity_term_20260526` (L0 → L1 on landing)
**Subagent**: `boostnerv-variant-c-i-residual-loss-with-sign-diversity-term-4th-order-recursive-doctrine-20260526`
**Predecessor**: `boostnerv-variant-c-ii-centered-base-recolor-training-dynamics-fix-20260526` (commit `86cfe4aad`; TaskCreate #1349; 3rd-order discovery surfaced this node)
**Operator authority**: 2026-05-26 cascade follow-up to sister #1349 operator-routable #1 (HIGHEST EV next decomposition node)

## Entropy-position declaration (per just-landed entropy-position discipline `1a85400dd`)

**Entropy position**: P2 = loss-shape (TRAIN phase). The intervention adds `λ * |global_positive_fraction - 0.5|` penalty to the L2 objective during gradient training. It MODIFIES the UPSTREAM entropy distribution that downstream P10 sidecar codecs (Variant A int8 / Variant B-d sign-bitmap) see. The downstream entropy-position is INDIRECTLY affected via the upstream loss-shape change.

**Sister entropy-position landings**:
- #1342 gain_clamp sweep = P10 codec hyperparameter (downstream)
- #1345 Variant B-d sign-bitmap codec = P10 codec design (downstream)
- #1349 Variant C-ii centered_base_recolor = P1 base-input transform (upstream)
- **#1351 (THIS) Variant C-i sign-diversity penalty = P2 loss-shape (upstream gradient field shaping)** ← canonical first attack on the L2+tanh+clip attractor topology itself

## Full-stack fractal optimization decomposition (per just-elevated GUIDING PRINCIPLE 2026-05-26T19:10Z)

```
BoostNeRV-PR110 substrate
├── ingredient #6 curriculum / loss-shape ← THIS WORK'S OPTIMIZATION TARGET
│   └── sub-ingredient L2 loss MSE ← sister #1345 identified as bottleneck
│       ├── sub-sub-ingredient base-output centering (Variant C-ii; #1349 REFUTED 3rd-order)
│       ├── sub-sub-ingredient sign-diversity regularizer (Variant C-i; **THIS FIX**) ← 4TH-ORDER NODE
│       │   ├── sub-sub-sub-ingredient λ scale (sweep {0.01, 0.1, 1.0})
│       │   ├── sub-sub-sub-ingredient global vs per-channel positive_fraction (next 5th-order if Variant C-i refuted)
│       │   └── sub-sub-sub-ingredient L1 vs L2 vs Huber base loss (next 5th-order alternative)
│       ├── sub-sub-ingredient paired +/- residual heads (Variant C-iii; future)
│       └── sub-sub-ingredient gain_clamp temperature schedule (Variant C-iv; future)
```

**Recursive doctrine trajectory**:

| Landing | Decomposition node | Outcome | Next node |
|---|---|---|---|
| #1337 L1 EMPIRICAL | ingredient #6 sub-ingredient training schedule | WIN (7.8% loss reduction) | gain_clamp value |
| #1342 gain_clamp sweep | ingredient #4 sub-ingredient codec hyperparameter | WIN (53% recon-MSE-reduction) | codec design |
| #1345 Variant B-d codec | ingredient #8 sub-ingredient codec design | REFUTED → 2nd-order training-dynamics | L2-loss-shape |
| #1349 Variant C-ii centering | ingredient #6 sub-ingredient L2-loss-shape sub-sub-ingredient base-output centering | REFUTED → 3rd-order L2+tanh+clip attractor topology | sign-diversity penalty (THIS) |
| **#1351 (THIS) Variant C-i sign-diversity** | **L2+tanh+clip optimizer attractor topology DIRECTLY** | **OPEN — empirical test will VALIDATE or surface 4th-order discovery** | If REFUTED: 5th-order = activation function change OR loss-norm change OR per-channel sign-diversity |

## Hypothesis under test (4TH-ORDER)

**3rd-order discovery from #1349 (commit `86cfe4aad`)**:
> "L2 loss + tanh output + clip([-gain_clamp, +gain_clamp]) composition has a STRUCTURAL SIGN ATTRACTOR independent of base-output position in the [0,1] domain. The optimizer's gradient field for L2 loss with tanh output and symmetric clip is dominated by a single sign mode at convergence; centering changes WHERE in residual space the optimizer lands (smaller magnitude residuals because base-alone MSE is smaller) but NOT WHICH sign attractor (still all-negative)."

**4th-order hypothesis (this work)**:
> Adding `λ * |global_positive_fraction - 0.5|` penalty to L2 objective DIRECTLY pulls the optimizer's gradient field away from the single-sign attractor by penalizing sign-degenerate distributions. The penalty's gradient w.r.t. residual sign is monotonically toward the 50/50 balance point. If the attractor is purely a function of the L2 loss landscape (and not of e.g. tanh activation saturation OR clip symmetry OR AdamW momentum smoothing), this DIRECT penalty SHOULD break the attractor and produce sign-balanced residuals.

**Predictions**:
- **IF VALIDATED**: At λ=0.1, global_positive_fraction approaches 0.5 (target: > 0.10 per sister thresholds); sign-bitmap entropy grows toward 1.0 bit (target: > 0.10 bits); Variant B-d sign-bitmap codec sidecar bytes grow above ~200B (entropy-driven RLE expansion) reflecting actual signed information; L2 recon proxy MSE reduction MAY regress slightly (the penalty term competes with L2 fit).
- **IF REFUTED**: positive_fraction stays at 0.0 even with λ=1.0; sign_entropy stays at 0.0; this surfaces a 4th-order discovery: the L2+tanh+symmetric-clip attractor is NOT escapable via direct sign-balance penalty (would imply the attractor is structural at the activation+clip composition level, not the loss landscape level; would route to 5th-order alternatives: replace tanh with asymmetric activation; replace L2 with Huber; per-channel sign-balance; paired +/- residual heads from Variant C-iii).

## Variant C-i implementation plan

**Sweep harness**: `.omx/tmp/boostnerv_variant_c_i_sign_diversity_sweep.py` (~620 LOC; sister-derived from `.omx/tmp/boostnerv_variant_c_ii_centered_base_recolor_sweep.py`)

**The fix is a 3-line training-pipeline insertion** (in `_loss_fn`):
1. Compute `positive_fraction = mx.mean(mx.greater(residual_clamped, 0.0).astype(mx.float32))` per training batch (over residual_pred AFTER clamp)
2. Compute `sign_diversity_penalty = λ_sign_diversity * mx.abs(positive_fraction - 0.5)`
3. Add to objective: `total_loss = mse + sign_diversity_penalty`

**Sweep grid (6 cells)**: `λ_sign_diversity ∈ {0.01, 0.1, 1.0} × gain_clamp ∈ {0.05, 0.20}` (preserving Carmack-best gain_clamp from #1342). Drops epochs grid axis to 30 epochs only per fixture parsimony (sister sweeps showed sign distribution stable across epochs in C-ii; this saves wallclock).

**Sister-coherence**: identical fixture (50 pairs × 96×128 / batch_size=8 / lr=1e-3 / seed=42 / Adam → AdamW) + identical PR110 base archive + identical GT video + identical canonical Variant A + Variant B-d codecs for sister-comparison.

**Provenance per Catalog #323**: every result row stamped `axis_tag=[macOS-MLX research-signal]` + `promotion_eligible=False` + `score_claim=False` + `ready_for_exact_eval_dispatch=False`.

## Refutation criteria (Catalog #307 paradigm-vs-implementation discipline pre-declared)

**Per pre-execution criteria locked BEFORE empirical run** (no post-hoc gating per "Forbidden empirical-claim-without-evidence-tag"):

- **VALIDATED**: avg positive_fraction across λ-sweep > 0.10 AND avg sign_entropy > 0.10 bits AND loss-axis WIN preserved (recon-MSE-reduction > 5%) at best cell
- **REFUTED (4th-order discovery)**: avg positive_fraction stays < 0.10 across all 6 cells DESPITE direct sign-diversity penalty (would imply L2+tanh+clip attractor is structural at activation+clip level not loss-landscape level)
- **PARTIAL**: positive_fraction moves toward 0.5 but sign_entropy stays low (entropy didn't grow proportionally; per-batch positive_fraction may oscillate while per-pixel sign stays uniform — possible cell-level patch issue)
- **REGRESSION**: loss-axis WIN destroyed (recon-MSE-reduction < 5%); penalty too dominant; investigate smaller λ OR per-channel formulation

## Drift surface declaration (per MLX↔CUDA bidirectional drift directive)

Per `feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md`, the Variant C-i sweep reuses sister #1349 + #1345 + #1342 canonical helpers verbatim (identical fp32 throughout / MLX defaults / NHWC / tanh+clip ordering / AdamW β₁/β₂ defaults / brotli q9 determinism).

**NEW drift surface introduced**:
- `mx.mean(mx.greater(residual_clamped, 0.0).astype(mx.float32))` — uses MLX greater + astype + mean (all bit-stable on MLX). CUDA sister would use `torch.mean((residual_clamped > 0).float())` which is bit-identical.
- `mx.abs(positive_fraction - 0.5)` — bit-stable everywhere.

**Portability verdict**: ZERO new training-time drift surface; penalty term is bit-stable across MLX/CUDA. Future paired CUDA verification inherits the L1+#1342+#1345+#1349 drift-surface declarations unchanged plus the bit-stable sign-diversity penalty.

## Canonical-vs-frontier-push decision (per pushing-the-frontier directive)

- **Sign-diversity penalty as loss term**: FRONTIER-PUSH. NO canonical literature directly cites "L2 loss + tanh + symmetric clip residual learner with explicit `|positive_fraction - 0.5|` penalty to break attractor topology". This is a novel objective-engineering primitive contribution to the canonical equation #347 domain.
- **AdamW + standard MSE base**: CANON-APPLICATION.
- **6-cell λ × gain_clamp grid**: CANON-APPLICATION (sister of #1342 9-cell pattern).
- **Empirical attractor-topology probe**: FRONTIER-PUSH EMPIRICAL CONTRIBUTION (sister of #1349's 3rd-order discovery).

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: ACTIVE — per-batch positive_fraction trajectory + per-channel positive_fraction breakdown feed sister `tac.sensitivity_map.*` ranker
2. **Pareto constraint**: ACTIVE — Variant C-i with sign-diversity penalty gives a NEW (recon_MSE_reduction, sidecar_bytes, sign_entropy) tradeoff point
3. **Bit-allocator hook**: N/A (uniform per-pixel allocation; codec structurally fixed)
4. **Cathedral autopilot dispatch hook**: ACTIVE — sweep_heatmap.json carries canonical Provenance per Catalog #323 + auto-discoverable per Catalog #335 contract
5. **Continual-learning posterior update**: ACTIVE — empirical anchor appended to canonical equation #347 via `tac.canonical_equations.update_equation_with_empirical_anchor` per Catalog #344
6. **Probe-disambiguator**: ACTIVE — Variant C-i IS the canonical operator-routable disambiguator probe per Catalog #313 between "L2+tanh+clip attractor is escapable via direct loss-term penalty (Variant C-i validates)" vs "L2+tanh+clip attractor is STRUCTURAL at activation+clip level requiring architectural change (Variant C-i refutes → 4th-order)"

## HORIZON-CLASS verdict per Catalog #309

`frontier_pursuit` (same as sister #1349; predicted PLATEAU-ADJACENT band; mechanism investigation at loss-landscape-level; canonical equation #347 domain refinement is structural-protection contribution).

## Cross-references

- Sister #1349 landing memo: `.omx/research/boostnerv_variant_c_ii_centered_base_recolor_landed_20260526.md` (3rd-order discovery anchor)
- Sister #1345 Variant B-d landing memo: `.omx/research/boostnerv_bpr1_variant_b_codec_redesign_landed_20260526.md` (2nd-order discovery anchor)
- Sister #1342 gain_clamp sweep landing memo: `.omx/research/boostnerv_pr110_gain_clamp_sweep_landed_20260526.md` (1st-order discovery anchor)
- Sister #1337 L1 EMPIRICAL landing memo: `.omx/research/boostnerv_pr110_l1_empirical_landed_20260526.md` (baseline)
- Sister sweep harness (Variant C-ii): `.omx/tmp/boostnerv_variant_c_ii_centered_base_recolor_sweep.py` (template)
- Canonical equation #347: `.omx/state/canonical_equations_registry.jsonl` (5 anchors as of #1349)
- GUIDING PRINCIPLE: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_pr95_sniped_lesson_GUIDING_PRINCIPLE_full_stack_fractal_optimization_elevation_20260526.md`
- Entropy-position cascade catalog: commit `1a85400dd` repo research memo
