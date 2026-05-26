<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #229 premise-verification-before-edit pre-execution gate report. -->
<!-- # FORMALIZATION_PENDING:variant_c_ii_centered_base_recolor_pre_execution_gate_report_training_dynamics_fix_per_recursive_sub_ingredient_doctrine_GUIDING_PRINCIPLE_elevated_2026_05_26T19_10Z_decomposition_node_ingredient_6_sub_l2_loss_shape_sub_sub_base_output_centering -->

# BoostNeRV-PR110 Variant C-ii `centered_base_recolor` — Pre-execution gate report (2026-05-26)

**Lane**: `lane_boostnerv_variant_c_ii_centered_base_recolor_training_dynamics_fix_20260526` (L1 target)
**Subagent**: `boostnerv-variant-c-ii-centered-base-recolor-training-dynamics-fix-20260526`
**Predecessor sister**: `boostnerv-bpr1-variant-b-codec-redesign-break-scale-invariance-20260526` (commit `57ccd2b1e`; TaskCreate #1345 landing)
**Operator authority**: 2026-05-26 cascade follow-up to operator-routable #1 of sister #1345 ("HIGHEST EV: Variant C-ii `centered_base_recolor`")
**Budget**: $0 MLX-local per "Remember all on MLX"
**Expected wallclock**: ~30s (sister sweep was 17.4s; +centering helper is O(num_pairs) numpy)

---

## Full-stack fractal optimization decomposition (per just-elevated GUIDING PRINCIPLE 2026-05-26T19:10Z)

Per the recursive per-sub-ingredient doctrine + the PR95-sniped-lesson-amendment GUIDING PRINCIPLE elevation, this work optimizes the next-level decomposition node identified by sister #1345's empirical 2nd-order discovery:

```
BoostNeRV-PR110 substrate
├── ingredient #1 architecture (residual head module)
├── ingredient #2 base substrate (PR110 fec6)
├── ingredient #3 latent representation (z_pr110)
├── ingredient #4 codec (BPR1 int8 / Variant B-d sign-bitmap)
├── ingredient #5 brotli compression (q9)
├── ingredient #6 curriculum / loss-shape ← TODAY'S OPTIMIZATION TARGET
│   ├── sub-ingredient L2 loss MSE ← sister #1345 identified as bottleneck
│   │   ├── sub-sub-ingredient base-output centering ← TODAY'S FIX (Variant C-ii)
│   │   ├── sub-sub-ingredient sign-diversity regularizer (Variant C-i; future)
│   │   ├── sub-sub-ingredient paired +/- residual heads (Variant C-iii; future)
│   │   └── sub-sub-ingredient gain_clamp temperature schedule (Variant C-iv; future)
│   └── sub-ingredient training schedule (epochs/batch/LR; already optimized at L1 #1337)
├── ingredient #7 EMA shadow (canonical)
├── ingredient #8 QAT (sister #1345 RULED OUT — codec not the bottleneck)
└── ingredient #9 inflate runtime (PR110 base + residual decoder)
```

**Sister #1345's 2nd-order discovery**: ingredient #8 QAT sub-ingredient codec design was NOT the bottleneck (Variant B-d sign-bitmap codec EMPIRICALLY produced identical-149B sidecar across ALL 9 sweep cells with `global_sign_entropy_bits=0.0000` — root cause is training-dynamics sign-axis bias). The fractal optimization next-step is to descend into ingredient #6 sub-ingredient L2-loss-shape sub-sub-ingredient base-output centering.

**The recursive doctrine in action**: each empirical landing surfaces the next decomposition node to optimize. Sister #1337 optimized ingredient #6 sub-ingredient training schedule (got the WIN pattern). Sister #1342 + #1345 optimized ingredients #4 + #8 sub-ingredient gain_clamp+codec (empirically refuted as bottlenecks). TODAY optimizes the ACTUAL bottleneck per the 2nd-order discovery.

---

## Hypothesis (KEY HYPOTHESIS, falsifiable)

**The base-output centering fix breaks the sign-axis bias**: by mean-subtracting the PR110 base output to match GT's median BEFORE the residual learner trains, the residual target distributes around zero (signed) rather than collapsing to all-negative. This is predicted to:

- **Residual sign-distribution**: `global_positive_fraction` shifts from ~0.0 (all-negative at #1345) to ~0.5 (balanced; signed)
- **Variant B-d sidecar bytes**: shifts from 149B (sign-bitmap entropy=0 → brotli RLE collapse to 13B) toward NEW band depending on actual sign entropy; predicted in range [149B, ~1700B] (1-bit/pixel × NUM_PIXELS = 1536B if entropy=1.0; less if entropy<1.0 due to brotli compression)
- **Variant A int8 sidecar bytes** (sister L1 BPR1 codec): unchanged (int8 quantization is sign-symmetric; the absolute magnitude distribution is what matters)
- **Loss-axis WIN preservation**: must hold (centering is theoretically loss-neutral; only changes WHERE in residual space the optimizer lands, not the achievable optimum)

**Refutation criteria**:
- IF residual sign distribution remains ~0.0 positive fraction → centering DID NOT fix the sign-axis bias (mechanism hypothesis wrong; the bias is NOT base-output overshoot)
- IF loss-axis WIN regresses (#1337 was 7.8% reduction; was 16.2% recon proxy MSE at gain_clamp=0.05 / 30ep) → centering introduces a confounding bias (training-dynamics regression)
- IF Variant B-d sidecar bytes still constant 149B across all 9 cells → scale-invariance is NOT a function of sign distribution (deeper mechanism unidentified)

---

## Implementation design (Variant C-ii `centered_base_recolor`)

### Source location (per Catalog #229 PV)

The sweep harness `.omx/tmp/boostnerv_pr110_bpr1_variant_b_sweep.py` (sister #1345) constructs the residual target at training time via:

```python
def _loss_fn(model_inner, batch_pr110, batch_gt, batch_z):
    rgb_base = batch_pr110[:, 0]          # PR110 base output (BHW3)
    gt_target = batch_gt[:, 0]            # GT (BHW3)
    residual_pred = model_inner(rgb_base, batch_z)
    composed = compose_pr110_base_plus_residual(rgb_base, residual_pred, gain_clamp)
    mse = mx.mean((composed - gt_target) ** 2)
    return mse
```

The implicit residual target is `gt_target - rgb_base` (since `composed = clip(rgb_base + residual_pred, 0, 1)` and L2 minimizes the difference from `gt_target`). The sign-axis bias arises because PR110 base typically overshoots GT (positive bias in `rgb_base - gt_target`) → the residual learner is pushed toward all-negative `residual_pred`.

### Centering fix (Variant C-ii)

**Insertion point**: BEFORE training begins, compute a per-channel median offset over the entire training set:

```python
def compute_centering_offset(pr110_pairs: mx.array, gt_pairs: mx.array) -> mx.array:
    """Compute per-channel median offset to recolor PR110 base to match GT median.

    Returns: shape (3,) tensor of median offsets to ADD to PR110 base.
    """
    # Per-channel: median(GT) - median(base) over all (pair, H, W) positions
    # Frame 0 of each pair is what's used in the loss; mirror that.
    rgb_base = pr110_pairs[:, 0]   # (NUM_PAIRS, H, W, 3)
    gt_target = gt_pairs[:, 0]      # (NUM_PAIRS, H, W, 3)
    # Flatten spatial+pairs dims; keep channel dim
    rgb_base_np = np.array(rgb_base).reshape(-1, 3)
    gt_target_np = np.array(gt_target).reshape(-1, 3)
    offset_np = np.median(gt_target_np, axis=0) - np.median(rgb_base_np, axis=0)
    return mx.array(offset_np.astype(np.float32))
```

Then in `_loss_fn`:

```python
def _loss_fn(model_inner, batch_pr110, batch_gt, batch_z, centering_offset_local):
    rgb_base_raw = batch_pr110[:, 0]
    rgb_base_centered = mx.clip(rgb_base_raw + centering_offset_local, 0.0, 1.0)
    gt_target = batch_gt[:, 0]
    residual_pred = model_inner(rgb_base_centered, batch_z)
    composed = compose_pr110_base_plus_residual(rgb_base_centered, residual_pred, gain_clamp)
    mse = mx.mean((composed - gt_target) ** 2)
    return mse
```

The model now sees `rgb_base_centered` (median-shifted) as input + composes residuals against the same centered base. The OFFSET is a property of the (frozen) PR110 base + GT pair; it stamps a 12-byte fp32 sidecar field (3 channels × 4 bytes). At inflate time, the same offset is added BACK to the PR110 base before composing the residual.

**Inflate-side**: the BPR1 sidecar will need to carry the 12-byte centering offset. For sweep diagnostic purposes, we don't actually build inflate runtime — we measure residual sign distribution + sidecar bytes (the bytes claim ASSUMES the 12-byte offset is added to the sidecar overhead).

### Sweep harness changes (`.omx/tmp/boostnerv_pr110_variant_c_ii_centered_base_recolor_sweep.py`)

Copy `.omx/tmp/boostnerv_pr110_bpr1_variant_b_sweep.py` verbatim and:

1. Add `compute_centering_offset` helper before `train_residual_head`
2. Add `centering_offset` parameter throughout the training pipeline
3. Use Variant A (L1 BPR1 int8) codec from `src/tac/substrates/boost_nerv_pr110_residual/archive.py` as the primary measurement (the L1 baseline at #1337 was 42B; centering should restore int8 path's full byte count if magnitudes increase)
4. ALSO measure Variant B-d sidecar bytes for the sister-comparison (was 149B at #1345; predicted to grow if sign entropy > 0)
5. Persist sidecar 12-byte centering offset accounting in heatmap JSON
6. Add 12B to all sidecar-byte reports per Catalog #229 honest accounting

Output: `.omx/research/boostnerv_variant_c_ii_centered_base_recolor_sweep_results_20260526/sweep_heatmap.json`

---

## Drift surface declaration (per MLX↔CUDA bidirectional drift directive 2026-05-26)

The Variant C-ii fix introduces a NEW drift surface: `numpy.median` over float32 arrays. NumPy median uses partition-based selection (deterministic for equal-length arrays); MLX may not have native median (uses sort+index workaround). To avoid drift:

- **Compute centering offset via numpy.median ONLY on CPU** (already off the MLX training graph since it's a one-time pre-training setup)
- **Stamp the offset as fp32 in the BPR1 sidecar** (canonical sister of grayscale-LUT pattern per Daubechies wavelet hierarchical priors)
- **Inflate-side applies the offset deterministically** (12-byte field; bit-stable round-trip)

Portability verdict: ZERO new drift surface introduced for the actual training (MLX defaults unchanged); ONE new field (12B centering offset) added to BPR1 sidecar grammar with explicit fp32 byte-stable round-trip.

---

## Canonical-vs-frontier-push decision (per pushing-the-frontier directive 2026-05-26)

- **Centering offset computation**: CANON-APPLICATION. Median-subtract for distribution centering is canonical preprocessing (sister of LayerNorm + standardization).
- **L2 loss preservation**: CANON. No new loss term.
- **Sign-axis diversification mechanism**: FRONTIER-PUSH. The hypothesis that base-output centering fixes the sign-axis bias is an EMPIRICAL one derived from sister #1345's 2nd-order discovery; canonical literature does NOT directly cite "centered base recolor for residual sign-distribution diversification". This is original empirical-grounded design.

---

## Catalog #303 cargo-cult audit (sub-sub-ingredient base-output centering)

| Assumption | Classification | Rationale |
|---|---|---|
| Median-offset centering preserves L2 loss optimum | HARD-EARNED-mathematical | L2 loss = `|composed - gt|² = |rgb_base + offset + residual - gt|² = |rgb_base + (offset + residual) - gt|²`. The optimizer can compensate by subtracting `offset` from `residual` to reach the same final composed value. Sister proof: classical preconditioning theorem. |
| Sign-distribution diversification translates to sidecar entropy growth | CARGO-CULTED-pending-empirical | This is THE hypothesis under test. Falsifiable per refutation criteria above. |
| `numpy.median` is the right centering statistic (vs `mean`) | CARGO-CULTED-design-choice | Median is robust to outliers + matches "typical-value" intuition. Mean would be cleaner mathematically but more susceptible to outlier-frame influence. Could try both in follow-up Variant C-ii-b. |
| 12-byte centering offset stamp is negligible vs sidecar bytes | HARD-EARNED-arithmetic | At gain_clamp=0.05 / 30ep / Variant A, sister #1337 sidecar was 42B; +12B = 54B (29% increase). At Variant B-d (149B), +12B = 161B (8% increase). NOT negligible at Variant A but acceptable since the WIN of restored magnitude distribution dominates. |

---

## Catalog #305 observability surface declaration

Per-cell instrumented:

1. **Inspectable per layer**: residual_pred raw tensor per pair (NUM_PAIRS × H × W × 3); centering_offset (3,) tensor
2. **Decomposable per signal**: sign distribution per pixel + per pair + per channel + globally; sidecar bytes split (header + len fields + brotli + magnitudes + centering offset)
3. **Diff-able across runs**: identical fixture+seed+training; ONLY difference is centering_offset application
4. **Queryable post-hoc**: heatmap_json carries all per-cell + per-pair stats
5. **Cite-able**: every result row stamps PR110 archive sha256 + canonical equation #347 anchor reference
6. **Counterfactual-able**: sister `--disable-centered-base-recolor` flag emits identical fixture WITHOUT centering for direct A/B comparison (functionally equivalent to sister #1345 with same seed)

---

## 6-hook wire-in declaration per Catalog #125 (planned for landing memo)

1. **Sensitivity-map contribution**: ACTIVE — centering_offset per-channel + residual sign distribution feed sister `tac.sensitivity_map.*` ranker
2. **Pareto constraint**: ACTIVE — NEW (gain_clamp, sidecar_bytes) Pareto point for Variant A + Variant B-d with centering applied
3. **Bit-allocator hook**: N/A (uniform per-pixel allocation; codec structurally fixed)
4. **Cathedral autopilot dispatch hook**: ACTIVE — sweep_heatmap.json stamped Catalog #323 canonical Provenance
5. **Continual-learning posterior update**: ACTIVE — canonical equation #347 anchor_append via `tac.canonical_equations.update_equation_with_empirical_anchor`
6. **Probe-disambiguator**: ACTIVE — the Variant C-ii sweep IS the canonical operator-routable disambiguator between "sign-axis bias is the L2-loss-shape sub-ingredient bottleneck" vs "the bottleneck is deeper (e.g., needs sign-diversity term Variant C-i)"

---

## HORIZON-CLASS per Catalog #309

`frontier_pursuit` (same as sister #1345; mechanism investigation; predicted PLATEAU-ADJACENT band for sidecar bytes; canonical equation #347 domain-of-validity refinement to add `_centered_base_recolor_training_dynamics_fix_at_PR110_overshoot_regime` context)

---

## Pre-dispatch waivers (Catalogs #303 / #294 / #305 / #309 satisfied above)

This is MLX-local research-signal only; NO paid dispatch. No symposium required per Catalog #325 (research_only signal not contest-promotion dispatch).
