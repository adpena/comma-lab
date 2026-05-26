<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — BoostNeRV-PR110 BPR1 Variant B codec redesign pre-execution gate report. DO NOT mutate after landing. -->
<!-- Catalog #229 PV: this memo verifies premises empirically — L1+sweep landings read in full, root-cause scale-invariance mechanism identified via heatmap+residual_stats inspection, Variant B design choice empirically grounded BEFORE harness emission. -->
<!-- # FORMALIZATION_PENDING:variant_b_pre_execution_gate_carries_design_choice_justification_no_canonical_equation_registration_at_pre_execution_stage_landing_memo_will_address_anchor_appended_events_per_catalog_344 -->

# BoostNeRV-PR110 BPR1 Variant B codec redesign — PRE-EXECUTION GATE REPORT 2026-05-26

**Subagent**: `boostnerv-bpr1-variant-b-codec-redesign-break-scale-invariance-20260526`
**Lane**: `lane_boostnerv_bpr1_variant_b_codec_redesign_break_scale_invariance_20260526`
**Predecessor**: `boostnerv-pr110-gain-clamp-sweep-20260526` (commit `8240aceda`)
**Operator authority**: 2026-05-26 cascade follow-up to Carmack-dissent dual-axis verdict in sweep landing memo (operator-routable #1 HIGHEST EV: codec-design FRONTIER-PUSH)

## Pre-execution checklist

| Item | Status | Evidence |
|---|---|---|
| Mandatory pre-flight: CLAUDE.md + AGENTS.md | DONE | NON-NEGOTIABLES honored ("MPS auth eval is NOISE" + Catalog #192/#317 advisory tagging + #287 placeholder rejection + #340 sister-checkpoint guard + #343 no hardcoded scores + #344 canonical-equation registry) |
| Both 2026-05-26 NEW standing-directive memos | DONE | (1) MLX↔CUDA bidirectional drift 5 sources; (2) pushing-the-frontier canon-vs-frontier-push framing |
| Read BoostNeRV L1 EMPIRICAL landing memo | DONE | Commit `b2fd3e587`; 42-byte BPR1 sidecar baseline |
| Read BoostNeRV gain_clamp sweep landing memo | DONE | Commit `8240aceda`; 9-cell heatmap; Carmack-dissent dual-axis verdict (LOSS validated 7.8%→34.6%; BYTES refuted 42B constant) |
| Inspect sweep heatmap JSON empirically | DONE | `sweep_heatmap.json` — confirmed 14B brotli of 1.84MB uncompressed int8 at ALL 9 cells; `clamped_fraction_at_boundary=1.0000` at ALL 9 cells |
| Sister-context disjoint scope check | DONE | NSCS06 v8 stacked paired Modal T4 (slot 1; DIFFERENT substrate) + PR110-OPT-3 Variant C variable-K (slot 2; selector-stream codec ≠ residual codec); MY scope = `src/tac/substrates/boost_nerv_pr110_residual/bpr1_variant_b_*.py` + `.omx/tmp/boostnerv_pr110_bpr1_variant_b_sweep.py` + `.omx/research/boostnerv_bpr1_variant_b_codec_redesign_*` + `.omx/research/boostnerv_pr110_bpr1_variant_b_sweep_results_20260526/` |
| Catalog #340 sister-checkpoint guard | PROCEED | All write targets are NEW files; ZERO existing-file mutations planned |
| Catalog #206 subagent checkpoint emitted | DONE | Step 1 in_progress at 2026-05-26T18:37:00Z |
| Canonical equation #347 `residual_hybrid_boosting_savings_v1` registry state | VERIFIED | Registered commit `04f34ea40` per Catalog #344 (sister script `tools/register_2_new_canonical_equations_20260526.py`); 0 empirical anchors currently appended; status PROVISIONAL pending empirical validation |

## ROOT-CAUSE empirical discovery (sweep heatmap re-inspection)

The sweep landing memo identified the scale-invariance fact (42B constant). **My re-inspection** of the heatmap JSON surfaces the underlying mechanism that the sweep memo did not name explicitly:

```
9-cell observation:
  residual_blob_uncompressed_bytes = 1,843,200 (50 pairs × 96 × 128 × 3 = identical at all cells)
  residual_blob_brotli_bytes       = 14 (identical at all cells, ratio = 7.6e-6)
  clamped_fraction_at_boundary     = 1.0000 (100% saturation at ALL 9 cells)
  raw_abs_residual p99             = 0.7759 / 0.7712 / 0.7576 (gain_clamp=0.05)
                                     0.9028 / 0.8990 / 0.8889 (gain_clamp=0.10)
                                     0.9595 / 0.9576 / 0.9527 (gain_clamp=0.20)
  raw_abs_residual p50             = ~0.56 / ~0.78 / ~0.91 (means ~MOST residuals are large too, not just tail)
```

**Mechanism chain** (frontier-push insight not in sweep memo):

1. The tanh-bounded residual learner outputs are saturated by the L2 loss landscape — the optimizer DRIVES residuals to ±tanh-asymptote.
2. The downstream `clip(±gain_clamp)` then truncates 100% of residuals to ±gain_clamp.
3. The int8 quantizer `round(residual_clamped / gain_clamp * 127)` → ±127 for 100% of elements (regardless of gain_clamp absolute value).
4. The result is a 1.84MB stream of ±127 bytes with low ENTROPY (sign-pattern only matters).
5. Brotli RLE+context-modeling collapses this to 14 bytes (essentially the dictionary header + sign-bitmap-context).

**This means**: the rate axis is NOT scale-invariant because of int8 quantization specifically. It is scale-invariant because **the trained residuals carry no per-pixel magnitude information** — all the information is in the SIGN. The canonical int8 quantizer + brotli pipeline preserves this fact, hence the 42B constant.

## Variant B design choice — analysis

Per operator-routable #1 (HIGHEST EV) cascade, three design alternatives were proposed:

### Variant B-a (signed-exponent encoding)

**Claim**: fp4/fp8 mantissa + exponent preserves magnitude across gain_clamp values.

**Empirical refutation**: residuals don't have wide magnitude variance (p50≈p99 at every gain_clamp). 100% saturation means magnitudes are nearly identical to gain_clamp. fp-exponent encoding has nothing to encode — it would still RLE-collapse.

**Verdict**: NOT-APPROPRIATE for this empirical regime.

### Variant B-b (gain_clamp-dependent bit-width)

**Claim**: int4 at gain_clamp=0.05; int8 at 0.10; int16 at 0.20. More bits → more savings space.

**Empirical refutation**: more bits with all-saturated distribution still RLE-compresses identically. Quantizing ±0.05 to int4 still produces ±7 uniformly; quantizing ±0.20 to int16 still produces ±32767 uniformly. Brotli RLE-collapses both to ~14B.

**Verdict**: NOT-APPROPRIATE for this empirical regime.

### Variant B-c (non-uniform / Lloyd-Max optimal quantization)

**Claim**: optimal quantizer levels per residual distribution shape.

**Empirical refutation**: optimal quantizer for uniform-saturated distribution is the same as uniform quantizer (saturated mode has 1 level). Lloyd-Max with K levels degenerates to K identical levels at the saturation point.

**Verdict**: NOT-APPROPRIATE for this empirical regime.

### Variant B-d (REFINED: explicit sign-bitmap codec) — SELECTED

**Frontier-push insight**: the empirical fact tells us all 100% of information is in the SIGN. Encode it directly.

**Design**:
1. Compute `sign_bitmap = (residual_clamped >= 0).astype(uint8)` → 1 bit per pixel.
2. Encode sign_bitmap as packed bits (8 pixels/byte; total = `N_PIXELS / 8`).
3. Encode per-pair magnitude as a single fp16 = gain_clamp (one scalar per pair).
4. Apply brotli quality 9 to the packed sign-bitmap (entropy depends on per-pair sign-pattern correlation).
5. Sidecar = BPR1 header + brotli(sign_bitmap) + per-pair_magnitudes_fp16.

**Predicted sidecar bytes** (Variant B-d):
- `sign_bitmap_raw_bytes` = 50 pairs × 96 × 128 × 3 / 8 = 230,400 bytes
- `sign_bitmap_brotli_bytes` ≈ depends on sign-pattern entropy; could be 1KB-50KB depending on spatial correlation
- `per_pair_magnitudes_fp16` = 50 × 2 = 100 bytes
- BPR1 header = 28 bytes
- **TOTAL** ≈ ~1.5KB-50KB depending on entropy (vs current 42B)

**Scale-invariance breaking**: Variant B-d sidecar bytes grow with **NUM_PIXELS** (1 bit/pixel). For a single fixture, this is FIXED across gain_clamp. **HOWEVER**, the sign-pattern entropy may differ with gain_clamp because the L2 loss landscape shape depends on gain_clamp (larger clamp → wider loss valley → trained sign-patterns may be more diverse → less brotli-compressible → MORE bytes).

**KEY HYPOTHESIS** (empirical question): does the trained sign-bitmap have higher entropy at larger gain_clamp? If yes → Variant B-d breaks scale-invariance. If no → Variant B-d sidecar bytes also constant across cells (different absolute value than 42B, but still gain_clamp-independent).

## Variant B design choice — SELECTED: Variant B-d (sign-bitmap + per-pair magnitude)

### Honest classification per Catalog #307 paradigm-vs-implementation

This design IS an IMPLEMENTATION-LEVEL response to the empirical falsification of Variants B-a/b/c. The PARADIGM (residual-correction-hybrid stacking) is unchanged. The design choice CARGO-CULTED assumption is "more flexible quantization → larger sidecar"; the HARD-EARNED-EMPIRICALLY-VERIFIED truth is "all information is in the sign". Variant B-d acknowledges this.

### Frontier-push canonical-vs-frontier-push decision per NEW pushing-the-frontier directive

- **Codec-design level**: FRONTIER-PUSH. Sign-bitmap encoding of trained residuals after saturation observation is a novel codec design specifically motivated by the EMPIRICAL sweep finding. No canonical literature directly cites "sign-bitmap codec for tanh-saturated residual learners"; this is original empirical-grounded design.
- **Brotli compression level**: CANON-APPLICATION. Brotli q9 is canonical (sister of L1 codec).
- **Per-pair magnitude encoding**: CANON-APPLICATION. fp16 scalar per pair is standard.

### Predicted outcomes (sweep)

| | epochs=30 | epochs=100 | epochs=300 |
|---|---|---|---|
| gain_clamp=0.05 | sidecar=?B (predicted ≥ 1KB) | ? | ? |
| gain_clamp=0.10 | ? | ? | ? |
| gain_clamp=0.20 | ? | ? | ? |

**Carmack-dissent verdict criteria** (per Catalog #307 sister):

- **IF Variant B-d sidecar bytes monotonically grow with gain_clamp**: SCALE-INVARIANCE BROKEN; canonical equation #347 predicate VALIDATED at Variant B-d codec surface; anchor_appended event registered.
- **IF Variant B-d sidecar bytes still constant across gain_clamp**: Variant B-d IMPLEMENTATION-LEVEL FALSIFIED (PARADIGM unchanged); canonical equation #347 predicate STILL FALSIFIED at this codec class; DEFER to Variant C (next subagent: training-side mitigation that prevents tanh saturation, e.g. softer-than-tanh activation OR magnitude-preserving residual loss).
- **IF Variant B-d sidecar bytes GROW with gain_clamp BUT distortion reduction does NOT**: net-negative trade-off; canonical equation predicate FALSIFIED in a different way (rate cost without distortion benefit).

## Sweep scope

**Identical 9-cell grid** (mirrors sister gain_clamp sweep for direct comparability):

| | epochs=30 | epochs=100 | epochs=300 |
|---|---|---|---|
| gain_clamp=0.05 | NEW | NEW | NEW |
| gain_clamp=0.10 | NEW | NEW | NEW |
| gain_clamp=0.20 | NEW | NEW | NEW |

**Identical fixture**: 50 pairs × 96×128 × NHWC float32; seeded RNG=42; AdamW β₁=0.9 β₂=0.999 lr=1e-3; tanh-bounded residual head; canonical compose; fp32 throughout.

**Variant B-d sidecar build difference**: replace `build_bpr1_sidecar` per-pair int8 quantize + brotli with:
1. `build_bpr1_sidecar_variant_b_d`:
   - For each pair: compute `residual_clamped`; extract `sign_bitmap = (residual_clamped >= 0).astype(uint8)`.
   - Pack 8 sign bits per byte via numpy `packbits`.
   - Compute per-pair magnitude scalar = gain_clamp (one fp16 per pair).
   - Concatenate all sign-bitmap bytes; brotli q9 compress.
   - Concatenate magnitudes (NUM_PAIRS × 2 bytes).
   - Sidecar = BPR1 header (28B) + len_packed_signs(4B) + len_brotli_signs(4B) + brotli_signs + magnitudes_fp16

**Estimated wallclock**: 9 × ~3s = ~30s (matches sister sweep; reuses identical training; only sidecar build differs).

## Drift surface declaration per NEW MLX↔CUDA bidirectional drift directive

Same as sister sweep memo (identical fp32 throughout / MLX defaults / NHWC / tanh+clip ordering / AdamW defaults / brotli q9 determinism). NEW surface added: `numpy.packbits` byte-order. Predicted PORTABILITY: identical between numpy CPU and any CUDA sister (packbits is canonical big-endian per numpy spec); zero drift surface introduced.

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: ACTIVE — per-pair sign-bitmap entropy + per-pair p99 magnitude feeds sister `tac.sensitivity_map.*`.
2. **Pareto constraint**: ACTIVE — Variant B-d gives a NEW (gain_clamp, sidecar_bytes) Pareto front to compare with L1 (constant 42B).
3. **Bit-allocator hook**: N/A at this sweep (1-bit-per-pixel uniform allocation).
4. **Cathedral autopilot dispatch hook**: ACTIVE — Variant B-d results JSON will carry canonical Provenance per Catalog #323 + `axis_tag=[macOS-MLX research-signal]` + non-promotable markers per Catalog #341.
5. **Continual-learning posterior update**: ACTIVE — Variant B-d anchors extend BoostNeRV-PR110 evidence; appended to canonical equation #347 via `tac.canonical_equations.update_equation_with_empirical_anchor`.
6. **Probe-disambiguator**: ACTIVE — Variant B-d sweep IS the canonical operator-routable #1 disambiguator probe per Catalog #313 between "scale-invariance is a CODEC-design artifact (Variant B-d should break it)" vs "scale-invariance is a TRAINING-DYNAMICS artifact (Variant C would be needed)".

## HORIZON-CLASS verdict per Catalog #309

`frontier_pursuit` (same as sister sweep; predicted PLATEAU-ADJACENT band; codec-redesign is mechanism investigation not scoring-floor pursuit).

## Operator-routable next steps (post-landing)

1. **IF Variant B-d breaks scale-invariance** + sidecar growth tracks meaningful information content: PROPOSED for inclusion in T3 PR110-stacking ordering memo as BoostNeRV-PR110 candidate #6 (already operator-routable #3 of sister sweep memo).
2. **IF Variant B-d still scale-invariant**: DEFER to Variant C (training-side fix); per CLAUDE.md "Forbidden premature KILL without research exhaustion".
3. **IF Variant B-d compression too aggressive** (e.g. ≥10KB sidecar without corresponding distortion benefit): IMPLEMENTATION-LEVEL trade-off; needs analytical balance.

## Cross-references

- Sister gain_clamp sweep landing memo: `.omx/research/boostnerv_pr110_gain_clamp_sweep_landed_20260526.md` (commit `8240aceda`)
- Sister gain_clamp sweep heatmap JSON: `.omx/research/boostnerv_pr110_gain_clamp_sweep_results_20260526/sweep_heatmap.json`
- Sister L1 EMPIRICAL landing memo: `.omx/research/boostnerv_pr110_l1_empirical_landed_20260526.md` (commit `b2fd3e587`)
- Canonical equation #347 registration script: `tools/register_2_new_canonical_equations_20260526.py` (commit `04f34ea40`)
- Sweep harness script (Variant B-d): `.omx/tmp/boostnerv_pr110_bpr1_variant_b_sweep.py` (NEW; same commit batch)
- NEW Variant B-d codec module: `src/tac/substrates/boost_nerv_pr110_residual/bpr1_variant_b_sign_bitmap_codec.py` (NEW; same commit batch)
- MLX↔CUDA bidirectional drift directive: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md`
- Pushing-the-frontier directive: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_pushing_the_frontier_of_research_on_optimization_algorithms_standing_directive_20260526.md`
