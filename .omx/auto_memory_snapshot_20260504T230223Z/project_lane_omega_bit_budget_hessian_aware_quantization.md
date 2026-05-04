---
name: Lane Ω — bit-budget Hessian-aware per-weight quantization driven by hard-pair contribution
description: Long-form design recalled by user 2026-04-27. Per-WEIGHT (not per-channel) bit allocation: profile ∂L/∂w on hard pairs → water-fill a fixed total bit budget → variable-length encoding → QAT. Extension of Lane W (per-channel hard-pair weighting) to per-weight resolution. Could recover 30-40% MORE rate at same distortion by sparing only the ~5% critical weights.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**The design we synthesized over multiple sessions** (referenced by user 2026-04-27):

The hierarchy of quantization resolution:
1. **Uniform**: every weight in the model gets the same bit-depth (FP4 = 4 bits everywhere). Lane F's approach. PROBLEM: 20× PoseNet penalty because critical weights lose precision identically to bulk weights.
2. **Per-layer**: each layer has its own bit-depth. Quantizr's approach (5-stage QAT with layer-specific scales). PROBLEM: still uniform within a layer.
3. **Per-channel**: each output channel has learnable bit-depth (Self-Compression, Lane S). PROGRESS: SC layers reach 2 bits avg without distortion penalty per `project_self_compression_breakthrough`. BUT still uniform within a channel.
4. **Per-channel + hard-pair weighted** (Lane W): per-channel SC bit-depth gradient is upweighted on hard pairs → SC steers bits to channels critical for the heavy tail. PROGRESS: protects the worst-pair channels.
5. **Per-weight + hard-pair Hessian** (Lane Ω, this design): each individual weight gets its own bit-depth based on ∂L/∂w² aggregated over hard pairs. NEW: per-weight resolution should recover the redundancy WITHIN each channel.

**Lane Ω algorithm**:

```
Phase 1: Profile (cheap, ~10 min)
  - Load Lane A renderer + Lane W's hard pair profile (top-K hardest)
  - For each weight w_ij in eligible layers (excluding protected: head, motion, FiLM, fuse_conv):
    - Compute Fisher I(w_ij) = mean(|∂L/∂w_ij|²) over hard pairs
  - Save I_tensor.pt: shape matches model state_dict, per-element importance

Phase 2: Water-Fill Allocation (instant)
  - Given target total bit budget B (e.g., 600_000 bits ≈ 75KB)
  - Sort weights by I descending
  - Greedily assign bits[w] = clamp(round(c × I(w)^α), 1, 8) where:
    - α controls allocation steepness (e.g., 0.5)
    - c chosen so Σ bits[w] = B
  - Save bits_per_weight.pt

Phase 3: Constrained QAT (~30 min - 2h)
  - Fine-tune model with FrozenBitFakeQuant(weight, bits[w_ij]) per weight
  - Use eval_roundtrip + KL distill + Lane A as init
  - Lr=2.5e-6 (low to preserve Lane A quality)
  - 200-500 epochs

Phase 4: Variable-Length Export
  - Pack weights with per-element bit-depth metadata
  - Use bit-packed storage format (similar to Lane S's SCv1 but per-weight)
  - Compute total bytes used, verify ≤ B/8

Phase 5: Inflate-Time Decode
  - Mirror the bit-packed format in inflate_renderer.py
  - Decode per-element bit-depth + value, reconstruct weight tensor
  - No scorer load (strict-scorer-rule compliant)
```

**Predicted EV**:
- Lane Ω at B=600,000 bits (75KB renderer.bin): predicted score band [0.70, 1.05]
- Lane Ω at B=480,000 bits (60KB renderer.bin): predicted [0.80, 1.20] (more aggressive)
- Lane Ω at B=300,000 bits (37KB renderer.bin = Quantizr's claimed FP4 size): predicted [1.10, 1.50] (probably hurts pose floor)

**Why this could work where Lane F failed**:
Lane F-V2's 20× PoseNet penalty came from quantizing critical FiLM-adjacent weights identically to bulk weights. Lane Ω measures per-weight importance and SPARES the critical 1-5% of weights at FP8 quality while crushing the 60-80% truly redundant weights to 1-2 bits. The total bit budget can then be MUCH lower than uniform FP4 because most weights are unimportant.

**Implementation complexity**:
- Phase 1 (profiler): ~50 lines, similar to Lane W's per-pair profiler but per-weight
- Phase 2 (water-fill): ~30 lines
- Phase 3 (QAT with frozen per-weight bits): MEDIUM — needs FakeQuantPerWeightBit autograd module (~80 lines)
- Phase 4 (variable-length export): MEDIUM — needs bit-packing format design (~150 lines)
- Phase 5 (inflate decode): MEDIUM — mirror format (~100 lines)
- Tests: ~200 lines

Total: ~600-700 lines new code. NOT trivial. Defer until Lane W lands and validates the per-pair-importance signal is real.

**Comparison to literature**:
- HAWQ (Hessian-Aware Quantization, Dong et al 2019): similar per-layer Hessian-based bit allocation
- OBQ (Optimal Brain Quantization, Frantar 2022): per-weight via inverse Hessian
- BRECQ (Block Reconstruction Quantization): block-level
- Lane Ω novelty: HARD-PAIR-WEIGHTED Hessian (not full-dataset). Adapts to the contest's per-pair score sensitivity rather than dataset-average.

**When to dispatch Lane Ω**:
1. AFTER Lane W (#a03807360be034810 in flight 2026-04-27) lands and beats Lane A. If Lane W beats 1.15, Lane Ω becomes high-EV.
2. If Lane W LOSES, the per-pair importance signal isn't strong enough → Lane Ω probably also loses → DON'T DISPATCH.
3. If Lane S beats Lane A (the underlying SC mechanism works), but Lane W doesn't add much (per-channel hard-pair weighting saturates at low gain), Lane Ω is the natural next step at finer resolution.

**Council voices on Lane Ω**:
- Yousfi: "Hessian-aware quantization is the literature-canonical answer to 'which weights matter most'. Combining with hard-pair sensitivity is the contest-specific extension."
- Fridrich: "Per-weight bit allocation IS the inverse-steganalysis principle at the finest granularity. Detector-informed embedding at the parameter level."
- Hotz: "OBQ-style per-weight Hessian + your variable-length encoding = a real research contribution. If Lane W shows hard-pair signal, this is the natural unlock."
- Quantizr: "I never tried per-weight bit allocation — it's complex. If you make it work, you've definitively beaten my approach."
- Contrarian: "600 lines of new code is high risk. Watch out for: (1) the bit-packed format is hard to debug, (2) per-weight QAT is slow (each gradient pass touches per-element bit-depth), (3) the inflate-time decode adds runtime cost (under contest budget?). Validate Lane W FIRST as a cheaper proxy for whether the hard-pair signal is real."

**Status**: Recalled by user 2026-04-27. Filed for later dispatch. PREREQUISITE: Lane W must land + validate hard-pair signal.

**Related memories**:
- `project_self_compression_breakthrough` — 2-bit deep conv improves score; FiLM 3rd most sensitive
- `project_lane_w_hard_pair_self_compress_premise_20260427` — per-channel hard-pair weighted (precursor)
- `project_5stage_quantization_advantage` — our 5-stage QAT stack
- `project_research_survey_20260420` — LoRA, Cool-chic, DSConv research synthesis
- `feedback_overfit_is_the_goal` — single video memorization
- `feedback_posenet_tracking` — heavy tail dominates score
