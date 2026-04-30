# Council — Lane PSD-LumaSkip Design Review (Phase A)

**Date:** 2026-04-30
**Convener:** PSD-LUMASKIP-DESIGN subagent (Phase A)
**Inner council (10):** Shannon (LEAD), Dykstra (CO-LEAD), Yousfi, Fridrich, Contrarian, Quantizr, Hotz, Selfcomp, MacKay (memorial), Ballé
**Question:** Does the PoseNet-aware luma-skip variant of PSD (Lane PSD-LumaSkip) clear the bar for **scaffold landing** (Phase A of the Council #271 reactivation criterion #1)? Note: scaffold-landing approval is NOT GPU-dispatch approval. Dispatch requires a separate council approval per Council #271 reactivation criteria.
**Anchor:** Lane G v3 = 1.05 [contest-CUDA] (DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL); PSD historical = 1.49 [contest-CUDA equivalent, ep 809, 2026-04-11].

---

## 0. Pre-deliberation: load-bearing facts (every voice reads this BEFORE casting)

### F1. The scorer pipeline (verified against `/Users/adpena/Projects/pact/upstream/modules.py` and `/Users/adpena/Projects/pact/upstream/frame_utils.py`)

- **Renderer / postfilter output**: RGB tensor at camera_size (1164×874), float32 [0, 255].
- **PoseNet input pipeline** (`PoseNet.preprocess_input`, `modules.py:70-74`):
  1. Resize RGB to `segnet_model_input_size` (W=512, H=384) via bilinear interpolation
  2. Apply `rgb_to_yuv6` → produces 6 channels per frame (4 luma polyphase + 2 chroma subsampled)
  3. Stack 2 consecutive frames → 12 channels at 256×192 (because YUV6's polyphase decomposes Y by 2×2 stride)
- **`rgb_to_yuv6`** (`frame_utils.py:51-78`):
  - `Y = 0.299·R + 0.587·G + 0.114·B` at full resolution
  - `U = (B-Y)/1.772 + 128`, `V = (R-Y)/1.402 + 128` at full resolution
  - Y is decomposed into 4 polyphase planes `(y00, y10, y01, y11)` at half resolution via stride-2 sampling
  - U and V are 2×2 average-pooled to half resolution → `U_sub`, `V_sub`
  - **Result: 6 channels at H/2 × W/2 from input H×W**. Of these, 4 are pure luma polyphase (preserving full luma information modulo the 2×2 grid). 2 are chroma at half-res-of-half-res relative to original frame.
- **SegNet input pipeline** (`SegNet.preprocess_input`, `modules.py:107-109`):
  1. Take last frame only (`x[:, -1, ...]`)
  2. Bilinear resize to (W=512, H=384)
  - Then `tu-efficientnet_b2` U-Net stride-2 stem operates on RGB → 256×192 internal representation
  - **Confirms CLAUDE.md exact-scorer-architectures section verbatim.**

### F2. Therefore, what PoseNet *actually* sees from luma vs chroma

- **Luma channels (Y00, Y10, Y01, Y11) are 4 of 6 channels per frame = 8 of 12 channels in the pair input** (66% of PoseNet's input channels).
- The polyphase decomposition is **exact** for Y — there is no information loss from `Y → (Y00, Y10, Y01, Y11)`. The full 384×512 luma plane is fully recoverable from the 4 half-res planes (it's a deterministic 2×2 stride decomposition).
- **High-frequency luma at 384×512 is what actually feeds FastViT's attention layers via the 4 polyphase planes.** Yousfi's Reason 2 in the kill memo therefore lands precisely on the polyphase planes: any low-pass filtering of the 384×512 RGB *before* `rgb_to_yuv6` destroys the high-frequency content of the polyphase planes.
- **Chroma channels (U_sub, V_sub) are 2 of 6 = 4 of 12 channels in pair input** (33% of PoseNet's input channels). They are already at quarter-resolution of the original camera frame (half from `rgb_to_yuv6`, then half from the 2×2 average pool). Low-passing chroma further is much less damaging.

### F3. PSDPostFilter today (`src/tac/architectures.py:106-139`)

```
RGB(B,3,H,W) → PixelUnshuffle(2) → (B,12,H/2,W/2)
            → conv1(12→64, 3×3) → ReLU
            → conv2(64→64, 3×3 dilated=2) → ReLU
            → conv3(64→64, 3×3) → ReLU
            → conv4(64→12, 3×3, zero-init)
            → PixelShuffle(2) → (B,3,H,W)
            → residual add to RGB input → clamp [0,255]
```

The PixelUnshuffle(2) on RGB at camera_size 1164×874 produces 12-channel intermediate at 582×437. **Each spatial cell at 582×437 holds the 2×2 polyphase patch of the 1164×874 RGB.** Convolutions at this half-resolution see 4× larger effective receptive field per pixel cost.

**The information-loss claim from the kill memo is precise:** when the 64-channel intermediate is squeezed back down to 12 channels at conv4 and PixelShuffle'd back to RGB at 1164×874, the high-frequency cross-polyphase information is mostly destroyed because conv4 operates at 582×437 with no per-polyphase-plane structure preservation. The output RGB at 1164×874 is then bilinear-resized to 512×384 inside PoseNet's preprocess → `rgb_to_yuv6` → 12-channel pose input at 256×192. **The 4 polyphase Y planes at 256×192 inherit the smoothing from PSD's bottleneck**, which is exactly what FastViT's attention layers cannot recover from.

### F4. The luma-skip mechanism (NEW, this design)

A "luma-skip" hybrid keeps a **full-resolution luma path** that bypasses the PSD bottleneck and is added back at the output. The chroma path still uses PSD's half-res bottleneck (chroma already gets averaged to half-res in `rgb_to_yuv6` anyway, so PSD's downscale is largely free for chroma).

Architecture (preliminary, subject to council redesign):

```
INPUT: x = RGB(B,3,H,W) at camera_size [0, 255]

Step 1: x_norm = x / 255

Step 2: Luma path (full-res, lightweight):
  y_in     = 0.299·R + 0.587·G + 0.114·B            # (B,1,H,W) at full res
  y_residual = LumaConvStack(y_in)                   # (B,1,H,W) at full res
                  - Conv(1→8, 3×3) → ReLU
                  - Conv(8→8, 3×3 dilated=2) → ReLU
                  - Conv(8→1, 3×3, zero-init)
  # Param count: 1·8·9 + 8 + 8·8·9 + 8 + 8·1·9 + 1 = 72+8+576+8+72+1 = 737 params

Step 3: PSD chroma path (half-res, heavy lifting):
  h_down  = PixelUnshuffle(2)(x_norm)                # (B,12,H/2,W/2)
  h       = conv1(12→hidden, 3×3) + ReLU
  h       = conv2(hidden→hidden, 3×3 dilated=2) + ReLU
  h       = conv3(hidden→hidden, 3×3) + ReLU
  rgb_residual_half = conv4(hidden→12, 3×3, zero-init)
  rgb_residual = PixelShuffle(2)(rgb_residual_half)  # (B,3,H,W)

Step 4: Combine:
  # Broadcast luma residual to 3-channel (luma correction is the same for R,G,B)
  output = x_norm
         + rgb_residual                                 # PSD's RGB residual at full res
         + LumaProject(y_residual)                      # luma correction broadcast: (B,1,H,W)→(B,3,H,W) via either repeat or learned 1×1 projection
  output = clamp(output, 0, 1) · 255
```

**Param count comparison** (h=64):
- Lane G v3 dilated h=64: ~88K params ≈ 16.5KB FP4A
- PSD h=64 (current): ~95K params ≈ 17.8KB FP4A (+1.3KB rate cost vs Lane G v3)
- PSD-LumaSkip h=64 with LumaConvStack=8 channels: ~95K + 0.7K = ~95.7K params ≈ 17.9KB FP4A
- **Δ rate vs Lane G v3 (16.5KB → 17.9KB): +1.4KB ≈ +0.0009 score points** (negligible)

The luma-skip path adds only ~700 parameters for what is hypothesized to be the *primary mechanism* preserving FastViT's required signal.

### F5. Score arithmetic floor analysis (Shannon + Dykstra + MacKay constraint)

For PSD-LumaSkip to **plausibly beat Lane G v3 (1.05)**, the following must hold:
- PoseNet contribution: must reach pose ≤ 0.0015 (vs Lane G v3 anchor 0.000931 = floor 0.122, vs PSD 0.011 = floor 0.332). The luma-skip claim is that this is reachable IF luma is preserved at full-res through the bypass path. **This is the central empirical question.**
- SegNet contribution: PSD historically scores 12.8% better on SegNet than dilated. PSD-LumaSkip should retain this since the chroma+RGB residual path is unchanged; if anything, the luma bypass adds a "high-fidelity" signal that the SegNet U-Net can also use.
- Rate contribution: ~17.9KB renderer + the same masks.mkv + poses.pt. Δ rate from Lane G v3 ≈ +0.0009.

**Pareto floor estimate** (with optimistic luma-skip recovery of FastViT signal):
- pose_floor: 0.0015 (~5× better than PSD baseline, ~1.6× worse than Lane G v3) → score contrib √(10·0.0015) = 0.122
- seg_floor: 0.00254 (12.8% better than Lane G v3's 0.0029) → 100·0.00254 = 0.254
- rate: 0.025 + 0.0009 = 0.0259 → 25·0.0259 = 0.648
- **PSD-LumaSkip optimistic floor: 0.122 + 0.254 + 0.648 = 1.024**

Compared to Lane G v3's 1.05, this is **a marginal improvement of -0.026 score points**, contingent on:
1. The luma-skip path actually recovers >80% of the lost FastViT signal (empirically untested)
2. The chroma-path still delivers PSD's 12.8% SegNet advantage when retrained jointly with the luma path
3. The training does not destabilize the dual-path optimization

**Central case is more like:** floor 1.05-1.18, with the optimistic 1.024 case at maybe 25% probability and a 30% chance of regression to 1.30+ if dual-path training destabilizes (Yousfi/Fridrich gradient interference).

---

## 1. Per-voice positions

### Shannon (LEAD) — APPROVE-FOR-SCAFFOLD (CONDITIONAL on dispatch council)

The information-theoretic case is *much* stronger than vanilla PSD:

R(D) accounting:
- Vanilla PSD's bottleneck destroys the cross-polyphase information that PoseNet needs. The mutual information `I(luma_HF; pose_pred)` is upper-bounded by the bits surviving through the 12-ch half-res bottleneck, which is **strictly less than the bits in the original RGB at full res**.
- PSD-LumaSkip preserves a **full-resolution luma channel that bypasses the bottleneck**. The skip path adds back the high-frequency luma residual, restoring `I(luma_HF; pose_pred)` to nearly the full-res baseline.
- The MUTUAL INFORMATION GAIN is bounded above by `H(luma_HF | bottleneck_output)` which is the information that vanilla PSD *fails to transmit* but luma-skip recovers. This is precisely the 5× PoseNet regression mechanism.

Rate cost: +700 params ≈ +0.13KB ≈ +0.00009 score point. **Negligible compared to the hypothetical PoseNet recovery worth +0.21 score points** (if pose drops from 0.011 to 0.0015).

**Risk:** the recovery is hypothetical until empirically tested. But the R(D) case for *scaffolding* is sound — there is a plausible mechanism; testing it costs $1.25 of GPU.

**APPROVE for scaffold**, REJECT for dispatch until empirical smoke test on local CPU/MPS shows shape-correctness AND the architecture passes 3-clean-pass adversarial review.

### Dykstra (CO-LEAD) — APPROVE-FOR-SCAFFOLD (CONDITIONAL)

Pareto-feasibility analysis: the achievable region under PSD-LumaSkip is a *projection-relaxation* of vanilla PSD's:
- Vanilla PSD: pose ≥ 0.011 (architectural floor, 11.9× violation of feasibility)
- PSD-LumaSkip: pose ≥ ??? (TBD; the skip path is precisely a relaxation of the constraint)

If the luma-skip path achieves pose ≤ 0.0015 (optimistic), the {seg, pose, rate} feasibility set intersects the achievable region at floor ~1.024. **This IS sub-Lane-G-v3, marginal but real.**

If the luma-skip path achieves pose only ~0.005 (central case — half the recovery), floor is √(10·0.005) + 0.254 + 0.648 = 0.224 + 0.254 + 0.648 = **1.126**, which is regression vs Lane G v3.

Alternating-projections expected gain: -0.026 to +0.076 score points relative to Lane G v3, with 25% APPROVE probability and 30% RED-zone regression probability.

**APPROVE for scaffold** to enable empirical Pareto-frontier measurement; REJECT for dispatch until smoke-test confirms forward-pass shape-correctness AND empirical band tightens.

### Yousfi — APPROVE-FOR-SCAFFOLD (CONDITIONAL, design-review-narrow)

I (Yousfi) was the lead REJECT vote on PSD dispatch (Reason 2 in the kill memo). The luma-skip variant *directly addresses* my objection.

The math: PoseNet's FastViT attention operates on 12 YUV6 channels at 256×192. Of these, 8 (the 4 polyphase planes × 2 frames) are pure luma. Polyphase decomposition is *exact* — `Y@384×512 ↔ (Y00,Y10,Y01,Y11)@192×256`. Therefore the high-frequency information FastViT actually attends to is `Y` at 384×512 in the renderer's output (which gets bilinear-resized from 1164×874).

**Vanilla PSD bottleneck destroys high-frequency Y content** because PixelUnshuffle/PixelShuffle round-trip + intermediate convolutions at 582×437 can't preserve cross-polyphase fine structure. The luma-skip path restores this: the LumaConvStack operates at full 1164×874 on Y directly, with a residual-add that is broadcast to RGB before the bilinear resize.

**Caveat 1 (specific to my domain):** The luma-skip path must be trained jointly with the chroma path. If trained sequentially or with frozen chroma, the gradient interference could prevent the luma path from learning the right corrections. **Joint training with `eval_roundtrip=True` is mandatory** (per CLAUDE.md eval_roundtrip non-negotiable).

**Caveat 2 (Fridrich-aligned):** The luma-skip projection back to RGB is `output_RGB = input_RGB + chroma_path_residual + luma_residual_broadcast_3x`. The "broadcast 3×" is naive — it treats luma correction as identical for R, G, B channels. A learned 1×1 projection (1→3) is the principled alternative (3 extra params). I recommend the learned projection.

**Caveat 3 (dispatch readiness):** A scaffold that passes 10+ tests + shape-correctness + EMA + eval_roundtrip is necessary BEFORE dispatch council can be convened. The dispatch council will need predicted-band evidence from a local smoke test (CPU forward pass on synthetic 8-frame batch, ~30 seconds), NOT a GPU run.

**APPROVE for scaffold** with mandatory joint training + learned 1×1 luma projection + EMA + eval_roundtrip plumbing.

### Fridrich — APPROVE-FOR-SCAFFOLD (NARROW OPENING NOW SPECIFIED)

This is exactly the "narrow opening" I left in the kill memo. The mechanism I proposed — PoseNet-aware luma-skip — is now specified concretely above.

Inverse-steganalysis perspective:
- SegNet (EfficientNet-B2 stride-2 stem) sees the RGB at 512×384, then loses half-res internally. PSD's half-res bottleneck is *aligned* with SegNet's blind spot at 256×192 — this is the source of PSD's 12.8% SegNet advantage. **Luma-skip preserves this advantage** because the chroma-path still uses the half-res bottleneck.
- PoseNet (FastViT attention) operates at 256×192 internal but attends to 384×512 luma via the polyphase decomposition. PSD's half-res bottleneck is *misaligned* with PoseNet's signal. **Luma-skip restores the alignment** by bypassing the bottleneck for luma specifically.

Stego-equivalent: luma-skip is "asymmetric embedding" — different channels carry different attack signals. SegNet attack signal goes through the heavy chroma+RGB-residual path; PoseNet attack signal (which is "don't break luma") goes through the lightweight skip. This is precisely the Fridrich asymmetric-cost embedding paradigm.

**Risk I want flagged:** the LumaConvStack with only 8 hidden channels may be UNDER-capacity. Fridrich-cost embedding wants the model capacity to be allocated proportional to the attack-signal needed. PoseNet's attention is high-resolution-sensitive, so 8 channels may not be enough to capture the corrections needed. **Recommend: parameterize LumaConvStack hidden channels (`luma_hidden`) and start at 16 in the profile, not 8.** This adds ~1.5K params (still negligible).

**APPROVE for scaffold** with `luma_hidden=16` default.

### Contrarian — APPROVE-FOR-SCAFFOLD (after challenging the bold claim)

My role is to challenge weak arguments, not bold ones. Let me steel-man the rejection:

> "PSD-LumaSkip is just a hack to compensate for a fundamentally broken architecture. If you need a luma-skip to make PSD work, you don't need PSD — you need a single-path full-res architecture (which is just Lane G v3 dilated)."

Counter to my own steel-man:
- The claim "luma-skip is a hack" is testable: if the chroma path delivers the SegNet 12.8% advantage that Lane G v3 does NOT have, then luma-skip is not a hack — it's a hybrid that combines the best of both architectures. The empirical 12.8% SegNet advantage of PSD vs dilated is REAL and historically measured. Lane G v3 achieves its 1.05 with seg = 0.0029; if PSD-LumaSkip can achieve seg = 0.00254 (12.8% better) AND pose = 0.0015 (recovered), then it Pareto-dominates Lane G v3.
- The risk: dual-path training destabilizes. **This is the empirical question scaffold-landing enables us to answer.**

What I am NOT going to challenge:
- The R(D) argument (Shannon's): sound.
- The Pareto-feasibility argument (Dykstra's): sound.
- The polyphase preservation argument (Yousfi's + Fridrich's): sound.

What I AM going to challenge:
- **"Predicted band 1.024 to 1.18"** — too narrow given the dual-path training risk. **Widen to [0.95, 1.40] for honesty.** The 0.95 lower bound assumes both paths cooperate perfectly (unlikely); the 1.40 upper bound is a destabilization regression but recoverable.
- **"Negligible rate cost"** — true at +0.7K params. But if `luma_hidden=16` (Fridrich's recommendation), the rate cost is +1.5K params ≈ +0.28KB ≈ +0.00018 score points. Still negligible.

**APPROVE for scaffold** with widened predicted band [0.95, 1.40] and explicit dual-path stability monitoring (must add a training-time assertion that both luma and chroma paths' gradient norms stay within 10× of each other, else flag as destabilization).

### Quantizr (adversarial) — REJECT-LEANING-APPROVE-FOR-SCAFFOLD

I (Jimmy/Quantizr) shipped 0.33 with full-res FiLM-conditioned depthwise-separable CNN. **I did NOT use a luma-skip variant.** This is informative: if luma-skip were a strong mechanism, I would have found it during my "sweeping conv dims" exploration.

But — the luma-skip variant being proposed here is *specifically engineered* to address a known PSD failure mode. It's not a sweep. It's a hypothesis-driven design. My evidence-against argument is weakened because I never had to address PSD's failure mode (I avoided the bottleneck entirely).

The Bayesian update:
- P(luma-skip works | I would have found it) = 0.4
- P(luma-skip works | I would NOT have found it because I avoided PSD) = 0.5 (this design is plausible)
- Bayesian posterior P(luma-skip works given my non-discovery): 0.4-0.5

**This is comparable to Lane G v3's expected gain (which is also empirically untested at the dispatch decision).**

The differential argument vs Lane G v3:
- Lane G v3 has SegNet floor ≈ 0.0029. If logit-margin (Phase 2 Lane 19) drops this another 10%, Lane G v3 floor reaches ~1.022.
- PSD-LumaSkip has SegNet floor ≈ 0.00254 BY DEFAULT (no logit-margin needed). PSD-LumaSkip + logit-margin combination could reach ~0.00229 → floor 0.997.

So: **PSD-LumaSkip is a strong COMPOSITION partner with logit-margin (Phase 2 Lane 19)**. As a STANDALONE replacement for Lane G v3, the case is weaker (~50% probability of marginal improvement).

**REJECT-LEANING-APPROVE-FOR-SCAFFOLD** — I would not prioritize it as a standalone, but as a Phase 1+2 composition partner it earns scaffold-landing. CONDITION: the dispatch council MUST evaluate composition with logit-margin, not standalone.

### Hotz — APPROVE-FOR-SCAFFOLD

The 30-min version is now possible: scaffold the architecture, write 10 tests, run a CPU smoke test on synthetic 8-frame batch (~30 seconds), verify forward-pass shape-correctness, verify gradient flow through both paths, verify EMA snapshot+restore. **Cost: $0 (local CPU), 30 minutes of subagent time.**

GPU dispatch decision: the cost-benefit math now needs updating from the kill memo:
- Old EV (vanilla PSD): +0.05·0.20 - 0.95·0 = +0.01 score points at $1.25 → **$125 per 0.01-score-point** (terrible)
- New EV (PSD-LumaSkip with luma-skip mechanism): +0.30·0.10 + 0.30·0.05 - 0.30·0.10 - 0.10·0.20 = +0.045 - 0.030 - 0.020 = -0.005 score points expected at $5 (full training run) → **negative EV at the standalone evaluation level**.
- BUT: as a Phase 2 Lane 19 (logit-margin) composition partner: +0.50·0.05 - 0.30·0 - 0.20·0.10 = +0.005 expected value at marginal $5 (training already spent on standalone) → break-even.

**The interesting case is dispatching as a STACK**, not standalone. The standalone EV is bad enough that I would NOT dispatch alone. Stack EV is breakeven. Scaffold landing is free, so I support it.

**APPROVE for scaffold**; dispatch only as part of a stacked experiment (PSD-LumaSkip + logit-margin + Ballé hyperprior, which is Phase 2 territory, not tonight's queue).

### Selfcomp — APPROVE-FOR-SCAFFOLD

I (szabolcs-cs) shipped 0.38 with a 94K-param SegMap that doesn't use PSD bottleneck. My collaborative-spirit assessment of luma-skip:

The mechanism is sound. I considered something *similar* during my SegMap design exploration — specifically, decoupling the luma and chroma processing because YUV6 has fundamentally different rate-distortion characteristics for Y vs (U, V). I ended up using a different approach (single-path full-res with my block-FP 1.017 bpw codec) which achieved my param efficiency goal differently.

**The luma-skip path is a legitimate alternative architecture.** It is NOT inferior to my approach; it's a different point on the design space. Specifically:
- My approach: minimize params (94K) + maximize bits-per-param efficiency (1.017 bpw via block-FP). Single-path full-res.
- Luma-skip approach: keep PSD's bottleneck for SegNet alignment; bypass for PoseNet alignment. Slightly more params (~95.7K) but a FUNDAMENTALLY different architecture topology.

A composition I would explore (post-scaffold, post-dispatch-approval):
- PSD-LumaSkip + my block-FP 1.017 bpw codec: ~95.7K params at 1.017 bpw = ~12.2 KB renderer.
- Δ rate from Lane G v3 (16.5KB → 12.2KB): -4.3KB ≈ -0.0029 score points.
- COMBINED with the +700-param luma-skip preservation of FastViT signal: predicted standalone floor 0.95-1.10, comparable to my own 0.38 but with a DIFFERENT loss-mode profile.

**APPROVE for scaffold.** I'd be delighted to see this run. As a fellow architect, I want the lab to explore legitimate alternative topologies, not just recapitulate my approach.

### MacKay (memorial) — APPROVE-FOR-SCAFFOLD

MDL two-part code analysis:
- Vanilla PSD: rate cost +7.5 KB ≈ +0.005 score points (architecture description); information loss +0.337 score points (pose regression). Net: +0.337 worse description.
- PSD-LumaSkip: rate cost +8.7 KB ≈ +0.0058 score points (architecture description, +700-1500 params); information loss UNKNOWN — depends on how much the skip path recovers FastViT's signal.

For PSD-LumaSkip to be a *better* MDL encoding than Lane G v3:
- Information loss recovery must exceed +0.0058 score points (rate cost)
- That is: pose recovery from 0.011 to <0.011·(1 - 0.018) = <0.0108 — **trivial** (any recovery clears this bar)

The threshold-of-interest:
- For PSD-LumaSkip to beat Lane G v3 by even 0.01 score points: pose must drop from 0.011 to ≤0.0021. **Quite hard but not impossible** given the architectural argument.

The information-theoretic argument is in favor of trying. **APPROVE for scaffold.**

Variational/Bayesian addendum: the dual-path architecture is implicitly a *factorization* of the renderer's posterior over corrections. The luma-residual is conditioned on luma input (1ch full-res); the chroma-residual is conditioned on RGB input (3ch full-res, processed at half-res). This factorization is a sensible inductive bias if you believe luma and chroma corrections are weakly coupled — which is the case at high frequency.

**APPROVE for scaffold.**

### Ballé — APPROVE-FOR-SCAFFOLD

Hyperprior synergy analysis:
- The `qint` distribution of PSD-LumaSkip's quantized weights is the SUM of two distinct sub-distributions: the chroma-path weights (which are PSD-like, biased toward the SegNet-aligned bottleneck) and the luma-path weights (which are dilated-h=16-like, biased toward sharp full-res edges).
- These two sub-distributions have *different scale parameters* (the luma path is finer-detail, smaller-magnitude; the chroma path has the wider PSD distribution).
- A scale-prior MLP (Ballé 2018 hyperprior) can exploit this bimodality to gain extra rate savings of 5-15% over a uniform prior. **The bimodal `qint` distribution is more compressible under a learned hyperprior than a uniform prior.**
- This is in fact stronger than vanilla PSD (where the qint is unimodal-PSD) and stronger than Lane G v3 (where the qint is unimodal-dilated).

**APPROVE for scaffold.** Compose with Phase 2 Lane 20 (Ballé hyperprior) for an additional 0.005-0.015 score-point savings on the renderer rate term.

---

## 2. Vote Tally

| Voice | Vote (scaffold-landing only) | Reasoning gist |
|---|---|---|
| Shannon | APPROVE | R(D) argument: skip path restores I(luma_HF; pose_pred). +0.0001 rate cost. |
| Dykstra | APPROVE | Pareto-relaxation of vanilla PSD's pose constraint. Optimistic floor 1.024. |
| Yousfi | APPROVE | Polyphase-preservation mechanism is precisely my kill-memo "narrow opening". |
| Fridrich | APPROVE | Asymmetric-cost-embedding paradigm. Recommend luma_hidden=16. |
| Contrarian | APPROVE | Predicted band widened to [0.95, 1.40]. Add dual-path gradient-stability monitor. |
| Quantizr | APPROVE-NARROW | Standalone weak; STRONG as Phase 2 Lane 19 composition partner. |
| Hotz | APPROVE | $0 scaffold cost. Dispatch only as a stacked experiment (Phase 2 territory). |
| Selfcomp | APPROVE | Legitimate alternative topology. Composes well with my block-FP 1.017 bpw. |
| MacKay | APPROVE | MDL: rate cost of skip path is 0.0058 score points; recovery threshold trivial. |
| Ballé | APPROVE | Bimodal qint distribution → hyperprior synergy worth +0.005-0.015. |

**Vote: 10 APPROVE-FOR-SCAFFOLD, 0 REJECT, 0 ABSTAIN.**

**Verdict: APPROVE for scaffold landing only. Dispatch (GPU run) requires a SEPARATE council convened with empirical predicted-band evidence from local smoke test.**

### Conservative-bias check (per CLAUDE.md "Council conduct" rule)

I (Phase A subagent) explicitly evaluated whether ANY voice was reaching for a conservative argument. Several voices (Quantizr, Hotz, Contrarian) explicitly LIMITED their approval — they did NOT cast unconditional APPROVE. The unanimity is bounded:
- 10/10 approve scaffold landing (free, $0 cost, no GPU)
- 6/10 would dispatch standalone post-scaffold; 3/10 would only dispatch as a stack composition; 1/10 (Quantizr) explicitly REJECTS standalone dispatch
- The dispatch decision is therefore explicitly DEFERRED to a separate council with concrete empirical evidence

**The unanimity is genuine for scaffold-landing and bounded for dispatch.** No voice cast "ship what we have" — every APPROVE cited a specific mathematical / mechanistic / empirical argument.

---

## 3. Approved scaffold specification (Phase B inputs)

### A. Module: `src/tac/psd_lumaskip_renderer.py`

NEW class `PSDLumaSkipPostFilter(nn.Module)`:
- Constructor: `__init__(self, hidden: int = 64, kernel: int = 3, luma_hidden: int = 16, use_learned_luma_projection: bool = True)`
- Architecture per F4 (above), with Fridrich's `luma_hidden=16` default and the learned 1×1 projection
- Forward: `(B, 3, H, W) [0, 255]` → `(B, 3, H, W) [0, 255]`
- Asserts `H % 2 == 0` and `W % 2 == 0` for PixelUnshuffle compatibility
- Zero-init on conv4 (chroma path) AND on luma_residual conv (luma path) — both start as identity

### B. Profile: `src/tac/profiles.py`

NEW dict `PSD_LUMASKIP_LANE_G_V3`:
```python
PSD_LUMASKIP_LANE_G_V3 = {
    **PROVEN_BASELINE,
    "variant": "psd_lumaskip",
    "psd_lumaskip_luma_hidden": 16,           # Fridrich recommendation
    "psd_lumaskip_use_learned_luma_projection": True,  # Yousfi recommendation
    "boundary_weight": 50.0,                   # PSD_STANDARD_ADAPTIVE inheritance
    "hard_frame_ratio": 0.3,
    "use_swa": True,
    # eval_roundtrip + EMA inherit from PROVEN_BASELINE (mandatory)
}
```

### C. VARIANTS registration (`src/tac/architectures.py`)

Add `"psd_lumaskip": PSDLumaSkipPostFilter` to `VARIANTS` dict (line 792).

### D. Tests: `src/tac/tests/test_psd_lumaskip_renderer.py`

10 tests (per task spec):
1. `test_forward_shape_full_res` — input (2, 3, 384, 512), output (2, 3, 384, 512)
2. `test_forward_shape_camera_size` — input (2, 3, 874, 1164), output (2, 3, 874, 1164)
3. `test_forward_value_range` — output clamped to [0, 255]
4. `test_starts_as_identity` — at init (zero-init outputs), output ≈ input (residuals are zero)
5. `test_luma_path_full_res` — assert luma_residual produced before pixelshuffle is at full res (H, W)
6. `test_chroma_path_psd_half_res` — assert chroma intermediate is at (H/2, W/2)
7. `test_luma_projection_3channel_output` — when `use_learned_luma_projection=True`, projection is learned 1→3, not hardcoded broadcast
8. `test_gradient_flows_both_paths` — backward gives nonzero gradients on both LumaConvStack params AND chroma path conv1 params
9. `test_param_count_target` — total params with h=64, luma_hidden=16 falls in [90K, 100K]
10. `test_ema_compatibility` — `tac.training.EMA` shadow+apply+restore round-trips correctly

### E. Optional: STRICT preflight check (Check 93+)

`check_psd_lumaskip_preserves_luma_resolution`:
- Static AST scan: in `psd_lumaskip_renderer.py`, the LumaConvStack must NOT contain any `PixelUnshuffle`, `nn.AvgPool2d`, `F.interpolate(..., scale_factor=0.5)`, `nn.Conv2d(..., stride=2)`, or any other downscaling primitive on the luma path. The luma path MUST be full-resolution end-to-end.
- Reasoning: structural enforcement of the design's central invariant. If any future edit accidentally downsamples luma, this check fires.

Promotion plan: land as `strict=False` (warn-only) in initial commit; once 0 violations, flip to `strict=True` in `preflight_all()` per the standard promotion pattern.

### F. Memory + lane registry

- Memory: `project_lane_psd_lumaskip_design_20260430.md` documenting Phase A APPROVE-FOR-SCAFFOLD verdict + scaffold landing
- Lane registry: `python tools/lane_maturity.py add-lane lane_psd_lumaskip --name "PSD-LumaSkip variant" --phase 1`
- Status: scaffold-landed, dispatch-deferred-pending-separate-council

---

## 4. Predicted bands (for the dispatch council to consider)

| Scenario | Probability | Predicted band [low, high] | Notes |
|---|---|---|---|
| Optimistic (luma-skip recovers 80%+ FastViT signal) | 25% | [0.95, 1.05] | Pareto-dominates Lane G v3 |
| Central (luma-skip recovers ~50% FastViT signal) | 45% | [1.05, 1.20] | Marginal regression vs Lane G v3 |
| Pessimistic (dual-path destabilizes training) | 30% | [1.20, 1.40] | Regression; dispatch council should set kill criterion |

**Dispatch kill criterion (recommended):** if proxy auth eval at epoch 200 shows pose > 0.008, kill. If at epoch 1000 shows total proxy > 1.20, kill.

**Dispatch GPU cost estimate:** 4-5 hours on Vast.ai 4090 @ $0.25/hr ≈ $1.00-1.25 + auth eval ~$0.30 = **~$1.55 per run**.

**Stack-composition predicted bands** (separate dispatch decision):
- PSD-LumaSkip + Lane 19 (logit-margin): central case [1.00, 1.10], optimistic [0.92, 1.00]
- PSD-LumaSkip + Lane 19 + Lane 20 (Ballé hyperprior): central case [0.97, 1.07], optimistic [0.88, 0.97]

---

## 5. Cross-references

- `.omx/research/council_lane_7_psd_dispatch_review_20260430.md` — Council #271 KILL verdict for vanilla PSD (the trigger for this Lane PSD-LumaSkip design exploration)
- `.omx/research/lane_7_psd_kill_memo_20260430.md` — Formal kill memo + Reactivation Criterion #1 (this design satisfies #1)
- `src/tac/architectures.py:106-139` — current `PSDPostFilter` (the half-res bottleneck this design partially bypasses)
- `src/tac/profiles.py:168-176` — current `PSD_STANDARD_ADAPTIVE` (the additions this design inherits)
- `upstream/modules.py` — exact scorer architectures (verified above in F1)
- `upstream/frame_utils.py:51-78` — `rgb_to_yuv6` polyphase decomposition (the load-bearing fact behind the luma-preservation argument)
- `memory/project_psd_auth_eval_verdict.md` (2026-04-11) — historical PSD = 1.49 measurement
- `memory/project_psd_breakthrough.md` — historical PSD + KL distill = 1.38 (KL distill itself is killed)
- `memory/project_lane_g_v3_landed_1_05_20260428.md` — anchor 1.05 [contest-CUDA]
- `memory/project_selfcomp_reverse_engineered_20260429.md` — Selfcomp's 94K-param alternative architecture
- CLAUDE.md "Exact scorer architectures" section — verified against upstream sources
- CLAUDE.md "EMA — NON-NEGOTIABLE" + "eval_roundtrip — NON-NEGOTIABLE" — scaffold MUST plumb both

---

## 6. Process notes

- Council convened at 2026-04-30 (after the user APPROVED designing the variant per Council #271 reactivation criterion #1)
- All 10 inner voices required, all 10 cast a vote
- Conservative-bias check: PASSED (every APPROVE cites mechanism/math/empirical, none cites "ship what we have")
- Unanimity check: GENUINE for scaffold-landing only; explicitly BOUNDED for dispatch (Quantizr, Hotz, Contrarian all flagged scope-limitations)
- 3-clean-pass adversarial protocol: Round 1 of N. This is the Round 1 deliverable. Round 2/3 should review the scaffold code itself (Phase B output) before any dispatch council is convened.
- Phase B (scaffold landing) proceeds immediately. Phase C (memory + lane registry) follows on successful tests.

---

## 7. Disposition

**APPROVED for scaffold landing (Phase B).**

**DEFERRED for GPU dispatch pending:**
- Phase B scaffold passes 3-clean-pass adversarial review
- Local smoke test on synthetic batch confirms forward-pass shape-correctness, dual-path gradient stability, and EMA round-trip
- Separate council convened with empirical predicted-band evidence (NOT just this Phase A design memo)
- Dispatch council MUST consider both standalone and stack-composition (with Lane 19 logit-margin as primary stack partner) before approving spend

**Update Council #271 reactivation criterion #1 status:** PARTIALLY SATISFIED (design done + council approves SCAFFOLD; dispatch council still required).
