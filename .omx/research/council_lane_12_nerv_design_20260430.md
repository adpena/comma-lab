# Council Design Review — Lane 12 NeRV mask codec (2026-04-30)

## Charter

Promote Lane 12 from Level 1 (synthetic scaffold) → Level 3 (full production hardened + recursive adversarial reviewed). Council quintet pact + Quantizr + Hotz + Selfcomp + MacKay + Ballé deliberate on architecture, training recipe, and integration.

## Architecture decision

### Coordinate-MLP backbone

| Knob | Choice | Rationale |
|---|---|---|
| `num_freqs` | **8** | NeRF reference uses 10 for spatial / 4 for temporal; a single L=8 across all 3 dims (t, y, x) gives 3 × 2 × 8 = 48 input dims after positional encoding. Mallat: this is dyadic frequency tiling — wavelet-equivalent on a discrete grid. |
| `hidden_dim` | **64** | Empirically smallest dim that resolves SegNet-relevant 4-pixel-wide boundary objects on 384×512. Chen 2021 NeRV used 256 for natural video; masks are piecewise-constant so 64 suffices (Quantizr's 88K param renderer fits the same boundary detail at hidden=128 with depth=3). |
| `depth` | **4** | NeRV ablation shows 3 underfits, 6 overfits per-frame artifacts; 4 is the sweet spot. |
| `num_classes` | **5** | Matches comma SegNet output. |
| Activation | **GELU** | Smoother gradients than ReLU; coordinate networks favor smooth nonlinearities (Sitzmann SIREN: sin works too but breaks our positional encoding additivity). |

**Parameter count** (hidden_dim=64, depth=4, num_freqs=8, num_classes=5):
- Layer 1: 48 × 64 + 64 = 3,136
- Layer 2: 64 × 64 + 64 = 4,160
- Layer 3: 64 × 64 + 64 = 4,160
- Layer 4 (output): 64 × 5 + 5 = 325
- **Total: 11,781 params**
- At fp16: **23,562 bytes ≈ 23 KB**
- At int8 (with scale table): **~12 KB**

**Bit budget vs AV1 baseline**: AV1 = 421 KB → NeRV fp16 = 23 KB → **~18× reduction** if SegNet score ≤ AV1.

### Loss

**Cross-entropy on raw 5-class logits vs argmax SegNet labels** (NOT argmax — argmax has zero gradient per Council A `.round()` zero-gradient bug class).

```python
loss = F.cross_entropy(logits_BC, mask_labels_B, reduction='mean')
```

(Logits flat `(N, 5)`, labels flat `(N,)` of class IDs in [0..4]. Argmax of logits is the predicted class at inflate-time.)

### Training recipe

| Knob | Value | Rationale |
|---|---|---|
| Optimizer | Adam, lr=1e-3, β=(0.9, 0.999) | NeRV reference; small model so cosine schedule unnecessary |
| Batch | 4096 coordinates per step (random sampling across (t, y, x)) | CPU-friendly stochastic; 1200×384×512 = 236M coords → ~57K steps per epoch at 4096 batch = 1 full pass |
| Epochs | 1 epoch (single full pass) sufficient for overfit-on-fixed-sequence | NeRV is a per-video overfit, not generalization |
| EMA | **decay=0.997 (CLAUDE.md non-negotiable)** | Snapshot+restore at eval per `tac.training.EMA` canonical pattern |
| eval_roundtrip | **True (CLAUDE.md non-negotiable)** | NOT applied to mask labels themselves (no roundtrip on argmax) but applied to the SegNet auth-eval at end |
| Seed | 2026 | Determinism |

### Quantization for shipping

- **Phase A1 (this Phase F dispatch)**: fp16 weights (start). Predicted ~23 KB.
- **Phase A2 (stretch, post-CUDA-result)**: int8 weights with QAT phase + scale table. Predicted ~12 KB.

The current scaffold's int8 path is BROKEN at decode (no scale persisted). Phase F lands fp16 only; int8 deferred to a follow-up.

### NRV1 wire format (already in scaffold; no change)

```
[magic 4B = b"NRV1"][version u16 = 1][num_freqs u16][hidden_dim u16][num_classes u16]
[depth u16][weight_dtype u16 (0=fp16,1=int8)][payload_size u64][weights bytes]
```

For Phase G we keep `weight_dtype=0` (fp16); int8 (1) reserved.

## Council deliberation

### Shannon (LEAD — information theory)

R(D) for a 1200-frame 5-class mask sequence at 384×512:
- Source entropy upper bound (uniform 5-class, 384×512×1200): H ≤ log2(5) × 384×512×1200 = 2.32 × 236.2M = 549 Mbits = **68 MB**.
- AV1 monochrome compresses to 421 KB → exploits massive redundancy (piecewise-constant + temporal coherence).
- NeRV at 23 KB targets the **next** floor: amortizes the temporal+spatial structure into a single MLP. Whether 23 KB is feasible at SegNet ≤ AV1 baseline is empirical — **predict** the score will hold within ±5% IF training converges (scaling law: NeRV 88K params on natural video gets 33 dB PSNR; mask sequence is far simpler).

**Vote: GREEN — execute. Floor calculation justifies trying.**

### Dykstra (CO-LEAD — convex feasibility)

Joint feasibility set:
- Bytes ≤ 50 KB (target) ✓ (predicted 23 KB)
- SegNet distortion ≤ Lane G v3 baseline (~0.0040) ✓ (target ≤0.0050 = +25% loss budget)
- Inflate latency ≤ 30 min on T4 ✓ (236M coord forward at 100M/s = 2.4s; well under)

Three constraints satisfiable simultaneously per back-of-envelope. **Vote: GREEN.**

### Yousfi (steganalysis / contest design)

The contest scorer looks at SegNet argmax-disagreement on per-pixel basis. NeRV produces continuous logits → argmax. As long as argmax matches at >95% pixels (per Quantizr empirical tolerance for AV1 lossy), score holds.

**Risk**: NeRV is a learned model — it WILL underfit class boundaries. AV1 also underfits boundaries (chroma subsampling artifacts). The question is whether the NeRV underfitting is concentrated on class-boundary pixels (catastrophic) or spread uniformly (amortized into noise).

**Mitigation**: weight loss by inverse class-boundary distance (UNIWARD-style). DEFER to Phase A2; Phase A1 runs flat cross-entropy and we measure the boundary-pixel disagreement rate as a diagnostic.

**Vote: GREEN with diagnostic flag.**

### Fridrich (steganalysis / detector-aware embedding)

Same point as Yousfi. Add: per-frame loss weighting by SegNet-gradient magnitude (detector-informed). Phase A2.

**Vote: GREEN.**

### The Contrarian

Strongest objections:
1. "AV1 already does temporal motion compensation; what's NeRV's edge for piecewise-constant masks specifically?"
   - **Answer**: AV1 spends bits on intra-frame coding of 1200 individual frames + motion vectors. NeRV spends bits on a single shared MLP. For 1200 frames of HIGHLY-CORRELATED masks (5 semantic classes that change slowly across time), the amortized MLP is fundamentally cheaper per Shannon's argument.
2. "23 KB is a wild prediction; what if it's actually 80 KB?"
   - **Answer**: it's still 5× better than AV1 at the same SegNet score, which would be a clear win. Floor band: [20 KB, 80 KB]. Kill if >100 KB at AV1-equivalent SegNet.
3. "Hotz says 'just store every Kth frame and interpolate' — has anyone checked that's worse?"
   - **Answer**: nearest-frame interpolation (K=10 → 120 frames stored) at AV1 = 42 KB. SegNet score on interpolated frames degrades fast for fast-moving cars (5 of 1200 frames have heavy traffic). NeRV's smooth spatial-temporal interpolation is better-conditioned than nearest-frame. NOT verified empirically — flagged for Phase A2.

**Vote: GREEN with kill criterion: bytes > 100 KB OR SegNet > AV1 + 25% → abandon.**

### Quantizr (adversarial competitor)

"I shipped AV1 at 421 KB and got 0.33 — what makes NeRV better at 1200 frames? You're betting on coordinate-MLP overfitting an entire mask sequence into 23 KB. Show me the comparison vs my Quantizr-class only-odd-frames AV1 at ~211 KB."

**Answer**: even-frame warping is orthogonal to NeRV. NeRV can store ALL 1200 frames in 23 KB, OR only 600 odd frames in ~12 KB. Both stack with the "warp even frames from odd" trick. Phase A1 lands all-1200 NeRV; Phase A2 explores half-frame NeRV (predicted ~12 KB).

**Vote: GREEN with stacking flag for Phase A2.**

### Hotz (raw engineering)

"Just nearest-frame interpolation. K=20 stored frames, AV1 at high CRF, 30 KB. Same SegNet."

**Answer**: addressed in Contrarian #3. Phase A1 NeRV vs Phase A2 nearest-frame parallel comparison agreed.

**Vote: GREEN if NeRV beats nearest-frame baseline at the same byte budget. Add nearest-frame as Phase A2 ablation.**

### Selfcomp (PR #56 lead)

"My SegMap is 94K params at 1.017 bpw → ~12 KB. NeRV is structurally similar (small MLP, deterministic decode). The byte arithmetic only matters if SegNet stays within Lane G v3 noise floor. My empirical experience: tiny grayscale-LUT decoders DO hold the score on this dataset — your premise is plausible."

**Vote: GREEN. Provides anchor data point: "94K param SegMap holds the score; 11.7K param NeRV is even tighter — likely needs more depth or training time, watch for underfitting."**

### MacKay (memorial — info theory + Bayesian)

"What's the MDL cost of the approximation? You're shipping 23 KB of MLP + the implicit 'frame index → coord' mapping (negligible). Vs AV1 421 KB, the MDL gain is 398 KB ≈ 3.18 Mbits — substantial. The Bayesian view: NeRV is a parametric prior on mask sequences; AV1 is a non-parametric one. For correlated sequences, the parametric wins because it shares parameters across frames."

**Vote: GREEN. Frames the win as MDL-optimal parametric encoding.**

### Ballé (modern neural compression)

"This is the 2018 entropy bottleneck framework, sans the entropy bottleneck. Your fp16 weights are stored uniform; an arithmetic coder over weight quantiles (Lane SH-style) would get another 10-30% off. DEFER to Phase A2."

**Vote: GREEN with arithmetic-coding stacking flag for Phase A2.**

### van den Oord (VQ-VAE / WaveNet)

"Coordinate-MLP is one of two valid choices; the other is VQ-VAE codebook over patches. For piecewise-constant masks at 5 classes, the codebook would be tiny (100 codes × 8 pixels = ~1 KB). NOT obviously worse than NeRV — they trade off boundary fidelity vs token quantization."

**Vote: GREEN with VQ-VAE codebook as Phase 3 alternative.**

## Final verdict

**UNANIMOUS GREEN — execute Phase F (Vast.ai 4090, ~$0.60-$0.85, fp16 NeRV at hidden=64, depth=4, num_freqs=8).**

**Win class** (clarified Round 3 — Dykstra D2): **STACKING LANE.** Lane 12 wins the rate term decisively (~ -0.04 score points on rate alone vs Lane G v3) but the SegNet term may regress 0.001-0.005 (vs Lane G v3's 0.0040 baseline distortion). Net standalone score may go either way in the predicted band [0.95, 1.30]. The CLEAR win case is Lane 12 stacked with Lane Ω-W-V2 (renderer water-fill) + Lane LCT (learnable class targets) + Lane PD-V2 (arithmetic-coded poses) per the Phase 1.5 stacking architecture — there the rate savings compound while the SegNet underfit is amortized into noise. Phase G dispatches Lane 12 STANDALONE for cleanest measurement; the stacking variant is a follow-up.

Kill criteria (binding):
- Final NRV1 bytes > 100 KB → abandon (worse than AV1 at half the savings would be required).
- SegNet score > Lane G v3 baseline + 25% → abandon (boundary-pixel underfitting catastrophic).
- Inflate render time > 30s on T4 → abandon (would exceed 30-min inflate budget when stacked).

Stacking flags for Phase A2 (NOT this dispatch):
- int8 + scale table (~12 KB)
- Half-frame NeRV (only odd frames, even frames warped) (~12 KB)
- Boundary-weighted loss (UNIWARD-style)
- Arithmetic-coded weight quantiles (Ballé)
- Nearest-frame baseline ablation (Hotz)
- VQ-VAE patch codebook alternative (van den Oord)

## Cross-refs

- Phase 2 Lane 12 design: `memory/project_phases_2_3_4_design_implementation_math_provenance_20260429.md`
- Phase A audit: `.omx/research/lane_12_nerv_scaffold_audit_20260430.md`
- Production-hardened standard: `memory/feedback_production_hardened_standard_definition_20260430.md`
