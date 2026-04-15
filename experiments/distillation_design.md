# Renderer Distillation: Experiment Design

## Council Endorsement

**Tripartite pact (Yousfi + Fridrich + Contrarian):** Approved. Distillation is the only remaining path to improve contest-compliant score (renderer-only, no scorer at inflate time). The v5a TTO result (auth 0.43) proves the target frames exist; the question is whether a 287K-param feedforward network can reproduce them.

**Hotz:** The capacity question is the crux. 287K params for 600 pairs = 478 params/pair. Each pair is 384x512x3x2 = 1.18M pixels. The compression ratio is 2,470:1. TTO gets to spend 500 gradient steps per pair; the renderer gets zero. This is fundamentally a distillation capacity problem.

**Quantizr:** Competitor mask2mask achieves 0.60 auth with ~308K FP4 params. Our current 287K achieves 0.87. The gap is not capacity but training signal quality. Distillation replaces noisy scorer-mediated gradients with direct pixel supervision from TTO frames, which are already scorer-optimal.

## 1. Problem Statement

**Current state:**
- Best renderer-only auth score: 0.87 (v5, Lagrangian annealing, 287K params)
- Best TTO auth score: 0.43 (v5a, gradient fix, 500 steps/pair)
- Gap: 0.44 score points
- Competitor (mask2mask): 0.60 auth with similar architecture

**Goal:** Train the renderer to produce frames as close to TTO quality as possible, without any scorer or gradient computation at inflate time.

**Constraint:** The only inputs at inflate time are the masks (from archive) and the renderer weights (from archive). No scorers, no GT frames, no gradient steps.

## 2. TTO Frame Acquisition

### 2.1 Source

TTO frames from the v5a run: 1200 frames at 384x512, stored as `tto_frames.pt` (float32, ~900MB) on the Modal volume.

Per-pair TTO metrics (from the v5a run log):
- Mean PoseNet distortion: 0.00263
- Mean SegNet distortion: 0.00148
- These are the distillation targets

### 2.2 Quality verification

Before distillation, verify TTO frame quality:
1. Run auth eval on TTO frames directly (should reproduce 0.43 +/- 0.02)
2. Compute per-pair distortion breakdown (600 values each for PoseNet and SegNet)
3. Identify the hard-pair distribution from Section 4 of the hard-pair analysis
4. Verify no degenerate pairs (NaN, all-black, all-white)

## 3. Distillation Loss Design

### 3.1 Baseline: MSE against TTO frames

$$\mathcal{L}_\text{MSE} = \frac{1}{N}\sum_{k}\|G_\theta(m_{2k}, m_{2k+1}) - x_\text{TTO}^{(k)}\|_2^2$$

where $G_\theta$ is the renderer and $x_\text{TTO}^{(k)}$ is the TTO pair.

**Concern (Fridrich):** TTO frames contain codec-specific artifacts from the 500-step optimization. MSE would force the renderer to reproduce these artifacts, even when they are not scorer-relevant. The renderer should match TTO frames *in scorer space*, not pixel space.

### 3.2 Scorer-guided distillation (recommended)

At compress time we CAN use scorers. The loss combines pixel regression with scorer feedback:

$$\mathcal{L} = \lambda_\text{pixel} \cdot \mathcal{L}_\text{pixel} + \lambda_\text{pose} \cdot \mathcal{L}_\text{pose} + \lambda_\text{seg} \cdot \mathcal{L}_\text{seg}$$

where:

- $\mathcal{L}_\text{pixel} = \frac{1}{N}\sum_k \|G_\theta(m^{(k)}) - x_\text{TTO}^{(k)}\|_1$ (L1, more robust than L2 to outlier pixels)
- $\mathcal{L}_\text{pose} = d_\text{pose}(G_\theta(m^{(k)}), \text{GT}^{(k)})$ (PoseNet distortion against GT, not TTO)
- $\mathcal{L}_\text{seg} = d_\text{seg}(G_\theta(m^{(k)}), \text{GT}^{(k)})$ (SegNet distortion against GT, not TTO)

**Rationale:** The pixel term provides a warm-start toward TTO quality. The scorer terms ensure the renderer doesn't just memorize TTO artifacts but learns the scorer-relevant structure.

### 3.3 Perceptual distillation (experimental)

Replace the pixel term with a feature-space loss using PoseNet's intermediate activations:

$$\mathcal{L}_\text{percept} = \sum_l \|F_l(G_\theta(m^{(k)})) - F_l(x_\text{TTO}^{(k)})\|_2^2$$

where $F_l$ extracts activations at layer $l$ of the frozen PoseNet. This focuses on features PoseNet actually uses, ignoring pixel-level differences that don't affect the score.

**Status:** Experimental. Feature extraction hooks need implementation. Prioritize 3.2 first.

## 4. Training Pipeline

### Phase 1: Pixel regression warm-start (epochs 0--2000)

- **Loss:** $\mathcal{L}_\text{pixel}$ only (L1 against TTO frames)
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4
- **Batch size:** 8 pairs (16 frames)
- **Data:** All 600 pairs, random sampling
- **Purpose:** Fast convergence to a reasonable initialization. Expected pixel L1 < 10 by epoch 2000.
- **No scorer forward passes needed** -- pure regression, very fast.
- **Estimated time:** ~20 min on RTX 4090 (600 pairs x 2000 epochs / 8 batch = 150K iters, ~8ms/iter)

### Phase 2: Scorer-guided fine-tuning (epochs 2000--6000)

- **Loss:** $\lambda_\text{pixel} \cdot \mathcal{L}_\text{pixel} + \lambda_\text{pose} \cdot \mathcal{L}_\text{pose} + \lambda_\text{seg} \cdot \mathcal{L}_\text{seg}$
- **Weights:** $\lambda_\text{pixel} = 0.1$, $\lambda_\text{pose} = 1.0$, $\lambda_\text{seg} = 100.0$ (matching scoring formula ratio)
- **Optimizer:** AdamW, lr=3e-4, cosine decay to 1e-5
- **Batch size:** 4 pairs (scorer forward/backward is VRAM-intensive)
- **Scorer setup:** Frozen PoseNet + SegNet, differentiable preprocessing via `make_scorers_differentiable()`
- **Estimated time:** ~2 hours on RTX 4090 (scorer backward is ~50ms/pair)

### Phase 3: Lagrangian annealing (epochs 6000--10000)

- **Loss:** Augmented Lagrangian as in v5, but initialized from Phase 2 checkpoint
- **Constraint targets:** $\epsilon_\text{pose} = 0.005$, $\epsilon_\text{seg} = 0.003$ (slightly above TTO levels to allow exploration)
- **Lagrangian caps:** Start at $\lambda_\text{cap} = 1000$, anneal to $\lambda_\text{cap} = 100$ at epoch 8000, snapshot transient improvement, re-tighten to 1000 at epoch 9000
- **Purpose:** The proven Lagrangian annealing protocol that produced the 0.87 result, but starting from a much better initialization (Phase 2 checkpoint)
- **Estimated time:** ~3 hours on RTX 4090

### Phase 4: Hard-pair fine-tuning (epochs 10000--12000)

- **Loss:** Same as Phase 2, but training only on the hardest 20% of pairs (120 pairs)
- **Purpose:** Allocate final gradient budget to the pairs where the renderer struggles most
- **Pair selection:** Use the difficulty prior $h^{(k)}$ from compress-time analysis
- **Estimated time:** ~30 min on RTX 4090 (fewer pairs per epoch)

## 5. Architecture Considerations

### 5.1 Current architecture: 287K params

| Component | Params | Notes |
|-----------|--------|-------|
| MaskRenderer (shared emb + U-Net) | 199K | base_ch=36, mid_ch=60, embed_dim=6 |
| MotionPredictor (U-Net) | 88K | hidden=32, output_channels=6 |
| **Total** | **287K** | FP4 archive: ~150KB |

### 5.2 Capacity analysis

**Shannon bound argument (advisory):** The TTO frames contain ~1.18M pixels per pair x 600 pairs = 708M pixel values. At 8 bits/pixel, the uncompressed information is ~708MB. The renderer must compress this into 287K float32 params = 1.1MB, or 150KB at FP4. The compression ratio is 708M / 150K = 4,720:1. But the renderer doesn't need to reproduce the TTO frames exactly --- it needs to reproduce them up to scorer sensitivity. The effective information content is much lower because PoseNet operates at 192x256 (1/4 resolution) and SegNet has only 5 output classes.

**Effective degrees of freedom in scorer space:**
- PoseNet: 6 floats per pair x 600 pairs = 3,600 scalar values
- SegNet: 384 x 512 x 5 logits per pair x 600 pairs, but argmax reduces this to ~20K boundary pixels per frame
- Total effective DoF: ~3,600 + 600 x 20K = ~12M effective scalar constraints

With 287K params, the renderer has 287K / 12M = 0.024 params per constraint. This is tight but not impossible --- the constraints are highly structured (spatial smoothness, class coherence).

### 5.3 Should we increase capacity?

**Option A: depth=2 (current depth=1)**
- Adds ~163K params (total ~450K). FP4 archive: ~225KB.
- Rate cost: $(225 - 150) / (1200 \times 384 \times 512 \times 3) \times 25 \approx 0.008$ score points
- Worth it if depth=2 reduces distortion by more than 0.008 points

**Option B: Increase motion_hidden from 32 to 48**
- Adds ~42K params (total ~329K). FP4 archive: ~165KB.
- Specifically improves flow prediction, which is the bottleneck for PoseNet

**Option C: Increase base_ch from 36 to 48**
- Adds ~130K params (total ~417K). FP4 archive: ~210KB.
- Improves rendering quality uniformly

**Council verdict (Youssi + Fridrich + Contrarian):** Start with current 287K architecture. If Phase 2 plateaus with pixel L1 > 5 or scorer distortion > 2x TTO levels, sweep depth and base_ch. The rate cost of extra params is small compared to the potential distortion gain.

**Hotz:** Agree. But if we need more capacity, increase motion_hidden first. The rendering head already produces decent frames (SegNet is ~0.003 at TTO level). The bottleneck is PoseNet, which depends entirely on inter-frame flow quality. Flow needs more params before rendering does.

## 6. Pre-computed Metadata Specification

### 6.1 What to store in the archive

| Data | Size (raw) | Size (FP16) | Purpose |
|------|-----------|-------------|---------|
| Renderer weights (FP4) | 287K x 4B | ~150KB | Frame generation |
| Affine flow params (FP4) | 600 x 6 x 4B | ~1.8KB | Per-pair ego-flow |
| Mask video (AV1) | - | ~240B | Semantic conditioning |
| **Pair difficulty scores** | 600 x 4B | **1.2KB** | Inflate-time pair conditioning |
| **Per-pair render hints** | 60 x 16 x 2B | **1.9KB** | Hard-pair conditioning vectors |
| **Total new metadata** | - | **~3.1KB** | Negligible rate cost |

### 6.2 How the renderer uses metadata at inflate time

**Pair difficulty scores** (600 floats): Not used directly by the renderer forward pass. Used at compress time for training loss weighting and hard-pair selection. At inflate time, they inform which pairs to spend more effort on if any iterative refinement is available.

**Per-pair render hints** (60 hard pairs x 16-dim vectors): Injected as additional conditioning into the renderer for the 60 hardest pairs. Implementation:

```python
# In AsymmetricPairGenerator.forward():
if pair_hint is not None:
    # pair_hint: (B, 16) -> broadcast to (B, 16, 1, 1) -> tile to spatial dims
    # Add to stem features via a learned projection (16 -> base_ch)
    hint_feat = self.hint_proj(pair_hint.unsqueeze(-1).unsqueeze(-1))  # (B, base_ch, 1, 1)
    stem = stem + hint_feat.expand_as(stem)
```

The hint projection is a 16 x 36 matrix = 576 params, negligible.

### 6.3 Format

All metadata is packed into `archive.zip` alongside existing data:
- `renderer.bin`: FP4 quantized weights
- `masks.av1`: entropy-coded segmentation masks
- `affine_flow.bin`: 600 x 6 FP4 affine parameters
- `pair_meta.bin` (NEW): 600 x 4B difficulty scores + 60 x 16 x 2B hard-pair hints

## 7. Success / Kill / Concern Criteria

### Success criteria
- Auth score < 0.60 (matching or beating mask2mask competitor)
- PoseNet distortion < 0.01 (3x improvement over current 0.03)
- SegNet distortion < 0.005 (2x improvement over current 0.01)
- Archive size < 200KB (rate contribution < 0.15)

### Kill criteria
- After Phase 2 (6000 epochs), auth score > 0.80 (less than 8% improvement over baseline)
- Pixel L1 against TTO frames > 15 at Phase 1 end (renderer cannot approximate TTO)
- PoseNet distortion increases during Phase 2 (scorer guidance backfiring)

### Concern criteria (investigate, don't kill)
- Phase 2 PoseNet and SegNet gradients conflict (PCGrad needed)
- Hard-pair fine-tuning degrades easy-pair performance (overfitting to hard set)
- FP4 quantization degrades distillation quality (switch to FP8 and accept rate cost)

## 8. Resource Estimates

| Phase | GPU | Time | Cost |
|-------|-----|------|------|
| Phase 1 (pixel regression) | RTX 4090 | 20 min | $0.08 |
| Phase 2 (scorer-guided) | RTX 4090 | 2 hours | $0.50 |
| Phase 3 (Lagrangian) | RTX 4090 | 3 hours | $0.75 |
| Phase 4 (hard-pair) | RTX 4090 | 30 min | $0.13 |
| Auth eval (3 checkpoints) | T4 (Modal) | 1 hour | $0.59 |
| **Total** | - | **~7 hours** | **$2.05** |

Budget-compatible with $24 Vast.ai hard cap. Total experiment cost is ~$2.

## 9. Comparison with Alternative Approaches

| Approach | Expected auth | Archive cost | Inflate time | Risk |
|----------|--------------|-------------|-------------|------|
| Current renderer (v5) | 0.87 | 150KB | <1s | Baseline |
| **Distillation (this proposal)** | **0.55--0.65** | **155KB** | **<1s** | **Medium** |
| TTO at inflate (forbidden) | 0.43 | 150KB | ~15 min | N/A (rule violation) |
| Larger renderer (2M params) | 0.60--0.70 | ~250KB | <1s | Rate cost |

## 10. Implementation Checklist

1. [ ] Download TTO frames from Modal volume
2. [ ] Verify TTO frame quality (auth eval reproduction)
3. [ ] Compute per-pair difficulty priors (Section 4 of hard-pair analysis)
4. [ ] Implement distillation training script (`experiments/train_distill.py`)
5. [ ] Phase 1: pixel regression (20 min)
6. [ ] Evaluate Phase 1 checkpoint (proxy eval)
7. [ ] Phase 2: scorer-guided fine-tuning (2 hours)
8. [ ] Evaluate Phase 2 checkpoint (proxy + auth eval)
9. [ ] Phase 3: Lagrangian annealing (3 hours)
10. [ ] Evaluate Phase 3 checkpoint (auth eval)
11. [ ] Phase 4: hard-pair fine-tuning (30 min)
12. [ ] Final auth eval on best checkpoint
13. [ ] FP4 quantization and archive packaging
14. [ ] Rate-distortion comparison with v5 baseline
15. [ ] Council review of results and next steps
