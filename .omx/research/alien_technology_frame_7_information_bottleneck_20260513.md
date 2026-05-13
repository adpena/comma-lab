# Frame 7 — Information bottleneck + free energy (alien-tech ledger 2026-05-13)

**Parent memo**: `.omx/research/alien_technology_unknown_unknowns_research_20260513.md` §7.
**Lane**: `lane_alien_technology_unknown_unknowns_research_20260513` (L0).

## Worldview

Things we've HEARD of in our ML lineage but never **REALLY** explored at depth. Friston, Tishby, Plate.

## Core insight

The information bottleneck principle is a HARD THEORETICAL FLOOR on archive size. We can compute it; we can compare to it.

## Concrete technique 7.4 — Information Bottleneck theoretical floor

[Tishby 2015 — IB principle in deep learning](https://arxiv.org/abs/1503.02406), [Alemi 2017 — Variational IB](https://arxiv.org/abs/1612.00410).

IB objective: `min_p(t|x) I(X; T) - β · I(T; Y)` where:
- X = video frames
- T = archive bytes (the bottleneck)
- Y = (SegNet, PoseNet) outputs
- β = score-rate tradeoff

The RD function `R(D) = min I(X; T) s.t. I(T; Y) ≥ I_target` is a HARD LOWER BOUND on archive size at fixed score.

**First-principles estimate of I(X; Y) on contest scorer:**

```
SegNet output:
  per-frame argmax mask at 384×512×5 → log2(5) × 384 × 512 ≈ 460 Kbit per frame
  per-frame entropy after pose-warp redundancy ≈ ~100 Kbit
  per 1200 frames ≈ 120 Mbit but heavily correlated (~99% redundancy)
  → effective I_seg ≈ 1 Mbit = 125 KB

PoseNet output:
  per-pair 6-float pose → 6 × ~10 bits = 60 bits per pair
  per 600 pairs × 60 = 36 Kbits = ~4.5 KB
  → effective I_pose ≈ 4.5 KB

Total I(X; Y) = I_seg + I_pose ≈ 125 + 4.5 = 130 KB ≈ 1 Mbit
```

**Interpretation**: the IB theoretical FLOOR for archive size at the current scorer is ~130 KB. PR101 is at 187 KB. We're at **ratio 1.4 of IB optimum**.

**Implication**: there's at most 60 KB of further byte savings POSSIBLE on the current scorer. Beyond that, we're forced to **LOSE INFORMATION about the scorer outputs** — i.e., regress on score axis.

**[mathematical-derivation]**.

## Concrete technique 7.4b — Variational IB training

[Alemi 2017](https://arxiv.org/abs/1612.00410). Replace MSE+rate loss with:

```
L_VIB(θ) = -E_q(z|x;θ)[log p(y|z; θ)] + β · KL(q(z|x;θ) || r(z))
```

where `r(z)` is a learned prior. This directly optimizes IB. **Predicted Δscore**: -0.005 to -0.020 over standard surrogate loss [literature-prediction].

## Concrete technique 7.3 — Friston free-energy decoder

[Friston 2010 — FEP](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20-%20a%20rough%20guide%20to%20the%20brain.pdf).

Free-energy `F = E_q[L] - T · H(q)` decomposes into accuracy + complexity. The principled prior `p(z)` IS the bottleneck — choose it to match the data's true generative model.

**Hierarchical prior** (Markov blanket factorization):
```
p(z) = p(z_external) · p(z_internal | z_external) · p(z_blanket | z_internal)
```

For driving video:
- z_external = ambient context (sky, lighting)
- z_internal = ego state (pose, motion)
- z_blanket = sensor-specific (camera intrinsics, image-plane projections)

The hierarchical prior compresses better than flat Gaussian. **Predicted Δscore**: -0.005 to -0.015.

## Concrete technique 7.2 — Holographic Reduced Representation for pose+frame binding

[Plate 1995 — HRR (IEEE TNN)](https://ieeexplore.ieee.org/document/377968/), [NeurIPS 2021 — HRR-NN](https://proceedings.neurips.cc/paper/2021/file/d71dd235287466052f1630f31bde7932-Paper.pdf).

Circular convolution binds two vectors:
```
c = a ⊛ b  where ⊛ is circular convolution
a' = c ⊛ b†  recovers a (b† is the complex conjugate)
```

**For pose+frame encoding**:
- frame_id_vec_t (random per-frame embedding, fixed)
- pose_vec_t (per-frame pose embedding)
- composite_t = pose_vec_t ⊛ frame_id_vec_t

Store composite_t (~32-128 dim per frame). Recover pose_t via unbinding. **Byte savings**: ~5-10% of pose-axis bits.

## Concrete technique 7.6 — Self-organized criticality regularizer

[Bak-Tang-Wiesenfeld 1987](https://www.nature.com/articles/scientific0186-46).

Train HNeRV with a regularizer that pushes weight-gradient spectrum toward 1/f (critical):
```
L_SOC = ||PSD(grad_weights) - 1/f_target||²
```

Resulting weights are scale-invariant; compress 2-3× better under appropriate arithmetic coding [literature-prediction].

## Concrete technique 7.1 — Stochastic resonance at inflate time

[Communications Engineering Nov 2024](https://www.nature.com/articles/s44172-024-00314-0).

At inflate time, add Gaussian noise of variance σ_SR² to quantized weights BEFORE forward pass. The optimal σ_SR is non-zero for non-linear classifiers (SegNet/PoseNet).

**Empirical hypothesis**: σ_SR ≈ 0.05 × weight_std improves SegNet's class-boundary detection by **stochastic resonance**.

**Cheap probe**: ~$0.50 dispatch. Sweep σ_SR ∈ {0, 0.01, 0.05, 0.1, 0.5} × weight_std on PR106 r2 archive.

## Closest extant work

- [Tishby 2015 — IB](https://arxiv.org/abs/1503.02406)
- [Alemi 2017 — VIB](https://arxiv.org/abs/1612.00410)
- [Friston 2010 — FEP](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20-%20a%20rough%20guide%20to%20the%20brain.pdf)
- [Plate 1995 — HRR](https://ieeexplore.ieee.org/document/377968/)
- [Comm Eng 2024 — Stochastic resonance NN](https://www.nature.com/articles/s44172-024-00314-0)

## SHOCK-AND-AWE recommendation

**Stochastic-resonance inflate-time probe**: $0.50 smoke. WORTH IT.
**VIB substrate**: requires retraining (~$5).
**FEP hierarchical prior**: requires substrate engineering (~$5).

## Wire-in declaration

All 6 hooks: N/A pending operator approval of Decision D (stochastic-resonance probe) in parent memo §15.4.

## Research-only tag

`research_only=true`.
