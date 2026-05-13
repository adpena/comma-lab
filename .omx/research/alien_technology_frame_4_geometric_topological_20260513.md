# Frame 4 — Geometric / topological foundations (alien-tech ledger 2026-05-13)

**Parent memo**: `.omx/research/alien_technology_unknown_unknowns_research_20260513.md` §4.
**Lane**: `lane_alien_technology_unknown_unknowns_research_20260513` (L0).

## Worldview

Euclid + Riemann + Poincaré + Lefschetz + Atiyah → topology and differential geometry as the DEEPEST way to know a signal. Compression = preserving topological invariants.

## Core inductive bias

**What matters is topology, not pixels.** Two videos that are topologically-equivalent compress to the same descriptor.

## Concrete technique 4A — Persistent-homology video descriptor

[Carlsson 2009](https://www.ams.org/notices/200906/rtx090600711p.pdf), [Edelsbrunner-Harer 2010](https://www.maths.ed.ac.uk/~v1ranick/papers/edelcomp.pdf).

Procedure:
1. For each frame, extract 3×3 patches (81-dim point cloud per frame).
2. Compute Vietoris-Rips filtration → persistence barcodes.
3. [Stanford SURJ 2024](https://ojs.stanford.edu/ojs/index.php/surj/article/view/1345) showed natural-image patches lie on a **connected bouquet of spheres**.
4. Encode: per-frame barcode (~20 birth-death pairs) + fixed bouquet prior + small per-frame decoder.

**Architecture**:
```python
def inflate_topology(barcode, prior_bouquet, decoder):
    # barcode: list of (birth, death) pairs per dim
    # bouquet: precomputed common topology
    submanifold = inflate_barcode_to_submanifold(barcode, prior_bouquet)
    frame = decoder(submanifold)
    return frame
```

**[literature-prediction]**: byte budget ~5 KB per frame for barcodes + decoder. Total: 6 MB / 1200 frames ≈ 5 KB/frame after coding. Comparable to HNeRV per-frame.

## Concrete technique 4B — Discrete differential geometry rendering

[Crane et al.](https://www.cs.cmu.edu/~kmcrane/Projects/DDG/paper.pdf).

Video as 3-form on discrete 3-manifold; encode gradient field (the 2-form `dV`) instead of 3-form V.

Helmholtz-Hodge decomposition:
```
V = grad(potential) + curl(vector_field) + harmonic_form
```

For smooth driving video, ~95% of variance is in the harmonic + grad components (low-frequency); curl is small (~5%).

**Byte savings**: encode harmonic + grad with high-precision (small bytes); encode curl with low precision. Predicted: 10-15% byte reduction at fixed score [literature-prediction].

## Concrete technique 4C — Knot-theoretic ego-trajectory encoding

The ego-trajectory in SE(3) is a curve. Compute:
- Jones polynomial (knot invariant)
- HOMFLY polynomial (link invariant)
- Reidemeister classes of the curve

For driving video: trajectories are unknots → trivial Jones polynomial. **BUT** the **branched 2-complex** (the trajectory with bifurcation points at lane changes / stops) has non-trivial topology.

**Pre-computation**: identify all "branch events" in the trajectory. Encode the branched 2-complex topology (~10 bytes) + per-branch trajectory delta (~100 bytes per branch).

**Total pose-encoding budget**: ~1-2 KB (vs current ~3-5 KB). Save 1-3 KB. **Marginal at PR106 frontier**.

## Concrete technique 4D — Sheaf-cohomology compression

[Sheaf cohomology](https://en.wikipedia.org/wiki/Sheaf_cohomology).

The video as sheaf of local sections (per-block pixel data). The **gluing axiom**: compatible local sections on overlapping patches join into a global section. The **cohomology obstructions**: non-trivial H^1 classes correspond to motion boundaries / occlusions where local sections DON'T glue smoothly.

**Concrete coding**:
- Encode local patch sections as usual (no change).
- At block boundaries, encode ONLY the cohomology obstruction (Čech 1-cocycle).
- Most block boundaries have trivial cohomology → only **boundary deltas** stored.

**Predicted savings**: 5-10% block-coding-rate reduction [literature-prediction].

## Concrete technique 4E — Hodge-Helmholtz decomposition prior

Train HNeRV with a **harmonic-+-grad-prior** instead of standard Gaussian. The latent is decomposed:

```
z = z_grad + z_curl + z_harmonic
```

with separate entropy coders for each component. Predicted: **3-7% byte budget reduction** at fixed score for driving video (which is grad-dominated).

## Closest extant work

- [GUDHI computational topology library](https://gudhi.inria.fr/)
- [Persistent homology of neural-network activations](https://arxiv.org/abs/2106.02797)
- [Functorial compression (Spivak)](https://math.mit.edu/~dspivak/CT4S.pdf)
- [Crane discrete DG](https://www.cs.cmu.edu/~kmcrane/Projects/DDG/paper.pdf)
- [Stanford SURJ — algebraic topology video compression](https://ojs.stanford.edu/ojs/index.php/surj/article/view/1345)

## Cheap-probe recommendation (~$0 offline)

Run GUDHI on `videos/0.mkv` frames; produce per-frame persistent-homology barcodes; identify **topologically-similar frame clusters**. Estimated: ~50 cluster representatives instead of 1200 frames → 24× reduction in latent diversity if exploited.

## Wire-in declaration

All 6 hooks: N/A (research-only frame ledger).

## Research-only tag

`research_only=true`.
