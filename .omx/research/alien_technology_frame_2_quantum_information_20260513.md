# Frame 2 — Quantum-information / tensor-network lineage (alien-tech ledger 2026-05-13)

**Parent memo**: `.omx/research/alien_technology_unknown_unknowns_research_20260513.md` §2.
**Lane**: `lane_alien_technology_unknown_unknowns_research_20260513` (L0).

## Worldview

A civilization that discovered quantum mechanics centuries before us. Their compression unit is the **qubit**; their natural data structure is the **density matrix** + **tensor network**. They never wrote backprop.

## Core inductive bias

**Entanglement IS the compression resource.** Classical correlations are a corner of the bigger quantum picture.

## Concrete technique 2A — MERA video compression (the MOST PROMISING candidate)

[Vidal 2007 — Entanglement Renormalization](https://arxiv.org/abs/cond-mat/0512165).

The Multi-scale Entanglement Renormalization Ansatz (MERA) is a tensor network with logarithmic bond dimension for representing 2D scale-invariant states.

```
|V⟩ = U_3 ∘ U_2 ∘ U_1 (|ψ_seed⟩)

where each U_k is a layer of disentanglers (2-site) + isometries (2→1 coarse-graining).
Bond dimension χ controls fidelity.
```

For YUV6 video frames as a 2D+1D tensor:

```python
import opt_einsum
import torch

# Each layer: (B, C, H, W) → coarser representation
# Disentangler at scale k: 2-site unitary in HxW plane
# Isometry: H/2 x W/2 reduction

def mera_forward(seed, layers, chi=8):
    state = seed
    for layer in layers:
        state = apply_disentanglers(state, layer.disent, chi=chi)
        state = apply_isometries(state, layer.iso)
    return decode_to_yuv6(state)
```

**Bond-dimension RD curve** [literature-prediction]: χ=8 ≈ 1.5 KB/frame; χ=16 ≈ 3 KB/frame; χ=32 ≈ 8 KB/frame. For 1200 frames at χ=8: ~1.8 MB seed-state + ~30 KB per-frame deltas.

**SUBSTRATE-ENGINEERING ESTIMATE**: 2-4 person-weeks to build a MERA-based HNeRV substrate. Council review required.

## Concrete technique 2B — Holographic compression (AdS/CFT-inspired)

[Ryu-Takayanagi 2006 — Holographic entanglement entropy](https://arxiv.org/abs/hep-th/0603001).

Encode boundary; reconstruct bulk via entanglement-entropy minimization.

For video: store border-pixel strip (32 px wide) + keyframes; reconstruct interior via learned bulk-reconstruction.

**Estimated rate**: 200 × 4 (RGBA border) × 1200 frames = 960 KB raw → ~30 KB after arithmetic coding of smooth gradients.

## Concrete technique 2C — Adiabatic / quantum-annealing archive search

[D-Wave Advantage](https://www.dwavesys.com/solutions-and-products/systems/) (~5000 qubits).

Post the contest as Ising / QUBO:

```
H(archive_bits) = score(decode(archive_bits)) + λ · |archive_bits|

H = Σ_i h_i σ_i + Σ_ij J_ij σ_i σ_j         # quadratic form
```

**Tractability**: 100 KB archive = 800K bits, far more than available qubits. **Block-decomposed annealing** ($1-2 of D-Wave): anneal 1 KB at a time conditioned on the rest.

## Concrete technique 2D — Compressed sensing in superposition

[Donoho 2006 — Compressed sensing](https://en.wikipedia.org/wiki/Compressed_sensing).

Apply a single random projection matrix Φ to the video, store the seed of Φ + the projection. At inflate, run L1-min to recover.

For video: K-sparse signals are recoverable from K · log(N) measurements [Candès-Romberg-Tao 2006]. If the YUV6 stream has effective sparsity K = 50 K (Council F's 10^9 free dims / scorer-equivalence-class cardinality), then ~50K · log(N) ≈ 50K × 24 = 1.2 M bits = 150 KB measurements. Marginally better than HNeRV. **[literature-prediction]**.

## Closest extant work (within reach)

- [TensorTrain / TT-decomposition](https://en.wikipedia.org/wiki/Matrix_product_state) (Oseledets 2011)
- [Stoudenmire-Schwab MNIST via MPS](https://arxiv.org/abs/1605.05775)
- [Beny 2013 — DL meets tensor networks](https://arxiv.org/abs/1301.3124)
- [Penrose graphical calculus](https://en.wikipedia.org/wiki/Penrose_graphical_notation)
- [TenPy library](https://github.com/tenpy/tenpy)

## SHOCK-AND-AWE recommendation

**MERA substrate** is the #2 most-likely-to-work alien-tech idea (after NCA). Recommended dispatch: $5 substrate engineering + smoke probe.

## Wire-in declaration

All 6 hooks: N/A pending operator approval of Decision B (lane pre-registration) in parent memo §15.2.

## Research-only tag

`research_only=true`.
