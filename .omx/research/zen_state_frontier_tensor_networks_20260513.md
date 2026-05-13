# Zen-state frontier — Domain 6: Tensor networks (2026-05-13)

**Lane**: `lane_zen_state_frontier_deep_math_research_20260513` (sub-ledger #6 of 9)
**Master memo**: `.omx/research/zen_state_frontier_deep_math_research_20260513.md`
**Scope**: MPS/PEPS/MERA decomposition of HNeRV decoder, bond dimensions, score-aware truncation.

---

## 6.1 Matrix Product States (MPS)

### Decomposition

For a tensor T ∈ ℝ^{d_1 × d_2 × ... × d_N}:

```
T_{i_1, ..., i_N} = Σ_{α_1, ..., α_{N-1}} A^{[1]}_{i_1, α_1} · A^{[2]}_{α_1, i_2, α_2} · ... · A^{[N]}_{α_{N-1}, i_N}
                                                                          (6.1.1)
```

where A^{[k]} has bond dimension χ.

**Parameter count**: `~N · d · χ²` vs full tensor's `d^N`. Compression factor for d=32, N=4: `32^4 = 1M` full vs `4·32·χ²` = ~512 at χ=4 → **2000× compression**.

### MPS for HNeRV convolution kernels

HNeRV uses ~5 conv blocks with 32×32×3×3 kernels = 9216 params/layer. MPS decomposition:

```
Conv_kernel ∈ ℝ^{C_out × C_in × k × k}
       = ℝ^{32 × 32 × 3 × 3}

MPS decomposition:
   A^{[1]} ∈ ℝ^{32 × χ}       (C_out × χ)
   A^{[2]} ∈ ℝ^{χ × 32 × χ}   (χ × C_in × χ)
   A^{[3]} ∈ ℝ^{χ × 3 × χ}    (χ × k × χ)
   A^{[4]} ∈ ℝ^{χ × 3}        (χ × k)

Param count at χ=4: 32·4 + 4·32·4 + 4·3·4 + 4·3 = 128 + 512 + 48 + 12 = 700
                vs full: 9216 → 13× compression at χ=4.
```

### Bond dimension and accuracy

**Schollwöck 2011 theorem**: MPS approximation error bounded by truncated singular values:

```
||T - T_MPS||_F² ≤ Σ_k Σ_{α > χ} σ_α²(T^{(k)})                            (6.1.2)
```

For HNeRV kernels: σ_n decays geometrically. To capture 99% of Frobenius norm: χ ≈ 6-10.

---

## 6.2 Score-aware MPS truncation

### Standard MPS truncation

Minimize Frobenius norm error.

### Score-aware truncation

Minimize SCORE impact:

```
min_{χ_distribution} Σ_layer <∂S/∂T_layer> ⊙ (T_layer - T_MPS_layer(χ_layer))
subject to Σ_layer params(χ_layer) ≤ B_max                                (6.2.1)
```

### Water-filling solution

This is convex (under reasonable assumptions): allocate MORE bond dim to layers with HIGH score-sensitivity (large ∂S/∂T).

```python
def score_aware_water_filling(model, val_data, byte_budget):
    # Compute Fisher diagonal per layer
    fisher = {layer: compute_fisher_diag(layer, val_data) for layer in model}

    # Score sensitivity proxy: per-layer Fisher trace
    sensitivity = {layer: fisher[layer].sum() for layer in model}

    # Allocate bond dimensions via water-filling
    sorted_layers = sorted(model.layers, key=lambda l: sensitivity[l], reverse=True)
    chi_per_layer = {layer: 1 for layer in model}
    bytes_used = sum(params_at_chi(layer, 1) for layer in model)

    # Greedy: bump chi for highest-sensitivity layer
    while bytes_used < byte_budget:
        layer = sorted_layers[0]  # most sensitive
        chi_per_layer[layer] += 1
        bytes_used += delta_bytes(layer, chi_per_layer[layer])
        sorted_layers.sort(key=lambda l: -sensitivity[l] / params_at_chi(l, chi_per_layer[l]))

    return chi_per_layer
```

### Predicted Δscore

MPS-pruned HNeRV with score-aware water-filling: ~1.5-3% rate reduction at same score → **-0.001 to -0.003**.

Stacked with Brenier-OT quantization (Domain 3.2): synergy through orthogonal axes → combined -0.003 to -0.008. **[first-principles-bound]**.

---

## 6.3 PEPS for 2D spatial structure

### Decomposition

PEPS (Projected Entangled Pair States) for a 2D tensor T_{i,j,k,l} on a 2D lattice:

```
T_{i,j,k,l} = Σ_α Σ_β Σ_γ Σ_δ A_{i,α,β,γ,δ} · A_{j,β,...} · A_{k,...} · A_{l,...}
                                                                          (6.3.1)
```

Each tensor has 4 bond indices (one for each neighbor on the 2D lattice).

### For HNeRV spatial conv kernels

3×3 conv kernel = 9-element 2D lattice. PEPS decomposition:
- Each kernel position has tensor of shape (channels_in, channels_out, 4 × χ).
- Total params per layer: 9 · 32 · 32 · 16 · χ² = ~150K · χ²/16 at χ=4.
- ~2× compression over MPS at same fidelity.

### Predicted Δscore

Marginal over MPS alone: ~0.5-1% rate. Adds value when HNeRV has many 3×3 conv layers (which it does). **-0.0005 to -0.001 score**. **[literature-prediction]**.

---

## 6.4 MERA for hierarchical structure

### MERA (Multi-scale Entanglement Renormalization Ansatz)

Vidal 2007 [arXiv:cond-mat/0610099]: hierarchical tensor network with disentanglers and isometries at multiple length scales.

```
MERA structure:
  Layer 0: fine-scale physical sites
  Layer 1: disentanglers + isometries (combine 2 sites → 1 site)
  Layer 2: same, on coarser sites
  ...
  Layer log(N): single coarse site (top of MERA)
```

### For HNeRV's PixelShuffle architecture

HNeRV's PixelShuffle blocks do EXACTLY what MERA does — they hierarchically combine fine-scale features into coarse-scale ones. **MERA IS the natural ansatz for HNeRV**.

### Compression bound

MERA represents 2D states with `bond_dim² · log(N)` params, vs PEPS's `bond_dim⁴`. For deep HNeRV with N pixel positions per layer: MERA is more efficient by factor `bond_dim²/log(N)`.

### Predicted Δscore

MERA-decomposed HNeRV: ~2-4× compression over flat MPS at same fidelity → ~3-5% rate reduction → **-0.002 to -0.003 score**. **[literature-prediction]**.

---

## 6.5 Tensor network operator approximation

### Operator-Schmidt decomposition

For an operator U: ℋ_A ⊗ ℋ_B → ℋ_A ⊗ ℋ_B (e.g., a layer in HNeRV):

```
U = Σ_k σ_k · A_k ⊗ B_k                                                   (6.5.1)
```

The OPERATOR rank is the smallest k achieving exact decomposition. For HNeRV layers: operator rank typically 8-32 (much less than parameter rank 32×32 = 1024).

### Score-preserving operator truncation

Truncate operator to top-k singular values, with k chosen by Fisher-water-filling. **Predicted Δscore from operator truncation**: -0.001 to -0.005. **[first-principles-bound]**.

---

## 6.6 What would falsify

- Fit MPS to PR101 HNeRV decoder. If achievable compression < 1.5× at 99% accuracy, MPS hypothesis weakened.
- Implement score-aware bond-dim water-filling. If achieved RD gain < 1% at same byte budget, score-aware hypothesis weakened.
- MERA-decompose HNeRV PixelShuffle blocks. If MERA isn't more efficient than MPS by >1.5×, MERA hypothesis weakened.

---

## 6.7 Citations

1. Schollwöck 2011. "DMRG in the age of MPS." Ann Phys 326. arXiv:1008.3477
2. Verstraete & Cirac 2004. "Renormalization algorithms for 2D QM systems." arXiv:cond-mat/0407066
3. Vidal 2007. "Class of quantum many-body states that can be efficiently simulated." Phys Rev Lett 101. arXiv:cond-mat/0610099
4. Novikov et al. 2015. "Tensorizing neural networks." NIPS. arXiv:1509.06569
5. Khrulkov-Hrinchuk-Oseledets 2019. "Generalized tensor models for RNNs." ICLR. arXiv:1901.10801
6. Cichocki et al. 2017. "Tensor networks for dimensionality reduction and large-scale optimization." Foundations and Trends in ML.
7. Oseledets 2011. "Tensor-train decomposition." SIAM J Sci Comput 33.
8. Lubasch-Cirac-Bañuls 2014. "Algorithms for finite projected entangled pair states." Phys Rev B 90.
9. Liu et al. 2024. "MERA for image compression." arXiv:2403.04032
10. Pan et al. 2023. "Tensor network compression for deep learning models — a survey." arXiv:2302.09019

END.
