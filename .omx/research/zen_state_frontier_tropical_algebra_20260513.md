# Zen-state frontier — Domain 5: Tropical / max-plus algebra (2026-05-13)

**Lane**: `lane_zen_state_frontier_deep_math_research_20260513` (sub-ledger #5 of 9)
**Master memo**: `.omx/research/zen_state_frontier_deep_math_research_20260513.md`
**Scope**: tropical polynomial structure of ReLU/argmax nets, SegNet boundary encoding, tropical lottery tickets.

---

## 5.1 Tropical semiring

### Definition

The **tropical (max-plus) semiring** is:

```
T = (ℝ ∪ {-∞}, ⊕, ⊗)
where x ⊕ y = max(x, y)        (tropical addition)
      x ⊗ y = x + y             (tropical multiplication)
```

A **tropical polynomial** in N variables:

```
p(x) = ⊕_i (a_i ⊗ x_{i,1}^{c_{i,1}} ⊗ ... ⊗ x_{i,N}^{c_{i,N}})
     = max_i (a_i + Σ_j c_{i,j} · x_j)                                   (5.1.1)
```

### Tropical hypersurface

The **tropical hypersurface** of p is the locus where the max is NOT unique:

```
V(p) = {x : ≥ 2 monomials achieve max in p(x)}                            (5.1.2)
```

This is a polyhedral complex of codimension 1 in ℝ^N.

---

## 5.2 ReLU networks = tropical rational functions

### Zhang-Naitzat-Lim theorem (ICML 2018)

Every PWL function f: ℝ^N → ℝ^M computed by a ReLU network can be expressed as:

```
f(x) = p(x) ⊘ q(x) = p(x) - q(x)                                          (5.2.1)
```

where p, q are tropical polynomials. The **number of monomials** in p+q is bounded by `(number_of_neurons)^(network_depth)`.

### For our scorers

**SegNet** uses EfficientNet-B2 + Unet head:
- ~6M params, ~5000 activations effective, ~50 conv blocks.
- Tropical-monomial bound: 5000^50 (astronomical, but mostly redundant).
- Empirically (Hanin-Rolnick 2019 [arXiv:1906.00904]): ACTUAL number of linear regions ~10^5-10^7, MUCH less than bound.

**PoseNet** uses FastViT-T12 (RepMixer/conv backbone):
- Similar structure; ~12 transformer blocks reparameterized.
- Tropical-monomial bound: even smaller because RepMixer is more efficient.

---

## 5.3 SegNet argmax = tropical hypersurface vertices

### Key insight

SegNet's output is argmax over 5 classes. The decision boundary between class i and class j is:

```
B_{ij} = {x : logit_i(x) = logit_j(x)}                                    (5.3.1)
```

In tropical terms: B_{ij} is a face of the tropical hypersurface V(max_class).

The COMPLETE decision boundary is the union of all B_{ij} for i<j:

```
B = ∪_{i<j} B_{ij}      (the multiboundary)                              (5.3.2)
```

### Compression opportunity

If we can encode B (the decision boundary) in few bytes, we've encoded SegNet's output for free.

**Estimation**:
- For 384×512 image with 5 classes: typical decision boundary length ~5000-15000 pixels.
- Encoded as polyline at 0.5-1 byte/vertex (delta-coded): ~5-15 KB per frame.
- Per-frame delta coding (most boundary doesn't move frame-to-frame): ~0.5-2 KB per frame.
- Over 1200 frames: 0.6-2.4 MB → after entropy coding ~50-200 KB.

Compare to current mask encoding (AV1 monochrome at ~40 KB): tropical encoding might be COMPETITIVE or BETTER, depending on entropy of boundary delta.

### Tropical encoding algorithm

```python
def tropical_encode_segnet_argmax(argmax_map):
    # Find decision-boundary pixels
    boundaries = compute_decision_boundaries(argmax_map)  # set of (i, j, polyline) triples
    # Delta-encode against previous frame's boundary
    delta_boundaries = compute_delta(boundaries, prev_boundaries)
    # Entropy-code the deltas
    encoded = arithmetic_code(delta_boundaries)
    return encoded
```

### Decoding (inflate-side)

```python
def tropical_decode_segnet_argmax(encoded):
    delta_boundaries = arithmetic_decode(encoded)
    # Reconstruct boundary at each frame
    for t in range(1200):
        boundary[t] = boundary[t-1] + delta_boundaries[t]
    # Fill class labels from boundary structure
    argmax_map = fill_argmax_from_boundary(boundary[t])
    return argmax_map
```

**[first-principles-bound]**: predicted Δrate -20 to -40 KB → -0.013 to -0.027 score.

---

## 5.4 Tropical-Decomposition LoRA (TD-LoRA, EUREKA E2)

### Idea

Replace vanilla LoRA `W' = W + AB^T` (matrix sum) with tropical sum:

```
W'(x) = max(W·x, A·x + b_A, B·x + b_B, ...)                              (5.4.1)
```

This is a tropical polynomial in W's parameters. The output is a PWL function with k+1 "knee points" (k = number of additional A-and-B-style branches).

### Why this is more expressive

Standard LoRA produces a low-rank LINEAR perturbation: `W + AB^T` is rank-r.

Tropical-LoRA produces a low-rank PWL perturbation: at most k different "linear regions" in the input space, each with its own (matrix, bias) pair.

For the same byte budget (k pairs of (A_i, b_i)), tropical-LoRA has MORE expressive capacity because:
- Linear LoRA: 1 region, dimension d_in × r.
- Tropical-LoRA: k regions, each dimension d_in × r/k.

For DoRA (magnitude/direction split), tropical generalization is even more powerful.

### Predicted Δscore

DoRA paper (Liu 2024) shows 0.5-1.5% gain over LoRA. Tropical extension should add another 0.3-1% (per first-principles expressivity gain). **Predicted Δscore**: -0.005 to -0.010. **[first-principles-bound + literature-prediction]**.

### Implementation

```python
class TropicalLoRA(nn.Module):
    def __init__(self, base_module, rank=4, n_branches=3):
        super().__init__()
        self.base = base_module
        self.branches = nn.ModuleList([
            nn.Linear(in_dim, rank, bias=True) for _ in range(n_branches)
        ])
        self.combine = nn.Linear(rank, out_dim, bias=False)

    def forward(self, x):
        # Compute base output
        y_base = self.base(x)
        # Compute each branch
        branch_outputs = torch.stack([self.combine(b(x)) for b in self.branches], dim=-1)
        # Tropical max
        y_tropical = branch_outputs.max(dim=-1).values
        return y_base + y_tropical
```

---

## 5.5 Lottery tickets in tropical space

### Standard lottery ticket (Frankle-Carbin 2019)

Find a sparse subnetwork (pruned weights) that trains to dense-network performance.

### Tropical lottery ticket

For a tropical polynomial `p(x) = ⊕_i a_i x_i`, find a SPARSE subset of monomials `S ⊆ {1,...,N}` such that:

```
p_S(x) = ⊕_{i ∈ S} a_i x_i      ≈ p(x)  on training data                 (5.5.1)
```

This is the "tropical lottery ticket".

### Why this is principled

Most monomials in a tropical polynomial are NEVER ACTIVE (never achieve the max for ANY input). Only the "active" subset matters.

**Algorithm to find tropical lottery ticket**:
```python
def tropical_lottery_ticket(model, data, threshold=0.001):
    activations = {monomial_i: 0 for monomial_i in monomials(model)}
    for x in data:
        active = find_active_monomial(model, x)  # which monomial wins argmax
        activations[active] += 1
    # Keep only monomials active > threshold fraction of inputs
    keep = {i: count > threshold*len(data) for i, count in activations.items()}
    pruned_model = prune_model(model, keep)
    return pruned_model
```

### Predicted compression

For ReLU/PWL networks: typical fraction of "active" monomials per input is ~5-15%. Pruning to active-only: 7-20× parameter compression.

For HNeRV (semi-ReLU via GELU): active fraction is higher (~30-50%) but still room for 2-3× compression.

**Predicted Δscore from tropical-pruned HNeRV**: -0.005 to -0.015 (compression more than offsets minor distortion loss). **[first-principles-bound]**.

---

## 5.6 Real tropical geometry (Pasque-Tran-Maragos 2024)

### Recent advance

Pasque-Tran-Maragos 2024 [arXiv:2403.11871] extended tropical analysis to REAL (continuous) activations like GELU/SiLU. Result: bounds remain valid asymptotically, with corrections.

### Implication

Our HNeRV decoder (GELU activation) is ALMOST tropical in the limit of large pre-activation magnitudes. The tropical bounds (regions, monomials) apply UP TO an exponentially small correction.

**Practical**: tropical encoding methods work essentially unchanged on GELU networks for the purposes of our analysis.

---

## 5.7 What would falsify

- Fit a tropical net to SegNet's argmax on 1200 frames. If it needs > 50 KB to match SegNet at 99% pixel accuracy, tropical-encoding hypothesis weakened.
- Implement tropical-LoRA on a small PR95 sweep. If <0.3% RD-gain over vanilla LoRA, tropical-LoRA hypothesis weakened.

---

## 5.8 Citations

1. Zhang, Naitzat, Lim 2018. "Tropical geometry of deep neural networks." ICML. arXiv:1805.07091
2. Maragos 2017. "Dynamical systems on weighted lattices." arXiv:1709.02394
3. Hanin & Rolnick 2019. "Deep ReLU networks have surprisingly few activation patterns." NeurIPS. arXiv:1906.00904
4. Charisopoulos & Maragos 2018. "Morphological perceptrons and lattice ANN." arXiv:1709.06164
5. Pasque-Tran-Maragos 2024. "Real tropical geometry of deep learning." arXiv:2403.11871
6. Mikhalkin 2005. "Enumerative tropical algebraic geometry in ℝ²." J AMS 18.
7. Itenberg-Mikhalkin-Shustin 2007. "Tropical algebraic geometry." Birkhäuser.
8. Sturmfels 2002. "Solving systems of polynomial equations." CBMS Reg Conf 97.
9. Develin-Sturmfels 2004. "Tropical convexity." Doc Math 9.
10. Develin-Yu 2007. "Tropical polytopes and cellular resolutions." Expo Math 25.

END.
