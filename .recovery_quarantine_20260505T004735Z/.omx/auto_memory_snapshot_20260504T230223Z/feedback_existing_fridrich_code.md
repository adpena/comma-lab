---
name: Fridrich/wavelet code ALREADY EXISTS in multiple places — don't rebuild
description: fridrich.py has real UNIWARD wavelet cost maps + Jacobian sensitivity. training.py has frequency_loss hook. Profiles have wavelet_renderer. fridrich_losses.py is a simplified duplicate.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Existing Fridrich/Wavelet Code (discovered 2026-04-22)

We have THREE separate implementations of similar concepts:

1. **src/tac/fridrich.py** (the BEST one):
   - `_uniward_cost()`: Real S-UNIWARD with Haar wavelet directional filters
   - `_jacobian_cost()`: Per-pixel scorer sensitivity via randomized Jacobian
   - `compute_pixel_cost_map()`: Hybrid UNIWARD + Jacobian cost map
   - `fridrich_constrained_optimize()`: Full constrained optimization
   - `estimate_detection_boundary()`: Find the scorer detection cliff
   - This is SOPHISTICATED and research-grade

2. **src/tac/fridrich_losses.py** (simplified duplicate I wrote 2026-04-22):
   - `compute_texture_complexity()`: Simple local variance (NO wavelets)
   - `texture_weighted_loss()`: Weighted pixel loss
   - `linf_penalty()`: Top-percentile error penalty
   - `boundary_sensitive_hinge()`: Boundary-weighted hinge
   - `markov_chain_loss()`: Gradient statistics preservation
   - USEFUL but simpler than fridrich.py

3. **src/tac/training.py** (wired into Trainer):
   - `use_frequency_loss` config flag (line 163)
   - `frequency_aware_loss()` called at line 1681
   - Already integrated into `fit_lazy()` path

**Why:** I didn't check existing code before writing fridrich_losses.py.

**How to apply:** Use `fridrich.py`'s `_uniward_cost()` for the real wavelet cost
map. Use `fridrich_losses.py`'s `linf_penalty` and `boundary_sensitive_hinge` which
are genuinely new. Wire them into the training loop via the existing
`use_frequency_loss` config hook or add new config flags.

DO NOT write more Fridrich code. Use what exists.
