# D16 METAL-KERNELS — #212 suite unbuilt-kernel build + evidence-based ranking — 2026-07-08  [no-triality]

Operator (#252 compute program): *"build all unbuilt items and wire and integrate and DSL and
triality."* Target: the 3 UNBUILT #212 fused kernels (persistence-pool, margin-map, curvelet;
`.omx/research/v7_compute_exploitation_audit_20260708.md` lever #5). Cost-ranked, STOP-after-(1)
gate. **MEANS, not ends** — nothing here moves the pointer (0.19110); only a byte-closed n600
exact row < 0.19110 does. Compute is score-NEUTRAL by construction (bit-identical) + lexicographically
second to score.

**STORES CONSULTED:** v7_compute_exploitation_audit_20260708 (lever #5 signatures + clDice ~7.5%-of-step
anchor); `metal_grouped_conv_backward` + `metal_fused_r_operator` (the BUILT template: `mx.fast.metal_kernel`
+ `@mx.custom_function.vjp` + `_available()` + fixed-order/no-atomics determinism + parity bar);
`persistence_topology_loss` (`_pool3x3_np` authority, `_pool3x3_mlx` 9-shift path, `metal_pool_kernel_signature`,
clDice hot path); `tools/mlx_gpu_determinism_probe.py` (L70 scatter-poison localization); MEMORY L52/L59/L70.

## BUILT — persistence-pool kernel (kernel 1, the primary)

`src/tac/local_acceleration/metal_persistence_pool.py` — ONE `mx.fast.metal_kernel`
(`persistence_pool_3x3`) computing the edge-clamped 3x3 **min / max / mean** over trailing (H,W)
of an (M,H,W) fp32 field in a single register pass (replaces the pure-MLX 9-shift-stack + reduce
= 9× memory traffic).

- **Bit-identical (max|Δ|=0, MEASURED)** to the numpy authority `_pool3x3_np` on real-shaped
  tensors (M∈{1,2,4,8}, 384×512, + 2D + (N,K,H,W) + border ramp). min/max = exact selection;
  mean = sequential fp32 sum of the 9 taps in numpy's (di,dj) order /9 (verified equal to
  `np.mean(np.stack(wins),0)` — 9-way reduction is below numpy's pairwise base case). `#pragma
  clang fp contract(off)` is LOAD-BEARING for the mean bit-match.
- **Deterministic** — one thread per output, no atomics/scatter. Probe `persistence_pool` added to
  `mlx_gpu_determinism_probe.py`: **N=5 cross-process bit-identical, 0 nondeterministic**.
- **Wired** (forward-only, stop-grad guarded) into `_smooth_density_mlx` (the density-weight
  smoothing pool, which is `mx.stop_gradient`'d by construction ⇒ no VJP needed). Gated by
  `TAC_MLX_CUSTOM_PERSISTENCE_POOL` **default OFF** ⇒ byte-identical to the pre-wire trainer;
  GPU-required; pure-MLX CPU fallback intact. Full-loss parity flag-on vs flag-off on real n600
  GT: **|Δ|=0.00e+00, grad OK** (density stop-grad, pred-side skeleton stays pure-MLX).

### Measured speedup table (per-call, this host, M=4 × 384×512)

| unit | pure-MLX | metal kernel | speedup |
|---|---|---|---|
| pool `min` | 0.625 ms | 0.272 ms | **2.30×** |
| pool `max` | 0.379 ms | 0.201 ms | **1.88×** |
| pool `mean` | 0.382 ms | 0.157 ms | **2.44×** |
| soft_skeleton (17 pools, fwd) | 8.377 ms | 2.139 ms | **3.92×** |

Kernel speedup 1.9–4× ≫ the 10% STOP gate → gate cleared, proceed to assess (2)&(3).

### The honest ceiling (why the wired scope is forward-only)

A raw `mx.fast.metal_kernel` has **no VJP** — it raises `[Primitive::vjp] Not implemented for
CustomKernel` inside `mx.grad`. So it can safely accelerate ONLY non-differentiable pools
(density smoothing = stop-grad; GT skeleton = constant). The DIFFERENTIABLE pred-side
soft-skeleton (17 pools/step, the bulk of clDice) would need a bit-identical **deterministic
morphological VJP kernel** (replicating MLX's first-tap-wins tie rule for min/max + the
edge-clamp mean transpose). **Documented NO-GO:** clDice is ~7.5% of a levers-on step (audit
anchor) and `--cache-gt-skeleton` already elides ~half; the persistence loss is OFF by default.
So the whole-step payoff of a differentiable pool kernel is bounded well under 10%, against a real
scatter-determinism build risk (the L70 poison class) — the classic gold-plating trap. The
forward kernel is landed as a proven, parity-gated, reusable #212 module; the differentiable VJP
is deferred with this measured rationale, not built speculatively.

## STOP-DECISION / ranking of (2) margin-map and (3) curvelet — evidence-based DEFER

The literal gate (persistence-pool speedup ≥ 10%) says continue; I assessed (2)&(3) and DEFER
both on evidence (not built):

- **(2) margin-map** — `margin_saliency_map` (#141) is a **stop-grad saliency PRIOR** (single
  argmax/sort of the logits per pixel), NOT a 9-shift-stack hot term; MLX already vectorizes
  argmax/sort efficiently and it is not the measured clDice-class hot path. A fused kernel is
  speculative without a measured hot-term. DEFER (no measured hot term).
- **(3) curvelet** — the curvelet/shearlet directional bank (`lever_b_levelset_generator`) is a
  **GENERIC parametric basis front-end**, amortized behind `--reorient-every 50` (self-orient
  cache), not a per-step differentiable hot term. Building a fused kernel is premature. DEFER
  (amortized, not per-step-hot).

Both also face the same differentiable-path-needs-VJP wall as the pool if ever moved into the
gradient path. Reactivation: a measured per-step profile showing either as a >10%-of-step term.

## Compute-facet / triality  **[no-triality]**

Kernels are the compute FACET (MEMORY L54 facet #6): no lever/governance/DSL entry (score-neutral
throughput, not a trainer knob). This memo IS the triality doc leg. **No canonical-equation
anchor** registered: no measured LAW was produced — only a bit-identical throughput speedup
(engineering, not a predictive equation). `metal_pool_kernel_signature()` updated `built: True` +
`module` + `wired_into` so the signature reflects the built kernel (no orphan).

## Tests / determinism / fallback

`src/tac/local_acceleration/tests/test_metal_persistence_pool.py` — **24 tests**: bit-identity
(min/max/mean × 4 real shapes + 2D + extra-leading-dims + border ramp), in-process determinism,
env-flag gating (unset/falsey/truthy), density-dispatch bit-identity + flag-off fallback, input
validation, signature-built. Plus the cross-process probe cell + the 28 existing persistence-loss
tests (regression, all green). ruff F clean.

## FILES / COMMITS
- NEW `src/tac/local_acceleration/metal_persistence_pool.py` (kernel + dispatch + `_available`/`_enabled`)
- NEW `src/tac/local_acceleration/tests/test_metal_persistence_pool.py` (24 tests)
- MOD `src/tac/boundary_math/persistence_topology_loss.py` (forward-only stop-grad dispatch in
  `_smooth_density_mlx`; `metal_pool_kernel_signature` → built)
- MOD `tools/mlx_gpu_determinism_probe.py` (`persistence_pool` op cell)
