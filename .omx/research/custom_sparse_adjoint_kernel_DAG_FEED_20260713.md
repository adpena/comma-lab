---
title: "DAG FEED - custom sparse/low-rank Metal adjoint primitive"
date_utc: "2026-07-13"
lane_id: "custom_sparse_adjoint_kernel"
research_only: true
score_claim: false
pointer_moved: false
training_performed: false
live_run_mutated: false
status: "BLOCKED_NO_METAL_PARITY_OR_WALL"
---

# FEED-custom-sparse-adjoint-kernel-20260713

## Scope and non-authority

This feed adds a **default-off compute primitive**, not a sparse-adjoint accuracy provider and not a
contest-score lever. It does not alter the live trainer, a run directory, `witness_control/*`, or a
v9/#432 surface. The caller must supply a per-convolution spatial support or an `r`-vector
cotangent basis. The primitive cannot discover the post-hoc oracle mask and cannot certify a
state-stable Jacobian.

## Nodes and edges

```text
[#486 MEASURED n600/heldout-120 costate + support receipt]
    |  DERIVED whole-network ideal spatial ceiling C*=2.208577465069467
    |  MEASURED 20%-oracle rel-L2=0.026206284007981848
    |  MEASURED 20%-source-margin rel-L2=0.5401736369574366
    v
[caller-supplied per-layer support M_l or rank-r cotangent basis B]
    |
    v
[metal_sparse_adjoint: no-atomic compact Conv2d input VJP]
    |-- fixed reduction order kh -> kw -> output-channel
    |-- fp contraction disabled
    |-- grouped/depthwise/stride/dilation geometry
    |-- fused rank <=8; deterministic chunks above 8
    v
[GATE P: N=10 cross-process Metal vs NumPy-fp32 dense-on-support]
    |  CURRENT: BLOCKED/UNMEASURED (execution sandbox has no Metal device)
    |  REFUSE on any per-config bit mismatch unless a documented #356/L70 fp wall is re-derived
    v
[GATE W: 125-real-SegNet-convolution wall replay vs existing #212 dense kernel]
    |  CURRENT: BLOCKED/UNMEASURED
    |  report A=T_dense/T_sparse, eta=A/C*, residual time, hardware fingerprint
    +-------------------------------+
    |                               |
    v                               v
[oracle-mask predictor provider]    [#487 state-stable Jacobian provider]
    | current named gap:            | K=2 is the only bounded survivor
    | 0.5139673529494547 abs rel-L2 | current n=1 smoke NOT_ADMITTED
    v                               v
[n600 renderer-gradient + optimizer-regret + in-loop-wall admission gates]
    |
    v
[typed DSL compute lever MAY be created]
```

There is deliberately no edge from the kernel directly to the frontier pointer. There is also no
edge from CPU NumPy parity to a Metal wall claim.

## Triality

- **Equation:** `custom_sparse_adjoint_achieved_vs_ceiling_v1` in
  `src/tac/canonical_equations/custom_sparse_adjoint_achieved_ceiling_20260713.py` defines
  `C=F_dense/F_sparse`, `A=T_dense/T_sparse`, `eta=A/C`, and the reuse crossover
  `T_basis/K<T_dense`.
- **DAG:** this file owns the gate order and makes both missing accuracy providers explicit.
- **DSL:** `REFUSED_NOT_FIREABLE`. A typed lever would falsely imply admission before the Metal
  parity/wall gates and before either accuracy provider passes n600 through-R/optimizer gates.

The canonical equation registry is a shared hot surface and already had unrelated in-flight edits.
The new equation module exposes an explicit append-only population function, but this lane does not
append an empirical anchor while Metal is unmeasured.

## Canonical consumers and hooks

- Sensitivity contribution: none; this lane consumes #486 support accounting and does not create a
  new gradient/sensitivity measurement.
- Pareto constraint: non-binding until both accuracy and wall gates are green.
- Bit allocator: no hook; this changes compute cost, not archive bytes.
- Cathedral/autopilot: fail-closed DAG node only; no dispatch while status is blocked.
- Continual-learning posterior: no empirical Metal anchor exists to append.
- Probe-disambiguator: the benchmark ships both spatial-support and rank-batched compositions; the
  parity/wall receipts arbitrate compute, while #486/#487 arbitrate accuracy.

## Scoped negative and reopen condition

`verdict_scope`: the current sandbox's ability to execute and measure the new Metal program. It is
not a verdict on the kernel's host performance or on the sparse-adjoint family.

`req_R`: execute `tools/run_custom_sparse_adjoint_kernel_host.command` on a real Metal device; first
require N=10 per-config NumPy-fp32 dense-on-support parity, then seal the 125-shape wall replay. Only
after that, connect an n600-admitted current-witness oracle-mask predictor or the governed K=2
state-stable-Jacobian provider.

