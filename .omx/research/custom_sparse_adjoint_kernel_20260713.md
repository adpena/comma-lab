---
title: "Custom sparse/low-rank Metal adjoint primitive"
date_utc: "2026-07-13"
lane_id: "custom_sparse_adjoint_kernel"
research_only: true
score_claim: false
pointer_moved: false
training_performed: false
paid_dispatch: false
live_run_mutated: false
verdict: "BUILT_DEFAULT_OFF; METAL_ADMISSION_BLOCKED"
verdict_scope: "default-off compact frozen-Conv2d input-adjoint source and current execution substrate; no Metal parity/wall, mask-predictor, nonlinear full-SegNet, training, score, or pointer verdict"
---

# Custom sparse adjoint kernel — primitive built, solution not claimed

> **MEANS caveat:** this work targets backward throughput only. It cannot move the contest pointer
> by itself. The kernel is a compute primitive whose accuracy is exactly the accuracy of the support
> mask or rank basis supplied by its caller. It does not solve the oracle-mask-predictor problem,
> global squeeze-excite support, Jacobian-state stability, or optimizer regret.

## Outcome first

**Kernel source: BUILT, default-off, not admitted or shipped into a live path. NumPy-fp32 compact
authority: MEASURED bit-identical to its dense-on-support reference in 40/40 deterministic CPU
trials across four grouped/depthwise/stride/rank configurations. Required Metal parity: UNMEASURED
because MLX fails closed with `No Metal device available`; therefore the mandatory N=10
cross-process per-config Metal gate is not green. Real Metal wall factor versus the DERIVED
`2.208577x` ceiling: UNMEASURED for the same reason. No memory/occupancy/gather attribution is
claimed without a wall run.**

One-line scoped verdict: **`BUILT_DEFAULT_OFF / BLOCKED-NO-METAL-ADMISSION` for this execution
substrate; `req_R` is N=10 Metal parity followed by the 125-shape wall replay, with accuracy still
owned by a current-witness oracle-mask predictor or the governed K=2 reuse provider.**

## What was built

The new module `src/tac/local_acceleration/metal_sparse_adjoint.py` reuses the #212 custom grouped
backward conventions instead of introducing a second dense baseline. It supplies:

1. a validated grouped-convolution geometry record;
2. deterministic output-mask compaction and exact local input-support propagation;
3. a fixed-order NumPy-fp32 dense input-adjoint authority;
4. a fixed-order NumPy-fp32 compact authority;
5. one no-atomic Metal thread per compact `(input-site,input-channel)`;
6. fused rank-`r` cotangent VJPs for `r<=8`, with deterministic consecutive chunks above eight; and
7. exact valid-tap FMA accounting distinct from nominal padded FLOPs.

For input site `(h_i,w_i)` and input channel `c_i`, the compact kernel evaluates only active output
sites in the caller's lookup:

```text
lambda_x[r,h_i,w_i,c_i]
  = sum_(k_h,k_w,c_o in group(c_i))
      1[M(h_o,w_o)=1]
      W[c_o,k_h,k_w,c_i-local]
      lambda_y[r,h_o,w_o,c_o].
```

The reduction order is `k_h -> k_w -> output-channel`; each thread owns its result, so there are no
atomics. `#pragma clang fp contract(off)` prevents multiply-add contraction. This deliberately
matches the NumPy authority's order. Rank is the leading dimension, so shared mask/weight lookup is
the exact compute composition needed for cached basis VJPs.

This is a **per-convolution input-VJP primitive**, composable layer-by-layer. It is not a monolithic
nonlinear SegNet-decoder VJP: nonlinearities, skip paths, and the 23 global squeeze-excite reductions
must supply their own exact or admitted adjoints and per-layer supports. Wiring such a wrapper before
the mask/state providers exist would overclaim what the kernel solves and would violate the request's
anti-collision boundary around the live trainer.

## Parity gate — authority and current verdict

Axis: **`[macOS-CPU NumPy-fp32 parity authority; macOS-MLX Metal unmeasured; non-promotable MEANS]`**.
MPS is not used as authority.

| Configuration | CPU compact vs NumPy-fp32 dense-on-support | Metal vs NumPy-fp32 | Required Metal gate |
|---|---:|---:|---:|
| standard 3x3, rank 1 | MEASURED bit-identical, 10/10 | UNMEASURED/BLOCKED | N=10 cross-process |
| grouped 3x3 stride 2, rank 2 | MEASURED bit-identical, 10/10 | UNMEASURED/BLOCKED | N=10 cross-process |
| depthwise 5x5 stride 2, rank 3 | MEASURED bit-identical, 10/10 | UNMEASURED/BLOCKED | N=10 cross-process |
| pointwise full support, rank 8 | MEASURED bit-identical, 10/10 | UNMEASURED/BLOCKED | N=10 cross-process |

The CPU trials are deterministic trials in one process, not a substitute for the requested
cross-process Metal gate. Unit verification is **MEASURED `14 passed, 3 Metal-skipped`**, including
an independent Torch-autograd geometry check for standard, grouped-stride, and depthwise cases; lint
and bytecode compilation are green. The static source guard requires fixed-order loops,
contraction-off, and no atomics.

The actual device probe is:

```text
RuntimeError: [metal::load_device] No Metal device available. This typically occurs in headless,
sandboxed, or virtualized macOS sessions where the GPU is not accessible.
```

The machine inventory reports an Apple M5 Max/40-core GPU, but visibility is not execution custody.
Approved Terminal/host-UI attempts were also safety-blocked. No sandbox bypass, paid dispatch, or
heavy launch was attempted. Consequently the kernel remains default-off and **not admitted into any
callable live trainer/DSL path**. This honors “no kernel ships without parity GREEN.”

Durable static receipt in the intended commit set:
`.omx/research/custom_sparse_adjoint_kernel_static_receipt_20260713.json`, SHA-256
`cb2a9d84194516ac61e86dbc33ccc26c4fbdbf7cb965edc39e5a02a633a7d565`.

## Achieved wall versus the 2.208577x ceiling

The predecessor's `2.208577465069467x` is **DERIVED**, not measured wall time. This landing
reconstructed all **MEASURED 125** real frozen-SegNet convolution shapes and weights on CPU:
`114 encoder + 10 decoder + 1 segmentation head`. At the sealed #486 family aggregate support
fractions, the static replay re-derives:

| Quantity | Label | Value |
|---|---|---:|
| Flagship whole-network ideal spatial ceiling | DERIVED predecessor | `2.208577465069467x` |
| 125-shape nominal replay ceiling | DERIVED here | `2.20861526222186x` |
| absolute replay/flagship gap | DERIVED here | `0.00003779715239327075x` |
| exact valid-tap replay ceiling | DERIVED here | `2.2002297677359484x` |
| achieved Metal wall factor | **UNMEASURED/BLOCKED** | — |
| achieved/ceiling efficiency | **UNMEASURED/BLOCKED** | — |

The nominal replay's tiny difference is integer site rounding under deterministic evenly spaced
synthetic masks. Those masks reproduce aggregate arithmetic pressure, not the real oracle geometry
and not fidelity.

The canonical achieved-versus-ceiling law is:

```text
C   = F_dense / F_sparse                         [DERIVED arithmetic ceiling]
A   = T_dense / T_sparse                         [MEASURED wall factor]
eta = A / C                                      [MEASURED/DERIVED efficiency]
R_t = T_sparse - T_dense/C                       [residual time above FLOP-scaled ideal]
```

`R_t` can charge map/gather traffic, launch latency, occupancy loss, or remaining dense work, but it
cannot identify which cause dominates. Since `T_dense` and `T_sparse` could not be measured here,
the **gap reason is execution-substrate blockage only**. “Memory-bound,” “gather overhead,” and
“occupancy” remain hypotheses, not findings.

The resumable host command performs, in order: ten fresh-process parity runs; fail-closed parity
aggregation; the 125-shape Metal replay against the existing #212 dense grad-input kernel; the rank-2
K=2 timing; and an atomic receipt. Each stage is distinct and resume-safe:

```text
tools/run_custom_sparse_adjoint_kernel_host.command
```

## Composition A — oracle mask plus a predictor

The useful post-hoc oracle is the **MEASURED heldout-120 20% output-support row**, not the cheap
4.7366% masks:

| 20% support provider | MEASURED global costate rel-L2 |
|---|---:|
| post-hoc top-output oracle | `0.026206284007981848` |
| cached source-margin mask | `0.5401736369574366` |

The named **`current_witness_oracle_mask_predictor_gap`** is therefore **DERIVED from those measured
rows** as an absolute relative-L2 gap of `0.5139673529494547`. This landing builds no predictor. The
open problem is a cheap current-witness provider that approaches the post-hoc oracle without first
computing the dense adjoint it is supposed to avoid, and that explicitly handles global SE.

The kernel makes such a provider computationally actionable once it exists; it does not make the
provider accurate. Admission still requires n600 renderer-gradient fidelity, optimizer-step regret,
through-R full-facet behavior, and in-loop wall improvement.

## Composition B — rank-r basis over K=2 state-stable reuse

The Metal program fuses `r` cotangents against one weight/support map. The compute law is:

```text
basis wins over K served steps iff T_basis / K < T_dense_per_step.
```

With no reuse (`K=1`), `r` VJPs do not beat one exact VJP merely because they are called a basis.
`K=2` is the bounded survivor because the separate #487 lane measured cached lag-5 costate cosines
near `0.85` while relative L2 stayed near `0.58`, then selected K=2 as the smallest changed-form
probe. The preserved K=2 n=1 diagnostic smoke is **MEASURED** costate cosine
`0.9989641547388` and renderer-gradient cosine `0.999494794470607`, but its admission is explicitly
`NOT_ADMITTED`. Its `1.6971077069821179x` teacher-slice number is **DERIVED DIAGNOSTIC**, and the
whole-epoch speedup is unknown.

This landing implements and stages a real rank-2/K=2 Metal timing against both two sequential dense
VJPs and the existing dense batched rank-2 kernel. That timing is **UNMEASURED/BLOCKED** here. It
does not touch or infer the concurrently owned n600 #487 files.

## Triality and fireability

- Equation: `src/tac/canonical_equations/custom_sparse_adjoint_achieved_ceiling_20260713.py`, with
  tests and an explicit append-only population function.
- DAG: `.omx/research/custom_sparse_adjoint_kernel_DAG_FEED_20260713.md`.
- DSL: **`REFUSED_NOT_FIREABLE`**. Creating a compute lever now would imply parity, wall, and
  accuracy authority that do not exist.

No canonical-registry empirical row was appended: the shared registry was already a hot in-flight
surface, and the only current Metal datum is a blocker, not a calibration anchor. The analytic
equation remains in its own module until a real host receipt exists.

## Storage, resumability, and containment

The benchmark creates only small JSON stage receipts and a log. Writes are atomic, the output
directory is single-writer locked, a rerun requires `--resume`, completed per-stage artifacts are
preserved, and contract drift fails closed. No bulky artifact is created, copied, moved, or deleted.
No training, live run, provider call, paid dispatch, archive evaluation, or pointer mutation occurred.

All implementation files are new. The live trainer, `witness_control/*`, pre-SE reopen,
`sparse_adjoint`, heavy-tail, vrghal, quant-tail-reliability, v9, and #432 files were not edited.

## Git transaction status

All new Python entities passed the repository review policy. The canonical serializer was invoked
with explicit file names, `base=new`, post-edit SHA-256 values, and `dag,dsl,equations` triality.
It failed before staging any byte because the current managed session exposes `.git` read-only:

```text
error: unable to create temporary file: Operation not permitted
fatal: adding files failed
```

`git diff --cached --name-only` remained empty. The source/artifacts are durable in the shared
working tree, but **no commit is claimed**. `req_R-git`: rerun the same canonical serializer from a
Git-writable custody surface; do not replace it with a direct commit or absorb unrelated dirty files.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `PROGRAM.md`, vehicle OS,
  sealed v7.5 §8 operating contract, and v8 decomposition spec.
- Canonical frontier, lane, subagent, task, equation, probe, gradient-anchor, dispatch, cost-band,
  continual-learning, sister findings/session/design/council, and last-24-hour directive surfaces.
- #486 memo and measurement receipt; #487 costate-reuse memo, closer memo, and preserved K=2 n=1
  smoke receipt.
- Existing #212 grouped Metal backward and #356 contraction/bit-identity kernel patterns.

## Pointer delta and exact reopen request

Pointer delta: **NONE**. This is a local throughput MEANS primitive.

`verdict_scope`: default-off compact frozen-Conv2d input-adjoint source plus inability of the current
execution substrate to run Metal. This negative does not reject the host kernel or sparse/low-rank
family.

`req_R`: on an actual Metal execution surface, run the staged host command; accept no timing unless
all four configurations are bit-identical to NumPy-fp32 dense-on-support in N=10 fresh processes.
Then measure the 125-shape wall factor and rank-2/K=2 factor. A live compute lever additionally
requires an n600-admitted oracle-mask predictor or state-stable reuse provider.
