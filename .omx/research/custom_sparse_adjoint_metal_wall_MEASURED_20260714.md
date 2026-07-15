---
title: "D43 custom sparse-adjoint Metal wall — MEASURED (blocker cleared)"
date_utc: "2026-07-14"
lane_id: "custom_sparse_adjoint_kernel"
research_only: true
score_claim: false
pointer_moved: false
training_performed: false
paid_dispatch: false
live_run_mutated: false
axis: "[macOS-MLX/Metal research-signal] — advisory, NON-score, non-promotable MEANS"
verdict: "PARITY-GREEN; WHOLE-NETWORK WALL 0.708x (SLOWDOWN) vs #212 dense; DERIVED 2.2086x ceiling NOT realized"
verdict_scope: "this custom per-input-site no-atomic compact Metal kernel scheme; this M5 Max/40-core-GPU + MLX fingerprint; synthetic evenly-spaced support geometry at sealed #486 family fractions; frozen-Conv2d input-VJP only. NO verdict on mask prediction, global-SE, nonlinear full-SegNet VJP, optimizer regret, another chip/runtime, score, or pointer."
supersedes: "custom_sparse_adjoint_kernel_20260713.md (BUILT_DEFAULT_OFF; METAL_ADMISSION_BLOCKED) — the Metal wall is no longer UNMEASURED"
---

# D43 — custom sparse-adjoint Metal micro-bench: the wall is MEASURED

> **MEANS.** Backward-throughput compute primitive only. Cannot move the contest pointer
> (0.19108 submittable / 0.18804 borrowed bank, both UNMOVED). Advisory `[macOS-MLX/Metal
> research-signal]`, never a score authority. NO training, NO paid dispatch, NO archive eval.

## What changed: the blocker is cleared

The 2026-07-13 predecessor (`custom_sparse_adjoint_kernel_20260713.md`) built the kernel and proved it
bit-identical on **CPU NumPy-fp32** in 40/40 trials, but the mandatory **Metal** parity gate + wall
measurement were `UNMEASURED/BLOCKED` because that session hit `No Metal device available`
(headless/sandboxed). **This session has a live Metal device** (`Device(gpu,0)`, MLX eval OK), so the
staged host bench (`tools/bench_custom_sparse_adjoint_kernel.py`) ran to completion. The
`METAL_WALL_OWED` half of the D43 ledger row is now discharged with a real measurement — not a
naive→binary NO-GO: this is the **OPTIMAL FORM** (the real custom Metal kernel, no-atomic per-thread,
`fp contract(off)`, fixed reduction order) measured against the existing #212 dense Metal grad-input
kernel across all 125 real frozen-SegNet convolution shapes.

Receipt (durable in the shared checkout, `experiments/results/` is gitignored):
`experiments/results/custom_sparse_adjoint_kernel_metal_bench_20260714/measurement_receipt.json`
(329 KB, `status=GREEN_PRIMITIVE_ONLY`, `pointer_delta=NONE`).

## MEASURED numbers (advisory)

**Parity gate — GREEN.** `GREEN_BIT_IDENTICAL`, 10 fresh cross-processes, **1** unique output hash
(every process produced the identical bytes). Max abs deviation of the sparse Metal output vs the
existing **#212 dense Metal** kernel over all 125 shapes: **6.68e-6** (fp32 reduction-order only; the
sparse kernel is bit-identical to the NumPy-fp32 authority by construction).

**Whole-network wall replay (125 shapes, warmups 2, repeats 7):**

| Quantity | Label | Value |
|---|---|---:|
| flagship DERIVED arithmetic ceiling | DERIVED | 2.208577x |
| exact valid-tap arithmetic ceiling | DERIVED | 2.200230x |
| dense median (existing #212 Metal) | MEASURED | 65.356 ms |
| sparse median (this kernel) | MEASURED | 92.342 ms |
| **achieved wall speedup** | **MEASURED** | **0.7078x (SLOWDOWN)** |
| efficiency η = achieved/ceiling | MEASURED/DERIVED | **0.3205** |
| achieved/flagship-ceiling ratio | MEASURED | 0.3205 |

**Per-family wall (support fraction = sealed #486 family-active):**

| Family | support frac | n shapes | median speedup | frac shapes >1x |
|---|---:|---:|---:|---:|
| segmentation_head | 0.0474 | 1 | **1.63x** (win) | 1.00 |
| decoder | 0.2803 | 10 | **1.06x** (marginal) | 0.60 |
| encoder | 0.9835 | 114 | **0.76x** (loss) | 0.12 |

**rank-2 / K=2 state-stable basis-fusion (one real decoder shape, `decoder.blocks.2.conv2.0`):**
amortized speed vs two sequential dense VJPs = **0.4923x** (SLOWDOWN); no-reuse vs one dense VJP =
0.2462x. Basis-fusion does **not** beat dense at K=2 here either. (Accuracy of the K=2 reuse is a
separate owned #487 concern; this is timing-only.)

## The honest verdict

The **DERIVED 2.2086× arithmetic ceiling is real but is NOT realized as wall time** — the custom
sparse kernel is **0.71× (a slowdown)** whole-network. Cause, decomposed by the receipt's own law
`R_t = T_sparse − T_dense/C`: the residual charges gather/map traffic, launch latency, and occupancy
loss on the per-`(input-site,input-channel)`-thread no-atomic scheme; the bench cannot attribute which
dominates without a profiler, but the **structural reason is clear from the per-family split**: 114 of
125 shapes are the **encoder at 98.3% support** — there is essentially no sparsity to exploit there, so
a sparse kernel is dominated *by definition* and only pays the compaction/gather overhead. The kernel
wins **only** where support is genuinely sparse (seg-head 1.63×, decoder marginal 1.06×), and those
layers are a small fraction of total backward FLOPs.

**Verdict scope = INSTANCE/FORMULATION**, not a family kill:
- **This whole-network formulation is DOMINATED** on this M5 Max/MLX substrate — the arithmetic ceiling
  oversold the realizable wall because it aggregated a near-dense encoder into the "sparse" budget.
- The family stays OPEN only via the **hybrid layer-routing reformulation** below — and even there the
  measured wins are modest and gated behind the unbuilt oracle-mask predictor (0.514 rel-L2 accuracy
  gap, MEASURED predecessor). Net EV of the whole D43 sparse-adjoint line is now **LOW** and honestly
  bounded, not "owed/unknown."

This is the value of running the optimal form: the predecessor's `BLOCKED` state left the 2.2086×
ceiling looking like banked headroom. The measurement shows it is not — one fewer phantom lever.

## Build-ticket (OWNED-BUILD — design now, build when the owning arm frees)

**Reformulation R-D43-hybrid — layer-routed sparse-adjoint (the only surviving optimal form).**

- **What:** route the backward input-VJP per layer by measured support: use `metal_sparse_adjoint`
  ONLY on layers whose active-support fraction is below the break-even (empirically ≈ 0.28: decoder
  ≥1.06× median, seg-head 1.63×); keep the existing **#212 dense** grouped-conv Metal kernel for the
  encoder (0.98 support → 0.76×). A static per-shape router keyed on the sealed #486 family fraction.
- **Where it wires (arm-owned — DO NOT touch from this arm):** the costate/backward path that composes
  the frozen-SegNet layer VJPs (`witness_control` / costate provider). The `metal_sparse_adjoint`
  primitive itself is complete and bit-identical; no edit needed there.
- **Hard precondition (dominates admission):** an accuracy provider — a **cheap current-witness
  oracle-mask predictor** that approaches the MEASURED 20% post-hoc oracle (rel-L2 0.0262) without
  first computing the dense adjoint it is meant to avoid, AND that explicitly handles the 23 global
  squeeze-excite reductions. Until that exists, the router has no sparse layers to route to (the cheap
  source-margin mask sits at 0.540 rel-L2 = the 0.514 gap). **The bench proves the kernel is
  actionable; it does NOT make the predictor accurate.**
- **Measurement gate (before any live-lever admission):** (1) hybrid-routed whole-network wall > 1.0×
  on this substrate; (2) n600 renderer-gradient cosine ≥ the #487 admission bar through-R;
  (3) optimizer-step regret null over a stage; (4) in-loop epoch wall improvement. Any miss → stays
  default-off research primitive.
- **EV note:** even a perfect router caps at the decoder/seg-head share of backward FLOPs at ~1.1–1.6×
  on those layers; whole-network best case is well below 2× and is bottlenecked by the accuracy
  provider, not the kernel. Prioritize the mask-predictor question over more kernel work.

## OWED triality legs (arm-owned → serialized/routed, NOT touched here)

- **eq (OWED):** append the MEASURED achieved/ceiling anchor
  (`achieved=0.7078x`, `η=0.3205`, `2.2086x` DERIVED unrealized) to
  `custom_sparse_adjoint_achieved_vs_ceiling_v1`
  (`src/tac/canonical_equations/custom_sparse_adjoint_achieved_ceiling_20260713.py`) — canonical_equations
  is arm-owned; the append lands via the owning arm's serializer.
- **DSL (OWED):** `REFUSED_NOT_FIREABLE` remains correct — no admitted accuracy provider exists, so no
  compute lever is created. Unchanged from predecessor.
- **ledger (OWED):** update `.omx/state/deferral_ledger.md` D43 row
  `BUILT-DEFAULT-OFF / METAL-WALL-OWED` → `METAL-WALL-MEASURED: 0.708x whole-network SLOWDOWN
  (parity GREEN); reopen = hybrid layer-routing + oracle-mask predictor, EV LOW` — `.omx/state/` is not
  in this arm's touch set; routed as an owed one-line edit.

## STORES CONSULTED

- `CLAUDE.md` (NO-FAKE; verdict-scope ladder; no-naive→binary-NO-GO; MLX/Metal never a score authority),
  `docs/operating_manual_craft_handoff.md`, the drain prompt.
- `.omx/research/naive_nogo_rescoping_audit_498_20260715.md` (D43 the #3 reopenable, cheap-local probe),
  `.omx/state/deferral_ledger.md` (D43 row `BUILT-DEFAULT-OFF / METAL-WALL-OWED`),
  `custom_sparse_adjoint_kernel_20260713.md` (predecessor, the BLOCKED state),
  `tools/bench_custom_sparse_adjoint_kernel.py`, `src/tac/local_acceleration/metal_sparse_adjoint.py`.

Pointer delta: **0.0000000000.**
