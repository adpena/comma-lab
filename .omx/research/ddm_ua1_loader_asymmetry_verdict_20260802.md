# ddm_ua1 — the GT/output loader asymmetry: characterized, priced, REFUTED as a lever

**Date:** 2026-08-02 · **Arm:** `ddm_ua1_loader_asymmetry` · **Axis:** `[macOS-CPU advisory]`,
canonical `DistortionNet` path · **score_claim=false, promotable=false**

---

## VERDICT (answer first)

**The asymmetry is REAL, exactly characterized, and NOT a lever. Pre-registered falsifier FIRES,
at FORMULATION scope, with one correction to its own wording.**

The decisive number: the GT manifold is **surjective onto the scorer's input space**. A
GT-manifold-constrained encoder can hit an arbitrary scorer target to **2.86e-14 rms** (fp64
machine precision). There is no scorer input we can produce that a GT-constrained encoder cannot.

The surplus 13.79% of colour space that only we can emit is **provably unused**: **0 of 12,208,032**
camera pixels emitted by the real solver landed in it.

Rate leg: constraining to the GT manifold at matched accuracy costs **+2.62% MORE** bytes, not
fewer. The GT manifold is not the cheap region.

**Correction to the falsifier's wording.** Leg (a) said "entirely within ker(D)". That is **FALSE** —
the extra DOF *do* move D's output. The operative condition is different and stronger, and it holds:
`D(GT-manifold) = D(our space)`. Surjectivity, not kernel-containment, is what makes the surplus
worthless.

---

## 1 · PREDICT-THEN-DIFF on the upstream reads

Predictions recorded before opening each file; `upstream/` never modified.

| Predicted | Actual | |
|---|---|---|
| `AVVideoDataset`: PyAV → yuv420p → `yuv420_to_rgb` | `frame_utils.py:200-201`, planes read raw, no reformat | MATCH |
| `TensorVideoDataset` = memmap uint8 (N,874,1164,3) | `frame_utils.py:231`, `format='raw'` | MATCH |
| BT.601 limited-range, clamp | `frame_utils.py:176-182` | MATCH |
| chroma upsample **bilinear** *(flagged as my highest uncertainty)* | `frame_utils.py:173-174` `mode='bilinear', align_corners=False` | MATCH |
| Dali path also calls `yuv420_to_rgb` | **WRONG.** DALI never calls it; the docstring says it *reimplements* nvdec. GT-on-CUDA is nvdec hardware. | **DIFF** |
| GT is float | **WRONG.** `frame_utils.py:183` `.round().to(torch.uint8)`. Both sides are uint8 in the same cube. | **DIFF** |

**Two corrections to the briefing that reshape the problem:**

1. **Luma is NOT subsampled.** Only U,V are 2×2. This is a *chroma* constraint, not a blanket
   "half the bits".
2. **GT is uint8.** The asymmetry is about *which* points are reachable, not about dtype or precision.

Both scorers read through the **same** D (`modules.py:109` and `:73`; PoseNet resizes *before*
`rgb_to_yuv6`), already pinned in-repo at fp64 parity 6.45e-12
(`src/tac/optimization/ddm_ll1_window_solve.py`).

---

## 2 · The manifold, measured

`experiments/ddm_ua1_loader_asymmetry_probe.py` → `.omx/research/ddm_ua1_manifold.json`

**Degrees of freedom** (MEASURED from the real decoded planes):

| | scalars | bits / camera px |
|---|---:|---:|
| GT: uint8 luma 874×1164 + uint8 chroma 2×437×582 | 1,526,004 | **12** |
| ours: unconstrained uint8 RGB | 3,052,008 | **24** |

**Occupied RGB sub-lattice** (EXACT enumeration of all 2²⁴ triples, not sampled — the decode is an
invertible affine map, so reachability is exactly "does the implied (y,u,v) fit the box?"):

| chroma model | GT-reachable fraction of the 256³ cube |
|---|---:|
| continuous (the real GT — bilinear blending makes chroma effectively continuous) | **86.21%** |
| integer (pure unblended sample) | 17.59% |

→ **13.79% of colour space is exclusive to us.**

> **I predicted ~50.4% and was wrong.** My luma-lattice argument ignored that the ±0.5 rounding
> slack on *each* of r,g,b widens the feasible g-interval to 1.704 — wider than the 1.164 luma
> lattice spacing, which dissolves the lattice constraint entirely. The binding constraint is the
> **chroma box**, not the luma lattice. Corrected by measurement.

**POSITIVE CONTROL:** real decoded GT frames are **100.0%** inside the YUV box (`in_yuv_box_frac`
= 1.0, n=2 frames, 1,017,336 px each). The gamut test is calibrated.

**The collapse (DERIVED, verified 1.7e-13):** because decode is affine, a weighted average of
decoded RGB equals the decode of the weighted-average YUV. So **the scorer sees the GT source only
through (Ȳ, Ū, V̄)** — three numbers per private 2×2 window.

---

## 3 · Exploitability: the constraint splits three ways, and only one binds

`experiments/ddm_ua1_gamut_exploitability.py` → `.omx/research/ddm_ua1_exploitability.json`
(n=6 pairs / 12 frames, canonical `DistortionNet`, all arms aiming at the identical target D(GT)).

| constraint | status | evidence |
|---|---|---|
| **gamut** (13.79% exclusive) | **NOT BINDING, NOT USABLE** | **0 / 12,208,032** camera px out-of-gamut |
| **chroma bandwidth** (254,334 shared vs 1,017,336 free) | **NOT BINDING** | CGNR residual **2.86e-14** |
| **chroma uint8 granularity** | the only term that binds | 0.148 rms, ≤0.031 S |

**Why bandwidth cannot bind (DERIVED, then MEASURED):** the composed operator `A = D∘U` maps
254,334 chroma unknowns → 196,608 scorer constraints — **under-determined by 1.29×**. A real-valued
solution generically exists, and CGNR finds it at machine precision. This is
**target-independent**: A is surjective onto *all* of R^(384×512), so the result holds for any
render, not just this frame's.

**Arms** (`S_dist` = 100·d_seg + √(10·d_pose); rate excluded, identical across arms):

| arm | d_seg | d_pose | S_dist | ΔS vs FREE | delivery rms |
|---|---:|---:|---:|---:|---:|
| X0 (unsolved bicubic) | 1.585e-4 | 2.020e-4 | 0.060801 | +0.0577 | 0.361 |
| **FREE** (unconstrained, = ddm_ll1) | 9.325e-6 | 4.539e-7 | 0.003063 | — | 0.024 |
| GTPROJ (projected onto manifold) | 2.009e-4 | 1.202e-4 | 0.054764 | +0.0517 | 0.990 |
| GTSOLVE (constrained optimum) | 1.263e-4 | 4.599e-5 | 0.034076 | **+0.0310** | 0.275 |

**POSITIVE CONTROL LIVE:** X0 scores +0.0577 worse than FREE. The harness moves.

**Two confounds I had to remove before this table was admissible:**

1. My first GTSOLVE used 150 Adam steps against a purpose-built exact window solve — comparing
   *solver strength*, not feasible-set size. Replacing Adam with CGNR halved ΔS (0.0253→0.0125) and
   exposed the real-valued residual as 2.86e-14. **Solve, don't search.**
2. A rounding-refinement loop I added produced **zero** improvement (0.14935 identical). Rather than
   ship a stub that claims work it does not do, I removed it and report the naive bound with its
   analytic floor: independent uint8 rounding predicts 0.2887·√0.25 = **0.1444**, measured **0.1480**.
   They agree, so naive rounding is *at* the independent-rounding floor.

---

## 4 · The price, honestly bounded

**ΔS ≤ +0.031, and it is an UPPER BOUND that is also unstable.**

- *Upper bound* because a true integer least-squares could exploit the 1.29× slack (57,726 spare
  chroma DOF/channel) and land below the independent-rounding floor. The lower bound is 0 (the
  continuum residual is exactly 0).
- *Unstable*: the estimate rises with n — **0.0125 (n=2) → 0.0164 (n=4) → 0.0310 (n=6)**, a 2.5×
  spread. This is a small-n variance signal, not a converged price. **Do not quote 0.031 as a number.**

Even taken at face value, 0.031 S is rate-equivalent to 0.031·37,545,489/25 ≈ **46.6 KB** — but it is
**not available to spend**, because it is not a lever we can pull. It is the amount we would *lose*
if someone forced us onto the GT manifold. Nobody is.

---

## 5 · Rate leg — closed, MEASURED (proxy)

The only mechanism by which the asymmetry could touch rate is if the cheapest scorer-acceptable
raster lay *outside* the GT manifold. lzma over each arm's raster (n=6 pairs):

| arm | lzma bytes | vs FREE |
|---|---:|---:|
| GTPROJ (constrained, inaccurate) | 16,886,268 | −9.59% |
| X0 (unsolved) | 16,999,756 | −8.99% |
| GT_reference (the real GT) | 17,763,120 | −4.90% |
| **FREE** | 18,678,408 | — |
| **GTSOLVE** (constrained, accurate) | 19,167,708 | **+2.62%** |

**GTSOLVE and GTPROJ sit on the same manifold and land on opposite sides of FREE.** What predicts
cost is *smoothness/accuracy*, not manifold membership. Constraining to the GT manifold at matched
accuracy costs **more**, not less. Falsifier leg (b) holds.

*Caveat, welded on:* this is a PROXY. `archive.zip` stores tokens/weights, not the camera raster
(`evaluate.py:63`). It is the right proxy for the structural question — is the GT manifold the cheap
region? — and the answer is no.

**Corollary (a would-be rate lever, now DEAD):** "our chroma is over-resolved relative to what the
scorer needs" is FALSE. A is under-determined, not over-determined — the scorer's 384×512 chroma
demand is full-rank against the 437×582 plane. There is no chroma rank deficiency to harvest.

---

## 6 · What this does NOT say

The FREE arm *is* `ddm_ll1`'s window solve, and it beats plain `clip(rint(U(r)))` by a measured
ΔS −0.01441 (its own docstring, n=3). **That value is real and is NOT what this memo refutes** —
it comes from D's disjoint-window structure, not from the loader asymmetry. `ddm_ll1` is
default-OFF in `ddm_tr1_runtime.render_frame1_camera_uint8` (line 1382), and that default is a
**reasoned deferral with a named blocker** — v4d's frame_0 warp resamples *across* private windows,
so d_pose through the warp is UNMEASURED — not a forgotten default. Closing that gate is a far
better-supported item than anything the asymmetry offers.

---

## 7 · Scope and what is owed

**verdict_scope: FORMULATION.** Refuted: *the GT/output loader asymmetry as a source of exploitable
headroom on seg, pose, or rate.* Not refuted: D-structure exploitation generally (`ddm_ll1` is
alive and measured).

Load-bearing claims and their strength:

- **Strong, n-independent:** surjectivity (2.86e-14), the 1.29× under-determination, the affine
  collapse (1.7e-13), the exact 86.21% enumeration, 0/12,208,032 out-of-gamut. These are structural;
  the first three are target-independent.
- **Weak, n=6, unstable:** the +0.031 price. Owed if anyone wants it: n600, and an integer
  least-squares chroma solve to tighten the upper bound.
- **Proxy:** the rate table. Owed if load-bearing: the real archive coder.

Nothing here moves the exact pointer. Own-vehicle frontier remains **v4d 0.9639878**.

**Artifacts:** `experiments/ddm_ua1_loader_asymmetry_probe.py`,
`experiments/ddm_ua1_gamut_exploitability.py`, `.omx/research/ddm_ua1_manifold.json`,
`.omx/research/ddm_ua1_exploitability.json`.
