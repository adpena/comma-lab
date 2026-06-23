# Frozen-instance horizon d_seg — cross-frame structure test → NO-GO (reconciliation FINAL)

- **Date:** 2026-06-23
- **Subagent:** `frozen_instance_horizon_20260623` ($0, CPU-light OMP=2; exact frozen SegNet).
- **Reconciles:** a90 deep-math byte-floor 16–262 B (`horizon_deepmath_multilens_20260623.md`)
  vs aa98 measured NO-GO −4.65e-9 d_seg/byte (`horizon_band_dseg_lever_20260623.md`), via the
  operator's frozen-instance reframe (`frozen_instance_horizon_crossframe_structure_directive_20260623.md`).
- **Axis:** `[contest-CPU advisory]` — single video `0.mkv` (1200 frames = 600 pairs) reproduces
  the 600-sample eval locally. Authority = exact frozen-SegNet argmax-disagreement.
- **Pointer:** UNMOVED at 0.19110. Any GO still needs byte-close + `upstream/evaluate.py`.
- **All score/byte math via `tac.contest_score`** (Catalog #391). Rate slope = `rate_term(1)/100`
  = **6.659e-9 d_seg/byte** (the canonical break-even a sidecar must beat).
- **Tool:** `experiments/probe_frozen_instance_horizon_crossframe.py` (stages trajectory /
  crossframe / taskspace). Reuses aa98's cached exact-SegNet argmaps
  `experiments/results/indep_dseg_bets_20260623_inflated/seg_argmaps.npz` — validated cached
  d_seg = **0.00055989** vs report 0.00055978 (Δ=1e-7, exact-scorer faithful). GT horizon row
  recovered from the EXACT GT argmax (road↔undrivable boundary) — no external pose file, no
  PyAV rgb24, fully authority-faithful.

## The decisive question

> Do the 1200 KNOWN per-frame horizon flip-sets collapse to a LOW-DIMENSIONAL function of the
> KNOWN ego-motion (horizon trajectory v_h(t) = cy + fy·tan(pitch(t))), so a PARAMETERIZED
> encoding (trajectory + low-entropy residual) is cheap enough to FLIP aa98's NO-GO economics?

**Answer: NO.** The horizon ROW v_h(t) is smooth + low-D exactly as a90's geometry predicts, but
the per-frame flip SET within the band is **near-full-rank, content-dependent scatter** that does
NOT track v_h(t). The ego-parameterized encoding is no cheaper than aa98's per-pixel encoding.
**aa98's NO-GO is FINAL; the horizon fix is trainer-side only.**

## Result 1 — v_h(t) IS smooth + low-D (a90's geometry CONFIRMED)

Per-frame GT horizon row v_h(t) = median over columns of the topmost-Road row (exact GT argmax):

| metric | value | reading |
|---|---|---|
| v_h mean | **193.6** | a90 predicted cy·sy = 192.0 (✓; the +1.6 is the road-edge vs vanishing-line offset) |
| v_h std | 2.45 rows | thin band — matches a90's ±3.5-row pitch-wander |
| v_h range | 189–200 | ±~5 rows over the whole clip = ~±0.75° pitch |
| temporal smoothness mean \|dv_h/dt\| | 0.59 rows/frame | smooth trajectory |
| trajectory byte cost (delta+zlib) | **213 B** | the LINE is cheap, as a90 said |

So the **WHERE** (the horizon line) is cheap (~213 B) — a90 was right about the geometry. The
question was always the **WHAT** (the per-pixel flip set), which the next result settles.

## Result 2 — the flip SET does NOT track v_h(t): cross-frame intrinsic dimension is HIGH (DECISIVE)

The whole reframe rested on the hope that the per-frame flip sets are a low-D function of the
smooth v_h(t). They are not:

| measurement | value | reading |
|---|---|---|
| corr(flip-centroid, v_h) | **0.20** | weak — flips do not follow the horizon row |
| flip-offset-vs-v_h std | **30.8 rows** | huge scatter (p10=−18, p90=+60) — flips spread far from v_h |
| **effective rank of the 600×band-pixel flip matrix** (SVD participation ratio) | **547 / 600** | **near-FULL-rank — flip patterns are ~independent across all 600 frames** |
| top-1 singular share | 0.43% | no dominant shared mode |
| components for 90% var | 502 / 600 | no low-D subspace |
| mean consecutive (v_h-aligned) Jaccard | **0.015–0.019** | aligned flip sets share almost nothing frame-to-frame |

**Cross-frame compression test (the byte-level proof):** encode the band flip set under three
models — (M0) aa98's per-frame-independent bitmap+class; (M1) v_h-row-aligned then zlib; (M2)
shared template + per-frame XOR residual:

| band | M0 independent | M1 v_h-aligned | M2 template-residual | best cross-frame / M0 |
|---|---:|---:|---:|---:|
| peak rows 180–200 (27,113 flips) | 49,381 B | 49,341 B | 82,746 B | **0.9992** |
| horizon rows 96–288 (64,608 flips) | 146,766 B | 146,782 B | 302,777 B | **1.0001** |

Aligning every frame to the ego-pitch horizon trajectory buys **0.08% / −0.01%** — i.e. nothing.
The template-residual model is strictly *worse* (no fixed template the flips XOR cheaply against).
**The cross-frame intrinsic dimension is ≈ the number of frames: each frame's horizon flips are a
fresh, content-dependent draw, not a low-D function of the known ego-motion.**

## Result 3 — the economics do NOT flip aa98's NO-GO (and the gap to a90's floor is 189–561×)

Parameterized encoding = trajectory (213 B) + best cross-frame flip-set bytes. Oracle Δd_seg =
force comp→GT in band (the maximum a perfect correction can buy). Net score via `tac.contest_score`:

| band | oracle Δd_seg | parameterized bytes | **Δd_seg / byte** | beats break-even (−6.659e-9)? | net ΔS | verdict |
|---|---:|---:|---:|:--:|---:|:--:|
| peak 180–200 | −2.30e-4 (−41%) | 49,554 | **−4.64e-9** | NO (0.70×) | +0.0100 | **NO-GO** |
| horizon 96–288 | −5.48e-4 (−98%) | 146,995 | −3.73e-9 | NO (0.56×) | +0.0431 | **NO-GO** |

The parameterized Δd_seg/byte (−4.64e-9) is **essentially identical to aa98's static per-pixel
−4.65e-9** — the frozen-instance / ego-parameterization gives **zero** improvement, because there
was no cross-frame structure to exploit. Every band still RAISES the score.

**Gap to a90's 16–262 B floor:** the measured parameterized cost is **49,554 B** (peak band) =
**189× a90's high floor (262 B)**, and **561×** for the horizon band. a90's floor assumed a clean
1-D curve with a near-deterministic 2-class label (its A1/A3 assumptions, marked
PROVISIONAL-PENDING-aa98). Those assumptions are **falsified by measurement**: the flips are not a
thin near-deterministic line (effective rank 547, offset std 30.8 rows, 5 mixed classes), so a90's
optimistic floor never materializes. a90's *mechanism* (geometry-pinned shallow-margin band) is
correct; a90's *byte floor* over-assumed cross-frame + cross-column determinism that does not exist
in the frozen instance.

## Result 4 — task-space angle (brief): direct argmax coding is NOT cheaper

| code | zlib bytes | vs frontier 177,169 B |
|---|---:|---:|
| full GT argmax field (dense task-space) | **581,266** | **3.28× larger** |
| full-resolution residual correction | 160,289 | 0.90× |

Storing the KNOWN dense argmax directly costs 3.3× the learned decoder — the HNeRV decoder is
already a far cheaper code for the scorer-targets than direct task-space storage. The only
conceivable task-space win is a *sparse structured* code, which is exactly the horizon flip-set the
crossframe stage proved is high-entropy. So the task-space angle does not rescue the horizon fix.

## Reconciled verdict (a90 ⊕ aa98 ⊕ frozen-instance)

- **a90 (geometry/mechanism): CONFIRMED.** v_h ≈ 192–194, thin band, shallow-margin flips on the
  road↔undrivable decision face. The horizon LINE is cheap (213 B).
- **a90 (byte floor 16–262 B): FALSIFIED by measurement.** It assumed a near-deterministic 1-D
  line + 2-class label; the actual flip set is full-rank content-dependent scatter (eff. rank 547,
  5-class, offset std 30.8 rows). The real parameterized cost is 49.5 KB+ = 189–561× the floor.
- **aa98 (NO-GO, −4.65e-9): FINAL.** The frozen-instance / ego-parameterized re-test — the one
  thing both prior memos left untested — gives the *same* −4.64e-9. There is no cross-frame
  structure to exploit; the per-frame flip sets are ~independent draws.

**The horizon d_seg is NOT a jump-independent pointer-move.** It is a diffuse, per-frame
content-dependent, near-full-rank scatter pinned to the geometric horizon but not encodable cheaply
either statically (aa98) or via ego-motion (this probe). The ONLY lever that pays is **trainer-side
base reconstruction fidelity at the horizon band** (horizon-band-weighted recon loss / decoder
capacity targeting camera rows ~421–444 = the shallow-margin road↔undrivable boundary). Any $0
sidecar — static or ego-parameterized — is byte-dominated even with a perfect oracle.

## 6-hook wire-in (Catalog #125)

- **#1 sensitivity-map: ACTIVE** — the flip set is geometry-pinned (v_h ≈ 193) but content-dependent
  + near-full-rank; a reusable seg-sensitivity prior: route trainer recon weight to the horizon band,
  do NOT spend sidecar bytes there.
- **#2 Pareto: N/A** — no admitted candidate (all NO-GO).
- **#3 bit-allocator: ACTIVE (advisory)** — "horizon d_seg is full-rank content scatter; sidecar
  byte-dominated; allocate to the trainer not the archive."
- **#4 cathedral-dispatch: N/A** — advisory, non-promotable, no archive change.
- **#5 continual-learning: N/A** — `[contest-CPU advisory]`, non-promotable.
- **#6 probe-disambiguator: ACTIVE** — `probe_frozen_instance_horizon_crossframe.py` is the
  cross-frame-structure disambiguator that FINALIZES the a90/aa98 reconciliation.

Mission contribution: **frontier_protecting** — closes the highest-prior remaining $0 d_seg path
(the frozen-instance ego-parameterized horizon sidecar) with a decisive intrinsic-dimension
measurement, and definitively redirects the horizon d_seg attack to the trainer side. Sister-DISJOINT
from aa98/a90 (it measures the cross-frame structure they left untested). NON-PROMOTABLE
[contest-CPU advisory]. Pointer UNMOVED 0.19110.
