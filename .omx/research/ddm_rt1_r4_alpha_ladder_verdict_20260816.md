# R4 — the flat band-repaint α-ladder: REFUTED 5/5, and the mechanism is a missing first-order term

**verdict:** `REFUTED` — sx1 §6's pre-registered falsifier fires. No α lowers d_seg.

verdict_scope: **FORMULATION** — flat prototype-palette band repaint at radius 1
(`band_repaint`, blend `out[band] ← (1−α)·out[band] + α·palette[lifted][band]`), over
α ∈ {0.02, 0.05, 0.10, 0.20, 0.40} plus the α=1 anchor, on the hv1 ep0634 base, **n600
full population — not a prefix, not a subset** (the tool has no stride flag; FRAMES=600
by construction, which exceeds sx1's "n≥120, never a prefix" minimum rather than meeting it).

**axis:** `[macOS-CPU advisory]` — never a score. `promotable: false`, `score_claim: false`.
**own-vehicle frontier UNMOVED:** hv1 ep0634 S 0.15959729295498598 @ 182,759 B `[contest-CUDA T4 n600]`.

Receipts (retained): `/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816/r4_alpha_ladder/`
— `R4_LADDER_VERDICT.json` · `R4_PREREGISTERED_POWERLAW.json` (written BEFORE the 4 test
rungs landed) · `ladder.log` · per-leg `RT1_LEG_band_r1_a*.json`.
Tool: `experiments/ddm_rt1_seg_roundtrip_decomposition.py leg --leg band --radius 1 --alpha A`.
Cost: 1,910 s wall, $0, rc=0.

## The measurement

α=0 needs no run: the blend is the identity at α=0, so the baseline is the ALREADY-MEASURED
base leg on the same work dir, same instrument, same 600 pairs — 34,938 flips_vs_gt,
d_seg 0.000296173095703125. Reusing it removes an instrument-drift confound rather than
adding one.

| α | flips_vs_gt | harm | seg (S units) | ΔS vs base | below base? |
|---:|---:|---:|---:|---:|:--|
| 0 (identity) | 34,938 | — | 0.029611 | — | — |
| 0.02 | 35,155 | +217 | 0.029801 | +0.000184 | no |
| 0.05 | 36,886 | +1,948 | 0.031269 | +0.001651 | no |
| 0.10 | 44,851 | +9,913 | 0.038021 | +0.008403 | no |
| 0.20 | 84,081 | +49,143 | 0.071276 | — | no |
| 0.40 | 305,655 | +270,717 | 0.259107 | — | no |

`REFUTED iff d_seg(α) > d_seg(0) for EVERY α` — **5 of 5, monotone, no ambiguity.**

## The mechanism — and it is stronger than the verdict

Fitting all five measured rungs:

    harm(α) = 2,319,821 · α^2.370268        r² = 0.999919

**p = 2.3703 > 1**, so `d(harm)/dα → 0` as `α → 0`. The flat band repaint has **no
first-order benefit term at all** — not a badly-tuned one, an absent one. Damage is purely
superlinear from the smallest α we can resolve (α=0.02 costs +217 flips and sits exactly on
the law). This is why the family cannot be rescued by choosing a gentler blend: there is no
small-α regime where flat content helps. The α knob is CLOSED, not mistuned.

## The pre-registered law survived, with a named drift

`R4_PREREGISTERED_POWERLAW.json` was written before the 4 test rungs ran, fixing
`harm(α) = 1,628,865·α^2.281044` from a 2-point exact fit (anchors α=0.02 and α=1; two
points determine two parameters, so those two rungs are anchors and the other four are
genuine out-of-sample tests).

| α | measured harm | predicted | ratio |
|---:|---:|---:|---:|
| 0.02 | 217 | 217.0 | 1.000 (anchor) |
| 0.05 | 1,948 | 1,754.6 | 1.110 |
| 0.10 | 9,913 | 8,527.9 | 1.162 |
| 0.20 | 49,143 | 41,448.0 | 1.186 |
| 0.40 | 270,717 | 201,449.6 | **1.344** |

All four survive the 1.5× bound — but the ratio drifts **monotonically upward** and reaches
1.344 at α=0.40, approaching its own falsifier. The 2-point fit anchored at the endpoints
systematically UNDER-predicts interior harm; the true interior exponent (2.3703) exceeds the
anchored one (2.2810). Report the law with that drift attached; do not extrapolate it past
α≈0.4 without re-fitting.

## What this does NOT close

The tool's own docstring is explicit that this rung is the **flat end** — the
FLAT-CONTENT CONTROL, not the family's cure. sq1 §2.4 measured both ends on the v4d vehicle:
flat/truth content pasted into the band gives η_net = −3.7640 (32/32 pairs harmed), while
paint **SOLVED against the frozen head** gives η_net = **+0.7895**. R4 prices the flat end
on hv1 for the first time and finds it superlinearly harmful with no first-order term.

**Solved-content band paint is a DIFFERENT family and is NOT tested here.** Do not cite R4
against it.

## Convergence (two instruments, one answer)

`.omx/research/ddm_et1_eta_on_the_priced_band_20260803.md` closed the r=1 address-band family
on a MEASURED pose-viable η (0.3017 ± 0.0246 vs a DERIVED bar 0.61491, 12.7σ short, with the
bar RISING in radius so widening cannot rescue). R4 reaches the same place from a completely
different direction — an α-sweep on the seg axis with a pre-registered power law — and adds
the mechanism et1 did not have: **the absent first-order term.** Independent instruments,
concordant verdict.

## Fire-order

1. The flat-content band rung is CLOSED. Do not re-run it at other α, other radius
   (et1 measured the bar RISES with radius), or other blend shape.
2. The live question the ladder does NOT answer: solved-vs-frozen-head band content
   (sq1's +0.7895 end). That is a separate charter and inherits none of R4's negative.
3. The measured law `harm(α) = 2,319,821·α^2.370268` (r² 0.999919, n=5, α∈[0.02,0.40]) is
   reusable as the flat-content damage price on this vehicle. Register it before citing.

Sisters: `[[ddm_et1_eta_on_the_priced_band_20260803]]` ·
`[[ddm_rt1_seg_roundtrip_decomposition]]` · `[[a_fit_residual_scatter_is_not_a_single_point_resolution_20260816]]`
(the sibling law from this same session: a fit's σ_log is scatter, not resolution — here the
relevant statistic is the 1.5× ratio bound, which the law clears while drifting toward it).
