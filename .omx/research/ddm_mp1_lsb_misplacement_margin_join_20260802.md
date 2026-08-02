---
title: "mp1 — the misplacement is a RESAMPLING error, it is 2.3x WORSE on the live vehicle than on the ideal render, and it IS margin-enriched"
lane_id: lane_ddm_mp1_lsb_misplacement_margin_join_20260802
arm: ddm_mp1 (task #897, paying task #898's code debt)
date_utc: 2026-08-02
authority: "[macOS-CPU advisory] SCORER-FREE sweep + join on cached frozen-SegNet fields. NON-PROMOTABLE. score_claim=false."
pointer_delta: "UNMOVED. No exact row. This arm sizes a lever and hands it to the build."
verdict: OPEN_AND_SIZED
verdict_scope: >
  FORMULATION-level for the ANSWER (the reach is DERIVED from a single-anchor
  first-order gain law, never scorer-measured). MEASUREMENT-level and n600-exact for
  the misplacement field itself, its ideal/live decomposition, the resample-vs-quant
  split, the camera lattice, and the margin join.
council_predicted_mission_contribution: frontier_breaking
---

# mp1 — LSB misplacement, joined against the margin field

**THE ASK (#897):** can misplacement carry a pixel across its margin? Under the null it is
negligible and the row closes. **THE DEBT (#898):** cg2 landed 8 scalars and NO code, so the
per-pixel field never existed.

**THE VERDICT: the null is REJECTED and the row OPENS.** Misplacement is margin-ENRICHED
(1.55–2.06x in the annulus), the live vehicle's misplacement is **2.31x LARGER** than cg2's
ideal-render figure (not smaller, as the transfer argument assumed), and the camera lattice can
absorb essentially all of it at **zero counted bytes**. First-order reach on the live vehicle:
**ΔS ≈ 0.0418** — 10.4 % of the seg-axis gap. That number is DERIVED, not measured; §7 names the
one measurement that would settle it.

---

## §1 — the debt is paid: the instrument exists, and it is controlled

`experiments/ddm_mp1_lsb_misplacement_margin_join.py`. Chunked, resumable (a completed chunk is
never recomputed), overlap-refusing **from the first line** rather than after a near-miss. It
persists the per-pixel field cg2 discarded: `field_l2_f16` + `field_maxabs_f16`, 600x384x512 per
leg, on the SSD tier (`/Volumes/VertigoDataTier/pact/ddm_mp1_20260802/`, 6 chunks x 75 MB x 2
legs). Every number below is re-derivable from those chunks without re-running the sweep.

**Five positive controls, each with its denominator.** A rebuilt instrument is a new error
source; none of its output is believable until these pass.

| # | control | result | denominator |
|---|---|---|---|
| A | this module's `D` vs `segnet.preprocess_input` | **max-abs 0.0** | 3 frames |
| A' | `SegNet(D(gt_f1))` argmax vs cached `lstars` | **0 mismatches** | 589,824 px |
| B | my `clip(rint(U(r)))` vs `render_frame1_camera_uint8` | **0 mismatches** | 9,156,024 camera values |
| C | cg2's 8 n600 scalars | **reproduced, ≤1e-6 rel** | 353,894,400 values |
| D | aggregator refuses overlap / gap / short-cover; accepts intact | **4/4** | 6 chunks |
| E | break-even `W` recomputed from the score function | **1.27310821533 B/flip** | exact |

Control C is the load-bearing one, and it is a genuine cross-instrument check: cg2 used a float64
separable matmul, this module calls the torch op the scorer itself calls. They agree to 1e-7 on
the mean and 1e-6 on the fractions. **Both instruments are sound; cg2's numbers stand.** Control A
is stronger than cg2 had available — it proves the operator is not merely *equivalent* to the
authority's resize, it **is** it (`upstream/modules.py:109`).

Two more cg2 results reproduced independently, at wider scope: blind camera rows/cols/pixels
**106 / 140 / 230,904**, and the block structure — but measured over the **full** grid (all 874
camera rows, all 1164 camera cols) rather than one probe pixel: **exactly 2 taps per output row
and 2 per output col, every camera row/col feeds at most ONE output, blocks disjoint, weights sum
to exactly 1.0**. cg2's blast-radius-1 claim holds everywhere, not just where it was probed.

---

## §2 — the headline number was never a rounding number (decomposition, MEASURED)

`e_real = D(clip(rint(U(r)))) - r` splits **exactly** into

* `e_resample = D(U(r)) - r` — U then D is not the identity;
* `e_quant = D(clip(rint(U(r)))) - D(U(r))` — uint8 rounding, D-averaged.

| n600, 353,894,400 values | ideal (`r = y*`) = cg2 | **live v4d** |
|---|---|---|
| mean abs | 0.19815 | **0.32926** |
| **rms** | 0.34517 | **0.79899** |
| max | 29.844 | 20.411 |
| values > 0.5 LSB | 8.233 % | **12.979 %** |
| values > 1 LSB | 1.056 % | **4.124 %** |
| **rms `e_resample`** | 0.30111 | **0.77254** |
| **rms `e_quant`** | 0.17200 | 0.19567 |
| resample share of variance | **76.1 %** | **93.5 %** |
| camera values clipped at 0/255 | 22,442,485 (1.23 %) | 2,238,378 (0.12 %) |

**"LSB misplacement" is a misnomer if read as a rounding problem.** Rounding is 25 % of the
variance on the ideal render and **6 %** on the live one. The dominant term is the resampling
round trip. This matters for the cure: a dither that only fixes rounding leaves 94 % of the live
error on the table.

---

## §3 — cg2's number does NOT transfer, and it errs in the opposite direction

The transfer argument was: the live render comes from a 24x32 uint4 token grid, so it must be
smoother than the true scene, so `U∘D` (which only bites high-frequency content) must hurt it
less. **MEASURED, on the real v4d archive (360,238 B, 6 members, sha256 `f1f32880…`): FALSE, and
backwards.**

| frame | live render 384-grid Laplacian rms | true scene (`y*`) | live render error vs `y*` |
|---|---|---|---|
| 0 | **21.49** | 13.07 | 44.07 |
| 137 | **21.23** | 10.98 | 45.80 |
| 599 | **21.44** | 11.63 | 43.83 |

The v4d renderer emits **~1.8x more high-frequency energy at the 384 grid than the scene actually
has** (it is in range — `[0.13, 255.0]`, zero out-of-range values — so this is not a clipping
artifact). It is not smoother than reality; it is **rougher**. That is the whole mechanism of the
2.31x: `U∘D` mangles exactly the content the renderer over-produces.

Consequence for the ledger: **cg2's `0.34517` is not an upper bound on the live vehicle's
realization error.** It is a *lower* bound. cg2 hedged correctly ("informative only if the
vehicle's render error is of comparable size") but the direction was unknown until now.

---

## §4 — the join: the null is REJECTED

Per-pixel misplacement aggregated to `q_p = ||e_p||_2` over the 3 channels. **Why that
aggregation:** the first-order margin perturbation is `dm = <grad m, e>`, so under an unaligned
gradient a pixel enters through its channel-L2. (`chan_maxabs` is persisted too — it is what
cg2's per-channel ">0.5 LSB" fraction keys on — and is reported in the JSON.)

Enrichment = bin-mean misplacement / global-mean misplacement, n600, 117,964,800 pixels:

| margin bin | area frac | ideal enrichment | live enrichment |
|---|---|---|---|
| [0, 0.05) | 0.00141 | **2.058** | **1.554** |
| [0.05, 0.1) | 0.00141 | 2.046 | 1.549 |
| [0.1, 0.2) | 0.00280 | 2.039 | 1.541 |
| [0.2, 0.5) | 0.00820 | 2.008 | 1.517 |
| [0.5, 1) | 0.01288 | 1.868 | 1.446 |
| [1, 2) | 0.02163 | 1.435 | 1.282 |
| [2, 4) | 0.05469 | 1.064 | 0.964 |
| [4, 8) | **0.87932** | 0.956 | 0.981 |
| [8, inf) | 0.01766 | 1.035 | 0.983 |

**Misplacement is NOT margin-independent.** It is 1.5–2.1x concentrated exactly where flips
happen, monotone in `-margin`, and it **saturates** below m ≈ 0.5 rather than diverging. The
mechanism is unsurprising in hindsight and worth stating plainly: realization error tracks image
detail, and argmax boundaries sit on image detail. The bin RMS moves with the bin mean (ideal
[0,0.05): mean 0.833 / rms 1.543 vs bulk mean 0.387 / rms 0.512), so this is a shift of the whole
bin, not a few outliers dragging a mean.

**Per class** (ideal leg): Lane carries **28.7 %** of the modelled reach on **0.59 %** of the
area, with mean misplacement 1.185 LSB = 2.93x the global mean. Road 39.0 %, Undrivable 13.6 %,
Movable 13.3 %, MyCar 5.5 %. Lane-dominance again, as in every other lens.

---

## §5 — the reach, and the honest label on it

Law (registered, DERIVED): `||grad m_p|| = G/cosh(m_p/2)`, `G = 0.060599`, from
`segnet_head_rank4_linear_flipdist_v1` + `tools/adversarial_evasion_fisher_null_probe.py`.
Flip probability is **one-sided** — `P = Phi(-m_p/sigma_p)` — because a positive margin
perturbation only deepens the winner's lead.

| | ideal | live |
|---|---|---|
| d_seg, measured field | 3.2443e-4 | **4.1849e-4** |
| d_seg, shuffled null (same magnitudes, margin-decorrelated) | 1.5862e-4 | 2.7113e-4 |
| **enrichment vs shuffled** | **2.045x** | **1.544x** |
| d_seg, uniform-rho null (the evasion probe's margin-blind model) | 2.3453e-4 | 5.4490e-4 |
| d_seg, 1-sigma threshold variant (probe-comparable) | 8.1624e-4 | 1.0555e-3 |

Two things worth reading carefully. (1) The **shuffled** null is the clean control: same marginal
distribution of misplacement, correlation with margin destroyed. Enrichment 2.05x / 1.54x is the
join's actual answer. (2) On the live leg `enrichment_vs_uniform` is **0.768 — below 1**. That is
not a contradiction: the live field is heavy-tailed (mean 0.694, rms 1.384), and much of its RMS
sits in a few very large errors parked in the deep interior where they cannot flip anything.
Concentration wastes amplitude. Both statements are true at once — the field is *enriched at the
boundary* relative to a shuffle, and *less efficient* than a uniform field of equal RMS.

Cross-check against the registered probe: its margin-blind table gives d_seg 0.001614 at rho=1.0;
this module's dose-response gives 6.818e-4 at rho=1.0, a factor 2.37 lower, which is the
one-sided-vs-threshold difference. The 1-sigma-threshold row above is the like-for-like number and
sits within 1.5x of the probe scaled to the same rho. **Consistent, not identical — and the
difference is accounted for, not waved at.**

---

## §6 — how much is recoverable: the camera lattice, solved EXACTLY

Because the blocks are disjoint (§1), each scorer value is an independent 4-tap problem: choose
`c in uint8^4` to hit a target `t`. Solved **exactly** over the full 256^4 lattice (meet-in-the-
middle, 64 sampled blocks x 256 targets = 16,384 exact solves):

| | achievable | plain `rint` at the same targets |
|---|---|---|
| mean abs residual | **2.06e-5 LSB** | 0.2462 LSB |
| rms residual | **2.98e-4 LSB** | 0.2854 LSB |
| p99 | 2.70e-4 LSB | — |
| max | 1.56e-2 LSB | — |

**The lattice is 958x finer than rounding — about 9.9 extra bits of amplitude authority per scorer
value, at zero counted bytes.** cg2 estimated 2.93 bits from the smallest single-uint8-tick step
(0.1315); that under-counts, because the reachable set is the Minkowski sum of four progressions,
not one tick. This also explains *why* the resample term is recoverable at all: `D` is surjective,
so a real-valued camera image can hit any `r` exactly; only the uint8 granularity is irreducible,
and it is ~3e-4 LSB.

**The S arithmetic** (`W = 1.27310821533 B/flip` reproduced; v4d rate term 0.2398677; own-vehicle
S 0.9639878; demonstrated floor PR130 0.172141; gap **0.7918468**; seg axis 0.4015):

| | d_seg | ΔS | % of gap | % of seg axis | B-equivalent |
|---|---|---|---|---|---|
| ideal-leg misplacement | 3.244e-4 | 0.03244 | 4.10 % | 8.08 % | 48,724 |
| **live-leg misplacement** | **4.185e-4** | **0.04185** | **5.29 %** | **10.42 %** | **62,850** |
| post-dither floor (rho = 2.98e-4) | 2.23e-7 | 0.00002 | 0.003 % | 0.006 % | 34 |
| **recoverable** | **4.183e-4** | **0.04183** | **5.28 %** | **10.42 %** | **62,816** |

62,816 B-equivalent against an archive of 360,238 B, for a decode-side rule that ships **zero**
counted bytes. It is not a rate lever and it is not a capacity lever; it is a realization lever,
and it is the cheapest thing on the seg axis by construction.

---

## §7 — what I did NOT measure, and the one thing that would settle it

**The reach is DERIVED, not measured.** `G = 0.0606` comes from a **single** anchor (median
Road-Lane boundary pixel, margin 0.516, flip-L2 8.8), is first-order, is `[macOS-CPU advisory]`
`research_only`, and has never been checked against a measured d_seg. Every ΔS in §6 inherits that.
Treat them as order-of-magnitude, not as a row.

**The live leg's reach is the weaker of the two.** It applies the GT margin field to the live
misplacement, but the live render is 44–46 LSB rms from GT — the operating point is far from where
the margin field was computed. The **ideal leg (0.0324 S) is the internally consistent one**; the
live leg's magnitude is indicative and its *comparison* to the ideal leg is what I trust.

**Not measured:** decode wall-clock of any dither (the exact 256^4 solve is ~65k ops/block x 118M
blocks — far too slow; a table-driven +/-1 correction is the obvious cheap form, and is unbuilt and
untimed); the actual d_seg delta; the d_pose coupling.

**One mechanism check that changes the risk profile, READ not assumed:** PoseNet's
`preprocess_input` (`upstream/modules.py:73`) applies **the same** `F.interpolate(..., (384,512),
bilinear)` to both frames before `rgb_to_yuv6`. So a dither that makes `D(cam)` hit `r` exactly
improves *both* heads' realization simultaneously — the seg/pose antagonism that gc17 §2b warned
about for a naive camera-res chroma dither does not apply to a dither designed against `D` itself.
**The residual pose risk is elsewhere:** v4d synthesizes frame_0 by warping frame_1 at camera
resolution, so changing frame_1's camera pixels changes what gets warped. That coupling is
UNMEASURED and is the pre-registered falsifier.

**THE ONE NEXT MEASUREMENT (gated).** Replace the receiver's `clip(rint(up))` with the generic
per-block dither, hold the archive **byte-identical** (the rule is generic ⇒ rule-118 free, zero
rate risk by construction), and re-measure d_seg AND d_pose at n600 through the frozen CPU-torch
scorers on v4d. Pre-registered falsifier: **if measured Δd_seg < 1.0e-4 (≈ 24 % of the modelled
4.18e-4), the gain law is over-predicting and this family is closed at FORMULATION level; if
d_pose worsens by more than the seg gain in S units, the frame_0-warp coupling kills it.**
**GATE: needs the full-n600 scorer slot.** Queued, not taken — this arm held none.

---

## §8 — re-anchor discipline, and the prior law's prediction

Per the standing rule, this is a **RE-ANCHOR of #149** (camera-res sub-pixel placement,
closed-form, COMPLETED, $0) and of cg2 §6a, not a discovery. **The prior law's prediction, stated
so it can be scored:** the registered evasion probe predicts d_seg ≈ 0.001614·rho under a
margin-BLIND model, i.e. it would have predicted **9.4e-4 at the live rho of 0.799** with **zero
enrichment**. Measured here: enrichment **1.54x vs shuffle**, and a one-sided reach of 4.19e-4.
So the prior law got the *scale* right within ~2x and the *margin-blindness* wrong. What is new in
this arm and was not in #149, cg2, or the probe: the per-pixel field itself, the ideal-vs-live
transfer (which reverses the expected direction), the resample-vs-quant split, the full-grid block
measurement, the exact lattice solve, and the enrichment curve.

## §9 — triality

* **DAG:** FEED-mp1 (this file + `.omx/research/ddm_mp1_realization_join_n600_20260802.json`).
  Pays task #898; answers #897 with OPEN, not CLOSED.
* **DSL:** N/A — a receiver realization rule, no trainer lever, no curriculum, no launch.
* **equations:** consumes `segnet_head_rank4_linear_flipdist_v1` (scope note: its pixel-space
  pullback is first-order and single-anchor; this arm is the first consumer to state that the
  resulting d_seg is therefore order-of-magnitude only). Independently reproduces the registered
  break-even `W = 1.27310821533 B/flip` and cg2's blind-set constants (106 / 140 / 230,904).
