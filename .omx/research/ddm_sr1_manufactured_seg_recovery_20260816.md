---
arm: ddm_sr1
title: "the archive format inserts an EXACTLY TRIDIAGONAL blur [0.101470, 0.797060, 0.101470] between the renderer and the scorer -- square, DC-preserving (which is why rt1 measured R=0 on flat paint), invertible at kappa 2.34 by a Thomas solve, and worth ZERO archive bytes to undo; and rt1's correction channel FLIPS SIGN when priced per cell instead of on average: +0.00269 S becomes -0.000595 S (6.2% of the gap) at the same measured eta, buying flips at 1.053 B against the 1.273 B bar, on a support the waterfill selects unaided as 56.4% Road<->Lane"
utc: 2026-08-16
parent: ".omx/research/ddm_rt1_seg_roundtrip_decomposition_20260816.md"
axis: "[macOS-CPU advisory] scorer-free -- NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "INSTANCE on the hv1 ep0634 vehicle; family verdicts only where named"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_sr1 — the manufactured seg error, priced into two actuators

STORES CONSULTED: parent `ddm_rt1_seg_roundtrip_decomposition_20260816.md` (all sections plus
its retained receipts `RT1_INSTRUMENT_CHECK/GEOMETRY/EDGESHAPE/LEDGER.json`) ·
`ddm_td1_token_drop_schur_arithmetic_20260816.md` · `ddm_av3_fresh_eyes_review_20260816.md` §F4 ·
`ddm_hg1_ring0_margin_hinge_20260816.md` (§ANSWER FIRST items 3–6) ·
`ddm_rc4_rung4_token_drop_verdict_20260816.md` (via its ledger row: the seg market bar and the
frame_1-only mechanism) · `ddm_wc1_decode_wallclock_verdict_20260816.md` (the retained decode) ·
the LIVE decoder `experiments/results/public_pr130_intake_20260725_fable/source/submissions/
semantic-pose-HPAC_CPR1/inflate.py` + its `LINEAGE_AND_CITATIONS.md` · `upstream/modules.py` ·
`experiments/ddm_sq1_eta_seg_realization.py` · memories [[m88]] [[m96]] [[m91]] and
`the_counted_byte_is_not_fungible_placement_beats_amount_20260816`.

## ANSWER FIRST

rt1 measured that 96.6% of the hv1 seg axis is MANUFACTURED between the labels we ship and the
argmax the scorer reads back, and closed every post-hoc lever it priced. It left exactly one
stage open, in its own words: *"R is not a supplier for any paint-shaped candidate, and it
remains unmeasured for the render."* I measured that stage, and I re-priced the one channel rt1
closed on an average.

**1. The operator is now exact, and it is not nothing.** The live decoder renders the seg
surface at **384×512 — byte for byte the size SegNet consumes** — then upsamples it to
874×1164, rounds to uint8, and hands it to a scorer that immediately bilinear-downsamples it
back to 384×512. The composite `A = D∘U` is a **square** map on the scorer's own lattice, so it
cannot add information. Measured exactly: **A is TRIDIAGONAL** (max |A| outside band 1 is
`0.000e+00`, both axes), its middle row is **[0.101470, 0.797060, 0.101470]**, and **every row
sums to 1.000000000**.

**2. That row sum is why rt1 measured S3 = 0, and why the zero it measured does not transfer.**
A constant field is an exact eigenvector of A with eigenvalue 1. Flat paint is piecewise
constant, so A is transparent to it — rt1's `S3 = 0 flips` is now explained mechanically rather
than observed. The render is not piecewise constant, and on the boundary A attenuates: the 2-D
singular values run **[0.4491, 1.0531]**, **83.0% of them below 0.9**, condition number
**2.3449**.

**3. Undoing it costs ZERO archive bytes and is a tridiagonal solve.** `inflate.py` is free and
unsized; a fixed resample matrix is a generic algorithm with no video-derived content, so
rule 118 is clean, and PR95's L28 decode-side channel postprocess is the in-tree precedent for a
0-byte decoder-side correction. Because A is tridiagonal, `A⁻¹` is a Thomas solve, not a dense
inverse. Inverse amplification is **≤ 1.4909 per axis, ≤ 2.223 in 2-D**. Clipping outside
[0,255] touches **0.0431%** of pixels.

**4. The perturbation lands exactly where the axis lives.** Undoing A moves the scorer's input
by **22.48 RGB levels on average at the label boundary** and **1.82 in the interior — a 12.34×
concentration** on the one-pixel curve that carries 99.22% of the seg axis. 97.4% of band pixels
move more than 4 levels.

**5. The CEILING, and it is the reason to spend the row:** a zero-byte actuator has no rate side,
so any recovery is pure profit. **It must recover 33.55% of the 33,743 round-trip flips to close
the whole −0.0095973 gap by itself.** Recovering 10% is 29.8% of the gap; 5% is 14.9%.

**6. The sign is NOT settled scorer-free, and I will not pretend it is.** Two scorer-free proxies
both failed, each for a measured reason, and I report both because a NO from either would have
been an artifact sold as physics. Against the GT the sharpening looks strictly bad (α\* = 0 on
16/16) — but the band RMSE is **102–113 levels**, because the semantic renderer is photometrically
nowhere near the video while scoring 0.0296% argmax error; a 22-level term is invisible under a
105-level residual. In the class-decision coordinate the sharpening is unanimous in magnitude
(**decisiveness +22.8%, 16/16 pairs**) but essentially null in direction (**wrong-side share
−0.219% relative, 11/16 pairs**) — which reproduces rt1's own §2.6 finding that **94.3% of the
error is symmetric jitter** from a second instrument. Symmetric error, amplified symmetrically,
does not obviously improve. **Honest prior: near-neutral, either sign, huge upside, zero cost.**

**7. rt1's correction channel FLIPS SIGN when it is priced per cell instead of on average.** The
break-even density is a fixed number at a given η — `η·p·1.2731 > H(p)/8` — so the whole-band
mean density of 1.359% is the wrong statistic. At the measured **η = 0.6235** the break-even
density is **3.821%**, and the density distribution is wide: **46.1% of flips sit in cells at or
above 2%, 35.9% above 3%, 15.2% above 5%**. Waterfilling the free band:

| framing at η = 0.6235 (measured) | flips | bytes | B / recovered flip | net ΔS | share of gap |
|---|---:|---:|---:|---:|---:|
| rt1: describe the whole band (its real coder) | 34,666 | 33,235 | 1.5376 | **+0.00381** | — (a loss) |
| describe the whole band (ideal limit) | 34,666 | 31,554 | 1.4599 | **+0.00269** | — (a loss) |
| **waterfill, unguarded** | **6,512** | **4,276** | **1.0532** | **−0.000595** | **6.20%** |
| waterfill, cells ≥ 500 band px | 6,007 | 4,035 | 1.0773 | −0.000488 | 5.09% |

The market bar is **1.2731 B/flip**. rt1's channel bought at **1.5376 — 20.8% above the bar**.
The waterfilled channel buys at **1.053 — 17.3% below it**. Same object, same η, same retained
masks; only the placement changed. This is
`the_counted_byte_is_not_fungible_placement_beats_amount` measured on rt1's own data.

**8. The waterfill selects rt1's named lever unaided.** Nothing in my cell key knows about
Road↔Lane. The selected support is nonetheless **56.4% Road↔Lane (3,673 flips at 6.638%
density)**, then Movable↔Road (1,572) and Road↔Undrivable (869 at 15.998%). rt1 named the
Road↔Lane edge from a flip census; the rate-distortion waterfill picks the same edge out of 74
cells on price alone. Two independent routes, one target.

**Net: the seg axis has two live actuators, not zero.** A zero-byte decoder-side de-blur whose
ceiling is 2.98× the gap and whose sign needs one local SegNet pass, and a byte-carrying
correction channel that is a supplier of 5–6% of the gap at the measured η — where rt1 recorded a
loss — and 84–87% at η = 1. Both are wired to the LIVE decode path. **Pointer UNMOVED; no score
claim; no dispatch spent.**

## §0 Prior-law prediction lines (stated BEFORE measuring, per the anti-re-anchor law)

1. **rt1 §2.5b** — "R supplies exactly zero" on flat paint, unmeasured on the render.
   PREDICTION: A will be a genuine low-pass, so the zero will be a property of *flat content*,
   not of the operator. **HELD, and sharpened**: A is a 3-tap blur whose rows sum to exactly 1,
   so flat content is an exact fixed point. The zero was never about R.
2. **rt1 §2.6** — 94.3% of the seg error is symmetric sub-pixel jitter, only 5.7% systematic
   bias. PREDICTION: a symmetric linear de-blur will restore *magnitude* but not *direction*.
   **HELD** — decisiveness +22.8% (16/16) against wrong-side −0.219% (11/16). I wrote this line
   before running §3.3 and it is the reason I did not report the tool's `LIVE_SIGN_CONFIRMED`
   string as a finding.
3. **rt1 §5.4 + §6.4** — the channel needs η > 0.753 and measures 0.6235, so it is a
   non-supplier. PREDICTION: that arithmetic is done at the band MEAN, and the band is
   heterogeneous, so a sub-support should clear the bar. **HELD, and it flips the sign.**
4. **m91 (pc2 hub law)** — seg is one graph with one hub, Road in 87.8% of flips.
   PREDICTION: the waterfilled support will be Road-dominated. **HELD, with the exception
   recorded**: 7 of the 9 selected edges touch Road or Lane, carrying **6,403 of 6,512 flips
   (98.33%)**. Two do not — Movable↔Undrivable (80 flips) and Movable↔MyCar (29) — and I am
   recording them rather than rounding the claim to "every edge", which is what I first wrote.
5. **hg1 item 6** — hv1's token trainer has zero SegNet, so the whole axis is renderer
   realization error. PREDICTION: the actuator must live in the decoder or the renderer, never
   in the token path. **HELD** — both actuators below are decoder-side.

## §1 Instrument — one positive control passed, two of my own instruments failed

Nothing below is claimed without its control. Receipt `SR1_ROPERATOR.json`.

| control | measured | verdict |
|---|---:|---|
| separable A vs the real `F.interpolate` up/down chain | max abs `8.53e-14`, rel `3.37e-16` | **PASS** |
| waterfill band support vs rt1's | 2,551,464 px vs 2,551,464 | **EXACT** |
| waterfill band flips vs rt1's | 34,666 vs 34,666 | **EXACT** |
| target-class cost vs rt1's §5.3 | 0.22531 vs 0.2226 bits/flip | 1.2% |

The band and flip counts are reproduced from independent code on the same retained fields, so
the waterfill is re-pricing rt1's object, not a different one.

**Failure 1 — my GT-referenced sign test is an invalid instrument, and I am retracting its
verdict.** I asked whether `x + α(A⁻¹x − x)` moves the scorer's input toward `D(GT_cam)`, swept
α ∈ [0, 1.5], and got **α\* = 0.00 on 16 of 16 pairs**. That reads as a clean kill. It is not
one: the band RMSE at α = 0 is **102.3–113.4 RGB levels**. The semantic renderer paints from five
class tokens plus coordinates; it is not trying to look like the video and it does not, yet its
argmax error is 0.0296%. A 22-level sharpening term cannot register under a 105-level residual
that is uncorrelated with it. The measurement is real; its interpretation as physics is void.
This is rt1 §6.1's defect class reached from the other side — there the target carried 0.025% of
the loss, here the signal carries ~4% of the residual. Receipt `SR1_SIGN.json` is retained with
this scope written into it.

**Failure 2 — my class-decision proxy over-predicts errors ~29× and cannot settle direction
either.** Projecting each band pixel onto the line between its two class anchors calls **39.71%**
of band pixels "wrong side", while the scorer actually flips **1.36%** of them. SegNet reads
REGIONS, not pixels (CLAUDE.md), so a per-pixel colour test is the wrong function. What survives
from it is the part that does not depend on the proxy being calibrated: the **direction-free**
statement that A⁻¹ increases decisiveness by 22.8% on 16/16 pairs, and the **null** result on
direction. Receipt `SR1_EMPHASIS.json`.

**A warning I chased rather than ignored.** numpy emits `divide by zero / overflow / invalid
encountered in matmul` on this host's BLAS for these products. I verified every input and output
finite (`x finite: True`, range [0, 254.643]) and the operator finite before using any number.
The warnings are a floating-point-status artifact of the BLAS call, not a value defect.

## §2 The per-stage attribution, re-derived with provenance per row

rt1's rows verified against its own receipts, then extended by the stage it left open. All rows
n600, `[macOS-CPU advisory]`, `seg_ΔS/flip = 100/117,964,800 = 8.477105e-07`.

| stage | flips | S | vs the −0.0095973 gap | provenance |
|---|---:|---:|---:|---|
| scored seg term | 34,938 | 0.029617 | 3.086× | `RT1_INSTRUMENT_CHECK.json` `advisory.flips_vs_gt`; 1.000213 of the contest-CUDA seg term |
| label channel (shipped labels vs GT) | 1,717 | 0.001456 | 0.152× | `RT1_INSTRUMENT_CHECK.json` `td1_control`, EXACT vs td1 |
| **MANUFACTURED round trip** | **33,743** | **0.028604** | **2.980×** | `RT1_INSTRUMENT_CHECK.json` `advisory.flips_vs_label` |
| …of which on ring 0 of the label boundary | 33,479 | 0.028381 | 2.957× | `RT1_GEOMETRY.json` `flips_vs_label_by_ring[0]` |

**Where inside the round trip the error is born.** rt1 established that the v14 stage taxonomy
does not nest here — flat paint reads back 35.4× *worse* than the trained render, so "render
deviation" is negative and the stages are alternatives, not layers. That verdict stands and I did
not re-derive it. What I add is the third stage, which was the only one still unmeasured:

| stage | supplies | how measured |
|---|---|---|
| S2 paint → SegNet | a CEILING 35.4× above the render, not a floor | rt1 §2.5, unchanged |
| **S3 = A = D∘U, on FLAT content** | **0 flips** | rt1 §2.5b — and now **explained**: rows of A sum to 1.000000000, so constants are fixed points |
| **S3 = A = D∘U, on the RENDER** | **a 3-tap blur, κ 2.3449, 83.0% of the 2-D spectrum below unit gain; 22.48 levels at the band vs 1.82 interior** | this unit, `SR1_ROPERATOR.json` |
| S4 GT flicker | 27.7% coincidence, a bound on smoothing cures | rt1 §2.2, unchanged |

The operator, exactly (`A_row_384x384.npy` sha `d884e8ec…`, `A_col_512x512.npy` sha `1a0fd4c4…`):

| property | rows (384) | columns (512) |
|---|---:|---:|
| max abs outside tridiagonal band | **0.000e+00** | **0.000e+00** |
| middle-row taps | 0.101470 / 0.797060 / 0.101470 | 0.101744 / 0.796513 / 0.035968 |
| row sum | 1.000000000 | 1.000000000 |
| diagonal range | 0.780326 – 0.997718 | 0.780082 – 0.996577 |
| ‖A − I‖_F / ‖I‖_F | 0.195941 | 0.196301 |
| condition number | 1.530498 | 1.532113 |
| distinct row patterns | 96 | 64 |

The kernel is position-dependent (874/384 = 2.2760 is not an integer, so the resample phase
cycles) with only 96 and 64 distinct patterns — a tiny deterministic rule, no stored table, no
video-derived content.

## §3 The actuators — both wired to the LIVE decode path

The #917 warning is the right one to fear here: most lever instruments in this repo aim at a
retired vehicle. So I traced the shipped chain to source before naming anything.

**The live seg surface, at source.** `inflate.py::render_video` writes `output[2i+1]` — frame_1 —
as `semantic(tokens, idx)` at `EVAL_H, EVAL_W = 384, 512`, then
`F.interpolate(..., size=(874,1164), mode="bilinear", align_corners=False).clamp(0,255).round()`.
`output[2i]` — frame_0 — is the 12-dimensional pose carrier only. `SegNet.preprocess_input`
(`upstream/modules.py:107-109`) takes `x[:, -1]` and interpolates bilinearly to
`(384, 512)`. So **frame_1 is 100% of the seg surface** and the round trip is square. This agrees
row-for-row with rc4's independent reading of the same file, and with hg1's finding that hv1's
token trainer contains no SegNet at all.

### A1 — zero-byte de-blur (`m ← A⁻¹m` before the up-sample)

- **Where it lands:** one line in `render_video`, before `F.interpolate`. A tridiagonal solve
  along each axis, applied to `master_eval`.
- **Wired to the live path:** YES — it modifies the exact tensor the shipped decoder writes to
  frame_1.
- **Counted bytes: ZERO.** The matrices are generated deterministically from `F.interpolate` on
  an identity, at decode time. No learned content, no video-derived content, rule 118 clean.
  Precedent in tree: PR95-family L28, a decode-side channel postprocess at 0 archive bytes.
- **Decode cost:** a Thomas solve is O(n) per line; 600 frames × 3 channels × (512+384) short
  tridiagonal solves. Negligible against the 30-minute budget.
- **Feasibility, measured:** invertible at κ 2.3449, amplification ≤ 2.223, clipping 0.0431%.
- **Free parameter:** α in `m ← m + α(A⁻¹m − m)`. α = 0 is the shipped decoder, so the actuator
  strictly contains the current behaviour and cannot be worse than the best α found.

### A2 — waterfilled free-band correction channel

- **Where it lands:** a new archive section naming which band pixels flip, plus the target class,
  restricted to the free cells the waterfill selects. The receiver recomputes the support from
  the transmitted labels it has already decoded, so the *selection* costs nothing.
- **Wired to the live path:** PARTLY. The coder exists and is built (rt1 §5, `M7.bin`, verified
  by decoding it back). The *realization* — turning a described flip into a scored flip — is
  sq1's solved paint, which rt1 measured on hv1 at η = 0.6235. Neither is landed in the shipped
  `inflate.py`; both are measured harnesses. **I am stating that plainly rather than calling it
  wired.**
- **Counted bytes:** 4,035–4,276 B at the measured η (vs rt1's 33,235 B for the whole band).

### Not actuators, and why

- **Better class prototypes** — dominated 35.4× (rt1 §2.5). CLOSED, unchanged.
- **Flat band repaint** — +1.3808 S (rt1 §2.7). CLOSED, unchanged.
- **Any cure aimed at R for paint-shaped content** — A is a fixed point on constants. Exactly
  zero, now with the mechanism.
- **The label channel** — 1,717 flips = 0.001456 S, 15.2% of the gap even if driven to zero, and
  hg1 measured the token field already 34.9× better than the axis it feeds. Not where the money
  is.

## §4 Ceiling arithmetic, in S units against the −0.0095973 gap

Exchange rates, exact contest arithmetic: `seg ΔS/flip = 8.477105e-07`,
`rate ΔS/byte = 25/37,545,489 = 6.658590e-07`, so the **seg market bar is 1.273108 B/flip**.

### A1 ceiling — zero bytes, so recovery share is the whole story

| recovered share of the 33,743 round-trip flips | ΔS | share of gap |
|---:|---:|---:|
| 5% | −0.001430 | 14.9% |
| 10% | −0.002860 | 29.8% |
| 20% | −0.005721 | 59.6% |
| **33.55%** | **−0.009597** | **100% — closes the gap alone** |
| 100% (unreachable; A is not the only error source) | −0.028604 | 298% |

A1 cannot go negative-value on rate because it has no rate. It can go negative on *seg* — a
sharpening that amplifies wrong decisions costs flips. That is precisely the risk §1 Failure 2
could not resolve, and precisely why the α sweep matters: α = 0 recovers the shipped behaviour
exactly, so the actuator's floor is the status quo.

### A2 ceiling — the waterfill curve

`SR1_WATERFILL.json`, ideal conditional-entropy limit (a CEILING: no coder inefficiency, model
cost 148 B for 74 cells). Guarded rows drop cells below 500 band pixels, which kills the
small-sample optimism.

| η | framing | flips | bytes | B/flip | net ΔS | share of gap |
|---:|---|---:|---:|---:|---:|---:|
| 0.6235 (MEASURED) | describe everything | 34,666 | 31,554 | 1.4599 | +0.002688 | loss |
| 0.6235 | **waterfill** | 6,512 | 4,276 | 1.0532 | **−0.000595** | **6.20%** |
| 0.6235 | waterfill, guarded | 6,007 | 4,035 | 1.0773 | −0.000488 | 5.09% |
| 0.7531 (rt1's bar) | waterfill | 18,381 | 14,390 | 1.0395 | −0.002153 | 22.44% |
| 0.7531 | waterfill, guarded | 17,871 | 14,144 | 1.0509 | −0.001991 | 20.75% |
| 0.85 | waterfill | 29,823 | 25,984 | 1.0250 | −0.004188 | 43.63% |
| 1.00 | waterfill | 33,893 | 30,561 | 0.9017 | −0.008382 | 87.34% |
| 1.00 | waterfill, guarded | 33,382 | 30,314 | 0.9081 | −0.008114 | 84.54% |

Break-even density by η: **0.6235 → 3.821% · 0.7531 → 1.548% · 0.85 → 0.784% · 1.00 → 0.273%.**
Flip mass above each density: **≥2% 46.10% · ≥3% 35.91% · ≥5% 15.16% · ≥10% 5.40% · ≥20% 0.70%.**
Overfitting exposure of the η = 0.6235 selection: **1.27% of selected flips come from cells under
100 band pixels, 7.75% from cells under 500.** The sign survives every guard down to two cells.

**A2 is still not a gap-closer alone**, and rt1's structural point stands: even at η = 1 the
channel reaches 84–87%, not 100%. What changed is that at the *measured* η it is a **supplier
instead of a cost**, and it is small and cheap enough (4 KB) to compose with anything.

**How far can η fall before A2 dies?** The specific 6,512-flip support above goes negative below
**η = 0.5158**. But the waterfill is self-correcting: at a lower η it retreats to a denser
support. With the ≥500-px guard the channel stays a supplier for any **η > 0.3871** — a 38%
margin below the measured 0.6235. That margin, not the headline 6.20%, is what makes A2 worth a
measurement rather than a build.

## §5 Ranking, and the single cheapest decisive measurement

Ranked by (S recoverable) ÷ (counted bytes + build risk):

| # | actuator | S ceiling | counted bytes | build risk | rank rationale |
|---|---|---:|---:|---|---|
| **1** | **A1 zero-byte de-blur** | **−0.028604 (2.98× the gap)** | **0** | LOW — one line in `render_video`, α = 0 recovers today's behaviour | infinite S per byte; the only unmeasured term is the sign, and one local pass settles it |
| 2 | A2 waterfilled correction channel | −0.000595 measured η, −0.008382 at η = 1 | 4,276 | MEDIUM — coder built and verified; the solved-paint realization is a harness, not shipped | positive at the measured η, but bounded and needs the per-cell η it has never had |
| 3 | edge-weighted Road↔Lane render objective | unbounded by this unit | 0 | HIGH — a training run | rt1's follow-on #3; **already live** as `hg1` ring-0 hinge A/B (pid 4832), so not mine to fire |

**The cheapest decisive measurement is A1's sign, and it does not need the renderer.**

The trick: the actuator writes `U(A⁻¹m)` where `m` is the master. We do not have `m`, but the
scorer's own view of the shipped decode is `x = D(cam) = A·m + q`, so `A⁻¹x ≈ m` up to one uint8
term. Therefore the post-fix camera frame can be synthesized directly from the retained decode:

```
cam' = round(clamp(U(A⁻¹ · D(cam)), 0, 255))
```

and pushed through the frozen CPU SegNet exactly as rt1's base leg was. **No renderer forward, no
archive rebuild, no decode rebuild, no Modal.** Cost is one SegNet pass per α — rt1's base leg
took 782 s for n600 at batch 1 with 8 threads.

## §6 FIRE-ORDER (sealed for MAIN — I did not run it; the scorer slot is held by pid 4832)

**FO-1 — A1 sign, the decisive $0 row.** Highest value in this unit.

- **Object:** for each pair, `x = D(cam_f1)` using the retained `A_row/A_col`; `xs = x + α(A⁻¹x − x)`;
  `cam' = round(clamp(U(xs)))`; frozen CPU SegNet argmax; count flips vs
  `gt_argmax_n600.npy` (sha `91d3ff11…`).
- **Inputs, all retained:** wc1 decode `0.raw` (3,662,409,600 B, sha `e5539653…`),
  `A_row_384x384.npy` (sha `d884e8ec…`), `A_col_512x512.npy` (sha `1a0fd4c4…`),
  rt1 `argmax_base.npy` (sha `2aeb1e6b…`) as the α = 0 control.
- **Ladder:** α ∈ {0, 0.25, 0.5, 0.75, 1.0}. α = 0 MUST reproduce 34,938 flips exactly — that is
  the positive control, and a miss invalidates the row.
- **Pre-registered bands, written before any α > 0 is scored:**
  - flips at best α **< 33,251** (≥ 5% of the round trip recovered) → **LIVE**, and the same tool
    then sweeps α finer and MAIN prices a 0-byte candidate archive;
  - flips **within ±1% of 34,938** at every α → **CLOSED as neutral**, verdict_scope FORMULATION
    (global linear de-blur of A on this vehicle), and the honest reason is rt1 §2.6's 94.3%
    symmetric jitter reproduced by my §1 Failure 2;
  - flips **> 35,287** at every α > 0 → **CLOSED as harmful**, same scope.
- **Instrument pins (et4 — batch shape is part of the forward instrument):** frozen CPU torch
  SegNet from `upstream/models/segnet.safetensors`, **batch = 1 pair**, `torch.set_num_threads(8)`,
  `SegNet.preprocess_input` verbatim — identical to rt1's, so leg-to-leg differences carry no
  instrument term.
- **Cost:** ~800 s per α on CPU, $0, no dispatch. Run it when the Metal/scorer slot frees.
- **Scope caveat to carry into the verdict:** `A⁻¹·D(cam)` inherits one round of uint8 noise the
  real decoder-side fix would not, so this realization is faithful but slightly PESSIMISTIC. A
  LIVE reading is therefore conservative; a CLOSED reading at the ±1% band should be re-checked
  against the true renderer output before the family is called dead.

**FO-2 — A2's per-cell η, the number the whole waterfill rests on.** Fire only after FO-1.

- **Object:** rt1's `experiments/ddm_rt1_eta_gate_pose_constrained.py` restricted to the
  waterfilled described set instead of the whole ring-0 band. The tool's live flags are
  `--mode {null,free,aggregate} --n-pairs --seed --steps --lr --radius --focus-weight
  --eval-every --full-population-n --retain-frames`; **there is no support-restriction flag
  today**, so this needs one added — I am naming the change, not inventing a flag.
- **Why it is load-bearing:** my whole §4 A2 arithmetic assumes η is constant across cells, and
  it is **UNMEASURED per cell**. rt1 §6.3 measured that collateral is what caps η, and the
  selected cells are exactly the ones where flips are densest — so η there could plausibly be
  LOWER.
- **Bands, computed on the retained cell arrays:**
  - the **specific 6,512-flip support** priced in §4 goes negative below **η = 0.5158**;
  - but the waterfill is **self-correcting** — at a lower η it simply retreats to a smaller,
    denser support. With the ≥500-px guard the waterfilled channel stays a supplier for any
    **η > 0.3871**. (Unguarded the figure is 0.0221, which is not a real bound: it is the
    small-cell tail of §4 doing the work, and I am not claiming it.)
  - So the verdict rule is: **η_selected > 0.3871 → A2 stays a supplier at some support size;
    ≤ 0.3871 → A2 CLOSED, and rt1's original verdict is restored on better evidence than it had.**
    Between 0.3871 and 0.5158 the support must be re-waterfilled at the measured η before any
    byte is spent.

## §7 What this unit did NOT establish

- **No argmax effect for A1.** The operator, its invertibility, its cost and its magnitude are
  measured; the flip count is not. Two scorer-free proxies failed, for the measured reasons in §1.
- **No per-cell η for A2.** The waterfill assumes the pooled 0.6235 applies uniformly. It is the
  single assumption that can invert §4's A2 rows, and FO-2 is the measurement.
- **No real coder for the waterfilled support.** §4's A2 bytes are the ideal conditional-entropy
  limit. rt1's real M7 beat its own i.i.d. floor by 2.5%, so the limit is not wildly optimistic,
  but a realized coder on 74 cells will pay some model cost above the 148 B I credited.
- **No claim that A is the only manufactured term.** The render→argmax map contains SegNet's own
  stride-2 stem and everything after it. A is one stage, exactly characterized.
- **No score, no pointer move, no dispatch.** Every number is `[macOS-CPU advisory]` or scorer-free
  arithmetic.

## §8 Retained payloads (ALWAYS KEEP THE PAYLOAD)

Root `/Volumes/APDataStore/pact/ddm_sr1_manufactured_seg_recovery_20260816/` (APDataStore:
VertigoDataTier is effectively full; APDataStore has 230 GiB free).

| artifact | bytes | sha256 (prefix) | what it is |
|---|---:|---|---|
| `A_row_384x384.npy` | 1,179,776 | `d884e8ecb9ab…` | the exact row operator D∘U |
| `A_col_512x512.npy` | 2,097,280 | `1a0fd4c49d25…` | the exact column operator D∘U |
| `band_blur_levels.npy` | 408,372 | `f0e376259f54…` | per-band-pixel realized blur, 24 pairs |
| `cell_band_px.npy` | 9,728 | — | the waterfill's per-cell band population |
| `cell_flip_px.npy` | 9,728 | — | the waterfill's per-cell flip count |

Receipts: `SR1_ROPERATOR.json` · `SR1_SIGN.json` (retracted verdict, retained with scope) ·
`SR1_EMPHASIS.json` · `SR1_WATERFILL.json` · `SR1_LEDGER.json`.
Tool: `experiments/ddm_sr1_manufactured_seg_recovery.py`
(stages `roperator` / `sign` / `emphasis` / `waterfill` / `ledger`).
Consumed unmodified: the wc1 retained decode `0.raw` (sha `e5539653…`), the hv1 ep0634
`decoded_spatial_tokens.rc64.bin`, the qs3 `gt_argmax_n600.npy` (sha `91d3ff11…`), and rt1's
`argmax_base.npy` (sha `2aeb1e6b…`).

**Own-vehicle frontier: hv1 ep0634, S 0.15959729295498598 @ 182,759 B `[contest-CUDA T4 n600]` —
UNMOVED by this unit.**
