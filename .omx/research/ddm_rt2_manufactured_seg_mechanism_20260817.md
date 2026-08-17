---
arm: ddm_rt2
title: "rt1 reproduces to the flip (34,938 / 33,743); the mechanism is that the decoder renders at 384x512 -- SegNet's exact input size -- then bilinear-lifts to 874x1164 and the scorer lifts it straight back down, and removing that gratuitous blur is measured DEAD on both legs (seg +98 flips, d_pose x38.6 at full strength); the durable result is that PoseNet's null space is EXACTLY 50% of the scorer-resolution RGB field, its chroma-clamp leak is 1e-5, and its only real loss channel is the uint8 camera grid -- worth 44x pose attenuation at matched perturbation"
utc: 2026-08-17
charter: "operator/MAIN charter to ddm_rt2, 2026-08-17"
axis: "[macOS-CPU advisory] frozen CPU-torch SegNet + PoseNet -- NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "INSTANCE on the hv1 ep0634 vehicle; family/mechanism verdicts only where named"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_rt2 — by what MECHANISM the round trip manufactures the seg axis

STORES CONSULTED: rt1 memo + charter (`ddm_rt1_seg_roundtrip_decomposition_20260816.md`,
`..._charter_20260816.md`) · sr1 `ddm_sr1_manufactured_seg_recovery_20260816.md` (its actuator
A1 and its sealed unfired FO-1) · rn1 `ddm_rn1_render_boundary_mechanism_20260816.md` §3/§4 ·
hg1 `ddm_hg1_ring0_margin_hinge_20260816.md` · td1 `ddm_td1_token_drop_schur_arithmetic_20260816.md` ·
rp1 `ddm_rp1_rangeA_cell_realized_probe_20260728.md` · mp1 `ddm_mp1_lsb_misplacement_margin_join_20260802.md` ·
ra1 `ddm_ra1_rasterization_crossing_20260802.md` · vs1 `ddm_vs1_20260805/SCORER_INVISIBLE_NAMING.md` ·
`#149` `dseg_side_feasibility_corners_verdict_20260619.md` · lr2 · `r_survival_physics_20260629T182659Z.md` ·
`upstream/modules.py:72-73,108-110` and `upstream/frame_utils.py:51-76` read at source ·
memories [[m88]] [[m96]] [[m91]].

## ANSWER FIRST

1. **rt1 reproduces EXACTLY.** An independently written tool, n600, full field, seeded-random
   never a prefix, reproduces **34,938 scored flips** and **33,743 round-trip flips** — delta
   **0** on both — at **1.000213×** the contest-CUDA `d_seg`. The round trip is **0.028604 S =
   96.58%** of the seg axis and **2.98×** the whole remaining −0.0095973 gap. rt1's headline
   stands. (The charter's "0.028155 S / 95%" is td1's earlier MODELLED figure; rt1's measured
   0.028604 / 96.58% supersedes it and I confirm rt1, not td1.)
2. **The mechanism is a gratuitous blur, and I identified the kernel.** The decoder renders
   frame_1 at **384×512 — byte for byte the size SegNet consumes** — bilinear-lifts it to
   874×1164, rounds to uint8, and the scorer bilinear-downsamples it straight back to 384×512.
   MEASURED: the shipped camera frames lie in `range(U_bilinear)` to **rms 0.233 / max 0.825**,
   i.e. pure uint8 residue; and `A = D∘U` is tridiagonal to **exactly 0.000e+00** off-band for
   **bilinear** while **bicubic** leaves **1.54e-2**. So the lift is bilinear, not bicubic — the
   charter's premise and every "bicubic-up" description of this chain are wrong for this vehicle.
3. **Removing that blur is measured DEAD, on both legs.** `D` is surjective, so I built the
   exact closed-form preimage `X' = X + R† (T − D X) C†ᵀ` and verified it: `α = 0` reproduces the
   base to the flip (positive control PASS), and a named target is realized to **rms 0.148**,
   entirely the uint8 grid. Handing SegNet the render's own native field then costs
   **+98 flips (+0.002077 S)** and **d_pose ×38.58 (+0.2475 S)** at full strength. **sr1's
   actuator A1 — its #1-ranked zero-byte lever with a stated ceiling of −0.028604 S — is closed.**
4. **Every other mechanism I could isolate is a non-supplier too.** uint8 at camera resolution
   is worth only **−0.8%** of the round trip. Anti-aliasing the scorer's read is **worse**
   (box +1.4%, triangle +16.5%); sharpening it is **worse** (monotone, +2% → +117%); a 2×2
   camera blur is **3.5×** worse. The blur axis has no minimum anywhere near us.
5. **The durable result is the pose-null subspace, and it is exact.** DERIVED from
   `frame_utils.py:51-76`: PoseNet's four luma planes are the full-resolution `Y` merely
   de-interleaved, and its two chroma planes are exact 2×2 box averages. So the invisible set is
   `{dY = 0 pointwise} ∩ {every channel zero-mean per 2×2 block}` = **exactly 6 of 12 DOF per
   block — 50% of the scorer-resolution RGB field**. Verified numerically: `dY ≤ 4.9e-16`,
   block-mean `≤ 3.3e-16`, idempotent to `6.7e-16`, 50.17% of a random field retained.
6. **Its only real loss channel is the uint8 camera grid — the chroma clamp is a non-issue.**
   At full strength the YUV6 leak from the `clamp_(0,255)` nonlinearity is **1e-5 rms** against a
   3.16 rms RGB perturbation (**0.0%** of the realized leak's energy); the realized leak is
   **0.266 rms**, all of it uint8 + box clipping at camera resolution. Measured pose attenuation:
   **d_pose ×38.58 → ×1.205** at matched α, and **44×** on the excess at matched perturbation size.
7. **Two of my own predictions were wrong and I am recording the diff**, per §0.

**Net: the whole camera-space blur family is now bounded on this vehicle — de-blur dead, blur
dead, sharpen dead, antialias dead, uint8 worth 0.8% — which reproduces rn1's law ("no undirected
operator is a reliable seg supplier, because the trade is symmetric") on a fifth independent
operator. The one thing that changed price is the pose constraint: a seg-side edit realized
through the exact preimage and projected onto the measured null space pays 44× less pose than the
same edit unprojected. That does not supply the gap by itself; it re-prices rt1's correction
channel, whose η gate is pose-constrained. Pointer UNMOVED.**

## §0 Prior-law prediction lines (written BEFORE measuring, per the anti-re-anchor law)

1. **ALIASING is the mechanism.** `D` reads a 2-tap kernel spanning ~1 camera pixel out of a
   5.17-pixel cell footprint at scale 2.276 — a point sample of a 2.28×-denser field. PREDICTION:
   antialiasing the read (box / triangle) will REDUCE flips, and rt1's measured salt-and-pepper
   signature (mean run 1.110, 92.2% runs of length 1) is the alias fingerprint.
   **WRONG, and I record it: antialiasing makes it WORSE** (box +7 flips, triangle +84 on 509 at
   n=8). The render is adapted to the scorer's exact narrow read; integrating the footprint
   destroys signal it depends on. The run-length signature is real but it does not imply aliasing.
2. **The render is natively camera-resolution** (rt1 §2.5b: "the render exists only at camera
   resolution, so there is no 'before R' version of it"). PREDICTION: no pre-R lever exists.
   **WRONG. The render is natively 384×512**, and the range-projection test settles it at rms
   0.233. rt1's scope note was too conservative; sr1 had it right and this unit confirms sr1
   independently and identifies the kernel sr1 did not name.
3. **rn1's law** — no undirected camera operator is a reliable seg supplier; pose costs 25–70×
   the seg leg. PREDICTION: the de-blur will be pose-killed. **HELD**, at 71× (α=0.25) to 119×
   (α=1.0).
4. **rp1/mp1** — uint8 is ~6% of the realization error and flips concentrate at near-zero margin.
   PREDICTION: the uint8 term of the round trip is small. **HELD**: −0.8%.
5. **m91 hub law / rt1 §2.8** — the residual is a tie, not a wall. PREDICTION: an undirected
   perturbation will move many pixels in both directions with `broken/fixed ≈ 1`.
   **HELD, and it is the mechanism**: `broken/fixed` = 0.848 / 1.000 / 1.150 / 1.349 across the α
   ladder — the trade is a fair coin that turns unfavourable as the move grows.

## §1 Instrument — PASS, with a caught inert leg

Independent tool, not rt1's. Pins per et4 (batch shape is part of the instrument): frozen
CPU-torch SegNet, **batch = 1 pair**, `torch.set_num_threads(8)`, `SegNet.preprocess_input`
verbatim. Pose legs import rn1's `Instrument` and its canonical `decode_gt`
(`frame_utils.yuv420_to_rgb` only) as reference form rather than reimplementing them.

| control | measured | reference | verdict |
|---|---:|---:|---|
| scored flips vs GT, n600 full field | **34,938** | rt1 34,938 | **EXACT** |
| round-trip flips vs shipped labels, n600 | **33,743** | rt1 33,743 | **EXACT** |
| advisory `d_seg` | **2.96173e-04** | contest-CUDA 2.9611e-04 | ratio **1.000213** |
| camera read support | 768 rows × 1024 cols = 786,432 px | rn1 768 / 1024 / 786,432 | **EXACT** |
| never-read fraction | **22.696926%** | rn1 22.6969% | **EXACT** |
| `A` row middle tap | [0.101470, 0.797058, 0.101472] | sr1 [0.101470, 0.797060, 0.101470] | 6 digits |
| my numpy `D` vs torch `interpolate` | max abs **4.23e-03** | — | 2e-5 relative, PASS |
| preimage right-inverse identity | **2.22e-16** | exact | PASS |
| de-blur ladder `α = 0` | **0 flips changed, 0 fixed, 0 broken, d_pose ×1.0000** | base | **PASS** |

**One inert leg, caught and fixed.** My first `dither` leg wrote `round(X + U(−0.5, 0.5))` on an
already-integer field — a no-op, and it returned a flip count *identical* to the base, which is
what exposed it. Corrected to score `X + U(−0.5, 0.5)` unrounded, which is the actual first-order
simulation of the pre-quantization render. Recorded because an inert leg that returns the base
count is indistinguishable from "this mechanism does nothing" unless you look at the digits.

**One self-caught bad probe, discarded.** My first structural probe estimated the camera field's
row rank by SVD of a column-subsampled matrix `g[:, ::8]` — 146 columns, so the reported "rank
146" was just the subsample width and carried no information. It is not used anywhere below; the
range-projection test in §2.1 replaced it.

## §2 The mechanism

### §2.1 The decoder renders at exactly the scorer's resolution, then blurs it

| test | measured | reading |
|---|---:|---|
| residual of the camera frame off `range(U_bilinear)`, frame_1 | **rms 0.2334, max 0.825, 1.22% > 0.5** | in-range to uint8 residue |
| same, frames 601 / 1199 | rms 0.2338 / 0.2340 | stable |
| same, frame_0 (the pose carrier) | rms 0.2624 / 0.2819 | also in-range |
| `A = D∘U` off-tridiagonal max, **U bilinear** | **0.000e+00** (rows and cols) | exact |
| `A = D∘U` off-tridiagonal max, **U bicubic** | 1.539e-02 / 1.542e-02 | not the kernel |
| `A` row sums | 1.000000000 | constants are fixed points |

So the chain is `m (384×512) → U_bilinear → 874×1164 → uint8 → D_bilinear → 384×512 → SegNet`.
The lift and the scorer's read are **inverse-shaped operations on the same lattice**, and their
composite is a tridiagonal low-pass with middle tap `[0.1015, 0.7971, 0.1015]`. The render is
never seen by the scorer; a blurred copy is. **That is where the round trip is manufactured.**

⚠ **Correction to two live descriptions.** The charter's chain ("bicubic up to 874×1164") and
CLAUDE.md's `differentiable_eval_roundtrip` docstring ("384 → 874 bicubic-up → uint8 → 384
bilinear-down") do not describe *this* vehicle's decoder: the lift here is **bilinear**, and the
bicubic hypothesis is refuted at 1.54e-2 against an exact zero. The historical PR95-lineage
modules that hardcode `mode="bicubic"` are a different vehicle and are not touched by this note.

### §2.2 The scorer's read operator, characterised

Scorer-free, from the operator alone (`RT2_OPERATOR.json`).

| quantity | value |
|---|---:|
| scale (rows / cols) | 2.276042 / 2.273438 |
| camera pixels read by `D` | **786,432 / 1,017,336 = 77.303%** |
| never read by either scorer | **22.696926%** |
| output-cell footprint | 5.174 camera px |
| effective taps (participation ratio 1/Σw²) | **2.467** |
| same, for a true box average | 7.138 |
| under-integration ratio | **2.097×** |
| rank of `D` / camera DOF | 196,608 / 1,017,336 = **19.326%** |

`D` is surjective, so **an exact preimage exists for any scorer-resolution target** — this is what
makes the whole cure family closed-form rather than a search. Note per vs1: the 22.70% blind mask
and the 80.67% resize nullity are **geometry, not bytes**; nothing here claims rate credit.

### §2.3 The mechanism ledger

n=8 seeded-random pairs (seed 20260817, never a prefix), base **509 flips vs GT**. SCOPE
reduction declared: these are mechanism *reads*, not counted ledger rows; the counted row is the
n600 base in §1 and the n=24 ladders in §3.

| mechanism isolated | leg | flips | Δ vs base | share of round trip |
|---|---|---:|---:|---:|
| **uint8 at camera resolution** | `dither` (score `X + U(−.5,.5)`, unquantized) | 505 | **−4** | **−0.8%** |
| — | **base (shipped round trip)** | **509** | **0** | — |
| under-integration, box | `area` (exact 5.17-px cell average) | 516 | +7 | +1.4% |
| preimage realization of `area` | `area_pre` (through `D`, uint8) | 518 | +9 | +1.8% |
| under-integration, triangle | `aa` (`antialias=True` kernel) | 593 | +84 | +16.5% |
| sharpen toward the narrow read | `unsharp` λ=0.25 | 519 | +10 | +2.0% |
| " | λ=0.5 | 528 | +19 | +3.7% |
| " | λ=1.0 | 569 | +60 | +11.8% |
| " | λ=2.0 | 659 | +150 | +29.5% |
| " | λ=4.0 | 1,106 | +597 | +117% |
| camera-resolution 2×2 blur | `boxblur` | 1,774 | +1,265 | +249% |

**Read three ways.** (a) **uint8 is 0.8%** — consistent with mp1's independent 6%-of-*field*-error
figure once you account for margin absorption (rp1: flipped sites sit at margin 0.0337 vs 5.6136
held, a 166× gap). (b) **The blur axis has no local minimum near us**: every step in either
direction is worse, and worse monotonically. The render sits at a point the scorer's exact narrow
read likes and both neighbours are downhill. (c) **`area_pre` reproduces `area` to 2 flips of
516** — the preimage machinery realizes an arbitrary named target with a **0.4%** uint8 penalty
(rms 0.148, max 0.496 = exactly the half-LSB bound), which is the positive control that licenses
§3.

## §3 The cure, built and measured — and CLOSED

`X' = X + R† (T − D X) C†ᵀ` with `R† = Rᵀ(RRᵀ)⁻¹`, `C† = Cᵀ(CCᵀ)⁻¹`. Two precomputed Gram solves
on diagonal matrices (the Gram of a 2-tap bilinear matrix at scale > 2 is diagonal, cond ≈ 2), so
this is closed-form and cheap, not a search. Target `T(α) = D X + α·(m̂ − D X)` with
`m̂ = U⁺X`. **Zero archive bytes**: every operand is the decoded render, no scorer is touched at
decode, and it is generic algorithm in `inflate.py` under rule 118. Both frames of every pair are
transformed, because PoseNet reads both.

### §3.1 Unprojected — pose-fatal

n=24 seeded-random pairs, seed 20260817, base 1,369 flips, `d_pose` aggregated in the scorer's
convention (mean of `d_pose` itself, never a mean of per-pair ratios — rt1 §6.2b).

| α | Δflips | fixed | broken | b/f | ΔS_seg | d_pose × | ΔS_pose | **net ΔS** | pose/seg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | **0** | 0 | 0 | — | 0 | 1.0000 | 0 | **0** | control PASS |
| 0.25 | **−16** | 105 | 89 | 0.848 | −0.000339 | 2.2701 | +0.024064 | **+0.023725** | **71×** |
| 0.50 | 0 | 179 | 179 | 1.000 | 0 | 6.4048 | +0.072703 | **+0.072703** | ∞ |
| 0.75 | +36 | 240 | 276 | 1.150 | +0.000763 | 16.6292 | +0.146182 | **+0.146945** | 192× |
| 1.00 | +98 | 281 | 379 | 1.349 | +0.002077 | 38.5809 | +0.247509 | **+0.249586** | 119× |

Removing the blur entirely costs **+0.2496 S — 26× the whole remaining gap, in the wrong
direction.** The single negative seg row (−16 at α=0.25) is **1.15σ**: 194 pixels traded, and
under a fair coin the net difference `fixed − broken` has sd `2√(npq) = 13.9`. It is not a
supplier.

### §3.2 Pose-null projected — the pose cost collapses, the seg verdict does not change

Same ladder, perturbation projected onto the measured null space before realization.

| α | Δflips | fixed | broken | b/f | ΔS_seg | d_pose × | ΔS_pose | **net ΔS** | scorer move (rms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | **0** | 0 | 0 | — | 0 | 1.0000 | 0 | **0** | 0.000 |
| 0.5 | +8 | 60 | 68 | 1.133 | +0.000170 | **1.0385** | +0.000906 | **+0.001075** | 0.801 |
| 1.0 | +27 | 108 | 135 | 1.250 | +0.000572 | **1.2052** | +0.004645 | **+0.005217** | 1.603 |
| 2.0 | +93 | 182 | 275 | 1.511 | +0.001971 | 2.5103 | +0.027755 | **+0.029726** | 3.205 |

**Matched-perturbation attenuation.** At α=1.0 unprojected the scorer input moves rms 2.406 for a
`d_pose` excess of **37.58**; the projected α=2.0 row moves rms **3.205** — 1.33× further — for an
excess of **1.510**. `d_pose` is an MSE, so scaling the unprojected excess to the same move gives
`37.58 × (3.205/2.406)² = 66.7`. Measured **1.510 vs 66.7 → 44.2× attenuation of the pose cost at
matched perturbation size.**

**The seg leg is unchanged by the projection, and stays negative.** So the de-blur is dead for a
reason that has nothing to do with pose: the trade is a fair coin (`b/f` 0.85 → 1.51) that turns
unfavourable as the move grows. This is rn1's law on a fifth operator, and it is the same reason
rt1's flat band repaint (+1.38 S) and sq1's truth paint (η −3.76) died: **the decoder cannot know
the sign of its own error, so an undirected move is a coin flip weighted against you.**

### §3.3 Where the pose-null leak actually comes from — MEASURED, not assumed

The projector is exact in exact arithmetic, so any residual pose cost is a realization leak.
Attributed on 4 seeded frames at α=1.0, in PoseNet's own YUV6 coordinates:

| channel | YUV6 rms leak | share of realized leak energy |
|---|---:|---:|
| chroma `clamp_(0,255)` nonlinearity (float target, no uint8) | **0.00001** | **0.0%** |
| realized through the camera preimage + uint8 + box clip | **0.26614** | 100% |
| unprojected, for scale | 3.08909 | — |

**The clamp is a non-issue on this vehicle: 5 orders of magnitude below the realized leak.** The
entire residual is the uint8 grid and the [0,255] box at camera resolution — the #532 mechanism,
in its exact place. Two consequences worth keeping:

- The leak grows **super**-linearly in α (realized target error rms 0.219 → 0.482 → 1.079 across
  α = 0.5 / 1.0 / 2.0) because the preimage's excursions start hitting the box. That caps how far
  any preimage-realized edit can be pushed before pose reappears.
- The leak is a property of **how many camera pixels the edit touches**, not of the edit's
  purpose. A full-frame edit pays the full 0.266; a small-support edit pays proportionally less.
  **DERIVED, not measured**: at rt1's ring-0 support (2,551,464 of 117,964,800 scorer cells =
  2.16%) the leak energy would be ~2.16% of full-frame, i.e. a `d_pose` ratio near ×1.004. That
  number is an extrapolation and must be measured before anything is built on it.

## §4 What this changes, and what it does not

| lever | prior status | this unit |
|---|---|---|
| sr1 **A1** zero-byte de-blur (its rank #1, ceiling −0.028604 S, sign "not settled scorer-free") | sealed FO-1, **never fired** | **CLOSED.** Seg 1.1σ at best and positive everywhere else; pose ×38.6. Both legs. |
| sr1 **FO-1**'s pre-registered bands | flips only | **The bands are unsafe as written.** rn1 measured pose at 25–70× the seg leg; a seg-only ladder on this vehicle would have read −16 flips at α=0.25 as within its LIVE band while the row actually costs **+0.0237 S**. Any successor ladder must carry pose. |
| "the round trip is aliasing" | my own §0 prediction | **REFUTED.** Antialiasing is worse in both kernels. |
| "the render is natively camera-res" (rt1 §2.5b) | scope caveat | **Refuted; it is natively 384×512.** rn1 §4 already lifted the caveat from the `D` side; this lifts it from the `U` side and names the kernel. |
| uint8 as a seg mechanism | mp1: 6% of field error | **0.8% of the round trip.** Not a supplier. |
| **PoseNet null space** | named in the Q3 family, exact kernel claimed **pre-quantization only** (hg1, #532) | **Now exact, sized, and its leak attributed**: 50% of scorer-res RGB DOF, clamp leak 1e-5, uint8-only loss channel, **44× pose attenuation at matched move.** |

**What it does NOT change.** The pointer. The seg axis is still 33,743 manufactured flips on a
one-pixel curve, still 43.4% Road↔Lane, still a tie at median deficit 0.105 (rt1 §2.8) — and this
unit adds a fifth undirected operator to the list that cannot break the tie. rt1's routing to the
renderer's own training stands, and hg1's sealed tr1 hinge A/B remains the live seg lever.

## §5 Sealed follow-on — NOT a fire-order for a cure, a fire-order for a PRICE

No cure cleared, so no cure is fired. What is worth one cheap row is the one quantity this unit
moved: **the pose price of a small-support seg edit.** It is the input rt1's correction channel
failed on (η 0.6235 against a 0.753 bar, pose-null-constrained) and sr1's waterfill re-priced
without ever measuring the pose leg per-cell.

**FO-A — the small-support pose leak.** Measure the realized `d_pose` of a *pose-null-projected,
preimage-realized* edit restricted to rt1's retained ring-0 support
(`free_band_mask.npy`, sha `649dd26f0843…`), at α ∈ {0.5, 1.0}, n ≥ 24 seeded-random pairs.
Command shape (this arm's tool, support flag NOT yet built — it is the one line of work FO-A
owes):

```
.venv/bin/python experiments/ddm_rt2_deblur_ladder.py \
    --n-pairs 24 --seed 20260817 --pose-null --alphas 0.0 0.5 1.0 \
    --support /Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816/free_band_mask.npy
```

Pre-registered bands, written before the run:
- `d_pose` ratio **≤ ×1.01** → the DERIVED ×1.004 holds; the pose constraint on rt1's channel is
  effectively free, and rt1 §6.4's third reopening condition (a better solver reaching η > 0.753)
  should be re-priced without the pose-null projection eating the seg gain. sr1's η-death margin
  (supplier for any η > 0.3871 with the ≥500-px guard) then becomes the live question.
- ratio **> ×1.05** → the small-support extrapolation is refuted; the uint8 leak does not scale
  with support the way the energy argument says, and that is itself the finding.
- `α = 0` must reproduce the base to the flip or the row is VOID.

Cost: $0, local CPU, ~15 min, no Modal, no launch. **Owner: MAIN.**

**Not queued, and why.** A directed (solver-driven) edit inside the null space is the obvious next
thought, and it is exactly what rt1 §6 already measured at η 0.6235 — re-running it inside the
null space is only worth it if FO-A shows the pose constraint was the binding half. That
ordering is deliberate: measure the price before paying for the solver.

## §6 Retained payloads (ALWAYS KEEP THE PAYLOAD)

Root `/Volumes/APDataStore/pact/ddm_rt2/` (APDataStore; VertigoDataTier has 893 MiB free and is
read-only for this arm).

| artifact | bytes | sha256 (prefix) | what it is |
|---|---:|---|---|
| `argmax_base_n600_s20260817.npy` | 117,964,928 | `2aeb1e6be0f7…` | **the n600 re-derivation's argmax field** |
| `argmax_area_n8_s20260817.npy` | 1,572,992 | — | box-average counterfactual |
| `argmax_aa_n8_s20260817.npy` | 1,572,992 | — | triangle counterfactual |
| `argmax_area_pre_n8_s20260817.npy` | 1,572,992 | — | preimage positive control |
| `argmax_dither_n8_s20260817.npy` | 1,572,992 | — | uint8 noise-floor leg |
| `argmax_boxblur_n8_s20260817.npy` | 1,572,992 | — | camera-blur control |
| `argmax_unsharp_l{0.25,0.5,1,2,4}_n8_s20260817.npy` | 1,572,992 ea | — | the sharpen ladder |

Receipts: `RT2_STRUCTURE_PROBE.json` · `RT2_OPERATOR.json` · `RT2_LEG_*.json` (11) ·
`RT2_DEBLUR_LADDER_n24_s20260817.json` · `RT2_DEBLUR_LADDER_n24_s20260817_posenull.json` ·
`RT2_DEBLUR_LADDER_n3_s20260817_nopose.json`. Logs `base_n600.log`, `ladder_n24.log`,
`ladder_posenull_n24.log`.

Tools: `experiments/ddm_rt2_structure_probe.py` · `experiments/ddm_rt2_mechanism_decomposition.py`
(stages `operator` / `leg` / `ledger`) · `experiments/ddm_rt2_deblur_ladder.py`.
Consumed unmodified: the wc1 retained decode `0.raw` (3,662,409,600 B, sha `e5539653…`), the hv1
ep0634 `decoded_spatial_tokens.rc64.bin`, the qs3 `gt_argmax_n600.npy`, and `upstream/videos/0.mkv`
via `frame_utils.yuv420_to_rgb` for the pose legs. `upstream/` was read, never written.

## §7 What this unit did NOT establish

- **No score, no pointer move.** Every number is `[macOS-CPU advisory]`.
- **The pose instrument's absolute level is not trusted.** rn1 measured it ~18.2× optimistic on
  pose. Every `d_pose` here is used as a RATIO against the same instrument's own base, which is
  the comparison that survives an offset; the absolute `ΔS_pose` figures are therefore
  **conservative in the wrong direction** — the real cost of the unprojected de-blur is larger,
  not smaller. Nothing in §3 depends on the absolute level.
- **n=24 is a SCOPE reduction, seeded-random.** Per m96 a random subset may REFUTE a bar (which
  is what happened: the de-blur loses on both legs at every α) but may not license a LIVE verdict.
  No LIVE verdict is claimed anywhere in this memo.
- **The n=8 mechanism reads in §2.3 are diagnostics, not counted ledger rows.** Only §1's n600
  base is a counted row.
- **The pose-null projection is measured only on the de-blur perturbation.** Its 44× attenuation
  is a property of that perturbation's spectrum realized through this preimage; a different edit
  could leak differently. FO-A is the first test of that.
- **The ×1.004 small-support pose ratio is DERIVED from an energy argument, not measured.** It is
  the thing FO-A exists to check, and no downstream claim may cite it as measured.
- **No causal claim about why the trade is a fair coin.** `broken/fixed ≈ 1` is measured on five
  operators now; the mechanism (rn1's "the decoder cannot know the sign of its own error") is
  inherited, not re-derived here.
