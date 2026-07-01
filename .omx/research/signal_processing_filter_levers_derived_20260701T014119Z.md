# Signal-processing / filter / physics levers for d_seg convergence + score — DERIVED from the measured pipeline transfer function

- **UTC** 2026-07-01T01:41Z · **authority** `[macOS-CPU signal-processing derivation] NON-PROMOTABLE`
- **pointer UNMOVED 0.19110** · score_claim **false** · promotable **false** · ready_for_exact_eval **false**
- **git** `50bb83e4c` · **Scope** $0, CPU-only, NO GPU, NO training, live n600 run (pid 38641) untouched.
- **Operator idea (2026-07-01)** the chain INR→render→R→SegNet→argmax is a MEASURED filter chain + threshold
  detector; derive amplification / signal-processing / physics / pass / range / gain levers that SPEED
  d_seg convergence AND help (not hurt) score. This memo is the "Understand"-mind deep-math axis.
- **Measured inputs** (all $0, CPU): (a) **R's transfer function measured directly** by grating MTF +
  dash-survival probe (`scratchpad/measure_R_mtf.py`, this session); (b) the n600 erasure/shift + persistence
  curves (`residual_inr_overturn_n600_full_erasure_shift_20260701T005600Z.md`, memo `bcf579509`); (c) the
  GT margin field (`gt_n600.npz['margins']`, 600×384×512, loaded once). The R operator source:
  `tac.local_acceleration.torch_levelset_inflate` (`torch_R`/`_R`) + the PR#95 roundtrip in
  `tac.differentiable_eval_roundtrip` (bicubic↑384→874 → uint8 → bilinear↓→384).

---

## 0. HEADLINE — the operator's central hypothesis is MEASURED-FALSE, and that is the finding

The operator framed R as "a KNOWN, computable low-pass transfer function" and proposed pre-emphasis /
deconvolution against it. **I measured R's actual MTF. R is very nearly ALL-PASS.** It is NOT the low-pass
in the chain. Deconvolving/pre-emphasizing R buys **≤ +1.25 dB (×1.15) at render-Nyquist — negligible.**

The chain has THREE cascaded low-pass / threshold stages; the measurement localizes the loss to the OTHER
two:

| stage | measured/derived attenuation at 2px feature | is it the bottleneck? |
|---|---|---|
| **R** (bicubic↑384→874 + uint8 + bilinear↓→384) | **\|H\|=0.842** (16% loss); dash contrast retained ≥92% | **NO — near all-pass** |
| **INR** (spectral-bias / NTK power-law) | dominant: composite 50%-survival at ~area 70px (~4× coarser than R Nyquist) | **YES (generator side)** |
| **SegNet stem** (stride-2 → 192×256 feature) + **argmax** (threshold on top1−top2 margin) | detector Nyquist ~2px@384; sub-2px features unresolvable at any contrast | **YES (detector side)** |

**Consequence (redirects the whole lever set):** the two $0-inflate-side amplitude levers the operator
hoped for (pre-emphasis-of-R, deconvolution-of-R) are near-worthless because R does not attenuate. The
real leverage is **training-side** (reshape the INR NTK so the finest *detector-resolvable* band converges
fast) plus a **phase** (not amplitude) inflate-side lever (sub-pixel pre-image for the shift slice). This
is the "negatives are suspect — measure, then overturn or accept" discipline applied to the operator's own
framing: measured, overturned the amplitude framing, re-derived the correct levers.

---

## 1. MEASURED — R's modulation transfer function |H_R(f)| (grating sweep, A_in=100 DN)

R applied to a pure sinusoid at spatial frequency `f` (cycles/pixel at the 384 render grid; Nyquist=0.5),
amplitude recovered by cos/sin projection. `_linear` = up→down (no round); `_contest` = up→uint8@874→down;
`_trainer` = up→down→uint8 (PR#95 order).

| f (cyc/px) | λ (px) | \|H\|_linear | \|H\|_contest | \|H\|_trainer |
|---:|---:|---:|---:|---:|
| 0.02 | 50.0 | 0.9999 | 0.9996 | 0.9984 |
| 0.10 | 10.0 | 0.9971 | 0.9960 | 0.9972 |
| 0.18 | 5.6 | 0.9834 | 0.9836 | 0.9830 |
| 0.26 | 3.8 | 0.9507 | 0.9507 | 0.9504 |
| 0.35 | 2.9 | 0.8965 | 0.8962 | 0.8965 |
| 0.40 | 2.5 | 0.8690 | 0.8686 | 0.8689 |
| 0.45 | 2.2 | 0.8486 | 0.8485 | 0.8482 |
| **0.48** | **2.1** | **0.8422** | **0.8418** | **0.8427** |

**No hard null in-band.** \|H\| falls monotonically from 1.0 to **0.842 at Nyquist** — a gentle roll-off, not
a cliff. **uint8 quantization is negligible** (linear≈contest≈trainer to 3 decimals; the round order does
not matter). This is because the up-to-874 adds no aliasing and the down-to-384 bilinear is the binding
Nyquist but only lightly attenuates the top octave.

**Dash-survival (isolated bar, peak retained contrast / input contrast):**

| width (px) | A=200 contest | A=8 (near quant floor) contest | A=8 output (DN) |
|---:|---:|---:|---:|
| 1 | 0.914 | 0.955 | 7.6 |
| 2 | 0.969 | 0.966 | 7.7 |
| 3 | 1.024 | 1.011 | 8.1 |
| ≥5 | ~1.01 | ~1.00 | ~8.0 |

Even a **1px** dash keeps **91%** of its contrast through R, and an 8-DN (≈3% of full-scale) dash comes out
at 7.6 DN — well above the 0.5-LSB uint8 floor. **R does not erase dashes.** (Contest vs linear identical →
quantization is not the eraser either.)

---

## 2. DERIVED — R deconvolution / pre-emphasis headroom (Wiener inverse of the measured MTF)

`G(f) = |H|/(|H|²+ε)`, ε=0.02. The inverse is well-conditioned everywhere (no null), so R is exactly
invertible — but the *gain* is the point:

| f | λ (px) | \|H\|_con | G_wiener | boost (dB) |
|---:|---:|---:|---:|---:|
| 0.30 | 3.3 | 0.928 | 1.053 | +0.45 |
| 0.40 | 2.5 | 0.869 | 1.122 | +1.00 |
| 0.48 | 2.1 | 0.842 | 1.155 | **+1.25** |

**Max achievable R-sharpening = +1.25 dB (×1.15) at Nyquist.** Nyquist limit: the render grid carries no
frequency above 0.5 cyc/px; the ↑874 step adds no information, the ↓384 step is the binding Nyquist. So
R-deconvolution's ceiling is ×1.15 — **not a d_seg lever.** (The physics: R is a resample+quantize, whose
composite kernel is close to a delta because ↑bicubic then ↓bilinear at the *same* final grid is a mild
smoothing, and uint8 at 255 levels over a ~235-DN headroom is ~0.2% relative — both far below the
generator/detector losses.)

---

## 3. DERIVED — where the low-pass ACTUALLY is (localized by elimination + the survival curve)

Since R is all-pass (§1), the measured composite survival-vs-scale curve (memo `bcf579509` §4) is
**entirely generator(INR)·detector(SegNet)**:

| dash area (px) | survival | | dash area | survival |
|---:|---:|---|---:|---:|
| 2–5 | 1.5% | | 40–80 | 55.2% |
| 5–10 | 5.1% | | 80–160 | 87.4% |
| 10–20 | 15.9% | | 160+ | 98.6% |
| 20–40 | 31.6% | | | |

50%-survival at **~area 70px ≈ ~8px linear** — **~4× coarser** than R's 2px Nyquist. The bottleneck scale is
4× above where R would put it → the loss is generator+detector, not R. Split of the two:

- **Detector floor (hard):** the SegNet EfficientNet-B2 **stride-2 stem** halves 384→192 immediately
  (CLAUDE.md: "artifacts below (256,192) invisible"). A feature < ~2px@384 → < ~1px@192 stem grid → below
  the stem Nyquist → cannot form an argmax class **at any contrast**. The area<5px dashes (3536 dashes,
  98.5% erased) sit at/below this floor → **unrecoverable by any amplitude lever**. This is a physics floor.
- **Generator gap (soft, the addressable part):** the area 5–80px dashes (survival 5–55%) ARE
  detector-resolvable but the INR under-produces them — the **NTK spectral bias**. This slice is the target
  of the training-side levers below.

**GT margin field (detector sensitivity, from `gt_n600['margins']`):** mean\|m\|=5.61, median 5.89; only
**1.38% of cells at \|m\|<0.5**, 2.67%<1.0, 4.83%<2.0. The detector-sensitive mass is a **thin codim-1
annulus** (matches the 94% flip-in-r2 localization). Fisher↔−margin Pearson 0.978 (memo) ⇒ Fisher (detector
gain ∂²/∂logit²) is concentrated in exactly this ~1.4–4.8% band. **This is the "WHERE" for every
capacity/gain lever.**

---

## 4. The six levers — derived EV (ΔS = d_seg-gap-closing; Δspeed = convergence acceleration)

Witness baseline d_seg = **0.006655**; sub-0.19-competitive needs ≤0.00118, sub-0.15 ≤0.00077 (memo §6).
Error split: **shift 76.5% (0.00509 d_seg) / erasure 23.5% (0.00156)**. ΔS below are DERIVED bounds/ranges
(labeled), NOT measured score rows.

**L1 — Pre-emphasis / gain kernel.** *Requested vs R; re-derived vs the true low-pass.* Kernel = 1/(gen·det
MTF), NOT 1/(R MTF). From §3 the boost needed at 5–80px is ×2–20 — but bounded two ways: (i) you cannot
pre-emphasize a frequency the INR cannot synthesize (spectral bias — L3 is the real fix), (ii) below the
detector Nyquist (2px@384) no contrast helps (§3 floor). **Real form:** a high-freq-band emphasis on the RGB
loss *localized to the annulus* (§3 margin band) that pushes lane-color contrast up for the
detector-RESOLVABLE-but-underproduced dashes (5–80px). ΔS: **moderate on the erasure slice** (attacks the
5–80px dashes ≈ ~15% of erasure mass ⇒ up to ~−0.0002 d_seg). **Training-side** (loss reweight), NOT $0.

**L2 — Deconvolution / pre-distortion of R.** *Amplitude form: measured-worthless* (§2, ≤+1.25 dB). *Phase
form: valuable.* R's resample shifts sub-pixel phase; the boundary that lands on the correct side of the
argmax separatrix after ↑↓ is a **sub-pixel pre-image solve** (generalize resize-null-preimage /
sub-pixel-placement). This is a **phase/position** lever for the **SHIFT slice** (76.5%, the ≤3px separatrix
wobble), not an amplitude lever for erasure. ΔS: **moderate-high on the shift slice** if it converts primed
flips (margin<0.5 = 64% of flips) by exact placement. **$0 inflate-side** (deterministic pre-image at
decode) for the amplitude=0 part; the placement target is set at train time.

**L3 — NTK spectral reshaping / band-pass preconditioner. ★ THE dominant lever (speed AND score).** The INR
is the real low-pass (§3). Fourier/curvelet-feature INR NTK eigenvalues λ_k decay as a power law in
frequency; GD converges mode k at rate ∝ λ_k, so fine-dash modes converge ∝ (λ_lo/λ_hi)× slower = the
~30k-epoch slow tail. **Whitening preconditioner:** set per-scale feature amplitude ∝ 1/√(λ_scale) so the
finest *detector-resolvable* band (down to 2px@384, NOT finer — no point synthesizing below the §3 floor)
has NTK eigenvalue ≈ the coarse band ⇒ **uniform convergence**. This is WHY the directional/curvelet
scaling already works (implicit NTK reshaping toward the boundary-tangent band); make it **explicit +
tunable**. Derived speedup: the fine band is ~1.5–16% represented vs ~99% coarse ⇒ effective λ ratio
~6–60× ⇒ whitening can cut fine-band convergence time by up to that factor (bounded by conditioning /
step-size stability; realistic **~3–10× on the finest band**). ΔS: attacks the generator gap = the
addressable erasure slice ⇒ up to ~−0.0003 d_seg AND reaches it in far fewer epochs. **Training-side; the
#1 speed win.** Composes with a frequency curriculum (coarse→fine anneal).

**L4 — Matched-filter margin-threshold placement.** From §3: 76.5% of flips are primed (realized margin
−0.60; 64% at GT margin<0.5). The max-SNR nudge to flip a cell is along **+∂logit/∂input** (the
margin-saliency #141 field) with magnitude ∝ margin/‖saliency‖ — tiny (~0.5-DN-equivalent). **Weight the
witness loss by the saliency direction at the annulus** so representational effort lands exactly on the
flip-reachable direction. ΔS: **high on the shift slice** (the primed flips are the cheapest d_seg). Δspeed:
also faster (gradient aimed at the binding cells). **Training-side** (needs the saliency = one backward
through SegNet; already the #141 field). Composes with L2-phase and L5.

**L5 — UNIWARD inverse-steg gain-placement (the "WHERE" prior).** Max detector-response per unit cost =
high Fisher (low margin = the §3 annulus, 1.4–4.8% of frame) × low cost (textured / high-local-variance,
UNIWARD embedding-suitability). This is the **capacity-routing prior** already central to the frontier;
the measurement PINS it to the 1.4–4.8% margin band (Fisher↔−margin 0.978). ΔS: it does not add flips by
itself — it MULTIPLIES L1/L3/L4 by telling them where to spend. Δspeed: prunes ~95% of the frame from the
optimization. **Train-side prior; free to compute** ($0, from the margin field). Composes with all.

**L6 — Subspace / dimensionality reduction.** The residual lives on **~8 nonlinear DOF (Whitney ~17–19)**.
Project the optimization / capacity onto that subspace ⇒ converge only the DOF that carry d_seg. Δspeed:
direct (fewer effective parameters in the binding directions). ΔS: neutral-to-positive (removes
off-manifold noise). **Training-side.** Composes with L3 (band-pass in frequency × subspace in the code
manifold = the natural joint preconditioner).

### Ranked lever table

| # | lever | derived ΔS (d_seg-gap) | Δconvergence-speed | $0-now vs needs-hours-run | measured/derived basis | composes with |
|---|---|---|---|---|---|---|
| **L3** | **NTK band-pass / whitening preconditioner** | up to ~−3e-4 (erasure, addressable part) | **~3–10× on finest band ★** | **training-side** (hours-run) | §3 survival curve = gen·det MTF (R eliminated §1); NTK power-law | L5, L6, freq-curriculum |
| **L4** | matched-filter margin-saliency placement | **high on shift** (~−1e-3 to −2.5e-3 if half the primed flips convert) | faster (grad aimed at binding cells) | training-side (hours-run) | margins §3 (64% flips <0.5); #141 saliency | L2-phase, L5 |
| **L2** | R **phase** pre-image (sub-pixel placement) — NOT amplitude | moderate-high on shift | n/a (placement) | **$0 inflate-side** (target set train-time) | §2 R invertible in phase; resize-null-preimage | L4 |
| **L5** | UNIWARD Fisher/cost gain-placement (WHERE prior) | multiplier on L1/L3/L4 | prunes ~95% of frame | **$0 to compute** (prior) | margin field §3, Fisher↔−margin 0.978 | all |
| **L6** | subspace (~8–19 DOF) projection | neutral-to-+ | direct (fewer binding DOF) | training-side (hours-run) | dimensionality ~8 / Whitney 17–19 | L3 |
| **L1** | pre-emphasis vs gen·det (NOT vs R) | moderate on erasure (~−2e-4, detector-resolvable slice only) | — | training-side (RGB-loss reweight) | §3 gen·det gap; §3 detector floor caps it | L5 |
| ~~L2a~~ | ~~amplitude deconvolution of R~~ | **~0 (measured ≤+1.25 dB)** | — | $0 but worthless | §1/§2 R near all-pass | — |

---

## 5. Bake-in recommendation (top-3) + the $0 vs training split

**Bake into the optimal-form (hours) run — the real d_seg + speed wins:**
1. **L3 NTK band-pass whitening** (per-scale curvelet amplitude ∝ 1/√λ_scale, capped at the 2px@384
   detector Nyquist) — the dominant convergence-speed lever; makes the finest addressable band converge in
   ~3–10× fewer epochs. Compose with a coarse→fine frequency curriculum.
2. **L4 matched-filter / margin-saliency loss weighting** at the annulus — the highest-d_seg lever (the
   shift slice is 76.5% and primed).
3. **L5 UNIWARD Fisher/cost WHERE-prior** — free to compute; multiplies L3+L4 and prunes 95% of the frame.
   (L6 subspace folds in naturally with L3.)

**Pure-$0 inflate-side:** only **L2-phase** (sub-pixel pre-image placement) is a genuine free-decode lever,
and only for the SHIFT slice. **L1/L2 amplitude pre-emphasis/deconvolution of R are NOT worth building** —
measured near-all-pass (this is the money-saving finding: don't spend a session on R deconvolution).

**Hard floor to respect (physics):** dashes < ~2px@384 (area<5px, ~3536 dashes, 98.5% erased) are below the
SegNet stem Nyquist and are **unrecoverable by any generator/amplitude lever** — they need a *store* (Yousfi
flip-sidecar) or a *deterministic openpilot-lane raster* (the #203 gate), NOT more contrast. Levers L1/L3
should target the **5–80px** detector-resolvable band and stop there.

---

## 6. Adversarial after-audit (prove from filter theory + measured response, not analogy)

- **The R-low-pass claim is refuted by direct measurement, not asserted.** §1 measured \|H_R\|≥0.842 to
  Nyquist with 3 methods (linear/contest/trainer) agreeing to 3 decimals, plus dash-survival ≥91% at 1px.
  The Wiener inverse (§2) is the filter-theory proof that ≤+1.25 dB is the *ceiling*, set by the render-grid
  Nyquist (the ↑874 adds no information). Analogy played no role.
- **The bottleneck localization is by elimination + the measured survival curve**, not vibe: R eliminated
  (§1) ⇒ the composite survival curve (§3) is gen·det ⇒ 50%-survival at 4× the R Nyquist ⇒ the loss is
  generator(NTK)+detector(stem). The detector floor at 2px@384 is the documented stride-2 stem
  (CLAUDE.md verified), giving a hard, physics-grounded lower bound on the erasure slice.
- **The speed claim for L3 is NTK theory + the measured eigenvalue ratio proxy** (fine-band representation
  1.5–16% vs coarse 99% ⇒ λ ratio 6–60×; realistic 3–10× after conditioning), not analogy. It also EXPLAINS
  the already-observed curvelet-scaling win (implicit NTK reshaping) — a consistency check.
- **Honest downgrades:** the operator's two named $0 levers (pre-emphasis-of-R, deconvolution-of-R
  amplitude) are measured near-worthless; L2 survives only in its *phase* form. ΔS figures are DERIVED
  ranges, not exact-eval rows.
- **Uncaptured:** I did not run SegNet ($0 but touches the scorer + slow); the detector floor is inferred
  from the stem architecture + the survival curve, not a per-dash SegNet ablation — a named residual for a
  future $0-CPU SegNet stem-response probe.

## 7. means≠ends

This DERIVES + MEASURES leverage (a MEANS): R is not the low-pass; the wins are training-side NTK/saliency
levers + one $0 phase-placement lever, all localized to the 1.4–4.8% margin annulus and the 5–80px
detector-resolvable band. The pointer moves only on a byte-closed `upstream/evaluate.py` row (CPU/CUDA,
never MPS) < 0.19110. Feeds #201 (synthesis); composes with #203 (the free lane-raster is the right
mechanism for the sub-2px hard floor L1/L3 cannot reach).
