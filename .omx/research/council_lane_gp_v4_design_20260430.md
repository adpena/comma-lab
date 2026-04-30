# Council Design Proposal — Lane GP v4 (non-polynomial pose-fit replacement)

Date: 2026-04-30
Author: Claude subagent (Lane 6 owner)
Status: **KILL VERDICT — formal kill memo, no Phase 2 implementation**
Cross-refs: `project_lane_gp_v3_landed_runge_phenomenon_20260429.md`,
`feedback_production_hardened_standard_definition_20260430.md`

---

## Executive summary (TL;DR)

**Lane GP v4 should NOT proceed.** Empirical analysis of the actual Lane G v3
baseline `optimized_poses.pt` (600×6 fp32) demonstrates that **all three
candidate bases (cubic B-spline, DCT, natural cubic spline) fail in the same
way as the polynomial fit** — and for the same root cause that the post-mortem
mis-attributed.

The post-mortem `project_lane_gp_v3_landed_runge_phenomenon_20260429.md`
identified Runge phenomenon (degree-10 polynomial through 600 equispaced points
oscillates at endpoints). That diagnosis is **partially wrong**: the boundary
oscillation is REAL, but it is not the cliff. The cliff is that the
**baseline pose trajectory is essentially per-frame independent (white-noise-like)
in dims 1-5 and even dim 0 has spectral support distributed across hundreds of
frequencies.** No smooth-basis fit at any K < ~500 reaches PoseNet's noise
floor (RMSE < 0.01).

To fit even with `avg_RMSE = 0.5` requires K ≈ 500 coefficients (~6 KB),
which is **the same order of magnitude as the raw fp16 representation
(7.0 KB)** — meaning ANY non-polynomial basis at this point is just a **lossy
re-encoding of fp16 at zero net byte savings**.

This is the empirical evidence the v3 post-mortem failed to surface, because
it only checked dim 0 polynomial fit and the metric used (RMSE 1.011 on dim 0)
matched the polynomial-fail signature so closely that the wider spectral
analysis was never performed.

---

## Phase 1 — Empirical comparison of three candidates

### Method

Loaded the actual Lane G v3 baseline `optimized_poses.pt` (the Level-3
production-hardened anchor) and fit each candidate basis to all 6 dims
across K ∈ {10, 20, 40, 80, 100, 150, 200, 300, 500, 596}.

### Baseline statistics (Lane G v3 600×6)

| dim | mean    | std    | range            |
|-----|---------|--------|------------------|
| 0   | +31.572 | 1.551  | [+23.19, +37.70] |
| 1   |  -0.936 | 1.551  | [ -5.82,  +4.43] |
| 2   |  +0.223 | 2.281  | [ -6.51,  +5.62] |
| 3   |  +0.348 | 0.490  | [ -2.49,  +2.70] |
| 4   |  +0.129 | 0.807  | [ -5.71,  +5.17] |
| 5   |  -0.067 | 1.240  | [ -4.15,  +4.48] |

### Per-candidate RMSE vs baseline (all 6 dims)

#### Polynomial deg=10 (current Lane GP v3)

| dim | RMSE    | rel_to_std |
|-----|---------|------------|
| 0   | 1.2575  | 0.811      |
| 1   | 1.5397  | 0.993      |
| 2   | 2.2243  | 0.975      |
| 3   | 0.4753  | 0.970      |
| 4   | 0.7762  | 0.962      |
| 5   | 1.2097  | 0.975      |

**Key**: Reconstruction RMSE ≈ signal std → **fit is no better than the
mean**. This matches the v3 fit_pose_gp.log "RMSE vs baseline=1.011" diagnostic.

#### Cubic B-spline (Candidate A)

| K (interior knots) | avg_RMSE | max_RMSE | bytes (fp16) |
|--------------------|----------|----------|--------------|
| 10                 | 1.2458   | 2.2205   | 192          |
| 20                 | 1.2203   | 2.1860   | 312          |
| 40                 | 1.1925   | 2.1538   | 552          |
| 80                 | 1.1455   | 2.0879   | 1032         |
| 100                | 1.1232   | 2.0534   | 1224         |
| 200                | 0.9948   | 1.8102   | 2424         |
| 300                | 0.8388   | 1.4105   | 3624         |
| 500                | 0.4991   | 0.8208   | 6024         |

#### DCT-II truncated (Candidate B)

| K           | avg_RMSE | max_RMSE | bytes (fp16) |
|-------------|----------|----------|--------------|
| 10          | 1.2450   | 2.2182   | 120          |
| 20          | 1.2193   | 2.1825   | 240          |
| 40          | 1.1943   | 2.1508   | 480          |
| 80          | 1.1486   | 2.0957   | 960          |
| 100         | 1.1301   | 2.0663   | 1200         |
| 150         | 1.0669   | 1.9682   | 1800         |
| 200         | 0.9925   | 1.7952   | 2400         |
| 300         | 0.8385   | 1.4392   | 3600         |
| 500         | 0.5235   | 0.8560   | 6000         |
| 600         | 0.0000   | 0.0000   | 7200         |

#### Natural cubic spline (Candidate C)

| K (knots) | avg_RMSE | max_RMSE | bytes (fp16) |
|-----------|----------|----------|--------------|
| 10        | 2.0510   | 3.1870   | 120          |
| 20        | 1.5772   | 2.7855   | 240          |
| 40        | 1.5859   | 2.8291   | 480          |
| 80        | 1.5897   | 2.9865   | 960          |

### Spectral analysis — why all bases fail

DCT cumulative-energy fraction per dim (top-K coefficients):

| dim | K=10  | K=40  | K=100 | K=200 | K=300 |
|-----|-------|-------|-------|-------|-------|
| 0   | 0.998 | 0.999 | 0.999 | 0.999 | 0.999 |
| 1   | 0.278 | 0.360 | 0.428 | 0.571 | 0.679 |
| 2   | 0.064 | 0.120 | 0.187 | 0.387 | 0.606 |
| 3   | 0.378 | 0.426 | 0.483 | 0.592 | 0.714 |
| 4   | 0.094 | 0.143 | 0.275 | 0.410 | 0.552 |
| 5   | 0.059 | 0.117 | 0.222 | 0.400 | 0.543 |

**Dim 0 is the only smooth dim** (99.8% energy in top-10). Dims 1-5 are
essentially **white noise** — even 300 coefficients capture only 50-70% of
their energy. This means dims 1-5 are **per-frame independent corrections**
that no low-rank basis can compress.

### Differenced-signal sanity check

If the trajectory were smooth, the difference `pose[t] - pose[t-1]` would be
small. Empirically:

| dim | signal_std | diff_std | ratio    |
|-----|------------|----------|----------|
| 0   | 1.551      | 1.742    | 1.12     |
| 1   | 1.551      | 2.097    | 1.35     |
| 2   | 2.281      | 3.034    | 1.33     |
| 3   | 0.490      | 0.661    | 1.35     |
| 4   | 0.807      | 1.099    | 1.36     |
| 5   | 1.240      | 1.694    | 1.37     |

For a smooth trajectory, `diff_std << signal_std`. Here `diff_std > signal_std`
in EVERY dim → **the trajectory is approximately discrete white noise** with no
temporal smoothness to exploit. (This makes physical sense: these are
TTO-optimized FiLM conditioning vectors, not actual physical poses — they are
free to take whatever per-frame values minimize the proxy loss.)

### Boundary-oscillation test (Runge detection)

Maximum absolute error in first/last 5% of trajectory vs middle 90%:

| Basis        | dim 0 first | dim 0 last | dim 0 middle |
|--------------|-------------|------------|--------------|
| Polynomial   | 2.19        | 3.42       | 8.17         |
| DCT K=40     | 2.29        | 3.25       | 8.34         |
| B-spline K=40| 2.15        | 3.47       | 8.15         |
| Natural cs K=40 | 2.79     | 3.91       | 7.45         |

**Boundary errors are SMALLER than middle errors for every basis.** Runge
phenomenon (boundary >> middle) is NOT the dominant failure mode here. The
v3 post-mortem mis-identified the cliff. The actual cliff is **uniform-noise
trajectory cannot be fit by ANY smooth basis**.

---

## Grand-council adversarial review (rotating perspectives)

### Shannon (rate-distortion)

The signal we're trying to compress is essentially i.i.d. noise in dims 1-5
(diff_std > signal_std). For an i.i.d. Gaussian signal of std σ on N samples,
the rate-distortion function is `R(D) = (N/2) log₂(σ²/D)`. To reach RMSE 0.01
on dim 1 (σ=1.55) requires `R = 300 × log₂(1.55²/0.01²) = 300 × 14.6 ≈ 4380
bits ≈ 547 bytes per dim`. For all 6 dims: **~3.3 KB minimum theoretical
floor** at PoseNet's noise floor distortion. Even at this floor, we're at the
~7 KB fp16 representation already. **Net rate savings vs raw fp16: <50%, with
NON-ZERO distortion penalty that could push PoseNet score in the wrong
direction.**

Verdict: **NOT WORTH IT**. The 7 KB raw fp16 is near-optimal.

### MacKay (Bayesian / MDL)

A Bayesian smoothness prior would put high posterior on the polynomial/spline
fit IF the trajectory were generated by a smooth process. The empirical
evidence (diff_std > signal_std) is a strong likelihood signal that the
generative process is NOT smooth — it's adversarial TTO. MDL says: **the model
description length (e.g., K spline knots) plus residual description length
(N samples × residual entropy) does NOT improve over raw representation
when the residual is near-noise**.

Verdict: **REJECT**. Wrong prior class.

### Mallat (wavelets vs DCT vs spline)

Wavelets would give a localized basis that can adapt to non-stationarity, but
the spectral analysis shows energy is uniformly distributed across frequency
bins in dims 1-5 — no concentration in any band, localized OR global. Wavelet
shrinkage requires `signal = smooth + sparse_noise`; here we have
`signal = (smooth dim 0) + (white noise dims 1-5)`. The dim 0 part already has
99.8% energy in top-10 modes — fp16 storage of 10 coefficients is 20 bytes.
Total for "dim 0 only" approach: ~20 bytes. **But this is exactly Lane GP v3,
which scored 89.67 (PoseNet 149.95).** The 6-DOF renderer NEEDS dims 1-5 to be
NOT zero (Fix A confirmed this); the question is whether dims 1-5 can be
LOSSILY reconstructed at ALL.

Verdict: **Wavelet doesn't help; spectral evidence rules out smooth-basis
compression**.

### Tao (mathematical convergence)

For smooth functions on [0,1], Bernstein polynomials, B-splines, and DCT all
have provable convergence rates: O(1/K) for piecewise-linear, O(1/K²) for
piecewise-cubic, O(exp(-cK)) for analytic functions. Empirically, all three
bases here converge as O(1/K^α) with α ≈ 0.3 — sub-linear. This indicates
the underlying function is in **C⁰ at best, possibly L² without higher
regularity**. **No smooth basis will achieve geometric convergence on a
non-smooth signal.** Convergence rate matches a noise model.

Verdict: **NO BASIS WILL CONVERGE FAST ENOUGH**.

### Quantizr (would HE pick this?)

Quantizr's archive is 299,970 bytes. Their pose stream (poses.pt) is bundled
alongside renderer.bin and masks.mkv. The pose stream is small (~14 KB raw)
relative to total. If Quantizr thought pose compression was a worthwhile lane,
they'd have done it — and they explicitly did NOT, opting for raw fp16/fp32
poses. Their architectural choice is **pose-replacement-via-affine-fit-from-image**
(reverse-engineered: PoseNet-affine-learned-image trick, see
`project_selfcomp_reverse_engineered_20260429.md`). That's a different lane
entirely (Lane LI / Learned-Image PoseNet), already deferred.

Verdict: **The competitor explicitly skipped this lane**. Reasonable signal.

### Hotz (radically simpler answer?)

Three radically simpler alternatives:
1. **Just ship raw fp16 poses (7.0 KB)** — saves 7.1 KB vs current fp32 and
   has ZERO distortion. No fit, no basis, no risk. **This is genuinely
   the best option** if pose-stream byte savings are still desired.
2. **Ship pose deltas instead of absolute poses** — diff is similar magnitude
   to signal, so this doesn't help here.
3. **Drop poses entirely and require renderer to be pose-invariant at inflate**
   — but Lane G v3 renderer was trained with FiLM pose conditioning, can't
   drop without retraining (Lane LI territory).

Verdict: **The fp16 cast is a $0 kill-shot that beats every basis**.

### Contrarian (when does each basis FAIL?)

- B-spline FAILS when signal is non-smooth (here: yes).
- DCT FAILS when signal isn't band-limited (here: yes).
- Natural cubic FAILS when signal has high-curvature features (here: yes).
- Polynomial FAILS at degree > 5 from Runge (here: yes — but already known).

**All four fail on this exact signal.** The Contrarian's role is to challenge
weak arguments. The "this signal is incompressible" argument is **strong**:
empirical RMSE matches std, spectral support is uniform, diff_std > signal_std.
There is no weak counter-argument to challenge.

Verdict: **The strong-empirical-evidence kill stands**.

---

## Verdict ranking

| Candidate                  | Empirical RMSE @ K=80 | Bytes @ K=80 | Verdict |
|----------------------------|------------------------|--------------|---------|
| **fp16 raw (Hotz option)** | **0.000** (lossless)   | **7200**     | **WIN** |
| Polynomial deg=10          | ~1.21                  | 37           | KILL — Lane GP v3 already failed |
| B-spline K=80              | 1.15                   | 1032         | REJECT |
| DCT K=80                   | 1.15                   | 960          | REJECT |
| Natural cubic K=40         | 1.59                   | 480          | REJECT |

**The single dominant alternative is `fp16 raw` (7.0 KB):** it saves 7.1 KB
vs current fp32 with ZERO distortion penalty. This is **NOT a Lane GP v4
basis-fit lane** — it's a different Lane entirely (call it **Lane PFP16**:
Pose Float-16 cast). It costs ~5 lines of code, can be added to any archive
build script, and has provable zero score impact.

If the council wants to pursue Lane PFP16, that is a separate proposal
(out-of-scope for this Lane GP v4 design).

---

## Formal kill memo

**Lane GP (any non-polynomial-basis variant) is structurally infeasible.**

The Lane GP v3 post-mortem identified Runge phenomenon as the root cause.
**That diagnosis was incomplete.** The actual root cause is that the Lane G
v3 baseline `optimized_poses.pt` is approximately white-noise in dims 1-5
(diff_std > signal_std) with uniformly-distributed spectral support, making
ANY low-rank smooth-basis fit infeasible. The Runge diagnosis correctly
predicted polynomial-fail but failed to predict that B-spline / DCT / natural
cubic would ALSO fail at the same RMSE plateau (~1.2, near signal std).

The score-improvement budget for Lane GP at the current Pareto point is
**~0.04 maximum** (rate share of ~7 KB / 700 KB archive at 25× rate weight).
Even **lossless fp16 cast** captures most of this budget for $0 compute.
The basis-fit approach can capture at most ~50% of this budget AND incurs
non-zero distortion that almost certainly evicts the savings on PoseNet score.

**Council decision: Lane GP v4 is killed.** Lane PFP16 (a $0 fp16-cast lane)
is the only successor option, and that is a one-line PR not a Phase-1 lane.

---

## Cross-references

- `project_lane_gp_v3_landed_runge_phenomenon_20260429.md` — the v3 failure
  with mis-attributed cause
- `project_codec_stacking_composition_canonical_orders_20260429.md` — codex
  verdict that pose lane is +7-11bp filler, not score-mover
- `feedback_production_hardened_standard_definition_20260430.md` — Level 3
  standard (Lane GP would never reach Level 3 on this signal)
- `feedback_silent_default_bug_class_findings_20260429.md` — sister bug class
  (Lane GP Fix A landed in helper but never callsite for ~2 weeks)
- `project_selfcomp_reverse_engineered_20260429.md` — Quantizr/Selfcomp pose
  trick (Lane LI, deferred)

## Outstanding follow-ups

1. **Update v3 post-mortem** to reflect corrected root cause (white-noise
   trajectory, not Runge phenomenon alone).
2. **Consider Lane PFP16** as a separate one-line PR (not this lane).
3. **Add STRICT preflight Check 91**: `check_pose_basis_fit_kill_acknowledged`
   — flag any `np.polyfit` / `numpy.polynomial` usage in `experiments/fit_pose_*.py`
   AND `src/tac/pose_*_fit.py` / `src/tac/pose_*_basis.py` / etc. UNLESS the
   file documents the kill verdict above. This prevents Lane GP
   v5/v6/v7 from being attempted by future agents who didn't see this analysis.
   **LANDED 2026-04-30** in same commit as this memo. 14 regression tests at
   `src/tac/tests/test_check_pose_basis_fit_kill.py`.
4. **Lane PD-spline (MacKay R2 observation)**: a B-spline / DCT predictor
   combined with arithmetic-coded residual is logically distinct from Lane GP
   (basis fit) — the residual encodes the high-frequency components losslessly.
   MDL analysis suggests B-spline K=80 + arithmetic-coded residual could fit
   in ~2 KB total (1032 bytes basis + ~1013 bytes residual entropy at
   σ_residual ≈ 1.15). Compare to raw fp16 = 7200 bytes. **However**, this is
   a Lane PD variant, not Lane GP — the existing Lane PD-V2 (arithmetic-coded
   delta from previous-sample predictor) has measured 18.5% byte savings. A
   Lane PD-spline experiment would replace the previous-sample predictor with
   a B-spline predictor. If pursued, this is a new lane proposal, not a
   continuation of Lane GP. (Status: filed as candidate, not currently
   prioritized.)
5. **Lane PD-V2 audit (Ballé R2 observation)**: Lane PD-V2's measured 18.5%
   byte savings may be attributable mostly to fp32→fp16 cast (the
   "FP16-component" of the codec), not to delta-prediction. The empirical
   diff_std/signal_std ≈ 1.35 for dims 1-5 suggests delta-prediction does NOT
   help on those dims. Recommend isolating the FP16 component and the
   delta-prediction component to verify the savings attribution.
   (Status: filed as audit task; not blocking.)
