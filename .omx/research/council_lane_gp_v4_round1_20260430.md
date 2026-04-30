# Lane GP v4 — Adversarial Review Round 1

Date: 2026-04-30
Reviewer perspectives: Yousfi, Fridrich, Contrarian, Quantizr, Hotz
Object of review: Lane GP v4 KILL VERDICT + Check 91 + kill marker + 11 tests
Counter on entry: 0/3 clean passes
Files reviewed:
- `.omx/research/council_lane_gp_v4_design_20260430.md` (the kill memo)
- `src/tac/preflight.py` Check 91 (`check_pose_basis_fit_kill_acknowledged`)
- `src/tac/tests/test_check_pose_basis_fit_kill.py` (11 tests)
- `experiments/fit_pose_gp.py` (kill marker added)

## Round 1 perspectives

### Yousfi (contest-design rigor)

The kill verdict rests on the empirical claim that the Lane G v3 600×6 baseline
trajectory is "approximately white-noise in dims 1-5" (`diff_std > signal_std`).
**Question**: is `diff_std > signal_std` actually a sufficient test for
incompressibility, or is it just a NECESSARY one?

**Counter**: For ANY truly white noise signal, `diff_std = sqrt(2) × signal_std ≈
1.414 × signal_std`. Empirical values are 1.12, 1.35, 1.33, 1.35, 1.36, 1.37 —
all near sqrt(2). This IS the white-noise signature, with the dim-0 outlier (1.12)
matching the spectral evidence (99.8% energy in top-10 modes).

**Status**: argument holds. NO ISSUE.

### Fridrich (steganalysis lens)

**Question**: could a Fridrich-style local-variance-weighted basis (UNIWARD-like)
exploit the texture-dependent scoring of PoseNet? The kill memo only tested
GLOBAL bases (polynomial / DCT / spline) — what about LOCAL bases like wavelet
shrinkage with adaptive thresholds?

**Counter**: The DCT spectral analysis shows energy is uniformly distributed
across ALL 600 frequency bins for dims 1-5 (e.g. dim 2 has only 60.6% energy
in top-300 of 600 bins → energy is approximately flat). Wavelet shrinkage
requires `signal = smooth + sparse_noise` decomposition; here the empirical
spectrum is FLAT (no sparse component to exploit). Wavelet was explicitly
addressed in the Mallat perspective in the design memo, with the verdict "no
spectral concentration in any band, localized OR global."

**Status**: NO ISSUE. The Mallat section of the design memo addresses this.

### Contrarian (challenge weak arguments)

**Question 1**: The kill verdict cites "RMSE 1.2 ≈ signal std → fit no better
than mean" but rest is the **score-translation argument**: how do we know that
RMSE 1.2 maps to PoseNet distortion 149.95 (Lane GP v3 score)? Could a basis
with RMSE 0.5 actually land at PoseNet distortion < 1.0 instead of just being
"50% as bad"?

**Counter**: The 89.67 score for Lane GP v3 was measured at PoseNet distortion
149.95 (vs Lane G v3 baseline 0.003) → ~50,000× regression. Lane GP v3's
recorded RMSE-on-dim-0 was 1.011, ~100% of signal std. If the basis hits
RMSE 0.5 (50% of signal std), the input is still off by ~σ/2 in EVERY frame.
Even if we are extremely generous and assume PoseNet distortion scales
quadratically with input perturbation σ², we get distortion → ~150 × 0.25 ≈
37.5, still 12,000× over baseline. The score budget for the pose lane is at
most 0.04 (rate share). To EVICT 0.04 in distortion, PoseNet must stay below
~0.05 — which requires basis RMSE in the ~0.005 range. Empirically that
needs K ≥ 500 → which is the same bytes as raw fp16 with ZERO distortion.

**Question 2**: Check 91 only scans `experiments/fit_pose_*.py`. What if a
future agent puts the smooth-basis fit in `src/tac/pose_*_fit.py` instead?
The check has a structural blind spot.

**Status**: REAL ISSUE. Check 91 needs to also scan `src/tac/pose_*_fit.py`
+ `src/tac/pose_*_basis.py` patterns to catch module-level evasion.

### Quantizr (competitor lens)

**Question**: Quantizr's 0.33 archive ships raw poses (~14KB). Selfcomp's 0.38
ships PoseNet-affine-learned-image (different lane entirely — Lane LI). Neither
attempts pose-fit-compression. **Is there ANY public competitor doing pose-fit
compression?** If so, the kill verdict is wrong.

**Counter**: No public leaderboard entry has surfaced a pose-fit lane. The
implicit market signal supports the kill verdict. Quantizr explicitly said
"sub 0.30 is possible just by sweeping conv dims" — pose-fit was never on his
list.

**Status**: NO ISSUE. Market signal aligns.

### Hotz (radical simplicity)

**Question**: The "fp16 cast" alternative was floated in the Hotz section
(7.0 KB raw, zero distortion). **Why isn't this commit ALSO landing the fp16
cast as Lane PFP16?** The kill memo says "that is a separate proposal" but
the parent agent might never get back to it.

**Status**: REAL OBSERVATION but OUT OF SCOPE for Lane GP v4. The PFP16 lane
is one-line in any archive build script and should be filed as a follow-up
task. The Lane GP v4 design memo already documents this as
"Outstanding follow-ups #2".

### Other observations

- The kill memo conflates "Lane GP v4 design" with "Lane GP class kill". This
  is intentional (the design exercise CONCLUDES the kill), but the framing
  is potentially confusing for future agents who only read the title. Should
  add a header note "Outcome: KILL — no Phase 2-7 implementation".
  → NOT A BUG; cosmetic. Will note for Round 2.

## Issues found

1. **CRITICAL** (Contrarian Q2): Check 91 scans only `experiments/fit_pose_*.py`,
   not `src/tac/pose_*_fit.py` / `src/tac/pose_*_basis.py`. A future agent
   bypassing this check by placing the basis-fit module under `src/tac/` would
   not trigger the gate.

## Round 1 verdict

**1 CRITICAL issue. Counter resets to 0/3.** Fix Issue 1, then re-run Round 2.
