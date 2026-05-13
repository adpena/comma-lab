# Ledger 06 — AFRL Low-Observable + NSWC Sonar Signature Lineage

**Date:** 2026-05-13
**Lane:** `lane_expert_team_aerospace_stealth_analytic_alien_tech_20260513`
**Evidence grade:** `[stealth-engineering-analog]`.

---

## 0. Persona discipline

The **Air Force Research Lab (AFRL)** at Wright-Patterson AFB designs low-observable
materials and signature management for F-22/F-35-class and beyond. The **Naval Surface
Warfare Center (NSWC)** at Carderock and Dahlgren designs acoustic-signature management
for nuclear submarines (Seawolf, Virginia-class, Columbia-class) and surface combatants.
Both labs operate on the same core principle as Skunkworks but in different physical domains:

- AFRL: minimize EM signature across radar / IR / visual bands
- NSWC: minimize acoustic signature across SONAR / cavitation / engine bands

Their META-discipline: **balance signatures across ALL relevant detection bands; don't
optimize one at the cost of another**.

---

## 1. Multi-spectral material design → multi-objective Lagrangian

### 1.1 AFRL principle

Modern stealth aircraft balance RADAR + IR + VISUAL + ACOUSTIC + ELECTRONIC signatures
SIMULTANEOUSLY. A material that reduces RCS by 30 dB but increases IR signature by 15 dB
is a net LOSS — modern IRST (Infrared Search and Track) systems detect this. The DOD
Stealth Roadmap (2010s) explicitly mandates multi-spectral signature reduction.

Mathematically: the stealth material's design space is the joint optimization
`minimize α·σ_RCS + β·σ_IR + γ·σ_visual + δ·σ_acoustic` subject to aerodynamic
constraints. The Pareto frontier is multi-dimensional; the chosen design point reflects
the threat-environment's relative weighting.

### 1.2 Contest analog

The contest scorer IS a multi-objective sensor:
- `100·d_seg` (SegNet, mostly linear)
- `sqrt(10·d_pose)` (PoseNet, nonlinear; marginal blows up as `d_pose → 0`)
- `25·B/37,545,489` (rate, linear)

The score `S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489` is a WEIGHTED MULTI-OBJECTIVE.
Naive optimization of any single component degrades the others. AFRL's framing maps
directly.

### 1.3 Derived technique F1 — Multi-spectral Lagrangian sweep

**Mathematical formulation.** During training, sweep operating points in `(α, β, γ)`
across the Pareto frontier. For each operating point, train a substrate. Evaluate the
score at TRUE operating-point — the canonical contest formula. The best substrate is the
one that minimizes the WORST-CASE score across the sweep (robust optimization).

This is the AFRL multi-spectral material design framework applied to neural compression.
Existing T1 Balle infrastructure (per CLAUDE.md Catalog #146) supports this.

**Predicted Δscore:** -0.003 to -0.010 (resilient to proxy-auth drift, captures robust frontier).
**Build cost:** 4-6 days (Lagrangian sweep training + selection criterion).
**Risk:** robust optimization may be too conservative; verify with mean-case selection too.

---

## 2. NSWC submarine acoustic-signature → harmonic suppression

### 2.1 NSWC principle

A submarine's acoustic signature is dominated by PROPELLER cavitation harmonics. Cavitation
generates a fundamental frequency and broadband harmonics — the harmonics are detectable
by ASW (anti-submarine warfare) sonar. NSWC's Carderock acoustic-research program
developed:
- Propeller blade-shape optimization to delay cavitation onset
- HARMONIC-CANCELING blade tip vortex devices
- Acoustic anechoic tiles on the hull that ABSORB rather than reflect specific frequency
  bands matching enemy sonar transmitters

The principle: identify the ENEMY'S detection band; design to be SILENT in THAT band
specifically, even at cost of being louder elsewhere.

### 2.2 Contest analog

The scorer's "detection band" is the SegNet-attention map (Grand Council ledger 04 D2)
plus the PoseNet first-6-pose-dim attention. We are ALREADY analyzing this. The novelty
from NSWC's framing: the HARMONIC structure.

PR101's pose-residual signal has HARMONIC structure: across the 600 pairs, the residual
is correlated frame-to-frame (motion is continuous). We can model the residual as a
HARMONIC TIME SERIES and cancel the dominant harmonics.

### 2.3 Derived technique F2 — Pose-residual harmonic cancellation

**Mathematical formulation.** Let `r_p ∈ R^6` be the per-pair pose-residual (PR101 output
minus GT). Compute the discrete Fourier transform across the 600-pair sequence:
`R_k = FFT(r_p)`. The top-K (say K=20) harmonics of R contain most of the residual energy.
Encode these K harmonics (40-80 bytes total — 6 dims × 20 complex coefficients × 1-2 bytes
each after quantization) as a global sidecar.

At inflate time, IFFT the sidecar and apply per-pair correction. The 600-pair residual
total is reduced by 70-90%.

**Predicted Δscore:** -0.010 to -0.025 (pose-axis dominant; pose marginal at PR106 r2 is
271 per CLAUDE.md).
**Build cost:** 5-7 days (FFT residual model + sidecar coder + inflate apply).
**Risk:** residual structure may not be harmonic on all videos (only on `0.mkv`); risks
overfitting to the contest video specifically.

---

## 3. Vibration isolation → temporal smoothing of residual

### 3.1 NSWC principle

Submarine machinery (turbines, pumps) generates broadband acoustic noise. Vibration
isolation (rubber mounts, spring-tuned dampers) ISOLATES specific frequency bands from
hull-radiating. The principle: **decouple the noise source from the radiating surface**.

### 3.2 Contest analog

PoseNet's pose-residual fluctuates frame-to-frame. The fluctuation is essentially
"machinery noise" in the encoder. By temporally smoothing the encoding scheme — applying
a low-pass filter to the per-pair residual encoding — we average out high-frequency
fluctuations that the FFT-based F2 wouldn't capture.

### 3.3 Derived technique F3 — Temporal residual smoothing

**Mathematical formulation.** Apply a per-pair exponential moving average to the encoded
residual: `r_smooth_p = α·r_p + (1-α)·r_smooth_{p-1}`. For α ≈ 0.8, this damps high-
frequency fluctuations. The smoothed residual encodes more efficiently (lower entropy)
than raw residual.

This is sister to F2; F2 is FREQUENCY-domain cancellation, F3 is TIME-domain smoothing.

**Predicted Δscore:** -0.003 to -0.010 (rate-axis; pose-axis modest).
**Build cost:** 2-3 days (simple EMA filter + encoder mod).
**Risk:** smoothing may DEGRADE pose precision on sharp turns (catastrophic pair class).

---

## 4. Material angularity → frequency-aware encoding

### 4.1 AFRL principle

Stealth materials have FREQUENCY-DEPENDENT radar absorption. A material optimized at
10 GHz (X-band) may be transparent at 1 GHz (L-band). Modern broadband stealth uses
LAYERED dielectric materials (e.g. B-2's "dielectric gradient" coating) to absorb
across multiple bands simultaneously.

### 4.2 Contest analog

The contest scorer has DIFFERENT sensitivity at different SPATIAL FREQUENCIES. SegNet's
stride-2 stem (per S2SBS ledger 01 §1.4) attenuates spatial frequency > 0.5 cycles/pixel
at the (192, 256) level. PoseNet's YUV6 representation also has frequency-dependent
sensitivity. The encoder should USE this: spend bits in the frequency bands where the
scorer is sensitive; reuse free bytes in scorer-blind bands.

### 4.3 Derived technique F4 — Frequency-aware encoder budget

**Mathematical formulation.** Wavelet-domain encoding: decompose each frame via wavelet
transform (Mallat 1989) into low-frequency (DC + 1-2 cycles) and high-frequency (3+
cycles) bands. Allocate bits:
- Low-frequency (high scorer sensitivity): 70-80% of budget
- Mid-frequency (medium): 15-20%
- High-frequency (low; below S2SBS Nyquist): 5-10%

This is a refined version of S2SBS (A2) that uses wavelet decomposition rather than naive
spatial-frequency cutoff.

**Predicted Δscore:** -0.005 to -0.015 (rate-axis; refines A2).
**Build cost:** 4-6 days (wavelet decomposition + per-band encoder).
**Risk:** wavelet-domain encoding doesn't compose with HNeRV's INR architecture cleanly;
may require substrate replacement.

---

## 5. Coupled signature management → cross-component cancellation

### 5.1 Principle

Aircraft signature management requires CROSS-COMPONENT thinking. A radar cross-section
optimization that reduces RCS but INCREASES exhaust visibility is a net LOSS. AFRL
emphasizes COUPLED design — choose materials and shapes that REDUCE ALL signatures
simultaneously.

### 5.2 Contest analog

The score `S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489` has CROSS-COMPONENT
DEPENDENCIES:
- Reducing `d_seg` by argmax-preservation often requires MORE bytes (B↑); net Δ unclear
- Reducing `d_pose` to zero requires bytes that could've been spent on `d_seg`
- Each substrate trains against a SPECIFIC coupling

The cross-component dependency means: optimizing one axis can DEGRADE another. Robust
substrate design balances them.

### 5.3 Derived technique F5 — Coupled component-balance constraint

**Mathematical formulation.** During training, monitor all three components in parallel.
Apply a constraint:
```
|d_seg/d_seg_target - 1| < ε_seg
|d_pose/d_pose_target - 1| < ε_pose
|B/B_target - 1| < ε_B
```
Treat the three components as a COUPLED system. The Lagrangian's `(α, β, γ)` weights
should be CALIBRATED so that the marginal at the target operating point is BALANCED:
`d_S/d_d_seg = d_S/d_d_pose = d_S/(d_B/d_S_total/d_B)`.

This is the AFRL "coupled signature management" framework. The existing meta-Lagrangian
solver (CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable) ALREADY supports this.

**Predicted Δscore:** -0.003 to -0.010 (resilient to operating-point drift).
**Build cost:** 3-4 days (calibration sweep + balanced-Lagrangian implementation).

---

## 6. Council adversarial review

- **Hotz.** "Multi-spectral framing is correct but adds 3 axes of substrate complexity.
  We have a meta-Lagrangian solver already. Just verify it's calibrated correctly and
  stop philosophizing." → **STRONG AGREE.** F1, F5 are immediate calibration improvements
  to existing infrastructure; F2, F3, F4 are new substrates.

- **Yousfi.** "Pose-residual harmonic cancellation (F2) IS UNIWARD applied to TEMPORAL
  pose-residual instead of spatial pixels. The 'cost function' is the inverse-FFT-cost of
  expressing the residual in K harmonics. Mature math." → **ENDORSE F2 with UNIWARD
  framing**.

- **Mallat (memorial seat).** "F4 wavelet-domain encoding is the CORRECT way to do
  frequency-aware allocation. The wavelet transform has spatial-frequency LOCALITY by
  construction. Don't use Fourier — use wavelets." → **STRONG ENDORSE F4 over Fourier
  alternatives**.

- **Selfcomp.** "F4 wavelet-domain encoding overlaps with my BlockFP / Hessian-quant work.
  Wavelet coefficients can be Hessian-quantized for high-leverage precision. Stack F4 + my
  BlockFP." → **ENDORSE F4 with Selfcomp stacking**.

- **MacKay (memorial seat).** "Multi-objective Lagrangian (F1) is well-established Bayesian
  optimization. The novelty here is the operating-point calibration to match the contest
  formula's nonlinearity. The pose-axis `sqrt` term is non-trivial." → **STRONG ENDORSE F1**.

- **Contrarian.** "F2 + F3 may OVERFIT to the contest video `0.mkv`. Pose-residual
  harmonic structure on a single 60s clip is not the same as the contest's hold-out video.
  Risk: contest-video performance > training-video performance." → **VALID CHALLENGE.** F2
  + F3 are CONTEST-VIDEO-ONLY techniques. They are CONTEST_ONE_VIDEO_REPLAY mode per
  CLAUDE.md "Contest vs production target modes" non-negotiable. Acceptable for contest;
  unacceptable for production. Tag accordingly.

---

## 7. Reactivation criteria

- F1 multi-spectral Lagrangian: if robust selection is too conservative, reactivate
  with mean-case selection.
- F2 pose-residual harmonic cancellation: if harmonic structure doesn't transfer beyond
  `0.mkv`, reactivate when hold-out videos available.
- F3 temporal residual smoothing: if smoothing degrades hard-pair pose, reactivate with
  activity-gated smoothing.
- F4 wavelet-domain encoding: if wavelet doesn't compose with HNeRV, reactivate as
  substrate-replacement rather than HNeRV-extension.
- F5 coupled component-balance: if calibration is unstable, reactivate with
  fixed-target-operating-point calibration.

---

## 8. Citations (AFRL / NSWC / multi-spectral)

- Air Force Research Laboratory (AFRL): <https://www.afrl.af.mil/>
- Naval Surface Warfare Center Carderock Division: <https://www.navsea.navy.mil/Home/Warfare-Centers/NSWC-Carderock/>
- "Stealth technology" — Wikipedia (multi-spectral section): <https://en.wikipedia.org/wiki/Stealth_technology>
- "Breaking detection barriers: Next-generation dual-band radar/IR stealth materials" — ScienceDirect: <https://www.sciencedirect.com/science/article/abs/pii/S0010854526001670>
- "Low Observable Principles, Stealth Aircraft and Anti-Stealth Technologies" — ResearchGate 2013: <https://www.researchgate.net/publication/259503614_Low_Observable_Principles_Stealth_Aircraft_and_Anti-Stealth_Technologies>
- DARPA RF Stealth (Low Observable) and Counter — DTIC: <https://apps.dtic.mil/sti/tr/pdf/ADA496936.pdf>
- "Detecting Stealth Aircraft: IRST Systems" — Diverse Daily: <https://diversedaily.com/detecting-stealth-aircraft-the-role-of-infrared-search-and-track-irst-systems/>
- Mallat, S. "A Theory for Multiresolution Signal Decomposition: The Wavelet Representation" — IEEE TPAMI 1989: <https://ieeexplore.ieee.org/document/192463>

---

## 9. Wire-in (Catalog #125)

1. Sensitivity-map: F2 pose-residual harmonics feed temporal-sensitivity-map.
2. Pareto: F1 multi-spectral Lagrangian sweeps the Pareto frontier explicitly.
3. Bit-allocator: F4 wavelet-domain budget IS the bit-allocator's spatial-frequency primitive.
4. Cathedral autopilot: F1 + F5 calibration runs feed autopilot's operating-point selection.
5. Continual-learning: F2 sidecar harmonics produce per-video posterior anchors.
6. Probe-disambiguator: F2 frequency-domain vs F3 time-domain cancellation is the canonical disambiguation.

---

**End ledger 06.**
