# Ledger 02 — Area 51 / Black-Project Signature & Deception Lineage

**Date:** 2026-05-13
**Lane:** `lane_expert_team_aerospace_stealth_analytic_alien_tech_20260513`
**Evidence grade:** `[classified-domain-derivation]` and `[stealth-engineering-analog]`.
**Score claim:** false. `promotion_eligible:` false. `ready_for_exact_eval_dispatch:` false. `research_only:` true.

---

## 0. Persona discipline

Area 51 / Groom Lake / Tonopah Test Range hosted classified flight test for HAVE BLUE
(F-117 prototype, 1977), Tacit Blue (1982 LO testbed), RQ-170 Sentinel (Beast of Kandahar),
the A-12 Oxcart, and the alleged RQ-180 / Aurora / TR-3B. The operational discipline at
these ranges is **signature management against KNOWN adversary SIGINT**: classified flight
test must happen without alerting Soviet RORSAT, Cold War HUMINT at McCarran approach
corridors, or the public's media presence.

The contest analog: the scorer is a known, fixed sensor (its source code is `upstream/modules.py`).
Our build pipeline must produce archives whose score-emission is precisely characterized in
advance. There is no "we'll measure on submission" — the auth-eval cost is real and burns
GPU. Classified flight test taught the discipline of **measuring signature BEFORE flight**.

---

## 1. Deception / decoy operations analog → false-target encoding

### 1.1 Classified flight-test deception

Tacit Blue (1982) flew 135 missions out of Groom Lake while remaining classified for 14
years. Operational technique: schedule flights during periods of known Soviet RORSAT
unavailability; spread activity across multiple inhabited ranges to mask the LO testbed
within ordinary military traffic; emit chaff and decoys to confuse passive radar.

The deception principle: **make the adversary's sensor see SOMETHING REAL and benign so
it doesn't look for the hidden signature.**

### 1.2 Contest analog (`[classified-domain-derivation]`)

The scorer expects certain frame statistics. If we render frames that exhibit the
"benign" statistics the scorer expects (smooth ego-motion deltas, road-plane geometry,
predictable lane markings), the scorer's PoseNet and SegNet pathways spend their
attention budget on these "decoys" — and we can hide perturbations in adjacent feature
maps.

### 1.3 Derived technique B1 — Decoy-target rendering (Tacit Blue analog)

**Mathematical formulation.** Build a per-pair "expected statistics" model `E[Stats|GT]`
covering: mean pose deltas, mean SegNet class-distribution histogram, mean YUV6 channel
power. Render a baseline `f_b` that matches these expected statistics EXACTLY (per-frame).
Then encode the per-pair DEVIATION `δ = (f_GT - f_b)` rather than the raw frame.

The deviation has lower entropy than the raw frame (predictable mean is removed). Plus,
when the scorer applies its preprocess+forward pipeline, the BASELINE'S forward already
matches the GT-baseline's forward (by construction), so the deviation must only correct
the scorer-visible part.

**Predicted Δscore:** -0.003 to -0.012 (rate-axis only; orthogonal to A1 SABOR which is
seg-axis).
**Build cost:** 4-6 days (statistics modeling + deviation encoder).
**Risk:** decoy-effectiveness depends on how well baseline matches GT statistics; if mismatch
is large, savings shrink.

---

## 2. Multi-range activity-spreading → multi-frame budget spreading

### 2.1 Operational principle

Classified test programs at Groom Lake routinely flew identical aircraft across multiple
inhabited ranges (Edwards, Nellis, Tonopah) to spread "obvious" activity and conceal "real"
activity. SIGINT analysts counting daily sortie patterns from a single base would have
flagged the Groom traffic as anomalous; spreading activity across ranges removed the
anomaly.

### 2.2 Contest analog

PR101's approach concentrates byte-budget on a single architecture (HNeRV-LC decoder).
Spread the budget across MULTIPLE inflate-time renderers: one for "easy" highway pairs,
one for "hard" turn-event pairs, one for "boundary-aware" lane-marking frames. Each
renderer is small (~5-20 KB), but the COMPOSITION operates against the scorer with full
fidelity.

This is the **DARPA Mosaic Warfare** principle applied to compression: many small,
specialized encoders > one large monolithic encoder.

### 2.3 Derived technique B2 — Mosaic encoder swarm (Mosaic Warfare / OFFSET analog)

**Mathematical formulation.** Partition the 600 pairs by behavior class:
- Class E (Easy, ~70-80%): highway cruise; pose delta in narrow band
- Class M (Medium, ~10-20%): mild turns; lane changes
- Class H (Hard, ~5-10%): braking events; tight turns
- Class C (Catastrophic, <2%): occlusions; outlier pairs

For each class, train a specialized renderer that compresses to ~5-10 KB. Inflate-time
dispatch reads a 2-bit class-label sidecar per pair and selects the renderer. Total
archive: 4 renderers × 7 KB avg + 600 × 0.25 B label = 28 KB + 150 B = 28.15 KB
(vs HNeRV ~120 KB).

The byte savings come at a cost: the per-pair distortion may be slightly higher for some
hard pairs, but the MOSAIC achieves better Pareto frontier on `(d_seg, d_pose, B)`.

**Predicted Δscore:** -0.008 to -0.025 (depends on class partition quality).
**Build cost:** 6-9 days (class-partition + 4 specialized renderers + inflate dispatch).
**Risk:** class boundaries may be hard to detect at inflate time; per-pair dispatch label
adds overhead.

---

## 3. SIGINT-clutter analog → frequency-domain byte spreading

### 3.1 Operational principle

When a known transmitter is monitoring a frequency band, classified ELINT operations
spread their emissions across the entire band so the adversary cannot distinguish signal
from noise. This is the spread-spectrum/CDMA principle.

### 3.2 Contest analog

The archive's byte distribution should match a "natural" Brotli-compressed file's byte
distribution. PR101's archive byte histogram is HEAVILY non-uniform (decoder weights cluster
at certain values; latents have predictable distributions). An adversarial scorer (not
ours, but a hypothetical) could detect this anomaly. More importantly: **bytes that follow
a high-entropy uniform distribution compress poorly under Brotli, but contain LOW
information per byte**.

Inverse design: **engineer the archive byte distribution to have HIGH entropy after
Brotli/zstd canonicalization**. This means: don't waste bytes encoding "easy" runs of zeros
that Brotli already compresses; concentrate bytes on Brotli-incompressible-but-scorer-
relevant information.

### 3.3 Derived technique B3 — Brotli-incompressible-aware allocation

**Mathematical formulation.** For each candidate byte position `i` in the archive's
PRE-BROTLI representation, compute the marginal Brotli-rate contribution `r_i = compressed
size after adding byte i - compressed size without byte i`. A byte with `r_i ≈ 0` is
Brotli-incompressible and contributes to entropy efficiency. A byte with `r_i ≈ 8` is
Brotli-redundant and contributes nothing.

Allocate the encoder's bit budget to MAXIMIZE `Σ_i informational(i) - r_i`, where
`informational(i)` measures the byte's contribution to score reduction.

**Predicted Δscore:** -0.002 to -0.008 (rate-axis only; orthogonal to other techniques).
**Build cost:** 2-4 days (Brotli-rate-aware bit-allocator in `tac.composition.registry`).

---

## 4. Have Blue / pre-F-117 prototype → existence-proof discipline

### 4.1 Operational principle

The XST program (Experimental Survivable Testbed, 1976-1977) flew the Lockheed Have Blue
prototype 36 times. Have Blue PROVED that the faceted-geometry mathematical predictions
held in flight. The F-117 production model came AFTER. Operational discipline: **prove the
principle on cheap prototype before committing to full production**.

### 4.2 Contest analog

The φ1 SABOR audit (2026-05-13), the φ2 PAYIC probe (planned), and the φ3 boundary audit
(planned) are the contest analogs of Have Blue. Each is a $0-5 GPU experiment that proves
or refutes a principle BEFORE we build a full $40-100 substrate around it.

### 4.3 Derived technique B4 — Cheap-prototype probe discipline

**Operational rule.** For every new substrate idea (A1, A2, A4, A5, A6, B1, B2, B3 etc.),
build a $0-5 cheap-prototype probe that:
1. Tests the core principle on 10-50 sample pairs
2. Returns a YES/NO/MAYBE verdict within 24h wall clock
3. Has structurally-different failure modes from a full substrate build

If the probe returns NO, the full build is DEFERRED-pending-research. If YES, full build
proceeds. If MAYBE, council review.

**Predicted Δscore:** N/A (process discipline, not a substrate).
**Build cost:** N/A (this is a meta-recommendation).
**Impact:** estimated 30-50% reduction in wasted GPU spend on dead-end substrates.

---

## 5. Council adversarial review

- **Quantizr.** "Decoy targets and mosaic encoders are classical compression
  techniques (zerotree, embedded zerotree wavelet, EZW from 1996). The aerospace framing
  is rhetorical; the math is just rate-distortion." → **REBUT.** The aerospace framing
  brings the MULTI-SPECTRAL discipline (separate budget per sensor, A4 from ledger 01) and
  the OPERATIONAL discipline (Have Blue probe, B4). Classical compression literature does
  NOT systematically separate per-sensor budgets or enforce probe-before-build. The framing
  is operational, not mathematical. **Verdict: ENDORSE B1+B2+B4; B3 is incremental.**

- **Carmack.** "Mosaic encoder swarm is an N+1 problem. Each new renderer adds inflate-time
  dispatch complexity, and inflate.py has a 100-LOC budget. You can't ship 4 renderers in
  100 LOC." → **CHALLENGE PARTIALLY ACCEPTED.** Realistic budget per renderer: 25 LOC
  including class-dispatch logic. 4 renderers = 100 LOC = right at the budget limit. Need
  CARMACK-style code golf: shared encoder backbone + 4 small heads = single dispatch
  parameter, not 4 separate renderers. Revised B2 estimate: -0.006 to -0.020 (down from
  -0.008 to -0.025).

- **Filler (Fridrich's other student; STC expert).** "B3 Brotli-incompressible-aware
  allocation is exactly STC (Syndrome Trellis Coding) — assign each modification a cost,
  optimize total cost s.t. parity-check constraint. STC is mature." → **AGREE.** B3 is
  STC-flavored. Suggest naming the bit-allocator `STCByteOptimizer` and citing Filler 2011.
  **Verdict: STRONG ENDORSE B3 with STC framing.**

- **Hassabis.** "Classified flight test discipline (B4 probe) is exactly DeepMind's
  AlphaFold approach: cheap structure-prediction prior to expensive wet-lab experiments.
  This is the right meta-discipline." → **STRONG ENDORSE B4.** Suggest formalizing B4 in
  CLAUDE.md as a non-negotiable: every substrate build must cite its cheap-probe verdict.

- **Contrarian.** "All these analogies are post-hoc rationalization. The contest is not
  literally a stealth problem; treating it as one is anthropomorphism." → **CHALLENGE
  REJECTED.** The contest IS literally a sensor-output minimization problem; the scorer is
  literally two CNN sensors with specific blindspots; "stealth" is the operational
  vocabulary for sensor-output minimization with engineering discipline. The vocabulary
  is precise, not rhetorical.

---

## 6. Reactivation criteria

- B1 decoy-target rendering: if `E[Stats|GT]` model is unreliable on out-of-distribution
  pairs, reactivate with per-class statistics model.
- B2 mosaic encoder swarm: if class boundaries can't be detected at inflate time within
  100-LOC budget, reactivate with shared-backbone+4-heads architecture.
- B3 STC byte optimizer: if Brotli compression doesn't dominate the archive's bottleneck
  (e.g. archive is < 50 KB and overhead dominates), reactivate when archive grows.
- B4 cheap-probe discipline: this is process, no reactivation needed; failure mode is
  "council doesn't enforce it" and is captured in CLAUDE.md.

---

## 7. Citations (Area 51 / black-project lineage)

- Area 51 (USAF Groom Lake) on Wikipedia: <https://en.wikipedia.org/wiki/Area_51>
- HAVE BLUE program — F-117 prototype: <https://roadrunnersinternationale.com/haveblue.html>
- Tacit Blue / "Whale" — Northrop LO testbed: <https://en.wikipedia.org/wiki/Northrop_Tacit_Blue>
- RQ-170 Sentinel — Wikipedia: <https://en.wikipedia.org/wiki/Lockheed_Martin_RQ-170_Sentinel>
- DARPA OFFSET (Offensive Swarm-Enabled Tactics): <https://www.darpa.mil/research/programs/offensive-swarm-enabled-tactics>
- "Mosaic Warfare: Small and Scalable are Beautiful" — War on the Rocks (2019): <https://warontherocks.com/2019/12/mosaic-warfare-small-and-scalable-are-beautiful/>
- "DARPA's Mosaic Warfare - Multi Domain Ops, But Faster" — Breaking Defense: <https://breakingdefense.com/2019/09/darpas-mosaic-warfare-multi-domain-ops-but-faster/>
- Filler, T., Judas, J., Fridrich, J. "Minimizing Additive Distortion in Steganography Using Syndrome-Trellis Codes" — IEEE TIFS 2011: <https://ieeexplore.ieee.org/document/5740835>
- National Security Archive Area 51 declassification: <https://nsarchive.gwu.edu/briefing-book/intelligence/2013-08-15/cia-finally-acknowledges-existence-area-51>

---

## 8. Wire-in (Catalog #125)

1. Sensitivity-map: B3 STCByteOptimizer feeds per-byte Brotli-rate map into bit-allocator.
2. Pareto: B1 decoy-targets shift the rate-axis Pareto frontier inward.
3. Bit-allocator: B2 mosaic encoder swarm requires shared-backbone+4-heads dispatch primitive.
4. Cathedral autopilot dispatch: B4 cheap-probe discipline IS the dispatch-ordering primitive.
5. Continual-learning posterior: each cheap-probe verdict (YES/NO/MAYBE) feeds an anchor.
6. Probe-disambiguator: B2 mosaic vs single-monolith is the canonical disambiguation;
   probe = run both on 10 sample pairs, compare Pareto frontier.

---

**End ledger 02.**
