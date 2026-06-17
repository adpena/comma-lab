# BLIND-SPOT PROBE C — measurement trust on d_seg / d_pose (2026-06-17)

**Authority:** `[contest-CPU advisory]` NON-PROMOTABLE. All numbers are exact
contest metrics from the vendored authority path (`score.evaluate_decoder`
streaming GT via `frame_utils.yuv420_to_rgb`, `score.compute_score`) on CPU —
NOT a score claim; a measurement-trust diagnostic. READ-ONLY on the running
trajectory + basin (checkpoints never mutated).

**Question:** is the d_seg "wall" the 5-day canonical run is fighting REAL, or
another measurement artifact (the capstone EMA-shadow-lag bug class that faked a
d_seg 0.505 plateau on the MLX capstone)?

**Tools (this lane):**
- `experiments/probe_c_measurement_trust_live_vs_shadow.py` — loads LIVE vs
  EMA-SHADOW decoder+latents from a checkpoint, computes exact d_seg/d_pose for
  BOTH on the SAME frames; FiLM-carrier isolation for ARM B.
- `experiments/probe_c_ce_power_law_trust.py` — power-law fit + window-stability
  on the d_seg trajectory.
- JSON: `.omx/research/probe_c_measurement_trust_live_vs_shadow_20260617.json`.

---

## VERDICT — the d_seg wall is REAL; the d_pose eval is the artifact-prone axis

**d_seg measurement is TRUSTED.** The 5-day run's d_seg signal is real, not a
shadow-lag artifact. The capstone bug (EMA shadow frozen near init faking a
plateau) is NOT present in the torch_vehicle runs. The "wall" is a genuine,
slow power-law — which is the *bad* news for the projection: even the optimistic
canonical fit lands ~2.5× ABOVE the sub-0.15 d_seg target. The d_seg axis is
trustworthy enough to act on; the *projection off it* is what should be
re-examined, not the measurement.

**d_pose measurement is the noisy / artifact-prone axis** — but ONLY in the
FiLM-carrier arm (ARM B), and the noise source is the **FiLM pose carrier**, not
the EMA shadow, not MPS. The simpler basin path (no FiLM) has a STABLE, trusted
d_pose.

---

## AUDIT 1 — LIVE vs EMA-SHADOW d_seg gap (the capstone bug class)

The torch_vehicle `exact_eval` (BEST tracker) renders the **EMA shadow**
(`ema_decoder`, `ema_latents`). Both audited runs ran with `ema_warmup=OFF`,
constant `ema_decay=0.999` — the exact config that froze the capstone MLX shadow.
BUT: the capstone froze because it was SHORT (6 steps/epoch, ~240 steps/stage <<
EMA time constant 1/(1-0.999)=1000 steps). The torch_vehicle runs are LONG: 600
pairs / bs 8 ≈ 75 steps/epoch × 2120–3072 epochs ≈ 160k–230k steps >> 1000-step
time constant. The shadow is fully warmed → NO freeze.

Direct live-vs-shadow exact d_seg (same frames, CPU authority):

| arm | d_seg LIVE | d_seg SHADOW | gap (live − shadow) |
|-----|-----------|--------------|---------------------|
| basin (n=2 smoke) | 0.002566 | 0.002505 | +6.1e-5 (~2.4%, shadow ≈ live) |
| armb  (n=2 smoke) | 0.002881 | 0.002520 | +3.6e-4 (shadow ≈/slightly better) |

The shadow d_seg ≈ live d_seg (shadow marginally *better*, the healthy smoothed
case) — the OPPOSITE of the capstone (shadow +0.466 ABOVE live). **No lag.**
The full n=64 confirmation is in the JSON (`live_vs_shadow_d_seg_gap`).

Corroboration from the trajectory itself: the basin's shadow-eval d_seg is a
clean monotone descent 0.0717 → 0.0026 over 2120 epochs with NO frozen-near-init
plateau. A capstone-style frozen shadow would read a near-constant ~0.5; it does
not.

---

## AUDIT 2 — pose eval-path variance root-cause: it is the FiLM carrier

Late-window (ep≥1000) trajectory noise, by axis:

| arm | d_seg CV | d_pose CV | d_pose mean | carrier |
|-----|---------:|----------:|------------:|---------|
| basin | 0.041 | **0.17** | 0.00047 | none (latent+decoder) |
| armb  | 0.144 | **0.97** | 0.01675 | FiLM v2 (rgb_0 head) |

Pose variance is ~6× higher in ARM B and lives entirely in d_pose. d_seg is
stable in both. The basin (no FiLM) has BOTH a lower mean d_pose (0.00047 vs
0.0168) AND far more stable d_pose (CV 0.17 vs 0.97).

**FiLM-carrier isolation (ARM B, same state):**
- FiLM-ON  d_pose = 0.00017
- FiLM-OFF d_pose = 2.27 (rgb_0 = vendored clean path)
- d_seg is EXACTLY invariant to FiLM (Δ = 0.0) — the v2 rgb_1-clean decoupling works.

The FiLM carrier does ~13,000× of the pose work; without it pose collapses. The
carrier is therefore the lever the whole d_pose axis hinges on — and it is the
source of the eval variance. The noise is NOT:
- (a) the EMA shadow — basin shares the identical shadow config and is pose-stable;
- (b) MPS — eval is CPU-authority in both; ARM B trained on CPU and is still noisy;
- (d) — ruled out.
It IS (c) the FiLM-pose-carrier reconstruction (+ its uint8/fp16 byte-close
round-trip: `encode_pose_section` quantizes stored_pose to uint8/254 per dim).

Why the carrier is noisy: the EMA shadow of the FiLM params + the rgb_0 head
wanders between eval snapshots; the rgb_0 pose head is sensitive, so adjacent
eval epochs swing d_pose 0.00065 ↔ 0.0628 (96×) even though the SHADOW d_seg
(rgb_1, FiLM-clean) and the train pose_mse (~0.0003) are stable.

**Recommended fix (highest-trust → lowest-cost):**
1. **Trust the basin pose path over the FiLM carrier for the score that picks
   BEST.** The basin reaches d_pose=0.00034 with CV 0.17 and NO carrier. The
   FiLM carrier's mean d_pose (0.0168) is WORSE than the basin's by ~35×; it is
   not earning its noise. Re-examine whether ARM B needs the FiLM carrier at all.
2. If the carrier is kept: **average the d_pose over the last K eval snapshots**
   (or evaluate an EMA-of-EMA / multi-snapshot mean) so a single lucky/unlucky
   carrier render does not pick BEST. ARM B's BEST (ep2700, d_pose=0.00065) was a
   lucky low-pose draw amid neighbors at 0.006–0.063 — BEST-by-single-noisy-pose
   is itself a measurement artifact.
3. Apply `ema_warmup` is irrelevant here (shadow already warmed); the carrier
   variance is the real target, not the shadow.

---

## AUDIT 3 — CE power-law trust on the LIVE d_seg trajectory

Fit d_seg = A·ep^(−p) on the shadow-eval trajectory (validated as ≈live by
Audit 1), warmup-skip ep≤40:

| arm | full p | r² | early-half p | late-half p | late r² |
|-----|-------:|---:|-------------:|------------:|--------:|
| basin | 0.235 | 0.979 | 0.276 | 0.192 | 0.995 |
| armb  | 0.245 | 0.927 | 0.170 | 0.431 | 0.989 |

**The basin "flattening tail" is REAL power-law decay, not a shadow floor.** A
shadow-lag floor would show the late exponent collapsing toward 0 with degrading
fit; instead the basin late window has r²=0.995 and p=0.19 — still cleanly
descending, just slowly. (ARM B's late p RISES because its d_seg actually
accelerated late after a sloppy early phase; its lower full r²=0.93 reflects the
pose-noise contamination of its score, not d_seg — d_seg itself is clean.)

**Projection (the part to distrust, not the measurement):**
- basin own fit (p=0.235): d_seg(50k) ≈ 0.0012 (~3.7× above sub-0.15 target 0.000322)
- canonical fit 0.0367·ep^−0.351: d_seg(50k) ≈ 0.00082 (~2.5× above target)
- the basin's late-window p=0.19 (slower than canonical 0.351) → 50k is even
  FURTHER from target than the canonical extrapolation suggests.

So: the d_seg wall is a trustworthy, slow power law. Epochs alone do not reach
the sub-0.15 d_seg target within 50k. This corroborates the symposium verdict
(d_seg/capacity-infeasible under epochs alone) — and it is REAL, not an artifact.

---

## TRUSTED / NOT-TRUSTED summary

| measurement | verdict |
|-------------|---------|
| d_seg (both arms, shadow eval) | **TRUSTED** — shadow ≈ live, clean power law, no freeze |
| d_pose (basin, no FiLM) | **TRUSTED** — CV 0.17, stable, mean 0.00047 |
| d_pose (ARM B, FiLM carrier) | **NOT TRUSTED** — CV 0.97, carrier-noise; BEST-by-single-pose is itself an artifact |
| CE power-law fit on d_seg | **TRUSTED as a fit; the 50k PROJECTION is the risk** — d_seg lands ~2.5–3.7× above the sub-0.15 target |

**De-risk outcome:** the 5-day run is NOT fighting a phantom d_seg wall (good —
the measurement is honest). But the wall is real and slow, so a 5-day epochs-only
d_seg descent will land ABOVE the sub-0.15 d_seg target — the projection, not the
measurement, is where the run is mis-calibrated. Redirect: trust d_seg, attack it
with the byte-neutral d_seg-aware structural levers (taper realloc / KD warm-start)
rather than epochs; and stop letting the noisy FiLM-carrier d_pose pick BEST
(average snapshots or drop the carrier in favor of the basin pose path).

## Wire-in (6-hook)
1. sensitivity-map: N/A (diagnostic, no per-byte map produced).
2. Pareto: informs the d_seg-vs-rate constraint (d_seg floor is slow → byte
   realloc dominates epochs).
3. bit-allocator: N/A.
4. cathedral autopilot: N/A (research diagnostic).
5. continual-learning posterior: this memo + JSON are the anchor; corroborates
   the symposium d_seg-infeasible-under-epochs verdict with a measurement-trust
   proof.
6. probe-disambiguator: this IS the disambiguator (real-wall vs shadow-artifact;
   carrier vs shadow vs MPS for pose). Verdict: real wall + carrier pose noise.
