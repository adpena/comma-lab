# Horizon-band-localized d_seg lever on the 0.19110 CPU frontier — measured verdict

- **Date:** 2026-06-23
- **Subagent:** `horizon_dseg_lever_20260623`
- **Axis:** `[contest-CPU advisory]` — single video `0.mkv` (1200 frames = 600 pairs)
  reproduces the contest 600-sample eval locally. Authority = exact frozen CPU SegNet
  argmax-disagreement through the exact preprocess. NOT a proxy.
- **Pointer:** UNMOVED at 0.19110. Any win still needs byte-close + exact eval.
- **All score/break-even math via `tac.contest_score`** (canonical helper, NOT hand-rolled
  — Catalog #391 compliance). rate slope = `rate_term(1)/100 = 6.659e-9` d_seg per byte.
- **Builds on** `independent_dseg_bets_frontier_20260623.md` (the prior probe that localized
  97.8% of d_seg to the horizon band and ruled out hood/road/selector). This probe tests the
  operator's NEW stark-edge-under-resolution hypothesis + the untested horizon-band
  flip-RESIDUAL sidecar (Lever-D localized).

## Authority / setup (NO FAKE)

Frontier archive `experiments/results/pr110_payload_entropy_recode_20260610/submission_dir/archive.zip`
sha256 `b46897267…`, 177169 bytes, report.txt d_seg 0.00055978 / d_pose 0.00002942 / score 0.19.
Reused the cached frozen-SegNet argmaps `(gt, comp)` (600 pairs, 384×512) from the prior probe's
prepare stage — validated: cached d_seg = **0.00055989** vs report 0.00055978 (Δ=1e-7, exact-scorer
faithful). GT decoded via `upstream/frame_utils.yuv420_to_rgb` (BT.601, never PyAV rgb24).
Tool: `experiments/probe_horizon_band_dseg_lever.py` (stages `margin`, `sidecar`).

## TASK A — how comma/openpilot handle the horizon

- **comma10k 5-class SegNet** (`commaai/comma10k`): 0=Road, 1=Lane Markings, 2=**Undrivable
  (includes sky)**, 3=Movable, 4=My Car. Classes are organized by motion (scene-fixed: road/lane/
  undrivable; self-moving: movable; ego: my car). This DIRECTLY explains our measured horizon flips:
  dominant flip pairs in the band are gt=1↔comp=0 (Road↔LaneMarking, 54% combined) and
  gt=0↔comp=2 (Road↔Undrivable/sky, 17%). The horizon is the road→distant-undrivable/sky and
  road→lane-marking semantic transition.
- **openpilot calibration** (`common/transformations`): images are warped into a "calibrated frame"
  aligned with the car frame in pitch+yaw. The **horizon row is the vanishing point**, a deterministic
  function of pitch/yaw extrinsics + intrinsics K. Our segment is the known comma2k19 RAV4 highway,
  `_neo_config` K with `camera_fl=910` and principal point cy≈437. On a flat highway at ~level pitch
  the calibration (hence the horizon row) is stable across the clip. **Measured tie-in:** our d_seg
  flip peak is at SegNet rows 185-195 → camera rows **421-444**, straddling cy≈437. The d_seg hotspot
  IS the calibrated horizon.
- **Is the horizon a near-zero-byte prior?** YES for the geometric LINE (cy + pitch ⇒ a known row,
  ~0 bytes). NO for the d_seg WHAT: the flips are at the *content-dependent semantic* boundary
  (distant road vs sky/lane), which moves per-frame with the scene, not at a static line. Comma's
  driving model represents road-edges/lane-lines PARAMETRICALLY (supercombo: 2 road edges + lane lines
  as mean/std polylines over 33 future timestamps, 0–192 m) — a cheap "where" — but that parametric
  boundary is the *driving-path* geometry, NOT the per-pixel SegNet argmax the contest scores. The
  near-0-byte geometric horizon prior cannot supply the per-pixel class corrections d_seg needs.

## TASK B — stark-edge-under-resolution mechanism (measured, exact SegNet + GT)

`stage margin` on 100 pairs (real GT decode + real frozen SegNet logits on the comp last-frame):

| signal | value | reading |
|---|---|---|
| horizon-band luma recon err / whole-frame | **1.14×** | modest elevation, not a dramatic spike |
| horizon-band chroma recon err / whole-frame | 1.03× | ~flat |
| local luma-err at camera row 410 vs row 430 | **59.5 vs 26.3** | a real LOCAL spike AT the sky/ground edge |
| global luma-err peak row | 670 (hood/dash), val 89.5 | biggest recon error is at the hood — **0 d_seg there** |
| GT-vs-comp luma-edge row shift (median) | 33 camera-rows | edge localization is noisy/shifted (multi-object band) |
| **flip-pixel SegNet margin (top1−top2)** | **mean 0.102, median 0.075** | knife-edge |
| non-flip margin | 0.981 | confidently decided |
| flip pixels with margin < 0.1 / < 0.2 | **61.8% / 86.6%** | the flips sit on the decision boundary |

**Mechanism verdict — PARTIALLY confirmed, refined.** The operator's intuition is right at the
*decision* level: the d_seg flips are overwhelmingly **shallow-margin** pixels (86.6% below 0.2 vs
non-flip 0.98) where SegNet is barely decided, and there IS a local luma-error spike at the horizon
edge (row 410). BUT the recon error is NOT dramatically concentrated at the horizon band
(only 1.14× frame mean), and the largest recon errors (hood, row 670) produce ZERO d_seg — proving
recon-error magnitude alone does not drive d_seg; **proximity to a shallow SegNet decision boundary**
does. So "stark edge under-resolved" is not the full story: the binding factor is that the distant
road/sky/lane boundary is *both* a high-contrast edge *and* a shallow-margin SegNet boundary, so a
small recon error there tips the argmax. (The at-flip recon-error refinement could not be re-run: the
3.66 GB `0.raw` was auto-cleaned mid-session; the saved per-row curves + margin distribution stand.)

This points the only real fix at the **base reconstruction fidelity / score-aware training at the
horizon band** (a trainer-side lever), consistent with the prior probe's structural conclusion.

## TASK C — horizon-band flip-RESIDUAL sidecar (Lever-D localized) → **NO-GO (decisive)**

`stage sidecar` on 600 pairs. The sidecar stores, per pair, the horizon-band positions where comp≠gt
plus the corrected (GT) class — its ORACLE Δd_seg = "force comp argmax → GT argmax in the band" (the
maximum a perfect localized correction can buy). Byte cost = real `zlib`-coded position-bitmap +
class-symbol stream (conservative vs arithmetic coding; entropy floor reported for reference). Net
score via `compute_contest_score(new_dseg, d_pose, 177169+sidecar_bytes)`:

| band (SegNet rows) | oracle Δd_seg | sidecar bytes (zlib) | Δd_seg / byte | net ΔS | verdict |
|---|---|---:|---:|---:|---|
| peak 180–200 | −2.30e-4 (−41%) | 49,381 | **−4.65e-9** | **+0.0099** | NO-GO |
| tight 160–220 | −4.41e-4 (−79%) | 101,033 | −4.36e-9 | +0.0232 | NO-GO |
| horizon 96–288 | −5.48e-4 (−98%) | 146,766 | −3.73e-9 | +0.0430 | NO-GO |
| all-residual 96–345 | −5.60e-4 (−100%) | 153,794 | −3.64e-9 | +0.0464 | NO-GO |

**The unit economics are the decisive killer.** The break-even efficiency a sidecar must beat is the
rate slope **6.659e-9 d_seg/byte** (canonical `rate_term`). Even the ORACLE (perfect, uses GT)
achieves only **−4.65e-9 d_seg/byte at best** (peak band) — i.e. **0.70× of break-even**. A flip is
worth ~3e-8 d_seg but each corrected flip costs ~1.8 zlib bytes (position+class), and there are
27,113 flips even in the 20-row peak band. So the sidecar is intrinsically ~30–45% too byte-expensive
per flip, and EVERY band raises the score (best case +0.0099). Tightening the band raises efficiency
(fewer pixels), but even the tightest peak band cannot cross break-even. Storing the residual is
strictly dominated by NOT storing it. Honest negative — even a perfect oracle correction does not
move the pointer.

- The "calibrated-horizon-line + step prior" variant is also NO-GO: the horizon LINE is near-0-byte,
  but the prior probe's #138 geo gate already proved a data-independent prior at the horizon is
  wrong ~half the time (mixed-class, 0-byte geometric prior = +0.100). The cheap "where" exists; the
  "what" (per-pixel class) is the byte-expensive part this sidecar measures, and it doesn't pay.

## Ranked verdict

1. **Horizon-band flip-residual sidecar — NO-GO (decisive).** Oracle Δd_seg/byte −4.65e-9 (best) <
   the 6.659e-9 break-even rate slope; every band net ΔS > 0. The d_seg is too diffuse (27k+ flips)
   and each correction symbol costs ~1.8 bytes for ~3e-8 d_seg.
2. **Calibrated-horizon geometric prior — NO-GO.** The line is near-0-byte but data-independent class
   priors at the horizon are wrong ~50% (prior probe #138: +0.100). Cheap "where", no cheap "what".

**Single most promising path: NONE on the $0 frontier-side.** Confirmed both the prior probe's
structural finding (diffuse content-dependent boundary noise) AND added the decisive unit-economics
reason: even a *perfect* localized correction is byte-dominated. The horizon d_seg is real and
shallow-margin, so the only lever that pays is **base reconstruction fidelity / score-aware training
at the horizon band** (a trainer-side lever — e.g. horizon-band-weighted recon loss or more decoder
capacity targeting camera rows ~421–444), NOT a $0 sidecar/transform. This rules out the
highest-prior cheap path with an exact-scorer measurement.

## 6-hook wire-in (Catalog #125)

- #1 sensitivity-map: ACTIVE — flip pixels are shallow-margin (86.6% < 0.2 margin) at camera rows
  421–444 (= calibrated horizon cy≈437); a reusable seg-sensitivity prior for the trainer/bit-allocator.
- #2 Pareto: N/A — no admitted candidate (all NO-GO).
- #3 bit-allocator: ACTIVE (advisory) — "d_seg lives at the calibrated horizon, shallow-margin"
  routes any future horizon-band-weighted recon/capacity spend.
- #4 cathedral autopilot dispatch: N/A — advisory, non-promotable, no archive change.
- #5 continual-learning posterior: N/A — `[contest-CPU advisory]`, non-promotable.
- #6 probe-disambiguator: ACTIVE — `probe_horizon_band_dseg_lever.py` (margin + sidecar stages)
  is the disambiguator for the stark-edge + sidecar hypotheses.

Mission contribution: `frontier_protecting` — rules out the highest-prior $0 d_seg path
(horizon-band flip-residual sidecar) with decisive unit-economics, and redirects the d_seg attack to
the trainer side. Pointer UNMOVED 0.19110.
