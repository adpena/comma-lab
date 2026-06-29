# BUDGET-GATE OVERTURN — EXACT comma2k19 GT pose + sharp render + structured descriptor + dither

**UTC** 2026-06-29T21:57:29Z · **authority** `[macOS advisory / CPU-torch research-signal]` · **pointer UNMOVED 0.19110**
**score_claim** false · **promotable** false · **ready_for_exact_eval_dispatch** false · this is a **MEANS**, not the end.
**Tool** `tools/measure_budget_gate_overturn.py` (new; extends a95b0ad6 clean-canonical + a23062c4 screw + a513372a pose machinery)
**JSON** `experiments/results/budget_gate_overturn_n96_r2/results.json` (n96, window ±2, elapsed 823s)
**Authority** FROZEN CPU-torch SegNet (`load_real_segnet("cpu")` + `measure_segnet_argmax`), NEVER MPS.
**Resolves** the A9 NOT-PESSIMISTIC OVERTURN of a95b0ad6's NEGATIVE budget-gate verdict (DAG FEED-jz; 9-axis audit FEED-jy A9).

> a95b0ad6 (FEED-jz) found the clean-canonical BULK floor through R = 0.00291 (n96) = 2.4× the 1.23e-3
> d_seg budget; the 9-axis audit flagged it as a likely UPPER BOUND inflated by (A4/A8) the inter-pair
> CONSTANT-VELOCITY pose proxy, (A5/M2) the RGB-median blur, and (A5/M3) the occupancy-mask strawman.
> This tool resolves all three at $0 using the LOCALLY-AVAILABLE comma2k19 GT global pose for this exact
> segment. **The overturn attempt was run in full and the negative SURVIVED on the binding axis.**

## Linchpin de-risking (both PASSED — the GT pose is usable)
- **Frame alignment PROVEN:** comma2k19 frame g == video frame g — mean per-step distance **1.6766 m == speed×dt 1.6768 m** (rel_err 1.0e-4; fps 20.000; segment `b0c9d2329ad1606b|2018-07-27--06-03-57/10`, 1200 frames = 600 pairs).
- **Convention VALIDATED:** comma6 = `[fwd=‖Δp‖, vert, lat, −aa_dev]` (device→camera mapping); the comma6 within-pair Road+Lane warp fit **0.02269 ≤** the PoseNet within-pair fit **0.02299** — comma6 even engages yaw (s_r=0.0215) the PoseNet fit could not (s_r=0). Rotation columns map cleanly (corr −0.93 / −0.94 / −0.94 vs the stored PoseNet rotation). NO-FAKE selfcheck `SegNet(gt_f1)==lstars` PASSED.

## M1 — EXACT-pose warp (resolves A4/A8): the constant-velocity proxy was NOT inflating the floor
pre-R VOTE bulk (the fair denoiser-ceiling, label-space), window sweep, n96, SAME (comma) calibration both arms:

| window | proxy VOTE bulk | EXACT VOTE bulk |
|---|---:|---:|
| ±1 | 0.00290 | **0.00251** |
| ±2 | **0.00256** | 0.00291 |
| ±3 | 0.00263 | 0.00322 |

- best proxy VOTE = 0.00256 (2.08× budget); best EXACT VOTE = 0.00251 (2.04× budget). Within-tool same-calibration delta at ±2 = **+0.00034 (exact WORSE)**; cross-window best Δ = −0.00005 (negligible).
- **VERDICT M1: exact comma2k19 GT poses do NOT lower the bulk floor below the proxy.** Both sit at ~2.0× budget. The constant-velocity proxy was NOT the cause of the floor. The VOTE was already alignment-robust to small misalignment; better pose accuracy does not help it, and **larger windows HURT** (far neighbors warp worse). **A4/A8 REFUTED.**

## M2 — SHARP render (resolves the RGB-blur confound): de-blur lands at the VOTE level, not the carrier
through-R bulk d_seg, window ±2, n96:

| aggregator | bulk d_seg | vs budget | note |
|---|---:|---:|---|
| naive persist | 0.00506 | 4.1× | store prev frame |
| EXACT RGB-median (R) | 0.00612 | 5.0× | ≈ a95b0ad6 proxy RGB-median 0.0055 |
| **EXACT SHARP** (vote-consistent real RGB, R) | **0.00369** | **3.0×** | de-blurred, sharp texture |
| EXACT pre-R VOTE (label) | 0.00291 | 2.4× | denoiser ceiling |
| per-frame-exact carrier floor (FEED-jk) | 5.9e-4 | 0.48× | the achievable carrier |

- the RGB-median is blur-confounded (5.0× budget); SHARP de-blurs it (0.00612→**0.00369**, −40%) and **beats** the median, but lands at **3.0× budget = 6.26× the carrier floor**, ABOVE the pre-R VOTE (2.4×).
- **VERDICT M2: sharp render does NOT collapse the floor toward the 5.9e-4 carrier.** The blur was a real confound on the through-R median but removing it only recovers the pre-R VOTE level (~2.4×), NOT the carrier. The carrier floor is reachable only by a per-frame-EXACT rendered contour (FEED-jk), i.e. by storing/generating per-frame content — not by any warp/vote/sharp-select of neighbours. **A5/M2 does not overturn.**
  - (apples-to-apples note: the tool's internal *proxy* RGB-median = 0.01400 is calibration-mismatched — it applies the comma metric calibration to the PoseNet-unit proxy poses; the faithful comparison is the EXACT RGB-median 0.00612 vs a95b0ad6's published proxy RGB-median 0.0055, which MATCH. The scale-robust VOTE arms are unaffected and use the same comma calibration both sides.)

## M3 — STRUCTURED descriptor RATE (resolves A5): occupancy WAS a strawman, but few-KB needs lossy geometry
n96 (192 frames), scaled to 600:

| representation | bytes/600 | RMS | vs target 0.5–5KB |
|---|---:|---:|---|
| BULK horizon (Undriv↔drivable) occupancy mask | 129,653 | — | strawman |
| BULK horizon LOSSY poly (deg2 coef) | **2,288** | 6.0 px | **hits** (but lossy → re-incurs d_seg) |
| BULK horizon LOSSY poly (deg6 coef) | 5,247 | 3.3 px | hits |
| BULK horizon LOSSLESS (poly coef + exact resid) | 55,650 | 0 | misses |
| LANE occupancy mask (≈ a95b0ad6 iid 285,309) | 285,309 | — | strawman (gnd-frame delta was 367,753) |
| LANE centerline LOSSY poly (deg2 coef) | 2,772 | **96.7 px (DEGENERATE)** | invalid proxy |
| LANE structured point-set (honest) | 46,516 | 0 | misses (but 6× < occupancy) |

- **VERDICT M3: A5 PARTIALLY confirmed.** The occupancy mask WAS a strawman — structured geometry is far cheaper (BULK 130KB→56KB lossless; LANE 285KB→47KB honest point-set). BUT the **0.5–5KB target is reached only by a LOSSY poly (BULK deg2 2.3KB at 6px RMS, which re-incurs d_seg)**; the lossless bulk is still 56KB and the LANE does NOT hit few-KB (the "2.8KB centerline" is a **degenerate single-centroid-per-row fit at 96px RMS** — meaningless; the honest lane structured cost is the 47KB point-set). So: smooth geometry is cheap; the lane's thin multi-line structure + the per-frame jitter on top are NOT few-KB.

## M4 — BULK-JITTER DITHER cost + budget closure (the binding question): does NOT close
- bulk flip fraction under the EXACT-pose VOTE = **0.00297** (= the must-store residual). Flip labels: **Road 22,279 · Undriv 15,102 · MyCar 17,451** over 192 frames (≈285 flips/frame).
- **SMOKING GUN for A1 (intrinsic target-side jitter, NOT warp error):** MyCar/hood uses the **identity regime (NOT warped at all)** yet contributes **17,451 of 54,832 bulk flips (32%)**. The static hood's SegNet argmax flickers per-frame with zero warp involved → the floor is intrinsic frozen-SegNet per-frame noise, unremovable by ANY pose/warp.
- flips are only MODERATELY annulus-localized: mean flip margin 0.73, **median 0.37**; only 9.2% of flips at margin<0.05, 60% at margin<0.5 → margin-keying gives limited savings.
- min margin-keyed dither bytes to FULLY store the bulk jitter (drive bulk d_seg→0) = **177,926 B/600 → rate 0.1185** (coincidentally ≈ the PR95 frontier rate). S(100·d_seg@budget 0.123 + pose 0.018 + dither 0.1185) = **0.2595 ≫ 0.15. closes_sub_0.15 = FALSE.**
- **VERDICT M4: the budget does NOT close via cheap stored dither.** The bulk jitter is high-entropy per-frame SegNet noise: neither warp-removable (M1) nor cheaply storable (M4 ~178KB ≈ PR95's whole archive).
  - **NOT-PESSIMISTIC caveat (the genuinely-open door):** the 178KB dither is an iid sparse-set UPPER BOUND; (a) temporal/spatial correlation of recurring boundary flips could lower it, and (b) PR95 reaches d_seg 6e-4 at ~118KB by storing a LEARNED per-pair latent a decoder expands into the full partition INCLUDING the jitter — i.e. the jitter comes FREE with per-frame content reconstruction, not as separate dither. A TRAINED content-aware generator producing the jitter from a compact code (the GPU run) remains UNTESTED; this $0 probe bounds only the warp-only and naive-store routes.

## OVERTURN VERDICT — a95b0ad6's negative is CONFIRMED (robust), NOT overturned
The NOT-PESSIMISTIC overturn was run in full; the negative SURVIVED on the binding axis:
1. **M1 (exact GT pose) FAILS to overturn** — exact poses give the SAME ~2.0× bulk floor as the proxy (best 0.00251 vs 0.00256). A4/A8 REFUTED; the floor is genuine target-side jitter (the hood smoking gun proves it: 32% of flips are on the un-warped static hood).
2. **M2 (sharp render) FAILS to overturn** — de-blurring recovers the pre-R VOTE (~2.4×), not the 5.9e-4 carrier (6.3× above it).
3. **M3 (structured descriptor) PARTIALLY overturns A5** — occupancy strawman confirmed (structured 6–46× cheaper), but few-KB needs lossy geometry (d_seg cost) and the lane stays ~47KB.
4. **M4 (dither) does NOT close the budget** — bulk jitter store ≈ 178KB (rate 0.118) → S~0.26.

**Bottom line:** the bulk-near-free thesis is genuinely **REFUTED, not recovered**. The binding wall is the per-frame frozen-SegNet jitter (~2× budget), which depends on actual per-frame RGB texture (proven by the un-warped hood flicker) and is therefore unreachable by a geometry-only warp witness and uncheaply-storable as dither. It is reproducible only by a content-reconstructing/generating witness, which at the demonstrated PR95 operating point costs ~118KB (≈ the frontier rate). The audit's optimistic reframe ("the must-store floor is an inflated upper bound") is itself REFUTED for the warp+store routes; the one genuine recovery is that the occupancy-mask rate strawman overstated the *geometry* cost. The remaining open door is a TRAINED generator that emits the jitter from a compact code at low rate (the GPU run) — which this $0 probe cannot settle.

## Honest caveats / NO-FAKE (MEANS, not the end)
- Warps GT RGB / votes GT argmaxes (bounds the deterministic part); authority = realized-through-R inside the witness INR + exact CPU/CUDA eval — NOT this advisory probe.
- comma2k19 poses are EXACT (metric step distance + quaternion rotation); the device→camera column mapping for the SMALL lateral/vertical translation is INFERRED (negligible vs forward+rotation, which map cleanly).
- the internal proxy RGB-median arm is calibration-mismatched (comma calib on PoseNet-unit poses → 0.014); the faithful proxy reference is a95b0ad6's own 0.0055 (matched by the EXACT median 0.0061). The decisive VOTE comparison is same-calibration both sides.
- M3 structured descriptors are low-DOF proxies (bulk = horizon-row poly; lane = single-centroid-per-row poly is degenerate at 96px RMS → use the 47KB point-set); they LOWER-BOUND a faithful multi-curve spline coder.
- M4 dither = iid sparse-set entropy upper bound (temporal correlation untested).
- camera-res R model excludes sub-874 bicubic-up aliasing of a low-capacity sub-camera INR.
- `[macOS advisory / CPU-torch research-signal]`; pointer **0.19110 UNMOVED**. The exact pointer did NOT move.

## rule-118 tags
- **FREE (generic, inflate.py, uncounted):** homography + expmap + per-step compose + window vote/median/sharp + R chain + polynomial rasterizer.
- **COUNTED-but-EXISTING:** per-pair 6-DOF pose (already stored for d_pose; +0 marginal). The comma2k19 GT pose is a derivation tool here, NOT an archive payload.
- **COUNTED:** static scene descriptor + structured boundary coeffs + bulk-jitter dither/residual + lane/movables residual + any stored canonical keyframe.
- **NOT FORBIDDEN:** honest geometry + GT comma2k19 pose used to DERIVE the warp; NOT a smuggled per-frame argmax/warp table.

## Wire-in (Catalog #125)
1. **sensitivity-map** ACTIVE — the bulk floor is robust to pose source; the un-warped hood (MyCar identity) contributes 32% of bulk flips = intrinsic SegNet jitter rows.
2. **Pareto** ACTIVE — the exact-pose warp prior removes ~0 MORE than the proxy already did; the residual jitter is the binding rate cost.
3. **bit-allocator** ACTIVE — do NOT spend bytes on a better pose (no d_seg return); do NOT store the bulk jitter as explicit dither (rate 0.118 ≈ PR95 whole archive); prefer a trained content generator OR accept the bulk lossy structured boundary (2–5KB at a measured d_seg cost).
4. **cathedral autopilot** N/A (advisory probe).
5. **continual-learning** — this memo + the JSON + the returned DAG FEED.
6. **probe-disambiguator** — `tools/measure_budget_gate_overturn.py` IS the disambiguator (exact-vs-proxy pose; sharp-vs-median; structured-vs-occupancy; dither cost vs budget). Next disambiguator = the TRAINED conditioned-residual generator (GPU) for the jitter.

## Primary citations
- a95b0ad6 clean-canonical budget gate (`.omx/research/clean_canonical_warp_budget_gate_20260629T203717Z.md`, DAG FEED-jz) + the FEED-jy 9-axis audit; a23062c4 screw-through-R (FEED-jq); a513372a screw probe; FEED-jk single-SDF carrier (5.9e-4); comma2k19 GT pose (`experiments/results/pose_feasibility_probe/comma2k19_gt_pose_raw.npz`, probe #155 / `experiments/probe_pose_side_feasibility_taskspace_155.py`).
- Hartley & Zisserman plane-induced homography `H=K(R−t nᵀ/d)K⁻¹`; Longuet-Higgins & Prazdny 1980; openpilot/comma2k19 EON intrinsics (fx=fy=910, pp=(582,437) @ 1164×874).
