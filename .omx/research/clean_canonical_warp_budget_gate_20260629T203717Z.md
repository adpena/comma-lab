# CLEAN-CANONICAL warp d_seg THROUGH R + ground-frame RATE gate (the v2 budget gate)

**UTC** 2026-06-29T203717Z · **authority** `[macOS advisory / CPU-torch research-signal]` · **pointer UNMOVED 0.19110**
**score_claim** false · **promotable** false · **ready_for_exact_eval_dispatch** false
**Tool** `tools/measure_clean_canonical_warp_through_R.py` (new; extends a23062c4 `measure_screw_warp_through_R.py` + a513372a `measure_pose_warp_dseg.py`)
**JSON** `experiments/results/clean_canonical_warp_n96_r2/results.json` + `…_n200strided_r2/results.json`
**Authority** FROZEN CPU-torch SegNet (`load_real_segnet("cpu")` + `measure_segnet_argmax`), NEVER MPS.
**Settles** the decisive $0 step DAG FEED-jq / a23062c4 named: is the bulk d_seg pose-explainable (a CLEAN canonical removes the inter-frame jitter) or genuine per-frame noise that must be stored?

> **The question a23062c4 left open.** a23062c4 warped the NOISY previous frame through R and got bulk
> d_seg ~0.0048 (~4x the 1.23e-3 budget), attributing the floor to the **inter-frame SegNet per-frame
> JITTER (~0.008)** that a single-neighbour warp cannot predict. It named the decisive open: how much of
> that floor is (a) POSE-EXPLAINABLE — removed by warping a CLEAN, temporally-aggregated, jitter-free
> canonical — vs (b) GENUINE per-frame SegNet noise that MUST be stored per-frame (what PR95 stores to
> reach ~6e-4)? This memo runs that gate.

## Method (the clean canonical)

For each target pair p (target = the f1 frame, global index t=2p+1): take a short window of NEIGHBOUR
frames {t−R..t+R} **excluding t** (the target's own jitter must NOT leak in), warp each into the target's
view via the screw per-class-regime homography composed along the per-frame ego-motion chain (Road→ground
plane homography, sky→rotation-only, hood→identity), and AGGREGATE → a denoised canonical. Two aggregators,
both measured:
- **through-R RGB-median**: per-pixel median of the warped RGB → R (warp@874→uint8@874→bilinear↓384) →
  frozen CPU-torch SegNet → argmax. The literal "warp a clean RGB canonical through R."
- **pre-R argmax-VOTE**: per-pixel majority vote of the warped cached SegNet argmaxes. Isolates the
  jitter-averaging effect WITHOUT the RGB-median blur confound → the FAIR upper bound on removable jitter.

Compared to: a23062c4's **prev-frame-warp** (re-measured here as the single-source baseline, same cache +
same calibration → apples-to-apples) and the **per-frame-exact carrier floor** (FEED-jk single-SDF lane
@render-192 = 5.9e-4, cited). Window radius R=2 global frames (4 neighbours; composition depth ≤2 steps).

- **NO-FAKE self-check (PASSED):** `SegNet(gt_f1)==lstars` exact on the 4-pair gate AND on ALL 96 pairs of
  the per-frame cache (`cache_selfcheck` 96/96, 0 px). The tool ABORTS rather than report a number if it fails.
- **prev-frame-warp REPRODUCED:** my n96 prev-frame through-R bulk = **0.00477** vs a23062c4's **0.0048**
  (apples-to-apples reproduction → the tool is sound).

## Results — TEST-1 (the BUDGET gate, d_seg THROUGH R, n96, window ±2)

Per-class d_seg (Road/Lane/Movable scored under ground regime, Undriv→rotonly, MyCar→identity):

| class | regime | area | naive-persist (R) | prev-warp (R) | canon RGB-median (R) | prev-label (preR) | **clean VOTE (preR)** |
|---|---|---|---|---|---|---|---|
| Road | ground | 0.230 | 0.0155 | 0.0142 | 0.0176 | 0.0135 | **0.0051** (−67%) |
| Lane | ground | 0.006 | 0.3934 | 0.3896 | 0.4023 | 0.3952 | **0.6557** (vote ERODES thin lane) |
| Undriv(sky) | rotonly | 0.493 | 0.0016 | 0.0016 | 0.0014 | 0.0016 | 0.0016 |
| Movable | ground | 0.016 | 0.0347 | 0.0348 | 0.0345 | 0.0336 | 0.0520 (vote can't track motion) |
| MyCar(hood) | identity | 0.256 | 0.0028 | 0.0028 | 0.0030 | 0.0028 | 0.0036 |

**BULK (Road+sky/Undriv+hood/MyCar) terms:**

| quantity | bulk d_seg | vs budget 1.23e-3 |
|---|---|---|
| naive-persist (store prev frame) | 0.00506 | 4.1× |
| **prev-frame-warp through R** (a23062c4) | **0.00477** | 3.9× |
| through-R RGB-median canonical | 0.00550 | 4.5× (HURTS — blur) |
| pre-R prev-frame label warp | 0.00459 | 3.7× |
| **pre-R clean-canonical VOTE (best clean)** | **0.00291** | **2.4×** |
| per-frame-exact carrier floor (FEED-jk @render-192) | 5.9e-4 | 0.5× |

### Two independent samplings AGREE (the verdict is robust)

| sampling | prev-warp bulk (R) | a23062c4 ref | best clean (VOTE) bulk | vs budget | pose-explainable % | genuine must-store |
|---|---|---|---|---|---|---|
| **n96 (within-pair)** | 0.00477 | 0.0048 ✓ | **0.00291** | **2.4×** | 37% | 0.0023 |
| **n200 (strided)** | 0.00512 | 0.0051 ✓ | **0.00427** | **3.5×** | 15% | 0.0037 |

Both reproduce a23062c4's prev-frame-warp bulk and both land the best clean-canonical bulk WELL ABOVE budget
(NO-FAKE cache self-check `SegNet(gt_f1)==lstars` PASSED on ALL pairs of BOTH samplings: 96/96 and 200/200).
The strided sampling has larger inter-frame baselines → larger warp error → LESS removable (15% vs 37%) and
the RGB-median blur hurts MORE (−78% vs −15%). So the pose-explainable fraction is **15–37%** (baseline-
dependent) and the genuine must-store residual is **0.0023–0.0037 (≈1.9–3.0× budget)** — verdict invariant.

## The DECISIVE decomposition of the ~0.008 inter-frame floor

The clean canonical (best case = blur-free VOTE) removes **37% of the bulk inter-frame jitter**
(pre-R bulk 0.00459 → 0.00291), almost entirely by denoising **Road −67%** (0.0155→0.0051). This 37% IS the
**pose-explainable / source-jitter** fraction — removable by a clean canonical for FREE (deterministic warp).

But the remaining bulk = **0.00291 = 2.4× budget**, and it is no longer Road-dominated: after Road is
denoised, the floor is split across Road (contrib 0.0012) + hood (0.0009) + sky (0.0008). Subtracting the
per-frame-exact carrier floor (5.9e-4), the **genuine target-side per-frame jitter that MUST be stored ≈
0.0023 (≈1.9× budget alone)**. This is the per-frame partition correction NO warp (clean or noisy) can
produce — it is exactly what PR95 stores per-pair to reach ~6e-4.

- **POSE-EXPLAINABLE fraction (removed by clean canonical): 37% (n96) / 15% (n200 strided)** — baseline-
  dependent; larger inter-frame motion → less removable.
- **GENUINE per-frame-noise fraction (must store): 63–85%** (≈0.0023–0.0037), of which ~5.9e-4 is the
  achievable R-carrier floor and the rest is the must-store per-frame payload above it.

**RGB-median HURTS (+15%):** median-of-misaligned-RGB blurs boundaries → SegNet shifts → worse. A real
witness must render a SHARP partition (the VOTE / a rendered carrier), not a blurred RGB. The VOTE is the
fair denoiser ceiling; the literal RGB-median through-R is pessimistic.

**The VOTE ERODES the thin Lane (0.39→0.66) and Movables (0.035→0.052)** — confirming (again) that the Lane
and Movables must be STORED/TRAINED, never voted/warped. Bulk → clean-canonical-warp (partial); Lane →
trained/stored residual; Movables → stored. (Consistent with FEED-jk/jq/jm.)

## VERDICT on the v2 budget thesis

**"Bulk near-free via warp" is REFUTED even for a CLEAN canonical.** Best-case clean-canonical bulk =
**0.00291 = 2.4× the 1.23e-3 budget**. The clean canonical recovers 37% of the inter-frame jitter for free
(deterministic), but the bulk's remaining ~0.0023 (≈1.9× budget) is genuine target-side per-frame SegNet
jitter that requires PER-FRAME partition storage (the thing PR95 stores), NOT a static canonical + warp.

**The v2 vehicle implication (refines, does NOT kill):**
1. The clean-canonical + warp is a real, FREE win on the BULK: it captures ~37% of the bulk jitter
   (Road −67%) deterministically. v2 SHOULD use it as the bulk prior.
2. But the bulk ALSO needs a per-frame trained/stored residual (~0.0023 d_seg, ~1.9× budget) — "bulk needs
   no INR" is wrong. The per-frame bulk residual is SMALLER than storing the full per-pair latent (PR95),
   so there is a partial compression win, but the bulk is not free.
3. Lane (0.39 flip, separate) + Movables → trained/stored (warp/vote both fail them).
4. **Untested next step (flagged):** the TRUE through-R clean canonical = RENDER the voted SHARP partition
   as a witness RGB → R → SegNet (between the pre-R vote 0.00291 and the carrier floor 5.9e-4). I measured
   the pre-R vote (denoiser ceiling) and the RGB-median (blur floor); the rendered-sharp-partition-through-R
   composite is the next $0 measurement.

## Results — TEST-2 (the RATE gate, lane bytes; closes FEED-jm's correction)

Lane occupancy (lstars==1) coding, n96 interleaved (192 consecutive frames), scaled to 600:

| scheme | bytes/600 | vs iid |
|---|---|---|
| iid per-frame (FEED-jm-style baseline) | 285,309 | 1.00× |
| image-space XOR temporal delta (no ego-comp) | 362,897 | 1.27× (WORSE) |
| **ground-frame ego-compensated delta** | **367,753** | **1.29× (WORSE)** |

Adjacent-frame lane IoU: **image-space 0.425, ground-aligned 0.421** (ego-comp does NOT improve alignment).

**RATE verdict:** at the occupancy-mask level, ground-frame ego-compensation does **NOT** recover lane
temporal redundancy and does **NOT** hit the 0.5–5KB target — it is ~1.29× WORSE than iid, and ego-comp
leaves the lane IoU essentially unchanged (0.425→0.421). The thin/jittery lane mask (2px sub-Nyquist per
FEED-jk) is temporally non-redundant in BOTH image and ground frames at 1-frame spacing, and XOR-delta of
two thin jittery masks costs MORE than independent coding. **Corroborates FEED-jm's "rate-half NOT
settled-tiny" correction.**
- **STRONG caveat:** FEED-jm/jd's 0.5–5KB target assumed a STRUCTURED centerline/spline ground-frame
  descriptor (sparse, low-DOF), NOT an occupancy mask. This measurement evaluates the occupancy mask (an
  upper bound). The structured-spline ground-frame coder remains the UNMEASURED path to the few-KB target;
  this result neither reaches it nor fully refutes it — it refutes only the occupancy-mask route.
- Pose stream marginal ≈ 0 (already stored for d_pose); ground-frame adds only the static descriptor +
  stored canonical keyframe bytes.

## Honest caveats / NO-FAKE (this is a MEANS, not the end)
- **Warps GT RGB / votes GT argmaxes, not a shipped witness.** Bounds the deterministic part; the authority
  is realized-through-R inside the witness INR + exact CPU/CUDA eval on byte-closed bytes — NOT this probe.
- **INTER-PAIR pose is a PROXY:** only WITHIN-pair poses are stored (`gt_poses[p]` = f0→f1); inter-pair
  steps use `0.5*(pose[p]+pose[p+1])` (constant-velocity-ish). Per-step plane-homography composition over
  the window is a small-motion approximation (window kept ≤±2). If this warp error inflates the residual,
  the genuine-must-store fraction is an UPPER bound (i.e., even MORE of the floor may be removable with
  exact inter-pair poses — but the bulk would still need per-frame info to clear 2.4× budget).
- **RGB-median blur vs VOTE:** the two aggregators bracket the truth; the rendered-sharp-partition-through-R
  composite (the real witness route) is unmeasured (flagged above).
- **Calibration = 3 global scalars** (s_t,s_r,pitch) fit in label space, applied to RGB; the raw PoseNet
  6-vector column interpretation is the flagged INFERRED assumption.
- **Camera-res R model** excludes sub-874 bicubic↑ aliasing of a low-capacity sub-camera INR.
- `[macOS advisory / CPU-torch research-signal]`; pointer **0.19110 UNMOVED**. **The exact pointer did not
  move.** This redirects the v2 witness build; it is not a byte-closed exact row.

## rule-118 tags
- **FREE (generic, expandable in inflate.py, uncounted):** plane-induced homography + expmap + per-step
  composition + window-median/vote + R chain — a deterministic geometric algorithm.
- **COUNTED-but-EXISTING:** the per-pair 6-DOF pose (already stored for d_pose; the screw adds NO per-pair payload).
- **COUNTED:** the static scene descriptor (n,d,hood-mask,calibration) + the per-frame BULK residual
  (~0.0023 d_seg must-store) + the Lane/Movables stored/trained residual + any stored canonical keyframes.
- **NOT FORBIDDEN:** honest geometry, not a smuggled per-frame argmax/warp table.

## Wire-in hooks (Catalog #125)
1. **sensitivity-map** ACTIVE — clean-canonical-vote per-class deltas are new rows (Road −67% removable;
   sky/hood per-frame jitter NOT removable; lane/movables erode under vote = store-only).
2. **Pareto** ACTIVE — the rate↔distortion arm: clean-canonical-warp (~0 byte) removes 37% of bulk d_seg;
   the remaining ~0.0023 bulk + lane is the per-frame stored/trained payload.
3. **bit-allocator** ACTIVE — spend bytes on (a) the per-frame BULK residual ~0.0023, (b) the Lane residual
   (binding), (c) small Movables; NOT on the bulk warp params (free); the clean-canonical prior is free.
4. **cathedral autopilot** N/A (advisory probe).
5. **continual-learning** this memo + the JSON + DAG FEED (next).
6. **probe-disambiguator** — `tools/measure_clean_canonical_warp_through_R.py` IS the disambiguator (pose-
   explainable vs genuine-jitter; RGB-median vs vote; occupancy-mask ground-frame rate). Next disambiguator
   = render-the-voted-sharp-partition-through-R + the structured-spline ground-frame coder.

## Primary citations
- a23062c4 screw-warp-through-R (`.omx/research/screw_warp_through_R_gap2_20260629T195829Z.md`, DAG FEED-jq);
  a513372a screw probe (`screw_twist_warp_dseg_probe_20260629T192609Z.md`); FEED-jk single-SDF lane carrier
  (`msdf_lane_carrier_probe_20260629T193309Z.md`, 5.9e-4 @render-192); FEED-jm openpilot lane head-start
  (`openpilot_lane_headstart_landed_20260629T193648Z.md`, 65KB image-space iid correction); F1 R-survival
  (`r_survival_physics_20260629T182659Z.md`).
- Longuet-Higgins & Prazdny 1980; Hartley & Zisserman plane-induced homography `H=K(R−t nᵀ/d)K⁻¹`;
  openpilot/comma2k19 EON intrinsics (fx=fy=910, pp=(582,437) @ 1164×874).
