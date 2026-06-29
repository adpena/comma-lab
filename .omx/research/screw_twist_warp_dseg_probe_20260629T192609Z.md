# SCREW / TWIST (Chasles–Hodge) pose-warp d_seg probe — is one ego-twist a ~0-byte per-class warp?

**UTC** 2026-06-29T19:26:09Z · **authority** `[macOS advisory / research-signal]` · **pointer UNMOVED 0.19110**
**score_claim** false · **promotable** false · **ready_for_exact_eval_dispatch** false
**Tool** `tools/measure_pose_warp_dseg.py` (extended; `screw_analysis`) · **runs** PRE-R, label-space, pure-numpy, $0
**JSON** `experiments/results/screw_twist_warp_dseg_n96/results.json` + `…_n200/results.json` (`screw_analysis` block)
**Tests** DAG FEED graphics_aa Task 6 (the Hodge/screw warp) · extends FEED-iv/iu (grok pose-warp) · settles whether
the single-twist parameterization is a real ~0-byte win for the v2 witness.

> **The claim under test (screw theory / Chasles + Longuet-Higgins–Prazdny).** Every rigid ego-motion is ONE twist
> `(t, ω) ∈ se(3)`. The per-class warps are DERIVED from that single twist + a tiny STATIC scene descriptor:
> Road/Lane → ground homography `H = K(R − t nᵀ/d)K⁻¹`; sky → rotation-only `H = K R K⁻¹` (the `Z→∞` limit, t dropped);
> hood → identity; movables → ground (best guess; the part it can't explain IS the residual). The twist **reuses the
> already-stored 6-DOF pose** (the d_pose sidecar) at **~0 marginal bytes**, vs independently-fit per-class
> homographies (~6,600 params/600 pairs, graphics_aa Task 6). **Question:** does the single-twist stratified warp (c)
> MATCH the per-class independent homography oracle (b) — i.e. is the compression d_seg-free?

## Method (extends the grok probe; same authority firewall)

`d_seg` = REAL argmax-disagreement of a forward-warped label map vs the frozen CPU-torch SegNet argmax `lstars`
(`experiments/results/mlx_fleet_gt_cache/{gt_n96,gt_strided_n200}.npz`; classes `[Road,Lane,Undriv,Movable,MyCar]`).
Local consequence of the canonical claim: predict `lstars[p+1] := warp(lstars[p], H_rel(pose))` and compare PER CLASS:

- **(a) persist** — naive copy, `pred = lstars[p]` (the no-motion null).
- **(b) per-class INDEPENDENT oracle** — each class fits its OWN `(s_t,s_r,pitch)` full ground homography on its own
  pixels (the per-class-homography upper bound; the expensive granularity the screw would replace).
- **(c) single-twist SCREW stratified** — ONE shared calibration (the Road+Lane fit) + the **physics regime** per
  class (Road/Lane/Movable=ground, sky=rotonly, hood=identity). **No per-class free parameters.**

**Fair, non-gameable accounting (a fix over the first pass):** where a warp maps off-frame (invalid), we **fall back to
persist** (and the oracle fit uses the same full-coverage objective) — so a per-class fit cannot drive its d_seg to 0
by *invalidating* pixels (an artifact that gave a spurious hood `0.0000`). d_seg is scored over **every** target-class
pixel (no coverage caveat; denominators of a/b/c are identical → directly comparable).

## Results — two independent samplings AGREE → `SCREW_WIN_ZERO_BYTE_PHYSICAL`

**Totals (d_seg):**

| sampling | (a) persist | (b) per-class oracle | (c) single-twist screw | gap c−b | of which non-physical oracle overfit | genuine residual |
|---|---|---|---|---|---|---|
| n96 (95 transitions) | 0.01148 | 0.00950 | **0.01074** | +0.00124 | 0.00105 (85%) | 0.00019 (15%) |
| n200 strided (199) | 0.01672 | 0.01372 | **0.01538** | +0.00166 | 0.00143 (86%) | 0.00025 (14%) |

**Per-class (n96; n200 in the same direction):**

| class | area | regime (c) | (a) persist | (b) oracle | (c) screw | c−b | oracle `s_t` | oracle non-physical? |
|---|---|---|---|---|---|---|---|---|
| Road | 0.230 | ground | 0.0231 | 0.0196 | **0.0196** | +0.0000 | −0.0055 | no (physical) |
| Lane | 0.006 | ground | 0.5795 | 0.5793 | 0.5851 | +0.0057 | 0.0 | no (survival residual) |
| Undriv (sky) | 0.493 | rotonly | 0.0024 | 0.0018 | 0.0024 | +0.0006 | **+0.0126** | **yes** (translation on Z→∞ sky) |
| Movable | 0.016 | ground | 0.0506 | 0.0424 | 0.0522 | +0.0099 | +0.0032 | no (genuine independent-motion residual) |
| MyCar (hood) | 0.256 | identity | 0.0031 | 0.0001 | 0.0031 | +0.0030 | **+0.0038** | **yes** (translation, opposite-sign, on a static hood) |

**Sky-divergence-null ablation (Task 6 #2): the t-term HURTS the sky in BOTH samplings → depth-independence confirmed.**

| sampling | sky rotonly (screw, t dropped) | sky ground (+t added back) | sky identity | t-term hurts sky? |
|---|---|---|---|---|
| n96 | 0.00241 | 0.00247 | 0.00241 | **True** |
| n200 | 0.00496 | 0.00604 | 0.00496 | **True** (penalty bigger — more forward motion ⇒ larger spurious sky translation) |

Fitted shared calibration: n96 `s_t=−0.0032, s_r=0, pitch=−0.01`; n200 `s_t=−0.0119, s_r=0, pitch=0.09`. `s_r=0` (the
learned-pose rotation columns don't drive a useful rotation at the fitted scale), so rotonly≈identity on the sky here;
the ablation still validates depth-stratification (adding the t-term to the sky hurts regardless of `s_r`).

## Verdict — the screw is a real ~0-byte parameterization WIN (at the physical level)

- **The single 6-DOF twist MATCHES the per-class independent homography oracle on every PHYSICALLY-MEANINGFUL class.**
  Road is reproduced **exactly** (c=b, both use the Road+Lane fit); the screw's physical choices for the static classes
  (hood=identity, sky=rotation-only) are the **correct, generalizable** ones. The single global homography of the prior
  grok probe DESTROYED those static classes (hood −525%/−2677%); the screw **fixes them by construction**.
- **The oracle's small raw-total edge is ~85% clip-specific NON-PHYSICAL overfit.** Its hood fit uses `s_t=+0.0038`
  (opposite sign to Road's `−0.0055`) — a rigidly-attached hood cannot translate, let alone against the ego-motion; its
  sky fit uses `s_t=+0.0126` — a `Z→∞` sky cannot translate. These per-class warps lower this clip's boundary flicker
  by 0.001-class amounts that **do not generalize** and **cost per-class bytes**. The screw deliberately forgoes them.
- **The only GENUINE residual the screw cannot capture is Movable independent motion** (+ a sliver of Lane). That is
  exactly the off-ego-orbit residual the grok probe predicted — `~0.00019` (n96) / `~0.00025` (n200) of total d_seg.
- **Byte verdict.** screw (c) = **~0 marginal** (reuses the stored 6-DOF pose) **+ O(10) static params for the whole
  clip** (calibration `s_t/s_r/pitch` + plane `n,d` + hood-mask) vs oracle (b) = 3 scalars × 5 classes = 15 globals at
  the measured granularity, or **~6,600 params** as per-pair per-class homographies (graphics_aa Task 6 — the expensive
  alternative the screw replaces). **Same d_seg fidelity on the physical classes, far fewer bytes ⇒ the screw WINS.**

This is a constructive **confirmation+refinement** of graphics_aa Task 6 and the grok probe's stratified-warp picture,
not a kill (Forbidden-premature-KILL: paradigm intact; the refinement is "one twist, physics-stratified, persist-
fallback").

## rule-118 tags
- **FREE (generic, expandable in `inflate.py`, uncounted):** the Longuet-Higgins–Prazdny / plane-induced-homography
  formula + `expmap`/Rodrigues + the per-class regime selection + the warp. A deterministic geometric algorithm.
- **COUNTED-but-EXISTING:** the per-pair 6-DOF pose stream (already stored for `d_pose`; the screw adds **no new
  per-pair payload**).
- **COUNTED-but-TINY:** the static scene descriptor `(n, d, hood-mask, calibration)` — stored ONCE for the clip.
- **NOT FORBIDDEN:** the descriptor is honest geometry, not a per-frame argmax/warp table smuggled as "code."

## Honest caveats / NO-FAKE (this is a MEANS, not the end)
- **PRE-R, label-space.** The existing tool has **no R operator**; the contest R (bicubic↑874 → uint8 → bilinear↓384 →
  argmax) acts on a **witness RGB** we do not have here, so a faithful through-R number is **not measurable in this
  pure-numpy probe** (I did NOT fabricate one). The **Lane-survival residual (0.58) is a lower bound** — through-R can
  only worsen the thin-lane flip (GAP-2, the binding wall, is blind here). The cache *does* hold `gt_f0/gt_f1` RGB, so a
  through-R probe is feasible as a next step but requires running the frozen SegNet (not $0-pure-numpy).
- **Twist source = the LEARNED PoseNet 6-vector** (the d_pose sidecar), **NOT comma2k19 metric pose** (not in the
  cache). The learned→ground calibration is the 3 fitted global scalars (low-capacity, data-determined; cannot overfit
  per-frame). An exact relative ego-pose would likely explain MORE on Road (the +Road match is a lower bound).
- **HHD labeling caveat (graphics_aa Task 6):** div↔translation / curl↔rotation is exact only for forward-translation +
  roll; yaw/pitch mix. The **depth-stratification itself** (translation ∝ 1/Z, rotation depth-independent) **is exact**
  (Longuet-Higgins–Prazdny) — and the sky-null ablation confirms its sign-of-effect.
- `[macOS advisory / research-signal]`; pointer **0.19110 UNMOVED**. This redirects the v2 witness build; it is not a
  byte-closed exact row.

## Single most decisive next step
**Take the screw warp THROUGH R.** GAP-2 (R-survival of the thin Lane) is the binding wall and this probe is blind to
it. Since the cache holds `gt_f0/gt_f1` RGB, warp the GT/witness RGB by the screw homographies → push through the R
operator (↑874 bicubic → uint8 → ↓384 bilinear) → frozen SegNet → measure d_seg. That converts the Lane lower bound
into the real binding number and tells us whether the screw-predicted front + the (SDF/MSDF) lane carrier survive R —
the actual sub-0.15 gate. (Secondary: per-object 6-DOF twist decomposition of Movables, GAP-1, a few extra streams.)

## Wire-in hooks (Catalog #125)
1. **sensitivity-map** ACTIVE — per-class warp-regime fidelity is a new sensitivity row; the non-physical-oracle flag is
   a new overfit-detector signal. 2. **Pareto** ACTIVE — `screw (~0 marginal bytes) ↔ d_seg` vs `per-class homographies
   (~6,600 params) ↔ d_seg` is a measured rate↔distortion arm; the screw dominates on the physical classes. 3.
   **bit-allocator** ACTIVE — spend per-pair bytes on the pose (free, dual-use) + the Movable/Lane residual, NOT on
   per-class warp params. 4. **cathedral autopilot** N/A (advisory probe, not archive-deployable yet). 5.
   **continual-learning** this memo + the DAG FEED. 6. **probe-disambiguator** — `tools/measure_pose_warp_dseg.py`
   `screw_analysis` IS the disambiguator (single-twist vs per-class oracle, with the non-physical-overfit decomposition);
   the named through-R follow-up is the next disambiguator.

## Primary citations
- Longuet-Higgins, H.C. & Prazdny, K. 1980. *The interpretation of a moving retinal image.* Proc. R. Soc. Lond. B 208,
  385–397 (ego-flow split: translation ∝ 1/Z, rotation depth-independent).
- Chasles' theorem / screw theory; `se(3)` twists; Helmholtz–Hodge decomposition of ego optical flow (IEEE 2010,
  omnidirectional-flow robot navigation). Rodrigues / `so(3)≅ℝ³` axis-angle exponential map.
- Hartley & Zisserman, *Multiple View Geometry*: plane-induced homography `H = K(R − t nᵀ/d)K⁻¹`.
- openpilot / comma2k19 EON intrinsics (fx=fy=910, pp=(582,437) at 1164×874; `HEIGHT_INIT`=1.22 m).
- graphics_aa Task 6 (`.omx/research/graphics_aa_astronomy_inverse_codec_crosscheck_20260629T184900Z.md`); grok pose-warp
  probe (`.omx/research/grok_pose_warp_dseg_test_20260629T181000Z.md`).
