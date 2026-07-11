# Partition anisotropy map — d_H = log(λmax/λmin) of the argmax boundary, per-edge / per-saddle / temporal — 2026-07-10

**Question (deep-insight dispatch):** WHERE does the geometry+factorization+SPD-cone treatment
(proven on pose — SPD-matched code buys ~27% bytes at pose d_H≈10.5, `spd_cone_pose_codec_ab_measured.json`;
being tested on the lane by a sister agent) have leverage BEYOND the lane? Rank every inter-class
edge and every Morse-Smale saddle by anisotropy so future carriers are routed to the right chart.

**Method ($0, memory-light, NO scorer forward, NO model inference):** on the CACHED argmax labels +
margin field (`gt_n96.npz` AND `gt_n600.npz` `lstars`/`margins`), compute the per-pixel local 2×2
**structure tensor** of the margin (Fisher-surrogate; margin↔Fisher Pearson 0.978) field
`J = G_σ*(∇m ∇mᵀ)` and `d_H = log(λmax/λmin)`. This is the SAME quantity whose byte-win the SPD-cone
pose codec measured (there: `log(cond)` of the 6×6 pose covariance). SADDLES use the margin **Hessian**
(mixed-sign eigenvalues = hyperbolic, distinct from the SPD structure tensor). Class order SELF-DETECTED
from spatial/static signature (never luma-sorted): `0=Road 1=Lane 2=Undriv 3=Movable 4=MyCar`.

**Authority: `[macOS-MLX advisory]`** — a geometric anisotropy map on cached argmax, NOT through R +
the frozen SegNet, NOT byte-closed. It ROUTES future carriers; it moves NO score. **Pointer 0.19108282 UNMOVED.**

**Apparatus:** `src/tac/boundary_math/partition_anisotropy_map.py` + CLI
`tools/probe_partition_anisotropy_map.py`; 10 tests incl. synthetic controls (straight edge → high d_H;
isotropic bump → low d_H; hyperbolic paraboloid → saddle; convex bowl → not-saddle; horizon-line recovery;
fail-closed class-order guard). Artifacts: `experiments/results/partition_anisotropy_map_20260710/probe.json`
(n96) + `probe_n600.json` (fullest scale). n96 and n600 AGREE (ranking identical, d_H within ±0.1).

## 1. The ranked EDGE anisotropy map (n600; n96 identical)

| Edge (class pair) | d_H (mean) | d_H (energy-wt) | pixel-share | leverage = d_H×share | temporal | TREATMENT |
|---|---|---|---|---|---|---|
| **Road–MyCar (hood)** | **6.73** | 6.96 | 0.196 | 1.32 | STATIC (IoU 0.992) | **static mask #139 — d_H is a red herring: no temporal variance to factor → store-once (already near-free)** |
| **Road–Lane (lanes)** | 3.74 | 3.88 | **0.503** | **1.88** | dynamic | ground-frame ξ chart — **SISTER agent owns this** |
| **Road–Undrivable (HORIZON)** | **4.45** | **4.75** | 0.179 | 0.80 | dynamic (pitch) | **ground-frame ξ chart — THE generalization target (see §3)** |
| Undrivable–Movable (car tops) | 2.86 | 2.99 | 0.061 | 0.18 | dynamic | moment-ellipse (mildly directional at car crown) |
| Road–Movable (car bottoms) | 2.55 | 2.84 | 0.056 | 0.14 | dynamic | **moment-ellipse / site-coder — LOW d_H = compact/isotropic; factorization has LOW leverage** |
| Lane–MyCar / Lane–Undriv / Lane–Movable / Movable–MyCar | 1.1–3.8 | — | <0.004 | ~0 | — | negligible mass |

**Reads:**
- **Anisotropy is REAL and it ranks cleanly.** Directional edges (hood 6.7, horizon 4.5, lane 3.7) are
  2–3× the d_H of the Movable blobs (2.5–2.9). The hypothesis holds on BOTH ends: lane/horizon/hood are
  strongly rank-1 directional; **Movable boundaries ARE the low-d_H isotropic class** (compact blobs) —
  confirming the v8 memo's moment-ellipse routing for Movable, and confirming factorization has little
  leverage there.
- **The hood is the anisotropy TRAP.** Road–MyCar has the HIGHEST d_H (6.7) but it is temporally STATIC
  (MyCar IoU 0.992, #139). d_H measures DIRECTIONAL rank-1-ness; the SPD-cone/factorization win is in
  FACTORING VARIANCE (temporal or across the fibre). A static edge has ~zero variance to factor → the
  right treatment is store-once, NOT a ξ chart. **Routing must be `d_H × share × (NOT static)`, not d_H
  alone.** This is the key correction to a naive "rank by d_H" reading.

## 2. The SADDLE map — FULL eigenstructure of the margin Morse-Smale complex (n600)

The edges + saddles here ARE the Morse-Smale complex of the cached margin field `m = φ_top − φ_runnerup`
(the margin field is CACHED, so the Hessian sign-signature is DIRECT, not owed-pending-logits). Edges are
rank-1 (a scalar d_H suffices); at saddle strata we report the FULL eigenstructure (λ₁, λ₂, ratio) of the
margin-Hessian AND the structure tensor, because that is the load-bearing edge-vs-saddle distinction.

- **11.2 triple junctions / frame** (n96: 9.1) — ≥3 distinct class labels in a 2×2 window (label-only,
  no logits). The vanishing point (Road+Lane+Undriv) is the canonical one; car ground/sky corners follow.
- **Hard-mass concentration = 17.1×** (n96: 16.7×): low-margin (bottom-5% = flip-prone) pixels are **17×
  over-represented** in saddle neighborhoods — the topologically-hard d_seg mass DOES sit at the junctions.
- **BUT the FULL eigenstructure REFUTES a naive "saddles are rank-2, the directional lever fails there":**
  measured AT the junction pixel, margin-Hessian median (λ₁, λ₂) = (0.150, 0.007) —
  - **only 37% are mixed-sign** (hyperbolic); the other 63% are near-isotropic ELLIPTIC minima (the margin
    dips to ~0 at the junction and rises outward = a bowl, NOT a saddle — three logits tie ⟹ a min of the
    top-2 gap, exactly as Morse theory predicts).
  - **among the mixed-sign ones the Hessian is quasi-rank-1:** |λmin|/|λmax| median = **0.082** (one
    curvature dominates ~12×). The structure-tensor d_H at junctions is 1.12 (lower than edges' 3.7–4.5 —
    junctions are less coherent, as expected — but not 2D-flat).
  - **⟹ ~98% of triple junctions are DIRECTIONALLY-CODEABLE** (rank-1 hyperbolic or an elliptic bowl the
    #1 all-class directional basis + a scalar offset handles); **only ~2% are GENUINELY 2D-hyperbolic**
    (comparable mixed eigenvalues) where a single-tangent directional basis truly fails.
- **Routing consequence (CORRECTED by the eigenstructure):** the #1 directional lever is NOT broadly
  defeated at saddles — it codes ~98% of junctions. **Genuine saddle-aware (2-parameter junction) coding is
  a SMALL, high-value ~2% RESIDUAL, not a big program.** The 17× hard-mass concentration is real, but it is
  a PRECISION problem at directionally-codeable loci, not a chart-change problem. (The earlier
  neighborhood-dilated "100% hyperbolic within radius 3" is a weak test — a saddle pixel almost always
  exists somewhere in a radius-3 window; the AT-junction eigenstructure is the honest measure and it says
  the opposite.) `[macOS-MLX advisory]`

## 3. VERDICT — is the horizon the lane's high-d_H sibling? **YES (with a precise caveat).**

- **High d_H:** Road–Undrivable d_H = 4.45 (energy-wt 4.75) — the highest-d_H DYNAMIC edge after the lane,
  and #2 by leverage among non-lane edges. ✓
- **Big edge:** 18% of all boundary cracks (290k / n600), 3rd-largest by count, 2nd by leverage. ✓
- **It sits on the ground-plane vanishing line:** the fitted Road↔Undriv line is at **v_center = 188.8 rows**,
  within **3.2 rows (0.8% of image height) of the intrinsics principal-point row cy = 192.0**, slope −0.075
  (4.3° from horizontal), residual 8.1 rows RMS (line-like modulo road crest/curvature), coverage 0.77.
  This is exactly where the ground plane's vanishing line projects for the near-zero-pitch openpilot camera.
- **Does the ground-frame ξ chart generalize to it? YES — it shares the SAME calibration and ξ dynamics.**
  Precise statement: the horizon is the **vanishing line of the ground plane** — under the IPM (plane→bird's-eye)
  homography the ground plane's line-at-infinity maps to it, so it is NOT a finite ground-plane CURVE you
  rasterize the way lanes are (lanes are on the plane at finite depth). BUT its image row `v_horizon(pitch,
  height)` is fixed by the SAME ground-plane calibration that defines the lane's ground-frame chart, and it
  moves frame-to-frame with the SAME ξ ego-motion (pitch). **⟹ ONE vanishing-point-anchored ground-frame ξ
  chart covers Road + Lane + Undrivable = the entire scene above the hood** — the horizon coded as an
  analytic line `v = v_horizon(ξ)` driven by the shared ξ, not as 290k free boundary pixels. This is the
  openpilot unified-physical-prior (v_horizon FEEDs #325–327) realized as a rate carrier. The lane's
  ground-frame factorization is not lane-specific; it is a SCENE-ABOVE-HOOD chart, and the horizon is its
  cheapest, most-directional sibling.

## 4. TEMPORAL anisotropy (spatio-temporal structure tensor, n600)

- `d_H_spatiotemporal = 2.20`, eigs [0.080, 0.076, **0.0089**] — a clear near-NULL third eigenvalue
  (~9× below the top). The boundary field has a coherent spatio-temporal orientation = locally a
  1-parameter moving-edge structure = **the time axis is factorable** (a moving edge with consistent
  velocity is exactly what the ego screw ξ predicts). This is WHY ξ-factorization compresses time.
- **Honest caveat on the radial/tangential split (0.137):** this is NOT "motion is tangential." It reflects
  that most boundary LENGTH is lane (radial edges whose normal is tangential ⟹ aperture-blind to the radial
  ego expansion). The **radial-normal edges (horizon, hood) are the ones that carry the ego-expansion
  temporal signal**; the tangential-normal lanes see it only through the ground-frame chart. So ξ compresses
  time via TWO routes: directly for the radial-normal horizon/hood, and via the ground chart for the lanes —
  it is not a single-direction claim.

## 5. Routing summary (the deliverable)

1. **Horizon (Road↔Undriv)** — high d_H, big, shares lane geometry ⟹ **fold into the lane's ground-frame ξ
   chart** as an analytic `v_horizon(ξ)` line. THE #1 generalization of the factorization treatment beyond
   the lane. (Sister owns the lane carrier; this says the same chart should ALSO emit the horizon.)
2. **Hood (Road↔MyCar)** — highest d_H but STATIC ⟹ store-once static mask (#139); do NOT spend a ξ chart.
3. **Movable** — LOW d_H (isotropic blobs) ⟹ moment-ellipse / site-coder (v8 memo already routes this;
   confirmed factorization has low leverage).
4. **Saddles (17× hard mass)** — the FULL eigenstructure says the directional basis codes ~98% of
   junctions (rank-1 hyperbolic + elliptic bowls); route only the ~2% GENUINELY 2D-hyperbolic residual to a
   small 2-parameter junction / persistence-preserving code. The 17× concentration is a PRECISION lever at
   directionally-codeable loci, not a chart change. (This CORRECTS the naive "saddles defeat the directional
   lever" reading — measured eigenstructure refutes it.)

**Candidate law (MEASURED, not yet an equation):** factorization/SPD-cone leverage on the argmax boundary
ranks as `d_H × pixel_share × (1 − static)`, EXCLUDING rank-2 saddle loci (which need a 2-param code).
d_H alone over-ranks the static hood. Registering deferred until it is exercised by a byte-closed carrier.

## Triality
- **DAG:** FEED-anisomap (this memo + tool + artifacts).
- **DSL:** N/A — a measurement/routing map, no new lever wired.
- **equations:** candidate `factorization_leverage_ranks_dH_times_share_not_static` — MEASURED routing
  heuristic; NOT registered yet (needs a byte-closed carrier to confirm the win scales with d_H the way
  the pose SPD-cone did). Flagged, not asserted.

**Sisters (disjoint files):** lane-factorization agent (afa9d885) owns the LANE carrier — this memo measures
the OTHER edges + saddles + the horizon-generalization verdict; costate/per-class + v8 build agents untouched.
**Pointer 0.19108282 UNMOVED — an anisotropy map routes carriers; it moves no score.**
