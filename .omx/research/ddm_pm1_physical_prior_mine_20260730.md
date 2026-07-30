---
schema: ddm_pm1_physical_prior_mine.v1
date_utc: 2026-07-30
arm: ddm_pm1 (physical-prior mine, operator-directed; industrializes ph2 prompt #8)
lane_id: "lane_ddm_pm1_physical_prior_mine_20260730"
research_only: true
score_claim: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU frozen-PoseNet advisory; per-pair realized through the real receiver + frozen PoseNet; composed byte-close + n600 evaluate gate OWED]"
operator_verbatim_1: "Stop waiting for my openpilot pointers — recursively fractally mine openpilot/upstream for EVERY physical prior and rank them as carriers."
operator_verbatim_2: "Also, photometric."
tool: "experiments/ddm_qa44_photometric_rungs_probe.py (commit 5cd3b88eac)"
data: "SSD ddm_qa44_20260730/{photometric_rungs_probe.partial.jsonl (25 aimed + full-tail-112), probe_run.log, probe_fulltail_run.log}"
sources_confirmed:
  - "upstream/{evaluate.py:92 score, modules.py:64-113 PoseNet/SegNet preproc, frame_utils.py:11-12,51,159-176 camera+YUV6+BT.601} (IMMUTABLE contest snapshot)"
  - "receiver .../ddm_pfs1_20260729/d1/submission/pfs1_warp_receiver.py (intrinsics 910/582/437, height 1.22, ground homography)"
  - ".omx/research/comma_openpilot_domain_tricks_20260619T035417Z.md (#156: camera/calib provenance, comma2k19 GT-pose unlock, kinematic null, preproc)"
  - ".omx/research/comma_openpilot_crossref_polynomial_geometry_20260619T014433Z.md (#145: lane-polynomial deg-4, comma10k classes, extrinsics pitch)"
  - "live physics wins: ddm_qa43 (two-plane -0.5754 S) + ddm_qa45 (static horizon v=437 BEATS GT masks) + ddm_ph2 convocation"
---

# ddm_pm1 — the physical-prior mine: recursive-fractal carrier enumeration + QA44 photometric rungs FIRED

**POINTER HONESTY FIRST: `0.1910828242 [contest-CPU]` UNMOVED.** Every number below is
`[macOS-CPU frozen-PoseNet advisory]`, `score_claim=false`, per-pair realized through the real
receiver + frozen PoseNet. This arm characterizes the pose member of an ADVISORY vehicle (the pfs1
warp base, S≈2.2566, far from the pointer) and mines its physical-prior carriers; it does NOT move
the pointer. The composed byte-close + n600 `evaluate.py` gate is OWED (v4b/v4c build).

## §0 What this arm did (two legs)

**Leg 1 — the MINE (centerpiece §2/§3):** recursively-fractally enumerated EVERY physical prior in
the openpilot ecosystem + the `upstream/` contest snapshot, organized by family (geometric /
dynamic / photometric / scene-structure), and ranked each AS A CARRIER against the meet-it-where-it-is
4-clause test. Split CONSUMED (already in the shipped/measured vehicle, with receipt) vs UNCONSUMED
(the value — the tail we have not yet spent). The industrialization of the operator's 2-for-2
openpilot pointers: stop waiting, mine the source.

**Leg 2 — the MEASUREMENT (§1, FIRED):** executed QA44's three photometric-physics rungs on the
realizable two-plane pose warp, aimed at the 17 hard-core two-plane losses + 8 deepest wins as
controls. **Result: PoseNet reads auto-exposure and rolling-shutter — the operator's "Also,
photometric" is a MEASURED, family-level pose carrier.**

## §1 QA44 PHOTOMETRIC RUNGS — FIRED (25 aimed pairs; full-tail-112 confirmation running)

The image-formation ORDER is `motion → per-depth projection → photometric response → rolling-shutter
→ uint8`; the shipped receiver implements ONLY the projection stage (the two-plane warp). QA44
measures the next two stages. Base = the qa45 REALIZABLE static two-plane (far = rows < v=437 → H∞,
ground → full H, hood → identity; 0 bytes) at each pair's qa43 GT-optimized `p_two_star` (advisory;
a static re-solve is owed at v4b, so ABSOLUTE totals are an upper bound — the RUNG DELTAS are the
measurement). PoseNet reads YUV6 of BOTH frames → every rung is scorer-visible by construction.

| rung | mechanism | price | HARDCORE-17 | WIN-CTRL-8 | verdict |
|---|---|---|---:|---:|---|
| **B auto-exposure** | `f0 := a·warp(f1)+b` after warp, before uint8 (camerad AE changes between consecutive frames); (a,b) GN-solved at fixed pose | **~4 B/pair** (2×f16) | **improves 17/17, degrades 0** | **improves 8/8, degrades 0** | **FIRED — FAMILY-level pose carrier** |
| **A rolling-shutter** | rotation sheared linearly across rows (yaw row-shear), mean rotation preserved; magnitude = global physical const, SIGN free from shipped ξ yaw | **~0 B** | improves 12/17, degrades 0 | improves 8/8, degrades 0 | **FIRED — ~free, wins on turns** |
| C Movable third plane | route Movable (GT class-3 mask, UB) through full H at per-pair effective depth s_t_mov (GN, 1 param) | ~2 B/pair | improves 10/17, **degrades 6** | improves 4/8, degrades 1 | FIRED — INSTANCE, needs selector; dominated by B |

**Composed selection Σmin(single-fallback) on the 25 aimed pairs (realizable, advisory):**
qa45 static base **14.599 → 11.187 with rungs A+B (−3.412, −23.4%)**; excluding the two
divergent GT→static transfer pairs (222/140, capped by single-fallback anyway) 13.646 → 10.500
(−23.0%). **The 8 deep-win controls all get DEEPER, not degraded** (e.g. pair 46: 0.22 → **0.0003**;
175: 0.26 → **0.0009**; 19: 0.038 → **0.0010**) — auto-exposure was an unmodeled residual even on the
pairs the geometry already "won". Which rung drives each pair's best: **B=17/25, C=5, A=3.**

**The mechanism is real (per-rung falsifiers did NOT fire):**
- **Rung B — camerad auto-exposure.** 23/25 pairs solved a non-trivial `(a,b)`; the shipped warp
  copies frame_1's exposure onto the reconstructed frame_0, but the REAL consecutive frames differ in
  exposure (camerad AE gain/bias re-tunes frame-to-frame), and PoseNet — a photometric-differential
  instrument reading YUV6 of BOTH frames — sees the mismatch. Correcting it is the cleanest, most
  universal rung measured on this vehicle. This is the operator's "Also, photometric," MEASURED.
- **Rung A — rolling shutter.** Winning β splits ±1.0 (11 pairs +1, 7 pairs −1): the shear SIGN
  tracks the turn direction = the sign of the shipped ξ yaw dim (FREE), so the magnitude is one global
  physical constant (readout/inter-frame ratio) → **~0 counted bytes.** 0 degradations anywhere.
- **Rung C — Movable third plane.** Real but weak and MIXED (degrades 6/17): cars are not on the
  ground plane, but a per-pair depth scalar only helps where Movable content is both present and
  mis-warped; it needs a per-pair selector and a realizable Movable mask (decoded partition). Rung B
  dominates it. INSTANCE scope; defer to QA48 (plane+parallax) which subsumes it.

**#404 (advisory) — FULL-TAIL-112 MEASURED (receipt sha 025898218825fdb8):** rung B improves
**112/112** tail pairs, 0 degradations; rung A improves 96/112, 0 degradations. The full-tail pose
member (Σmin(single), non-tail 488 held at P0 sum 8.6199): qa45 static base tail-selection 22.734
(contribution 0.7229) → **rung B only 15.040 (0.6280, −0.0949 S) → rungs A+B 14.365 (0.6189,
−0.1039 S)**. **So the photometric axis is a MEASURED −0.1039 S at ≤4 B/pair (rung B) + ~0 B (rung
A), ON TOP of qa45's −0.7683 — a NEW orthogonal (photometric) axis, 2.5× the 25-pair projection**
(the projection was conservative; the full tail has many more improvable pairs). **Grammar impact:
v4c gains 2 photometric coeffs/pair (rung B) + a 1-line free rolling-shutter receiver amendment
(rung A); rung C stays deferred.** Absolute totals remain an UPPER BOUND (GT-optimized poses, no
static re-solve); the composed byte-close + n600 evaluate gate is OWED at v4c.

## §2 THE CARRIER MINE — CONSUMED (already in the shipped/measured vehicle, with receipt)

Redundancy discipline (meet-it-where-it-is 4-clause: scorer-visible × sensitivity-priced × compact ×
non-redundant): many of the strongest physical priors are ALREADY spent. Marking them CONSUMED with
the receipt is the point — it stops re-proposal and isolates the UNCONSUMED tail (§3).

| # | prior (family) | source (file/receipt) | how it's consumed | axis | receipt |
|---|---|---|---|---|---|
| C1 | Camera intrinsics K=[910,0,582;0,910,437] (GEO) | frame_utils.py:11-12; camera.py `_neo_config` | receiver `intrinsics_native()` | geo base | pfs1_warp_receiver |
| C2 | Camera height 1.22 m (GEO) | calibrationd.py:6; HEIGHT_INIT | `CAMERA_HEIGHT_M` in `pose_to_homography` | geo base | pfs1 |
| C3 | Ground-plane vanishing row **cy=437** (GEO/SCENE) | K → l=K⁻ᵀ[0,−1,0] | qa45 static far/ground split (0 bytes, DERIVED not tuned; ±40 rows collapses = positive control) | seg/pose | ddm_qa45 |
| C4 | Ground-plane homography H=K(R−t·nᵀ/d)K⁻¹ (GEO) | camera.py ground homog | the entire warp base | pose | pfs1 D1 |
| C5 | Far/ground/hood 3-region partition (SCENE) | comma10k classes + horizon | two-plane per-class warp | pose | ddm_qa43 |
| C6 | Near/far parallax (two-plane) (GEO) | operator pointer 07-29 | qa43 −0.5754 S measured | pose | ddm_qa43 §5 |
| C7 | Hood static region (class-4 my-car, bottom) (SCENE) | comma10k my-car; #139 | warp identity for hood | seg/pose | pfs1 receiver |
| C8 | Roll ≡ 0 (image not roll-corrected) (GEO) | calibrationd.py:6 | n=[0,−cos,−sin], no roll DOF | pose | pfs1 |
| C9 | Pose 6-dim = se(3) ego screw [v_fwd,v_lat,v_vert,ω_r,ω_p,ω_y] (DYN) | fill_model_msg.py:186-191 | the warp pose is this screw; GN solves it | pose | pfs1 D2 |
| C10 | Pose effective low-rank (dim-0 forward-speed dominates, rank-1 98%) (DYN) | pose stats; measured | GN finds it; e_p rank-measured | pose | pfs1 D2 / uh1 |
| C11 | YUV6 BT.601 preproc (4 luma phases + 4:2:0 chroma, both frames) (PHOTO) | frame_utils.py:51,159-176; modules.py:74 | the scorer path itself; the reason B/A are scorer-visible | pose | upstream |
| C12 | Chroma <2px INVISIBLE / 4:2:0 sub-sample (PHOTO) | modules.py YUV6; frozen_scorer_exact_factorization | seg token-stream rate lever (burn) | rate | that memo |
| C13 | SegNet reads LAST frame only, unnormalized (SCENE/PHOTO) | modules.py:108 | frame_0 is seg-free → the free-frame_0 pose carrier | seg/pose | frozen_scorer_exact_factorization |
| C14 | Out-of-gamut clip (.clamp 0,255) free (PHOTO) | frame_utils yuv420_to_rgb | known exploit (PR95 sigmoid·255) | rate | #156 §4 |
| C15 | 20 Hz cadence, 1200 frames = 600 pairs (DYN) | arXiv 1812.05752 | context; pairing | — | #156 §0 |
| C16 | Sky/top-band ≈ undrivable high-margin (SCENE) | homography + comma10k | seg dominated-rung; far region | seg | #156 §1 |
| C17 | Road trapezoid geometrically bounded (SCENE) | homography | seg interior prior | seg | #145 |

## §3 THE CARRIER MINE — UNCONSUMED (the tail; the value) — RANKED

Ranked by (measured-or-derived magnitude × cheapness × non-redundancy). The 4-clause carrier test
column: SV=scorer-visible · SP=sensitivity-priced · CO=compact · NR=non-redundant vs shipped streams.

| rank | prior (family) | source | DERIVES FREE at decode | replaces/cures | axis | 4-clause | magnitude (#404) | $0 falsifier | status |
|---:|---|---|---|---|---|---|---|---|---|
| **1** | **Auto-exposure gain/bias per pair (PHOTO)** | camerad AE; QA44-B | nothing free — 2 coeffs/pair COUNTED (~4 B) applied `a·warp+b` | cures the exposure-mismatch residual the warp cannot (copies f1 exposure) | pose | SV✓ SP✓ CO✓(4B) NR✓ | **MEASURED −0.041 S proj @ ~4B/pair; 25/25 improve** | done (FIRED §1) | **MEASURED — v4c grammar +2 coeffs/pair** |
| **2** | **Rolling-shutter row-shear (PHOTO/GEO)** | rolling-shutter sensor; QA44-A | row-dependent rotation from shipped ξ yaw × global readout const; SIGN free from ξ | cures within-frame yaw shear the single homography can't | pose | SV✓ SP✓ CO✓(~0B) NR✓ | **MEASURED 20/25 improve, 0 degrade, ~0 B** | done (FIRED §1) | **MEASURED — free receiver amendment** |
| 3 | ξ trajectory / bicycle-model B-spline coding (DYN) | POLY_PATH_DEGREE=4; se(3) B-spline; QA52 | receiver expands a few se(3) spline knots → 600×6 field | 7.2 KB pose field → ~1–2 KB; AND solve-conditioning (easy→hard transfer, anti speed-for-turn aliasing) | rate+pose | SV✓ SP✓ CO✓ NR✓ | rate −0.004 minor + conditioning (24% tail in runs≥4) | spline re-solve degrades tail d_pose > byte saving | UNCONSUMED (QA52 DUE) |
| 4 | Movable inverse-depth third plane (GEO) | QA44-C / QA48 (Irani-Anandan) | per-pair depth scalar for Movable region | cures cars-not-on-ground-plane | pose | SV✓ SP~ CO✓(2B) NR~ | MEASURED weak/mixed (10/17, degrades 6) — needs selector+realizable mask | done (FIRED, INSTANCE) | UNCONSUMED-WEAK (→QA48) |
| 5 | Mount pitch global refinement (GEO) | calibrationd.py pitch∈[−0.09,0.17]; #145 extrinsics pitch −0.02 | receiver ground normal n=[0,−cos p,−sin p] with p≠0 | refines the ground-plane warp (receiver ships pitch=0) | pose | SV✓ SP~ CO✓(~0B) NR~ | UNMEASURED, ~0 B (1 global const) | pitch sweep {−.03,0,.03} on ground warp: no pair improves → pitch=0 optimal | UNCONSUMED — $0 probe candidate |
| 6 | comma2k19 GT ego-motion arrays (answer key) (DYN) | #156 §2; public global_pos/ for b0c9d2329… | verification oracle + smoothness prior for QA52 | prices the pose floor; seeds trajectory code | pose(prior) | SV✗(oracle) SP✓ CO✓ NR✓ | prior only (NO-FAKE: contest scores PoseNet on OUR frames, not comma2k19 pose) | fit code to GT, measure d_pose it produces | UNCONSUMED-as-oracle |
| 7 | Kinematic null coding (4 of 6 dims fixed maps of v_fwd) (DYN) | #156 §5; pose_kf.py Q-ranks | v_vert=−v_fwd·tan(pitch), v_lat≈0, ω_r≈0, ω_p≈0 | shrinks the pose field's coded DOF for QA52 rate | rate | SV✓ SP✓ CO✓ NR~ | folds into QA52 (GN already exploits implicitly) | code 2 DOF + fixed maps; d_pose vs full 6 | UNCONSUMED (→QA52) |
| 8 | Lane-polynomial deg-4 geometry (road/lane as coeffs) (GEO) | #145; POLY_PATH_DEGREE=4 | receiver rasterizes lane curve from a few coeffs | seg-frame lane carrier (38.7% of flip mass) | seg | SV✓ SP✓ CO✓ NR✓ | UNMEASURED on this vehicle | lane-coeff raster vs shipped lane cells at matched bytes | UNCONSUMED (→QA28 lane pool) |
| 9 | Per-channel (luma-only) exposure gain (PHOTO) | QA44-B refinement; YUV6 | luma-only a·Y+b (PoseNet reads luma parallax) | tests whether rung-B win is luma-driven | pose | SV✓ SP✓ CO✓ NR~ | UNMEASURED; rung-B variant | luma-only vs RGB gain: no delta → RGB fine | UNCONSUMED — $0 probe candidate |

## §4 Families measured / assessed EMPTY (do not re-mine)

- **Bayer/demosaic artifacts (PHOTO):** EMPTY for our path — the contest goes through an RGB-uint8
  intermediate (`yuv420_to_rgb → rgb_to_yuv6`), NOT raw Bayer; sub-Bayer structure is unreachable and
  unscored (#156 §4 caveat). Do not mine.
- **Sensor noise model / HEVC source statistics (PHOTO/rate):** CONSUMED-as-finding, not a carrier —
  matching STATISTICAL texture (not exact HEVC blocks / noise) is sufficient; the archive holds the
  INR not a re-encode (#156 §6). LOW/empty as a byte-lever.
- **Vignetting / lens shading (PHOTO):** LOW prior — camerad ISP largely corrects it; a static radial
  gain would be a spatial rung-B variant, dominated by the per-pair global gain (rung B). Not worth a
  slot unless rung-B residual shows a radial pattern (it did not, in the 25-pair (a,b) fits).
- **Headlight/shadow dynamics, sun/sky radiometry (PHOTO):** data-dependent, not codeable as a global
  prior; scene-specific. EMPTY as a physical-prior carrier.
- **Mount yaw window ±0.069 rad (GEO):** effectively CONSUMED — the vanishing point sits within a few
  px of the principal point; the warp rotation already carries it. No separate carrier.

## §5 ROUTING (defer-at-source; same commit)

- **QA44 ledger row → FIRED** (results §1). New grammar consequence: **v4c pose grammar gains rung-B
  2 photometric coeffs/pair (~4 B) + rung-A free rolling-shutter receiver amendment**; rung C deferred
  into QA48.
- **New ledger rows** for the top UNCONSUMED $0-probe candidates: **QA53** (mount-pitch global
  refinement, rank 5) · **QA54** (per-channel/luma-only exposure gain, rank 9, rung-B variant, $0,
  DUE). Ranks 3/6/7/8 already have homes (QA52 / oracle / QA52 / QA28) — folded, not duplicated.
- **QA53 FIRED 07-30 → NULL** (mount-pitch, full-tail-112 transfer, receipt sha 671a947ef49b7e4e):
  **pitch=0 IS the aggregate global optimum** (tail-sel 22.734 / contribution 0.7229); every non-zero
  global pitch is worse (p+0.02→25.49, p−0.02→84.46). Per-pair scatter (57/112 prefer +0.02/+0.04) is
  not capturable by one free constant, and a per-pair pitch costs bytes. VERDICT: the ground-plane
  calibration is already optimal at pitch=0 — a clean honest negative (constants-are-poison clean, not
  a missed lever). Confound: transfer only (pose solved at pitch=0); a pitch≠0 re-solve is a v4c option
  but NOT free. This RESOLVES mine rank 5 to NULL and removes it from the UNCONSUMED tail.
- **Full-tail-112 QA44 confirmation** running detached (upgrades the −0.041 S projection to a
  measured full-tail pose-member number); §1 table updates at completion (receipt appended).
- **Standing (PH-4 / QA50):** the residual-axis loop's next firing = SVD the post-rung residual on the
  17 hard core — after rung B removes the exposure axis, is the remaining tail residual the geometric
  parallax field (QA48) or still photometric? $0, owed to QA50.

## §6 Confounds + discipline

- **`tac` import HIJACK control-guarded (QD15):** ran with `PYTHONPATH=$PWD/src:$PWD/upstream:$PWD/
  experiments` → `import tac` resolves to main `src/tac`, NOT the eg1 codex worktree. Positive control:
  the rung-ctrl (static two-plane at p_two_star) reproduces the qa45 h437 numbers per pair (substrate
  identity), so the runtime decode is the qa45/qa43 instrument.
- **Advisory everywhere:** frozen-PoseNet, macOS-CPU, non-promotable, `score_claim=false`. Absolute
  totals are an UPPER BOUND (GT-optimized poses transferred, no static re-solve); the RUNG DELTAS +
  the win/degrade counts are the measurement. The two divergent transfer pairs (222/140) are capped by
  single-fallback and excluded from the honest-subset number.
- **Verdict scope:** rung B = FAMILY (photometric is a real pose carrier on this vehicle, 25/25);
  rung A = FAMILY (~free, 20/25, 0 degrade); rung C = INSTANCE (mixed, needs selector). Full-tail
  extension + the composed byte-close + n600 gate are OWED (v4c build).
