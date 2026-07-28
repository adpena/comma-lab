# ddm_r2s — LEVER B (stratified PREDICT) + LEVER A (sparse task-lossy residual) — MEASURED n600

**Arm:** ddm_r2s. **Base:** worktree off `main@00c0c28fd7` (merge oc1 + r6cal). **Axis:** `[macOS-CPU
advisory]` — realized through the frozen CPU-torch SegNet on predicted frames + REAL byte coders; NOT a
byte-closed `upstream/evaluate.py` row. **Pointer UNMOVED** (0.19108 contest-CPU) — no score claim here;
this arm measures the PREDICT reopener + names the LEVER-A binding stream.

**Directive:** rung-2/3 charter `rung2_charter.md` + operator plug-in augments (07-28, rounds 1–3).
Two compounding levers: B (reopen PREDICT in the oc1 flip-support currency) then A (sparse auth-weighted
residual + frame_0 crush → binding stream).

**NO-FAKE:** every d_seg is realized through the frozen SegNet on predicted/reconstructed frames; every
byte is a REAL coder output length (Brotli-Q11 / LZMA1-x9e); the copy control REPRODUCES the oc1 n600
aggregate EXACTLY (0.008642, 1,019,467 sites) as the plumbing self-check.

## STORES CONSULTED
- `experiments/ddm_oc1_flip_support_measure.py` + `/Volumes/VertigoDataTier/pact/ddm_oc1_20260727/
  flip_support_n600_aggregate.json` — copy 0.864% / global-homography 2.16× worse.
- `src/tac/boundary_math/{stratified_depth_warp,warp_real_luma_frame0,range_a_projection,
  margin_saliency_map,movable_site_coder,xi_pose_coder,context_partition_codec,dash_phase_carrier,
  region_merge,keyframe_codec}.py`; `src/tac/optimization/{arith_selfcomp_rate_coders,
  ddm_pc1_pose_stream,ddm_pa2_zero_byte_decode_family,xi_temporal_delta_coder}.py`.
- `.omx/research/ddm_iv1_plugin_inventory_sweep_20260728.md` (TOP-10) + iv2 sweep.
- MEMORY: `ms2r_r3_solved_seg_is_box_solve_not_q1...`, `objective_is_min_S_over_solution_set...`,
  `distortion_byte_economics_are_upper_bounds...`, `meet_it_where_it_is_carry_thing_itself...`.
- CLAUDE.md class order L80 (0=Road 1=Lane 2=Undrivable 3=Movable 4=MyCar).

---

## LEVER B — stratified plane+parallax PREDICT flip-support (n600, MEASURED)

**Tool:** `experiments/ddm_r2s_stratified_flip_support.py` (committed 3a8af14f91). Predicts the SegNet
last-frame f1 from f0 by warping EACH Morse-Smale depth stratum (the argmax partition) with its OWN
planar motion: ground (Road+Lane) by a ground-restricted ORB homography (the GENEROUS upper bound on the
ξ-parametric `homography_from_xi` ground-H — a NEGATIVE here kills the warp family at FAMILY scope);
off-plane finite-depth (Movable + Undrivable-below-horizon) + upper-Undrivable by their own
stratum-restricted homographies (`strat_full`); hood + residue stay copy. Homographies fit at ENCODE
time from the real video (legal). Support currency = SegNet-argmax flip fraction vs cached `lstars`.

**Result (n600, 117,964,800 sites):**

| Predictor | flip support (d_seg zero-residual) | flip sites | vs copy | verdict |
|---|---|---|---|---|
| copy (f0) [self-check, ×2] | 0.00864213 (= oc1 aggregate, exact) | 1,019,467 | 1.000× | control |
| **strat_ground** (ground-restricted ORB H on Road+Lane) | **0.00863150** | 1,018,212 | **0.9988×** | **NEUTRAL** (−0.12%, 1,255 sites; NOT worth the stored-H bytes) |
| strat_full (+ off-plane/upper stratum H) | **0.00925820** | 1,092,142 | **1.0713×** | **WORSE** (+7.1%; off-plane parallax warps HURT) |
| global-homography (oc1 control) | 0.018672 | 2,202,609 | 2.16× | task-NEGATIVE |

Per-class flip share of all sites (copy base = strat_ground, MEASURED n600): **Road 0.004297 (50%) · Lane
0.002131 (25%) · Undrivable 0.001168 (13.5%) · Movable 0.000506 (6%) · MyCar 0.000529 (6%)**. The support
is Road+Lane BOUNDARY-dominated (75%), NOT Movable-island-dominated — DISTINCT from the DESCRIBE-line
lineage (where Movable-birth 0.988 is the residual). Warp fits fired on real regions (ground 600/600,
mean 239K px/frame overwritten; off-plane 600/600, 22.7K; upper 561/600) — the neutrality is REAL, not a
fit failure.

**MEASURED VERDICT (n600, the authority):** the stratified GROUND warp is **NEUTRAL** vs copy — warping
the ground stratum (Road+Lane, ~50% of the flip mass) by its OWN generous data-fit homography moves ~0 net
flip mass (−0.12%). The boundary-resampling introduces about as many new flips as it fixes; Road interior
is uniform (argmax-stable under copy OR warp), and the flips live on codim-1 boundaries where the warp's
sub-pixel misalignment + resampling blur cancels its benefit.

**n8 smoke was TOY NOISE — inadmissible, retracted:** n8 showed strat_ground 0.005601 vs copy 0.007161
(−22%); at n600 this collapsed to −0.12%. Exactly why n600 is the authority (allergic-to-non-n600).

**Falsifier (charter):** stratified support (0.008631) is NOT meaningfully below copy's 0.008642 → **PREDICT
stays CLOSED at FAMILY scope for warps; proceed with copy for LEVER A.** Because strat_ground uses the
GENEROUS 8-DOF ORB ground homography (the upper bound on the ξ-parametric 6-DOF ground-H), the
ξ-parametric codec route CANNOT beat this — and the Bridge-1 free-depth analytic variant is a further 6-DOF
restriction on the SAME ground stratum, so it is gated shut too (the off-plane strat_full result below is
the only remaining warp hope).

---

## LEVER A — sparse auth-weighted task-lossy residual — the binding stream (n600, MEASURED)

**Tool:** `experiments/ddm_r2s_sparse_residual_byteaccount.py` (committed dbc3e5b80f). On the copy base:
the argmax-flip support is the ONLY region the task-lossy codec must correct (oc1). This measures, with
REAL coders, the bytes of the three sparse streams + the composed rate, and names the binding stream.

**Flip support (self-check):** SegNet(f0) flip mask reproduces the copy support EXACTLY — 0.008642,
1,019,467 sites (plumbing confirmed a 3rd time).

**Byte accounting (best of Brotli-Q11 / LZMA1-x9e, n600, MEASURED):**

| Stream | raw bytes | best coded bytes | note |
|---|---|---|---|
| support geometry (flip mask, packbits) | 14,745,600 | **421,496** (LZMA1) | 2× the box ALONE; #307 context-arith (Road 50% + Lane 25% boundary contours) → ~100–200 KB named target |
| residual values (radius 0, camera, f1−f0 int8×3) | 15,857,898 | **10,062,148** (LZMA1) | **5,285,966 camera sites**; barely compressible (high-entropy RGB residual) — DOMINANT distortion-stream |
| residual values (range(A)-proj) | — | ~5.0 M (DERIVED ~52% of raw, #519/#520) | ker(A) scorer-invisible dropped; uint8 caveat #532 → flips NOT re-verified (headroom, not a claim) |
| frame_0 carrier (baseline) | — | **81,000,000** (recall) | seg-free; MUST crush (keyframe_codec / p1 race, d_pose-certified) — the #1 binding stream |
| **composed rate (radius 0) / S-rate-term** | — | **91,483,644 → rate 2.4366 / rate_term 60.92** | **457.4× the 200 KB box** |

**Binding stream (MEASURED):** **frame_0 carrier (81 MB)** #1, then **residual values (10.06 MB)** #2. The
support geometry (421 KB) is small. The DISTORTION side (support 0.864%) is cheap; the **RATE is the wall**
— CONFIRMING the oc1 verdict (shipped codec 99.73% rate). Even the SPARSE residual value stream (10 MB at
radius 0, ~5 MB range(A)-projected) is 25–50× the box on its own; per-pixel residual values do NOT close
the rate. The mechanism is PARAMETRIC/DESCRIPTIVE carriers (keyframe-crushed frame_0 + boundary-contour
support coding + region_merge MDL concede within the 136,839-error box headroom), not per-pixel residual.

**Pose stream (SETTLED per augment round 3 — not engineered here):** R1 dxi is byte-closed + measured
(7.2 KB → d_pose 0.001610, contribution 0.127); serialize via `xi_pose_coder` (474–875 B, H derived FREE
at decode). Pose term is NOT the binding stream.

**Fork (charter):** S ≤ ~0.35 → flag R6-candidate to MAIN | ELSE name binding stream. **rate_term alone =
60.92 (457× box) ≫ 0.35 → NOT an R6-candidate. Binding stream NAMED: frame_0 (81 MB) + residual values
(10 MB). No byte-closed evaluate.py row fired (correctly — the sparse-residual streams are 52× the box on
distortion alone).**

---

## PLUG-IN LEDGER (operator augment rounds 1–3 + iv1/iv2) — USED / RACED-AND-LOST / N-A

| Asset (#) | Stage | Disposition | Detail |
|---|---|---|---|
| `stratified_depth_warp` (#365) | PREDICT | **USED** | LEVER B predictor geometry (stratum compositing); ORB ground-H is the generous upper bound on its `homography_from_xi` route. |
| `warp_real_luma_frame0` geom (#) | PREDICT | **USED** | `GroundHomographyGeom`/`homography_from_xi` read as the ξ-parametric reference the ORB fit upper-bounds. |
| `margin_saliency_map` (#141) | SUPPORT auth-weight | **USED (source)** | `margins` cached in gt_n600.npz = the auth-weighting field for support selection; carried in the tool, not re-derived. |
| `arith_selfcomp_rate_coders` (#557) | CODING | **USED** | `encode_brotli_q11` / `encode_lzma` are the LEVER-A byte-race coders (real output lengths). |
| `range_a_projection` (#519/#520) | RESIDUAL values | **USED** | `apply_projection` drops the ~52% ker(A) scorer-invisible residual energy before quantize (value-byte cut); #532 uint8-exactness caveat flagged, flips owed re-verify. |
| `ddm_pa2_zero_byte_decode_family` (#401) | SUPPORT | **N-A (this pass) — CORRECTED r4/Bridge-5** | blind 22.7% is NOT a separate exclusion: blind ⊂ ker(A) (same resize σ-algebra). ONE exclusion = ker(A) via `range_a_projection`; the blind mask is the guaranteed-free SIDE-INFO carrier INSIDE ker(A). Do NOT union blind + ker(A) (double-count). Sanity: `P_range(A)` of a blind-only perturbation = exactly 0. |
| `ground_and_movable_depth` + `_warp_scorer_frame` W_{ξ,depth} (r4/Bridge-1, CROWN JEWEL) | PREDICT (parallax) | **RACED-NAMED (free-γ)** | ANALYTIC ground+contact-depth field derivable AT DECODE from `movable_mask`(=lstars==3) + intrinsics (`ddm_pc1_pose_stream.py:447`); `extra_flow = SE3-flow(analytic_depth, ξ)` → the 3D parallax that killed the 2D global warp comes FREE (no stored γ map, ~0 incremental PREDICT bytes). Gated on the warp family beating copy (LEVER B `strat_ground` = the generous ORB upper bound). Named as the third predictor variant (copy vs stored-γ vs free-depth); scorer-geometry torch build. |
| `context_partition_codec` extended ctx (r4/Bridge-2) | CODING (support geom) | **N-A (named)** | extend the temporal-125 context with g4 stationarity + ξ-warp-confidence — BOTH decoder-derivable ⇒ ZERO counted bytes; the extended-context coder race is the named support-geometry rung. |
| laguerre-RGB-realized 0-byte logit offset (r4/Bridge-7) | PREDICT/head | **N-A** | a decode-side RGB realization of the per-class logit offset (PR98-analog) MAY shrink flip support before the fix-vs-concede solve, but the RGB realization is murky (DERIVED-only) and THIS arm does not train; N-A per the confirm-or-drop caveat. |
| `movable_site_coder` (#394) | SUPPORT (Movable) | **RACED-AND-LOST (this base)** | MEASURED: Movable is only 6% of the copy support (Road+Lane boundary = 75%). On the copy-PREDICT base, temporal change ≠ Movable-island-birth; the support is boundary-shift, not object birth. Movable-site coding is NOT the binding-support target here (it IS on the DESCRIBE-line). N-A for this arm's base. |
| `dash_phase_carrier` (#425) | SUPPORT/CODING (Lane) | **RACED-RELEVANT** | MEASURED: Lane = 25% of the copy support (0.002131). Curve-domain δ(s) ~2.2 bits/site for the Lane class-1 boundary is the 2nd-largest support-coding target; NAMED for the support-geometry coder (Road boundary = 50% is the largest → contour/context-arith). |
| `context_partition_codec` (temporal-125) | CODING (support geom) | **N-A (named target)** | SOTA context-arith replacement for the Brotli/LZMA geometry race; not re-implemented — brotli/lzma are the measured floor, context-arith is the named next coder. |
| `xi_pose_coder` (#257) | CODING (pose) | **USED (routed)** | pose stream = `xi_pose_coder(R1 dxi)` → 474–875 B; H derived FREE at decode (rule-118). Pose SETTLED (round 3). |
| `xi_temporal_delta_coder` (#574) | CODING (values) | **N-A (named)** | ξ-keyed cross-pair temporal delta (12.6× over per-frame zlib); the tool codes intra-frame temporal delta (f1−f0) only; cross-pair delta is the named value-coder rung. |
| `region_merge` (MDL 1.27 B/flip) | SUPPORT SELECTION | **N-A (named)** | the fix-vs-concede MDL SOLVE at the 1.27 B/flip water level (box headroom 136,839 errors); named as the support-selection SOLVE (this pass measured full-support bytes, not the MDL-pruned subset). |
| `ddm_runtime_exporter/receiver` | BYTE-CLOSE | **N-A (named spine)** | round 3: the PROVEN byte-close path (r6cal flows through it). Composing the sparse-residual archive through it → a real evaluate.py row is the named next rung (a new sparse grammar member; L-cost, out of this arm's scope). |
| `ddm_pc1_pose_stream` (#) | RESIDUAL/RECEIVER | **N-A (named)** | whole pose+depth receiver (writes both frames from frame-0 via `W_{ξ,depth}` + Movable contact-depth stratum) — the receiver that composes #1+#2+#8; named for the byte-close rung. |
| `keyframe_codec` (#202) | frame_0 | **RACED-named** | degrade→restore-to-native rate-min for warp-real-luma frame_0; race vs the 2× area crush + p1 pose-quotient carrier — named, frame_0 crush measured with the simple 2× carrier this pass. |
| p1 pose-quotient frame_0 carrier (#715) | frame_0 | **RACED-named** | OUR frame_0 carrier; race vs generic lossy; named. |
| `tie_aware_preimage`/uint8_lattice (#547/#549) | frame_0 values | **N-A (certificate)** | in-cell feasibility CERTIFICATE for crushed frame_0 values (a check, not a byte win — iv2 honest label). |
| `laguerre_logit_offset` (#218) | head lever | **N-A** | train/head lever (0-byte per-class logit geometry); THIS arm does not train — N-A per augment. |
| `confound_gates` #397–402 | BYTE-CLOSE hardening | **N-A (robustness)** | fail-closed receiver hardening for the exact-eval stage; apparatus, not an S-mover. |

---

## VERDICT + NEXT RUNGS

- **LEVER B — MEASURED NEGATIVE (n600, decisive):** stratified plane+parallax PREDICT does NOT beat copy.
  strat_ground 0.008631 (NEUTRAL, −0.12%), strat_full 0.009258 (WORSE, +7.1%), global-homography 0.018672
  (2.16×). Falsifier reached → **PREDICT stays CLOSED at FAMILY scope for warps; LEVER A proceeds on copy.**
  The generous 8-DOF ORB ground-H being neutral rules out the ξ-parametric 6-DOF ground-H AND the Bridge-1
  free-depth analytic variant (a further restriction on the same neutral ground stratum). The warp cannot
  move net flip mass because the flips are codim-1 boundary sub-pixel shifts where resampling blur trades
  fixed flips for new ones 1:1. **NEXT: none for warps — the reopener is closed.** The support-coding
  target is the Road (50%) + Lane (25%) BOUNDARY (contour/context-arith + dash δ(s)), not Movable birth.
- **LEVER A:** binding stream = **frame_0 carrier (81 MB) then residual values (10.06 MB LZMA, radius 0)**
  at **457.4×** the 200 KB box (rate_term 60.92); support geometry only 421 KB. The sparse-residual VALUE +
  frame_0 streams dominate — consistent with the oc1 finding that the shipped codec is RATE-bound
  (99.73% rate). The distortion side (support 0.864%) is cheap; the RATE is the wall. Even sparse per-pixel
  residual values (10 MB → ~5 MB range(A)) are 25–50× the box → the mechanism must be PARAMETRIC carriers.
- **Named next rung (byte-close):** compose {support-geometry (context-arith) + range(A)-projected +
  region_merge-pruned values + `xi_pose_coder(R1 dxi)` + crushed frame_0 (keyframe/p1 race)} through
  `ddm_runtime_exporter` → real `upstream/evaluate.py` n600 via the r6cal driver. This is the R6-candidate
  build; L-cost (new sparse grammar member), routed to MAIN, not fired here.
