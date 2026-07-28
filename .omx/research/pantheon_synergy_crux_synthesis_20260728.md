# PANTHEON SYNERGY + CRUX SYNTHESIS — the task-lossy ego-scene codec (2026-07-28)

`research_only=true` · `score_claim=false` · `promotion_eligible=false` ·
`evidence_axis=[recall/synthesis over receipts; advisory rows labeled]` ·
best submittable exact row `0.19108 [contest-CPU]` — **0.019 BEHIND the 0.172 bar**
(effective bar = min(0.15, official leaderboard best 0.172 [PR130 display]); routing card §9).
Operator charge: find THE synergy and THE crux against the frozen contest space; operator steer
(memory `codec_archetype_mpeg4_object_x_netflix_percontent_x_robotics_worldmodel_task_lossy_20260728`)
names the codec form; operator amendment binds every section to **n600** (all 600 pairs, real gt).

## THE SYNERGY (the one representation all lenses converge on)

**The task-lossy ego-scene codec: ONE world-program W = { scene atlas (sprite, image-chart) ·
ξ(t) ego-screw (dual-use) · sparse dynamic events (VOPs) }, decoded by the FREE inflate.py
interpreter into the ONE 384×512 scorer-input plane both heads read, quantized in-cell on the
uint8 lattice.** This is MPEG-4 sprite/GMC/VOP coding made LOSSY-TO-THE-TASK (distortion =
E-cell violation, never PSNR/VMAF), Netflix per-content RD optimization retargeted from VMAF to
the frozen E's measured sensitivity atlas, and the robotics world-model (map + ego-trajectory +
tracked objects) as the state being coded. The coupling is MEASURED, not metaphor: SegNet argmax
and PoseNet-6 are two projections of the SAME bilinear-resized plane (`A_seg ≡ A_pose → (512,384)`,
`frozen_scorer_exact_factorization_20260715`; the 07-19 lattice solve pinned that one plane and
bought BOTH heads at once — d_seg 0.0, d_pose 9.3e-10 [MEASURED n6→n600 chain, real DistortionNet]).
So the minimal sufficient statistic is a description of ONE object; term-wise composition
(staple seg-object + pose-stream) and storing solved frames are anti-compression [MEASURED:
raw exact realization 409.5 MB → S 272.7 advisory; the compact-vs-raw span is ~2,000×, routing
card §2]. n600 support: 98.806% of residual flip mass is image-stationary (G4), joint debt is
BROAD not heavy-tailed (G3 top-10 pairs = 1.98%) — amortize ONCE across the 600-pair drive.

## THE CRUX (one wall, three measured faces, n600-localized)

**The in-cell REALIZATION operator in the right chart — turning scene-coordinate descriptions
into uint8 frames whose E-cells hold, without the describe-then-repaint loss.** MEASURED faces:
(1) **Chart face** — the BEV/ξ-warp does NOT place ground strata at scorer precision: n600 p50
ruling residual Road 39.0 px, Lane 47.1 px, ≤1 px fraction 4.3% (`bev_staticity_v2`, D1/D2
NEGATIVE at the calibrated chart) — while the SAME strata are 97.8–99.1% stationary in IMAGE
coordinates (G4). The one-program hypothesis holds statistically in the image chart, not
geometrically in BEV. (2) **Paint face** — description→RGB regeneration loses 159×
(is1: exact solve 17,927 errors vs best described base 2,845,843; v14: mask promise 2.83e-4 →
0.0275 after fixed-prototype RGB projection). (3) **Direction face** — at current endpoints NO
measured single-coordinate move is downhill in joint S (j12: 16 sealed singles active-zero;
pf3: 16/16 uphill; pf3b: first distortion-downhill edge rate-dominated, +860 B vs ~162 B
break-even) — only JOINT/composed moves pay (cb1 MyCar carrier +319 B → ΔS −0.0516; wf7 −1,776 B
rate-only [macOS-CPU advisory]). Sharpest attack (all lenses agree): **solve IN description
coordinates** — Gauss-Newton/CG directly over {atlas, ξ, event} parameters using the custodied
exact linearizations (rank-4 SegNet head law + ≤6-dim Pose quadratic), uint8-STE and the real
coder INSIDE the objective, Fisher-margin trust region, so every iterate is by construction
~130 KB and receiver-closed. is1 ranked this family (d) prospective-#1; it has never been built.

---

## §1 The frozen information space (what everything optimizes against)

- E = frozen deterministic map: SegNet (smp.Unet tu-efficientnet_b2, rank-4 linear head
  [MEASURED, `segnet_head_rank4_linear_flipdist_v1`]) on last-frame RGB at 512×384 → argmax
  d_seg; PoseNet (FastViT-T12) on two-frame YUV6 → 6-dim MSE d_pose. Both consume the SAME
  resized plane (resize-first, `upstream/modules.py:71-75`) [MEASURED].
- S = 100·d_seg + √(10·d_pose) + 25·B/37,545,489 (`upstream/evaluate.py:92`). inflate.py is
  FREE and unsized (rule-118); only video-derived payload counts.
- Bar: min(0.15, 0.172). #613 representation gate: ≤200 KB, d_seg ≤ 0.00116, d_pose ≤ 0.00161
  [DERIVED gate, is1]. Sub-0.15 at the solved distortion point ⟺ B ≤ 154,522 [DERIVED, is1].
- Reachability existence proof: PR130 (d_seg 2.966e-4, d_pose 2.331e-5, 191,052 B) = 0.1721
  [contest-CUDA external display; existence only, banned lineage, zero design transfer].
- Distortion is SOLVED; description is not: exact uint8-lattice solve d_seg 1.51969e-4
  (17,927/117,964,800), d_pose 1.02e-4, survives real R + uint8 [MEASURED, is1/rd1 receipts].
  The box has 7.6× error headroom (136,839 allowed vs 17,927 exact) [MEASURED, card §4].

## §2 n600 EVIDENCE BASE — where the one-program hypothesis holds and where it breaks

All rows below are across ALL 600 pairs, real gt (no subset promoted; G3's own subset contract:
top24/full r=0.595 prohibits subset-only conclusions).

**Holds (the amortizable core):**
- **98.806%** of the 4,011,236 exact flip events (v12-base residual field) recur at the SAME
  512×384 pixel with the SAME class transition ≥2× across pairs [MEASURED, G4 #623]. Top 1% of
  pixels = 21.2% of flip mass; top 10% = 89.9%. The stationary bands ARE the scene-through-
  fixed-camera geometry: horizon/road-edge bands, the two Lane-corridor edges, the Movable
  mid-band (rows ~174–215), the hood rim.
- Hood/MyCar: temporal IoU 0.994; BEV-probe hood control p50 0.0 px, 91.3% ≤1 px at n600
  [MEASURED, bev_staticity_v2 D0 PASS] — the truly rigid-static stratum, and the first admitted
  strictly-joint-improving carrier (cb1: +319 B → Δd_pose −0.1795, ΔS −0.0516 [advisory]).
- Joint debt is BROAD: top-10 pairs carry 1.98% of joint mass, top-100 carry 18.7% [MEASURED,
  G3 #622]. No per-pair hero fixes; the win is shared structure amortized across the drive.
- Road–Lane interfaces = 61% of all boundary content [MEASURED, MPEG-4 CAE intake] — the bulk
  mass lives exactly where the sprite/atlas arm operates.

**Breaks (the crux-localizing complement, with fraction + mass):**
- **Chart break (the deep one):** in the calibrated absolute BEV chart, Road p50 residual
  39.0 px / Lane 47.1 px, only ~4.3% within the 1 px floor [MEASURED n600, bev_staticity_v2;
  verdict_scope: this chart + registered floor]. The ξ-proxy links just **2 of 47,882**
  singleton flip events [MEASURED, G4]. So ξ-advection does NOT currently supply
  scorer-pixel boundary placement — placement is counted content (RG3 class-birth 0/10:
  generic receiver geometry cannot supply video-specific interface placement free [MEASURED]).
- **Movable band** (independent agents; ξ cannot predict them by physics): 1,083,972 flips =
  **27.0%** of flip mass; BEV Movable p50 10.8 px [MEASURED]. This is the VOP arm's workload.
- **Transients** (single-occurrence events): 47,880 flips = **1.19%** [MEASURED, G4].
- **Timeline covariate spikes** (G3 proxies): lead_car_pass pair 452 (rank 33, visually
  confirmed), intersection pair 279 (rank 133), lane_change pair 286 (rank 204) [MEASURED
  covariates; semantic names not promoted]. Even the worst pairs (523, 54, 1, 90…) carry only
  ~0.086–0.091 joint mass each — turns/dynamics degrade gracefully, they do not dominate.
- **Pose mass is everywhere:** per-pair d_pose 178–205 at the v12 advisory base — pose debt is
  uniform across n600, and the exchange law dS/dd_pose = 5/√(10·d_pose) (0.124 @163 → 158 @1e-4)
  makes the pose endgame a near-exactness game [MEASURED/DERIVED, card §1].

**Reconciliation (the synthesis fact):** the scene decomposition n600 actually supports is
**image-chart stationarity + sparse innovations**, not literal BEV re-projection. For THIS
drive (forward motion, mostly straight, fixed camera), the projection of static world structure
is statistically fixed in image coordinates; ξ enters (a) as the pose payload itself (dual-use)
and (b) as the innovation PREDICTOR for the Movable band and transients — exactly MPEG-4's
sprite+GMC split, but with the sprite charted in the image plane and the warp demoted from
"places boundaries" to "predicts context" (CAE crosswalk §4b: ξ-warped prediction confidence as
coding context; raw label-diff temporal factorization measured 3.5× WORSE — temporal structure
must ride geometry).

## §3 PER-LENS MINING (each lens: minimal sufficient representation + where it locates the crux)

- **Kolmogorov/Solomonoff/Schmidhuber** [EXTERNAL: Solomonoff 1964; Schmidhuber OOPS 2004,
  speed prior 2002]: minimal statistic = shortest E-accepted program; S_min = 25·K/37.5M once
  distortion is in-cell. Brackets MEASURED: MDL(MS) upper ~236 KB (contour+Brotli) vs 154.5 KB
  strict line vs S_floor 0.11797 ⇒ K ≈ 177 KB-class if the floor is tight [eureka §5.5]. Crux:
  the description LANGUAGE (what the generic prior ignores: scene structure).
- **Shannon/Rissanen/MacKay/Willems/Duda** [EXTERNAL: Rissanen 1978 MDL; Willems CTW 1995;
  Duda ANS 2013]: the unmeasured tier-moving scalar is **H(flip-field | free decoder context)**
  — the 405.5 B/error greedy price is a channel upper bound only [MEASURED law, economics
  07-24]. G4's real KT rows: context-free 25.25 Mbit; causal per-pixel KT 12.34 Mbit (51%
  ideal); real generic traversal saved 18.2% — entropy proxies ≠ real coder (the
  boundary-distance proxy REJECTED despite 41.9% ideal) [MEASURED]. Crux: real-coder-in-loop.
- **Neural/overfitted codecs — Ballé/Minnen, Cool-Chic/Ladune, COIN++/Dupont** [EXTERNAL: Ballé
  2017/18 hyperprior; Ladune Cool-Chic 2023; Dupont COIN++ 2022]: per-content overfitting is
  the legal super-power here (one video, frozen metric). Their lesson: fit the DECODER-side
  representation with rate term in-loss — exactly "solve in description coordinates." Our
  measured twin: the 83,838 B compact generator beat direct-plane storage by orders of
  magnitude (representational lesson only, wrong objective) [MEASURED, is1].
- **Video codecs — Sikora/MPEG-4, Girod/Wiegand, Taubman** [EXTERNAL: ISO/IEC 14496-2 sprite/
  GMC/VOP + CAE; Sikora 1997]: the archetype itself. Sprite = amortized static scene (G4's
  static field: 4,107 B raw-LZMA reaches 920,921 cell events = Δd_seg 0.0078 cell-space
  [MEASURED]); GMC = ξ; VOPs = Movable band; CAE with ξ-confidence context = the shape coder
  [intake memo, built crosswalk]. Their hard-won law transfers: temporal deltas ride geometry,
  never raw label diffs [MEASURED external + our CAE crosswalk].
- **Netflix per-title/dynamic optimizer** [EXTERNAL: Aaron et al. 2015; Katsavounidis dynamic
  optimizer 2018]: shot-aware convex-hull RD per region against the METRIC. Swap VMAF → E-cell
  distortion, shots → G3 covariate segments, ladder → per-stratum tolerance (the box's 7.6×
  headroom is EXACTLY the "which tolerance to sell for which bytes" axis; rd1 λ-continuation
  re-read [MEASURED headroom, DERIVED framing]). Waterfill precedent in-tree: KKT
  `boundary_routing.py`; non-additive pools law binds (co-measure, never sum).
- **Robotics/VO/SfM — Hartley-Zisserman, Davison, Engel, Triggs, Scaramuzza** [EXTERNAL: H&Z
  2004 homography; Engel DSO 2016; Irani-Peleg mosaics 1995]: the world-model state = map +
  trajectory + objects; bundle adjustment = joint refinement of ALL of it against reprojection
  — the exact analogue of our joint solve (and the anti-analogue of single-coordinate edits,
  which our j12/pf3 measured dead). Crux located: the CHART — our BEV probe measured the naive
  ground-plane chart fails at 1 px (39–47 px p50); robotics' answer is photometric/direct
  alignment in the IMAGE plane (DSO), which is where G4 says our stationarity lives.
- **Costate/optimal control — Pontryagin, Bellman**: λ = marginal S per unit payload = the
  live costate organ (#247) + pf3b's break-even arithmetic; fire rules D3 already implement
  Hamiltonian switching (fire descent iff measured ΔS/hour beats priced ΔS/hour) [in-tree].
  Crux located: prices must be finite and same-object (the §5 materialization layer, mostly
  closed by pf3; direction layer remains).
- **PDE — Osher-Sethian, Crandall-Lions, Cole-Hopf** [EXTERNAL: Osher-Sethian 1988; Crandall-
  Lions 1983]: the partition is a level-set object; boundary moves are normal-velocity fields;
  viscosity solutions are the stable chart for argmax interfaces (our witness line's math).
  Crux located: island BIRTHS (class birth 0/10 MEASURED) — topology change needs explicit
  sources (VOP events), not smooth advection.
- **Geometry — Amari/Fisher-Rao, Brenier/Villani OT, Aurenhammer** [EXTERNAL: Amari 1998;
  Brenier 1991 (crosswalk in-tree); Aurenhammer 1987 power diagrams]: the argmax partition of
  a linear head IS a power/Laguerre diagram — store GENERATORS, not pixels (v8 rate-falls-out
  law [in-tree]); Fisher-margin metric = where bytes buy flips (hb1 just built per-stratum
  BN-capacity addresses for the FISHER_MARGIN_SITE_LOCAL family — one of rg4's 3 named missing
  coordinate families now has a builder [MEASURED, 07-27]). Trust region = Fisher ball.
- **Topology — Morse-Smale, Edelsbrunner-Harer** [EXTERNAL: persistent homology 2002]: lane
  dashes/birth-death pairs are the low-persistence tail that erasure kills first [MEASURED,
  witness era]; persistence ranks which events are worth VOP bytes.
- **HOPE/Hilbert rank-1 (Mobahi-Bartlett)** [EXTERNAL arXiv 2607.21366, sealed crosswalk]:
  capacity-weighted per-unit addressing on the frozen net; consumed with rate denominators
  OWED (no parameter-count-as-rate fake) [MEASURED custody, hb1].

**Convergence check:** every lens lands on the same object — a compact generator of the ONE
scorer-input plane, parameterized as {static scene atlas in the image chart, ξ(t) screw,
sparse events}, fit JOINTLY against exact E with the real coder in-loss, tolerance-managed
per-region. No lens dissents on the object; they dissent only on the CHART (BEV vs image) —
and the n600 measurements adjudicate: image chart for placement, ξ for pose + prediction.

## §4 THE MINIMAL COMPOSED PROGRAM (spec + n600-projected bytes + feasibility)

**FREE in inflate.py (rule-118, deterministic, 30-min budget):** homography constructor H(ξ);
sprite/atlas renderer + per-frame composite; Laguerre/arc partition rasterizer (generators →
argmax plane); level-set/viscosity smoother; uint8 lattice in-cell solve at decode (proven
free-at-decode [MEASURED 07-19]); CAE/ANS decoder with ξ-warp-confidence context; seeded tables.

**COUNTED payload (per-pair distribution → n600 totals; box ≤200 KB ⇒ ~333 B/pair avg;
sub-0.15 line 154,522 B ⇒ ~257 B/pair):**

| # | section | content | n600-grounded bracket | label |
|---|---|---|---|---|
| 1 | Scene atlas (sprite) | image-chart static strata + boundary arcs/generators (Road/Lane/Undrivable/hood; 61% of boundary) | 90–130 KB target; brackets: generic-AR partition ~114 KB [external MEASURED], MDL(MS) ~236 KB upper [MEASURED], ws1_seglex96 138 KB @2.85M err [MEASURED]; G4 static field 4,107 B closes 0.0078 of 0.0340 cell-space d_seg [MEASURED, cell-space NOT receiver-realized] | DERIVED target |
| 2 | ξ(t) trajectory | ~600×6 screw knots, AR/spline-coded (#574 unbuilt = strongest unbuilt rate axis); dual-use as pose payload + prediction context | 0.3–2 KB (knots) + pose finisher: quotient CLOSED on describe line; PR130 existence 23 KB → 2.33e-5-class; p1 floor: storage-shaped frame-0 carriers wall at d_pose≈19.9 (FORMULATION-scoped) ⇒ budget 5–23 KB via descent-fit carrier (A3) | MEASURED anchors, CONJECTURE composition |
| 3 | Sparse events (VOPs) | Movable-band objects (27.0% of flip mass) + transients (1.19%) + lane birth/death tokens | movable-band field 1,533 B cell-space [MEASURED]; realized VOP price UNKNOWN ⇒ budget 10–30 KB | OPEN |
| 4 | Residual syndrome | context-coded flip corrections where 1–3 miss | sized by H(flip|context) — UNMEASURED (the tier-moving scalar; E7/R5/R6 preregistered race unrun) | OPEN |

**Per-head feasibility vs the box (n600):** POSE — feasible: exact quotient closed (solve line
d_pose 1.02e-4; source-proximity law: pose binds only at ≈2.5e-4 [MEASURED]); existence proof
at 23 KB [external]; the open question is OUR carrier at ≤23 KB via joint descent (pc2 loop
16/16, ΔS −0.2475 [advisory MEASURED]). SEG — the open axis: box allows 136,839 errors; best
DESCRIBED base today 2,845,843 (0.0241) ⇒ the crux workload is ~2.7M errors of realization,
which is exactly face (2)+(3) of THE CRUX — closable only by solve-in-description-coordinates
(is1 family (d)) or live joint descent, NOT by more single-coordinate pricing [MEASURED walls].
RATE — seg secant break-even ≈150 B per 1e-6 d_seg [MEASURED instrument]; amortization is the
lever: the atlas is bought ONCE for 600 pairs (G4's 98.8% stationarity is precisely the
statement that this amortization is lawful).

**Honest anti-claims:** (a) all inherited describe-line exchange rates are upper-bound,
proposal-search-channel [MEASURED law]; (b) cell-space Δd_seg ≠ receiver-realized Δd_seg (v14
paint gap); (c) nothing here is a score claim — composed arithmetic must land ≤~0.17 aimed at
~0.15 BEFORE buying the exact row (§9 re-anchor); (d) the BEV-chart negative is chart-scoped,
not a ban on ξ (pose dual-use + context prediction survive it); (e) sy1's fail-closed edges
remain binding (fixed-atlas transmit cannot close the Seg box: short 192,020 px [MEASURED]) —
the atlas arm therefore MUST be fit jointly (live parameters), not stapled as an oracle field.

## §5 THE ONE CONFIRMING MEASUREMENT (confirm-or-kill)

**km1-class composed fit, n600, byte-closed:** fit the three payload sections JOINTLY in
description coordinates — image-chart atlas generators (Laguerre/arc, seeded from G4's
stationary field + the v8/v9 per-class carrier table) + ξ knots (+ descent-fit frame-0 pose
carrier) + Movable VOP tokens — by Gauss-Newton/CG against the exact rank-4 head + pose
quadratic with uint8-STE and the real coder in the objective; realize through the e5a/E4
adapter; measure ONE triple (bytes, d_seg, d_pose) across all 600 pairs on the frozen scorer.
- **CONFIRM:** described base ≤ ~0.5M errors at ≤130 KB with pose leg ≤2.5e-4-class ⇒ composed
  S arithmetic ≤ ~0.17 ⇒ spend the Modal envelope on the R6 dual-axis exact row (the end).
- **KILL:** if the joint fit cannot beat the ws1 138 KB / 2.85M-error endpoint by ≥3× at equal
  bytes, the image-chart atlas FORMULATION (this chart, this generator family) is falsified —
  weight shifts to the descent line as finisher (G2 fork), and the negative is scoped, not a
  paradigm kill.

## §6 Honesty + walls already measured (do not re-walk)

Fixed-atlas oracle transmit: REFUSED (192,020 px short). Per-cell waterfill granularity:
lawfully empty (ev2 0/134,211 separable). Single-coordinate edges: dead at this endpoint
(j12/pf3/pf3b). Storage-shaped pose carriers: floor 19.9 (p1, formulation-scoped). Tie-aware
preimage: NO-OP (fp32-exact 0/117.96M). Borrowed-incumbent rate polish: PERMANENTLY DEAD
(operator ban). M2-widens-box citations: stale. The describe line is NOT sealed (§8 wave:
cb1/wf7 admitted) — but its remaining openings are composed moves, which is what this codec IS.

STORES CONSULTED: routing card `council_coherent_optimal_path_routing_20260725.md` §1–§9 ·
G3 `codex_findings_ddm_g3_score_atlas_20260722T204813Z` (#622) · G4
`codex_findings_ddm_g4_spatial_stationarity_20260722T212138Z` (#623) ·
`bev_staticity_v2_absolute_trajectory_20260721T183219Z.md` · is1 findings (#684) · rd1 (#667,
via card/eureka) · sy1 FEED-603 + fail-closed edges (#712) · pan1 (#709, via card) ·
`fable_eureka_hunt_tier_breakthrough_20260725.md` (a381cd5166) · HOPE crosswalk 0058123af3 +
hb1 findings (#725) · `mpeg4_shape_coding_intake_and_crosswalk_20260719.md` ·
`ddm_ms2rp`/`pf3`/`pf3b`/`wf7`/`cb1` findings (card §§5–8) · memories:
`codec_archetype_mpeg4…20260728` · `master_thesis_invert_frozen_space…20260720` ·
`seg_and_pose_solved_exact_lattice…20260719` · `frozen_scorer_exact_factorization_20260715` ·
`opportunity_pools_non_additive…20260718` · `distortion_byte_economics_are_upper_bounds…20260724` ·
`goal_is_sub015_or_below_official_leaderboard_best…20260727` · CLAUDE.md frontier sections ·
canonical equations (`segnet_head_rank4_linear_flipdist_v1` · `hope_bn_capacity_per_stratum_codebook_v1`).

Bar honesty: no score claim anywhere above; the bar moves only via a real sub-bar contest row
through `upstream/evaluate.py` on exact archive bytes.
