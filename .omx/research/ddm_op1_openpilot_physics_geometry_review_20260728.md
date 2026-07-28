# ddm_op1 — OpenPilot + physics/deep-math/geometry review for the tr1 renderer build (tb1 = named consumer)

**Pointer honesty first.** `0.1910828242 [contest-CPU]` — our submittable original-work frontier — is
**UNMOVED**. This arm is a review/derivation support arm for the LIVE renderer build (ddm_tb1,
`SPEC_tr1_trained_partition_renderer_20260728.md`); it ran **no scorer jobs, no launches, $0 spend**,
and moved no exact score. GOAL bar unchanged: `min(0.15, 0.172)` (effective_frontier = PR130 0.172).
`evidence_axis: review/derivation · research_only=true · score_claim=false · paid_dispatch=false`.

**Honesty labels used throughout:** `MEASURED` (committed artifact, path cited) · `DERIVED`
(arithmetic/derivation shown here from MEASURED inputs) · `CONJECTURE` (falsifier named) ·
`OPEN-QUESTION`.

STORES CONSULTED: CLAUDE.md + AGENTS.md (full; NO-FAKE, THE GOAL, rule-118, scorer facts, class-order
law, eval_roundtrip/EMA/QAT, MPS train/authority split); `scratchpad/op1_charter.md`;
`SPEC_tr1_trained_partition_renderer_20260728.md` (full read — the consumer's blueprint, §S1.2 grid
derivation + §S3 gates); `.omx/research/bev_staticity_developability_probe_20260721T172426Z.md` (+
receipt shas) ; `.omx/research/bev_staticity_v2_absolute_trajectory_20260721T183219Z.md`;
`.omx/research/openpilot_cross_surface_audit_20260706.md` (incl. the #327 MEASURED ADDENDUM);
`.omx/research/comma_openpilot_crossref_polynomial_geometry_20260619T014433Z.md`;
`.omx/research/comma_openpilot_domain_tricks_20260619T035417Z.md`;
`.omx/research/openpilot_world_model_free_prior_v2_20260629T190505Z.md`;
`.omx/research/openpilot_lane_headstart_landed_20260629T193648Z.md`;
`.omx/research/openpilot_comma_repo_wider_exploit_sweep_pose_cereal_hood_20260617T192718Z.md`;
`.omx/research/pantheon_synergy_crux_synthesis_20260728.md` (via its memory hook, commit a3281cf312);
sub015 DAG FEED rows (FEED-BINDING-PRINCIPLE-…-20260721, FEED-603-gestalt + dr2 correction,
FEED-603-lane-grammar); `.omx/state/canonical_task_status.jsonl` (charter-ID reconciliation);
MEMORY.md current-state rows (pantheon 07-27, codec archetype, pose-terminal law, frozen-scorer
factorization). Existing crosswalk corpus NOT duplicated: #476 five-slot geometry, #284 diff-geo,
#593 ACME, #584 matrix calculus, #616 Brenier, #550 Nielsen.

---

## RECALL DISCREPANCY REPORT (RECALL-FIRST binding — reported, not silently proceeded)

**D1 — The charter's #609 characterization contradicts the committed receipts.** The charter says
*"#609 (BEV staticity/developability probe — road/lane stratum measured MORE STATIC in BEV,
ξ-registered)"*. The committed record says the opposite for the tested chart:
- v1 (`bev_staticity_developability_probe_20260721T172426Z.md`): `NO_VERDICT_C1_HOOD_CONTROL_FAILED`
  — no Road/Lane verdict authorized at all `[MEASURED]`.
- v2 (`bev_staticity_v2_absolute_trajectory_20260721T183219Z.md`): D0 hood control PASS (p50 0.0 px,
  0.913 ≤1 px, n600), then **Road p50 39.0226 px / Lane p50 47.1192 px ruling residual; only
  0.043093 / 0.043713 of samples ≤1 px** — Road/Lane are **NON-static** in the exact G1-calibrated
  absolute chart; probe ledger verdict `KILL` scoped to that exact chart, with 3 reactivation
  criteria (new custodied absolute-motion source/calibration preserving the D0 pass AND raising BOTH
  Road and Lane to p50 ≤1 px and fraction ≥0.5 at n600) `[MEASURED]`.
- The pantheon synthesis (07-27, commit a3281cf312) already canonicalized this: *"atlas lives in the
  IMAGE chart, NOT BEV… ξ predicts motion but cannot place boundaries"* `[MEASURED: pantheon memo]`.
Everything in §P2 below treats the v2 numbers as the truth and the charter parenthetical as a
mis-recall. No number was consumed from the charter's framing.

**D2 — charter task-IDs are memo handles, not `canonical_task_status.jsonl` task_ids.** The ledger's
task_id namespace does not contain bare `325/326/327/145/156/609/263`; the receipts live as the
`.omx/research` memos cited above (the #327 receipt is the MEASURED ADDENDUM inside the #326 audit
file). Reported for future charter hygiene; the custody itself was located and consumed.

---

## P1 — DISTILLED OPENPILOT CUSTODY TABLE (what binds tb1; settled vs ancestor-scoped)

All rows below are from committed artifacts; nothing was re-derived. "Scope" says whether the number
transfers to tb1 (scorer/scene facts are vehicle-independent; carrier-level results are family- or
formulation-scoped).

| # | custody (memo) | MEASURED facts that bind tb1 | scope / status |
|---|---|---|---|
| 1 | #326 cross-surface audit (openpilot @ee54e82) | Canonical constants: native cam 1164×874, f=910, pp (582,437); SEGNET_SIZE (512,384); camera height **1.22 m** (`HEIGHT_INIT`); scorer-plane intrinsics **fx=400.27, fy=399.82, c=(256,192)** (derived arithmetic, MEASURED against source); SE(3)/frame conventions of the live ξ/pose path CONFIRMED-CORRECT (translation-first `[ρ,ω]`, view y=down, n(p=0)=[0,−1,0]) — **do not change**; NO openpilot upstream candidates; do-NOT list (don't blind-set v_h=192, don't hardcode dash period before geometry, don't wire `calibrated_geometry.py` into live path). | SETTLED, vehicle-independent. Direct input to any tr1 geometric conditioning feature. |
| 2 | #327 addendum (same file, n600 through real GT argmax) | **v_h=174 is OPTIMAL** for the lane IPM band: recall 0.5475 / precision 0.6585 / FP_far 4.0e-4 / band_err 0.00462, all monotonically WORSE at 188 and 192 — the inferred "174 is wrong" finding was FALSIFIED by the n600 row. cam_h 1.2→1.22: Δband_err exactly 0.0 (cosmetic). **Two-horizon-roles law:** 174 = lane-IPM saliency VP (measured-optimal), 192 = zero-pitch geometric horizon for pose projection; do not force to one. | SETTLED (n600, `[macOS-CPU advisory]` measurement of a geometric fact). Inherit 174 for any lane/ground feature; 192 for pose geometry. |
| 3 | #609 v1+v2 BEV staticity | v1: NO_VERDICT (hood control failed, p50 7.75 px). v2: hood D0 PASS (p50 0.0 px / 0.913 ≤1 px n600; SE(3) closure ≤3.6e-15 m — the `tac.lie` chain is exact); **Road 39.02 px / Lane 47.12 px p50 ruling residual, 4.31%/4.37% ≤1 px → NON-static in the exact G1 chart**; Movable non-static control passes (10.76 px). KILL scoped to the exact chart; 3 reactivation criteria (above). | SETTLED negative for the tested chart; verdict_scope = exact-chart, NOT all BEV families. THE decisive P2 input. |
| 4 | #145 polynomial-geometry crossref | Contest clip IS comma EON native footage (intrinsics exact match) and IS comma2k19 segment `b0c9d2329ad1606b\|2018-07-27--06-03-57/10` (per #156 §0); comma10k 5-class scheme byte-exact (Lane = thin strokes only, no arrows/crosswalks); openpilot's native lane rep = degree-4 polynomial (`POLY_PATH_DEGREE=4`, 33-pt quadratic grid to 192 m); VP (256,174), horizon band rows 155–195 @512×384. | Geometry facts SETTLED. The "polynomial as the lane CARRIER" idea is superseded by row 6 (measured floor). |
| 5 | #156 domain tricks | Ground row map (native frame): `v = 910·1.22/d + 437`; class layout geometrically pinned (sky band / road trapezoid / static hood); pose is ~1–2 effective DOF (`v_fwd` + `ω_yaw`; `v_vert/v_lat/ω_roll/ω_pitch` kinematically null for this fixed mount) with EKF noise floors 0.5 m/s / 0.05 rad/s; chroma pre-declared cheap by the contest's own 4:2:0 preprocessing; every YUV step clamps (out-of-gamut free); SegNet reads frame_1 only. | Scene/scorer facts SETTLED. The frame-role split is REFINED by newer law: frame_0 is structurally seg-free (d_seg obligation 8.5e-9, Unit C) and pose is the TERMINAL 6-eq solve (#383) — do not resurrect the older "frame_0 = pose carrier" framing beyond that. The memo's class-order "MUST-VERIFY" was settled 2026-06-27 (CLAUDE.md canonical order, MEASURED). |
| 6 | #325 world-model prior + lane headstart | **Polynomial lane-carrier floor (n96, oracle width, per-dash): d_seg 0.002144 > the ~1.23e-3 need → smooth-curve lane carriers CANNOT alone reach the bar; residual is FN-dominated boundary raggedness, not curvature (deg 1→4 saturates)**. The prior still recovers ~64% of lane d_seg as a free positional prior. **Image-space explicit centerline is NOT cheap: adjacent-frame lane IoU 0.284, ~65.2 KB/600 zlib (rate 0.043)**; the 0.5–5 KB figure requires the ground frame and was NEVER measured. supercombo compress-time analyzer CONFIRMED $0 CPU (onnxruntime 1.27, v0.9.7 LFS 49.1 MB sha-verified, ~15 ms/frame, `ORT_DISABLE_ALL` workaround). rule-118: coeffs COUNTED / rasterizer FREE. | Carrier-family-scoped negative (smooth-curve carriers), still binding: tr1's renderer must learn the ragged ±1 px lane boundary — this is exactly SPEC G1's job. The 65 KB image-centerline number is representation-specific; pp1's 173.6 KB explicit / ~117 KB learned-token numbers are the current pricing frame. |
| 7 | #263 wider exploit sweep | Ranked design candidates: pose trajectory-smoothness low-rank coding (600×6 is EKF-smooth); hood static-region clamp (since absorbed as the #139 static core, MyCar temporal IoU 0.994); road-edge prior (contingent, IoU gate never passed); per-dim pose weighting + unscored-back-6-dims audit. | Mostly ABSORBED or SUPERSEDED: pose coding is now the terminal 6-eq law + rank-1 e_p ~2 KB (MEASURED-CLOSED per MEMORY); hood is the #139 core. Keep only as provenance. |
| 8 | Image-chart staticity (dr2 + G4, via pantheon/DAG) | **98.806% of flip mass is image-stationary (G4)**; movable-band 99.149% / lane-corridor 97.78% static-in-image — with dr2's caveat: *pixel recurrence ≠ record constancy* (the "Road 99.1%" attribution was corrected to the movable band). Partition temporal disagreement **1.246%**/frame (pp1). ξ-proxy explains **2 of 47,882** singleton flips. | SETTLED. The image chart is where the partition is (approximately) static; ξ does not place boundaries. |

**Net P1 verdict for tb1:** the openpilot custody is CONSUMED and largely SETTLED; nothing in it
forces a change to SPEC_tr1 §S1.2's image-plane grid derivation, and rows 3+8 actively confirm it.
The live free gifts to tr1 are: the reconciled geometry constants (rows 1–2) as decode-side
conditioning features, the compress-time supercombo/poly analyzer (row 6) as an initializer, and the
already-correct SE(3)/ξ path (row 1) for the terminal pose leg.

---

## P2 — BEV vs IMAGE-PLANE TOKEN GRID: DERIVED RECOMMENDATION + FALSIFIER

**Recommendation (DERIVED from MEASURED inputs): tb1's token grid stays in the IMAGE (scorer) plane.
A BEV-warped grid variant does NOT currently earn a lane in the D∈{8,12,16} race.** BEV/openpilot
geometry enters tr1 in three subordinate FREE roles instead (below). Confidence: HIGH on the default;
the falsifier keeps the BEV door honest.

### The derivation (five independent legs, each from a measured anchor)

1. **The staticity premise inverted (the charter's own argument, corrected).** The only measured
   BEV-advection test (#609 v2) puts Road/Lane boundary ruling residuals at **39–47 px p50** in the
   exact G1 chart (≤1 px fractions 4.3%), while the SAME strata are **97.8–99.1% image-stationary**
   and 98.806% of flip mass is image-stationary `[MEASURED, P1 rows 3+8]`. The d_cov law
   (pair-dependence factors through (ξ,R)) is not violated — but the measured split says the
   surviving temporal variation in the image chart is dominantly **d_gauge** (sub-pixel dash phase ×
   R's resampling grid × stride-2 stem phase; g1's lane knee 0.004946 ≈ the flicker floor 0.005318
   `[MEASURED: DAG FEED-603-lane-grammar]`). BEV advection does NOT remove d_gauge (it is generated
   in the image/R chart) and it ADDS the 39–47 px placement error. ξ predicts motion; it cannot
   place boundaries; placement is counted content.

2. **The scorer metric is uniform in the image, so the image grid IS the perspective-optimal
   measure.** d_seg counts pixels on the 512×384 lattice; each D=16 cell holds an equal 256-px slice
   of maximal d_seg exposure (0.13%). A BEV-uniform grid allocates cells ∝ ground area. With the
   reconciled geometry (fx≈400.3, h=1.22, cy=192 `[MEASURED: P1 row 1]`), the scorer-plane ground map
   is `v(d) = 192 + 488.3/d` `[DERIVED]`: d=10 m→row 240.8, 20 m→216.4, 50 m→201.8, 100 m→196.9.
   A BEV cell of length s at distance d occupies `Δv ≈ fx·h·s/d²` image rows, so BEV cell density per
   image pixel grows ∝ d²: at 100 m a 1 m BEV cell is 0.049 px tall — ~326 BEV cells per D=16 token
   row. A uniform BEV grid to openpilot's own 192 m horizon spends essentially all its cells inside
   rows 192–202 (~half a token row), while the near field (rows 240–384) is undersampled relative to
   its pixel mass. The image grid gets this allocation right by construction.

3. **Lane thinness at range dies in the BEV→image→R round trip.** A ~15 cm lane stroke has image
   width ≈ `fx·0.15/d` = 60/d px `[DERIVED from P1 row 1]`: 3 px at 20 m, 1 px at 60 m, sub-pixel
   beyond. The fd2 lesson (SPEC §S2) is that realization lives at the image/uint8/R staircase; a
   BEV-parametrized Lane must survive an EXTRA resampling (BEV→image homography) BEFORE the R chain,
   exactly where the class is already sub-pixel — the thin-stroke erasure mode the lane long-tail
   already exhibits (error ∝ 1/persistence). The lane-carrier floor 0.002144 `[MEASURED: P1 row 6]`
   was FN-dominated raggedness — structure a ground-frame smooth parametrization is intrinsically
   blind to.
4. **More than half the scored field has no BEV chart.** Undrivable (49.5%, sky-dominated) and MyCar
   (25.4%, image-static hood, #139) are not ground-plane strata; Movable is independently non-static
   in ANY ego chart (`[MEASURED: #609 v2 control]`). A BEV grid could parametrize only Road+Lane
   (23.8% of pixels — though 61% of boundary content per pp1), forcing a two-chart hybrid with a
   counted seam. The image chart is the ONLY chart all five classes share — and it is the chart the
   scorer reads.
5. **The BEV upside is already partially captured by the renderer itself.** What BEV would buy —
   store the ground strata once, replay through ξ — is the same shared-structure absorption the
   trained renderer performs by training against ALL 600 pairs (fd1 Rung-2 routing; ee1 C10
   convergence). pp1's +57 KB explicit-vs-learned gap is the measured size of that absorption
   `[MEASURED via SPEC §S1.4]`; a BEV chart would compete for the same redundancy, not add to it.

### What openpilot geometry SHOULD do in tr1 (all rule-118 FREE, no BEV grid needed)

- **(a) Decode-side deterministic conditioning features** computed FROM the decoded tokens/partition
  + fixed constants (never counted): horizon-relative row `(v−174)` for the lane band / `(v−192)`
  for pose geometry, IPM forward distance `d(v)=488.3/max(v−192,1)` on the ground band,
  distance-to-boundary maps. This is the CLADE-ICPE pattern (P3 row 1) — perspective awareness as
  free positional encoding instead of a warped grid.
- **(b) Compress-time analyzer/initializer:** supercombo + degree-4 poly fits, CONFIRMED $0 CPU
  `[MEASURED: P1 row 6]` — token-grid + renderer warm-start, never in the archive.
- **(c) The terminal pose leg** keeps the CONFIRMED-CORRECT openpilot-frame SE(3)/ξ path unchanged
  `[MEASURED: P1 row 1, finding 8]`.

### Pre-registered falsifier (the honest BEV re-entry gate; $0, no scorer jobs)

A BEV-warped grid variant earns a race lane ONLY if BOTH hold:

- **F1 (chart gate, inherited from #609 v2 reactivation):** a new custodied absolute-motion source or
  calibration that preserves the v2 D0 hood pass AND raises BOTH Road and Lane to ruling residual
  p50 ≤1 px with fraction ≥0.5 at n600. (Without a chart that places boundaries, BEV tokens cannot
  realize — this is the measured blocker, not an opinion.)
- **F2 (token-level gate, new, runs only if F1 passes):** tokenize the GT partition at the raced
  (D,c); reproject the token field into the passing ground chart; measure per-cell temporal change
  rate across n600 vs the image-chart change rate (image-chart pixel-level bound: 1.246%/frame
  `[MEASURED: pp1]`). Require BEV-chart token change rate < 0.5× image-chart on Road+Lane AND ≥99%
  Lane token mass preserved through the BEV→image→R round trip (thin-stroke erasure check).

Predicted outcome given current evidence: F1 FAILS (that is what #609 v2 measured). The falsifier
converts any future "BEV should work" intuition into a measurement with named thresholds.

---

## P3 — FRESH PHYSICS / DEEP-MATH / GEOMETRY SWEEP (ranked; each row: label · prior art · falsifier/first measurement · tb1 consumption point)

Ranked by (leverage on SPEC G1 `native d_seg ≤5e-4 at ≤64 KB`) × (cheapness of the first
measurement). Rows 1–4 are recommended for tb1 attention before T2 freeze; 5–8 are named-and-parked.

| # | row | label | prior art (concrete) | falsifier / first measurement | tb1 consumption |
|---|---|---|---|---|---|
| 1 | **Class-adaptive (not spatially-adaptive) modulation + deterministic positional encoding.** With K=5 classes, per-class γ/β costs `2·K·C` params per norm layer (tens of params) vs a SPADE modulation subnet (avg ~39% param overhead per the CLADE analysis, worst layers >600%). CLADE's core finding — modulation benefits from semantic-awareness more than spatial-adaptiveness — lands exactly at tr1's budget; CLADE-ICPE's intra-class positional map is computed FROM the layout at decode ⇒ rule-118 FREE, and OUR positional maps can be the openpilot-geometry features (P2(a): v−174, d(v), dist-to-boundary). | DERIVED (transfer) | CLADE, Tan et al., TPAMI 2021 (arXiv 2012.04644); SPADE, Park et al. 2019 (arXiv 1903.07291) | Matched A/B at (D=16,c=4), equal counted bytes: CLADE+geo-ICPE conditioning vs a mini-SPADE 2-layer modulation net; adopt the lower native-d_seg arm; transfer falsified if mini-SPADE wins by >10% | T0 architecture — the S1.3 conditioning block; DSL lever `renderer_conditioning ∈ {clade_geo_icpe, spade_mini}` |
| 2 | **Row-anisotropic D (perspective foveation WITHOUT BEV).** The boundary/flip mass concentrates in the horizon–midfield band (Movable band rows 174–215 `[MEASURED: CLAUDE.md class geometry]`; 50 m→∞ road compresses into rows 192–202 `[DERIVED: P2 leg 2]`; sky mean row 95 is near-free; hood static). A row-banded grid (e.g. D=8 for rows 160–240, D=16 elsewhere) buys boundary pitch where flips live at ~constant token count. | DERIVED | Foveated/spatially-variant sampling (foveal adaptive pyramid, PMC5190984); quadtree CU splitting (HEVC) | $0 from existing artifacts: per-row flip-mass histogram from `gt_n600` margins/g3 data; adopt iff ≥50% of flip mass lies in rows 160–240 (~21% of rows); else uniform D stands | T1 grid race — ONE extra variant lane in the S1.2 Pareto sweep, replacing any BEV lane |
| 3 | **Boundary-gated token code width (PointRend logic at the coder).** The partition is piecewise-constant: boundary is 2,436 px/frame ≈1.2% of pixels `[MEASURED: pp1]`, so interior cells are class-constant and should carry ~0 extra bits beyond class; spend c only at boundary-crossing cells. | DERIVED | PointRend, Kirillov et al., CVPR 2020 (boundary-adaptive subdivision); JBIG/crack-edge region coding | $0: on GT tokens at (D,c), measure H(cell\|neighbors) split interior vs boundary cells; adopt iff boundary-gating saves ≥15% of the token stream vs uniform c | T1 token-coder design — the S1.2 "small learned prior" becomes boundary-gated; feeds G4 (≤130 KB, target ~117 KB) |
| 4 | **OASIS lesson — per-pixel class-BALANCED seg feedback.** OASIS showed a segmentation-network discriminator with (N+1)-class per-pixel CE + class balancing is what makes label alignment work, especially for rare classes; perceptual losses become superfluous. tr1's "discriminator" is the REAL frozen SegNet — stronger than OASIS's learned one — but the class-balancing term transfers: Lane is 0.59% of pixels yet 36% of partition cost `[MEASURED: pp1]`. | DERIVED (transfer) | OASIS, Sushko et al., ICLR 2021 / IJCV 2022 | A/B on the S2.1 loss: boundary-weighted margin loss ± inverse-class-frequency balancing; read PER-CLASS native d_seg (Lane column decides); falsified if Lane d_seg does not improve at equal total | T0 loss — one swept Lever on the seg trunk (`class_balance_weights`) |
| 5 | **Free interior fill (PDE renderer for the interiors, learned capacity only at the separatrix).** Represent interiors as deterministic diffusion/screened-Poisson fill from a thin boundary band; the solver is GENERIC ⇒ FREE in inflate; ALL learned capacity routes to the boundary annulus (matches margin-field physics + Lane cost concentration). | CONJECTURE | Poisson image editing (Pérez 2003); screened Poisson (Bhat 2008); Diffusion Curves (Orzan 2008) | $0 smoke: diffusion-fill GT interiors from a 3 px GT boundary band, score through the frozen SegNet; adopt-relevant iff interior flips <5% of total flips | T1 smoke — architecture split (boundary head + free fill) as a race variant only if rows 1–3 leave G1 unmet |
| 6 | **Crack-edge (inter-pixel) boundary tokens.** Code the 1-cochain (boundary segments between cells), not the 0-cochain (cell classes): the argmax field is determined by its crack-edge set + one seed class per region; chain-coded cracks are the classical minimal code for piecewise-constant images. | CONJECTURE | Freeman chain codes (1961); Kovalevsky cell complexes; JBIG2 region coding | $0: chain-code the GT boundary at scorer res, compare bytes vs pp1's context-arith 173.6 KB baseline at exact fidelity; feed the token prior iff <0.5× | T2 window — alternative token-prior family if G4 misses ~117 KB |
| 7 | **Dirty-paper anchor for the S2.4 control-token re-solve.** Costa/Gelfand-Pinsker: with interference (the frozen renderer's bias) known at the encoder, capacity is as if it were absent — the theory row justifying re-solving tokens against the frozen renderer+SegNet; measurement gate already pre-registered in SPEC §S2.4. | DERIVED (theory) | Costa 1983 "Writing on dirty paper"; Gelfand–Pinsker 1980 | Already pre-registered in SPEC (no d_seg gain at equal bytes ⇒ skip) | T2 window — no new lane; prior-art anchor only |
| 8 | **Schur/arrowhead structure for the GN legs.** Token params are per-frame (pose-like), renderer weights global (landmark-like) ⇒ bundle-adjustment Schur complement applies to the terminal 6-eq pose solve and any control-token re-solve; pure optimizer engineering, no score claim. | DERIVED (structure) | Triggs et al. 2000 (bundle adjustment); Agarwal et al. BAL 2010 | n/a (engineering); verify step wall-clock vs fd1r's 99.6%-verdict-dominated law before optimizing the 0.4% | T2 optimizer engineering — only if solve time ever matters (fd1r says it does not today) |

**Anti-duplication note:** rows deliberately do NOT re-cover the existing crosswalk corpus (#476,
#284, #593, #584, #616, #550), the SPEC's own fd2/STE physics, or the settled pose law.

---

## HONEST BOUNDARIES

- $0 arm: no scorer jobs, no launches, no paid dispatch; every number above is quoted from a
  committed artifact (path given) or derived here with the arithmetic shown. NOTHING here is a
  score; the pointer `0.1910828242 [contest-CPU]` is UNMOVED and moves only through a byte-closed
  `upstream/evaluate.py` row.
- The P2 recommendation is DERIVED, not itself an n600 measurement; its authority rests on the cited
  MEASURED anchors (#609 v2, G4/dr2, pp1, #326/#327). The falsifier F1/F2 is the pre-registered
  re-entry gate — the BEV family is NOT killed (verdict_scope of #609 v2 is the exact G1 chart).
- P3 rows are design candidates with falsifiers, not adopted levers; per the DSL-as-SoT law, any
  adopted row must land as a `Lever` factory in the BUILD arm, never a hand-added flag.
- Web-verified prior art consulted this session: CLADE (arXiv 2012.04644 + github.com/tzt101/CLADE),
  SPADE (arXiv 1903.07291), OASIS (IJCV 2022, lamarr-institute.org publication page), foveal
  adaptive pyramid (PMC5190984). Other prior-art citations (PointRend, Poisson/diffusion-curves,
  Freeman/JBIG, Costa/Gelfand-Pinsker, Triggs BAL) are standard results cited from domain knowledge
  and flagged for spot-check if any row is adopted.

---
## MAIN post-merge annotation (append-only; verdict-scope + relative-significance compliance)

**verdict_scope declarations for negative tokens in this memo:**
- The #609-v2 BEV staticity **KILL** cited herein: `verdict_scope: formulation — BEV-staticity
  of the road/lane stratum in the EXACT G1 chart (custodied homography, v_h ladder)`. The BEV
  FAMILY is NOT dead: the pre-registered re-entry gate F1∧F2 (new custodied chart with D0 pass
  + Road AND Lane p50 ≤1px at n600 ∧ token-change-rate + survival tests) is the falsifier that
  keeps the door honest. One chart formulation failed; the family re-enters on a passing chart.
- Any **FALSIFIED** token referring to the op1 charter's "MORE STATIC in BEV" premise:
  `verdict_scope: instance — one charter-composition recall error by MAIN (receipts prevail)`;
  it falsifies the charter sentence, not any measurement or family.

**Relative-significance annotation for parked row 5 (PDE interior fill)** [magnitude-ok]: the
parking is SEQUENCING under fleet-cap, not a magnitude kill — `verdict_scope: instance —
prioritization decision, row stays a live candidate with its falsifier`. The arithmetic AGAINST
dismissal, stated per the relative-significance law: the row's value is on the RATE axis
(interior tokens → deterministic fill); a 10KB token-stream saving = 25·10,240/37,545,489 =
**0.00682 S**, vs the remaining mid-corner gap to the 0.172 bar of ~0.004 (SPEC_tr1 composed
mid 0.176) — i.e. ~1.7× the gap; even 3KB saved (0.00205 S) is half the gap. NOT negligible.
Adoption gate: tb1's MEASURED T2 token-stream occupancy decides whether interior tokens are a
material fraction; if interiors occupy ≥10% of the token stream, the PDE-fill race fires as a
named variant. Un-recoverability is NOT claimed — no measurement supports it.
