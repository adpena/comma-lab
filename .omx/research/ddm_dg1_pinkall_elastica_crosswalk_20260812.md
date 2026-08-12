# DDM-DG1 — Pinkall elastica / Willmore crosswalk

**Date:** 2026-08-12  
**Arm:** `ddm_dg1`  
**Verdict scope:** source-book rigor triage plus a bounded crosswalk to the current CP135/HY1/EC1/JS7 representation surface; no new scorer call, receiver run, archive, or exact evaluation  
**Outcome:** one immediately testable proposal-ranking feature and two representation classes survive. The book does **not** supply a discrete DDG codec, a minimum-description theorem, or evidence that Pact lane boundaries are Euler elastica or that the full time sheet is Willmore.

The effective frontier remains **CP135: S=0.16195513827824176 at 186,252 B** on contest-CUDA T4, n600. The own-vehicle anchor remains **LC2: S=0.16959899569230852 at 187,226 B**. This book arm did not move either pointer.

## Source custody and rigor triage

- Canonical publication: Pinkall and Gross, *Differential Geometry: From Elastic Curves to Willmore Surfaces*, Springer, 2024, DOI `10.1007/978-3-031-39838-4`.
- The Springer PDF endpoint in the charter could not be resolved from the managed shell, and the in-app browser was unavailable. No local PDF payload materialized, so no payload was discarded and no local-custody claim is made. The intended SSD directory exists at `/Volumes/VertigoDataTier/pact/ddm_dg1/retained/` but contains no claimed book payload.
- Rigor triage used the authors' complete, exact-edition open-access PDF: <https://olligross.github.io/projects/DGCS/figs/DifferentialGeometryFromElasticCurvesToWillmoreSurfaces.pdf>. The official metadata surface is <https://link.springer.com/book/10.1007/978-3-031-39838-4>.
- The source is a smooth differential-geometry and variational textbook. It has no discrete-curves chapter, discrete-surfaces chapter, mesh algorithm, quantizer, entropy coder, receiver implementation, or benchmark. Its only `discrete` hit is the bibliography entry for Bergou et al., *Discrete Elastic Rods*. Therefore it can justify mathematical priors and falsifiable diagnostics, but not a codeable discrete method by citation alone.
- No literature ratio, score, or byte projection is transferred into Pact. All proposed tests below are `$0` tests on existing artifacts, and any later materializer is bound by payload retention and exact decode/receiver semantics.

## What the book actually establishes

| Source object | Source-level fact | Pact-safe interpretation |
|---|---|---|
| Arc length and bending, pp. 7–11 | Arc length and `B(γ)=1/2 ∫ κ² ds` are invariant under orientation-preserving reparameterization. | Parameterization should not change the geometric prior. It does not make the prior rate-free or scorer-preserving. |
| Variations, pp. 15–28 | Higher-order bending variations require the appropriate endpoint data; fixed-length elastica are critical points of bending energy and obey a fourth-order Euler–Lagrange system with a tension multiplier. | A segment codec must transmit or deterministically recover anchors/tangents and length constraints. Omitting those conditions is not an elastica construction. |
| Plane curves, pp. 31–36 | A unit-speed planar curve is determined up to an orientation-preserving rigid motion by its curvature function. Planar fixed-length elastica satisfy `κ'' + κ³/2 + λκ = 0`; an area constraint adds a constant multiplier. | Curvature is a legitimate coordinate for a **correspondence-resolved local curve**. An arbitrary curvature function is still infinite-dimensional; finite bytes require a new sampled/residual codec and a measured reconstruction rule. |
| General curves, p. 54 | Curvature data determine a unit-speed curve up to rigid motion under the stated regularity conditions. | This is a reconstruction theorem, not a statement that real lane masks lie in a low-parameter elastica family. |
| Willmore energy, pp. 181–190 | `W(f)=∫H² dA`; critical surfaces satisfy `ΔH + 2H(H²-K)=0`. The energy is scale invariant, and its modified form is Möbius invariant. A cylinder over a planar curve is Willmore exactly when the generating curve is a free elastica. | A Willmore-style energy can diagnose smooth local space-time strips. The cylinder equivalence applies to a static straight extrusion, not to arbitrary moving masks with births, deaths, contact changes, or scorer-lattice resampling. |

These are source facts. The adoption decisions below are Pact inferences constrained by the current measured representation surface.

## Ranked crosswalk

| Rank | Candidate and disposition | Why it survives or fails now | `$0` falsifier and named consumer |
|---:|---|---|---|
| 1 | **Change in discrete bending energy as an event-ranking feature — ADOPT-with-named-consumer** | EC1 already materialized 200 receiver-effective boundary/lane proposals, while JS7 showed that proxy-selected stacking can reverse sign at exact n600. A reparameterization-invariant measure of how violently an edit changes local boundary turning is a plausible *ranking feature*. It is not acceptance authority. | On the existing 200-proposal tensors, reconstruct the affected class boundary before/after, split at junctions, resample by arc length, and compute a robust discrete `ΔB_h = ΔΣ κ_i² Δs`. Join it to existing singleton outcomes and compare held-out-frame pairwise ranking/top-k useful-yield against the current `B/flip` feature, only inside the existing pose gate. **Falsify** if held-out ranking does not improve or if the preferred sign is unstable by boundary/lane stratum. **Consumer:** the EC1/JS7 corrected-stack successor, using `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/follow_on/realized_acceptance_200` and `/Volumes/APDataStore/pact/ddm_js7_20260812/ACCEPTANCE_TABLE.json`. |
| 2 | **Curvature-function residual codec for topology-stable Road↔Lane components — ADOPT-CLASS** | The plane-curve theorem makes `(anchor frame, κ(s))` a complete local coordinate under its assumptions. This can be raced against current chain/poly summaries only after correspondence. It does not revive the globally coherent normal-offset formulation that OF1 closed. | For correspondence-resolved components with no birth/death/contact event, encode a matched quantized boundary both as the current SP1/poly object and as anchor/tangent plus quantized curvature residuals; reconstruct the exact same mask coordinates, run real coders, retain every payload, and compare complete section bytes at exact geometry equality. **Falsify** if the curvature form is not smaller, needs untransmitted endpoint data, or changes the decoded mask. **Consumer:** EC1 alphabet-v2 / the `#939` description successor, with retained outputs under `/Volumes/VertigoDataTier/pact/ddm_ec1_20260812/`. |
| 3 | **Local Willmore-style strip energy as an event diagnostic — ADOPT-CLASS** | The existing worldsheet evidence has a mostly subpixel body plus an 8.25–8.40% `>4 px` tail and explicit topology events. A local second-order surface diagnostic may separate smooth transport from event debt. The book provides no reason to regularize the full sheet globally. | On existing G1 topology-stable tracks only, compute a discrete mean-curvature/second-difference energy without new scorer calls; test whether it improves held-out prediction of the measured residual tail or event-section bytes after conditioning on class, length, and motion. **Falsify** if it has no held-out association or merely proxies track length. **Consumer:** the worldsheet/event successor feeding EC1 alphabet-v2 and the TF1 event-sparse survivor; evidence roots `.omx/research/g1_worldsheet_g3_cellcode_measurements_20260720T210000Z.md` and `/Volumes/VertigoDataTier/pact/ddm_ec1_20260812/`. |
| 4 | **Arc-length parameterization, correspondence before smoothing, and first-order boundary regularity — ALREADY-EMBODIED** | The source supplies clean mathematical rationale, but the live system already contains IPM polynomial/geodesic lane objects, correspondence-first tracking, temporal smoothing receipts, and eikonal plus boundary-length energy. | Do not open a new arm for these ingredients. Curvature-squared is **not** claimed already embodied; only the lower-order geometry discipline is. **Consumers already present:** `dm2_lane_ipm_polynomial_geodesic_v1`, `correspondence_first_lane_coding_optimal_pipeline_v1`, `lane_band_source_reparam_measured_resolution_v1`, and `theta_star_eikonal_length_boundary_energy_v1`. |
| 5 | **Pure Euler-elastica lane carrier — LESSON-ONLY** | Elastica are critical curves under named constraints, not arbitrary lane boundaries and not guaranteed minimizers of description length. Current OpenPilot geometry evidence says degree increases saturate while false negatives and raggedness remain; exact-chart #609 evidence says Road/Lane are nonstatic in the tested BEV chart. | No standalone fire. It may reopen only if a topology-stable subset first passes the curvature residual byte race and fitted Euler–Lagrange residuals are small out of sample. Until then it is a restrictive model class, not the representation. |
| 6 | **Global Willmore worldsheet regularizer or cylinder reduction — LESSON-ONLY** | The cylinder theorem assumes static extrusion. Pact has measured nonstatic Road/Lane geometry, an 8% large-residual tail, and contact/birth/death events. A global fairness term can erase precisely the sparse events the codec must preserve. | No global-loss fire. Only the local diagnostic in rank 3 survives. Any training use would require a separate exact-semantics proposal after the diagnostic passes. |
| 7 | **Discrete conformal/DDG map supplied by this book — N-A** | This source does not contain a discrete conformal algorithm. Its Möbius invariance result concerns smooth Willmore energy, not invariance of the frozen SegNet/PoseNet cells, archive bytes, or the resize receiver. | Do not cite this book to implement a mesh/DDG/conformal codec. A future source must provide an explicit discretization, convergence/consistency conditions, and a code path compatible with the Pact receiver before it can enter this crosswalk. |

## Why the live evidence narrows the adoption

- **EC1/JS7:** EC1's 200 realized proposals are predominantly boundary/lane objects (151 boundary, 48 lane, 1 island), which gives rank 1 an existing, relevant data surface. JS7's exact composed row was worse than CP135 by `+0.00147089912796 S`: `+3.2e-7 d_seg`, `+2.18e-6 d_pose`, and `+323 B`. Therefore no geometric feature can inherit acceptance authority from an n32 proxy; it may only generate or rank candidates before exact, pose-aware selection.
- **TF1:** exact raster intra coding was 356,636 B while global transport was 453,449 B. That closes the tested global xi-raster transport formulation, not local curve/event coordinates.
- **OF1:** the tested coherent one-dimensional normal-offset bands had median arc length only 2–3 px and 5–21× worse useful flips per degree of freedom. Rank 2 is consequently restricted to correspondence-resolved component geometry; it cannot be used to relabel the closed global `δ(s)` offset formulation.
- **#609 / G1:** the exact tested G1 chart had Road/Lane median residuals of 39.02/47.12 px with only 4.31/4.37% within 1 px. A different pairwise ground-homography surface had a subpixel body but an 8.25–8.40% `>4 px` tail and topology events. These facts reject a global static-cylinder interpretation while leaving topology-stable local strips testable.
- **#934 / LP1:** pixel compositing harmed the result; the same geometry became useful only after frozen-head color solving/re-rendering. Any eventual curve carrier must feed a solve/re-render path, not paste a geometric mask into RGB.
- **#939 / BF1:** the description is only partially settled: its lane-crop section was lossy and receiver-closed survival was unmeasured. This arm does not promote that object or call it a canonical task owner.
- **#580:** the resize-nullity/gauge result is lesson-only and reclaims 0 B on the current archive. It is not a conformal-equivalence result and cannot be combined with smooth Willmore invariance to claim a free receiver gauge.

## TOP-2 FIRE ORDERS

1. **FIRE-1 — ADOPT-with-named-consumer.** Owner: EC1/JS7 corrected-stack successor. Consumer store: the existing JS5 realized-proposal directory plus JS7 `ACCEPTANCE_TABLE.json`. Trigger: source and acceptance hashes match the existing 200-event census and the successor uses the corrected n600 pose marginal; then compute the rank-1 feature with no new scorer call. Stop on the registered held-out falsifier.
2. **FIRE-2 — ADOPT-CLASS.** Owner: EC1 alphabet-v2 / `#939` description successor. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_ec1_20260812/`. Trigger: FIRE-1 has identified a topology-stable Road↔Lane stratum and correspondence is explicit; then run the exact-geometry, real-coder curvature residual race, retaining per-candidate payloads and hashes. Stop if complete bytes do not beat the matched current representation.

Rank 3 is deliberately third: it fires only after a topology-stable track census is available and cannot delay the two direct local representation tests.

## RECALL EVIDENCE

| Recalled object | Current bounded fact used here | Disposition in this arm |
|---|---|---|
| `.omx/research/ddm_ec1_event_coordinate_producer_20260812.md` | 200 receiver-effective proposals and exact boundary/lane/event inventories already exist. | Existing data surface for FIRE-1; no rematerialization. |
| `.omx/research/ddm_js7_acceptance_sweep_and_compose_20260812.md` and exact verdict | The selected 44-event stack reversed the projected sign at n600 and worsened all three score terms. | Geometric energy is ranking-only, never acceptance authority. |
| `.omx/research/ddm_tf1_theoretical_floor_and_beyond_20260812.md` | Global xi raster transport is larger than exact raster intra. | Global transport remains closed; local curve/event class remains open. |
| `.omx/research/ddm_lp1_lane_program_20260803.md` | Geometry needs solve/re-render; post-hoc pixel compositing was harmful. | Receiver path constraint on any later carrier. |
| `.omx/research/ddm_bf1_20260805/BF1_RECEIPT_20260805.md` | The `#939` description is partial and receiver survival is unmeasured. | Named as a successor description, not promoted evidence. |
| `.omx/research/ddm_op1_openpilot_physics_geometry_review_20260728.md` plus the #609 receipts | Static exact-chart BEV motion failed for Road and Lane; the original hood-control probe was no-verdict. | Forbids a static/global cylinder inference. |
| `.omx/research/g1_worldsheet_g3_cellcode_measurements_20260720T210000Z.md` | Smooth transport body coexists with a substantial large-residual tail and topology events. | Motivates local-only rank 3. |
| `.omx/research/ddm_of1_offset_field_and_flicker_coherence_20260729.md` | Coherent global normal offsets were too short/speckled and inefficient. | Prevents curvature coordinates from resurrecting global `δ(s)`. |
| Canonical equation/DAG objects named in rank 4 | Arc-length, correspondence, smoothing, polynomial geometry, and first-order boundary regularity are present. | `ALREADY-EMBODIED`; no duplicate arm. |

Task-like labels `#580`, `#609`, `#934`, and `#939` are treated as memo handles where the canonical task-status/bridge surfaces do not provide matching task rows. No ownerless, done, or queued claim is inferred from the labels.

## Measurement and authority boundary

- **Measured and recalled:** the repository receipts and exact rows explicitly cited above, within their recorded scopes.
- **Derived here:** the mapping from smooth variational facts to the three bounded candidate classes.
- **Untested:** all three proposed geometric features/codecs. No association, byte win, d_seg improvement, pose safety, receiver survival, archive score, or frontier movement is claimed.
- **Not run:** scorer, receiver, materializer, archive mutation, exact evaluator, paid dispatch, training, or payload-producing probe.
- **Authority:** the source book is authority for its smooth mathematical statements only. Exact Pact authority remains the actual receiver/evaluator on exact bytes.

## NEXT_IF_RESUMED

- **ADOPT-with-named-consumer:** owner `EC1/JS7 corrected-stack successor`; consumer store `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/follow_on/realized_acceptance_200` joined to `/Volumes/APDataStore/pact/ddm_js7_20260812/ACCEPTANCE_TABLE.json`; fire when both source hashes validate and the corrected n600 pose marginal is in the selection logic; compute and falsify the rank-1 held-out bending-change feature without a new scorer run.
- **ADOPT-CLASS:** owner `EC1 alphabet-v2 / #939 description successor`; consumer store `/Volumes/VertigoDataTier/pact/ddm_ec1_20260812/`; fire when a topology-stable Road↔Lane stratum has explicit correspondence; race retained curvature-residual payloads against the matched current representation at exact decoded geometry equality.
- **ADOPT-CLASS:** owner `worldsheet/event successor`; consumer store `/Volumes/VertigoDataTier/pact/ddm_ec1_20260812/` with the G1 receipt as the source ledger; fire only after topology-stable tracks are separated from births/deaths/contact changes; test the local Willmore-style diagnostic and stop if it adds no held-out event-tail information.

## LIVE-HYPOTHESES

- **Boundary-event bending change:** among already-realized, pose-gated proposals, small or beneficial `ΔB_h` may identify edits that respect the smooth local partition while avoiding ragged spill. It is plausible because most EC1 proposals touch boundary/lane geometry, but it has not been joined to held-out outcomes.
- **Curvature residual entropy:** after correspondence and topology gating, curvature residuals plus anchors may compress a smooth Road↔Lane component more cheaply than chain/poly coordinates. It is plausible because curvature is a complete unit-speed plane-curve coordinate; the byte advantage is wholly unmeasured.
- **Local worldsheet fairness:** a discrete Willmore-style energy may distinguish the smooth transport body from the sparse event tail. It is plausible because the G1 evidence contains both regimes; it may still collapse to a length/motion proxy.

## DEAD-ENDS

- **Book-as-discrete-algorithm:** closed for this source because it contains smooth theory but no discrete DDG/conformal construction, coder, or receiver.
- **Wholesale pure-elastica lane replacement:** closed on the current evidence because constrained critical curves are not a minimum-description family for arbitrary lane masks, and the live residual is dominated by missed/ragged structure rather than polynomial degree.
- **Global Willmore/static-cylinder sheet:** closed for the measured chart because Road/Lane motion and topology-event tails violate the static-extrusion premise.
- **Global coherent normal-offset resurrection:** closed in the OF1 formulation; its bands were too short, speckled, and inefficient. Only component-local, correspondence-resolved curvature coordinates remain live.
- **Smooth conformal invariance as receiver/scorer gauge:** closed because Willmore/Möbius invariance does not imply frozen-evaluator, resampling, or byte invariance; #580 provides no such bridge.
