# GENERATOR-DESCRIPTION CRUX SYNTHESIS — bytes(generator + band-slack) is the ONE open axis (2026-07-19)

**Posture:** RECALL-BEFORE-DECIDE consolidation. No experiment run, no launch, no dispatch.
**Pointer:** `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**. Every number below is MEASURED/DERIVED
advisory (macOS/Darwin CPU n6/n24/n48/n600 local) unless explicitly tagged as the official-evaluator spine
row; nothing here is a new score claim.
**Crux (operator 2026-07-19):** plane-storage is RATE-DEAD as a family (~336 KB/pair floor). The one open
axis is a minimum-byte DESCRIPTION of the scorer planes, expanded free at decode (rule 118) by the proven
C1 receiver spine, plus band-slack. This memo is the single consumable map of everything we have MEASURED /
BUILT / DESIGNED on that axis, with sources verified first-hand.

---

## 0. The budget box (re-derived here; every input MEASURED)

The official-evaluator spine row (`.omx/research/v10_capstone_first_byteclosed_row_20260719.md`,
unmodified `upstream/evaluate.py`, 600 samples, `--device cpu`): **S = 272.73** on the exact-plane
predictor-residual archive (409,526,925 B) — d_seg **1.5196e-4** (contribution 0.0152), d_pose
**1.0184e-4** (contribution **0.0319**), rate 272.687 = 99.98% of S. The chain (payload → archive.zip →
contest-signature inflate.sh → scorer-free receiver → exact factor-2 integer solve → official evaluate.py)
is PROVEN end-to-end; any compact description is a drop-in payload swap into this spine
(`capstone_submission/`, #571).

Byte boxes (all DERIVED from `S = 100·d_seg + √(10·d_pose) + 25·B/37,545,489`):

| box | assumption | total archive B | B/pair (600) |
|---|---|---:|---:|
| **Operator box (sub-pointer)** | d_seg 1.52e-4, pose contribution ≈ 0 | **≤ 264,320** | **440.5** |
| Honest sub-pointer box at the MEASURED spine distortions | d_seg 1.52e-4 + d_pose 1.02e-4 (contribution 0.0319) | ≤ ~216,300 | ~360 |
| Sub-0.15-BY-RATE line (zero distortion) | d=0 corner | ≤ 225,272 | 375.5 |
| SPEC ceiling at measured positive-band pose | d_seg=0, d_pose 2.522e-5 | ≤ 201,422 | 335.7 |

**Honesty flag:** the 264 KB/440 B figure appears in NO corpus file (checked by the R-D extraction pass and
by direct grep) — it is the operator's derivation, re-verified arithmetically here, and it silently assumes
the fp32-preimage pose noise (1.02e-4 → 0.0319 S) is eliminated. Until the tie-aware preimage selector
exists, the real box is ~216 KB. The Seg break-even that prices every marginal trade:
**150,181,956 B per unit d_seg = 150.18 B per 1e-6 d_seg = 0.2503 B/pair per 1e-6**
(`seg_secant_rd_curve_20260719_codex.md:27`).

**The measured R-D curve today has a 4-order-of-magnitude hole.** The two measured endpoints:
- generator endpoint (#548 rung B): **139.7 B/pair @ d_seg 3.455e-3** (and d_pose 63 — plane RMSE 25 off source);
- residual endpoint (seg-secant): **1.77 MB/pair @ d_seg 1.63e-4** (precision_drop1; whole family 1.12–2.22 MB/pair).
Nothing has ever been measured in between. The open axis IS this hole; every post-hoc way of crossing it is
measured dominated (rows 6–8 below). The bytes must buy STRUCTURE inside the generator/solve, not
per-pixel residual.

---

## 1. RANKED MASTER TABLE — description families

Ranking = proximity to the 440 B/pair box AT capstone distortion (rate AND distortion jointly), i.e. which
family can plausibly place a point inside the box soonest.

| # | family | what exists (built/measured/designed) | best MEASURED datum (source) | extrapolated B/pair | gap vs 440 B/pair box | single named blocker |
|---|---|---|---|---|---:|---|
| 1 | **Trained/solved ŷ-GENERATOR (witness-as-ŷ; C2 integer-plane emitter)** | ep725 witness archive byte-closed through the exact lattice (#548 rung B); C2 emitter BUILT (`src/tac/boundary_math/integer_plane_emitter.py`, 228 tests, NumPy/Torch parity, U4 head basis, train-least deletion executable; fixture exact 589,824/589,824 numerators) | **83,838 B total (139.7 B/pair), n600 rate 0.0558, d_seg 0.003455, d_pose 63** (`yhat_rd_ladder_20260719_codex.md` rung B, MEASURED archive bytes; d_pose is a GENERATOR artifact of plane RMSE 25 — instance scope) | 140 (today) | rate **3.1× UNDER** the box; d_seg **23× OVER** the 1.52e-4 endpoint; pose fails until plane≈source | **no trained run of the emitter within source-centered margin bands** — the operator-confirmed reframe ("solve ŷ within seg bands AROUND THE SOURCE, pay only residual-vs-free-predictor; pose falls out") has zero measured points (C2 charter owed) |
| 2 | **Per-stratum TASK-SPACE generators (Road-Lane polynomial + dash-phase; static MyCar seed)** | Wave-F LBND2 coder LANDED + bit-exact inflate (n600); openpilot deg-4 poly cross-ref + IPM geometry (focal 910, VP (256,174)); lane head-start conditioning measured; analytic lane band L71 | **41,526 Brotli B n600 = 69 B/pair** for 2,967 lane lines (Shannon floor 26,179 B = 44 B/pair; quantization-induced lateral RMS 0.0212 m) (`wave_f_lane_band_rd_rate_n600_measured.json`); lane-conditioning recovers **64.7%** of lane d_seg (0.005855→0.002069 n600, `openpilot_lane_headstart_landed_20260629T193648Z.md`); analytic band d_seg **0.00087** (L71) | 44–69 (lane component only); ground-frame SE(3) target 1–4 KB total UNMEASURED | lane component fits inside the box with ~370 B/pair to spare — but covers ONLY the Road-Lane stratum (61% of edge H-floor, 66% of witness residual) | **post-hoc composition is measured ≈0/negative** (Wave-F band composed with witness ≈ break-even; flat-amplitude on witness +80.1%): the component must enter as a SEED/conditioning inside the generator/solve, and that joint form is unmeasured |
| 3 | **Contour/partition GEOMETRY codecs (H/K floors, MS codec, seg-core)** | seg-core contour codec (d_seg=0.0 by construction, #52); MS partition codec RD-swept (#180); necessity per-stratum floors registered (`realization_necessity_preimage_per_stratum_v1`) | H-floor **303,047 B n600 (505 B/frame)**, Road-Lane 309.6 B/f = **61%**; K-ladder eps=1px **143,552 B (239 B/pair), K/H=0.47** (`necessity_solver_inverse_factorization_20260715.md`); MS codec eps=0.5: 740 B/frame @ d_seg 5.57e-4 → S≈0.370 DOMINATED (`morse_smale_partition_codec_feasibility_20260626.md`) | 239–505 | K-ladder INSIDE the box at geometric eps=1px; H-ladder 15% over; MS codec dominated standalone | **DP-eps→d_seg calibration + self-delimiting codec + receiver absent** — floors are geometric, not d_seg-verified; and temporal coding measured NOT to collapse rate (motion-comp ceiling −3.3%) |
| 4 | **Head-description packet (PDW1→PDW2 gauge-fixed, #539/#553)** | power-diagram identity EXACT (head = rank-4 Laguerre diagram, 0/200,000 label mismatches); PDW2 codec byte-identical parse-back, frame-195 fp32 tie reproduced exactly | **138 B raw / 133 B Brotli** margin-preserving (PDP2 134/122), one GLOBAL packet vs PDW1 338 B (`pdw2_gauge_packet_probe_20260719_codex.md`, byte-anchor MEASURED) | **0.22 amortized** (global packet ÷ 600) | negligible bytes — but it describes only the HEAD's channel-space cells, not the video | **the spatial/RGB feature-field pullback is absent** (`TARGET_ONLY_VS_REALIZATION_NON_EQUIVALENT`; Nielsen stop rule fired — stop packet polishing) |
| 5 | **Band-slack (source-centered margin bands + KKT waterfill, #549/#536/secant)** | VJP custody closed (24 sidecars, Seg field-VJP + PoseNet-6 Jacobian); positive-band curve measured; secant curve 9 points; #536 returns `MEASURED_SECANT_KKT_CANDIDATE` (margin_m0p3 ↔ precision_drop1, marginal gap 4.14e-12 score/global-B) | bands buy **−37.6073%** rate (1,474,580 vs 2,363,386 Brotli B/pair at Seg scale 1e-4, τ 2.5e-4, d_seg=0); **wider bands cost MORE** (scale 1e-3: 1.81–1.90 MB/pair, 40–41 repairs vs 9 — repair-dominated) (`vjp_custody_positive_bands_20260719_codex.md`) | n/a as standalone (1.47 MB/pair) | as a full-plane residual: **3,300× OVER**; as the SLACK TERM of a generator description: the −37.6% law is the only measured price of slack | **only measured in full-plane range-coordinate form**; the banded-description-of-a-GENERATOR point (slack around a compact base) is unmeasured |
| 6 | **Exact-residual / predictor-residual planes (#541/#549/direct)** | 3 rule-118-free predictors built + measured (copy/affine6-Q12/smooth-121); zero-band joint solve exact; rung-E n48 archive parses + inflates | floor = **335,729.6 B/pair conditional / 664,002 complete** (smooth-121, n48 MEASURED codec bytes) (`constructive_solver_541_20260719_codex.md`); zero-band range residual 2,337,608 B/pair (#549); direct plane i32 1,572,790 B/pair, rate 628 (#548 rung A); direct frames 1.70 MB/frame, rate ≈680 | 335,730+ | **762× OVER** — RATE-DEAD, formulation scope, family CLOSED for payload use | none — this is the settled negative; do not re-spend here (its live value: residual byte mass is dominated by the HIGH-margin interior — `[2,∞)` band 15.3 MB vs sub-1.0 bands <0.9 MB — i.e. a band solve spends nothing where the exact residual spends most) |
| 7 | **Post-hoc flip/edit sidecars (#307 contour-string, #369 DMTz)** | contour-string flip coder measured on real witness residual n600; DMTz adversarial review completed | **0.8201 B/flip MEASURED** (witness 441,329 flips, 142,270 components); DMTz 1.254 B/flip DERIVED; admit bar **0.45–0.65 B/flip** (`contour_string_flip_coding_n600_20260707.md`, `dmtz_adversarial_review_and_technique_search_20260709.md`) | rate_S if stored ≈ 0.241 | DOMINATED (bar missed 1.3–2.8×; residual = "fragmented confetti", mean component 3.1 px, 44.6% singletons) | reactivation requires ~3× residual coherence — **a TRAINING outcome, not a coder trick** (routes back to family 1) |
| 8 | **Quotient/setoid rate law (#155) + Cole-Hopf (#542)** | rate law DERIVED-EXACT: `R_sem = H(U(W))`; pay only the shortest receiver-reachable representative of the evaluator-kernel class; constructive-quotient debt `H(q_G|U) ≥ 0`. Cole-Hopf = logsumexp-of-argmax; n600 Gibbs target build 2.51 s, top-1 prob 0.9958 | no byte gain measured for either (`infdesc_setoid_quotient_rate_law_equation_feed_20260713.md`; state review §alt-forms) | UNKNOWN | design guidance only | quotient: fiber-completeness + receiver section owed; Cole-Hopf: preregistered preimage gate never fired (initializer-only claim) |
| 9 | **MDL/K brackets (bounds, not carriers)** | contour-form Seg+ξ upper ≈ **235,974 B** (rate-only 0.1571 — 10.7 KB OVER the 225,272 sub-0.15 line); real exact-Seg context code **255,288 B** measured; K lower bound honestly TRIVIAL_ONLY (`mdl_ms_complex_K_lower_bound_20260718.md` — FALSIFIED_AT_CLAIM_LEVEL for any "MDL=lower bound" reading) | n/a | brackets the box: best-known LOSSLESS-Seg description is ~5–13% over the sub-0.15 line, so lossy-within-bands is REQUIRED, not optional | the bracket's lower side is trivial — K itself unmeasured |

**Supporting geometry (constrains what the description must pay for — all verified first-hand):**
strict-necessary camera support **1.66%** of pixels (edges 1.646% + saddles 0.016%); interiors 97.8% of
scorer pixels with margin median 5.89 (2.67% < 1.0, 0.28% < 0.1) — **except Lane interiors (margin med
1.54, no safe interior)**; ker(A) = 22.70% of camera pixels certified free + ~52% of decoded-frame energy
scorer-invisible (precision lever, not rate lever — canonicalize gauge BEFORE int8 for 22.3% finer scale);
head EXACT rank-4 with flip law `d = |m|/‖Δw‖`, all four Lane normals largest (3.75–4.01); Road-Lane =
77% of skip-limited flips, ERF r50≈85 px; frame_0 seg-free; chroma <2px pose-invisible (boundary-RGB
carries pose-safe by construction); witness-own residual = **90.6% edge-flicker, Road-Lane 66%**,
flat-amplitude EXHAUSTED (+80.1% on witness), cure = non-local LUMA (92–94% of cure-gradient energy),
crisp>blurry. Sources: `necessity_solver_inverse_factorization_20260715.md` ·
`null_subspace_rate_measure_20260717.md` · `frozen_scorer_exact_factorization_20260715.md` ·
`segnet_recursive_fractal_factorization_20260715.md` · `c2_witness_own_decomp_20260716.md`.

---

## 2. THE COMPOSED BEST-CASE STACK (with non-additive-pools honesty)

Composition (each element cites its measured anchor; the stack is a DESIGN, not a measured row):

1. **Base: counted ŷ-generator** (family 1) — the C2 integer-plane emitter trained/solved
   **within source-centered margin bands** (the operator-confirmed reframe: pose then falls out via
   `pose_plane_proximity_corollary_v1`, confirmed 96/96 inactive at in-band solutions). Byte anchors for
   the base: ep725 archive 83,838 B; donor int8+Brotli coder 63,394 B base + 20,518 B pair codes (SPEC E6);
   block-FP 13,957 B @1.005 bits/param (E7 — inadmissible until charter C7). **Plausible base: 60–85 KB.**
2. **Per-stratum seeds inside the base (NOT post-hoc):** Road-Lane openpilot deg-4 polynomial + dash-phase
   conditioning (Wave-F coder 41.5 KB n600 today; SE(3) ego-factorization Stage-2 target 1–4 KB;
   analytic-band d_seg existence proof 0.00087) + one-time static MyCar hood seed (IoU 0.994) + horizon
   band. **Plausible increment: +4–42 KB.**
3. **Head packet** PDW2 margin-preserving +133 B (contingent on the spatial-receiver gate C3) — negligible.
4. **Band-slack repair term:** the #536 waterfill allocates residual bytes ONLY where the generator misses
   its band — priced by the measured slack law (−37.6% at scale 1e-4; wider = repair-dominated, so the
   allocator must stay at small scale) and the secant break-even 0.2503 B/pair per 1e-6 d_seg.
   **UNMEASURED for a generator base — this is the open quantity.**
5. **Preimage tie-aware selection** in the lattice solve — removes the coarse-description fp32 floor
   (d_seg ~1.2e-4 measured for predictor-optimal preimages vs 0.19 mismatch/pair replay) AND the spine's
   pose 1.02e-4 → recovers ~0.047 S of distortion and widens the box 216→264 KB. Zero payload bytes.

**Best-case DERIVED total: ~65–130 KB (108–217 B/pair) + the unmeasured repair term**, vs the 264 KB box —
i.e. the rate side has ~2–4× headroom IF AND ONLY IF the banded-generator reaches d_seg ≲ 1e-3 territory,
which no measured point supports yet (best generator d_seg 0.003455). Applying the break-even, closing
0.003455 → 1.5e-4 by residual bytes alone would cost ≥ 3,300e-6 × 150.18 B ≈ 496 KB even at the
theoretical break-even price — over the whole box on its own. **The distortion gap cannot be bought with
residual bytes; it must be closed inside the generator.**

**Non-additive-pools note (binding, `opportunity_pools_non_additive_rate_distortion_reachable_20260718.md`):**
Road-Lane is ONE pool (≈0.208 S of the palette-total 0.3146; 66% of the witness residual) claimed
simultaneously by the lane-polynomial seed, the contour floor share, the skip-limited flip mechanism, and
the generator's own training — these COMPETE, never sum. Two measured composition failures enforce this:
(a) Wave-F band composed post-hoc with the witness ≈ 0/slightly negative on flips; (b) the palette-vehicle
β=2 one-sided band (−29.5% @0 B on palette) is **+80.1% on the witness** (flat-amplitude exhaustion,
formulation-scoped law). Every stack element must therefore enter as conditioning/seed INSIDE the joint
solve/training, and the stack's value must be measured as ONE row, never as summed per-element ΔS.

**The single binding blocker of the whole program:** no measured point exists in the byte-vs-d_seg hole
between 140 B/pair @ 3.455e-3 and 1.12 MB/pair @ 7.5e-3 / 1.77 MB/pair @ 1.63e-4 — the banded-generator
mid-curve is empty. Everything else (receiver, lattice, pose, KKT instrument, break-even, geometry) is
built and measured.

---

## 3. TOP-3 NEXT MEASUREMENTS

1. **First mid-curve point: C2 banded-generator run at n600 through the byte-close.** Tooling EXISTS:
   `integer_plane_emitter.py` (built, 228 tests) + margin-band law in `joint_seg_pose_rate.py` + #543
   receiver + `levelset_byte_close_and_eval.py`; glue ≈ trainer config + band loss (~300–600 LOC).
   Decisive answer: (bytes, d_seg, d_pose) of a COUNTED generator solved within source bands — does it land
   under ~150 KB at d_seg ≤ 1e-3? This single row populates the hole and prices the whole program.
2. **Tie-aware preimage selector A/B on the officially-scored spine.** ~200–400 LOC inside the factor-2
   lattice solve (choose among exact preimages by fp32 margin at ties), then re-measure the n24/n600
   advisory chain (and re-score the spine archive when GO-gated). Decisive: does d_seg 1.52e-4 → ULP class
   (~1e-6) and d_pose 1.02e-4 → 1e-9 class? Worth ~0.047 S of pure distortion at ZERO payload bytes, and it
   converts the honest 216 KB box into the operator's 264 KB box.
3. **Stratum-seed efficacy inside the solve (Road-Lane generator swap).** Compose the Wave-F/analytic lane
   band as a SEED of the band solve (not post-hoc) on n24: measure Δbytes and Δd_seg against the 0.2503
   B/pair-per-1e-6 break-even. Tooling: Wave-F coder (landed) + #549 solver; glue ~300 LOC. Decisive: does
   per-stratum conditioning beat the flat-amplitude wall when it enters INSIDE the solve — i.e. is the
   composed stack real or does pool-competition eat it?

---

## 4. OPEN QUESTIONS FOR THE ONLINE ARM (each marked ONLINE)

1. **ONLINE — coding piecewise-constant argmax/label fields near a known partition:** best-known
   rate-distortion results for lossy coding of label maps / Laguerre-power-diagram cell complexes
   (computational-geometry compression, e.g. optimal bit allocation for weighted Voronoi/power diagrams;
   tropical-polynomial sparsification). Target: does published theory beat our measured K-ladder
   (239 B/pair at eps=1px) for equivalent boundary fidelity?
2. **ONLINE — constraint-satisfying implicit/learned generators:** prior art on training compact
   generators subject to per-pixel half-space (margin/hinge) constraints around a reference field
   ("certified" INR compression, learned quantization with feasibility constraints, ROI/margin-constrained
   neural codecs). Anything that prices bytes-vs-constraint-slack directly informs the banded-generator run.
3. **ONLINE — float32 argmax tie certification at scale:** interval/affine-arithmetic or exact-rational
   certification of fp32 argmax stability through fixed bilinear resize + frozen conv nets (the frame-195
   ULP class) — methods cheap enough for 118M pixels to drive the tie-aware preimage selector.
4. **ONLINE — SE(3)-factorized lane/trajectory stream coding:** HD-map / lane-graph / clothoid-spline
   compression literature (ego-motion-compensated polyline coding, ODE/clothoid intrinsic coders) — the
   Wave-F Stage-2 target (41.5 KB → 1–4 KB) needs the best-known factorization of per-frame lane fits into
   one ego trajectory + slowly-varying ground-frame geometry.
5. **ONLINE — MDL for cell complexes with lower-bound legitimacy:** any published technique giving
   NON-trivial lower bounds on description length of planar partitions (e.g. via combinatorial entropy of
   triangulations/flip graphs) — our K lower side is honestly TRIVIAL_ONLY.

---

## 5. DAG FEED stub (FEED-generator-description-crux-synthesis)

- **Node:** consolidation of the ONE open axis (bytes(generator + band-slack)) into a ranked map;
  no pointer motion; no new measurement.
- **Edges in:** #548 rung B (83,838 B / 0.003455) · seg-secant (break-even 150.18 B/1e-6; 9 points) ·
  #549/#536 (`MEASURED_SECANT_KKT_CANDIDATE`) · vjp-bands (−37.6%, repair-domination, 96/96 pose inactive) ·
  #541 (~336 KB/pair floor; fp32 debt 1.23e-4) · #553 PDW2 (138/133 B) · #539 (rank-4 power diagram exact) ·
  official spine (S=272.73; d_seg 1.5196e-4; d_pose 1.0184e-4) · necessity (1.66%; Road-Lane 61%) ·
  Wave-F (41,526 B n600) · lane head-start (64.7% recovered) · MS codec REVISE · #307/#369 dominated ·
  flat-amplitude exhaustion (+80.1%) · MDL bracket (~236 KB vs 225,272 line).
- **Edges out (owed):** M1 C2 banded-generator n600 row → the first mid-curve point; M2 tie-aware preimage
  A/B → distortion recovery ~0.047 S; M3 stratum-seed-inside-solve A/B → composed-stack admissibility;
  ONLINE arm items 1–5.
- **Consistency note for the triality:** the operator's 264 KB/440 B box is DERIVED-here (not in any prior
  artifact) and is conditional on M2; the corpus-native anchors are 225,272 B (sub-0.15-by-rate) and the
  spine's measured 0.047 distortion. No equation registration is performed by this memo
  (# FORMALIZATION_PENDING: consolidation memo — every constituent number is already anchored in its source
  artifact's equation/law; the only new derivation is the budget-box arithmetic, which is the frozen score
  law evaluated at measured points).

## STORES CONSULTED

CLAUDE.md · docs/operating_manual_craft_handoff.md · memory/MEMORY.md + the 07-19 lattice ledger
(`seg_and_pose_solved_exact_lattice_realization_one_rd_axis_20260719.md`) ·
`.omx/research/{yhat_rd_ladder,seg_secant_rd_curve,joint_seg_pose_inverse_solve,vjp_custody_positive_bands,
constructive_solver_541,pdw2_gauge_packet_probe,power_diagram_witness_20260718,v10_lattice_rate_verdict_and_composition,
v10_flattened_lagrangian_kkt_derivation,spec_v10_reconciliation_and_kkt_verify_20260719_fable,
SPEC_v10_capstone_RECONCILED_20260719,SPEC_v10_integer_plane_vehicle_20260719,
v10_capstone_first_byteclosed_row_20260719,c1_two_plane_receiver_timing_20260719_codex,
c2_integer_plane_emitter_build_20260719_codex,inverse_solve_completeness_matrix_20260718,
mdl_ms_complex_K_lower_bound_20260718,necessity_solver_inverse_factorization_20260715,
null_subspace_rate_measure_20260717,frozen_scorer_exact_factorization_20260715,
segnet_recursive_fractal_factorization_20260715,morse_smale_partition_codec_feasibility_20260626,
dmtz_adversarial_review_and_technique_search_20260709,infdesc_setoid_quotient_rate_law_equation_feed_20260713,
contour_string_flip_coding_n600_20260707,wave_f_lane_band_rd_code_LANDED_stage1_measured_20260702,
comma_openpilot_crossref_polynomial_geometry_20260619T014433Z,openpilot_lane_headstart_landed_20260629T193648Z,
c2_perclass_stratum_carrier_taxonomy_20260716,c2_witness_own_decomp_20260716}.md ·
`wave_f_lane_band_rd_rate_n600_measured.json` ·
memory/{opportunity_pools_non_additive_rate_distortion_reachable_20260718,
c2_witness_own_decomp_flat_amplitude_exhaustion_20260716,MEMORY_established_findings_cluster_20260717}.md.

**Pointer delta: 0.** This memo is MEANS (a map), not an exact row; the next exact-relevant actions are M1–M3.
