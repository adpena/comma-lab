# v2 witness — PER-ELEMENT smallest-representation spec + the n600 STACK BUDGET (Lens-D C1)

- **UTC:** 20260630T013253Z
- **Authority:** `[macOS advisory / research-MARSHALING]` — `score_claim=false`, `promotable=false`,
  `ready_for_exact_eval_dispatch=false`. **Pointer UNMOVED contest-CPU 0.19110.** This is a MEANS
  (a build spec + a paper byte budget). No score moves; means≠ends.
- **Mode:** $0, CPU/numpy, NO GPU, NEVER-MPS-authority. MARSHALING pass — every decision is CITED to a
  prior measured artifact; nothing here is reinvented. GROUNDED (measured) vs ESTIMATE flagged per row.
- **Deliverable for:** the Lens-D C1 gap — *"NO end-to-end stack arithmetic exists; every favorable
  number is a SLICE; the excluded terms ALL push score UP"* (`adversarial_review_round1_coherence_
  20260630T004003Z.md`). This memo writes down the sum the chain never wrote down.

---

## 0. The two prior budgets DISAGREE — this is the headline finding

The chain has TWO assembled budgets that disagree by ~50× on the dominant (bulk) term:

| Budget | source | bulk treatment | counted bytes/600 | implied rate | projected S |
|---|---|---|---:|---:|---:|
| **OPTIMISTIC (F4)** | `free_generator_irreducible_info_byte_budget_20260629T182515Z.md` | bulk = **FREE pose-warp** of ONE canonical (needs NO trained INR) | **~3,180 B** | 0.0021 | **0.107** |
| **PESSIMISTIC (C1/C2)** | `adversarial_review_round1_coherence_*.md` | bulk = **stored warped CONTENT** that turns over with n | bulk-only ≈ **155 KB** (rate 0.103) | ≥0.103 + stack | **≥0.19, plausibly WORSE than frontier** |

Both budgets AGREE the learned lane-survival long-tail residual is the genuine unknown. They DISAGREE on
whether the BULK (Road/sky/hood, ~98% of area, 3 of 5 classes) is **cheap** (F4: deterministic warp of one
canonical scene → ~0 marginal bytes) or **expensive** (C2: a 60-s forward-driving clip turns the scene
over ~15–17×, so a single canonical can't cover 1.7 km of novel road → you need MPEG-style keyframes whose
cost GROWS with n). **The deciding measurement — "how few canonical keyframes can generate the 1200 frames
at task-survival d_seg?" — does NOT exist.** It is the build-gated unit (FEED phase-1 §"Next unit" step 2).
This is the honest C1 answer: the favorable per-element slices have a sub-0.15 path ON PAPER **only in the
optimistic-bulk regime, which is unmeasured at n600.** (§2 closes this out.)

---

## 1. THE PER-ELEMENT DECISION TABLE

Witness = the SDS-TSC 6-section grammar (`stratified_dynamic_sfm_taskspace_codec_design_20260629T182602Z.md`)
crossed with the generate-vs-store partition (`v2_originality_provenance_synergy_citations_20260629.md`
§"generate-vs-store partition", decided EMPIRICALLY per component by measured $0 score-cost).

Columns: **FITS** = which archive section · **S/L/G** = Store / Learn(trained weights) / Generate(free
algorithm) · **QUANT** · **MEAS** = how its size/d-cost was measured · **BYTES/600** (G=GROUNDED, E=ESTIMATE)
· **CITE** (DAG-FEED / artifact).

| # | Component | FITS | S/L/G | QUANT | MEAS | BYTES/600 | CITE |
|---|---|---|---|---|---|---:|---|
| 1 | **Calib header / globals** (EON intrinsics, scene globals) | S0 | **GENERATE** intrinsics (pinned, FREE) + **STORE** ~globals | raw f32 → ~32–128 B | known fixed intrinsics | **~64 B** (E) | `calibrated_geometry.py` pinned EON @384×512; SDS-TSC S0 |
| 2 | **Ego-pose / SE(3) twist stream** | S2 | **STORE** (tiny, dual-use with d_pose) | temporal-Δ + range-code; col0 (fwd-speed) is the sole non-trivial cost, cols1–5 near-static | order-0 entropy of Δ × N + real LZMA; d_pose floor = `mean((round_q−true)²)` on EXACT PoseNet target | **~875 B** (G, range-code @ d_pose 6.3e-5) | F4 table; `gt_n600.npz['gt_poses']`; range_coder.py |
| 3 | **Canonical scene descriptor** (static IPM/ground-frame scene C) | S1 | **STORE** descriptor + **GENERATE** rasterization | ground-frame coords; lossless mode-partition = 480 B | per-pixel temporal-MODE partition, lossless `partition_description_bytes`; **LOSSY @ d_seg 0.021** | **see §2 BULK** — 480 B (lossless static) **only covers the LOW-turnover regime** | F4 (b); FEED-jh (openpilot centerline recovers 64%); FEED-jm (ground-frame 0.5–5 KB target vs 65 KB image-space) |
| 4 | **Bulk warp-type mask** (class→regime dispatch) | S3 | **GENERATE** (FREE: Road→ground-H, MyCar→identity, sky→rotation) | n/a (class-keyed dispatch) | per-class dispatch is parameter-free | **0 B** (G) | generate-vs-store table; FEED-iz stratified warp |
| 5 | **Bulk per-frame jitter** (the SegNet flip-jitter on Road/sky/hood) | S1/S4 | **OPEN** (generate-via-clean-canonical **vs** store-per-frame) | TBD | per-frame-warp through R = **4× budget** = the SegNet jitter floor | **UNRESOLVED** (a95b0ad6 budget gate) | FEED-jq; clean-canonical budget gate `a95b0ad6` |
| 6 | **Lane SDF / boundary carrier** | S1 | **GENERATE** (FREE 1-Lipschitz SDF rasterizer; survives R) | SDF ramp; render ≥192 (ideally 320) to survive R Nyquist | single-SDF lane d_seg 5.9e-4 @192 / 1e-5 @320; MSDF FALSIFIED | **0 B** (rasterizer FREE; G) | FEED-jk; `lane_sdf_component.py`; `eikonal_sdf_dseg_recovery_test_*` |
| 7 | **Lane ragged / survival residual** (THE binding wall) | S4 | **LEARN** (trained flow-matching from prior) + **STORE** irreducible jitter as margin-keyed annulus dither | trained weights quantized (int8 sensitivity-aware, §3); dither = annulus-only | post-R lane d_seg 4.2e-4→8e-4 (structured-SDF existence proof); polynomial can't collapse it | **~1.5 KB** (lane SDF manifold, CITED-GROUNDED) **+ trained-weight VARIABLE (the unknown)** | F4 (c); FEED-dm/du; FEED-js (GNVC-VD/OT-NFM/PICM-Net) |
| 8 | **Movables (class-3 / cars)** | S5 | **STORE** (templates + low-rank trajectories) | template + rank-≤2 trajectory; range-code | warp-predict floor 0.00082 (=67% of movable budget) → store ≈ 0 | **~0.75–2.7 KB** (E, CITED grok GAP-1 / F3-FEED-je) | F4 (movables); `movables_multibody_residual_*` |
| 9 | **Margin field** (if stored separately) | S4 | **GENERATE** (margin = Fisher surrogate, co-located w/ curvature ρ=0.978) — **do NOT store** | n/a | curvature↔−margin 0.978 co-location → margin is derivable, not a stored channel | **0 B** (G; derive) | `colocation_fisher_stress_anisotropy_test_*` (b0bee924e) |
| 10 | **Trained residual WEIGHTS** (the LEARNED long-tail generator) | S4 | **LEARN** → store minimal-by-construction rep (§4 collapse) | int8 sensitivity-aware mixed (§3); rank-floored to intrinsic dim | bc20 full vehicle = 89,244 B @ rate 0.0594; the v2 residual-ONLY generator is far smaller but **UNMEASURED** | **VARIABLE — THE GENUINE UNKNOWN** | bc20 `_fire_g3_basin_baseline`; SDS-TSC S4 6–20 KB ESTIMATE |

**Reading of the table:** rows 1,2,4,6,8,9 are GROUNDED-small or FREE (sum ≈ **~3.2 KB** counted). Rows 3,5,7,10
— the **canonical-content / bulk-jitter / lane-survival / trained-weights** — are the entire game, and they are
exactly the four the chain has NOT byte-closed at n600. The optimistic budget zeroes rows 3+5 (bulk free);
the pessimistic budget says rows 3+5 dominate.

---

## 2. THE END-TO-END n600 BYTE BUDGET (the C1 sum — written down on paper)

`S = 100·d_seg + √(10·d_pose) + 25·bytes/37,545,489`. Budget anchors: sub-0.19 ⟺ archive ≲ **177 KB**
(rate 0.118); sub-0.15 ⟺ archive ≲ 108 KB at PR95-class d_seg+pose. Three honest regimes:

### Regime A — OPTIMISTIC (bulk is FREE pose-warp; F4)
| section | bytes/600 | seg contrib | pose contrib | rate |
|---|---:|---:|---:|---:|
| pose stream | 875 | (Road d_seg FREE via grok +16%) | 0.025 (d_pose 6.3e-5) | — |
| canonical per-class SDF (lane+hood) | 1,556 | lane 4.2e-4 / hood 7.4e-4 | — | — |
| warp-type mask | 0 | — | — | — |
| movables | 750 | ~8e-4 | — | — |
| **trained long-tail residual** | **VARIABLE** | **drives d_seg 0.018→~8e-4** | — | — |
| **TOTAL counted (excl. learned residual)** | **~3,180** | d_seg≈8e-4 → **0.080** | **0.025** | **0.0021** |
| **→ projected S** | | | | **≈ 0.107 ✓ sub-0.15** |
**Threshold (rate negligible, pose fixed): sub-0.15 ⟺ d_seg ≤ 1.23e-3; sub-0.19 ⟺ d_seg ≤ 1.63e-3.** The
cited frontier d_seg need (~6e-4–1e-3) sits INSIDE the sub-0.15 window. **In this regime rate is NOT the
constraint; sub-0.15 reduces ENTIRELY to the learned d_seg residual + R-survival.** (F4, MEASURED-rate.)

### Regime B — PESSIMISTIC (bulk content turns over; C1/C2)
- bulk-only deterministic store (Road/Undriv/MyCar) ≈ **0.103** at n96 (FEED-ko, real clustering coder).
- C2 physics: 60 s × ~28 m/s ≈ 1.7 km, depth ~100 m → scene turns over **~15–17×** → one canonical can't
  cover it → MPEG-style **keyframes whose cost GROWS with n** (n96 = the artificially-cheap low-turnover
  regime). FEED-jz independently measured d_seg WORSENING with n (bulk floor 2.4×@n96 → 3.5×@n200).
- Stack: bulk **0.103** + lane wall (unmeasured) + movables + pose + keyframe-turnover (a rate-INCREASER) +
  coder overhead → **plausibly EXCEEDS 0.19110 (WORSE than the current frontier).** No path shown to sub-0.19.

### Regime C — bc20 standalone (the PROVEN byte-close, a DIFFERENT vehicle = HNeRV recode, not the warp codec)
- GROUNDED: archive **89,244 B**, rate **0.0594**, d_pose **0.00034**, parity-verified
  (`_fire_g3_basin_baseline/g3_packet_manifest.json`).
- Live S = 100·0.00385 + √(10·0.0002) + 0.0594 = **0.489** (k=8); Muon-floor-projected ≈ **0.31**. BOTH ≫
  0.19110. To beat the pointer bc20 needs **d_seg ≤ ~0.00087** (a ~2.4× close from the Muon floor) — NOT yet
  shown. So even the proven small-basis is NOT sub-0.19 standalone. (DAG line 540, GROUNDED.)

### HONEST C1 VERDICT
**The favorable slices DO NOT obviously sum.** A sub-0.19 / sub-0.15 path exists ON PAPER **only in Regime A**,
and Regime A's central assumption (bulk = free pose-warp of one/few canonical scenes across the full turning-
over clip) is **the single biggest UNMEASURED quantity in the plan** — directly contradicted by Regime B's
keyframe-turnover physics. The two regimes differ ~50× on the dominant (bulk) term. **The chain cannot
truthfully claim a sub-0.19 path until the one deciding measurement is run: how many canonical keyframes are
needed to generate the 1200 frames at frontier d_seg, and what do they cost at n600.** Until then the honest
status is: *rate floor is tiny IF bulk warps for free; that "if" is unproven and the physics argues against it.*

---

## 3. QUANTIZATION (per-tensor sensitivity bit-allocation) — the marshaled scheme for rows 7 & 10

For the LEARNED weights (lane-survival generator + any trained residual), the researched-optimal allocator is
**reverse-waterfill / KKT on the exact-sensitivity field** (NOT a flat int4/int8):
- **#157 exact-sensitivity KKT + codec co-design** + **#69 score-aware weight re-quant**: allocate bits so the
  marginal d_seg/byte is EQUALIZED across tensors (reverse-waterfill on the saliency map) — concentrate ~all
  capacity on the **0.72% lane boundary band** (the only factor with multiplicative headroom; DAG line 510/597).
- Hooks (GROUNDED in-tree): `tac.sensitivity_map`, `tac.pareto_*`, `tac.score_lagrangian`
  (`score_marginal_lagrange_multipliers_v1`: λ_pose/λ_seg = 5/(√(10·d_pose)·100), operating-point dependent),
  `tac.uncertainty_weighted_loss` (Kendall+Lin per-pair). Per-byte leverage is ~uniform on entropy-coded
  archives (`per_byte_leverage_uniformly_distributed_v1`) → the win is the **allocation across tensors**, not
  per-byte edits.
- **Finishing-kit coders (L21–L32, ALL in-tree)** for the stored streams: per-tensor byte-maps (#21),
  split-brotli (#23), raw-LZMA (#24), temporal-Δ uint8 (#25), range/arithmetic coding
  (`src/tac/lossless/range_coder.py`, #30), colex-rank no-op (#31). **CALIBRATION (FEED-lb, GROUNDED):** the
  L21–L32 kit is **EXHAUSTED on the 0.19110 frontier** (already at entropy floor; byte_delta=0). It is NOT a
  free banked win on the current archive — it is the coding layer the v2 stored streams (rows 2,7,8,10) must
  use, where it pays on NOT-yet-coded bytes.
- **Distortion-quant of the frontier** (gentle/sensitivity-aware mixed int8/int6) is the alternative rate lever;
  prior naive int5 score-aware QAT capped ~0.49 (d_seg walls ~0.0035) — so it needs the sensitivity-aware
  mixed allocation above, not a flat low-bit pass. (GROUNDED `frontier_int5_score_aware_qat_*`.)

---

## 4. THE LEARN-STRATEGY — emergent collapse to intrinsic dim (rows 5, 7, 10)

**The smallest representation is best obtained by EMERGENT COLLAPSE, not hand-sizing.** For every LEARNED /
GENERATED element: INDUCE the latent/code to collapse to the task-relevant intrinsic dimension via a
rank/spectral regularizer DURING training, then store ONLY the discovered minimal rep → minimal BY
CONSTRUCTION. Anchor: a prior run's code blob "naturally shrank as the arch found the lowest-dimensional
representation possible and collapsed and fell through."

**The regularizers (GROUNDED, in-tree):** #110 latent-structure-inducing regularizer · the A6 nuclear-norm
θ*-TIER2 lever · the code-spectral-entropy DM1 build.

**GROUNDED intrinsic-dim anchors (the dims the arch keeps re-discovering — these are the STORE targets):**
| element | intrinsic dim | source (GROUNDED) |
|---|---|---|
| ego-coherent partition variation | **rank-8 (95.6% var)** | `gr-unified` / per-pair code manifold |
| boundary effective rank (participation ratio) | **~4.07 / 600** (top-1 mode 46%, k=6 for 80%) | `frozen_partition_topology_ego_deformation_*` |
| lane orbit | **~8-dim** | `eikonal_sdf_dseg_recovery_test_*` |
| pose codec | **rank-2 SVD → deeper = 1 scalar/frame (zoom rate)** | #140; `closure_reaudit_round2_*` (geometry-optimal) |
| FiLM DOF actually used | **~1.2 of 768** | `gr-unified-action-*` |

**THE OVER-COLLAPSE GUARD (CALIBRATION — both sides MEASURED):** collapse to the TRUE intrinsic dim =
minimal-sufficient (good); collapse PAST it = pathology. The **DM1 FiLM rank-1 collapse couldn't localize the
moving lane annulus → DEMOTED** (`project_gr_unified_action_*`). So the learn-strategy = induce collapse
*toward* the intrinsic dim **with a FLOOR/guard at the task-necessary rank (~8 for the lane orbit, ≥4 for the
boundary)**. ⚠️ Honest nuance (`deepmath_multiscale_bridge_hunt_*`): "~8-dim lane-orbit" is REFUTED as a
chart for the FULL partition — it is the LANE-orbit intrinsic dim specifically, not a full-scene latent. Set
the rank floor per-element (lane ≥8, boundary ≥4, pose ≥1–2), never a single global rank.

---

## 5. v2-DONE-RIGHT BUILD SPEC — BUILT vs UNBUILT + the per-element optimal choice

Per the session CURRENT memo + phase-1 smoke (in-tree audit):

**BUILT (primitives exist; the missing piece is COMPOSITION, not from-scratch geometry):**
- Real ground-homography warp: `src/tac/calibrated_geometry.py` (pinned EON intrinsics, Faugeras decomp) +
  `src/tac/se3.py` (exp/log map). REPLACES the broken d_pose=190 crude integer roll.
- Real residual entropy coder: `src/tac/lossless/range_coder.py` (RangeEncoder/Decoder, encode_static_symbols).
- Byte-close + small-n exact-eval harness: `experiments/contest_auth_eval.py` + the `--batch-size n` $0
  reduced-n trick (GROUNDED: store_raw n24 → d_seg=0,d_pose=0 exactly → apparatus trustworthy).
- Boundary_math S1 components: `lane_sdf_component`, `hood_static_component`, `road_horizon_component`,
  `context_partition_codec`; `witness_dsl/gauge.py` (warp-chart selector); the bc20 G3 byte-close (89 KB).

**UNBUILT (the actual vehicle):**
1. **v2 6-section codec is DESIGN, NOT BUILT** — SDF/carrier/coder modules wired into nothing; the
   "store-canonical" scaffold is actually per-pair raw f0 (the 150–450× byte-close rate is the SCAFFOLD).
2. **The warp is BROKEN** (d_pose=190, units mismatch). FIX = wire the FEED-iz STRATIFIED per-class
   ground-homography (Road=ground-H(pose), MyCar=identity, sky=rotation-only KRK⁻¹) + learned survival
   residual; then MEASURE d_pose-through-R (NEVER measured with the real warp — Lens-D C3). Agent a6774036c.
3. **θ* campaign CONFOUNDED** (`--anneal-epochs` not threaded → warm-start arms re-heat temp → junk A/B).

**Per-element optimal choice the Round-2 build should use (do NOT reinvent):** rows 1–9 of §1's table as-is
(STORE pose ~875 B / GENERATE bulk-warp + SDF / STORE movables / derive margin); row 10 (the learned residual)
trained with the §4 emergent-collapse strategy and §3 sensitivity-aware int8 allocation; coded with the
in-tree L21–L32 range coder.

---

## 6. WHERE IT'S GROUNDED vs ESTIMATE — the $0 measurements that close C1

| claim | status | the $0 check that closes it |
|---|---|---|
| pose stream ~875 B @ d_pose 6.3e-5 | **GROUNDED** (F4 real entropy/LZMA on EXACT target) | — |
| lane SDF survives R (4.2e-4→8e-4) | **GROUNDED** (FEED-dm/du/jk existence proof) | — |
| bc20 89 KB / rate 0.0594 / d_pose 0.00034 | **GROUNDED** (parity-verified manifest) | — |
| L21–L32 exhausted on frontier | **GROUNDED** (FEED-lb, byte_delta=0) | — |
| **bulk = free pose-warp covers full clip at frontier d_seg** | **ESTIMATE (the C1/C2 crux)** | **keyframe-count probe: how few canonicals generate 1200 frames at task-survival d_seg + their n600 byte cost** (optical-flow/feature-track scene-turnover on 0.mkv) — THE deciding measurement |
| v2 warp-codec d_pose | **UNMEASURED for v2** (3.4e-5 transplanted from Quantizr supervised-render) | **PoseNet-through-R on the geometrically-warped frames** (materializer already builds them) — Lens-D C3, the cheapest check |
| movables ~0.75–2.7 KB | **ESTIMATE** (CITED grok GAP-1 / F3) | byte-close the movable templates+trajectories through range_coder |
| trained long-tail residual bytes + achieved d_seg | **THE UNKNOWN** (GPU-gated) | the exact byte-closed n600 row (the END) |

---

## 7. Wire-in (Catalog #125)
- Hook #1 sensitivity-map: ACTIVE — §3 allocator concentrates capacity on the 0.72% lane band (the only term
  with multiplicative headroom).
- Hook #2 Pareto: ACTIVE — §2 shows rate has slack ONLY in Regime A; in Regime B rate is the binding term.
- Hook #3 bit-allocator: ACTIVE — §1/§3 the per-element STORE/GENERATE split + sensitivity-aware int8 IS the
  allocation; pruned the lossy label-space coding (dominated).
- Hook #4 cathedral autopilot: N/A (advisory spec).
- Hook #5 continual-learning: ACTIVE — this memo + DAG FEED; closes Lens-D C1 (the n600 stack sum) on paper.
- Hook #6 probe-disambiguator: ACTIVE — §6 names the ONE deciding probe (keyframe-turnover) that disambiguates
  Regime A vs B.

## 8. DAG-FEED SUMMARY (tight)
**C1 CLOSED ON PAPER (the sum the chain never wrote):** per-element table (10 components) + n600 budget in 3
regimes. **Headline: the two prior budgets disagree ~50× on the BULK term** — OPTIMISTIC F4 (~3,180 B, S≈0.107,
sub-0.15 IFF d_seg≤1.23e-3) assumes bulk = FREE pose-warp; PESSIMISTIC C1/C2 (bulk ≈0.103 GROWING with
keyframe turnover) plausibly EXCEEDS 0.19110. **A sub-0.19 path exists ON PAPER only in Regime A, whose
central assumption is the single biggest UNMEASURED quantity and is contradicted by the C2 turnover physics.**
Even the PROVEN bc20 byte-close (89 KB, rate 0.0594) is S≈0.31–0.49 standalone, ≫ frontier. **The honest
verdict: the favorable slices do NOT obviously sum; the deciding $0-ish measurement (keyframe count to
generate 1200 frames at task-survival + v2's actual d_pose-through-R) is UNBUILT.** Marshaled (not reinvented):
per-element STORE/LEARN/GENERATE verdicts (generate-vs-store partition), sensitivity-aware KKT bit-allocation
(#69/#157), emergent-collapse-to-intrinsic-dim learn-strategy (#110/A6/DM1 + rank floor ~8 lane / ~4 boundary,
over-collapse guard from the demoted DM1 rank-1). means≠ends; **pointer 0.19110 UNMOVED.**
