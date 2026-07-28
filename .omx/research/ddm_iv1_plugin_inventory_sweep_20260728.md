# ddm_iv1 — MECHANICAL PLUG-IN INVENTORY SWEEP (2026-07-28)

**Arm:** ddm_iv1 (mechanical inventory sweep), READ-ONLY. Base: `main@00c0c28fd7` (merge oc1).
**Directive:** operator 2026-07-28 — *"There's a lot of other stuff we've already built that you've
probably forgotten about that just plugs in and is super optimal."* Enumerate the full built
inventory, DIFF against the live r2s 5-stage pipeline, surface the forgotten PLUGS-IN with their
MEASURED receipt. Per the `composition_requires_mechanical_inventory_sweep` discipline.

**NO-FAKE / calibration:** every "super optimal" label carries a MEASURED receipt (DAG line or module
docstring) OR is explicitly labeled DERIVED / UNMEASURED. No score claims — the pointer (0.19108,
contest-CPU) moves ONLY through a byte-closed `upstream/evaluate.py` n600 row. This memo names PLUG-IN
targets + integration cost; it does not itself lower S. Every receipt tagged `[macOS-CPU advisory]`
unless it is a bit-exact cached-artifact fact.

---

## 0. THE LIVE PIPELINE (the DIFF baseline — 5 stages, arm ddm_r2s)

1. **PREDICT** — stratified plane+parallax warp (`stratified_depth_warp.py`; ground-H(ξ) + γ map).
2. **SUPPORT SELECTION** — argmax-flip support (0.864% sites) + SegNet-RF dilation + auth-weighting.
3. **RESIDUAL VALUES** — solve/project/quantize on the support (range(A) projection, uint8 lattice).
4. **CODING** — support geometry (contour/context-arith) + values (Brotli-Q11/LZMA1/arith races) +
   frame_0 crush (seg-free carrier, pose-certified).
5. **BYTE-CLOSE / EVAL** — `tools/r6cal_byteclose_and_eval.py` → real `upstream/evaluate.py` n600.

**ALREADY ROUTED to r2s (NOT re-listed as discoveries — the baseline):** range_a_projection /
instant_projected_adjoint (#580/#519), predict_project_receiver (#597), margin_saliency_map (#141),
#391 resize-adjoint/flip-ledger/waterfill, #149 pre-R placement, g4 stationarity maps (#623),
arith_selfcomp_rate_coders (#557), #307 contour coders, ddm_pa2_zero_byte_decode_family (#401),
p1 frame_0 pose-quotient carrier (#715), ms4 pose-quadratic custody bundle, lattice solve (#547/#549).

---

## 1. ENUMERATED COUNTS (the "what we've built" totals)

| Source | Enumerated total | Method |
|---|---|---|
| DSL Lever factories | **443** (363 mapped + 80 unmapped) | `tac.witness_dsl.lever_registry.completeness()` |
| Canonical equations | **412** | `tools/list_canonical_equations.py --json` |
| DAG FEED blocks (dated) | **61** blocks (1,246 FEED-mentions, 24,976 lines) | grep `sub015_DAG_*.md` |
| Modules (candidate assets) | **~660** | `boundary_math ~85 · optimization ~350 · witness_control ~90 · witness_dsl ~130 · v2_compose ~7` |
| tools/ scripts | 1,959 | `ls tools/` |

**Sweep method honesty:** the ~660 modules were NOT hand-classified one-by-one (that is not tractable
or useful in one pass). The DIFF was RECEIPT-DRIVEN: cross-reference the module inventory against the
DAG's measured receipts + the adversarial-pass asset classes (pose / coding / cross-pair / receiver),
then confirm mechanism + integration status per candidate. The classification counts below are of the
**surfaced pipeline-relevant subset**, not of all 660 modules.

### Classification of the surfaced subset

| Class | Count | Notes |
|---|---|---|
| PLUGS-IN (receipt-backed) | **22** | TOP-10 below + 12 honorable mentions (§4) |
| SUPERSEDED | 5 | contour/LZMA-labels · raw-7.2KB-dxi · fp16-pose · partition_contour_entropy(as-coder) · power_diagram(as-spatial-codec, self-disclaimed) |
| IRRELEVANT to codec pipeline | large | `l5_staircase_v2` (382 KB, HNeRV/PR95 BANNED lineage) · `pr95_muon_local_training` · `mamba2_predictor` · `aurora_mlx` · all modal/dispatch/cost-band/council/cuda_cpu_axis apparatus = MEANS, not codec plug-ins |

Per-source PLUGS-IN density: modules 22/~660 surfaced-relevant · equations ~100/412 touch the 5 stages
(realiz 12 · pose 27 · margin 10 · flip 6 · coder 9 · chroma 6 · brotli 6 · waterfill 5 · quotient 3 · arith 3)
· levers mostly N/A-DSL for coders/solves (DAG-20602: coder/solve surfaces are not trainer levers).

---

## 2. THE DOMINANT OPEN RESIDUAL (what the forgotten assets must attack)

The live post-solve receiver's n600 residual (DAG 21238 / 21264, MEASURED):
- **Movable conditional d_seg 0.988–0.9895** — "the predictor essentially does not render Movable
  islands at all"; "chart/event corrections through the post-solve receiver CANNOT birth the car
  islands." **This is the #1 residual and it is STRUCTURAL** (a support-birth gap, not a value gap).
- **Lane 0.437** (class-1 IoU 0.263, the unstable orbit; Lane↔Road = ~57% of all flips, #209).
- boundary codim-1 0.427 vs interior 0.0235; Road 0.070 / Undrivable 0.0051 / MyCar 0.0013 near-clean.
- 29× gap to the 0.00116 bar "lives almost entirely in structural Movable-island birth + Lane debt."

The forgotten assets that map onto exactly these two holes (Movable birth, Lane debt) are the
highest-leverage plug-ins — they are not incremental, they attack the residual that dominates S.

---

## 3. THE TOP-10 PLUGS-IN (ranked by leverage = receipt × addresses-binding-residual × 1/integration-cost)

| # | Stage | Asset (path) | MEASURED receipt | Int. cost | Replaces in a naive build |
|---|---|---|---|---|---|
| 1 | SUPPORT/PREDICT | **movable_site_coder.py** `src/tac/boundary_math/` (#394) | Attacks the #1 residual: Movable d_seg **0.988–0.9895** untouched (DAG 21238/21264). Stores class-3 as sparse `(cx,cy,w,h)` boxes + Hungarian temporal track + delta. **1 importer = orphaned.** | **M** | whole-scene Movable bitmap / (currently: NOTHING — islands unrendered) |
| 2 | CODING (pose values) | **xi_pose_coder.py** `src/tac/boundary_math/` (#257) | ξ-only SE(3) payload, temporal-delta + arithmetic-code: **474 B @ d_pose 6.8e-4 / 875 B @ solved-grade 6.3e-5** vs 7.2 KB banked dxi (DAG 6182, FEED-jd). H derived FREE at decode (rule-118). | **S** | 7.2 KB raw dxi bank (live r2s banks this) → verify-and-swap |
| 3 | CODING (support geom) | **context_partition_codec.py** `src/tac/boundary_math/` | SOTA context-arith codec (JBIG/LOCO-I/CABAC), spatial-25 + **temporal-125** ctx; floor = Σ N_ctx·H(p_ctx). Explicit TOP-AIML replacement for LZMA-over-labels (prototype 669–873 B/frame → rate 0.27–0.35 DEAD). **1 importer = orphaned.** | **M** | contour/LZMA support-geometry coder |
| 4 | CODING/SUPPORT (Lane) | **dash_phase_carrier.py** `src/tac/boundary_math/` (#425) | Curve-domain per-dash δ(s) codec at **~2.2 bits/site**; unit = lane DASHES (~20.6/frame world objects), not raster sites. Attacks Lane 0.437 debt. Jitter prior d=0 40.4%/≤1px 72.3% (module docstring). 6 importers. | **M** | raster transport-conditional coding (amortization 0.71 <1 = LOSES) |
| 5 | PREDICT/RESIDUAL (0-byte) | **laguerre_logit_offset.py** `src/tac/boundary_math/` (#218) | **ZERO archive bytes.** 3 head levers fixing class-imbalance under-prediction; Lane↔Road = 57% of flips (#209). Byte-free per-class logit geometry. 2 importers. | **S** | plain-softmax head (systematically drops minority Lane/Movable) |
| 6 | SUPPORT SELECTION | **region_merge.py** `src/tac/boundary_math/` | MDL region-merge SOLVE at the **analytic 1.27 B/flip** water level (weighted greedy graph-cut over the RAG, NOT a sweep). Water level derived exactly from evaluate.py rate term. 3 importers. | **M** | heuristic support thresholding |
| 7 | CODING (residual streams) | **xi_temporal_delta_coder.py** `src/tac/boundary_math/` (#574) | ξ-keyed temporal delta coder on residual/exception streams; published ~40–50% shape-bit cut; our B5 single-object factorization **12.6× over per-frame zlib** (DAG 20596/20050). **0 non-test importers = fully orphaned.** | **S** | per-frame independent residual coding |
| 8 | RESIDUAL/RECEIVER (pose+depth) | **ddm_pc1_pose_stream.py** `src/tac/optimization/` | Typed counted PC1 pose-stream + deterministic multi-depth receiver; writes BOTH frames from one decoded frame-0 via `W_{xi,depth}` (ground-plane depth field + **Movable contact-depth stratum**). 10 importers (mature). Standalone member; does NOT read R1 dxi. | **M** | independent frame_1 store / naive pose bank |
| 9 | CODING (frame0 carrier) | **keyframe_codec.py** `src/tac/boundary_math/` (#202) | Rate-min primitives (degrade→restore-to-native) for warp-real-luma frame0 keyframe bytes (the #202 rate crux, ~0.03–0.07 for 13 keyframes). 4 importers. Companion to #2 IF frame_0 crush uses warp-real-luma. | **M** | raw keyframe store |
| 10 | BYTE-CLOSE/EVAL (hardening) | **confound_gates.py #397–402** + `ddm_runtime_receiver` fail-closed | #402 telemetry-rows-carry-liveness + #398 generalized reject-filter gate (DAG 8346, 77 tests, live-count 0). Receiver/export robustness for the exact-eval stage. **DERIVED/apparatus — robustness, not an S-mover.** | **S** | unguarded inflate (silent-freeze / non-fail-closed receiver) |

**Integration-cost key:** S = a config/adapter wire-in (hours); M = a stage-swap + parity test (day-ish);
L = new receiver surface. None are L.

---

## 4. HONORABLE MENTIONS (surfaced, receipt-backed, below the TOP-10 cut)

- **ego-motion warp-predicted partition residual** (DAG 801, lever ON TOP of #3): warp prev partition by
  the already-paid ξ homography, code only the post-warp residual → cuts the changed-pixel count below
  the per-pixel 0.174-B/change floor. Companion lever to context_partition_codec, not a standalone module.
- **partition_contour_entropy.py** (0 importers) — the information-theoretic FLOOR oracle (Σ N_ctx·H)
  that #3 codes against; SUPERSEDED as a coder, RETAINED as the floor measurement.
- **inverse_depth_compander.py** — depth-companding for the stratified warp (PREDICT); pairs with #8.
- **movable_deshare.py** — Movable de-sharing for #1/#8 (Movable stratum isolation).
- **seven_home_stream_allocator.py** (opt, 1 importer) — deterministic byte-home allocation over EV2's
  7 counted homes; research-only planning apparatus (self-disclaimed), CODING stage-4 allocator.
- **jacobian_fisher_importance_allocator.py** — Fisher-weighted value allocation (RESIDUAL stage); sister
  of the already-routed margin_saliency #141.
- **road_horizon_component.py / road_undriv_bulk_field.py / hood_static_component.py** — per-class bulk
  generators (Road/Undrivable/MyCar near-clean 0.005–0.07; these carry the FREE 80% temporal-constant bulk).
- **step_native_activation.py** — step/hosc head basis (topology-matched to piecewise-constant argmax;
  0-byte train lever; the stable trainable-slope survivor per CLAUDE.md launch caveat).
- **xi_spline_residual_coder.py** — SE(3) spline-knot ξ storage (B3 wrong-levels sweep item, DAG 20660).
- **e5a/E4 adapter + E2 governed inflate** (receiver-hardening class the adversarial pass named) —
  surfaced by name in the DAG apparatus lineage but WITHOUT a standalone measured S-receipt in the FEED
  blocks; folded into the #10 receiver-hardening class rather than claimed as a distinct S-mover.

---

## 5. SUPERSEDED / IRRELEVANT (explicit, so they are not re-proposed)

- **SUPERSEDED:** `contour_codec`/LZMA-over-labels → by #3 context_partition_codec · raw 7.2 KB dxi →
  by #2 xi_pose_coder · fp16 pose → by per-column fixed-point bit-alloc (FEED-db, 2.3 KB, MSE better
  than fp16) · `power_diagram_witness` as a spatial partition codec → self-disclaims ("does not claim a
  channel-space target is a spatial partition codec"; it is the PDW1/PDW2 custody core, 63 importers,
  NOT forgotten).
- **IRRELEVANT (MEANS, not codec plug-ins):** `l5_staircase_v2` (382 KB) + `pr95_muon_local_training`
  = HNeRV/PR95 BANNED lineage (no-old-lineage ban) · `mamba2_predictor` / `aurora_mlx` = alt vehicles ·
  all modal/dispatch/cost-band/council/cuda_cpu_axis/repair_campaign apparatus = infra.

---

## 6. INTEGRATION INSTRUCTIONS FOR THE r2s ARM (one line each)

1. **movable_site_coder** — add a SUPPORT-side Movable birth stage: `extract_movable_sites(L*)` →
   `track_sites` (Hungarian) → byte-account the box+delta stream; render the boxes as class-3 sites in
   PREDICT so the receiver births the islands it currently cannot. Target the 0.988 residual first.
2. **xi_pose_coder** — swap the 7.2 KB dxi bank for `coder="delta_ar"` (~875 B solved-grade); verify
   bit-parity of decoded ξ → identical H → identical d_pose before/after; ~−0.004 S if the swap holds.
3. **context_partition_codec** — replace the stage-4 contour/LZMA support-geometry coder with the
   temporal-125-ctx RangeEncoder; add the DAG-801 ξ-warp-prediction on top (code post-warp residual only).
4. **dash_phase_carrier** — route the Lane class-1 support through the δ(s) per-dash carrier
   (~2.2 bits/site) instead of raster sites; pairs with #1 as the two class-specific SUPPORT carriers.
5. **laguerre_logit_offset** — turn on the 3 head levers at train/render time (0 bytes) to reduce the
   Lane↔Road flip mass (57%) before any support is even selected; cheapest possible d_seg win.
6. **region_merge** — gate SUPPORT SELECTION with the 1.27 B/flip MDL SOLVE (keep a region iff its
   contour bytes < flips-avoided × 1.27); replaces the current heuristic dilation threshold.
7. **xi_temporal_delta_coder** — wrap the residual/exception value streams (stage-4) in the ξ-keyed
   temporal delta coder (0 importers today → pure upside; 12.6× over per-frame zlib on B5).
8. **ddm_pc1_pose_stream** — evaluate as the whole pose+depth RECEIVER (writes both frames from frame-0
   via `W_{xi,depth}` with a Movable contact-depth stratum) — composes #1+#2+#8 into one member.
9. **keyframe_codec** — if the frame_0 crush carrier is warp-real-luma, minimize its keyframe bytes with
   the degrade→restore-to-native primitives (the #202 rate crux).
10. **confound_gates #397–402 / ddm_runtime_receiver fail-closed** — wire into the BYTE-CLOSE stage so a
    frozen/inert receiver can never emit a silent-green exact row (robustness, not S).

---

## 7. VERDICT

The two forgotten SUPPORT-side carriers (**movable_site_coder** #1, **dash_phase_carrier** #4) map
directly onto the two structural residuals that dominate the 29× gap to the 0.00116 bar (Movable-island
birth + Lane debt) — the post-solve receiver CANNOT reach these, so a bolt-on value coder never closes
them; a support-birth carrier is the mechanism. The two forgotten pose/partition coders
(**xi_pose_coder** #2, **context_partition_codec** #3) are near-free rate wins on the CODING stage (pose
7.2 KB → 875 B; partition LZMA → temporal-125-ctx arith). **laguerre_logit_offset** #5 is a 0-byte head
lever that shrinks the flip mass before support is even chosen. All are BUILT, tested, and either
orphaned (1–3 importers) or banking a naive fallback in the live path. Pointer UNMOVED 0.19108; every
receipt is advisory/derived until a byte-closed n600 row lands.
