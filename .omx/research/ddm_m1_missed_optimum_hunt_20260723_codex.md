---
schema: ddm_m1_missed_optimum_hunt.v1
date_utc: 2026-07-23
lane_id: lane_ddm_m1_missed_optimum_hunt_20260723
research_only: true
execution_allowed: false
score_claim: false
candidate_archive: false
evidence_axis: "[read-only receipt and source audit]"
verdict: TWO_GENUINELY_OPEN_REPRESENTATION_RACES_FOUND
verdict_scope: "DESIGN x current DDM vehicle and cited finite formulations; no family, launch, contest-axis, score, or promotion verdict"
pointer: "0.1910828242 [contest-CPU Linux x86_64]"
pointer_moved: false
main_landing_review_required: true
---

# DDM M1 missed-optimum hunt

## Outcome first

**Yes—but not in the literal forms suggested by the headlines.** Two representation races are
genuinely open on the current vehicle:

1. **A kinetic anisotropic Laguerre / regular-triangulation event code** that stores one evolving,
   shared cell complex and renders it through a scorer-free RGB pullback. This is the
   highest-leverage open representation because it could replace the current 100,099-byte
   predictor home (**50.0495% of the 200 KB box**) while describing shared boundaries once. It is
   not the already-refuted “few Euclidean sites per frame,” and no matched
   `d_seg <= 0.00116` rate exists.
2. **A conditional frame-0 Pose preimage**: hold frame 1 exactly fixed for Seg, derive frame 0
   from it and the counted `xi`, and store only the least code needed to hit the six Pose outputs.
   The scorer factorization makes frame 0 an exact Seg null direction. The current DDM vehicle has
   a 3,721-byte `xi/Pose6` home but `d_pose=163`; the separate 7,195-byte R1 result reaches
   `d_pose=0.001610` only as a co-adapted complete artifact and is not transferable.

The strongest additional signal is v19b: exact joint replay of ten correction moves is
**synergistic**, not merely non-additive. It adds `0.0804967212` score units of amplification,
with eight of ten moves amplified (up to `9.058356x`) and 73,945 realized flips in
Road+Undrivable+MyCar. Therefore any new representation must optimize and replay compound
description elements jointly. Independent per-role rate curves are useful controls, not a valid
composition theorem.

This memo is a design-only ranking. No probe was fired, no vehicle was built, and no score or
pointer was changed.

## Fresh source and live-state facts

The current `upstream/modules.py`, SHA-256
`065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa`, establishes:

- `SegNet.preprocess_input` selects `x[:, -1]`, resizes only frame 1 to `384x512`, and scores the
  five-class `argmax`. Thus `D_seg(I_0,I_1)=D_seg(I_1)` exactly.
- `PoseNet.preprocess_input` resizes **both** frames, maps each RGB frame to official YUV6,
  concatenates the twelve planes, and scores six outputs. Frame 0 is not Pose-null, and chroma is
  not assumed null.
- The Seg head has the exact rank-4 affine/Laguerre quotient, but its affine coefficients alone
  do not determine the spatial quotient field or a scorer-free RGB preimage.

The latest consumed v19b receipt, SHA-256
`4bb5d6b4b793b667c7cbe15e37cbf9a27f6c0e75451374839fb5df8ca1c1b8e8`, measures on n600
`[macOS-CPU frozen-scorer advisory]`:

- 137,825 B, `d_seg=0.026594424778`, `d_pose=163.061176604795`;
- 3,137,206 Seg errors, hence **3,000,367 above** the integer 136,839-error target;
- 103,322 net flips for +3,884 B versus v15, `delta_S=-0.085019605373`;
- 29,377 Lane+Movable flips and 73,945 Road+Undrivable+MyCar flips.

The c1 number **2,377,273** is still only the optimistic residual floor after maximal
preregistered REALIZE credit; it is not the latest realized error count. Likewise, c1's
7,232-byte row is a final coder/container contingency—not an already composed Pose stream. The
current nested `xi/Pose6` home is 3,721 B and its measured Pose distortion is 163.061..., so
“Pose is already banked” is false for this vehicle.

## Ranked missed-optimum table

Rate entries are complete measured or derived rates only. `N/A` means no current receipt reaches
the required tolerance; it is not filled by extrapolation. Leverage percentages are ceilings,
not forecasts. The residual denominator is c1's requested optimistic 2,377,273 errors unless a row
explicitly says otherwise.

| Rank | Candidate and `modules.py` grounding | Coverage verdict | Rate at required tolerance versus live line | Maximum leverage | Cheapest decisive $0 probe |
|---:|---|---|---|---:|---|
| **1** | **Kinetic anisotropic Laguerre / regular-triangulation event code.** Seg sees one frame's argmax partition; code initial weighted sites, low-degree site/weight trajectories, shared dual adjacency, and sparse triangulation flips; render a scorer-free RGB preimage. | **GENUINELY OPEN.** Literal Euclidean per-frame sites are dead; v8 hybrid and v13 worldsheet are partial controls. No current arm races a temporally linked, shared-edge-once cell complex through RGB/R at matched tolerance. | **N/A at `d_seg<=0.00116`.** Closest: v8 hybrid 229 B/frame = 137,400 B, residual .0079 (6.81x too high, not through R); global K=1024 = 2,227,200 B, residual .0073 (6.29x too high). Live v19b is 137,825 B at .026594, not target-green. | **<=100,099 B = 50.0495% of box** if it replaces the predictor home; tautologically <=100% of residual, not forecast. | Execute the bounded label-space then RGB-pullback ladder in `ddm_m1_kinetic_laguerre_at_tolerance_probe_20260723.json`. Kill the registered form if all site/degree/metric/event cells fail 136,839 errors in <=100,099 serialized bytes. |
| **2** | **Conditional frame-0 Pose preimage.** Seg's exact `x[:,-1]` selection permits bit-identical frame 1; Pose still constrains both YUV6 frames. Optimize only counted frame-0 coefficients conditional on frame 1 and `xi`. | **GENUINELY OPEN / infrastructure PARTIAL.** The current line stores `xi/Pose6`, but has not substituted a minimal frame-0 preimage. R1 proves a co-adapted complete artifact, not a graft. | Current: 3,721 B home at `d_pose=163.061...` (**fails**). Nontransferable comparator: R1 7,195 B at `.001610`. **Current-vehicle rate at `.001610`: N/A.** | Known comparator <=7,195 B = **3.5975% of box**, plus an unmeasured frame-0-specific saving. It can remove the entire current Pose gate; no honest “half the predictor” byte claim is available because current homes are not frame-attributed. | Hold frame 1 byte-identical; race `xi` warp, luma-only, and joint-YUV6 low-order preimages under exact Pose+byte replay. Confirm only at n600 `d_pose<=.00161` and <=7,195 complete bytes. |
| **3** | **SE-coupled compound motif dictionary.** Both scorers are nonlinear after resize; v19b directly measures positive higher-order coupling. Store jointly useful worldsheet/template/Q8 motifs instead of independently selected atoms. | **PARTIAL, not missed by principle.** v19b exact greedy composition sees ten given motifs; #366/J5 and #604 global-description solve are the live consumers. What remains open is interaction-aware *motif discovery*, not joint replay itself. | v19b measured +3,884 B, 103,322 net flips, `delta_S=-.0850196`; amplification alone `.0804967`. No target-green rate. | Measured flips equal **4.3462%** of 2,377,273; residual-bucket flips alone **3.1105%**. Further reach N/E. | On n64, exact-replay all singleton/pair/triple combinations of the top costate-ranked atoms, fit a Möbius interaction table, then compare a compound-code dictionary with v19b greedy at identical bytes. |
| **4** | **One global min-description n600 solve over the complete counted description.** `modules.py` gives the exact objective; optimize archive description through free decode and both frozen scorers. | **COVERED / execution bridge PARTIAL.** This is #604 U1's direct-description minimizer plus #366 trunk descent; it is the correct end-state, not a newly missed representation. c1 remains piecewise in current execution. | **N/A at target.** No full n600 end-to-end row. The 200 KB box and exact target define the admission gate. | `N/E (<=100%)`; the full residual ceiling is tautological. | Do not spawn a duplicate arm. Complete the existing #604 description-variable bridge with exact code length inside selection, then replay one common n600 master. |
| **5** | **Target-boundary curvelet/shearlet residual.** Seg error is concentrated on curved codimension-1 separatrices; directional systems are the right approximation class. | **PARTIAL; literal full-RGB test does not win.** Genuine #502 n600 equal-value/support receiver ranks Fourier `.409722` < shearlet `.428860` < curvelet `.504824`; it is not equal bytes and not a target-boundary inverse. v13's receiver schema has a shearlet field but active count zero. | **N/A at target and equal bytes.** No custodied target-boundary pullback or complete ZIP race. Do not turn the theorem into a measured rate. | N/E; boundary mass is large, but no receipt maps it to current target errors at a byte price. | Reuse c1/A1 column generation: same exact boundary targets, equal complete ZIP bytes, curvelet and shearlet only, joint RGB/R replay. Kill only the registered finite bank if no negative exact reduced-cost column appears. |
| **6** | **Physical BEV-static ground collapse.** A correct full homography could make road/lane structure one-time while `xi` supplies motion. | **PARTIAL, not highest.** BEV v2 fixes the near-field hood control but far-field Road/Lane 39--47 px remains confounded by yaw/trajectory drift. Exact G1-PoseNet chart is scoped dead; true absolute ego GT family remains open. | N/A. No target-green, receiver-closed byte row. | **<=9.6841%** Lane-only ceiling from the audited receiver debt; not “majority of all current debt.” | v3 with independent absolute trajectory and far-field yaw control; require the far-field control to pass before Road/Lane rate is interpretable. |
| **7** | **Literal few global Euclidean power generators / affine-head packet.** | **MEASURED-DEAD / N/A as a self-contained receiver.** Few global sites flatten near `.0073`; the 133--138 B affine packet is spatially non-identifying without the quotient field, whose preserved memmap is about 1.887 GB. | No target-green rate. The historical SPEC `.02-.05` rate is not validated at `.00116`. | 0 for packet-only spatial reconstruction. | None. Do not rerun. Rank-1 deliberately replaces this dead form with a richer kinetic, spatially explicit formulation. |
| **8** | **Camera resize/gauge/null coordinates as stored-rate savings.** Both scorers see a resize and have large exact kernels. | **MEASURED-DEAD as a rate lever on the current procedural receiver.** The current generator stores no camera pixels, so zeroing unsampled/gauge coordinates saves no payload. Dense head-gauge int8+Brotli was rate-neutral within 11 B. | N/A; no bytes exist to remove. | **0% current rate leverage.** | None unless a future representation actually stores camera-space pixels; then re-open against its own byte homes. |
| **9** | **Pose-from-embedding MLP replacement.** Pose has only six outputs, suggesting a small map. | **INSTANCE-DEAD.** The measured 4,782-parameter fp16 MLP costs 9,564 B, above the 6,791 B stream it replaces, and worsens score by `.00185` even if other weights are free. | Measured non-dominating instance. | 0 under tested instance. | None; reopen only with a materially different compressed/shared weight realization and exact bytes. |
| **10** | **Raw q8 Lane phase deviations.** Seg sees thin, resize-phase-sensitive Lane boundaries. | **INSTANCE-DEAD; family PARTIAL.** v13 raw pre-addendum phase symbols worsened the joint objective. AR(1)-whitened anisotropic BEV innovation remains unmeasured. | Raw form measured non-dominating; successor N/A. | <=9.6841% Lane ceiling. | Only the already named raw-vs-whitened innovation race under the repaired receiver; do not relabel raw q8 as new. |

## Why rank 1 is a representation change rather than a power-diagram rerun

The exact Seg factorization says that for a rank-4 quotient field `z(x,t)`, the cell label is

`c*(x,t) = argmax_c <a_c,z(x,t)> + b_c`.

The dead shortcut stores only `{a_c,b_c}` or independent Euclidean sites. It omits the expensive
object: the spatial-temporal field and its topology. The proposed object instead stores a
**kinetic regular triangulation**:

`P_j(x,t) = ||x-q_j(t)||^2_{M_j(t)} - w_j(t)`,

with class `c_j`, low-degree trajectories for `q_j,w_j,M_j`, and sparse combinatorial flip events.
Its dual graph makes adjacency explicit and charges every shared boundary once. BEV/projective
charts are optional, gated coordinates—not assumed facts. A generic decoder rasterizes the cells
and a counted palette/template state pulls them back to RGB; neither scorer nor its weights is
shipped.

This form attacks two measured inefficiencies simultaneously:

- v8's independent per-frame generator cost and `.0073-.0079` boundary jitter floor; and
- v13/c1's separately coded per-role structures and later correction atoms.

It remains a hypothesis until both the target-tolerance rate and scorer-free RGB realization are
measured. Calling the head packet itself the generator would repeat the PDW2 non-identifiability
error.

## Top-2 full preregistrations

Both configurations are config-complete execution contracts but their named runners are absent in
this delegated snapshot. That is an explicit fail-closed blocker, not permission to fire or a
claim that a measurement exists. A later builder must land and MAIN-review the runner without
changing the fixed endpoints; execution requires separate authority.

### Probe 1 — kinetic anisotropic Laguerre at tolerance

**Config:** `.omx/research/configs/ddm_m1_kinetic_laguerre_at_tolerance_probe_20260723.json`

**Question.** Can one temporally linked cell complex reach 136,839 n600 label errors in no more
than the current 100,099-byte predictor home, then survive a scorer-free RGB/R pullback with
`d_pose<=.00161` inside the 200 KB box?

**Controls.** Import, do not rerun, the sealed v8 global K=1024 and per-class hybrid rows; compare
the current v13 predictor home and exact v19b archive.

**Fixed ladder.** Site counts `{64,128,256,512}` x trajectory degrees `{1,2,3}` x isotropic,
shared-chart anisotropic, and projective-depth-stratified metrics x independent/kinetic temporal
coding. Race real complete serializers. Preserve every cell atomically.

**Primary endpoint.** Minimum complete serialized bytes subject to at most 136,839 label errors.
Stage B runs only for Stage-A winners and measures exact RGB -> uint8 -> R -> Seg/Pose plus complete
archive bytes.

**Confirm.** Stage A `errors<=136839` and `B<=100099`; Stage B `archive<=200000`,
`d_seg<=.001159998576`, `d_pose<=.00161`, deterministic double-decode.

**Falsifier.** No registered Stage-A cell meets both error and byte limits. If Stage A passes but
all winners fail RGB pullback, close only
`FORMULATION:KINETIC_ANISOTROPIC_LAGUERRE_REGISTERED_LADDER`, not every generator/grammar family.

### Probe 2 — conditional frame-0 Pose preimage

**Config:** `.omx/research/configs/ddm_m1_conditional_frame0_pose_preimage_probe_20260723.json`

**Question.** With exact v19b frame 1 fixed, can existing `xi` plus a low-order counted frame-0
preimage reach the Pose tube more cheaply than the 7,195-byte nontransferable R1 comparator?

**Controls.** Exact v19b hold; `xi` warp with zero residual; luma-only basis; joint-YUV6 basis.
The YUV6 arm prevents the unproved “chroma is null” assumption.

**Fixed ladder.** Luma coefficients `{6,12,24,48}`; joint-YUV6 `{12,24,48,96}`; quantizers
`{4,6,8,10}`. Relinearize after every accepted exact finite step; stop at the exact byte-price
break-even. Preserve n64 and n600 checkpoints.

**Confirm.** `d_pose<=.00161`, complete pose payload `<=7195 B`, frame-1 camera bytes and exact-R
Seg cells bit-identical, and no scorer/target table in the archive.

**Falsifier.** No preregistered basis/quantizer cell reaches the Pose tube inside 7,195 B, or any
winner mutates frame 1. Close only
`FORMULATION:XI_CONDITIONED_LOW_ORDER_FRAME0_POSE_PREIMAGE`.

## Decision and routing

1. **Race the kinetic cell-complex representation first** because it has the largest bounded
   byte leverage and directly tests whether the current line codes the wrong object.
2. **Race conditional frame 0 second** because it cleanly isolates the still-open Pose gate with
   an exact Seg invariant and a real 7,195-byte comparator.
3. Feed the v19b interaction result into both: selection authority is the exact composed master,
   not independent component deltas.
4. Do not spawn duplicates for global min-description solve (#604/#366), BEV (#609 successor), or
   c1/A1 curvelet column generation. Those are already routed.
5. Do not resurrect literal global Euclidean sites, packet-only PDW2, raw q8 phase, camera-null
   deletion, or pose-from-embedding MLP as new ideas.

## Triality

- **DSL/data:** the two typed, execution-disabled JSON probe configurations above.
- **DAG:** `.omx/research/ddm_m1_missed_optimum_hunt_DAG_FEED_20260723.md`.
- **Equations:** `.omx/research/ddm_m1_missed_optimum_hunt_canonical_equations_20260723.md`.

## MAIN landing review required

MAIN must independently:

1. verify the authority prompt SHA and this branch diff;
2. ensure v19b receipt `4bb5d6b4...` and archive `74ede419...` are present before accepting their
   rows, because this branch consumed them read-only from later main state;
3. re-derive the 136,839 target, 3,000,367 latest residual, c1 2,377,273 optimistic residual, and
   all byte/leverage percentages;
4. confirm the v8 rows are geometric cache comparisons, not through-R scores;
5. confirm R1 is a complete-artifact prior, not a composable DDM component;
6. preserve `execution_allowed=false`, no score/promotion claim, and pointer immobility;
7. review the two falsifier scopes before any runner is built or fired.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; `PROGRAM.md`;
`/Users/adpena/Projects/pact/upstream/modules.py`; `reports/latest.md`; FEED-603 window in
`.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`;
`.omx/research/SPEC_v8_perclass_decomposition_20260708.md`;
`.omx/research/SPEC_v10_capstone_cold_start_seeded_20260717.md`;
`.omx/research/ddm_c1_composed_candidate_spec_603_613_20260723.md` and its JSON/equations/DAG;
all 18 SHA-bound c1 receipts; v13, v14, v15, v16, v19 and later main v19b receipts;
`.omx/research/v8_laguerre_generator_feasibility_and_perclass_hybrid_20260710.md`;
`.omx/research/v10_power_diagram_byteclose_findings_20260718.md`;
`.omx/research/pdw2_spatial_receiver_576_implementation_spec_20260719.md`;
`.omx/research/frozen_scorer_exact_factorization_20260715.md`;
`.omx/research/null_subspace_rate_measure_20260717.md`;
`.omx/research/ddm_a1_naive_verdict_audit_20260723_codex.md`;
genuine #502 curvelet receipts and operator broadcast custody; BEV v1/v2 FEED rows;
Claude memory L-v8/L1/L10 and `meet_it_where_it_is_carry_thing_itself_smallest_basis_n600`;
latest Codex findings/session memos; lane/task/subagent state; per-arm inbox through
`2026-07-23T06:28:18Z`; fleet broadcast through `2026-07-21T13:15:53Z`.
