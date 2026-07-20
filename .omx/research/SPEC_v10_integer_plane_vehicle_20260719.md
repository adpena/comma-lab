# SPEC v10 integer-plane vehicle — measured-parts successor section
<!-- # DUPLICATE_SOT_OK: additive successor SECTION to spec_v10_capstone_reconciled (registered relation in canonical_doc_registry); per §1 it supersedes ONLY the stale status prose in reconciled §§R2/R3/R6/R7 — both docs are intentionally canonical for vehicle v10; registered 2026-07-19 at MAIN review of the #567/#566 drift finding -->

**Date:** 2026-07-19
**Vehicle ID:** `v10.integer-plane.two-independent-planes.v0`
**Maturity:** L0 composition SPEC; `research_only=true`; `launch_ready=false`
**Successor relation:** additive successor section for
`.omx/research/SPEC_v10_capstone_RECONCILED_20260719.md`; it does not rename or rewrite the
reconciled capstone.
**Pointer delta:** exactly zero. The preserved pointer is
`0.1910828242 [contest-CPU Linux x86_64]`.
**Authority:** composition and gap-listing only. No training, measurement, archive evaluation,
paid dispatch, score claim, promotion, submission, or pointer mutation occurred here.

## Verdict

**COMPOSED, NOT BUILD-READY:** the measured parts select a two-independent-integer-plane vehicle,
but the fast receiver has only a one-plane-plus-copy timing receipt, the learned integer-plane
emitter and pose-proximity objective are not built, and the PDW2 packet remains target-only until
a spatial consumer exists. The ordered charters in §8 are owed before any training launch.

This document follows `docs/operating_manual_craft_handoff.md`, `PROGRAM.md`,
`docs/vehicle_operating_system.md`, the binding operating contract in
`.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md` §8, and the v8 successor
constraints in `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`.

## 1. Verdict scope and succession boundary

The program object remains the exact contest action

\[
S = 100d_{\rm seg}+\sqrt{10d_{\rm pose}}+
    25\frac{B_{\rm archive}}{37{,}545{,}489}.
\]

The proposed vehicle changes the representation, not the evaluator:

1. a learned, compact description emits **two independently described** scorer planes;
2. each plane is projected to exact `uint8` at scorer resolution;
3. a deterministic factor-2 receiver separately solves camera-frame preimages for frame 0 and
   frame 1; and
4. the exact frozen evaluator, exact archive bytes, and exact contest axes remain the only verdict
   authorities.

The scorer-plane tensor is width×height×channels `512×384×3`, represented in arrays as
`[384,512,3]`; a pair therefore has shape `[2,384,512,3]`, and an n-pair batch has shape
`[n,2,384,512,3]`. The receiver emits two `uint8` camera frames of shape `[874,1164,3]` per pair.
These dimensions come from the frozen scorer and factor-2 receiver contracts, not a new choice
([E15], [E3]).

This section supersedes only the now-stale status prose in reconciled §§R2/R3/R6/R7 that called
the positive-band KKT and pose intermediate regime unmeasured. The 2026-07-19 secant arm supplied
the missing measured rows ([E9], [E10]). All other reconciled statements stand unless MAIN makes
an explicit append-only amendment.

## 2. Evidence register — the only admitted parts

Labels mean exactly: **MEASURED** is an observed receipt; **DERIVED** is arithmetic or a source
consequence; **STRUCTURAL** is present in code but lacks the named runtime receipt; **DESIGN** is
specified here and remains owed. Advisory measurements are not contest scores.

| ID | Admitted fact | Label and exact value | Source row and landed merge | Authority limit |
|---|---|---|---|---|
| E1 | Native-plane replay shape | **MEASURED:** 24 real pairs / 48 planes; `28,311,552 / 28,311,552` rational samples exact; zero rational failures. **MEASURED:** `0/24` pairs frozen-f32 bit-identical and 7 Seg argmax disagreements. | `.omx/research/yhat_native_generator_20260719_codex.md:28-56`; merge `9b25ba3ce0` | macOS-CPU advisory; donor-derived feasible planes; rational equality is not oracle equality. |
| E2 | Arbitrary-rational receiver runtime | **MEASURED n24:** `597.7790400451 s` solve total, `12.4537300009 s/plane`, `25.7238992518 s/two-plane pair`. **DERIVED n600:** `14,944.4760011276 s = 249.0746000188 min`. | `.omx/research/yhat_native_generator_20260719_codex.md:58-70`; merge `9b25ba3ce0` | n600 was projected, not run; expansion, packaging, and output I/O excluded. This formulation is dead only for this runtime contract. |
| E3 | Fast factor-2 receipt | **MEASURED n12 double decode:** `4.53 s` and `4.50 s`; `14,155,776 / 14,155,776` numerator values exact; all `36,624,096` frame bytes satisfy frame0==frame1. **DERIVED n600:** `226.5 s = 3.775 min`. | `.omx/research/production_receiver_543_20260719_codex.md:33-68`; source commit `c03c2e1389` | exactly one integer frame-1 plane is solved and copied under `repeat-frame1`; not a two-independent-plane timing result. |
| E4 | Distinct-plane receiver surface | **STRUCTURAL:** `description-frame0.v1` accepts exact `uint8` frame0 matching frame1 geometry and independently realizes both planes. **MEASURED Rung E n48:** `56,623,104` exact numerator values, but 1,165 Seg flips, `d_seg=0.00012344784206814235`, `d_pose=0.00005041551356414436`, archive `31,873,460 B`. | `src/tac/witness_dsl/v10_production_receiver.py:404-470,1127-1137`; `.omx/research/constructive_solver_541_20260719_codex.md`, Rung E; merge `1461c0fedb` | no distinct-plane wall-time receipt; Darwin-arm64 research-only; this preimage policy is rate-dead and exposes native-f32 debt. |
| E5 | Head-quotient certificate | **MEASURED:** PDW2 margin packet `138 B` raw / `133 B` Brotli-q11, layout `12+10+80+36`, 20 float32 coefficients; PDP2 partition-only `134 B` / `122 B`, 19 coefficients. Frame-195 tie passes only with deterministic zero-sum float32 reconstruction. | `.omx/research/pdw2_gauge_packet_probe_20260719_codex.md`, packet table and frame-195 gate; `.omx/research/pdw2_gauge_packet_probe_20260719_receipt.json`; merge `0e0c1c4f1a` | literal verdict `TARGET_ONLY_VS_REALIZATION_NON_EQUIVALENT`; no spatial/RGB receiver or through-R savings. |
| E6 | Baseline learned-tensor coder | **MEASURED on the mod32cap donor `levelset_n600_witness_mod32cap_20260706T115554Z` (source SHA `6dd28a6e295d007ef0e53ae3e0e792a517a5708394a17d2185870e44920dedca`):** per-tensor int8+Brotli-q11 is `63,394 B` for 72,695 base-weight symbols and `20,518 B` for 38,400 pair-code symbols; tested hand-built entropy contexts were larger. | `.omx/research/arith_selfcomp_rate_coders_20260719_codex.md`, complete-coder table; merge `209c9cc3a6` | donor-specific byte anchors; they do not predict the new vehicle’s bytes. The negative scopes only the measured coder/configurations. |
| E7 | Block-FP alternative | **MEASURED rate-only candidate:** block size 32, threshold 0.25, LZMA-framed `13,957 B` over 111,095 parameters, `1.0050497322111707 bits/parameter`. | same coder memo, block-FP section; merge `209c9cc3a6` | no matched n≥24 through-R Seg/Pose receipt; inadmissible until charter C7. |
| E8 | Post-training and joint self-compression | **MEASURED control on the distinct ep725 donor `levelset_n600_witness_20260717T113932Z` (archive SHA `b0a431e9259cd3c54ae53b677076823f36e096b27eb0d9ba74ed7c54c9113cef`):** canonical archive `83,838 B`; DeepCABAC replacement `85,274 B`, `+1,436 B`; base stream `61,598→63,007 B`, `+1,409 B`. **DERIVED knee:** deleting all 61,598 base bytes saves at most `0.0410156` score, so pose-neutral `Delta d_seg < 0.000410156` is required before overhead. | `.omx/research/neural_selfcomp_sota_20260719_codex.md`, receiver-closed table, size-in-loss formulation, and knee; merge `c3ba14cd07` | measured DeepCABAC donor/configuration only. Joint size-in-loss remains an unmeasured new-vehicle arm and is default OFF. |
| E9 | Seg/rate economics | **DERIVED law:** `150,181,956 B` per unit `d_seg` = `150.181956 B` per `1e-6 d_seg`, pose fixed. **MEASURED inputs / DERIVED advisory closest-secant KKT candidate:** the closest adjacent endpoints are `margin_m0p3` and `precision_drop1`; their derived marginal-score gap is `4.142214713626108e-12` per global byte. Each measured endpoint covers 24/24 real pairs and has Pose inactive. | `src/tac/canonical_equations/seg_rate_breakeven_and_head_gauge_laws_20260719.py:4-12,93-139`; `.omx/research/seg_secant_rd_curve_20260719_codex.md`, “Verdict” and “Adjacent Seg secants and break-even sign”; merge `c6b798f146` | the price is an indifference law, not an achieved curve. Closest secants do not establish a continuous optimum. The candidate uses n600-equivalent conditional payload rows, not exact receiver-closed archive bytes. |
| E10 | Pose/proximity law | **MEASURED four-anchor registry:** near-source solved planes are Pose-inactive; margin rows are 96/96 inactive; precision-drop1 is 24/24 inactive; precision-drop2 has 2/24 violations and precision-drop3 has 4/24, six pair-point violations total; the two spatial rows violate 48/48. One measured positive-band value is `d_pose=2.521975392375284e-5`. The **DERIVED** Pose-vs-Seg marginal crossover is `d_pose=2.5e-4`; per-pair activity against the literal `<2.5e-4` gate is measured. Far-plane instance: plane RMSE about 25 and `d_pose=63.031066895`. | `.omx/research/seg_secant_rd_curve_20260719_codex.md`, “Pose activity and violations”; `.omx/state/canonical_equations_registry.jsonl:745`; merge `c6b798f146`, registry commit `7b9a3a7d3c2` | proximity-conditional, not “Pose is always free.” The Python builder still emits three anchors and stale “intermediate unmeasured” text; registry row 745 also retains a stale `domain_of_validity.unmeasured` field; see §10. |
| E11 | Non-additive pools | **MEASURED/DERIVED advisory decomposition:** rounded pool totals `T≈0.3146`, `A≈0.208`, `B≈0.002`, `C≈0.1046`; levers inside one pool compete and cannot be added as independent score gains. | `.omx/research/SPEC_v10_capstone_cold_start_seeded_20260717.md:829-901`, non-additive-pool law | summary-rounded legacy pool accounting, not a byte allocation or promise for this vehicle. |
| E12 | Integer lattice foundation | **MEASURED n600 advisory:** 114 Seg mismatches / `117,964,800`, `d_seg=9.663899739583e-7`, max rational residual `8.527e-14`; all mismatches were ULP-tie class. Clip-round comparator: 42,817 mismatches, 376× worse. | `.omx/research/v10_lattice_rate_verdict_and_composition_20260719.md:108-123`; merge `00b40c58ce` | macOS-CPU advisory, frame1/Seg only; aggregate custody is external and must be brought into tracked/content-addressed custody. |
| E13 | Resume apparatus | **MEASURED:** CPU-locked bit identity `13/13`; control twice deterministic; SIGKILL at epoch 3 then resume made an exact continuation, not a re-anchor, restoring live/EMA/optimizer/RNG/event/stage state. Stage checkpoints were byte-close-loadable, distinctly named, and bounded-retention. GPU aggregate resume deltas `0.0083<0.0102` live/EMA host floor and `0.0185<0.0212` optimizer floor, with discrete state identical. | `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md:19752-19769`; merge `74d511b9ae` | proves the reusable apparatus, not this unbuilt vehicle’s state coverage. Final receipts are external; older tracked status is stale. |
| E14 | Native-f32 arithmetic law | **MEASURED registry has three anchors:** frame195 margin `4.76837158203125e-7`; pair125 has one flip despite exact rational equality (`d_seg=5.086263020833333e-6`); Rung-E preimage policy has 1,165 flips over 48 pairs despite exact numerators. | `.omx/state/canonical_equations_registry.jsonl:742`; `.omx/research/constructive_solver_541_20260719_codex.md:120-147`; registry commit `d864b267a7` | Python builder still emits two anchors; hard-oracle admission or a measured margin rule is mandatory. |
| E15 | Frozen scorer factorization | **DERIVED from pinned source:** Pose uses both frames, shared bilinear resize, then RGB→YUV6; per 2×2 scorer-plane RGB block, luma is lossless four-phase space-to-depth while U/V chroma are lossy box averages. Seg uses frame1 only and the same resize; Pose distortion consumes the first six head outputs; Seg distortion is argmax disagreement. | `.omx/research/frozen_scorer_exact_factorization_20260715.md:16-58`; source commits `58dfcfac51`, `7dc017fe88`; `upstream/modules.py:70-84,107-113,143-158`; `upstream/frame_utils.py:51-79`; `upstream/evaluate.py:63,92` | exact pinned evaluator only. `upstream/` is absent from this worktree; the content-addressed source-derived memo is tracked, and the exact source lines were independently reopened read-only at `/Users/adpena/Projects/pact/upstream/`. |
| E16 | Quantization-in-training corpus | **CODE-VERIFIED:** `Uint8STE` forward is `clip(round(x),0,255)` and backward is identity only in the unsaturated range; tests verify integer forward values and nonzero gradient through R. | `src/tac/quantization.py:190-220`; `experiments/tests/test_witness_realized_through_R.py:59-93` | Torch reference exists; an MLX implementation and deterministic NumPy parity for this vehicle are owed. |
| E17 | Frozen SegNet weight geometry | **MEASURED, macOS-CPU advisory, on frozen weights SHA `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`:** the head is `Conv2d(16→5,k=3,activation=None)`, hence five affine logits over each 144-D `16×3×3` penultimate patch. Centering its five rows gives singular values `[3.1283763256,2.1542713873,2.0247078699,1.7962638357,3.7304e-16]`, rank 4, rank-4 reconstruction max error `5.96e-8`, and `sigma1/sigma4=1.74160`. The ten pairwise hyperplanes lie in that four-dimensional difference span; mutual unoriented normal angles are `25.8°–90°` (median `62°`). Lane-pair normal norms are `4.007,3.953,3.862,3.748`, versus `2.602–2.946` for non-Lane pairs. | `.omx/research/segnet_recursive_fractal_factorization_20260715.md:1-19,50-97`; `.omx/research/power_diagram_witness_20260718.md:149-160`; canonical equation `src/tac/canonical_equations/segnet_head_rank4_flipdist_20260715.py`; merges `6207f8a889`, `18247802a6` | exact only in the local 144-D penultimate-patch space. Pixel/RGB pullback crosses the nonlinear trunk and remains first-order/realization-limited. |
| E18 | Immutable weight-response custody | **MEASURED:** 24 immutable real-pair VJP sidecars. Seg stores aggregate active-field `g_y [384,512,3]`, `g_x=A^Tg_y [874,1164,3]`, `Lip_local=||g_y||`, and `q=g_y/Lip_local`; Pose stores `J_y=d pose[:6]/d(y0,y1) [6,2,384,512,3]` and `J_x=A^TJ_y [6,2,874,1164,3]`. | `.omx/research/vjp_custody_positive_bands_20260719_codex.md:13-20,35-112,240-257`; manifests SHA `3d1218a52e...` and `200e8cfa37...`; final merge `6704c3857c` | proposal/constraint surfaces only; the full hard oracle admits. Sidecars are encode-side custody, never payload, and must be recalled rather than re-derived. |
| E19 | Channel, scale, and resize-response priors | **MEASURED advisory, scope-separated:** destroying stride-2 skip detail on n16 induces 8,072 flips, of which 6,205 (`76.87%`) are Road–Lane; the single `192×256×16` skip is shared, not a private Lane channel. Margin-gradient ERF has median `r50≈85 px` and `r90≈206–424 px` (about `300 px` working guard). The historical n600 operating-point flip-mass summary is approximately `50% Road / 19% Lane / 13% Undrivable`. One donor has `52.42%` raw / `52.88%` mean-removed output energy in `ker(A)`, with about `50–53%` of marginal output-layer effect there. | `.omx/research/segnet_recursive_fractal_factorization_20260715.md:119-190`, merge `6207f8a889`; `.omx/research/lane_channel_deep_refactorization_20260716.md:55-70`, merge `477ec610c5`; `.omx/research/canonical_research_index_dseg_20260629.md:49-62`, merge `73e39c402e`; `.omx/research/null_subspace_rate_measure_20260717.md:93-114,142-161`, commit `1dbdfff7ea` | induced-flip share, historical class share, ERF, and null-energy are diagnostic priors, not byte quotas or achieved gains. They must never be multiplied or added. The approximately 52% value is sample-specific capacity waste, not an automatic byte saving. |
| E20 | Basis-before-capacity, filter-scale, and full-RGB channel role | **OPERATOR-BINDING DESIGN from measured structure:** use the four left-singular head directions and ten pairwise margins as the first Seg coordinate system at fixed capacity; test that basis before increasing width/modulation dimension. **DERIVED:** frozen Seg consumes full RGB. **MEASURED advisory filter census:** `74.3%` of 14,784 depthwise kernels are low-pass; high-pass/oriented mass lives mainly at strides 2–8, while deep oriented kernels are predominantly axis-aligned. **MEASURED n600 advisory necessity ceilings:** removing all chroma costs `0.005384 d_seg`-equivalent; removing annulus chroma costs `0.002972`; keeping chroma only in the annulus costs `0.006293`. | rank-4 and filter sources `.omx/research/segnet_recursive_fractal_factorization_20260715.md:50-97,137-155`, merge `6207f8a889`; `.omx/research/frozen_scorer_exact_factorization_20260715.md:20-28,44-57`, merges `58dfcfac51`, `7dc017fe88`; `.omx/research/rgb_at_boundaries_derivation_20260715.md:60-101,120-143`, merge `222ee235e9`; MAIN directive `2026-07-19T13:05:03Z` | no superseded directional-gain figure is consumed. Deep-filter orientation causing any task deficit is only a hypothesis. Chroma ablation costs are necessity/worth ceilings, not achieved vehicle gains; candidate-local luma/chroma and class-pair effects remain owed. |

No donor byte count, advisory distortion, or projected runtime above is silently promoted to a new
vehicle prediction.

## 3. Frozen-cell optimization map

Every stage is optimized against an actual frozen evaluator surface. “No direct modules line”
means the stage is governed by archive accounting or an internal certificate and must still close
through the hard evaluator before admission.

| Vehicle stage | Frozen surface | Exact pinned source boundary | Admission consequence |
|---|---|---|---|
| Pair semantics | Pose sees frame0 and frame1; Seg sees frame1 only | `upstream/modules.py:70-84,107-113`; dimensions `upstream/frame_utils.py:10-13` [E15] | two independently described planes are required; `repeat-frame1` is a control, not the vehicle. |
| Learned plane emitter | joint preprocessor forks Pose and Seg from the same video pair | `upstream/modules.py:143-148`; Pose block map `upstream/frame_utils.py:51-79` [E15] | training must preserve both branches in one objective and cannot certify plane1 alone. |
| Seg term / cell certificate | last-frame select, bilinear resize, Seg forward, argmax disagreement | `upstream/modules.py:105,107-113` [E15] | train with frozen-logit/margin losses and admit actual native-f32 argmax pixels; PDW2 may describe target cells but does not realize them. |
| Pose proximity term | both-frame resize, RGB→YUV6, Pose forward, first-six-output MSE | `upstream/modules.py:64-84`; `upstream/frame_utils.py:51-79` [E15] | carry a per-pair Pose constraint on the exact 2×2 luma/chroma geometry. |
| Factor-2 solve | the shared resize that maps camera frames to scorer planes | `upstream/modules.py:73,109` [E15] | solve `A_num(X_t)=D Y_t` independently for `t∈{0,1}` under the exact integer ABI. |
| Hard admission | joint frozen forward and exact distortion composition | `upstream/modules.py:150-158` [E15] | native-f32 hard-oracle replay is required; rational equality alone refuses authority [E14]. |
| Rate allocation | exact archive byte term while frozen distortions remain invariant | `upstream/modules.py:155-158`; `upstream/evaluate.py:63,92` [E9], [E15] | allocate only measured exact archive bytes using §7; proxy entropy cannot authorize an update. |

The composition is deliberately ordered:

```text
counted learned description
  -> generic deterministic expander
  -> (Y0_uint8, Y1_uint8) at 384x512x3
  -> independent factor-2 integer preimages (X0_uint8, X1_uint8)
  -> frozen shared resize A
  -> {PoseNet(Y0,Y1), SegNet(Y1)}
  -> exact d_pose, exact argmax d_seg, exact archive bytes
```

The PDW2 relation is orthogonal but subordinate:

```text
PDW2 head-quotient cells/margins --target--> SegNet quotient features/cells
                                      ^
                                      |
                         spatial trunk applied to Y1_uint8
```

### 3.1 Exact 2×2 Pose geometry

After the shared resize, `rgb_to_yuv6` maps each 2×2 RGB block to six values at `192×256`:

- four luma phases `[y00,y10,y01,y11]` preserve every full-resolution luma sample by
  space-to-depth; this branch is lossless, so **there is no fine-scale luma slack**;
- `U_sub` and `V_sub` are one box average each over the four pixels; only these chroma means are
  Pose-visible; and
- within-block zero-mean chroma modes are in the pre-Pose kernel. Fine chroma below 2 pixels at
  `512×384` (about 4 pixels at camera resolution) is therefore a legal **d_seg-only** carrier axis
  with zero Pose cost by construction, subject still to exact uint8, resize, and Seg admission.

This is frozen-source structure, not a learned empirical correlation:
`upstream/modules.py:70-84` composes the Pose path and
`upstream/frame_utils.py:51-79` implements the four luma phases and two chroma box averages [E15].
The Pose-proximity objective in §4.3 and Pose-byte pricing in §7 therefore operate on this 2×2
block lattice. They may not invent per-pixel chroma obligations or false luma-null freedom.

### 3.2 Weight-level Seg operator binding

**Frozen lines optimized against:** the frozen `Conv2d(16→5,k=3,activation=None)` head and argmax
at `upstream/modules.py:103-113`, inside the joint fork and hard evaluator at
`upstream/modules.py:136-158` [E15], [E17]. The vehicle’s Seg content is not RGB fidelity. For a
local 144-D penultimate patch `f`, the exact pairwise head margin is

\[
m_{c,c'}(f)=\langle w_c-w_{c'},f\rangle+(b_c-b_{c'}),\qquad
d_{flip}^{feat}(f)=\frac{|m_{c,c'}(f)|}{\lVert w_c-w_{c'}\rVert_2}.
\]

The distance identity is exact in this local feature space only. Pullback to scorer-plane RGB
crosses the nonlinear Seg trunk and is realization-limited. Given desired winner `c*`, the
vehicle’s content obligations are the halfspaces

\[
m_{c^*,r}(f_{h,w}(Y_1))\ge \mu_{h,w,r}\quad\text{for every rival }r\ne c^*.
\]

Each `mu` is a typed, candidate-local safety margin settled by native-f32 replay; this SPEC guesses
no universal value. PDW2 stores four reference-relative affine rows—20 fp32 coefficients for
`K=5,d=4`—which are algebraically sufficient to derive all ten pairwise differences. Its measured
n600 packet declares only the nine observed adjacency edges, omitting `(2,4)`, and ships zero
explicit normal/offset bytes. Its certificate API is therefore adjacency-scoped. For canonical
`i<j`, PDW2 uses `g_ij=ell_j-ell_i`: class `i` wins when `g_ij<=0`, class `j` strictly when `g_ij>0`.
(**#564 correction:** `torch.argmax`/NumPy `argmax` return the FIRST index on exact co-maxima, so
cells are lexicographically half-open — at `g_ij=0` only the lower index `i` wins; the earlier
symmetric prose double-assigned equality. Executable consumers already use first-max
[`f32_receiver_arithmetic_exactness_admissibility_v1`]; this is a prose/certificate correction only.)
The receiver must preserve its zero-sum native-fp32 score-reconstruction order because collapsing
or reassociating a normal/offset can alter ULP ties [E5].

The first Seg basis is the measured four-dimensional quotient, not more generator capacity:

- project centered five-class logits into the sign-fixed left-singular `U4` frame with singular
  values `3.1283763256, 2.1542713873, 2.0247078699, 1.7962638357`, without singular-value
  division; retain all ten pairwise margin constraints in the coupled four-dimensional span;
- at fixed parameter/code count, A/B this `U4`/pair-margin parameterization against the raw
  output basis **before** widening the trunk or modulation dimension; no superseded directional
  gain is used as a prior or acceptance threshold; and
- use raw RGB error only as a logged diagnostic or hard-render check. If a cheaper exact
  hyperplane, quotient-basis, channel, or Pose-block expression exists, training on raw RGB as the
  content target violates the intrinsic-complexity cut in §5.1.

Channel and spatial allocation must remain resolved. Seg consumes full RGB [E20]. The historical
`50% Road / 19% Lane / 13% Undrivable` shares only seed candidate-local acquisition order. The
`76.87%` Lane statistic is narrower: it attributes Road–Lane flips induced by destroying detail in
one shared 16-channel stride-2 skip; it is not Lane’s error share and proves no private Lane
channel. The empirical `r50≈85 px`, `r90≈206–424 px` demands full-neighborhood collateral checks,
with about `300 px` as a working guard rather than a local-patch assumption [E19]. Region chroma,
pair-annulus chroma, luma, shared-skip, and deep-path effects remain distinct conditional rows;
the ablation costs in [E20] are necessity ceilings, not gains. The measured filter census in
[E20] supplies the complementary scale basis: high-pass/oriented response is concentrated at
strides 2–8 and deep paths are predominantly low-pass/axis-aligned. That stratifies skip/deep and
spatial-scale rows; it does not establish the causal orientation hypothesis without through-R A/B.

Weight-response guidance is already in immutable custody and must be consumed, not re-derived:
Seg `g_y`, `q`, `Lip_local`, and `g_x=A^Tg_y`; Pose6 `J_y` and `J_x=A^TJ_y`, for all 24 real
pairs [E18]. These first-order surfaces propose and constrain updates; native-f32 hard replay
admits them. The integer-plane emitter spends learned capacity in `range(A)`, while generic
factor-2 preimage solving supplies camera bytes and may choose `ker(A)` fill without learning it.
The observed roughly `52%` null-response energy is a sample-specific capacity-waste prior, not a
byte saving [E19]. Frozen weights, VJPs, source planes, and safety-margin tables are encode-side
instruments only: none may enter the counted payload or scorer-free decoder.

PDW2 is not a substitute for either integer plane. It is admitted in one of two mutually exclusive
roles only:

- **training-only certificate:** its cells/margins constrain the learned emitter and are compiled
  into counted weights/codes; no separate packet is shipped; or
- **receiver-consumed payload:** a scorer-free deterministic pullback consumes the counted packet
  and changes emitted plane bytes. The frozen spatial trunk is used encode-side only to optimize
  and validate that `Y1` realizes the target cells; it is never present in the decoder. A
  packet-mutation/no-op test must prove consumption.

Until charter C3 closes, the first role is the only admissible role and the packet’s `133 B` is not
claimed as archive savings [E5].

## 4. Train-least design: learned, projected, solved, and verified

### 4.1 Learned state

**Frozen line optimized against:** the joint fork and both frozen consumers at
`upstream/modules.py:143-158`, with Pose’s exact block map at
`upstream/frame_utils.py:51-79` [E15].

The learned state is only the compact information needed to emit two scorer planes:

- shared generator parameters `theta`;
- independently addressable per-pair/per-frame codes `c[p,0]` and `c[p,1]`;
- typed topology/group state required by the generator;
- optional PDW2-target loss state, if training-only; and
- only after charter C8, optional per-output-channel bit-depth/exponent/topology state.

Those codes control the operator-native `U4`/pair-margin, class/channel, and Pose-block
coordinates specified in §§3.1–3.2. Raw RGB is the derived renderer output and hard-admission
surface, not the fidelity target. A raw-output-basis control remains mandatory at fixed capacity
so basis value is measured before capacity changes.

The generator emits continuous precursor planes

\[
Z_{p,t}=G_\theta(c_{p,t}),\qquad
Z\in\mathbb{R}^{N\times2\times384\times512\times3}.
\]

There is no learned camera-resolution texture, camera preimage, resize kernel, scorer, or evaluator
lookup. Those are solved or frozen.

### 4.2 Projected in the forward pass

**Frozen line optimized against:** the shared bilinear scorer-plane inputs at
`upstream/modules.py:73,109`, followed by Seg argmax at `:111-113` and the Pose block transform at
`upstream/frame_utils.py:51-79` [E15].

The scorer-facing training value is

\[
Y_{p,t}=Q_{u8}(Z_{p,t})=
\operatorname{uint8}(\operatorname{clip}(\operatorname{round}(Z_{p,t}),0,255)).
\]

Backward differentiation uses the saturation-aware straight-through rule already present in the
Torch corpus [E16]. The same forward bytes must be implemented in MLX and in a deterministic
NumPy-fp32 reference. This is the rounding-in-training choice: no float-space plane may receive an
admission verdict.

The integer-lattice feasibility operator is a **forward feasibility projection/check**, not an
additional learnable decoder. Admission requires exact `uint8 [384,512,3]`, fixed
`874×1164→384×512` exact half-pixel geometry, disjoint non-overlapping supports, at most two taps
per axis, and support numerators summing to each denominator. Under those contract conditions,
canonical support-fill constructs an exact preimage for every uint8 plane. For each plane
independently, the solved camera frame must satisfy

\[
A_{\rm num}(X_{p,t})=D\,Y_{p,t},\qquad X_{p,t}\in\{0,\ldots,255\}^{874\times1164\times3}.
\]

Refusal applies to dtype, shape, geometry, contract, or numerator-verification failure—not to an
otherwise valid contract plane. Gradients do not flow through an invented soft preimage solver.

### 4.3 Pose proximity is part of the optimization object

**Frozen line optimized against:** both-frame preprocessing and scored Pose outputs at
`upstream/modules.py:70-84`, with the exact 2×2 luma/chroma map at
`upstream/frame_utils.py:51-79` [E15].

Let `Ysrc[p,t]` be the frozen source scorer plane used only during training, never shipped. Let
`Phi_pose` be exactly `[y00,y10,y01,y11,U_sub,V_sub]` from `rgb_to_yuv6`. Define the
Pose-visible proximity diagnostic

\[
\rho_p^{2\times2}(Y,Y^{src})=
\operatorname{RMSE}\left(\Phi_{pose}(Y_p),\Phi_{pose}(Y_p^{src})\right).
\]

This keeps all four lossless luma phases but constrains chroma only through its two 2×2 block
means. Within-block zero-mean chroma is deliberately absent from `rho_p^{2×2}` because Pose cannot
see it; raw RGB RMSE may be logged as a diagnostic but may not tax that legal d_seg-only carrier.

The base constrained program is

\[
\begin{aligned}
\min_{\theta,c}\quad &
L_{\rm seg,train}^{f32}(Y_1,L^*)
+ \eta_\rho\sum_p\rho_p^{2\times2}(Y,Y^{src})
+ J_{\rm rate,enabled} \\
\text{s.t.}\quad &d_{{\rm pose},p}^{f32}(Y_{p,0},Y_{p,1}) < \tau_{\rm pose}
\quad\forall p,\\
&\tau_{\rm pose}=2.5\times10^{-4},\\
&A_{\rm num}(X_{p,t})=D\,Y_{p,t}\quad\forall p,t.
\end{aligned}
\]

`L_seg,train` is a differentiable frozen-logit/margin loss through the exact uint8 forward; the
hard `d_seg` argmax value remains admission-only at `upstream/modules.py:111-113`. `eta_rho` is
typed but has no sealed numeric value here; C4 must derive it from the measured
proximity/Pose response rather than guess it. Duals and `eta_rho` may
change only at stage boundaries.

**#564 POOLED-POSE AMENDMENT (DERIVED source contradiction, supersedes the per-pair hard-cap
prose above and in rows C4/C9):** the frozen evaluator pools Pose globally —
`S_pose = sqrt(10·D)`, `D = (1/N)·Σᵢ qᵢ = ‖e‖²/(6N)` (`modules.py:82-84`, `evaluate.py:81-92`) —
so the score-term sublevel set is ONE L2 ball in `R^{6N}`, never 600 per-pair balls. A literal
per-pair `qᵢ<2.5e-4` veto is STRICTER than the score and forbids cross-pair rate allocation the
score explicitly allows (measured witness: precision-drop rows with global `D<2.5e-4` but 2–4
pair-cap violations). The binding constraint is the pooled `D` (equivalently the norm
`S_pose(e)=‖e‖/sqrt(360)` at N=600, whose gradient has CONSTANT norm away from 0 — the
`1/sqrt(D)` coefficient blow-up cancels in native error coordinates). The `2.5e-4` crossover is a
coordinate-derivative identity, NOT a feasibility wall; it may inform telemetry, never a hard
veto. A typed per-pair tail-risk cap may exist only as `ASSUMED_ROBUSTNESS_GUARD`, default OFF,
until an axis-drift probe earns it. Consumers: C4 objective (one pooled term + one dual), C9
(one global Pose dual in the KKT), #536 waterfill. Per-pair qᵢ maxima/quantiles REMAIN mandatory
diagnostics. This objective still rejects both false simplifications: “Pose is free” and “Pose
requires copying frame1.”

The optimization consumes the immutable sidecars in [E18] rather than launching a new derivative
harvest: `g_y/q/Lip_local` propose and trust-region Seg scorer-plane updates, `J_y` constrains the
six Pose outputs on the two-plane tensor, and `g_x/J_x` audit the solved camera-preimage pullback.
They are local proposal surfaces, not verdicts. Every proposed step is rounded through the actual
integer planes and admitted by the full native-f32 frozen hard oracle; a sidecar-predicted gain or
zero alone cannot enter an R-D row.

### 4.4 Default-OFF joint size-in-loss term

**Frozen line optimized against:** it may reduce only the exact archive-rate term while preserving
both distortion outputs at `upstream/modules.py:155-158`; the exact rate is read at
`upstream/evaluate.py:63,92` [E8], [E15].

The only high-EV neural training candidate identified by [E8] is the following
specification-only, default-OFF grouped weight-quantization term:

\[
q(w;b,e)=2^e\operatorname{round}(\operatorname{clip}(2^{-e}w,
-2^{b-1},2^{b-1}-1)),
\]

\[
B_{\rm relaxed,bits}=\sum_{g\in active}n_g b_g+B_{\rm topology,bits},
\qquad
J_{SC}=J_{\rm witness}+\gamma\frac{25 B_{\rm relaxed,bits}}
{8\cdot37{,}545{,}489}.
\]

The typed default is `mode=OFF`, `gamma=0`, output-channel grouping, `b_g∈{0,...,8}`, and one
learned exponent per output channel. For `b_g=0`, the group is pruned/absent and the displayed
quantizer is not evaluated; `b_g≥1` uses the formula. The MLX custom VJP must return gradients for
weight, bit depth, and exponent. Activation is permitted only at a stage boundary, after the OFF
control has a receiver-closed baseline and C8 passes. All `b`, `e`, topology, EMA, optimizer, RNG,
stage, and epoch state is resume state. Pruning occurs only at stage boundaries.

### 4.5 Solved and verified state

**Frozen line optimized against:** the two exact camera→scorer resizes at
`upstream/modules.py:73,109`, and final hard distortions at `upstream/modules.py:155-158` [E15].

The receiver solves, rather than learns:

- exact factor-2 camera preimages for both integer planes;
- canonical support fill and declared integer arithmetic;
- camera-frame assembly and raw-video serialization; and
- packet/hash verification.

Encode-side admission then runs the actual native-f32 frozen hard oracle. The three-anchor law
[E14] proves that exact numerators alone are insufficient: preimage selection changes native-f32
resize accumulation from camera bytes, which changes scorer inputs and may alter frozen-trunk
logits/argmax. A tie-aware preimage policy is therefore a measured charter, not an assumption.
No scorer weights or ground-truth table enter the decode path.

## 5. Payload grammar and free-versus-counted split

The baseline grammar is a design contract, not an implemented byte claim. Section byte values are
`UNKNOWN` until the new vehicle emits an exact archive. Canonical order is mandatory so parsing,
hashing, and double decode are deterministic.

### 5.1 Intrinsic-complexity cut

The representation is bounded both ways. Completeness (§9) requires every force measured in the
frozen dynamics; Kolmogorov minimality refuses everything else. For fixed generic decoder `D`, the
rate object is

\[
B_{intrinsic}=|p^*|+|seed|,\qquad
p^*=\arg\min_{p:\;D(p,seed)=(Y_0,Y_1)} |p|.
\]

Kolmogorov complexity is not numerically computable here; exact receiver-closed archives are
measured upper bounds. The operational rule is still falsifiable: every payload field and decoder
mechanism must cite a measured frozen demand, and deleting it must either change the decoded
sufficient statistic or make decoding ambiguous. Otherwise it is cut. Incidental complexity is a
defect, not completeness.

This rule removes three initially conceivable sections: topology is folded into the generator
stream; integrity/custody hashes live outside the archive unless a minimal parser field is needed;
and speculative repair records are absent until a measured residual proves they beat changing the
base description. No source/runtime hash is paid inside the payload merely for documentation.

| Order | Minimal section | Required content | Baseline coder | Counted/free | Measured demand and exact frozen line optimized against |
|---:|---|---|---|---|---|
| 0 | `vehicle_header.v1` | only decoder-essential magic/version, section lengths, and a seed if the expander actually consumes it | canonical fixed-width bytes, no entropy coder | **COUNTED** when physically present | deterministic unambiguous decode while leaving both distortions at `upstream/modules.py:155-158` invariant; dimensions are fixed generically by `upstream/frame_utils.py:10-13`; minimize against `upstream/evaluate.py:63,92` |
| 1 | `generator_base.v1` | quantized shared tensors, typed shapes/scales, and only active topology/exponent fields needed to decode them; no frozen head weights or VJPs | per-tensor int8 + Brotli-q11 | **COUNTED** | encode `U4`/pair-margin and channel/Pose-block control only insofar as it changes sufficient statistics at `upstream/modules.py:70-84,103-113,136-158` and `upstream/frame_utils.py:51-79`; coder control [E6], bytes unknown |
| 2 | `pair_frame_codes.v1` | independently addressable `c[p,0]`, `c[p,1]` and only charged operator-native modes that change an emitted plane | per-tensor int8 + Brotli-q11 | **COUNTED** | pair-local margin/class/channel coordinates serve frame1 Seg at `upstream/modules.py:103-113`; block coordinates serve two-frame Pose at `upstream/modules.py:70-84` and `upstream/frame_utils.py:51-79`; raw RGB is derived output, coder control [E6], bytes unknown |
| 3 | `pdw2_margin.v1` | exact PDW2 packet **only** if its scorer-free pullback changes `Y1`; otherwise absent | PDW2 raw grammar + Brotli-q11 | **COUNTED** only in consumed mode | Seg head/cell demand at `upstream/modules.py:105,111-113`; measured inner payload `138/133 B` [E5], but length/hash/container overhead is unknown and counted by C6; role blocked by C3 |

The block-FP/LZMA representation [E7] is an alternate encoding only of tensor arrays in sections
1–2, not an additional stacked section. No header or topology encoding was measured; all framing
and any active topology remain separately counted and unknown. Block-FP is default OFF and cannot
be selected before C7. Likewise, a learned entropy model, fitted prior, table, or topology is
video-derived state and is counted in full.

**Rule-118 free mechanism:** generic deterministic NumPy-fp32 expansion code, fixed arithmetic,
factor-2 solve, parser logic, raster/video writer, and non-video-derived constants are mechanism,
not learned payload only when they reside solely in the generic uncounted inflate/runtime surface.
If any copy or fitted value is physically stored in `archive.zip`, those bytes count. **Counted
information:** every source/video-fitted parameter, code, mode, margin packet, fitted table, header,
and container field. Regardless of this design ledger, the final authority is the exact physical
`archive.zip` byte count; nothing is subtracted after packaging.

Forbidden payload includes scorer weights, SegNet/PoseNet executables or parameters, source frames,
ground-truth argmax tables, source-derived lookup tables disguised as code, or any uncatalogued
side channel.

## 6. Receiver chain and runtime budget

The deterministic receiver chain is:

1. strict parse and canonical re-encode of minimal sections 0–2, plus section 3 only if consumed;
2. generic deterministic NumPy-fp32 expansion into two precursor planes;
3. exact clip/round to independent `Y0_uint8` and `Y1_uint8`;
4. separate factor-2 solve and numerator verification for frame 0;
5. separate factor-2 solve and numerator verification for frame 1;
6. camera-frame assembly, ordered raw-video write, and success-only scratch cleanup;
7. independent second decode and byte-identical output comparison; and
8. encode-side only: native-f32 hard-oracle replay, exact archive parse-back, and score custody.

| Receiver stage | Exact frozen line it serves |
|---|---|
| parse + expand | must reconstruct only state consumed by `upstream/modules.py:70-84,105-113,143-158` and `upstream/frame_utils.py:51-79` |
| uint8 integer planes | exact inputs to the shared resize at `upstream/modules.py:73,109` |
| independent frame0/frame1 preimage solves | preserve the same resize outputs at `upstream/modules.py:73,109`; frame0 remains Pose-visible at `:70-84` |
| camera/video assembly | exact pair layout consumed at `upstream/modules.py:143-148`; dimensions from `upstream/frame_utils.py:10-13` |
| double decode | prove deterministic invariance of hard outputs at `upstream/modules.py:155-158` |
| encode-side admission | hard distortions at `upstream/modules.py:150-158`, with exact bytes at `upstream/evaluate.py:63,92` |

The decode path contains no scorer. Resume state lives on disk per pair/stage; completed pair files
are preserved and content-addressed, with atomic manifests.

### Runtime accounting

| Formulation | What was actually timed | n600 status | Verdict |
|---|---|---|---|
| arbitrary rational, two planes | n24 solve `597.7790400451 s` | derived `249.0746000188 min`, before expansion/I/O [E2] | dead for this formulation under the 30-minute limit |
| factor-2 integer, `repeat-frame1` | n12 full double decode `4.53 s` slower run | derived `3.775 min` [E3] | viable control class only; cannot be transferred to two distinct solves |
| factor-2 integer, `description-frame0.v1` | structural independent frame0/frame1 solves [E4] | **UNMEASURED** | C1 is the first build gate |

**CORRECTED 2026-07-19 (operator: "It's 30 minutes total for full auth eval on contest
hardware"; source `upstream/README.md:113` — "The official evaluation has a time limit of 30
minutes").** The 1800 s limit binds the ENTIRE official evaluation — inflate PLUS the 600-sample
scoring pass — not the inflate invocation alone. The prior per-invocation reading was wrong in
both directions it could matter: it granted each inflate its own 1800 s, and it omitted
`T_scoring` entirely. The only admissible runtime inequality is:

\[
\underbrace{T_{parse}+T_{expand}+T_{solve0}+T_{solve1}+T_{assemble/I/O}+T_{verify}}_{T_{inflate}}
\;+\;T_{scoring}\;<\;1800\;\mathrm{s}\quad\text{on contest hardware.}
\]

`T_scoring` (evaluate.py's SegNet/PoseNet pass over 600 samples plus archive accounting) is not
ours to tune and is NOT guessed — it is measured only by running the actual official evaluation
on contest-class hardware (Modal, #381 envelope; operator pre-authorized that measurement on a
CLOSE verdict). The decode headroom is therefore `1800 − T_scoring`, itself a MEASURED quantity.
No component budget is guessed. C1 must report every inflate term and total them on full n600;
local wall-clock is adjudicated ONLY via a measured local↔contest calibration or the Modal
actual-time run (never an invented margin — the timing axis obeys the same apples-to-apples
discipline as scores). A second timed invocation is reproducibility evidence, reported
separately, never added to the contest runtime. The 3.775-minute projection is an optimistic
inflate-only control datum: it never included `T_scoring` and is not a two-plane forecast.

**HARDWARE EXPLOITATION (operator 2026-07-19: "we can fully leverage and exploit the hardware
available ... CPU and GPU cuda and multiprocessing and threading and async, anything within the
contest rules").** The receiver is NOT constrained to a single-threaded CPU process. Per
`upstream/README.md:113` the INSTANCE CHOICE IS OURS: declaring a GPU requirement buys a T4
(26 GB RAM, 16 GB VRAM, CUDA); otherwise a 4-core/16 GB CPU instance. Within the 30-minute total,
multiprocessing, threading, async I/O, and CUDA are all legal. Design consequences (binding on
C1/C2/C11):
- the 600 pairs are embarrassingly parallel — the baseline CPU receiver uses a 4-worker
  process pool (deterministic: per-pair outputs are independent; assembly order fixed);
- the factor-2 preimage solve is EXACT INTEGER arithmetic — it is CUDA-parallelizable
  DETERMINISTICALLY (integer ops carry no accumulation-order drift; the fp32 hazard lives in
  float paths only, which the receiver does not execute). A T4 lattice-solve path is therefore
  a legal, bit-exact throughput lever, subject to the C1 measurement discipline;
- `T_scoring` itself depends on the instance choice (T4 scoring pass vs 4-core CPU scoring
  pass) — the instance decision is made from MEASURED totals on both instance classes, not
  assumed. C1's receipt reports the single-worker baseline AND the exploited configuration
  actually intended for the contest run, per stage, on each candidate instance class;
- determinism remains the hard constraint: any parallel schedule must produce byte-identical
  output across runs and hosts (double-decode identity is already a C1 pass condition).


## 7. Byte-budget derivation and one shared KKT

**Frozen line optimized against:** exact Pose/Seg distortions at
`upstream/modules.py:155-158`, exact archive accounting at `upstream/evaluate.py:63,92`, and Pose
block visibility at `upstream/frame_utils.py:51-79` [E9], [E15].

### 7.1 Candidate-specific strict budget

For a strict target `S_target`, a candidate with measured `(d_seg,d_pose)` may contain at most

\[
B_{max}=\left\lceil
\frac{37{,}545{,}489}{25}
\left(S_{target}-100d_{seg}-\sqrt{10d_{pose}}\right)
\right\rceil-1
\]

integer bytes, provided the parenthesized term is positive. This is a conditional ceiling, not an
allocation and not evidence that the vehicle can achieve the distortions.

For the program target `S_target=0.15`:

- at hypothetical `d_seg=d_pose=0`, `B_max=225,272 B` (**DERIVED** from the frozen score law);
- at hypothetical `d_seg=0` and the **DERIVED** Pose crossover `d_pose=2.5e-4`,
  `B_max=150,181 B` (**DERIVED** from [E10]);
- at `d_seg=0` and the **MEASURED positive-band** Pose value
  `d_pose=2.521975392375284e-5`, `B_max=201,422 B` (**DERIVED conditional ceiling**, not achieved
  archive bytes; [E10]); and
- any actual candidate recomputes the ceiling from its exact receiver-closed values. It does not
  inherit any worked ceiling.

### 7.2 Marginal admission price

For a change from old to new candidate, the real-valued byte indifference allowance paid by the
exact distortion gain is

\[
\Delta B_{break\text{-}even}=\frac{37{,}545{,}489}{25}
\left[100(d_{seg}^{old}-d_{seg}^{new})
+\sqrt{10d_{pose}^{old}}-\sqrt{10d_{pose}^{new}}\right].
\]

Strict improvement requires
`Delta B_archive < Delta B_break-even`; the strict integer allowance, when the right side is
positive, is `ceil(Delta B_break-even)-1`. At fixed Pose the real-valued coefficient reduces to
`150,181,956 B` per unit Seg improvement [E9]. A payload section is admitted only when its
**marginal exact archive bytes** are below the value of its realized Seg/Pose gain. Proxy loss,
unframed tensors, and raw section bytes cannot satisfy this rule.

### 7.3 Pose bytes are priced on the 2×2 block lattice

The Pose-visible sufficient statistic per frame and 2×2 scorer-plane block is exactly four luma
phases plus two chroma box averages (`upstream/frame_utils.py:51-79` [E15]). Consequently:

- chroma proximity state and its bytes are priced per `192×256` 2×2 block, a factor-4 coarser than
  a `384×512` per-pixel chroma field;
- the four luma phases preserve every luma sample, so luma state receives no factor-4 discount;
- within-block zero-mean chroma belongs to the d_seg-only channel and is priced only by its realized
  Seg gain after a hard zero-Pose-effect check; and
- the final admission price is still the nonlinear exact `d_pose` term in §7.2, not a proxy count.

Any allocator that charges Pose per raw chroma pixel or treats luma as box-averaged is rejected.

### 7.4 Non-additive allocation

There is one shared KKT across all pool×channel consumers:

- generator base, pair/frame codes, PDW2-consumed state, and minimal headers all draw from the same
  archive-rate term;
- multiple levers acting on the same measured pool compete; their gains are not summed;
- section order is evaluated marginally after prior admitted sections; and
- the advisory `margin_m0p3 ↔ precision_drop1` closest-secant KKT candidate [E9] seeds measurement
  order only. It is not a continuous optimum and is not transplanted as a new-vehicle optimum.

The rounded historical pool totals [E11] are a topology map for overlap, not an initial vehicle
budget. The new vehicle earns its own receiver-closed R-D curve before allocation.

Channel resolution does not create multiple rate terms. Conditional deletion/addition deltas are
measured from a named parent after full archive re-encode and need not add because entropy and
overlap are non-additive; the physical section/container reconciliation remains total-byte
authority. Each active joint-pool frontier carries at least
`class_or_pair × U4 direction × skip/deep × range(A)/ker(A) × Pose block`, plus luma/chroma and
region/annulus roles. The single shared `lambda` is applied to adjacent secants of those joint
frontiers, never to separately summed Lane, skip, chroma, or nullspace “headrooms.”

## 8. Owed-before-build chain — falsifiable charters

These charters are ordered. Local implementation may prepare a later interface, but no training
launch becomes authorized until all predecessors required by that launch are closed. Each negative
is formulation-scoped; a failure records the exact rejected formulation and leaves the broader
family open.

| Order | Charter and object | Corpus / axis | Pass condition | Literal falsifier | Durable output owed |
|---:|---|---|---|---|---|
| C0 | **Canonical-law/source reconciliation.** Reconcile the four-anchor Pose registry with the three-anchor Python builder, the three-anchor f32 registry with the two-anchor builder, stale predecessor status prose, and registry row 745’s internally stale `domain_of_validity.unmeasured` field. | source-only, MAIN review | builders re-emit the latest append-only anchors without changing historical registry rows; registry payload/domain agrees with its appended anchors; successor supersession map approved | any builder or registry payload still reports stale anchor count/domain, or MAIN finds an unauthorized rewrite | reviewed source diff, focused regression, new append-only reconciliation memo |
| C1 | **Two-independent-plane receiver ABI and timing.** Use distinct exact integer planes under `description-frame0.v1`; measure parse, expansion, solve0, solve1, assembly/I/O, verification. | full n600; local timing first, then contest-compatible runtime custody | exact numerator equality on both planes; strict parse/re-encode; double-decode byte identity; timing adjudicated against the 30-min TOTAL official-eval budget (T_inflate + T_scoring < 1800 s on contest hardware; corrected 2026-07-19 — see §Runtime accounting): measured local↔contest calibration or Modal actual-eval time, never a bare local bound; no `repeat-frame1` shortcut | either plane is copied/aliased; any numerator mismatch; output drift; timing verdict issued from local wall-clock alone or per-invocation 1800 s reading; missing component timing | content-addressed n600 timing receipt, commands/env/source hashes, stage manifests, preserved output digest |
| C2 | **Learned integer-plane emitter and basis-before-capacity A/B.** Implement two independently indexed outputs with forward `uint8` rounding, saturation-aware STE, exact contract validation, and NumPy-fp32 reference. At identical parameter/code count, compare raw output coordinates with sign-fixed `U4`/pair-margin coordinates before any width/modulation increase. | synthetic fixtures, real n24 advisory, both frames | MLX/Torch/NumPy forward integer bytes match; nonzero in-range gradient; saturation blocks gradient; both planes differ on a fixture; every contract-valid plane passes exact factor-2 numerator verification; fixed-capacity basis A/B emits matched exact bytes and hard Seg/Pose rows | float values reach scorer; one plane is ignored/copied; parity below the repository threshold; invalid dtype/shape/geometry accepted; numerator verification fails; capacity changes before basis value is resolved; a superseded directional-gain number is used as acceptance evidence | typed DSL/config, parity receipt, exact fixture hashes, fixed-capacity basis A/B receipt, per-stage resume schema |
| C3 | **PDW2 spatial composition.** Choose training-only or receiver-consumed mode explicitly. In consumed mode, a scorer-free deterministic pullback changes emitted plane1; the frozen spatial trunk is encode-side optimization/admission only. | frame195 plus ordinary fixtures, then real n24 through R | zero-sum native-fp32 score reconstruction retained; packet mutation changes emitted plane/cell result; deleting packet refuses or changes result; all declared adjacency halfspaces realize after R; full-neighborhood collateral is checked through at least the measured `r90` support; exact archive and Seg/Pose deltas measured | packet is a no-op, target cells are not realized, decoder contains scorer state, collapsed-normal arithmetic changes a tie, patch-local evaluation omits full-neighborhood collateral, or Pose regresses beyond gate | receiver-consumption receipt or explicit training-only decision record; adjacency/margin and collateral rows; no double counting |
| C4 | **Pose-plane proximity objective (#564-AMENDED: pooled).** Add `rho_p^{2×2}` and ONE pooled Pose constraint on the frozen global term `D=(1/N)Σqᵢ` (equivalently the norm `‖e‖/sqrt(360)`); per-pair caps only as `ASSUMED_ROBUSTNESS_GUARD` default OFF; derive `eta_rho`/dual schedule only at stage boundaries from real response rows. | real n24 first, then n600; native-f32 hard oracle | pooled `D` passes the declared global budget with per-pair maxima/quantiles emitted as diagnostics; four lossless luma phases and two block-mean chroma values are accounted exactly; within-block chroma is untaxed only after zero-Pose-effect verification; d_pose, d_seg, block-granular bytes, and diagnostics are emitted together; OFF control retained | re-introducing evaluator-underived per-pair hard vetoes; mean emitted WITHOUT per-pair diagnostics; luma is treated as lossy; chroma is charged per pixel; proximity term is inert; coefficient is unproven magic | typed objective/DSL, joint curve receipt, Pose-law successor consumer test, resume state for the ONE global dual |
| C5 | **Preimage/native-f32 admission.** Compare canonical support-fill and predictor-optimal preimages for identical exact integer planes; test a tie-aware policy against all three f32 anchors. | frame195, pair125, Rung-E n48, n600 lattice corpus | declared policy reproduces hard-oracle labels or a measured winner/rival safety margin closes every case; policy ID is payload-bound | exact numerators still produce unexplained flips; anchor behavior changes across decode; margin rule is inferred rather than measured | preimage-policy A/B receipt, updated canonical-law regression, hard-oracle refusal artifact for failures |
| C6 | **Baseline grammar and exact coder/channel accounting.** Serialize minimal sections 0–2 and section 3 only if C3 admits it; prove canonical parse/re-encode and count all header/container overhead. Use int8+Brotli only as baseline for tensor sections. | real n24 then n600 exact archive bytes | every field passes the intrinsic-complexity deletion test; every source-fitted bit is counted; double serialization is identical; physical section sums reconcile to exact archive bytes. Every learned/optional field also emits a conditional row keyed by `parent_candidate_hash,section,pool,frame_role,class_or_class_pair,spatial_role,color_role`, with exact post-reencode byte delta, total/by-class/by-pair Seg delta, per-pair Pose vector/max, and `overlap_pool` | donor byte anchors reused as predictions; uncounted fitted state; nonessential field survives; section sum differs from archive; aggregate-only rows; conditional channel marginals are falsely required to sum despite entropy interaction | grammar implementation, physical byte ledger, conditional channel-role ledger, parse-back tests, exact archive hash |
| C7 | **Block-FP admissibility A/B.** From the same source checkpoint/corpus compare block32/threshold0.25/LZMA with int8+Brotli, allowing each its specified reconstruction. | real n≥24 through actual R for advisory acquisition; n600 contest-axis confirmation before admission | n24 emits receiver-closed advisory `Delta S_n24`, exact framed bytes, per-pair d_seg/d_pose, and deterministic reference parity; n600 exact archive/contest-axis result is lower before promotion | rate-only win disappears through R; any Pose gate violation; Block-FP parse-back differs from its deterministic reference reconstruction; n600 fails to confirm | paired archive receipts and formulation-scoped accept/reject row |
| C8 | **Default-OFF joint size-in-loss.** Build `WitnessSelfCompressionGauge` exactly as §4.4; start from an admitted OFF baseline. | deterministic OFF/ON twin n24, then n600 only if n24 passes | ON exact archive has `Delta S<0`; full fitted overhead counted; pose-neutral response clears the conditional `Delta d_seg<0.000410156` knee when applicable; resume covers `b/e/topology` | OFF path changes; proxy bits do not predict archive; knee fails; cross-host decode drift; incomplete resume state | typed DSL, OFF no-op test, twin archive receipts, stage checkpoints |
| C9 | **Receiver-closed channel R-D and one KKT.** Sweep only pre-registered section/pool levers after C1–C8; price Pose state at 2×2 block granularity; recompute candidate-specific §7 budget. Required axes are `pool × class/pair-hyperplane × U4 direction × {skip,deep} × {range(A),ker(A)} × Pose block`, with luma/chroma and region/annulus roles retained. | real n24 acquisition, n600 confirmation; exact archive bytes | every active non-additive pool first forms a receiver-closed joint frontier carrying the C6 channel-role rows. One shared lambda is applied to adjacent pool-frontier secants; each selected secant brackets lambda or satisfies the correct boundary inequality; ONE global Pose dual on the pooled `D` (#564 amendment — never 600 pair duals; per-pair maxima emitted as diagnostics), neighbors/residual, parent hashes, and overlap map are emitted | closest-pair alone is closure; one-point curve; aggregate-only bytes or missing class/pair/color/skip attribution; `19%×77%`; chroma-ablation worth called gain; per-pixel chroma or luma-slack fiction; raw payload substituted for archive; additive pool gains | canonical joint-pool R-D and channel-role rows, parent hashes, overlap map, shared-lambda KKT receipt, continual-learning ingest |
| C10 | **Vehicle-specific resumability.** Apply the #537 apparatus to every training stage and receiver stage. | CPU locked control/resume; available GPU parity as separate axis | control twice deterministic; forced interruption loses at most one checkpoint interval; live/EMA/optimizer/RNG/dual/topology/event/stage/epoch and receiver pair progress restore; all stage checkpoints preserved | any loop-end-only save; EMA absent; stage file overwritten; resumed discrete state differs | tracked CPU receipt, GPU receipt if run, checkpoint inventory with hashes, resume regression |
| C11 | **Full custody and launch gate.** Run the same exact candidate bytes through full n600 receiver, storage hygiene, parse-back, and separate contest CPU/CUDA evaluation only after explicit authority. | `[contest-CPU Linux x86_64]` and `[contest-CUDA]` kept separate | full command/hardware/runtime/archive/source hashes; measured T_inflate + T_scoring < 1800 s TOTAL per official evaluation on contest hardware (corrected 2026-07-19); deterministic inflate; exact score components; both axes reported without inference; MAIN and operator GO | missing custody field, axis substitution, either receiver timeout, archive drift, or any scorer/source payload | promotion-grade receipt bundle, lane claim, governed-launch log, MAIN-reviewed pointer decision |

The first concrete build unit is C0→C1, not a long generator training run.

## 9. Completeness table — demanded force versus present mechanism

| Force demanded by measured dynamics | Evidence demanding it | Mechanism presently available | Status | Exact missing leg |
|---|---|---|---|---|
| two-frame semantic independence | Pose uses both frames while Seg uses frame1 [E15] | `description-frame0.v1` structural path [E4] | **PARTIAL** | distinct-plane n600 runtime and custody C1 |
| integer-plane rather than arbitrary-rational realization | arbitrary rational projects to 249.07 min [E2] | factor-2 exact integer solve; 3.775-min copy control [E3] | **PARTIAL** | independent-plane timing and end-to-end total C1 |
| quantization in the training forward | exact integer receiver ABI and clip-round failure [E12] | Torch `Uint8STE` corpus [E16] | **PARTIAL** | MLX emitter, NumPy parity, feasibility refusal C2 |
| Seg cell/margin force | 7 native-f32 disagreements after rational replay [E1] | PDW2 exact head packet [E5] | **PARTIAL** | spatial pullback or explicit training-only consumption C3 |
| exact head-halfspace content | ten coupled pair hyperplanes in a four-dimensional quotient [E17] | `U4`/pair-margin objective §3.2; PDW2 has nine observed adjacency certificates [E5] | **DESIGN ONLY** | fixed-capacity basis A/B, candidate-local margins, nonlinear spatial realization C2/C3 |
| immutable derivative guidance | 24 custodied Seg/Pose VJP sidecars [E18] | proposal/trust-region consumption in §4.3 | **PRESENT INPUT / MISSING VEHICLE CONSUMER** | hash-bound recall, sidecar consumer test, hard-oracle admission C2/C4 |
| class/class-pair Seg debt | historical n600 `~50/19/13` Road/Lane/Undrivable prior [E19] | aggregate emitter only | **MISSING VEHICLE BINDING** | candidate-local hard d_seg by class and pair at every R-D point C9 |
| shared stride-2 Lane path | `6,205/8,072` induced Road–Lane flips; one shared 16-channel skip [E19] | generic skip/deep axis, no private Lane channel | **PRESENT GENERIC / MISSING VEHICLE BINDING** | full-cell trust region and exact-byte conditional skip/deep A/B C3/C9 |
| nonlocal Seg collateral | empirical `r50/r90` support [E19] | working full-neighborhood guard in §3.2 | **DESIGN ONLY** | candidate-local collateral map through actual R at/above measured support C3/C9 |
| resize range/null capacity | about `52%` sample-specific response energy in `ker(A)` [E19] | direct scorer-plane emitter plus generic preimage solve | **PARTIAL** | conditional range/ker rows proving learned camera-null capacity is absent or worth its bytes C2/C9 |
| full-RGB luma/chroma role | source law and n600 region/annulus chroma necessity ceilings [E20] | exact Pose-block kernel plus channel-resolved ledger | **PARTIAL** | region chroma, annulus chroma, and luma exact-byte/Seg/Pose rows C4/C6/C9 |
| plane proximity / Pose force | four-anchor bindingness map [E10] | canonical registry law and objective specified in §4.3 | **PARTIAL** | built joint objective and measured coefficient/curve C4 |
| exact 2×2 Pose visibility | four lossless luma phases plus two chroma box averages [E15] | block-lattice objective/pricing in §§3.1, 4.3, 7.3 | **DESIGN ONLY** | implementation, zero-Pose chroma-kernel check, block-granular byte curve C4/C9 |
| native-f32 preimage/tie force | three f32 anchors [E14] | encode-side hard oracle; policy IDs in receiver [E4] | **PARTIAL** | tie-aware preimage A/B and builder reconciliation C0/C5 |
| exact rate force | score law and secant price [E9] | per-tensor int8+Brotli baseline [E6] | **PARTIAL** | new-section bytes, headers, exact archive reconciliation C6 |
| shortest sufficient program / no incidental state | frozen scorer sees only the sufficient statistics in [E15] | intrinsic-complexity cut and minimal grammar §5.1 | **DESIGN ONLY** | per-field deletion tests and exact receiver-closed archive C6 |
| lower-rate weight representation | block-FP rate anchor [E7] | alternate format candidate | **MISSING ADMISSION** | matched through-R Seg/Pose/bytes C7 |
| joint distortion/rate training | conditional 4.1e-4 knee [E8] | default-OFF formulation in §4.4 | **DESIGN ONLY** | implementation, OFF no-op, exact twin archives C8 |
| non-additive allocation | shared pools [E11] | one-KKT rule in §7 | **PARTIAL** | receiver-closed multi-point curves C9 |
| crash-resumable execution | binding operating contract and #537 [E13] | proven reusable apparatus | **PRESENT GENERIC / MISSING VEHICLE BINDING** | vehicle state inventory and interruption receipt C10 |
| deterministic portable decode | contest contract | existing production receiver surface [E4] | **PARTIAL** | new expander parity, full timing, exact same archive across axes C1/C2/C11 |
| certificate-to-payload causality | PDW2 target-only verdict [E5] | packet codec only | **MISSING** | mutation/no-op proof and spatial consumption C3 |
| exact promotion custody | pointer contract | none for the unbuilt vehicle | **MISSING** | C11 and explicit GO |

The completeness table is deliberately not a mechanism inventory that mistakes code presence for
scientific closure. The inverse audit is the minimal grammar in §5.1: every surviving payload
section cites one or more rows above and exact frozen lines, while every uncited field is deleted.
Thus the table tests demand→mechanism and the grammar tests mechanism→measured demand.

## 10. Drift, stores consulted, and review boundary

### Canonical drift requiring MAIN adjudication

1. `pose_plane_proximity_corollary_v1`: registry row 745 contains four measured anchors after
   `7b9a3a7d3c2`; `src/tac/canonical_equations/pose_plane_proximity_law_20260719.py` still emits
   three anchors and calls the intermediate regime unmeasured. Row 745 itself appends the measured
   intermediate anchor while retaining a contradictory `domain_of_validity.unmeasured` field.
2. `f32_receiver_arithmetic_exactness_admissibility_v1`: registry row 742 contains the third preimage-noise
   anchor after `d864b267a7`; the Python builder still documents and emits two anchors.
3. The earlier reconciled spec predates the positive-band secant/KKT landing `c6b798f146`.
4. The final #537 proof is recorded in the append-only DAG and external receipts, while an older
   tracked receipt/registry status remains blocked. This spec consumes the final proof only as a
   generic apparatus anchor [E13].

This delegated composition does not silently repair those shared authority surfaces. C0 assigns
the repair and its regression to MAIN review.

### Stores consulted

- `CLAUDE.md` and `AGENTS.md`, fully;
- `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, and
  `docs/vehicle_operating_system.md`;
- v7.5/v8 canonical vehicle specs and the reconciled/cold-start v10 specs;
- all five 2026-07-19 arm memos/receipts named in [E1]–[E11];
- frozen scorer factorization, lattice, constructive-solver, resumability, weight-factorization,
  immutable-VJP, channel/ERF/nullspace, RGB/chroma, and canonical-law sources named in
  [E12]–[E20];
- `.omx/state/canonical_equations_registry.jsonl`, `.omx/state/lane_registry.json`, and
  `.omx/state/subagent_progress.jsonl`;
- `reports/latest.md`; and
- the high-priority MAIN inbox directives dated `2026-07-19T13:03:26Z` and
  `2026-07-19T13:05:03Z`. Their frozen-line, intrinsic-complexity, exact-Pose, and weight-level
  hyperplane/basis/channel/VJP bindings are integrated in §§2–9.

The sacred run directory was treated as read-only and no bytes within it were changed.

### Triality and pointer honesty

- **DSL leg:** proposed typed vehicle/header/section IDs, `description-frame0.v1`, default-OFF
  `WitnessSelfCompressionGauge`, and explicit policy IDs. Implementation is owed.
- **DAG leg:** counted description → deterministic expand → two integer planes → two independent
  preimage solves → frozen Seg/Pose → exact bytes, with C0–C11 as the launch-predecessor chain.
- **Equations leg:** exact score law, strict conditional byte ceiling, marginal Seg/Pose byte value,
  pose constraint, factor-2 equality, and non-additive one-KKT rule.
- **Pointer delta:** zero; no candidate archive or score was produced.

### MAIN landing review is required

MAIN must review the complete base-to-head diff, the evidence/merge mapping, both canonical-law
drifts, the distinction between `repeat-frame1` timing and two-plane structural support, the
free-versus-counted grammar, the operator-native `U4`/pair-margin and immutable-VJP binding, the
channel-resolved KKT ledger, and every C0–C11 falsifier before merging. Merge of this SPEC does not
authorize C1 execution, training, paid work, or any pointer change.

---

## §ADDENDUM 2026-07-19-B — capstone proven · family closed · budget box · description-axis pivot · ROADMAP

Append-only (no section above rewritten). Durability sync of the day's landed facts; every number MEASURED
or labeled DERIVED. Pointer **0.19108 [contest-CPU] UNMOVED**. Pointers to memos; numbers inline.

- **(A) Capstone spine PROVEN** (`v10_capstone_first_byteclosed_row_20260719.md`). predictor-residual-u8.v1
  archive 409,526,925 B (sha e4cd154f…) → OFFICIAL upstream/evaluate.py n600 --device cpu seed 1234:
  **S=272.73, d_seg 1.5196e-4, d_pose 1.0184e-4, rate 272.687**. Measured TWICE bit-identical: macOS-CPU
  advisory AND [contest-CPU] Modal Linux x86_64 (fc-01KXXRAR7341QCJ6XWKV4S3QCW). Cross-host determinism of
  the C1 receiver spine PROVEN; inflate **215.8 s / 1800 s** (§6 C1 canonical-full-evaluate receipt now
  satisfied). score_claim=false — rate-dead spine, not a candidate; any compact §5 payload inherits this decode.
- **(B) Plane-storage family CLOSED** (`v10_ratecrush_phase1_20260719.md`). Exact-plane lossless storage
  RATE-DEAD at FAMILY scope: 5 codec families within 1.9× of the ~334 KB/pair floor (#541 n48) vs the box —
  ~700× over. JXL e9 best rung implied S=168.71, KEPT as donor. This retires the §5.1 "store the residual"
  reading of the intrinsic-complexity cut: the cut must buy STRUCTURE (generator + slack), not per-pixel residual.
- **(C) THE BUDGET BOX** (`generator_description_crux_synthesis_20260719.md` §0; extends §7.1 ceilings).
  S<0.19108 at capstone distortion (pose≈0) ⇒ TOTAL ≤ **264,320 B (440.5 B/pair)**; the HONEST box at the
  MEASURED spine distortions ≤ **~216,300 B (~360 B/pair)**; the box widens toward 264 KB as the residual
  distortion is recovered by **realization/target-selection** (aim the plane at the unrounded scorer reference
  per the identified-optimal preimage; NOT the pre-rounded Y) — a live axis, NOT a closed conditional. Even the
  honest 216 KB box is **not binding**: the byte-closed witness GENERATOR is already `yhat_rd_ladder` rung B =
  **83,838 B / n600 = 139.7 B/pair** (2.6× UNDER the honest box) — so REALIZATION (get the solved continuous
  target into the uint8 lattice at the counted-byte optimum), not bytes, is what the remaining work buys.
  # MAGNITUDE_DISMISSAL_OK: not a dismissal — exact budget-box arithmetic on the registered §7 score law; the
  216→264 widening is a MEASURED realization axis (identified-optimal preimage), not an eyeball claim. # FORMALIZATION_PENDING: budget-box constants are arithmetic
  on the §7 score law + seg_rate_breakeven_v1 — no new equation.
- **(D) Description-axis pivot: 9-family map, composed stack, binding blocker** (crux memo §1–2). Composed
  best-case stack **~65–130 KB TOTAL (108–217 B/pair) + unmeasured repair term** = 2–4× under the box on RATE.
  Binding blocker = a **4-order empty hole** in the measured (bytes, d_seg) curve (140 B/pair @ 3.455e-3 ↔ 1.77
  MB/pair @ 1.63e-4); for in-box descriptions (**PDW2 gauge-fixed generator-only packet 138 B partition / 142 B
  margin** = the cheapest measured, frame-195 receiver PASS; PDW1 306 B; MS-contour ~236 KB) **bytes are NOT
  binding — REALIZATION is** (frame-195 fp32 authority; absent receiver grammar). **PDW2's OPEN half is the SAME
  receiver problem as STEP 2's d_B-attack:** its verdict is `TARGET_ONLY_VS_REALIZATION_NON_EQUIVALENT`
  (`pdw2_gauge_packet_probe_20260719_receipt.json`, `through_r_authority: false`) — the 138 B gauge-fixed
  coefficients need a receiver that expands them into the spatial partition (a channel feature field) and is
  through-R equivalent (consumption discipline #417). So PDW2 (rate frontier, ~1/3 of PDW1's raw) and the STEP-2
  realization CLOSE TOGETHER through one shared receiver. Non-additive pools: elements must enter as
  seeds/conditioning INSIDE the joint solve (two post-hoc composition failures forbid composition-after).
- **(E) Survey adoptions folded into the §5/§C description ladder** (`generator_description_online_survey_20260719.md`).
  ADOPT: Aurenhammer min-generator LP · kinetic regular-triangulation event grammar (~2-3 B/gen/frame, temporal)
  · geogram BSD SDOT + voro++ cross-check · Schuster-Katsaggelos operational-R-D DP (retargeted to d_seg) ·
  Rissanen MDL stop-rule · margin-aware flip-slack weight quantization (OURS). TEST: ξ-keyed temporal delta w/
  MPEG-4 INTER-CAE ~40-50% (#574) · Apollonius/anisotropic cells ε^(-1/2)→ε^(-1/3) · SAD fitter. Originality
  double-confirmed (NO-FAKE #7): no era's codec is task-lossy vs a frozen known argmax.

### The M1/M2/M3 measurement ladder (owners)

- **M1** — C2 banded-generator n600 row (owner: operator-GO launch; tooling `integer_plane_emitter.py` built,
  228 tests + margin-band law + #543 receiver + `levelset_byte_close_and_eval.py`) → first mid-curve point.
- **M2** — preimage/target-selection A/B (the REALIZATION axis). The narrow tie-aware arm (eb51dab964) is exact
  for the pre-rounded-Y target only (measured NO-OP against Y). The LIVE M2 is **aim the plane at the unrounded
  scorer reference** (identified-optimal preimage: `resize_null_preimage` #49/S12 min-description + bounded-uint8
  Diophantine feasibility #532/#547) → ~0.047 S recovery at ZERO payload bytes, widening toward the 264 KB box.
- **M3** — stratum-seed-INSIDE-the-solve A/B (Road-Lane generator swap on n24, Wave-F coder landed + #549
  solver) → composed-stack admissibility against the 0.2503 B/pair-per-1e-6 break-even (does pool-competition
  eat it?).

### ROADMAP TO S<0.19108 (and toward sub-0.15) — the causal chain, gates, owners

- **STEP 0 (DONE, proven):** C1 receiver spine — official evaluator loop closed, cross-host bit-identical,
  inflate 215.8s/1800s. Any compact payload inherits this decode path.
- **STEP 1 (M2 — RE-SCOPED; the NO-OP conclusion was a mis-scoped binary, operator 2026-07-19):** the M2 arm
  (eb51dab964) MEASURED one narrow true thing — the canonical factor-2 support-fill preimage reproduces the
  **already-chosen** plane `Y = round(exact_resize)` exactly (‖A_fp32(canonical)−Y‖ = 0 across 117.96M n600
  values; tie-aware is bit-identical GIVEN that target). But its conclusion — "preimage selection is a NO-OP,
  box closed" — is a **binary collapse of the realization axis** and does NOT hold: the 0.047 S it attributes
  to "plane-quantization" IS the preimage/**target-selection** axis (M2 aimed at the pre-rounded `Y`). The
  **extensive prior preimage corpus identifies the optimal** and it is NOT the pre-rounded-Y target:
  (a) `resize_null_preimage_compiler` (#49/S12) — 80.67% resize nullity, min-**description** preimage, optimal
  fill chosen BY MEASUREMENT (a 0-byte RATE lever, not a noise-kill); (b) `bounded_uint8_resize_preimage_cell_feasibility_v1`
  (#532/#547) — exact uint8 realization is a 4-var Diophantine feasibility whose HARD_ACCEPT depends on the
  **decoded-uint8 argmax**, so preimage choice IS a d_seg lever at tie-tight cells; (c) `yhat_rd_ladder` (#548,
  rung B) — the byte-closed witness generator is **83,838 B / n600 = 139.7 B/pair**, 2.6× UNDER the honest box.
  **Correct reading:** aim the plane at the **unrounded scorer reference** (the identified-optimal target), not
  at pre-rounded `Y` → recovers the 0.047 as a target-selection/realization move; and emit the min-description
  preimage as the rate lever. **The box question stays OPEN** and is pursued via the identified-optimal preimage
  (STEP 2 realization + STEP 3 generator), not declared closed. verdict_scope: the M2 measurement is exact for
  the pre-rounded-Y target ONLY; the realization/target-selection family is LIVE. [magnitude-ok — the correction
  is a scope/target reconciliation against the measured prior corpus, not an eyeball dismissal of M2's numbers]
- **STEP 2 (PDW1 realization — PARTLY MEASURED, `pdw1_fp32_realization_receipt_20260719.json`):** frame-195
  authority **CLOSED** (Phase B: fp32 first-max contract `pdw1-native-f32-power-first-max.v1` resolves the exact
  class-0/1 tie at px(195,112,214) to class 0 == L*). Phase C **named the dominating error term**: `d_A = 0.0`
  (contract labels ARE exactly L* — encoding/bytes perfect) but `d_B = 0.008069` (38,077 px) = the **REALIZATION
  gap** — the realized uint8 frame re-scored through the real R+SegNet flips argmax (Road→Lane 28.1%,
  Movable→Road 16%). So the 23.8×-over-need d_seg is 100% realization, 0% encoding — the empirical confirmation
  of "realization not bytes is binding." **The path to in-box is now sharp:** apply the identified-optimal
  preimage (`bounded_uint8_resize_preimage_cell_feasibility_v1` HARD_ACCEPT — choose the uint8 cell whose
  *re-scored* argmax == L*) to drive `d_B → 0`. **Gate:** a point at ≤477.8 B/pair with d_seg ≤3.39e-4
  (in-box), via the d_B-attack arm (dispatched, parallel with M1). [advisory macOS-CPU; pointer UNMOVED]
- **STEP 3 (live, M1, operator GO):** first C2 banded-around-source generator n600 row → fills the 4-order
  (bytes,d_seg) hole from the cheap end. Target zone **60–85 KB total at d_seg ≤~1e-3**. **Gate:** byte-closed
  row through the production receiver + hard oracle.
- **STEP 4 (designed, #574):** ξ-keyed temporal amortization — kinetic event grammar (~2-3 B/gen/frame) or
  INTER-CAE fallback per class — multiplies whichever single-frame point wins (published precedent 40-50%).
  **Gate:** fires on the Step-2/3 point landing.
- **STEP 5 (the composed row):** assemble the stack INSIDE the joint solve (two measured post-hoc failures
  forbid composition-after) — banded generator base + Road-Lane seed-in-solve + PDW2 head packet (133 B) +
  tie-aware preimage + KKT band-slack repair — measured as ONE byte-closed row through upstream/evaluate.py.
  **Projection** from the synthesis: 65–130 KB total ⇒ rate term 0.043–0.087 ⇒ **S ≈ 0.06–0.14 IF distortion
  holds at capstone grade** — i.e. the composed row is a candidate not just for sub-0.19108 but for SUB-0.15
  DIRECTLY. This is a **projection, not a claim** (NO-FAKE #8), under two explicit conditionals: (1) distortion
  holds through composition; (2) realization closes (frame-195 authority + receiver grammar).
- **STEP 6 (authority):** dual-axis exact eval on the composed archive → pointer move.

**Failure-routing:** if STEP 2's d_seg blows out on curved Road-Lane → Apollonius/anisotropic cells (survey
P4); if STEP 3 misses the target zone → the dominating-byte-term diagnosis (crux §2, dense residual vs frame-0
bootstrap) routes the next unit.

**Today's honest position:** pointer 0.19108 UNMOVED; everything above is the measured path, with three of six
steps live.

## §ADDENDUM 2026-07-20-W — THE WORLDSHEET OBJECT (operator-directed design + philosophy elevation)

**Provenance:** operator 2026-07-20 ("World sheet object should be added to our v10 capstone design doc and
philosophy"), from the Time-Traveler doctrine-gap convening (G1, `.omx/research/time_traveler_doctrine_gaps_20260720.md`)
— itself a recognition-and-binding of corpus we already hold: the Chasles screw engine (`tac.lie`), the ξ-keyed
temporal coder design (#574), the measured flicker mechanism, the phase-advection lever (#424).

### The object

The scored video is not 600 independent pairs. The SegNet separatrix (the codim-1 argmax boundary) exists ONCE
as a **2+1-dimensional worldsheet** W ⊂ (x, y, t): the 1-D boundary curves of each frame are time-slices of one
ruled surface swept by the ego-screw ξ(t). Formally, W is (to first order) the orbit of the t=0 separatrix under
the one-parameter screw family — a ruled/developable surface in spacetime, with deviations from exact
ξ-transport concentrated at scene events (occlusions, birth/death of dashes, specular stress).

### Design consequences (each with its measured anchor)

1. **SOLVE dimension, not just coder dimension.** The interaction inversion (steer-4 components) extends to
   SPACETIME components: adjacency = spatial support-overlap ∪ temporal ξ-transport adjacency. One worldsheet
   solve replaces up to 600 per-slice solves; flip debt clusters along worldsheet geodesics (the flicker
   decomposition — 90.6% edge-flicker — IS this fact measured as a defect). Anchor: r2b margin histogram +
   R1 boundary concentration (77.5% within 1px of the slice curve).
2. **DESCRIPTION dimension (the rate half, elevates #574).** The minimal description of 600 boundary
   configurations is one curve description + the (already-stored or derived) ξ trajectory + a sparse ledger of
   deviation EVENTS where transport fails. Expected shape: keyframe curve + per-slice residuals that are
   near-zero away from events — the MPEG-INTER structure applied to the separatrix itself. Per the tiebreak law
   this is also the LEAST-COMPLEX carrier: one object instead of 600.
3. **POSE UNIFICATION.** ξ is dual-use by construction (Chasles): the same screw that transports the worldsheet
   IS the pose observable (dim-0 dominant). On the worldsheet there is no separate pose problem — pose is the
   transport field of the seg object. The seg-conditioned-pose ordering law becomes geometric: the sheet must be
   correct before its transport field is read.
4. **COMPLEXITY.** The worldsheet is intrinsically low-dimensional: per-curve ~8-dim lane manifold × 1-dim time,
   with events sparse. Intrinsic-complexity ops (steer 6) on W: banded along curve-parameter, banded along t,
   per-event local. Ambient-sized ops on W are structurally forbidden.

### Philosophy (the eightfold/master-thesis fold)

The worldsheet is the master thesis applied to TIME: the frozen space contains one spacetime object; slicing it
into 600 problems was a coordinate choice, not a fact. "It all falls out" — seg boundary, pose transport,
temporal coding, flicker cure, and rate floor are FACETS of W, exactly as the level-set doctrine's facets were
of the witness. The joint solve on W is beautiful because it is cheaper (one object), and cheaper because it is
the right object (least-complex tiebreak).

### Binding plan (no new build wave; sequenced)

- NOW (r1b5 line): unchanged — per-pair row first; the worldsheet does not gate the first candidate row.
- POST-ROW: #574 elevated from coder to worldsheet solve+description (queue #582(e)/G1); spacetime component
  partition A/B vs per-pair components (verdict test: worldsheet carrier bytes vs 600× per-pair carrier at
  equal d_seg); event-ledger schema shares the #315/#344 event vocabulary.
- Equations leg owed at binding time: worldsheet transport law (ξ-advection of the separatrix + measured
  deviation-event rate) as a canonical equation with a real evaluator.
