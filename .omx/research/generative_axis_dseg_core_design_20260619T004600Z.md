---
title: GENERATIVE/ALGORITHMIC d_seg-core design + feasibility-gate spec — store the GENERATOR, replay to reconstruct
authority: "[contest-CPU advisory] — NON-PROMOTABLE; exact pointer UNMOVED at 0.19110"
score_claim: false
promotable: false
frontier_pointer_moved: false
mission_contribution: frontier_breaking_enabler
date: 2026-06-19
status: design_plus_gate_spec
verdict_of_design: BUILD_NCA_DSEG_CORE_GATE_FIRST
cross_refs:
  - .omx/research/campaign_inflection_three_paths_capped_concentrated_saliency_20260618.md
  - .omx/research/factored_lf_core_capacity_gate_20260618T233940Z.md
  - .omx/research/campaign_math_review_dynamics_and_optimization_20260618.md
  - experiments/probe_curve_core_dseg_feasibility_gate.py
  - experiments/probe_nca_dseg_feasibility_gate.py
---

# Generative/algorithmic d_seg-core — design + the $0 feasibility gate

**The operator's reframe: store the GENERATOR (process + seed), replay to reconstruct.** Every prior
family in the sub-0.15 campaign stored a STATIC description (decoder weights / polygon control points /
partition stores) and hit the capacity↔rate wall — a wall on *static descriptions*. The generative axis
is the ONE representational family the campaign has not tested. It is the Kolmogorov/MDL-optimal axis
(Schmidhuber, grand-council seat): **the shortest description of a complex pattern is the shortest
PROGRAM that outputs it.** Iteration generates detail for free — a few-KB rule, run N steps, can grow
structure whose static description would need many KB.

All numbers in this design + its gate are `[contest-CPU advisory]` NON-PROMOTABLE; the exact pointer is
UNMOVED at **0.19110**. This unit does not move the pointer — it is a measured GREEN/RED that re-routes
the sub-0.15 search.

## 0. What the two prior geometry gates established (the design constraints)

The generative axis must beat BOTH walls the static families hit. The exact numbers (measured, this
campaign):

| Family | mechanism | best d_seg | best S | wall |
|---|---|---:|---:|---|
| factored RANK-1 LF (learned pixel decoder) | small narrow-channel HNeRV trained 100% on d_seg | 0.0169 (bc12) | 1.79 | **CAPACITY**: d_seg ~ 29.3·params^−0.71; sub-0.15 d_seg needs ~4M params (forfeits rate) |
| curve-core (static-stored geometry) | decimated boundary polygons, colours fit through SegNet+roundtrip | **0.0071 realized** (mp32, geo_recon 0.0033) | 0.77 | **SURVIVAL**: geometry matches L* (geo_recon→0.002) but realized d_seg plateaus ~0.007 — the bilinear-874→384 downsample mixes the 1px boundary band → SegNet argmax flips **17-26% of boundary px** regardless of geometry |

**The decisive curve-gate fact (the survival wall, read directly off the live gate_state):** as control
points rise (mp8→mp64), `geometric_dseg_recon` falls monotonically (0.0575→0.0019) — geometry CAN match
L* — but `realized_dseg` PLATEAUS at ~0.007 and `boundary_band_flip` stays stuck at 0.17-0.26. The
survival wall is a property of **the eval roundtrip + the SegNet decision boundary**, NOT of the
representation. So the generative axis faces an unavoidable design split:

- **Survival is NOT escapable by changing the representation.** Any vehicle that renders an RGB frame,
  passes it through the *exact same* camera-res→384 bilinear roundtrip and the *exact same* frozen
  SegNet, faces the *identical* boundary-band mixing. An NCA frame is just as exposed as a polygon frame.
- **THEREFORE the generative axis's only candidate escape is the BYTE/CAPACITY axis** (a few-KB shared
  rule that grows the partition cheaply), PLUS whatever pre-compensation the differentiable
  through-roundtrip fit can buy at the boundary band (the same lever the curve core already has and which
  only reached 0.007). The honest hypothesis the gate tests is therefore narrow and falsifiable.

## 1. The generative-axis mechanisms, ranked for OUR exact objective

Objective: minimize realized d_seg (binding, ∂S/∂d_seg=100) + held d_pose (0.00034) at byte-cheap rate
(∂S/∂B=6.66e-7/byte), surviving the eval roundtrip, replayable in inflate.sh ≤30min CPU.

### Rank 1 (BUILD): Neural Cellular Automata (NCA) d_seg-core
Mordvintsev "Growing NCA". A few-KB local conv update rule (perception conv + tiny MLP — the ONLY stored
params), iterated N steps from a tiny seed, GROWS a per-class partition / decision-band frame. Rendered →
exact roundtrip → real SegNet → realized d_seg vs L*.

- **Byte model:** the rule (a 3×3 depthwise perception × C channels + a 1×1 conv hidden→C, no bias) is
  the only stored weight. For C channels, hidden H: params ≈ C·3·3 (perception, fixed identity+Sobel or
  learned) + C·H + H·C (the MLP) ≈ C·H·2. At C=12,H=48 → ~1.2K params; C=16,H=96 → ~3.1K; C=24,H=128 →
  ~6.4K. At int8 that is **1.2-6.4 KB**. The seed is a tiny init grid (a few classes × a coarse grid,
  ~hundreds of bytes). Amortized over 600 frames the per-frame delta is a tiny ego-motion warp of the
  seed (the contour barely moves — geometric-solve identity residual 0.33px). **This is the cheap-bytes
  bet, and it is structurally MUCH cheaper than the 270-650 control points the curve core needed.**
- **The capacity-escape claim (why it could beat params^−0.71):** temporal weight-SHARING across
  iteration steps. An N-step NCA has effective depth N with the SAME rule params. A one-shot learned
  decoder of P params can represent functions of bounded complexity ∝ P; an N-step NCA of P params can
  represent functions of complexity ∝ P·(growth from iteration) — the iteration is a free depth
  multiplier. IF the SegNet decision boundary is generable by local growth (it is largely a smooth
  road↔lane↔other partition with locally-coherent structure), the NCA's few-KB rule could reach the same
  d_seg a many-KB static description needs. This is the EXACT bet the static families could not make.
- **The survival story (honest):** the NCA grows a partition that is then rendered and roundtripped. It
  inherits the survival wall. Its ONLY survival lever is the same as the curve core: fit the grown
  frame's per-class colours / boundary band THROUGH the roundtrip so gradients pre-compensate mixing. We
  do NOT expect the NCA to beat the curve core's 0.007 survival floor on the boundary band — we expect it
  to (a) reach comparable realized d_seg at FAR fewer bytes (the capacity escape), and we test whether
  that combination crosses S<0.15. If realized d_seg is survival-floored at ~0.007, S~0.77 even at near-
  zero bytes (100·0.007 = 0.70), so the gate will RED on survival even with the byte win — which is
  itself the airtight finding (survival, not capacity, is the terminal wall).
- **Inflate-time replay cost:** N conv iterations over a 384×512×C grid on CPU. At C=16, N=64, that is
  ~64 × (384·512·16 × 3×3 perception + MLP) ≈ a few hundred MFLOP × 600 frames — trivially ≤30min CPU.
  Numpy-portable (conv2d + a tiny matmul). The generator IS a tiny program — the openpilot-friendly,
  Kolmogorov-cheap representation.

### Rank 2 (contingency): A*/Dykstra geodesic / active-contour boundary fixed-point
Store a cheap cost field (downsampled SegNet margin / edge map, a few hundred bytes) + recover the
boundary as a minimal-path / level-set fixed point at inflate. Extends the legal-frame Dykstra solve
(#73) from one-shot to iterated. **Why ranked below NCA:** it still produces a *static* boundary
(geometry), so it inherits the curve core's survival wall directly with no new pre-compensation lever,
and it does NOT have the free-detail-via-iteration capacity escape (the iteration converges to a fixed
boundary, it does not GROW unbounded detail). It is a cheaper *encoder* of the same static geometry the
curve core already showed walls at 0.007. Useful only if the NCA gate shows the byte win is real and we
want an even cheaper static fallback.

### Rank 3 (temporal complement, near-free): Brownian/projective ego-path replay
Store ONE keyframe partition + the ~1-DOF ego trajectory; per-frame boundary = deterministic projective
warp of the keyframe. This is the TEMPORAL amortization (it makes the per-frame delta near-zero) and is
ORTHOGONAL to the per-frame d_seg-core question this gate answers. It is already implicitly in every
family's amortization model (the 0.10 per-frame-delta factor). It does not by itself lower the per-frame
realized d_seg, so it is not the family the gate tests; it is a multiplier to apply to whichever per-
frame core wins. Folded in as the amortization model, not a separate gate.

**Design verdict: BUILD THE NCA d_seg-CORE GATE.** It is the only mechanism with a genuine capacity
escape (iteration = free depth) AND a survival lever (through-roundtrip colour/band fit), and it is the
operator's headline candidate. The gate measures whether the capacity escape is real AND whether the
combination crosses S<0.15 or hits the survival wall the curve core hit.

## 2. The gate (the decisive $0 measurement)

`experiments/probe_nca_dseg_feasibility_gate.py`, REUSING the curve harness's exact roundtrip
(`_eval_roundtrip_t`), realized-d_seg measurement (`_segnet_argmax_of_frame`), GT load (gt_targets_n16),
byte model (`curve_param_bytes`-analog with NCA param counts), and GREEN/RED/AMBER verdict structure.
The NCA is the swapped representation; everything downstream of "render a frame" is identical to the
curve gate so the two are apples-to-apples.

**Mechanism.** A tiny shared NCA rule (the stored bytes) over a C-channel grid, seeded from a coarse
class-init, iterated N steps to produce per-class logits → softmax → differentiable per-class-colour
render → exact roundtrip (STE round) → real SegNet → CE vs L*. The rule weights + per-class colours +
boundary-band offset are trained THROUGH the chain on MPS fp32 gradient; the realized d_seg of the HARD
(argmax-decoded, then through the no-STE roundtrip) frame through the CPU-authority SegNet IS the verdict.

**Sweep RULE-SIZE** (the generative analog of curve-complexity): hidden width H ∈ {32, 64, 128, 256}
(optionally also C and N), giving rule-param counts spanning ~1KB→~30KB at int8. Report per size:
rule_param_count, amortized rate (rule bytes once + tiny per-frame seed/delta, entropy-coded, amortized
over 600 — the same honest model as the curve gate, documented), the 3 separated d_seg numbers
(geo-analog if meaningful, geometric_dseg no-roundtrip, **realized_dseg through-roundtrip = AUTHORITY**),
S_projected (100·realized + √(10·HELD_POSE) + 25·rate/B0), elapsed.

**Verdict.** GREEN iff some rule-size has realized d_seg < ~0.0012 AND rate < 0.05 AND S < 0.15. RED
otherwise, with wall-diagnosis: CAPACITY (the NCA rule cannot grow a partition matching L* even with no
roundtrip → geometric_dseg high) vs SURVIVAL (geometric matches but realized >> geometric, the boundary-
band mixing) vs INTRINSIC-FLOOR (both low but S high on bytes). MEASUREMENT-FIRST: driven by realized-
through-the-chain d_seg, NEVER the CE training loss (the factored-LF degenerate-fit false-GREEN is the
cautionary anchor — guard against false-GREEN AND false-RED). Recompute S from components in the JSON.

**MVP-first scope** (per directive): dominant class-pair first option, n_frames small (3), modest N
(48-64) and iters — a feasibility ESTIMATE, not a final train. Detached nohup resumable daemon with
per-size JSON checkpoint (`experiments/results/nca_dseg_feasibility_gate/gate_state.json`, incremental +
`final_verdict`). `python -u`.

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| eval roundtrip (`_eval_roundtrip_t`) | **ADOPT_CANONICAL** (reuse curve gate verbatim) | The exact contest roundtrip is fixed; survival apples-to-apples with the curve gate REQUIRES identical roundtrip. |
| realized-d_seg measurement (`_segnet_argmax_of_frame` + flip-rate vs L*) | **ADOPT_CANONICAL** (reuse) | The d_seg metric is the exact contest functional; reusing guarantees the survival comparison is honest. |
| GT load + L* targets (gt_targets_n16 `seg` field) | **ADOPT_CANONICAL** (reuse) | Same GT, same L*, same per-class mu colours as the curve gate. |
| byte model (rule params int8 + seed + amortization) | **FORK_PRINCIPLED** | NCA bytes = rule weights, NOT control points; the byte driver differs structurally, but the amortization factor (0.10 per-frame delta, /600) is ADOPTED from the curve gate for apples-to-apples rate. |
| the representation (NCA rule + iteration) | **FORK** (the whole point) | This IS the unique mechanism under test; nothing canonical exists. |
| verdict logic (GREEN/RED/AMBER + wall_diagnosis) | **ADOPT_CANONICAL** (mirror curve gate) | Same thresholds (GREEN_DSEG 0.0012, byte-cheap 0.05, S 0.15), same measurement-first basis, same false-GREEN guard. |
| optimizer (AdamW through chain, MPS fp32 + CPU authority) | **ADOPT_CANONICAL** | The train/authority split is the canonical MPS discipline. |

## Observability surface

- **Inspectable per layer:** the gate logs per-frame, per-rule-size: rule_param_count, geometric_dseg
  (no-roundtrip), realized_dseg (through-roundtrip), boundary_band_flip, interior_flip, CE first/last,
  bytes breakdown, rate, S — so the capacity-vs-survival decomposition is directly readable.
- **Decomposable per signal:** S is recomputed from its 3 components (100·realized + √(10·pose) +
  25·B/B0) in every row; the survival_gap (realized − geometric) isolates the survival contribution; the
  boundary vs interior flip split isolates WHERE the d_seg lives.
- **Diff-able across runs:** per-size JSON checkpoint + the deterministic seed make two runs comparable;
  the same-grid comparison to the curve gate's gate_state is the cross-family diff.
- **Queryable post-hoc:** the result JSON (`.omx/research/nca_dseg_feasibility_gate_<UTC>.json`) carries
  all rows + thresholds + verdict + wall_diagnosis, machine-readable.
- **Cite-able:** every number anchored to the producer path + the GT cache + the frozen SegNet + the
  rule-size; axis_tag `[contest-CPU advisory]`, score_claim=false, pointer_moved=false.
- **Counterfactual-able:** the rule-size sweep IS the counterfactual ("what if the rule had H params?");
  the geometric-vs-realized split answers "what if there were no roundtrip?".

## Master-gradient grounding

Per the math review: ∂S/∂d_seg=100 (binding, linear), ∂S/∂d_pose=85.8 at the operating point (pose held
on the trunk at 0.00034, NOT chased — the NCA targets seg only), ∂S/∂B=6.66e-7/byte (bulk, tiny). The
NCA bet is precisely a capacity↔rate decoupling: it spends ~1-6 KB of bytes (rate ~0.001, negligible vs
the 0.05 byte-cheap bar) to buy d_seg via iteration. The gate measures whether the 100·d_seg term can be
driven below ~0.07 (realized d_seg < 0.0007) at that near-zero rate. If realized d_seg is survival-
floored at ~0.007 (the curve core's number), 100·d_seg = 0.70 dominates and no byte win can rescue S —
the airtight terminal finding that survival, not capacity, is the wall across ALL families.
