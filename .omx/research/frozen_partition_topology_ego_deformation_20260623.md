---
title: "Frozen-instance partition TOPOLOGY + ego boundary-deformation — is the d_seg-optimal code topological? VERDICT: LIMITED"
authority: "[contest-CPU advisory] NON-PROMOTABLE — pointer UNMOVED 0.19110; $0; CPU-only, NEVER MPS; no PR; no exact-eval dispatch"
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
pointer_moved: false
date: 2026-06-23
verdict: LIMITED_TOPOLOGY_DECOMPOSES_BUT_NO_VIABLE_TOPOLOGICAL_DSEG_CODE
subagent: frozen-topology-measure-20260623
cross_refs:
  - .omx/research/frozen_instance_horizon_crossframe_result_20260623.md   # a3061: flip-RESIDUAL full-rank 547/600 (DIFFERENT object)
  - .omx/research/partition_store_realization_gate_DEFER_20260617T024639Z.md  # the boundary-survival wall (24% flip)
  - .omx/research/boundary_math_seg_core_20260610T101618Z.md   # #52: static partition ~896 B/frame, ~36 components/frame, d_seg=0 lossless
  - .omx/research/nonrgb_capstone_reopen_verdict_20260623.md   # the L13 72 KB witness / rate-vs-d_seg framing
  - experiments/results/indep_dseg_bets_20260623_inflated/seg_argmaps.npz   # EXACT frozen-SegNet argmax cache (authority-faithful)
  - upstream/modules.py   # frozen PoseNet 6-dim d_pose target
  - upstream/frame_utils.py   # yuv420_to_rgb (GT decode; NEVER PyAV rgb24)
tool: experiments/probe_frozen_partition_topology.py
result_json: reports/frozen_partition_topology.json
tests: src/tac/tests/test_frozen_partition_topology.py   # 7 NO-FAKE, all green
---

# Frozen-instance partition topology + ego boundary-deformation — reopen measurement

**TL;DR.** The operator's topological decomposition is **REAL at the coarse level but does NOT
yield a viable frozen-instance d_seg code.** Measured on all 600 scored frames (the EXACT
frozen-SegNet argmax cache, authority-faithful to 1e-7):

1. **Coarse topology IS near-constant.** The large scene regions (classes 0/2/3/4 = 99.3% of the
   pixel mass) carry ~1–4 connected components each with low variance; the class-level adjacency
   graph is the modal graph in 35% of frames (only 11 distinct graphs over 600). The dominant
   boundary CURVE (per-column horizon-type boundary) is genuinely **low-dimensional: effective
   rank 4.07** (participation ratio), top-1 mode 46% of variance, k=2 modes for 50% / k=6 for 80%.
2. **BUT the deformation is NOT ego-driven, and the FINE structure is content-noise.** The real
   frozen PoseNet 6-dim ego-motion (the d_pose target) explains only **R²=0.23** of the boundary
   deformation (R²=0.38 with cumulative-trajectory regressors); the horizon-row proxy v_h(t)
   explains **R²=0.034** and correlates with PoseNet at max |r|=0.29. The fine structure (~31
   small islands/frame, 0.72% of pixels) has **effective rank 52.9/60** — content-noise, EXACTLY
   like a3061's flip-residual (547/600), NOT an ego-deformation.
3. **A constant-topology code pays d_seg ≈ 0.0071** (12.6× the frontier d_seg 0.00056) — measured
   as the d_seg of dropping the fine islands. That is the SAME order as the partition-store survival
   wall (0.0064) and WORSE than the frontier. The cheap low-dim coarse code does not realize a
   competitive d_seg; the expensive fine islands that WOULD are full-rank content-noise.

**VERDICT: LIMITED.** The topological reframe decomposes the partition cleanly but does **not**
help beyond what #52 (static partition, d_seg=0 lossless costs ~524 KB) and a3061 (flip set is
full-rank, not ego-low-dim) already found. The frozen-instance-optimal d_seg code is NOT a cheap
topological template+ego-deformation; the d_seg signal lives in content-dependent fine structure
that is neither cheaply storable (a3061/#52) nor ego-parameterizable (this probe). **It belongs in
TRAINING (the live generator d_seg campaign), not a $0 topological sidecar.**

All numbers `[contest-CPU advisory]` NON-PROMOTABLE. Pointer UNMOVED 0.19110. $0, CPU-only, NEVER
MPS, no GPU, no PR, no exact-eval dispatch.

---

## What this measures (and how it differs from the prior NO-GOs)

The operator's claim: GT seg partition = (near-CONSTANT topology) + (LOW-DIM ego-driven boundary
deformation) → if true, the d_seg code is topological (constant template + ego-deformation +
rendered boundary), unifying with d_pose via the shared ego-motion.

This is a **DIFFERENT object** from the two prior NO-GOs, which is why it had to be measured:
- **a3061** (`frozen_instance_horizon_crossframe_result`) measured the **flip RESIDUAL** — the
  per-frame *correction* set (where comp argmax ≠ GT argmax) — and found it full-rank (547/600),
  not ego-low-dim. That is the *error*, not the GT partition.
- **#52** (`boundary_math_seg_core`) measured the **static per-frame** partition byte cost (~896
  B/frame LZMA, ~36 components/frame, d_seg=0 lossless) — no cross-frame / ego structure tested.
- **This probe** measures the **GT partition TOPOLOGY itself** (adjacency + component counts +
  Euler) across all 600 frames, its **boundary-CURVE deformation** intrinsic dimension (the GT
  boundary, not the residual), and its **ego-explained fraction** (real PoseNet + horizon proxy).

Authority: the cached `gt` argmaps are the EXACT frozen-SegNet argmax (validated d_seg 0.00055989
vs report 0.00055978, Δ=1e-7 — exact-scorer faithful). PoseNet ego = the real frozen 6-dim
d_pose-target output on GT pairs (non-overlapping seq_len=2, evaluate.py-faithful), CPU-only.

---

## Result 1 — partition TOPOLOGY: coarse near-constant, fine volatile (600 frames)

| measurement | value | reading |
|---|---|---|
| distinct class-adjacency graphs | **11 / 600** | the class-touch graph is nearly fixed |
| modal adjacency-graph frequency | **34.8%** | one graph dominates (5 classes mostly mutually border) |
| components / class (mean ± std) | 0:2.1±1.7 · 1:**27.6±4.3** · 2:1.1±0.4 · 3:3.7±1.5 · 4:1.0±0.0 | classes 0/2/3/4 ≈ constant; **class 1 is the volatile one** |
| total components / frame | **35.5 ± 4.9** | matches #52's ~36 |
| Euler characteristic (mean ± std) | **14.0 ± 3.5** | a true topology invariant (components − holes); varies via the islands |
| distinct FULL signatures (adj+comp+euler) | **573 / 600** | the *full* topology is distinct nearly every frame — driven by class-1 island count |
| pixel mass in small (<500 px) components | **0.72%** | the volatile islands are a tiny fraction of pixels |
| small components / frame | **31.1** | ~all of the 36 components are small islands |

**Reading:** the partition is **two-scale**. The COARSE topology (the big regions: road, sky,
undrivable, lane = 99.3% of mass) is near-constant (1–4 components, low variance, stable adjacency).
The FINE topology (≈31 small islands/frame, 0.72% of mass, mostly class 1) is volatile — it makes
the full signature distinct in 573/600 frames. The operator's "near-constant topology" hypothesis
holds **for the coarse scene** but **fails for the full partition** because of the fine islands.

## Result 2 — boundary DEFORMATION: the coarse curve IS low-dim (DECISIVE for the cheap half)

Per-column boundary-curve representation (upper scene boundary + road-top horizon per column),
SVD across 600 frames:

| measurement | value | reading |
|---|---|---|
| **boundary effective rank (participation ratio)** | **4.07 / 600** | **LOW-DIM** — the dominant boundary lives in ~4 modes |
| top-1 singular var share | **46.1%** | one dominant deformation mode |
| top-3 / top-6 var share | 67.2% / 82.7% | a handful of modes capture most motion |
| k for 50% / 80% / 90% / 95% var | **2 / 6 / 11 / 21** | low-dim subspace (cf. a3061 flip-residual 502 for 90%) |
| horizon-row v_h(t) mean ± std | 286.4 ± 0.70 rows | thin band — smooth trajectory (a3061's v_h ≈ 194 was the road-top; this curve uses a different boundary convention but same low-D conclusion) |

**This is the genuinely new finding:** unlike a3061's flip-residual (eff. rank 547, no low-D
subspace), the GT boundary CURVE is low-dimensional (eff. rank 4). The coarse scene boundary moves
in a few modes. The question the next stage settles: are those modes the EGO-motion?

## Result 3 — ego unification: the low-dim deformation is NOT primarily ego-driven

Linear regression of the boundary deformation on the ego regressors (R² = explained variance):

| ego regressor set | R² explained | reading |
|---|---:|---|
| horizon-row proxy v_h(t) + dv_h/dt | **0.034** | the horizon-row trajectory explains ~nothing |
| **real PoseNet 6-dim ego-motion** (d_pose target) | **0.231** | ego explains <¼ of the boundary deformation |
| PoseNet 6-dim CUMULATIVE (trajectory) | **0.379** | even integrated ego explains <⅖ |
| max |corr(v_h, PoseNet dim)| | **0.288** | the horizon row weakly couples to one pose dim |
| PoseNet pose std per dim | [1.256, 0.036, 0.030, 0.010, 0.007, 0.029] | dim 0 (forward/yaw-like) dominates the ego signal |

**Reading:** the boundary deformation is low-dim but its modes are **content-driven, not
ego-driven**. The shared-ego-code unification the hypothesis needs (one ego trajectory parameterizes
BOTH d_seg-boundary AND d_pose) **does not hold**: 62–77% of the boundary motion is unexplained by
the actual 6-dim ego-motion. The d_seg boundary and the d_pose target do not collapse to one cheap
ego code.

## Result 4 — the fine islands are content-noise (the d_seg-binding half), like a3061

| measurement | value | reading |
|---|---:|---|
| island-mask effective rank (60-frame SVD) | **52.9 / 60** | **near-full-rank** — island locations are ~independent draws |
| island-mask top-1 var share | 7.1% | no dominant shared spatial mode |
| **d_seg cost of dropping the fine islands** (coarse partition vs GT) | **0.00705** | **12.6× the frontier d_seg (0.00056)** |

**This is the decisive negative.** A topological code that stores the constant coarse template + the
low-dim ego/boundary deformation but NOT the per-frame islands realizes d_seg ≈ 0.0071 — worse than
the frontier and the same order as the partition-store survival wall (0.0064). The islands that
WOULD fix it are full-rank content-noise (eff. rank 52.9/60) — exactly a3061's flip-residual class —
so they cannot be ego-coded or template-coded cheaply.

## Result 5 — byte / S projection (via tac.contest_score)

| code | bytes | vs L13 witness (72,217 B) | vs frontier (177,169 B) |
|---|---:|---:|---:|
| constant template (#52, once) | 896 | — | — |
| low-dim boundary deformation (k=21 coeffs, Δ+zlib, 600 frames) | 11,801 | — | — |
| **topological low-rank total** | **12,697** | 0.176× | 0.072× |
| lossless full partition (#52 extrapolation) | 537,600 | 7.4× | 3.0× |

**The byte side LOOKS attractive** (the coarse topological code is ~12.7 KB, 0.07× the frontier) —
BUT it is **byte-cheap precisely because it is d_seg-lossy** (it drops the islands → d_seg 0.0071).
Plugging into `tac.contest_score`:
- Topological coarse code at its own realized d_seg: S = 100·0.0071 + √(10·d_pose) + 25·12,697/N ≈
  **0.71 + pose + 0.0085 ≈ 0.73** — DOMINATED (the d_seg term alone is +0.71). Same failure class as
  the partition-store DEFER (S=0.84).
- To beat the frontier at this rate the code would need d_seg < ~3.2e-4, but the coarse code's
  floor is 0.0071 (the islands it cannot cheaply represent). The break-even (rate slope 6.659e-9
  d_seg/byte) is irrelevant — the topological code is d_seg-bound, not byte-bound.

**The byte attractiveness is an artifact of d_seg-lossiness.** A topological code that ACTUALLY
realizes a competitive d_seg must store the full per-frame partition (the islands), which is #52's
~524 KB (3× the frontier) — the rate then dominates. Either way it loses: cheap-but-d_seg-lossy
(S≈0.73) or d_seg-faithful-but-byte-fat (rate 3× frontier). There is no middle vertex because the
binding signal (islands) is full-rank content-noise.

---

## VERDICT — LIMITED (no premature kill, no over-claim)

Applying the existence-proof discipline both directions:

- **I do NOT over-claim a topological exploit.** The decomposition is real (coarse low-dim,
  fine volatile) but every realization is dominated: cheap coarse code → d_seg 0.0071 → S≈0.73;
  d_seg-faithful code → #52's 524 KB → rate 3× frontier. The shared-ego unification (the hypothesis'
  load-bearing claim) is FALSIFIED — ego explains only 23–38% of even the low-dim boundary.
- **I do NOT kill the paradigm.** Per Catalog #307 the IMPLEMENTATION (constant-template + low-dim
  ego-deformation + rendered boundary) is falsified at the measured operating point; the PARADIGM
  (topological/structured task-space code) is the same family as the nonrgb-capstone witness, whose
  open term is the SAME thing — the generator's d_seg power-law under training. This probe REINFORCES
  that verdict: the d_seg-binding structure (the islands) is content-noise that only a TRAINED
  full-grid generator (scored on its own frame-1) can reproduce — not a $0 topological sidecar.

**The single next step:** none on the topological-sidecar axis — it walls here. Route the d_seg
lever to the live **generator d_seg training campaign** (Muon stage-8 + d_seg-aware taper on the
real renderer, currently ~12× from frontier per the capstone) — the ONLY lever that touches the
full-rank content-noise islands, because there the renderer's own frame-1 is scored. This probe's
contribution to that campaign is the sensitivity prior: **0.72% of pixels (the fine islands) carry
the residual d_seg debt; the coarse 99.3% is near-free and near-constant → weight trainer capacity
toward the small-component / class-1 regions, not the stable large regions.**

---

## 6-hook wire-in (Catalog #125)

1. **Sensitivity-map — ACTIVE:** the d_seg debt concentrates in 0.72% of pixels (fine islands,
   mostly class 1), full-rank content-noise; the coarse 99.3% is near-constant near-free. Reusable
   seg-sensitivity prior: trainer capacity → small-component/class-1 regions; do NOT spend sidecar
   bytes on the coarse template.
2. **Pareto — ACTIVE (records dominated vertex):** the topological coarse code is dominated
   (cheap-d_seg-lossy S≈0.73; d_seg-faithful rate 3× frontier) — a new dominated point confirming
   no topological sidecar vertex beats the frontier.
3. **Bit-allocator — ACTIVE (advisory):** "frozen partition d_seg = full-rank fine-island content
   scatter; topological template+ego-deformation is d_seg-bound at 0.0071; allocate to the trainer
   not the archive."
4. **Cathedral-dispatch — N/A:** advisory, non-promotable, no archive change, no paid eval.
5. **Continual-learning — ACTIVE:** probe outcome `frozen_partition_topology_20260623` (verdict
   LIMITED, reactivation = live-run generator d_seg crossing the beat-frontier line) to register via
   `tac.probe_outcomes_ledger.register_probe_outcome`.
6. **Probe-disambiguator — ACTIVE:** `experiments/probe_frozen_partition_topology.py` is the
   disambiguator that arbitrates "is the partition a cheap topological template+ego-deformation?"
   (hypothesis) vs "is the d_seg-binding structure full-rank content-noise?" (measured) — the math
   arbitrates: the coarse half is low-dim but the binding half walls.

Mission contribution: **frontier_protecting** — closes the topological-sidecar $0 d_seg path with a
decisive two-scale intrinsic-dimension + ego-explained-fraction measurement, and reconfirms the
d_seg attack belongs in TRAINING (sister to a3061's trainer-side redirect). Sister-DISJOINT from
a3061 (flip-residual) and #52 (static byte cost): this measures the GT topology constancy + GT
boundary ego-deformation they left untested. NON-PROMOTABLE [contest-CPU advisory]. Pointer UNMOVED
0.19110.

## NO-FAKE ledger
- MEASURED (this unit, exact frozen-SegNet argmax cache + real frozen PoseNet, 600 frames, CPU,
  NEVER MPS): topology constancy (Result 1); boundary effective rank 4.07 (Result 2); PoseNet R²
  0.231 / horizon R² 0.034 (Result 3); island eff. rank 52.9 + d_seg-drop cost 0.0071 (Result 4);
  byte projection via `tac.contest_score` (Result 5).
- DERIVED: the S projections (contest_score arithmetic on the measured d_seg/byte anchors); the
  two-scale interpretation.
- NOT claimed: NO score moved; pointer UNMOVED 0.19110; no archive built/byte-closed; no exact-eval
  dispatch; no PR. The topological code is NOT viable as a frozen-instance d_seg sidecar (this
  verdict); the d_seg lever is training-side. Tool + 7 NO-FAKE tests committed; ruff clean.
