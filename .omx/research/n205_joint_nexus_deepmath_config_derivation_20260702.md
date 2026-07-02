---
title: "#205 PRE-LAUNCH GATE — PHASE 1: the DEEP-MATH JOINT-NEXUS config derivation"
date: 2026-07-02
axis: "[macOS-MLX advisory / design] — NON-PROMOTABLE. means != ends: this derives a config (a MEANS); the ONLY end is a byte-closed n600 exact row < 0.19110 from upstream/evaluate.py (contest-CPU/CUDA, NEVER MPS)."
pointer: "0.19110 UNMOVED (contest-CPU recoded-R3)"
scope: "RESEARCH/DESIGN ONLY. NO code edits, NO GPU, NO launch. Phase 1 of the 3-phase #205 gate (P1=this derivation; P2=MLX/Metal wall-clock verify; P3=3-clean-pass recursive adversarial review). FINALIZED after #234 net-S + #237 pose-reconcile + #223 parametrization land."
provenance_discipline: "Every number tagged MEASURED (byte-closed EXACT / MLX-rs advisory / macOS-CPU advisory) vs DERIVED (theory/arithmetic) vs PENDING (needs-measurement). All non-EXACT are advisory. Assembled from existing anchors via proactive recall (4 parallel Explore agents + direct reads); nothing re-derived that we already measured."
---

# #205 P1 — Joint-Nexus Deep-Math Config Derivation

**Operator 2026-07-02:** *"the #205 crux is d_seg convergence BUT there's a NEXUS with pose + rate + training-time + total-score; ensure full-stack + meta + meta-meta all optimized/calibrated/configured optimally taking the dynamics + interactions among/between all into account from deep math."* + *"your candidate S-optimal argv must be a REALIZABLE config … flag each lever's WIRING STATE against the LIVE levelset trainer."*

This is the marshaling pass: assemble the already-measured/derived response surfaces of the five nexus axes (d_seg × d_pose × rate × training-time × total-S), derive the couplings from the GR unified action (δS/δθ=0 in the frozen-scorer Fisher metric), and SOLVE for the candidate S-optimal optimal-form config — **using ONLY grepped-real flags** — plus the wiring-gap list (levers still MODULE-ONLY, needing #224 wire-in).

---

## 0. FRAME — the nexus, the common unit, the math-first tier order

**The objective (frozen, don't re-derive):** `S = 100·d_seg + √(10·d_pose) + 25·B/N`, `N = 37,545,489` bytes. The witness amortizes the frozen-SegNet argmax Morse-Smale partition; it is ONE variational action whose stationarity **δS_τ/δφ = 0 holds in the FIXED frozen-scorer Fisher metric G = E[Jᵀ F J]** (QFT-on-fixed-background, well-posed + measurable). [GR-action memory; veh-G1 formalized in `tac.canonical_equations`]

**⚠️ TRAINING-TIME IS A SECONDARY LEXICOGRAPHIC OBJECTIVE — NOT AN S-TERM, NEVER SCALARIZED (operator + compute-facet 2026-07-02).** `S` has **NO time term** (`evaluate.py:92`); the ONLY time constraint is the 30-min DECODE (inflate) budget. The objective is strictly **lexicographic**:
- **PRIMARY (inviolable): minimize contest-S** (d_seg + d_pose + rate).
- **SECONDARY: minimize training-time T, ONLY among equal-or-better-S configs** — `min T(θ) s.t. S(θ) = S*`.

**NEVER a scalarization `S + λ·T`; NEVER a faster-but-worse-S config.** Procedure: FIRST pick the S-optimal argv (§5, primary); THEN add EVERY **S-NEUTRAL** speed lever (free — bit-identical to the numpy-fp32 authority so S is unchanged to the bit). T is surfaced as an INFORMATIONAL secondary metric, explicitly **"not a term S was traded against."** Training-time enters the nexus ONLY through the convergence DYNAMICS (how fast d_seg descends per the curriculum/optimizer) and the wall-clock budget — never as an S addend.

**The calibration metric + common unit (the "how", operator Q 2026-07-01):** calibrate in the **Fisher geometry** — the top1−top2 **margin field is the MEASURED Fisher surrogate (Pearson 0.978, Spearman 0.908; anisotropy 9.56:1; 96.8% flip-mass in a 2px band)**, so it is BOTH the metric (where score-leverage lives = the codim-1 boundary annulus) AND the capacity target. The ONE common unit is **Δ(score-term)/byte realized-through-R on the n600 numpy-fp32 authority** — every knob (basis, mod-dim, β₂, stored coord, lane byte) scored in this ONE currency ⇒ representation/rate/optimizer knobs sit on ONE Pareto frontier (the Meta-Lagrangian typed atoms). This turns "sweep 12 knobs" into **(i) make the parametrization Fisher-efficient (directional/curvelet basis PRIOR to capacity) + (ii) KKT-waterfill capacity onto the annulus (margin-saliency).** [calibrate-parametrization memory]

**The math-first, measurement-gated 4-tier protocol + the ORDER (a DAG not a grid):**
- **T0 DERIVE-free** (don't sweep what geometry gives closed-form): Whitney mod-dim from intrinsic ~8-9; boundary-Nyquist bank-freq; curvelet parabolic freq_across²∝freq_along; β₂ floor from batch n; Muon NS steps K=5.
- **T1 MEASURE-through-R $0** on a FROZEN ckpt (render-grid/AA/analytic-lane/activation/basis — the two live gates ARE this tier).
- **T2 MEASURE-in-training** cheap warm-started off per-stage ckpts (β₂/temps/LR/Muon/persistence — only exist in the descent).
- **T3 EXACT byte-closed** (final composed config only = the pointer; one row per config, not per knob).
- **ORDER:** basis-BEFORE-capacity (capacity on isotropic basis HURTS +6%, on directional pays −48%); representation-BEFORE-dynamics (**the oracle-R AA floor caps the ceiling at ZERO training → Tier-2's job is REACH-the-floor, not lower-it**).

**The nexus reduction (the load-bearing structural simplification).** Two couplings collapse the 5-axis problem:
1. **seg⊥pose is FREE (measured):** render-space cos(∂d_seg/∂F, ∂d_pose/∂F) median **5.9e-5**, 99.95% pose-null, PoseNet Jacobian rank-6 FULL; **frame0 is seg-free** (`modules.py:108`, SegNet reads only `x[:,-1]`). ⇒ pose can be carried on a DISJOINT frame (frame0) with EXACT freeze-and-add (additive-S provably holds; PCGrad is a FALSE FRIEND — nothing to project at cos 6e-5). [pose survey SURVEY:18; veh-G2]
2. **pose is REMOVED from the witness's d_seg optimization** (`--w-pose` gates the pose term). ⇒ the witness's binding controllable job is **d_seg × rate × training-time**; pose is a near-separable block (see §2B for the CRITICAL task-space caveat that pose is NOT free-by-store on the SDF witness).

So the nexus is: **minimize the contest-S [100·d_seg(H, basis, curriculum, epochs) + √(10·d_pose) + 25·B(H)/N] over capacity/basis/curriculum, SUBJECT TO (i) a POSE side-constraint √(10·d_pose) ≤ ε via a disjoint-frame carrier, and (ii) the training-time/convergence budget** — reaching the d_seg REPRESENTATION FLOOR is the goal; doing it in the shortest wall-clock is the campaign co-objective (a constraint, not an S-addend).

---

## 1. THE OBJECTIVE + MARGINAL STRUCTURE (operating-point-dependent; the effort-allocation law)

Marginals (DERIVED, exact arithmetic; N=37,545,489):
- **d_seg marginal `∂S/∂d_seg = 100`** (CONSTANT — the only axis with a flat marginal).
- **d_pose marginal `∂S/∂d_pose = 5/√(10·d_pose)`** → **∞ as d_pose→0** (the nonlinear √ term).
- **rate marginal `∂S/∂B = 25/N = 6.66e-7 /byte`** → **1 KB = 0.00068 S** (CONSTANT).

**Operating-point flip (the measured 77×→2.71× importance inversion, CLAUDE.md):**

| operating point | d_pose | pose term √(10·d_pose) | pose marginal 5/√(10·d_pose) | pose/seg marginal |
|---|---:|---:|---:|---:|
| old 1.x scores | ~0.18 | 1.34 | 3.7 | 0.04× (seg 27× more) |
| **seg==pose crossover (DERIVED)** | **2.5e-4** | 0.05 | 100 | **1.0× (crossover)** |
| frontier / sidecar-solved | 2.94e-5 | 0.0172 | 292 | 2.9× (pose more) |
| task-space witness pose-blind | ~190 | 43.5 | 0.11 | 0.001× (CATASTROPHIC) |

**Frontier component split (MEASURED EXACT, contest-CPU 0.19110):** `d_seg 0.00056→0.05598 (29.3%) + d_pose 2.94e-5→0.01715 (9.0%) + rate 0.11797 (61.7%, BINDING) = 0.19110`. On the BORROWED frontier, rate binds. **On our witness, rate de-risks CHEAP (byte-close 83,062 B = 0.0553; RD-optimum 122 KB = 0.081) so d_seg becomes the binding term** — this is the whole reason for the pivot.

**Effort-allocation law (the nexus verdict):**
- **d_seg is the binding controllable term** (flat marginal 100, largest recoverable headroom on our vehicle). Spend the training + capacity + basis budget here.
- **rate is de-risked** — hold it at the RD-optimum KKT point (§3B), do NOT over-shrink (int8+brotli already at the entropy floor).
- **pose is a SIDE-CONSTRAINT** — hold √(10·d_pose) ≤ ε≈0.018–0.02 via a disjoint-frame carrier; do NOT chase d_pose below the crossover (2.5e-4) — its high marginal there is irrelevant because the pose is solved by STORE/carrier, not by witness capacity. **BUT (task-space caveat §2B): reaching ε at all requires w_pose>0 + a real-luma carrier — it is NOT free on the SDF witness.**

---

## 2. THE RESPONSE SURFACES (assembled from existing anchors; confidence + gaps flagged)

Calibration legend: **EXACT** = byte-closed n600; **MLX-rs** = macOS-MLX research-signal (advisory); **CPU-adv** = frozen CPU-torch through-R (advisory); **DERIVED** = theory/arithmetic; **PENDING** = needs measurement.

### 2A. d_seg surface (the binding axis)

| knob | measured response | maturity | source |
|---|---|---|---|
| **AA-SDF observation render × grid** | g192 0.01116→AA 0.00241; g256 0.00810→**0.00129**; **g384 0.00549→0.00086** (AA ss=2). Oracle-R floor @384 **0.00091**, @192 0.00247. | **WELL-CHARACTERIZED** (n600 through real R) | DAG:7223-7230; eq `aa_sdf_observation_footprint_render_dseg_v1` |
| **directional (curvelet/anisotropic) basis** | **−48% all-class** vs −8% lane-only (n96); n600 **−31% alone**; +cap n96 **−64%**. ⚠️ circular-GT (built from `gt.lstars`); realized −48% UNVERIFIED → self-orient fixed-point is the byte-closeable form. | SINGLE circular-GT point; realized-axis PENDING | §3A-D1; DAG:633-637 |
| **capacity (hidden-dim) AFTER basis-match** | isotropic capacity-ALONE HURTS **+6%**; after basis-match, h96-dir 0.003679 → h128/mod48-dir+cap **0.002447 (−70%)** @ep400; h192 OVERFITS (0.003362→0.005514). | 3 n600 points; clean isolated curve QUEUED | §3A-D2; DAG:645-673 |
| **mod-dim** | code SVD eff-rank PR=13.5 (90%@21); mod-32 is 1.5× over-wide; mod16-vs-21 d_seg-neutrality **UNMEASURED**. mod +1 = +1765 B×1200 (expensive per-frame). | SINGLE-POINT / PROBE | §3A; DAG:3406 |
| **epochs / curriculum length** | long900 ep450 0.002423 → ep800 **0.002176** (best DIRECT, still descending); overfit onset ~ep400 for cap-config. | WELL-CHARACTERIZED trajectory | §3A-D5; DAG:647 |
| **curriculum stage** | CE 0.01045→0.00643 · **tau_softplus(0.3) →0.00396 (THE primary drop)** · smooth →0.00423 (**RAISES +6.8%, DROP**) · l7 →0.00369 (small drop in MLX-trace BUT see l7-DEFECT reconcile) · **Muon = THE conditioning drop (PREDICTED ~6–9e-4, UNMEASURED — the binding open cell)**. | per-stage WELL-CHARACTERIZED; Muon-final PREDICTION | §3A-C2; DAG:532-539 |
| **activation** | **hosc+SIREN-init+β-anneal(1→4) = the HEALTHY regime** (drift-resolved `17f8fc663`). Fixed-β hosc DIVERGES standalone (β4 0.00723→0.01357; β8 →0.03996; d/dx tanh(β·sin)≈0 a.e. → AdamW random-walk); SIREN-init FIXES it (0.689→0.221). FINER 0.002915 / SIREN 0.003051 (−4.5% n600). step_basis stable but needs MLX `_act` port (DEFER). | WELL-CHARACTERIZED screen; a SPEED knob not a floor lever | §3A-D7; DAG:461-473,1610-1615 |
| **persistence/topology soft-clDice loss** | **111× more erasure-sensitive than CE** (span 8.83 vs 0.08, monotone); CE-only recovers 0.000 of erased islands vs CE+topology **+0.443**; 90.4% density-mass on flipped px. | NEW, BUILT + $0 signal; d_seg-drop is a PROBE (block-mean surrogate, no through-R row) | 1B state; DAG:7162; eq `persistence_topology_cldice_betti_island_recall_v1` |
| **island seed/contain/amplify** | LANE erased-recall 0.5646 → seeded **0.9304 (+0.366)** → bulk-wash 0.7961 → contained **0.9531 (+0.157)**; movable seeded +0.067. | NEW, BUILT + $0 signal | 1B state; DAG:7194; eq `island_finest_scale_protection_survival_v1` |
| **margin-saliency (LEVER-4) + KKT waterfill** | grad 1.0 on confident flip vs soft-cosine 1.9e-22; realized −16–36% vs CE. All-class flip-band Road47/Lane19/Undriv14/Movable9/MyCar11 — LEVER-4 defends 100% vs LEVER-3's 19%. | WELL-CHARACTERIZED (n96+n600) | §3A-D2/D3 |
| **analytic-lane render band (openpilot deg-3)** | FN (shape) 0.00046 < target; FP (dash on/off) **0.00396 = 90% of recon d_seg**; ground-frame 0.000439 vs image 0.000858. Post-hoc: c_naive **+0.000622 HURT** → witness-uncertainty gate **+0.000012 BREAK-EVEN** (kills 98% FP). Net-positive REQUIRES training-in. | render-band FREE (compose_fn, bytes unchanged); net-S PENDING (#234) | §3A-D10/D11; 1B state |
| **UNIWARD texture down-weight (β=4)** | Fridrich square-root-law; late-stage (l7/Muon) lever; BUILT smoke-verified, convergence A/B DEFERRED. | SINGLE / BUILT, no A/B | §3A-D8 |
| **chroma** | SegNet argmax on RGB ⇒ chroma carries argmax signal at the annulus; every pre-chroma verdict PROVISIONAL. Baked into BASELINE. | OPEN — UNMEASURED on realized axis | §3A-D9 |

**d_seg FLOORS/CAPS (DERIVED/MEASURED — don't re-derive):**
- **Representation floor is BELOW target:** AA-SDF @g384 = **0.00086 < sub-0.15 need-band [0.00077, 0.00118]**; oracle-R floor @384 = 0.00091. **⇒ THE PIVOTAL NEXUS FACT: sub-0.15 d_seg is reachable by REPRESENTATION; the ~0.003 gap from our best training (0.002447) is a TRAINING (reach-the-floor) problem, not a representation problem.** [DAG:7226-7230]
- Label-noise confident-GT cap **ΔS≈0.012** (93.9% of flips at GT-margin <0.5) → seg-only best-case **S≈0.184 > 0.15** → sub-0.15 needs the cheap rate too.
- Deterministic-render floor (k=0, no trained generator) ≈ 0.0185 bulk / 0.023 full = 15–40× budget → the **trained amortized-residual generator is REQUIRED** (residual-mode NO-GO, §2C).
- Manifold ~8-dim NONLINEAR lane-orbit (AE-knee 8 / MLE 13); linear "store-the-flips" sidecar NO-GO ×3.
- **Best measured d_seg by layer (use the right one):** proven Muon arm realized **0.003698 @ep1000** (n200, MLX-rs); converged cap-config **0.002447 @ep400** (n600, MLX-rs); long900 DIRECT **0.002176**; SDF-hosc n96 **0.00124**; n600 EXACT-eval pre-train pose-blind **0.006655**.

### 2B. d_pose surface (the SIDE-CONSTRAINT — with a CRITICAL task-space caveat)

| lever | measured d_pose | maturity | source |
|---|---|---|---|
| **stored-target sidecar (P1)** | 6 PoseNet scalars/pair → d_pose≈0; 7,200 B raw / <5 KB zlib; low-rank rank-4/511 **2,563 B, MSE 2.7e-5, −0.0004 rate**; bit-alloc 2.3 KB MSE 1.8e-6 (< fp16). **BUT this is the RGB-witness / training-GT solve — see caveat.** | GROUNDED deployed | §3D-P1/P4; `scorer_targets.py` |
| **warp-real-luma frame0 (H1, the task-space carrier)** | n600 byte-closed **d_pose 163.1→1.367 (−99%)**; oracle dense-flow 182→1.42; warp-alone −94% (182→10.53). SE(3) screw + ground-homography; frame0 seg-free ⇒ **ZERO d_seg cost**. ξ store 2,424 B. | MEASURED (advisory); residual→3.4e-5 UNMEASURED on witness | 1B state; pose-solved memory; eq `warp_real_luma_frame0_pose_carrier_dpose_v1` |
| **D1 disjoint-freeze-add + trunk-stopgrad** | 0-byte structural enabler; makes additive-S EXACT (cos 5.9e-5). | DESIGN on measured orthogonality | SURVEY:101-102 |
| **E1 KKT/Lagrangian pose-tube** | minimize d_seg s.t. √(10·d_pose) ≤ ε≈0.02; 0-byte training reformulation. | DESIGN | SURVEY:115-116 |

**⚠️ THE #237 RECONCILE (the CRITICAL nexus finding — pose is NOT free on the task-space witness):** the scorer runs PoseNet on the decoded FRAMES; a **pose-BLIND SDF render** (`--w-pose 0`) sits at **d_pose ~189.59 (term 43.5, CATASTROPHIC)** — a stored 6-scalar *deploy* sidecar is bytes PoseNet never reads. The "pose SOLVED via `--w-pose 0` sidecar" line (P3, §4) holds ONLY for an RGB witness that carries pose in its rendered frames. For the non-photoreal SDF witness the pose MUST be realized in the render:
- **PRIMARY (operator 2026-07-02):** WARP-REAL-KEYFRAME-LUMA by the stored twist ξ (real luma advected by ξ → PoseNet reads real warped motion → pose-valid BY CONSTRUCTION, frame0 seg-free). Measured −99% (163→1.37); but **1.37 → term 3.70 is STILL far from ε** — the residual to ~0.018 needs the trained per-pair dξ/FiLM residual (**REACHABLE per rank-6, UNMEASURED on witness**).
- **The stored 6-scalar targets become the TRAINING GT + `--w-pose>0` supervision** (not a free deploy byte); `--w-pose>0` is a **hard row prerequisite** for the task-space witness.
- **RECONCILE VERDICT for the config:** pose = ξ PURE-POSE carrier (2,424 B, ξ-coding of lanes REFUTED — Pareto-dominated, ξ is pure-pose) + `--w-pose>0` supervising warp-real-luma frame0. The n600 baseline's `--w-pose 0` is the pose-BLIND d_seg-isolation leg, NOT the shippable config.
- **Pose collapse (d_pose 2.67–12.66)** = the amortized-luma RECONSTRUCTION carrier (AVOID); the warp-real-luma carrier is a DIFFERENT mechanism (real luma, not reconstructed).
- **d_pose descends free with training when rendered** (ep50 0.0072→ep488 0.0002, MLX-rs) — so once the carrier is wired, pose is not expected to fight d_seg.

**HONEST GAP:** the 3.4e-5→0.018 pose term is the ANCESTOR-RGB anchor, NOT witness-validated. The #221 fine-tune (w_pose>0) + #206 A/B (FiLM-SDF vs warp-real-luma) MEASURE it. This is the single largest open uncertainty in the S-budget.

### 2C. rate surface (de-risked CHEAP; hold at the KKT optimum)

| lever | measured/derived | maturity | source |
|---|---|---|---|
| **RD-optimum B\*** | **mod-32/h96 → ~122 KB, rate 0.081** (predicted S 0.134 sub-0.15 IF directional-ON); mod-48/h128 → 161 KB (0.107, **+0.026 S overshoot**). Curve `d_seg(B)=d0·(B0/B)^α`, **α≈2.34 BORROWED from 2 points (LOAD-BEARING; response-surface sweep QUEUED)**. | curve DERIVED-from-2-anchors; config bytes MEASURED | §3B-R; DAG:1471-1483 |
| **witness byte-close (measured)** | levelset arm int8+brotli **83,062 B → rate 0.0553** (EXACT stat); g3 bc20 **89,244 B → 0.0594** (dual-exact). | MEASURED byte-closed | §2 rows; git `09c397a0a` |
| **KKT capacity↔rate** | at the interior optimum **∂d_seg/∂byte = 25/(100·N) = 6.66e-9/byte**; equivalently **1 KB of extra weights pays iff it lowers d_seg by > 6.82e-6**. Solving with α=2.34 lands B*≈122 KB. H-invariance: bits vs channels is a distinction without a difference at the optimum. | law DERIVED exact; crossover DERIVED-from-fit | §3B-R2; DAG:423-433 |
| **quant/entropy levers** | int8 is S-min (sub-8-bit PTQ RED: int8→int7 saves 1.4 KB for +0.044 S); DeepCABAC/order-2 EXHAUSTED (brotli-q11 6.891 vs H(W) 6.884); int5-QAT caps S=0.483; **WRQ score-aware per-tensor = largest post-T1 lever, UNGROUNDED magnitude, OPEN**. | MEASURED byte-closed (mostly EXHAUSTED) | §3B-R14/R15 |
| **code low-rank penalty** | nuclear-norm REFUTED (magnitude≠spread); replacement = **spectral-entropy + Stiefel-W (byte-free): PR(M) 1.19→4.57, effR 6.5×, −60% code bytes**. DEMOTED 2nd-order (PR collapses WHILE d_seg improves). | MEASURED rank effect; DEMOTED | §3B-R22; veh-G4 |
| **residual-mode v2 hybrid** | **MEASURED NO-GO (decisive):** 50–86% of residual is INTERIOR (not a thin annulus); `unreachable_dseg` floors at ≈0.010 (=+1.0 S) independent of INR capacity; shrinking it needs keyframes → store_rate 0.055–0.092 (~entire budget). BOTH axes catastrophic. | MEASURED (k* RD sweep) | residual-memo; §3B-9 |
| **analytic-lane serializer (#234 COUNTED half — MEASURED n600, a2688263)** | naive float64 ~220 KB was the pre-coding estimate; the coherent-tracking build MEASURED the coded RATE: **`coherent_slot_none` 0.02750 (lossless) · `coherent_slot_rpca` 0.02621 (RMS 2.96 m) · `sort_MA_win25` 0.01489 (−46%, LOSSY RMS 6.07 m)**. Clip is swap-LIGHT (fit-jitter, not slot-swaps → correspondence only 0.5%). **Net-S UNMEASURED = #205 through-R A/B; NOT assumed net-positive.** | render-band FREE; serializer a SLOT (default lossless; A/B in #205) | coordinator 2026-07-02; wave_f |
| **rule-118 FREE/COUNTED law** | FREE = generic forward-pass + seed-generated bases + openpilot rasterizer + AA render (deterministic decode op); COUNTED = ~8-dim lane coords + learned INR weights + per-pair latent; FORBIDDEN = scorer weights / GT-argmax table / hide-data-in-code. | binding law | §3B-R19 |

**Rate verdict for the config:** ship the trained witness weights **int8+brotli** (already at entropy floor → ~83 KB / 0.055 measured); hold mod-32/h96 (RD-optimum region); `--code-nuclear-weight 0` + `--film-stiefel`/`--code-spectral-entropy` as byte-free DEMOTED ablation arms (default OFF for attribution-clean first). **`--residual-mode OFF` (NO-GO).** The analytic-lane #234 COUNTED serializer is IN-if-net-positive PENDING (temporal-AR coding + T5 gate); the render-time band is FREE-but-net-positive-needs-training-in.

### 2D. training-time / convergence dynamics (the reach-the-floor axis — a CAMPAIGN CONSTRAINT, NOT an S-term; enters via convergence dynamics + wall-clock budget)

**Realized through-R per-stage trace (the most-trustworthy, n96/n200 through R + frozen SegNet — use THIS over the MLX-port surrogate):** CE 0.01045→0.005443 · **tau_softplus(0.3) →0.004563 (−0.000879, THE primary single drop)** · l7 →0.004287 (−0.000276, short knee ~ep700 — but l7 DEMOTED, see below) · **Muon →0.003718 (−0.000569, still descending/decelerating)**. So the realized Muon floor reached so far is **0.003718** (not yet the predicted 6–9e-4 — the decisive converged Muon run OOM-died before a converged eval; the floor is the binding open cell).


| element | measured/derived | maturity | source |
|---|---|---|---|
| **short curriculum** | S0 seed → S1 CE → S2 tau_softplus(0.3) → **[l7 DEMOTE]** → S4 Muon; SKIP smooth (+6.8%) + QAT/C1a/λ/σ; **~1100–2100 ep vs PR95's 29,650**. | GROUNDED deploy design | §3D-C1c/C19c |
| **per-stage d_seg dirs** | CE ↓ · tau_softplus **THE primary drop (→0.00396)** · smooth **↑ DROP IT** · Muon **THE conditioning drop (predicted 6–9e-4)**. | WELL-CHARACTERIZED; Muon-final PREDICTION | §3D-C2c |
| **l7 RECONCILE** | MLX-port trace shows a SMALL l7 drop (0.00396→0.00369) BUT the 2026-07-01 5-agent deep pass (DAG:7142) + drift-fixer eq `l7_linf_sharpening_defect` reclassify l7 a **DEFECT** (L∞ sharpening inside a smoothing/viscosity flow = the measured d_seg-decoupling). **LATEST verdict = DEMOTE l7** (target d_seg ≤ 0.00077, rate no-slack). | reconciled → DEMOTE | DAG:7142; 1B state |
| **Muon finisher** | Muon descends **~32% MORE than AdamW** (gap widens monotone; AdamW grad-norm collapses on κ~19 Hessian); **muon-lr = 2e-3** (band 1e-3..2e-3, ceiling 5e-3; NOT 0.03; trainer-default None→0.1·lr=1e-4 is 20× too low). Final-stage-only, NS steps K=5, 2-D hidden weights only (heads/codes stay AdamW). | GROUNDED (witness band) | §3D-C4c/C5c |
| **two temps** | `--tau-softplus-tau`=0.3 (seg-surrogate reachability floor Δ_min≈0.3) vs render `--softmax-temp` **1.0→0.05 anneal** (frozen 0.05 for Muon). | GROUNDED | §3D-C7c |
| **stage-transition REHEAT** | rewarmup floor 0.1×/8 ep + reset-moments (partial restart; full 1.0× re-destabilizes); the l7→Muon switch already re-treats via fresh optimizer. | MEASURED | §3D-C9c |
| **EMA** | decay 0.997, save SHADOW (not live), eval-only snapshot+restore (EMA-shadow-lag up to 78× = the "0.505 wall" artifact); wider finisher 0.999/0.9995 SWA-style (`--ema-decay-finisher`) from Muon-start. | GROUNDED (non-neg) | §3D-C11c |
| **NCA stabilizers** | grad-clip 1.0 + spike-factor 5.0 (5×-median) + per-boundary reset; n_restarts≥2 keep-best at CAMPAIGN level. | GROUNDED | §3D-C12c |
| **β₂ (AdamW second moment)** | **MEASURED anchor = β₂=0.999** (~1000-step second-moment memory) + `--stage-transition-reset-moments` (flush stale moments at each boundary). ⚠️ **the "0.9999999" recall is a MIS-ANCHOR** (not in the training-dynamics corpus; appears only in unrelated scorer/kahan-ema files). DERIVE-T0 `1−β₂* ≲ (1−β₁⁵)/n^3.5` from batch n; #222 sweep = the disambiguating optimizer-vs-representation point. **NOT a flag** (AdamW hardcodes 0.999, line 1235). | β₂=0.999 MEASURED; higher-floor PENDING/needs-flag | calibrate memory; n600-cert; #222 |
| **MD-Decoupling** | `--optimizer md --md-base {adam,muon}`: byte-identical default, 13/13 tests, stable transitions by construction; but $0 smoke = STABLE (29× lower gnorm) yet UNDER-STEPS at adamw-lr → **PARALLEL ABLATION ARM, not a blind drop-in.** ⚠️ **WIRED in the BASE trainer (line 2296), NOT the levelset entry point (subset-miss).** | WIRED-in-base-only; DEMOTED ablation | DAG:1417-1420 |
| **structured-init S0 seed** | `--structured-init --structured-init-include-lane --lane-prior-phi1` (openpilot deg-3 centerline SDF, FREE 0 bytes; separatrix residual 1.9e-5); seed → low-freq free → jump to high-freq annulus (NTK). MEASURED CAVEAT: no epoch-0 realized win (texture-gated); trajectory A/B only. | GROUNDED (train-time init) | §3D-C14c; FEED-ef |
| **wall-clock (MLX/Metal)** | custom Metal grouped backward (`TAC_MLX_CUSTOM_GROUPED_BACKWARD=1`, **16.9× MEASURED, banked, carries the seg-only step — build nothing new**) = **0.098 s/step → 54–59 s/ep @ n600** (26.4× over reference; grad-cosine **1.000000**). **99% of the step is the frozen SegNet+PoseNet fwd+bwd; the render is 0.25%** (⇒ render-384-vs-192 is dominated by the scorer, not the render → 384 is ~free). **K=1 throughput-optimal (pair-batching MEASURED NEGATIVE — compute-bound convs)**; fp32 sweet spot; CPU-torch verdict over 600 pairs is the new bottleneck (`--async-verdict` ~10.9% reclaim, `--eval-every 25`). Total-run: fast+accurate 300 ep ≈ **5 h**; a full ~1500-ep n600 h96 run ≈ **~22 h**; full-cap through-R with verdicts ≈ **~70 h** → **resumability + per-stage checkpoints MANDATORY.** | MEASURED | DAG:900-910,1702,2163 |
| **mx.compile (config win — NOT yet a flag)** | the trunk-INR + seg-loss closure runs **UN-compiled** on the hot path; shapes static (accum_pairs=8, 384×512). Wiring an `mx.compile` step ≈ **~5% (ESTIMATE)**, LOW risk, parity-gated (`assert_compile_bit_identical`, max\|Δ\|=1e-6). On a multi-day run 5% = hours. **WIRING GAP: no `--compile-step` flag in either trainer (grep empty) → cannot go in the argv; wire-in + parity-gate first.** | config-win PENDING (wire-in) | coordinator/compute-facet 2026-07-02 |
| **freeze-decoder-fit-codes amortization** | train shared decoder on subset (n96/n192), FREEZE, fit ONLY per-pair codes for all pairs (embarrassingly-parallel) — days→hours. Incompatible with Muon/residual-mode/structured-init. | BUILT | trainer `--freeze-decoder-fit-codes` |

---

## 3. THE COUPLINGS (deep-math, from δS/δθ = 0 in the Fisher metric — they FALL OUT, not bolted on)

1. **seg ⊥ pose = FREE (additive-S EXACT).** cos(∂d_seg/∂F, ∂d_pose/∂F) = 5.9e-5; frame0 seg-free. ⇒ carry pose on the disjoint frame0 (warp-real-luma) with trunk-stopgrad; the pose term adds without perturbing d_seg. PCGrad FORBIDDEN (false friend). **Config consequence:** `--w-pose>0` on a frame0-luma carrier is safe for d_seg. [pose survey; veh-G2]

2. **capacity ↔ rate = KKT waterfill.** `∂d_seg/∂byte = 25/(100·N) = 6.66e-9/byte` at the optimum → B*≈122 KB (mod-32/h96). Over-capacity (mod-48/h128, 161 KB) overshoots +0.026 S; under-capacity (bc20) leaves d_seg-debt. **Config consequence:** mod-32/h96, hold rate at the RD-optimum, spend marginal bytes only where Δd_seg/byte > 6.82e-6/KB. [rate agent; §3B]

3. **basis ↔ capacity = STRICT ORDER (basis BEFORE capacity).** Capacity-ALONE on isotropic basis HURTS +6%; after directional basis-match it pays −64/−70%. **Config consequence:** `--self-orient` directional/curvelet basis is baked into the BASELINE; capacity/routing levers engage AFTER. [D1/D2]

4. **representation ↔ dynamics = FLOOR CAPS CEILING.** The oracle-R AA floor (0.00086 @g384) is BELOW the sub-0.15 need at ZERO training. ⇒ **Tier-2's job is REACH-the-floor, not lower-it** — the ~0.003 gap is a training/conditioning problem (Muon finisher + reheat + the AA/persistence/island levers), not a representation deficit. This is why the config front-loads the representation (AA render, directional basis, structured-init seed) and the Muon conditioning finisher. [calibrate memory; DAG:7230]

5. **training-time ↔ d_seg convergence DYNAMICS.** The curriculum is a homotopy of relaxations (CE → tau → Muon) = deterministic annealing = coarse-to-fine curvelet scale = Morse-Smale persistence order = temperature annealing (ONE object). Muon is THE conditioning drop (orthogonalized finisher on a formed partition; AdamW collapses on the κ~19 Hessian). Stage transitions exhibit critical-slowing → REHEAT (rewarmup 0.1×/8ep + reset-moments) makes them stable; MD-Decoupling makes them stable-by-construction (ablation). **l7 is a DEFECT (L∞ decoupling) → DEMOTE.** [§2D; GR-action; DAG:7142]

6. **pose carrier ↔ d_seg = ZERO-COST (frame0 seg-free) BUT NOT free-by-store.** The warp-real-luma carrier lives on frame0 (seg-free) so it costs 0 d_seg, AND reuses the stored ξ (dual-use: same ξ that would advect the partition IS the pose). BUT the carrier is the ONLY way to escape the pose-blind d_pose~190 catastrophe on the SDF witness. **This is the #237 reconcile — the config MUST wire the carrier (w_pose>0), it is not optional.** [pose-solved memory; §2B]

7. **rate ↔ analytic-lane (#234) = FREE-render vs COUNTED-serialize fork; a SLOT, NOT assumed net-positive.** The render-time band is FREE (compose_fn, 0 bytes) but net-positive only if TRAINED-IN; the COUNTED serialized lane costs a MEASURED 0.0149–0.0275 rate (coherent-tracking build a2688263; clip swap-light → correspondence only 0.5%). **Config consequence:** default the lane band to `coherent_slot_none` (lossless, safe) and decide (`coherent_rpca` low-distortion vs `sort_MA` low-rate vs band-OFF) as an **IN-#205-RUN through-R A/B** — the band may not earn its bytes; the net-S is the #205 gate, not a settled rate-win. [coordinator 2026-07-02; wave_f; §2C]

---

## 4. THE THREE LEVELS (full-stack × meta × meta-meta — calibrated + interactions respected)

- **Full-stack (arch/basis/curriculum/optimizer/losses/parametrization):** the §5 argv — mod-32/h96/nh4, hosc+SIREN+β-anneal, self-orient directional basis, CE→tau→[l7-demote]→Muon short curriculum, muon-lr 2e-3, EMA-shadow 0.997, REHEAT transitions, structured-init+lane-prior seed, chroma, eikonal/length regularizers. **Interaction respected:** basis-before-capacity (#3), representation-before-dynamics (#4), l7-demote (#5).
- **Meta (config/DSL-gauges/campaign):** the config IS a DSL program (`tac.witness_dsl.curriculum_dsl.openpilot_seeded_opening` + `campaign.plan_adaptive_step`, validate()-refuses-invented-flags, 294 tests). Adaptive stacking (#188 `campaign.decide_next_stage`, window 300): EXTEND / ADVANCE / RERUN_NEW_CONFIG / ROLLBACK_BRANCH; curvelet-scale climb via warm-safe `--max-bank-freq` 16→32→64. **Interaction respected:** shape-changing flags (bank-n-scales/hidden/mod) force a FRESH arm; loss/projection levers land as warm-start re-treatments.
- **Meta-meta (triality-consistency / Fisher-common-unit / response-surfaces):** every calibrated knob compounds into `tac.canonical_equations` EmpiricalAnchor (residual = pred-vs-measured) → the triality DAG↔DSL↔equations stays consistent (calibrate once, write registry, never re-sweep). The common unit (Δscore-term/byte through-R, margin=Fisher 0.978) keeps all knobs on ONE Pareto frontier. **Interaction respected:** info-gain ranking (β₂ sweep #222 = the maximally-disambiguating optimizer-vs-representation point), not a uniform grid.

---

## 5. THE CANDIDATE S-OPTIMAL OPTIMAL-FORM ARGV (grepped-real flags ONLY; slots explicit)

Basis = `witness_autoconfig.proven_base` (recalled verbatim from the 0.003698 Muon arm) + the calibrate-parametrization T0-derived updates + the §4 launch-config marshaling. **Every flag below was grep-verified in `experiments/train_levelset_witness_realized_through_R_mlx.py`'s argparse (NEVER-INVENT).** The attribution-clean FIRST launch keeps surgical levers + DM1 OFF (they re-treat as shape-compatible warm-starts).

```
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 \
.venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py \
  --out-dir <run_dir> --gt-cache <n600_cache> --num-pairs 600 \
  --mlx-device gpu --seed 0 --async-verdict \
  --epochs 1500 --eval-every 25 --verdict-pairs 96 \
  --curriculum \
  --tau-softplus-start-epoch 450 --tau-softplus-tau 0.3 \
  --l7-start-epoch 1499            # l7 DEMOTED: park at end (curriculum guard needs tau<l7<=epochs) \
  --muon-start-epoch 1088 --muon-lr 0.002 --muon-momentum 0.95 --muon-ns-steps 5 \
  --stage-transition-rewarmup-epochs 8 --stage-transition-rewarmup-floor 0.1 \
  --stage-transition-rewarmup-shape linear --stage-transition-reset-moments \
  --w-seg 100 --w-pose 0          # SLOT-POSE: 0 = pose-BLIND d_seg-isolation leg; >0 required for a shippable row (see §2B/#237) \
  --score-domain-loss \
  --mod-dim 32 --hidden-dim 96 --n-hidden 4 \
  --activation hosc --hosc-beta 1.0 --hosc-beta-end 4.0 --hosc-beta-anneal linear --hosc-omega 1.0 --siren-init \
  --softmax-temp-start 1.0 --softmax-temp-end 0.05 --tau-anneal-shape cosine \
  --self-orient --n-dir-freqs 2 --freq-across 32 --freq-along 4 --reorient-every 50 --max-bank-freq 64 \
  --chroma --palette-anchor \
  --eikonal-weight 0.01 --length-weight 0.001 \
  --render-h 384 --render-w 512 --accum-pairs 8 --grad-clip 1.0 --spike-factor 5.0 \
  --ema-decay 0.997 --ema-decay-finisher 0.9995 --ema-decay-finisher-start-epoch 1088 \
  --structured-init --structured-init-include-lane \
  --lane-prior-phi1 --lane-prior-phi1-mode replace --lane-prior-phi1-dash-gate \
  --ckpt-every 25 --stage-checkpoints
```

**Curriculum boundaries (1500 ep):** CE ep1–450 · tau_softplus ep450–1088 (THE primary drop) · **l7 DEMOTED (parked @1499)** · Muon ep1088–1500 (the conditioning drop, tau+render-temp frozen 0.05). Rationale: proven fractions (tau@0.30, muon@0.726) scaled to 1500; the 5-agent-pass l7-demote skips the L∞ defect. Wall-clock ≈ 1500 × ~54 s/ep ≈ **~22.5 h** (n600, h96) — a multi-day-class run → resumable + per-stage ckpts (both ON).

**Explicit SLOTS (do NOT hardcode; resolve at finalize):**
- **SLOT-POSE (#237):** `--w-pose` 0 → >0 + warp-real-luma carrier wire-in. The pose term is NOT closed at w_pose=0 (d_pose~190). The first attribution-clean launch may run w_pose=0 to isolate d_seg, but the shippable S-optimal row REQUIRES the carrier (MODULE-ONLY, §6). Default the shippable config to `--w-pose 1.0` once the carrier is wired.
- **SLOT-234 (analytic-lane, a SLOT — default lossless, A/B decided IN the #205 run; MEASURED n600 rate, coherent-tracking build a2688263):** correspondence-first did NOT beat the moving-average on rate — this highway clip is **swap-LIGHT** (edge-free parallel lanes; the delta is far-field fit-JITTER, not slot-swaps → correspondence is only a 0.5% lossless win). Measured lane-coeff RATE options: **`coherent_slot_none` = 0.02750 (lossless, −0.5%) = the SAFE DEFAULT** · `coherent_slot_rpca` = 0.02621 (RMS 2.96 m, lowest DISTORTION) · `sort_MA_win25` = 0.01489 (−46%, but LOSSY geom RMS 6.07 m). **The band net-S (lane-recall d_seg win MINUS rate cost) is UNMEASURED = the #205 through-R A/B gate — do NOT assume net-positive; it may not earn its ~0.015–0.0275 bytes.** IN THE CONFIG: default the lane band to `coherent_slot_none` (lossless, safe); represent (`coherent_rpca` lowest-distortion vs `sort_MA` lowest-rate) as an IN-#205-RUN A/B, NOT a fixed rate contribution. The task-λ seam (`per_dim_lambda`: edge-preserving on ∂d_seg/∂coeff, not geometric RMS) is where correspondence+denoise could still beat the MA on the metric that matters.
- **SLOT-223 (parametrization):** mod-dim ∈ {32 overfit / 26 review / 19 aggressive-Whitney-floor}; the T0-derive is Whitney(2m+1) for measured intrinsic m~9 → 19; overfit headroom → 26; proven arm → 32. The #223 parametrization pass (byte-close sweep) resolves the RD-optimal mod-dim; default 32 (proven), fold to 26 if the byte-close sweep confirms −0.004 S.
- **SLOT-β₂ (#222):** AdamW β₂ default 0.999 (no flag); the T0-derive + #222 disambiguating sweep may motivate a `--beta2` flag (WIRING GAP, §6).
- **DEMOTED byte-free ablation arms (OFF for attribution-clean first, warm-start re-treat):** `--film-stiefel --code-spectral-entropy-weight <β>` (DM1 rank, DEMOTED 2nd-order); `--margin-saliency-weight <w> --margin-saliency-start-epoch 1088 --margin-saliency-uniward --margin-saliency-uniward-beta 4.0` (LEVER-4 late-stage); `--lane-thin-weight <w>` (LEVER-B birth-death dashes); `--head etf`/`--additive-margin` + `--logit-adjust-per-class` (#218 margin-field). **`--residual-mode` OFF (NO-GO).** `--code-nuclear-weight 0` (REFUTED).

**S-NEUTRAL SPEED LEVERS (the SECONDARY lexicographic objective — applied AFTER the S-optimal argv is fixed; each is bit-identical to the numpy-fp32 authority so S is unchanged to the bit; NONE trades S).** These are FREE wins on `min T s.t. S=S*`:
- `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` — **16.9× MEASURED, banked, grad-cosine 1.000000, correctness-critical** (already in the argv env prefix; MUST verify active per the launch-gate-throughput discipline).
- **K=1** (pair-batching is a MEASURED NEGATIVE — compute-bound convs; do NOT batch).
- `--async-verdict` — the CPU-torch verdict runs off a snapshot in a background thread (~10.9% reclaim; **bit-identical training**, the verdict is never read back).
- `--eval-every 25` — the verdict over 600 pairs is the wall-clock bottleneck; keep the cadence coarse.
- fp32 (accurate sweet spot; fp16 is pose-`1/√`-fragile — NOT a speed lever here).
- **`mx.compile` step (~5% ESTIMATE, WIRING GAP):** the trunk-INR + seg-loss closure is UN-compiled; wiring an `mx.compile` + parity gate (`assert_compile_bit_identical`, max\|Δ\|=1e-6) is a pure S-neutral win — **but `--compile-step` is NOT a real flag (grep empty) → wire-in + parity-gate first, then add.** Compute facet is READY otherwise; build nothing new pre-launch (compute-facet ledger `16cb540f0` / `n205_mlx_metal_new_kernel_plan`).

**T (informational secondary metric, NOT a term S was traded against):** ~54–59 s/ep @ n600 h96; 1500-ep run ≈ ~22 h (full-cap-with-verdicts ≈ ~70 h); resumable + per-stage checkpoints mandatory.

---

## 5b. PHASE-2 FINALIZED SHIPPABLE ARGV (post-#224 wire-in, 2026-07-02; HEAD `e28cbab63`)

**GO received 2026-07-02. Durability CONFIRMED** (crash-resume smoke ALL_PASS on `e28cbab63`, ~47 s; base + Muon-finisher bit-identical resume 0/28; fail-closed all pass — commit `e28cbab63`/`2ca1726ae`). **#224 CLOSED the §6 blocking gaps:** the LIVE levelset trainer (`experiments/train_levelset_witness_realized_through_R_mlx.py`) now has `--pose-carrier` (WarpRealLumaFrame0Carrier — FULLY WIRED, the FEED-224 fail-closed guard REPLACED), `--render-aa`, `--lane-render-band`, `--adam-beta2`, `--persistence-loss-weight`, `--amplify-weight`, `--seed-islands` (also fully wired) all reaching render/loss (callsite-traced, NOT argparse-only). This subsection SUPERSEDES the §5 candidate argv for the shippable row; §5 is preserved (Phase-1 provenance).

**The reconcile (§5 candidate → shippable), 4 load-bearing deltas — each toward the MEASURED/REQUIRED value:**
1. **render-aa: `supersample 2` → `none` + `--lane-render-band` (Wave D AA CORRECTION, DECISIVE).** The §5 "wire supersample as the #1 floor lever" is SUPERSEDED by `aa_feasibility_reconciliation_20260702.md` (same-day, MEASURED): brute supersample is DISQUALIFIED on TWO independent grounds — (a) it HURTS the witness **−49%** (the 0.00086 is a REAL-FRAME *ceiling* SIGNAL-A, NOT the witness-realized SIGNAL-B — supersampling an already-smooth softmax-of-SDF partition recovers nothing), and (b) fp64 decode **41.3 min > the 30-min budget** AND neither shipped inflate applies ss (archive stores render_h/w=384; decode point-samples) = a train/decode observation **MISMATCH** ⇒ the AA benefit is NOT realized at decode = a FAKE optimization. The contest-feasible OPTIMAL AA is `--render-aa none` + the analytic coverage-integrated `--lane-render-band` (O(1)/pixel, base-grid, decode IN budget, MEASURED to HELP via the witness-uncertainty FP gate). This is the code + capstone-artifact + Wave-D consensus.
2. **pose: `--w-pose 0` → `--w-pose 1.0 --pose-carrier` (means/ends firewall; #237/SLOT-POSE).** A `--w-pose 0` leg is ADVISORY (pose-blind SDF → d_pose~190, S-catastrophic; does NOT move the pointer). The pose-carrier is now WIRED (requires w-pose>0; default `--pose-carrier-residual-mode table` = a SEPARATE per-pair dξ table ⇒ the code manifold stays PURE d_seg (seg⊥pose additive-S EXACT, cos 5.9e-5); frame0 seg-free ⇒ 0 d_seg cost). Warp-real-luma measured d_pose 163→1.37; residual→ε≈0.018 is the largest open uncertainty (Phase-3 Q).
3. **mod-dim: `32` (SLOT-223, task-directed) — NOT the capstone's aggressive `19`.** d_seg is the BINDING term; 19 (Whitney floor for m~9) is rate-saving but its d_seg-neutrality is UNMEASURED (§2A). Rate has slack (0.055 vs 0.081 RD-optimum). 32 is the PROVEN arm value (reached the measured d_seg 0.003698) AND covers the composite m~13 (d_seg⊕screw) with headroom. #223 byte-close sweep may fold 32→26→19 ONLY if measured d_seg-neutral.
4. **β₂: `0.999` (MEASURED anchor) — NOT the capstone's `0.9999999`.** §2D flags 0.9999999 as a possible MIS-ANCHOR + says #222 must disambiguate; 0.999 is the MEASURED anchor == MLX default (byte-identical, no bias-correction-gate confound on the first attribution row). 0.9999999 is the T0-DERIVED candidate (arXiv 2603.02092 small-n) gated on #222.

**Reasoned deep-math call on the finest-scale erasure levers (persistence/amplify/seed):** persistence-loss + amplify are **ON** (the §4 coupling #4 "floor caps ceiling": with render-aa=none, NO supersample floor-setter, the finest-scale ERASURE recovery becomes the PRIMARY ~0.003→0.00086 gap-closer; persistence is 111× more erasure-sensitive than CE (+0.443 island recall), amplify rides the SHARED `_signed` margin (+0.366) — both ride the shared seg forward = no extra cost). weights = ENGAGE (T2-calibration start, LABELLED not measured-optima). `--seed-islands` **OFF** (now wired, but a separate optimizer-group + grad-shield restructure = more confound; amplify carries the island mechanism loss-only). DEMOTED ablation arms (margin-saliency/lane-thin/film-stiefel/head-etf/code-nuclear/residual-mode) OFF (attribution-clean first; warm-start re-treat). l7 DEMOTED (parked `--l7-start-epoch 1000` = epochs; L∞-defect).

**THE FINALIZED SHIPPABLE ARGV (every flag grep-verified in the LIVE trainer; 83/83 in-argparse, 0 invented; `--help` parser builds clean; all choice-values valid):**

```
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 \
.venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py \
  --out-dir experiments/results/levelset_n600_witness_capstone_<UTC> \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 \
  --mlx-device gpu --seed 0 \
  --epochs 1000 --eval-every 25 --verdict-pairs 0 --async-verdict \
  --curriculum \
  --tau-softplus-start-epoch 300 --tau-softplus-tau 0.3 \
  --l7-start-epoch 1000 \
  --muon-start-epoch 726 --muon-lr 0.002 --muon-momentum 0.95 --muon-ns-steps 5 \
  --stage-transition-rewarmup-epochs 8 --stage-transition-rewarmup-floor 0.1 \
  --stage-transition-rewarmup-shape linear --stage-transition-reset-moments \
  --w-seg 100 --w-pose 1.0 --score-domain-loss \
  --pose-carrier --pose-carrier-residual-mode table \
  --mod-dim 32 --hidden-dim 96 --n-hidden 4 \
  --activation hosc --hosc-beta 1.0 --hosc-beta-end 4.0 --hosc-beta-anneal linear \
  --hosc-omega 1.0 --siren-init \
  --softmax-temp-start 1.0 --softmax-temp-end 0.05 --tau-anneal-shape cosine \
  --self-orient --n-dir-freqs 2 --freq-across 32 --freq-along 4 --reorient-every 50 \
  --max-bank-freq 64 \
  --chroma --palette-anchor \
  --eikonal-weight 0.01 --length-weight 0.001 \
  --render-h 384 --render-w 512 --render-aa none \
  --lane-render-band --lane-band-start-epoch 300 --lane-band-uncertainty-source witness \
  --lane-band-tau 0.85 --lane-band-eps 0.35 --lane-band-softness 1.0 \
  --lane-band-dash-forward-max-m 55.0 --lane-band-weight 1.0 \
  --persistence-loss-weight 1.0 --persistence-recall-weight 1.0 --cldice-iters 5 \
  --persistence-warmup-epochs 300 --persistence-classes auto \
  --amplify-weight 1.0 --amplify-form hinge --amplify-margin-target 1.0 \
  --amplify-persist inverse_thickness --island-dilate-px 1 \
  --structured-init --structured-init-include-lane \
  --lane-prior-phi1 --lane-prior-phi1-mode replace --lane-prior-phi1-dash-gate \
  --accum-pairs 8 --grad-clip 1.0 --ema-decay 0.997 \
  --lr 1e-3 --lr-end 1e-4 --weight-decay 1e-4 --adam-beta2 0.999 \
  --ckpt-every 25 --stage-checkpoints
```

**S-NEUTRAL speed (lexicographic-secondary, bit-identical → S unchanged):** `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` (16.9× banked, verify `custom_grouped_backward active=true`); K=1 (pair-batch MEASURED NEGATIVE); `--async-verdict --eval-every 25`; fp32. **metal-VJP verdict (`840e39a56` p2b_metal_vjp):** fully-fused metal transpose VJP for fused-R, **S-NEUTRAL + determinism-clean (bit-identical to numpy authority, atol=0, atomics-free)**, 3.06× fwd+bwd / 5.12× bwd-only vs pure-MLX — a candidate S-neutral render-path lever whose #205 wire-in is **Phase-2b-audit-gated (NOT yet on the #205 render path); negligible #205 impact; KEEP for #214 inflate** (decode parallelization, sister of the 10.8× bit-exact inflate `db264bb2f`). The island/persistence Metal kernels are NOT-YET-BUILT (do NOT set `TAC_MLX_CUSTOM_ISLAND_BIRTH`/`_PERSISTENCE_POOL`; the `mx.compile`'d path is the authority). `--compile-step` remains a NON-flag (grep empty) → wire-in + parity-gate is a FUTURE S-neutral ~5% win.

**Launch (operator-gated, NOT this pass):** `.venv/bin/python tools/launch_witness_run.py --gt-cache … --num-pairs 600 --epochs 1000 --all-levers` reproduces the CAPSTONE all-levers config (mod-dim 19 / β₂ 0.9999999 / w-pose 0) — the shippable argv above HAND-DIVERGES on the 4 deltas (a hand-assembled bash `launch.sh`, or a small `witness_autoconfig` extension). If the Muon finisher is still descending @ep1000, EXTEND from the ep1000 ckpt (`--resume-from`, warm-start, free) rather than committing to 1500 upfront — reaches ≤ the same d_seg, training-time-efficient.

---

## 5c. THE CURRICULUM SCHEDULE AS A CALIBRATED OPTIMAL-CONTROL PROGRAM (operator 2026-07-02: co-equal with lever SELECTION)

**Frame:** the schedule is the CONTROL TRAJECTORY `u(t)` of the level-set annealing flow (CE→tau→Muon = coarse-to-fine curvelet scale = Morse-Smale persistence order = temperature anneal). Objective (lexicographic): **minimize terminal d_seg (reach 0.00086) in minimal wall-clock, s.t. the flow dynamics + critical-slowing at transitions.** NOT a fixed proven-fraction pin — a calibrated program. **It ALREADY IS a first-class object:** `tac.witness_dsl.WitnessProgram` (`stages: tuple[Stage,…]` = the ORDER, `.with_lever(Lever(start_epoch=…))` = the STACKING, `Anneal`/`Freeze` = intra-stage, `.compile_trainer_argv()`→ the §5b argv, `.validate()` refuses invented flags) + `tac.witness_dsl.campaign` (`decide_next_stage` / `plan_adaptive_step` / `Cycle` = the CLOSED loop). I express the finalized schedule as that program; I do NOT reinvent.

### (a) The coupling-ordered stage DAG (a partial order — the §3 couplings collapse the permutation explosion; each edge justified)

```
[INIT ep0]  self-orient DIRECTIONAL basis + structured-init + lane-prior-φ1 + palette-anchor  (rule-118 FREE priors seed the coarse partition)
   │ edge: basis-BEFORE-capacity (§3-3: capacity on isotropic basis HURTS +6%, on directional pays −64%)
   │ edge: representation-BEFORE-dynamics (§3-4: the chart/floor is set before any optimizer dynamics)
   ▼
[S1 CE  ep0–300]  coarse partition formation (softmax-temp 1.0)
   │ edge: tau-BEFORE-Muon (§3-5: the softplus surrogate must be REACHABLE before orthogonalized conditioning)
   ▼
[S2 tau_softplus(0.3)  ep300–726]  THE primary drop (softmax-temp anneal 1.0→0.05)
   ├─ branch: persistence-loss + island-amplify (warmup@300, ramp S2→S3)   edge: finest-scale erasure long-tail EMERGES as the partition sharpens (late-emergent → late-engaged)
   ├─ branch: lane-render-band (@300)                                       edge: the witness-uncertainty FP gate is only VALID once the partition is formed (§2A)
   ├─ branch: pose-carrier dξ (trains throughout, w-pose>0)                 edge: seg⊥pose FREE (§3-1); d_pose descends free once rendered
   └─ (l7 DEMOTED — parked @epochs)                                         edge: l7 = L∞ sharpening inside a viscosity flow = d_seg-DECOUPLING defect (§3-5, eq l7_linf_sharpening_defect)
   ▼
[S3 Muon  ep726–1000]  conditioning finisher on the FORMED partition (tau+temp FROZEN 0.05)   edge: §3-5 Muon orthogonalizes on a formed partition; AdamW grad-norm COLLAPSES on the κ~19 Hessian
   ▼
[CLOSED-LOOP GATE @ep1000]  campaign.decide_next_stage → EXTEND / ADVANCE / RERUN_NEW_CONFIG / ROLLBACK_BRANCH
```

### (b) The lever × stage STACKING matrix (coupling-justified per cell)

| lever | INIT | S1 CE (0–300) | S2 tau (300–726) | S3 Muon (726–1000) | justification |
|---|:--:|:--:|:--:|:--:|---|
| self-orient directional basis | ● | ● | ● | ● | basis-BEFORE-capacity (§3-3); −48%/−31% |
| structured-init + lane-prior-φ1 + palette-anchor | SEED | — | — | — | rule-118 FREE coarse seed; separatrix residual 1.9e-5 |
| chroma | ● | ● | ● | ● | SegNet argmax on RGB → chroma is a d_seg lever |
| AA render | `none` | `none` | `none` | `none` | Wave D: supersample DISQUALIFIED (−49% + decode/mismatch) |
| CE loss | — | ● | — | — | coarse partition formation |
| tau_softplus(0.3) | — | — | ● | frozen | primary drop; Δ_min≈0.3 reachability floor |
| softmax-temp | 1.0 | 1.0 | anneal→0.05 | frozen 0.05 | deterministic annealing (cosine) |
| lane-render-band (witness-gated) | — | — | ●@300 | ● | uncertainty gate needs a FORMED partition |
| persistence-loss + island-amplify | — | warmup ramp | ●@300 | ● | **finest-scale erasure long-tail = the late-emergent ~0.003→0.00086 gap-closer** |
| pose-carrier dξ (w-pose>0) | init | ● | ● | ● | seg⊥pose FREE; descends free when rendered |
| Muon finisher | — | — | — | ●@726 | conditioning on FORMED partition (§3-5) |
| margin-saliency / UniWARD | — | — | — | (late-slot, DEMOTED) | texture regime; OFF attribution-clean first → warm-start re-treat |
| EMA | 0.997 | 0.997 | 0.997 | 0.997 (finisher 0.9995 optional) | shadow ships (non-neg) |

### (c) Calibrated epoch allotment (from per-stage DESCENT DYNAMICS, NOT proven-fraction-scaled)

| stage | epochs | Δd_seg realized through-R (MLX-rs/CPU-adv advisory) | calibration evidence |
|---|:--:|---|---|
| **S1 CE** | 0–300 (300) | 0.01045 → 0.005443 (−0.0050) | the coarse-formation knee; the partition forms in ~300 ep |
| **S2 tau_softplus** | 300–726 (426) | 0.005443 → 0.004563 (**−0.000879 = THE primary single drop**) | tau-knee = the largest single-stage drop; held to 726 (the proven Muon-start) so tau saturates before conditioning |
| l7 | DEMOTED (parked @1000, ≤1 trailing ep) | (L∞ defect: decouples d_seg) | eq `l7_linf_sharpening_defect` + the 5-agent pass |
| **S3 Muon** | 726–1000 (274) | 0.004563 → 0.003718 (−0.000569, **STILL DESCENDING/decelerating @ep800**) | long900 @ep800 = 0.002176 still descending → **EXTEND-eligible (the (e) closed loop decides, not a pin)**; overfit-onset ~ep400 is CAP-config-specific (h192 overfits), NOT h96 |

Total ~1000 ep × ~54 s/ep ≈ **~15 h** (T = informational secondary; NOT an S-term). Allotment is the OPENING trajectory; (e) governs the terminal length.

### (d) Intra-stage + inter-stage (transition) controls

**Intra-stage** (`Anneal`/`Freeze` + the schedule flags): LR `1e-3→1e-4` (`--lr-schedule`, `--warmup-epochs`, `--anneal-epochs`); softmax-temp `1.0→0.05` (`--tau-anneal-shape cosine`, FROZEN 0.05 through Muon); tau_softplus_tau `0.3` (`--tau-softplus-tau`; the seg-surrogate Δ_min≈0.3 reachability floor); hosc-β `1.0→4.0` (`--hosc-beta-anneal linear`; the drift-fix — fixed-β4 diverges); max-bank-freq `64` (curvelet stem-Nyquist; the 16→32→64 warm-safe climb is an INTER-LAUNCH escalation, fixed within a launch).

**Inter-stage (transition)** — critical-slowing-aware: `--stage-transition-rewarmup-epochs 8 --stage-transition-rewarmup-floor 0.1 --stage-transition-rewarmup-shape linear --stage-transition-reset-moments` (partial restart; MEASURED — full 1.0× re-destabilizes; flush stale AdamW moments at each boundary). The tau→Muon switch ALSO re-treats via a fresh MultiOptimizer (bit-faithful finisher continuation on resume, `2ca1726ae`). **MD-Decoupling** (`--optimizer md --md-base {adam,muon}`) = a stable-BY-CONSTRUCTION alternative to reheat, BUT SUBSET-MISS (wired in the BASE trainer only, not the levelset entry) → DEMOTED ablation arm.

### (e) The ENGAGED dynamical (closed-loop) control — the real optimality (NOT blind open-loop)

The §5b fixed schedule is the **OPENING trajectory, not a pin.** Two facts MAKE the loop closeable: `--stage-checkpoints --ckpt-every 25` (every stage independently resumable) + `--verdict-pairs 0 --async-verdict --eval-every 25` (all-600 realized-d_seg verdicts, bit-identical training). The closed loop is `tac.witness_dsl.campaign` (BETWEEN-launch, `--resume-from` warm-start — CONTAINMENT: never auto-fires heavy GPU; operator-gated):
- **`decide_next_stage`** (#188 early-stop policy; trailing window=4, `plateau_abs_slope 1e-6`, `descend_slope -1e-5`, extend/advance/rerun window 300): at the ep1000 Muon boundary → **EXTEND** (steep-negative slope, still descending — the long900 evidence) · **ADVANCE** (plateau at/below floor + reheat) · **RERUN_NEW_CONFIG** (plateau but best > rerun_floor → sharper same-stage) · **ROLLBACK_BRANCH** (regressed vs own best → roll to BEST ckpt + branch). Deterministic + recorded → bit-faithful replay.
- **`Cycle` / `expand_cycles`** = the cyclic-stage RECURSION (Muon-priming: re-enter a stage for `cycles[i].window` more ep) — the "recursive" facet.
- **curvelet-scale climb** (`--max-bank-freq` 16→32→64) = the warm-safe inter-launch escalation (shape-changing → FRESH arm per `plan_adaptive_step`; loss/projection levers land as warm re-treatments).

### Schedule-OPTIMIZATION open questions for Phase-3 (DERIVED-now vs NEEDS-TRAJECTORY-MEASUREMENT)

1. **tau→Muon boundary @726** — DERIVED from the proven fraction; the OPTIMAL boundary needs a per-stage-ckpt A/B (measure d_seg vs muon-start-epoch). NEEDS-TRAJECTORY.
2. **Muon length (274 ep)** — long900 still descending @ep800; RESOLVED DYNAMICALLY by (e) EXTEND, not pinned. DERIVED-now (the policy), measured-at-run.
3. **persistence/amplify warmup@300 + ramp SHAPE** — start=tau DERIVED; the linear ramp is a default, no measured optimum. NEEDS-TRAJECTORY.
4. **l7 DEMOTE vs small-drop** — MLX-trace −0.00027 vs the 5-agent DEFECT verdict; the through-R A/B at the tau-converged ckpt resolves it. NEEDS-TRAJECTORY.
5. **reheat floor/shape per boundary (0.1×/8ep)** — MEASURED partial-restart; the exact floor/shape per transition is calibratable. DERIVED-now, refine-measurable.
6. **closed-loop thresholds (`plateau_abs_slope 1e-6`, `descend_slope -1e-5`)** — DERIVED defaults; they must calibrate to the n600 verdict-NOISE floor (needs the verdict-variance measurement so a plateau isn't called on noise). NEEDS-TRAJECTORY.
7. **stage ORDER robustness** — the DAG edges are coupling-DERIVED (basis-before-capacity etc.); a single-swap ablation (e.g. tau-before-directional) would CONFIRM the partial order empirically. NEEDS-TRAJECTORY (low-priority; couplings are measured).

---

## 6. THE WIRING-GAP LIST (coordinator requirement — grepped against the LIVE levelset trainer)

> **PHASE-2 UPDATE (2026-07-02, HEAD `e28cbab63`): #224 CLOSED both BLOCKING gaps + the 3 high-value + the β₂ optional.** The two "MODULE-ONLY BLOCKING" gaps below — **(A) warp-real-luma pose carrier** and **(B) AA-SDF render** — are now FULLY WIRED with real flags reaching render/loss (callsite-traced): `--pose-carrier` (build+child-attach+render-dispatch+NO-FAKE-verdict, requires `--w-pose>0`; the FEED-224 fail-closed guard REPLACED) + `--render-aa {none,supersample,ipe}` / `--aa-supersample`. The 3 high-value (`--lane-render-band`, `--persistence-loss-weight`, `--amplify-weight`) + `--seed-islands` + `--adam-beta2` are ALSO wired. Residual gaps: **MD-optimizer into levelset** (still base-only subset-miss) + **`--compile-step`** (S-neutral ~5%, still NON-flag). The AA reconcile flipped the (B) verdict: `--render-aa none` + `--lane-render-band`, NOT supersample (Wave D, §5b delta 1). The list below is the Phase-1 as-of-derivation snapshot (PRE-#224).

Every lever's state vs `experiments/train_levelset_witness_realized_through_R_mlx.py` argparse + body (the base's flags are a SUBSET — the subset-miss is real and bit us before).

**WIRED (real grepped flag reaching render/loss — usable in the argv NOW):**
- mod-dim / hidden-dim / n-hidden ✓
- activation hosc + `--siren-init` + `--hosc-beta`/`--hosc-beta-end`/`--hosc-beta-anneal` (β-anneal 1→4, drift-resolved) ✓
- curriculum: `--curriculum` `--tau-softplus-start-epoch` `--l7-start-epoch` `--muon-start-epoch` `--muon-lr` (full Muon group) ✓
- self-orient directional/curvelet basis + `--bank-*`/`--n-dir-freqs`/`--freq-across`/`--freq-along`/`--max-bank-freq` ✓
- `--chroma` `--palette-anchor` ✓
- structured-init + `--lane-prior-phi1` (openpilot deg-3 centerline SDF, S0 seed) ✓
- margin-saliency LEVER-4 + UNIWARD; lane-edge LEVER-3; lane-thin LEVER-B (birth-death dashes); hardness LEVER-5 ✓
- `--eikonal-weight` `--length-weight` `--code-nuclear-weight` ✓
- `--film-stiefel` `--code-spectral-entropy-weight` (DM1 byte-free) ✓
- `--head etf/additive-margin` `--margin-field-head-weight` `--logit-adjust-per-class` (#218) ✓
- EMA `--ema-decay` + `--ema-decay-finisher` (SWA); REHEAT `--stage-transition-rewarmup-*` + `--stage-transition-reset-moments` ✓
- `--residual-mode` (v2 hybrid — WIRED but NO-GO, keep OFF) ✓
- `--render-h/w` `--accum-pairs` `--grad-clip` `--spike-factor` ✓
- `--w-pose` (the pose-term gate; scaffold present) ✓ but see carrier gap below
- MLX perf env `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` ✓ (env, in `to_command`)

**MODULE-ONLY — NEEDS WIRE-IN (module built in `src/tac/boundary_math/` or `tac.optimization`, but ZERO live-trainer flag — the #224 gaps; the argv CANNOT use these until wired):**
- ★ **AA-SDF observation render** (`aa_sdf_observation_render.py`, 14.6 K) — **THE #1 representation lever** (d_seg 0.00549→0.00086 @g384, reaches the sub-0.15 floor). No `--render-aa`/`--supersample` flag. **HIGHEST-priority wire-in.**
- **persistence/topology soft-clDice loss** (`persistence_topology_loss.py`, 30.8 K; gauge TopologyLossGauge) — 111× erasure-sensitive, +0.443 island recovery. No loss-weight flag.
- **island seed/contain/amplify** (`island_protection.py`, 40.0 K; gauge IslandProtectionGauge; rides LEVER-4) — LANE 0.56→0.93→0.95. No flag.
- **warp-real-luma frame0 pose carrier** (`warp_real_luma_frame0.py`, 31.0 K) — d_pose 163→1.37, wire-in = parity-dispatch render_fn hook (even code_idx=f0→carrier, odd=f1→witness) + `--w-pose>0`. **The #237 SLOT-POSE depends on this.** No render hook wired.
- **analytic-lane AA-SDF render band** (`analytic_lane_render_band.py`, 21.9 K) — #234; wire-in = `render_through_R_mlx` compose_fn hook + the missing `render_batch_through_R_mlx` hook. Render-band FREE; serializer rate-not-viable naive. No flag.
- **MD-Decoupling optimizer** (`--optimizer md --md-base {adam,muon}`) — **WIRED in the BASE trainer (line 2296), NOT the levelset entry point (SUBSET-MISS).** DEMOTED ablation arm.
- **β₂ override** — AdamW uses hardcoded default 0.999 (`optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)`, line 1235); no `--beta2` flag. #222 sweep needs it.

**PENDING / config-only (controllable via existing flags but needs a wire-in decision):**
- **l7-demote** — the curriculum guard requires `0 < tau < l7 <= epochs`, so l7 can only be parked (l7-start@epochs) not cleanly removed; a clean `--no-l7`/skip-stage wire-in is the tidy fix. (The argv parks it @1499.)
- **stored-pose training-GT + supervision** — `--w-pose>0` supervises toward stored PoseNet targets, but the target-loading path for the SDF witness (frame0-null space) is the #221 fine-tune wiring.
- **`mx.compile` step (S-NEUTRAL speed lever, secondary objective)** — no `--compile-step` flag in either trainer (grep empty); wiring `mx.compile` on the static-shape hot path + the `assert_compile_bit_identical` parity gate is a ~5% FREE win (bit-identical → S unchanged). WIRE-IN before it can enter the argv.

**Wiring-gap summary for gate phase-2b (integration audit, after #237):** the S-optimal SHIPPABLE config depends on **2 BLOCKING wire-ins** — **(A) warp-real-luma pose carrier (#237/SLOT-POSE, else d_pose~190 catastrophe)** and **(B) AA-SDF observation render (the #1 d_seg lever to the floor)** — plus **3 high-value non-blocking** wire-ins (persistence loss, island protection, analytic-lane band) and **3 optional/secondary** (MD-optimizer into levelset, β₂ flag, `mx.compile` S-neutral speed lever). The argv in §5 is runnable TODAY as the **attribution-clean d_seg-isolation leg** (w_pose=0, AA-render off) but is NOT the shippable S-optimal row until (A)+(B) land in #224. Ordering per the representation-before-dynamics coupling: wire **(B) AA-render FIRST** (it sets the d_seg floor), then **(A) the pose carrier**, then the 3 high-value loss/render levers.

---

## 7. THE S-BUDGET (derived, with the binding uncertainties honest)

At optimal form, composing the measured/derived surfaces:

| term | value | basis | confidence |
|---|---|---:|---|
| **d_seg** | 0.00077–0.00118 (sub-0.15 need) reachable; representation floor 0.00086; current best 0.00245 | AA-render + Muon-conditioning + directional basis reach the floor | **TRAINING gap (reach-the-floor), floor MEASURED below target** |
| 100·d_seg | 0.077–0.118 | | |
| **d_pose** | ε ≈ 0.018 (term) IF the carrier residual closes; warp-alone 1.37 (term 3.70) is NOT enough | warp-real-luma + trained dξ residual, `--w-pose>0` | **UNMEASURED on witness (#237/#221) — the largest open uncertainty** |
| √(10·d_pose) | ~0.018 (target) | | |
| **rate** | 0.055 (measured byte-close) – 0.081 (RD-optimum) | int8+brotli witness weights + pose ξ 2.4 KB + lane slot | **MEASURED (byte-close 83 KB)** |
| 25·B/N | 0.055–0.081 | | |
| **S (predicted, optimal-form)** | **~0.13 – 0.15 first-row band; ~0.134 at the RD-optimum (directional-ON)** | sum | **DESIGN prediction; MEANS≠ends — only a byte-closed exact row is a score** |

**The three binding uncertainties that gate the prediction:** (1) realized directional d_seg (−48% is circular-GT; self-orient fixed-point UNVERIFIED); (2) witness pose closure (3.4e-5→0.018 is ancestor-anchored, UNMEASURED); (3) the α≈2.34 RD-curve is borrowed from 2 points. All three resolve into the #205 GPU run + #221 pose FT + #223 byte-close sweep.

---

## 8. OPEN QUESTIONS for the 3-CLEAN-PASS RECURSIVE ADVERSARIAL REVIEW (Phase 3)

1. **l7 reconcile:** MLX-trace shows l7 lowering d_seg (−0.00027) but the 5-agent pass calls it a defect. Is DEMOTE right, or is l7 a small-but-real drop we're discarding? (Resolve: measure l7-on vs l7-off through-R at the tau-converged ckpt, T2.)
2. **Pose SLOT (#237) — is w_pose=0 first-launch legitimate?** The attribution-clean d_seg-isolation leg runs pose-blind, but then the FIRST byte-closed row has d_pose~190 (S catastrophic). Should the first shippable run wait for the carrier wire-in (A), or is a d_seg-only advisory leg acceptable as a gate? (This is the means/ends firewall — a d_seg-only row does NOT move the pointer.)
3. **AA-render wire-in (B) sequencing:** it's the #1 d_seg lever (reaches the floor) but MODULE-ONLY. Should #205 launch WITHOUT it (leaving d_seg ~0.0024, S~0.16) or block on the wire-in? (The representation-before-dynamics coupling says wire AA-render FIRST — it's the floor-setter.)
4. **mod-dim (#223):** 32 (proven) vs 26 (review, −0.004 S) vs 19 (Whitney floor). Is the byte-close sweep worth a $0 pass before launch, or launch at 32 and fold at warm-start?
5. **β₂ (#222):** derive-T0 says a higher β₂ floor from batch-n; but MD-Decoupling under-steps at adamw-lr. Is β₂ a real lever or an optimizer-vs-representation red herring? (The #222 sweep is the disambiguator.)
6. **epochs = 1500:** proven arm was 1000 (best 0.003698). Is 1500 enough to reach the 0.00086 floor, or does the Muon finisher need more (long900 still descending @ep800)? Wall-clock ~22.5 h vs a longer run.
7. **Directional basis realized verdict:** the −48% is circular-GT. Does the self-orient fixed-point actually deliver it through R? (The #1 d_seg-lever's realized magnitude is UNVERIFIED — a $0 numpy de-risk on `generator_n600.npz` precedes GPU.)
8. **The `--async-verdict` bit-identity:** confirmed bit-identical training (verdict never read back); confirm the CPU-torch verdict bottleneck doesn't stall the multi-day run.

---

## HONEST GAP LEDGER (DERIVED vs NEEDS-MEASUREMENT vs PENDING-SLOT)

| item | status |
|---|---|
| Objective + marginals + KKT | **DERIVED (exact arithmetic)** |
| d_seg per-stage / AA-floor / capacity-after-basis | **MEASURED (MLX-rs/CPU-adv advisory)** |
| d_seg realized directional −48% | **NEEDS-MEASUREMENT (circular-GT; self-orient de-risk)** |
| Muon-final d_seg 6–9e-4 | **PENDING (predicted, the binding open cell = #205 GPU run)** |
| d_pose warp-real-luma 163→1.37 | **MEASURED (advisory); residual→0.018 UNMEASURED (#237/#221)** |
| rate byte-close 83 KB / 0.055; RD-optimum 122 KB | **MEASURED byte-close; curve DERIVED-from-2-anchors (α load-bearing)** |
| residual-mode NO-GO | **MEASURED (decisive; keep OFF)** |
| training dynamics (Muon/EMA/reheat/two-temps/wall-clock) | **MEASURED/GROUNDED** |
| β₂ / MD-Decoupling-into-levelset | **PENDING (β₂ derive-T0 + #222 sweep; MD subset-miss)** |
| analytic-lane #234 net-S | **PENDING (T5 gate; render-band FREE, serializer needs AR-coding)** |
| mod-dim #223 | **PENDING (byte-close sweep; default 32)** |
| candidate argv | **DESIGN (grepped-real flags; 2 blocking wire-ins A+B before shippable)** |

**FINALIZATION:** this derivation is FINALIZED after (a) #234 net-S lands (analytic-lane slot resolves), (b) #237 pose-reconcile lands (SLOT-POSE resolves: warp-real-luma carrier vs sidecar), (c) #223 parametrization byte-close sweep lands (mod-dim resolves). Phase-2 = MLX/Metal wall-clock verify (confirm `TAC_MLX_CUSTOM_GROUPED_BACKWARD` active + K=1 + s/ep at n600 h96). Phase-3 = the 3-clean-pass recursive adversarial review of the argv + §8 open questions. **Pointer 0.19110 UNMOVED — this is a MEANS; the END is a byte-closed n600 exact row below it.**
