---
title: CONCENTRATED-SALIENCY VEHICLE — $0 design + feasibility probe (capstone #78 design phase)
authority: "[contest-CPU advisory] — NON-PROMOTABLE; exact pointer UNMOVED at 0.19110"
score_claim: false
promotable: false
frontier_pointer_moved: false
mission_contribution: frontier_breaking_enabler
date: 2026-06-18
supersedes: none
cross_refs:
  - .omx/research/campaign_inflection_three_paths_capped_concentrated_saliency_20260618.md
  - .omx/research/campaign_math_review_dynamics_and_optimization_20260618.md
  - .omx/research/frontier_margin_saliency_qat_bitalloc_prior_20260618T183000Z.md
  - .omx/research/frontier_int5_score_aware_qat_finetune_20260618T211958Z.md
  - .omx/research/label_noise_floor_RESOLUTION_frontier_existence_proof_20260618.md
  - .omx/research/SESSION_SYNTHESIS_SoT_20260617_20260618.md
---

# Concentrated-saliency vehicle — $0 design + feasibility probe (capstone #78)

**The decisive $0 gate the inflection memo demanded, run.** All numbers `[contest-CPU
advisory]` NON-PROMOTABLE; the exact pointer is UNMOVED at 0.19110 — stated plainly per
the GOAL firewall: **this unit did NOT move the pointer.** Its value is a measured GO/NO-GO
that re-routes the sub-0.15 search, plus a ranked design + a gated build spec for the winning
direction. $0: local CPU authority + a short MPS fp32 gradient smoke; no paid GPU, no PR.

The capstone (#78 "our own small learned basis, extremely scorer-optimized") rests on ONE
load-bearing assumption: that a decoder's d_seg-criticality can be **concentrated** into a
small high-precision byte-cheap CORE so the rate-win (shed the periphery) and the d_seg-floor
(protect the core) stop fighting. This unit tested whether that concentration is even possible,
and ranks the mechanisms that could achieve it.

---

## 0. The headline (read this first)

- **The d_seg signal is NATURALLY concentrated in PIXEL space** (frontier per-pixel saliency
  Gini 0.50, boundary/interior 2.23×, 4.1% of pixels carry 8.7% of the margin-gradient mass;
  bc20 boundary ratio 3.15×). The detector decides on a thin perimeter — the criticality lives
  on the SegNet decision boundary, exactly as the campaign established (d_seg = the argmax-flip
  rate ≈ a perimeter integral, 882× concentrated at margin<0.5).
- **But a DENSE decoder SPREADS that pixel-concentration across its WEIGHTS** (frontier
  per-weight-tensor saliency spread 5.5×, Gini ~0.27; bc20 spread 7.2×, Gini 0.276;
  decision-band weight-mass fraction only 0.44, geometry-ordering FALSE — the coarse early
  stages carry as much margin-saliency as the late decision-band stages).
- **THE FEASIBILITY PROBE (decisive, two configs): you CANNOT concentrate the per-weight
  criticality by a regularizer on a dense decoder.** A penalty that drives the periphery's
  margin-gradient norm down 10–12× moves the periphery's saliency-MASS FRACTION by only
  0.2–0.4% and REGRESSES d_seg (+30% to +84%). The penalty shrinks the gradient magnitude by
  DAMAGING the decoder, not by redistributing criticality. **The spread is intrinsic to the
  dense, fully-shared rendering path** (every stage couples into the boundary via the
  bilinear-skips + sin — the Yousfi-seam coupling the frontier #141 memo already flagged).

**VERDICT (nuanced):**
- **NO-GO** for the **saliency-concentrating REGULARIZER on a dense decoder** (capstone
  candidate #2). Confirmed twice, configs robust. This direction is dead.
- **GO (conditional)** for the **architecturally-FACTORED vehicle** (candidate #1) — but with a
  CORRECTED rationale: the concentration must be **structural** (a separate, narrow LF/structure
  sub-network whose capacity is physically distinct from the HF/detail periphery), NOT a penalty
  on a shared dense path. The probe rules out "make the dense path concentrate"; it does NOT rule
  out "build a path that is concentrated by construction." This is the surviving capstone design.
- The deeper implication: because d_seg-criticality is pixel-boundary-bound and the dense decoder
  diffuses it, **sub-0.15 likely needs a representation where the boundary is carried by a
  distinct, compact sub-module** — at the extreme, a non-dense (parametric/geometry) boundary
  core. That re-routes the search toward the factored and geometry-anchored candidates, away
  from any "regularize the existing decoder" hope.

---

## 1. The feasibility probe (the load-bearing measurement)

**Producer:** `experiments/probe_concentrated_saliency_feasibility.py` (resumable, $0).
**Real components (NO-FAKE):** the bc20 basin decoder (83,356 params, byte-exact
`ema_decoder`), the real frozen contest SegNet (CPU), real GT argmax hard labels
(`gt_targets_n100.pt`), real autograd of the SegNet top1−top2 margin through the decoder
weights. d_seg measured on the EXACT eval path (decoder pair → uint8 roundtrip
bicubic-874→bilinear-384→round → SegNet scores last frame x[:,-1] → argmax-flip-rate vs GT) on
a 24-frame subset — a faithful-DEFINITION proxy (validated: subset d_seg 0.00256 vs the basin's
reported full-600 d_seg 0.0026, ~1% agreement).

**The mechanism tested:** designate a small CORE (top-k most-d_seg-critical weight tensors by
baseline saliency) + a PERIPHERY (the rest); short MPS fp32 train with
loss = ce_weight·CE(scored-frame, GT) + reg_weight·‖∂margin/∂(periphery weights)‖² ; re-measure
the per-weight saliency Gini / periphery-mass and d_seg. If criticality concentrates (Gini up,
periphery mass down) at ~flat d_seg → GO; if the periphery cannot be made blind without raising
d_seg → NO-GO.

### Results (two configs, 150 ep each)

| config | core | core %params | reg_w | ce_w | periph_sal penalty | periph saliency-mass | Gini | d_seg(subset) | verdict |
|---|---|---:|---:|---:|---|---|---|---|---|
| A | top-3 tensors | 54.7% | 1.0 | 3.0 | 1.53 → 0.15 (**10×↓**) | 67.2% → 66.7% (−0.4%) | 0.276 → 0.290 (+0.014) | 0.00256 → **0.00472 (+84%)** | NO-GO (no-conc + d_seg regress) |
| B | top-6 tensors | 92.7% | 0.3 | 5.0 | 0.73 → 0.06 (**12×↓**) | 38.0% → 37.7% (−0.2%) | 0.276 → 0.282 (+0.006) | 0.00256 → **0.00334 (+30%)** | NO-GO (spread intrinsic) |

**The invariant finding across both:** the penalty achieves its OWN objective (periphery
margin-gradient norm down 10–12×) but the **saliency-MASS FRACTION is invariant** (moves <0.5%)
and **d_seg always regresses.** Reducing the periphery gradient norm degrades the periphery's
contribution to the rendered boundary → MORE flips → higher d_seg → the global margin shrinks
everywhere → the *fraction* is unchanged. There is no "spare" periphery whose criticality can be
relocated: every weight tensor's saliency is the boundary it helps render, and the boundary is
rendered jointly.

### Why this is a SOUND verdict, not a premature kill (per "Forbidden premature KILL")

- **Two independent configs** (small vs large core; aggressive vs gentle reg; weak vs strong CE)
  converge on the same invariant. The result is not a single-config artifact.
- It is grounded in an **operating-point-INVARIANT quantity** (the per-pixel→per-weight diffusion
  is a structural property of the shared rendering path, confirmed independently on BOTH the bc20
  basin AND the frontier vehicle by the #141 saliency memos — spread 5.5–7.2×, geometry-ordering
  FALSE on both). Per the campaign META-pattern, invariant-grounded closures hold; choice-grounded
  ones are where errors concentrate. This is invariant-grounded.
- The closure is **scoped precisely**: it kills the REGULARIZER-on-a-dense-decoder, NOT the
  factored-architecture or geometry-core candidates (those change the structure, not just the
  loss). The reactivation criterion for the regularizer is explicit (below).

---

## 2. RANKED DESIGN — the concentrated-saliency mechanisms (sub-0.15-EV × feasibility)

The score function S = 100·d_seg + √(10·d_pose) + 25·B/B₀ governs everything (master gradient:
∂S/∂d_seg = 100 constant; ∂S/∂d_pose = 5/√(10·d_pose) ≈ 86 at the operating point; ∂S/∂B =
6.66e-7/byte). The design goal: a vehicle where the 100·d_seg term is carried by a small
high-precision core (so it floors low) AND the 25·B/B₀ term is dominated by a cheap periphery (so
it shrinks) — decoupling the capacity↔rate tension that capped all three prior paths.

### RANK 1 (GO — the surviving capstone) — FACTORED decoder: structural LF d_seg-core + cheap HF periphery

**Mechanism:** split the decoder into two physically distinct branches. A **narrow LF/structure
core** renders the coarse class-region geometry at the SegNet decision grid (≥192×256 post
stride-2 stem) — this carries the 100·d_seg term, stays high-precision, and is *small* (few
channels at the decision band). A **wide HF/detail+pose periphery** adds reconstruction texture +
the pose-relevant high frequencies — this is d_seg-blind by CONSTRUCTION (it operates above the
SegNet's Nyquist / outside the decision band), so it can be int4 / 2:4-sparse / low-rank / pruned
for the rate win.

**Why it survives the probe:** the probe killed "make a SHARED dense path concentrate." A factored
vehicle does not share the path — the LF core and HF periphery have separate parameters, so the HF
periphery's d_seg-saliency is structurally near-zero (it is filtered out by the SegNet's stride-2
stem / the decision-grid downsample). The concentration is by construction, not by penalty.

**The math (how it decouples):** S = 100·d_seg(LF core) + √(10·d_pose(periphery)) + 25·B_LF/B₀ +
25·B_HF·q/B₀, where q is the periphery's bits-per-param fraction (int4 ⇒ q≈0.36 vs FP). Because
d_seg depends only on the LF core, you can drive q→0.36 (or prune the HF) WITHOUT spilling d_seg —
the exact thing int5-QAT on the dense frontier could NOT do (it capped d_seg at 0.0035, 6× the
floor, because the d_seg-critical structure was in the dense early stages the coarse grid hit).

**Byte/d_seg model (advisory):** if the LF core is ~25–35% of params at FP and the HF periphery
~65–75% at int4: B ≈ (0.30 + 0.70·0.36)·B_dense ≈ 0.55·B_dense. On the small basis (89KB) that is
~49KB → rate ≈ 0.033. With a LF core sized for frontier-grade d_seg (~0.0004) and pose held at
the trunk value (0.00034): S ≈ 100·0.0004 + √(10·0.00034) + 0.033 ≈ 0.040 + 0.058 + 0.033 ≈ **0.131**
— sub-0.15 on the CPU ranking axis IF the LF core reaches frontier-grade d_seg at small size. That
"IF" is the build's central risk (the capacity question the inflection memo named).

**Real grounding (papers/OSS):**
- **SNeRV** (Kim, Lee, Kang, ECCV 2024, arXiv 2501.01681; `github.com/qwertja/SNeRV`) — DWT-splits
  video, stores only the LF subband, generates HF in the decoder. Almost exactly this design at
  NeRV scale. **Read first.**
- **Octave Convolution** (Chen et al., ICCV 2019, arXiv 1904.05049; `github.com/d-li14/octconv.pytorch`)
  — the cleanest droppable conv primitive to build an LF-at-half-res / HF split into the existing
  HNeRV decoder without a rewrite. **Most droppable-in.**
- **MWCNN** (Liu et al., 2018, arXiv 1805.07071; `github.com/lpj0/MWCNN`) — invertible DWT/IDWT
  U-Net; makes the HF subbands genuinely sparse/compressible (the periphery-shed mechanism).
- **LapSRN** (Lai et al., CVPR 2017, arXiv 1704.03915) — the canonical "coarse base = structure
  core + light HF residual branches" pyramid.
- Risk: the existing HNeRV bilinear-skip+sin coupling is the very thing that diffuses saliency
  (probe-confirmed); the factored design must SEVER that coupling for the LF core (no skip from the
  HF branch into the decision band), else it re-diffuses.

**Build cost:** medium (a new decoder architecture + a from-scratch or partial-transfer train; the
LF/HF split is a structural change, not a bolt-on). **EV × feasibility: HIGHEST.**

### RANK 2 (GO-adjacent, the rate-side enabler) — MIXED-PRECISION periphery via HAWQ-trace bit-allocation + 2:4 sparsity

**Mechanism:** given a factored (or even the existing dense) decoder, allocate weight bits by a
task-sensitivity score (Hessian trace per HAWQ-V2, or the margin-saliency map #141 we already
have), then crush the lowest-sensitivity tensors to int4 + 2:4 semi-structured sparsity, with QAT
to train robustness to the grid.

**Why it is RANK 2 not RANK 1 alone:** on a DENSE decoder this is exactly Path B, which CAPPED
(int5-QAT d_seg plateau 0.0035, the saliency is too flat to find cheap-bytes-at-zero-d_seg — #141
measured only 5.5× spread, so the int4 budget gain from coarsening the "blind" tensors is modest).
It only becomes powerful ON TOP OF a factored vehicle, where the HF periphery is genuinely blind
(then int4 + 2:4 on the periphery is a clean ~50–64% byte cut at ~0 d_seg cost). **It is the
periphery-shed actuator for RANK 1, not a standalone win.**

**Real grounding:** HAWQ-V2 (Dong et al., NeurIPS 2020, arXiv 1911.03852; `github.com/Zhen-Dong/HAWQ`)
— Hessian-trace sensitivity → Pareto bit allocation. HAWQ-V3 (arXiv 2011.10680) — ILP allocation
under a hard byte budget (use when the rate target is fixed). NVIDIA 2:4 sparsity (Mishra et al.,
arXiv 2104.08378; PyTorch `torch.sparse.to_sparse_semi_structured`) — deterministic 50% on the
periphery. We ALREADY HAVE the score-aware QAT codec (`tac.torch_vehicle.score_aware_qat`,
`per_tensor_levels_from_sensitivity`) + the #141 saliency map — this is wired, just mis-targeted
(it was applied to a dense decoder). **EV × feasibility: HIGH (but gated on RANK 1 landing first).**

### RANK 3 (GO, the periphery compressor) — LOW-RANK / structured periphery (LoRA-style + Monarch)

**Mechanism:** parameterize the HF/detail periphery as a low-rank or structured-matrix factorization
(rank-r conv/linear factors, or Monarch/butterfly blocks) so it stores O(r·n) instead of O(n²) bytes
natively, while the LF core stays full-rank high-precision.

**Why RANK 3:** complementary to RANK 1+2 (it is another way to make the periphery cheap), but it is
a periphery-compression technique, not the concentration mechanism itself — it presupposes the
factored split. Native low-rank design (Tai et al., arXiv 1511.06067) is cleaner than retrofit.

**Real grounding:** LoRA (Hu et al., arXiv 2106.09685; `github.com/microsoft/LoRA`) — the canonical
"frozen high-precision core + cheap low-rank delta periphery" map. Monarch (Dao et al., ICML 2022,
arXiv 2204.00595; `github.com/HazyResearch/fly`) — closed-form dense→structured projection,
GEMM-friendly. Tai et al. (arXiv 1511.06067) — train low-rank conv from scratch. **EV × feasibility:
MEDIUM (a periphery option for the RANK-1 build, not standalone).**

### RANK 4 (HIGH-RISK class-shift, the deepest decoupling) — GEOMETRY-ANCHORED parametric d_seg-core

**Mechanism:** replace the implicit pixel-rendered boundary with an explicit low-dimensional
parametric core — the road↔lane boundary is a handful of Bézier/polyline control points (d_seg is
64% road↔lane markings per the campaign), differentiably rasterized into the decision-band, with a
cheap learned periphery for everything else. The d_seg-critical capacity becomes literally a few
dozen curve parameters (bytes ≈ 0), maximally concentrated.

**Why RANK 4 (last, despite the highest decoupling):** highest EV (the core is near-free bytes AND
frontier-grade if the boundary is well-approximated by curves) but highest risk + a true class-shift
(a new representation + a differentiable rasterizer + a from-scratch training contract). It is the
direction the NO-GO points at IF the factored dense-conv core (RANK 1) also walls on the capacity
question. The campaign's #137/#138 lane-poly priors are the down-payment on this.

**Real grounding:** LSTR (Liu et al., WACV 2021, arXiv 2011.04233; `github.com/liuruijin17/LSTR`) —
lane = tiny curve-parameter vector. BézierLaneNet (Feng et al., CVPR 2022, arXiv 2203.02431;
`github.com/voldemortX/pytorch-auto-drive`) — lane = Bézier control points. DiffVG (Li et al.,
SIGGRAPH Asia 2020; `github.com/BachiLi/diffvg`) — **the keystone**: differentiable rasterization of
Bézier/polygon control points, backprops a pixel/margin loss into curve params (this is how you wire
the geometry core into the score-aware loop). Curve-GCN (arXiv 1903.06874) for general contours.
**EV × feasibility: HIGH-EV / LOW-feasibility — the contingency if RANK 1 caps.**

### RANK 5 (RULED OUT this unit) — saliency-concentrating REGULARIZER on a dense decoder

The probe's NO-GO. Penalizing periphery margin-saliency does not redistribute criticality; it
damages the decoder. **DEFER (not kill) with explicit reactivation:** re-open ONLY if applied to an
ALREADY-FACTORED vehicle as a *secondary* sharpener (to push residual decision-band saliency from
the HF branch back into the LF core) — i.e. the regularizer may help maintain a structural split, but
it cannot CREATE one on a shared dense path. Real grounding if reactivated: Movement Pruning (Sanh
et al., NeurIPS 2020, arXiv 2005.07683), L0 reg (Louizos et al., ICLR 2018, arXiv 1712.01312), SNIP
saliency (Lee et al., arXiv 1810.02340). **EV × feasibility: dead as primary; LOW as secondary.**

---

## 3. Canonical-vs-unique decision per layer (UNIQUE-AND-COMPLETE-PER-METHOD discipline)

| layer | decision | rationale (falling-rule) |
|---|---|---|
| decoder architecture | **FORK_PRINCIPLED** — new factored LF/HF decoder | the dense HNeRV path STRUCTURALLY diffuses saliency (probe-proven); the whole point is to NOT share that path |
| score-aware QAT codec | **ADOPT_CANONICAL** — `tac.torch_vehicle.score_aware_qat` | already exists + correct; just re-target it at the HF periphery (where it works) not the dense decoder (where it capped) |
| margin-saliency map | **ADOPT_CANONICAL** — `tac.margin_saliency_map` | the real cost map; the keeper asset (boundary ratio 2.2–3.15×); used to size + verify the LF core |
| GT-target / d_seg eval path | **ADOPT_CANONICAL** — exact eval roundtrip + last-frame scoring | validated faithful (subset 0.00256 ≈ full-600 0.0026); do not fork |
| pose carrier | **ADOPT_CANONICAL** — trunk pose + the #140 1-DOF radial-zoom codec | pose is ~free on bytes (1.9% of store) + held stable at 0.00034; the √-derivative punishes variance, so the periphery must NOT route pose through a noisy carrier |
| LF/HF split primitive | **FORK_PRINCIPLED then ADOPT** — OctConv / DWT (SNeRV/MWCNN) | use the published frequency-split primitive (don't reinvent), but the wiring into HNeRV + the SegNet-decision-band alignment is unique to this vehicle |

## 4. 18-shared-assumption profile (the plateau-breaker check — score-relevant assumptions)

| assumption | classification | note |
|---|---|---|
| EMA shadow at inference | ADOPT_CANONICAL | hard-earned; warmup-decay fix already landed |
| eval_roundtrip in training | ADOPT_CANONICAL | mandatory; the d_seg path uses it (probe validated) |
| dense fully-shared decoder | **FORK_PRINCIPLED** | THE class-shift: the probe proves dense diffuses saliency; the factored vehicle violates this assumption deliberately |
| canonical scorer-preprocess (x[:,-1] last-frame) | ADOPT_CANONICAL | the exact metric; the LF core renders the scored frame |
| d_seg lives in capacity (more params) | **FORK_EMPIRICAL** | the probe + #141 show d_seg lives in pixel-BOUNDARY structure that a dense net diffuses; capacity helps only if STRUCTURED at the decision band |
| margin-hinge is the seg lever | **UNCLEAR_NEEDS_EMPIRICAL** | margin-hinge over-sharpens under coarse grids (int5-QAT finding); the factored core may prefer CE or boundary-weighted TCKD |
| rate is at the entropy floor (lossless recode = 0) | ADOPT_CANONICAL | confirmed; the rate win MUST be a lossy/structural model change (the factored int4 periphery) |

(Remaining assumptions inherit ADOPT_CANONICAL; none score-flips this design.)

## 5. Observability surface

- **Inspectable per layer:** the LF core vs HF periphery saliency is `compute_decoder_tensor_margin_saliency` per branch (the probe's `_weight_saliency`); the build must report the periphery saliency-mass fraction (target: HF periphery < 10% of d_seg mass, vs the dense 67%).
- **Decomposable per signal:** S decomposes into LF-core-d_seg + periphery-d_pose + B_LF + B_HF·q — each separately measurable via the exact eval + byte-close.
- **Diff-able across runs:** the feasibility-state JSON schema (`concentrated_saliency_feasibility.v1`) is the before/after concentration diff; the build extends it.
- **Queryable post-hoc:** advisory JSONs in `.omx/research/concentrated_saliency_*`.
- **Cite-able:** (basin_ckpt sha, base_channels, n_params, reported full-600 d_seg) tuple recorded.
- **Counterfactual-able:** the probe IS the counterfactual ("what if we penalize periphery saliency?" → answered: no concentration, d_seg regresses).

---

## 6. GO/NO-GO and the BUILD SPEC (gated — do NOT launch the multi-day build)

**GO/NO-GO:** **NO-GO for the regularizer (RANK 5); GO for the architecturally-factored vehicle
(RANK 1 + RANK 2 periphery-shed).** The feasibility probe re-routes the capstone: the path is NOT
"regularize the existing decoder into concentration" (dead) but "build a vehicle whose LF d_seg-core
and HF periphery are structurally distinct." If RANK 1's small LF core also walls on the capacity
question (cannot reach frontier-grade d_seg at small size), the contingency is RANK 4 (geometry core).

### The RANK-1 build spec (the gated follow-on)

1. **Architecture** (FORK): a two-branch HNeRV. LF/structure branch = narrow channels, renders the
   coarse class-region geometry at ≥192×256 (the SegNet decision grid), no skip from the HF branch
   into it (sever the saliency-diffusing coupling). HF/detail branch = wide, renders the 384×512
   residual texture + pose-relevant HF. Sum at output. Build via OctConv or a DWT LF/HF split
   (SNeRV/MWCNN primitive). Total params target ≈ the bc20 basis (≈83K) but reallocated: ~30% in the
   LF core (high-precision), ~70% in the HF periphery (int4-able).
2. **Training (score-aware, eval-roundtrip):** the LF core trained against 100·d_seg (CE or
   boundary-weighted TCKD on the scored frame, NOT large-target margin-hinge — it over-sharpens
   coarse grids); the HF periphery trained against reconstruction + the √(10·d_pose) term. EMA
   shadow, warmup-decay. Pose held on the trunk + the #140 radial-zoom codec.
3. **Rate-shed (RANK 2):** after convergence, apply score-aware QAT (`tac.torch_vehicle.score_aware_qat`)
   to the HF periphery ONLY → int4 + optional 2:4 sparsity; the LF core stays high-precision.
   Verify the HF periphery's margin-saliency-mass is < 10% (the structural-concentration gate) BEFORE
   shedding — this is the probe's metric, now as a build acceptance test.
4. **Byte-close + exact row:** reuse the G3 packet chain (`tools/build_torch_vehicle_g3_contest_packet.py`)
   → inflate.sh → dual CPU+CUDA exact eval. The break-even: the factored vehicle must beat the bc20
   basin's S=0.378 first (the small-basis frontier), then the 0.191 pointer.
5. **Falsification gate (the capacity question):** if the small LF core cannot reach d_seg < ~0.001
   at its size, RANK 1 walls → route to RANK 4 (geometry core) per the contingency. Measure this
   EARLY (a short LF-core-only train), $0, before committing the full multi-day build.

### Honest framing

- This is a multi-day R&D build, NOT a near-term pointer-mover. The easy routes are exhausted
  (three capped paths). Sub-0.15 is a representation problem, not a tuning problem.
- The probe's NO-GO is genuine signal: it kills the regularizer hope and sharpens the design to
  "structural factorization," saving the multi-day cost of a regularizer build that would have
  capped. That is the MVP-first $0 gate working as designed.
- `[contest-CPU advisory]`, exact pointer UNMOVED at 0.19110.

## 7. 6-hook wire-in (Catalog #125)

- **#1 sensitivity-map:** the probe consumes + extends `tac.margin_saliency_map` (per-weight + the
  frontier per-pixel char); the NO-GO REFINES it (per-weight criticality is non-relocatable on a
  dense path) — ACTIVE.
- **#2 Pareto constraint:** the measured (concentration-invariant, d_seg-regress) point is a Pareto
  constraint on the regularizer operating point (it cannot reach the low-d_seg / concentrated corner)
  — ACTIVE.
- **#3 bit-allocator:** the verdict re-targets the score-aware QAT bit-allocator at a FACTORED
  periphery (where it works) vs a dense decoder (where it capped) — ACTIVE.
- **#4 cathedral autopilot:** N/A (a feasibility verdict + design, not an archive-deployable surface).
- **#5 continual-learning posterior:** the GO/NO-GO is a probe outcome (regularizer-on-dense = NO-GO,
  factored-vehicle = GO-gated); register in `probe_outcomes.jsonl` on the next wiring pass — ACTIVE.
- **#6 probe-disambiguator:** the probe IS the disambiguator between "concentration is feasible by
  regularizer" (NO) and "concentration must be structural" (YES → RANK 1 build) — ACTIVE.

## 8. Files
- `experiments/probe_concentrated_saliency_feasibility.py` — the resumable $0 feasibility probe.
- `.omx/research/concentrated_saliency_feasibility_20260618T225128Z.json` — config A (top-3 core, reg 1.0).
- `.omx/research/concentrated_saliency_feasibility_20260618T230036Z.json` — config B (top-6 core, reg 0.3).
- `experiments/results/concentrated_saliency_feasibility{,_coreB}/` — checkpoints + state (344K each; rebuildable).

NO-FAKE: the saliency is the REAL autograd of the REAL frozen SegNet margin w.r.t. the REAL bc20
decoder weights; d_seg is the REAL argmax-flip-rate on the EXACT eval path (subset, validated
≈full-600); the concentration penalty was actually applied and actually reduced the periphery
gradient norm 10–12×. The only authoritative d_seg is upstream/evaluate.py; these are advisory.
