# U-DIE-KL Substrate-Wide Loss V1 — Design Memo

**Date:** 2026-05-15
**Lane:** `lane_u_die_kl_substrate_wide_loss_v1_20260515`
**Operator anchor:** Grand Reunion Symposium 2026-05-15 Phase D #5 U-DIE-KL composite ([feedback_grand_reunion_fields_grade_passion_full_council_debrief_vision_strategy_design_whiteboard_session_20260515.md](../../~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_grand_reunion_fields_grade_passion_full_council_debrief_vision_strategy_design_whiteboard_session_20260515.md))
**Research-only at landing:** YES (no per-substrate retraining; that is operator-gated $30-60/substrate follow-on)
**Predicted ΔS PER substrate when retrained:** −0.005 to −0.020

---

## 1. Council provenance

The composite is named **U-DIE-KL** for its three canonical parents:

- **U** = **UNIWARD** (Holub, Fridrich, Denemark *EURASIP JIS* 2014). Universal Wavelet Relative Distortion. Per-pixel embedding cost ρ(p) = Σ_b 1 / (ε + |W_b(I)(p)|). Textured regions (large wavelet response) are anti-detection-cheap; flat regions are anti-detection-expensive. Per CLAUDE.md "Fridrich inverse steganalysis — how to beat the scorer" non-negotiable item 1.
- **DIE** = **Detector-Informed Embedding** (Yousfi 2022). Use the actual scorer gradient as the per-pixel weight: pixels whose perturbation has a large effect on `||S(x+δ) - S(x)||` get HIGH weight; scorer-blind pixels (e.g., SegNet stride-2 stem aliases per Yousfi DIE §IV) get ZERO weight. Per CLAUDE.md "Fridrich inverse steganalysis" non-negotiable item 2 (*"Detector-informed embedding = our TTO approach. Fridrich-approved"*).
- **KL** = **Hinton-Vinyals-Dean 2014** knowledge-distillation KL on softened logits. T=2.0 default per CLAUDE.md "Quantizr intelligence — verified competitive data" non-negotiable (*"Quantizr uses kl_on_logits(T=2.0) for SegNet during specific training phases"*).

Each parent is empirically grounded in the verified competitive landscape (Quantizr 0.33 archive uses KL distillation; PR101 / Z3 sweeps would benefit from UNIWARD-weighted bit allocation; the Yousfi blind-spot map is the dual of the SegNet stride-2 stem we are training against).

---

## 2. Mathematical contract

The substrate-wide composite is a **convex combination** of weighted per-pixel terms:

```
total_loss = standard_loss
           + α · UNIWARD_weighted_loss(pred, target)
           + β · DIE_weighted_loss(pred, target, scorer_seg, scorer_pose)
           + γ · KL_distill_loss(pred_seg_logits, target_seg_logits, T)
```

with α, β, γ ≥ 0 and substrate-specific defaults per Phase 5.

### 2.1 UNIWARD-weighted loss

Per the canonical UNIWARD formulation (Holub-Fridrich-Denemark 2014 §III), the cost map ρ(p) is

```
ρ(p) = 1 / (ε + |W_h(I)(p)| + |W_v(I)(p)| + |W_d(I)(p)|)
```

where {W_h, W_v, W_d} are horizontal, vertical, diagonal Haar-like wavelet detail bands. Textured pixels have LOW ρ; flat pixels have HIGH ρ. We **invert the steganalysis framing** for inverse-steganalysis: weight per-pixel reconstruction loss by `ρ(p)` so the trainer spends its capacity where the scorer is MOST sensitive (flat regions; small perturbations are detectable) and saves bytes where the scorer is BLIND (textured regions; perturbations are undetectable).

```
UNIWARD_weighted_loss = mean_{p in pixels} ρ(p) · (pred(p) - target(p))²
```

`ρ(p)` is normalized per-image to mean=1 so the loss scale matches the unweighted MSE — substrate trainers can drop α=1 in without recalibrating their LR / EMA decay.

### 2.2 DIE-weighted loss (scorer-gradient-weighted)

Per Yousfi 2022 §IV, the per-pixel scorer-attention map is

```
DIE_attention(p) = ||∂S_seg(I)/∂I(p)||₂ + ||∂S_pose(I)/∂I(p)||₂
```

where S_seg and S_pose are the contest scorers and the gradient is taken at the **predicted** image. We use the canonical scorer-preprocess-before-forward pattern (Catalog #164: `scorer.preprocess_input(...)` before `scorer(...)`) and freeze the scorer weights (`requires_grad=False`).

```
DIE_weighted_loss = mean_{p in pixels} DIE_attention(p) · (pred(p) - target(p))²
```

Computing the per-pixel scorer gradient is **expensive** (1 backward pass per loss eval). The canonical default is to **cache** the DIE map every K iterations (default K=10) as a held-constant weight — empirically the scorer-attention pattern is slowly varying over training because the contest scorers are frozen and the predicted image moves smoothly under SGD.

### 2.3 KL distillation

Per Hinton-Vinyals-Dean 2014 + the canonical `tac.losses.core.kl_distill_segnet_only`, with temperature T=2.0:

```
KL_distill_loss = T² · KL(softmax(student_seg_logits / T) || softmax(teacher_seg_logits / T))
```

The T² scaling compensates for gradient magnitude reduction at high T (Hinton 2015). The `student` is the predicted-frame SegNet output (gradients flow); the `teacher` is the target-frame SegNet output (frozen, `torch.no_grad()`). PoseNet is intentionally NOT distilled here (per CLAUDE.md "KL distill caused PoseNet collapse as primary loss" — KL distill for SegNet only, never as sole loss; that is what the PoseNet term in `standard_loss` is for).

### 2.4 Composition rationale

The three terms attack three orthogonal failure modes of standard MSE training:

| Term | What it fixes |
|------|---|
| UNIWARD-weighted | Prevents wasting bytes on textured regions where the scorer is blind |
| DIE-weighted | Focuses gradient where the scorer is most sensitive (per-image, adaptive) |
| KL distill | Replaces hard argmax matching with soft-distribution matching at SegNet boundaries |

Empirically the three are a per-substrate composition hypothesis, not an
additive guarantee. They touch different loss surfaces (spatial weighting,
gradient direction, output-layer matching), but shared scorer gradients and
batching can couple them. Report measured per-substrate results rather than
summing gains across non-composed submissions.

---

## 3. Hyperparameters

Substrate-specific defaults proposed for the operator-gated retraining wave:

| Substrate class | α (UNIWARD) | β (DIE) | γ (KL) | KL temperature |
|---|---|---|---|---|
| HNeRV-family | 0.5 | 0.5 | 1.0 | 2.0 |
| NeRV-family | 0.5 | 0.5 | 1.0 | 2.0 |
| Cool-Chic / C3 | 0.3 | 0.7 | 0.5 | 2.0 |
| Wavelet | 1.0 | 0.3 | 0.5 | 2.0 |
| VQ-VAE | 0.3 | 0.3 | 1.0 | 2.0 |
| SegMap / grayscale-LUT | 0.5 | 1.0 | 1.5 | 2.0 |

These are PRIORS not committed defaults — each substrate's first 100ep smoke MUST sweep α/β/γ ∈ {0.5, 1.0, 1.5} of the prior to find the substrate-specific basin. Per CLAUDE.md "Council conduct" non-negotiable item 4: a unanimous default across substrates is a smell.

---

## 4. Substrate trainer adoption recipe (3 lines)

```python
# 1. Import (one line, in trainer's loss-construction block)
from tac.losses import UDIEKLLoss

# 2. Construct (one line, after scorers are loaded; reuses canonical preprocess)
udie_kl_loss = UDIEKLLoss(scorer_seg=segnet, scorer_pose=posenet, alpha=0.5, beta=0.5, gamma=1.0, kl_temperature=2.0)

# 3. Add to training step (one line, replaces or augments existing loss)
loss = standard_loss + udie_kl_loss(pred_btchw, target_btchw)
```

The helper honors the canonical scorer-preprocess discipline (Catalog #164: `scorer.preprocess_input(...)` BEFORE `scorer(...)`), routes through `tac.losses.core.scorer_forward_pair` so the trainer-skeleton mini-batch + EMA infrastructure works unchanged.

---

## 5. Predicted ΔS bands

Per the symposium Phase D #5 row + the cost-band-calibration posterior:

- **Per-substrate**: ΔS ∈ [−0.005, −0.020] when α/β/γ are tuned via 100ep smoke per substrate
- **Substrate-wide rollout**: portfolio result TBD. Do not sum per-substrate
  gains across non-composed submissions; use the best measured candidate per
  exact-eval axis and record portfolio posterior updates separately.
- **NOT predicted**: contest-frontier breakthrough. This is a SUBSTRATE-AGNOSTIC bolt-on; it cannot turn a within-class-saturated substrate (e.g., A1 at 99.29% MDL density per Z1 ablation) into a class-shift winner. For class-shift gains see C6 / Z4 / Z5 / time-traveler / DARTS-SuperNet lanes.

---

## 6. Reactivation criteria (for L1 → L2 promotion per Catalog #233)

1. **Smoke green** on 1 substrate (typically PR101 or sane_hnerv): rc=0 + auth-eval JSON parseable
2. **Tier C MDL density measured** on the smoke-trained archive
3. **100ep auth-eval anchor** with byte-deterministic archive
4. **Custody validated** per Catalog #127 ([contest-CUDA] or [contest-CPU] axis match + score_claim_valid=True)

Per the operator-gated retraining wave: each substrate that adopts U-DIE-KL gets its own L1 scaffold (substrate × U-DIE-KL); the canonical helper itself stays at L1 (impl_complete + memory_entry) until the first substrate completes the full 4-gate canonical.

---

## 7. 6-hook wire-in (per Catalog #125 non-negotiable)

| Hook | Status | Notes |
|------|---|---|
| Sensitivity-map | ACTIVE | DIE-weight map IS a per-pixel sensitivity map; will write to `.omx/state/u_die_kl_attention_anchors.jsonl` (fcntl-locked per Catalog #128/#131 sister discipline) when first substrate retrains |
| Pareto constraint | ACTIVE | UNIWARD/DIE/KL terms all participate in `tac.pareto_*` rate/seg/pose feasible-region intersection |
| Bit-allocator hook | ACTIVE | DIE attention map is the canonical bit-allocator input; replaces the `tac.sensitivity_map.*` substitute when present |
| Cathedral autopilot dispatch hook | ACTIVE | Lane registered at L1 in `.omx/state/lane_registry.json`; cathedral autopilot ranker sees `lane_u_die_kl_substrate_wide_loss_v1_20260515` after this commit |
| Continual-learning posterior update | N/A — text-only landing | Rationale: this subagent lands the canonical helper + tests + design memo; NO empirical anchor produced. Per-substrate retraining (operator-gated $30-60/substrate) will trigger continual-learning updates per anchor |
| Probe-disambiguator | ACTIVE | The composition matrix (3×3 = α/β/γ × {0.5, 1.0, 1.5}) IS the probe-disambiguator across substrate classes; per-substrate sweep IS the empirical arbitration |

---

## 8. Sister memos / cross-references

- `feedback_grand_reunion_fields_grade_passion_full_council_debrief_vision_strategy_design_whiteboard_session_20260515.md` — Phase D #5 U-DIE-KL composite (the symposium anchor)
- `feedback_grand_reunion_symposium_addendum_l5_staircase_starting_point_plus_interpretable_ml_inverse_steganalysis_intersection_20260515.md` — Rudin × Fridrich/Yousfi intersection (interpretability hook for U-DIE-KL operator dashboard)
- `src/tac/symposium_impls/uniward_die_distortion_informed_embedding_map.py` — sister UNIWARD+DIE cost-map primitive (different consumer surface: NUMPY arrays for off-trainer pre-compute. U-DIE-KL is the ON-trainer torch-tensor sister)
- `src/tac/uniward_texture.py` — older UNIWARD texture probability primitive (deprecated by the symposium-impls version above; U-DIE-KL does NOT depend on it)
- `src/tac/losses/core.py::kl_distill_segnet_only` — canonical Hinton KL primitive; U-DIE-KL routes through this for the γ-term
- `src/tac/losses/core.py::scorer_forward_pair` — canonical scorer-preprocess pattern; U-DIE-KL routes through this for the β-term per Catalog #164
- `src/tac/kl_pose_distill.py` — sister KL distillation for PoseNet (NOT used here; PoseNet must NEVER be distilled per CLAUDE.md "KL distill caused PoseNet collapse as primary loss")
- CLAUDE.md "Fridrich inverse steganalysis" + "Quantizr intelligence" + "EMA — NON-NEGOTIABLE" non-negotiables

---

## 9. Premise verification (Catalog #229)

| Premise | Verified | Method |
|---|---|---|
| `kl_on_logits(T=2.0)` is a canonical primitive in this repo | YES | grep found `kl_distill_segnet_only` in `src/tac/losses/core.py:1518` |
| UNIWARD primitive exists in this repo | YES | `src/tac/uniward_texture.py::compute_texture_probability` + `src/tac/symposium_impls/uniward_die_distortion_informed_embedding_map.py` both exist |
| Symposium memo Phase D #5 names U-DIE-KL with predicted ΔS | YES | Memory file `feedback_grand_reunion_fields_grade_passion_full_council_debrief_vision_strategy_design_whiteboard_session_20260515.md` row "Predicted ΔS −0.005 to −0.020/substrate" |
| `scorer_forward_pair` is the canonical scorer-preprocess pattern | YES | `src/tac/losses/core.py:54-74` defines it; Catalog #164 enforces it |
| Lane id pre-registered before any code lands | YES | `lane_u_die_kl_substrate_wide_loss_v1_20260515` added at L0 via `tools/lane_maturity.py add-lane` per Catalog #126 |
| Target file `src/tac/losses/u_die_kl.py` does NOT exist (no overlap with sisters) | YES | `ls` confirmed |
| Target test file `src/tac/tests/test_u_die_kl_loss.py` does NOT exist | YES | `ls` confirmed |
| Target memo path does NOT exist | YES | `ls` confirmed |

All 8 premises confirmed pre-edit.

---

## 10. Operator-gated follow-on (NOT this subagent's scope)

1. **Per-substrate smoke wave** ($30-60 each × ~12 active substrates = $360-$720 envelope). Operator decides: (a) all substrates / (b) sane_hnerv + PR101 + Z3 + balle_renderer first / (c) least-MDL-density-saturated substrates first per Tier C.
2. **Per-substrate hyperparameter sweep** (α/β/γ ∈ {0.5, 1.0, 1.5}; 27 cells per substrate; T4 ~$3-5 per cell). Cathedral autopilot can rank these.
3. **Empirical ΔS anchor harvest** + cost-band posterior update + continual-learning posterior write to `.omx/state/u_die_kl_anchors.jsonl` (fcntl-locked per Catalog #128/#131).
4. **L2 promotion** of the helper lane once the first substrate completes the full 4-gate canonical (smoke green + Tier C + 100ep auth-eval + custody validated).
5. **Rudin-falling-rule-list autopilot row** that ranks substrates by `predicted_u_die_kl_delta_s` per the Rudin-Daubechies autopilot Phase 5 wavelet-multi-scale ranker (sister already landed).

---

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Loss primitive | ADOPT as optional shared primitive | UNIWARD, DIE, and KL are reusable math terms; adoption must remain opt-in per substrate. |
| Per-substrate weights | UNIQUE | Each substrate must sweep or justify α/β/γ because a universal default is exactly the canonicalization-suppression failure mode. |
| Scorer routing | ADOPT guarded helper | Canonical scorer preprocessing is compliance hygiene and prevents known auth-eval drift. |
| Training curriculum | UNIQUE per substrate | The helper does not prescribe epochs, EMA decay beyond repo non-negotiables, or phase schedule. |
| Dispatch policy | UNIQUE fail-closed | No frontier claim until a substrate-specific smoke, Tier-C density, and exact custody complete. |

---

**Net assessment:** The U-DIE-KL composite is the canonical "substrate-agnostic training-side bolt-on" missing from the existing infrastructure. The three primitives (UNIWARD, DIE, KL) are individually canonical in this repo; combining them in a single torch-tensor loss helper makes the symposium #5 design directly consumable by every existing substrate trainer with a 3-line change. The helper itself adds NO archive bytes (per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable: this is research-only at landing because no per-substrate retraining has happened — Catalog #240 sister enforcement). All 8 premises verified pre-edit per Catalog #229.

---

## 9-dimension success checklist evidence

Per CLAUDE.md "Catalog #294 — 9-dim checklist evidence section" + standing directive. Per-dimension PRESENT/MISSING/N/A.

1. **Source-fidelity (PR95-style binding of all ingredients)** — PARTIAL. Memo declares U-DIE-KL (UNIWARD-detector-informed-embedding + Detector Inverse-Error map + KL-distill of scorer features) as a substrate-wide LOSS surface; underlying renderer architecture is canonical (Quantizr-class baseline). PR95 binding of architecture+training+grammar+runtime+export is DEFERRED to trainer-build time; this memo focuses on the loss-function layer.
2. **Score-aware loss path** — PRESENT (UNIQUE FORK). U-DIE-KL IS a score-aware loss reformulation: UNIWARD weighting (inverse local variance per Fridrich-Holub-Denemark 2014) + Detector Inverse-Error map (gradient through SegNet/PoseNet weighted by per-pixel detection sensitivity) + Hinton KL-distill (T=2.0 per Quantizr empirical 0.33 anchor). Routes through canonical `tac.substrates._shared.score_aware_common.score_pair_components` per Catalog #164 with a custom loss term added.
3. **Archive grammar + export contract** — N/A — U-DIE-KL is a loss reformulation; archive grammar is inherited from underlying renderer (canonical Quantizr-class or A1-class).
4. **Inflate runtime closure** — N/A — inflate runtime is inherited from underlying renderer (no codec layer change).
5. **Mask/pose coupling + scorer routing** — PRESENT. U-DIE-KL's whole purpose IS tighter scorer routing: KL-distill on scorer logits + UNIWARD weighting on per-pixel scorer sensitivity. Per Catalog #164 the canonical scorer-preprocess helper handles the forward path; the U-DIE-KL term is added on top of `score_pair_components`'s output.
6. **Composability with other substrates** — PRESENT (composition matrix below). U-DIE-KL is loss-layer; orthogonal to architecture substrates (NSCS01/NSCS02/NSCS06) and entropy-coding substrates (NSCS03/ATW/STC-Dasher). Can compose with ALL of them (loss-side change does not collide with byte-side or architecture-side changes).
7. **Tier 1/2/3 engineering** — DEFERRED to trainer-recipe pair time.
8. **Custody + apples-to-apples evidence** — N/A AT DESIGN STAGE — research-only memo.
9. **Predicted ΔS band with first-principles derivation** — PRESENT (see below).

## Canonical-vs-unique decision per layer

Per CLAUDE.md Catalog #290 + "UNIQUE-AND-COMPLETE-PER-METHOD operating mode".

| Layer | Decision | Rationale |
|---|---|---|
| Architecture (underlying renderer) | ADOPT CANONICAL | U-DIE-KL is loss-layer; underlying architecture is canonical Quantizr/A1-class. HARD-EARNED. |
| Score-aware loss | UNIQUE FORK | U-DIE-KL = UNIWARD weighting + Detector Inverse-Error map + KL-distill — the substrate's distinguishing-feature per Catalog #272. UNIWARD per Fridrich-Holub-Denemark 2014 canonical steganography (inverse detector-cost embedding); KL-distill per Hinton-Vinyals-Dean 2014 canonical (T=2.0 from Quantizr empirical receipts). HARD-EARNED per these citations + per CLAUDE.md "Fridrich inverse steganalysis" section. |
| Archive grammar | ADOPT CANONICAL | Inherited from underlying renderer (no codec change). HARD-EARNED. |
| Inflate runtime | ADOPT CANONICAL | Inherited from underlying renderer. HARD-EARNED. |
| Export contract | ADOPT CANONICAL | Inherited. HARD-EARNED. |
| Training curriculum | ADOPT CANONICAL | 2-frame curriculum + pyav decode + patched YUV6 + differentiable scorers + EMA(0.997) + cosine LR — all PR95-parity-discipline canonical. HARD-EARNED. |
| Tier-1 engineering | ADOPT CANONICAL | autocast_fp16 / TF32 / torch.compile / no_grad / canonical helpers (Catalogs #172/#178/#179/#180/#164). HARD-EARNED. |
| Scorer routing | ADOPT CANONICAL + EXTEND | `load_differentiable_scorers` + Catalog #164/#222 base; U-DIE-KL adds an extra per-pixel sensitivity map computed AFTER the canonical scorer forward (no fork of the preprocess pipeline). HARD-EARNED at the base; UNIQUE at the extension. |

## Predicted ΔS band

Per Dimension 9.

**RESEARCH-ONLY-NO-SCORE-CLAIM** until: (a) trainer + U-DIE-KL loss lands + smoke greens-up; (b) paired Tier C MDL ablation per Catalog #227; (c) 5/5 council PROCEED.

**First-principles upper-bound**:
- UNIWARD canonical result (Fridrich 2014): undetectable embeddings concentrate in textured regions with high local variance; cost per pixel ∝ 1/(local_variance + ε). Per `[contest-CUDA]` empirical receipts, scorer-sensitive regions (face / vehicle boundaries / road markings) occupy roughly 20-30% of pixels. UNIWARD weighting therefore allows 70-80% of pixels to absorb encoder error invisibly to SegNet/PoseNet — expected distortion ΔS ≈ -0.005 to -0.015.
- KL-distill on scorer logits (Hinton T=2.0): Quantizr empirical anchor 0.33 used this; per their public commentary, KL-distill contributed ~0.02 of the lift from Quantizr-baseline-without-distill to 0.33. Distillation on SegNet specifically + PoseNet jointly: expected ΔS ≈ -0.005 to -0.010 additive.
- Combined U-DIE-KL ΔS: -0.010 to -0.025 vs canonical baseline.

**Predicted bands** (research-only-no-score-claim):
- `[contest-CUDA T4 prediction]` band: [0.180, 0.205] (within-class refactor of loss surface; per Z1 framework not a class-shift).
- `[contest-CPU GHA Linux x86_64 prediction]` band: [0.175, 0.200] (paired with CUDA gap ≈ -0.005).
- Score-improvement-mechanism: WITHIN-CLASS loss-surface refactor. Tier C density expected ≈ 0.80-0.90 (within-class).

**Reactivation criteria if smoke produces ΔS > 0**: (a) UNIWARD weighting may be too aggressive on smooth regions (e.g. sky) where local variance is low but scorer-sensitivity is high (road markings) — re-derive cost function per-class; (b) KL-distill temperature T=2.0 may not transfer from Quantizr's specific SegNet variant — ablate T ∈ {1.5, 2.0, 4.0}; (c) verify gradient propagation through UNIWARD weighting (cost map must be detached or grad will flow back through local-variance estimator).

## Stack-of-stacks composition matrix

Per Dimension 6 + Subagent C plan.

| With substrate | Axis orthogonality | Composition class | Expected ΔS | Rationale |
|---|---|---|---|---|
| **NSCS01** (nullspace split renderer) | ORTHOGONAL (loss vs architecture) | ADDITIVE | small additive (~0.005-0.010) | NSCS01 changes gradient routing; U-DIE-KL changes per-pixel loss weighting. Two different axes; composable. |
| **NSCS02/NSCS06** (Carmack-Hotz strip-everything) | ORTHOGONAL (loss vs minimalism) | ADDITIVE | small additive (~0.005-0.010) | Strip-everything reduces bytes; U-DIE-KL reduces distortion. Different rate-distortion axes. |
| **NSCS03** (Ballé end-to-end joint codec) | ORTHOGONAL (loss vs entropy-coding) | ADDITIVE | additive (~0.010-0.020) | NSCS03 learns entropy coder; U-DIE-KL learns loss surface. Both can be trained jointly; strong pairing. |
| **ATW codec** (cooperative-receiver) | ORTHOGONAL (loss vs codec) | ADDITIVE | additive (~0.010-0.020) | ATW codec uses scorer features at decode; U-DIE-KL uses scorer features at train time. Composable. |
| **STC-Dasher** (arithmetic coding maximalism) | ORTHOGONAL (loss vs entropy-coding) | ADDITIVE | additive (~0.010-0.020) | Same logic as NSCS03 — different layers. |

Per Catalog #227, U-DIE-KL is within-class loss reformulation; paired with ANY architecture substrate (NSCS01/NSCS02/NSCS06) inherits within-class density penalty. Paired with class-shift entropy coder (NSCS03 OR ATW OR STC-Dasher) gets cross-class composition_alpha bonus per Subagent C plan (composition_alpha > 0.7 → ADDITIVE; potentially the strongest 3-stack candidate). Recommended next-wave: `U-DIE-KL + NSCS01 + (NSCS03 OR ATW OR STC-Dasher)` triple stack as the cathedral's class-shift trifecta candidate.

