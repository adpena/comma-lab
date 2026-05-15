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

Empirically the three should ADD (not multiply) per Boyd convex-combination geometry: no antagonism is expected because they touch DIFFERENT axes of the loss landscape (spatial weighting × gradient direction × output-layer matching).

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
- **Substrate-wide rollout** (15+ active substrates × $30-60 retraining each = $450-$900 envelope): ΔS_aggregate ∈ [−0.05, −0.20] when the best-per-substrate gains compose (per Boyd Tier C composition matrix; not all substrates will land all of their per-substrate gain because of within-class saturation per Catalog #219 / #227, but the bottom-half of the substrate inventory has not been within-class-saturated at landing)
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

**Net assessment:** The U-DIE-KL composite is the canonical "substrate-agnostic training-side bolt-on" missing from the existing infrastructure. The three primitives (UNIWARD, DIE, KL) are individually canonical in this repo; combining them in a single torch-tensor loss helper makes the symposium #5 design directly consumable by every existing substrate trainer with a 3-line change. The helper itself adds NO archive bytes (per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable: this is research-only at landing because no per-substrate retraining has happened — Catalog #240 sister enforcement). All 8 premises verified pre-edit per Catalog #229.
