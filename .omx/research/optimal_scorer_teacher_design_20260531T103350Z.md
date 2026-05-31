# Optimal scorer-teacher design + multi-granularity sensitivity tooling

- UTC: 2026-05-31T10:33:50Z
- Lane: `lane_optimal_teacher_and_sensitivity_tools_20260531`
- Author: subagent `optimal-teacher-and-sensitivity-tools`
- Scope: DESIGN-ONLY for the teacher wire-in (sister `z8-hier-pc-full-stack-longrun` holds the GPU + is mid-flight using the live teacher path); BUILD for the sensitivity tools.
- Axis discipline: every emitted score-relevant number is `[predicted]` / `[macOS-MLX research-signal]` non-promotable per Catalog #341 / #192 / #127 / #323. No paid GPU. No PR. No contest-score claim.

## Predicted ΔS band

`[macOS-MLX research-signal]` advisory only; this memo proposes a teacher change + sensitivity tooling, not a contest archive. The falsifiable prediction (§4) is the only numeric claim and it is a research-signal proxy delta, NOT a contest-CPU/CUDA score. Dykstra-feasibility: the proposed seg-objective change is a re-weighting WITHIN the existing rate/seg/pose feasible set (it changes which pixels the KL gradient targets, not the archive grammar), so it does not move the polytope; it moves the operating point along the seg face. First-principles bound: §1 derives the objective directly from the contest score's `∂S/∂(teacher output)` so the prediction is grounded in `S = 100·d_seg + sqrt(10·d_pose) + 25·R`, not a sweep. <!-- PREDICTED_BAND_VIBES_OK:research-signal-advisory-only-derived-from-contest-score-marginals-not-a-contest-archive-band -->

---

## 0. The exact contest scorer math (ground truth, `upstream/modules.py`)

Verified directly from the pinned upstream snapshot (NOT from memory):

**SegNet** (`upstream/modules.py:103-112`, `smp.Unet('tu-efficientnet_b2', classes=5)`):
```python
def preprocess_input(self, x):
    x = x[:, -1, ...]                       # LAST frame only of the pair
    return F.interpolate(x, size=(H,W), mode='bilinear')
def compute_distortion(self, out1, out2):
    diff = (out1.argmax(dim=1) != out2.argmax(dim=1)).float()   # per-pixel argmax disagreement
    return diff.mean(dim=tuple(range(1, diff.ndim)))            # → d_seg = disagreement RATE
```
So **`d_seg` = per-pixel argmax-class-disagreement rate of the LAST frame**, 5 classes, at SegNet input resolution.

**PoseNet** (`upstream/modules.py:61-84`, FastViT-T12, 12-ch YUV6 of both frames):
```python
def compute_distortion(self, out1, out2):
    distortion_heads = ['pose']
    return sum((out1[h.name][..., :h.out//2] - out2[h.name][..., :h.out//2]).pow(2)
               .mean(dim=tuple(range(1, out1[h.name].ndim)))
               for h in self.hydra.heads if h.name in distortion_heads)   # MSE
```
So **`d_pose` = MSE on the FIRST HALF (`:h.out//2`) of the pose head's output**, per-pair scalar. With the canonical 6-dim pose head this is the first 3 dims; the substrate teacher code (`pose_distillation_mse_loss`) distils on the head's `:out//2` slice.

**Score** (canonical, `tac.master_gradient.compute_marginal_coefficients`):
```
S        = 100·d_seg + sqrt(10·d_pose) + 25·R          (R = archive_bytes / 37_545_489)
∂S/∂d_seg  = 100                          (constant)
∂S/∂d_pose = 5 / sqrt(10·d_pose)          (hyperbolic; diverges as d_pose→0; at ~0.192 frontier ≈ 25.6)
∂S/∂byte   = 25 / 37_545_489
```

## 0.1 The live teacher (verified, do NOT edit — sister is mid-flight)

`src/tac/substrates/_shared/mlx_score_aware/{loss.py,bundle.py}` +
`src/tac/substrates/hinton_distilled_scorer_surrogate/mlx_loss.py`:

- **Seg**: `hinton_distilled_kl_t2_loss(student_logits, teacher_logits, T=2.0)` =
  `T² · KL(softmax(student/T) ‖ softmax(teacher/T))` over **all 5 classes at every pixel**.
  Teacher = real `smp.Unet` EfficientNet-B2 per-pixel logits; student = a learnable 1×1-conv head on the
  decoded last frame (full-PoseNet/SegNet backprop NaNs in MLX 2nd-order autograd — the learnable head is the
  finite-gradient surrogate; this is HARD-EARNED, see §5).
- **Pose**: `pose_distillation_mse_loss(student_pose, teacher_pose, per_dim_scale=teacher_std)` =
  `mean( ((student-teacher)/std)² )` — **already per-dim-std standardized** (a Mahalanobis-like distance), already
  NON-uniform. Teacher = real FastViT-T12 pose vector.

The teacher is already sophisticated. The two open questions the operator named map to two precise gaps below.

---

## 1. The math-derived optimal seg-distillation objective

### 1.1 The structural mismatch: KL-on-full-logits vs argmax-flip-rate

`d_seg` is a step function of the argmax. The teacher's per-pixel decision is `argmax_c z_c`. A perturbation
of the rendered frame only moves `d_seg` if it **flips an argmax** at some pixel. An argmax flip at pixel `i`
can only happen when the **top-2 logit margin** `m_i = z_{(1)} - z_{(2)}` is small — i.e. the pixel is in the
**decision-boundary band**. For a high-confidence interior pixel (`m_i` large), no achievable rendering
perturbation flips the label, so its contribution to `∂d_seg/∂(rendered pixel)` is ≈ 0.

Hinton KL T=2.0 distributes its gradient over **every pixel and every class** in proportion to the teacher's
soft probabilities. On a confident interior pixel `softmax(z/T)` is still a full 5-vector, so the student is
pushed to match the teacher's exact probability mass on directions (interior-class confidence) **that can never
change d_seg**. The KL objective spends gradient budget where the score is flat. This is the cargo-cult risk
(§5): KL T=2.0 is the canonical *classification*-distillation objective; the contest seg score is a
*decision-boundary-agreement-rate* objective, which is a strictly different mathematical functional.

### 1.2 DKD decomposition makes the mismatch precise (Zhao 2022 CVPR, verified Eq.6/7)

Decoupled KD (`arXiv:2203.08679`, ar5iv-verified) proves classical KD decomposes **exactly** as:

```
Eq.6:  KD = TCKD + (1 − p_t^T) · NCKD
```
where `p_t^T = softmax(z/T)` at the **target (argmax) class**, TCKD = binary KD on {target, non-target} mass,
NCKD = KD over the non-target classes' relative shape. The `(1 − p_t)` coupling **suppresses NCKD whenever the
teacher is confident** (`p_t → 1 ⇒ (1−p_t) → 0`). DKD reformulates as:
```
Eq.7:  DKD = α · TCKD + β · NCKD     (α, β decoupled)
```

For the contest: the score-relevant signal is **exactly TCKD-on-the-boundary** (does the student's argmax
agree with the teacher's argmax = does the {target-class vs rest} split match). NCKD — the relative shape of the
non-target classes — is **irrelevant to d_seg** because d_seg never reads non-target rank, only the argmax. So
the contest math says: **for seg, set β (NCKD) toward 0 and concentrate α (TCKD) on the boundary band, weighted
by the inverse margin.** This is the opposite of generic DKD (which boosts NCKD); the contest's argmax-only
metric is the rare regime where TCKD-only is score-optimal.

### 1.3 BPKD confirms the spatial form (Liu 2024 WACV, verified)

Boundary-Privileged KD (`BPKD`, WACV 2024) independently arrives at the spatial structure: distil **edge regions
with a dedicated edge loss** (high spatial sensitivity, ambiguous-class disambiguation) separately from **body
regions** (shape constraints), because student edge predictions are the high-uncertainty / high-discrepancy
locus. For the contest, the d_seg signal IS the edge band; BPKD's edge-loss is the spatial realization of §1.2's
"concentrate on the boundary band."

### 1.4 The optimal seg-distillation objective (the design)

Define the teacher per-pixel **boundary-band weight**:
```
m_i      = z_teacher,(1),i − z_teacher,(2),i          # top-2 teacher logit margin at pixel i
w_i      = exp(−m_i / τ_b)        OR    w_i = 1[ m_i < m_thresh ]   # boundary-band indicator / soft
```
The optimal seg-distillation loss is **boundary-weighted target-class-KD** (DKD α-only, BPKD edge-loss):
```
L_seg* = Σ_i  w_i · TCKD_i(student_logits_i, teacher_logits_i, T)
       = Σ_i  w_i · T² · KL( binarize_to_target(student/T) ‖ binarize_to_target(teacher/T) )
```
Plus an optional small **margin-preservation** hinge that pulls the student's margin toward the teacher's
**sign** (not magnitude) on each boundary pixel, so the student learns where the teacher's decision flips, not
the teacher's interior confidence. Properties:
1. **Score-faithful**: gradient concentrates exactly where `∂d_seg/∂rendered ≠ 0` (the boundary band).
2. **Gradient budget**: re-allocated from `O(H·W)` confident interior pixels to `O(boundary_band ≈ 5–15%·H·W)`
   informative pixels → higher effective signal at the same step count.
3. **Reduces to the current objective** when `w_i ≡ 1` and TCKD→full-KL → safe, auditable superset.

`τ_b` / `m_thresh` are derived empirically from the boundary-band fraction (§3, the sensitivity tool measures
the real margin distribution on `upstream/videos/0.mkv` so `τ_b` is calibrated, not guessed).

## 2. The math-derived optimal pose-distillation objective

`d_pose = sum_k (Δpose_k)² / K` over `k ∈ [0, K=out//2)`. `∂S/∂d_pose = 5/sqrt(10·d_pose)` is a per-pair
scalar; within a pair, the score-sensitivity to pose dim `k` is `∂S/∂(student_pose_k) ∝ (student_pose_k −
teacher_pose_k)`. So the optimal per-dim weight is NOT the teacher std (current `per_dim_scale`) but the
**inverse-variance Mahalanobis weight times the contest's equal per-dim weighting in the MSE**:
```
L_pose* = (5 / sqrt(10·d_pose_running)) · mean_k[ ((student_pose_k − teacher_pose_k) / σ_k)² ]   for k < K
```
Two refinements vs the current `pose_distillation_mse_loss`:
1. **Only the first `K = out//2` dims enter the contest** — confirm the teacher distils on the `:out//2`
   slice (it does, but the per-dim weight should be 0 for `k ≥ K` to avoid wasting capacity on dims the
   scorer ignores). The sensitivity tool (§3) measures which of the `K` dims actually dominate per-pair so
   the weight is empirical, not assumed-uniform.
2. **Pose-regressor dark-knowledge caveat (Saputra 2019, `arXiv:1908.00858`, verified)**: regression
   distillation cannot use softmax dark-knowledge; the Attentive Imitation Loss weights the per-sample
   teacher-imitation term by the teacher's **reliability**. Translation here: weight each pair's pose-distill
   term by the local `∂S/∂d_pose = 5/sqrt(10·d_pose)` so pairs near the pose floor (small d_pose, large
   marginal) get more gradient — exactly the operating-point-aware reweighting CLAUDE.md "SegNet vs PoseNet
   importance" already flags (pose is ~2.71× seg by marginal at the 0.192 frontier).

The current per-dim-std scaling is **already most of the way there** (it is the correct scale-stabilization);
the refinement is (a) zero-out `k ≥ K`, (b) per-pair sensitivity-reweight by `5/sqrt(10·d_pose)`, (c) measure
the real per-dim dominance instead of assuming uniform.

## 3. Bleeding-edge survey + which technique the math selects

| Technique | Paper (verified) | Fit to the contest seg/pose structure |
|---|---|---|
| Hinton KL (current) | Hinton-Vinyals-Dean 2014/2015 | classification dark-knowledge; spreads gradient over score-flat interior. CARGO-CULTED for argmax-only d_seg (see §5). |
| **Decoupled KD (DKD)** | **Zhao 2022 CVPR `2203.08679` (Eq.6/7 verified)** | **SELECTED for seg.** d_seg = argmax agreement = TCKD-only; set β(NCKD)→0, α(TCKD) boundary-weighted. The contest is the rare TCKD-dominant regime. |
| **BPKD (boundary-privileged)** | **Liu 2024 WACV (verified)** | **SELECTED for seg spatial form.** edge-loss = boundary band; body-loss = ~0 weight for the contest. |
| Relational KD (RKD) | Park 2019 CVPR | distils sample-pair distances; orthogonal — could help the renderer preserve inter-pixel structure but not directly d_seg. DEFERRED. |
| Attention transfer | Zagoruyko 2017 | distils feature attention maps; needs full-SegNet feature access (the head surrogate has none). DEFERRED-pending-feature-access. |
| FitNets (hint) | Romero 2015 | intermediate feature hints; same feature-access blocker. DEFERRED. |
| Contrastive rep. distill (CRD) | Tian 2020 ICLR | mutual-info between teacher/student reps; heavy, feature-access blocker. DEFERRED. |
| **Attentive Imitation Loss** | **Saputra 2019 `1908.00858` (verified)** | **SELECTED for pose.** reliability/sensitivity-weighted regression imitation = per-pair `5/sqrt(10·d_pose)` reweight. |
| Born-Again Networks | Furlanello 2018 (verified) | dark-knowledge gradient = ground-truth-rescale + dark term; confirms the dark term carries the inter-class signal that d_seg discards → supports stripping NCKD for seg. |

**Verdict: the contest score's argmax-only seg + sensitivity-scaled pose structure selects (DKD-α-only ⊕ BPKD-edge)
for seg and (per-dim-std ⊕ AIL-sensitivity-reweight ⊕ zero `k≥K`) for pose.** The full-feature distillation
families (RKD/AT/FitNets/CRD) are DEFERRED-pending-feature-access (the learnable-head surrogate exposes no
teacher features; reactivation = a teacher that surfaces intermediate features without 2nd-order MLX NaNs).

## 4. The concrete wire-in plan (DESIGN ONLY — operator-attended next slot)

The sister `z8-hier-pc-full-stack-longrun` is mid-flight using the live teacher. **Do NOT edit** the live path now.
When the GPU frees:

1. **New canonical kernel** `tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss.boundary_weighted_tckd_loss(student_logits, teacher_logits, *, temperature=2.0, boundary_tau=None, margin_threshold=None)`:
   - compute teacher top-2 margin `m_i`; `w_i = exp(−m_i/τ_b)` (soft) or `1[m_i<m_thresh]` (hard).
   - TCKD = KL on the {target-mass, 1−target-mass} binary split (DKD Eq.6 TCKD term).
   - return `mean(w_i · T² · TCKD_i)`; `w_i ≡ 1, full-KL` recovers the current objective (auditable superset).
   - sister kernel `pose_sensitivity_weighted_mse_loss(student_pose, teacher_pose, *, per_dim_scale, d_pose_running, out_half)` that (a) slices `[:out_half]`, (b) per-dim-std scales, (c) multiplies by `5/sqrt(10·max(d_pose_running, ε))`.
2. **Flag, not default-swap** (Catalog #240 / #326 driver-mode discipline): add `--seg-distill-objective {kl_t2|boundary_tckd}` and `--pose-distill-objective {std_mse|sensitivity_mahalanobis}` to the substrate trainer argparse, default `kl_t2`/`std_mse` (the proven baseline). The optimal objectives are opt-in until a paired smoke ratifies. Per CLAUDE.md "Experiment design — Multiple contenders → multiple paths": ship BOTH, let the score arbitrate.
3. **`bundle.py` RendererBundle** gains `seg_distill_objective` + `pose_distill_objective` enum fields (default = current); `score_aware_loss` dispatches on them. No change to the default code path → the in-flight Z8 run is unaffected even after wire-in lands (it pins the defaults).
4. **Calibrate `τ_b` from the boundary-band sensitivity tool** (§3 / Part B) on `upstream/videos/0.mkv` before the first opt-in run (no guessed threshold — Catalog #296 Dykstra-feasibility / first-principles bound).

This is a ≤200-LOC bolt-on (HNeRV parity L7: bolt-on, not substrate-engineering). It REUSES `softmax_with_temperature` + `kl_divergence_between_softmax` (verified present).

## 4.1 Falsifiable prediction

On a paired MLX-local smoke (same substrate, same seed, same epochs, `[macOS-MLX research-signal]` only):
**boundary-weighted-TCKD seg-distill will reduce the substrate's REAL-SegNet d_seg by ≥ 8% relative vs KL-T=2.0
at the same step count**, because the gradient budget is re-allocated from `O(H·W)` score-flat interior pixels
to `O(0.05–0.15·H·W)` boundary pixels (a 6–20× per-informative-pixel gradient density increase), and d_seg is a
function of argmax flips that only occur on those boundary pixels. **Refutation**: if boundary-TCKD d_seg ≥
KL-T=2.0 d_seg (no improvement or worse) at matched steps, the boundary-concentration hypothesis is
IMPLEMENTATION-LEVEL falsified (Catalog #307) for this substrate; reactivation = check the boundary-band
fraction (if >40% the band is so wide that concentration gives no density gain) and τ_b calibration. Pose
prediction: sensitivity-Mahalanobis pose-distill reduces REAL-PoseNet d_pose by ≥ 5% relative vs std-MSE,
refuted if the per-dim dominance (§Part B tool) shows a single dim already saturating the std-scaled gradient.

## 5. Canonical-vs-unique decision per layer + cargo-cult audit per assumption

### Canonical-vs-unique decision per layer
| Layer | Decision | Rationale (falling-rule) |
|---|---|---|
| seg-distill objective | **FORK_PRINCIPLED** | canonical KL T=2.0 assumes a classification metric; the contest seg metric is argmax-flip-rate (decision-boundary functional). PRINCIPLED mismatch → boundary-weighted TCKD. |
| pose-distill objective | **ADOPT+REFINE** | per-dim-std scaling is HARD-EARNED (the empirical mean~34/std~0.6 dim-0 dominance is real). Keep it; ADD sensitivity reweight + `k≥K` zeroing. |
| student head (learnable 1×1 conv) | **ADOPT_CANONICAL** | HARD-EARNED: full-SegNet/PoseNet 2nd-order MLX autograd NaNs (documented in `mlx_loss.py`); the head is the finite-gradient surrogate. Not cargo-cult. |
| KL temperature T=2.0 | **ADOPT (with the seg objective change)** | T=2.0 is HARD-EARNED at the Quantizr 0.33 anchor; the change is WHERE the KL applies (boundary band), not T. |
| reuse `softmax_with_temperature`/`kl_divergence_between_softmax` | **ADOPT_CANONICAL** | OBVIOUS-FIT: shared kernels, no substrate-specific reason to fork. |

### Cargo-cult audit per assumption
| Assumption | HARD-EARNED vs CARGO-CULTED | Unwind path |
|---|---|---|
| "KL T=2.0 on full logits is the optimal seg-distill objective" | **CARGO-CULTED** | inherited from classification distillation; d_seg is argmax-only so NCKD is score-irrelevant. Unwind = boundary-weighted TCKD (§1.4). **This is the central finding.** |
| "T=2.0 temperature" | HARD-EARNED | Quantizr 0.33 anchor + AAA T4 §6.5; T is orthogonal to the WHERE-the-gradient-goes question. |
| "per-dim-std pose scaling" | HARD-EARNED | empirical dim-0 magnitude dominance is real; documented; verified. |
| "learnable head not full-scorer backprop" | HARD-EARNED | MLX 2nd-order NaN is documented + reproduced; the head gives finite scorer-bound gradient. |
| "distill all `out` pose dims" | CARGO-CULTED (minor) | contest uses only `:out//2`; unwind = zero-weight `k≥K`. |
| "uniform per-pair pose weight" | CARGO-CULTED (minor) | `∂S/∂d_pose` is per-pair-variable; unwind = AIL sensitivity reweight. |

**Bottom line: KL T=2.0 is HARD-EARNED as a temperature but CARGO-CULTED as the seg-distillation *functional* — the
contest's argmax-only d_seg selects boundary-weighted TCKD (DKD-α-only ⊕ BPKD-edge), not full-logit KL.** This is
a FORK_PRINCIPLED, not a kill of the teacher; the current teacher is a safe superset (`w_i≡1, full-KL`).

## 6. Observability surface

The teacher change + sensitivity tools are observable via: (1) per-layer — the boundary-weight map `w_i` is
inspectable per pixel; (2) per-signal — seg/pose/rate decomposed by the §Part B tools; (3) diff-able — the
boundary map diffs across runs (where did the band move); (4) queryable — sensitivity maps persist as
machine-readable JSON; (5) cite-able — anchored to `(archive_sha256, master_gradient anchor, upstream video sha)`;
(6) counterfactual — "if this pixel's margin were larger, would d_seg change" is the boundary-band definition.

## 7. 6-hook wire-in (Catalog #125)
1. **sensitivity-map**: Part B tools ARE new sensitivity-map contributions (per-frame/pair/region/boundary axis).
2. **Pareto constraint**: the seg objective change moves the operating point along the seg face within the existing polytope (no grammar change) — Pareto-non-binding by construction; recorded.
3. **bit-allocator**: the boundary-band map is a per-region capacity-allocation prior (allocate render capacity to boundary pixels) — ACTIVE for future substrate bit allocation.
4. **cathedral autopilot**: the sensitivity tools emit `[predicted]` non-promotable rows; consumable by the ranker but never promotable (Catalog #341).
5. **continual-learning posterior**: the falsifiable prediction (§4.1) registers a canonical-equation anchor on the paired smoke (FORMALIZATION_PENDING until the paired smoke lands).
6. **probe-disambiguator**: the two-objective ship-both design (§4 step 2) IS the disambiguator — the score arbitrates kl_t2 vs boundary_tckd.

## Sources
- Decoupled KD: Zhao et al., CVPR 2022, `arXiv:2203.08679` (Eq.6 `KD = TCKD + (1−p_t)·NCKD`, Eq.7 `DKD = α·TCKD + β·NCKD`; ar5iv-verified).
- BPKD boundary-privileged KD: Liu et al., WACV 2024 (edge-loss vs body-loss).
- Pose-regressor distillation / Attentive Imitation Loss: Saputra et al., `arXiv:1908.00858` (reliability-weighted regression imitation; softmax dark-knowledge inapplicable to regression).
- Born-Again Networks: Furlanello et al., ICML 2018 (dark-knowledge = ground-truth-rescale + dark term).
- Contest scorer: `upstream/modules.py` SegNet/PoseNet `compute_distortion` (verified directly).
- Canonical marginals: `tac.master_gradient.compute_marginal_coefficients`.
