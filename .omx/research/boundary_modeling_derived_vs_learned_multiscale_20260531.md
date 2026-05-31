# Boundary modelling for the optimal SegNet teacher: derived vs learned, across lower + higher dimensions

`[macOS-MLX research-signal]` design memo — answers the operator's two questions
(2026-05-31): (1) *"should the [boundary cost-maps] be learned or trained or
derived...? make sure the regions map onto what openpilot / comma.ai / the SegNet
implementation actually use; especially perfectly understanding and modelling and
interacting with the boundaries"*; (2) *"lower and higher dimension via
abstractions and whatever tools."*

Sister of `.omx/research/optimal_scorer_teacher_design_20260531T103350Z.md` (the
boundary-TCKD derivation) and the kernels in
`src/tac/substrates/hinton_distilled_scorer_surrogate/mlx_loss.py`.

NON-PROMOTABLE per Catalog #341/#192/#127/#323: training-dynamics design, NOT a
contest-score claim.

---

## 0. What the SegNet decision actually IS (the ground truth we model)

From the pinned upstream snapshot (do not edit per CLAUDE.md):

- `upstream/frame_utils.py:11` — `camera_size = (1164, 874)` (W, H of the frame).
- `upstream/frame_utils.py:13` — `segnet_model_input_size = (512, 384)` (W, H of
  the **SegNet scorer input/output argmax grid**). `SegNet.preprocess_input`
  bilinearly interpolates the camera frame to (384, 512) and slices the LAST
  frame `x[:, -1, ...]`.
- `upstream/modules.py:105` — `smp.Unet('tu-efficientnet_b2', classes=5, ...)`.
  EfficientNet-B2 has internal stride-2/downsampling stages, but that is a
  feature-extraction/receptive-field fact, **not** the distortion-grid size.
  The UNet decoder returns 5-class logits on the 384×512 grid.
- `upstream/modules.py:111-112` — `compute_distortion`:
  `d_seg = (out1.argmax(dim=1) != out2.argmax(dim=1)).float().mean()`. **Per-pixel
  argmax-disagreement RATE, uniform per flip.**
- Live contract check (2026-05-31, dummy forward, no weights needed):
  `SegNet.preprocess_input -> (1,3,384,512)`, `SegNet(...) -> (1,5,384,512)`.

The 5 comma10k classes (the openpilot/comma.ai segmentation, confirmed by the
contest community's own `upstream/submissions/v4_qp_aq2_roi/generate_qpmap.py`,
which keys on `cell == 0` for road and gives road-boundary blocks more bits):

| idx | class | spatial role |
|---|---|---|
| 0 | road | dominant drivable region (huge confident interiors) |
| 1 | lane markings | thin high-frequency structures on road |
| 2 | undrivable | sky + buildings + off-road (large uniform regions) |
| 3 | movable | vehicles / pedestrians (compact blobs) |
| 4 | my car | ego hood (fixed bottom region) |

**Two facts that are DERIVED, not chosen, and bound everything below:**

1. **Scorer grid + measured frequency response.** The distortion is the mean over
   384×512 SegNet output argmax cells after bilinear downsample from the 874×1164
   last frame. Camera-space structure below that sampling grid is attenuated by
   resize, and the EfficientNet encoder's internal downsampling/receptive fields
   add feature-path frequency bias; however, the current code does **not** justify
   the stronger claim that `d_seg` is computed at <=256px or that all higher
   boundary frequencies are invisible. High-frequency boundary signal must be
   kept as a candidate signal until a frequency-response / argmax-visual bridge
   measures which bands actually change the 384×512 argmax map.
2. **Uniform flip cost.** `d_seg` weights a road↔lane flip and a movable↔my-car
   flip **identically** (`1/N`). The contest does not care which class-pair the
   boundary is between.

---

## 1. The boundary is THREE objects, with three different answers

The naive question "should the cost-map be learned or derived?" hides that the
cost-map is not one object. Decompose it:

### Object 1 — boundary GEOMETRY (where a flip is achievable): **DERIVED, 0 params**

A pixel's argmax flips iff a perturbation crosses the decision boundary, which
costs exactly the top-2 logit margin `m_i = logit₁ − logit₂`. So
`∂d_seg/∂logit` is a Dirac AT the boundary and zero in the confident interior.
The achievability map IS the margin:

```
w_i = exp(-m_i / τ_b)          # Boltzmann flip-probability under logit noise
```

There is nothing to learn — the teacher's own logits hand you the map.
Implemented: `segnet_boundary_band_weights_mlx`.

### Object 2 — argmax decision condition: **DERIVED target, margin-calibrated surrogate**

Every soft probability-matching loss (KL / target-vs-rest TCKD / 3-bucket
decision-KD) teaches the student to match the teacher's probability softness. At
a true boundary pixel the teacher is genuinely uncertain (for example top1≈0.30,
top2≈0.28 at T=2.0), so soft losses can teach the student to stay uncertain
instead of becoming argmax-correct. They do not have the score-faithful property:
`loss = 0` iff the candidate argmax is correct.

The score target is simpler and harsher:

```
y_i          = argmax(real SegNet(source)_i)
m_impostor   = max_{j != y_i} candidate_logit_ij
hinge_i      = relu(m_impostor - candidate_logit_i,y_i + margin_i)
```

This is the Crammer-Singer multiclass hinge on raw logits. Its max over
`j != y_i` sweeps every impostor class, including out-of-pair class 3/4/5
failures that a top1/top2-only decision objective can miss. It replaces
temperature/softness with a margin buffer: zero loss means the candidate chooses
the correct class with margin. With `margin -> 0`, the zero set approaches exact
argmax correctness; with positive margin, it optimizes robustness against
codec/postfilter/runtime perturbation.

Implemented: `boundary_argmax_hinge_loss`. The older soft objectives remain useful
diagnostics and historical controls, but they are not the mathematical endpoint
for `d_seg`.

**Margin is the learnable/derived degree of freedom.** A fixed unit margin is only
the first safe default. The optimal `margin_i` should be calibrated from full
dataset evidence: source/candidate margin gaps, runtime perturbation noise,
semantic region, frame/pair ego-motion, repair/postfilter side effects, and
exact-axis drift. This is where deterministic repair policies, learned
postfilters, LoRA/adapters, and full-dataset boundary behavior models become
parallel candidate lanes: they should learn *where* to spend margin/repair, not
replace the argmax condition.

### Object 3 — class-pair IMPORTANCE (does road matter more?): **NEITHER — forbidden**

`d_seg` is uniform-per-flip. Imposing "road boundaries matter more" (exactly the
openpilot qpmap production heuristic, which gives road-boundary blocks `-5 QP`)
would optimise a DIFFERENT objective than the contest. The comma10k class
*semantics* tell you which surfaces dominate spatially (road↔undrivable is the
big one) but the *score* treats them equally. So: derive the geometry + identity,
**do NOT learn or impose class-pair weights.** This is precisely where comma.ai's
visual-quality priorities diverge from the contest scorer — and the divergence is
a TRAP for anyone who imports the qpmap intuition into a distillation loss.

### Summary

> All score-relevant boundary structure is **DERIVED** — read off the teacher's
> logits + the scorer's fixed architecture, zero params, zero training, zero risk
> of distilling the wrong functional. The only thing a substrate may legitimately
> **learn** is the spatial render head that produces boundary-correct frames. That
> learner is bounded by the derived 384×512 scorer grid, uniform class weight, and
> the empirically measured frequency response of the resize + SegNet path.

---

## 2. Lower and higher dimension, via abstractions

The boundary admits a multi-scale representation. Each abstraction level is the
right tool for a different sub-problem, and each is DERIVED:

| dim | abstraction | object | derived? | tool |
|---|---|---|---|---|
| **0-D** | scalar margin `m_i` | per-pixel flip cost | yes | `segnet_boundary_band_weights_mlx` |
| **1-D** | K×K class-pair transition matrix `T[a,b]` | which decision *surfaces* carry the d_seg mass | yes | `segnet_class_pair_transition_matrix` (NEW today) |
| **2-D** | spatial boundary band (the weight map at 384×512) | where boundaries live on the scorer grid | yes | the per-pixel weight map (already at scorer-output res) |
| **higher-D** | multi-scale / wavelet bands of the boundary | which spatial frequencies survive resize + SegNet feature path | derived + empirical | `tac.wavelet_variance.wavelet_variance_map` plus argmax/margin response probes |

**Lower-dimensional (the operator's "lower dimension via abstraction").** The
1-D `T[a,b]` collapses the entire 2-D spatial boundary into a 5×5 class-pair
histogram. It answers "which decision surfaces dominate" — for comma10k that is
road↔undrivable (sky horizon) and road↔lane (lane edges). The diagonal is
structurally 0 (top1≠top2). This is pure OBSERVABILITY (Catalog #305), NOT a loss
term — per Object 3 it must not become a weight, or it diverges from the uniform
`d_seg`. Verified today: on a synthesized road(0)↔undrivable(2) band, `argmax(T)`
came out exactly `(0,2)` without any hardcoded semantics.

**Higher-dimensional (the operator's "higher dimension via abstraction").** The
fixed facts are bilinear resize to 384×512, then an EfficientNet/UNet
encoder-decoder with internal downsampling and a 384×512 argmax output. That
architecture implies attenuation and receptive-field mixing, but it is not by
itself a proof that every sub-256px boundary feature is score-invisible. A wavelet
(Daubechies) decomposition of the margin map is still the right abstraction, but
the gate must be learned from **empirical frequency response**: perturb or compare
real frames by band, run the upstream SegNet path, and measure which bands change
the 384×512 argmax / margin maps. Until that bridge exists, high-frequency
boundary fidelity is a candidate allocation signal, not dead signal.

---

## 3. Concrete state after today's landing

Implemented + tested (28/28 green) in `mlx_loss.py`:

- `segnet_boundary_band_weights_mlx` — 0-D geometry (Object 1).
- `boundary_argmax_hinge_loss` — raw-logit all-impostor hinge (the `d_seg`-faithful
  Object 2).
- `boundary_decision_tckd_loss` — guarded top1/top2/other soft diagnostic objective.
- `boundary_weighted_tckd_loss` — the prior target-vs-rest version (kept; auditable
  soft baseline; the A/B σ-sweep was run against it).
- `segnet_class_pair_transition_matrix` — 1-D lower-D abstraction (NEW; observability).
- `pose_sensitivity_weighted_mse_loss` — the pose-axis sister (AIL reweight).

## 4. Operator-routable semantic bridge

Before discarding high-frequency boundary signal or promoting a loss mode, build a
bridge artifact that compares the contest video against a candidate inflated video
on the actual 384×512 SegNet argmax surface:

1. **Pair/component tails.** Run `tools/xray_pair_component_errors.py` on the
   candidate `inflated/` tree to rank pair-level SegNet, PoseNet, frame0 L1, and
   frame1 L1 tails. This is diagnostic only: `score_claim=false`,
   `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`.
2. **Argmax + margin maps.** Use / extend `tools/build_segnet_boundary_marginals.py`,
   `src/tac/analysis/segnet_boundary_marginals.py`, and
   `src/tac/xray/segnet_margin_polytope.py` to emit GT and inflated SegNet argmax
   maps, top-2 margins, boundary masks, and boundary mass at 384×512. Existing
   visual surfaces such as `src/tac/visualization/segnet_viz.py`,
   `src/tac/visualization/comparison_video.py`, and
   `src/tac/research/generate_visual_comparison.py` are suitable for operator
   inspection, but the machine-routing artifact must be JSON/NPY, not screenshots.
3. **Class-pair transitions.** Record both teacher top1↔top2 transition mass
   (`segnet_class_pair_transition_matrix`) and candidate-vs-GT argmax confusion
   `argmax_gt -> argmax_inflated`. Stratify by `tac.semantic_label_contract`
   names: road, lane_markings, undrivable/horizon/sky, movable objects, and my_car.
   This is the empirical answer to whether real errors are top1↔top2 boundary
   flips or spread into top3/4/5.
4. **Scene semantics.** Overlay the transition surfaces with object/road/lane/
   horizon regions and pair-level ego-motion / PoseNet tails. SegNet sees only the
   last frame; PoseNet sees the pair. Keep lane/horizon boundary flips separate
   from frame0/frame1 motion or ego-motion artifacts.
5. **Planner + MLX linkage.** Feed the bridge into acquisition surfaces such as
   `tools/build_pair_frame_scorer_geometry_lattice.py` and into MLX loss selection:
   use `boundary_argmax_hinge_loss` as the score-faithful local objective; sweep
   `margin_i` by semantic region / boundary band / pair motion; keep soft KL/TCKD
   arms as diagnostics and ablation controls; add a frequency gate only after band
   perturbations show stable argmax benefit.

This bridge remains `[macOS-MLX research-signal]` / diagnostic design evidence.
It may select local follow-up or exact-eval candidates, but it does not claim,
rank, kill, promote, or replace `[contest-CPU]` / `[contest-CUDA]` auth eval.

## 5. Next steps (operator-routable, all $0 MLX-first)

1. **Trainer objective:** route MLX scorer-aware training through
   `boundary_argmax_hinge_loss` for SegNet repair when optimizing `d_seg`; expose
   `margin` as an acquisition-controlled hyperparameter rather than a hidden
   constant. Keep KL/TCKD/decision-KD as explicit diagnostic arms.
2. **Semantic bridge artifact:** produce the JSON/NPY bridge above for at least one
   real candidate inflated video and the contest source video. Acceptance is a
   per-pair report containing component tails, GT/candidate argmax maps,
   class-pair transition matrices, margin/boundary maps, and explicit false-
   authority fields.
3. **Full-dataset repair lanes:** run deterministic postfilter/engineered repair,
   small LoRA/adapter repair, and margin-sweep lanes against the full training
   video set as `[macOS-MLX research-signal]` candidates. Promote only byte-closed
   archive/runtime candidates through receiver proof and exact CPU/CUDA gates.
4. **Frequency-response probe:** multiply `segnet_boundary_band_weights_mlx` by a
   wavelet band-survival mask only after perturbation or candidate-vs-GT band
   comparisons show which wavelet bands alter the 384×512 argmax/margin surface.
5. **A/B matrix:** extend `tools/ab_boundary_tckd_vs_kl_t2.py` into a four-arm
   matrix: KL, target-vs-rest TCKD, decision-KD, and argmax hinge, each measured
   under the observed transition regime and margin distribution.
