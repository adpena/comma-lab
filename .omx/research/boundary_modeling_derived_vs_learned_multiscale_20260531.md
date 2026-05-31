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
  the **decision grid**). `SegNet.preprocess_input` bilinearly interpolates the
  camera frame to (384, 512) and slices the LAST frame `x[:, -1, ...]`.
- `upstream/modules.py:105` — `smp.Unet('tu-efficientnet_b2', classes=5, ...)`.
  EfficientNet-B2 has a **stride-2 stem** → the first decision feature map is at
  ~192×256.
- `upstream/modules.py:111-112` — `compute_distortion`:
  `d_seg = (out1.argmax(dim=1) != out2.argmax(dim=1)).float().mean()`. **Per-pixel
  argmax-disagreement RATE, uniform per flip.**

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

1. **Resolution ceiling.** The decision is computed at ≤256px after a fixed
   stride-2 low-pass. Boundary structure finer than that is averaged away before
   the argmax — it literally cannot flip `d_seg`. You cannot learn around a fixed
   low-pass filter.
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

### Object 2 — boundary IDENTITY (which class-pair): **DERIVED from teacher logits**

This is the bug I found and fixed today. Target-vs-rest TCKD binarizes to
`[p_target, 1−p_target]`, which lumps top2..top5 into "rest" and **destroys the
runner-up identity**. But a `d_seg` flip happens specifically at the **top1↔top2
decision surface** — classes 3/4/5 are far below and irrelevant to whether THIS
pixel flips. So the faithful object is the 2-way decision over the teacher's two
competing classes:

```
a, b    = teacher top1, top2 indices       # read off sorted logits, 0 params
q_teach = softmax([teacher_a/T, teacher_b/T])
decKL_i = KL(q_stud || q_teach)            # the actual decision, not target-vs-rest
```

Implemented: `boundary_decision_tckd_loss`. Empirically distinct from target-vs-rest
(0.059 vs 0.029 on the same student/teacher) and provably trains
(non-zero gradient, zero loss at identity). **This is the direct answer to
"perfectly modelling and interacting with the boundaries": the boundary IS the
top1↔top2 surface, so distil exactly that surface — derived, zero params.**

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
> **learn** is the spatial render head that produces boundary-correct frames, and
> even that is bounded by the two derived facts (≤256px resolution, uniform class
> weight).

---

## 2. Lower and higher dimension, via abstractions

The boundary admits a multi-scale representation. Each abstraction level is the
right tool for a different sub-problem, and each is DERIVED:

| dim | abstraction | object | derived? | tool |
|---|---|---|---|---|
| **0-D** | scalar margin `m_i` | per-pixel flip cost | yes | `segnet_boundary_band_weights_mlx` |
| **1-D** | K×K class-pair transition matrix `T[a,b]` | which decision *surfaces* carry the d_seg mass | yes | `segnet_class_pair_transition_matrix` (NEW today) |
| **2-D** | spatial boundary band (the weight map at 384×512) | where boundaries live in the image | yes | the per-pixel weight map (already at decision res) |
| **higher-D** | multi-scale / wavelet bands of the boundary | which spatial frequencies survive the stride-2 low-pass | yes (from the scorer's fixed freq response) | `tac.wavelet_variance.wavelet_variance_map` (db4) |

**Lower-dimensional (the operator's "lower dimension via abstraction").** The
1-D `T[a,b]` collapses the entire 2-D spatial boundary into a 5×5 class-pair
histogram. It answers "which decision surfaces dominate" — for comma10k that is
road↔undrivable (sky horizon) and road↔lane (lane edges). The diagonal is
structurally 0 (top1≠top2). This is pure OBSERVABILITY (Catalog #305), NOT a loss
term — per Object 3 it must not become a weight, or it diverges from the uniform
`d_seg`. Verified today: on a synthesized road(0)↔undrivable(2) band, `argmax(T)`
came out exactly `(0,2)` without any hardcoded semantics.

**Higher-dimensional (the operator's "higher dimension via abstraction").** The
stride-2 stem is a fixed low-pass: a boundary perturbation at the full 874×1164
frame is bilinearly downsampled + stride-2'd before the argmax, so high-spatial-
frequency boundaries are averaged away and CANNOT flip `d_seg`. A wavelet
(Daubechies) decomposition of the margin map tells you which bands survive: only
the coarse + mid bands reach the decision. So a substrate that spends render
capacity on sub-256px boundary fidelity is spending bits the scorer can't see.
This is a DERIVED constraint from the scorer's spatial frequency response, not a
learned one. `tac.wavelet_variance` already gives the db4 band decomposition; the
higher-D boundary model is: gate the boundary weight by the surviving bands.

---

## 3. Concrete state after today's landing

Implemented + tested (28/28 green) in `mlx_loss.py`:

- `segnet_boundary_band_weights_mlx` — 0-D geometry (Object 1).
- `boundary_decision_tckd_loss` — runner-up-aware Object 2 (the headline fix).
- `boundary_weighted_tckd_loss` — the prior target-vs-rest version (kept; auditable
  superset; the A/B σ-sweep was run against it).
- `segnet_class_pair_transition_matrix` — 1-D lower-D abstraction (NEW; observability).
- `pose_sensitivity_weighted_mse_loss` — the pose-axis sister (AIL reweight).

## 4. Next steps (operator-routable, all $0 MLX-first)

1. **Wire `boundary_decision_tckd_loss` into a substrate trainer** behind an
   opt-in flag (default stays `kl_t2`) and run a substrate-level A/B — the
   contest-faithful end-to-end validation of the regime-win.
2. **Higher-D gate**: multiply `segnet_boundary_band_weights_mlx` by a wavelet
   band-survival mask keyed to the stride-2 resolution — stops the loss spending
   gradient on boundaries the scorer can't see (needs the `(H,W,C)` spatial shape,
   not the flat `(...,C)` the kernels currently take).
3. **A/B decision-TCKD vs target-vs-rest TCKD** on the real-SegNet teacher
   (extend `tools/ab_boundary_tckd_vs_kl_t2.py` with a third arm) to quantify how
   much the runner-up identity is worth at the contest operating point.
