# d_seg loss-conditioning decisive test — PRE-REGISTRATION (task #63)

**Subagent:** `task63-dseg-loss-decisive`. **Written BEFORE any arm-2/arm-3 measurement** (the arm-1
argmax-CE baseline is the already-landed #62 result, reproduced here as the control). **Authority of
every number this test will produce:** `[local CPU-torch advisory]` — exact upstream PoseNet/SegNet
(`DistortionNet`) on CPU, GT decoded via `upstream/frame_utils.yuv420_to_rgb` ONLY, S-terms recomputed
from components (the rounded `final_score` lies). `[macOS-MLX research-signal]` for the conv-decoder
forward (numpy↔torch RGB parity gate). **NOT** the contest 600-sample harness → non-promotable per the
authority ladder. `$0` spend, no GPU, **no paid dispatch**, **NO MPS**. `promotable=false`,
`score_claim=false`, `ready_for_exact_eval_dispatch=false`.

**Frontier (pointer, not hardcoded):** `0.19109982` `[contest-CPU]`, 177,169 bytes (archive sha
`b4689726…`). Secondary gate: sub-0.15.

---

## 1. The decisive question

#62 proved a small `conv_pair_decoder` trained with **argmax cross-entropy** (boundary-weighted CE
against the GT SegNet argmax labels) cannot move exact `d_seg` below the flat-frame floor: trained
d_seg `0.50732` vs constant-mid-gray-frame control `0.50692` (Δ +0.0004 = ZERO improvement), while it
crushed pose 114× (d_pose 11.99→0.105). The RGB→frozen-EfficientNet-B2-SegNet→argmax-CE path carries
no usable d_seg gradient at score-native capacity.

BUT two facts contradict a "fundamental wall" reading:
- the **boundary-solver (#55)** flips SegNet argmax in *closed form* via the margin-polytope gradient
  (the real SegNet input-Jacobian `∂margin/∂pixel`);
- **PR95** (the leaderboard substrate the 0.19 cluster sits on) drove d_seg down via **KL-T=2.0
  SegNet-LOGIT distillation** — matching the teacher's full soft-logit distribution, NOT the hard
  argmax.

So the decisive question: **is the #62 d_seg wall fundamental to the small-RGB-renderer family, or is
it a wrong-loss (argmax-CE) artifact?** argmax-CE backprops through `out.argmax` only at the chosen
class; the soft-KL and the margin-hinge both expose a denser, better-conditioned gradient. If either
moves d_seg materially below the flat-frame floor where argmax-CE stalls, the cheap-small-renderer
family reopens — and the same wrong-loss diagnosis plausibly explains the contest-wide 0.19 cluster.

## 2. The test — three d_seg losses, head-to-head, matched

Train the SAME `conv_pair_decoder` (config A: seed32, ch 32-24-16-12-8, latent 24 ≈ 118 K params,
≈104 KB int8+brotli — the #62 config A so the byte/arch axis is held fixed) on the SAME pairs (8 for
the decisive smoke; 16 for a confirmation arm), with matched epochs / lr / seed / EMA-0.997 shadow /
eval_roundtrip-STE / differentiable-yuv6, varying **only the seg-loss term**:

1. **`argmax_ce`** — the #62 baseline. `mean( w_boundary · CE(student_logits, GT_argmax_labels) )`.
   The gradient touches the SegNet only through the GT-argmax-class log-prob.
2. **`kl_distill_t2`** — PR95's actual trick (canonical `tac.losses.u_die_kl.kl_distill_segnet_term`
   math). Teacher = frozen SegNet logits on **GT frame1** (the same `preprocess_input`→resize→forward
   path the d_seg metric reads). Loss = `T^2 · KL( log_softmax(student/T) ‖ softmax(teacher/T) )`,
   T=2.0. The gradient flows through the FULL soft logit distribution of all 5 classes (boundary-aware
   by construction — soft targets near a boundary carry the runner-up mass).
3. **`margin_hinge`** — the boundary-solver gradient lifted into a differentiable training term.
   For the GT-argmax source class `s = A_GT(p)` at each pixel, penalize
   `max(0, γ − ( student_logit[s] − max_{c≠s} student_logit[c] ))`. This *directly* pushes the
   student argmax margin toward (and past) the flip point, using the real SegNet input-Jacobian via
   autograd through the frozen SegNet (the same `g_p = J_{s,p} − J_{c2,p}` the #52/#55 polytope is
   written over). Boundary-weighted by the same `w_boundary`.

The recon-anchor (null-space margin-weighted MSE) and the pose term (exact 6-dim PoseNet MSE +
Jacobian-saliency-weighted recon) are **identical across all three arms** — the ONLY thing that varies
is the seg loss. Each arm re-measures EXACT `d_seg`/`d_pose` on the frozen CPU-torch scorer (GT via
`yuv420_to_rgb`) at the EMA-shadow inference checkpoint, plus the constant-frame control.

## 3. PRE-REGISTERED PREDICTION

The argmax-CE → softmax-KL → margin-hinge ordering is increasing in *gradient conditioning toward the
argmax decision boundary*. My prediction:

- **arm 1 (`argmax_ce`)** reproduces #62: d_seg stalls at the flat-frame floor (~0.50), Δ vs constant
  control < 0.01.
- **arm 3 (`margin_hinge`)** is the most likely to move d_seg, because it targets the argmax-flip
  quantity the metric reads *directly* (it is the differentiable form of the closed-form #55 flip).
  PREDICTED to move d_seg materially below the floor (I will call "material" < 0.40, i.e. a ≥ 20%
  relative reduction below the 0.507 floor, AND a monotone-down training trend).
- **arm 2 (`kl_distill_t2`)** is the PR95 mechanism; PREDICTED to move d_seg *some* (below the floor)
  but I am genuinely uncertain whether a SMALL-capacity conv RGB carrier can match the teacher logit
  distribution well enough to flip argmax pixels — PR95 used a far larger full-RGB renderer. I expect
  arm 2 > arm 1 (better than the floor) but possibly not down to medal-band.

**Net prediction: at least one of arm 2 / arm 3 moves d_seg materially below the flat-frame floor**
(i.e. the wall is a wrong-loss artifact, the cheap-small-renderer family reopens). If TRUE → PASS.

## 4. PRE-REGISTERED KILL / FAIL CRITERION

**If ALL THREE arms stall at the flat-frame floor** (every arm's exact d_seg within 20% relative of the
constant-frame control ~0.50, i.e. ≥ ~0.405, with no arm showing a monotone-down d_seg trend across
checkpoints) → **FAIL → the d_seg-via-small-RGB-renderer wall is FUNDAMENTAL.** The frozen-SegNet
argmax conditioning is not bridgeable by *any* of {hard-label CE, soft KL, margin hinge} at
score-native conv capacity. This is the publishable original finding (WHY the field clusters at 0.19 —
the frozen-SegNet-argmax conditioning wall is loss-invariant for a cheap RGB carrier) + the pivot
(defensive rate banking R1+R2+R3 + lever F floor analysis on the existing frontier).

## 5. The verdict branches (both high-value)

- **PASS** (arm 2 or 3 materially moves d_seg, e.g. < 0.1 and trending — STRONG pass; or < 0.40 and
  monotone-down — WEAK pass): the cheap-small-renderer family REOPENS; argmax-CE was the wrong loss.
  Pre-register the lever-C/AFSR-1 campaign with the WINNING loss (timing smoke → 600-pair → byte-close
  → dual exact eval, per the long-burn default). **Do NOT fire paid dispatch from this smoke.** A
  strong pass would also be a candidate explanation for the contest-wide 0.19 cluster.
- **FAIL** (all three stall): write the honest publishable analysis (the loss-invariant frozen-SegNet
  argmax conditioning wall) + recommend the pivot (R1+R2+R3 lossless rate bank + lever F floor).

## 6. Dykstra-feasibility note (Catalog #296)

The d_seg-reduction half of the feasible region is what is being probed. The closed-form #55 polytope
proves a per-pixel correction `δ` exists that flips argmax (the feasible set is nonempty at the
PIXEL/logit level); the open question this test answers is whether a small CONV RGB carrier's
representable-frame manifold *intersects* that feasible set under each of the three losses' gradient
geometry. This is a representability∩conditioning question, not a polytope-emptiness question — the
test is the empirical intersection probe (no closed-form bound asserted; this is the disambiguator).

## 7. Reused surfaces (NO new architecture; only the seg-loss term is added)

- `src/tac/boundary_math/conv_pair_decoder.py` (the #62 conv decoder — UNCHANGED).
- `tools/lever_c_train_conv_pair_decoder.py` (the #62 trainer — extended with `--seg-loss {argmax_ce,
  kl_distill_t2,margin_hinge}`; arm 1 == the existing path).
- `tac.losses.u_die_kl.kl_distill_segnet_term` math (T^2·KL, T=2.0) — arm 2 (computed inline against
  the single-frame1 SegNet path the trainer already uses, with the live GT-frame1 teacher logits).
- `src/tac/boundary_math/margin_polytope.py` (margin → boundary weight) — the `w_boundary` shared by
  arms 1 & 3 and the margin concept arm 3 differentiates.
- `tac.differentiable_eval_roundtrip` (`patch_upstream_yuv6_globally` / eval_roundtrip STE) — shared.
- `src/tac/boundary_math/posenet_jacobian_saliency.py` — the pose term (shared, UNCHANGED).
