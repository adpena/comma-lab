# MLX score-aware harness scorer-binding fix — the C6 IBPS / DreamerV3 RSSM scorer-blindness shared-foundation bug, EXTINCT

- **Date:** 2026-05-27T19:02Z
- **Lane:** `lane_mlx_harness_scorer_binding_fix_20260527` (L1; impl_complete + strict-self-protect-via-fail-closed-invariant + tests + memory_entry)
- **Cost:** $0 (MLX-local M5 Max; NO paid CUDA/CPU/Modal dispatch; Catalog #325 dispatch_enabled untouched)
- **Axis tags:** `[macOS-MLX research-signal]` (training) + `[macOS-CPU advisory]` (distortion measurement) — NON-PROMOTABLE per Catalog #192/#127/#323.
- **Scope owned:** `src/tac/substrates/_shared/mlx_score_aware/{bundle,loss,adapter,__init__}.py` + the `mlx_score_aware_full_main.py` facade + `tests/test_scorer_binding.py` + the dreamer driver/run-dir. Did NOT touch the 4 class-shift PyTorch substrates, the numpy bridge, `pr95_hnerv*`, or `*pact_nerv*`.
- **Evidence consumed:** `.omx/research/dreamer_v3_rssm_mlx_advisory_verdict_20260527T183556Z.md` (FIRE-NO; §4 defect #1 "MSE-proxy loss does not bind the scorer").

## TL;DR

The DreamerV3 RSSM advisory verdict diagnosed the symptom (decoder reconstructs
pixels well — recon-MSE 0.0055 — while SegNet/PoseNet collapse — seg 0.52, pose
185, advisory 95.7 vs frontier 0.192). I traced the ROOT CAUSE in the SHARED
`mlx_score_aware` harness ALL class-shift candidates route through, EMPIRICALLY
PROVED the harness loss was scorer-blind, FIXED it to bind the real contest
SegNet via a gradient-reachable surrogate, added gradient-reachability tests,
and VERIFIED on a short MLX-local run that scorer-binding materially moves the
SegNet axis (seg disagreement 0.61 → 0.52 in 100 steps; recon-MSE barely
changes). The fix FAILS CLOSED so no future class-shift candidate can silently
re-enter the scorer-blind trap.

## 1. DIAGNOSIS — was the loss scorer-blind? YES.

The harness's `score_aware_loss` (`loss.py`) default distillation term used
`MockTeacherLogitsProvider` — a FIXED cosine of RGB pixel means
(`cos((k*0.07 + R + 0.5G + 0.25B)*π)`). It has **ZERO SegNet/PoseNet weights**,
so it cannot encode class-boundary semantics. The DreamerV3 driver passed
`distillation_weight=0.5`, but that only activated this scorer-blind mock — so
training was effectively recon-MSE + scorer-blind noise. Identical failure mode
to C6 IBPS v1 (105.15 contest-CUDA).

**Empirical gradient-reachability probe** (`.omx/tmp/dreamer_scorer_blindness_probe.py`,
real `upstream/videos/0.mkv` targets, dreamer renderer, MLX `value_and_grad`):

| Comparison | cos vs recon grad | Interpretation |
|---|---:|---|
| mock-teacher distill (full loss) | **0.985** | ≈parallel to recon → scorer-BLIND |
| mock-teacher distill (isolated) | **+0.362** | partially recon-aligned (pixel-stat projection) |
| **real-SegNet distill (isolated)** | **−0.105** | nearly ORTHOGONAL to recon → genuinely scorer-BOUND |
| cos(real-distill, mock-distill) | **0.71** | real teacher carries info the mock cannot |

The mock distill gradient is ~parallel to / partially redundant-with recon; it
adds no SegNet class signal. The real-SegNet-bound gradient points in a
direction reconstruction does NOT already provide.

**Secondary finding (why the obvious fix doesn't work):** backprop through the
FULL ported MLX SegNet (`MLXSegNetAdapter`, a pure-MLX graph) composed with the
DreamerV3 renderer's PixelShuffle/bilinear backward produces **NaN gradients
(27/29 renderer tensors)** in MLX's second-order autograd — even though the
SegNet forward is finite and the gradient w.r.t. a leaf input is finite, and
even when the SegNet contribution is scaled to 1e-6 (so it is NOT a
cotangent-magnitude overflow; it is a structural second-order-composition NaN).
The fix must therefore bind the scorer WITHOUT full-SegNet backprop.

## 2. THE FIX — real-scorer-bound distillation via learnable student head

The canonical scorer-binding (Catalog #164 + the C6 IBPS lesson) does NOT
require backprop through the full SegNet; it requires the loss to BIND THE
SCORER'S SEMANTICS so the renderer gradient is pulled toward what the scorer
rewards. Architecture (all pieces already existed in
`tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss`; the harness simply
never wired them):

- **Teacher** = the REAL contest SegNet's per-pixel class distribution on the
  pair's TARGET SegNet frame (upstream slices the LAST frame `x[:, -1, ...]` →
  frame 1). Built ONCE pre-training via `build_mlx_segnet_pair_teacher` (NEW
  canonical harness helper): one gradient-free `MLXSegNetAdapter` forward per
  target frame, cached + indexed by pair → `RealSegNetTeacherLogitsCache`.
  Gradient-blocked (`mx.stop_gradient`).
- **Student** = `LearnableConv1x1StudentHead` (~20 params) mapping the DECODED
  frame's RGB → class logits. The renderer gradient flows
  `KL → student_head(decoded) → renderer`. The head learns
  decoded-RGB → real-SegNet-class-logits, so the renderer is pulled toward
  frames whose REAL SegNet class distribution matches the target's. **Finite
  gradient (0 NaN)** — avoids the full-SegNet-backprop NaN entirely.

### Files changed

- **`bundle.py`**: added `ScorerTeacherProvider` protocol + `scorer_teacher` /
  `learnable_student_head` / `allow_mock_scorer_teacher` /
  `segnet_teacher_frame_index` fields. **FAIL-CLOSED invariant** in
  `__post_init__`: if `distillation_weight > 0` AND no real `scorer_teacher`
  AND not `allow_mock_scorer_teacher=True` → raise. This structurally extincts
  the scorer-blind trap: a distillation term without a real teacher is REFUSED
  unless the caller explicitly accepts the scorer-blind mock for a smoke.
- **`loss.py`**: `score_aware_loss` routes the distill term through the real
  `scorer_teacher` + `learnable_student_head` when present (student on the
  contest SegNet frame; teacher gradient-blocked); falls back to the mock ONLY
  when `allow_mock_scorer_teacher`. Added `build_mlx_segnet_pair_teacher`.
- **`adapter.py`**: `train_step` trains the student head JOINTLY via a sibling
  `mx.value_and_grad` AdamW step on the same score-aware loss (the renderer is
  differentiated by the canonical `nn.value_and_grad(self.model, ...)`; the
  head's ~20 params descend the SAME loss). Verified the head params move +
  loss decreases.
- **driver**: `--scorer-binding {real_segnet,mock}` flag; default `real_segnet`
  builds the real teacher + head and routes the bundle through them. `mock` is
  the explicit scorer-blind smoke opt-in.

## 3. THE GRADIENT-REACHABILITY TEST (self-protection)

`tests/test_scorer_binding.py` (6 tests, all PASS on MLX; skip cleanly on
non-Apple-Silicon CI):

- `test_bundle_fails_closed_on_scorer_blind_distill` — distill>0 with no real
  teacher and no mock opt-in → `MlxScoreAwareHarnessError` ("SCORER-BLIND").
- `test_bundle_requires_head_when_scorer_teacher_set` — teacher without head → reject.
- **`test_real_scorer_distill_grad_is_reachable_finite_nonzero_and_scorer_bound`**
  (DECISIVE): the real-scorer distill-only gradient (a) is nonzero
  (gradient-REACHABLE through the renderer), (b) is FINITE (no full-SegNet NaN),
  (c) has `cos < 0.5` vs the recon gradient (scorer-BOUND — a direction
  reconstruction does not already provide; empirically −0.105).
- `test_real_scorer_distill_grad_differs_from_mock` — `cos(real, mock) < 0.95`
  (the real SegNet carries class info the pixel-cosine mock cannot; empirically 0.71).
- `test_adapter_trains_head_jointly_and_reduces_loss` — sibling head step works.

The fail-closed invariant + these tests are the structural extinction of the
scorer-blindness bug class for EVERY substrate that routes through the shared
harness (per CLAUDE.md "Bugs must be permanently fixed AND self-protected
against").

## 4. PROOF THE FIX MATTERS — short MLX-local verification ($0)

`.omx/tmp/dreamer_scorer_binding_verification.py` trains the SAME dreamer model
(seed 0, real video, NP=16) for 100 MLX steps under two regimes, measures
decoded-frame distortion through the REAL MLX SegNet + PoseNet adapters:

| Regime | recon_mse | seg_disagree | pose_proxy |
|---|---:|---:|---:|
| A: recon-MSE only (verdict baseline) | 0.00670 | **0.6085** | 175.6 |
| B: real-scorer-bound (the fix) | 0.00636 | **0.5178** | 186.2 |
| **Δ (B − A)** | −0.0003 | **−0.0907** | +10.6 |

**Binding the real SegNet teacher materially improves the SegNet axis** (seg
disagreement 0.61 → 0.52, ~15% relative, in 100 steps) while recon-MSE barely
changes — DIRECT empirical confirmation of the verdict's claim that the
recon-MSE minimum is uncorrelated with scorer distortion, and that scorer-binding
redirects training toward what the scorer rewards.

**Honest caveat (apples-to-apples discipline):** the teacher binds **SegNet**
only; pose moved the wrong way (+10.6) in this short run, partly because the
verification's `_posenet_inputs_yuv6` is a crude advisory proxy (not the real
YUV6 transform) and partly because no PoseNet teacher is wired yet. The harness
now has the canonical `ScorerTeacherProvider` mechanism to plug a PoseNet
teacher in identically (`MLXPoseNetAdapter` exists). A complete two-axis binding
is the named follow-on. All numbers `[macOS-MLX research-signal]` /
`[macOS-CPU advisory]` — NON-PROMOTABLE; the DreamerV3 archive is STILL not
class-shift-competitive (rate 0.354 alone exceeds the frontier — the verdict's
defect #2, which this fix does NOT address). This fix repairs the SHARED loss
foundation; it does not by itself make DreamerV3 a FIRE candidate.

## Reactivation / follow-on (DEFERRED-pending-research per Catalog #313)

1. Add a `MLXPoseNetAdapter`-backed `ScorerTeacherProvider` for the pose axis
   (the YUV6 student-input + pose-MSE distillation), so the harness binds BOTH
   contest scorer components.
2. Re-run the DreamerV3 converged run with `--scorer-binding real_segnet` to
   re-measure the advisory band; defect #2 (517 KB cat→continuous rate) must
   ALSO be addressed (per-group embedding tables) before any FIRE consideration.

## Canonical-vs-unique decision per layer (Catalog #290)

- ADOPT_CANONICAL: `MLXSegNetAdapter` (real SegNet MLX port);
  `RealSegNetTeacherLogitsCache` + `LearnableConv1x1StudentHead` +
  `hinton_distilled_kl_t2_loss` (existing canonical distillation pieces);
  `run_long_training` (EMA/telemetry/Provenance/posterior).
- FORK (harness-UNIQUE): the `scorer_teacher` + `learnable_student_head` bundle
  fields + the fail-closed scorer-blind invariant + `build_mlx_segnet_pair_teacher`
  + the sibling-head joint training in `train_step` — these are the harness's
  own canonical-vs-unique boundary extension; no canonical helper existed to
  wire the real teacher into the harness loss.

## Observability surface (Catalog #305)

- Inspectable per layer: `score_aware_loss` returns `parts` with `recon` +
  `distill` decomposed; the verification script prints per-regime seg/pose/recon.
- Diff-able across runs: the gradient-reachability probe + verification scripts
  are deterministic (seed 0).
- Cite-able: this memo + the two `.omx/tmp/` probe scripts + the test file.
- Counterfactual-able: `--scorer-binding {real_segnet,mock}` IS the
  counterfactual switch; the fail-closed invariant makes the scorer-blind path
  reachable ONLY by explicit opt-in.

## 6-hook wire-in (Catalog #125)

- hook #1 sensitivity-map: ACTIVE — the real-scorer distill gradient IS the
  per-SegNet-class sensitivity signal at the renderer boundary (replaces the
  scorer-blind pixel-cosine).
- hook #2 Pareto constraint: N/A (loss-foundation fix; no new Pareto point).
- hook #3 bit-allocator: N/A (no rate change; defect #2 untouched).
- hook #4 cathedral autopilot dispatch: ACTIVE — the fail-closed invariant
  prevents any future class-shift candidate routed through the harness from
  dispatching a scorer-blind training run.
- hook #5 continual-learning posterior: ACTIVE — this memo + the verification
  anchor; the DreamerV3 verdict's defect #1 is now CLOSED at the shared-harness
  surface (the verdict's reactivation criterion (a) is satisfied for SegNet).
- hook #6 probe-disambiguator: ACTIVE — the `_distill_only_grad` isolation +
  the real-vs-mock cos comparison IS the canonical disambiguator between
  scorer-bound and scorer-blind training.

## Discipline

- CLAUDE.md 8th MLX-first directive ✓ (training MLX-local; inflate untouched/numpy-portable).
- Catalog #1 / #114 real-video targets (`upstream/videos/0.mkv`, never synthetic) ✓.
- Catalog #164 canonical scorer-loss binding (real SegNet teacher, gradient-reachable) ✓.
- Catalog #192/#127/#323 non-promotable markers preserved ✓.
- Catalog #325 dispatch_enabled untouched; $0 ✓.
- Catalog #206 checkpoint discipline (3+ checkpoints) ✓.
- Catalog #290 canonical-vs-unique decision per layer ✓.
- Catalog #307 paradigm-vs-implementation: this is an IMPLEMENTATION-LEVEL fix
  of the shared loss foundation; the class-shift paradigm is INTACT ✓.
- $0 cost; ~25 min wall-clock (diagnosis + fix + tests + 2 verification runs).
