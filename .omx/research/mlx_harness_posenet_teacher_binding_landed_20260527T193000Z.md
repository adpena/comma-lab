# MLX score-aware harness PoseNet-teacher binding — the POSE-axis foundation, WIRED + tested + honestly verified

- **Date:** 2026-05-27T19:30Z
- **Lane:** `lane_mlx_harness_posenet_teacher_binding_20260527` (L1; impl_complete + three_clean_review-via-gradient-reachability-tests + memory_entry)
- **Cost:** $0 (MLX-local M5 Max; NO paid CUDA/CPU/Modal; Catalog #325 dispatch_enabled untouched)
- **Axis tags:** `[macOS-MLX research-signal]` (training) + `[macOS-CPU advisory]` (distortion measurement) — NON-PROMOTABLE per Catalog #192/#127/#323.
- **Scope owned:** `src/tac/substrates/_shared/mlx_score_aware/{bundle,loss,adapter,__init__}.py` + `src/tac/substrates/hinton_distilled_scorer_surrogate/mlx_loss.py` (pose head + pose teacher cache + pose-MSE loss) + `tests/test_pose_scorer_binding.py` + the `.omx/tmp/` verification scripts. Did NOT touch the numpy bridge, `pr95_hnerv*`, `*pact_nerv*`, or the 4 class-shift PyTorch substrates.
- **Predecessor:** `.omx/research/mlx_harness_scorer_binding_fix_landed_20260527T190221Z.md` (SegNet teacher; commit `6e02e2875`). This is the named follow-on #1 ("add a MLXPoseNetAdapter-backed ScorerTeacherProvider for the pose axis").

## TL;DR

The SegNet teacher landed; PoseNet is the missing — and dominant-at-frontier —
axis. I built the POSE-axis sister of the SegNet teacher (exact pattern,
least-duplication): a real-contest-PoseNet teacher cache + a learnable
pose-student head + a pose-MSE distill loss, all gradient-reachable through the
renderer with a FINITE gradient (the full-FastViT-PoseNet second-order backprop
NaNs in MLX, identical to SegNet). I extended the fail-closed invariant so a
frontier-targeting candidate MUST bind BOTH scorers (with an explicit
`allow_segnet_only_research` opt-out). The mechanism is correctly wired + fully
tested (12 pose tests + the decisive gradient-reachability proof).

**Honest empirical verification (apples-to-apples, faithful YUV6):** binding the
real SegNet teacher alone ALREADY moves BOTH axes the right way — seg
−0.0907 AND **pose −24.15** (the prior SegNet-only landing's reported pose drift
of +10.6 was an ARTIFACT of its crude RGB-duplicate pose proxy; the FAITHFUL
`rgb_to_yuv6` measurement shows segnet-only IMPROVES pose). The pooled-RGB
pose-student head, however, does NOT yet IMPROVE real PoseNet pose at any tested
weight (0.01–0.5) or with head warmup — its gradient is `cos 0.185` from recon
(directionally "scorer-bound") but is NOT aligned with what improves real
PoseNet distortion. The foundation binds both axes structurally; the pose head's
gradient quality is a NAMED follow-on (richer geometry/motion-aware student).

## 1. What was built (the POSE-axis sister of the SegNet teacher)

### Teacher cache (`tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss.RealPoseNetTeacherCache` + `build_mlx_posenet_pair_teacher`)

The real contest PoseNet (PyTorch, CPU per "MPS auth eval is NOISE") runs ONE
gradient-free forward per pair's TWO target frames (PoseNet consumes the full
pair, not a single frame) via the canonical `preprocess_input` (interpolate →
`rgb_to_yuv6` → 12-channel YUV6 pair). Caches the per-pair 6-dim pose vector
(first 6 of the 12-dim pose head, matching `compute_distortion`) indexed by PAIR
index → O(1) MLX lookup per step. Satisfies the new `PoseScorerTeacherProvider`
protocol (`pose_dims` + `teacher_pose_for_indices`).

### Student head (`LearnablePoseStudentHead` + `build_learnable_pose_student_head`)

Maps the decoded frame PAIR → `(B, 6)` pose via a coarse 4×4 RGB pool per frame
+ linear projection (582 params for the canonical config). Gradient flows
pose-MSE → `pose_head(decoded_0, decoded_1)` → renderer params. The head can fit
the teacher pose well in isolation (diagnostic: head-only std-loss 318 → 0.17 in
300 steps; the per-dim bias absorbs the ~34 dim-0 depth offset). NOT full-PoseNet
backprop — that NaNs in MLX's second-order autograd composed with the renderer's
PixelShuffle/bilinear backward (the first-order grad-to-input IS finite; the
second-order composition is not — identical to the SegNet finding).

### Pose-MSE distill loss (`pose_distillation_mse_loss`)

The contest pose distortion is MSE on the first 6 pose dims (NOT KL — pose is a
continuous ego-motion vector). Adds canonical BOUNDED-AMPLIFICATION per-dim
standardization: the raw teacher per-dim std spans ~3 orders of magnitude (dim 0
std ~0.9 vs rotation dims ~0.001), so dividing by raw std would AMPLIFY the
near-constant dims ~1000× and make them dominate. The scale is floored at 10% of
the max std → amplification ratio capped at 10× (Mahalanobis-like with bounded
condition number). Without this the raw MSE (O(180)) swamps recon (O(0.006)) +
SegNet-KL (O(3)).

### Fail-closed invariant extension (`bundle.py`)

- Pose distill > 0 requires a REAL `pose_scorer_teacher` (no pose mock — pose is
  continuous, not a class distribution) AND a `learnable_pose_student_head`.
- **Frontier both-scorer invariant:** a SegNet-bound candidate that does NOT
  also bind PoseNet is REFUSED unless `allow_segnet_only_research=True`. PoseNet
  is dominant at the ~0.192 frontier (CLAUDE.md "SegNet vs PoseNet importance").

### Joint training (`adapter.py`)

The pose head trains JOINTLY via a sibling `mx.value_and_grad` AdamW step on the
SAME score-aware loss (exact SegNet-head pattern). A single trailing `mx.eval`
realizes the renderer + both heads + both optimizer states.

## 2. The gradient-reachability test (self-protection)

`tests/test_pose_scorer_binding.py` (12 tests, all PASS on MLX; skip on non-Apple
CI). The DECISIVE test
(`test_real_pose_distill_grad_is_reachable_finite_nonzero_and_scorer_bound`)
proves the pose-distill-only gradient is (1) reachable+nonzero (norm 1.78),
(2) FINITE (no full-PoseNet NaN), (3) scorer-bound (`cos 0.185 < 0.5` vs recon —
a direction reconstruction does not provide). Plus: all 5 fail-closed invariants,
teacher-cache shape+contract, wrong-size-target rejection, `pose_distill` part
emission, joint pose-head training, AND joint BOTH-heads training. 130 tests pass
across the package + hinton module (1 non-MLX-host skip); 0 regressions.

## 3. PROOF — short MLX-local verification (3 regimes, faithful YUV6, $0)

`.omx/tmp/dreamer_pose_scorer_binding_verification.py` trains the SAME dreamer
model (seed 0, real video, NP=16) 100 steps under three regimes; measures decoded
distortion through the REAL MLX SegNet + PoseNet adapters with the FAITHFUL
`rgb_to_yuv6` transform (NOT the crude RGB-duplicate proxy the SegNet-only
verification used):

| Regime | recon_mse | seg_disagree | pose_mse | Δseg vs A | Δpose vs A |
|---|---:|---:|---:|---:|---:|
| A: recon-only | 0.00670 | 0.6085 | 178.17 | — | — |
| B: SegNet-only (real SegNet teacher) | 0.00636 | 0.5178 | 154.01 | **−0.0907** | **−24.15** |
| C: both-scorer (SegNet + PoseNet, bounded-scale) | 0.11540 | 0.5067 | 181.53 | −0.1018 | **+3.36** |

**Two findings:**

1. **The faithful YUV6 measurement CORRECTS the predecessor's +10.6 pose-drift
   artifact.** SegNet-only ACTUALLY IMPROVES pose (−24.15 here). The
   predecessor's reported `pose +10.6` was an artifact of its
   `_posenet_inputs_yuv6` crude RGB-duplicate proxy (explicitly flagged as
   advisory in its own code). Apples-to-apples discipline: with the real
   contest preprocessing, binding the SegNet teacher already moves the pose
   axis the right way (pose distortion is correlated with overall reconstruction
   fidelity at this operating point).

2. **The pooled-RGB pose-student head does NOT yet IMPROVE real PoseNet pose.**
   At weight 0.5 (and swept to 0.01, and with 300-step pose-head warmup) the
   pose binding hurts both recon (0.006 → 0.04–0.16) and real PoseNet pose
   (+3 to +33). The pose-distill gradient is directionally "scorer-bound"
   (cos 0.185 from recon) but is NOT aligned with what improves real PoseNet
   distortion: the head learns "what pooled RGB predicts the teacher pose," and
   backpropping that to the renderer changes pooled colors in ways that damage
   the geometric/motion content PoseNet actually measures.

**Honest verdict:** the FOUNDATION (teacher + head + loss + fail-closed
both-scorer invariant + joint training + tests) is sound and binds both axes
structurally. The pose head's gradient QUALITY is the named follow-on: the
pooled-RGB linear student is too weak/wrong an architecture to provide a
pose-IMPROVING gradient. A richer geometry/motion-aware pose student (e.g. a
small CNN on the decoded pair, or a flow/displacement feature) is required before
the pose-distill gradient improves real PoseNet pose. All numbers
`[macOS-MLX research-signal]` / `[macOS-CPU advisory]` — NON-PROMOTABLE.

## Reactivation / follow-on (DEFERRED-pending-research per Catalog #313)

1. Richer pose-student head: a small CNN / displacement-feature head on the
   decoded pair (the pooled-RGB linear head is geometry-blind). The teacher cache
   + loss + fail-closed wiring are already in place; only the head architecture
   changes.
2. Re-run the 3-regime verification with the richer head; the pose binding must
   IMPROVE real PoseNet pose over segnet-only (Δpose < 0) before any FIRE
   consideration.
3. The DreamerV3 archive is STILL not class-shift-competitive (rate 0.354 alone
   exceeds the frontier — the predecessor's defect #2, untouched here).

## Canonical-vs-unique decision per layer (Catalog #290)

- ADOPT_CANONICAL: `MLXPoseNetAdapter` (real PoseNet MLX port); PyTorch
  `rgb_to_yuv6` + `PoseNet.preprocess_input` for the teacher cache build;
  `hinton_distilled_scorer_surrogate` module home (sister of the SegNet head);
  `run_long_training` (EMA/telemetry/Provenance/posterior via the adapter).
- FORK (harness-UNIQUE): the `pose_scorer_teacher` + `learnable_pose_student_head`
  + `pose_distillation_weight` bundle fields + the frontier both-scorer
  fail-closed invariant + `build_mlx_posenet_pair_teacher` + `LearnablePoseStudentHead`
  + `pose_distillation_mse_loss` (bounded-amplification standardization) + the
  sibling pose-head joint training. Pose is MSE-not-KL + global-not-per-pixel, so
  the SegNet head could not be reused; this is the canonical pose-axis extension.

## Observability surface (Catalog #305)

- Inspectable per layer: `score_aware_loss` `parts` decomposes `recon` +
  `distill` (SegNet-KL) + `pose_distill` (pose-MSE); the verification prints
  per-regime seg/pose/recon.
- Diff-able across runs: the gradient-reachability probe + 3-regime verification
  + weight sweep + warmup probe are deterministic (seed 0).
- Cite-able: this memo + `tests/test_pose_scorer_binding.py` + 3 `.omx/tmp/`
  scripts (`dreamer_pose_scorer_binding_verification.py`, `pose_weight_sweep.py`,
  `pose_warmup_verification.py`).
- Counterfactual-able: the 3-regime (recon / segnet / both) verification IS the
  counterfactual; `allow_segnet_only_research` is the explicit opt-out switch.

## 6-hook wire-in (Catalog #125)

- hook #1 sensitivity-map: ACTIVE — the pose-distill gradient IS the per-pose-dim
  sensitivity signal at the renderer boundary (the POSE-axis sister of the SegNet
  per-class signal).
- hook #2 Pareto constraint: N/A (loss-foundation extension; no new Pareto point).
- hook #3 bit-allocator: N/A (no rate change).
- hook #4 cathedral autopilot dispatch: ACTIVE — the frontier both-scorer
  fail-closed invariant prevents a future frontier-targeting candidate from
  dispatching with only the SegNet axis bound (the dominant pose axis left
  unbound).
- hook #5 continual-learning posterior: ACTIVE — this memo + the verification
  anchor; the predecessor's reactivation criterion #1 (pose teacher) is now
  SATISFIED at the mechanism level (the gradient-quality refinement is the new
  open criterion).
- hook #6 probe-disambiguator: ACTIVE — the `_pose_distill_only_grad` isolation +
  the cos-vs-recon comparison + the 3-regime verification IS the canonical
  disambiguator between "pose mechanism wired" (YES) and "pose gradient improves
  real PoseNet pose" (NOT YET — the open follow-on).

## Discipline

- CLAUDE.md 8th MLX-first directive ✓ (training MLX-local; inflate untouched).
- Catalog #1 / #114 real-video targets (`upstream/videos/0.mkv`, never synthetic) ✓.
- Catalog #164 canonical scorer-loss binding (real PoseNet teacher, gradient-reachable) ✓.
- Catalog #192/#127/#323 non-promotable markers preserved ✓.
- Catalog #325 dispatch_enabled untouched; $0 ✓.
- Catalog #206 checkpoint discipline (8 checkpoints) ✓.
- Catalog #290 canonical-vs-unique decision per layer ✓.
- Catalog #307 paradigm-vs-implementation: the pose-head gradient-quality gap is
  an IMPLEMENTATION-LEVEL finding (richer head needed), NOT a paradigm-level
  refutation of scorer-binding ✓.
- Apples-to-apples discipline: the faithful-YUV6 measurement CORRECTED the
  predecessor's crude-proxy pose-drift artifact; reported honestly ✓.
- $0 cost; ~30 min wall-clock (build + tests + 3 verification passes).
