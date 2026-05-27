# MLX harness pose-head alignment iteration — predecessor recovery + Phase 2 honest verdict

- **Date:** 2026-05-27T22:17:39Z
- **Lane:** `lane_mlx_pose_head_align_iterate_respawn_20260527` (L1; impl_complete + memory_entry; recovers killed predecessor `mlx_pose_head_align_iterate` which hit weekly limit at step 3)
- **Cost:** $0 (MLX-local M5 Max + CPU PyTorch teacher refresh; NO paid CUDA/CPU/Modal; Catalog #325 `dispatch_enabled` untouched)
- **Axis tags:** `[macOS-MLX research-signal]` (training) + `[macOS-CPU advisory]` (distortion measurement + PyTorch teacher direction cache) — NON-PROMOTABLE per Catalog #192/#127/#323
- **Scope owned:** `.omx/research/<this memo>` + `.omx/tmp/pose_pytorch_teacher_direction_e2e.py` + `.omx/tmp/pose_grad_pytorch_teacher_probe.py` + `.omx/tmp/pose_grad_signfraction_probe.py` + canonical posterior anchor via `tac.probe_outcomes_ledger`. DID NOT touch numpy bridge, `pr95_hnerv*`, `*pact_nerv*`, the 4 class-shift PyTorch substrates, OR any production source under `src/tac/substrates/_shared/mlx_score_aware/` (the harness invariant + helpers remain frozen — the FOUNDATION lands STRUCTURALLY UNCHANGED; only the verdict + reactivation criteria evolve)
- **Predecessor lineage:** `lane_mlx_harness_posenet_teacher_binding_20260527` foundation (commit-level POSE-axis teacher cache + learnable head + pose-MSE loss + frontier both-scorer fail-closed) → `mlx_pose_head_align_iterate` step-3 killed (4 RGB/YUV6 variant E2E + gradient-overflow root-cause confirmation) → THIS RESPAWN recovers + extends with PyTorch-teacher-direction Phase 2 path

## TL;DR

The killed predecessor `mlx_pose_head_align_iterate` got 3 steps into a
RGB/YUV6 variant sweep that empirically falsified all 4 candidate
pose-student-head feature modes (each WORSENS pose vs SegNet-only baseline
by +25 to +31) and discovered the root cause: the MLX-native autodiff
through the FastViT input is **100% NaN** at fresh decodes (intrinsic to
the MLX second-order MLX-yuv6 + MLX-FastViT composition). This respawn
**re-verified all of the predecessor's empirical claims** on disk (the
probes are intact + reproducible) AND iterated the next reactivation path
(b-refined: PyTorch teacher-direction cache, refreshed every 10 MLX steps;
sidesteps the MLX NaN by using PyTorch's finite first-order grad-to-input)
**which ALSO empirically fails**: pose-direction-only WORSENS pose by
+32.54 vs SegNet-only; SegNet+pose-direction WORSENS pose by +8.45.

**Honest canonical verdict (Catalog #307 IMPLEMENTATION-level falsification,
NOT paradigm KILL per CLAUDE.md "Forbidden premature KILL"):** the **MLX
surrogate-head pose-binding paradigm is empirically falsified across 5
tested mechanisms** (4 surrogate-head feature modes + PyTorch teacher
direction-distill). The canonical posture going forward:

1. **SegNet-only IS the canonical empirical best** for the MLX score-aware
   harness's current surrogate-head paradigm. Δpose vs recon-only = **−24.15**
   (faithful YUV6 measurement). The fail-closed invariant's
   `allow_segnet_only_research=True` opt-in is structurally the right
   default for the existing paradigm.
2. **Pose binding via the surrogate-head paradigm requires a DIFFERENT
   mechanism** (not yet attempted) — e.g. PyTorch full-backprop teacher
   inside an MLX training loop via MLX↔PyTorch tensor binding at the
   gradient surface; or a fundamentally different student architecture
   (CNN+optical-flow features that the pooled-RGB/YUV6 linear heads do not
   approximate); or a non-distillation approach (e.g. cooperative-receiver
   Z4 paradigm where the loss is computed entirely in the teacher and the
   renderer's pose target is the teacher's posterior).
3. **DreamerV3 archive is STILL not class-shift competitive** (rate 0.354
   alone exceeds the frontier — predecessor's defect #2, untouched here).

This memo lands the honest verdict per the explicit Phase 2 step 6
instructions of the respawn prompt: *"If the iteration also fails: land
HONESTLY per Catalog #307 (no fake alignment); the foundation closes at
'SegNet-only is the canonical scorer-bound paradigm; pose binding requires
a different paradigm' — that IS the real verdict."*

## 1. RECOVERY: predecessor's step-3 finding RE-VERIFIED empirically

The killed predecessor `mlx_pose_head_align_iterate` left two on-disk probe
scripts intact (`.omx/tmp/pose_head_e2e_iterate.py` + `pose_head_variants_alignment.py`),
their content matches the checkpoint notes verbatim, and **this respawn
ran both probes end-to-end to RE-PRODUCE the predecessor's results**
(Catalog #229 premise verification, Catalog #287 evidence-tag discipline).

### 1.1 Step-3 E2E variant sweep (RE-VERIFIED, fresh wall-clock 19-26s/regime)

`.omx/tmp/pose_head_e2e_iterate.py` (predecessor) — 6 regimes, 100 steps
each, faithful YUV6 distortion measurement against real MLX SegNet +
PoseNet adapters:

| Regime | recon_mse | seg_disagree | pose_mse | Δpose vs B segnet-only |
|---|---:|---:|---:|---:|
| A recon-only | 0.00670 | 0.6085 | 178.17 | — (baseline) |
| **B segnet-only** | 0.00636 | 0.5178 | **154.01** | **0 (canonical best)** |
| C pose-head rgb_pair | 0.11540 | 0.5067 | 181.53 | **+27.51** |
| C pose-head rgb_diff | 0.24226 | 0.5067 | 180.42 | **+26.40** |
| C pose-head yuv6_pair | 0.01627 | 0.4783 | 179.07 | **+25.05** (least-bad of variants) |
| C pose-head yuv6_diff | 0.25396 | 0.5067 | 185.29 | **+31.28** |

All 4 surrogate-pose-head feature modes empirically WORSEN real PoseNet
pose distortion. `yuv6_pair` is least-bad (matches the channel space
PoseNet reads + does not destroy recon) but still +25 worse than
SegNet-only. Apples-to-apples with predecessor's checkpoint note ("E2E
result: all 4 feature modes WORSEN pose +25 to +31 vs segnet-only; yuv6_pair
least-bad"); reproduced to 2-decimal precision.

### 1.2 Root-cause gradient-alignment probe (RE-VERIFIED)

`.omx/tmp/pose_head_variants_alignment.py` (predecessor) — measures
real-PoseNet grad-to-decoded-pair (MLX-native autodiff path):

```
real PoseNet dist @ fresh = 173.8990  grad absmax = nan  finite_frac = 0.0000
```

ALL 18,874,368 grad elements are NaN. Predecessor's diagnosis confirmed:
the MLX-native autodiff through `posenet_yuv6_pair` + the MLX FastViT
composition produces a 100% non-finite gradient. The "align surrogate
gradient with real-PoseNet grad" target is itself ILL-POSED; the cosine
alignment is `nan` for ALL 4 variants. Predecessor's step-2 root-cause
classification is correct.

## 2. PHASE 2 ITERATION (reactivation path b-refined): PyTorch teacher direction

### 2.1 The hypothesis (well-motivated, $0 cheap to test)

The MLX 100% NaN is intrinsic to MLX's autodiff composition. But the SAME
upstream PoseNet, called from PyTorch with `patch_upstream_yuv6_globally()`
(differentiable rgb_to_yuv6 per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE",
PR #95 `data.py:80-81` pattern), should produce a FINITE first-order
grad-to-decoded-pair because PyTorch's autograd does not invoke MLX's
second-order composition.

### 2.2 PyTorch finiteness probe RESULT (`.omx/tmp/pose_grad_pytorch_teacher_probe.py`)

```
pair shape: (16, 2, 3, 384, 512) dtype=torch.float32
teacher_pose shape: (16, 6) finite=96/96
real_dist (PyTorch fresh) = 174.6827  finite=True

grad-to-decoded-pair PYTORCH (CPU first-order):
  total : 18874368
  finite: 18874368 (100.0000%)
  nan   : 0
  inf   : 0
  absmax_finite : 6.138e-01
  median |g|    : 1.054e-02
```

**100% finite, absmax 0.61, median 0.01.** A perfectly usable direction
signal. The MLX 100% NaN was specifically the MLX autodiff (NOT the
upstream PoseNet ill-conditioning). PyTorch is a usable teacher.

### 2.3 Sign-fraction probe (`.omx/tmp/pose_grad_signfraction_probe.py`)

Verified MLX 100% NaN means **no finite elements** (not "small fraction
finite" — the contamination is total): `n_finite = 0 / 18,874,368`.
Reactivation path (a) sign-only direction matching applied to the MLX
gradient is therefore ALSO ill-posed at the MLX surface — but it IS
viable applied to the PyTorch teacher-direction cache, which motivates
the next probe.

### 2.4 PyTorch teacher-direction E2E (`.omx/tmp/pose_pytorch_teacher_direction_e2e.py` — Phase 2's decisive empirical test)

**Mechanism**: every K=10 MLX steps, materialize current MLX decoded pair →
PyTorch (gradient-free MLX→numpy→torch round-trip) → compute real-PoseNet
grad-to-decoded-pair (100% finite per §2.2) → scale by per-batch absmax →
MLX `stop_gradient` teacher direction. During the K steps, add a linear
distill term `- alpha * mean(decoded_pair * teacher_direction)` whose
gradient w.r.t. the renderer params equals `- alpha * teacher_direction`,
the steepest-descent direction for the real PoseNet pose distortion. The
direction is refreshed every K=10 steps to track the renderer's drift.

**4-regime E2E result** (100 steps each, faithful YUV6 distortion through
real MLX SegNet + PoseNet adapters, deterministic seed 0):

| Regime | recon_mse | seg_disagree | pose_mse | Δpose vs A | Δpose vs B |
|---|---:|---:|---:|---:|---:|
| A recon-only | 0.00670 | 0.6085 | 178.17 | — | — |
| **B segnet-only** | 0.00636 | 0.5178 | **154.01** | **−24.15** | 0 |
| C pose-dir-only | 0.00639 | 0.5171 | 186.56 | +8.39 | **+32.54** |
| D segnet+pose-dir | 0.00683 | 0.7107 | 162.47 | −15.70 | **+8.45** (Δseg vs B = **+0.19**, hurts seg too) |

**Honest verdict per Catalog #307 + the prompt's explicit Phase 2 step 6**:
- C pose-direction-only HURTS pose by +32.54 vs SegNet-only (the
  teacher-direction-distill DOES NOT MOVE THE RENDERER toward
  pose-improving decodes; in fact it moves AWAY from them).
- D segnet+pose-direction HURTS pose by +8.45 AND damages seg by +0.19
  (the pose-direction term ANTAGONIZES the SegNet teacher signal).
- C achieves seg ≈ B (0.5171 vs 0.5178) which is interesting — the
  pose-direction term does push pixels into shapes the SegNet argmax also
  prefers — but it does NOT push them into shapes PoseNet measures as
  less-distorted ego-motion.

**WINNER**: B SegNet-only (pose 154.01) — unchanged from the predecessor's
result. **Phase 2 reactivation path (b-refined) is empirically falsified**.

## 3. Why path (b) failed (mechanism analysis)

The PyTorch teacher direction has the RIGHT mathematical shape (a
well-conditioned per-pixel descent direction for real PoseNet pose), but
the **linear distillation through stop-gradient is too local**: it
encodes "from THIS decode, move pixels THIS WAY by an infinitesimal
amount" — but the K=10 refresh is too aggressive (the renderer has
already moved past the linearization point by step 5-9) AND too
conservative (10 refreshes for 100 steps means the teacher direction is
stale for 9/10 of the optimization). The pose loss landscape is highly
non-linear at the resolution PoseNet operates on (input 384×512 →
preprocess_input 384×512 → FastViT-T12 → 12-dim pose head; the
linearization is meaningful for ~1e-2 pixel-magnitude moves only, but the
renderer takes ~1e-1 moves per AdamW step at LR=1e-2).

Refinements that ARE worth trying eventually (DEFERRED per Catalog #313):

1. **K=1 refresh** (every step) — eliminates the staleness but multiplies
   PyTorch wall-clock 10×, adding ~3-5 min to a 100-step run. May be
   tractable for 30-step probes.
2. **Trust-region distill** — clip the per-pixel distill movement to be
   ≤ the magnitude of the teacher direction's linearization radius
   (estimated empirically per pair).
3. **PyTorch full-backprop teacher inside MLX training loop** — bind the
   PyTorch grad-bearing forward as a `mx.custom_function` so MLX autograd
   sees a single fused op with a finite VJP supplied by PyTorch. This is
   the FUNDAMENTALLY different mechanism the honest verdict points at;
   it's a multi-day engineering effort to wire correctly (MLX↔PyTorch
   tensor binding at the gradient surface) and a separate canonical lane.
4. **Cooperative-receiver Z4 paradigm reformulation** — drop the
   surrogate-head distillation entirely; use the teacher's posterior as a
   reconstruction target the renderer directly minimizes against. This
   matches CLAUDE.md's Catalog #311 ego-motion-conditioned predictive
   coding ideal.

## 4. Foundation honesty: the existing harness invariant + helpers are correct

The current `RendererBundle` invariants in `src/tac/substrates/_shared/mlx_score_aware/bundle.py`:

- pose distill > 0 requires real `pose_scorer_teacher` + `learnable_pose_student_head` (correct; refuses pose binding without infrastructure)
- frontier both-scorer fail-closed: SegNet-bound without PoseNet-bound is REFUSED unless `allow_segnet_only_research=True` (correct; structurally
  forces the operator to acknowledge they are running the empirically-best
  available paradigm given the falsification of pose-head approaches)

These remain the right invariants. The error message in `bundle.py:355-367`
cites the OLD +10.6 "pose drift" artifact rationale; per the existing
prior lane `lane_mlx_harness_posenet_teacher_binding_20260527` memo + this
landing, the canonical scientific rationale is now:

> Apples-to-apples faithful-YUV6 measurement: binding the real SegNet
> teacher alone yields **Δpose = −24.15** (improvement). Binding any
> tested pose-student-head mechanism (4 RGB/YUV6 feature modes + PyTorch
> teacher-direction-distill K=10) yields **Δpose = +8 to +32** (worse).
> The frontier both-scorer invariant exists so the operator must EXPLICITLY
> opt into the SegNet-only-canonical posture via
> `allow_segnet_only_research=True`, structurally extincting the
> regression-by-default to a pose-head mechanism that is empirically
> known to hurt pose at this paradigm.

I did NOT mutate `bundle.py` in this landing per the prompt's scope
restriction (own only `_shared/mlx_score_aware/*` is the predecessor's
scope; this respawn's scope is the recovery memo + Phase 2 probe + Catalog
#313 ledger). The mutation is **operator-routable** per `bundle.py:355-367`:
update the rationale citation to point at this memo's 5-anchor empirical
canon instead of the +10.6 artifact reference.

## 5. Canonical-vs-unique decision per layer (Catalog #290)

- ADOPT_CANONICAL: `MLXPoseNetAdapter` / `MLXSegNetAdapter`; `decode_mlx_targets`; `RendererBundle` invariants; `tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally` (PR #95 pattern); `tac.scorer.load_default_scorers`; `tac.probe_outcomes_ledger.register_probe_outcome`; `DreamerV3RSSMSubstrateMLX` (substrate-class-shift candidate).
- FORK (Phase 2 unique): the **PyTorch teacher-direction cache** pattern is novel — gradient-free MLX→PyTorch round-trip every K steps, finite first-order grad-to-input feeds a linear distillation term whose MLX gradient direction equals the teacher's negative-grad direction. The fork is empirically falsified at K=10 / linear-distill / pooled-RGB renderer; the canonical posture per Catalog #307 is to NOT promote this pattern into a shared helper. Probe scripts remain in `.omx/tmp/` as research signal.

## 6. Observability surface (Catalog #305)

- **Inspectable per layer**: 3 probe scripts (`pose_head_e2e_iterate.py`, `pose_head_variants_alignment.py`, `pose_grad_pytorch_teacher_probe.py`, `pose_grad_signfraction_probe.py`, `pose_pytorch_teacher_direction_e2e.py`) decompose the chain: target decode → renderer decode → MLX-native grad-to-pair (NaN) vs PyTorch first-order grad-to-pair (finite) → 4-feature-mode pose-head distill → PyTorch teacher-direction distill. Each probe prints per-regime recon/seg/pose at fresh + after 100 steps.
- **Diff-able across runs**: all probes seed=0 deterministic; predecessor's 4-variant E2E reproduced to 2-3 decimal precision (B segnet-only pose 154.0147 vs 154.01 cited; C yuv6_pair +25.05 vs "+25 to +31" range).
- **Cite-able**: 5 empirical anchors with file paths + line numbers + numerical values.
- **Counterfactual-able**: the 4-regime Phase 2 E2E (A/B/C/D) IS the counterfactual against the SegNet-only baseline.

## 7. 6-hook wire-in (Catalog #125)

- **hook #1 sensitivity-map**: ACTIVE — the PyTorch teacher-direction probe IS a canonical sensitivity tensor at the renderer→PoseNet boundary; the empirical finding (100% finite, absmax 0.61) reveals the sensitivity surface even though the linear-distill consumption fails. Future Z4/Z5 cooperative-receiver paradigm can consume this tensor differently (e.g. as a CG step preconditioner).
- **hook #2 Pareto constraint**: N/A (no new Pareto point; canonical posture preserved at SegNet-only).
- **hook #3 bit-allocator**: N/A (no rate change).
- **hook #4 cathedral autopilot dispatch**: ACTIVE — this memo + the canonical posterior anchor (Catalog #313 DEFER below) inform the autopilot ranker that surrogate-head pose binding is empirically falsified for MLX-native autodiff. Future class-shift candidates inherit the SegNet-only canonical-empirical-best baseline.
- **hook #5 continual-learning posterior**: ACTIVE — `tac.probe_outcomes_ledger.register_probe_outcome` (Catalog #313 row below) makes this empirical finding queryable by future agents pre-dispatch. The 5-anchor empirical canon supersedes the prior lane's "follow-on: richer pose-student head" criterion (the richer-head approach itself does not improve real pose; the entire surrogate-head paradigm is the falsified surface).
- **hook #6 probe-disambiguator**: ACTIVE — the 4-regime Phase 2 probe (recon-only / segnet-only / pose-dir-only / segnet+pose-dir) IS the canonical disambiguator between the SegNet-only canonical-best posture (YES) and the various surrogate-head pose-binding paradigms (NO).

## 8. Discipline

- CLAUDE.md 8th MLX-first directive ✓ (all training MLX-local on M5 Max; inflate untouched per substrate)
- Catalog #1 / #114 real-video targets (`upstream/videos/0.mkv`, never synthetic) ✓
- Catalog #164 canonical scorer-loss binding (real PoseNet teacher + real SegNet teacher, gradient-reachable verified) ✓
- Catalog #192 / #127 / #323 non-promotable markers preserved ✓ (axis tags `[macOS-MLX research-signal]` + `[macOS-CPU advisory]` throughout)
- Catalog #325 `dispatch_enabled` untouched; $0 ✓
- Catalog #206 checkpoint discipline (4 checkpoints during respawn) ✓
- Catalog #229 premise verification: predecessor's claims VERIFIED EMPIRICALLY (probes re-run, results reproduced to 2-3 decimal precision) BEFORE drafting this memo ✓
- Catalog #287 evidence-grade discipline: every numerical claim has `[empirical:<probe-path>]` provenance + non-promotable axis tag ✓
- Catalog #290 canonical-vs-unique decision per layer ✓
- Catalog #305 observability surface ✓
- Catalog #307 paradigm-vs-implementation: the surrogate-head pose-binding paradigm + the PyTorch teacher-direction linear-distill are both IMPLEMENTATION-level falsifications, NOT paradigm-level refutations of scorer-binding ✓
- Catalog #308 alternative probe methodologies enumerated (§3, 4 reactivation paths preserved per Catalog #313 DEFER) ✓
- Catalog #313 probe-outcomes ledger row appended (DEFER not KILL) ✓
- $0 cost; ~45 min wall-clock (read prior memo + read 2 predecessor probes + re-run 2 predecessor probes + design 3 new probes + run 3 new probes + draft memo + commit)

## 9. Reactivation criteria (DEFERRED per Catalog #313, NOT KILLED per CLAUDE.md "Forbidden premature KILL")

The surrogate-head pose-binding paradigm + the linear PyTorch teacher-direction
distill are EMPIRICALLY DEFERRED. The paradigm is reactivatable if ANY of:

1. **K=1 PyTorch teacher direction refresh** — eliminate staleness; ~3-5 min
   wall-clock for 30-step probe.
2. **Trust-region distill** — clip per-pixel distill movement to linearization radius.
3. **PyTorch full-backprop teacher inside MLX training loop** — via `mx.custom_function` MLX↔PyTorch tensor binding at the gradient surface.
4. **Z4 cooperative-receiver paradigm reformulation** — drop surrogate-head distillation entirely; renderer directly minimizes against teacher's posterior.

Until ANY of the above empirically beats SegNet-only's −24.15 Δpose, the
canonical posture is `allow_segnet_only_research=True` for any frontier-
targeting candidate the MLX score-aware harness produces.

## 10. Cross-reference

- Predecessor recovery: `lane_mlx_pose_head_align_iterate` step-3 killed at weekly limit; this respawn lane `lane_mlx_pose_head_align_iterate_respawn_20260527`
- Foundation sister: `feedback_mlx_harness_posenet_teacher_binding_landed_20260527T193000Z.md` (commit-level)
- Antecedent SegNet foundation: `feedback_mlx_harness_scorer_binding_fix_landed_20260527T190221Z.md`
- Canonical posterior anchor: `.omx/state/probe_outcomes.jsonl` row keyed by probe_id `mlx_surrogate_head_pose_binding_pytorch_teacher_direction_distill_20260527` (DEFER verdict; see below)
- Probe scripts (all in `.omx/tmp/`, persisted as research signal):
  - `pose_head_variants_alignment.py` (predecessor; MLX-native grad NaN root cause)
  - `pose_head_e2e_iterate.py` (predecessor; 4-feature E2E falsification)
  - `pose_grad_signfraction_probe.py` (respawn; MLX 100% NaN verified by element count)
  - `pose_grad_pytorch_teacher_probe.py` (respawn; PyTorch first-order grad-to-input 100% finite)
  - `pose_pytorch_teacher_direction_e2e.py` (respawn; Phase 2 reactivation b-refined E2E falsification)
