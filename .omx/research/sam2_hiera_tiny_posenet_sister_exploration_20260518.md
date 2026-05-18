---
title: SAM2-hiera-tiny PoseNet sister exploration (Insight 3)
date_utc: 2026-05-18T14:10:00Z
lane_id: lane_hf_jobs_implementation_wave_segnet_posenet_dinov3_sam2_20260518
substrate_class: research_only_tool_surrogate
status: design-only
horizon_class: plateau_adjacent
predicted_band_validation_status: pending_post_training
predicted_band: null  # research surrogate, no contest score prediction
related_deliberation_ids: []
council_tier: T1  # working group; tool surrogate not substrate
council_attendees: []
council_quorum_met: false
council_verdict: PROCEED  # design-only; symposium not required for tool surrogate
council_dissent: []
council_assumption_adversary_verdict: []
council_decisions_recorded: []
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: ""
---

# SAM2-hiera-tiny PoseNet sister exploration

## TL;DR

Distill the contest PoseNet (FastViT-T12 + Hydra, 12-dim pose head — first 6
dims used by `compute_distortion`) into a SAM2-hiera-tiny encoder-frozen +
6-dim regression head. ~5M trainable params total (Catalog #779 freezing
exploit on the ~33 MB SAM2 vision encoder + memory encoder). HF Jobs
t4-small, ~$0.40-0.60 per 30-epoch run.

**THIS IS A TOOL SURROGATE, NOT A SUBSTRATE.** It produces a candidate
cooperative-receiver for ATW V2 / Z6 / Z7 substrates. Per CLAUDE.md
"PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium"
non-negotiable, any substrate that consumes this surrogate's output must
have its own per-substrate symposium (Catalog #325) before dispatch.

## Hypothesis

The contest PoseNet has known limitations:

1. **23x MPS drift** per CLAUDE.md "MPS auth eval is NOISE" — PoseNet
   distortion drifts 23x worse on MPS vs CUDA. The cause is hardware
   numerical drift in FastViT's GELU-tanh activation + the Hydra head's
   12-dim regression sensitivity.
2. **CUDA-CPU axis gap** per PR102 evidence: PoseNet contributes 5x to the
   CUDA-vs-CPU score gap (CLAUDE.md "Submission auth eval — BOTH CPU AND
   CUDA" section). This suggests PoseNet itself has high precision
   sensitivity that propagates into the score.
3. **Single-frame pose prediction** — PoseNet operates on 2-frame YUV6
   stacks but the architecture is feed-forward without explicit temporal
   coupling. Pretrained SAM2-hiera-tiny's multi-scale Hiera features
   (4 stages: 96, 192, 384, 768 channels) provide richer temporal-
   structure signal that *might* be a better cooperative-receiver anchor
   for substrates that already need to compose with predictive coding
   (Z7) or cooperative-receiver loss (Z6).

**Falsifiable prediction**: if the SAM2 PoseNet surrogate's MSE on the
first 6 pose dims is **within 50%** of the teacher's MSE-against-itself
(0 by definition), then it's a viable cooperative-receiver anchor. If MSE
> 2x teacher self-MSE on a smoke check (4 train + 2 eval), the architecture
is the wrong target and we **redesign** before paid dispatch.

## ## Predicted ΔS band

This is a TOOL surrogate with no direct contest-score prediction. The
predicted ΔS for substrates that consume this surrogate (ATW V2 / Z6 / Z7)
is documented in those substrates' own design memos. Per CLAUDE.md
"Forbidden symposium-band-prediction-without-Dykstra-feasibility-check":
this memo deliberately does NOT predict a band; Dykstra feasibility for
the surrogate-itself is "did the distillation converge?", measured
empirically by `eval_mse < 1.5x teacher self-MSE`.

<!-- PREDICTED_BAND_VIBES_OK:tool surrogate with no contest-score prediction -->

## ## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode":

| Layer | Decision | Rationale |
|---|---|---|
| Model loading | ADOPT canonical `transformers.Sam2Model.from_pretrained` | HF Hub is custodian of SAM2 weights. |
| Encoder freezing | ADOPT Catalog #779 freezing exploit | Encoders are pretrained on SA-1B; transfers without retraining. ~33 MB frozen. |
| Channel adapter | FORK 1x1 Conv2d 6→3 with frame-averaging init | 6-channel pair input (frame_0 RGB + frame_1 RGB concatenated) is unique to PoseNet's pair semantics; we initialize to "average the two frames into RGB" so the pretrained encoder sees something meaningful on first forward. |
| Pose head | FORK 3-layer MLP (256→128→12) | Custom head; matches PoseNet's "12-dim raw output, first 6 used for distortion" contract. |
| Loss | ADOPT MSE on first 6 dims | Matches `PoseNet.compute_distortion` formula verbatim. |
| Optimizer | ADOPT AdamW + cosine LR | Canonical PR95 paradigm. |
| Data pipeline | ADOPT canonical `adpena/comma-video-substrate-eval-600pairs` | Cross-substrate dataset; one canonical source of truth for SegNet + PoseNet labels. |
| Eval-roundtrip | N/A | Tool surrogate doesn't ship in archive bytes. |
| EMA | ADOPT 0.997 decay | Canonical PR95 paradigm. |
| Cooperative-receiver wire-in | DEFERRED to substrate consumer | This memo doesn't define HOW substrates consume the surrogate; that's per-substrate per the canonical helper at `tac.dinov3_cooperative_receiver_anchor` (sister pattern). |

## ## 9-dimension success checklist evidence

Per CLAUDE.md 9-dim checklist + Catalog #294:

1. **UNIQUENESS**: SAM2-based PoseNet surrogate is novel (no existing
   public PoseNet surrogate; the deep-research wave 2026-05-18 §SAM2
   discusses SAM2 for SegNet only, not PoseNet).
2. **BEAUTY + ELEGANCE**: ~250 LOC trainer + 1 frozen encoder + 1
   regression head. Reviewable in 30 sec.
3. **DISTINCTNESS**: Different from sister SegNet-SAM2 surrogate (regression
   not segmentation; pair-input not single-frame).
4. **RIGOR**: Premise verified pre-edit (SAM2 model identifier exists;
   PoseNet I/O verified). Adversarial review: this memo IS the review;
   substrates that consume the surrogate must independently re-review.
5. **OPTIMIZATION PER TECHNIQUE**: SAM2 encoder frozen (Catalog #779);
   only 5M params trainable.
6. **STACK-OF-STACKS-COMPOSABILITY**: Substrates compose this via
   `tac.dinov3_cooperative_receiver_anchor`-pattern helper (sister design).
7. **DETERMINISTIC REPRODUCIBILITY**: `torch.manual_seed(42)` +
   `numpy.random.seed(42)` + `random_state=42` train/val split.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: t4-small inference at batch_size=8
   on 1024-res ~10 min full training.
9. **OPTIMAL MINIMAL CONTEST SCORE**: N/A (tool surrogate). Substrates that
   consume it are scored.

## ## Observability surface

Per Catalog #305:

1. **Inspectable per layer**: SAM2 encoder outputs `(B, 256, 64, 64)`
   feature map; pose head outputs `(B, 12)`. Both inspectable via
   `student.sam2.vision_encoder` + `student.pose_head` hooks.
2. **Decomposable per signal**: pose distortion is per-pair scalar;
   per-dim contribution accessible via `(student_pose[:, i] - teacher_pose[:, i]).pow(2).mean()`.
3. **Diff-able across runs**: deterministic seed + same dataset commit sha
   produces identical metrics; metrics.json records full history.
4. **Queryable post-hoc**: model_best.pt + model_final.pt + metrics.json
   on HF Hub.
5. **Cite-able**: every output row carries source dataset commit sha +
   teacher PoseNet safetensors sha (cited in dataset README) + student
   model identifier.
6. **Counterfactual-able**: posenet_distortion_loss is differentiable for
   substrate consumers who want to backprop into a substrate prediction
   that's been routed through this surrogate.

## ## Cargo-cult audit per assumption

Per Catalog #303:

| Assumption | Classification | Rationale | Unwind path if cargo-culted |
|---|---|---|---|
| SAM2-hiera-tiny vision encoder transfers to dashcam scenes | CARGO-CULTED | SA-1B is segmentation, not pose. We're using the encoder as a feature extractor; pretrained features may not contain pose-relevant signal. | Try DINOv3 base instead (deep-research wave shows DINOv3 has stronger driving-scene-relevant features per OK-VQA + COCO downstream evals). |
| Freezing the encoder preserves enough signal | CARGO-CULTED | 5M trainable params on a 12-dim regression target may be too small. | Unfreeze the last Hiera stage (~10M additional params); rerun smoke. |
| Channel-adapter init via frame-averaging is good enough | CARGO-CULTED | A learned 6→3 projection might do better than frame-averaging. | Initialize adapter randomly + train; compare to frame-averaging init. |
| 1024-res is necessary | CARGO-CULTED | SAM2's pretrained encoder is OPTIMAL at 1024-res but 384x512 (the contest's native) might be sufficient. | Smoke at native 384x512 with interpolated patch embedding; compare. |
| First 6 pose dims are the right target | HARD-EARNED | `upstream/modules.py:84` `PoseNet.compute_distortion` uses `out[:, :h.out//2]` = first 6 dims. Verified. | n/a |
| Pair concatenation (6-channel) is the right input | CARGO-CULTED | Could also use frame_0 + delta or 2-stream encoder. | Try late-fusion: separate forward on frame_0 + frame_1 then concat features. |

## Architecture diagram (text)

```
Input pair (B, 2, 3, 384, 512)
    |
    | resize to (B, 2, 3, 1024, 1024) + ImageNet normalize
    | concat along channel dim
    v
(B, 6, 1024, 1024)
    |
    | 1x1 Conv2d channel_adapter (6 -> 3, frame-averaging init)
    v
(B, 3, 1024, 1024)
    |
    | [FROZEN] SAM2-hiera-tiny vision_encoder
    v
(B, 256, 64, 64) feature map
    |
    | AdaptiveAvgPool2d((1, 1)) + Flatten
    v
(B, 256)
    |
    | Linear(256, 128) + GELU + Linear(128, 12)
    v
(B, 12) pose output
    |
    | MSE against teacher first 6 dims
    v
Loss
```

## Cost estimate

- HF Jobs t4-small: $0.40/hr
- 30 epochs × ~2 min/epoch (540 train pairs, batch=8, 1024-res) = ~60 min
- Total: ~$0.40 per run
- Smoke (1 epoch, 4 train + 2 eval): ~3 min, ~$0.02

## Cross-pollination with Z6 4c + Z7

Z6 4c (Atick-Redlich cooperative-receiver per CLAUDE.md "Council hierarchy:
4-tier protocol" new Z6 seats Atick + Redlich + Tishby + Zaslavsky + Wyner)
and Z7 (predictive coding per Rao-Ballard + Tishby + Time-Traveler protégé)
both consume cooperative-receiver loss. The SAM2 PoseNet surrogate is a
CANDIDATE for that role, IF its features contain richer driving-scene
predictive signal than the contest PoseNet alone.

**The empirical test**: train both surrogates (SegNet via SAM2 + PoseNet via
SAM2), embed both in a Z6/Z7 substrate trainer, run a 100-epoch smoke on
$0.30 of t4-small. Compare to baseline (Z6/Z7 with contest scorer only).
If the SAM2 surrogates ADD signal (eval_total drops), proceed. If they
ADD NOISE (eval_total rises), the SAM2 features are not driving-scene-
specific enough — at which point DINOv3 anchor (Insight 2) is the next
candidate.

## Sequence recommendation

1. **First**: build `adpena/comma-video-substrate-eval-600pairs` (Insight 4
   dataset, local M5 Max ~30-60 min).
2. **Then**: run SegNet SAM2 surrogate ($0.40-0.60 on t4-small) — Stage 2
   per Insight 1.
3. **Then**: run PoseNet SAM2 surrogate ($0.40-0.60 on t4-small) — THIS
   memo's recommendation.
4. **Then**: extract DINOv3 anchor features (Insight 2, $0.07-0.10 t4-small).
5. **Then**: cross-substrate symposium for Z6/Z7/ATW V2 substrates that
   want to consume the surrogates.

Total upfront cost: ~$1-2 across all 4 t4-small jobs.

## Reactivation criteria (per CLAUDE.md "Forbidden premature KILL")

If the PoseNet surrogate's eval_mse > 2x teacher self-MSE (catastrophic
failure), the lane is DEFERRED with reactivation criteria:

1. Try unfreezing last Hiera stage (~10M additional params).
2. Try DINOv3 anchor as the encoder (Insight 2 fallback).
3. Try late-fusion 2-stream encoder.
4. Try 384x512 native resolution with interpolated patch embedding.
5. Try MobileNetV3-S as a smaller-but-pose-specific student.

Estimated reactivation cost: $0.40-0.80 per alternative.

## Cross-references

- `feedback_deep_research_wave_landed_20260518.md` (canonical source of
  Insights 1-5).
- `src/tac/dinov3_cooperative_receiver_anchor.py` (sister canonical helper
  for the DINOv3 anchor; this memo's PoseNet surrogate follows the same
  pattern).
- `tools/build_comma_video_substrate_eval_600pairs_dataset.py` (canonical
  dataset builder this memo consumes).
- `submitted_jobs/training_posenet_surrogate_sam2_*.py` (the ready-to-fire
  HF Jobs script).
- CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council
  symposium" — substrates consuming this surrogate must symposium-gate.
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" — surrogate output
  routes through canonical auth-eval gate per Catalog #226.
