# Scorer Response Surface Analysis - Source-Grounded Facts

Date: 2026-05-17
Lane: `lane_scorer_response_surface_analysis_20260517`
Author: codex
Authority: planning/control-plane only; `score_claim=false`;
`promotion_eligible=false`; `ready_for_exact_eval_dispatch=false`;
`dispatch_attempted=false`.

## Purpose

This memo closes the T4 Decision 3 gap by separating what the contest scorer
source code proves from hypotheses that still need empirical probes. The
immediate engineering consequence is that byte liveness, latent liveness, and
"feature consumed by inflate" are not enough: every Rule #6 bolt-on or
high-risk per-pair-conditioning substrate needs matched scorer-response
evidence before it changes lane status.

Follow-up implementation note: `src/tac/scorer_response_probe.py` now
normalizes historical `contest_auth_eval`-style fields
(`score_axis`/`canonical_score`/`avg_segnet_dist`/`avg_posenet_dist` plus
nested provenance) into the flat exact-eval evidence schema while preserving
axis separation. In particular, `cpu_advisory` and macOS CPU aliases normalize
to `macos_cpu_advisory`, not `[contest-CPU]`.

## Verified Source Facts

1. The official evaluation shape is two full RGB frames. `upstream/frame_utils.py:10-13`
   fixes `seq_len = 2`, camera size `(1164, 874)`, and scorer input size
   `(512, 384)`.

2. `DistortionNet.preprocess_input` receives batches shaped `b t h w c`,
   rearranges them to `b t c h w`, then sends the same two-frame tensor through
   both scorer preprocessors (`upstream/modules.py:143-148`).

3. PoseNet sees both frames. It flattens `(batch, time)` before resize, converts
   each frame to six YUV channels, then rearranges back to `b (t c) h w`
   (`upstream/modules.py:70-74`). With `seq_len=2`, PoseNet receives 12
   channels. Its distortion uses the first half of the 12 pose outputs as MSE
   (`upstream/modules.py:82-84`).

4. SegNet sees only the last frame. Its preprocess path slices `x[:, -1, ...]`
   before resize (`upstream/modules.py:107-109`), then distortion is hard
   argmax disagreement over the 5-class output (`upstream/modules.py:111-113`).

5. The upstream RGB-to-YUV helper is decorated with `@torch.no_grad()` and uses
   in-place clamps (`upstream/frame_utils.py:50-78`). Any training path that
   expects PoseNet gradient through the unmodified upstream helper is suspect
   until it proves gradient reachability with a differentiable replacement.

6. The official score formula is applied after averaging component distances:
   `100 * segnet_dist + sqrt(posenet_dist * 10) + 25 * rate`
   (`upstream/evaluate.py:89-92`). The internal scorer-response probe mirrors
   those terms explicitly (`src/tac/scorer_response_probe.py:122-150`).

## Hard Conclusions

- Frame 0 is SegNet-null but not scorer-null. It is still PoseNet-visible, so
  frame-0-only information is a PoseNet-budgeted channel, not free capacity.

- Frame 1 is both SegNet-visible and PoseNet-visible. Any artifact or residual
  on frame 1 must survive hard SegNet argmax boundaries and PoseNet YUV6
  sensitivity.

- A low-level byte/latent change can be real and still not score-lowering.
  The score-lowering question is whether the matched baseline/candidate
  comparison moves the SegNet/PoseNet terms on the same evidence axis.

- The `@torch.no_grad()` scorer preprocess is a training/optimization hazard,
  not an eval ambiguity. Contest eval is what it is; training code must either
  use a differentiable faithful surrogate or prove the intended gradient path.

## Still-Hypotheses

The following claims are not proven by source inspection alone and must remain
probe-gated:

- "SegNet stride/downsampling makes sub-grid artifacts free." Source proves
  resize to `(512, 384)` and EfficientNet-B2 UNet architecture, but not the
  safe perturbation radius or margin distribution.

- "Per-pair conditioning has at most N useful bits per pair." Source proves
  the tensor route; the information bound is a modeling claim and needs
  scorer-response evidence on real artifacts.

- "A Rule #6 bolt-on is enough to beat A1." PR101 is the external anchor, but
  A1-specific bolt-ons need matched CPU/CUDA scorer-response probes and exact
  eval custody before promotion.

## Operational Rule

For every high-risk substrate or Rule #6 bolt-on, require one of:

- `SCORER_RESPONSE_POSITIVE` from `tools/probe_substrate_score_response.py`
  on matched exact-eval evidence for the same axis; or
- `SCORER_RESPONSE_PRESENT_RATE_NEGATIVE`, which keeps the method alive as a
  byte-optimization target but does not allow a score-improvement claim; or
- an explicit `BLOCKED_*` / `NO_MEASURABLE_RESPONSE` / `SCORE_REGRESSION`
  result with reactivation criteria.

`RATE_ONLY_IMPROVEMENT` is useful, but it proves byte savings rather than
scorer-visible method response.

Historical/advisory artifacts may be used for scouting only. If the probe
reports `macos_cpu_advisory`, the result is not a `[contest-CPU]` promotion
claim and cannot be used to retire, promote, or submit a lane.

## Next Concrete Work

1. Use `tools/probe_substrate_score_response.py` as the required post-harvest
   classifier for A1 Rule #6 bolt-on cells and for NSCS/Z6/ATW-like
   per-pair-conditioning ablations.

2. Add scorer-response probe output paths to future campaign ledgers before
   paid full runs, so the harvest artifact classifies the result immediately
   instead of requiring ad hoc chat interpretation.

3. When a substrate claims frame-0 nullspace or SegNet-margin exploitation,
   require a component-separated CPU/CUDA pair: frame-0-only perturbation,
   frame-1-only perturbation, and paired both-frame perturbation.

## Verification

Source inspection performed against:

- `upstream/modules.py:70-84`, `107-113`, `143-158`
- `upstream/frame_utils.py:10-13`, `50-78`
- `upstream/evaluate.py:89-92`
- `src/tac/scorer_response_probe.py:122-150`
- `src/tac/scorer_response_probe.py:130-247` for historical artifact
  normalization and advisory-axis preservation

Focused code verification already at current `main`:

```bash
.venv/bin/python -m pytest src/tac/tests/test_scorer_response_probe.py
.venv/bin/python -m py_compile src/tac/scorer_response_probe.py tools/probe_substrate_score_response.py
```

No provider dispatch was attempted. No score claim is made.
