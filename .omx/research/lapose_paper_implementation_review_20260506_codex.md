# LA-Pose Paper Implementation Review - 2026-05-06

## Sources Reviewed

- Project page: `https://la-pose.github.io/`
- arXiv: `https://arxiv.org/abs/2604.27448`
- PDF: `https://arxiv.org/pdf/2604.27448`
- Wayve blog: `https://wayve.ai/thinking/la-pose/`
- GitHub/code search: no official public implementation link was found on the
  project page, arXiv page, Wayve blog, or author page during this pass.

## Paper Contract

LA-Pose is a two-stage camera-pose method:

1. Large-scale self-supervised latent-action pretraining from consecutive
   driving frames.
2. A pose post-training stage that discards the forward dynamics model and
   attaches a lightweight pose head to the pretrained inverse-dynamics encoder.

The paper-specific ingredients are:

- inverse-forward dynamics pretraining;
- video tokenization over 16-frame clips;
- latent action tokens from a causal ST-transformer inverse-dynamics module;
- optional 1536 -> 50 -> 1536 latent bottleneck;
- forward dynamics next-token prediction during pretraining;
- a pose head predicting relative translation, quaternion rotation, field of
  view, and metric scale;
- supervised post-training on Waymo, nuScenes, and Argoverse, with Waymo and
  PandaSet evaluation;
- known limitation around reverse motion.

## Current Repo Implementation

Current files:

- `src/tac/analysis/lapose_lite_inputs.py`
- `src/tac/analysis/lapose_motion_atoms.py`
- `src/tac/analysis/lapose_motion_evidence.py`
- compatibility wrappers under `src/tac/lapose_*.py`
- tools under `tools/build_lapose_*.py`

This is not a paper-faithful LA-Pose implementation. It is a LA-Pose-inspired
planning layer that converts contest CUDA pair metrics and component-response
artifacts into deterministic motion/action proposal atoms for meta-lagrangian
allocation. It does not train an inverse-forward dynamics model, does not
tokenize video, does not produce latent action tokens from frames, and does not
predict metric camera pose through a learned pose head.

## Fix Landed

The planning manifests now carry explicit paper-alignment metadata:

- `implementation_alignment=inspired_planning_only_not_paper_faithful_model`
- `arxiv=2604.27448`
- missing paper components list

The manifests also keep `score_claim=false`, `ready_for_exact_eval_dispatch=false`,
and a dispatch blocker `lapose_lite_is_not_paper_faithful_lapose_model`.

## Engineering Verdict

The current LA-POSE-lite surface is still useful, but only as compiler-profile
feedback:

- hard-pair routing;
- categorical/foveation/pose proposal ranking;
- openpilot/camera prior attachment;
- meta-lagrangian atom construction;
- eventual charged archive-builder policies.

It must not be described as implementing LA-Pose unless and until a real
latent-action encoder, pretraining/post-training recipe, and pose-head contract
exist.

## Data Strategy

For the fixed contest objective, the highest-EV path remains overfitting the
single scored video and exact-evaluating deterministic charged archives. Broad
comma/openpilot video data is useful when it creates a smaller or more stable
program for that same video, but it is not automatically useful just because it
is more data.

Recommended split:

- Contest/frontier lane: overfit to the one scored video, use exact CUDA
  component response, hard pairs, categorical masks, foveation, and HNeRV
  section anatomy as the optimization signal.
- LA-Pose/generalization lane: use the broader driving-video corpus for
  inverse-dynamics latent-action pretraining, then distill the learned motion
  prior into either compress-time proposal features or a tiny charged archive
  component.
- Safe bridge: treat broad-data LA-Pose features as proposal/ranking feedback
  unless the final archive consumes a deterministic charged payload and exact
  CUDA proves the score.

## Next Correct Implementation Slices

1. Add `LaposePaperContract` docs/tests that keep the current planner labeled
   as inspired/proposal-only.
2. If pursuing paper-faithful LA-Pose, implement a separate module, not by
   stretching the current planner:
   - frame-pair tokenizer/adapter;
   - inverse-dynamics encoder interface;
   - latent-action artifact manifest with model/checkpoint SHA;
   - pose-head output contract for translation/quaternion/FOV/scale;
   - contest-frame deterministic inference mode;
   - charged archive consumer if any latent/action bytes affect inflate.
3. Use paper-faithful latent actions as inputs to the existing
   meta-lagrangian/categorical/foveation stack only after their artifacts have
   source SHAs, runtime hashes, no-op controls, and exact CUDA archive evidence.
