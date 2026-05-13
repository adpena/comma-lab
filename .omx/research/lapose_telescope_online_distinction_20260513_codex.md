# LA-Pose vs Telescope online source distinction (2026-05-13)

Status: research-only taxonomy correction
Score claim: false
Dispatch attempted: false
Promotion eligible: false

## Finding

The operator's correction is valid. LA-Pose and Telescope are different 2026
papers and should not be collapsed into one "LA-POSE foveation" concept.

## Source-verified distinction

### LA-Pose

Primary source: https://arxiv.org/abs/2604.27448
Project page: https://la-pose.github.io/

LA-Pose is "Latent Action Pretraining Meets Pose Estimation" by Zhengqing Wang,
Saurabh Nair, Prajwal Chidananda, Pujith Kachana, Samuel Li, Matthew Brown, and
Yasutaka Furukawa. The arXiv page says it was submitted on 2026-04-30 and
describes inverse- and forward-dynamics models trained on large-scale driving
videos to learn latent action representations, then repurposes those latent
actions as inputs to a camera pose estimator. The project page describes it as
a feed-forward pose estimator that converts self-supervised latent actions from
driving video into camera motion. It predicts relative camera translation,
rotation, field of view, and metric scale from motion-centric latent actions.

Pact interpretation: LA-Pose belongs to the pose/motion prior lane. It should
inform latent-action features, hard-pair motion records, ego-motion
parameterization, pose-axis bit allocation, and PoseNet-focused scorer probes.
It is not itself a foveated image-resampling paper.

### Telescope

Primary source: https://arxiv.org/abs/2604.06332
Project page: https://princeton-computational-imaging.github.io/Telescope/

Telescope is "Learnable Hyperbolic Foveation for Ultra-Long-Range Object
Detection" by Parker Ewen, Dmitriy Rivkin, Mario Bijelic, and Felix Heide. The
arXiv page says it was submitted on 2026-04-07 and introduces a two-stage
ultra-long-range autonomous-driving detector with a resampling layer and image
transformation. The project page says stage one predicts a learnable
hyperbolic foveation transform from a low-resolution image and stage two
applies it to the full-resolution image before a foundation encoder and
Deformable DETR head. The paper reports a 76 percent relative mAP improvement
for ultra-long-range detection, from 0.185 to 0.326 beyond 250 m.

Pact interpretation: Telescope belongs to the foveation/resampling lane. It
should inform hyperbolic warp parameters, foveation side payloads, SegNet
small-object/boundary sensitivity routing, and scorer-visible runtime
consumption proofs. It is not a latent-action pose-estimation paper.

### Predecessor: FOVEA

Primary source: https://openaccess.thecvf.com/content/ICCV2021/html/Thavamani_FOVEA_Foveated_Image_Magnification_for_Autonomous_Navigation_ICCV_2021_paper.html

FOVEA is the older ICCV 2021 autonomous-navigation foveated magnification
paper. It elastically magnifies regions likely to contain objects while keeping
a small detector input canvas. Telescope explicitly compares against and
extends this line with a hyperbolic/Riemannian transform and low-resolution
parameter prediction.

## Correction for current Pact naming

Existing Pact files named `lapose_foveation_*` are semantically overloaded:
they combine LA-Pose-like latent action / hard-pair motion features with
Telescope-like foveation tuple payloads. That is acceptable only as historical
implementation naming. New papers, ledgers, and dispatch decisions should use
this split:

- `lapose_motion`: latent-action camera-motion prior from LA-Pose.
- `telescope_foveation`: hyperbolic foveation / image resampling prior from
  Telescope and FOVEA.
- `lapose_plus_telescope`: composed lane where LA-Pose selects motion/hard
  pairs and Telescope supplies a byte-closed foveation payload.

## Score-lowering implication

The highest-EV contest path is not "LA-Pose foveation" as one primitive. It is
a two-part stack:

1. LA-Pose-style latent actions select where pose/motion bytes matter most
   across frame pairs.
2. Telescope-style hyperbolic foveation changes scorer-visible pixels or
   decoded geometry in those pairs.

This split keeps the math non-arbitrary: LA-Pose gives the temporal/ego-motion
latent prior; Telescope gives the spatial resampling transform. A byte-closed
candidate must prove both are consumed by `inflate.sh` and must remain
`score_claim=false` until exact auth eval.

## Immediate implementation rule

Do not dispatch or promote any candidate under the name "LA-Pose foveation"
without an explicit taxonomy field:

```json
{
  "motion_prior_source": "LA-Pose latent-action pose estimation",
  "foveation_source": "Telescope hyperbolic foveation",
  "score_claim": false,
  "ready_for_exact_eval_dispatch": false
}
```

The LFV1 local payload candidate emitted on 2026-05-13 should be described as a
Telescope-style foveation payload selected by LA-Pose-like motion telemetry, not
as paper-faithful LA-Pose.
