# ROI teacher-first notes

## why this lane still matters

Region-aware coding remains a viable lane because the challenge score is task-based rather than human-visual-quality based. The right map is not simply "center good, sky bad". It is a teacher-sensitivity map after the evaluator's own resize and preprocessing steps.

## recommended ladder

1. Build teacher-first heatmaps:
   - patch-drop or patch-degrade ablations
   - score-delta estimates against SegNet / PoseNet outputs
2. Compare against cheap priors:
   - temporal variance
   - optical flow
   - encoder motion vectors / bitrate metadata
   - edge density
   - horizon / center priors
3. Only then add heavier mask proposal tools:
   - segmentation / tracking tools as offline mask stabilizers only

## implementation guidance

- keep heavy models out of `inflate.sh` unless a future measured result clearly justifies the byte/runtime burden
- prefer one strong global base stream plus spatial quality shaping or sparse residual/overlay channels over tiled mixed-codec layouts
- if revisiting parity-aware or luma/chroma-aware ROI, bias protection toward luma edges and motion-sensitive areas first

## specific caution

Do not automatically down-rank the near-road region in front of the car. In forward dashcam data, it can carry strong motion, lane texture, and expansion cues that matter to the temporal teacher even if it looks visually boring to a human.

## current decision

This lane stays speculative until a cheaper side-information story beats the current honest Track B floor.
