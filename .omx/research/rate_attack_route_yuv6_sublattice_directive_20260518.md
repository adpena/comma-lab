# RATE-ROUTE-YUV6-SUBLATTICE-20260518

Status: `research_only=true`; compact routing directive extracted from `rate_attack_novel_vectors_29_deep_research_20260518.md`.
No `.omx/state` mutation in this turn.

## Vectors

Primary: `Y1,Y2,Y4,Y7`.
Support: `A1,A2,M3`.

## Premise

PoseNet consumes two frames after a BT.601-style RGB-to-YUV420 transform and channel stack. An RGB-first packet can waste entropy on degrees of freedom that immediately collapse into six YUV channels per frame. The route tests whether a YUV6-native latent/render representation plus luma-sublattice allocator can lower charged bytes without violating full-frame output and SegNet constraints.

`prediction_only DeltaS=[-0.016,-0.004]`.

## Minimal smoke

Required outputs:

- Per-channel entropy for `Y00,Y10,Y01,Y11,U,V`.
- Per-channel and per-sublattice byte allocation proposal.
- PoseNet and SegNet component deltas with explicit axis label.
- Full-frame inflate/eval status, not only decoded latent parity.
- RGB-first baseline and YUV6-native candidate manifests.

Kill if:

- SegNet loss dominates rate/PoseNet movement.
- Candidate only proves tensor/YUV parity without official full-frame replay.
- Differentiable YUV6 path is not active during training/prototype gradient collection.

## 6-hook wire-in

- Sensitivity map: add YUV6 channel and luma-sublattice fields.
- Pareto: require component-specific deltas, not aggregate-only ranking.
- Bit allocator: allocate by channel, sublattice, frame role, and hard-pair class.
- Cathedral autopilot: rank this before post-hoc recompression.
- Continual learning: update only after empirical artifact exists.
- Probe-disambiguator: callable `rgb_first` and `yuv6_native` modes.
