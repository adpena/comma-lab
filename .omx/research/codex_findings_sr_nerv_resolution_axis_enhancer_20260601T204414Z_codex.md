# SR-NeRV Resolution-Axis Enhancer

Date: 2026-06-01T20:44:14Z
Author: Codex

## Verdict

SR-NeRV should be treated as a high-priority enhancer/design knob for the
SNeRV/HiNeRV carrier campaign, not as a separate top-priority carrier stack.
The useful principle is low-resolution carrier state plus deterministic
super-resolution to the contest-required frame size.

The reason is scorer-specific and testable: the submitted video must inflate to
the camera geometry, but the upstream scorer resizes inputs before SegNet and
PoseNet. Therefore high-frequency detail above scorer-visible resolution is a
rate sink unless it indirectly preserves a scorer decision after the scorer's
own resize path.

## Engineering Consequence

The next carrier runner should expose a resolution-axis mode:

- train/encode internal frames at scorer-visible or slightly above
  scorer-visible resolution;
- emit required `1164x874` frames through a deterministic SR/output head;
- make the SR head task-aware, not perceptual: preserve SegNet argmax and
  PoseNet geometry after the official resize/YUV6 path;
- keep the mode portable through numpy/PyTorch receiver code, with MLX used for
  local acquisition/training only;
- gate the mode with a mirror check:
  `lowres -> SR -> 1164x874 -> official scorer resize` versus
  `direct scorer-visible representation -> official scorer resize`.

This composes with SNeRV/HiNeRV, FFNeRV-flow, and BoostNeRV:

- SNeRV/HiNeRV remain the carrier candidates.
- SR-NeRV-style low-res encoding is the first enhancer to test because it
  attacks the resolution-axis rate dead-zone.
- FFNeRV-flow is a pose-channel enhancer.
- BoostNeRV is a decoder/temporal-affine synergy bolt-on after the carrier
  shows rate/score promise.

## Required Probe

Before making score claims, build a queue-owned invariance probe that records:

- archive/runtime/input hashes;
- official frame geometry and scorer input geometry;
- SegNet hard argmax delta after the full resize path;
- PoseNet delta after official YUV6 construction;
- byte estimate for low-res carrier state plus SR head;
- false-authority fields, because this is still a local design probe until
  byte-closed replay and exact auth.

If the probe says the scorer-visible output is preserved, this becomes a native
training/export option for HiNeRV/SNeRV. If it fails, the blocker should name
the exact failing axis: SegNet boundary loss, PoseNet geometry loss, SR runtime
byte overhead, or MLX/CPU mirror drift.
