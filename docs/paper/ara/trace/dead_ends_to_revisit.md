# trace/dead_ends_to_revisit.md

Live list of dead-ends from the exploration graph that are NOT permanently
falsified. Each entry has a revisit protocol so a future agent can decide
whether to spend compute on it.

---

## Lane M-V3 — clean retest of the rank-1 renderer-input subspace

- **dead-end id**: DE2.1
- **why on this list**: the original Lane M-V2 regression (1.84) was caused
  by a train/inference pose-pad asymmetry, not by the rank-1 hypothesis
  itself. The hypothesis remains untested at the renderer input subspace.
- **revisit protocol**:
  1. Pass `init_poses[:, 1:6]` through `_project_to_renderer_pose` at BOTH
     train and inference time (the asymmetry that caused BUG-1).
  2. Verify Check 42 STRICT passes.
  3. Run with $0.30 / 2h cap on Modal T4.
  4. Predicted contest-CUDA in [1.05, 1.20].
- **kill criterion**: any contest-CUDA score > 1.40 (decisively worse than
  Lane A 1.15).

## Lane GP polynomial-fit — DCT or B-spline basis variant

- **dead-end id**: DE2.2
- **why on this list**: degree-10 polynomial through 600 equispaced points
  has Runge oscillations at the endpoints; switching to DCT (low-frequency
  cutoff) or B-spline (piecewise low-degree) would avoid the failure mode.
- **revisit protocol**:
  1. Fit DCT-coefficients with low-frequency cutoff to the per-frame pose
     trajectory.
  2. Verify RMSE < 0.001 on held-out frames before integrating into the
     archive.
  3. Net rate gain ~14KB / 700KB = 2%.
- **kill criterion**: any RMSE > 0.005, OR net score gain < 0.01 at archive
  size budget.
- **priority**: low (2% rate gain is dominated by Era 3 portfolio gains).

## Lane UNIWARD — paired with SLI1 inflate-time decoder

- **dead-end id**: DE2.3
- **why on this list**: encoder-only UNIWARD is a no-op on the archive
  bitstream. With a paired inflate-time decoder that understands the
  cost-weighted encoding, the lane could deliver real bits-where-blind.
- **revisit protocol**:
  1. Implement SLI1 inflate-time decoder (compliance work; requires
     human approval per strict-scorer rule).
  2. Pair encoder + decoder; verify inflated bytes match the encoder's
     intended cost-weighted layout.
  3. Run as a stack with Lane G v3.
- **kill criterion**: any contest-CUDA score regression vs Lane G v3.
- **priority**: medium-low (compliance work + uncertain delta).

## Era 1 SegNet hard-argmax STE training mode

- **dead-end id**: implicit (Era 1 "SegNet attack" lane, scored 1.84)
- **why on this list**: SegNet attack with hard STE improved SegNet from
  0.006 to 0.005, but PoseNet regressed enough to net 1.84. Could be
  revisited paired with the Era 2 renderer where SegNet leverage is
  different.
- **revisit protocol**: in Era 3 portfolio (Lane SC++ already incorporates
  KL distill T=2.0; the hard-STE variant is a second-order revisit).
- **priority**: low — superseded by KL distill weight=0.002 at Lane G v3.
