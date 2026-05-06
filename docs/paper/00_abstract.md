# Abstract

The comma.ai video compression challenge asks participants to compress a
60-second dashcam video so two frozen perception networks (a road segmentation
U-Net and a 6-DOF ego-motion estimator) produce outputs as close as possible
to those on the original frames. We treat the contest as **inverse
steganalysis** — the scorers are detectors. We submitted PR #107 `apogee` at
**0.2293** (about 11th place). The final official leaderboard winner was PR
#101 at 0.193; our strongest local public-archive replay/control was PR #106
at 0.20945.

We make three contributions. **(1)** We identify and fix a gradient
obstruction bug — `@torch.no_grad` on PoseNet's RGB-to-YUV preprocessing
silently zeroed all PoseNet gradients during test-time optimization. The fix
was worth 0.27 score points, larger than any architectural change, and was
caught only by adversarial review (§3). **(2)** We derive a Yousfi-Fridrich
rate-distortion floor (~0.155) and document a game-theoretic mechanism by
which deadline-bounded public-PR contests converge above this floor (§6.6,
§7.7). **(3)** We document a *planner-without-actuator* failure mode that
decided the final 4-hour race window (§7.8): we built a meta-Lagrangian
ranker with refusal modes and a sanity ladder but not the parallel-dispatch
actuator. The corrective closed-loop toolchain ships post-deadline as the OSS
`tac` library.
