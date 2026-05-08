# Abstract

> **Claim status.** This is a community/historical-record writeup draft, not
> an arXiv/preprint commitment. Ranked score claims must come from
> `docs/paper/04_results.md` and carry `A++`/`A` evidence. Roadmap,
> derivation, prediction, and public-PR context below should stay labeled as
> non-ranking unless exact CUDA custody exists.

The comma.ai video compression challenge asks participants to compress a
60-second dashcam video so two frozen perception networks (a road segmentation
U-Net and a 6-DOF ego-motion estimator) produce outputs as close as possible
to those on the original frames. We treat the contest as **inverse
steganalysis** — the scorers are detectors. We submitted PR #107 `apogee` at
**0.2293** (about 11th place). The final official leaderboard winner was PR
#101 at 0.193. Our strongest May 4 public-archive replay/control was PR #106
at 0.20945; post-deadline PR103-on-PR106 now supersedes it as the active local
A++ HNeRV rate anchor at 0.20898.

We make three contributions. **(1)** We identify and fix a gradient
obstruction bug — `@torch.no_grad` on PoseNet's RGB-to-YUV preprocessing
silently zeroed all PoseNet gradients during test-time optimization. The fix
was worth 0.27 score points, larger than any architectural change, and was
caught only by adversarial review (§3). **(2)** We outline a Yousfi-Fridrich
rate-distortion floor hypothesis (~0.155) as a derivation/roadmap claim and
document a game-theoretic mechanism by which deadline-bounded public-PR
contests converge above this floor (§6.6, §7.7). **(3)** We document a
*planner-without-actuator* failure mode that
decided the final 4-hour race window (§7.8): we built a meta-Lagrangian
ranker with refusal modes and a sanity ladder but not the parallel-dispatch
actuator. The corrective closed-loop toolchain ships post-deadline as the OSS
`tac` library.
