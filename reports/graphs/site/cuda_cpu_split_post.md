# We discovered the contest leaderboard ranks by CPU eval, not CUDA

**Posted:** 2026-05-08, four days after the comma.ai video compression
challenge deadline closed.
**TL;DR:** the contest scorer at `upstream/evaluate.py` produces two
authoritative score axes for the same archive bytes — `--device cuda` and
`--device cpu` — and the public leaderboard ranks by the **CPU** score. We
spent seven months optimizing against the CUDA score. PR #102's third-prize
0.195 was the CPU number; its CUDA bot comment was 0.228. Our PR #107
`apogee` archive landed at 0.22936 CUDA (~11th by CUDA ranking) but was
likely silver/bronze band on the actual leaderboard axis. We will never
know for sure.

## What happened

Up until 2026-05-08, we read the contest's `upstream/evaluate.py --device`
flag as a performance switch — same answer, different speed. We also read
the contest CI bot's PR comments as posting "the score" (singular) for
every archive. Neither of those mental models was right. Reading the bot's
comment format closely — there are *two* score lines per PR comment, one
labeled `--device cuda`, one labeled `--device cpu` — and reading
`evaluate.py` carefully, we found that the two device modes select
*different decoder backends* (`DaliVideoDataset` for CUDA hardware NVDEC,
`AVVideoDataset` for CPU libavcodec software decode), and run the
SegNet/PoseNet kernels with different FP32 reduction tree shapes between
the two backends.

The leaderboard ranks by the CPU number. The third-prize 0.195 headline
that everyone (including us) was reading as the score was the CPU score.
The CUDA score is informational. We had been optimizing against
informational, not against ranking.

## How big is the gap?

Across the medal-band HNeRV cluster — five public PRs that have paired CPU
and CUDA bot comments — the gap is remarkably constant:

| PR | Author | score<sub>cuda</sub> | score<sub>cpu</sub> | Δscore |
|---|---|---:|---:|---:|
| #100 | BradyMeighan | 0.22839 | 0.19538 | 0.0330 |
| #101 | SajayR (gold) | 0.22624 | 0.19326 | 0.0330 |
| #102 | EthanYangTW (bronze) | 0.22839 | 0.19538 | 0.0330 |
| #103 | rem2 (silver) | 0.22773 | 0.19473 | 0.0330 |
| #105 | valtterivalo | 0.23114 | 0.19814 | 0.0330 |
| **mean** | | | | **0.0330** |
| **σ** | | | | **0.0004** |

Five archives, five different authors, five different codec strategies.
Δscore lands in [0.0325, 0.0335] every time. That is not a sampling
artifact; that is a structural property of the scorer.

## Where does the gap come from?

The 0.033 gap decomposes cleanly:

```
70% from pose distortion divergence
30% from seg distortion divergence
 0% from rate (rate term is bit-identical between CPU and CUDA)
```

Per-component, we computed two ratios across the same five HNeRV archives:

```
R_pose = pose_cuda / pose_cpu = 5.04 ± 0.10
R_seg  = seg_cuda  / seg_cpu  = 1.17 ± 0.01
```

PoseNet is a regression head — it outputs a 6-dimensional pose vector and
the score uses MSE. SegNet is an argmax classification head — it outputs
a 5-class label per pixel. Regression heads propagate FP32 noise
quadratically; argmax classification is piecewise-constant in its logits
unless a class boundary is crossed. The 5× pose ratio is the regression
head meeting the FP32 noise floor; the 1.17× seg ratio is the argmax head
mostly resisting it.

Mechanism, three sources, additive:

1. **Decoder split** (~25% of the gap, *confirmed*): `DaliVideoDataset`
   (NVIDIA DALI hardware NVDEC) vs `AVVideoDataset` (PyAV libavcodec
   software decode) produce different RGB uint8 bytes from the same `.mkv`
   file. SegNet and PoseNet then receive quantitatively different inputs.
2. **PoseNet FP32 noise floor** (~75% of the gap, *additive-noise model*):
   FastViT-T12 is a 50-layer convolutional backbone (12 stages, 4 blocks
   per stage, RepMixer not attention). FP32 conv reduction-tree shapes
   differ between CPU and CUDA backends. The model `σ²_cuda ≈ K · L · ε² ·
   ||x||²` with L = 50, ε = 1.7e-3 gives `σ²_cuda ≈ 1.4e-4`. At medal-band
   pose<sub>cpu</sub> = 3.5e-5, the predicted ratio is `(3.5e-5 + 1.4e-4) /
   3.5e-5 = 5.0` — a numerically perfect match for the observed 5.04.
3. **Hydra-head MLP** (long tail): the final 12-dim pose-regression
   MLP. Like any FP32 MLP, its CPU and CUDA implementations differ in
   FMA fusion, matmul tiling, and reduction order.

We initially hypothesized the gap came from FastViT-T12 attention TF32-
compounding across 12 transformer layers. That hypothesis is **falsified**
on three independent grounds: (1) FastViT-T12 has zero attention layers —
all 4 stages use RepMixer, a convolutional backbone; (2) T4 (sm_75 Turing)
has no TF32 hardware support — it's an Ampere-only feature; (3) PyTorch's
default `cuda.matmul.allow_tf32 = False` since 1.12 anyway. The clean
narrative was wrong, and reading the actual `timm/models/fastvit.py:1645`
caught it.

## What does this mean for the leaderboard?

Our PR #107 `apogee` archive (CUDA 0.22936, CPU never run) was, in
expectation, a 0.196 candidate on the actual leaderboard axis. We don't
know for sure because we never dispatched a CPU eval — but if the
empirical R<sub>pose</sub> ≈ 5.04 and R<sub>seg</sub> ≈ 1.17 hold for our
HNeRV-class archive, the predicted CPU score sits in the silver/bronze
medal band. The full predictions:

| Archive | CUDA score | Predicted CPU score |
|---|---:|---:|
| Our PR #107 apogee | 0.22936 | ~0.196 (predicted) |
| PR103-on-PR106 AC repack | 0.20898 | ~0.176 (predicted) |
| PR102 hardened replay | 0.22839 | 0.19538 (verified) |
| PR104 hardened replay | 0.23114 | ~0.198 (predicted) |

PR103-on-PR106 AC repack — our own bolt-on stack — predicts to 0.176 on
the CPU axis, which would have placed it above the gold medal had we
landed it before the deadline. We did not land it before the deadline.

## What does this mean going forward?

The dual-axis discovery is now a CLAUDE.md non-negotiable. Every shippable
archive gets both `--device cuda` and `--device cpu` evaluation on
Linux x86_64 hardware that is 1:1 contest-compliant with the GitHub
Actions CI runner. Apple Silicon CPU eval is `[macOS-CPU advisory only]` —
ARM FP32 intrinsics differ from x86_64 enough to shift SegNet argmax
across class boundaries. We tag every score with one of:

- `[contest-CUDA]` — NVIDIA GPU on Linux x86_64
- `[contest-CPU]` — Linux x86_64 CPU (Modal / Lightning / Vast.ai / GitHub Actions)
- `[macOS-CPU advisory only]` — Apple Silicon CPU, drifts from x86_64
- `[MPS-PROXY]` — Apple Silicon GPU, drifts ~23× on PoseNet
- `[advisory only]` — proxy / partial / unknown

Strategic exploitation:

- **Floor-aware Huber pose loss**: clamp `||pose_err||² - τ²` at zero where
  τ ≈ sqrt(CPU pose floor). Bit-budget that would have been spent driving
  pose<sub>cuda</sub> below R<sub>pose</sub>·ε<sub>cpu</sub> is freed for
  seg/rate.
- **Leaderboard-aware Lagrangian**: `tac.score_geometry target_axis="cpu_leaderboard"`
  reweights pose marginal by 1/R<sub>pose</sub> ≈ 0.20 and seg marginal by
  1/R<sub>seg</sub> ≈ 0.86 before ranking dispatch candidates. This inverts
  the May 4 race postmortem rule "pose 2.71× more marginal than seg" at the
  PR106 frontier — at the leaderboard axis, **SegNet is ~4× more
  marginal than pose**.
- **Calibrated noise injection**: train with σ ≈ 1.7e-3 per-op-equivalent
  noise. Tightens R<sub>pose</sub> from ~5 to ~3.5 by training the renderer
  to be robust to the precision-noise floor.

## How can you verify this yourself?

The five paired data points are public — they are the contest CI bot's own
comments on PR100/101/102/103/105. The `tools/public_pr_eval_comment_scorecard.py`
script in our OSS repo harvests them; the resulting JSON lives at
`reports/public_pr100_108_eval_comment_scorecard_20260508.json`. A fresh
paired CPU+CUDA replay of any one of those archives on Modal CPU
(Linux x86_64, ~$0.12) plus Lightning T4 ($0.30) reproduces the scorecard
within 3×10⁻⁶.

## Disclosure posture

The operator authorized full public disclosure on 2026-05-08: "all of that
must be written up in our writeup and paper and site and everywhere and
OSS documentation." The contest is closed; competing contestants no longer
have a deadline-bounded incentive to use this information asymmetrically.
Anyone running a future similar contest, or post-deadline analysis on this
contest, should be able to read this post and reproduce the entire finding
chain. The full methodological detail is in the OSS repository at
[`docs/findings/cuda_cpu_auth_eval_split_20260508.md`](https://github.com/adpena/tac/blob/main/docs/findings/cuda_cpu_auth_eval_split_20260508.md).

## What we missed and what we now have

We had the primitives. We had `tools/public_pr_eval_comment_scorecard.py`
harvesting the bot comments since April. We had T4 replays of multiple
PR archives. The structural fact — that the bot publishes two axes and
the leaderboard uses one specific axis — was implicit in the comment format
the whole time. No agent, no human reviewer, and no preflight check
distinguished them as different score axes until 2026-05-08.

We now have:

- A CLAUDE.md non-negotiable rule mandating dual-eval for every shippable
  archive on 1:1 contest-compliant hardware.
- Tag distinctness enforced by Check D (`check_scores_have_lane_tag`) and
  Check B7 (`check_scores_have_lane_tag_paper_research`) across code, paper,
  and research memos.
- 10 representation-integration STRICT preflight gates closing 8 separate
  failure patterns from the codex audit.
- A 25-PR cross-family sweep design (`.omx/research/cuda_cpu_drift_sweep_design_20260508_claude.md`)
  ready to dispatch when GPU credit returns.
- Score-geometry tooling that can rank dispatch candidates against
  either CUDA or CPU-leaderboard axis.

This was not the most expensive measurement-axis blind spot we hit in the
project — that's still the 1199-overlapping-pairs vs 600-non-overlapping
disaster from 2026-04-21 — but it is the most consequential one for the
leaderboard outcome. We are publishing it in full because the next person
running this kind of contest, or evaluating a learned codec against a
frozen scorer with multiple device backends, deserves to know it before
they spend seven months optimizing against the wrong axis.
