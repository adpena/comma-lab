# Apogee — comma video compression challenge write-up

**Submission**: PR #107 `apogee`, score **0.2293** [contest-CUDA T4 A++],
ranked about 11th. Final official winner: **0.193** (PR #101
`hnerv_ft_microcodec`). Strongest local public-archive replay/control:
**0.20945** (PR #106 `belt_and_suspenders`).

This write-up is a tight summary of three results that have leverage beyond
this contest. The full paper, with method, architecture, ablations, and
65 pages of supporting analysis, is in the parent comma-lab repository at
`docs/paper/`. The reusable codec primitives, score-band predictor with
refusal modes, and parallel-dispatch toolchain are MIT-licensed at
<https://github.com/adpena/tac>.

---

## 1. The gradient bug worth more than any architectural change

Test-time optimization (TTO) v1–v4 showed a consistent pattern: SegNet
distortion improved steadily over 500 optimization steps; PoseNet
distortion either stagnated or got worse. We attributed this to PoseNet
being "harder to optimize" — its loss surface is noisier, it operates on
frame pairs, and ego-motion estimation is geometrically more complex than
segmentation. These are plausible explanations. They are also wrong.

The upstream scorer code (`frame_utils.py`, line 50) contains:

```python
@torch.no_grad()
def rgb_to_yuv6(x):
    ...
```

This decorator creates an autograd barrier. PoseNet's `preprocess_input()`
calls `rgb_to_yuv6` to convert RGB frames to YUV 4:2:0 (6 channels). Any
gradient flowing backward through PoseNet's loss, through the network,
through preprocessing, hits this barrier and becomes zero. **The autograd
graph is silently detached at the color-space conversion. One decorator.
Zero PoseNet gradients. Every TTO experiment in the project was blind for
weeks.**

PoseNet loss still *changed* during optimization — because SegNet gradients
moved pixels and those changes incidentally affected PoseNet output. Loss
fluctuated; it sometimes improved. It looked like noisy optimization. The
distinction between "optimizing PoseNet" and "PoseNet changing as a side
effect of optimizing SegNet" is invisible without inspecting the gradient
tensor directly.

The fix is a 6×3 differentiable BT.601 conversion matrix replacing the
`@torch.no_grad`-decorated function. **Worth 0.27 score points** (TTO
result improved from 0.70 to 0.43) — larger than any architectural change
in the project. Caught only by adversarial review when the council demanded
an explanation for why gradient descent made PoseNet worse, not better.

**The 1ms validation check** (full code in `docs/paper/03_gradient_bug.md`
§3.6): take a `requires_grad=True` clone of the input, call the model's
preprocess + forward, take a scalar loss, call `loss.backward()`, assert
`x.grad.abs().max() > 0`. One assertion catches the entire bug class. Any
project optimizing through frozen networks should run this check before
trusting any optimization result.

## 2. Game-theoretic premature convergence in deadline-bounded contests

The empirical leaderboard floor at the time of submission was approximately
**0.31** (top three contestants in a tight band), against a realistic
Shannon floor of **~0.155** derived from explicit per-component R(D)
analysis. The Yousfi-Fridrich floor (the rate-distortion bound under the
contest's task-aware distortion, accounting for SegNet/PoseNet blind spots)
lies strictly below 0.155.

The gap is structural, not effort-related. The contest has these features:
public PRs with `inflate.py` and full archive bytes, a public leaderboard
with rounded scores, a hard deadline, single-archive winner, and no formal
collaboration mechanism. These create a multi-player extensive-form game
with the following equilibrium property:

- **Early-mover information disclosure is asymmetric** — the early sharer
  (PR #55) provides architectural blueprint to all subsequent contestants.
- **Late-mover incremental refinement dominates under deadline pressure** —
  once the leader-band is competitive at ~0.31, the marginal expected score
  from incremental rate-side packer-layout refinement on a *known-good*
  paradigm is high; the marginal expected score from architectural
  exploration is low and high-variance.
- **As the deadline approaches, the variance budget for architectural
  exploration collapses to zero.**

The combination predicts that the contest converges *prematurely* to a
leader-band well above the Yousfi-Fridrich floor. Paradigm-shift candidates
(Score-Jacobian Karhunen-Loève residual coding, Ballé hyperprior over the
5-class mask symbol stream, Selfcomp's block-FP at 1.017 bpw) exist as
known techniques implementable within the contest's evaluation budget, but
do not reach contest-CUDA archive evidence in the public leaderboard
because the contest's incentive structure punishes high-variance
architectural exploration relative to incremental rate-side refinement.

This is not a critique of any individual contestant. It is a structural
observation about how multi-week public-PR contests with hard deadlines
reach floors. Full derivation in `docs/paper/06_related_work.md` §6.6 +
`docs/paper/07_discussion.md` §7.7.

## 3. The May 4 race-window planner-without-actuator failure

PR #95 (`hnerv_muon`, AaronLeslie138) — the seminal HNeRV-class submission
— published at 2026-05-04 07:47:15 UTC, scoring 0.20. The final top three
all landed within a **4 hour 8 minute window** that followed it:

| Rank | PR | Author | Created (UTC) | Lines | Files | Score |
|:----:|:----:|:--------:|:-------------:|:-----:|:-----:|:------:|
| 🥇 | #101 | SajayR | 11:50:13 | 660 | 5 | 0.193 |
| 🥈 | #103 | rem2 | 11:55:56 | **241** | **2** | 0.195 |
| 🥉 | #102 | EthanYangTW | 11:54:32 | 367 | 7 | 0.195 |

PR #105 (`kitchen_sink`, valtterivalo) — 1,776 lines across 21 files,
throwing every available technique at HNeRV — landed at 0.198 and lost to
PR #103's 241 lines. Each medalist iterated **publicly**: BradyMeighan
shipped #97 → #99 → #100 in 2h12m; rem2 went #96 → #103 in 3h24m;
EthanYangTW went #98 → #102 in 2h23m.

We had every primitive needed pre-built. PR #107 `apogee` landed at 0.229
(~11th). We did not ship a competitive submission in the 4-hour window
because we used the window to build infrastructure rather than to dispatch
candidates.

**The architectural failure**: our meta-Lagrangian engine was *conceptually*
a parallel-dispatch system — rank N candidates locally in microseconds,
select top-K, fire K dispatches in parallel to N concurrent paid GPUs,
harvest empirical anchors, reseed the calibration. With 16 concurrent
dispatches at $0.11 each, $2 buys 16 simultaneous empirical anchors per
cycle, and the loop converges in 2-3 cycles. What we built was the
**ranking layer**. What we did *not* build was the **actuator**:
`concurrent.futures.ThreadPoolExecutor` over the existing dispatch wrapper,
~150 lines. *Right design with the actuator missing.*

The deeper observation: when a metric is publicly contested with a hard
deadline, harness rigor and competitive performance trade off directly
inside the deadline window. Pre-deadline harness work is investment;
intra-window harness work is forfeit. The kitchen-sink author who threw
1,776 lines at HNeRV and lost to 241 lines is the same failure pattern at
a different layer.

Full postmortem with five lessons + reactivation criteria in
`docs/paper/99_appendix_postmortem.md`.

## 4. Practical takeaway: the closed-loop parallel-dispatch toolchain

The corrective is OSS at <https://github.com/adpena/tac> as the `tac`
library (MIT). Three composable primitives:

- **`tac.optimizer.MetaLagrangianSearch`** — Boyd-style multi-constraint
  ranker integrating a closed-form distortion proxy, a score-band predictor
  with refusal modes, and a 5-gate predispatch sanity ladder. Refused
  candidates sort to the bottom of the dispatch queue regardless of nominal
  score. Engine is deterministic, uses the exact contest score formula, and
  is arch-agnostic (accepts arbitrary calibration anchors).

- **`tac.predictor.score_band`** — predicts a contest-CUDA score band from
  rel_err and archive bytes, but *refuses* when calibration support is
  insufficient (`insufficient_anchors`, `extrapolation`,
  `lossy_better_than_lossless_incoherent`, ...). Built after the apogee_int4
  8× miss (predicted [0.155, 0.180]; landed 1.4287 [contest-CUDA]) —
  refusal is the feature, not the bug.

- **Production wrappers** in the parent comma-lab repo
  (`tools/parallel_dispatch_top_k.py`, `tools/harvest_and_reseed.py`,
  `tools/feedback_loop_sweep.py`) close the rank → dispatch → harvest →
  reseed → re-rank loop end-to-end with `--max-cycles`,
  `--max-total-cost`, `--max-cost-per-cycle`, `--top-k`,
  `--convergence-eps`, and a `--race-mode` flag that forces the
  leadership-shift prior.

Run the toy walk-through (no GPU needed):

```bash
git clone https://github.com/adpena/tac && cd tac
python examples/quickstart.py
```

It exercises the four primitives end-to-end on a synthetic problem:
build 3 calibration anchors, rank 5 candidates with the Lagrangian, trigger
the predictor's refusal modes, fan out a (mocked) parallel dispatch,
harvest and reseed.

---

## Visualizations

The 6-panel diagnostic visualization (GT Original | Reconstruction | Pixel
Error | GT SegNet masks | Our SegNet masks | SegNet Disagreement) is the
canonical evidence for the gradient-bug story and the architectural
trade-off discussion. Generation status is tracked in
`docs/paper/figures/MISSING.md` of the parent repo; if the figure has
landed by PR submission time, it should appear inline here.

The three present figures in `docs/paper/figures/`:

- `leaderboard_comparison.png` — our score trajectory vs. the public band
- `score_decomposition.png` — per-component (seg / pose / rate)
  contributions across our key milestones
- `step_curve_comparison.png` — gradient-fix before/after on the TTO loop

## Acknowledgements

Built as a one-week collaboration between one human engineer and LLM-assisted
review/code generation. The methodological pattern — the
"adversarial review council" of 10 simulated domain experts (Shannon-LEAD,
Dykstra-CO-LEAD, Yousfi, Fridrich, Contrarian, Quantizr, Hotz, Selfcomp,
MacKay, Ballé) reviewing every design decision adversarially — is documented
in `docs/paper/07_discussion.md` §7.1–§7.2 of the parent repo as the most
transferable artifact from the project. The gradient bug, in particular,
was a council finding that no unit test would have flagged.

Thanks to Yassine Yousfi and the comma.ai team for designing a contest
that doubled as a steganalysis problem — and to the public-PR culture that
let us reverse-engineer the leading submissions in detail.
