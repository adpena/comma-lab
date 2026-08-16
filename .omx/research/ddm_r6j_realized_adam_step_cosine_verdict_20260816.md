# R6 judge repair, leg 3 — the realized-AdamW step cosine

**verdict:** `JUDGE_DEAD_SURVIVES_STEP_IS_GENUINELY_ROTATED` (6 of 6 matched steps)
**verdict_scope:** INSTANCE — two arms (stock `ddm_lr1/A2`, band `ddm_rg1/band_a1`),
600 steps, this init, MPS-trained checkpoints read back on CPU.
**axis:** `[macOS-CPU advisory]` — read-back of retained payloads. NEVER a score.
**own-vehicle frontier UNMOVED:** hv1 ep0634 S 0.15959729295498598 @ 182,759 B `[contest-CUDA T4 n600]`.

Receipt: `/Volumes/APDataStore/pact/ddm_r6j_realized_adam_cosine_20260816/R6J_REALIZED_ADAM_COSINE.json`
(+ per-tensor cosine payloads `per_tensor_cos_step*.npz`, 6 files, retained).
Tool: `experiments/ddm_r6j_realized_adam_step_cosine.py`. Cost: **0.7 s, $0**.

## The question

rg1b measured two facts that sit awkwardly together. The band objective **rotates the
gradient** (`cos` < 0.90 on the Adam-relevant metric in 9 of 9 cells), and yet the band arm's
flips land **on** the 5-arm displacement law `peak_flips = 118,563.2·‖Δw‖₁₀₀^0.457640`
(r² 0.9969) at **−0.871σ** — a *smaller* residual than the worst stock arm's 1.070σ.

Two readings survived that pair:

- **(J) JUDGE-DEAD** — the objective steers, but a 600-step peak-flip probe cannot see
  direction. This was rg1b's own adjudication.
- **(M) MECHANISM-DEAD** — AdamW's per-coordinate normalisation `m̂/(√v̂+ε)` **divides the
  reweighting out**. A per-pixel loss reweight that scales gradient *magnitudes* without
  changing their sign pattern is largely undone by dividing by `√v̂`, which carries the same
  scaling. Under (M) the objective cannot steer this optimiser at all, the flip law is a
  consequence rather than a coincidence, and repairing the judge buys nothing.

I entered this expecting **(M)**, and said so in the tool's docstring. The prior was not idle:
task #903 measured, on this stack, an upsample-VJP scatter interacting with Adam's sign
behaviour such that the **loss scalar was identical while 40 of 41 arrays diverged** — direct
evidence that Adam's normalisation can dominate direction here.

**The measurement refuted my hypothesis.** That is the result.

## What was measured

Both arms retain full AdamW state (`training_state.optimizer.state` → `exp_avg` = m,
`exp_avg_sq` = v) at matched steps, so the realized update direction is reconstructible from
the checkpoint **alone** — no re-forward, no scorer, no new run:

    u = m̂/(√v̂ + ε),   m̂ = m/(1−β₁ᵗ),   v̂ = v/(1−β₂ᵗ)

Hyperparameters verified matched and fail-closed at load: β = (0.9, 0.999), ε = 1e-8, and
lr **identical to 17 digits** (1.8673651497465946e-05) on both arms. 38 tensors, 66,339
coordinates.

| step | cos_m (momentum, cross-arm) | cos_u (**realized step**, cross-arm) | s − g |
|---:|---:|---:|---:|
| 100 | +0.300964 | +0.342021 | +0.041057 |
| 200 | +0.432363 | +0.356423 | −0.075940 |
| 300 | +0.325556 | +0.306114 | −0.019442 |
| 400 | −0.099590 | **−0.120834** | −0.021243 |
| 500 | +0.290544 | +0.303079 | +0.012536 |
| 600 | +0.540744 | +0.554099 | +0.013355 |

**cos_u never approaches the 0.95 bar. At step 400 it is NEGATIVE** — the two arms' realized
updates point into opposite half-spaces. (M) fails its own pre-registered test 6 times out of 6.

## Why the PAIRED difference is the sound statistic

I pre-registered the test as **one-sided** because A2 and band sit at different weight-space
points by step t (‖Δw‖₁₀₀ 0.047400 vs 0.055976, an 18% gap), and that confound can only push
a cross-arm cosine **down** — so a high cos_u would have been sound evidence for (M), while a
low cos_u alone would not have been sound evidence for (J).

The difference **s − g** escapes that limit. Both cosines are taken between the *same two
weight-space points*; the location confound enters `cos_m` and `cos_u` nearly identically and
largely cancels in their difference. Measured: **mean s − g = −0.0083, max |s − g| = 0.0759**.

Whatever rotation exists in the momentum **survives into the realized step essentially intact**.
Adam neither collapses the objective nor amplifies it.

**Per-arm control (no location confound at all):** `cos(m, u)` — how far Adam's own
normalisation rotates that arm's momentum — is **0.42–0.54 for both arms at every step**. Adam
applies a large (~60°) rotation, but it is *common-mode*: it rotates both arms by about the
same amount and does **not** make them agree. That is the precise sense in which the
normalisation is direction-preserving between arms while being far from direction-neutral
within one.

## What this licenses, and what it does not

**Licenses.** rg1b's adjudication stands on a second, independent instrument. The band
objective genuinely moves the weights somewhere else. **R6's blocker is the judge and the
window, not the optimiser** — so rg1b's named repair is the right route:
(a) compare arms at **matched ‖Δw‖** rather than matched steps; (b) a long window where
direction can compound; (c) a judge that is not peak/end flips.

R6 therefore keeps its position: **top of the sx1 ladder, the only rung whose ceiling
(−0.028604 S, 298% of the 0.0095973 gap) exceeds the gap, at zero bytes, needing only 33.55%
realized.**

**Does NOT license.** This measures that the two arms' steps **differ**. It does **not** measure
that the band arm's step is **better**. Direction difference is necessary, not sufficient. The
genuine remaining puzzle is unchanged and sharpened: a materially different direction produced
the *same flip count the displacement law predicts*. Resolving that is exactly what a
matched-‖Δw‖ or long-window judge is for — and until one runs, no descent claim is admissible.

**Also not established.** Six checkpoints, not an integrated trajectory. One α (=1). One init.
The `sign`-limit proxy rg1b measured and this exact-moment reconstruction agree in direction,
but neither is a multi-step compounding argument.

## Fire-order

1. **rg1b §6.6 item (a) — matched-‖Δw‖ comparison.** Now the highest-value R6 leg: the law
   predicts expected flips at any displacement, so the **residual off the curve is the
   direction signal** at σ_log 0.0728 resolution. Free to compute for any existing arm.
2. **rg1b §6.6 item (b) — the long window.** The untested question is whether direction
   compounds past the diffusive noise; rg1b's §6.5 names it as *the* live question.
3. The **F3/ns1 protection-list blocker** (rg1b §5) still binds the rg1 full burn and is
   unaffected by this result.

Sisters: [[ddm_rg1b_band_objective_build_20260816]] · [[ddm_sx1_seg_cure_ladder_20260816]] ·
[[cross-regime-constant-transfer-genus-finishing-stage]] (the genus this *avoided*: I did not
transfer #903's Adam-dominance finding as a conclusion — I tested it on this vehicle and it
did not hold).
