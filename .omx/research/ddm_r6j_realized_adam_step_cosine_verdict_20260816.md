# R6 judge repair, leg 3 — the realized-AdamW step cosine

> # ⛔ VERDICT WITHDRAWN 2026-08-16 — THE NULL CONTROL REFUTES IT
>
> **Round-1 recursive adversarial review (fresh-eyes reviewer, finding 1 CRITICAL) named the
> missing control; MAIN ran it in 0.7 s at $0. It was free the whole time and this memo
> shipped without it.**
>
> `ddm_lr1/{A1,A2,A3,C0}` each retain all seven `full_state` checkpoints — the same six steps
> this memo compares. Running the SAME tool on two arms that SHARE an objective:
>
> | pair | objective | `cos_realized_step_cross_arm`, steps 100…600 |
> |---|---|---|
> | A1 vs A2 | SAME | +0.3716 +0.6449 +0.4206 +0.0410 +0.4279 +0.6767 |
> | A1 vs A3 | SAME | +0.1991 +0.0755 +0.0012 −0.0291 +0.1136 +0.0997 |
> | A1 vs C0 | SAME | +0.4969 +0.7842 +0.6128 +0.5133 +0.6027 +0.6449 |
> | A2 vs band | **DIFFERENT** | +0.3420 +0.3564 +0.3061 **−0.1208** +0.3031 +0.5541 |
>
> **NULL pool** (n=18, same objective): mean **+0.3721**, range [−0.0291, +0.7842].
> **BAND** (n=6, different objective): mean **+0.2902**, range [−0.1208, +0.5541].
> Twelve of eighteen null values fall inside the band's own range, and **A1-vs-A3 — two arms
> that share an objective — are LESS aligned than band-vs-stock.**
>
> **What this kills.** The pre-registered falsifier was `cos_u ≥ 0.95 ⇒ MECHANISM-DEAD`. With
> no null, "cos_u never approaches 0.95" was read as refuting MECHANISM-DEAD. But arms that
> share an objective *also* never approach 0.95 — the bar was unreachable for any pair, so the
> test could never have fired. It discriminated nothing. **Both readings survive the data:**
> under MECHANISM-DEAD the band arm should look like an ordinary stock pair, and it does
> (0.2902 vs null 0.3721, well inside). The honest verdict is
> `INDETERMINATE_NO_DISCRIMINATING_POWER`, not `JUDGE_DEAD`.
>
> **What still stands.** The measurement itself (moments read back from two byte-verified
> checkpoints, 66,339 coords, 38 tensors) is sound and reproduces. The paired `s − g` statistic
> and the per-arm `cos(m,u)` 0.42–0.54 control are unaffected. The *conclusion drawn from them*
> — "the band objective genuinely moves the weights somewhere else" — is withdrawn: nothing
> here separates objective-steering from ordinary lr-driven trajectory divergence.
>
> **Consequence for the fire-order.** The addendum's re-ranking (long window promoted over
> matched-‖Δw‖) rested on the licensing paragraph and is also withdrawn. A discriminating
> instrument must be established BEFORE either judge is ranked: any future arm comparison must
> report its same-objective null alongside the treatment. Genus:
> `[[m94]]` (a negative measures the instrument) — here the instrument measured, and its
> capacity was zero.
>
> Null receipts: `/Volumes/APDataStore/pact/ddm_r6j_nullcontrol_20260816/{A1_vs_A2,A1_vs_A3,A1_vs_C0}/`.

**verdict (SUPERSEDED, retained per APPEND-ONLY):** `JUDGE_DEAD_SURVIVES_STEP_IS_GENUINELY_ROTATED` (6 of 6 matched steps)

verdict_scope: instance — two arms (stock `ddm_lr1/A2`, band `ddm_rg1/band_a1` at α=1),
6 checkpoints over 600 steps, this init, MPS-trained state read back on CPU.

**The words MECHANISM-DEAD and JUDGE-DEAD in this memo are HYPOTHESIS NAMES under test,
never family verdicts.** Nothing here kills any optimizer family or any objective family.
What died is one specific *explanation* — that AdamW's per-coordinate normalisation divides
the band reweight out — refuted on two arms of one vehicle. The band objective itself is
neither confirmed nor killed by this measurement (see "Does NOT license" below).

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

## ADDENDUM — the free judge is UNDER-POWERED, and that re-orders the fire-order

rg1b §6.6 item 2 proposes the residual-off-the-law as "a usable free judge … it is
~0.07σ-resolvable." I computed the detection arithmetic before spending a Metal slot on it.
**σ_log 0.0728 is the fit's residual SCATTER, not the resolution of a single new point** — a
new arm's residual must BEAT that scatter to be distinguishable, and the band arm's sits
*inside* one sigma.

| what | improvement over the law needed |
|---|---:|
| **band arm DELIVERED** | **6.14%** (residual −0.871σ, peak/prediction 0.9386) |
| single new arm to clear 1σ | 7.02% |
| single new arm to clear 2σ | **13.55%** |
| law-vs-law, 3 band arms vs the 4 stock | 10.53% |
| law-vs-law, 4 band arms | 9.79% |
| law-vs-law, 8 band arms | 8.53% |

(law-vs-law: SE(Δ log A) ≈ σ√(1/n₁+1/n₂), leverage ignored — an optimistic bound, so the real
bars are *higher* than shown.)

**The measured effect is ~0.45× the single-arm detection threshold, and adding arms does not
rescue it**: even 8 band arms leave a 8.53% bar against a 6.14% effect. The residual judge
cannot resolve this effect size **at any affordable n**. rg1b's own receipt anticipated the
shape of this — `instrument_capacity_caveat`: *"n=4, dof=2 … this bar detects a LARGE direction
change only. A 20% improvement is inside the noise."* — but item 2 then proposed the same
statistic as the cheap route. It is cheap; it is also blind here.

**Consequence:** item (a) drops below item (b). The discriminator must be the **LONG WINDOW**,
where a per-step direction advantage compounds instead of being read off a single noisy
displacement point.

## Fire-order (re-ordered by the addendum)

1. **rg1b §6.6 item (b) — the long window.** Promoted to rank 1. The untested question is
   whether direction compounds past the diffusive noise; rg1b §6.5 names it as *the* live
   question, and the addendum shows it is now the only judge with the power to answer.
2. **rg1b §6.6 item (a) — matched-‖Δw‖ comparison.** Demoted, NOT dead: matching displacement
   removes the location confound, which is a different (and real) gain from resolving the
   residual. Worth doing *inside* the long window, not as a standalone cheap substitute for it.
3. The **F3/ns1 protection-list blocker** (rg1b §5) still binds the rg1 full burn and is
   unaffected by this result.

Sisters: [[ddm_rg1b_band_objective_build_20260816]] · [[ddm_sx1_seg_cure_ladder_20260816]] ·
[[cross-regime-constant-transfer-genus-finishing-stage]] (the genus this *avoided*: I did not
transfer #903's Adam-dominance finding as a conclusion — I tested it on this vehicle and it
did not hold).
