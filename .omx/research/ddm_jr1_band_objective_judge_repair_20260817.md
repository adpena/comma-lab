# ddm_jr1 — repair the judge on R6, the only seg rung whose ceiling exceeds the gap

`axis: [macOS-CPU advisory] read-back of retained payloads — NEVER a score`
`score_claim: false` · `verdict_scope: stated per finding`

Own-vehicle frontier, **unmoved by this unit**: hv1 ep0634 **S 0.15959729295498598 @ 182,759 B**
`[contest-CUDA T4 n600]`, archive sha256 `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`.
Gap to 0.15: **−0.0095973**.

---

## §0 PRE-REGISTRATION — written and committed BEFORE any measurement

Committed at the state recorded in this file's first commit. Everything below §0 was written after.

### §0.1 What I already established at source, before pre-registering

Reading receipts rather than memos changed the charter's premise. Three facts, each pinned:

1. **The displacement law was fit on FOUR arms, not five.**
   `.omx/research/ddm_rg1b_lr1_refit_and_bar_20260816.json` → `fit_peak_vs_dw100.n = 4`, `dof = 2`,
   `residuals_log` has exactly 4 entries, and the arm list is `C0, A1, A2, A3`.
2. **The band arm was NOT in that fit.** `/Volumes/APDataStore/pact/ddm_rg1/grad_cosine/RG1B_BAND_ARM_ON_THE_LAW.json`
   names its own key **`law_from_the_four_stock_arms`** and carries A/exponent/σ identical to the
   4-arm fit. The −0.871σ is therefore **genuinely out-of-sample as a point**.
   → **The charter's stated defect ("the band arm is the fifth point the law is fit on") is FALSE,
   and I do not inherit it.** It entered via loose language: `sx1` §5.3 calls it "the **5-arm**
   displacement law" and `rg1b` §6.6 says "the 5-arm curve … n now 5", both meaning *the curve has
   5 points on it now*, not *the fit used 5*.
3. **The five `ddm_lr1` directories are `C0, A1, A2, A3, W1` — and `W1` was EXCLUDED from the fit.**
   W1 is `float_warmup_steps=100` at `lr 2e-5` (A2's lr), stock objective, same seed/steps/curriculum
   fractions. **W1 is therefore a held-out STOCK arm: the null control the judge never had.**

### §0.2 The defect I am actually repairing (not the charter's)

The −0.871σ is out-of-sample as a *point* but its **denominator is wrong**, in two compounding ways:

- It divides by **σ_log = 0.072827**, the *in-sample residual RMS* of a 4-point, 2-parameter fit.
  The correct denominator for a NEW observation is the prediction standard error
  `SE_pred(d) = σ·√(1 + 1/n + (ln d − x̄)²/Sxx)`, which adds parameter uncertainty and the new
  point's own scatter. **`rg1b`'s own §3 bar section uses `SE_pred` correctly** (`SE_pred = 0.08257`
  at A2's d) — so the memo is internally inconsistent: `SE_pred` for the BREAK bar, bare `σ` for the
  σ-report. Using σ **overstates** the residual's apparent size.
- It reports the result in "σ" as if referred to a normal. With `dof = 2` the reference distribution
  is **t₂**, whose 95% two-sided critical value is **4.303**, not 1.96.

And there was no **null control**: W1 sat on disk unused.

### §0.3 PRE-REGISTERED BARS — stated before measuring

**LEG A.**
- `t_band = r_band / SE_pred(d_band)`, referred to **t₂**. Critical values pinned now:
  `t₉₅,₂ = 4.3027` (two-sided 95%), `t₉₀,₂ = 2.9200` (90%), `t₉₉,₂ = 9.9248` (99%).
- **SIGNIFICANCE BAR:** `|t_band| > 4.3027` → the residual is a direction signal at 95%.
  `> 2.9200` → SUGGESTIVE, promoted to nothing on its own.
- **NULL-CONTROL BAR (the decisive one):** compute `t_W1` by the identical construction. If
  `|t_band| ≤ |t_W1|`, then the band arm's residual is **not distinguishable from ordinary
  stock-arm prediction scatter**, and the residual-off-the-law judge is refuted as a direction test
  at this n regardless of the t-value.
- **LOO:** for each stock arm, refit on the other three (`dof = 1`, `t₉₅,₁ = 12.7062`), predict the
  held-out arm, studentize. Report the honest LOO scatter and place band + W1 on it. If LOO is not
  computable or is degenerate at this n, **say so with a number** rather than fitting a story.

**LEG B.** "Gradient rotated" (rg1b, measured) vs "**step** rotated" (open). Two measurements, both
from retained payloads, no new gradient:
- **B1 — realized trajectory rotation.** `cos(Δw_band(0→t), Δw_A2(0→t))` at `t ∈ {100…600}`.
  Both arms share the identical init, seed, data order, lr and curriculum; the ONLY difference is
  `--band-objective-weight`. This is the realized update, integrated — strictly stronger than a
  per-step cosine. **BAR:** rotated iff `cos < 0.95` (rg1b's own collinearity threshold).
- **B2 — realized AdamW next-step direction** reconstructed from the retained
  `training_state.optimizer.state[i].{exp_avg, exp_avg_sq, step}`:
  `u = m̂/(√v̂ + ε)`, `m̂ = m/(1−β₁^t)`, `v̂ = v/(1−β₂^t)`, `β=(0.9,0.999)`, `ε=1e-8` (read from the
  retained `param_groups`, not assumed). **BAR:** same 0.95.
- **B3 — the init identity.** Prove analytically and verify numerically that at `t=1`
  (`m=v=0` initially, both arms identical) the AdamW step reduces to the **sign limit**, so rg1b's
  `cos(sign g)` column **is** the realized first-step cosine and the gap it declared open at its
  §6.5 is closed at the init by identity.

**LEG C.** Design only. Derive the per-arm step budget to a common `‖Δw‖` target **from the measured
displacement-vs-step curve**, not from a guess. `b2e`'s warning binds: 3,000 steps at lr 2e-7 moved
ΔS_adv by +0.000336 and weight entropy by 9 B. **If the matched displacement target is unreachable at
a working lr inside a governed window, that is a Leg-C BLOCKER and I say so rather than paper over it.**

### §0.4 PRE-REGISTERED FORK

- **DIRECTION REAL** — `|t_band| > 4.3027` **AND** `|t_band| > |t_W1|` **AND** Leg B shows the
  realized step (not merely the gradient) rotated below 0.95. → R6 is a live supplier on the only
  rung whose ceiling exceeds the gap; Leg C's ticket is the campaign's next heavy fire and I say so.
- **DIRECTION NULL** — residual inside noise out-of-sample, or not separated from the W1 null. →
  the pixel-reweighting family closes at **FORMULATION** scope with a measured law, and I name where
  the seg axis routes: R5 (solved-prototype ordered camera paint, ancestor-scoped per [[m18]] — the
  *mechanism* transfers, the numbers do not) or the vehicle question.
- **UNDERPOWERED** — n too small to separate. → I name the exact arm count and the exact displacement
  spread that would separate them, and price it as Leg C's real cost. A refusal with a number is the
  deliverable; a fabricated significance is not.

### §0.5 Instrument control, pre-declared

The instrument must reproduce, from the retained checkpoints alone, all four stock arms'
`dw_100`, `dw_600`, `peak_dpx`, `end_dpx` to the digits in
`ddm_rg1b_lr1_refit_and_bar_20260816.json`, and the band arm's to
`RG1B_BAND_ARM_ON_THE_LAW.json`, **before** any new number is read. A control failure voids the
unit; I do not proceed past it.

---
