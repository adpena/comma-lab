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

## ANSWER FIRST

**The judge is not weak. It is structurally incapable, at any n — and I can put a number on it.**

The residual-off-the-law test that `rg1b` §6.6 item 2 proposed as "a usable free judge" has an
**irreducible resolution floor of a 15.3% flip reduction** (`1.96·σ_log = 0.1427` log). The band
arm's entire measured effect is **6.1%** (`0.0634` log). The floor is **2.25× coarser than the
signal**, and adding arms cannot close it: `SE_pred = σ·√(1 + 1/n + leverage)` — more arms shrink
`1/n` and the leverage term, but never the leading `1`, which is the new observation's own scatter.
**No number of arms makes this judge work.** That is the refusal-with-a-number the charter asked for,
and it retires a proposed free instrument before anyone spends on it.

**Corrected significance.** The honest studentized residual is **t = −0.763, p = 0.525** against
`t₂` — a coin flip — where `rg1b` reported "−0.871σ". And the band arm's `|t|` is **smaller than the
stock arms' own leave-one-out scatter** (max `|t| = 2.007` at A2). The band arm is *less* anomalous
than an ordinary stock arm's own prediction error.

**But the objective did work on the axis it was built for.** Leg B closes the gap `rg1b` §6.5 left
open: the **realized trajectory rotated 66.3°** — `cos(Δw_band, Δw_A2) = +0.524` at step 100 decaying
to **+0.402** and holding flat to step 600, against a 0.95 bar, on arms that differ in exactly one
CLI flag. Not the gradient — the *step*, integrated over the whole window.

**And the law-free comparison points the right way on both axes.** At step 600 the band arm sits at
**7,538 flips at ‖Δw‖ 0.064803** against A2's **8,049 at 0.050453** — **6.4% fewer flips at 28.5%
larger displacement**. Correct sign, twice. Not significant against this instrument, and no
instrument of this family can make it significant.

**Two further defects the judge carried, both found by reading receipts rather than memos:**

1. **The law is not universal, and a stock arm already on disk breaks it by 3×.** `W1` — stock
   objective, float warmup, excluded from the fit — lands at **t = +12.87, p = 0.006** (119,914
   measured vs 39,966 predicted). `rg1b` §6.4's upgrade ("in this regime flips are a function of
   displacement alone") must be narrowed to *the four QAT-from-step-0 arms*.
2. **A3 peaks at step 400; every other arm peaks at step 100.** The law regresses "peak, whenever it
   occurs" against "displacement at step 100" — and for one of its own four points those are
   different moments in the run.

**Verdict on the fork: DIRECTION NULL on the judge · UNDERPOWERED on the objective · and the repair
is not "more arms", it is "a different judge".** R6 stays open and unjudged. Leg C's ticket is
sealed below and it is now justified by a derived mechanism rather than by rg1b's intuition.

**Frontier UNMOVED.** No score claim anywhere in this unit.

---

## §1 Instrument control — 20 of 20, fail-closed

`experiments/ddm_jr1_judge_repair.py` recomputes every retained quantity from the checkpoints before
reading a single new number, and exits non-zero on any mismatch. It **did** exit non-zero twice
during the build, both times correctly:

- **Catch 1 — AppleDouble sidecars.** The SSD tiers are ExFAT, so macOS writes a `._<name>` sidecar
  beside every checkpoint. Those match `*.full_state.pt`. Reading one as a checkpoint would have
  silently corrupted the measurement; they are now excluded **by name**, not by `try/except`.
- **Catch 2 — the norm convention.** Full-float64 accumulation disagreed with every retained `dw` in
  the 9th significant figure. The retained convention sums squares **per tensor in float32**, then
  widens to float64 to add across tensors. Reproduced exactly rather than substituted; the float64
  value is reported alongside (`dw_by_step_float64`), never in place of.

Two conventions pinned that no memo states:

| convention | value | why it matters |
|---|---|---|
| weights used for `‖Δw‖` | **`training_state.model_state_dict`** (LIVE) | the top-level `state_dict` is the deployed/EMA copy and gives 0.0013798 where the receipt says 0.0014435 for C0@100 |
| flip metric | `max_t (qes(t) − init)·117,964,800` | reproduces C0 peak 5879.000000000001 / end 2164.0 exactly |

Receipts: `/Volumes/APDataStore/pact/ddm_jr1/JR1_JUDGE_REPAIR.json`
sha256 `639cbd3d2ff258c60d4e958c498ce0f68f0d226db0183bdc2ba203e9bd4d2f33` (29,485 B) ·
`JR1_VECTORS.npz` sha256 `c701f5290d53757b1ef449377c3b17e8052cd12926be42cfff0054b64606fb20`
(12,424,461 B — every displacement vector and every AdamW direction vector, 66,339 params × 6 arms ×
7 steps, so no successor re-reads 40 checkpoints to re-ask this).

---

## §2 LEG A — the judge, repaired and then retired

### §2.1 The charter's premise, corrected from the receipts

The charter warned me not to inherit `−0.871σ` because "the band arm is the fifth point the law is
fit on." **It is not.** The receipt's own key is `law_from_the_four_stock_arms`; the fit carries
`n = 4`, `dof = 2`, four residuals, arms `C0/A1/A2/A3`. The band arm was genuinely held out.

The `{C0,A1,A2,A3,W1}` / "fifth point" discrepancy resolves cleanly: the five directories are the
five **`ddm_lr1`** arms, and **`W1` is the one excluded from the fit**; "fifth point" in `rg1b` §6.3
means the band arm is the fifth point now *on the curve*, not the fifth point *in the fit*. `sx1`
§5.3 then compressed this to "the **5-arm** displacement law", and the charter inherited that.
**No in-sample defect. A different one.**

### §2.2 The defect that is real — the denominator

`rg1b` divided the residual by **`σ_log`**, the in-sample residual RMS of a 4-point 2-parameter fit,
and reported the result in "σ" as though referred to a normal. For a NEW observation the denominator
is the prediction standard error `SE_pred(d) = σ·√(1 + 1/n + (ln d − x̄)²/Sxx)` — **which `rg1b`'s own
§3 BREAK bar uses correctly** (`SE_pred = 0.08257` at A2's d). The memo used `SE_pred` for the bar and
bare `σ` for the σ-report. With `dof = 2` the reference is `t₂`, whose 95% two-sided value is
**4.3027**, not 1.96.

| quantity | `rg1b` as published | honest |
|---|---:|---:|
| log residual | −0.06342 | −0.06342 |
| denominator | σ_log = 0.072827 | **SE_pred = 0.083117** |
| reported | **−0.871 σ** | **t = −0.763** |
| reference distribution | (implied normal) | **t₂**, crit 4.3027 |
| p, two-sided | not stated | **0.525** |

Both errors push the same way: they make the residual look larger than it is. The corrected number
is a coin flip.

### §2.3 The null control the judge never had — leave-one-out

| held-out arm | log residual | SE_pred | t (dof 1) | p |
|---|---:|---:|---:|---:|
| C0 | −0.03582 | 0.18424 | −0.194 | 0.878 |
| A1 | +0.08468 | 0.08844 | +0.957 | 0.514 |
| **A2** | **−0.10906** | 0.05433 | **−2.007** | 0.294 |
| A3 | +0.10401 | 0.16580 | +0.627 | 0.643 |
| **band_a1** *(true hold-out)* | **−0.06342** | 0.08312 | **−0.763** | **0.525** |

**The band arm's `|t| = 0.763` is below the stock arms' own LOO scatter (max 2.007, mean 0.947), and
its raw log residual (0.0634) is below their RMS LOO residual (0.0883).** A stock arm predicted from
its three siblings misses by more than the band arm does. There is no signal here.

### §2.4 The structural power result — the judge cannot be fixed by more arms

| quantity | log | as a flip ratio |
|---|---:|---:|
| minimum detectable effect at n = 4 (95%) | 0.3553 | **1.427×** (a 30% reduction) |
| **asymptotic floor as n → ∞** | **0.1427** | **1.153×** (a 13.3% reduction) |
| **band arm's measured effect** | **0.0634** | **1.065×** (a 6.1% reduction) |

`SE_pred = σ·√(1 + 1/n + leverage)`. Arms shrink `1/n` and leverage; the leading `1` — the new
observation's own scatter — is irreducible. So the floor is `1.96·σ_log`, and it sits **2.25× above
the effect**. **The residual-off-the-law judge is refuted, not underpowered.**

And the usual escape does not exist here: `σ_log = 0.0728` is **not** stochastic noise. These arms
are seed-matched and deterministic in configuration; `σ_log` is the **misspecification of a single
power law across three decades of learning rate**. Replicate seeds would reproduce, not average it
down. What *does* remove it is **matching** — at equal displacement the misspecification is common to
both arms and cancels in the difference. That is precisely why Leg C's matched-‖Δw‖ design is the
correct repair, and it is a derived reason, not a preference.

### §2.5 Two more cracks in the law

**W1 breaks it by 3×.** W1 is stock-objective, `float_warmup_steps=100`, `lr 2e-5`, same seed and
curriculum fractions — and it was left out of the fit.

| arm | ‖Δw‖₁₀₀ | measured peak | law's prediction | log residual | t | p |
|---|---:|---:|---:|---:|---:|---:|
| **W1** | 0.092907 | **119,914** | 39,966 | **+1.0988** | **+12.87** | **0.006** |

⚠ **W1 is not a matched null for the band arm, and I will not use it as one.** Its step-100 point is
the float→quantized shock boundary: 100 steps of *float* CE, then evaluated *quantized*. That is a
mechanically different event, and it is the honest explanation for the 3×. What W1 **does** establish
is that the law is **curriculum-conditional**. `rg1b` §6.4's upgrade — "in this regime flips are a
function of displacement alone" — holds for *the four QAT-from-step-0 arms across three decades of
lr*, and a fifth stock arm sitting on the same disk violates it by a factor of three. That
qualification should travel with the law wherever it is cited.

**A3 peaks at the wrong step.** `peak_step` = **400** for A3, **100** for C0/A1/A2/W1/band. The law
pairs "peak, wherever it occurs" with "displacement at step 100"; for one of its own four fit points
those are different moments in the run.

### §2.6 The END law — fit by `rg1b`, never applied to the band arm

`rg1b` §3 fit `end ∝ ‖Δw‖₆₀₀^0.5573` (R² 0.9460) and then scored the band arm only on the PEAK law.
Scoring it on the END law costs nothing and I owed it:

| arm | ‖Δw‖₆₀₀ | measured end | predicted | log residual | t | p |
|---|---:|---:|---:|---:|---:|---:|
| band_a1 | 0.064803 | **7,538** | 13,738 | **−0.6002** | **−1.278** | 0.329 |
| W1 | 0.085273 | 12,422 | 16,008 | −0.2536 | −0.534 | 0.647 |

Same direction, still not significant (`σ_log = 0.4116` here — 5.7× looser than the peak law). Worth
recording because it is the third independent read that points the same way and the third that
cannot resolve it.

---

## §3 LEG B — the STEP rotated, not merely the gradient

`rg1b` §6.5 listed "**No realized-AdamW-step cosine**" as an open gap. Closed, three ways, all from
retained payloads and all at full precision.

### §3.1 B1 — the realized trajectory rotation (the decisive one)

A2 and band_a1 differ in **exactly one CLI flag**: `--band-objective-weight`. Same init, seed, data
order, lr, steps, curriculum, `--weight-qat-q3q4`, device. So `cos(Δw_band(0→t), Δw_A2(0→t))` is the
realized update direction, **integrated over the window** — no counterfactual, no limit proxy, no
choice of whose moments.

| step | cos(Δw_band, Δw_A2) | angle | below 0.95? |
|---:|---:|---:|:--:|
| 100 | **+0.5244** | 58.4° | yes |
| 200 | +0.4710 | 61.9° | yes |
| 300 | +0.4113 | 65.7° | yes |
| 400 | +0.4011 | 66.4° | yes |
| 500 | +0.4012 | 66.3° | yes |
| 600 | **+0.4019** | **66.3°** | yes |

**The trajectories separate immediately and stay separated.** The rotation is not a transient at the
start that decays back — it *grows* from 58° to 66° and then holds flat for the last 300 steps. The
two runs are exploring materially different directions for the whole window, and the flip trajectory
does not care. That is the strongest possible form of `rg1b`'s finding, and it makes the
instrumental verdict airtight: **the judge could not see a 66° rotation.**

### §3.2 B2 — the realized AdamW next-step direction, reconstructed from retained moments

`u = m̂/(√v̂ + ε)`, `m̂ = m/(1−β₁^t)`, `v̂ = v/(1−β₂^t)`, with `β = (0.9, 0.999)` and `ε = 1e-8` **read
from the retained `param_groups`**, not assumed.

| step | cos(u_band, u_A2) | sign agreement |
|---:|---:|---:|
| 100 | +0.3420 | 0.634 |
| 200 | +0.3564 | 0.628 |
| 300 | +0.3061 | 0.607 |
| 400 | **−0.1208** | 0.471 |
| 500 | +0.3031 | 0.608 |
| 600 | +0.5541 | 0.702 |

Every cell far below 0.95; one cell is **anti-aligned**. ⚠ Caveat stated plainly: by step *t* the two
arms sit at **different weights**, so this measures "do the two runs step alike", not a same-point
counterfactual. B1 is the load-bearing measurement; B2 corroborates.

### §3.3 B3 — the init identity closes the gap analytically

After one AdamW step from zero moments, `m = (1−β₁)g` and `v = (1−β₂)g²`, so `m̂ = g`, `v̂ = g²`, and

> `u = g/(|g| + ε) = sign(g) · |g|/(|g| + ε)`

**Therefore `rg1b`'s `cos(sign g)` column IS the realized first-step cosine** — 0.2087 / 0.5235 /
0.6185 across the three phases — not a proxy for it. The gap §6.5 declared open is closed at the init
**by identity**, and only the `t > 1` behaviour ever needed measuring (which B1 and B2 now supply).

Precondition `|g| ≫ ε`, checked on the retained moments: **31,568 of 919,146 live coordinates
(3.43%)** sit within 100× ε; the smallest live `√v̂` is 2.09e-10. So the identity is exact for 96.6%
of live coordinates and ε-softened on 3.4%. An ε-distortion on 3.4% of coordinates cannot turn a
cosine of 0.209 into 0.95. *(Coordinates with `v == 0` exactly never received gradient and take a
zero step under either objective; they are excluded, not counted as violations.)*

---

## §4 The law-free comparison — what the data says with no fit at all

No law, no σ, no fit. Just the two matched arms side by side.

| step | A2 flips | A2 ‖Δw‖ | band flips | band ‖Δw‖ | band/A2 flips |
|---:|---:|---:|---:|---:|---:|
| 100 | 27,170 | 0.047400 | 29,747 | 0.055976 | 1.0948 |
| 200 | 14,237 | 0.049407 | 15,456 | 0.062182 | 1.0856 |
| 300 | 12,009 | 0.050815 | 10,214 | 0.062915 | **0.8505** |
| 400 | 9,607 | 0.050836 | 9,021 | 0.064109 | 0.9390 |
| 500 | 8,415 | 0.050572 | 7,981 | 0.064733 | 0.9484 |
| **600** | **8,049** | 0.050453 | **7,538** | 0.064803 | **0.9365** |

The band arm starts worse (it moves further), crosses over at step 300, and finishes with **6.4%
fewer flips while sitting 28.5% further from the init**. Both coordinates favour the objective. And
**every single number here is inside the instrument's noise** — which is the entire point of this
unit. The band arm is **unjudged**, not refuted.

⚠ **The bar this does not clear, and it matters.** `improved_over_init` is **false for all six arms**,
band included: every arm's minimum `quantized_exact_seg` over the window is the init, at step 0.
Nothing in this family has ever descended, at any lr across three decades. So the honest reading of
the table above is *"the band arm climbs less"*, not *"the band arm supplies flips."* Those are
different claims and only the first is in evidence.

---

## §5 LEG C — the sealed matched-‖Δw‖ ticket (MAIN fires; this arm does not)

### §5.1 What the ticket buys, stated before the command

It buys a **direction verdict at matched displacement**, where the law's misspecification cancels in
the difference (§2.4) instead of being carried as irreducible scatter. It is the correct **gate**
before spending on the long window `rg1b` §6.5 named — roughly 1/10 the cost of a 3,000-step run.

It does **not** establish that R6 supplies flips (see the §4 blocker). It answers exactly one
question: **at equal weight displacement, does the band objective damage less than stock?**

### §5.2 The derivation of the step budget, from the measured curve

Target: A2's `‖Δw‖₁₀₀ = 0.047399682085730054`. The band arm at `lr 2e-5` reached **0.055976**, i.e.
**1.181× too far** — so the ticket lowers lr rather than steps (the curriculum fractions are defined
against total steps, so changing steps would change the schedule and break the match).

Fitting `‖Δw‖₁₀₀ ∝ lr^p` on the four stock arms gives **p = 0.7907** (`rg1b` reports 0.7893 fitting
the other direction; the difference is regression asymmetry, not disagreement), so

> `lr_matched = 2e-5 × (0.047400/0.055976)^(1/0.7907) = ` **`1.6207e-05`**

⚠ **This is an extrapolation from ONE band point and it must not be trusted as a law.** The band
arm's own `dw`-vs-`lr` exponent is **unmeasured** — the band gradient is 4.6–100× larger in norm than
stock (`rg1b` §6.2), and although Adam divides that out, the *curvature* it meets need not match. So
the ticket **brackets** instead of trusting the point estimate.

### §5.3 The four runs

Three band arms bracketing the target, plus **one determinism control that the comparison cannot do
without**:

| # | arm | lr | purpose |
|---|---|---:|---|
| 1 | `band_m_lo` | 1.50e-5 | bracket below |
| 2 | `band_m_mid` | 1.62e-5 | point estimate |
| 3 | `band_m_hi` | 1.75e-5 | bracket above |
| 4 | **`A2_repeat`** | 2.0e-5 | **byte-identical re-run of A2 — establishes the run-to-run floor** |

Run 4 is not optional. MPS is not bit-deterministic, and without `|Δpeak|` from an identical re-run
there is **no bar** to compare the matched difference against — the unit would repeat exactly the
error it was chartered to fix. **If `A2_repeat` reproduces A2's 27,170 to within a few flips, the
matched comparison resolves effects far below the 13.3% floor of §2.4 — that is the whole reason this
design beats the residual judge.** If it does not reproduce, the run-to-run spread *is* the new floor
and must be reported as such before any band arm is read.

Then interpolate the three band `(‖Δw‖₁₀₀, peak)` points to `‖Δw‖₁₀₀ = 0.047400` and compare against
A2's 27,170 **directly** — no law, no σ_log, no `SE_pred`.

**Pre-registered bar:** the band objective damages less iff the interpolated band peak at matched
displacement is below `27,170 − 3·(A2_repeat run-to-run spread)`. Fixing the bar to a *measured*
noise floor rather than to a fitted scatter is the repair.

### §5.4 The command (MAIN fires; re-hash every pin at fire time)

```bash
# Run 4 of 4 shown; runs 1-3 are identical with --lr 1.50e-5 / 1.62e-5 / 1.75e-5,
# --band-objective-weight 1.0, and --label/--out/--save renamed to match.
.venv/bin/python tools/safe_run.py \
  --rss-mb 12288 --timeout 4200 \
  --label ddm_jr1_A2_repeat \
  --status-receipt /Volumes/APDataStore/pact/ddm_jr1/A2_repeat/safe_run_status.json \
  --child-pidfile  /Volumes/APDataStore/pact/ddm_jr1/A2_repeat/child.pid \
  -- \
  .venv/bin/python -m tac.pr130_lift.train_semantic_quantized_resumable \
  --challenge-root upstream \
  --cache /Volumes/VertigoDataTier/pact/ddm_op1r_20260809/authority_cache/gt_cache_600_official_ada.pt \
  --init  /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/\
checkpoints/semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt \
  --bits 4 \
  --weight-qat-q3q4 \
  --steps 600 --lr 2.0e-5 --float-warmup-steps 0 \
  --ce-fraction 0.50 --softplus-fraction 0.85 \
  --eval-every 100 --checkpoint-every 100 \
  --device mps --seed 20260715 \
  --band-objective-weight 0.0 \
  --out  /Volumes/APDataStore/pact/ddm_jr1/A2_repeat/result.json \
  --save /Volumes/APDataStore/pact/ddm_jr1/A2_repeat/ckpt
```

**Budget** ~33 min each, **4 runs ≈ 2.2 h**, one governed Metal fire at a time (never concurrent —
the composed-preflight OOM law binds). **Resumability** unchanged: 6 stage checkpoints, resumable
from disk, `best_step`/`init_seg` round-trip through resume. **Memory preflight at the real config**
before launch. **Payload** `/Volumes/APDataStore/pact/ddm_jr1/<arm>/` — every checkpoint, full
`history`, safe_run receipt, launcher tree.

**Read the receipt with `experiments/ddm_jr1_judge_repair.py`**, which already reproduces every
retained quantity fail-closed; add the four new arms to `ARMS` and the control extends automatically.

### §5.5 The b2e blocker, checked rather than assumed

`b2e` warns that 3,000 steps at `lr 2e-7` moved ΔS_adv by +0.000336 and weight entropy by 9 B — "did
this trainer train at all?" **Checked: `lr 2e-7` is C0, which moves `‖Δw‖₁₀₀ = 0.0014` and then
plateaus (`‖Δw‖₆₀₀ = 0.0014`). The matched target 0.0474 is 33× further, and it is reached in 100
steps at `lr 1.6e-5` — 81× above b2e's lr.** So the displacement target is **reachable** and b2e's
non-response does not block this ticket; b2e was measuring the flat end of the lr range.

**The real Leg-C blocker is a different one, and I name it rather than paper over it:** no arm in this
family has ever satisfied `improved_over_init`, at any lr across three decades, including the band
arm. A matched-displacement comparison can rank *damage rates*. It **cannot** show that R6 supplies
the 33.55% of the round trip the gap needs. That requires a window in which something descends, which
is `rg1b` §6.5's untested long-window question and is **not** what this ticket buys.

---

## §6 What this unit did NOT establish

- **No score.** Frontier UNMOVED: hv1 ep0634 S 0.15959729295498598 @ 182,759 B `[contest-CUDA T4 n600]`.
- **No verdict on the band objective.** It is **unjudged**, and §4's 6.4%-fewer-flips-at-28%-further
  is *not* evidence at this instrument's resolution. I decline to promote it.
- **No proof that no direction descends.** Still two directions, not a spanning set.
- **No long-window result.** Whether direction compounds past diffusive noise over 3,000+ steps is
  untested and remains the live question behind Leg C.
- **W1 is not a matched null.** Its 3× miss is explained by the float→quantized shock; I use it to
  qualify the law's universality and for nothing else.
- **The matched-lr point estimate is an extrapolation** from one band point; §5.3 brackets rather
  than trusting it.

## §7 Verdict scope

`verdict_scope: FORMULATION` on **the residual-off-the-law judge** — refuted as a direction test for
effects below ~15% flip reduction, at any n, by the `SE_pred` floor argument in §2.4. This is a
property of the estimator, not of one dataset.

`verdict_scope: INSTANCE` on **the law's curriculum-conditionality** (§2.5) — one held-out arm, one
warmup configuration.

`verdict_scope: none` on **the band objective itself** — it is unjudged, and R6 remains the only seg
rung whose ceiling (−0.028604 S, 298% of the gap, at zero bytes) exceeds the gap.
