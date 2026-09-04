---
title: "sd1's 85% wasted gradient splits 77.7 / 7.7 at the n600 m_safe, and the correct-pixel share has a FLOOR at 52% that no tau can cross — the pre-registered >=50% prediction is NOT met (falsifier FIRES on the all-correct reading at 19.9%, does not fire on the wasted-only reading at 45.6%); the tau band beats row 1 on waste removal 45.6-96.8% vs 3.7-37.3%, they compose near-multiplicatively, and row 1's SHIPPED default is near-inert while its headline bottom-k@0.05 setting is EXACTLY inert"
arm: ddm_gm1
charter: .omx/research/charters/ddm_gm1_gradient_mass_at_n600_msafe_20260904.md
charter_commit: 57a6c3f36
utc: 2026-09-04T02:00:20Z
verdict_scope: "[macOS-CPU advisory . retained EMA-shadow scorer logits . frozen CPU-torch SegNet argmax . QBF1-born vehicle . n32 sealed selection . seed 20260902 . NON-PROMOTABLE]"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_gm1 — where the seg gradient lands, split at the n600 `m_safe`

## The pre-registered prediction and falsifier, READ OUT BEFORE THE NUMBERS

The charter (`.omx/research/charters/ddm_gm1_gradient_mass_at_n600_msafe_20260904.md`, §Measure 4)
pre-registered, verbatim:

> *prediction (from sd1): a τ band starting at 2·δ_R removes ≥50% of the correct-pixel gradient
> share; falsifier: < 30%.*

"Correct-pixel gradient share" admits two readings and the pre-registration did not disambiguate
them, so both are reported and neither is chosen after the fact:

| reading | share at τ = 0.15 | share at τ = 2·δ_R | relative removal | verdict |
|---|---:|---:|---:|---|
| **A — ALL correct pixels** (sd1's quantity) | 85.385% | 68.423% | **19.87%** | **FALSIFIER FIRES** (< 30%) |
| **B — WASTED only** (correct AND outside `m_safe`) | 77.715% | 42.278% | **45.59%** | falsifier does not fire; **prediction still NOT met** (< 50%) |

**Under both readings the ≥50% prediction FAILS.** Under the reading sd1's own headline uses
(A) the falsifier fires outright. Control cell, step 0, DALI authority, HT-weighted; the treatment
cell is identical at step 0 (shared init) and within 1.0 pp everywhere.

## The finding, first

**sd1's "85% of the gradient is spent on already-correct pixels" is two populations with opposite
meanings, and at the trainer's τ they split 77.715 / 7.670.** Only the first is waste: those pixels
sit outside `m_safe`, so R cannot un-do them and the gradient buys nothing the score can see. The
second defends correct pixels the uint8 roundtrip can still flip — legitimate spend.

**And the correct-pixel share has a FLOOR.** Scanning τ from 0.200 down to 0.004 (9.14 → 0.18 δ_R),
the all-correct share falls monotonically to **52.159%** and stops: the waste is driven to 0.002%
but the *defence* group rises to 52.156%. **The charter asked for the τ at which the correct share
first drops below 50% and below 25%. MEASURED answer: NEITHER CROSSING EXISTS on this vehicle.**
The floor is the near-boundary correct/wrong ratio of the field itself — a property of the FIELD, not
of τ, and no temperature can move it.

The *wasted* share does cross, and cheaply: below 50% at **τ ≈ 2.49–2.99 δ_R** and below 25% at
**τ ≈ 1.25–1.40 δ_R**, across all 6 milestones of both cells.

Three further MEASURED results that rank the third race:

* **The τ band dominates row 1 on waste removal.** τ alone removes 45.6% (at 2 δ_R) → 77.7% (at
  1 δ_R) → 96.8% (at 0.5 δ_R) of the τ=0.15 waste. Row 1's best tested setting removes **37.3%**,
  and its **SHIPPED default `inverse@temp=1.0` removes 3.7%**.
* **Row 1's headline setting `bottom-k@0.05` is EXACTLY inert — 0.0% at every milestone and every
  τ.** The measured concentration that motivated row 1 (FEED-bp: 89.2% of d_seg in the
  bottom-5%-margin pixels) is the same fact that makes it inert: the mask's selected set already
  contains the gradient support, so a mean-1 mask is a uniform rescale on that support and every
  share is invariant.
* **They are not substitutes — they compose near-multiplicatively on the waste** (9 combinations,
  max deviation from the multiplicative prediction **2.5 pp**), so a race may fire both.

## Verified at source (VERIFIED-AT-SOURCE LAW — `path:line` for every premise this arm adds)

| claim | evidence | label |
|---|---|---|
| the loss is `sigmoid(-margin/τ)`, pixel-mean, then HT-weighted over pairs | `experiments/ddm_qbt1_qbflow_trainer.py:544-565` | MEASURED |
| `margin = logits[GT] − max_{c≠GT} logits[c]`, target channel masked with −1e9 | `ddm_qbt1_qbflow_trainer.py:556-559` | MEASURED |
| τ is one global scalar, linear `start=0.15 → end=0.05` over `total_steps`, and `tau_for_step` takes those as DEFAULT ARGUMENTS — the band is two constants in one signature | `ddm_qbt1_qbflow_trainer.py:643-646` | MEASURED |
| HT weights are `(15.0,)*24 + (30.0,)*8` over `SELECTION_IDS` (n32) | `ddm_qbt1_qbflow_trainer.py:78,112` | MEASURED |
| row 1 `_live_margin_weight` allocates over the **UNSIGNED top1−top2 gap** (`sort(...)[-1] − sort(...)[-2]`), mean-1, `stop_gradient`; three allocators `inverse` / `exp` / `bottom-k` | `experiments/train_witness_realized_through_R_mlx.py:1086,1107-1108,1110-1120` | MEASURED |
| row 1's SHIPPED defaults are `--margin-weight-fn inverse` and `--margin-weight-temp 1.0`, with a documented anneal from `--margin-weight-temp-start 1.0` toward a "moderate TARGET, e.g. 0.3" | `train_witness_realized_through_R_mlx.py:3293,3299,3314-3316` | MEASURED |
| `m_safe` **resolved through the canonical law**, not carried as a literal: `delta_r 0.021881818771362305`, `headroom 2.0`, `m_safe 0.04376363754272461`, `n_frames 600`, `artifact reports/delta_R_noise_floor_n600.json`, `artifact_fallback_used False` | `tac.canonical_equations.margin_band_satisficing_threshold_20260712.resolve_margin_band_threshold`, live call | MEASURED |
| per-class `δ_R_c` read from dr1's receipts artifact (never retyped) | `/Volumes/APDataStore/pact/ddm_dr1_delta_R_n600/delta_R_receipts_n600.json` → `per_class_annulus_pooled[*].p95` | MEASURED (dr1) |
| the milestone retains `segnet_logits_f16 (5,384,512)` + `segnet_argmax_u8` + `target_argmax_u8` per pair, under `ema_scope` | `ddm_qbt1_qbflow_trainer.py:1907-1936`; `ddm_qbr1_born_fairform_burn_prep.py:420-439` | MEASURED (sd1, re-read here) |
| sd1's bins are **UNWEIGHTED** across pairs — no HT multiplier at the accumulator | `experiments/ddm_sd1_surrogate_exact_map.py:400-405` | MEASURED |
| vr1 row 1 identification + FEED-bp's bottom-5% concentration | `.omx/research/ddm_vr1_v7_v11_signal_recall_20260903.md:137` | TRANSFERRED |
| Lane = 0.59% of area, 33.56% of model bits, ~90.1% of rate demand | `[[m131]]` | TRANSFERRED |

## Calibration receipts (the instrument, before any finding)

1. **Exact, and bit-for-bit.** The HT-weighted `d_seg` recomputed from the retained argmax reproduces
   each milestone's own recorded `d_seg_hat` with difference **0.000000000e+00**, and the per-pair
   `|recorded − recomputed|` maximum is **0.000e+00**, at all **6 milestones × 2 cells**.
2. **Differential against the objective itself.** `test_surrogate_matches_the_trainers_own_loss_exactly`
   calls `qbt.expected_flip_margin_loss` and asserts this instrument's surrogate matches it to
   `rel=1e-6` at τ ∈ {0.15, 2 δ_R, 0.5 δ_R}. The instrument cannot drift from the loss it decomposes.
3. **Cross-instrument reconciliation with sd1, and the resolution of a real 0.28 pp gap.** My first
   read showed 85.385% correct-share where sd1's memo reports 85.10%. Traced, not waved at: sd1
   accumulates **unweighted** (`:400-405`), this arm accumulates both. My **unweighted** value is
   **0.850981 = sd1's 85.10% to four significant figures**. The two instruments agree exactly; the
   difference is HT weighting, which the trainer's own loss applies (`:559-565`) and sd1 omits. HT is
   this memo's primary mode; unweighted is reported wherever sd1 is compared.
4. **A numerical defect found and fixed in this arm's own gradient kernel, with its direction
   TRACED rather than assumed.** `|d sigmoid(-m/τ)/dm|` computed as `p*(1-p)` from a branch-split
   sigmoid loses the confidently-**wrong** tail (`m ≪ 0`, where `p → 1`) to cancellation against one
   ulp of 1.0: MEASURED relative error **3.6e-08 at z=20, 4.2e-06 at z=25, 1.0e-03 at z=30**. My first
   write-up of this had the sign backwards (I assumed it was the correct tail); the trace corrected
   it. Cured by `a/(1+a)²` with `a = exp(-|z|)`, exact on both tails and exactly even by construction.
   **Aggregate significance below 1e-12 of the total mass** — hygiene on an exact group share, not a
   finding, and labelled as such.
5. **Both GT lineages computed, never mixed.** DALI (authority) and PyAV (what the loss actually saw)
   are separate columns throughout; DALI flips exceed PyAV by **+0.97% to +1.59%** across the 12 milestone-cells.
6. **The source changed after the measurement ran, and the output is byte-identical.** Three edits
   landed after the cells were measured (a dead unreachable branch removed, the `--reanalyse` path
   added, a fail-closed guard on >2 cells). All three are outside the measurement path, and the
   claim is a RECEIPT rather than an argument: re-deriving `GM1_COMBINED.json` from the stored bins
   under the final source reproduces sha256 `dc138b60a092b122…` — the same 16 hex digits as the
   pre-edit derivation. The bins themselves were never re-read from the run directory.
7. **Cross-cell agreement, stated as a number.** Control vs treatment of the same seed differ by at
   most **0.68 pp (correct_outside) / 1.00 pp (correct_inside) / 1.02 pp (wrong)** over all
   6 milestones × 41 τ. That is a same-seed repeat, never a seed replication.

## 1. The three-way split, per milestone (MEASURED)

Control cell, DALI authority, HT-weighted. Cells are **outside% / inside% / wrong%**.

| step | τ_own | τ=0.15 (6.86 δ_R) | τ=0.05 (2.29 δ_R) | τ=2 δ_R | τ=1 δ_R | τ=0.5 δ_R |
|---:|---:|---|---|---|---|---|
| 0 | 0.15000 | 77.71 / 7.67 / 14.62 | 47.07 / 23.18 / 29.75 | 42.28 / 26.14 / 31.58 | 17.33 / 43.18 / 39.49 | 2.49 / 53.08 / 44.44 |
| 1,000 | 0.13000 | 74.08 / 7.50 / 18.42 | 42.99 / 22.41 / 34.60 | 38.48 / 25.21 / 36.31 | 15.64 / 41.28 / 43.08 | 2.24 / 50.95 / 46.82 |
| 2,000 | 0.10999 | 72.99 / 7.40 / 19.61 | 42.01 / 22.18 / 35.82 | 37.59 / 24.96 / 37.45 | 15.16 / 41.10 / 43.74 | 2.13 / 51.00 / 46.87 |
| 3,000 | 0.08999 | 73.07 / 7.50 / 19.43 | 41.99 / 22.41 / 35.60 | 37.59 / 25.20 / 37.21 | 15.27 / 41.27 / 43.47 | 2.16 / 50.87 / 46.97 |
| 4,000 | 0.06998 | 74.41 / 7.51 / 18.08 | 43.53 / 22.49 / 33.97 | 39.04 / 25.32 / 35.64 | 15.98 / 41.75 / 42.27 | 2.26 / 51.62 / 46.12 |
| 5,000 | 0.05000 | 75.47 / 7.62 / 16.91 | 44.44 / 22.99 / 32.57 | 39.82 / 25.90 / 34.28 | 16.16 / 42.68 / 41.16 | 2.30 / 52.61 / 45.09 |

The split is **remarkably stable in training** — every column moves by less than 5 pp across
5,000 updates while τ moves it by 75 pp. **τ, not training progress, sets where the gradient lands.**

**The population underneath (τ-independent, MEASURED):** 99.657% of pixels are correct-and-outside,
**0.088%** are correct-and-inside, 0.256% are wrong (step 0). So at τ=0.15 the loss puts 7.67% of its
gradient on 0.088% of pixels (an 87× enrichment) and 77.7% on the 99.66% that cannot move the score.

### The floor — the answer to the charter's crossing question

| τ | τ/δ_R | outside% | inside% | **correct%** | wrong% |
|---:|---:|---:|---:|---:|---:|
| 0.200 | 9.14 | 82.830 | 5.674 | 88.503 | 11.497 |
| 0.150 | 6.86 | 77.715 | 7.670 | **85.385** | 14.615 |
| 0.050 | 2.29 | 47.066 | 23.185 | 70.251 | 29.749 |
| 0.043764 | 2.00 | 42.278 | 26.144 | **68.423** | 31.577 |
| 0.021882 | 1.00 | 17.326 | 43.183 | 60.509 | 39.491 |
| 0.010941 | 0.50 | 2.487 | 53.077 | 55.563 | 44.437 |
| 0.004000 | 0.18 | **0.002** | 52.156 | **52.159** | 47.841 |

**MEASURED crossings** (both cells, all 6 milestones):

| level | all-correct share | wasted share (correct AND outside `m_safe`) |
|---|---|---|
| < 50% | **never — floor 52.159% at τ = 0.004** | τ = **0.0544–0.0655** = **2.49–2.99 δ_R** |
| < 25% | **never** | τ = **0.0274–0.0306** = **1.25–1.40 δ_R** |

DERIVED: the floor is the near-boundary correct:wrong mass ratio (≈52:48). The field is 99.5%
correct, so even at the boundary the correct side carries slightly more mass. **Only a change of
FIELD — or a cap that stops pushing R-safe pixels — moves that ratio; τ cannot.**

## 2. Per class — Lane is the most-wasted class at every τ (MEASURED)

Control, DALI, HT. Cells are **class share of total grad% | outside/inside/wrong % of that class**.

| τ | step | Road | **Lane** | Undrivable | Movable | MyCar |
|---|---:|---|---|---|---|---|
| 0.15 | 0 | 52.37% \| 73.1/8.4/18.5 | **17.59% \| 87.9/5.5/6.5** | 15.05% \| 75.1/8.4/16.4 | 7.11% \| 83.9/6.3/9.8 | 7.88% \| 84.8/7.2/8.0 |
| 0.15 | 2,000 | 56.22% \| 67.0/8.4/24.6 | **14.77% \| 89.5/4.4/6.1** | 16.28% \| 69.1/8.0/22.9 | 4.64% \| 88.7/4.0/7.3 | 8.08% \| 83.3/6.7/10.0 |
| 0.15 | 5,000 | 54.83% \| 69.5/8.8/21.7 | **16.22% \| 89.8/4.5/5.7** | 15.48% \| 72.6/8.1/19.4 | 5.38% \| 88.7/4.3/7.1 | 8.10% \| 84.1/7.2/8.6 |
| 2 δ_R | 0 | 57.36% \| 37.9/26.2/35.9 | 13.73% \| 56.6/24.0/19.4 | 16.10% \| 40.3/27.0/32.6 | 6.01% \| 49.8/25.6/24.6 | 6.79% \| 48.4/28.4/23.2 |
| 1 δ_R | 0 | 59.45% \| 15.6/41.7/42.7 | 11.76% \| 23.9/45.5/30.6 | 16.47% \| 16.5/44.2/39.3 | 5.68% \| 21.7/45.1/33.3 | 6.65% \| 19.6/48.0/32.4 |

**Lane spends 87.9–89.8% of its gradient on pixels R cannot flip — the highest waste of any class —
and only 5.5–6.5% on repair, the lowest of any class.** This is the [[m131]] join in gradient units:
the class that is 0.59% of area but 33.56% of the model bits and ~90.1% of the rate demand is also
the class whose gradient is most nearly all spent on nothing.

**The cost of the τ band that nobody has priced (MEASURED).** Lowering τ *de-prioritizes Lane*:

| τ | Lane share of total grad (step 0 / 2,000 / 5,000) | Road share (step 0 / 2,000 / 5,000) |
|---|---|---|
| 0.15 | 17.59% / 14.77% / 16.22% | 52.37% / 56.22% / 54.83% |
| 2 δ_R | 13.73% / 9.28% / 10.86% | 57.36% / 63.54% / 62.21% |
| 1 δ_R | 11.76% / 7.76% / 8.74% | 59.45% / 65.18% / 64.73% |
| 0.5 δ_R | 11.00% / 7.36% / 7.81% | 60.03% / 65.52% / 65.54% |

Lane loses **1.60–2.08×** of its relative gradient share as τ falls 0.15 → 0.5 δ_R; Road gains it.
A τ-band race that does not watch the per-class share will silently trade the rate-binding class
for the majority class.

## 3. Per-class `m_safe_c` vs the global cap (MEASURED)

Per-class `m_safe_c = headroom · δ_R_c` (dr1's construction; headroom is a DERIVED policy factor
applied to each class's MEASURED `δ_R_c`, and dr1's caveat that a per-class headroom is *not*
derived travels with it):

| class | `m_safe_c` | ratio to global 0.043764 | direction of the global cap's error |
|---|---:|---:|---|
| **Lane** | 0.025712 | **0.5875** | **OVER-pushes** (global is 1.702× too high) |
| Movable | 0.036866 | 0.8424 | OVER-pushes (1.187×) |
| Road | 0.045154 | 1.0318 | under-protects (1.032×) |
| MyCar | 0.047705 | 1.0901 | under-protects (1.090×) |
| Undrivable | 0.052052 | 1.1894 | under-protects (1.189×) |

**The fraction of each class's gradient a GLOBAL cap over-pushes** (inside the global band, already
outside the class's own band) — step 0, control, DALI, HT; `+over / −under`:

| τ | Road | **Lane** | Undrivable | Movable | MyCar |
|---|---|---|---|---|---|
| 0.15 | +0.00 / −0.29 | **+2.76** / −0.00 | +0.00 / −1.47 | +1.30 / −0.00 | +0.00 / −0.69 |
| 0.05 | +0.00 / −0.73 | **+9.76** / −0.00 | +0.00 / −3.65 | +4.21 / −0.00 | +0.00 / −2.12 |
| 2 δ_R | +0.00 / −0.78 | **+11.23** / −0.00 | +0.00 / −3.91 | +4.69 / −0.00 | +0.00 / −2.32 |
| 1 δ_R | +0.00 / −0.80 | **+17.45** / −0.00 | +0.00 / −3.78 | +5.90 / −0.00 | +0.00 / −2.43 |
| 0.5 δ_R | +0.00 / −0.25 | **+10.73** / −0.00 | +0.00 / −1.04 | +2.46 / −0.00 | +0.00 / −0.72 |

**The per-class cap's value is a function of τ and PEAKS at τ = δ_R.** At the shipped τ = 0.15 a
global cap over-pushes only 2.76% of Lane's gradient — a per-class cap is nearly worthless there.
At τ = δ_R it over-pushes **17.45%** (14.11–17.87% across all 6 milestones of both cells). **The two levers
are coupled: the tighter the τ band, the more a global cap misallocates Lane.** A τ-band race that
goes below ~1.5 δ_R without the per-class cap is choosing to waste up to 17% of the gradient of the
class that holds 90.1% of the rate demand.

## 4. Row 1 (`_live_margin_weight`) — how much waste it removes, and where it puts it (MEASURED)

At the trainer's own τ for each milestone. Every allocator is mean-1, so the total budget is
identical by construction and only the ALLOCATION changes.

| config | step 0 (τ=0.15) waste removed | → to wrong | step 2,000 (τ=0.110) | step 5,000 (τ=0.05) |
|---|---:|---:|---:|---:|
| **`inverse@1.0` (SHIPPED DEFAULT)** | **3.7%** | +10.7% | 4.0% | 3.7% |
| `inverse@0.3` (documented anneal target) | 9.5% | +26.3% | 10.5% | 10.2% |
| `inverse@0.1` | 19.0% | +49.5% | 21.1% | 21.9% |
| `exp@1.0` | 4.4% | +13.1% | 4.5% | 4.0% |
| `exp@0.3` | 14.0% | +39.4% | 14.7% | 12.6% |
| **`exp@0.1` (best tested)** | **37.3%** | +95.8% | 38.1% | 33.4% |
| `bottom-k@0.01` | 11.8% | +39.4% | 8.6% | 1.0% |
| **`bottom-k@0.05` (FEED-bp's headline)** | **0.0%** | +0.1% | 0.0% | −0.0% |

**Two decisive negatives.**

* **`bottom-k@0.05` is EXACTLY inert — 0.0% at every milestone and at every one of the six charter
  τ.** DERIVED mechanism: a mean-1 hard mask is `1/0.05 = 20×` on the selected set and 0 elsewhere.
  FEED-bp MEASURED that the bottom-5%-margin set already contains ~89% of d_seg; this arm measures
  that it also already contains essentially all of the *gradient*, so the mask is a uniform rescale
  on the gradient's own support and every share is invariant. **The measurement that motivated row 1
  is the same measurement that makes its headline setting a no-op.** A row-1 race that fires
  `bottom-k@0.05` will measure nothing and report it as a null result for the family.
* **The SHIPPED default removes 3.7%.** Row 1 is not wrong; it is shipped **10× below** its own best
  tested setting. `[[m56]]` (unwired-but-built) in its temperature.

**Where the removed waste goes (step 0, `exp@0.1`):** outside 77.71 → 48.73 (−28.98 pp), inside
7.67 → 22.66 (**+14.99 pp**), wrong 14.62 → 28.61 (**+13.99 pp**). So **52% of the freed mass goes to
DEFENCE and 48% to REPAIR**, and the defence share rises as τ falls (67/33 at step 5,000). Row 1 does
not preferentially buy repair; it buys the annulus, both sides of it.

## 5. The head-to-head, and the DERIVED (τ_start, τ_end)

**Which lever removes more wasted mass per unit of change — DERIVED.**

| | τ band | row 1 |
|---|---|---|
| waste removed, best measured | **45.6% (2 δ_R) → 77.7% (1 δ_R) → 96.8% (0.5 δ_R)** | 37.3% (`exp@0.1`); **3.7% as shipped** |
| unit of change | **two default arguments** in one existing signature (`tau_for_step(step, total_steps, start=0.15, end=0.05)`, `ddm_qbt1_qbflow_trainer.py:643`) | a ~30-line MLX→torch port + a new hyperparameter whose shipped value is near-inert and whose headline value is exactly inert |
| also fixes | **sd1's schedule-leg defect** (below) | nothing on the schedule |
| also costs | Lane's gradient share falls 1.60–2.08×; Lane over-push rises to 17.45% at 1 δ_R | nothing measured here |

**The τ band wins on this vehicle**, on both legs of "per unit of change": larger effect, smaller
change. It is the third race's first move.

**Bonus, MEASURED — the τ band also repairs most of sd1's #0 item.** sd1's finding was that the
anneal deflates the reported loss by −40.54% on a frozen field, 8.4× the field's own signal. On this
arm's step-0 field the current band 0.15 → 0.05 gives a schedule leg of **−41.30%**; the proposed
band **2 δ_R → 1 δ_R gives −8.57%** — a **4.8× reduction of the artefact**, from a change that was
being made for a different reason. (It does not *remove* it: sd1's τ-invariant reporting cell is
still the cheapest full cure and is still worth firing.)

**DERIVED recommendation: `tau_for_step(start = 2·δ_R = 0.04376363754272461, end = δ_R = 0.021881818771362305)`.**

| | today 0.15 → 0.05 | proposed 2 δ_R → 1 δ_R |
|---|---|---|
| band in δ_R units | 6.86 → 2.29 | **2.00 → 1.00** |
| waste at τ_start | 77.72% | **42.28%** (−45.6%) |
| waste at τ_end | 44.44% | **16.16%** (−63.6%) |
| schedule leg on a frozen field | −41.30% | **−8.57%** |
| Lane over-push under a global cap | 1.90–2.76% → 1.90–2.30% | **9.2–11.2% → 14.1–17.9%** ⚠ |

Why these two ends, derived rather than picked:

1. **τ_end = δ_R, not lower.** Below δ_R the loss concentrates its gradient inside the band where the
   roundtrip's own noise decides the class — it would be optimizing structure R can erase. δ_R is the
   physical floor of "decided", so it is the physical floor of a useful temperature.
2. **τ_start = 2·δ_R = `m_safe`.** At τ = `m_safe` the gradient half-max sits at 1.76 `m_safe`, so the
   loss's soft band is matched to the satisficing band instead of being 4–12× wider than it (sd1
   MEASURED τ=0.15 = 6.86 δ_R with half-max at 12.08 δ_R). This is the same constant already resolved
   by `margin_band_satisficing_threshold_v1`, so the schedule stops being a free pair of literals and
   becomes a DERIVED consequence of the measured noise floor.
3. **The ratio 2:1 preserves an anneal** (coarse→fine survives) while cutting the schedule leg 4.8×.
4. **⚠ Co-fire the per-class cap, or hold τ_end at 1.5 δ_R.** The over-push table is the binding
   caveat: at τ_end = δ_R a global cap wastes 14.1–17.9% of Lane's gradient. **The τ band and the
   per-class `m_safe_c` are not independent races on this vehicle — the second is created by the
   first.**

**Row 1 composes; it does not compete.** Row 1's relative waste-removal is nearly constant in τ
(`exp@0.1`: 37.3 / 33.1 / 32.8 / 32.3 / 33.5% at τ = 0.15 / 0.05 / 2 δ_R / 1 δ_R / 0.5 δ_R), so the
two act near-multiplicatively — MEASURED across 9 (config × τ) combinations, deviation from the
multiplicative prediction within **±2.5 pp** (e.g. at 2 δ_R, `exp@0.1`: predicted 65.9%, measured
63.4%). A race may fire both; it should fire the τ band **first**, because row 1's contribution is
measured relative to whatever τ is already doing.

## 6. A MEASURED correction to a TRANSFERRED number

Row 1 allocates over the **unsigned** top1−top2 gap, which equals `|signed margin|` only when GT is
the top-2. dr1 carried hg1's figure that **GT is the runner-up on 98.018% of flips** — TRANSFERRED
from another vehicle. **On the QBF1-born vehicle I MEASURE 90.138%** (14,715 of 16,325 flips, PyAV
target, step 0; identical in both cells, which share the init; 1,531–1,684 divergent sites at every
milestone). **100.00% of the divergence sits on flipped pixels**, exactly as
`scalar_top1_top2_margin_is_exact_distance_to_flip_v1` predicts — on a correct pixel GT *is* top1, so
the two quantities are identically equal.

The divergence is **5.0× larger** than the transferred number. Its direction is benign for row 1: on
those pixels `gap < |signed margin|`, so the allocator over-weights genuinely-wrong pixels. But the
number itself must not be re-transferred at 98% on this vehicle.

## Scope and limits (these travel with every number above)

* **verdict_scope = FORMULATION.** Scoped to the sealed QBR1 objective (`expected_flip_margin_loss`
  + linear `tau_for_step` 0.15→0.05 over 5,000 updates) on the QBF1-born vehicle, seed 20260902,
  n32 sealed selection. It is a statement about where a temperature-annealed sigmoid's gradient lands
  relative to a measured noise floor; it does not close the surrogate family and it is not a verdict
  on any lever — **only a cell can give that.**
* **Axis** `[macOS-CPU advisory]`. Every input is the burn's own `[macOS-MPS n32 stratified advisory]`
  retained payload. **No score claim. Nothing here is promotable. The pointer is untouched.**
* **n32 TRAINED-selection only, STRUCTURALLY.** `_evaluate_milestone` materializes only
  `qbt.SELECTION_IDS` (`ddm_qbr1_born_fairform_burn_prep.py:433`), so the 568 unfitted pairs have no
  retained logits at any milestone. Every number is the trained n32 population and never mixed with
  an unfitted read.
* **n = 2 cells, 1 seed** — a same-seed repeat, bounded at ≤1.02 pp (Calibration 6), NOT a seed
  replication.
* **`δ_R` and `m_safe` are TRANSFERRED-from-n600 constants applied to an n32 field.** dr1 measured
  them over 600 PyAV frames; this arm applies them to the 32 sealed pairs. The n32 selection's own
  δ_R is not measured here and could differ — [[m88]] cuts both ways, and this is the honest exposure.
* **Row 1 is ported, not the MLX original.** Mean-1 is applied **per pair**, matching the per-forward
  normalization of `:1120`; a batch-level mean would give different shares. Its `stop_gradient` is a
  no-op here because this instrument never differentiates.
* **Row 1's numbers are a STATIC re-weighting of a FROZEN field.** They say where a weight would put
  the gradient at that milestone, NOT what a run trained with the weight would do. Its
  `--margin-weight-start-epoch 80` curriculum (`:3305`) exists precisely because the allocator starves
  the base from random init; nothing here measures that dynamic.
* **The τ recommendation is DERIVED, not raced.** `treatment_delta_s` for both levers remains
  **UNMEASURED**. This arm produced inputs for a charter; it did not run a cell.
* **float16 storage** perturbs the retained logits; sd1 MEASURED 1.97e-05 of sites. Cured for the
  denominators here by taking the flip indicator from the retained argmax — which is why the
  recompute is bit-exact (Calibration 1).

## Equations leg (`tac.canonical_equations`)

**`margin_band_satisficing_threshold_v1` — CONSUMED, IN-DOMAIN, and consumed the RIGHT WAY.**
Its `domain_of_validity.included` names "SegNet signed-margin units measured by the delta_R
artifact", which is exactly the unit of every split above. This arm resolves `m_safe` through
`resolve_margin_band_threshold()` at runtime rather than carrying the decimal — pinned by
`test_module_carries_no_hardcoded_m_safe_literal`, which fails if any of the three historical
`δ_R`/`m_safe` literals appears in the source. That is the `[[m107]]` split-banks cure applied at
the point of use, and it is why this arm automatically read dr1's n600 repoint (`fallback_used
False`, `n_frames 600`) instead of the retired n96 value.

**No anchor appended.** The law's `empirical_output` is `δ_R` / `m_safe`; this arm measures neither —
it measures how gradient mass distributes *relative to* `m_safe`. An anchor whose output is a
gradient share would teach the posterior a quantity the law does not predict. Appending it would be
the shape-mismatch sibling of the cross-vehicle transfer the campaign has extincted.

**`scalar_top1_top2_margin_is_exact_distance_to_flip_v1` — CONSUMED as the premise, and REFINED.**
It is why `1[margin < 0]` is the exact flip term and why "the gap equals `|signed margin|` on every
correct pixel" is a theorem rather than a hope — MEASURED here: 100.00% of the 1,531–1,684 divergent
sites per milestone are flips, zero are correct pixels. **REFINEMENT owed to its downstream transfer,
MEASURED here:** the "GT is the runner-up on 98.018% of flips" figure is **90.138% on this vehicle**,
a 5.0× larger divergence. **No anchor appended** — the law's `domain_of_validity.vehicle` is
`softmax_of_sdf_levelset_witness` + `frozen_contest_segnet`, and QBF1-born is a different vehicle
sharing only the frozen scorer (sd1 declined for the same reason; [[m21]], [[m143]], [[L18]]).

**FORMALIZATION_PENDING** — the law this arm's headline needs does not exist:

> *For a temperature-annealed sigmoid surrogate over a piecewise-constant argmax field, the fraction
> of gradient mass on already-correct pixels does not vanish as τ → 0: it converges to the
> near-boundary correct:wrong mass ratio of the field, a FIELD invariant independent of τ. Only the
> sub-fraction outside the R-noise band `m_safe` is removable by temperature; the remainder is
> irreducible defence, and its floor is what a satisficing cap — not a temperature — must address.*

MEASURED floor on this vehicle: **52.159% at τ = 0.18 δ_R**, with the removable part driven to
0.002%. It should be registered once a τ-band cell has been burned, so it anchors on a measurement of
the cure rather than on this diagnosis. Its callable is already
`ddm_gm1_gradient_mass_msafe.group_shares` + `first_crossing` over `accumulate_pair`'s bins.

## GESTALT-DELTA

The campaign's gestalt, from sd1, held that **"67–85% of the seg gradient is wasted on already-correct
pixels"** and that the cure was a sharper τ. **Both halves need correcting.**

**First: "wasted" was over-counted, and the over-count grows as the cure is applied.** At the shipped
τ the waste is 77.7%, not 85.4% — the missing 7.7 pp is legitimate defence of pixels R can still flip.
As τ falls the two invert: at 1 δ_R the split is 17.3 waste / 43.2 defence, and the *defence* group is
then 2.5× the waste. **A "wasted-gradient" number quoted without its `m_safe` split is a different
number at every τ, and it moves in the direction that flatters the lever being sold.** This is the
`[[m99]]` units×level×aggregation genus at the *partition* axis: sd1's quantity and mine are both
correct and they are not the same quantity.

**Second: the correct-pixel share has a FLOOR that no temperature crosses.** τ is a *removal* lever
for the outside-band waste (96.8% of it by 0.5 δ_R) and a *conversion* lever for the rest — it turns
waste into defence, not into repair. The wrong-pixel share only goes 14.6% → 44.4% while the defence
share goes 7.7% → 53.1%. **Sharpening the loss does not make it a repair loss; it makes it an annulus
loss.** Anything that wants *repair* to dominate has to change the field's near-boundary correct:wrong
ratio, which is a capacity/representation question, not a schedule question.

**Third, and the one that ranks the third race: the two candidate levers are coupled, and their
coupling runs through Lane.** Lowering τ (a) removes 45.6–96.8% of the waste, (b) drops Lane's share
of the gradient by 1.60–2.08×, and (c) raises the Lane over-push of a *global* cap from 2.76% to
17.45%. Lane is 0.59% of area, 33.56% of the model bits, ~90.1% of the rate demand — so the τ band
buys d_seg-relevant focus by spending exactly the class the rate corner cannot afford, **and creates
the per-class-cap race it would otherwise have made unnecessary.** `[[m148]]` (object-change, not
jointness) reads directly: the τ band changes the *object* the per-class cap acts on, which is why
these are one race with two legs and not two races.

**Fourth, a cheap general detector:** row 1's `bottom-k@0.05` being **exactly** inert is a reusable
shape. **A mean-1 hard-mask allocator whose selected set is a superset of the gradient's support is a
uniform rescale and cannot re-allocate anything.** The very measurement used to justify such an
allocator (the concentration statistic) is the measurement that predicts its inertness. Before racing
any masking allocator, check whether its mask already contains the mass — that check is free and it
would have saved this race a null cell.

## NEXT_IF_RESUMED — every row carries a disposition, an owner and a fire condition ([[m113]])

| # | follow-on | disposition | owner | fire condition |
|---|---|---|---|---|
| 1 | **`TAU-BAND-RACE` at (2 δ_R, 1 δ_R)** — the DERIVED first move of the third race; change is two default args at `ddm_qbt1_qbflow_trainer.py:643`. Pre-registerable falsifiers from this arm: waste share at step 0 lands 42.3% ± 1.0 pp; the schedule leg lands −8.6% ± 2 pp; Lane's gradient share falls to 13.7% ± 1.0 pp. | **QUEUED-WITH-FIRE-ORDER, fires FIRST** | MAIN to assign a cell | the next QBR1-lineage burn slot |
| 2 | **Co-fire the per-class `m_safe_c` cap with #1** — NOT a separate race. At τ_end = δ_R a global cap wastes 14.1–17.9% of Lane's gradient; dr1's row-6 `m_safe_c` (Lane 0.025712) is the cure and is already derived. | **FOLDED into #1** | same cell as #1 | fires with #1; do not fire standalone |
| 3 | **Fix row 1's default before any row-1 race** — the shipped `inverse@1.0` removes 3.7%, `exp@0.1` removes 37.3%, and the headline `bottom-k@0.05` removes **exactly 0.0%**. A race at the shipped or headline setting measures a no-op and would report it as a family negative. | **QUEUED, fires only if row 1 is raced** | row-1 racer | fires when row 1 is raced; sweep the temperature, never the allocator name alone |
| 4 | **sd1's `TAU-INVARIANT-REPORTING-CELL` is still worth firing** even though #1 cuts the schedule leg 4.8× (−41.3% → −8.6%). It is ~1 extra sigmoid per update and it makes #1's own falsifiers readable in `history.jsonl` in real time. | **QUEUED-WITH-FIRE-ORDER, fires WITH #1** | MAIN | the next QBR1-lineage burn |
| 5 | **Re-measure `δ_R` on the n32 sealed selection** — every constant here is an n600 population value applied to 32 pairs; [[m88]] says a sub-population is a different population and dr1 MEASURED that annulus-restricted constants are exactly where that bias hides (+11.70% annulus vs +0.45% global). $0, reuses dr1's retained `m0`/`m1`. | **QUEUED, no fire order** | unowned; MAIN to assign or close | fires if #1's measured waste share misses this arm's ±1.0 pp prediction |
| 6 | **Sweep the mask-superset detector across the lever corpus** — §GESTALT-DELTA 4 gives a free static check for inert mean-1 masking allocators. UNOWNED; naming it without owning it would be the deferral scatter this repo extincted ([[m36]]), so it is explicitly unowned. | **QUEUED, needs a census first** | unowned | fires only if a second masking allocator is proposed for a race |

## DEAD-ENDS

* **"Lower τ until the loss stops defending correct pixels" is CLOSED.** The all-correct share has a
  MEASURED floor of 52.159% at τ = 0.18 δ_R and never reaches 50%, let alone 25%, anywhere in
  τ ∈ [0.004, 0.200]. Do not re-open the crossing question for this field.
* **"`bottom-k@0.05` concentrates the loss on the annulus" is CLOSED as a re-allocation claim** — it
  is exactly a uniform rescale on the gradient's own support, 0.0% share change at 6 milestones × 6 τ
  × 2 cells. It survives only as an interaction with terms this arm does not model.
* **"Row 1 and the τ band are alternatives" is CLOSED** — they compose near-multiplicatively on the
  waste (9 combinations, ≤2.5 pp from the multiplicative prediction). The race is an ordering
  question, not a selection question.
* **"The per-class cap is an independent row-6 variant" is CLOSED** — its value is 2.76% of Lane's
  gradient at the shipped τ and 17.45% at 1 δ_R. It is a *dependent* leg of the τ-band race.
* **"98% of flips have GT as the runner-up" is CLOSED as a transfer to this vehicle** — MEASURED
  90.138%, a 5.0× larger divergence. The 98.018% figure remains hg1's, on hg1's object.
* **Re-rendering or re-training to get these splits is CLOSED as unnecessary** — every number above
  came from sd1's already-retained milestone logits at $0. sd1's instrument was imported, not rebuilt.

## Custody (ALWAYS KEEP THE PAYLOAD)

Store root `/Volumes/APDataStore/pact/ddm_gm1_gradient_mass/` (27 MB). **Nothing was written under
the live chain's `runs/`, `authorized_configs/` or `CHAIN_LEDGER.jsonl`; the QBR1 run directory was
opened read-only and the claims ledger was not touched.**

| artifact | bytes | sha256 (first 16) |
|---|---:|---|
| `measure/GM1_COMBINED.json` (all four analysis views per cell + cross-cell agreement) | 6,436,517 | `dc138b60a092b122…` |
| `measure/control_native100/GM1_REPORT.json` (all bins, 41 τ) | 8,268,971 | `6fd6ef1650480909…` |
| `measure/control_native100/pair_rows.jsonl` (192 rows) | 74,082 | `a0939a21ce134fb8…` |
| `measure/treatment_zero_native/GM1_REPORT.json` | 8,269,098 | `66ddc4913cd80cd3…` |
| `measure/treatment_zero_native/pair_rows.jsonl` (192 rows) | 74,094 | `0e016aed03f90906…` |
| `run_both_cells.sh` | 490 | `e1d693b0cd8f902e…` |
| `launch/` launch manifest, `run.log`, `safe_run` status receipt (status ok, 176.5 s) | — | — |
| instrument + 54 tests | — | `experiments/ddm_gm1_gradient_mass_msafe.py`, `src/tac/tests/test_ddm_gm1_gradient_mass_msafe.py` |

The payload is COMPLETE, not decorative: `--reanalyse` re-derives every table in this memo from the
stored bins without re-reading the QBR1 run directory, so any τ, class, group or cell is recomputable
without the 87 s/cell pass. Per-bin sha256s and file facts are in `GM1_REPORT.fact.json` beside each.

## Apparatus

* sd1's instrument was **imported, never rebuilt** (SPEC_v75 §8B): `margin_and_competitor`,
  `stable_sigmoid`, `tau_for_milestone`, `read_pair_arrays`, `read_milestone_json`,
  `sample_weight_lookup`, `atomic_json`, and ar1's GT-lineage loader all come from
  `experiments/ddm_sd1_surrogate_exact_map.py`. This arm adds only the `m_safe` partition, the
  row-1 allocator port, and the crossing solve.
* Governed launch: `tools/launch_detached_process.py`, nice 10, `torch.set_num_threads(4)`,
  wall cap 3600 s. MEASURED wall **176.5 s total** (86.9 s + 87.0 s), peak RSS ~0.6 GiB, exit ok.
* 54 tests, ruff clean, review-gate two passes per entity. One defect was found by the tests and
  fixed in this arm's own kernel (Calibration 4), and its direction was corrected by tracing rather
  than by assumption.

---

Pointer honesty: this arm measured the INPUTS to a race. It trained nothing, byte-closed nothing, and
could not move the frontier.

Own-vehicle frontier: **afr1 S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]** — UNMOVED.
