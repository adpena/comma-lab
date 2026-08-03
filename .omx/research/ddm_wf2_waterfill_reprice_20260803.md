# `ddm_wf2` — the waterfill re-price: one exchange rate, many mechanism prices

**Arm:** `ddm_wf2` · 2026-08-03 · **scorer-free** (0 scorer forwards; `ddm_pu2` holds the slot)
**Axis:** `[macOS-CPU advisory]` — `score_claim=false`, `promotion_eligible=false`,
`rank_or_kill_eligible=false`, `pointer_moved=false`. Exact contest pointer `0.1910828242` UNMOVED.

**BASELINE for every ΔS below:** live best **S = 0.7910689**, **353,805 B**, seg leg **0.4311790**,
`cx1` flips **508,639**. **TARGET:** PR130 bar **0.172141**. **LIVE gap = 0.6189279.**
Every number is recomputed from a named receipt in this session; none is re-typed.

---

## §0 HEADLINE — four claims, in the order they change decisions

1. **The single `W` is NOT the defect.** `seg_term = 100·flips/PX` and `rate = 25·B/DEN` are both
   **exactly linear**, so seg+rate has no diminishing returns and therefore **no equal-marginal
   condition to satisfy**. The optimal rule over mechanisms is a **threshold** ("take everything
   priced better than `W`"), not an equalization — and `W` is exactly invariant (`ddm_op3`).
   Differing mechanism prices are *expected and harmless* at the portfolio level.
   **The genuine waterfill lives at exactly one place: the concave pose leg `√(10·d_pose)`.**
2. **The defect is real, it is INSIDE an allocator rather than between them, and it is confined to
   ONE allocator.** Where a fixed byte budget is split across many channels the greedy needs a
   correct **per-channel** price, and `wr1`'s is corrupted on **both legs** — bytes `ρ = 0.513`,
   damage support **24.2× too small** (`rs2`, MEASURED). **The other three named allocations have
   ZERO measured prices** and therefore cannot be re-priced at all: `c1`/EV2 is **100% unallocated**
   (`FORMULATION_MISPOSED`), `rd1` has **162/162 null duals**, `ms2r`/`r3` executed **zero rungs**
   with a **null knee** (§4.1, all recomputed from their own artifacts). **`wr1` is the only
   allocator in scope that ever assigned bytes by a price — so it is the whole exposure.**
3. **The quoted prices are NOT comparable as printed — five distinct incomparabilities**, §2. The
   sharpest: **`sx1`'s 0.5349 B/flip reproduces EXACTLY from its own artifact**
   (253,341 / 473,651 gap-flips = 0.5348684). `ddm_sx2`'s "defect found" is a **denominator
   substitution, not an arithmetic error** — and my charter inherited the mis-attribution.
   Correcting the correction: **§2.2.**
4. **The dimension tax is real, per-mechanism, and MEASURED for exactly one mechanism.** It is
   **1.000 by construction** for every description-space mechanism now at the top of the list, and
   **2.07–2.17×** for `rz1`'s chroma actuator (efficacy 0.484 proxy / 0.460 corpus-direct). The
   36× head-space tax (rank 4 of 144) applies to a space **no mechanism acts in**.
   **Net: the tax prices one qualitative claim; it reorders nothing.**

5. **The one place re-pricing pays TODAY is the banked pose shelf.** `W` is identical to 1e-12
   across the two live-best baselines in circulation; `dS/d(d_pose)` moved **1.285×** between them
   (§7.6, an exact reconciliation — the two "competing" live-bests are the **same archive**, and the
   whole 0.0354 S difference is pose). Composed with `op3`'s measured **1.73× since `pw1`**:
   **pose levers banked at `pw1`-era prices are under-priced by ≥ 2.22×, and the error COMPOUNDS as
   pose improves.** No new build required — this is a re-read of an existing shelf.

**Honest bound (§4): the ordering changes WITHIN families; the top of the list does NOT change.**
The one measured inversion in the campaign (`#826`, `−0.0983 → +0.0035`) was a **baseline** move,
not a **price** move — and my charter conflated them.

---

## §1 CONSTANTS — recomputed this session, never re-typed

| quantity | value | derivation / receipt |
|---|---:|---|
| `DEN` | 37,545,489 | `upstream/evaluate.py` rate denominator (**live quantity** — Catalog #812) |
| `PX` | 117,964,800 | `600 × 384 × 512` |
| `W` | **1.2731082153320312** B/flip | `4·DEN/PX` — **DERIVED, exactly invariant** |
| `cx1` flips (total) | **508,639** | `d_seg 0.004311790 × PX = 508,639.44` |
| PR130 floor `d_seg` | 0.0002966 | `rz1` §R3 |
| **gap-flips** | **473,651** | `(0.004311790 − 0.0002966)·PX = 473,651.09` |
| **gap-flip byte budget** | **603,009 B** | `473,651.09 × W` (reproduces `rz1` R3's 603,009 ✅) |
| `d_pose` at live best | 0.00154519 | `(0.7910689 − 0.4311790 − 25·353805/DEN)² / 10` |
| `dS/d(d_pose)` at live best | **40.223** | `5/√(10·d_pose)` — **strongly point-dependent** |

**`W` is an EXCHANGE RATE (what a flip is *worth*), never a mechanism COST.** The distinction is
load-bearing for everything below and is the reason a "price table" is not automatically a ranking.

---

## §2 COMPARABILITY FIRST — five ways these prices are not the same quantity

*(Round-1 self-review named this as the likeliest failure: a beautiful table whose entries are not
comparable. So this section comes BEFORE the table, and it disqualifies three of the seven rows
from ranking at all.)*

### 2.1 — SIGN: buy-side and sell-side have **opposite** good directions

| kind | mechanism spends | mechanism gains | good when |
|---|---|---|---|
| **BUY** | bytes | flips | price **< `W`** |
| **SELL** | flips | bytes | price **> `W`** |

`mf1` (0.01564) and `ba31` (0.6498) are both "B/flip" and `mf1`'s is 41× *smaller* — yet `mf1` is a
**81.4× win** and `ba31` is **dominated 1.96×**. **Ranking these two by numeric value is not merely
imprecise; it inverts the verdict.** No document in the corpus prints the direction column.

### 2.2 — DENOMINATOR: total-flips vs gap-flips, and the mis-attribution my charter inherited

`ddm_sx2` §1 reports a "defect found in `sx1`'s headline": *"0.5349 B/flip … does not reproduce from
its own artifact: 253,341 B ÷ 508,640 flips = 0.4981 … The quoted figure implies a denominator of
473,623 flips, which is neither `cx1`'s 508,640 nor `pc2`/`tb1`'s 458,738."*

**Recomputed — 473,651 IS a named quantity, and it is the right one for a headroom question:**

```
253,341 / 473,651 (GAP-flips) = 0.5348684  → rounds to 0.5349   ✅ sx1 reproduces EXACTLY
253,341 / 508,640 (TOTAL flips) = 0.4980753 → 0.4981            ✅ sx2 also correct
ratio = 508,640 / 473,651 = 1.07387        → sx2's "7.4% cheaper"
```

`sx2`'s back-derived 473,623 differs from 473,651 by **28 flips = the rounding of a 4-dp quote**.
**`sx1` had no arithmetic error.** It priced the description against the flips it must *fix*
(gap-flips); `sx2` re-priced it against the flips that *exist* (total). Both are defensible; they
answer different questions. **Two arms quoted "B/flip", differed by 7.4% for purely denominational
reasons, and one reported the other as defective. My charter carried the mis-attribution forward.**

**The full spread is a 2×2 that nobody has stated** — {which description} × {which denominator}:

| description of exact GT `L*`, n600 | bytes | ÷ gap-flips | ÷ total flips |
|---|---:|---:|---:|
| `sx1` H1 order-1 (the quoted floor) | 253,341 | **0.5349** | **0.4981** |
| `sx1` H2 order-2 | 237,136 | 0.5007 | 0.4662 |
| `sx2` order-4 unconditional | 238,945 | 0.5045 | 0.4698 |
| **`sx2` order-4 + free canny (cheapest MEASURED)** | **216,395** | **0.4569** | **0.4254** |
| `lzma9e` ×10 (real codec) | 428,120 | 0.9039 | 0.8417 |

*(`sx1` H1 recomputed from `ddm_sx1_label_field_mdl_n600.json`: `(1,986,727.72 + 40,000)/8 =
253,340.96` ✅. `sx2` best from `ddm_sx2_conditional_mdl.json` `conditional_dilated.total_bytes` ✅.)*

**Spread across the 2×2: 0.4254 → 0.5349 = 1.2572×, i.e. 25.7%** — from choices, not measurements.
And the *cheapest measured description is `sx2`'s 216,395 B, not the 253,341 B every downstream
headroom figure uses.* Corrected headroom: **6.530 bits/gap-flip, not 5.906.**

### 2.3 — KIND: description-cost, purchase-price, and residual-budget are three different objects

`rz1` §2.5's three rows read like three prices. **They are ONE budget over three denominators** —
and that budget is *derived from* `sx1`'s description cost, so **`rz1` and `sx1` are not independent
measurements**:

```
budget          = 473,651.09 × W − 253,341 = 349,668 B          (rz1 R3: 349,668 ✅)
÷ 473,651 flips = 0.73824 B = 5.906 bits/gap-flip               (rz1: 5.91 ✅)
÷ 2,566,212 sep px = 0.13626 B = 1.090 bits/px                  (rz1: 1.09 ✅)
÷ 4,684,382 band px = 0.074646 B = 0.5972 bits/px               (rz1: 0.60 ✅)
1 flat bit/px over the band = 585,548 B = 1.6746 × budget       (rz1: 1.67 ✅)
```

All five reproduce to the printed precision. **`ddm_dd1` §4 independently caught the sister half of
this** — that `rz1`'s 0.60 bits/px (a *budget* over the dilated 4,684,382-px band) and `sx1`'s 0.6984
bits/px (an *entropy* over the strict 2,551,382-px band) "are a different quantity … noted so the two
are never averaged." **I confirm dd1 and extend it: all three `rz1` rows collapse to one number, and
that number is a residual, not a price. It cannot enter a price ranking at all.**

### 2.4 — BETWEEN-ARCHIVE difference quotients are not mechanism prices

`qd1`'s **32.53 B/flip** recomputes exactly (`5,413 B ÷ 166.45 flips = 32.5206`; `25.544 × W`) — but
its numerator and denominator are the **total difference between two whole archives from two
lineages** (`gr1_cell_drop50`, a v4d-era candidate, vs `cx1`, the live best). It is a
difference quotient, not the marginal cost of any mechanism. **Its correct use is exactly the one
`qd1` gives it — a SPECIFICATION ("re-encode within 212 B of `cx1`") — not a rank.**

### 2.5 — BASE: three flip counts are in circulation and they are not interchangeable

**508,639** (`cx1` live) · **473,651** (gap to PR130) · **458,738** (`pc2`/`tb1` burn, `d_seg`
0.0038892). Any "B/flip" inherits whichever base its author had open. Spread 508,639/458,738 = 1.109.

---

## §3 THE PRICE TABLE — with unit, direction, acting space, and measured overlap

**RANKABLE rows** (a real marginal cost of a real mechanism, direction stated):

| # | mechanism | receipt | recomputed price | unit | dir. | acting space | `range(A)` overlap | **effective price** | vs `W` |
|---|---|---|---:|---|---|---|---:|---:|---|
| 1 | **`mf1`** Movable per-component displacement | `ddm_mf1` §5.3 | **0.0156417** | B / flip bought | **BUY** | label-component (2-vector) | **1.000** by construction (translating `L*` regions *is* the seg-visible object) | **0.0156417** | **81.39× BETTER** |
| 2 | **`sx2`** sub-pixel phase carrier, normal-gauge halved | `dd1` §3 | 1,098 B → **549 B** | B / carrier | **BUY** | boundary-segment **normal** | **0.500** in DOF terms (tangential `δ_t` is a gauge while `κL ≪ 4`) | **2× cheaper than as shipped** | not yet flip-denominated |
| 3 | **`ba31`** drop-more | `rs2` §2.2 (n600) | **0.6498** | B / flip sold | **SELL** | token lattice cells | support **24.2× larger** than the key assumes (`rs2`) | 0.6498 | **dominated 1.959×** |
| 4 | **`gr1`** cell sweep | `rs2` §2.2 (**n48 — PRIOR, not evidence**) | **0.6298** | B / flip sold | **SELL** | token lattice cells | as #3 | 0.6298 | **dominated 2.021×** |

**NOT-RANKABLE rows** (recomputed and reported, but disqualified from ranking by §2):

| mechanism | recomputed | why it cannot be ranked |
|---|---:|---|
| `sx1`/`sx2` exact-`L*` description | 0.4254 – 0.5349 B/flip (2×2, §2.2) | an **average description cost**, not a marginal purchase price; and it is a **floor**, not a mechanism |
| `rz1` §2.5 "three prices" | ONE budget, **349,668 B** | a **residual**, and *derived from* the row above — not independent (§2.3) |
| `qd1`/`gr1_cell_drop50` | 32.5206 B/flip | **between-archive** difference quotient (§2.4) |
| `rz1` A2 pose-free chroma steering | **B/flip UNMEASURED** | see §3.1 — the tax is measured, the price is not |
| `wr1` Knee A "free" tranche | claimed ∞ (zero flips) | **falsified for 144 of 486 cells** by `rs2`; the true price is finite and unmeasured |
| head rank-4 quotient (140 of 144 null) | tax 36.0× | `sx1`'s own caveat: lives in **decoder feature space**; exploiting it "requires inverting the encoder … not directly a byte lever" |

### 3.1 — The dimension tax, derived per mechanism (operator §2)

The correction is `p_eff = p / overlap`, where `overlap` is the fraction of the mechanism's spend
that lands in the **visible subspace of the space the mechanism acts in**. It is **not one number**,
because the mechanisms do not act in one space. Four measured nullities, four different spaces:

| space | ambient | visible | nullity | isotropic tax | which mechanisms act here |
|---|---:|---:|---:|---:|---|
| SegNet head feature window | 144 | **4** (σ₅ = 3.7e-16, `sx1` §2.3) | 97.2% | **36.0×** | **NONE** — requires encoder inversion |
| camera plane → scorer plane via `D` | 1,017,336 /frame | 196,608 | **80.6742%** (`rz1` R1b ≡ corpus #580 to 4 dp) | **5.174×** | any raw pixel-residual carrier |
| — of which **blind to BOTH scorers** | 230,904 px | 0 | 100% | **∞** | must receive zero bits |
| PoseNet `yuv6` per scorer block | 12 | 6 | 50% | 2× | `rz1` A2 (pose leg — the point is nullity, not tax) |

**The decisive consequence, and it is the opposite of the intuition:** `D` is a **disjoint 4→1
partition** (`rz1` R1: each scorer px reads a *private* 2×2 camera block, no cross-talk). So the
5.174× is the tax on an **uncoordinated** camera-plane edit only. A **block-coordinated** edit pays
**1.000** — "for any target scorer-plane value, setting the 4 private camera pixels to it realizes
it exactly." **The tax is a property of the carrier's design, not of the operator.**

**Measured overlap, the one mechanism that has one.** `rz1` §2.1 measures the pose-free chroma
actuator retains **48.4%** of the discriminative direction (spatial-chain proxy, `INFERRED`), with
the corpus's *direct* gradient measurement at **46.0%**:

```
tax = 1 / 0.484 = 2.066×      tax = 1 / 0.460 = 2.174×
```

`rz1` already prices this itself — *"paying a 2× directional loss to buy exact pose-freeness"*
against a **79× pose penalty** for luma edits. **The tax is 2.07–2.17× and the trade still clears by
36×.** Its authority is separately MEASURED and large (isoluminant chroma at amp 32 moves
`Δd_seg = 2.73e-3` at n96 = **68% of the entire 4.015e-3 seg gap**) — but **its B/flip is UNMEASURED**,
so it cannot be ranked against `mf1` today. *That is the single highest-value owed price in this memo.*

**Why the tax reorders nothing today:** rows #1 and #2 act in **description space** — they emit a
description of `L*` (a component translation; a boundary-segment phase), and `L*` *is* the object
`d_seg` reads. Overlap 1.000 by construction. **A description-space mechanism cannot pay a
visibility tax.** What it pays instead is a **realization** risk — and that is the same crux the
campaign already named (`crux = REALIZATION`), now with a price attached:

> **The dimension tax and the realization crux are one object seen from two sides.** A carrier that
> acts in camera space pays the tax *up front and measurably* (`rz1`: 2.07–2.17×). A carrier that
> acts in description space defers the identical cost to *whether the decoder can realize the
> description at all* — where it is **unmeasured**, and therefore looks free.

**This is why `mf1` looks 81× better than everything.** Its 0.0156417 B/flip prices the
**description** (a 2-vector per component). Its realization multiplier is exactly its blocker **A8**
("displacement is rigid per component"), which `dd1` §2 has now moved from
`ASSUMED_AWAITING_VERIFICATION` to **actively contradicted** (`mf1`'s two supporting cross-checks
both fail: the continuum `(2/π)` flip model undercounts by **2.37×** vs `sx2`'s measurement, and the
2.26 px it implies produces ≈2.4% off-3px against the 18–20% observed, which needs `d ≈ 5 px`).
**`mf1`'s price survives; its support does not.** Priced as a realization-efficiency `η`:

| `η` (rigid-translation adequacy) | effective B/flip | vs `W` |
|---:|---:|---|
| 1.00 | 0.01564 | 81.4× better |
| 0.50 | 0.03128 | 40.7× better |
| 0.10 | 0.15642 | 8.1× better |
| **0.0123 (break-even)** | **1.2731** | **1.0× — the floor** |

**`mf1` clears `W` for any `η > 1.23%`.** That is the honest shape of its 81.4×: not a precise
price, but a *very* wide margin over a *contradicted* assumption. It stays #1 on the BUY list under
any plausible `η`, and its rank is therefore **not price-sensitive** — it is **support-sensitive**.

---

## §4 THE RE-ALLOCATION — which allocations move, named one by one

**The governing fact, and it decides four of the five: `seg_term` and `rate` are EXACTLY linear in
(flips, bytes).** A linear objective has no equal-marginal condition. So *across* mechanisms the
optimal policy is a threshold at `W`, `W` never moves (`op3`: "exactly invariant"), and **discovering
that prices differ by 65× does not make a threshold policy suboptimal — it makes it emphatic.**
The equal-marginal condition binds only where the objective is **concave**: `√(10·d_pose)`.

| allocation | what it actually prices | does re-pricing move it? |
|---|---|---|
| **`#766` / `wr1`** | `np.lexsort((-residual_mass, flip_mass))` — an implicit per-cell price `residual_mass / flip_mass` | **YES — MEASURED, and both legs are corrupted.** Bytes: `residual_mass` correlates **ρ = 0.513** with the real per-cell byte marginal over 384 exact re-encodes. Damage: the key's 16×16 tile (256 px) understates the MEASURED receptive field (84×82 = 6,192 px) by **24.2×**, so **144 of 486** "provably safe" zero-flip cells are not — the free tranche is **29.6% smaller** than Knee A claims. `rs2` has already **BUILT** the byte-matched A/B (274,631 vs 274,321 B, residual 310 B = 0.4%); at equal bytes the re-priced key carries **27.9% less ambient flip mass**. **Direction measured; ΔS owed (needs the scorer slot `pu2` holds).** |
| **`c1` waterfill tables** (via EV2) | the C1 composition's pair-cell dual allocation | **NO — it allocated NOTHING.** EV2 *"conserves the C1 byte total by leaving **100% unallocated**: assigned pair-cell bytes are zero, all 162 duals are non-computable"*, exact verdict **`FORMULATION_MISPOSED_FOR_CURRENT_C1_COMPOSITION`** (`codex_findings_…366box…`). There is no price to re-price. |
| **`rd1` λ-continuation frontier** | per-dimension duals `λ_bytes_per_D_dimension` over 162 cells | **NO — 162/162 prices are `None`.** Recomputed from `rd1_162_dual_backfill.json`: `actionable_cell_count = 0`, `lambda_measured_cell_count = 0`, `rung_measured_cell_count = 0`, `effective_quantum_D = 0.0` on **162/162**, `lambda_measurement_status = STILL_NULL_…` on **162/162**. `metric_context_cell_count = 162` — every cell has metric *context* and no *price*. |
| **`#869` token-by-token** | **UNKNOWN** — not resolved | **Did not find in the named scope** {the four receipt dirs + `codex_findings_*` read this session}. A parallel extraction over `.omx/research`, `.omx/state`, `experiments/`, `src/` was dispatched and had not returned at seal. **Honestly open.** |
| **`ms2r`/`r3` typed Fisher waterfill** | a **typed** (per-box) Fisher marginal — the "many prices" structure the operator's §1 asks for | **NO — IT NEVER FIRED.** From `priced_rung_table.json`: `measured_task_rungs = []` (**zero**), `knee = None`, `knee_status = NULL_NO_TYPED_HOMOTOPY_CURVE`, `preregistered_rung_status = NOT_EXECUTABLE_UNTIL_ALL_TYPED_PRECONDITIONS_CLOSE`; all six ladder rungs carry `epistemic_status = DERIVED_PREREGISTRATION_NOT_MEASURED` + `execution_status = BLOCKED_PRECONDITION_NOT_RUN`. `MS4D` records *"zero waterfilled rungs."* |
| **`qd1` / `#826`** | `gr1_cell_drop50` vs a v4d-era reference | **Already inverted (`−0.0983 → +0.0035`) — but by a BASELINE move, not a price move.** My charter cited this as price-sensitivity evidence; it is not. It is the sister defect ("a ΔS without its baseline is unanchored"). Keeping the two apart is the point of §2.4. |
| **the pose leg** | `dS/d(d_pose) = 40.223` at live best | **YES, and this is the ONE true waterfill.** `op3` measured this marginal has risen **1.73×** since `pw1`, so **banked pose levers are UNDER-priced, not stale.** Unlike `W`, it moves with every improvement. A 1% relative cut in `d_pose` is worth **933 B** at today's point — and *more* tomorrow. |

### 4.1 — **Three of the four named allocations have ZERO measured prices. That relocates the defect.**

This is the sharpest thing §4 found, and it inverts the shape of the question. `c1`/EV2 allocated
**100% unallocated**; `rd1` has **162/162 null duals**; `ms2r`/`r3` executed **zero rungs** with a
**null knee**. **None of them can be "re-priced," because none of them was ever priced.**

> **Therefore the operator's §1 defect is CONFINED to `wr1`/`#766` — because `wr1` is the only
> allocator in the named scope that ever actually assigned bytes using a per-channel price.** And
> that is precisely the one `rs2` proved is corrupted on **both** legs (`ρ = 0.513`; support 24.2×).
> The defect is narrower than feared and it is already located, measured, and has a built replacement.

**This is the `[[VACUITY == PASS]]` genus applied to allocators.** Three sophisticated typed
allocators sit in the corpus reading like live machinery; their measured-rung denominators are
`0`, `0`, and `0`. Reporting them as "waterfill surfaces" without their denominator is the same
silent-instrument failure as a skipped gate emitting `PASS`.

### 4.2 — The operator's §2 is ALREADY a typed field in the `rd1` schema

`rd1_162_dual_backfill.json` carries **`scorer_visibility ∈ {ker(A)-invisible, seg-visible,
pose-visible}`** per cell, and pairs `ker(A)-invisible` with
`source_metric_status = STRUCTURAL_ZERO_SCORER_EFFECT_NO_TYPED_RATE_HOME`. **That is exactly the
dimension tax, correctly instantiated:** a cell with zero scorer effect buys zero flips for any
bytes ⇒ its price is `+∞` ⇒ non-actionable. The null `λ` on those cells is not a gap; it is the
right answer.

**Caveat, stated because it is the trap:** the split is **54 / 54 / 54** and the strata are
**27 × 6** and the temporal classes **54 × 3** — a **preregistered factorial grid**, not a sampled
population. **"33% of cells are `ker(A)`-invisible" is a DESIGN count, not a measured nullity.**
Anyone quoting it as a population fraction has re-created the prefix-vs-population error
(`bp2`). The measured nullities are the four in §3.1's table; this schema field is the *vocabulary*,
not the *measurement*.

**Corroboration in passing:** the same table registers
`pose_exchange_law = dS/dd_pose = 5/sqrt(10*d_pose)` — the campaign's own registered form of §1's
pose marginal, which my 40.223 and `op3`'s 31.302 both evaluate.

---

## §5 THE HONEST BOUND — how much of the ranking is actually price-sensitive?

The directive asks for the distinction that costs money: *does the ordering change, or does the
**top** of the list change?*

**The top does not change. The ordering inside two families does.**

| claim | verdict | evidence |
|---|---|---|
| The BUY list's #1 (`mf1`, 81.4×) is price-robust | **YES** | clears `W` at any realization efficiency `η > 1.23%`. **It is also the ONLY flip-denominated BUY mechanism with a measured price in the whole scope** — so "#1" is a list of one, which is itself a finding: the campaign has priced its SELL side four ways and its BUY side once. No pricing correction in this memo moves it. Its risk is **support** (`A8`, now actively contradicted), not price. |
| The SELL family's #1 ("drop zero-flip cells") is price-robust *as a strategy* | **YES** | no re-pricing makes a *nonzero*-flip cell preferable to a genuinely zero-flip one. |
| …but its **membership** is not | **NO — 29.6% wrong** | 144 of 486 cells (`rs2`, MEASURED). The strategy survives; the tranche shrinks. |
| The SELL family's *within-tranche* ordering is price-robust | **NO — half noise** | `ρ = 0.513` on the byte proxy (`rs2`, 384 exact re-encodes). |
| The description floor's absolute level | **25.7% soft** | the {description} × {denominator} 2×2, §2.2 — but it is a **slack** quantity (headroom 5.906 → 6.530 bits/gap-flip), and no decision keys off it. |
| Any inversion attributable to **mechanism price** | **NONE FOUND** | Exhaustively, within the named scope {`sx1`, `sx2`, `rz1`, `mf1`, `qd1`, `rs2`, `wr1`}: the single measured inversion (`#826`) is baseline-driven (§2.4). **Did not find a price-driven inversion in that scope.** |

**So: the defect the operator identified is real, is MEASURED, and is worth fixing — and it costs us
nothing at the top of the list today.** What it costs is *inside* the rate lever, which is the
campaign's largest byte axis: a 29.6%-overstated free tranche and a half-noise ordering within it.
`rs2` has already built the replacement; it needs the scorer slot to price it.

**And the exposure is bounded by a second, blunter fact (§4.1): three of the four named allocations
have never priced anything.** `c1`/EV2 100% unallocated · `rd1` 162/162 null duals · `ms2r`/`r3`
zero measured rungs, null knee. **`wr1` is the entire surface.** A defect that looked campaign-wide
is one allocator deep — which is good news for the fix and bad news for the shelf, because it means
the campaign's three most sophisticated allocators are not competing candidates for re-pricing;
they are **unbuilt**, and §4.1 says so with their denominators attached.

**And the sharpest thing in this memo is not a price at all.** It is that a "B/flip" figure has, in
this corpus, silently carried **five** hidden arguments — direction, denominator, kind, base, and
acting space — any one of which moves it by 7%–65×, and one of which (direction) **inverts the
verdict**. That is precisely `op3`'s genus: *argument loss, not arithmetic error*. Every number in
§2 passed every arithmetic check it was ever given.

> **The law this memo asks to bank:** a `B/flip` is inadmissible without its **five arguments**
> — `{BUY|SELL}` · `{gap|total|burn}`-flips · `{description|purchase|residual|between-archive}` ·
> base `d_seg` · acting space. A bare `B/flip` is unanchored in the same way a bare `ΔS` is.

---

## §6 SELF-REVIEW (round 1) — three attacks on my own conclusion

1. **"You claim linearity kills the waterfill — did you check the objective actually used?"**
   The score `S = 100·d_seg + √(10·d_pose) + 25·B/DEN` is linear in `d_seg` and `B` by inspection of
   `upstream/evaluate.py`'s form, and `d_seg = flips/PX` exactly. **But** the allocators do not
   optimize `S` directly — they spend a *fixed byte budget*, which is a knapsack, and a knapsack's
   correct greedy needs per-channel ratios. **Both statements are true and they are not in
   conflict**; §4 keeps them apart deliberately. I flag that a reader who takes only §0.1 would
   wrongly conclude "per-channel prices don't matter."
2. **"Is `overlap = 1.000` for description-space mechanisms a measurement or an assumption?"**
   It is a **DERIVATION** (`L*` is definitionally what `d_seg` reads), and it is **only true of the
   priced object**. The realization multiplier is a *separate, unmeasured* factor — which is why §3.1
   presents `mf1` as an `η`-table rather than a single number. If a reader collapses that table back
   to "0.01564", they have re-created the exact fake this memo exists to prevent.
3. **"Did you disprove the operator, or refine them?"** Refined — and I should say which. The
   operator's §1 ("waterfill may not be optimal") is **correct at the allocator level and MEASURED
   there** (`rs2`); it is *not* correct at the portfolio level, for a reason (linearity) that no
   memo had stated. The operator's §2 (dimension tax) is **correct and I measured its magnitude**;
   its practical bite today is smaller than it sounds, because the mechanisms that lead are already
   in the scorer's own coordinates — **which is itself the reason they lead.**

---

## §7 OWED, named

1. **`rz1` A2's B/flip.** The only top-tier mechanism with a measured *tax* (2.07–2.17×) and no
   measured *price*. Highest-value single number in this memo.
2. **`mf1`'s `η`** — decode `cx1`, fit per-component `δ`, measure residual. `mf1` §5.4 blocker 1,
   escalated by `dd1` §2 from due-diligence to load-bearing. Break-even 1.23%.
3. **`rs2`'s A/B ΔS** — built, byte-closed, blocked on the scorer slot (`pu2`).
4. **The four allocator price bases** (§4.1) — `c1`, `rd1`, `#869`, `ms2r`/`r3`.
5. **The contour radius-of-curvature distribution** on `lstars` n600 (`dd1` §3) — gates whether the
   normal-gauge halving (row #2) applies at the segment lengths we would actually ship.
6. **A live-best reconciliation.** This memo's charter names `S = 0.7910689` / 353,805 B; `op3`'s
   instrument (14/14 evaluator receipts) names `0.8264972` / gap `0.6543559`; `qd1` names `cx1` at
   353,808 B (3 B from the charter). **RESOLVED — they are the SAME ARCHIVE, and the entire
   difference is pose.** Decomposing both against the same seg leg:

   | | bytes (from rate) | seg | `d_pose` | pose leg | S |
   |---|---:|---:|---:|---:|---:|
   | `op3` `v4d_cx1_pj2ix2` | **353,808.001** | 0.4311790 | 0.00255143 | 0.1597320 | 0.8264972 |
   | charter (`pu2`) | 353,805 | 0.4311790 | 0.00154519 | 0.1243057 | 0.7910689 |
   | delta | **−3 B** | **0** | **1.651× better** | −0.0354263 | −0.0354283 |

   `−0.0354263` (pose) vs `−0.0354283` (total); the residual **2.0e-6 is exactly the 3-byte rate
   difference** (`3 × 25/DEN`). **Nothing is unexplained.** `op3`'s figure is the best *evaluator
   receipt*; the charter's is `pu2`'s pose improvement on the identical archive. Both are correct;
   they are not competing live-bests. *(MAIN: the resolution is that `op3`'s `live_operating_point`
   parses receipts and `pu2`'s pose result is not yet one.)*

**And this reconciliation is itself the memo's cleanest measurement of its own thesis.** Across two
baselines of the same archive:

- **`W` is IDENTICAL to 1e-12** — 1.2731082153320312, both points. Exchange rate: invariant.
- **`dS/d(d_pose)` moved 31.302 → 40.223 = 1.285×** — exactly `√1.651`, the concave term doing what
  a concave term does.

`op3` already measured this marginal had risen **1.73× since `pw1`** and concluded *"banked pose
levers are UNDER-priced, not stale."* **Composing: `1.73 × 1.285` ⇒ pose levers banked at `pw1`-era
prices are now under-priced by ≥ 2.22×, and the under-pricing COMPOUNDS every time pose improves.**
This is the one place in the campaign where the operator's "different prices break the waterfill" is
literally true — and it points at the *banked* shelf, not at a new build.
