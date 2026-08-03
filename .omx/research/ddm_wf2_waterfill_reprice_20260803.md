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
4. **The dimension tax is RETRACTED for token mechanisms — `ddm_rs2` MEASURED that `D∘U` annihilates
   nothing.** Closed-form `M = D∘U` (validated: BLAS vs `einsum` differ by **0.0**; matches the real
   receiver to **1.7e-07**) has full 196,608-dim gain range **[0.6866, 1.0283]**, cond **1.223**, and
   **0.0%** of directions attenuated below 0.5 / 0.1 / 1e-2 / 1e-3. The **80.6742%** null is `D`'s
   fraction **on the CAMERA plane**, structurally unreachable from the token lattice; our renderer
   emits into `range(U)`, where `D` is near-isometric. **What survives is a BINARY on the acting
   plane** (camera-plane carriers can use the 230,904 blind px/frame; token carriers cannot) plus one
   **constraint** cost (`rz1`'s pose-null chroma, 2.07–2.17×). §3.1 carries the retraction and the
   self-review of how I made the error — transporting a camera-plane number to mechanisms that never
   touch that plane, the manual's *borrowed number*.
   **In its place, a more useful column: HOW EACH PRICE WAS OBTAINED.** The `clip(rint())` dead zone
   is **amplitude-dependent**, and **a linearisation cannot represent a dead zone at all** — so
   gradient and geometric keys are blind to the quantiser **exactly at small amplitudes, i.e. the
   separatrix**. Both of `wr1`'s keys are that class; both exact refutations (`rs2`, `tw1`) are
   realized finite differences. **And the proxy is not cheaper:** 36.4 s (gradient, cached) vs
   388.8 s = **1.01 s/cell** exact — **both `#766` axes can be made EXACT in ≈7 minutes from on-disk
   data.** A proxy price here is a *wrong price at comparable cost*, which removes its only defence.

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

## §0.5 THE WORKED EXAMPLE — `W`-as-cost, caught live, one message after this arm was chartered

**This is the failure mode, in the wild, on the largest axis, costing an 81× verdict error.**

`ddm_pu2` measured the Road↔Lane edge at **235,148 flips** on `cx1`'s own n600 argmax (fail-closed
control rel **1.09e-06**; **508,640** flips). It then priced the repair at `235,148 × W` and the
result was relayed to the operator as **"not buyable — must come from the base representation."**

| priced at | bytes | as % of the 353,805 B archive | verdict it produces |
|---|---:|---:|---|
| **`W` = 1.2731082153** (the **exchange rate**) | **299,368.9** | **84.61%** | *"not buyable — must come from the base"* |
| **`mf1` = 0.0156417** (a **measured mechanism cost**) | **3,678.1** | **1.04%** | *the cheapest open lever on the largest axis* |

**Ratio: 81.39×.** Recomputed this session; both figures reproduce exactly.

**`W` is what a flip is WORTH, not what any mechanism CHARGES.** Pricing a repair at `W` asks *"how
many bytes would I be willing to pay?"* and then reports the answer as *"how many bytes it costs."*
Those are the same number only for a mechanism that is exactly break-even — i.e. for the **worst**
mechanism we would still accept. Using it as a cost systematically prices every repair at the
worst admissible rate, which is why it produces a **closed** verdict on an **open** lever.

**Two more measured prices show the spread is not an artifact of one lucky mechanism:**
`ddm_sx2`'s static prior — **49 B removing 57.2% of a 36,798 B term = 21,048 B saved = 429.6×
leverage**; `ddm_qd1`'s cell-drop at **32.5206 B/flip = 25.544× WORSE than `W`**. **Real mechanism
prices span 0.0156 → 32.52 B/flip = 2,079×.** A single `W` cannot stand in for any of them.

> **The rule this memo asks to bank, in one line:** *`W` may appear on the **right** side of a
> comparison (is this mechanism worth doing?) and never on the **left** (what does this cost?).*

*(Baseline note, non-material: `pu2`'s "30.46% of gap" uses `op3`'s gap 0.6543559; against this
charter's 0.6189279 the same 0.1993374 S is **32.21%**. Both correct at their own baseline — §7.6 —
and the verdict is unchanged either way.)*

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

### 2.6 — OPERATION and ADDRESS: two more required columns, both from `ddm_hs1` (`9da1e4afa1`)

**(a) Direction of operation is required, and it is NOT the same axis as §2.1's buy/sell.** `hs1`
found the reconciliation on a **single cell index**: `sx2` reads **430× leverage** and `qd1` reads
**25.5× worse**, and the resolution is not magnitude — **one ADDS a shared description, the other
REMOVES per-pair payload.** That is orthogonal to buy/sell:

| operation | spends | gains | example | price sense |
|---|---|---|---|---|
| ADD shared description | a few static bytes | many per-pair bytes | `sx2` static prior, 49 B → 21,048 B | **bytes-for-bytes**, no flip term at all |
| REMOVE per-pair payload | flips | per-pair bytes | `qd1` cell-drop, 32.52 B/flip | SELL (§2.1) |
| ADD per-pair correction | bytes | flips | `mf1` component carrier | BUY (§2.1) |

**A bytes-for-bytes re-description is not on the flip axis at all** — quoting it as a "B/flip
leverage" against `W` is a category error even before the sign question. `sx2`'s 430× is a
**compression ratio**; `mf1`'s 81.4× is a **flip-purchase discount**. They are not the same 400-ish.

**(b) ADDRESS cost must be separated from PAYLOAD cost, and it can DOMINATE.** `hs1` measured, at
the same payload mechanism: **static top-128 = 91.22% capture for 62 B** of address vs **per-pair
top-64 = 88.71% capture for 23,516 B** — **more capture at 379.3× less address.** So two mechanisms
with *identical payload prices* differ by 379× in what it costs to point at them, and a
single-number B/flip hides all of it. **The table needs `price = address + payload`, itemized** —
and this distinction is **price-independent**, so it survives every re-pricing in this memo.

**(c) ~~Every capture figure in (b) is an UPPER BOUND until the dimension correction is applied.~~
RETRACTED — the correction is ZERO for these mechanisms.** `hs1`'s static/per-pair address figures
are **token-lattice** quantities, and `rs2` measured `D∘U` near-isometric on the token lattice
(§3.1). `cell-mass × visible-fraction` under `D`'s 80.6742% camera-plane null **does not apply**;
these captures are **not** upper-bounds-pending-a-discount. **The 379× address ratio and the 91.22%
capture stand as measured**, and the correct caveat on them is `tw1`'s state-dependence (§4.4), not
a visibility fraction.

---

## §3 THE PRICE TABLE — with unit, direction, acting space, and measured overlap

**RANKABLE rows** (a real marginal cost of a real mechanism, direction stated):

**Required columns, final (four of them added by review, one retracted):** `unit` · **`dir.`**
(BUY/SELL, §2.1) · **`operation`** (ADD-shared / REMOVE-per-pair, §2.6a) · **`address | payload`**
(§2.6b) · **`acting plane`** (token vs camera — the *binary* that replaced the retracted per-mechanism
overlap, §3.1) · **`OBTAINED`** (realized finite difference vs linearisation, §3.1a).

| # | mechanism | receipt | recomputed price | unit | dir. | operation | acting plane | **OBTAINED** | vs `W` |
|---|---|---|---:|---|---|---|---|---|---|
| 1 | **`mf1`** Movable per-component displacement | `ddm_mf1` §5.3 | **0.0156417** | B / flip bought | **BUY** | ADD per-pair correction | **token** (no discount) | numerator DERIVED-exact bit count; denominator MEASURED debt — but its **effectiveness model is a continuum linearisation, wrong 2.37×** (`dd1` §2) | **81.39× BETTER** |
| 2 | **`sx2`** sub-pixel phase carrier, normal-gauge halved | `dd1` §3 | 1,098 B → **549 B** | B / carrier | **BUY** | ADD per-pair correction | **token** | DERIVED (DOF count: tangential `δ_t` is a gauge while `κL ≪ 4`) | not yet flip-denominated |
| 3 | **`ba31`** drop-more | `rs2` §2.2 (n600) | **0.6498** | B / flip sold | **SELL** | REMOVE per-pair payload | **token** | realized n600 | **dominated 1.959×** |
| 4 | **`gr1`** cell sweep | `rs2` §2.2 | **0.6298** | B / flip sold | **SELL** | REMOVE per-pair payload | **token** | **CORRECTED provenance:** the key `gr1_sensitivity_gabs.npy` is `(600,24,32,4)`, all 600 pairs nonzero — **the key IS n600**; the `n48` caveat applied only to `gr1`'s realized-`d_seg` rows. **Real caveat: ANCESTOR-LATTICE (pre-drop model).** Key class = **linearisation** ⇒ dead-zone-blind | **dominated 2.021×** |
| 5 | **`sx2`** static prior | `hs1` / `sx2` | 49 B → **21,048 B saved = 429.6×** | **B / B** (compression ratio) | — | **ADD shared description** | **token** | measured byte counts | **not on the flip axis** (§2.6a) |
| 6 | **`hs1`** address: static top-128 vs per-pair top-64 | `hs1` (`9da1e4afa1`) | **62 B @ 91.22%** vs **23,516 B @ 88.71%** = **379.3×** | B / address, at equal payload | — | ADD shared address | **token** | measured, n600 | **price-independent** — survives every re-pricing here |

**NOT-RANKABLE rows** (recomputed and reported, but disqualified from ranking by §2):

| mechanism | recomputed | why it cannot be ranked |
|---|---:|---|
| `sx1`/`sx2` exact-`L*` description | 0.4254 – 0.5349 B/flip (2×2, §2.2) | an **average description cost**, not a marginal purchase price; and it is a **floor**, not a mechanism |
| `rz1` §2.5 "three prices" | ONE budget, **349,668 B** | a **residual**, and *derived from* the row above — not independent (§2.3) |
| `qd1`/`gr1_cell_drop50` | 32.5206 B/flip | **between-archive** difference quotient (§2.4) |
| `rz1` A2 pose-free chroma steering | **B/flip UNMEASURED** | see §3.1 — the tax is measured, the price is not |
| `wr1` Knee A "free" tranche | claimed ∞ (zero flips) | **falsified for 144 of 486 cells** by `rs2`; the true price is finite and unmeasured |
| head rank-4 quotient (140 of 144 null) | ratio 36.0× (**not a tax** — see §3.1) | `sx1`'s own caveat: lives in **decoder feature space**; exploiting it "requires inverting the encoder … not directly a byte lever". Unaffected by the `D∘U` retraction (different, downstream operator) and still applies to **no mechanism** |

### 3.1 — The dimension tax: **RETRACTED for token mechanisms.** `ddm_rs2` measured `D∘U` and it annihilates nothing.

**I asserted a per-mechanism `p_eff = p / overlap` correction. For every token-lattice mechanism in
this table that correction is ZERO, and the retraction is MEASURED, not argued.**

`rs2` (`84367be88e`, `55a786f8db`) computed the render→scorer operator `M = D∘U` in **closed form**
(it is linear and separable) and **validated before believing** — the result looked too clean and
the matmul raised divide-by-zero flags, so: BLAS vs `einsum` differ by **0.0** (the flags were
spurious) and **`M` matches the real receiver pipeline to 1.7e-07 relative.**

| measured on `M = D∘U` | value |
|---|---|
| row / col singular values | [0.82899, 1.01417] cond **1.2234** / [0.82829, 1.01390] cond **1.2241** |
| full **196,608-dim** gain range | **[0.6866, 1.0283]** |
| directions attenuated below 0.5 / 0.1 / 1e-2 / 1e-3 | **0.0% / 0.0% / 0.0% / 0.0%** |

**`D∘U` ANNIHILATES NOTHING.** `rs2` reproduced **80.6742315%** exactly as `1 − 196,608/1,017,336`
— confirming it is **`D`'s null fraction on the CAMERA plane**, and that our renderer emits into
**`range(U)`, where `D` is near-isometric.** **The null space is real and STRUCTURALLY UNREACHABLE
FROM THE TOKEN LATTICE.** ⇒ **no invisible complement, no discount, no tax** for token carriers.

**What survives is a BINARY on the acting plane, not a per-mechanism magnitude:**

| acting plane | blind fraction usable? | mechanisms |
|---|---|---|
| **token lattice** (through `U`) | **NO — `D∘U` is near-isometric, cond 1.22** | `wr1`, `gr1`, `qd1`, `ms2r`, `mf1`'s realization, every carrier in §3 |
| **camera plane** (direct) | **YES — 230,904 px/frame blind to both scorers** | `#401` blind-coordinate exploit; `rz1` A2 |

**Every "upper bound pending the visible-fraction correction" caveat in this memo is VOID for token
mechanisms.** §2.6(c) is retracted on the same grounds. And the SegNet-head rank-4 row (4 of 144,
`sx1` §2.3) is unaffected but still applies to **no mechanism** — it is a downstream feature-space
fact requiring encoder inversion, not a property of `M`.

**Self-review on my own retracted claim:** the error was §8.5 of the manual — *the borrowed number*.
`rz1` measured `D`'s nullity **on the camera plane** and I transported it to mechanisms that never
touch that plane. `rz1`'s number was right; my *transport* of it was the fake. The tell was
available and I missed it: `rz1` R1(a) itself says `D` is *"neutral … this says where the wall is
not."* I read a neutrality result as a tax.

### 3.1a — What replaces it, and it is more useful: **how the price was OBTAINED**

`rs2`'s replacement finding is a *mechanism-price* fact, not a geometry fact: **the `clip(rint())`
dead zone is AMPLITUDE-dependent, not direction-dependent — and a linearisation cannot represent a
dead zone at all.** Neither a gradient key nor a geometric key sees the quantiser; **only DRIVE — a
realized finite difference — does.** So **any price derived from a linear surrogate is blind to the
quantiser exactly where amplitudes are small, which is the separatrix** — i.e. precisely where all
our flips live. **This is a required column.**

| price | how OBTAINED | sees the dead zone? |
|---|---|---|
| `wr1` `residual_mass` byte key | **linearisation** (Σ\|signed delta\|) | **NO** → `ρ = 0.513` vs truth |
| `wr1`/`gr1` gradient & 16×16 damage keys | **linearisation / geometry** | **NO** → support 24.2× wrong |
| `rs2` exact per-cell byte marginal | **realized finite difference** (384 exact re-encodes) | **YES** |
| `tw1` state-conditioned marginal | **realized finite difference** (real r7/SMEVR coder, 4/4 controls) | **YES** |
| `mf1` 0.0156417 | numerator DERIVED-exact (bit count); denominator MEASURED debt | n/a — but its *effectiveness* model **is** a continuum linearisation, wrong by **2.37×** (`dd1` §2) |

**And a proxy price is not even a cheaper price here.** `rs2` timed both: gradient **36.4 s** cached
per-(pair,cell,channel); **exact byte marginal 388.8 s = 1.01 s/cell**. **Both axes of `#766`'s
lexsort can be made EXACT from on-disk data in ≈7 minutes.** *A proxy price is a wrong price at
comparable cost* — which removes the only defence `wr1`'s keys had.

*(Provenance correction carried from `rs2`: `gr1_sensitivity_gabs.npy` is `(600,24,32,4)` with all
600 pairs nonzero — **the key IS n600**; the `n_pairs_realized: 48` caveat applied only to `gr1`'s
realized-`d_seg` rows. **The real caveat is different: it is an ANCESTOR-LATTICE measurement**
(pre-drop model). §3's row #4 is corrected accordingly.)*

### 3.1b — The one surviving measured efficacy, correctly scoped

`rz1` §2.1's chroma figure is **not** a `D`-nullity tax (retracted above). It is a
**subspace-restriction efficacy**: the *exactly pose-free* chroma subspace (294,912 of 1,017,336
camera dims/frame) retains only part of SegNet's discriminative direction —

**0.484** (spatial-chain proxy) / **0.460** (corpus direct gradient) ⇒ a **2.07–2.17× directional
loss**, which `rz1` itself already prices — *"a 2× directional loss to buy exact pose-freeness"*
against a **79× pose penalty** for luma edits, so **the trade still clears by ~36×.**

**This is a constraint cost, not a geometry tax:** it is what you pay for *restricting the actuator
to be exactly pose-null*, and it would exist even if `D` were the identity. It is the only measured
efficacy discount in the table, and it is the correct scoped survivor of the operator's §2 —
**paired with the acting-plane binary in §3.1, since `rz1` A2 is one of the two camera-plane
mechanisms that can still use the 230,904 blind px/frame.** Its authority is separately MEASURED and
large (isoluminant chroma at amp 32 moves `Δd_seg = 2.73e-3` at n96 = **68% of the entire 4.015e-3
seg gap**) — but **its B/flip is UNMEASURED**, so it cannot be ranked against `mf1` today. *That is
the single highest-value owed price in this memo.*

**What the retraction does NOT dissolve — the realization risk.** Rows #1 and #2 act in
**description space**: they emit a description of `L*` (a component translation; a boundary-segment
phase), and `L*` *is* the object `d_seg` reads. With `D∘U` near-isometric there is no *visibility*
question for them at all. But their price still prices the **description**, while what ships is the
**realization**, and those are different objects:

> **Corrected statement.** I claimed the dimension tax and the realization crux were one object.
> **They are not** — `rs2` killed the tax and left the crux standing alone. The realization risk is
> *not* a visibility discount and cannot be computed from any nullity; it is the empirical question
> *"can the renderer produce the described change from the token lattice?"* — **which must be
> MEASURED per carrier, and for `mf1` is exactly blocker A8.**

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
| **`c1` waterfill tables** | **ONE global scalar** — `λ_B = 25/37,545,489 = 6.658589531221713e-7` S/byte (`spec:84`), rule *"allocate until `−dD/db ≤ λ_B`"* (`spec:87`). **`λ_B` IS `W` in disguise:** `(100/PX)/(25/DEN) = 1.27310821533` B/flip | **NO — nothing to re-price.** The cost side is DERIVED-exact; the **benefit side was never measured**: `ledger:/waterfill/current_status = NO_MEASURED_KKT_FEASIBLE_ALLOCATION_CLOSES_THE_BOX`, reserves 8–11 `DERIVED_COMPUTABLE_NOT_YET_COMPUTED`. EV2's realization leaves **100% unallocated**, verdict **`FORMULATION_MISPOSED_FOR_CURRENT_C1_COMPOSITION`**. |
| **`rd1` λ-continuation frontier** | objective `counted_bytes + λ·(100·d_seg + √(10·d_pose))`, **λ swept over a 10-point ladder** | **NO — and it is the one surface STRUCTURALLY immune.** A λ-continuation *sweeps* the exchange rate instead of fixing it, so a wrong `W` cannot bias it (**my §4 prediction, now confirmed against the artifact**). Its per-cell refinement is null: `lambda_bytes_per_D_dimension = None` on **162/162** (`rd1_162_dual_backfill.json`: `actionable=0`, `lambda_measured=0`, `rung_measured=0`, `effective_quantum_D=0.0`; `metric_context_cell_count=162` — metric *context* on every cell, *price* on none). Its own note: *"valid scalarization controls, but not train-decision exchange rates."* |
| **`#869` = `ddm_tw1`** | **measured marginal bytes per cell drop, CONDITIONED ON THE CURRENT DROP STATE** | **YES — and it has ALREADY MEASURED the operator's §1. See §4.4.** |
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
`source_metric_status = STRUCTURAL_ZERO_SCORER_EFFECT_NO_TYPED_RATE_HOME`. **The PRINCIPLE is
correctly instantiated** — a cell with zero scorer effect buys zero flips for any bytes ⇒ price
`+∞` ⇒ non-actionable, so the null `λ` there is the right answer, not a gap.

**But after §3.1's retraction, the `ker(A)-invisible` LEVEL IS EMPTY ON THE TOKEN LATTICE.** `rs2`
measured `D∘U` with **0.0%** of its 196,608 directions attenuated below 1e-3 — there is no
structurally-zero-scorer-effect direction reachable from the lattice. So the schema encodes a real
distinction that, for the plane our allocators act in, **has no members**. It is correct vocabulary
over an empty class — and a schema level with no members is the sister of the vacuity trap: it
*looks* like a live typology.

**Second caveat, the arithmetic one:** the split is **54 / 54 / 54**, strata **27 × 6**, temporal
classes **54 × 3** — a **preregistered factorial grid**, not a sampled population. **"33% of cells
are `ker(A)`-invisible" is a DESIGN count, not a measured nullity.** Quoting it as a population
fraction re-creates the prefix-vs-population error (`bp2`). Between the two caveats: the field is
**vocabulary, not measurement**, and the only measurement that bears on it says the class is empty
where we allocate.

**Corroboration in passing:** the same table registers
`pose_exchange_law = dS/dd_pose = 5/sqrt(10*d_pose)` — the campaign's own registered form of §1's
pose marginal, which my 40.223 and `op3`'s 31.302 both evaluate.

### 4.4 — `#869` = `ddm_tw1`: the operator's §1 is not a hypothesis. It was MEASURED on 2026-08-01.

`.omx/research/ddm_tw1_token_waterfill_state_dependence_20260801.md` — *"the waterfill's per-unit
byte price is a FUNCTION OF STATE, not a constant (task #869)"* — plus
`experiments/ddm_tw1_token_waterfill_state_dependence.py`. It measures exactly the quantity `wr1`'s
greedy assumes away: **marginal bytes saved by dropping a fixed cell, conditioned on how many cells
are already dropped**, through the **real shipped r7/SMEVR coder**.

| drop state | `k` cells dropped | mean marginal B/cell |
|---|---:|---:|
| base | 0 | **771.8** |
| | 100 | 845.1 |
| | 300 | 849.5 |
| Knee A | 486 | 835.1 |
| Knee B | 600 | **871.2** |

- **TW1-1:** the marginal saving of a *fixed* cell **rises +13.1%** from `k=0` to Knee B, **on 52 of
  53 cells**. Per-cell range **371–1029 B** (2.77× spread across cells at one state).
- **TW1-2:** the cost is **superadditive** — joint saving exceeds the singleton sum by up to **7.0%**.
- **Controls: 4/4 pass** (557,253/557,253; **261,590/261,590 reproducing `wr1`'s independent receipt
  at k=486**; 161,835/161,835 at k=600). Denominator: **768 cells**; price samples **53 cells** (29×5
  states + 24×3 states) — a sample, and labelled as one.

**What this settles, and it is the memo's strongest result.** `wr1`'s greedy sorts once, on
`residual_mass` evaluated at `k=0`, and then drops 486–600 cells as if that price still held.
**`tw1` measured that it does not** — and both corrections point the same way:

> **The marginal saving GROWS with depth (+13.1%) and joint > sum (+7.0%) ⇒ `wr1` systematically
> UNDER-values deep drops ⇒ the true optimum is DEEPER than `wr1`'s knee, not shallower.**
> This is a **directional, actionable** conclusion, and it is opposite to what `rs2`'s support
> finding alone would suggest (which shrinks the *free* tranche). The two compose: **fewer cells are
> genuinely free than `wr1` claims, and the ones that are, are worth more than `wr1` claims.**

**So the four allocators resolve into a clean trichotomy** — and only one of them is exposed:

| | allocator | status |
|---|---|---|
| **never priced** | `c1` (benefit side unmeasured, 100% unallocated) · `ms2r`/`r3` (zero rungs, null knee) | cannot be re-priced |
| **immune by construction** | `rd1` (sweeps λ rather than fixing it) | not exposed |
| **priced, and MEASURED WRONG** | **`wr1`/`#766`** — bytes `ρ=0.513` · support **24.2×** (`rs2`) · price **state-dependent +13.1%, superadditive +7.0%** (`tw1`) | **the entire exposure** |

### 4.5 — Corpus-wide: the constant is not one constant

The repo carries **~60 distinct numeric B/flip values**. Three are worth naming because they are
*used* rather than reported: **`0.65`** is a **registered engineering GO bar** (59 hits) at
`0.5106 × W`; **`0.905`** is hard-coded as `SIDECAR_BYTES_PER_FLIP` in
`experiments/measure_symbolic_topological_partition_mdl.py:78` — an **ASSUMED** rate baked into
code with no receipt; and **`1.2727`** appears in `ddm_iv3` as a region-merge solve's own analytic
water level, **slightly but genuinely different from `W`**. Notably, **none of the four allocator
directories spells `1.2731…` at all** — they carry it only as `6.658589531221714e-07` S/byte, which
is why "we used one `W` everywhere" was not visible by grep.

### 4.3 — Why `#869` was initially unresolvable: the task-ledger split, demonstrated on this charter

MEASURED from `.omx/state/canonical_task_status.jsonl`: **42 ids, range 383–909.**
**`869 ∉` and `766 ∉`.** Both task numbers this arm was chartered against are **harness-TaskList
ids that do not exist in the store arms can read** — the known split (*harness TaskList ~911 vs repo
ledger; arms see ONLY the repo*). `#766` resolved **only because the charter also named its content
(`wr1`)**; `#869` was given as a bare id plus three words and resolved to nothing.

> **This is the bridge gap producing its predicted failure inside the very memo sent to find price
> defects.** The one-line fix is the one already banked: **cite CONTENT, never a bare id.** For
> `#869` that means an arm name or a file — with either, it would have taken one grep.
> **"id not found" ≠ "row absent"; it is a missing JOIN**, and I am reporting it as such rather than
> as a negative-existence claim about a token-by-token allocator.

---

## §5 THE HONEST BOUND — how much of the ranking is actually price-sensitive?

The directive asks for the distinction that costs money: *does the ordering change, or does the
**top** of the list change?*

**Answer: the top of the list does not change. The ORDERING inside the SELL family does, and — the
one thing that costs us — its DEPTH does.**

*(Revised after review. My first pass said "ordering only." `tw1`'s state-dependence is a
depth error, not an ordering error, and depth is a decision.)*

| claim | verdict | evidence |
|---|---|---|
| The BUY list's #1 (`mf1`, 81.4×) is price-robust | **YES** | clears `W` at any realization efficiency `η > 1.23%`. **It is also the ONLY flip-denominated BUY mechanism with a measured price in the whole scope** — so "#1" is a list of one, which is itself a finding: the campaign has priced its SELL side four ways and its BUY side once. No pricing correction in this memo moves it. Its risk is **support** (`A8`, now actively contradicted), not price. |
| The SELL family's #1 ("drop zero-flip cells") is price-robust *as a strategy* | **YES** | no re-pricing makes a *nonzero*-flip cell preferable to a genuinely zero-flip one. |
| …but its **membership** is not | **NO — 29.6% wrong** | 144 of 486 cells (`rs2`, MEASURED). The strategy survives; the tranche shrinks. |
| The SELL family's *within-tranche* ordering is price-robust | **NO — half noise** | `ρ = 0.513` on the byte proxy (`rs2`, 384 exact re-encodes). |
| The SELL family's **DEPTH** is price-robust | **NO — and this one moves the answer, not just the order** | `tw1` (#869): the marginal saving of a fixed cell **rises +13.1%** with drop depth (52/53 cells) and joint > singleton sum by **+7.0%**. `wr1` prices at `k=0` and drops 486–600. **Both corrections say the true optimum is DEEPER than `wr1`'s knee.** This is the one re-pricing in the memo that changes *what to do*, not merely *in what order*. |
| The two keys that drive `#766` see the quantiser | **NO — structurally** | `clip(rint())`'s dead zone is amplitude-dependent and **a linearisation cannot represent a dead zone at all** (`rs2`). Both `wr1` keys are linearisations, so both are blind **exactly at the separatrix**. Fixable for ≈7 min of on-disk compute (§7.0). |
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
3. **"Did you disprove the operator, or refine them?"** Split verdict, and I should say which.
   The operator's **§1 is CONFIRMED and was already MEASURED** — `tw1` (#869) measured the per-unit
   byte price is state-dependent (+13.1%, superadditive +7.0%) on `wr1`'s own lattice, and `rs2`
   measured both of `wr1`'s keys wrong. It is *not* correct at the portfolio level, for a reason
   (linearity) that no memo had stated.
   The operator's **§2 is REFUTED for token mechanisms** — `rs2` measured `D∘U` near-isometric
   (0.0% attenuated below 1e-3); the 80.67% null is camera-plane and unreachable from the lattice.
   **I initially confirmed §2 with a magnitude, which was wrong, and the error was mine, not the
   operator's:** I transported `rz1`'s camera-plane nullity onto token carriers. **The retraction is
   in §3.1 with the tell I missed.** What survives is a binary on the acting plane plus one
   constraint cost — and the replacement (`OBTAINED`: realized-vs-linearised, §3.1a) is strictly more
   useful, because it indicts the two keys that actually drive `#766`.
4. **"Is a memo that retracts its own §3.1 mid-flight still coherent?"** Only if the retraction is
   propagated, not appended. I propagated it to §0.4, §2.6(c), the §3 table's column set, §3.1b,
   and §7 — and I am naming that as the check a reader should re-run, because a partially-propagated
   retraction is worse than none: it leaves the killed claim alive in whichever section a reader
   happens to open.

---

## §7 OWED, named

0. **Make both `#766` axes EXACT — ≈7 minutes, on-disk, no scorer slot, no new build.** `rs2` timed
   it: 388.8 s for the exact per-cell byte marginal (1.01 s/cell) vs 36.4 s cached gradient. This is
   the highest ratio of *decision moved* to *cost* anywhere in this memo, and it retires **both**
   linearised keys at once.
1. **`rz1` A2's B/flip.** The only top-tier mechanism with a measured *efficacy discount*
   (2.07–2.17×, a constraint cost — **not** the retracted geometry tax) and no measured *price*.
   Highest-value single number in this memo after item 0.
2. **`mf1`'s `η`** — decode `cx1`, fit per-component `δ`, measure residual. `mf1` §5.4 blocker 1,
   escalated by `dd1` §2 from due-diligence to load-bearing. Break-even 1.23%.
3. **`rs2`'s A/B ΔS** — built, byte-closed, blocked on the scorer slot (`pu2`).
4. **`#869`'s identity** — the ONLY allocator of the four still unresolved, and it is blocked on a
   **JOIN**, not on analysis (§4.3). One arm name or file path unblocks it in one grep.
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
