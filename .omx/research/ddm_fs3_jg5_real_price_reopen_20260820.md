# ddm_fs3 — the reopen was real on the average price and dies on the marginal one

> **ROUND 2 VERDICT (§R), 2026-08-20.** The 38-pair reopen was built and priced for real:
> **+223 archive bytes over 300 tokens = 5.9467 bits/token**, against a falsifier of 3.5139
> registered before the encode. **NET +5.724484e-05 — a LOSS at 16.36× the bar. REFUSED on rate.**
> Round 1's 7.11× is **WITHDRAWN as a realized claim**; it was attribution-calibrated and said so.
> The re-screen's controls were perfect (38/38 pairs landed on the exact configuration the census
> predicted), so the census arithmetic was right and the PRICE TRANSFER was wrong — on the boundary
> round 1 named in its own §5. jg3's stopping rule is vindicated by measurement. The refutation
> opens a mirror worth more than the reopen was: **§R6, a −4.45e-05 tightening at 12.72× the bar.**

**Task #1173 (leg 1) + rv17 W2-F8 / E4 (leg 2)** · **date** 2026-08-20 · **arm** `ddm_fs3`
(lane `ddm_fs3_jg5_real_price_reopen`)
**Axis** rate legs **EXACT** (archive `stat` / re-encoder code bits, byte-identical controls) ·
pose leg `[macOS-CPU advisory, jg5 retained compensated arrays]` · seg leg **MEASURED** (jg3
realized repaired-cell counts through the receiver forward + frozen CPU SegNet).
`score_claim=false`, `promotion_eligible=false`. **No Modal dispatch, no scorer forward, no
frozen `#1111` packet custody touched.**
**Store** `/Volumes/APDataStore/pact/ddm_fs3/`

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B `[contest-CUDA T4, n600]`, archive
`df7fd266…` — UNMOVED by this arm.** I produced no byte-closed candidate. What I produced is a
measured reopen with a size, a measured price that settles an open erratum, and a named next build.

---

## ANSWER FIRST

**Leg 1.** The charter's premise was wrong in its numbers and right in its instinct.

1. **The 19% overcharge does not exist, and the price the charter named was never a price.**
   fs2 §7's "3.8373 realised against a 4.718 model" compares a *realised encoding* against
   `ddm_jg3`'s `LogitPrice` **RANKER**, which jg3's own docstring labels "a RANKER, not a price."
   fs2's two numbers also disagree with each other (3.8373/4.718 = 0.8133, not the 0.877 printed
   beside them); jg5's own receipt settles it at `realized_over_modelled = 0.8133320623591156`.
   The price jg3 actually charged is a flat **4.1379 bits/token**, itself measured by jg2's
   byte-identical re-encoder. Against the **MEASURED** realised **3.813767**, the overcharge is
   **1.084990× — 8.50%**, not 19% and not the charter's own 12%.
2. **jg5's waterfill was never priced by a model at all.** Its `delta_bytes` is
   `(bits_candidate − bits_control)/8` over `ddm_jg4`'s retained **per-frame re-encoder code-bit
   arrays**. Re-pricing it corrects by **−1%, in the adverse direction** (the built 455-subset cost
   180,580 B against a modelled 180,540.4). So the charter's literal reading — reprice the
   waterfill's KEEP/DROP — yields an **empty** reopen set by direction.
3. **The reopen lives one level up, at jg3's configuration sweep**, and it is **not empty**:
   **46 pairs re-select at the measured price; 38 of them are pairs jg5 actually ships.** On seg+rate,
   pose-free, that is **−3.105753e-06 S = 0.887× the −3.5e-6 bar. It FAILS.**
4. **Except that census mis-prices exactly the pairs it selects.** The 38 reopened pairs encode at a
   token-weighted **2.7357 bits/token** against a population **3.8002** — and that is neither a
   token-count confound (Pearson(tokens, bits/token) = **+0.1987**, the wrong sign) nor chance
   (token-matched permutation **z = −3.19, p = 0.0005**). Re-priced per pair the census is
   **−3.711936e-05 S = 10.6× the bar.** A 12× bracket straddling the admission bar.
5. **So I measured the marginal price instead of arguing about it.** One real `ddm_jg2_tail_reencode`
   of the shipped field with those 38 pairs' 569 tokens **held out**, against a control that
   reproduces the base stream byte-identically. Pre-registered on disk before running: 195 B if the
   attribution is right, 270 B if the population price is right. **MEASURED: 189 B — 2.6573
   bits/token.** The attribution was **2.9%** out; the population price was **30.1%** out.
   **The attribution is validated and the population price is refuted for these pairs.**
6. **The row is LIVE.** Identifying the 38 at the population price and scoring them at their measured
   marginal: **seg+rate −3.198e-05 · pose +4.585e-06 (DERIVED) · carrier +2.508e-06 (MEASURED, leg 2)
   → net −2.489e-05 S = 7.11× the admission bar.** I did **not** byte-close it, and §5 says exactly
   why: jg3 did not retain the per-site candidate gains, so materialising the reopened configurations
   needs a 38-pair re-screen. That is the next build, costed in §8.

**Leg 2 (charter addition, rv17 W2-F8).** The carrier-coefficient move is **essentially free, and
every price at source is wrong.**

| price for the jg1-class carrier move | B/pair | vs the MEASURED 0.0991 |
|---|---:|---|
| `fs1` §3 table "jg1 re-solve midpoint" | 10.500 | **106× too high** — and a coefficient COUNT, not a price |
| `up3` §5 isolated-coefficient | 27–36 | **272–363× too high** |
| `na10` stated | 0.83 | 8.4× too high |
| `na10` own internal arithmetic | 0.0083 | 11.9× too low |
| **`ddm_fs3` MEASURED, built and stat'd** | **0.0991** | — (0.0088 B/coefficient) |

Both of fs1's break-evens are cleared — the median (1.429 B/pair) by **14.4×**, the mean (2.909) by
**29.4×**. **But the reactivation trigger is still NOT met, for a reason the erratum did not reach:**
the jg1 re-solve is a **COMPENSATION** encoding, and fs1 §3's table scored it in a column of **EDIT**
encodings. The blanket-27 pose credit requires a token edit; fs1 §4 measured that a carrier move
alone shifts those 27 pairs by **≤4.35e-10**. Priced correctly the move is
`edit 5.667 + compensation 0.0991 = 5.766 B/pair` and **still loses**. ERRATUM §E's alarming
"−2.384e-05, a gain ≈6.8× the bar at 0.83 B/pair" is **refuted**: it charged the whole move at the
compensation price. **fs1 §3's verdict SURVIVES; E1 is ADJUDICATED CLOSED at a measured price.**

The durable output of leg 2 is not the close. It is that **pose compensation costs ~0.099 B/pair on
this vehicle**, which removes a charge from every future lever that needs it — including leg 1's.

---

## §1 THE TWO REJECTION SURFACES (leg 1, and why only one is censusable)

`ddm_jg3` rejects edit-class moves in two places and prices them differently.

| | Surface A — per-site inner gate | Surface B — per-configuration sweep |
|---|---|---|
| location | `ddm_jg3_joint_solve.py:695` | `ddm_jg3_joint_solve.py:807` |
| rule | `if repaired * 10.18477 <= cost: continue` | winner = `argmin` of `net_delta_S` over the separation × keep grid |
| price used | `LogitPrice` `log2(p_old/p_new)`, per move | flat `RATE_PRIOR_BITS_PER_TOKEN = 4.1379` |
| that price's own status | jg3: "a RANKER, not a price"; measured **under**-charging 2.2× (1.91 vs 4.1379) | jg2's byte-identical re-encoder measurement |
| rejected moves retained? | **NO** — bare `continue`, no record on disk | **YES** — every configuration, with tokens, measured repaired cells and net |

**Surface A cannot be censused at $0 and I did not estimate it.** Its price under-charges in
aggregate, so re-pricing it at the real rate makes that gate *stricter*, not looser — the reopen
would be empty there and the *tightening* set is the interesting one. Recorded as
`rejected_moves_retained: false`, not guessed at.

**Surface B is censused exactly**, 573 pairs and 11,654 configurations, and it is where a flat-prior
overcharge would live.

### The three controls that make the census a measurement

| control | result |
|---|---|
| **1** — every retained `net_delta_S` rebuilt from `(repaired, tokens)` and the published constants | **max abs residual 0.000e+00** — exact float reproduction; my arithmetic *is* the ledger's |
| **2** — jg3's recorded choice is present in its own sweep AND is the modelled argmin | **PASS**, 573/573 |
| **3** — chosen configurations' tokens and repaired cells sum to jg3's independently published shard totals | **PASS** — 10,900 tokens, 15,155 cells |

Control 1 at exactly zero is what licenses the whole re-pricing; without it I would be re-scoring a
ledger I had only guessed the arithmetic of.

---

## §2 THE REAL PRICE, MEASURED (leg 1)

Not asserted, and not taken from a memo. From `ddm_jg4`'s retained per-frame code-bit arrays — the
output of the re-encoder whose unedited control reproduces the shipped RC64 stream byte-identically —
divided by jg3's own measured token count:

| quantity | value |
|---|---:|
| delta bytes, edited pairs | 5,196.258 |
| delta bytes, **unedited** pairs (context bleed) | 10.277 |
| tokens changed | 10,900 |
| **MEASURED real** | **3.813767 bits/token** |
| jg3's accept price | 4.1379 |
| **overcharge** | **1.084990× (8.50%)** |

The bleed line is worth keeping: an edit in one pair moves 10.3 B of code length in frames that were
never edited. Per-pair attribution is **leaky**, and §4 is where that matters.

---

## §3 THE CENSUS (leg 1)

`experiments/ddm_fs3_jg5_real_price_reopen.py census` → `FS3_REOPEN_CENSUS.json`.

**A pair jg5 DROPPED ships base tokens and the base carrier, so re-selecting its configuration
changes no archive byte.** The census reports the split and the SHIPPING column is the only one that
means anything — counting the other 8 rows in the headline would be a phantom credit.

| price | bits/tok | reopen (all) | reopen (**ships**) | **SHIPPING ΔS seg+rate** | × bar | Δtokens |
|---|---:|---:|---:|---:|---:|---:|
| jg3 accept prior (control) | 4.1379 | 0 | 0 | 0 | 0.000 | +0 |
| **MEASURED real** | **3.8138** | 46 | **38** | **−3.105753e-06** | **0.887** | **+300** |
| jg5 realised 455-subset | 3.8373 | 29 | 23 | −2.733590e-06 | 0.781 | +156 |
| charter's assumed 0.88× | 3.6414 | 52 | 44 | −7.871298e-06 | 2.249 | +374 |
| fs2 ranker ratio 0.8133× | 3.3654 | 90 | 74 | −1.723814e-05 | 4.925 | +574 |

Two derived falsifiers, by bisection on the retained grid:

* **highest price at which any shipping pair still re-selects: 4.073946 bits/token** = 0.9845× the
  accept prior. The flat prior only had to be 1.6% wrong before it started costing configurations.
* **price at which the shipping re-selection first clears the bar: 3.797978 bits/token** = **0.9959×
  the measured real price.** The row misses by 11.3% in ΔS, which is a **0.41%** error in the token
  price. That is a knife-edge, and a knife-edge is not a verdict.

---

## §4 THE CENSUS MIS-PRICES THE PAIRS IT SELECTS (leg 1 — the finding)

Pricing every reopened pair at the population mean assumes the reopened pairs are typical. They are
not.

| population | token-weighted bits/token |
|---|---:|
| all 573 edited pairs (full-set attribution) | 3.8124 |
| jg5's shipped 455 (shipped-configuration attribution) | 3.8002 |
| **the 38 reopened pairs** | **2.7357** |

Two adversarial checks, because a cheap subpopulation is exactly what a selection effect looks like:

* **Not a token-count confound.** Pearson(tokens, bits/token) = **+0.1987**, Spearman +0.2087 — more
  tokens costs *more* per token, and the reopened pairs have *fewer* tokens (15.0 vs 19.0). The
  correlation runs the wrong way to manufacture this.
* **Not chance.** Unmatched permutation over 20,000 random 38-pair draws: **z = −4.54, p < 1e-5**.
  Token-**matched** permutation: **z = −3.19, p = 0.00050**.

Re-priced at each pair's own attributed price: **28 pairs still re-select, ΔS = −3.711936e-05 =
10.6× the bar.** And it is not fragile in the way a noisy tail would be — clipping the per-pair price
from below:

| price floor (bits/token) | ΔS seg+rate | × bar |
|---:|---:|---:|
| none | −3.712e-05 | 10.61 |
| 2.0 | −3.077e-05 | 8.79 |
| 3.0 | −1.671e-05 | 4.78 |
| 3.4 | −1.006e-05 | 2.88 |
| 3.8138 (population) | −2.448e-06 | 0.70 |

The crossover is ≈3.5 bits/token. **But 70.7% of the −3.712e-05 comes from pairs attributed below 2
bits/token**, one of them *negative* — and §2 measured that per-pair attribution leaks. So the
optimistic column is not citable either. **The seg+rate leg is bracketed [−3.11e-06, −3.71e-05],
straddling the bar by 12×, and only a real encode collapses it.**

---

## §5 THE MEASUREMENT THAT COLLAPSES THE BRACKET (leg 1)

Pre-registered before running (`FS3_PREREGISTERED_PREDICTION.json`, written first): hold the 38
reopened pairs' **569 shipped tokens** out of the 455-edit field, re-encode for real, and read the
marginal price off the archive.

* attribution right (2.7357) → saving ≈ 195 B
* population right (3.8002) → saving ≈ 270 B

**Controls first.** My field construction reproduces jg5's `tokens_changed = 8654` exactly. My
freshly built RC64 encoder reproduces `ddm_jg4`'s retained per-frame control ledger **exactly**
(78,963.87 code bytes at frame 450, against jg4's 78,963.9) — two instruments, different code paths,
same number.

**The control.** `byte_identical: true`, 109,696 B emitted, sha
`15054e5da33640bcb2e9d4589615c3b89b1312ce27fd9aa8e2a0ec0284b506f2` — the shipped base token stream,
`prefix_bytes_matching: 109696`. Its per-frame ledger hashes to `23005a8b0994b058…`, **bit-identical
to `ddm_jg4`'s retained control ledger**. Two instruments, different code paths, different days, the
same bytes.

**The candidate.** `delta_trustworthy: true`, `tokens_changed: 8085` (= 8,654 − 569, exact).

| quantity | shipped 455-edit field | 417-edit field (38 held out) | **held-out cost** |
|---|---:|---:|---:|
| token stream | 113,847 B | **113,658 B** | **189 B** |
| archive | 180,580 B | **180,391 B** | **189 B** |
| tokens | 8,654 | 8,085 | **569** |
| **bits per changed token** | 3.8373 (population) | — | **2.6573** |

| prediction | bytes | error |
|---|---:|---:|
| per-pair attribution (2.7357 b/tok) | 194.6 | **2.9%** |
| population price (3.8002 b/tok) | 270.3 | **30.1%** |
| **MEASURED** | **189** | — |

**The attribution method is validated on exactly the pairs whose price decides the row**, and the
population price — the one the census used to refuse — is the one that is 30% wrong about them. So
the bracket collapses to its optimistic end, and it does so on a measurement rather than on the
argument I could have made from the permutation test alone.

Re-scoring the 38 (identified at the population price, so the cheap price is never used to *select*
— that would be fs1 §4's population defect again):

| leg | value | authority |
|---|---:|---|
| seg + rate | **−3.198257e-05** | seg MEASURED (jg3 realized cells) · rate MEASURED (this encode) |
| pose (+300 tokens) | **+4.584744e-06** | **DERIVED** cross-sectional OLS, §8 |
| carrier (38 pairs × 0.0991 B) | **+2.507971e-06** | **MEASURED**, leg 2 §6 |
| **net** | **−2.4889853e-05** | **7.11× the −3.5e-6 bar — CLEARS** |

### The boundary this measurement does NOT cross

It prices the 38 pairs' **existing** edits, which validates or refutes the attribution **method** on
exactly those pairs. It is **not** the price of the **+300 additional** tokens the re-selection would
add — those are different, lower-ranked tokens, and jg3 did not retain the per-site `best` dict that
would let me materialise them. Materialising the reopened configurations needs a **38-pair
re-screen**. That boundary is §8's next build, and no number here is quoted as if it had been
crossed.

---

## §6 LEG 2 — THE CARRIER PRICE, MEASURED AT EVERY DENSITY

`experiments/ddm_fs3_carrier_price_density_ladder.py ladder` → `FS3_CARRIER_PRICE_LADDER.json`.

The three prices at source disagree because they are quoting **different points on one curve**:
up3's "absorbed" and "isolated" regimes are its two ends, and the blanket-27 move sits at the sparse
end while jg5's +45 B sits at the dense end. So I measured the curve. `build_archive` copies the
hpac stream, the semantic stream and the section tail verbatim, re-encodes only the carrier, and
parses the finished bytes back through the receiver before it will return them.

**Two byte-identity controls, one of them unplanned:**

| control | result |
|---|---|
| rebuild the body from its OWN codes | `f3bce5d2…` @ **180,625 B — byte-identical** to the shipped jg5 archive |
| revert **all** 454 moved pairs to br1 codes | `30d372ae…` @ **180,580 B — byte-identical to jg5's own `candidate_jg5_subset455.zip`** |

The second is an independent reproduction of a body jg5 built a day earlier, from the other
direction, and it anchors the far end of the ladder.

| pairs reverted | coefficients | archive B | Δ vs shipped | **B/pair** | B/coefficient |
|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 180,668 | −43 | −43.000 | −3.583 |
| 3 | 36 | 180,644 | −19 | −6.333 | −0.528 |
| 9 | 99 | 180,643 | −18 | −2.000 | −0.182 |
| **27** | **300** | **180,626** | **−1** | **−0.037** | **−0.003** |
| 55 | 610 | 180,634 | −9 | −0.164 | −0.015 |
| 110 | 1,236 | 180,613 | +12 | +0.109 | +0.010 |
| 227 | 2,556 | 180,656 | −31 | −0.137 | −0.012 |
| **454** | **5,119** | **180,580** | **+45** | **+0.099** | **+0.009** |

Read it honestly. **The per-rung deltas are non-monotone and of both signs, with a ±45 B spread** —
that is the brotli container search and the Rice `k` quantisation, not a per-pair price. The only
clean number is the **endpoint**, which is anchored by a byte-identity control: **+45 B over 454
pairs / 5,119 coefficients = 0.0991 B/pair, 0.0088 B/coefficient**, independently reproducing jg5's
published +45 B. At the blanket-27 density the point estimate is a **1-byte credit**, and the honest
adverse bound given the container noise is **|price| ≤ 1.7 B/pair at n = 27**.

Either way, `up3`'s 27–36 B/pair is **refuted at every measured density** by two orders of magnitude,
and `fs1`'s 10.5 is refuted by 106×.

### The adjudication of ERRATUM §E1

fs1 §3's table has a column of **edit** encodings (qs2 5.667, jg5 9.123, rc4 12.830) and the jg1 row
is a **compensation** encoding. Those are not substitutes — a compensation cost is *added to* an edit
cost, not swapped for it. The erratum's alarming scenario charged the entire blanket-27 move at the
compensation price and got a gain; priced correctly:

| reading of the actuator | cost B/pair | pose credit available | net |
|---|---:|---|---|
| edit (qs2) + carrier compensation | 5.667 + 0.099 = **5.766** | fs1 §3's credit distribution | **LOSS** (fs1's own net table, marginally worse) |
| carrier re-solve **alone** | **0.099** | **≤4.35e-10** (fs1 §4, measured) | **LOSS** — rate +1.78e-06 against ~nothing |

**Both readings lose. fs1 §3's verdict SURVIVES the erratum, and E1 is closed at a measured price
rather than a cited one.** `verdict_scope: FORMULATION` — the pose-only-edit actuator on the
unbanked pairs, now at a *measured* compensation encoding rather than a misread one.

---

## §7 LEG 2 / E4 — THE js6b RE-SCREEN, POPULATION CURED

`… rescreen` → `FS3_JS6B_RESCREEN.json`. fs1 §4 refused the pose actuator's population defect and
fs1 §5 did not apply the same cure to js6b. Applied here.

**18 of 200 rows sit on pairs jg5 never edited** — confirmed exactly. **Positive control:** at
`c = 1` and fs1's original price my implementation reproduces js6b's **0 survivors**, so the
compensated rows are interpretable.

| rate leg | c = 1 | c = 8.11338 | c = 13.7356 |
|---|---:|---:|---:|
| fs1 original 5.667 B/pair | 0 → 0 | 59 → **47** (cure removed 12) | 110 → **96** (cure removed 14) |
| compensation-free (leg 2) | 0 → 0 | 199 → **181** (cure removed 18) | 210 → **188** (cure removed 22) |

Two things follow. The cure removes 12–14 admits at fs1's own price, so §5's reopen stands but its
counts move. And at the compensation-free price the screen admits nearly every remaining row — which
says **the rate leg was doing essentially all of the refusing**, exactly as fs1 §7 predicted for this
operating point.

**No js6b row may be cited as an admit.** The cheaper column rests on a carrier-compensation price
leg 2 measured and an **EDIT**-encoding price for js6b's semantic cells that **nobody has measured**.
That is the same category error §6 just caught in fs1 §3, and I am not going to commit it one section
later.

---

## §8 WHAT SURVIVES, AND THE NEXT BUILD

**What is MEASURED and LIVE.**

| row | number | what it needs next |
|---|---|---|
| **The jg3 configuration re-selection** | 38 shipping pairs, **net −2.489e-05 S = 7.11× the bar** at the measured marginal price | a **38-pair re-screen** — the build below |
| **The carrier compensation price** | **0.0991 B/pair, 0.0088 B/coefficient**, built and stat'd, two byte-identity controls | nothing. It is measured and reusable today |
| **The attribution method** | validated to **2.9%** on the pairs that matter; population price 30.1% out | apply per-pair, never population, when the subpopulation is selected |

**What is CLOSED.**

* **ERRATUM §E1**, adjudicated at a measured price (§6). fs1 §3's verdict survives; the erratum's
  "gain at 0.83 B/pair" scenario is refuted as a compensation-vs-edit category error.
* **The charter's own premise** — there is no 19% overcharge, jg5's waterfill was never model-priced,
  and the literal reopen it named is empty by direction (§1–§2). The reopen that exists is one level
  up and was found by looking, not by assuming.

**What is OPEN and must not be cited.**

* **The +300 marginal tokens' own price.** I measured the price of the 38 pairs' *existing* edits.
  The re-selection adds *different*, lower-ranked tokens. The validated attribution makes the
  estimate credible; it does not make it measured.
* **Surface A** (jg3's per-site inner gate). Not retained, not estimated. Its price under-charges, so
  the interesting set there is a *tightening* set, not a reopen.
* **Every js6b row** (§7).

### NEXT_IF_RESUMED — SUPERSEDED BY §R; see §R7 below

*(Round 1's build order. It was executed in round 2 and the row it aimed at is refused. Retained so
the costing stays inspectable — the 78-min estimate came in at ~70 min across 4 CPU shards.)*

1. **Re-screen the 38 reopened pairs** with jg3's solver at a corrected token price so the argmin
   *is* the reopened configuration and its `accepted` coordinate list is emitted.
   `ddm_jg3_joint_solve.py solve --pair-list <the 38> --store …`. Budget: jg3 measured ~123 s/pair →
   **~78 min** of render + SegNet forwards. **Trap, named:** `RATE_PRIOR_BITS_PER_TOKEN` is read as a
   global at `:807` and `:410` but is bound as a **default argument** at `:902` — reassigning the
   module constant changes two of the three and silently leaves the third. Rebind the *function*, as
   `ddm_fs2_jg5_on_candidate.py` does for jg5's constants; do not reassign the constant.
2. **Price the result for real** with `ddm_jg2_tail_reencode` against the control this arm already
   emitted (`S1_control_600.json`, byte-identical). ~13 min.
3. **Carrier re-solve on the 38**, then splice — at a price leg 2 has already measured (~3.8 B total).
4. **Byte-close, advisory n600 via `tools/fire_local_advisory.py`, seal via
   `tools/make_candidate_seal.py`, hand the fire-order to MAIN.** No Modal from the arm.

**What a successor must NOT do.** Do not identify the reopen set at the cheap price — that applies a
price measured on 38 pairs to all 573 and is fs1 §4's population defect wearing a new hat; the census
receipt carries that row for shape only and labels it unsupported. Do not cite fs2 §7's "0.877×" or
"4.718" as a price: the first is a transcription slip and the second is a ranker mean. Do not price a
carrier move off any of the four numbers in §6's table — they are wrong by 8× to 363×; use 0.0991
B/pair.

---

## §R REROUND — MATERIALISING THE REOPEN (leg 3)

Round 1 ended with a row **projected** at 7.11× the bar and a named blocker: jg3 emits an
`accepted` coordinate list only for the *winning* configuration, so the reopened configurations did
not exist on disk. Round 2 executes the build. **The 7.11× was attribution-calibrated, not realized;
this section is where that word changes or the row closes.**

### R1 The trap, disarmed and PROVED

`RATE_PRIOR_BITS_PER_TOKEN` is read in four places in **two different ways**:

| site | how it reads the constant | reassignment cures it? |
|---|---|---|
| `:410` `LogitPrice.bits_for` fallback | module global, at call | yes |
| `:807` the configuration sweep's cost | module global, at call | yes |
| `:972` `break_even_yield` | module global, at call | yes |
| **`:902`** `project(..., bits_per_token=X)` | **default argument, at import** | **NO — silent no-op** |

So `jg3.RATE_PRIOR_BITS_PER_TOKEN = new` cures three of four and leaves the fourth quietly reporting
a projection computed at the **old** price, in a run whose entire purpose is the new one. That is the
same class `ddm_fs2_jg5_on_candidate.py` exists for.

`experiments/ddm_fs3_jg3_repriced_rescreen.py` disarms **both** classes and then **proves** the
disarm from the live module rather than assuming the writes took: it re-reads the global, re-reads
`project.__kwdefaults__['bits_per_token']`, and re-evaluates `break_even_yield()` against
`new_price / BITS_PER_SEG_CELL`. It also **AST-audits the module source** for any *other*
default-argument binding of the constant and **REFUSES** on an unaccounted one — so a future jg3 edit
that adds a fifth site cannot slip past this shim silently.

Measured disarm, recorded per shard in `DISARM_PROOF_shard*.json`:
**4.1379 → 2.657293497363796 bits/token · `break_even_yield` 0.406279 → 0.260906 · verdict
`DISARMED_AND_PROVED`.**

**What was deliberately NOT re-priced:** the per-site inner gate at `:695`, which prices each
candidate move with the `LogitPrice` ranker. That gate builds the candidate **site pool**; re-pricing
it would change every sweep entry and destroy the reproduction control below. Left alone, on purpose,
and recorded in the proof as `left_alone`.

### R2 The two controls — both PASS, and the census predicted every configuration exactly

The re-screen changed exactly one thing: the price inside the sweep's `argmin`. So two things had to
hold, and a pair failing either would have been EXCLUDED from the candidate rather than shipped on
the hope that it was close enough.

| control | what it proves | result |
|---|---|---|
| **A** — every `(separation, keep_fraction)` entry carries the same `(tokens, repaired)` as jg3's retained entry | the site pool is untouched and only the argmin moved; this is the same solver | **PASS, 0 of 38 failing** |
| **B** — the winner is the configuration `FS3_RESCREEN_PREREG.json` registered *before* the run finished | the census predicted a specific object and it arrived | **PASS, 0 of 38 failing** |

**38 of 38 pairs admitted.** And the agreement is not merely directional — the census predicted
*which configuration each pair lands on*, and every pair hit its exact `(tokens, repaired)`:

| | before (jg5 shipped) | after (reopened) | delta | **pre-registered** |
|---|---:|---:|---:|---:|
| tokens | 569 | 869 | **+300** | **+300** |
| repaired cells | 876 | 992 | **+116** | **+116** |

| shard | pairs | tokens | cells | yield (cells/token) |
|---|---:|---:|---:|---:|
| s0 | 10 | 254 | 267 | 1.0512 |
| s1 | 10 | 205 | 228 | 1.1122 |
| s2 | 9 | 214 | 263 | 1.2290 |
| s3 | 9 | 196 | 234 | 1.1939 |
| **total** | **38** | **869** | **992** | — |

**Seg leg: −9.833442e-05 S**, MEASURED — 116 realized argmax cells through the receiver's own forward
model and the frozen CPU SegNet, not a projection. Composed field
`seg_edits_reopen_composed.npz`, 639,153 B, sha `f795fd92c598b333`.

**A control on the compose path itself:** a zero-replacement round-trip reproduces jg5's shipped
455-pair field with every plane identical and 8,654 tokens against base — jg5's own published count.

### R2b A correction to round 1's pose leg, in the conservative direction

Round 1 charged pose at **+4.5847e-06** from an OLS slope fitted over jg5's 455 kept pairs. That is a
**population** slope applied to a **subpopulation selected for being pose-favourable** — all 38 sit in
jg5's KEPT set, 76.3% of them already land at or below their base pose, and their compensated
residuals **SUM to a credit of −2.32e-05**. That is the same population-transfer genus fs1 §4 refused
and §4 of this memo caught in the rate leg. It erred **conservative** — it charged a cost where the
pair-specific data shows none — but it is a transfer and it is labelled one here.

Control on the reading: reconstructing the pointer's mean `d_pose` from jg5's arrays gives
**6.365684192e-06**, matching jg5's published value exactly.

The measured prior — the compensated residual is amplitude-INDEPENDENT (Spearman(tokens, residual)
**−0.0002**; fs2 §5 measured Spearman 0.100 over a 277× damage span) — puts the leg at **0**.
**Policy: the conservative +4.5847e-06 stays the headline until the carrier re-solve MEASURES it.
The row clears at both ends of the bracket (7.11× conservative, 8.42× measured).**

### R3 The falsifier, registered before the encode

`FS3_RATE_FALSIFIER.json`, written **before** the composed field was priced:

| leg | value |
|---|---:|
| seg (116 realized cells) | **−9.833442e-05** |
| pose (conservative) | +4.584744e-06 |
| carrier (MEASURED, leg 2) | +2.507971e-06 |
| **rate budget to the bar** | **+8.774170e-05 = 131.8 B over 300 tokens** |

**FALSIFIER: the row dies if the +300 marginal tokens price above 3.5139 bits/token.**
Calibration 2.6573 (32.2% headroom); the population price 3.8138 and jg3's prior 4.1379 both kill it.

### R4 The realized row — REFUSED on rate, on the boundary round 1 named

I built the composed field and had the proven `ddm_jg2_tail_reencode` price it for real, against this
arm's own control (`S1_control_600.json`, `byte_identical: true`, 109,696 B, sha `15054e5d…`, whose
per-frame ledger is **bit-identical to `ddm_jg4`'s retained control**).

| quantity | shipped (jg5 455) | composed (38 reopened) | delta |
|---|---:|---:|---:|
| token stream | 113,847 B | **114,070 B** | **+223 B** |
| archive | 180,580 B | **180,803 B** | **+223 B — visibly LARGER** |
| tokens | 8,654 | 8,954 | +300 |
| **marginal price** | — | — | **5.9467 bits/token** |

`delta_trustworthy: true`, 618 s. The candidate archive is **bigger than the body it came from**,
which is the refusal in one number a reader can `stat` for themselves.

| leg | value | authority |
|---|---:|---|
| seg (116 realized cells) | **−9.833442e-05** | MEASURED |
| rate | **+1.484865e-04** | **MEASURED — exact archive delta** |
| pose | +4.584744e-06 | conservative (R2b) |
| carrier | +2.507971e-06 | MEASURED, leg 2 |
| **NET** | **+5.724484e-05 — a LOSS, 16.36× the bar in the wrong direction** | |

| pre-registered | value | realized | verdict |
|---|---:|---:|---|
| falsifier (row dies above) | 3.5139 b/tok | **5.9467** | **REFUTED by 1.69×** |
| calibration (average price) | 2.6573 b/tok | **5.9467** | **the marginal price is 2.24× the average** |

**The +300 marginal tokens do not price like the 569 they were added to.** Round 1's holdout measured
the **average** price of those pairs' shipped edits at 2.6573 bits/token and validated it to 2.9% —
and round 1 stated, in `§5`'s own words, that this "is **not** the price of the **+300 additional**
tokens — those are different, lower-ranked tokens." That boundary was load-bearing and it is what
broke the row.

**Why, mechanically.** jg3's greedy orders candidate sites by **gain**, so a denser configuration adds
exactly the sites it already ranked worst. Both terms degrade on the margin, in the same direction:

| | tokens | cells | yield (cells/token) |
|---|---:|---:|---:|
| the 38 pairs' **shipped** edits | 569 | 876 | **1.5395** |
| the **reopened** configurations | 869 | 992 | 1.1415 |
| **the marginal sites they add** | **300** | **116** | **0.3867 — 3.98× worse** |

That marginal yield of **0.3867** is essentially `ddm_jg1` S1e's measured exhaustion yield of
**0.390**, which `ddm_jg3`'s own docstring already declared **below** its break-even of 0.406 —
*"the stopping rule is not a refinement; it is the whole result."* **The reopen was trying to buy
precisely the sites jg3's stopping rule exists to refuse.** They looked admissible only because the
measured 8.50% overcharge moved the break-even yield from 0.406 down to 0.261, and the marginal
sites sit between the two.

**And no subset rescues it.** The marginal yield is flat across all 38 pairs (0.375–0.400, median
0.400): admitting only the best five still needs ≤4.07 bits/token against all thirty-eight's ≤3.94.
The row is all-or-nothing on the marginal price, so there is no selective candidate to fall back to.

**The falsifier fired before the encode even finished.** At 208 of 300 tokens the reopen had already
spent ~144.9 B against a total budget of 131.8 B — **over budget with 92 tokens still unpriced**, so
that even at a price of **zero** for the remainder the net is **+5.23e-06, a loss**. The refusal does
not depend on the final byte.

### R5 Disposition

**The row is REFUSED on rate. The 7.11× from round 1 is WITHDRAWN as a realized claim** — it was
attribution-calibrated, said so, and the pre-registered falsifier caught it rather than a post-hoc
rationalisation. No candidate was byte-closed, no advisory was fired, no seal was cut, and MAIN has
no fire-order from this arm. The pointer is unmoved.

`verdict_scope`: **INSTANCE** for this 38-pair composed candidate (refused at exact re-encode bytes).
**FORMULATION** for the jg3 configuration-re-selection family under real marginal prices — warranted,
not asserted, by three measurements: the marginal yield is flat across all 38 pairs so no subset
selection exists; the marginal price is stable across 208 priced tokens; and the mechanism (greedy
ordering makes marginal sites both dearer and less productive) is jg3's own measured decay.
**It is not a family kill of the paradigm** — buying seg with bytes is exactly what the shipped
pointer already does.

**What this arm VINDICATED, which is worth as much as the refusal:** jg3's stopping rule is correct
by measurement. Its flat 4.1379 prior over-charged the set by 8.50% (round 1) — but it
**UNDER**-charged the marginal sites, which really cost ~5.6. Both errors point the same way: stop
where jg3 stopped, or earlier.

### R6 The mirror — the refutation opens a tightening worth more than the reopen was

If the marginal sites cost ~5.57 and jg3 priced them at 4.1379, then **jg3 OVER-admitted at its own
margin**, and those sites can be DROPPED for a byte credit. This is the same boundary from the other
side, and the arithmetic supports it: a re-selection delta prices **only** the Δtokens — the tokens
common to both configurations cancel exactly — so the marginal price is the correct price here too.

`FS3_TIGHTENING_CENSUS.json`, $0, over jg3's retained sweep:

| marginal price | shipping pairs | Δtokens | Δcells | ΔS (seg+rate) | × bar |
|---:|---:|---:|---:|---:|---:|
| 4.1379 (jg3's prior — control) | 0 | +0 | +0 | 0 | 0.00 |
| 4.5 | 19 | −163 | −69 | −2.559e-06 | 0.73 |
| 5.0 | 30 | −300 | −131 | −1.380e-05 | 3.94 |
| **5.5722 (MEASURED marginal)** | **120** | **−818** | **−395** | **−4.453e-05** | **12.72** |
| 6.0 | 138 | −1,014 | −507 | −7.660e-05 | 21.88 |

**Named caveat, not buried:** the marginal price was measured just **below** the cut (sites a denser
configuration adds) and is applied just **above** it (sites a sparser one drops). Those are adjacent
in jg3's own greedy ranking so the transfer is short — but it **is** a transfer, of exactly the class
this memo has now caught twice, and it must be closed by a real re-encode of a tightened field before
any of it is quoted. The control at jg3's own prior returning **0 pairs** is the positive control that
the census is not manufacturing rows.

### R7 NEXT_IF_RESUMED (round 2)

1. **QUEUED-WITH-A-FIRE-ORDER — the tightening (§R6).** Re-screen the ~120 tightening pairs at a price
   ABOVE jg3's prior using this arm's proved disarm shim (`--census` pointed at
   `FS3_TIGHTENING_CENSUS.json`), compose the sparser field, and price it with one real
   `ddm_jg2_tail_reencode` against this arm's existing byte-identical control. Cost: ~120 pairs ×
   ~150 s / 4 CPU shards ≈ **75 min**, plus a 10-min encode. **Register the falsifier first**, as
   round 2 did — the tightening dies if the dropped sites price BELOW the yield they give up.
   This row drops bytes AND cells, so unlike the reopen it is not all-or-nothing: a subset that
   clears is a real candidate.
2. **DO NOT re-run the reopen at another price.** Any price below jg3's 4.1379 selects configurations
   whose marginal yield is ~0.39 — jg1's measured exhaustion yield — and the marginal price
   measurement (5.9467) refuses all of them. The flat marginal yield across all 38 pairs means no
   subset exists. This door is measured shut.
3. **REUSABLE, no further work needed:** the carrier-compensation price **0.0991 B/pair** (leg 2,
   built and stat'd, two byte-identity controls) and the disarm shim with its AST audit.
4. **The law this arm adds to the fs2 direction-dependent family** — see §R8.

### R8 The law, stated so the next arm can price without re-measuring

`ddm_fs2` established that the `-log2 p` model prices token edits at ~0.88× moving AWAY from the
model argmax and ~0.09× moving TOWARD it. This arm adds the axis fs2 could not see, because fs2 only
ever priced whole sets:

**Within the away-from-argmax direction, the price depends on WHERE IN THE GREEDY RANKING the token
sits, and the AVERAGE price of a set is not the price of its marginal member.**

| object | measured price | vs jg3's flat 4.1379 prior |
|---|---:|---:|
| a jg3-selected edit set, AVERAGE (n=10,900 tokens) | 3.8138 b/tok | prior **over**-charges by 8.50% |
| the 38 reopened pairs' shipped edits, AVERAGE (n=569) | 2.6573 b/tok | prior over-charges by 56% |
| **the MARGINAL sites a denser configuration adds (n=300)** | **5.9467 b/tok** | prior **UNDER**-charges by 44% |

The two errors point in **opposite directions**, and that is the whole trap: a flat prior fitted to a
set's average is simultaneously too dear for the set and too cheap for its margin. Correcting only
the first — which is what round 1's census did, correctly and with a validated measurement — admits
exactly the configurations the second error forbids.

**Operational rule:** never price a token-field re-selection off a set average. A re-selection delta
prices only its Δtokens, and those Δtokens are by construction the ranking's worst members. Price
them at the MARGINAL rate or measure them.

---

## §9 BOUNDARIES

No `upstream/` or protected file changed. No Modal dispatch, no paid spend, no contest-CPU or
contest-CUDA row produced, no scorer forward, no frozen `#1111` packet custody touched. The pointer
did not move and this arm did not move it.

**The strongest numbers here are the rate ones**, and they rest on controls that reproduce prior work
byte-for-byte from different directions: the census arithmetic reproduces jg3's ledger at exactly
zero residual; my RC64 encoder reproduces jg4's retained control ledger; the carrier ladder's
endpoints reproduce both jg5 archives byte-identically.

**The soft numbers, named:** the pose leg (§8) is a **DERIVED cross-sectional** OLS slope over pairs
with different token counts, which is not the same object as adding tokens to a *given* pair; the
per-pair rate attribution is **leaky** by the 10.3 B of measured context bleed; the +300 marginal
tokens' own price is **UNMEASURED**.

`verdict_scope` per claim: §1–§5 **FORMULATION** (jg3-class edit-configuration re-selection under
real prices, on the rc2-lineage body) · §4's permutation tests **INSTANCE** (these 38 pairs) ·
§6 **FORMULATION** (the jg1-class carrier-coefficient move across densities on the jg5 body) ·
§6's E1 adjudication **supersedes** fs1's cited price with a measured one · §7 **FORMULATION**
(js6b's 182 population-clean rows). None is a family kill. None is a score.

---

## §10 OBSERVABILITY SURFACE

**Inspectable per layer** — all 11,654 sweep configurations retained per pair with tokens, measured
repaired cells and both modelled and re-priced nets; every carrier-ladder rung's archive AND its
code array retained. **Decomposable per signal** — seg, rate, pose and carrier legs priced
separately and never summed into a headline without the split shown. **Diff-able across runs** —
the carrier ladder's two endpoints are byte-identity diffs against independently built prior
archives. **Queryable post-hoc** — `FS3_REOPEN_CENSUS.json` carries seven price columns; the census
re-runs at any price in seconds. **Cite-able** — every input digest is computed at read time, never
hardcoded. **Counterfactual-able** — `--rungs` re-prices the carrier at any density and
`census_at_price` at any token price; the pre-registration is on disk and was written before the
encode.

---

## §11 ARTIFACTS (ALWAYS KEEP THE PAYLOAD)

Store root `/Volumes/APDataStore/pact/ddm_fs3/`, every file with bytes + sha256 in
`FS3_RETENTION_MANIFEST.json`. **Every carrier-ladder rung was BUILT and retained, not merely
priced** — eight archives plus their code arrays, so any reader can `stat` the curve themselves.

| artifact | bytes | sha256 (first 16) |
|---|---:|---|
| `FS3_REOPEN_CENSUS.json` | 497,272 | `3a48e845af9cdcef` |
| `FS3_PREREGISTERED_PREDICTION.json` | 1,679 | `80daa5873a1beda3` |
| `FS3_CARRIER_PRICE_LADDER.json` | 6,044 | `2388d1110b3e0b10` |
| `FS3_JS6B_RESCREEN.json` | 3,367 | `23f3c425c7381fba` |
| `FS3_HOLDOUT38.json` | 629 | `a4796f1009e48c7e` |
| `reencode/retained/S1_control_600.json` (byte-identical control) | 2,086 | `04cfb87579185392` |
| `reencode/retained/S1_encode_fs3_holdout38.json` (the decisive row) | 5,864 | `b757d4d9941e2527` |
| `reencode/retained/candidate_fs3_holdout38.zip` (built, 180,391 B) | 180,391 | `94903d7daa7fc3f3` |
| **round 2** `retained/fields/seg_edits_reopen_composed.npz` | 639,153 | `f795fd92c598b333` |
| **round 2** `reencode/retained/S1_encode_fs3_reopen_composed.json` (the refusal) | 6,216 | `81735438782f4219` |
| **round 2** `reencode/retained/candidate_fs3_reopen_composed.zip` (**LARGER than its body**) | **180,803** | `6dee4ac8260bd5b6` |
| **round 2** `rescreen38/retained/seg_solve_fs3_reopen38_s{0..3}.json` + `seg_edits_*.npz` | — | 8 files, in the manifest |
| **round 2** `FS3_RESCREEN_PREREG.json` · `FS3_RATE_FALSIFIER.json` · `FS3_REOPEN_COMPOSE.json` · `FS3_POSE_PRIOR.json` · `FS3_TIGHTENING_CENSUS.json` | — | pre-registrations + verdicts |
| `carrier_ladder/retained/archive_identity.zip` | 180,625 | `f3bce5d259a08183` ← shipped jg5, reproduced |
| `carrier_ladder/retained/archive_revert_0027.zip` | 180,626 | `c980ba441b73c78e` |
| `carrier_ladder/retained/archive_revert_0454.zip` | 180,580 | `30d372aefdbc1ada` ← jg5's own body, reproduced |
| `retained/fields/seg_edits_subset417_holdout38.npz` | 585,335 | `2c2942d466874c6c` |
| `reencode/` (control + holdout, per-frame ledgers, RC64 build) | — | RC64 base `5c75e2c70b89f148` = jg2's pinned source |

Landed instruments: `experiments/ddm_fs3_jg5_real_price_reopen.py` (the census, three fail-closed
arithmetic controls, seven price columns, two derived-by-bisection falsifiers) ·
`experiments/ddm_fs3_carrier_price_density_ladder.py` (`ladder` = the density curve with two
byte-identity controls; `rescreen` = E4's population-cured js6b screen).

Own-vehicle frontier: **S 0.14827847122030852 @ 180,456 B `[contest-CUDA T4, n600]` — UNMOVED by
ddm_fs3.**
