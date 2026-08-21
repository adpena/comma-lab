# ddm_fs1 — pose-only-edit actuator rate-dead at MEASURED encodings; js6b's closure does not survive a compensated envelope  
**[ERRATUM 2026-08-20, §E below — rv17 wave-2 W2-F8: the jg1 10.5 B/pair row was a UNITS MISREAD (coefficient count, not a price); §3 is CLOSED at qs2 pricing but OPEN-PENDING-ADJUDICATION for the jg1 re-solve encoding. "3.97×" is the MEDIAN statistic; the blanket move's mean break-even is 1.95×.]**

**Task #1142** · **axis** `[macOS-CPU advisory, scorer-free retained-array arithmetic]` ·
`score_claim=false` · `promotion_eligible=false` · **no Modal dispatch, no scorer forward, no archive built.**
**Payloads** `/Volumes/APDataStore/pact/ddm_fs1/retained/`.

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B [contest-CUDA T4, n600], archive
`df7fd266…` — UNMOVED by this arm.** I produced no byte-closed row. Everything below is means.

---

## ANSWER FIRST

I closed one family and reopened another, both at $0, and I caught my own headline being wrong
before it shipped.

1. **The pose-only-edit actuator is RATE-DEAD.** `ddm_jg5` proved that re-solving the pose carrier
   against a pair's own edited render is usually a pose *credit*. The obvious next move is to
   propose edits on the 145 pairs that never banked that credit. It does not pay: the median credit
   pair justifies an edit encoding of **1.429 B/pair**, and the cheapest encoding this vehicle has
   ever measured is **5.667 B/pair** — **3.97× too expensive**. Applied to the 27 genuinely
   unmeasured pairs the blanket move is a **net loss of +6.31e-05 S**.
2. **`js6b`'s FORMULATION closure no longer holds.** Its 200/200 HELD verdict was measured against
   an *uncompensated* pose envelope. At the compensation this vehicle actually measures, rows clear
   js6b's own admission standard. **REOPENED**, not admitted.
3. **My own selective estimate was refuted by my own control.** I first priced a selective actuator
   at 22× the bar using the kept-pair credit distribution. That distribution is sampled from pairs
   *selected for being credits*. The 145-pair population it was applied to is 118 measured negatives
   plus 27 pairs with no measurement at all. The estimate is withdrawn; the check that killed it is
   now a first-class function.
4. **The jg5 waterfill is sound.** Control: **0** of the 118 dropped pairs is a pose credit.

Charter legs A and D were already executed by sister arms before I spawned (§5). I did not re-run them.

---

## §1 THE INSTRUMENT

Everything derives from four retained `ddm_jg5` n600 arrays, content-hashed at read time.

| input | bytes | sha256 | meaning |
|---|---:|---|---|
| `d_pose_per_pair_base_odd_frames.npy` | 4,928 | `891f8dce60b6…` | base odd frame, br1 carrier |
| `d_pose_per_pair_candidate.npy` | 4,928 | `23d16bcfbd26…` | edited odd frame, **stale** carrier |
| `d_pose_per_pair_refined_matched.npy` | 4,928 | `033d8d7379c1…` | edited odd frame, **re-solved** carrier |
| `kept_pairs.json` | 2,205 | `4afea7f76196…` | the 455 pairs the waterfill admitted |

Tool `experiments/ddm_fs1_composition_law_headroom.py`; result
`FS1_COMPOSITION_LAW_HEADROOM.json` (108,823 B, sha256
`c2f51a59e4b363094e6450f009a137140c8f8d4d1b8eaebfabc7cc78c2bbc2c3`). A determinism repeat
reproduced that sha **byte-identically**. Four derived per-pair arrays are retained beside it, each
with its own bytes and sha256.

**Reconstruction control.** The shipped mixture (kept → re-solved, dropped → base) reproduces jg5's
published `d_pose = 6.365684e-06` to 7 significant figures. The instrument is jg5's instrument.

---

## §2 THE COMPENSATION IS DERIVED, NOT TRANSFERRED

For each edited pair, `c_i = (candidate_i − base_i) / (refined_i − base_i)` — uncompensated
edit-induced pose damage over what survives the carrier re-solve.

| quantity | value |
|---|---:|
| edited pairs | 573 |
| pairs landing **at or below base** after re-solve | **355 (62.0%)** |
| pairs with residual damage | 218 |
| aggregate factor (sum/sum) | **8.1134×** |
| residual-damage factor, median | **13.7356×** |
| residual-damage factor, p25 / p75 | 2.818× / 216.6× |

The 355 below-base pairs carry non-positive compensated damage, so no finite factor describes them.
I count them separately rather than let them dominate a percentile silently.

**This is a bracket, not a constant.** It is measured on jg3-derived seg edits on the br1/rc2 body.
Carrying the number onto a different edit family is the cross-regime-constant-transfer genus that
jg5 itself named as its own fifth instance. I use it below only to test whether a closure's *margin*
survives — never to admit anything.

---

## §3 THE POSE-ONLY-EDIT ACTUATOR (FORMULATION) — CLOSED AT qs2 PRICING; jg1 ENCODING UNADJUDICATED [see ERRATUM §E]

An edit buying a per-pair pose credit `k` is worth `k · dS/dd_i` and costs `B · 25/37,545,489`.
Re-deriving the slope at the live operating point (`d_pose = 6.365684e-06`) gives
`dS/dd_i = 1.044471` — jg5 quoted 1.043784 at its own slightly different point, so I did not reuse it.

**Break-even byte budget:**

| credit percentile | per-pair credit | edit must cost ≤ |
|---|---:|---:|
| p25 | 3.346e-07 | **0.525 B/pair** |
| median | 9.112e-07 | **1.429 B/pair** |
| mean | 1.855e-06 | **2.909 B/pair** |
| p75 | 2.171e-06 | 3.405 B/pair |
| p90 | 4.353e-06 | 6.828 B/pair |

**Against every encoding this vehicle has measured:**

| encoding | B/pair | credit pairs that pay | over the median budget |
|---|---:|---:|---:|
| `qs2` cheapest measured | 5.667 | 14.1% | **3.97×** |
| `jg5` live seg edits (180,580−176,429)/455 | 9.123 | 4.8% | 6.38× |
| `jg1` re-solve midpoint | ~~10.500~~ MISREAD — see §E | — | — |
| `rc4` rung-4 reference | 12.830 | 3.4% | 8.98× |

Blanket application to the 27 genuinely unmeasured pairs, at the cheapest encoding:

| assumed credit | ΔS_pose | rate | **net** |
|---|---:|---:|---:|
| median | −2.268e-05 | +1.019e-04 | **+7.92e-05 (loss)** |
| mean | −3.876e-05 | +1.019e-04 | **+6.31e-05 (loss)** |

**Verdict: CLOSED, `verdict_scope: FORMULATION`** — the pose-only-edit actuator on the unbanked
pairs, at every edit encoding measured on this vehicle. It is not a family kill: the mechanism
(edits move pose) is real and already banked in the pointer. What is closed is *paying new bytes to
buy more of it.*

**Reactivation, named and numeric:** an edit encoding at **≤1.429 B/pair** (median) or **≤2.909
B/pair** (mean) flips the blanket case. That is a rate-side result, so the lever is the encoder, not
the solver.

---

## §4 THE CONTROL THAT KILLED MY OWN HEADLINE

My first pass priced a *selective* actuator — solve all 145 unbanked pairs, ship only the payers —
at **−7.75e-05 S, 22× the bar**. It was wrong, and the way it was wrong is the dominant
false-verdict genus.

The estimate named "145 unbanked pairs" and measured "355 pairs selected for being credits."
Measuring the unbanked population directly:

| unbanked group | n | measured state |
|---|---:|---|
| dropped | 118 | **every one is a pose COST** (median 4.412e-04). Not unknowns — measured negatives. |
| unedited | 27 | re-solve moves them by ≤4.35e-10, i.e. **no edit and no measurement exists** |

So the prior is drawn from a different population than the one it is applied to. The selective
number is **withdrawn**. Any credit assumed for these 145 pairs is a hypothesis about a *different,
milder* edit, and must be labelled one. The ceiling row for the 118 dropped pairs
(ΔS_pose −2.105e-04 against +4.453e-04 of rate = **+2.348e-04 net loss**) is reported as hypothetical
for exactly this reason.

This check is now `unbanked_population_validity()`, returns
`kept_pair_prior_is_admissible_for_unbanked: False`, and two tests pin it — including one that flips
the all-costs claim to `False` the moment a dropped pair is genuinely a credit, so the function
cannot pass by always refusing.

**Sister control that passed:** 0 of the 118 dropped pairs is a pose credit, so the jg5 waterfill
never discarded a free win.

---

## §5 THE js6b BANK IS REOPENED

`ddm_na10` row 5. js6b screened `−seg + pose_risk + 0` and held all 200 rows. Two of those three
terms were wrong in a known direction: the pose envelope was measured with a stale carrier (too
high) and rate was set to zero (too low).

**Ceiling first** (the `ra3` law): all 200 rows admitted, full seg credit, zero pose, real rate over
147 distinct pairs at `qs2`'s 5.667 B/pair → **−9.500e-04 S, 271× the −3.5e-6 bar.** The bank is not
vacuous, so screening it is worth the minutes.

**Positive control:** re-screened at `c = 1` with zero rate, my implementation reproduces js6b's
**0 survivors on both bounds**. Without that the compensated rows would be uninterpretable.

| compensation | rows admitting (lower envelope) | best net | rows admitting (upper envelope) | best net |
|---|---:|---|---:|---|
| `c = 1` (js6b's screen) | 0 | — | 0 | — |
| `c = 8.1134` (measured aggregate) | 58 | −2.994e-05 | **1** | −3.248e-06 |
| `c = 13.7356` (measured median) | 108 | −3.626e-05 | **2** | −1.182e-05 |

The minimum break-even compensation across the bank is **7.024×** on the upper envelope — *below*
the 8.1134× this vehicle measures. js6b's own admission standard was "upper bound strictly
negative," and at the measured compensation that standard is met by at least one row.

**Verdict: REOPENED, `verdict_scope: FORMULATION`** — the sealed 200-row unprojected bank on CP135.
Three honest limits keep this from being an admission: (a) the compensation factor is a cross-regime
bracket (§2); (b) js6b's seg term is an optimistic upper bound; (c) the bank sits on **CP135**, and
the live body has since taken 8,654 changed tokens through jg2's re-encode, so the proposals are not
transferable as-is. At `c = 8.1134` the single surviving row nets **−3.248e-06**, which does not even
clear the −3.5e-6 bar; only the `c = 13.7356` rows do.

**Resolving measurement:** re-derive the ≤2 surviving proposals against the rc2 token stream, then
measure d_seg and d_pose through the real receiver with a carrier re-solve in the loop. Until then
this reopens a closure; it does not create a candidate.

---

## §6 THE CHARTER'S OTHER LEGS — ALREADY EXECUTED BEFORE I SPAWNED

I validated the charter at the spawn site rather than executing it literally. Three of four legs had
moved.

| leg | charter premise | state at source | disposition |
|---|---|---|---|
| **A** DALI GT wiring into `qs1.GT_POSE` | "unwired, ~8 consumers on PyAV" | **cured** by MAIN `809199d24f` (2026-08-19); `ddm_dg1` found and fixed the dead `#351` gate (139 candidates, 0 scanned, `VACUITY==PASS`) and registered the cure target; `ddm_gt2` proved the selector already DALI-backed and drove the undeclared population to **0/11**, landed `5c60d32af3` | **CLOSED — not re-run** |
| **B** `sq2` R8 + carrier re-solve | "re-run on the rc2 body" | `SL2` already ran it (`ddm_sl2_20260805`, 32/32 stratified pairs). Its `d_seg 0.0043 → 0.0010` is **21× the live body's 2.014e-04** — an ancestor vehicle. Under the L18 ancestor law those numbers do not transfer | **RE-AIMED** onto the live body (§2–§4) |
| **C** qs3 / js6 re-screen | "$0" | `gt2` re-ran the qs3 resume (`rc=0`, `B=108 H=76 W=5`, 0 toy-bracket rows admitted). The js6b half was **not** done | **js6b DONE (§5)**; qs3 folded |
| **D** ps135b / ps1u DALI re-run | "cheap, if A landed" | superseded at stronger scope by UP2/UP3 — all 600 pairs solved against DALI, 429 improved / 0 worsened, byte-closed at 176,420 B, T4 row `S 0.15652626` | **FOLDED — no arithmetic owed** |

Eleventh-plus instance of `[[charter_recall_validation_is_apparatus_not_volition_20260816]]`.
Executing leg A literally would have re-migrated already-migrated consumers and called it a landing.

---

## §7 WHAT THIS MEANS FOR THE SEVENTEENTH MOVE

The live S decomposes as **rate 0.120159 (81.0%) · seg 0.020139 (13.6%) · pose 0.007979 (5.4%)**, and
one archive byte is 6.659e-07 S — so the −3.5e-6 admit bar is **5.26 bytes**.

That is the finding underneath both verdicts. At this operating point the distortion axes are small
and the rate axis is everything, so *any* actuator that spends bytes to buy distortion must clear a
brutal exchange rate. The pose actuator misses it by 3.97×. The js6b bank clears it only under a
compensation factor borrowed across regimes. **The next pointer move is far more likely to come from
the rate axis than from either.** `ddm_na10` row 1 — the `rc4` rung-4 token *drop*, refused on pose
517× and never re-tried with a carrier re-solve — is the same composition law pointed at rate
instead of seg, and it is the row I would fire next.

---

## §8 BOUNDARIES

No `upstream/` or protected file changed. No Modal dispatch, no scorer forward, no archive built, no
contest-CPU or contest-CUDA row produced, no frozen `#1111` packet custody touched. Every number here
is scorer-free arithmetic over retained arrays and is advisory. The pointer did not move.

`verdict_scope` per claim: §3 **FORMULATION** (pose-only-edit actuator at measured encodings); §4
**INSTANCE** (my own withdrawn selective estimate); §5 **FORMULATION** (the js6b closure, reopened).
None is a family kill.

---

## §9 OBSERVABILITY SURFACE

Inspectable per layer (per-pair arrays for base / stale / re-solved / credit, each retained) ·
decomposable per signal (seg, pose and rate legs priced separately; every break-even reported as its
own row) · diff-able across runs (determinism repeat byte-identical) · queryable post-hoc
(`FS1_COMPOSITION_LAW_HEADROOM.json`, one record per js6b proposal) · cite-able (input digests
recorded at read time, not hardcoded) · counterfactual-able (`--bytes-per-pair` re-prices the whole
screen; the compensation sweep re-runs at any factor).

---

## §10 NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — `ddm_na10` row 1, the `rc4` rung-4 token drop re-measured with a
  carrier re-solve. Owner: MAIN or a fresh arm. Trigger: immediate; it is $0 and it is on the rate
  axis, which §7 shows is where the exchange rate is favourable. It needs the jg1 re-solve against
  rc4's retained deltas over ≥60 seeded-random pairs, ratio-of-sums.
- **FOLDED** — the pose-only-edit actuator, until an edit encoding ≤1.429 B/pair exists.
- **QUEUED** — the ≤2 surviving js6b proposals, re-derived on the rc2 token stream. Low priority: the
  best clears the bar only at the median compensation factor, and that factor is a cross-regime
  bracket.
- **OWED (inherited, not mine)** — `ddm_dg1` §8 rows 1–3 and 5: widen `#351` to the seg artifacts,
  declare the 11 label-only findings, flip `#351` STRICT at live-count 0, and route callers through
  `receipt_delta` so that guard stops being an unwired cure itself.

Own-vehicle frontier: **S 0.14827847122030852 @ 180,456 B [contest-CUDA T4, n600] — UNMOVED by ddm_fs1.**

---

## §E ERRATUM (2026-08-20, from rv17 wave-2 round-1 — findings adopted, cures applied)

**E1 (W2-F8, HIGH — the jg1 price is a units misread; §3's universal quantifier fails).** The
§3 table's `jg1 re-solve midpoint 10.500 B/pair` came from truncating `ddm_na10:562` at
"moves 9–12" — but the source sentence reads "moves 9–12 already-shipped **coefficients**, which
up2 measured the Rice stream absorbing at +5 B for all 7,200 ≈ 0.83 B/pair." 10.5 is a
COEFFICIENT COUNT entered as a byte price, inside a table whose preamble claims every row is a
measured encoding. The candidate prices for that same move are mutually inconsistent at source:
na10's stated 0.83 B/pair (itself a 100× internal slip — 5 B / 7,200 over 600 pairs = 0.0083);
up3 §5's correction (~0.08 B/pair absorbed as shipped, but +3 B per ISOLATED coefficient →
~27–36 B/pair for isolated moves). At 0.83 B/pair the blanket-27 move nets **−2.384e-05 — a GAIN
≈6.8× the admission bar**; at 27–36 it stays dead. **Consequence: §3's verdict is DOWNGRADED
from "CLOSED at every edit encoding measured on this vehicle" to CLOSED AT qs2 PRICING (5.667
B/pair leg unaffected) and OPEN-PENDING-ADJUDICATION for the jg1 re-solve encoding.** The
governing price is not decidable by citation — none of the three numbers is a real re-encode of
THIS move on THIS body. The adjudicating measurement (real Rice-stream re-encode of the blanket-27
re-solved coefficients, per the fs2 direction-dependent pricing law) is assigned to **ddm_fs3
leg 2**; its result settles whether the ≤1.429 B/pair reactivation trigger is already met.

**E2 (MED — median statistic on a blanket claim).** The title's "3.97×" is the MEDIAN-budget
statistic (5.667 / 1.429). The blanket-27 move is additive, so its break-even is the MEAN budget:
5.667 / 2.909 = **1.95×**. The qs2-priced conclusion (loss at both mean and median, §3's own
net table) survives; the published magnitude is corrected everywhere headline-visible.

**E3 (MED — "family" over-label).** Title, §3 header, ANSWER-FIRST and the landing commit
message said "family" while the typed verdict correctly says `verdict_scope: FORMULATION`
(pose-only-edit actuator at measured encodings). Title and header corrected above; the commit
message (fa1c61ac64) is immutable — this erratum is its correction of record.

**E4 (MED — §5 population control not applied).** §4 refuses the pose actuator's prefix/
population defect, but §5's js6b re-screen carries the same class: 18 of 200 rows sit on pairs
never edited in jg5, and one of the two median-calibration admits is among them. §5's reopen
stands but inherits an explicit population caveat; the re-screen re-runs under fs3's real prices
before any js6b row is cited as an admit.

Review provenance: `.omx/research/ddm_rv17_wave2_review_round1c_fs1_20260820.md` (reviewer
commit 89de978499); receipt `/Volumes/APDataStore/pact/ddm_rv17/rv17_wave2_round1c_receipt.json`.
