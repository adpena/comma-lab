# THE BANNER'S BINDING ARITHMETIC IS STALE BY 453.6 B — on the afr1 body, zeroing distortion CLEARS sub-0.12 with 216.35 B of margin, so "sub-0.12 REQUIRES rate-representation cuts" is FALSE as written

Date: 2026-08-31 · Author: MAIN · Cost: **$0** (exact arithmetic on the landed afr1 receipt)
Axis: no new measurement. `score_claim=false` · `promotable=false`
`verdict_scope`: **arithmetic correction to a binding instruction** — CLAUDE.md's 🎯 banner clause,
re-derived on the current pointer. Does NOT claim the distortion corner is easy; it claims the
banner's *structural exclusion* of that corner no longer holds.

---

## 1. The clause, and what it says

CLAUDE.md's always-loaded 🎯 banner:

> **BINDING ARITHMETIC (rc2 body, MEASURED): zeroing ALL distortion leaves S = rate 0.120158 > 0.12
> — sub-0.12 is unreachable by distortion work alone; it REQUIRES rate-representation cuts**

The premise is labelled with its own scope — **rc2 body** — and it was true there. Back-solving the
archive it describes:

```
rc2 implied archive = 0.120158 × 37,545,489 / 25 = 180,455.6 B
afr1 archive (live)                             = 180,002.0 B
                                            delta =     453.6 B
```

**Twenty-three pointer moves have taken 453.6 B off the body since that clause was written**, and the
clause was never re-derived. It sits 453.6 B on the wrong side of its own threshold.

## 2. The re-derivation on the live body

```
rate = 25 × 180,002 / 37,545,489 = 0.11985594327989708      ← this IS S at zero distortion
0.12 − 0.11985594327989708      = 0.00014405672010291137    ← a real, positive distortion budget
B_max at zero distortion        = 0.12 × 37,545,489 / 25 = 180,218.347 B
margin                          = 180,218.347 − 180,002 = 216.347 B
```

**At zero distortion the current archive is already 216.35 B UNDER the sub-0.12 threshold.** The
banner's conclusion — *"unreachable by distortion work alone"* — is false on this body. Cross-checked
two ways: the 0.00014405672010291137 S budget converts at the campaign exchange rate
6.658589531221714e-7 S/B to **216.35 B**, matching the direct byte arithmetic.

## 3. What this does and does NOT mean

It removes a **structural exclusion**, not a difficulty. The two corners, priced honestly:

| corner | what must move | measured requirement |
|---|---|---|
| **rate** (the one 6 swarms worked) | archive, distortion held at 0.028120 | **−42,016 B** = 23.3% of the archive |
| **distortion** (excluded by the stale clause) | d_seg + d_pose, bytes held at 180,002 | **195.2× reduction** (0.028120 → 0.000144) |

Both are hard. **Neither is closed by the other, and the banner closed one of them by arithmetic that
expired 453.6 B ago.** That matters more than the difficulty ranking, because CLAUDE.md is loaded into
every unit: the clause has been telling every arm for weeks that only one axis can work.

The genus is [[m106]] — a stale headline surviving a corrected body — with the aggravating feature
that this headline lives in the always-loaded instructions, so its blast radius is every future unit.
It is also exactly [[m124]]'s own warning ("the floor you divide by decides the answer") fired at the
campaign's top-level threshold rather than at a sub-measurement.

**Already-correct sibling:** memory [[m124]] records *"DEMAND READS TWO WAYS: 42,382 B at fixed
distortion OR 150 B at ZERO; price levers in BOTH."* That memory had the structure right and its
zero-distortion figure (150 B, on the lb1 body) is superseded by afr1's **216.35 B** — improved, in
our favour, by the same −81 B move. The banner and the memory have disagreed since m124 was written;
the banner is the one that is wrong.

## 4. Honest read on the distortion corner

195.2× is a long way. What the corpus already measures against it:

- `msr1` (#1235): the manufactured seg error is **90.12% balanced two-way flow**; the zero-byte
  boundary family is CLOSED at an oracle ceiling of ONE PIXEL.
- `mst1` (#1211): **78.71%** of manufactured seg error appears at the NATIVE RENDER — R and uint8 are
  net REPAIRERS, not destroyers.
- `#1110`/`up2`/`up3`: pose has repeatedly yielded to exact solves at 0 B (up3: −6.847e-5 at zero
  bytes, zero seg).

So the free half of the distortion corner is largely swept too. But **the corner is no longer excluded
by arithmetic**, and its cheapest members are pose-side exact solves that have historically cost 0 B —
a different economics from the rate side, where every win must buy bytes.

The practical consequence is the one [[m124]] already states and the banner contradicts: **price
levers on BOTH axes at the exchange rate, and let the measurement pick.** A mixed move (shed some
bytes, cut some distortion) faces neither the full 42,016 B nor the full 195.2×.

## 5. Action

- **The banner clause needs its scope stated or its number refreshed.** It is operator-authored
  binding text in CLAUDE.md; MAIN does not silently rewrite it. Recording the correction here and in
  the hot state, and surfacing it to the operator as the one instruction-level change this unit found.
- No arm is spawned on this. `ddm_wwc1_winwin_cone_sweep` remains the live decisive question.

## 6. Denominator

Binding clauses re-derived: **1**. Found stale: **1**. Independent cross-checks of the correction:
**2** (direct byte threshold · exchange-rate conversion). New measurements run: **0**. Arms spawned:
**0**. Dollars: **0**.

The exact pointer did not move. This unit re-derived one always-loaded clause that had been steering
every arm toward a single axis, and found it expired 453.6 B ago.

`[contest-CUDA T4 n600] own-vehicle frontier: AFR1 — S=0.14797617125559104, archive=180,002 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25; sub-0.12 gap 0.027976171255591042.`
