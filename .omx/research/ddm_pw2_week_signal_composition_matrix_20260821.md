# ddm_pw2 — the week's un-consumed signal, and the composition cells nobody has tried

`date_utc: 2026-08-21` · `arm: ddm_pw2` (past-week signal harvest + composition coverage)
`axis: [local-CPU $0 exact arithmetic over MEASURED receipts]` — **never a score**
`score_claim: false` · `promotable: false` · `pointer_moved: false` · cost **$0**
No Modal, no scorer forward, no archive built, no `upstream/` or frozen-packet file touched.

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B `[contest-CUDA T4, n600]`, archive
`df7fd266…` — UNMOVED by this arm.** Everything below is MEANS.

---

## ANSWER FIRST

**1. The largest unfired delta in the whole inventory is `sr1` FO-1 — the zero-byte de-blur sign
row. Its ceiling on the CURRENT body is −0.019313 S. It has never been run, it costs $0 and
~67 minutes of local CPU, and the blocker its own author recorded ("the scorer slot is held by
pid 4832") was verified nonexistent on 2026-08-16 — five days ago.** Nothing else in the
inventory is within two orders of magnitude of it per unit of work. At a 1% recovery share it
still clears the −3.5e-6 admission bar by **55×**; at the 5% share `sr1` pre-registered as its
LIVE threshold it is **276×** the bar.

**2. The top composed candidate nobody has tried is the JOINT (DALI ∥ PyAV) carrier re-solve.**
`ddm_cpu1` MEASURED, n600, by swapping only the GT table on one forward pass, that **100.02% of
the CPU-vs-CUDA gap is the ground-truth DECODER, not the CPU**. Every carrier solve in this
lineage — `up2`, `up3`, `br1`, `jg5` — optimized against the **DALI** lineage alone. That single
choice costs **+0.030363 S of pose and +0.014605 S of seg on the contest-CPU axis**, the axis the
public leaderboard actually ranks. The carrier re-solve is the machinery that already exists, it
costs **0.0991 B/pair** (MEASURED at n=454), and it has never once been pointed at a two-lineage
objective. This cell is empty because the axis was assumed, never chosen.

**3. Four of the seven items the harvest brief listed as "banked-unfired" are STALE HEADLINES.**
`qs2` (−4.375e-6) and `re1` (−1.207e-6) were consumed by `mc35`→`mc36` and are inside the
frontier (`bu1`, 08-17, event-id-level proof). The `ra2` CPR1 rider is `rr5` and shipped in `rc2`
as −169 B. `#869`'s −113,555 B is a foreign-vehicle (IX2/fz4) projection that `gx1` row 6
measured at **+9.002e-03 — a LOSS** on n600, and `ma1` independently measured the token stream's
real remaining free-rate reservoir at **a few hundred bytes, not 77 KB**. This is the
`consumption-recorded-at-the-consuming-row-only` genus firing for the second time in five days
([[orphan_sweeps_that_do_not_write_the_store_are_the_disease_20260803]],
[[corrections_land_in_bodies_headlines_keep_the_stale_number_20260805]]).

**4. Matrix summary: 45 unordered pairs over 10 mover classes — 11 COMPOSED-with-receipt, 8
ANTAGONISTIC-measured, 26 UNFIRED.** Of the 26 unfired, **21 are unfired because at least one leg
is itself family-closed** (they are empty for a good reason). Only **5 cells are genuinely open**,
and 4 of the 5 involve a leg that has never been fired at all (`M10` de-blur, `M4×lineage`).

---

## §0 THE BASE, RE-DERIVED — the control before any table

Recomputed from components, not quoted (Catalog #877: the receipt's `final_score` reads `0.15`).

| term | value | S contribution | share |
|---|---:|---:|---:|
| `d_seg` | 0.00020139 | 0.020139 | 13.58% |
| `d_pose` | 6.37e-06 | 0.007981227975693965 | 5.38% |
| archive | 180,456 B | 0.12015824324461455 | **81.04%** |
| **S** | | **0.14827847122030852** | **residual vs pointer 0.000e+00** |

Exact exchange rates (properties of the score function — the one thing in this campaign that
transfers across vehicles, per `gx1`):

```
dS/dB     = 25 / 37,545,489        = 6.658589531221714e-07  S/B
dS/dflip  = 100 / 117,964,800      = 8.477105034722222e-07  S/flip
breakeven = 0.7854791823326633 flips/B   ==   1.273108215332031 B/flip
dS/d_pose = 5/sqrt(10*d_pose) = 626.4700137907352  →  1e-7 d_pose = 94.08 B-equivalent
```

**Rate is 81% of S.** Any lever that is not a rate lever must clear a bar set by an axis four
times its size. Conversely the two distortion legs together are only 0.028120 S — so a
distortion lever with a large *share* recovery is worth more than its axis's small share suggests
only if the recovery share is large. `M10` is the only such lever in the inventory.

**The round-trip pool, re-derived on THIS body.** `jg1` MEASURED that the stored tokens are
99.9985% identical to the DALI GT argmax (only 1,714 of 117,964,800 cells differ), so **95.9% of
the seg debt is render/re-segment loss, not stored-label error**. On rc2:

```
total scored flips        = 0.00020139 * 117,964,800 = 23,756.9
round-trip share (95.9%)  = 22,782.9 flips
round-trip ceiling in S   = 22,782.9 * 8.477105e-07 = 0.019313
```

---

## §1 THE UNFIRED-DELTA TABLE

**Body attribution is the load-bearing column.** A delta measured on a different body is a
*translation owed*, not a free credit — this campaign has been burned by cross-regime constant
transfer five times ([[cross-regime-constant-transfer-genus-finishing-stage]]); `rr5` alone moved
183 B → 169 B and `fs3` caught the same class twice inside one memo.

Lineage of bodies, newest first:
`rc2` 180,456 B → `jg5` 180,625 B → `br1` 176,429 B → `up3`/`to1` 176,420 B → `ck2`/`ck1` →
`fx1` 180,601 B → `hv1` 182,759 B → `mc36` 186,269 B → `cp135`/`e480b`.

### 1A — GENUINELY OPEN (fire-order absent or blocker discharged and ignored)

| # | mechanism | body measured on | measured / projected ΔS | translation risk to rc2 | cost to fire | blocking condition | named fire-order? |
|---|---|---|---:|---|---|---|---|
| **U1** | **`sr1` A1 zero-byte de-blur** (`m ← m + α(A⁻¹m − m)` before `F.interpolate`; A is EXACTLY tridiagonal `[0.101470, 0.797060, 0.101470]`, row sums 1.000000000, κ 2.3449) | hv1 182,759 B — but **the operator is a property of the FORMAT, not the body**: A is generated from `F.interpolate` on an identity at decode time | **ceiling −0.019313 S on rc2** (100% recovery, unreachable). 5% → **−9.657e-04 (276× bar)**; 1% → −1.931e-04 (55× bar). **SIGN UNMEASURED** | **LOW on the operator, HIGH on the flip counts.** A transfers exactly; the 33,743-flip pool does not — re-derived above as 22,783 on rc2 | **$0**, ~800 s/α × 5 α, local CPU, no renderer, no archive rebuild, no Modal | **DISCHARGED 2026-08-16** — `qw1` verified pid 4832 does not exist. Still not run 5 days later | YES, sealed: `sr1` §6 FO-1, with a positive control (α=0 MUST reproduce the base flip count) and three pre-registered bands |
| **U2** | **JOINT (DALI ∥ PyAV) carrier re-solve** — point the existing `br1`/`jg5` GN machinery at a two-lineage pose objective instead of DALI alone | the *gap* is MEASURED on jg5 by `cpu1` (single forward, GT table swapped): PyAV `d_pose 1.4701090981e-04` vs DALI `6.3658738313e-06` | CPU-axis pose leg **0.038341 vs 0.007978 = +0.030363 S**; seg leg **+0.014605 S**; total gap **+0.04496856 = 100.02% attributed to GT lineage** | **LOW** — `cpu1`'s swap is on the jg5 raws, one step (−169 B) from rc2 | ~1 n600 GN solve (`jg5`'s took hours, resumable, checkpointed) + `+45 B` splice measured at n=454 | **The contest-CPU decode wall REFUSES by 3,037.6 s** (`cpu1`: 4,369.6 s vs `[1,044, 1,332]` band). `cd1`'s corrector-port verdict is the cure and is BUILD-approved | **NO — this cell has never been named** |
| **U3** | **`dx1` −18 B CABAC re-code of the dxi section** | **jg5 180,625 B** — one −169 B step from rc2 | **−1.198546e-05 S, 3.4× the bar**, decode-identical, bit-exact control passed | **LOW** — same section, one lossless rider away | $0 build; needs a runtime-tree revision | `dx1` §6: *"fold the −18 B into the rr2 receiver revision, when the tree is being rebuilt for reasons that already justify a seal chain."* **That rebuild LANDED the same day as `rc2` and the −18 B was not folded in.** | Recommendation only — **no ledger row** |
| **U4** | **`sr1` A2 waterfilled free-band correction channel** | hv1 | **−0.000595 S at the MEASURED η 0.6235** (6.2% of the then-gap), 4,276 B; −0.008382 at η=1 | **MEDIUM** — η is a per-cell quantity never measured; `rt1` measured the *aggregate* η only | $0 after FO-1 | `sr1` §6 FO-2: *"fire only after FO-1"* | YES, sealed, gated behind U1 |
| **U5** | **`mz2`/`mp2` deep-prune FiLM keep62/50/37/25** | **e480b 183,502 B** — a semantic-section body 3 KB larger and off the ck1 row-prune lineage | rate MEASURED **−748 … −2,051 B = −4.980e-04 … −1.366e-03 S**; **pose leg NEVER SCORED** | **HIGH.** `ck2` row 5: *"superseded: keep01 (keep 1%) is already IN ck1"* — the ck1 row-prune ate part of this pool. Byte credits do not transfer | $0 local (receiver-closed, retained, parse-backed) | **An apparatus bug, not physics**: `launch_detached_process.py` wrote `.done.done` receipts the queue could not observe; class fixed `0286280f95`, **queue NEVER RESTARTED** | Ledger row `qw1_mp2_deep_prune_fd_pose_probe_and_queue_restart_20260816`, still `pending` |

**Flag per the brief (ΔS > 1e-5 with NO named fire-order): U2 and U3.** U2 is the larger by three
orders of magnitude and has never been written down anywhere as a candidate.

### 1B — STALE HEADLINES: already consumed, or measured on a vehicle that does not transfer

These were carried into this harvest as "banked-unfired". They are not.

| mechanism | claimed | what is actually true, at source | disposition |
|---|---:|---|---|
| `qs2` micro-edit | −4.374918e-06 @ +34 B | **CONSUMED.** `bu1` proved subsumption at the EVENT ID: 5 of qs2's 6 events are in `mc36` Variant C's parse-back set with *better* (joint) compensation; pair 532 was measured NET HARMFUL (−2 flips) and dropped, saving +21 B. `hv1` then carried mc36's frames at 3,510 fewer bytes | **INSIDE THE FRONTIER** |
| `re1` micro-edit | −1.206874e-06 @ 0 B | **CONSUMED.** Round-1 pair 96 and Round-2 pair 7 both in mc36 Variant C. Pair 73 was never sign-verified | **INSIDE THE FRONTIER** |
| `ra2` CPR1 inner-coder repack | "+263 B raw MEASURED … the ONLY measured unfired win on the hv1 base" | **FIRED as `rr5`.** Re-measured on the live jg5 body: **−169 B, ΔS −1.125302e-04**, three losslessness controls PASS (C1 arithmetic round-trip 27,648/27,648; C2 carrier-body byte identity; C3 real-receiver parse 9-of-10 parts byte-equal). It IS the rate half of the sixteenth pointer move | **INSIDE THE FRONTIER (rc2)** |
| `#869` adaptive per-cell token map | −113,555 B | **DOES NOT TRANSFER.** `gx1` row 6, MEASURED n600 on fz4: 113,648 B at +8.4675e-04 seg = **0.8789 flips/B = 1.12× breakeven → +9.002e-03 S, a LOSS.** `ma1` independently withdrew the "77,241 B reservoir" framing as a vacuous denominator: the real remaining token free-rate is **~400 B (hit event) + ~180 B (within-miss)**, of which 797 B and 105 B are already taken | **DEAD as a number; ledger row `tz1_…_20260804` should be closed** |
| `mz2` mixed q3/q4 | −823 B | **REFUSED on pose.** `gx1` row 9 / `mp2` advisory n600: net **+7.549e-02**. `ck2` row 4 additionally: *"superseded: ck1 already 3-bits `frame_embed` + `blocks.0.film`"* | REFUSED + superseded |
| `br1` 12-dim basis **re-orientation** | "splice-ready composable transform — was it ever spliced?" | **The re-orientation move is PROVABLY NULL** — re-mixing the basis leaves the reachable pose correction invariant to **1.9e-08** over 24 random pairs. What `br1` actually shipped is a *different* mechanism (damped Gauss-Newton multi-coordinate step replacing `up2`'s ±2 coordinate descent), and **that IS spliced**: `44e9e650…` @ 176,429 B is the body `jg5` built on | **NULL as named; the real mechanism is INSIDE THE FRONTIER** |
| `t1h` zero-byte pose re-solve headroom | −0.002970 projected | **T4-REFUSED.** `fc-01M092BSTGSHRQAS2BJ1KCVARC`: `d_pose` ROSE **6.31×** (6.886e-6 → 4.346e-5), S 0.158534 → **0.171091**. The CPU-torch accept oracle does not merely fail to transfer — it **ANTI-transfers** | **WITHDRAWN — DO NOT FIRE** |
| `up3` pose byte-close | −6.85e-5 @ 0 B | **FIRED** — thirteenth pointer move, T4-measured −6.833250614765585e-05, ΔB = 0, all four pre-registered falsifiers passed | INSIDE THE FRONTIER |
| `to1`/`ma1` tail override | S 0.156595 | **FIRED** — twelfth move, −105 B, ΔS −6.991519e-05 | INSIDE THE FRONTIER |
| `fs3` §R6 tightening | −4.453e-05 at 12.72× the bar | **FIRED as task #1176 and REFUSED.** Rate leg survived its falsifier by 1.9% (664 B credit, 5.3280 b/dropped-token, a *trend edge* not a plateau), but the same-instrument pose leg measured **+3.590433e-02 = 81× the entire rate credit**. `§T12`: the carrier-re-solve rescue needs a **696×** pose recovery where `jg5` delivered **8×** — short by 87× | REFUSED, task closed |

### 1C — FAMILY-CLOSED (do not re-open without a new precondition)

| family | closing receipt | the number that closed it |
|---|---|---|
| carrier rank / atom truncation / refit (6 treatments) | `ra3` + `rr1` | rank-4 returns 14,709 B = 102.1% of the then-bar, but the score functional misses by **1,498×–3,139×**; best realised (`ra3` trust-regioned subspace refit, the untried 2×2 cell) still **35.5×**. **Even at ZERO pose damage the rung returns only 913–1,847 B = 6.3–12.8% of the gap** — the ceiling is below the target |
| rung-4 confidence-threshold token drop, Path A and Path B | `fs2` §6, **on the live rc2-lineage body** | refused on **rate, before pose**. `-log2 p` overstates the credit ~11× moving TOWARD the argmax (real/model 0.0872 at u=7.75; **−0.0145 at u=12 — it COSTS 37 B**). Path A's agreeing-position bits are 6.3× short on their own |
| pose-only-edit actuator on the 145 dropped pairs | `fs1` §3 | median credit pair justifies **1.429 B/pair**; the cheapest encoding this vehicle has ever measured is **5.667 B/pair** — 3.97× the median budget, 1.95× the mean. Blanket move nets **+6.31e-05** |
| section-coding / byte-plane de-interleave | `bp1` → **REOPENED by `ck2`** | `bp1` closed it at −5 B on sz1's body; on ck1's row-pruned body the *same* transform is **−613 B**. Not a closed door — a calibrated knob whose sign moved with the body ([[read-closed-negatives-as-actuator-datasheets]]) |
| SSE/APM second coder stage | `fx2` §2 | LOSES in **6 of 6** formulations (+9.91 B to +139.56 B) |
| exact fixed-schema semantic re-parametrizations (dense / zero-sparse / row-dictionary / hybrid) | `mz2` | all four rebuild to **+340 B**; 0/64 tensor selectors chose sparse; 38/38 tensors receiver-required; 0/38 derivable at decode |

---

## §2 THE COMPOSITION COVERAGE MATRIX

### The ten proven mover classes

| id | class | representative receipt | in rc2? |
|---|---|---|---|
| **M1** | micro-edit compensated seg edits (sparse, joint Schur compensation) | `mc36` Variant C, −2.068040e-05 `[contest-CUDA]` | YES |
| **M2** | token-field edit waterfill (margin-saliency sites, realized acceptance) | `jg3`→`jg5`, 455 of 573 admitted | YES |
| **M3** | token DROP (confidence-threshold / over-admission tightening) | `rc4`, `fs2`, `fs3` §T | **NO — closed** |
| **M4** | carrier pose GN re-solve (damped Gauss-Newton + ±2 polish, materiality stop) | `br1` + `jg5` mixed splice, +45 B / 454 pairs | YES |
| **M5** | pose basis re-orientation / rank reduction | `br1` (null) + `ra2c`/`ra3` (closed) | orientation N/A; rank NO |
| **M6** | token-tail coder + probability-MODEL axis (causal geometry, miss-cost law) | `fx1`/`fx2` −711 B; `ma1`/`to1` −105 B | YES |
| **M7** | lossless riders (CPR1 arithmetic repack; semantic metadata split; container plane2) | `rr5` −169 B; `sz1` mechanism via `ck2` −657 B | YES |
| **M8** | native-port decode rate/wall (FreeCorrector in C) | `rr6`/`rr8`→`rc2`, 2.90× end-to-end | YES |
| **M9** | semantic-section quantization / row sparsity | `mz2`/`mp2` | partial (ck1 row-prune + 3-bit `frame_embed`) |
| **M10** | **zero-byte decode-side de-blur (`A⁻¹` pre-compensation)** | `sr1` A1 | **NEVER FIRED — no receipt of any kind** |

### The grid

Legend: **C** = COMPOSED-with-receipt · **A** = ANTAGONISTIC-measured · **U** = UNFIRED ·
**U\*** = UNFIRED **and genuinely open** (neither leg is family-closed) · `—` = diagonal.

|      | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 | M10 |
|------|----|----|----|----|----|----|----|----|----|-----|
| **M1**  | — | U | U | C | U | C | C | U | U | U\* |
| **M2**  |    | — | **A** | **C** | U | C | C | U | U | **U\*** |
| **M3**  |    |    | — | **A** | U | A | U | U | U | U |
| **M4**  |    |    |    | — | **A** | C | **A** | U | U | U\* |
| **M5**  |    |    |    |    | — | U | U | U | U | U |
| **M6**  |    |    |    |    |    | — | C | U | U | U |
| **M7**  |    |    |    |    |    |    | — | **C** | U | U |
| **M8**  |    |    |    |    |    |    |    | — | U | U |
| **M9**  |    |    |    |    |    |    |    |    | — | U |
| **M10** |    |    |    |    |    |    |    |    |    | — |

**Counts: 11 COMPOSED · 8 ANTAGONISTIC · 26 UNFIRED, of which 4 are genuinely open (U\*).**
Plus the axis cell **M4 × GT-lineage**, which the class grid cannot express and which is §1A U2.

### The cells with receipts — every non-U entry, named

| cell | verdict | the receipt, with its number |
|---|---|---|
| **M2 × M4** | **COMPOSED — and the composition is the whole sixteenth-move body** | `jg1` found the hard negative (two token edits move `d_pose` **6.5×**; the full set **387×**) AND its reversal (carrier coordinate descent recovers to **1.073×** of original at ~0 bytes). `jg5` then proved the naive compose FAILS: the `jg4` full-edit composite scored **S 0.3192**, seg bought −0.012847 while pose cost **+0.172 — a 13.4× loss**. Only the JOINT waterfill admission (455 of 573, swept over a Lagrange multiplier, every subset scored through the exact formula) lands **S 0.14838267**. **Compose ≠ stack.** |
| **M7 × M8** | **COMPOSED** | `rc2`: native FreeCorrector port × RR5 rider. Two independent full-n600 local receiver runs, both `rc=0`, both emitting jg5's raw digest `7246a4ff…`; T4 row confirms decode identity at the **TOKEN** level (`decoded_token_sha256 cc10a7b0…` matches the CPU leg). ΔS −1.125302e-04, entirely rate. Wall 513.8 s vs jg5's 1,491.6 s |
| **M1 × M4** | **COMPOSED, and it beat the naive union by 3.705×** | `mc36` did not stack two compensation objects: it fresh-Schur-solved compensation **jointly** over the composed 7-object stream. −2.068040e-05 realized vs −5.581792e-06 projected. **+34 B → +17 B; 5.67 → 2.43 B/pair; 0.941 → 2.18 flips/B** |
| **M1 × M6**, **M1 × M7**, **M2 × M6**, **M2 × M7**, **M4 × M6**, **M6 × M7** | **COMPOSED (byte-disjoint sections)** | Rate credits across disjoint archive sections are **exactly additive** — `sz1` measured **0 B interaction** between its semantic split and fx2's token stream; `jg2` measured edit costs superposing at **union/sum = 1.0258** with **exact** additivity at the archive layer (10 + 6 + 14 = 30) |
| **M2 × M3** | **ANTAGONISTIC — MEASURED twice, both directions** | `fs3` §T: dropping 137 over-admitted pairs' sites credits **664 B (5.3280 b/token)** but the *same-instrument* pose leg is **+3.590433e-02 = 81× the rate credit**, because the pair still ships an EDITED frame 1 with a carrier solved for a *different* edit set. NET **+3.579520e-02, a LOSS at 10,227× the bar** |
| **M3 × M4** | **ANTAGONISTIC — arithmetically dead** | `fs3` §T12: the rescue needs a **696×** pose recovery; `jg5`'s measured whole-set carrier re-solve delivers **8×**. Shortfall **87×** |
| **M3 × M6** | **ANTAGONISTIC** | `fs2`: the `-log2 p` model **overstates the drop credit ~11×**; at u=12 the substitution *costs* 37 B, so `rate+seg > 0` for any non-negative seg-transfer coefficient A — the refusal survives A falling to **zero** |
| **M4 × M5** | **ANTAGONISTIC / null** | `br1`: re-orienting the basis is invariant to **1.9e-08**; `jg1` §S3 prices *enlarging* it at up to **+0.008175 S** against a whole pose leg of 0.008746 — even `d_pose → 0` nets **+0.000571 S**. `ra3` closed the shrink direction at 35.5× |
| **M4 × M7** | **ANTAGONISTIC on the constants, benign on the mechanism** | `rr5`'s own prestage: the −183 B *"is proven to survive a TAIL edit but explicitly NOT proven across a carrier re-solve."* Re-measured after `jg5`'s re-solve it is **−169 B**, not −183. The rider survives; its constant does not |

### The four genuinely open cells

| cell | why it is empty | what it would test | projected ΔS |
|---|---|---|---:|
| **M2 × M10** | M10 has never been fired at all, so no composition with it exists | The 455 `jg5` edits are **PRE-DISTORTION against the α=0 renderer** (`jg1` finding 2). A1 changes that renderer. The two levers therefore compete for the SAME 22,783-flip round-trip debt, so the composition must be **strongly sub-additive** and needs a joint re-solve, not a merge | conditional on M10's sign; bounded above by −0.019313 |
| **M1 × M10** | same | Whether the sparse micro-edits survive an α>0 renderer, or whether their sign flips | small (M1 is ~0.06–0.2% of a gap per `bu1`) |
| **M4 × M10** | same | A1 changes frame 1 → the carrier goes stale on every pair, exactly the `fs3` §T failure shape. **Any A1 candidate MUST carry a carrier re-solve in the same build** | −45 B-scale cost, measured |
| **M4 × GT-lineage** | the axis was never chosen — it was inherited | Solve the carrier against a two-lineage objective instead of DALI alone | up to **−0.030363 S on the contest-CPU pose leg** |

---

## §3 TOP-5 NEXT COMPOSED CANDIDATES, ranked by (projected ΔS)/(cost + risk)

### 1 — `sr1` FO-1: the A1 de-blur SIGN row. `$0`, ~67 min local CPU.

**Projected S.** The measurement returns a flip count, not an S. Translating the pre-registered
bands onto rc2's 22,783-flip round-trip pool:

| recovery share | ΔS | S | × the −3.5e-6 bar |
|---:|---:|---:|---:|
| 1% | −1.931e-04 | 0.14808534 | 55× |
| **5% (sr1's LIVE threshold)** | **−9.657e-04** | **0.14731281** | **276×** |
| 10% | −1.931e-03 | 0.14634714 | 552× |
| 33.55% | −6.480e-03 | 0.14179886 | 1,851× |

**Why first.** Zero counted bytes (the matrices are generated deterministically from
`F.interpolate` on an identity at decode time — rule 118 clean, with PR95-family L28 as the
in-tree precedent). `α = 0` reproduces the shipped decoder exactly, so **the actuator's floor is
the status quo**: it cannot be worse than the best α found. It needs no renderer forward, no
archive rebuild, no decode rebuild and no Modal, because the post-fix camera frame can be
synthesized from the retained decode as `cam' = round(clamp(U(A⁻¹·D(cam))))`. And it is the only
lever in the entire inventory that attacks the **95.9%** of the seg debt every other lever ignores.

**Falsifier (pre-registered by `sr1`, re-scoped to rc2 by this arm).** Positive control: α=0 MUST
reproduce rc2's base flip count exactly; a miss invalidates the row. Then —
(a) flips at best α **< 22,618** (≥5% of the 22,783 round-trip pool recovered) → **LIVE**, sweep α
finer and price a 0-byte candidate; (b) flips within **±1%** of base at every α → **CLOSED as
neutral**, `verdict_scope: FORMULATION` (global linear de-blur of A on this vehicle); (c) flips
**>1% above base** at every α>0 → **CLOSED as harmful**, same scope. Carry `sr1`'s own caveat: the
synthesized realization inherits one round of uint8 noise the real decoder-side fix would not, so
a LIVE reading is **conservative** and a CLOSED reading at the ±1% band must be re-checked against
true renderer output before the family is called dead.

**Risk.** The sign is genuinely unknown, and `sr1` §1 recorded that two of its own instruments
failed on this question. `rt1` §2.6 measured 94.3% symmetric jitter, which is the mechanism by
which the answer could be exactly zero. Treat this as a **free ticket on an unmeasured sign**, and
bank the result either way — **never the ceiling** (`qw1`'s wording, and it is right).

### 2 — the JOINT (DALI ∥ PyAV) carrier re-solve.

**Projected S.** rc2's contest-CPU projection is `0.19335280 − 0.00011253 = 0.19324027`
(`cpu1`'s measured PyAV row on the jg5 bytes, less the rider's 169 B). The pose leg alone is
**0.038341** of that. If a two-lineage solve reached PyAV `d_pose ≈ 3e-5` — a **4.9×** recovery,
comfortably inside `jg5`'s MEASURED **8×** — the CPU pose leg falls to 0.017321, a **−0.021 S**
move on the axis the public leaderboard ranks.

**Why second and not first.** Three real discounts. (i) It cannot be *scored* until the
contest-CPU decode wall is cured: `cpu1` measured **4,369.6 s against a `[1,044, 1,332] s` band —
REFUSE by 3,037.6 s**, with 90.8% of it in token decode. `cd1`'s verdict is **BUILD the corrector
port** (break-even 2.03–2.22× frame B / 2.77–3.08× frame A, and the later port clears the
conservative endpoints), so the cure is already approved and specified. (ii) The codes are ONE
object: optimizing PyAV degrades DALI, so this is a trade, not a free win. (iii) It costs a full
n600 GN solve.

**Falsifier.** Register before the solve: if a PyAV-targeted re-solve cannot reach PyAV
`d_pose ≤ 6e-5` **without** pushing DALI `d_pose` above **1.3e-5** (the point at which the CUDA
pose leg's growth eats the CPU gain in a single shipped object), the joint objective is dominated
and the honest finding is that the two axes demand different carriers — which the contest does not
permit. Report both legs, never their sum: the pose sensitivity `5/sqrt(10·d_pose)` is **626.47 on the
DALI axis against 130.41 on PyAV — a 4.80× difference**, so the addends are unequal by
construction and must never be folded into one figure.

### 3 — fold `dx1`'s −18 B CABAC dxi re-code into the native runtime.

**Projected S: 0.14826648575915233 (ΔS −1.198546e-05, 3.4× the bar).** Decode-identical, bit-exact
control already passed, $0 to build. Its author's stated blocking condition — *"when the tree is
being rebuilt for reasons that already justify a seal chain"* — was **satisfied by `rr6`/`rc2` on
the same day** and nobody folded it. It is 0.018% of the archive; it is on this list purely
because the cost is near zero and the condition is already met.

**Falsifier.** If the re-code does not reproduce the decoded token SHA `cc10a7b0…` byte-for-byte
through the composed receiver, refuse — this is the `rr2` #1096 desync shape (CPU-prob encode vs
CUDA-prob decode produced **S 27.83**, not identity), and the composed tree must prove it on its
own axis, not inherit it.

### 4 — A1 × token-edit JOINT re-solve (conditional on #1 returning LIVE).

**Projected S: not quotable.** The two levers share the debt, so a merge would double-count. What
is quotable is the *shape*: `jg5` measured that the naive compose of two separately-finished
candidates costs **13.4× more pose than it buys seg** (S 0.3192), while the joint admission lands
0.14838. The same law applies here, and A1 additionally invalidates every carrier code in the
build (`fs3` §T's stale-carrier failure at **81× the rate credit**). **Any A1 candidate must
therefore carry: (a) a re-run of the `jg3` edit configuration sweep against the α>0 renderer, and
(b) a full carrier re-solve — in ONE build.**

**Falsifier.** If the joint solve's realized seg gain is less than the *max* of the two levers
measured alone, the levers are pure substitutes and only the better one ships. Price the edits at
a **REAL re-encode**, never at the `-log2 p` model, and never off a set average — `fs3` §R8: a
flat prior fitted to a set's average is simultaneously **8.50% too dear for the set and 44% too
cheap for its margin**, and correcting only the first admits exactly what the second forbids.

### 5 — restart the `mp2` deep-prune queue and measure the pose leg it never measured.

**Projected S: rate-only −4.980e-04 (keep62) … −1.365677e-03 (keep25); pose leg UNKNOWN and
projects WRONG-SIGN.** Listed fifth and honestly: the only monotone measured pair (keep87 −130 B
pose +0.044296 → keep75 −471 B pose +0.041366, slope −8.59e-6 S/B) extrapolates to **net +0.0264 S
— the wrong sign by 2.8× the gap**, needing a 20× further pose reduction.

**Why it is on the list at all.** The reason those four candidates were never scored is an
**apparatus bug, not physics** — `launch_detached_process.py` appended `.done` to a name already
ending in `.done`, producing `.done.done` receipts the queue could not observe. The class was
fixed (`0286280f95`) and **the queue was never restarted**. `mz2` registered "pose-null or
pose-positive FiLM rows likely exist" as a live hypothesis, and `gx1` §E showed the pose *screen*
is linear in bytes saved, so the miss factor is a function of byte credit rather than of the edit.

**Falsifier.** Refuse if measured Δd_pose exceeds the budget `5.114e-9 × ΔB_exact` on the advisory
axis. But carry `gx1` §E(ii): **the advisory axis screens ~4.6× stricter than the CUDA axis that
ships**, so escalate to a CUDA read only if a candidate lands within 4.6× of its bar. At 139–3,191×
these stay refused under any transfer model — which is exactly why this is fifth, and why the row
should be closed by measurement rather than left `pending` for a fourth week.

---

## §4 BOUNDARIES

No `upstream/`, `submissions/exact_current/` or frozen `#1111` packet file was read for mutation
or changed. No Modal dispatch, no paid spend, no scorer forward, no archive built, no seal cut,
no lane claimed. Every number above is either quoted from a named receipt with its body and axis
attached, or is exact arithmetic over the contest score function, re-derived in this arm's own
hands (§0 reproduces the pointer with residual `0.000e+00`).

**Two unreconciled decimals, stated rather than folded away.** (i) `jg5` §5 reports **455**
admitted edits; `dx1` §5 counts **454** pairs re-solved in the same final build. Both may be
right if one admitted pair had no carrier change, but no receipt says so — worth one grep before
either count travels. (ii) `rc2`'s component sum reproduces the pointer exactly at the reported
8 dp, but `d_seg` is *carried* at 8 dp while `cpu1` measures it at 10
(`2.0134819878e-04`); the difference is below the report bound and is not material to any row
here, but a candidate whose whole margin is under 4e-8 S must not be adjudicated on the 8 dp form.

**What this arm did NOT do.** It measured nothing new about the vehicle. It fired nothing, sealed
nothing, and moved no pointer. It is an inventory and a set of pre-registered falsifiers — MEANS,
not the END. **The goal is a lower exact score, and this unit did not deliver one.**

## §5 NEXT_IF_RESUMED

**QUEUED-WITH-A-FIRE-ORDER — `sr1` FO-1 (candidate 1).** Owner: MAIN routes to an arm or runs
inline. Consumer store: a new receipt beside
`/Volumes/APDataStore/pact/ddm_sr1/` plus `.omx/state/main_hot_state.md`. Fire trigger: no local
scorer/Metal slot held (verified 08-16 and again at spawn). Action: run the α ladder
{0, 0.25, 0.5, 0.75, 1.0} with `sr1` §6's exact instrument pins — frozen CPU torch SegNet from
`upstream/models/segnet.safetensors`, **batch = 1 pair**, `torch.set_num_threads(8)`,
`SegNet.preprocess_input` verbatim — against rc2's retained decode, not hv1's; register the
rc2-scoped bands from §3 **before** any α>0 is scored; retain every per-α argmax array, not only
its flip count.

**ROUTED, not fired — U2 (candidate 2)** behind the `cd1` corrector port. **ROUTED — U3**
(candidate 3) into the next runtime-tree revision. **LEDGER HYGIENE OWED:** close
`tz1_adaptive_percell_869_joint_remeasure_20260804` (the −113,555 B premise is dead on two
independent measurements), and add the `consumed_by` back-reference `bu1` asked for to the `qs2`,
`re1` and `ra2`-CPR1 bank rows, which are still advertising **BANKED, HELD** for deltas that are
inside the frontier.

**OWN-VEHICLE FRONTIER: S 0.14827847122030852 @ 180,456 B `[contest-CUDA T4, n600]`, archive
`df7fd266…` — UNMOVED by this arm.**
