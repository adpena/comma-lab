# ddm_na11 — re-grading the negative corpus against the rc2 operating point

- **arm** `ddm_na11` · **date** 2026-08-21
- **axis** This arm MEASURES NOTHING. It ADJUDICATES. Every number below is attributed to the
  artifact that measured it, carrying that artifact's own axis label. `score_claim=false` ·
  `promotion_eligible=false` · no Modal · no scorer forward · $0.
- **verdict_scope** `corpus-adjudication`. REOPEN means *the negative no longer binds*, never
  *the positive is proven*. Each REOPEN carries its resolving measurement and cost.
- **Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B `[contest-CUDA T4, n600]`, archive
  `df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080` — UNMOVED by this arm.**

---

## ANSWER FIRST

**The operating point re-grades more rows than all six laws combined, and it re-grades them in the
direction nobody wants: it closes the rate axis and hands the whole remaining descent to
distortion.**

At rc2 the rate term is **0.120158 — already INSIDE the 0.07–0.13 theoretical floor band.** Zeroing
both distortion legs lands S = 0.120158 without removing one archive byte. So rate is 81.0% of S and
approximately 0% of the addressable descent. Two independent measured closures agree: the coder axis
returns **−5 B** (`bp1`, byte-exact through the real coders) and the probability-model axis has
**~475 B** left (`tx1`: ~400 B hit-event + ~75 B within-miss, after fx1/fx2/ma1 took 560+711+105).
Together that is **ΔS −3.2e-4 against a gap of 0.018278 to the floor band's top — 1.7%.**

The remaining **0.028120 S** is distortion: seg 0.020139 (23,757 flips) + pose 0.007981. And the
actuator that took the last 11,997 flips is at its **measured marginal stopping point**. Per `cf2`'s
registered equation `greedy_set_average_vs_marginal_price_v1`, the shipped edit set returned 1.5395
cells/token at 2.6573 b/token — value/cost **5.90×, a large win**. The marginal member returns 0.3867
cells/token at 5.9467 b/token — value/cost **0.662, a 1.51× LOSS**. Yield fell 3.98× while price rose
2.24×: **8.91× joint degradation at the margin.** The solve is correctly stopped.

**Top three reopens, and they are all small:**

| # | row | projected ΔS `[baseline:rc2=0.14827847122030852]` | cost |
|---|---|---|---|
| **R1** | `cb1` MyCar hood — na10 REOPENED it, two audits passed, **nobody fired it** | **−2.34e-05** (6.7× bar) DERIVED via the relative-pose law | **$0, ~140 s** |
| **R2** | `fx2` 19-member build — refused on decode margin; the wall moved 2.90× | **−5.77e-05** (16.5× bar) from a MEASURED −86.58 B | rebuild + **$0.16** |
| **R3** | `qs1`/`qs5` re-price at the measured compensation cost | **−3.98e-06 / −3.27e-06** (1.14× / 0.94× bar) PROJECTION | $0 fold, no new candidate |

**Everything large in the reopen column is gone.** na10's headline reopen (`rc4` rung-4, 50% of the
then-gap) was fired and **refused twice over** — on rate by `fs2`, and its reopening rationale
demolished by `dx1`. Two of na10's ten reopens I **downgrade to STANDS** on arithmetic na10 did not
run. That leaves 17 STANDS against 6 REOPENS.

---

## §0 THE SIX LAWS, VERIFIED AT SOURCE

Per the charter-recall law ([[charter_recall_validation_is_apparatus_not_volition_20260816]]) I
re-derived each from its artifact before applying it. All six verify. Two carry refinements that
change how the audit runs.

| # | law | verified at | refinement the charter compressed away |
|---|---|---|---|
| **L1** | **Composition.** Seg edit × carrier re-solve composes; in-compile Schur compensation exists and works. | `jg1` §S2b; `jg5` law 1 | **The REACH is bimodal and its estimate spans 1,000×.** See §2. |
| **L2** | **Real re-encode prices.** `−log2 p` is DIRECTION-DEPENDENT: away from argmax real/model **0.93×**; toward argmax **0.09×**. The 0.77–0.88 figures divide by 4.718, which is jg3's `LogitPrice` **RANKER**. | `fs2` memo + memory `price-token-field-levers-by-real-reencode`; `cf2` equation `token_rate_model_direction_dependence_v1` | The two directions move verdicts **opposite ways**. See §4. |
| **L3** | **One waterfill.** Edits and drops are one joint solve; separate composition double-spends the shared compensation budget (`bu1`/`mc36`: joint beat the naive union **3.705×**). | memory `edits_and_drops_are_one_waterfill_solve_jointly_20260819` | jg5's "DROP" means *revert the edit to a byte-identical base*, **not** drop a token. Do not conflate. |
| **L4** | **GT-lineage fork**, in `pi2`/`na10`'s corrected form: `d_pose` differs by an **additive floor C = 1.4061e-04**, never a ratio; `d_seg` by 1.4425×. | `na10` §4.3 L1′; `gt_lineage_registry.json` | Already applied corpus-wide by na10. I add no new L4 reopen. |
| **L5** | **Native decode wall CLOSED.** Whole Modal job **1,491.6 s → 513.8 s = 2.90×**; harness `Wall budget: PASS [contest-CUDA] … 498.476 s charged ≤ 822 s cold-cache ceiling`. | `rc2` row memo, lines 22 / 41 / 47–51 | **The slack has a size: 323.5 s on the shipping axis.** That is the currency R2 spends. |
| **L6** | **Placement beats amount, 26×.** Learned prior returns 3.810 token-B per counted byte; best hand table 0.146. | memory `the_counted_byte_is_not_fungible_placement_beats_amount_20260816` | Its own worked example CLOSED a family by ceiling-first. It closes more rows here than it opens. |

**One law was already spent.** L4 is na10's L1′, applied across 24 verdicts on 08-19. Re-running it
would be the ninth audit on top of eight ([[m18]]). I applied it only where a row was untouched.

---

## §1 THE ARITHMETIC OF THE NEW OPERATING POINT

Recomposed from components; reproduces the given S exactly (12 decimals).

| leg | value | share | what it means for a re-grade |
|---|---:|---:|---|
| rate `25·180,456/37,545,489` | 0.120158243 | **81.04%** | **inside the 0.07–0.13 floor band already** |
| seg `100·2.0139e-4` | 0.020139000 | 13.58% | **23,757 flips outstanding** |
| pose `√(10·6.37e-6)` | 0.007981228 | 5.38% | **already BELOW the historic base floor** 6.993e-6 (jg5 law 1) |
| **S** | **0.148278471220** | | gap to floor-band top 0.13 = **0.018278** |

**Exchange rates, and one independent control.** 1 B = **6.658590e-07 S**. One seg flip =
**8.477105e-07 S = 1.2731 B-equivalent**; breakeven **0.7855 flips/B**. `cg2` derived the same
constant independently at `W = 1.273108215332031 B/flip` — a clean positive control on the
S-arithmetic, from an arm that ran no scorer. Pose marginal `dS/d(d_pose) = 626.47`, so 1e-7 of
`d_pose` is worth **94.1 B**. The T4 admit band is **3.5e-6 S = 5.26 B = 4.1 flips**; nothing below
that is measurable and no projection below it is quotable.

**Two consequences that bind every row below.**

1. **A pose reopen has a hard ceiling of 0.007981 S** — the entire pose term — and the carrier
   already sits below the base floor. Any pose-carrier row must be priced against that ceiling, not
   against the ancestor bodies where `d_pose` was 1e-3 to 31.
2. **A rate reopen competes with two measured closures.** `bp1`: −5 B on the section-coding axis,
   instrument first calibrated to reproduce the shipped `semantic` stream **to the byte**. `tx1`:
   model axis closed at ~475 B. `tx1` states the consequence plainly and I adopt it: **"rate cannot
   close the gap alone… representation axis OPEN."**

---

## §2 THE REACH OF THE CARRIER RE-SOLVE — the load-bearing unknown behind half the corpus

L1 is the law with the most reopen leverage, and its critical quantity is measured **three times,
across a 1,000× span, and jg5's own law says it is BIMODAL and must never be modelled with a mean.**

| source | n | cancellation | residual factor |
|---|---:|---:|---:|
| `jg1` §S2b aggregate (the number that reopened `rc4`) | **3** | 99.9874% | 7,936× |
| `dx1` §5, same mechanism at scale | **454** | **99.725360%** | **364×** |
| `jg5` whole-set re-solve, as read by `fs3` §T12 | 600 | 87.5% | **8×** |

Against the doors that need clearing: `rc4` 99.807% · `sa1` keep87 99.805% · `fs3` mirror 99.856%
(696×). **At n=3 every door clears. At n=454 rc4's door FAILS. At the whole-set figure every door
fails by 8.5–87×.**

`dx1` states it directly: *"the reopening's headline cancellation is 21.80× optimistic at scale."*
Under **L5-of-na10** (pose estimate band 13.4× seg's; ~100× the pairs for equal precision) an n=3
pose aggregate is exactly where estimates wander.

**Ruling for this audit: no row is reopened on the n=3 reach.** Where a reopen needs the carrier,
I price it at `dx1`'s n=454 figure or refuse it. This is the single largest correction I make to
na10, and it removes na10's own top row.

---

## §3 REOPEN — ranked, top rows first

### R1 · `cb1` MyCar hood — the favourable-direction row that two audits reopened and nobody fired

| | |
|---|---|
| **original verdict** | ADMITTED **−0.051646**, then quarantined |
| **precondition that killed it** | its ΔS is **98% an ancestor-vehicle pose absolute** (implied base `d_pose ≈ 31` vs rc2's 6.37e-6) — an L18 ancestor-transfer defect, not a mechanism failure |
| **law that moves it** | **L4** (re-score on the authority lineage) + the relative-pose law [[m87]] |
| **status** | na10 row 9, **NOT FIRED**. Confirmed by corpus sweep: no 08-19/20/21 memo touches it. |
| **projected ΔS** | ancestor pose term √(10·31) = 17.607; −0.051646 = **0.2933% relative**. Same relative effect on rc2's 0.007981 pose term → **−2.34e-05 S = 6.7× the bar.** DERIVED, `verdict_scope: instance`. |
| **resolving measurement** | re-score the retained MyCar-hood candidate's pose6 against `gt_cache_dali.pt["pose"]`. Pure arithmetic on retained bytes. |
| **cost** | **$0, ~140 s** |

**Why it ranks first despite not being the largest.** It is the only reopened row whose *direction is
favourable*, its resolving measurement is arithmetic on bytes already on disk, and it has survived two
audits unfired. Per [[m48]] the uncomfortable direction is the one an audit must not skip.

### R2 · `fx2` 19-member build — refused on decode margin, and the wall has since been measured open

| | |
|---|---|
| **original verdict** | best measured architecture **−797.42 B**; the 13-member **−710.84 B** shipped instead |
| **precondition that killed it** | **decode wall-clock, on a projected budget.** `fx1` §6 is titled *"Decode wall-clock — the constraint that picks the candidate"*: allfam_fast cost +127.3 s = **42.7% of a 297.7 s LOCAL ADVISORY headroom**. `fx2` §8: the 19-member build *"leaves only 29 s of margin"*. `dx1` then refused the remainder: *"+89 s to a body already decode-REFUSED"* — jg5 at **1,419.9 s vs CI [822, 1302]**. |
| **law that moves it** | **L5**, and the row named its own door: *"**it becomes the pick the moment somebody measures real T4 headroom instead of projecting it.**"* |
| **the door, measured** | `rc2`, contest-CUDA, on the shipping object: **498.476 s charged ≤ 822 s ceiling.** Slack **323.5 s**. dx1's +89 s lands at 587.5 s — **234.5 s under the ceiling.** The 1,419.9 s that refused it was the pre-port Python decoder; `cd1` priced the corrector port (71.7% of the T4 token stage = 917.929 s) and `rc2` landed it. |
| **projected ΔS** | −86.58 B = **−5.765e-05 S = 16.5× the bar.** The −86.58 B is MEASURED (fx2's own race); the ΔS conversion is exact arithmetic. |
| **placement** | the corrector adds **zero counted bytes** — it is generic receiver code, free under rule 118. This is the far-favourable side of L6's 26×. |
| **honest caveat** | fs3's average-vs-marginal law applies to the member ladder too: member 1 returned 340.82 B for +13.1 s (26.0 B/s); members 2–11 returned 21.9 B/member for 11.4 s/member (1.92 B/s) — a **13.5× collapse**. Do not extrapolate a 20th member. |
| **resolving measurement** | rebuild the 19-member build on the rc2 body, real re-encode, one T4 row |
| **cost** | rebuild + **$0.16** |

### R3 · `pg1`-GN celldrop50 — NOT FIRED, and it decides itself for $0

na10 row 10, never touched. Margin **3.06× over break-even**, and na10's own flag is that the margin
is *smaller than the lineage fork*, on a row **double-inflated** (PyAV lineage + hardest-first
selection). **Law: L4.** Resolving measurement: re-score 113 pairs against the DALI table. **$0,
~140 s.** Projected ΔS: unresolved by design — the row exists to be settled, not projected.

### R4 · `qs1` / `qs5` / `qs4` — re-price at the MEASURED compensation cost, then FOLD

**The precondition that killed them was the rate leg, and that rate leg was compensation coding.**

- `qs1`: REFUSED **+2.425702e-5** `[contest-CUDA T4 dual-axis, n600]`. seg −32 flips = −2.712674e-5;
  pose +1.126177e-7; **rate +77 B = +5.127114e-5 — "the dominating term."** 32 flips / 77 B = 0.416
  flips/B against the 0.785 breakeven — **1.9× short.** Coding price **12.8 B/pair × 6 pairs = 76.8 B
  ≈ the entire 77 B.**
- `qs5`: REFUSED **+2.519822e-6** (a near-miss). seg −17 flips; pose −3.814320e-7 (**below base**);
  **rate +26 B.** Its bar: *"≥21 flips negative"* at ≤26 B.
- `qs4`: REFUSED **+2.44e-4**, caused by a **stale carried compensation** — a defect, not a physics
  result. `qs5` cured it: *"the exact-object Schur solve not only cancelled frame-1 pose leakage, it
  slightly improved it."*

**The law that moves them: L1's rate half.** `fs3` measured the carrier compensation at
**0.0991 B/pair (0.0088 B/coefficient)**, with a byte-identity control — reverting all 454 moved
pairs reproduces the jg5 body byte-for-byte. That refutes every price at source: `fs1` 10.5 (**106×
high**, and a coefficient COUNT misread as a price), `up3` 27–36 (272–363× high), `na10` 0.83 (8.4×
high). Priced correctly the move is **edit 5.667 + compensation 0.0991 = 5.766 B/pair** — which
**satisfies `qs1`'s own reactivation lever ("get ≤6.8 B/pair at equal cancellation") by 1.2×.**

| row | as paid | re-priced at 5.766 B/pair | verdict |
|---|---|---|---|
| `qs1` (6 pairs, 32 flips) | 77.00 B → net **+2.4257e-05** | 34.60 B → net **−3.978e-06** | **1.14× bar, a win** |
| `qs5` (3 pairs, 17 flips) | 26.00 B → net **+2.5198e-06** | 17.30 B → net **−3.274e-06** | **0.94× bar, still short** |

**These are PROJECTIONS and are not quotable as rows.** `fs3` parked its own §R6 tightening for
exactly this reason — a price transferred across an admission cut. Both sit within 1.2× of the
measurement band, which is where estimates are least trustworthy.

**Disposition, per L3: FOLD, do not re-run.** Both target a field the live solve already edits, so
they enter the canonical F1/`cw1` descent engine as proposal classes inside one realized-acceptance
loop. Spawning a parallel candidate chain double-spends the shared compensation budget — the exact
error `bu1`/`mc36` measured at **3.705×**. `verdict_scope`: `qs5`'s micro-edit family stays **PARKED
at FORMULATION**, which is the correct resting state; this reopen supplies its price, not its verdict.

### R5 · The `na2` / `na5` pose re-measure queue — six rows, blocker dissolved, still never fired

na10 reopened this as a queue. **Confirmed still exactly where it left it.**
`ddm_na2_strided_rerun_four_pose_family_verdicts_20260803` carries **one ledger event —
`registered`, status `pending`** — and the five `na5` rows have **no repo-ledger presence at all**.
Four pose-family verdicts still rest on **n8/n24 contiguous prefix** documents against a measured
serial effective N of **40.22/600**, and pose prefixes measure **2.54–4.21× HARDER** than the
population ([[m96]]) — the exact false-negative shape. **Law: L4 supplies the missing apparatus**
(jg1/up2 $0 DALI instruments; `gt2`'s fail-closed `assert_gt_lineage` at the read). **$0 local,
gated only on rebuilding absent harnesses.**

This is also a live instance of [[m89]]: five rows exist only in prose and never reached the repo
ledger, so no arm can see them. **Registering them is the cure, and it is free.**

### R6 · `qs2` dead-zone steps 2–4 — one $0.16 T4 row, never fired

na10 row 7. Confirmed NOT FIRED: none of the 14 Modal calls in the 08-19→08-21 window is a qs2
lattice row. Cheapest unfired na10 row that needs paid compute.

---

## §4 STANDS — the honest non-reactivations

First class ([[m48]]). An audit that reopens everything has measured nothing. **Two of these are
na10 reopens I DOWNGRADE**, on arithmetic na10 did not run.

| # | row | law tested | why it STANDS |
|---|---|---|---|
| S1 | **`rc4` rung-4 token drop** | L1, L2 | **STANDS, and now on RATE — refused twice over.** `fs2` built it and had jg2 price it for real: modelled 11,716.7 B, **MEASURED 1,022 B = 0.0872× realised/modelled**; needs 9,305.1 B → **9.10× short**. At u=12 it **COSTS 37 B** (archive 180,493 vs 180,456 — visibly larger). Separately `dx1` measured the reopening rationale **21.80× optimistic** (99.725% at n=454 FAILS rc4's own 99.807% door). |
| S2 | **`ps1u` r2** | L4 | **na10 REOPEN → DOWNGRADED TO STANDS.** The lineage fix cannot reach it. With pose set to **exactly zero**: seg −3.137e-05 + rate +3.915e-04 (588 B) = **+3.602e-04 S, 103× the bar in the wrong direction.** 37 flips are worth 47.1 B; the row spent 588 B — **12.5× short on rate alone.** `gt2` also found the selector was already repointed by commit `809199d2`, so the premise was stale too. |
| S3 | **`ps135b` pass-4 carrier** | L4 | **na10 REOPEN → DOWNGRADED TO STANDS (FOLDED).** `up2`/`up3` executed it at stronger scope: all 600 independent 12-DOF carrier solves targeting DALI, **429 improved / 0 worsened**, byte-closed, T4 row. And at rc2's operating point the archive is **+6,767 B = +4.506e-03 S — 0.56× of the ENTIRE pose term.** Pose cannot pay for its own rate. |
| S4 | **`sq2` R8 pose guard** | L1 | **FOLDED** onto `SL2`, whose `d_seg 0.0043` is **21× the live body's 2.014e-04** — an ancestor vehicle. L18 forbids the transfer. |
| S5 | **`js6b`** | L1, L3 | `fs1` reopened it; `fs3` §7 then found **18 of 200 rows sit on pairs jg5 never edited** and ruled **"No js6b row may be cited as an admit."** The cheap column rests on an edit encoding for semantic cells **that nobody has measured** — the same compensation-vs-edit category error fs3 had just caught. STANDS pending that price. |
| S6 | **`sa1` family** (FAMILY) | L1 | **STANDS.** Killing leg is pose at **68–512× the rate credit**. Its own reactivation criterion is compensated editing — but the compensation-price law moves the **RATE** of compensation, not its **REACH**. keep87 needs **99.805%** cancellation; `dx1`'s n=454 figure is 99.725% and jg5's whole-set is 87.5%. Short by 1.4–64×. na10's √-deflation already put the legs at +0.0328/+0.0124/+0.0096 against a −3.5e-6 bar. |
| S7 | **`SD1M` mass ladder** (FORMULATION) | L1 | **STANDS.** Killing leg is **SEG** (render amplification ~38,700×; damage ∝ mse^0.4 falls slower than rate credit). Its own memo: *"the in-compile Schur compensation addresses pose only; **the seg leg has no compensator in this family**."* No law touches a seg kill. |
| S8 | **`et1`/`ob1`/`gp1` address band** | L2, L6 | **STANDS on a structural kill**: the band is computed from *"a label field the receiver does not have and cannot compute."* Receiver custody is not a pricing question. η 0.3017 vs a bar of 0.61491 that **rises** with radius (0.615/0.658/0.679) — *"the band cannot be widened into profitability."* |
| S9 | **`cg2` camera grid** (FAMILY, indep. corrections) | L6 | **STANDS.** Camera-grid addressing costs exactly 2 bits more per correction and **blast radius is exactly 1**, so it can never amortise. Its break-even `W = 1.273108215332031 B/flip` reproduces my §1 constant exactly — an independent control. |
| S10 | **`lr1` lattice-solve rebase** (FAMILY) | — | **STANDS.** A deterministic instrument over 600/600 pairs: the teacher is a **strictly worse copy** of a free reference (corr 0.9867; shuffled control 0.5606). My charter forbids reopening a family-scope refutation by a deterministic instrument absent a law touching the mechanism. None does. |
| S11 | **`bp1` section coding** | L2, L6 | **STANDS, −5 B.** Instrument calibrated to reproduce the shipped stream to the byte first. Also contains its own NO-FAKE catch: a −215 B result was an artifact of a lossy transform *"throwing bytes away"*; the invertible version is +7 **worse** than shipped. |
| S12 | **`dc1` coder swap ≤7.8 B** | L5 | **STANDS**, and **L5 does not reopen it** — dc1's own verdict is *"decode compute is a sound currency with no market"*; the budget was never its killing precondition. Its over-wide naming was already corrected in-band by `fx1`. |
| S13 | **`pz4a` coarsening** | L6 | **STANDS on a ceiling argument that no law reaches**: *"even a hypothetical zero-byte allocation map would leave the measured gross ceiling at 500 B, 1,500 B below the gate."* Ceiling-first, exactly as L6 prescribes. |
| S14 | **`rr7` native token decoder** | L5 | **STANDS — and this is where L5 is most tempting and most wrong.** The port that won was the **CORRECTOR** port (`cd1`→`rc2`). The **token-decoder** port measured byte-perfect and **SLOWER**: 1,612.579 s on the shipping axis against a 797.7 s projection — *"putting the accelerator in the tree costs 193 s."* |
| S15 | **`pk3`/`pk4` frame-0 overlays** (FORMULATION) | L4 | **STANDS/SHARPENED.** na10 re-scored on DALI at $0 and **the sign did not change** (rungs 42/250/1000 all hurt or do nothing). Scope narrows to *linear overlays fit against a PyAV-contaminated residual*. |
| S16 | **`cpu1` CPU axis** | L5 | **STANDS.** CPU inflation measured **4,369.6 s = 2.428× the entire CI job wall**, and CPU is **0.0450 worse structurally** (a different GT decode selected by device at `evaluate.py:39-42`), not tunable. |
| S17 | **`fo2h` eta channel** | L4 | **STANDS, fresh.** Re-scored directly on DALI GT, the **pose leg costs 41.3× the seg gain, 6/6 pairs.** NET NON-SUPPLIER on the shipping axis. |

**Tally: 6 REOPEN · 17 STANDS**, of which **2 are downgrades of na10 reopens** and **5 stand
*more strongly* under the sharper instruments** (S1, S2, S3, S6, S15).

**Denominator** ([[m50]] — report it): **401 memos under `.omx/research/` dated 2026-08-* carry a
strict negative token** (131 in 08-0x, 231 in 08-1x, 39 in 08-2x). I adjudicated 23 at receipt,
drawn from na10's 24-row inventory plus the charter's 10 seeds plus a mechanical precondition-class
sweep. **I did not adjudicate the other ~378 and make no claim about them.**

---

## §5 THE PATTERN — what single precondition class moved the most rows

1. **The price and reach of POSE COMPENSATION moved the most rows — and it moved them in opposite
   directions, netting near zero.** Its RATE half collapsed **129×** (12.8 B/pair paid → 0.0991
   measured, byte-identity-controlled), reopening exactly the rows whose refusal was
   compensation-coding-dominated: `qs1`, `qs5`, `qs4`. Its REACH half went the other way: measured at
   n=454 the cancellation is **99.725%, 21.80× worse** than the n=3 figure that reopened `rc4`, and
   the whole-set re-solve delivers **8×** where `sa1` and `fs3`'s mirror need 68–696×. **Compensation
   is now cheap to ship and weak to rely on.**
2. **Real re-encode pricing moved rows almost entirely in the STANDS direction, because the law is
   asymmetric.** Toward-argmax credits are **91.3% phantom** (0.0872× realised/modelled), so every
   drop-family negative was refused *despite* an 11×-inflated credit and now stands **stronger**. The
   away-from-argmax correction is only **8.5%** — too small to flip anything, and it moved nothing.
3. **The decode wall moved exactly one row, and it was the row that named the door itself.** `fx2`
   wrote *"it becomes the pick the moment somebody measures real T4 headroom"*; `rc2` measured it. But
   L5 is also the most over-applied law in this audit: `dc1` was never budget-bound, and `rr7`'s port
   was slower, not blocked. **A moved precondition only reopens the rows it was actually the
   precondition FOR.**
4. **Placement (L6) and the lineage fork (L4) closed more than they opened.** L6 kills by ceiling
   before anyone builds (`pz4a`, `cg2`); L4 was already spent corpus-wide by na10 and its remaining
   value is three $0 re-scores nobody has run.
5. **The operating point out-graded every law.** Rate is 81% of S and already inside the floor band;
   coder −5 B and model ~475 B are both measured shut; and the seg-edit actuator that bought the last
   11,997 flips is at its marginal stopping point — **0.662 value/cost against 1.0 breakeven, 8.91×
   degraded from its own set average.** No re-pricing restarts it. **`tx1` has it right and this audit
   confirms it independently: rate cannot close the gap; the representation axis is the open one.**

---

## §6 WHAT I ASK FOR, AND IT IS NOT A NINTH AUDIT

na10 asked for a fail-closed GT-lineage gate; `gt2` built it (11 consumers declared, undeclared
population 0/11, red positive control fired). That ask is **discharged**.

**Mine is smaller and it is the reason four of na10's ten reopens are still untouched: three of them
cost $0 and ~140 s, and no surface schedules them.** Rows R1, R3 and R5 are arithmetic on retained
bytes. They sat through two audits because a reopen is written into a memo and a memo is not a queue
([[m89]] — arms see the repo ledger, not prose). **Register R1/R3/R5 as repo tasks with owners, and
the queue drains itself.** `follow_on_work_fires_immediately_or_it_is_orphan_poison_20260803`.

Sisters: [[m48]] (negative ↔ cure) · [[m56]] (unwired-but-built — `wc2` found a 4.4× token
accelerator refused by every candidate because the corrector was wired to Python only) ·
[[same_defect_negatives_masquerade_as_family_convergence_20260805]].

---

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B `[contest-CUDA T4, n600]` — UNMOVED by
ddm_na11.** This arm produced no row and claims none.
