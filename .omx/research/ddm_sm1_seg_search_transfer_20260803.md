# ddm_sm1 — the seg solve's search defects, measured: both present, both small, and the detector that was hiding them

`verdict_scope` on every row. Exact contest pointer **0.1910828242 UNMOVED**.
Own-vehicle rows are `[macOS-CPU advisory]`, `score_claim=false`, `promotion_eligible=false`.
Live best **S = 0.7910689 @ 353,805 B** (`ddm_pu2`); gap **0.6189279**; 1% of gap = 0.0061893 S = 9,295 B.
Per-flip price on this base **8.477105e-07 S/flip** (QA03's own `seg_S_delta / net_flips_total`).

STORES CONSULTED: `ddm_xa1` (**the arm whose §4 experiment this executes**), `ddm_ss1`, `ddm_pu2`,
`ddm_sq1`, `ddm_dc1`, `ddm_sv1`, `ddm_ob1`, `ddm_gt3`, `ddm_os1`, `ddm_pc2`, `ddm_na2`, `ddm_mh1`;
`tools/sb1_seg_batch.py`; `/Volumes/VertigoDataTier/pact/ddm_sb1_20260729/qa03/` (the shipped cap-4
store); `/Volumes/VertigoDataTier/pact/ddm_sq1_20260803/receipts/sq1_stage_n32.json`;
`/Volumes/VertigoDataTier/pact/ddm_pb1_20260729/p2c_aimed_archive.zip` (sha `b9a7983b`).

---

## §0 — HEADLINE

`ddm_pu2` measured the POSE defect to be **SEARCH, not model**, in two parts: under-convergence
(the solve stops while still descending) and basin-trapping (the result depends on where it
started). `ddm_xa1` §4 designed the seg-side transfer test, priced it, and ran no scorer. This arm
ran it.

**Both pathologies are PRESENT on seg. The transfer premise HOLDS mechanically. And closing both
completely buys ~0.06% of the gap** — on a base the live container's parser refuses.

| question | verdict | measured |
|---|---|---:|
| UNDER-CONVERGENCE present? | **YES** | uncapping 4→32 on the full 120: **+382 flips (+20.5%)** = 3.238e-04 S = **0.0523% of gap** |
| BASIN-TRAPPING present? | **YES** | 4 starts beat the shipped single start on **11/16** instances: **+72 flips (+15.7%)** = 6.104e-05 S = **0.0099% of gap** |
| is greedy at a JOINT optimum? | **NO** | exact ±2 joint optimum beats coordinate-wise greedy on **5/7** instances where the box is a valid bound |
| does `pu2`'s TAIL-specificity transfer? | **NO — it INVERTS** | tail +7.5% vs **non-tail control +28.3%** (pose was tail 0.2013 / control 0.9388) |

**So the answer to the charter is neither of the two it anticipated.** The pathology is not absent
(that would have closed the transfer) and it is not a large unexploited lever (that would have been
the first thing to move seg). It is present, real, mechanically identical to pose — and **priced
out**. Seg is 64.9% of the remaining gap; the seg *token solver's* entire search headroom is
~0.06% of it. **The seg lever is not search quality on this actuator.** That is a specification,
not a kill: to matter, a seg lever must be ~1000× this.

Along the way the arm found and fixed **three compounding defects in the censoring detector that
was supposed to be measuring exactly this**, and lost a 28-row store to a duplicate writer.

---

## §1 — DENOMINATORS (`m50`)

| population | count |
|---|---:|
| QA03 instances re-run at cap 32 (full shipped top-k) | **120 / 120** |
| — matched 1:1 against the shipped cap-4 store | **120** (0 duplicates, 0 chain breaks) |
| probe instances (multi-start + box oracle) | **16** = 8 tail-censored + 8 non-tail control |
| — starts per instance | 4 (`base`, 2× local ±3, 1× uniform over the whole lattice) |
| — box-oracle configs enumerated per instance | 625 (L∞ ≤ 2, exhaustive) |
| — instances where greedy STAYED in the box (oracle is a valid bound) | **7 / 16** |
| scorer evaluations this arm | ~19,600 (measured 0.389–0.7 s each) |
| distinct pairs holding >1 instance (coupling is testable) | **11 / 108** |

**Where the instrument stops, in the claim's own units (`m94`).** The actuator is the 4-channel
integer token code of one 16×16 cell, 16 levels per channel. The objective is the **whole-pair
realized SegNet argmax flip count through the real TR1 receiver** (one render + one SegNet forward
per evaluation) — not a proxy. The oracle's reach is L∞ ≤ 2 (625 of 65,536 configurations, 0.95%
of the cell's lattice). **"Seg search defects are small" is scoped to THAT actuator.** It says
nothing about seg levers that change the address, the basis, or the carrier.

---

## §2 — THE BASIN VERDICT (charter priority 1)

### 2.1 Multi-start pays, on 11 of 16

| stratum | n | single-start banked | multi-start adds | non-base winner |
|---|---:|---:|---:|---:|
| tail_censored | 8 | 279 flips | **+21 (+7.5%)** | 4/8 |
| control_nontail | 8 | 180 flips | **+51 (+28.3%)** | 7/8 |
| **all** | **16** | **459** | **+72 (+15.7%)** | **11/16** |

Winning starts: `base` 5, `local_r3_3` 5, `uniform_2` 4, `local_r3_1` 2. **A uniform-random start
over the whole 16⁴ lattice won 4 times** — `pu2`'s "a start 8.2× worse descends past the shipped
point", reproduced on a different axis with a different actuator.

Spread across starts is large and two-sided: per-instance range in final flips runs from 6 to
**210** (p182: `[808, 813, 1017, 807]` — the uniform start lands 209 flips worse than base, and a
local start lands 1 better). **Distant starts are usually much worse and occasionally best.** That
is the signature of a rugged objective, which is what a CNN argmax over an integer lattice should
be.

### 2.2 The tail-specificity INVERTS — the most transferable finding here

`pu2` measured the pose multi-start defect as **TAIL-SPECIFIC** (tail median ratio 0.2013, 5/6
improved; non-tail control 0.9388, 2/6), and `xa1` §4 made a non-tail control **mandatory** for
exactly this reason. It earned its place: on seg the **non-tail control benefits 3.8× MORE than
the tail** (+28.3% vs +7.5%), the reverse of pose.

Mechanism, and it is not mysterious: QA03 selects its tail by *atlas flip count*, so tail cells
have the most room and the greedy descent from base already finds a lot of it; the shallow non-tail
cells are where the *choice of basin* is proportionally decisive. **The mechanism transfers; the
targeting rule does not.** Anyone carrying a multi-start cure across axes on `pu2`'s authority
would have aimed it at the tail and captured the smaller half.

### 2.3 Greedy is NOT at a joint optimum — structurally confirmed

`xa1` Probe B predicted this from source (`_best_single_quantum` evaluates 4 channels × ±1 and
accepts strict improvement ⇒ terminates at a **coordinate-wise** local minimum, which for a
non-separable CNN-argmax objective need not be a joint one) and called it "structural, not
speculative". Measured:

Restricting to the **7/16 instances where greedy stayed inside L∞ ≤ 2** (so the box optimum is a
genuine bound on it — on the other 9 greedy *escaped* the box, up to L∞ = 5, and beat the oracle
by construction):

```
p167 L∞=2: greedy 712   exact joint opt 709   (+3)
p514 L∞=2: greedy 933   exact joint opt 933   ( 0)
p182 L∞=2: greedy 808   exact joint opt 806   (+2)
p145 L∞=2: greedy 724   exact joint opt 724   ( 0)
p250 L∞=1: greedy 643   exact joint opt 639   (+4)
p242 L∞=1: greedy 645   exact joint opt 641   (+4)
p542 L∞=2: greedy 1007  exact joint opt 1002  (+5)
```

**5/7 — the exact joint optimum beats coordinate-wise greedy, total 18 flips.** The prediction is
confirmed with an instrument whose reach is enumerated rather than argued. The residual is 2–5
flips per cell: real, and tiny.

### 2.4 Price

`72 flips × 8.477105e-07 = 6.104e-05 S = **0.0099% of gap**`, for **4× the solve compute**.

Extrapolating the +15.7% rate onto the full 120-instance cap-32 yield (2,248 flips) would give
~353 flips ≈ 0.048% of gap — but **the 16 instances are a deliberately stratified half-tail
half-control sample, not a random subset of the 120**, so that is an assumption-laden ceiling, not
a point estimate. **I refuse a point estimate** (the `os1` discipline; and `pu2`'s own precedent —
it cut its multi-start forward estimate from 3.6–7.1% of gap to ~0.6% once it ran a non-tail
control).

---

## §3 — UNDER-CONVERGENCE, clean and full-population (charter priority 2)

### 3.1 The clean matched comparison — the withdrawn result REPRODUCES, and grows

I withdrew a +13.4% figure because the rows it used sat inside a duplicate-writer window (§5).
Re-measured on a clean single-writer run, **all 120 instances, same order, 0 duplicates, 0 chain
breaks**:

| | cap 4 (shipped) | cap 32 | delta |
|---|---:|---:|---:|
| total net flips | **1,866** | **2,248** | **+382 (+20.5%)** |
| `seg_S_delta` | −0.0015818278 | **−0.0019056532** | −3.238e-04 |
| instances improved / worse / unchanged | — | — | **29 / 1 / 90** |

The cap-4 total reproduces the shipped receipt **exactly** (1,866), which is the harness control
`xa1` §4 asked for (arm C). **+382 flips = 3.238e-04 S = 0.0523% of gap.**

Note the shape: **29 of 120 instances carry the entire gain, 90 do not move at all.** The uncap is
not a broad improvement; it is a small number of cells that had much further to go
(p66 52→95, p138(12,1) 17→50, p72 9→40, p569 37→66, p586 36→64).

### 3.2 The corrected censoring rate — and a correction to my own correction

`dc1` reported **51/120 (42.5%) stopped on the cap**, and `xa1` Probe A amplified it to "still
descending at essentially their initial rate". Measured directly, by raising the bound and seeing
who actually moves:

| | |
|---|---:|
| cap-4 rows labelled `"cap"` | 51 (42.5%) — `dc1`'s number |
| of those, **actually still descending** (took >4 steps at cap 32) | **29** |
| **FALSE-CENSORED** | **22 = 43.1%** |
| **TRUE censoring rate** | **29/120 = 0.2417** |

`for/else` cannot distinguish "ran out while descending" from "converged exactly at the bound", and
**43.1% of the labels were the coincidence.** `dc1`'s 42.5% and `xa1`'s amplification of it are
**both overstated; the true rate is 0.2417.**

⚠ **This also corrects THIS arm's own earlier figure.** On a 12-instance prefix I measured 58.3%
false-censored and reported it. At full population it is **43.1%**. My 12-instance sample was
drawn from the atlas-rank prefix — **`m88`/`m96` applied to my own measurement**: a prefix of a
rank-ordered population is a different population, and it biased my estimate 15 points high. The
corrected rate is 43.1%; **43.1% is the number to carry**, and the commit message landed with the
old 58.3% figure is superseded by this row.

### 3.3 At cap 32 the bound genuinely stops binding

`stop_reason` over the 120 cap-32 rows: **`converged` 107, `no_move` 13, `cap` 0.** The receipt
emits `cap_saturated_frac: 0.0` with `censoring_determinable: true`, `n_cap_unknown: 0`.

**This 0.0 is EARNED, and the distinction from the defect's 0.0 is the whole of §4.** It is earned
because every row carries a precise label from the probing solver, and because 29 rows demonstrably
took more than 4 steps — proving the bound was raised into genuine slack rather than declared
clean. The defect's 0.0 was produced by classifying rows against a bound nothing could reach. Same
number, opposite epistemic status. **This is the first `cap_saturated_frac` in the project's
history that is a measurement.**

### 3.4 Cell-order coupling — reproduces, but I over-claimed it

I provisionally reported that deeper search on one cell made its *pair* worse. On clean data it
**reproduces exactly** — and is rarer than I implied:

```
pair 138: cap4 [17, 42] = +59    cap32 [50, 0] = +50    delta -9   <-- deeper search HURT the pair
pair 183: cap4 [0, 23] = +23     cap32 [0, 36] = +36    delta +13
9 other multi-instance pairs: delta 0
multi-instance pairs (11): cap4 317 -> cap32 321, net +4;  1/11 hurt
single-instance pairs (97): cap4 1549 -> cap32 1927, net +378
```

p138's cell (12,1) descended 13 steps instead of 4, taking the pair to 1036 flips, at which point
its sibling cell (12,2) found **no improving move at all** (was +42). The mechanism is real: the
instances are **not independent** — they share a whole-pair objective and are solved sequentially
and greedily, so a deeper solve on one cell can strand the next.

**But the honest calibration is: 1 of 11 coupled pairs, and the aggregate over coupled pairs is
+4, not negative.** "More search is non-monotone across cells sharing a pair" is correct as an
**existence** claim and wrong as a **tendency** claim. I made it as the latter; it is the former.
It still matters for the transfer premise — the pose-side result was measured on independent
per-pair solves, and this actuator's units are coupled — but it does not offset the uncap.

---

## §4 — THE DETECTOR THAT WAS HIDING THIS: three compounding defects (landed `f542e43663`)

Each was found by reviewing the fix for the previous one. All three make the censoring statistic
read *clean*.

**(1) The detector zeroed itself precisely when the cure was applied.** The resume path classified
persisted rows using the **reader's** `--max-quanta`:

```python
r.get("stop_reason", "cap" if len(r["accepted_steps"]) >= args.max_quanta else "converged")
```

Rows never recorded the bound they ran under. Replaying the shipped 120-row cap-4 store:
`cap_saturated_frac` = **0.425** at `--max-quanta 4`, and **0.000000 at 8, 12, 32 and 48**
(reproduced). The receipt's own note reads *">0 here means the solve is CENSORED, not solved"*, so
0.0 reads as SOLVED. Since `dc1` raised the default **4 → 32 specifically to cure the censoring**,
the obvious next action — resume the existing run at the new default — **certifies the censoring
cured having done no work.** *Fix:* rows co-record `max_quanta_at_write`; undeclared legacy rows
classify `unknown`; the fraction emits **`None`**, never a misleading 0.0.

**(2) `stop_reason="cap"` was 43.1% false-censored** (§3.2). *Fix:* the `for/else` probes once more
(≤8 evals, score-neutral — `_best_single_quantum` restores the codes it tries) and emits
`converged_at_bound` when the solve had already converged. Pre-cure `"cap"` labels are kept as a
distinct class (`cap_conflated`) so the two semantics are never averaged.

**(3) Vacuity through the operator's own flag.** With the reader's cap no longer leaking in, an
operator can still *declare* `--legacy-cap 32` over a store whose rows max out at 4 steps — every
row classifies "converged" **by construction** and the fraction reports a clean 0.0 having tested
nothing. `m50`: a check that cannot return a negative is not a check. *Fix:* `legacy_cap_is_vacuous`
detects a declared bound that cannot bind any row and downgrades those rows to undetermined.

Result: `cap_saturated_frac` is now `None` on **every** path where it cannot be honestly measured.
Verified against the real shipped store under all three flag settings. **34 tests**, ruff-F clean.

**The general bug class, and it is not specific to this tool: a derived label computed from a
parameter that is not co-recorded with the data.** The label silently re-derives itself against
whatever the current reader happens to be holding. `os1`'s census law is the same disease one level
down (*emission* fixes the future while the past stays unreadable); `m97` is its sister (*a cure
landed in source is not a measurement*). This is the third form: **a cure that disables its own
detector.**

---

## §5 — THE DUPLICATE-WRITER INCIDENT, and why the procedural guard was not enough

The harness reported a detached background job as **`failed exit code 144`** (SIGURG). The signal
killed the harness's *wrapper*; the Python child kept running. Relaunching on that notification —
the correct response to "your job died" — produced a **second writer on an append-only JSONL**:
28 rows / 24 unique, three duplicated instances, one broken pair-state chain, in a store I had
already read and reasoned over. The rows behind my provisional +13.4% and p138 findings were inside
that window, which is why both were withdrawn (§3.1, §3.4 — both then reproduced).

**The part worth keeping:** I wrote the law *"a launcher's nonzero exit is not evidence the job
died — check the process table"*, then **followed it, and `ps` returned empty twice while the
process was demonstrably alive and 17 minutes into its run.** The procedural guard failed. What
caught the second writer was `src/tac/single_writer_lock.py` — a fail-closed `flock` whose holder
self-identifies — refusing my next launch and naming PID 68737.

**Liveness must be enforced by something that does not depend on a reader correctly interpreting a
signal.** The lock is now wired into every `sb1_seg_batch` subcommand and the new probe, with the
incident as its docstring and 13 tests including a genuinely cross-process case. Contaminated data
is quarantined at `.../ddm_sm1_20260803/CONTAMINATED_evidence/` with a README, not deleted.

Sister defect, same landing: the probe's first version checkpointed **per instance (~8 min)**
against a reaper firing in minutes, and banked **nothing across three launches**. Resumability's
actual requirement is **checkpoint interval < failure interval**, not "has a resume path". Now
checkpoints per arm and per 80-eval oracle chunk.

---

## §6 — CORRECTION TO `ddm_sq1` / `ddm_xa1`: a transfer that happened in name only (VERIFIED)

`xa1` §0.2 R2 refuted its charter's premise with this argument: the pose→seg multi-start transfer
*already happened* the same day, because `sq1`'s headline seg cure (realizer v1,
`eta_net −3.7640 → +0.7895`, 32/32 pairs) lists among its riders **"multi-start (`dec` and `truth`
inits, `pu2` mechanism)"**. `xa1` concluded the disease is not non-transfer but **non-banking** of
transfers that already occurred.

**Verified at source, and the conclusion does not survive.** `sq1_stage_n32.json` records
`starts = ['dec', 'truth']` and a per-pair `solved_start_tag`:

```
solved_start_tag over 32 pairs:  {'dec@25': 31, 'dec@20': 1}
```

* **The `truth` start won 0 of 32 pairs.** `sq1`'s multi-start contributed **nothing measurable**
  to its headline result. The transfer is present in the *prose*, not in the *outcome*. `xa1`'s R2
  is **implementation-refuted**: this is not an unbanked transfer, it is an unmeasured one. (The
  paradigm survives — §2 shows multi-start genuinely pays elsewhere on seg — so this narrows R2's
  evidence, it does not reinstate the charter's strong "transfers never reach other axes" claim,
  which §2's own inverted targeting rule also complicates.)
* **31 of 32 winners are `dec@25` — the TERMINAL iterate at the 25-step budget.** That is an
  **independent, unremarked under-convergence signal on a completely different seg surface** (the
  realizer's Adam descent on the scorer lattice, not the token codebook). `sq1` itself half-saw it,
  recommending *"raise the pose-null solver to Job 1b's budget (25 steps, 2 starts)"* and calling
  its own eta a floor — but the 31/32-at-the-bound fact was never stated.

**So under-convergence is now measured on BOTH seg surfaces, independently: the token solver
(§3, 29/120 genuinely truncated) and the realizer (31/32 winners at the step bound).** That
convergence is the strongest structural claim this arm makes, and it costs one line to act on:
`sq1`'s realizer budget is a raise-and-re-measure, not a redesign.

---

## §7 — WHAT I REFUTE IN MY CHARTER

**R1. "The only seg-axis solver row is MORE-SEARCH, and its cure was never run."** True when
written, and now false in a way the charter could not have known: running it revealed the cure's
own detector was broken three ways (§4), so the cure could not have been *verified* even if it had
been run. The charter framed this as an unexecuted action; it was also an uninstrumented one.

**R2. "`xa1` measured no saturation signature at the current cap; confirm or refute at n600 scale."**
**REFUTED, and the refutation is the point.** `xa1` Probe A compared per-step yield ACROSS strata
(different instances) and found the capped stratum still descending at 86.3% of the shallowest
stratum's rate — reading it as no decay. Measured WITHIN instances by raising the bound: **43.1% of
the capped set took no further step at all.** The cross-stratum inference was structurally unable
to see this, and `xa1` said so honestly (*"this is a cross-stratum comparison, not a within-instance
decay curve"*) and predicted its bias ran conservative. The bias ran the other way.

**R3. The charter's two anticipated outcomes were "pathology present → price the cure" and
"pathology absent → clean negative". Neither happened.** Both pathologies are present and both are
priced out at ~0.01–0.05% of gap each. The useful deliverable is a **threshold**, not a verdict:
a seg lever must clear ~1000× this to matter, and search quality on the token actuator cannot.

**R4. I do NOT reproduce my own 58.3% false-censoring figure** (§3.2). It was an atlas-rank-prefix
estimate; at full population it is 43.1%. I applied `m88`/`m96` to sister arms and not to myself.

---

## §8 — SCOPE (binding, stated so nobody cites this as a win)

**QA03's entire yield is 0.2556% of the current gap**, and the base it runs on is the
**767,812 B TR1** archive (sha `b9a7983b`). The live-best **353,805 B** container **refuses**
`ddm_tr1_runtime.parse_archive` (`TR1RuntimeError: TR1 archive members/order differ`) — verified
this arm, not inherited. Every number here is `verdict_scope: INSTANCE` on that base and **does not
compose with the live row without a re-derivation**, and the TR1→live bridge is OWED and unbuilt.

**This arm was run for the MECHANISM, not for its ledger value.** The +382 flips are not a
score-mover and must never be cited as one. What is portable is: (a) both search pathologies are
present on seg, (b) their combined price is ~0.06% of gap, (c) `pu2`'s tail-targeting rule inverts
on this axis, (d) the censoring detector was broken three ways, (e) under-convergence is
independently present on `sq1`'s realizer.

---

## §9 — ROUTING

Landed as rows in `canonical_task_status.jsonl` (per `mh1`:288 and `m89` — cite CONTENT, never bare
ids). **No ΔS is claimed on any row.**

1. **`sq1` realizer budget raise** — 31/32 winners at the 25-step bound; raise and re-measure.
   Cheapest live under-convergence item on the board, and it sits on the surface that produced the
   only large seg η win we own (`−3.7640 → +0.7895`). ~$0 to specify, one scorer job to price.
2. **Multi-start on the token solver: BUILT, priced, NOT recommended** — +15.7% for 4× compute =
   0.0099% of gap. Registered so nobody re-derives it. If it is ever switched on, aim it at the
   **non-tail** (28.3%), not the tail (7.5%).
3. **Seg lever threshold** — any future seg-search proposal on this actuator must clear ~1000× the
   measured headroom or it is dominated before it starts.
4. **The bug class `a derived label computed from a parameter not co-recorded with the data`** —
   generalize the §4 audit beyond `sb1_seg_batch`. Every resumable store that classifies its own
   rows is a candidate.
