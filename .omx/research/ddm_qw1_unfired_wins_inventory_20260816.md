---
arm: ddm_qw1
title: "The unfired MEASURED-wins queue is ONE row worth 1.6% of the gap and it needs no scorer slot; the one genuinely large unfired object (mp2 keep25, 14.2% of the gap in real bytes) is unmeasured behind a $0 probe nobody ran because a .done.done defect killed the queue"
utc: 2026-08-16
axis: "[local-CPU $0 arithmetic over MEASURED primary receipts] — NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] — UNMOVED by this unit"
verdict_scope_default: "stated inline per row; no new family verdict is issued here"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_qw1 — every win sitting unfired, priced against the gap

**STORES CONSULTED** (primary, re-derived not quoted): `.omx/state/canonical_frontier_pointer.json` ·
`.omx/state/canonical_task_status.jsonl` (561 rows / 212 tasks) ·
`.omx/state/codex_arm_queue.next_if_resumed.jsonl` (252 rows) · `.omx/state/main_hot_state.md` ·
`.omx/state/lever_activation_ledger.jsonl` · `ddm_ra2_pose_metric_subspace_and_cpr1_coder_20260816.md` ·
`ddm_pz5_stage0_rederivation_gate_20260816.md` · `ddm_ps1u_uncapped_pose_solve_20260816.md` ·
`ddm_pv1_pose_floor_and_admission_bar_20260816.md` · `ddm_sr1_manufactured_seg_recovery_20260816.md` ·
`ddm_hv2_arm_final_harvest_20260816.md` + its staged task-rows JSON · `ddm_fb1_…_20260816.md` ·
`ddm_gx1_…_20260816.md` · `ddm_dc1_…_20260816.md` · `ddm_kp2_payload_retention_census_20260816.md` ·
`ddm_mh1_month_harvest_20260803.md` ·
`/Volumes/VertigoDataTier/pact/ddm_pz2_pose_representation_20260810_v3/PZ2_MEASUREMENT_RECEIPT.json`
(919 candidates, re-priced) ·
`/Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815/{MP2_ADVISORY_ADJUDICATION.json,
DIFFERENTIAL_N600_FIRE_ORDER.json, advisory_n600_cpu/**/contest_auth_eval.json}` ·
memories [[m34]] [[m45]] [[m96]] [[L18]] [[the_denominator_and_the_falsifier_can_both_be_vacuous_20260816]].

---

## ANSWER FIRST

**The unfired queue of MEASURED wins on the live base is ONE row worth −1.531e-4 S = 1.596% of the
−0.0095973 gap. It does not deserve a scorer slot — because it does not need one. It is lossless.**

That row is `ra2`'s CPR1 **inner** entropy coder: +263 B raw (MEASURED, round-trip exact on
27,648/27,648 symbols), ~230 B realized after Brotli (PROVISIONAL). Because the swap is lossless,
`d_seg` and `d_pose` are exactly unchanged and **no scorer runs at all** — the whole cost is local
byte-close work at $0.

Everything else in the pile with a negative sign is **spent, refused, a ceiling, or on a dead base.**
I found no second measured win. Firing every measured credit I could locate — including the cp135
bank that `fb1` measured is *not* composable — reaches **−1.587e-4 S = 1.654% of the gap** and leaves
**14,183 of the 14,413 bytes still owed.**

**The one-sentence answer to "is the unfired queue worth a scorer slot": no. Stop mining the bank
and go build.** The bank is exhausted to three decimal places; the gap needs a mechanism that
returns thousands of bytes or moves `d_seg`, and no such mechanism is sitting built and measured.

**One thing genuinely large IS sitting there, and it is not a win yet.** `mp2` built **nine**
generations on the correct hv1 pins and scored **four**. `keep25` is retained, receiver-closed, and
**never measured** at 180,708 B — **−2,051 B = 14.2% of the gap in real bytes.** Its pose leg is
unmeasured and the measured dose-response projects it at **+0.0264 S net, 2.8× the gap in the wrong
direction**. But the trend across the only monotone pair is *favourable*, `mp2` registered
"pose-null rows likely exist" as a live hypothesis, and the named prerequisite is a **$0 local
finite-difference probe, not a scorer run** — which nobody has run, partly because a `.done.done`
receipt-naming defect silently killed the queue. **Fire the $0 probe. Never the n600 score.** §B row 4.

**The structural finding, which is worth more than the row.** `gx1` (10:17), `hv2` (18:37) and
`fb1` (19:12) each independently enumerated "the bank" today and each concluded it totals
−5.5818e-06 (qs2 + re1). `ra2` landed at **20:50**. Its credit is **27× the entire bank they
summed**, and `grep -ci cpr1` returns **0** in all three. **They were not careless — they could not
see it.** The defect is that the bank total is a hand-recomputed number in prose with no live
consumer, so *every* aggregate is stale the moment the next credit lands, and three arms burned a
day converging on a figure that a fourth arm obsoleted two hours later. Cure named in §E.

---

## A. The arithmetic every row is priced against (DERIVED, re-derived at source)

| quantity | value | label |
|---|---:|---|
| frontier `S` | `0.15959729295498598` | MEASURED, `canonical_frontier_pointer.json` |
| frontier archive | `182,759 B`, sha `80d9c8c6…0178e` | MEASURED, same pointer |
| components | seg `0.029611` + pose `0.0082945765` + rate `0.1216917` | MEASURED |
| `d_pose` | `6.880e-06` | DERIVED `= pose²/10` |
| gap to 0.15 | `0.0095973 S` | DERIVED |
| `dS/dbyte` | `25/37,545,489 = 6.658590e-07` | DERIVED |
| rate-only close | **`−14,413.4 B`** | DERIVED |

I recomputed `seg+pose+rate` and got `0.15959729291` against the pointer's `…295498598`, residual
`−4.1e-11` — the components are consistent with the pointer, so pricing off them is safe.

---

## B. THE RANKED TABLE

`%gap` is `|ΔS| / 0.0095973`. **LEVEL** is the discipline that matters most here: an AVERAGE, a
CEILING and a MARGINAL are three different quantities and only one of them prices a next step.

| # | row | ΔS | %gap | LEVEL | BASE | scope | composable onto hv1? | cost to fire | owner |
|---|---|---:|---:|---|---|---|---|---|---|
| **1** | **ra2 CPR1 inner coder** (basis stream: static order-0 Huffman → adaptive arithmetic) | **−1.531e-4** (realized) / −1.751e-4 (raw) | **1.60%** | raw = **MEASURED**; realized = **PROVISIONAL** | **hv1**, the live 22,161 B carrier | INSTANCE | **YES — lossless, seg+pose exactly unchanged** | **$0, no scorer** | unowned |
| 2 | ps1u sealed T4 order (frame-0 carrier delta, container BUILT + PROVEN + dispatcher-VALIDATED) | rate leg **+3.915e-4** MEASURED cost; pose leg **UNMEASURED**; ceiling −7.903e-3 | ceiling 82% | rate MEASURED, pose **UNMEASURED** | hv1 (candidate 183,347 B, sha `97048f9f…`) | INSTANCE | n/a — it *is* a candidate | **$0.16 Modal** (of $1.38 left) | MAIN |
| 3 | sr1 FO-1 — A1 zero-byte de-blur, **sign unmeasured** | ceiling −2.860e-2 @100% recovery; −9.597e-3 @33.55% | ceiling 298% | **CEILING with UNMEASURED SIGN** | hv1, rt1 n600 retained masks | INSTANCE | 0 bytes, so rate-neutral by construction | **$0**, ~800 s/α local CPU | MAIN |
| 4 | **mp2 keep25** (+ keep37/50/62) — retained, receiver-closed on hv1 pins, **NEVER SCORED** | rate credit **−1.366e-3** MEASURED; pose leg **UNMEASURED**, projected **+0.0278** | rate credit **14.2%**; projected net **+2.8× the gap, wrong sign** | rate MEASURED, pose **PROJECTED (n=2)** | **hv1** (180,708 B) | INSTANCE, under a FAMILY refusal | bytes are real and correctly pinned | **$0** per-row FD pose probe first — **never an n600 score** | MAIN |
| 5 | qs2 + re1 banked micro-edits | −5.582e-6 | 0.058% | MEASURED | **cp135** (superseded ×3) | INSTANCE | **NO** — 4 independent grounds (fb1 §B.2) | $0.16 + a rebuild + a *re-search* | MAIN |
| 6 | sr1 FO-2 — waterfilled correction channel | −5.95e-4 | 6.20% | **CEILING** (ideal conditional-entropy limit, no coder) | hv1 | INSTANCE | 4 KB, composes in principle | $0, after FO-1, **+ a tool flag that does not exist** | MAIN |

### Row 1 — the only real one, in detail

`ra2` raced the **inner** CPR1 coder, which nobody had touched (`mp2` raced all 12 Brotli qualities
and tied at 0 B — that races the *outer* coder). Measured, with a real adaptive arithmetic coder and
a control requiring the shipped encoder to reproduce the shipped blob byte-for-byte:

| stream | shipped | best adaptive | result |
|---|---:|---:|---|
| basis (27,648 sym) | 12,277 B | 12,046 B | **+231 B** (+32 B dead table) |
| coefficients (7,200 val) | 9,878 B | 10,293 B | **−415 B — Rice WINS, do not swap** |

Net **+263 B raw**, **~230 B** after Brotli. `dc1` (independent, same day) reached the same place
from the other direction: *"that is the only section with any measured slack."*

- **Why it is real:** lossless ⇒ `d_seg`/`d_pose` exactly unchanged ⇒ no scorer, no advisory row,
  no axis-transfer question. It clears the `−3.5e-6` admission bar by **44×**.
- **The honest caveat, from ra2 itself:** the 230 B is Brotli measured on the *canonical* CPR1 form
  (22,307 B); the archive ships a *compacted* body (22,219 B) that `materialize_cpr1` expands. The
  repack was not reproduced. The uncertainty scale is the 117 B gap between ra2's Brotli-of-canonical
  (22,278 B) and the shipped 22,161 B. **The raw 263 B is solid; the realized 230 B is PROVISIONAL.**
- **The blocker is a judgment call, not physics.** `ra2` row 3: *"fire only when a rung of ≥2 KB is
  also in flight — 230 B does not justify a receiver change alone."* **I checked: nothing ≥2 KB is
  in flight.** wd2/wd3 closed (family negative), mp2 closed (family refused), mz1 closed (0 B),
  carrier rank closed (`ra2c`/`ra3`), token drop closed (`td1`/`rc4`), `dc1` closed the token stream
  at +5 B. **By its own gate this row never fires.** That gate should be retired: the bundling
  rationale assumed a companion rung would appear, and the measured record now says none will.

### Row 2 — the only sealed, dispatcher-validated order in the pile, and it is a bad buy

`ps1u` built, proved and *dispatcher-validated* a sealed T4 request (sha `9863de20…`, 6,217 B) and
fired nothing. This is a genuine class-5 row: the order exists, the candidate archive exists, the
runtime bundle exists, and no one pulled the trigger. Priced honestly it is still a bad buy:

- The rate leg is a **measured cost**: +588 real bytes = **+3.915e-4 S**, paid before any pose gain.
- The seg leg is **ASSERTED decode-identical, never measured** (`seg_leg_measured = false`). ps1u
  labels this correctly and says any T4 seg drift is signal.
- `pv1` (10:22, after ps1u) drew the inference nobody had: **ps1u's converged pose floor is
  `1.286e-05` on the advisory decode, while the shipped CUDA decode already sits at `6.88e-06` with
  ZERO carrier optimization — the CUDA decode is `1.87×` better than the solver's converged answer.**
  The solved deltas correct an error that is largely absent on the object that ships. `pv1`
  pre-registered the falsifiable prediction: **ps1u r2 REFUSES on the CUDA axis.**
- **The order also has a measured false-admit window.** Its "equivalent" rule is stated off
  `d_pose = 6.885642960696714e-06`, which is the **cp135** value at 186,252 B, not hv1's `6.88e-06`.
  A T4 row landing in **[6.245822e-06, 6.251199e-06)** ADMITS on the sealed rule while **RAISING** S.
- **FIRE-ORDER:** do **not** buy it as a win. If MAIN wants the transfer number, fix the false-admit
  window first (re-derive the threshold off `6.88e-06`), then treat the $0.16 as an *information*
  purchase with a pre-registered expectation of REFUSE. With $1.38 of Modal left and one detached
  T4 dispatch already running (`ddm_sa1_modal_t4_sign_gate`, pid 63260), I would not spend it here.

### Row 3 — the cheapest ticket, and it is not a win

`sr1` FO-1 is $0, local, ~800 s per α, with a proper positive control (α=0 must reproduce 34,938
flips exactly) and pre-registered bands. Its ceiling is 2.98× the gap. **Its sign is unmeasured** —
two scorer-free proxies failed, and sr1's own prior is *"near-neutral, either sign, huge upside,
zero cost."* **It is a coin flip with a free ticket, not a win**, and it must not be banked as one.

**Its stated blocker is DISCHARGED.** sr1 wrote *"I did not run it; the scorer slot is held by pid
4832."* I checked: **pid 4832 does not exist.** The only live jobs are a detached Modal dispatcher
and two pollers; no local scorer holds the lane. This is the clearest WON-THEN-STRANDED case in the
inventory — the prerequisite was met and nobody re-checked.

### Row 4 — the largest retained byte credit, and why it is still not a win

A sister sweep of both SSD tiers (692 `retained*` directories enumerated; **nothing unreachable**,
no eviction observed despite VertigoDataTier at 893 MiB free) found the biggest omission in my first
pass. `ddm_mp2` built **nine** generations against the correct hv1 archive/runtime pins and scored
only **four**. Five are receiver-closed, retained, and carry a literal `NOT MEASURED` in every score
column:

| generation | bytes | Δ vs frontier | rate credit | %gap |
|---|---:|---:|---:|---:|
| keep62 | 182,011 | −748 | −4.981e-4 | 5.19% |
| keep50 | 181,694 | −1,065 | −7.091e-4 | 7.39% |
| keep37 | 181,235 | −1,524 | −1.015e-3 | 10.57% |
| **keep25** | **180,708** | **−2,051** | **−1.366e-3** | **14.23%** |
| keep75−keep87 | 182,734 | −25 | −1.665e-5 | 0.17% |

**This is 14.2% of the gap in real, correctly-pinned bytes, and nobody has measured it.** That is a
far bigger unfired object than anything else in this inventory, and it is why I am reporting it even
though I recommend against scoring it.

**Why it is still not a win.** `mp2`'s three scored siblings all failed on pose, and the adjudication
closes the family with a measured dose-response: *"even −25 B costs +0.0362 S net; pose damage per
byte removed ~1.4e-3 S/B vs rate value 6.66e-7 S/B (~2000× over)."* Pricing keep25 against the only
monotone pair we have (keep87 −130 B → pose leg +0.044296; keep75 −471 B → +0.041366, slope
**−8.59e-6 S/B, the favourable direction**) projects a keep25 pose leg of **+0.0278** against a
**+0.0014** rate credit — net **+0.0264 S, 2.8× the gap in the wrong direction**, needing a **20×**
further pose reduction to break even. `mp2`'s own DEAD-ENDS say it: *"Blindly scoring
keep62/50/37/25 is closed until the pose-sensitivity prerequisite exists."*

**The honest counterweight, and it is why the row stays open rather than closed.** The trend is
favourable and it is `n=2`: pruning *more* improved pose from keep87 to keep75. `mp2` registered
this as a live hypothesis — *"pose-null or pose-positive FiLM rows likely exist."* The named
prerequisite is a **$0 local per-row finite-difference pose-sensitivity probe**, not a scorer run.
**FIRE-ORDER: run the FD probe, never the n600 score.** If pose-null rows exist, a re-selected
candidate at keep25's byte scale is worth 14% of the gap; if they do not, the family closes with a
second measured leg instead of a projection.

**An apparatus defect is part of why these are unscored, and it is already fixed but not restarted.**
`launch_detached_process.py` appended `.done` to a name already ending in `.done`, producing
`.done.done` receipts the queue could not observe; `ADVISORY_QUEUE_STATE.json` still labels keep75
`RUNNING` and **no keep62 attempt directory was ever created.** The class was fixed by `0286280f95`
— **the queue was never restarted.** That is a silent-control-plane instance
([[the-control-plane-fails-silently-make-the-silence-loud-at-launch-20260816]]): the science reason
(pose prerequisite) and the plumbing reason (dead queue) coincided, so nobody noticed the second one.

Also retained-and-unscored, reported for completeness and **not** counted as candidates: `mz2`'s 11
archives (**permanently unscoreable in place** — `mp2` closed it: they are not candidate-bound to the
hv1 pins), `pz4`'s ~67 archives (cp135 base, family folded after direct-v6's `+2.4715 S` pose
collapse), `lr2`'s 16 carrier bitstreams (sweep intermediates, no scorer contract), and ~163 analysis
payloads across six 08-16 arms (`hm1`, `dc1`, `ra1b`, `rc4`, `ra2crr`, `cl2`).

### Row 5 — the bank, confirmed dead independently

I re-derived `fb1`'s finding rather than quoting it. `qs2 + re1 = −5.5817878492e-06` = **55.8%** of
its own 1e-5 naming bar, so *waiting for the pool to grow cannot satisfy the rule*. Not composable
onto hv1: hv1's admission proof requires exact decoded-token **and** raw-output byte identity, which
either edit breaks; `re1`'s probability object no longer exists on hv1's HPAC model (a re-*search*,
not a port); and `qs2` carries a compensation solved against a base whose `d_pose` differs by
`3.4009e-06 S` — **60.9% of the entire pool.** `fb1`'s retire-the-rule verdict stands. I add nothing
except confirmation.

---

## C. SPENT, REFUSED, OR DEAD — with the traps that are still live in writing

These are the rows that *look* like wins in a document someone will read tomorrow.

| item | advertised | actual | status |
|---|---|---|---|
| **mp2 keep75−keep87 differential** | `hv2` rank 3: *"built, −25 B, unscored"*, −1.7e-5 S | **FIRED and REFUSED at +0.0362 S** (pose 3.77× base). Adjudicated `2026-08-16T05:20:06Z` | **hv2's row is STALE — a live re-banking trap** |
| **pz2 pose-carrier direct replacement** | `hv2` rank 1: −0.0135 S = 141% of gap, *"UNOWNED ← the finding"* | **`pz5`: STAGE_0_REFUSED — DO NOT BUILD.** FAMILY scope | dead; see §D for my own retraction |
| mz2 q3/q4 + FiLM keep87/keep75 (retained payloads) | −823 B / −130 B / −471 B | all three FIRED and REFUSED (+0.0467 / +0.0443 / +0.0414). **Family CLOSED with a measured dose-response** | spent |
| `mh1_orphan_int8_posthoc` | *"highest-value in the harvest"*, d_seg −4.301114e-4 | v7.5.2 **witness** vehicle; TR1 gate; TR1 is `S≈0.75 @ 357,836 B` | unfireable as written |
| `mh1_orphan_chroma_hf`, `_lane_skipband`, `_bayes_floors`, `_seg_rd_secant` | measured directions | all TR1/witness-gated | unfireable as written |
| `mh1_recover_blind_coordinate_generic_fill_401` | *"vehicle-independent, no re-derivation gate blocks it"*, ~230,904 blind px/frame | **NEW KILL (this unit)** — see below | premise false on this vehicle |
| #826 FIRE-ORDER-0 `gr1_cell_drop50` | −0.0983195 | re-priced by `op3` against the live best: **+0.0034632 — an inversion** | spent |
| `ra2` row 4, token-stream alphabet test | $0, largest section | **DISCHARGED by `dc1` the same day: +5 B (a LOSS)** | closed |
| `fp_shrink_qat` (−0.022 to −0.029 S), `#807` (−0.044), `PDW1` (−0.0170) | large bands in the open task ledger | bc20/witness/v9 vehicles, 2026-06/07 | ancestor numbers, [[L18]] |
| eu4 (−1.122e-5), pb2 (−3.745e-5), js7, vp1 ANS (−2,120 B) | projected/measured negatives | cp135-superseded / v4d 360,323 B / refuted on exact CUDA / won against PR130's *Range* coder | do not re-propose |

**NEW KILL — `mh1_recover_blind_coordinate_generic_fill_401`.** This row claimed to be the one
mh1 orphan with *no* re-derivation gate. It has one. `td1` measured at source that **the token field
is exactly the scored population — 600 × 384 × 512 = 117,964,800, one token per scored pixel** — and
`pz5` parsed the archive header: tokens 112,110 B, semantic model 34,763 B, carrier 22,161 B, HPAC
13,515 B, residual 96 B, header+ZIP 114 B. **Nothing in the archive stores camera-resolution pixels.**
The blind camera pixels already cost **zero** bytes; the renderer generates them. The lever was real
on a per-pixel vehicle and is worth **0 B** here. A weaker residual claim survives — that generic-
filling blind regions might let the *renderer* be smaller — but that is a different, unmeasured
mechanism and must not inherit the 230,904-pixel headline. Scope: INSTANCE (this row, this vehicle).

---

## D. ATTACKING MY OWN CONCLUSION — one finding I retract

**I found the right point on a curve that does not exist.** Acting on `hv2`'s own hard gate (*"the
1,817 B must be re-derived from the receipt before any launch is routed on it"*), I opened
`PZ2_MEASUREMENT_RECEIPT.json` and re-priced all **919** candidates against hv1's components. The
arithmetic said `hv2` had picked the wrong Pareto point: minimum-bytes (1,817 B, qMSE `2.329e-05`)
is the wrong objective when the pose term is a square root. The joint optimum is
`direct_p138_b14-8-8-7-7-8` at **3,759 B**, qMSE `3.110e-08`, giving a perfect-realization
**S = 0.13960724** — `−0.01999` vs hv1 and **0.01039 below 0.15** — robust across every plausible
carrier size (20,000–23,384 B).

**That is refuted, and I am retracting it as a route.** `pz5` read `cpr1/inflate.py::render_video`
at source: the 22,161 B carrier **renders frame_0** (`einsum("bk,kchw->bchw", coefficients, basis)`,
bicubic to 874×1164), while the pz2 packet stores **PoseNet's six output scalars**. Targets are not
images, and no receiver in the archive converts one into the other. `pz2` labelled this itself
(`frame_parity: NOT_RUN_NO_FRAME_REALIZATION`, projection axis literally `[TOY-BRACKET … no
receiver/scorer]`). The `−20,524 B` was never a rate saving — it is **the price of an unbuilt
receiver, quoted as if the receiver were free.** My re-pricing inherited exactly that premise.

What survives is a **CONDITIONAL SPEC, not a row**: *if* someone ever builds `pz5`'s item-4
target-conditioned frame_0 **generator**, the packet should be sized at ~3,759 B, not 1,817 B —
quantization then consumes 0.26% of the pose budget and leaves `1.196e-05` of `d_pose` headroom for
realization error. For calibration: the one realization ever attempted (`pz4r` direct-v6) measured
`d_pose = 0.631014` — **52,760× over that headroom.** The representation side is solved with room to
spare; the entire problem is realization. Nobody should spend another byte on the packet.

**Other ways this unit could be wrong.**
1. **My "one row" claim is a negative-existence claim over a scope I did not exhaust.** I searched
   `.omx/research/*.md` (7,304 top-level), `arm_final_messages/` (343), the 252-row arm queue and the
   561-row task ledger. Honest wording: **I did not find a second measured unfired win in those
   scopes.** A bar or credit phrased as a percentage, or living only in the 454 gitignored
   `.omx/tmp/codex_runs/` finals, is invisible to me — and `fb1` recorded the same blind spot today.
2. **Row 1's realized 230 B is not mine.** I verified ra2's method, control and honesty at source; I
   did not re-run the coder. If the repack measurement lands below ~113 B the row drops under 1% of
   the gap, though it stays above the admission bar. It does not change the verdict.
3. **The `%gap` framing flatters small rows.** 1.6% of the gap sounds like progress. In bytes it is
   230 of 14,413 — **1.6% of the distance, and the remaining 98.4% has no measured candidate at all.**
4. **My ps1u pricing rests on pv1's advisory-to-CUDA reasoning, which is DERIVED, not measured.** The
   falsifier is the T4 row itself. I am recommending against buying the falsifier, so I am recording
   an unfalsified prediction as such rather than as a fact.

---

## E. THE ONE CURE WORTH LANDING (apparatus, $0, not this arm's to land)

Three arms recomputed "the bank" by hand today and all three were obsolete within hours. The bank is
prose. **Make the bank a file.** A JSONL of admitted credits — `{arm, ΔS, level, base_sha, scope,
composable_onto, fire_state}` — appended by any arm that measures one, with the total computed at
read time and every row carrying the base it was measured on. Then:

- `ra2`'s credit is visible the moment it lands instead of two hours after three aggregators finish;
- a credit on a superseded base (qs2, re1) is *structurally* prevented from being summed with one on
  the live base, which is the double-count `fb1` caught by hand;
- "does the pool clear 1e-5?" becomes a query, not a day of arm-time.

This is the [[m45]] own-everything law applied to credits rather than tasks, and the sister of
`fb1`'s stale-bar cure: **quote the invariant, compute the total.** Owner: an apparatus arm.
Blocker: Catalog #299 quota discipline — it should extend the existing ledger surface, not become
orphan-tracking surface #17.

---

## F. DENOMINATOR — what I actually scanned

| surface | enumerated | inspected |
|---|---:|---|
| `.omx/research/*.md` (top level) | **7,304** | grep-swept ×12; **~30 read** in whole or substantial part |
| `.omx/research/**` recursive `.md` | 8,759 | grep-swept |
| `.omx/research/arm_final_messages/*.md` | **343** | grep-swept in full; 4 read |
| `.omx/research/charters/*` | 128 | listed, not read |
| `.omx/state/codex_arm_queue.next_if_resumed.jsonl` | **252 rows** | **all parsed**; 138 in the ≥08-10 live window; **40 QUEUED-WITH-A-FIRE-ORDER blocks extracted and read** |
| `.omx/state/canonical_task_status.jsonl` | **561 rows / 212 tasks** | **all parsed**; 111 open rows listed; 20 full records read |
| memos dated 2026-08-16 | 54 | 11 carrying sealed fire-orders inspected |
| primary receipts opened on SSD | 6 | pz2 (919 candidates re-priced), mp2 ×3, mp2 adjudication, mp2 fire order |

**Verified, not assumed:** the frontier components (recomputed, residual −4.1e-11); pz2's 1,817 B and
its qMSE at source; the mp2 differential's `contest_auth_eval.json` **existing** (my first `find` was
truncated and I nearly reported it as unscored — caught by re-reading the directory); `pid 4832`
absent; the task ledger's strict load (repaired — 559 served, **2 still UNREADABLE and excluded**:
`1079_pv1_…` and `1082_ddm_hm1_…`); `cpr1` absent from all three aggregator memos; the memo
timestamps that explain *why* it is absent.

**Retained-payload sweep (sister agent, both SSD tiers).** 142 `retained*` dirs on VertigoDataTier +
550 on APDataStore enumerated (**692**); 81 arm-level queue/fire-order manifests opened, 16 parsed in
full. **Nothing unreachable** — both volumes mounted, and despite VertigoDataTier sitting at 893 MiB
free / 100% capacity **no eviction was observed**; every path stat'd existed. Triaged by (a) archive
candidates near 182,759 B, (b) presence of a queue/fire-order manifest, (c) presence of a score
receipt. **Explicitly not individually adjudicated:** the pre-2026-08-09 cp135/e480b-lineage arms
(`sm3`, `sm4`, `sd1`, `sd2`, `ai1`, `cp2`, `hp3`) — enumerated only. `experiments/results/` (145 GB)
contains **0** `retained*` dirs; it predates the retention policy.

**What I could not reach.** The 454 arm finals that exist only in gitignored `.omx/tmp/codex_runs/`.
A full read of the 28k-line DAG (grepped only; it is a June v4d/v19b lineage). The
`lever_activation_ledger.jsonl` claim that *26 of 31 fire-now rows never fired* — the file is
**unmodified since Jul 27**, which corroborates that nothing drained it, but those levers are
witness/v9-vehicle and stranded exactly like the mh1 orphans. `kp2`'s census is the standing evidence
that retention is systematically incomplete (**272** genuine RECORD-class discards live at the
preflight scope), so a payload that was *never persisted* cannot appear in any sweep, including this
one.

---

## NEXT_IF_RESUMED

- **`QUEUED-WITH-A-FIRE-ORDER`** — owner **MAIN**. **Row 1, ra2 CPR1 inner coder.** Action: retire
  ra2's own "≥2 KB companion rung" gate (nothing ≥2 KB is in flight and the measured record says
  nothing will be), then re-measure the +230 B at the repack layer against the shipped compacted
  body, receiver-close, byte-close. **No scorer, no dispatch, $0.** Fire trigger: immediately.
  Admission: it is lossless, so the only question is the realized byte count.
- **`QUEUED-WITH-A-FIRE-ORDER`** — owner **MAIN**. **Row 3, sr1 FO-1.** Its stated blocker (pid 4832)
  is **discharged** — verified absent. $0, local, ~800 s/α, positive control pre-registered. Fire
  trigger: immediately. **Bank the result either way; do not bank the ceiling.**
- **`QUEUED-WITH-A-FIRE-ORDER`** — owner **MAIN**. **Row 4, mp2's four unscored deep prunes.** Action:
  run the **$0 per-row finite-difference pose-sensitivity probe** mp2 named as the prerequisite, and
  restart the advisory queue (the `.done.done` class was fixed by `0286280f95` and never restarted;
  `ADVISORY_QUEUE_STATE.json` still shows keep75 `RUNNING` and keep62 has no attempt directory).
  **Do NOT fire an n600 score on keep62/50/37/25** — mp2's DEAD-ENDS forbid it and the projection is
  +2.8× the gap. Fire trigger: immediately. Admission for any successor candidate: a re-selection
  built from measured pose-null rows, not a deeper blind prune.
- **`DEFERRED, blocker named`** — owner **MAIN**. **Row 2, ps1u sealed T4.** Blocker: the admission
  rule's false-admit window `[6.245822e-06, 6.251199e-06)`, measured by `pv1`. Re-derive the
  threshold off hv1's `6.88e-06` before any dispatch. Even then it is an information buy against a
  pre-registered REFUSE prediction, at $0.16 of $1.38 remaining. **My recommendation: do not buy.**
- **`QUEUED-WITH-A-FIRE-ORDER`** — owner **MAIN**. **Correct two stale rows that will be re-banked
  by the next reader:** `hv2` rank 3 (mp2 differential — FIRED and REFUSED at +0.0362) and `hv2`
  rank 1 (pz2 — `pz5` STAGE_0_REFUSED; hv2's own re-derivation gate is hereby discharged: REFUSED).
  Both are append-only corrections, $0. `pz5` already asked for the rank-1 fix; this is the second
  independent request.
- **`CLOSED`** — `mh1_recover_blind_coordinate_generic_fill_401`. Premise false on this vehicle:
  blind camera pixels already cost 0 B. Retire with the lesson, do not re-open.
- **`DEFERRED, blocker named`** — owner: an apparatus arm. §E, the bank-as-a-file cure. Blocker:
  Catalog #299 quota — extend an existing ledger, do not create orphan-tracking surface #17.

Own-vehicle frontier line: **hv1 ep0634 S 0.15959729295498598 @ 182,759 B `[contest-CUDA T4, n600]`
— UNMOVED by this unit.** This unit measured, re-priced, retracted one of its own findings, and
recommended against the only paid buy on the table. It did not lower the score and did not try to.
Modal untouched: **$18.62 / $20.**
