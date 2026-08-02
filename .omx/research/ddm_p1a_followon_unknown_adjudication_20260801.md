---
schema: ddm_p1a_followon_adjudication.v1
date_utc: 2026-08-01
arm: ddm_p1a (population 1 — the 86 UNKNOWN follow-on rows)
task_row: "#879"
lane_id: "lane_ddm_p1a_followon_unknown_adjudication_20260801"
research_only: true
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU advisory; $0 read-only adjudication over memo text + artifact existence; no scorer, no dispatch, no candidate, no pointer mutation]"
inherits: "ddm_fo1 (#870) — tac.followon_ledger; 102 rows / 0 ORPHANED / 4 STAGED / 86 UNKNOWN / 12 EXECUTED"
verdict_scope: "INSTANCE (this extraction window since=2026-07-18, this artifact scope)"
empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_p1a — the 86 UNKNOWN adjudicated: 29 distinct open items, not 86, and QA52 was already fired

## §0 Answer first

**The 86 UNKNOWN rows are 25 not-a-follow-on + 21 already-done + 40 real-debt rows that collapse
to 29 distinct open items.** The partition is exact and machine-checked (25+21+40 = 86, no
overlaps, no gaps; the 40 RD rows map onto 29 items with zero rows uncovered).

**QA52 is ALREADY-DONE — FIRED 2026-07-30 by `ddm_kl1`.** The seed's framing ("the one STAGED row
with a plausible runnable follow-on that no arm has claimed") is FALSE. `ddm_fo1` correctly wrote
"unadjudicated"; that became "unclaimed" in one hop. `experiments/ddm_kl1_pose_field_receiver.py`
is not an unfired runner — it is the arm's *product*.

**Three instrument facts that change how the fo1 numbers should be read:**

1. **All 86 UNKNOWN carry ONE reason — "no artifact-shaped join token", `joinable=False`.** The
   detector did not try these rows and fail; it declined to try. UNKNOWN here is a property of the
   EXTRACTOR's line-scoping, not a statement about the debt.
2. **The closure evidence was in the text the detector already read.** 32 of the rows name a
   canonical queue id whose STATUS cell in `ddm_deferral_queue_ledger_20260729.md` says FIRED /
   CLOSED-MOOTED / BUILT / MEASURED-DOMINATED. The detector joins only on filenames and ignored a
   closed-vocabulary status column sitting on the same line. Cheapest fix in the population.
3. **`ORPHANED` is structurally unreachable on this host for its main branch.**
   `/Volumes/APDataStore/pact` does not exist (only `VertigoDataTier` is mounted), so
   `artifact_scope_complete` is permanently `False` and every output-missing row degrades to
   UNKNOWN by construction. In *this* run zero rows reached that branch, so it did not cause
   `ORPHANED=0` — but the 0 cannot be read as evidence in either direction, and every future run
   inherits the block. fo1 said ORPHANED precision is "UNVALIDATED at n=0"; the sharper statement
   is that one of its two ORPHANED branches cannot fire here at all.

Pointer `0.1910828242 [contest-CPU]` **UNMOVED** (the borrowed harvest-only line). The own-vehicle
line moved today by another arm — v4d 0.9639878 → pw1 0.9476091, on the **pose** axis — which this
arm did not cause and does not claim. This arm is apparatus: it measured nothing about score.

---

## §1 The population, with its denominator

Reproduced exactly from `tac.followon_ledger.audit(since=date(2026,7,18))`:

```
ScopeLedger(surface='followon-extractor', examined=1184, declared=1184, population=7378,
            note='102 follow-on rows extracted')
{'ORPHANED': 0, 'STAGED': 4, 'UNKNOWN': 86, 'EXECUTED': 12}
```

1,184 memos examined of 7,378 in `.omx/research` — the `since` filter is doing most of the
narrowing, and **6,194 memos (83.9%) were never looked at**. Anything below is scoped to the
2026-07-18→08-01 window. Rows outside it are not adjudicated and are not claimed to be clean.

**UNKNOWN reason histogram (n=86):** `no artifact-shaped join token` — 86. One bucket, one cause.

---

## §2 86 rows are not 86 debts

| collapse | count |
|---|---|
| rows in scope | 86 |
| — NOT-A-FOLLOW-ON | **25** |
| — ALREADY-DONE | **21** |
| — REAL-DEBT rows | **40** |
| REAL-DEBT rows → distinct items | **29** |

**Why the rows over-count.** Two independent multipliers:

- *Re-citation.* 32 rows name ≥1 canonical queue id; they resolve to 31 distinct ids. QA03 appears
  4×, QA06 4×, QA11 4×, QA24 3×, QA02 3× — each convocation that re-listed the queue minted new
  rows for the same item. This is the detector working as designed (memo-scoped identity is
  correct), but it means the raw count is a citation count, not a debt count.
- *Line-scoping.* The extractor tests individual lines ≥25 chars. A wrapped sentence yields a
  fragment; a frontmatter `axis:` string, a verdict line, and a design matrix all match
  `ACTION_RX ∧ CHEAP_RX`. Nine of the 25 NAF rows are bare sentence shards
  (e.g. `PH-1, so Eqs. 3–5 do NOT transfer as-is. However (DERIVED, cheap owed`).

**Self-match.** Two of the 86 rows are `ddm_fo1`'s own memo — its H1 title
(`the orphaned-cheap-follow-on class…`) and a prose sentence about the class. A detector that
flags the memo *describing* the class as an *instance* of it is harmless but inflates the count;
worth a filter.

### The 21 ALREADY-DONE, by queue status (VERIFIED_VIA_SOURCE_INSPECTION, ledger status cell)

FIRED — QA02 (07-29, merge c53939477b) · QA03 (07-29 sb1) · QA04 (07-29 sb1) · QA05 (07-30 qp1) ·
QA06 Knee-A realized gate (07-29) · QA11 (07-29 sb1) · QA39 (07-29 xi1) · QA41 (07-30 qp1) ·
QA44 (07-30 pm1) · **QA52 (07-30 kl1)** · QA89 (07-31 pa1r) · QA24 (bc1 07-30, BUILT+FIRED
detached, pid 68621, git e0a37e82f4).
CLOSED-MOOTED — QA18 (zb1 07-30: consumer wr1 dead).
CEILING-PRICED-EFFECTIVELY-CLOSED — QA08 (uh1 07-29).
MEASURED-DOMINATED (INSTANCE) — QA07 (gr1 07-30).
BUILT — QA88 (hw1 07-30) · QA80 producer (b2p 07-30) + field pass RUN n600 (zb1 07-30).
PARTIAL — QA58: `ddm_v4c` §9 records the exposure race as `$0, FIRED`, while the ledger still
carries `OWED` for the **full-600 rung-B (a,b) fit**. Fired on the tail; the full-600 leg is the
residual, carried below as a real-debt item. *This ledger/memo disagreement is itself a finding.*

---

## §3 The 29 open items, ranked by COST-TO-FALSIFY

**Ranking discipline, and why it is not stylistic.** These are ordered by what it costs to make
the claim *false*, never by predicted ΔS. The population contains its own proof: **QA03** is cited
4× in my 86 rows; `gc7r` ranked it #2 with a booked band up to **−0.138 ΔS_seg**; the realized
measurement was **−0.001582 = 1.15% of the ceiling** — ~87× optimistic, re-derived here from the
ledger status cell, not inherited from `gc17`. A predicted ΔS is an unfalsified belief being used
to order work, and it compounds: the top-ranked row gets built first.

Every item carries its source memo date. **Rows dated 2026-07-19…07-25 predate the current
own-vehicle line** (MEMORY: box retired 07-28; own-vehicle v4d 0.9639878 measured 07-31), so
several target lineages MEMORY records as retired or rate-dead. I flag these `VEHICLE-SCOPE-OWED`
rather than silently ranking them — I did not verify each against the live vehicle, and doing so
is itself the first measurement for those rows.

### T0 — a READ, no run (minutes)

| # | item | rows | evidence / next measurement |
|---|---|---|---|
| 1 | **φ / 1-over-φ efficiency multiplier from the composite-R adjoint** | gc15:390, gc15:466 | gc15's own words: *"$0 but needs the composite-R adjoint from the ms4d bundle — a read, not a run"*. It is UNMEASURED and **gates arms D±**. Lowest cost-to-falsify in the entire population. Next: read the adjoint out of the ms4d bundle, report 1/φ. |
| — | *(instrument, not debt)* **read the ledger STATUS cell** | n/a | Would have resolved 21 of 86 rows at zero marginal cost. See §6. |

### T1 — $0 local, cached data, minutes

| # | item | rows | evidence / next measurement |
|---|---|---|---|
| 2 | **LP1 G4 same-object context price for the v15 stream** | blindspot:130/161/173 (07-25) | Existence proof measured: LP1 G4 context beat explicit bytes by **89,161 B with 0 counted context params**. LP1 *explicitly refuses transfer without the same-object price*. Memo says `$0 local encode — do first`. VEHICLE-SCOPE-OWED (07-25). |
| 3 | **tw1 greedy-under-joint-remeasure knee** ★ | tw1:324 (08-01) | The re-priced knee — see §5. tw1: *"My harness already exposes `state_bytes` over arbitrary state sets, so this is a driver, not new machinery"*; ~1 encode/candidate at 1–5 s. Newest row in the population and the only one on the live vehicle. |
| 4 | **W1-COH Yousfi flicker-Bayes-floor preflight + #535 Fisher-spectrum** | cn3:451, gc14:339, gc14:340, ub1:82 | Both `$0` from **existing** W1-COH receipts. gc14 states the #535 measurement *"should be queued regardless of branch, because it is the cheapest way to convert a never-fired design into a priced one"* — and `ub1` (08-01) independently re-flagged that gc14 already said so. Cited 4×, never fired. |
| 5 | **g4 boundary-gated code-width `$0 H(cell\|neighbors)` gate** | lv1:323, tb1:181 | Named a trainer DUTY row (`boundary_gated_token_code_width`, "feeds G4"); zero-payload decoder-derived contexts are the CAE context source. |
| 6 | **gc13 R1 endpoint consumption bundle** | gc13:343 (07-31) | `$0`, gated on *"scorer slot freed at burn4.done"*. **The gate is open**: no literal `burn4.done` marker exists, but `burn4_endpoint_decision_MAIN.json` (2026-07-31T18:35Z) records *"NO WINDOW_04 — window_03 is the last training window"*. Bundle = xp1 per-class re-measure · 5×5 pair flip matrix · protected per-class descent rates · Undriv-erosion typing. |
| 7 | **fp1 component-erasure labels** | fp1:125 (07-31) | *"need the endpoint realized argmax (one scorer pass, NOT run per the `$0` fold)"* — exactly one scorer pass. |
| 8 | **QA58 residual: full-600 rung-B (a,b) fit** | deferral QA58, v4c:193 | Tail leg FIRED; the full-600 fit then races AR(1)/spline vs raw byteplane. Also resolves the ledger/memo status disagreement noted in §2. |

### T2 — $0 but 30–60+ min

| # | item | rows | evidence / next measurement |
|---|---|---|---|
| 9 | **pn1 S1 dress-rehearsal Stage A** | fu1:19, pn1:53 | `$0` local full-n600 on the eg1 packet, ~35–60 min. *"the chain has NEVER run end-to-end."* **`ddm_fu1` already stamped this ORPHANED on 07-30** — a prior sweep adjudicated it and it still has not fired. Falsifier is pre-registered: Stage-A d_seg disagreeing with tb1 full-confirm 0.013833 beyond the drift band ⇒ deploy-parity bug found. |
| 10 | **gc5 B2: n600-full flip-amplitude confirmation + first real coder bytes** | gc5:153 (07-28) | Measured on 60 pairs / 100,596 sites (median 1.11 uint8 steps, 64% ≤2). Two named rungs: n600-full confirmation before the 10.06 MB→low-single-MB reprice is citable, and no coder has emitted bytes yet. n600 discipline applies. |
| 11 | **pn1 Higham deploy-parity drift hardening** | pn1:59 | `$0`, part of byte-close; eg1's 63/3,052,008 camera-byte MLX↔NumPy mismatches are uint8 tie-rounding. Risk is TRAIN↔DEPLOY realized-d_seg shift. |
| 12 | **QA54 per-channel / luma-only exposure gain** | ba31:386, ja1:52 | `DUE (defer-at-source: ddm_pm1)`. |
| 13 | **QA48 plane+parallax (Irani-Anandan) + layered motion** | pm1:169 | `DUE (defer-at-source: MAIN, ph2)`. |
| 14 | **QA40 (ξ-advected temporal-innovation probe) + QA47 (pose-steering shared basis)** | fu1:25 | QA40 `DUE-LOW`; QA47 `DUE` **with a precondition**: ps1 07-30 requires a pose-conditioned base — on the seg-native parent the field is walled (solved d_pose 20.41 n600). Do not aim QA47 at the wrong base. |

### T3 — build or run required

| # | item | rows | evidence |
|---|---|---|---|
| 15 | **KD-from-warm-into-fresh (`kd_warm_start_dir`, #74/#129)** | gc15:303 | **BUILT, 6 NO-FAKE tests, DEFAULT-OFF, never fired on tr1.** Verified: `src/tac/torch_vehicle/kd_warm_start.py` + tests + 2 launchers reference it; **no `.sh`/`.json` in the repo tree names it** (SSD run-config tier NOT exhaustively searched — see §7). This is the *default-off orphan* class from CLAUDE.md, and the "off" carries no recorded reason. |
| 16 | **Structured warm reset (#725)** | gc15:298 | *"NEVER RUN; #725 built, ms4d complete"*; `$0` build on existing artifacts. gc15 ranked this **#1** — per the QA03 arithmetic above, treat that rank as a belief, not evidence; its real merit is that the build cost is already paid. |
| 17 | **QA62 pm1 rungs A+B fold into v4c** | deferral QA62 | `DUE-AT-v4c`; A+B measured −0.1039 S on its own basis. |
| 18 | **QA25 v10 SPEC amendments (pose-in-burn charter)** | deferral QA25, gc7r:164 | `DUE (doc)`; pfs1 D2 07-29 supersedes the p3v2 framing — pose-in-burn is REQUIRED. |
| 19 | **QD04 registry + disk hygiene** | deferral:100 | `DUE`: 6 owed lane-registry rows via `lane_maturity.py`, 8 prune-eligible + 13 dirty worktrees certify-or-block. |
| 20 | **QA32 tb1 registered DUTY_TO_MEASURE levers** | deferral QA32 | `HELD (registry)` — a *set* of never-fired levers incl. MarginBandSatisficing #459; the activation-ledger class. |
| 21 | **zb1 post-snap continuation-knee re-measure** | gc10:72 (+:98) | Cheap, rides gc10 row 1; marginal price 444→1332 B/1e-4 was measured on the UN-snapped SMEVR stream and **re-prices** if row 1 banks ~40% null deltas. Same state-dependence physics as item 3. |
| 22 | **gc15 Q2 named `$0` measurement (fresh-asymmetry premise)** | gc15:101 | Self-assessed *"PLAUSIBLE-with-named-`$0`-measurement — and yes, mildly overrated"*. |
| 23 | **rg3 bit-allocator `score_units_per_byte_status`** | rg3:48 (07-24) | Deferred until a byte-closed score measurement exists; gated, not runnable alone. VEHICLE-SCOPE-OWED. |
| 24 | **c2 integer-plane-emitter rate/receiver custody** | c2:139 (07-19) | *"the current smoke is not byte-closed."* VEHICLE-SCOPE-OWED — the C2/description lineage is what MEMORY records as plane-family rate-dead. Adjudicate the vehicle before the item. |
| 25 | **r1b3 `E_n600` producer** | r1b3:22 (07-20) | `NOT RUN`. VEHICLE-SCOPE-OWED (pre-arc). |
| 26 | **QE10 / R15 — Martin-Löf randomness battery + FRI innovation-rate audit** | deferral:126, round3:264 | `HELD`, card-queued post-refoundation; sequencing already honored. Not orphaned — parked with a gate. |
| 27 | **OP-R3-6 rule-118 adjudication memo (COIN++ pretrained receiver)** | round3:50 (07-25) | `$0` memo, but R16 is COMPLIANCE-CONTESTED and the pretrained-receiver lineage sits against the no-old-lineage ban. Lowest value in the list; adjudicate cheaply or retire the row. |

### T4 — costs money

| # | item | rows | evidence |
|---|---|---|---|
| 28 | **Modal `gpu="T4"` smoke** | ua2:419 (07-31) | ~**$0.20**, and the smoke is **already staged** by #214 deliverable (c) — built, never fired. Converts the entire CUDA-axis ladder from PROJECTED to MEASURED (the ~98× T4 multiplier is projected, never run). Best paid-per-dollar row in the population. |
| 29 | **blindspot op-routable-4 `$0` batch** | blindspot:38 (07-25) | Five sub-items (v19c tail-exponent fit; margin-mass N(δ); predictor-stream deletion ablation batch32; LP1 G4 price — *duplicates item 2*; gc1 CONNECTION probe). Partly subsumed; VEHICLE-SCOPE-OWED. |

---

## §4 QA52 — adjudicated: ALREADY-DONE (FIRED 2026-07-30)

**Verdict: ALREADY-DONE.** Re-derived from two independent primary artifacts, neither of which is
the fo1 memo:

1. **`ddm_deferral_queue_ledger_20260729.md#QA52` (L85), status cell:**
   `**FIRED 07-30 (ddm_kl1; memo ddm_kl1_law_plus_noise_20260730.md): RATE HALF measured-FALSIFIED
   …**`
2. **`ddm_kl1_law_plus_noise_20260730.md` frontmatter:** `fires: "ledger QA52 (xi-trajectory
   coding); + new QA55 (container recompression), QA52-b defer"`, and
   `experiments/ddm_kl1_pose_field_receiver.py` is listed under that memo's own `tools:`.

**What was measured (kl1, `[macOS-CPU advisory]`, `score_claim=false`):** the ξ-trajectory premise
is **FALSIFIED**. The 600×6 pose field is temporally **WHITE** —
`std(diff)/std(value) = 1.14` (dim0) … `1.40` (dims 1-5) — not a smooth vehicle trajectory.
Poly/delta/spline predictors *lose* to distributional byte-plane coding. Lossless floor **5,948 B**
(byte-plane colmajor brotli), not the hypothesized 1–2 KB (that figure was the LOSSY rank-1 →
d_pose 0.207, already in the d2 ladder). The se(3) B-spline (`tac.lie`, never fired here) is the
**wrong chart for this field's rate**.

**What was delivered:** a real, bit-exact verified codec — `pose_warp.stp` 6,844 → 5,964 member,
= −880 B member / −687 B payload vs the .stp's internal row-major-brotli 6,635. **#404: −0.00059 S,
zero d_pose risk.**

**Why the detector said STAGED and why that was wrong.** `classify_execution` saw a runner token
(`ddm_kl1_pose_field_receiver.py`) present with no output token, which is the built-never-fired
signature. But the file is the *product* of the executed follow-on, not its unfired runner — and
the word `FIRED` was on the same ledger line the extractor read. The runner-vs-output distinction
is sound; it is defeated when an arm's deliverable *is* code.

**The residue, correctly located.** QA52-b (dynamics-regularized GN re-solve) was deferred to
**QA57**, whose status is `DEFER (defer-at-source: ddm_kl1 → MAIN/pi2; strongly-predicted negative,
not worth confirming here)`. That is a reasoned defer with a named owner, not an orphan.

**A REAL open item QA52 did leave, which nobody has raised:** the verified −880 B codec is
**built and unwired**. Repo-wide grep (positive + negative control, both passed):
`ddm_kl1_pose_field_receiver` appears in exactly **4 files** — its own source and three memos
(`ddm_fo1`, the deferral ledger, the kl1 memo). **No exporter, builder, or packet path imports
it.** Per `built_elsewhere_unwired_is_p0_20260801` that is the P0 grade-5 class. Its sister
**QA55** (container/member lossless recompression — the ck1 composed archive stores all members
`method=0`) is `DUE` and is the natural recipient. Cost-to-falsify: minutes to check whether the
real builder already deflates members.

**On the seed's framing — the motivation was sound and I verified it.** Checked against
`.omx/state/main_hot_state.md` (not taken on trust): the own-vehicle frontier **moved today**,
v4d 0.9639878 → **pw1 0.9476091**, and `pose 0.2765059 (sqrt(10*0.00764555)) dS -0.0164351 <- the
entire win`. So pose is the live axis and #741's ξ dual-use stands. The claim that failed is
narrower and worth naming precisely: **"unadjudicated" (fo1's word) became "unclaimed by any arm"
in one restatement.** That is the confidence-laundering step from §5 of the operating manual, and
it cost this arm's first hour. fo1 wrote it honestly; the compression did the damage.

**Consequence for the ranking.** Because the win that moved the own-vehicle line today was on the
*pose* axis, the QA52 residue matters more than its size suggests — not the falsified trajectory
prior, but the **unwired −880 B codec** and its recipient **QA55**.

---

## §5 Scope guard applied: the wr1 knee

**Applied, not just noted — and it is stronger than the seed states.** The two knees fo1 rejected
are `verdict_scope: INSTANCE`: Knee A **+0.153086**, Knee B **+1.055658**. Re-derived from the
ledger QA06 status cell, the Knee-A gate measured rate 0.1827 (−0.197, exact as predicted) but
d_seg 0.00553676 and d_pose 0.28002128 → realized S **2.4097 vs ref 2.2566 = +0.153 net**. Matches.

Two independent reasons these do not carry forward as a family verdict:

1. **`ddm_tw1` (08-01) measured a state-dependent byte price**, so a re-priced knee is a different
   object — the seed's guard, and item 3 above is exactly that measurement.
2. **The INSTANCE reject was already OVERTURNED on 07-29 by `ddm_ck1`**, which the ledger records
   in the same cell: on the tail-112, two-plane wins **96/112 (86%)**, tail best_mean 0.3609 ≈
   full-base 0.3692 (**recovery parity 0.98×**) — *"the 486 dropped cells cost pose ~nothing once
   re-solved; the gate's +0.185 S pose damage was ENTIRELY stale params"*. Composed (Knee-A tokens
   + two-plane re-solve, GT-mask UB) beats ref 2.2566 by **−0.270 to −0.616 ADVISORY**.

So the honest state is not "reverse-waterfill rejected". It is: **rejected at INSTANCE with stale
pose params, overturned on the tail, full-600 sweep owed, and now due a third re-pricing under
tw1's state-dependence.** Sister #841 holds: wr1 published its descent table without its own
ceiling column and the ceiling reverses past the recommended knee — so any re-priced knee must
carry the ceiling column.

---

## §6 Defects found in the inherited instrument

| # | defect | severity | fix / cost |
|---|---|---|---|
| D1 | **The ledger STATUS column is ignored.** Rows sourced from a table with a closed status vocabulary (DUE/ORPHAN/HELD/FOLD/FIRED/SUPERSEDED-PENDING) carry their own closure evidence in-line. | HIGH — it is the difference between 86 unadjudicated and 21 resolved-for-free | Parse the status cell for ledger-sourced rows before the artifact join. Minutes. |
| D2 | **`ORPHANED` unreachable for the output-missing branch** while `/Volumes/APDataStore/pact` is absent — and that tier does not exist on this host at all, so the state is permanent, not transient. | HIGH (silent) | Distinguish *"declared tier absent"* from *"declared tier unmounted"*; a fallback tier that is optional by design should not gate completeness. |
| D3 | **Asymmetric tier-gating inside `classify_execution`.** The output-missing → ORPHANED branch is gated on `artifact_scope_complete`; the **runner-missing → ORPHANED branch is not**. The same argument applies to both. | MEDIUM | Gate both, or document why the runner case is exempt. |
| D4 | **Line-scoped extraction** yields sentence shards, frontmatter `axis:` strings, verdict lines and design-matrix rows. 25 of 86 are not follow-ons. | MEDIUM (inflates the count) | Require the line to parse as a table row or a bulleted action; drop frontmatter blocks. |
| D5 | **Self-match:** the memo describing the class matches as an instance (2 rows). | LOW | Exclude the detector's own memo family. |
| D6 | **Runner-vs-output fails when the deliverable IS code** — QA52's receiver is a product, read as an unfired runner. | MEDIUM | When a runner token is co-located with a FIRED status or a `tools:` frontmatter listing, prefer that evidence. |

None of these make fo1's report wrong. Its headline — *"measured much smaller than claimed…they get
drained fast"* — is **confirmed and strengthened** by this adjudication: 46 of 86 rows were never
debt at all, and the real backlog is 29 items, most of them parked with named gates rather than
forgotten.

---

## §7 Round-1 adversarial review of my own work

Attacking my own conclusion, per §6 of the operating manual.

1. **My first grep failed silently-ish on zsh globbing** (`--include=*.py` was glob-expanded and
   the command errored). I caught it because the output was an error, not an empty result — had it
   returned empty I would have had a false negative-existence claim. Re-run with quoting **plus a
   positive control** (the file's own path) and a negative control (a token that cannot exist);
   both passed. This is the exact failure class the seed warned about, and it fired on me.
2. **My ledger-status parser reads the LAST cell as status. That is wrong for rows with extra
   columns** — QA24 (11 cells) and QA80 (13 cells) returned mid-row prose. I detected this by
   printing full rows and re-read both by hand. Any status I quote for a row I did *not* print in
   full inherits this risk; I printed QA52, QA06, QA24, QA80, QA03, QA43, QA47, QA55, QA57 in full.
   **The remaining ~22 statuses are from the last-cell parse and are UNVERIFIED at cell level**,
   though all are short status-vocabulary strings, which is weak corroboration.
3. **My ALREADY-DONE verdicts trust the ledger's self-report.** A row saying `FIRED` with a receipt
   path is testimony, not an artifact check — I verified receipts for QA52 only. The damaging
   direction here is a false ALREADY-DONE (it deletes real backlog), so treat the other 20 as
   *candidate*-closed pending a receipt check. That check is cheap and is the natural next arm.
4. **My NOT-A-FOLLOW-ON calls are judgment, not measurement.** 25 rows classified by reading. A
   second reader would likely disagree on 3–5 borderline cases (rows 64, 70, 89, 90 are the ones I
   was least sure of; I put 70 and 90 in REAL-DEBT and 64 and 89 in NAF, and I could defend the
   swap). The partition is exact; the *assignment* is not certified.
5. **Class vs instance.** The fixes I name in §6 are class fixes (D1 changes every ledger-sourced
   row, D4 changes the extractor predicate). But I did not *land* them — this arm is read-only, so
   §6 is a debt I created, not one I paid. Named here so it does not become the thing this memo
   accuses others of.
6. **What I did not check.** The SSD run-config tier was not exhaustively searched for
   `kd_warm_start_dir` (the full-tier grep exceeded this arm's time budget), so item 15's
   "never fired" is scoped to **the repo tree only**. And 6,194 memos outside the `since` window
   were never examined by the extractor at all.

---

## §8 The honest residue

- **6,194 of 7,378 memos (83.9%) are outside the extraction window** and are not adjudicated. No
  claim, in any direction, is made about follow-on debt there.
- **20 of 21 ALREADY-DONE verdicts rest on ledger self-report**, not on a receipt check.
- **The 29 open items are not ranked by value** and this memo deliberately does not estimate one.
  Nine of them are `VEHICLE-SCOPE-OWED` (dated 07-19…07-25, targeting lineages MEMORY records as
  retired or rate-dead); for those, the first measurement is *"does this still apply to the
  own-vehicle line?"*, and it is cheaper than the item itself.
- **Nothing here moved the pointer.** `0.1910828242 [contest-CPU]` **UNMOVED**. This is apparatus:
  it converted an 86-row unadjudicated bucket into a 29-item queue with named next measurements,
  and it corrected one propagated false claim. That is a means, not an end.

## §9 The single next measurement

If exactly one thing fires from this memo, make it **item 1 (φ from the composite-R adjoint)** —
it is a *read*, it costs minutes, it is the only T0 row, and gc15 states it **gates arms D±**. The
runner-up is **item 3** (tw1 re-priced knee), because it is the only open item measured on the live
own-vehicle line and it discharges the §5 scope guard directly.
