---
schema: ddm_op3_canonical_operating_point.v1
date_utc: 2026-08-03
arm: ddm_op3 (one canonical live operating point; make the stale-fit genus representable)
lane_id: "lane_ddm_op3_20260803"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false   # exact contest pointer 0.1910828242 UNMOVED. No eval was run by this arm.
verdict_scope: CLASS
axis: "[macOS-CPU advisory - real evaluator receipts, recomputed from components] NON-PROMOTABLE.
  Every number is either recomputed by the executable equation from a real report.txt or
  measured by counting ledger rows. NO new scorer run, NO training, NO paid dispatch, NO
  pointer mutation. $0."
consumes:
  - /Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/*/report.txt  (14 receipts, 14 parsed)
  - .omx/state/canonical_frontier_pointer.json
  - .omx/state/canonical_task_status.jsonl  (417 rows at entry, 422 at exit)
  - .omx/research/pr86_pr130_fullstack_intake_20260728.md  (PR130 floor triple)
  - .omx/research/ddm_qd1_backlog_drain_20260803.md
  - .omx/research/ddm_qd2_rebaseline_against_cx1_20260803.md
  - .omx/research/ddm_na1_negative_audit_20260802.md
produces:
  - src/tac/canonical_equations/gap_decomposition_against_floor_20260802.py  (custody readers + transformation laws)
  - src/tac/canonical_task_status/contract.py  (delta_s_custody_findings)
  - src/tac/canonical_task_status/writer.py  (DeltaSCustodyError)
  - 5 append-only ledger corrections (826, 827, 850, 873, 882)
consumers: [MAIN]
tokens: [no-triality, p0-ledger-ok]
---

# ddm_op3 — the operating point, made un-typeable

## §0 ANSWER FIRST

**The obvious fix would not have worked.** I was chartered to add a baseline clause to
`contract.py:160`, which guards `actual_delta_s`. **The two rows that actually misdirected
readers this week both carry `actual_delta_s = None` and state their ΔS in `title`.** A clause
on the typed field passes both. Measured over the 417-row ledger: **15 distinct rows assert a
ΔS, and the pre-existing invariant can see exactly one of them.** I mutation-tested this —
reducing my clause to the typed-field-only design breaks 5 of my controls, including both real
rows.

**One canonical live operating point now exists and every input to it is read from a receipt.**
`live_operating_point()` parses 14/14 evaluator receipts, selects the minimum recomputed S,
and decomposes the gap. It independently reproduces every figure `ddm_qd2` derived by hand:

| quantity | value | agrees with |
|---|---:|---|
| live best (`v4d_cx1_pj2ix2`) | **0.8264971874742499** | qd2 §1, to 1e-7 |
| PR130 floor | 0.1721412975 | published 0.172141 |
| **total gap** | **0.6543558900** | qd2 0.6543559 |
| **1% of gap** | **9,827.2 B** | qd2 9,827.2 |
| shares seg / pose / rate | **61.4% / 22.1% / 16.6%** | qd2 exactly |
| `W` (seg↔rate) | 1.2731082153320312 | banked constant, 1e-12 |
| `dS/d(d_pose)` at `cx1` | 31.302439 | qd2 31.3024 |

**The seven circulating gap variants reconcile completely, and the spread has two separate
causes that were being read as one.** The *material* spread is six frozen ERAS
(0.7918465 … 0.6543559 — a 21% range). The residual 3e-7 between MAIN's `0.6543562` and qd2's
`0.6543559` is **not an error at all**: it is PR130's `d_pose` being published to four
significant figures (2.331e-5 vs 2.3311e-5 moves the floor by 3.3e-7). Stating that band
matters, because otherwise the seventh decimal invites a hunt for a bug that does not exist.

**A finding the instrument surfaced that no memo states:** our receipts are
`[macOS-CPU advisory]` and the PR130 floor is `[contest-CUDA]`. **Every gap figure this
campaign quotes is a CROSS-AXIS comparison.** It ranks the axes correctly — the gap is 3.8× the
entire floor — but its low-order digits are not a paired same-instrument measurement, and
nothing said so. `cross_axis_warning()` now says it on every run.

**Depth amendment, answered structurally rather than by enumeration.** A quoted scalar is the
value of a function **at a point in an 8-coordinate argument space**; "a number that was true
and silently stopped being true" is exactly the loss of one coordinate. It is **argument loss,
not arithmetic error** — which is why every instance passed every arithmetic check it was ever
given. The ladder could not represent the genus because it grades *how a value was obtained*,
never *what it was obtained relative to*. §2.

**And the three coordinates are not alike under a move**, which is the distinction nothing
carried: `W` is **exactly invariant** (a ratio of two linear terms has no operating point),
while `dS/d(d_pose)` is the derivative of a **concave** term and has risen **1.73×** since
`pw1`. So **banked pose levers are UNDER-priced, not stale** — re-pricing finds value in that
direction, and nobody had swept it.

**Pointer delta: NONE.** Exact contest pointer `0.1910828242` UNMOVED; own-vehicle best `cx1`
`0.8264972` unchanged. This arm moved no score. It removed a phantom `−0.2026353` from the
ledger's apparent bank and made the next stale figure fail loudly instead of quietly.

---

## §1 WHAT WAS BUILT, AND WHERE THE DEBT WAS PAID

Two existing surfaces, extended. **No new registry, no new state file, no new tool** — the
failure this arm exists to prevent is a beautiful instrument nobody calls.

### 1a. `tac.canonical_equations.gap_decomposition_against_floor_20260802` (existing)

The equation was **right both times MAIN was wrong**; its INPUTS were hand-typed. So the fix is
not more arithmetic care, it is removing the typing opportunity:

| addition | what it removes |
|---|---|
| `parse_evaluator_report` / `triple_from_evaluator_report` | every scoring input read from `report.txt`, **including the rate denominator** (`Original uncompressed size`) — Catalog #812's dynamic-denominator hazard answered by custody, not by a constant that was right the day it was typed |
| `demonstrated_floor_pr130` | the floor cross-checked **twice**: it must reproduce the published `0.172141`, AND agree with `canonical_frontier_pointer.json`. If the leaderboard moves, it REFUSES |
| `live_operating_point` | one call answers live-best / bar / gap / 1%-in-bytes, and **reports its scope denominator** |
| `marginals`, `seg_rate_exchange_bytes_per_flip`, `restate_pose_delta_at` | the invariance class of each marginal, so an exactly-invariant exchange rate and a strongly point-dependent one stop reading alike |
| `cross_axis_warning` | announces the advisory-vs-CUDA leg mismatch on every run |

### 1b. `tac.canonical_task_status` contract + writer (existing)

`delta_s_custody_findings()` — **one implementation, two policies**:

- **the reader WARNS.** The ledger is append-only; 15 historical rows predate this rule. A
  reader that RAISED would break campaign-wide recall — strictly worse than the defect it
  reports. This is the same split already used for malformed `event_notes` in that module.
- **the writer REFUSES** (`DeltaSCustodyError`), at **both** `update_status` (typed assertion)
  and `register_task` (**where the class actually entered** — both real rows were minted with
  their ΔS in the title). Live count for NEW rows is **0 by construction**, so no backfill is
  owed before it binds, and the strict-flip atomicity rule is satisfied without a warn-only
  purgatory.
- **carry-forward is not gated.** You must custody what you ASSERT; you may carry forward what
  someone else asserted. Otherwise historical tasks become un-annotatable.

Four clauses, each traceable to a coordinate that was separately lost: `MISSING_BASELINE`,
`UNDECLARED_PARTIAL_COMPOSITE`, `PARTIAL_IN_TYPED_FIELD`, `MISSING_POPULATION`.

---

## §2 THE GENUS, DERIVED

Enumerating incidents does not close a class. The structural statement:

> A quoted scalar is the value of a function **at a point**. Carrying the number without the
> point is the defect. Every listed instance is one coordinate of that point going missing.

| # | coordinate | what silently moved | transformation law |
|---|---|---|---|
| 1 | **baseline** | our own frontier moved **6×** in one day | `NOT_RESTATEABLE` — re-measure |
| 2 | **floor / denominator** | six frozen gap eras, 0.7918465 … 0.6543559 (21%) | `RESTATEABLE` — absolute ΔS invariant, % of gap rescales |
| 3 | **operating point** | `dS/d(d_pose)` rose **1.73×** (concave term) | `RESTATEABLE` — and **upward** |
| 4 | **population** | n=73 read −0.122 WIN, its own n600 read +0.152 LOSS | `NOT_RESTATEABLE` |
| 5 | **scope** | empty scope reported as PASS, 7 instances in one day | `NOT_RESTATEABLE` |
| 6 | **term set** | "−0.0866 UNLOCK" whose composed row is **+19.22** | `NOT_RESTATEABLE` — it was never a ΔS |
| 7 | **formulation** | divided by a smooth-label floor we are 27% BELOW | `NOT_RESTATEABLE` |
| 8 | **instrument** | advisory-vs-CUDA legs (found by this arm, §0) | `RESTATEABLE` with a stated band |

**Why the ladder could not represent this.** The value-provenance ladder grades **how a value
was obtained** — measured, derived, waived. Coordinates 1–8 are all about **what it was obtained
RELATIVE TO**. A number can sit on the highest rung — genuinely MEASURED, byte-closed, exact
evaluator — and still be false today, because its rung says nothing about its reference. `#826`
is precisely that: a real measurement against a real archive, correctly recorded, now inverted.
**The ladder is orthogonal to the genus, which is why it could not represent two of its own
three instances.** The fix is not a new rung; it is that a rung and a reference are different
things and a claim needs both.

**Collapsing the three laws into "stale" is what made re-pricing look purely destructive.** Of
the eight coordinates, three are `RESTATEABLE`, and one of those points *upward*.

### 2a. An instance measured inside the toolchain, by accident

Mutation-testing my own controls, a restore appeared not to take: the source read `191_052`
while the imported module reported `190952`. Measured directly:

```
src  mtime 1785758969.795791   size 27150
pyc  mtime 1785758969          size 27150
```

CPython invalidates a `.pyc` on **(mtime truncated to seconds, size)**. My edit was
**byte-length-preserving** (`191_052` → `190_952`, both 7 chars) and landed **within the same
second**, so both coordinates were unchanged and the interpreter served **mutant bytecode from
restored source**. That is coordinate #1 of the genus inside the tool I was using to test the
fix for coordinate #1 — a cached derived value whose validity key lacks the coordinate that
actually changed. (CPython's own answer is PEP 552 hash-based invalidation.)

The second-order lesson is the one that matters: for ~10 minutes the evidence said my gate
**failed open**. Had I trusted that reading I would have "discovered" a bug that did not exist
and weakened a working clause to chase it.

---

## §3 THE RECONCILIATION — SEVEN VARIANTS, TWO CAUSES

```
                                     gap to bar      1% of gap
  v4d          0.9639878  -> floor    0.7918465      11,892.1 B
  pw1          0.9476092                0.7754679    11,646.1 B
  (DAG row)                             0.7631413    11,461.0 B
  ms8/dc1_fold 0.8983775                0.7262362    10,906.8 B   <- the constant in my charter
  pj2          0.8308905                0.6587492     9,893.2 B
  cx1  LIVE    0.8264972                0.6543559     9,827.2 B   <- recomputed here
```

- **Material cause — era drift (21% range).** Absolute ΔS is invariant under this; a *fraction*
  of gap is not. Fractions **GROW as the gap shrinks**, so a "% of gap" claim is *under*-stated
  by its own drift. **The absolute belongs in the ledger; the percentage belongs to the moment.**
- **Immaterial cause — floor input precision (3.3e-7).** PR130's `d_pose` is published to 4 s.f.
  `2.331e-5` → floor `0.1721413`, gap `0.6543559`; `2.3311e-5` → floor `0.1721416`, gap
  `0.6543556`. That 3.3e-7 band **fully accounts** for MAIN's `0.6543562` vs qd2's `0.6543559`.
  Relative size 5.0e-7. **No bug. Do not hunt it.**

The `190,952` vs `191,052` question is settled by reproduction, and the provenance is now clean:
**190,952 is PR130's inner `p` payload; 191,052 is `archive.zip`, and `evaluate.py:63` charges
`archive.zip`.** 190,952 gives 0.1720747, which misses the published row by 130× the loader's
tolerance. Pinned as a test.

**Live gap decomposition (supersedes the `dc1_fold`-era figures in `cv1` / `gd3` / `gd4` /
MEMORY):**

| axis | gap | share | vs the ms8-era figure carried in MEMORY |
|---|---:|---:|---|
| **seg** | **0.4015190** | **61.4%** | 55.3% — understated by 6.1 points |
| pose | 0.1444644 | 22.1% | 29.2% |
| rate | 0.1083725 | 16.6% | 15.5% |

`rank_by_gap() = ('seg', 'pose', 'rate')` — a MEASURED output, not an assumption. **Seg's share
rose without seg moving**: `d_seg = 0.00431179` is bit-identical across all eight rows from
`v4c_static_photo_celldrop50` to `cx1`. Every own-vehicle move since 2026-07-30 was pose or
rate. The standing "seg is the majority of what is left" reading is **strengthened**, not
weakened.

---

## §4 POSITIVE CONTROLS — 85 TESTS ON THE TOUCHED SURFACES, MUTATION-PROVEN

Two single-file detectors written the previous day both passed their own positive control while
being unable to return the negative. So every clause is exercised **twice**, and then the code
is **mutated** to prove the tests can fail.

| mutation | tests that failed | proves |
|---|---:|---|
| byte regex made tolerant (the real qd2 bug) | 4 | the comma-truncation guard is load-bearing |
| population check disabled | 1 | the n600 requirement is reachable |
| floor bytes reverted to 190,952 | 6 | the reproduction cross-check binds |
| **free-text detection disabled** (i.e. **the obvious typed-field-only design**) | **5** | **both real misdirecting rows become invisible** |

The last row is the argument for the whole design. Controls also cover: empty scope refused;
malformed pointer raises rather than skipping its own check; a fully-custodied row **passes**
(a clause that refuses everything is not a gate); the 402-of-417 rows that assert nothing are
**untouched** (a gate that fires everywhere is noise); reader stays total over history.

**Scope denominators, all measured, none inferred:**

- 14 of 14 evaluator receipts parsed (0 unparsed).
- 417 ledger rows at entry → **16 flagged (3.8%), 13 distinct tasks**; 8 typed + 8 free-text-only,
  1 overlapping = **15 distinct ΔS assertions**, of which the pre-existing invariant saw **1**.
- Widening to reference-comparison prose added **exactly 1** further real row (2 of 417 = 0.5%
  carry that prose; both are genuine ΔS claims). Specific, not noisy.
- After the §5 corrections: **8 of 148 tasks** flagged on the latest-row view a reader consumes.

### 4a. The control I had to correct — my own fabrication

My first `_REAL_826_TITLE` rendered the delta as `= -0.0983195 S`. **The real row writes
`vs ref 0.7685479 (-0.0983195)` — a bare parenthesised number with no `S` label anywhere.** I
had edited a case to suit my detector and called it a replay. Caught by testing the clause
against the ledger bytes instead of against my own string.

Fixing it produced the **sharper rule**. `#826` **did** name its reference. The defect was never
a missing baseline — it was an **un-re-derivable** one: a bare number nobody can date, locate,
or recompute, so nobody could see it had been superseded five times. **Naming a baseline is
necessary; naming a re-derivable one is the requirement**, and that is exactly what the
`[baseline:<locator>=<S>]` token buys over prose.

**The honest limit, pinned as a test** (`test_MEASURED_LIMIT_an_unlabelled_bare_delta_is_still_invisible`):
detection keys on **claim language** — an `S` label or reference-comparison prose. A delta
written as a naked parenthesised number with neither still passes. Widening to "any signed
decimal" would fire on coordinates, byte counts and ratios across the whole ledger. The next
reader learns that boundary from the suite rather than from a missed row.

---

## §5 THE LEDGER CORRECTIONS (append-only; the convention's first users)

Five rows corrected. Each pre-verified against the clause **before** writing — and the clause
**refused my first attempt**, because my verification harness probed a `[partial:]` note with a
typed delta, which is the one combination the clause forbids. The gate refusing its own author
on the live path is the strongest control I have.

| task | what it advertised | re-priced vs live best | restated as |
|---|---|---:|---|
| **826** | −0.0983195 vs a v4d-era ref | **+0.0034632** (INVERSION) | exchange rate **32.53 B/flip = 25.5× W**, and a **SPECIFICATION**: re-encode within **212 B** of `cx1` and it is net-positive again |
| **827** | "seg+rate −0.0866789 S = 11.178% of gap" | **+19.2200801** | `[partial:seg+rate]`; omitted pose term **+19.302316 = 234.7×** the advertised prize; own seg+rate leg re-prices to −0.0822362 (it SHRANK — `cell_drop50` is its REFERENCE, so it sheds that deficit rather than inheriting it) |
| **850 / 873 / 882** | −0.0675451 each | **0.0000000** | ABSORBED-INTO-`cx1`, **TRIPLE-STAMPED**: one `pj2` run's total on three distinct scopes. Summing banks **−0.2026353** for a delta worth zero. "DO NOT SUM THESE THREE" |

Every correction carries `[baseline:…=0.8264972] [n600] [empirical:…]` and, where applicable,
`[partial:…]`. Originals stand as HISTORICAL_PROVENANCE per Catalog #110/#113. `score_claim=false`
throughout; no status transitions; pointer untouched.

**Nothing here is a new win.** The honest total of re-priceable banked ΔS against `cx1` remains
**0.0000000**.

---

## §6 REVIEW PASSES — COUNTER STANDS AT 2 CLEAN, NOT 3

**Stated plainly because round-finished is not clean-pass.** Six rounds ran. **Four found real
defects and reset the counter; the last two were clean.** The protocol asks for three
consecutive clean passes and I have two — so this landing is **one clean round short of the
protocol bar**, and the next reader should run that round rather than assume it happened.

Every one of the four defects was in **my own work**, and three of the four were instances of
the very genus this arm exists to close (§7). Assumption classifications:

| assumption | classification | note |
|---|---|---|
| `cx1` is the live own-vehicle best | `VERIFIED_VIA_EMPIRICAL_ANCHOR` | min over 14/14 receipts, recomputed from components |
| PR130 floor = (2.966e-4, 2.331e-5, 191,052) | `VERIFIED_VIA_SOURCE_INSPECTION` | intake memo line 140 + reproduces the published row |
| the 3e-7 spread is floor-input precision | `VERIFIED_VIA_EMPIRICAL_ANCHOR` | both floors computed; band 3.3e-7 brackets the observed 3e-7 |
| `W` is exactly invariant | `VERIFIED_VIA_SOURCE_INSPECTION` | derived from `evaluate.py:92`; ratio of two linear terms |
| the 8 argument axes are COMPLETE | **`ASSUMED_AWAITING_VERIFICATION`** | **PROVISIONAL.** 8 coordinates from 7 known instances + 1 found here. A ninth is likely; the structure (argument loss) survives adding one, the enumeration does not |
| free-text detection precision holds on future rows | `INFERRED_FROM_DOMAIN_LITERATURE` | **PROVISIONAL.** 3.8% fire rate measured on today's 417 rows only |

---

## §7 ROUND-BY-ROUND ADVERSARIAL REVIEW

**Round 1 — FOUND AN ISSUE, counter reset.** The fabricated `_REAL_826_TITLE` (§4a). Shared
assumption surfaced: *"the ΔS lives in the typed field."* Violating it was the entire unlock —
it is what revealed that the obvious design passes both live failures, and it is now
mutation-proven.

**Round 2 — clean.** Shared assumption surfaced: *"a stale number is a wrong number."* False
for 3 of 8 coordinates; `restate_pose_delta_at` exists because of this round. Also checked: the
reader must not raise (would break recall on 15 rows — verified by test); carry-forward must not
be gated (would make historical tasks un-annotatable — verified by test); blast radius of the
writer refusal enumerated across `tools/` and `src/` (only `probe_jrd_coefficient_prefix.py`
touches `actual_delta_s`, and only as `None`).

**Test totals, counted not estimated:** the three suites I touched carry **85** tests
(23 canonical-task-status + 38 gap-decomposition + 24 ingestion), all green. A further **98**
downstream consumer tests are green on the same tree — 59 across duckdb-read-model /
jrd-coefficient-prefix / per-pair-difficulty-atlas, and 42 across the two graph-memory suites,
which construct `CanonicalTaskStatusRow` over the live ledger and are therefore the sharpest
check that the reader stayed total.

**Round 3 — FOUND AN ISSUE, counter reset.** Shared assumption surfaced: *"a gate that fires is
a gate that works."* The counter-check is the 402 rows it must NOT fire on and the known-good
cases it must pass; both are now pinned. The defect:
`test_the_live_ledger_loads_and_any_anomaly_is_visible` asserted `len(caught) == malformed`,
which is **broader than its own stated intent** — it required that event-notes coercion be the
only warning the reader can EVER emit, so any second warning class fails it. Corrected to count
**by class**, preserving the real invariant (one warning per malformed row, never a silent pass)
and adding an explicit empty-scope guard.

**Round 4 — FOUND AN ISSUE, counter reset. My own code committing the genus.**
`demonstrated_floor_pr130` defaulted to the RELATIVE path `.omx/state/canonical_frontier_pointer.json`.
From any other working directory that file does not exist, the loader takes the "pointer absent"
branch, and **the bar cross-check silently does not run** — a check that skips itself under an
unstated condition, which is coordinate #5 (scope) of my own §2 table, inside the function
written to enforce coordinate #2. Fixed three ways: resolve from `__file__`; WARN loudly when
the pointer is genuinely absent ("the check did not run" and "the check passed" must not look
alike); and require an explicit `SKIP_POINTER_CROSSCHECK` sentinel to opt out, because "use the
default" and "deliberately skip" must never share a value. Three controls added.

**Round 5 — FOUND AN ISSUE, counter reset.** `ParsedEvaluatorReport.total` could
`ZeroDivisionError` deep inside a caller, because the receipt regexes accept `0`. A zero-sample
receipt is also the empty-scope failure wearing a report header. Moved the refusal to the
parser, where the bad receipt can be named. Three parametrised controls added.

**Round 6 — clean.** Checked: `DeltaSCustodyError` subclasses `ValueError`, so existing callers
that catch `ValueError` are unaffected; `title` is isinstance-validated as `str` before the
custody call reads it; the keyword-only signature prevents positional confusion; the writer
refusal's blast radius is empty in `tools/` (the only `actual_delta_s` writer passes `None`).
Also verified against the LIVE ledger that
`check_canonical_task_status_no_dangling_transitions` reports **13 violations, all of which
exist byte-identically at HEAD** (tasks `438` ×7, `z6_v2_phase_c…` ×3, `deferred_items_feeder…`
×3, all `source_design_memo missing`, all predating this arm). **My five appended rows introduce
zero.**

**Round 7 — clean.** Full regression: **183 tests green** across the three touched suites and
five consumer suites, including both graph-memory suites, which construct
`CanonicalTaskStatusRow` over the live 422-row ledger and are therefore the sharpest available
proof that the reader stayed total.

**What would change my mind.** If the free-text clause's 3.8% fire rate rises materially on
rows written *after* today, the pattern is over-broad and should narrow to the typed field plus
an explicit opt-in marker. And if a ninth argument coordinate appears, §2's *enumeration* is
incomplete — though its *structure* (argument loss) is what the clauses are built on, and that
survives.

**What I did not do.** I did not touch the 8 remaining flagged tasks (4 are `0.0000000`
no-ops, 3 predate July, 1 is in-flight under `ddm_bs2`). I did not correct the `dc1_fold`-era
gap constants in `cv1` / `gd3` / `gd4` / MEMORY — §3 supersedes them in text, but those files
are other arms' surfaces. I did not add a preflight gate: the enforcement is at the writer,
which is where rows are minted, and a preflight scan would be a second surface for one rule.

---

## §7a LANDING STATUS — COMMIT BLOCKED BY AN UNRELATED NATIVE MLX CRASH

The work is complete and green in the working tree; **the commit is blocked by the pre-commit
hook's CI-blind (MLX-gated) step**, which dies with `Fatal Python error: Bus error` /
`Segmentation fault` inside
`src/tac/substrates/_shared/mlx_score_aware/adapter.py::_score_aware_loss_part_metrics`.

**PROVEN PRE-EXISTING, not argued.** I created a detached `git worktree` at pristine `HEAD`
(`git status --porcelain` empty — none of my changes present) and ran the hook's exact 40
targets with the hook's exact pytest invocation. **It crashes identically**, and from the clean
tree the traceback is finally readable:

```
test_compact_renderer_mlx_spine_runner.py:18593 test_hinerv_execute_runs_training_archive_and_receiver_proof
 -> run_compact_renderer_mlx_spine_runner.py:10646  execute_hi_nerv_mlx_scoreaware_and_adapt
 -> run_compact_renderer_mlx_spine_runner.py:18775  _run_hi_nerv_mlx_scoreaware_smoke
 -> mlx_score_aware/harness.py:508                  run_mlx_score_aware_full_main
 -> training/long_training_canonical.py:3760        run_long_training
 -> training/long_training_canonical.py:2883        run_step
 -> <native fault: Bus error / Segmentation fault>
```

**This is a repo-wide commit blocker at HEAD**, not a property of my landing: it fires for ANY
commit whose staged files select `test_compact_renderer_mlx_spine_runner.py` into the CI-blind
set. The earlier `mlx_score_aware/adapter.py` line numbers were red herrings — with the process
already corrupted, the reported frame moved run to run (5410, 5369, `<invalid frame>`), which is
why the frame must not be trusted and the pristine-HEAD reproduction was worth the ten minutes.

Supporting evidence gathered along the way:

- **The crashing file is at HEAD**, unmodified by any arm (`git status --porcelain` empty).
- **It imports nothing I changed** — no `canonical_task_status`, no `gap_decomposition` anywhere
  under `mlx_score_aware/`.
- **The hook's selection contains 40 MLX test targets and none of my files.** The selection is
  identical whether I stage my files or a single unrelated one.
- **It is order-dependent, not deterministic-by-line.** Three runs reported three different
  crash sites (adapter lines 5410, 5369, and `<invalid frame>`), and `<invalid frame>` in the
  traceback means the Python-level line is not trustworthy at all. The hook runs pytest without
  `-p no:randomly`, so ordering varies per run.
- **Every implicated module passes in isolation**: target #11 alone = 35 passed; the first 11
  targets together = 47 passed in 53s; `test_compact_renderer_mlx_spine_runner.py` alone ran
  139 tests without faulting.

That signature — a native fault whose location moves, reproducible only in a large mixed-module
process, in unmodified code — is a **Metal/MLX resource-state fault**, plausibly aggravated by
the other arms running MLX concurrently on this machine.

**I did not set `PREFLIGHT_SKIP_CI_BLIND_TESTS=1`.** The charter forbids normalising it and the
hook is explicit that it is the only automated surface that runs these modules; bypassing it
would re-create the skip-as-green silence that this arm's entire §2 is about. The correct
resolution is an operator/MAIN decision: either re-run the commit when MLX contention drops, or
adjudicate the crash on its own merits as a separate defect. **Nothing about this landing should
be taken as evidence the crash is benign** — it is a real, unexplained native fault on the only
machine that can see it, and it deserves its own arm.

---

## §8 NEXT-IF-RESUMED

1. **Re-price the pose backlog UPWARD.** `restate_pose_delta_at` is built and tested; nothing
   has been run through it. Every pose lever parked at a `pw1`- or `ms8`-era operating point is
   under-priced by **1.73×** / **1.42×**. This is the only direction where re-pricing finds
   value, and it is now one function call.
2. **Correct the era-frozen gap constants** in `cv1` §, `gd3`/`gd4` headers and MEMORY to the
   §3 figures (append-only). Seg's share is **61.4%**, not 55.3%.
3. **Resolve the cross-axis leg (§0).** Either measure `cx1` on a contest-CPU/CUDA rail, or
   carry the advisory band explicitly in every gap quote. Currently the instrument warns; nobody
   has priced the warning.
4. **`#826` is a live specification, not a kill:** re-encode `gr1_cell_drop50` within **212 B**
   of `cx1` (a 25.5× byte cut) and it inverts back to net-positive.
5. **Drain the remaining 8 flagged tasks** — 4 are `0.0000000` and can be closed with a one-line
   custodied note; the other 4 need their owners.
6. **Consider PEP 552 hash-based `.pyc` invalidation** (§2a) for this repo's test runs. Cost is
   a small hash per import; the failure it removes is a same-second byte-length-preserving edit
   silently serving stale bytecode — which cost this arm ten minutes and could cost a verdict.

**Pointer delta: NONE.** `0.1910828242` UNMOVED, own-vehicle best `cx1 0.8264972` unchanged.
This arm produced no score; it made the next stale figure fail loudly.
