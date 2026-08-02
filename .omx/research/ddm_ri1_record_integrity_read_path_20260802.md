---
schema: ddm_arm_finding.v1
arm: ddm_ri1
date_utc: 2026-08-02
axis: "[macOS-CPU advisory] — 0 scorer forwards, 0 bytes shipped, no live run touched"
score_claim: false
promotable: false
pointer: "UNMOVED — effective_frontier 0.172 (upstream official); own-vehicle frontier v4d 0.9639878"
rows: ["#899 (A, LIVE)", "#898 (B)", "#878 (C)", "#870/#879/#880 (D)", "ROW E (fl1 driver)", "ROW F (gt_n600 identity)"]
verdict_scope: "formulation — the three Row-B/E detector formulations are refuted AS FORMULATIONS on this corpus; the CLASS is real and has two confirmed members"
---

# RECORD INTEGRITY — the apparatus can hold a claim it cannot verify

One genus, six rows. The unifying test: **what does a consumer see when the claim is false, and is
it distinguishable from what they see when it is true?** Every instance below answered "no."

**Headline: ROW A is FIXED with a two-sided control. ROWS B/E are REFUSED with three measured
formulations and a structural reason. ROWS C/D are adjudicated as already-built-or-blocked, with the
blocker measured. ROW F gets a costed recommendation, not a build.**

---

## §1 ROW A (#899) — the P0 gate's READ path bypassed its own WRITE gate. FIXED.

`ddm_wd2` (`f953cfa3ba`) signed the harm clause on the WRITE path and left the READ path open
**deliberately and in writing**: dropping unverified rows is signal loss, silently trusting them is
a false claim. The third option is to **TYPE what was loaded**.

Why it mattered more than it looked: `ddm_gd5` DELETED the grade-5 detector, so **declaration is
currently the only route into `built-elsewhere-unwired`** — and declaration is exactly what the read
path failed to check.

### The defect, MEASURED against HEAD before the change (two-sided)

Hand-appended one grade-5 row with **no `live_measured` / `candidate_measured` /
`metric_direction`** — a shape `record_required_component` REFUSES — beside one legitimately
written, measured row, plus one truncated line:

```
PRE-FIX   read_required_components -> [('HandAppendedNoEvidence', <NO SUCH FIELD>), ('RealWinner', ...)]
          built_elsewhere_unwired  -> ['HandAppendedNoEvidence', 'RealWinner']    # P0 queue, admitted
          build_completeness_report()[0] -> HandAppendedNoEvidence               # ABOVE the measured row
          RAW LINES 3 | rows returned 2 | SILENTLY DROPPED 1                     # malformed line vanished

POST-FIX  typed rows      -> [('HandAppendedNoEvidence','declared-unverified'), ('RealWinner','verified')]
          P0 queue order  -> [('RealWinner','verified'), ('HandAppendedNoEvidence','declared-unverified')]
          report[0]       -> RealWinner  sort_rank 0  verified
          HandAppendedNoEvidence: grade built-elsewhere-unwired, sort_rank 3, declared-unverified
          integrity summary -> rows_read 2 · verified 1 · declared_unverified 1 · malformed_lines 1
          malformed detail  -> line_no 3, "unparseable JSON: Unterminated string…"
```

The **negative control is embedded**: `RealWinner` still verifies, still leads, still holds rank 0 —
so the demotion is not an unconditional reject. Without that leg the fix could be "refuse
everything" and every test would still pass.

### The fix — one existing module, no new surface

`src/tac/witness_dsl/activation_ledger.py`:

1. **`_validate_required_component()`** — the write-path admission logic extracted verbatim into
   ONE predicate. The write path and the read path now decide admissibility with the *same code*; a
   duplicated predicate is a drift generator (this module already says so about `BUILD_GRADE_ORDER`).
2. **`verify_required_component_row(row) -> (integrity, reason)`** — replays that predicate on a
   STORED row. It decides nothing new, so a row can never verify on weaker evidence than it would
   have needed to be written. The **reason travels with the row**; a bare flag makes the operator
   re-derive the defect.
3. **`read_required_components()`** stamps `record_integrity` ∈ {`verified`, `declared-unverified`}
   and `record_integrity_reason` on every row. **Nothing is dropped** — that was wd2's named failure
   mode.
4. **`required_component_integrity_summary()`** — the DENOMINATOR: rows read, verified,
   declared-unverified, and **malformed lines with their line numbers and reasons**. Previously an
   unparseable line and an empty file emitted the same symbol (`vacuity_is_indistinguishable_from_
   pass_empty_scope_confound_20260801`, at the record level).
5. **`built_elsewhere_unwired()`** orders verified-first. **`_effective_grade_rank()`** demotes an
   unverified grade-5 row out of rank 0. The demotion TARGET is **derived, not chosen**: the write
   path's own refusal text says such a row "is indistinguishable from built-never-fired", so it
   reads at built-never-fired's rank. **The row keeps its declared grade label** — silently
   relabelling would swap one false record for another; only the read ORDER changes.

Rows of every *other* grade keep their rank: their rank is not evidence-justified, so an unverified
charter there is a debt with a thin description, not an unproven harm claim.

### Tests + live state

8 new tests (56 pass in `test_build_completeness_grades.py`, was 48). Includes a guard that runs
against the **REAL** ledger and asserts `rows_read > 0` first, so it cannot pass vacuously. Live
store today (RUN, not asserted): **28 raw lines → 22 unique rows** after latest-row-wins, **22
verified, 0 declared-unverified, 0 malformed** — the defect was latent, not live, and the guard fails
the moment anyone hand-edits the store into a state its own writer would refuse. `ddm_wt1`'s rows
share the schema and are covered by the same typing (all 22 are). The store currently holds **zero**
`built-elsewhere-unwired` rows, so the demoted-rank path has no live instance either.

**Round-2 self-review caught a real one** (`_fire_order_key`): the sort key read `fire_order`
straight from the stored row, so a row with a **string** `fire_order` made `sorted` raise and took
down the read of the **whole** store — one bad row costing the entire corpus, which is precisely the
2026-08-01 recall-layer failure. Such a row is already typed `declared-unverified`; it must not also
be able to silence its healthy neighbours. Fixed and guarded by its own test.

**Two pre-existing failures** (`test_newly_mapped_flags_leave_unmapped`,
`test_p1_repo_live_count_bounded`) are **NOT mine — PROVEN**: I copied my two files aside, reverted
them to HEAD, and both still failed identically; restoring reproduced my 55 passes. Neither test
references any symbol I touched.

---

## §2 ROWS B + E (#898) — the detector is REFUSED, on three measured formulations

The class is real and now has **two confirmed members**: `ddm_cg2`'s n600 sweep (memo + JSON, zero
executable files committed) and `ddm_fl1`'s driver (exists only on `/Volumes/`, not in the repo).
MAIN correctly raised the bar for a refusal once the class recurred. Here is the argument.

| # | formulation | fires on cg2 | fires on fl1 | base rate on real corpus |
|---|---|---|---|---|
| 1 | zero executable files in the landing's commit set | **YES** | not established | **281 / 1,751 commits = 16.0%** (since 2026-07-20) |
| 2a | memo cites an **off-repo** (`/Volumes/…`) `.py`/`.sh` producer | NO | NO | 5 / 6,953 memos = 0.07% |
| 2b | memo cites a repo-relative `.py`/`.sh` that does not resolve at HEAD | NO | NO | 668 / 6,953 = 9.6% |
| 3 | measurement memo names **no** executable path at all | NO | **YES** | **1,172 / 3,193 = 36.7%** |

Scopes and controls: formulation 1 over all 1,751 commits since 2026-07-20 (of which 1,036 have zero
executable files and 417 claim a measurement in the subject). Formulations 2–3 over all 6,953
`.omx/research/*.md`; positive control = the tracked-path set (39,679 paths) against which resolution
is decided, and the regexes returned real hits, so the zeros are real rather than empty-scope vacuity.

**No formulation catches both members, and every formulation that catches either carries a
double-digit base rate.** Formulation 1's ~281 firings are overwhelmingly legitimate — DAG FEED rows,
`current_focus` updates, receipt landings, and memos recording measurements made with tools that were
already committed. A gate with hundreds of false positives per true positive is not a gate; it is
noise that trains its readers to ignore it.

**The structural reason, which is why tuning will not rescue it:** *the producer relation is not
written down.* cg2 cites `tools/levelset_byte_close_and_eval.py` and
`tools/probe_einstein_kolmogorov_xi_bridge.py` — real, resolvable paths that are simply **not** its
producer. fl1 cites no path at all. **Nothing in a memo distinguishes "the path I ran" from "the path
I read."** A text predicate cannot recover a relation the text does not encode. This is the same wall
`ddm_gd5` hit on the grade-5 detector (three formulations, measured against controls, honest refusal
as the deliverable), reproduced independently here, and the same wall §4 measures for #880.

**On #670 (`LandingDiffManifest` enforcement is warn-only, HELD in the deferral ledger, QD07).** I did
not have to choose between (a) pay #670 first and (b) land warn-only: **the measurement removes the
predicate, so the surface question is moot.** Landing any of these three onto a warn-only gate would
have added a double-digit-FP check to a gate already in warn-only purgatory — compounding exactly the
debt #670 names. That is the strong argument MAIN asked for, and it is measured, not asserted.

**What WOULD work is a WRITE-side field, not a READ-side detector:** a measurement receipt that names
its own producer, checked for resolution at HEAD. That is exact, needs no prose parsing, and cleanly
separates "measured with a committed tool" from "measured with a tool that no longer exists."
`ddm_cg2_realization_n600_20260802.json` has no such field. **I did not build it**: `ddm_wi1` deleted
an untracked `audit_receipt_field_coverage` tool TODAY as built-instead-of-paying, so a receipt
auditor is a known live trap. It is filed below with an owner rather than left as prose.

**The working cure for existing members is `ddm_ti1`'s, and it is a practice, not a gate:**
independently RE-DERIVE rather than cite. ti1 reproduced fl1 exactly (spike count 625,297; atlas
total 0.00388854 reproduces the registered burn seg). That is what makes fl1's numbers safe to keep
using despite the missing tool, and it is the template. `ddm_mp1` is live rebuilding cg2's instrument.

---

## §3 ROW C (#878) — the instrument EXISTS; it is unwired, and that is the finding

`audit_handoffs` / `classify_handoff` / `summarise_handoffs` / `handoff_join_canary` landed today in
`src/tac/followon_ledger.py` (`ddm_oh1`), with tests and a canary. Row C is **built**, so I did not
rebuild it.

**MEASURED, exhaustive over the whole tracked tree (`git grep` over `.`):** `audit_handoffs` and
`audit_tasks` have **zero** non-test, non-memo callers. Positive control: `audit`, `summarise`,
`cache_age_s`, `ORPHANED`, `STAGED` from the same module ARE imported by `tools/costate_digest.py`,
so the zeros are real. `tools/costate_digest.py:1016` wires only the **memo** follow-on join.

That is grade-5 shape *inside the module built to detect orphans*. I flag it rather than wire it,
because §4 measures why one of the two cannot honestly be wired yet.

---

## §4 ROW D (#870/#879/#880) — THE JOIN is blocked by a measured absence, not by missing code

#880's join is **already built** (`audit_tasks`, `classify_task_execution`, `task_join_canary`) and
correctly takes its rows as caller-supplied input, refusing to enumerate a population it cannot see.

**The blocker, measured independently of the module's own claim:** across all **155** `.jsonl` files
under `.omx/state`, **4** carry a `task_id` key (positive control — the scan finds them where they
exist: `canonical_task_status.jsonl`, `canonical_equations_registry.jsonl`,
`canonical_anti_patterns_registry.jsonl`, `probe_outcomes.jsonl`), and **zero** carry any of task ids
**870 / 878 / 879 / 880 / 898 / 899**. `canonical_task_status.jsonl` holds 398 rows and none of the six.

**The operator task backlog has no on-disk mirror.** #880's "zero coverage on the task-row side" is
therefore structural: the join's population does not exist on disk. Wiring `audit_tasks` into the
costate digest today would produce a vacuous scope — the exact genus this cluster exists to extinct.
Building a mirror would be new machinery on a surface I do not own. **So the correct state is: the
join stays unwired, and the blocker is now recorded as a measured fact rather than a coverage gap.**

---

## §5 ROW F (gt_n600.npz scorer identity, 318 consumers) — recommendation, NOT a build

`ddm_sf1` measured the artifact as currently CORRECT (re-hashed both weight files against the pin,
exact match) while the loader checks `n_pairs` only and `upstream/` is gitignored. Right today;
undetectable if it stops being right. My unifying test verbatim.

**Recommendation: the read-side identity CHECK is sufficient, and it does NOT require rewriting the
5.08 GB artifact.** The identity does not have to live *in* the npz: pin the scorer weight sha256 in
a few bytes under `.omx/state`, and have the loader re-hash at load and refuse on mismatch — exactly
the move sf1 already performed by hand. This is the same read-path-typing pattern as §1: the cheap
repair is to make the reader verify, not to make every past writer re-emit. The `np.savez` kwarg
remains the belt-and-braces version and should ride the next legitimate regeneration, never a
rewrite fired for its own sake. **Cost of the recommended path: no rewrite, no re-measure, one pin
file plus a load-time check.** Not built here — sf1 owns the measurement and it is not my surface.

---

## §6 What is OWED, with owners (never bare prose — that is Row D's own class)

| item | why it is not built here | owner | price |
|---|---|---|---|
| Measurement-receipt `producer` field + HEAD-resolution check (the §2 write-side cure) | `ddm_wi1` deleted a receipt-field auditor TODAY as built-instead-of-paying; needs the receipt-schema owner, not a detector arm | receipt-schema owner (MAIN to assign) | ~1 field + 1 check |
| Wire `audit_handoffs` into `tools/costate_digest.py` | built by `ddm_oh1` today; consumer choice is oh1's/MAIN's, not mine to pre-empt | `ddm_oh1` / MAIN | ~10 lines |
| Wire `audit_tasks` | **BLOCKED, measured §4** — no on-disk task mirror; wiring it now yields a vacuous scope | blocked, not owed | — |
| Row F loader identity pin + check | §5; sf1 owns the measurement | `ddm_sf1` / MAIN | no rewrite |

---

## §7 Honest boundaries

- Everything here is **apparatus**. Pointer UNMOVED. Zero scorer forwards, zero bytes, no live run
  touched. `score_claim=false`, `promotable=false`.
- `verdict_scope: formulation` on §2 — three formulations are refuted **on this corpus**; the CLASS
  is real with two confirmed members, and the write-side cure is named and priced, not killed.
- §3's and §4's "zero callers" are **exhaustive in a named scope** (whole tracked tree, `git grep`)
  **with a positive control**; they are not "I did not find any."
- Formulation 1's base rate is over **commit subjects**, not memo bodies — a subject-line proxy for
  "claims a measurement". A body-level scan could shift the 16.0% somewhat; it would have to fall by
  two orders of magnitude to change the verdict, and formulations 2–3 fail on the founding cases
  regardless of scan depth.
- I did **not** establish whether formulation 1 fires on fl1's landing; the table says so rather than
  guessing.
