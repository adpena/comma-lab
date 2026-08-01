# ddm_wi1 — the WRONG-INSTRUMENT sweep, and a wrong-instrument verdict reached with a wrong instrument

**Date:** 2026-07-31 · **Arm:** ddm_wi1 · **Axis:** `[macOS-CPU advisory]` ·
`score_claim=false`, `promotable=false`, `promotion_eligible=false`,
`rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false` ·
**Pointer: `0.1910828242` [contest-CPU] UNMOVED.** No number here is a score. `upstream/` untouched.

**Why this exists.** Operator 2026-07-31: *"Where else is that disease infected?"* +
*"Remember, you can always use FM tools for classification as well."* +
*"FM tools should be able to do classification and value, key and value."*

**Output-form rule.** Per `boolean_flags_are_a_ui_over_a_continuum_never_binary_judgment_20260731`,
every row below is a **placed point**, not a pass/fail. Each carries a `verdict_scope`
(INSTANCE < FORMULATION < FAMILY < PARADIGM). Denominators and SOUND control rows are reported in
§5 — an audit that returns only hits has not shown it can discriminate.

---

## 0. THE DISEASE, stated so it is distinguishable from its neighbour

- **C1 WRONG-INSTRUMENT** — an instrument **validated against, or generalized from, a case it does
  not actually cover**. It answers a *different* question than the evidence that motivated it.
- Neighbour (already audited by `ddm_cf1`): **COARSE FRAMING** — an instrument answering a strictly
  *weaker* question. Different axis; do not merge them.
- **C2 BUILT-INSTEAD-OF-PAID** — a NEW surface (tool/gate/registry/scaffold) added when the
  identified debt was to FIX or WIRE an EXISTING one. This is the generator of the orphan population.

The canonical one-line statement of C1 is already in-tree, written today, and I did not improve on
it — `src/tac/preflight.py:89311`:

> *"A reader cannot distinguish 'nothing is wrong' from 'nothing was looked at', which is how a
> hollow gate survives review."*

---

## 0.5 THE RANKED TABLE (ranked by LOAD-BEARING NOW, then by blast radius)

"Load-bearing" = gates a launch / prices a rate column / closes a family. Rank is an ordering for
attention, never a verdict; every row's evidence is in its section.

| # | instrument | file:line | disease | load-bearing | the gap, in one line | scope |
|---|---|---|---|---|---|---|
| 1 | `check_lane_scripts_have_e2e_smoke_proof` | `preflight.py:28416` (STRICT `:5653`) | C1 | **YES — gates lane dispatch** | Built because static checks were insufficient; satisfied by 10 static checks in 40 ms, on a proof store 17 d stale against a 7 d window | INSTANCE |
| 2 | `check_refusal_gates_have_live_positive_control` | `confound_gates.py:3332` (STRICT `preflight.py:6726`) | C1 | **YES** | Covers 2 of the 5 incidents it names; ratchet met by 2 unrelated controls | INSTANCE + FORMULATION |
| 3 | `check_lane_classes_have_pipeline_proof` | `preflight.py:28628` (STRICT `:5665`) | C1 | **YES** | "Complete cycle demonstrated" satisfied by 13/17 plumbing-only rows, `score: None`, and no staleness window at all | INSTANCE |
| 4 | costate `_LEDGER_GATE_OPEN_TOKENS` | `ddm_costate_organ.py:2202,2260` | C1 (#829) | **YES — core sense organ** | `"MET"` matched by substring; **85% false-match** on this corpus (`PHOTOMETRIC`/`TELEMETRY`/`GEOMETRY`); 1 live false OPEN today | INSTANCE + FORMULATION |
| 5 | p0 ledger `next_action` | `.omx/state/operator_p0_ledger.jsonl` (last row) | C1 staleness | **YES — routing surface** | Declares OWED a defect repaired ~3 h earlier (commit `990674f4a9`) | INSTANCE |
| 6 | P4 fmtools advisory layer | `confound_gates.py:1278` + `preflight.py:6774` | C2 / orphan grade 2 | YES (warn-only) | `use_fmtools=False` and the orchestrator never passes it ⇒ structurally un-fireable | INSTANCE |
| 7 | `tools/audit_receipt_field_coverage.py` | *(deleted)* | **C2** | was none | New machinery while `lever_registry` 0.585% and findings-gate 0% went unpaid | INSTANCE |
| 8 | `_validate_amrc_artifacts` | `preflight.py:15555` (STRICT `:1890`) | C1 | DORMANT | Glob matches **0** files repo-wide; validator returns `[]` unconditionally | INSTANCE |
| 9 | orphan-grade population | `ddm_ba31:281-291` (`ddm_sb2` #819) | **C2** | YES | **2 of 197 built-and-fired (1.0%)**; grade 3 (8 rows) detectable by *nothing automated* | FAMILY |

**The single most consequential row is not in the table: §1** — the wrong-instrument *verdict* on row 7
was itself reached with a wrong instrument. That is the arm's headline.

---

## 1. HEADLINE — the deletion was right, its stated reason was measurably wrong

`tools/audit_receipt_field_coverage.py` was **deleted** (first action of this arm; it was UNTRACKED,
`git status` = `??`, no git log — nothing was committed and nothing was lost).

**The deletion is correct on C2 grounds** (§4 row C2-1): it was new machinery built while identified
debt went unpaid, it was unvalidated, and it reintroduced three already-recorded poison classes
(substring-scan #829; hand-rolled keyword list where fmtools exists #259/#522; coarse name-for-value
proxy).

**The deletion's *other* stated reason is REFUTED by measurement.** The claim was: *"there is no wr1
receipt JSON; `S_ref_ceiling` is memo-internal only; a receipt↔memo set difference cannot see the case
it was built for."* Both legs are false:

| claim | measured |
|---|---|
| "there is no wr1 receipt JSON at all" | **FALSE.** `/Volumes/VertigoDataTier/pact/ddm_wr1_20260729/wr1_descent_receipt.json`, 7.4 KB, schema `ddm_wr1_reverse_waterfill.v1` |
| "`S_ref_ceiling` lives ONLY in the memo" | **FALSE in substance.** The receipt stores the quantity in all 10 descent rows as **`S_vs_ref_flipceiling`**; also `dseg_pred_ceiling`. Only the *spelling* differs from the memo's prose name |

**Why the error happened — and it is the disease itself.** Receipts live on the SSD custody tier per
CLAUDE.md's own *"Local Disk, SSD Spill, Auto-Cleanup, And Provenance"* non-negotiable. The search was
scoped to the repo (`.omx/research/`, which holds `ddm_wr1_reverse_waterfill_20260729.md` and **no**
wr1 `.json`). *"Not found in my search scope"* was reported as *"does not exist"* — a
**negative-existence claim made without exhaustive search**, which is the dominant false-claim class
already recorded today (`negative_existence_claims_are_the_days_dominant_error_class_20260731`). The
memo names the SSD path in plain text at `ddm_wr1_reverse_waterfill_20260729.md:159`.

**So the original `ddm_cf1` diagnosis was right and is fully reproducible.** Recovering the stored
column from the receipt:

| k cells | archive B | printed `S_ref (flipfree)` | stored `S_vs_ref_flipceiling` | ceiling vs ref 2.256641 |
|---:|---:|---:|---:|---:|
| 100 | 482,742 | 2.1985 | 2.1985 | −0.0581 |
| 200 | 409,534 | 2.1498 | 2.1498 | −0.1068 |
| 300 | 346,671 | 2.1079 | 2.1079 | −0.1487 |
| 400 | 297,368 | 2.0751 | 2.0751 | −0.1815 |
| **486 (Knee A)** | 274,333 | 2.0598 | 2.0598 | −0.1969 |
| 540 | 227,327 | 2.0285 | 2.0304 | −0.2262 |
| **600 (Knee B)** | 174,578 | 1.9933 | 2.0058 | −0.2509 |
| 660 | 118,245 | 1.9558 | **2.0108** ↑ | −0.2459 |
| 730 | 51,128 | 1.9111 | **2.1186** ↑ | −0.1381 |
| 768 (all) | 14,303 | 1.8866 | **2.2755** ↑ | **+0.0189** |

Three facts the printed table cannot show, all recovered from the stored column:

1. **For every k ≤ 486 the bracket is exactly 0.0000** — flipfree and ceiling coincide. Knee A's
   free-safe-floor is **STRONGER** than the memo argued, not weaker.
2. **The ceiling REVERSES beyond k ≈ 600** (2.0058 → 2.0108 → 2.1186 → 2.2755) while the printed
   column descends monotonically (1.9933 → 1.8866). A sign flip hidden behind a column choice.
3. **At k=768 the worst case is WORSE THAN DOING NOTHING** (2.2755 > 2.256641, **+0.0189**) while the
   printed column reads −0.370.

This is decision-relevant, not cosmetic: the memo's §3 recommends **Knee B (k=600)** as the sub-0.15
byte target, and the risk direction reverses immediately past it.

**The memo-internal tell, independent of any receipt.** `ddm_wr1_reverse_waterfill_20260729.md:54-56`
promises **three** S readings — `S_ref_flipfree` · `S_ref_ceiling` · `S_if_solved`. The table header at
line 58 carries **two** S columns — `S_ref (flipfree)` and `S_if_solved`. The omitted one is exactly
the one that reverses. *Three promised, two tabulated* is checkable inside the single document, with
no receipt and no SSD access.

`verdict_scope: INSTANCE` (this specific misdiagnosis) — but the *mechanism* (repo-scoped search over
an SSD-tier artifact class) is `FORMULATION`-scoped and will recur for any receipt-consuming audit.

**Net disposition:** DELETE stands. The instrument *shape* (receipt↔memo set difference) was
**correct** for wr1; its real defects were a one-line scope bug (globbed `.omx/research` only, never
the SSD custody tier) and name-matching keyed to prose spelling rather than receipt key. Recording
that honestly matters because the wrong lesson — *"receipt↔memo differencing cannot see this class"* —
would foreclose a shape that demonstrably works.

---

## 2. JOB B — fmtools: located, capability confirmed, and where a hand-rolled list stands in for it

**Located.** `~/Projects/fmtools` (NOT `src/tac/fmtools`). Consumed from this repo by SUBPROCESS
under a separate venv so the pact venv gains zero deps — the #259 firewall, documented at
`src/tac/confound_gates.py:1193-1198`.

**Key AND value confirmed** (operator's phrasing is accurate):
- `fmtools/decorators.py:20` — `local_extract(schema: type[T], ...)` returns a **caller-defined
  structured schema**, so the extracted object carries arbitrary fields.
- `fmtools/auditor.py` ships `audit_file` / `audit_directory` / `audit_diff` returning issues with
  **`severity` + `category`** (classification KEYS/LABELS) alongside **`message` + `suggestion`**
  (VALUES) — see the shape at `examples/code_auditor.py:36-45`. Classification and value extraction
  in one call.

**B-1 — the live stand-in: a semantic layer that is structurally un-fireable.**
`src/tac/confound_gates.py:1278` defines the P4 gate with `use_fmtools: bool = False`. Its UNCERTAIN
branch (a class that defines `observe`/`detect`/`classify` but whose *name* does not match the meter
regex) is explicitly **not counted as a violation** and is meant to be resolved by the fmtools
advisory `_fm_meter_advisory` (`confound_gates.py:1190`).

The orchestrator at `src/tac/preflight.py:6774-6775` is:

```python
for _eightfold_gate in _EIGHTFOLD_GATES:
    _eightfold_gate(strict=False, verbose=verbose)
```

It passes **only** `strict` and `verbose`. `use_fmtools` is never passed, and I found no call site
that sets it True (searched `src/`, `tools/`; the only other `use_fmtools` is
`tools/auto_push_main.py:318`, a *separate* function's own parameter, defaulting True — so fmtools is
live there). The one test that touches it,
`src/tac/tests/test_eightfold_gates.py:273`, asserts the **OFF** path performs no subprocess.

So inside P4 the fmtools layer is a **grade-2 orphan (built-never-fired)**, and un-fireable through
the orchestrator without a signature change. Per the standing law
(`default_off_is_orphaned_signal_activation_ledger_reconciliation_20260706`), "off" here is a silent
hardcoded default, not a tracked queue state. Sharpest form: **P4 is the "no meter without a canary"
gate, and its own uncertain branch has no canary — it defers to an advisory that cannot fire.**
`verdict_scope: INSTANCE`. LOAD-BEARING NOW (wired into `preflight_all`, warn-only).

**B-2 — a hand-rolled substring list doing semantic classification in a core sense organ.** See row
C1-1 below; it is both a job-B row and the arm's one newly-measured C1 row.

---

## 3. C1 — WRONG-INSTRUMENT rows

### C1-1 (NEW, measured by this arm) — `"MET"` matches inside `PHOTOMETRIC`

`src/tac/ddm_costate_organ.py:2202-2203` classifies deferral-ledger gate status with substring tests:

```python
_LEDGER_GATE_OPEN_TOKENS = ("OPEN", "MET", "FIRED MID-ARM", "GATE FIRED", "MEASURABLE_NOW")
_LEDGER_GATE_CLOSED_TOKENS = ("CLOSED", "BLOCKED", "PRE-ARC", "HELD", "STAGED")
```

applied at `:2260` as `gate_open = any(tok in joined_upper for tok in _LEDGER_GATE_OPEN_TOKENS)`.

**The asymmetry is the tell.** One line earlier, `:2257`, the *status* axis DOES carry a guard —
`and "FIRED" not in status`. The *gate* axis at `:2260` carries none. The author knew the failure mode
on one axis and not the other.

MEASURED on the live ledger `.omx/research/ddm_deferral_queue_ledger_20260729.md` (125 `| Q` rows):

| quantity | measured |
|---|---:|
| rows where `"MET"` appears as a substring | **33 / 125** |
| ...of which a real word-boundary `MET` | **5** |
| **substring false-match rate for the token `MET`** | **28/33 = 85%** |
| carriers | `PHOTOMETRIC` ×24, `TELEMETRY` ×10, `GEOMETRY` ×10, `METAL` ×3, `ARITHMETIC` ×2, `GEOMETRIC` ×2, `METHOD` ×2 |
| rows surfaced as gate-OPEN today | **29** |
| **of those, open SOLELY by substring accident** | **1** (QA55, via `METHOD`) |

**Placed point, not a verdict.** Today the instrument is 28/29 correct — the damage is bounded at 1
row because the *status* pre-filter (`DUE`/`ORPHAN`, no `FIRED`) happens to exclude most carriers. But
the token itself is 85% wrong on this corpus, so the true reading is *"one status-vocabulary change
away from a large false-positive wave"*, not *"a 3% error"*. In a campaign whose vocabulary is
`PHOTOMETRIC`/`GEOMETRY`/`TELEMETRY`, this token is maximally badly chosen.

LOAD-BEARING NOW: the costate digest is a **core sense organ**, auto-surfaced at SessionStart per
CLAUDE.md; this scan is the OWNERSHIP-ON-GATE-OPEN surface (`:2210`) built precisely so open-gate/
no-owner items are *machine-surfaced, not audit-discovered*. Actuation is ADVISORY.
`verdict_scope: INSTANCE` for QA55; `FORMULATION` for the token list.

**The fix is to REPAIR, not to build** (and is locally precedented): word-boundary matching already
exists in the sister module at `src/tac/confound_gates.py:2571` (`r"(?i)\bspec\b..."`). One regex at
`:2260`. I did **not** land it — see §6.

### C1-2 (already self-caught in-corpus; I verified it first-hand rather than citing)

`check_codex_findings_memos_consumed` (`src/tac/preflight.py:89295`) **was** the purest C1 specimen —
it aged memos by **mtime, a checkout artifact**, and measured **0 of 1,260 memos in scope while
printing LIVE COUNT 0** (docstring `:89309-89315`). A gate that looked at nothing reported clean.

It was **FIXED today** by `ddm_rg5` (#825/#821): age now derives from the filename stamp. My
independent re-run (not a citation):

```
10 violation(s) (218 in-window memo(s) scanned of 1258 total, window 30d by filename date)
```

The docstring records "10 violations over 273 in-window memos"; I measure **218** in-window with the
same 10 violations — the window slid, exactly the *running-census* behaviour `ddm_ba31` already named.

**Two derived findings, both mine:**
- **The fix is SOUND and belongs in the control set** (§5). It also supplies the campaign's best
  one-line statement of C1 (quoted in §0).
- **A STALENESS row, independently confirmed twice.** `.omx/state/operator_p0_ledger.jsonl` (last row,
  `p0_bug_class_sweep_20260717`, `written_at_utc` **2026-07-31T18:45:33Z**) still carries
  `next_action: "OWED, no owner assigned: check_codex_findings_memos_consumed 3-day scan window
  (0 of 1260 files in scope)"`. The fix landed in commit **`990674f4a9` at 2026-07-31 15:43:38 -0500**
  (`preflight.py:89118`: `_CODEX_FINDINGS_FRESH_SECONDS = 30 * 24 * 3600  # 30-day routing window
  (was 3d on a dead mtime axis)`) — i.e. the ledger row describing the defect as OWED was written
  **~3 hours after it was repaired**. Two independent derivations (my live gate re-run; a parallel
  sweep's static read of the commit timestamp) agree. Per
  `staleness_is_a_named_confound_class_freshness_at_consumption_20260723`, freshness belongs at
  CONSUMPTION. The C1 here is in the **ledger row**, not the gate.
  `verdict_scope: INSTANCE`. LOAD-BEARING NOW (the p0 ledger is a routing surface).

### C1-3 — `check_refusal_gates_have_live_positive_control`: covers 2 of the 5 incidents that motivated it

**The strongest C1 row in the sweep, because it is the meta-gate built to catch exactly this class.**
Definition `src/tac/confound_gates.py:3332`; wired **STRICT** via `_CONFOUND_STRICT` at
`src/tac/preflight.py:6726`. **LOAD-BEARING NOW.**

The inventory the operator's brief referred to as "4 of 23 covered, 19 NAMED" is **not a memo** — it
is computed at runtime by `positive_control_coverage()` (`confound_gates.py:3320-3331`) and emitted
only into the gate's `ok_detail` string. Reproduced statically: **5 registered controls covering 4
distinct gates out of 23 refuse-capable gates; 19 uncovered and NAMED**, against
`MIN_POSITIVE_CONTROL_COVERAGE = 4` (`:3316`).

Its docstring (`:3207-3210`) names **five** motivating incidents. The registered controls reproduce
**two**:

| motivating incident | control registered? |
|---|---|
| #829 — `ps -axo command` guard family skipped at FILE level | **YES** (`:3260`, `planted_split.py`) |
| #830 — raw-vm gate declared live-count 0 while measuring 6 | **YES** (`:3236`, `planted.py`) |
| lever registry AST'd **1 of 171** modules | **NO** — owning gates are in the uncovered 19 |
| duty queue enumerated **116 of 177** | **NO** — same two gates |
| findings gate scanned **0 of 1,260** | **NO** — `check_codex_findings_memos_consumed` lives in `preflight.py:89295`, **outside `CONFOUND_GATES`**, so it is not in the denominator at all |

**And the coverage ratchet is met by padding.** Controls 4 and 5 (`:3292`
`check_levelset_hosc_requires_beta_end`, `:3302` `check_no_duplicate_long_flags_in_launch`) belong to
gates that were **not** among the five incidents; their `why` fields cite generic hazards
(*"argparse last-wins silently discards the earlier value"*), not the narrowing class. They lift
distinct-gate coverage 2 → exactly **4**, which is exactly `MIN_POSITIVE_CONTROL_COVERAGE`. The floor
is satisfied by two incident-derived controls plus two convenient ones.

Two sub-rows:
- **Denominator mismatch** (`:3320`): `total_refuse_capable_gates = len(CONFOUND_GATES)` = 23, but the
  class header at `:3204` defines narrowing across the whole preflight surface — and one of its own
  five instances is a `preflight.py` gate, of which there are **440** `def check_*`. The instrument
  reports "4 of 23" for a class defined over a far larger population.
- **Comment/computation disagreement** (`:3403-3405`): the docstring says it *"deliberately does NOT
  include itself in its own coverage denominator"*, but `positive_control_coverage()` derives `gates`
  from `CONFOUND_GATES`, which at `:3405` **does** include it — and it duly appears in the uncovered 19.

`verdict_scope: INSTANCE` for the padding; `FORMULATION` for the denominator scoping.
**Important counter-weight (§5 control row 5): the gate's own mutation test is genuinely decisive** —
this row is about *coverage breadth*, not about a broken mechanism.

### C1-4 — the gate built because static checks were insufficient is satisfied by ten static checks in 40 ms

`check_lane_scripts_have_e2e_smoke_proof`, `src/tac/preflight.py:28416`; **STRICT** at `:5653`;
**gates lane dispatch**. Its stated purpose (`:5644-5648`): *"Closes the structural gap that cost Lane
RM-d 3.5h GPU: 63 STATIC preflight checks above all guard CODE PATTERNS, none actually run the deploy
→ inflate → contest_auth_eval pipeline locally."*

What it accepts: a JSON row from `experiments/canonical_local_auth_eval_smoke.py`, whose own docstring
(`:39-40`) states it *"short-circuits BEFORE the expensive 600-pair scorer loop. The smoke proves the
pipeline plumbing is correct, **NOT that the archive scores**."* Measured: all 205 proofs carry the
identical 10 stages, every one a static file/arity check, `elapsed_seconds` **0.036–0.09**. No stage
runs `contest_auth_eval`.

**Plus a dead window.** `SMOKE_PROOF_MAX_AGE_DAYS = 7` (`:28413`), but all 205 proofs in
`.omx/state/lane_e2e_smoke_proofs.json` are stamped between `2026-07-14T23:22:51Z` and
`2026-07-14T23:23:02Z` — an **11-second bulk write**, now 17 days old against a 7-day window, with only
20 of 204 lane scripts carrying an `E2E_SMOKE_OPT_OUT` waiver.

This is the purest C1 specimen found: the instrument answers *"does a plumbing artifact exist"* while
its charter is *"has the pipeline actually been run."* `verdict_scope: INSTANCE`. LOAD-BEARING NOW.

### C1-5 — `check_lane_classes_have_pipeline_proof`: "complete cycle demonstrated" backfilled as plumbing-only, no staleness window

`src/tac/preflight.py:28628`; **STRICT** at `:5665`. Charter (`.omx/state/lane_class_proofs.json`
header): *"a complete dispatch → train → archive → auth_eval cycle has been demonstrated."* Docstring
`:28653`: *"This catches the Lane RM-d class of bug PERMANENTLY."*

Measured: 17 proof rows; **13 are `proof_kind: "canonical-local-smoke"` with `score: None`**, noted
*"Plumbing-only Check 65 backfill"*. Only 4 are `production-deploy` with a score. The gate reads **no
timestamp window at all**, so a backfill row is permanently valid. Secondary drift: the docstring at
`:28657` says *"SHIPS WARN-ONLY initially (strict=False)"* while the call site passes `strict=True` —
the promotion happened in code and never in text. `verdict_scope: INSTANCE`. LOAD-BEARING NOW.

### C1-6 — `_validate_amrc_artifacts`: a STRICT validator whose glob matches zero files repo-wide

`src/tac/preflight.py:15555`, called at `:15536` from `preflight_filename_contract` (`:15329`), which
runs **STRICT** at `:1890` and `:4695`. Built 2026-04-26 for the *"Yousfi council #8 lossless
argmax-RLE mask codec"* (`:14946`). It scans `submissions/robust_current/**/*.amrc` and
`experiments/results/**/*.amrc` (`:15566-15569`); a repo-wide `find` returns **0** `.amrc` files, so the
validator returns `[]` unconditionally. `"masks.amrc"` is still listed as a live mask artifact at
`:14954` — the contract is declared while the artifact class is extinct in-tree.

Placed point: this is the *benign* end of the ladder (a validator guarding nothing costs a glob, not a
false clean), but it is the same shape as C1-2's pre-fix state and belongs on the same axis.
`verdict_scope: INSTANCE`. **DORMANT** (strict, but structurally inert).

### C1-7 — detection coverage, already measured by `ddm_sb2`, cited not re-derived

`.omx/research/ddm_ba31_negative_surfaces_20260731.md:270-273` already tabulates this class. Reproduced
verbatim because re-deriving it would be the rediscovery sin:

| instrument | coverage | reading |
|---|---:|---|
| `lever_registry` | **1/171 modules = 0.585%** | blind |
| findings gate (pre-fix) | **0/1,260 = 0%** | blind |
| duty-queue | **116/177 = 65.5%** | partial |

**Recall note, and it is the honest headline of this section:** the C1 sweep I was asked to run was
**already substantially performed in-corpus** by `ddm_sb2` (#819) and `ddm_ba31` on 2026-07-31, before
this arm existed. My additions are C1-1 (new), the C1-2 first-hand re-derivation + staleness row, and
the §1 correction. I am not claiming the class as a discovery.

---

## 4. C2 — BUILT-INSTEAD-OF-PAID

### C2-1 — the arm's own origin (the cleanest specimen, because it is ours)

`tools/audit_receipt_field_coverage.py`: ~300 lines, **untracked**, no production caller, built to
detect a restatement gap **while the identified debt was to fix existing detection surfaces** —
`lever_registry` at 0.585% coverage and the findings gate at 0% (§3 C1-3), both named in
`ddm_sb2`/`ddm_ba31` the same day. It also re-imported three recorded poison classes into a repo that
had already paid to name them. **Right action: DELETE** — executed. `verdict_scope: INSTANCE`.

This is why C2 outranks its own symptoms: each of the four defects in that tool was individually
fixable, and fixing them would still have left the debt unpaid.

### C2-2 — `S_vs_ref_flipceiling` needed no tool at all

The wr1 gap was recoverable by reading a stored column in an existing receipt (§1). The identified
debt was **restatement discipline in the memo** — print the third promised column. A ~300-line
cross-corpus detector was built for a gap that a table edit closes. `verdict_scope: INSTANCE`.

### C2-2b — the creation/repair rate, measured, with the caveat that keeps it honest

MEASURED over the 7 days 2026-07-24 → 07-31 (`git log --diff-filter=A`):

| quantity | measured |
|---|---:|
| NEW `.py` files added under `tools/` + `src/tac/` | **430** (~61/day) |
| ...of which under `tools/` | **130** |
| distinct tasks in `canonical_task_status.jsonl` | 143 (**44 pending**) |
| pending tasks explicitly naming WIRING / ORPHAN / STUB / registry debt | **4** (#819, #820, #821, #825) |

**The caveat is load-bearing and I will not drop it: 430 is not 430 orphans.** Each `ddm_*` arm
legitimately writes one-shot measurement scripts (`tools/analyze_ddm_*`, `tools/aggregate_ddm_*`), and
a one-shot analysis script is *supposed* to be single-use — calling those orphans would be the same
name-for-substance error this arm is auditing. The honest reading is a **rate asymmetry**: new-surface
creation runs at ~61/day while the named wiring debt sits at 4 open rows.

**And the debt is being paid, partially — the counter-evidence belongs here.** #821 (*"two hollow
UNOWNED gates"*) was **discharged in part today** by `ddm_rg5`: the findings gate went from vacuous
(0/1,260) to honest (10 violations over 218 in-window). That is the C2-correct action — repair an
existing surface — executed the same day, and it is why this row is a *rate* observation and not an
indictment. `verdict_scope: FORMULATION` (the rate), `INSTANCE` (each named task).

### C2-3 — the orphan-grade inventory is the standing measurement of this class

`ddm_sb2` (#819), via `ddm_ba31_negative_surfaces_20260731.md:281-291`:

| grade | count | detectable by |
|---|---:|---|
| 1. built-and-fired | **2** | everything |
| 2. built-never-fired | **165** | registry |
| 3. **BUILT-ELSEWHERE-UNWIRED-HERE** | **8** | **nothing automated** |
| 4. designed-stub | **10** (2 silent) | partial |
| 5. not-even-designed | **12** | nothing |
| **total** | **197** | — |

**2 of 197 = 1.0% built-and-fired.** That ratio *is* C2 measured: the repo's dominant output is
surfaces that were built and never fired. Repairs landed in the same unit (registry 1→171 modules,
factories 116→177, stubs 0→10, caching 1096×, *"because a slow gate is a disabled gate — which is how
this survived"*). **B-1 in §2 is a previously-unlisted grade-2 instance** — the P4 fmtools layer, built
and structurally un-fireable. `verdict_scope: FAMILY`.

---

## 5. DENOMINATOR AND CONTROL ROWS

**Examined:**
- **23** refuse-capable gates enumerated by name (the full `CONFOUND_GATES` tuple,
  `confound_gates.py:3170-3193` + the self-registration at `:3405`).
- **14** instruments opened at implementation / docstring / call-site level; **6** placed as C1 rows
  ⇒ **8 examined and not flagged**.
- **4** instruments measured against the live tree (proof-store timestamps, lane-script counts,
  findings-memo filename stamps, `.amrc` glob).
- **9** `preflight.py` glob patterns tested for zero-match: **1** returned zero (`*.amrc`); the other
  **8** matched between 2 and 3,710 files.
- **4** hardcoded day-windows swept in `preflight.py` (`SMOKE_PROOF_MAX_AGE_DAYS=7`,
  `_CHECK_298_STALENESS_DAYS=30`, `_CODEX_FINDINGS_FRESH_SECONDS`,
  `_CODEX_FINDINGS_CONSUMER_MTIME_WINDOW_SECONDS`).
- **189** hand-rolled token/keyword lists across 85 files (`rg` for
  `^_[A-Z_]*(TOKENS|KEYWORDS|WORDS|TERMS|MARKERS|PATTERNS)\s*[:=]` in `src/tac/` + `tools/`, excluding
  tests); **125** ledger rows; **1,258** findings memos via the live gate; **10** receipt descent rows;
  the 8-gate eightfold set and its orchestration.

**Hit rate, stated so it can be judged:** 6 of 14 opened instruments placed as C1 rows; 1 of 9 globs
degenerate; 1 of 4 windows dead. The instrument is not returning "everything is broken."

**Cross-derivation of the delegated rows (this arm re-measured them rather than citing).** Every
quantitative claim in ranked rows 1 and 8 was re-run first-hand and reproduced exactly:

| claim | re-measured |
|---|---|
| 205 proofs / 204 lane scripts / 20 opt-outs | **confirmed** (205 / 204 / 20) |
| proofs written in an 11-second bulk window | **confirmed** — 12 distinct timestamps, `2026-07-14T23:22:51` → `23:23:02` |
| every proof is the same 10 static stages | **confirmed** — exactly **1** distinct stage-set across 205 rows |
| `elapsed_seconds` 0.036–0.09 | **confirmed** — min 0.036 / max 0.090 |
| `*.amrc` matches zero files | **confirmed** — repo-wide `find` returns **0** |

This matters methodologically: the one place my own reading and a second derivation *disagreed*
(control row 6) is the one place an error was hiding. Agreement across two derivations is the only
reason these rows are stated as measured.

**Discrimination — most hand-rolled lists are FINE.** The large majority of the 189 match *literal
structural markers*, where substring matching is correct and fmtools would be wrong: e.g.
`src/tac/cost_band_calibration.py:1216` `_SMOKE_LABEL_TOKENS = ("__smoke__", "_smoke_", ...)` matches
a naming convention, not a meaning. **A hand-rolled list is only an infection when it stands in for a
SEMANTIC judgment** (is this load-bearing? is this verdict negative? is this gate open?). I did not
attempt to classify all 189 on that axis — see §6.

**CONTROL ROW 1 — SOUND, and it refines the disease definition.**
`src/tac/tests/test_rehearse_tr1_quant_engagement.py` (the #828 fix). Its fixture is *synthetic and
hermetic* (`:19-20`) — which by a naive reading would make it a "stand-in, not the incident." It is
not, and the reason is precise:
- it names the **real** incident with the real checkpoint path and MEASURED numbers (`:13-17`:
  `intra_seg_trunk_tau_ep00644.npz`, `token_max_abs` 0.06666672 = 1.0000008× the 1/15 half-step → 0.0
  exactly; 230 affected checkpoints);
- it asserts the incident's **PRECONDITION** is present in the fixture — `:93`,
  `assert engagement["build_default_quant_engaged"] is False  # the latent defect's precondition`;
- it asserts the **exact quantity the gate thresholds** (`:112`, `token_max_abs == 0.0`), naming the
  pre-fix value;
- it ships a **negative control** (`:118`, the `off` lineage must be unaffected) and a
  provenance test against the sister silent-correction bug (`:128`).

**Therefore "synthetic fixture" is NOT the C1 signature.** The signature is whether the fixture carries
the incident's *precondition* and the *thresholded quantity*. This distinction is measured, not
asserted, and it should govern any future positive-control audit.

**CONTROL ROW 2 — SOUND.** `check_codex_findings_memos_consumed` post-fix (§3 C1-2): honest scope
printed alongside the count (`218 of 1258`), a NAMED strict-flip condition, a named owner, and an
in-memo waiver path. Printing *scope* next to *count* is the structural cure for C1 and this gate now
does it.

**CONTROL ROW 3 — SOUND (self-correction, not mine).** `ddm_ba31:271` records that the
silently-wrong-instrument count moved **4 → 3 → 4 in one day**, with MAIN refuting its own round-1 root
cause on two independent legs. Framing the count as *"a coordinate that moves as the census proceeds"*
rather than a fact is the correct output form.

**CONTROL ROW 4 — SOUND, and it is the fixture standard.**
`check_process_guard_excludes_observer_flag_values`, control at `confound_gates.py:3260`. The control
reproduces the incident's *structure* rather than standing in for it: *"#829: the EXACT pre-fix
ru1/sb1 shape — `ps -axo command` enumeration with the enumeration, the token test and the decision
SPLIT across functions and module scope. Every leg was individually invisible to the function-scoped
predicate."* Sister: `check_no_raw_virtual_memory_safety_basis` +
`src/tac/tests/test_raw_vm_basis_gate_scope_and_denominator.py:63`, a **per-scope-leg** positive
control parameterized over `experiments/guard.py` and `scripts/guard.py`, each annotated *"#830: was
silently OUT of scope"*, and pinning the denominator against re-widening (`:93`, `:104`, `:112`).

**CONTROL ROW 5 — SOUND, and it is the direct counter-weight to C1-3.**
`src/tac/tests/test_refusal_gate_positive_control_class_guard.py:46`: *"THE DECISIVE TEST. Narrow a
real gate exactly the way all five measured instances were narrowed — make it scan nothing — and the
class guard must refuse."* It monkeypatches a real gate to return a clean OK over
`ok_detail="0 source file(s) scanned"`. On the axis *"does the meta-gate catch a gutted detector,"*
this is a genuine live assertion. **C1-3 is about coverage breadth, not a broken mechanism** — stating
both is the whole point of carrying controls.

**CONTROL ROW 6 — my own near-miss, reported because it is the same disease.** I first ran the costate
scan reading key `open_rows` and got `[]`, and was about to record "the scan returns 0 rows." The real
key is `open_gate_unfired_rows` (29). My independent re-derivation of the documented logic *also* gave
29 — the disagreement between my two methods is what caught it. **Cross-derivation caught what a
single instrument would have reported as a clean zero**, which is exactly the failure this arm audits.

---

## 6. WHAT I DID NOT COVER (explicit; no negative-existence claims)

1. **I did not land the C1-1 fix.** The one-line word-boundary repair at
   `ddm_costate_organ.py:2260` is a `.py` edit requiring the two-pass review gate; I chose to report
   rather than start a landing this arm could not finish. **Owed, owner unassigned.** It is a
   REPAIR of an existing surface — the C2-correct action.
2. **The 189 token lists are not individually classified** into literal-marker (fine) vs
   semantic-stand-in (infection). I examined roughly a dozen closely. The remainder is *unclassified*,
   not *clean* — this is precisely the "nothing looked at ≠ nothing wrong" distinction, and I decline
   to launder it.
3. **The grade-3 component inventory (8 BUILT-ELSEWHERE-UNWIRED-HERE rows) is cited from
   `ddm_sb2`/`ddm_ba31`, not re-derived.** I did not open those 8 components or verify their unwired
   status first-hand; a parallel sweep tasked with it had not reported at write time. The one grade-2
   instance I *did* derive myself is B-1 (the P4 fmtools layer). The 430-vs-4 rate in C2-2b **is**
   mine, with its caveat.
3b. **Positive-control coverage outside `CONFOUND_GATES` was not swept.** The 4-of-23 figure is now
   VERIFIED (C1-3) but its denominator is the 23-gate confound subset. The ~440 `def check_*` in
   `preflight.py` were **not** enumerated for positive-control coverage; that population is outside
   the meta-gate's declared denominator. Unknown, not clean.
3c. **The "4 of 23 / 19 NAMED" inventory is not a document.** Searched `.omx/research/*2026073*`,
   filename greps for `positive|control|refusal`, `docs/meta_bug_class_catalog.md`, and
   `.omx/state/canonical_task_status.jsonl` without finding a memo carrying it. It is computed at
   runtime by `positive_control_coverage()` and emitted only into the gate's `ok_detail` string —
   which is itself worth noting: **the coverage queue has no durable artifact.**
5. **I did not audit the wr1 memo's other columns**, nor re-run the wr1 gates. The §1 table is a
   READ of a stored receipt, not a re-measurement; `S_vs_ref_flipceiling` is the receipt author's
   own worst-case construction and I did not re-derive it from flip mass.
6. **Scope searched, stated plainly:** `src/tac/`, `tools/`, `experiments/` (Python);
   `.omx/research/` (md + json); `.omx/state/*.jsonl`; `/Volumes/VertigoDataTier/pact/ddm_wr1_20260729/`.
   I did not search other SSD custody directories, `reverse_engineering/`, or `docs/` beyond the
   catalog pointer. Anything I report as absent is absent **from that scope**, not from the repo.

---

## 7. THE ONE-LINE LESSON

The tool built to catch *"a quantity stored but never restated"* was itself judged by a search that
never looked where the quantity was stored. **Before an instrument may return "not found," it must
report its scope next to its count** — the cure `check_codex_findings_memos_consumed` already
implements (§5 control row 2), and the cure C1-1 still needs.

**Pointer `0.1910828242` [contest-CPU] UNMOVED. Nothing here is a score claim.**
