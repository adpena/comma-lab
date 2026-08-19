# ddm_sp3 — preflight-debt consolidation (4 items)

**Arm:** `ddm_sp3` · **Date:** 2026-08-19
**Authority:** `[macOS-CPU advisory]` — apparatus repair. No scorer run, no archive, no exact row.
**Pointer delta: NONE.** This is a means. It is not frontier progress, and it does not move the
own-vehicle frontier.

Four filed items, each verified at source before any edit per the charter-recall clause.

| # | Item | Premise | Outcome |
|---|---|---|---|
| 1 | M1 meta-cure: the CLASS-POPULATION line (task #1146) | **LIVE** | **LANDED** warn-only, Catalog #348 scope extension |
| 2 | fp16 clamp-then-cast gate (fx4's debt) | **DONE BY SIBLING** `98f24b3379` | verified; my 3 hardening repairs were absorbed into it |
| 3 | fx3's two #330 detector improvements | **(a) LIVE · (b) LIVE but unsound-as-specified** | (a) **LANDED** with measured evidence; (b) **NOT landed**, spec gap named |
| 4 | #1149 registry round-trip, HEAD RED (Catalog #344) | **LIVE** | **FIXED** — gate green on HEAD |

**Gate status on HEAD after this landing:** `check_evidence_authority_claims_are_custodied(strict=True)`
**GREEN** (item 4). `check_quantize_degenerate_range_clamped_correctly(strict=True)` GREEN.
`check_modal_harvesters_record_call_id_outcome(strict=True)` GREEN (item 3a did not add a row).

No new catalog number is claimed. All three protections are **scope extensions of existing gates**,
per the 2026-07-20 Catalog #351 precedent recorded in CLAUDE.md's "2026-07-14 catalog amendments"
section: *"a scope extension, not a new gate or number, per the post-#400 Catalog #299 consolidation
rule."* The #299 quota brake stands at 407/400.

---

## ITEM 1 — the CLASS-POPULATION line (meta-bug M1)

### The meta-bug, restated from the measurement

The two-landing rule's TRIGGER is manual for arm-discovered defects. Measured anchor: `ddm_jg2` fixed
its own 3-byte zip defect at ONE site and reported it; nobody swept the 20-site population until the
operator asked. A class-fix that stops at the incident site leaves the class live everywhere else,
wearing green.

### Host gate: Catalog #348, chosen by invariant not by convenience

Two candidates were read before choosing. **Catalog #229**
(`check_subagent_landing_includes_premise_verification_evidence`) has the right *machinery* — dated
memo glob, cutoff, waivers — but the wrong *surface*: `_check_229_memo_dir` resolves
`~/.claude/projects/.../memory/`, not `.omx/research/`. **Catalog #348**
(`check_new_gate_landing_includes_retroactive_sweep_evidence`) reads `.omx/research/` and carries the
same invariant: *a landing that changes the defect state owes a sweep beyond its own incident site.*
#348 enforces that on the TEMPORAL axis (sweep history for verdicts the class already tainted); this
extension enforces it on the SPATIAL axis (state how many other live sites the class has). Same
invariant, different field — exactly the #351 precedent's phrasing.

### The predicate was CALIBRATED on the live corpus, not guessed

The charter's literal predicate (fix-markers `FIXED|CURED|defect|bug` + a code-fix commit reference)
was measured first and **rejected on the evidence**: over `.omx/research/*.md` dated ≥ 2026-08-12
(280 memos) it fired on **198**, of which **196** would have been flagged. A 196-row warn-only gauge
is one nobody reads — the flood failure `ddm_sp2` refused when it kept its ZIP guard at library
scope, and the sister of vacuity-equals-pass. Tightening to uppercase-verdict + commit-proximity
overshot the other way (1–2 triggers, and it MISSED `ddm_fx4`'s own memo, which writes "fixed"
lowercase).

The landed trigger is three shapes, calibrated against three known-positive exemplars:

| trigger shape | 7-day hits |
|---|---|
| `two-landing` / `second landing` / `self-protection` | 23 |
| `fixed … N sites/files/callsites` | 2 |
| `**FIXED**` / `\| FIXED` verdict marker | 4 |
| **union (the landed trigger)** | **25 of 280** |

All three exemplars (`ddm_fx3`, `ddm_fx4`, `ddm_sp2`) trigger AND pass. Requiring a commit reference
in addition was measured and **dropped**: it cost a true positive (`ddm_fx4` stopped triggering).

**Acceptance** is a population token (`CLASS-POPULATION` | `live count` | `live sites` |
`population count/measured/swept`) **with a digit within 4 lines**, or a substantive
`# CLASS_POPULATION_WAIVED:<reason>` (Catalog #287 placeholder rejection).

### What the gauge reads if the cure is applied and nothing else changes

It requires a NUMBER, not a word: adding the header "CLASS-POPULATION" with no measurement moves it
by exactly zero. **Executed negative control:** neutering the digit-proximity check makes the
word-alone fixture wrongly PASS (1 → 0 violations); neutering the trigger makes the positive control
go dead (scanned 1 → 0). Both halves are load-bearing.

**The honest limit:** this gate can only force the arm to STATE a population. It cannot verify the
number is correct. That limit is real, and it is one reason this lands warn-only.

### Live count, and the forward-binding cutoff

Cutoff `20260820` — **the gate binds forward, and the 7-day corpus was deliberately NOT backfilled**
per the charter. Live count today is therefore **0 violations / 0 memos scanned**, and the gate
reports that denominator every run so it can never read as silently green.

Measured calibration (what the gate WOULD have flagged over the trailing 7 days, recorded as evidence
and left alone): **18 offenders of 25 triggering memos.** Top offenders:
`p0_1111_round11_fixes_20260818` · `ddm_wc2_hpac_mps_port_20260814` ·
`ddm_up3_thirteenth_move_byteclose_20260819` · `ddm_sr1_submission_gauntlet_20260817` ·
`ddm_rg2_red_gate_triage_20260816` · `ddm_pu3_falsified_premise_propagation_20260816` ·
`ddm_lw2_liveness_cure_20260816` · `ddm_hd1_na9_hazard_hardening_20260818` ·
`ddm_er1_error_class_ledger_and_determinization_20260817` · `ddm_av3_fresh_eyes_review_20260816`.

**Strict-flip condition:** the extension's rows are returned to the caller but are excluded from the
raise (`test_extension_rows_are_returned_but_never_raise` pins this). Flip when the forward window
has run long enough to show a steady-state live count near 0 — i.e. arms are writing the line — at
which point move `class_pop_violations` into the `PreflightError` in
`check_new_gate_landing_includes_retroactive_sweep_evidence`. Note that host gate is itself wired
`strict=False` in `preflight_all`, so nothing here can block a commit today.

---

## ITEM 2 — fp16 clamp-then-cast: DONE BY SIBLING, and my hardening is inside it

**Verdict: done by `ddm_sp2`, commit `98f24b3379`** — landed STRICT as a Catalog #161 scope extension
at live count **0 violations / 36 guard-and-cast sites scanned**, with the predicate extracted to
`src/tac/fp16_floor_guard.py` and `ddm_fx4`'s 87-line duplicate deleted (one detector, not two).
MAIN routed this debt to two owners; this arm did not re-land it.

I reviewed the module before the deconfliction arrived and found **three defects**, all of which
`ddm_sp2` absorbed into `98f24b3379` (the 3-tuple producer contract won). Re-verified against the
COMMITTED module:

| defect I found | why it mattered | control vs committed module |
|---|---|---|
| the canonical cure written on the NEXT statement was reported as a violation | a STRICT gate refusing a correctly-cured site tells the engineer to do what they already did | `cured NEXT statement` → clear ✔ |
| `#` inside a string literal blinded the WHOLE file | truncation unbalanced brackets → statement splitter returned nothing → file scanned clean AND added 0 to the denominator, so the vacuity guard could not see it either | `hash-in-string + defect` → CAUGHT ✔ |
| no waiver path at all on a STRICT gate | no escape hatch for a genuinely unauditable site | substantive waiver → clear; `<reason>` → CAUGHT ✔ |

`src/tac/fp16_floor_guard.py` is byte-clean against `98f24b3379` in my tree; I did not recommit it.
The committed suite `test_fp16_scale_floor_guard.py` has **zero** coverage of these three repairs, so
this landing adds that regression coverage in `test_ddm_sp3_*` — consuming the SAME shared detector,
not a second one. Wall-clock measured after the repairs: **1.54 s**, 5.1% of the 30.0 s
`DEFAULT_PREFLIGHT_CLI_TIMEOUT_S`.

---

## ITEM 3 — fx3's two Catalog #330 detector improvements

### (a) String-literal awareness — LANDED

**Premise VERIFIED at source.** `_check_330_line_is_comment_or_literal` was a line-level lexical
guess: it asked *"does this line START with a quote?"*, so it DISAGREED WITH ITSELF on one file — a
hit inside an f-string was skipped while a docstring CONTINUATION beginning with `print(` was falsely
flagged.

**Cure: imported, never re-typed** (per the deconfliction and the split-bank discipline).
`_check_330_line_is_comment_or_literal` is deleted; `_check_330_code_lines` now calls
`tac.fp16_floor_guard.neutralize_prose`, the shared tokenize-based neutralizer.

**One measured adjustment was required.** `neutralize_prose` only *sanitizes* single-line strings (it
strips brackets but keeps text) because fp16 needs `astype("float16")` to remain visible. Catalog
#330's token, `FunctionCall.from_id`, carries **no bracket** — so a one-line f-string holding a
printed recovery command survives sanitization and reads as a live harvester. Measured on this repo:
the weaker form adds **3 false positives** to a STRICT, currently-green gate
(`modal_scorer_introspection.py:556`, `modal_click_polish_cpu.py:395`, `modal_train_lane.py:3087` —
all f-string recovery commands). Rather than fork the detector, `neutralize_prose` gained a
keyword-only **`blank_all_strings=False`** parameter: additive, default-preserving (fp16 behaviour
byte-identical), one implementation serving both questions.

**Measured after:** `check_modal_harvesters_record_call_id_outcome` still **0 violations**; all 3
f-string cases still skipped; `experiments/modal_ot_offset_n600_gate.py` — the file fx3 measured the
self-disagreement on — now has **0 hits treated as code**, i.e. the false flag on its docstring
continuation is cured and the correct skip is preserved. **Executed negative control:** the
reconstructed old guard gets 2 of 3 fixtures wrong, so the new tests are load-bearing.

### (b) The asymmetric `-20/+180` detection window — NOT LANDED, deliberately

`ddm_sp2`'s recorded caution is binding and I confirmed it rather than re-deriving it: **naively
widening the backward window trades a false positive for a false NEGATIVE** — an unrelated
`append_terminal_call_id_ledger_event(` far above would falsely clear a real offender. The cheap half
is not a cure; it moves the error to the direction that hides defects.

**Spec gap, named:** the sound half is call-graph resolution, and it is not specified. To land it one
must decide (1) what counts as "the mirroring helper is reachable from this call site" — same
function only, same module, or across imports; (2) how to resolve indirection (a helper called
through a dict/dispatch table or passed as a callable is not statically resolvable); (3) what the
gate does when resolution FAILS — fail-open (re-introducing the false negative) or fail-closed
(re-introducing the false positive). Until (3) has an answer, the improvement has no defined
behaviour on the cases that motivated it. Owner: the next `preflight.py` owner; do not land the
window-widening half alone.

### A regression I introduced and caught in my own review pass

Worth stating plainly, because a fix is unreviewed new code. My first version of (a) called
`_check_330_code_lines(text)` for **every** `.py` file under the scan roots, tokenizing thousands of
files that never mention the token. Measured: **17.98 s — 60% of the 30.0 s
`DEFAULT_PREFLIGHT_CLI_TIMEOUT_S`, inside a STRICT gate.** Review pass 2 caught it; the cure is the
same superset pre-filter discipline `ddm_sp2` applied to fp16 (skip a file that cannot host a
violation).

| | wall-clock | violations |
|---|---|---|
| HEAD (old lexical guard) | 4.23 s | 0 |
| my first version | **17.98 s** | 0 |
| landed | **3.79 s** | 0 |

The landed version is faster than HEAD, because the pre-filter also skips the per-line scan on files
without the token.

`ddm_fx3`'s remaining named debt is not preflight-surface and stays with its owners:
`tools/codex_companion_spawn.sh:96-100` (`rc=unknown_detached`, blocked while the codex lane is walled
to Aug 20) and the absent shared waiter helper.

---

## ITEM 4 — task #1149, the registry round-trip (Catalog #344)

### Premise VERIFIED: HEAD was RED

`check_evidence_authority_claims_are_custodied(strict=True)` raised on
`.omx/state/canonical_equations_registry.jsonl:750` — `equation=realization_necessity_preimage_per_stratum_v1`,
`anchor=3` not JSON-round-trip exact at `$[3].vs1_rescope_reason`, `$[3].vs1_rescope_utc`.
(`ddm_sp2` flagged this file as "dirty, another arm owns it" — that is now **stale**: the file was
clean in the working tree and was mine to repair.)

### What the red actually was — not a serialization nit

Root cause, from git rather than inference: commit **`5ab6506630` "VS1 hygiene receipt" (2026-08-05)**
edited the **2026-07-19** row **in place** — 1 insertion, 1 deletion, no new row. That single edit:

1. changed `anchor_id` `exact_plane_storage_rate_dead_family_20260719` → `…formulation_20260719`;
2. narrowed `empirical_output.verdict_scope` from *"family (exact-plane storage under ANY lossless
   entropy stage)"* → `FORMULATION:EXACT_REVERSIBLE_L3_RASTER_RESIDUAL_RATE_DEAD`;
3. added `non_closed_plane_description_classes`;
4. added the two keys the gate can see: `vs1_rescope_reason`, `vs1_rescope_utc`.

So the gate's red is the visible **residue of a history rewrite** — a Catalog #110/#113 APPEND-ONLY
violation. Corroborating tell: `vs1_rescope_utc` = `2026-08-05`, later than the row's own
`written_at_utc` = `2026-07-19`.

### The repair, split on a principled line

The auditor scans **every** row, so appending a superseding row cannot clear line 750 — confirmed by
reading `audit_empirical_anchor_roundtrip_fidelity`. The repair therefore had to touch that row, and
I split it where the evidence splits:

* **The 2 unmodeled keys are INERT and mine to remove.** `_equation_from_dict` does not model them, so
  they were dropped on every read — **measured: the before-repair consumer object already lacked
  them**, and nothing in the entire repo reads `vs1_rescope_*` (1 occurrence, that line). They are
  not authored history; they are the residue of the violating commit. Removing exactly those bytes is
  partial restoration, not mutation. Done through the writer's own locked primitives
  (`_registry_lock` + `load_equation_registry_strict` + `_save_ledger` — the same load/modify/save the
  writer performs internally).
* **The `anchor_id` / `verdict_scope` changes are LIVE canonical state and NOT mine to guess.** Reads
  are latest-row-wins, and that row is the latest for its equation, so every consumer has read the
  rescoped values for two weeks. Reverting them would adjudicate CONTENT. Per the charter I stopped
  and am reporting the decision instead (below).

I also did **not** manufacture a `domain_refined` event to re-home the rescope prose: the rescope is
anchor-scoped, `domain_of_validity` is equation-scoped, and inventing canonical structure to hold two
inert strings would be worse than recording them here. **Preserved verbatim, so nothing is lost:**

> `vs1_rescope_reason` = *"prior family wording was overbroad; five lossless codecs close exact
> reversible L3 raster residual storage only"*
> `vs1_rescope_utc` = *"2026-08-05T00:00:00Z"* (also recoverable from commit `5ab6506630`)

The rescope's substance already lives in modeled, round-trip-safe fields on that same row:
`anchor_id`, `empirical_output.verdict_scope`, `non_closed_plane_description_classes`, and the event
`notes` (*"VS1 re-scoped exact-plane storage from family to FORMULATION:…"*).

### Proof the repair changed nothing a consumer can see

| check | result |
|---|---|
| reconstructed equation `to_dict()` sha256, before | `85c5000fc361926d4f3b3ffd98a9c52afd72a520f20fd8ff548038f7c727aecd` |
| reconstructed equation `to_dict()` sha256, after | `85c5000fc361926d4f3b3ffd98a9c52afd72a520f20fd8ff548038f7c727aecd` — **identical** |
| rows in file | 905 → 905 |
| lines changed | exactly 1 (line 750); `HEAD-line-750 minus the 2 keys == current` verified |
| round-trip violations over all 905 rows | 5 → **0** |
| `check_evidence_authority_claims_are_custodied(strict=True)` | **no longer raises** |

The 11 rows that gate still returns are `ddm_sp2`'s warn-only GT-lineage extension, which by design
never raises — unchanged by this repair.

### DECISION OWED to the operator (not guessed)

The 2026-08-05 in-place narrowing of `anchor_id` + `verdict_scope` on a 2026-07-19 row stands. Two
options, both content decisions:

1. **Leave it.** Current consumer state is the rescoped (narrower, more accurate) verdict. Cost: an
   APPEND-ONLY violation stays in the ledger, and the family-scoped anchor as originally authored is
   only recoverable from git.
2. **Restore + re-append.** Restore line 750 to its authored bytes and append the formulation-scoped
   anchor as a NEW anchor via `update_equation_with_empirical_anchor` — the shape 2026-08-05 should
   have used. Cost: the anchor list goes 4 → 5 and the superseded family-scoped anchor becomes live
   again in that list, where it can be mis-cited as a current family verdict.

I recommend (2) with the superseded anchor explicitly labeled, but it changes what consumers read and
is not mine to decide.

---

## CLASS-POPULATION

Eating this landing's own dogfood. (This memo is dated `20260819`, before the gate's `20260820`
cutoff, so the gate does not scan it — the line is written because the discipline binds, not because
a gauge demands it.)

```
CLASS 1 — anchor keys the canonical model does not model (item 4)
  registry events swept ........ 905
  anchors swept ................ 1,982      <- the denominator
  rows with dropped keys ....... 1          (line 750)
  keys ......................... 2          (vs1_rescope_reason, vs1_rescope_utc)
  fixed ........................ 1 row / 2 keys
  out of scope ................. 0
  SINGLETON, not a class: the sweep proves it, rather than assuming it.

CLASS 2 — "leading quote => prose" lexical guesses in src/tac/preflight.py (item 3a)
  candidate sites (lexical census) .. 18
  measured-defective and fixed ...... 1     (_check_330_line_is_comment_or_literal)
  UNADJUDICATED debt ................ 17
  NOT closed. Each remaining site needs its own check of whether the guess is
  load-bearing for that gate; several are the benign full-line `#` skip wearing
  a similar shape. This is honest debt, not a cleared class.

CLASS 3 — fp16 floor destroyed by its own cast (item 2)
  owned and closed by ddm_fx4 (35 sites / 22 files) + ddm_sp2 (98f24b3379).
  This arm added 3 detector repairs, population 1 (the single shared detector).
```

---

## Verification performed

* **`ruff check --select F`** clean on all three edited files.
* **`test_ddm_sp3_preflight_debt_consolidation.py` — 27 passed** (positive + negative controls for
  every protection: caught / cured / waiver-accept / waiver-placeholder-reject / pre-cutoff not
  backfilled / non-fix memo not scanned / missing dir / warn-only isolation / vacuity denominator).
* **107 passed** across `test_fp16_scale_floor_guard.py` + `test_ddm_sp2_two_landing_protections.py`
  + this suite.
* **429 passed** across `test_preflight_meta_bugs.py` + `test_v9_provenance_gates.py` +
  `test_check_351_canonical_producer_identity_scope_extension.py`.
* **Executed negative controls** (not asserted — run): item 1's two cure-halves each neutered in
  process and shown to break the protection; item 3a's old guard reconstructed and shown wrong on 2
  of 3 fixtures.
* **Live gate runs on HEAD:** `check_evidence_authority_claims_are_custodied(strict=True)` green ·
  `check_quantize_degenerate_range_clamped_correctly(strict=True)` green (0/36) ·
  `check_modal_harvesters_record_call_id_outcome(strict=False)` 0 violations ·
  `check_new_gate_landing_includes_retroactive_sweep_evidence` extension prints its denominator.
* No stale references to the deleted `_check_330_line_is_comment_or_literal` remain outside the
  docstring that records what it replaced.

## Files

* `src/tac/preflight.py` — item 1 gate + helper; item 3a rewire (deletes the lexical guard).
* `src/tac/fp16_floor_guard.py` — item 3a: `neutralize_prose` made public + additive
  `blank_all_strings` parameter. No change to fp16 behaviour.
* `src/tac/tests/test_ddm_sp3_preflight_debt_consolidation.py` — 27 tests (new).
* `.omx/state/canonical_equations_registry.jsonl` — item 4, 1 line, 2 inert keys removed.
