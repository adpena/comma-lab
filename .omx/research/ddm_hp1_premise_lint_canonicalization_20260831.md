# ONE MATCHER, BOTH SURFACES — the falsified-premise lint canonicalized (tools/premise_lint.py), the keeper delegated, the MEMO surface finally guarded; plus the 7 registry receipts the schema test correctly demanded

Date: 2026-08-31 · Author: MAIN · Cost: **$0**
Axis: apparatus. `score_claim=false` · `promotable=false`
`verdict_scope`: instrument build + wiring — operator 2026-08-31 "Continue hardening and polishing
and optimizing and making automatic…" + "All can be engineered and designed optimally there are no
walls" (the no-walls steer upgraded this from duplicate-with-note to ONE canonical module).

## 1. What landed

1. **`tools/premise_lint.py`** — the CANONICAL matcher. `lint_text(text, registries, subject=…)`
   ports the keeper's implementation verbatim (same normalisation, same warning format, same
   advisory/silent-on-every-failure contract), with the surface noun parameterized and a CLI
   (`--file --subject --registry --strict-rc`; rc=0 always unless `--strict-rc`).
2. **Keeper delegation** — `codex_arm_queue._lint_falsified_premises` now delegates to the module
   (importlib by path; registries resolution stays keeper-side so test overrides of
   `FALSIFIED_PREMISE_REGISTRY` keep working; any failure → the documented silent `[]`). The
   retired inline body was DELETED, not kept — git history is the forensic record.
3. **The memo surface** — `subagent_commit_serializer` gained a fail-open advisory sibling of the
   SHIFT-LEFT block: staged `.omx/research/*.md` content runs through the same matcher BEFORE the
   commit; warnings print to stderr as `PREMISE-LINT …`; NEVER blocks. This closes fpr1 §4's open
   design decision ("whether MAIN's memo-write path gets the same check"): it does, structurally.
   The jt1 propagation vector — a MAIN memo restating a number whose owning memo had published a
   do-not-cite list, crossing zero lint — is now a fired advisory at commit time.
4. **Registry receipts backfilled** — the keeper's schema test correctly REFUSED my 7 new rows
   ("asserts death with no receipt"): every row now carries `falsifications` with the correcting
   memo path + measured value + scale. The test was right; the rows were under-schema'd.

## 2. Executed controls

- Keeper delegation: positive (double-hit text → 2 warnings, `charter restates …`) + negative
  (clean text citing the CORRECTED numbers → silent) + full suite
  `tools/tests/test_codex_arm_queue_falsified_premises.py` **11/11 green**.
- Memo surface: THIS memo is the positive control — it quotes the retired wrong usage
  "349× as the distortion ratio" (do NOT cite; the honest number is a proxy-understatement factor,
  see the registry row's receipt), so committing it through the serializer must print a
  `PREMISE-LINT` advisory AND still commit (advisory ≠ veto). The .py-only commit alongside is the
  negative direction (no memo staged → silent). Both observed at commit time; receipts in the
  commit stderr capture below.

## 3. Design notes (why this shape)

- **Curated registry only** — the auto-scraped corrections index stays retired for this purpose
  (window-adjacency rows cannot carry quantity identity; fpr1 §1's fake-precision verdict stands).
- **A correction memo that quotes a wrong usage fires the advisory honestly** — the warning is a
  re-derive prompt, not a veto; that is the intended behavior, documented in the serializer block.
- Warning cap 5/memo; unicode-dash normalisation preserved so "13–23%" matches "13-23%".

## 4. Denominator

Files: 1 new + 2 edited (.py ×3, two review passes each via review_tracker). Dead code deleted:
2,687 chars. Registry rows patched: 7 (receipts). Tests: 11/11 keeper + module import + executed
controls. Dollars: 0. Pointer: UNMOVED (apparatus).

---

## ADDENDUM (ddm_eq1, 2026-09-04) — the equations leg

<!-- # FORMALIZATION_PENDING:canonicalization of the premise linter: an apparatus change plus its lint counts. The counts are the tool's own scope, not a physical row. The law it would need is a premise-decay law (how fast a charter's premises go stale); not derivable until premises carry measured expiry evidence rather than dates. -->

**No canonical equation.** Canonicalization of the premise linter: an apparatus change plus its lint counts. The counts are the tool's own scope, not a physical row. The law it would need is a premise-decay law (how fast a charter's premises go stale); not derivable until premises carry measured expiry evidence rather than dates.
