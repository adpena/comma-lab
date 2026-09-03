# ddm_ql1_retired_lineage_test_quarantine — make the witness_dsl suite honest: adjudicate the 51 red tests (35 failed / 16 errors) of the retired July taskspace/ep725/G-series lineage — PIN-REFRESH with a current bit-exact receipt, or QUARANTINE with the owning memo + drift commit, or name a REAL regression — so the semantic vehicle's tests never hide behind a red baseline

## MANDATE

Operator standing GO + the proactive-harden law. `ddm_cd1_working_tree_debt_landing_20260903.md` (d50da5258)
left three typed fire orders for "retired maintainers" (EP725 decode receipts · V15 compile custody · PBR2/V9
teacher census) and MAIN's directive `ddm_cd1_directive_test_drift_ledger_20260903.md` retains the full
failure list (log: `/Volumes/VertigoDataTier/pact/ddm_cd1_working_tree_debt_landing/witness_dsl_full_suite_20260903.log`).
A suite with 51 known-red tests cannot catch a 52nd. This arm owns the adjudication; cd1 already repaired the
registry race and the stale pointer fixture (landed 563b093e3).

## SCOPE

1. For each red module (ep725 adapter/bounded encoder · monolithic PGA receiver · selected-preimage v1/v2 ·
   g17 actuator IR · g17/g49 bridge · g72 analytic factor compiler · g82 lowering · inverse-stack receipt):
   name the pinned source/renderer/custody that drifted, the commit that moved it (`git log -S`), and whether
   the current output is bit-exact against the retained receipt (RUN the decode where custody exists on the
   SSDs; do not assume).
2. Disposition per module: **PIN-REFRESH** (current output bit-exact → refresh the pin with a receipt) ·
   **QUARANTINE** (retired lineage, no current consumer: mark the tests `xfail(strict=True)` with the owning
   memo + drift commit in the reason, never delete) · **REGRESSION** (a live consumer depends on it: name it,
   do not quarantine; hand MAIN a typed fix order).
3. Re-run the full `src/tac/witness_dsl` suite: target 0 failed / 0 errors with every quarantine documented.
   Also clear the 3 canonical-equation "surfaces once" tests if cd1's repair did not (verify).

## HARD CONSTRAINTS

- `upstream/` READ-ONLY; `submissions/semantic_joint_ctxmix/` READ-ONLY. NO scorer/Modal/Metal.
- Never delete a test; never weaken an assertion to pass; quarantine = strict xfail with provenance.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes; the suite's
  before/after counts in the memo.
- Do not touch gc1/gf2/xr1 files.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_cd1_working_tree_debt_landing_20260903.md` — the three fire orders and their live hypotheses (EP725
  drift from safe-extraction hardening outside decoder arithmetic; V15's last change validation-oriented).
- `ddm_cd1_directive_test_drift_ledger_20260903.md` — the verbatim failure list.
- memory `harness_monitor_dies_rc144_use_bg_until_loop_20260903` — hook import class (fixed 6079666de7).

## OPTIMAL FORM

- Family exemplar: cd1's landing discipline, reference `.omx/research/ddm_cd1_working_tree_debt_landing_20260903.md`
  (commit d50da5258) and its registry/fixture repair (563b093e3).
- SCOPE reductions: none. MECHANISM reductions FORBIDDEN: no blanket skip; no assertion weakening.
- **PRIOR-LAW PREDICTION (falsifiable):** cd1's hypotheses predict EP725 and V15 are PIN-REFRESH (output-identical)
  and the PGA/preimage/g17/g82 family is QUARANTINE (retired, no live consumer); zero REGRESSIONS. FALSIFIER:
  any module whose current output differs from its receipt AND has a live consumer — count it plainly.

## DELIVERABLE

`.omx/research/ddm_ql1_retired_lineage_test_quarantine_20260903.md` — the per-module table, suite counts
before/after, RECALL EVIDENCE, NEXT_IF_RESUMED, LIVE-HYPOTHESES, DEAD-ENDS. Commit via the serializer. Cite
`docs/operating_manual_craft_handoff.md`. End with the own-vehicle frontier line.
