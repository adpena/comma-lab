# ddm_ql2_apparatus_debt — close the three NON-scorer items ql1 left owed: (1) the pre-existing STRICT gate `check_no_bulk_write_strands_the_ready_record` at live count 11 > 10, (2) V9/PBR2 rematerialization + the broken `tools/build_taskspace_inverse_stack_receipt.py` (hardcoded strict_source_reopen=True), (3) the bounded-target-G receipt regeneration (no scorer; delta already measured) — with the authority MAIN grants below

## MANDATE

Operator standing GO + the proactive-harden law. `ddm_ql1_retired_lineage_test_quarantine_20260903.md` (f1f0abd27,
commit f1f0abd27) took the witness_dsl suite from 35F/16E to 0/0 and left four owed rows; three need no scorer.
MAIN AUTHORITY GRANTED HERE: (a) regenerating the sealed July bounded-target-G receipt is authorized as an
APPEND-ONLY supersession (new receipt file + a provenance row naming the pin-refresh cause and the measured
3-field delta; the old receipt is retained, never overwritten); (b) fixing the tool's hardcoded flag is a bug fix.

## SCOPE

1. **Gate live-count 11 > 10** (locate at source: `grep -rn check_no_bulk_write_strands_the_ready_record src tools` — it is NOT in tac.preflight's namespace; found in: src/tac/confound_gates.py src/tac/tests/test_payload_write_order_gate.py ): enumerate the
   11 violations with file:line; for each, decide FIX (route the write through the canonical ready-record path)
   or WAIVE with the gate's own same-line waiver and a real rationale; drive the live count under the cap and
   record before/after. Do not raise the cap.
2. **V9/PBR2:** rematerialize the teacher-census receipt per ql1's row (three of 13 manifest sources moved);
   fix `tools/build_taskspace_inverse_stack_receipt.py` so `strict_source_reopen` is a real CLI flag (grep its
   argparse; never invent), with a test; un-quarantine the inverse-stack test if the receipt regenerates cleanly.
3. **bounded-target-G:** regenerate the receipt under the authority above; un-quarantine its test; retain both
   receipts with shas.
4. Run the full `src/tac/witness_dsl` suite detached (`.venv/bin/python tools/launch_detached_process.py
   --output-dir <run_dir> --done-receipt <name> --nice 10 --nice-best-effort -- <cmd...>`) and report counts.

## HARD CONSTRAINTS

- `upstream/` and `submissions/semantic_joint_ctxmix/` READ-ONLY. NO scorer/Modal/Metal. Never touch
  `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/`, ft1's or xr1's files/stores.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 review passes; ruff clean; never bare git.
- The V15 compile receipt (19 tests) needs scorer authority — NOT in scope; leave its quarantine.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_ql1_retired_lineage_test_quarantine_20260903.md` — the four owed rows and the two-drift-commit finding.
- `ddm_cd1_working_tree_debt_landing_20260903.md` — the registry/fixture repair already landed (563b093e3).

## OPTIMAL FORM

- Family exemplar (reference): ql1's disposition discipline, `.omx/research/ddm_ql1_retired_lineage_test_quarantine_20260903.md`
  (commit f1f0abd27); the gate's own docstring + tests in `src/tac/preflight.py` / `src/tac/tests/`.
- SCOPE reductions: none. MECHANISM reductions FORBIDDEN: no cap raise; no test deletion; no assertion weakening.
- **PRIOR-LAW PREDICTION (falsifiable):** the 11 violations are ≤ 3 distinct write sites (bulk writers repeat);
  fixing them lands the count ≤ 10 without a waiver. FALSIFIER: ≥ 6 distinct sites — count it plainly.

## DELIVERABLE

`.omx/research/ddm_ql2_apparatus_debt_20260903.md` — per-item table, suite counts, RECALL EVIDENCE,
NEXT_IF_RESUMED, DEAD-ENDS. Commit via the serializer. Cite `docs/operating_manual_craft_handoff.md`. End with
the own-vehicle frontier line.
