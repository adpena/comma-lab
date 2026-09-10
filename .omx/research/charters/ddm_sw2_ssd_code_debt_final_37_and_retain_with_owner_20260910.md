# ddm_sw2 — sw1's follow-on (gpt-5.6-sol, medium): drive the SSD authored-code debt from 37 to 0 and close the 20 retain-with-owner rows by naming each owner's fire trigger in the cs1 ledger; re-apply the one vr5 row (bz2d) the process gate refused on a false positive (charter, 2026-09-10)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm (gpt-5.6-sol, medium — mechanical custody) · Spawned by MAIN 2026-09-10. Sources: sw1 (`.omx/research/ddm_sw1_screening_census_and_ssd_code_disposition_20260910.md`; 223 blobs landed by MAIN in nine batches, bb9a4a48e; audit after: 37 AUTHORED-OWED, 83 certified), cs1 (`ce582b160`), vr5 apply journal (`484971b9c`: bz2d row refused `LSOF_SCAN_FAILED_AT_APPLY:LIVE_OWNER_PROCESS:ddm_bz2d` because a concurrent ruff argv named the arm — a pattern match on argv text, not a live consumer), `tools/audit_ssd_authored_signal.py`. Axes: apparatus; `score_claim=false`.

## MANDATE
(A) For each of the 37 remaining AUTHORED-OWED blobs: commit through the serializer where the source is real and an owner memo exists (batches ≤ 25; `.py` = 2 review passes + ruff), or certify with `--certify <sha> --owner <arm> --rationale`, or record retain-with-owner with the owner named. Re-run the audit; target 0 owed. (B) For the 20 retain-with-owner rows: append one cs1 ledger row each naming the owner, the exact fire trigger, and the date. (C) The bz2d raw: propose (do not run) the narrowed process-gate rule — match the owner token against process COMMAND names and cwd, not against the full argv text — as a code change + test in `experiments/ddm_vr3_certified_raw_reclaim.py`; MAIN re-runs the apply.

## PRIOR-LAW PREDICTION (m38)
- ≥ 30 of 37 committable; ≤ 7 retain-with-owner; after-count ≤ 7. The process-gate narrowing keeps every vr3/vr5 test green and refuses a real owner process in the test fixture.
- **FALSIFIER:** if the after-count cannot reach ≤ 7 without weakening certification, report the residue with reasons.

## SCOPE
Custody + one small gate-precision change. No SSD deletions (MAIN applies). No score work.

## HARD CONSTRAINTS
- `upstream/` READ-ONLY; never touch live trees or payloads. Serializer commits w/ post-edit `--expected-content-sha256`; if git object writes are refused (sw1's rc=17), leave `landing.patch` + manifest as sw1 did. Tokens `[no-triality] [p0-ledger-ok]`; NEVER a Co-Authored-By or AI-attribution trailer. Checkpoint `tools/subagent_checkpoint.py --subagent-id ddm_sw2`.
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).

## PRIOR NEGATIVE SIGNAL
- sw1: source-blob insertion is refused in the sandbox (rc=17) — do not retry nine times; one attempt, then the patch.
- vr4/vr5: an empty lsof in a sandbox proves nothing; the narrowed gate must still fail closed when `ps`/`pgrep` are unavailable.

## OPTIMAL FORM
- Reference form: sw1 (`b4c23432f`, `bde9d7453`) and vr5's executor tests (`463337c85`). SCOPE reductions: none. MECHANISM reductions FORBIDDEN: no bulk certification without rationale; no gate weakening.
- **PRIOR-LAW PREDICTION (falsifiable):** as above.

## DELIVERABLE
Memo `.omx/research/ddm_sw2_ssd_code_debt_final_20260910.md` with before/after counts, the 20 owner rows, the gate change + test. Commit via the serializer. End with the live frontier line.
