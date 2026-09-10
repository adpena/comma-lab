Prepared the fix, but **landing and host validation remain blocked**.

- **67 tests passed; 2 live controls skipped** because `ps` is sandbox-blocked. Ruff and two review passes passed.
- Real `lsof` capture contains 422 valid rows but omits PID 1, which the old gate requires.
- Serializer refused with **rc=9**: abandoned `ddm_sw2` checkpoint ownership overlaps both files.
- No apply, deletion, or score measurement ran.

[Memo and exact re-apply command](/Users/adpena/Projects/pact/.omx/research/ddm_vg1_reclaim_gate_cwd_census_20260910.md) · [Verified patch](/Users/adpena/Projects/pact/.omx/research/ddm_vg1_20260910/landing.patch) · [Hash manifest](/Users/adpena/Projects/pact/.omx/research/ddm_vg1_20260910/landing_manifest.json)

Frontier unchanged: **S 0.13766931482209038 @ 180,186 B [contest-CUDA T4 n600]**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — Owner MAIN; consumer store `ddm_vg1_20260910/landing_manifest.json`; trigger: reconcile `ddm_sw2` ownership. Land the verified patch through the serializer.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner MAIN; consumer store `ddm_vg1_20260910/`; trigger: visible host with patched source. Capture paired ps/lsof output and run both live controls without skips.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner MAIN; consumer store `ddm_vr5_apply_journal_bz2d_20260910.jsonl`; trigger: host validation passes and ledger hash remains unchanged. Execute the memo’s exact re-apply command.

## LIVE-HYPOTHESES

- Missing cwd rows explain MAIN’s failure: the captured omission conflicts directly with the old predicate; paired host verification remains owed.
- The admitted bz2d row may become reclaimable after validation; current apply-time safety checks remain necessary.

## DEAD-ENDS

- Requiring every ps PID to expose cwd rejects the captured valid output.
- Empty/self-only censuses and unavailable process tools remain insufficient clearance.
- Full-argv matching misidentifies reviewers; command/cwd matching remains preserved.