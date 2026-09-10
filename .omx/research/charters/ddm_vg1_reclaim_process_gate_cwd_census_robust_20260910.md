# ddm_vg1 — make the narrowed reclaim process gate's cwd census robust on this host (it now FAILS CLOSED on MAIN's own apply with "PROCESS_VISIBILITY_UNAVAILABLE: malformed or incomplete cwd census"), keep every fail-closed property, and hand MAIN the exact re-apply command for the bz2d row (charter, 2026-09-10)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm (gpt-6-astra, high) · Spawned by MAIN 2026-09-10. Sources: cd3 (`.omx/research/ddm_cd3_ssd_code_debt_final_20260910.md`, `a7cb75034`: the gate narrowed from full-argv matching to command name + cwd; 57 tests), the failed apply (`.omx/tmp/codex_runs/vr5_apply_bz2d_20260910/run.log`: `FATAL: CertifyError: PROCESS_VISIBILITY_UNAVAILABLE:malformed or incomplete cwd census` — run by MAIN outside any sandbox, with `ps`/`pgrep`/`lsof` available), the fresh plan ledger `.omx/research/ddm_vr5_reclaim_plan_bz2d_revalidate_20260910.jsonl` (1 DELETABLE row: bz2d, 3,662,409,600 B; sha 36416b17…), vr4/vr5 (an empty lsof under a blind sandbox proves nothing — the gate MUST still fail closed when process visibility is genuinely absent). Axes: apparatus; `score_claim=false`.

## MANDATE
Find why the cwd census is malformed on a visible host (likely: `lsof -d cwd` output shape on macOS — kernel/zombie rows without a cwd, permission-denied rows for other users' processes, or a parse that requires every pid to have a cwd row) and fix the census so that (a) a visible host with some unreadable cwd rows yields a COMPLETE census with those pids recorded as `cwd_unreadable` (and the gate refuses only if an unreadable pid's COMMAND matches the owner token), (b) a host with no process visibility still raises PROCESS_VISIBILITY_UNAVAILABLE. Tests: the malformed shape captured from this host (record the real `lsof`/`ps` output as the fixture), the blind-sandbox shape, a real owner process (command named ddm_bz2d with cwd under the target) → REFUSE, a reviewer process whose argv merely names the arm → PASS. Do not run apply; hand MAIN the exact command (ledger sha unchanged unless you re-plan).

## PRIOR-LAW PREDICTION (m38)
- Root cause is a parse that treats any pid lacking a cwd row as "incomplete"; ≥ 5 % of pids on this host lack one (kernel_task, launchd children, other users). Fix ≤ 40 lines; 57 + ~6 tests pass.
- **FALSIFIER:** if the census is malformed because `lsof` itself fails (permissions), say so and propose the `ps -o pid,comm` + `lsof -p <pid> -Fn` per-candidate path instead.

## SCOPE
One gate function + tests + memo. No SSD deletions.

## HARD CONSTRAINTS
- `upstream/` READ-ONLY; no live-tree writes; `.py` = 2 visible review passes + ruff; serializer commits w/ post-edit `--expected-content-sha256`; if git object writes are refused, leave `landing.patch` + manifest after ONE attempt. Tokens `[no-triality] [p0-ledger-ok]`; NEVER a Co-Authored-By or AI-attribution trailer. Checkpoint `tools/subagent_checkpoint.py --subagent-id ddm_vg1`.
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).

## PRIOR NEGATIVE SIGNAL
- vr4: blind-sandbox lsof; cd3: full-argv false positive — both fail-closed properties must survive this fix (tests).

## OPTIMAL FORM
- Reference form: cd3's gate change + tests (`a7cb75034`) and vr5's executor tests (`463337c85`). SCOPE reductions: none. MECHANISM reductions FORBIDDEN: no "skip the census when lsof is odd".
- **PRIOR-LAW PREDICTION (falsifiable):** as above.

## DELIVERABLE
The fix + tests + memo `.omx/research/ddm_vg1_reclaim_gate_cwd_census_20260910.md` + the re-apply command for MAIN. Commit via the serializer. End with the live frontier line.
