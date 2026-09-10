# ddm_vg1 — reclaim cwd census fix; visible-host validation pending

2026-09-10. `[no-triality] [p0-ledger-ok]`. Axis: **[macOS-CPU process/custody apparatus]**. `research_only=true`; `score_claim=false`.

The proposed fix passes **67 tests; 2 live integration tests are SKIPPED** because this session cannot launch ps. Ruff passes and two visible delta reviews are recorded through the unmodified review tracker in the isolated workspace. No reclaim/apply ran, no SSD payload was changed, and **0 bytes were reclaimed**. The exact-score frontier did not move. The charter is **PARTIAL: source and controlled-input verification complete; paired visible-host capture and real-process controls remain owed**. The one serializer attempt returned **rc=9, WAIT_AND_RETRY**, before Git object insertion: the older ddm_sw2 checkpoint still claims the same two Python files, although the hot state records sw2 as quit and cd3 as landed. No ownership guard was bypassed. The final verified patch and post-edit hashes are retained in `ddm_vg1_20260910/landing.patch` and `landing_manifest.json`; **no live commit landed**. The attempted packet is separately retained as `attempt1_landing.patch`; only this memo changed afterward to record the refusal.

## Finding and scope

MEASURED: `lsof -n -P -d cwd -F pcn` returns rc=0, empty stderr, and **422 distinct process rows**, all with COMMAND, fcwd and absolute name fields, but **no PID 1**. Its original stdout is retained byte-for-byte as text in the fixture. `ps -axo pid=,ppid=,comm=` raises PermissionError; `pgrep -x ddm_bz2d` returns rc=3 with process-list-unavailable diagnostics. Those failures are retained too. This is a real sandbox capture, NOT a paired unsandboxed ps/lsof census.

SOURCE-VERIFIED: the old gate demands PID 1 in ps and then demands every ps PID in lsof's cwd set. Thus omitting PID 1 from otherwise well-formed lsof output suffices to raise MAIN's exact error. The charter's likely missing-cwd root cause is supported by code and this capture; it is not fully established on MAIN's original process population. The prediction that at least 5% of host PIDs lack cwd remains UNMEASURED: no successful ps denominator exists here. lsof itself did NOT fail in this capture, so the charter's lsof-command-failure falsifier did not fire here.

Only `process_gate` changes in the executor (AST comparison verified). It keeps a ps command map and a separate lsof PID set. A missing cwd is classified as `cwd_unreadable`, with PID and command, including omitted ps PIDs and command-only lsof records. Either command source still refuses an owner token. `COMPLETE` means every observed PID has a classification; it does not mean every cwd was readable. An unreadable generic-command process could have an owner cwd; acceptance of that residual uncertainty is the charter's explicit command-only policy for unreadable PIDs, not proof that all processes are idle.

Independent visibility anchors remain mandatory: ps must see init and self; lsof must see self plus another ps-listed PID. An lsof-only observer cannot establish that second anchor. Tool failures/warnings, malformed or duplicate records, missing COMMAND fields, empty/self-only/observer-only censuses, owner commands, readable owner cwd, and exact-name pgrep matches still refuse. Full argv is never searched. Unrecognized relative-name/error fields still refuse; no speculative permission-string allowlist was added. All archive/runtime/hash/reference/descriptor and protected-tree gates are unchanged.

## Verification

- Existing vr3/vr5 suite: 57 cases retained. Added controlled cases: 10 passes for omitted/command-only/final unreadable rows, owner refusal from either command source, actual lsof replay with explicitly controlled ps/self identity, actual ps-denial replay, observer-only, warning, and duplicate unreadable records.
- Both real process controls are implemented: copied executable named `ddm_bz2d` in a matching cwd must refuse; a waiting reviewer shell with the token only in argv must pass. Both skip before launching a child when ps is unavailable. Neither was exercised here. Run these on MAIN's visible host before treating this change as host-validated.
- The actual lsof fixture is 422 rows; its test uses a SYNTHETIC ps table. It is parser regression evidence, not a claim that real ps succeeded.
- Review receipts: `ddm_vg1_20260910/review_pass1.json`, `review_pass2.json`; commands/results: `tests.json`, `ruff.json`. Reviews apply to post-edit hashes. No Python review override was used.
- Source edits were made only in an isolated small copy. Shared production source, upstream, live candidate trees, staged index and the reclaim ledger were not edited by this arm. No large payload, job or scorer was launched. Local artifacts are small source/manifests, not discarded payload measurements.

## MAIN re-apply command — NOT executed

The unchanged ledger SHA is `36416b17a70b4814fee681bc5c438b0ef87dec25f4b418eae5317c4ff97b6413`: **20 rows, 1 DELETABLE row, 3,662,409,600 admitted B**. This is a ledger admission, not fresh raw/hash/liveness clearance. After landing and visible-host verification, from `/Users/adpena/Projects/pact`:

```sh
.venv/bin/python experiments/ddm_vr3_certified_raw_reclaim.py apply \
  --ledger .omx/research/ddm_vr5_reclaim_plan_bz2d_revalidate_20260910.jsonl \
  --expected-ledger-sha256 36416b17a70b4814fee681bc5c438b0ef87dec25f4b418eae5317c4ff97b6413 \
  --journal .omx/research/ddm_vr5_apply_journal_bz2d_20260910.jsonl \
  --target-bytes 3662409600
```

The argv was copied from MAIN's failed launch manifest and verified against the actual parser; it is also in `ddm_vg1_20260910/main_reapply_argv.json`. It intentionally retains every fail-closed revalidation. If the ledger changes, this command must refuse and MAIN must use a newly verified plan digest.

Visible-host test command after landing:

```sh
.venv/bin/python -m pytest -q -rs experiments/tests/test_ddm_vr3_certified_raw_reclaim.py experiments/tests/test_ddm_vr5_certified_raw_reclaim.py
```

Require both live cases to run, not skip. Preserve a paired ps/lsof capture and the gate receipt. If lsof itself fails there, retain the block and investigate the charter's proposed ps-command census plus per-candidate `lsof -p PID -d cwd -Fn` path; do not waive visibility.

## RECALL EVIDENCE

Read the charter/common contract, PROGRAM, governing CLAUDE/AGENTS no-fake, mutation, storage, checkpoint, review and serializer rules, operating manual, current hot state, actual gate/tests, failed launch and prior cd3/vr4 receipts. No ddm_vg1 predecessor checkpoint existed. Memory-registry query `reclaim|cwd.census|process.gate|vg1` found no match; no memory fact is used.

Independent research content query: `cwd census|cwd_unreadable|PROCESS_VISIBILITY_UNAVAILABLE|process_gate` over `.omx/research` Markdown. Index/DAG/design/task query: `certif.*reclaim|process.gate|cwd census` over the canonical research index, capstone DAG, docs and canonical task JSONL. Exact queries/results are retained in `research_recall.json` and `design_tasks_recall.json`. The canonical-equations CLI returned its registry; no reclaim/cwd-census entry was found in that returned scope (`equations_recall.json`). Beyond charter seeds, the local-disk reclaim cadence distinguishes deleted logical bytes from actual freed capacity, and the task ledger retains owner/liveness fire conditions. This kept our result at 0 reclaimed B and our re-apply explicitly gated; it did not justify broadening this patch into cleanup. No alternative cwd parser was found in these searched index/DAG/design scopes.

The contract's old frontier literals are superseded by the current canonical pointer/hot state. Six scientific integration hooks are N/A: this apparatus unit changes no scorer, codec, law, optimizer, allocator or dispatch. Consumer wiring is the retained landing packet and MAIN's existing bz2d apply journal.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN; consumer store `.omx/research/ddm_vg1_20260910/landing_manifest.json` and main Git; fire trigger this packet is harvested and MAIN reconciles the abandoned ddm_sw2 checkpoint ownership claim. Verify the patch/hash manifest, rerun the serializer with the final post-edit hashes, and land only this reviewed delta in a Git-writable session if not already landed.
- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN; consumer store `.omx/research/ddm_vg1_20260910/`; fire trigger the reviewed source is available on a visible host. Retain paired ps/lsof output and run both real-process integration controls without skips; confirm unreadable rows are classified and the gate preserves refusal cases.
- **QUEUED-WITH-A-FIRE-ORDER** — owner MAIN; consumer store `.omx/research/ddm_vr5_apply_journal_bz2d_20260910.jsonl`; fire trigger host validation passes and the ledger SHA still matches. Execute the exact command above, allowing every existing revalidation to refuse on drift.

## LIVE-HYPOTHESES

- Missing cwd rows explain MAIN's failure: the captured lsof excludes PID 1 and the old predicate requires it. The paired unsandboxed capture is still needed to determine the full missing-PID population and confirm all host-specific shapes.
- The unchanged bz2d admission may become reclaimable after the fix: the ledger still admits exactly one row. Current apply-time references, payload hashes and liveness remain unmeasured here.

## DEAD-ENDS

- Requiring cwd for every ps PID is rejected as a reliable macOS parser rule for the captured shape: valid lsof output omits init.
- Empty/self-only lsof or unavailable ps/pgrep as idleness proof remains rejected; retained regressions refuse these.
- Full-argv owner matching remains rejected: a reviewer may merely name the arm, and the existing narrowed command/cwd tests preserve that distinction.
- Claiming all charter validation complete is rejected for this session: both live controls skipped and no successful paired ps capture exists. Negative scope: this session/host-visibility instance, not the parser formulation generally.

Existing own-vehicle frontier, unchanged by this arm: **S 0.13766931482209038 @ 180,186 B [contest-CUDA T4 n600]** (current local canonical pointer; no new score measured).
