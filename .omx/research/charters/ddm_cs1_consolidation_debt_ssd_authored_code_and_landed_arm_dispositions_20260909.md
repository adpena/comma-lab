# ddm_cs1 — consolidation debt: certify-or-land the SSD-authored code the sweep flags, disposition every landed arm of the 09-08/09 wave in the lane registry, and refresh the consolidation caches (charter, 2026-09-09)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm (astra medium) · Spawned by MAIN 2026-09-09 under the operator's standing GO. Source: the SessionStart consolidation-debt monitor (`[consolidation-debt] CONSOLIDATE-NOW … stale_commits: 176 · ssd_only_code: 118 (cache stale since 2026-09-01) → commit clean artifacts · disposition landed arms · route findings to canonical_equations/DSL/tasks/memory`). Apparatus: `score_claim=false`; no scorer, no Modal.

## MANDATE

The wave of 2026-09-08/09 landed fourteen arms (sj1, pc2, fe1, rw1, rc2, rc3, tc1, gf2, gf3, gb1, ane3, cl3c, gov3, scg1/scg2, ql1, vr3) and MAIN landed nine fallback bundles by hand. The consolidation monitor's SSD sweep cache is eight days stale, so its `ssd_only_code: 118` is not current, and the lane registry / task ledger dispositions for the landed arms are partial. This arm makes the debt CURRENT and drains what is drainable: refresh the SSD-authored-code cache, certify-or-land every code file that lives only on an SSD tier (the `tools/audit_ssd_authored_signal.py` contract: a `.py`/`.sh` under `/Volumes/*/pact/**` that no committed file matches is either LANDED into the repo through the serializer, or CERTIFIED as rebuildable scratch with its reproducer, or BLOCKED with a named reason), and write one disposition row per landed arm (lane registry gate marks with evidence paths; task-ledger completions with the memo filename beside each id — the m89 rule).

## SCOPE

1. Recall: `tools/audit_ssd_authored_signal.py` (`--write-cache`), `tools/lane_maturity.py` (audit/mark/validate), `tools/extract_canonical_tasks_from_directive.py`, `.omx/research/arm_final_messages/*_2026090[89]T*.md` (every final message of the wave), `.omx/state/codex_arm_queue.next_if_resumed.jsonl`. `tools/subagent_checkpoint.py read --subagent-id ddm_cs1` first.
2. Refresh: run the SSD sweep with `--write-cache`; report the CURRENT counts (the denominator, not the stale 118).
3. Certify-or-land: for each SSD-only code file — if it is an arm's committed-elsewhere script (compare by content hash against HEAD), record MATCHED; if it is genuinely unlanded arm code that a memo cites, land it through the serializer (two review passes per `.py`); if it is scratch, write a certify row (path, sha256, bytes, reproducer, reason) — never delete. Report the three counts.
4. Dispositions: for each landed arm of the wave, `tools/lane_maturity.py mark <lane> --gate <gate> --evidence <memo/seal path>` for the gates its memo proves (impl_complete / real_archive_empirical / contest_cuda where a T4 row exists — moves 33/34 and, if promoted by then, 35 — memory_entry where a memory file exists); `validate` must pass. Task ledger: mark every arm-registered ITEM whose memo says DONE as completed (with the memo filename), leave the rest pending with their fire triggers.
5. Memo `.omx/research/ddm_cs1_consolidation_debt_20260909.md`: the before/after of every monitor component; equations leg: apparatus → `# FORMALIZATION_PENDING:consolidation apparatus — no measured law`; run the Catalog #344 check before the final message.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. No scorer. NEVER delete, move, or rewrite any file on the SSD tiers (certify rows only). Never edit the live pointer tree, `submissions/semantic_joint_ctxmix/`, or the trees of live arms (sj1, fe1, rw1 — Opus; cmp1 pending) beyond reading them; do not land code from a LIVE arm's tree (its own commits own it) — land only from arms marked landed/closed in the keeper (`tools/codex_arm_queue.py status`).
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).
- CPU ≤ 2 procs; hashing tens of GB is detached via `tools/launch_detached_process.py … --nice 10 --nice-best-effort` with a resumable ledger; no `nohup`/`&`/clock waiters; artifact-bound waits ≤ 780 s.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 review passes (`review_tracker mark-file` rescans). Tokens `[no-triality] [p0-ledger-ok]`. NEVER a Co-Authored-By or AI-attribution trailer. Sandbox git refusal → serializer fallback bundle + receipt, named in the final message (MAIN lands it).
- ALWAYS KEEP THE PAYLOAD (certify ledger with shas). VERIFIED-AT-SOURCE for every count you report (rerun the tool; never quote the stale cache).
- Checkpoint discipline: `tools/subagent_checkpoint.py --subagent-id ddm_cs1 …` every ~10 tool uses.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- Hand `rm`/`mv` on SSD tiers is CLOSED by the custody contract (vr2/vr3 memos `.omx/research/ddm_vr2_vertigo_reclaim_round2_20260831.md`, `ddm_vr3_*`) — this arm writes certify rows, nothing else.
- Task-ledger ids without the owning memo filename are unresolvable by arms (m89; `.omx/research/ddm_cl3c_closer_20260908.md` registered five ids WITH the filename — the reference shape).
- "Finished, unharvested" charters get resurrected by `saturate --spawn` (memory `codex_keeper_saturate_resurrects_stale_rank1_finished_charters_20260908`) — when you disposition a landed arm, also `mark --status landed` in the keeper if it is not already.
- ExFAT AppleDouble `._*` companions are not code (memory `both_ssds_usable_20260810`) — exclude them from the sweep's denominator and say so.

## OPTIMAL FORM

- Family exemplar: cl3c's closer (`.omx/research/ddm_cl3c_closer_20260908.md`, commit e57f42d32) — the reference form for a bookkeeping closer: re-prove the row, regenerate the report, register tasks with filenames, append-only edits to the owning memo.
- SCOPE reductions declared per row: dispositions limited to the 09-08/09 wave's arms (SCOPE; older debt is a later closer). MECHANISM reductions FORBIDDEN: no cache quotes without a rerun; no "landed" gate without the evidence path; no deletion.
- **PRIOR-LAW PREDICTION (falsifiable):** the current SSD-only code count is ≥ 150 (the stale 118 plus this wave's arm scripts) and ≥ 80 % of it is MATCHED-by-content or landable arm code; the residual scratch certifies without a single BLOCKED row. FALSIFIER: any SSD-only file that is neither matchable nor certifiable (no reproducer) — list it as BLOCKED with the reason; that is a real finding.

## DELIVERABLE

The memo with the before/after monitor table, the certify ledger `.omx/research/ddm_cs1_ssd_code_certify_20260909.jsonl`, the lane-registry/task-ledger dispositions. Commit via the serializer. End with the live frontier line.
