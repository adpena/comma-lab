# ddm_cd1_working_tree_debt_landing — land or revert every stale in-flight edit in the shared working tree (the "ssd_only_code / stale_commits" hook debt): per path, find the owning memo/arm, verify its tests, commit through the serializer with an attributing message, or revert with a receipt — never leave a silent diff that a later landing inherits

## MANDATE

Operator standing GO + the proactive-harden law (m141) + the consolidation hook (CONSOLIDATE-NOW: pile
files 55, stale commits 60, ssd-only code 118 blobs). MEASURED cost today: an uncommitted in-flight edit
to `tools/preflight_hook.py` (a `from tools.argv_role import` from a blocked-git arm) made EVERY commit
that staged a test module fail with `ModuleNotFoundError` until MAIN found and fixed it (6079666de7);
fpc1/fpc2/wc3 each needed MAIN to land fallback bundles by hand. The working tree still carries ~31
modified and several untracked files from arms whose sandboxes could not write git objects. Every one
is a landmine for the next landing and a possible lost result.

## SCOPE

1. **Inventory** every `git status --porcelain` path EXCEPT live-arm WIP (arms ddm_fpc3, ddm_gc1,
   ddm_ol1 are LIVE — leave any file they own or are writing; check the arm keeper logs / their charters
   for ownership) and the charters/arm_final_messages dirs (MAIN custody). For each path: the diff size,
   the last memo/arm that names it (grep `.omx/research` + `.omx/state/commit-serializer.log` +
   serializer fallback bundles under `/Volumes/*/pact/*/commit_serializer_fallback*`), and whether a
   fallback bundle commit contains the SAME bytes (then the landing is a bundle import, byte-verified).
2. **Verify** per group: `ruff` on `.py`; the tests the owning memo names (or the module's own test
   file); preflight hook fast pass. Report per path: LAND / REVERT / HOLD(live-arm) with the reason.
3. **Land** groups through `tools/subagent_commit_serializer.py` with post-edit `--expected-content-sha256`,
   ONE logical unit per commit, message naming the owning memo + bundle sha; `.py` needs the review
   gate (two passes via `tools/review_tracker.py mark-file`); non-code with `REVIEW_GATE_OVERRIDE=1`.
   If git writes are denied in your sandbox, produce the verified bundle + a landing table for MAIN
   (that is a valid outcome; do not fake a landing).
4. **Revert** only a diff whose owner memo says it was superseded/abandoned AND whose bytes are already
   preserved in a fallback bundle or SSD custody — write a receipt naming where the bytes live. When
   in doubt: HOLD with the reason, never delete.
5. **Test drift ledger:** the 3 canonical-equations "surfaces once" failures and the 4 witness_dsl
   source-drift/renderer-oracle failures on HEAD (see `ddm_gs3`/hot state) — diagnose each (which pin
   drifted, which commit moved it), fix if the fix is a pin refresh with a receipt, else list as owed.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY; `submissions/semantic_joint_ctxmix/` READ-ONLY. Do NOT touch any file owned
  by a LIVE arm or the QBR1 burn's sealed source tree (`/Volumes/VertigoDataTier/pact/ddm_wc3_*`).
- No scorer/Modal/Metal. Serializer only; never bare `git commit`; never `git checkout --`/`git stash`
  on files you did not adjudicate.
- ALWAYS KEEP THE PAYLOAD: any reverted bytes must already exist in a bundle/SSD custody named in the
  receipt under `/Volumes/VertigoDataTier/pact/ddm_cd1_working_tree_debt_landing/`.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- memory `harness_monitor_dies_rc144_use_bg_until_loop_20260903` — the hook import bug class.
- `feedback_concurrent_subagent_commit_message_swap_20260429` (memory) + the serializer's rc=12/13
  refusals (override with Python; gitignored paths) — the landing rules.
- `ddm_dd1_drift_debt_ledger_verdict_20260901.md` — the ledger-drift audit; cite, don't repeat.
- `ddm_ht1_red_debt_hygiene_verdict_20260901.md` — gate-4 law; 84 lanes research_only.

## OPTIMAL FORM

- Family exemplar: MAIN's fpc1/fpc2/wc3 bundle landings today (commits d5a721f9f7, 2079b4bb93 and the
  wc3 code commit; procedure: `git bundle verify` → byte-compare working tree vs bundle commit → tests →
  review marks → serializer), reference `.omx/research/ddm_fpc1_full_pipeline_compress_20260903.md`
  (commit 2296c6bad8 bundle) and the serializer contract `tools/subagent_commit_serializer.py`
  (commit 31e716f64a).
- SCOPE reductions: none needed. MECHANISM reductions FORBIDDEN: no `git add -A`; no landing without
  the owner memo named; no revert without a custody receipt.
- **PRIOR-LAW PREDICTION (falsifiable):** ≥ 80% of the stale diffs correspond byte-for-byte to a
  serializer fallback bundle (blocked-git arms), so most landings are bundle imports; FALSIFIER: fewer
  than half match any bundle — count it plainly (it means arms edit without the serializer).

## DELIVERABLE

`.omx/research/ddm_cd1_working_tree_debt_landing_20260903.md` — the per-path table (owner memo, bundle
sha or none, tests, LAND/REVERT/HOLD, commit sha), the test-drift ledger, RECALL EVIDENCE,
NEXT_IF_RESUMED, LIVE-HYPOTHESES, DEAD-ENDS. Commit via the serializer. Cite
`docs/operating_manual_craft_handoff.md`. End with the own-vehicle frontier line.
