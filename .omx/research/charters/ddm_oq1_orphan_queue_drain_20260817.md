# ddm_oq1 — THE ORPHAN-QUEUE DRAIN (the never-spawned owner, finally spawned)

Operator 2026-08-17 verbatim: "We have a ton of orphaned signal." Standing laws that bind this arm:
YOU-OWN-EVERYTHING (every row exits OWNED) · follow-ons-fire-at-harvest · audits-launder (m49 —
this arm EXECUTES, it does not re-audit) · staleness-at-consumption (m37) · vehicle-scope (#917:
the lever instruments all point at a retired vehicle) · ALWAYS KEEP THE PAYLOAD (DEF CON 1000) ·
never-invent-flags · serializer commits with POST-EDIT working-tree sha · .py needs 2 review
passes, REVIEW_GATE_OVERRIDE ok on .md/.json only · NO Modal/paid dispatch (MAIN owns all fires) ·
NO scorer n600 runs (queue those to MAIN with a sealed fire-order) · upstream/ READ-ONLY ·
payloads → /Volumes/APDataStore/pact/ddm_oq1/ (Vertigo is READ-ONLY).

## The measured orphan population (recalled from stores, 2026-08-17)

1. `.omx/research/ddm_qj1_followon_backlog_join_20260804.json` — **437 rows
   QUEUED-WITH-FIRE-ORDER, untouched since 08-04. 233 rows are owned by
   `codex-qj1-followon-drain`, an arm that was never spawned — YOU are that arm.**
   Tiers: 1×T0-read · 7×T1-$0-minutes · 6×T2-$0-long · 13×T3-build · 2×T4-money ·
   16×p2a-commit-sweep · 392×untiered.
2. Unowned pending task rows from the 08-16/17 audit waves (MAIN will apply ledger updates at
   harvest from your table; you CANNOT touch the harness task ledger): #833 #834 #835 #840 #844
   #848 #849 #857 #860 #862 #875 #882 #894 #896 #901 #905 #906 #911 #912 #914 #915 #916 #917
   #918 #919 #920 #924 #926 #933 #939 #940 #949 #954 #971 #974 #977 #979 #984 #985 #986 #990
   #994 #1086 #1087 #1088 #1089 #1090 #1091 #1092 #1093 #1094 #1095. Read each row's CONTENT
   from `.omx/state/canonical_task_status.jsonl` sources or the memos they cite — never bare ids.
3. `.omx/tmp/codex_runs/` holds 1,128 `.done` receipts with ZERO harvest markers. The reader
   surface is the np1 store: `.omx/state/codex_arm_queue.final_messages.jsonl` +
   `codex_arm_queue.next_if_resumed.jsonl` — consume THOSE, never the 1,128 raw files.
4. Costate `DDM-vehicle-harvest: 10/1524 routed artifacts` — sample the largest gap families
   (v:121, ms:49, j:45) only where a row in (1)/(2) points into them.

## THE VEHICLE-SCOPE FILTER — apply FIRST, to every row (AMENDED at spawn, operator 08-17)

**Operator steer 2026-08-17, verbatim, received during charter authoring: "TR1 class needs a
lot of work and optimization and composition pre and during and post training including seeding
and solving and conditioning and all."** Consequence: there are TWO live lines, not one.

- **LIVE-FRONTIER**: the PR130-lineage HPAC vehicle (cp135 → hv1 ep0634, S 0.15959729295498598
  @ 182,759 B [contest-CUDA T4 n600], sha 80d9c8c6…).
- **LIVE-ACHIEVER**: the **TR1 class** (trained partition→pixel renderer + its
  seeding/solving/conditioning/composition machinery) — NOT retired, per the operator steer.
  TR1-class rows are SURVIVORS: tag them `oq1_vehicle_scope=live_achiever_tr1` and route them
  as inputs to the sister arm **ddm_tc1** (TR1 lifecycle composition program), phase-tagged
  {PRE-seeding | DURING-conditioning/objective | POST-solving | COMPOSITION}.
- **RETIRED**: witness v9/v10/inverse-solve/levelset lineage vehicles → FOLD(vehicle-retired)
  with one line, UNLESS the row is a mechanism/law that transfers to either live line — then
  re-scope it explicitly to the named live consumer.
Do not spend execution time on retired-vehicle rows beyond the one-line fold.

## Execution mandate (this is a DRAIN, not an audit)

Per row, in rank order within tier (T0 → T1 → p2a-sweep → untiered triage → T2 → T3/T4):
- **T0/T1/p2a rows: EXECUTE NOW yourself** (reads, greps, cached-data checks, closing-artifact
  location). The row's own fire_order text tells you the check.
- **Untiered 392: triage in bulk** — vehicle-scope filter + a one-line tier assignment; execute
  the ones that turn out T0/T1; do NOT deep-read every memo (batch by source memo).
- **T2: execute if your remaining budget allows; else OWNER + fire-condition.**
- **T3/T4 + anything needing scorer/Metal/Modal: sealed fire-order row for MAIN** (exact command,
  cost, expected ΔS, falsifier).
- Every row exits exactly one of: FIRED(result) · FOLDED(reason) · QUEUED(owner, fire-condition)
  · DEFERRED(named-measured-blocker). No row may remain ownerless.

## Deliverables (append-only; never edit qj1 in place)

1. `.omx/research/ddm_oq1_drain_dispositions_20260817.json` — versioned successor of qj1's
   dispositions (same schema + `oq1_disposition`, `oq1_evidence`, `oq1_vehicle_scope` fields),
   ALL 437 queued rows + the task-row table + final-message-store rows dispositioned.
2. `.omx/research/ddm_oq1_orphan_queue_drain_20260817.md` — memo: counts
   (fired/folded/queued/deferred), the TOP-10 LIVE-VEHICLE wins surfaced (with byte/ΔS estimates
   and evidence paths), the sealed MAIN fire-orders, and the task-row disposition table for MAIN
   to apply. STORES CONSULTED lines per the recall-depth law.
3. Commits via `tools/subagent_commit_serializer.py` with `--expected-content-sha256` (post-edit
   working-tree sha). End with the arm final message summarizing counts + top wins (it is
   persisted byte-for-byte; NEXT_IF_RESUMED block if you run out of budget).

## Budget + loop

Loop-until-dry on T0/T1/p2a (a round that executes zero new rows = done). Hard cap ~6h. If the
untiered 392 cannot all be triaged, triage by source-memo batches and record the honest
denominator (report N-triaged / 392 — silence-is-not-success; the vacuity law binds).

## Falsifier honesty

Expected: most of the 437 fold as vehicle-retired, and the live-vehicle survivors number in the
tens. If the survivors number near ZERO, say so plainly — "the backlog was already dead mass" is
a valid, valuable verdict and shrinks the standing queue to nothing. Do not manufacture wins.
