# ddm_hv2_harvest_consumption_sweep — drain the FINISHED-unharvested arm backlog: extract every NEXT_IF_RESUMED / follow-on / dead-end block, route each to an owned exit, surface the fire-ready head

## MANDATE

Operator 2026-08-26 "Continue with all." The m113 law (follow-ons fire AT HARVEST, exits
FIRED/FOLDED/QUEUED-with-order) and the m45 law (every row exits OWNED) have a measured
debt: ~429 arm final messages exist under `.omx/research/arm_final_messages/` and the
hot-state has carried "~79 FINISHED-unharvested keeper arms" for days; the
consolidation-debt monitor reads signal_ratio 50.0 (100 research memos vs 2 canonical
routings in 24h) — un-routed findings are the campaign's #1 signal-loss class
([[orphan_sweeps_that_do_not_write_the_store_are_the_disease_20260803]]). This arm is the
sweep's OWNER: compute the true unharvested set, read every unconsumed final message,
extract the typed blocks, and route EVERY row to an owned exit. No score work — pure
signal recovery; the deliverable is the routed queue the NEXT score arms fire from.

## SCOPE

1. COMPUTE THE TRUE SET: diff `.omx/research/arm_final_messages/*` + the keeper's .done
   receipts (`.omx/tmp/codex_runs/*.done` and the keeper ledger) against consumption
   evidence (ledger rows via tools/canonical_task_status.py, memo citations, harness-task
   references). An arm counts HARVESTED only if its final message's follow-on blocks have
   routed exits — a .done receipt alone is NOT consumption.
2. EXTRACT: for each unconsumed final message, pull NEXT_IF_RESUMED / LIVE-HYPOTHESES /
   DEAD-ENDS / QUEUED-WITH-A-FIRE-ORDER blocks via the np1 extractor lineage (commit
   `499ffd68a1` — the persisted-final-message + extractor surface; reuse it, do not
   rebuild). Where a message predates the np1 format, extract by reading.
3. ROUTE every extracted row to exactly one exit (m113/m45): (a) ALREADY-CONSUMED (cite
   the consuming memo/commit) · (b) STALE-SUPERSEDED (cite the superseding receipt —
   staleness at consumption, m37) · (c) QUEUED-with-order (ledger row w/ named gate +
   fire-condition + owner) · (d) FIRE-NOW head: the ranked top ≤5 rows whose
   fire-conditions are MET, delivered as a table for MAIN (do not fire score work
   yourself). Dead-ends route to the negative-signal registry only if not already there.
4. LEDGER + HYGIENE: write the routed rows (tools/canonical_task_status.py, actor
   ddm_hv2); mark the harvest state durably so the next sweep diffs instead of re-reading
   (a machine-readable harvest ledger keyed by final-message filename + content sha).

## HARD CONSTRAINTS

- READ-heavy arm: no score measurements, no Modal, no heavy compute. `upstream/` READ-ONLY.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- No naive/toy/generic (operator 08-26): the routed table's ALREADY-CONSUMED claims must
  cite the consuming artifact (never asserted from memory — m44); negative-existence
  claims ("never consumed") require the grep evidence line (m53).
- Do not modify any arm's final message (append-only corpus); the harvest ledger is a NEW
  file, never an edit of the sources.

## PRIOR NEGATIVE SIGNAL (bearing laws)

- #878/np1 (commit `499ffd68a1`): final messages ARE persisted byte-for-byte + extractor
  built w/ 2 real readers — REUSE, the build exists.
- #879/#880 (their adjudication rows in the harness ledger): the follow-on detector's
  corpus is MEMOS while the backlog is TASK ROWS — this sweep must join BOTH sides, that
  was the measured gap.
- m49 (audits launder): an audit that only COUNTS unharvested rows re-creates the debt —
  the deliverable is ROUTED EXITS, not a census.
- m36 (deferral scatter): queued rows go to ONE canonical ledger, never per-memo scatter.

## OPTIMAL FORM

- Family REFERENCE exemplars w/ provenance pins (receipt-backed): np1's persisted-message
  + extractor surface, commit `499ffd68a1` · the oq1 orphan-queue drain pattern (harness
  #1101's landed memo in `.omx/research/`) · the canonical ledger discipline
  (tools/canonical_task_status.py lineage, gb1-era rows commit `884bb65f1e`).
- SCOPE reductions declared: if the true unconsumed set exceeds ~120 messages, rank by
  recency×axis-relevance and declare the processed/deferred split explicitly (the
  remainder gets a QUEUED-sweep row with this same charter as its fire-order — no silent
  truncation, m50: report the denominator).
- **PRIOR-LAW PREDICTION (falsifiable):** the sweep finds ≥10 unconsumed follow-on rows
  with MET fire-conditions (the m113 debt made visible), including ≥1 on the token/rate
  axis relevant to the live d3b line. FALSIFIER: the backlog is already fully consumed —
  then the "79 unharvested" hot-state figure was stale (m106 headline debt), correct it at
  the source with the measured number.

## DELIVERABLE

`.omx/research/ddm_hv2_harvest_consumption_sweep_20260826.md` — the true-set computation
(denominators explicit) + the routed-exit table + the FIRE-NOW head (≤5, fire-conditions
quoted) + the machine-readable harvest ledger path + ledger receipts + GESTALT-DELTA line.
Commit via the serializer. End with the own-vehicle frontier line.
