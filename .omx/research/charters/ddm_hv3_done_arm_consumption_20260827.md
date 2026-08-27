# ddm_hv3_done_arm_consumption — drain the 18 FINISHED-unharvested .done arms into owned exits (hv2 pattern round 3; owning memo: ddm_hv2_harvest_consumption_sweep_20260826.md)

## MANDATE

Operator 20260827: *"Frontier score lowering priority"* + standing 08-16 law *"KEEP THE QUEUE FED —
a WAIT ⇒ fire parallel work SAME TURN"* (memory m158). The keeper shows 18 FINISHED arms with
unread/undispositioned .done state while the Metal slot burns r8 (~3-6h wait). MAIN already
harvested the 5 decision-bearing heads (fb2 route table · d3b · bs3 · bs4x · sr3) and adjudicated
route 2 DEAD (`.omx/research/ddm_bs3_route2_dead_adjudication_20260827.md`, 32488cafc6). This arm
consumes the REMAINDER so no follow-on sits orphaned when the r8 endpoint fires: read every final
message + NEXT_IF_RESUMED row, disposition every follow-on to an owned exit, and mark each arm
harvested in the keeper.

## SCOPE

1. Enumerate the FINISHED-unharvested set from `tools/codex_arm_queue.py status` (18 at charter
   time: bs4 · bs4x · d3b · fb2 · pf4x · w96b · jf2 · pc2 · rb1 · sr3 · w96a · bs3 + ~6 more not
   shown in the top-12) — authority: the keeper's tracked-charter registry + `.done` markers +
   `.omx/research/arm_final_messages/` byte-exact captures (the persisted-final-message surface,
   `.omx/research/ddm_np1_20260805/NP1_RECEIPT.md`) +
   `.omx/state/codex_arm_queue.next_if_resumed.jsonl`.
2. Per arm, emit ONE typed disposition row: {arm, verdict_summary (≤2 lines, content not bare ids
   per m89), follow_ons: [{item, exit ∈ CONSUMED-ALREADY / ROUTED-to-<task-or-ledger-row> /
   CLOSED-dead-with-reason / QUEUED-W-FIRE-ORDER-<trigger>}], payload_custody_check (retained
   receipts exist per ALWAYS-KEEP-THE-PAYLOAD)}. The m113 law binds: every follow-on exits
   FIRED/FOLDED/QUEUED-W-ORDER — no UNKNOWN residue.
3. Cross-check the 5 MAIN-harvested arms' dispositions against MAIN's routing (fb2 → route table
   in hot-state · d3b → REFUSED_AT_FORMULATION_SCOPE closed · bs3 → route-2-dead memo · bs4x →
   storage-blocked + DE-PRIORITIZED per the route-2 adjudication · sr3 → reclaim green consumed
   by the r8 fire) — flag any follow-on MAIN missed, do not re-adjudicate settled verdicts.
4. Mark each dispositioned arm harvested via the keeper's `mark` subcommand so the SATURATION
   view stops listing consumed arms as spawnable; append rows to
   `.omx/research/ddm_hv2_harvest_consumption_ledger_20260826.jsonl` (extend the existing ledger,
   do NOT fork a parallel one — the anti-duplicate-SoT rule: one canonical ledger per surface).
5. Rank any surviving FIRE-NOW heads (frontier-relevant, unowned) at the memo top for MAIN —
   named trigger + cost + falsifier each; expect few (the decision-bearing heads were taken).

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; bulky receipts to `/Volumes/APDataStore/pact/ddm_hv3_done_arm_consumption/`.
- NO new measurements, NO scorer runs, NO spawns — this is a CONSUMPTION/ROUTING arm only. Do not
  touch the Metal slot, the r8 launcher dir, or `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/`
  (live burn custody). Do not close/edit MAIN-owned task rows — propose routing in the memo.
- Storage claims must cite LIVE `df` numbers at write time, never charter-time figures (the m37
  staleness law; a stale capacity note has killed arms before).

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- ddm_hv2_harvest_consumption_sweep_20260826.md: the 84-unknown-follow-on backlog arose exactly
  because harvest lagged spawn — the detector's corpus is MEMOS while the backlog is TASK ROWS;
  route by CONTENT.
- `.omx/research/ddm_np1_20260805/NP1_RECEIPT.md`: final messages are now persisted
  byte-for-byte — read the capture, never reconstruct an arm's verdict from memory (the
  not-re-derivable-receipt genus: na2's pose ratios were quoted all session while the receipt
  was absent from disk).
- ddm_bs3_route2_dead_adjudication_20260827.md: born-small B+C is DEAD on distortion — do not
  re-open bs3/bs4x follow-ons as live routes; their exits are CLOSED-dead or LESSON-ONLY.
- The 08-23 stale-capacity-note incident (keeper retry-loop lesson): a stale note killed three
  arms and four kill attempts missed the keeper's retry loop — when marking arms, use the
  keeper's own subcommands, never hand-edit its state files.

## OPTIMAL FORM

- Family exemplar: `.omx/research/ddm_hv2_harvest_consumption_sweep_20260826.md` @ commit
  88ecc0a9b1 is the reference form (82-row true-set sweep, typed exits, 5 fire-now heads ranked,
  ledger receipt `.omx/research/ddm_hv2_harvest_consumption_ledger_20260826.jsonl`).
- SCOPE reductions declared per row: this round covers ONLY the 18 currently-FINISHED arms (legal
  n reduction vs hv2's 82 — the older population is already dispositioned). MECHANISM reductions
  FORBIDDEN: every arm gets the full read-capture → typed-exit → keeper-mark chain; no
  headline-only skims (the m106 stale-headline law — read BODIES).
- **PRIOR-LAW PREDICTION (falsifiable):** the hv2 law predicts ≥1 decision-bearing follow-on
  still sits unowned in the 13 unread arms (hv2 found 5 fire-now heads in 82; base rate ~6%
  ⇒ expect ~1 in 13). FALSIFIER: all 13 arms' follow-ons already owned/consumed ⇒ count it
  plainly — the harvest-before-spawn discipline has caught up and the next hv round can cheapen.

## DELIVERABLE

`.omx/research/ddm_hv3_done_arm_consumption_20260827.md` — the per-arm disposition table (typed
rows per SCOPE 2) + FIRE-NOW ranks for MAIN + ledger append receipt + keeper mark receipts. Commit via the
serializer. End with the own-vehicle frontier line.
