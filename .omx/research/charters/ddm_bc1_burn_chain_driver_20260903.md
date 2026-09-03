# ddm_bc1_burn_chain_driver — a resumable, reaper-safe CHAIN DRIVER for the QBR1 six-cell burn: authorizes and fires cells 2…6 in the sealed order (each via its exact launcher argv), waits on each cell's terminal receipt, then runs the sealed adjudication — so the burn survives MAIN's session and never waits on a human wake-up

## MANDATE

Operator standing GO ("as long as it takes"). The QBR1 six-cell burn is the sub-0.12 critical path
(`ddm_gs3_gestalt_after_submission_20260903.md` + addendum). Cell 1 is LIVE (MAIN fired it by hand per
`/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/SEALED_MAIN_FIRE_ORDER.json`, arm wc3 9df085160). Cells
2–6 each need: copy `sealed_configs/<cell>.json` → `authorized_configs/<cell>.json` with
`launch_authorized=true`, `scorer_lane={claimed:true, claim_id:ddm_qbr1_scorer_20260903}`,
`metal_lane={claimed:true, claim_id:ddm_qbr1_metal_20260903}`; then the cell's `launcher_argv` with
`AUTHORIZED_CONFIG_PATH` bound; then wait for its terminal receipt; then the next. Five manual fires over
~15 h through a harness that reaps foreground tasks at ~4 min is the "control plane fails silently" genus.
The chain must be durable disk state.

## SCOPE

1. `experiments/ddm_qbr1_cell_chain.py`: reads the fire order JSON; for each cell in `order`: if its
   `RESULT.json` (consumer store `runs/<seed>/<arm>/RESULT.json`) is complete → skip (resumable); else
   verify preconditions (source pins re-hash; AP free ≥ 8 GiB reserve; no other live cell — pidfile check;
   claims present in `.omx/state/active_lane_dispatch_claims.md` and unexpired) → write the authorized
   config (exactly the claim_mutation) → run the cell's `launcher_argv` verbatim with the path bound →
   poll its launch manifest's done receipt / pidfile until terminal → record a chain ledger row (JSONL,
   atomic) → continue. After cell 6: run `adjudication_argv` verbatim; write `CHAIN_DONE.json`.
   Fail-closed: any precondition or receipt failure halts the chain with a typed reason (never skips a cell,
   never re-fires a completed cell, never runs two cells at once).
2. Tests (`tests/test_ddm_qbr1_cell_chain.py`): resumability (completed cells skipped), single-flight
   refusal, precondition refusals, verbatim argv binding, adjudication trigger — all with a fake fire order
   and fake launcher (no real launches).
3. `--dry-run` prints the exact sequence for the live fire order. The arm does NOT launch the chain;
   MAIN launches it via `tools/launch_detached_process.py` (`--done-receipt qbr1_chain_20260903`).

## HARD CONSTRAINTS

- Never launch a cell, the adjudication, a scorer, Metal, or Modal from the arm; never write to
  `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/runs/` or `authorized_configs/` except in the dry-run's
  temp dir. The sealed source tree is READ-ONLY. `upstream/` READ-ONLY.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes; ruff clean.
- Keep it small and reviewable; no new orchestration layer — it is a sequencer over the sealed order.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- memory `harness_monitor_dies_rc144_use_bg_until_loop_20260903` — MAIN's session cannot be the chain.
- `ddm_wc2_qbr1_bug_wallclock_realization_audit_20260902.md` — never alter a cell's config beyond the
  claim_mutation; the seal is the intervention.
- `ddm_qbr1_born_fairform_burn_prep_20260902.md` — the per-cell contract (5,000 updates; sequential; ONE Metal fire).

## OPTIMAL FORM

- Family exemplar: the sealed fire order itself, reference `SEALED_MAIN_FIRE_ORDER.json` (arm wc3, memo commit
  9df085160), the burn entry `experiments/ddm_qbr1_born_fairform_burn_prep.py` (commit 42d322db5), and the launcher
  `tools/launch_detached_process.py` (commit ed3f29000, `--nice-best-effort` available).
- SCOPE reductions: none. MECHANISM reductions FORBIDDEN: no config edits beyond claim_mutation; no parallel cells.
- **PRIOR-LAW PREDICTION (falsifiable):** the dry-run reproduces the six `launcher_argv` arrays byte-for-byte
  with only the config path substituted. FALSIFIER: any argv token differs — count it plainly.

## DELIVERABLE

`.omx/research/ddm_bc1_burn_chain_driver_20260903.md` — the driver contract, test transcript, the dry-run
sequence, RECALL EVIDENCE, NEXT_IF_RESUMED, LIVE-HYPOTHESES, DEAD-ENDS. Commit via the serializer. Cite
`docs/operating_manual_craft_handoff.md`. End with the own-vehicle frontier line.
