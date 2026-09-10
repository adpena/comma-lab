# ddm_pr15 — second-family ratification of MAIN's fire-tool fix, freeze amendment[1], and the three findings the first REAL first-measurement dispatch surfaced (charter, MAIN 2026-09-10)

You are the pr-family reviewer (pr8→pr14 lineage; second family; read-only). The amended contract met its first REAL
producer PASS (rlc5: candidate 180,406 B sha 04758c0d…, intent rc 0, both refusal controls exact; memo
`.omx/research/ddm_rlc5_rebase_counted_rider_cure_onto_move43_amended_prefire_intent_20260910.md`, sha 59f8fed003e03c05…) and
MAIN then ran the committed-consumer half. Three tool defects surfaced, all in the pass path, none in a rule:
1. **Dispatch-path check ordering (FIXED by MAIN, commit 808da43c; consumer change → freeze amendment[1] at ff75f2f9f, receipt
   sha 57f613af8c82c7d7…):** `tools/fire_modal_auth_eval.py` first-measurement mode ran `verify_dispatch_paths` over an argv that names
   `FIRST_MEASUREMENT_CONTEXT.json`, which the same function writes only AFTER the check (`_pf_write_new` refuses a
   pre-existing file), so no real intent could reach dispatch. Fix: exclude only that self-written path from the check.
   RATIFY / AMEND / REFUSE the fix and the amendment row (same definition, new implementation pins; adjudication memo pr14).
2. **Replay guard vs the tool's own refusal receipts (WORKED AROUND):** `FIRST_MEASUREMENT_REPLAY_REFUSED: output contains an
   existing or ambiguous dispatch` fired on an output dir holding only the tool's own `PREFIRE_REFUSAL_*.json` receipts and
   an empty launch dir. MAIN issued authorization v3 on a clean dir. Rule on whether the guard should ignore its own
   refusal receipts (exact patch text if AMEND) — never loosen: a dir with any dispatch/claim/result artifact must still refuse.
3. **Authorize tool import path (WORKED AROUND with PYTHONPATH):** `tools/authorize_candidate_first_measurement.py` adds only
   `src`; `tac.decode_wall_clock.measure_t4_runtime_digest` imports `experiments.contest_auth_eval` → ModuleNotFoundError
   from a bare invocation. Rule on the fix (add the repo root like `tools/quiesced_decode_timing.py`) and whether it is a
   pinned consumer change (amendment[2]).
Also record: the superseded, unconsumed authorizations v1/v2 (custody kept; state AUTHORIZED_ONCE, never reserved) and
whether the contract needs an explicit SUPERSEDED marker for them.

## Method
Read pr12 (sha 50d00e3956dc7ae5…), pr13, pr14, the freeze (both amendment rows), `src/tac/candidate_seal.py`,
`tools/fire_modal_auth_eval.py` (diff 808da43c^..808da43c), `tools/authorize_candidate_first_measurement.py`, rlc5's
research dir (intent v1 88204b88… / v2 8370bcf6…; refusal receipts under the run dirs). Run the suites on the host. Read-only
on code and every `/Volumes/...` path; no Modal, no fires. Verdict per item with `clause → code → test`; AMEND text
literal. Memo `.omx/research/ddm_pr15_ratify_fire_tool_fix_amendment1_and_first_measurement_findings_20260910.md`;
serializer commit LAST (`REVIEW_GATE_OVERRIDE=1` ok for .md); rc 17 is NOT a stop. Checkpoint as `ddm_pr15`.

## OPTIMAL FORM
- Reference form: pr12's contract + pr13/pr14 rulings as normative text; the landed code at 808da43c as the object; the
  pr13 clause-table form. No delta.
- Provenance pins (sha256 prefixes): freeze receipt 57f613af8c82c7d7…; rlc5 memo 59f8fed003e03c05…; pr14 memo 6e1732baea1912731fef6a53abca0cb94faf2e9440accc304dd9b34e593bade9;
  pr13 memo 3156992449ea4f71…; fire-tool fix commit 808da43c02a3; freeze append ff75f2f9f; intent v2 sha 8370bcf6401a40b9…;
  pointer move 43 commit 48109233e / archive 7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e.

## Prior negatives accounted (operator 2026-08-15)
- pr10 (rule tuned after data): MAIN changed a CONSUMER (not a rule) mid-cycle to reach the pass path; you decide whether
  that ordering (fix → amendment → re-emit → authorize → fire) preserved the contract's guarantees; nothing was loosened.
- dwc1 (gate with no door): finding 1 is that genus inside the fire tool; finding 2 is the same genus in the replay guard.
- r9m: the manifest re-pin is content-only.

Final message: three verdicts (one line each), any AMEND text, the superseded-authorization ruling, serializer rc, and
the frontier line — quote the pointer at the time you finish (move 43 or 44) from `.omx/state/canonical_frontier_pointer.json`.

<!-- # FORMALIZATION_PENDING: review charter; no measured row -->
