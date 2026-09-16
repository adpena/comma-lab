# ddm_ffi6 — amend the frozen pre-fire contract's consumer pins for the 2026-09-11 poller fix (da66353b7), so first-fire intents are producible again (charter, MAIN 2026-09-16; Opus)

## Why (mrs5 memo `.omx/research/ddm_mrs5_minimal_packet_geometry_native_and_prefire_intent_20260916.md`, refusal
`.omx/research/ddm_mrs5_20260916/prefire_attempt/PREFIRE_REFUSAL_43ca2b00aebd4c4296d8356283327f0c.json`)
The intent producer (`tools/make_candidate_seal.py --first-fire-intent`) refuses EVERY candidate since 2026-09-11 with
gate 0: `PREFIRE_CONTRACT_DRIFT_REFUSED: live file differs from committed blob: tools/modal_harvest_poller.py`. The newest
amendment row (`a9fd1270…`) pins 15 consumers; 14 are live-clean; da66353b7 (MAIN's poller fix binding the auth-eval
runtime-tree sha in terminal claim rows — a reviewed, tested, intended change, commit da66353b7, test in
`src/tac/tests/test_pointer_move_packet.py`) moved that one. The contract's own cure is a consumer-fix amendment row —
exactly what ffi5 did for the 2026-09-10 consumer fixes (`PREFIRE_IMPLEMENTATION_MANIFEST_AMENDMENT3.json`, commit
0919b62e4, "amendments 4 rows"; read the ffi5 memo and rlc5's custody under `.omx/research/ddm_rlc5_20260910/` for the
exact amendment procedure and the freeze receipt `.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json`).
Second finding (record, do NOT fix here): the frozen intent object has no shape for a receiver-only candidate at the
pointer's own bytes (net dS = 0 against a strictly-negative bar; a timing ratio < 1 is clamped) — a contract GAP, to be
adjudicated by a pr-family memo, not patched in this arm.

## Deliverable
1. Produce the next amendment row exactly per the contract's amendment procedure (whatever ffi5 used: a new
   `PREFIRE_IMPLEMENTATION_MANIFEST_AMENDMENT<N>.json` naming the moved consumer with its committed blob sha at
   da66353b7's landing and the rationale "consumer fix: poller binds auth-eval runtime-tree sha in terminal claims;
   reviewed + tested"), re-run the contract's own validator so gate 0 passes on the live tree, and prove it with a
   real producer run on rlc5's committed intent inputs or mrs5's real inputs: gate 0 must no longer fire (gates 1–2 may
   still refuse mrs5 — that is the recorded gap, report it verbatim). Do not touch `src/tac/candidate_seal.py`,
   `src/tac/decode_wall_clock.py`, or `tools/modal_harvest_poller.py`; if the procedure requires code, STOP and report.
2. A memo `.omx/research/ddm_ffi6_prefire_contract_consumer_pin_amendment_20260916.md`: the drift, the amendment row,
   the validator receipt before/after, the gate-1/2 gap stated as a pr-family question with the exact sentences.
3. Serializer commits (`[no-triality] [p0-ledger-ok]`; two review passes if any .py changes — there should be none);
   no co-author trailer, no AI attribution. Checkpoint `ddm_ffi6`; lane `ddm_ffi6_prefire_contract_consumer_pin_amendment_20260916`.

## Boundaries
No Modal, fire, packet, PR, push, authorize_*; never edit upstream/, sealed trees, the PR trees, contract code; do not
touch `/Volumes/APDataStore/pact/ddm_pd7/` or `ddm_mrs5_fire/`. Label MEASURED / DERIVED / INFERRED / ASSUMED.

## OPTIMAL FORM
Reference form: ffi5's amendment landing (0919b62e4) and the frozen contract (a47543199 + PREFIRE_CONTRACT_FROZEN.json).
Declared deltas: one amendment row for one moved consumer; nothing else. Provenance pins: da66353b7; mrs5 refusal file
sha 8df50f1c0ee52069 (prefix); ffi5 landing 0919b62e4.

## Prior negatives accounted (operator 2026-08-15)
rlc3/rlc4/rlc5 (the contract's refusals are real controls; a drift refusal is cured by an amendment, never by editing
the contract); ffi1/ffi2 (clause questions → STOP with the exact sentence, no patching); hpr1 (move-48 intent blocked by
the same class of drift after pr19's commit — the precedent for this cure).

<!-- # FORMALIZATION_PENDING: charter, not a finding -->
