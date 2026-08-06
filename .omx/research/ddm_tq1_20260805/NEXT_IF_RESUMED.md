# ddm_tq1 NEXT_IF_RESUMED

Status: QUEUED-WITH-FIRE-ORDER. Phase A built the derived menu and byte-priced smoke candidates; Phase B is gated on a jd5 endpoint receipt with `status=complete`.

1. Re-check the gate without touching the jd5 run dir:
   `.venv/bin/python experiments/ddm_tq1_optimal_token_edit.py --phase-b --require-jd5`
2. If the gate opens, wire/run the v19/v19b realized move-level scorer stack on the queued candidate ledger:
   `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/optimal_form/tq1_phase_a_candidate_prices.jsonl`
3. Accept a move only when its realized receiver -> R -> uint8 -> frozen-scorer components give joint `delta_S < 0` and pose-term erosion <= 0.005.
4. Checkpoint about every 20 accepted/rejected scorer moves under the SSD root before any n600 greedy-to-saturation run.
5. If a full derived menu produces zero accepted realized moves, close the family at optimal-form scope; otherwise stage the accepted endpoint for n600.

No score is claimed by this Phase A receipt.
