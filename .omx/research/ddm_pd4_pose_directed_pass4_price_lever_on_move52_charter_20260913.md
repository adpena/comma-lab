# ddm_pd4 — pose-directed pass 4 on the move-52 field: the PRICE lever (credit per real bit; clustered proposals) (charter, MAIN 2026-09-13; operator "Continue frontier score lowering"; Opus)

## Why
Moves 50–52 (pd1–pd3) came from the same actuator — token edit → re-render → per-pair carrier re-solve → pose credit, selected per pair —
and the family is draining on DEPTH: admitted-per-walked 0.49 → 0.24 → 0.09; K=12 beat K=8 on 24 of 125 shared pairs and lost on none.
What is binding is the PRICE: admitted edits cost 12.9–17.0 bits per changed token (pd1 16.98; pd2 15.4; pd3 12.9), against pass 8's
measured 8.93 bits/token for CLUSTERED edits (adjacent changed tokens share context under the HPAC prior). In every admitted subset the rate
leg is now the largest cost (move 52: +3.1e-5 rate against −6.3e-5 pose). Move 52: S 0.13620226906030858 @ 179,332 B; d_seg 0.00010304;
d_pose 4.21e-6; archive sha ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e; tree
`/Volumes/APDataStore/pact/ddm_pd3/candidate/candidate_runtime`; memo `.omx/research/ddm_pd3_pose_directed_pass3_on_move51_20260913.md`.

## The rung
1. **Price every proposal by its REAL marginal bits** under the shipped HPAC prior on the CURRENT field (the RLC1 known-symbol rail prices a
   single-token change exactly: encode the pair's plane with and without the change — pd1/pd2 measured the ledger 1.9× optimistic, so
   ledger-only ranking is forbidden; a per-pair real encode of the changed plane is ~2 s on the shipped native decoder — measure it in the
   timing smoke), and rank by **resolved-pose credit per real bit**.
2. **Clustered proposals**: for each credited pair from pd1–pd3's rows (their retained `rows` under `/Volumes/APDataStore/pact/ddm_pd{1,2,3}/`)
   and for the tier-1 pairs, propose second-token moves ADJACENT (8-neighbourhood) to an already-changed token or to the pair's best
   single-token move, and price the PAIR of moves jointly by real encode (the clustered price is what pass 8 measured at 8.93 bits/token);
   accept the joint move only if its resolved-pose credit per real bit beats the single move's.
3. K_refine 12 (pd3's), the 12 floor pairs excluded, the 26 move-52 pairs re-searched from their new renders (pose credits re-open on
   re-render — measured), pair budget from a timing smoke (~4 h on 8 shards; APDataStore is at ~29 GiB free — report before every heavy step
   and keep retention ≤ 6 GiB; Vertigo is under its reserve — never lower it).
4. Three-leg Lagrange admission on the RESOLVED pose (frame-0 repair inside; seg on the shipped-mode decode; real-encode rate, twins);
   composition by re-verification with the realized fraction; pre-registered band −2e-5 … −6e-5 S; falsifier: admitted set > −2e-5, or the
   per-bit ranking does not lower the admitted subset's bits/token below pd3's 12.9 (then the price lever is closed at formulation scope).
5. If it nets: byte-close on the move-52 base (prior/semantic member/receiver code byte-identical), twins, cold n600 public parse-back
   (decoded scorer numbers = in-loop), manifest via the Catalog #420 producer, census, smokes candidate + frontier (four native-library exports
   as inflate.sh sets them), retention with shas, NORMAL seal inheriting move 52's own minted leg
   (`/Volumes/APDataStore/pact/ddm_pd3/SEAL_ddm_pd3_pose_directed_pass3_contest_cuda.json.decode_wall_clock.json`, 1,067.8 s; committed copy
   `.omx/research/ddm_pd3_packet_inputs_20260913/MOVE52_T4_DIRECT_LEG.decode_wall_clock.json`; pr18 behaviour digest 9f6e7168… must match).
   NO Modal, NO fire, NO packet (MAIN fires).
6. Memo `.omx/research/ddm_pd4_pose_directed_pass4_price_lever_on_move52_20260913.md` (bits/token table single vs clustered; the per-bit
   ranking's effect on the admitted set; three-leg table; margins); serializer commits (two visible review passes per .py; `[no-triality]
   [p0-ledger-ok]`; NEVER a co-author trailer or AI attribution); lane `ddm_pd4_pose_directed_pass4_price_lever_20260913` (claim it);
   checkpoint `ddm_pd4`. Heavy steps via `tools/launch_detached_process.py --output-dir /Volumes/APDataStore/pact/ddm_pd4/<stage> --nice 0
   --done-receipt …`; waits as background receipt-only until-loops; `sys.dont_write_bytecode` against read-only trees.

## Boundaries
No Modal, no `authorize_*`, no `fire_modal_auth_eval.py`; never edit `upstream/`, the PR tree, sealed trees, contract code, receiver code,
the renderer (weights or per-pair frame_embed), the basis, or the prior; never lower a reserve; pd1/pd2/pd3/sj1/pp1 directories read-only;
no ScheduleWakeup. Label MEASURED / DERIVED / INFERRED / ASSUMED; state the solver behind every pose number.

## OPTIMAL FORM
Reference forms: pd3's search/admission/pricer/seal inputs. Declared deltas: the per-bit RANKING and the clustered proposal family (the
rung); pair budget (SCOPE). Provenance pins: HEAD (record); pd3 memo/packet shas (record); move 52 seal + leg; pass-8 memo for the 8.93
bits/token clustered price.

## Prior negatives accounted (operator 2026-08-15)
rp1 §8d (first-order token price is a ranking, and at depth it inverts — price by real encode); pd1/pd2 (ledger 1.9× optimistic); pass 8
(clustered edits 8.93 bits/token — the measured target); pp1 floor pairs excluded; the 34.8 B lottery; pr19 (inherit move 52's leg once).

Final message: bits/token single vs clustered, the per-bit ranking's effect, the histogram, the three-leg table with exact bytes + sha,
projected S and margins, the seal path with file sha or the typed blocker, retained bytes and free space, every boundary, ending with
`composition S 0.13620226906030858 @ 179,332 B [contest-CUDA T4 n600] (move 52)` — a new number only if MAIN's fire moved it.

<!-- # FORMALIZATION_PENDING: charter, not a finding; the row, if any, carries its equations leg through tools/pointer_move_packet.py --equations-leg at harvest -->
