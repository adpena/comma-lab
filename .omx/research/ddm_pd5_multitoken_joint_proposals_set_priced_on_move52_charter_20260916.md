# ddm_pd5 — pose-directed MULTI-TOKEN joint proposals, priced as a SET by real encode (charter, MAIN 2026-09-16; operator "Continue with all, no signal loss"; Opus)

## Why (the one open formulation on this object; gs3 Addendum 65)
pd4 measured the price lever: clustered 2-token proposals cost 11.4 bits/token median vs 15.8 for singles (−26 %), the per-bit ranking
added five payable pairs, and the admitted 18-pair subset still landed at 12.0 bits/token — 1.1 bits (9.2 %) above the 10.9 the same pool
needs to clear the bar. pd4 also measured WHY its ledger under-charged by 36 %: selecting the argmin of a few noisy real prices is biased
low; re-pricing the selected SET as one object is the cure. Two mechanisms remain unpriced: (a) 3–4-token joint proposals along a pair's
pose-saliency ridge (the prior's context model rewards runs of changed tokens more than pairs — pass 8's clustered field priced at 8.93
bits/token), and (b) admission on a SET price (every candidate set re-encoded as one field, twins) instead of per-proposal argmins.
Move 52: S 0.13620226906030858 @ 179,332 B; d_seg 0.00010304; d_pose 4.21e-6; archive sha
ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e; tree `/Volumes/APDataStore/pact/ddm_pd3/candidate/candidate_runtime`;
leg `/Volumes/APDataStore/pact/ddm_pd3/SEAL_ddm_pd3_pose_directed_pass3_contest_cuda.json.decode_wall_clock.json` (1,067.8 s; unconsumed).
pd4's memo `.omx/research/ddm_pd4_pose_directed_pass4_price_lever_on_move52_20260913.md` and its retained sheets/rows under
`/Volumes/APDataStore/pact/ddm_pd4/` are the starting material (read-only): reuse its 156-pair pool, its per-pair single/2-token rows,
its known-symbol sheet pricer (13.7 s/proposal) and its selection-bias finding.

## The rung
1. Base = move 52 by parse-back (read-only, copy); pose base under the measured-tolerance gate; RLC1 pricer proved by byte-identical repack
   of move 52's tail (119,097 B) and carrier (pd4's control receipts may be reused if their shas bind to the same tree — re-verify).
2. **Joint proposals**: for each of pd4's 156 pool pairs (floor pairs excluded), build 3- and 4-token runs along the pose-saliency ridge
   (contiguous or 8-adjacent cells, deltas ±1 per cell, masked to argmax-interior), seeded from pd4's best single/2-token move; realize each
   run (re-render; per-pair carrier re-solve with frame-0 repair inside; seg on the frozen argmax); K_refine 12 per pair (pd3/pd4's);
   pair budget from a timing smoke (~4 h on 8 shards; APDataStore is at ~22 GiB — report before every heavy step; retention ≤ 5 GiB; Vertigo
   untouched under its reserve).
3. **Set pricing**: rank per pair by resolved-pose credit per real bit using SHEET encodes (pd4's method), then — the cure — re-encode each
   candidate admitted SET as one field (twins) and run the Lagrange admission on the SET's real bits, iterating the sweep until the selected
   set's real price and its ledger agree within the measured noise (report the iteration count and the residual bias).
4. Three-leg admission on the RESOLVED pose (seg on the shipped-mode decode; real-encode rate on the set; composition by re-verification
   with the realized fraction). Pre-registered band: −2e-5 … −5e-5 S; falsifiers: admitted set's real bits/token ≥ 11.0, or net > −2e-5,
   or the set re-price differs from the ledger by > 10 % after iteration. Any fired falsifier closes the multi-token formulation on this object.
5. If it nets: byte-close on the move-52 base (prior/semantic member/receiver code byte-identical), twins, cold n600 public parse-back
   (decoded scorer numbers = in-loop), manifest via the Catalog #420 producer, census, smokes candidate + frontier (four native-library
   exports as inflate.sh sets them), retention with shas, NORMAL seal inheriting move 52's leg (pr18 behaviour digest 9f6e7168… must match);
   NO Modal, NO fire, NO packet (MAIN fires).
6. Memo `.omx/research/ddm_pd5_multitoken_joint_proposals_set_priced_on_move52_20260916.md` (bits/token by run length; the set-vs-ledger
   residual; three-leg table; margins); serializer commits (two visible review passes per .py; `[no-triality] [p0-ledger-ok]`; NEVER a
   co-author trailer or AI attribution); lane `ddm_pd5_multitoken_joint_proposals_set_priced_20260916` (claim it); checkpoint `ddm_pd5`.
   Heavy steps via `tools/launch_detached_process.py --output-dir /Volumes/APDataStore/pact/ddm_pd5/<stage> --nice 0 --done-receipt …`;
   waits as background receipt-only until-loops; `sys.dont_write_bytecode` against read-only trees.

## Boundaries
No Modal, no `authorize_*`, no `fire_modal_auth_eval.py`; never edit `upstream/`, the PR tree, sealed trees, contract code, receiver code,
the renderer (weights or per-pair frame_embed), the basis, or the prior; never lower a reserve; pd1–pd4/sj1/pp1 directories read-only; no
ScheduleWakeup. Label MEASURED / DERIVED / INFERRED / ASSUMED; state the solver behind every pose number.

## OPTIMAL FORM
Reference forms: pd4's pool, pricer and admission; pass 8's clustered price as the target. Declared deltas: run length 3–4 (the rung) and
set-pricing (the cure); pair budget (SCOPE). Provenance pins: HEAD (record); pd4 memo sha (record); move 52 seal + leg.

## Prior negatives accounted (operator 2026-08-15)
pd4 (selection on a real price still ranks — hence set pricing; 12.0 vs 10.9 bits/token); pass 8 (8.93 bits/token is a CLUSTERED-FIELD
figure — a target, not a promise for sparse runs); the family's decay (0.49 → 0.24 → 0.09 → 0.12 admitted/walked — expect few payable pairs;
the bet is price, not count); rp1 §8d; the 34.8 B lottery; pr19 (inherit move 52's leg once).

Final message: bits/token by run length, the set-vs-ledger residual and iteration count, the histogram, the three-leg table with exact
bytes + sha, projected S and margins, the seal path with file sha or the typed blocker, retained bytes and free space, every boundary, ending
with `composition S 0.13620226906030858 @ 179,332 B [contest-CUDA T4 n600] (move 52)` — a new number only if MAIN's fire moved it.

<!-- # FORMALIZATION_PENDING: charter, not a finding; the row, if any, carries its equations leg through tools/pointer_move_packet.py --equations-leg at harvest -->
