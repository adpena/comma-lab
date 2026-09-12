# ddm_pd2 — pose-directed token pre-distortion, pass 2 on the move-50 field, FIRST-MEASUREMENT chain (charter, MAIN 2026-09-13; operator "Continue frontier score lowering"; Opus)

## Why
pd1 landed move 50 (S 0.13628342713679067 @ 179,195 B; d_seg 0.00010294; d_pose 4.45e-6; archive sha
1ea274f612a26183f31f6d439505d0289bd3751b4da9467343fc156203503cd7; tree `/Volumes/APDataStore/pact/ddm_pd1/candidate/candidate_runtime`; packet
`.omx/research/ddm_pd1_pose_directed_token_predistortion_all_pairs_20260913_pointer_move_50_20260912.md`; memo
`.omx/research/ddm_pd1_pose_directed_token_predistortion_all_pairs_20260913.md`). Its search refined only 3 of ≤ 32 screened proposals per
pair under a 3-hour budget, carried 41 credited pairs (median −21.6 % of the pair's pose; 26 of 41 at ≤ 0 seg cells) and admitted 20; 580
pairs were searched shallowly and 21 credited pairs were dropped by the sweep. The rate leg measured 16.98 bits/token (1.9× the modelled
ledger) — the binding cost. Pass 2 on the move-50 field: (a) more refines per pair (K_refine ≥ 8, chosen from pd1's measured 130 s per
pair-with-3-refines and ~3–4 h of 8 CPU shards — declare it), (b) rank proposals by resolved-pose credit PER BIT using pd1's measured
per-token prices (the real encode of each pair's edit, or the RLC1 ledger calibrated by pd1's 1.9× factor and re-verified by real encode),
(c) the 20 pairs already edited at move 50 are re-searched from their NEW renders (a repair consumes local slack — pass 7's law — so expect
less there), the 12 floor pairs excluded. Pre-registered band: −2e-5 … −8e-5 S net; falsifier: the admitted set projects net > −2e-5 on the
RESOLVED pose with real-encode rate, or the composition realizes < 0.8 of the sum.

## The contract route (binding): FIRST-MEASUREMENT chain
Move 50 inherited move 49's leg (spent) and its own T4 decode measured 1,375.8 s, above the 1,260 s t4_direct limit, so no leg can be minted
for move 50. Your candidate (receiver code byte-identical; archive + two pins change) takes the FIRST-MEASUREMENT route exactly as moves
46/48 did: `tools/make_candidate_seal.py --first-fire-intent … --timing-risk-evidence …` with risk mode `measured_t4_identity_class_envelope`
over the identity class's declared T4 legs (moves 46: 1,140.8 s; 48: 1,023.3 s; 49: 1,106.2 s; 50: 1,375.8 s — the envelope max is 1,375.8 s;
local ratio kept as ceiling × (1 + fraction) ≤ 1,800 s; fraction 0 REFUSED), the frozen contract
`.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json` (drift refuses emission), pr18's behaviour digest v2 (9f6e7168… must match);
record the normal validator's typed refusal; COMMIT the intent JSON; stage every seal input. MAIN alone claims, authorizes
(`tools/authorize_candidate_first_measurement.py`), fires, harvests, completes, packets move 51. Worked example:
`.omx/research/ddm_hpr1_20260911/v2/` and `.omx/research/ddm_pr19_20260911/TIMING_RISK_FIELD_SPEC_FOR_HPR1.json`; hpr1's
`experiments/ddm_hpr1_seal_inputs.py` modes. Read `src/tac/candidate_seal.py` (intent/authorization/completion paths) — never edit it.

## Deliverable
Base = move 50 by parse-back (read-only, copy); per-pair pose base under pd1's measured-tolerance gate; the RLC1 pricer proved by
byte-identical repack of move 50's tail (118,978 B; carrier repacked from the pointer). Search as above; three-leg Lagrange admission on the
RESOLVED pose (frame-0 repair inside; seg on the shipped-mode decode; real-encode rate, twins); composition by re-verification with the
realized fraction reported; if it nets: byte-close on the move-50 base, twins, cold n600 public parse-back (decoded scorer numbers =
in-loop), manifest via the Catalog #420 producer, census, smokes candidate + frontier (four native-library exports as inflate.sh sets them),
retention ≤ 8 GiB with shas on APDataStore (Vertigo is under its reserve — never lower it), timing-risk receipt + COMMITTED intent + typed
refusal + seal inputs staged. NO authorization, NO fire, NO completion, NO packet (MAIN's). Memo
`.omx/research/ddm_pd2_pose_directed_pass2_on_move50_20260913.md`; serializer commits (two visible review passes per .py; `[no-triality]
[p0-ledger-ok]`; NEVER a co-author trailer or AI attribution); lane `ddm_pd2_pose_directed_pass2_first_measurement_20260913` (claim it);
checkpoint `ddm_pd2`. Heavy steps via `tools/launch_detached_process.py --output-dir /Volumes/APDataStore/pact/ddm_pd2/<stage> --nice 0
--done-receipt …`; waits as background receipt-only until-loops; `sys.dont_write_bytecode` against read-only trees.

## Boundaries
No Modal, no `authorize_*`, no `fire_modal_auth_eval.py`; never edit `upstream/`, the PR tree, sealed trees, contract code, receiver code,
the renderer (weights or per-pair frame_embed), the basis, or the prior; never lower a reserve; pd1/sj1/pp1/cb1/cr1/so2 directories
read-only; no ScheduleWakeup. Label MEASURED / DERIVED / INFERRED / ASSUMED; state the solver behind every pose number.

## OPTIMAL FORM
Reference forms: pd1's search/admission/pricer (its producers `experiments/ddm_pd1_*.py` and `ddm_pp1_pose_actuation.py::cmd_search_b`),
hpr1's first-measurement inputs. Declared deltas: K_refine and the per-bit ranking (SCOPE + the rung); the contract route (a CONTRACT
requirement). Provenance pins: HEAD (record); pd1 memo/packet shas (record); move 50 seal + receipt; frozen contract sha (record).

## Prior negatives accounted (operator 2026-08-15)
pd1's rate prior 1.9× optimistic (price by real encode); pass 8's same-field decay (re-searching edited pairs yields less); pp1's floor
pairs (excluded); rp1 §8d (first-order token price is a ranking); pr19 (no transitive inheritance — this charter obeys it); the T4
variance (1,023–1,376 s in one identity class — the envelope, not a single leg, is the timing authority); the 34.8 B lottery.

Final message: K_refine and the timing smoke, the per-pair credit/cost histogram, the three-leg table with exact bytes + sha, projected S
and margins, the intent path with file sha/bytes/digest and its commit, the typed refusal, the timing-risk receipt, seal-inputs path,
retained bytes, every boundary, ending with `composition S 0.13628342713679067 @ 179,195 B [contest-CUDA T4 n600] (move 50)` — a new
number only if MAIN's fire moved it.

<!-- # FORMALIZATION_PENDING: charter, not a finding; the row, if any, carries its equations leg through tools/pointer_move_packet.py --equations-leg at harvest -->
