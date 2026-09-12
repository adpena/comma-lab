# ddm_pd1 — POSE-DIRECTED token pre-distortion over ALL 600 pairs: the re-solve credit as the objective (charter, MAIN 2026-09-13; operator "Continue frontier score lowering"; Opus)

## The measured actuator nobody has run as an objective
- **Move 42** (rp1 round 2, `.omx/research/ddm_rp1_round2_move40_20260910.md` §8d–8e): rate-directed edits whose rate leg LOST (+5 B) landed
  −1.62e-4 S ENTIRELY as a pose credit — d_pose 4.89e-6 → 4.66e-6 — because the per-pair carrier re-solve on the re-rendered pairs found
  better coefficients (stale 216× worse → resolved +1.9 % on the full set; the Lagrange sweep kept the 160 of 576 pairs that credited).
- **Move 49** (sj1 pass 7): seg-directed edits on 130 pairs; the re-solve credited 35 pairs (sum −5.5e-8 d_pose mean units) and cost 96; the
  subset's pose leg was −3.44e-5 S, larger than its seg leg. MAIN's per-pair read of `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/pose*/`:
  credits exist on ~27 % of edited pairs regardless of the edit's objective.
- **pp1** ran actuator B (pose-ranked token edits) ONLY on the 12 floor pairs (52 % of the pose mass, where the resolved pose is a floor)
  and found 0.25 %. The other 588 pairs — 48 % of the pose mass, and the pairs where moves 42/49 found their credits — were never searched
  for pose. `experiments/ddm_pp1_pose_actuation.py::cmd_search_b` + `pose_saliency_on_token_grid` exist; reuse them.
- Arithmetic (DERIVED): the pose leg is 741 S per unit d_pose at 4.55e-6; a credit of 1 % of d_pose is −3.4e-5 S (1.7 bars). Moves 42/49
  found −4.7 % and −0.9 % as side effects. Band (pre-registered): −3e-5 … −2e-4 S net after the token bytes (pass 8 measured 8.9 bits per
  changed token on this field; pass 7 5.0 bits/token in its subset) and seg re-verification. Falsifier: the Lagrange-admitted set projects
  net ΔS > −2e-5 on the RESOLVED pose with real-encode rate; or the composed object realizes < 0.8 of the per-pair sum.

## Deliverable
1. Base = move 49 (tree `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/candidate/candidate_runtime`, read-only, copy; field by parse-back;
   per-pair pose base reproduced under pp1's measured-tolerance gate; the RLC1 pricer proved by byte-identical repack of the 118,938 B tail).
2. **Search, all 600 pairs, pose as the objective**: for each pair, propose single-token moves at cells where the pose saliency on the token
   grid is largest (pp1's `pose_saliency_on_token_grid`; K proposals per pair — pick K from a timing smoke so the search fits ~3 h on this
   host's CPU shards; declare it), realize each (re-render the pair; `refine_pair` carrier re-solve; frame-0 repair inside), accept per pair
   the best RESOLVED d_pose gain; seg re-verified on the pair's re-render (frozen CPU SegNet argmax, DALI GT lineage; net flipped cells
   charged at 8.48e-7 S/cell, no seg-neutral requirement — the admission prices it); record the credit/cost per pair.
3. **Admission**: the three-leg Lagrange sweep on the RESOLVED pose (pass 7's `ddm_sj1_joint_admission.py` rule; seg on the shipped-mode
   decode; real-encode rate by twins), composed by re-verification (never additive; report the realized fraction of the sum; pass 8's law:
   the full set resolves to a cost, the subset keeps the credits). Exclude the 12 floor pairs [88,87,73,316,89,70,63,448,64,91,66,67] from
   the search (pp1 measured them) unless a proposal there is free.
4. If the admitted subset nets ΔS < −2e-5 on exact bytes: byte-close on the move-49 base (field + carrier; prior/semantic member/receiver
   code byte-identical), twins, cold n600 public parse-back (decoded scorer numbers = in-loop), manifest via the Catalog #420 producer,
   census, smokes candidate + frontier (four native-library exports as inflate.sh sets them), retention ≤ 8 GiB with shas (APDataStore;
   Vertigo is under its 40 GiB reserve — never lower it), seal on the NORMAL path inheriting move 49's adopted leg
   (`/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/SEAL_ddm_sj1_token_predistortion_pass7_contest_cuda.json.decode_wall_clock.json`; pr18
   behaviour digest 9f6e7168… must match). `tools/make_candidate_seal.py`; NO Modal, NO fire, NO packet (MAIN fires).
5. Memo `.omx/research/ddm_pd1_pose_directed_token_predistortion_all_pairs_20260913.md`: per-pair credit/cost histogram, the three-leg
   table, projected S, margins in units of the 34.8 B lottery and the instrument's pose reproduction, falsifier verdicts, every boundary;
   serializer commits (two visible review passes per .py; `[no-triality] [p0-ledger-ok]`; NEVER a co-author trailer or AI attribution);
   lane `ddm_pd1_pose_directed_token_predistortion_all_pairs_20260913` (claim it); checkpoint `ddm_pd1`. Heavy steps via
   `tools/launch_detached_process.py --output-dir /Volumes/APDataStore/pact/ddm_pd1/<stage> --nice 0 --done-receipt …`; waits as background
   receipt-only until-loops; `sys.dont_write_bytecode` against read-only trees.

## Boundaries
No Modal, no `authorize_*`, no `fire_modal_auth_eval.py`; never edit `upstream/`, the PR tree, sealed trees, contract code, receiver code, the
renderer (weights OR per-pair frame_embed — pp1 closed that), the basis, or the prior; never lower a reserve; sj1/pp1/cb1/cr1/so2 directories
read-only; no ScheduleWakeup. Label MEASURED / DERIVED / INFERRED / ASSUMED; state which solver each pose number uses (jg5.refine_pair =
pass 8's refine = cr1's; they are the same at these budgets).

## OPTIMAL FORM
Reference forms: pass 7's search/admission/pricer; pp1's actuator B and instrument; rp1 §8e's three-column pose leg. Declared deltas: the
OBJECTIVE (pose-ranked proposals over all pairs — the rung) and K (SCOPE, from a timing smoke). Provenance pins: HEAD (record); rp1 r2,
pass 7, pp1 memo shas (record); move 49 packet + seal + leg.

## Prior negatives accounted (operator 2026-08-15)
- pp1: the 12 floor pairs are outside every actuator — excluded.
- rp1 §8d: first-order token price is a ranking, never a charge — every rate number by real encode, twins.
- pass 8: a same-field pass decays and the full set resolves to a pose COST — per-pair selection is the whole method.
- cr1/cb1: the re-solve at the SHIPPED render is converged (1 of 600) — credits require a CHANGED render; that is what the edit supplies.
- m98 (address is the tax) and the 34.8 B lottery: quote margins.

Final message: the histogram, the three-leg table with exact bytes + sha if built, fraction of the sum, the seal path with file sha or the
typed blocker, commit shas, retained bytes, every boundary, ending with `composition S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600]
(move 49)` — a new number only if MAIN's fire moved it.

<!-- # FORMALIZATION_PENDING: charter, not a finding; the row, if any, carries its equations leg through tools/pointer_move_packet.py --equations-leg at harvest -->
