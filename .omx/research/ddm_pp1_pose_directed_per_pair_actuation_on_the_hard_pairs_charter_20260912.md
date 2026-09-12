# ddm_pp1 — POSE-DIRECTED per-pair actuation on the pairs that carry the pose term (charter, MAIN 2026-09-12; operator full-authority GO + "creative and divergent"; Opus)

## The bet, derived from three measured facts (read each at source; never from memory)
1. **Pose is concentrated.** On move 49's own per-pair receipt (`/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/pose/POSE_base.json` → `per_pair_path`;
   MAIN re-read 2026-09-12): the top-12 pairs [88, 87, 73, 70, 316, 89, 63, 448, 64, 91, 66, 67] hold **52.7 %** of d_pose (worst pair 2.13e-4 vs
   median 3.4e-7 = 630×); **65.7 %** of the pose mass sits in pairs pass 7 never edited; the hard pairs cluster in the 60–120 block (the two
   hardest 60-pair blocks, per the prefix-bias memory). The pose term is √(10·d_pose) = 0.006745 S at d_pose 4.55e-6: cutting the top-12
   mass by 10 % is −1.6e-4 S (8× the −2e-5 bar); by 50 % is −8.5e-4 S. Rate cost of per-pair code moves: fe1's container law (+70 B flat,
   recoverable to +1.3 B (N=1) / +28 B (N=72) by the q/lgwin/ck2 search) — tens of bytes.
2. **A per-pair render change is payable on pose** (fe1, `.omx/research/ddm_fe1_per_pair_frame_embedding_realized_search_20260908.md` §"THE LAW"):
   the pair's twelve carrier coefficients re-aim after it (`jg5.refine_pair`); post-re-solve cost +2.6e-8 S per moved pair; the 170–220×
   coupling belongs to ALL-pairs changes only. fe1 was SEG-directed and found the 3-bit FiLM lattice too coarse for seg; it did not aim at pose.
3. **The per-pair `frame_embed` codes are a working POSE actuator** (iv1, `.omx/research/ddm_iv1_inversion_pose_actuator_20260818.md`): re-solving
   them in the d_pose-descent direction cut realized d_pose −90.48 % on 48 seeded-random pairs through the exact shipping decode, with near-zero
   seg cross-talk (+5 flipped pixels on 12 pairs); honest transfer ~7× smaller than the headline. That was the 08-18 object (d_pose ~10× today's);
   nobody has aimed it at TODAY's hard pairs on TODAY's field and carrier.
Pre-registered band (DERIVED, not measured): −5e-5 … −5e-4 S net on the top-K pairs; falsifier: if the resolved per-pair d_pose on the treated
pairs does not fall by ≥ 5 % at seg-neutral admission, the actuator has no supply on the hard pairs and the family closes there.

## Deliverable
1. **Base**: move 49 (S 0.13632299781031237 @ 179,153 B; archive sha 73e41a66…; tree `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/candidate/candidate_runtime`,
   read-only, copy; field by parse-back). Reproduce the per-pair pose base on your instrument (`[cpu_torch fp32, DALI GT, n600]`) and the
   pair ranking above (MEASURED; F1). Controls: re-solving an untouched pair moves d_pose by exactly 0 (fe1's converged-carrier control); the
   pointer repacks byte-identically through your pricer (twins).
2. **Actuator A — per-pair `frame_embed` pose-descent** (iv1's mechanism; locate its code by `git grep -n "def refine_pair"` and the iv1/fe1
   scripts; reuse, do not rewrite): for each of the top-K pose pairs (K = 12 first, then extend down the ranking while the marginal pair still
   pays), search the pair's 8 signed 3-bit codes (single- and two-code moves; the shipped lattice, no receiver change) for the code vector that
   minimizes the RESOLVED d_pose of that pair after `refine_pair` (carrier re-solved on the new render), subject to seg neutrality re-verified
   on the frozen CPU SegNet argmax of the pair's re-render (accept ≤ 0 net flipped cells; record any positive as a cost at 8.48e-7 S/cell).
3. **Actuator B — token edits on the same pairs, pose-directed**: for the same pairs, propose sj1-family single-token moves ranked by resolved
   pose gain (not seg), admitted on the three legs (seg on the shipped-mode decode, RESOLVED pose, real-encode rate). Report A, B, and A+B
   per pair; composition by re-verification (never additive).
4. **Price exactly**: semantic member re-encode after the frame_embed moves (fe1's container search; shipped shape q10/lgwin16+CK2 per ren2's
   correction), token stream by the RLC1 pricer (`experiments/ddm_sj1_rlc1_price.py`, proved by byte-identical repack), carrier re-solved and
   repacked; twins; S from components; frame-0 repair inside the admission (op 09-10). Fire bar net ΔS < −2e-5 on the RESOLVED pose.
5. **If it nets**: byte-close on the move-49 base (member values + field + carrier; receiver code byte-identical), cold n600 public parse-back
   with the decoded output's scorer numbers equal to the in-loop numbers, manifest regenerated from outside the tree (Catalog #420 producer),
   census, smokes candidate + frontier with the four native-library exports as inflate.sh sets them, retention ≤ 8 GiB with shas (APDataStore
   overflow if Vertigo < 42 GiB free). **Seal on the NORMAL path inheriting move 49's OWN measured `t4_direct` leg** — MAIN adopted it today at
   `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/SEAL_ddm_sj1_token_predistortion_pass7_contest_cuda.json.decode_wall_clock.json` (1,106.2 s ≤ 1,260;
   `inherit_decode_wall_clock(source_leg_path=…, runtime_dir=…, archive_path=…, pointer_archive_sha256='73e41a66…')` verified) — pr18's
   behaviour digest 9f6e7168… must match. Call `tools/make_candidate_seal.py`; NO Modal, NO fire, NO packet (MAIN fires).
6. Memo `.omx/research/ddm_pp1_pose_directed_per_pair_actuation_on_the_hard_pairs_20260912.md`: the pair table (base d_pose, A / B / A+B resolved
   d_pose, seg cells, bytes, net S per pair), the band vs the measurement, falsifier verdicts, margins in units of the 34.8 B lottery and the
   instrument's pose reproduction, every boundary; serializer commits (two visible review passes per .py; `[no-triality] [p0-ledger-ok]`;
   NEVER a co-author trailer or AI attribution); lane id `ddm_pp1_pose_directed_per_pair_actuation_20260912` (claim it); checkpoint `ddm_pp1`.
   Heavy steps via `tools/launch_detached_process.py --output-dir /Volumes/VertigoDataTier/pact/ddm_pp1/<stage> --nice 0 --done-receipt …`; waits
   as background receipt-only until-loops; `sys.dont_write_bytecode` in every run against the read-only tree (pass 8's lesson).

## Boundaries
No Modal, no `authorize_*`, no `fire_modal_auth_eval.py`; never edit `upstream/`, the PR tree, sealed trees, contract code, receiver CODE, or
renderer WEIGHTS (the per-pair `frame_embed` VALUES are field-like counted data — the only renderer-side object you may move, and only per pair);
never lower a reserve; sj1/ren/dpi1/rq1/so1 directories read-only; no ScheduleWakeup; DALI GT lineage authority (`up2.verify_gt_lineage`).
Label MEASURED / DERIVED / INFERRED / ASSUMED.

## OPTIMAL FORM
Reference forms: iv1's actuator + `refine_pair` re-solve; fe1's realized per-pair search harness and container law; sj1 pass 7's three-leg
admission and RLC1 pricer; the n600 CPU-torch pose instrument. Declared deltas: the OBJECTIVE (pose-descent on pose-ranked pairs — the rung
itself) and K (SCOPE). Provenance pins: HEAD (record), iv1/fe1/pass-7 memo shas (record), move 49 packet + seal + adopted leg (above).

## Prior negatives accounted (operator 2026-08-15)
- iv1's 7× transfer shrink: report per-pair realized numbers only; no headline from a subsample.
- fe1: seg-directed FiLM moves make seg WORSE on 95 % of pairs — this arm admits on seg NEUTRALITY and aims at pose; a seg cost is charged.
- pc2/pc3: the carrier alone (field fixed) is closed at 350 B family slack — this arm moves the RENDER, then re-solves; different door.
- pass 7: the full 130-pair field resolved to a pose COST — the credits are per pair; keep the Lagrange selection per pair.
- m98 (address is the tax): per-pair addressing is paid by the container search — price it, never assume "zero bytes".
- Container lottery sd 34.8 B; instrument pose reproduction 0.07 % (ren1): quote margins in both.

Final message: the pair table, the band vs the measured net, A/B/A+B verdicts, exact archive bytes + sha if built, the seal path or the typed
blocker, commit shas, retained bytes, every boundary, and `composition S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600] (move 49)`
— a new number only if MAIN's fire moved it.

<!-- # FORMALIZATION_PENDING: charter, not a finding; the row, if any, carries its equations leg through tools/pointer_move_packet.py --equations-leg at harvest -->
