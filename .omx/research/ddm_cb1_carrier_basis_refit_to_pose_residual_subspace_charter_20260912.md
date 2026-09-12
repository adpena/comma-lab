# ddm_cb1 — re-FIT the pose carrier's twelve basis atoms to the CURRENT pose-residual subspace (fixed dims, fixed lattice, receiver unchanged); price on move 49 (charter, MAIN 2026-09-12; operator full-authority GO; Opus)

## The door, from four measured facts (read at source)
1. pp1 (`.omx/research/ddm_pp1_pose_directed_per_pair_actuation_on_the_hard_pairs_20260912.md`): on the 12 pairs holding 52.5 % of d_pose, the
   resolved pose is a FLOOR — render-side actuators fall 0.45 % at best, and an unmoved carrier re-solve moves 0.000 — so the residual is outside
   the carrier BASIS's reach, not a solve or render problem.
2. up2 (`.omx/research/ddm_up2_shipping_object_pose_solve_20260819.md` ~lines 280–300 and "follow-ons" item 2): the 12-atom basis moves pose hugely
   along one (global photometric) direction and barely along the residual's; relaxing the basis at the same 24×32 band-limit cuts the demanded
   step 1.8–18.3× per pair; "re-fit the 12-dim basis to the measured pose-residual subspace" was named the single highest-value follow-on
   (6.4× median demanded-step reduction; coefficients nearly free; the basis section must be re-priced). It was never fired.
3. br1 (`.omx/research/ddm_br1_pose_basis_reorientation_20260819.md`): re-ORIENTING the atoms inside their own span is provably null (the
   coefficient re-solve already covers it) — this charter changes the SPAN, not its orientation; and the free-field ceiling at that band-limit
   is d_pose ratio 0.7347 (d_pose is not fully cancellable). Pre-registered band (DERIVED): −1e-4 … −9e-4 S; falsifier: the refit basis +
   full re-solve does not cut n600 d_pose by ≥ 3 % vs move 49 (4.55e-6) at Δ bytes ≤ +200 B.
4. pc3 (`ddm_pc3_pose_carrier_rate_distortion_curve_on_move44_20260911.md` §4): rank/precision/width are receiver constants (closed); the basis
   VALUES (27,648 five-bit symbols, Huffman → RR5 adaptive rider, 12,277 B) and the coefficients (Rice/DX2) are DATA. A refit at fixed
   dims/lattice is a normal-seal row inheriting move 49's own measured leg (adopted today at
   `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/SEAL_ddm_sj1_token_predistortion_pass7_contest_cuda.json.decode_wall_clock.json`, 1,106.2 s).
   SegNet scores frame 1 only and the carrier warps frame 0 ⇒ d_seg unchanged BY CONSTRUCTION — still re-verify it on the decoded output.

## Deliverable
1. **Base and controls** on move 49 (S 0.13632299781031237 @ 179,153 B; tree `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/candidate/candidate_runtime`,
   read-only, copy): parse the carrier (`cpr1/carrier_codec.py`, `runtime/dx2_cabac_coefficients.py`, `runtime/rr5_arith_basis.py` — read the
   receiver's own parsers), reproduce the shipped basis/coefficients and the n600 per-pair pose base (4.5436e-6 mean; pp1's tolerance law:
   measure the batch-order band, gate at 10×), and repack the shipped carrier byte-identically (twins). Recall first: grep the store for any
   prior basis REFIT (not re-orientation) and cite it; if one exists and was priced, STOP and report.
2. **Measure the residual subspace**: for all 600 pairs, the per-pair pose-gradient / minimum-norm warp-field residual at the 3×24×32 band-limit
   (up2's instrument — reuse `experiments/ddm_up2_*` / `ddm_br1_*` producers; do not rewrite), weighted by pose mass; SVD; report the energy
   captured by 12 atoms of the residual subspace vs by the shipped span (MEASURED), and the top-12 pairs' share.
3. **Refit**: new 12 atoms = the pose-mass-weighted residual subspace (optionally a blend with the shipped span if the SVD says the shipped
   photometric direction still carries mass — report both), quantized to the shipped 5-bit lattice with the shipped per-atom scale rule; then the
   canonical per-pair coefficient re-solve on ALL 600 pairs (`jg5.refine_pair` / `up2.solve_pair_realized`, int12 codes, frame-0 repair inside),
   pose measured n600 on the frozen CPU-torch PoseNet, DALI GT lineage (`up2.verify_gt_lineage` fails closed).
4. **Price exactly**: basis section through the shipped chain (Huffman lengths + RR5 rider as shipped — the receiver decodes any 5-bit basis),
   coefficients through Rice/DX2, carrier body repacked into the archive on the move-49 base (token stream and semantic member byte-identical),
   twins; S from components. Fire bar net ΔS < −2e-5. Report the curve: d_pose vs basis bytes for ≥ 3 blends.
5. **If it nets**: byte-close, cold n600 public parse-back (decoded scorer numbers = in-loop; d_seg must equal move 49's 0.00010287 on the decoded
   output), manifest via the Catalog #420 producer, census, smokes candidate + frontier (four native-library exports as inflate.sh sets them),
   retention ≤ 8 GiB with shas (APDataStore overflow; Vertigo ~38 GiB free), seal on the NORMAL path inheriting move 49's leg (pr18 behaviour
   digest 9f6e7168… must match; receiver code byte-identical). `tools/make_candidate_seal.py`; NO Modal, NO fire, NO packet (MAIN fires).
6. Memo `.omx/research/ddm_cb1_carrier_basis_refit_to_pose_residual_subspace_20260912.md`: subspace energy table, the refit curve, per-pair
   table for the top-12, the exact price with margins (34.8 B lottery; instrument pose reproduction), falsifier verdict, every boundary;
   serializer commits (two visible review passes per .py; `[no-triality] [p0-ledger-ok]`; NEVER a co-author trailer or AI attribution); lane
   `ddm_cb1_carrier_basis_refit_pose_residual_subspace_20260912` (claim it); checkpoint `ddm_cb1`. Heavy steps via
   `tools/launch_detached_process.py --output-dir /Volumes/VertigoDataTier/pact/ddm_cb1/<stage> --nice 0 --done-receipt …`; waits as background
   receipt-only until-loops; `sys.dont_write_bytecode` against the read-only tree.

## Boundaries
No Modal, no `authorize_*`, no `fire_modal_auth_eval.py`; never edit `upstream/`, the PR tree, sealed trees, contract code, receiver CODE
(`CARRIER_DIM`, `BASIS_BITS`, `COEFFICIENT_BITS`, the band-limit, the rider set are all fixed), the renderer, or the token field; never lower a
reserve; sj1/ren/dpi1/rq1/so1/pp1 directories read-only; no ScheduleWakeup. Label MEASURED / DERIVED / INFERRED / ASSUMED.

## OPTIMAL FORM
Reference forms: up2's residual instrument and solve; br1's producers; the receiver's own carrier parsers/coders; pp1's tolerance law. Declared
deltas: the basis VALUES (the rung itself); blend count (SCOPE). Provenance pins: HEAD (record); up2/br1/pc3/pp1 memo shas (record); move 49
packet + seal + adopted leg; carrier body sha as parsed (record).

## Prior negatives accounted (operator 2026-08-15)
- br1: re-orientation is null — this is a span change; prove the new span is not in the old span's orbit (subspace angle, MEASURED).
- pc3: rank ×2 dead by arithmetic (12,277 B/12 atoms); this rung adds no atoms; basis bytes may still MOVE — price them.
- ra2: the basis stream is spatially white (no context to harvest) — do not expect the coder to rescue a denser basis.
- br1's free-field ceiling 0.7347: no claim beyond it; the band's far edge assumes ~half of the reachable quarter.
- Pose re-solve law (op 09-10): resolve on every pair; frame-0 repair inside; admit on the RESOLVED pose only.
- pp1: the hard pairs are a floor for render-side moves — this charter does not touch the render.

Final message: subspace energy table, refit curve, top-12 per-pair table, exact bytes + sha if built, the seal path with file sha or the typed
blocker, commit shas, retained bytes, every boundary, ending with `composition S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600] (move 49)`
— a new number only if MAIN's fire moved it.

<!-- # FORMALIZATION_PENDING: charter, not a finding; the row, if any, carries its equations leg through tools/pointer_move_packet.py --equations-leg at harvest -->
