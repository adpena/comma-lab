# ddm_jrd1 — JOINT rate-aware descent of renderer + field + prior: STEP 0, the joint exchange probe on move 52 (charter, MAIN 2026-09-16; operator standing GO "be creative and weird, think divergently"; codex astra xhigh)

## Why this object and not another arm on the field
Every single axis of the shipped object is at its knee (gs3 `.omx/research/ddm_gs3_gestalt_after_submission_20260903.md`
Addenda 59 and 65): the field (sj1 passes 1–8, pd1–pd4), the renderer basis (cb1: "bytes buy a position, not a spectrum"),
the in-place renderer refit (ren2), the int4 weight grid (rw1: smallest action breaks 240 cells), the carrier lattice
(pc2/pc3), the tail coder (ls1/ls2, bd1). Each closure moved ONE part while the other three stayed frozen. MAIN's answer to
the operator's "what would break through to below 0.12" (2026-09-13): not this object under single-axis moves; a JOINT
rate-aware descent of renderer + field + prior, where the field's real coded bits and the renderer's quantized-weight bits
are in the loop together with the frozen scorer's argmax and the resolved pose. Nobody has measured whether a joint direction
exists that the single-axis closures cannot see. That is the whole of this arm: MEASURE THE JOINT EXCHANGE. No burn.

Arithmetic at move 52 (DERIVED from the packet; the arm re-derives from components): S 0.13620226906030858 =
100·0.00010304 (0.010304) + sqrt(10·4.21e-6) (0.006488) + 25·179,332/37,545,489 (0.119410). Rate is 87.7 % of S. Sub-0.12
needs −0.01620 S. At held distortion the archive must fall to ≤ 155,0xx B (−24.3 kB, 13.6 %); at held bytes the whole
distortion (0.0168) must vanish, which the round-trip intercept forbids (accuracy half closed above 140,477 B). Only a
direction that moves bytes AND distortion together is unmeasured. Section map (DERIVED, re-measure through the packet
grammar): token tail 119,097 B (66 %); renderer + prior + carrier + headers ≈ 60,135 B.

## The probe (SCOPE-reduced to K pairs; MECHANISM at full form — real coder, real scorer, real renderer, resolved pose)
Base = move 52 by parse-back (tree `/Volumes/APDataStore/pact/ddm_pd3/candidate/candidate_runtime`, read-only; copy; archive
sha ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e). Pick K = 24 pairs by a SEEDED random draw (seed
recorded; NOT the prefix — prefix bias inverts by axis, m88/m96) stratified 12 from pd4's residual-heavy pool and 12 from
the untouched remainder. For each pair, measure three exchange rows, all in S per byte with the real price:
1. **Field-only** (the control; must reproduce pd4's knee): resolved-pose seg credit per real coded bit for the best
   single/2-token move (pd4's machinery; twins encode of the pair plane through the real coder). Expected ≈ 1/12 bar per
   bit-equivalent — the arm states the number it reproduces and the tolerance.
2. **Renderer-only at held field**: the gradient of the boundary-jitter seg residual (86 % of the residual sits at correct
   tokens, 1 px jitter — memory `residual_seg_debt_is_renderer_boundary_jitter_at_correct_tokens_20260908`) w.r.t. the
   renderer's quantized weights, projected onto the int4 grid's nearest representable step per weight, priced by the REAL
   change in the weight section's coded bytes (re-encode the section; do not assume 0 B). rw1 says the smallest grid action
   breaks 240 cells globally; the arm measures the PER-PAIR-RESTRICTED version and its global collateral on the other
   599 pairs (the B/H collateral law, m132) — this is the row rw1 did not price.
3. **Joint**: the direction in (field tokens, renderer weights) that minimizes seg debt at ZERO net coded bytes to first
   order (Lagrange with the real prices from rows 1–2 as the multipliers), realized by the smallest admissible discrete
   step in both parts, re-rendered, pose re-solved (per-pair carrier re-solve with frame-0 repair; state the solver),
   seg on the frozen argmax, bytes by real re-encode of both sections (twins). Report the realized exchange vs the sum of
   rows 1 and 2 (UNION ≠ SUM, m164) and vs the pointer exchange 6.658589531221714e-7 S/B.
Pre-registered gate (the only question): does the joint row's realized exchange exceed the better single-axis row by
≥ 2× on the median pair, with global collateral (all 600 pairs) priced in? If YES: the joint formulation has a lever above
the knees and MAIN charters the burn (jrd2). If NO: the joint direction is closed at this operating point and the object
is finished; say so plainly. Falsifiers: joint realized exchange < 1.5× the best single row; renderer-row collateral on
untouched pairs exceeds its per-pair credit; row 1 fails to reproduce pd4's knee within tolerance (then the instrument,
not the object, is under test — STOP and report).

## Apparatus, boundaries, retention
No burn, no training, no Modal, no fire, no packet; never edit `upstream/`, the PR tree, sealed trees, contract code, the
receiver, the shipped renderer weights, basis or prior IN THE TREE (all changes on copies under
`/Volumes/APDataStore/pact/ddm_jrd1/`; report free space before every heavy step; retain ≤ 3 GiB with sha256; Vertigo
untouched under its reserve). pd5 (Opus) shares the host and owns `/Volumes/APDataStore/pact/ddm_pd5/` and pd4's pool
files — read them, never write them. Heavy steps via `tools/launch_detached_process.py --output-dir … --nice 0
--done-receipt …`; background receipt-only waits. Label MEASURED / DERIVED / INFERRED / ASSUMED. Serializer commits with
post-edit shas; two visible review passes per .py; `[no-triality] [p0-ledger-ok]`; NEVER a co-author trailer or AI
attribution. If the serializer refuses with a Git-object write denial (rc 17), that is NOT a stop condition: continue to
the end, keep every file in the working tree, leave the verified bundle + format-patch under
`.omx/research/ddm_jrd1_20260916/`; MAIN lands. Commit LAST. Checkpoint as `ddm_jrd1`; lane
`ddm_jrd1_joint_rate_aware_descent_step0_exchange_probe_20260916` (claim it).

## OPTIMAL FORM
Reference forms: pd4's per-pair pricer + resolved-pose admission (row 1); rw1's int4-grid action + ren2's refit machinery
(row 2); the real coder for every byte. Declared deltas: K = 24 pairs (SCOPE); first-order Lagrange direction realized by
one discrete step (SCOPE — the burn is jrd2's). No MECHANISM reduction: no ledger sums, no first-order token prices as
charges (m: first-order token price is a ranking, never a charge), no proxy scorer, no unresolved pose. Provenance pins: HEAD 59bcc5a9f (charter commit 59bcc5a9f); pd4 memo `.omx/research/ddm_pd4_pose_directed_pass4_price_lever_on_move52_20260913.md` sha 48767941b8a9cf1e; rw1 memo `.omx/research/ddm_rw1_boundary_local_renderer_weight_foldback_20260909.md` sha 233ed1b53a35585b; ren2 memo `.omx/research/ddm_ren2_renderer_refit_in_place_on_the_coded_field_20260912.md` sha 3b9cdbbdb6a7bfdf; cb1 memo `.omx/research/ddm_cb1_carrier_basis_refit_to_pose_residual_subspace_20260912.md` sha 07b6e9ecdc911116; move 52 seal `/Volumes/APDataStore/pact/ddm_pd3/SEAL_ddm_pd3_pose_directed_pass3_contest_cuda.json` sha 9905cb9238cccf29, archive sha256 ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e, pointer commit d1fc2a1c2.
HEAD (record), pd4 memo sha (record), rw1/ren2/cb1 memo shas (record), move 52 seal + leg.

## Prior negatives accounted (operator 2026-08-15)
rw1 (int4 grid: global smallest action breaks 240 cells — this arm prices the per-pair-restricted action WITH collateral);
ren2 (refit in place loses — that was renderer-only at held field; here the field moves too); cb1 (basis refit
monotonically worse — basis is NOT a free variable here); pd4 (selection on a real price still ranks — K pairs are drawn,
not selected); m164 UNION ≠ SUM; m132 collateral; m88 prefix bias.

Final message: the three exchange rows (median + IQR over K pairs, S/B and bits), the joint-vs-single ratio, the collateral
row, the gate verdict in one sentence, every path + sha, the serializer rc, every boundary, ending with
`composition S 0.13620226906030858 @ 179,332 B [contest-CUDA T4 n600] (move 52)`.

<!-- # FORMALIZATION_PENDING: charter, not a finding; the probe's exchange rows land as an EmpiricalAnchor on the exchange law at harvest -->
