# ddm_sj1 pass 7 — Lagrange seg subset on the move 48 field (charter, MAIN 2026-09-12; operator full-authority GO)

## Why
Pass 6 (move 43, −1.9165e-4 S) reached 2.74 % of the remaining residual — ABOVE your own 1 %/pass convergence rule, so the family
was not closed. Every pointer move since (44 rider, 45 predictor refit, 46/48 rounding, 47 prior refit) was RATE-ONLY: the token field
of move 48 is byte-for-byte pass 6's output (field sha a92e7d90…; d_seg 0.00010345; d_pose 4.59e-6; archive 179,111 B, sha
d830edd3…; S 0.13638261682704697). No pair has been re-rendered since pass 6, so this is a SAME-FIELD pass: expect the decay the
same-field passes showed (3→4→5: 10.2 % → 3.3 % → 1.9 %), not pass 6's re-render rebound. Pre-registered expectation ≈ 1 % of
12,196 residual cells ≈ 120 cells ≈ 1.0e-4 S gross before token bytes; the fire bar is net ΔS < −2e-5 on the RESOLVED pose. If the
admitted subset does not project under the bar, the family closes at formulation scope on this field — say so with the curve.
rq1 (2026-09-12) and ren2 closed the renderer/realization axes in both directions; the token/field axis is the only live one.

## Resume surface (read first)
- Your pass-6 charter `.omx/research/ddm_sj1_pass6_seg_lagrange_on_move42_charter_20260910.md` and memo
  `.omx/research/ddm_sj1_t4_token_predistortion_pass6_20260910_pointer_move_43_20260910.md`; receipts under
  `/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/` (read-only; PASS_RESULT, REACH_DECOMPOSITION, DECISION_PREREGISTRATION, admission_pass6*).
- Base = move 48's SHIPPED bytes: `/Volumes/VertigoDataTier/pact/ddm_hpr1/public/retrain_frame_even/candidate_runtime` (read-only; copy).
  Read the field by parse-back of archive.zip, never from a sizing plane. The receiver of move 48 differs from move 43's (hpac rounding
  in the receiver — moves 46/48 were first-measurement rows): use move 48's receiver as shipped; your candidate changes archive.zip +
  the two pins only.
- Laws: pose re-solve mandatory on the edited field, admit only on the RESOLVED pose, frame-0 repair INSIDE the admission; bind base
  field + ranking by sha; token −log2p is a ranking never a charge; seg sub-additive by pair overlap (compose by re-verification);
  subset writer carries the live row's plane for dropped pairs (42d5fc651); stage-tail successor baseline rule (e6141fa4e).
- Seal contract: receiver byte-identical to move 48's except archive.zip and the two pins ⇒ NORMAL seal inheriting move 48's
  MEASURED `t4_direct` leg via `--inherit-decode-wall-clock` from
  `.omx/research/ddm_hpr1_20260911/v2/SEAL_ddm_hpr1_comp_even_on_refit_contest_cuda_v3.json.decode_wall_clock.json` (pr19: a measured
  leg may be inherited once; the pointer archive sha at seal time must be d830edd3…; report if the pairing refuses). pr18's behaviour
  digest must match (9f6e7168…). Timing risk mode `measured_t4_identity_class_envelope`.

## Deliverable
Pass 7 on the move-48 field: rank the residual boundary-jitter cells with your instrument (DALI GT lineage; `up2.verify_gt_lineage`
fails closed), admit a Lagrange subset under the three legs (seg on the shipped-mode decode, RESOLVED pose, real-encode rate) with
frame-0 repair inside the admission, twin-price by real encode against move 48's tail (never the ledger sum), pre-register falsifiers,
stage the seal inputs and call `tools/make_candidate_seal.py` on the normal path if the projection clears the bar. MAIN fires; packet
move 49 iff exact S < 0.13638261682704697. Report the seg/pose/rate decomposition, subset size, the yield curve vs the bar, and the
reach decomposition (how many admitted positions are carryover vs new). Heavy steps through `tools/launch_detached_process.py
--output-dir /Volumes/VertigoDataTier/pact/ddm_sj1_pass7/<stage> --nice 0 --done-receipt …`; waits as background receipt-only until-loops.

## Boundaries
CLAUDE.md non-negotiables; commits ONLY via the serializer with post-edit shas, `[no-triality] [p0-ledger-ok]`, two visible review passes
per .py, NEVER a co-author trailer or AI attribution; never edit `upstream/`, the PR tree, sealed trees, contract code
(`candidate_seal.py`, `decode_wall_clock.py`), or the receiver; no Modal, no `authorize_*`, no `fire_modal_auth_eval.py` (MAIN fires);
every payload on the SSD tier ≤ 8 GiB with sha certificates; never lower a reserve; no ScheduleWakeup. Host is otherwise idle.
Checkpoint as `ddm_sj1` (step numbering continues). Lane id `ddm_sj1_t4_token_predistortion_pass7_20260912` (claim it).

## OPTIMAL FORM
Reference form: pass 6 exactly (same machinery, same three-leg admission, same Lagrange rule, same n600 instrument); the only delta is the
base field/pointer (SCOPE). Provenance pins: HEAD (record), pass-6 memo + charter shas (record), move 48 seal (above), field a92e7d90….

## Prior negatives accounted (operator 2026-08-15)
- Pass 5 closed at 0.634 % on a same-field pass; pass 6 re-opened only because 365 planes were re-rendered — the honest prior here is the
  same-field decay; report the measured fraction against it. If < 1 %, the family closes on this field at formulation scope.
- rp1 r2 / sj1 silent revert: bind the base by sha; the subset carries the live plane for dropped pairs.
- Container lottery sd 34.8 B: quote the margin in its units.
- ntb2's prize is FALSIFIED (rq1): do not touch the renderer or its realization.

Final message: MEASURED vs not, the yield curve and fraction, subset size, the three-leg decomposition, the seal path or the typed blocker,
every boundary, and the frontier line `composition S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600] (move 48)` — a new number
only if MAIN's fire moved it.

<!-- # FORMALIZATION_PENDING: charter, not a finding; the pass-7 row, if any, carries its equations leg through tools/pointer_move_packet.py --equations-leg at harvest -->
