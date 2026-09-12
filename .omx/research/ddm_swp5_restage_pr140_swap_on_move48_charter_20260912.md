# ddm_swp5 — re-stage the PR #140 swap packet on pointer move 48 WITHOUT publishing (charter, MAIN 2026-09-12; operator full-authority GO; publish stays gated on the operator's one-line confirm)

## Why
swp4 (`.omx/research/ddm_swp4_restage_pr140_swap_on_move47_20260911.md`, sha 8062f5c7…) staged move 47 at 91/93 with exactly the two
known open items. The pointer moved once more: **move 48** (hpr1 even rounding on the refit prior; S **0.13638261682704697**, archive
**179,111 B**, sha `d830edd371641e1968765ae6be27120a6c55a9b1ca4b3747158e603a43ef149c`; rate-only; distortion unchanged d_seg 0.00010345 /
d_pose 4.59e-6; raw 2b762eba… identical; behaviour digest 9f6e7168…). Every refit rung on every counted section is now PRICED (tmx1, dpi1,
ren2 all lose), so move 48 is the day's final pointer and the packet must sit on it. NOTHING is pushed, hosted, or made public; no
Claude/AI attribution in any public text; the live PR tree `submissions/semantic_joint_ctxmix/` is never touched.

## Deliverable
1. Re-stage `submissions/_staging_move48_pr140_swap/` from move 48's promoted tree
   `/Volumes/VertigoDataTier/pact/ddm_hpr1/public/retrain_frame_even/candidate_runtime` (read/copy only; archive sha d830edd3…,
   179,111 B) exactly as swp4 staged move 47 (same layout and sanitisation: no private paths, no provider logs, no operator state;
   README/report text user-attributed).
2. `scripts/pre_submission_compliance_check.py --contest-final --strict` with `--expected-archive-sha256 d830edd3…` (full),
   `--expected-archive-size-bytes 179111`, the canonical auth-eval JSON
   `/Volumes/VertigoDataTier/pact/ddm_hpr1_first_measurement/run1/MODAL_REMOTE_RESULT.json` (sha 9bd402ca…, 242,361 B; call
   fc-01M29A1CWQBBDT4XXTQ1K7TG88; measured t4_direct 1,023.26 s), the archive manifest, dispatch-claim linkage (read the lane id from
   `.omx/research/ddm_hpr1_20260911/v2/FIRST_MEASUREMENT_AUTHORIZATION.json` / `.omx/state/active_lane_dispatch_claims.md`), and the
   typed CPU-axis refusal receipt `.omx/research/ddm_hpr1_packet_inputs_20260911/CPU_AXIS_ADJUDICATION_move48.json`
   (`--submission-score-axis contest_cuda`). Record the 93-item state; the two open items (receiver `experiments` fallback-import hygiene
   = operator's receiver-change decision; hosted manifest = at publish) stay open — do not close them by assumption.
3. Fresh `report.txt` from the T4 receipt with the score recomputed from components (100·0.00010345 + √(10·4.59e-6) + 25·179,111/37,545,489
   → 0.13638261682704697; state each component to full precision from the receipt); diff the staged tree vs the live PR tree and vs
   swp4's staging (member-level: byte counts, shas — only the `hpac` member and the archive should differ from move 47).
4. Memo `.omx/research/ddm_swp5_restage_pr140_swap_on_move48_20260912.md` (`# FORMALIZATION_PENDING:<rationale>`), checkpoint as
   `ddm_swp5`, serializer commits (no .py expected; `REVIEW_GATE_OVERRIDE=1` for non-.py; no co-author trailer; `[no-triality]
   [p0-ledger-ok]`); update P0 row `p0_swap_procedure_no_push_without_confirm_20260817` via
   `tools/operator_p0_digest.py --update … --status open` (single quotes; no backticks inside).

## Boundaries
NO push, NO PR update, NO hosting, NO public text with AI attribution; never edit `upstream/`, the live PR tree, sealed trees, or other
arms' directories. $0. No Modal. If any compliance item needs a receiver change, STOP and report — receiver changes are an operator decision.

## OPTIMAL FORM
Reference form: swp4's staging on move 47 (91/93; measured with cpx3's checker) — same tools, same checklist; the only delta is the pointer
(SCOPE). Provenance pins: pointer move 48 (commit a3777a2b2), seal
`.omx/research/ddm_hpr1_20260911/v2/SEAL_ddm_hpr1_comp_even_on_refit_contest_cuda_v3.json`, packet memo
`.omx/research/ddm_hpr1_comp_even_on_refit_move47_first_measurement_20260911_pointer_move_48_20260911.md`, swp4 memo sha 8062f5c7….

## Prior negatives accounted (operator 2026-08-15)
- swp2/swp3/swp4: staged, never pushed — the confirm is the operator's; never infer it from a GO.
- swp4's lineage note: cite the checker that produced the 93-count (cpx3's), not an inherited number.
- cpx1/cpx2: the CPU axis is a declaration + refusal receipt, never an inferred score.

Final message: staged path, checklist state (N/93 with the open items named), report.txt score line, the member-level diff, commit shas,
and `composition S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600] (move 48)` unchanged (NOT published).
