# ddm_swp4 — re-stage the PR #140 swap packet on pointer move 47 WITHOUT publishing (charter, MAIN 2026-09-11; operator full-authority GO; publish stays gated on the operator's one-line confirm)

## Why
The staged swap (swp3, `.omx/research/ddm_swp3_pr140_swap_packet_restage_move44_charter_20260910.md`) sits at 91/93 on MOVE 44 (180,406 B). The pointer moved three times today —
45 (pc3 carrier refit, 180,246 B), 46 (ntb2 prior rounding, 180,001 B), 47 (hpr1 prior refit, 179,359 B; S 0.13654774984742127) — all
rate-only, distortion unchanged, receiver unchanged (behavior digest 9f6e7168…). A staged packet three moves stale is publish latency;
the operator's confirm is a one-liner and the packet must be ready the minute it comes. NOTHING is pushed, hosted, or made public by
this arm; no Claude/AI attribution in any public text; the live PR tree `submissions/semantic_joint_ctxmix/` is never touched.

## Deliverable
1. Re-stage `submissions/_staging_move47_pr140_swap/` from move 47's promoted tree (`/Volumes/VertigoDataTier/pact/ddm_hpr1/public/retrain_control/candidate_runtime`,
   archive sha d1fab05d69f31c90ac55173fa87072949e5ea1e069a0b7614337089b7a2a0ce9, 179,359 B) exactly as swp3 staged move 44 (same layout,
   same sanitisation: no private paths, no provider logs, no operator state; README/report text user-attributed).
2. `scripts/pre_submission_compliance_check.py --contest-final --strict` with `--expected-archive-sha256 d1fab05d…`,
   `--expected-archive-size-bytes 179359`, the canonical auth-eval JSON (`/Volumes/VertigoDataTier/pact/ddm_hpr1_fire_move47/run1/MODAL_REMOTE_RESULT.json`),
   the archive manifest, dispatch-claim linkage (lane `ddm_hpr1_retrain_control_move46_contest_cuda_20260911`, call fc-01M29166GTAZC3TA5V8W8Q54E1),
   and the typed CPU-axis refusal receipt (`.omx/research/ddm_hpr1_packet_inputs_20260911/CPU_AXIS_ADJUDICATION.json`; cpx3 path,
   `--submission-score-axis contest_cuda`). Record the 93-item checklist state; the two known open items are the receiver-hygiene
   decision (operator's) and the hosted manifest (at publish) — do not close them by assumption.
3. Fresh `report.txt` from the T4 auth-eval receipt (score recomputed from components: d_seg 0.00010345, d_pose 4.59e-6, 179,359 B →
   0.13654774984742127); diff the staged tree vs the live PR tree and vs swp3's staging (member-level, byte counts, shas).
4. Memo `.omx/research/ddm_swp4_restage_pr140_swap_on_move47_20260911.md` (`# FORMALIZATION_PENDING:<rationale>`), checkpoint as
   `ddm_swp4`, serializer commits (two review passes per .py if any .py changes — none expected), no co-author trailer, tags
   `[no-triality] [p0-ledger-ok]`. Update `.omx/state` P0 row `p0_swap_procedure_no_push_without_confirm_20260817` evidence via
   `tools/operator_p0_digest.py --update … --status open` (single quotes; no backticks inside).

## Boundaries
NO push, NO PR update, NO hosting, NO public text with AI attribution; never edit `upstream/`, the live PR tree, sealed trees,
or other arms' directories (ddm_hpr1 is live — read/copy only). $0. If any compliance item needs a receiver change, STOP and
report — receiver changes are an operator decision.

## OPTIMAL FORM
- Reference form: swp3's staging on move 44 (91/93) and cpx3's typed CPU-axis path; same tools, same checklist; the only delta is
  the pointer (SCOPE), no mechanism change.
- Provenance pins: pointer move 47 (commit 40518b844), seal `.omx/research/ddm_hpr1_20260911/SEAL_ddm_hpr1_retrain_control_contest_cuda.json`,
  packet memo `.omx/research/ddm_hpr1_retrain_control_move46_contest_cuda_20260911_pointer_move_47_20260911.md`, swp3 memo (record sha).

## Prior negatives accounted (operator 2026-08-15)
- swp2/swp3: the packet was staged, never pushed — the confirm is the operator's; do not infer it from a GO.
- cpx1/cpx2: the CPU axis is a declaration + refusal receipt, never an inferred score.
- Public disclosure hygiene: private paths/logs leaked into an earlier push (jg5 custody P0) — sanitise and list every path checked.
