# ddm_swp3 — re-stage the PR #140 swap packet on MOVE 44's bytes (prepare ONLY; nothing publishes) — charter, MAIN 2026-09-10

## Why
The submittable frontier is **move 44: S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600]**, archive sha
04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e (packet memo
`.omx/research/ddm_rlc5_counted_rider_rebase_move43_first_measurement_20260910_pointer_move_44_20260910.md`, sha f7638e1e171e0d19…, commit 99625f32f).
swp2 staged move 43's packet (`.omx/research/ddm_swp2_pr140_swap_packet_restage_move43_20260910.md`, sha 8e0a3ebe66895315…; compliance 91/93
after cpx2's checker path; two refusals: static import hygiene = receiver change, hosted manifest = at publish). Same rule: NO
publish/push/PR update without the operator's one-line confirm. Re-stage on the CURRENT bytes; everything else as swp2.

## Inputs
- Candidate tree (read-only): `/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/candidate_runtime` (archive 180,406 B; the
  rider cure moved the receiver's fitted constants INTO the archive — the tree's receiver differs from move 43's by the seven-path delta;
  the census is in `.omx/research/ddm_rlc5_20260910/LITERAL_CENSUS.json`).
- T4 receipt: `/Volumes/VertigoDataTier/pact/ddm_rlc5_first_measurement_run5/MODAL_REMOTE_RESULT.json` (call fc-01M26NNV3WR2XXXV914S8BTDR4);
  custody mirror `/Volumes/APDataStore/pact/ddm_rlc5_counted_rider_rebase_move43_first_measurement_20260910/custody_pointer44`.
- CPU axis: MAIN fired the CPU-axis dispatch on move 44's own bytes (job ddm_rlc5_move44_cpu_20260910, pair group
  ddm_rlc5_move44_pair_20260910, output `/Volumes/VertigoDataTier/pact/ddm_rlc5_move44_cpu_20260910/`); it refuses by design in
  seconds. POLL that dir for `MODAL_REMOTE_RESULT.json` (artifact-bound wait), then write the move-44 CPU adjudication record in the
  shape of `.omx/research/ddm_sj1_pass6_packet_inputs_20260910/CPU_AXIS_ADJUDICATION.json` (verdict REFUSED_BY_DESIGN; call id;
  receipt `{path,bytes,sha256}`) at `.omx/research/ddm_rlc5_packet_inputs_20260910/CPU_AXIS_ADJUDICATION.json`, and feed it to
  the checker's typed `contest_cpu_axis_refusal.v1` path exactly as swp2/cpx2 did.
- swp2's memo, BLOCKERS.json, COMPLIANCE_COMMAND.json; `docs/submission_template.md`; `scripts/pre_submission_compliance_check.py --help`.

## Deliverable (all staged under `submissions/_staging_move44_pr140_swap/`, none published; commit manifests + text only)
1. Byte-exact staged copy of the shippable members with per-file sha256; archive sha/size = 04758c0d… / 180,406 B; grep the staged
   tree for retracted receiver code (`tc3`/`tc4`/lane-predictor symbols) and prove absence; confirm the rider's constants are in
   the ARCHIVE, not in free code (census).
2. `report.txt` from the T4 receipt; score recomputed from components must equal 0.1372449041713402 to the last digit
   (d_seg 0.00010345, d_pose 4.59e-06, 180,406 B) — else STOP and report.
3. `scripts/pre_submission_compliance_check.py --contest-final --strict` with expected sha/size, the auth-eval JSON, the move-44 CPU
   refusal receipt (typed path), archive manifest, dispatch-claim linkage; full output retained; expect 91/93 with the same two
   remaining (import hygiene = operator decision; hosted manifest = at publish); any NEW refusal named with its cure class.
4. `PR_BODY.md` (declares `linux-nvidia-t4`; move 44's components; the neutral disclosure sentence from swp2, unchanged) +
   `SWAP_COMMANDS.md` (exact commands the confirm would run; marked NOT RUN).
5. Memo `.omx/research/ddm_swp3_pr140_swap_packet_restage_move44_20260910.md`: verification table, compliance count vs swp2's 91/93,
   blocker table, boundaries. Serializer commit LAST, once (`REVIEW_GATE_OVERRIDE=1` for non-.py); rc 17 is NOT a stop. Checkpoint as
   `ddm_swp3` and mark it COMPLETE at the end.

## Hard boundaries
Do NOT edit the live PR tree, `upstream/`, any `/Volumes/...` path, or any receiver file; no `gh pr`, no push, no hosting, no
Modal/GPU dispatch (the CPU refusal is MAIN's, already fired), no score claim beyond the retained T4 receipt; no Claude/AI
attribution in public-facing text.

## OPTIMAL FORM
- Reference form: swp2's staging + cpx2's typed CPU-refusal path; scope delta: bytes = move 44 and the rider tree. No mechanism delta.
- Provenance pins (sha256 prefixes): packet memo f7638e1e171e0d19…; swp2 memo 8e0a3ebe66895315…; archive sha 04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e;
  pointer commit 99625f32f; run5 receipt path above (record sha); CPU refusal receipt (record sha when it lands).

## Prior negatives accounted (operator 2026-08-15)
- hv1: never retype a sha or a score; copy from receipts and recompute.
- cpx1: a CPU refusal bound to a DIFFERENT pointer is not evidence — use only the move-44 receipt.
- swp2's blockers: import hygiene is a receiver change (fresh exact rows) — carry as an operator decision, do not apply.
- pr5/pr6: disclosure must not name assistants.

Final message: what is staged, the recomputed score, compliance count + remaining blockers, the disclosure sentence, serializer rc,
and the frontier line `composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)`.

<!-- # FORMALIZATION_PENDING: staging charter; no measured row of its own -->
