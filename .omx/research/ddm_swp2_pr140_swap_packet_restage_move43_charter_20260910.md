# ddm_swp2 — re-stage the PR #140 swap packet on move 43's bytes (prepare ONLY; nothing publishes) — charter, MAIN 2026-09-10

## Why
PR #140 carries pointer move 23. The submittable frontier is now **move 43: S 0.1372848557085275 @ 180,466 B
[contest-CUDA T4 n600]**, archive sha 7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e (packet memo
`.omx/research/ddm_sj1_t4_token_predistortion_pass6_20260910_pointer_move_43_20260910.md`, sha d57ab4d55b3d7b0d…, commit 48109233e). swp1/pk1 staged move 40's packet (`.omx/research/ddm_pk1_pr140_swap_packet_staging_20260910.md`, sha d61b93ba6ec7460c…; compliance 70/76 with
blockers adjudicated); that staging is superseded. Same rule: NO publish/push/PR update without the operator's one-line
confirm (`p0_swap_procedure_no_push_without_confirm_20260817`). You make the confirm a one-command act on the CURRENT bytes.

## Hard boundaries (binding)
- Do NOT edit `submissions/semantic_joint_ctxmix/` (the live PR tree), `upstream/`, or anything under
  `/Volumes/*/pact/ddm_sj1_pass6/` (sealed custody, read-only source). Stage under NEW `submissions/_staging_move43_pr140_swap/`
  (check `.gitignore`; commit only manifests + text, never binaries).
- No `gh pr`, no `git push`, no hosting upload, no Modal/GPU dispatch, no score claim beyond the retained T4 receipt.
- NO receiver code changes. pk1 found `except ImportError` fallbacks importing `experiments` in `runtime/rc3_shared_mixer.py`
  + `sm1_semantic_mixer.py`; a fix is a RECEIVER CHANGE that needs fresh exact rows — do NOT apply it; carry it as a named
  operator decision with the exact lines, and prove with a bare-venv smoke whether the fallback path is ever reached from
  the shipped tree (if never reached, say so with the receipt; that is the packet's answer).
- Moves 41/tc4 receiver code must be absent (rule 118): grep the staged tree for `tc3`/`tc4`/lane-predictor symbols.
- No Claude/AI attribution in any public-facing text; reuse pk1's neutral disclosure draft (`.omx/research/ddm_pk1_20260910/PR_BODY.md`)
  and keep the before/after in your memo for the operator.

## Inputs
- Candidate tree (read-only): `/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/candidate/candidate_runtime`.
- T4 receipt: `/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/fire/MODAL_REMOTE_RESULT.json` (Modal call fc-01M2691VTJYKNCXX01CKKTN44N;
  custody mirror `/Volumes/APDataStore/pact/ddm_sj1_t4_token_predistortion_pass6_20260910/custody_pointer43`).
- CPU axis: REFUSED BY DESIGN (receiver declares `linux-nvidia-t4`; `inflate.sh` raises on the CPU path); the packet's CPU
  requirement is met by the runtime DECLARATION plus the refusal receipt: `.omx/research/ddm_rp1_round2_packet_inputs_20260910/CPU_AXIS_ADJUDICATION.json`
  (Modal call fc-01M25W34DGHHJES03531X3VDXV). Use `--submission-score-axis contest_cuda` and pass the refusal receipt as
  `--contest-cpu-auth-eval-json` exactly as pk1's memo records; if the checker refuses that form, record the exact refusal.
- pk1's memo and its 6 blockers; `docs/submission_template.md`; `scripts/pre_submission_compliance_check.py --help`.

## Deliverable (all staged, none published)
1. Byte-exact staged copy of the shippable members with per-file sha256; archive sha/size = 7beb6a5f… / 180,466 B.
2. `report.txt` from the T4 receipt; score recomputed from components must equal 0.1372848557085275 to the last digit
   (d_seg 0.00010345, d_pose 4.59e-06, 180,466 B) — else STOP and report.
3. `scripts/pre_submission_compliance_check.py --contest-final --strict` with expected sha/size, the auth-eval JSON, the
   CPU refusal receipt, archive manifest, dispatch-claim linkage; full output retained; every remaining refusal named
   with its cure class (declaration / receiver change / manifest) — nothing waived.
4. `PR_BODY.md` (declares `linux-nvidia-t4`; move 43's components; neutral disclosure) + `SWAP_COMMANDS.md` (exact
   commands the confirm would run; marked NOT RUN).
5. Memo `.omx/research/ddm_swp2_pr140_swap_packet_restage_move43_20260910.md`: verification table, compliance count vs
   pk1's 70/76, blocker table, boundaries. Serializer commit LAST, once (`REVIEW_GATE_OVERRIDE=1` allowed for non-.py); a
   Git-object write denial (rc 17) is NOT a stop — leave the bundle, report the rc, MAIN lands. Checkpoint as `ddm_swp2`.

## OPTIMAL FORM
- Reference form: pk1's staging (`.omx/research/ddm_pk1_pr140_swap_packet_staging_20260910.md`) and the packet that produced PR #140; scope delta: bytes = move 43. No mechanism delta.
- Provenance pins: packet memo sha d57ab4d55b3d7b0d…; pk1 memo sha d61b93ba6ec7460c…; archive sha 7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e;
  pointer commit 48109233e; CPU adjudication record path above (record its sha).

## Prior negatives accounted (operator 2026-08-15)
- hv1: five hand-typed errors in packet consequences — copy every sha/score from receipts; recompute the score.
- pk1's blockers: CPU (declaration + refusal receipt), import-hygiene (receiver change → operator decision, not applied),
  runtime-tree digest vs auth-eval manifest (two definitions coexist; record both values, cure owed to r9m), manifests.
- pr5/pr6: disclosure named assistants — carry the neutral draft.

Final message: what is staged, the recomputed score, compliance count + remaining blockers, the disclosure before/after,
the serializer rc, and the frontier line `composition S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600] (move 43)`.

<!-- # FORMALIZATION_PENDING: staging charter; no measured row of its own -->
