# ddm_pk1 — stage the PR #140 swap packet for move 40 (prepare ONLY; nothing publishes) — charter, MAIN 2026-09-10

## Why
PR #140 on the contest repo carries the afr1 bytes (pointer move 23, S 0.14797617). The submittable
frontier is move 40 (S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600], archive sha
986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857), seventeen moves ahead. The operator's
standing rule (`p0_swap_procedure_no_push_without_confirm_20260817`) is that NO publish/push/PR update
happens without the operator's one-line confirm. This arm makes that confirm a one-command act by staging
everything beside the live tree, and touches nothing public.

## Hard boundaries (binding)
- Do NOT edit `submissions/semantic_joint_ctxmix/` (the live PR tree), `upstream/`, or any file under
  `/Volumes/*/pact/ddm_sj1_compose39_price/` (sealed custody). Stage under a NEW directory
  `submissions/_staging_move40_pr140_swap/` (gitignored? check `.gitignore`; if not ignored, add the
  staged tree to a dated `.omx/research/ddm_pk1_20260910/` manifest and commit only the manifest + PR
  body text, not the binaries).
- Do NOT run `gh pr`, `git push`, any hosting upload, or any Modal/GPU dispatch. No score claim beyond the
  retained T4 receipt.
- Receiver code of moves 41 and tc4 must not appear anywhere in the packet (rule-118 retraction).
- No Claude/AI attribution in any public-facing text you draft. The PR body's existing AI-disclosure
  sentence (pr5/pr6 found it names specific assistants) must be rewritten to a neutral disclosure of
  tooling without naming assistants; put the exact before/after in your memo for the operator to approve.

## Inputs
- The move 40 candidate tree: `/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/candidate/candidate_runtime`
  (archive.zip, inflate.py, inflate.sh, runtime/, cpr1/, README.md, MANIFEST.sha256) — read-only source.
- Its T4 receipt: `/Volumes/APDataStore/pact/ddm_sj1_t4_compose39_rp1_union_20260910/MODAL_REMOTE_RESULT.json`
  (contest_auth_eval.json inside: d_seg, d_pose, bytes, inflate 990.054 s, evaluate 45.23 s).
- The move 40 packet: `.omx/research/ddm_frontier_pointer_move_40_20260910.md`; the canonical pointer
  `.omx/state/canonical_frontier_pointer.json`; the PR #140 memos (`grep -rl "PR #140\|PR140" .omx/research | head`),
  especially the disclosure findings in `ddm_pr5_*`/`ddm_pr6_*` memos.
- `docs/submission_template.md`, `scripts/pre_submission_compliance_check.py --help`,
  `tools/pointer_move_packet.py --help` (the packet apparatus writes consequences; you do not retype shas).

## Deliverable (all staged, none published)
1. `submissions/_staging_move40_pr140_swap/` = a byte-exact copy of the candidate tree's shippable members
   (archive.zip, inflate.sh, inflate.py, runtime/, cpr1/, README.md, MANIFEST.sha256) with sha256 of every
   file recorded; archive sha and size must equal the pointer's (986d536b…, 180,233 B).
2. `report.txt` regenerated from the T4 receipt in the contest's format (see the live PR tree's report.txt
   for the shape), with the score recomputed from components: 100·d_seg + sqrt(10·d_pose) + 25·bytes/37,545,489
   — it must equal 0.13763861019288715 to the last digit; if it does not, STOP and report the discrepancy.
3. `scripts/pre_submission_compliance_check.py --contest-final --strict` run against the staged tree with
   `--expected-archive-sha256` / `--expected-archive-size-bytes` and the auth-eval JSON; its full output
   retained; if it refuses, record the exact refusal (do not waive anything).
4. `PR_BODY.md`: the updated PR #140 body (declares `linux-nvidia-t4`; move 40's components; the
   neutral disclosure sentence), plus `SWAP_COMMANDS.md`: the exact commands the operator's confirm would
   execute (copy staged tree over the live tree, commit via the serializer, `gh pr` update), clearly
   marked NOT RUN.
5. Memo `.omx/research/ddm_pk1_pr140_swap_packet_staging_20260910.md` with a verification table and
   every boundary; commit memo + manifests via the serializer (`REVIEW_GATE_OVERRIDE=1` allowed for
   .md/.json/.txt; no co-author trailer; tags `[no-triality] [p0-ledger-ok]`).

## OPTIMAL FORM
- Reference form: the packet that produced PR #140 itself (grep the memos for its staging steps) and
  `docs/submission_template.md`. Scope delta: staging only — the publish step is the operator's.
- Provenance pins: archive sha 986d536b…, T4 receipt sha cd6d5ef5243e26fa1868100a2bbb34efe0d6446b4a7fb9bd1eab7f309f5d3d00,
  pointer commit 4aa86522c.

## Prior negatives accounted (operator 2026-08-15)
- hv1 found five hand-typed errors in packet consequences: never retype a sha or a score; copy from the
  receipts and recompute the score from components.
- pr5/pr6: the PR body's disclosure sentence named assistants — fix it, do not carry it forward.
- Moves 41/tc4 receiver code is retracted content — grep the staged tree for `tc3`/`tc4`/lane-predictor
  symbols and prove absence.

Checkpoint as `ddm_pk1` every ~10 tool uses. Final message: what is staged, the recomputed score, the
compliance verdict, the before/after disclosure sentence, and the frontier line
`composition S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600] (move 40)` unchanged.
