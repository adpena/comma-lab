# CHARTER ddm_ps1 — PR #140 update packet PREP for the fs1 bytes (24th pointer move) — STAGING ONLY, NO PUBLISH

Tokens: `[no-triality] [p0-ledger-ok]`. Owner: Opus arm (operator-authorized "opus subagents for now"). Spawned 2026-09-04 ~17:55Z.

## PRIOR-LAW PREDICTION (owed line)
The frozen packet `submissions/semantic_joint_ctxmix/` reproduces afr1 (180,002 B, sha cbb8d928…) from a pinned base via five
lossless stages. fs1's archive (180,022 B, sha 50fcaf1a…) differs from afr1 by ONE 14-byte selector blob + container; PREDICTION:
a sixth lossless stage (the fs1 selector ENCODER applied to the shipped decode: `encode(select(decode(afr1)))`) rebuilds fs1's
archive bit-exactly from afr1's, so the packet keeps its "refuses unless the rebuilt archive matches its pinned SHA-256" contract
with a new pin. Falsifier: the rebuilt archive's sha ≠ 50fcaf1ac3c8504abdf3e0daff7c5bce32104f19d8de4a7ba207816f32e708cf.

## Objective
Prepare — in a STAGING tree, never in the live PR tree — everything the operator's one-line confirm would need to update PR #140
to the fs1 bytes. p0_swap_procedure binds: NO push, NO release upload, NO PR edit, NO comment. Deliver a ready packet + a diff
summary + a compliance dry-run + a one-paragraph PR-body delta the operator can approve verbatim.

## Inputs (pinned)
- Live PR tree (READ-ONLY): `submissions/semantic_joint_ctxmix/` (MANIFEST 39/39; README pins; compress.py five stages).
- fs1 runtime tree: `/Volumes/VertigoDataTier/pact/ddm_fs1_frame0_selector/fire_runtime_B_byte_optimal_101/` (the g8v1 tree with
  two `inflate.py` pin lines; tree sha fbf4aaf436aa02814d0558bfbc2bf4307502bdac49a7616b66bcfa31b44ca43c) + its `archive.zip`
  (sha 50fcaf1a…708cf, 180,022 B). Second copy: `/Volumes/APDataStore/pact/ddm_fs1_frame0_selector/custody_pointer24/`.
- Exact row: `/Volumes/APDataStore/pact/ddm_fs1_frame0_selector/t4_buy_20260904/MODAL_REMOTE_RESULT.json` (sha 4bcecd01…);
  artifacts include `report.txt` (verbatim evaluator output: pose 0.00000617, seg 0.00020139, 180,022 bytes, "Final score … = 0.15";
  recomputed 0.14786319521362173); inflate 492.7 s + evaluate 38.2 s on T4.
- fs1 encoder + scan: commits 7ec320551 / 7f64bd379 (read `git show --stat`); seal
  `/Volumes/VertigoDataTier/pact/ddm_fs1_frame0_selector/SEAL_fs1_frame0_selector_byte_optimal_contest_cuda.json`.
- Pointer memo: `.omx/research/ddm_fs1_pointer_move_24_20260904.md`. PR record: `.omx/research/ddm_pr140_submission_posted_20260903.md`,
  posted body `experiments/results/ddm_fr2_final_review_20260903/pr_body_FINAL_POSTED.md`.

## Deliverables (all in a staging tree `/Volumes/APDataStore/pact/ddm_ps1_pr140_update_prep/` + small diffs in git)
1. **Stage 6 in compress.py (staged copy):** the selector re-selection as a lossless replay stage — consumes the afr1 archive
   (pinned df7fd266… base → five stages → afr1 cbb8d928…) and emits fs1's archive; refuses unless sha == 50fcaf1a… exactly. The 21-pair
   selection is VIDEO-DERIVED content: it must be carried as data the stage reads (the 14-byte blob or the 21 (pair,k) choices),
   never hidden as code constants disguised as an algorithm beyond what the archive itself stores (rule 118). Determinism repeat.
2. **Runtime:** the staged tree = fs1's runtime tree (two pin lines). Prove the diff vs the live PR tree is EXACTLY those two lines
   in `inflate.py` (fs1 proved it; re-prove in the staging tree). MANIFEST rehashed for the staged tree.
3. **README.md / report.txt / PR-body delta:** numbers only — 180,022 B, sha 50fcaf1a…, S 0.14786319521362173, pose 0.00000617,
   the row's provenance (T4, 492.7 s + 38.2 s), the sixth stage named in one bullet ("a per-pair frame-0 selector re-selection,
   +20 bytes, pose 6.37e-6 → 6.17e-6, segmentation output unchanged"). Disclosure paragraph UNCHANGED. Lineage/credits UNCHANGED.
   Keep the PR title convention `semantic_joint_ctxmix (0.148)` (0.14786 → still 0.148 at 3 dp — state that explicitly).
4. **Compliance dry-run:** `scripts/pre_submission_compliance_check.py --contest-final --strict` against the staged packet with
   `--expected-archive-sha256 50fcaf1a… --expected-archive-size-bytes 180022` + the harvested auth-eval JSON; record GREEN/RED
   counts vs the frozen pq12 adjudication (80/7); any NEW RED class is a blocker you report, not a thing you waive.
5. **Release-asset plan (not executed):** the exact `gh release create/upload` + `gh pr edit` commands the operator would run, with
   the fetchback verification step, written to `RELEASE_PLAN.md` in the staging dir. NOTHING RUNS.
6. Memo `.omx/research/ddm_ps1_pr140_update_packet_prep_20260904.md` + task registration via `tools/register_task.py` for the
   operator decision gate; hot-state line proposal in the memo's last section.

## OPTIMAL FORM
Reference form = the pq-series packet prep (pq1…pq12) that produced the live PR tree: full-manifest custody, compress.py replay
with sha refusal, compliance --contest-final. No scope reduction: all six deliverables. Mechanism reductions: none. Provenance
pins: every input above by path+sha.

## Rules that bind
NO-FAKE (stage 6 must actually rebuild the bytes; sha-refuse); ALWAYS KEEP THE PAYLOAD (retain every rebuilt archive with sha);
upstream/ READ-ONLY; the live PR tree READ-ONLY; commits ONLY via `tools/subagent_commit_serializer.py --message … --files … 
--expected-content-sha256 <file>=<post-edit sha>` with `[no-triality] [p0-ledger-ok]`; NO co-author trailers; .py files: two
review-gate passes (`tools/review_tracker.py mark-file`); checkpoints every 10 tool uses via `tools/subagent_checkpoint.py`;
`docs/operating_manual_craft_handoff.md` binds; no `/tmp` in any persisted evidence; end with the frontier line
`fs1 S 0.14786319521362173 @ 180,022 B [contest-CUDA T4 n600]`.
