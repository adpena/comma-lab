# ddm_pr140 — semantic_joint_ctxmix SUBMITTED: contest PR #140 posted (2026-09-03)

Tokens: `[no-triality] [p0-ledger-ok]`

## The event

**PR #140 — https://github.com/commaai/comma_video_compression_challenge/pull/140** —
posted 2026-09-03 under the operator's explicit order chain:

1. 2026-08-17 ×2: "SUBMIT AFTER A+B+C EXHAUSTION" (task #1111, p0_swap_procedure gate).
2. 2026-09-03: "Okay, we are ready to submit it and I want you to do it for me without
   Claude or AI attribution."
3. 2026-09-03: "The version of the PR in our Google Drive is the final approved version
   unless codex has any final suggestions."
4. 2026-09-03 (the final one-line confirm): **"Rank 1 do nothing and submit as is.
   Rank 2-6 suggested changes are approved."** — this resolved the codex fr2c policy
   stop-condition (task #1363) by operator decision: submit with the transparent
   disclosure, maintainer adjudicates; and approved all five wording refinements.
5. 2026-09-03: "I need you to do that using gh cli" (the archive-asset upload) +
   "Make sure the archive is hosted and our related wording is correct."

## What shipped

- **PR branch:** `adpena:semantic_joint_ctxmix`, head `7f29354`, base upstream master
  `db52c5a9f0`. 40 files; MANIFEST verifies 39/39 in the shipped tree. Commits authored
  by the operator alone ("Alejandro Peña <adpena@gmail.com>") — no co-author trailers,
  no AI attribution anywhere in commits, branch, or body beyond the operator's own
  disclosure paragraph (codex rank-2 replacement form).
- **Archive delivery (PR #135 convention):** fork release tag `semantic_joint_ctxmix-afr1`
  (id 382143393), asset `archive.zip` uploaded via `gh release upload` on the operator's
  explicit instruction after the auto-mode classifier blocked the binary upload 4×.
  **Hosted-bytes fetchback verified:** SHA-256
  `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`, 180,002 bytes —
  byte-exact vs the retained afr1 archive. Download URL (in the PR body):
  https://github.com/adpena/comma_video_compression_challenge/releases/download/semantic_joint_ctxmix-afr1/archive.zip
- **PR body:** the operator's approved Google-Doc text verbatim
  (`pr_body_snapshot_v3_FINAL.md`, sha 7c7816a6…) with exactly six deltas:
  the delivery line realized as the release link (operator-amended convention), and the
  five operator-approved codex fr2c replacements (ranks 2–6: ownership wording ×2 ·
  pk4 n64-scope pose bullet · fresh-measurement TODO · entropy-coder scope ·
  reorder scope). Posted file: `experiments/results/ddm_fr2_final_review_20260903/pr_body_FINAL_POSTED.md`.
- **Rank-4 README twin:** `submissions/semantic_joint_ctxmix/README.md` TODO bullet
  aligned to the same fresh-measurement wording; manifest rehashed (39/39 both trees);
  pact commit `695a80f0bb`; PR-branch commit `7f29354`.
- **Title:** `semantic_joint_ctxmix (0.148)` — the name (score) convention of PR #133/#135;
  0.148 = the component-recomputed 0.14797617125559104 at 3 decimals; the body's first
  section explains the exact recompute per #877 discipline.

## Review provenance

- Opus fr2 review (R1–R10) → all cures verified present in v3 by codex fr2c.
- Codex fr2c final fresh-eyes (memo `ddm_fr2c_codex_fresh_eyes_pr_review_20260903.md`):
  ranks 7–10 VERIFIED-OK (arithmetic bit-exact, archive identity, current-tree T4 row,
  packet, lineage, priority dates); rank 1 policy BLOCKER → **operator-resolved:
  submit as-is** (cross-model disagreement recorded: Opus held the disclosure was the
  right posture; codex read the literal all-code clause as an eligibility stop);
  ranks 2–6 → applied verbatim.
- Compliance: frozen pq12 adjudication (80 GREEN / 7 RED, receipts) binds these exact
  bytes; fresh strict re-run this session showed zero NEW defect classes (19 REDs all
  staging-input/known-survivor categories; parser-schema note: per-check key is
  `passed`, not `ok`).

## State changes

- Task #1111 (SUBMIT) → executed; #1363 (policy decision) → resolved by operator.
- p0_swap_procedure_no_push_without_confirm: the gate FIRED correctly — nothing
  published until the operator's one-line confirm; then push/host/post executed.
- canonical_frontier_pointer `submitted_pr_number_for_current_frontier`: now PR #140
  (pointer-file refresh rides the next pointer touch; this memo is the record).
- Post-submission roadmap already tracked: #1390 (--device / full-pipeline / per-clip),
  #1389 (gen-8 public-minimal packet), #1381 (Drive package email leg).

Own-vehicle frontier (unchanged by publication): **S 0.14797617125559104 @ 180,002 B
[contest-CUDA T4 n600]**, archive sha cbb8d928…d405bf25.
