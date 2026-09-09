# PR #140 public-state audit

`[no-triality] [p0-ledger-ok]` · `score_claim=false` · observed
2026-09-09T23:20Z through the unauthenticated public GitHub page.

Source: <https://github.com/commaai/comma_video_compression_challenge/pull/140>

- The page showed PR #140 open, titled `semantic_joint_ctxmix (0.148)`, from
  `adpena:semantic_joint_ctxmix`, with two commits.
- The public body still identified archive SHA-256
  `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`,
  180,002 B, and recomputed score `0.14797617125559104`.
- The public body included this sentence: “I used coding agents (Claude as
  orchestrator of Codex subagents) extensively as research and engineering
  tools for the work behind this submission.” The checked-in posted-body
  snapshot carries the same sentence at
  `experiments/results/ddm_fr2_final_review_20260903/pr_body_FINAL_POSTED.md:75`.
- Therefore the broad claim “no public text carries an AI attribution” is
  false. The narrower local receipt says the two commits have no co-author or
  AI-attribution trailer
  (`.omx/research/ddm_pr140_submission_posted_20260903.md:24-28`); that does not
  remove the explicit public-body disclosure.
- No public state was changed. A `gh api` cross-check was attempted but network
  access to `api.github.com` was unavailable; the direct public HTML page was
  available and was the live verification surface.
