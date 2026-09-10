# PR #140 current public-state audit

`[no-triality] [p0-ledger-ok]` · `score_claim=false` · observed during the
2026-09-10 ddm_pr6 review through the unauthenticated public GitHub page.

Source: <https://github.com/commaai/comma_video_compression_challenge/pull/140>

- The visible page still showed PR #140 open, titled
  `semantic_joint_ctxmix (0.148)`, from `adpena:semantic_joint_ctxmix`, with two
  commits. Its body still identified the same 180,002-B AFR1 archive and score
  that pr5 recorded.
- The pre-existing public-body disclosure was still present: “I used coding
  agents (Claude as orchestrator of Codex subagents) extensively as research
  and engineering tools for the work behind this submission.” The checked-in
  posted-body snapshot has the same text at
  `experiments/results/ddm_fr2_final_review_20260903/pr_body_FINAL_POSTED.md:75`.
- Compared with pr5's public-page observation at
  `.omx/research/ddm_pr5_20260910/pr140_public_audit.md:3-22`, this review did
  not find a newly added AI-attribution sentence in the visible PR body or
  conversation. This is a bounded visible-page finding, not a claim about all
  GitHub objects.
- A `gh pr view` API cross-check was attempted and failed at the network edge
  with `error connecting to api.github.com`; the direct public page was the
  live verification surface. No public state was changed.
