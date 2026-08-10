# ddm_pi136 — reverse engineer ALL un-intaken public PRs (operator 2026-08-10 "reverse engineer all")

## Mission

Our public-frontier intake stops at PR #132 (`.omx/research/public_pr129_132_intake_20260725.md`).
The live leaderboard snapshot (pointer refresh 2026-08-10T16:53Z) shows PR #135
at rank 1 (0.162). The sister arm ddm_pi135 owns PR135 DEPTH. You own BREADTH:
every public PR to commaai/comma_video_compression_challenge that we have never
intaken — at minimum PR #133 and PR #134, plus any PR numbered >135 or any
non-PR leaderboard row that appeared since 2026-07-25. Reverse engineer each per
the CLAUDE.md "Public frontier watch and intake" default order.

## Deliverables (per PR)

1. Identity: number, title, author, head SHA, claimed score(s) with axis, bot
   comments (CPU + CUDA rows if present).
2. Archive: exact bytes, sha256, section map (bit-level anatomy where the
   archive is downloadable).
3. Mechanism read: what the vehicle IS; diff vs its nearest ancestor
   (PR130 lineage? PR128 rhnerv? new family?).
4. Transfer ranking vs OUR live base (lc2, 187,226 B, S 0.16959899569230852
   [contest-CUDA]): {TRANSFERS / ALREADY-OURS / NEW-MECHANISM / COSMETIC /
   DOMINATED}. Every transfer claim carries borrowed_substrate_accounting.
5. One ranked table across ALL intaken PRs, consumers named
   (#984 composed campaign, #995 roadmap, #1009 pi135 depth).

## Ground rules (binding)

- Detached READ-ONLY clones under /Volumes/VertigoDataTier/pact/pr_breadth_intake_20260810/
  (SSD tier); never the shared worktree; no in-place edits; certify-or-block.
- Claimed public scores stay `external` until exact replay — no score claims;
  axis labels on every number; every byte figure MEASURED from downloaded bytes.
- Full research authority (online/GitHub/papers) per the standing sol-ultra
  clause; internal-leverage authority applies.
- De-dup vs ddm_pi135: PR135 rows are pi135's — link, do not duplicate.
- Durable memo `.omx/research/ddm_pi136_leaderboard_breadth_intake_20260810.md`
  + serializer commit (post-edit --expected-content-sha256, tags
  [no-triality] [p0-ledger-ok]). Checkpoint per the subagent protocol.

## OPTIMAL FORM

Reference form: `public_pr129_132_intake_20260725.md` (the 07-25 breadth pass)
+ pi1's bit-level standard. SCOPE-equal per PR, no mechanism reduction.
Provenance pins: PR130 bar archive sha
0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd; our anchor
sha f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45.
PRIOR-LAW PREDICTION: PR133/134 are the polish steps BETWEEN PR130 (0.172) and
PR135 (0.162) from the same author lineage — expect an incremental rate-recode
ladder; if either is a DIFFERENT family beating 0.19, that is the surprising
finding — flag it loudly.
