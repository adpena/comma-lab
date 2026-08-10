# ddm_pi135 — PR #135 "semantic-pose-HPAC_CPR1_polished" full intake (THE NEW BAR, 0.162)

## Mission

PR #135 (https://github.com/commaai/comma_video_compression_challenge/pull/135)
is the new official rank-1 at **0.162** — PR130's author, polished, the SAME
lineage our live vehicle builds on. Our freshest exact row is
S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4], sha
`f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45` — we are
**+0.007599 ABOVE the new bar** (≈11,414 B rate-equivalent). Your job: full
public-frontier intake per CLAUDE.md "Public frontier watch and intake" so MAIN
can re-derive the aim against PR135 instead of PR130.

## Deliverables (ranked)

1. **Bit-level anatomy** of PR135's archive: exact bytes, sha256, section map
   diffed against PR130's 191,052 B (0491d5df…) — WHERE did the −0.010 S come
   from? Split it into seg / pose / rate deltas from the bot comments if
   present; else from the archive structure + code diff.
2. **Code diff PR130 → PR135**: every changed file, mechanism-level read. Rank
   each change {TRANSFERS-TO-OUR-BASE / ALREADY-OURS (e.g., did they adopt an
   ANS/constriction recode like our lc2? our lc2 is 187,226 B — did they beat
   it on the same axis?) / NEW-MECHANISM / COSMETIC}.
3. **Both-ways gap table**: their 0.162 vs our 0.169599 — which of their deltas
   compose with OUR landed levers (lc2 ANS recode, pk2 pose-carrier attack
   23,384 B, #869 adaptive token map −113,555 B projected, hp1 learned AR
   prior)? Name the composed best-case arithmetic vs 0.162.
4. **Ranked lessons table** with named consumers (#984 composed campaign,
   #995 roadmap, #1007 successor row).

## Ground rules (binding)

- Intake clone is READ-ONLY: detached clone under
  /Volumes/VertigoDataTier/pact/pr135_intake_20260810/ (SSD tier), NEVER the
  shared worktree; no in-place edits; certify-or-block on any cleanup.
- PR130 off-the-shelf grant (operator 2026-08-06) extends to LESSONS from this
  lineage; honesty-half unchanged: borrowed_substrate_accounting on every
  transfer claim.
- Claimed public scores stay `external` until exact replay; do NOT write any
  score claim. Axis labels on every number.
- Every byte figure MEASURED from the downloaded archive, never inferred.
- Full research authority (online, GitHub, papers) per the standing sol-ultra
  clause; internal leverage authority applies (our own modules are
  off-the-shelf).
- Durable memo: .omx/research/ddm_pi135_pr135_intake_20260810.md + commit via
  tools/subagent_commit_serializer.py (post-edit --expected-content-sha256,
  tags [no-triality] [p0-ledger-ok]). Checkpoint per the subagent protocol.

## OPTIMAL FORM

Reference form: the pi1 PR86+PR130 intake (`.omx/research/` pi1 memo lineage,
receipts commit fb34d3c6aae696a2f5d1070e5adf66ac69713d9d era) — bit-level
anatomy + L1–L13 binding read + both-ways gap table. This charter is
SCOPE-equal to pi1 on ONE PR (no mechanism reduction). Provenance pins: our
row sha `f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45`,
PR130 bar archive sha
`0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.
PRIOR-LAW PREDICTION: from the lineage's history (CPR1 = carriage + HPAC), the
most likely −0.010 S source is a rate-side recode of the token carriage (the
axis our lc2 already attacked, −3,826 B) plus a pose-section trim (pk2's axis,
23,384 B available). If the diff shows instead a SEG-side change, that is the
surprising finding — flag it loudly.
