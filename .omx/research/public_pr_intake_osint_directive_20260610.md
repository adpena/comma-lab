# OSINT WIDENING DIRECTIVE → public_pr_frontier_beat_intake_20260610 (operator, 2026-06-10)

The author SHARES A LOT in the PR — and there is more beyond the diff. Expand intake step 1 to a full
open-source-intelligence sweep on the author + method BEFORE concluding the anatomy. The writeup often
states the method INTENT that the bytes alone obscure.

Sources to harvest (record each in the intake ledger with URL + retrieved-at + relevance):
1. **The PR body + ALL comments** verbatim (gh pr view --json body,comments) — authors routinely
   describe the mechanism, the curriculum, the byte budget, the "what I tried that failed."
2. **Any linked writeup/report/blog/gist/notebook** in the PR or comments — fetch in full.
3. **The author's GitHub**: `gh api users/<login>` + their repos (`gh repo list <login>`), pinned repos,
   the fork of the contest repo (their branch history `gh api repos/<login>/<fork>/commits` often shows
   the iteration path — failed attempts, parameter sweeps, the order they found things), any
   contest-related repo, READMEs, commit messages.
4. **Their other PRs/issues** on the contest repo (`gh pr list --author <login>` / `gh issue list`) —
   earlier submissions show the trajectory to this win.
5. **Web presence**: WebSearch the author handle + "comma video compression" / the method names they
   use; fetch any X/twitter thread, LinkedIn, personal site, arXiv, that describes the approach.
6. **The contest discussion**: any leaderboard thread / discussion where they explain the gain.

Synthesis additions:
- Build a METHOD-INTENT vs BYTES table: what they SAY they did (writeup) cross-checked against what the
  bytes PROVE they did (anatomy) — discrepancies are either modesty (more to absorb) or
  misdirection/error (verify, don't trust the claim).
- The iteration path (from fork commits) tells us what they ALREADY tried and abandoned — we skip those
  dead ends and start from their frontier.
- Keep claimed scores `external` until our paired exact replay; the writeup informs the leapfrog, the
  replay sets authority.

Everything else in the original intake prompt stands. This only widens the information-gathering net.
