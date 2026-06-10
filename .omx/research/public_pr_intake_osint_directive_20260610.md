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

## ADDENDUM (operator, 2026-06-10): COMPARATIVE TOOLING + FULL-STACK CENSUS

Beyond the score + anatomy, produce a side-by-side OURS-vs-THEIRS map so we know what to absorb vs
what we already out-tool. Two tables:

### Table A — their TOOLING vs ours (do they have analogs?)
For each of OUR instruments, find whether they have an equivalent (in their repo/writeup/fork) and who
is stronger:
- evaluator response surface / sensitivity map / Jacobian atlas (ours: #36, 600-pair, MLX)
- flip map / SegNet margin field (ours: #35/#51, 66,039 flips mapped, 91% margin<0.5)
- joint safe cone (seg-margin ∧ pose-Jacobian) (ours: #35)
- resize/YUV null-space / invisibility basis (ours: #47, 22.7% certified / 80.67% nullity)
- minimum-description preimage postprocessor (ours: #49, 10-19.5% free coded bytes)
- on-host exact mode tables + noise-floor tie law (ours: R3)
- per-pair/per-mode selector search (ours + theirs: the PR110 lineage)
- score-aware retraining harness w/ differentiable YUV6 + QAT-in-loop (ours: AFSR-1)
- composition algebra / V3 exact-ΔS waterfiller (ours)
- did they hand-tune what we DERIVE? (the leapfrog wedge — if yes, our derived version dominates)
Verdict per row: THEY-HAVE-STRONGER / PARITY / WE-HAVE-STRONGER / NEITHER. The WE-HAVE-STRONGER rows
are the leapfrog levers (our tool applied to their method).

### Table B — their FULL STACK, end to end (encoder → archive → decoder/inflate)
Reverse-engineer and document every stage:
- ENCODER / compress-time: what produces the archive? (trained renderer? selector search? codec? what
  architecture — HNeRV-family params/blocks? SNeRV? something new?) training recipe / curriculum /
  epochs / optimizer if stated.
- ARCHIVE GRAMMAR: every section, bytes, format, entropy coder per section (vs our decoder
  162,127B/latents 15,387B/selector 220B/sidecar 607B PR101-family).
- DECODER / inflate.py: the call graph, what each blob decodes to, the math of reconstruction, LOC +
  deps + runtime, whether it loads scorers (forbidden) / has sidecars.
- THE DELTA vs PR101/PR110 baseline: what stage did they change/add, and is the win in encoder
  (better-trained weights), grammar (better coding), or decoder (better reconstruction math)?
Cross-reference our reference_carrier_comparison + frontier anatomy memos so the census is apples-to-apples.

Deliverable: fold both tables into the intake memo. Return adds: the WE-HAVE-STRONGER tool rows + the
one stage of their stack that holds their win.
