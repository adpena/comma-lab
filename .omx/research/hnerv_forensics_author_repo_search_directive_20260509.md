# HNeRV forensics — operator directive: SEARCH AUTHOR REPOS + WRITEUPS (2026-05-09)

<!-- generated_at: 2026-05-09T05:50:00Z, from_state_hash: hnerv_forensics_author_search -->

## Operator directive (verbatim)

> "the PR author may cite in the report or in their repo or other repos the training scripts look everywhere"

## Critical insight

**The contest submission ships only the INFLATE runtime, NEVER the training script.** The training script — which contains the actual hyperparameters, loss formulation, optimizer schedule, EMA decay, conditioning vector strategy, and every other secret we want to extract — must be searched for OUTSIDE the submission archive itself.

Authors who land sub-0.20 scores typically have ONE of these public-by-default places:

1. **A sibling GitHub repo** by the same author (often named like `<contest>-training` / `<author>-comma-experiments` / `<paper-name>-codebase`)
2. **A linked arXiv preprint** describing the method
3. **A blog post** on Medium / Substack / personal site explaining the approach
4. **A YouTube or Twitch stream** showing the training loop
5. **A Discord / Slack discussion** in the comma community
6. **A Twitter / X thread** breaking down their submission
7. **The PR description body itself** (often has a writeup paragraph + repo link)
8. **The PR comments** (author may respond to maintainer questions with implementation details)
9. **Co-authors / acknowledgments** in the PR (other contributors may have their own repos)
10. **A LICENSE / NOTICE / CITATION file** in the submission that points to upstream code
11. **The author's pinned repos** on their GitHub profile (they curate these)
12. **Forks of the comma challenge** by the author with their training infra checked in
13. **A Hugging Face model / dataset card** uploaded by the author
14. **A WandB / TensorBoard / MLflow public run** linked from the PR
15. **A submission to a related challenge / paper** using the same architecture

## Required additions to your investigation (for subagent a30f2ade)

In your forensic dossier (`.omx/research/hnerv_leaderboard_binary_forensics_dossier_20260509.md`), add a NEW top-level section:

### §10: Training-script and writeup hunt — per author

For each PR's author (rem2, EthanYangTW, BradyMeighan, Quantizr/Jimmy at UCLA, and any other HNeRV-cluster author):

#### A. Inside the PR clone

- Read PR description body — capture every URL, every GitHub @mention, every paper citation
- Read README.md / docs/ / writeup.md inside the submission tree
- Read CITATION.cff / LICENSE / NOTICE / THIRD_PARTY_NOTICES.md
- Check `.github/` for issue templates referencing training infra
- Check git log for commit messages referencing external training code
- Check git remotes for any non-default remotes (sometimes points to private fork with training code)

#### B. PR thread harvest (via WebFetch if accessible)

- Fetch `https://github.com/commaai/comma_video_compression_challenge/pull/<NUMBER>` for each PR
- Capture all PR comments (especially:
  - Maintainer score-bot comments — `[contest-CUDA]` and `[contest-CPU]` numbers
  - Author replies to maintainer questions — often includes implementation details
  - Cross-PR discussion threads — authors often credit each other's tricks
- Look for any "I trained with X" / "my training code is at Y" / "see also Z" mentions
- Capture image attachments (training curves, ablation tables) — these often encode hyperparameters

#### C. Author GitHub profile + adjacent repos

- Fetch `https://github.com/<author>` (their profile)
- Fetch their pinned repos (curated; most likely to be relevant)
- Search their public repo list for keywords: `comma`, `nerv`, `hnerv`, `mnerv`, `video`, `codec`, `compression`, `hyperprior`, `quantizr`, `selfcomp`, `ballé`, `posenet`, `segnet`
- For any matching repo:
  - Fetch the README
  - Identify training scripts (`train*.py`, `experiments/`, `scripts/`, `notebooks/`)
  - Extract hyperparameters mentioned
  - Identify any custom loss function or optimizer schedule
- Fetch their gists (`https://api.github.com/users/<author>/gists`) — sometimes contains scratch training code

#### D. Search the public web

- Web search: `"<author>" comma video compression`
- Web search: `<author> NeRV training`
- Web search: `<author> HNeRV hyperprior`
- Web search: `comma video compression PR<NUMBER> training`
- ArXiv search: same author name
- Hugging Face search: same author
- Papers With Code search: same author + nerv/hnerv/comma

#### E. Discord / Slack / forum discussions

- Comma's Discord (#research / #comma-video-compression channels) — search for author name + "training"
- Reddit r/SelfDrivingCars or r/openpilot — search for the author or PR numbers
- Comma's openpilot Discord — often has tangential discussions

#### F. The "Quantizr" archetype

Quantizr (Jimmy, UCLA CSE/Neuro) achieved 0.33 with 88K params + FiLM + KL distill T=2.0. He's documented in our memory at `feedback_quantizr_competitive_intelligence_20260421` (or similar). His own assessment: "sub 0.30 is possible just by sweeping conv dims" — meaning HE knows there's headroom but stopped optimizing. His GitHub + UCLA page may have follow-up work. The HNeRV-cluster authors may have built directly on Quantizr's foundation.

Search specifically:
- `https://github.com/jimmyhliu` or similar Jimmy-at-UCLA candidates
- UCLA CSE / Neuro PhD students with comma references
- Any UCLA paper citing the comma challenge

#### G. Yousfi (challenge creator) connection

Yousfi was Fridrich's PhD student at Binghamton DDE Lab. The challenge IS inverse steganalysis. Authors who beat the challenge may explicitly cite:
- Fridrich's UNIWARD / HUGO / WOW
- Yousfi's `OneHotConv` / `alaska` / `comma10k-baseline`
- Filler's syndrome trellis coding (STC)

Check each PR for these citations — they often indicate the author's training-time secret IS one of these published methods.

## Output format (additions to dossier §10)

Per author, structured table:

| Field | Value |
|---|---|
| Author | rem2 (PR #103 silver) |
| GitHub profile URL | https://github.com/... |
| Pinned repos | [list] |
| Sibling training repo | [URL or "not found"] |
| Linked writeup | [URL or "not found"] |
| Discord/forum activity | [summary or "none found"] |
| External CITATION | [paper / blog / repo URL or "none"] |
| Specific hyperparameter cited anywhere | [e.g., "EMA 0.9995, AdamW lr=1e-4 cosine, 100K steps"] |
| Specific loss formulation cited anywhere | [e.g., "MSE + 0.1 * KL_distill_T=2.0"] |
| Specific arch detail cited anywhere | [e.g., "ConvNeXt encoder, depthwise decoder, FiLM at every block"] |
| Confidence training script is recoverable | [HIGH / MEDIUM / LOW] |
| Notes | [free text] |

## Coordination with sibling subagents

- a1a9359d (HNeRV retrospective) has LANDED — its memo is the higher-level lesson catalog. Your dossier provides the EVIDENCE the catalog cites.
- a8c01d31 (coherence council) has LANDED 3/3 CLEAN — its §3.A T7+T8+T11 sub-additivity finding means the seg-axis loss surrogate matters URGENTLY. If you find an author cites a SPECIFIC seg-axis loss (e.g., "we use Lovász on the EfficientNet-B2 logits"), surface it as a CRITICAL finding to the council's recommendation.
- a30f2ade (you): produce the evidence corpus that lets a1a9359d's catalog and a8c01d31's framework be CONCRETE.

## Priority

This is HIGHEST priority within your remaining scope. The "what specific training-time thing" question CANNOT be answered from the inflate runtime alone — most of the secret lives in code that NEVER ships with the submission. Author-repo search is the single biggest chance of recovering it.

## References

- Original forensic prompt: `.omx/research/hnerv_retrospective_user_clarification_20260509.md`
- HNeRV retrospective memo: `~/.claude/projects/.../feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md`
- Coherence council memo: `~/.claude/projects/.../feedback_grand_council_portfolio_coherence_journal_grade_20260509.md`
- Codex review findings: `.omx/research/codex_adversarial_review_findings_for_inflight_subagents_20260509.md`
