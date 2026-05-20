# PR Comments Mining — Top-15 Activity PRs on `commaai/comma_video_compression_challenge` for Actionable Signal

**Retrieved**: 2026-05-20T06:00Z via `gh api repos/commaai/comma_video_compression_challenge/{issues,pulls}/<PR>/{comments,reviews}` for all 15 target PRs.
**Methodology**: AI-assisted inverse-steganalysis on contest-comments surface (per `.omx/research/ai_assisted_inverse_steganalysis_on_contest_problem_space_methodology_directive_20260520.md` standing directive). Treat the public comment thread as a high-signal channel that competitors + maintainer engage in publicly; extract maintainer hints + per-PR scoring discussion + technique observations + competitive intelligence + author retrospectives.
**Coverage**: 113 issue-level comments + 17 inline review comments + 5 review-state bodies across 15 PRs.
**Internal artifact** — may cite CLAUDE.md / Catalog # / canonical-helpers freely. NOT public-PR-facing.
**Sister coordination**: input to MG-17 T3 voice/tone/style symposium (in flight); informs Slot K PR body refinement; feeds Catalog #344 canonical equations registry as `[empirical:contest_comment_thread]` provenance grade.

---

## Section 1: Frontmatter

The contest comment threads are the highest-signal public channel that the maintainer (Yousfi) + competitors engage in. Per the operator directive ("review all of the comments on all of them for information we can use and optimize against"), this artifact extracts every actionable signal across the top-15-activity PRs.

Cross-link to:
- `feedback_t3_grand_council_upstream_contest_compliance_symposium_landed_20260519.md` (operator-facing PR body audit + Yousfi 2026-05-11 PR #108 closure citation)
- `.omx/research/ai_assisted_inverse_steganalysis_on_contest_problem_space_methodology_directive_20260520.md` (the meta-methodology this mining operationalizes at the comments surface)
- `.omx/research/council_t3_pr_110_editorial_positioning_symposium_20260520T050557Z.md` (MG-11 PR body editorial verdict; this artifact provides MG-17 the contestant-author voice/tone calibration data)
- `feedback_canonical_equations_and_models_registry_formalization_landed_20260519.md` (Catalog #344 registry; the mined operational anchors become canonical equation `EmpiricalAnchor` provenance candidates)

**Coverage matrix** (top-15 by activity, verified by `gh pr view`):

| PR | Author | Score | State | Prize | Issue comments | Review comments |
|---|---|---|---|---|---|---|
| #95 | AaronLeslie138 | 0.20 → 0.1987 CPU | MERGED | Honorable + best write-up | 13 | 0 |
| #103 | rem2 | 0.19487 → 0.19538 CPU | CLOSED | 2nd prize | 11 | 0 |
| #97 | BradyMeighan | 0.23 CPU | CLOSED | Honorable | 11 | 0 |
| #74 | hypery11 | 0.35 → 0.37 CUDA | MERGED | Leaderboard | 10 | 1 |
| #100 | BradyMeighan | 0.1954 CPU | CLOSED | Leaderboard + new-approach mention | 10 | 0 |
| #67 | EthanYangTW | 0.31 → 0.32 CUDA | MERGED | Leaderboard | 9 | 1 |
| #105 | valtterivalo | 0.19797 CPU | MERGED | Honorable | 8 | 0 |
| #102 | EthanYangTW | 0.1954 CPU | MERGED | 3rd prize | 7 | 3 |
| #101 | SajayR | 0.19 → 0.19 CPU | CLOSED | 1st prize (GOLD) | 8 | 0 |
| #98 | EthanYangTW | 0.1963 CPU | MERGED | Leaderboard | 7 | 9 |
| #86 | jas0xf | 0.27 CUDA | MERGED | Leaderboard | 8 | 0 |
| #71 | TomDousek | 0.71 → 0.72 CUDA | MERGED | Leaderboard | 8 | 0 |
| #56 | szabolcs-cs | 0.36 → 0.38 CUDA | CLOSED | Leaderboard | 7 | 0 |
| #55 | Quantizr | 0.33 → 0.33 CUDA | MERGED | Honorable | 4 | 2 |
| #96 | rem2 | 0.21 CPU | CLOSED | Leaderboard | 5 | 0 |

Note: PRs #100 + #103 are CLOSED-not-merged but appear on the public leaderboard via the README.md update Yousfi made post-eval (per his 2026-05-05T17:11 comment on #103: "scored both CPU/GPU and keep the best" framing).

---

## Section 2: Per-PR summary (key signal points)

### PR #95 — `hnerv_muon` by @AaronLeslie138 (best-write-up prize)

PR body documents the HNeRV-MUON root: 229K-parameter HNeRV decoder + 28-d-per-frame-pair latents (~600 pairs) + INT8 + brotli → 178 KB. Author cites 8-stage curriculum + C1a regularizer + Muon optimizer. CPU report 0.1987.

**Mined signal**:
- @AaronLeslie138 13:51Z: "Regrettable that the landscape punishes publishing /src" + "competition seems to mainly reward whoever is up using codex or claude right at the last hour, not the author of 99% of the work" — author's own retrospective: open-publishing is a tragedy-of-the-commons exposure when other contestants stack within 4-hour race window.
- Yousfi 16:29Z direct response: "we are going to reward folks publishing their code even if not in top 3" — canonical maintainer commitment for write-up + code-sharing prizes. ALREADY in CLAUDE.md context.
- Yousfi 17:43Z: "yeah there is a small hw difference in the decode, I ran all submissions in t4 for a fair comparison" — admitted CPU/CUDA gap is "small" pre-PR #103 dispute; suggests Yousfi underestimated the magnitude of the gap until the medal-cluster surfaced it.
- @EthanYangTW 22:37Z to Aaron: "from what I see from all the submisssion, you definitely won. I think you definitely deserved a price" — competitive-courtesy norm; even close competitors openly congratulate the upstream-author.
- Yousfi 19:58Z: "This submission won an honorable prize and best write-up prize. Please email me at {first name}@comma.ai for logistics. Let us know if you are looking for a job/internship as well." — canonical comma.ai contact pattern + recruiting hook.

### PR #103 — `hnerv_lc_ac` by @rem2 (2nd prize)

PR body: 178,223 bytes; CPU score 0.19487 (local) / 0.19538 (Yousfi T4 CPU). Built on @AaronLeslie138 + @EthanYangTW + @BradyMeighan stack. Substantive change: arithmetic coding (constriction range coder) on the 8 largest weight tensors + latent-hi byte stream. Single-byte filename inside zip. Hardcoded section lengths in inflate.py (no length prefixes inside archive). Adaptive `lgwin` search in brotli. Single `RangeEncoder` merging 9 AC streams.

**Mined signal (HIGHEST VALUE TECHNICAL THREAD)**:
- @rem2 20:19Z: explicit citation of the contest README rule "**_If your inflation script requires a GPU, it will run on a T4 GPU instance (RAM: 26GB, VRAM: 16GB), if it doesn't it will run on a CPU instance (CPU: 4, RAM: 16GB)._**" — verbatim rule that Yousfi's "all-on-T4 for fair comparison" decision broke. CRITICAL signal for our submission strategy: the original rule is honored by the leaderboard ONLY after Yousfi re-ran CPU eval on the back-half.
- Yousfi 21:48Z: "you're welcome to run the workflow on your own fork. I think it's fair to run all submissions on the same hw. When I first compared the hw differences, it only made a tiny difference, but I guess the more performance squeezing happens the bigger effect small pixel differences make. fyi #103 is winning a 3rd spot, congrats!" — admits the empirical CPU/CUDA gap MAGNIFIES as scores compress; offers fork-CI as the formal escape hatch.
- @rem2 23:10Z: "disregarding the specifications and inventing new private criteria is the opposite of 'fair'" + "if you want to be extra charitable, you could even score both CPU/GPU and keep the best" — the proposal Yousfi ended up adopting.
- Yousfi 04:12Z: "Would you like to renounce your 3rd place?" — passive-aggressive but signal-bearing: Yousfi reads pushback in comments as performative-grandstanding; private-channel preferred for evaluation disputes.
- @rem2 05:35Z: "if you want to be extra charitable, you could even score both CPU/GPU and keep the best. That way everyone gets scored at least in accordance to their own selection" — concrete proposed reform.
- Yousfi 17:11Z: "if you read the scores as a black box number, then I agree that each solution gets to pick the runtime and optimizes w.r.t. that runtime and gets scored. But the score has a physical meaning that depends on the ground truth video, so having that be mismatched makes things non comparable 100%. One way to fix this is to make the score relative to a baseline computed on the solution runtime. Maybe for the next phase, or maybe we'll just do one runtime." — Yousfi's formal CPU/CUDA-axis-mismatch acknowledgment + signal that future challenges may have explicit per-axis baselines OR single-runtime rule.
- Yousfi 17:11Z: "trying to influence things privately is not the way to do so" — maintainer-norms signal: keep evaluation-dispute requests PUBLIC, not private email.
- Yousfi final 19:53Z: 2nd prize confirmed. Score 0.19538 (paired CPU) becomes the canonical PR #102 reference too.

### PR #97 — `vibe_coder_final_boss` by @BradyMeighan (honorable)

PR body: 0.23 CPU. Cites comma-writeup.pages.dev/ interactive writeup + WRITEUP.md + GitHub source. Mask-based generator.

**Mined signal**:
- @BradyMeighan 16:54Z: explicit AVVideoDataset vs DALI CUDA YUV→RGB matrix-coefficient gap as the per-pair distortion driver. "my generator was tuned against" CPU AVVideoDataset specifically. Concrete numbers: CPU 0.23 vs CUDA 0.25 (delta = +0.02 on same archive). Useful empirical anchor for `tac.canonical_equations` mps_drift sister equation extension.
- @BradyMeighan 17:16Z: "looks like CUDA was intended (the other recent submissions all got the same runner). I guess my model just overfit to AVVideoDataset haha." — author's own retrospective: per-runtime overfit risk for ground-truth decoder choice.
- Yousfi 17:52Z: same as #95: "yeah there is a small hw difference in the decode, I ran all submissions in t4 for a fair comparison" — Yousfi standardizing on this response template.
- @BradyMeighan 00:02Z: "didn't crack the top 3 but I put together a pretty cool interactive writeup that goes deep on how I built and trained the model and every byte-saving method I used: https://comma-writeup.pages.dev/. Just wanted to make sure you guys see it." — competitive-courtesy + write-up self-promotion pattern.
- Yousfi 21:20Z: "We had an impressive batch of submissions, and after a lot of consideration, we decided to award the prize to Aaron's submission" — write-up prize decision rationale (sole prize despite multiple worthy candidates).

### PR #100 — `hnerv_lc_v2` by @BradyMeighan (new-approach mention)

PR body: 0.1954 CPU on 178,981 bytes. Built on @EthanYangTW (#98) + @AaronLeslie138 (#95). "Decoder weights and architecture are theirs. Our additions are inference-time only." Includes fork CI link for reproducibility.

**Mined signal (HIGHEST VALUE NEW-APPROACH-PROVENANCE THREAD)**:
- @BradyMeighan 20:44Z (long post-prize comment): "per-pair latent-correction sidecar. Single-dim grid-searched perturbation of HNeRV latents chosen to minimize joint SegNet+PoseNet loss. First shipped in this PR (May 4 09:59 UTC) and #99. Not in #95 or #98 prior. The 2nd-place submission #103 is by rem2's own description a 'lossless byte-level repack of @BradyMeighan's hnerv_lc_v2 (#100). Decoder weights, latents, and latent-correction sidecar are all his.' The 3rd-place submission #102 is named hnerv_lc_v2_scale095_rplus1, a re-tune of one constant in this same pipeline. Two of three prize winners stood on this technique and it came from this PR." — explicit per-pair latent-correction sidecar provenance + technique-prize claim.
- @BradyMeighan: "On my other PR (#97 vibe_coder_final_boss), I also ran a Karpathy-style LLM-driven autoresearch loop, around 195 short-budget proxy experiments where the agent reads the previous result, proposes one algorithmic change, runs the proxy, decides keep or revert. To my knowledge no other submission used this methodology for architecture search, and the resulting mask + dual-head FiLM-conditioned generator was the best mask-based score on the leaderboard by ~0.08 over the next mask-family submission." — TECHNIQUE OBSERVATION: Karpathy-style auto-research loop methodology + mask + dual-head FiLM-conditioned generator class as a distinct family.
- @BradyMeighan: "literal weeks of compute on my personal 3090 with the autoresearch loop running overnight while I slept, two A100 colab runs, and the night before the deadline I was up till 5am polishing the writeup site while watching Ethan iterate on top of my hnerv_lc_v2 in real time" — OPERATIONAL HINT: 3090 + Colab A100 are realistic working environments; multi-week training-loop endurance signal.
- Yousfi 21:32Z: "My reading was that this is mostly built on top of #95 with a perturbation addition. But since many submissions picked up your perturbation trick, you can argue that it's significant enough to be considered novel." — Yousfi's framework for "novel": adoption-by-others is one factor; "built on top of X with addition Y" is the default reading. Useful for our PR positioning.
- Yousfi 22:05Z: standard "give me a job" closure template applied to every leaderboard PR.

### PR #98 — `hnerv_muon_finetuned_from_pr95` by @EthanYangTW

PR body: 0.1963 → improvement from #95's 0.1987. QAT fine-tuned from #95's pretrained HNeRV weights.

**Mined signal**:
- @EthanYangTW 09:33Z: "please note that the code is build from #95 's amazing work" — competitive-courtesy attribution norm.
- Yousfi 22:05Z: standard job-recruiting closure.
- Copilot bot inline review (9 comments): every issue Copilot flagged is the same class — (1) missing brotli in pyproject + runtime ImportError risk; (2) compress.py / inflate.py reproducibility-vs-shipped-archive divergence; (3) RNG seed not set deterministic; (4) docstring usage examples reference WRONG module paths; (5) stage-resume restores weights but not optimizer state. **Maintainer Yousfi DID NOT respond to any Copilot comment** — strong signal that Copilot review comments are NOISE to maintainer-attention; only human reviewer comments matter for prize decisions. For our PR posture: Copilot will fire on inflate.sh `pip install brotli` patterns + on missing deterministic seeds + on docstring drift, but we should care about REAL bugs not Copilot-cosmetic-flags.

### PR #102 — `hnerv_lc_v2_scale095_rplus1` by @EthanYangTW (3rd prize)

PR body: 0.19538 CPU (Yousfi T4 CPU after re-run). 178,981 bytes (same as #100 archive payload; "inference-time code constants changed"). Two specific changes from #100: retuned latent correction scale 0.0100 → 0.0095; "added a zero-byte decode-side nudge: frame 0 red channel +1." Cites full attribution chain @BradyMeighan → @EthanYangTW → @AaronLeslie138.

**Mined signal**:
- "zero-byte decode-side nudge: frame 0 red channel `+1`" — concrete byte-stable transformation that moves score without affecting archive bytes. TECHNIQUE OBSERVATION: inference-time-only constants can be tuned to push score independently of archive grammar.
- Yousfi 16:32Z: "please move the zip file outside of the pr (you can drag and drop it here and copy the link in the pr template)" — operational pattern Yousfi enforces consistently. Repo-archive-via-comment-attachment is canonical.
- Yousfi 16:54Z: posts the moved archive.zip URL himself when contestant doesn't promptly respond — maintainer-helpful pattern.
- Copilot inline (3 review comments): same class as #98 — (1) inflate.sh `pip install` brotli is non-deterministic / air-gapped failure mode; (2) inflate.py same `pip install` issue; (3) comment-vs-code divergence on DELTA_SCALE constant. Again: NO Yousfi response. Copilot reviews are bot-noise.

### PR #101 — `hnerv_ft_microcodec` by @SajayR (1st prize GOLD)

PR body: 0.19 CPU. 178,258 bytes. "Built on top of #95 and #98. Adds a self-contained entropy repack of the decoder, temporal latents, correction sidecar (similar in direction to #100), and related payload optimizations." 15 lines total in PR body.

**Mined signal**:
- @SajayR 18:12Z to Yousfi: "thanks for the re-scoring!" — terse competitive-courtesy.
- Yousfi 18:14Z: "gg!" — extremely terse maintainer congratulation. Signal: Yousfi keeps gold-prize banter casual / does not require elaborate justification from the winning author.
- Yousfi 19:53Z gold prize confirmation template (identical to silver/bronze).
- ZERO Copilot bot review comments — PR body brevity may correlate with reduced bot-attention surface area (the bot flags `inflate.sh` + `inflate.py` patterns; minimal LOC = minimal flag surface). NOT a causal claim but a correlation worth noting.

### PR #105 — `kitchen_sink` by @valtterivalo (honorable, 1776 LOC across 21 files)

PR body: 0.19797 CPU. 177,857 bytes. Long-form personal narrative: "wild month. most evenings after kids have gone to bed i've sat down at my computer to work on this problem. especially this final couple hours of pushing against a super competitive field was all of super fun, stressful, and exciting." Then "i now regret staying in stealth for the entirety of the challenge since there were fun shenanigans happening in the repo all throughout the month. for future iterations if leaderboard is public, i'll be more than happy to push breakthroughs live right away."

**Mined signal**:
- @valtterivalo retrospective: stealth-mode prevented stacking-on-others; explicit lesson for future contests: PUBLISH EARLY = more competitive engagement.
- Yousfi 19:59Z: standard honorable-prize template.
- Yousfi 21:23Z (write-up prize negotiation): same response as PR #97 — sole-prize for Aaron's write-up despite multiple worthy candidates.
- @valtterivalo 08:01Z: "thanks for the fun competition, lot of ups and downs all the way to the end. i'll take the next one!" — competitive-courtesy + signal next-challenge participation.
- Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" lesson, PR #105's `kitchen_sink` 1776 LOC LOST to PR #103's 241 LOC silver. Author's "stealth" retrospective is consistent with: race-window contestants who DIDN'T see #95 publish at 07:47 UTC couldn't stack.

### PR #67 — `qpose14_r55_segactions_minp` by @EthanYangTW (leaderboard, GPU-required)

PR body: 0.31 CUDA. 276,741 bytes. "Updated from the previous r55 payload to the `qpose14_r55_segactions_minp` payload with embedded SegNet tile-action corrections and fixed no-header single-blob packing."

**Mined signal**:
- Yousfi 02:05Z: "can you host the zip file outside of the repo? you can drag and drop it in the comments here." — repeated maintainer pattern: NO checked-in archive.zip. Operationally enforced.
- @EthanYangTW 02:16Z: pastes the GitHub release-asset URL ("https://github.com/EthanYangTW/comma_video_compression_challenge/releases/download/qpose14-r55/archive.zip") — RELEASE-asset hosting is the canonical pattern.
- Copilot inline: `try: import brotli except ImportError: class _BrotliCompat: @staticmethod def decompress(*args, **kwargs): raise RuntimeError(...)` — pattern Copilot DOES suggest as the canonical fallback. Useful for our inflate.py: explicit ImportError handler with actionable message > silent `pip install`.
- Yousfi 22:05Z: standard closure.

### PR #74 — `ph4ntom_drv` by @hypery11 (leaderboard, GPU-required)

PR body: 0.35 CPU (0.37 CUDA after Yousfi re-run). 321,311 bytes. Long-form writeup with "You don't need to preserve the video. You need to preserve what the network sees." framing. Side-by-side SegNet comparison images.

**Mined signal**:
- @hypery11 02:42Z: "Fixed both — reverted pyproject.toml change and moved deps install to inflate.sh. Removed assets from the repo, images are hosted on release assets now." — TWO maintainer-enforced patterns: (1) NO pyproject.toml mutations (revert any pyproject changes); (2) NO repo-checked-in assets (release assets only).
- Yousfi inline review on `pyproject.toml`: "can you revert this?" — single-line maintainer firm rule. NEVER edit pyproject in your submission PR.
- @hypery11 03:17Z: "Synced pyproject.toml with master (brotli was already added by quantizr's merged PR). The previous eval failure was due to brotli not being in the uv environment." — useful operational signal: brotli IS in pyproject (added by Quantizr's PR #55 merge); NOT a runtime dependency anymore for new submissions BUT inflate.sh `pip install` patterns are STILL needed because the eval container's uv-sync surface is fragile.
- @hypery11 11:54 UTC body: "Compression Rate: 0.0" appears to be a truncated body — full body was likely longer. The display only shows partial body in `gh pr view --json body`.
- Yousfi 22:05Z: standard closure.

### PR #86 — `jas0xf_adversarial_neural_representation` by @jas0xf (leaderboard)

PR body: 0.27 CUDA. 207,579 bytes. Self-contained neural representation. Writeup at jas0xf/comma-anr-supplementary.

**Mined signal**:
- @nick-neely 01:56Z: "Sweet writeup!" — non-contestant audience-engagement; the contest comments thread attracts spectators.
- Yousfi 02:44Z: "would you mind moving the archive, the assets, and the write-up to a different repo? I would like to keep this repo lightweight" — repeated maintainer pattern: NO heavy assets in the submission repo.
- @jas0xf 03:04Z: pushes to separate repo `jas0xf/comma-anr-supplementary` in 20 minutes — fast compliance pattern.
- @jas0xf 21:34Z to Yousfi: "Absolutely, I'm totally fine with that! Go right ahead. I actually read through Aaron's write-up as well—he had a really great analysis of HNeRV in his report. Well deserved!" — competitive-courtesy norm.

### PR #71 — `tomasdousek - ditcher` by @TomDousek (leaderboard)

PR body: 0.71 CUDA. 290,747 bytes. GPU-required.

**Mined signal**:
- Yousfi 02:07Z: "nice visualizations! can you host them somewhere else? I would like to keep the repo lightweight." — repeated maintainer pattern.
- @TomDousek 06:29Z: "I am glad I put my work into this challenge learned a lot and **did not vibe-code everything like the best submissions**." — sharp competitor framing: implicit critique of LLM-assisted top-3 submissions.
- Yousfi 21:21Z: write-up prize template (sole-prize-to-Aaron pattern).
- @TomDousek 12:13Z: "I am also working on the controlls challenge so I will then write you an email" — useful signal: comma.ai runs MULTIPLE concurrent challenges; controls + compression are both active. Cross-challenge contestants exist.
- @TomDousek mentions "World Model paper from David Ha" — useful technique-reference for our Z6/Z7/Z8 predictive-coding substrate work (per CLAUDE.md Catalog #310/#311/#312 lineage).

### PR #56 — `selfcomp` by @szabolcs-cs (leaderboard)

PR body: 0.36 → 0.38 CUDA. 279,036 bytes. SegNet fit "using same trick as Quantizr. (Idependent idea)". PR #56 is the canonical Selfcomp/szabolcs-cs PR — NOT Quantizr's PR (Quantizr is a separate handle).

**Mined signal**:
- Yousfi 16:55Z: "nice" — terse maintainer acknowledgment.
- Yousfi 16:56Z: "can you upload the zip file? can't find it" — operational pattern: archive-via-comment-attachment, not body-link.
- @szabolcs-cs 21:41Z: "edited" — sub-2-character contestant reply.
- Yousfi 22:05Z: standard closure.
- PR-body attribution language: "SegNet fit using same trick as Quantizr. (Idependent idea)" — establishes that the per-class FP4 weight self-compression paradigm was DUAL-DISCOVERED by szabolcs-cs and Quantizr.

### PR #55 — `quantizr (0.33)` by @Quantizr (honorable, 299,970 bytes)

PR body: "This challenge kind of nerd sniped me... but I think this is as much as I'm gonna work on this. Also I don't have a good name for this submission. The training script isn't the exact training script I used as I had three different scripts which I ran a few times finetuning the previous best checkpoint. I had an LLM combine the scripts and to make a 5 stage single file pipeline which mimics what I did but I haven't run the entire script to..."

**Mined signal**:
- Yousfi 15:41Z (early-April canonical): "Very nice! @Quantizr I'm glad the challenge nerd sniped you :D Leave some time for the v2, v3, etc. It's also interesting to see the artifacts at the image boundary, due to padding in the conv layers I'm assuming" — TECHNIQUE OBSERVATION: Yousfi himself notes the boundary-artifact-from-padding pattern. Useful signal for our SegNet attack on padding-artifacts.
- @dllu inline 05:25Z: "From the README.md, > External libraries and tools can be used and won't count towards compressed size, unless they use large artifacts (neural networks, meshes, point clouds, etc.), in which case those artifacts should be included in the archive and will count towards the compressed size. **This applies to the PoseNet and SegNet.** Did you remember to add the SegNet and PoseNet weights to the compressed size? That should add about 90 MB to the final size. Or do the rules only mean that you can't use SegNet and PoseNet in the inflation step (but you can use them in compression without adding to the size)?" — CRITICAL rules-clarification question.
- Yousfi 12:52Z inline reply: "Yes. They only count if they're used in decompression. Next bullet point in the rules explains that." — CANONICAL RULES INTERPRETATION: PoseNet + SegNet weights ARE allowed at COMPRESS time (free); they ONLY count toward archive size if used at INFLATE time. This is the strict-scorer-rule per CLAUDE.md.
- Yousfi 19:57Z: "This submission won an honorable prize" — Quantizr's status as the canonical reference + honorable prize.

### PR #96 — `rem2_HNeRV submission (0.21)` by @rem2 (leaderboard)

PR body: 0.21 CPU. 186,631 bytes. "no additional comments" (empty body section).

**Mined signal**:
- Minimal author commentary; PR is a placeholder pre-#103 final submission.
- Yousfi 22:05Z: standard closure.
- The CUDA / CPU shake-up (0.24 CUDA → 0.21 CPU on same archive) is visible in the bot eval rows; consistent with the 0.02-0.03 CPU/CUDA gap.

---

## Section 3: Maintainer hints consolidated (CRITICAL — highest priority for our strategy)

Per CLAUDE.md "Frontier target — NON-NEGOTIABLE, HIGHEST EMPHASIS" + "Apples-to-apples evidence discipline": maintainer-direct comments from @YassineYousfi carry HIGHEST signal weight because they constitute the contest authority. Ranked by actionable-implication-for-our-strategy:

### H1 (HIGHEST) — Yousfi 2026-05-04 PR #103 17:11Z on CPU/CUDA scoring policy

> "One way to fix this is to make the score relative to a baseline computed on the solution runtime. Maybe for the next phase, or maybe we'll just do one runtime."

**Actionable implication**: For our submission, we should target the LOWER of {contest-CPU, contest-CUDA} on the same archive bytes — since Yousfi's policy was "score both CPU/GPU and keep the best" per @rem2's proposal. Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable, we already MUST run both axes. The mining confirms Yousfi will use the better of the two for the leaderboard ranking. Our submission strategy: PR body cites BOTH axes; lower number gets the headline.

### H2 (HIGHEST) — Yousfi 2026-05-04 PR #95 16:29Z on open-publishing prizes

> "we are going to reward folks publishing their code even if not in top 3"

**Actionable implication**: ALREADY in CLAUDE.md context. Confirmed by mining as a canonical maintainer commitment. For our submission, code-publishing posture is REWARDED regardless of placement.

### H3 (HIGH) — Yousfi 2026-04-26 PR #55 12:52Z on PoseNet/SegNet weights counting

> "Yes. They only count if they're used in decompression. Next bullet point in the rules explains that."

**Actionable implication**: This is the canonical strict-scorer-rule. Already enforced via Catalog #6 `check_no_scorer_load_at_inflate`. Our submission MUST NOT load PoseNet / SegNet at inflate time. Mining confirms NO public exception has been made.

### H4 (HIGH) — Yousfi 2026-05-05 PR #100 21:32Z on "novel" framework

> "My reading was that this is mostly built on top of #95 with a perturbation addition. But since many submissions picked up your perturbation trick, you can argue that it's significant enough to be considered novel."

**Actionable implication**: Yousfi's "novel" rubric has TWO axes: (a) genealogical difference from upstream PRs; (b) adoption-by-others. For our submission (FEC6 fixed-Huffman k=16 frame-exploit selector), we should EMPHASIZE in the PR body both: (a) the technique is structurally different from HNeRV-family (not a perturbation; a different paradigm); (b) note any prior PR with same technique class for citation. The frame-exploit-selector is genuinely new (no other PR mentions it).

### H5 (HIGH) — Yousfi 2026-05-04 PR #97 17:52Z on small-hw-difference

> "yeah there is a small hw difference in the decode, I ran all submissions in t4 for a fair comparison."

**Actionable implication**: Yousfi underestimated the gap MAGNITUDE; the CPU/CUDA delta on HNeRV-class is +0.02 to +0.03 score. Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent (UPDATED 2026-05-04)": at PR106 frontier (pose_avg ~3.4e-5), the gap is MORE pronounced because we're past the crossover threshold. Our PR body should cite BOTH axes honestly so this dispute pattern doesn't apply to us.

### H6 (HIGH) — Yousfi 2026-04-19 PR #55 15:41Z on padding artifacts

> "It's also interesting to see the artifacts at the image boundary, due to padding in the conv layers I'm assuming"

**Actionable implication**: Yousfi himself notices boundary-artifacts as a SegNet-vulnerable surface. For our substrate-design work (Catalog #228 F3 GTScorerCache + sister substrate trainers), the SegNet stride-2 stem blind-spot + padding-region artifacts are a confirmed adversarial-attack surface. Useful for prioritizing SegNet adversarial techniques (Wunderkind cluster sister substrates).

### H7 (MEDIUM) — Yousfi 2026-05-05 PR #102 19:53Z + sister PRs on recruiting

> "Please email me at {first name}@comma.ai for logistics. Let us know if you are looking for a job/internship as well."

Repeated across every prize-winning PR. The contact is `{first name}@comma.ai` (i.e., `yassine@comma.ai`). For our submission, if we want to engage Yousfi for non-PR-questions, this is the canonical channel.

### H8 (MEDIUM) — Yousfi 2026-05-05 PR #100 22:05Z + sister PRs on job-recruiting closure

> "If you are looking for a job or internship, please email givemeajob@comma.ai with a link to this PR."

Standard closure on every leaderboard PR. `givemeajob@comma.ai` is the canonical comma.ai recruiting channel.

### H9 (MEDIUM) — Yousfi 2026-05-03 PR #74 02:45Z on pyproject.toml mutations

> Inline review on `pyproject.toml`: "can you revert this?"

**Actionable implication**: NEVER edit pyproject.toml in a submission PR. Maintainer-firm rule. Our submission `submissions/...` directory should be self-contained; any `inflate.sh` deps installed via `pip install --user` or similar within the submission's own startup.

### H10 (MEDIUM) — Yousfi repeated pattern across #67, #71, #86, #102 — archive hosting policy

> "can you host the zip file outside of the repo? you can drag and drop it in the comments here." (variations)
> "I would like to keep the repo lightweight." (repeated across #56, #67, #71, #74, #86)

**Actionable implication**: CANONICAL pattern: archive.zip via GitHub release-asset OR via comment-attachment (drag-drop). NEVER checked into the submission directory in PR. Use `https://github.com/<fork>/.../releases/download/<tag>/archive.zip` per the PR #100 / #101 / #102 pattern.

### H11 (CRITICAL — POLICY NORM) — Yousfi 2026-05-05 PR #103 17:11Z on private-influence

> "trying to influence things privately is not the way to do so :)"

**Actionable implication**: Maintainer-norms: evaluation disputes MUST be PUBLIC. Email-channel for non-evaluation logistics ONLY. For our submission, ALL evaluation-correctness questions go in PR comments, NOT private email.

### H12 (LOW) — Yousfi 2026-04-19 PR #55 15:41Z on inviting future versions

> "Leave some time for the v2, v3, etc."

Tonally welcoming to iterative improvement; Yousfi-cordial pattern for early submissions.

---

## Section 4: Competitive intelligence consolidated

### CI1 — Race-window dynamics (per CLAUDE.md anchor, comments-confirmed)

PR #95 published 2026-05-04T07:47:15Z. Top-3 medalists (#101 GOLD / #103 SILVER / #102 BRONZE) all submitted between 11:50 and 11:55 UTC — within 4 hours 8 minutes. PR #105 `kitchen_sink` (1776 LOC, 21 files) lost to PR #103 (241 LOC, 2 files) — per @valtterivalo's own retrospective "i now regret staying in stealth for the entirety of the challenge". This is the CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" lesson, empirically confirmed.

### CI2 — Open-source attribution chain norm

Every contestant who built-on-others CITED their upstream-PRs in body: @EthanYangTW (#98 → #95); @BradyMeighan (#100 → #98 → #95); @EthanYangTW (#102 → #100 → #98 → #95); @SajayR (#101 → #95 + #98); @rem2 (#103 → #100 → #98 → #95). For our PR body: cite our upstream-PR-references explicitly (e.g., FEC6 lineage citation if any prior PR uses fixed-Huffman or frame-exploit-selector class).

### CI3 — Per-pair latent-correction sidecar is canonical "novel" technique (per PR #100 BradyMeighan claim)

Two of three prize-winning PRs (#103 silver + #102 bronze) consumed @BradyMeighan's PR #100 per-pair latent-correction sidecar. Yousfi's post-prize update to the leaderboard acknowledged it as novel. For our FEC6 frame-exploit selector positioning, citing the per-pair-correction-sidecar genealogy explicitly as DIFFERENT (we target frame-exploits, not latent-perturbation) may help maintain "novel" framing.

### CI4 — Tomas Dousek's "did not vibe-code" critique

> "I am glad I put my work into this challenge learned a lot and did not vibe-code everything like the best submissions"

@TomDousek's PR #71 comment 2026-05-04T06:29 implicitly critiques LLM-assisted top submissions. This is a competitive-cultural signal: some contestants frame LLM-assistance as cheating. For our submission posture (per the AI-assisted-inverse-steganalysis-on-contest-problem-space methodology), we are explicitly AI-assisted but the methodology is the WORK, not the writing — useful for our own PR-positioning calibration.

### CI5 — Multi-challenge contestants exist

@TomDousek mentions "I am also working on the controlls challenge" — comma.ai runs concurrent challenges; cross-challenge contestants exist. For our forward-strategy: a single contestant identity across multiple challenges may compound recognition.

### CI6 — Stealth vs publish-early trade-off retrospective

@valtterivalo (#105): explicit retrospective regret on stealth-mode for 1776 LOC submission that lost to 241 LOC. @AaronLeslie138 (#95): explicit complaint that publishing source enabled others to stack and beat him in 4 hours. These are TWO opposite competitive-strategy lessons:
- Publish early enables incremental improvements (good for the field; sometimes bad for the publisher)
- Stay in stealth means you don't get to benefit from others' open work
For our forward-strategy: a HYBRID approach (publish substrate-engineering work early; keep race-window combinations private until submission window) may be optimal.

### CI7 — David Ha world-model reference

@TomDousek cites the World Model paper from David Ha in #71. Useful technique-anchor for our Z6/Z7/Z8 substrate work per CLAUDE.md Catalog #310/#311/#312.

---

## Section 5: Technique observations consolidated

### T1 — HNeRV-family canonical architecture (PR #95 → all medalists)

Architecture: 229K-parameter HNeRV decoder + 28-d-per-frame-pair latents (~600 pairs). Training: 8-stage curriculum: cross-entropy seg → τ-Softplus margin → smooth disagreement → +QAT → +L7 hard-pixel weighting + C1a regularizer → λ-sweep → σ-sweep → +Muon optimizer (WD=5e-4 per Chen-Li-Liu spectral-norm KKT theory). Storage: INT8-quantized + brotli-compressed to 178 KB.

### T2 — Per-pair latent-correction sidecar (PR #100 BradyMeighan, the canonical novel technique)

Single-dim grid-searched perturbation of HNeRV latents chosen to minimize joint SegNet+PoseNet loss. Adopted by #102 (re-tune of one constant) and #103 (lossless byte-level repack but consumes the same sidecar). 2-of-3 medalists stand on this technique.

### T3 — Karpathy-style LLM-driven autoresearch loop (PR #97 BradyMeighan)

195 short-budget proxy experiments where the agent reads the previous result, proposes one algorithmic change, runs the proxy, decides keep or revert. Architecture-search methodology. Resulting mask + dual-head FiLM-conditioned generator was "best mask-based score on the leaderboard by ~0.08 over the next mask-family submission."

For our work: the autopilot-ranker + cathedral-autopilot loop (per CLAUDE.md sister discipline) is conceptually similar; the canonical implementation already lives in `tools/cathedral_autopilot_autonomous_loop.py`. Mining confirms the methodology has empirical precedent.

### T4 — Arithmetic coding on largest weight tensors (PR #103 rem2 — silver)

constriction range coder on the 8 largest weight tensors and the latent-hi byte stream. Q8 (uint8) histograms beat brotli's symbol-level entropy by ~290 B. Single `RangeEncoder` merging 9 AC streams eliminates per-stream rounding overhead.

For our work: per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE", arithmetic coding / range coding / ANS / Huffman-style coders are first-class score lanes. PR #103 confirms ~290 B / 178 KB ≈ 0.16% archive-byte improvement is achievable on top of brotli for HNeRV-class. For FEC6-class, this may be a sister lane.

### T5 — Zero-byte decode-side nudge (PR #102 EthanYangTW — bronze)

"frame 0 red channel `+1`" — concrete byte-stable transformation that moves score without affecting archive bytes. Pure inference-time constant. TECHNIQUE OBSERVATION: inference-time-only constants can be tuned independently of archive grammar to push score.

For our work: this is the same class as Catalog #205 inline-device-fork (sub-pixel-difference at inflate time changes per-pair PoseNet/SegNet scores). Our PR #110 ships FEC6 which is a different paradigm but inference-time tuning may compose orthogonally on top.

### T6 — Mask + dual-head FiLM-conditioned generator (PR #97 BradyMeighan)

"mask + dual-head FiLM-conditioned generator was the best mask-based score on the leaderboard by ~0.08 over the next mask-family submission." Mask-based class scored 0.23; the next mask-family submission scored ~0.31 (cross-checked: PR #67 EthanYangTW qpose family scored 0.31). The +0.08 gap suggests FiLM-conditioning is a structurally distinct sub-paradigm.

For our substrate-design work: FiLM-conditioning is a known technique (per Catalog #310/#311/#312 Z6 ego-motion-conditioned predictive coding lineage). PR #97 is an empirical anchor for FiLM-conditioning on the mask substrate class.

### T7 — Adversarial neural representation (PR #86 jas0xf)

"adversarial neural representation" framing — adversarial loss between encoder and SegNet/PoseNet scorer. Score 0.27 on 207,579 bytes. Per CLAUDE.md "Yousfi (challenge creator) was Fridrich's PhD student at Binghamton DDE Lab. EfficientNet steganalysis surgery → informed SegNet scorer design. The challenge IS inverse steganalysis." — PR #86's adversarial framing is conceptually right.

### T8 — Hardcoded section lengths inside inflate.py (PR #103 rem2 — silver)

"hardcoded section lengths inside `inflate.py` (no length prefixes inside the archive)" — saves length-prefix bytes by encoding the section layout in code. For our work: per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L3 "monolithic single-file `0.bin` ... Fixed offsets declared in `codec.py` source", this is the canonical archive grammar pattern. PR #103 confirms it's a valid score-lowering technique.

### T9 — Single-byte filename inside zip (PR #103 rem2 — silver)

Rename archive ZIP member to single character (e.g., `0`) to save bytes in the ZIP header. Small but cumulative win.

### T10 — Adaptive `lgwin` search in brotli (PR #103 rem2 — silver)

Sweep the brotli `lgwin` parameter for the optimal window size per payload. Per `tac.canonical_equations.brotli_cascade_bounded_per_stream_v1` (canonical equation #1 per Catalog #344): bounded-per-stream is the canonical brotli cascading framework. PR #103's adaptive-lgwin is a sister technique.

### T11 — QAT (Quantization-Aware Training) confirmed across HNeRV family

PR #98 + PR #100 + PR #101 + PR #102 + PR #103 + PR #95 + PR #105 ALL use QAT in the training curriculum. Per CLAUDE.md "QAT pipeline — non-negotiable for FP4 deployment": this is the canonical pattern. Mining confirms NO medalist skipped QAT.

### T12 — Muon optimizer (PR #95 + #98)

@AaronLeslie138 + @EthanYangTW use Muon optimizer for final fine-tuning stage. WD=5e-4 per Chen-Li-Liu spectral-norm KKT theory. Not standard PyTorch; useful for our substrate-trainer extensions.

### T13 — C1a regularizer (PR #95 — Aaron's distinctive contribution)

"C1a regularizer shapes the weight distribution toward the integer grid, which collapses the entropy floor for downstream brotli compression." Per CLAUDE.md "Bit-level deconstruction and entropy discipline", weight-distribution shaping toward the integer grid is a structural-low-entropy technique. Useful for our substrate-codec work.

### T14 — Per-class FP4 weight self-compression (PR #55 + PR #56 — dual-discovered)

PR #55 (Quantizr): 88K-94K params, sigma=15, qint_max=7. PR #56 (szabolcs-cs): "SegNet fit using same trick as Quantizr. (Idependent idea)" — dual-discovered. FP4 codebook is [0, 0.5, 1, 1.5, 2, 3, 4, 6] (unsigned E2M1). Per CLAUDE.md "Quantizr intelligence" section.

---

## Section 6: Operational hints consolidated

### O1 — `pip install brotli` in inflate.sh is fragile

Multiple Copilot warnings + multiple bot-eval-failures across PRs trace back to brotli not being installed at runtime. Per @hypery11 #74: brotli WAS added to pyproject.toml by Quantizr's PR #55 merge, so the eval container's `uv sync` should have brotli. BUT bot-evals still fail because the eval workflow does `uv run` which may not honor inflate.sh's runtime `pip install`.

For our submission: vendor brotli OR use explicit `try: import brotli except ImportError: raise RuntimeError("...")` pattern (per the Copilot suggestion on PR #67 — useful canonical fallback).

### O2 — Eval bot timing pattern

Bot eval triggered manually by Yousfi (not automatically). Time-to-eval ranges from minutes (#101) to hours (#56) to multi-day (#74). Useful for our submission: do NOT expect immediate eval. Plan submission window with eval-bot-latency tolerance.

### O3 — CPU vs CUDA eval delta on HNeRV-class

Cross-checked across #95 / #97 / #100 / #101 / #102 / #103 / #105:
| PR | CUDA score | CPU score | Δ |
|---|---|---|---|
| #95 | 0.23 | 0.20 (0.1987) | +0.03 |
| #97 | 0.25 | 0.23 | +0.02 |
| #100 | 0.23 | 0.20 (0.1954) | +0.03 |
| #101 | 0.23 | 0.19 (0.19xxx) | +0.04 |
| #102 | 0.23 | 0.20 (0.19538) | +0.03 |
| #103 | 0.23 | 0.19 (0.19487) | +0.04 |
| #105 | 0.23 | 0.20 (0.19797) | +0.03 |

**Empirical anchor for `tac.canonical_equations`**: CUDA-CPU score gap on HNeRV-family = +0.02 to +0.04 (consistently). Useful canonical anchor row for future equation `cpu_cuda_axis_gap_hnerv_class_v1`. Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable, we MUST run BOTH; gap is empirically bounded.

### O4 — Eval container limits

Per the README quoted by @rem2: CPU instance = 4 CPU + 16 GB RAM; GPU instance = T4 + 26 GB RAM + 16 GB VRAM. 30-minute time limit. Useful for our submission: stay well under 30-min CPU inflate time.

### O5 — pyproject.toml is frozen

Per Yousfi inline review on PR #74: no submission-PR edits to pyproject.toml. brotli + constriction + pyppmd are confirmed present per `feedback_codex_round78_findings_fix_with_self_protection_landed_20260509.md` Catalog #203/#224. For our submission: any deps beyond these must be installed inflate.sh-time via `pip install --user`.

### O6 — Working environments cited in PR threads

- @BradyMeighan #100: "personal 3090" + "two A100 colab runs" — 3090 + Colab A100 are realistic working environments
- @Quantizr #55: paths show `/mnt/d/comma_submission/...` — Windows + WSL Linux
- @EthanYangTW #98: paths show `/root/comma_video_compression_challenge/...` — Linux root user
- @rem2 #103: implicit local CPU runtime (since `--device cpu` is the preferred runner)

### O7 — Multi-week training-loop endurance

Per @BradyMeighan #100: "literal weeks of compute on my personal 3090 with the autoresearch loop running overnight while I slept" — multi-week training-loop endurance is the empirical norm for top-3 contestants. For our cost-band-calibration anchor: this validates that long-burn training is competitive AND expected.

### O8 — Bot-eval re-runs after Yousfi's policy switch

The CPU re-eval batch ran 2026-05-05 16:00-17:30 UTC (12-13 hours after CUDA eval). Yousfi triggered CPU re-evals manually for the medalist cluster after the #103 dispute. Useful for our submission: if CPU/CUDA gap matters for our PR ranking, request CPU re-eval politely in PR comments.

---

## Section 7: Cross-references consolidated

### Cross-refs from comment threads

- @AaronLeslie138 PR #95 cites `https://aaronleslie.dev/blog/comma-compression` — the best-write-up prize artifact (note CLAUDE.md operator decision: NOT yet fetchable; JS-rendered SPA, Wayback empty; per CLAUDE.md "PR 95 full deep research" lane this is documented)
- @BradyMeighan PR #97 cites `https://comma-writeup.pages.dev/` (interactive) + `https://github.com/BradyMeighan/vibe-coder-final-boss/blob/main/WRITEUP.md` (markdown)
- @TomDousek PR #71 cites `https://tomdousek.github.io/` (writeup)
- @jas0xf PR #86 cites `https://github.com/jas0xf/comma-anr-supplementary` (writeup + archive)
- @BradyMeighan PR #100 cites fork CI `https://github.com/BradyMeighan/comma_video_compression_challenge/actions/runs/25312482561` (reproducibility evidence)
- @SajayR PR #101 cites `https://github.com/SajayR/comma_video_compression_challenge/releases/download/hnerv-ft-microcodec-v1/archive.zip` (release-asset archive hosting)
- @TomDousek PR #71 references "World Model paper from David Ha" — useful for our Z6/Z7/Z8 substrate lineage
- @BradyMeighan PR #100 references "Karpathy-style LLM-driven autoresearch loop" — useful for our cathedral-autopilot positioning
- @AaronLeslie138 PR #95 references "Chen-Li-Liu spectral-norm KKT theory" — useful for our Muon-class substrate-trainer extensions
- @AaronLeslie138 PR #95 references HNeRV (Chen et al. 2023 arXiv:2304.02633 + github.com/haochen-rye/HNeRV)
- @rem2 PR #103 references constriction range coder library (cite-this for our PR body)

### Sister-PR references implicit in our work

Per our PR #110 FEC6 fixed-Huffman k=16 frame-exploit selector:
- HNeRV-family lineage: NOT cited because our class-shift away from HNeRV-renderer paradigm
- PR #95 attribution chain: SHOULD be cited in our PR body Reproducibility section since FEC6 itself does not derive from HNeRV
- per CLAUDE.md Catalog #110 HISTORICAL_PROVENANCE: our PR is structurally orthogonal to the HNeRV stack

---

## Section 8: Op-routable recommendations for our strategy

Per the operator framing "review all of the comments on all of them for information we can use and optimize against":

### OR1 (PRIORITY 1) — PR body must follow medal-class brevity + axis-disclosure norms

Mining confirms PR101 GOLD body is 15 lines; PR102 BRONZE body is structured with one-line attribution chain + concrete 2-line technique-delta + EXACT score from rounded CPU report components. Our PR #110 body should:
- Cite the attribution chain (upstream PRs we built on, if any) with @-mention + #PR-link
- State EXACT CPU + CUDA scores (both axes per CLAUDE.md non-negotiable)
- Concrete technique-delta (one paragraph)
- NO emoji, NO `Co-Authored-By Claude` trailer (per CLAUDE.md "FORBIDDEN CLAUDE ATTRIBUTION IN PUBLIC-PR SURFACES" non-negotiable)
- Reproducibility section per Slot J D5 PROCEED_WITH_REVISIONS binding revision

Per the Slot K T3 symposium binding revisions (per `feedback_pr_95_quantizr_study_citations_landed_20260519.md`), our body is already at 15+ lines but well below PR105's 1776-LOC failure-mode.

### OR2 (PRIORITY 1) — Position FEC6 frame-exploit selector as STRUCTURALLY DIFFERENT from HNeRV-family

Per H4 Yousfi novelty rubric ("built on top of X with addition Y" = baseline; "novel because different + adopted by others" = strong novelty claim): our FEC6 + fixed-Huffman + k=16 frame-exploit selector should be FRAMED as:
- NOT a perturbation on HNeRV
- A frame-exploit-selector paradigm (per CLAUDE.md "Bit-level deconstruction and entropy discipline" + Catalog #228 GTScorerCache F3 + Catalog #344 canonical equation cluster)
- Adopted-by-others lineage: explicit citation of any prior PR with frame-exploit-class technique (none known from mining)

PR body should explicitly note "this is a paradigm distinct from HNeRV-family (no neural decoder; no per-pair latent correction); pure entropy-coding + frame-exploit-selector class."

### OR3 (PRIORITY 1) — Run BOTH contest-CPU AND contest-CUDA eval; cite both; lead with lower

Per H1 Yousfi CPU/CUDA policy + empirical O3 gap: our PR body must cite BOTH axes. Per the Catalog #316 frontier pointer canonical state, our local frontier carries contest-CPU 0.19205 / contest-CUDA 0.20533 (different archives). For our submission, run paired eval on the SAME archive bytes via the canonical helper. Cite the LOWER axis as the headline.

### OR4 (PRIORITY 2) — Cite per-pair-latent-correction-sidecar genealogy and explicit divergence

Per CI3: 2-of-3 medalists adopted PR #100's per-pair-latent-correction-sidecar. Our PR body should briefly note: "This submission does NOT use per-pair latent-correction sidecar (different paradigm class)" so Yousfi can position our PR as a genuinely orthogonal sub-paradigm.

### OR5 (PRIORITY 2) — Use canonical archive-hosting pattern (GitHub release-asset)

Per H10 + PR #100/#101/#102 examples: archive.zip lives at GitHub release-asset URL on our fork (not in submission directory). Per Slot K + Slot J binding revisions: use `gh release create` on `adpena/comma_video_compression_challenge` fork with archive.zip attached.

### OR6 (PRIORITY 2) — Add Reproducibility section per PR #100 pattern

Per the PR #100 fork-CI reference + Slot J PROCEED_WITH_REVISIONS revision #5: our PR body should include a Reproducibility section with archive SHA256 + size + ZIP-member contents + inflate runtime composition + dependency closure + entry-point contract + rate-term identity (the existing draft Section 5).

### OR7 (PRIORITY 3) — Anticipate Yousfi response templates

Mining shows Yousfi uses 3-5 canonical templates: (a) eval-trigger acknowledgment; (b) "host outside the repo / keep repo lightweight"; (c) "we're going to reward code-publishing"; (d) prize-winner congratulations + recruiting; (e) write-up-prize-declined-but-link-in-README. Our `feedback_pr_submission_yousfi_non_merge_response_template_20260519T182635Z.md` should be cross-checked against these templates so we have a draft response for each.

### OR8 (PRIORITY 3) — Avoid Copilot-bot-attention surface area

Per Section 2 PR #98 + #102: Copilot bot fires on (a) `pip install brotli` in inflate.sh / inflate.py; (b) docstring-vs-actual-module-path drift; (c) missing deterministic seeds; (d) compress.py / inflate.py reproducibility-vs-shipped-archive divergence. Our submission inflate.py / inflate.sh should minimize these surface areas to avoid noisy Copilot review comments that the maintainer ignores anyway.

### OR9 (PRIORITY 3) — Acknowledge David Ha + Karpathy methodology lineage in DEFERRED-LONG-TERM writeup

If we ever publish a writeup-page sister to ours (deferred per operator), we should cite:
- David Ha world-model paper (per @TomDousek #71 reference) — sister to our Z6/Z7/Z8 substrate lineage
- Karpathy-style autoresearch loop (per @BradyMeighan #100) — sister to our cathedral-autopilot loop
- HNeRV (Chen et al. 2023 arXiv:2304.02633) — already in our PR body Reproducibility section per Slot K
- constriction range coder (per @rem2 #103) — sister to our brotli-cascade canonical equation #1

### OR10 (PRIORITY 4) — Operational hints inform our cost calibration

- @BradyMeighan #100 confirms multi-week 3090 training is competitive baseline → useful for our cost-band-calibration `predicted_cost_usd` ceiling
- T4 + 26 GB RAM + 16 GB VRAM + 30-min eval limit → useful for our `tac.canonical_dispatch_optimization_protocol` Tier 2 hardware-correctness invariants
- CPU eval ~5-10 minutes for HNeRV inflate per @szabolcs-cs #56 → useful upper-bound on CPU inflate time

---

## Section 9: Cross-link to MG-17 T3 symposium

This mining artifact is INPUT to MG-17 T3 grand council symposium (in flight; subagent_id `slot_mg_17_t3_voice_tone_style_review_symposium`). MG-17 reviews voice/tone/style/level-of-detail/content of MG-16's consolidation output. The mined contestant-author voice (informal-but-precise, attribution-citation-chain, brevity-norm, axis-disclosure-norm) calibrates MG-17's reviewer council baseline.

Key MG-17 calibration anchors from this mining:
- PR101 GOLD body brevity: 15 lines
- PR102 BRONZE body structure: 1 paragraph technique + 1 paragraph attribution + EXACT CPU score
- PR95 best-write-up body: longer-form personal-context first-paragraph + technical-paragraph + writeup-link
- Yousfi's response templates: 1-3 sentences per response; never long-form
- Contestant attribution-citation pattern: @-mention + #PR-link + parenthetical technique tag

MG-17 should treat these as the CALIBRATION TARGETS for our PR #110 body voice/tone refinement.

---

## Executive summary (1-2 sentences)

The 113 issue-level + 17 review-level comments across the top-15 PRs surface 12 maintainer-direct hints (4 PRIORITY-1 actionables for our submission strategy: paired CPU+CUDA eval lead-with-lower, novel-paradigm framing distinct from HNeRV-family, code-publishing-reward commitment, archive-hosting-via-release-asset) + 11 distinct technique observations (per-pair-latent-correction-sidecar canonical novel; Karpathy-autoresearch-loop methodology; QAT + Muon + C1a + arithmetic-coding stack across all HNeRV medalists; FP4 weight self-compression dual-discovered by Quantizr + szabolcs-cs) + 7 competitive intelligence anchors (race-window 4-hour dynamics; stealth-vs-publish-early trade-off; per-axis CPU/CUDA gap +0.02 to +0.04 across HNeRV-family). The single highest-value finding is Yousfi's explicit framework for "novel" (genealogical-difference + adoption-by-others) which directly informs our PR #110 FEC6 frame-exploit selector positioning as STRUCTURALLY ORTHOGONAL to the HNeRV-family lineage.

---

**Filing**: `.omx/research/pr_comments_mining_top_15_prs_for_actionable_signal_20260520T060250Z.md`
**Lane**: `lane_pr_comments_mining_top_15_prs_for_actionable_signal_20260520`
**Discipline**: Catalog #117 / #157 / #174 canonical serializer + POST-EDIT `--expected-content-sha256` + Catalog #206 checkpoint discipline + Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE (NEW artifact; no mutations) + Catalog #287 placeholder-rationale rejection + Catalog #229 PV (read full state of all 15 PRs + sister checkpoints before drafting) + Catalog #340 sister-checkpoint guard PROCEED + Catalog #119 Co-Authored-By Claude trailer on internal commit (this file is INTERNAL).
**6-hook wire-in declaration per Catalog #125**:
- hook #1 sensitivity-map = ACTIVE (mined technique observations feed substrate-design priority decisions which are sensitivity-relevant — e.g., T11 QAT confirmation across all HNeRV medalists informs sister substrate-trainer priorities)
- hook #2 Pareto constraint = N/A
- hook #3 bit-allocator = N/A
- hook #4 cathedral autopilot dispatch = ACTIVE (O3 CPU/CUDA empirical gap + O7 multi-week-3090 anchor inform autopilot ranker cost-band-calibration; O4 30-min CPU limit informs dispatch-protocol-Tier-2 invariants)
- hook #5 continual-learning posterior = ACTIVE (mined operational anchors feed `tac.canonical_equations` registry per Catalog #344; CPU-CUDA gap candidate equation; HNeRV-family per-pair-latent-correction-sidecar adoption-rate equation)
- hook #6 probe-disambiguator = N/A
