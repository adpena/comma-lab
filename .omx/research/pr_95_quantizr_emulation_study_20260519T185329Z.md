# PR 95 + Quantizr emulation study (PR body posture extraction)

**Date:** 2026-05-19
**Subagent:** `pr_95_quantizr_study_citations_20260519`
**Lane:** `lane_pr_95_quantizr_study_citations_landed_20260519`
**Operator directive 2026-05-19:** *"review the best writeup winner PR 95 and quantizr's submissions because those are standouts with respectable submissions and conversation histories"* + *"we can do even better but don't want to be seen as cringe or going overboard; we want to act in such a way as they will want to hire us"*

---

## 1. Source PRs studied (verified via `gh pr view`)

| PR | Author | Title | Score | Status | Body length | Tone |
|---|---|---|---|---|---|---|
| #56 | `szabolcs-cs` (Szabolcs) | selfcomp | 0.36 → 0.38 CUDA | leaderboard, not merged | ~30 lines | terse, 4-bullet technique list |
| #95 | `AaronLeslie138` (Aaron Leslie) | hnerv_muon submission (0.20) | 0.20 CPU / 0.23 CUDA | merged, **best write-up + honorable prize** | ~40 lines | technical narrative + external blog link |
| #98 | `EthanYangTW` | hnerv_muon_finetuned_from_pr95 (0.1963) | 0.20 CPU | merged | ~30 lines | clean attribution, terse |
| #100 | `BradyMeighan` | hnerv_lc_v2 submission (0.1954) | 0.20 CPU | leaderboard | ~25 lines | terse + repro CI link |
| #101 | `SajayR` | add hnerv ft microcodec submission | 0.19 CPU / 0.23 CUDA | **GOLD** | **~15 lines** | shortest body of the medal class |
| #102 | `EthanYangTW` | hnerv_lc_v2_scale095_rplus1 (0.19538 CPU) | 0.19538 CPU | **BRONZE**, merged | ~35 lines | attribution chain explicit |
| #103 | `rem2` | hnerv_lc_ac submission (0.19) | 0.19 CPU | **SILVER** | ~25 lines | substantive change in 1 paragraph |
| #108 | `andrei-minca` | andimin01 | 3.59 | **closed** 2026-05-11 | ~40 lines | the rejected pattern (selling/explaining) |

## 2. The headline finding — restraint is the signal

The single strongest stylistic pattern across the medal class (PR101 gold, PR103 silver, PR102 bronze) is **terseness**. PR101's entire `# additional comments` section is **one sentence**:

> "Built on top of #95 and #98. Adds a self-contained entropy repack of the decoder, temporal latents, correction sidecar (similar in direction to #100), and related payload optimizations."

That's it. SajayR won gold with a one-sentence technical summary that does three things:
1. **Attribution** (`#95` and `#98`)
2. **What's new** (entropy repack + sidecar + payload optimizations)
3. **Honest provenance** (`similar in direction to #100` rather than claiming originality)

Aaron Leslie's PR 95 (best write-up prize) ran longer (~10 sentences) but the extra length all goes into **substantive technical detail** of the 8-stage curriculum, not into framing/selling. The body ends with one bare link to the external blog and stops.

Both winning patterns SHARE:
- No "Happy to discuss…" closing flourish
- No "let me know if you have questions"
- No emoji
- No bold-italic emphasis on the score
- No self-congratulation on novelty
- **One inline link per claim that needs evidence; bare attribution otherwise**

## 3. The rejected pattern (PR 108)

`andrei-minca`'s PR 108 was closed 2026-05-11 with the verbatim:

> "closing this pr per the new submission guidelines, the tricks used are already established in several past submissions"

What's instructive about PR 108's BODY style (not its result):
- "feat:" prefix from conventional commits (signals "I'm using a process other than what's asked")
- **Bullet list of marketing-tone benefits** ("improved score", "competitive score by strategically reducing bitrate")
- **Bold "Key changes include"** + sub-bullets
- Wraps with "This approach targets a competitive score by..." — explaining intent rather than showing math
- Score `3.59` (genuinely non-competitive); the closure cited "tricks already established" + "is this competitive or innovative?"

The closure crystallized the maintainer's **post-deadline gate**:

> "is this submission competitive or innovative? explain why
>  competitive: better than top # 1 submission
>  innovative: it has a novel idea that is not on the leaderboard yet, might not be competitive, but has potential"

This is the rubric our PR body MUST satisfy explicitly.

## 4. Maintainer (Yousfi) conversation tone

Across PR 95/56/100/101/102/103 Yousfi's interactions are characterized by:

- **Brevity** ("nice", "gg!", "great submission and write-up!")
- **Direct technical answers** ("yeah there is a small hw difference in the decode, I ran all submissions in t4 for a fair comparison")
- **No ceremony** on prize announcements ("This submission won # 1 prize. Please email me at {first name}@comma.ai for logistics. Let us know if you are looking for a job/internship as well.")
- **Equal treatment** — PR 56's `szabolcs-cs` and PR 95's `AaronLeslie138` both got the "email for logistics + job/internship" closing
- **Catches typos kindly** ("small typo in your write-up title 'Spritually' -> 'Spiritually'") — proofreading is appreciated
- **Recognizes good faith** — when AaronLeslie138 vented about people fine-tuning his archive in the last 3 hours, Yousfi responded "`<re:vent>` we are going to reward folks publishing their code even if not in top 3" — open-source ethos is real and rewarded
- **Asks clarifying questions when needed** ("can you upload the zip file? can't find it" on PR 56)

The **CONTRIBUTOR ↔ COLLABORATOR** dynamic is: contributor ships a small clean PR, maintainer scores it, body length matches the substance. There is no "thank you for your time" or "please consider" mid-body.

## 5. Patterns to EMULATE (operator's "hire us" framing)

1. **Brevity over comprehensiveness in the body.** The body is a contract surface for the score + reproducibility + attribution. Long-form goes in linked writeups, NOT in the body. PR 95's blog link is the only external link in the body.

2. **Attribution chain that names every prior contributor by GitHub handle.** PR 102 explicitly chains: `@BradyMeighan's hnerv_lc_v2 PR #100 → @EthanYangTW's hnerv_muon_finetuned_from_pr95 PR #98 → @AaronLeslie138's hnerv_muon PR #95`. Three authors, three PRs, in one sentence. We should do the same: explicit GitHub handles, explicit PR numbers, named-by-name.

3. **Verbatim report.txt block.** Every medal-class PR shipped the exact `report.txt` output formatted as a code block, with the upstream `=== Evaluation config ===` / `=== Evaluation results ===` / `Final score:` lines preserved. This is non-negotiable — the maintainer scans these blocks first.

4. **One-line "what's new" framing.** PR 101: "Adds a self-contained entropy repack of the decoder, temporal latents, correction sidecar (similar in direction to #100), and related payload optimizations." Our equivalent should be one paragraph max — FEC6 + K=16 + fixed-Huffman + offline selector + byte-stable.

5. **Honest hedging on hardware-axis split.** PR 102 explicitly distinguishes "Fast PyAV/CUDA scorer during tuning gave exact `0.194986956`" from CPU-axis numbers. PR 95 took Yousfi's "yeah there is a small hw difference in the decode" answer in stride and didn't argue. We should pre-emptively document the CPU/CUDA split with the same matter-of-fact tone.

6. **No prediction about merge prospects.** None of the medal-class PRs say "I hope this is merged" or "consider merging". They ship the work and stop.

7. **Yes/no answers on the template questions.** PR 101: "no" / "no". That's literally the answer to the GPU + compress.sh questions. Don't dress them up.

8. **Cite the upstream maintainer's framing when it applies.** Our submission post-dates the 2026-05-11 closure of PR 108; the body should explicitly answer the "competitive or innovative" rubric the maintainer codified there. **Our current body already does this** in the "Competitive + innovative per the 2026-05-11 new-submission gate" section — good.

## 6. Patterns to AVOID (operator's "no cringe / no overboard" framing)

1. **"Happy to discuss…" closing.** Both our current body and the canonical body end with this exact phrase. NONE of the medal-class PR bodies do. The maintainer will reach out if they want to; the offer reads as performative.

2. **Self-congratulatory novelty framing.** Avoid "innovative", "novel approach", "key insight", "breakthrough". Use neutral verbs ("adds", "encodes", "compacts"). The "Novel in this submission" header in our body is already restrained — keep that exact phrasing, don't escalate.

3. **Bold-italic emphasis on the score.** The medal-class PRs let the bare number speak. Our current body bolds `**0.1920513169**` once in the score table — that's fine; don't add more.

4. **"Operator-honest" or other meta-framing labels in section headers.** Our current body has `**Limitations (operator-honest):**` — drop the parenthetical. "Limitations" is enough; the prose IS the honesty.

5. **Extensive limitations sections.** PR 95 has zero limitations text. PR 101 has zero. PR 102 has one matter-of-fact note about CPU/CUDA paired scores. Our body's 6-bullet limitations section is the longest in the medal class. Compress it.

6. **External links that aren't reproducibility-critical.** PR 95's only external link is the blog. The medal class doesn't link to internal research dossiers, code reviews, or "see also" memos. The 3 links to internal `.omx/research/` documents in our canonical body should NOT be on the upstream-template body (they're for internal use).

7. **Source-tree LOC accounting in the body.** Our current "Inflate runtime is 1140 LOC across 4 files" is responsive to a probable maintainer question, but reads defensive. The medal class doesn't pre-empt critiques in the body. Two options: (a) trim to one sentence ("Inflate runtime: 4 Python files, fully self-contained; no scorer weights at inflate time"), or (b) move into a separate "Reproducibility" section that frames the LOC as a deterministic asset rather than a defensiveness target.

8. **Apologetic framing.** Our body says "report.txt shipped at `submission_dir/report.txt` contains an absolute path ... is not redacted because doing so would diverge from the upstream output contract." This is correct discipline but the explanation reads defensive. Either redact and explain in a footnote, or ship verbatim and don't explain — the medal class would do the latter.

## 7. Direct application to our PR body (recommendations)

Based on the study, apply these revisions:

### KEEP

- The 5-required-template-headings structure (already template-conformant)
- The verbatim `report.txt` block under `# report.txt`
- The Modal-CPU-not-host-bot-validated honesty (PR 102 sets the precedent for explicit-axis-disclosure)
- The CPU/CUDA paired score table (PR 102 has the same)
- The attribution chain in `# additional comments` (PR 102's exact pattern)
- The competitive-or-innovative gate satisfaction (the maintainer codified this; we answer it directly)
- Appendix A (upstream template citation) — this is unusual but reads as conformance evidence, not flourish
- Appendix B (pre-submission compliance gate verdict) — this is operator-routable transparency

### REVISE

- **Line 85**: DELETE the "Happy to discuss engineering details or run additional auth-eval verifications if useful." closing. Replace with nothing (let Appendix A be the natural end), OR with a single neutral line if needed.
- **Line 76 header**: Change `**Limitations (operator-honest):**` → `**Limitations:**`
- **Limitations section**: Compress 6 bullets to 4. Merge the inflate.py LOC bullet (~5 lines of text) with the report.txt absolute-path bullet into a single "Operational notes:" section since both are reproducibility-context not method limitations.

### ADD (per operator's deterministic-reproducibility directive)

Add a **Reproducibility section** that explicitly frames:
- Archive SHA-256 + size + ZIP-member name (member name `x`)
- Deterministic-ZIP discipline (fixed timestamps, fixed central-directory ordering)
- Inflate runtime tree composition (4 files, byte-stable for fixed hardware axis)
- Dependency closure (stdlib + torch + brotli only; no scorer weights at inflate)
- CPU/CUDA rate-term identity decomposition (`25·R = 0.118867`)

This section's role is to make the reproducibility contract **immediately auditable** by the maintainer scanning the body — same role the report.txt block plays for the score claim.

### ADD (per operator's citations + URL hyperlinks directive)

Inline citations + URL hyperlinks for the techniques claimed. Specifically:
- HNeRV → arXiv:2304.02633 (Chen, Gwilliam, Lim, Shrivastava 2023)
- FastViT (PoseNet backbone) → arXiv:2303.14189
- EfficientNet-B2 (SegNet backbone) → arXiv:1905.11946
- `segmentation_models.pytorch` → github.com/qubvel/segmentation_models.pytorch
- Brotli → RFC 7932
- PR101 / PR103 / PR100 / PR98 / PR95 → already in body as PR numbers; add `commaai/comma_video_compression_challenge#101` etc. or full URLs where they help

## 8. Author identification cross-reference (for accurate attribution)

The CLAUDE.md project-level memory refers to "Quantizr (Jimmy, UCLA CSE/Neuro)" as the leader at 0.33 — this APPEARS to be a misidentification.

Verified facts from `gh pr view`:
- **PR 56** is `szabolcs-cs` (GitHub login `szabolcs-cs`, no display name), submission `selfcomp`, score 0.36 CPU / 0.38 CUDA — the canonical "self-compression at ~1.017 bits/weight" pattern.
- **PR 101 GOLD** is `SajayR` (GitHub login + display name "SajayR"), submission `hnerv_ft_microcodec`, score 0.19 CPU.
- The PR 56 body itself says: *"**SegNet** was fit using the same trick as Quantizr. (Idependent idea)."* — `szabolcs-cs` references "Quantizr" as a separate person who also used the same SegNet trick.

So "Quantizr" is a third name (possibly a contest handle or an earlier PR's author) that `szabolcs-cs` cites in PR 56. Our PR body's reference to "Jimmy / 'Quantizr'" as the canonical PR101 author IS inaccurate. The correct attribution is:

- **HNeRV decoder architecture + FP4 codebook + qpose14/qzs3 wire format**: `@SajayR` (PR101)
- **"Encode only frame-0 masks; warp frame-1" insight**: this is in PR101's body as part of the entropy-repack work, but the underlying concept may trace to the earlier HNeRV-family stack (PR95 / PR98 / PR100). Without a confirmed citation in any PR body, we should attribute to PR101 GOLD itself as the merged anchor.
- **Composable selector-axis pattern**: `@rem2` (PR103) — explicitly: "arithmetic coding (constriction range coder) on the 8 largest weight tensors" + "merging all 9 AC streams into one constriction `RangeEncoder` to eliminate per-stream rounding overhead"

**This is a substantive correction to the current PR body** which credits "Jimmy / 'Quantizr'" — we should rewrite the attribution to use the verified GitHub handles + PR numbers and drop the "Jimmy / 'Quantizr'" framing.

## 9. Posture summary for body revision

**Target posture (extracted from medal-class):** Direct + technical + accessible + high-signal. Cite work. Show the math. Link reproducibility. Get out of the way.

**Concrete rewrite directives for Phase 3:**
1. Delete line 85 ("Happy to discuss…")
2. Drop `(operator-honest)` parenthetical on line 76
3. Compress limitations from 6 bullets to 4 + merge into "Operational notes"
4. Add `## Reproducibility` section (deterministic ZIP + dependency closure + axis-rate identity)
5. Add inline citation links for HNeRV, FastViT, EfficientNet, smp, Brotli (canonical-paper URLs)
6. **Correct the attribution**: `Jimmy / "Quantizr"` → `@SajayR (PR #101)` for HNeRV decoder + FP4 codebook + qpose14/qzs3
7. If Slot H delivers tac URL map: add 1-2 inline references to specific tac file URLs in the FEC6 description; otherwise omit (don't reach)

## 10. Cross-reference

- `.omx/research/pr_submission_check_in_package_20260519/PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md` (revision target)
- `.omx/research/pr_submission_check_in_package_20260519/PR_BODY_CANONICAL.md` (preserved per Catalog #110)
- Maintainer (Yousfi) PR review patterns: github.com/commaai/comma_video_compression_challenge/pull/{56,95,98,100,101,102,103,108}
- Upstream PR template: `upstream/.github/pull_request_template.md`
- Operator directive 2026-05-19: in-session verbatim quoted above
- 6-hook wire-in declaration per Catalog #125: hooks N/A (this is a documentation + research artifact; no signal-producing surface)
