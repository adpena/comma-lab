# Template B — Response scaffold for a substantive technical question from @YassineYousfi, @geohot, or sister maintainers/researchers on PR #110

**Scope**: INTERNAL scaffold held for use when an inner-council-tier reviewer (Yousfi the challenge designer, Hotz at comma.ai, a domain-adjacent researcher) asks a substantive technical question on https://github.com/commaai/comma_video_compression_challenge/pull/110 or via the discussion threads on the comma.ai Discord / email.

**Sole-author voice**: Alejandro Peña per `user_pr_attribution.md`. Zero Claude / Anthropic / AI-assisted tokens per `feedback_forbidden_claude_attribution_in_public_pr_surfaces.md`.

**Deploy via**: `gh pr comment 110 --repo commaai/comma_video_compression_challenge --body "<FILLED TEMPLATE>"` OR direct email reply OR Discord thread reply, depending on the channel the question came in on.

**Hold until**: actual substantive question lands. Do not deploy as preemptive engagement.

---

## TEMPLATE BODY (~200-400 words, fill placeholders before deployment)

> Thanks for the [question / pointer / context — pick one based on what was asked].
>
> [ACKNOWLEDGMENT PARAGRAPH — 1-3 sentences restating the question or the specific technical point being raised. Demonstrates the reviewer's signal landed accurately. If the question contains a correction or pushback on the PR, acknowledge the correction directly here without hedging.]
>
> [TECHNICAL DEPTH PARAGRAPH — 2-5 sentences responding substantively. Cite specific files / line numbers / measurements from the PR or the cross-linked context. Format choices:
>
> - If the question is about the FEC6 frame-exploit selector: cite `submissions/hnerv_fec6_fixed_huffman_k16/src/frame_selector.py` (213 LOC) + the 31-mode K=16-active-palette structure + the per-pair selection criterion.
> - If the question is about the fixed-Huffman k=16 codebook: cite `submissions/hnerv_fec6_fixed_huffman_k16/inflate.py#L64-L88` + the contrast with PR #101's canonical Huffman applied to the latent sidecar.
> - If the question is about the underlying substrate paradigm: cross-reference https://github.com/adpena/comma-lab/blob/main/docs/asymptotic_floor_candidate_inventory.md for the broader candidate landscape this submission sits within.
> - If the question is about reproducibility: cross-reference the sister library at https://github.com/adpena/tac (MIT-licensed; the canonical helpers the submission runtime imports from).
> - If the question is about methodology: cross-reference one of the methodology memos at https://github.com/adpena/comma-lab/blob/main/docs/cargo_cult_unwind_methodology.md / strict_preflight_catalog_summary.md / canonical_equations_tour.md / master_gradient_extractor_tour.md as appropriate.
> - If the question challenges a measurement: be precise about hardware (Modal Linux x86_64 vs T4 vs A100) + axis tag (`[contest-CPU]` vs `[contest-CUDA T4]`) + the specific archive SHA the measurement refers to. Never aggregate measurements across different operating points.]
>
> [OFFER OF NEXT-STEP ENGAGEMENT — 1-3 sentences. Pick the most-fitting options:
>
> - **If the question opens a research direction**: "If you'd be interested in collaborating on a workshop paper exploring [this direction], I'd be open to that — the broader candidate inventory at [inventory link] has [N] candidates that hit on this question from different angles."
> - **If the question is about tooling**: "The relevant tooling is at https://github.com/adpena/tac under MIT — happy to discuss the API design via the project's discussions or a direct email at adpena@gmail.com."
> - **If the question is about extending to other comma.ai or openpilot use cases**: "The substrate-design patterns in this submission generalize past the contest scorer — happy to discuss applications to [openpilot model_replay / driver-monitoring / sister production paths] in whatever channel works for you."
> - **If the question is about the contest itself (deadlines, scoring, future directions)**: defer to maintainer's framing; do not propose contest-policy changes.]
>
> [CLOSING — 1-2 sentences preserving long-term relationship. Examples:
>
> - "Either way, thanks for running this challenge — the year of cumulative work the leaderboard has surfaced (PR #56 / #95 / #98 / #100 / #101 / #102 / #103 / #110) is a real contribution to the field."
> - "Glad to continue this conversation in any channel you prefer."
>
> Never imply the submission is closed; never imply the conversation is over. Preserve the option for follow-up.]

---

## Scaffold-specific guidance

- **Length calibration**: 200-400 words total. Medal-class PR comment threads (PR #95 / #100 / #101 / #102 / #103) skew toward brevity; never exceed 500 words in a single response unless the question genuinely requires it. If the question requires more, split across two comments or move to a sister channel (Discord / email).
- **Tone**: technical, matter-of-fact, warm. Match the @-mention attribution conventions of the prior PR comment threads (PR #102 names 3 prior PR authors by `@<github-handle>` in its body). Do not use emojis unless the reviewer used them first.
- **Citation discipline**: every claim that names a number, file, or measurement must include the cross-link or the file:line reference. Reviewers in this thread (Yousfi, Hotz, the medal-class authors) verify claims; unverifiable claims erode credibility.
- **Anti-pattern to refuse**: any phrasing that positions the submission against the maintainer's framing ("you might want to reconsider X" / "the contest rule Y should be different" / "actually..."). The submission's positioning is "here is what I built; here is the technical context; the maintainer's framing is the canonical authority." Inverting that is closure-category-eroding behavior.

---

## Sister cross-links the response may invoke

- **Inventory**: https://github.com/adpena/comma-lab/blob/main/docs/asymptotic_floor_candidate_inventory.md
- **Cargo-cult unwind methodology**: https://github.com/adpena/comma-lab/blob/main/docs/cargo_cult_unwind_methodology.md
- **Strict preflight catalog summary**: https://github.com/adpena/comma-lab/blob/main/docs/strict_preflight_catalog_summary.md
- **Canonical equations tour**: https://github.com/adpena/comma-lab/blob/main/docs/canonical_equations_tour.md
- **Master-gradient extractor tour**: https://github.com/adpena/comma-lab/blob/main/docs/master_gradient_extractor_tour.md
- **Sister library**: https://github.com/adpena/tac
- **Direct email**: adpena@gmail.com (per `user_pr_attribution.md` public-PR-attribution email)

---

## Discipline applied

- `user_pr_attribution.md` sole-author voice
- `feedback_forbidden_claude_attribution_in_public_pr_surfaces.md` zero-Claude
- CLAUDE.md "Apples-to-apples evidence discipline" (every measurement tagged with axis + hardware substrate)
- CLAUDE.md "Bit-level deconstruction and entropy discipline" (every technical claim grounded in file:line citation)
- CLAUDE.md "Council conduct — non-negotiable" (matter-of-fact, technical, never conservative-by-default)
- CLAUDE.md "Strategic Secrecy" applied generatively: sharing cross-links to the public methodology memos rather than withheld details
- CLAUDE.md "Beauty, simplicity, and developer experience" (offer of MIT-licensed tooling cross-reference; OSS-default posture)

---

## Related

- `pr_110_hnerv_fec6_fixed_huffman_k16_submitted.md` (canonical PR #110 record)
- `pr_submission_yousfi_non_merge_response_template_20260519T182635Z.md` (non-merge verdict sister templates)
- `response_to_maintainer_bot_eval_comment_20260520.md` (sister Template A for eval-only comments)
- `response_to_merge_or_nonmerge_verdict_20260520.md` (sister Template C for verdict-bearing comments)
