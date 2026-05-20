# Template C — Response to merge or non-merge verdict on PR #110

**Scope**: INTERNAL templates held for use when @YassineYousfi posts an explicit verdict on https://github.com/commaai/comma_video_compression_challenge/pull/110 — either "merging" / "merged" (C-merge) or any of the five non-merge closure categories enumerated in the sister non-merge template (C-non-merge).

**Sole-author voice**: Alejandro Peña per `user_pr_attribution.md`. Zero Claude / Anthropic / AI-assisted tokens per `feedback_forbidden_claude_attribution_in_public_pr_surfaces.md`.

**Deploy via**: `gh pr comment 110 --repo commaai/comma_video_compression_challenge --body "<TEMPLATE BELOW>"`

**Hold until**: explicit verdict comment from @YassineYousfi or the maintainer-bot's merge action. Do not deploy on neutral comments or non-verdict engagement.

---

## C-merge — merged verdict (~80 words)

> Thanks for the merge.
>
> The tooling that produced this — substrate registry, master-gradient extractor, canonical equations registry, cargo-cult unwind methodology — is open at https://github.com/adpena/tac (sister library, MIT) + https://github.com/adpena/comma-lab (working repo). If any of it is useful past the contest for openpilot / driver-monitoring / sister production paths, I'd be glad to discuss in whatever channel works — adpena@gmail.com is the most direct.
>
> Either way, glad to have contributed to the leaderboard's cumulative work.

---

## C-non-merge — closure-category responses

Five sub-templates aligned with the closure categories enumerated in the sister template `pr_submission_yousfi_non_merge_response_template_20260519T182635Z.md` (`B1 score-gap` / `B2 not-innovative` / `B3 post-deadline` / `B4 runner-busy` / `B5 modifications`).

Each sub-template is ~120 words. Each is calibrated for **long-term-relationship preservation** above closure-category dispute: the merge is the maintainer's call; the response demonstrates that the work, the discipline, and the relationship survive a non-merge verdict intact.

### B1 — Non-merge due to score gap (`0.192051` vs PR #101 GOLD `0.192845` deemed insufficient delta)

> Understood — the `-0.000794` CPU delta is small enough that the maintainer-policy call on whether to displace the current GOLD is reasonable either way. Thanks for running the eval honestly.
>
> The 2 bolt-ons (FEC6 frame-exploit selector + fixed-Huffman k=16 selector codebook) are documented at the cross-links in the PR body in case either is useful as a building block for a future stack. The substrate-design tooling that produced them is open at https://github.com/adpena/tac (sister library) + https://github.com/adpena/comma-lab (broader inventory at docs/asymptotic_floor_candidate_inventory.md).
>
> Glad to continue contributing — adpena@gmail.com is the most direct channel.

### B2 — Non-merge due to "not innovative enough" (PR #108 closure category)

> Understood — the FEC6 + fixed-Huffman bolt-ons live on top of the PR #101 substrate rather than introducing a paradigm-class shift, so the "competitive OR innovative" rubric from PR #108 closure can read either way depending on weight. Thanks for the call.
>
> The broader work past this submission targets class-shift candidates (predictive-coding world models, cooperative-receiver framings, information-bottleneck substrates) at https://github.com/adpena/comma-lab/blob/main/docs/asymptotic_floor_candidate_inventory.md. None has reached a paired CPU + CUDA anchor yet at contest scale. Happy to share progress as those land — adpena@gmail.com if you'd like to be tagged.

### B3 — Non-merge due to post-deadline timing

> Understood — past-deadline submissions land in a different review bucket than the contest-window submissions and the maintainer call on whether to evaluate is reasonable either way. Thanks for the context.
>
> Glad to have contributed the FEC6 + fixed-Huffman bolt-ons to the public record regardless of merge status. The tooling that produced them is at https://github.com/adpena/tac (MIT, sister library) for anyone who wants to build on top.
>
> Happy to continue the conversation in any channel — adpena@gmail.com is the most direct.

### B4 — Non-merge due to maintainer-runner busy / capacity-constrained

> Understood — the maintainer-bot pipeline at contest cadence is real work and the queue is the queue. Thanks for the context.
>
> No urgency on my end. The CPU `0.192051` / CUDA `0.226210` numbers in the PR body are my own Modal measurements against the hosted archive (`b0571bc` fork commit, SHA `6bae0201fb08...`); I trust those for my own decision-making and can wait on the upstream eval whenever it fits the queue.
>
> The substrate-design tooling at https://github.com/adpena/tac and https://github.com/adpena/comma-lab is open regardless — adpena@gmail.com if useful.

### B5 — Non-merge requesting modifications (e.g., remove FEC6 / split into two PRs / address specific concern)

> [Address the specific modification request first, in 1-2 sentences.]
>
> I can land that. [Outline the concrete next steps: which files / which commits / which timeline; or, if the modification is incompatible with the submission's substrate design, say so directly and propose the closest-fit alternative.]
>
> If the alternative path doesn't fit, the broader candidate inventory at https://github.com/adpena/comma-lab/blob/main/docs/asymptotic_floor_candidate_inventory.md has [N] other class-shift candidates that may align better with what you're looking for. Happy to discuss in whatever channel works — adpena@gmail.com.

---

## Tone notes

- Acknowledge the verdict first. Do not lead with disagreement or framing-pushback.
- Brief. None of the sub-templates exceeds 200 words.
- Long-term-relationship-first: every sub-template ends with an open offer of continued engagement, never a closure of the conversation. The relationship outlasts the verdict.
- Cite tooling cross-links (sister library + comma-lab repo + email) consistently. Reviewers who later want to build on the work or hire from the talent pool find them in the closure record itself.
- No emojis. No "no worries" / "all good" softeners. Match the maintainer's matter-of-fact tone from prior PR threads.

---

## Discipline applied

- `user_pr_attribution.md` sole-author voice
- `feedback_forbidden_claude_attribution_in_public_pr_surfaces.md` zero-Claude
- CLAUDE.md "Council conduct — non-negotiable" (never conservative-by-default; never closure-category disputation by default)
- CLAUDE.md "Strategic Secrecy" applied to long-term relationship: share the cross-links generously; do not withhold the broader work behind closure-status
- CLAUDE.md "Frontier target — NON-NEGOTIABLE" extended to the discipline of separating contest-verdict from research-frontier (the broader work past the contest is the durable value; merge is a checkpoint)

---

## Related

- `pr_110_hnerv_fec6_fixed_huffman_k16_submitted.md` (canonical PR #110 record)
- `pr_submission_yousfi_non_merge_response_template_20260519T182635Z.md` (the prior 5-category non-merge template this builds on; references the categories B1-B5 explicitly)
- `response_to_maintainer_bot_eval_comment_20260520.md` (sister Template A for eval-only comments)
- `response_to_substantive_yousfi_or_hotz_question_20260520.md` (sister Template B for substantive technical questions)
