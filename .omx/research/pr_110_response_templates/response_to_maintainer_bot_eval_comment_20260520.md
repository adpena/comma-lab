# Template A — Response to maintainer-bot eval comment on PR #110

**Scope**: INTERNAL template held for use when @YassineYousfi or the maintainer-bot posts a CPU+CUDA score comment on https://github.com/commaai/comma_video_compression_challenge/pull/110.

**Sole-author voice**: Alejandro Peña per `user_pr_attribution.md`. Zero Claude / Anthropic / AI-assisted tokens per `feedback_forbidden_claude_attribution_in_public_pr_surfaces.md`.

**Deploy via**: `gh pr comment 110 --repo commaai/comma_video_compression_challenge --body "<TEMPLATE BELOW>"`

**Hold until**: maintainer bot or @YassineYousfi posts an explicit CPU+CUDA score comment. Do not deploy on neutral acknowledgments or non-eval engagement.

---

## TEMPLATE BODY (~50 words)

> Thanks for running the eval. CPU [contest-CPU SCORE] / CUDA [contest-CUDA SCORE] matches what I measured against the same archive on Modal Linux x86_64 + T4 within the expected drift band. Happy to share the substrate-design context that produced the FEC6 selector + fixed-Huffman-k=16 bolt-ons if that would help — there's a candidate inventory at https://github.com/adpena/comma-lab/blob/main/docs/asymptotic_floor_candidate_inventory.md and the sister tooling lives at https://github.com/adpena/tac.

---

## Variant adjustments

- **If maintainer-bot CPU score matches our `0.192051 [contest-CPU]` exactly**: open with "Thanks for running the eval — matches exactly."
- **If CPU score differs slightly (`±0.0001`)**: open with "Thanks for running the eval — `[their score]` vs my `0.192051` matches within the expected upstream-evaluator-vs-my-Modal-eval drift band."
- **If CPU score differs substantively (`> ±0.001`)**: omit the "matches" framing entirely; open with "Thanks for running the eval — [their score] is meaningfully different from my `0.192051` Modal measurement; let me reproduce on the same hardware path." (Investigate before further response.)
- **If only CUDA reported (no CPU)**: omit CPU clause; "CUDA [contest-CUDA SCORE] matches what I measured on Modal T4." Add "Happy to also run the GHA CPU eval myself if useful."

---

## Tone notes

- Brief. Single short paragraph.
- Technical, not promotional. State the match and the cross-links; do not push for engagement.
- Open offer of context (inventory + sister library link) without demanding response.
- No emojis. No "thanks for your time" / "looking forward to" closers. Match the maintainer's brevity from PR #56 / #95 / #100 / #101 / #102 / #103 / #108 comment threads.

---

## Discipline applied

- `user_pr_attribution.md` sole-author voice
- `feedback_forbidden_claude_attribution_in_public_pr_surfaces.md` zero-Claude
- CLAUDE.md "Apples-to-apples evidence discipline" (explicit axis tags `[contest-CPU]` / `[contest-CUDA]`; hardware substrate cited only when relevant to drift discussion)
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" (referencing the canonical paired-eval contract)
- CLAUDE.md "Strategic Secrecy" (sharing concrete cross-links rather than withheld claims)

---

## Related

- `pr_110_hnerv_fec6_fixed_huffman_k16_submitted.md` (canonical PR #110 record)
- `pr_submission_yousfi_non_merge_response_template_20260519T182635Z.md` (non-merge verdict sister templates)
- `response_to_substantive_yousfi_or_hotz_question_20260520.md` (sister Template B for substantive technical questions)
- `response_to_merge_or_nonmerge_verdict_20260520.md` (sister Template C for verdict-bearing comments)
