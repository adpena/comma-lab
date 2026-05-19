# Yousfi Non-Merge Response Template — pr101_fec6_k16_clean Submission

**Issued:** 2026-05-19
**Purpose:** Pre-prepared response template for case where Yousfi closes our PR per the 2026-05-11 new-submission gate (PR #108 closure pattern). Per T3 grand council symposium 2026-05-19T180611Z op-routable #7.
**Operator-routable:** YES — use as starting point; tailor to specific reasoning Yousfi cites in the closure comment.
**Authority:** Not yet posted. This is preparation for case (B) of the 3-outcome cascade (merge / non-competitive close / modification request).

---

## Yousfi's binding gate (verbatim from PR #108 closure 2026-05-11T19:19:57Z)

> closing this pr per the new submission guidelines, the tricks used are already established in several past submissions
>
> 'is this submission competitive or innovative? explain why
> competitive: better than top # 1 submission
> innovative: it has a novel idea that is not on the leaderboard yet, might not be competitive, but has potential'

## Our position (already stated in PR body)

This submission satisfies BOTH criteria of the gate:

1. **Competitive:** `0.1920513169` `[contest-CPU]` improves on top-merged PR #102's reported `0.19538` `[contest-CPU]` by `-0.00333`.
2. **Innovative:** the FEC6 fixed-Huffman k=16 per-pair frame-exploit selector composition is not currently merged on the leaderboard.

## Pre-prepared response templates by closure category

### Category B1: closure cites "not better than top #1"

**Suggested response:**

> Hi @YassineYousfi, thank you for reviewing. I want to clarify the score comparison so we can align on the numbers:
>
> - This submission's headline CPU score is `0.1920513169` (Modal Linux x86_64 CPU reproduction).
> - Top-merged PR #102's reported CPU score is `0.19538` per the [PR #102 comment thread](https://github.com/commaai/comma_video_compression_challenge/pull/102).
> - Delta: `0.1920513169 − 0.19538 = −0.00333` (lower is better).
>
> If the bot has a different number for either archive, I'd appreciate seeing the host-bot CPU comment for this archive so we can identify the source of the gap. If the gap is from MPS vs Modal CPU vs GHA Linux CPU axis drift, I can re-run on a different substrate.
>
> Happy to provide additional verification on either archive if useful.

### Category B2: closure cites "tricks already established"

**Suggested response:**

> Hi @YassineYousfi, thank you for reviewing. The FEC6 composition does build on PR #101's HNeRV substrate and PR #103's composable selector pattern (cited in the PR body), but the specific composition here introduces:
>
> 1. **K=16 frame-conditional per-pair mode palette** (vs PR #101's K=8 static modes; vs PR #103's per-pair selector but on a different substrate axis).
> 2. **Fixed-Huffman codebook on selector indices** (a different entropy-coding approach than raw-byte storage in PR #101 GOLD; compacts 300 bytes naïve cost to ~107 bytes).
> 3. **Composition of these two over the HNeRV-ft substrate** — neither PR #101 nor PR #103 currently merged on the leaderboard ships this specific stack.
>
> If you would prefer this stack be evaluated as a comment / discussion thread rather than a merge, I am happy to convert it. If there is a specific past submission that already ships this exact composition, please point me to it and I will withdraw.
>
> Thanks for the contest!

### Category B3: closure cites "post-deadline; no new submissions accepted"

**Suggested response:**

> Hi @YassineYousfi, understood — thank you for the clarification on the post-deadline policy. I appreciate you reviewing.
>
> I'll keep this PR as a public-record reference of the composition for anyone working on follow-on entropy-coded selector designs over HNeRV-family substrates, but I understand it won't be merged or evaluated by the bot.
>
> If there is a future call for post-contest write-ups or a follow-up evaluation window, please consider this PR a candidate.
>
> Thanks for running such a well-designed contest!

### Category B4: closure cites "GHA runner busy; please defer"

**Suggested response:**

> Hi @YassineYousfi, understood — no rush, happy to defer.
>
> If it helps, I have paired CUDA + CPU axis reproductions on the same archive bytes (cited in the PR body); the CUDA axis number is `0.2262100217` `[Modal T4 CUDA replay]`. Both axes derive from the same archive.zip bytes (sha256 `6bae0201...`).
>
> Whenever the runner has time, I'm happy to provide additional verification.
>
> Thanks!

### Category B5: closure cites "modifications required" (specific request)

**Suggested response (template; tailor to the request):**

> Hi @YassineYousfi, thank you for the feedback. Re: [SPECIFIC REQUEST], I can [SPECIFIC ACTION]:
>
> - [Action 1]
> - [Action 2]
> - Re-submit with the modifications applied.
>
> Expected turnaround: [TIMEFRAME].
>
> Please confirm this addresses the concern, and I'll prepare a new PR.

## Internal follow-up actions (per category of closure)

| Category | Internal action |
|---|---|
| B1 (score gap) | Re-run paired CPU+CUDA auth-eval on a different CPU substrate (e.g. GHA Linux runner via fork PR); compare to host-bot result if available. |
| B2 (not innovative) | Document which sister archive ships the same FEC6 composition + recipe; if no such archive exists, request specific reasoning. |
| B3 (post-deadline) | DEFER; mark lane as `archived` per Catalog #298 retirement-discipline (research_only). |
| B4 (runner busy) | DEFER; no action required. Periodically check PR for runner status. |
| B5 (modifications) | Address the specific request; iterate per CLAUDE.md "Recursive adversarial review protocol — close paths". |

## Discipline + cross-references

- **CLAUDE.md "Executing actions with care":** the operator decides whether to post any response; this is a template only.
- **CLAUDE.md "Public Disclosure Hygiene":** no internal process language (no Catalog #, no Rudin-Daubechies, no cathedral autopilot, no subagent IDs); reads as authored by a human contestant.
- **CLAUDE.md "Council conduct":** the Contrarian's gate at op-routable #7 explicitly called for a courteous response template; this satisfies that gate.
- **Catalog #208 disclosure hygiene:** no local paths; no provider URLs (except the canonical hosted-archive URL which is public per D3 Option A); no credentials.
- **T3 grand council symposium 2026-05-19T180611Z op-routable #7:** "DEFERRED-to-operator after D5 fires: operator-facing acknowledgment that Yousfi MAY close the PR per the new-submission gate (2026-05-11 PR #108 precedent). Document expected outcome cascade: PR opens → maintainer reviews against gate → (A) merges + GHA eval runs + score recorded; (B) closes per non-competitive gate; (C) requests modifications. Have Slot F prepare a courteous response template for case (B)." — this template satisfies that prepare.

## What's NOT in this template

- Threats / demands / ultimatums.
- Internal process disclosure (cathedral autopilot, council voting, subagent IDs, lane registry).
- Score literals for sister archives (only PR #102 0.19538 + our 0.1920513169 + paired CUDA 0.2262100217 are cited; all are public-axis numbers).
- Promises of future submissions / contest engagement we cannot honor.
- Any request for compensation, sponsorship, employment, contracting, or collaboration with comma.ai.

## When to use this template

- ONLY if Yousfi (or another contest maintainer) posts a closure comment.
- After waiting at least 1 hour to confirm the closure is not auto-revoked or accompanied by additional explanation.
- After confirming the closure category (B1-B5 above) by reading the maintainer's comment carefully.
- Operator approves the response text before posting; this template is a starting point, not the final text.

## When NOT to use this template

- If the PR is merged or auto-eval succeeds — no response needed beyond a thank-you.
- If the PR is queued for eval (status pending) — wait for the eval to complete or for an explicit decision.
- If the maintainer asks a clarifying question — respond directly to the question, not via this template.

---

**Operator-routable:** review + approve + tailor before posting any response. This template is preparation, not authorization.

**Sister cite-chain:**
- `.omx/research/grand_council_t3_upstream_contest_compliance_conformance_symposium_20260519T180611Z.md` — the symposium memo that mandated this template.
- `.omx/research/pr_submission_check_in_package_20260519/PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md` — the PR body this template responds to.
- `.omx/research/pr_body_canonical_pre_submission_adversarial_review_20260519_codex.md` — the codex adversarial review of the PR body.
