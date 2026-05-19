# Codex Routing Directive: PR Submission Body Pre-Submission Adversarial Review

**Issued:** 2026-05-19
**Source slot:** PR-PRE-SUBMISSION-CANONICALIZATION-AND-VALIDATION (Claude subagent `pr_pre_submission_canonicalization_20260519`)
**Operator directive verbatim:** *"we should check in together prior to final submisison of the new PR; we will want to talk through it together and also have codex review it, and we also already have some PR body draft work done somewehere we can start from perhaps but eneds to be canonicalized and hardened and tested and recursively reviewed and auth eval duplicated and proved again for ultimate confidence so we don't embarass ourselves"*
**Target codex review surface:** the canonical PR body draft at `.omx/research/pr_submission_check_in_package_20260519/PR_BODY_CANONICAL.md`

## Authority

- `score_claim=false` — this is a pre-submission review request, not a score claim itself
- `promotion_eligible=false`
- `ready_for_submission=false` — explicitly: PR is NOT submitted; operator check-in required before submission
- `ready_for_provider_dispatch=false`

## Request

Codex: please run an adversarial senior-engineering taste review on `.omx/research/pr_submission_check_in_package_20260519/PR_BODY_CANONICAL.md` from the perspective of:

1. **Skeptical comma.ai-style senior reviewer** (the maintainer who would decide whether to merge / score this packet)
2. **Yousfi-like contest-review lens** (axis labels, artifact custody, score discipline must be impossible to misunderstand)
3. **Hotz-like engineering-taste lens** (cut ceremony; the bytes, the command, the score; nothing else on first screen)
4. **Production-mindset lens** (comma.ai is a production company; how does the PR body read to someone optimizing for trust per minute?)

Your prior reviews:
- `.omx/research/pr_body_senior_engineering_taste_review_20260517_codex.md` (full senior-engineering review of the long draft at `docs/pr_writeups/cpu_frontier_fec6_20260517.md`)
- `.omx/research/fec6_cpu_frontier_submission_surface_adversarial_review_20260517_codex.md` (submission-surface compliance review)

Confirm whether the canonical draft has addressed all P0/P1 blockers from those reviews:

- **P0 — Headline CPU claim axis labeling** (was `[contest-CPU GHA Linux x86_64]`; now `[Modal Linux x86_64 CPU reproduction; contest/GHA host validation pending]`)
- **P0 — Submission gate claim consistency** (no claim of "submission compliance passing" without compliance JSON evidence)
- **P0 — Reproduction + provenance one runnable path** (one archive path, one inflate path, one canonical command)
- **P1 — CUDA comparison score-direction** (lower-is-better correctly applied; we describe `0.22621` vs PR101 GOLD `0.22936` as a CUDA improvement, but we explicitly do NOT promote it as the primary axis)
- **P1 — CPU/CUDA mechanism causality** (no "fundamental crux" overclaim; observed-split with paired hashes)
- **P1 — Internal process language stripped** (no Catalog numbers, cathedral autopilot, Rudin-Daubechies, hiring/funding appeal, internal nicknames)
- **P0 — Hidden-better-score contradiction** (no `~0.171` reference; that figure was a proxy/theoretical floor and is removed from the public body)
- **P0 — Employment/sponsorship ask** (removed; replaced with single closing sentence "Happy to discuss engineering details...")

Additional review axes (per CLAUDE.md "Recursive adversarial review protocol — close paths" + "Submission PR gate — non-negotiable"):

1. **Does the body overstate evidence anywhere?** Specifically: does any claim require evidence we don't have / haven't validated?
2. **Are the reproduction commands runnable as written?** Specifically: would a comma.ai reviewer running on a fresh Linux x86_64 checkout get the headline number without additional steps?
3. **Are limitations honest?** Specifically: is there any limitation we should be disclosing that isn't already in §5?
4. **Public-Disclosure-Hygiene per CLAUDE.md:** Does the body leak local absolute paths (`/Users/adpena/...`), private infrastructure URLs, unpublished operator state, or account metadata?
5. **Apples-to-apples evidence discipline per CLAUDE.md:** Does every numerical claim carry an axis tag + hardware substrate tag + archive sha tag where appropriate?
6. **The assumption-challenge axis per CLAUDE.md Recursive review protocol item 8:** What shared assumption is the submission body operating within (e.g., "comma.ai will validate the Modal CPU number as a contest-CPU host-bot run") and would violating that assumption invalidate the claim?

## Expected codex output format

A short adversarial review at `.omx/research/pr_body_canonical_pre_submission_adversarial_review_20260519_codex.md` with:

1. **Verdict**: APPROVE / APPROVE_WITH_REVISIONS / REFUSE
2. **Per-blocker status** from the P0/P1 list above (resolved / regressed / still-open)
3. **New findings** (if any)
4. **Operator-routable recommended actions** (if APPROVE_WITH_REVISIONS or REFUSE)

## Cross-context

The PR is NOT submitted. Operator check-in REQUIRED before submission per operator's 2026-05-19 directive. Slot 27 (`operator_admin_bundle_20260519`) was in-flight at time of this directive and intended to land a "PR submission DEFER report"; this directive supersedes slot 27's PR-submission scope. The frontier itself (`0.19205 [contest-CPU]` archive `6bae0201fb08...` + `0.20533 [contest-CUDA]` archive `9cb989cef519...`) is canonical per `tac.canonical_frontier_pointer` last_refreshed 2026-05-19T13:26:11 UTC + Catalog #316 frontier alignment = 0 drift.

## Authority statement

This request does not consume new GPU spend, does not modify any canonical posterior, and does not constitute a score claim. It requests adversarial review of a candidate PR body before the operator decides on final submission.
