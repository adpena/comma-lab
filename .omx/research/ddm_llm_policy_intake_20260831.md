# CONTEST LLM-POLICY INTAKE — Yousfi added a "coding agents and LLMs policy" and enforced it TODAY (closed #137 + #138); this is the decisive new input to the #1111 submission decision

Date: 2026-08-31 · Author: MAIN · Cost: $0 (gh api reads)
Axis: intake/adjudication input. `score_claim=false` · `promotable=false`
Trigger: operator 2026-08-31 "Also search for any new PRs."

## 1. What changed upstream (verified via gh api this turn)

- **No NEW PRs since #138** (2026-08-17, already intaken by pq2/#1104).
- **Both #137 (metric_shift_av1) and #138 (opal_v1) were CLOSED by YassineYousfi personally
  on 2026-08-31 at 10:08Z**, 15 seconds apart, with a comment on #138 linking the policy:
  `https://github.com/commaai/comma_video_compression_challenge#coding-agents-and-llms-policy`.
- **The README now carries the policy** (inspired by Rust's LLM usage policy; verbatim core):
  - Allowed: "write, refine, check, suggest, review **parts** of the code" · "document,
    organize, answer questions, analyze information for personal and internal use."
  - **Banned: "write all of the code" · "write full PR description and public facing comments."**
  - Enforcement: "Any violation of this policy will result in a closed PR, repeated violations
    will result in a ban."
- **The challenge is STILL OPEN for submissions** ("Submit to get on the leaderboard, apply for
  a job/internship, or just for fun") — refines the 2026-07-06 "CONTEST CLOSED" note: prizes
  are awarded (winners §: PR101/PR103), but the submission/leaderboard channel remains live.

## 2. Bearing on #1111 (the pending submission, operator-confirm-gated)

Our candidate packet (rr4-lineage → rc2 swap, #1170) is agent-authored end-to-end — code,
archive, compression scripts, and (as drafted) the PR description. Under a plain reading of the
policy AND today's demonstrated enforcement (two closures within a minute), submitting the
current packet as-is would very likely be closed as a violation of BOTH banned clauses.
Honesty rules (NO-FAKE; public-attribution discipline) forbid misrepresenting authorship to
route around it.

**Options (operator decision — NOT adjudicated by MAIN):**
- (a) **No contest PR; publish independently.** The 2026-07-06 directive already makes our
  methods/writeup open-source-destined. Leaderboard placement forgone; full research record
  published on our own surface. Zero policy risk.
- (b) **Operator-authored submission:** operator personally writes the PR description + all
  public comments (cures the second banned clause) and decides how to represent code
  authorship honestly — but "write all of the code" likely remains violated on the facts;
  high risk of closure, ban on repeat.
- (c) **Ask first:** operator contacts the maintainer (Discord/issue) about whether a fully
  disclosed agent-built research submission is welcome outside job-application intent, citing
  the policy's stated purpose ("fun or applying for a job").

MAIN's honest read: (a) or (c). (b) as-is fails the policy on its face.

## 3. Consumers

- #1111 / p0_swap_procedure_no_push_without_confirm_20260817 — this memo is a REQUIRED input
  at the confirm boundary; the confirm ask must present the policy.
- pq-line packet work (pq11 frozen swap) — unaffected mechanically; the freeze-not-publish
  stance is vindicated.
- Frontier pointer — no leaderboard movement (neither closed PR was evaluated).

## 4. Denominator

Sources read: PR list (15 rows) · comments on #137/#138 · close-event timelines · README
(policy section + leaderboard section). New PRs found: 0. Policy sections found: 1 (new).
Pointer: UNMOVED.
