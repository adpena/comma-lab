# Yousfi review-behavior census — the bar our PR must clear

`date_utc: 2026-08-20` · `owner: ddm_pq7` · `score_claim: false` · `frontier_moved: false`

**Own-vehicle frontier: S = 0.14839100138338618 @ 180,625 B `[contest-CUDA T4, n600]`.** Unmoved here.

## Method and scope

`pq1` owed a live census; `api.github.com` was unreachable on 2026-08-15. **It is reachable now.**
I fetched PRs #60–#140 with `gh pr view <N> --repo commaai/comma_video_compression_challenge
--comments --json number,title,author,state,body,comments,reviews` (81 threads retrieved, 0 fetch
failures) and extracted every comment and review authored by `YassineYousfi`: **76 comments.**
Raw JSON retained at `_census_raw/pr*.json`. I also fetched the **live** `README.md` from
`master` and diffed it against our pinned snapshot.

This supersedes the retained-only bound in `CONTRIBUTION_ETIQUETTE.md`. Quotes below are from the
live API, ≤15 words each, attributed by PR number and date.

---

# ⛔ THE FIVE THINGS HE MOST CONSISTENTLY DEMANDS

## 1. Do not duplicate code that is already merged — this is THE merge gate

Three PRs, **all within 47 seconds on 2026-07-12**, got the identical verdict. This is a standing
policy, not a one-off reaction.

> "the code has too much overlap with code that is already merged"
> — YassineYousfi, **#125**, **#127**, **#128**, 2026-07-12 (verbatim in all three)

> "we won't merge it unless you refactor to reuse bits and pieces that are already there"
> — same three comments

**What makes this the operator's model of the bar:** all three PRs were *competitive* and all three
*shipped a compression script*. #128 (0.187946) beat the then-leader and its author was **sent an
internship email in the same comment** — and it was still refused merge. Score and script are not
sufficient. Non-duplication is the binding constraint.

Each was still added to the leaderboard ("added to leaderboard, congrats"), and each was told
"feel free to reopen". **Leaderboard ≠ merge.** Refusal is reversible by refactoring.

## 2. Comply with the coding-agents and LLMs policy — it is NEW and our pinned snapshot predates it

Four PRs in one day were closed with nothing but a link to the policy:

> "see https://github.com/.../README.md#coding-agents-and-llms-policy"
> — YassineYousfi, **#133**, **#134**, **#135** (2026-08-07), **#136** (2026-08-07)

The live policy text (`README.md` §"coding agents and LLMs policy", absent from our pinned
`upstream/README.md`) reads in part:

> "If you're not writing and reading most of the code you are submitting, then what's the point?!"

> **banned uses** — "write all of the code" · "write full PR description and public facing comments"

> "Any violation of this policy will result in a closed PR, repeated violations will result in a ban."

**The accepted remediation is receipted.** #135 was closed under the policy, re-described, and then
accepted onto the leaderboard on 2026-08-08. The two comments that define what "acceptable" means:

> "you have to show that there was some human work" — #135, 2026-08-07

> "We don't need more verbose, we need more precise." — #135, 2026-08-07

And he gave the **literal PR-body template** he wants:

> "**THIS** is the baseline submission and score…" — #135, 2026-08-07
> "**THIS** file/function/etc was changed to do **THIS** instead and achieved score…"
> "Optionally: **THIS** didn't work better…"
> "Optionally: **THIS** is my llm setup and prompts…"

## 3. Host the archive outside the repo, and keep the repo lightweight

Five separate PRs, one consistent request:

> "can you host the zip file outside of the repo?" — **#67**, 2026-05-03
> "can you link the zip file?" — **#73**, 2026-05-03
> "please host the assets somewhere else, I would like to keep the repo lightweight" — **#74**, 2026-05-03
> "can you host them somewhere else? I would like to keep the repo lightweight" — **#71**, 2026-05-03
> "please move the zip file outside of the pr" — **#102**, 2026-05-04

He accepts a GitHub comment attachment: he posted the drag-and-drop link himself on #102.

## 4. Every added byte must pay for itself, and the idea must not already be on the leaderboard

> "too micro optimized, the files and arrays introduced are larger than the bytes saved"
> — YassineYousfi, **#121**, 2026-07-03

> "closing as these methods are already well represented in the leaderboard" — **#117**, **#120**, 2026-06-24/29

> "the tricks used are already established in several past submissions" — **#108**, 2026-05-11

On what *does* clear the novelty bar, from #100:

> "many submissions picked up your perturbation trick, you can argue that it's … novel"
> — **#100**, 2026-05-05

## 5. Keep evaluation disputes public, and accept his hardware call

> "I ran all submissions in t4 for a fair comparison" — **#95** and **#97**, 2026-05-04

> "the score has a physical meaning that depends on the ground truth video" — **#103**, 2026-05-05

> "trying to influence things privately is not the way to do so" — **#103**, 2026-05-05

> "the more performance squeezing happens the bigger effect small pixel differences make" — **#103**

---

# What he PRAISES (the positive signal)

| Quote (≤15 words) | PR | Date |
|---|---|---|
| "we are going to reward folks publishing their code even if not in top 3" | #95 | 2026-05-04 |
| "great submission and write-up!" | #95 | 2026-05-04 |
| "nice visualizations!" | #71 | 2026-05-03 |
| "great write-up!" | #86 | 2026-05-04 |
| "Thanks for the amazing write-up." | #71, #86, #90, #97, #105 | 2026-05-05 |

He also proofreads: he flagged a one-word typo in a write-up title (#95, 2026-05-05). **Prose
quality is read.**

# What he FLAGS

| Quote (≤15 words) | PR | Class |
|---|---|---|
| "the code has too much overlap with code that is already merged" | #125/#127/#128 | duplication — merge blocker |
| "can you add your deps in your submission folder instead of project level?" | #74 | repo-wide dependency mutation |
| "the files and arrays introduced are larger than the bytes saved" | #121 | net-negative byte accounting |
| "this write up only reports on 4 negative experiments" | #118 | insufficient evidence for the conclusion |
| "I am happy to merge the write-up … but only if you fix the conclusion" | #118 | over-claimed conclusion |
| "yeah gaming the scoring script details is pretty easy" | #87 | scorer gaming |
| "closing for now, please re-open when ready" | #116 | incomplete submission |

**#118 is the closest analogue to our own risk profile** — a rigorous negative-results write-up
closed because the *conclusion* over-reached the evidence, not because the work was bad. He
separated the artifacts cleanly: "No need to have the code here but the description of the
experiments have to be improved" (#118, 2026-06-29).

# What he asks FOR

- The current template, filled literally: "can you update the pr with the new template" (#110, 2026-05-25),
  including "an easy to understand response to" the competitive-or-innovative question.
- A precise, human-authored, baseline→change→score description (#135, above).
- Method identification: "is the downscale with Lanczos? Spline?" (#116, 2026-06-24).

---

# Live-README deltas our pinned snapshot does not carry (all verified by diff)

`upstream/README.md` (pinned, 2026-04-13) vs live `master` (fetched 2026-08-20):

1. **NEW §"coding agents and LLMs policy"** — see item 2. Absent from our pin.
2. **The merge promise was DELETED.** Pinned: *"If your submission includes a working compression
   script, and is competitive we'll merge it into the repo."* Live: *"Open a Pull Request with your
   submission and follow the template instructions to be evaluated."* **Merge is now discretionary
   and the duplication rule is how that discretion is exercised.**
3. **Runtime is now the submitter's choice.** Pinned: *"If your inflation script requires a GPU, it
   will run on a T4 … if it doesn't it will run on a CPU instance."* Live: **"Pick your runtime:
   github's 'linux-nvidia-t4' GPU instance (RAM: 26GB, VRAM: 16GB) or github's 'ubuntu-latest' CPU
   instance (CPU: 4, RAM: 16GB)."** This materially de-risks `FREEZE_CHECKLIST` item (b): the
   T4 routing is ours to declare, not the maintainer's to guess.
4. **Live leaderboard head:** 0.162 (#135) · 0.166 (#133) · 0.172 (#130) · 0.187 (#128).
   Our 0.14839100138338618 would lead by 0.0136.

**MEASURED precedent for our runtime:** the eval bot ran **#130 and #133 on `device=cuda`,
`num_threads=2`**. Our body is the CPR1 descendant of exactly those. CUDA routing for this lineage
is established practice, not a hope.

---

# Cross-check: do our #402 receiver-hardening defects trace to Yousfi comments on #128?

**No — and the packet must not imply they do.** The charter asked me to check whether our #402 work,
derived from a "PR128 §8.2 defect list", traces to Yousfi comments on that PR. **PR #128 has exactly
three comments: two from the `github-actions` bot and one from Yousfi.** That single Yousfi comment
is the duplication verdict quoted in item 1. **He posted no defect list on #128.** Any "PR128 §8.2"
defect list is *our own* reverse-engineering of PR #128's source, not maintainer feedback.

Consequence: we can claim we cleared defects *we found in #128's code*; we may **not** present them
as maintainer-raised. The one thing Yousfi actually said on #128 is the duplication rule — and that
is the item our packet must demonstrate it clears.

---

# The bar, as a checklist for our packet

| # | Demand | Our state | Verdict |
|---|---|---|---|
| 1 | No duplication of already-merged code | 0 byte-identical copies of any merged submission; CPR1 lineage is **not merged** so it cannot be duplicated (see `DUPLICATION_AUDIT.md`) | **PASS, with attribution owed** |
| 2 | Coding-agents policy: human writes/reads most code; human writes the PR description | Packet drafted by agents end-to-end | **⛔ BLOCKING — unresolved** |
| 3 | Archive hosted outside the repo | HOLD pending operator-authorized hosting | Blocked on operator, correctly |
| 4 | Added bytes pay for themselves | Archive is 180,625 B vs #135's leader; net −0.0136 S | **PASS** |
| 5 | Runtime declared | `inflate.py` auto-selects CUDA; PR body must say `linux-nvidia-t4` | **SHOULD — state it explicitly** |
| 6 | Template answered literally, precisely, baseline→change→score | Body is long-form prose, not the #135 4-bullet shape | **SHOULD — restructure** |
| 7 | Deps inside the submission folder, not project level | No repo-level dependency mutation | **PASS** |
| 8 | Conclusion matched to evidence (#118 lesson) | Wall-clock WARN/REFUSE band unreconciled and disclosed | Honest; resolve before publish |

**Item 2 is the one that closes a PR on sight.** It is not an arm's call to resolve; it is the
operator's. Nothing in this file is a submission action.
