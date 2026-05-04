---
name: post-deadline continuous-engineering mandate (no-meat-left-on-bone, no-resource-constraint)
description: User directive 2026-05-02 — keep all lanes alive past contest deadline, no time/money constraint, extreme rigor, submissions/writeup/site continuously updated post-deadline as investment in better results that compound into better contest outcomes.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
User directive sequence 2026-05-02 (rapid succession during C-067 ship-prep):

1. "kepe pushing outside the deadline window too"
2. "we will update our writeup and results and submissions online even after the deadline"
3. "keep it all alive and push all extreme rigor"
4. "no meat left on the bone and time and money are no issue"
5. "we are on track to win and we can spend the winnings we will get on better results that will enhance our likelihood of winning"

**Why:** Capital-recycling framing — each post-deadline sub-frontier improvement increases expected payout of the next contest cycle (or this cycle's tiebreakers / runner-up considerations / writeup-driven recognition). It's the engineering equivalent of compounding. Cost-of-experimentation shifts from "sunk" to "investment".

**How to apply:**

1. **Council KILL verdicts that were deadline-window-scoped now reactivate post-deadline.** Specifically: the 2026-05-02 Council 5/0 SHIP-C-067 verdict (`project_grand_council_lane_12_nerv_unblock_decision_20260502.md`) KILLED Lane 12 NeRV "for the 46h deadline window" only. Post-deadline, Lane 12 Path B (parser + Alpha-Geo contracts + L2 clearance JSON + tests + 3-pass review) is LIVE active workstream. Also active: Quantizr's #4 recipe (KL-soft-distill T=2.0 against C-067's deployed AV1 mask logits, NOT raw argmax). Memory: `project_grand_council_lane_12_nerv_unblock_decision_20260502.md`.

2. **All deferred sub-0.3 ambitious lanes reactivate.** Block-FP transplant, joint training, Self-Compress NN, MDL/Bayesian, multi-paradigm stack composition, NeRV-stacked-on-C-067, Quantizr recipe #4, Pint12-PCA, etc. None of these are killed — only deadline-window-deferred.

3. **Subagent cap relaxes from "1-2 max" to ~3-4 max** under the no-resource-constraint mandate. Anthropic session quota IS still a real constraint (rate limits don't help anyone), but the prior cost-discipline cap is dropped. Default mode shifts from "1 subagent unless 2 needed" to "3-4 subagents in parallel for high-throughput dispatch + 1-2 research-slot for council/audit".

4. **C-067 packet remains immutable as the historical-deadline submission record.** Append-only: each new sub-frontier archive lands a new `experiments/results/submission_packet_<id>_<date>/` directory.

5. **No "lane killed" verdicts in the registry mean "permanently dead"** — all kills are deadline-window-scoped and reactivate post-window unless an empirical-falsification has been recorded with full council review per `feedback_grand_council_imp_permanent_fix_review_20260430` standard (3 sections: council positions + internal-consistency check + what-would-change-my-mind).

6. **Public-facing artifact updates are continuous.** writeup (`docs/paper/`), site, GitHub PR description, leaderboard submission may be continuously updated as the post-deadline measurement cadence produces new sub-frontier results. The leaderboard submission mechanic + cadence is contest-organizer dependent and must be checked for each update. Strategic Secrecy Rule still applies: don't publish unpublished secret sauce until user explicitly approves.

7. **EXTREME RIGOR remains non-negotiable.** Looser resource constraint does NOT mean looser engineering rigor. Council reviews, 3-clean-pass gates, empirical verification, lane maturity gates — all stay in force. The mandate is "more rigor with more resources," not "skip rigor because we can afford rework."

8. **Every dispatch still needs council sign-off + specific approval per CLAUDE.md "Design decisions — non-negotiable."** Aggregate "no time/money constraint" mandate does NOT auto-approve any specific dispatch. Surface concrete options with predicted ROI; user approves specific dispatches.

Cross-references:
- `project_grand_council_lane_12_nerv_unblock_decision_20260502.md` — Council 5/0 SHIP-C-067 verdict + dissent positions
- `project_lane_owv3_0120_orthogonal_stack_LANDED_0_997_20260501.md` — earlier sub-1.000 frontier
- `project_leaderboard_0_32_0_33_floor_or_irrelevant_20260501.md` — competitive landscape (now deadline-relevant; post-deadline target is sub-0.30)
- `feedback_fast_chip_directive_no_waiting_20260501.md` — fast-chip mandate (H100/A100 SXM preferred, RTX 4090 OK for <1h)
- `feedback_grand_council_imp_permanent_fix_review_20260430.md` — KILL-verdict canonical-example standard (3 sections required)
