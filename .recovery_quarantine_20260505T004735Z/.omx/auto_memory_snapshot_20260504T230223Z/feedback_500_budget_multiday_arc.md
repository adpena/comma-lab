---
name: BUDGET = $500 over multi-day iteration arc — push for absolute best score
description: User authorized 2026-04-27: $500 Vast.ai credit limit (NOT the historical $24 cap). Multi-day iteration over many experiment versions for absolute best score before May 3 deadline. Council rigor still required per session.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Binding update (2026-04-27, supersedes all prior $24 budget references):**

- Total Vast.ai budget: **$500** (was $24).
- Time horizon: 7 days to May 3 deadline.
- Iteration budget: ~$70/day average. Can run 3-5 instances in parallel.
- Per-experiment $ ceiling: ~$10-20 acceptable for high-EV bets (not $1).
- ~50-100 individual experiment runs feasible.

**How to apply:**

1. **Don't over-optimize for $0.10-0.20 savings.** A $1.50 experiment that produces a verified score is cheap.
2. **Run experiments in parallel** — 3 simultaneous instances at $0.30/hr = $0.90/hr is fine for 6+ hours.
3. **Iterate.** Each experiment is one of MANY versions. If iteration N fails, plan N+1 with the learnings — don't abandon entire technique on first failure.
4. **Council rigor still mandatory.** Even with $500, every experiment must:
   - Pre-registered hypothesis + predicted score
   - Single-variable change
   - Smoke check before full commit
   - Auth-eval via `experiments/contest_auth_eval.py`
   - Documented provenance (gpu, sha, git commits)
5. **No more "stop, ask user, wait" before every spend.** With $500 the user wants forward motion. Save the user-confirmation gate for: (a) >$5 single-experiment commits, (b) committing a final submission archive, (c) when results contradict hypothesis.

**Session pattern shift:** Stop the timid $0.20-at-a-time pattern. Plan a daily slate of 2-3 parallel experiments, dispatch them, monitor, synthesize results, plan next day's slate. The rigor is in the EXPERIMENTAL DESIGN, not in the budget conservatism.
