---
name: NEVER Conflate Contest-Compliant vs Unlimited-Compute Scores
description: HARD BLOCKER — two lanes produce different scores. Never mix them. Contest-compliant is priority. Label everything.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
There are TWO distinct score lanes. They MUST NEVER be conflated, confused, or mixed.

**Lane 1: Contest-Compliant (PRIORITY)**
- Goes through inflate.sh → inflate_renderer.py → evaluate.py
- Must complete within 30 min total on T4 (inflate + scoring)
- Archive must contain ALL artifacts used at inflate time
- NO TTO at inflate time (takes ~3 hours, far exceeds 30 min budget)
- Score = renderer-only output = ~0.87 auth (current best)
- This is what gets submitted. This is what counts.

**Lane 2: Unlimited Compute (Research/Paper)**
- TTO optimization at compress time, bake results into submission
- Or: TTO at inflate time with unlimited time budget
- Scores: v5a auth 0.43 (output MSE), v5b auth 0.41 (embedding)
- For the paper's scalability story, NOT for contest submission
- Shows what's possible with more compute

**Why:** Previous sessions confused TTO auth scores (0.41-0.43) with contest-compliant scores. The TTO scores bypass inflate.sh entirely. They cannot be achieved within the contest's 30-min inflate budget. Claiming 0.41 as "our score" when the contest pipeline produces 0.87 is scientifically dishonest.

**How to apply:**
- EVERY score must be labeled: `[contest-compliant]` or `[unlimited-compute]`
- NEVER say "our score is X" without specifying which lane
- The submission PR uses ONLY contest-compliant scores
- The paper can show both, clearly labeled, in the scalability section
- When optimizing, always ask: "does this help Lane 1 (contest) or Lane 2 (research)?"
- Lane 1 improvements (faster inflate, better renderer, lighter TTO) are PRIORITY
- Lane 2 improvements are valuable for the paper but NOT for winning
