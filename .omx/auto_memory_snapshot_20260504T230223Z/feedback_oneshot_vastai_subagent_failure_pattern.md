---
name: One-shot Vast.ai-eval subagent pattern fails repeatedly — use launch+return-early instead
description: 2026-04-27: 5+ recurring failure mode. Subagents told to do the FULL Vast.ai eval lifecycle (build → upload → SSH → run → capture → destroy) consistently bail out mid-flight, leaving instances burning. Lane G + Lane D subagents that launch-in-tmux-and-return-early consistently succeed. Use the latter pattern.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Verified 2026-04-27 across 5 separate subagent dispatches:**

| Lane | Subagent ID | Outcome | Wasted |
|---|---|---|---|
| Lane A+Brotli (1st) | a9885eccaaf111503 | Returned mid-flight ("container not yet pulled") | ~$0 (no real spend, but stuck) |
| Lane A+Brotli (2nd) | ade40a32b1db19ebe | Left orphan 35706650 ($0.10) | $0.10 |
| Lane G (1st) | a6d1d0571bc9b42f3 | Aborted correctly ($0 — KL not wired) | $0 |
| Lane H CRF56 | a5b051bb03ddd7f62 | Left idle instance 35708015 ($0.27, idle 50 min) | $0.27 |
| Lane A+Brotli (3rd, redispatch) | ade40a32... | Left idle 35707726 ($0.30, idle 56 min) | $0.30 |

**Total wasted on this pattern: ~$0.67 + 4-6h subagent-time confusion.**

**Common failure mode:**
1. Subagent SSH's into the instance.
2. Does some setup (rsync, env exports).
3. Returns control to parent with a "still in progress" message.
4. Never completes the actual workflow (run, capture, destroy).
5. Instance keeps burning at $0.32/hr.

**Working pattern (Lane G + Lane D):**
- Subagent's job is LAUNCH + verify health + report back.
- Launch tmux session that runs the workload.
- Verify probe passes + tmux alive + first 3 batches of telemetry.
- REPORT BACK to parent with: instance ID, telemetry, ETA, cost-burn.
- PARENT (me) monitors via SchedulewWakeup + fetches RESULT_JSON when done.
- PARENT destroys the instance at end.

This pattern has worked successfully:
- Lane D 5h half-frame retrain — landed launch + telemetry visible.
- Lane G — subagent wrote script + launched + reported KL imbalance + parent destroyed.
- Lane A original (the first 3.4h pose TTO) — landed cleanly.

**Why one-shot fails:**
- Subagent token budget is bounded (~200K).
- Vast.ai container pull on slow hosts is 5-10 min wall time.
- During that wait, subagent context burns on idle SSH attempts.
- By the time container is up, subagent has spent its budget polling.
- Returns "still in progress" without doing the actual work.

**Permanent fix:**
1. NEVER dispatch a subagent that does FULL Vast.ai eval lifecycle.
2. Subagent's only Vast.ai job: launch + verify health + report back.
3. PARENT handles monitoring + RESULT_JSON capture + destroy via SchedulewWakeup loops.
4. For HUMAN-ATTENDED runs, the operator can run the script manually instead of using a subagent.

**Cost paranoia rule** (CLAUDE.md addition candidate): every Vast.ai instance launched MUST have a kill-after-N-minutes watchdog OR a "destroy at end of subagent" must be explicit. If the subagent is dispatched in launch-and-return mode, the parent must scheduleWakeup-and-destroy.

**What NOT to do:** dispatch another one-shot Vast.ai-eval subagent. The pattern has failed 5/5 times in a single session. Use the launch+return-early pattern.
