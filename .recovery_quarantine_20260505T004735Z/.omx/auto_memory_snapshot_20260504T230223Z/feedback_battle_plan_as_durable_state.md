---
name: Battle Plan IS the Endgame Durable State — Read and Write Every Turn
description: Non-negotiable — docs/BATTLE_PLAN.md (no date) is the canonical endgame state. Read at start of every task, update at end.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
The battle plan document at `docs/BATTLE_PLAN.md` (NOT dated — it's a living document) is the CANONICAL endgame durable state.

**Why:** The battle plan contains the current scores, the three-lane strategy, the exact deployment sequence, the kill criteria, the budget tracking, and what's next. It is the single source of truth for where we are and where we're going. Every agent and every task should read it first and update it when done.

**How to apply:**
- Name the file `docs/BATTLE_PLAN.md` (no date suffix — it's always current)
- READ it at the start of every conversation, every task, every agent dispatch
- WRITE to it at the end of every task that changes scores, discovers bugs, or shifts priorities
- It replaces scattered state across `.omx/state/current_focus.md`, `next_experiments.md`, and session handoffs
- It should contain: current scores (both lanes, labeled), deployment queue, budget remaining, kill criteria, red flags
- It is the FIRST thing a fresh agent reads and the LAST thing it updates
