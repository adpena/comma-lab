---
name: Don't abandon higher-score lanes — raw score ≠ stacking potential
description: 2026-04-28 user clarification after Lane G v3 1.05 frontier landed. Ranking lanes by RAW score is wrong for the stacking phase. A lane with raw 1.20 may compose better than a lane with raw 1.10 because of orthogonality. Continue iterating + experimenting on ALL lanes — some "worse" raw lanes are stack-essentials. The optimal stack is empirically determined, not predicted.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The rule

**Raw single-lane score ≠ stack value.** A lane that scores 1.20 raw but composes orthogonally with N other lanes may dominate a lane that scores 1.10 raw but conflicts with everything else.

Concrete examples from current portfolio:
- Lane V (Quantizr replica) — predicted raw [0.50, 1.10] BIG SWING; but if it lands at 1.10, stacking with Lane W + Lane MAE-V might produce ~0.50 stack
- Lane I (Cool-Chic) — predicted raw [0.95, 1.30]; if it lands at 1.20, stacking with Lane SI-V2 + Lane T2-MASK might get to 0.85
- Lane MOS — predicted raw [0.95, 1.10]; modest single-lane but UNIQUE pose-from-embedding mechanism
- Lane SAUG-V2 — predicted raw [0.70, 1.00]; but its real value is closing the proxy/auth gap so Lane Ω-V2 et al can stack

## How to apply

1. **Continue iterating on every dispatched lane** even after a lower-score lane lands. Don't pivot all GPU to "the winner".
2. **Stacking experiments require empirical results** — don't predict-then-pick. Run each candidate lane to see its actual landed score + decomposition (rate vs distortion components), THEN compose.
3. **Multiple lanes to "best raw score" target** are a feature, not redundancy. The portfolio gives the council options; some will dominate after measurement.
4. **Don't kill in-flight Vast.ai work** unless it's verifiably broken (no heartbeat for 30+ min). Higher-score lanes might land valuable diagnostic info even if they can't be the new frontier alone.

## CRITICAL ADDENDUM (2026-04-28 user clarification)

**The first landed score is NOT the lane's true potential.** Every lane needs MULTIPLE ROUNDS of audit + debugging + engineering rigor before its final score is known.

Examples of past lanes that improved on iteration:
- Lane F V1 → V2 (bug 2 fix): 2.73 → 1.79 (-0.94)
- Lane G v1/v2 → v3 (KL weight bug 14000× fix): killed → 1.05
- Lane M+N V1 → V2 (1-DOF save bug + baseline padding): 2.35 → in flight
- Lane D V1 → V3 (LR-starvation hypothesis → KL-distill drowning fix): killed → in flight

A lane scoring "1.30 raw" might be:
- A bug (silent default, missing flag, simulated-FP4-not-hardware-FP4) that fixes to 0.90
- A misconfiguration (wrong profile, wrong anchor) that fixes to 1.10
- A genuine architectural ceiling (rare; usually fixable)

**Default assumption for any landed score above 1.0**: it's not the floor — there's a bug or missing engineering rigor. Council should default to "audit + debug + iterate" before "accept and abandon".

## Why this matters now (2026-04-28)

Lane G v3 just landed at 1.05 [contest-CUDA] (memory `project_lane_g_v3_landed_1_05_20260428`). The instinct is to:
- Pivot all attention to Lane G v3 stacks (Lane G + Lane W, Lane G + Lane SAUG, etc.)
- Mentally write off Lane V, Lane I, Lane M-V2 as "worse than 1.05"

That's wrong. Lane V is still in flight at 4.6h on Vast.ai (35733832, 47% util). It might land at 0.70 (Quantizr-replica architecture) and become the new anchor. Lane I might land at 1.15 but compose orthogonally with everything. Run them ALL.

## Council non-conservative principle reinforcement

CLAUDE.md "Council conduct — non-negotiable":
> The only valid arguments are mathematical, scientific, geometric, or empirical.

"This lane's raw score is worse than the new frontier" is a SCORE argument, not a STACK argument. Council should debate stack potential separately from raw score.

## Cross-references
- `project_lane_g_v3_landed_1_05_20260428` — the new raw-score frontier
- `project_lane_taxonomy_stacking_strategy_20260427` — full lane composition analysis
- `project_outstanding_work_and_stacks_20260428` — TIER 3 catalog
- `project_cosmos_deep_dive_addendum_20260428` — orthogonality analysis (Pattern 4 + 5)
