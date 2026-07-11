# Papers checked — "Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents" (2026-07-11)

Anti-re-research ledger row (operator + coordinator routed, folded same-day).

| field | value |
|---|---|
| paper | Wu et al., *Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents*, arXiv **2607.08716** |
| headline | SELECTIVE INTERVENTION (a memory/control agent alongside the action agent, injecting only when the trajectory state calls) OUTPERFORMS always-on injection, passive-bank exposure, advisor-only guidance, and general retrieval: **+8.3pp Terminal-Bench, +6.8pp τ²-Bench** (MEASURED-elsewhere). Intervention policy is RL-learned (GRPO). "Behavioral state decay" named: critical context buried in expanding trajectories → proactive selective recall > passive retrieval. |
| our fold (measured same-day) | (1) **#430**: the coherent curriculum is shaped as a selective-intervention (state-triggered) policy and BACKTESTED selective-vs-always-on-vs-hand on the #205 replay (`tac.witness_control.schedule_backtest`): organ cascade beats the hand schedule −27% ∫d_seg·dep on the WF-winning model; selective ≈ always-on IN-MODEL on the transient-only prefix (gate question not in-model-resolvable — this paper's external measurement carries it, cited as such). (2) **organ architecture**: the paper's structure (control agent alongside action agent, deciding WHEN+WHAT to inject) IS the costate organ ↔ witness relationship; validates the spread-gate/System-2-on-disagreement + PowerPlay duty-to-measure "act only when necessary" discipline. (3) "behavioral state decay" = the #411 reconstructable-graph-memory / retrieval-first #346 concern, externally measured. (4) RL-learned when-to-intervene composes with the GEPA/SAO RL stage — backtest-gated, duty-queued. |
| status | FOLDED (memo `scorer_model_arms_430_schedule_20260711.md`; ticket + replay artifact carry the citation). Not reproduced independently; the paper's numbers are MEASURED-elsewhere, ours are our own replay. |

**Pointer 0.19108282 [contest-CPU] UNMOVED.**
