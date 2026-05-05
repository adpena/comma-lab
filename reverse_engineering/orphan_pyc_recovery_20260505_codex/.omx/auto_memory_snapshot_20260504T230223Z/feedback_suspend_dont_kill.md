---
name: Suspend Don't Kill — Keep All Experimental Lanes Open
description: Human overrides council's kill decisions. Suspend experiments, don't kill them. LoRA TTO stays alive.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
The human prefers to SUSPEND experiments rather than kill them. Killing closes a lane permanently. Suspending preserves the option to resume if new information arrives.

**Specific overrides (2026-04-20):**
- Council voted to "kill LoRA TTO" — OVERRIDDEN by human. LoRA TTO stays alive as an experimental lane.
- Council voted to "kill Phase 1 at loss < 0.5" — OVERRIDDEN by human. Suspend (checkpoint and move to Phase 2) but don't discard the Phase 1 state. Can resume later.

**Why:** The contest is about finding the OPTIMAL approach. Killing lanes prematurely is how labs fall behind. LoRA TTO is a genuinely novel idea (per-video low-rank adaptation stored as compact delta) that the literature survey validated (Cool-chic, LoRA for neural fields). The council's objection was complexity vs latent codes, not fundamental unsoundness.

**How to apply:**
- When the council votes to kill an experiment, present it as "suspended" unless the failure is mathematically proven
- Always checkpoint before transitioning phases (don't lose Phase 1 state)
- Keep the `experiments/train_lora_tto.py` script and `src/tac/lora.py` module active
- Consider running LoRA TTO after distillation as an alternative to latent codes

**Reminder:** The skunkworks council is INHERENTLY non-conservative and aggressive per its charter in CLAUDE.md. "Don't change working code" is NOT a valid argument. "Ship what we have" is NOT a valid argument. The burden of proof is always on NOT trying something, never on trying it. When the council votes conservative (e.g., "kill LoRA"), challenge them — they're violating their own charter. Bold, well-reasoned proposals should survive the Contrarian. Lazy consensus should not.
