---
name: Reviews Must Check STRATEGY, Not Just Bugs
description: Code can be bug-free but strategically wrong. Reviews must challenge design decisions and training recipes, not just syntax and crashes.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
The distillation run used --skip-phase1, which is valid code that runs without errors. But it was the WRONG STRATEGY — skipping Phase 1 meant the FiLM renderer trained from a cold start with noisy scorer gradients instead of warm-starting from TTO-aligned pixel regression. This wasted ~$0.70 and 2 hours.

**Why paranoia didn't catch it:**
- Reviews focused on: crashes, wrong values, missing imports, index arithmetic, default overrides
- Reviews did NOT challenge: "is --skip-phase1 the right choice?" "is the training recipe optimal?"
- The code was CORRECT. The STRATEGY was wrong.

**How to apply:**
- Every deployment review must include a STRATEGY section: "Is this the RIGHT experiment to run?"
- Check: are ALL available techniques enabled? If not, WHY is each one disabled?
- Check: is the training recipe optimal? Warm-start, learning rate, loss weights?
- Check: does the experiment design match the council's binding decisions?
- A review that says "code looks clean" but doesn't question the experiment design is INCOMPLETE
- The council recommended Phase 1 warm-start. We deployed with --skip-phase1. Nobody caught the contradiction.
