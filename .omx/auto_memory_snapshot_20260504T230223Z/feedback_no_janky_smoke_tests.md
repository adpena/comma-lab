---
name: No Janky Smoke Tests — Experiments Must Produce Actionable Signal
description: Smoke tests must be designed to Yousfi/Fridrich standards. No toy experiments that produce misleading kill/promote decisions.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
CRITICAL RULE: Smoke tests must be designed to produce ACTIONABLE signal — enough fidelity to promote, kill, or flag for deeper investigation.

**What went wrong (2026-04-12):** A SIREN smoke test ran at 1/4 resolution for 500 steps without mask conditioning, then declared "SIREN is dead." This was not what the council designed. The test was too simplified to produce valid signal. A kill decision on data that thin is malpractice.

**Yousfi and Fridrich's standard:**
- Smoke test must be faithful to the ACTUAL proposed experiment design
- Must use representative resolution, step count, and conditioning
- Must test the actual hypothesis, not a strawman version
- Bias toward keeping lanes open until CONCLUSIVELY closed
- No premature kills based on undertrained/misconfigured tests
- Yousfi and Fridrich have final say as steganalysis experts

**How to apply:**
- Before running ANY experiment: have the council review the exact config
- "Is this what you would run? Does it test the real hypothesis?"
- Experiments must use enough steps/resolution/data to produce clear signal
- A negative result on a janky test means "test was janky," not "technique is dead"
- Every experiment has a pre-registered hypothesis + success criteria + kill criteria
- No GPU time wasted on tests that can't produce conclusions
