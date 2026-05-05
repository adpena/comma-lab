---
name: Council Decides All Design Questions
description: Never make design decisions unilaterally — always consult the tripartite pact (Yousfi+Fridrich+Contrarian) first
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
Always ask the council about design decisions instead of deciding yourself.

**Why:** The user explicitly corrected me after I unilaterally changed the GT resize from bicubic to bilinear. Design decisions — especially ones that change training behavior — must go through the tripartite pact. I should identify the issue, present the options, and let the council decide.

**How to apply:**
- When a review identifies a design question (not a clear bug), present options to the council
- Do NOT implement the change before the council decides
- Clear bugs (crashes, wrong formulas, missing imports) can be fixed immediately
- Design tradeoffs (interpolation method, loss function choice, architecture config, boundary values) must be council-approved
- If unsure whether something is a bug or a design decision, ask the council
