---
name: Review Loop Before Every Deploy
description: Every implementation must go through implement→review→repeat loop until zero issues before building or deploying
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
Always follow this sequence: implement → review → repeat until zero issues → build → deploy.

**Why:** The user explicitly corrected the proposed sequence of implement → build → deploy. Review is not optional and not a one-shot gate. It's a loop that repeats until the reviewers find nothing. This applies to every piece of new infrastructure, not just the training script.

**How to apply:** After implementing any council-mandated change or new module, launch a review agent before proceeding to the next step. Only move to "build" when the review returns clean. Only move to "deploy" when the build is also reviewed clean.
