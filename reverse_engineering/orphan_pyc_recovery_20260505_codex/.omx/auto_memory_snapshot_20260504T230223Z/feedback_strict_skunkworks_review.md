---
name: Strict Skunkworks Review for All New Implementations
description: All new code from here on must pass the strictest review and approval by the skunkworks team (Yousfi+Fridrich+Contrarian) before merge or deploy
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
All implementations from here on must be subject to the strictest standards of review and approval by the skunkworks team.

**Why:** The user explicitly mandated this after the paranoia sweep revealed 28 bugs in previously "reviewed" code. The bar is higher now. No more "good enough" — every new module must be council-approved before it touches the main branch.

**How to apply:**
- After building any new module (renderer_export.py, inflate_renderer.py, etc.), launch a full tripartite review before committing
- The review must cover: correctness, gradient flow, numerical stability, training/eval consistency, rate calculation, round-trip exactness
- No janky smoke tests — tests must be at representative resolution with enough signal to validate
- A test at 1/4 res for 500 steps CANNOT kill a technique. Bias toward keeping lanes open.
- The implement→review→repeat loop runs until ZERO issues, not "good enough"
