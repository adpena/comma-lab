---
name: Recursive Senior Engineer Greenup Protocol
description: Every agent's code changes must pass recursive council review before committing. No exceptions.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The Protocol

When ANY agent returns with code changes:

1. **Commit the agent's work** to a descriptive commit immediately
2. **Spawn a council senior engineer review** of the changed files
3. **Fix all issues found** by the review
4. **Re-review** the fixes
5. **Repeat** until a review pass finds ZERO issues
6. Only then: launch training or deploy

## Why
- The MRS 10x bug (200 should have been 20) was caught by council review
- The SWA dead code was caught by council review  
- The profile architecture mismatch was caught by running code
- The CRF distribution shift was caught by auth eval
- Every bug costs hours of wasted GPU time

## When to skip
- Pure documentation/memory changes (no code)
- Trivial one-line fixes where the change is obviously correct
- Emergency hotfixes during active training (fix, deploy, review after)

## The standard for "clean"
A pass is clean ONLY if ZERO issues are found — not "fewer than 3."
If the Contrarian finds even 1 issue, it's not clean.
Three successive ZERO-ISSUE passes = greenlit for deployment.
The Contrarian should try their hardest to find issues every time.

**How to apply:** This is non-negotiable from here forward.
Every line of code that runs on expensive GPU time must be reviewed first.
