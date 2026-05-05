---
name: Never Bypass Review Gate on Python Files
description: NEVER use REVIEW_GATE_OVERRIDE=1 on .py files. Documentation and non-code files are okay to override.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
RULE: NEVER use REVIEW_GATE_OVERRIDE=1 when committing .py files. Work with the review tracker instead.

**Exception:** Documentation files (.md, .json, .env, .sh, config files) may use REVIEW_GATE_OVERRIDE=1 since the review tracker is designed for code review.

**Why:** Used REVIEW_GATE_OVERRIDE=1 on every commit in a session. Every implementation shipped with bugs. The review gate exists to catch bugs before commit — bypassing it on code files is how bugs ship.

**How to apply:**
- For .py files: run `python tools/review_tracker.py mark-file <file> --status reviewed` after review
- Let the review gate pass naturally for Python code
- For docs/config/non-code: REVIEW_GATE_OVERRIDE=1 is acceptable
- If the gate blocks a .py commit, that means it NEEDS review first
- The inconvenience is the point — it forces verification
