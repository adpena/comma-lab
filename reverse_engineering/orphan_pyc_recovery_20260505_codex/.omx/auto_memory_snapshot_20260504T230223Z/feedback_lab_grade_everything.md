---
name: Lab-Grade Everything — Non-Negotiable Standard
description: Every piece of DX, reporting, logging, scoring, and infrastructure must be journal-grade hardened
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
ALL infrastructure, reporting, logging, and DX must meet lab-grade / journal-grade standards:

**Reporting:**
- Every auth eval result must trace to the exact upstream evaluate.py invocation
- Every proxy score must be labeled "PROXY" and never compared to auth scores
- Component breakdowns (seg, pose, rate) must always show both raw distortion AND weighted contribution
- results.json must include: git commit hash, full config, timing, hardware, software versions

**Logging:**
- Every experiment must have incremental logging (not just end-of-run)
- Volume commits at regular intervals for crash recovery
- Start log with config committed BEFORE experiment begins
- Final log with exit status committed AFTER experiment ends

**DX:**
- VRAM estimation before any GPU launch
- Checkpoint resume for long-running experiments
- P100 detection with clear marker files
- One-command status check for all platforms (modal_check.py, kaggle_check.py)

**Scoring:**
- Single source of truth: upstream evaluate.py
- Never claim a score without auth eval confirmation
- Proxy scores labeled as such in all reporting
- Score formula verified: 100*seg_dist + sqrt(10*pose_dist) + 25*rate

**Code:**
- No code smell anywhere
- Recursive review until clean pass
- All issues fixed regardless of severity
- Every function documented, every magic number commented

**Why:** This is going to be open-sourced. Every file will be read by competitors,
reviewers, and the community. The quality of the infrastructure IS the argument
for the methodology.
