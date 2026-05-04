---
name: No Signal Loss Ever — Full Provenance and History
description: Non-negotiable requirement to maintain complete provenance of all experiments, results, and decisions with full nuance
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
Never lose signal. Every experiment result, council decision, score measurement, and strategic insight must be recorded with full provenance and nuance.

**Why:** The user demands zero signal loss across sessions. Previous sessions lost critical context (e.g., the "0.43" score was initially misattributed, proxy vs auth confusion caused wrong optimization priorities). Complete provenance prevents these errors.

**How to apply:**
- Every experiment result must be saved locally with: platform, GPU, cost, timestamp, config, raw numbers
- Every score must be labeled: [contest-compliant] vs [unlimited-compute], proxy vs auth, simulate_resize or not
- Every council decision must be committed with the reasoning, vote count, and dissents
- State files (.omx/state/) must be updated after every significant discovery
- Memory files must be updated when findings change our understanding
- Commit messages must include provenance (which platform, which experiment, what changed)
- The git history IS the research timeline — never batch unrelated changes
- When a previous finding is REVISED (e.g., "SegNet 77x" was proxy-specific), update the original memory with the correction, don't just add a new one
- Track Vast.ai/Modal/GPU spend per experiment in commit messages
- Never assume a result from a previous session is still valid — verify against current code
