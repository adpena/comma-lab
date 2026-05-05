---
name: Code Quality Standard — Portfolio Grade, No Ad Hoc, Ever
description: Every line of code must be written as if researchers, judges, and employers are reading it. Build for the future. No ad hoc ever.
type: feedback
originSessionId: 47bf3dd8-df75-4271-9ce1-428c19c2eb32
---
Every line of code must be written as if researchers, judges, and employers are reading it.

**Why:** This codebase IS the portfolio. The paper, the competition entry, the open-source release — they all point here. Every abstraction, every naming choice, every module boundary is a statement about engineering quality. Ad hoc code is technical debt that becomes permanent the moment it's committed.

**How to apply:**
- Build for the future, not just the current experiment
- Every module should be independently understandable by a stranger
- Abstractions must be justified by real reuse, not hypothetical
- Naming must be precise and domain-appropriate (steganalysis terms, compression terms, not generic ML jargon)
- No "quick and dirty" — if it's worth doing, it's worth doing right
- No dead code, no commented-out blocks, no TODO-without-issue
- Tests for anything that could break silently
- Type hints everywhere, pydantic for configs
- Docstrings that explain WHY, not just WHAT
- If a researcher reads this in 2028, they should be able to understand and reproduce
- If an employer reads this, they should want to hire the author
- If a judge reads this, they should see engineering excellence
