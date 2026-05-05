---
name: All Deploy Scripts Must Use Canonical tac Deploy Infrastructure
description: Vast.ai and all deployment scripts must live in src/tac/deploy/ and follow the modular canonical patterns. No ad-hoc standalone scripts.
type: feedback
originSessionId: 47bf3dd8-df75-4271-9ce1-428c19c2eb32
---
All deployment infrastructure MUST be canonical and modular within the tac library. Smart, Pythonic, maintainable, composable, creative, expressive abstractions. Clean OSS quality — zero code smell.

**Why:** User explicitly requires no ad-hoc scripts. This is intended to be open-sourced. Every abstraction should be thoughtful, every interface composable. The code should read like a well-written library, not a hackathon project.

**How to apply:**
- Vast.ai deployment goes in `src/tac/deploy/vastai/` (matching `src/tac/deploy/modal/` and `src/tac/deploy/kaggle/`)
- Reuse shared infrastructure: cost tracking, experiment configs, results management
- Extract common patterns across Modal/Kaggle/Vast.ai into shared base (e.g., `src/tac/deploy/base.py`)
- Experiment configs should be declarative, platform-agnostic, composable
- Clean abstractions: `Experiment`, `Platform`, `BudgetTracker`, `ResultStore`
- Entry points follow consistent patterns across all platforms
- A thin CLI script at `scripts/vastai_deploy.py` can exist but MUST delegate to `src/tac/deploy/vastai/`
- Budget tracking, instance management, results download — all in `src/tac/deploy/vastai/`
- Any platform-specific code isolated behind clean interfaces
- Pythonic: dataclasses/pydantic, context managers, generators where natural
- Expressive: clear naming, self-documenting structure, minimal comments needed
- No code smell: no dead code, no god functions, no magic numbers, DRY
