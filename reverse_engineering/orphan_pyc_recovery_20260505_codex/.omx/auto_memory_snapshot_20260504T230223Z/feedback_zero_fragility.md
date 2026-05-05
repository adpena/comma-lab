---
name: Zero Fragility — No Silent Failures, No Broken Paths, Full Reproducibility
description: Non-negotiable code robustness requirements. Every path validated. Every config explicit. Every failure loud. Full reproducibility.
type: feedback
originSessionId: 47bf3dd8-df75-4271-9ce1-428c19c2eb32
---
The codebase must have ZERO fragility. This is non-negotiable.

**Why:** Silent failures wasted weeks of GPU time (the gradient bug). Hardcoded paths break across machines. Default overrides create invisible behavior changes. Every fragile pattern is a future debugging session.

**How to apply:**
- **No silent failures**: Every operation that can fail must fail LOUDLY with a clear error message pointing to the fix. No bare excepts. No swallowed errors. No "if None, use default" patterns that hide bugs.
- **No broken paths**: Every file path must be constructed from config or discovered dynamically. No hardcoded absolute paths. Relative paths must be validated before use.
- **No hardcoded configs**: Every parameter that could change between runs must be configurable (CLI arg, config file, or explicit constant with a name and comment).
- **No default overrides**: If a function has a default that callers override, grep ALL callers when changing the default. The "default override antipattern" has caused 4 bugs in this project.
- **No fragile bugs**: Add assertions and validation at function boundaries. If a tensor should be (N, H, W, 3), assert it. If a file should exist, check and raise before proceeding.
- **Full reproducibility**: Every experiment must be reproducible from its config. Seeds, hyperparameters, model versions, data versions — all recorded. A stranger should be able to reproduce any result from the git history.
- **Gradient validation**: The coupled_trajectory_optimize gradient check is the MODEL for this — 1ms cost prevents hours of wasted GPU time. Apply this pattern everywhere.
