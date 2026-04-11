# Lossless Findings

## 2026-04-11 current state

- The commavq lossless challenge contract is private-submission driven, not PR-driven.
- Visible leaderboard methods cluster around:
  - arithmetic coding with GPT
  - generic compressors like `zpaq`
  - a stronger private winner labeled `self-compressing neural network`
- The right workflow is stealth-first, exact-eval-first, and stateful from day one.
- `tac/lossless` now exists as a real subsystem with canonical:
  - profiles
  - packaging
  - file-based exact evaluation
  - promotion/state rendering
- The remaining gap is not scaffolding anymore; it is running the first true commavq-measured baseline and then iterating from evidence.
