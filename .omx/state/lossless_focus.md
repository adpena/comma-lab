# Lossless Focus

## Objective
- Build a canonical `tac/lossless` workflow for the commaVQ lossless challenge.

## Status
- `tac/lossless` now has real modules for:
  - contracts
  - data
  - exact evaluation
  - submission packaging
  - profiles
  - state rendering
- `tac` now has real `lossless` CLI commands:
  - `profiles`
  - `compress`
  - `package`
  - `evaluate`
  - `promote`
- No measured leaderboard-grade lossless baseline has been recorded yet.

## Constraints
- No legacy compatibility.
- No lossy state contamination.
- Exact commavq contract compliance before any promotion.
