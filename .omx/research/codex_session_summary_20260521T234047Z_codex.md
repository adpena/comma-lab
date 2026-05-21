# Codex Session Summary: MLX auth scorer recursive greenup

## Work completed

- Re-reviewed the landed MLX scorer-response path after newer main commits.
- Closed the remaining false-authority gap by requiring explicit GPU research
  allowance before any MLX GPU scorer-response/profile execution.
- Reconciled the concurrent `main` advance to `a71ef3500`, which also added
  the MLX scorer batch-invariance audit for GPU batch-shape drift.
- Updated `.gitignore` for ad hoc MLX scorer cache/response/profile roots so
  local generated artifacts do not leak into staging by accident.
- Preserved prior Markdown provenance append-only and wrote a successor finding
  documenting that historical GPU-profile rerun commands now need
  `--allow-gpu-research-signal`.
- Confirmed no untracked files were present at the initial signal-loss audit;
  after authoring this summary, only these two Codex memos remained untracked
  pending exact-file staging.
- Left unrelated partner dirty files unstaged and untouched.

## Verification

- `16 passed in 3.27s` for scorer-response/profiler/profile-stability greenup.
- `ruff check` passed on the MLX batch/scorer-response tools, modules, and
  tests involved in this review.
- `87 passed in 16.84s` for the recursive MLX local-acceleration suite,
  including the new batch-invariance guard.

## Remaining cautions

- MLX CPU remains the local calibration reference for scorer-response work.
- MLX GPU is fast prescreen/profiling signal only and must be paired with CPU
  transfer checks before influencing exact-eval dispatch choices.
- No contest score, promotion, rank/kill, or dispatch-readiness claim is made
  from this pass.

## Next recommended action

Run a future CPU-transfer calibration on the exact target scorer-response window
before allowing GPU-derived candidates into any paid exact-eval selection queue.
