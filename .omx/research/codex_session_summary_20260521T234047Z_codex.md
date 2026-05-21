# Codex Session Summary: MLX auth scorer recursive greenup

## Work completed

- Re-reviewed the landed MLX scorer-response path after newer main commits.
- Closed the remaining false-authority gap by requiring explicit GPU research
  allowance before any MLX GPU scorer-response/profile execution.
- Updated `.gitignore` for ad hoc MLX scorer cache/response/profile roots so
  local generated artifacts do not leak into staging by accident.
- Preserved prior Markdown provenance append-only and wrote a successor finding
  documenting that historical GPU-profile rerun commands now need
  `--allow-gpu-research-signal`.
- Confirmed no untracked files were present at signal-loss audit time.
- Left unrelated partner dirty files unstaged and untouched.

## Verification

- `16 passed in 3.27s` for scorer-response/profiler/profile-stability greenup.
- `83 passed in 16.78s` for the recursive MLX local-acceleration suite.

## Remaining cautions

- MLX CPU remains the local calibration reference for scorer-response work.
- MLX GPU is fast prescreen/profiling signal only and must be paired with CPU
  transfer checks before influencing exact-eval dispatch choices.
- No contest score, promotion, rank/kill, or dispatch-readiness claim is made
  from this pass.

## Next recommended action

Run a future CPU-transfer calibration on the exact target scorer-response window
before allowing GPU-derived candidates into any paid exact-eval selection queue.

