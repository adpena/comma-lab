# 2026-04-05 dynamic main ROI experiment

## purpose

Implement the next ROI-aware experiment without neglecting the main ROI.

This run family is intended to test a **dynamic, temporally smoothed main ROI** with an **optional auxiliary ROI**. The main ROI remains mandatory and central; the auxiliary ROI is additive only.

## status

- Planning complete
- Implementation not started in this artifact yet
- No measured score recorded yet

## constraints

- Use `uv` for Python tooling
- Keep heavy analysis on the compression side only
- Keep local CPU scorer authoritative for any promotion
- Keep byte accounting honest in both `current_workflow` and `rule_faithful`

## metadata contract (planned)

Per video / time window JSON should include:

- `video`
- `window_start`
- `window_end`
- `main_roi` with `x`, `y`, `w`, `h`
- `aux_roi` with `x`, `y`, `w`, `h` or `null`
- summary stats for motion/staticness and smoothing

## first measured success bar

1. package succeeds
2. inflate succeeds
3. shape/frame-count checks pass
4. one full local CPU score is recorded
5. promotion only if the score beats `3.33`
