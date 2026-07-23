---
utc: 2026-07-23T03:50:00Z
lane_id: ddm_j3_366_fullrun_mode_and_ticket_reseal
verdict: BLOCKED_REALIZED_DSEG_REGRESSION
research_only: true
score_claim: false
---

# Session anchor

- Built the only `--full-run` path: exact sparse-secant MLX loop, strict chunked n600 scorer decisions, immutable periodic/stage checkpoints, EMA/Adam/cursor persistence, same-outdir guard, SSD preflight, and governed memory admission.
- Resealed the executable ticket at semantic SHA `df8db01f60d582b0a716ae62af3422997fcc12c014364939ab2935a2c403b824` and typed hash `fa63e79492d916a9cc6fe144207bdcb627d07e416883e131ecb90c289f8ccec0`.
- Derived 450 steps and a `13.31387624311125–13.79371420691443 h` band from measured timings plus ten strict n600 decisions.
- Full-path memory is SAFE, and forced-kill/new-process resume is byte-faithful through step 4.
- Exact bounded efficacy is blocked: both Seg and Pose regressed. No launch, score claim, or pointer move.
- Verification: ruff, py_compile, final hash dry-run, 41 related tests, and three clean review passes.

Next: MAIN performs mandatory landing review. Preserve the build/ticket if accepted, but route the efficacy blocker to a separate realized n600 warm-start-selection reformulation before any governed fire.
