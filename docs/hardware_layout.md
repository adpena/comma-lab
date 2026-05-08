# hardware layout

## MacBook Pro

Use for:
- orchestration
- search-space bookkeeping
- codec sweeps
- MPS experiments
- dashboards, plots, and report generation

## Windows RTX 2070 Super box

Prefer WSL2 Ubuntu or Linux if possible.

Use for:
- CUDA teacher runs
- residual optimization
- surrogate training
- promoted-candidate checks on CUDA

## Validation rule

Final promoted candidates MUST be re-validated on both official axes when a
leaderboard/frontier claim is being made:

- `[contest-CUDA]`: exact archive/runtime custody on T4-equivalent or better
  CUDA remains the internal promotion, regression, and kill/retire axis.
- `[contest-CPU]`: Linux x86_64 CPU eval through the same
  `archive.zip -> inflate.sh -> upstream/evaluate.py` path is the public
  leaderboard reproduction axis and is not interchangeable with CUDA.

Local macOS CPU is a high-velocity advisory proxy only; it can guide sweeps and
catch bugs but cannot promote, rank, kill, or retire a lane. MPS remains proxy
only for smoke tests and research-signal sweeps; it is never an auth-eval score
axis.
