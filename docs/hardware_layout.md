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

Final promoted candidates should be checked in environments that resemble the official CPU and CUDA runners as closely as practical.
