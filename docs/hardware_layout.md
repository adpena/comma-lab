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

Final promoted candidates MUST be re-validated against the official `[contest-CUDA]` runner (Vast.ai 4090 / A100 / Modal T4 with the pinned upstream `evaluate.py`) BEFORE any kill/promote decision lands. CPU and MPS results are advisory only — see CLAUDE.md "MPS auth eval is NOISE" non-negotiable. The MPS-vs-CUDA score drift on PoseNet is 23×.
