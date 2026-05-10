# Provider-Agnostic Deploy Contracts — 2026-05-10 Codex Landing

Scope: reusable deploy abstraction/docs/tests only. No remote jobs launched. No
active T1 result/custody files edited.

Artifact changes:

- `src/tac/deploy/provider_contracts.py` adds the static registry for Modal,
  Kaggle, AWS, Azure, and GCP.
- `src/tac/deploy/gcp/` adds a fail-closed dry-run GCP scaffold.
- `docs/deploy/provider_contracts.md` records the deterministic operator
  runbook.
- `src/tac/preflight.py` exposes the registry as a strict deploy guard.

Custody invariants preserved:

- Claim before dispatch.
- Terminal claim row after dispatch.
- Plan-only default path; explicit execution flag required for implemented
  providers.
- No MPS auth-eval score truth.
- No score claim from Kaggle or other provider proxy substrates.
- CUDA exact-eval artifacts still require byte-closed custody and adjudication.

Solver wire-in:

- `research_only=true` for this infrastructure landing.
- Sensitivity-map contribution: N/A, no scored payload or empirical anchor.
- Pareto constraint: N/A, no new candidate frontier point.
- Bit-allocator hook: N/A, no per-tensor importance signal.
- Cathedral autopilot dispatch hook: deferred until lane-specific actuators
  consume the registry.
- Continual-learning posterior update: N/A, no empirical anchor.
- Probe-disambiguator: N/A, single fail-closed interpretation.
