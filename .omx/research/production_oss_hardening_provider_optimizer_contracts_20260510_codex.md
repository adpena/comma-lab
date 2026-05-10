# Production/OSS hardening: provider and optimizer evidence contracts

generated_at_utc: 2026-05-10T15:50:00Z
research_only: true
score_claim: false
dispatch_attempted: false

## Scope

This tranche hardens reusable infrastructure used by score-lowering work. It
does not claim a new score and does not launch a remote job.

## Durable changes

- Provider contracts now fail closed when a scaffold provider advertises exact
  CUDA support before lifecycle/harvest implementation exists. Modal/Azure are
  implemented exact-CUDA surfaces; AWS/GCP remain scaffolds.
- Modal T1 mounts now flow through one mount manifest used by both the Modal
  image and the mounted-code custody snapshot. The active actuator file itself,
  `experiments/__init__.py`, and explicit upstream data mounts are visible in
  the custody surface.
- Modal remote summaries are non-authoritative: `score_claim=false` until local
  recover applies the exact-CUDA custody gate. Raw remote adjudication intent is
  recorded under `remote_adjudication_*`.
- Proxy optimizer rows now share
  `tac.optimization.proxy_candidate_contract`, which forces Kaggle/Optuna/CMA
  planning rows to stay non-promotable.
- `tools/parallel_dispatch_top_k.py` now requires explicit contest target
  metadata. Missing `target_modes` no longer defaults into paid contest
  dispatch.

## Verification

- Focused production/OSS hardening suite:
  `84 passed in 1.97s`.
- Kaggle proxy manifest queue smoke:
  `n_candidates=1`, `dispatch_ready=0`,
  `target_modes=["contest_exact_eval_planning"]`,
  `score_claim=false`.
- All-lanes preflight before staging completed in `2.36s`; the only failure was
  expected untracked-source custody for the new files in this tranche.

## Score-lowering implication

This does not lower score directly. It reduces false authority and duplicate
provider risk around the active T1 Modal run and makes optimizer/CMA/Optuna/
Kaggle output safely composable with the exact-eval actuator. The next
score-lowering implementation gap remains the exact-readiness promoter that
takes one planning row and proves byte-closed archive/runtime custody before
dispatch.
