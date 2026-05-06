# PR106 Latent Sidecar Manifest Canonicalization - 2026-05-06 Codex

## Scope

This tranche canonicalizes the latent sidecar smoke manifest so the PR106
sidechannel stack gate can distinguish local scaffold custody from dispatchable
score evidence.

## Patch

- `experiments/build_pr106_latent_sidecar.py` now writes:
  - `source_archive_sha256`
  - `sidecar_path`
  - `archive_path`
  - `remote_jobs_dispatched=false`
  - `ready_for_exact_eval_dispatch=false`
  - explicit `dispatch_blockers`
- `src/tac/tests/test_pr106_latent_sidecar.py` now runs the CPU smoke builder
  against the PR106 archive and verifies the generated manifest is fail-closed.
- The tracked latent smoke manifest was regenerated through the builder rather
  than hand-edited.
- The PR106 sidechannel production-readiness dry-run now passes against the
  local advisory artifacts after latent manifest regeneration.

## Evidence Discipline

Evidence grade: `empirical` local smoke/custody only.

This does not claim score movement. The latent sidecar remains non-promotable
until a scorer-backed CUDA search and exact CUDA auth eval exist.
