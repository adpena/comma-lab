# Documentation Index

Start with the files that describe the current public repository contract:

- [README.md](../README.md): project overview, evidence grades, quick start,
  and package map.
- [src/tac/README.md](../src/tac/README.md): Task-Aware Compression library
  scope, terminology, core surfaces, and package boundary.
- [src/comma_lab/README.md](../src/comma_lab/README.md): lab operations and
  state-projection package boundary.
- [docs/terminology_and_boundaries.md](terminology_and_boundaries.md):
  canonical naming rule for `tac`, `codec`, `comma-lab`, and `comma_lab`.
- [docs/contest_compliance_authority.md](contest_compliance_authority.md):
  upstream rule reading, public PR precedent ladder, and procedural-generation
  authority packet protocol.
- [docs/archive_bound_candidate_pipeline.md](archive_bound_candidate_pipeline.md):
  shared archive-bound contract, adapter spine, acquisition input, and
  fail-closed exact handoff rules.

## Current Public Docs

- [docs/paper/](paper/): technical paper sources and result methodology.
- [docs/pr_writeups/](pr_writeups/): public challenge writeups and release
  cut material.
- [docs/runbooks/](runbooks/): operator runbooks for exact eval, public
  intake, hidden-gem readiness, and dispatch hygiene.
- [docs/findings/](findings/): durable engineering findings such as CPU/CUDA
  auth-eval split methodology.
- [docs/release/](release/): OSS release notes and disclosure hygiene.

## Historical And Internal Plans

`docs/superpowers/` contains older specs and implementation plans. They are
kept for provenance, but they are not the live task queue unless a newer
`.omx/research/` directive or canonical task-status row explicitly revives one.
When a historical plan disagrees with the root README, package READMEs,
`AGENTS.md`, or `CLAUDE.md`, prefer the current contract.

## Documentation Gates

Run the public terminology guard before documentation commits:

```bash
.venv/bin/python tools/check_tac_terminology.py --strict
```
