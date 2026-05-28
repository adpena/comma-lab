# comma-lab

`comma_lab` is the comma-lab operations layer around `tac`. It turns Task-Aware
Compression primitives into reproducible lab workflows: state projection,
preflight orchestration, public-frontier intake, reverse-engineering hygiene,
provider dispatch support, artifact custody, and operator-facing reports. `pact`
is retained only as a historical/internal workspace alias in older docs and
local paths.

It is intentionally not the compression engine. Reusable compression math,
archive grammars, scorer contracts, byte profilers, packet compilers, and
runtime-safe algorithmic code belong in `tac`. `comma_lab` owns the surrounding
control plane that keeps those artifacts auditable, release-ready, and
score-authority-safe.

## Scope

`comma_lab` owns:

- state models and state synchronization for promoted results;
- strict preflight adapters that expose canonical `tac` checks to operator
  flows;
- research-state and reverse-engineering disposition helpers;
- scheduler models, run registries, and report formatting;
- install/bootstrap/smoke helpers for a live repository checkout;
- thin CLI entry points for lab operations that should not live inside `tac`.

`comma_lab` does not own:

- codec implementations or entropy coders;
- scorer math or exact score formula helpers;
- inflate/runtime decoder internals;
- model architectures, losses, or quantization kernels;
- reusable packet grammars or archive parsers.

## Module Map

| Module | Role |
|---|---|
| `comma_lab.cli` | Operator-facing command surface |
| `comma_lab.preflight` | Adapters around canonical `tac.preflight` checks |
| `comma_lab.research_state` | Track, summarize, externalize, or ignore research artifacts |
| `comma_lab.reverse_engineering` | Lab-facing public-submission hygiene import surface backed by `tac.reverse_engineering_curation` |
| `comma_lab.scheduler` | Lightweight scheduler data models and reporting |
| `comma_lab.scheduler.queue_fleet` | Live queue discovery, health classification, and bounded supervision across `.omx` / `experiments` queues |
| `comma_lab.state_models` | Promoted-result and custody state models |
| `comma_lab.state_sync` | Deterministic state projection and doctor/sync flows |
| `comma_lab.snapshot` | Repository snapshot helpers |
| `comma_lab.task_codec` | Legacy compatibility records for early post-filter state only; not a new-code namespace |

## Active Workflows

This package is the live operations harness around the `tac` compression
library. Current active workflows include:

- projecting promoted score and custody state into operator-facing read models;
- exposing strict `tac.preflight` checks through normal lab/operator flows;
- auditing which `.omx/research`, state, report, and reverse-engineering files
  should be tracked, externalized, summarized, or ignored;
- keeping public-frontier intake and deconstruction trees clean;
- producing release, paper, and OSS hygiene reports without moving algorithmic
  compression logic out of `tac`;
- preserving exact evidence labels so advisory local or proxy signals never
  masquerade as score-bearing contest authority.

## Boundary Rule

When adding code, ask which layer owns the durable abstraction:

- **Algorithmic compression primitive**: put it in `tac`.
- **Contest/runtime validity check** reusable outside one tool: put it in
  `tac.preflight` or another `tac` module, then expose it through
  `comma_lab.preflight` if operators need it.
- **Reusable repository/package hygiene check**: put the pure checker in `tac`
  when `tac.preflight` enforces it; expose a `comma_lab` adapter only for lab
  dashboards and operator workflows.
- **Lab state, provider state, public intake, or report projection**: put it in
  `comma_lab`.
- **One-off CLI glue**: keep it thin and delegate to `tac` or `comma_lab`.

This keeps `tac` publishable as Task-Aware Compression software while
`comma_lab` remains the rigorous lab notebook and operations harness around it.

## Development Contract

- Never create score authority from `comma_lab` alone; score authority comes
  from byte-closed archives, exact eval artifacts, runtime custody, and
  canonical `tac` validators.
- Prefer structured state models over ad hoc JSON mutation.
- Preserve historical research signal by summarizing it into dated ledgers or
  manifests rather than committing raw provider logs or rebuildable artifacts.
- Keep commands reproducible from repository-relative paths.
- Keep large or private artifacts out of tracked source unless a committed
  manifest points to their custody location.

## Related Packages

- `tac`: Task-Aware Compression library and reusable algorithmic engine.
- `tools/`: thin operator CLIs, launchers, validators, and recovery scripts.
- `.omx/research/`: dated scientific, adversarial, and progress ledgers.

The repository-wide terminology authority is
[`docs/terminology_and_boundaries.md`](../../docs/terminology_and_boundaries.md).
Contest rule and public-PR precedent authority is
[`docs/contest_compliance_authority.md`](../../docs/contest_compliance_authority.md).
