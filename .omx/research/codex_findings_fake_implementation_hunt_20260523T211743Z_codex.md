# Codex Findings: Fake-Implementation Hunt

utc: 2026-05-23T21:17:43Z
agent: codex
topic: fake implementation and false-green authority guards
research_only: false

## Scope

Spawned and harvested xhigh read-only subagents over provider/final-compliance,
scheduler/storage/DAG, and optimizer/MLX advisory surfaces. Promoted the
highest-risk findings into fail-closed code and focused regressions.

## Fixed Bug Classes

- Contest-final compliance no longer trusts `--submission-score` over the
  selected-axis auth-eval artifact. Final mode derives the score from strict
  component recomputation and treats explicit CLI score as an assertion that
  must match.
- Contest-final compliance now fails on raw `promotion_blockers` and
  `rank_or_kill_blockers`, including malformed blocker fields, and fails when
  adjudicated raw-policy gates report `scientific_score_eligible=false`.
- Frontier scan failure is no longer warning-only in `--contest-final`; missing
  or crashing canonical frontier state is an error in final mode.
- Review-gate JSON fallback no longer counts reviewed critical/standard
  entities as full-policy-compliant when DuckDB review-policy evidence is
  unavailable.
- `greenup-import` no longer marks arbitrary file bullets reviewed; only
  explicit CLEAN verdicts grant review status.
- Lightning exact-eval adjudication now fails closed by default for component
  and sane-score gates. Forensic rc0 requires explicit `--allow-*` flags.
- MLX dynamic learned sweep no longer treats missing
  `exact_cpu_calibrated_estimate_scope` as candidate-specific exact CPU signal.
- Decoder-q next-candidate selection now rejects advisory measured rows without
  false-authority fields and custody identity before using them for slopes.
- Staircase/DAG dispatch revalidates storage and cleanup artifacts before
  selecting materializers, and Dask specs include enforceable per-machine
  resource tokens.
- Storage tier planning no longer selects a missing workload root when
  `create=false`.
- `.gitignore` keeps raw Lightning exact-eval roots ignored but permits curated
  tracked custody anchors (`custody_anchor.json`, `git_review_anchor.json`).

## Verification

- `149 passed` focused regression bundle:
  - pre-submission compliance frontier/final checks
  - review gate fallback and greenup import
  - Lightning adjudication opt-in behavior
  - MLX dynamic learned sweep
  - decoder-q next-candidate selector
  - staircase DAG and storage tiers
  - comma-lab research-state boundary
- `ruff check` passed on touched Python files excluding legacy
  `tools/review_tracker.py` full-file lint debt.
- `compileall` passed for touched production modules.

## Outstanding

- Read-only subagents also flagged deeper planning/advisory weaknesses:
  composition ranking prior-only clean rows, byte-shaving fallback economics,
  scorer-response axis-label assumptions, materializer schema-only completion,
  and cleanup pressure-volume accounting. These should be next tranche items.
- Concurrent unrelated work appeared in `runtime-rs/`,
  `src/tac/optimization/cross_family_candidate_portfolio.py`, and related
  Rust combo acquisition files. It was not staged by this pass.
