# Harness consolidation fixes — DAG FEED (2026-07-15)

`research_only=true`

`score_claim=false`

`verdict_scope=canonical local Codex delegation, status, drain, and tracked-shell lint surfaces only`

## Verdict

`BUILT_AND_STRICT_SELF_PROTECTED_LOCAL_CPU_$0`. No GPU, paid dispatch, evaluator run, score movement, or frontier-pointer change occurred.

## Per-class disposition

| Class | Disposition | Runtime fix | STRICT/self-protect | Exact remaining blocker |
|---|---|---|---|---|
| Retry restarts long arm from zero | `RESOLVED / gate-landed` | Retry cap 8→2; exact delegation checkpoint required; rc=20 fail-closed; compact resume prompt continues from `next_action` | `check_codex_retry_preserves_original_sandbox_authority` behaviorally checks exact-key custody, bounded prompts, capped retry, and review marker | Application-specific trainers still own their own complete per-stage checkpoint schema; this harness only proves subagent checkpoint custody |
| Pre-CFL in-flight writers | `RESOLVED / gate-landed` | Legacy live writer without `isolate=true` becomes `STRAND_DOOMED` and actionable; drain treats it as `WEDGED` | `check_codex_nonisolated_writer_cap` executes missing-field and safe-control probes | Any bytes already stranded in a shared tree still need explicit MAIN harvest and review; classification does not confer attribution |
| Drain timeout exits zero | `RESOLVED / gate-landed` | `DRAINED=0`, `TIMED_OUT=2`, `WEDGED=3` | `check_codex_drain_timeout_uses_liveness` executes fresh-log, progress, stale, and strand-doomed controls | None inside verdict scope |
| zsh nomatch optional glob | `RESOLVED / gate-landed` | Tracked shell monitor lint rejects bare `*.last.txt` / `*stage*.npz`; allows `null_glob`, `(N)`, `find(1)`, quoted literals | Existing STRICT `check_dispatch_cli_shell_hazards` now covers `.command` and source-index paths | Ad-hoc untracked interactive zsh commands remain outside static repository lint scope |
| `codex_probe_token_limit` | `DIAGNOSIS FALSIFIED; harness pressure/custody gate-landed; external blocker remains` | Full authority is file-backed with SHA+bytes and 32768-byte read direction; default effort `high`; compact resume/final-output direction; explicit landing-review marker | Retry STRICT gate checks prompt size and `review_required=1` | Persistent external model capacity can outlast both bounded retries; repository code cannot supply service capacity |
| workspace-write git objects | `RESOLVED / gate-landed` | Exact fix `9cc9eb830b` | Independent proof commit `d1510a9cd1` | Explicit noncanonical sandbox invocations outside `codex_delegate` remain outside verdict scope |

## Authority and provenance

- `MEASURED`: focused behavior/STRICT tests passed locally on macOS CPU; no network/GPU use.
- `DERIVED`: retry rc=20 follows from absent exact-key checkpoint; timeout rc=2 follows from remaining live arms at deadline; `STRAND_DOOMED` follows from writer authority plus `isolate is not true`.
- `CONFIG`: retry cap is 2 and chunk direction is 32768 bytes.
- `ANCHOR`: operating manual resumability P0, NO-FAKE, and two-landing fix+STRICT contract.
- `WAIVER`: none.

## Local verification receipt

- `69 passed`: retry, status reconciliation, apparatus behavior/negate probes, and dispatch-shell hazard suites.
- `27 passed`: failure-ledger and producer-bridge suites.
- `tools/check_dispatch_cli_shell_hazards.py --strict`: rc=0 on the live tracked repository.
- All three apparatus STRICT gates returned `[]` on the live repository.
- Targeted Ruff check: clean (the two pre-existing `SIM103`/`SIM110` findings at `confound_gates.py:81/89` were explicitly excluded and are outside this landing).

## Triality and canonical wire-in

- DSL leg: `N/A` — harness process custody only; no trainer lever or configuration flag added.
- DAG leg: this durable FEED records terminal statuses and blockers; campaign queue pointer is unchanged.
- Equations leg: `N/A` — no scientific law changed. The process invariant is `success ⇔ status=DRAINED`; all other terminal states return nonzero.
- Sensitivity map / Pareto / bit allocator / cathedral dispatch / posterior / probe-disambiguator hooks: `N/A`, because this is apparatus-only and makes no empirical witness or score claim. The reusable consumers are the canonical delegation status, drain, preflight, and failure-ledger surfaces.

## STORES CONSULTED

- `docs/operating_manual_craft_handoff.md`
- `CLAUDE.md`
- `AGENTS.md`
- `.omx/research/P0_campaign_queue_20260715.md`
- `.omx/research/means_audit_4enum_negatives_open_derivation_fourier_20260715.md`
- `.omx/research/consolidation_drain_cherrypick_prune_driftgate_20260714.md`
- `.omx/research/ledger_debt_drain_disposition_20260714.md`
- `.omx/state/harness_failure_ledger.jsonl`
- `.omx/state/subagent_progress.jsonl`

## Pointer delta

`NONE`. This landing changes harness reliability and custody only. It neither authorizes a campaign launch nor changes `reports/latest.md` or any score pointer.
