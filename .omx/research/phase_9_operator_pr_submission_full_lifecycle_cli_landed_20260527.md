# Phase 9 — canonical PR-submission full-lifecycle CLI (Layer 7) LANDED 2026-05-27

**Lane**: `lane_phase_9_full_lifecycle_cli_20260527` L1 (impl_complete + memory_entry)

**Subagent**: `phase_9_full_lifecycle_cli_23F7BEBD` (parent session
`b74f6039-6caf-44f2-a2c3-cd8156acd447`)

**Mission contribution per Catalog #300**: `apparatus_maintenance` /
`frontier_protecting`. Closes the canonical-submission-pipeline 7-layer
architecture at Layer 7 (the LAST layer). This single CLI is the
single-command default-path that collapses the prior ~3h x 4-subagent +
~5K-LOC + 6-phase manual PR-submission anti-pattern (2026-05-19 PR101 anchor)
to one command. Per CLAUDE.md "Race-mode rigor inversion" + the May 4 race
postmortem: shipping cadence dominates frontier when the leaderboard moves;
collapsing the submission lifecycle unlocks more PR111+ attempts per remaining
contest window.

## What landed

| Surface | Path | LOC |
|---|---|---|
| Phase 9 full-lifecycle CLI | `tools/operator_pr_submission_full_lifecycle.py` | ~570 |
| Test suite | `src/tac/tests/test_operator_pr_submission_full_lifecycle_cli.py` | ~620 |
| Landing memo | THIS file | ~360 |

**Tests**: 45 pass (0.66s). Sister regression: Phase 5 + Phase 6 + Catalog
#370 gate tests = **242 passed, 1 skipped** (the skip is pre-existing
structural-script skip in `test_submission_compliance.py`). Ruff clean.

## The canonical single-command future statement

Per the 12th canonicalization x standardization x ease-of-contest-compliance
trinity + the 13th OPTIMAL-TRIO standing directive: PR111+ contest submissions
now collapse from the 2026-05-19 PR101 anti-pattern (~3h x 4 sister subagents
Slot K + L + M + J + ~5K LOC + 6 phases of manual hand-editing per surface) to:

```bash
.venv/bin/python tools/operator_pr_submission_full_lifecycle.py \
    --lane-id <lane> \
    --substrate-trainer experiments/train_substrate_<id>.py \
    --recipe-path .omx/operator_authorize_recipes/substrate_<id>_<platform>_dispatch.yaml \
    --archive-path experiments/results/<lane>/archive.zip \
    --target-repo commaai/comma_video_compression_challenge \
    --predecessors @SajayR:56:HNeRV_substrate @AaronLeslie138:95:fec_curriculum \
    --output-dir submissions/pr<N>_<lane>/ \
    --execute
```

`--execute` to PACKET-CLEAN (exit 4 OPERATOR-GATED, gh commands emitted but
not fired) in < 60 seconds excluding the paid Modal paired-CUDA wall-clock
(vs prior ~3h manual lifecycle = ~180x wall-clock collapse).

## Canonical orchestration (Layer 0 -> Layer 6)

Each layer routes through its canonical Python API helper (NOT the per-layer
sub-CLIs; direct API for clean end-to-end orchestration). Sub-CLIs remain
callable for sister consumers.

1. **Layer 0** `build_compression_pipeline` -> `CompressionPipelineResult`
2. **Layer 1** `build_archive_grammar_from_compression_pipeline_result` -> `ArchiveGrammarManifest`
3. **Layer 2** `build_submission_bundle` -> `SubmissionBundleResult`
   (+ `submission_bundle_result.json` sidecar)
4. **Layer 3** `lint_submission_bundle` -> `LintVerdict`
   (+ `lint_verdict.json` sidecar)
5. **Layer 4** `enforce_contest_compliance` -> `ComplianceVerdict`
   (+ `compliance_verdict.json` sidecar)
6. **Layer 5** `plan_paired_auth_eval` -> `PairedAuthEvalVerdict`
   (+ `paired_auth_eval_verdict.json` sidecar)
7. **Layer 6** Phase 8 Catalog #370 gate
   `check_no_pr_submission_without_canonical_compliance_verdict` (4-verdict-
   chain verification over the just-emitted sidecars)

The 4 sidecars are emitted to the EXACT filenames the Catalog #370 gate
searches (`submission_bundle_result.json` / `lint_verdict.json` /
`compliance_verdict.json` / `paired_auth_eval_verdict.json` — verified by the
`test_canonical_sidecar_names_match_phase_8_gate` test against the live
`_CHECK_370_PHASE_*_SIDECAR_FILENAMES` constants).

## Exit-code taxonomy (binding per 9th-directive amendment Layer 7)

| Code | Verdict | Meaning |
|---|---|---|
| 0 | PACKET-CLEAN | all 7 layers PASS; ready for operator `gh pr create` |
| 1 | LINT-VIOLATIONS | Layer 3 ERROR-severity findings |
| 2 | COMPLIANCE-ERRORS | Layer 4 structural / D3+D5 blockers |
| 3 | MISSING-PAIRED-AXIS | Layer 5 verdict not PAIRED_PASS |
| 4 | OPERATOR-GATED | packet clean; gh commands emitted (NEVER fired) |
| 5 | CLI / usage error | bad arg / path / Layer 0 / Layer 1 failure / gate violation |

Note: when all 4 verdict sidecars are clean AND the Layer 6 gate passes, the
CLI reaches exit 4 OPERATOR-GATED (not 0), because `gh pr create` +
`gh release create` remain operator-gated per CLAUDE.md "Executing actions
with care". Exit 0 PACKET-CLEAN is reserved for a future fully-hosted-
submission verdict where even the hosting artifact is present. Exit 4 IS the
canonical happy-path terminal verdict for this CLI's scope.

## CRITICAL attribution discipline (per user_pr_attribution memory)

The CLI's Layer 3 linter invocation is the canonical enforcer of:
- ZERO `Claude` / `Anthropic` / `Co-Authored` / `claude.com` / `anthropic.com`
  tokens in PR-facing surfaces
- ZERO first-person-plural `\b(we|our|us|we're|we've|we'll|we'd)\b`
- ZERO emdash (U+2014)
- sole-author Alejandro Pena <adpena@gmail.com>

The CLI ALSO self-lints its own generated attribution-chain markdown (built
from `--predecessors @handle:PR:slug` specs) via `_scan_forbidden_pr_tokens`
BEFORE the canonical linter runs, so it can never emit a forbidden token into
a PR-facing surface (defense-in-depth; the linter is the canonical surface,
the self-lint is the producer-side guard).

The fork-branch-vs-internal-commit Co-Authored-By distinction is preserved:
the CLI never emits a Co-Authored-By trailer into any PR-facing surface
(forbidden per memory for commaai/* fork-branch commits); the internal
adpena/pact landing commit DOES carry the Catalog #119 trailer.

## NO auto-gh execution (per "Executing actions with care")

The CLI source contains ZERO `subprocess` / `os.system` / `Popen`
invocations (verified by `test_source_does_not_subprocess_gh`). At exit 4 it
EMITS the operator-routable `gh release create` (host archive on fork) +
`gh pr create` (submit) commands as text; the operator runs them. The report
carries `gh_commands_fired: False` permanently.

## --execute paired-env discipline (Catalog #199)

`--dry-run` (default) runs Layers 0-4 + 6 at $0 and Layer 5 prescreen-only
(MLX-local + macOS-CPU advisory plan, NO paid dispatch) per the 8th MLX-first
standing directive. `--execute` runs the full pipeline; the Layer 5
paired-CUDA GATED escalation requires BOTH
`OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE` AND a numeric
`OPERATOR_AUTHORIZE_SESSION_BUDGET_USD`. Bare CONFIRMED without a valid BUDGET
is hard-rejected (exit 5). Even with paired-env active, the CLI STILL stops at
exit 4 before any gh command. Layer 5 directs to
`tools/dispatch_modal_paired_auth_eval.py` for the ACTUAL paid execution; this
subagent NEVER fired a paid dispatch.

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Orchestration via Python API (not sub-CLIs) | FORK | direct typed-result orchestration is cleaner than subprocess-chaining the 4 sub-CLIs; per CLAUDE.md "Beauty, simplicity, and developer experience" |
| Sidecar filenames | ADOPT_CANONICAL | emit to the exact `_CHECK_370_PHASE_*_SIDECAR_FILENAMES` the Phase 8 gate searches |
| Exit-code taxonomy | ADOPT (9th-directive amendment) | 0/4/1-3/5 binding contract from the spawn directive; the spec memo's fuller per-layer taxonomy is preserved as the per-layer named blocker |
| `--dry-run` / `--execute` split | ADOPT_CANONICAL | sister of `tools/operator_authorize.py` + `tools/paired_auth_eval_cli.py` |
| Paired-env discipline | ADOPT_CANONICAL | Catalog #199 sister; `tools/operator_authorize.py` precedent |
| Attribution self-lint | NEW (unique-and-complete) | producer-side guard; no sister generator emits attribution markdown from `--predecessors` specs |
| Layer 6 gate invocation | ADOPT_CANONICAL | lazy import (fail-closed per Catalog #279 pattern) + scope to this submission_dir |

## 9-dimension success checklist evidence

1. **UNIQUENESS** — Layer 7 is the canonical end-to-end orchestrator; no
   sister CLI composes all 6 layers + the gate end-to-end.
2. **BEAUTY + ELEGANCE** — ~570 LOC; single `run_full_lifecycle` entry +
   per-layer helpers; typed report dict; canonical data flow Layer 0 -> 6.
3. **DISTINCTNESS** — explicitly different from `operator_authorize.py`
   (single dispatch), `operator_briefing.py` (situational awareness),
   `refresh_canonical_frontier.py` (pointer hygiene), and the 4 per-layer
   sub-CLIs (single-layer). This CLI covers the END-TO-END lifecycle ONLY.
4. **RIGOR** — 45 dedicated tests covering exit-code routing, sidecar
   emission to canonical Phase 8 filenames, attribution discipline,
   paired-env discipline, NO-auto-gh regression, and live-repo guards.
   Premise verification: read Phase 1 spec + all 6 sister landings + 6
   canonical helper API surfaces + Phase 8 gate sidecar contract BEFORE a
   line of code; dry-run smoke against a real archive confirmed Layer 0-1
   orchestration.
5. **OPTIMIZATION PER TECHNIQUE** — short-circuits on first failing layer
   (lint failure does not run compliance; compliance failure does not run
   paired); each layer's typed verdict is the canonical signal.
6. **STACK-OF-STACKS COMPOSABILITY** — composes 6 independent canonical
   helpers + the gate; acyclic dependency graph; each helper still callable
   standalone via its sub-CLI.
7. **DETERMINISTIC REPRODUCIBILITY** — sidecars are sorted-keys byte-stable
   JSON; exit-code routing is deterministic given the layer verdicts;
   monkeypatch-based tests verify routing without paid dispatch.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — pure-Python orchestration; $0
   in dry-run; the only paid spend (Layer 5 paired-CUDA) is operator-gated
   and directs to the canonical dispatch tool.
9. **OPTIMAL MINIMAL CONTEST SCORE** — does NOT directly lower score;
   frontier-protecting apparatus that collapses the lifecycle so MORE
   PR111+ substrate attempts ship per remaining contest window.

## Cargo-cult audit per assumption

| Assumption | HARD-EARNED / CARGO-CULTED | Unwind path |
|---|---|---|
| Direct Python API orchestration (not subprocess sub-CLIs) | HARD-EARNED per "Beauty, simplicity, DX" + cleaner typed-result composition | N/A |
| Exit 4 OPERATOR-GATED is the happy-path terminal (not exit 0) | HARD-EARNED per CLAUDE.md "Executing actions with care" (gh remains operator-gated) | N/A |
| Sidecars emitted inside submission_dir (not experiments/results) | HARD-EARNED per Phase 8 gate sidecar cascade (submission_dir is the first search location) | If a future submission_dir is read-only, fall back to experiments/results/<lane>/ per gate cascade |
| Layer 5 prescreen-only in dry-run is sufficient | HARD-EARNED per 8th MLX-first directive + Catalog #192 (macOS never promotable) | N/A; paired-CUDA is operator-gated escalation |
| Self-lint of generated attribution markdown is necessary | HARD-EARNED per user_pr_attribution memory (defense-in-depth before canonical linter) | N/A |
| Lazy import of the Phase 8 gate is canonical | HARD-EARNED per Catalog #279 fail-closed-on-import pattern | N/A |

## Observability surface

| Facet | Implementation |
|---|---|
| **Inspectable per layer** | `report["layers"][<layer_key>]` carries `{ok, ...}` per layer; each typed verdict dataclass inspectable |
| **Decomposable per signal** | per-layer verdict + per-layer sidecar + per-layer error message |
| **Diff-able across runs** | sorted-keys byte-stable sidecar JSON; deterministic report shape |
| **Queryable post-hoc** | `--json` emits the canonical machine-readable report; sidecars persisted in submission_dir |
| **Cite-able** | report carries lane_id + target_repo + archive_sha256 + per-layer sidecar paths |
| **Counterfactual-able** | adding/removing a sidecar OR flipping a layer verdict immediately changes the exit code; tests verify the cascade |

## 6-hook wire-in declaration per Catalog #125

- **Hook #1 SENSITIVITY_MAP** — N/A (orchestrator glue; no signal contribution)
- **Hook #2 PARETO_CONSTRAINT** — N/A
- **Hook #3 BIT_ALLOCATOR** — N/A
- **Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH** — **ACTIVE** (`--json` lifecycle
  verdict is consumable by the cathedral autopilot ranker /
  `pr_submission_compliance_consumer` per Catalog #335 to weight PR-readiness)
- **Hook #5 CONTINUAL_LEARNING_POSTERIOR** — **ACTIVE** (the Phase 10
  first-PR111-candidate end-to-end run will land the first empirical anchor
  for the canonical equation candidate below; Layer 5 paired results land in
  the canonical posterior per Phase 6/7 hook #5)
- **Hook #6 PROBE_DISAMBIGUATOR** — **ACTIVE** (the lifecycle verdict
  PACKET-CLEAN vs LINT-VIOLATIONS vs COMPLIANCE-ERRORS vs MISSING-PAIRED-AXIS
  vs OPERATOR-GATED IS the canonical disambiguator between full-pipeline
  completion and per-layer operator-routable blocker)

## Canonical equation #344 candidate (FORMALIZATION_PENDING)

- **id**: `full_lifecycle_cli_consolidation_savings_v1`
- **shape**: `Δ_wall_clock_per_PR = T_lifecycle_canonical - T_lifecycle_manual`
- **expected sign**: Δ << 0 (canonical CLI FASTER than manual)
- **prediction**: < 60 seconds canonical (excluding paid Modal wall-clock) vs
  ~3h x 4 subagents manual = ~180x wall-clock collapse per PR submission.
- **status**: FORMALIZATION_PENDING (per Catalog #344 — promotes at the first
  PR111-candidate end-to-end regression per Phase 1 spec memo §7 Phase 10
  acceptance criterion #5). Sister of the Phase 2-8 candidates
  (`compression_pipeline` + `archive_grammar` + `submission_bundle` +
  `submission_linter` + `pr_submission_compliance_gate`).

## Sister landings verified

- **Phase 2** `compression_pipeline.py` (`b96329a71`) — `build_compression_pipeline` consumed
- **Phase 3** `archive_grammar.py` (`1d4753f65`) — `build_archive_grammar_from_compression_pipeline_result` consumed
- **Phase 4** `builder.py` (`1de30160e`) — `build_submission_bundle` consumed
- **Phase 5** `linter.py` (`2b2f3148f`) — `lint_submission_bundle` consumed
- **Phase 6** `compliance.py` (`2d3042d14`) — `enforce_contest_compliance` consumed
- **Phase 7** `paired_auth_eval.py` (`61213aede`) — `plan_paired_auth_eval` consumed
- **Phase 8** STRICT gate Catalog #370 (`71772a30d`) — `check_no_pr_submission_without_canonical_compliance_verdict` consumed at Layer 6

## Sister coordination (Catalog #230 ownership map)

In-flight at landing (per `.omx/state/subagent_progress.jsonl` audit):
- `v15_sister_e_extrema_attempt3_C4D449C0` (V15 UNIWARD weighted-extrema) —
  SISTER-DISJOINT (UNIWARD per-pixel reactivation surface; zero overlap)
- `meta_resurrection_v2_op_routables_canonicalization_D346AA8D` —
  SISTER-DISJOINT (op-routables canonicalization; zero overlap)

Phase 9 owns: `tools/operator_pr_submission_full_lifecycle.py` (NEW) +
`src/tac/tests/test_operator_pr_submission_full_lifecycle_cli.py` (NEW) +
THIS memo (NEW). ZERO file collision. No new catalog claim (Phase 9 is a CLI,
not a STRICT gate, per the spawn directive).

## canonical-submission-pipeline 7/7-LAYER-CLOSED declaration

| Layer | Phase | Surface | Status |
|---|---|---|---|
| 0 | 2 | `compression_pipeline` | LANDED |
| 1 | 3 | `archive_grammar` | LANDED |
| 2 | 4 | `builder` | LANDED |
| 3 | 5 | `linter` | LANDED |
| 4 | 6 | `compliance` | LANDED |
| 5 | 7 | `paired_auth_eval` | LANDED |
| 6 | 8 | Catalog #370 STRICT gate | LANDED |
| 7 | **9** | **full-lifecycle CLI (THIS)** | **LANDED** |

**VERDICT: PHASE_9_FULL_LIFECYCLE_CLI_LANDED — 7/7 architecture closed;
PR111+ single-command lifecycle unblocked.**

## Operator-routable next (Phase 10)

1. **PR101 baseline regression** — run the CLI `--dry-run` against the
   canonical PR101 fec6 lane; expect structural-only re-emission.
2. **First PR111-candidate** — whichever of Cascade A FEC10 / Cascade C' /
   NSCS06 v8 lands paired-CUDA first goes through this CLI end-to-end.
3. **Catalog #370 STRICT-flip** — Live count: 0 verified at the Phase 10
   landing per Phase 1 spec §7 acceptance criterion #6.
4. **Canonical equation registration** — promote
   `full_lifecycle_cli_consolidation_savings_v1` (+ the 5 Phase 2-8 sisters)
   to REGISTERED with the first measured `T_lifecycle_canonical` anchor.

## Cross-references

- Phase 1 spec memo: `.omx/research/canonical_submission_pipeline_specification_memo_20260526.md`
- Phase 8 sister landing: `.omx/research/phase_8_strict_gate_catalog_370_canonical_submission_compliance_landed_20260526.md`
- User attribution memory: `~/.claude/projects/-Users-adpena-Projects-pact/memory/user_pr_attribution.md`
- Forbidden Claude attribution memory: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_forbidden_claude_attribution_in_public_pr_surfaces.md`

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
