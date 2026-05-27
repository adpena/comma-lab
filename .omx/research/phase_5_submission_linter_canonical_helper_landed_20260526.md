# Phase 5 submission linter canonical helper LANDED 2026-05-26

Landing memo for Phase 5 of the canonical submission pipeline per
`.omx/research/canonical_submission_pipeline_specification_memo_20260526.md`
§3 Layer 3 (operator prompt redefinition: linter, sister-disjoint from
parallel Phase 6 compliance spawn).

## What landed

| Surface | Path | LOC |
|---|---|---|
| Canonical helper module | `src/tac/submission_packet/linter.py` | 1020 |
| Public API re-exports | `src/tac/submission_packet/__init__.py` | +30 |
| Operator-facing CLI | `tools/submission_linter_cli.py` | 340 |
| Cathedral consumer | `src/tac/cathedral_consumers/submission_linter_consumer/__init__.py` | 117 |
| Tests | `src/tac/tests/test_submission_linter.py` | 950 |

**Tests**: 99 pass + 1 skip (canonical PR body template gracefully skipped
when not in repo; per CLAUDE.md "Forbidden premature KILL").

## Canonical contract per Phase 1 spec memo Layer 3

`tac.submission_packet.lint_submission_bundle(submission_bundle_result,
*, target_repo, pr_body_path, pr_body_text, inflate_py_loc_waiver_rationale)
-> LintVerdict` consumes a Phase 4 `SubmissionBundleResult` and runs the
canonical 5-surface lint cascade in the 11th ORDER-MATTERS canonical
ordering:

1. **PR body** (`lint_pr_body`) — forbidden tokens (Claude / Anthropic /
   Co-Authored / claude.com / anthropic.com) + first-person plural
   (we / our / us word-boundary regex) + emdash (U+2014) + tone
   violations (signoff flourish / marketing hype / AI-tell / excessive
   punctuation) + emoji + attribution chain (≥1 @-mention + ≥1 PR#) +
   axis tag presence + Catalog #208 local-absolute-paths.
2. **inflate.py** (`lint_inflate_py`) — HNeRV parity L4 LOC budget +
   substantive waiver discipline per Catalog #287 + Catalog #205
   canonical `select_inflate_device` routing + Catalog #295 PYTHONPATH
   self-containment.
3. **archive.zip** (`lint_archive_zip`) — sha + size match against
   `SubmissionBundleResult.archive_sha256` + `archive_bytes`.
4. **compliance placeholder** (`lint_compliance_placeholder`) —
   sister-disjoint observability surface that detects parallel Phase 6
   compliance enforcer sidecar JSON without invoking the compliance
   subprocess (sister-disjoint parallel spawn boundary respected).
5. **README.md** (`lint_readme`) — forbidden token + Catalog #208
   local-absolute-paths scoped to `submission_dir/README.md`.

Every emitted `LintVerdict` carries canonical Provenance per Catalog
#323: `axis_tag=[predicted]` + `score_claim=False` + `promotable=False`
+ `evidence_grade=[predicted; submission-linter-canonical]` +
`canonical_helper_invocation=tac.submission_packet.lint_submission_bundle`.

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| `LintFinding` dataclass | ADOPT_CANONICAL | sister of `LintFinding` shape across Phase 2/3/4 frozen dataclasses |
| `LintVerdict` dataclass | ADOPT_CANONICAL | sister of `CompressionPipelineResult` / `ArchiveGrammarManifest` / `SubmissionBundleResult` |
| Forbidden-token grep | FORK_BECAUSE_PRINCIPLED_MISMATCH | builder's `_scan_for_forbidden_pr_tokens` is a one-shot internal helper; linter needs per-finding line/col + fix-suggestion shape |
| First-person plural | NEW (unique-and-complete-per-method) | no canonical sister exists; operator first-person-only directive is the canonical authority |
| Emdash audit | NEW (unique-and-complete-per-method) | no canonical sister exists; PR 95 medal-class typography discipline is the canonical authority |
| Catalog #208 path scan | ADOPT_CANONICAL | reuses regex pattern set from `tac.submission_packet.builder._LOCAL_ABSOLUTE_PATH_PATTERNS` |
| Compliance placeholder | SISTER_DISJOINT_PARALLEL | scope explicitly bounded to OBSERVATION ONLY; Phase 6 spawn owns subprocess invocation |
| CLI exit codes | NEW (unique-and-complete-per-method) | 6 canonical exit codes per operator prompt; no sister CLI has this taxonomy |

## 9-dimension success checklist evidence

1. **UNIQUENESS** — Linter is the canonical Phase 5 Layer 3 surface; no
   sister exists at this surface (Phase 4 builder + Phase 6 compliance
   are upstream/downstream sisters).
2. **BEAUTY + ELEGANCE** — 1020 LOC + 5 per-surface helpers + 1 main
   entry point + canonical CLI; reviewable in 30 seconds per PR101
   medal-class precedent.
3. **DISTINCTNESS** — Lint surfaces are explicitly separate from Phase
   4 (BUILDS) and Phase 6 (ENFORCES). Linter does NOT modify any
   artifact; OBSERVABILITY-ONLY by construction.
4. **RIGOR** — Premise verification: read full Phase 1 spec + Phase 2/3/4
   sister landings + user_pr_attribution memory + PR 95 medal-class
   study before writing a line. Per-finding fix_suggestion ≥4 chars +
   placeholder rationale rejection per Catalog #287.
5. **OPTIMIZATION PER TECHNIQUE** — Per-rule regex compiled at module
   load (no runtime recompilation); LOC counter is a single line-count
   call; sha256 streams via 64KB chunks.
6. **STACK-OF-STACKS COMPOSABILITY** — Layer 3 sits cleanly between
   Layer 2 (builder) and Layer 4 (compliance); linter consumes
   `SubmissionBundleResult` cleanly; cathedral consumer observes lint
   verdict without mutating ranker.
7. **DETERMINISTIC REPRODUCIBILITY** — All regex patterns are pinned at
   module load; sorted/deterministic finding emission order; canonical
   sha256 deterministic.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — Lint is pure-Python regex
   over bounded source files (PR body ~1KB; inflate.py ≤200 LOC;
   archive.zip metadata only) — runs in milliseconds.
9. **OPTIMAL MINIMAL CONTEST SCORE** — Linter is observability-only per
   Catalog #341; it does NOT contribute to score directly; it
   structurally extincts the bug class of submitting an unclean PR
   body that gets refused by maintainer review (saving a re-submission
   cycle).

## Cargo-cult audit per assumption

| Assumption | HARD-EARNED vs CARGO-CULTED | Rationale |
|---|---|---|
| Word-boundary regex catches `\bwe\b` not `weave` | HARD-EARNED | regex tested with positive + negative cases including `weave` / `swept` / `Power` |
| Tone violations are 8-pattern list per PR 95 study | HARD-EARNED | derived from PR #56/95/100/101/102/103 actual bodies (all single-author, matter-of-fact, no flourishes) |
| `[predicted]` axis tag for observability-only | HARD-EARNED | per Catalog #341 canonical-routing-markers + Catalog #323 canonical Provenance |
| `attribution_no_at_mention` is WARN not ERROR | HARD-EARNED | per PR 95 study some PR bodies are 15-line minimum and skip @-mention chain |
| inflate.py over-budget is ERROR (no waiver) | HARD-EARNED | per HNeRV parity L4 + sister Phase 4 builder discipline |
| inflate.py over-budget WITH waiver is WARN | HARD-EARNED | per Catalog #287 substantive-rationale acceptance pattern |
| Compliance placeholder is INFO not WARN | CARGO-CULTED-MAYBE | parallel Phase 6 spawn may upgrade to WARN if absence becomes a blocker; operator-routable per Phase 6 landing |
| Emoji is ERROR not WARN | HARD-EARNED | per PR #56/95/100/101/102/103 zero-emoji empirical convention |

## Observability surface

The linter exposes ALL 6 canonical facets per Catalog #305:

1. **Inspectable per layer** — `LintVerdict.findings` is a typed tuple
   of `LintFinding` rows; each row carries surface + severity + rule +
   file_path + line_number + matched_text + fix_suggestion.
2. **Decomposable per signal** — `error_count` / `warn_count` /
   `info_count` per-severity; `surfaces_scanned` per-surface.
3. **Diff-able across runs** — `LintVerdict.as_dict()` round-trips via
   JSON; two runs over the same submission_dir emit byte-identical
   verdicts (modulo `measurement_utc` + `elapsed_seconds`).
4. **Queryable post-hoc** — CLI `--json` emits the canonical
   `as_dict()` payload; cathedral consumer's `submission_linter_verdict`
   field accepts the same shape.
5. **Cite-able** — `canonical_provenance` dict carries
   `canonical_helper_invocation` + `captured_at_utc` + `target_repo` +
   `archive_sha256` (when present) per Catalog #323.
6. **Counterfactual-able** — every `LintFinding.fix_suggestion`
   surfaces the operator-actionable canonical recommendation; "what if
   I changed this byte" is answered by the per-rule fix.

## Predicted ΔS band

**N/A** — this layer is OBSERVABILITY-ONLY per Catalog #341. No score
contribution; `predicted_delta_adjustment=0.0` in the cathedral
consumer's contribution dict. The 8th MLX-first numpy-portable directive
applies via the linter being pure-Python (no MLX or numpy dep beyond the
sister Phase 2/3/4 helpers it imports from). The 13th `[predicted]` axis
tag honored.

## Horizon class

**plateau_adjacent** — this is apparatus growth (Layer 3 of the canonical
8-layer pipeline) per the 13th OPTIMAL-TRIO standing directive. It does
NOT directly lower contest score; it structurally extincts the
PR-submission rejection bug class so future PR111-candidate landings
flow through ONE canonical helper rather than ad-hoc per-PR linter logic.

## Canonical equation status

`submission_linter_canonical_helper_consolidation_savings_v1` is
FORMALIZATION_PENDING per the spec memo Phase 1 §13 + Catalog #344
sister discipline. The equation will land in the registry at Phase 10
first PR111-candidate end-to-end regression when the predicted savings
(ad-hoc per-PR linter logic consolidated to ONE canonical helper) is
empirically anchored.

This is the SAME FORMALIZATION_PENDING posture as sister Phase 2
(`compression_pipeline_canonical_helper_consolidation_savings_v1`),
Phase 4 (`submission_bundle_canonical_helper_consolidation_savings_v1`),
and the sister Phase 6 compliance enforcer's equation. The shared
posture means a single Phase 10 regression promotes ALL 4 equations
together via the canonical `update_equation_with_empirical_anchor`
helper.

## 6-hook wire-in declaration per Catalog #125

| Hook | Status | Rationale |
|---|---|---|
| #1 SENSITIVITY_MAP | N/A | Defensive observability gate; no signal contribution |
| #2 PARETO_CONSTRAINT | N/A | No Pareto-relevant signal |
| #3 BIT_ALLOCATOR | ACTIVE | Cathedral consumer surfaces LINT_CLEAN / BLOCKED verdict so bit-allocator priority cascade routes LINT_CLEAN candidates first |
| #4 CATHEDRAL_AUTOPILOT_DISPATCH | ACTIVE PRIMARY | Cathedral consumer at `tac.cathedral_consumers.submission_linter_consumer` auto-discovered per Catalog #335/#336/#337 |
| #5 CONTINUAL_LEARNING_POSTERIOR | ACTIVE | Per-lint-verdict anchor feeds canonical posterior so Phase 6 + Phase 10 empirical anchor landings inherit the apriori lint signal |
| #6 PROBE_DISAMBIGUATOR | N/A | Verdict is structurally deterministic per canonical regex set; no probe needed |

## 12th canonicalization × standardization × ease-of-contest-compliance trinity

- **Canonical helper**: ONE `lint_submission_bundle` entry point at
  `tac.submission_packet.lint_submission_bundle`.
- **Canonical CLI**: ONE `tools/submission_linter_cli.py` per the
  operator-facing 4-layer pattern (sister of
  `tools/list_canonical_equations.py` /
  `tools/check_predecessor_probe_outcome.py`).
- **Canonical cathedral consumer**: ONE
  `tac.cathedral_consumers.submission_linter_consumer` per Catalog #335.
- **Canonical 6 exit codes**: 0 LINT-CLEAN / 1 FORBIDDEN-TOKEN /
  2 FIRST-PERSON-PLURAL / 3 EMDASH / 4 INFLATE-PY-OVER-BUDGET /
  5 TONE-VIOLATION / 6 CLI-error.

## 13th OPTIMAL-TRIO declaration

- **AUTOMATED**: cathedral autopilot auto-discovers + invokes the
  consumer per Catalog #335/#336/#337; CLI is operator-facing manual
  trigger for one-off lint passes; STRICT preflight gate (Phase 8 sister)
  will fire on every PR-submission candidate by structural composition.
- **COMPOUNDING**: per the 12th canonicalization principle — every
  future PR111-candidate landing uses THE SAME linter; per-PR lint
  drift is structurally extincted.
- **OPTIMAL**: linter is pure-Python regex with no GPU + no paid
  dispatch; runs in milliseconds; observability-only by construction.

## Catalog #370 — claimed but not used

I claimed Catalog #370 at session start (commit `82206aec8` "state: claim
catalog #370 (git-transactional)") expecting to land a STRICT preflight
gate as the 4th surface of the canonical 4-layer pattern. Per the
operator prompt the 4-layer pattern for Phase 5 is (1) canonical helper
module + (2) operator-facing CLI + (3) cathedral consumer + (4)
**Phase 8 = Catalog #362 STRICT preflight gate** (separate phase per
spec memo §3 Phase 8). Phase 5 does NOT land a STRICT preflight gate;
Catalog #370 is therefore **claimed-but-unused** per Catalog #110/#113
APPEND-ONLY HISTORICAL_PROVENANCE discipline (the claim is preserved;
the gate is reserved for future use OR explicit RETIRE via Catalog #299
"stop and consolidate" pause discipline). Operator-routable next:
either (a) Phase 8 lands `Catalog #370` as the canonical sister of
`Catalog #362` (if numbering needs to remain monotonic per Catalog #118),
or (b) Phase 8 lands `Catalog #362` per spec memo + Catalog #370 is
RETIRED via the next Catalog #299 audit pass.

## Sister coordination

- **Phase 2** `b96329a71` (compression_pipeline): consumed via
  `CompressionPipelineResult` in the test fixture; no collision.
- **Phase 3** `1d4753f65` (archive_grammar): consumed via
  `ArchiveGrammarManifest` in the test fixture; no collision.
- **Phase 4** `1de30160e` (builder): consumed via
  `SubmissionBundleResult` as the canonical lint input; sister
  `_FORBIDDEN_PUBLIC_PR_TOKENS` mirrored into the linter's public
  `FORBIDDEN_PUBLIC_PR_TOKENS` for downstream consumer routing.
- **Phase 6 parallel spawn** (compliance enforcer): sister-disjoint per
  operator prompt; sister added `tac.submission_packet.compliance`
  imports to `__init__.py` during my work; merged cleanly (`__init__.py`
  shows BOTH Phase 5 + Phase 6 imports + `__all__` entries in sequence
  with NO collision). The 5th lint surface `lint_compliance_placeholder`
  surfaces parallel Phase 6 compliance sidecar JSON without invoking
  the compliance subprocess directly — the sister-disjoint boundary
  is respected by construction.
- **No other in-flight subagents touched the lint surface during my
  work** per `.omx/state/subagent_progress.jsonl` audit.

## Operator-routable next

1. **Phase 7** `paired_auth_eval` — Layer 5 per spec memo. Consumes
   `SubmissionBundleResult` + emits `PairedAuthEvalResult` for
   paired CPU+CUDA on 1:1 contest-compliant hardware. Routes Catalog
   #226 / #245 / #339 / #360 / #192 / #313.
2. **Phase 8** STRICT preflight gate `Catalog #362` — `check_pr_submission_packet_canonical`.
   Refuses `submissions/*/` dirs that fail any of the linter's ERROR
   rules. Per Catalog #299 quota brake: catalog # is currently 370
   (just-claimed) well under 400 quota. Decision: either land as
   #362 per spec memo (deprecate #370) or land as #370 (sister of
   #362; rename in spec). Operator-routable.
3. **Phase 10** first PR111-candidate end-to-end regression — promotes
   ALL 4 FORMALIZATION_PENDING equations (compression_pipeline,
   archive_grammar, submission_bundle, submission_linter) to REGISTERED
   via the canonical posterior anchor at
   `tac.canonical_equations.update_equation_with_empirical_anchor`.

## Lane

`lane_phase_5_submission_linter_canonical_helper_20260526` L1
(impl_complete + memory_entry).

## Cross-references

- Phase 1 spec memo: `.omx/research/canonical_submission_pipeline_specification_memo_20260526.md`
- Phase 4 sister landing: `feedback_phase_4_submission_bundle_canonical_landed_20260526.md`
- PR 95 medal-class study: `feedback_pr_95_full_deep_research_landed_20260519T192300Z.md`
- User attribution memory: `~/.claude/projects/-Users-adpena-Projects-pact/memory/user_pr_attribution.md`
- Forbidden Claude attribution memory: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_forbidden_claude_attribution_in_public_pr_surfaces.md`
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L4
- CLAUDE.md "Public Disclosure Hygiene"
- CLAUDE.md "Apples-to-apples evidence discipline"
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
- Catalog #205 (inline device fork) + #208 (docs no local paths) +
  #287 (placeholder rationale rejection) + #295 (PYTHONPATH self-
  containment) + #323 (canonical Provenance) + #335 (cathedral
  consumer canonical contract) + #341 (canonical routing markers) +
  #344 (canonical equations registry).
