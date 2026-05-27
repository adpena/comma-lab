# Phase 8 STRICT gate Catalog #370 — canonical-submission-pipeline LANDED 2026-05-26

**Lane**: `lane_phase_8_strict_gate_canonical_submission_compliance_20260526` L1
(impl_complete + strict_preflight + cathedral_consumer + memory_entry)

**Subagent**: `phase_8_strict_gate_c4d75058` (parent session
`b74f6039-6caf-44f2-a2c3-cd8156acd447`)

**Mission contribution per Catalog #300**: `apparatus_maintenance`
(closes the canonical-submission-pipeline 7-layer architecture at Layer 6;
THIS gate is the structural protection of the canonical default-path so future
PR111+ submissions are STRUCTURALLY incapable of shipping ad-hoc-hand-edited
without going through the canonical Phase 4 → 5 → 6 → 7 verdict chain;
unblocks the Phase 10 operator-runbook end-to-end CLI which consumes THIS
gate's verdict at pre-`gh pr create` time).

## What landed

Per Phase 1 audit specification memo
(`.omx/research/canonical_submission_pipeline_specification_memo_20260526.md`)
§3 Phase 8 (Layer 6 STRICT gate) acceptance contract:

| Surface | Path | LOC |
|---|---|---|
| Canonical STRICT preflight gate | `src/tac/preflight.py` (~660 LOC inserted) | +660 |
| Cathedral consumer | `src/tac/cathedral_consumers/pr_submission_compliance_consumer/__init__.py` | 235 |
| Tests | `src/tac/tests/test_check_370_no_pr_submission_without_canonical_compliance.py` | 605 |
| CLAUDE.md catalog row | `CLAUDE.md` line 3757 | +1 |
| Retroactive sweep memo | `.omx/research/retroactive_sweep_for_catalog_370_20260527T040829Z.md` | 120 |
| Landing memo | THIS file | 175 |

**Tests**: 46 pass. Live count: 4 PR-facing submissions without canonical
4-verdict chain (warn-only baseline canonical per Phase 1 spec memo prediction).

## Canonical contract per Phase 1 spec memo §3 Phase 8

`tac.preflight.check_no_pr_submission_without_canonical_compliance_verdict(*,
repo_root=None, strict=False, verbose=False) -> list[str]` scans
`submissions/*/` directories (excluding `submissions/exact_current/` per
CLAUDE.md mutation frontier; excluding `_intake_` vendored clones) for
PR-facing artifacts:

1. **PR-facing trigger** — `PR_BODY*.md` / `PR_DESCRIPTION.md` (existence
   alone qualifies) OR `README.md` body carrying canonical sentinel tokens
   (`# PR ` / `## Submission` / `## Score` / `## Reproducibility` /
   `## Attribution` / `## Citations` / `[contest-CPU]` / `[contest-CUDA]` /
   `commaai/comma_video_compression_challenge`).

2. **Required 4-verdict chain** (all must be present + clean):
   - Phase 4 builder `SubmissionBundleResult` (`overall_pass=true`)
   - Phase 5 linter `LintVerdict` (`overall_clean=true`)
   - Phase 6 compliance `ComplianceVerdict` (`overall_clean=true`)
   - Phase 7 paired_auth_eval `PairedAuthEvalVerdict` (`verdict=PAIRED_PASS`)

3. **Sidecar search cascade** per phase:
   - Inside `submissions/<sub>/` (canonical helper-emitted sidecar)
   - Inside `experiments/results/<sub>*/` (lane-tagged)
   - Inside `reports/pr_pre_submission/` (Phase 6 canonical persistence)
   - Most-recently-modified wins

4. **Acceptance cascade**:
   - (a) All 4 sidecars present + clean (canonical default-path)
   - (b) Same-line `# PR_SUBMISSION_NO_CANONICAL_COMPLIANCE_OK:<rationale>`
     waiver in README.md or PR_BODY*.md first 30 lines (non-placeholder
     rationale ≥4 chars; placeholder `<rationale>` / `<reason>` literals
     rejected per Catalog #287)
   - (c) Submission is not PR-facing (no PR_BODY*.md AND README.md lacks
     PR-facing sentinel tokens)

5. **Violation messages** include operator-actionable remediation hints per
   CLAUDE.md "Operator gates must be wired and used": each missing phase
   names the canonical CLI command to generate the sidecar.

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Helper function structure | ADOPT_CANONICAL | sister of `check_substrate_inflate_consumes_real_trained_weights_not_synthetic_frame_base` (Catalog #369) at the per-file scan + acceptance-cascade + waiver pattern surface |
| Cathedral consumer | ADOPT_CANONICAL | sister of `submission_compliance_consumer` (Phase 6) at the same per-candidate readiness annotation surface; Catalog #335 contract honored |
| CLAUDE.md row | ADOPT_CANONICAL | sister of all 369 prior catalog entries; required by Catalog #176 META-meta |
| Catalog #348 retroactive sweep | ADOPT_CANONICAL | sister of `retroactive_sweep_for_catalog_368/369*.md` 4-field contract |
| Test file structure | ADOPT_CANONICAL | sister of `test_check_365_366_367_cascade_c_prime_bug_class_extinction.py` tmp_path fixture pattern |
| Catalog # selection | FORK (Option A) | per spawn directive: Option A = use Catalog #370 (claimed at session start commit `82206aec8` per Phase 5 op-routable resolution); honors apparatus efficiency + Catalog #299 quota brake |
| Waiver token | NEW (unique-and-complete-per-method) | `PR_SUBMISSION_NO_CANONICAL_COMPLIANCE_OK` per-gate canonical; no sister waiver exists |
| Sidecar discovery (3-tier cascade) | NEW (unique-and-complete-per-method) | Layer 6 spec requires consulting 3 canonical persistence locations (submission_dir + experiments/results + reports/pr_pre_submission); no sister cascade exists |

## 9-dimension success checklist evidence

1. **UNIQUENESS** — Catalog #370 is the canonical Phase 8 Layer 6 surface; no
   sister gate exists at this surface (Phase 6 compliance / Phase 5 linter /
   Phase 4 builder are upstream sisters that this gate composes).
2. **BEAUTY + ELEGANCE** — ~660 LOC + 7 per-helper functions + 1 main entry
   point; reviewable in 30 seconds per PR101 medal-class precedent; canonical
   data flow: scan → trigger-classify → waiver-check → 4-verdict-evaluate →
   format-violation.
3. **DISTINCTNESS** — Layer 6 STRICT gate is structurally separate from Phase
   6 (which ENFORCES compliance) + Phase 5 (which LINTS) + Phase 4 (which
   BUILDS). THIS gate is REFUSE-AT-SOURCE-LEVEL (preflight surface); it does
   NOT modify any artifact; OBSERVABILITY-+-REFUSAL by construction.
4. **RIGOR** — Premise verification: read full Phase 1 spec + Phase 5/6 sister
   landings + Phase 7 paired_auth_eval in-flight scope + submissions/ + JSON
   sidecar conventions + canonical Provenance shapes BEFORE writing a line.
   Per-finding fix_suggestion ≥4 chars + placeholder rationale rejection per
   Catalog #287. 46 dedicated tests covering all acceptance cascades + sister
   cross-references.
5. **OPTIMIZATION PER TECHNIQUE** — Per-phase regex tokenization pre-compiled
   at module load; sidecar discovery scoped by `submission_dir.name`-anchored
   glob (avoids scanning unrelated experiment results); mtime-based latest-wins
   reduces I/O; gate runs in 30ms on live repo.
6. **STACK-OF-STACKS COMPOSABILITY** — Layer 6 sits cleanly between Layer 5
   (paired_auth_eval) and Phase 10 (operator-runbook CLI); consumes 4
   canonical verdict sidecars cleanly; cathedral consumer observes
   PR-submission readiness without mutating ranker.
7. **DETERMINISTIC REPRODUCIBILITY** — All sentinel tokens / sidecar filenames
   / glob patterns pinned at module load; sorted submission_dir iteration;
   deterministic finding emission order; canonical JSON parsing.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — Pure-Python file-system + regex
   over bounded submission directories (~50 in live repo); runs in
   milliseconds. No GPU + no paid dispatch.
9. **OPTIMAL MINIMAL CONTEST SCORE** — Gate is structural-refusal only per
   Catalog #341; it does NOT contribute to score directly; it structurally
   extincts the bug class of submitting a PR without canonical 4-verdict
   chain (saving the 3h × 4-subagent hand-editing cycle the 2026-05-19 PR101
   anchor exhibited; the canonical pipeline collapses to <60 seconds per
   Phase 10 acceptance criterion #5).

## Cargo-cult audit per assumption

| Assumption | HARD-EARNED vs CARGO-CULTED | Unwind path |
|---|---|---|
| `submissions/exact_current/` exempt (pinned upstream snapshot) | HARD-EARNED per CLAUDE.md mutation frontier non-negotiable | N/A; exempt by canonical rule |
| `_intake_` vendored clones exempt (public-PR snapshots) | HARD-EARNED per Catalog #109 sister discipline | N/A; exempt by canonical rule |
| PR-facing sentinel tokens recognize PR101+102+103 medal-class shapes | HARD-EARNED per `feedback_pr_95_full_deep_research_landed_20260519T192300Z.md` study | Add new tokens to `_CHECK_370_PR_FACING_README_TOKENS` as future medal-class shapes emerge |
| 4-verdict chain (Phase 4 + 5 + 6 + 7) is the canonical default-path | HARD-EARNED per Phase 1 spec memo §2 7-layer architecture | N/A; the spec memo IS the canonical authority |
| `dual_eval_adjudicated.json` accepted as Phase 7 fallback | HARD-EARNED per PR101+PR102 medal-class precedent | N/A; preserves canonical-frontier-pointer routing per Catalog #343 |
| Most-recent-mtime sidecar wins | CARGO-CULTED-from-canonical-Modal-call-id-ledger-pattern | Could fork to sha-stable matching if future sidecars carry archive_sha256 cross-reference; deferred to Phase 10 |
| Initial wire-in is WARN-ONLY | HARD-EARNED per CLAUDE.md "Strict-flip atomicity rule" + Phase 1 spec memo §3 Phase 8 acceptance contract | N/A; strict-flip atomic with Phase 10 PR101 baseline migration + first NEW submission both PACKET-CLEAN |

## Observability surface

| Facet | Implementation |
|---|---|
| **Inspectable per layer** | Each helper function is independently callable + testable (`_check_370_iter_submission_dirs` / `_check_370_path_in_scope` / `_check_370_submission_has_pr_facing_artifact` / `_check_370_evaluate_submission` / `_check_370_format_violation`) |
| **Decomposable per signal** | Per-phase verdict carries `{present, clean, sidecar_path}` triple; aggregate violation message decomposes into per-phase missing/unclean sections |
| **Diff-able across runs** | Gate emits the same canonical violation message format every run; mtime-based latest-wins is deterministic given filesystem state |
| **Queryable post-hoc** | Live count + per-submission verdict surfaced via `verbose=True`; cathedral consumer surfaces per-candidate readiness via `consume_candidate` |
| **Cite-able** | Every violation message names the submission_dir + cites Catalog #370 + cites Phase 1 spec memo + cites the canonical CLI command for each missing phase |
| **Counterfactual-able** | Adding/removing a sidecar OR adding/removing a waiver immediately changes the gate's verdict; test suite verifies the cascade |

## 6-hook wire-in declaration per Catalog #125

- **Hook #1 SENSITIVITY_MAP** — N/A (defensive validator gate, no signal contribution)
- **Hook #2 PARETO_CONSTRAINT** — N/A
- **Hook #3 BIT_ALLOCATOR** — N/A
- **Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH** — **ACTIVE PRIMARY** (THIS gate IS the structural protection that PR submissions cannot ship without the full 4-verdict chain; companion cathedral consumer `tac.cathedral_consumers.pr_submission_compliance_consumer` surfaces per-candidate readiness verdicts via Catalog #335 auto-discovery)
- **Hook #5 CONTINUAL_LEARNING_POSTERIOR** — **ACTIVE** (per-submission PR-readiness verdict feeds the canonical posterior so Phase 10 first-PR111-candidate landings inherit the apriori canonical-compliance signal)
- **Hook #6 PROBE_DISAMBIGUATOR** — **ACTIVE** (PR_READY vs BLOCKED_ON_<phase> vs BLOCKED_FORBIDDEN_MACOS_AXIS IS the canonical disambiguator between full-canonical-pipeline-completion vs operator-routable blocker at the cathedral ranker surface)

## Canonical equation candidate (FORMALIZATION_PENDING per Catalog #344)

- **id**: `pr_submission_canonical_compliance_gate_consolidation_savings_v1`
- **shape**: `Δ_wall_clock_per_PR = canonical_pipeline_wall_clock - manual_4_subagent_wall_clock`
- **expected sign**: Δ ≤ 0 (canonical pipeline FASTER than manual)
- **prediction**: <60 seconds canonical vs ~3h × 4 subagents manual = 100x to
  1000x wall-clock reduction per PR submission. Empirical anchor will be
  registered at Phase 10 first PR111-candidate end-to-end regression per
  Phase 1 spec memo §6 Phase 10 acceptance criterion #5.
- **status**: FORMALIZATION_PENDING (per Catalog #344 — required 3+ empirical
  anchors before posterior calibration)

## Sister landings verified

- **Phase 2** (`compression_pipeline.py`, commit `b96329a71`) ✓ Verified via
  `_REQUIRED_PHASE_VERDICTS` cathedral consumer routing
- **Phase 3** (`archive_grammar.py`, commit `1d4753f65`) ✓ Verified via
  Catalog #335 canonical contract sister
- **Phase 4** (`builder.py`, commit `1de30160e`) ✓ Verified via
  `submission_bundle_result.json` canonical sidecar
- **Phase 5** (`linter.py`, commit per `feedback_phase_5_submission_linter_canonical_helper_landed_20260526.md`) ✓
  Verified via `lint_verdict.json` canonical sidecar
- **Phase 6** (`compliance.py`, commit per `feedback_phase_6_submission_compliance_canonical_helper_landed_20260526.md`) ✓
  Verified via `compliance_verdict.json` + `pre_submission_compliance.contest_final.json` canonical sidecars + `tools/submission_compliance_cli.py` CLI
- **Phase 7** (`paired_auth_eval.py`, in-flight sister subagent
  `phase7_paired_183A1060`) ⚠ IN-FLIGHT at landing; gate accepts
  `dual_eval_adjudicated.json` sister sidecar (PR101+PR102 medal-class
  precedent) as Phase 7 fallback so gate is functional regardless of
  in-flight subagent landing state

## Integration verified

- Sister Phase 5 + Phase 6 tests (`test_submission_linter.py` +
  `test_submission_compliance.py`) = **196/196 PASS** (regression preserved)
- New Catalog #370 tests = **46/46 PASS** in 0.66s
- Live-repo gate execution = 30ms (negligible DX overhead)
- Live-repo violation count = 4 (warn-only baseline canonical per Phase 1
  spec memo prediction)
- Orchestrator wire-in `preflight_all()` invokes gate with `strict=False`
  (warn-only initial per Phase 1 spec memo §3 Phase 8)
- CLAUDE.md catalog row line 3757 satisfies Catalog #176 META-meta
- Cathedral consumer at
  `src/tac/cathedral_consumers/pr_submission_compliance_consumer/__init__.py`
  satisfies Catalog #335 canonical contract + Catalog #341 Tier-A
  canonical-routing markers; auto-discovered per Catalog #336/#337

## Sister coordination

- **Phase 7 paired_auth_eval** (in-flight `phase7_paired_183A1060` at landing
  time): SISTER-DISJOINT per Catalog #230 ownership map. Phase 7 owns
  `src/tac/submission_packet/paired_auth_eval.py` +
  `src/tac/cathedral_consumers/paired_auth_eval_consumer/__init__.py` +
  `tools/paired_auth_eval_cli.py`; Phase 8 owns `src/tac/preflight.py`
  insertion + `src/tac/cathedral_consumers/pr_submission_compliance_consumer/`
  + new test file + CLAUDE.md row + memos. ZERO file collision. The Phase 7
  helper's eventual landing extends the canonical 4-verdict chain Phase 8
  consumes (Phase 8 accepts `dual_eval_adjudicated.json` fallback so it works
  regardless of Phase 7 landing state).
- **NSCS06 v8 arith-coded cls_stream PV probe** (in-flight `arith_pv_418c7efb`):
  SISTER-DISJOINT. Probe operates on `.omx/tmp/` arith-coding entropy scripts;
  zero overlap with submission pipeline surfaces.
- **T3 grand council negotiator** (in-flight `t3gc_neg_2cb43695`): SISTER-DISJOINT.
  Council operates on grand-council memos under `.omx/research/`; my landing
  memo lives in Claude memory + `.omx/research/retroactive_sweep_for_catalog_370_*`
  + canonical sister `.omx/research/phase_8_strict_gate_catalog_370_canonical_submission_compliance_landed_20260526.md`.
- **V15 UNIWARD Sister B reactivation** (in-flight `v15_uniward_sister_b_C1990441`):
  SISTER-DISJOINT. V15 operates on UNIWARD per-pixel reactivation criterion 1
  N=200 macOS-CPU advisory probe; zero overlap with submission pipeline.
- **Catalog #340 sister-checkpoint guard**: SISTER-DISJOINT across all 5
  in-flight subagents per `.omx/state/subagent_progress.jsonl` audit at
  landing time.

## Operator-routable next

1. **Phase 7 paired_auth_eval landing** — Layer 5 per spec memo. Phase 8
   accepts `dual_eval_adjudicated.json` sister sidecar but the canonical
   `paired_auth_eval_verdict.json` shape will benefit from Phase 7's typed
   verdict dataclass landing.
2. **Phase 10 first PR111-candidate end-to-end regression** — promotes ALL 5
   FORMALIZATION_PENDING equations (compression_pipeline + archive_grammar +
   submission_bundle + submission_linter + pr_submission_compliance_gate) to
   REGISTERED via the canonical posterior anchor at
   `tac.canonical_equations.update_equation_with_empirical_anchor`. Catalog
   #370 strict-flips from WARN-ONLY to STRICT (Live count: 0 verified) at
   that landing per Phase 10 acceptance criterion #1.
3. **2026-05-19 PR101 baseline backfill sweep** — operator-routable per the 3
   resolution paths in the retroactive sweep memo §4: (a) generate 4 canonical
   sidecars via `tools/submission_*_cli.py` for each of the 4 in-scope
   submissions; (b) add substantive waiver if the submission predates the
   canonical pipeline and will not be re-submitted; (c) strip PR-facing
   tokens from README.md if the submission is research-artifact-only.
4. **CASCADE A FEC10 hybrid / Cascade C' / NSCS06 v8 PR111-candidates** —
   first future PR submissions MUST go through canonical Phase 4 → 5 → 6 → 7
   → 8 chain per Phase 1 spec memo Phase 10 acceptance criterion #3.

## Lane

`lane_phase_8_strict_gate_canonical_submission_compliance_20260526` L1
(impl_complete + strict_preflight + cathedral_consumer + memory_entry).

## Cross-references

- Phase 1 spec memo: `.omx/research/canonical_submission_pipeline_specification_memo_20260526.md`
- Phase 5 sister landing: `feedback_phase_5_submission_linter_canonical_helper_landed_20260526.md`
- Phase 6 sister landing: `feedback_phase_6_submission_compliance_canonical_helper_landed_20260526.md`
- Retroactive sweep per Catalog #348: `.omx/research/retroactive_sweep_for_catalog_370_20260527T040829Z.md`
- 2026-05-19 PR101 cascade anchor: `feedback_pr_submission_d5_prerequisites_executed_landed_20260519T182635Z.md`
- PR 95 medal-class study: `feedback_pr_95_full_deep_research_landed_20260519T192300Z.md`
- User attribution memory: `~/.claude/projects/-Users-adpena-Projects-pact/memory/user_pr_attribution.md`
- Forbidden Claude attribution memory: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_forbidden_claude_attribution_in_public_pr_surfaces.md`

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
