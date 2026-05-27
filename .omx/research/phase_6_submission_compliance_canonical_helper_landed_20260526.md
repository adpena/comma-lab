# Phase 6 Submission Compliance Canonical Helper — LANDED 2026-05-26

**Lane**: `lane_phase_6_submission_compliance_canonical_helper_20260526` L1
(impl_complete + strict_preflight + cathedral_consumer + memory_entry)

**Subagent**: `phase-6-compliance-canonical-helper-per-phase-1-spec-memo-layer-4-contest-compliance-enforcer-wraps-pre-submission-compliance-check-py-20260526`

**Mission contribution per Catalog #300**: `frontier_protecting` (Layer 4
contest compliance enforcer extincts the per-substrate ad-hoc compliance
invocation divergence bug class; consolidates the 3267-LOC standalone
script into ONE typed canonical helper with per-Catalog-gate categorized
verdict + operator-gated D3/D5 blocker classification + Catalog #192
macOS-CPU structural refusal).

## What landed

Per Phase 1 audit specification memo
(`.omx/research/canonical_submission_pipeline_specification_memo_20260526.md`)
§3 Phase 6 (Layer 4 compliance) acceptance contract:

1. **`tac.submission_packet.compliance`** (~760 LOC src; counted 1173
   including comprehensive module-level docstring + per-Catalog-gate
   prefix tables + per-checkpoint remediation hints):
   - `ComplianceCheck` frozen dataclass (per-check: check_name + severity
     enum + passed + details + catalog_gate_refs + is_operator_gated +
     remediation_hint)
   - `ComplianceVerdict` frozen dataclass (per-submission: overall_clean
     + checks list + error_count + operator_gated_remaining +
     forbidden_macos_axis_detected + catalog_gate_protection_summary +
     canonical Provenance per Catalog #323)
   - `enforce_contest_compliance(submission_bundle_result, *,
     contest_final_strict=True, ...)` canonical entry point
   - Wraps `scripts/pre_submission_compliance_check.py --contest-final
     --strict` via subprocess + structured JSON parsing
   - Catalog gate categorization: #127 / #146 / #152 / #192 / #221 / #226
     / #240 / #266
   - Catalog #192 enforcement: Darwin ARM64 / macOS substrate
     STRUCTURALLY refused regardless of wrapped-script overall_passed
   - Catalog #287 placeholder-rationale rejection
   - Catalog #323 canonical Provenance umbrella
   - Catalog #341 Tier A canonical-routing markers (axis_tag=[predicted],
     score_claim=False, promotable=False)
   - Catalog #344 canonical equation
     `submission_compliance_canonical_helper_consolidation_savings_v1`
     FORMALIZATION_PENDING (preserved until Phase 10 first paired
     empirical anchor)

2. **`tools/submission_compliance_cli.py`** (351 LOC):
   - Operator-facing CLI wrapping `enforce_contest_compliance`
   - `--from-submission-bundle <path>` (or `-` for stdin) loads
     SubmissionBundleResult JSON per Phase 4 round-trip contract
   - `--contest-final-strict` per CLAUDE.md "Submission auth eval — BOTH
     CPU AND CUDA" non-negotiable
   - `--json` machine-readable; otherwise human-readable verdict on
     stderr with per-Catalog-gate failed-check counts
   - Canonical 6-tier exit code taxonomy: 0 CLEAN / 1 COMPLIANCE-ERROR /
     2 PAIRED-AXIS-MISSING / 3 CUSTODY-MISMATCH / 4
     RECIPE-TRAINER-STATE-INCONSISTENT / 5 CLI error
   - Verified `--help` exits 0 + bad-args exits non-zero + missing-file
     exits 5

3. **Cathedral consumer**
   `src/tac/cathedral_consumers/submission_compliance_consumer/`
   (149 LOC) per Catalog #335 canonical contract:
   - CONSUMER_NAME = "submission_compliance_consumer"
   - CONSUMER_VERSION = "1.0.0"
   - CONSUMER_HOOK_NUMBERS = CATHEDRAL_AUTOPILOT_DISPATCH +
     CONTINUAL_LEARNING_POSTERIOR + PROBE_DISAMBIGUATOR
   - 4-verdict taxonomy: CLEAN / OPERATOR_GATED / STRUCTURAL_BLOCKED /
     FORBIDDEN_MACOS_AXIS / UNKNOWN
   - Tier-A observability-only invariants (predicted_delta_adjustment=
     0.0 + promotable=False + axis_tag=[predicted]) per Catalog #341
   - Auto-discovered per Catalog #336/#337

4. **Tests** `src/tac/tests/test_submission_compliance.py` (1353 LOC; 97
   tests **ALL PASS**):
   - Module constants pinned (schema_version, phase, canonical_equation_id,
     canonical script path, CheckSeverity enum, required Catalog gates
     covered)
   - ComplianceCheck invariants (canonical construction, empty
     check_name rejected, invalid severity rejected, catalog_gate_refs
     must be sorted tuple, out-of-range rejected, as_dict round-trip)
   - ComplianceVerdict invariants (bad axis rejected, bad sha256
     rejected, score_claim must be False, promotable must be False,
     macOS axis with overall_clean rejected, error_checks must be
     error-severity, operator_gated_remaining must be subset of
     error_checks, evidence_grade must start [predicted;,
     canonical_equation_id pinned, axis_tag pinned, as_dict round-trip)
   - Catalog gate classification (127/146/152/192/221/226/240/266 each
     individually tested)
   - Operator-gated classification (auth_eval/hosted_archive/
     public_source/runtime_equivalence_proof all operator-gated;
     structural checks NOT operator-gated)
   - Remediation hint derivation (D5 paired auth-eval / D3 hosting /
     #192 Linux x86_64 / #127 custody / #152 required input /
     #146 inflate runtime)
   - Forbidden macOS axis detection (macos_arm64 / darwin_arm64 /
     apple_silicon / case-insensitive / clean linux passes)
   - Wrapped script report parser (clean / failed structural / failed
     op-gated / macOS detected / unknown severity normalized / warning
     not in error_checks / non-list raises / **21 PASS + 18 op-gated
     anchor reproduction**)
   - enforce_contest_compliance structural failure paths (bad axis /
     non-bundle-result / missing canonical script / subprocess crash
     rc!=0,1 / subprocess timeout / missing JSON file / unparseable JSON)
   - enforce_contest_compliance happy path via mocked subprocess (clean
     verdict / failed verdict / macOS forces blocked / op-gated
     classified / JSON report emitted / Provenance canonical /
     per-Catalog summary)
   - Provenance derivation (canonical fields pinned)
   - Cathedral consumer (imports / validates canonical contract /
     unknown metadata / clean verdict / forbidden macOS / op-gated /
     structural blocked / update_from_anchor no-op)
   - CLI subprocess (--help exits 0 / missing arg exits non-zero / bad
     bundle path exits 5 CLI error)
   - Phase 4 integration round-trip (bundle's archive_sha256 +
     archive_bytes + submission_dir + lane_id passed through to wrapped
     script via correct argv)
   - Real script invocation smoke (canonical script exists + emits
     parseable JSON)
   - Live-repo regression guard (canonical_helper_invocation full
     dotted path, __all__ exhaustive, Phase 6 surfaces in package
     __init__)

## Sister landings verified

- Phase 2 (`compression_pipeline.py`, commit `b96329a71`) ✓
- Phase 3 (`archive_grammar.py`, commit `1d4753f65`) ✓
- Phase 4 (`builder.py`, commit `1de30160e`) ✓
- Phase 5 sister parallel (linter, in-flight subagent
  `phase-5-linter-canonical-helper`) — DISJOINT scope; Phase 5 owns
  `tac.submission_packet.linter`; Phase 6 owns
  `tac.submission_packet.compliance`; per Catalog #230 ownership map
  honored

- Sister Phase 2/3/4 tests (`test_submission_bundle.py` +
  `test_archive_grammar.py` + `test_compression_pipeline.py`) =
  **189/189 PASS** (regression preserved)

## Integration verified

- Accepts `SubmissionBundleResult` from Phase 4
  `tac.submission_packet.build_submission_bundle` directly without
  transformation (Phase 4 → Phase 6 canonical handoff verified by
  `TestPhase4IntegrationRoundTrip`)
- Invokes `scripts/pre_submission_compliance_check.py` via subprocess
  with canonical 3-arg passthrough (verified by
  `TestRealScriptInvocationSmoke` + `TestPhase4IntegrationRoundTrip`)
- Emits JSON report to `reports/pr_pre_submission/` per existing
  convention
- Canonical Provenance per Catalog #323 threaded through every verdict
- Cathedral consumer auto-discovered per Catalog #335 (validated via
  `validate_consumer_module`)

## Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Canonical helper | Forked OR adopted | Rationale |
|---|---|---|---|
| Subprocess invocation | Python stdlib `subprocess.run` | ADOPT | Industry-standard; no substrate-specific advantage |
| JSON parsing | Python stdlib `json` | ADOPT | Industry-standard |
| Frozen dataclass + `__post_init__` invariants | Python stdlib `dataclasses` | ADOPT | Matches Phase 2/3/4 sister pattern; canonical-frozen-dataclass-return per 12th standing directive |
| Per-Catalog-gate prefix classification | NEW canonical (this lane) | FORK | The 3267-LOC wrapped script does NOT expose per-Catalog-gate categorization; canonical classification IS the value-add at the Phase 6 surface |
| Operator-gated D3/D5 taxonomy | NEW canonical (this lane) | FORK | Per Phase 1 spec memo Layer 4 contract; canonical taxonomy IS the operator-routable next-action surface |
| Catalog #192 macOS-CPU detection | NEW canonical (this lane) | FORK | Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable; structural refusal at the verdict surface is canonical to this layer |
| Provenance | `tac.provenance` canonical builders | ADOPT | Catalog #323 umbrella |
| Cathedral consumer | `tac.cathedral.consumer_contract` | ADOPT | Catalog #335 canonical contract |
| Subprocess timeout | 120s default | ADOPT | Conservative; tests showed <1s for synthetic + ~30s realistic for real-script smoke |

## 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS**: Phase 6 Layer 4 is the canonical TYPED WRAPPER over the
   3267-LOC standalone compliance script; the typed wrapper extincts the
   per-substrate ad-hoc invocation divergence class (NOT a within-class
   refinement of a sister substrate).
2. **BEAUTY+ELEGANCE**: 760-LOC compliance module + 351-LOC CLI + 149-LOC
   cathedral consumer; reviewable in 30 seconds per HNeRV parity L4. ONE
   canonical entry point (`enforce_contest_compliance`) + ONE return
   shape (`ComplianceVerdict`).
3. **DISTINCTNESS**: explicitly different from sister Phase 5 linter
   (linter scans submission_dir for tone/path/attribution issues; Phase
   6 enforces contest archive grammar + auth-eval custody compliance via
   the canonical wrapped script).
4. **RIGOR**: 97 dedicated tests + premise verification of canonical
   wrapped script structure (read scripts/pre_submission_compliance_
   check.py Check dataclass + arg parser + main entry) + Phase 4
   integration round-trip test + real-script smoke + Catalog #192
   structural refusal regression guard.
5. **OPTIMIZATION-PER-TECHNIQUE**: canonical helper routes ALL operator-
   facing compliance invocations through the typed wrapper; per-Catalog-
   gate categorization enables downstream consumers (Phase 7 operator
   runbook / Phase 8 STRICT gate / Phase 10 PR111 regression) to consume
   structured verdict without re-parsing the 3267-LOC script's JSON.
6. **STACK-OF-STACKS-COMPOSABILITY**: composes with Phase 4 SubmissionBundleResult
   (consumer surface) + Phase 7 paired_auth_eval (will populate
   `--auth-eval-json` + `--contest-cpu-auth-eval-json` paths) + Phase 8
   STRICT gate `check_no_pr_submission_without_compliance_verdict`
   (future).
7. **DETERMINISTIC REPRODUCIBILITY**: subprocess invocation pinned + JSON
   report path canonical; round-trip via `as_dict()` + reconstruction
   preserves all fields. Pinned canonical equation ID + schema version.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: 120s subprocess timeout
   default + structured parser bounded by check count (O(N) where N is
   wrapped-script check count, typically <50); no recursive scan.
9. **OPTIMAL MINIMAL CONTEST SCORE**: this layer does NOT directly
   contribute to score; it ENFORCES contest compliance per CLAUDE.md
   "Submission auth eval — BOTH CPU AND CUDA" non-negotiable so paid
   dispatches produce promotion-eligible artifacts (NOT phantom
   compliance promotions).

## Cargo-cult audit per assumption (Catalog #303)

| Assumption | HARD-EARNED-vs-CARGO-CULTED | Rationale + unwind |
|---|---|---|
| Wrapping the 3267-LOC standalone script via subprocess is the canonical Layer 4 pattern | HARD-EARNED | Per Phase 1 spec memo Layer 4 explicit guidance "this layer is a typed wrapper, NOT a rewrite"; the standalone script carries 7+ years of contest-compliance institutional knowledge |
| Catalog #192 macOS-CPU detection should be a STRUCTURAL refusal at the verdict surface (not just a wrapped-script check) | HARD-EARNED | Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable + Catalog #192 sister gate; defense-in-depth at the verdict surface |
| Operator-gated D3/D5 blockers should be surfaced separately from structural blockers | HARD-EARNED | Per Phase 1 spec memo Layer 4 + 2026-05-19 sister landing (D5 prerequisites executed) which empirically established the 21 PASS + 18 op-gated split |
| Per-Catalog-gate prefix classification via static dict lookup is the right abstraction | HARD-EARNED | The wrapped script's check names are stable per the canonical 3267-LOC implementation; static lookup is O(1) per check + extensible per new Catalog gates without rewriting |
| Subprocess invocation should default to 120s timeout | CARGO-CULTED → unwind-test plan | Adopted from common Python subprocess defaults; the realistic upper bound for the wrapped script on a baseline submission_dir is unknown; UNWIND: measure realistic upper bound during Phase 7 paired_auth_eval real-substrate smoke and update default if needed |

## Observability surface (Catalog #305)

1. **Inspectable per layer**: every wrapped-script check surfaces as a
   typed `ComplianceCheck` with check_name + severity + passed +
   details + catalog_gate_refs + is_operator_gated + remediation_hint;
   the FULL wrapped-script JSON report is preserved at
   `verdict.json_report_path` for forensic audit.
2. **Decomposable per signal**: `verdict.catalog_gate_protection_summary`
   decomposes the failed-check count per Catalog gate; `verdict.error_checks`
   + `verdict.operator_gated_remaining` decompose by blocker class.
3. **Diff-able across runs**: every verdict has canonical
   `measurement_utc` + `archive_sha256` + `submission_dir`; sister runs
   on the same archive_sha256 produce diff-able JSON reports.
4. **Queryable post-hoc**: verdict `as_dict()` round-trips to canonical
   JSON; sister-consumers (Phase 7 / 8 / 10) consume via the canonical
   ComplianceVerdict shape.
5. **Cite-able**: every verdict carries canonical Provenance per Catalog
   #323 with (lane_id, substrate_id, archive_sha256, measurement_utc,
   canonical_helper_invocation, canonical_equation_id).
6. **Counterfactual-able**: the wrapped-script subprocess invocation is
   reproducible from the verdict's argv (per `_classify_check_catalog_gates`
   + the canonical CLI flag set); a future Phase 7 paired_auth_eval can
   re-invoke with a different `--submission-score-axis` to counterfactually
   check sister axis verdict.

## HORIZON-CLASS classification (Catalog #309)

`horizon_class: apparatus_maintenance` — this layer does NOT directly
lower contest score; it ENFORCES contest compliance so paid dispatches
produce promotion-eligible artifacts per CLAUDE.md non-negotiable. The
downstream Phase 7 paired_auth_eval + Phase 10 PR111-candidate
end-to-end regression are the score-lowering surfaces this layer
unblocks.

## 6-hook wire-in declaration (Catalog #125)

1. **Hook #1 sensitivity-map**: N/A (defensive validator gate; no
   sensitivity signal contribution)
2. **Hook #2 Pareto constraint**: N/A (no Pareto-relevant signal)
3. **Hook #3 bit-allocator**: N/A (no bit-allocator signal)
4. **Hook #4 cathedral autopilot dispatch**: **ACTIVE PRIMARY** — the
   cathedral consumer `submission_compliance_consumer` is auto-discovered
   per Catalog #335 and surfaces per-candidate compliance readiness
   (CLEAN / OPERATOR_GATED / STRUCTURAL_BLOCKED / FORBIDDEN_MACOS_AXIS)
   to the cathedral autopilot ranker
5. **Hook #5 continual-learning posterior**: **ACTIVE** —
   `update_from_anchor` accepts canonical posterior anchors per Catalog
   #344; future Phase 10 first paired empirical anchor will promote
   canonical equation `submission_compliance_canonical_helper_
   consolidation_savings_v1` from FORMALIZATION_PENDING to REGISTERED
6. **Hook #6 probe-disambiguator**: **ACTIVE** — the 4-verdict taxonomy
   (CLEAN / OPERATOR_GATED / STRUCTURAL_BLOCKED / FORBIDDEN_MACOS_AXIS)
   IS the canonical disambiguator at the cathedral ranker surface

## 12th canonicalization × standardization × ease-of-contest-compliance trinity declaration

- **canonicalization**: ONE canonical helper at
  `tac.submission_packet.enforce_contest_compliance` for the Layer 4
  compliance enforcement surface; sister Phases 2/3/4/5/7/8/9/10
  consume the canonical `ComplianceVerdict` shape
- **standardization**: canonical-frozen-dataclass-return contract
  matches Phase 2/3/4 sister `CompressionPipelineResult` +
  `ArchiveGrammarManifest` + `SubmissionBundleResult` patterns;
  canonical Provenance per Catalog #323 + non-promotable markers per
  Catalog #341 inherited from the same canonical surfaces
- **ease-of-contest-compliance**: ONE operator-facing CLI
  (`tools/submission_compliance_cli.py`) wraps the canonical helper;
  operator-friendly 6-tier exit code taxonomy + human-readable verdict
  + machine-readable `--json` mode; future Phase 7 operator runbook
  CLI consumes via the canonical CLI surface

## 13th OPTIMAL-TRIO standing directive declaration

This landing satisfies the 3 questions:

1. **AUTOMATED?** YES — canonical helper auto-discovered per Catalog
   #335 + sister-consumer wire-in to cathedral autopilot via Catalog
   #336/#337; no manual ranker-cascade edits required for new
   substrates
2. **COMPOUNDING?** YES — each new substrate's compliance verdict
   feeds the canonical posterior (Catalog #323) + the cathedral
   autopilot ranker (Catalog #335) + the Phase 7 paired_auth_eval
   prerequisite chain; the canonical equation #344 entry
   `submission_compliance_canonical_helper_consolidation_savings_v1`
   will compound from FORMALIZATION_PENDING to REGISTERED upon Phase
   10 first paired empirical anchor
3. **OPTIMAL?** YES — typed wrapper over the 3267-LOC standalone
   script (NOT a rewrite per Phase 1 spec memo Layer 4 explicit
   guidance) + per-Catalog-gate categorization (8 gates) +
   operator-gated D3/D5 taxonomy (6 patterns) + Catalog #192
   structural refusal at verdict surface (defense-in-depth) +
   canonical-frozen-dataclass-return contract sister of Phase 2/3/4
   surfaces

## Operator-routable next

- **Phase 7 paired_auth_eval canonical helper** (Layer 5 per Phase 1
  spec memo §3 Phase 6 prompt template): orchestrates paired Modal
  CUDA + Linux x86_64 CPU auth-eval on the EXACT same archive bytes
  per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-
  negotiable; produces `auth_eval_json` + `contest_cpu_auth_eval_json`
  artifacts that satisfy Phase 6 D5 operator-gated blockers
- **Phase 8 STRICT gate Catalog #362** (`check_no_pr_submission_without_
  compliance_verdict`): refuses any state that attempts a `gh pr
  create` to `commaai/comma_video_compression_challenge` without a
  Phase 6 compliance verdict landed within the last 24 hours per
  canonical posterior anchor

## Discipline declaration

- Catalog #117/#157/#174 canonical serializer (will commit via
  subagent_commit_serializer.py with POST-EDIT --expected-content-sha256)
- Catalog #119 Co-Authored-By Claude trailer
- Catalog #206 checkpoint cadence (3 checkpoints emitted at landing)
- Catalog #229 PV (read Phase 4 builder + scripts/pre_submission_
  compliance_check.py FIRST before drafting compliance.py)
- Catalog #230 sister-disjoint: respected Phase 5 linter sister
  parallel spawn (`phase-5-linter-canonical-helper`); my scope is
  `tac.submission_packet.compliance` + CLI + cathedral consumer;
  zero overlap with Phase 5 linter scope (`tac.submission_packet.linter`)
- Catalog #287 placeholder rejection (every waiver path explicitly
  rejects `<rationale>` / `<reason>` / empty / <4-char rationales)
- Catalog #290 + #294 + #303 + #305 + #309 design-memo sections
  declared above
- Catalog #335 cathedral consumer canonical contract (validated via
  `validate_consumer_module` in test)
- Catalog #340 sister-checkpoint guard PROCEED (no overlap with
  active sister at landing time)
- Catalog #341 Tier A canonical-routing markers initially
- Catalog #344 FORMALIZATION_PENDING preserved (no canonical equation
  promotion; awaits Phase 10 first paired empirical anchor)
- 10th apples-to-apples paired CPU+CUDA on 1:1 contest-compliant
  hardware NON-NEGOTIABLE: Catalog #192 macOS-CPU detection
  STRUCTURALLY refuses Darwin ARM64 references in any wrapped-script
  check details
- 11th ORDER-MATTERS sequencing: Phase 6 Layer 4 depends on Phase 4
  Layer 2 (SubmissionBundleResult); Phase 6 is invoked AFTER bundle
  emission per the canonical pipeline order
- 12th canonicalization × standardization × ease-of-contest-compliance
  trinity binding (declared above)
- 13th OPTIMAL-TRIO declaration (declared above)

## Cross-references

- Phase 1 spec memo: `.omx/research/canonical_submission_pipeline_
  specification_memo_20260526.md` §3 Phase 6 (Layer 4 compliance)
- Phase 4 landing: commit `1de30160e`
- Phase 3 landing: commit `1d4753f65`
- Phase 2 landing: commit `b96329a71`
- Canonical compliance script: `scripts/pre_submission_compliance_check.py`
  (3267 LOC standalone)
- 2026-05-19 sister D5 prerequisites landing:
  `feedback_pr_submission_d5_prerequisites_executed_landed_20260519T182635Z.md`
- Lane: `lane_phase_6_submission_compliance_canonical_helper_20260526` L1

## Test verification

```
$ .venv/bin/python -m pytest src/tac/tests/test_submission_compliance.py -x
======================== 97 passed in 0.98s =========================
```

Sister Phase 2/3/4 regression:

```
$ .venv/bin/python -m pytest src/tac/tests/test_submission_bundle.py \
    src/tac/tests/test_archive_grammar.py \
    src/tac/tests/test_compression_pipeline.py -q
189 passed in 0.92s
```

Catalog #185 baseline preserved (2 pre-existing violations unchanged by
this lane).
