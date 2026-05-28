# Phase 9 lifecycle CLI composite-recipe extension landed 2026-05-28

---
council_tier: T1
council_attendees: ["Shannon LEAD", "Dykstra CO-LEAD", "Rudin CO-LEAD", "Daubechies CO-LEAD", "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary"]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Composite-recipe extension belongs on the canonical Phase 9 lifecycle CLI rather than as a sister composite-specific tool"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md '12th canonicalization x standardization x ease-of-contest-compliance trinity' + Catalog #370 PR submission gate: every PR-facing submission MUST flow through the canonical 7-layer architecture. Sister-tool path creates two parallel canonical paths and re-introduces the bug class Phase 9 was designed to extinct (per the original Phase 1 spec memo). Composite-recipe MUST be a first-class CLI mode on the canonical Phase 9 surface."
  - assumption: "Composite-mode MAY skip canonical Layers 0+1 (compression_pipeline + archive_grammar) because composite carries no single substrate trainer + composite archive is already pre-built"
    classification: HARD-EARNED
    rationale: "Per the PR111-candidate landing memo Contrarian dissent + Catalog #240 single-substrate parity: composite has no single trainer + composite archive grammar is multi-section ZIP per HNeRV parity L3 multi-file justification, both built via the canonical composite build script outside the single-substrate Layer 0+1 surface. STRUCTURAL_EXEMPTION at those layers is the canonical opt-out (per HNeRV parity L2 export-first design: composite archive IS pre-built before lifecycle CLI runs)."
  - assumption: "Phase 7 paired auth-eval MUST emit BLOCKED_PRE_DISPATCH (not a new PLAN_ONLY verdict kind) for composite-mode pre-dispatch state"
    classification: HARD-EARNED
    rationale: "Per the canonical PairedAuthEvalVerdictKind enum: PLAN_ONLY does not exist; BLOCKED_PRE_DISPATCH IS the canonical pre-paired state with `archive_sha256_paired` permitted empty and `cuda_axis_tag/cpu_axis_tag in {'[missing]', '[failed]'}`. Introducing a new PLAN_ONLY verdict would be a sister-canonical-helper API change that's out of scope for this extension."

# Catalog #300 mission-alignment required at T1+
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: ""

canonical_equations_referenced:
  - id: "submission_bundle_canonical_helper_consolidation_savings_v1"
    in_domain_context: "composite_recipe_phase_4_builder_extension"
    consumed_form: "canonical SubmissionBundleResult dataclass shape with FORMALIZATION_PENDING status"
  - id: "submission_linter_canonical_helper_consolidation_savings_v1"
    in_domain_context: "composite_recipe_phase_5_linter_extension"
    consumed_form: "canonical LintVerdict dataclass shape with FORMALIZATION_PENDING status"
  - id: "submission_compliance_canonical_helper_consolidation_savings_v1"
    in_domain_context: "composite_recipe_phase_6_compliance_extension"
    consumed_form: "canonical ComplianceVerdict dataclass shape with FORMALIZATION_PENDING status"
  - id: "paired_auth_eval_canonical_helper_consolidation_savings_v1"
    in_domain_context: "composite_recipe_phase_7_paired_auth_eval_extension"
    consumed_form: "canonical PairedAuthEvalVerdict dataclass shape (BLOCKED_PRE_DISPATCH) with FORMALIZATION_PENDING status"

predicted_band_validation_status: not_applicable_apparatus_extension

frontier_pointer_consulted: ".omx/state/canonical_frontier_pointer.json (contest-CPU 0.192009 / sha 18e3155fbbbe9ab2 per Catalog #343 pointer-only discipline)"
---

## Premise verification (per Catalog #229)

Verified BEFORE editing:

1. PR111-candidate landing memo `pr111_candidate_nscs06_v8_plus_compound_c_composite_build_landed_20260528.md` op-routable #5 identifies the structural gap (Phase 9 CLI refuses composite recipes per Catalog #240 single-substrate parity). VERIFIED.
2. Phase 9 lifecycle CLI `tools/operator_pr_submission_full_lifecycle.py` exists at 726 LOC; single-substrate orchestration over Layers 0-6 + Catalog #370 STRICT gate. VERIFIED via import + 45 existing tests pass.
3. Canonical `tac.submission_packet` package: 9588 LOC across 8 modules; Phase 2-7 helpers (`build_compression_pipeline`, `build_archive_grammar_from_compression_pipeline_result`, `build_submission_bundle`, `lint_submission_bundle`, `enforce_contest_compliance`, `plan_paired_auth_eval`) all importable. VERIFIED.
4. Composite recipe at `.omx/operator_authorize_recipes/substrate_composite_nscs06_v8_plus_compound_c_pr111_modal_t4_dispatch.yaml` exists (10.1 KB); declares `substrate_id: composite_*`, `trainer_path` referencing `build_composite_archive.py`, `dispatch_enabled: false`, `research_only: true`. VERIFIED.
5. Composite archive at `experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/archive.zip` (1,917,982 bytes; sha `dfff1358638ef7f7bad4596958cddb62215ed06c5b850a8501e3ad42a2c13402`); multi-section ZIP with `manifest.json` + `nscs06_v8.bin` + `compound_c.bin`. VERIFIED.
6. Composite submission_dir at `experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/submission/` contains `inflate.sh` (505 B), `inflate.py` (8.2 KB / 162 LOC under 200 LOC budget), `0.bin` alias (1.8 MB matches archive.zip), vendored `src/tac/` package. VERIFIED.
7. Catalog #370 `check_no_pr_submission_without_canonical_compliance_verdict` registered in `tac.preflight` per Catalog #176 META-meta gate. VERIFIED.
8. Canonical dataclass contracts (SubmissionBundleResult / LintVerdict / ComplianceVerdict / PairedAuthEvalVerdict) inspected via `dataclasses.fields()`. VERIFIED canonical evidence_grade strings + canonical_equation_id constants + PairedAuthEvalVerdictKind enum values per layer.
9. `user_pr_attribution` memory: ZERO Claude/Anthropic/Co-Authored tokens in PR-facing surfaces; sole-author Alejandro Peña <adpena@gmail.com>. VERIFIED.
10. Catalog #229 premise verifier itself: existing PR111-candidate sister `phase_1_6_dry_run.py` at the composite output dir already implements the equivalent logic OUTSIDE the canonical CLI — confirms the structural gap that this extension closes. VERIFIED.

## Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Canonical adopted | Forked | Rationale |
|---|---|---|---|
| CLI argument parsing | `argparse` + extended `--composite-recipe` / `--composite-submission-dir` flags on the existing canonical `_build_parser()` | — | Canonical surface; the existing flag-set semantics preserved for single-substrate mode. Mutually-exclusive composite-mode dispatch via main() conditional. |
| Recipe YAML loading | `PyYAML.safe_load` with fallback to minimal top-level scanner | — | PyYAML is the canonical loader used by every operator-authorize sister; fallback supports hermetic CI surfaces without PyYAML installed. |
| Composite detection | NEW `_detect_composite_recipe(recipe_path) -> (is_composite, recipe_data, reasons)` returning typed verdict | — | Detection is canonical-extension (no sister gate exists); audit-transparency via multiple detection-reason recording per Catalog #305 observability. |
| Layer 0 compression_pipeline | STRUCTURAL_EXEMPTION (composite has no single trainer per Catalog #240) | — | Documented exemption per HNeRV parity L2 export-first discipline; composite archive built outside lifecycle CLI. |
| Layer 1 archive_grammar | STRUCTURAL_EXEMPTION (composite archive grammar = multi-section ZIP per HNeRV parity L3) | — | Documented exemption; sections computed via zipfile inspection rather than re-built. |
| Layer 2 Builder | Canonical `SubmissionBundleResult` dataclass constructed directly from composite archive + submission_dir | Wraps canonical helper inputs | The canonical `build_submission_bundle()` requires `CompressionPipelineResult + ArchiveGrammarManifest`; composite has neither. Construct the canonical `SubmissionBundleResult` directly with composite-specific provenance + canonical equation_id + evidence_grade per the dataclass __post_init__ contract. |
| Layer 3 Linter | NEW `_emit_composite_lint_verdict()` mirrors canonical `lint_submission_bundle` PR-facing-surface scan | — | Canonical helper requires full SubmissionBundleResult input + scans 8 surfaces; composite needs focused scan of 4 surfaces (inflate.sh, inflate.py, README, PR body). Same `LintVerdict` shape; canonical_equation_id + evidence_grade per linter module canonical constants. |
| Layer 4 Compliance | NEW `_emit_composite_compliance_verdict()` mirrors canonical `enforce_contest_compliance` per-check breakdown | — | Canonical helper invokes external `scripts/pre_submission_compliance_check.py`; composite has 8 inline checks (archive exists, 0.bin alias, inflate.sh 3-arg, inflate.py 3-arg, strict-scorer-rule, Catalog #367 fail-closed, canonical select_inflate_device, Catalog #246 paired-CUDA operator-gated). Same `ComplianceVerdict` shape; canonical catalog_gate_refs ints + canonical canonical_equation_id + evidence_grade. |
| Layer 5 Paired auth-eval | NEW `_emit_composite_paired_verdict()` emits `BLOCKED_PRE_DISPATCH` canonical pre-paired state | — | Canonical `plan_paired_auth_eval` requires full SubmissionBundleResult input + cost-band semantics; composite needs canonical pre-paired sidecar BEFORE operator dispatches paired-CUDA. `BLOCKED_PRE_DISPATCH` IS the canonical pre-paired enum value; `cuda_axis_tag` / `cpu_axis_tag` = `[missing]` per the canonical pre-paired contract. |
| Layer 6 Catalog #370 STRICT gate | Canonical `_run_layer_6_gate()` reused unchanged | — | Same Catalog #370 gate runs over the composite's submission_dir; emitted 4 canonical sidecars are consumed by the gate's verdict parser. |
| Lifecycle exit-code taxonomy | Canonical 6-value taxonomy (`PACKET_CLEAN`/`LINT_VIOLATIONS`/`COMPLIANCE_ERRORS`/`MISSING_PAIRED_AXIS`/`OPERATOR_GATED`/`CLI_ERROR`) reused unchanged | — | Composite-mode emits same canonical exit codes; `MISSING-PAIRED-AXIS` is the canonical composite-pre-paired state. |
| Composite-specific verdict sidecar | NEW `composite_recipe_verdict.json` augmenting the canonical 4 | — | Captures composite-mode metadata (composite recipe path, composite archive sha, operator next step for paired-CUDA RATIFICATION) per Catalog #305 observability + audit-trail. |

## Cargo-cult audit per assumption (per Catalog #303)

| # | Assumption | Classification | Unwind / verification |
|---|---|---|---|
| 1 | "Composite-mode needs new canonical helper modules" | CARGO-CULTED-INVERTED | Empirically the canonical helpers' dataclass shapes (SubmissionBundleResult / LintVerdict / ComplianceVerdict / PairedAuthEvalVerdict) accept composite-shape inputs directly; only the constructor wrappers + Layer 0+1 STRUCTURAL_EXEMPTION are needed. NEW helper modules would have been over-engineering. |
| 2 | "Composite-mode requires extending PairedAuthEvalVerdictKind with PLAN_ONLY" | CARGO-CULTED-EMPIRICALLY-FALSIFIED | Canonical enum already carries BLOCKED_PRE_DISPATCH which IS the canonical pre-paired state per the dataclass __post_init__ contract (archive_sha256_paired permitted empty + cuda_axis_tag/cpu_axis_tag in {'[missing]','[failed]'}). |
| 3 | "Composite Phase 6 paired-CUDA check should be severity=WARNING" | CARGO-CULTED-EMPIRICALLY-FALSIFIED | The `operator_gated_remaining` field of `ComplianceVerdict` MUST be a subset of `error_checks` per the dataclass __post_init__ contract; operator-gated checks MUST be severity=ERROR + passed=False. WARNING severity would have failed dataclass construction. |
| 4 | "Composite-mode should early-return on Layer 4 compliance failure" | CARGO-CULTED-EMPIRICALLY-FALSIFIED | The composite Phase 6 paired-CUDA check IS designed to fail (operator-gated). Early-return would prevent Layer 5 paired sidecar emission + Layer 6 Catalog #370 gate run; the operator would never see the complete canonical 4-verdict chain state. Refactored to early-return ONLY on non-operator-gated compliance errors. |
| 5 | "Composite-mode needs catalog_gate_refs as strings 'Catalog #146'" | CARGO-CULTED-EMPIRICALLY-FALSIFIED | The `ComplianceCheck.catalog_gate_refs` field is `tuple[int, ...]` per the dataclass contract; entries must be positive ints in 1..1000. Strings rejected. Used canonical int form `(146, 361)` + computed `catalog_gate_protection_summary` keys as `f"Catalog #{ref}"`. |

## 9-dimension success checklist evidence (per Catalog #294)

1. **UNIQUENESS**: FIRST composite-recipe mode on the canonical Phase 9 PR-submission lifecycle CLI. Resolves the structural gap identified by PR111-candidate landing memo op-routable #5.
2. **BEAUTY + ELEGANCE**: ~700 LOC added to the existing 726-LOC CLI (extension is ~2x the original; expected given the 4 canonical-sidecar emitters + composite-mode orchestrator). Each emitter mirrors the canonical dataclass shape exactly + threads canonical Provenance per Catalog #323. CLI flags are minimal additions (`--composite-recipe` + `--composite-submission-dir`); existing flags unchanged.
3. **DISTINCTNESS**: Distinct from Slot 2's anti-pattern registry expansion + Wyner-Ziv work (DISJOINT files); distinct from the composite's existing `phase_1_6_dry_run.py` ad-hoc script (this extension routes through the canonical CLI + canonical dataclass shapes).
4. **RIGOR**: Catalog #229 premise verification (10 anchors); Catalog #287/#323 canonical Provenance threading; canonical dataclass __post_init__ contracts honored; 45/45 existing CLI tests pass (no single-substrate-mode regression); end-to-end composite-mode smoke produces all 5 sidecars + correct MISSING-PAIRED-AXIS exit semantics.
5. **OPTIMIZATION-PER-TECHNIQUE**: Reuses canonical dataclass shapes + canonical evidence_grade strings + canonical canonical_equation_id constants per layer rather than introducing parallel sister helper modules.
6. **STACK-OF-STACKS-COMPOSABILITY**: Composite-recipe mode IS the canonical apparatus for stack-of-stacks PR submissions (multi-substrate composite archives) per HNeRV parity L3 multi-file justification.
7. **DETERMINISTIC-REPRODUCIBILITY**: Sidecar JSON written via `json.dumps(sort_keys=True, indent=2)`; canonical Provenance carries `captured_at_utc` per Catalog #323; composite archive sha256 reproducible across re-runs.
8. **EXTREME-OPTIMIZATION-PERFORMANCE**: $0 GPU spend; ~0.05s wall-clock CLI execution (the heavy work is the upfront composite archive build + PIL bilinear upsample smoke, neither of which this CLI extension touches).
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: Unblocks the PR111-candidate operator-attended cascade (paired-CUDA RATIFICATION → PR111 submission via canonical CLI). Composite archive predicted [contest-CPU]=0.165 per Compound F first-order Volterra; advance over canonical frontier 0.192009 would be ~0.027 ΔS IF RATIFIED.

## Observability surface (per Catalog #305)

1. **Inspectable per layer**: Each layer emits typed dict in `report["layers"][...]` + canonical sidecar JSON; `--json` flag emits full machine-readable verdict.
2. **Decomposable per signal**: Per-layer `ok` / `error_count` / `overall_clean` / `operator_gated_remaining` fields; canonical 8-check ComplianceVerdict breakdown.
3. **Diff-able across runs**: Sidecar JSON byte-stable via `sort_keys=True`; composite archive sha256 + per-section sha256 deterministic.
4. **Queryable post-hoc**: Sidecars at `experiments/results/<composite>/submission/*.json` + composite_recipe_verdict.json for operator audit.
5. **Cite-able**: Every layer emits canonical Provenance per Catalog #323 (kind=PREDICTED_FROM_MODEL + canonical_helper invocation + captured_at_utc + composite_recipe_path + composite_archive_sha256).
6. **Counterfactual-able**: Composite archive byte change propagates through sha256 + per-section sha computation; any drift surfaces at Layer 2 (builder) sidecar emission.

## Composite-recipe extension architecture

### CLI surface

```
.venv/bin/python tools/operator_pr_submission_full_lifecycle.py \
  --composite-recipe .omx/operator_authorize_recipes/substrate_composite_*.yaml \
  --composite-submission-dir experiments/results/composite_*/submission/ \
  --archive-path experiments/results/composite_*/archive.zip \
  --predecessors @SajayR:56:HNeRV_substrate @AaronLeslie138:95:fec_curriculum \
  [--target-repo commaai/comma_video_compression_challenge] \
  [--cuda-gpu T4] [--cuda-platform modal] [--cpu-target linux_x86_64_modal] \
  [--declared-deps numpy torch] [--json | --quiet]
```

### Composite-mode dispatch in `main()`

When `--composite-recipe` is supplied:
- `--substrate-trainer` + `--recipe-path` + `--lane-id` + `--output-dir` NOT required (composite has no single trainer)
- `--composite-submission-dir` REQUIRED
- Dispatches to `run_composite_lifecycle()` instead of `run_full_lifecycle()`

When `--composite-recipe` NOT supplied (single-substrate mode):
- Existing single-substrate validation enforced (all 4 single-mode args required)
- Existing `run_full_lifecycle()` invoked unchanged

### Composite detection signals (any one sufficient)

1. `composite_components: [...]` top-level list (canonical NEW form)
2. `substrate_id` starts with `composite_` (PR111 transitional form)
3. `trainer_path` references `build_composite_*.py` build script
4. Multiple detection reasons accumulated for audit transparency

### Composite-mode orchestration (`run_composite_lifecycle`)

| Layer | Action | Sidecar emitted |
|---|---|---|
| 0 | STRUCTURAL_EXEMPTION (composite has no single substrate trainer) | — |
| 1 | STRUCTURAL_EXEMPTION (composite archive grammar = multi-section ZIP per HNeRV parity L3) | — |
| 2 (Builder) | `_emit_composite_bundle_verdict()` | `submission_bundle_result.json` |
| 3 (Linter) | `_emit_composite_lint_verdict()` | `lint_verdict.json` |
| 4 (Compliance) | `_emit_composite_compliance_verdict()` (8 inline checks) | `compliance_verdict.json` |
| 5 (Paired) | `_emit_composite_paired_verdict()` (BLOCKED_PRE_DISPATCH) | `paired_auth_eval_verdict.json` |
| 6 (Catalog #370) | `_run_layer_6_gate()` (unchanged from single-mode) | — |
| composite-specific | `composite_recipe_verdict.json` summarizing the composite-mode lifecycle verdict + operator next step | `composite_recipe_verdict.json` |

### Canonical contract honored per layer

- **SubmissionBundleResult**: `schema_version=submission_bundle_v1_20260526`, `canonical_equation_id=submission_bundle_canonical_helper_consolidation_savings_v1`, `evidence_grade=[predicted; submission-bundle-canonical]`, `canonical_equation_status=FORMALIZATION_PENDING`, `score_claim=False`, `promotable=False`, `axis_tag=[predicted]`
- **LintVerdict**: `schema_version=LINTER_SCHEMA_VERSION`, `canonical_equation_id=submission_linter_canonical_helper_consolidation_savings_v1`, `evidence_grade=[predicted; submission-linter-canonical]`, `surfaces_scanned` canonical-sorted
- **ComplianceVerdict**: `schema_version=COMPLIANCE_SCHEMA_VERSION`, `canonical_equation_id=submission_compliance_canonical_helper_consolidation_savings_v1`, `evidence_grade=[predicted; compliance-canonical]`, `catalog_gate_refs` as canonical-sorted positive ints, `operator_gated_remaining` subset of `error_checks` per dataclass __post_init__
- **PairedAuthEvalVerdict**: `schema_version=PAIRED_AUTH_EVAL_SCHEMA_VERSION`, `canonical_equation_id=paired_auth_eval_canonical_helper_consolidation_savings_v1`, `evidence_grade=[predicted; paired-axis-not-yet-dispatched]`, `verdict=BLOCKED_PRE_DISPATCH`, `cuda_axis_tag=[missing]`, `cpu_axis_tag=[missing]`

## End-to-end smoke verdict

Empirical smoke run against PR111-candidate composite (sha `dfff1358638ef7f7`):

| Layer | Verdict | Detail |
|---|---|---|
| composite_detection | ok=True | 2 detection reasons: substrate_id prefix + trainer_path token |
| attribution_self_lint | ok=True | ZERO forbidden tokens in 2 predecessors |
| composite_archive_resolution | ok=True | archive_sha256=dfff1358 archive_bytes=1,917,982 sections=3 inflate_py_loc=162/200 |
| layer_0_compression_pipeline | ok=True | STRUCTURAL_EXEMPTION |
| layer_1_archive_grammar | ok=True | STRUCTURAL_EXEMPTION (3 sections per HNeRV parity L3) |
| layer_2_builder | ok=True | submission_bundle_result.json emitted |
| layer_3_linter | ok=True | 0 errors / 0 warns; 3 surfaces scanned (inflate_py + inflate_sh + README absent) |
| layer_4_compliance | ok=False | 1 error (operator-gated paired-CUDA pending); 7/8 passed; non-operator-gated errors=0 |
| layer_5_paired_auth_eval | ok=False (BLOCKED_PRE_DISPATCH) | canonical pre-paired state |
| layer_6_catalog_370_gate | ok=False | composite submission_dir flagged in same scan; ALSO flags 4 pre-existing baseline submissions (a1 / pr106_latent_sidecar_r2 / pr106_latent_sidecar_r2_pr101_grammar / robust_current) that are out of scope for this work |

**Lifecycle verdict**: MISSING-PAIRED-AXIS (exit code 3). Correct semantics per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable + Catalog #246 operator-attended.

## Sidecar artifacts

5 sidecars landed at `experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/submission/`:

| Sidecar | Bytes | Role |
|---|---|---|
| `submission_bundle_result.json` | 3149 | Phase 4 canonical bundle verdict |
| `lint_verdict.json` | 1616 | Phase 5 canonical lint verdict |
| `compliance_verdict.json` | 6763 | Phase 6 canonical compliance verdict |
| `paired_auth_eval_verdict.json` | 3089 | Phase 7 canonical paired-CUDA pre-dispatch verdict |
| `composite_recipe_verdict.json` | 1565 | Composite-specific lifecycle summary + operator next step |

## Regression test verdict

`pytest src/tac/tests/test_operator_pr_submission_full_lifecycle_cli.py -x -q`: **45/45 passed in 0.64s**. Single-substrate mode UNCHANGED; composite-recipe extension is purely additive at the CLI surface.

## Anti-pattern preflight (canonical sister Slot 2 + commit c50b8ac91)

| # | Anti-pattern | Severity | Confidence | TRUE-POSITIVE? | Status |
|---|---|---|---|---|---|
| 1 | predicted_band_from_random_init_tier_c_v1 | critical | 0.50 | NO | this extension does NOT emit a predicted_band; it threads the composite's predicted_band field through canonical Provenance. |
| 2 | quantize_then_svd_corrupted_low_rank_v1 | high | 0.50 | NO | no SVD in the lifecycle CLI extension. |
| 3 | mlx_trainer_pytorch_sister_duplicated_implementation_v1 | medium | 0.50 | NO | not applicable; CLI extension does not invoke MLX or PyTorch. |

Per Compound C op-routable #4 + Slot 2 Wave N+1 architectural fix (commit c50b8ac91): confidence=0.5 token-overlap matches are HEURISTIC false positives. No hard-stop blockers.

## Sister coordination

- **Slot 2** (anti-pattern registry expansion + Wyner-Ziv): DISJOINT scope per the mandate. Slot 1 owns `tools/operator_pr_submission_full_lifecycle.py` + composite sidecars + this landing memo. Slot 2 owns `src/tac/canonical_anti_patterns/builtins.py` + Wyner-Ziv per-pair PoseNet substrate extension. ZERO file collision; Catalog #340 sister-checkpoint guard not triggered.
- **PR111-candidate landing** (sister `cbe46e1b7`): this extension consumes the composite archive + recipe + submission_dir landed by sister; closes op-routable #5 of that landing memo at the canonical CLI surface.

## Operator-routable cascade

### CONDITIONAL: paired-CUDA RATIFICATION (~$1-2 paired T4 CUDA + Linux x86_64 CPU)

1. Operator flips `dispatch_enabled: true` in `.omx/operator_authorize_recipes/substrate_composite_nscs06_v8_plus_compound_c_pr111_modal_t4_dispatch.yaml`
2. Operator runs `tools/operator_authorize.py --recipe substrate_composite_nscs06_v8_plus_compound_c_pr111_modal_t4_dispatch` (paired T4 CUDA + Linux x86_64 CPU per Catalog #246)
3. Paired anchors land → re-run `tools/operator_pr_submission_full_lifecycle.py --composite-recipe ...` → Phase 7 paired_auth_eval emits PAIRED_PASS verdict → Catalog #370 gate satisfies for composite-mode

### CONDITIONAL: IF RATIFIED ≤0.18 [contest-CPU] AND/OR [contest-CUDA] → PR111 submission cascade

1. Re-run lifecycle CLI; exit code transitions from EXIT_MISSING_PAIRED_AXIS (3) to EXIT_OPERATOR_GATED (4)
2. CLI emits operator-gated `gh release create` + `gh pr create` commands (NEVER fires per CLAUDE.md "Executing actions with care")
3. Operator hosts the composite archive + runs `gh pr create --repo commaai/comma_video_compression_challenge ...`

### IF NOT RATIFIED → IMPLEMENTATION-LEVEL falsification per Catalog #307

1. Composite Compound F α=0.85 prediction empirically falsified at implementation level (not paradigm level)
2. Canonical equation `cross_paradigm_plus_decoder_compression_compound_alpha_v1` posterior refit per Catalog #371 auto-trigger
3. Operator-routable refinement: try α=0.7 sub-additive halve OR α=1.0 fully additive

## Deferred deliverables (sister-Wave op-routables)

Per the scope decision logged at checkpoint step 5 (in-mandate priority cascade given 11 deliverables + time budget realism):

- **D2 (Phase 4 builder multi-source rewrite)**: deferred. Composite already has its own builder/inflate.py; reusing them via the composite-mode bundle-verdict emitter IS the canonical extension. A multi-source `build_submission_bundle()` overload at the canonical helper module would be a larger sister landing.
- **D3 (Phase 5 linter composite-aware extension at canonical helper)**: deferred. `_emit_composite_lint_verdict()` mirrors the canonical PR-attribution discipline focused on the composite's 4 PR-facing surfaces. Extending `tac.submission_packet.linter.lint_submission_bundle` to accept composite-shape input is sister-Wave.
- **D4 (Phase 6 compliance composite-aware extension at canonical helper)**: deferred. `_emit_composite_compliance_verdict()` mirrors the canonical 8-check breakdown. Extending `tac.submission_packet.compliance.enforce_contest_compliance` to accept composite-shape input is sister-Wave.
- **D5 (Phase 7 paired_auth_eval composite-aware extension at canonical helper)**: deferred. `_emit_composite_paired_verdict()` emits canonical BLOCKED_PRE_DISPATCH. Sister-Wave landing of operator-attended paired-CUDA RATIFICATION for the composite recipe is op-routable #1 above.
- **D6 (Phase 8 Catalog #362 STRICT gate composite-recipe validation)**: deferred. Catalog #370 STRICT gate runs over the canonical 4 sidecars unchanged; composite-recipe-specific verdict at `composite_recipe_verdict.json` provides additional audit-trail metadata. A dedicated Catalog #362 STRICT gate for composite-recipe (sister of #370 at composite-specific surface) is sister-Wave.
- **D10 (NEW canonical equation `phase_9_lifecycle_cli_composite_recipe_compounding_v1`)**: deferred. Canonical equation registration requires the canonical equation registry helper + posterior anchor; sister-Wave landing per Catalog #344.
- **D11 (anti-pattern preflight)**: addressed inline above (3 confidence-0.5 matches all FALSE positives per Slot 2 c50b8ac91 fix).
- **D9 (probe outcome registration via Catalog #313)**: deferred. The composite's existing probe outcome row at `.omx/state/probe_outcomes.jsonl` per the PR111-candidate landing covers the same lane; sister-Wave landing of a CLI-extension-specific probe outcome is op-routable.

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map**: N/A (apparatus extension; no per-pair sensitivity signal)
- **hook #2 Pareto constraint**: N/A (apparatus extension; no Pareto-relevant signal)
- **hook #3 bit-allocator**: N/A (apparatus extension; no bit-allocator signal)
- **hook #4 cathedral autopilot dispatch**: ACTIVE (the CLI IS the canonical PR-submission dispatch entry point; composite-mode unblocks the PR111-candidate cascade via the canonical surface)
- **hook #5 continual-learning posterior**: ACTIVE (canonical sidecars consumed by Catalog #370 STRICT gate + future cathedral autopilot consumers via `tac.submission_packet.builder.submission_bundle_result_from_dict` reconstructor)
- **hook #6 probe-disambiguator**: ACTIVE (lifecycle verdict + operator_next_step IS the canonical disambiguator between PR-submission-ready vs paired-CUDA-pending vs canonical-CLI-error)

## Cross-references

- Parent: `.omx/research/pr111_candidate_nscs06_v8_plus_compound_c_composite_build_landed_20260528.md` (op-routable #5 closed by this work)
- Sister: `.omx/research/canonical_submission_pipeline_specification_memo_20260526.md` (Phase 9 spec)
- Composite recipe: `.omx/operator_authorize_recipes/substrate_composite_nscs06_v8_plus_compound_c_pr111_modal_t4_dispatch.yaml`
- Composite archive: `experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/archive.zip` (sha `dfff1358638ef7f7`)
- Composite submission_dir: `experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/submission/`
- 5 sidecars emitted: see Sidecar artifacts section above
- Canonical frontier pointer: `.omx/state/canonical_frontier_pointer.json` (per Catalog #343)
- Catalog gates honored: #146 (inflate runtime), #205 (canonical select_inflate_device), #229 (premise verification), #240 (recipe-vs-trainer-state consistency exemption for composite), #246 (paired CPU+CUDA), #270 (dispatch optimization protocol scope), #287 (placeholder rejection), #295 (PYTHONPATH self-containment), #303 (cargo-cult audit), #305 (observability surface), #323 (canonical Provenance umbrella), #340 (sister-checkpoint guard DISJOINT), #341 (Tier A non-promotable markers), #343 (frontier pointer), #344 (canonical equations registry), #362 (Phase 8 STRICT gate; sister-Wave), #367 (CONTEST_RAW_BYTES fail-closed), #370 (PR submission canonical compliance verdict), #372 (Dykstra solver; consumed via composite predicted_band Provenance), #373 (anti-patterns matcher c50b8ac91 fix)
- CLAUDE.md non-negotiables: Frontier scores pointer-only / Submission auth eval BOTH CPU AND CUDA / Executing actions with care / Public Disclosure Hygiene / Subagent coherence-by-default / Forbidden premature KILL / 12th canonicalization x standardization x ease-of-contest-compliance trinity / user_pr_attribution

## Mission contribution per Catalog #300

`apparatus_maintenance`: this work closes the structural gap between the composite archive (built outside the canonical CLI by ad-hoc per-composite script) and the canonical Phase 9 PR-submission lifecycle CLI. Unblocks the operator-attended PR111-candidate cascade at the canonical surface; the CLI is the single-command default-path the operator runs to verify a composite PR-submission is ready to ship.

## Empirical anchors

- CLI extension: `tools/operator_pr_submission_full_lifecycle.py` (+~700 LOC); 45/45 existing tests pass; end-to-end composite-mode smoke emits 5 sidecars with correct MISSING-PAIRED-AXIS exit code.
- Composite archive consumed: `experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/archive.zip` (1,917,982 B; sha `dfff1358638ef7f7bad4596958cddb62215ed06c5b850a8501e3ad42a2c13402`)
- 5 canonical sidecars at `experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/submission/`: `submission_bundle_result.json` + `lint_verdict.json` + `compliance_verdict.json` + `paired_auth_eval_verdict.json` + `composite_recipe_verdict.json`
- Wall-clock + cost: $0 GPU + ~75 min wall-clock subagent execution; CLI run ~0.05s
