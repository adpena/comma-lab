# SPDX-License-Identifier: MIT
"""Canonical-automated submission packet pipeline — public API.

Per operator NON-NEGOTIABLE 2026-05-26 verbatim:
*"Remember everything we had to do to clean up and properly bundle our
submission, let's make that canonical and automated moving forward"*
(9th standing directive) and the amendment
*"Remember contest compliance and bundling full compression script and
all and everything"* (FULL lifecycle: compression + compliance +
everything).

Phase 2 lands ``compression_pipeline`` (Layer 0 per Phase 1 audit
specification memo at
``.omx/research/canonical_submission_pipeline_specification_memo_20260526.md``).
Phase 3 lands ``archive_grammar`` (Layer 1 per Phase 1 audit spec memo).

Per the 11th ORDER-MATTERS standing directive: Layer 0 is the canonical
FIRST encoder pipeline orchestrator + Layer 1 is the canonical SECOND
archive grammar builder. Every future submission phase (Phase 4 builder
/ Phase 5 compliance / Phase 6 paired_auth_eval / Phase 7 operator
runbook CLI / Phase 8 Catalog #362 STRICT preflight gate / Phase 9
cathedral consumer / Phase 10 PR111-candidate end-to-end regression)
consumes :class:`CompressionPipelineResult` + :class:`ArchiveGrammarManifest`
directly.

Per the 12th canonicalization × standardization × ease-of-contest-
compliance trinity: this package's API is canonical-frozen-dataclass-
return + canonical-Provenance-routing + 4-layer canonical-helper-pattern
sister of :mod:`tac.deploy.modal.call_id_ledger` (Catalog #245),
:mod:`tac.probe_outcomes_ledger` (Catalog #313), and
:mod:`tac.canonical_equations` (Catalog #344).

Per the 8th MLX-first numpy-portable standing directive: training routes
through MLX-first encoder on Apple Silicon; weights export to portable
``.npz`` for numpy-only inflate per HNeRV parity L4 (≤200 LOC, ≤2 deps).
Per HNeRV parity L3: archive grammar is monolithic single-file ``0.bin``
(or explicitly justified multi-file).

Quick start::

    from tac.submission_packet import (
        ArchiveGrammarManifest,
        ArchiveSectionSpec,
        CompressionPipelineResult,
        build_archive_grammar_from_compression_pipeline_result,
        build_compression_pipeline,
    )

    pipeline_result = build_compression_pipeline(
        lane_id="lane_pr111_candidate_20260601",
        video_path=Path("upstream/videos/0.mkv"),
        substrate_trainer=Path("experiments/train_substrate_nscs06_v8_chroma_lut.py"),
        recipe_path=Path(".omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_local_apple_silicon_dispatch.yaml"),
        hardware_substrate="auto",
        qat_enabled=True,
        output_dir=Path("experiments/results/pr111_candidate"),
    )
    grammar = build_archive_grammar_from_compression_pipeline_result(
        compression_pipeline_result=pipeline_result,
        archive_path=Path("experiments/results/pr111_candidate/archive.zip"),
        monolithic_single_file=True,
    )
    print(grammar.archive_sha256[:12], grammar.section_specs[0].section_name)

Discipline cross-references:
  * Catalog #245 / #313 / #344 / #354 / #355 canonical 4-layer pattern
  * Catalog #270 dispatch optimization protocol (Tier1 + Tier2 + Tier3 umbrella)
  * Catalog #146 contest-compliant inflate runtime template
  * Catalog #361 Modal artifact filter preserves output/submission/
  * Catalog #205 canonical select_inflate_device routing
  * Catalog #295 PYTHONPATH self-containment
  * Catalog #339 + #360 silent-no-spawn extinction
  * Catalog #190 hardware_substrate detection (NO false precision)
  * Catalog #226 canonical scorer-loss helper routing
  * Catalog #228 GTScorerCache F3 consumption
  * Catalog #323 canonical Provenance umbrella
  * Catalog #340 sister-checkpoint guard
  * Catalog #356 per-axis decomposition (if Tier B available)
  * Catalog #365 + #366 + #367 just-landed STRICT gates respected
  * Catalog #139 + #105 + #266 + #272 archive bytes consumed by inflate
  * Catalog #220 substrate L1+ scaffold operational mechanism declaration
  * CLAUDE.md "Beauty, simplicity, and developer experience"
  * CLAUDE.md "Subagent coherence-by-default"
  * CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"
"""
from __future__ import annotations

from tac.submission_packet.archive_grammar import (
    ARCHIVE_GRAMMAR_SCHEMA_VERSION,
    BYTE_MUTATION_SMOKE_MIN_BYTES,
    CANONICAL_ARCHIVE_NAME,
    CANONICAL_MONOLITHIC_MEMBER_NAME,
    PHASE_3_LAYER_VERSION,
    ArchiveGrammarError,
    ArchiveGrammarManifest,
    ArchiveSectionSpec,
    ByteMutationSmokeVerdict,
    OperationalMechanismStatus,
    SectionKind,
    build_archive_grammar_from_compression_pipeline_result,
    derive_archive_grammar_provenance,
    discover_section_specs_from_archive,
    emit_parser_section_manifest_sidecar,
    verify_byte_mutation_smoke_via_canonical_helper,
)
from tac.submission_packet.archive_grammar import (
    CANONICAL_EQUATION_ID as ARCHIVE_GRAMMAR_CANONICAL_EQUATION_ID,
)
from tac.submission_packet.builder import (
    CANONICAL_EQUATION_ID as SUBMISSION_BUNDLE_CANONICAL_EQUATION_ID,
)
from tac.submission_packet.builder import (
    DEFAULT_INFLATE_DEPS_BUDGET,
    DEFAULT_INFLATE_PY_LOC_BUDGET,
    HNERV_CLASS_INFLATE_DEPS,
    NUMPY_PORTABLE_INFLATE_DEPS,
    PHASE_4_LAYER_VERSION,
    SUBMISSION_BUNDLE_SCHEMA_VERSION,
    BundleComponentKind,
    DependencyClosureManifest,
    PythonpathSelfContainmentStatus,
    SelectInflateDeviceRouting,
    SubmissionBundleError,
    SubmissionBundleResult,
    build_dependency_closure_manifest,
    build_submission_bundle,
    derive_submission_bundle_provenance,
    submission_bundle_result_from_dict,
)
from tac.submission_packet.compliance import (
    CANONICAL_COMPLIANCE_SCRIPT_PATH,
    COMPLIANCE_SCHEMA_VERSION,
    PHASE_6_LAYER_VERSION,
    CheckSeverity,
    ComplianceCheck,
    ComplianceVerdict,
    SubmissionComplianceError,
    derive_compliance_provenance,
    enforce_contest_compliance,
)
from tac.submission_packet.compliance import (
    CANONICAL_EQUATION_ID as COMPLIANCE_CANONICAL_EQUATION_ID,
)
from tac.submission_packet.compression_pipeline import (
    CANONICAL_EQUATION_ID,
    COMPRESSION_PIPELINE_SCHEMA_VERSION,
    PHASE_2_LAYER_VERSION,
    CompressionPipelineError,
    CompressionPipelineResult,
    HardwareSubstrateClass,
    PerAxisPredictedBand,
    build_compression_pipeline,
    classify_hardware_substrate_for_dispatch,
    derive_compression_pipeline_provenance,
    validate_recipe_trainer_pair,
    verify_compression_pipeline_protocol_complete,
)
from tac.submission_packet.linter import (
    CANONICAL_AXIS_TAGS,
    EMDASH_CHARACTER,
    FIRST_PERSON_PLURAL_PATTERNS,
    FORBIDDEN_PUBLIC_PR_TOKENS,
    LINTER_SCHEMA_VERSION,
    PHASE_5_LAYER_VERSION,
    TONE_VIOLATION_PATTERNS,
    LintFinding,
    LintSeverity,
    LintSurface,
    LintVerdict,
    SubmissionLinterError,
    derive_linter_provenance,
    lint_archive_zip,
    lint_compliance_placeholder,
    lint_inflate_py,
    lint_pr_body,
    lint_readme,
    lint_submission_bundle,
    lint_tone,
)
from tac.submission_packet.linter import (
    CANONICAL_EQUATION_ID as LINTER_CANONICAL_EQUATION_ID,
)
from tac.submission_packet.linter import (
    EVIDENCE_GRADE as LINTER_EVIDENCE_GRADE,
)
from tac.submission_packet.paired_auth_eval import (
    CANONICAL_EQUATION_ID as PAIRED_AUTH_EVAL_CANONICAL_EQUATION_ID,
)
from tac.submission_packet.paired_auth_eval import (
    PAIRED_AUTH_EVAL_SCHEMA_VERSION,
    PHASE_7_LAYER_VERSION,
    PairedAuthEvalError,
    PairedAuthEvalVerdict,
    PairedAuthEvalVerdictKind,
    derive_paired_auth_eval_provenance,
    plan_paired_auth_eval,
    reconstruct_verdict_from_disk,
)

__all__ = [  # noqa: RUF022 - grouped by pipeline phase, not alphabetically.
    # Phase 2 (Layer 0)
    "CANONICAL_EQUATION_ID",
    "COMPRESSION_PIPELINE_SCHEMA_VERSION",
    "HardwareSubstrateClass",
    "PHASE_2_LAYER_VERSION",
    "CompressionPipelineError",
    "CompressionPipelineResult",
    "PerAxisPredictedBand",
    "build_compression_pipeline",
    "classify_hardware_substrate_for_dispatch",
    "derive_compression_pipeline_provenance",
    "validate_recipe_trainer_pair",
    "verify_compression_pipeline_protocol_complete",
    # Phase 3 (Layer 1)
    "ARCHIVE_GRAMMAR_CANONICAL_EQUATION_ID",
    "ARCHIVE_GRAMMAR_SCHEMA_VERSION",
    "BYTE_MUTATION_SMOKE_MIN_BYTES",
    "CANONICAL_ARCHIVE_NAME",
    "CANONICAL_MONOLITHIC_MEMBER_NAME",
    "PHASE_3_LAYER_VERSION",
    "ArchiveGrammarError",
    "ArchiveGrammarManifest",
    "ArchiveSectionSpec",
    "ByteMutationSmokeVerdict",
    "OperationalMechanismStatus",
    "SectionKind",
    "build_archive_grammar_from_compression_pipeline_result",
    "derive_archive_grammar_provenance",
    "discover_section_specs_from_archive",
    "emit_parser_section_manifest_sidecar",
    "verify_byte_mutation_smoke_via_canonical_helper",
    # Phase 4 (Layer 2)
    "SUBMISSION_BUNDLE_CANONICAL_EQUATION_ID",
    "SUBMISSION_BUNDLE_SCHEMA_VERSION",
    "PHASE_4_LAYER_VERSION",
    "DEFAULT_INFLATE_DEPS_BUDGET",
    "DEFAULT_INFLATE_PY_LOC_BUDGET",
    "HNERV_CLASS_INFLATE_DEPS",
    "NUMPY_PORTABLE_INFLATE_DEPS",
    "BundleComponentKind",
    "DependencyClosureManifest",
    "PythonpathSelfContainmentStatus",
    "SelectInflateDeviceRouting",
    "SubmissionBundleError",
    "SubmissionBundleResult",
    "build_dependency_closure_manifest",
    "build_submission_bundle",
    "derive_submission_bundle_provenance",
    "submission_bundle_result_from_dict",
    # Phase 6 (Layer 4) - compliance enforcer (this lane)
    "COMPLIANCE_CANONICAL_EQUATION_ID",
    "COMPLIANCE_SCHEMA_VERSION",
    "PHASE_6_LAYER_VERSION",
    "CANONICAL_COMPLIANCE_SCRIPT_PATH",
    "CheckSeverity",
    "ComplianceCheck",
    "ComplianceVerdict",
    "SubmissionComplianceError",
    "derive_compliance_provenance",
    "enforce_contest_compliance",
    # Phase 5 (Layer 3)
    "LINTER_CANONICAL_EQUATION_ID",
    "LINTER_EVIDENCE_GRADE",
    "LINTER_SCHEMA_VERSION",
    "PHASE_5_LAYER_VERSION",
    "FORBIDDEN_PUBLIC_PR_TOKENS",
    "FIRST_PERSON_PLURAL_PATTERNS",
    "EMDASH_CHARACTER",
    "TONE_VIOLATION_PATTERNS",
    "CANONICAL_AXIS_TAGS",
    "LintFinding",
    "LintSeverity",
    "LintSurface",
    "LintVerdict",
    "SubmissionLinterError",
    "derive_linter_provenance",
    "lint_archive_zip",
    "lint_compliance_placeholder",
    "lint_inflate_py",
    "lint_pr_body",
    "lint_readme",
    "lint_submission_bundle",
    "lint_tone",
    # Phase 7 (Layer 5) - paired auth-eval orchestrator (this lane)
    "PAIRED_AUTH_EVAL_CANONICAL_EQUATION_ID",
    "PAIRED_AUTH_EVAL_SCHEMA_VERSION",
    "PHASE_7_LAYER_VERSION",
    "PairedAuthEvalError",
    "PairedAuthEvalVerdict",
    "PairedAuthEvalVerdictKind",
    "derive_paired_auth_eval_provenance",
    "plan_paired_auth_eval",
    "reconstruct_verdict_from_disk",
]
