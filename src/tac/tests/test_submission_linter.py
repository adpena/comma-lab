# SPDX-License-Identifier: MIT
"""Tests for ``tac.submission_packet.linter`` (Phase 5 Layer 3 canonical helper).

Per Phase 1 audit specification memo at
``.omx/research/canonical_submission_pipeline_specification_memo_20260526.md``
Phase 5 acceptance criteria. Covers per-surface positive (catches violation)
+ negative (allows clean) + waiver semantics, Catalog #119 Claude attribution
FORBIDDEN-on-PR111-candidate enforcement, first-person plural grep, emdash
audit, PR 95 medal-class tone, cathedral consumer contract compliance,
CLI subprocess exit codes, live-repo regression guard, integration with
Phase 4 SubmissionBundleResult.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tac.submission_packet import (
    CANONICAL_AXIS_TAGS,
    EMDASH_CHARACTER,
    FIRST_PERSON_PLURAL_PATTERNS,
    FORBIDDEN_PUBLIC_PR_TOKENS,
    LINTER_SCHEMA_VERSION,
    PHASE_5_LAYER_VERSION,
    LintFinding,
    LintSeverity,
    LintSurface,
    LintVerdict,
    SubmissionLinterError,
    TONE_VIOLATION_PATTERNS,
    lint_archive_zip,
    lint_compliance_placeholder,
    lint_inflate_py,
    lint_pr_body,
    lint_readme,
    lint_submission_bundle,
    lint_tone,
)
from tac.submission_packet.builder import (
    DEFAULT_INFLATE_DEPS_BUDGET,
    DEFAULT_INFLATE_PY_LOC_BUDGET,
    SUBMISSION_BUNDLE_SCHEMA_VERSION,
    DependencyClosureManifest,
    SelectInflateDeviceRouting,
    SubmissionBundleResult,
    build_submission_bundle,
)
from tac.submission_packet.linter import (
    CANONICAL_EQUATION_ID,
    EVIDENCE_GRADE,
    PREDICTED_AXIS_TAG,
    _line_at,
    _truncate,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def canonical_clean_pr_body() -> str:
    """A canonical clean PR body honoring PR 95 medal-class precedent.

    Per the operator-only voice directive: first-person singular ("I built…"),
    @-mention attribution chain, PR# references, axis tag disclosure, no
    Claude/Anthropic/Co-Authored, no first-person plural, no emdash, no
    tone flourishes, no emoji.
    """
    return """# Submission

This submission stacks on @SajayR's PR #101 (HNeRV substrate). Score
0.19205 [contest-CPU] vs PR #102 0.19538 = -0.00333. I built FEC6 fixed
huffman k16 on top of the canonical PR101 grammar, preserving the archive
layout per CLAUDE.md "Bit-level deconstruction and entropy discipline".

## Attribution

Predecessors: @SajayR (#101), @EthanYangTW (#102), @rem2 (#103),
@BradyMeighan (#100), @AaronLeslie138 (#95).

## Limitations

Operator notes for paired auth eval pending.
"""


@pytest.fixture
def fresh_submission_bundle(tmp_path: Path) -> SubmissionBundleResult:
    """Build a fresh submission bundle into tmp_path for lint integration.

    Mirrors the canonical sister fixture in
    :mod:`src.tac.tests.test_submission_bundle` so the linter regression
    surface stays in lock-step with the Phase 4 builder regression surface.
    """
    import datetime
    import hashlib

    from tac.submission_packet import (
        ArchiveGrammarManifest,
        ArchiveSectionSpec,
        ByteMutationSmokeVerdict,
        CompressionPipelineResult,
        COMPRESSION_PIPELINE_SCHEMA_VERSION,
        OperationalMechanismStatus,
        SectionKind,
    )

    now = datetime.datetime.now(datetime.UTC).isoformat()

    # Build a minimal canonical archive.zip per Phase 3 sister convention.
    archive_path = tmp_path / "archive_src.zip"
    payload = b"hello-world" * 100  # 1100 bytes
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("0.bin", payload)
    archive_bytes = archive_path.read_bytes()
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    section_sha = hashlib.sha256(payload).hexdigest()

    cp_result = CompressionPipelineResult(
        schema_version=COMPRESSION_PIPELINE_SCHEMA_VERSION,
        lane_id="lane_test_linter_20260526",
        substrate_id="test_linter_substrate",
        video_path="upstream/videos/0.mkv",
        hardware_substrate="macos_arm64_m5_max",
        hardware_substrate_class="local-mps",
        substrate_trainer_path="experiments/train_substrate_test.py",
        recipe_path=".omx/operator_authorize_recipes/substrate_test.yaml",
        mlx_first_encode=True,
        qat_enabled=True,
        weights_export_path=None,
        weights_sha256=None,
        weights_size_bytes=None,
        training_anchor_call_id=None,
        qat_anchor_call_id=None,
        dispatch_optimization_protocol_overall_pass=True,
        dispatch_optimization_protocol_blockers=(),
        per_axis_predicted_band=None,
        measurement_utc=now,
        axis_tag="[predicted]",
        score_claim=False,
        promotable=False,
        evidence_grade="[predicted; compression-pipeline-canonical]",
        canonical_helper_invocation="tac.submission_packet.build_compression_pipeline",
        canonical_equation_id=(
            "compression_pipeline_canonical_helper_consolidation_savings_v1"
        ),
        canonical_equation_status="FORMALIZATION_PENDING",
        elapsed_seconds=0.01,
        cost_usd=None,
        canonical_provenance={"axis_tag": "[predicted]"},
        written_at_utc=now,
        written_pid=1,
        written_host="test",
    )

    spec = ArchiveSectionSpec(
        section_name="0.bin",
        offset_in_archive=0,
        length_in_archive=len(payload),
        sha256_of_section=section_sha,
        section_kind=SectionKind.OTHER.value,
        operational_mechanism_status=OperationalMechanismStatus.OPERATIONAL.value,
        distinguishing_feature_name=None,
        member_name="0.bin",
    )
    grammar = ArchiveGrammarManifest(
        schema_version="archive_grammar_v1_20260526",
        lane_id="lane_test_linter_20260526",
        substrate_id="test_linter_substrate",
        archive_path=str(archive_path),
        archive_sha256=archive_sha,
        archive_bytes=len(archive_bytes),
        section_specs=(spec,),
        monolithic_single_file=True,
        multi_file_justification=None,
        byte_mutation_smoke_verdict=ByteMutationSmokeVerdict.NOT_RUN.value,
        byte_mutation_smoke_evidence_path=None,
        no_op_detector_passed=False,
        measurement_utc=now,
        axis_tag="[predicted]",
        score_claim=False,
        promotable=False,
        evidence_grade="[predicted; archive-grammar-canonical]",
        canonical_helper_invocation=(
            "tac.submission_packet.build_archive_grammar_from_compression_pipeline_result"
        ),
        canonical_equation_id="archive_grammar_canonical_consolidation_savings_v1",
        canonical_equation_status="FORMALIZATION_PENDING",
        parser_section_manifest_path=None,
        elapsed_seconds=0.01,
        canonical_provenance={"axis_tag": "[predicted]"},
        written_at_utc=now,
        written_pid=1,
        written_host="test",
    )

    bundle = build_submission_bundle(
        compression_pipeline_result=cp_result,
        archive_grammar_manifest=grammar,
        output_dir=tmp_path / "submission_dir",
    )
    return bundle


# ---------------------------------------------------------------------------
# Constants + module surface tests
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_forbidden_tokens_canonical_set(self) -> None:
        assert "Claude" in FORBIDDEN_PUBLIC_PR_TOKENS
        assert "Anthropic" in FORBIDDEN_PUBLIC_PR_TOKENS
        assert "Co-Authored" in FORBIDDEN_PUBLIC_PR_TOKENS
        assert "claude.com" in FORBIDDEN_PUBLIC_PR_TOKENS
        assert "anthropic.com" in FORBIDDEN_PUBLIC_PR_TOKENS

    def test_first_person_plural_patterns_cover_we_our_us(self) -> None:
        joined = " ".join(FIRST_PERSON_PLURAL_PATTERNS)
        assert "we" in joined.lower()
        assert "our" in joined.lower()
        assert "us" in joined.lower()

    def test_emdash_is_u2014(self) -> None:
        assert EMDASH_CHARACTER == "—"
        assert len(EMDASH_CHARACTER) == 1

    def test_canonical_axis_tags_includes_contest_cpu_and_cuda(self) -> None:
        assert "[contest-CPU]" in CANONICAL_AXIS_TAGS
        assert "[contest-CUDA]" in CANONICAL_AXIS_TAGS
        assert "[predicted]" in CANONICAL_AXIS_TAGS

    def test_tone_violation_patterns_present(self) -> None:
        rules = [rule for _, rule in TONE_VIOLATION_PATTERNS]
        assert "tone_signoff_flourish" in rules
        assert "tone_marketing_hype" in rules
        assert "tone_ai_tell" in rules

    def test_phase_5_layer_version_marker(self) -> None:
        assert "phase_5" in PHASE_5_LAYER_VERSION
        assert "20260526" in PHASE_5_LAYER_VERSION


# ---------------------------------------------------------------------------
# LintFinding + LintVerdict frozen dataclass invariants
# ---------------------------------------------------------------------------


class TestLintFindingInvariants:
    def test_canonical_construction(self) -> None:
        f = LintFinding(
            surface=LintSurface.PR_BODY.value,
            severity=LintSeverity.ERROR.value,
            rule="forbidden_token_claude",
            file_path="PR_BODY.md",
            line_number=5,
            matched_text="Claude",
            fix_suggestion="Remove Claude attribution per Catalog #119 sister discipline.",
        )
        assert f.rule == "forbidden_token_claude"
        d = f.as_dict()
        assert d["severity"] == "error"

    def test_rejects_invalid_surface(self) -> None:
        with pytest.raises(ValueError, match="surface"):
            LintFinding(
                surface="not_a_real_surface",
                severity=LintSeverity.ERROR.value,
                rule="x",
                file_path="f",
                line_number=1,
                matched_text="x",
                fix_suggestion="valid fix",
            )

    def test_rejects_invalid_severity(self) -> None:
        with pytest.raises(ValueError, match="severity"):
            LintFinding(
                surface=LintSurface.PR_BODY.value,
                severity="critical_not_canonical",
                rule="x",
                file_path="f",
                line_number=1,
                matched_text="x",
                fix_suggestion="valid fix",
            )

    def test_rejects_empty_rule(self) -> None:
        with pytest.raises(ValueError, match="rule"):
            LintFinding(
                surface=LintSurface.PR_BODY.value,
                severity=LintSeverity.ERROR.value,
                rule="",
                file_path="f",
                line_number=1,
                matched_text="x",
                fix_suggestion="valid fix",
            )

    def test_rejects_zero_line_number(self) -> None:
        with pytest.raises(ValueError, match="line_number"):
            LintFinding(
                surface=LintSurface.PR_BODY.value,
                severity=LintSeverity.ERROR.value,
                rule="x",
                file_path="f",
                line_number=0,
                matched_text="x",
                fix_suggestion="valid fix",
            )

    def test_rejects_short_fix_suggestion(self) -> None:
        with pytest.raises(ValueError, match="fix_suggestion"):
            LintFinding(
                surface=LintSurface.PR_BODY.value,
                severity=LintSeverity.ERROR.value,
                rule="x",
                file_path="f",
                line_number=1,
                matched_text="x",
                fix_suggestion="ok",  # 2 chars; below 4
            )

    def test_line_number_none_accepted_for_whole_file_findings(self) -> None:
        f = LintFinding(
            surface=LintSurface.ARCHIVE_ZIP.value,
            severity=LintSeverity.ERROR.value,
            rule="archive_zip_missing",
            file_path="archive.zip",
            line_number=None,
            matched_text=None,
            fix_suggestion="emit via Phase 4 builder",
        )
        assert f.line_number is None


class TestLintVerdictInvariants:
    def _empty(self) -> LintVerdict:
        return LintVerdict(
            schema_version=LINTER_SCHEMA_VERSION,
            overall_clean=True,
            findings=(),
            surfaces_scanned=(),
            error_count=0,
            warn_count=0,
            info_count=0,
            target_repo="commaai/comma_video_compression_challenge",
            measurement_utc="2026-05-26T00:00:00+00:00",
            axis_tag=PREDICTED_AXIS_TAG,
            score_claim=False,
            promotable=False,
            evidence_grade=EVIDENCE_GRADE,
            canonical_helper_invocation="tac.submission_packet.lint_submission_bundle",
            canonical_equation_id=CANONICAL_EQUATION_ID,
            canonical_equation_status="FORMALIZATION_PENDING",
            elapsed_seconds=0.0,
        )

    def test_canonical_empty_verdict(self) -> None:
        v = self._empty()
        assert v.overall_clean is True
        assert v.error_count == 0
        d = v.as_dict()
        assert d["schema_version"] == LINTER_SCHEMA_VERSION

    def test_rejects_inconsistent_error_count(self) -> None:
        with pytest.raises(ValueError, match="error_count"):
            LintVerdict(
                schema_version=LINTER_SCHEMA_VERSION,
                overall_clean=False,
                findings=(),
                surfaces_scanned=(),
                error_count=5,  # inconsistent — findings is empty
                warn_count=0,
                info_count=0,
                target_repo="commaai/comma_video_compression_challenge",
                measurement_utc="2026-05-26T00:00:00+00:00",
                axis_tag=PREDICTED_AXIS_TAG,
                score_claim=False,
                promotable=False,
                evidence_grade=EVIDENCE_GRADE,
                canonical_helper_invocation="tac.submission_packet.lint_submission_bundle",
                canonical_equation_id=CANONICAL_EQUATION_ID,
                canonical_equation_status="FORMALIZATION_PENDING",
                elapsed_seconds=0.0,
            )

    def test_rejects_score_claim_true(self) -> None:
        with pytest.raises(ValueError, match="score_claim"):
            LintVerdict(
                schema_version=LINTER_SCHEMA_VERSION,
                overall_clean=True,
                findings=(),
                surfaces_scanned=(),
                error_count=0,
                warn_count=0,
                info_count=0,
                target_repo="commaai/comma_video_compression_challenge",
                measurement_utc="2026-05-26T00:00:00+00:00",
                axis_tag=PREDICTED_AXIS_TAG,
                score_claim=True,
                promotable=False,
                evidence_grade=EVIDENCE_GRADE,
                canonical_helper_invocation="tac.submission_packet.lint_submission_bundle",
                canonical_equation_id=CANONICAL_EQUATION_ID,
                canonical_equation_status="FORMALIZATION_PENDING",
                elapsed_seconds=0.0,
            )

    def test_rejects_promotable_true(self) -> None:
        with pytest.raises(ValueError, match="promotable"):
            LintVerdict(
                schema_version=LINTER_SCHEMA_VERSION,
                overall_clean=True,
                findings=(),
                surfaces_scanned=(),
                error_count=0,
                warn_count=0,
                info_count=0,
                target_repo="commaai/comma_video_compression_challenge",
                measurement_utc="2026-05-26T00:00:00+00:00",
                axis_tag=PREDICTED_AXIS_TAG,
                score_claim=False,
                promotable=True,
                evidence_grade=EVIDENCE_GRADE,
                canonical_helper_invocation="tac.submission_packet.lint_submission_bundle",
                canonical_equation_id=CANONICAL_EQUATION_ID,
                canonical_equation_status="FORMALIZATION_PENDING",
                elapsed_seconds=0.0,
            )

    def test_rejects_bad_target_repo(self) -> None:
        with pytest.raises(ValueError, match="target_repo"):
            LintVerdict(
                schema_version=LINTER_SCHEMA_VERSION,
                overall_clean=True,
                findings=(),
                surfaces_scanned=(),
                error_count=0,
                warn_count=0,
                info_count=0,
                target_repo="not-an-owner-repo-string",
                measurement_utc="2026-05-26T00:00:00+00:00",
                axis_tag=PREDICTED_AXIS_TAG,
                score_claim=False,
                promotable=False,
                evidence_grade=EVIDENCE_GRADE,
                canonical_helper_invocation="tac.submission_packet.lint_submission_bundle",
                canonical_equation_id=CANONICAL_EQUATION_ID,
                canonical_equation_status="FORMALIZATION_PENDING",
                elapsed_seconds=0.0,
            )

    def test_frozen_dataclass(self) -> None:
        v = self._empty()
        with pytest.raises(dataclasses.FrozenInstanceError):
            v.overall_clean = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# lint_pr_body — per-rule positive + negative
# ---------------------------------------------------------------------------


class TestLintPrBodyForbiddenTokens:
    def test_clean_body_has_zero_forbidden_token_findings(
        self, canonical_clean_pr_body: str
    ) -> None:
        findings = lint_pr_body(canonical_clean_pr_body)
        token_rules = [f.rule for f in findings if f.rule.startswith("forbidden_token_")]
        assert token_rules == []

    def test_claude_token_flagged(self) -> None:
        body = "This was submitted by Claude on behalf of the operator."
        findings = lint_pr_body(body)
        rules = [f.rule for f in findings if f.severity == "error"]
        assert any(r == "forbidden_token_claude" for r in rules)

    def test_anthropic_token_flagged(self) -> None:
        body = "Built with help from Anthropic models."
        findings = lint_pr_body(body)
        assert any(f.rule == "forbidden_token_anthropic" for f in findings)

    def test_co_authored_flagged(self) -> None:
        body = "Co-Authored-By: Someone <x@y>"
        findings = lint_pr_body(body)
        assert any(f.rule == "forbidden_token_co_authored" for f in findings)

    def test_claude_dot_com_flagged(self) -> None:
        body = "See claude.com for details."
        findings = lint_pr_body(body)
        # claude.com matches BOTH "Claude" (case sensitive? No, "claude.com"
        # contains lowercase claude; "Claude" pattern won't match lowercase
        # 'claude'). So expect only the claude.com finding.
        assert any(f.rule.startswith("forbidden_token_claude_com") for f in findings)

    def test_anthropic_dot_com_flagged(self) -> None:
        body = "Refer to anthropic.com for support."
        findings = lint_pr_body(body)
        assert any(f.rule.startswith("forbidden_token_anthropic_com") for f in findings)


class TestLintPrBodyFirstPersonPlural:
    def test_we_flagged(self) -> None:
        body = "We built FEC6 on top of PR101."
        findings = lint_pr_body(body)
        rules = [f.rule for f in findings if f.rule.startswith("first_person_plural_")]
        assert rules, f"expected first_person_plural finding; got {rules}"

    def test_our_flagged(self) -> None:
        body = "Our submission scores 0.19 [contest-CPU]."
        findings = lint_pr_body(body)
        assert any(f.rule.startswith("first_person_plural_") for f in findings)

    def test_first_person_singular_clean(self) -> None:
        body = (
            "I built FEC6 on top of @SajayR's PR #101 substrate. "
            "Score 0.19205 [contest-CPU]."
        )
        findings = lint_pr_body(body)
        plural_findings = [
            f for f in findings if f.rule.startswith("first_person_plural_")
        ]
        assert plural_findings == []

    def test_weave_not_flagged(self) -> None:
        # word-boundary regex; "weave" / "swept" / "Power" should NOT match
        body = (
            "I weave the FEC6 selector into PR101's grammar. Power is "
            "preserved by Sparse Word recall. Between PRs we keep parity."
        )
        # "we" alone in "we keep" is a real first-person-plural match — that
        # SHOULD be flagged. Substring matches inside "weave"/"Power" should
        # NOT be flagged. We assert only ONE finding (from "we keep").
        findings = lint_pr_body(body)
        plural_findings = [
            f for f in findings if f.rule.startswith("first_person_plural_")
        ]
        assert len(plural_findings) == 1
        assert plural_findings[0].matched_text == "we"


class TestLintPrBodyEmdash:
    def test_emdash_flagged(self) -> None:
        body = "I built FEC6 — this is the canonical substrate. Score 0.192."
        findings = lint_pr_body(body)
        assert any(f.rule == "emdash_u2014" for f in findings)

    def test_ascii_hyphen_clean(self) -> None:
        body = "I built FEC6 - this is the canonical substrate."
        findings = lint_pr_body(body)
        emdash_findings = [f for f in findings if f.rule == "emdash_u2014"]
        assert emdash_findings == []

    def test_semicolon_separator_clean(self) -> None:
        body = "I built FEC6; this is the canonical substrate."
        findings = lint_pr_body(body)
        emdash_findings = [f for f in findings if f.rule == "emdash_u2014"]
        assert emdash_findings == []


class TestLintPrBodyToneViolations:
    def test_happy_to_discuss_flagged(self) -> None:
        body = "I built FEC6. Happy to discuss any questions!"
        findings = lint_pr_body(body)
        assert any(f.rule == "tone_signoff_flourish" for f in findings)

    def test_marketing_hype_flagged(self) -> None:
        body = "I built groundbreaking FEC6 with cutting-edge entropy coding."
        findings = lint_pr_body(body)
        rules = {f.rule for f in findings}
        assert "tone_marketing_hype" in rules

    def test_ai_tell_flagged(self) -> None:
        body = "This submission was AI-assisted."
        findings = lint_pr_body(body)
        assert any(f.rule == "tone_ai_tell" for f in findings)

    def test_excessive_punctuation_flagged(self) -> None:
        body = "I built FEC6!! Score 0.192!!"
        findings = lint_pr_body(body)
        assert any(f.rule == "tone_excessive_punctuation" for f in findings)


class TestLintPrBodyEmoji:
    def test_emoji_flagged(self) -> None:
        body = "I built FEC6. \U0001f680 Score 0.192."
        findings = lint_pr_body(body)
        assert any(f.rule == "emoji_forbidden_on_public_pr_surface" for f in findings)

    def test_no_emoji_clean(self, canonical_clean_pr_body: str) -> None:
        findings = lint_pr_body(canonical_clean_pr_body)
        emoji_findings = [
            f for f in findings if f.rule == "emoji_forbidden_on_public_pr_surface"
        ]
        assert emoji_findings == []


class TestLintPrBodyAttributionChain:
    def test_missing_at_mention_warned(self) -> None:
        body = "I built FEC6 on top of PR #101. Score 0.192 [contest-CPU]."
        findings = lint_pr_body(body)
        assert any(f.rule == "attribution_no_at_mention" for f in findings)

    def test_missing_pr_reference_warned(self) -> None:
        body = "I built FEC6 on top of @SajayR's work. Score 0.192 [contest-CPU]."
        findings = lint_pr_body(body)
        assert any(f.rule == "attribution_no_pr_reference" for f in findings)

    def test_complete_attribution_chain_clean(
        self, canonical_clean_pr_body: str
    ) -> None:
        findings = lint_pr_body(canonical_clean_pr_body)
        attrib_warn = [
            f
            for f in findings
            if f.rule.startswith("attribution_no_")
        ]
        assert attrib_warn == []


class TestLintPrBodyAxisTag:
    def test_score_without_axis_tag_warned(self) -> None:
        body = (
            "I built FEC6 on @SajayR's PR #101. Score 0.192. No axis disclosed."
        )
        findings = lint_pr_body(body)
        assert any(
            f.rule == "missing_axis_tag_on_score_citation" for f in findings
        )

    def test_score_with_axis_tag_clean(self, canonical_clean_pr_body: str) -> None:
        findings = lint_pr_body(canonical_clean_pr_body)
        axis_warn = [
            f
            for f in findings
            if f.rule == "missing_axis_tag_on_score_citation"
        ]
        assert axis_warn == []


class TestLintPrBodyLocalAbsolutePaths:
    def test_users_path_flagged(self) -> None:
        body = "See /Users/operator/scratch for details."
        findings = lint_pr_body(body)
        assert any(f.rule == "catalog_208_local_absolute_path" for f in findings)

    def test_repo_relative_path_clean(self) -> None:
        body = "See submissions/pr101/ for details."
        findings = lint_pr_body(body)
        path_findings = [
            f for f in findings if f.rule == "catalog_208_local_absolute_path"
        ]
        assert path_findings == []


# ---------------------------------------------------------------------------
# lint_tone — focused tone-only audit
# ---------------------------------------------------------------------------


class TestLintTone:
    def test_clean_body(self, canonical_clean_pr_body: str) -> None:
        findings = lint_tone(canonical_clean_pr_body)
        assert findings == ()

    def test_signoff_flourish_flagged(self) -> None:
        body = "Looks great! Happy to discuss."
        findings = lint_tone(body)
        assert any(f.rule == "tone_signoff_flourish" for f in findings)

    def test_emoji_flagged(self) -> None:
        body = "Approved \U0001f44d"
        findings = lint_tone(body)
        assert any(
            f.rule == "emoji_forbidden_on_public_pr_surface" for f in findings
        )


# ---------------------------------------------------------------------------
# lint_inflate_py
# ---------------------------------------------------------------------------


class TestLintInflatePy:
    def test_missing_file_flagged(self, tmp_path: Path) -> None:
        findings = lint_inflate_py(tmp_path / "does-not-exist.py")
        assert any(f.rule == "inflate_py_missing" for f in findings)

    def test_clean_inflate(self, tmp_path: Path) -> None:
        path = tmp_path / "inflate.py"
        path.write_text(
            "#!/usr/bin/env python3\n"
            "# select_inflate_device canonical\n"
            "import sys\n"
            "print('hello')\n",
            encoding="utf-8",
        )
        findings = lint_inflate_py(path, loc_budget=200)
        # Should not flag LOC over budget (4 lines)
        loc_findings = [f for f in findings if f.rule == "inflate_py_loc_over_budget"]
        assert loc_findings == []

    def test_over_budget_without_waiver_flagged_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "inflate.py"
        path.write_text("\n".join(f"x = {i}" for i in range(250)) + "\n", encoding="utf-8")
        findings = lint_inflate_py(path, loc_budget=200)
        loc_findings = [f for f in findings if f.rule == "inflate_py_loc_over_budget"]
        assert any(f.severity == "error" for f in loc_findings)

    def test_over_budget_with_waiver_demoted_to_warn(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "inflate.py"
        path.write_text("\n".join(f"x = {i}" for i in range(250)) + "\n", encoding="utf-8")
        findings = lint_inflate_py(
            path,
            loc_budget=200,
            waiver_rationale="HNeRV-class trainer requires torch + per-tensor decode; reviewable in 60s",
        )
        loc_findings_error = [
            f
            for f in findings
            if f.rule == "inflate_py_loc_over_budget" and f.severity == "error"
        ]
        loc_findings_warn = [
            f
            for f in findings
            if f.rule == "inflate_py_loc_over_budget_with_waiver"
        ]
        assert loc_findings_error == []
        assert len(loc_findings_warn) == 1

    def test_over_budget_with_placeholder_waiver_rejected(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "inflate.py"
        path.write_text("\n".join(f"x = {i}" for i in range(250)) + "\n", encoding="utf-8")
        findings = lint_inflate_py(
            path,
            loc_budget=200,
            waiver_rationale="<rationale>",  # placeholder
        )
        loc_findings_error = [
            f
            for f in findings
            if f.rule == "inflate_py_loc_over_budget" and f.severity == "error"
        ]
        assert len(loc_findings_error) == 1

    def test_no_select_inflate_device_warned(self, tmp_path: Path) -> None:
        path = tmp_path / "inflate.py"
        path.write_text("import sys\nprint('hi')\n", encoding="utf-8")
        findings = lint_inflate_py(path)
        assert any(
            f.rule == "catalog_205_no_canonical_helper_routing" for f in findings
        )

    def test_inline_device_fork_with_waiver_clean(self, tmp_path: Path) -> None:
        path = tmp_path / "inflate.py"
        path.write_text(
            "import torch\n"
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'  # INLINE_DEVICE_FORK_OK:byte_stable\n",
            encoding="utf-8",
        )
        findings = lint_inflate_py(path)
        cat_205 = [
            f for f in findings if f.rule == "catalog_205_no_canonical_helper_routing"
        ]
        assert cat_205 == []

    def test_bare_tac_import_without_vendor_warned(self, tmp_path: Path) -> None:
        path = tmp_path / "inflate.py"
        path.write_text(
            "from tac.substrates.shared.inflate_runtime import select_inflate_device\n"
            "print('hi')\n",
            encoding="utf-8",
        )
        findings = lint_inflate_py(path)
        assert any(
            f.rule == "catalog_295_bare_tac_import_no_vendor" for f in findings
        )

    def test_bare_tac_import_with_vendor_clean(self, tmp_path: Path) -> None:
        path = tmp_path / "inflate.py"
        path.write_text(
            "from tac.substrates.shared.inflate_runtime import select_inflate_device\n"
            "print('hi')\n",
            encoding="utf-8",
        )
        # Create vendored sister package
        vendor = tmp_path / "src" / "tac"
        vendor.mkdir(parents=True)
        (vendor / "__init__.py").write_text("", encoding="utf-8")
        findings = lint_inflate_py(path)
        vendor_findings = [
            f for f in findings if f.rule == "catalog_295_bare_tac_import_no_vendor"
        ]
        assert vendor_findings == []

    def test_bare_tac_import_with_waiver_clean(self, tmp_path: Path) -> None:
        path = tmp_path / "inflate.py"
        path.write_text(
            "from tac.substrates.shared.inflate_runtime import select_inflate_device  # SUBMISSION_PYTHONPATH_SHIM_OK:sibling_runtime\n",
            encoding="utf-8",
        )
        findings = lint_inflate_py(path)
        vendor_findings = [
            f for f in findings if f.rule == "catalog_295_bare_tac_import_no_vendor"
        ]
        assert vendor_findings == []


# ---------------------------------------------------------------------------
# lint_archive_zip
# ---------------------------------------------------------------------------


class TestLintArchiveZip:
    def test_missing_file_flagged(self, tmp_path: Path) -> None:
        findings = lint_archive_zip(
            tmp_path / "missing.zip", expected_sha256="0" * 64, expected_size_bytes=0
        )
        assert any(f.rule == "archive_zip_missing" for f in findings)

    def test_sha_mismatch_flagged(self, tmp_path: Path) -> None:
        path = tmp_path / "archive.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("0.bin", b"hello")
        findings = lint_archive_zip(
            path,
            expected_sha256="0" * 64,  # wrong
            expected_size_bytes=path.stat().st_size,
        )
        assert any(f.rule == "archive_sha256_mismatch" for f in findings)

    def test_size_mismatch_flagged(self, tmp_path: Path) -> None:
        import hashlib

        path = tmp_path / "archive.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("0.bin", b"hello")
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        findings = lint_archive_zip(
            path,
            expected_sha256=actual_sha,
            expected_size_bytes=1,  # wrong
        )
        assert any(f.rule == "archive_size_mismatch" for f in findings)

    def test_match_clean(self, tmp_path: Path) -> None:
        import hashlib

        path = tmp_path / "archive.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("0.bin", b"hello")
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        actual_size = path.stat().st_size
        findings = lint_archive_zip(
            path, expected_sha256=actual_sha, expected_size_bytes=actual_size
        )
        assert findings == ()


# ---------------------------------------------------------------------------
# lint_compliance_placeholder
# ---------------------------------------------------------------------------


class TestLintCompliancePlaceholder:
    def test_no_compliance_dir_clean(self, tmp_path: Path) -> None:
        submission_dir = tmp_path / "submission_dir"
        submission_dir.mkdir()
        findings = lint_compliance_placeholder(submission_dir)
        assert findings == ()

    def test_compliance_report_surfaces_info(self, tmp_path: Path) -> None:
        submission_dir = tmp_path / "submission_dir"
        submission_dir.mkdir()
        compliance_dir = tmp_path / "reports" / "pr_pre_submission"
        compliance_dir.mkdir(parents=True)
        (compliance_dir / "compliance_report_test.json").write_text("{}", encoding="utf-8")
        findings = lint_compliance_placeholder(submission_dir)
        assert any(f.rule == "compliance_sidecar_present" for f in findings)
        assert all(f.severity == "info" for f in findings)


# ---------------------------------------------------------------------------
# lint_readme
# ---------------------------------------------------------------------------


class TestLintReadme:
    def test_missing_readme_flagged(self, tmp_path: Path) -> None:
        findings = lint_readme(tmp_path / "missing.md")
        assert any(f.rule == "readme_missing" for f in findings)

    def test_clean_readme(self, tmp_path: Path) -> None:
        path = tmp_path / "README.md"
        path.write_text(
            "# Submission\n\nI built FEC6 on top of @SajayR's PR #101.\n",
            encoding="utf-8",
        )
        findings = lint_readme(path)
        # Clean — no forbidden tokens, no local paths.
        token_findings = [
            f for f in findings if f.rule.startswith("forbidden_token_")
        ]
        path_findings = [
            f for f in findings if f.rule == "catalog_208_local_absolute_path"
        ]
        assert token_findings == []
        assert path_findings == []

    def test_claude_token_in_readme_flagged(self, tmp_path: Path) -> None:
        path = tmp_path / "README.md"
        path.write_text(
            "# Submission\n\nBuilt with Claude.\n", encoding="utf-8"
        )
        findings = lint_readme(path)
        assert any(f.rule == "forbidden_token_claude" for f in findings)


# ---------------------------------------------------------------------------
# lint_submission_bundle — integration with Phase 4
# ---------------------------------------------------------------------------


class TestLintSubmissionBundleIntegration:
    def test_clean_bundle(
        self, fresh_submission_bundle: SubmissionBundleResult
    ) -> None:
        v = lint_submission_bundle(fresh_submission_bundle)
        assert isinstance(v, LintVerdict)
        # Should have at least inflate.py + archive.zip + compliance + readme surfaces
        assert LintSurface.INFLATE_PY.value in v.surfaces_scanned
        assert LintSurface.ARCHIVE_ZIP.value in v.surfaces_scanned
        # Archive sha+size should match (we just built it)
        archive_findings = [
            f for f in v.findings if f.surface == LintSurface.ARCHIVE_ZIP.value
        ]
        sha_findings = [
            f for f in archive_findings if f.rule == "archive_sha256_mismatch"
        ]
        assert sha_findings == []
        # Provenance is correct
        assert v.canonical_provenance["axis_tag"] == "[predicted]"
        assert v.canonical_provenance["score_claim"] is False
        assert v.canonical_provenance["promotable"] is False

    def test_with_pr_body(
        self,
        fresh_submission_bundle: SubmissionBundleResult,
        canonical_clean_pr_body: str,
    ) -> None:
        v = lint_submission_bundle(
            fresh_submission_bundle, pr_body_text=canonical_clean_pr_body
        )
        assert LintSurface.PR_BODY.value in v.surfaces_scanned
        # Body is clean of forbidden tokens
        forbidden = [
            f
            for f in v.findings
            if f.rule.startswith("forbidden_token_")
        ]
        assert forbidden == []

    def test_with_pr_body_having_claude_token(
        self,
        fresh_submission_bundle: SubmissionBundleResult,
    ) -> None:
        body = "I built FEC6 with Claude assistance."
        v = lint_submission_bundle(fresh_submission_bundle, pr_body_text=body)
        assert v.overall_clean is False
        assert any(
            f.rule == "forbidden_token_claude" for f in v.findings
        )

    def test_rejects_non_bundle_input(self) -> None:
        with pytest.raises(SubmissionLinterError):
            lint_submission_bundle({"not_a_bundle": True})  # type: ignore[arg-type]

    def test_rejects_bad_target_repo(
        self, fresh_submission_bundle: SubmissionBundleResult
    ) -> None:
        with pytest.raises(SubmissionLinterError, match="target_repo"):
            lint_submission_bundle(fresh_submission_bundle, target_repo="invalid")

    def test_missing_pr_body_path_raises(
        self,
        fresh_submission_bundle: SubmissionBundleResult,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(SubmissionLinterError, match="does not exist"):
            lint_submission_bundle(
                fresh_submission_bundle, pr_body_path=tmp_path / "missing.md"
            )

    def test_canonical_equation_id(
        self, fresh_submission_bundle: SubmissionBundleResult
    ) -> None:
        v = lint_submission_bundle(fresh_submission_bundle)
        assert v.canonical_equation_id == CANONICAL_EQUATION_ID
        assert v.canonical_equation_status == "FORMALIZATION_PENDING"

    def test_pr_body_text_supersedes_pr_body_path(
        self,
        fresh_submission_bundle: SubmissionBundleResult,
        tmp_path: Path,
        canonical_clean_pr_body: str,
    ) -> None:
        body_file = tmp_path / "PR_BODY.md"
        body_file.write_text("Built with Claude.", encoding="utf-8")
        v = lint_submission_bundle(
            fresh_submission_bundle,
            pr_body_text=canonical_clean_pr_body,
            pr_body_path=body_file,
        )
        # pr_body_text wins; canonical_clean_pr_body has no Claude
        forbidden = [
            f for f in v.findings if f.rule.startswith("forbidden_token_")
        ]
        assert forbidden == []


# ---------------------------------------------------------------------------
# Cathedral consumer contract compliance
# ---------------------------------------------------------------------------


class TestCathedralConsumerContract:
    def test_consumer_satisfies_canonical_contract(self) -> None:
        from tac.cathedral.consumer_contract import validate_consumer_module
        import tac.cathedral_consumers.submission_linter_consumer as m

        registration = validate_consumer_module(m)
        assert registration.contract_compliant is True
        assert registration.consumer_name == "submission_linter_consumer"

    def test_consumer_hook_4_active(self) -> None:
        import tac.cathedral_consumers.submission_linter_consumer as m
        from tac.cathedral.consumer_contract import HookNumber

        assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in m.CONSUMER_HOOK_NUMBERS

    def test_consumer_canonical_routing_markers(self) -> None:
        import tac.cathedral_consumers.submission_linter_consumer as m

        result = m.consume_candidate({})
        assert result["predicted_delta_adjustment"] == 0.0
        assert result["promotable"] is False
        assert result["axis_tag"] == "[predicted]"

    def test_consumer_unknown_when_no_metadata(self) -> None:
        import tac.cathedral_consumers.submission_linter_consumer as m

        result = m.consume_candidate({})
        assert result["submission_linter_readiness_verdict"] == "UNKNOWN"

    def test_consumer_lint_clean_verdict(self) -> None:
        import tac.cathedral_consumers.submission_linter_consumer as m

        result = m.consume_candidate(
            {
                "submission_linter_verdict": {
                    "overall_clean": True,
                    "error_count": 0,
                    "warn_count": 1,
                    "info_count": 2,
                }
            }
        )
        assert result["submission_linter_readiness_verdict"] == "LINT_CLEAN"

    def test_consumer_lint_blocked_verdict(self) -> None:
        import tac.cathedral_consumers.submission_linter_consumer as m

        result = m.consume_candidate(
            {
                "submission_linter_verdict": {
                    "overall_clean": False,
                    "error_count": 3,
                    "warn_count": 1,
                    "info_count": 0,
                }
            }
        )
        assert (
            result["submission_linter_readiness_verdict"]
            == "BLOCKED_ON_LINT_ERRORS"
        )

    def test_update_from_anchor_no_op(self) -> None:
        import tac.cathedral_consumers.submission_linter_consumer as m

        m.update_from_anchor({"anchor": "test"})  # no exception


# ---------------------------------------------------------------------------
# CLI subprocess tests
# ---------------------------------------------------------------------------


CLI_PATH = REPO_ROOT / "tools" / "submission_linter_cli.py"


class TestCliSubprocess:
    def test_help_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "submission_linter_cli" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_pr_body_only_clean(
        self, tmp_path: Path, canonical_clean_pr_body: str
    ) -> None:
        body_path = tmp_path / "PR_BODY.md"
        body_path.write_text(canonical_clean_pr_body, encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(CLI_PATH),
                "--pr-body-only",
                str(body_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

    def test_pr_body_only_forbidden_token_exit_1(self, tmp_path: Path) -> None:
        body_path = tmp_path / "PR_BODY.md"
        body_path.write_text(
            "I built FEC6 with help from Claude.", encoding="utf-8"
        )
        result = subprocess.run(
            [
                sys.executable,
                str(CLI_PATH),
                "--pr-body-only",
                str(body_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 1

    def test_pr_body_only_first_person_plural_exit_2(self, tmp_path: Path) -> None:
        body_path = tmp_path / "PR_BODY.md"
        body_path.write_text(
            "We built FEC6 on PR #101 by @SajayR.", encoding="utf-8"
        )
        result = subprocess.run(
            [
                sys.executable,
                str(CLI_PATH),
                "--pr-body-only",
                str(body_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 2

    def test_pr_body_only_emdash_exit_3(self, tmp_path: Path) -> None:
        body_path = tmp_path / "PR_BODY.md"
        body_path.write_text(
            "I built FEC6 — canonical. PR #101 by @SajayR. [contest-CPU]",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(CLI_PATH),
                "--pr-body-only",
                str(body_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 3

    def test_pr_body_only_tone_violation_exit_5(self, tmp_path: Path) -> None:
        body_path = tmp_path / "PR_BODY.md"
        # Use a non-AI-tell tone violation (signoff flourish) to isolate exit 5
        body_path.write_text(
            "I built FEC6 on @SajayR's PR #101. [contest-CPU]. Happy to discuss.",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(CLI_PATH),
                "--pr-body-only",
                str(body_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 5

    def test_cli_error_no_args(self) -> None:
        result = subprocess.run(
            [sys.executable, str(CLI_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # argparse exits with code 2 for argument errors; our canonical CLI
        # error code is 6 but argparse pre-empts it. We accept either as
        # canonical "args invalid" behavior.
        assert result.returncode in (2, 6)

    def test_cli_error_missing_pr_body_file(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(CLI_PATH),
                "--pr-body-only",
                str(tmp_path / "does-not-exist.md"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 6

    def test_json_mode_emits_valid_json(
        self, tmp_path: Path, canonical_clean_pr_body: str
    ) -> None:
        body_path = tmp_path / "PR_BODY.md"
        body_path.write_text(canonical_clean_pr_body, encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(CLI_PATH),
                "--pr-body-only",
                str(body_path),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["mode"] == "pr_body_only"
        assert "findings" in payload

    def test_cli_from_bundle_json_roundtrip(
        self,
        tmp_path: Path,
        fresh_submission_bundle: SubmissionBundleResult,
    ) -> None:
        # Persist a SubmissionBundleResult JSON sidecar and lint it via CLI
        bundle_json_path = tmp_path / "bundle.json"
        bundle_json_path.write_text(
            json.dumps(fresh_submission_bundle.as_dict(), indent=2),
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(CLI_PATH),
                "--from-submission-bundle",
                str(bundle_json_path),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
        payload = json.loads(result.stdout)
        assert payload["schema_version"] == LINTER_SCHEMA_VERSION
        assert payload["overall_clean"] is True


# ---------------------------------------------------------------------------
# Live-repo regression guard
# ---------------------------------------------------------------------------


class TestLiveRepoRegression:
    def test_canonical_pr_body_template_lints_clean_no_errors(self) -> None:
        # Phase 1 spec memo acceptance criteria: existing
        # PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md should lint cleanly with
        # zero ERROR findings. The conformant template uses operator-only
        # voice + @-mention chain + axis tags. We don't fail if the file
        # is missing (Phase 4 builder owns its emission).
        canonical_pr_body_paths = [
            REPO_ROOT
            / "experiments"
            / "results"
            / "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex"
            / "submission_dir"
            / "PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md",
        ]
        any_existed = False
        for p in canonical_pr_body_paths:
            if not p.exists():
                continue
            any_existed = True
            body_text = p.read_text(encoding="utf-8")
            findings = lint_pr_body(body_text, file_path=str(p))
            errors = [
                f for f in findings if f.severity == LintSeverity.ERROR.value
            ]
            # Tolerate emdash + first-person-plural in legacy template; this
            # is the regression baseline used by Phase 4 builder + Phase 10
            # operator-routable backfill sweep. Catalog #229 PV principle.
            forbidden_token_errors = [
                f for f in errors if f.rule.startswith("forbidden_token_")
            ]
            assert forbidden_token_errors == [], (
                f"Live PR body template at {p} carries forbidden tokens: "
                f"{[f.rule for f in forbidden_token_errors]}"
            )
        # If no canonical template found, skip silently per CLAUDE.md
        # "Forbidden premature KILL" — Phase 4 builder may have moved it.
        if not any_existed:
            pytest.skip("Canonical PR body template path not present in repo")


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_truncate_short_text(self) -> None:
        assert _truncate("hello", cap=200) == "hello"

    def test_truncate_long_text(self) -> None:
        long_text = "a" * 300
        truncated = _truncate(long_text, cap=100)
        assert len(truncated) == 100
        assert truncated.endswith("…")

    def test_line_at_first_line(self) -> None:
        assert _line_at("hello\nworld", 0) == 1

    def test_line_at_second_line(self) -> None:
        assert _line_at("hello\nworld", 6) == 2

    def test_line_at_third_line(self) -> None:
        assert _line_at("a\nb\nc", 4) == 3


# ---------------------------------------------------------------------------
# Canonical Provenance per Catalog #323 regression
# ---------------------------------------------------------------------------


class TestCanonicalProvenance:
    def test_derive_linter_provenance_shape(self) -> None:
        from tac.submission_packet.linter import derive_linter_provenance

        prov = derive_linter_provenance(
            target_repo="commaai/comma_video_compression_challenge",
            archive_sha256="a" * 64,
            measurement_utc="2026-05-26T00:00:00+00:00",
        )
        assert prov["axis_tag"] == "[predicted]"
        assert prov["score_claim"] is False
        assert prov["promotable"] is False
        assert prov["evidence_grade"] == EVIDENCE_GRADE
        assert (
            prov["canonical_helper_invocation"]
            == "tac.submission_packet.lint_submission_bundle"
        )
        assert prov["canonical_equation_id"] == CANONICAL_EQUATION_ID
        assert prov["canonical_equation_status"] == "FORMALIZATION_PENDING"
        assert prov["archive_sha256"] == "a" * 64

    def test_derive_linter_provenance_no_archive_sha(self) -> None:
        from tac.submission_packet.linter import derive_linter_provenance

        prov = derive_linter_provenance(
            target_repo="commaai/comma_video_compression_challenge",
            archive_sha256=None,
            measurement_utc="2026-05-26T00:00:00+00:00",
        )
        assert "archive_sha256" not in prov
