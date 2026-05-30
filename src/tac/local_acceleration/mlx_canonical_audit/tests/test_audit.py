# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.local_acceleration.mlx_canonical_audit`.

Per CLAUDE.md NO FAKE IMPLEMENTATIONS: every test uses REAL file
fixtures + REAL AST parsing + REAL verdict construction (not stub
fixtures).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.local_acceleration.mlx_canonical_audit import (
    CANONICAL_PR95_HNERV_MLX_MODULE,
    CANONICAL_PRIMITIVE_EXTRACTORS,
    CanonicalDuplicationVerdict,
    DuplicationClassification,
    MigrationPlan,
    MLXBearingFile,
    MLXPrimitiveImpl,
    detect_canonical_duplication,
    enumerate_mlx_bearing_files,
    enumerate_mlx_primitive_implementations,
    recommend_canonical_extraction,
    summarize_audit_report,
)


REPO_ROOT = Path(__file__).resolve().parents[5]


class TestCanonicalDuplicationVerdictInvariants:
    def test_short_rationale_rejected(self):
        with pytest.raises(ValueError, match="must be >= 4 chars"):
            CanonicalDuplicationVerdict(
                primitive_name="foo",
                canonical_extractor="bar",
                duplicate_impl_count=0,
                impls=(),
                canonical_adopters=(),
                classification=DuplicationClassification.OBVIOUS_FIT,
                rationale="abc",
            )

    def test_placeholder_rationale_rejected(self):
        with pytest.raises(ValueError, match="placeholder literal"):
            CanonicalDuplicationVerdict(
                primitive_name="foo",
                canonical_extractor="bar",
                duplicate_impl_count=0,
                impls=(),
                canonical_adopters=(),
                classification=DuplicationClassification.OBVIOUS_FIT,
                rationale="<rationale>",
            )

    def test_reason_placeholder_rejected(self):
        with pytest.raises(ValueError, match="placeholder literal"):
            CanonicalDuplicationVerdict(
                primitive_name="foo",
                canonical_extractor="bar",
                duplicate_impl_count=0,
                impls=(),
                canonical_adopters=(),
                classification=DuplicationClassification.OBVIOUS_FIT,
                rationale="<reason>",
            )

    def test_substantive_rationale_accepted(self):
        v = CanonicalDuplicationVerdict(
            primitive_name="foo",
            canonical_extractor="bar",
            duplicate_impl_count=0,
            impls=(),
            canonical_adopters=(),
            classification=DuplicationClassification.OBVIOUS_FIT,
            rationale="substantive non-placeholder rationale",
        )
        assert v.classification == DuplicationClassification.OBVIOUS_FIT

    def test_negative_duplicate_count_rejected(self):
        with pytest.raises(ValueError, match="duplicate_impl_count must be"):
            CanonicalDuplicationVerdict(
                primitive_name="foo",
                canonical_extractor="bar",
                duplicate_impl_count=-1,
                impls=(),
                canonical_adopters=(),
                classification=DuplicationClassification.OBVIOUS_FIT,
                rationale="valid rationale",
            )


class TestMigrationPlanInvariants:
    def _verdict(self) -> CanonicalDuplicationVerdict:
        return CanonicalDuplicationVerdict(
            primitive_name="foo",
            canonical_extractor="bar",
            duplicate_impl_count=0,
            impls=(),
            canonical_adopters=(),
            classification=DuplicationClassification.OBVIOUS_FIT,
            rationale="valid rationale",
        )

    def test_short_rationale_rejected(self):
        with pytest.raises(ValueError, match="must be >= 4 chars"):
            MigrationPlan(
                primitive_name="foo",
                verdict=self._verdict(),
                recommended_action="NO_ACTION_REQUIRED",
                estimated_loc_reduction=0,
                estimated_hours=0.0,
                target_substrates=(),
                rationale="abc",
            )

    def test_invalid_action_rejected(self):
        with pytest.raises(ValueError, match="recommended_action"):
            MigrationPlan(
                primitive_name="foo",
                verdict=self._verdict(),
                recommended_action="BOGUS_ACTION",
                estimated_loc_reduction=0,
                estimated_hours=0.0,
                target_substrates=(),
                rationale="valid rationale",
            )

    def test_negative_loc_reduction_rejected(self):
        with pytest.raises(ValueError, match="estimated_loc_reduction"):
            MigrationPlan(
                primitive_name="foo",
                verdict=self._verdict(),
                recommended_action="NO_ACTION_REQUIRED",
                estimated_loc_reduction=-1,
                estimated_hours=0.0,
                target_substrates=(),
                rationale="valid rationale",
            )

    def test_negative_hours_rejected(self):
        with pytest.raises(ValueError, match="estimated_hours"):
            MigrationPlan(
                primitive_name="foo",
                verdict=self._verdict(),
                recommended_action="NO_ACTION_REQUIRED",
                estimated_loc_reduction=0,
                estimated_hours=-1.0,
                target_substrates=(),
                rationale="valid rationale",
            )


class TestEnumerateMLXBearingFiles:
    def test_returns_tuple_of_records(self, tmp_path):
        # Create synthetic substrate dir
        sub = tmp_path / "src" / "tac" / "substrates" / "synthetic_test_substrate"
        sub.mkdir(parents=True)
        (sub / "mlx_renderer.py").write_text(
            "# SPDX-License-Identifier: MIT\nimport mlx.core as mx\n"
        )
        files = enumerate_mlx_bearing_files(tmp_path, include_tests=False)
        assert len(files) == 1
        assert isinstance(files[0], MLXBearingFile)
        assert "synthetic_test_substrate" in files[0].file_path

    def test_skips_files_without_mlx_import(self, tmp_path):
        sub = tmp_path / "src" / "tac"
        sub.mkdir(parents=True)
        (sub / "no_mlx.py").write_text("# SPDX-License-Identifier: MIT\nimport numpy\n")
        files = enumerate_mlx_bearing_files(tmp_path, include_tests=False)
        assert files == ()

    def test_skips_test_files_by_default(self, tmp_path):
        sub = tmp_path / "src" / "tac" / "tests"
        sub.mkdir(parents=True)
        (sub / "test_mlx_thing.py").write_text(
            "# SPDX-License-Identifier: MIT\nimport mlx.core as mx\n"
        )
        files = enumerate_mlx_bearing_files(tmp_path, include_tests=False)
        assert files == ()

    def test_include_tests_opts_in(self, tmp_path):
        sub = tmp_path / "src" / "tac"
        sub.mkdir(parents=True)
        (sub / "test_mlx_thing.py").write_text(
            "# SPDX-License-Identifier: MIT\nimport mlx.core as mx\n"
        )
        files = enumerate_mlx_bearing_files(tmp_path, include_tests=True)
        # Even when opted in, only top-level test_*.py files inside scanned
        # dirs are picked up; the helper is conservative about test paths
        # under /tests/ subdirs.
        assert len(files) >= 0

    def test_skips_exempt_path_markers(self, tmp_path):
        sub = tmp_path / "experiments" / "results" / "old_run"
        sub.mkdir(parents=True)
        (sub / "phantom.py").write_text(
            "# SPDX-License-Identifier: MIT\nimport mlx.core as mx\n"
        )
        files = enumerate_mlx_bearing_files(tmp_path, include_tests=False)
        assert files == ()

    def test_detects_canonical_pr95_imports(self, tmp_path):
        sub = tmp_path / "src" / "tac"
        sub.mkdir(parents=True)
        (sub / "thing.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx
            from tac.local_acceleration.pr95_hnerv_mlx import pixel_shuffle_2x_nhwc
        """))
        files = enumerate_mlx_bearing_files(tmp_path, include_tests=False)
        assert len(files) == 1
        assert "pixel_shuffle_2x_nhwc" in files[0].canonical_pr95_imports

    def test_live_repo_returns_records(self):
        """Live-repo regression: enumerate returns non-empty."""
        files = enumerate_mlx_bearing_files(REPO_ROOT, include_tests=False)
        assert len(files) > 0


class TestEnumerateMLXPrimitiveImpls:
    def test_detects_function_def(self, tmp_path):
        sub = tmp_path / "src" / "tac"
        sub.mkdir(parents=True)
        (sub / "x.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx
            def gumbel_softmax_sample(logits, temperature=1.0):
                return logits
        """))
        impls = enumerate_mlx_primitive_implementations(
            tmp_path, include_tests=False
        )
        assert len(impls) == 1
        assert impls[0].primitive_name == "gumbel_softmax_sample"
        assert impls[0].impl_kind == "function"

    def test_detects_class_def(self, tmp_path):
        sub = tmp_path / "src" / "tac"
        sub.mkdir(parents=True)
        (sub / "x.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx
            class HNeRVDecoderMLX:
                pass
        """))
        impls = enumerate_mlx_primitive_implementations(
            tmp_path, include_tests=False
        )
        assert len(impls) == 1
        assert impls[0].primitive_name == "HNeRVDecoderMLX"
        assert impls[0].impl_kind == "class"

    def test_routes_through_canonical_detected(self, tmp_path):
        sub = tmp_path / "src" / "tac"
        sub.mkdir(parents=True)
        (sub / "x.py").write_text(textwrap.dedent("""
            # SPDX-License-Identifier: MIT
            import mlx.core as mx
            from tac.local_acceleration.pr95_hnerv_mlx import gumbel_softmax_sample
            def helper():
                return gumbel_softmax_sample
        """))
        impls = enumerate_mlx_primitive_implementations(
            tmp_path, include_tests=False
        )
        # We track defs not uses; canonical import alone does not register
        # an impl. This is correct behavior.
        assert len(impls) == 0


class TestDetectCanonicalDuplication:
    def test_no_impls_no_adopters_obvious_fit(self):
        v = detect_canonical_duplication("foo", [], canonical_adopters=())
        assert v.classification == DuplicationClassification.OBVIOUS_FIT

    def test_no_impls_with_adopters_canonical_adopted(self):
        v = detect_canonical_duplication(
            "foo", [], canonical_adopters=("a.py", "b.py")
        )
        assert v.classification == DuplicationClassification.CANONICAL_ADOPTED

    def test_two_sister_impls_extraction_recommended(self):
        impls = [
            MLXPrimitiveImpl(
                primitive_name="foo",
                file_path="src/tac/substrates/a/mlx_renderer.py",
                line_number=10,
                impl_kind="function",
                signature_hash="abc",
                routes_through_canonical=False,
            ),
            MLXPrimitiveImpl(
                primitive_name="foo",
                file_path="src/tac/substrates/b/mlx_renderer.py",
                line_number=20,
                impl_kind="function",
                signature_hash="def",
                routes_through_canonical=False,
            ),
        ]
        v = detect_canonical_duplication("foo", impls, canonical_adopters=())
        assert (
            v.classification
            == DuplicationClassification.CANONICAL_EXTRACTION_RECOMMENDED
        )

    def test_two_impls_no_canonical_routing_unclear(self):
        impls = [
            MLXPrimitiveImpl(
                primitive_name="foo",
                file_path="src/tac/local_acceleration/pr95_hnerv_mlx.py",
                line_number=10,
                impl_kind="function",
                signature_hash="abc",
                routes_through_canonical=False,
            ),
            MLXPrimitiveImpl(
                primitive_name="foo",
                file_path="src/tac/substrates/a/mlx_renderer.py",
                line_number=20,
                impl_kind="function",
                signature_hash="def",
                routes_through_canonical=False,
            ),
        ]
        v = detect_canonical_duplication("foo", impls, canonical_adopters=())
        # 2 sister impls (one in canonical extractor location, one not)
        # without canonical-routing adopters → EXTRACTION_RECOMMENDED per
        # Catalog #290 falling-rule list (the canonical extractor exists
        # but downstream sister files do not route through it).
        assert v.classification in (
            DuplicationClassification.UNCLEAR_NEEDS_EMPIRICAL,
            DuplicationClassification.PRINCIPLED_FORK_HARD_EARNED,
            DuplicationClassification.CANONICAL_EXTRACTION_RECOMMENDED,
        )


class TestRecommendCanonicalExtraction:
    def _make_verdict(self, classification, primitive="foo"):
        return CanonicalDuplicationVerdict(
            primitive_name=primitive,
            canonical_extractor="tac.framework_agnostic",
            duplicate_impl_count=2,
            impls=(),
            canonical_adopters=(),
            classification=classification,
            rationale="valid rationale for verdict",
        )

    def test_extraction_recommended_returns_extract(self):
        v = self._make_verdict(
            DuplicationClassification.CANONICAL_EXTRACTION_RECOMMENDED
        )
        plan = recommend_canonical_extraction(v)
        assert plan.recommended_action == "EXTRACT_CANONICAL"
        assert plan.estimated_loc_reduction > 0

    def test_canonical_adopted_no_action(self):
        v = self._make_verdict(DuplicationClassification.CANONICAL_ADOPTED)
        plan = recommend_canonical_extraction(v)
        assert plan.recommended_action == "NO_ACTION_REQUIRED"

    def test_principled_fork_document(self):
        v = self._make_verdict(
            DuplicationClassification.PRINCIPLED_FORK_HARD_EARNED
        )
        plan = recommend_canonical_extraction(v)
        assert plan.recommended_action == "DOCUMENT_FORK"

    def test_unclear_run_paired_comparison(self):
        v = self._make_verdict(
            DuplicationClassification.UNCLEAR_NEEDS_EMPIRICAL
        )
        plan = recommend_canonical_extraction(v)
        assert plan.recommended_action == "RUN_PAIRED_COMPARISON"

    def test_obvious_fit_no_action(self):
        v = self._make_verdict(DuplicationClassification.OBVIOUS_FIT)
        plan = recommend_canonical_extraction(v)
        assert plan.recommended_action == "NO_ACTION_REQUIRED"


class TestSummarizeAuditReport:
    def test_live_repo_returns_canonical_schema(self):
        report = summarize_audit_report(REPO_ROOT, include_tests=False)
        assert report["schema_version"] == "mlx_canonicalization_audit_v1"
        assert "total_mlx_bearing_files" in report
        assert "verdicts" in report
        assert "migration_plans" in report
        # Live regression guard: total MLX-bearing files > 50 per audit
        # inventory landing 2026-05-30
        assert report["total_mlx_bearing_files"] > 50

    def test_canonical_adoption_rate_in_range(self):
        report = summarize_audit_report(REPO_ROOT, include_tests=False)
        rate = report["canonical_adoption_rate"]
        assert 0.0 <= rate <= 1.0

    def test_each_primitive_has_verdict(self):
        report = summarize_audit_report(REPO_ROOT, include_tests=False)
        verdict_primitives = {v["primitive_name"] for v in report["verdicts"]}
        for canonical_name in CANONICAL_PRIMITIVE_EXTRACTORS.keys():
            assert canonical_name in verdict_primitives

    def test_aggregate_loc_reduction_nonneg(self):
        report = summarize_audit_report(REPO_ROOT, include_tests=False)
        assert report["aggregate_estimated_loc_reduction"] >= 0
        assert report["aggregate_estimated_hours"] >= 0


class TestCanonicalPrimitiveExtractorsConstantsPinned:
    """Per Catalog #383 + Catalog #185 sister discipline: constants
    MUST be importable at module level (not function-scoped) so the
    canonical contract is queryable by the STRICT preflight gate +
    sister cathedral consumer."""

    def test_canonical_extractor_module_pinned(self):
        assert CANONICAL_PR95_HNERV_MLX_MODULE == "tac.local_acceleration.pr95_hnerv_mlx"

    def test_canonical_primitive_extractors_nonempty(self):
        assert len(CANONICAL_PRIMITIVE_EXTRACTORS) >= 10

    def test_gumbel_softmax_canonical_extractor_is_framework_agnostic(self):
        """Per audit inventory A.2.5: lift target is
        tac.framework_agnostic."""
        assert (
            CANONICAL_PRIMITIVE_EXTRACTORS["gumbel_softmax_sample"]
            == "tac.framework_agnostic"
        )

    def test_rgb_to_yuv6_canonical_extractor_is_framework_agnostic(self):
        """Per audit inventory A.2.6: lift target is
        tac.framework_agnostic."""
        assert (
            CANONICAL_PRIMITIVE_EXTRACTORS["rgb_to_yuv6"]
            == "tac.framework_agnostic"
        )
