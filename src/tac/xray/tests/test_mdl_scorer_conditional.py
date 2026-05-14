"""Tests for F1: tac.xray.mdl_scorer_conditional.

The structural-tier MDL estimator wraps tac.analysis.scorer_conditional_mdl.
We test both the typed API surface (frozen dataclass invariants, hooks
declaration) and end-to-end behavior against synthetic archives.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tac.xray.base import XRayPrimitive, XRayPrimitiveResult
from tac.xray.mdl_scorer_conditional import (
    MDLDeltaReport,
    MDLDensityResult,
    ScorerConditionalMDLEstimator,
)


# ── MDLDensityResult dataclass invariants ────────────────────────────────


def test_mdl_density_result_rejects_negative_bytes():
    with pytest.raises(ValueError, match="non-negative"):
        MDLDensityResult(
            total_archive_bytes=-1,
            measurable_payload_bytes=0,
            aggregated_role_entropy_bits_per_byte=0.0,
            per_section_breakdown=(),
            mdl_density=0.0,
        )


def test_mdl_density_result_rejects_entropy_above_8():
    with pytest.raises(ValueError, match="0.0, 8.0"):
        MDLDensityResult(
            total_archive_bytes=100,
            measurable_payload_bytes=50,
            aggregated_role_entropy_bits_per_byte=9.0,
            per_section_breakdown=(),
            mdl_density=0.5,
        )


def test_mdl_density_result_rejects_density_above_1():
    with pytest.raises(ValueError, match="mdl_density must be"):
        MDLDensityResult(
            total_archive_bytes=100,
            measurable_payload_bytes=50,
            aggregated_role_entropy_bits_per_byte=4.0,
            per_section_breakdown=(),
            mdl_density=1.5,
        )


def test_mdl_density_result_accepts_well_formed():
    r = MDLDensityResult(
        total_archive_bytes=100,
        measurable_payload_bytes=80,
        aggregated_role_entropy_bits_per_byte=4.0,
        per_section_breakdown=(("sec1", "decoder", 80, 4.0),),
        mdl_density=0.5,
    )
    assert r.mdl_density == 0.5
    assert len(r.per_section_breakdown) == 1


# ── MDLDeltaReport dataclass invariants ──────────────────────────────────


def test_mdl_delta_report_rejects_density_delta_out_of_range():
    with pytest.raises(ValueError, match="density_delta must be"):
        MDLDeltaReport(
            archive_a_path=Path("a"),
            archive_b_path=Path("b"),
            archive_a_sha256="x",
            archive_b_sha256="y",
            density_a=0.5,
            density_b=0.5,
            density_delta=2.0,
            classification="within_class",
        )


def test_mdl_delta_report_accepts_well_formed():
    r = MDLDeltaReport(
        archive_a_path=Path("a"),
        archive_b_path=Path("b"),
        archive_a_sha256="x" * 64,
        archive_b_sha256="y" * 64,
        density_a=0.5,
        density_b=0.45,
        density_delta=-0.05,
        classification="within_class",
    )
    assert r.classification == "within_class"


# ── Estimator API surface ────────────────────────────────────────────────


def test_estimator_implements_protocol():
    est = ScorerConditionalMDLEstimator()
    assert isinstance(est, XRayPrimitive)


def test_estimator_name():
    est = ScorerConditionalMDLEstimator()
    assert est.name == "mdl_scorer_conditional"


def test_estimator_wire_in_hooks():
    est = ScorerConditionalMDLEstimator()
    assert "continual_learning" in est.wire_in_hooks
    assert "probe_disambiguator" in est.wire_in_hooks
    assert "cathedral_autopilot" in est.wire_in_hooks


def test_estimator_compute_refuses_nonexistent_archive(tmp_path):
    est = ScorerConditionalMDLEstimator()
    with pytest.raises(ValueError, match="does not exist"):
        est.compute(tmp_path / "no_such_archive.zip")


# ── End-to-end against synthetic archive ──────────────────────────────────


def _build_pr101_minimum_archive(path: Path) -> bytes:
    """Build a zip archive containing a PR101-parseable payload.

    The PR101 parser requires ``DECODER_BLOB_LEN + LATENT_BLOB_LEN``
    bytes; we copy those constants from the existing module so the
    test stays in sync with the upstream parser. The byte stream is a
    deterministic counter (1103515245-style LCG) so test runs reproduce.
    """
    from tac.analysis.hnerv_packet_sections import (
        PR101_DECODER_BLOB_LEN,
        PR101_LATENT_BLOB_LEN,
    )

    size_bytes = PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN
    out = bytearray()
    state = 0x91827364
    for _ in range(size_bytes):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(state & 0xFF)
    rng = bytes(out)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        # Name contains "pr101" which routes parser to PARSER_PR101.
        zf.writestr("pr101_hnerv_ft_microcodec_payload.bin", rng)
    return rng


# Back-compat alias for tests below.
_build_random_archive = _build_pr101_minimum_archive


def test_estimator_compute_returns_canonical_result_envelope(tmp_path):
    archive = tmp_path / "test_archive.zip"
    _build_random_archive(archive)
    est = ScorerConditionalMDLEstimator()
    # The structural-tier estimator may not have a parser for arbitrary
    # zip files; the function still returns a manifest with an empty
    # sections list. We tolerate that here — the API surface is what's
    # tested, not the structural tier's parser coverage.
    result = est.compute(archive)
    assert isinstance(result, XRayPrimitiveResult)
    assert result.primitive_name == "mdl_scorer_conditional"
    assert result.archive_or_video_path == archive
    assert result.archive_sha256 is not None
    assert len(result.archive_sha256) == 64
    assert result.evidence_grade == "mathematical-derivation"
    assert result.confidence_band is not None
    assert isinstance(result.primitive_value, MDLDensityResult)
    assert result.primitive_value.total_archive_bytes > 0


def test_estimator_compute_metadata_records_delegated_module(tmp_path):
    archive = tmp_path / "x.zip"
    _build_random_archive(archive)
    est = ScorerConditionalMDLEstimator()
    result = est.compute(archive)
    assert result.metadata["delegated_module"] == "tac.analysis.scorer_conditional_mdl"
    assert result.metadata["tier"] == "structural"


# ── compose_with returns ComposedXRayPrimitive ───────────────────────────


def test_estimator_compose_with_returns_composed():
    from tac.xray.base import ComposedXRayPrimitive

    est = ScorerConditionalMDLEstimator()

    class _Other:
        name = "other"
        wire_in_hooks = ("sensitivity_map",)

        def compute(self, target, **kw):
            return XRayPrimitiveResult(
                primitive_name="other",
                archive_or_video_path=None,
                archive_sha256=None,
                primitive_value=1.0,
                evidence_grade="mathematical-derivation",
                confidence_band=None,
                composes_with=(),
                wire_in_hooks_engaged=("sensitivity_map",),
            )

        def compose_with(self, other):
            return ComposedXRayPrimitive(left=self, right=other)

    composed = est.compose_with(_Other())
    assert isinstance(composed, ComposedXRayPrimitive)
    assert composed.name == "mdl_scorer_conditional+other"


# ── A1-archive integration regression ─────────────────────────────────────


def test_estimator_on_real_a1_archive_if_present():
    """Optional regression test against the canonical A1 archive.

    Skipped if submissions/a1/archive.zip is not present (CI / fresh checkout).
    """
    a1 = Path("submissions/a1/archive.zip")
    if not a1.exists():
        pytest.skip("A1 archive not present in this checkout")
    est = ScorerConditionalMDLEstimator()
    result = est.compute(a1)
    v = result.primitive_value
    assert v.total_archive_bytes == 178262
    # PR101-family substrate: brotli output is near-uniform-random; entropy
    # should be > 6.5 bits/byte on the measurable payload.
    assert v.aggregated_role_entropy_bits_per_byte > 6.5
    assert v.mdl_density > 0.8
    # The canonical A1 archive sha must match.
    assert result.archive_sha256 == (
        "87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5"
    )


# ── compare() probe-disambiguator API ─────────────────────────────────────


def test_estimator_compare_classifies_identical_within_class(tmp_path):
    archive_a = tmp_path / "a.zip"
    archive_b = tmp_path / "b.zip"
    _build_random_archive(archive_a)
    # b is a byte-identical copy of a.
    archive_b.write_bytes(archive_a.read_bytes())
    est = ScorerConditionalMDLEstimator()
    report = est.compare(archive_a, archive_b)
    assert isinstance(report, MDLDeltaReport)
    assert report.classification == "within_class"
    assert abs(report.density_delta) < 0.005


def test_estimator_compare_returns_report_with_shas(tmp_path):
    archive_a = tmp_path / "a.zip"
    archive_b = tmp_path / "b.zip"
    _build_random_archive(archive_a)
    _build_random_archive(archive_b)
    est = ScorerConditionalMDLEstimator()
    report = est.compare(archive_a, archive_b)
    assert report.archive_a_sha256 is not None
    assert report.archive_b_sha256 is not None
    assert report.density_a >= 0.0
    assert report.density_b >= 0.0


def test_estimator_compare_threshold_borderline(tmp_path):
    archive_a = tmp_path / "a.zip"
    archive_b = tmp_path / "b.zip"
    _build_random_archive(archive_a)
    archive_b.write_bytes(archive_a.read_bytes())
    est = ScorerConditionalMDLEstimator()
    # Force the within-class threshold to be tiny -> identical archive falls
    # into "within_class" by default (delta == 0).
    report = est.compare(
        archive_a, archive_b, within_class_threshold=0.0, across_class_threshold=0.01
    )
    # Identical archives have density_delta=0.0; depending on rounding may be
    # "within_class" or "borderline".
    assert report.classification in ("within_class", "borderline")


def test_estimator_compare_archive_a_sha_matches_compute(tmp_path):
    archive_a = tmp_path / "a.zip"
    archive_b = tmp_path / "b.zip"
    _build_random_archive(archive_a)
    _build_random_archive(archive_b)
    est = ScorerConditionalMDLEstimator()
    result_a = est.compute(archive_a)
    report = est.compare(archive_a, archive_b)
    assert report.archive_a_sha256 == result_a.archive_sha256


# ── Confidence band invariants ────────────────────────────────────────────


def test_confidence_band_lower_is_density(tmp_path):
    archive = tmp_path / "x.zip"
    _build_random_archive(archive)
    est = ScorerConditionalMDLEstimator()
    result = est.compute(archive)
    band = result.confidence_band
    assert band is not None
    assert band[0] == result.primitive_value.mdl_density


def test_confidence_band_capped_at_one(tmp_path):
    """A near-1.0 density still has an upper band <= 1.0."""
    archive = tmp_path / "x.zip"
    _build_random_archive(archive)
    est = ScorerConditionalMDLEstimator()
    result = est.compute(archive)
    band = result.confidence_band
    assert band is not None
    assert band[1] <= 1.0
