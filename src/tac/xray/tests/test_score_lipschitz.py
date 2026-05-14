"""Tests for F4: tac.xray.score_lipschitz."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.xray.base import XRayPrimitive
from tac.xray.score_lipschitz import (
    ScoreLipschitzReport,
    ScoreVsArchiveLipschitz,
)


def test_protocol():
    assert isinstance(ScoreVsArchiveLipschitz(), XRayPrimitive)


def test_name():
    assert ScoreVsArchiveLipschitz().name == "score_lipschitz"


def test_hooks():
    h = ScoreVsArchiveLipschitz().wire_in_hooks
    assert "pareto_constraint" in h
    assert "bit_allocator" in h
    assert "probe_disambiguator" in h


def test_report_rejects_zero_bits():
    with pytest.raises(ValueError, match="archive_n_bits"):
        ScoreLipschitzReport(
            archive_n_bits=0,
            n_flip_samples=0,
            max_delta_score=0.0,
            mean_delta_score=0.0,
            median_delta_score=0.0,
            empirical_lipschitz_per_bit=0.0,
            lagrangian_lipschitz_per_bit=1e-8,
            flip_score_distribution=(),
        )


def test_report_rejects_negative_max_delta():
    with pytest.raises(ValueError, match="max_delta_score"):
        ScoreLipschitzReport(
            archive_n_bits=1024,
            n_flip_samples=0,
            max_delta_score=-1.0,
            mean_delta_score=0.0,
            median_delta_score=0.0,
            empirical_lipschitz_per_bit=0.0,
            lagrangian_lipschitz_per_bit=1e-8,
            flip_score_distribution=(),
        )


def test_compute_lagrangian_only_no_score_fn(tmp_path):
    """Without score_fn, the primitive returns the closed-form Lagrangian
    Lipschitz only."""
    archive = tmp_path / "x.bin"
    archive.write_bytes(b"x" * 256)
    result = ScoreVsArchiveLipschitz().compute(target=archive)
    report = result.primitive_value
    assert report.n_flip_samples == 0
    assert report.lagrangian_lipschitz_per_bit > 0
    # Closed-form: 25 / (37,545,489 * 8) ≈ 8.32e-8.
    assert report.lagrangian_lipschitz_per_bit == pytest.approx(
        25.0 / (37_545_489 * 8), rel=1e-12
    )


def test_compute_with_score_fn_runs_flips(tmp_path):
    archive = tmp_path / "x.bin"
    archive.write_bytes(b"\x00" * 32)

    def score_fn(b: bytes) -> float:
        # Score = popcount(b) / total_bits; flipping one bit changes score by 1/N.
        n_bits = 8 * len(b)
        ones = sum(bin(byte).count("1") for byte in b)
        return ones / n_bits

    result = ScoreVsArchiveLipschitz().compute(
        target=archive, score_fn=score_fn, n_samples=10, seed=0xDEAD
    )
    report = result.primitive_value
    assert report.n_flip_samples == 10
    # Each flip changes score by exactly 1 / (8 * 32) = 1/256.
    expected_per_flip = 1.0 / (8 * 32)
    assert report.max_delta_score == pytest.approx(expected_per_flip)
    assert report.mean_delta_score == pytest.approx(expected_per_flip)


def test_compute_refuses_empty_archive(tmp_path):
    archive = tmp_path / "empty.bin"
    archive.write_bytes(b"")
    with pytest.raises(ValueError, match="archive is empty"):
        ScoreVsArchiveLipschitz().compute(target=archive)


def test_compute_refuses_missing_archive(tmp_path):
    missing = tmp_path / "missing.bin"
    with pytest.raises(ValueError, match="does not exist"):
        ScoreVsArchiveLipschitz().compute(target=missing)


def test_compute_accepts_bytes_input():
    """target can be raw bytes (no archive file required)."""
    raw = b"\x00" * 16

    def score_fn(b: bytes) -> float:
        return sum(b) / 256.0

    result = ScoreVsArchiveLipschitz().compute(
        target=raw, score_fn=score_fn, n_samples=4
    )
    assert result.archive_or_video_path is None
    assert result.archive_sha256 is None
    assert result.primitive_value.n_flip_samples == 4


def test_compute_default_n_samples_64_when_score_fn_provided(tmp_path):
    archive = tmp_path / "x.bin"
    archive.write_bytes(b"x" * 32)

    def score_fn(b: bytes) -> float:
        return 0.0

    result = ScoreVsArchiveLipschitz().compute(
        target=archive, score_fn=score_fn, n_samples=0
    )
    assert result.primitive_value.n_flip_samples == 64


def test_compute_explicit_flip_positions(tmp_path):
    archive = tmp_path / "x.bin"
    archive.write_bytes(b"\x00" * 8)
    flips_called = []

    def score_fn(b: bytes) -> float:
        flips_called.append(b)
        return float(b[0])

    result = ScoreVsArchiveLipschitz().compute(
        target=archive,
        score_fn=score_fn,
        flip_bit_positions=[0, 1, 2],
    )
    # base call + 3 flips = 4.
    assert len(flips_called) == 4
    assert result.primitive_value.n_flip_samples == 3


def test_compute_confidence_band_uses_lagrangian_lower():
    raw = b"\x00" * 16

    def score_fn(b: bytes) -> float:
        return 0.0  # Always returns 0; max_delta=0; empirical < lagrangian.

    result = ScoreVsArchiveLipschitz().compute(
        target=raw, score_fn=score_fn, n_samples=4
    )
    band = result.confidence_band
    assert band is not None
    lo, hi = band
    # Lagrangian must be in band.
    assert lo == pytest.approx(
        result.primitive_value.lagrangian_lipschitz_per_bit
    )


def test_compute_evidence_grade_first_principles_without_samples(tmp_path):
    archive = tmp_path / "x.bin"
    archive.write_bytes(b"x" * 16)
    result = ScoreVsArchiveLipschitz().compute(target=archive)
    assert result.evidence_grade == "first-principles-bound"


def test_compute_evidence_grade_math_with_samples(tmp_path):
    archive = tmp_path / "x.bin"
    archive.write_bytes(b"\x00" * 16)
    result = ScoreVsArchiveLipschitz().compute(
        target=archive,
        score_fn=lambda b: 0.0,
        n_samples=2,
    )
    assert result.evidence_grade == "mathematical-derivation"


def test_compute_flip_distribution_capped_at_1024():
    raw = b"\x00" * 256

    def score_fn(b: bytes) -> float:
        return float(b[0]) / 256.0

    result = ScoreVsArchiveLipschitz().compute(
        target=raw, score_fn=score_fn, n_samples=2048
    )
    # Distribution capped at 1024 entries.
    assert len(result.primitive_value.flip_score_distribution) == 1024


def test_compute_records_archive_sha_for_path_input(tmp_path):
    archive = tmp_path / "x.bin"
    archive.write_bytes(b"abc")
    result = ScoreVsArchiveLipschitz().compute(target=archive)
    assert result.archive_sha256 is not None
    assert len(result.archive_sha256) == 64


def test_compute_record_n_bits_correct(tmp_path):
    archive = tmp_path / "x.bin"
    archive.write_bytes(b"abc")  # 3 bytes = 24 bits.
    result = ScoreVsArchiveLipschitz().compute(target=archive)
    assert result.primitive_value.archive_n_bits == 24


def test_compute_median_handles_even_count(tmp_path):
    archive = tmp_path / "x.bin"
    archive.write_bytes(b"\x00" * 8)

    # score_fn returns popcount(byte_0) -> 0 baseline; flipping any
    # bit increments by 1 (one new bit set).
    # All flips have identical |delta_S|, so median == max == mean.
    def score_fn(b: bytes) -> float:
        return float(sum(bin(byte).count("1") for byte in b))

    result = ScoreVsArchiveLipschitz().compute(
        target=archive,
        score_fn=score_fn,
        flip_bit_positions=[0, 1, 2, 3],
    )
    assert (
        result.primitive_value.median_delta_score
        == result.primitive_value.max_delta_score
    )


def test_metadata_includes_score_fn_provided_flag():
    raw = b"\x00" * 4
    result_no_fn = ScoreVsArchiveLipschitz().compute(target=raw)
    result_with_fn = ScoreVsArchiveLipschitz().compute(
        target=raw, score_fn=lambda b: 0.0
    )
    assert result_no_fn.metadata["score_fn_provided"] is False
    assert result_with_fn.metadata["score_fn_provided"] is True


def test_compose_with_returns_composed():
    from tac.xray.base import ComposedXRayPrimitive

    p = ScoreVsArchiveLipschitz()

    class _Other:
        name = "other"
        wire_in_hooks = ("sensitivity_map",)

        def compute(self, target, **kw):
            from tac.xray.base import XRayPrimitiveResult

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

    composed = p.compose_with(_Other())
    assert isinstance(composed, ComposedXRayPrimitive)
