# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.symposium_impls.mackay_conditional_entropy_a1_archive`."""
from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import pytest

from tac.symposium_impls.mackay_conditional_entropy_a1_archive import (
    A1_ARCHIVE_PATH,
    A1ConditionalEntropyEstimate,
    A1SectionEntropy,
    compute_a1_conditional_entropy_estimate,
    estimate_brotli_context_bits,
    estimate_scorer_conditional_bits,
    estimate_zero_context_bits,
    load_cached_a1_conditional_entropy_estimate,
    save_a1_conditional_entropy_estimate,
    shannon_entropy_bits,
    update_from_anchor,
)


def _build_synthetic_archive(tmp_path: Path, members: dict[str, bytes]) -> Path:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return archive


# --------------------------------------------------------------------------- Shannon entropy unit tests


def test_shannon_entropy_uniform_eight_classes_is_three_bits() -> None:
    """H(uniform N=8) = log2(8) = 3.0 bits exactly."""
    counts = [10, 10, 10, 10, 10, 10, 10, 10]
    assert shannon_entropy_bits(counts) == pytest.approx(3.0, abs=1e-12)


def test_shannon_entropy_uniform_two_classes_is_one_bit() -> None:
    assert shannon_entropy_bits([5, 5]) == pytest.approx(1.0, abs=1e-12)


def test_shannon_entropy_deterministic_is_zero() -> None:
    """Single-class distribution has zero entropy."""
    assert shannon_entropy_bits([100]) == pytest.approx(0.0, abs=1e-12)


def test_shannon_entropy_empty_iterable_is_zero() -> None:
    assert shannon_entropy_bits([]) == 0.0


def test_shannon_entropy_skips_zero_counts() -> None:
    """Zero-count symbols must not contribute to entropy (would be 0·log2(0) = NaN)."""
    counts_with_zeros = [10, 0, 10, 0, 0]
    counts_without = [10, 10]
    assert shannon_entropy_bits(counts_with_zeros) == pytest.approx(
        shannon_entropy_bits(counts_without), abs=1e-12
    )


def test_shannon_entropy_binary_symmetric_p_quarter() -> None:
    """H(p=1/4) = -1/4 log 1/4 - 3/4 log 3/4 ≈ 0.811."""
    expected = -0.25 * math.log2(0.25) - 0.75 * math.log2(0.75)
    assert shannon_entropy_bits([1, 3]) == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------- zero-context entropy


def test_estimate_zero_context_bits_empty_payload_is_zero() -> None:
    assert estimate_zero_context_bits(b"") == 0.0


def test_estimate_zero_context_bits_constant_payload_is_zero() -> None:
    """All-same bytes have zero entropy regardless of length."""
    assert estimate_zero_context_bits(b"\x00" * 1000) == pytest.approx(0.0, abs=1e-12)


def test_estimate_zero_context_bits_uniform_full_alphabet_is_eight_bits_per_byte() -> None:
    """Each of 256 symbols equally frequent → exactly 8 bits / byte."""
    payload = bytes(range(256)) * 8  # 2048 bytes; uniform over the byte alphabet
    bits = estimate_zero_context_bits(payload)
    assert bits / len(payload) == pytest.approx(8.0, abs=1e-12)


# --------------------------------------------------------------------------- scorer-conditional entropy


def test_scorer_conditional_bits_empty_payload_is_zero() -> None:
    assert estimate_scorer_conditional_bits(b"") == 0.0


def test_scorer_conditional_bits_constant_payload_is_zero() -> None:
    """Constant payload has zero per-bucket entropy."""
    assert estimate_scorer_conditional_bits(b"x" * 5000) == pytest.approx(0.0, abs=1e-12)


def test_scorer_conditional_bits_invalid_buckets_raises() -> None:
    with pytest.raises(ValueError):
        estimate_scorer_conditional_bits(b"abc", n_buckets=0)


def test_scorer_conditional_bits_at_most_zero_context_bits() -> None:
    """Conditioning reduces entropy: H(X|Y) <= H(X) per Cover & Thomas Theorem 2.6.5."""
    payload = bytes(range(256)) * 4
    zero_bits = estimate_zero_context_bits(payload)
    cond_bits = estimate_scorer_conditional_bits(payload)
    assert cond_bits <= zero_bits + 1e-9


# --------------------------------------------------------------------------- brotli-context entropy


def test_brotli_context_bits_empty_payload_is_zero() -> None:
    assert estimate_brotli_context_bits(b"") == 0.0


def test_brotli_context_bits_uses_supplied_compressed_size_when_provided() -> None:
    """Bits = compressed_bytes × 8 when supplied directly."""
    bits = estimate_brotli_context_bits(b"unused", brotli_size_bytes=10)
    assert bits == pytest.approx(80.0, abs=1e-12)


# --------------------------------------------------------------------------- end-to-end on synthetic archive


def test_compute_estimate_synthetic_archive_yields_expected_aggregates(tmp_path: Path) -> None:
    members = {
        "constant.bin": b"\x00" * 1024,
        "uniform.bin": bytes(range(256)) * 4,
    }
    archive_path = _build_synthetic_archive(tmp_path, members)
    estimate = compute_a1_conditional_entropy_estimate(archive_path=archive_path)
    section_names = {s.section_name for s in estimate.sections}
    assert section_names == {"constant.bin", "uniform.bin"}
    constant_section = next(s for s in estimate.sections if s.section_name == "constant.bin")
    assert constant_section.zero_context_bits_per_byte == pytest.approx(0.0, abs=1e-12)
    uniform_section = next(s for s in estimate.sections if s.section_name == "uniform.bin")
    assert uniform_section.zero_context_bits_per_byte == pytest.approx(8.0, abs=1e-12)
    assert estimate.aggregate_zero_context_bits >= estimate.aggregate_scorer_prior_context_bits - 1e-6
    assert estimate.score_claim is False
    assert estimate.evidence_grade == "theoretical-bound-prediction"
    # Slack non-negative iff brotli >= scorer-prior; not guaranteed for synthetic, but typed correctly.
    assert isinstance(estimate.slack_brotli_minus_scorer_prior_fraction, float)


def test_compute_estimate_missing_archive_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_a1_conditional_entropy_estimate(archive_path=tmp_path / "does-not-exist.zip")


# --------------------------------------------------------------------------- save / load round-trip


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    members = {"only.bin": b"\xfe" * 256}
    archive = _build_synthetic_archive(tmp_path, members)
    state_path = tmp_path / "state.json"
    estimate = compute_a1_conditional_entropy_estimate(archive_path=archive)
    save_a1_conditional_entropy_estimate(estimate, state_path=state_path)
    assert state_path.is_file()
    parsed = json.loads(state_path.read_text())
    assert parsed["archive_path"] == str(archive)
    loaded = load_cached_a1_conditional_entropy_estimate(state_path=state_path)
    assert loaded is not None
    assert loaded.archive_sha256 == estimate.archive_sha256
    assert loaded.aggregate_zero_context_bits == pytest.approx(
        estimate.aggregate_zero_context_bits, abs=1e-9
    )


def test_load_returns_none_when_state_absent(tmp_path: Path) -> None:
    assert load_cached_a1_conditional_entropy_estimate(state_path=tmp_path / "absent.json") is None


# --------------------------------------------------------------------------- continual-learning hook


def test_update_from_anchor_skips_when_archive_sha_mismatches(tmp_path: Path) -> None:
    members = {"x.bin": b"abc"}
    archive = _build_synthetic_archive(tmp_path, members)
    state_path = tmp_path / "state.json"
    fake_anchor = {"archive_sha256": "0" * 64}
    result = update_from_anchor(
        fake_anchor, state_path=state_path, archive_path=archive
    )
    assert result is None
    assert not state_path.exists()


def test_update_from_anchor_recomputes_on_match(tmp_path: Path) -> None:
    members = {"x.bin": b"abcdef"}
    archive = _build_synthetic_archive(tmp_path, members)
    state_path = tmp_path / "state.json"
    actual_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    result = update_from_anchor(
        {"archive_sha256": actual_sha}, state_path=state_path, archive_path=archive
    )
    assert result is not None
    assert state_path.is_file()


def test_update_from_anchor_skips_when_archive_missing(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    result = update_from_anchor(
        {"archive_sha256": "abc"}, state_path=state_path, archive_path=tmp_path / "missing.zip"
    )
    assert result is None


# --------------------------------------------------------------------------- live A1 archive integration (skipped when absent)


@pytest.mark.skipif(not A1_ARCHIVE_PATH.is_file(), reason="A1 archive not committed in this checkout")
def test_live_a1_archive_compute_returns_well_formed_estimate() -> None:
    estimate = compute_a1_conditional_entropy_estimate()
    assert estimate.archive_size_bytes > 0
    assert estimate.aggregate_zero_context_bits > 0
    assert estimate.aggregate_brotli_context_bits > 0
    assert estimate.aggregate_scorer_prior_context_bits > 0
    # Conditioning reduces or holds entropy (Cover & Thomas Theorem 2.6.5).
    assert estimate.aggregate_scorer_prior_context_bits <= estimate.aggregate_zero_context_bits + 1e-6
    assert -1.0 <= estimate.slack_brotli_minus_scorer_prior_fraction <= 1.0
