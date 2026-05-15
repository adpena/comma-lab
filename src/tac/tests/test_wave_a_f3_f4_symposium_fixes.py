# SPDX-License-Identifier: MIT
"""WAVE-A F3 + F4 symposium_impls source-fix regression tests.

Per `feedback_wave_a_f3_f4_symposium_impls_fix_landed_20260515.md`:

* F3 (Catalog #268, codex bkrbqet3p HIGH): `compute_contest_theoretical_floor`
  in `tac.symposium_impls.blahut_arimoto_theoretical_floor` previously added
  `25.0 * r_combined` directly to `contest_score_floor` where `r_combined`
  is bits-per-unit; the contest formula at `upstream/evaluate.py:92` is
  `25 * archive_bytes / 37,545,489`. The fix adds:
    - `CONTEST_RATE_DENOM_BYTES = 37_545_489` module constant
    - `bits_per_unit_to_contest_rate_term(r, num_units)` canonical converter
    - per-axis conversion inside `compute_contest_theoretical_floor`
    - `theoretical_floor_units_calibrated=True` field on the result

* F4 (Catalog #269, codex bkrbqet3p MEDIUM):
  `tac.symposium_impls.mackay_conditional_entropy_a1_archive` previously
  claimed `H(A1_archive | scorer_state_dict)` while the partition was
  computed by hashing byte position only. The fix relabels the helpers
  and adds module-level disclosure constants:
    - `true_scorer_conditional_entropy_claim = False`
    - `position_partition_proxy = True`
    - `EVIDENCE_GRADE_POSITION_PARTITION_PROXY = "position-partition-proxy"`
    - new canonical `estimate_position_partition_bits(...)` helper
    - back-compat alias `estimate_scorer_conditional_bits(...)` retained
    - `A1ConditionalEntropyEstimate.evidence_grade` set to
      `"position-partition-proxy"`
"""
from __future__ import annotations

import math
import zipfile
from pathlib import Path

import pytest

from tac.symposium_impls.blahut_arimoto_theoretical_floor import (
    CONTEST_RATE_DENOM_BYTES,
    ContestTheoreticalFloor,
    bits_per_unit_to_contest_rate_term,
    compute_contest_theoretical_floor,
)
from tac.symposium_impls.mackay_conditional_entropy_a1_archive import (
    EVIDENCE_GRADE_POSITION_PARTITION_PROXY,
    A1ConditionalEntropyEstimate,
    compute_a1_conditional_entropy_estimate,
    estimate_position_partition_bits,
    estimate_scorer_conditional_bits,
    position_partition_proxy,
    true_scorer_conditional_entropy_claim,
)


# --------------------------------------------------------------------- F3 calibration --


def test_contest_rate_denom_bytes_matches_upstream_evaluate() -> None:
    """Per CLAUDE.md Contest compliance + ``upstream/evaluate.py:92``:
    the contest rate denominator is the sum of uncompressed video bytes.
    For ``upstream/videos/0.mkv`` decoded raw the canonical value is
    37,545,489 bytes."""
    assert CONTEST_RATE_DENOM_BYTES == 37_545_489


def test_bits_per_unit_to_contest_rate_term_a1_anchor_calibration() -> None:
    """A1 archive bytes = 178,262 (live A1 archive on disk 2026-05-15).
    Rate term = 25 * 178262 / 37545489 ≈ 0.118696.

    For an arbitrary archive size, the converter computes:
        archive_bytes_estimate = (r_bits_per_unit * num_units) / 8
        rate_term = 25 * archive_bytes_estimate / 37_545_489

    Round-trip test: choose num_units + r_bits_per_unit so that
    archive_bytes_estimate exactly equals A1's 178,262 bytes, then
    verify rate_term matches the closed-form expected value within 1e-5.
    """
    a1_archive_bytes = 178_262
    expected_rate_term = 25.0 * a1_archive_bytes / 37_545_489
    # Construct r,n such that (r * n) / 8 == a1_archive_bytes:
    num_units = 1_000_000
    r_bits_per_unit = (a1_archive_bytes * 8.0) / num_units
    actual = bits_per_unit_to_contest_rate_term(r_bits_per_unit, num_units)
    assert actual == pytest.approx(expected_rate_term, abs=1e-5)
    # Sanity: A1 rate term is close to ~0.118696 (CPU axis)
    assert 0.10 < actual < 0.14


def test_bits_per_unit_to_contest_rate_term_zero_rate_is_zero() -> None:
    assert bits_per_unit_to_contest_rate_term(0.0, 1_000_000) == 0.0


def test_bits_per_unit_to_contest_rate_term_negative_rate_raises() -> None:
    with pytest.raises(ValueError):
        bits_per_unit_to_contest_rate_term(-0.001, 1_000_000)


def test_bits_per_unit_to_contest_rate_term_zero_units_raises() -> None:
    with pytest.raises(ValueError):
        bits_per_unit_to_contest_rate_term(1.0, 0)


def test_bits_per_unit_to_contest_rate_term_negative_units_raises() -> None:
    with pytest.raises(ValueError):
        bits_per_unit_to_contest_rate_term(1.0, -100)


def test_contest_theoretical_floor_emits_calibration_flag_true() -> None:
    floor = compute_contest_theoretical_floor(
        target_d_seg=0.01, target_d_pose=0.001
    )
    assert isinstance(floor, ContestTheoreticalFloor)
    assert floor.theoretical_floor_units_calibrated is True


def test_contest_theoretical_floor_carries_rate_term_contest_normalized_field() -> None:
    floor = compute_contest_theoretical_floor(
        target_d_seg=0.05, target_d_pose=0.005
    )
    # Rate term must be in normalized-bytes domain (small positive),
    # NOT in bits-per-unit domain (could be large).
    assert 0.0 <= floor.rate_term_contest_normalized < 100.0
    # Score floor = 100 * D_seg + sqrt(10 * D_pose) + rate_term_contest_normalized
    expected_floor = (
        100.0 * 0.05
        + math.sqrt(10.0 * 0.005)
        + floor.rate_term_contest_normalized
    )
    assert floor.contest_score_floor == pytest.approx(expected_floor, abs=1e-9)


def test_contest_theoretical_floor_carries_num_units_metadata() -> None:
    floor = compute_contest_theoretical_floor(
        target_d_seg=0.01, target_d_pose=0.001
    )
    # Defaults: 600 frames × 384 × 512 pixels = 117,964,800 seg units
    #           600 frames × 6 pose components = 3,600 pose units
    assert floor.num_units_seg == 600 * 384 * 512
    assert floor.num_units_pose == 600 * 6


def test_contest_theoretical_floor_caller_overrides_num_units() -> None:
    floor_default = compute_contest_theoretical_floor(
        target_d_seg=0.01, target_d_pose=0.001
    )
    floor_smaller = compute_contest_theoretical_floor(
        target_d_seg=0.01, target_d_pose=0.001,
        num_units_seg=1_000, num_units_pose=10,
    )
    # Smaller num_units → smaller rate term contribution → smaller score floor
    assert floor_smaller.rate_term_contest_normalized < floor_default.rate_term_contest_normalized
    assert floor_smaller.contest_score_floor < floor_default.contest_score_floor


def test_contest_theoretical_floor_post_fix_is_orders_smaller_than_buggy_form() -> None:
    """Regression: pre-fix code added ``25.0 * r_combined`` where r_combined
    was bits-per-unit (often > 1). For realistic distortions this produced
    contest_score_floor values that were orders of magnitude wrong.
    Post-fix, the rate term is the contest-normalized ``25 * archive_bytes
    / 37,545,489`` which for any reasonable codec is < 1.0.

    Synthetic verification: at D_seg=0.01, D_pose=0.001 the per-pixel
    seg rate is 0.5*log2(1/0.01) ≈ 3.32 bits/pixel and the per-component
    pose rate is 6 * 0.5 * log2(1/(0.001/6)) ≈ 38.7 bits-per-frame-pair.
    Buggy: 25 * (3.32 + 38.7) ≈ 1050.
    Fixed: ~0.1 (dominated by rate, since rate-bytes are large here).
    """
    floor = compute_contest_theoretical_floor(
        target_d_seg=0.01, target_d_pose=0.001
    )
    # Distortion contribution: 100*0.01 + sqrt(10*0.001) = 1.0 + 0.1 = 1.1
    distortion_term = 100.0 * 0.01 + math.sqrt(10.0 * 0.001)
    assert floor.contest_score_floor < distortion_term + 200.0  # MUCH smaller than buggy ~1050


def test_catalog_268_gate_passes_post_fix() -> None:
    """Catalog #268 self-protection: post-fix the gate must accept the file."""
    from tac.preflight import check_theoretical_floor_rate_term_unit_calibrated

    v = check_theoretical_floor_rate_term_unit_calibrated(strict=False, verbose=False)
    assert v == [], f"Catalog #268 expected 0 violations, got: {v}"


# --------------------------------------------------------------------- F4 disclosure --


def test_module_level_disclosure_constants_present() -> None:
    """Per Catalog #269: the module must export the disclosure constants
    so autopilot / Pareto consumers can refuse to treat the estimate as
    a true scorer-conditional entropy."""
    assert true_scorer_conditional_entropy_claim is False
    assert position_partition_proxy is True
    assert EVIDENCE_GRADE_POSITION_PARTITION_PROXY == "position-partition-proxy"


def test_position_partition_bits_canonical_helper_matches_legacy_alias() -> None:
    """Back-compat: ``estimate_scorer_conditional_bits`` is preserved as
    an alias and must return identical results to the new canonical
    helper ``estimate_position_partition_bits``."""
    payload = bytes(range(256)) * 4
    canonical = estimate_position_partition_bits(payload)
    legacy = estimate_scorer_conditional_bits(payload)
    assert canonical == legacy


def test_compute_estimate_evidence_grade_is_position_partition_proxy(tmp_path: Path) -> None:
    """Per Catalog #269: the estimate's ``evidence_grade`` must disclose
    its position-partition-proxy nature, NOT claim
    ``theoretical-bound-prediction``."""
    archive = tmp_path / "syn.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("only.bin", b"\xaa\xbb\xcc" * 256)
    estimate = compute_a1_conditional_entropy_estimate(archive_path=archive)
    assert estimate.evidence_grade == "position-partition-proxy"
    assert estimate.true_scorer_conditional_entropy_claim is False
    assert estimate.position_partition_proxy is True
    assert estimate.score_claim is False


def test_compute_estimate_notes_carries_position_partition_disclosure(tmp_path: Path) -> None:
    """Per Catalog #269: the ``notes`` field must explicitly disclose the
    position-partition nature so downstream Markdown readers (operator
    review, autopilot summary) cannot mistake this for a true
    scorer-conditional entropy."""
    archive = tmp_path / "syn.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("only.bin", b"\xaa\xbb\xcc" * 256)
    estimate = compute_a1_conditional_entropy_estimate(archive_path=archive)
    assert "position-partition-proxy" in estimate.notes
    assert "true_scorer_conditional_entropy_claim=False" in estimate.notes
    assert "Catalog #269" in estimate.notes


def test_compute_estimate_section_field_carries_position_partition_alias(tmp_path: Path) -> None:
    """Per Catalog #269: each section row must expose both the back-compat
    ``scorer_prior_context_bits_per_byte`` field AND the canonical
    ``position_partition_context_bits_per_byte`` field (same value)."""
    archive = tmp_path / "syn.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("only.bin", b"\xff" * 200)
    estimate = compute_a1_conditional_entropy_estimate(archive_path=archive)
    for section in estimate.sections:
        assert section.scorer_prior_context_bits_per_byte == pytest.approx(
            section.position_partition_context_bits_per_byte, abs=1e-12
        )


def test_compute_estimate_aggregate_position_partition_matches_legacy(tmp_path: Path) -> None:
    """Per Catalog #269: ``aggregate_position_partition_context_bits`` is
    the canonical name; ``aggregate_scorer_prior_context_bits`` is the
    back-compat alias (same value)."""
    archive = tmp_path / "syn.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a.bin", bytes(range(256)) * 2)
        zf.writestr("b.bin", b"\x00" * 512)
    estimate = compute_a1_conditional_entropy_estimate(archive_path=archive)
    assert estimate.aggregate_position_partition_context_bits == pytest.approx(
        estimate.aggregate_scorer_prior_context_bits, abs=1e-9
    )


def test_catalog_269_gate_passes_post_fix() -> None:
    """Catalog #269 self-protection: post-fix the gate must accept the file."""
    from tac.preflight import check_scorer_conditional_entropy_actually_uses_scorer_state

    v = check_scorer_conditional_entropy_actually_uses_scorer_state(
        strict=False, verbose=False
    )
    assert v == [], f"Catalog #269 expected 0 violations, got: {v}"


# --------------------------------------------------------------------- synthetic negative regression --


def test_synthetic_buggy_blahut_arimoto_fixture_would_be_refused(tmp_path: Path) -> None:
    """Synthetic regression: rebuild the F3 buggy pattern in a tmp file
    and verify Catalog #268's gate refuses it. This pins the gate's
    behavior so a future regression to the buggy form is caught."""
    from tac.preflight import check_theoretical_floor_rate_term_unit_calibrated

    target_dir = tmp_path / "src" / "tac" / "symposium_impls"
    target_dir.mkdir(parents=True)
    buggy = target_dir / "blahut_arimoto_theoretical_floor.py"
    buggy.write_text(
        "# Synthetic buggy stub — bits-per-unit added directly without "
        "the converter or advisory tag\n"
        "def compute_contest_theoretical_floor():\n"
        "    r_combined = 3.3\n"
        "    return 100.0 * 0.01 + 25.0 * r_combined\n",
        encoding="utf-8",
    )
    v = check_theoretical_floor_rate_term_unit_calibrated(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1


def test_synthetic_buggy_mackay_fixture_would_be_refused(tmp_path: Path) -> None:
    """Synthetic regression: rebuild the F4 buggy pattern in a tmp file
    and verify Catalog #269's gate refuses it. This pins the gate's
    behavior so a future regression to the overstated form is caught."""
    from tac.preflight import check_scorer_conditional_entropy_actually_uses_scorer_state

    target_dir = tmp_path / "src" / "tac" / "symposium_impls"
    target_dir.mkdir(parents=True)
    buggy = target_dir / "mackay_conditional_entropy_a1_archive.py"
    # No disclosure token, no feature binding — claims scorer-conditional
    # entropy via prose alone.
    buggy.write_text(
        '"""Claims H(X | scorer_state_dict) via position partition only."""\n'
        "def estimate_scorer_conditional_bits(payload):\n"
        "    return len(payload) * 8\n",
        encoding="utf-8",
    )
    v = check_scorer_conditional_entropy_actually_uses_scorer_state(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) == 1


def test_synthetic_mackay_fixture_with_disclosure_is_accepted(tmp_path: Path) -> None:
    """Negative-of-the-negative: the disclosure token alone (without
    feature binding) is sufficient acceptance per the gate's acceptance
    contract."""
    from tac.preflight import check_scorer_conditional_entropy_actually_uses_scorer_state

    target_dir = tmp_path / "src" / "tac" / "symposium_impls"
    target_dir.mkdir(parents=True)
    accepted = target_dir / "mackay_conditional_entropy_a1_archive.py"
    accepted.write_text(
        '"""Position-partition proxy disclosed."""\n'
        "# true_scorer_conditional_entropy_claim=false\n"
        "def estimate_position_partition_bits(payload):\n"
        "    return len(payload) * 8\n",
        encoding="utf-8",
    )
    v = check_scorer_conditional_entropy_actually_uses_scorer_state(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_live_repo_catalog_268_269_both_at_zero() -> None:
    """Live-repo regression guard: this test pins the symposium_impls
    files at 0 violations across both gates so a future regression that
    re-introduces either bug class is caught at test time."""
    from tac.preflight import (
        check_scorer_conditional_entropy_actually_uses_scorer_state,
        check_theoretical_floor_rate_term_unit_calibrated,
    )

    v268 = check_theoretical_floor_rate_term_unit_calibrated(strict=False, verbose=False)
    v269 = check_scorer_conditional_entropy_actually_uses_scorer_state(
        strict=False, verbose=False
    )
    assert v268 == [], f"Catalog #268 regression: {v268}"
    assert v269 == [], f"Catalog #269 regression: {v269}"
