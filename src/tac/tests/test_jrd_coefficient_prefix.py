# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac.packet_compiler.jrd_coefficient_prefix import (
    MAX_INT8_PREFIX_PLANES,
    PREFIX_FAMILIES,
    LaplaceHistogramFit,
    PrefixMeasurement,
    coefficient_sections,
    component_safe,
    fit_laplace_histogram,
    generate_prefix_chain,
    quantize_prefix,
    read_section,
    replace_section,
    select_best_byte_safe,
    select_last_safe_plane,
)


def test_uniform_prefix_is_literal_twos_complement_prefix() -> None:
    q = np.array([-128, -127, -3, -2, -1, 0, 1, 2, 3, 126, 127], dtype=np.int8)
    actual = quantize_prefix(q, bits_removed=1, family="uniform")
    expected = np.array([-128, -128, -4, -2, -2, 0, 0, 2, 2, 126, 126], dtype=np.int8)
    np.testing.assert_array_equal(actual, expected)


def test_dead_zone_known_vector_with_exact_histogram_fit() -> None:
    q = np.array([-128, -127, -3, -2, -1, 0, 1, 2, 3, 126, 127], dtype=np.int8)
    fit = fit_laplace_histogram(q)
    actual = quantize_prefix(
        q, bits_removed=1, family="laplace_dead_zone", fit=fit
    )
    expected = np.array([-128, -126, 0, 0, 0, 0, 0, 0, 0, 126, 126], dtype=np.int8)
    np.testing.assert_array_equal(actual, expected)


def test_dead_zone_refuses_forged_fit_even_when_count_matches() -> None:
    q = np.array([-2, -1, 0, 1, 2], dtype=np.int8)
    forged = LaplaceHistogramFit(q.size, 1, 0.0, 0.0, "0" * 64)
    with pytest.raises(ValueError, match="exact coefficient histogram"):
        quantize_prefix(q, bits_removed=1, family="laplace_dead_zone", fit=forged)


@pytest.mark.parametrize("family", PREFIX_FAMILIES)
def test_prefix_chain_is_deterministic_nested_identity_to_completion(family: str) -> None:
    q = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
    first = generate_prefix_chain(q, family=family)  # type: ignore[arg-type]
    second = generate_prefix_chain(q, family=family)  # type: ignore[arg-type]
    assert len(first) == MAX_INT8_PREFIX_PLANES + 1
    np.testing.assert_array_equal(first[0], q)
    np.testing.assert_array_equal(first[-1], np.zeros_like(q))
    for lhs, rhs in zip(first, second, strict=True):
        np.testing.assert_array_equal(lhs, rhs)


def test_laplace_fit_is_histogram_content_addressed_and_exact_mle() -> None:
    q = np.array([-2, -1, 0, 0, 1, 4], dtype=np.int8)
    fit = fit_laplace_histogram(q)
    assert fit.count == 6
    assert fit.zero_count == 2
    assert fit.mean_abs == pytest.approx(8.0 / 6.0)
    assert fit.scale_b == fit.mean_abs
    assert len(fit.histogram_sha256) == 64
    assert fit == fit_laplace_histogram(q.copy())


@pytest.mark.parametrize(
    ("values", "error"),
    [
        (np.array([], dtype=np.int8), ValueError),
        (np.array([1], dtype=np.int16), TypeError),
        ([1, 2], TypeError),
    ],
)
def test_prefix_input_refuses_empty_or_non_int8(values: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        quantize_prefix(values, bits_removed=1, family="uniform")  # type: ignore[arg-type]


def test_prefix_refuses_invalid_plane_family_and_mismatched_fit() -> None:
    q = np.array([1, 2], dtype=np.int8)
    with pytest.raises(ValueError, match="bits_removed"):
        quantize_prefix(q, bits_removed=9, family="uniform")
    with pytest.raises(ValueError, match="unknown prefix family"):
        quantize_prefix(q, bits_removed=1, family="bad")  # type: ignore[arg-type]
    fit = LaplaceHistogramFit(1, 0, 1.0, 1.0, "0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        quantize_prefix(q, bits_removed=1, family="laplace_dead_zone", fit=fit)


def test_section_layout_rederives_offsets_and_preserves_untouched_bytes() -> None:
    manifest = {
        "base_param_order": ["a", "b"],
        "base_shapes": {"a": [2], "b": [2, 2]},
        "code_shape": [2, 2],
    }
    base = bytes(np.array([1, 2, 3, 4, 5, 6], dtype=np.int8))
    code = bytes(np.array([7, 8, 9, 10], dtype=np.int8))
    sections = coefficient_sections(manifest, base_raw_len=len(base), code_raw_len=len(code))
    assert [(s.name, s.stream, s.offset, s.count) for s in sections] == [
        ("a", "base", 0, 2),
        ("b", "base", 2, 4),
        ("code", "code", 0, 4),
    ]
    np.testing.assert_array_equal(read_section(base, code, sections[1]), [[3, 4], [5, 6]])
    new_base, new_code = replace_section(
        base, code, sections[1], np.zeros((2, 2), dtype=np.int8)
    )
    assert new_base[:2] == base[:2]
    assert new_base[2:] == b"\x00" * 4
    assert new_code == code


def test_section_layout_refuses_manifest_stream_mismatch() -> None:
    manifest = {
        "base_param_order": ["a"],
        "base_shapes": {"a": [2]},
        "code_shape": [2],
    }
    with pytest.raises(ValueError, match="stream has"):
        coefficient_sections(manifest, base_raw_len=3, code_raw_len=2)
    with pytest.raises(ValueError, match="stream has"):
        coefficient_sections(manifest, base_raw_len=2, code_raw_len=3)


def test_pair_scoped_code_section_preserves_unscored_rows() -> None:
    manifest = {
        "n_pairs": 3,
        "base_param_order": ["a"],
        "base_shapes": {"a": [2]},
        "code_shape": [6, 2],
    }
    base = bytes(np.array([1, 2], dtype=np.int8))
    code = bytes(np.arange(12, dtype=np.int8))
    sections = coefficient_sections(
        manifest,
        base_raw_len=len(base),
        code_raw_len=len(code),
        eval_pairs=1,
    )
    code_section = sections[-1]
    assert (code_section.name, code_section.shape, code_section.count) == (
        "code_scored_pair_prefix",
        (2, 2),
        4,
    )
    _base_new, code_new = replace_section(
        base,
        code,
        code_section,
        np.zeros((2, 2), dtype=np.int8),
    )
    assert code_new[:4] == b"\x00" * 4
    assert code_new[4:] == code[4:]


def _row(bits: int, archive: int, d_seg: float, d_pose: float) -> PrefixMeasurement:
    return PrefixMeasurement("a", "uniform", bits, archive, d_seg, d_pose)


def test_exact_selectors_separate_last_safe_from_best_bytes() -> None:
    baseline = _row(0, 100, 0.10, 0.20)
    rows = [
        _row(1, 90, 0.10, 0.20),
        _row(2, 95, 0.09, 0.20),
        _row(3, 80, 0.11, 0.20),
    ]
    assert select_last_safe_plane(
        rows, baseline, seg_tolerance=0.0, pose_tolerance=0.0
    ) == rows[1]
    assert select_best_byte_safe(
        rows, baseline, seg_tolerance=0.0, pose_tolerance=0.0
    ) == rows[0]


def test_component_guard_is_noncompensating_and_fail_closed() -> None:
    baseline = _row(0, 100, 0.10, 0.20)
    assert component_safe(
        _row(1, 90, 0.09, 0.20), baseline, seg_tolerance=0.0, pose_tolerance=0.0
    )
    assert not component_safe(
        _row(1, 90, 0.11, 0.0), baseline, seg_tolerance=0.0, pose_tolerance=0.0
    )
    with pytest.raises(ValueError, match="finite"):
        component_safe(
            _row(1, 90, float("nan"), 0.2),
            baseline,
            seg_tolerance=0.0,
            pose_tolerance=0.0,
        )


def test_selectors_refuse_mixed_identity_and_negative_units() -> None:
    baseline = _row(0, 100, 0.1, 0.2)
    with pytest.raises(ValueError, match="match the baseline"):
        select_last_safe_plane(
            [PrefixMeasurement("b", "uniform", 1, 90, 0.1, 0.2)],
            baseline,
            seg_tolerance=0.0,
            pose_tolerance=0.0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        select_best_byte_safe(
            [_row(1, -1, 0.1, 0.2)],
            baseline,
            seg_tolerance=0.0,
            pose_tolerance=0.0,
        )
