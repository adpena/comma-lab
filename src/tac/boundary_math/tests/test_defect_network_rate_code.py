# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from tac.boundary_math.defect_network_rate_code import (
    DEFECT_TUBE_MAGIC,
    DefectTubeRateCodeError,
    _component_indices,
    decode_defect_tube_to_phase_section,
    encode_defect_tube_recode,
)
from tac.boundary_math.phase_residual_carrier import (
    PhaseCarrierConfig,
    compute_tie_field_from_margins,
    decode_phase_carrier,
    encode_phase_carrier,
)


def _fixture(seed: int = 7):
    rng = np.random.default_rng(seed)
    p_count, h, w = 5, 18, 24
    labels = np.zeros((p_count, h, w), dtype=np.int64)
    margins = np.empty((p_count, h, w), dtype=np.float32)
    for p in range(p_count):
        edge = 7 + (p % 3)
        labels[p, :, edge:] = 1
        labels[p, 3:8, 15:21] = 2
        x = np.arange(w)[None, :]
        margins[p] = np.abs(x - edge).astype(np.float32) * rng.uniform(0.15, 1.1, (h, w))
    cfg = PhaseCarrierConfig(q_step=1.0 / 32.0)
    ties, masks, class_maps = [], [], []
    for p in range(p_count):
        tie, mask, class_map = compute_tie_field_from_margins(labels[p], margins[p], cfg)
        ties.append(tie)
        masks.append(mask)
        class_maps.append(class_map)
    xi = rng.normal(0.0, 0.01, size=(p_count, 6))
    incumbent, _ = encode_phase_carrier(ties, masks, class_maps, xi, cfg)
    return incumbent, masks, class_maps


def test_component_oracle_is_eight_connected_not_four_connected() -> None:
    diagonal = np.asarray([[True, False], [False, True]])

    components = _component_indices(diagonal)

    assert len(components) == 1
    assert np.array_equal(components[0], np.asarray([0, 3]))


def test_component_oracle_orders_components_and_pixels_by_raster_index() -> None:
    mask = np.zeros((4, 6), dtype=bool)
    mask[0, 3] = True
    mask[1, 4] = True
    mask[3, 0] = True

    components = _component_indices(mask)

    assert len(components) == 2
    assert np.array_equal(components[0], np.asarray([3, 10]))
    assert np.array_equal(components[1], np.asarray([18]))


@pytest.mark.parametrize("z2", [False, True])
def test_recode_is_exact_at_residual_and_phase_field_boundaries(z2: bool) -> None:
    incumbent, masks, class_maps = _fixture()
    candidate, report = encode_defect_tube_recode(
        incumbent, masks, class_maps, z2_orientation_quotient=z2
    )
    rebuilt, residuals = decode_defect_tube_to_phase_section(candidate, masks, class_maps)

    assert candidate.startswith(DEFECT_TUBE_MAGIC)
    assert residuals.size == report.residual_count
    assert report.exact_residual_roundtrip is True
    assert report.exact_phase_field_roundtrip is True
    assert report.z2_orientation_quotient is z2
    base_fields = decode_phase_carrier(incumbent, masks, class_maps)
    new_fields = decode_phase_carrier(rebuilt, masks, class_maps)
    assert all(np.array_equal(a, b) for a, b in zip(base_fields, new_fields, strict=True))


def test_report_accounts_for_every_component_and_stream_byte() -> None:
    incumbent, masks, class_maps = _fixture()
    candidate, report = encode_defect_tube_recode(incumbent, masks, class_maps)

    assert report.component_count > 0
    assert report.residual_count >= report.component_count
    assert report.component_base_bytes > 0
    assert report.intracomponent_delta_bytes >= 0
    assert report.group_label_bytes == 0
    assert report.component_stream_bytes == (
        report.component_base_bytes + report.intracomponent_delta_bytes
    )
    assert report.component_stream_delta_bytes == (
        report.component_stream_bytes - report.incumbent_residual_stream_bytes
    )
    assert report.bytes_saved == len(incumbent) - len(candidate)
    for value in (
        report.constant_component_fraction,
        report.constant_pixel_fraction,
        report.singleton_component_fraction,
        report.sign_consistent_component_fraction,
        report.canonical_sequence_orbit_reuse_fraction,
    ):
        assert 0.0 <= value <= 1.0


def test_z2_group_labels_are_counted_and_lossless() -> None:
    incumbent, masks, class_maps = _fixture()
    candidate, report = encode_defect_tube_recode(
        incumbent, masks, class_maps, z2_orientation_quotient=True
    )

    assert report.group_label_bytes == (report.component_count + 7) // 8
    rebuilt, _ = decode_defect_tube_to_phase_section(candidate, masks, class_maps)
    base_fields = decode_phase_carrier(incumbent, masks, class_maps)
    new_fields = decode_phase_carrier(rebuilt, masks, class_maps)
    assert all(np.array_equal(a, b) for a, b in zip(base_fields, new_fields, strict=True))


def test_mask_change_fails_closed_instead_of_silently_reindexing() -> None:
    incumbent, masks, class_maps = _fixture()
    bad_masks = [m.copy() for m in masks]
    bad_masks[0][0, 0] = ~bad_masks[0][0, 0]

    with pytest.raises(DefectTubeRateCodeError):
        encode_defect_tube_recode(incumbent, bad_masks, class_maps)


def test_topology_preserving_class_map_change_fails_geometry_commitment() -> None:
    incumbent, masks, class_maps = _fixture()
    candidate, _ = encode_defect_tube_recode(incumbent, masks, class_maps)
    swapped_maps = [m.copy() for m in class_maps]
    for class_map in swapped_maps:
        zero = class_map == 0
        one = class_map == 1
        class_map[zero] = 1
        class_map[one] = 0

    with pytest.raises(DefectTubeRateCodeError, match="geometry commitment"):
        decode_defect_tube_to_phase_section(candidate, masks, swapped_maps)


def test_class_traversal_order_is_bound_by_geometry_commitment() -> None:
    incumbent, masks, class_maps = _fixture()
    candidate, _ = encode_defect_tube_recode(incumbent, masks, class_maps)
    off = len(DEFECT_TUBE_MAGIC)
    (header_len,) = struct.unpack("<I", candidate[off : off + 4])
    header_start = off + 4
    header = json.loads(candidate[header_start : header_start + header_len])
    header["classes"] = [1, 0, 2]
    mutated_header = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    assert len(mutated_header) == header_len
    corrupt = b"".join(
        (
            candidate[:header_start],
            mutated_header,
            candidate[header_start + header_len :],
        )
    )

    with pytest.raises(DefectTubeRateCodeError, match="geometry commitment"):
        decode_defect_tube_to_phase_section(corrupt, masks, class_maps)


def test_extra_geometry_frame_fails_closed() -> None:
    incumbent, masks, class_maps = _fixture()
    candidate, _ = encode_defect_tube_recode(incumbent, masks, class_maps)

    with pytest.raises(DefectTubeRateCodeError, match="geometry frame count"):
        decode_defect_tube_to_phase_section(
            candidate,
            [*masks, masks[-1]],
            [*class_maps, class_maps[-1]],
        )


def test_declared_unused_label_bytes_fail_when_z2_is_disabled() -> None:
    incumbent, masks, class_maps = _fixture()
    candidate, _ = encode_defect_tube_recode(incumbent, masks, class_maps)
    off = len(DEFECT_TUBE_MAGIC)
    (header_len,) = struct.unpack("<I", candidate[off : off + 4])
    header_start = off + 4
    header = json.loads(candidate[header_start : header_start + header_len])
    lengths_off = header_start + header_len + int(header["n_frames"]) * 12
    label_len, base_len, delta_len = struct.unpack(
        "<III", candidate[lengths_off : lengths_off + 12]
    )
    assert label_len == 0
    payload_off = lengths_off + 12
    corrupt = b"".join(
        (
            candidate[:lengths_off],
            struct.pack("<III", 1, base_len, delta_len),
            b"\x00",
            candidate[payload_off:],
        )
    )

    with pytest.raises(DefectTubeRateCodeError, match="must not contain"):
        decode_defect_tube_to_phase_section(corrupt, masks, class_maps)


def test_z2_group_label_length_mismatch_fails_closed() -> None:
    incumbent, masks, class_maps = _fixture()
    candidate, _ = encode_defect_tube_recode(
        incumbent,
        masks,
        class_maps,
        z2_orientation_quotient=True,
    )
    off = len(DEFECT_TUBE_MAGIC)
    (header_len,) = struct.unpack("<I", candidate[off : off + 4])
    header_start = off + 4
    header = json.loads(candidate[header_start : header_start + header_len])
    lengths_off = header_start + header_len + int(header["n_frames"]) * 12
    label_len, base_len, delta_len = struct.unpack(
        "<III", candidate[lengths_off : lengths_off + 12]
    )
    payload_off = lengths_off + 12
    corrupt = b"".join(
        (
            candidate[:lengths_off],
            struct.pack("<III", label_len + 1, base_len, delta_len),
            candidate[payload_off : payload_off + label_len],
            b"\x00",
            candidate[payload_off + label_len :],
        )
    )

    with pytest.raises(DefectTubeRateCodeError, match="bitstream length mismatch"):
        decode_defect_tube_to_phase_section(corrupt, masks, class_maps)


def test_corrupt_candidate_trailing_bytes_fail_closed() -> None:
    incumbent, masks, class_maps = _fixture()
    candidate, _ = encode_defect_tube_recode(incumbent, masks, class_maps)

    with pytest.raises(DefectTubeRateCodeError, match="trailing bytes"):
        decode_defect_tube_to_phase_section(candidate + b"x", masks, class_maps)
