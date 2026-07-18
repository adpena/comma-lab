# SPDX-License-Identifier: MIT
"""Behavior and custody tests for the frozen-head power-diagram target."""

from __future__ import annotations

import hashlib
import json
import struct
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tac.boundary_math.power_diagram_witness import (
    PDW1_HEADER,
    TARGET_COMPARISON_VERDICT,
    PowerDiagramWitnessError,
    affine_head_to_power_diagram,
    compare_target_to_realizations,
    decode_pdw1,
    encode_pdw1,
    fit_power_diagram_from_labels,
    fit_power_diagram_from_paired_features,
    initialize_video_fed_target,
    is_co_maximum_tie,
    make_power_diagram_target,
    measure_f32_target_parity,
    observed_four_neighbour_adjacency,
    open_stored_npy_memmap,
    pair_tie_value,
    power_assign,
    project_channel_features,
    read_frozen_segmentation_head,
    read_safetensors_tensors,
)


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write the small F32 safetensors subset used by the head-only reader tests."""

    header: dict[str, object] = {}
    chunks: list[bytes] = []
    offset = 0
    for name, raw in tensors.items():
        array = np.asarray(raw, dtype="<f4")
        data = array.tobytes(order="C")
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(chunks))


def _write_raw_safetensors(path: Path, header_json: bytes, data: bytes) -> None:
    header_json += b" " * (-len(header_json) % 8)
    path.write_bytes(struct.pack("<Q", len(header_json)) + header_json + data)


def test_affine_head_and_f32_power_target_have_random_argmax_parity_with_error_receipt() -> None:
    rng = np.random.default_rng(20260718)
    weight = rng.normal(size=(5, 11))
    bias = rng.normal(size=5)
    features = rng.normal(size=(400, 11))
    head = affine_head_to_power_diagram(weight, bias)

    assert head.quotient_basis.shape == (11, 4)
    np.testing.assert_allclose(head.quotient_basis.T @ head.quotient_basis, np.eye(4), atol=1e-12)
    points = project_channel_features(features, head.quotient_basis)
    affine_logits = np.einsum("nd,kd->nk", features, weight, optimize=False) + bias
    affine_labels = np.argmax(affine_logits, axis=1)
    np.testing.assert_array_equal(power_assign(points, head.target), affine_labels)
    parity = measure_f32_target_parity(features, head)
    assert parity.exact_on_samples
    assert parity.mismatch_count == 0
    assert parity.max_pair_score_error > 0
    assert parity.boundary_exactness == "NO_GENERAL_VERDICT_WITHIN_F32_TIE_UNCERTAINTY"

    for i, j in ((0, 1), (1, 4), (2, 3)):
        edge = head.target.adjacency.index((i, j))
        pair_margin = np.einsum("nd,d->n", features, weight[j] - weight[i], optimize=False) + bias[j] - bias[i]
        tie_value = (
            np.einsum("nr,r->n", points, head.target.tie_normals[edge], optimize=False) + head.target.tie_offsets[edge]
        )
        np.testing.assert_allclose(tie_value, pair_margin, atol=2e-6, rtol=2e-6)


def test_f32_target_receipt_marks_constructed_near_tie_as_uncertain() -> None:
    weight = np.array([[0.1, 0.2], [-0.3, 0.7], [0.0, 0.0]], dtype=np.float64)
    bias = np.array([0.123456789, -0.234567891, -10.0], dtype=np.float64)
    difference = weight[1] - weight[0]
    x0 = 1.0
    x1 = -((difference[0] * x0) + bias[1] - bias[0]) / difference[1]
    feature = np.array([[x0, x1]], dtype=np.float64)
    affine = np.einsum("nd,kd->nk", feature, weight, optimize=False) + bias
    assert abs(float(affine[0, 1] - affine[0, 0])) < 1e-14
    parity = measure_f32_target_parity(feature, affine_head_to_power_diagram(weight, bias))
    assert parity.max_pair_score_error > 0
    assert parity.f32_tie_uncertain_count == 1
    assert parity.boundary_exactness == "NO_GENERAL_VERDICT_WITHIN_F32_TIE_UNCERTAINTY"


def test_basis_and_gauge_are_deterministic_and_gauge_does_not_move_cells() -> None:
    rng = np.random.default_rng(3)
    weight = rng.normal(size=(5, 9))
    bias = rng.normal(size=5)
    first = affine_head_to_power_diagram(weight, bias)
    second = affine_head_to_power_diagram(weight, bias)
    shifted = affine_head_to_power_diagram(weight, bias, common_gauge=first.common_gauge + 17.0)
    np.testing.assert_array_equal(first.quotient_basis, second.quotient_basis)
    np.testing.assert_allclose(shifted.target.weights, first.target.weights - 17.0, atol=2e-6)
    points = rng.normal(size=(100, first.target.rank))
    np.testing.assert_array_equal(power_assign(points, first.target), power_assign(points, shifted.target))


def test_default_basis_does_not_tolerance_erase_small_decision_direction() -> None:
    weight = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1e-11]])
    bias = np.zeros(3)
    feature = np.array([[0.0, 1e12]])
    head = affine_head_to_power_diagram(weight, bias)
    assert head.target.rank == 2
    parity = measure_f32_target_parity(feature, head)
    assert parity.exact_on_samples
    assert power_assign(project_channel_features(feature, head.quotient_basis), head.target)[0] == 2


def test_class_ids_reject_lossy_numeric_coercion() -> None:
    with pytest.raises(PowerDiagramWitnessError, match="lossless integers"):
        make_power_diagram_target(
            np.array([[-1.0], [1.0]]),
            np.zeros(2),
            class_ids=(0.5, 1.5),
        )
    with pytest.raises(PowerDiagramWitnessError, match="lossless integers"):
        make_power_diagram_target(
            np.array([[-1.0], [1.0]]),
            np.zeros(2),
            adjacency=((0.5, 1.0),),
        )


def test_formal_losing_equality_is_not_a_co_maximum_boundary() -> None:
    losing = make_power_diagram_target(
        np.array([[-1.0], [1.0], [0.0]]),
        np.array([0.0, 0.0, 2.0]),
    )
    origin = np.array([[0.0]])
    np.testing.assert_allclose(pair_tie_value(origin, losing, 0, 1), 0.0)
    assert not bool(is_co_maximum_tie(origin, losing, 0, 1)[0])

    active = make_power_diagram_target(
        np.array([[-1.0], [1.0], [0.0]]),
        np.array([0.0, 0.0, -2.0]),
    )
    assert bool(is_co_maximum_tie(origin, active, 0, 1)[0])
    with pytest.raises(PowerDiagramWitnessError, match="distinct"):
        is_co_maximum_tie(origin, active, 0, 0)


def test_paired_feature_inverse_is_deterministic_and_exact_when_separable() -> None:
    features = np.array([[-3.0, 0.2], [-2.0, -0.4], [2.0, 0.3], [3.0, -0.1]])
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    first = fit_power_diagram_from_paired_features(features, labels, n_classes=2, regularization=1e-5)
    second = fit_power_diagram_from_paired_features(features, labels, n_classes=2, regularization=1e-5)
    assert first.exact_on_samples
    assert first.sample_agreement == 1.0
    assert first.minimum_true_label_margin > 0
    assert first.residual_rms > 0  # ridge fit is honest about nonzero target residual
    np.testing.assert_array_equal(first.affine_weight, second.affine_weight)
    np.testing.assert_array_equal(first.affine_bias, second.affine_bias)
    np.testing.assert_array_equal(encode_pdw1(first.head.target), encode_pdw1(second.head.target))


def test_paired_feature_inverse_reports_nonexact_nonlinearly_separable_fixture() -> None:
    features = np.array([[-2.0], [-1.0], [0.0], [2.0]])
    labels = np.array([0, 1, 0, 0], dtype=np.int64)
    receipt = fit_power_diagram_from_paired_features(features, labels, n_classes=2, regularization=1e-3)
    assert not receipt.exact_on_samples
    assert receipt.sample_agreement < 1.0
    assert receipt.minimum_true_label_margin <= 0
    assert receipt.residual_rms > 0
    assert receipt.objective > 0


def test_labels_only_inverse_refuses_instead_of_manufacturing_features() -> None:
    with pytest.raises(PowerDiagramWitnessError, match="paired channel features are required"):
        fit_power_diagram_from_labels(np.array([0, 1, 0], dtype=np.int64))


def test_zip_stored_npy_member_is_a_real_read_only_memmap(tmp_path: Path) -> None:
    values = np.arange(60, dtype=np.uint8).reshape(3, 4, 5)
    cache = tmp_path / "cache.npz"
    np.savez(cache, lstars=values, unused=np.ones(4))
    with zipfile.ZipFile(cache) as archive:
        assert archive.getinfo("lstars.npy").compress_type == zipfile.ZIP_STORED
    mapped = open_stored_npy_memmap(cache, "lstars")
    assert isinstance(mapped, np.memmap)
    assert not mapped.flags.writeable
    np.testing.assert_array_equal(mapped, values)


def test_compressed_npy_member_refuses_zero_copy_claim(tmp_path: Path) -> None:
    cache = tmp_path / "compressed.npz"
    np.savez_compressed(cache, lstars=np.arange(8, dtype=np.uint8))
    with pytest.raises(PowerDiagramWitnessError, match="compressed"):
        open_stored_npy_memmap(cache, "lstars")


def test_npy_memmap_rejects_local_central_compression_method_disagreement(tmp_path: Path) -> None:
    cache = tmp_path / "method_mismatch.npz"
    np.savez(cache, lstars=np.arange(8, dtype=np.uint8))
    with zipfile.ZipFile(cache) as archive:
        local_header = archive.getinfo("lstars.npy").header_offset
    corrupted = bytearray(cache.read_bytes())
    corrupted[local_header + 8 : local_header + 10] = struct.pack("<H", zipfile.ZIP_DEFLATED)
    cache.write_bytes(corrupted)
    with pytest.raises(PowerDiagramWitnessError, match="local/central"):
        open_stored_npy_memmap(cache, "lstars")


def test_npy_memmap_rejects_local_central_size_disagreement(tmp_path: Path) -> None:
    cache = tmp_path / "size_mismatch.npz"
    np.savez(cache, lstars=np.arange(8, dtype=np.uint8))
    with zipfile.ZipFile(cache) as archive:
        local_header = archive.getinfo("lstars.npy").header_offset
    corrupted = bytearray(cache.read_bytes())
    corrupted[local_header + 22 : local_header + 26] = struct.pack("<I", 1)
    cache.write_bytes(corrupted)
    with pytest.raises(PowerDiagramWitnessError, match=r"CRC or sizes|ZIP64"):
        open_stored_npy_memmap(cache, "lstars")


def test_head_only_safetensors_reader_recovers_exact_tensors(tmp_path: Path) -> None:
    weight = np.arange(24, dtype=np.float32).reshape(3, 2, 2, 2) / 7.0
    bias = np.array([0.25, -0.5, 1.0], dtype=np.float32)
    path = tmp_path / "segnet.safetensors"
    _write_safetensors(
        path,
        {
            "unrelated": np.ones((100,), dtype=np.float32),
            "segmentation_head.0.weight": weight,
            "segmentation_head.0.bias": bias,
        },
    )
    loaded_weight, loaded_bias = read_frozen_segmentation_head(path)
    np.testing.assert_array_equal(loaded_weight, weight)
    np.testing.assert_array_equal(loaded_bias, bias)


@pytest.mark.parametrize("corruption", ["overlap", "duplicate", "trailing"])
def test_head_only_safetensors_reader_rejects_noncanonical_extents(tmp_path: Path, corruption: str) -> None:
    path = tmp_path / f"{corruption}.safetensors"
    if corruption == "overlap":
        header = (
            b'{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
            b'"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
        )
        data = b"\0" * 4
        expected = "overlapping"
    elif corruption == "duplicate":
        header = (
            b'{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
            b'"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
        )
        data = b"\0" * 4
        expected = "duplicate"
    else:
        header = b'{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
        data = b"\0" * 8
        expected = "trailing"
    _write_raw_safetensors(path, header, data)
    with pytest.raises(PowerDiagramWitnessError, match=expected):
        read_safetensors_tensors(path, ("a",))


def test_video_fed_init_uses_labels_for_statistics_and_head_for_sites(tmp_path: Path) -> None:
    labels = np.array(
        [
            [[0, 0, 1], [0, 2, 1]],
            [[2, 2, 1], [2, 0, 0]],
        ],
        dtype=np.uint8,
    )
    cache = tmp_path / "gt.npz"
    np.savez(cache, lstars=labels, margins=np.zeros_like(labels, dtype=np.float32))
    weight = np.array(
        [
            [[[1.0]], [[0.0]]],
            [[[0.0]], [[1.0]]],
            [[[-1.0]], [[-1.0]]],
        ],
        dtype=np.float32,
    )
    bias = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    head_path = tmp_path / "segnet.safetensors"
    _write_safetensors(
        head_path,
        {
            "segmentation_head.0.weight": weight,
            "segmentation_head.0.bias": bias,
        },
    )
    receipt = initialize_video_fed_target(cache, head_path)
    assert receipt.active_classes == (0, 1, 2)
    assert receipt.class_counts == (5, 3, 4)
    expected_edges = set()
    for frame in labels:
        expected_edges.update(observed_four_neighbour_adjacency(frame, n_classes=3))
    assert receipt.adjacency == tuple(sorted(expected_edges))
    assert receipt.selected_partition_sha256 == hashlib.sha256(labels.tobytes()).hexdigest()
    assert receipt.frozen_head_sha256 == hashlib.sha256(head_path.read_bytes()).hexdigest()
    exact_head = affine_head_to_power_diagram(weight, bias, adjacency=receipt.adjacency)
    np.testing.assert_array_equal(receipt.target.sites, exact_head.target.sites)
    np.testing.assert_array_equal(receipt.target.weights, exact_head.target.weights)


def test_pdw1_is_little_endian_canonical_and_decode_reencode_identical() -> None:
    target = make_power_diagram_target(
        np.array([[-1.25, 0.5], [0.75, -0.25], [2.0, 1.0]]),
        np.array([0.1, -0.2, 0.3]),
        adjacency=((1, 2), (0, 1)),
    )
    payload = encode_pdw1(target)
    assert payload[:4] == b"PDW1"
    assert PDW1_HEADER.unpack_from(payload)[1:] == (3, 2, 2)
    decoded = decode_pdw1(payload)
    assert decoded.adjacency == ((0, 1), (1, 2))
    assert encode_pdw1(decoded) == payload


def test_pdw1_rejects_direct_noninteger_class_id_dataclass_mutation() -> None:
    target = make_power_diagram_target(np.array([[-1.0], [1.0]]), np.zeros(2), adjacency=((0, 1),))
    corrupted = replace(target, class_ids=np.array([0.5, 1.5]))
    with pytest.raises(PowerDiagramWitnessError, match="canonical little-endian uint16"):
        encode_pdw1(corrupted)


@pytest.mark.parametrize("field", ["sites", "weights", "tie_normals", "tie_offsets"])
def test_pdw1_rejects_direct_float64_array_dataclass_mutation(field: str) -> None:
    target = make_power_diagram_target(np.array([[-1.0], [1.0]]), np.zeros(2), adjacency=((0, 1),))
    corrupted = replace(target, **{field: getattr(target, field).astype(np.float64)})
    with pytest.raises(PowerDiagramWitnessError, match="canonical little-endian float32"):
        encode_pdw1(corrupted)


@pytest.mark.parametrize("mutation", ["magic", "truncated", "trailer", "nan", "edge", "tie", "negative_zero"])
def test_pdw1_rejects_corruption_and_every_trailing_byte(mutation: str) -> None:
    target = make_power_diagram_target(
        np.array([[-1.0], [0.0], [1.0]]),
        np.array([0.0, 0.2, -0.1]),
        adjacency=((0, 1), (1, 2)),
    )
    payload = bytearray(encode_pdw1(target))
    if mutation == "magic":
        payload[0] ^= 0xFF
    elif mutation == "truncated":
        del payload[-1]
    elif mutation == "trailer":
        payload.extend(b"x")
    elif mutation == "nan":
        site_start = PDW1_HEADER.size + 2 * target.n_classes
        payload[site_start : site_start + 4] = struct.pack("<f", float("nan"))
    elif mutation == "edge":
        edge_start = PDW1_HEADER.size + 2 * target.n_classes + 4 * target.n_classes * target.rank + 4 * target.n_classes
        payload[edge_start : edge_start + 4] = struct.pack("<HH", 1, 0)
    else:
        normal_start = (
            PDW1_HEADER.size
            + 2 * target.n_classes
            + 4 * target.n_classes * target.rank
            + 4 * target.n_classes
            + 4 * len(target.adjacency)
        )
        old = struct.unpack("<f", payload[normal_start : normal_start + 4])[0]
        replacement = -0.0 if mutation == "negative_zero" else old + 0.5
        payload[normal_start : normal_start + 4] = struct.pack("<f", replacement)
    expected = "negative zero" if mutation == "negative_zero" else None
    with pytest.raises(PowerDiagramWitnessError, match=expected):
        decode_pdw1(payload)


def test_description_comparator_is_target_only_and_makes_no_score_or_k_claim(
    tmp_path: Path,
) -> None:
    target = make_power_diagram_target(np.array([[-1.0], [1.0]]), np.array([0.0, 0.0]), adjacency=((0, 1),))
    checkpoint = tmp_path / "coordinate_target.bin"
    packed = tmp_path / "packed_carrier.bin"
    checkpoint.write_bytes(b"checkpoint" * 9)
    packed.write_bytes(b"carrier" * 5)
    receipt = compare_target_to_realizations(target, (checkpoint, packed))
    assert receipt.verdict == TARGET_COMPARISON_VERDICT
    assert receipt.spatial_k_lower_bound == "NO_VERDICT"
    assert receipt.renderer_authority == "NOT_RUN_NO_AUTHORITY"
    assert not receipt.score_claim
    assert not receipt.archive_saving_claim
    assert [row.bytes for row in receipt.realization_files] == [checkpoint.stat().st_size, packed.stat().st_size]
    assert receipt.target_sha256 == hashlib.sha256(encode_pdw1(target)).hexdigest()
