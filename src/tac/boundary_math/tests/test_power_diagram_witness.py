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
    PDW2_HEADER,
    PDW2_MARGIN_MODE,
    PDW2_PARTITION_MODE,
    TARGET_COMPARISON_VERDICT,
    PowerDiagramWitnessError,
    affine_head_to_power_diagram,
    compare_target_to_realizations,
    decode_pdw1,
    decode_pdw2,
    encode_pdw1,
    encode_pdw2,
    fit_power_diagram_from_labels,
    fit_power_diagram_from_paired_features,
    gauge_fixed_assign_f32,
    gauge_fixed_pair_tie_value_f32,
    gauge_fixed_scores_f32,
    initialize_video_fed_target,
    is_co_maximum_tie,
    make_gauge_fixed_affine_target,
    make_power_diagram_target,
    measure_f32_target_parity,
    observed_four_neighbour_adjacency,
    open_stored_npy_memmap,
    pair_tie_value,
    pdw1_to_pdw2,
    power_assign,
    project_channel_features,
    read_frozen_segmentation_head,
    read_safetensors_tensors,
    realized_margin_and_gradient,
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


def _five_class_nine_edge_target(*, partition_only: bool = False):
    weight = np.array(
        [
            [-1.25, 0.5, 0.25, -0.75],
            [0.75, -0.25, 1.0, 0.5],
            [2.0, 1.0, -0.5, 0.25],
            [-0.5, -1.5, 0.75, 1.25],
            [1.25, 0.25, -1.0, -0.5],
        ],
        dtype=np.float32,
    )
    bias = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float32)
    adjacency = tuple((i, j) for i in range(5) for j in range(i + 1, 5) if (i, j) != (3, 4))
    return make_gauge_fixed_affine_target(
        weight,
        bias,
        adjacency=adjacency,
        partition_only=partition_only,
    )


def test_pdw2_measures_138_margin_bytes_and_134_partition_bytes() -> None:
    margin = _five_class_nine_edge_target()
    partition = _five_class_nine_edge_target(partition_only=True)
    margin_bytes = encode_pdw2(margin)
    partition_bytes = encode_pdw2(partition)
    assert len(margin_bytes) == 138
    assert len(partition_bytes) == 134
    assert decode_pdw2(margin_bytes).mode == PDW2_MARGIN_MODE
    assert decode_pdw2(partition_bytes).mode == PDW2_PARTITION_MODE
    assert encode_pdw2(decode_pdw2(margin_bytes)) == margin_bytes
    assert encode_pdw2(decode_pdw2(partition_bytes)) == partition_bytes
    assert decode_pdw2(margin_bytes).verdict == TARGET_COMPARISON_VERDICT
    assert decode_pdw2(partition_bytes).verdict == TARGET_COMPARISON_VERDICT


def test_pdw2_reproduces_pdw1_ordinary_random_and_small_direction_parity() -> None:
    rng = np.random.default_rng(20260718)
    weight = rng.normal(size=(5, 11))
    bias = rng.normal(size=5)
    features = rng.normal(size=(400, 11))
    head = affine_head_to_power_diagram(weight, bias)
    points = project_channel_features(features, head.quotient_basis)
    pdw2 = decode_pdw2(encode_pdw2(pdw1_to_pdw2(head.target)))
    np.testing.assert_array_equal(gauge_fixed_assign_f32(points, pdw2), power_assign(points, head.target))

    small_head = affine_head_to_power_diagram(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1e-11]]),
        np.zeros(3),
    )
    small_feature = np.array([[0.0, 1e12]])
    small_point = project_channel_features(small_feature, small_head.quotient_basis)
    small_pdw2 = decode_pdw2(encode_pdw2(pdw1_to_pdw2(small_head.target)))
    np.testing.assert_array_equal(
        gauge_fixed_assign_f32(small_point, small_pdw2),
        power_assign(small_point, small_head.target),
    )


def test_pdw2_gauge_composition_is_a_fixture_assertion() -> None:
    weight = np.array(
        [[-1.0, 0.5], [0.25, 1.0], [1.5, -0.75]],
        dtype=np.float32,
    )
    bias = np.array([-0.5, 0.25, 1.0], dtype=np.float32)
    common_affine = np.array([2.0, -4.0], dtype=np.float32)
    common_bias = np.float32(8.0)
    base = make_gauge_fixed_affine_target(weight, bias)
    composed = make_gauge_fixed_affine_target(weight + common_affine, bias + common_bias)
    assert encode_pdw2(base) == encode_pdw2(composed)
    np.testing.assert_array_equal(base.tie_normals, composed.tie_normals)
    np.testing.assert_array_equal(base.tie_offsets, composed.tie_offsets)


def test_pdw2_exact_frame195_native_f32_tie_after_parseback() -> None:
    pdw1_hex = (
        "50445731050004000a0000000000010002000300040076e0343fd7abd23ee1c8aa3e5467673e"
        "2895a2bfd7abd23ee1c8aa3e5467673eaea5363ebd0247bfe1c8aa3e5467673ef8033f3e385a"
        "88bd62db74bf5467673ec37d4b3e8845ac3c021f35bd546767bfc14195be1f02543fad2942be"
        "21b22abe2ba938be000001000000020000000300000004000100020001000300010004000200"
        "0300020004000300040063057dc00000000000000000000000000a3787bf542c18c000000000"
        "00000000781f85bf65c274bfe91f25c000000000050182bf7ee747bfc16c41bf94a010c0de69"
        "3940542c18c00000000000000000a7753a4065c274bfe91f25c000000000e0043c407ee747bf"
        "c16c41bf94a010c0a0e4853c76f7b53fe91f25c000000000a8c0263de964cc3fc16c41bf94a0"
        "10c0b09cc73c9a6b333e7289e93f94a010c0407ab53bd980093e8c9a4abdf622aa3d07d5033e"
        "d44961bd52cb9e3d7c273cbe78bd51bd1eb8073e"
    )
    point = np.array([[0.0014007954, 5.7772665, 1.7700754, 3.2382076]], dtype=np.float32)
    pdw1 = decode_pdw1(bytes.fromhex(pdw1_hex))
    assert hashlib.sha256(bytes.fromhex(pdw1_hex)).hexdigest() == (
        "f2178e123c12cc062812a93f046174b7d5716df2edba0545733cad1c09795758"
    )
    for partition_only in (False, True):
        pdw2 = decode_pdw2(encode_pdw2(pdw1_to_pdw2(pdw1, partition_only=partition_only)))
        naive_reference_zero = np.add(
            np.sum(
                np.multiply(point, pdw2.relative_coefficients[0, :-1], dtype=np.float32),
                axis=-1,
                dtype=np.float32,
            ),
            pdw2.relative_coefficients[0, -1],
            dtype=np.float32,
        )
        assert float(naive_reference_zero[0]) > 0.0  # the naive ordering flips to class 1
        scores = gauge_fixed_scores_f32(point, pdw2)
        assert scores[0, 0] == scores[0, 1]
        assert int(gauge_fixed_assign_f32(point, pdw2)[0]) == 0
        assert float(gauge_fixed_pair_tie_value_f32(point, pdw2, 0, 1)[0]) == 0.0


@pytest.mark.parametrize(
    "mutation",
    ["magic", "truncated", "trailer", "nan", "edge", "negative_zero", "partition_pivot"],
)
def test_pdw2_rejects_malformed_and_every_trailing_byte(mutation: str) -> None:
    target = _five_class_nine_edge_target()
    if mutation == "partition_pivot":
        partition = _five_class_nine_edge_target(partition_only=True)
        payload = bytearray(encode_pdw2(partition))
        magic, k, rank, metadata = PDW2_HEADER.unpack_from(payload)
        bad_metadata = (metadata & 0x8000FFFF) | (((k - 1) * (rank + 1)) << 16)
        PDW2_HEADER.pack_into(payload, 0, magic, k, rank, bad_metadata)
    else:
        payload = bytearray(encode_pdw2(target))
        coefficient_start = PDW2_HEADER.size + 2 * target.n_classes
        edge_start = coefficient_start + 4 * target.relative_coefficients.size
        if mutation == "magic":
            payload[0] ^= 0xFF
        elif mutation == "truncated":
            del payload[-1]
        elif mutation == "trailer":
            payload.extend(b"x")
        elif mutation == "nan":
            payload[coefficient_start : coefficient_start + 4] = struct.pack("<f", float("nan"))
        elif mutation == "edge":
            payload[edge_start : edge_start + 4] = struct.pack("<HH", 1, 0)
        else:
            payload[coefficient_start : coefficient_start + 4] = struct.pack("<f", -0.0)
    expected = "negative zero" if mutation == "negative_zero" else None
    with pytest.raises(PowerDiagramWitnessError, match=expected):
        decode_pdw2(payload)


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


# --------------------------------------------------------------------------- #
# realized_margin_and_gradient -- the exact top-2 margin primitive (#539 -> #888)
# --------------------------------------------------------------------------- #


def _reference_geometric_margin(
    features: np.ndarray, weight: np.ndarray, bias: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent top-2 geometric margin straight from the affine head."""

    logits = np.einsum("nd,kd->nk", features, weight, optimize=False) + bias
    order = np.argsort(-logits, axis=-1, kind="stable")
    top, runner = order[:, 0], order[:, 1]
    gap = (
        np.take_along_axis(logits, top[:, None], axis=-1)[:, 0]
        - np.take_along_axis(logits, runner[:, None], axis=-1)[:, 0]
    )
    return gap / np.linalg.norm(weight[top] - weight[runner], axis=-1), top, runner


def test_realized_margin_top_class_matches_power_assign_and_margin_is_non_negative() -> None:
    rng = np.random.default_rng(20260802)
    weight, bias = rng.normal(size=(5, 11)), rng.normal(size=5)
    head = affine_head_to_power_diagram(weight, bias)
    points = project_channel_features(rng.normal(size=(2000, 11)), head.quotient_basis)

    realized = realized_margin_and_gradient(points, head.target)

    np.testing.assert_array_equal(realized.top_class, power_assign(points, head.target))
    assert np.all(realized.margin >= 0.0)
    assert np.all(realized.top_class != realized.runner_up_class)
    np.testing.assert_allclose(np.linalg.norm(realized.gradient, axis=-1), 1.0, atol=1e-12)


def test_realized_margin_equals_geometric_margin_of_the_affine_head() -> None:
    rng = np.random.default_rng(20260802)
    weight, bias = rng.normal(size=(5, 11)), rng.normal(size=5)
    features = rng.normal(size=(4000, 11))
    head = affine_head_to_power_diagram(weight, bias)

    realized = realized_margin_and_gradient(
        project_channel_features(features, head.quotient_basis), head.target
    )
    expected, top, runner = _reference_geometric_margin(features, weight, bias)

    # The target is serialized float32, so the identity holds to f32 epsilon in
    # ABSOLUTE terms (relative error is unbounded only where the gap -> 0).
    np.testing.assert_allclose(realized.margin, expected, atol=5e-6, rtol=0.0)
    np.testing.assert_array_equal(realized.top_class, top)
    np.testing.assert_array_equal(realized.runner_up_class, runner)


def test_realized_margin_gradient_matches_finite_differences_on_stable_sites() -> None:
    rng = np.random.default_rng(20260802)
    weight, bias = rng.normal(size=(5, 7)), rng.normal(size=5)
    head = affine_head_to_power_diagram(weight, bias)
    points = project_channel_features(rng.normal(size=(1500, 7)), head.quotient_basis)
    realized = realized_margin_and_gradient(points, head.target)

    step, rank = 1e-5, head.target.rank
    finite = np.empty_like(realized.gradient)
    stable = np.ones(points.shape[0], dtype=bool)
    for axis in range(rank):
        offset = np.zeros(rank)
        offset[axis] = step
        plus = realized_margin_and_gradient(points + offset, head.target)
        minus = realized_margin_and_gradient(points - offset, head.target)
        finite[:, axis] = (plus.margin - minus.margin) / (2 * step)
        stable &= (plus.top_class == realized.top_class) & (minus.top_class == realized.top_class)
        stable &= (plus.runner_up_class == realized.runner_up_class)
        stable &= (minus.runner_up_class == realized.runner_up_class)

    assert stable.sum() > 0.9 * points.shape[0]
    relative = np.linalg.norm(finite - realized.gradient, axis=-1)
    # Pre-registered falsifier leg: relative gradient error <= 1e-3.
    assert relative[stable].max() < 1e-3
    # Positive control: a deliberately wrong gradient must FAIL the same bar.
    wrong = np.linalg.norm(finite - realized.gradient[:, ::-1], axis=-1)
    assert wrong[stable].max() > 1e-3


def test_realized_margin_flags_the_uncovered_codim_two_junction_stratum() -> None:
    # Three unit sites co-dominating the origin; coordinates are chosen exactly
    # representable in float32 so the constructed tie survives serialization.
    sites = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    target = make_power_diagram_target(sites, np.array([0.0, 0.0, 0.0, -9.0]))

    junction_point = realized_margin_and_gradient(np.zeros((1, 2)), target, junction_tolerance=1e-9)
    assert bool(junction_point.junction[0])

    generic = realized_margin_and_gradient(np.array([[3.0, 0.05]]), target, junction_tolerance=1e-9)
    assert not bool(generic.junction[0])
    # A tolerance wide enough to swallow every gap flags everything.
    assert bool(realized_margin_and_gradient(np.array([[3.0, 0.05]]), target, junction_tolerance=1e6).junction[0])


def test_junction_tolerance_must_clear_the_float32_target_noise_floor() -> None:
    """A constructed exact tie is broken by ~1e-8 once the target is float32.

    Consumers therefore may not use ``junction_tolerance=0`` (or a margin floor)
    below the serialization floor -- at that scale they would be deciding on
    float32 noise rather than on geometry.
    """

    root_three_over_two = 0.8660254037844386  # not representable in float32
    sites = np.array([[1.0, 0.0], [-0.5, root_three_over_two], [-0.5, -root_three_over_two], [0.0, 0.0]])
    target = make_power_diagram_target(sites, np.array([0.0, 0.0, 0.0, -9.0]))
    origin = np.zeros((1, 2))

    assert not bool(realized_margin_and_gradient(origin, target, junction_tolerance=0.0).junction[0])
    assert bool(realized_margin_and_gradient(origin, target, junction_tolerance=1e-6).junction[0])


def test_realized_margin_preserves_batch_shape_and_rejects_bad_input() -> None:
    rng = np.random.default_rng(7)
    target = make_power_diagram_target(rng.normal(size=(3, 2)), rng.normal(size=3))

    batched = realized_margin_and_gradient(rng.normal(size=(4, 6, 2)), target)
    assert batched.margin.shape == (4, 6)
    assert batched.gradient.shape == (4, 6, 2)
    assert batched.junction.shape == (4, 6)
    assert batched.junction_tolerance == 0.0

    with pytest.raises(PowerDiagramWitnessError, match="final dimension"):
        realized_margin_and_gradient(np.zeros((5, 3)), target)
    with pytest.raises(PowerDiagramWitnessError, match="final dimension"):
        realized_margin_and_gradient(np.full((5, 2), np.nan), target)
    with pytest.raises(PowerDiagramWitnessError, match="junction_tolerance"):
        realized_margin_and_gradient(np.zeros((5, 2)), target, junction_tolerance=-1.0)
    with pytest.raises(PowerDiagramWitnessError, match="junction_tolerance"):
        realized_margin_and_gradient(np.zeros((5, 2)), target, junction_tolerance=np.inf)


def test_realized_margin_fails_closed_when_finite_points_overflow_the_score() -> None:
    """A finite point can still drive the score to +/-inf, making the margin NaN.

    The primitive exists to carry an objective, so it refuses rather than
    seeding a descent with a silent NaN.
    """

    target = make_power_diagram_target(
        np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]), np.zeros(3)
    )
    assert np.isfinite(np.full((1, 2), 1e308)).all()  # the INPUT is finite
    with pytest.raises(PowerDiagramWitnessError, match="non-finite"):
        realized_margin_and_gradient(np.full((1, 2), 1e308), target)


def test_realized_margin_is_invariant_to_head_rescaling_unlike_cross_entropy() -> None:
    """argmax ignores a common positive rescale of the head; so must the margin.

    This is the structural reason the primitive exists: the describe loop's CE
    leg is scale-sensitive where the scored argmax quantity is not.
    """

    rng = np.random.default_rng(20260802)
    weight, bias = rng.normal(size=(5, 11)), rng.normal(size=5)
    features = rng.normal(size=(3000, 11))
    base = affine_head_to_power_diagram(weight, bias)
    base_margin = realized_margin_and_gradient(
        project_channel_features(features, base.quotient_basis), base.target
    )

    def mean_cross_entropy(scale: float) -> float:
        logits = scale * (np.einsum("nd,kd->nk", features, weight, optimize=False) + bias)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_sum = np.log(np.exp(shifted).sum(axis=-1))
        target_logit = np.take_along_axis(shifted, base_margin.top_class[:, None], axis=-1)[:, 0]
        return float(np.mean(log_sum - target_logit))

    baseline_ce = mean_cross_entropy(1.0)
    for scale in (0.5, 2.0, 10.0):
        scaled = affine_head_to_power_diagram(weight * scale, bias * scale)
        rescaled = realized_margin_and_gradient(
            project_channel_features(features, scaled.quotient_basis), scaled.target
        )
        np.testing.assert_array_equal(rescaled.top_class, base_margin.top_class)
        np.testing.assert_allclose(rescaled.margin, base_margin.margin, atol=5e-6, rtol=0.0)
        # ...while the cross-entropy leg the describe loop currently minimizes
        # moves substantially under the very same score-preserving rescale.
        assert abs(mean_cross_entropy(scale) / baseline_ce - 1.0) > 0.25
