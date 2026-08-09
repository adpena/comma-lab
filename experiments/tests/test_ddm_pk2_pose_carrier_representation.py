from __future__ import annotations

import importlib.util
import io
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "ddm_pk2_pose_carrier_representation.py"
SPEC = importlib.util.spec_from_file_location("ddm_pk2_repr", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pk2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pk2
SPEC.loader.exec_module(pk2)


def test_absolute_int12_delta_round_trip_is_not_a_stub() -> None:
    rng = np.random.default_rng(20260809)
    codes = rng.integers(-2047, 2048, size=(pk2.N, pk2.DIM), dtype=np.int32)
    encoded = pk2.encode_absolute_codes(codes)
    assert encoded.shape == codes.shape
    assert not np.array_equal(encoded, codes)
    assert np.array_equal(pk2.decode_absolute_codes(encoded), codes)
    changed = codes.copy()
    changed[17, 3] += 1
    assert not np.array_equal(
        pk2.encode_absolute_codes(changed),
        encoded,
    )


def test_basis_quantizer_performs_real_low_precision_work() -> None:
    value = np.linspace(-2.0, 2.0, np.prod(pk2.BASIS_SHAPE), dtype=np.float32)
    value = value.reshape(pk2.BASIS_SHAPE)
    restored, codes, scales = pk2.quantize_basis(value, 4, 99.5)
    assert codes.min() >= -7 and codes.max() <= 7
    assert scales.shape == (pk2.DIM,)
    assert restored.shape == pk2.BASIS_SHAPE
    assert not np.array_equal(restored, value)


def _baseline_arrays():
    codec, _ = pk2.setup_imports()
    bundle = pk2.extract_bundle(pk2.DEFAULT_ARCHIVE)
    decoded = codec.decode_compact_carrier(
        bundle.carrier,
        basis_count=math.prod(pk2.BASIS_SHAPE),
        frames=pk2.N,
        dimensions=pk2.DIM,
    )
    basis_scales, basis_codes_flat, coefficient_scales, encoded = decoded
    basis_codes = basis_codes_flat.reshape(pk2.BASIS_SHAPE)
    coefficient_codes = pk2.decode_absolute_codes(encoded)
    basis = basis_codes.astype(np.float32) * basis_scales[:, None, None, None]
    coefficients = coefficient_codes.astype(np.float32) * coefficient_scales[None]
    return (
        codec,
        bundle,
        basis_scales,
        basis_codes,
        coefficient_scales,
        coefficient_codes,
        basis,
        coefficients,
    )


def test_exact_predictor_and_low_rank_residual_packets_reconstruct_declared_arrays() -> None:
    (_, _, _, _, coefficient_scales, coefficient_codes, _, coefficients) = _baseline_arrays()
    predictor, predictor_declared = pk2.encode_predictor_coefficient_component(
        coefficient_codes,
        coefficient_scales,
        pk2.PRED_AR,
        order=4,
    )
    assert np.array_equal(
        pk2.decode_predictor_coefficient_component(predictor),
        predictor_declared,
    )
    assert np.array_equal(predictor_declared, coefficients)

    low_rank, low_rank_declared = pk2.encode_low_rank_coefficient_component(
        coefficient_codes,
        coefficient_scales,
        rank=4,
        factor_bits=6,
        residual_step=1,
    )
    assert np.array_equal(
        pk2.decode_low_rank_coefficient_component(low_rank),
        low_rank_declared,
    )
    assert np.array_equal(low_rank_declared, coefficients)


def test_reversible_haar_and_overlay_dispatch_are_real() -> None:
    (codec, bundle, basis_scales, basis_codes, _, _, basis, coefficients) = _baseline_arrays()
    _, coefficient_component = pk2.split_cpr1_components(bundle.carrier, codec)
    haar_component = pk2.encode_haar_basis_component(basis_codes, basis_scales)
    overlay = pk2.encode_overlay(
        pk2.BASIS_HAAR,
        haar_component,
        pk2.COEFF_CPR1,
        coefficient_component,
    )
    decoded_basis, decoded_coefficients = pk2.decode_overlay_carrier(overlay, codec)
    assert np.array_equal(decoded_basis, basis)
    assert np.array_equal(decoded_coefficients, coefficients)
    mutated_codes = basis_codes.copy()
    mutated_codes[0, 0, 0, 0] += 1
    mutated = pk2.encode_overlay(
        pk2.BASIS_HAAR,
        pk2.encode_haar_basis_component(mutated_codes, basis_scales),
        pk2.COEFF_CPR1,
        coefficient_component,
    )
    assert mutated != overlay
    assert not np.array_equal(pk2.decode_overlay_carrier(mutated, codec)[0], decoded_basis)


def test_overlay_metadata_is_counted_and_truncation_refuses() -> None:
    codec, bundle, *_ = _baseline_arrays()
    basis_component, coefficient_component = pk2.split_cpr1_components(bundle.carrier, codec)
    first = pk2.encode_overlay(
        pk2.BASIS_CPR1,
        basis_component,
        pk2.COEFF_CPR1,
        coefficient_component,
    )
    second = pk2.encode_overlay(
        pk2.BASIS_CPR1,
        basis_component + b"\x00",
        pk2.COEFF_CPR1,
        coefficient_component,
    )
    assert len(second) == len(first) + 1
    assert second != first
    with pytest.raises(ValueError):
        pk2.decode_overlay_carrier(first[:-1], codec)


def test_declared_array_mismatch_is_detectable() -> None:
    codec, bundle, *_rest, basis, coefficients = _baseline_arrays()
    basis_component, coefficient_component = pk2.split_cpr1_components(bundle.carrier, codec)
    overlay = pk2.encode_overlay(
        pk2.BASIS_CPR1,
        basis_component,
        pk2.COEFF_CPR1,
        coefficient_component,
    )
    decoded_basis, decoded_coefficients = pk2.decode_overlay_carrier(overlay, codec)
    fake_declared = decoded_coefficients.copy()
    fake_declared[0, 0] += 1.0
    assert not np.array_equal(decoded_coefficients, fake_declared)
    assert np.array_equal(decoded_basis, basis)
    assert np.array_equal(decoded_coefficients, coefficients)


def test_deterministic_archive_has_one_stored_member() -> None:
    first = pk2.deterministic_archive(b"payload")
    second = pk2.deterministic_archive(b"payload")
    assert first == second
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.namelist() == ["p"]
        assert archive.getinfo("p").compress_type == zipfile.ZIP_STORED
        assert archive.read("p") == b"payload"


def test_joint_delta_prices_full_archive_bytes() -> None:
    terms = pk2.joint_delta_s(
        baseline_d_pose=2.0e-5,
        candidate_d_pose=2.0e-5,
        baseline_d_seg=3.0e-4,
        candidate_d_seg=3.0e-4,
        baseline_archive_bytes=191_052,
        candidate_archive_bytes=187_052,
    )
    assert terms["seg"] == 0.0
    assert terms["pose"] == 0.0
    assert terms["rate"] == pytest.approx(-25 * 4_000 / pk2.REFERENCE_BYTES)
    assert terms["total"] == terms["rate"]


def test_measured_dimension_order_uses_joint_cost_per_archive_byte() -> None:
    rows = []
    for dimension in range(pk2.DIM):
        rows.append(
            {
                "name": f"capacity_drop_dim{dimension:02d}",
                "delta_archive_bytes": -(100 + dimension),
                "delta_s_seg": 0.0,
                "delta_s_pose": float(pk2.DIM - dimension),
            }
        )
    order = pk2.measured_dimension_order(rows)
    assert order[0] == pk2.DIM - 1
    assert order[-1] == 0


def test_score_selection_refuses_toy_prefix_scale() -> None:
    targets = np.zeros((pk2.N, 6), dtype=np.float32)
    import torch

    with pytest.raises(ValueError, match="n>=120"):
        pk2.selection(8, 20260809, torch.from_numpy(targets))


def test_score_selection_is_seeded_stratified_random() -> None:
    import torch

    targets = torch.arange(pk2.N * 6, dtype=torch.float32).reshape(pk2.N, 6)
    first, provenance = pk2.selection(120, 20260809, targets)
    second, _ = pk2.selection(120, 20260809, targets)
    assert first == second
    assert first != list(range(120))
    assert provenance["pair_selection"] == "stratified_blocks"
    assert provenance["representative_mode"] is True


def test_persisted_output_refuses_tmp_and_non_ssd_paths(tmp_path: Path) -> None:
    for value in (Path("/tmp/pk2"), Path("/private/tmp/pk2"), tmp_path):
        with pytest.raises(ValueError, match="persisted experiment output"):
            pk2.validate_out_dir(value)


def test_persisted_output_accepts_arm_specific_ssd_path() -> None:
    value = Path("/Volumes/VertigoDataTier/pact/ddm_pk2_20260809")
    assert pk2.validate_out_dir(value) == value
