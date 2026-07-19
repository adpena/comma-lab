from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.boundary_math.integer_plane_emitter import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    FROZEN_PAIR_COEFFICIENTS_F64_SHA256,
    FROZEN_SEGNET_HEAD_FILE_SHA256,
    FROZEN_U4_F64_SHA256,
    MEASURED_U4_SINGULAR_VALUES,
    CapacitySignature,
    EncodeOnlyVJPGuidance,
    FixedCapacityBasisAB,
    IntegerPlaneEmitterError,
    IntegerPlaneGeometry,
    LatticeBatchProof,
    LatticePlaneProof,
    QuotientResidualState,
    SignFixedU4Basis,
    StructuredEmitterState,
    deterministic_coordinate_basis,
    mlx_uint8,
    numpy_precursor,
    numpy_uint8,
    realize_all_factor2,
    rgb_pair_to_yuv6,
    torch_uint8,
    torch_uint8_bytes,
    validate_scorer_uint8,
)


@pytest.fixture(scope="module")
def structured() -> StructuredEmitterState:
    base = np.full((1, 2, 384, 512, 3), 128.0, dtype=np.float32)
    base[:, 1] = np.float32(131.0)
    return StructuredEmitterState.from_base(base, residual_width=2)


@pytest.fixture(scope="module")
def residual(structured: StructuredEmitterState) -> QuotientResidualState:
    return QuotientResidualState.fresh(structured, seed=173, scale=0.03)


def _synthetic_custodied_head() -> np.ndarray:
    # Q spans 1^perp; V uses four distinct coordinates.  The resulting centered
    # five-row matrix has exactly the measured four singular values before its
    # required float32 serialization.
    projector = np.eye(5, dtype=np.float64) - np.ones((5, 5), dtype=np.float64) / 5.0
    q, _ = np.linalg.qr(projector[:, :4])
    right = np.zeros((4, 144), dtype=np.float64)
    right[:, :4] = np.eye(4, dtype=np.float64)
    rows = q @ np.diag(np.asarray(MEASURED_U4_SINGULAR_VALUES)) @ right
    return rows.astype(np.float32).reshape(5, 16, 3, 3)


@pytest.fixture(scope="module")
def u4() -> SignFixedU4Basis:
    head_path = Path(
        "/Users/adpena/Projects/pact/experiments/results/"
        "public_pr101_hnerv_ft_microcodec_intake_20260504_codex/"
        "source/models/segnet.safetensors"
    )
    if not head_path.is_file():
        pytest.skip("SHA-pinned PR101 frozen head is absent")
    from safetensors.torch import load_file

    weight = load_file(str(head_path))["segmentation_head.0.weight"].numpy()
    return SignFixedU4Basis.from_head_weight(
        weight,
        frozen_head_sha256=FROZEN_SEGNET_HEAD_FILE_SHA256,
    )


def test_geometry_is_exactly_frozen() -> None:
    geometry = IntegerPlaneGeometry()
    assert geometry.scorer_shape == (384, 512, 3)
    assert geometry.camera_shape == (874, 1164, 3)
    assert geometry.plane_count == 2


def test_geometry_refuses_variant() -> None:
    with pytest.raises(IntegerPlaneEmitterError, match="sealed"):
        IntegerPlaneGeometry(scorer_width=511)


def test_coordinate_basis_is_deterministic_f32_and_immutable() -> None:
    first = deterministic_coordinate_basis(9)
    second = deterministic_coordinate_basis(9)
    assert first.shape == (384, 512, 9)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert not first.flags.writeable


def test_coordinate_basis_has_no_duplicate_or_rank_deficient_capacity() -> None:
    basis = deterministic_coordinate_basis(12).reshape(-1, 12)
    assert np.linalg.matrix_rank(basis.astype(np.float64)) == 12
    for left in range(12):
        for right in range(left + 1, 12):
            assert not np.array_equal(basis[:, left], basis[:, right])


def test_structured_state_refuses_wrong_base_dtype() -> None:
    base = np.zeros((1, 2, 384, 512, 3), dtype=np.float64)
    with pytest.raises(IntegerPlaneEmitterError, match="dtype float32"):
        StructuredEmitterState.from_base(base, residual_width=1)


def test_structured_state_refuses_nonfinite_basis() -> None:
    base = np.zeros((1, 2, 384, 512, 3), dtype=np.float32)
    basis = deterministic_coordinate_basis(1).copy()
    basis[0, 0, 0] = np.nan
    with pytest.raises(IntegerPlaneEmitterError, match="finite"):
        StructuredEmitterState(base, basis)


def test_structured_state_refuses_cross_pair_autoregression() -> None:
    base = np.zeros((1, 2, 384, 512, 3), dtype=np.float32)
    with pytest.raises(IntegerPlaneEmitterError, match="pair-parallel"):
        StructuredEmitterState(
            base,
            deterministic_coordinate_basis(1),
            cross_pair_autoregression=True,
        )


def test_fresh_residual_is_seed_deterministic(
    structured: StructuredEmitterState,
) -> None:
    first = QuotientResidualState.fresh(structured, seed=99)
    second = QuotientResidualState.fresh(structured, seed=99)
    assert np.array_equal(first.pair_plane_codes, second.pair_plane_codes)
    assert np.array_equal(first.shared_rgb_head, second.shared_rgb_head)


def test_fresh_residual_indexes_planes_independently(
    structured: StructuredEmitterState,
) -> None:
    state = QuotientResidualState.fresh(structured, seed=100)
    assert not np.array_equal(state.pair_plane_codes[:, 0], state.pair_plane_codes[:, 1])


def test_numpy_emits_exact_uint8_shape_and_distinct_planes(
    structured: StructuredEmitterState, residual: QuotientResidualState
) -> None:
    emitted = numpy_uint8(structured, residual, require_distinct_planes=True)
    assert emitted.shape == (1, 2, 384, 512, 3)
    assert emitted.dtype == np.uint8
    assert not emitted.flags.writeable
    assert not np.array_equal(emitted[:, 0], emitted[:, 1])


def test_float_scorer_handoff_is_refused() -> None:
    with pytest.raises(IntegerPlaneEmitterError, match="dtype uint8"):
        validate_scorer_uint8(np.zeros((1, 2, 384, 512, 3), dtype=np.float32))


def test_wrong_scorer_geometry_is_refused() -> None:
    with pytest.raises(IntegerPlaneEmitterError, match="shape"):
        validate_scorer_uint8(np.zeros((1, 2, 383, 512, 3), dtype=np.uint8))


def test_empty_scorer_batch_is_refused() -> None:
    with pytest.raises(IntegerPlaneEmitterError, match="shape"):
        validate_scorer_uint8(np.zeros((0, 2, 384, 512, 3), dtype=np.uint8))


def test_copied_plane_collapse_is_refused() -> None:
    value = np.zeros((1, 2, 384, 512, 3), dtype=np.uint8)
    with pytest.raises(IntegerPlaneEmitterError, match="collapsed"):
        validate_scorer_uint8(value)


def test_copied_plane_collapse_is_refused_by_every_public_byte_path(
    structured: StructuredEmitterState,
) -> None:
    copied_base = np.full((1, 2, 384, 512, 3), 128.0, dtype=np.float32)
    copied = StructuredEmitterState.from_base(copied_base, residual_width=1)
    deleted = QuotientResidualState.deleted(copied)
    with pytest.raises(IntegerPlaneEmitterError, match="collapsed"):
        numpy_uint8(copied, deleted)
    with pytest.raises(IntegerPlaneEmitterError, match="collapsed"):
        torch_uint8_bytes(
            copied,
            torch.zeros((1, 2, 1), dtype=torch.float32),
            torch.zeros((1, 3), dtype=torch.float32),
        )
    with pytest.raises(IntegerPlaneEmitterError, match="collapsed"):
        realize_all_factor2(np.full((1, 2, 384, 512, 3), 17, dtype=np.uint8))


def test_deleting_only_residual_recovers_solved_base(
    structured: StructuredEmitterState,
) -> None:
    deleted = QuotientResidualState.deleted(structured)
    assert np.array_equal(numpy_precursor(structured, deleted), structured.base)


def test_pair_expansion_is_independent() -> None:
    base = np.full((2, 2, 384, 512, 3), 100.0, dtype=np.float32)
    state = StructuredEmitterState.from_base(base, residual_width=1)
    codes = np.ones((2, 2, 1), dtype=np.float32)
    head = np.ones((1, 3), dtype=np.float32)
    first = QuotientResidualState(codes, head, 0)
    changed_codes = codes.copy()
    changed_codes[0] = 9.0
    second = QuotientResidualState(changed_codes, head, 0)
    first_out = numpy_precursor(state, first)
    second_out = numpy_precursor(state, second)
    assert not np.array_equal(first_out[0], second_out[0])
    assert np.array_equal(first_out[1], second_out[1])


def test_torch_bytes_match_numpy_exactly(structured: StructuredEmitterState, residual: QuotientResidualState) -> None:
    codes = torch.tensor(np.array(residual.pair_plane_codes), dtype=torch.float32)
    head = torch.tensor(np.array(residual.shared_rgb_head), dtype=torch.float32)
    actual = torch_uint8_bytes(structured, codes, head)
    assert np.array_equal(actual, numpy_uint8(structured, residual))


def test_numpy_torch_match_half_even_ties_and_clip_edges() -> None:
    base = np.zeros((1, 2, 384, 512, 3), dtype=np.float32)
    base[0, 0, 0, 0] = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    base[0, 1, 0, 0] = np.array([-0.5, 254.5, 255.5], dtype=np.float32)
    state = StructuredEmitterState.from_base(base, residual_width=1)
    deleted = QuotientResidualState.deleted(state)
    expected = np.array([[0, 2, 2], [0, 254, 255]], dtype=np.uint8)
    numpy_bytes = numpy_uint8(state, deleted)
    torch_bytes = torch_uint8_bytes(
        state,
        torch.zeros((1, 2, 1), dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float32),
    )
    assert np.array_equal(numpy_bytes[0, :, 0, 0], expected)
    assert np.array_equal(torch_bytes, numpy_bytes)


def test_torch_in_range_ste_gradient_is_nonzero() -> None:
    base = np.full((1, 2, 384, 512, 3), 128.0, dtype=np.float32)
    state = StructuredEmitterState.from_base(base, residual_width=1)
    codes = torch.ones((1, 2, 1), dtype=torch.float32, requires_grad=True)
    head = torch.ones((1, 3), dtype=torch.float32, requires_grad=True)
    torch_uint8(state, codes, head).sum().backward()
    assert codes.grad is not None and bool(torch.count_nonzero(codes.grad))
    assert head.grad is not None and bool(torch.count_nonzero(head.grad))


@pytest.mark.parametrize("base_value", [-10.0, 300.0])
def test_torch_saturated_ste_gradient_is_zero(base_value: float) -> None:
    base = np.full((1, 2, 384, 512, 3), base_value, dtype=np.float32)
    state = StructuredEmitterState.from_base(base, residual_width=1)
    codes = torch.ones((1, 2, 1), dtype=torch.float32, requires_grad=True)
    head = torch.ones((1, 3), dtype=torch.float32, requires_grad=True)
    torch_uint8(state, codes, head).sum().backward()
    assert codes.grad is not None and not bool(torch.count_nonzero(codes.grad))
    assert head.grad is not None and not bool(torch.count_nonzero(head.grad))


def test_torch_refuses_non_f32_residual(structured: StructuredEmitterState) -> None:
    codes = torch.zeros((1, 2, 2), dtype=torch.float64)
    head = torch.zeros((2, 3), dtype=torch.float32)
    with pytest.raises(IntegerPlaneEmitterError, match=r"torch\.float32"):
        torch_uint8(structured, codes, head)


def test_mlx_ste_source_has_exact_stop_gradient_rounding() -> None:
    # Execution is separately environment-gated; this structural proof prevents
    # silent replacement with an identity-gradient clip or a float scorer path.
    source = inspect.getsource(mlx_uint8)
    assert "mx.clip" in source
    assert "mx.stop_gradient(mx.round(clipped) - clipped)" in source


def test_mlx_contract_accepts_actual_float32_dtype_equality(
    monkeypatch: pytest.MonkeyPatch, structured: StructuredEmitterState
) -> None:
    import tac.boundary_math.integer_plane_emitter as module

    class FakeMX:
        float32 = np.float32

        @staticmethod
        def array(value: np.ndarray, *, dtype: object) -> np.ndarray:
            return np.asarray(value, dtype=dtype)

        @staticmethod
        def einsum(equation: str, *values: np.ndarray) -> np.ndarray:
            return np.einsum(equation, *values, optimize=False)

        zeros_like = staticmethod(np.zeros_like)
        isfinite = staticmethod(np.isfinite)
        all = staticmethod(np.all)
        eval = staticmethod(lambda *values: None)
        clip = staticmethod(np.clip)
        round = staticmethod(np.round)
        stop_gradient = staticmethod(lambda value: value)

    monkeypatch.setattr(module, "_load_mlx", lambda: FakeMX)
    codes = np.zeros((1, 2, 2), dtype=np.float32)
    head = np.zeros((2, 3), dtype=np.float32)
    actual = module.mlx_uint8(structured, codes, head)
    assert actual.dtype == np.float32
    assert np.array_equal(actual, np.rint(structured.base))


def test_mlx_contract_refuses_nonfinite_residual_before_expansion(
    monkeypatch: pytest.MonkeyPatch, structured: StructuredEmitterState
) -> None:
    import tac.boundary_math.integer_plane_emitter as module

    class FakeMX:
        float32 = np.float32
        isfinite = staticmethod(np.isfinite)
        all = staticmethod(np.all)
        eval = staticmethod(lambda *values: None)

    monkeypatch.setattr(module, "_load_mlx", lambda: FakeMX)
    codes = np.zeros((1, 2, 2), dtype=np.float32)
    codes[0, 0, 0] = np.nan
    head = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(IntegerPlaneEmitterError, match="finite"):
        module.mlx_uint8(structured, codes, head)


def test_factor2_realization_proves_every_plane() -> None:
    targets = np.empty((1, 2, 384, 512, 3), dtype=np.uint8)
    targets[:, 0] = 37
    targets[:, 1] = 211
    receipt = realize_all_factor2(targets)
    assert receipt.camera_planes.shape == (1, 2, CAMERA_HEIGHT, CAMERA_WIDTH, 3)
    assert len(receipt.rows) == 2
    assert all(row.numerator_exact and row.certified_exact for row in receipt.rows)
    assert all(row.numerator_equal_values == row.scorer_values for row in receipt.rows)
    assert receipt.rows[0].camera_sha256 != receipt.rows[1].camera_sha256


def test_lattice_proof_rows_are_canonical_and_byte_bound() -> None:
    targets = np.empty((1, 2, 384, 512, 3), dtype=np.uint8)
    targets[:, 0] = 29
    targets[:, 1] = 203
    receipt = realize_all_factor2(targets)
    with pytest.raises(IntegerPlaneEmitterError, match="canonical pair/plane order"):
        LatticeBatchProof(
            receipt.scorer_planes,
            receipt.camera_planes,
            (replace(receipt.rows[0], pair_index=1), receipt.rows[1]),
        )
    with pytest.raises(IntegerPlaneEmitterError, match="bind the supplied"):
        LatticeBatchProof(
            receipt.scorer_planes,
            receipt.camera_planes,
            (replace(receipt.rows[0], camera_sha256="a" * 64), receipt.rows[1]),
        )
    with pytest.raises(IntegerPlaneEmitterError, match="successful exactness"):
        LatticePlaneProof(
            pair_index=0,
            plane_index=0,
            scorer_sha256="a" * 64,
            camera_sha256="b" * 64,
            denominator=1,
            numerator_equal_values=384 * 512 * 3,
            scorer_values=384 * 512 * 3,
            numerator_exact=False,
            certified_exact=True,
        )


def test_u4_refuses_wrong_head_custody() -> None:
    with pytest.raises(IntegerPlaneEmitterError, match="custody mismatch"):
        SignFixedU4Basis.from_head_weight(_synthetic_custodied_head(), frozen_head_sha256="0" * 64)


def test_u4_has_measured_singulars_rank_and_signs(u4: SignFixedU4Basis) -> None:
    assert np.allclose(
        u4.singular_values,
        np.asarray(MEASURED_U4_SINGULAR_VALUES, dtype=np.float64),
        rtol=0.0,
        atol=5e-12,
    )
    assert np.linalg.matrix_rank(u4.centered_weight) == 4
    for vector in u4.right_vectors:
        assert vector[int(np.argmax(np.abs(vector)))] > 0.0
    assert u4.u4_sha256 == FROZEN_U4_F64_SHA256
    assert u4.pair_coefficients_sha256 == FROZEN_PAIR_COEFFICIENTS_F64_SHA256


def test_u4_projection_reconstructs_centered_logits(u4: SignFixedU4Basis) -> None:
    features = np.arange(288, dtype=np.float32).reshape(2, 144) / np.float32(97.0)
    coordinates = u4.u4_coordinates(features)
    reconstructed = u4.centered_logits_from_u4(coordinates)
    expected = features.astype(np.float64) @ u4.centered_weight.T
    assert np.allclose(reconstructed, expected, rtol=0.0, atol=2e-14)


def test_u4_logit_equations_are_exact(u4: SignFixedU4Basis) -> None:
    logits = np.arange(15, dtype=np.float64).reshape(3, 5)
    centered = u4.centered_logits(logits)
    q = u4.logit_u4_coordinates(logits)
    assert np.allclose(u4.centered_logits_from_u4(q), centered, atol=5e-15, rtol=0.0)
    direct = centered @ u4.pair_difference_map.T
    assert np.allclose(u4.all_pair_margins_from_u4(q), direct, atol=1e-14, rtol=0.0)


def test_u4_raw_coordinates_and_all_ten_margins(u4: SignFixedU4Basis) -> None:
    logits = np.arange(10, dtype=np.float32).reshape(2, 5)
    raw = u4.raw_four_coordinates(logits)
    margins = u4.all_pair_margins(logits)
    assert raw.shape == (2, 4)
    assert margins.shape == (2, 10)
    assert np.array_equal(raw[0], np.array([1, 2, 3, 4], dtype=np.float32))
    expected = np.array(
        [logits[0, i] - logits[0, j] for i, j in __import__("itertools").combinations(range(5), 2)],
        dtype=np.float32,
    )
    assert np.array_equal(margins[0], expected)


def test_u4_forbids_sigma_division(u4: SignFixedU4Basis) -> None:
    with pytest.raises(IntegerPlaneEmitterError, match="sigma normalization"):
        u4.coordinates(np.zeros((1, 144), dtype=np.float32), divide_by_sigma=True)


def test_fixed_capacity_ab_shares_signature_and_bytes(
    structured: StructuredEmitterState,
    residual: QuotientResidualState,
    u4: SignFixedU4Basis,
) -> None:
    harness = FixedCapacityBasisAB(structured, residual, u4)
    expected = CapacitySignature.from_states(structured, residual)
    assert harness.capacity_signature == expected
    emitted = numpy_uint8(structured, residual)
    harness.require_same_emitted_bytes(emitted, emitted.copy())
    logits = np.arange(5, dtype=np.float32)[None]
    assert harness.objective_view("raw_centered", logits).shape == (1, 4)
    assert harness.objective_view("sign_fixed_u4_pair_margin", logits).shape == (1, 10)


def test_fixed_capacity_ab_refuses_output_change(
    structured: StructuredEmitterState,
    residual: QuotientResidualState,
    u4: SignFixedU4Basis,
) -> None:
    harness = FixedCapacityBasisAB(structured, residual, u4)
    first = numpy_uint8(structured, residual)
    second = first.copy()
    second[0, 0, 0, 0, 0] ^= np.uint8(1)
    with pytest.raises(IntegerPlaneEmitterError, match="identical"):
        harness.require_same_emitted_bytes(first, second)


def test_fixed_capacity_ab_blocks_capacity_mutation_before_verdict(
    structured: StructuredEmitterState,
    residual: QuotientResidualState,
    u4: SignFixedU4Basis,
) -> None:
    harness = FixedCapacityBasisAB(structured, residual, u4)
    changed = replace(harness.capacity_signature, total_parameters=999)
    with pytest.raises(IntegerPlaneEmitterError, match="blocked"):
        harness.refuse_capacity_change_until_verdict(changed)
    harness.refuse_capacity_change_until_verdict(harness.capacity_signature)


def test_vjp_guidance_is_hash_bound_encode_only() -> None:
    guidance = EncodeOnlyVJPGuidance("a" * 64, "b" * 64, (0, 7, 19))
    metadata = guidance.proposal_metadata()
    assert metadata["decoder_serializable"] is False
    assert metadata["candidate_admission_authority"] is False
    assert metadata["pair_ids"] == [0, 7, 19]


def test_vjp_guidance_has_no_decoder_or_admission_surface() -> None:
    guidance = EncodeOnlyVJPGuidance("a" * 64, "b" * 64, (0,))
    with pytest.raises(IntegerPlaneEmitterError, match="no decoder"):
        guidance.decoder_payload()
    with pytest.raises(IntegerPlaneEmitterError, match="cannot admit"):
        guidance.admit_candidate()


def test_rgb_pair_to_yuv6_matches_frozen_2x2_formula() -> None:
    pair = np.zeros((1, 2, 2, 2, 3), dtype=np.float32)
    pair[0, 0, :, :, 0] = np.array([[10, 20], [30, 40]], dtype=np.float32)
    pair[0, 1, :, :, 2] = np.array([[50, 60], [70, 80]], dtype=np.float32)
    actual = rgb_pair_to_yuv6(pair)
    assert actual.shape == (1, 12, 1, 1)
    first_luma = pair[0, 0, :, :, 0] * np.float32(0.299)
    expected_luma_order = first_luma[(0, 1, 0, 1), (0, 0, 1, 1)]
    assert np.allclose(actual[0, :4, 0, 0], expected_luma_order, atol=2e-6)
    second_luma = pair[0, 1, :, :, 2] * np.float32(0.114)
    expected_u = np.mean((pair[0, 1, :, :, 2] - second_luma) / 1.772 + 128.0)
    assert actual[0, 10, 0, 0] == pytest.approx(expected_u, abs=2e-5)


def test_pose_chroma_is_visible_only_through_2x2_average() -> None:
    base = np.full((1, 2, 2, 2, 3), 100, dtype=np.uint8)
    permuted = base.copy()
    # Opposing within-block red/blue swaps preserve the block chroma average.
    permuted[0, 0, 0, 0] = [120, 100, 80]
    permuted[0, 0, 1, 1] = [80, 100, 120]
    baseline = rgb_pair_to_yuv6(base)
    changed = rgb_pair_to_yuv6(permuted)
    assert changed[0, 4, 0, 0] == pytest.approx(baseline[0, 4, 0, 0], abs=2e-5)
    assert changed[0, 5, 0, 0] == pytest.approx(baseline[0, 5, 0, 0], abs=2e-5)
    # Luma samples remain separately visible, so the transform is not collapsed.
    assert not np.array_equal(changed[0, :4], baseline[0, :4])
