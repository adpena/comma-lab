# SPDX-License-Identifier: MIT
"""Bounded real-fixture proof for the exact V15-native G74 overlay."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as torch_functional

import tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 as g74
from tac.optimization.direct_description_carrier_compose import (
    BoundaryShearletAtomV1,
)
from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    RoleAwareBoundaryShearletOperandV1,
    V15RoleAwareOverlayDecoderV1,
    V15RoleAwareOverlayError,
    audit_v15_legacy_coordinate_mismatch,
    parse_role_aware_boundary_shearlet_operand,
    parse_v15_role_aware_overlay_receipt,
    resolve_source_pair_ids,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    SelectedPreimageFrameSelectorV1,
)

_FRESH_V15_SHA256 = "759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df"
_MAX_OPERAND_BYTES = 4096


def _fresh_v15_path() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
        / "fresh_v15_semantic_base_n600_20260726"
        / "ddm_v15_solved_templates_n600.not_a_candidate.zip.receipt-bytes"
    )


@pytest.fixture(scope="module")
def real_decoder() -> V15RoleAwareOverlayDecoderV1:
    archive_path = _fresh_v15_path()
    if not archive_path.is_file():
        pytest.skip("retained fresh V15 semantic custody archive is absent")
    archive = archive_path.read_bytes()
    return V15RoleAwareOverlayDecoderV1.open(
        archive,
        expected_archive_bytes=133_941,
        expected_archive_sha256=_FRESH_V15_SHA256,
        verify_member_effects=True,
    )


def _road_atom(*, pair_index: int = 0) -> BoundaryShearletAtomV1:
    return BoundaryShearletAtomV1(
        pair_index=pair_index,
        role="Road",
        center_y=160,
        center_x=256,
        scale_y=24,
        scale_x=96,
        shear_q4=0,
        amplitude_q4=64,
    )


def _operand(
    selector: SelectedPreimageFrameSelectorV1,
    *,
    atom: BoundaryShearletAtomV1 | None = None,
) -> RoleAwareBoundaryShearletOperandV1:
    return RoleAwareBoundaryShearletOperandV1(
        frame_selector=selector,
        atoms=(atom or _road_atom(),),
    )


def _decode(
    decoder: V15RoleAwareOverlayDecoderV1,
    selector: SelectedPreimageFrameSelectorV1,
) -> object:
    operand = _operand(selector)
    payload = operand.to_bytes()
    return decoder.decode(
        payload,
        expected_operand_sha256=operand.sha256,
        maximum_operand_bytes=_MAX_OPERAND_BYTES,
        local_pair_ids=(0,),
    )


@pytest.fixture(scope="module")
def both_result(real_decoder: V15RoleAwareOverlayDecoderV1) -> object:
    return _decode(real_decoder, SelectedPreimageFrameSelectorV1.BOTH)


def _torch_resize(frame: np.ndarray) -> np.ndarray:
    tensor = (
        torch.from_numpy(np.array(frame, copy=True, order="C")).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    )
    with torch.inference_mode():
        output = torch_functional.interpolate(
            tensor,
            size=(384, 512),
            mode="bilinear",
        )
    return np.ascontiguousarray(output[0].permute(1, 2, 0).numpy())


def _expected_owner_mask(
    decoder: V15RoleAwareOverlayDecoderV1,
    changed_support_values: np.ndarray,
) -> np.ndarray:
    owned = np.zeros((874, 1164, 3), dtype=bool)
    rows, cols, channels = np.nonzero(changed_support_values)
    for row_offset in range(2):
        camera_rows = np.asarray(
            [support.indices[row_offset] for support in decoder.operator.row_supports],
            dtype=np.intp,
        )[rows]
        for col_offset in range(2):
            camera_cols = np.asarray(
                [support.indices[col_offset] for support in decoder.operator.col_supports],
                dtype=np.intp,
            )[cols]
            owned[camera_rows, camera_cols, channels] = True
    return owned


def test_real_v15_prepaint_overlay_matches_native_torch_on_exact_supports(
    real_decoder: V15RoleAwareOverlayDecoderV1,
    both_result: object,
) -> None:
    result = both_result
    base = real_decoder.receiver.render_camera_pairs((0,))
    ephemeral = replace(
        real_decoder.receiver,
        boundary_shearlets=(_road_atom(),),
    )
    native_mutated = ephemeral.render_camera_pairs((0,))

    for frame_index in range(2):
        base_numerators, denominator = real_decoder.operator.apply_numerators(base[0, frame_index])
        mutated_numerators, mutated_denominator = real_decoder.operator.apply_numerators(native_mutated[0, frame_index])
        output_numerators, output_denominator = real_decoder.operator.apply_numerators(
            result.camera_pairs[0, frame_index]
        )
        assert denominator == mutated_denominator == output_denominator
        assert np.array_equal(output_numerators, mutated_numerators)
        unchanged = ~result.changed_support_values[0, frame_index]
        assert np.array_equal(output_numerators[unchanged], base_numerators[unchanged])
        assert np.array_equal(
            _torch_resize(result.camera_pairs[0, frame_index]),
            _torch_resize(native_mutated[0, frame_index]),
        )
        assert np.array_equal(
            result.owned_camera_values[0, frame_index],
            _expected_owner_mask(
                real_decoder,
                result.changed_support_values[0, frame_index],
            ),
        )
        unowned = ~result.owned_camera_values[0, frame_index]
        assert np.array_equal(
            result.camera_pairs[0, frame_index][unowned],
            base[0, frame_index][unowned],
        )

    receipt = result.receipt
    assert receipt.base_archive_bytes == 133_941
    assert receipt.base_archive_sha256 == _FRESH_V15_SHA256
    assert receipt.operand_parse_reencode_identical is True
    assert receipt.operand_roles == ("Road",)
    assert receipt.realization_profile_consumed is True
    assert receipt.scorer_solved_template_count == 6
    assert receipt.legacy_render_pairs_used is False
    assert receipt.ephemeral_receiver_only is True
    assert receipt.ephemeral_receiver_archive_claim is False
    assert receipt.changed_support_cells_per_frame == (69, 69)
    assert receipt.changed_support_values_per_frame == (207, 207)
    assert receipt.changed_integer_numerator_values_per_frame == (207, 207)
    assert receipt.owned_camera_values_per_frame == (828, 828)
    assert receipt.actually_changed_camera_values_per_frame == (633, 633)
    assert receipt.preserved_unowned_camera_values == 6_102_360
    assert receipt.unchanged_scorer_numerator_values == 1_179_234
    assert receipt.exact_resize_denominator == 786_432
    assert receipt.selected_frames_match_native_mutated_numerators is True
    assert receipt.selected_frames_match_native_torch_bilinear is True
    assert receipt.torch_version == torch.__version__
    assert receipt.cross_host_torch_parity_claim is False
    assert receipt.deterministic_double_decode is True
    assert receipt.scorer_invoked is False
    assert receipt.pose_invoked is False
    assert receipt.pose_preservation_claim is False
    assert receipt.score_claim is False
    assert receipt.candidate_claim is False
    assert receipt.public_runtime_claim is False
    assert receipt.n600_evidence_claim is False
    assert parse_v15_role_aware_overlay_receipt(receipt.to_bytes()) == receipt


def test_integer_numerator_nullspace_is_owned_when_torch_float32_changes(
    real_decoder: V15RoleAwareOverlayDecoderV1,
) -> None:
    base = np.full((874, 1164, 3), 128, dtype=np.uint8)
    native_mutated = base.copy()
    row_support = real_decoder.operator.row_supports[13].indices
    col_support = real_decoder.operator.col_supports[0].indices
    native_mutated[row_support[0], col_support[0], 0] += 29
    native_mutated[row_support[1], col_support[0], 0] -= 99

    base_num, denominator = real_decoder.operator.apply_numerators(base)
    mutated_num, mutated_denominator = real_decoder.operator.apply_numerators(native_mutated)
    assert denominator == mutated_denominator
    assert base_num[13, 0, 0] == mutated_num[13, 0, 0]
    assert _torch_resize(base)[13, 0, 0] != _torch_resize(native_mutated)[13, 0, 0]

    changed_support = g74._support_tap_difference_mask(
        real_decoder.operator,
        base,
        native_mutated,
    )
    assert changed_support[13, 0, 0]
    replaced, owned = g74.DonorTapCopyPolicyV1().replace_changed_supports(
        base_frame=base,
        native_mutated_frame=native_mutated,
        changed_support_values=changed_support,
        operator=real_decoder.operator,
    )
    assert owned[row_support[0], col_support[0], 0]
    assert owned[row_support[1], col_support[0], 0]
    assert np.array_equal(_torch_resize(replaced), _torch_resize(native_mutated))


def test_counted_frame_selector_changes_only_selected_chronological_member(
    real_decoder: V15RoleAwareOverlayDecoderV1,
) -> None:
    base = real_decoder.receiver.render_camera_pairs((0,))
    y0 = _decode(real_decoder, SelectedPreimageFrameSelectorV1.Y0)
    y1 = _decode(real_decoder, SelectedPreimageFrameSelectorV1.Y1)

    assert y0.receipt.frame_selector == "Y0"
    assert y0.receipt.changed_support_values_per_frame[1] == 0
    assert np.any(y0.camera_pairs[0, 0] != base[0, 0])
    assert np.array_equal(y0.camera_pairs[0, 1], base[0, 1])
    assert y0.receipt.unselected_frames_byte_identical_to_base is True

    assert y1.receipt.frame_selector == "Y1"
    assert y1.receipt.changed_support_values_per_frame[0] == 0
    assert np.array_equal(y1.camera_pairs[0, 0], base[0, 0])
    assert np.any(y1.camera_pairs[0, 1] != base[0, 1])
    assert y1.receipt.unselected_frames_byte_identical_to_base is True


def test_role_is_receiver_live_and_zero_effect_role_refuses(
    real_decoder: V15RoleAwareOverlayDecoderV1,
) -> None:
    undrivable = replace(_road_atom(), role="UndrivableBoundary")
    operand = _operand(
        SelectedPreimageFrameSelectorV1.BOTH,
        atom=undrivable,
    )
    with pytest.raises(V15RoleAwareOverlayError, match="zero native support effect"):
        real_decoder.decode(
            operand.to_bytes(),
            expected_operand_sha256=operand.sha256,
            maximum_operand_bytes=_MAX_OPERAND_BYTES,
            local_pair_ids=(0,),
        )


def test_operand_roundtrip_tamper_constructor_seal_and_global_streaming(
    real_decoder: V15RoleAwareOverlayDecoderV1,
) -> None:
    operand = _operand(SelectedPreimageFrameSelectorV1.BOTH)
    payload = operand.to_bytes()
    assert (
        parse_role_aware_boundary_shearlet_operand(
            payload,
            expected_sha256=operand.sha256,
            maximum_operand_bytes=_MAX_OPERAND_BYTES,
        )
        == operand
    )
    with pytest.raises(V15RoleAwareOverlayError, match="body length/EOF"):
        parse_role_aware_boundary_shearlet_operand(
            payload + b"x",
            maximum_operand_bytes=_MAX_OPERAND_BYTES,
        )

    with pytest.raises(TypeError, match=r"constructed through \.open"):
        replace(real_decoder, receiver=real_decoder.receiver)

    pair_one = _operand(
        SelectedPreimageFrameSelectorV1.BOTH,
        atom=_road_atom(pair_index=1),
    )
    streamed = real_decoder.decode(
        pair_one.to_bytes(),
        expected_operand_sha256=pair_one.sha256,
        maximum_operand_bytes=_MAX_OPERAND_BYTES,
        local_pair_ids=(0,),
    )
    assert not np.any(streamed.changed_support_values)
    assert np.array_equal(
        streamed.camera_pairs,
        real_decoder.receiver.render_camera_pairs((0,)),
    )

    with pytest.raises(V15RoleAwareOverlayError):
        RoleAwareBoundaryShearletOperandV1(
            frame_selector=SelectedPreimageFrameSelectorV1.BOTH,
            atoms=(_road_atom(), _road_atom()),
        )


def test_nonzero_source_window_mapping_keeps_atoms_global() -> None:
    assert resolve_source_pair_ids(
        (0, 2, 5),
        source_pair_start=37,
        pair_count=6,
    ) == (37, 39, 42)
    with pytest.raises(V15RoleAwareOverlayError, match="escaped"):
        resolve_source_pair_ids(
            (5,),
            source_pair_start=595,
            pair_count=6,
        )


def test_real_v15_legacy_and_whole_factor2_coordinates_are_not_native(
    real_decoder: V15RoleAwareOverlayDecoderV1,
) -> None:
    audit = audit_v15_legacy_coordinate_mismatch(
        real_decoder,
        local_pair_id=0,
    )
    assert audit["legacy_vs_native_rounded_u8_changed_values_per_frame"] == [
        17_570,
        17_732,
    ]
    assert audit["legacy_vs_native_rounded_u8_changed_cells_per_frame"] == [
        6_728,
        6_928,
    ]
    assert audit["legacy_vs_native_rounded_u8_max_abs_per_frame"] == [116, 116]
    assert audit["whole_factor2_vs_native_camera_changed_values_per_frame"] == [
        745_497,
        745_958,
    ]
    assert audit["whole_factor2_zero_camera_values_per_frame"] == [
        708_180,
        708_180,
    ]
    assert audit["native_v15_zero_camera_values_per_frame"] == [20_680, 20_680]
    assert audit["diagnostic_only"] is True
    assert audit["score_claim"] is False
