# SPDX-License-Identifier: MIT
"""Contract tests for the DDM family-(d) score-quotient representation."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tac.contest_score import compute_contest_score
from tac.optimization.ddm_min_description_contract import LayerHome, StreamType
from tac.optimization.ddm_score_quotient_functional_contract import (
    AXIS,
    DEMAND_ROW_COUNT,
    FRONTIER_POINTER,
    V14_BASELINE_ARCHIVE_BYTES,
    V14_BASELINE_D_SEG,
    DecodedDemandPlacementV1,
    ExternallyPricedDemandPlacementV1,
    FunctionalParametersV1,
    LaneOrbitRankCertificateV1,
    PixelExceptionV1,
    ScoreQuotientContractError,
    TemporalLatentV1,
    build_ddm_event_continuation_v1_fit_request,
    compile_score_quotient_packet,
    derive_score_quotient_capacity,
    parse_score_quotient_packet,
    receive_score_quotient_packet,
    score_quotient_functional_objective,
    v14_baseline_falsifier,
)
from tac.through_r.resolution_chain import CAMERA_H, CAMERA_W


def _parameters() -> FunctionalParametersV1:
    return FunctionalParametersV1(
        base_rgb_u8=np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
        row_basis_i8=np.zeros((2, 3, 384), dtype=np.int8),
        col_basis_i8=np.zeros((2, 3, 512), dtype=np.int8),
    )


def _latents(n: int = 24) -> tuple[TemporalLatentV1, ...]:
    return tuple(
        TemporalLatentV1(
            pair_index=pair,
            coefficients_q8=(0, 0, 0, 0, 0, 0),
            xi_q12=(pair, -pair, 2 * pair, -2 * pair, 3 * pair, -3 * pair),
        )
        for pair in range(n)
    )


def _placements() -> tuple[ExternallyPricedDemandPlacementV1, ...]:
    decoded_sha = hashlib.sha256(b"dm1-decoded-fixture").hexdigest()
    return tuple(
        ExternallyPricedDemandPlacementV1(
            pair_index=slot,
            bucket_index=100 + slot,
            slot_index=slot,
            coder_id="dm1.real-coder.v1",
            decoded_sha256=decoded_sha,
            coded_record=b"DM1" + bytes([slot]),
        )
        for slot in range(DEMAND_ROW_COUNT)
    )


def test_inactive_streams_preserve_named_base_byte_identity() -> None:
    base = b"named-v14-base-bytes"
    compiled = compile_score_quotient_packet(
        named_base="v14.receiver-baseline",
        named_base_bytes=base,
    )
    assert compiled.payload is base
    assert compiled.receipt.inactive_base_identity is True
    assert compiled.receipt.packet_sha256 == hashlib.sha256(base).hexdigest()
    assert compiled.receipt.total_counted_bytes == len(base)
    assert compiled.receipt.sections == ()


def test_packet_is_real_coded_typed_crc_closed_and_parseback_exact() -> None:
    base = b"v14"
    exception = PixelExceptionV1(
        pair_index=0,
        frame_index=1,
        y=8,
        x=9,
        channel=2,
        value_u8=77,
    )
    compiled = compile_score_quotient_packet(
        named_base="v14",
        named_base_bytes=base,
        parameters=_parameters(),
        temporal_latents=_latents(),
        demand_placements=_placements(),
        exceptions=(exception,),
    )
    assert compiled.payload != base
    assert compiled.receipt.inactive_base_identity is False
    assert compiled.receipt.total_counted_bytes == len(base) + len(compiled.payload)
    assert [section.kind.value for section in compiled.receipt.sections] == [
        "PARAMETERS",
        "TEMPORAL_LATENTS",
        "DEMAND_PLACEMENTS",
        "EXCEPTIONS",
    ]
    assert compiled.receipt.sections[0].typed_tag.type is StreamType.SKELETON
    assert compiled.receipt.sections[0].typed_tag.layer_home is LayerHome.L1_PROGRAM
    assert compiled.receipt.sections[2].typed_tag.type is StreamType.FIBER
    assert compiled.receipt.sections[2].typed_tag.layer_home is LayerHome.L3_RASTER
    assert compiled.receipt.sections[2].coder.value == "PASSTHROUGH"
    assert sum(
        section.typed_tag.counted_bytes for section in compiled.receipt.sections
    ) == len(compiled.payload)

    parsed = parse_score_quotient_packet(
        compiled.payload,
        named_bases={"v14": base},
    )
    assert parsed.base_sha256 == hashlib.sha256(base).hexdigest()
    assert parsed.temporal_latents == _latents()
    assert parsed.demand_placements == _placements()
    assert parsed.exceptions == (exception,)
    assert np.array_equal(parsed.parameters.base_rgb_u8, _parameters().base_rgb_u8)

    corrupt = bytearray(compiled.payload)
    corrupt[-1] ^= 1
    with pytest.raises(ScoreQuotientContractError, match="CRC"):
        parse_score_quotient_packet(bytes(corrupt), named_bases={"v14": base})


def test_demand_records_require_all_25_canonical_external_rows() -> None:
    with pytest.raises(ScoreQuotientContractError, match=r"slots 0\.\.24"):
        compile_score_quotient_packet(
            named_base="v14",
            named_base_bytes=b"v14",
            parameters=_parameters(),
            temporal_latents=_latents(),
            demand_placements=_placements()[:-1],
        )
    reversed_rows = tuple(reversed(_placements()))
    with pytest.raises(ScoreQuotientContractError, match="canonical"):
        compile_score_quotient_packet(
            named_base="v14",
            named_base_bytes=b"v14",
            parameters=_parameters(),
            temporal_latents=_latents(),
            demand_placements=reversed_rows,
        )


def test_receiver_exact_parseback_on_n24_hard_tail_first_through_real_r() -> None:
    base = b"v14"
    pairs = tuple(range(24))
    compiled = compile_score_quotient_packet(
        named_base="v14",
        named_base_bytes=base,
        parameters=_parameters(),
        temporal_latents=_latents(),
        demand_placements=_placements(),
    )

    def no_op_dm1(
        planes: np.ndarray,
        _row: ExternallyPricedDemandPlacementV1,
    ) -> DecodedDemandPlacementV1:
        return DecodedDemandPlacementV1(b"dm1-decoded-fixture", planes)

    def constant_camera(_pair: int, _frame: int, plane: np.ndarray) -> np.ndarray:
        assert np.all(plane == plane[0, 0])
        return np.broadcast_to(
            plane[0, 0],
            (CAMERA_H, CAMERA_W, 3),
        ).copy()

    proof = receive_score_quotient_packet(
        compiled.payload,
        named_bases={"v14": base},
        pair_indices=pairs,
        hard_tail_order=pairs,
        realize_plane_to_camera=constant_camera,
        placement_applier=no_op_dm1,
    )
    assert proof.pair_count == 24
    assert proof.exact_parseback is True
    assert proof.exact_through_r is True
    assert proof.cpu_threads == 4
    assert proof.axis == AXIS
    assert proof.score_claim is False
    assert proof.pairs[7].pose_stats[0] == pytest.approx(7 / 4096)

    with pytest.raises(ScoreQuotientContractError, match="n>=24"):
        receive_score_quotient_packet(
            compiled.payload,
            named_bases={"v14": base},
            pair_indices=pairs[:23],
            hard_tail_order=pairs,
            realize_plane_to_camera=constant_camera,
            placement_applier=no_op_dm1,
        )
    with pytest.raises(ScoreQuotientContractError, match="DM1 decoder"):
        receive_score_quotient_packet(
            compiled.payload,
            named_bases={"v14": base},
            pair_indices=pairs,
            hard_tail_order=pairs,
            realize_plane_to_camera=constant_camera,
        )

    def wrong_dm1_hash(
        planes: np.ndarray,
        _row: ExternallyPricedDemandPlacementV1,
    ) -> DecodedDemandPlacementV1:
        return DecodedDemandPlacementV1(b"wrong-decoded-value", planes)

    with pytest.raises(ScoreQuotientContractError, match="decoded_sha256"):
        receive_score_quotient_packet(
            compiled.payload,
            named_bases={"v14": base},
            pair_indices=pairs,
            hard_tail_order=pairs,
            realize_plane_to_camera=constant_camera,
            placement_applier=wrong_dm1_hash,
        )


def test_exact_s_functional_delegates_to_canonical_contest_score() -> None:
    compiled = compile_score_quotient_packet(
        named_base="v14",
        named_base_bytes=b"v14",
        parameters=_parameters(),
        temporal_latents=_latents(),
    )
    objective = score_quotient_functional_objective(0.01, 0.0004, compiled.receipt)
    assert objective.archive_bytes == compiled.receipt.total_counted_bytes
    assert objective.score == compute_contest_score(
        0.01,
        0.0004,
        compiled.receipt.total_counted_bytes,
    )
    assert objective.exact_real_coder_bytes is True
    assert objective.score_claim is False
    assert objective.frontier_pointer == FRONTIER_POINTER


def test_capacity_keeps_approximate_lane_orbit_null_until_certified() -> None:
    pending = derive_score_quotient_capacity()
    assert pending.seg_head.exact_value == 4
    assert pending.lane_orbit.exact_value is None
    assert pending.lane_orbit.approximate_hint == 8
    assert pending.pose_xi.exact_value == 6
    assert pending.demand_rows.exact_value == 25
    assert pending.exact_total is None
    assert pending.status == "NULL_DERIVATION_OWED"

    certificate = LaneOrbitRankCertificateV1(
        exact_rank=8,
        source_artifact=".omx/research/fixture.json",
        source_sha256="a" * 64,
        measurement_axis="[macOS-CPU frozen-scorer advisory]",
        realized_through_r=True,
    )
    certified = derive_score_quotient_capacity(
        lane_orbit_rank_certificate=certificate
    )
    assert certified.exact_total == 4 + 8 + 6 + 25
    assert certified.status == "COMPLETE"


def test_v14_falsifier_and_future_fit_request_fail_closed() -> None:
    incomplete = v14_baseline_falsifier(
        candidate_d_seg=None,
        candidate_archive_bytes=None,
        receiver_closed=False,
    )
    assert incomplete.verdict == "INCOMPLETE"
    assert incomplete.missing_stream == "FIT_RESULT_RECEIVER_CLOSED_V14_OR_BETTER"

    passing = v14_baseline_falsifier(
        candidate_d_seg=V14_BASELINE_D_SEG,
        candidate_archive_bytes=V14_BASELINE_ARCHIVE_BYTES,
        receiver_closed=True,
    )
    assert passing.verdict == "EXPRESSIBLE_V14_OR_BETTER"
    assert passing.missing_stream is None

    compiled = compile_score_quotient_packet(
        named_base="v14",
        named_base_bytes=b"v14",
        parameters=_parameters(),
        temporal_latents=_latents(),
    )
    request = build_ddm_event_continuation_v1_fit_request(compiled.receipt)
    assert request.schema == "DDMEventContinuationV1"
    assert request.status == "INTERFACE_ONLY_NOT_EXECUTABLE"
    assert request.real_coder_in_loss is True
    assert request.execution_allowed is False
    assert request.score_claim is False
    assert request.support_gaps
