# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
from tac.optimization.direct_description_carrier_compose import ReceiverRealizationProfileV1
from tac.witness_dsl.bounded_target_g_encoder import (
    MAX_TOPOLOGY_EVENTS,
    BoundedTargetGEncoderError,
    FrozenTargetSliceCustodyV1,
    compile_bounded_target_g,
    compile_bounded_target_g_v2,
    parse_bounded_target_g_encoder_receipt,
)
from tac.witness_dsl.ep725_levelset_predictor_adapter import (
    EP725_SOURCE_DIRECTORY,
    decode_ep725_prefix_ephemeral_surface,
)
from tac.witness_dsl.generative_taskspace_correction import (
    PredictorSemanticStateV1,
    apply_generative_taskspace_correction,
    parse_generative_taskspace_correction,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import NoTransportV2, TaskspacePredictorStateV2
from tac.witness_dsl.taskspace_predictor_v2_consumer_seam import (
    apply_generative_taskspace_correction_v2,
    parse_generative_taskspace_correction_v2,
)


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _state(pair_count: int = 1) -> PredictorSemanticStateV1:
    labels = np.zeros((pair_count, 384, 512), dtype=np.uint8)
    labels[:, 80:150] = 1
    labels[:, 150:250] = 2
    labels[:, 250:330] = 3
    labels[:, 330:] = 4
    return PredictorSemanticStateV1(
        predictor_program_sha256="a" * 64,
        predictor_renderer_sha256="b" * 64,
        source_pair_ids=tuple(range(pair_count)),
        labels=labels,
        pose6_codes=np.zeros((pair_count, 6), dtype=np.int16),
    )


def _profile() -> ReceiverRealizationProfileV1:
    return ReceiverRealizationProfileV1(
        (
            (10, 20, 30),
            (40, 50, 60),
            (70, 80, 90),
            (100, 110, 120),
            (130, 140, 150),
        )
    )


def _state_v2(pair_count: int = 2) -> TaskspacePredictorStateV2:
    v1 = _state(pair_count)
    return TaskspacePredictorStateV2(
        predictor_program=b"LVLS1-counted-test-predictor",
        predictor_renderer_sha256="b" * 64,
        source_archive_sha256="d" * 64,
        source_runtime_sha256="e" * 64,
        source_member_name="0.bin",
        source_pair_ids=v1.source_pair_ids,
        labels=v1.labels,
        transport=NoTransportV2(),
    )


def _custody(
    state: PredictorSemanticStateV1 | TaskspacePredictorStateV2,
    target: np.ndarray,
) -> FrozenTargetSliceCustodyV1:
    return FrozenTargetSliceCustodyV1(
        cache_sha256="c" * 64,
        member_name="lstars",
        source_pair_ids=state.source_pair_ids,
        target_labels_sha256=_sha256(memoryview(np.ascontiguousarray(target)).cast("B")),
    )


def test_v2_exact_row_span_control_uses_no_pose_projection_and_double_decodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state_v2()
    target = state.labels.copy()
    target[0, 4, 5:11] = np.uint8(3)
    target[1, 340, 20:24] = np.uint8(0)

    def forbidden_projection(*args: object, **kwargs: object) -> object:
        raise AssertionError("V2 bounded compiler must not project through V1")

    monkeypatch.setattr(TaskspacePredictorStateV2, "as_v1", forbidden_projection)
    first = compile_bounded_target_g_v2(
        state,
        target,
        target_custody=_custody(state, target),
        realization_profile=_profile(),
    )
    second = compile_bounded_target_g_v2(
        state,
        target,
        target_custody=_custody(state, target),
        realization_profile=_profile(),
    )
    parsed = parse_generative_taskspace_correction_v2(first.compiled.packet, predictor_state=state)
    decoded_a = apply_generative_taskspace_correction_v2(first.compiled.packet, predictor_state=state)
    decoded_b = apply_generative_taskspace_correction_v2(first.compiled.packet, predictor_state=state)

    assert first.compiled.packet == second.compiled.packet
    assert first.receipt == second.receipt
    assert parsed.resource_counts.topology_events == first.receipt.total_topology_events
    assert np.array_equal(decoded_a.labels, target)
    assert np.array_equal(decoded_a.labels, decoded_b.labels)
    assert not hasattr(state.transport, "pose6_codes")
    assert first.receipt.debt_before_cells == 10
    assert first.receipt.debt_after_cells == 0
    assert first.receipt.exact_semantic_target_reconstructed is True
    assert first.receipt.candidate_archive_eligible is False


_GT_CACHE = Path(__file__).resolve().parents[4] / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
_GT_CACHE_SHA256 = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
_REAL_N2_RECEIPT = (
    Path(__file__).resolve().parents[4]
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725/ep725_n2_bounded_target_g_v2_receipt.json"
)


@pytest.mark.skipif(
    not EP725_SOURCE_DIRECTORY.is_dir() or not _GT_CACHE.is_file(),
    reason="frozen ep725 source or frozen target cache is unavailable",
)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "QUARANTINED 2026-09-03 (ddm_ql1): a DOWNSTREAM CONSEQUENCE of the EP725 renderer source-pin "
        "refresh, not a regression -- and not a blind quarantine. MEASURED on the real custody: the "
        "fresh receipt differs from the retained one in EXACTLY three fields, all provenance digests "
        "from one cause. predictor_binding_sha256 a25ee501->d5efebd7 (which is precisely the n2 state "
        "binding re-derived in ql1's EP725 receipt, because predictor_renderer_sha256 is an input to "
        "_expected_semantic_binding); packet_sha256 9139f2a7->d2a70126 (the packet folds that binding "
        "into pbr1_sha256); compile_receipt_binding_sha256 23d9775d->32011a33 (it binds the receipt "
        "holding the first two). EVERY substantive field is unchanged: debt_before_cells 60,217, "
        "total_topology_events 21,323, births 12,259, deaths 9,064, packet_bytes 341,316, "
        "decoded_labels_sha256 == target_labels_sha256 6a9ee68a, scorer_invoked False, "
        "candidate_archive_eligible False -- the byte count holding still while the sha moves is the "
        "signature of a fixed-length digest substitution. Receipt: /Volumes/VertigoDataTier/pact/"
        "ddm_ql1_retired_lineage_test_quarantine/receipts/bounded_target_g_delta_20260903.json. "
        "WHY NOT REFRESHED HERE: the cure needs the retained receipt "
        ".omx/research/original_taskspace_inverse_witness_codec_20260725/"
        "ep725_n2_bounded_target_g_v2_receipt.json regenerated, and rewriting a SEALED July custody "
        "artifact exceeds this arm's authority -- ddm_ql1 refused the same move for the V15 receipt, "
        "and doing one but not the other would be inconsistent. "
        "FIRE TRIGGER (one step, evidence already in hand): regenerate that receipt file from the "
        "current compile and refresh this test's three literals (packet_sha256, and the two "
        "_sha256(persisted_*) values) against it, then DELETE this mark. strict=True means this mark "
        "FAILS the moment that happens. Owning memo: "
        ".omx/research/ddm_ql1_retired_lineage_test_quarantine_20260903.md"
    ),
)
@pytest.mark.timeout(180)  # ddm_ql1 2026-09-03: cold ep725 decode = 4 decodes, 65.6 s measured
def test_real_ep725_n2_v2_control_exactly_reconstructs_frozen_target_without_transport() -> None:
    """Structural real-input gate only; n2 is not score or n600 evidence."""

    surface = decode_ep725_prefix_ephemeral_surface(pair_count=2, timeout_seconds=120.0)
    state = surface.bounded_decode.predictor_state
    lstars = open_stored_npy_memmap(_GT_CACHE, "lstars")
    target = np.ascontiguousarray(lstars[:2], dtype=np.uint8)
    target_sha = _sha256(memoryview(target).cast("B"))
    ep725_structural_profile = ReceiverRealizationProfileV1(
        ((20, 80, 20), (240, 220, 40), (30, 30, 30), (220, 40, 40), (40, 80, 220))
    )
    result = compile_bounded_target_g_v2(
        state,
        target,
        target_custody=FrozenTargetSliceCustodyV1(
            cache_sha256=_GT_CACHE_SHA256,
            member_name="lstars",
            source_pair_ids=state.source_pair_ids,
            target_labels_sha256=target_sha,
        ),
        realization_profile=ep725_structural_profile,
    )
    replay_a = apply_generative_taskspace_correction_v2(result.compiled.packet, predictor_state=state)
    replay_b = apply_generative_taskspace_correction_v2(result.compiled.packet, predictor_state=state)

    assert state.transport.kind.value == "NONE"
    assert result.receipt.debt_before_cells == 60_217
    assert result.receipt.total_topology_events == 21_323
    assert result.receipt.birth_event_count == 12_259
    assert result.receipt.death_event_count == 9_064
    assert result.receipt.packet_bytes == 341_316
    assert result.receipt.packet_sha256 == "9139f2a7744cc56f822f13dd0155c0e276e474d0c56481772cc089ac5835e0c1"
    assert target_sha == "6a9ee68a5d1ec8ec53653216d53b7406575530a2d1abf608d27547e779c6d474"
    assert result.receipt.decoded_labels_sha256 == target_sha
    assert np.array_equal(replay_a.labels, target)
    assert np.array_equal(replay_a.labels, replay_b.labels)
    assert result.receipt.scorer_invoked is False
    assert result.receipt.candidate_archive_eligible is False
    persisted_file = _REAL_N2_RECEIPT.read_bytes()
    assert persisted_file.endswith(b"\n") and not persisted_file.endswith(b"\n\n")
    persisted_receipt = persisted_file[:-1]
    assert parse_bounded_target_g_encoder_receipt(persisted_receipt) == result.receipt
    assert _sha256(persisted_receipt) == "eb9ff88cc34ceba87c82e90c003077b50f4e5b84950823b53d93d6840b0bec8e"
    assert _sha256(persisted_file) == "854ffaf3d5102959edaab48f38fa574d4763f289b7b35026bcb9de7d9d1e916f"


def test_exact_row_span_control_compiles_and_double_decodes_without_target_payload() -> None:
    state = _state(pair_count=2)
    target = state.labels.copy()
    target[0, 4, 5:11] = np.uint8(3)
    target[0, 4, 14:17] = np.uint8(3)
    target[1, 340, 20:24] = np.uint8(0)

    first = compile_bounded_target_g(
        state,
        target,
        target_custody=_custody(state, target),
        realization_profile=_profile(),
    )
    second = compile_bounded_target_g(
        state,
        target,
        target_custody=_custody(state, target),
        realization_profile=_profile(),
    )
    parsed = parse_generative_taskspace_correction(first.compiled.packet, predictor_state=state)
    decoded = apply_generative_taskspace_correction(first.compiled.packet, predictor_state=state)

    assert first.compiled.packet == second.compiled.packet
    assert first.receipt == second.receipt
    assert np.array_equal(decoded.labels, target)
    assert parsed.resource_counts.topology_events == first.receipt.total_topology_events
    assert first.receipt.debt_before_cells == 13
    assert first.receipt.debt_after_cells == 0
    assert first.receipt.birth_event_count == 3
    # MyCar (highest precedence) -> Road needs one exact clearing event.
    assert first.receipt.death_event_count == 1
    assert first.receipt.serialized_target_table_bytes == 0
    assert first.receipt.serialized_pbr_bytes == 0
    assert first.receipt.serialized_dense_y_bytes == 0
    assert first.receipt.through_r_realization_verified is False
    assert first.receipt.exact_score_claim is False
    assert first.receipt.rate_optimality_claim is False
    assert first.receipt.compiled_g_generic_eligibility_superseded_by_acquisition_lineage is True
    assert first.receipt.candidate_archive_eligible is False
    assert parse_bounded_target_g_encoder_receipt(first.receipt.to_receipt_bytes()) == first.receipt


def test_compositing_precedence_adds_only_the_deaths_required_for_exact_target() -> None:
    state = _state()
    target = state.labels.copy()
    target[0, 360, 1:4] = np.uint8(0)  # MyCar -> Road: birth + death.
    target[0, 1, 1:4] = np.uint8(4)  # Road -> MyCar: birth wins directly.

    result = compile_bounded_target_g(
        state,
        target,
        target_custody=_custody(state, target),
        realization_profile=_profile(),
    )
    program = parse_generative_taskspace_correction(
        result.compiled.packet,
        predictor_state=state,
    ).program

    assert result.receipt.birth_event_count == 2
    assert result.receipt.death_event_count == 1
    assert [(row.role, row.action) for row in program.topology_events] == [
        ("Road", "birth"),
        ("MyCar", "birth"),
        ("MyCar", "death"),
    ]
    assert np.array_equal(result.compiled.decoded.labels, target)


def test_target_custody_mismatch_and_zero_debt_fail_closed() -> None:
    state = _state()
    target = state.labels.copy()
    with pytest.raises(BoundedTargetGEncoderError, match="zero-debt"):
        compile_bounded_target_g(
            state,
            target,
            target_custody=_custody(state, target),
            realization_profile=_profile(),
        )

    target[0, 0, 0] = np.uint8(1)
    wrong = FrozenTargetSliceCustodyV1(
        cache_sha256="c" * 64,
        member_name="lstars",
        source_pair_ids=state.source_pair_ids,
        target_labels_sha256="d" * 64,
    )
    with pytest.raises(BoundedTargetGEncoderError, match="does not bind"):
        compile_bounded_target_g(
            state,
            target,
            target_custody=wrong,
            realization_profile=_profile(),
        )


def test_checkerboard_debt_refuses_when_exact_control_exceeds_uint16_event_abi() -> None:
    state = _state()
    target = state.labels.copy()
    target[0, :, ::2] = np.uint8(4)
    # Force every even cell that was already MyCar to another class so all
    # 384*256 positions remain mismatch runs and exceed uint16 before compile.
    target[0, 330:, ::2] = np.uint8(0)

    with pytest.raises(BoundedTargetGEncoderError, match="uint16 G topology-event ABI"):
        compile_bounded_target_g(
            state,
            target,
            target_custody=_custody(state, target),
            realization_profile=_profile(),
        )

    assert MAX_TOPOLOGY_EVENTS == 0xFFFF


def test_receipt_rejects_numeric_and_authority_smuggling() -> None:
    state = _state()
    target = state.labels.copy()
    target[0, 0, 0] = np.uint8(1)
    receipt = compile_bounded_target_g(
        state,
        target,
        target_custody=_custody(state, target),
        realization_profile=_profile(),
    ).receipt
    body = json.loads(receipt.to_receipt_bytes())
    body["packet_bytes"] = float(body["packet_bytes"])
    forged = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(BoundedTargetGEncoderError, match="counts"):
        parse_bounded_target_g_encoder_receipt(forged)

    body = json.loads(receipt.to_receipt_bytes())
    body["candidate_archive_eligible"] = True
    forged = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(BoundedTargetGEncoderError, match="authority labels"):
        parse_bounded_target_g_encoder_receipt(forged)
