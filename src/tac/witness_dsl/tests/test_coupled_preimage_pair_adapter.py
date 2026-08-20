# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import tac.witness_dsl.coupled_preimage_program as coupled_preimage_module
from tac.optimization.direct_description_carrier_compose import (
    ReceiverRealizationProfileV1,
    TopologyEventV1,
)
from tac.witness_dsl.coupled_preimage_pair_adapter import (
    A_ARTIFACT_CLASS,
    A_SECTION_ROLE,
    CoupledPreimagePairAdapterError,
    CoupledPreimageSourceCustodyV1,
    append_source_custodied_coupled_preimage,
    build_target_only_a_counterfactual,
    consume_counted_coupled_preimage,
)
from tac.witness_dsl.coupled_preimage_program import (
    CoupledPreimageMode,
    Frame1AnchoredY0FibreControlV1,
    JointSharedSkeletonTwoFibreControlV1,
    compile_coupled_preimage_program,
)
from tac.witness_dsl.coupled_witness_state import canonical_json_bytes, decode_canonical_json
from tac.witness_dsl.generative_taskspace_correction import (
    DecodedGenerativeCorrectionV1,
    EncoderOnlyTeacherEvidenceV1,
    GenerativeCorrectionProgramV1,
    PredictorSemanticStateV1,
    compile_generative_taskspace_correction,
)
from tac.witness_dsl.pair_population_envelope import (
    CountedArtifactClass,
    CountedPayloadLineage,
    CountedPayloadProvenance,
    CountedProgramSection,
    CountedSectionRole,
    PairPopulation,
    validate_counted_program_sections,
    validate_reverse_causal_counted_program_sections,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _profile(*, blue_offset: int = 0) -> ReceiverRealizationProfileV1:
    return ReceiverRealizationProfileV1(
        (
            (12, 24, 36 + blue_offset),
            (48, 60, 72),
            (84, 96, 108),
            (120, 132, 144),
            (156, 168, 180),
        )
    )


def _predictor_state() -> PredictorSemanticStateV1:
    labels = np.zeros((2, 384, 512), dtype=np.uint8)
    labels[:, 80:160] = 1
    labels[:, 160:240] = 2
    labels[:, 240:320] = 3
    labels[:, 320:] = 4
    labels[1] = np.roll(labels[1], 9, axis=1)
    return PredictorSemanticStateV1(
        predictor_program_sha256="a" * 64,
        predictor_renderer_sha256="b" * 64,
        source_pair_ids=(20, 21),
        labels=labels,
        pose6_codes=np.arange(12, dtype=np.int16).reshape(2, 6),
    )


def _teacher(state: PredictorSemanticStateV1) -> EncoderOnlyTeacherEvidenceV1:
    target = np.ascontiguousarray(state.labels, dtype=np.uint8)
    return EncoderOnlyTeacherEvidenceV1(
        pbr1_sha256="1" * 64,
        pbr2_sha256="2" * 64,
        target_labels_sha256=_sha256(memoryview(target).cast("B")),
        obligation_ir_sha256="3" * 64,
        oracle_evidence_sha256="4" * 64,
        dense_y_sha256="5" * 64,
        target_labels=target,
        teacher_event_count=0,
    )


def _compiled_g(state: PredictorSemanticStateV1):
    return compile_generative_taskspace_correction(
        state,
        GenerativeCorrectionProgramV1(
            topology_events=(TopologyEventV1(20, "MyCar", "birth", "box", 1, 40, 40, 42, 43),),
            realization_profile=_profile(),
        ),
        teacher_evidence=_teacher(state),
    )


def _anchored_controls(*, delta: int = 0) -> tuple[Frame1AnchoredY0FibreControlV1, ...]:
    return (
        Frame1AnchoredY0FibreControlV1(20, 1, -2, (3 + delta, -4, 5)),
        Frame1AnchoredY0FibreControlV1(21, -1, 2, (-2, 4, 1)),
    )


def _joint_controls(*, delta: int = 0) -> tuple[JointSharedSkeletonTwoFibreControlV1, ...]:
    return (
        JointSharedSkeletonTwoFibreControlV1(
            20,
            2,
            0,
            (
                (1 + delta, 0, 0),
                (0, 2, 0),
                (0, 0, 3),
                (-4, 0, 0),
                (0, -5, 0),
            ),
        ),
        JointSharedSkeletonTwoFibreControlV1(
            21,
            0,
            -3,
            (
                (0, 0, -1),
                (2, 0, 0),
                (0, 3, 0),
                (0, 0, 4),
                (-5, 0, 0),
            ),
        ),
    )


def _compiled_a(
    state: PredictorSemanticStateV1,
    decoded_g: DecodedGenerativeCorrectionV1,
    mode: CoupledPreimageMode,
    *,
    delta: int = 0,
):
    kwargs = (
        {"anchored_controls": _anchored_controls(delta=delta)}
        if mode is CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE
        else {"joint_controls": _joint_controls(delta=delta)}
    )
    return compile_coupled_preimage_program(state, decoded_g, mode=mode, **kwargs)


def _population(*, v10_ids: tuple[int, ...] = (20, 21)) -> PairPopulation:
    return PairPopulation.derive(
        source_pair_ids=(20, 21),
        v9_local_to_source_pair_ids=(20, 21),
        pbr_local_to_source_pair_ids=(20, 21),
        ir_local_to_source_pair_ids=(20, 21),
        v10_local_to_source_pair_ids=v10_ids,
    )


def _provenance(role: CountedSectionRole, state: PredictorSemanticStateV1) -> CountedPayloadProvenance:
    artifact_class = {
        CountedSectionRole.GENERATIVE_CORRECTION: CountedArtifactClass.GENERATOR_PARAMETERS,
        CountedSectionRole.TERMINAL_QUOTIENT: CountedArtifactClass.TERMINAL_QUOTIENT,
    }[role]
    source_sha256 = _sha256(Path(__file__).read_bytes())
    return CountedPayloadProvenance(
        artifact_class=artifact_class,
        lineage=CountedPayloadLineage.ORIGINAL_OWN,
        producer_id="test.coupled_preimage_pair_adapter_fixture",
        producer_source_sha256=source_sha256,
        derivation_input_sha256=state.binding_sha256,
        video_derived=True,
    )


def _g_prefix(state: PredictorSemanticStateV1, g_packet: bytes) -> tuple[bytes, tuple[CountedProgramSection, ...]]:
    section = CountedProgramSection(
        role=CountedSectionRole.GENERATIVE_CORRECTION,
        offset=0,
        byte_length=len(g_packet),
        payload_sha256=_sha256(g_packet),
        provenance=_provenance(CountedSectionRole.GENERATIVE_CORRECTION, state),
    )
    return g_packet, (section,)


def _custody(path: Path) -> CoupledPreimageSourceCustodyV1:
    return CoupledPreimageSourceCustodyV1.reopen(
        path,
        producer_source_path=Path(coupled_preimage_module.__file__),
        producer_id="tac.compile_coupled_preimage_program.fixture",
        payload_lineage=CountedPayloadLineage.ORIGINAL_OWN,
        video_derived=True,
    )


def _adapt(tmp_path: Path, mode: CoupledPreimageMode, *, delta: int = 0):
    state = _predictor_state()
    compiled_g = _compiled_g(state)
    compiled_a = _compiled_a(state, compiled_g.decoded, mode, delta=delta)
    artifact_path = tmp_path / f"{mode.value}.{delta}.a.json"
    artifact_path.write_bytes(compiled_a.packet)
    g_bytes, g_sections = _g_prefix(state, compiled_g.packet)
    adapted = append_source_custodied_coupled_preimage(
        g_bytes,
        g_sections,
        source_custody=_custody(artifact_path),
        pair_population=_population(),
        predictor_state=state,
        decoded_g=compiled_g.decoded,
    )
    return state, compiled_g, compiled_a, artifact_path, adapted


def _raw_a_variant(
    adapted,
    payload: bytes,
) -> tuple[bytes, tuple[CountedProgramSection, ...]]:
    a_section = adapted.section_manifest[-1]
    packet = adapted.program_bytes[: a_section.offset] + payload
    section = CountedProgramSection(
        role=A_SECTION_ROLE,
        offset=a_section.offset,
        byte_length=len(payload),
        payload_sha256=_sha256(payload),
        provenance=a_section.provenance,
    )
    return packet, (*adapted.section_manifest[:-1], section)


@pytest.mark.parametrize("mode", tuple(CoupledPreimageMode))
def test_both_modes_are_exact_counted_a_sections_and_double_decode(mode: CoupledPreimageMode, tmp_path: Path) -> None:
    state, compiled_g, compiled_a, artifact_path, adapted = _adapt(tmp_path, mode)

    assert tuple(section.role for section in adapted.section_manifest) == (
        CountedSectionRole.GENERATIVE_CORRECTION,
        A_SECTION_ROLE,
    )
    assert adapted.program_bytes == compiled_g.packet + compiled_a.packet
    assert adapted.consumed.section.offset == len(compiled_g.packet)
    assert adapted.consumed.section.byte_length == len(compiled_a.packet)
    assert adapted.consumed.section.payload_sha256 == _sha256(compiled_a.packet)
    assert adapted.consumed.section.provenance.artifact_class is A_ARTIFACT_CLASS
    assert adapted.consumed.counted_g_packet_bytes == len(compiled_g.packet)
    assert adapted.consumed.counted_g_packet_sha256 == _sha256(compiled_g.packet)
    assert adapted.consumed.counted_g_strict_reapplied is True
    assert adapted.consumed.caller_decoded_g_expected_value_matched is True
    assert adapted.source_custody.artifact.path == str(artifact_path.resolve())
    assert adapted.source_custody.artifact.byte_length == len(compiled_a.packet)
    assert adapted.source_custody.artifact.sha256 == _sha256(compiled_a.packet)
    assert adapted.consumed.mode is mode
    assert np.array_equal(adapted.consumed.y0, compiled_a.decoded.y0)
    assert np.array_equal(adapted.consumed.y1, compiled_a.decoded.y1)
    assert adapted.consumed.decoded.exact_y1_content_sha256 == compiled_a.receipt.exact_y1_content_sha256
    assert adapted.consumed.absolute_pose6_values_present is False
    assert adapted.consumed.pose6_consumed_by_materializer is False
    assert b"pose6" not in compiled_a.packet.lower()

    replay = consume_counted_coupled_preimage(
        adapted.program_bytes,
        adapted.section_manifest,
        pair_population=_population(),
        predictor_state=state,
        decoded_g=compiled_g.decoded,
    )
    assert replay.packet == adapted.consumed.packet
    assert replay.as_dict() == adapted.consumed.as_dict()
    assert replay.to_receipt_bytes() == adapted.consumed.to_receipt_bytes()
    assert replay.y0.tobytes() == adapted.consumed.y0.tobytes()
    assert replay.y1.tobytes() == adapted.consumed.y1.tobytes()
    receipt = decode_canonical_json(replay.to_receipt_bytes())
    assert receipt["body"]["counted_g_packet_bytes"] == len(compiled_g.packet)
    assert receipt["body"]["counted_g_packet_sha256"] == _sha256(compiled_g.packet)
    assert receipt["body"]["counted_g_artifact_class"] == CountedArtifactClass.GENERATOR_PARAMETERS.value
    assert receipt["body"]["counted_g_strict_reapplied"] is True
    assert receipt["body"]["caller_decoded_g_expected_value_matched"] is True
    assert receipt["body"]["source_runtime_closure"] == "DIRECT_MODULE_ONLY_NONTRANSITIVE"
    assert receipt["body"]["candidate_payload_eligible"] is False


@pytest.mark.parametrize("mode", tuple(CoupledPreimageMode))
def test_a_only_counterfactual_changes_y0_preserves_y1_and_non_target_bytes(
    mode: CoupledPreimageMode,
    tmp_path: Path,
) -> None:
    state, compiled_g, _compiled_a0, _artifact_path, adapted = _adapt(tmp_path, mode)
    changed = _compiled_a(state, compiled_g.decoded, mode, delta=1)
    changed_path = tmp_path / f"{mode.value}.counterfactual.a.json"
    changed_path.write_bytes(changed.packet)
    terminal = canonical_json_bytes({"schema": "test.terminal.v1", "bias": 0})
    terminal_section = CountedProgramSection(
        role=CountedSectionRole.TERMINAL_QUOTIENT,
        offset=len(adapted.program_bytes),
        byte_length=len(terminal),
        payload_sha256=_sha256(terminal),
        provenance=_provenance(CountedSectionRole.TERMINAL_QUOTIENT, state),
    )
    baseline_program = adapted.program_bytes + terminal
    baseline_sections = (*adapted.section_manifest, terminal_section)

    result = build_target_only_a_counterfactual(
        baseline_program,
        baseline_sections,
        counterfactual_custody=_custody(changed_path),
        pair_population=_population(),
        predictor_state=state,
        decoded_g=compiled_g.decoded,
    )
    assert result.causality.y0_changed is True
    assert result.causality.y1_changed is False
    assert result.counterfactual.program_bytes[: len(compiled_g.packet)] == compiled_g.packet
    assert result.section_manifest[0] == adapted.section_manifest[0]
    assert result.section_manifest[1].payload_sha256 == _sha256(changed.packet)
    assert result.counterfactual.program_bytes[result.section_manifest[-1].offset :] == terminal
    assert result.section_manifest[-1].payload_sha256 == terminal_section.payload_sha256
    decoded = consume_counted_coupled_preimage(
        result.counterfactual.program_bytes,
        result.section_manifest,
        pair_population=_population(),
        predictor_state=state,
        decoded_g=compiled_g.decoded,
    )
    assert np.array_equal(decoded.y1, adapted.consumed.y1)
    assert not np.array_equal(decoded.y0, adapted.consumed.y0)


@pytest.mark.parametrize("mode", tuple(CoupledPreimageMode))
def test_strict_pair_sliced_a_rejects_truncation_trailing_noncanonical_and_deletion(
    mode: CoupledPreimageMode,
    tmp_path: Path,
) -> None:
    state, compiled_g, compiled_a, _artifact_path, adapted = _adapt(tmp_path, mode)

    for malformed in (compiled_a.packet[:-1], compiled_a.packet + b"x", compiled_a.packet + b"\n"):
        program, sections = _raw_a_variant(adapted, malformed)
        with pytest.raises(CoupledPreimagePairAdapterError, match="strict reverse-causal consumption"):
            consume_counted_coupled_preimage(
                program,
                sections,
                pair_population=_population(),
                predictor_state=state,
                decoded_g=compiled_g.decoded,
            )

    with pytest.raises(CoupledPreimagePairAdapterError, match="manifest failed"):
        consume_counted_coupled_preimage(
            compiled_g.packet,
            adapted.section_manifest[:-1],
            pair_population=_population(),
            predictor_state=state,
            decoded_g=compiled_g.decoded,
        )


def test_unknown_mode_hash_mismatch_source_mismatch_and_pair_order_fail_closed(tmp_path: Path) -> None:
    state, compiled_g, compiled_a, _artifact_path, adapted = _adapt(
        tmp_path,
        CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
    )
    envelope = json.loads(compiled_a.packet.decode("ascii"))
    envelope["body"]["mode"] = "UNKNOWN_A_MODE"
    envelope["body_sha256"] = _sha256(canonical_json_bytes(envelope["body"]))
    unknown = canonical_json_bytes(envelope)
    program, sections = _raw_a_variant(adapted, unknown)
    with pytest.raises(CoupledPreimagePairAdapterError, match="strict reverse-causal consumption"):
        consume_counted_coupled_preimage(
            program,
            sections,
            pair_population=_population(),
            predictor_state=state,
            decoded_g=compiled_g.decoded,
        )

    corrupted_program = bytearray(adapted.program_bytes)
    corrupted_program[-1] ^= 1
    with pytest.raises(CoupledPreimagePairAdapterError, match="manifest failed"):
        consume_counted_coupled_preimage(
            bytes(corrupted_program),
            adapted.section_manifest,
            pair_population=_population(),
            predictor_state=state,
            decoded_g=compiled_g.decoded,
        )

    foreign_g = DecodedGenerativeCorrectionV1(
        labels=compiled_g.decoded.labels,
        realization_profile=_profile(blue_offset=1),
        correction_packet_sha256=compiled_g.decoded.correction_packet_sha256,
    )
    with pytest.raises(CoupledPreimagePairAdapterError, match="caller decoded_g"):
        consume_counted_coupled_preimage(
            adapted.program_bytes,
            adapted.section_manifest,
            pair_population=_population(),
            predictor_state=state,
            decoded_g=foreign_g,
        )

    with pytest.raises(CoupledPreimagePairAdapterError, match="source-pair IDs/order"):
        consume_counted_coupled_preimage(
            adapted.program_bytes,
            adapted.section_manifest,
            pair_population=_population(v10_ids=(21, 20)),
            predictor_state=state,
            decoded_g=compiled_g.decoded,
        )


def test_counted_g_is_strictly_reapplied_and_requires_generator_provenance(tmp_path: Path) -> None:
    state, compiled_g, _compiled_a0, _artifact_path, adapted = _adapt(
        tmp_path,
        CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
    )
    false_g_provenance = replace(
        adapted.section_manifest[0].provenance,
        artifact_class=CountedArtifactClass.COUPLED_PREIMAGE_PARAMETERS,
    )
    false_g_sections = (
        replace(adapted.section_manifest[0], provenance=false_g_provenance),
        adapted.section_manifest[1],
    )
    with pytest.raises(CoupledPreimagePairAdapterError, match="not typed as generator parameters"):
        consume_counted_coupled_preimage(
            adapted.program_bytes,
            false_g_sections,
            pair_population=_population(),
            predictor_state=state,
            decoded_g=compiled_g.decoded,
        )

    invalid_g = b"not-a-valid-G-packet"
    a_packet = adapted.consumed.packet
    forged_g_sections = (
        CountedProgramSection(
            role=CountedSectionRole.GENERATIVE_CORRECTION,
            offset=0,
            byte_length=len(invalid_g),
            payload_sha256=_sha256(invalid_g),
            provenance=adapted.section_manifest[0].provenance,
        ),
        CountedProgramSection(
            role=A_SECTION_ROLE,
            offset=len(invalid_g),
            byte_length=len(a_packet),
            payload_sha256=_sha256(a_packet),
            provenance=adapted.section_manifest[1].provenance,
        ),
    )
    with pytest.raises(CoupledPreimagePairAdapterError, match="counted G packet failed strict reapply"):
        consume_counted_coupled_preimage(
            invalid_g + a_packet,
            forged_g_sections,
            pair_population=_population(),
            predictor_state=state,
            decoded_g=compiled_g.decoded,
        )


def test_exact_y1_mismatch_fails_before_y0_materializer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state, compiled_g, _compiled_a0, _artifact_path, adapted = _adapt(
        tmp_path,
        CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
    )
    called = False

    def forbidden_materializer(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("Y0 materializer ran before exact Y1 validation")

    monkeypatch.setattr(coupled_preimage_module, "_materialize_y0", forbidden_materializer)
    envelope = json.loads(adapted.consumed.packet.decode("ascii"))
    forged_y1_sha256 = "f" * 64
    envelope["body"]["source_binding"]["exact_y1_content_sha256"] = forged_y1_sha256
    for item in envelope["body"]["lineage"]:
        if item["role"] == "EXACT_FRAME1_UINT8":
            item["content_sha256"] = forged_y1_sha256
    envelope["body_sha256"] = _sha256(canonical_json_bytes(envelope["body"]))
    forged_packet = canonical_json_bytes(envelope)
    forged_program, forged_sections = _raw_a_variant(adapted, forged_packet)
    with pytest.raises(CoupledPreimagePairAdapterError, match="strict reverse-causal consumption"):
        consume_counted_coupled_preimage(
            forged_program,
            forged_sections,
            pair_population=_population(),
            predictor_state=state,
            decoded_g=compiled_g.decoded,
        )
    assert called is False


def test_source_artifact_mutation_is_refused(tmp_path: Path) -> None:
    state = _predictor_state()
    compiled_g = _compiled_g(state)
    compiled_a = _compiled_a(
        state,
        compiled_g.decoded,
        CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
    )
    path = tmp_path / "mutated.a.json"
    path.write_bytes(compiled_a.packet)
    custody = _custody(path)
    path.write_bytes(compiled_a.packet + b"\n")
    g_bytes, g_sections = _g_prefix(state, compiled_g.packet)
    with pytest.raises(CoupledPreimagePairAdapterError, match="custody identity"):
        append_source_custodied_coupled_preimage(
            g_bytes,
            g_sections,
            source_custody=custody,
            pair_population=_population(),
            predictor_state=state,
            decoded_g=compiled_g.decoded,
        )


def test_source_artifact_leaf_symlink_is_refused(tmp_path: Path) -> None:
    state = _predictor_state()
    compiled_g = _compiled_g(state)
    compiled_a = _compiled_a(
        state,
        compiled_g.decoded,
        CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
    )
    target = tmp_path / "source.a.json"
    target.write_bytes(compiled_a.packet)
    link = tmp_path / "source-link.a.json"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("leaf symlinks are unavailable on this platform")

    with pytest.raises(CoupledPreimagePairAdapterError, match="regular non-symlink"):
        _custody(link)


def test_producer_source_leaf_symlink_is_refused(tmp_path: Path) -> None:
    state = _predictor_state()
    compiled_g = _compiled_g(state)
    compiled_a = _compiled_a(
        state,
        compiled_g.decoded,
        CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
    )
    artifact = tmp_path / "source.a.json"
    artifact.write_bytes(compiled_a.packet)
    target = Path(coupled_preimage_module.__file__)
    link = tmp_path / "producer-source.py"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("leaf symlinks are unavailable on this platform")

    with pytest.raises(CoupledPreimagePairAdapterError, match="regular non-symlink"):
        CoupledPreimageSourceCustodyV1.reopen(
            artifact,
            producer_source_path=link,
            producer_id="test.coupled_preimage_pair_adapter_fixture",
            payload_lineage=CountedPayloadLineage.ORIGINAL_OWN,
            video_derived=True,
        )


def test_producer_source_mutation_is_refused_by_custody_reopen(tmp_path: Path) -> None:
    state = _predictor_state()
    compiled_g = _compiled_g(state)
    compiled_a = _compiled_a(
        state,
        compiled_g.decoded,
        CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
    )
    artifact = tmp_path / "source.a.json"
    artifact.write_bytes(compiled_a.packet)
    producer_source = tmp_path / "producer-source.py"
    producer_source.write_bytes(Path(coupled_preimage_module.__file__).read_bytes())
    custody = CoupledPreimageSourceCustodyV1.reopen(
        artifact,
        producer_source_path=producer_source,
        producer_id="test.coupled_preimage_pair_adapter_fixture",
        payload_lineage=CountedPayloadLineage.ORIGINAL_OWN,
        video_derived=True,
    )
    producer_source.write_bytes(producer_source.read_bytes() + b"\n")

    with pytest.raises(CoupledPreimagePairAdapterError, match="custody identity"):
        custody.reopen_packet()


def test_descriptor_snapshot_refuses_leaf_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _predictor_state()
    compiled_g = _compiled_g(state)
    compiled_a = _compiled_a(
        state,
        compiled_g.decoded,
        CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
    )
    artifact = tmp_path / "source.a.json"
    artifact.write_bytes(compiled_a.packet)
    replacement = tmp_path / "replacement.a.json"
    replacement.write_bytes(compiled_a.packet)
    real_open = os.open

    def open_then_replace(path: os.PathLike[str] | str, flags: int) -> int:
        fd = real_open(path, flags)
        Path(path).unlink()
        Path(path).symlink_to(replacement)
        return fd

    monkeypatch.setattr(os, "open", open_then_replace)
    with pytest.raises(CoupledPreimagePairAdapterError, match="mutated during reopen"):
        _custody(artifact)


def test_portable_no_nofollow_fallback_refuses_preopen_leaf_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = _predictor_state()
    compiled_g = _compiled_g(state)
    compiled_a = _compiled_a(
        state,
        compiled_g.decoded,
        CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
    )
    artifact = tmp_path / "source.a.json"
    artifact.write_bytes(compiled_a.packet)
    replacement = tmp_path / "replacement.a.json"
    replacement.write_bytes(compiled_a.packet)
    real_open = os.open

    def replace_then_open(path: os.PathLike[str] | str, flags: int) -> int:
        Path(path).unlink()
        Path(path).symlink_to(replacement)
        return real_open(path, flags)

    monkeypatch.setattr(os, "O_NOFOLLOW", 0, raising=False)
    monkeypatch.setattr(os, "open", replace_then_open)
    with pytest.raises(CoupledPreimagePairAdapterError, match="mutated during reopen"):
        _custody(artifact)


def test_reverse_causal_adapter_refuses_legacy_prefix_while_legacy_pair_remains_readable(tmp_path: Path) -> None:
    state, compiled_g, _compiled_a0, _artifact_path, adapted = _adapt(
        tmp_path,
        CoupledPreimageMode.JOINT_SHARED_SKELETON_TWO_FIBRE,
    )
    terminal = canonical_json_bytes({"schema": "test.terminal.v1", "bias": 0})
    terminal_section = CountedProgramSection(
        role=CountedSectionRole.TERMINAL_QUOTIENT,
        offset=len(adapted.program_bytes),
        byte_length=len(terminal),
        payload_sha256=_sha256(terminal),
        provenance=_provenance(CountedSectionRole.TERMINAL_QUOTIENT, state),
    )
    validate_reverse_causal_counted_program_sections(
        adapted.program_bytes + terminal,
        (*adapted.section_manifest, terminal_section),
    )

    legacy_frame1 = canonical_json_bytes({"schema": "test.legacy.frame1.v1"})
    legacy_frame0 = canonical_json_bytes({"schema": "test.legacy.frame0.v1"})
    g = adapted.program_bytes[: adapted.section_manifest[0].stop]
    legacy_program = g + legacy_frame1 + legacy_frame0
    producer_sha = _sha256(Path(__file__).read_bytes())
    legacy_provenance = CountedPayloadProvenance(
        artifact_class=CountedArtifactClass.FRAME_PREIMAGE_PARAMETERS,
        lineage=CountedPayloadLineage.ORIGINAL_OWN,
        producer_id="test.legacy",
        producer_source_sha256=producer_sha,
        derivation_input_sha256=state.binding_sha256,
        video_derived=True,
    )
    legacy_pose_provenance = replace(
        legacy_provenance,
        artifact_class=CountedArtifactClass.POSE_RESIDUAL_PARAMETERS,
    )
    legacy_sections = (
        adapted.section_manifest[0],
        CountedProgramSection(
            CountedSectionRole.FRAME1_PREIMAGE,
            len(g),
            len(legacy_frame1),
            _sha256(legacy_frame1),
            legacy_provenance,
        ),
        CountedProgramSection(
            CountedSectionRole.FRAME0_POSE_RESIDUAL,
            len(g) + len(legacy_frame1),
            len(legacy_frame0),
            _sha256(legacy_frame0),
            legacy_pose_provenance,
        ),
    )

    with pytest.raises(CoupledPreimagePairAdapterError, match="requires exactly counted G"):
        append_source_custodied_coupled_preimage(
            g + legacy_frame1,
            legacy_sections[:2],
            source_custody=adapted.source_custody,
            pair_population=_population(),
            predictor_state=state,
            decoded_g=compiled_g.decoded,
        )

    validate_counted_program_sections(legacy_program, legacy_sections)
