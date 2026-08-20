# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import (
    ReceiverRealizationProfileV1,
    TopologyEventV1,
)
from tac.witness_dsl import generative_taskspace_receiver as receiver_module
from tac.witness_dsl.generative_taskspace_correction import (
    DecodedGenerativeCorrectionV1,
    EncoderOnlyTeacherEvidenceV1,
    GenerativeCorrectionProgramV1,
    PredictorSemanticStateV1,
    compile_generative_taskspace_correction,
    parse_generative_taskspace_correction,
)
from tac.witness_dsl.generative_taskspace_receiver import (
    ARCHIVE_MEMBER_NAME,
    CountedTaskspaceRole,
    GenerativeTaskspaceReceiverError,
    GReceiverConsumptionProofV1,
    build_g_only_counterfactual_fragment_archive,
    build_generative_taskspace_receiver_fragment_archive,
    decode_generative_taskspace_fragment_to_y1_raw,
    encode_counted_taskspace_payload,
    parse_counted_taskspace_payload,
    parse_generative_taskspace_receiver_fragment_archive,
    prove_generative_taskspace_receiver_consumption,
    verify_generative_taskspace_consumption_proof,
    write_generative_taskspace_consumption_proof,
)

_A_PACKET = b"TACA-adapter-owned-structural-fixture-v1:" + bytes(range(64))
_T_PACKET = b"TACT-terminal-quotient-structural-fixture-v1:" + bytes(reversed(range(64)))


def _state() -> PredictorSemanticStateV1:
    labels = np.full((1, 384, 512), 2, dtype=np.uint8)
    labels[:, 210:, :] = 0
    labels[:, 275:290, 230:270] = 3
    return PredictorSemanticStateV1(
        predictor_program_sha256="a" * 64,
        predictor_renderer_sha256="b" * 64,
        source_pair_ids=(0,),
        labels=labels,
        pose6_codes=np.zeros((1, 6), dtype=np.int16),
    )


def _profile() -> ReceiverRealizationProfileV1:
    return ReceiverRealizationProfileV1(
        (
            (20, 80, 20),
            (240, 220, 40),
            (30, 30, 30),
            (220, 40, 40),
            (40, 80, 220),
        )
    )


def _teacher(state: PredictorSemanticStateV1) -> EncoderOnlyTeacherEvidenceV1:
    target = state.labels.copy()
    target[0, 0, 0] = 4
    target = np.ascontiguousarray(target)
    return EncoderOnlyTeacherEvidenceV1(
        pbr1_sha256="1" * 64,
        pbr2_sha256="2" * 64,
        target_labels_sha256=hashlib.sha256(memoryview(target).cast("B")).hexdigest(),
        obligation_ir_sha256="4" * 64,
        oracle_evidence_sha256="5" * 64,
        dense_y_sha256="6" * 64,
        target_labels=target,
        teacher_event_count=1,
    )


def _packet(*, y0: int, x0: int) -> bytes:
    state = _state()
    program = GenerativeCorrectionProgramV1(
        topology_events=(
            TopologyEventV1(
                0,
                "Lane",
                "birth",
                "box",
                1,
                y0,
                x0,
                y0 + 12,
                x0 + 18,
            ),
        ),
        realization_profile=_profile(),
    )
    return compile_generative_taskspace_correction(
        state,
        program,
        teacher_evidence=_teacher(state),
    ).packet


def _archives() -> tuple[bytes, bytes]:
    state = _state()
    baseline = build_generative_taskspace_receiver_fragment_archive(
        _packet(y0=40, x0=30),
        predictor_state=state,
        coupled_preimage_packet=_A_PACKET,
        terminal_quotient_packet=_T_PACKET,
    )
    counterfactual = build_g_only_counterfactual_fragment_archive(
        baseline,
        _packet(y0=80, x0=100),
        predictor_state=state,
    )
    return baseline, counterfactual


def test_canonical_counted_archive_strictly_consumes_g_and_double_decodes_raw(tmp_path: Path) -> None:
    state = _state()
    baseline, _counterfactual = _archives()
    parsed = parse_generative_taskspace_receiver_fragment_archive(baseline)
    roles = tuple(row.role for row in parsed.counted.sections)
    assert roles == (
        CountedTaskspaceRole.GENERATIVE_CORRECTION,
        CountedTaskspaceRole.COUPLED_PREIMAGE,
        CountedTaskspaceRole.TERMINAL_QUOTIENT,
    )
    assert parsed.counted.section(CountedTaskspaceRole.COUPLED_PREIMAGE).payload == _A_PACKET
    assert parsed.counted.section(CountedTaskspaceRole.TERMINAL_QUOTIENT).payload == _T_PACKET

    output = tmp_path / "baseline_y1.raw"
    first = decode_generative_taskspace_fragment_to_y1_raw(
        baseline,
        predictor_state=state,
        output_path=output,
    )
    second = decode_generative_taskspace_fragment_to_y1_raw(
        baseline,
        predictor_state=state,
        output_path=output,
    )

    assert first.receipt == second.receipt
    assert np.array_equal(first.decoded.labels, second.decoded.labels)
    assert first.receipt.archive_sha256 == hashlib.sha256(baseline).hexdigest()
    assert first.receipt.rate_charged_archive_bytes == len(baseline)
    assert first.receipt.generic_receiver_code_counted_bytes == 0
    assert first.receipt.g_section_sha256 == first.decoded.correction_packet_sha256
    assert first.receipt.predictor_binding_sha256 == state.binding_sha256
    assert first.receipt.predictor_state_contract == "PredictorSemanticStateV1_with_explicit_Pose6_transport.v1"
    assert first.receipt.pose6_serialized_in_g is False
    assert first.receipt.levelset_latents_relabelled_as_pose6 is False
    assert first.receipt.ep725_frame1_predictor_compatible is False
    assert first.receipt.counted_temporal_coordinate_adapter_owed is True
    assert first.receipt.changed_semantic_cells > 0
    assert first.receipt.realized_y1_raw_bytes == 874 * 1164 * 3
    assert first.receipt.realized_y1_raw_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert first.receipt.factor2_scorer_values_verified == 384 * 512 * 3
    assert first.receipt.factor2_numerator_values_verified == 384 * 512 * 3
    assert first.receipt.strict_g_parse_back is True
    assert first.receipt.g_applied_to_predictor_state is True
    assert first.receipt.realized_through_factor2_uint8 is True
    assert first.receipt.raw_output_materialized is True
    assert first.receipt.raw_output_role == "realized_frame1_Y1_raw_fragment"
    assert first.receipt.complete_chronological_pair_output is False
    assert first.receipt.non_g_sections_consumed_by_g_receiver is False
    assert first.receipt.literal_counted_sections_ge_32_absent_from_runtime_sources is True
    assert first.receipt.literal_scan_min_section_bytes == 32
    assert first.receipt.complete_data_in_code_lineage_audit is False
    assert first.receipt.repository_decoder_dependency is True
    assert first.receipt.standalone_runtime_closure_bound is False
    assert first.receipt.research_only is True
    assert first.receipt.score_claim is False
    assert first.receipt.originality_claim is False
    assert first.receipt.promotion_eligible is False


def test_g_only_counterfactual_changes_decoded_state_and_actual_raw_with_non_g_fixed(tmp_path: Path) -> None:
    state = _state()
    baseline, counterfactual = _archives()
    baseline_parsed = parse_generative_taskspace_receiver_fragment_archive(baseline)
    counterfactual_parsed = parse_generative_taskspace_receiver_fragment_archive(counterfactual)
    assert tuple(row.payload for row in baseline_parsed.counted.non_g_sections) == tuple(
        row.payload for row in counterfactual_parsed.counted.non_g_sections
    )

    proof = prove_generative_taskspace_receiver_consumption(
        baseline,
        counterfactual,
        predictor_state=state,
        baseline_output_path=tmp_path / "proof_baseline_y1.raw",
        counterfactual_output_path=tmp_path / "proof_counterfactual_y1.raw",
    )
    assert proof.baseline_replay_identical is True
    assert proof.same_runtime_manifest is True
    assert proof.same_predictor_binding is True
    assert proof.target_g_section_only_changed is True
    assert proof.unchanged_non_g_sections is True
    assert proof.decoded_state_changed is True
    assert proof.scorer_y1_changed is True
    assert proof.realized_raw_output_changed is True
    assert proof.archive_member_to_strict_g_parse_apply_to_raw_causal is True
    assert proof.baseline.g_section_sha256 != proof.counterfactual.g_section_sha256
    assert proof.baseline.non_g_manifest_sha256 == proof.counterfactual.non_g_manifest_sha256
    assert proof.baseline.runtime_manifest_sha256 == proof.counterfactual.runtime_manifest_sha256
    assert proof.baseline.decoded_labels_sha256 != proof.counterfactual.decoded_labels_sha256
    assert proof.baseline.scorer_y1_sha256 != proof.counterfactual.scorer_y1_sha256
    assert proof.baseline.realized_y1_raw_sha256 != proof.counterfactual.realized_y1_raw_sha256
    assert proof.candidate_archive_eligible is False
    assert proof.n600_complete_pair_archive_present is False
    assert proof.research_only is True
    assert proof.score_claim is False
    assert proof.originality_claim is False
    assert proof.promotion_eligible is False

    proof_path = tmp_path / "g_receiver_consumption_receipt.json"
    written = write_generative_taskspace_consumption_proof(
        proof,
        proof_path,
        baseline_archive=baseline,
        counterfactual_archive=counterfactual,
        predictor_state=state,
        scratch_root=tmp_path / "writer_replay_scratch",
    )
    assert written == proof
    verified = verify_generative_taskspace_consumption_proof(
        proof_path.read_bytes(),
        baseline,
        counterfactual,
        predictor_state=state,
        scratch_root=tmp_path / "verify_scratch",
    )
    assert verified == proof


def test_strict_container_rejects_truncation_trailing_unknown_duplicate_and_bad_zip() -> None:
    g_packet = _packet(y0=40, x0=30)
    payload = encode_counted_taskspace_payload(
        g_packet,
        coupled_preimage_packet=_A_PACKET,
        terminal_quotient_packet=_T_PACKET,
    )
    with pytest.raises(GenerativeTaskspaceReceiverError, match="truncated"):
        parse_counted_taskspace_payload(payload[:-1])
    with pytest.raises(GenerativeTaskspaceReceiverError, match="trailing"):
        parse_counted_taskspace_payload(payload + b"x")

    prefix_size = struct.calcsize(">8sBBH")
    descriptor_size = struct.calcsize(">BQI32s")
    unknown = bytearray(payload)
    unknown[prefix_size] = 99
    with pytest.raises(GenerativeTaskspaceReceiverError, match="unknown section role"):
        parse_counted_taskspace_payload(bytes(unknown))
    duplicate_g = bytearray(payload)
    duplicate_g[prefix_size + descriptor_size] = 1
    with pytest.raises(GenerativeTaskspaceReceiverError, match="section order"):
        parse_counted_taskspace_payload(bytes(duplicate_g))

    corrupted = bytearray(payload)
    corrupted[-1] ^= 1
    with pytest.raises(GenerativeTaskspaceReceiverError, match="hash or CRC"):
        parse_counted_taskspace_payload(bytes(corrupted))

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(ARCHIVE_MEMBER_NAME, payload)
        archive.writestr("extra.bin", b"forbidden")
    with pytest.raises(GenerativeTaskspaceReceiverError, match="exactly canonical"):
        parse_generative_taskspace_receiver_fragment_archive(output.getvalue())


def test_invalid_or_deleted_g_never_reaches_raw_receiver(tmp_path: Path) -> None:
    state = _state()
    packet = _packet(y0=40, x0=30)
    with pytest.raises(GenerativeTaskspaceReceiverError, match="strict G receiver refused"):
        build_generative_taskspace_receiver_fragment_archive(packet[:-1], predictor_state=state)
    with pytest.raises(GenerativeTaskspaceReceiverError, match="empty, non-bytes, or too large"):
        encode_counted_taskspace_payload(b"")

    baseline, _counterfactual = _archives()
    parsed = parse_generative_taskspace_receiver_fragment_archive(baseline)
    invalid_packet = bytearray(parsed.counted.section(CountedTaskspaceRole.GENERATIVE_CORRECTION).payload)
    invalid_packet[-1] ^= 1
    invalid_payload = encode_counted_taskspace_payload(
        bytes(invalid_packet),
        coupled_preimage_packet=_A_PACKET,
        terminal_quotient_packet=_T_PACKET,
    )
    invalid_archive = receiver_module._canonical_zip(invalid_payload)
    with pytest.raises(GenerativeTaskspaceReceiverError, match="strict parse/apply"):
        decode_generative_taskspace_fragment_to_y1_raw(
            invalid_archive,
            predictor_state=state,
            output_path=tmp_path / "must_not_exist.raw",
        )
    assert not (tmp_path / "must_not_exist.raw").exists()


def test_noop_apply_mutation_cannot_fabricate_receiver_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    baseline, counterfactual = _archives()

    def inert_apply(
        packet: bytes,
        *,
        predictor_state: PredictorSemanticStateV1,
    ) -> DecodedGenerativeCorrectionV1:
        parsed = parse_generative_taskspace_correction(packet, predictor_state=predictor_state)
        return DecodedGenerativeCorrectionV1(
            labels=predictor_state.labels,
            realization_profile=parsed.program.realization_profile,
            correction_packet_sha256=hashlib.sha256(packet).hexdigest(),
        )

    monkeypatch.setattr(receiver_module, "apply_generative_taskspace_correction", inert_apply)
    with pytest.raises(GenerativeTaskspaceReceiverError, match="did not change decoded semantic state"):
        prove_generative_taskspace_receiver_consumption(
            baseline,
            counterfactual,
            predictor_state=state,
            baseline_output_path=tmp_path / "inert_baseline.raw",
            counterfactual_output_path=tmp_path / "inert_counterfactual.raw",
        )


def test_resume_recovers_raw_fsync_before_first_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state()
    baseline, _counterfactual = _archives()
    output = tmp_path / "resumable_y1.raw"
    original = receiver_module._atomic_write

    def fail_checkpoint(path: Path, payload: bytes) -> None:
        if path.name.endswith(".resume.json"):
            raise RuntimeError("fault after raw fsync")
        original(path, payload)

    monkeypatch.setattr(receiver_module, "_atomic_write", fail_checkpoint)
    with pytest.raises(RuntimeError, match="fault after raw fsync"):
        decode_generative_taskspace_fragment_to_y1_raw(
            baseline,
            predictor_state=state,
            output_path=output,
        )
    partial = output.with_name("resumable_y1.raw.partial")
    assert partial.stat().st_size == 874 * 1164 * 3
    assert not output.exists()

    monkeypatch.setattr(receiver_module, "_atomic_write", original)
    resumed = decode_generative_taskspace_fragment_to_y1_raw(
        baseline,
        predictor_state=state,
        output_path=output,
    )
    assert output.stat().st_size == resumed.receipt.realized_y1_raw_bytes
    assert hashlib.sha256(output.read_bytes()).hexdigest() == resumed.receipt.realized_y1_raw_sha256
    assert not partial.exists()


@pytest.mark.parametrize(
    ("surface", "suffix"),
    (
        ("output", ""),
        ("partial", ".partial"),
        ("resume", ".resume.json"),
        ("lock", ".lock"),
    ),
)
def test_receiver_refuses_every_symlinked_output_surface(
    tmp_path: Path,
    surface: str,
    suffix: str,
) -> None:
    state = _state()
    baseline, _counterfactual = _archives()
    output = tmp_path / "safe_y1.raw"
    attacked = output.with_name(f"{output.name}{suffix}")
    victim = tmp_path / f"{surface}.victim"
    victim.write_bytes(b"must-remain-unchanged")
    attacked.symlink_to(victim)

    with pytest.raises(GenerativeTaskspaceReceiverError, match="regular non-symlink file"):
        decode_generative_taskspace_fragment_to_y1_raw(
            baseline,
            predictor_state=state,
            output_path=output,
        )
    assert victim.read_bytes() == b"must-remain-unchanged"
    assert attacked.is_symlink()


@pytest.mark.parametrize("suffix", ("", ".partial", ".resume.json", ".lock"))
def test_receiver_refuses_nonregular_output_surfaces(tmp_path: Path, suffix: str) -> None:
    state = _state()
    baseline, _counterfactual = _archives()
    output = tmp_path / "nonregular_y1.raw"
    attacked = output.with_name(f"{output.name}{suffix}")
    attacked.mkdir()

    with pytest.raises(GenerativeTaskspaceReceiverError, match="regular non-symlink file"):
        decode_generative_taskspace_fragment_to_y1_raw(
            baseline,
            predictor_state=state,
            output_path=output,
        )
    assert attacked.is_dir()


def test_atomic_write_uses_collision_safe_temp_and_cleans_failed_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "receipt.json"
    legacy_fixed_temp = tmp_path / "receipt.json.tmp"
    colliding_unique_shape = tmp_path / ".receipt.json.preexisting.tmp"
    legacy_fixed_temp.write_bytes(b"legacy-collision-sentinel")
    colliding_unique_shape.write_bytes(b"unique-shape-collision-sentinel")

    receiver_module._atomic_write(destination, b"first")
    receiver_module._atomic_write(destination, b"second")
    assert destination.read_bytes() == b"second"
    assert legacy_fixed_temp.read_bytes() == b"legacy-collision-sentinel"
    assert colliding_unique_shape.read_bytes() == b"unique-shape-collision-sentinel"
    assert sorted(tmp_path.glob(".receipt.json.*.tmp")) == [colliding_unique_shape]

    failed_destination = tmp_path / "failed.json"
    original_replace = receiver_module.os.replace

    def fail_install(source: object, target: object) -> None:
        if Path(target) == failed_destination:
            raise OSError("injected atomic install failure")
        original_replace(source, target)

    monkeypatch.setattr(receiver_module.os, "replace", fail_install)
    with pytest.raises(OSError, match="injected atomic install failure"):
        receiver_module._atomic_write(failed_destination, b"never-installed")
    assert not failed_destination.exists()
    assert list(tmp_path.glob(".failed.json.*.tmp")) == []


def test_atomic_write_refuses_symlink_destination(tmp_path: Path) -> None:
    victim = tmp_path / "receipt-victim.json"
    victim.write_bytes(b"must-remain-unchanged")
    destination = tmp_path / "receipt.json"
    destination.symlink_to(victim)
    with pytest.raises(GenerativeTaskspaceReceiverError, match="regular non-symlink file"):
        receiver_module._atomic_write(destination, b"forbidden")
    assert destination.is_symlink()
    assert victim.read_bytes() == b"must-remain-unchanged"


def test_runtime_source_read_refuses_symlink_and_detects_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    victim = tmp_path / "runtime-victim.py"
    victim.write_bytes(b"victim-runtime-source")
    symlink = tmp_path / "runtime-link.py"
    symlink.symlink_to(victim)
    with pytest.raises(GenerativeTaskspaceReceiverError, match="regular non-symlink file"):
        receiver_module._read_stable_regular_bytes(symlink, label="receiver runtime source")

    source = tmp_path / "runtime.py"
    source.write_bytes(b"original-runtime-source")
    replacement = tmp_path / "runtime-replacement.py"
    replacement.write_bytes(b"replaced-runtime-source")
    original_lstat = Path.lstat
    source_lstat_calls = 0

    def replace_before_final_identity_check(path: Path) -> object:
        nonlocal source_lstat_calls
        if path == source:
            source_lstat_calls += 1
            if source_lstat_calls == 3:
                replacement.replace(source)
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", replace_before_final_identity_check)
    with pytest.raises(GenerativeTaskspaceReceiverError, match="changed identity"):
        receiver_module._read_stable_regular_bytes(source, label="receiver runtime source")
    assert source.read_bytes() == b"replaced-runtime-source"


def test_short_counted_literal_has_no_unqualified_data_in_code_absence_claim(tmp_path: Path) -> None:
    state = _state()
    short_runtime_literal = b"research_only"
    assert len(short_runtime_literal) < 32
    assert short_runtime_literal in Path(receiver_module.__file__).read_bytes()
    archive = build_generative_taskspace_receiver_fragment_archive(
        _packet(y0=40, x0=30),
        predictor_state=state,
        coupled_preimage_packet=short_runtime_literal,
    )
    result = decode_generative_taskspace_fragment_to_y1_raw(
        archive,
        predictor_state=state,
        output_path=tmp_path / "short_literal_y1.raw",
    )
    receipt = result.receipt.as_dict()
    assert "literal_counted_payload_absent_from_runtime_sources" not in receipt
    assert receipt["literal_counted_sections_ge_32_absent_from_runtime_sources"] is True
    assert receipt["literal_scan_min_section_bytes"] == 32
    assert receipt["complete_data_in_code_lineage_audit"] is False


def test_public_proof_writer_refuses_caller_constructed_forgery(tmp_path: Path) -> None:
    state = _state()
    baseline, counterfactual = _archives()
    honest = prove_generative_taskspace_receiver_consumption(
        baseline,
        counterfactual,
        predictor_state=state,
        baseline_output_path=tmp_path / "honest_baseline_y1.raw",
        counterfactual_output_path=tmp_path / "honest_counterfactual_y1.raw",
    )
    forged = GReceiverConsumptionProofV1(
        baseline=honest.baseline,
        counterfactual=honest.baseline,
        baseline_replay_identical=True,
        same_runtime_manifest=True,
        same_predictor_binding=True,
        target_g_section_only_changed=True,
        unchanged_non_g_sections=True,
        decoded_state_changed=True,
        scorer_y1_changed=True,
        realized_raw_output_changed=True,
        archive_member_to_strict_g_parse_apply_to_raw_causal=True,
    )
    output = tmp_path / "forged-proof.json"
    with pytest.raises(GenerativeTaskspaceReceiverError, match="differs from fresh behavioral replay"):
        write_generative_taskspace_consumption_proof(
            forged,
            output,
            baseline_archive=baseline,
            counterfactual_archive=counterfactual,
            predictor_state=state,
            scratch_root=tmp_path / "forged_writer_scratch",
        )
    assert not output.exists()


def test_forged_caller_receipt_is_rejected_by_fresh_behavioral_replay(tmp_path: Path) -> None:
    state = _state()
    baseline, counterfactual = _archives()
    proof = prove_generative_taskspace_receiver_consumption(
        baseline,
        counterfactual,
        predictor_state=state,
        baseline_output_path=tmp_path / "forgery_baseline.raw",
        counterfactual_output_path=tmp_path / "forgery_counterfactual.raw",
    )
    envelope = json.loads(proof.to_bytes())
    envelope["body"]["baseline"]["realized_y1_raw_sha256"] = "0" * 64
    envelope["body_sha256"] = hashlib.sha256(
        json.dumps(
            envelope["body"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    forged = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    with pytest.raises(GenerativeTaskspaceReceiverError, match="differs from fresh behavioral replay"):
        verify_generative_taskspace_consumption_proof(
            forged,
            baseline,
            counterfactual,
            predictor_state=state,
            scratch_root=tmp_path / "forgery_scratch",
        )
