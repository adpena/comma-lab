# SPDX-License-Identifier: MIT
from __future__ import annotations

import fcntl
import hashlib
from pathlib import Path

import numpy as np
import pytest

from tac.witness_dsl import c0b_counted_receiver_codec as counted_codec_module
from tac.witness_dsl.c0b_counted_receiver_codec import (
    SECTION_ORDER,
    CountedReceiverCodecError,
    SparseRGBOverride,
    build_counted_receiver_archive,
    decode_counted_receiver_archive,
    parse_counted_receiver_archive,
    parse_counted_receiver_packet,
    write_counted_receiver_raw,
)
from tac.witness_dsl.factorized_v9_predictor import receive_factorized_v9_predictor
from tac.witness_dsl.tests.test_factorized_v9_predictor import _program


@pytest.fixture(scope="module")
def counted_inputs() -> tuple[bytes, bytes, np.ndarray, tuple[int, ...]]:
    program = _program()
    receiver = receive_factorized_v9_predictor(program)
    target = receiver.decode_all_semantics()
    target[0, 100, 100] = 0
    pbr1 = receiver.build_pbr1(target)
    palette = np.asarray(
        [
            [
                [[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42], [50, 51, 52]],
                [[110, 111, 112], [120, 121, 122], [130, 131, 132], [140, 141, 142], [150, 151, 152]],
            ]
        ],
        dtype=np.uint8,
    )
    source_pair_ids = tuple(
        getattr(
            receiver,
            "source_pair_ids",
            range(
                int(receiver.receiver.predictor.source_pair_start),
                int(receiver.receiver.predictor.source_pair_start) + receiver.pair_count,
            ),
        )
    )
    return program, pbr1, palette, source_pair_ids


def _archive(
    counted_inputs: tuple[bytes, bytes, np.ndarray, tuple[int, ...]],
    *,
    pbr1: bytes | None = None,
    palette: np.ndarray | None = None,
    pair_ids: tuple[int, ...] | None = None,
    overrides: tuple[SparseRGBOverride, ...] = (),
) -> bytes:
    program, default_pbr1, default_palette, default_pair_ids = counted_inputs
    return build_counted_receiver_archive(
        predictor_program=program,
        pbr1=default_pbr1 if pbr1 is None else pbr1,
        pair_ids=default_pair_ids if pair_ids is None else pair_ids,
        palette=default_palette if palette is None else palette,
        overrides=overrides,
    )


def test_n1_structural_receiver_is_byte_closed_and_double_decode_identical(
    counted_inputs: tuple[bytes, bytes, np.ndarray, tuple[int, ...]],
) -> None:
    archive = _archive(counted_inputs)
    parsed = parse_counted_receiver_archive(archive)
    first = decode_counted_receiver_archive(archive)
    second = decode_counted_receiver_archive(archive)

    assert first.raw == second.raw
    assert first.receipt == second.receipt
    assert first.receipt.archive_sha256 == hashlib.sha256(archive).hexdigest()
    assert first.receipt.raw_sha256 == hashlib.sha256(first.raw).hexdigest()
    assert first.receipt.raw_bytes == 1 * 2 * 874 * 1164 * 3
    assert first.receipt.factor2_certified_exact is True
    assert first.receipt.factor2_scorer_values_verified == 1 * 2 * 384 * 512 * 3
    assert first.receipt.factor2_numerator_values_verified == 1 * 2 * 384 * 512 * 3
    assert first.receipt.exact_archive_parse_back is True
    assert first.receipt.codec_role == "abi_causality_ablation"
    assert first.receipt.capstone_eligible is False
    assert first.receipt.shared_semantic_partition_across_planes is True
    assert first.receipt.repository_decoder_dependency is True
    assert first.receipt.standalone_inflate_source_custody is False
    assert first.receipt.standalone_inflate_owed is True
    assert first.receipt.authority_receipt_owed is True
    assert first.receipt.research_only is True
    assert first.receipt.score_claim is False
    assert first.receipt.promotion_eligible is False
    assert parsed.header["separate_dense_target_table_section_bytes"] == 0
    assert parsed.header["dense_y_table_bytes"] == 0
    assert parsed.header["decode_scorer_dependency"] is False
    assert parsed.header["pbr1_is_target_derived"] is True
    assert parsed.header["codec_role"] == "abi_causality_ablation"
    assert parsed.header["capstone_eligible"] is False
    assert parsed.header["shared_semantic_partition_across_planes"] is True
    assert first.receipt.pbr1_counted_bytes == first.receipt.section_bytes["pbr1"]
    assert first.receipt.pbr1_target_derived_section_bytes == first.receipt.pbr1_counted_bytes
    assert first.receipt.pbr1_event_count == 1
    assert first.receipt.pbr1_event_density_numerator == 1
    assert first.receipt.pbr1_event_density_denominator == 384 * 512
    assert first.receipt.separate_dense_target_table_section_bytes == 0
    assert first.receipt.target_derived_residual_promotion_admitted is False
    assert first.receipt.exact_target_semantic_reconstruction is True
    assert first.receipt.candidate_payload_allowed is False
    assert first.receipt.candidate_archive_blocker == (
        "contains lossless predictor-conditional target-semantic-table residual"
    )
    assert first.receipt.pbr1_nested_semantic_residual_bytes == parsed.header["pbr1_nested_semantic_residual_bytes"]
    assert first.receipt.archive_container_bytes + first.receipt.packet_bytes == len(archive)
    assert (
        first.receipt.packet_framing_and_header_bytes + sum(first.receipt.section_bytes.values())
        == first.receipt.packet_bytes
    )
    assert parsed.header["pair_ids"] == [0]
    assert parsed.header["evaluator_obligation_ir_bound"] is False
    assert parsed.header["coupled_witness_state_bound"] is False
    assert parsed.header["hard_oracle_admission_bound"] is False
    assert parsed.header["independent_frame0_preimage"] is False
    assert first.receipt.evaluator_obligation_ir_bound is False
    assert first.receipt.coupled_witness_state_bound is False
    assert first.receipt.hard_oracle_admission_bound is False
    assert first.receipt.independent_frame0_preimage is False
    assert tuple(first.receipt.section_bytes) == SECTION_ORDER
    assert sum(first.receipt.section_bytes.values()) == parsed.header["section_payload_bytes"]


def test_each_counted_section_is_causally_checked_and_pair_swap_refuses(
    counted_inputs: tuple[bytes, bytes, np.ndarray, tuple[int, ...]],
) -> None:
    archive = _archive(counted_inputs)
    parsed = parse_counted_receiver_archive(archive)
    for section in parsed.sections:
        offset = parsed.packet.find(section)
        assert offset >= 0
        mutated = bytearray(parsed.packet)
        mutated[offset + len(section) // 2] ^= 1
        with pytest.raises(CountedReceiverCodecError, match="section hash"):
            parse_counted_receiver_packet(bytes(mutated))

    with pytest.raises(CountedReceiverCodecError, match="ordered pair population"):
        _archive(counted_inputs, pair_ids=(1,))


def test_valid_pbr1_alternative_causally_changes_target_and_raw(
    counted_inputs: tuple[bytes, bytes, np.ndarray, tuple[int, ...]],
) -> None:
    program = counted_inputs[0]
    receiver = receive_factorized_v9_predictor(program)
    zero_pbr1 = receiver.build_pbr1(receiver.decode_all_semantics())

    corrected = decode_counted_receiver_archive(_archive(counted_inputs))
    uncorrected = decode_counted_receiver_archive(_archive(counted_inputs, pbr1=zero_pbr1))

    assert corrected.receipt.predictor_semantic_sha256 == uncorrected.receipt.predictor_semantic_sha256
    assert corrected.receipt.section_sha256["pbr1"] != uncorrected.receipt.section_sha256["pbr1"]
    assert corrected.receipt.target_semantic_sha256 != uncorrected.receipt.target_semantic_sha256
    assert corrected.receipt.raw_sha256 != uncorrected.receipt.raw_sha256
    assert corrected.receipt.pbr1_event_count == 1
    assert uncorrected.receipt.pbr1_event_count == 0


def test_palette_and_sparse_absolute_rgb_override_change_realized_raw(
    counted_inputs: tuple[bytes, bytes, np.ndarray, tuple[int, ...]],
) -> None:
    baseline_archive = _archive(counted_inputs)
    baseline = decode_counted_receiver_archive(baseline_archive)

    palette = counted_inputs[2].copy()
    palette[0, 0, 2, 0] += 1
    palette_result = decode_counted_receiver_archive(_archive(counted_inputs, palette=palette))
    assert palette_result.receipt.target_semantic_sha256 == baseline.receipt.target_semantic_sha256
    assert palette_result.receipt.scorer_rgb_sha256 != baseline.receipt.scorer_rgb_sha256
    assert palette_result.receipt.raw_sha256 != baseline.receipt.raw_sha256

    override = SparseRGBOverride(0, 1, 0, 0, 201, 202, 203)
    override_result = decode_counted_receiver_archive(_archive(counted_inputs, overrides=(override,)))
    assert override_result.receipt.target_semantic_sha256 == baseline.receipt.target_semantic_sha256
    assert override_result.receipt.scorer_rgb_sha256 != baseline.receipt.scorer_rgb_sha256
    assert override_result.receipt.raw_sha256 != baseline.receipt.raw_sha256


def test_streaming_writer_emits_the_same_raw_and_durable_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    counted_inputs: tuple[bytes, bytes, np.ndarray, tuple[int, ...]],
) -> None:
    archive = _archive(counted_inputs)
    expected = decode_counted_receiver_archive(archive)
    output = tmp_path / "camera.raw"
    receipt = write_counted_receiver_raw(archive, output)
    assert output.read_bytes() == expected.raw
    assert receipt.raw_sha256 == expected.receipt.raw_sha256
    assert output.with_name("camera.raw.resume.json").is_file()
    assert not output.with_name("camera.raw.partial").exists()

    # Recreate the completed pair-boundary checkpoint and prove continuation
    # verifies/reuses it instead of silently accepting unrelated partial bytes.
    partial = output.with_name("camera.raw.partial")
    output.replace(partial)
    original = counted_codec_module._atomic_write_resume_state
    resumed_states: list[dict[str, object]] = []

    def record_state(path: Path, state: object) -> None:
        assert isinstance(state, dict)
        resumed_states.append(dict(state))
        original(path, state)

    monkeypatch.setattr(counted_codec_module, "_atomic_write_resume_state", record_state)
    resumed = write_counted_receiver_raw(archive, output)
    assert resumed.raw_sha256 == receipt.raw_sha256
    assert output.read_bytes() == expected.raw
    assert len(resumed_states) == 1
    assert resumed_states[0]["completed"] is True


def test_streaming_writer_recovers_raw_fsync_before_first_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    counted_inputs: tuple[bytes, bytes, np.ndarray, tuple[int, ...]],
) -> None:
    archive = _archive(counted_inputs)
    expected = decode_counted_receiver_archive(archive)
    output = tmp_path / "crash.raw"
    original = counted_codec_module._atomic_write_resume_state

    def fail_after_raw_fsync(_path: Path, _state: object) -> None:
        raise RuntimeError("fault after raw fsync")

    monkeypatch.setattr(counted_codec_module, "_atomic_write_resume_state", fail_after_raw_fsync)
    with pytest.raises(RuntimeError, match="fault after raw fsync"):
        write_counted_receiver_raw(archive, output)
    partial = output.with_name("crash.raw.partial")
    assert partial.stat().st_size == len(expected.raw)
    assert not output.with_name("crash.raw.resume.json").exists()

    monkeypatch.setattr(counted_codec_module, "_atomic_write_resume_state", original)
    resumed = write_counted_receiver_raw(archive, output)
    assert output.read_bytes() == expected.raw
    assert resumed.raw_sha256 == expected.receipt.raw_sha256
    assert not partial.exists()


def test_streaming_writer_refuses_concurrent_output_owner(
    tmp_path: Path,
    counted_inputs: tuple[bytes, bytes, np.ndarray, tuple[int, ...]],
) -> None:
    archive = _archive(counted_inputs)
    output = tmp_path / "owned.raw"
    lock_path = output.with_name("owned.raw.lock")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(CountedReceiverCodecError, match="another counted receiver writer"):
                write_counted_receiver_raw(archive, output)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
