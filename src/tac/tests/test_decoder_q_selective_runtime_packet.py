# SPDX-License-Identifier: MIT
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tac.optimization import decoder_q_selective_runtime_packet as packet
from tac.optimization.fec6_byte_targets import ByteRange, Fec6Section
from tac.optimization.fec6_decoder_mutations import (
    DecoderMutationResult,
    DecoderQTensorSpan,
)


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }


def _write_zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("x", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def test_dqs1_trailer_payload_round_trip() -> None:
    payload = packet.pack_dqs1_payload(
        pair_indices=[2, 5, 9],
        frame_policy="pair_all_frames",
        storage_index=7,
        q_offset=123,
        delta=-1,
    )

    assert payload[:4] == b"DQS1"
    assert packet.unpack_dqs1_payload(payload) == {
        "frame_policy": "pair_all_frames",
        "frame_policy_code": 1,
        "mode_byte": 1,
        "pair_encoding": "raw_u16",
        "pair_encoding_code": 0,
        "storage_index": 7,
        "q_offset": 123,
        "delta": -1,
        "pair_indices": [2, 5, 9],
    }


def test_dqs1_payload_supports_sorted_gap_uleb_pair_encoding() -> None:
    payload = packet.pack_dqs1_payload(
        pair_indices=[2, 5, 9],
        frame_policy="pair_all_frames",
        storage_index=7,
        q_offset=123,
        delta=-1,
        pair_encoding="sorted_gap_uleb",
    )

    assert payload[:4] == b"DQS1"
    assert payload[4] == 0x11
    assert len(payload) == 14
    assert packet.unpack_dqs1_payload(payload) == {
        "frame_policy": "pair_all_frames",
        "frame_policy_code": 1,
        "mode_byte": 0x11,
        "pair_encoding": "sorted_gap_uleb",
        "pair_encoding_code": 1,
        "storage_index": 7,
        "q_offset": 123,
        "delta": -1,
        "pair_indices": [2, 5, 9],
    }


def test_dqs1_pair_encoding_selector_prefers_smallest_then_raw_compatibility() -> None:
    compact = packet.choose_dqs1_pair_encoding([2, 5, 9])
    assert compact["selected"] == {
        "pair_encoding": "sorted_gap_uleb",
        "pair_encoding_code": 1,
        "pair_index_payload_bytes": 3,
        "descriptor_bytes": 14,
    }

    raw_tie = packet.choose_dqs1_pair_encoding([599])
    assert raw_tie["selected"] == {
        "pair_encoding": "raw_u16",
        "pair_encoding_code": 0,
        "pair_index_payload_bytes": 2,
        "descriptor_bytes": 13,
    }


@pytest.mark.parametrize("pairs", ([3, 1], [1, 1]))
def test_dqs1_payload_rejects_noncanonical_pair_order(pairs: list[int]) -> None:
    with pytest.raises(packet.DecoderQSelectiveRuntimePacketError, match="pair indices"):
        packet.pack_dqs1_payload(
            pair_indices=pairs,
            frame_policy="pair_all_frames",
            storage_index=0,
            q_offset=0,
            delta=1,
        )


def test_packet_plan_preserves_false_authority_and_trailer_byte_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_member = b"aaaaaaaa"
    candidate_member = b"baaaaaaa"
    base_archive = tmp_path / "base.zip"
    candidate_archive = tmp_path / "candidate.zip"
    _write_zip(base_archive, base_member)
    _write_zip(candidate_archive, candidate_member)

    source_sha = packet.sha256_bytes(base_member)
    mutated_sha = packet.sha256_bytes(candidate_member)

    def fake_parse_fec6_sections(_payload: bytes, **_kwargs: object) -> list[Fec6Section]:
        return [Fec6Section("decoder", ByteRange(0, len(base_member)), "fixture", "fixture")]

    def fake_probe(_prepared: object, mutation: object) -> DecoderMutationResult:
        return DecoderMutationResult(
            mutation=mutation,  # type: ignore[arg-type]
            tensor=DecoderQTensorSpan(
                name="rgb.0.weight",
                storage_index=3,
                storage_position=0,
                shape=(8,),
                numel=8,
                byte_map="zig",
                stream_index=0,
                raw_q_range=ByteRange(0, 8),
                raw_scale_range=ByteRange(8, 10),
            ),
            q_before=4,
            q_after=5,
            source_decoder_len=len(base_member),
            mutated_decoder_len=len(candidate_member),
            source_decoder_sha256=source_sha,
            mutated_decoder_sha256=mutated_sha,
            fixed_length_runtime_compatible=True,
        )

    monkeypatch.setattr(packet, "extract_fec6_decoder_blob", lambda payload: payload)
    monkeypatch.setattr(packet, "prepare_decoder_blob", lambda payload: {"payload": payload})
    monkeypatch.setattr(packet, "probe_q_mutation", fake_probe)
    monkeypatch.setattr(packet, "parse_fec6_sections", fake_parse_fec6_sections)

    bridge_plan = {
        "schema": packet.BRIDGE_SCHEMA,
        **_false_authority(),
        "candidate_generation_only": True,
        "evidence_grade": "macOS-MLX-research-signal",
        "materialized_decoder_q_candidate": {
            "archive_zip_path": str(candidate_archive),
            "mutation": {
                "tensor_name": "rgb.0.weight",
                "q_offset": 0,
                "delta": 1,
                "source_decoder_sha256": source_sha,
                "mutated_decoder_sha256": mutated_sha,
            },
        },
        "work_units": [
            {
                **_false_authority(),
                "pair_window": [2, 3],
                "observed_mlx_gain": 0.002,
            },
            {
                **_false_authority(),
                "pair_window": [5, 6],
                "observed_mlx_gain": 0.001,
            },
        ],
    }

    plan = packet.build_decoder_q_selective_runtime_packet_plan(
        bridge_plan,
        base_archive=base_archive,
        repo_root=tmp_path,
    )

    assert plan["score_claim"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["selective_packet"]["wire_format"].startswith(
        "member = legacy_FP11_FEC6_member || DQS1_trailer"
    )
    assert plan["selective_packet"]["selected_pair_indices"] == [2, 5]
    assert plan["selective_packet"]["pair_encoding"] == "sorted_gap_uleb"
    assert plan["selective_packet"]["pair_index_payload_bytes"] == 2
    assert plan["selective_packet"]["payload_bytes"] == 13
    assert plan["selective_packet"]["pair_encoding_candidates"] == [
        {
            "pair_encoding": "raw_u16",
            "pair_encoding_code": 0,
            "pair_index_payload_bytes": 4,
            "descriptor_bytes": 15,
        },
        {
            "pair_encoding": "sorted_gap_uleb",
            "pair_encoding_code": 1,
            "pair_index_payload_bytes": 2,
            "descriptor_bytes": 13,
        },
    ]
    assert plan["selective_packet"]["wrapper_header_bytes"] == 0
    assert plan["selective_packet"]["estimated_archive_byte_delta_if_appended_to_member"] == 13
    assert (
        "packet plan must be materialized with selective runtime adapter before use"
        in plan["dispatch_blockers"]
    )
    assert "decode the full same local batch with the mutated decoder" in plan[
        "runtime_adapter_contract"
    ]["batch_decode_strategy"]
