from __future__ import annotations

from pathlib import Path

from experiments import ddm_mz1_model_section_rate_race as mz1
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def test_mz1_files_pass_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=Path.cwd(),
        strict=False,
        roots=(
            "experiments/ddm_mz1_model_section_rate_race.py",
            "experiments/tests/test_ddm_mz1_model_section_rate_race.py",
        ),
    )
    assert findings == []


def test_mzc1_identity_container_roundtrip(tmp_path: Path) -> None:
    segments = [
        mz1.LogicalSegment("header", mz1.BYTE_SEGMENT, 4, b"IHS1"),
        mz1.LogicalSegment("bits", mz1.BIT_SEGMENT, 11, b"\xaa\x07"),
    ]
    encoded = [mz1.EncodedSegment(segment, "identity", segment.data) for segment in segments]
    container = mz1._pack_container(encoded)

    decoded = mz1._unpack_container(container, (None, None, tmp_path / "unused"))

    assert [(item.kind, item.units, item.data) for item in decoded] == [
        (item.kind, item.units, item.data) for item in segments
    ]


def test_ihs1_rebuild_preserves_partial_final_byte() -> None:
    segments = [
        mz1.LogicalSegment("magic", mz1.BYTE_SEGMENT, 4, b"IHS1"),
        mz1.LogicalSegment("depths", mz1.BYTE_SEGMENT, 2, b"\x01\x02"),
        mz1.LogicalSegment("weight", mz1.BIT_SEGMENT, 11, b"\xaa\x07"),
        mz1.LogicalSegment("fixed", mz1.BYTE_SEGMENT, 2, b"\x03\x04"),
    ]

    assert mz1._rebuild_ihs1(segments) == b"IHS1\x01\x02\xaa\x07\x03\x04"


def test_byte_map_q11_roundtrip() -> None:
    source = bytes(range(256)) * 3 + b"model-section"

    payload = mz1._byte_map_encode(source)

    assert mz1._byte_map_decode(payload, len(source)) == source
