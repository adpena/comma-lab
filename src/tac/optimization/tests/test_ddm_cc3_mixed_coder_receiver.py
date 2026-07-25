# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.optimization import ddm_cc3_mixed_coder_receiver as cc3
from tac.optimization import ddm_runtime_receiver as runtime
from tac.optimization.arith_selfcomp_rate_coders import (
    encode_bellard_class_mixing,
    encode_g4_decoder_context,
)
from tac.optimization.ddm_cc2_coder_races import extract_recursive_zip_leaves
from tac.optimization.ddm_pc1_pose_stream import (
    PC1PosePacketV1,
    build_counted_composition_archive,
)


def _stored_zip(members: list[tuple[str, bytes]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, payload)
    return output.getvalue()


def _fixture() -> tuple[bytes, dict[str, object]]:
    parent = _stored_zip([(f"state/leaf_{index:02d}.bin", bytes([65 + index]) * (1024 + index)) for index in range(25)])
    packet = PC1PosePacketV1(
        active=True,
        pair_count=600,
        xi_scales=(1.0,) * 6,
        residual_scale=1.0,
        q_xi=np.zeros((2, 6), dtype=np.int16),
        q_luma_phase=np.zeros((2, 4), dtype=np.int8),
    )
    source = build_counted_composition_archive(
        parent_archive=parent,
        parent_sha256=hashlib.sha256(parent).hexdigest(),
        packet=packet,
    )
    leaves, overhead = extract_recursive_zip_leaves(source)
    assert len(leaves) == 27
    selected_ids = [row.stream_id for row in leaves[2:10]]
    g4_id = selected_ids[0]
    rows: list[dict[str, object]] = []
    selected_leaf_bytes = 0
    for leaf in leaves:
        if leaf.stream_id in selected_ids:
            codec = "G4_FREE_DECODER_CONTEXT" if leaf.stream_id == g4_id else "BELLARD_CLASS_MIXING"
            frame = (
                encode_g4_decoder_context(leaf.payload)
                if leaf.stream_id == g4_id
                else encode_bellard_class_mixing(leaf.payload)
            )
            assert len(frame) < len(leaf.payload)
            rows.append(
                {
                    "stream_id": leaf.stream_id,
                    "current_bytes": len(leaf.payload),
                    "current_sha256": hashlib.sha256(leaf.payload).hexdigest(),
                    "selected_codec": codec,
                    "selected_framed_bytes": len(frame),
                    "delta_bytes": len(frame) - len(leaf.payload),
                    "arms": [
                        {
                            "codec": codec,
                            "frame_sha256": hashlib.sha256(frame).hexdigest(),
                        }
                    ],
                }
            )
            selected_leaf_bytes += len(frame)
        else:
            rows.append(
                {
                    "stream_id": leaf.stream_id,
                    "current_bytes": len(leaf.payload),
                    "current_sha256": hashlib.sha256(leaf.payload).hexdigest(),
                    "selected_codec": "RAW_CURRENT",
                    "selected_framed_bytes": len(leaf.payload),
                    "delta_bytes": 0,
                    "arms": [],
                }
            )
            selected_leaf_bytes += len(leaf.payload)
    selected_total = overhead + selected_leaf_bytes
    price_table: dict[str, object] = {
        "price_table_schema": cc3.PRICE_TABLE_SCHEMA,
        "composition_archive_bytes": len(source),
        "composition_archive_sha256": hashlib.sha256(source).hexdigest(),
        "selected_total_archive_estimate_bytes": selected_total,
        "selected_total_delta_bytes": selected_total - len(source),
        "rows": rows,
    }
    return source, price_table


def _tamper_leaf(payload: bytes, *, target: str, owner: str = cc3.COMPOSITION_OWNER) -> bytes:
    stream = io.BytesIO(payload)
    if not zipfile.is_zipfile(stream):
        if owner != target:
            return payload
        mutated = bytearray(payload)
        mutated[-1] ^= 1
        return bytes(mutated)
    rows, suffix = cc3._read_stored_zip(payload, owner=owner)
    rewritten = tuple(
        (
            info,
            _tamper_leaf(
                member,
                target=target,
                owner=f"{owner}!/{info.filename}",
            ),
        )
        for info, member in rows
    )
    return cc3._write_stored_zip(rewritten, suffix=suffix)


def test_exact_eight_leaf_mixed_archive_restores_source_and_extracted_bridge() -> None:
    source, price_table = _fixture()
    mixed, build = cc3.build_mixed_archive(source, price_table)
    restored, receipt = cc3.restore_mixed_archive(mixed)
    assert restored == source
    assert len(mixed) - len(source) == price_table["selected_total_delta_bytes"]
    assert build["selected_leaf_count"] == 8
    assert build["raw_leaf_count"] == 19
    assert receipt["physical_leaf_count"] == 27
    assert receipt["codec_counts"] == {
        "G4_FREE_DECODER_CONTEXT": 1,
        "BELLARD_CLASS_MIXING": 7,
    }

    members = cc3._composition_members(mixed)
    extracted_source, parent, packet, bridge = cc3.restore_extracted_composition(members)
    assert extracted_source == source
    assert parent == cc3._composition_members(source)["parent/ws1.zip"]
    assert packet.active is True
    assert bridge["decoded_frame_count"] == 8


def test_each_selected_frame_terminal_byte_is_consumed_and_refused() -> None:
    source, price_table = _fixture()
    mixed, build = cc3.build_mixed_archive(source, price_table)
    for row in build["replacement_rows"]:
        tampered = _tamper_leaf(mixed, target=row["stream_id"])
        with pytest.raises(cc3.MixedCoderReceiverError):
            cc3.restore_mixed_archive(tampered)


def test_recursive_zip_suffix_may_contain_false_eocd_signature() -> None:
    base = _stored_zip([("leaf.bin", b"payload")])
    suffix = b"W_joint-state-prefix:PK\x05\x06:not-an-EOCD"
    rows, observed_suffix = cc3._read_stored_zip(base + suffix, owner="synthetic.zip")
    assert observed_suffix == suffix
    assert cc3._write_stored_zip(rows, suffix=observed_suffix) == base + suffix


def _write_extracted_fixture(root: Path) -> None:
    for name in runtime.EXPECTED_CC3_MEMBERS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("ascii"))


def test_extracted_member_reader_rejects_extra_directories_and_symlinks(
    tmp_path: Path,
) -> None:
    clean = tmp_path / "clean"
    _write_extracted_fixture(clean)
    assert tuple(runtime._read_exact_cc3_members(clean)) == runtime.EXPECTED_CC3_MEMBERS

    extra = tmp_path / "extra"
    _write_extracted_fixture(extra)
    (extra / "unowned").mkdir()
    with pytest.raises(runtime.ReceiverError, match="extra directory"):
        runtime._read_exact_cc3_members(extra)

    linked = tmp_path / "linked"
    _write_extracted_fixture(linked)
    target = linked / "target.json"
    target.write_bytes(b"target")
    member = linked / "manifest/pc1.json"
    member.unlink()
    member.symlink_to(target)
    with pytest.raises(runtime.ReceiverError, match="symlink"):
        runtime._read_exact_cc3_members(linked)
