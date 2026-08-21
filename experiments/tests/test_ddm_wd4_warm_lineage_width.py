from __future__ import annotations

import io
import struct
import zipfile

import brotli
import torch

from experiments import ddm_wd2_student_receiver as receiver
from experiments import ddm_wd4_warm_lineage_width as wd4


def test_ck2_round_trip_is_total() -> None:
    for payload in (b"", b"a", b"ab", bytes(range(255))):
        assert wd4.ck2_uninterleave(wd4.ck2_interleave(payload)) == payload


def test_stratified_pair_ids_are_deterministic_and_cover_bins() -> None:
    observed = wd4.stratified_pair_ids(32, wd4.SEED)
    assert observed == wd4.stratified_pair_ids(32, wd4.SEED)
    assert len(observed) == len(set(observed)) == 32
    edges = torch.linspace(0, 600, 33, dtype=torch.int64).tolist()
    assert all(edges[index] <= value < edges[index + 1] for index, value in enumerate(observed))


def test_nested_slice_strict_loads_same_mechanism_student() -> None:
    torch.manual_seed(7)
    teacher = receiver.StudentSemanticRenderer(
        receiver.StudentSpec("teacher", "dense", 96, 4)
    )
    state = teacher.state_dict()
    order = wd4.salience_order(state)
    assert sorted(order) == list(range(96))
    assert all(
        order[start : start + 8] == list(range(order[start], order[start] + 8))
        for start in range(0, 96, 8)
    )
    sliced = wd4.slice_dense_state(state, order[:64])
    student = receiver.StudentSemanticRenderer(
        receiver.StudentSpec("student", "dense", 64, 4)
    )
    student.load_state_dict(sliced, strict=True)
    packet = receiver.pack_student(student)
    parsed = receiver.unpack_student(packet)
    assert parsed.spec.width == 64
    assert tuple(parsed.state_dict()) == tuple(student.state_dict())
    assert len(parsed.state_dict()) == 38


def test_real_coder_archive_replaces_only_semantic_section() -> None:
    container = {
        "magic": b"RX1M",
        "version": 1,
        "codec": 2,
        "table_mode": 0,
        "reserved": 0x0A,
        "hpac": b"h" * 11,
        "carrier": b"c" * 13,
        "tail": b"t" * 17,
    }
    packet = b"WD2S" + bytes(range(64))
    semantic, member, archive = wd4.build_archive(container, packet)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        assert bundle.namelist() == ["p"]
        assert bundle.read("p") == member
    fields = wd4.RX1_HEADER.unpack_from(member)
    assert fields[:5] == (b"RX1M", 1, 2, 0, 0x0A)
    assert fields[5:] == (11, len(semantic), 13)
    offset = wd4.RX1_HEADER.size + 11
    assert wd4.ck2_uninterleave(brotli.decompress(member[offset : offset + len(semantic)])) == packet
    assert member[-17:] == container["tail"]
    assert struct.calcsize("<4sBBBBHHH") == wd4.RX1_HEADER.size
