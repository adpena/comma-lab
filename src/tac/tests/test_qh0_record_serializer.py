# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import brotli
import pytest

from tac.pr85_bundle import parse_pr85_bundle
from tac.qh0_record_serializer import (
    QH0Record,
    build_serialized_variants,
    choose_byte_win_candidates,
    pack_hilo_fp4_bytes,
    prove_decoded_tensor_parity,
    record_set_summary,
    serialize_records,
    split_even_odd_bytes,
    unpack_hilo_fp4_bytes,
    unsplit_even_odd_bytes,
)


def test_qh0_splits_and_synthetic_record_serializer_are_deterministic() -> None:
    direct = bytes(range(18))
    split = split_even_odd_bytes(direct)
    assert unsplit_even_odd_bytes(split) == direct
    assert split_even_odd_bytes(direct) == split

    packed = bytes([0x12, 0x34, 0xAB, 0xCD])
    hilo = pack_hilo_fp4_bytes(packed)
    assert unpack_hilo_fp4_bytes(hilo, len(packed)) == packed
    assert pack_hilo_fp4_bytes(packed) == hilo

    record = QH0Record(
        name="synthetic.weight",
        category="module_weight",
        record_kind="fp16",
        offset=3,
        source_nbytes=1 + len(direct),
        direct_record=b"\x00" + direct,
        qh0_record=b"\x00" + split,
        tensor_shape=(len(direct) // 2,),
        element_count=len(direct) // 2,
        kind_byte=0,
    )
    assert serialize_records([record], magic=b"QH0") == b"QH0\x00" + split
    assert serialize_records([record], magic=b"QM0") == b"QM0\x00" + direct
    assert serialize_records([record], magic="QM0") == b"QM0\x00" + direct


def test_byte_win_candidate_filter_keeps_only_runtime_compatible_wins() -> None:
    rows = [
        {
            "candidate_id": "win_ok",
            "candidate_model_delta_bytes_vs_source": -3,
            "decoded_tensor_parity": True,
            "runtime_compatibility": {"runtime_can_decode_without_edits": True},
        },
        {
            "candidate_id": "byte_negative",
            "candidate_model_delta_bytes_vs_source": 4,
            "decoded_tensor_parity": True,
            "runtime_compatibility": {"runtime_can_decode_without_edits": True},
        },
        {
            "candidate_id": "win_runtime_blocked",
            "candidate_model_delta_bytes_vs_source": -8,
            "decoded_tensor_parity": True,
            "runtime_compatibility": {"runtime_can_decode_without_edits": False},
        },
    ]

    selected = choose_byte_win_candidates(rows)

    assert [row["candidate_id"] for row in selected] == ["win_ok"]


def test_public_pr85_qh0_to_qm0_tensor_parity() -> None:
    archive = Path("experiments/results/public_pr85_intake_20260503_codex/archive.zip")
    if not archive.is_file():
        pytest.skip("public PR85 intake archive is not present")

    import zipfile

    with zipfile.ZipFile(archive, "r") as zf:
        bundle = parse_pr85_bundle(zf.read("x"))
    source_model = brotli.decompress(bundle.segments["model"])
    record_set, variants = build_serialized_variants(source_model)
    by_magic = {variant.magic: variant for variant in variants}

    summary = record_set_summary(record_set)
    assert summary["source_magic"] == "QH0"
    assert summary["record_kind_counts"]["fp4"] > 0
    assert by_magic["QH0"].same_as_source is True
    assert by_magic["QM0"].payload.startswith(b"QM0")

    parity = prove_decoded_tensor_parity(source_model, by_magic["QM0"].payload)

    assert parity["decoded_tensor_parity"] is True
    assert parity["max_abs_diff"] == 0.0
