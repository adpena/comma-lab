from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from experiments.build_pr85_qh0_serializer_candidates import (
    build_candidates,
    runtime_compatibility,
)
from tac.pr85_bundle import parse_pr85_bundle
from tac.qh0_record_serializer import (
    QH0Record,
    build_serialized_variants,
    choose_byte_win_candidates,
    pack_hilo_fp4_bytes,
    prove_decoded_tensor_parity,
    serialize_records,
    split_even_odd_bytes,
    unpack_hilo_fp4_bytes,
    unsplit_even_odd_bytes,
)
from tac.qh0_renderer_codec import QH0_MAGIC, QM0_MAGIC


brotli = pytest.importorskip("brotli")

REPO = Path(__file__).resolve().parents[3]


def test_low_level_qh0_splits_and_synthetic_record_serializer_are_deterministic() -> None:
    direct = bytes(range(17))
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
    assert serialize_records([record], magic=QH0_MAGIC) == QH0_MAGIC + b"\x00" + split
    assert serialize_records([record], magic=QM0_MAGIC) == QM0_MAGIC + b"\x00" + direct
    assert serialize_records([record], magic=QM0_MAGIC) == serialize_records(
        [record],
        magic=QM0_MAGIC,
    )


def test_byte_win_candidate_filter_keeps_only_runtime_compatible_wins() -> None:
    rows = [
        {
            "candidate_id": "win_ok",
            "candidate_model_delta_bytes_vs_source": -3,
            "runtime_compatibility": {"runtime_can_decode_without_edits": True},
        },
        {
            "candidate_id": "byte_negative",
            "candidate_model_delta_bytes_vs_source": 4,
            "runtime_compatibility": {"runtime_can_decode_without_edits": True},
        },
        {
            "candidate_id": "win_runtime_blocked",
            "candidate_model_delta_bytes_vs_source": -8,
            "runtime_compatibility": {"runtime_can_decode_without_edits": False},
        },
    ]

    selected = choose_byte_win_candidates(rows)

    assert [row["candidate_id"] for row in selected] == ["win_ok"]


def test_runtime_compatibility_fails_closed_when_replay_loader_lacks_magic(tmp_path: Path) -> None:
    replay = tmp_path / "inflate.py"
    replay.write_text(
        'def load_compact_archive_bundle(data_dir):\n'
        '    path = data_dir / "x"\n'
        "def get_decoded_state_dict_custom(payload_data, device):\n"
        '    if payload_data[:3] == b"QH0":\n'
        "        return {}\n",
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate_renderer.py").write_text("# QH0 only\n", encoding="utf-8")
    (runtime / "unpack_renderer_payload.py").write_text("# no single x support\n", encoding="utf-8")

    compat = runtime_compatibility("QM0", replay_inflate_py=replay, robust_current_dir=runtime)

    assert compat["runtime_can_decode_without_edits"] is False
    assert compat["dispatch_unlocked"] is False
    assert compat["blocker_class"] == "runtime_incompatibility"
    assert "public_pr85_replay_missing_QM0_model_loader" in compat["blockers"]


def test_real_pr85_qh0_to_qm0_serializer_parity_and_local_smoke(tmp_path: Path) -> None:
    archive = REPO / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
    if not archive.is_file():
        pytest.skip("public PR85 intake archive is not present")

    with zipfile.ZipFile(archive, "r") as zf:
        source_x = zf.read("x")
    bundle = parse_pr85_bundle(source_x)
    source_model = brotli.decompress(bundle.segments["model"])

    record_set, variants = build_serialized_variants(source_model)
    by_id = {variant.variant_id: variant for variant in variants}

    assert record_set.source_magic == "QH0"
    assert by_id["qh0_canonical"].payload == source_model
    assert by_id["qm0_direct"].payload.startswith(b"QM0")
    parity = prove_decoded_tensor_parity(source_model, by_id["qm0_direct"].payload)
    assert parity["decoded_tensor_parity"] is True

    summary = build_candidates(
        archive,
        tmp_path / "out",
        qualities=(0, 11),
        lgwins=(18, 24),
    )

    assert summary["score_claim"] is False
    assert summary["dispatch_performed"] is False
    assert summary["remote_gpu_dispatch_performed"] is False
    assert summary["source_model_segment"]["bytes"] == len(bundle.segments["model"])
    assert summary["best_screened_candidate"] is not None
    assert (tmp_path / "out" / "candidate_summary.json").is_file()
