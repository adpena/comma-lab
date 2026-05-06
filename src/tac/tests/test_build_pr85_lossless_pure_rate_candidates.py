from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from experiments.build_pr85_lossless_pure_rate_candidates import (
    _encode_p1d1,
    _parse_p1d1,
    _p1d1_semantic_sha256,
    _write_uvarint,
    build_candidates,
)


REPO = Path(__file__).resolve().parents[3]


def _p1d1_stream(values: list[int]) -> bytes:
    previous = 0
    out = bytearray()
    for value in values:
        delta = value - previous
        previous = value
        out += _write_uvarint((delta << 1) ^ (delta >> 31))
    return bytes(out)


def _synthetic_p1d1(order: tuple[int, ...]) -> bytes:
    q_by_dim = {
        0: [10240 + (idx % 3) for idx in range(600)],
        2: [idx % 17 - 8 for idx in range(600)],
    }
    header = bytearray(b"P1D1")
    header.append(len(order))
    body = bytearray()
    for dim in order:
        stream = _p1d1_stream(q_by_dim[dim])
        header.append(dim)
        header += len(stream).to_bytes(2, "little")
        body += stream
    return bytes(header + body)


def test_p1d1_reorder_preserves_decoded_pose_semantics() -> None:
    source = _synthetic_p1d1((0, 2))
    parsed = _parse_p1d1(source)

    reordered = _encode_p1d1(parsed, (2, 0))

    assert reordered != source
    assert _p1d1_semantic_sha256(reordered) == _p1d1_semantic_sha256(source)
    assert _parse_p1d1(reordered)["dims"] == (2, 0)


def test_build_candidates_screens_real_pr85_without_dispatch(tmp_path: Path) -> None:
    archive = REPO / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
    if not archive.is_file():
        pytest.skip("public PR85 archive is not present")

    summary = build_candidates(
        archive,
        tmp_path,
        qualities=(10,),
        lgwins=(16,),
        build_limit=2,
    )

    assert summary["score_claim"] is False
    assert summary["dispatch_performed"] is False
    assert summary["remote_gpu_dispatch_performed"] is False
    assert summary["source_archive"]["archive_bytes"] == 236_328
    assert summary["frontier_context"]["score_recomputed_from_components"] is not None
    assert {"PR86", "PR90", "PR91"}.issubset(summary["public_anatomy"])
    assert summary["exact_dispatch_gate"]["claim_required"].startswith("tools/claim_lane_dispatch.py")

    if summary["best_built_candidate"] is None:
        assert summary["built_candidate_count"] == 0
        assert summary["reason_no_candidate_built"]
        best_delta = summary["best_screened_candidate"].get(
            "archive_delta_bytes_vs_source_formula",
            summary["best_screened_candidate"].get("archive_delta_bytes_vs_source"),
        )
        assert best_delta >= 0
    else:
        best = summary["best_built_candidate"]
        candidate_path = REPO / best["archive_path"]
        assert candidate_path.is_file()
        assert best["archive_delta_bytes_vs_source"] < 0
        with zipfile.ZipFile(candidate_path, "r") as zf:
            assert [info.filename for info in zf.infolist() if not info.is_dir()] == ["x"]
