"""Exact tiny-fixture tests for the C0B semantic/quotient seam."""

from __future__ import annotations

import hashlib
import lzma
import os
import zipfile
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import tac.witness_dsl.c0b_semantic_quotient as quotient
from tac.witness_dsl.c0b_semantic_quotient import (
    PlaneChunk,
    RendererIdentity,
    SemanticQuotientError,
    build_semantic_quotient_archive,
    double_decode_archive,
    parse_semantic_quotient_archive,
)

CAMERA_HW = (8, 10)
SCORER_HW = (3, 4)
CHANNELS = 3
PAIR_COUNT = 4
CHUNK_PAIRS = 2
SEMANTIC_PACKET = b"synthetic generic semantic program v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _planes(offset: int) -> np.ndarray:
    values = np.arange(PAIR_COUNT * SCORER_HW[0] * SCORER_HW[1] * CHANNELS, dtype=np.uint16)
    return ((values * 37 + offset) % 256).astype(np.uint8).reshape(PAIR_COUNT, *SCORER_HW, CHANNELS)


class SyntheticRenderer:
    """Tiny-test-only caller-supplied base planes; never a full E1 renderer."""

    def __init__(
        self,
        y0: np.ndarray,
        y1: np.ndarray,
        *,
        chunk_pairs: int = CHUNK_PAIRS,
        source_tag: bytes = b"renderer-source-v1",
    ) -> None:
        self.y0 = y0
        self.y1 = y1
        self.chunk_pairs = chunk_pairs
        self._identity = RendererIdentity(
            renderer_id="test.synthetic_semantic_renderer.v1",
            renderer_source_sha256=_sha256(source_tag),
            semantic_packet_schema="test.synthetic_semantic_packet.v1",
            expected_semantic_packet_sha256=_sha256(SEMANTIC_PACKET),
        )

    @property
    def identity(self) -> RendererIdentity:
        return self._identity

    def render_chunks(
        self,
        semantic_packet: bytes,
        *,
        work_root: Path,
        chunk_pairs: int,
        resume: bool,
    ) -> Iterable[PlaneChunk]:
        assert semantic_packet == SEMANTIC_PACKET
        assert chunk_pairs == self.chunk_pairs
        assert isinstance(resume, bool)
        for chunk_index, start in enumerate(range(0, self.y0.shape[0], chunk_pairs)):
            stop = min(start + chunk_pairs, self.y0.shape[0])
            yield PlaneChunk(
                chunk_index,
                tuple(range(start, stop)),
                self.y0[start:stop],
                self.y1[start:stop],
            )


def _teacher_chunks(y0: np.ndarray, y1: np.ndarray, *, chunk_pairs: int = CHUNK_PAIRS) -> Iterable[PlaneChunk]:
    for chunk_index, start in enumerate(range(0, y0.shape[0], chunk_pairs)):
        stop = min(start + chunk_pairs, y0.shape[0])
        yield PlaneChunk(chunk_index, tuple(range(start, stop)), y0[start:stop], y1[start:stop])


def _teacher_custody(y0: np.ndarray, y1: np.ndarray, *, chunk_pairs: int = CHUNK_PAIRS) -> dict[str, object]:
    rows = [
        {
            "chunk_index": chunk.chunk_index,
            "pair_ids": list(chunk.pair_ids),
            "y0_sha256": _sha256(chunk.y0.tobytes(order="C")),
            "y1_sha256": _sha256(chunk.y1.tobytes(order="C")),
        }
        for chunk in _teacher_chunks(y0, y1, chunk_pairs=chunk_pairs)
    ]
    return {
        "schema": quotient.TARGET_TEACHER_CUSTODY_SCHEMA,
        "teacher_id": "test.synthetic-independent-two-plane.v1",
        "pair_count": y0.shape[0],
        "chunk_count": len(rows),
        "chunk_pairs": chunk_pairs,
        "scorer_hw": list(y0.shape[1:3]),
        "channels": y0.shape[3],
        "y0_sha256": _sha256(y0.tobytes(order="C")),
        "y1_sha256": _sha256(y1.tobytes(order="C")),
        "consumed_chunk_target_hashes": rows,
        "consumed_chunk_target_hashes_sha256": _sha256(quotient.canonical_json(rows)),
        "provenance": {"test_only_small_fixture": True},
        "all_video_derived_metadata_counted": True,
        "score_claim": False,
        "promotion_eligible": False,
    }


def _build(tmp_path: Path, name: str = "work") -> tuple[object, SyntheticRenderer, np.ndarray, np.ndarray]:
    base_y0 = _planes(11)
    base_y1 = np.flip(_planes(23), axis=2).copy()
    target_y0 = np.bitwise_xor(base_y0, np.uint8(0x5A))
    target_y1 = ((base_y1.astype(np.uint16) + 73) % 256).astype(np.uint8)
    renderer = SyntheticRenderer(base_y0, base_y1)
    root = tmp_path / name
    result = build_semantic_quotient_archive(
        SEMANTIC_PACKET,
        renderer,
        _teacher_chunks(target_y0, target_y1),
        archive_path=root / "output" / "archive.zip",
        work_root=root,
        target_teacher_custody=_teacher_custody(target_y0, target_y1),
        camera_hw=CAMERA_HW,
        scorer_hw=SCORER_HW,
        channels=CHANNELS,
        pair_count=PAIR_COUNT,
        chunk_pairs=CHUNK_PAIRS,
        resume=True,
        test_only_small_fixture=True,
    )
    return result, renderer, target_y0, target_y1


def test_semantic_packet_plus_exact_quotient_recovers_both_planes_and_factor2(tmp_path: Path) -> None:
    result, renderer, target_y0, target_y1 = _build(tmp_path)
    parsed = parse_semantic_quotient_archive(result.archive_path)

    assert parsed.semantic_packet == SEMANTIC_PACKET
    assert parsed.manifest["scientific_label"] == quotient.SCIENTIFIC_LABEL
    assert parsed.manifest["semantic_base"]["base_type"] == quotient.SEMANTIC_BASE_TYPE_ID
    assert parsed.manifest["quotient_codec"]["entropy_optimal_claim"] is False
    assert parsed.manifest["semantic_base"]["renderer"]["renderer_embedded_in_archive"] is False
    assert parsed.manifest["promotion_eligible"] is False
    assert parsed.manifest["score_claim"] is False

    recovered_y0: list[np.ndarray] = []
    recovered_y1: list[np.ndarray] = []
    with zipfile.ZipFile(parsed.archive_path, "r") as archive:
        assert "base/y0.bin" not in archive.namelist()
        assert "base/y1.bin" not in archive.namelist()
        for chunk_index, row in enumerate(parsed.manifest["chunks"]):
            start = chunk_index * CHUNK_PAIRS
            stop = start + CHUNK_PAIRS
            for key, base, recovered in (
                ("y0", renderer.y0[start:stop], recovered_y0),
                ("y1", renderer.y1[start:stop], recovered_y1),
            ):
                leg = row["quotient"][key]
                residual = lzma.decompress(
                    archive.read(leg["member"]),
                    format=lzma.FORMAT_RAW,
                    filters=list(quotient.LZMA_FILTERS),
                )
                decoded = np.bitwise_xor(
                    base,
                    np.frombuffer(residual, dtype=np.uint8).reshape(base.shape),
                )
                recovered.append(decoded)
    np.testing.assert_array_equal(np.concatenate(recovered_y0), target_y0)
    np.testing.assert_array_equal(np.concatenate(recovered_y1), target_y1)

    first = result.double_decode["first"]
    assert first["y0_sha256"] == _sha256(target_y0.tobytes(order="C"))
    assert first["y1_sha256"] == _sha256(target_y1.tobytes(order="C"))
    assert first["factor2_verified_values"] == PAIR_COUNT * 2 * SCORER_HW[0] * SCORER_HW[1] * CHANNELS
    assert first["camera0_chunk_hashes_sha256"] is not None
    assert first["camera1_chunk_hashes_sha256"] is not None
    assert result.double_decode["byte_identical"] is True
    assert result.double_decode["factor2_verified"] is True


def test_build_is_deterministic_and_resume_adopts_only_identical_stages(tmp_path: Path) -> None:
    first, renderer, target_y0, target_y1 = _build(tmp_path, "first")
    resumed = build_semantic_quotient_archive(
        SEMANTIC_PACKET,
        renderer,
        _teacher_chunks(target_y0, target_y1),
        archive_path=first.archive_path,
        work_root=tmp_path / "first",
        target_teacher_custody=_teacher_custody(target_y0, target_y1),
        camera_hw=CAMERA_HW,
        scorer_hw=SCORER_HW,
        channels=CHANNELS,
        pair_count=PAIR_COUNT,
        chunk_pairs=CHUNK_PAIRS,
        resume=True,
        test_only_small_fixture=True,
    )
    second, _renderer, _target_y0, _target_y1 = _build(tmp_path, "second")
    assert resumed.archive_sha256 == first.archive_sha256
    assert second.archive_sha256 == first.archive_sha256
    assert second.archive_path.read_bytes() == first.archive_path.read_bytes()

    chunk = tmp_path / "first" / "quotient_chunks" / "chunk-0000.y0.xor.lzma"
    chunk.write_bytes(chunk.read_bytes() + b"drift")
    with pytest.raises(SemanticQuotientError, match="preserved write-once bytes drifted"):
        build_semantic_quotient_archive(
            SEMANTIC_PACKET,
            renderer,
            _teacher_chunks(target_y0, target_y1),
            archive_path=first.archive_path,
            work_root=tmp_path / "first",
            target_teacher_custody=_teacher_custody(target_y0, target_y1),
            camera_hw=CAMERA_HW,
            scorer_hw=SCORER_HW,
            channels=CHANNELS,
            pair_count=PAIR_COUNT,
            chunk_pairs=CHUNK_PAIRS,
            resume=True,
            test_only_small_fixture=True,
        )


def test_decode_refuses_renderer_identity_substitution(tmp_path: Path) -> None:
    result, renderer, _target_y0, _target_y1 = _build(tmp_path)
    substituted = SyntheticRenderer(renderer.y0, renderer.y1, source_tag=b"different-renderer-source")
    with pytest.raises(SemanticQuotientError, match="renderer identity differs"):
        double_decode_archive(
            result.archive_path,
            substituted,
            work_root=tmp_path / "substituted-decode",
            verify_factor2=True,
        )


def test_nonfinal_and_final_chunk_arithmetic_is_closed(tmp_path: Path) -> None:
    pair_count = 5
    chunk_pairs = 2
    shape = (pair_count, *SCORER_HW, CHANNELS)
    base_y0 = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape).astype(np.uint8)
    base_y1 = np.flip(base_y0, axis=2).copy()
    target_y0 = np.bitwise_xor(base_y0, np.uint8(0xA5))
    target_y1 = np.bitwise_xor(base_y1, np.uint8(0x3C))
    renderer = SyntheticRenderer(base_y0, base_y1, chunk_pairs=chunk_pairs)
    result = build_semantic_quotient_archive(
        SEMANTIC_PACKET,
        renderer,
        _teacher_chunks(target_y0, target_y1, chunk_pairs=chunk_pairs),
        archive_path=tmp_path / "partial" / "output" / "archive.zip",
        work_root=tmp_path / "partial",
        target_teacher_custody=_teacher_custody(target_y0, target_y1, chunk_pairs=chunk_pairs),
        camera_hw=CAMERA_HW,
        scorer_hw=SCORER_HW,
        channels=CHANNELS,
        pair_count=pair_count,
        chunk_pairs=chunk_pairs,
        resume=True,
        test_only_small_fixture=True,
    )

    manifest = result.manifest
    plane_value_count = SCORER_HW[0] * SCORER_HW[1] * CHANNELS
    assert manifest["chunk_count"] == 3
    assert [len(row["pair_ids"]) for row in manifest["chunks"]] == [2, 2, 1]
    for row, expected_pairs in zip(manifest["chunks"], (2, 2, 1), strict=True):
        assert row["quotient"]["y0"]["raw_bytes"] == expected_pairs * plane_value_count
        assert row["quotient"]["y1"]["raw_bytes"] == expected_pairs * plane_value_count
        assert row["factor2"]["verified_values"] == 2 * expected_pairs * plane_value_count
    assert manifest["decoded_targets"]["y0_bytes"] == pair_count * plane_value_count
    assert manifest["decoded_targets"]["y1_bytes"] == pair_count * plane_value_count
    assert manifest["rate_accounting"]["quotient_raw_bytes"] == 2 * pair_count * plane_value_count


def test_manifest_arithmetic_and_teacher_custody_tampering_refuse(tmp_path: Path) -> None:
    result, _renderer, _target_y0, _target_y1 = _build(tmp_path)
    mutations: list[tuple[str, object]] = []

    wrong_chunk_count = deepcopy(result.manifest)
    wrong_chunk_count["chunk_count"] = 3
    mutations.append(("chunk count arithmetic", wrong_chunk_count))

    wrong_raw_bytes = deepcopy(result.manifest)
    wrong_raw_bytes["chunks"][0]["quotient"]["y0"]["raw_bytes"] -= 1
    wrong_raw_bytes["rate_accounting"]["quotient_raw_bytes"] -= 1
    mutations.append(("quotient raw chunk bytes", wrong_raw_bytes))

    wrong_decoded_bytes = deepcopy(result.manifest)
    wrong_decoded_bytes["decoded_targets"]["y1_bytes"] -= 1
    mutations.append(("decoded target byte arithmetic", wrong_decoded_bytes))

    wrong_factor2 = deepcopy(result.manifest)
    wrong_factor2["chunks"][0]["factor2"]["verified_values"] -= 1
    mutations.append(("factor2 verified-value arithmetic", wrong_factor2))

    wrong_custody_geometry = deepcopy(result.manifest)
    wrong_custody_geometry["target_teacher_custody"]["pair_count"] -= 1
    mutations.append(("teacher custody geometry", wrong_custody_geometry))

    wrong_consumed_hash = deepcopy(result.manifest)
    wrong_consumed_hash["target_teacher_custody"]["consumed_chunk_target_hashes"][0]["y0_sha256"] = "0" * 64
    mutations.append(("consumed chunk hash list digest", wrong_consumed_hash))

    for expected, candidate in mutations:
        with pytest.raises(SemanticQuotientError, match=expected):
            quotient._validate_manifest(candidate)


def test_parse_refuses_noncanonical_zip_bytes_and_symlink_path(tmp_path: Path) -> None:
    result, _renderer, _target_y0, _target_y1 = _build(tmp_path)
    original = result.archive_path.read_bytes()
    result.archive_path.write_bytes(original + b"noncanonical-trailer")
    with pytest.raises(SemanticQuotientError, match="canonical ZIP encoding"):
        parse_semantic_quotient_archive(result.archive_path)

    real_archive = result.archive_path.with_name("real.zip")
    real_archive.write_bytes(original)
    result.archive_path.unlink()
    result.archive_path.symlink_to(real_archive)
    with pytest.raises(SemanticQuotientError, match="no-follow regular file"):
        parse_semantic_quotient_archive(result.archive_path)


@pytest.mark.parametrize("racer_identical", [True, False])
def test_archive_publication_never_overwrites_concurrent_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    racer_identical: bool,
) -> None:
    real_link = os.link
    raced = False

    def racing_link(source: os.PathLike[str], destination: os.PathLike[str], *args: object, **kwargs: object) -> None:
        nonlocal raced
        destination_path = Path(destination)
        if destination_path.name == "archive.zip" and not raced:
            raced = True
            competing = Path(source).read_bytes() if racer_identical else b"different-concurrent-archive"
            destination_path.write_bytes(competing)
            raise FileExistsError(destination_path)
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(quotient.os, "link", racing_link)
    if racer_identical:
        result, _renderer, _target_y0, _target_y1 = _build(tmp_path, "race-identical")
        assert parse_semantic_quotient_archive(result.archive_path).archive_sha256 == result.archive_sha256
    else:
        with pytest.raises(SemanticQuotientError, match="preserved write-once bytes drifted"):
            _build(tmp_path, "race-different")
        assert (tmp_path / "race-different" / "output" / "archive.zip").read_bytes() == b"different-concurrent-archive"


def test_e1_adapter_binds_executed_module_path_and_live_source_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.build_c0b_semantic_quotient_archive as production

    packet = b"tiny-not-a-real-e1-packet"
    fake_source = tmp_path / "ddm_runtime_receiver.py"
    fake_source.write_text("def inflate(*args):\n    raise AssertionError\n", encoding="utf-8")
    monkeypatch.setattr(production, "E1_PACKET_SHA256", _sha256(packet))
    renderer = production.E1SemanticPlaneRenderer(
        runtime_source=fake_source,
        expected_runtime_sha256=production.sha256_file(fake_source),
    )
    with pytest.raises(SemanticQuotientError, match="module path differs"):
        next(
            iter(
                renderer.render_chunks(
                    packet,
                    work_root=tmp_path / "wrong-module",
                    chunk_pairs=production.CHUNK_PAIRS,
                    resume=True,
                )
            )
        )

    renderer = production.E1SemanticPlaneRenderer(
        runtime_source=fake_source,
        expected_runtime_sha256=production.sha256_file(fake_source),
    )
    fake_source.write_text("def inflate(*args):\n    return None\n", encoding="utf-8")
    with pytest.raises(SemanticQuotientError, match="changed before execution"):
        next(
            iter(
                renderer.render_chunks(
                    packet,
                    work_root=tmp_path / "changed-source",
                    chunk_pairs=production.CHUNK_PAIRS,
                    resume=True,
                )
            )
        )
