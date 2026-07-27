# SPDX-License-Identifier: MIT
"""Exact custody and fail-closed tests for the G85 portable semantic recode."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tac.witness_dsl import taskspace_g85_portable_semantic_recode_v1 as g85p

_ROOT = Path(__file__).resolve().parents[4]
_RUN_DIR = (
    _ROOT / ".omx/research/original_taskspace_inverse_witness_codec_20260725" / "fresh_v15_semantic_base_n600_20260726"
)
_SEMANTIC_PATH = _RUN_DIR / "ddm_v15_solved_templates_n600.not_a_candidate.zip.receipt-bytes"
_SOURCE_SEMANTIC_SHA256 = "759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df"
_PORTABLE_SEMANTIC_SHA256 = "f44dd0e1f2fd76847a1109080fa3f3594572fd124fd90a9adafc682eb0916657"


def _semantic() -> bytes:
    if not _SEMANTIC_PATH.is_file():
        pytest.skip("retained fresh V15 semantic custody is absent")
    payload = _SEMANTIC_PATH.read_bytes()
    assert len(payload) == 133_941
    assert hashlib.sha256(payload).hexdigest() == _SOURCE_SEMANTIC_SHA256
    return payload


@pytest.fixture(scope="module")
def recode() -> g85p.PortableSemanticRecodeV1:
    return g85p.transcode_portable_semantic_p(_semantic())


def test_exact_semantic_recode_closes_all_brotli_sections(
    recode: g85p.PortableSemanticRecodeV1,
) -> None:
    receipt = recode.receipt
    assert receipt.source_semantic_p_bytes == 133_941
    assert receipt.source_semantic_p_sha256 == _SOURCE_SEMANTIC_SHA256
    assert receipt.portable_semantic_p_bytes == 161_915
    assert receipt.portable_semantic_p_sha256 == _PORTABLE_SEMANTIC_SHA256
    assert len(receipt.section_rows) == receipt.changed_brotli_section_count == 12
    assert all(row["decoded_bytes_identical"] is True for row in receipt.section_rows)
    assert {row["portable_codec"] for row in receipt.section_rows} == {
        "zlib9",
        "pzsm1_zlib9",
    }


def test_semantic_recode_is_deterministic_and_parseback_exact(
    recode: g85p.PortableSemanticRecodeV1,
) -> None:
    replay = g85p.transcode_portable_semantic_p(_semantic())
    assert replay.semantic_p_archive == recode.semantic_p_archive
    assert replay.receipt.to_bytes() == recode.receipt.to_bytes()
    assert hashlib.sha256(replay.semantic_p_archive).hexdigest() == _PORTABLE_SEMANTIC_SHA256


def test_receipt_refuses_public_or_score_claims(
    recode: g85p.PortableSemanticRecodeV1,
) -> None:
    receipt = recode.receipt
    assert receipt.output_video_equality_proven is False
    assert receipt.opencv_rasterizer_replaced is False
    assert receipt.tree_shaken_public_receiver_closed is False
    assert receipt.upstream_default_double_decode_proven is False
    assert receipt.evaluator_invoked is False
    assert receipt.score_claim is False
    assert receipt.candidate_claim is False
    assert receipt.research_only is True
    assert receipt.open_blockers == (
        g85p.RASTERIZER_BLOCKER,
        g85p.TREE_SHAKE_BLOCKER,
    )


def test_portable_g1_reader_fails_closed_on_truncation(
    recode: g85p.PortableSemanticRecodeV1,
) -> None:
    semantic_members = g85p._read_store_zip(
        recode.semantic_p_archive,
        expected_names=(
            "manifest.json",
            "predictor.zip",
            "predict/movable_polygon_worldsheet.g1s",
            "render/receiver_realization.ddrp",
            "render/scorer_solved_templates.ddst",
        ),
    )
    g1 = semantic_members["predict/movable_polygon_worldsheet.g1s"]
    with pytest.raises(g85p.PortableSemanticRecodeError, match="truncated"):
        g85p._read_portable_g1_sections(g1[:8])
    with pytest.raises(g85p.PortableSemanticRecodeError, match="truncated"):
        g85p._read_portable_g1_sections(g1[:-1])


def test_nonsemantic_input_is_refused() -> None:
    with pytest.raises(g85p.PortableSemanticRecodeError):
        g85p.transcode_portable_semantic_p(b"not-a-semantic-program")
