# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.proxy_candidate_contract import validate_proxy_candidate
from tac.substrates.coord_mlp_residual_sidecar import (
    CoordMlpPatch,
    CoordMlpResidualSidecarError,
    CoordMlpResidualWeights,
    apply_sidecar_to_rgb,
    build_readiness_manifest,
    pack_sidecar,
    parse_sidecar,
)


def _weights(*, red_bias: int = 16, hidden_dim: int = 2) -> CoordMlpResidualWeights:
    return CoordMlpResidualWeights(
        w1_int8=np.zeros((hidden_dim, 3), dtype=np.int8),
        b1_int16=np.zeros((hidden_dim,), dtype=np.int16),
        w2_int8=np.zeros((3, hidden_dim), dtype=np.int8),
        b2_int16=np.array([red_bias, 0, 0], dtype=np.int16),
    )


def _sidecar(*, red_bias: int = 16) -> bytes:
    return pack_sidecar(
        (CoordMlpPatch(frame_index=0, y=1, x=1, height=2, width=2),),
        _weights(red_bias=red_bias),
        metadata={"test_vector": "h15_coord_mlp_residual_sidecar"},
    )


def test_pack_parse_roundtrip_is_deterministic_and_sections_are_charged() -> None:
    blob_a = _sidecar(red_bias=16)
    blob_b = _sidecar(red_bias=16)
    parsed = parse_sidecar(blob_a)

    assert blob_a == blob_b
    assert parsed.to_bytes() == blob_a
    assert parsed.hidden_dim == 2
    assert parsed.patches == (CoordMlpPatch(0, 1, 1, 2, 2),)
    assert parsed.charged_bytes == len(blob_a)
    assert [section.name for section in parsed.sections] == [
        "HEADER",
        "PATCH_TABLE",
        "WEIGHT_BLOB",
        "META_JSON",
    ]
    assert all(section.charged for section in parsed.sections)
    assert parsed.metadata["score_claim"] is False
    assert parsed.metadata["ready_for_exact_eval_dispatch"] is False
    assert parsed.metadata["scorer_at_inflate"] is False


def test_inflate_consumes_sidecar_bytes_and_mutation_changes_output() -> None:
    base = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    blob_red_1 = _sidecar(red_bias=16)
    blob_red_2 = _sidecar(red_bias=32)

    result_1 = apply_sidecar_to_rgb(base, blob_red_1)
    result_2 = apply_sidecar_to_rgb(base, blob_red_2)

    assert result_1.consumed_bytes == len(blob_red_1)
    assert result_1.consumed_sha256 == hashlib.sha256(blob_red_1).hexdigest()
    assert result_1.consumed_sections == (
        "HEADER",
        "PATCH_TABLE",
        "WEIGHT_BLOB",
        "META_JSON",
    )
    assert result_1.applied_pixels == 4
    assert np.all(result_1.frames[0, 1:3, 1:3, 0] == 1)
    assert np.all(result_1.frames[0, :, :, 1:] == 0)
    assert np.all(result_2.frames[0, 1:3, 1:3, 0] == 2)
    assert not np.array_equal(result_1.frames, result_2.frames)


def test_noop_payload_fails_inflate_and_is_not_dispatch_ready() -> None:
    noop = pack_sidecar(
        (CoordMlpPatch(frame_index=0, y=0, x=0, height=1, width=1),),
        _weights(red_bias=0),
    )
    manifest = build_readiness_manifest(noop)

    assert manifest["structural_noop"] is True
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "coord_mlp_residual_sidecar_structural_noop" in manifest["dispatch_blockers"]
    with pytest.raises(CoordMlpResidualSidecarError, match="structural no-op"):
        apply_sidecar_to_rgb(np.zeros((1, 2, 2, 3), dtype=np.uint8), noop)


def test_readiness_blocks_exact_eval_until_archive_runtime_custody_exists() -> None:
    manifest = build_readiness_manifest(_sidecar())

    assert manifest["charged_section"] is True
    assert manifest["charged_bytes"] > 0
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["rank_or_kill_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["dispatch_attempted"] is False
    assert validate_proxy_candidate(manifest) == []
    assert "archive_custody_missing" in manifest["dispatch_blockers"]
    assert "runtime_tree_custody_missing" in manifest["dispatch_blockers"]
    assert (
        "coord_mlp_sidecar_requires_byte_closed_exact_eval_adjudication"
        in manifest["dispatch_blockers"]
    )


def test_false_authority_metadata_is_rejected() -> None:
    with pytest.raises(CoordMlpResidualSidecarError, match="score_claim"):
        pack_sidecar(
            (CoordMlpPatch(frame_index=0, y=0, x=0, height=1, width=1),),
            _weights(),
            metadata={"score_claim": True},
        )


def test_parser_rejects_trailing_bytes() -> None:
    with pytest.raises(CoordMlpResidualSidecarError, match="trailing bytes"):
        parse_sidecar(_sidecar() + b"x")


def test_inflate_helper_has_no_scorer_imports() -> None:
    source = Path(__file__).parents[1].joinpath("inflate.py").read_text()
    forbidden = ("tac.scorer", "upstream.evaluate", "PoseNet", "SegNet")
    for token in forbidden:
        assert token not in source


def test_probe_cli_writes_proxy_safe_consumed_byte_manifest(tmp_path) -> None:
    output = tmp_path / "probe.json"
    repo_root = Path(__file__).resolve().parents[5]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools/probe_coord_mlp_residual_sidecar.py"),
            "--output",
            str(output),
        ],
        cwd=repo_root,
        env={
            **os.environ,
            "PYTHONPATH": f"{repo_root / 'src'}:{repo_root / 'upstream'}:{repo_root}",
        },
        text=True,
        capture_output=True,
        check=True,
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert "wrote coord_mlp_residual_sidecar_probe" in proc.stdout
    assert manifest["frames_changed"] is True
    assert manifest["readiness"]["score_claim"] is False
    assert manifest["readiness"]["ready_for_exact_eval_dispatch"] is False
    assert manifest["inflate_consumption"]["consumed_bytes"] == manifest["readiness"]["charged_bytes"]
    assert manifest["inflate_consumption"]["scorer_at_inflate"] is False
