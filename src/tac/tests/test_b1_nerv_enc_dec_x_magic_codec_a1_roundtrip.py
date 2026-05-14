# SPDX-License-Identifier: MIT
"""Roundtrip + structural test for B1 cell: nerv_enc_dec_x_magic_codec on A1.

Catalog #91 ENCODE_INFLATE_ROUNDTRIP coverage. This cell composes
nerv_enc_dec_separated (compress-time only architecture) with magic_codec.
On the archive surface the bytes are byte-identical to a singleton
magic_codec on A1 - only the composition_metadata reflects the architectural
context.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "build_b1_nerv_enc_dec_x_magic_codec_a1.py"
HELPER = REPO / "tools" / "_b1_composition_on_a1_helper.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def helper():
    return _load_module("_b1_helper_under_test_nerv", HELPER)


@pytest.fixture(scope="module")
def tool():
    return _load_module("_b1_nerv_under_test", TOOL)


def test_predicted_band_for_cell(helper):
    band = helper.predicted_band_for_cell("nerv_enc_dec_x_magic_codec")
    assert band[0] < band[1]


def test_emit_archive_without_film_slot_omits_film_member(helper, tmp_path):
    """When film_pose_slot=None, the archive should NOT contain a FILM member."""
    inner = b"\x04\x00\x00\x00" + b"\x10" * 16
    archive_path, sha, n_bytes = helper.emit_b1_archive(
        cell_id="test_no_film",
        out_dir=tmp_path,
        inner_payload=inner,
        film_pose_slot=None,
        composition_metadata={},
    )
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        assert "x" in names
        assert "FILM" not in names


def test_end_to_end_build_against_real_a1_archive(tool, tmp_path):
    a1_path = REPO / (
        "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_"
        "20260509T012628Z_modal/harvested_artifacts/finetuned_archive/archive.zip"
    )
    if not a1_path.exists():
        pytest.skip("A1 archive not present")
    out_dir = tmp_path / "b1_nerv_e2e"
    rc = tool.main(
        [
            "--source-archive",
            str(a1_path),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    archive = out_dir / "archive.zip"
    rm = json.loads((out_dir / "runtime_manifest.json").read_text())
    sm = json.loads((out_dir / "selection_manifest.json").read_text())
    proof = json.loads((out_dir / "no_op_proof.json").read_text())
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        assert "x" in names
        # NeRV cell does NOT include a FILM slot.
        assert "FILM" not in names
    assert rm["cell_id"] == "nerv_enc_dec_x_magic_codec"
    assert rm["score_claim"] is False
    assert sm["composition_metadata"]["byte_identity_to_singleton_magic_codec_on_a1"] is True
    assert proof["runtime_consumes_bytes"] is False
