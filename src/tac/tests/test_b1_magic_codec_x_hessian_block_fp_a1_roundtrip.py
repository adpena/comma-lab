"""Roundtrip + structural test for B1 cell: magic_codec_x_hessian_block_fp on A1.

Catalog #91 ENCODE_INFLATE_ROUNDTRIP coverage. Sequential composition:
hessian_block_fp coarsening + magic_codec wrap of the coarsened decoder.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "build_b1_magic_codec_x_hessian_block_fp_a1.py"
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
    return _load_module("_b1_helper_under_test_mc_hessian", HELPER)


@pytest.fixture(scope="module")
def tool():
    return _load_module("_b1_mc_hessian_under_test", TOOL)


def test_predicted_band_for_cell(helper):
    band = helper.predicted_band_for_cell("magic_codec_x_hessian_block_fp")
    assert band[0] < band[1]


def test_tool_refuses_without_acknowledgement_flag(tool, tmp_path):
    out_dir = tmp_path / "b1_mc_hessian_no_ack"
    with pytest.raises(SystemExit):
        tool.main(
            [
                "--source-archive",
                "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_"
                "20260509T012628Z_modal/harvested_artifacts/finetuned_archive/archive.zip",
                "--output-dir",
                str(out_dir),
            ]
        )


def test_end_to_end_build_against_real_a1_archive(tool, tmp_path):
    a1_path = REPO / (
        "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_"
        "20260509T012628Z_modal/harvested_artifacts/finetuned_archive/archive.zip"
    )
    if not a1_path.exists():
        pytest.skip("A1 archive not present")
    out_dir = tmp_path / "b1_mc_hessian_e2e"
    rc = tool.main(
        [
            "--source-archive",
            str(a1_path),
            "--output-dir",
            str(out_dir),
            "--target-decoder-bytes",
            "155000",
            "--proxy-acknowledged-non-score-aware",
        ]
    )
    assert rc == 0
    archive = out_dir / "archive.zip"
    rm = json.loads((out_dir / "runtime_manifest.json").read_text())
    sm = json.loads((out_dir / "selection_manifest.json").read_text())
    proof = json.loads((out_dir / "no_op_proof.json").read_text())
    with zipfile.ZipFile(archive) as zf:
        assert "x" in zf.namelist()
        assert "FILM" not in zf.namelist()  # this cell has NO FILM slot
    assert rm["cell_id"] == "magic_codec_x_hessian_block_fp"
    assert rm["score_claim"] is False
    assert rm["measured_config_status"] == "byte_proxy_only_advisory_saliency"
    assert sm["empirical_archive_bytes"] == archive.stat().st_size
    # The composition_metadata should record both the lossy stage AND the
    # magic_codec wrap.
    cm = sm["composition_metadata"]
    assert "lossy_a" in cm and "meta_codec_b" in cm
    assert cm["coarsened_decoder_bytes_pre_magic_codec"] > 0
    assert cm["magic_codec_payload_bytes"] > 0
    assert proof["runtime_consumes_bytes"] is False
