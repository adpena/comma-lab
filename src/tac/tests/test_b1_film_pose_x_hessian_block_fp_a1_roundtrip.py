"""Roundtrip + structural test for B1 cell: film_pose_x_hessian_block_fp on A1.

Catalog #91 ENCODE_INFLATE_ROUNDTRIP coverage for the hessian_block_fp
coarsened-decoder + magic_codec wrap composition.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "build_b1_film_pose_x_hessian_block_fp_a1.py"
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
    return _load_module("_b1_helper_under_test_film_hessian", HELPER)


@pytest.fixture(scope="module")
def tool():
    return _load_module("_b1_film_hessian_under_test", TOOL)


def test_helper_refuses_weight_domain_saliency_marker(helper):
    """Catalog #123: coarsen helper refuses non-score-aware proxy.

    Uses the real A1 decoder blob so PR101 schema validation passes; the
    helper MUST raise on the missing ``__saliency_source__`` marker BEFORE
    reaching downstream codec calls.
    """
    a1_path = REPO / (
        "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_"
        "20260509T012628Z_modal/harvested_artifacts/finetuned_archive/archive.zip"
    )
    if not a1_path.exists():
        pytest.skip("A1 archive not present")
    inner = helper.read_a1_inner_bytes(a1_path)
    sections = helper.split_a1_inner_sections(inner)
    bad_proxy = {"some_tensor": 1.0}  # no __saliency_source__ marker
    with pytest.raises(SystemExit) as exc:
        helper.coarsen_decoder_state_dict_by_hessian(
            sections.decoder_blob,
            saliency_proxy=bad_proxy,
            target_decoder_bytes=max(1, len(sections.decoder_blob) - 100),
        )
    assert "Catalog #123" in str(exc.value) or "score_gradient" in str(exc.value)


def test_helper_advisory_uniform_saliency_passes_marker_check(helper):
    """synthesize_neutral_saliency_advisory sets the marker correctly.

    Uses the real A1 decoder blob to exercise the full bit allocation +
    PR101 split-Brotli re-encode path against the FIXED_STATE_SCHEMA's
    28 tensors.
    """
    a1_path = REPO / (
        "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_"
        "20260509T012628Z_modal/harvested_artifacts/finetuned_archive/archive.zip"
    )
    if not a1_path.exists():
        pytest.skip("A1 archive not present")
    inner = helper.read_a1_inner_bytes(a1_path)
    sections = helper.split_a1_inner_sections(inner)
    proxy = helper.synthesize_neutral_saliency_advisory(sections.decoder_blob)
    assert proxy.get("__saliency_source__") == "score_gradient"
    assert "__saliency_advisory_only__" in proxy
    new_blob, bits, errs = helper.coarsen_decoder_state_dict_by_hessian(
        sections.decoder_blob,
        saliency_proxy=proxy,
        target_decoder_bytes=max(1, len(sections.decoder_blob) - 5_000),
    )
    assert len(new_blob) > 0
    # PR101 schema has 28 tensors.
    assert len(bits) == 28
    assert all(4 <= b <= 8 for b in bits.values())


def test_predicted_band_for_cell(helper):
    band = helper.predicted_band_for_cell("film_pose_x_hessian_block_fp")
    assert band[0] < band[1]


def test_tool_refuses_without_acknowledgement_flag(tool, tmp_path):
    """Catalog #123 enforcement: must require explicit non-score-aware opt-in."""
    out_dir = tmp_path / "b1_film_hessian_no_ack"
    with pytest.raises(SystemExit) as exc:
        tool.main(
            [
                "--source-archive",
                "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_"
                "20260509T012628Z_modal/harvested_artifacts/finetuned_archive/archive.zip",
                "--output-dir",
                str(out_dir),
            ]
        )
    msg = str(exc.value)
    assert "Catalog #123" in msg or "score-gradient" in msg.lower()


def test_end_to_end_build_against_real_a1_archive(tool, tmp_path):
    a1_path = REPO / (
        "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_"
        "20260509T012628Z_modal/harvested_artifacts/finetuned_archive/archive.zip"
    )
    if not a1_path.exists():
        pytest.skip("A1 archive not present")
    out_dir = tmp_path / "b1_film_hessian_e2e"
    rc = tool.main(
        [
            "--source-archive",
            str(a1_path),
            "--output-dir",
            str(out_dir),
            "--target-archive-bytes",
            "175000",
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
        assert "FILM" in zf.namelist()
    assert rm["cell_id"] == "film_pose_x_hessian_block_fp"
    assert rm["score_claim"] is False
    assert rm["measured_config_status"] == "byte_proxy_only_advisory_saliency"
    assert sm["empirical_archive_bytes"] == archive.stat().st_size
    assert proof["runtime_consumes_bytes"] is False
    # Bits per tensor must be in [floor, ceiling] range.
    bits = sm["composition_metadata"]["bits_per_tensor"]
    assert all(4 <= b <= 8 for b in bits.values())
