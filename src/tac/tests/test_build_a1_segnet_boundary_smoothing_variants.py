# SPDX-License-Identifier: MIT
"""Tests for the A1 SegNet boundary smoothing inflate-time variants builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "build_a1_segnet_boundary_smoothing_variants.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("a1_smoothing_tool", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_inflate_py_inserts_smoothing_after_bias_block() -> None:
    tool = load_tool()
    template = "\n".join(
        [
            "def inflate():",
            "    with torch.inference_mode():",
            "        for i in range(1):",
            "            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)",
            "            up[:, 0, 0].sub_(1.0)",
            "            up[:, 0, 2].sub_(1.0)",
            "            up[:, 1, 1].sub_(1.0)",
            "            frames = (",
            "                up.reshape(batch * 2, 3, CAMERA_H, CAMERA_W)",
            "            )",
            "",
        ]
    )

    out = tool.build_inflate_py(template, ["up_smooth = boundary_filter(up)"])

    # Bias block must be PRESERVED (load-bearing for A1's score)
    assert "up[:, 0, 0].sub_(1.0)" in out
    assert "up[:, 0, 2].sub_(1.0)" in out
    assert "up[:, 1, 1].sub_(1.0)" in out
    # Smoothing block must be inserted AFTER the last bias line
    assert "up_smooth = boundary_filter(up)" in out
    bias_pos = out.index("up[:, 1, 1].sub_(1.0)")
    smooth_pos = out.index("up_smooth = boundary_filter(up)")
    frames_pos = out.index("            frames = (")
    assert bias_pos < smooth_pos < frames_pos, (
        f"smoothing block must be between bias and frames; got "
        f"bias={bias_pos} smooth={smooth_pos} frames={frames_pos}"
    )


def test_build_inflate_py_v_baseline_returns_template_unchanged() -> None:
    tool = load_tool()
    template = "\n".join(
        [
            "            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)",
            "            up[:, 0, 0].sub_(1.0)",
            "            up[:, 0, 2].sub_(1.0)",
            "            up[:, 1, 1].sub_(1.0)",
            "            frames = (",
            "                up.reshape(batch * 2, 3, CAMERA_H, CAMERA_W)",
            "            )",
            "",
        ]
    )

    out = tool.build_inflate_py(template, [])

    # V_baseline: no smoothing → exact template echo
    assert out == template


def test_build_inflate_py_raises_if_anchor_missing() -> None:
    tool = load_tool()
    template = "\n".join(
        [
            "def inflate():",
            "    pass",
            "",
        ]
    )
    with pytest.raises(RuntimeError, match="could not find anchor line"):
        tool.build_inflate_py(template, ["dummy"])


def test_build_inflate_py_raises_if_frames_anchor_missing() -> None:
    tool = load_tool()
    # Has bias-block but no frames= within 5 lines
    template = "\n".join(
        [
            "            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)",
            "            up[:, 0, 0].sub_(1.0)",
            "            up[:, 0, 2].sub_(1.0)",
            "            up[:, 1, 1].sub_(1.0)",
            "            # no frames anchor in the next 5 lines",
            "            other = stuff",
            "            more = code",
            "            and = more",
            "            and = more",
            "            and = more",
            "            and = more",
            "            and = more",
            "            and = more",
            "            and = more",
            "            and = more",
            "",
        ]
    )
    with pytest.raises(RuntimeError, match="expected.*frames"):
        tool.build_inflate_py(template, ["dummy"])


def test_variants_table_has_expected_ids() -> None:
    tool = load_tool()
    ids = [v["variant_id"] for v in tool.VARIANTS]
    assert ids == [
        "v_baseline",
        "v_smooth_3x3",
        "v_smooth_5x5",
        "v_smooth_class_aware",
    ]
    # Baseline must have empty smoothing
    baseline = next(v for v in tool.VARIANTS if v["variant_id"] == "v_baseline")
    assert baseline["smoothing_lines"] == []
    assert baseline["smoothing_kernel_h"] == 0
    # Smoothing variants must have non-empty smoothing lines
    for vid in ["v_smooth_3x3", "v_smooth_5x5", "v_smooth_class_aware"]:
        v = next(x for x in tool.VARIANTS if x["variant_id"] == vid)
        assert v["smoothing_lines"], f"{vid} must have non-empty smoothing_lines"
        assert v["smoothing_kernel_h"] > 0


def test_manifest_path_repo_relative_and_external() -> None:
    tool = load_tool()
    assert tool.manifest_path(REPO_ROOT / "tools" / "x.py") == "tools/x.py"
    # External path returns absolute string
    assert tool.manifest_path(Path("/var/tmp/x.py")) == "/var/tmp/x.py"


def test_predicted_delta_band_is_two_tuple() -> None:
    tool = load_tool()
    for v in tool.VARIANTS:
        band = v["predicted_delta_band"]
        assert isinstance(band, tuple) and len(band) == 2
        lo, hi = band
        assert isinstance(lo, (int, float))
        assert isinstance(hi, (int, float))
        assert lo <= hi, f"variant {v['variant_id']} has lo > hi: {band}"


def test_write_variant_packages_full_submission_dir(tmp_path: Path) -> None:
    """Verify write_variant produces all required files in the submission dir."""
    tool = load_tool()
    if not tool.A1_ARCHIVE_PATH.exists():
        pytest.skip("A1 archive not present in this checkout")

    template = (tool.A1_SUBMISSION_DIR / "inflate.py").read_text()
    manifest = tool.write_variant(
        tool.VARIANTS[0],  # V_baseline (faster, no no-op detector run)
        tmp_path,
        "TEST",
        template,
        run_no_op=False,
    )
    out_dir = tmp_path / "segnet_boundary_smoothing_v_baseline_TEST" / "submission_dir"
    assert (out_dir / "archive.zip").exists()
    assert (out_dir / "inflate.py").exists()
    assert (out_dir / "inflate.sh").exists()
    assert (out_dir / "src" / "codec.py").exists()
    assert (out_dir / "src" / "model.py").exists()

    # archive.zip must be bit-identical to A1
    assert manifest["archive_sha256"] == tool.A1_EXPECTED_ARCHIVE_SHA
    assert manifest["archive_size_bytes"] == tool.A1_EXPECTED_ARCHIVE_BYTES
    assert manifest["archive_unchanged_from_a1"] is True
    # inflate.sh must be executable
    assert (out_dir / "inflate.sh").stat().st_mode & 0o111

    # Manifest custody fields
    assert manifest["lane_id"] == "lane_a1_segnet_boundary_smoothing_inflate"
    assert manifest["domain_catalog_atom_id"] == "domain_exploit_scorer_1"
    assert manifest["score_claim"] is False
    assert (
        manifest["evidence_grade"]
        == "[predicted; SegNet boundary smoothing on A1 substrate; pre-GHA-dispatch]"
    )
    # V_baseline: ready_for_exact_eval_dispatch is True because the variant
    # bytes are identical to A1 (no smoothing to verify; no-op detector skipped
    # but treated as PASSED for the baseline case).
    assert manifest["ready_for_exact_eval_dispatch"] is True
    assert manifest["no_op_detector_passed"] is True
    assert manifest["no_op_detector_result"].get("skipped") is True


def test_write_variant_smoothing_with_no_op_detector_passes(tmp_path: Path) -> None:
    """End-to-end: verify the no-op detector confirms smoothing changes pixels."""
    tool = load_tool()
    if not tool.A1_ARCHIVE_PATH.exists():
        pytest.skip("A1 archive not present in this checkout")
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        pytest.skip(".venv not present (no-op detector requires venv python)")

    template = (tool.A1_SUBMISSION_DIR / "inflate.py").read_text()
    smooth_3x3 = next(v for v in tool.VARIANTS if v["variant_id"] == "v_smooth_3x3")
    manifest = tool.write_variant(
        smooth_3x3,
        tmp_path,
        "TEST",
        template,
        run_no_op=True,
    )
    # The no-op detector renders 1 pair (2 frames). It should pass with > 0
    # differing bytes (smoothing is consumed).
    nop = manifest["no_op_detector_result"]
    assert manifest["no_op_detector_passed"] is True
    if "diff_ratio" in nop:
        assert nop["diff_ratio"] > 0.5, f"low diff_ratio: {nop['diff_ratio']}"
