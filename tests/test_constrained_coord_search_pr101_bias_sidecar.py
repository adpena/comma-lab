"""Tests for ``tools/constrained_coord_search_pr101_bias_sidecar.py``.

Per CLAUDE.md "Subagent coherence-by-default" + lesson 11 (no-op detector for
inflate-time variants) + the convergent finding from a3c89347's V2 regression.

Coverage:

- Coord convention: ``coord = -1.0`` produces ``sub_(1.0)`` (PR101 baseline).
- Coord convention: ``coord = +1.0`` produces ``sub_(-1.0)`` (effective add).
- All-zero coords skip lines (degenerate no-op).
- 4D-grid sidecar appends a 4th line on a valid (frame, channel) cell.
- Coarse and coarse-coarse grid sizes are correct.
- Refined grid centers around ``--center-coord``.
- Variant ID encoding is reversible enough to round-trip into a valid
  inflate.py.
- Inflate.py builder respects the canonical anchor block.
"""
# ROUNDTRIP_TESTED:test_constrained_coord_search_pr101_bias_sidecar.py
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL_PATH = REPO_ROOT / "tools/constrained_coord_search_pr101_bias_sidecar.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "constrained_coord_search_pr101_bias_sidecar", TOOL_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["constrained_coord_search_pr101_bias_sidecar"] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


# ---------------------------------------------------------------------------
# Coord convention tests (the V1/V2 anchors)
# ---------------------------------------------------------------------------


def test_coord_neg_one_produces_pr101_baseline(tool):
    """coord = -1.0 → sub_(1.0) — matches PR101's verified baseline.

    The canonical PR101 inflate has:
        up[:, 0, 0].sub_(1.0)
        up[:, 0, 2].sub_(1.0)
        up[:, 1, 1].sub_(1.0)
    """
    lines = tool.build_bias_lines(c0_0=-1.0, c0_2=-1.0, c1_1=-1.0)
    assert len(lines) == 3
    assert lines[0] == "up[:, 0, 0].sub_(1.000000)"
    assert lines[1] == "up[:, 0, 2].sub_(1.000000)"
    assert lines[2] == "up[:, 1, 1].sub_(1.000000)"


def test_coord_neg_half_produces_v2_regression_anchor(tool):
    """coord = -0.5 → sub_(0.5) — matches the V2 half-magnitude regression."""
    lines = tool.build_bias_lines(c0_0=-0.5, c0_2=-0.5, c1_1=-0.5)
    assert lines == [
        "up[:, 0, 0].sub_(0.500000)",
        "up[:, 0, 2].sub_(0.500000)",
        "up[:, 1, 1].sub_(0.500000)",
    ]


def test_coord_pos_one_produces_effective_add(tool):
    """coord = +1.0 → sub_(-1.0), which is mathematically add_(1.0)."""
    lines = tool.build_bias_lines(c0_0=1.0, c0_2=0.0, c1_1=0.0)
    assert lines == ["up[:, 0, 0].sub_(-1.000000)"]


def test_coord_zero_skips_line(tool):
    lines = tool.build_bias_lines(c0_0=0.0, c0_2=0.0, c1_1=0.0)
    assert lines == []


def test_coord_partial_zero_skips_only_that_line(tool):
    lines = tool.build_bias_lines(c0_0=-1.0, c0_2=0.0, c1_1=-0.5)
    assert lines == [
        "up[:, 0, 0].sub_(1.000000)",
        "up[:, 1, 1].sub_(0.500000)",
    ]


# ---------------------------------------------------------------------------
# Sidecar (4th coord) tests
# ---------------------------------------------------------------------------


def test_sidecar_appends_fourth_line_on_valid_cell(tool):
    """The sidecar 4th coord uses up[:, 1, 0] (frame 1 red), not up[:, 2, 0]
    (which is OUT OF RANGE — frames are only 0 or 1)."""
    lines = tool.build_bias_lines(c0_0=-1.0, c0_2=-1.0, c1_1=-1.0, sidecar_c2_0=0.25)
    assert len(lines) == 4
    assert lines[-1] == "up[:, 1, 0].sub_(-0.250000)"


def test_sidecar_zero_does_not_append(tool):
    lines = tool.build_bias_lines(c0_0=-1.0, c0_2=-1.0, c1_1=-1.0, sidecar_c2_0=0.0)
    assert len(lines) == 3
    assert all("up[:, 1, 0]" not in ln for ln in lines)


def test_sidecar_none_does_not_append(tool):
    lines = tool.build_bias_lines(c0_0=-1.0, c0_2=-1.0, c1_1=-1.0, sidecar_c2_0=None)
    assert len(lines) == 3


# ---------------------------------------------------------------------------
# Variant id encoding
# ---------------------------------------------------------------------------


def test_variant_id_encoding_is_filename_safe(tool):
    """Variant IDs must be valid filenames and reversible into the coord triple."""
    variants = tool.enumerate_variants(
        grid_c0_0=(-1.0, 0.0),
        grid_c0_2=(-0.5,),
        grid_c1_1=(0.5,),
        grid_sidecar=None,
    )
    assert len(variants) == 2
    ids = [vid for vid, _ in variants]
    assert ids == [
        "v_n1_00_n0_50_p0_50",
        "v_p0_00_n0_50_p0_50",
    ]


def test_variant_id_with_sidecar_includes_4th_dim(tool):
    variants = tool.enumerate_variants(
        grid_c0_0=(-1.0,),
        grid_c0_2=(-1.0,),
        grid_c1_1=(-1.0,),
        grid_sidecar=(0.25,),
    )
    assert variants[0][0] == "v_n1_00_n1_00_n1_00_scp0_25"
    assert variants[0][1] == {
        "c0_0": -1.0,
        "c0_2": -1.0,
        "c1_1": -1.0,
        "sidecar_c2_0": 0.25,
    }


# ---------------------------------------------------------------------------
# Grid enumeration
# ---------------------------------------------------------------------------


def test_default_coarse_grid_size_is_343(tool):
    """7³ = 343 candidates on the coarse grid."""
    variants = tool.enumerate_variants(
        grid_c0_0=tool.DEFAULT_COARSE_GRID,
        grid_c0_2=tool.DEFAULT_COARSE_GRID,
        grid_c1_1=tool.DEFAULT_COARSE_GRID,
        grid_sidecar=None,
    )
    assert len(variants) == 7 ** 3 == 343


def test_default_coarse_coarse_grid_size_is_64(tool):
    """4³ = 64 candidates on the coarse-coarse grid."""
    variants = tool.enumerate_variants(
        grid_c0_0=tool.DEFAULT_COARSE_COARSE_GRID,
        grid_c0_2=tool.DEFAULT_COARSE_COARSE_GRID,
        grid_c1_1=tool.DEFAULT_COARSE_COARSE_GRID,
        grid_sidecar=None,
    )
    assert len(variants) == 4 ** 3 == 64


def test_coarse_with_sidecar_is_343x5_eq_1715(tool):
    """7³ × 5 = 1715 — matches the operator's brief total."""
    variants = tool.enumerate_variants(
        grid_c0_0=tool.DEFAULT_COARSE_GRID,
        grid_c0_2=tool.DEFAULT_COARSE_GRID,
        grid_c1_1=tool.DEFAULT_COARSE_GRID,
        grid_sidecar=tool.DEFAULT_SIDECAR_GRID,
    )
    assert len(variants) == 7 ** 3 * 5 == 1715


def test_coarse_coarse_with_sidecar_is_64x5_eq_320(tool):
    variants = tool.enumerate_variants(
        grid_c0_0=tool.DEFAULT_COARSE_COARSE_GRID,
        grid_c0_2=tool.DEFAULT_COARSE_COARSE_GRID,
        grid_c1_1=tool.DEFAULT_COARSE_COARSE_GRID,
        grid_sidecar=tool.DEFAULT_SIDECAR_GRID,
    )
    assert len(variants) == 4 ** 3 * 5 == 320


# ---------------------------------------------------------------------------
# Inflate.py builder
# ---------------------------------------------------------------------------


@pytest.fixture
def a1_inflate_template(tool):
    """Read the actual A1 inflate.py template."""
    template_path = tool.A1_SUBMISSION_DIR / "inflate.py"
    if not template_path.exists():
        pytest.skip(f"A1 inflate.py template missing at {template_path}")
    return template_path.read_text()


def test_build_inflate_py_replaces_anchor(tool, a1_inflate_template):
    """Verify the inflate.py builder injects bias lines at the canonical anchor."""
    new_lines = ["up[:, 0, 0].sub_(0.42)"]
    new_text = tool.build_inflate_py(a1_inflate_template, new_lines)
    assert "up[:, 0, 0].sub_(0.42)" in new_text
    # The anchor line itself must still be present.
    assert tool.ANCHOR_START in new_text
    # The frames-conversion line must still be present.
    assert tool.ANCHOR_END_PREFIX in new_text


def test_build_inflate_py_baseline_matches_pr101_form(tool, a1_inflate_template):
    """When the bias_lines reproduce PR101's three lines, the rendered
    inflate.py should contain them inside the canonical loop body."""
    new_lines = [
        "up[:, 0, 0].sub_(1.0)",
        "up[:, 0, 2].sub_(1.0)",
        "up[:, 1, 1].sub_(1.0)",
    ]
    new_text = tool.build_inflate_py(a1_inflate_template, new_lines)
    # Lines should appear as a single block of 3 consecutive sub_ statements.
    block = (
        "            up[:, 0, 0].sub_(1.0)\n"
        "            up[:, 0, 2].sub_(1.0)\n"
        "            up[:, 1, 1].sub_(1.0)\n"
    )
    assert block in new_text


def test_build_inflate_py_empty_bias_lines_produces_empty_block(tool, a1_inflate_template):
    """coord_zero produces zero bias lines — the rendered inflate.py should
    have NO lines between the reshape anchor and the frames conversion."""
    new_text = tool.build_inflate_py(a1_inflate_template, [])
    # The anchor and the frames= line should appear back-to-back.
    lines = new_text.splitlines()
    anchor_idx = next(i for i, ln in enumerate(lines) if ln == tool.ANCHOR_START)
    next_line = lines[anchor_idx + 1]
    assert next_line.startswith(tool.ANCHOR_END_PREFIX)


# ---------------------------------------------------------------------------
# Smoke: build a single variant and verify the output structure
# ---------------------------------------------------------------------------


def test_smoke_write_one_variant(tool, tmp_path):
    """Build one variant end-to-end and verify the submission_dir layout."""
    if not tool.A1_ARCHIVE_PATH.exists():
        pytest.skip(f"A1 archive missing at {tool.A1_ARCHIVE_PATH}")
    template_text = (tool.A1_SUBMISSION_DIR / "inflate.py").read_text()
    bias_lines = tool.build_bias_lines(c0_0=-1.0, c0_2=-1.0, c1_1=-1.0)
    manifest = tool.write_variant(
        variant_id="v_n1_00_n1_00_n1_00_smoke",
        bias_lines=bias_lines,
        coords={"c0_0": -1.0, "c0_2": -1.0, "c1_1": -1.0},
        out_root=tmp_path,
        inflate_template=template_text,
    )
    assert manifest["archive_size_bytes"] == tool.A1_EXPECTED_ARCHIVE_BYTES
    assert manifest["archive_sha256"] == tool.A1_EXPECTED_ARCHIVE_SHA
    assert manifest["bias_spec"]["n_bias_lines"] == 3
    # Verify written submission_dir layout.
    sub_dir = tmp_path / "v_n1_00_n1_00_n1_00_smoke" / "submission_dir"
    assert (sub_dir / "archive.zip").exists()
    assert (sub_dir / "inflate.py").exists()
    assert (sub_dir / "inflate.sh").exists()
    assert (sub_dir / "src" / "model.py").exists()
    assert (sub_dir / "src" / "codec.py").exists()


def test_anchors_match_a1_sweep_tool(tool):
    """The anchor constants must match the canonical A1 sweep tool's anchors
    so both tools insert at the same location in the inflate.py loop."""
    a1_tool_path = REPO_ROOT / "tools/build_a1_inflate_time_bias_correction_sweep.py"
    if not a1_tool_path.exists():
        pytest.skip("A1 sweep tool not present")
    a1_text = a1_tool_path.read_text()
    assert tool.ANCHOR_START in a1_text, \
        "constrained-coord-search ANCHOR_START must match build_a1_inflate_time_bias_correction_sweep"
    assert tool.ANCHOR_END_PREFIX in a1_text, \
        "constrained-coord-search ANCHOR_END_PREFIX must match build_a1_inflate_time_bias_correction_sweep"
