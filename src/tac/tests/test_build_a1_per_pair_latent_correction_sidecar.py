"""Tests for the A1 per-pair latent sidecar resampling helper."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "build_a1_per_pair_latent_correction_sidecar.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("a1_sidecar_tool", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_packed_sidecar_encodes_high_dims_that_uint8_layout_cannot() -> None:
    tool = load_tool()
    dims = np.full(tool.N_PAIRS, 255, dtype=np.int64)
    delta_idx = np.full(tool.N_PAIRS, -1, dtype=np.int64)
    dims[0] = 27
    delta_idx[0] = 15

    packed = tool.encode_sidecar_huff_enum(dims, delta_idx)

    assert len(packed) == 661
    assert packed != b"\x00" * 661


def test_uint8_sidecar_rejects_high_dim_choices() -> None:
    tool = load_tool()
    dims = np.full(tool.N_PAIRS, 255, dtype=np.int64)
    delta_idx = np.full(tool.N_PAIRS, -1, dtype=np.int64)
    dims[0] = 27
    delta_idx[0] = 15

    try:
        tool.encode_sidecar_n_pairs(dims, delta_idx)
    except ValueError as exc:
        assert "cannot encode" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected high-dim uint8 sidecar choice to fail")


def test_manifest_path_allows_external_outputs(tmp_path: Path) -> None:
    tool = load_tool()

    assert tool.manifest_path(REPO_ROOT / "tools" / "x.py") == "tools/x.py"
    assert tool.manifest_path(tmp_path / "x.py") == str(tmp_path / "x.py")


def test_ground_truth_loader_uses_upstream_yuv420_helper() -> None:
    tool_text = TOOL_PATH.read_text()

    assert "load_upstream_yuv420_to_rgb" in tool_text
    assert "yuv420_to_rgb(f)" in tool_text
    assert "to_ndarray(format=\"rgb24\")" not in tool_text
