"""Tests for the A1 per-pair latent sidecar resampling helper."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


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


def test_infer_sidecar_choices_preserves_old_unsearched_pairs() -> None:
    tool = load_tool()
    base = torch.zeros((tool.N_PAIRS, tool.LATENT_DIM), dtype=torch.float32)
    old = base.clone()
    old[0, 3] += 0.04
    old[10, 27] -= 0.10

    dims, delta_idx = tool.infer_sidecar_choices_from_latents(
        {
            "latents_pre_sidecar": base,
            "latents_with_sidecar_old": old,
        }
    )

    assert dims[0] == 3
    assert tool.SIDECAR_DELTAS_X100[delta_idx[0]] == 4
    assert dims[10] == 27
    assert tool.SIDECAR_DELTAS_X100[delta_idx[10]] == -10
    assert dims[1] == 255
    assert delta_idx[1] == -1


def test_infer_sidecar_rejects_multi_dim_pair_delta() -> None:
    tool = load_tool()
    base = torch.zeros((tool.N_PAIRS, tool.LATENT_DIM), dtype=torch.float32)
    old = base.clone()
    old[0, 0] += 0.02
    old[0, 1] += 0.03

    with pytest.raises(ValueError, match="<=1 changed latent dim"):
        tool.infer_sidecar_choices_from_latents(
            {
                "latents_pre_sidecar": base,
                "latents_with_sidecar_old": old,
            }
        )


def test_manifest_readiness_fails_closed_without_materialized_archive(tmp_path: Path) -> None:
    tool = load_tool()
    manifest = {
        "dispatch_blockers": [],
        "new_archive_sha256": "missing",
        "new_archive_bytes": 123,
        "ready_for_exact_eval_dispatch": True,
        "runtime_smoke_checked": True,
        "smoke_only": False,
        "no_op_detector": {"sidecar_changed": True},
    }

    out = tool.enforce_manifest_dispatch_readiness(
        manifest,
        archive_path=tmp_path / "submission_dir" / "archive.zip",
    )

    assert out["ready_for_exact_eval_dispatch"] is False
    assert any(
        item.startswith("materialized_archive_missing:")
        for item in out["dispatch_blockers"]
    )


def test_manifest_readiness_fails_closed_for_smoke_or_no_runtime_smoke(tmp_path: Path) -> None:
    tool = load_tool()
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    manifest = {
        "dispatch_blockers": [],
        "new_archive_sha256": tool.sha256_of(archive),
        "new_archive_bytes": archive.stat().st_size,
        "ready_for_exact_eval_dispatch": True,
        "runtime_smoke_checked": False,
        "smoke_only": True,
        "no_op_detector": {"sidecar_changed": True},
    }

    out = tool.enforce_manifest_dispatch_readiness(manifest, archive_path=archive)

    assert out["ready_for_exact_eval_dispatch"] is False
    assert "smoke_only_not_exact_eval_ready" in out["dispatch_blockers"]
    assert "runtime_smoke_not_checked" in out["dispatch_blockers"]
