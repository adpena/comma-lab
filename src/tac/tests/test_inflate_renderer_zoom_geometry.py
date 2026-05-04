from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
INFLATE_RENDERER_PATH = REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer.py"


def _load_inflate_renderer():
    spec = importlib.util.spec_from_file_location(
        "inflate_renderer_zoom_geometry_test",
        INFLATE_RENDERER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _half_frame_masks(n_pairs: int) -> torch.Tensor:
    masks = torch.zeros(n_pairs, 384, 512, dtype=torch.long)
    masks._half_frame_only = True  # type: ignore[attr-defined]
    return masks


def test_half_frame_masks_load_charged_zoom_even_when_renderer_does_not_use_ego_flow(
    tmp_path: Path,
) -> None:
    inflate = _load_inflate_renderer()
    masks = _half_frame_masks(3)
    scalars = torch.tensor([0.05, -0.025, 0.0], dtype=torch.float16)
    (tmp_path / "zoom_scalars.bin").write_bytes(scalars.numpy().tobytes())
    renderer = torch.nn.Module()
    renderer.use_zoom_flow = False

    zoom_warp = inflate._load_zoom_warp_from_archive_dir(
        tmp_path,
        masks=masks,
        renderer=renderer,
        device="cpu",
    )

    assert zoom_warp is not None
    torch.testing.assert_close(
        zoom_warp.zoom_scalars.detach().cpu(),
        scalars.float(),
        rtol=0,
        atol=0,
    )


def test_half_frame_zoom_pair_count_mismatch_fails_closed(tmp_path: Path) -> None:
    inflate = _load_inflate_renderer()
    masks = _half_frame_masks(3)
    scalars = torch.tensor([0.0, 0.0], dtype=torch.float16)
    (tmp_path / "zoom_scalars.bin").write_bytes(scalars.numpy().tobytes())
    renderer = torch.nn.Module()
    renderer.use_zoom_flow = False

    with pytest.raises(RuntimeError, match="pair count mismatch"):
        inflate._load_zoom_warp_from_archive_dir(
            tmp_path,
            masks=masks,
            renderer=renderer,
            device="cpu",
        )

