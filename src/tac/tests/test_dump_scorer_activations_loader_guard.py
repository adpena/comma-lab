from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


def _load_tool():
    repo_root = Path(__file__).resolve().parents[3]
    tool_path = repo_root / "experiments" / "dump_scorer_activations.py"
    spec = importlib.util.spec_from_file_location("dump_scorer_activations", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {tool_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_real_video_cuda_path_fails_before_av_cuda_dataset(tmp_path: Path) -> None:
    mod = _load_tool()
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "0.mkv").write_bytes(b"not a real video; guard should fire first")

    with pytest.raises(RuntimeError, match="AVVideoDataset\\(device='cuda'\\)"):
        mod._gt_video_to_pair_input(tmp_path, 0, torch.device("cuda"))
