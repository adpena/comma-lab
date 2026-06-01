# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]


def _load_tool():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(
        "check_sr_nerv_resolution_axis_mirror_test",
        REPO / "tools/check_sr_nerv_resolution_axis_mirror.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_roundtrip_lowres_to_camera_preserves_shape() -> None:
    tool = _load_tool()
    x = torch.arange(2 * 2 * 3 * 8 * 10, dtype=torch.float32).reshape(
        2,
        2,
        3,
        8,
        10,
    )

    y = tool._roundtrip_lowres_to_camera(
        x,
        internal_hw=(4, 5),
        camera_hw=(8, 10),
        upsample_mode="bilinear",
    )

    assert y.shape == x.shape
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 255.0


def test_scorer_preprocess_tensors_separates_pose_and_last_frame_seg() -> None:
    tool = _load_tool()
    x = torch.zeros(1, 2, 3, 8, 10, dtype=torch.float32)
    x[:, 0, ...] = 10.0
    x[:, 1, ...] = 30.0

    def fake_rgb_to_yuv6(rgb: torch.Tensor) -> torch.Tensor:
        small = rgb[..., ::2, ::2]
        return torch.cat([small, small], dim=1)

    out = tool._scorer_preprocess_tensors(
        x,
        scorer_hw=(4, 6),
        rgb_to_yuv6=fake_rgb_to_yuv6,
        seq_len=2,
    )

    assert out["segnet_rgb"].shape == (1, 3, 4, 6)
    assert out["posenet_yuv6"].shape == (1, 12, 2, 3)
    assert torch.allclose(out["segnet_rgb"], torch.full((1, 3, 4, 6), 30.0))
    assert torch.allclose(out["posenet_yuv6"][:, :6], torch.full((1, 6, 2, 3), 10.0))
    assert torch.allclose(out["posenet_yuv6"][:, 6:], torch.full((1, 6, 2, 3), 30.0))


def test_main_writes_false_authority_json_with_stubbed_decode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = _load_tool()
    video = tmp_path / "0.mkv"
    video.write_bytes(b"fake-video")

    def fake_decode(**_kwargs):
        return torch.full((2, 2, 8, 10, 3), 64, dtype=torch.uint8)

    fake_frame_utils = types.ModuleType("frame_utils")
    fake_frame_utils.camera_size = (10, 8)
    fake_frame_utils.segnet_model_input_size = (10, 8)
    fake_frame_utils.seq_len = 2

    def fake_rgb_to_yuv6(rgb: torch.Tensor) -> torch.Tensor:
        small = rgb[..., ::2, ::2]
        return torch.cat([small, small], dim=1)

    fake_frame_utils.rgb_to_yuv6 = fake_rgb_to_yuv6
    monkeypatch.setattr(tool, "_decode_real_pairs", fake_decode)
    monkeypatch.setitem(sys.modules, "frame_utils", fake_frame_utils)

    out_json = tmp_path / "mirror.json"
    rc = tool.main(
        [
            "--video",
            video.as_posix(),
            "--output-json",
            out_json.as_posix(),
            "--num-pairs",
            "2",
            "--internal-width",
            "10",
            "--internal-height",
            "8",
            "--repo-root",
            REPO.as_posix(),
        ]
    )

    payload = json.loads(out_json.read_text())
    assert rc == 0
    assert payload["schema"] == tool.SR_NERV_RESOLUTION_AXIS_MIRROR_SCHEMA
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["mirror_gate"]["pass"] is True
