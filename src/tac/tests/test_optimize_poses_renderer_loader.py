# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


def test_optimize_poses_loader_dispatches_qzs3_without_torch_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Packed contest renderer formats must use the inflate runtime loader.

    The C-063 CRF52 stale-pose isolation run failed before evidence because
    ``experiments.optimize_poses.load_renderer`` knew ASYM/FP4A/OWV3 only;
    QZS3 fell through to ``torch.load`` and crashed on non-pickle bytes.
    """
    from experiments import optimize_poses
    from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
    from tac.quantizr_qzs3_codec import encode_qzs3_state_dict

    renderer_path = tmp_path / "renderer.bin"
    renderer_path.write_bytes(
        encode_qzs3_state_dict(build_quantizr_faithful_renderer().eval())
    )

    def _forbid_torch_load(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("QZS3 renderer path fell through to torch.load")

    monkeypatch.setattr(optimize_poses.torch, "load", _forbid_torch_load)
    loaded = optimize_poses.load_renderer(str(renderer_path), torch.device("cpu"))

    assert getattr(loaded, "q_faithful", False)
    assert getattr(loaded, "pose_dim", None) == 6
    assert not loaded.training
    assert all(not p.requires_grad for p in loaded.parameters())


def test_optimize_poses_loader_rejects_unknown_non_pickle_before_torch_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments import optimize_poses

    renderer_path = tmp_path / "renderer.bin"
    renderer_path.write_bytes(b"QZSX" + b"\0" * 64)

    def _forbid_torch_load(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("unknown binary renderer path fell through to torch.load")

    monkeypatch.setattr(optimize_poses.torch, "load", _forbid_torch_load)
    with pytest.raises(RuntimeError, match="unknown binary bytes"):
        optimize_poses.load_renderer(str(renderer_path), torch.device("cpu"))


def test_preflight_accepts_qzs3_and_rejects_unknown_non_pickle(tmp_path: Path) -> None:
    from tac.preflight import PreflightError, preflight_check
    from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
    from tac.quantizr_qzs3_codec import encode_qzs3_state_dict

    qzs3_path = tmp_path / "renderer_qzs3.bin"
    qzs3_path.write_bytes(
        encode_qzs3_state_dict(build_quantizr_faithful_renderer().eval())
    )
    preflight_check(renderer_path=qzs3_path, verbose=False)

    unknown_path = tmp_path / "renderer_unknown.bin"
    unknown_path.write_bytes(b"QZSX" + b"\0" * 64)
    with pytest.raises(PreflightError, match="unknown non-pickle binary format"):
        preflight_check(renderer_path=unknown_path, verbose=False)


def test_optimize_poses_parse_args_accepts_skip_proxy_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments import optimize_poses

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "optimize_poses.py",
            "--checkpoint",
            "renderer.bin",
            "--masks",
            "masks.pt",
            "--skip-proxy-score",
        ],
    )
    args = optimize_poses.parse_args()
    assert args.skip_proxy_score is True


def test_qzs3_cuda_pose_batch_safety_caps_index_math_risk(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from experiments import optimize_poses

    args = SimpleNamespace(device="cuda", batch_pairs=100)
    renderer = SimpleNamespace(q_faithful=True)

    capped = optimize_poses.apply_renderer_cuda_batch_safety(args, renderer)

    assert capped == optimize_poses.QZS3_CUDA_MAX_BATCH_PAIRS
    assert args.batch_pairs == optimize_poses.QZS3_CUDA_MAX_BATCH_PAIRS
    assert "capping --batch-pairs 100" in capsys.readouterr().out
