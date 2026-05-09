from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_train_module():
    path = REPO_ROOT / "experiments/train_score_gradient_pr101_finetune.py"
    name = "train_score_gradient_pr101_finetune_real_data_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FakeFrame:
    def __init__(self, value: int):
        self.value = value

    def to_ndarray(self, *, format: str):
        raise AssertionError(
            f"load_real_frame_pairs must use upstream yuv420_to_rgb, not {format}"
        )


class _FakeContainer:
    def __init__(self, frames: list[_FakeFrame]):
        self._frames = frames
        self.streams = types.SimpleNamespace(video=[object()])
        self.closed = False

    def decode(self, _stream):
        yield from self._frames

    def close(self):
        self.closed = True


def test_load_real_frame_pairs_uses_upstream_yuv420_converter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_train_module()
    frames = [_FakeFrame(10), _FakeFrame(20), _FakeFrame(30)]
    container = _FakeContainer(frames)
    monkeypatch.setitem(sys.modules, "av", types.SimpleNamespace(open=lambda _path: container))
    calls: list[int] = []

    def fake_yuv420_to_rgb(frame: _FakeFrame) -> torch.Tensor:
        calls.append(frame.value)
        return torch.full((4, 6, 3), frame.value, dtype=torch.uint8)

    monkeypatch.setattr(mod, "_load_upstream_yuv420_to_rgb", lambda: fake_yuv420_to_rgb)
    video = tmp_path / "0.mkv"
    video.write_bytes(b"fake-video")

    pairs = mod.load_real_frame_pairs(video, frame_h=2, frame_w=3)

    assert calls == [10, 20, 30]
    assert container.closed is True
    assert pairs.shape == (1, 2, 2, 3, 3)
    assert torch.all(pairs[0, 0] == 10)
    assert torch.all(pairs[0, 1] == 20)


def test_load_real_frame_pairs_honors_max_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_train_module()
    frames = [_FakeFrame(1), _FakeFrame(2), _FakeFrame(3)]
    monkeypatch.setitem(
        sys.modules,
        "av",
        types.SimpleNamespace(open=lambda _path: _FakeContainer(frames)),
    )
    seen: list[int] = []

    def fake_yuv420_to_rgb(frame: _FakeFrame) -> torch.Tensor:
        seen.append(frame.value)
        return torch.full((2, 2, 3), frame.value, dtype=torch.uint8)

    monkeypatch.setattr(mod, "_load_upstream_yuv420_to_rgb", lambda: fake_yuv420_to_rgb)
    video = tmp_path / "0.mkv"
    video.write_bytes(b"fake-video")

    pairs = mod.load_real_frame_pairs(video, frame_h=2, frame_w=2, max_frames=2)

    assert seen == [1, 2]
    assert pairs.shape == (1, 2, 2, 2, 3)


def test_train_requires_explicit_batch_source() -> None:
    mod = _load_train_module()

    with pytest.raises(ValueError, match="explicit batch_source"):
        mod.train(
            decoder=object(),
            posenet=object(),
            segnet=object(),
            epochs=1,
            steps_per_epoch=1,
            batch_size=1,
            latent_dim=1,
            frame_h=1,
            frame_w=1,
            lr=1e-4,
            device=torch.device("cpu"),
            output_dir=Path("."),
            aux_kl_weight=1.0,
            aux_pixel_l1_weight=0.0,
            batch_source=None,
        )


def test_real_pair_batch_source_uses_archive_latent_rows() -> None:
    mod = _load_train_module()
    frame_pairs = torch.stack(
        [
            torch.full((2, 2, 2, 3), float(idx), dtype=torch.float32)
            for idx in range(6)
        ],
        dim=0,
    )
    latents = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
    source = mod.RealPairBatchSource(
        frame_pairs=frame_pairs,
        latents=latents,
        device=torch.device("cpu"),
        seed=123,
    )

    z, gt = source.next_batch(5)

    assert z.shape == (5, 2)
    for row in range(z.shape[0]):
        idx = int(z[row, 0].item() // 2)
        assert torch.equal(z[row], latents[idx])
        assert torch.all(gt[row] == float(idx))


def test_real_pair_batch_source_rejects_missing_latent_rows() -> None:
    mod = _load_train_module()
    frame_pairs = torch.zeros((4, 2, 2, 2, 3), dtype=torch.float32)
    latents = torch.zeros((3, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="latents row count"):
        mod.RealPairBatchSource(
            frame_pairs=frame_pairs,
            latents=latents,
            device=torch.device("cpu"),
            seed=1,
        )
