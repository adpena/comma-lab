# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch


def test_decode_video_stops_at_max_frames(monkeypatch) -> None:
    from tac import data

    class FakeContainer:
        def __init__(self) -> None:
            self.streams = SimpleNamespace(video=[object()])
            self.decoded = 0
            self.closed = False

        def decode(self, stream):  # noqa: ANN001
            assert stream is self.streams.video[0]
            for i in range(10):
                self.decoded += 1
                yield SimpleNamespace(index=i)

        def close(self) -> None:
            self.closed = True

    container = FakeContainer()
    monkeypatch.setattr(data.av, "open", lambda path: container)
    monkeypatch.setattr(
        data,
        "yuv420_to_rgb",
        lambda frame: torch.zeros((2, 2, 3), dtype=torch.uint8),
    )

    frames = data.decode_video(Path("unused.mkv"), target_h=2, target_w=2, max_frames=3)

    assert len(frames) == 3
    assert container.decoded == 3
    assert container.closed


def test_load_gt_video_passes_n_frames_to_decoder(monkeypatch) -> None:
    from tac import data

    seen: dict[str, int | None] = {}

    def fake_decode_video(path, *, target_h, target_w, max_frames=None):  # noqa: ANN001
        seen["max_frames"] = max_frames
        return [torch.zeros((target_h, target_w, 3), dtype=torch.uint8) for _ in range(5)]

    monkeypatch.setattr(data, "decode_video", fake_decode_video)
    frames = data.load_gt_video(Path("unused.mkv"), n_frames=4)

    assert seen["max_frames"] == 4
    assert len(frames) == 4
