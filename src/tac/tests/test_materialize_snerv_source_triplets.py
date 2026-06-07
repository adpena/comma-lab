# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import tools.materialize_snerv_source_triplets as tool


class _FakeFrame:
    def __init__(self, value: int) -> None:
        self.value = value


class _FakeContainer:
    def __init__(self, values: list[int]) -> None:
        self.streams = types.SimpleNamespace(video=[object()])
        self._frames = [_FakeFrame(value) for value in values]
        self.closed = False

    def decode(self, _stream):
        yield from self._frames

    def close(self) -> None:
        self.closed = True


def _install_fake_decode(
    monkeypatch: pytest.MonkeyPatch,
    values: list[int],
) -> _FakeContainer:
    container = _FakeContainer(values)
    monkeypatch.setitem(
        sys.modules,
        "av",
        types.SimpleNamespace(open=lambda _path: container),
    )

    def fake_yuv420_to_rgb(frame: _FakeFrame) -> np.ndarray:
        return np.full((2, 4, 3), frame.value, dtype=np.uint8)

    monkeypatch.setattr(
        tool,
        "load_upstream_yuv420_to_rgb",
        lambda **_kwargs: fake_yuv420_to_rgb,
    )
    monkeypatch.setattr(tool, "git_head_sha", lambda **_kwargs: "a" * 40)
    return container


def test_materialize_pair_zero_current_previous_next_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = _install_fake_decode(monkeypatch, [0, 1, 2])
    source = tmp_path / "0.mkv"
    source.write_bytes(b"fake-video")
    out = tmp_path / "triplets.npy"
    manifest = tmp_path / "triplets.manifest.json"

    payload = tool.materialize_source_frame_triplets(
        video_path=source,
        pair_ids=[0],
        output_npy=out,
        manifest_path=manifest,
        min_free_bytes=0,
        command_argv=["tool", "--pair-ids", "0"],
        generated_utc="20260607T000000Z",
    )

    arr = np.load(out, allow_pickle=False)
    assert container.closed is True
    assert arr.shape == (1, 3, 3, 2, 4)
    assert arr.dtype == np.uint8
    assert np.all(arr[0, 0] == 1)  # current = frame 2p + 1
    assert np.all(arr[0, 1] == 0)  # previous = frame 2p
    assert np.all(arr[0, 2] == 2)  # next = frame 2p + 2
    assert payload["output"]["shape"] == [1, 3, 3, 2, 4]
    assert payload["output"]["geometry"] == {
        "height": 2,
        "width": 4,
        "coordinate_system": "source_rgb_frame_geometry",
        "scorer_resized": False,
    }
    assert payload["output"]["sha256"] == tool.sha256_file(out)
    assert payload["authority_boundary"] == {
        "source_frame_triplets_for_official_snerv_t_forward": True,
        "scorer_cache": False,
        "receiver_output": False,
        "source_forward_authority": False,
        "reason": (
            "Triplets are necessary inputs to the strict official Torch "
            "SNeRV_T.forward witness, but do not by themselves prove "
            "MFU/HFR/TUB/output_2 checkpoint source authority."
        ),
    }
    assert payload["false_authority_flags"]["score_claim"] is False
    assert payload["source_forward_authority"] is False

    persisted = json.loads(manifest.read_text(encoding="utf-8"))
    assert persisted["source"]["sha256"] == tool.sha256_file(source)
    assert persisted["frame_plan"][0]["source_frame_indices"] == {
        "current": 1,
        "next": 2,
        "previous": 0,
    }
    assert persisted["provenance"]["decode_semantics"].startswith("PyAV plus")


def test_materialize_preserves_requested_pair_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_decode(monkeypatch, list(range(7)))
    source = tmp_path / "0.mkv"
    source.write_bytes(b"fake-video")
    out = tmp_path / "triplets.npy"

    tool.materialize_source_frame_triplets(
        video_path=source,
        pair_ids=[2, 0],
        output_npy=out,
        min_free_bytes=0,
        command_argv=["tool"],
    )

    arr = np.load(out, allow_pickle=False)
    assert arr.shape == (2, 3, 3, 2, 4)
    assert np.all(arr[0, 0] == 5)
    assert np.all(arr[0, 1] == 4)
    assert np.all(arr[0, 2] == 6)
    assert np.all(arr[1, 0] == 1)
    assert np.all(arr[1, 1] == 0)
    assert np.all(arr[1, 2] == 2)


def test_materialize_refuses_silent_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_decode(monkeypatch, [0, 1, 2])
    source = tmp_path / "0.mkv"
    source.write_bytes(b"fake-video")
    out = tmp_path / "triplets.npy"
    out.write_bytes(b"existing")

    with pytest.raises(tool.ArtifactWriteError, match="refusing to overwrite"):
        tool.materialize_source_frame_triplets(
            video_path=source,
            pair_ids=[0],
            output_npy=out,
            min_free_bytes=0,
            command_argv=["tool"],
        )


def test_materialize_missing_next_frame_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_decode(monkeypatch, [0, 1])
    source = tmp_path / "0.mkv"
    source.write_bytes(b"fake-video")
    out = tmp_path / "triplets.npy"

    with pytest.raises(RuntimeError, match="ended before all requested"):
        tool.materialize_source_frame_triplets(
            video_path=source,
            pair_ids=[0],
            output_npy=out,
            min_free_bytes=0,
            command_argv=["tool"],
        )
    assert not out.exists()


def test_parse_pair_ids_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        tool.parse_pair_ids("0,-1")
