# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import tools.build_comma_video_substrate_eval_600pairs_dataset as builder


def test_resolve_public_video_path_accepts_stem_and_mkv_suffix(tmp_path: Path) -> None:
    assert builder.resolve_public_video_path("0", tmp_path) == tmp_path / "0.mkv"
    assert builder.resolve_public_video_path("0.mkv", tmp_path) == tmp_path / "0.mkv"


def test_resolve_public_video_path_refuses_unknown_suffix(tmp_path: Path) -> None:
    try:
        builder.resolve_public_video_path("0.mp4", tmp_path)
    except ValueError as exc:
        assert "bare stems or `.mkv`" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for non-mkv public video name")


def test_resolve_device_labels_macos_cpu_as_advisory(monkeypatch) -> None:
    monkeypatch.setattr(builder.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(builder.platform, "system", lambda: "Darwin")

    device, axis = builder.resolve_device()

    assert str(device) == "cpu"
    assert axis == "[macOS-CPU advisory]"


def test_dataset_card_calls_segnet_surface_logits_not_softmax() -> None:
    provenance = builder.BuildProvenance(
        upstream_modules_py_sha256="modules-sha",
        segnet_safetensors_sha256="seg-sha",
        posenet_safetensors_sha256="pose-sha",
        device="cpu",
        torch_version="test",
        n_pairs=1,
        seg_label_axis="[macOS-CPU advisory]",
        pose_label_axis="[macOS-CPU advisory]",
    )

    card = builder._build_dataset_card(provenance, 1)

    assert "full logits" in card
    assert "full softmax logits" not in card
    assert "[macOS-CPU advisory]" in card
