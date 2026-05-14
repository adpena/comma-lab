# SPDX-License-Identifier: MIT
"""Focused train/inflate plumbing tests for SegMap learnable class targets."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from tac.learnable_class_targets import LearnableClassTargets
from tac.mask_grayscale_lut import (
    NUM_CLASSES,
    encode_masks_grayscale,
    grayscale_to_probability_map,
)


def _load_module(rel_path: str, module_name: str):
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _toy_masks() -> torch.Tensor:
    return torch.tensor(
        [
            [[0, 1], [2, 3]],
            [[4, 0], [1, 2]],
        ],
        dtype=torch.int64,
    )


def test_train_segmap_lct_flag_defaults_off(monkeypatch, tmp_path) -> None:
    module = _load_module("experiments/train_segmap.py", "train_segmap_lct_args")
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_segmap.py", "--output-dir", str(tmp_path)],
    )

    args = module._parse_args()

    assert args.learnable_class_targets is False
    assert args.class_targets_filename == "class_targets.fp16"


def test_train_segmap_lct_flag_is_explicit_opt_in(monkeypatch, tmp_path) -> None:
    module = _load_module("experiments/train_segmap.py", "train_segmap_lct_args_on")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_segmap.py",
            "--output-dir",
            str(tmp_path),
            "--learnable-class-targets",
        ],
    )

    args = module._parse_args()

    assert args.learnable_class_targets is True


def test_build_pair_tensors_default_preserves_fixed_soft_lut() -> None:
    module = _load_module("experiments/train_segmap.py", "train_segmap_lct_pairs_off")
    mask_classes = _toy_masks()
    gt_frames = torch.zeros(2, 3, 4, 4)

    mask_pairs, gt_pairs = module._build_pair_tensors(mask_classes, gt_frames)

    expected = grayscale_to_probability_map(
        encode_masks_grayscale(mask_classes), sigma=15.0, channel_first=True
    ).view(1, 2, NUM_CLASSES, 2, 2)
    assert mask_pairs.shape == (1, 2, NUM_CLASSES, 2, 2)
    assert gt_pairs.shape == (1, 2, 3, 4, 4)
    assert torch.allclose(mask_pairs, expected, atol=1e-7)


def test_build_pair_tensors_lct_returns_grayscale_pairs() -> None:
    module = _load_module("experiments/train_segmap.py", "train_segmap_lct_pairs_on")
    mask_classes = _toy_masks()
    gt_frames = torch.zeros(2, 3, 4, 4)

    mask_pairs, gt_pairs = module._build_pair_tensors(
        mask_classes,
        gt_frames,
        learnable_class_targets=True,
    )

    expected = encode_masks_grayscale(mask_classes).view(1, 2, 2, 2)
    assert mask_pairs.dtype == torch.uint8
    assert mask_pairs.shape == (1, 2, 2, 2)
    assert gt_pairs.shape == (1, 2, 3, 4, 4)
    assert torch.equal(mask_pairs, expected)


def test_train_writes_lct_payload_and_metadata(tmp_path) -> None:
    module = _load_module("experiments/train_segmap.py", "train_segmap_lct_payload")
    targets = LearnableClassTargets(
        torch.tensor([0.0, 255.0, 70.25, 185.5, 130.75])
    )

    meta = module._write_class_targets_payload(
        tmp_path,
        "nested/class_targets.fp16",
        targets,
    )

    payload_path = tmp_path / meta["payload_filename"]
    metadata_path = tmp_path / meta["metadata_filename"]
    assert payload_path.read_bytes() == targets.serialize_to_bytes()
    assert payload_path.stat().st_size == 10
    assert meta["payload_bytes"] == 10
    assert meta["format"] == "fp16_class_targets_v1"
    restored = LearnableClassTargets.deserialize_from_bytes(payload_path.read_bytes())
    assert torch.allclose(restored(), targets().to(torch.float16).to(torch.float32))
    assert json.loads(metadata_path.read_text())["payload_sha256"] == meta["payload_sha256"]


def test_train_rejects_unsafe_lct_payload_filename(tmp_path) -> None:
    module = _load_module("experiments/train_segmap.py", "train_segmap_lct_bad_name")
    with pytest.raises(ValueError, match="relative archive member"):
        module._write_class_targets_payload(
            tmp_path,
            "../class_targets.fp16",
            LearnableClassTargets(),
        )


def test_inflate_soft_lut_accepts_custom_class_targets() -> None:
    module = _load_module(
        "submissions/robust_current/inflate_segmap.py",
        "inflate_segmap_lct_features",
    )
    gray = torch.tensor([[[0, 70], [130, 255]]], dtype=torch.uint8)
    targets = torch.tensor([0.0, 255.0, 70.0, 185.0, 130.0])

    out = module._grayscale_to_mask_features(
        gray,
        device=torch.device("cpu"),
        mode="soft_lut",
        class_targets=targets,
    )

    expected = grayscale_to_probability_map(
        gray,
        sigma=15.0,
        targets=targets,
        channel_first=True,
    )
    assert torch.allclose(out, expected, atol=1e-7)


def test_inflate_loads_lct_payload(tmp_path) -> None:
    module = _load_module(
        "submissions/robust_current/inflate_segmap.py",
        "inflate_segmap_lct_payload",
    )
    source = LearnableClassTargets(
        torch.tensor([0.0, 255.0, 70.25, 185.5, 130.75])
    )
    payload_path = tmp_path / "class_targets.fp16"
    payload_path.write_bytes(source.serialize_to_bytes())

    targets = module._load_class_targets_payload(payload_path)

    assert torch.allclose(targets, source().to(torch.float16).to(torch.float32))


def test_inflate_rejects_unsafe_lct_payload_filename() -> None:
    module = _load_module(
        "submissions/robust_current/inflate_segmap.py",
        "inflate_segmap_lct_bad_member",
    )

    with pytest.raises(RuntimeError, match="relative archive member"):
        module._resolve_archive_member(
            Path("archive"),
            "../class_targets.fp16",
            "class_targets_filename",
        )


def test_inflate_rejects_custom_targets_with_hard_onehot() -> None:
    module = _load_module(
        "submissions/robust_current/inflate_segmap.py",
        "inflate_segmap_lct_hard_reject",
    )
    gray = torch.zeros(1, 2, 2, dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="custom class targets require"):
        module._grayscale_to_mask_features(
            gray,
            device=torch.device("cpu"),
            mode="hard_onehot",
            class_targets=torch.tensor([0.0, 255.0, 70.0, 185.0, 130.0]),
        )


def test_inflate_raw_output_path_matches_auth_eval_contract(tmp_path: Path) -> None:
    module = _load_module(
        "submissions/robust_current/inflate_segmap.py",
        "inflate_segmap_lct_raw_path",
    )

    assert module._raw_output_path(tmp_path, "0.mkv") == tmp_path / "0.raw"
    assert module._raw_output_path(tmp_path, "nested/0.mkv") == tmp_path / "nested" / "0.raw"
    with pytest.raises(ValueError, match="unsafe"):
        module._raw_output_path(tmp_path, "../0.mkv")


def test_film_canvas_raw_output_path_matches_auth_eval_contract(tmp_path: Path) -> None:
    module = _load_module(
        "submissions/robust_current/inflate_segmap_film_canvas.py",
        "inflate_segmap_film_canvas_raw_path",
    )

    assert module._raw_output_path(tmp_path, "0.mkv") == tmp_path / "0.raw"
    assert module._raw_output_path(tmp_path, "nested/0.mkv") == tmp_path / "nested" / "0.raw"
    with pytest.raises(ValueError, match="unsafe"):
        module._raw_output_path(tmp_path, "../0.mkv")
