# SPDX-License-Identifier: MIT
"""Tests for Lane SH arithmetic SegMap inflate parity."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) in sys.path:
        sys.path.remove(str(_path))
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_submissions = sys.modules.get("submissions")
if _submissions is not None:
    expected = (REPO_ROOT / "submissions").resolve()
    package_paths = {
        Path(path).resolve() for path in getattr(_submissions, "__path__", ())
    }
    module_file = getattr(_submissions, "__file__", None)
    file_ok = module_file is not None and expected in Path(module_file).resolve().parents
    if expected not in package_paths and not file_ok:
        for _name in list(sys.modules):
            if _name == "submissions" or _name.startswith("submissions."):
                del sys.modules[_name]

from submissions.robust_current import inflate_segmap as standard
from submissions.robust_current import inflate_segmap_arithmetic as arith
from tac.learnable_class_targets import LearnableClassTargets
from tac.mask_grayscale_lut import grayscale_to_probability_map


def test_arithmetic_default_soft_lut_matches_standard_segmap_path() -> None:
    gray = torch.tensor(
        [
            [
                [0, 255, 64],
                [192, 128, 17],
            ]
        ],
        dtype=torch.uint8,
    )
    device = torch.device("cpu")

    got = arith._grayscale_to_mask_features(gray, device=device, mode="soft_lut")
    expected = standard._grayscale_to_mask_features(gray, device=device, mode="soft_lut")

    assert got.shape == (1, 5, 2, 3)
    assert torch.allclose(got, expected)
    assert torch.allclose(got.sum(dim=1), torch.ones(1, 2, 3))


def test_arithmetic_hard_onehot_mode_preserves_legacy_projection() -> None:
    gray = torch.tensor([[[0, 255, 64, 192, 128]]], dtype=torch.uint8)
    got = arith._grayscale_to_mask_features(
        gray, device=torch.device("cpu"), mode="onehot"
    )

    assert got.shape == (1, 5, 1, 5)
    assert torch.equal(got.argmax(dim=1), torch.tensor([[[0, 1, 2, 3, 4]]]))
    assert torch.all(got.sum(dim=1) == 1.0)


def test_arithmetic_soft_lut_accepts_custom_class_targets() -> None:
    gray = torch.tensor([[[0, 70], [130, 255]]], dtype=torch.uint8)
    targets = torch.tensor([0.0, 255.0, 70.0, 185.0, 130.0])

    got = arith._grayscale_to_mask_features(
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

    assert torch.allclose(got, expected, atol=1e-7)


def test_arithmetic_rejects_custom_targets_with_hard_onehot() -> None:
    gray = torch.zeros(1, 2, 2, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="custom class targets require"):
        arith._grayscale_to_mask_features(
            gray,
            device=torch.device("cpu"),
            mode="hard_onehot",
            class_targets=torch.tensor([0.0, 255.0, 70.0, 185.0, 130.0]),
        )


def test_arithmetic_loads_lct_payload(tmp_path) -> None:
    source = LearnableClassTargets(
        torch.tensor([0.0, 255.0, 70.25, 185.5, 130.75])
    )
    payload_path = tmp_path / "class_targets.fp16"
    payload_path.write_bytes(source.serialize_to_bytes())

    targets = arith._load_class_targets_payload(payload_path)

    assert torch.allclose(targets, source().to(torch.float16).to(torch.float32))


def test_arithmetic_rejects_unsafe_lct_payload_filename() -> None:
    with pytest.raises(RuntimeError, match="relative archive member"):
        arith._resolve_archive_member(
            Path("archive"),
            "../class_targets.fp16",
            "class_targets_filename",
        )


@pytest.mark.parametrize("alias", ["hard", "one_hot", "onehot", "hard_onehot"])
def test_arithmetic_grayscale_mode_aliases(alias: str) -> None:
    assert arith._normalize_grayscale_mode(alias) == "hard_onehot"


def test_arithmetic_raw_output_path_matches_auth_eval_contract(tmp_path) -> None:
    assert arith._raw_output_path(tmp_path, "0.mkv") == tmp_path / "0.raw"
    assert arith._raw_output_path(tmp_path, "nested/0.mkv") == tmp_path / "nested" / "0.raw"
    with pytest.raises(ValueError, match="unsafe"):
        arith._raw_output_path(tmp_path, "../0.mkv")


def test_arithmetic_rejects_unknown_grayscale_mode() -> None:
    with pytest.raises(RuntimeError, match="SEGMAP_GRAYSCALE_MODE"):
        arith._normalize_grayscale_mode("nearest")


def test_arithmetic_inflate_uses_bicubic_resize_for_standard_runtime_parity() -> None:
    source = inspect.getsource(arith.inflate)
    assert 'mode="bicubic"' in source
    assert 'mode="bilinear"' not in source


def test_arithmetic_build_segmap_honors_homography_env(monkeypatch) -> None:
    from tac.segmap_renderer import SegMapHomography

    source = SegMapHomography(hidden=4, block_hidden=4, num_blocks=1, max_frame_index=2)
    monkeypatch.setenv("SEGMAP_ARCH", "segmap_homography")
    model = arith._build_segmap(
        source.state_dict(),
        hidden=4,
        block_hidden=4,
        num_blocks=1,
        max_frame_index=2,
    )

    assert model.__class__.__name__ == "SegMapHomography"
