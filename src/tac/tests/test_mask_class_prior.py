# SPDX-License-Identifier: MIT
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.build_mask_class_prior import (
    build_mask_class_prior,
    write_prior_npz,
)
from tac.mask_prior import (
    apply_prior_weighting,
    load_prior,
    save_prior_to_archive,
)


def _toy_masks() -> torch.Tensor:
    masks = torch.zeros((6, 8, 10), dtype=torch.long)
    masks[:, :, 5:] = 1
    masks[2:, 2:6, 2:8] = 2
    masks[4:, :2, :] = 3
    masks[5:, 6:, :] = 4
    return masks


def test_build_mask_class_prior_produces_valid_npz(tmp_path: Path) -> None:
    prior = build_mask_class_prior(_toy_masks(), prior_resolution=(5, 4), sigma=1.0)
    out = tmp_path / "mask_class_prior.npz"

    write_prior_npz(prior, out)
    loaded = np.load(out)

    assert loaded["version"].item() == 1
    assert loaded["prior"].shape == (5, 4, 5)
    assert loaded["prior"].dtype == np.float16


def test_prior_sums_to_one_along_class_dim() -> None:
    prior = build_mask_class_prior(_toy_masks(), prior_resolution=(5, 4), sigma=1.0)

    assert np.allclose(prior.sum(axis=0), 1.0, atol=1e-5)


def test_load_prior_validates_shape(tmp_path: Path) -> None:
    bad = tmp_path / "bad_prior.npz"
    np.savez_compressed(bad, prior=np.ones((4, 4, 5), dtype=np.float16), version=np.array(1))

    with pytest.raises(ValueError, match="shape"):
        load_prior(bad)


def test_apply_prior_weighting_returns_same_shape() -> None:
    prior = build_mask_class_prior(_toy_masks(), prior_resolution=(5, 4), sigma=1.0)
    logits = torch.zeros(2, 5, 8, 10)

    weighted = apply_prior_weighting(logits, prior, alpha=0.1)

    assert weighted.shape == logits.shape


def test_save_prior_to_archive_appends_without_corrupting_zip(tmp_path: Path) -> None:
    prior = build_mask_class_prior(_toy_masks(), prior_resolution=(5, 4), sigma=1.0)
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("renderer.bin", b"renderer")

    save_prior_to_archive(prior, archive)

    with zipfile.ZipFile(archive) as zf:
        assert zf.read("renderer.bin") == b"renderer"
        assert "prior.npz" in zf.namelist()
        with zf.open("prior.npz") as fh:
            loaded = np.load(fh)
            assert loaded["prior"].shape == (5, 4, 5)


def test_roundtrip_save_load_apply_produces_finite_outputs(tmp_path: Path) -> None:
    prior = build_mask_class_prior(_toy_masks(), prior_resolution=(5, 4), sigma=1.0)
    out = tmp_path / "prior.npz"
    write_prior_npz(prior, out)

    loaded = load_prior(out)
    logits = torch.randn(1, 5, 7, 9)
    weighted = apply_prior_weighting(logits, loaded, alpha=0.2)

    assert torch.isfinite(weighted).all()
