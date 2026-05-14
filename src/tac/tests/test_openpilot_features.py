# SPDX-License-Identifier: MIT
"""Lane DI tests: openpilot supercombo feature extraction and distillation."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

# Lane DI / scene-embedding distillation is in-flight (#145 source recovery).
# tac.scene_embedding_distiller is pending — skip until subagent lands.
try:
    from tac.openpilot_features import (
        SupercomboFeatureUnavailable,
        extract_supercombo_features,
    )
    from tac.scene_embedding_distiller import SceneEmbeddingDistiller, train_scene_embedding_distiller
except ImportError:
    pytest.skip("Lane DI scene-embedding distiller pending", allow_module_level=True)


class _FakeSupercombo:
    def __init__(self) -> None:
        self.calls = 0

    def get_inputs(self):
        class _In:
            name = "input_imgs"
            shape = (1, 12, 128, 256)
        return [_In()]

    def get_outputs(self):
        class _Out:
            name = "features_buffer"
            shape = (1, 512)
        return [_Out()]

    def run(self, output_names, feed):
        self.calls += 1
        x = torch.from_numpy(feed["input_imgs"]).float()
        val = float(x.mean().item() + self.calls)
        return [torch.full((1, 512), val, dtype=torch.float32).numpy()]


def _video(n: int = 6) -> torch.Tensor:
    return torch.randint(0, 255, (n, 32, 48, 3), dtype=torch.uint8)


def test_forward_shape_with_injected_session() -> None:
    feats = extract_supercombo_features(_video(), supercombo_path=None, session=_FakeSupercombo())
    assert feats.shape == (3, 512)
    assert torch.isfinite(feats).all()


def test_gradient_flow_in_distiller() -> None:
    model = SceneEmbeddingDistiller(input_dim=512, output_dim=32)
    x = torch.randn(4, 512, requires_grad=True)
    out = model(x)
    loss = out.square().mean()
    loss.backward()
    assert x.grad is not None
    assert model.net[0].weight.grad is not None


def test_edge_cases_reject_bad_video_and_empty_training() -> None:
    with pytest.raises(ValueError, match=r"\(N, H, W, 3\)"):
        extract_supercombo_features(torch.zeros(4, 3, 32, 48), session=_FakeSupercombo())
    with pytest.raises(ValueError, match="at least"):
        train_scene_embedding_distiller(torch.zeros(1, 512), torch.zeros(1, 6), torch.zeros(1, 5))


def test_determinism_same_seed_same_distiller_output() -> None:
    torch.manual_seed(5)
    a = SceneEmbeddingDistiller(input_dim=512, output_dim=32)
    torch.manual_seed(5)
    b = SceneEmbeddingDistiller(input_dim=512, output_dim=32)
    x = torch.randn(2, 512)
    assert torch.allclose(a(x), b(x))


def test_cuda_only_enforcement_raises_on_cpu() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        extract_supercombo_features(_video(), session=_FakeSupercombo(), require_cuda=True)


def test_missing_supercombo_fails_closed_unless_zero_fallback_explicit(tmp_path: Path) -> None:
    missing = tmp_path / "missing_supercombo.onnx"
    with pytest.raises(SupercomboFeatureUnavailable):
        extract_supercombo_features(_video(), supercombo_path=missing)

    with pytest.warns(RuntimeWarning, match="zero embedding"):
        feats = extract_supercombo_features(
            _video(n=4),
            supercombo_path=missing,
            feature_dim=7,
            allow_zero_fallback=True,
        )
    assert feats.shape == (2, 7)
    assert torch.count_nonzero(feats).item() == 0


def test_distiller_save_load_roundtrip(tmp_path: Path) -> None:
    model = SceneEmbeddingDistiller(input_dim=512, output_dim=32)
    path = tmp_path / "distiller.pt"
    model.save(path)
    loaded = SceneEmbeddingDistiller.load(path)
    x = torch.randn(2, 512)
    assert torch.allclose(model(x), loaded(x))
