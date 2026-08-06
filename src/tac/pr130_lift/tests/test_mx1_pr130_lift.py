from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from tac.pr130_lift.mlx_semantic_renderer import (
    MlxSemanticConfig,
    curriculum_loss_mlx,
    mlx_device_probe,
)


LIFTED = Path(__file__).resolve().parents[1] / "lifted"


def _load_lifted_semantic_module():
    sys.path.insert(0, str(LIFTED))
    spec = importlib.util.spec_from_file_location(
        "mx1_lifted_semantic_renderer_oracle",
        LIFTED / "semantic_renderer_oracle.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lifted_torch_forward_and_curriculum_are_real_behavior() -> None:
    lifted = _load_lifted_semantic_module()
    torch.manual_seed(7)
    model = lifted.SemanticTokenRenderer(
        width=8, blocks=2, frame_dim=4, num_pairs=6, num_tokens=5
    )
    tokens = torch.randint(0, 5, (2, 12, 16), dtype=torch.long)
    pair_idx = torch.tensor([0, 3], dtype=torch.long)
    out = model(tokens, pair_idx)
    assert tuple(out.shape) == (2, 3, 12, 16)
    assert float(out.detach().min()) >= 0.0
    assert float(out.detach().max()) <= 255.0
    logits = torch.randn(2, 5, 12, 16)
    loss, phase = lifted.curriculum_loss(
        logits, tokens, step=0, total_steps=20, ce_fraction=0.5, softplus_fraction=0.8
    )
    assert phase == "ce"
    assert loss.requires_grad is False
    assert float(loss) > 0.0


def test_torch_reference_deterministic_same_seed() -> None:
    lifted = _load_lifted_semantic_module()
    tokens = torch.randint(0, 5, (2, 8, 8), dtype=torch.long)
    pair_idx = torch.tensor([1, 5], dtype=torch.long)
    torch.manual_seed(11)
    a = lifted.SemanticTokenRenderer(width=8, blocks=2, frame_dim=4, num_pairs=8)
    out_a = a(tokens, pair_idx)
    torch.manual_seed(11)
    b = lifted.SemanticTokenRenderer(width=8, blocks=2, frame_dim=4, num_pairs=8)
    out_b = b(tokens, pair_idx)
    assert torch.equal(out_a, out_b)


def test_mlx_port_imports_without_eager_mlx_runtime() -> None:
    cfg = MlxSemanticConfig(width=96, blocks=4)
    assert cfg.width == 96
    assert cfg.blocks == 4
    assert callable(curriculum_loss_mlx)


def test_mlx_device_probe_is_fail_closed_or_available() -> None:
    probe = mlx_device_probe(device="cpu")
    assert probe["status"] in {"available", "blocked"}
    if probe["status"] == "blocked":
        assert probe["error"]
