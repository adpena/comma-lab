"""Regression tests for qat_finetune.py loader fixes.

Mirrors pipeline.step_export's parametrize-strip + build_renderer dispatch
fix from c5214993. The LANE-D crash (2026-04-26) was the same bug class:

    RuntimeError: Error(s) in loading state_dict for AsymmetricPairGenerator:
      Missing key(s): "renderer.embedding.weight" ...
      Unexpected key(s): "renderer.embedding.parametrizations.weight.original" ...
      size mismatch for motion.head.bias: copying a param with shape
      torch.Size([2]) from checkpoint, the shape in current model is torch.Size([6])

Two layers of bug:
  1. parametrize-hooked checkpoint keys don't match plain weight keys
  2. hardcoded AsymmetricPairGenerator constructor ignored use_zoom_flow,
     loading PairGenerator state into AsymmetricPairGenerator architecture
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

REPO = Path(__file__).resolve().parents[3]
for sub in ("src", "upstream", "."):
    p = str(REPO / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.qat_finetune import QATConfig, create_model, load_float_checkpoint  # noqa: E402


@pytest.fixture
def tmp_ckpt():
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    yield Path(path)
    if Path(path).exists():
        os.remove(path)


def _save(model: nn.Module, path: Path) -> None:
    torch.save({"model_state_dict": model.state_dict()}, path)


def test_use_zoom_flow_false_returns_pair_generator():
    """create_model(use_zoom_flow=False) must return PairGenerator,
    NOT AsymmetricPairGenerator. The two have different motion.head shapes."""
    from tac.renderer import PairGenerator
    cfg = QATConfig(use_zoom_flow=False, pose_dim=0, padding_mode="zeros")
    m = create_model(cfg, torch.device("cpu"))
    assert isinstance(m, PairGenerator), f"Expected PairGenerator, got {type(m).__name__}"


def test_use_zoom_flow_true_returns_asymmetric():
    """create_model(use_zoom_flow=True) must return AsymmetricPairGenerator."""
    from tac.renderer import AsymmetricPairGenerator
    cfg = QATConfig(use_zoom_flow=True, pose_dim=6, padding_mode="zeros")
    m = create_model(cfg, torch.device("cpu"))
    assert isinstance(m, AsymmetricPairGenerator), f"Got {type(m).__name__}"


def test_load_pt_checkpoint_with_parametrize_hooks(tmp_ckpt):
    """A .pt checkpoint saved with nn.utils.parametrize hooks attached
    must load cleanly into a fresh (no-hooks) model. This was the LANE-D
    'Unexpected key(s): renderer.embedding.parametrizations.weight.original'
    crash."""
    cfg = QATConfig(use_zoom_flow=False, pose_dim=0, padding_mode="zeros")
    src = create_model(cfg, torch.device("cpu"))

    # Attach parametrize hooks to ALL conv layers (mimicking self-compression
    # / FakeQuant during training).
    class Identity(nn.Module):
        def forward(self, w: torch.Tensor) -> torch.Tensor:
            return w

    n_wrapped = 0
    for _, mod in src.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
            P.register_parametrization(mod, "weight", Identity())
            n_wrapped += 1
    assert n_wrapped > 0, "test setup wrong: no conv layers found"

    # Verify the saved state_dict has the parametrize-renamed keys.
    saved_keys = list(src.state_dict().keys())
    assert any(".parametrizations." in k for k in saved_keys), (
        "test setup wrong: parametrize hooks didn't rename any keys"
    )

    _save(src, tmp_ckpt)

    # Load into a FRESH model (no hooks) — must not raise.
    dst = create_model(cfg, torch.device("cpu"))
    load_float_checkpoint(dst, str(tmp_ckpt), torch.device("cpu"))


def test_load_pt_checkpoint_arch_mismatch_raises(tmp_ckpt):
    """If a checkpoint was trained with use_zoom_flow=False but the loader
    cfg says use_zoom_flow=True (or vice versa), we must raise loudly with
    the actual missing/unexpected keys — not silently load garbage."""
    # Train with use_zoom_flow=False (PairGenerator)
    cfg_train = QATConfig(use_zoom_flow=False, pose_dim=0, padding_mode="zeros")
    src = create_model(cfg_train, torch.device("cpu"))
    _save(src, tmp_ckpt)

    # Try to load into use_zoom_flow=True (AsymmetricPairGenerator)
    cfg_load = QATConfig(use_zoom_flow=True, pose_dim=6, padding_mode="zeros")
    dst = create_model(cfg_load, torch.device("cpu"))

    # PyTorch raises size-mismatch errors during load_state_dict before
    # our strict=False handler can format a custom message. Either error
    # is acceptable as long as the load LOUDLY fails (not silently
    # corrupts the model). Both contain the channel counts so the user
    # can debug arch drift.
    with pytest.raises(RuntimeError, match=r"size mismatch|shape mismatch|missing|unexpected"):
        load_float_checkpoint(dst, str(tmp_ckpt), torch.device("cpu"))


def test_load_plain_pt_checkpoint_no_hooks(tmp_ckpt):
    """A plain .pt checkpoint (no parametrize hooks) should load
    unchanged — the strip path is a no-op."""
    cfg = QATConfig(use_zoom_flow=False, pose_dim=0, padding_mode="zeros")
    src = create_model(cfg, torch.device("cpu"))
    _save(src, tmp_ckpt)

    dst = create_model(cfg, torch.device("cpu"))
    load_float_checkpoint(dst, str(tmp_ckpt), torch.device("cpu"))

    # Verify weights are bit-identical
    for (k1, v1), (k2, v2) in zip(src.state_dict().items(), dst.state_dict().items()):
        assert k1 == k2
        assert torch.equal(v1, v2), f"weight mismatch at {k1}"


def test_parametrize_strip_drops_codebook_buffers(tmp_ckpt):
    """Parametrizations with internal buffers (e.g. FP4 codebook) save
    extra keys like `<layer>.parametrizations.weight.0.codebook`. Those
    must be DROPPED, not loaded as-is — they're QAT internals, not part
    of the plain weight tensor."""
    cfg = QATConfig(use_zoom_flow=False, pose_dim=0, padding_mode="zeros")
    src = create_model(cfg, torch.device("cpu"))

    class WithBuffer(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("codebook", torch.zeros(16))

        def forward(self, w: torch.Tensor) -> torch.Tensor:
            return w

    # Wrap a single conv to keep the test small
    for _, mod in src.named_modules():
        if isinstance(mod, nn.Conv2d):
            P.register_parametrization(mod, "weight", WithBuffer())
            break

    saved = src.state_dict()
    has_codebook = any("codebook" in k for k in saved)
    assert has_codebook, "test setup wrong: codebook buffer not in state_dict"

    _save(src, tmp_ckpt)

    # Must load without raising (codebook keys dropped, not 'unexpected')
    dst = create_model(cfg, torch.device("cpu"))
    load_float_checkpoint(dst, str(tmp_ckpt), torch.device("cpu"))
