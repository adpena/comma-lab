# SPDX-License-Identifier: MIT
"""Lane SI-V3 tests: UNIWARD texture-probability masks."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tac.saliency_inversion import apply_saliency_weighted_compression, unpack_saliency_payload
from tac.uniward_texture import compute_texture_probability


class _TinyScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).mean(dim=(1, 2, 3))


def test_forward_shape_and_finite_values() -> None:
    frames = torch.rand(3, 3, 16, 24)
    sigma2 = compute_texture_probability(frames, [_TinyScorer()])
    assert sigma2.shape == (16, 24)
    assert torch.isfinite(sigma2).all()
    assert (sigma2 >= 0).all()


def test_gradient_flow_when_not_detached() -> None:
    frames = torch.rand(2, 3, 8, 8, requires_grad=True)
    sigma2 = compute_texture_probability(frames, [_TinyScorer()], detach=False)
    sigma2.mean().backward()
    assert frames.grad is not None


def test_edge_cases_zero_and_max_inputs_are_finite() -> None:
    scorer = _TinyScorer()
    assert torch.isfinite(compute_texture_probability(torch.zeros(1, 3, 8, 8), [scorer])).all()
    assert torch.isfinite(compute_texture_probability(torch.ones(1, 3, 8, 8) * 255, [scorer])).all()


def test_determinism_same_seed_same_probability() -> None:
    torch.manual_seed(11)
    scorer = _TinyScorer()
    frames = torch.rand(2, 3, 12, 12)
    a = compute_texture_probability(frames, [scorer])
    torch.manual_seed(11)
    b = compute_texture_probability(frames, [scorer])
    assert torch.allclose(a, b)


def test_cuda_only_enforcement_raises_on_cpu() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        compute_texture_probability(torch.rand(1, 3, 8, 8), [_TinyScorer()], require_cuda=True)


def test_saliency_inversion_uniward_texture_mode() -> None:
    masks = torch.randint(0, 5, (2, 8, 8), dtype=torch.uint8)
    texture = torch.zeros(8, 8)
    texture[:, 4:] = 10.0
    payload = apply_saliency_weighted_compression(
        masks,
        mode="uniward_texture",
        texture_probability=texture,
        texture_quantile=0.5,
    )
    parsed = unpack_saliency_payload(payload)
    assert parsed["region_mask"].shape == (8, 8)
    assert parsed["region_mask"][:, 4:].float().mean() > 0.9


# --- Coverage for the patched apply_saliency_weighted_compression helpers ---
# Wave 3 audit (2026-04-28) found these 3 helpers had F821 names that were
# late-bound through __code__ swap — only the "uniward_texture" branch above
# exercised them. These tests force the other branches to run.


def test_patched_apply_mode_none_delegates_to_original() -> None:
    """mode=None must reach the legacy path (binds _UNIWARD_ORIGINAL_APPLY)."""
    masks = torch.randint(0, 5, (2, 8, 8), dtype=torch.uint8)
    # Lane SI-V1 path: supply saliency_inv directly. mode=None means the
    # patched function must call _UNIWARD_ORIGINAL_APPLY which dispatches
    # to the legacy implementation.
    saliency_inv = torch.zeros(8, 8, dtype=torch.bool)
    saliency_inv[:, 4:] = True
    payload = apply_saliency_weighted_compression(masks, saliency_inv=saliency_inv)
    parsed = unpack_saliency_payload(payload)
    assert parsed["region_mask"].shape == (8, 8)
    assert parsed["region_mask"][:, 4:].float().mean() > 0.9


def test_patched_apply_uniward_texture_encoder_default_resolves() -> None:
    """encoder=None in uniward_texture mode must resolve _default_zlib_encoder + _encode_with_inv."""
    masks = torch.randint(0, 5, (2, 8, 8), dtype=torch.uint8)
    texture = torch.zeros(8, 8)
    texture[4:, :] = 5.0
    # encoder NOT supplied → forces _default_zlib_encoder lookup AND
    # _encode_with_inv invocation. Both were the F821 names.
    payload = apply_saliency_weighted_compression(
        masks,
        mode="uniward_texture",
        texture_probability=texture,
        texture_quantile=0.5,
        encoder=None,
    )
    parsed = unpack_saliency_payload(payload)
    assert parsed["region_mask"].shape == (8, 8)
    assert parsed["region_mask"][4:, :].float().mean() > 0.9


def test_patched_apply_unsupported_mode_raises() -> None:
    masks = torch.randint(0, 5, (2, 8, 8), dtype=torch.uint8)
    texture = torch.zeros(8, 8)
    with pytest.raises(ValueError, match="unsupported saliency compression mode"):
        apply_saliency_weighted_compression(
            masks,
            mode="not_a_real_mode",
            texture_probability=texture,
        )


def test_patched_apply_uniward_texture_validates_inputs() -> None:
    """Texture-mode validation paths must trigger before any late-bound name."""
    masks = torch.randint(0, 5, (2, 8, 8), dtype=torch.uint8)
    # Missing texture_probability
    with pytest.raises(ValueError, match="requires texture_probability"):
        apply_saliency_weighted_compression(masks, mode="uniward_texture")
    # Bad texture_quantile
    with pytest.raises(ValueError, match="texture_quantile"):
        apply_saliency_weighted_compression(
            masks,
            mode="uniward_texture",
            texture_probability=torch.zeros(8, 8),
            texture_quantile=1.5,
        )
    # Shape mismatch
    with pytest.raises(ValueError, match="shape mismatch"):
        apply_saliency_weighted_compression(
            masks,
            mode="uniward_texture",
            texture_probability=torch.zeros(4, 4),
        )

