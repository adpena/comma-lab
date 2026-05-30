# SPDX-License-Identifier: MIT
"""Tests for the canonical MLX HWIO -> PyTorch OIHW transpose helper.

Empirically anchored by 5-of-6 MLX -> PyTorch export tools that duplicated
verbatim the HWIO -> OIHW transpose logic per Catalog #290 falling-rule
OBVIOUS-FIT classification. The 6th tool (VQ) remains a PRINCIPLED FORK
preserved via the optional ``skip_buffer_name_predicate`` callback.

Per CLAUDE.md NO FAKE IMPLEMENTATIONS Slot EEE Class 2: these tests verify
BEHAVIOR (canonical OIHW shape after transpose; sha256 stability; fp32
canonicalization; predicate skip) — NOT just metadata constants.

Per CLAUDE.md "Apples-to-apples evidence discipline": comparisons against
the canonical MLX-HWIO -> PyTorch-OIHW reference pattern at
``tools/export_pact_nerv_ia3_mlx_to_pytorch_state_dict.py`` (the canonical
reference implementation of the duplicated-then-extracted pattern).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tac.framework_agnostic.helpers import convert_mlx_state_dict_to_pytorch_oihw

torch = pytest.importorskip("torch")


def _make_canonical_mlx_state_dict() -> dict[str, np.ndarray]:
    """Build a canonical MLX state_dict mirroring real substrate shapes.

    Includes:
      * 4-D Conv2d weight in MLX HWIO layout (out_channels=8, kH=3, kW=3,
        in_channels=4)
      * 2-D Linear weight (no transpose; passes through unchanged)
      * 1-D bias (no transpose)
      * Per-pair tensor (no transpose; canonical PR101 latent shape)
      * Scalar (0-dim; canonical EMA shadow)
    """
    rng = np.random.default_rng(seed=0xCAFEF00D)
    return {
        "decoder.layer0.weight": rng.standard_normal((8, 3, 3, 4), dtype=np.float32),
        "decoder.layer0.bias": rng.standard_normal((8,), dtype=np.float32),
        "head.linear.weight": rng.standard_normal((16, 32), dtype=np.float32),
        "head.linear.bias": rng.standard_normal((16,), dtype=np.float32),
        "latents": rng.standard_normal((600, 28), dtype=np.float32),
    }


def test_canonical_hwio_to_oihw_transpose_for_conv2d_weight() -> None:
    """4-D ``.weight`` MLX HWIO -> PyTorch OIHW canonical transpose."""
    mlx_sd = _make_canonical_mlx_state_dict()
    pytorch_sd, per_tensor = convert_mlx_state_dict_to_pytorch_oihw(mlx_sd)
    # Pre-transpose shape: (out=8, kH=3, kW=3, in=4); post: (out=8, in=4, kH=3, kW=3)
    assert tuple(pytorch_sd["decoder.layer0.weight"].shape) == (8, 4, 3, 3)
    assert per_tensor["decoder.layer0.weight"]["shape_mlx"] == [8, 3, 3, 4]
    assert per_tensor["decoder.layer0.weight"]["shape_pytorch"] == [8, 4, 3, 3]
    assert per_tensor["decoder.layer0.weight"]["layout"] == "mlx_hwio_to_pytorch_oihw"


def test_byte_stable_transpose_matches_canonical_reference_implementation() -> None:
    """Canonical helper output bytes match the inline pattern from the 5 tools.

    Apples-to-apples per CLAUDE.md: reproduce the inline pattern from
    ``tools/export_pact_nerv_ia3_mlx_to_pytorch_state_dict.py:115-130``
    and confirm the canonical helper produces byte-identical output.
    """
    mlx_sd = _make_canonical_mlx_state_dict()
    canonical_sd, _ = convert_mlx_state_dict_to_pytorch_oihw(mlx_sd)

    # Reproduce the inline pattern from the 5 export tools:
    inline_sd: dict[str, torch.Tensor] = {}
    for name, arr in mlx_sd.items():
        out_arr = arr
        if name.endswith(".weight") and arr.ndim == 4:
            out_arr = np.transpose(arr, (0, 3, 1, 2))
        out_arr = np.ascontiguousarray(out_arr).astype(np.float32)
        inline_sd[name] = torch.from_numpy(out_arr.copy())

    # Verify every tensor is byte-identical (canonical apples-to-apples).
    assert canonical_sd.keys() == inline_sd.keys()
    for key in canonical_sd:
        canonical_bytes = canonical_sd[key].numpy().tobytes()
        inline_bytes = inline_sd[key].numpy().tobytes()
        assert canonical_bytes == inline_bytes, f"byte divergence at {key}"


def test_non_conv2d_weights_preserved_unchanged() -> None:
    """Linear weights / biases / per-pair tensors pass through unchanged."""
    mlx_sd = _make_canonical_mlx_state_dict()
    pytorch_sd, per_tensor = convert_mlx_state_dict_to_pytorch_oihw(mlx_sd)
    # 2-D Linear weight: preserved
    assert tuple(pytorch_sd["head.linear.weight"].shape) == (16, 32)
    assert per_tensor["head.linear.weight"]["layout"] == "preserved"
    # 1-D bias: preserved
    assert tuple(pytorch_sd["decoder.layer0.bias"].shape) == (8,)
    assert per_tensor["decoder.layer0.bias"]["layout"] == "preserved"
    # 2-D per-pair latents: preserved
    assert tuple(pytorch_sd["latents"].shape) == (600, 28)
    assert per_tensor["latents"]["layout"] == "preserved"


def test_fp32_canonicalization_for_storage_dtype() -> None:
    """All tensors cast to fp32 per canonical contest-faithful storage."""
    mlx_sd_fp16 = {
        "conv.weight": np.random.default_rng(0).standard_normal((4, 3, 3, 2)).astype(np.float16),
        "bn.running_mean": np.array([0.1, 0.2, 0.3], dtype=np.float16),
    }
    pytorch_sd, per_tensor = convert_mlx_state_dict_to_pytorch_oihw(mlx_sd_fp16)
    for name, tensor in pytorch_sd.items():
        assert tensor.dtype == torch.float32, f"{name} not fp32: got {tensor.dtype}"
        assert per_tensor[name]["dtype"] == "float32"


def test_per_tensor_sha256_canonical_provenance_field() -> None:
    """Each tensor carries a 16-char sha256 prefix per Catalog #323 Provenance."""
    mlx_sd = _make_canonical_mlx_state_dict()
    _, per_tensor = convert_mlx_state_dict_to_pytorch_oihw(mlx_sd)
    for name, meta in per_tensor.items():
        sha = meta["sha256"]
        assert isinstance(sha, str)
        assert len(sha) == 16, f"{name} sha256 not 16-char prefix"
        # Confirm prefix is hex
        int(sha, 16)


def test_per_tensor_sha256_byte_stable_across_runs() -> None:
    """Identical inputs produce identical sha256 sidecars (deterministic)."""
    mlx_sd_a = _make_canonical_mlx_state_dict()
    mlx_sd_b = _make_canonical_mlx_state_dict()
    _, per_tensor_a = convert_mlx_state_dict_to_pytorch_oihw(mlx_sd_a)
    _, per_tensor_b = convert_mlx_state_dict_to_pytorch_oihw(mlx_sd_b)
    for name in per_tensor_a:
        assert per_tensor_a[name]["sha256"] == per_tensor_b[name]["sha256"]


def test_skip_buffer_name_predicate_preserves_vq_quantizer_principled_fork() -> None:
    """VQ ``quantizer.*`` buffers skip transpose via canonical predicate.

    Empirical anchor per ``tools/export_pact_nerv_vq_mlx_to_pytorch_state_dict.py``
    canonical PRINCIPLED FORK per VQ-VAE §3.2 (quantizer.codebook is a
    learnable 2-D embedding table; ema_cluster_size / ema_w are 1-D + 2-D
    EMA buffers; none are Conv2d weights despite the ``.weight``-ish
    naming).
    """
    mlx_sd = {
        # Real Conv2d weight: should transpose (HWIO -> OIHW)
        "decoder.conv.weight": np.zeros((4, 3, 3, 2), dtype=np.float32),
        # VQ quantizer codebook: 2-D embedding table (no transpose; ndim=2 already
        # passes; this test verifies the PREDICATE not the ndim-gate)
        "quantizer.codebook": np.zeros((512, 64), dtype=np.float32),
        # Synthetic 4-D buffer prefixed with quantizer.*: predicate MUST skip
        # even though it would otherwise match the .weight + ndim==4 rule
        "quantizer.synthetic_4d_buffer.weight": np.zeros((8, 3, 3, 4), dtype=np.float32),
    }
    skip_predicate = lambda name: name.startswith("quantizer.")
    pytorch_sd, per_tensor = convert_mlx_state_dict_to_pytorch_oihw(
        mlx_sd, skip_buffer_name_predicate=skip_predicate
    )
    # Real Conv2d weight transposed
    assert tuple(pytorch_sd["decoder.conv.weight"].shape) == (4, 2, 3, 3)
    assert per_tensor["decoder.conv.weight"]["layout"] == "mlx_hwio_to_pytorch_oihw"
    # VQ quantizer.codebook: ndim=2, NOT eligible for transpose anyway, but
    # predicate path takes precedence so layout token is the predicate token
    assert tuple(pytorch_sd["quantizer.codebook"].shape) == (512, 64)
    assert per_tensor["quantizer.codebook"]["layout"] == "skipped_by_predicate"
    # Synthetic 4-D quantizer buffer: predicate skip preserves HWIO shape
    assert tuple(pytorch_sd["quantizer.synthetic_4d_buffer.weight"].shape) == (8, 3, 3, 4)
    assert (
        per_tensor["quantizer.synthetic_4d_buffer.weight"]["layout"]
        == "skipped_by_predicate"
    )


def test_predicate_none_applies_canonical_transpose_to_every_4d_weight() -> None:
    """``skip_buffer_name_predicate=None`` matches the inline 5-tool default."""
    mlx_sd = _make_canonical_mlx_state_dict()
    pytorch_sd_default, _ = convert_mlx_state_dict_to_pytorch_oihw(mlx_sd)
    pytorch_sd_explicit_none, _ = convert_mlx_state_dict_to_pytorch_oihw(
        mlx_sd, skip_buffer_name_predicate=None
    )
    for key in pytorch_sd_default:
        assert torch.equal(pytorch_sd_default[key], pytorch_sd_explicit_none[key])


def test_load_state_dict_strict_round_trip_with_real_module() -> None:
    """Canonical helper output loads into a real torch.nn.Module strict=True.

    Per CLAUDE.md NO FAKE IMPLEMENTATIONS Slot EEE Class 2: this verifies
    BEHAVIOR (actual nn.Module.load_state_dict round-trip) not just shape
    metadata.
    """
    class TinyModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    mlx_sd = {
        "conv.weight": np.random.default_rng(0).standard_normal((8, 3, 3, 4)).astype(np.float32),
        "conv.bias": np.zeros((8,), dtype=np.float32),
    }
    pytorch_sd, _ = convert_mlx_state_dict_to_pytorch_oihw(mlx_sd)
    module = TinyModule().eval()
    load_result = module.load_state_dict(pytorch_sd, strict=True)
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    # Verify the actual forward pass works
    dummy_input = torch.zeros(1, 4, 5, 5)
    with torch.no_grad():
        output = module(dummy_input)
    assert tuple(output.shape) == (1, 8, 3, 3)


def test_empty_state_dict_round_trip() -> None:
    """Empty input produces empty output (canonical no-op)."""
    pytorch_sd, per_tensor = convert_mlx_state_dict_to_pytorch_oihw({})
    assert pytorch_sd == {}
    assert per_tensor == {}


def test_3d_weight_not_transposed() -> None:
    """3-D ``.weight`` tensors are NOT transposed (only ndim==4 Conv2d)."""
    mlx_sd = {"depthwise.weight": np.zeros((8, 3, 4), dtype=np.float32)}
    pytorch_sd, per_tensor = convert_mlx_state_dict_to_pytorch_oihw(mlx_sd)
    assert tuple(pytorch_sd["depthwise.weight"].shape) == (8, 3, 4)
    assert per_tensor["depthwise.weight"]["layout"] == "preserved"


def test_non_weight_4d_tensor_not_transposed() -> None:
    """4-D tensors NOT ending in ``.weight`` pass through unchanged."""
    mlx_sd = {"buffer.running_var": np.zeros((4, 3, 3, 2), dtype=np.float32)}
    pytorch_sd, per_tensor = convert_mlx_state_dict_to_pytorch_oihw(mlx_sd)
    assert tuple(pytorch_sd["buffer.running_var"].shape) == (4, 3, 3, 2)
    assert per_tensor["buffer.running_var"]["layout"] == "preserved"


def test_canonical_helper_importable_via_package_facade() -> None:
    """``from tac.framework_agnostic import convert_mlx_state_dict_to_pytorch_oihw`` works."""
    from tac.framework_agnostic import (
        convert_mlx_state_dict_to_pytorch_oihw as via_facade,
    )

    assert via_facade is convert_mlx_state_dict_to_pytorch_oihw


def test_canonical_helper_in_module_all_export() -> None:
    """Helper is in ``tac.framework_agnostic.helpers.__all__``."""
    from tac.framework_agnostic import helpers as helpers_mod

    assert "convert_mlx_state_dict_to_pytorch_oihw" in helpers_mod.__all__
