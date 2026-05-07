from __future__ import annotations

import pytest
import torch

from tac.block_fp_jfg import (
    BlockFPConfig,
    compress_jfg_block_fp,
    decompress_jfg_block_fp,
    is_film_protected,
    quantize_jfg_block_fp,
    validate_film_layer_block_fp,
)


def _state_dict() -> dict[str, torch.Tensor]:
    return {
        "decoder.conv.weight": torch.linspace(-0.25, 0.25, steps=2 * 3 * 2 * 2).reshape(
            2, 3, 2, 2
        ),
        "decoder.conv.bias": torch.linspace(-0.1, 0.1, steps=2),
        "film.gamma.weight": torch.linspace(-0.02, 0.02, steps=16).reshape(4, 4),
    }


def test_block_fp_jfg_roundtrip_is_deterministic_and_protects_film() -> None:
    cfg = BlockFPConfig(block_size=8, lzma_preset=1)
    quantized = quantize_jfg_block_fp(_state_dict(), cfg)

    assert quantized["film.gamma.weight"].protected is True
    assert quantized["decoder.conv.weight"].was_hwoi_permuted is True

    blob1 = compress_jfg_block_fp(quantized, lzma_preset=1)
    blob2 = compress_jfg_block_fp(quantized, lzma_preset=1)
    assert blob1 == blob2

    restored = decompress_jfg_block_fp(blob1)
    assert set(restored) == set(_state_dict())
    assert (
        restored["decoder.conv.weight"].shape
        == _state_dict()["decoder.conv.weight"].shape
    )
    assert restored["decoder.conv.bias"].shape == _state_dict()["decoder.conv.bias"].shape
    torch.testing.assert_close(
        restored["film.gamma.weight"],
        _state_dict()["film.gamma.weight"],
        atol=3e-5,
        rtol=0,
    )


def test_block_fp_jfg_validation_gate_and_input_checks() -> None:
    cfg = BlockFPConfig(block_size=8, film_block_size=4, lzma_preset=1)
    result = validate_film_layer_block_fp(
        torch.linspace(-0.05, 0.05, steps=16).reshape(4, 4),
        cfg,
        layer_name="film.gamma.weight",
        mse_threshold=1e-2,
    )

    assert result.passed is True
    assert result.kill is False
    assert result.effective_bpw > 0.0
    assert is_film_protected("film.gamma.weight", cfg.protect_patterns) is True

    with pytest.raises(ValueError, match="only supports float tensors"):
        quantize_jfg_block_fp({"bad": torch.ones(3, dtype=torch.int64)}, cfg)
