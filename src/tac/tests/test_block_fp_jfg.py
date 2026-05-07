from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tac.block_fp_jfg import (
    BFJ1_VERSION,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_PROTECT_PATTERNS,
    MAGIC_BFJ1,
    BlockFPConfig,
    BlockFPTensor,
    ValidationResult,
    compress_jfg_block_fp,
    decompress_jfg_block_fp,
    is_film_protected,
    quantize_jfg_block_fp,
    validate_film_layer_block_fp,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
UNPACK_RENDERER_PAYLOAD_PATH = (
    REPO_ROOT / "submissions" / "robust_current" / "unpack_renderer_payload.py"
)


def _load_unpack_renderer_payload_module():
    spec = importlib.util.spec_from_file_location(
        "bfj1_unpack_renderer_payload_under_test",
        UNPACK_RENDERER_PAYLOAD_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


# ── Deliverable-required tests (per Track F spec §5) ─────────────────────


def test_block_fp_roundtrip_small() -> None:
    """Tiny 2-layer Conv2d; per-layer MSE < 0.01 after roundtrip."""
    torch.manual_seed(0)
    sd: dict[str, torch.Tensor] = {
        "conv_a.weight": torch.randn(8, 4, 3, 3) * 0.1,
        "conv_a.bias": torch.randn(8) * 0.01,
        "conv_b.weight": torch.randn(4, 8, 1, 1) * 0.05,
        "conv_b.bias": torch.randn(4) * 0.01,
    }
    cfg = BlockFPConfig(block_size=8, protect_film_layers=True, lzma_preset=1)
    quantized = quantize_jfg_block_fp(sd, cfg)
    blob = compress_jfg_block_fp(quantized, lzma_preset=1)
    restored = decompress_jfg_block_fp(blob)
    for name, orig in sd.items():
        recon = restored[name]
        assert recon.shape == orig.shape, name
        assert recon.dtype == orig.dtype, name
        mse = (recon.to(torch.float32) - orig.to(torch.float32)).pow(2).mean()
        assert mse.item() < 0.01, f"{name} mse={mse.item():.6f}"


def test_film_layer_protection_default() -> None:
    """Default config: FiLM-named layers go through FP16 protected path."""
    torch.manual_seed(1)
    sd = {
        "block.film_proj.weight": torch.randn(112, 48) * 0.1,
        "block.film_proj.bias": torch.randn(112) * 0.01,
        "pose_mlp.0.weight": torch.randn(48, 6) * 0.1,
        "pose_mlp.2.weight": torch.randn(48, 48) * 0.1,
        "block.conv1.weight": torch.randn(56, 56, 1, 1) * 0.05,
    }
    cfg = BlockFPConfig(block_size=16, protect_film_layers=True, lzma_preset=1)
    quantized = quantize_jfg_block_fp(sd, cfg)
    assert quantized["block.film_proj.weight"].protected is True
    assert quantized["block.film_proj.weight"].int8_mantissa == b""
    assert len(quantized["block.film_proj.weight"].fp16_payload) == 112 * 48 * 2
    assert quantized["block.film_proj.bias"].protected is True
    assert quantized["pose_mlp.0.weight"].protected is True
    assert quantized["pose_mlp.2.weight"].protected is True
    assert quantized["block.conv1.weight"].protected is False
    assert len(quantized["block.conv1.weight"].int8_mantissa) > 0


def test_film_validation_kill_on_high_mse() -> None:
    """FiLM layer with ONE huge value alongside O(1) values in the SAME
    per-block window: the per-block exponent must scale to the huge value,
    so the small values collapse to 0 → high reconstruction MSE on the
    small entries → ValidationResult.kill=True."""
    torch.manual_seed(2)
    # Single block of 64 entries: one outlier 1e6, the rest order-1 random.
    # Per-block scale 2^e ≈ 2^14, so order-1 entries / 2^14 ≈ 6e-5 → rounds
    # to 0 → reconstruction loses them → MSE ≈ mean(order_1**2) ≈ O(1).
    inner = torch.cat(
        [torch.tensor([1e6]), torch.randn(63) * 0.5]
    ).reshape(1, 64)
    cfg = BlockFPConfig(block_size=64, film_block_size=64, lzma_preset=1)
    result = validate_film_layer_block_fp(
        inner, cfg, layer_name="film_proj.weight", mse_threshold=1e-3
    )
    assert isinstance(result, ValidationResult)
    assert result.kill is True, (
        f"expected kill=True given outlier+small mix; got "
        f"mse={result.roundtrip_mse} threshold={result.threshold}"
    )
    assert result.passed is False
    assert result.roundtrip_mse > result.threshold


def test_hwoi_permutation_roundtrip() -> None:
    """Conv2d (3, 4, 5, 6); HWOI permute + inverse via codec roundtrip."""
    torch.manual_seed(4)
    w = torch.randn(3, 4, 5, 6) * 0.1
    cfg = BlockFPConfig(block_size=8, hwoi_permute=True, lzma_preset=1)
    blob = compress_jfg_block_fp(
        quantize_jfg_block_fp({"conv.weight": w}, cfg), lzma_preset=1
    )
    restored = decompress_jfg_block_fp(blob)
    assert restored["conv.weight"].shape == (3, 4, 5, 6)
    mse = (restored["conv.weight"] - w).pow(2).mean().item()
    assert mse < 0.001


def test_bfj1_magic_byte_constant() -> None:
    assert MAGIC_BFJ1 == b"BFJ1"
    assert BFJ1_VERSION == 1
    assert DEFAULT_BLOCK_SIZE == 64


def test_compressed_size_is_bounded_for_jfg_class_fixture() -> None:
    """Synthetic ~80K-param JFG-class state-dict stays in a sane byte band.

    This is a structural guard, not a score or compression-ratio claim. The
    actual frontier question is whether a trained model plus decoder overhead
    beats the current packed baseline after exact archive custody.
    """
    rng = torch.Generator().manual_seed(7)

    def laplacian(*shape: int, scale: float = 0.05) -> torch.Tensor:
        """Sparse-near-zero (Laplacian-like) trained-weight surrogate.
        ~70% of values within ±0.01; long tail to ±0.5."""
        u = torch.rand(*shape, generator=rng) - 0.5
        return torch.sign(u) * (-torch.log(1 - 2 * u.abs() + 1e-9)) * scale

    sd: dict[str, torch.Tensor] = {}
    sd["shared_trunk.stem.weight"] = laplacian(56, 8, 1, 1)
    sd["shared_trunk.stem.bias"] = laplacian(56)
    sd["shared_trunk.down.weight"] = laplacian(64, 56, 1, 1)
    sd["shared_trunk.down.bias"] = laplacian(64)
    sd["shared_trunk.fuse.weight"] = laplacian(56, 112, 1, 1)
    sd["shared_trunk.fuse.bias"] = laplacian(56)
    sd["frame1_head.conv1.weight"] = laplacian(56, 56, 1, 1)
    sd["frame2_head.conv1.weight"] = laplacian(56, 56, 1, 1)
    sd["frame1_head.head.weight"] = laplacian(3, 52, 1, 1)
    sd["frame1_head.head.bias"] = laplacian(3)
    sd["frame2_head.head.weight"] = laplacian(3, 52, 1, 1)
    sd["frame2_head.head.bias"] = laplacian(3)
    sd["frame1_head.block1.film_proj.weight"] = laplacian(112, 48, scale=0.1)
    sd["frame1_head.block1.film_proj.bias"] = laplacian(112)
    sd["pose_mlp.0.weight"] = laplacian(48, 6, scale=0.1)
    sd["pose_mlp.2.weight"] = laplacian(48, 48, scale=0.1)
    sd["dense_a.weight"] = laplacian(64, 64, 3, 3)
    sd["dense_b.weight"] = laplacian(56, 56, 3, 3)
    total = sum(t.numel() for t in sd.values())
    assert total >= 70_000, total
    cfg = BlockFPConfig(block_size=64, lzma_preset=9)
    blob = compress_jfg_block_fp(
        quantize_jfg_block_fp(sd, cfg), lzma_preset=9
    )
    assert len(blob) < 100 * 1024, (
        f"compressed blob is {len(blob)} bytes for {total} params"
    )
    assert len(blob) > 256


def test_inflate_dispatch_can_load_bfj1_archive(tmp_path) -> None:
    """End-to-end seam: BFJ1 archive on disk → state_dict round-trip.

    On-disk format:
        outer 4-byte BFJ1 magic + lzma-compressed inner envelope.
    The inner envelope ALSO begins with BFJ1 (canonical envelope-magic
    after lzma decompression). This double-magic layout makes inflate
    dispatch a simple ``magic == b"BFJ1"`` check on the file's first
    4 bytes; the codec's decompressor handles everything else.
    """
    torch.manual_seed(8)
    sd = {
        "stem.weight": torch.randn(16, 8, 3, 3) * 0.1,
        "stem.bias": torch.randn(16) * 0.01,
        "head.weight": torch.randn(3, 16, 1, 1) * 0.05,
        "head.bias": torch.randn(3) * 0.01,
    }
    cfg = BlockFPConfig(block_size=8, lzma_preset=1)
    blob = compress_jfg_block_fp(
        quantize_jfg_block_fp(sd, cfg), lzma_preset=1
    )
    bin_path = tmp_path / "renderer.bin"
    bin_path.write_bytes(blob)
    raw = bin_path.read_bytes()
    # Inflate-side dispatch checks: first 4 bytes are BFJ1 magic.
    assert raw[:4] == MAGIC_BFJ1
    # After magic strip, the rest is lzma.
    assert raw[4:10] == b"\xfd7zXZ\x00"
    import lzma as _lzma
    envelope = _lzma.decompress(raw[4:])
    assert envelope[:4] == MAGIC_BFJ1
    (version,) = struct.unpack("<I", envelope[4:8])
    assert version == BFJ1_VERSION
    restored = decompress_jfg_block_fp(raw)
    assert set(restored.keys()) == set(sd.keys())
    for name, orig in sd.items():
        mse = (restored[name] - orig).pow(2).mean().item()
        assert mse < 0.01


def test_robust_inflate_renderer_loads_bfj1_without_torch_load(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contest runtime dispatches BFJ1 before any pickle/torch fallback."""
    from submissions.robust_current import inflate_renderer
    from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer

    torch.manual_seed(9)
    model = build_quantizr_faithful_renderer().eval()
    cfg = BlockFPConfig(block_size=32, lzma_preset=1)
    blob = compress_jfg_block_fp(
        quantize_jfg_block_fp(model.state_dict(), cfg),
        lzma_preset=1,
    )
    assert blob[:4] == MAGIC_BFJ1
    renderer_path = tmp_path / "renderer.bin"
    renderer_path.write_bytes(blob)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("BFJ1 runtime path fell through to torch.load")

    monkeypatch.setattr(inflate_renderer.torch, "load", forbidden)
    wrapped = inflate_renderer._load_renderer(str(renderer_path), "cpu")

    assert getattr(wrapped, "q_faithful", False) is True
    assert inflate_renderer._is_asymmetric_model(wrapped)
    assert wrapped.pose_dim == 6

    mask = torch.zeros((1, 8, 8), dtype=torch.long)
    pose = torch.zeros((1, wrapped.pose_dim), dtype=torch.float32)
    with torch.no_grad():
        pair = wrapped(mask, mask, pose=pose)
    assert pair.shape == (1, 2, 384, 512, 3)
    assert torch.isfinite(pair).all()


def test_unpack_renderer_payload_recognizes_bfj1_renderer_payload() -> None:
    unpacker = _load_unpack_renderer_payload_module()
    payload = MAGIC_BFJ1 + b"synthetic-bfj1-body"

    assert unpacker._looks_like_renderer_payload(payload) is True
    assert unpacker._renderer_payload_codec_label(payload) == "brotli_bfj1"


def test_bfj1_is_visible_to_runtime_preflight_and_composition_allowlist(
    tmp_path,
) -> None:
    from tac.preflight import preflight_check
    from tac.stack_compositions import _SCORER_FREE_RENDERER_MAGICS

    renderer = tmp_path / "renderer.bin"
    renderer.write_bytes(MAGIC_BFJ1 + b"synthetic-bfj1-body")

    assert preflight_check(renderer_path=renderer, verbose=False) == []
    assert MAGIC_BFJ1 in _SCORER_FREE_RENDERER_MAGICS


def test_compress_empty_dict_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        compress_jfg_block_fp({}, lzma_preset=1)


def test_decompress_against_skeleton_strict_keys() -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(4, 8, 1)

    model = TinyModel()
    blob = compress_jfg_block_fp(
        quantize_jfg_block_fp(
            {"conv.weight": torch.randn(8, 4, 1, 1) * 0.1},
            BlockFPConfig(block_size=4, lzma_preset=1),
        ),
        lzma_preset=1,
    )
    with pytest.raises(ValueError, match="state_dict key mismatch"):
        decompress_jfg_block_fp(blob, model_skeleton=model)


def test_protected_pattern_override() -> None:
    sd = {
        "layer.special.weight": torch.randn(8, 4, 1, 1) * 0.1,
        "layer.normal.weight": torch.randn(8, 4, 1, 1) * 0.1,
    }
    cfg = BlockFPConfig(
        block_size=4, protect_patterns=("special",), lzma_preset=1
    )
    quantized = quantize_jfg_block_fp(sd, cfg)
    assert quantized["layer.special.weight"].protected is True
    assert quantized["layer.normal.weight"].protected is False


def test_layer_order_is_deterministic() -> None:
    torch.manual_seed(9)
    a = torch.randn(8, 4, 1, 1) * 0.1
    b = torch.randn(8, 4, 1, 1) * 0.1
    cfg = BlockFPConfig(block_size=4, lzma_preset=1)
    blob_ab = compress_jfg_block_fp(
        quantize_jfg_block_fp({"a.weight": a, "b.weight": b}, cfg),
        lzma_preset=1,
    )
    blob_ba = compress_jfg_block_fp(
        quantize_jfg_block_fp({"b.weight": b, "a.weight": a}, cfg),
        lzma_preset=1,
    )
    assert blob_ab == blob_ba


def test_default_protect_patterns_constant() -> None:
    assert "film" in DEFAULT_PROTECT_PATTERNS
    assert is_film_protected("frame1.film_proj.weight", DEFAULT_PROTECT_PATTERNS)
    assert not is_film_protected("conv.weight", DEFAULT_PROTECT_PATTERNS)


def test_blockfptensor_dataclass_frozen() -> None:
    bft = BlockFPTensor(
        int8_mantissa=b"",
        block_exponents=b"",
        block_size=0,
        original_shape=(2, 3),
        dtype_target="float32",
        actual_param_count=6,
        was_hwoi_permuted=False,
        protected=True,
        fp16_payload=b"\x00" * 12,
    )
    with pytest.raises((AttributeError, Exception)):
        bft.protected = False  # type: ignore[misc]
