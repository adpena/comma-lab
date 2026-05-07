"""Phase 1 import + dataclass-validation tests for epsilon paradigm.

Tests live within Phase 1 scope:
- All public symbols import cleanly.
- ``MAGIC_LEPR`` magic-byte constant has the expected value + type.
- ``MAX_LEPR_BYTES`` has the expected default budget.
- ``HyperEncoderConfig`` / ``HyperDecoderConfig`` validate their fields.
- ``HyperEncoder`` / ``HyperDecoder`` instantiate but raise
  NotImplementedError on forward (Phase 2 pending).
- ``LearnableEntropyModel`` rate/encode/decode raise NotImplementedError.
- ``LearnableEntropyModelCodec`` build/parse raise NotImplementedError.
"""
from __future__ import annotations

import pytest

from tac.learnable_entropy_model import (
    MAGIC_LEPR,
    MAX_LEPR_BYTES,
    HyperDecoder,
    HyperDecoderConfig,
    HyperEncoder,
    HyperEncoderConfig,
    LearnableEntropyModel,
    LearnableEntropyModelCodec,
    LearnableEntropyModelError,
)

# -- Public-symbol import sanity ----------------------------------------


def test_module_exports_expected_symbols() -> None:
    from tac import learnable_entropy_model as m

    expected = {
        "LearnableEntropyModelError",
        "MAGIC_LEPR",
        "MAX_LEPR_BYTES",
        "HyperEncoderConfig",
        "HyperDecoderConfig",
        "HyperEncoder",
        "HyperDecoder",
        "LearnableEntropyModel",
        "LearnableEntropyModelCodec",
    }
    for name in expected:
        assert hasattr(m, name), f"missing {name}"


def test_magic_lepr_constant() -> None:
    """Magic byte must be exactly b'LEPR' (4 ASCII bytes)."""
    assert isinstance(MAGIC_LEPR, bytes)
    assert MAGIC_LEPR == b"LEPR"
    assert len(MAGIC_LEPR) == 4


def test_max_lepr_bytes_default_budget() -> None:
    """L(M) Occam ceiling - default 5000 bytes per blueprint section 300."""
    assert isinstance(MAX_LEPR_BYTES, int)
    assert MAX_LEPR_BYTES == 5000


# -- HyperEncoderConfig validation --------------------------------------


def test_hyper_encoder_config_accepts_valid_inputs() -> None:
    cfg = HyperEncoderConfig(
        channel_dim=64, hidden_dim=128, n_layers=3, z_dim=16, kernel_size=3
    )
    assert cfg.channel_dim == 64
    assert cfg.kernel_size == 3


def test_hyper_encoder_config_rejects_zero_channel_dim() -> None:
    with pytest.raises(LearnableEntropyModelError, match="channel_dim"):
        HyperEncoderConfig(channel_dim=0, hidden_dim=128, n_layers=3, z_dim=16)


def test_hyper_encoder_config_rejects_zero_n_layers() -> None:
    with pytest.raises(LearnableEntropyModelError, match="n_layers"):
        HyperEncoderConfig(channel_dim=64, hidden_dim=128, n_layers=0, z_dim=16)


def test_hyper_encoder_config_rejects_even_kernel_size() -> None:
    """Kernel must be odd for symmetric padding - Phase 1 enforces."""
    with pytest.raises(LearnableEntropyModelError, match="odd"):
        HyperEncoderConfig(
            channel_dim=64, hidden_dim=128, n_layers=3, z_dim=16, kernel_size=2
        )
    with pytest.raises(LearnableEntropyModelError, match="odd"):
        HyperEncoderConfig(
            channel_dim=64, hidden_dim=128, n_layers=3, z_dim=16, kernel_size=4
        )


def test_hyper_encoder_config_default_kernel_is_three() -> None:
    cfg = HyperEncoderConfig(channel_dim=64, hidden_dim=128, n_layers=3, z_dim=16)
    assert cfg.kernel_size == 3


# -- HyperDecoderConfig validation --------------------------------------


def test_hyper_decoder_config_accepts_gaussian_default() -> None:
    cfg = HyperDecoderConfig(z_dim=16, hidden_dim=128, n_layers=3)
    assert cfg.mode == "gaussian"
    assert cfg.n_mixture_components == 1


def test_hyper_decoder_config_accepts_mixture_mode() -> None:
    cfg = HyperDecoderConfig(
        z_dim=16, hidden_dim=128, n_layers=3, mode="mixture", n_mixture_components=3
    )
    assert cfg.mode == "mixture"
    assert cfg.n_mixture_components == 3


def test_hyper_decoder_config_rejects_unknown_mode() -> None:
    with pytest.raises(LearnableEntropyModelError, match="mode"):
        HyperDecoderConfig(
            z_dim=16, hidden_dim=128, n_layers=3, mode="laplace"  # type: ignore[arg-type]
        )


def test_hyper_decoder_config_rejects_gaussian_with_multiple_components() -> None:
    """Inconsistency: gaussian mode requires n_mixture_components=1."""
    with pytest.raises(LearnableEntropyModelError, match="gaussian"):
        HyperDecoderConfig(
            z_dim=16,
            hidden_dim=128,
            n_layers=3,
            mode="gaussian",
            n_mixture_components=3,
        )


def test_hyper_decoder_config_rejects_zero_components() -> None:
    with pytest.raises(LearnableEntropyModelError, match="n_mixture_components"):
        HyperDecoderConfig(
            z_dim=16,
            hidden_dim=128,
            n_layers=3,
            mode="mixture",
            n_mixture_components=0,
        )


# -- HyperEncoder / HyperDecoder instantiation + Phase 2 stubs -----------


def test_hyper_encoder_instantiates() -> None:
    cfg = HyperEncoderConfig(channel_dim=64, hidden_dim=128, n_layers=3, z_dim=16)
    enc = HyperEncoder(config=cfg)
    assert enc.config is cfg


def test_hyper_encoder_rejects_non_config() -> None:
    with pytest.raises(LearnableEntropyModelError, match="HyperEncoderConfig"):
        HyperEncoder(config="not_a_config")  # type: ignore[arg-type]


def test_hyper_decoder_instantiates() -> None:
    cfg = HyperDecoderConfig(z_dim=16, hidden_dim=128, n_layers=3)
    dec = HyperDecoder(config=cfg)
    assert dec.config is cfg


# -- LearnableEntropyModel ----------------------------------------------


def test_learnable_entropy_model_instantiates_with_matching_z_dims() -> None:
    enc_cfg = HyperEncoderConfig(channel_dim=64, hidden_dim=128, n_layers=3, z_dim=16)
    dec_cfg = HyperDecoderConfig(z_dim=16, hidden_dim=128, n_layers=3)
    model = LearnableEntropyModel(encoder_config=enc_cfg, decoder_config=dec_cfg)
    assert isinstance(model.encoder, HyperEncoder)
    assert isinstance(model.decoder, HyperDecoder)


def test_learnable_entropy_model_rejects_z_dim_mismatch() -> None:
    enc_cfg = HyperEncoderConfig(channel_dim=64, hidden_dim=128, n_layers=3, z_dim=16)
    dec_cfg = HyperDecoderConfig(z_dim=32, hidden_dim=128, n_layers=3)
    with pytest.raises(LearnableEntropyModelError, match="z_dim"):
        LearnableEntropyModel(encoder_config=enc_cfg, decoder_config=dec_cfg)


def test_learnable_entropy_model_rate_raises_phase2_pending() -> None:
    import torch

    enc_cfg = HyperEncoderConfig(channel_dim=64, hidden_dim=128, n_layers=3, z_dim=16)
    dec_cfg = HyperDecoderConfig(z_dim=16, hidden_dim=128, n_layers=3)
    model = LearnableEntropyModel(encoder_config=enc_cfg, decoder_config=dec_cfg)
    with pytest.raises(NotImplementedError, match="Phase 2"):
        model.rate(torch.zeros(1))


def test_learnable_entropy_model_encode_decode_raise_phase2_pending() -> None:
    import torch

    enc_cfg = HyperEncoderConfig(channel_dim=64, hidden_dim=128, n_layers=3, z_dim=16)
    dec_cfg = HyperDecoderConfig(z_dim=16, hidden_dim=128, n_layers=3)
    model = LearnableEntropyModel(encoder_config=enc_cfg, decoder_config=dec_cfg)
    with pytest.raises(NotImplementedError, match="Phase 2"):
        model.encode(torch.zeros(1))
    with pytest.raises(NotImplementedError, match="Phase 2"):
        model.decode(b"", n_symbols=64)


# -- LearnableEntropyModelCodec -----------------------------------------


def test_codec_build_raises_phase2_pending() -> None:
    enc_cfg = HyperEncoderConfig(channel_dim=64, hidden_dim=128, n_layers=3, z_dim=16)
    dec_cfg = HyperDecoderConfig(z_dim=16, hidden_dim=128, n_layers=3)
    model = LearnableEntropyModel(encoder_config=enc_cfg, decoder_config=dec_cfg)
    with pytest.raises(NotImplementedError, match="Phase 2"):
        LearnableEntropyModelCodec.build(model)


def test_codec_parse_raises_phase2_pending() -> None:
    with pytest.raises(NotImplementedError, match="Phase 2"):
        LearnableEntropyModelCodec.parse(b"LEPR\x00\x00\x00\x00")
