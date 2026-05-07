"""Phase 1 tests for zeta full-renderer self-compression scaffolding."""
from __future__ import annotations

import pytest
import torch.nn as nn

from tac.self_compress_full_renderer import (
    DEFAULT_FILM_PROTECT_PATTERNS,
    DEFAULT_QAT_STEPS,
    MAGIC_ZETA,
    FullRendererSelfCompress,
    FullRendererSelfCompressConfig,
    FullRendererSelfCompressError,
    export_full_renderer_self_compress,
    layer_name_matches_film_protect_pattern,
    load_full_renderer_self_compress,
    train_full_renderer_self_compress,
)


def test_zeta_constants_are_ascii_wire_contracts() -> None:
    assert MAGIC_ZETA == b"ZETA"
    assert DEFAULT_QAT_STEPS == 2000
    assert "film" in DEFAULT_FILM_PROTECT_PATTERNS
    assert "gamma" in DEFAULT_FILM_PROTECT_PATTERNS


def test_layer_name_matches_film_protect_pattern_case_insensitive() -> None:
    assert layer_name_matches_film_protect_pattern("renderer.FiLM.gamma")
    assert layer_name_matches_film_protect_pattern("decoder.cond_shift")
    assert not layer_name_matches_film_protect_pattern("renderer.conv1")


def test_layer_name_matcher_validates_inputs() -> None:
    with pytest.raises(FullRendererSelfCompressError, match="qualified_name"):
        layer_name_matches_film_protect_pattern(123)  # type: ignore[arg-type]
    with pytest.raises(FullRendererSelfCompressError, match="patterns"):
        layer_name_matches_film_protect_pattern("x", patterns=["film"])  # type: ignore[arg-type]
    with pytest.raises(FullRendererSelfCompressError, match="every pattern"):
        layer_name_matches_film_protect_pattern("x", patterns=("film", 1))  # type: ignore[arg-type]


def test_layer_name_matcher_empty_patterns_fail_closed() -> None:
    assert not layer_name_matches_film_protect_pattern("renderer.film", patterns=())


def test_full_renderer_self_compress_config_accepts_valid_inputs() -> None:
    cfg = FullRendererSelfCompressConfig(
        target_bits_total=50_000,
        qat_steps=DEFAULT_QAT_STEPS,
        lambda_rate_sc=1e-7,
        protect_film_layers=True,
        bit_depth_init=8.0,
        prune_threshold=0.5,
    )
    assert cfg.target_bits_total == 50_000
    assert cfg.protect_patterns == DEFAULT_FILM_PROTECT_PATTERNS


def test_full_renderer_self_compress_config_rejects_bad_inputs() -> None:
    with pytest.raises(FullRendererSelfCompressError, match="target_bits_total"):
        FullRendererSelfCompressConfig(target_bits_total=0)
    with pytest.raises(FullRendererSelfCompressError, match="qat_steps"):
        FullRendererSelfCompressConfig(
            target_bits_total=1, qat_steps=DEFAULT_QAT_STEPS - 1
        )
    with pytest.raises(FullRendererSelfCompressError, match="lambda_rate_sc"):
        FullRendererSelfCompressConfig(target_bits_total=1, lambda_rate_sc=0.0)
    with pytest.raises(FullRendererSelfCompressError, match="protect_film_layers"):
        FullRendererSelfCompressConfig(
            target_bits_total=1, protect_film_layers="yes"  # type: ignore[arg-type]
        )
    with pytest.raises(FullRendererSelfCompressError, match="bit_depth_init"):
        FullRendererSelfCompressConfig(target_bits_total=1, bit_depth_init=9.0)
    with pytest.raises(FullRendererSelfCompressError, match="prune_threshold"):
        FullRendererSelfCompressConfig(target_bits_total=1, prune_threshold=1.0)


def test_full_renderer_self_compress_wrapper_instantiates() -> None:
    cfg = FullRendererSelfCompressConfig(target_bits_total=50_000)
    renderer = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1))
    wrapper = FullRendererSelfCompress(renderer=renderer, config=cfg)
    assert wrapper.renderer is renderer
    assert wrapper.config is cfg


def test_full_renderer_self_compress_wrapper_validates_inputs() -> None:
    cfg = FullRendererSelfCompressConfig(target_bits_total=50_000)
    with pytest.raises(FullRendererSelfCompressError, match="renderer"):
        FullRendererSelfCompress(renderer=object(), config=cfg)  # type: ignore[arg-type]
    with pytest.raises(FullRendererSelfCompressError, match="config"):
        FullRendererSelfCompress(renderer=nn.Identity(), config=object())  # type: ignore[arg-type]


def test_full_renderer_self_compress_phase2_methods_fail_loud() -> None:
    cfg = FullRendererSelfCompressConfig(target_bits_total=50_000)
    wrapper = FullRendererSelfCompress(renderer=nn.Identity(), config=cfg)

    with pytest.raises(NotImplementedError, match="Phase 2"):
        wrapper.swap_renderer_convs_with_self_compress()
    with pytest.raises(NotImplementedError, match="Phase 2"):
        wrapper.forward()
    with pytest.raises(NotImplementedError, match="Phase 2"):
        train_full_renderer_self_compress(
            model=wrapper, frames=None, scorers=None, config=cfg
        )
    with pytest.raises(NotImplementedError, match="Phase 2"):
        export_full_renderer_self_compress(wrapper)
    with pytest.raises(NotImplementedError, match="Phase 2"):
        load_full_renderer_self_compress(b"ZETA", arch_config={})
