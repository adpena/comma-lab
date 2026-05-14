# SPDX-License-Identifier: MIT
"""Phase 2 tests for zeta full-renderer self-compression."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tac.self_compress_full_renderer import (
    DEFAULT_FILM_PROTECT_PATTERNS,
    DEFAULT_QAT_STEPS,
    MAGIC_ZETA,
    FullRendererSelfCompress,
    FullRendererSelfCompressConfig,
    FullRendererSelfCompressError,
    compute_renderer_rate_penalty,
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


def test_full_renderer_self_compress_config_unions_custom_protect_patterns() -> None:
    cfg = FullRendererSelfCompressConfig(
        target_bits_total=50_000,
        protect_patterns=("custom_gate",),
    )
    assert cfg.protect_patterns[: len(DEFAULT_FILM_PROTECT_PATTERNS)] == (
        DEFAULT_FILM_PROTECT_PATTERNS
    )
    assert "custom_gate" in cfg.protect_patterns


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
    with pytest.raises(FullRendererSelfCompressError, match="protect_patterns"):
        FullRendererSelfCompressConfig(target_bits_total=1, protect_patterns=())
    with pytest.raises(FullRendererSelfCompressError, match="film_unprotect_override"):
        FullRendererSelfCompressConfig(target_bits_total=1, protect_film_layers=False)
    cfg = FullRendererSelfCompressConfig(
        target_bits_total=1,
        protect_film_layers=False,
        film_unprotect_override="ALLOW_UNPROTECTED_FILM_COMPRESSION",
    )
    assert cfg.protect_film_layers is False
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


# -- Phase-2 CPU-feasible swap / forward / export / load ----------------


def _make_tiny_renderer() -> nn.Sequential:
    """Tiny CPU-friendly renderer with mixed conv types for swap testing."""
    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=1),  # eligible
        nn.ReLU(),
        nn.Conv2d(4, 4, kernel_size=3, padding=1),  # eligible
        nn.ReLU(),
        nn.Conv2d(4, 2, kernel_size=1),  # eligible
    )


def test_swap_renderer_convs_replaces_eligible_layers() -> None:
    from tac.self_compress import SelfCompressingConv2d

    cfg = FullRendererSelfCompressConfig(target_bits_total=50_000)
    renderer = _make_tiny_renderer()
    wrapper = FullRendererSelfCompress(renderer=renderer, config=cfg)
    diag = wrapper.swap_renderer_convs_with_self_compress()
    assert isinstance(diag, dict)
    # All 3 top-level convs in the Sequential are eligible.
    assert len(diag["swapped"]) == 3
    assert len(diag["protected"]) == 0
    # Every direct child of the Sequential that was originally a Conv2d
    # is now a SelfCompressingConv2d. (SelfCompressingConv2d internally
    # wraps a Conv2d at .conv, so a flat modules() walk still surfaces
    # those inner Conv2d layers — we only care about the swapped roots.)
    swapped_root_types = [
        type(c).__name__ for c in renderer.children()
        if isinstance(c, (nn.Conv2d, SelfCompressingConv2d))
    ]
    assert swapped_root_types == ["SelfCompressingConv2d"] * 3


def test_swap_renderer_convs_protects_film_layers() -> None:
    from tac.self_compress import SelfCompressingConv2d

    cfg = FullRendererSelfCompressConfig(target_bits_total=50_000)
    # Wrap convs under names that match FiLM patterns.
    class _FiLMRenderer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.film_decoder = nn.Conv2d(3, 4, kernel_size=1)
            self.body = nn.Conv2d(4, 4, kernel_size=1)

        def forward(self, x):  # pragma: no cover
            return self.body(self.film_decoder(x))

    renderer = _FiLMRenderer()
    wrapper = FullRendererSelfCompress(renderer=renderer, config=cfg)
    diag = wrapper.swap_renderer_convs_with_self_compress()
    assert "film_decoder" in diag["protected"]
    assert "body" in diag["swapped"]
    assert isinstance(renderer.film_decoder, nn.Conv2d)
    assert isinstance(renderer.body, SelfCompressingConv2d)


def test_full_renderer_forward_runs_after_swap_cpu() -> None:
    cfg = FullRendererSelfCompressConfig(target_bits_total=50_000)
    renderer = _make_tiny_renderer()
    wrapper = FullRendererSelfCompress(renderer=renderer, config=cfg)
    wrapper.swap_renderer_convs_with_self_compress()
    out = wrapper(torch.randn(1, 3, 8, 8))
    assert out.shape[0] == 1
    assert out.shape[1] == 2


def test_compute_renderer_rate_penalty_finite_after_swap() -> None:
    cfg = FullRendererSelfCompressConfig(target_bits_total=50_000)
    renderer = _make_tiny_renderer()
    wrapper = FullRendererSelfCompress(renderer=renderer, config=cfg)
    wrapper.swap_renderer_convs_with_self_compress()
    rate = compute_renderer_rate_penalty(wrapper)
    assert torch.isfinite(rate).item()
    assert rate.item() > 0


def test_train_orchestrator_cpu_smoke_runs_one_step() -> None:
    cfg = FullRendererSelfCompressConfig(target_bits_total=50_000)
    renderer = _make_tiny_renderer()
    wrapper = FullRendererSelfCompress(renderer=renderer, config=cfg)
    out = train_full_renderer_self_compress(
        model=wrapper,
        frames=torch.randn(1, 3, 8, 8),
        scorers=None,
        config=cfg,
        device="cpu",
        smoke_steps=1,
    )
    assert out is wrapper


def test_train_orchestrator_rejects_cpu_without_smoke_steps() -> None:
    cfg = FullRendererSelfCompressConfig(target_bits_total=50_000)
    wrapper = FullRendererSelfCompress(renderer=_make_tiny_renderer(), config=cfg)
    with pytest.raises(FullRendererSelfCompressError, match="smoke_steps"):
        train_full_renderer_self_compress(
            model=wrapper,
            frames=torch.zeros(1, 3, 8, 8),
            scorers=None,
            config=cfg,
            device="cpu",
        )


def test_train_orchestrator_rejects_cuda_when_unavailable() -> None:
    """Per CLAUDE.md `Forbidden device-selection defaults`, no MPS/CPU fallback."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is available — cannot test the no-CUDA refusal path")
    cfg = FullRendererSelfCompressConfig(target_bits_total=50_000)
    wrapper = FullRendererSelfCompress(renderer=_make_tiny_renderer(), config=cfg)
    with pytest.raises(FullRendererSelfCompressError, match="CUDA is not available"):
        train_full_renderer_self_compress(
            model=wrapper,
            frames=torch.zeros(1, 3, 8, 8),
            scorers=None,
            config=cfg,
            device="cuda",
        )


def test_export_load_roundtrip_cpu() -> None:
    cfg = FullRendererSelfCompressConfig(target_bits_total=50_000)
    renderer = _make_tiny_renderer()
    wrapper = FullRendererSelfCompress(renderer=renderer, config=cfg)
    wrapper.swap_renderer_convs_with_self_compress()
    blob = export_full_renderer_self_compress(wrapper, arch_fingerprint="tiny_v0")
    assert blob[:4] == MAGIC_ZETA
    parsed = load_full_renderer_self_compress(blob, arch_config={"name": "tiny_v0"})
    assert parsed["arch_fingerprint"] == "tiny_v0"
    assert isinstance(parsed["layers"], dict)
    assert len(parsed["layers"]) == 3


def test_export_rejects_collapsed_layer() -> None:
    """Pruning every channel triggers FullRendererSelfCompressError."""
    from tac.self_compress import SelfCompressingConv2d

    cfg = FullRendererSelfCompressConfig(
        target_bits_total=50_000, prune_threshold=0.99, bit_depth_init=0.5
    )
    renderer = _make_tiny_renderer()
    wrapper = FullRendererSelfCompress(renderer=renderer, config=cfg)
    wrapper.swap_renderer_convs_with_self_compress()
    # Force every learned bit-depth to a value below prune_threshold.
    for m in renderer.modules():
        if isinstance(m, SelfCompressingConv2d) and hasattr(m.bit_depth, "weight"):
            with torch.no_grad():
                m.bit_depth.weight.fill_(0.1)
    with pytest.raises(FullRendererSelfCompressError, match="0 channels"):
        export_full_renderer_self_compress(wrapper)


def test_load_rejects_bad_magic() -> None:
    with pytest.raises(FullRendererSelfCompressError, match="MAGIC_ZETA"):
        load_full_renderer_self_compress(b"WRNG\x00\x00\x00\x00", arch_config={})
