# SPDX-License-Identifier: MIT
"""Tests for decoder-weight coder-aware QAT rate pressure."""

from __future__ import annotations

import pytest

from tac.substrates._shared.mlx_score_aware.coder_qat import (
    CoderAwareQATConfig,
    build_decoder_c1a_entropy_term,
    build_decoder_coder_qat_terms,
    coder_qat_loss_weights,
    coder_qat_metadata,
)
from tac.substrates._shared.mlx_score_aware.device_gate import (
    MlxScoreAwareHarnessError,
)

try:
    import mlx.core as _mx  # noqa: F401

    _MLX = True
except ImportError:
    _MLX = False

mlx_only = pytest.mark.skipif(not _MLX, reason="MLX required (Apple Silicon)")


class _TinyParamTree:
    def __init__(self, decoder_values, latent_values=None):
        import mlx.core as mx

        self._params = {
            "decoder": {
                "blocks": {"weight": mx.array(decoder_values, dtype=mx.float32)},
            },
            "latents": mx.array(
                latent_values if latent_values is not None else [0.2, -0.4],
                dtype=mx.float32,
            ),
        }

    def parameters(self):
        return self._params


@mlx_only
def test_disabled_qat_emits_no_terms_or_weights() -> None:
    cfg = CoderAwareQATConfig(enabled=False)
    model = _TinyParamTree([0.1, -0.2, 0.3])

    assert build_decoder_coder_qat_terms(model, cfg) == {}
    assert coder_qat_loss_weights(cfg) == {}


@mlx_only
def test_quant_residual_is_zero_on_symmetric_int_grid() -> None:
    import mlx.core as mx

    cfg = CoderAwareQATConfig(enabled=True, quant_bits=2)
    model = _TinyParamTree([-1.0, 0.0, 1.0])
    terms = build_decoder_coder_qat_terms(model, cfg)
    mx.eval(terms["coder_qat_quant_residual"])

    assert float(terms["coder_qat_quant_residual"].item()) == pytest.approx(0.0)


@mlx_only
def test_quant_residual_uses_archive_axis0_scale_for_decoder_matrices() -> None:
    import mlx.core as mx

    cfg = CoderAwareQATConfig(enabled=True, quant_bits=2)
    model = _TinyParamTree([[0.0, 0.0, 4.0], [0.5, 0.0, 0.0]])
    terms = build_decoder_coder_qat_terms(model, cfg)
    mx.eval(terms["coder_qat_quant_residual"])

    assert float(terms["coder_qat_quant_residual"].item()) == pytest.approx(0.0)


@mlx_only
def test_quant_residual_penalizes_off_grid_decoder_weights() -> None:
    import mlx.core as mx

    cfg = CoderAwareQATConfig(enabled=True, quant_bits=2)
    model = _TinyParamTree([0.0, 0.25, 1.0])
    terms = build_decoder_coder_qat_terms(model, cfg)
    mx.eval(terms["coder_qat_quant_residual"])

    assert float(terms["coder_qat_quant_residual"].item()) > 0.0


@mlx_only
def test_qat_selection_excludes_latents_from_decoder_pressure() -> None:
    import mlx.core as mx

    cfg = CoderAwareQATConfig(enabled=True, quant_bits=2)
    baseline = _TinyParamTree([0.0, 1.0], latent_values=[0.25, 0.5])
    changed_latents = _TinyParamTree([0.0, 1.0], latent_values=[0.125, 0.75])

    terms_a = build_decoder_coder_qat_terms(baseline, cfg)
    terms_b = build_decoder_coder_qat_terms(changed_latents, cfg)
    mx.eval(terms_a["coder_qat_magnitude"], terms_b["coder_qat_magnitude"])

    assert float(terms_a["coder_qat_magnitude"].item()) == pytest.approx(
        float(terms_b["coder_qat_magnitude"].item())
    )


@mlx_only
def test_c1a_entropy_is_finite_and_positive_for_decoder_weights() -> None:
    import mlx.core as mx

    cfg = CoderAwareQATConfig(enabled=True, quant_bits=4)
    model = _TinyParamTree([-1.0, -0.2, 0.0, 0.35, 0.9])

    entropy = build_decoder_c1a_entropy_term(model, cfg, sigma=1.0)
    mx.eval(entropy)

    assert float(entropy.item()) > 0.0


@mlx_only
def test_c1a_entropy_selection_excludes_latents_from_decoder_pressure() -> None:
    import mlx.core as mx

    cfg = CoderAwareQATConfig(enabled=True, quant_bits=4)
    baseline = _TinyParamTree([0.0, 0.5, 1.0], latent_values=[0.25, 0.5])
    changed_latents = _TinyParamTree([0.0, 0.5, 1.0], latent_values=[8.0, -9.0])

    entropy_a = build_decoder_c1a_entropy_term(baseline, cfg, sigma=0.75)
    entropy_b = build_decoder_c1a_entropy_term(changed_latents, cfg, sigma=0.75)
    mx.eval(entropy_a, entropy_b)

    assert float(entropy_a.item()) == pytest.approx(float(entropy_b.item()))


@mlx_only
def test_c1a_entropy_fails_closed_on_invalid_controls() -> None:
    cfg = CoderAwareQATConfig(enabled=True, quant_bits=4)
    model = _TinyParamTree([0.0, 0.5, 1.0])

    with pytest.raises(MlxScoreAwareHarnessError, match="sigma"):
        build_decoder_c1a_entropy_term(model, cfg, sigma=0.0)
    with pytest.raises(MlxScoreAwareHarnessError, match="sample_size"):
        build_decoder_c1a_entropy_term(model, cfg, sigma=1.0, sample_size=0)


def test_qat_config_fails_closed_on_invalid_values() -> None:
    with pytest.raises(MlxScoreAwareHarnessError, match="quant_bits"):
        CoderAwareQATConfig(enabled=True, quant_bits=0).validated()
    with pytest.raises(MlxScoreAwareHarnessError, match="non-negative"):
        CoderAwareQATConfig(enabled=True, magnitude_weight=-1.0).validated()
    with pytest.raises(MlxScoreAwareHarnessError, match="include_substrings"):
        CoderAwareQATConfig(enabled=True, include_substrings=()).validated()


def test_enabled_qat_keeps_zero_weight_terms_explicit() -> None:
    weights = coder_qat_loss_weights(
        CoderAwareQATConfig(
            enabled=True,
            quant_residual_weight=1.0e-4,
            magnitude_weight=0.0,
            delta_weight=0.0,
        )
    )

    assert weights["coder_qat_quant_residual"] == pytest.approx(1.0e-4)
    assert weights["coder_qat_magnitude"] == pytest.approx(0.0)
    assert weights["coder_qat_delta"] == pytest.approx(0.0)


def test_qat_metadata_avoids_duplicate_authority_fields() -> None:
    metadata = coder_qat_metadata(CoderAwareQATConfig(enabled=True, quant_bits=4))

    assert metadata["schema"] == "coder_aware_decoder_qat.v1"
    assert metadata["enabled"] is True
    assert metadata["quant_bits"] == 4
    assert metadata["quantizer_geometry"] == (
        "symmetric_signed_axis0_fp16_scale_for_matrix_conv_weights_"
        "per_tensor_fp16_scale_for_biases"
    )
    assert metadata["authority"] == "false_macos_mlx_research_signal"
    assert "score_claim" not in metadata
    assert "promotion_eligible" not in metadata
    assert "ready_for_exact_eval_dispatch" not in metadata
