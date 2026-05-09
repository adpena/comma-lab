"""T15 — Time-varying (per-pair pose-conditioned) FiLM modulators.

Council provenance
------------------
This module operationalizes Track T15 from the Fields-Medal Grand Council
Eureka session (memory ``feedback_grand_council_fields_medal_eureka_mode_implement_landing_20260509.md``,
§Quantizr eureka).

Quantizr's 88K-param FiLM-DSConv (the 0.33 archive baseline) uses a
**static** FiLM conditioning: a single ``γ, β ∈ R^C`` learned at
training time and shared across all pairs. T15 generalizes this to
**per-pair** FiLM where the modulators ``γ_t, β_t ∈ R^C`` are themselves
a function of the per-pair pose delta (which we already transmit in the
archive).

This is a **hyperprior on the renderer**. A small modulator MLP takes the
per-pair pose-delta vector as input and outputs ``(γ_t, β_t)``. The MLP
is shipped in the archive (~2K params); the per-pair pose is already in
the archive. Net cost is 0 added per-pair bytes, plus one tiny MLP.

Mathematical core
-----------------
Static FiLM (Quantizr):

    h'(x) = γ ⊙ h(x) + β
    where γ, β ∈ R^C are learned scalars

Time-varying FiLM (T15):

    h'_t(x) = γ_t ⊙ h(x) + β_t
    where (γ_t, β_t) = MLP_modulator(pose_delta_t)
    and MLP_modulator: R^pose_dim → R^{2C}

The modulator MLP is small (default: 1 hidden layer, 32 units → 2C
outputs). For 88K-param Quantizr-style decoder with C=64 channels, the
MLP is ~6K params * one MLP shared across all per-pair invocations.
The per-pair pose vector (6 dims) is already transmitted in the archive
per the contest pose-stream contract.

CLAUDE.md compliance
--------------------
- **Dev-only ($0 GPU)**: pure CPU module construction + tests; no GPU
  spend in this scaffold.
- **EMA-compatible**: the modulator MLP is a normal nn.Module — when
  used inside a training loop, the trainer's existing EMA(0.997) wraps
  it like any other parameter set.
- **No silent defaults**: ``TimeVaryingFiLMConfig`` validates all
  fields in ``__post_init__``.
- **No score claims**: this module produces nn.Modules + byte
  estimators; any score claim derived from inserting time-varying FiLM
  into a renderer MUST be tagged ``[predicted; council eureka source]``.
- **Strict-scorer-rule**: this module performs NO scorer load.
- **No /tmp paths**: in-memory modules + closed-form byte counts.

Public API
----------
- ``TimeVaryingFiLMConfig`` — dataclass + validation
- ``TimeVaryingFiLM`` — nn.Module factory; produces a per-pair FiLM
  modulator MLP and the apply-FiLM forward
- ``compute_per_pair_gamma_beta`` — pure helper invoking the modulator
  MLP on a batch of pose deltas
- ``time_varying_film_state_bytes`` — closed-form archive-byte estimate
  for the modulator MLP at given quantization (FP4 default; FP8/FP16
  also supported)

Cross-references
----------------
- Council eureka memo: ``feedback_grand_council_fields_medal_eureka_mode_implement_landing_20260509.md`` (Quantizr eureka)
- Phase 2 launch memo (this session): ``feedback_paradigm_dezeta_phase2_architectural_launch_20260509.md``
- Quantizr static FiLM: ``architectures.py::FiLMQATPostFilter`` (the
  static analog T15 generalizes)
- T10 IB-Lagrangian aux scorer: :mod:`tac.ib_lagrangian_aux_scorer`
- T17 shared VQ-VAE codebook: :mod:`tac.shared_vq_codebook` (Phase 3
  composition target)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


__all__ = [
    "TimeVaryingFiLMConfig",
    "TimeVaryingFiLM",
    "compute_per_pair_gamma_beta",
    "time_varying_film_state_bytes",
    "TimeVaryingFiLMError",
]


class TimeVaryingFiLMError(RuntimeError):
    """Raised on configuration or runtime invariant violations."""


# Bytes-per-parameter at common quantization levels. Conservative picks
# matching the council's byte-budget reasoning (FP4 = 0.5 bytes/param,
# FP8 = 1, FP16 = 2, FP32 = 4).
_BYTES_PER_PARAM_BY_QUANT: dict[str, float] = {
    "fp4": 0.5,
    "fp8": 1.0,
    "fp16": 2.0,
    "fp32": 4.0,
}


@dataclass(frozen=True)
class TimeVaryingFiLMConfig:
    """Config for the per-pair pose-conditioned FiLM modulator MLP.

    Attributes
    ----------
    pose_dim
        Per-pair pose vector dimensionality. Contest = 6 (FastViT-T12
        hydra head first 6 dims). MUST be > 0.
    feature_channels
        Number of channels ``C`` in the renderer's intermediate feature
        map being modulated. Quantizr canonical ``C=64``; T15 supports
        any ``C >= 1``. MUST be > 0.
    hidden_dim
        Hidden-layer width of the modulator MLP. Default 32 per council
        eureka memo. MUST be > 0.
    activation
        Activation function name. ``"relu"`` (default), ``"gelu"``,
        ``"silu"``. Must be in supported set.
    quantization
        Modulator MLP quantization for archive-byte estimation. Default
        ``"fp4"`` matches Quantizr archive convention. Must be in
        ``_BYTES_PER_PARAM_BY_QUANT``.
    label
        Operator label for the modulator instance; used for log file
        naming and EMA-checkpoint identification. Required, non-empty.
    """

    pose_dim: int
    feature_channels: int
    hidden_dim: int
    activation: str
    quantization: str
    label: str

    def __post_init__(self) -> None:
        if not (self.pose_dim > 0):
            raise TimeVaryingFiLMError(
                f"pose_dim must be > 0; got {self.pose_dim}"
            )
        if not (self.feature_channels > 0):
            raise TimeVaryingFiLMError(
                f"feature_channels must be > 0; got {self.feature_channels}"
            )
        if not (self.hidden_dim > 0):
            raise TimeVaryingFiLMError(
                f"hidden_dim must be > 0; got {self.hidden_dim}"
            )
        if self.activation not in {"relu", "gelu", "silu"}:
            raise TimeVaryingFiLMError(
                f"activation must be one of relu/gelu/silu; got {self.activation!r}"
            )
        if self.quantization not in _BYTES_PER_PARAM_BY_QUANT:
            raise TimeVaryingFiLMError(
                f"quantization must be one of "
                f"{sorted(_BYTES_PER_PARAM_BY_QUANT.keys())}; "
                f"got {self.quantization!r}"
            )
        if not isinstance(self.label, str) or not self.label.strip():
            raise TimeVaryingFiLMError("label must be a non-empty string")

    @classmethod
    def quantizr_canonical(
        cls,
        *,
        label: str,
        feature_channels: int = 64,
    ) -> "TimeVaryingFiLMConfig":
        """Return the Quantizr-canonical config (pose=6, C=64, hidden=32, FP4)."""
        return cls(
            pose_dim=6,
            feature_channels=feature_channels,
            hidden_dim=32,
            activation="relu",
            quantization="fp4",
            label=label,
        )


def _require_torch() -> "tuple[Any, Any]":
    try:
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415
    except ImportError as exc:
        raise TimeVaryingFiLMError(
            "torch is required for TimeVaryingFiLM instantiation."
        ) from exc
    return torch, nn


def _build_activation(name: str) -> Any:
    torch, nn = _require_torch()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=False)
    # Defensive — config validates name; this path is unreachable.
    raise TimeVaryingFiLMError(f"Unknown activation: {name!r}")


def TimeVaryingFiLM(config: TimeVaryingFiLMConfig) -> Any:  # noqa: N802
    """Factory: build the per-pair FiLM modulator nn.Module.

    The returned module exposes:

      - ``forward(pose_delta) -> (γ, β)``: ``pose_delta`` shape
        ``(B, pose_dim)``; outputs each shape ``(B, feature_channels)``.
      - ``apply_film(features, pose_delta) -> features'``: applies
        ``γ ⊙ features + β`` channel-wise on a ``(B, C, H, W)`` feature
        map, with γ/β computed from ``pose_delta``.

    The modulator MLP is small (typically <10K params @ FP4 ≤ 5KB).
    """
    if not isinstance(config, TimeVaryingFiLMConfig):
        raise TimeVaryingFiLMError(
            f"TimeVaryingFiLM requires TimeVaryingFiLMConfig; "
            f"got {type(config).__name__}"
        )
    torch, nn = _require_torch()

    activation_name = config.activation
    pose_dim = config.pose_dim
    feature_channels = config.feature_channels
    hidden_dim = config.hidden_dim

    class _TimeVaryingFiLMModule(nn.Module):
        """Per-pair pose-conditioned FiLM modulator (T15)."""

        def __init__(self) -> None:
            super().__init__()
            self.modulator = nn.Sequential(
                nn.Linear(pose_dim, hidden_dim),
                _build_activation(activation_name),
                # Output 2 * C: first half = γ (gain), second half = β (bias).
                nn.Linear(hidden_dim, 2 * feature_channels),
            )
            # Quantizr convention: initialize γ near 1, β near 0 so the
            # initial modulator is approximately identity (the static
            # FiLM baseline).
            nn.init.zeros_(self.modulator[-1].weight)
            with torch.no_grad():
                bias = self.modulator[-1].bias
                # First half: γ initialized to 1.
                bias[:feature_channels].fill_(1.0)
                # Second half: β initialized to 0.
                bias[feature_channels:].zero_()

        def forward(self, pose_delta: Any) -> "tuple[Any, Any]":
            """Compute (γ, β) per-pair from pose_delta.

            Parameters
            ----------
            pose_delta
                ``(B, pose_dim)`` per-pair pose vector.

            Returns
            -------
            (γ, β)
                Each shape ``(B, feature_channels)``.
            """
            if pose_delta.dim() != 2 or pose_delta.shape[1] != pose_dim:
                raise TimeVaryingFiLMError(
                    f"pose_delta shape must be (B, {pose_dim}); "
                    f"got {tuple(pose_delta.shape)}"
                )
            modulators = self.modulator(pose_delta)
            gamma = modulators[:, :feature_channels]
            beta = modulators[:, feature_channels:]
            return gamma, beta

        def apply_film(
            self, features: Any, pose_delta: Any
        ) -> Any:
            """Apply per-pair γ⊙f + β to a feature map.

            Parameters
            ----------
            features
                ``(B, C, H, W)`` feature map. ``C`` must equal
                ``feature_channels``.
            pose_delta
                ``(B, pose_dim)`` per-pair pose vector.

            Returns
            -------
            ``(B, C, H, W)`` modulated feature map.
            """
            if features.dim() != 4:
                raise TimeVaryingFiLMError(
                    f"features must be 4D (B, C, H, W); got "
                    f"{features.dim()}D ({tuple(features.shape)})"
                )
            if features.shape[1] != feature_channels:
                raise TimeVaryingFiLMError(
                    f"features channel dim {features.shape[1]} != "
                    f"config.feature_channels {feature_channels}"
                )
            if features.shape[0] != pose_delta.shape[0]:
                raise TimeVaryingFiLMError(
                    f"batch dims must match: features {features.shape[0]} "
                    f"vs pose_delta {pose_delta.shape[0]}"
                )
            gamma, beta = self.forward(pose_delta)
            # Broadcast (B, C) → (B, C, 1, 1).
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            return features * gamma + beta

    return _TimeVaryingFiLMModule()


def compute_per_pair_gamma_beta(
    modulator: Any, pose_deltas: Any
) -> "tuple[Any, Any]":
    """Pure-helper: invoke a TimeVaryingFiLM modulator on a batch of pose deltas.

    Equivalent to ``modulator(pose_deltas)`` but exported as a public
    name so external callers (Phase 2 trainer, codec_pipeline_*
    callbacks) have a stable surface that does not depend on the
    private inner-class.
    """
    return modulator(pose_deltas)


def time_varying_film_state_bytes(config: TimeVaryingFiLMConfig) -> int:
    """Closed-form: archive byte cost of the modulator MLP at given quantization.

    Parameter count:
        layer 1: pose_dim * hidden_dim + hidden_dim
        layer 2: hidden_dim * (2 * feature_channels) + (2 * feature_channels)

    Quantization byte cost: ``params * bytes_per_param`` with
    ``bytes_per_param`` from ``_BYTES_PER_PARAM_BY_QUANT``.

    Returns int (rounded up to whole bytes).
    """
    params_layer1 = config.pose_dim * config.hidden_dim + config.hidden_dim
    params_layer2 = (
        config.hidden_dim * (2 * config.feature_channels)
        + (2 * config.feature_channels)
    )
    total_params = params_layer1 + params_layer2
    bytes_per_param = _BYTES_PER_PARAM_BY_QUANT[config.quantization]
    return int(math.ceil(total_params * bytes_per_param))
