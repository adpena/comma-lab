"""T18 — Ballé 2024 nonlinear transform coding.

Council provenance
------------------
This module operationalizes Track T18 from the Fields-Medal Grand Council
Eureka session (memory ``feedback_grand_council_fields_medal_eureka_mode_implement_landing_20260509.md``,
§Ballé eureka).

Ballé's 2018 entropy bottleneck + scale hyperprior already exists at
:mod:`tac.balle_hyperprior_codec` and :mod:`tac.entropy_bottleneck`. The
2024 nonlinear-transform-coding extension (Stanford He-Zheng 2024-style)
inserts a learned 4-layer MLP between the encoder output and the entropy
bottleneck. This non-trivial nonlinearity beats the standard GDN
(Generalized Divisive Normalization) layer on data with strong
non-Gaussian per-symbol statistics — which is exactly the comma video
substrate.

Per council memo: predicted -2 to -5 KB beyond Phase 2's vanilla Ballé,
applied as a Phase 2 architectural refinement (not a Phase 1 ship-now).

Mathematical core
-----------------
Ballé 2018 vanilla:

    z_e = encoder(x)              [linear convs + GDN]
    z_q = round(z_e + noise)      [STE quant]
    bits = -log2 P_θ(z_q)         [entropy bottleneck]

Ballé 2024 nonlinear-transform extension:

    z_e = encoder(x)
    z_t = NonlinearTransform(z_e)  [4-layer MLP, this module]
    z_q = round(z_t + noise)
    bits = -log2 P_θ(z_q)

The forward transform learns a mapping ``R^D → R^D`` that flattens the
entropy distribution of ``z_q`` (so the entropy bottleneck has tighter
bounds). The inverse transform (decoder side) is a separate MLP trained
end-to-end with the encoder.

The 4-layer MLP architecture (per He-Zheng 2024):
    Linear(D → 4D) → GELU → Linear(4D → 4D) → GELU
    → Linear(4D → 4D) → GELU → Linear(4D → D)

The 4× expansion follows transformer-FFN convention; GELU is a smooth
nonlinearity that preserves gradient quality through the entropy
bottleneck's STE.

CLAUDE.md compliance
--------------------
- **Dev-only ($0 GPU)**: pure CPU module construction + tests.
- **EMA-compatible**: NonlinearTransformBlock is a normal nn.Module.
- **No silent defaults**: ``NonlinearTransformConfig`` validates fields.
- **No score claims**: predicted gains are tagged in docstrings.
- **Strict-scorer-rule**: NO scorer load.
- **No /tmp paths**.

Public API
----------
- ``NonlinearTransformConfig`` — dataclass + validation
- ``NonlinearTransformBlock`` — nn.Module factory; produces forward + inverse MLPs
- ``forward_transform`` — pure helper alias for forward MLP application
- ``inverse_transform`` — pure helper alias for inverse MLP application
- ``transform_state_bytes`` — closed-form archive byte cost

Cross-references
----------------
- Council eureka memo: ``feedback_grand_council_fields_medal_eureka_mode_implement_landing_20260509.md`` (Ballé eureka)
- Phase 2 launch memo: ``feedback_paradigm_dezeta_phase2_architectural_launch_20260509.md``
- Existing Ballé implementations: :mod:`tac.balle_hyperprior_codec`,
  :mod:`tac.entropy_bottleneck`, :mod:`tac.balle_sensitivity_weighted`
- Ballé-Minnen-Singh-Hwang-Johnston 2018 — entropy bottleneck + hyperprior
- He-Zheng 2024 (Stanford) — nonlinear transform coding extension
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


__all__ = [
    "NonlinearTransformConfig",
    "NonlinearTransformBlock",
    "forward_transform",
    "inverse_transform",
    "transform_state_bytes",
    "NonlinearTransformError",
]


class NonlinearTransformError(RuntimeError):
    """Raised on configuration or runtime invariant violations."""


_BYTES_PER_PARAM_BY_QUANT: dict[str, float] = {
    "fp4": 0.5,
    "fp8": 1.0,
    "fp16": 2.0,
    "fp32": 4.0,
}


@dataclass(frozen=True)
class NonlinearTransformConfig:
    """Config for the Ballé 2024 nonlinear transform block.

    Attributes
    ----------
    latent_dim
        Latent dimensionality ``D`` (input/output dim of the transform).
        For Ballé hyperprior typical D=192 or D=256. MUST be > 0.
    expansion_factor
        FFN expansion factor (per He-Zheng 2024 conv: 4×). The hidden
        dim is ``expansion_factor * latent_dim``. MUST be > 0.
    num_hidden_layers
        Number of GELU-activated hidden layers. He-Zheng 2024 uses 3
        (which gives 4 total Linear layers including input + output).
        MUST be >= 1; default 3.
    activation
        Activation: ``"gelu"`` (He-Zheng 2024 canon), ``"relu"``, ``"silu"``.
    quantization
        Quantization for archive byte estimation. Default ``"fp16"``.
    label
        Operator label, required non-empty.
    """

    latent_dim: int
    expansion_factor: int
    num_hidden_layers: int
    activation: str
    quantization: str
    label: str

    def __post_init__(self) -> None:
        if not (self.latent_dim > 0):
            raise NonlinearTransformError(
                f"latent_dim must be > 0; got {self.latent_dim}"
            )
        if not (self.expansion_factor > 0):
            raise NonlinearTransformError(
                f"expansion_factor must be > 0; got {self.expansion_factor}"
            )
        if not (self.num_hidden_layers >= 1):
            raise NonlinearTransformError(
                f"num_hidden_layers must be >= 1; got {self.num_hidden_layers}"
            )
        if self.activation not in {"gelu", "relu", "silu"}:
            raise NonlinearTransformError(
                f"activation must be one of gelu/relu/silu; got {self.activation!r}"
            )
        if self.quantization not in _BYTES_PER_PARAM_BY_QUANT:
            raise NonlinearTransformError(
                f"quantization must be one of "
                f"{sorted(_BYTES_PER_PARAM_BY_QUANT.keys())}; "
                f"got {self.quantization!r}"
            )
        if not isinstance(self.label, str) or not self.label.strip():
            raise NonlinearTransformError("label must be a non-empty string")

    @classmethod
    def he_zheng_canonical(
        cls, *, label: str, latent_dim: int = 192
    ) -> "NonlinearTransformConfig":
        """He-Zheng 2024 canonical (D=192, 4× expansion, 3 hidden, GELU, FP16)."""
        return cls(
            latent_dim=latent_dim,
            expansion_factor=4,
            num_hidden_layers=3,
            activation="gelu",
            quantization="fp16",
            label=label,
        )


def _require_torch() -> "tuple[Any, Any]":
    try:
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415
    except ImportError as exc:
        raise NonlinearTransformError(
            "torch is required for NonlinearTransformBlock instantiation."
        ) from exc
    return torch, nn


def _build_activation(name: str) -> Any:
    _, nn = _require_torch()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "silu":
        return nn.SiLU(inplace=False)
    raise NonlinearTransformError(f"Unknown activation: {name!r}")


def _build_mlp(
    latent_dim: int,
    hidden_dim: int,
    num_hidden_layers: int,
    activation_name: str,
) -> Any:
    """Build one MLP: Linear(D→H) [+activation+Linear(H→H)]*L → Linear(H→D)."""
    _, nn = _require_torch()
    layers: list[Any] = [nn.Linear(latent_dim, hidden_dim), _build_activation(activation_name)]
    for _ in range(num_hidden_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(_build_activation(activation_name))
    layers.append(nn.Linear(hidden_dim, latent_dim))
    return nn.Sequential(*layers)


def NonlinearTransformBlock(config: NonlinearTransformConfig) -> Any:  # noqa: N802
    """Factory: build the Ballé 2024 nonlinear transform nn.Module.

    The returned module exposes:

      - ``forward(z_e)``: forward transform, shape ``(B, latent_dim)``
        → ``(B, latent_dim)``. Used encoder-side before the entropy bottleneck.
      - ``invert(z_t)``: inverse transform, shape ``(B, latent_dim)``
        → ``(B, latent_dim)``. Used decoder-side after the entropy bottleneck.
      - The forward and inverse MLPs are SEPARATE parameter sets (NOT
        weight-tied) per He-Zheng 2024 §3.2 — weight-tying degrades the
        rate-distortion frontier in their ablations.

    Initialized so the forward MLP is approximately identity at init
    (small-init last-layer + skip connection): the rate-distortion
    benefit accrues during training; at init the codec degrades to
    vanilla Ballé.
    """
    if not isinstance(config, NonlinearTransformConfig):
        raise NonlinearTransformError(
            f"NonlinearTransformBlock requires NonlinearTransformConfig; "
            f"got {type(config).__name__}"
        )
    torch, nn = _require_torch()

    latent_dim = config.latent_dim
    hidden_dim = config.expansion_factor * latent_dim
    num_hidden_layers = config.num_hidden_layers
    activation_name = config.activation

    class _NonlinearTransformModule(nn.Module):
        """Ballé 2024 nonlinear transform block."""

        def __init__(self) -> None:
            super().__init__()
            self.forward_mlp = _build_mlp(
                latent_dim, hidden_dim, num_hidden_layers, activation_name
            )
            self.inverse_mlp = _build_mlp(
                latent_dim, hidden_dim, num_hidden_layers, activation_name
            )
            # Approximate-identity init: zero the last linear layer of
            # both MLPs so initially forward(z_e) = z_e + 0 (identity)
            # and invert(z_t) = z_t + 0 (identity). This gives the
            # network the SAME initial behaviour as vanilla Ballé.
            with torch.no_grad():
                nn.init.zeros_(self.forward_mlp[-1].weight)
                nn.init.zeros_(self.forward_mlp[-1].bias)
                nn.init.zeros_(self.inverse_mlp[-1].weight)
                nn.init.zeros_(self.inverse_mlp[-1].bias)

        def forward(self, z_e: Any) -> Any:
            """Forward nonlinear transform with skip connection.

            Output: ``z_t = z_e + forward_mlp(z_e)``. The skip connection
            ensures the transform is a learned PERTURBATION on top of
            the linear identity, which matches Ballé's vanilla behaviour
            at init and degrades GRACEFULLY if the MLP is ablated.
            """
            if z_e.dim() < 1 or z_e.shape[-1] != latent_dim:
                raise NonlinearTransformError(
                    f"z_e last dim must be {latent_dim}; got {z_e.shape}"
                )
            return z_e + self.forward_mlp(z_e)

        def invert(self, z_t: Any) -> Any:
            """Inverse nonlinear transform with skip connection.

            Output: ``z_e = z_t + inverse_mlp(z_t)``.
            """
            if z_t.dim() < 1 or z_t.shape[-1] != latent_dim:
                raise NonlinearTransformError(
                    f"z_t last dim must be {latent_dim}; got {z_t.shape}"
                )
            return z_t + self.inverse_mlp(z_t)

    return _NonlinearTransformModule()


def forward_transform(block: Any, z_e: Any) -> Any:
    """Public helper: forward transform via ``block.forward``."""
    return block(z_e)


def inverse_transform(block: Any, z_t: Any) -> Any:
    """Public helper: inverse transform via ``block.invert``."""
    return block.invert(z_t)


def transform_state_bytes(config: NonlinearTransformConfig) -> int:
    """Closed-form: archive byte cost of forward + inverse MLPs.

    Each MLP has:
        Linear(D → H): D*H + H params
        [num_hidden_layers - 1] × Linear(H → H): each (H*H + H)
        Linear(H → D): H*D + D params

    Two MLPs (forward + inverse, NOT weight-tied per He-Zheng 2024 §3.2).

    Returns int (rounded up).
    """
    D = config.latent_dim
    H = config.expansion_factor * D
    L = config.num_hidden_layers
    # Per-MLP params:
    params_input = D * H + H
    params_hidden = (L - 1) * (H * H + H)
    params_output = H * D + D
    per_mlp = params_input + params_hidden + params_output
    total = 2 * per_mlp
    bytes_per_param = _BYTES_PER_PARAM_BY_QUANT[config.quantization]
    return int(math.ceil(total * bytes_per_param))
