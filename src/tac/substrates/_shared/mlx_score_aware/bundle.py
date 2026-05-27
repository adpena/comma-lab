# SPDX-License-Identifier: MIT
"""The substrate-specific axis passed into the MLX score-aware harness.

Separation of concerns: this module owns ONLY the canonical-vs-unique BOUNDARY
contract (Catalog #290). Everything in :class:`RendererBundle` is the
substrate's UNIQUE axis; the rest of the harness package (device gate / loss /
adapter / portability / orchestrator) is substrate-AGNOSTIC.

A substrate satisfies the harness by passing a ``RendererBundle`` describing
its MLX renderer + real-video targets + (optional) extra-loss callback +
(optional) distillation weight. The ``MlxRenderer`` Protocol documents the two
canonical forward conventions a renderer may expose.

[verified-against: tac.substrates.dreamer_v3_rssm.module.DreamerV3RSSMSubstrateMLX call_b2chw_255 reference]
[verified-against: tac.substrates.atw_v2_cooperative_receiver_v2.mlx_renderer reconstruct_pair_nchw01 reference]
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from tac.substrates._shared.mlx_score_aware.device_gate import (
    MlxScoreAwareHarnessError,
)

#: The two canonical renderer forward conventions the harness auto-detects.
FORWARD_CONVENTIONS: frozenset[str] = frozenset(
    {"reconstruct_pair_nchw01", "call_b2chw_255"}
)


@runtime_checkable
class MlxRenderer(Protocol):
    """Documentation Protocol for a harness-compatible MLX renderer.

    A renderer MUST be differentiable by MLX ``value_and_grad`` — i.e. it is an
    ``mlx.nn.Module`` (or exposes ``.parameters()`` + ``.update()``) — AND
    expose ONE of the two canonical forwards (selected via the bundle's
    ``forward_convention``):

    * ``reconstruct_pair(idx) -> (rgb_0, rgb_1)`` each ``(B, 3, H, W)`` in
      ``[0, 1]`` (the Z6 / atw_v2 / faiss / coin_pp convention), OR
    * ``__call__(idx) -> (B, 2, 3, H, W)`` in ``[0, 255]`` (the dreamer / z8
      HNeRV convention).

    This Protocol is ``runtime_checkable`` for ``.parameters()`` presence only;
    the forward method name varies by convention so it is validated at call
    time by the loss module rather than by ``isinstance``.
    """

    def parameters(self) -> Any: ...


@dataclass
class RendererBundle:
    """Substrate-specific renderer + targets + optional extra-loss callback.

    The canonical-vs-unique boundary: everything in this bundle is the
    substrate's UNIQUE axis; the harness owns everything else (AGNOSTIC).

    Attributes:
        model: the MLX renderer. MUST be an ``mlx.nn.Module`` (or expose
            ``.parameters()`` + ``.update()`` so MLX ``value_and_grad`` can
            differentiate it) AND expose ONE of:
              * ``reconstruct_pair(idx) -> (rgb_0, rgb_1)`` with each
                ``(B, 3, H, W)`` in ``[0, 1]`` (the Z6 / atw_v2 convention), OR
              * ``__call__(idx) -> (B, 2, 3, H, W)`` in ``[0, 255]`` (the
                dreamer / z8 HNeRV convention).
            The harness auto-detects the convention via ``forward_convention``.
        target_rgb_0: MLX float32 ``(num_pairs, H, W, 3)`` in ``[0, 1]``.
        target_rgb_1: MLX float32 ``(num_pairs, H, W, 3)`` in ``[0, 1]``.
        num_pairs: total trainable pair count.
        forward_convention: ``"reconstruct_pair_nchw01"`` (model returns
            ``(rgb_0, rgb_1)`` NCHW in ``[0, 1]``) or ``"call_b2chw_255"``
            (model returns ``(B, 2, 3, H, W)`` in ``[0, 255]``).
        extra_loss_terms: optional callback ``(model, idx) -> {name: scalar}``
            for the variant's UNIQUE extra terms (residual L2 / commitment /
            MINE). Each scalar is weighted by ``loss_weights[name]`` (default
            weight from ``extra_loss_weights``). The harness adds them to the
            reconstruction + score-aware terms.
        extra_loss_weights: default Lagrangian weights for ``extra_loss_terms``
            keys.
        distillation_weight: weight ``lambda`` on the gradient-reachable
            Hinton-KL T=2.0 score-aware surrogate term. ``0.0`` disables it
            (pure reconstruction). Default ``0.0`` so a substrate opts INTO the
            scorer surrogate explicitly (the canonical reconstruction-proxy
            posture mirrors the Z6 reference which DEFERS per-axis to the
            PyTorch sister).
        distillation_temperature: Hinton-KL temperature ``T`` (default 2.0).
        distillation_num_classes: SegNet surrogate class count (default 5).
        export_state_dict_fn: optional ``(model, path) -> None`` PyTorch-export
            bridge; threaded into the adapter's ``export_state_dict``.
        export_archive_fn: optional ``(model, output_dir) -> (path, sha, bytes)``
            numpy-portable archive builder; threaded into the adapter's
            ``export_archive``.
    """

    model: Any
    target_rgb_0: Any
    target_rgb_1: Any
    num_pairs: int
    forward_convention: str = "call_b2chw_255"
    extra_loss_terms: Callable[[Any, Any], Mapping[str, Any]] | None = None
    extra_loss_weights: Mapping[str, float] = field(default_factory=dict)
    distillation_weight: float = 0.0
    distillation_temperature: float = 2.0
    distillation_num_classes: int = 5
    export_state_dict_fn: Callable[[Any, Path], None] | None = None
    export_archive_fn: (
        Callable[[Any, Path], tuple[Path, str, int] | None] | None
    ) = None

    def __post_init__(self) -> None:
        if self.forward_convention not in FORWARD_CONVENTIONS:
            raise MlxScoreAwareHarnessError(
                f"forward_convention must be one of {sorted(FORWARD_CONVENTIONS)}; "
                f"got {self.forward_convention!r}"
            )
        if self.num_pairs < 1:
            raise MlxScoreAwareHarnessError(
                f"num_pairs must be >= 1; got {self.num_pairs}"
            )
        if self.distillation_weight < 0.0:
            raise MlxScoreAwareHarnessError(
                f"distillation_weight must be >= 0 (0.0 disables); got "
                f"{self.distillation_weight}"
            )
        if self.distillation_temperature <= 0.0:
            raise MlxScoreAwareHarnessError(
                f"distillation_temperature must be > 0; got "
                f"{self.distillation_temperature}"
            )
        if self.distillation_num_classes < 1:
            raise MlxScoreAwareHarnessError(
                f"distillation_num_classes must be >= 1; got "
                f"{self.distillation_num_classes}"
            )


__all__ = [
    "FORWARD_CONVENTIONS",
    "MlxRenderer",
    "RendererBundle",
]
