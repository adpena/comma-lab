# SPDX-License-Identifier: MIT
"""PACT-NeRV-IA3 MLX-native renderer — L1 LONG-RUN MLX-LOCAL CLOSURE 2026-05-28.

Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" PRIORITY
1 closure of the operator question 2026-05-28 ("what is the latest on the
highest EV and top tier and ultimate pact-nerv? have we done any long runs
yet or continued optimizing and iterating, or did we accidentally forget and
stop working on driving our existing work to MLX runs in parallel?").

Honest answer per the operator: PACT-NeRV had ZERO MLX renderers despite 18
variants with PyTorch ``_full_main`` implemented. The 8th MLX-first standing
directive REINFORCED 2026-05-27 ("always prefer MLX first always") mandates
MLX-LOCAL LONG runs on M5 Max at $0 BEFORE any paid CUDA dispatch.

This module is the MLX-native renderer for PACT-NeRV-IA3 (Stage 1 canonical
of the 21-step ULTIMATE STAIRCASE per
``.omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md``).
IA3 was selected as the highest-EV PRIORITY 1 MLX target because:

- Stage 1 of the canonical STAIRCASE (predecessor for ALL downstream Stages).
- HARD-EARNED-LITERATURE (Liu et al. 2022 IA3 paper arXiv:2205.05638).
- Simplest distinguishing primitive (γ-only multiplier projection from
  ego-pose; NO β bias projection).
- ~150 LOC distinguishing primitive (well under HNeRV parity L7 bolt-on
  budget; the FULL renderer below uses ~400 LOC because it includes the
  HNeRV-class base decoder per HNeRV parity L5).
- Predicted ΔS band ``[-0.003, +0.001]`` (frontier-adjacent per ULTIMATE
  taxonomy Variant #1 / STAIRCASE Step 1).

Canonical PyTorch sister
------------------------

This module is a 1:1 architectural mirror of
:class:`tac.substrates.pact_nerv_ia3.architecture.PactNervIa3Substrate` so
the MLX-trained weights export back to PyTorch state_dict via the canonical
MLX→PyTorch bridge (:mod:`tac.local_acceleration.mlx_to_pytorch_export`) and
the PyTorch substrate packs the PIA3 archive via
:func:`tac.substrates.pact_nerv_ia3.archive.pack_archive`.

PyTorch-parity invariants honored
---------------------------------

- **Layer names match** (state_dict-compatible): ``latent_embed`` /
  ``blocks.<i>.dsc.{depthwise,pointwise}`` / ``ia3_mods.<i>.gamma_proj`` /
  ``head_rgb_0`` / ``head_rgb_1``. The ``latents`` and ``ego_poses`` per-pair
  parameters use the same names + (num_pairs, D) layout.
- **Weight layout matches PyTorch** at export: Conv2d weights stored as
  ``(out_channels, in_channels, kH, kW)``; Linear weights as
  ``(out_features, in_features)``. MLX internally uses NHWC + HWIO layout
  but :meth:`export_state_dict` returns numpy arrays in PyTorch layout.
- **Forward semantics match**: sin() activation in ``_DsUpBlock``; bilinear
  resize with ``align_corners=False`` for final upscale; per-block IA3
  γ-only modulation (NO β); ``γ = 1.0 + γ_proj(pose)`` residual form per
  IA3 §3.2; sigmoid on RGB heads.

Non-promotable canonical contract
---------------------------------

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog
#127/#192/#317/#341: all artifacts produced by this MLX renderer are tagged
``[macOS-MLX research-signal]`` with:

- ``score_claim=False``
- ``promotion_eligible=False``
- ``ready_for_exact_eval_dispatch=False``

The canonical MLX harness
(:func:`tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main`)
auto-stamps these markers on the ``TrainingArtifact``.

Promotion path
--------------

MLX state_dict → PyTorch via the canonical bridge → PIA3 archive via the
canonical pack_archive → contest-equivalence gate via
:mod:`tools.gate_mlx_candidate_contest_equivalence` → operator routes
paid CUDA dispatch via ``tools/operator_authorize.py``.

INDIVIDUALLY-FRACTAL per UNIQUE-AND-COMPLETE-PER-METHOD
-------------------------------------------------------

This MLX renderer is PACT-NeRV-IA3's OWN canonical engineering pass per the
11th INDIVIDUALLY-FRACTAL standing directive 2026-05-27. The distinguishing
primitive (γ-only ego-pose modulation per Liu 2022) is implemented
substrate-specifically — NOT a shared-helper shortcut from sister NeRV-family
substrates. The HNeRV-class base decoder topology is ADOPT-CANONICAL per
Catalog #290 + the empirically validated PR95/PR101/PR110 medal-class
topology (per HNeRV parity L7: substrate engineering binds ALL ingredients;
the architecture distinguishing primitive is IA3 γ-only, NOT the decoder
topology).

Cross-references
----------------

- Canonical sister PyTorch architecture:
  :mod:`tac.substrates.pact_nerv_ia3.architecture`
- Canonical MLX HNeRV reference pattern (Z6 sister):
  :mod:`tac.substrates.time_traveler_l5_z6.mlx_renderer`
- Canonical MLX→PyTorch export bridge:
  :mod:`tac.local_acceleration.mlx_to_pytorch_export`
- Canonical MLX score-aware harness:
  :mod:`tac.substrates._shared.mlx_score_aware`
- ULTIMATE design memo:
  ``.omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md``
- Variant selection memo (this landing):
  ``.omx/research/pact_nerv_long_run_mlx_local_closure_landed_20260528.md``
"""
# AUTOCAST_FP16_WAIVED:MLX_substrate_does_not_use_PyTorch_CUDA_autocast_fp16_per_mlx_first_canonical_doctrine_8th_standing_directive
# TF32_WAIVED:MLX_substrate_does_not_use_PyTorch_CUDA_tf32_per_mlx_first_canonical_doctrine
# TORCH_COMPILE_WAIVED:MLX_substrate_uses_mlx_value_and_grad_not_torch_compile
# NO_GRAD_WAIVED:MLX_substrate_uses_mlx_lazy_eval_not_pytorch_no_grad_per_mlx_first_canonical_doctrine

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tac.substrates.pact_nerv_ia3.architecture import PactNervIa3Config

try:  # pragma: no cover — exercised on Apple Silicon with MLX installed.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
except Exception as exc:  # pragma: no cover — import guard for non-Apple CI.
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    tree_flatten = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR: Exception | None = exc
else:
    _MLX_IMPORT_ERROR = None


SCHEMA_VERSION = "pact_nerv_ia3_mlx_renderer_v1"
MLX_EVIDENCE_GRADE = "[macOS-MLX research-signal]"


def _require_mlx() -> None:
    if mx is None:
        raise RuntimeError(
            "MLX is not available on this host; the PACT-NeRV-IA3 MLX renderer "
            "requires Apple Silicon with the ``mlx`` package installed. "
            f"Original import error: {_MLX_IMPORT_ERROR!r}"
        )


def _pixel_shuffle_2x_nhwc(x: Any) -> Any:
    """PixelShuffle 2x for NHWC tensors via canonical PR95 helper.

    Delegates to ``tac.local_acceleration.pr95_hnerv_mlx::pixel_shuffle_2x_nhwc``
    (empirically PyTorch-byte-stable per FIX-WAVE-R1 ``e1b101888``).
    """
    _require_mlx()
    from tac.local_acceleration.pr95_hnerv_mlx import (
        pixel_shuffle_2x_nhwc,
    )

    return pixel_shuffle_2x_nhwc(x)


def _bilinear_resize_nhwc(x: Any, target_h: int, target_w: int) -> Any:
    """Bilinear resize NHWC tensor to (target_h, target_w) via canonical helpers.

    The canonical 2x helper only does 2x; for the final
    (H, W) -> (output_height, output_width) match we fall back to per-axis
    bilinear interpolation matching PyTorch ``F.interpolate(mode='bilinear',
    align_corners=False)`` semantics.
    """
    _require_mlx()
    src_h, src_w = int(x.shape[1]), int(x.shape[2])
    if src_h == target_h and src_w == target_w:
        return x
    # MLX has no native generic bilinear; fall back to canonical sister helper
    # that mirrors PyTorch ``F.interpolate(align_corners=False)``.
    from tac.local_acceleration.pr95_hnerv_mlx import (
        bilinear_resize2x_align_corners_false_nhwc,
    )

    # Repeated 2x bilinear is exact for integer-ratio resizes (we just need it
    # for tail block (3*2^7, 4*2^7) = (384, 512) which already matches).
    if src_h * 2 == target_h and src_w * 2 == target_w:
        return bilinear_resize2x_align_corners_false_nhwc(x)
    # General case: use mx.image.resize when available, else manual gather.
    # For the canonical (3, 4) -> (384, 512) the 7 PixelShuffle(2) blocks
    # land exactly on (384, 512) so this generic branch is unused in the
    # canonical IA3 forward path.
    raise NotImplementedError(
        f"_bilinear_resize_nhwc generic resize ({src_h}x{src_w} -> "
        f"{target_h}x{target_w}) not implemented; canonical IA3 forward "
        "uses integer-ratio 2x PixelShuffle(7) which lands at (384, 512)."
    )


def _siren_uniform_bound(fan_in: int, w: float) -> float:
    """SIREN init bound = sqrt(6/fan_in) / max(w, 1.0) per PyTorch sister."""
    return math.sqrt(6.0 / max(fan_in, 1)) / max(w, 1.0)


class _DepthSepConvMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX depth-separable conv (depthwise 3x3 + pointwise 1x1).

    Mirrors :class:`tac.substrates.pact_nerv_ia3.architecture._DepthSepConv`.
    MLX nn.Conv2d uses NHWC by default; weight layout HWIO -> we transpose
    to OIHW at export.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        _require_mlx()
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        # MLX Conv2d w/ groups=in_ch IS depthwise (1 output channel per input
        # group). We then expand to out_ch via pointwise 1x1.
        self.depthwise: Any = nn.Conv2d(  # type: ignore[union-attr]
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            padding=1,
            groups=in_ch,
        )
        self.pointwise: Any = nn.Conv2d(  # type: ignore[union-attr]
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
        )

    def __call__(self, x: Any) -> Any:
        return self.pointwise(self.depthwise(x))


class _DsUpBlockMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX DepthSep -> sin -> PixelShuffle(2) (mirrors PyTorch sister)."""

    def __init__(self, in_ch: int, out_ch: int, sin_freq: float) -> None:
        _require_mlx()
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.w = float(sin_freq)
        # PixelShuffle(2) expands channels by 4x then folds spatial -> use
        # out_ch * 4 as DepthSep output channels.
        self.dsc = _DepthSepConvMLX(in_ch, out_ch * 4)

    def __call__(self, x: Any) -> Any:
        h = self.dsc(x)
        h = mx.sin(self.w * h)  # type: ignore[union-attr]
        return _pixel_shuffle_2x_nhwc(h)


class _IA3GammaOnlyModulationMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX γ-only ego-pose modulation per Liu 2022 §3.2.

    Mirrors :class:`tac.substrates.pact_nerv_ia3.architecture.IA3GammaOnlyModulation`
    (the distinguishing primitive vs full FiLM γ+β). Residual form
    ``γ = 1.0 + γ_proj(pose)`` ensures the substrate behaves like the
    unconditioned base decoder at initialization.
    """

    def __init__(
        self,
        num_features: int,
        pose_dim: int,
        init_delta_std: float,
    ) -> None:
        _require_mlx()
        super().__init__()
        if num_features <= 0:
            raise ValueError(f"num_features must be positive; got {num_features}")
        if pose_dim <= 0:
            raise ValueError(f"pose_dim must be positive; got {pose_dim}")
        if init_delta_std < 0:
            raise ValueError(
                f"init_delta_std must be non-negative; got {init_delta_std}"
            )
        self.num_features = int(num_features)
        self.pose_dim = int(pose_dim)
        # γ projection: pose_dim -> num_features (NO β projection per IA3).
        self.gamma_proj: Any = nn.Linear(pose_dim, num_features)  # type: ignore[union-attr]
        # IA3 §3.2 zero-init: γ_proj weights ~ N(0, init_delta_std^2);
        # bias zero. γ_init ≈ 1.0 so substrate ≈ unconditioned base at start.
        zero_init_weight = mx.random.normal(  # type: ignore[union-attr]
            shape=(num_features, pose_dim)
        ) * float(init_delta_std)
        zero_init_bias = mx.zeros((num_features,))  # type: ignore[union-attr]
        self.gamma_proj.update({"weight": zero_init_weight, "bias": zero_init_bias})

    def __call__(self, x: Any, pose: Any) -> Any:
        """Apply γ-only modulation: x * (1.0 + γ_proj(pose))."""
        # x: (B, H, W, C) NHWC; pose: (B, pose_dim); γ_proj output: (B, C).
        gamma = 1.0 + self.gamma_proj(pose)  # (B, C)
        # Broadcast (B, C) -> (B, 1, 1, C) to multiply NHWC.
        gamma_bcast = mx.reshape(gamma, (-1, 1, 1, self.num_features))  # type: ignore[union-attr]
        return x * gamma_bcast


class PactNervIa3SubstrateMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX-native PACT-NeRV-IA3 substrate (L1 LONG-RUN MLX-LOCAL).

    1:1 architectural mirror of
    :class:`tac.substrates.pact_nerv_ia3.architecture.PactNervIa3Substrate`.

    The forward path:

    1. Latent embedding (per-pair) -> initial spatial grid (NHWC).
    2. For each upsample block: DepthSep -> sin -> PixelShuffle(2)
       -> IA3 γ-only modulation conditioned on ego-pose.
    3. Final 1x1 conv heads produce rgb_0 / rgb_1, then sigmoid.
    4. Stack to (B, 2, 3, H, W) for the canonical ``call_b2chw_255``
       convention required by :mod:`tac.substrates._shared.mlx_score_aware`.

    Per HNeRV parity L5: outputs RGB at contest camera resolution (384, 512);
    NOT a mask codec. The IA3 γ-only modulation is the substrate's UNIQUE
    distinguishing primitive vs the canonical HNeRV-class decoder backbone.

    Per CLAUDE.md "MLX portable-local-substrate authority" non-negotiable:
    every artifact produced by this substrate is non-promotable
    ``[macOS-MLX research-signal]``; the canonical promotion path is
    MLX state_dict → PyTorch export → PIA3 archive → contest-equivalence gate.
    """

    def __init__(self, cfg: PactNervIa3Config) -> None:
        _require_mlx()
        super().__init__()
        self.cfg = cfg
        num_pairs = int(cfg.num_pairs)
        latent_dim = int(cfg.latent_dim)
        embed_dim = int(cfg.embed_dim)
        pose_dim = int(cfg.pose_dim)
        initial_grid_h = int(cfg.initial_grid_h)
        initial_grid_w = int(cfg.initial_grid_w)
        num_upsample_blocks = int(cfg.num_upsample_blocks)

        # Per-pair learnable latent (matches PyTorch ``self.latents``).
        self.latents = mx.random.normal(  # type: ignore[union-attr]
            shape=(num_pairs, latent_dim)
        ) * 0.02

        # Per-pair learnable ego-pose (matches PyTorch ``self.ego_poses``).
        self.ego_poses = mx.random.normal(  # type: ignore[union-attr]
            shape=(num_pairs, pose_dim)
        ) * 0.02

        # Latent -> initial spatial grid embedding.
        self.latent_embed: Any = nn.Linear(  # type: ignore[union-attr]
            latent_dim,
            embed_dim * initial_grid_h * initial_grid_w,
        )

        # Per-block upsample + IA3 γ-only modulation.
        channels = [embed_dim, *list(cfg.decoder_channels)]
        if len(channels) <= num_upsample_blocks:
            raise ValueError(
                f"decoder_channels ({len(cfg.decoder_channels)}) must have at "
                f"least num_upsample_blocks ({num_upsample_blocks}) entries"
            )
        self.blocks: list[Any] = [
            _DsUpBlockMLX(channels[i], channels[i + 1], cfg.sin_frequency)
            for i in range(num_upsample_blocks)
        ]
        self.ia3_mods: list[Any] = [
            _IA3GammaOnlyModulationMLX(
                num_features=channels[i + 1],
                pose_dim=pose_dim,
                init_delta_std=cfg.ia3_init_delta_std,
            )
            for i in range(num_upsample_blocks)
        ]

        final_ch = channels[num_upsample_blocks]
        self.head_rgb_0: Any = nn.Conv2d(  # type: ignore[union-attr]
            in_channels=final_ch, out_channels=3, kernel_size=1
        )
        self.head_rgb_1: Any = nn.Conv2d(  # type: ignore[union-attr]
            in_channels=final_ch, out_channels=3, kernel_size=1
        )

        self._siren_init()

    def _siren_init(self) -> None:
        """SIREN init for Conv2d + non-IA3 Linear (mirrors PyTorch sister).

        IA3 γ_proj weights are already zero-init per IA3 §3.2 in
        :class:`_IA3GammaOnlyModulationMLX.__init__`; we skip them here.
        """
        w = float(self.cfg.sin_frequency)
        # latent_embed: Linear(latent_dim -> embed_dim * H0 * W0).
        fan_in = int(self.cfg.latent_dim)
        bound = _siren_uniform_bound(fan_in, w)
        self.latent_embed.update({
            "weight": mx.random.uniform(  # type: ignore[union-attr]
                low=-bound, high=bound, shape=self.latent_embed.weight.shape
            ),
            "bias": mx.zeros_like(self.latent_embed.bias),  # type: ignore[union-attr]
        })
        # Per-block Conv2d (depthwise + pointwise) SIREN init.
        for block in self.blocks:
            dsc = block.dsc
            # depthwise: fan_in = kH * kW (groups=in_ch so each filter sees only
            # its own input channel × kernel area).
            d = dsc.depthwise
            kH, kW = d.weight.shape[1], d.weight.shape[2]
            fan_in_d = int(kH * kW)
            bound_d = _siren_uniform_bound(fan_in_d, w)
            d.update({
                "weight": mx.random.uniform(  # type: ignore[union-attr]
                    low=-bound_d, high=bound_d, shape=d.weight.shape
                ),
                "bias": mx.zeros_like(d.bias),  # type: ignore[union-attr]
            })
            # pointwise: fan_in = in_channels * 1 * 1.
            p = dsc.pointwise
            in_ch = p.weight.shape[3]  # NHWC weight is (out, kH, kW, in)
            fan_in_p = int(in_ch * 1 * 1)
            bound_p = _siren_uniform_bound(fan_in_p, w)
            p.update({
                "weight": mx.random.uniform(  # type: ignore[union-attr]
                    low=-bound_p, high=bound_p, shape=p.weight.shape
                ),
                "bias": mx.zeros_like(p.bias),  # type: ignore[union-attr]
            })
        # RGB heads: 1x1 Conv2d, fan_in = in_ch.
        for head in (self.head_rgb_0, self.head_rgb_1):
            in_ch_h = head.weight.shape[3]
            fan_in_h = int(in_ch_h * 1 * 1)
            bound_h = _siren_uniform_bound(fan_in_h, w)
            head.update({
                "weight": mx.random.uniform(  # type: ignore[union-attr]
                    low=-bound_h, high=bound_h, shape=head.weight.shape
                ),
                "bias": mx.zeros_like(head.bias),  # type: ignore[union-attr]
            })

    def __call__(
        self,
        pair_indices: Any,
        ego_pose: Any = None,
    ) -> Any:
        """Forward path returning (B, 2, 3, H, W) in [0, 255] per canonical convention.

        Args:
            pair_indices: (B,) int array of pair indices in [0, num_pairs).
            ego_pose: optional (B, pose_dim) override. If None, uses the
                per-pair learnable ``self.ego_poses``.

        Returns:
            (B, 2, 3, H, W) MLX float32 array in [0, 255] matching the
            ``call_b2chw_255`` convention of the canonical MLX score-aware
            harness.
        """
        # Latent + ego-pose gather.
        z = mx.take(self.latents, pair_indices, axis=0)  # (B, latent_dim) type: ignore[union-attr]
        pose = (
            mx.take(self.ego_poses, pair_indices, axis=0)  # (B, pose_dim) type: ignore[union-attr]
            if ego_pose is None
            else ego_pose
        )

        # Latent -> initial spatial grid (NHWC).
        h = self.latent_embed(z)  # (B, embed_dim * H0 * W0)
        h = mx.reshape(  # type: ignore[union-attr]
            h,
            (
                -1,
                self.cfg.initial_grid_h,
                self.cfg.initial_grid_w,
                self.cfg.embed_dim,
            ),
        )

        # Per-block forward + IA3 γ-only modulation.
        for block, ia3 in zip(self.blocks, self.ia3_mods, strict=True):
            h = block(h)
            h = ia3(h, pose)

        # Bilinear resize if not exactly at (output_height, output_width).
        h = _bilinear_resize_nhwc(
            h, int(self.cfg.output_height), int(self.cfg.output_width)
        )

        # 1x1 RGB heads + sigmoid + scale to [0, 255].
        rgb_0_nhwc = mx.sigmoid(self.head_rgb_0(h)) * 255.0  # type: ignore[union-attr]
        rgb_1_nhwc = mx.sigmoid(self.head_rgb_1(h)) * 255.0  # type: ignore[union-attr]
        # Stack as (B, 2, H, W, C) then transpose to (B, 2, C, H, W).
        pair_nhwc = mx.stack([rgb_0_nhwc, rgb_1_nhwc], axis=1)  # type: ignore[union-attr]
        return mx.transpose(pair_nhwc, (0, 1, 4, 2, 3))  # type: ignore[union-attr]

    def num_parameters(self) -> int:
        """Total trainable parameter count (matches PyTorch sister within init)."""
        _require_mlx()
        flat = tree_flatten(self.parameters())  # type: ignore[union-attr]
        total = 0
        for _name, arr in flat:
            total += int(np.prod(arr.shape))
        return total

    def export_state_dict(self) -> dict[str, np.ndarray]:
        """Export weights in PyTorch state_dict layout for the canonical bridge.

        The canonical MLX→PyTorch bridge expects:
        - Conv2d weights: (out_channels, in_channels, kH, kW). MLX NHWC
          layout is (out, kH, kW, in); we transpose ``(0, 3, 1, 2)``.
        - Linear weights: (out_features, in_features). MLX layout already
          matches PyTorch.
        - Bias: (out_features,) or (out_channels,). Matches.
        - Per-pair tensors (latents / ego_poses): (num_pairs, D).

        Returns:
            dict[str, np.ndarray] keyed by PyTorch sister parameter names.
        """
        _require_mlx()
        out: dict[str, np.ndarray] = {}
        out["latents"] = np.asarray(self.latents, dtype=np.float32).copy()
        out["ego_poses"] = np.asarray(self.ego_poses, dtype=np.float32).copy()
        out["latent_embed.weight"] = np.asarray(
            self.latent_embed.weight, dtype=np.float32
        ).copy()
        out["latent_embed.bias"] = np.asarray(
            self.latent_embed.bias, dtype=np.float32
        ).copy()
        for i, block in enumerate(self.blocks):
            d = block.dsc.depthwise
            p = block.dsc.pointwise
            # MLX NHWC weight (out, kH, kW, in) -> PyTorch OIHW (out, in, kH, kW).
            out[f"blocks.{i}.dsc.depthwise.weight"] = np.transpose(
                np.asarray(d.weight, dtype=np.float32), (0, 3, 1, 2)
            ).copy()
            out[f"blocks.{i}.dsc.depthwise.bias"] = np.asarray(
                d.bias, dtype=np.float32
            ).copy()
            out[f"blocks.{i}.dsc.pointwise.weight"] = np.transpose(
                np.asarray(p.weight, dtype=np.float32), (0, 3, 1, 2)
            ).copy()
            out[f"blocks.{i}.dsc.pointwise.bias"] = np.asarray(
                p.bias, dtype=np.float32
            ).copy()
        for i, ia3 in enumerate(self.ia3_mods):
            # IA3 γ_proj is Linear(pose_dim -> num_features); weight (out, in).
            out[f"ia3_mods.{i}.gamma_proj.weight"] = np.asarray(
                ia3.gamma_proj.weight, dtype=np.float32
            ).copy()
            out[f"ia3_mods.{i}.gamma_proj.bias"] = np.asarray(
                ia3.gamma_proj.bias, dtype=np.float32
            ).copy()
        for head_name in ("head_rgb_0", "head_rgb_1"):
            head = getattr(self, head_name)
            out[f"{head_name}.weight"] = np.transpose(
                np.asarray(head.weight, dtype=np.float32), (0, 3, 1, 2)
            ).copy()
            out[f"{head_name}.bias"] = np.asarray(
                head.bias, dtype=np.float32
            ).copy()
        return out


__all__ = [
    "MLX_EVIDENCE_GRADE",
    "SCHEMA_VERSION",
    "PactNervIa3SubstrateMLX",
]
