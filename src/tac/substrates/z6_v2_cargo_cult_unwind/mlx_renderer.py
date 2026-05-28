# SPDX-License-Identifier: MIT
"""z6_v2_cargo_cult_unwind MLX-native renderer — L1 LONG-RUN MLX-LOCAL 2026-05-28.

Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" + the
8th MLX-first standing directive REINFORCED 2026-05-27 ("always prefer MLX
first always"): this module promotes Z6-v2 from L0 SCAFFOLD to L1 LONG-RUN
MLX-LOCAL via the canonical pattern landed by sister PACT-NeRV-IA3 (commit
``9ecc75a2d``) + sister PACT-NeRV-SELECTOR cascade.

Z6-v2 distinguishing primitives (per Catalog #272; from architecture.py):

1. **2-level Rao-Ballard hierarchical FiLM-ego-motion predictor**: depth=3
   FiLM-conditioned blocks (~300K params) organized as level-0 micro
   (first 3 blocks) + level-1 meso (remaining 4 blocks). Direct unwind of
   Z6-v1's single-layer FiLM CARGO-CULT per Rao verbatim critique 2026-05-17.
2. **FoE (focus-of-expansion) ego-motion prior conditioning**: per-pair
   6-dim ego-motion vector (tx, ty, tz, rx, ry, rz) feeds FiLM (γ, β)
   modulation at every block.
3. **Atick-Redlich cooperative-receiver gradient binding**: implemented at
   the score_aware_loss.py surface per Catalog #311 sister Z4 routing.

PyTorch-parity invariants honored (mirrors sister PACT-NeRV pattern):

- **Layer names match** (state_dict-compatible): ``latents`` / ``ego_vecs`` /
  ``latent_embed`` / ``blocks.<i>.dsc.{depthwise,pointwise}`` /
  ``blocks.<i>.film_gen.mlp.<j>`` / ``head_rgb_0`` / ``head_rgb_1``.
- **Weight layout matches PyTorch** at export: Conv2d weights stored as
  ``(out_channels, in_channels, kH, kW)``; Linear weights as
  ``(out_features, in_features)``.
- **Forward semantics match**: latent + ego gather → embed → 7 FiLM-modulated
  DepthSep → sin → PixelShuffle(2) blocks → 1x1 RGB heads → sigmoid;
  outputs RGB at contest camera resolution (384, 512).

INDIVIDUALLY-FRACTAL per UNIQUE-AND-COMPLETE-PER-METHOD: this is Z6-v2's
OWN canonical MLX engineering pass; the FiLM-ego-motion conditioning is
Z6-v2-specific (NOT shared with IA3 γ-only modulation NOR SELECTOR Rice-Golomb
coder NOR SELECTOR-V2 arithmetic coder).
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
    from tac.substrates.z6_v2_cargo_cult_unwind.architecture import Z6V2Config

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


SCHEMA_VERSION = "z6_v2_cargo_cult_unwind_mlx_renderer_v1"
MLX_EVIDENCE_GRADE = "[macOS-MLX research-signal]"


def _require_mlx() -> None:
    if mx is None:
        raise RuntimeError(
            "MLX is not available on this host; the Z6-v2 MLX renderer requires "
            "Apple Silicon with the ``mlx`` package installed. Original import "
            f"error: {_MLX_IMPORT_ERROR!r}"
        )


def _pixel_shuffle_2x_nhwc(x: Any) -> Any:
    """PixelShuffle 2x for NHWC tensors via canonical PR95 helper.

    Delegates to ``tac.local_acceleration.pr95_hnerv_mlx::pixel_shuffle_2x_nhwc``
    (empirically PyTorch-byte-stable per FIX-WAVE-R1 ``e1b101888``; 0.0
    absolute drift per CONSOLIDATE-OP-1 2026-05-26 extraction wave).
    """
    _require_mlx()
    from tac.local_acceleration.pr95_hnerv_mlx import pixel_shuffle_2x_nhwc

    return pixel_shuffle_2x_nhwc(x)


def _bilinear_resize_nhwc(x: Any, target_h: int, target_w: int) -> Any:
    """Bilinear resize NHWC tensor to (target_h, target_w) — sister of PACT-NeRV."""
    _require_mlx()
    src_h, src_w = int(x.shape[1]), int(x.shape[2])
    if src_h == target_h and src_w == target_w:
        return x
    from tac.local_acceleration.pr95_hnerv_mlx import (
        bilinear_resize2x_align_corners_false_nhwc,
    )
    if src_h * 2 == target_h and src_w * 2 == target_w:
        return bilinear_resize2x_align_corners_false_nhwc(x)
    raise NotImplementedError(
        f"_bilinear_resize_nhwc generic resize ({src_h}x{src_w} -> "
        f"{target_h}x{target_w}) not implemented; canonical Z6-v2 forward "
        "uses integer-ratio 2x PixelShuffle(7) at 3x4 -> 384x512."
    )


def _siren_uniform_bound(fan_in: int, w: float) -> float:
    """SIREN init bound = sqrt(6/fan_in) / max(w, 1.0) per PyTorch sister."""
    return math.sqrt(6.0 / max(fan_in, 1)) / max(w, 1.0)


class _SinActMLX:
    """MLX sin(w * x) activation (matches PyTorch sister ``_SinAct``)."""

    def __init__(self, w: float) -> None:
        _require_mlx()
        self.w = float(w)

    def __call__(self, x: Any) -> Any:
        return mx.sin(self.w * x)  # type: ignore[union-attr]


class _DepthSepConvMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX depth-separable conv (depthwise 3x3 + pointwise 1x1)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        _require_mlx()
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
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


class _FiLMGeneratorMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX multi-layer FiLM (γ, β) generator from (latent ⊕ ego_vec).

    Mirrors :class:`tac.substrates.z6_v2_cargo_cult_unwind.architecture._FiLMGenerator`.
    Direct unwind of Z6-v1's single-layer FiLM CARGO-CULT per the depth=3 MLP
    structure (Z6-v2 Candidate 1 spec per design memo).
    """

    def __init__(
        self,
        latent_dim: int,
        ego_dim: int,
        out_ch: int,
        depth: int,
        sin_freq: float,
        hidden_width: int = 24,
    ) -> None:
        _require_mlx()
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1; got {depth}")
        if hidden_width < 1:
            raise ValueError(f"hidden_width must be >= 1; got {hidden_width}")
        self.out_ch = int(out_ch)
        self.depth = int(depth)
        self.sin_freq = float(sin_freq)
        self.hidden_width = int(hidden_width)

        # MLP layers: (depth - 1) hidden Linear(in_dim -> hidden_width) +
        # final Linear(in_dim -> 2 * out_ch).
        layers: list[Any] = []
        in_dim = int(latent_dim + ego_dim)
        hidden = int(hidden_width)
        for _ in range(depth - 1):
            layers.append(nn.Linear(in_dim, hidden))  # type: ignore[union-attr]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 2 * out_ch))  # type: ignore[union-attr]
        self.layers: list[Any] = layers
        # SIREN activation between linear layers (NOT after final).
        self.act = _SinActMLX(sin_freq)

    def __call__(self, latent_plus_ego: Any) -> tuple[Any, Any]:
        h = latent_plus_ego
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.act(h)
        # h shape: (B, 2 * out_ch); split into γ + β.
        gamma_raw = h[..., : self.out_ch]
        beta = h[..., self.out_ch :]
        gamma = 1.0 + gamma_raw  # γ near-identity at init per PyTorch sister
        return gamma, beta


class _FiLMUpBlockMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX FiLM-conditioned DepthSep -> sin -> PixelShuffle(2) (mirrors PyTorch sister)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        sin_freq: float,
        latent_dim: int,
        ego_dim: int,
        film_depth: int,
        film_hidden_width: int = 24,
    ) -> None:
        _require_mlx()
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.w = float(sin_freq)
        # DepthSep outputs out_ch * 4 channels for the PixelShuffle(2) factor.
        self.dsc = _DepthSepConvMLX(in_ch, out_ch * 4)
        self.act = _SinActMLX(sin_freq)
        self.film_gen = _FiLMGeneratorMLX(
            latent_dim=latent_dim,
            ego_dim=ego_dim,
            out_ch=out_ch * 4,
            depth=film_depth,
            sin_freq=sin_freq,
            hidden_width=film_hidden_width,
        )

    def __call__(self, x: Any, latent_plus_ego: Any) -> Any:
        # x: NHWC (B, H, W, C)
        h = self.dsc(x)  # (B, H, W, out_ch * 4)
        gamma, beta = self.film_gen(latent_plus_ego)  # each (B, out_ch * 4)
        # Reshape γ, β to (B, 1, 1, C) for broadcast across H, W (NHWC).
        gamma_bcast = mx.reshape(  # type: ignore[union-attr]
            gamma, (-1, 1, 1, int(gamma.shape[-1]))
        )
        beta_bcast = mx.reshape(  # type: ignore[union-attr]
            beta, (-1, 1, 1, int(beta.shape[-1]))
        )
        h = gamma_bcast * h + beta_bcast
        h = self.act(h)
        return _pixel_shuffle_2x_nhwc(h)


class Z6V2SubstrateMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX-native Z6-v2 substrate (L1 LONG-RUN MLX-LOCAL).

    1:1 architectural mirror of
    :class:`tac.substrates.z6_v2_cargo_cult_unwind.architecture.Z6V2Substrate`.

    Forward path:

    1. Latent (per-pair) + ego_vec (per-pair) gather, concatenation.
    2. Latent → initial spatial grid (NHWC).
    3. For each of 7 upsample blocks: DepthSep → FiLM(γ, β) → sin →
       PixelShuffle(2). Blocks 0-2 are level-0 micro (Rao-Ballard); blocks 3-6
       are level-1 meso.
    4. Final 1x1 conv heads produce rgb_0 / rgb_1, then sigmoid.
    5. Stack to (B, 2, 3, H, W) per canonical ``call_b2chw_255`` convention.

    Per HNeRV parity L5: outputs RGB at contest camera resolution (384, 512);
    NOT a mask codec. Per CLAUDE.md "MLX portable-local-substrate authority":
    every artifact is non-promotable ``[macOS-MLX research-signal]``.
    """

    def __init__(self, cfg: "Z6V2Config") -> None:
        _require_mlx()
        super().__init__()
        self.cfg = cfg
        num_pairs = int(cfg.num_pairs)
        latent_dim = int(cfg.latent_dim)
        ego_dim = int(cfg.ego_dim)
        embed_dim = int(cfg.embed_dim)
        initial_grid_h = int(cfg.initial_grid_h)
        initial_grid_w = int(cfg.initial_grid_w)
        num_upsample_blocks = int(cfg.num_upsample_blocks)

        # Per-pair learnable latent + ego-motion vector.
        self.latents = mx.random.normal(  # type: ignore[union-attr]
            shape=(num_pairs, latent_dim)
        ) * 0.02
        self.ego_vecs = mx.random.normal(  # type: ignore[union-attr]
            shape=(num_pairs, ego_dim)
        ) * 0.02

        # Latent -> initial spatial grid embedding.
        self.latent_embed: Any = nn.Linear(  # type: ignore[union-attr]
            latent_dim,
            embed_dim * initial_grid_h * initial_grid_w,
        )

        # Per-block FiLM-conditioned upsample.
        channels = [embed_dim, *list(cfg.decoder_channels)]
        if len(channels) <= num_upsample_blocks:
            raise ValueError(
                f"decoder_channels ({len(cfg.decoder_channels)}) must have at "
                f"least num_upsample_blocks ({num_upsample_blocks}) entries"
            )
        self.blocks: list[Any] = [
            _FiLMUpBlockMLX(
                in_ch=channels[i],
                out_ch=channels[i + 1],
                sin_freq=cfg.sin_frequency,
                latent_dim=latent_dim,
                ego_dim=ego_dim,
                film_depth=int(cfg.film_generator_depth),
                film_hidden_width=int(cfg.film_hidden_width),
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
        """SIREN init for Conv2d + Linear (mirrors PyTorch sister)."""
        w = float(self.cfg.sin_frequency)
        # latent_embed.
        fan_in = int(self.cfg.latent_dim)
        bound = _siren_uniform_bound(fan_in, w)
        self.latent_embed.update({
            "weight": mx.random.uniform(  # type: ignore[union-attr]
                low=-bound, high=bound, shape=self.latent_embed.weight.shape
            ),
            "bias": mx.zeros_like(self.latent_embed.bias),  # type: ignore[union-attr]
        })
        # Per-block Conv2d + FiLM-generator Linear SIREN init.
        for block in self.blocks:
            dsc = block.dsc
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
            p = dsc.pointwise
            in_ch = p.weight.shape[3]  # NHWC (out, kH, kW, in)
            fan_in_p = int(in_ch * 1 * 1)
            bound_p = _siren_uniform_bound(fan_in_p, w)
            p.update({
                "weight": mx.random.uniform(  # type: ignore[union-attr]
                    low=-bound_p, high=bound_p, shape=p.weight.shape
                ),
                "bias": mx.zeros_like(p.bias),  # type: ignore[union-attr]
            })
            # FiLM generator MLP layers.
            for layer in block.film_gen.layers:
                fan_in_lin = int(layer.weight.shape[1])
                bound_lin = _siren_uniform_bound(fan_in_lin, w)
                layer.update({
                    "weight": mx.random.uniform(  # type: ignore[union-attr]
                        low=-bound_lin, high=bound_lin, shape=layer.weight.shape
                    ),
                    "bias": mx.zeros_like(layer.bias),  # type: ignore[union-attr]
                })
        # RGB heads.
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

    def __call__(self, pair_indices: Any) -> Any:
        """Forward returning (B, 2, 3, H, W) in [0, 255] per ``call_b2chw_255``."""
        # Latent + ego_vec gather.
        z = mx.take(self.latents, pair_indices, axis=0)  # type: ignore[union-attr]
        ego = mx.take(self.ego_vecs, pair_indices, axis=0)  # type: ignore[union-attr]
        latent_plus_ego = mx.concatenate([z, ego], axis=-1)  # type: ignore[union-attr]

        # Latent -> initial spatial grid (NHWC).
        h = self.latent_embed(z)
        h = mx.reshape(  # type: ignore[union-attr]
            h,
            (
                -1,
                self.cfg.initial_grid_h,
                self.cfg.initial_grid_w,
                self.cfg.embed_dim,
            ),
        )

        # Per-block FiLM-conditioned forward.
        for block in self.blocks:
            h = block(h, latent_plus_ego)

        h = _bilinear_resize_nhwc(
            h, int(self.cfg.output_height), int(self.cfg.output_width)
        )

        rgb_0_nhwc = mx.sigmoid(self.head_rgb_0(h)) * 255.0  # type: ignore[union-attr]
        rgb_1_nhwc = mx.sigmoid(self.head_rgb_1(h)) * 255.0  # type: ignore[union-attr]
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

        Mirrors sister PACT-NeRV pattern:
        - Conv2d weights: (out, in, kH, kW). MLX NHWC (out, kH, kW, in) → transpose ``(0, 3, 1, 2)``.
        - Linear weights: (out, in). MLX matches.
        - Per-pair latents / ego_vecs: (num_pairs, D).
        - FiLM-generator MLP weights: ``blocks.<i>.film_gen.layers.<j>.{weight,bias}``.
        """
        _require_mlx()
        out: dict[str, np.ndarray] = {}
        out["latents"] = np.asarray(self.latents, dtype=np.float32).copy()
        out["ego_vecs"] = np.asarray(self.ego_vecs, dtype=np.float32).copy()
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
            # FiLM-generator MLP layers. PyTorch sister wraps in nn.Sequential
            # so layer indices are 0, 2, 4, ... (even = Linear; odd = _SinAct).
            # MLX sister stores layers as a list of (depth) Linears + a single
            # _SinActMLX between them; map to PyTorch nn.Sequential indices.
            for j, layer in enumerate(block.film_gen.layers):
                # PyTorch nn.Sequential: Linear at index 2*j; _SinAct at 2*j+1.
                pyt_idx = 2 * j
                out[f"blocks.{i}.film_gen.mlp.{pyt_idx}.weight"] = np.asarray(
                    layer.weight, dtype=np.float32
                ).copy()
                out[f"blocks.{i}.film_gen.mlp.{pyt_idx}.bias"] = np.asarray(
                    layer.bias, dtype=np.float32
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
    "Z6V2SubstrateMLX",
]
