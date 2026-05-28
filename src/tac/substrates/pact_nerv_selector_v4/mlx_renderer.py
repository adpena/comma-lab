# SPDX-License-Identifier: MIT
"""PACT-NeRV-SELECTOR-V4 MLX-native renderer - L1 LONG-RUN MLX-LOCAL 2026-05-28.

Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" + the
8th MLX-first standing directive REINFORCED 2026-05-27 ("always prefer MLX
first always"): this module is the FOURTH PACT-NeRV variant promoted from
L0 SCAFFOLD to L1 LONG-RUN MLX-LOCAL via the canonical pattern landed by
sister PACT-NeRV-IA3 (commit ``9ecc75a2d``), PACT-NeRV-SELECTOR-V2 (commit
``fee801ac7``), and PACT-NeRV-SELECTOR-V3 (commit ``2f69d0ea6``).

Variant selected per the operator's individually-fractal next-variant
selection criteria in the parent prompt + ULTIMATE design memo STAIRCASE
Step 13 (next after SELECTOR-V3's Step 12):

- (i) Most-canonical "next" per ULTIMATE STAIRCASE Step 13: SELECTOR-V4 is
  PRIORITY 1 per CROSS-CANDIDATE finding #1 empirical headroom + cascade
  continuation after SELECTOR-V3 L1 landed at commit ``2f69d0ea6``.
- (ii) Highest predicted-DeltaS-per-MLX-hour EV: SELECTOR-V4 inherits the
  same FEC6 k=16 palette empirical headroom anchor as SELECTOR-V2/V3 with
  predicted DeltaS [-0.005, +0.001] (HARD-EARNED-EMP per ULTIMATE Variant
  taxonomy table row #13); the L1 MLX research-signal probes whether the
  L0 SCAFFOLD's run-length coder (Robinson-Cherry 1967 + Capon 1959;
  optimal for temporally-coherent runs) achieves rate savings proportional
  to mean run length on the FEC6 selector stream during static scenes.
- (iii) MLX-implementable at L1 ~3-6h: SELECTOR-V4's base HNeRV decoder is
  structurally identical to SELECTOR-V3 (DepthSep + SIREN + PixelShuffle x7);
  the RLE primitive operates at ARCHIVE-ENCODE TIME so the MLX renderer is
  the BASE HNeRV decoder WITHOUT the RLE modulation.
- (iv) DISJOINT from IA3 + SELECTOR-V2 + SELECTOR-V3 at the primitive
  surface: run-length encoding via (value, varint) pairs is a different
  fractional-bit-precision strategy than SELECTOR-V2's arithmetic coding
  (Witten 1987) and SELECTOR-V3's Rice-Golomb unary+binary coding
  (Golomb 1966 + Rice 1971); SELECTOR-V4 is the next SELECTOR-PARADIGM-
  EXTENSIONS family member, ideal for temporally-coherent streams where
  consecutive pairs select the same mode during static scenes.

Reference: ``.omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md``
Step 13 verdict + Variant taxonomy table Variant #13.

Note on ULTIMATE spec vs L0 scaffold ambiguity
---------------------------------------------

The ULTIMATE memo Variant #13 description Section says "per-pair-per-class
arithmetic coder via per_segnet_class_chroma_priors", but the actual L0
SCAFFOLD landed at commit per the G3 design memo
(``.omx/research/pact_nerv_g3_selector_extensions_l0_scaffold_design_20260520T204641Z.md``)
implements a RUN-LENGTH coder per Robinson-Cherry 1967 / Capon 1959 with
varint (LEB128-style) run-length encoding. Both fall within "SELECTOR-
PARADIGM-EXTENSIONS" Group 3. Per Catalog #229 PV (premise verification
before edit) + Catalog #303 cargo-cult audit + CLAUDE.md "UNIQUE-AND-
COMPLETE-PER-METHOD": this L1 LONG-RUN MLX-LOCAL promotion follows the
EXISTING L0 SCAFFOLD (RLE), since the base HNeRV decoder is the MLX
training surface regardless of which selector primitive operates at
archive-encode time. The ULTIMATE-aligned per-pair-per-class arithmetic
coder VARIANT is queued as the operator-routable next step OR a sister
substrate (e.g. PACT-NeRV-SELECTOR-V5 per the V2 -> V3 -> V4 -> V5 cascade
pattern).

Canonical PyTorch sister
------------------------

This module is a 1:1 architectural mirror of
:class:`tac.substrates.pact_nerv_selector_v4.architecture.PactNervSelectorV4Substrate`
so the MLX-trained weights export back to PyTorch state_dict via the canonical
MLX->PyTorch bridge (:mod:`tac.local_acceleration.mlx_to_pytorch_export`) and
the PyTorch substrate packs the PSV4 archive via
:func:`tac.substrates.pact_nerv_selector_v4.archive.pack_archive`.

PyTorch-parity invariants honored
---------------------------------

- **Layer names match** (state_dict-compatible): ``latent_embed`` /
  ``blocks.<i>.dsc.{depthwise,pointwise}`` / ``head_rgb_0`` / ``head_rgb_1``.
  The ``latents`` per-pair parameter uses the same name + (num_pairs, D) layout.
  NO ``ego_poses`` (SELECTOR-V4 has no pose conditioning - the selector
  primitive operates at archive-encode time on FEC6 k=16 palette indices).
  NO ``ia3_mods`` (this is the distinguishing primitive vs IA3 family).
  The non-trainable ``selectors`` buffer (LongTensor (num_pairs,)) is NOT a
  parameter; only the trainable params are exported via ``export_state_dict``.
- **Weight layout matches PyTorch** at export: Conv2d weights stored as
  ``(out_channels, in_channels, kH, kW)``; Linear weights as
  ``(out_features, in_features)``. MLX internally uses NHWC + HWIO layout
  but :meth:`export_state_dict` returns numpy arrays in PyTorch layout.
- **Forward semantics match**: latent embedding -> DepthSep -> sin ->
  PixelShuffle(2) -> ... (7 upsample blocks) -> 1x1 RGB heads -> sigmoid;
  outputs RGB at contest camera resolution (384, 512).

Non-promotability per Catalog #127/#192/#317/#341
-------------------------------------------------

Every artifact produced by this MLX renderer is tagged
``[macOS-MLX research-signal]`` with:

- ``score_claim=False``
- ``promotion_eligible=False``
- ``ready_for_exact_eval_dispatch=False``

The canonical MLX harness
(:func:`tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main`)
auto-stamps these markers on the ``TrainingArtifact``.

Promotion path
--------------

MLX state_dict -> PyTorch via the canonical bridge -> PSV4 archive via the
canonical pack_archive -> contest-equivalence gate via
:mod:`tools.gate_mlx_candidate_contest_equivalence_pact_nerv_selector_v4` ->
operator routes paid CUDA dispatch via ``tools/operator_authorize.py``.

INDIVIDUALLY-FRACTAL per UNIQUE-AND-COMPLETE-PER-METHOD
-------------------------------------------------------

This MLX renderer is PACT-NeRV-SELECTOR-V4's OWN canonical engineering pass
per the 11th INDIVIDUALLY-FRACTAL standing directive 2026-05-27. The
distinguishing primitive (run-length coder over k=16 palette via varint per
Robinson-Cherry 1967 + Capon 1959) is implemented substrate-specifically at
the ARCHIVE-ENCODE TIME path - NOT a shared-helper shortcut from sister
PACT-NeRV variants. The HNeRV-class base decoder topology is ADOPT-CANONICAL
per Catalog #290 (same decoder backbone as PACT-NeRV-IA3 / SELECTOR-V2 /
SELECTOR-V3 / sister NeRV-family substrates per the empirically validated
PR95/PR101/PR110 medal-class topology); the SUBSTRATE-DISTINGUISHING
primitive is the RUN-LENGTH CODER over the k=16 palette (NOT the forward
pass - which is the same as the base decoder).

This is the canonical Catalog #303 cargo-cult-unwind insight: for
SELECTOR-V4, the IA3 gamma-only modulation IS A CARGO-CULT if grafted onto
the selector primitive (which operates at archive-encode time independent of
forward conditioning). The MLX renderer correctly omits the IA3 modulation,
the SELECTOR-V2 arithmetic coder, AND the SELECTOR-V3 Rice-Golomb coder
(each is the WRONG primitive for the SELECTOR-V4 RLE distribution
assumption - temporal-coherence rather than arbitrary or geometric-decay).

Cross-references
----------------

- Canonical sister PyTorch architecture:
  :mod:`tac.substrates.pact_nerv_selector_v4.architecture`
- Canonical MLX HNeRV reference pattern (PACT-NeRV-IA3 sister):
  :mod:`tac.substrates.pact_nerv_ia3.mlx_renderer`
- Canonical SELECTOR family reference patterns:
  :mod:`tac.substrates.pact_nerv_selector_v2.mlx_renderer`
  :mod:`tac.substrates.pact_nerv_selector_v3.mlx_renderer`
- Canonical MLX->PyTorch export bridge:
  :mod:`tac.local_acceleration.mlx_to_pytorch_export`
- Canonical MLX score-aware harness:
  :mod:`tac.substrates._shared.mlx_score_aware`
- ULTIMATE design memo (Step 13 / Variant #13):
  ``.omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md``
- L0 SCAFFOLD design memo (RLE selector primitive):
  ``.omx/research/pact_nerv_g3_selector_extensions_l0_scaffold_design_20260520T204641Z.md``
- Variant selection memo (this landing):
  ``.omx/research/pact_nerv_selector_v4_l1_long_run_mlx_landed_20260528.md``
- SELECTOR-V3 reference landing (cascade continuation):
  ``.omx/research/pact_nerv_selector_v3_l1_long_run_mlx_landed_20260528.md``
- IA3 reference landing (canonical L1 promotion pattern):
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
    from tac.substrates.pact_nerv_selector_v4.architecture import (
        PactNervSelectorV4Config,
    )

try:  # pragma: no cover - exercised on Apple Silicon with MLX installed.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
except Exception as exc:  # pragma: no cover - import guard for non-Apple CI.
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    tree_flatten = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR: Exception | None = exc
else:
    _MLX_IMPORT_ERROR = None


SCHEMA_VERSION = "pact_nerv_selector_v4_mlx_renderer_v1"
MLX_EVIDENCE_GRADE = "[macOS-MLX research-signal]"


def _require_mlx() -> None:
    if mx is None:
        raise RuntimeError(
            "MLX is not available on this host; the PACT-NeRV-SELECTOR-V4 "
            "MLX renderer requires Apple Silicon with the ``mlx`` package "
            f"installed. Original import error: {_MLX_IMPORT_ERROR!r}"
        )


def _pixel_shuffle_2x_nhwc(x: Any) -> Any:
    """PixelShuffle 2x for NHWC tensors via canonical PR95 helper.

    Delegates to ``tac.local_acceleration.pr95_hnerv_mlx::pixel_shuffle_2x_nhwc``
    (empirically PyTorch-byte-stable per FIX-WAVE-R1 ``e1b101888``; 0.0
    absolute drift per CONSOLIDATE-OP-1 2026-05-26 extraction wave).
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
    align_corners=False)`` semantics. For the canonical SELECTOR-V4 decoder
    (initial grid 3x4 + 7 PixelShuffle(2) blocks -> 384x512) the bilinear
    fallback is unused (final block lands exactly at contest resolution).
    """
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
        f"{target_h}x{target_w}) not implemented; canonical SELECTOR-V4 "
        "forward uses integer-ratio 2x PixelShuffle(7) at 3x4 -> 384x512."
    )


def _siren_uniform_bound(fan_in: int, w: float) -> float:
    """SIREN init bound = sqrt(6/fan_in) / max(w, 1.0) per PyTorch sister."""
    return math.sqrt(6.0 / max(fan_in, 1)) / max(w, 1.0)


class _DepthSepConvMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX depth-separable conv (depthwise 3x3 + pointwise 1x1).

    Mirrors :class:`tac.substrates.pact_nerv_selector_v4.architecture._DepthSepConv`.
    MLX nn.Conv2d uses NHWC by default; weight layout HWIO -> we transpose
    to OIHW at export.
    """

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


class _DsUpBlockMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX DepthSep -> sin -> PixelShuffle(2) (mirrors PyTorch sister)."""

    def __init__(self, in_ch: int, out_ch: int, sin_freq: float) -> None:
        _require_mlx()
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.w = float(sin_freq)
        self.dsc = _DepthSepConvMLX(in_ch, out_ch * 4)

    def __call__(self, x: Any) -> Any:
        h = self.dsc(x)
        h = mx.sin(self.w * h)  # type: ignore[union-attr]
        return _pixel_shuffle_2x_nhwc(h)


class PactNervSelectorV4SubstrateMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX-native PACT-NeRV-SELECTOR-V4 substrate (L1 LONG-RUN MLX-LOCAL).

    1:1 architectural mirror of
    :class:`tac.substrates.pact_nerv_selector_v4.architecture.PactNervSelectorV4Substrate`.

    The forward path:

    1. Latent embedding (per-pair) -> initial spatial grid (NHWC).
    2. For each upsample block: DepthSep -> sin -> PixelShuffle(2).
       (NO IA3 gamma-only modulation - that's the IA3 variant's distinguishing
       primitive; SELECTOR-V4's distinguishing primitive is the run-length
       coder over k=16 palette which operates at ARCHIVE-ENCODE TIME.)
    3. Final 1x1 conv heads produce rgb_0 / rgb_1, then sigmoid.
    4. Stack to (B, 2, 3, H, W) for the canonical ``call_b2chw_255``
       convention required by :mod:`tac.substrates._shared.mlx_score_aware`.

    Per HNeRV parity L5: outputs RGB at contest camera resolution (384, 512);
    NOT a mask codec.

    Per CLAUDE.md "MLX portable-local-substrate authority" non-negotiable:
    every artifact produced by this substrate is non-promotable
    ``[macOS-MLX research-signal]``; the canonical promotion path is
    MLX state_dict -> PyTorch export -> PSV4 archive -> contest-equivalence gate.
    """

    def __init__(self, cfg: PactNervSelectorV4Config) -> None:
        _require_mlx()
        super().__init__()
        self.cfg = cfg
        num_pairs = int(cfg.num_pairs)
        latent_dim = int(cfg.latent_dim)
        embed_dim = int(cfg.embed_dim)
        initial_grid_h = int(cfg.initial_grid_h)
        initial_grid_w = int(cfg.initial_grid_w)
        num_upsample_blocks = int(cfg.num_upsample_blocks)

        # Per-pair learnable latent (matches PyTorch ``self.latents``).
        self.latents = mx.random.normal(  # type: ignore[union-attr]
            shape=(num_pairs, latent_dim)
        ) * 0.02

        # Latent -> initial spatial grid embedding.
        self.latent_embed: Any = nn.Linear(  # type: ignore[union-attr]
            latent_dim,
            embed_dim * initial_grid_h * initial_grid_w,
        )

        # Per-block upsample (NO IA3 modulation - see class docstring).
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

        final_ch = channels[num_upsample_blocks]
        self.head_rgb_0: Any = nn.Conv2d(  # type: ignore[union-attr]
            in_channels=final_ch, out_channels=3, kernel_size=1
        )
        self.head_rgb_1: Any = nn.Conv2d(  # type: ignore[union-attr]
            in_channels=final_ch, out_channels=3, kernel_size=1
        )

        # Per-pair selector buffer (filled at archive build time; mirrors the
        # PyTorch sister's register_buffer("selectors", torch.zeros(...))).
        # NOT a trainable parameter; NOT exported via export_state_dict.
        # The L0 SCAFFOLD default = zeros = "none" mode per FEC6 palette.
        self._selectors_np: np.ndarray = np.zeros(num_pairs, dtype=np.int64)

        self._siren_init()

    def _siren_init(self) -> None:
        """SIREN init for Conv2d + Linear (mirrors PyTorch sister)."""
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

    def __call__(self, pair_indices: Any) -> Any:
        """Forward returning (B, 2, 3, H, W) in [0, 255] per ``call_b2chw_255``.

        Args:
            pair_indices: (B,) int array of pair indices in [0, num_pairs).

        Returns:
            (B, 2, 3, H, W) MLX float32 array in [0, 255] matching the
            ``call_b2chw_255`` convention of the canonical MLX score-aware
            harness.
        """
        # Latent gather (no pose; SELECTOR-V4 has no pose conditioning).
        z = mx.take(self.latents, pair_indices, axis=0)  # type: ignore[union-attr]

        # Latent -> initial spatial grid. PyTorch sister views the linear
        # projection as NCHW; mirror that memory order before converting to
        # MLX's NHWC Conv2d convention.
        h = self.latent_embed(z)
        h = mx.reshape(  # type: ignore[union-attr]
            h,
            (
                -1,
                self.cfg.embed_dim,
                self.cfg.initial_grid_h,
                self.cfg.initial_grid_w,
            ),
        )
        h = mx.transpose(h, (0, 2, 3, 1))  # type: ignore[union-attr]

        # Per-block forward (NO IA3 gamma-only modulation; SELECTOR-V4 primitive
        # is run-length coding at archive-encode time only).
        for block in self.blocks:
            h = block(h)

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

    @property
    def selectors(self) -> np.ndarray:
        """Per-pair selector buffer (LongTensor (num_pairs,)) mirror.

        Returns a numpy view; mutate via ``set_selectors`` (PyTorch sister
        uses ``register_buffer`` semantics - non-trainable + persistent).
        """
        return self._selectors_np

    def set_selectors(self, selectors: np.ndarray) -> None:
        """Replace the per-pair selector buffer (non-trainable; archive-encode time)."""
        if selectors.ndim != 1:
            raise ValueError(f"selectors must be 1-D; got shape {selectors.shape}")
        if int(selectors.shape[0]) != int(self.cfg.num_pairs):
            raise ValueError(
                f"selectors length {selectors.shape[0]} != num_pairs "
                f"{self.cfg.num_pairs}"
            )
        if not np.issubdtype(selectors.dtype, np.integer):
            raise ValueError(
                f"selectors must be integer dtype; got {selectors.dtype}"
            )
        if selectors.min() < 0 or selectors.max() >= int(
            self.cfg.selector_palette_size
        ):
            raise ValueError(
                f"selectors must be in [0, {self.cfg.selector_palette_size}); "
                f"got range [{selectors.min()}, {selectors.max()}]"
            )
        self._selectors_np = selectors.astype(np.int64).copy()

    def export_state_dict(self) -> dict[str, np.ndarray]:
        """Export weights in PyTorch state_dict layout for the canonical bridge.

        The canonical MLX->PyTorch bridge expects:
        - Conv2d weights: (out_channels, in_channels, kH, kW). MLX NHWC
          layout is (out, kH, kW, in); we transpose ``(0, 3, 1, 2)``.
        - Linear weights: (out_features, in_features). MLX layout matches.
        - Bias: (out_features,) or (out_channels,). Matches.
        - Per-pair tensors (latents): (num_pairs, D).

        NOTE: the ``selectors`` buffer is NOT exported here (it's
        non-trainable and lives in the archive-encode path; PyTorch sister
        loads via ``register_buffer`` which is not in ``parameters()``).

        Returns:
            dict[str, np.ndarray] keyed by PyTorch sister parameter names.
        """
        _require_mlx()
        out: dict[str, np.ndarray] = {}
        out["latents"] = np.asarray(self.latents, dtype=np.float32).copy()
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
    "PactNervSelectorV4SubstrateMLX",
]
