# SPDX-License-Identifier: MIT
"""MLX-native Selfcomp grayscale_lut substrate variant.

OVERNIGHT-WW Phase 3 per operator directive 2026-05-21 *"perhaps we should
start writing portable reusable composable primitives in MLX and PyTorch as
well"*: this module is the MLX-native sister of the canonical PyTorch
:class:`tac.substrates.grayscale_lut.architecture.GrayscaleLutSubstrate`.

The existing PyTorch trainer at ``experiments/train_substrate_grayscale_lut.py``
remains the canonical contest-axis path; this MLX-native variant lets the
operator (a) iterate Selfcomp paradigm experiments on M5 Max at $0 cost,
(b) export trained weights via canonical pipeline at
``tac.local_acceleration.mlx_to_pytorch_export``, (c) load the exported
weights on CUDA T4 via the canonical PyTorch architecture for authoritative
contest-axis eval.

Per CLAUDE.md non-negotiables PRESERVED:
- **Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE**: this module is
  a NEW sister to the existing PyTorch architecture; ZERO mutation of the
  canonical architecture or trainer.
- **Catalog #1 MPS auth eval is NOISE** + **Catalog #192 macOS
  non-promotable**: MLX-native trained weights are research-signal until
  exported to PyTorch + evaluated on CUDA T4 via
  ``experiments/contest_auth_eval.py``.
- **Catalog #287/#323 canonical Provenance**: any persisted artifact from
  this module carries `evidence_grade="macOS-MLX-research-signal"` per
  the OO scaffold.
- **Beauty + DX**: 1:1 architectural mirror of the PyTorch variant so the
  canonical weight-export pipeline is byte-stable (tensor names + shapes
  match).

Architecture mirror (PyTorch -> MLX field names preserved):
    grayscale: (num_pairs, 1, H/D, W/D) fp32 -- analog grayscale stream
    pair_embedding: (num_pairs, embedding_dim) fp32 -- FiLM conditioning
    stem.weight + stem.bias: Conv2d(1, decoder_hidden, k=3, p=1)
    blocks.<i>.conv.weight + blocks.<i>.conv.bias: Conv2d(hidden, hidden, k=3, p=1)
    blocks.<i>.film_gen.weight + blocks.<i>.film_gen.bias: Linear(emb_dim, 2*hidden)
    head_rgb_0.weight + head_rgb_0.bias: Conv2d(hidden, 3, k=3, p=1)
    head_rgb_1.weight + head_rgb_1.bias: Conv2d(hidden, 3, k=3, p=1)

Forward: render frame-pairs at the given pair indices via FiLM-conditioned
RGB decoder over upsampled grayscale.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tac.portable_primitives import Backend, is_mlx_available
from tac.portable_primitives import nn as pnn
from tac.substrates.grayscale_lut.architecture import GrayscaleLutConfig

__all__ = [
    "GrayscaleLutMLXNative",
    "render_pair_mlx",
]


class GrayscaleLutMLXNative:
    """MLX-native Selfcomp grayscale_lut substrate.

    Constructor: ``GrayscaleLutMLXNative(cfg)`` with the same
    :class:`GrayscaleLutConfig` as the PyTorch variant. The constructor
    raises :class:`RuntimeError` if MLX is not available on the host (the
    operator must fall back to the PyTorch variant on Linux / CI).

    Forward signature matches the PyTorch variant:
        ``substrate(pair_indices_mlx) -> (rgb_0_mlx, rgb_1_mlx)``
    where each output is shape ``(B, 3, H, W)`` in [0, 1].
    """

    def __init__(self, cfg: GrayscaleLutConfig, *, seed: int = 0) -> None:
        if not is_mlx_available():
            raise RuntimeError(
                "MLX framework not available on this host; use the canonical "
                "PyTorch GrayscaleLutSubstrate at "
                "tac.substrates.grayscale_lut.architecture.GrayscaleLutSubstrate "
                "for non-Apple-Silicon dispatch (CUDA / CPU / MPS)."
            )
        import mlx.core as mx

        self.cfg = cfg
        self.backend = Backend.MLX

        h_g = cfg.output_height // cfg.grayscale_downsample
        w_g = cfg.output_width // cfg.grayscale_downsample

        rng = np.random.RandomState(seed)

        # Per-pair grayscale field initialized to mid-gray (matches PyTorch
        # variant's torch.full(..., 0.5)).
        self.grayscale = mx.full((cfg.num_pairs, 1, h_g, w_g), 0.5, dtype=mx.float32)

        # Per-pair embedding initialized with std=0.02 normal (matches PyTorch).
        embedding_np = rng.standard_normal((cfg.num_pairs, cfg.embedding_dim)).astype(np.float32) * 0.02
        self.pair_embedding = mx.array(embedding_np)

        # Decoder stem
        self.stem = pnn.PortableConv2d(
            1, cfg.decoder_hidden, kernel_size=3, backend="mlx", seed=seed + 1
        )

        # Decoder FiLM blocks: each is conv + film_gen pair
        self._block_convs: list[pnn.PortableConv2d] = []
        self._block_film_gens: list[pnn.PortableLinear] = []
        for i in range(cfg.decoder_blocks):
            conv = pnn.PortableConv2d(
                cfg.decoder_hidden, cfg.decoder_hidden, kernel_size=3,
                backend="mlx", seed=seed + 2 + 2 * i,
            )
            film_gen = pnn.PortableLinear(
                cfg.embedding_dim, 2 * cfg.decoder_hidden,
                backend="mlx", seed=seed + 3 + 2 * i,
            )
            self._block_convs.append(conv)
            self._block_film_gens.append(film_gen)

        # Two RGB heads
        self.head_rgb_0 = pnn.PortableConv2d(
            cfg.decoder_hidden, 3, kernel_size=3, backend="mlx",
            seed=seed + 100,
        )
        self.head_rgb_1 = pnn.PortableConv2d(
            cfg.decoder_hidden, 3, kernel_size=3, backend="mlx",
            seed=seed + 101,
        )

    def __call__(self, pair_indices_mlx: Any) -> tuple[Any, Any]:
        """Render frame-pairs at the given pair indices."""
        import mlx.core as mx

        # Fancy indexing: gs = grayscale[pair_indices]; emb = pair_embedding[pair_indices]
        gs = self.grayscale[pair_indices_mlx]  # (B, 1, H/D, W/D)
        emb = self.pair_embedding[pair_indices_mlx]  # (B, embedding_dim)

        # Bilinear upsample to (H, W)
        gs_up = pnn.bilinear_upsample(
            gs, size=(self.cfg.output_height, self.cfg.output_width), backend="mlx"
        )

        h = self.stem(gs_up)  # (B, hidden, H, W)
        for conv, film_gen in zip(self._block_convs, self._block_film_gens):
            h_new = conv(h)
            gamma_beta = film_gen(emb)  # (B, 2*hidden)
            # Chunk into gamma + beta
            hidden = self.cfg.decoder_hidden
            gamma = gamma_beta[:, :hidden]
            beta = gamma_beta[:, hidden:]
            # Broadcast (B, hidden) -> (B, hidden, 1, 1)
            gamma = mx.expand_dims(mx.expand_dims(gamma, -1), -1)
            beta = mx.expand_dims(mx.expand_dims(beta, -1), -1)
            h = h_new * (1.0 + gamma) + beta
            h = pnn.gelu(h, backend="mlx")

        rgb_0 = pnn.sigmoid(self.head_rgb_0(h), backend="mlx")
        rgb_1 = pnn.sigmoid(self.head_rgb_1(h), backend="mlx")
        return rgb_0, rgb_1

    def export_state_dict(self) -> dict[str, np.ndarray]:
        """Export model weights as a numpy state-dict in PyTorch layout.

        The keys match the PyTorch variant's :meth:`state_dict` output (so
        :mod:`tac.local_acceleration.mlx_to_pytorch_export` can load this
        directly into the canonical PyTorch architecture).

        Returns:
            Dict mapping PyTorch-canonical parameter name -> numpy array.
        """
        import mlx.core as mx

        # Materialize MLX tensors then convert to numpy.
        mx.eval(self.grayscale, self.pair_embedding)
        result: dict[str, np.ndarray] = {
            "grayscale": np.array(self.grayscale),
            "pair_embedding": np.array(self.pair_embedding),
        }

        stem_w, stem_b = self.stem.export_weights()
        result["stem.weight"] = stem_w
        result["stem.bias"] = stem_b

        for i, (conv, film_gen) in enumerate(
            zip(self._block_convs, self._block_film_gens)
        ):
            cw, cb = conv.export_weights()
            fw, fb = film_gen.export_weights()
            result[f"blocks.{i}.conv.weight"] = cw
            result[f"blocks.{i}.conv.bias"] = cb
            result[f"blocks.{i}.film_gen.weight"] = fw
            result[f"blocks.{i}.film_gen.bias"] = fb

        h0_w, h0_b = self.head_rgb_0.export_weights()
        h1_w, h1_b = self.head_rgb_1.export_weights()
        result["head_rgb_0.weight"] = h0_w
        result["head_rgb_0.bias"] = h0_b
        result["head_rgb_1.weight"] = h1_w
        result["head_rgb_1.bias"] = h1_b

        return result

    def load_state_dict_from_numpy(self, state: dict[str, np.ndarray]) -> None:
        """Load numpy state-dict (PyTorch layout) into MLX weights.

        Inverse of :meth:`export_state_dict`. Used for testing the
        round-trip MLX -> numpy -> MLX equivalence.
        """
        import mlx.core as mx

        if "grayscale" in state:
            self.grayscale = mx.array(state["grayscale"].astype(np.float32))
        if "pair_embedding" in state:
            self.pair_embedding = mx.array(state["pair_embedding"].astype(np.float32))

        if "stem.weight" in state and "stem.bias" in state:
            self.stem.load_weights(state["stem.weight"], state["stem.bias"])

        for i, (conv, film_gen) in enumerate(
            zip(self._block_convs, self._block_film_gens)
        ):
            cw_key = f"blocks.{i}.conv.weight"
            cb_key = f"blocks.{i}.conv.bias"
            fw_key = f"blocks.{i}.film_gen.weight"
            fb_key = f"blocks.{i}.film_gen.bias"
            if cw_key in state:
                conv.load_weights(state[cw_key], state[cb_key])
            if fw_key in state:
                film_gen.load_weights(state[fw_key], state[fb_key])

        if "head_rgb_0.weight" in state:
            self.head_rgb_0.load_weights(state["head_rgb_0.weight"], state["head_rgb_0.bias"])
        if "head_rgb_1.weight" in state:
            self.head_rgb_1.load_weights(state["head_rgb_1.weight"], state["head_rgb_1.bias"])

    def num_parameters(self) -> int:
        """Count trainable parameters (matches the PyTorch sister)."""
        cfg = self.cfg
        h_g = cfg.output_height // cfg.grayscale_downsample
        w_g = cfg.output_width // cfg.grayscale_downsample
        # grayscale + pair_embedding
        n = cfg.num_pairs * h_g * w_g
        n += cfg.num_pairs * cfg.embedding_dim
        # stem: in=1, out=hidden, k=3
        n += 1 * cfg.decoder_hidden * 9 + cfg.decoder_hidden
        # blocks: each = conv (h*h*9 + h) + film_gen (e * 2h + 2h)
        per_block = cfg.decoder_hidden * cfg.decoder_hidden * 9 + cfg.decoder_hidden
        per_block += cfg.embedding_dim * 2 * cfg.decoder_hidden + 2 * cfg.decoder_hidden
        n += cfg.decoder_blocks * per_block
        # heads: 2 x (h*3*9 + 3)
        n += 2 * (cfg.decoder_hidden * 3 * 9 + 3)
        return n


def render_pair_mlx(
    substrate: GrayscaleLutMLXNative,
    pair_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Render the requested pairs and return numpy arrays.

    Convenience helper for tests and the canonical export pipeline.
    """
    import mlx.core as mx

    indices_mlx = mx.array(pair_indices.astype(np.int32))
    rgb_0_mlx, rgb_1_mlx = substrate(indices_mlx)
    mx.eval(rgb_0_mlx, rgb_1_mlx)
    return np.array(rgb_0_mlx), np.array(rgb_1_mlx)
