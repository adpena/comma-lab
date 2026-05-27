# SPDX-License-Identifier: MIT
"""mlx_renderer — MLX-first renderer for J=MDL-IBPS DISCRETE-CATEGORICAL-MINE-HYBRID.

Per AMENDMENT #3 axis 2 (MLX drift minimization per primitive): every MLX
primitive in this module has a sister numpy reference implementation in
``numpy_reference.py``; tests measure MLX <-> numpy parity (<=1e-5) and
MLX <-> PyTorch parity (<=0.001 per Catalog #1265 gate threshold; 90x
margin over empirical anchor 0.000011).

Per CC-J-10 unwind: all artifacts produced by this module carry
``[macOS-MLX research-signal]`` evidence tag + ``score_claim=false`` +
``promotion_eligible=false`` + ``ready_for_exact_eval_dispatch=false``
per Catalog #192/#317 non-promotable markers.

Per CLAUDE.md FORBIDDEN_PATTERNS:
- No silent device defaults (MLX is explicit; no MPS-fallback ternary)
- No scorer load at inflate time (this module renders only)
- No /tmp paths in persisted artifacts

MLX drift mitigation per Catalog #1255:
- Sinusoidal positional encoding: fp32 explicit cast for ``mx.sin`` / ``mx.cos``
- MINE critic forward: fp32 accumulation per ``mx.matmul`` discipline
- Categorical posterior sampling: ``mx.random.gumbel(dtype=mx.float32)`` per sister A=DreamerV3 pattern
- Softmax: explicit ``axis`` + numerical-safe via max-subtraction (no `mx.softmax` without `axis`)
- KL Gaussian formula (for comparison-test only): Kahan summation per Catalog #962 / slot 16

Architecture summary (see ``__init__.py`` docstring for full details):
    categorical_indices m_i in {0, ..., K-1}^G [K=16; G=12; 48 bits per pair]
        -> one_hot(m_i) in R^{G*K}
        -> film_proj -> (scales, shifts) in R^{NUM_HIDDEN_LAYERS x HIDDEN_DIM x 2}
        -> procedural coord-MLP with FiLM modulation -> rgb in [0, 1]^3
"""

from __future__ import annotations

import math
from typing import Any

try:
    import mlx.core as mx
    import mlx.nn as nn
    _MLX_AVAILABLE = True
except ImportError:
    # Sister substrates follow this pattern: numpy_reference.py is the
    # fallback when MLX is not installed (e.g. GHA CPU CI). Importing this
    # module on a non-Apple-Silicon host without MLX produces an explicit
    # ImportError at first MLX-API call, not a silent fallback.
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _MLX_AVAILABLE = False

from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid import (
    CATEGORICAL_G,
    CATEGORICAL_K,
    EVAL_HW,
    HIDDEN_DIM,
    NUM_HIDDEN_LAYERS,
    POS_DIM,
)


def _require_mlx() -> None:
    """Raise an explicit ImportError if MLX is not available.

    Per CC-J-10 unwind: NO silent fallback. Operator-facing diagnostic
    points at the numpy_reference.py sister for non-Apple-Silicon hardware.
    """
    if not _MLX_AVAILABLE:
        raise ImportError(
            "MLX is not installed; use tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid"
            ".numpy_reference for CPU-only fallback per AMENDMENT #3 axis 3"
        )


def sinusoidal_positional_encoding_mlx(coords, pos_dim: int = POS_DIM):
    """Sinusoidal positional encoding (NeRF-style) - MLX implementation.

    Args:
        coords: ``(N, 3)`` mx.array of (x, y, t) coordinates in [0, 1].
        pos_dim: number of frequency bands per coordinate axis.

    Returns:
        ``(N, pos_dim * 2 * 3)`` mx.array; for each (x, y, t) we emit
        [sin(2^k * pi * c), cos(2^k * pi * c)] for k in 0..pos_dim-1, c in (x, y, t).
    """
    _require_mlx()
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (N, 3); got {coords.shape}")
    # Explicit fp32 per Catalog #1255 MLX drift mitigation
    coords_f32 = coords.astype(mx.float32)
    freqs_np = [(2.0 ** k) * math.pi for k in range(pos_dim)]
    freqs = mx.array(freqs_np, dtype=mx.float32)
    # (N, 3) -> (N, 3, pos_dim) via outer product
    scaled = coords_f32[:, :, None] * freqs[None, None, :]
    sins = mx.sin(scaled)
    coss = mx.cos(scaled)
    # Concatenate sin + cos along the freq axis: (N, 3, 2 * pos_dim)
    encoded = mx.concatenate([sins, coss], axis=-1)
    # Flatten last two axes: (N, 3 * 2 * pos_dim)
    return encoded.reshape(coords.shape[0], -1)


def film_modulation_mlx(h, scale, shift):
    """FiLM modulation: h * scale + shift (Perez et al. 2017) - MLX impl."""
    _require_mlx()
    if not (h.shape == scale.shape == shift.shape):
        raise ValueError(
            f"h/scale/shift must have matching shapes; got "
            f"h={h.shape}, scale={scale.shape}, shift={shift.shape}"
        )
    return h * scale + shift


def categorical_to_one_hot_mlx(indices, K: int = CATEGORICAL_K):
    """Categorical index -> one-hot encoding - MLX implementation.

    Args:
        indices: ``(B, G)`` mx.array of categorical indices (int dtype).
        K: alphabet size.

    Returns:
        ``(B, G * K)`` flattened one-hot mx.array in fp32.
    """
    _require_mlx()
    if indices.ndim != 2:
        raise ValueError(f"indices must be 2D (B, G); got {indices.shape}")
    B, G = indices.shape
    # Cast to int32 for safe indexing
    idx_i32 = indices.astype(mx.int32)
    # One-hot via comparison to range
    range_K = mx.arange(K, dtype=mx.int32)
    # (B, G, 1) == (K,) -> (B, G, K) bool -> fp32
    one_hot = (idx_i32[:, :, None] == range_K[None, None, :]).astype(mx.float32)
    return one_hot.reshape(B, G * K)


class FilmProjMLX:
    """Project categorical one-hot -> FiLM (scale, shift) parameters - MLX impl.

    Simple linear projection without bias regularization at scaffold time;
    full training adds sparse-Laplacian regularizer per ``ib_loss_mine.py``.
    """

    def __init__(self, weights: mx.array, biases: mx.array) -> None:
        _require_mlx()
        self.weights = weights.astype(mx.float32)
        self.biases = biases.astype(mx.float32)
        expected_out = NUM_HIDDEN_LAYERS * HIDDEN_DIM * 2
        expected_in = CATEGORICAL_G * CATEGORICAL_K
        if self.weights.shape != (expected_in, expected_out):
            raise ValueError(
                f"film_proj weights must have shape ({expected_in}, {expected_out}); "
                f"got {self.weights.shape}"
            )
        if self.biases.shape != (expected_out,):
            raise ValueError(
                f"film_proj biases must have shape ({expected_out},); "
                f"got {self.biases.shape}"
            )

    def __call__(self, one_hot):
        """Project one-hot (B, G*K) -> (scales, shifts) each (B, L, H)."""
        _require_mlx()
        B = one_hot.shape[0]
        proj = one_hot @ self.weights + self.biases
        proj = proj.reshape(B, NUM_HIDDEN_LAYERS, HIDDEN_DIM, 2)
        scales = proj[..., 0]
        shifts = proj[..., 1]
        return scales, shifts


class CoordMLPBaseMLX:
    """Procedural coord-MLP base with FiLM modulation - MLX implementation.

    Architecture per ``__init__.py`` module docstring.
    """

    def __init__(
        self,
        weights_first: mx.array,
        biases_first: mx.array,
        weights_hidden: list,
        biases_hidden: list,
        weights_out: mx.array,
        biases_out: mx.array,
    ) -> None:
        _require_mlx()
        self.weights_first = weights_first.astype(mx.float32)
        self.biases_first = biases_first.astype(mx.float32)
        self.weights_hidden = [w.astype(mx.float32) for w in weights_hidden]
        self.biases_hidden = [b.astype(mx.float32) for b in biases_hidden]
        self.weights_out = weights_out.astype(mx.float32)
        self.biases_out = biases_out.astype(mx.float32)
        if len(self.weights_hidden) != NUM_HIDDEN_LAYERS:
            raise ValueError(
                f"expected {NUM_HIDDEN_LAYERS} hidden weight matrices; got "
                f"{len(self.weights_hidden)}"
            )

    def __call__(self, coords, film_scales, film_shifts):
        """Forward pass; see ``numpy_reference.CoordMLPBaseNumpy.forward`` for contract."""
        _require_mlx()
        x = sinusoidal_positional_encoding_mlx(coords, pos_dim=POS_DIM)
        h = x @ self.weights_first + self.biases_first
        for layer_idx in range(NUM_HIDDEN_LAYERS):
            scale = film_scales[:, layer_idx, :]
            shift = film_shifts[:, layer_idx, :]
            h = film_modulation_mlx(h, scale, shift)
            h = mx.sin(h)
            if layer_idx < NUM_HIDDEN_LAYERS - 1:
                h = h @ self.weights_hidden[layer_idx] + self.biases_hidden[layer_idx]
        rgb = h @ self.weights_out + self.biases_out
        # Sigmoid via stable formulation
        rgb = mx.sigmoid(rgb)
        return rgb


class MINECriticMLX:
    """MINE critic (Belghazi 2018) - MLX implementation.

    2-layer MLP critic; produces scalar T_theta(z, frames) per sample.
    """

    def __init__(
        self,
        weights: list,  # 3 weight matrices
        biases: list,  # 3 bias vectors
    ) -> None:
        _require_mlx()
        if len(weights) != 3 or len(biases) != 3:
            raise ValueError(
                f"MINE critic requires 3 weight matrices + 3 biases; got "
                f"{len(weights)} weights, {len(biases)} biases"
            )
        self.weights = [w.astype(mx.float32) for w in weights]
        self.biases = [b.astype(mx.float32) for b in biases]

    def __call__(self, z, frames_features):
        """Forward T_theta(z, frames); see numpy_reference.mine_critic_forward_numpy."""
        _require_mlx()
        # Explicit fp32 for matmul accumulation per Catalog #1255
        z_f32 = z.astype(mx.float32)
        f_f32 = frames_features.astype(mx.float32)
        h = mx.concatenate([z_f32, f_f32], axis=-1)
        h = h @ self.weights[0] + self.biases[0]
        h = mx.maximum(0.0, h)  # ReLU (mx.relu also acceptable)
        h = h @ self.weights[1] + self.biases[1]
        h = mx.maximum(0.0, h)
        out = h @ self.weights[2] + self.biases[2]
        # Squeeze trailing scalar dimension
        return out.squeeze(-1)


def mine_lower_bound_mlx(critic_joint, critic_marginal) -> float:
    """Compute MINE Donsker-Varadhan lower bound on mutual information - MLX.

    Numerically stable via max-subtraction; matches numpy reference contract.
    """
    _require_mlx()
    if critic_joint.shape != critic_marginal.shape:
        raise ValueError(
            f"critic_joint and critic_marginal must have matching shape; got "
            f"{critic_joint.shape} vs {critic_marginal.shape}"
        )
    # Stable log-mean-exp: subtract max for numerical stability
    max_marginal = mx.max(critic_marginal)
    log_mean_exp = max_marginal + mx.log(
        mx.mean(mx.exp(critic_marginal - max_marginal))
    )
    bound = mx.mean(critic_joint) - log_mean_exp
    return float(bound.item())


def gumbel_softmax_sample_mlx(
    logits, temperature: float = 1.0, hard: bool = False
):
    """Gumbel-Softmax reparametrization (Jang et al. 2016) - MLX implementation.

    Args:
        logits: ``(B, G, K)`` mx.array of categorical logits.
        temperature: Gumbel-Softmax temperature (default 1.0).
        hard: if True, return one-hot via straight-through estimator;
              if False, return soft sample (probability distribution).

    Returns:
        ``(B, G, K)`` mx.array sample.
    """
    _require_mlx()
    if logits.ndim != 3:
        raise ValueError(f"logits must be 3D (B, G, K); got {logits.shape}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0; got {temperature}")
    # Sample Gumbel noise per Catalog #1255 mitigation: explicit fp32
    # MLX gumbel: -log(-log(uniform)) via uniform + transform
    uniform = mx.random.uniform(
        low=1e-10, high=1.0 - 1e-10, shape=logits.shape, dtype=mx.float32
    )
    gumbel_noise = -mx.log(-mx.log(uniform))
    # Add noise + scale by 1/temperature
    perturbed = (logits.astype(mx.float32) + gumbel_noise) / temperature
    # Numerically-stable softmax: subtract per-group max
    max_per_group = mx.max(perturbed, axis=-1, keepdims=True)
    exp_shifted = mx.exp(perturbed - max_per_group)
    soft = exp_shifted / mx.sum(exp_shifted, axis=-1, keepdims=True)
    if hard:
        # Straight-through: argmax forward, soft gradient backward
        # MLX does not yet support stop_gradient elegantly; for inference use argmax
        K = logits.shape[-1]
        argmax_idx = mx.argmax(soft, axis=-1)
        # Manual one-hot
        range_K = mx.arange(K, dtype=mx.int32)
        one_hot_hard = (argmax_idx[..., None] == range_K[None, None, :]).astype(mx.float32)
        return one_hot_hard
    return soft


def make_pixel_coords_mlx(
    height: int = EVAL_HW[0], width: int = EVAL_HW[1], t: int = 0
):
    """Generate (height * width, 3) pixel coords (x, y, t) normalized to [0, 1] - MLX."""
    _require_mlx()
    if t not in (0, 1):
        raise ValueError(f"t must be 0 or 1; got {t}")
    ys = mx.linspace(0.0, 1.0, height).astype(mx.float32)
    xs = mx.linspace(0.0, 1.0, width).astype(mx.float32)
    # Meshgrid (indexing="ij" not supported in older MLX; manual broadcast)
    grid_y = ys[:, None] * mx.ones((1, width), dtype=mx.float32)
    grid_x = mx.ones((height, 1), dtype=mx.float32) * xs[None, :]
    grid_t = mx.full((height, width), float(t), dtype=mx.float32)
    return mx.stack([grid_x.flatten(), grid_y.flatten(), grid_t.flatten()], axis=-1)


class MDLIBPSJRendererMLX:
    """Full J=MDL-IBPS renderer composing all primitives - MLX.

    Composition:
        categorical_indices -> one_hot -> film_proj -> (scales, shifts)
                                                        |
        pixel_coords (x, y, t) -------------------------+
                              \\                         |
                               -> CoordMLPBaseMLX -> rgb
    """

    def __init__(
        self,
        film_proj: FilmProjMLX,
        coord_mlp: CoordMLPBaseMLX,
    ) -> None:
        _require_mlx()
        self.film_proj = film_proj
        self.coord_mlp = coord_mlp

    def render_pair(
        self,
        categorical_indices,
        height: int = EVAL_HW[0],
        width: int = EVAL_HW[1],
    ):
        """Render rgb_0, rgb_1 for given per-pair categorical indices.

        Args:
            categorical_indices: ``(G,)`` int mx.array of indices for ONE pair.
            height: image height (default EVAL_HW[0] = 384).
            width: image width (default EVAL_HW[1] = 512).

        Returns:
            ``(2, 3, height, width)`` mx.array of rgb for (frame_0, frame_1).
        """
        _require_mlx()
        if categorical_indices.ndim != 1:
            raise ValueError(
                f"categorical_indices must be 1D (G,); got {categorical_indices.shape}"
            )
        # Expand to (1, G) for one_hot
        idx_2d = categorical_indices[None, :]
        one_hot = categorical_to_one_hot_mlx(idx_2d, K=CATEGORICAL_K)
        # FiLM projection -> (1, NUM_HIDDEN_LAYERS, HIDDEN_DIM) for scales + shifts
        scales, shifts = self.film_proj(one_hot)
        # Render both frames
        frames = []
        for t in (0, 1):
            coords = make_pixel_coords_mlx(height=height, width=width, t=t)
            N = coords.shape[0]
            # Broadcast scales/shifts to all N pixels
            scales_b = mx.broadcast_to(scales, (N, NUM_HIDDEN_LAYERS, HIDDEN_DIM))
            shifts_b = mx.broadcast_to(shifts, (N, NUM_HIDDEN_LAYERS, HIDDEN_DIM))
            rgb_flat = self.coord_mlp(coords, scales_b, shifts_b)  # (N, 3)
            rgb = rgb_flat.reshape(height, width, 3).transpose(2, 0, 1)  # (3, H, W)
            frames.append(rgb)
        return mx.stack(frames, axis=0)


class MDLIBPSJTrainableRendererMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Trainable ``nn.Module`` J=MDL-IBPS renderer for the MLX score-aware harness.

    MLX-SCORE-AWARE-HARNESS-WAVE 2026-05-27: the prior blocker was that
    :class:`MDLIBPSJRendererMLX` holds FIXED loaded weights (``FilmProjMLX`` /
    ``CoordMLPBaseMLX`` take pre-built arrays — loaded-weight inference, NOT
    learnable params; not an ``mlx.nn.Module``). This class extinguishes the
    blocker: it wraps the discrete-categorical-posterior + FiLM-proj + FiLM-
    conditioned CoordMLP as a single trainable ``nn.Module`` whose params MLX
    ``value_and_grad`` discovers via ``.parameters()``.

    Trainable parameters:

    - ``logits`` ``(num_pairs, G, K)`` per-pair categorical logits (the discrete
      IB code; argmax indices stored in the archive at inflate).
    - ``film_proj`` ``nn.Linear(G*K, NUM_HIDDEN_LAYERS*HIDDEN_DIM*2)`` (categorical
      one-hot -> per-layer FiLM scale+shift).
    - ``input_proj`` ``nn.Linear(pos_enc_dim, HIDDEN_DIM)`` (positional encoding).
    - ``hidden`` ``[nn.Linear(HIDDEN_DIM, HIDDEN_DIM)] x NUM_HIDDEN_LAYERS``.
    - ``output_proj`` ``nn.Linear(HIDDEN_DIM, 3)`` (RGB head).

    Forward (training): per-pair logits -> Gumbel-Softmax (soft, STE) ->
    one-hot -> FiLM proj -> CoordMLP(coords, scales, shifts) -> sigmoid RGB.
    Exposes the harness ``reconstruct_pair(idx) -> (rgb_0, rgb_1)`` NCHW
    ``[0, 1]`` convention (frame_0 t=0, frame_1 t=1).

    The MINE critic (``MINECriticMLX``) is the substrate's UNIQUE distinguishing
    extra term; it is wired separately via the harness ``extra_loss_terms``
    callback (NOT folded into the renderer forward) so the canonical loop never
    assumes a fixed loss signature.

    Non-promotable ``[macOS-MLX research-signal]`` per CLAUDE.md "MLX
    portable-local-substrate authority".
    """

    def __init__(
        self,
        num_pairs: int,
        *,
        gumbel_temperature: float = 1.0,
        height: int = EVAL_HW[0],
        width: int = EVAL_HW[1],
    ) -> None:
        _require_mlx()
        super().__init__()
        self.num_pairs = int(num_pairs)
        self.gumbel_temperature = float(gumbel_temperature)
        self.height = int(height)
        self.width = int(width)
        pos_enc_dim = POS_DIM * 2 * 3
        key = mx.random.key(0)
        self.logits = (
            mx.random.normal(
                shape=(self.num_pairs, CATEGORICAL_G, CATEGORICAL_K), key=key
            )
            * 0.01
        )
        self.film_proj = nn.Linear(
            CATEGORICAL_G * CATEGORICAL_K, NUM_HIDDEN_LAYERS * HIDDEN_DIM * 2
        )
        self.input_proj = nn.Linear(pos_enc_dim, HIDDEN_DIM)
        self.hidden = [
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(NUM_HIDDEN_LAYERS)
        ]
        self.output_proj = nn.Linear(HIDDEN_DIM, 3)

    def _render_frame(self, scales: Any, shifts: Any, t: int) -> Any:
        """Render one (B, 3, H, W) frame at coord-time t for FiLM (scales, shifts).

        Args:
            scales / shifts: ``(B, NUM_HIDDEN_LAYERS, HIDDEN_DIM)`` per-pair FiLM.
            t: frame index (0 or 1).

        Returns:
            ``(B, 3, H, W)`` RGB in ``[0, 1]`` (NCHW).
        """
        b = int(scales.shape[0])
        coords = make_pixel_coords_mlx(height=self.height, width=self.width, t=t)
        n = coords.shape[0]  # H*W
        pos_enc = sinusoidal_positional_encoding_mlx(coords, pos_dim=POS_DIM)
        # Shared base over all pixels; broadcast to batch.
        h = self.input_proj(pos_enc)  # (N, HIDDEN_DIM)
        h = mx.broadcast_to(h[None], (b, n, HIDDEN_DIM))  # (B, N, HIDDEN_DIM)
        for i in range(NUM_HIDDEN_LAYERS):
            h = self.hidden[i](h)
            scale = scales[:, i, :][:, None, :]  # (B, 1, HIDDEN_DIM)
            shift = shifts[:, i, :][:, None, :]
            h = mx.sin(h * scale + shift)
        rgb = mx.sigmoid(self.output_proj(h))  # (B, N, 3)
        rgb = mx.reshape(rgb, (b, self.height, self.width, 3))
        return mx.transpose(rgb, (0, 3, 1, 2))  # (B, 3, H, W)

    def reconstruct_pair(self, pair_indices: Any) -> tuple[Any, Any]:
        """Harness forward: per-pair categorical logits -> (rgb_0, rgb_1).

        Args:
            pair_indices: ``(B,)`` int tensor in ``[0, num_pairs)``.

        Returns:
            ``(rgb_0, rgb_1)`` each ``(B, 3, H, W)`` in ``[0, 1]``.
        """
        _require_mlx()
        logits = mx.take(self.logits, pair_indices, axis=0)  # (B, G, K)
        soft = gumbel_softmax_sample_mlx(
            logits, temperature=self.gumbel_temperature, hard=False
        )  # (B, G, K) simplex
        b = int(soft.shape[0])
        one_hot = mx.reshape(soft, (b, CATEGORICAL_G * CATEGORICAL_K))
        proj = self.film_proj(one_hot)  # (B, L*H*2)
        proj = mx.reshape(proj, (b, NUM_HIDDEN_LAYERS, HIDDEN_DIM, 2))
        scales = proj[..., 0]
        shifts = proj[..., 1]
        rgb_0 = self._render_frame(scales, shifts, t=0)
        rgb_1 = self._render_frame(scales, shifts, t=1)
        return rgb_0, rgb_1

    def __call__(self, pair_indices: Any) -> tuple[Any, Any]:
        """Alias for :meth:`reconstruct_pair` (default forward)."""
        return self.reconstruct_pair(pair_indices)

    def argmax_indices(self) -> Any:
        """Return ``(num_pairs, G)`` int32 argmax category indices for archive.

        The distinguishing-feature bytes per Catalog #272: the per-pair discrete
        categorical IB code stored as ``G`` int indices per pair.
        """
        return mx.argmax(self.logits, axis=-1)


__all__ = [
    "CoordMLPBaseMLX",
    "FilmProjMLX",
    "MDLIBPSJRendererMLX",
    "MDLIBPSJTrainableRendererMLX",
    "MINECriticMLX",
    "categorical_to_one_hot_mlx",
    "film_modulation_mlx",
    "gumbel_softmax_sample_mlx",
    "make_pixel_coords_mlx",
    "mine_lower_bound_mlx",
    "sinusoidal_positional_encoding_mlx",
]
