# SPDX-License-Identifier: MIT
"""numpy_reference — Numpy sister reference implementation for J=MDL-IBPS.

Per AMENDMENT #3 axis 3 (portability via numpy per primitive): every MLX
primitive in `mlx_renderer.py` has a sister numpy reference implementation
here. This module is PORTABLE to non-Apple-Silicon CPU-only test rigs
without MLX dependency.

Enables:
- (a) GHA CPU CI testing without MLX install
- (b) sister cathedral consumer cross-validation
- (c) operator-portable diagnostic on non-Apple-Silicon hardware

Test parity contract: MLX <-> numpy <= 1e-5 per primitive (tighter than
MLX <-> PyTorch <= 0.001 because both numpy and IEEE-754 reference reduce
through the same fp32 arithmetic).

All primitives match the architecture declared in `__init__.py` module
docstring; see `mlx_renderer.py` for MLX-side counterparts.

Per CLAUDE.md FORBIDDEN_PATTERNS:
- No silent device defaults (numpy is CPU-only by construction)
- No scorer load (this module renders only; loss + scorer routing live elsewhere)
- No /tmp paths in persisted artifacts
"""

from __future__ import annotations

import math

import numpy as np

from tac.substrates.mdl_ibps_j_discrete_categorical_mine_hybrid import (
    CATEGORICAL_G,
    CATEGORICAL_K,
    EVAL_HW,
    HIDDEN_DIM,
    MINE_HIDDEN_DIM,
    NUM_HIDDEN_LAYERS,
    POS_DIM,
)


def sinusoidal_positional_encoding_numpy(
    coords: np.ndarray, pos_dim: int = POS_DIM
) -> np.ndarray:
    """Sinusoidal positional encoding (NeRF-style).

    Args:
        coords: ``(N, 3)`` array of (x, y, t) coordinates in [0, 1].
        pos_dim: number of frequency bands per coordinate axis.

    Returns:
        ``(N, pos_dim * 2 * 3)`` array; for each (x, y, t) we emit
        [sin(2^k * pi * c), cos(2^k * pi * c)] for k in 0..pos_dim-1, c in (x, y, t).
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"coords must have shape (N, 3); got {coords.shape}"
        )
    freqs = (2.0 ** np.arange(pos_dim, dtype=np.float32)) * math.pi
    # (N, 3) -> (N, 3, pos_dim) via outer product
    scaled = coords.astype(np.float32)[:, :, None] * freqs[None, None, :]
    sins = np.sin(scaled)
    coss = np.cos(scaled)
    # Concatenate sin + cos along the freq axis: (N, 3, 2 * pos_dim)
    encoded = np.concatenate([sins, coss], axis=-1)
    # Flatten last two axes: (N, 3 * 2 * pos_dim)
    return encoded.reshape(coords.shape[0], -1).astype(np.float32)


def film_modulation_numpy(
    h: np.ndarray, scale: np.ndarray, shift: np.ndarray
) -> np.ndarray:
    """FiLM modulation: h * scale + shift (Perez et al. 2017).

    Args:
        h: ``(N, HIDDEN_DIM)`` hidden activations.
        scale: ``(N, HIDDEN_DIM)`` per-sample scale factor.
        shift: ``(N, HIDDEN_DIM)`` per-sample shift factor.

    Returns:
        ``(N, HIDDEN_DIM)`` modulated activations.
    """
    if not (h.shape == scale.shape == shift.shape):
        raise ValueError(
            f"h/scale/shift must have matching shapes; got "
            f"h={h.shape}, scale={scale.shape}, shift={shift.shape}"
        )
    return h * scale + shift


def categorical_to_one_hot_numpy(
    indices: np.ndarray, K: int = CATEGORICAL_K
) -> np.ndarray:
    """Categorical index -> one-hot encoding.

    Args:
        indices: ``(B, G)`` int array of categorical indices in [0, K).
        K: alphabet size (default CATEGORICAL_K).

    Returns:
        ``(B, G * K)`` flattened one-hot vector per sample.
    """
    if indices.ndim != 2:
        raise ValueError(f"indices must be 2D (B, G); got {indices.shape}")
    if np.any(indices < 0) or np.any(indices >= K):
        raise ValueError(
            f"indices out of range; must be in [0, {K}); got "
            f"min={indices.min()} max={indices.max()}"
        )
    B, G = indices.shape
    one_hot = np.zeros((B, G, K), dtype=np.float32)
    # Use np.arange for B and G axes; indices[i, j] selects the position
    b_arange = np.arange(B)[:, None]
    g_arange = np.arange(G)[None, :]
    one_hot[b_arange, g_arange, indices] = 1.0
    return one_hot.reshape(B, G * K).astype(np.float32)


class CoordMLPBaseNumpy:
    """Procedural coord-MLP base with FiLM modulation (numpy reference).

    Architecture per __init__.py docstring:
        Input: (x, y, t) -> sinusoidal positional encoding -> R^(3*2*POS_DIM)
        Hidden layers: NUM_HIDDEN_LAYERS x HIDDEN_DIM with FiLM modulation
        Output: linear -> R^3 -> sigmoid -> rgb in [0, 1]^3
    """

    def __init__(
        self,
        weights_first: np.ndarray,
        biases_first: np.ndarray,
        weights_hidden: list[np.ndarray],
        biases_hidden: list[np.ndarray],
        weights_out: np.ndarray,
        biases_out: np.ndarray,
    ) -> None:
        """Initialize with explicit weight arrays for deterministic testing.

        Args:
            weights_first: ``(POS_FEAT_DIM, HIDDEN_DIM)`` input layer.
            biases_first: ``(HIDDEN_DIM,)`` input bias.
            weights_hidden: list of ``(HIDDEN_DIM, HIDDEN_DIM)`` per hidden layer.
            biases_hidden: list of ``(HIDDEN_DIM,)`` per hidden layer.
            weights_out: ``(HIDDEN_DIM, 3)`` output layer.
            biases_out: ``(3,)`` output bias.
        """
        self.weights_first = weights_first.astype(np.float32)
        self.biases_first = biases_first.astype(np.float32)
        self.weights_hidden = [w.astype(np.float32) for w in weights_hidden]
        self.biases_hidden = [b.astype(np.float32) for b in biases_hidden]
        self.weights_out = weights_out.astype(np.float32)
        self.biases_out = biases_out.astype(np.float32)
        if len(self.weights_hidden) != NUM_HIDDEN_LAYERS:
            raise ValueError(
                f"expected {NUM_HIDDEN_LAYERS} hidden weight matrices; got "
                f"{len(self.weights_hidden)}"
            )

    def forward(
        self,
        coords: np.ndarray,
        film_scales: np.ndarray,
        film_shifts: np.ndarray,
    ) -> np.ndarray:
        """Forward pass.

        Args:
            coords: ``(N, 3)`` pixel-coord tensor (x, y, t in [0, 1]).
            film_scales: ``(N, NUM_HIDDEN_LAYERS, HIDDEN_DIM)`` FiLM scale.
            film_shifts: ``(N, NUM_HIDDEN_LAYERS, HIDDEN_DIM)`` FiLM shift.

        Returns:
            ``(N, 3)`` rgb in [0, 1].
        """
        x = sinusoidal_positional_encoding_numpy(coords, pos_dim=POS_DIM)
        h = x @ self.weights_first + self.biases_first
        # Apply NUM_HIDDEN_LAYERS hidden layers with FiLM modulation
        for layer_idx in range(NUM_HIDDEN_LAYERS):
            scale = film_scales[:, layer_idx, :]
            shift = film_shifts[:, layer_idx, :]
            h = film_modulation_numpy(h, scale, shift)
            h = np.sin(h)
            if layer_idx < NUM_HIDDEN_LAYERS - 1:
                # Apply hidden weight matrix
                h = h @ self.weights_hidden[layer_idx] + self.biases_hidden[layer_idx]
        # Output projection
        rgb = h @ self.weights_out + self.biases_out
        # Sigmoid clamp to [0, 1]
        rgb = 1.0 / (1.0 + np.exp(-rgb))
        return rgb.astype(np.float32)


def film_proj_numpy(
    one_hot: np.ndarray,
    weights_proj: np.ndarray,
    biases_proj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one-hot categorical -> FiLM (scale, shift) parameters.

    Args:
        one_hot: ``(B, G * K)`` flattened one-hot vector.
        weights_proj: ``(G * K, NUM_HIDDEN_LAYERS * HIDDEN_DIM * 2)`` projection.
        biases_proj: ``(NUM_HIDDEN_LAYERS * HIDDEN_DIM * 2,)`` projection bias.

    Returns:
        (scales, shifts) where each is ``(B, NUM_HIDDEN_LAYERS, HIDDEN_DIM)``.
    """
    if one_hot.ndim != 2:
        raise ValueError(f"one_hot must be 2D (B, G*K); got {one_hot.shape}")
    B = one_hot.shape[0]
    expected_dim = CATEGORICAL_G * CATEGORICAL_K
    if one_hot.shape[1] != expected_dim:
        raise ValueError(
            f"expected one_hot of shape (B, {expected_dim}); got {one_hot.shape}"
        )
    proj = one_hot @ weights_proj + biases_proj
    expected_proj_dim = NUM_HIDDEN_LAYERS * HIDDEN_DIM * 2
    if proj.shape[1] != expected_proj_dim:
        raise ValueError(
            f"projection output must be (B, {expected_proj_dim}); got {proj.shape}"
        )
    proj = proj.reshape(B, NUM_HIDDEN_LAYERS, HIDDEN_DIM, 2)
    scales = proj[..., 0].astype(np.float32)
    shifts = proj[..., 1].astype(np.float32)
    return scales, shifts


def mine_critic_forward_numpy(
    z: np.ndarray,
    frames_features: np.ndarray,
    weights_critic: list[np.ndarray],
    biases_critic: list[np.ndarray],
) -> np.ndarray:
    """MINE critic forward (Belghazi 2018).

    Critic structure: 2-layer MLP with ReLU activation.

    Args:
        z: ``(B, G * K)`` flattened one-hot latent.
        frames_features: ``(B, D_features)`` compact frame features.
        weights_critic: list of 3 weight matrices [W_in, W_hidden, W_out].
        biases_critic: list of 3 bias vectors.

    Returns:
        ``(B,)`` critic output T_theta(z, frames) (real-valued).
    """
    if z.ndim != 2 or frames_features.ndim != 2:
        raise ValueError(
            f"z and frames_features must be 2D; got z={z.shape}, frames={frames_features.shape}"
        )
    if z.shape[0] != frames_features.shape[0]:
        raise ValueError(
            f"batch mismatch; z={z.shape[0]}, frames={frames_features.shape[0]}"
        )
    if len(weights_critic) != 3 or len(biases_critic) != 3:
        raise ValueError(
            f"MINE critic must have 3 weight matrices + 3 biases; got "
            f"{len(weights_critic)} weights, {len(biases_critic)} biases"
        )
    # Concatenate inputs
    h = np.concatenate([z, frames_features], axis=-1)
    # Hidden 1
    h = h @ weights_critic[0] + biases_critic[0]
    h = np.maximum(0.0, h)  # ReLU
    # Hidden 2
    h = h @ weights_critic[1] + biases_critic[1]
    h = np.maximum(0.0, h)  # ReLU
    # Output (scalar critic value per sample)
    out = h @ weights_critic[2] + biases_critic[2]
    return out.squeeze(-1).astype(np.float32)


def mine_lower_bound_numpy(
    critic_joint: np.ndarray, critic_marginal: np.ndarray
) -> float:
    """Compute MINE Donsker-Varadhan lower bound on mutual information.

    Formula: I(z; frames) >= E_p(z, frames)[T(z, frames)] - log E_p(z) p(frames) [exp T(z, frames)]

    Args:
        critic_joint: ``(B,)`` critic outputs on joint samples (z, frames) drawn from joint p(z, frames).
        critic_marginal: ``(B,)`` critic outputs on marginal samples (z', frames) where z' is shuffled.

    Returns:
        scalar lower bound on I(z; frames) in nats.
    """
    if critic_joint.shape != critic_marginal.shape:
        raise ValueError(
            f"critic_joint and critic_marginal must have matching shape; got "
            f"{critic_joint.shape} vs {critic_marginal.shape}"
        )
    # Stable log-sum-exp for E[exp(T)]: subtract max for numerical stability
    max_marginal = np.max(critic_marginal)
    log_mean_exp = max_marginal + np.log(
        np.mean(np.exp(critic_marginal - max_marginal))
    )
    return float(np.mean(critic_joint) - log_mean_exp)


def kl_gaussian_to_standard_normal_numpy(
    mu: np.ndarray, logvar: np.ndarray
) -> np.ndarray:
    """KL(N(mu, sigma^2) || N(0, I)) per-sample.

    Sister formula to parent C6 ``tac.substrates.c6_e4_mdl_ibps.mdl_loss.kl_gaussian_to_standard_normal``
    (PyTorch); preserved here for comparison-tests (CC-J-4 unwind documented MINE vs KL upper bound).

    Args:
        mu: posterior mean ``(B, d_z)``.
        logvar: posterior log-variance ``(B, d_z)``.

    Returns:
        kl per sample ``(B,)`` in nats.
    """
    if mu.shape != logvar.shape:
        raise ValueError(
            f"mu and logvar must have matching shape; got "
            f"{mu.shape} vs {logvar.shape}"
        )
    if mu.ndim != 2:
        raise ValueError(f"mu must be 2D (B, d_z); got {mu.shape}")
    # Stable: 0.5 * (mu^2 + exp(logvar) - logvar - 1)
    return 0.5 * (mu ** 2 + np.exp(logvar) - logvar - 1.0).sum(axis=-1)


def sparse_laplacian_l1_numpy(matrices: list[np.ndarray]) -> float:
    """Sparse-Laplacian L1 regularizer (MacKay 2003 ch. 28).

    Encourages dimension-wise sparsity in FiLM modulation matrices.

    Args:
        matrices: list of np.ndarray (FiLM weight matrices).

    Returns:
        scalar L1 norm sum.
    """
    return float(sum(np.abs(m).sum() for m in matrices))


def make_pixel_coords_numpy(
    height: int = EVAL_HW[0], width: int = EVAL_HW[1], t: int = 0
) -> np.ndarray:
    """Generate (height * width, 3) pixel coords (x, y, t) normalized to [0, 1].

    Args:
        height: image height (default EVAL_HW[0] = 384).
        width: image width (default EVAL_HW[1] = 512).
        t: time index (0 or 1 for two-frame pair).

    Returns:
        ``(height * width, 3)`` coords.
    """
    if t not in (0, 1):
        raise ValueError(f"t must be 0 or 1; got {t}")
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    grid_t = np.full_like(grid_x, float(t))
    return np.stack([grid_x.flatten(), grid_y.flatten(), grid_t.flatten()], axis=-1)


__all__ = [
    "CoordMLPBaseNumpy",
    "categorical_to_one_hot_numpy",
    "film_modulation_numpy",
    "film_proj_numpy",
    "kl_gaussian_to_standard_normal_numpy",
    "make_pixel_coords_numpy",
    "mine_critic_forward_numpy",
    "mine_lower_bound_numpy",
    "sinusoidal_positional_encoding_numpy",
    "sparse_laplacian_l1_numpy",
]
