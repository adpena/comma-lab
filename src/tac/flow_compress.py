# SPDX-License-Identifier: MIT
"""DCT-based flow compression for archive storage (Fridrich's proposal).

Compresses dense per-pair optical flow into low-rank DCT coefficients.
At inflate time, reconstructs the dense flow from stored coefficients.

Archive cost: n_pairs × n_coeffs × 2 channels × 2 bytes (float16)
  With 32 coefficients: 600 × 32 × 2 × 2 = 76.8 KB

Usage (compress time):
    flow = torch.load("raft_flow.pt")["flow"]  # (600, 2, H, W)
    coeffs = compress_flow_dct(flow, n_coeffs=32)
    torch.save(coeffs, "flow_dct.pt")

Usage (inflate time):
    coeffs = torch.load("flow_dct.pt")
    flow = decompress_flow_dct(coeffs, H=384, W=512)
"""

from __future__ import annotations

import torch


def compress_flow_dct(
    flow: torch.Tensor,
    n_coeffs: int = 32,
) -> dict:
    """Compress dense flow fields using 2D DCT truncation.

    Args:
        flow: (N, 2, H, W) dense flow in normalized coordinates
        n_coeffs: number of DCT coefficients to keep per channel

    Returns:
        dict with "coeffs_x", "coeffs_y" (N, n_coeffs) float16,
        "basis_indices" (n_coeffs, 2) int16, and metadata
    """
    N, C, H, W = flow.shape
    assert C == 2

    # Compute 2D DCT basis (using cosine transform via matmul)
    # DCT-II basis for rows and columns — moved to flow device to avoid CUDA crash
    dev = flow.device
    dct_h = _dct_matrix(H).to(dev)  # (H, H)
    dct_w = _dct_matrix(W).to(dev)  # (W, W)

    # Find the n_coeffs with largest average energy across all pairs
    # First compute DCT of all flow fields
    flow_x = flow[:, 0]  # (N, H, W)
    flow_y = flow[:, 1]  # (N, H, W)

    dct_x = dct_h @ flow_x @ dct_w.T  # (N, H, W) DCT coefficients
    dct_y = dct_h @ flow_y @ dct_w.T

    # Average energy across pairs for each coefficient position
    energy = (dct_x.pow(2).mean(0) + dct_y.pow(2).mean(0))  # (H, W)

    # Select top-k coefficient positions
    flat_energy = energy.reshape(-1)
    _, top_indices = flat_energy.topk(n_coeffs)
    basis_h = top_indices // W
    basis_w = top_indices % W
    basis_indices = torch.stack([basis_h, basis_w], dim=1)  # (n_coeffs, 2)

    # Extract coefficients at selected positions (must be on same device as dct_x/dct_y)
    coeffs_x = torch.zeros(N, n_coeffs, device=dev)
    coeffs_y = torch.zeros(N, n_coeffs, device=dev)
    for k in range(n_coeffs):
        hi, wi = basis_indices[k]
        coeffs_x[:, k] = dct_x[:, hi, wi]
        coeffs_y[:, k] = dct_y[:, hi, wi]

    return {
        "coeffs_x": coeffs_x.half(),
        "coeffs_y": coeffs_y.half(),
        "basis_indices": basis_indices.short(),
        "n_pairs": N,
        "n_coeffs": n_coeffs,
        "resolution": (H, W),
    }


def decompress_flow_dct(
    compressed: dict,
    H: int | None = None,
    W: int | None = None,
) -> torch.Tensor:
    """Reconstruct dense flow from DCT coefficients.

    Args:
        compressed: output of compress_flow_dct
        H, W: override resolution (defaults to compressed["resolution"])

    Returns:
        (N, 2, H, W) float32 flow in normalized coordinates
    """
    if H is None or W is None:
        H, W = compressed["resolution"]

    coeffs_x = compressed["coeffs_x"].float()  # (N, n_coeffs)
    coeffs_y = compressed["coeffs_y"].float()
    basis_indices = compressed["basis_indices"].long()  # (n_coeffs, 2)
    N = compressed["n_pairs"]
    n_coeffs = compressed["n_coeffs"]
    # All tensors on same device as coefficients
    dev = coeffs_x.device

    # Reconstruct DCT coefficient grid (sparse)
    dct_x = torch.zeros(N, H, W, device=dev)
    dct_y = torch.zeros(N, H, W, device=dev)
    for k in range(n_coeffs):
        hi, wi = basis_indices[k]
        dct_x[:, hi, wi] = coeffs_x[:, k]
        dct_y[:, hi, wi] = coeffs_y[:, k]

    # Inverse DCT (on same device as data)
    idct_h = _dct_matrix(H).T.to(dev)
    idct_w = _dct_matrix(W).T.to(dev)

    flow_x = idct_h @ dct_x @ idct_w.T
    flow_y = idct_h @ dct_y @ idct_w.T

    return torch.stack([flow_x, flow_y], dim=1)  # (N, 2, H, W)


def _dct_matrix(N: int) -> torch.Tensor:
    """Orthonormal DCT-II basis matrix."""
    n = torch.arange(N, dtype=torch.float32)
    k = torch.arange(N, dtype=torch.float32)
    basis = torch.cos(torch.pi * (n.unsqueeze(0) + 0.5) * k.unsqueeze(1) / N)
    # Normalize: first row by 1/sqrt(N), rest by sqrt(2/N)
    basis[0] *= 1.0 / (N ** 0.5)
    basis[1:] *= (2.0 / N) ** 0.5
    return basis
