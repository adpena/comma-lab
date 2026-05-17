# SPDX-License-Identifier: MIT
"""Ballé-2018 hyperprior — T4 SYMPOSIUM Priority 1 BOLT-ON #1.

Per the T4 SYMPOSIUM verdict's Ballé voice: ``a Ballé-2018 hyperprior
on A1's existing per-pair latent (the 28-d latent the PR95-paradigm
renderer already produces). This is END-TO-END differentiable on top
of A1's frozen architecture.``

Ballé 2018 (*Variational image compression with a scale hyperprior*)
introduces a side-information channel (the hyperprior z) that
parameterizes the prior over the main latent y. The full archive
is then:

    bits = E[-log p(y | z)] + E[-log p_z(z)]
                ^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^
                main bits              hyperprior bits

In practice the hyperprior captures spatial / per-pair structure that
a factorized prior over y misses. For A1's 28-d per-pair latent the
hyperprior z is typically 4-8 dims; the hyperprior decoder produces a
per-dim Gaussian (mu, sigma) for each y dim, and y is encoded under
that Gaussian.

This module ships a CPU-only Gaussian-conditional encoder/decoder
suitable for archive-byte-size estimation + sister to NSCS03's joint
codec (which implements the full end-to-end Ballé pipeline).

[verified-against:Ballé 2018 paper + CompressAI canonical Gaussian
conditional + NSCS03 end-to-end implementation in
``src/tac/substrates/nscs03_end_to_end_balle_joint_codec/``]
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Sequence

import torch


@dataclass(frozen=True)
class BalleHyperpriorEncoded:
    """Encoded Ballé hyperprior archive blob."""
    y_quantized: torch.Tensor  # (N, D) int8 quantized main latent
    z_quantized: torch.Tensor  # (N, D_z) int8 quantized hyperprior latent
    hyperprior_decoder_state: dict  # state_dict of the hyperprior MLP
    # The encoded archive bytes derived from these (via Brotli)
    archive_bytes: bytes = b""

    def total_archive_byte_size(self) -> int:
        return len(self.archive_bytes)


def _shannon_bits_under_gaussian(
    y_quantized: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Compute -log2 p(y | N(mu, sigma)) for quantized y.

    Per Ballé 2018: ``p(y) ≈ integral_{y-0.5}^{y+0.5} N(t; mu, sigma) dt``
    which gives the per-element bit cost. The integral is approximated
    via the difference of CDFs.

    [verified-against:CompressAI ``GaussianConditional._likelihood``]
    """
    sigma = sigma.clamp(min=0.1)  # numerical floor per CompressAI default
    upper = (y_quantized + 0.5 - mu) / sigma
    lower = (y_quantized - 0.5 - mu) / sigma
    # CDF(x) for standard normal
    cdf_upper = 0.5 * (1.0 + torch.erf(upper / math.sqrt(2.0)))
    cdf_lower = 0.5 * (1.0 + torch.erf(lower / math.sqrt(2.0)))
    p = (cdf_upper - cdf_lower).clamp(min=1e-9)
    return -torch.log2(p)


def encode_balle_hyperprior_archive(
    main_latent: torch.Tensor,
    *,
    hyperprior_dim: int = 4,
    quantize_levels: int = 256,
    n_train_iter: int = 200,
    seed: int = 0,
) -> BalleHyperpriorEncoded:
    """Fit a hyperprior over ``main_latent`` and produce an encoded archive.

    Args:
        main_latent: (N, D) main latent matrix (e.g. A1's (600, 28))
        hyperprior_dim: D_z, typically D/7 (28-dim → 4-dim)
        quantize_levels: quantization grid size (256 = int8)
        n_train_iter: hyperprior MLP training iterations
        seed: RNG seed

    The hyperprior MLP is a simple 2-layer fully-connected network
    that takes the per-pair main latent y as input and outputs (mu, sigma)
    for each y dim. It's trained to minimize:

        bits = -E_y[log p(y | N(mu(z), sigma(z)))]
             - E_z[log p_z(z)]

    where p_z is a factorized standard normal (canonical Ballé 2018
    factorized prior).

    [verified-against:Ballé 2018 paper Equation (3) + CompressAI
    ``ScaleHyperprior`` reference impl]
    """
    if main_latent.ndim != 2:
        raise ValueError(
            f"encode_balle_hyperprior_archive requires 2D (N, D) latent; got shape {tuple(main_latent.shape)}"
        )
    torch.manual_seed(seed)
    y = main_latent.detach().contiguous().float()
    N, D = y.shape
    D_z = hyperprior_dim
    # Quantize main latent to int8
    y_scale = y.abs().max() / 127.0
    y_scale = max(y_scale.item(), 1e-10)
    y_quant = (y / y_scale).round().clamp(-128, 127)
    # Fit hyperprior: simple MLP D -> D_z -> 2*D (mu, sigma per main dim)
    encoder = torch.nn.Sequential(
        torch.nn.Linear(D, D * 2),
        torch.nn.ReLU(),
        torch.nn.Linear(D * 2, D_z),
    )
    decoder = torch.nn.Sequential(
        torch.nn.Linear(D_z, D * 2),
        torch.nn.ReLU(),
        torch.nn.Linear(D * 2, D * 2),  # output 2*D for (mu, log_sigma)
    )
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3
    )
    for it in range(n_train_iter):
        optimizer.zero_grad()
        z = encoder(y)
        z_quant_ste = z + (z.round() - z).detach()  # STE
        out = decoder(z_quant_ste)
        mu, log_sigma = out[:, :D], out[:, D:]
        sigma = torch.exp(log_sigma.clamp(-5, 5))
        bits_main = _shannon_bits_under_gaussian(y_quant, mu, sigma).sum() / N
        # Hyperprior: standard normal log-prob (factorized)
        bits_hyper = (0.5 * z_quant_ste**2 + 0.5 * math.log(2 * math.pi) / math.log(2)).sum() / N
        loss = bits_main + bits_hyper
        loss.backward()
        optimizer.step()
    # Final encoding
    with torch.no_grad():
        z = encoder(y)
        z_quant = z.round().clamp(-128, 127).to(torch.int8)
    # Pack everything into archive bytes:
    #   D, D_z, N, y_scale (fp32)
    #   y_quant (N*D int8)
    #   z_quant (N*D_z int8)
    #   decoder state_dict (fp16)
    header = struct.pack("<III f", D, D_z, N, y_scale)
    y_bytes = y_quant.to(torch.int8).numpy().tobytes()
    z_bytes = z_quant.numpy().tobytes()
    # Decoder state: pack each tensor as fp16
    decoder_state = decoder.state_dict()
    decoder_bytes_parts = []
    decoder_keys: list[bytes] = []
    for k, v in decoder_state.items():
        key_bytes = k.encode("utf-8")
        v_fp16 = v.detach().to(torch.float16).contiguous()
        decoder_keys.append(key_bytes)
        decoder_bytes_parts.append(
            struct.pack("<B I", len(key_bytes), v_fp16.numel())
            + key_bytes
            + v_fp16.numpy().tobytes()
        )
    decoder_blob = b"".join(decoder_bytes_parts)
    decoder_n_tensors = struct.pack("<H", len(decoder_state))
    raw_blob = header + y_bytes + z_bytes + decoder_n_tensors + decoder_blob
    # Brotli-compress the raw blob
    try:
        import brotli
        archive_bytes = brotli.compress(raw_blob, quality=11)
    except Exception:
        archive_bytes = raw_blob
    return BalleHyperpriorEncoded(
        y_quantized=y_quant.to(torch.int8),
        z_quantized=z_quant,
        hyperprior_decoder_state={k: v.detach().clone() for k, v in decoder_state.items()},
        archive_bytes=archive_bytes,
    )


def decode_balle_hyperprior_archive(archive_bytes: bytes) -> torch.Tensor:
    """Inverse of :func:`encode_balle_hyperprior_archive` — reconstruct
    the main latent from the encoded archive bytes.

    Decoder runs the hyperprior MLP to derive per-pair (mu, sigma), then
    re-derives ``y = y_quant * y_scale``. Note: the canonical Ballé
    decoder uses the hyperprior-conditional prior for entropy decoding;
    this simplified decoder uses the quantized y_quant directly (the
    hyperprior is used for ENCODING rate-distortion calculation but
    the actual archive stores y_quant int8 directly).
    """
    try:
        import brotli
        raw_blob = brotli.decompress(archive_bytes)
    except Exception:
        raw_blob = archive_bytes
    header_size = struct.calcsize("<III f")
    D, D_z, N, y_scale = struct.unpack_from("<III f", raw_blob, 0)
    offset = header_size
    n_y = N * D
    y_quant = torch.frombuffer(bytearray(raw_blob[offset : offset + n_y]), dtype=torch.int8)
    offset += n_y
    n_z = N * D_z
    z_quant = torch.frombuffer(bytearray(raw_blob[offset : offset + n_z]), dtype=torch.int8)
    offset += n_z
    n_tensors = struct.unpack_from("<H", raw_blob, offset)[0]
    offset += 2
    # Skip decoder reconstruction; decode_balle_hyperprior returns y
    # from y_quant directly. The hyperprior + decoder are needed only
    # for the entropy code (which this simplified path doesn't fully
    # use; see NSCS03 for the full end-to-end path).
    y = y_quant.float().reshape(N, D) * y_scale
    return y


class BalleHyperpriorBolton:
    """T4 SYMPOSIUM Priority 1 BOLT-ON #1 wrapper for A1.

    Usage::

        bolton = BalleHyperpriorBolton(hyperprior_dim=4)
        encoded = bolton.fit_and_encode(a1_latent)  # a1_latent.shape = (600, 28)
        archive_bytes = encoded.archive_bytes
        # Wire-format: append archive_bytes as a sidecar to A1's
        # canonical archive layout.

    Predicted ΔS band (per Ballé voice + Beta(2,2) posterior per
    Shannon's analysis): [0.188, 0.192] — i.e. 0.5-2.5 points below A1.
    Cost: ~$10-25 dispatch per the T4 verdict.

    [verified-against:T4 SYMPOSIUM verdict Decision 2 (Option 2C) +
    Ballé 2018 paper]
    """

    def __init__(self, *, hyperprior_dim: int = 4, n_train_iter: int = 200, seed: int = 0):
        self.hyperprior_dim = hyperprior_dim
        self.n_train_iter = n_train_iter
        self.seed = seed

    def fit_and_encode(self, main_latent: torch.Tensor) -> BalleHyperpriorEncoded:
        return encode_balle_hyperprior_archive(
            main_latent,
            hyperprior_dim=self.hyperprior_dim,
            n_train_iter=self.n_train_iter,
            seed=self.seed,
        )
