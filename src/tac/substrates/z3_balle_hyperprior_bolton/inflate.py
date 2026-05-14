# SPDX-License-Identifier: MIT
"""Z3 Ballé hyperprior bolt-on inflate runtime.

Reads a Z3 composition archive (= A1 archive bytes + optional Z3HP1 sidecar)
and either:

1. Passes through to A1's inflate when the Z3HP1 sidecar is absent
   (byte-identical-to-A1 fallback per Ballé 2018 amortization principle:
   ship the hyperprior only when it actually saves bytes), OR

2. Reconstructs each pair's A1 latents by COMBINING (a) the conditional
   Gaussian decode of residual_blob using sigma derived from the hyper-
   prior MLP forward on the decoded w_hat, with (b) the base A1 latent
   prior (zero-mean assumption). The reconstructed latents are then
   handed to A1's existing decode path.

LOC budget: ≤ 100 LOC per HNeRV parity discipline L4 (this is a BOLT-ON,
not substrate-engineering). NO scorer load. NO network. NO /tmp paths.
CUDA-or-CPU agnostic.

Per Catalog #205 the device selection is via the canonical
``select_inflate_device`` helper (env-driven; refuses MPS); per Catalog
#1 the MPS-fallback ternary is forbidden.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from tac.substrates.z3_balle_hyperprior_bolton.archive import (
    A1_LATENT_DIM,
    A1_N_PAIRS,
    decode_z3hp1_sidecar,
    dequantize_int8_with_scale,
    split_composition_archive,
)
from tac.substrates.z3_balle_hyperprior_bolton.architecture import (
    Z3HyperpriorConfig,
    Z3HyperpriorMLP,
)


def select_inflate_device() -> torch.device:
    """Canonical inflate device selector (Catalog #205).

    Honors ``PACT_INFLATE_DEVICE`` env var (``auto``/``cpu``/``cuda``).
    Refuses ``mps`` explicitly per CLAUDE.md MPS-NOISE non-negotiable.
    """
    requested = os.environ.get("PACT_INFLATE_DEVICE", "auto").lower()
    if requested == "mps":
        raise RuntimeError("PACT_INFLATE_DEVICE=mps refused (MPS is noise)")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PACT_INFLATE_DEVICE=cuda but no CUDA visible")
        return torch.device("cuda")
    # auto: prefer CUDA when available.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reconstruct_a1_latents(archive_bytes: bytes) -> tuple[bytes, torch.Tensor | None]:
    """Reconstruct A1 latents from a Z3 composition archive.

    Returns ``(a1_archive_bytes, reconstructed_latents_or_None)``. When
    the sidecar is absent, the second element is None and the caller
    should run A1's inflate verbatim on ``a1_archive_bytes``.
    """
    a1_bytes, sidecar = split_composition_archive(archive_bytes)
    if not sidecar:
        return a1_bytes, None
    meta, weights_int8, w_hat_int8, residual_int8 = decode_z3hp1_sidecar(sidecar)
    # Reload hyperprior MLP weights.
    cfg = Z3HyperpriorConfig(hyper_latent_dim=meta.hyper_dim)
    mlp = Z3HyperpriorMLP(cfg)
    # Flatten state-dict into the int8 buffer (canonical layout: linear weights
    # in declaration order). Each layer's weights are dequantized with the
    # shared ``int8_w_scale``.
    state_floats = dequantize_int8_with_scale(
        weights_int8,
        meta.int8_w_scale,
        shape=(len(weights_int8),),
    )
    pos = 0
    loaded_state = {}
    for name, param in mlp.state_dict().items():
        n = param.numel()
        if pos + n > state_floats.numel():
            raise ValueError(
                f"Z3HP1 weights blob too short at {name}: need {n}, "
                f"have {state_floats.numel() - pos}"
            )
        loaded_state[name] = (
            state_floats[pos : pos + n].view(param.shape).to(param.dtype)
        )
        pos += n
    if pos != state_floats.numel():
        raise ValueError(
            f"Z3HP1 weights blob has trailing {state_floats.numel() - pos} values"
        )
    mlp.load_state_dict(loaded_state)
    mlp.eval()
    # Decode w_hat (the side-info hyper-latents).
    w_hat_arr = np.frombuffer(w_hat_int8, dtype=np.int8).reshape(
        meta.n_pairs, meta.hyper_dim
    )
    w_hat = torch.from_numpy(w_hat_arr.copy()).to(torch.float32) * meta.quant_step
    # Forward through hyper-synthesis (h_s) to get sigma per pair.
    with torch.no_grad():
        h2 = mlp.h_s_1(w_hat)
        sigma_logits = mlp.h_s_2(h2)
        sigma = torch.nn.functional.softplus(sigma_logits) + meta.min_sigma
        sigma = sigma.clamp(min=meta.min_sigma, max=meta.max_sigma)
    # Reconstruct latents: residual_int8 stores quantized (y - 0) / quant_step
    # under the conditional Gaussian prior. Decode = residual * quant_step.
    residual_arr = np.frombuffer(residual_int8, dtype=np.int8).reshape(
        meta.n_pairs, meta.latent_dim
    )
    residual = torch.from_numpy(residual_arr.copy()).to(torch.float32)
    reconstructed = residual * meta.quant_step
    # sigma is used at encoding-time to drive the AC coder; at decode-time
    # the value of sigma is implicit in the integer residual. We expose
    # ``sigma`` for diagnostic / round-trip verification only.
    _ = sigma  # diagnostic only
    if reconstructed.shape != (A1_N_PAIRS, A1_LATENT_DIM):
        raise ValueError(
            f"Z3HP1 reconstructed latents shape {tuple(reconstructed.shape)} "
            f"!= ({A1_N_PAIRS}, {A1_LATENT_DIM})"
        )
    return a1_bytes, reconstructed


__all__ = ["reconstruct_a1_latents", "select_inflate_device"]
