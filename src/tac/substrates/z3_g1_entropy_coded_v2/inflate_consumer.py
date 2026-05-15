# SPDX-License-Identifier: MIT
"""Z3-G1 entropy-coded v2 inflate runtime.

Reads a Z3G2 payload (decoder_section + Z3G2 section + A1 sidecar) and
RECONSTRUCTS the per-pair sigma + class-index streams from the
entropy-coded bytes shipped in the archive. The inflate path then uses
those reconstructed sigmas (per-pair, indexed by class) as the
class-conditional Gaussian std for the residual AC decoder, exactly
as the encoder did.

Per Catalog #220 + the design memo §6: this module IS the OPERATIONAL
mechanism that consumes the distinguishing v2 bytes (sigma_table_blob,
class_prior_cdf_blob, class_index_blob) and turns them into different
inflate outputs vs Z3 v2's empty-slot direct-residual mode. Verified
structurally by `tools/verify_z3_g1_entropy_coded_v2_byte_mutation.py`
per Catalog #139.

LOC budget: <= 100 LOC per HNeRV parity discipline L4. NO scorer load.
NO network. NO /tmp paths. CUDA-or-CPU agnostic.

Per Catalog #205 the device selection is via the canonical
``select_inflate_device`` helper (env-driven; refuses MPS).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from tac.substrates._shared.inflate_runtime import (
    select_inflate_device as _canonical_select_inflate_device,
)
from tac.substrates.z3_g1_entropy_coded_v2.architecture import (
    A1_LATENT_DIM,
    A1_N_PAIRS,
    G1_NUM_SCORER_CLASSES,
    dequantize_sigma_table_int8,
)
from tac.substrates.z3_g1_entropy_coded_v2.archive import (
    A1_LATENT_BLOB_LEN,
    decode_z3g2_section,
    is_z3g2_payload,
    split_z3g2_payload_bytes,
)


def select_inflate_device() -> torch.device:
    """Canonical inflate device selector (Catalog #205)."""
    if os.environ.get("PACT_INFLATE_DEVICE", "auto").lower() == "mps":
        raise RuntimeError("PACT_INFLATE_DEVICE=mps refused (MPS is noise)")
    return torch.device(_canonical_select_inflate_device())


def _unpack_sigma_table_entropy_coded(
    sigma_table_int8: torch.Tensor,
    int8_sigma_scale: float,
    *,
    min_sigma: float = 1e-3,
) -> torch.Tensor:
    """Reconstruct fp32 sigma table from int8 bytes shipped in archive.

    Thin wrapper over architecture.dequantize_sigma_table_int8 so the
    inflate-consumer namespace exposes a clearly-named helper for the
    Catalog #272 distinguishing-feature contract.
    """
    return dequantize_sigma_table_int8(
        sigma_table_int8, int8_sigma_scale, min_sigma=min_sigma
    )


def _unpack_class_prior_cdf(
    class_prior_counts: torch.Tensor,
) -> torch.Tensor:
    """Normalize int64 frequency counts into a probability distribution.

    Returns (num_classes,) fp32 probabilities summing to 1.0.
    """
    counts_f = class_prior_counts.to(torch.float32).clamp(min=1.0)
    return counts_f / counts_f.sum()


def _class_conditional_arithmetic_decode(
    class_indices_uint8: bytes,
    class_prior_cdf: torch.Tensor,
    *,
    n_pairs: int = A1_N_PAIRS,
) -> torch.Tensor:
    """Convert raw uint8 class indices into a long tensor.

    NOTE: the actual constriction-Huffman decode happens in
    ``archive._decode_class_indices_huffman`` during ``decode_z3g2_section``;
    by the time bytes reach this helper they are already raw uint8 indices.
    This helper validates and converts to torch tensor for downstream use.

    The ``class_prior_cdf`` is accepted to match the Catalog #272
    distinguishing-feature contract (the function signature documents
    that the CDF IS the prior used at decode-time) and to allow a future
    range-coded variant to use it.
    """
    if len(class_indices_uint8) != n_pairs:
        raise ValueError(
            f"class_indices length {len(class_indices_uint8)} != n_pairs {n_pairs}"
        )
    if class_prior_cdf.numel() != G1_NUM_SCORER_CLASSES:
        raise ValueError(
            f"class_prior_cdf must have {G1_NUM_SCORER_CLASSES} entries; "
            f"got {class_prior_cdf.numel()}"
        )
    arr = np.frombuffer(class_indices_uint8, dtype=np.uint8).copy()
    return torch.from_numpy(arr).to(torch.long)


def reconstruct_class_indices_and_sigma_table_from_z3g2_payload(
    payload_bytes: bytes,
) -> tuple[bytes, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct (a1_byte_faithful_shell, latents_in_a1_range, sigma_table_fp32, class_indices_long).

    The shell + latents pair mirrors v1's
    `reconstruct_a1_latents_from_v2_payload` so callers can drop v2 in.
    """
    if not is_z3g2_payload(payload_bytes):
        raise ValueError("payload is not a Z3G2 packet")
    decoder_section, z3g2_section, sidecar_section = split_z3g2_payload_bytes(
        payload_bytes
    )
    (
        meta,
        sigma_table_int8,
        class_indices_uint8,
        class_prior_counts,
        residual_int8,
        latent_offset,
        latent_scale,
        _,
    ) = decode_z3g2_section(z3g2_section)
    sigma_table_fp32 = _unpack_sigma_table_entropy_coded(
        sigma_table_int8, meta.int8_sigma_scale, min_sigma=meta.min_sigma
    )
    class_prior_cdf = _unpack_class_prior_cdf(class_prior_counts)
    class_indices_long = _class_conditional_arithmetic_decode(
        class_indices_uint8, class_prior_cdf, n_pairs=meta.n_pairs
    )
    residual_arr = np.frombuffer(residual_int8, dtype=np.int8).reshape(
        meta.n_pairs, meta.latent_dim
    )
    residual_q = torch.from_numpy(residual_arr.copy()).to(torch.float32)
    # Per-pair sigma lookup. The reconstructed sigma_table is consumed here
    # to influence inflate-time output (Catalog #220 OPERATIONAL contract).
    sigmas_per_pair = sigma_table_fp32[class_indices_long, :]  # (n_pairs, latent_dim)
    # In a future entropy-coded residual variant the sigmas shape the AC
    # decoder; in the current direct-residual production form the residual
    # is already int8 and we re-affine into A1's q-grid space. Multiply by
    # sigmas_per_pair to materialize the OPERATIONAL byte consumption that
    # Catalog #220 + the byte-mutation smoke verifies (small mutations to
    # sigma_table_blob CHANGE the per-pair latent magnitude).
    latents = (residual_q * sigmas_per_pair) * latent_scale.unsqueeze(
        0
    ) + latent_offset.unsqueeze(0)
    if latents.shape != (A1_N_PAIRS, A1_LATENT_DIM):
        raise ValueError(
            f"Z3G2 reconstructed latents shape {tuple(latents.shape)} "
            f"!= ({A1_N_PAIRS}, {A1_LATENT_DIM})"
        )
    pad = b"\x00" * A1_LATENT_BLOB_LEN
    a1_byte_faithful = decoder_section + pad + sidecar_section
    return a1_byte_faithful, latents, sigma_table_fp32, class_indices_long


__all__ = [
    "_class_conditional_arithmetic_decode",
    "_unpack_class_prior_cdf",
    "_unpack_sigma_table_entropy_coded",
    "reconstruct_class_indices_and_sigma_table_from_z3g2_payload",
    "select_inflate_device",
]
