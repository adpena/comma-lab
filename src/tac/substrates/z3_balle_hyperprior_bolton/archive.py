# SPDX-License-Identifier: MIT
"""Wire format for the Z3 Ballé hyperprior bolt-on composition substrate.

The production score-moving Z3 contract is a latent replacement contract:
Z3 must replace A1's latent bytes inside the inner ZIP member ``x`` and
inflate must reconstruct latents before HNeRV decode. An append-only Z3HP1
trailer is valid only as a research diagnostic because it adds bytes to A1
instead of replacing the A1 latent stream.

Legacy diagnostic inner wire (single ZIP member ``x`` in archive.zip)::

    [A1 wire format] (verbatim — uint32 LE decoder section total +
                       decoder blob + 15387 B latent blob + A1 sidecar blob)
    [Z3HP1 sidecar] (this module)
        magic         : 4 bytes ASCII ``"Z3H1"``
        version       : uint8 (== 1)
        n_pairs       : uint16 LE (== 600)
        hyper_dim     : uint8  (hyper-latent dimensionality, e.g. 8)
        latent_dim    : uint8  (must == A1_LATENT_DIM == 28)
        int8_w_scale  : float32 LE (4 bytes; w_int8 / scale = w_real)
        quant_step    : float32 LE (4 bytes; latent residual quantization Δ)
        min_sigma     : float32 LE (4 bytes; sigma clamp lower bound)
        max_sigma     : float32 LE (4 bytes; sigma clamp upper bound)
        reserved      : 2 bytes (=0)
        hyperprior_weights_blob : brotli-compressed int8 MLP weights
            (length encoded as uint16 LE prefix; then the blob bytes)
        w_hat_blob    : brotli-compressed int8 hyper-latent codes
            (n_pairs * hyper_dim int8 values; length uint32 LE prefix)
        residual_blob : brotli-compressed int8 latent residual codes
            (n_pairs * latent_dim int8 values; length uint32 LE prefix)

Per Catalog #146 the inflate runtime that consumes this format is at
``tac.substrates.z3_balle_hyperprior_bolton.inflate`` (≤ 200 LOC per HNeRV
parity discipline L4 — but Z3 is a bolt-on not substrate-engineering,
so we MUST stay closer to ≤ 100 LOC inflate.py).

Per Catalog #19 (deterministic zip): the archive builder uses ZipInfo +
writestr with fixed timestamps (called by trainer; not in this module).

Magic ``Z3H1`` distinct from ``WAV1`` (a1+wavelet) and ``LPA1`` (a1+lapose)
so cross-substrate split operations refuse unknown magic loudly.

NOTE: at the smoke validation stage (Step 1 staircase) the Z3HP1 sidecar
is OPTIONAL and research-only. When the latent-replacement contract is not
implemented, the trainer MUST either omit the sidecar and emit byte-identical
A1 bytes, or explicitly build an append-only diagnostic contract whose
``byte_saving`` and exact-eval readiness flags are false.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import brotli
import torch

Z3HP1_MAGIC = b"Z3H1"
Z3HP1_VERSION = 1
# magic(4s) + version(B) + n_pairs(H) + hyper_dim(B) + latent_dim(B)
# + int8_w_scale(f) + quant_step(f) + min_sigma(f) + max_sigma(f) + reserved(2s)
Z3HP1_HEADER_STRUCT = struct.Struct("<4sBHBBffff2s")

A1_LATENT_DIM = 28
A1_N_PAIRS = 600
Z3_APPEND_ONLY_CONTRACT_BLOCKER = (
    "append-only Z3HP1 trailer adds bytes to A1 and cannot realize predicted "
    "byte savings; replace A1 latent_blob with a Z3-coded latent section before "
    "marking this archive byte-saving or exact-eval-ready"
)
Z3_BYTE_IDENTICAL_CONTRACT_BLOCKER = (
    "Z3 sidecar omitted; archive is byte-identical to A1 and carries no Z3 "
    "score-moving payload"
)


@dataclass(frozen=True)
class Z3HP1SidecarMeta:
    """Decoded Z3HP1 sidecar metadata."""

    n_pairs: int
    hyper_dim: int
    latent_dim: int
    int8_w_scale: float
    quant_step: float
    min_sigma: float
    max_sigma: float


@dataclass(frozen=True)
class Z3CompositionArchiveContract:
    """Typed authority contract for a Z3 composition payload.

    ``payload_bytes`` is the inner ``x`` member content. Authority flags are
    intentionally duplicated here so callers cannot accidentally treat an
    append-only diagnostic packet as byte-saving or exact-eval-ready.
    """

    payload_bytes: bytes
    layout: str
    base_archive_bytes: int
    z3hp1_sidecar_bytes: int
    archive_bytes: int
    byte_saving: bool
    byte_savings_bytes: int
    score_claim: bool
    promotion_eligible: bool
    ready_for_exact_eval_dispatch: bool
    exact_eval_ready: bool
    result_review_blockers: tuple[str, ...]

    def as_manifest(self) -> dict[str, Any]:
        """Return JSON-safe manifest fields for stats/provenance outputs."""
        return {
            "layout": self.layout,
            "base_archive_bytes": self.base_archive_bytes,
            "z3hp1_sidecar_bytes": self.z3hp1_sidecar_bytes,
            "archive_bytes": self.archive_bytes,
            "byte_saving": self.byte_saving,
            "byte_savings_bytes": self.byte_savings_bytes,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "ready_for_exact_eval_dispatch": self.ready_for_exact_eval_dispatch,
            "exact_eval_ready": self.exact_eval_ready,
            "result_review_blockers": list(self.result_review_blockers),
        }


def encode_z3hp1_sidecar(
    *,
    hyperprior_weights_int8: bytes,
    w_hat_int8: bytes,
    residual_int8: bytes,
    hyper_dim: int,
    int8_w_scale: float,
    quant_step: float,
    min_sigma: float,
    max_sigma: float,
    n_pairs: int = A1_N_PAIRS,
    latent_dim: int = A1_LATENT_DIM,
) -> bytes:
    """Pack the Z3HP1 hyperprior sidecar.

    Args:
        hyperprior_weights_int8: Raw int8 bytes for the hyperprior MLP
            weights (will be brotli-compressed in the sidecar).
        w_hat_int8: ``n_pairs * hyper_dim`` int8 bytes for the
            quantized hyper-latents (one per pair).
        residual_int8: ``n_pairs * latent_dim`` int8 bytes for the
            quantized latent residuals (one row per pair).
        hyper_dim: Hyper-latent dimensionality (e.g. 8).
        int8_w_scale: scale s.t. w_int8 / scale = w_real.
        quant_step: quantization step Δ for the conditional-Gaussian AC coder.
        min_sigma, max_sigma: bounds used at encoding-time (must match
            decoder for byte-faithful round-trip).
        n_pairs: Number of pairs (default A1_N_PAIRS = 600).
        latent_dim: Latent dim (default A1_LATENT_DIM = 28).

    Returns:
        Sidecar bytes (header + length-prefixed brotli blobs).
    """
    if n_pairs != A1_N_PAIRS:
        raise ValueError(f"n_pairs must be {A1_N_PAIRS}; got {n_pairs}")
    if latent_dim != A1_LATENT_DIM:
        raise ValueError(f"latent_dim must be {A1_LATENT_DIM}; got {latent_dim}")
    if hyper_dim <= 0 or hyper_dim > 255:
        raise ValueError(f"hyper_dim must be in [1, 255]; got {hyper_dim}")
    if len(w_hat_int8) != n_pairs * hyper_dim:
        raise ValueError(
            f"w_hat_int8 length {len(w_hat_int8)} != n_pairs*hyper_dim "
            f"{n_pairs * hyper_dim}"
        )
    if len(residual_int8) != n_pairs * latent_dim:
        raise ValueError(
            f"residual_int8 length {len(residual_int8)} != n_pairs*latent_dim "
            f"{n_pairs * latent_dim}"
        )

    header = Z3HP1_HEADER_STRUCT.pack(
        Z3HP1_MAGIC,
        Z3HP1_VERSION,
        n_pairs,
        hyper_dim,
        latent_dim,
        int8_w_scale,
        quant_step,
        min_sigma,
        max_sigma,
        b"\x00\x00",
    )

    weights_compressed = brotli.compress(hyperprior_weights_int8, quality=11)
    if len(weights_compressed) > 0xFFFF:
        raise ValueError(
            f"hyperprior weights blob too large: {len(weights_compressed)} > 65535"
        )
    w_hat_compressed = brotli.compress(w_hat_int8, quality=11)
    residual_compressed = brotli.compress(residual_int8, quality=11)

    return (
        header
        + struct.pack("<H", len(weights_compressed))
        + weights_compressed
        + struct.pack("<I", len(w_hat_compressed))
        + w_hat_compressed
        + struct.pack("<I", len(residual_compressed))
        + residual_compressed
    )


def decode_z3hp1_sidecar(
    data: bytes,
) -> tuple[Z3HP1SidecarMeta, bytes, bytes, bytes]:
    """Unpack the Z3HP1 sidecar.

    Returns ``(meta, hyperprior_weights_int8, w_hat_int8, residual_int8)``.
    Raises ``ValueError`` on bad magic / truncated payload.
    """
    if len(data) < Z3HP1_HEADER_STRUCT.size:
        raise ValueError("Z3HP1 sidecar too short for header")
    fields = Z3HP1_HEADER_STRUCT.unpack_from(data, 0)
    magic, version, n_pairs, hyper_dim, latent_dim = fields[:5]
    int8_w_scale, quant_step, min_sigma, max_sigma = fields[5:9]
    if magic != Z3HP1_MAGIC:
        raise ValueError(f"bad Z3HP1 magic: {magic!r}")
    if version != Z3HP1_VERSION:
        raise ValueError(f"unsupported Z3HP1 version: {version}")

    meta = Z3HP1SidecarMeta(
        n_pairs=int(n_pairs),
        hyper_dim=int(hyper_dim),
        latent_dim=int(latent_dim),
        int8_w_scale=float(int8_w_scale),
        quant_step=float(quant_step),
        min_sigma=float(min_sigma),
        max_sigma=float(max_sigma),
    )

    pos = Z3HP1_HEADER_STRUCT.size
    # Weights blob (uint16 LE length prefix).
    if pos + 2 > len(data):
        raise ValueError("Z3HP1 truncated before weights length prefix")
    (weights_len,) = struct.unpack_from("<H", data, pos)
    pos += 2
    if pos + weights_len > len(data):
        raise ValueError("Z3HP1 truncated mid-weights blob")
    weights_compressed = data[pos : pos + weights_len]
    pos += weights_len
    weights_int8 = brotli.decompress(weights_compressed)

    # w_hat blob (uint32 LE length prefix).
    if pos + 4 > len(data):
        raise ValueError("Z3HP1 truncated before w_hat length prefix")
    (w_hat_len,) = struct.unpack_from("<I", data, pos)
    pos += 4
    if pos + w_hat_len > len(data):
        raise ValueError("Z3HP1 truncated mid-w_hat blob")
    w_hat_compressed = data[pos : pos + w_hat_len]
    pos += w_hat_len
    w_hat_int8 = brotli.decompress(w_hat_compressed)
    if len(w_hat_int8) != meta.n_pairs * meta.hyper_dim:
        raise ValueError(
            f"Z3HP1 w_hat decoded {len(w_hat_int8)} != "
            f"n_pairs*hyper_dim {meta.n_pairs * meta.hyper_dim}"
        )

    # Residual blob (uint32 LE length prefix).
    if pos + 4 > len(data):
        raise ValueError("Z3HP1 truncated before residual length prefix")
    (residual_len,) = struct.unpack_from("<I", data, pos)
    pos += 4
    if pos + residual_len > len(data):
        raise ValueError("Z3HP1 truncated mid-residual blob")
    residual_compressed = data[pos : pos + residual_len]
    pos += residual_len
    if pos != len(data):
        raise ValueError(
            f"Z3HP1 sidecar has trailing {len(data) - pos} bytes after residual blob"
        )
    residual_int8 = brotli.decompress(residual_compressed)
    if len(residual_int8) != meta.n_pairs * meta.latent_dim:
        raise ValueError(
            f"Z3HP1 residual decoded {len(residual_int8)} != "
            f"n_pairs*latent_dim {meta.n_pairs * meta.latent_dim}"
        )

    return meta, weights_int8, w_hat_int8, residual_int8


def split_composition_archive(archive_bytes: bytes) -> tuple[bytes, bytes]:
    """Split a Z3 composition archive into (a1_bytes, z3hp1_sidecar_bytes).

    The split is by magic-byte trailer detection: scans backwards from the
    end of the archive for the ``Z3H1`` magic at a position that produces
    a valid Z3HP1 header. If not found, returns the entire archive as
    A1 bytes with empty sidecar (byte-identical-to-A1 fallback).

    This is deliberately permissive: per Ballé 2018 amortization principle
    the trainer MAY emit a sidecar-less archive when bytes_saved < overhead,
    and the inflate must still work in that case (== A1 inflate behavior).
    """
    # Common case: try the last (4 + 5 + 4*4 + 2) = 27 byte header position.
    # Scan from the end for Z3H1 magic.
    for pos in range(len(archive_bytes) - Z3HP1_HEADER_STRUCT.size, -1, -1):
        if archive_bytes[pos : pos + 4] == Z3HP1_MAGIC:
            try:
                # Validate by decoding the full sidecar.
                _ = decode_z3hp1_sidecar(archive_bytes[pos:])
                return archive_bytes[:pos], archive_bytes[pos:]
            except ValueError:
                continue
    # No Z3HP1 sidecar: byte-identical-to-A1.
    return archive_bytes, b""


def build_composition_archive_contract(
    a1_bytes: bytes,
    z3hp1_sidecar_bytes: bytes,
) -> Z3CompositionArchiveContract:
    """Build the fail-closed typed contract for a Z3 inner archive payload.

    Non-empty Z3HP1 currently means append-only diagnostic layout. It is
    allowed as an artifact, but its authority flags are false by construction.
    """
    if not z3hp1_sidecar_bytes:
        return Z3CompositionArchiveContract(
            payload_bytes=a1_bytes,
            layout="a1_byte_identical_fallback",
            base_archive_bytes=len(a1_bytes),
            z3hp1_sidecar_bytes=0,
            archive_bytes=len(a1_bytes),
            byte_saving=False,
            byte_savings_bytes=0,
            score_claim=False,
            promotion_eligible=False,
            ready_for_exact_eval_dispatch=False,
            exact_eval_ready=False,
            result_review_blockers=(Z3_BYTE_IDENTICAL_CONTRACT_BLOCKER,),
        )
    if z3hp1_sidecar_bytes[:4] != Z3HP1_MAGIC:
        raise ValueError(
            f"z3hp1_sidecar_bytes does not start with magic {Z3HP1_MAGIC!r}"
        )
    # Full decode validates length prefixes and refuses trailing bytes.
    decode_z3hp1_sidecar(z3hp1_sidecar_bytes)
    payload = a1_bytes + z3hp1_sidecar_bytes
    return Z3CompositionArchiveContract(
        payload_bytes=payload,
        layout="append_only_z3hp1_diagnostic",
        base_archive_bytes=len(a1_bytes),
        z3hp1_sidecar_bytes=len(z3hp1_sidecar_bytes),
        archive_bytes=len(payload),
        byte_saving=False,
        byte_savings_bytes=0,
        score_claim=False,
        promotion_eligible=False,
        ready_for_exact_eval_dispatch=False,
        exact_eval_ready=False,
        result_review_blockers=(Z3_APPEND_ONLY_CONTRACT_BLOCKER,),
    )


def pack_composition_archive(
    a1_bytes: bytes,
    z3hp1_sidecar_bytes: bytes,
    *,
    allow_append_only_diagnostic: bool = False,
) -> bytes:
    """Pack a Z3 composition archive, fail-closed for append-only sidecars.

    Empty sidecar returns A1 bytes unchanged. Non-empty sidecar raises unless
    the caller explicitly sets ``allow_append_only_diagnostic=True``. That flag
    is for smoke/research artifacts only and does not make the packet
    byte-saving or exact-eval-ready; use ``build_composition_archive_contract``
    when emitting provenance.
    """
    contract = build_composition_archive_contract(a1_bytes, z3hp1_sidecar_bytes)
    if contract.layout == "append_only_z3hp1_diagnostic" and not allow_append_only_diagnostic:
        raise ValueError(Z3_APPEND_ONLY_CONTRACT_BLOCKER)
    return contract.payload_bytes


def quantize_int8_with_scale(
    tensor: torch.Tensor, *, scale_clip_range: float = 7.0
) -> tuple[bytes, float]:
    """Quantize a float tensor to int8 with a single scale factor.

    Returns (int8_bytes, scale) where ``tensor_q / scale ≈ tensor``.
    The scale is chosen so that ``abs(tensor).max() / scale = scale_clip_range``
    (clipping rare outliers to the int8 range).
    """
    abs_max = float(tensor.detach().abs().max().item())
    if abs_max <= 1e-12:
        # All zeros — pick scale=1.0 to avoid div-by-zero.
        return bytes(tensor.numel()), 1.0
    scale = scale_clip_range / abs_max
    q = (tensor.detach() * scale).round().clamp(-128, 127).to(torch.int8)
    return q.cpu().numpy().tobytes(), scale


def dequantize_int8_with_scale(
    int8_bytes: bytes,
    scale: float,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Inverse of ``quantize_int8_with_scale``."""
    import numpy as np

    arr = np.frombuffer(int8_bytes, dtype=np.int8).reshape(shape)
    tensor = torch.from_numpy(arr.copy()).to(dtype)
    return tensor / scale


__all__ = [
    "A1_LATENT_DIM",
    "A1_N_PAIRS",
    "Z3HP1_HEADER_STRUCT",
    "Z3HP1_MAGIC",
    "Z3HP1_VERSION",
    "Z3_APPEND_ONLY_CONTRACT_BLOCKER",
    "Z3_BYTE_IDENTICAL_CONTRACT_BLOCKER",
    "Z3CompositionArchiveContract",
    "Z3HP1SidecarMeta",
    "build_composition_archive_contract",
    "decode_z3hp1_sidecar",
    "dequantize_int8_with_scale",
    "encode_z3hp1_sidecar",
    "pack_composition_archive",
    "quantize_int8_with_scale",
    "split_composition_archive",
]
