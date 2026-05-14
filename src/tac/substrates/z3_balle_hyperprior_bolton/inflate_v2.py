# SPDX-License-Identifier: MIT
"""Z3 v2 latent-replacement inflate runtime (council omnibus Decision 3).

Consumes a Z3 v2 payload (decoder_section + Z3HV2 section + A1 sidecar) and
reconstructs A1 latents from the Ballé-2018 hyperprior-coded residual + the
per-dim affine reload, returning latents in A1's quantized-range space so
A1's HNeRV decoder can be applied as if they had been read out of A1's own
``decode_latents_compact``.

For backward compatibility, this module's public ``reconstruct_a1_latents``
ALSO accepts a v1 payload (Z3HP1 sidecar) and the legacy A1 byte-identical
fallback. Detection is by magic at offset ``A1_DECODER_SECTION_TOTAL``:

- If the next 4 bytes == ``Z3V2``: v2 latent-replacement path.
- Else, fall back to v1's ``reconstruct_a1_latents`` (sidebar magic ``Z3H1``)
  or A1 verbatim.

LOC budget: ≤ 200 LOC per HNeRV parity discipline L4 (this module is the
v2 runtime; v1's lives in sibling ``inflate.py``). NO scorer load. NO network.
NO /tmp paths. CUDA-or-CPU agnostic.

Per Catalog #205 the device selection is via the canonical
``select_inflate_device`` helper (env-driven; refuses MPS); per Catalog
#1 the MPS-fallback ternary is forbidden.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from tac.substrates._shared.inflate_runtime import (
    select_inflate_device as _canonical_select_inflate_device,
)
from tac.substrates.z3_balle_hyperprior_bolton.architecture import (
    A1_LATENT_DIM,
    A1_N_PAIRS,
    Z3HyperpriorConfig,
    Z3HyperpriorMLP,
)
from tac.substrates.z3_balle_hyperprior_bolton.archive import (
    dequantize_int8_with_scale,
)
from tac.substrates.z3_balle_hyperprior_bolton.archive_v2 import (
    A1_DECODER_SECTION_TOTAL,
    A1_LATENT_BLOB_LEN,
    Z3HV2_MAGIC,
    decode_z3hv2_section,
    split_z3v2_payload_bytes,
)


def select_inflate_device() -> torch.device:
    """Canonical inflate device selector (Catalog #205).

    Thin wrapper over ``tac.substrates._shared.inflate_runtime.select_inflate_device``
    (per HOTZ-1 finding R1 + Catalog #205 canonical-helper discipline). The
    canonical helper returns a ``str`` (``"cuda"``/``"cpu"``); this wrapper
    converts to ``torch.device`` for backward compatibility with existing
    Z3 v2 callers + tests that rely on ``.type`` attribute access. The
    canonical helper raises on ``PACT_INFLATE_DEVICE=mps`` per CLAUDE.md
    MPS-NOISE non-negotiable.

    Per CLAUDE.md "Forbidden device-selection defaults" + Catalog #205
    (`check_inflate_py_uses_canonical_select_inflate_device`).
    """
    if os.environ.get("PACT_INFLATE_DEVICE", "auto").lower() == "mps":
        # Canonical helper raises a generic message ("unsupported"); preserve
        # the explicit MPS-refusal banner the v2 contract documented.
        raise RuntimeError("PACT_INFLATE_DEVICE=mps refused (MPS is noise)")
    return torch.device(_canonical_select_inflate_device())


def is_v2_payload(payload_bytes: bytes) -> bool:
    """True iff the bytes start with the Z3 v2 layout (Z3HV2 magic at decoder boundary).

    Returns False on any byte-shape mismatch (e.g. payload too short, A1
    legacy, or v1 Z3HP1 sidecar at end of A1).
    """
    if len(payload_bytes) < A1_DECODER_SECTION_TOTAL + len(Z3HV2_MAGIC):
        return False
    return (
        payload_bytes[
            A1_DECODER_SECTION_TOTAL : A1_DECODER_SECTION_TOTAL + len(Z3HV2_MAGIC)
        ]
        == Z3HV2_MAGIC
    )


def _load_hyperprior_from_bytes(
    weights_int8: bytes,
    int8_w_scale: float,
    config: Z3HyperpriorConfig,
) -> Z3HyperpriorMLP:
    """Reload the Z3 hyperprior MLP from int8 weights blob + scale."""
    mlp = Z3HyperpriorMLP(config)
    state_floats = dequantize_int8_with_scale(
        weights_int8,
        int8_w_scale,
        shape=(len(weights_int8),),
    )
    pos = 0
    loaded_state: dict[str, torch.Tensor] = {}
    for name, param in mlp.state_dict().items():
        n = param.numel()
        if pos + n > state_floats.numel():
            raise ValueError(
                f"Z3HV2 weights blob too short at {name}: need {n}, "
                f"have {state_floats.numel() - pos}"
            )
        loaded_state[name] = (
            state_floats[pos : pos + n].view(param.shape).to(param.dtype)
        )
        pos += n
    if pos != state_floats.numel():
        raise ValueError(
            f"Z3HV2 weights blob has trailing {state_floats.numel() - pos} values"
        )
    mlp.load_state_dict(loaded_state)
    mlp.eval()
    return mlp


def reconstruct_a1_latents_from_v2_payload(
    payload_bytes: bytes,
) -> tuple[bytes, torch.Tensor]:
    """Reconstruct A1 latents from a v2 payload via the Ballé hyperprior.

    Args:
        payload_bytes: A v2 inner payload (decoder_section + Z3HV2 + sidecar).

    Returns:
        ``(a1_byte_faithful_bytes, latents_in_a1_range)`` — a synthetic A1
        archive whose ``latent_blob`` slot is filled with the Z3-decoded
        latents pre-affine'd into A1's quantized-range space, plus the
        latents tensor (shape ``(N_PAIRS, A1_LATENT_DIM)``, fp32) ready for
        A1's ``apply_latent_sidecar``.

        The byte-faithful synthetic A1 archive is suitable for handing to
        A1's ``parse_archive(...)`` directly, because:

        - bytes 0..162168 = decoder_section (verbatim A1)
        - bytes 162168..177555 = synthetic latent_blob = LZMA-compressed
          (mins, scales, latent codes) reconstructed from the Z3 decode
        - bytes 177555.. = sidecar (verbatim A1)

        However, materialising A1's exact LZMA bytes requires the original
        per-dim mins/scales AND the byte-identical re-encode of the latent
        codes. v2 instead returns the LATENTS DIRECTLY (the Z3 decode is
        rate-saving, not bit-exact A1 latents), and the inflate runtime
        composes A1's HNeRVDecoder + apply_latent_sidecar on top.

        Per HNeRV parity discipline L4: callers should treat the returned
        ``latents_in_a1_range`` as the canonical input to A1's
        ``apply_latent_sidecar`` followed by the HNeRV decoder.
    """
    decoder_section, z3hv2_section, sidecar_section = split_z3v2_payload_bytes(
        payload_bytes
    )
    (
        meta,
        weights_int8,
        w_hat_int8,
        residual_int8,
        latent_min,
        latent_scale,
        _,
    ) = decode_z3hv2_section(z3hv2_section)

    cfg = Z3HyperpriorConfig(hyper_latent_dim=meta.hyper_dim)
    mlp = _load_hyperprior_from_bytes(weights_int8, meta.int8_w_scale, cfg)

    # Decode w_hat (the side-info hyper-latents).
    w_hat_arr = np.frombuffer(w_hat_int8, dtype=np.int8).reshape(
        meta.n_pairs, meta.hyper_dim
    )
    w_hat = torch.from_numpy(w_hat_arr.copy()).to(torch.float32) * meta.quant_step

    # Forward through hyper-synthesis (h_s) for sigma (diagnostic / round-trip).
    with torch.inference_mode():
        h2 = mlp.h_s_1(w_hat)
        sigma_logits = mlp.h_s_2(h2)
        sigma = torch.nn.functional.softplus(sigma_logits) + meta.min_sigma
        sigma = sigma.clamp(min=meta.min_sigma, max=meta.max_sigma)
    _ = sigma  # Diagnostic only; the residual already encodes the sigma-scaled state.

    # Reconstruct latents: residual_int8 stores quantized-q latents in
    # A1's q-grid space; we re-affine via per-dim (min, scale) to the
    # A1 latent range.
    residual_arr = np.frombuffer(residual_int8, dtype=np.int8).reshape(
        meta.n_pairs, meta.latent_dim
    )
    residual_q = torch.from_numpy(residual_arr.copy()).to(torch.float32)
    # Per-dim affine: latents = residual_q * scale + min. (residual_q is
    # the centered int8-quantized representation; scale + min restore A1's
    # learned-quantized fp32 range.)
    latents = residual_q * latent_scale.unsqueeze(0) + latent_min.unsqueeze(0)
    if latents.shape != (A1_N_PAIRS, A1_LATENT_DIM):
        raise ValueError(
            f"Z3HV2 reconstructed latents shape {tuple(latents.shape)} "
            f"!= ({A1_N_PAIRS}, {A1_LATENT_DIM})"
        )

    # Build the byte-faithful A1 archive bytes for callers that prefer to
    # round-trip through A1's parse_archive (this preserves the
    # decoder + sidecar bytes verbatim and pads the latent_blob slot with
    # zeros — callers that use this path MUST also pass the returned
    # latents tensor to A1's HNeRV decoder directly, NOT call A1's
    # decode_latents_compact, because the v2 latent_blob bytes are NOT
    # an LZMA-compressed A1-format blob).
    pad = b"\x00" * A1_LATENT_BLOB_LEN
    a1_byte_faithful = decoder_section + pad + sidecar_section

    return a1_byte_faithful, latents


def reconstruct_a1_latents(
    payload_bytes: bytes,
) -> tuple[bytes, torch.Tensor | None]:
    """Public adapter: detect v2 vs v1 vs A1-verbatim and dispatch.

    Returns ``(a1_compatible_bytes, latents_or_None)``. When ``latents`` is
    None the caller should run A1's ``decode_latents_compact`` on the
    A1 latent_blob slice (legacy v1 byte-identical path); when ``latents``
    is non-None the caller should USE it directly and skip A1's
    ``decode_latents_compact``.
    """
    if is_v2_payload(payload_bytes):
        return reconstruct_a1_latents_from_v2_payload(payload_bytes)
    # Legacy v1 fallback (Z3HP1 sidecar at end OR A1 byte-identical).
    from tac.substrates.z3_balle_hyperprior_bolton.inflate import (
        reconstruct_a1_latents as v1_reconstruct,
    )

    return v1_reconstruct(payload_bytes)


__all__ = [
    "is_v2_payload",
    "reconstruct_a1_latents",
    "reconstruct_a1_latents_from_v2_payload",
    "select_inflate_device",
]
