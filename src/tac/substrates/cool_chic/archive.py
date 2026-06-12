# SPDX-License-Identifier: MIT
"""cool_chic archive grammar CCV1 — monolithic single-file ``0.bin``.

Catalog #124 STRICT archive-grammar 8 fields are declared in the package
``__init__``. This file IS the export-first grammar (HNeRV parity discipline
lesson L2):

::

    MAGIC(4)             b"CCV1"   Cool-Chic Variant 1
    VERSION(1)           u8        schema version (currently 1)
    LATENT_C_COARSE(2)   u16       cfg.latent_channels_coarse
    LATENT_C_FINE(2)     u16       cfg.latent_channels_fine
    NUM_PAIRS(2)         u16       cfg.num_pairs
    H_COARSE(2)          u16       latent_coarse spatial height (= H/16)
    W_COARSE(2)          u16       latent_coarse spatial width (= W/16)
    H_FINE(2)            u16       latent_fine spatial height (= H/8)
    W_FINE(2)            u16       latent_fine spatial width (= W/8)
    SYNTHESIS_BLOB_LEN(4) u32      brotli(state_dict bytes) of shared synthesis MLP
    AR_PRIOR_BLOB_LEN(4) u32       brotli(state_dict bytes) of AR prior nets (coarse+fine)
    LATENT_COARSE_LEN(4) u32       int16 latent_coarse bytes len
    LATENT_FINE_LEN(4)   u32       int16 latent_fine bytes len
    META_BLOB_LEN(4)     u32       utf-8 json meta bytes len
    SYNTHESIS_BLOB       ...       brotli(pickled synthesis state_dict, fp16 cpu)
    AR_PRIOR_BLOB        ...       brotli(pickled ar_prior state_dict, fp16 cpu)
    LATENT_COARSE_BLOB   ...       int16 quantized coarse latents
    LATENT_FINE_BLOB     ...       int16 quantized fine latents
    META_BLOB            ...       json: {synthesis_hidden, ar_prior_hidden, ..., quant scales/zps}

CCV1 = "Cool-Chic Variant 1". The grammar is fixed at design-time; mutating it
changes the schema VERSION and requires a new inflate.py.

The added "autoregressive_prior_params" section (vs sane_hnerv's grammar) is the
AR_PRIOR_BLOB — it is what makes this substrate distinct.

CLAUDE.md compliance:
- Deterministic (sorted-keys JSON, fp16 CPU state_dict, fixed brotli quality)
- No /tmp paths
- No scorer load
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

CCV1_MAGIC: bytes = b"CCV1"
"""cool_chic variant 1 archive magic."""

CCV1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

CCV2_MAGIC: bytes = b"CCV2"
"""cool_chic variant 2 archive magic — AR-Gaussian ENTROPY-CODED latents.

CCV2 replaces CCV1's RAW int16 LATENT_*_BLOB sections with entropy-coded blobs
produced by ``tac.substrates.cool_chic.entropy_coder`` (range coding the latent
integer grid against the AR prior's per-element Gaussian density). The synthesis
+ AR-prior + meta sections are byte-identical to CCV1. This is the section that
makes Cool-Chic a REAL rate carrier: the latents (99.9 % of the archive) are no
longer stored at fixed 16-bit width but at their AR-conditional entropy.

::

    MAGIC(4)              b"CCV2"
    VERSION(1)            u8         schema version (== CCV2_SCHEMA_VERSION)
    LATENT_C_COARSE(2)    u16
    LATENT_C_FINE(2)      u16
    NUM_PAIRS(2)          u16
    H_COARSE(2)/W_COARSE(2)/H_FINE(2)/W_FINE(2)  u16 each (decode shape check)
    SYNTHESIS_BLOB_LEN(4) u32
    AR_PRIOR_BLOB_LEN(4)  u32
    LATENT_COARSE_LEN(4)  u32        entropy-coded coarse blob len (ECC1 section)
    LATENT_FINE_LEN(4)    u32        entropy-coded fine blob len (ECC1 section)
    META_BLOB_LEN(4)      u32
    SYNTHESIS_BLOB / AR_PRIOR_BLOB / LATENT_COARSE_BLOB(ECC1) /
        LATENT_FINE_BLOB(ECC1) / META_BLOB

The header layout is byte-for-byte the CCV1 layout (same 39-byte struct) so the
parser dispatches purely on MAGIC; only the LATENT_*_BLOB CONTENTS differ.
"""

CCV2_SCHEMA_VERSION: int = 2
"""Schema version byte for the entropy-coded grammar."""

# Header layout: MAGIC(4) + VERSION(1) + 7 u16 + 5 u32 = 4+1+14+20 = 39 bytes
# But per design (cool_chic operator memo) we use a 30-byte explicit subset
# of the above: collapse the 6 spatial u16 fields into a single 12-byte
# "latent_shape_packed" payload AFTER the header proper.
# To keep the parser obvious + deterministic, we use the full 39-byte form
# and document the section manifest as 6 sections (5 data + meta).
CCV1_HEADER_FMT: str = "<4sBHHHHHHHIIIII"
CCV1_HEADER_SIZE: int = struct.calcsize(CCV1_HEADER_FMT)
assert CCV1_HEADER_SIZE == 39, "CCV1 header size invariant (4+1+14+20 = 39)"

# Brotli quality
BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class CoolChicArchive:
    """Parsed archive structure — the inflate-time data contract."""

    synthesis_state_dict: dict[str, torch.Tensor]
    """Shared synthesis MLP state_dict."""

    ar_prior_state_dict: dict[str, torch.Tensor]
    """AR prior nets (coarse + fine) combined state_dict, prefixed by module."""

    latents_coarse: torch.Tensor
    """``(num_pairs, C_coarse, H/16, W/16)`` float32 dequantized latents."""

    latents_fine: torch.Tensor
    """``(num_pairs, C_fine, H/8, W/8)`` float32 dequantized latents."""

    meta: dict[str, object]
    """Sidecar JSON meta with quant scales/zps + arch hparams."""

    schema_version: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Pickle + brotli a state_dict deterministically (fp16 cpu)."""
    buf = io.BytesIO()
    sd_cpu = {k: v.detach().to("cpu", dtype=torch.float16).contiguous() for k, v in sd.items()}
    pickle.dump(sd_cpu, buf, protocol=4)
    return bytes(brotli.compress(buf.getvalue(), quality=BROTLI_QUALITY))


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    raw = brotli.decompress(blob)
    sd = pickle.loads(raw)
    if not isinstance(sd, dict):
        raise ValueError("state_dict blob did not unpickle to a dict")
    return sd


def _quantize_to_int16(t: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Quantize a tensor to int16 with affine (scale, zero_point) -> int16.

    Mirror of sane_hnerv's mapping:
        ``q = round((f - zp) / scale) - 32767``
    so ``f = (q + 32767) * scale + zp`` recovers the float.
    """
    if t.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"tensor must be float; got {t.dtype}")
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        # FFFF Catalog #159 fix: -32767 fill so dequant = 0*scale + lo = lo
        return (torch.full_like(f, -32767, dtype=torch.int16), 1.0, lo)
    scale = (hi - lo) / 65534.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 65534.0)
    q = (q_unsigned - 32767.0).to(torch.int16)
    return (q, scale, lo)


def _dequantize_from_int16(q: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
    q_unsigned = q.to(torch.float32) + 32767.0
    return q_unsigned * float(scale) + float(zero_point)


def pack_archive(
    synthesis_state_dict: dict[str, torch.Tensor],
    ar_prior_state_dict: dict[str, torch.Tensor],
    latents_coarse: torch.Tensor,
    latents_fine: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = CCV1_SCHEMA_VERSION,
) -> bytes:
    """Serialize trained substrate state into the monolithic 0.bin bytes."""
    if schema_version != CCV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if latents_coarse.dim() != 4 or latents_fine.dim() != 4:
        raise ValueError("latents must be 4-D (num_pairs, C, H, W)")
    if latents_coarse.shape[0] != latents_fine.shape[0]:
        raise ValueError("num_pairs mismatch between coarse and fine latents")

    num_pairs = int(latents_coarse.shape[0])
    c_coarse = int(latents_coarse.shape[1])
    c_fine = int(latents_fine.shape[1])
    h_coarse = int(latents_coarse.shape[2])
    w_coarse = int(latents_coarse.shape[3])
    h_fine = int(latents_fine.shape[2])
    w_fine = int(latents_fine.shape[3])

    for name, v in (
        ("num_pairs", num_pairs),
        ("c_coarse", c_coarse),
        ("c_fine", c_fine),
        ("h_coarse", h_coarse),
        ("w_coarse", w_coarse),
        ("h_fine", h_fine),
        ("w_fine", w_fine),
    ):
        if v <= 0 or v > 0xFFFF:
            raise ValueError(f"{name}={v} out of u16 range")

    q_coarse, scale_c, zp_c = _quantize_to_int16(latents_coarse)
    q_fine, scale_f, zp_f = _quantize_to_int16(latents_fine)
    coarse_bytes = q_coarse.contiguous().numpy().tobytes()
    fine_bytes = q_fine.contiguous().numpy().tobytes()

    synth_blob = _serialize_state_dict(synthesis_state_dict)
    ar_blob = _serialize_state_dict(ar_prior_state_dict)

    meta_full = dict(meta)
    meta_full["_quant_scale_coarse"] = float(scale_c)
    meta_full["_quant_zp_coarse"] = float(zp_c)
    meta_full["_quant_scale_fine"] = float(scale_f)
    meta_full["_quant_zp_fine"] = float(zp_f)
    meta_bytes = json.dumps(meta_full, separators=(",", ":"), sort_keys=True).encode("utf-8")

    header = struct.pack(
        CCV1_HEADER_FMT,
        CCV1_MAGIC,
        schema_version,
        c_coarse,
        c_fine,
        num_pairs,
        h_coarse,
        w_coarse,
        h_fine,
        w_fine,
        len(synth_blob),
        len(ar_blob),
        len(coarse_bytes),
        len(fine_bytes),
        len(meta_bytes),
    )
    return header + synth_blob + ar_blob + coarse_bytes + fine_bytes + meta_bytes


def _rebuild_ar_prior_net(ar_state_dict: dict[str, torch.Tensor], prefix: str, in_ch: int, hidden: int):
    """Rebuild a single ``_ARPriorNet`` (coarse/fine) from the combined AR blob.

    Local import of architecture to avoid a module-level cycle (architecture does
    not import archive, so this is safe; kept local for clarity).
    """
    from .architecture import _ARPriorNet  # local: avoid import cycle

    net = _ARPriorNet(in_ch=in_ch, hidden=hidden)
    sub_sd = {
        k.replace(f"{prefix}.", ""): v
        for k, v in ar_state_dict.items()
        if k.startswith(f"{prefix}.")
    }
    net.load_state_dict({k: v.to(torch.float32) for k, v in sub_sd.items()}, strict=False)
    return net.eval()


def pack_archive_ccv2(
    synthesis_state_dict: dict[str, torch.Tensor],
    ar_prior_state_dict: dict[str, torch.Tensor],
    latents_coarse: torch.Tensor,
    latents_fine: torch.Tensor,
    meta: dict[str, object],
    *,
    bits_per_std: float = 5.0,
    grid_step_coarse: float | None = None,
    grid_step_fine: float | None = None,
) -> bytes:
    """Serialize substrate state with ENTROPY-CODED latents (CCV2 grammar).

    The latents are range-coded against the AR prior's per-element Gaussian
    density (see ``entropy_coder``). The AR prior net MUST be reconstructible
    from ``ar_prior_state_dict`` + ``meta['ar_prior_hidden']`` so decode replays
    the identical causal chain. ``grid_step_*`` override the auto-chosen step
    (Layer-2 / a sweep may pin them); otherwise ``choose_grid_step(bits_per_std)``.
    """
    from .entropy_coder import LatentGrid, choose_grid_step, encode_latent_chain

    if latents_coarse.dim() != 4 or latents_fine.dim() != 4:
        raise ValueError("latents must be 4-D (num_pairs, C, H, W)")
    if latents_coarse.shape[0] != latents_fine.shape[0]:
        raise ValueError("num_pairs mismatch between coarse and fine latents")

    num_pairs = int(latents_coarse.shape[0])
    c_coarse = int(latents_coarse.shape[1])
    c_fine = int(latents_fine.shape[1])
    h_coarse = int(latents_coarse.shape[2])
    w_coarse = int(latents_coarse.shape[3])
    h_fine = int(latents_fine.shape[2])
    w_fine = int(latents_fine.shape[3])

    ar_hidden = int(meta["ar_prior_hidden"])

    # CRITICAL (AR causal-chain determinism): the AR prior that ENCODES must be
    # bit-identical to the one that DECODES. The archive stores the AR net as a
    # fp16-roundtripped brotli blob, so the decoder reconstructs the fp16 net. We
    # must therefore ENCODE with the SAME fp16-roundtripped net — otherwise the
    # causal chain (pair t conditioned on z_{t-1} through the AR net) drifts and
    # decode diverges (the values grow pair-over-pair). Serialize FIRST, then
    # deserialize, and build the encoding prior from the deserialized blob.
    ar_blob = _serialize_state_dict(ar_prior_state_dict)
    ar_sd_roundtripped = _deserialize_state_dict(ar_blob)
    prior_coarse = _rebuild_ar_prior_net(ar_sd_roundtripped, "coarse", c_coarse, ar_hidden)
    prior_fine = _rebuild_ar_prior_net(ar_sd_roundtripped, "fine", c_fine, ar_hidden)

    step_c = float(grid_step_coarse) if grid_step_coarse is not None else choose_grid_step(
        latents_coarse, bits_per_std=bits_per_std
    )
    step_f = float(grid_step_fine) if grid_step_fine is not None else choose_grid_step(
        latents_fine, bits_per_std=bits_per_std
    )
    grid_c = LatentGrid(step=step_c)
    grid_f = LatentGrid(step=step_f)

    coarse_blob = encode_latent_chain(latents_coarse.to(torch.float32), prior_coarse, grid_c)
    fine_blob = encode_latent_chain(latents_fine.to(torch.float32), prior_fine, grid_f)

    synth_blob = _serialize_state_dict(synthesis_state_dict)

    meta_full = dict(meta)
    meta_full["_entropy_grid_step_coarse"] = step_c
    meta_full["_entropy_grid_step_fine"] = step_f
    meta_bytes = json.dumps(meta_full, separators=(",", ":"), sort_keys=True).encode("utf-8")

    header = struct.pack(
        CCV1_HEADER_FMT,
        CCV2_MAGIC,
        CCV2_SCHEMA_VERSION,
        c_coarse,
        c_fine,
        num_pairs,
        h_coarse,
        w_coarse,
        h_fine,
        w_fine,
        len(synth_blob),
        len(ar_blob),
        len(coarse_blob),
        len(fine_blob),
        len(meta_bytes),
    )
    return header + synth_blob + ar_blob + coarse_blob + fine_blob + meta_bytes


def parse_archive(blob: bytes) -> CoolChicArchive:
    """Parse 0.bin bytes back into the typed CoolChicArchive.

    Dispatches on MAGIC: ``CCV1`` = raw int16 latents (fallback), ``CCV2`` =
    AR-Gaussian entropy-coded latents (the real rate carrier).
    """
    if len(blob) < CCV1_HEADER_SIZE:
        raise ValueError(f"archive too short ({len(blob)} bytes; need >= {CCV1_HEADER_SIZE})")
    (
        magic,
        version,
        c_coarse,
        c_fine,
        num_pairs,
        h_coarse,
        w_coarse,
        h_fine,
        w_fine,
        synth_len,
        ar_len,
        coarse_len,
        fine_len,
        meta_len,
    ) = struct.unpack(CCV1_HEADER_FMT, blob[:CCV1_HEADER_SIZE])
    if magic not in (CCV1_MAGIC, CCV2_MAGIC):
        raise ValueError(f"bad magic: {magic!r} (expected {CCV1_MAGIC!r} or {CCV2_MAGIC!r})")
    is_ccv2 = magic == CCV2_MAGIC
    if is_ccv2:
        if version != CCV2_SCHEMA_VERSION:
            raise ValueError(f"unsupported CCV2 schema version: {version}")
    elif version != CCV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    if not is_ccv2:
        expected_coarse_bytes = num_pairs * c_coarse * h_coarse * w_coarse * 2
        expected_fine_bytes = num_pairs * c_fine * h_fine * w_fine * 2
        if coarse_len != expected_coarse_bytes:
            raise ValueError(f"coarse_len {coarse_len} != expected {expected_coarse_bytes}")
        if fine_len != expected_fine_bytes:
            raise ValueError(f"fine_len {fine_len} != expected {expected_fine_bytes}")

    pos = CCV1_HEADER_SIZE
    synth_blob = blob[pos : pos + synth_len]
    pos += synth_len
    ar_blob = blob[pos : pos + ar_len]
    pos += ar_len
    coarse_blob = blob[pos : pos + coarse_len]
    pos += coarse_len
    fine_blob = blob[pos : pos + fine_len]
    pos += fine_len
    meta_blob = blob[pos : pos + meta_len]
    pos += meta_len
    if pos != len(blob):
        raise ValueError(f"archive size {len(blob)} != expected {pos} from header")

    synth_sd = _deserialize_state_dict(synth_blob)
    ar_sd = _deserialize_state_dict(ar_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np  # local

    if is_ccv2:
        from .entropy_coder import decode_latent_chain

        ar_hidden = int(meta["ar_prior_hidden"])
        prior_coarse = _rebuild_ar_prior_net(ar_sd, "coarse", c_coarse, ar_hidden)
        prior_fine = _rebuild_ar_prior_net(ar_sd, "fine", c_fine, ar_hidden)
        latents_coarse = decode_latent_chain(coarse_blob, prior_coarse)
        latents_fine = decode_latent_chain(fine_blob, prior_fine)
        if tuple(latents_coarse.shape) != (num_pairs, c_coarse, h_coarse, w_coarse):
            raise ValueError("CCV2 decoded coarse latent shape mismatch")
        if tuple(latents_fine.shape) != (num_pairs, c_fine, h_fine, w_fine):
            raise ValueError("CCV2 decoded fine latent shape mismatch")
        meta.pop("_entropy_grid_step_coarse", None)
        meta.pop("_entropy_grid_step_fine", None)
        return CoolChicArchive(
            synthesis_state_dict=synth_sd,
            ar_prior_state_dict=ar_sd,
            latents_coarse=latents_coarse,
            latents_fine=latents_fine,
            meta=meta,
            schema_version=int(version),
        )

    q_coarse = torch.from_numpy(
        np.frombuffer(coarse_blob, dtype=np.int16).copy()
    ).view(num_pairs, c_coarse, h_coarse, w_coarse)
    q_fine = torch.from_numpy(
        np.frombuffer(fine_blob, dtype=np.int16).copy()
    ).view(num_pairs, c_fine, h_fine, w_fine)

    scale_c = float(meta.pop("_quant_scale_coarse"))
    zp_c = float(meta.pop("_quant_zp_coarse"))
    scale_f = float(meta.pop("_quant_scale_fine"))
    zp_f = float(meta.pop("_quant_zp_fine"))

    latents_coarse = _dequantize_from_int16(q_coarse, scale_c, zp_c)
    latents_fine = _dequantize_from_int16(q_fine, scale_f, zp_f)

    return CoolChicArchive(
        synthesis_state_dict=synth_sd,
        ar_prior_state_dict=ar_sd,
        latents_coarse=latents_coarse,
        latents_fine=latents_fine,
        meta=meta,
        schema_version=int(version),
    )
