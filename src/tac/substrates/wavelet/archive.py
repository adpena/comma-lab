"""wavelet archive grammar WLV1 — monolithic single-file ``0.bin``.

Catalog #124 STRICT archive-grammar 8 fields are declared in the package
``__init__``. The grammar embeds the Mallat subband hierarchy at design-time:

::

    MAGIC(4)              b"WLV1"   Wavelet Variant 1
    VERSION(1)            u8        schema version (currently 1)
    COEFF_CHANNELS(2)     u16       cfg.coeff_channels
    NUM_PAIRS(2)          u16       cfg.num_pairs
    H_HALF(2)             u16       subband spatial height (= H/2)
    W_HALF(2)             u16       subband spatial width  (= W/2)
    SYNTHESIS_BLOB_LEN(4) u32       brotli(state_dict bytes) of synthesis MLP
    FILM_BLOB_LEN(4)      u32       brotli(state_dict bytes) of FiLM params (as tensor)
    LL_BLOB_LEN(4)        u32       int16 LL subband bytes
    LH_BLOB_LEN(4)        u32       int16 LH subband bytes
    HL_BLOB_LEN(4)        u32       int16 HL subband bytes
    HH_BLOB_LEN(4)        u32       int16 HH subband bytes
    META_BLOB_LEN(4)      u32       utf-8 json meta bytes
    SYNTHESIS_BLOB        ...       brotli(pickled synthesis state_dict, fp16 cpu)
    FILM_BLOB             ...       brotli(pickled FiLM state_dict, fp16 cpu)
    LL_BLOB               ...       int16 LL subband, row-major
    LH_BLOB               ...       int16 LH subband
    HL_BLOB               ...       int16 HL subband
    HH_BLOB               ...       int16 HH subband
    META_BLOB             ...       json: {synthesis_hidden, synthesis_layers, output_h/w, quant per-band}

WLV1 = "WaveLet Variant 1". 4 wavelet subbands + 3 metadata sections (synthesis,
FiLM, meta) = 7 sections total per the task description.

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


WLV1_MAGIC: bytes = b"WLV1"
"""wavelet variant 1 archive magic."""

WLV1_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header: MAGIC(4) + VERSION(1) + 4 u16 + 7 u32 = 4+1+8+28 = 41 bytes
# Task description's "32-byte header" was a target sketch; the actual grammar
# uses the 41-byte form for safety. Documented in the section manifest.
WLV1_HEADER_FMT: str = "<4sBHHHHIIIIIII"
WLV1_HEADER_SIZE: int = struct.calcsize(WLV1_HEADER_FMT)
assert WLV1_HEADER_SIZE == 41, "WLV1 header size invariant (4+1+8+28 = 41)"

BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class WaveletArchive:
    """Parsed archive structure — the inflate-time data contract."""

    synthesis_state_dict: dict[str, torch.Tensor]
    """Synthesis MLP state_dict."""

    film_state_dict: dict[str, torch.Tensor]
    """FiLM params (stored as a single-key dict {'film': Tensor(2, 2, C, 1, 1)})."""

    LL: torch.Tensor
    LH: torch.Tensor
    HL: torch.Tensor
    HH: torch.Tensor
    """Per-subband dequantized float32 tensors, each (num_pairs, C, H/2, W/2)."""

    meta: dict[str, object]
    """Sidecar JSON meta with quant scales/zps + arch hparams."""

    schema_version: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
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
    film_state_dict: dict[str, torch.Tensor],
    LL: torch.Tensor,
    LH: torch.Tensor,
    HL: torch.Tensor,
    HH: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = WLV1_SCHEMA_VERSION,
) -> bytes:
    """Serialize trained substrate state into the monolithic 0.bin bytes."""
    if schema_version != WLV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")

    for name, sb in (("LL", LL), ("LH", LH), ("HL", HL), ("HH", HH)):
        if sb.dim() != 4:
            raise ValueError(f"subband {name} must be 4-D; got {tuple(sb.shape)}")
    if not (LL.shape == LH.shape == HL.shape == HH.shape):
        raise ValueError("all subbands must have identical shape")

    num_pairs = int(LL.shape[0])
    c = int(LL.shape[1])
    h_half = int(LL.shape[2])
    w_half = int(LL.shape[3])

    for nm, v in (("num_pairs", num_pairs), ("coeff_channels", c), ("h_half", h_half), ("w_half", w_half)):
        if v <= 0 or v > 0xFFFF:
            raise ValueError(f"{nm}={v} out of u16 range")

    q_ll, scale_ll, zp_ll = _quantize_to_int16(LL)
    q_lh, scale_lh, zp_lh = _quantize_to_int16(LH)
    q_hl, scale_hl, zp_hl = _quantize_to_int16(HL)
    q_hh, scale_hh, zp_hh = _quantize_to_int16(HH)

    ll_bytes = q_ll.contiguous().numpy().tobytes()
    lh_bytes = q_lh.contiguous().numpy().tobytes()
    hl_bytes = q_hl.contiguous().numpy().tobytes()
    hh_bytes = q_hh.contiguous().numpy().tobytes()

    synth_blob = _serialize_state_dict(synthesis_state_dict)
    film_blob = _serialize_state_dict(film_state_dict)

    meta_full = dict(meta)
    meta_full["_quant_scale_ll"] = float(scale_ll)
    meta_full["_quant_zp_ll"] = float(zp_ll)
    meta_full["_quant_scale_lh"] = float(scale_lh)
    meta_full["_quant_zp_lh"] = float(zp_lh)
    meta_full["_quant_scale_hl"] = float(scale_hl)
    meta_full["_quant_zp_hl"] = float(zp_hl)
    meta_full["_quant_scale_hh"] = float(scale_hh)
    meta_full["_quant_zp_hh"] = float(zp_hh)
    meta_bytes = json.dumps(meta_full, separators=(",", ":"), sort_keys=True).encode("utf-8")

    header = struct.pack(
        WLV1_HEADER_FMT,
        WLV1_MAGIC,
        schema_version,
        c,
        num_pairs,
        h_half,
        w_half,
        len(synth_blob),
        len(film_blob),
        len(ll_bytes),
        len(lh_bytes),
        len(hl_bytes),
        len(hh_bytes),
        len(meta_bytes),
    )
    return (
        header
        + synth_blob
        + film_blob
        + ll_bytes
        + lh_bytes
        + hl_bytes
        + hh_bytes
        + meta_bytes
    )


def parse_archive(blob: bytes) -> WaveletArchive:
    """Parse 0.bin bytes back into the typed WaveletArchive."""
    if len(blob) < WLV1_HEADER_SIZE:
        raise ValueError(f"archive too short ({len(blob)} bytes; need >= {WLV1_HEADER_SIZE})")
    (
        magic,
        version,
        c,
        num_pairs,
        h_half,
        w_half,
        synth_len,
        film_len,
        ll_len,
        lh_len,
        hl_len,
        hh_len,
        meta_len,
    ) = struct.unpack(WLV1_HEADER_FMT, blob[:WLV1_HEADER_SIZE])
    if magic != WLV1_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {WLV1_MAGIC!r})")
    if version != WLV1_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_sb = num_pairs * c * h_half * w_half * 2  # int16 = 2 bytes
    for nm, sl in (("LL", ll_len), ("LH", lh_len), ("HL", hl_len), ("HH", hh_len)):
        if sl != expected_sb:
            raise ValueError(f"{nm}_len {sl} != expected {expected_sb}")

    pos = WLV1_HEADER_SIZE
    synth_blob = blob[pos : pos + synth_len]; pos += synth_len
    film_blob = blob[pos : pos + film_len]; pos += film_len
    ll_blob = blob[pos : pos + ll_len]; pos += ll_len
    lh_blob = blob[pos : pos + lh_len]; pos += lh_len
    hl_blob = blob[pos : pos + hl_len]; pos += hl_len
    hh_blob = blob[pos : pos + hh_len]; pos += hh_len
    meta_blob = blob[pos : pos + meta_len]; pos += meta_len
    if pos != len(blob):
        raise ValueError(f"archive size {len(blob)} != expected {pos} from header")

    synth_sd = _deserialize_state_dict(synth_blob)
    film_sd = _deserialize_state_dict(film_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np  # local

    def _decode_subband(b: bytes, scale: float, zp: float) -> torch.Tensor:
        q = torch.from_numpy(np.frombuffer(b, dtype=np.int16).copy()).view(num_pairs, c, h_half, w_half)
        return _dequantize_from_int16(q, scale, zp)

    scale_ll = float(meta.pop("_quant_scale_ll"))
    zp_ll = float(meta.pop("_quant_zp_ll"))
    scale_lh = float(meta.pop("_quant_scale_lh"))
    zp_lh = float(meta.pop("_quant_zp_lh"))
    scale_hl = float(meta.pop("_quant_scale_hl"))
    zp_hl = float(meta.pop("_quant_zp_hl"))
    scale_hh = float(meta.pop("_quant_scale_hh"))
    zp_hh = float(meta.pop("_quant_zp_hh"))

    LL = _decode_subband(ll_blob, scale_ll, zp_ll)
    LH = _decode_subband(lh_blob, scale_lh, zp_lh)
    HL = _decode_subband(hl_blob, scale_hl, zp_hl)
    HH = _decode_subband(hh_blob, scale_hh, zp_hh)

    return WaveletArchive(
        synthesis_state_dict=synth_sd,
        film_state_dict=film_sd,
        LL=LL,
        LH=LH,
        HL=HL,
        HH=HH,
        meta=meta,
        schema_version=int(version),
    )
