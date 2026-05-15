# SPDX-License-Identifier: MIT
"""NSCS03 archive grammar — monolithic single-file ``0.bin``.

UNIQUE-AND-COMPLETE grammar for the end-to-end Ballé joint codec. Differs
from sister grammars (`balle_renderer` BRV1, `atw_codec_v1` ATWC) by
carrying:

* TWO latent streams of DIFFERENT spatial shapes (main y_hat at H/16, W/16
  vs hyper z_hat at H/64, W/64), where each stream is row-major-flattened
  per-pair and quantized to int16.
* FIVE state-dict blobs (encoder g_a, decoder g_s, hyper-analysis h_a,
  hyper-synthesis h_s, entropy-bottleneck factorized-prior parameters)
  serialized with brotli compression of a deterministic state-dict pickle
  in fp16.

Per the audit memo NSCS03_END_TO_END_DIFFERENTIABLE_CODEC declared
archive_grammar field.

Grammar (declared at design-time per HNeRV parity L2):

::

    MAGIC(4)              b"NS03"   NSCS03 schema version 1
    VERSION(1)            u8        schema version (currently 1)
    NUM_PAIRS(2)          u16       number of pose pairs (typically 600)
    MAIN_C(2)             u16       main_latent_channels
    MAIN_H(2)             u16       latent y spatial height
    MAIN_W(2)             u16       latent y spatial width
    HYPER_C(2)            u16       hyper_latent_channels
    HYPER_H(2)            u16       latent z spatial height
    HYPER_W(2)            u16       latent z spatial width
    GA_BLOB_LEN(4)        u32       brotli(state_dict(g_a)) bytes
    GS_BLOB_LEN(4)        u32       brotli(state_dict(g_s)) bytes
    HA_BLOB_LEN(4)        u32       brotli(state_dict(h_a)) bytes
    HS_BLOB_LEN(4)        u32       brotli(state_dict(h_s)) bytes
    EB_BLOB_LEN(4)        u32       brotli(state_dict(entropy_bottleneck_z)) bytes
    MAIN_LATENTS_LEN(4)   u32       raw int16 (num_pairs * MAIN_C * MAIN_H * MAIN_W * 2) bytes
    HYPER_LATENTS_LEN(4)  u32       raw int16 (num_pairs * HYPER_C * HYPER_H * HYPER_W * 2) bytes
    META_BLOB_LEN(4)      u32       utf-8 json meta bytes
    GA_BLOB              ...
    GS_BLOB              ...
    HA_BLOB              ...
    HS_BLOB              ...
    EB_BLOB              ...
    MAIN_LATENTS         ...
    HYPER_LATENTS        ...
    META_BLOB            ...

The grammar is FIXED at design-time per HNeRV parity L2; mutating it
requires schema VERSION bump + new inflate.py.

Round-trip contract (tested in tests/test_nscs03_roundtrip.py per Catalog #91):

    bytes -> parse_archive -> NSCS03Archive
    NSCS03Archive components -> pack_archive -> bytes
    (must produce byte-identical output for the SAME inputs)

CLAUDE.md compliance:
- No silent defaults (caller passes config)
- No /tmp paths (all bytes are in-memory; archive consumer chooses storage)
- No scorer load
- Deterministic: same input -> same bytes (no timestamps, no host info)
- Catalog #158 degenerate-range fix: int16 quantize fills with -32767 sentinel
  when min == max so dequant returns the constant value.
"""

from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

NS03_MAGIC: bytes = b"NS03"
"""NSCS03 archive magic (Newer Substrate-Class-Shift 03)."""

NS03_SCHEMA_VERSION: int = 1
"""Schema version byte. Bump when grammar changes."""

# Header layout:
#   MAGIC(4) + VERSION(1) + NUM_PAIRS(2) + MAIN_C(2) + MAIN_H(2) + MAIN_W(2)
#   + HYPER_C(2) + HYPER_H(2) + HYPER_W(2)
#   + GA_LEN(4) + GS_LEN(4) + HA_LEN(4) + HS_LEN(4) + EB_LEN(4)
#   + MAIN_LATENTS_LEN(4) + HYPER_LATENTS_LEN(4) + META_LEN(4)
# Total = 4 + 1 + 2*7 + 4*8 = 51 bytes
NS03_HEADER_FMT: str = "<4sBHHHHHHHIIIIIIII"
NS03_HEADER_SIZE: int = struct.calcsize(NS03_HEADER_FMT)
assert NS03_HEADER_SIZE == 51, f"header size invariant; got {NS03_HEADER_SIZE}"

BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class NSCS03Archive:
    """Parsed archive structure — the inflate-time data contract.

    Attributes:
        encoder_state_dict: g_a (analysis transform) state_dict.
            (Carried for archive completeness even though inflate uses
            only decoder + hyper-synthesis + entropy-bottleneck;
            preserves the substrate's full encode-decode round-trip.)
        decoder_state_dict: g_s (synthesis transform) state_dict.
        hyper_analysis_state_dict: h_a state_dict (carried for completeness).
        hyper_synthesis_state_dict: h_s state_dict.
        entropy_state_dict: entropy_bottleneck_z state_dict (factorized prior).
        main_latents: ``(num_pairs, main_c, main_h, main_w)`` float main
            latents y_hat (dequantized from int16 archive bytes).
        hyper_latents: ``(num_pairs, hyper_c, hyper_h, hyper_w)`` float
            hyper latents z_hat.
        meta: sidecar JSON meta (config, quant scales/zero-points,
            arch params, ...).
        schema_version: archive schema version (must == NS03_SCHEMA_VERSION).
    """

    encoder_state_dict: dict[str, torch.Tensor]
    decoder_state_dict: dict[str, torch.Tensor]
    hyper_analysis_state_dict: dict[str, torch.Tensor]
    hyper_synthesis_state_dict: dict[str, torch.Tensor]
    entropy_state_dict: dict[str, torch.Tensor]
    main_latents: torch.Tensor
    hyper_latents: torch.Tensor
    meta: dict[str, object]
    schema_version: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Serialize + brotli a state_dict deterministically (fp16 CPU).

    Uses ``torch.save`` (NOT plain pickle) because plain pickle of detached
    + contiguous tensors is non-deterministic (pickle's REDUCE captures the
    storage object identity which varies across calls). ``torch.save``
    with stable key ordering (sorted) is byte-stable.
    """
    sd_sorted: dict[str, torch.Tensor] = {
        k: sd[k].detach().to("cpu", dtype=torch.float16).contiguous()
        for k in sorted(sd.keys())
    }
    buf = io.BytesIO()
    torch.save(sd_sorted, buf)
    return bytes(brotli.compress(buf.getvalue(), quality=BROTLI_QUALITY))


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    raw = brotli.decompress(blob)
    sd = torch.load(io.BytesIO(raw), weights_only=False)
    if not isinstance(sd, dict):
        raise ValueError("state_dict blob did not unpickle to a dict")
    return sd


def _quantize_to_int16(t: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Quantize ``t`` to int16. Returns ``(q, scale, zero_point)``.

    ``f = (q_int16 + 32767) * scale + zero_point``

    Catalog #158 fix: degenerate-range branch fills with -32767 (NOT 0)
    so that dequant = -32767+32767 = 0; 0*scale + zero_point = zero_point = lo.
    """
    if t.dtype not in (torch.float32, torch.float16):
        raise ValueError(f"tensor must be float; got {t.dtype}")
    f = t.detach().to(dtype=torch.float32, device="cpu")
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return (torch.full(f.shape, -32767, dtype=torch.int16), 1.0, lo)
    scale = (hi - lo) / 65534.0
    q_unsigned = ((f - lo) / scale).round().clamp(0.0, 65534.0)
    q = (q_unsigned - 32767.0).to(torch.int16)
    return (q, scale, lo)


def _dequantize_from_int16(
    q: torch.Tensor, scale: float, zero_point: float
) -> torch.Tensor:
    q_unsigned = q.to(torch.float32) + 32767.0
    return q_unsigned * float(scale) + float(zero_point)


def pack_archive(
    encoder_state_dict: dict[str, torch.Tensor],
    decoder_state_dict: dict[str, torch.Tensor],
    hyper_analysis_state_dict: dict[str, torch.Tensor],
    hyper_synthesis_state_dict: dict[str, torch.Tensor],
    entropy_state_dict: dict[str, torch.Tensor],
    main_latents: torch.Tensor,
    hyper_latents: torch.Tensor,
    meta: dict[str, object],
    *,
    schema_version: int = NS03_SCHEMA_VERSION,
) -> bytes:
    """Serialize all NSCS03 components into the monolithic ``0.bin`` bytes.

    The trainer ONLY calls this; everything else (framing, padding,
    section CRCs if added later) is the codec's responsibility, not the
    training loop's.

    Catalog #210 forensic provenance: caller MUST populate `meta` with
    license_tags / dataset_provenance / distillation_version /
    random_seed / basis_sha256 / num_frames_used (this gate enforces only
    archive grammar; provenance discipline is the caller's contract).

    Args:
        encoder_state_dict: g_a state_dict.
        decoder_state_dict: g_s state_dict.
        hyper_analysis_state_dict: h_a state_dict.
        hyper_synthesis_state_dict: h_s state_dict.
        entropy_state_dict: entropy_bottleneck_z state_dict.
        main_latents: ``(num_pairs, main_c, main_h, main_w)`` float main latents.
        hyper_latents: ``(num_pairs, hyper_c, hyper_h, hyper_w)`` float hyper latents.
        meta: sidecar JSON meta dict.
        schema_version: must equal ``NS03_SCHEMA_VERSION``.
    """
    if schema_version != NS03_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {schema_version}")
    if main_latents.dim() != 4:
        raise ValueError(
            f"main_latents must be 4-D (num_pairs, c, h, w); "
            f"got {tuple(main_latents.shape)}"
        )
    if hyper_latents.dim() != 4:
        raise ValueError(
            f"hyper_latents must be 4-D (num_pairs, c, h, w); "
            f"got {tuple(hyper_latents.shape)}"
        )
    if main_latents.shape[0] != hyper_latents.shape[0]:
        raise ValueError(
            f"num_pairs mismatch: main {main_latents.shape[0]} vs hyper "
            f"{hyper_latents.shape[0]}"
        )

    num_pairs = int(main_latents.shape[0])
    main_c = int(main_latents.shape[1])
    main_h = int(main_latents.shape[2])
    main_w = int(main_latents.shape[3])
    hyper_c = int(hyper_latents.shape[1])
    hyper_h = int(hyper_latents.shape[2])
    hyper_w = int(hyper_latents.shape[3])

    for name, val in (
        ("num_pairs", num_pairs),
        ("main_c", main_c),
        ("main_h", main_h),
        ("main_w", main_w),
        ("hyper_c", hyper_c),
        ("hyper_h", hyper_h),
        ("hyper_w", hyper_w),
    ):
        if val <= 0 or val > 0xFFFF:
            raise ValueError(f"{name}={val} out of u16 range")

    q_main, main_scale, main_zp = _quantize_to_int16(main_latents)
    q_hyper, hyper_scale, hyper_zp = _quantize_to_int16(hyper_latents)
    main_bytes = q_main.contiguous().numpy().tobytes()
    hyper_bytes = q_hyper.contiguous().numpy().tobytes()

    ga_blob = _serialize_state_dict(encoder_state_dict)
    gs_blob = _serialize_state_dict(decoder_state_dict)
    ha_blob = _serialize_state_dict(hyper_analysis_state_dict)
    hs_blob = _serialize_state_dict(hyper_synthesis_state_dict)
    eb_blob = _serialize_state_dict(entropy_state_dict)

    meta_with_quant = dict(meta)
    meta_with_quant["_main_quant_scale"] = float(main_scale)
    meta_with_quant["_main_quant_zero_point"] = float(main_zp)
    meta_with_quant["_hyper_quant_scale"] = float(hyper_scale)
    meta_with_quant["_hyper_quant_zero_point"] = float(hyper_zp)
    meta_bytes = json.dumps(
        meta_with_quant, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    header = struct.pack(
        NS03_HEADER_FMT,
        NS03_MAGIC,
        schema_version,
        num_pairs,
        main_c,
        main_h,
        main_w,
        hyper_c,
        hyper_h,
        hyper_w,
        len(ga_blob),
        len(gs_blob),
        len(ha_blob),
        len(hs_blob),
        len(eb_blob),
        len(main_bytes),
        len(hyper_bytes),
        len(meta_bytes),
    )
    return (
        header
        + ga_blob
        + gs_blob
        + ha_blob
        + hs_blob
        + eb_blob
        + main_bytes
        + hyper_bytes
        + meta_bytes
    )


def parse_archive(blob: bytes) -> NSCS03Archive:
    """Parse the ``0.bin`` bytes back into the substrate components.

    Pure-bytes function — no model class needed. inflate.py imports this +
    the model class + builds + loads + decodes, in ~200 LOC total.
    """
    if len(blob) < NS03_HEADER_SIZE:
        raise ValueError(
            f"archive too short ({len(blob)} bytes; need >= {NS03_HEADER_SIZE})"
        )
    (
        magic,
        version,
        num_pairs,
        main_c,
        main_h,
        main_w,
        hyper_c,
        hyper_h,
        hyper_w,
        ga_len,
        gs_len,
        ha_len,
        hs_len,
        eb_len,
        main_lat_len,
        hyper_lat_len,
        meta_len,
    ) = struct.unpack(NS03_HEADER_FMT, blob[:NS03_HEADER_SIZE])
    if magic != NS03_MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {NS03_MAGIC!r})")
    if version != NS03_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {version}")

    expected_main_bytes = num_pairs * main_c * main_h * main_w * 2  # int16 = 2 bytes
    expected_hyper_bytes = num_pairs * hyper_c * hyper_h * hyper_w * 2
    if main_lat_len != expected_main_bytes:
        raise ValueError(
            f"main_lat_len {main_lat_len} != "
            f"num_pairs*main_c*main_h*main_w*2 = {expected_main_bytes}"
        )
    if hyper_lat_len != expected_hyper_bytes:
        raise ValueError(
            f"hyper_lat_len {hyper_lat_len} != "
            f"num_pairs*hyper_c*hyper_h*hyper_w*2 = {expected_hyper_bytes}"
        )

    end_header = NS03_HEADER_SIZE
    end_ga = end_header + ga_len
    end_gs = end_ga + gs_len
    end_ha = end_gs + ha_len
    end_hs = end_ha + hs_len
    end_eb = end_hs + eb_len
    end_main = end_eb + main_lat_len
    end_hyper = end_main + hyper_lat_len
    end_meta = end_hyper + meta_len
    if end_meta != len(blob):
        raise ValueError(
            f"archive size {len(blob)} != expected {end_meta} from header"
        )

    ga_blob = blob[end_header:end_ga]
    gs_blob = blob[end_ga:end_gs]
    ha_blob = blob[end_gs:end_ha]
    hs_blob = blob[end_ha:end_hs]
    eb_blob = blob[end_hs:end_eb]
    main_blob = blob[end_eb:end_main]
    hyper_blob = blob[end_main:end_hyper]
    meta_blob = blob[end_hyper:end_meta]

    enc_sd = _deserialize_state_dict(ga_blob)
    dec_sd = _deserialize_state_dict(gs_blob)
    ha_sd = _deserialize_state_dict(ha_blob)
    hs_sd = _deserialize_state_dict(hs_blob)
    eb_sd = _deserialize_state_dict(eb_blob)
    meta = json.loads(meta_blob.decode("utf-8"))

    import numpy as np  # local import; keep module's import-time light

    q_main = torch.from_numpy(
        np.frombuffer(main_blob, dtype=np.int16).copy()
    ).view(num_pairs, main_c, main_h, main_w)
    q_hyper = torch.from_numpy(
        np.frombuffer(hyper_blob, dtype=np.int16).copy()
    ).view(num_pairs, hyper_c, hyper_h, hyper_w)
    main_scale = float(meta["_main_quant_scale"])
    main_zp = float(meta["_main_quant_zero_point"])
    hyper_scale = float(meta["_hyper_quant_scale"])
    hyper_zp = float(meta["_hyper_quant_zero_point"])
    main_latents = _dequantize_from_int16(q_main, main_scale, main_zp)
    hyper_latents = _dequantize_from_int16(q_hyper, hyper_scale, hyper_zp)

    return NSCS03Archive(
        encoder_state_dict=enc_sd,
        decoder_state_dict=dec_sd,
        hyper_analysis_state_dict=ha_sd,
        hyper_synthesis_state_dict=hs_sd,
        entropy_state_dict=eb_sd,
        main_latents=main_latents,
        hyper_latents=hyper_latents,
        meta=meta,
        schema_version=int(version),
    )


__all__ = [
    "NS03_HEADER_FMT",
    "NS03_HEADER_SIZE",
    "NS03_MAGIC",
    "NS03_SCHEMA_VERSION",
    "NSCS03Archive",
    "pack_archive",
    "parse_archive",
]
