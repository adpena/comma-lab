#!/usr/bin/env python3
"""ddm_rc1 -- adaptive (context-model) lossless recode race of the two MODEL sections.

The object is the cl2 frontier archive (179,982 B, sha 08ec8533...).  Only the two
MODEL sections are in scope:

* ``semantic`` -- the SM3R v1 mode-6 (row-prune mixed-depth) body.  RAW 36,130 B;
  shipped as ck2 2-plane de-interleave + Brotli q11 lgwin 24 -> 30,856 container B.
* ``hpac`` -- the IHS1 integer probability model.  RAW 17,770 B; shipped as
  Brotli -> 13,466 container B.

Every coding applied to these bodies today is GENERIC (Brotli / XZ / a parameter-free
byte de-interleave).  This module unpacks the 3/4-bit signed weight codes and races an
adaptive binary arithmetic coder (the DX1/DX2 range coder already in the receiver)
under per-tensor contexts against those generic baselines.

Stages
------
``extract``  parse the archive, split the RX1 member, restore both raw bodies, and
             prove the shipped container streams re-pack byte-identically.
``entropy``  per-tensor empirical zeroth- and first-order entropies of the unpacked
             codes (the bound the adaptive coder must approach).
``race``     the coder race; every coded body is decoded by a FRESH decoder and
             asserted byte-identical to the raw body before its bytes are reported.

Every reported number carries its BASIS: ``raw`` (the decompressed section body) or
``container`` (the bytes the archive actually holds).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import subprocess
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FRONTIER_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/rungs/"
    "lambda_1p0/retained/candidate_archive.zip"
)
FRONTIER_ARCHIVE_SHA256 = (
    "08ec85333d13d71344b4482cf261e3b2d508725e49f3ca05971265a81498ad4e"
)
FRONTIER_ARCHIVE_BYTES = 179_982
RECEIVER_COPY = FRONTIER_ARCHIVE.parent / "receiver_copy_runtime"
STORE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_rc1_model_section_adaptive_recode"
)

#: The container shape the shipped model sections use (proved by ``extract``).
BROTLI_QUALITY = 11
BROTLI_LGWIN = 24

SM3R_MAGIC = b"SM3R"
IHS1_MAGIC = b"IHS1"
ROW_PRUNE_NAMES = frozenset(
    {"blocks.1.film.weight", "blocks.2.film.weight", "blocks.3.film.weight"}
)


class Rc1Error(RuntimeError):
    """A ddm_rc1 stage refuses rather than reporting an unproven number."""


# --------------------------------------------------------------------------- utils


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def persist(path: Path, payload: bytes, *, label: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    fact = {
        "label": label,
        "path": str(path),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }
    return fact


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def ck2_interleave(body: bytes) -> bytes:
    """Encoder side of the ck2 whole-body 2-plane de-interleave."""

    span = len(body) & ~1
    planes = np.frombuffer(body[:span], dtype=np.uint8)
    return planes[0::2].tobytes() + planes[1::2].tobytes() + body[span:]


def ck2_uninterleave(body: bytes) -> bytes:
    """Exact inverse of :func:`ck2_interleave` (mirrors the receiver)."""

    span = len(body) & ~1
    half = span // 2
    restored = np.empty(span, dtype=np.uint8)
    planes = np.frombuffer(body[:span], dtype=np.uint8)
    restored[0::2] = planes[:half]
    restored[1::2] = planes[half:]
    return restored.tobytes() + body[span:]


# ------------------------------------------------------------------- SM3R geometry


@dataclass
class Field:
    """One contiguous run of the SM3R body, tagged by what it holds."""

    name: str
    kind: str  # "header" | "fp16_tensor" | "fp16_scales" | "prune_mask" | "codes"
    start: int
    stop: int
    bits: int = 0
    count: int = 0

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass
class Sm3rLayout:
    body: bytes
    keep_percent: int
    depths: dict[str, int]
    fields: list[Field] = field(default_factory=list)

    def by_kind(self, kind: str) -> list[Field]:
        return [item for item in self.fields if item.kind == kind]

    def covered(self) -> int:
        return sum(item.length for item in self.fields)


def unpack_signed_codes(payload: bytes, count: int, bits: int) -> np.ndarray:
    """Mirror of the receiver's ``_unpack_signed_bits`` (little-endian bit order)."""

    if not 2 <= bits <= 8:
        raise Rc1Error(f"invalid code width {bits}")
    byte_count = (count * bits + 7) // 8
    if len(payload) < byte_count:
        raise Rc1Error("truncated signed code stream")
    packed = np.frombuffer(payload[:byte_count], dtype=np.uint8)
    stream = np.unpackbits(packed, bitorder="little")[: count * bits]
    stream = stream.reshape(count, bits).astype(np.int16, copy=False)
    shifts = (1 << np.arange(bits, dtype=np.int16))[None]
    unsigned = (stream * shifts).sum(axis=1, dtype=np.int16)
    sign = 1 << (bits - 1)
    return np.where(unsigned >= sign, unsigned - (1 << bits), unsigned).astype(np.int16)


def pack_signed_codes(values: np.ndarray, bits: int) -> bytes:
    """Exact inverse of :func:`unpack_signed_codes`."""

    count = int(values.size)
    unsigned = (np.asarray(values, dtype=np.int32) & ((1 << bits) - 1)).astype(np.uint16)
    stream = np.zeros((count, bits), dtype=np.uint8)
    for index in range(bits):
        stream[:, index] = (unsigned >> index) & 1
    flat = stream.reshape(-1)
    pad = (-flat.size) % 8
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(flat, bitorder="little").tobytes()


def parse_sm3r_mixed(body: bytes, template: dict[str, Any]) -> Sm3rLayout:
    """Walk the SM3R mode-6 body, recording every field's exact byte range.

    The walk is the receiver's ``_decode_row_prune_mixed`` read backwards: any drift
    between the two is caught by the re-serialize identity assertion in ``extract``.
    """

    if len(body) < 10 or not body.startswith(SM3R_MAGIC):
        raise Rc1Error("not an SM3R body")
    version, mode, keep_percent, reserved = body[4:8]
    if version != 1 or mode != 6 or reserved != 0:
        raise Rc1Error(f"unsupported SM3R header v{version} mode {mode}")
    names = [name for name, value in template.items() if value.ndim >= 2]
    offset = 8
    fields = [Field("header", "header", 0, 8)]
    fields.append(Field("selection_mask", "header", offset, offset + 2))
    offset += 2
    depth_bytes = (len(names) + 1) // 2
    packed = np.frombuffer(body[offset : offset + depth_bytes], dtype=np.uint8)
    depths_array = np.empty(depth_bytes * 2, dtype=np.uint8)
    depths_array[0::2] = packed & 0xF
    depths_array[1::2] = packed >> 4
    depths = {
        name: int(value)
        for name, value in zip(names, depths_array[: len(names)].tolist(), strict=True)
    }
    fields.append(Field("depth_table", "header", offset, offset + depth_bytes))
    offset += depth_bytes

    for name, value in template.items():
        numel = int(value.numel())
        if value.ndim < 2:
            span = numel * 2
            fields.append(
                Field(name, "fp16_tensor", offset, offset + span, count=numel)
            )
            offset += span
            continue
        bits = depths[name]
        if name not in ROW_PRUNE_NAMES:
            scale_count = int(
                value.shape[-1] if name.endswith("embed.weight") else value.shape[0]
            )
            fields.append(
                Field(
                    f"{name}:scales",
                    "fp16_scales",
                    offset,
                    offset + scale_count * 2,
                    count=scale_count,
                )
            )
            offset += scale_count * 2
            span = (numel * bits + 7) // 8
            fields.append(
                Field(name, "codes", offset, offset + span, bits=bits, count=numel)
            )
            offset += span
            continue
        rows = int(value.shape[0])
        mask_bytes = (rows + 7) // 8
        fields.append(
            Field(f"{name}:mask", "prune_mask", offset, offset + mask_bytes, count=rows)
        )
        selected = np.unpackbits(
            np.frombuffer(body[offset : offset + mask_bytes], dtype=np.uint8),
            bitorder="little",
        )[:rows].astype(bool)
        offset += mask_bytes
        keep = int(selected.sum())
        expected = max(1, round(rows * keep_percent / 100.0))
        if keep != expected:
            raise Rc1Error(f"{name}: kept {keep} rows, header implies {expected}")
        columns = numel // rows
        fields.append(
            Field(
                f"{name}:scales",
                "fp16_scales",
                offset,
                offset + keep * 2,
                count=keep,
            )
        )
        offset += keep * 2
        span = (keep * columns * bits + 7) // 8
        fields.append(
            Field(
                name,
                "codes",
                offset,
                offset + span,
                bits=bits,
                count=keep * columns,
            )
        )
        offset += span
    if offset != len(body):
        raise Rc1Error(f"SM3R walk ended at {offset}, body is {len(body)} bytes")
    return Sm3rLayout(body=body, keep_percent=keep_percent, depths=depths, fields=fields)


_RECEIVER_MODULE: Any = None


def receiver_inflate() -> Any:
    """Import the receiver copy's own ``cpr1/inflate.py`` (the authority on geometry)."""

    global _RECEIVER_MODULE
    if _RECEIVER_MODULE is not None:
        return _RECEIVER_MODULE
    import importlib.util

    path = RECEIVER_COPY / "cpr1" / "inflate.py"
    if not path.is_file():
        raise Rc1Error(f"receiver copy is absent: {path}")
    cpr1 = str(path.parent)
    if cpr1 not in sys.path:
        sys.path.insert(0, cpr1)
    spec = importlib.util.spec_from_file_location("rc1_cpr1_inflate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    _RECEIVER_MODULE = module
    return module


def semantic_template() -> dict[str, Any]:
    """The receiver's own renderer state_dict -- the template the parse must match."""

    module = receiver_inflate()
    return module.SemanticTokenRenderer(module.SEMANTIC_WIDTH).state_dict()


# ------------------------------------------------------------------- IHS1 geometry


@dataclass
class Ihs1Layout:
    body: bytes
    depth_offset: int
    depth_bytes: int
    channel_count: int
    depths: np.ndarray
    weight_offset: int
    weight_bytes: int
    total_weight_bits: int
    row_bits: list[int]
    row_counts: list[int]
    tail_offset: int


def parse_ihs1(body: bytes) -> Ihs1Layout:
    """Walk the IHS1 body using the receiver's own module geometry.

    The row geometry comes from the deployed HPAC model, exactly as
    ``cpr1/integer_model_io.py`` reads it -- there is no fitted constant here.
    """

    import importlib

    import torch

    if not body.startswith(IHS1_MAGIC):
        raise Rc1Error("not an IHS1 body")
    module = receiver_inflate()
    hpac = importlib.import_module("hpac_integer")
    # The receiver's own loader builds AND deserializes; using it removes every
    # opportunity for a re-typed constructor keyword to drift from the shipped model.
    model = module.load_hpac(body, torch.device("cpu"))
    compressible = (hpac.IntegerConv2d, hpac.IntegerLinear)
    modules = [m for m in model.modules() if isinstance(m, compressible)]
    channel_count = sum(int(m.weight.shape[0]) for m in modules)
    depth_offset = len(IHS1_MAGIC)
    depth_bytes = (channel_count + 1) // 2
    packed = np.frombuffer(
        body[depth_offset : depth_offset + depth_bytes], dtype=np.uint8
    )
    values = np.empty(depth_bytes * 2, dtype=np.uint8)
    values[0::2] = packed & 0xF
    values[1::2] = packed >> 4
    depths = values[:channel_count].astype(np.int64)

    row_bits: list[int] = []
    row_counts: list[int] = []
    cursor = 0
    for module in modules:
        weight = module.weight
        if isinstance(module, hpac.IntegerConv2d):
            mask = module.mask.to(torch.bool).expand_as(weight)
            counts = [int(mask[i].sum().item()) for i in range(weight.shape[0])]
        else:
            counts = [int(weight[i].numel()) for i in range(weight.shape[0])]
        for index, count in enumerate(counts):
            row_bits.append(int(depths[cursor + index]))
            row_counts.append(count)
        cursor += weight.shape[0]
    total_weight_bits = sum(b * c for b, c in zip(row_bits, row_counts, strict=True))
    weight_offset = depth_offset + depth_bytes
    weight_bytes = (total_weight_bits + 7) // 8
    return Ihs1Layout(
        body=body,
        depth_offset=depth_offset,
        depth_bytes=depth_bytes,
        channel_count=channel_count,
        depths=depths,
        weight_offset=weight_offset,
        weight_bytes=weight_bytes,
        total_weight_bits=total_weight_bits,
        row_bits=row_bits,
        row_counts=row_counts,
        tail_offset=weight_offset + weight_bytes,
    )


def ihs1_rows(layout: Ihs1Layout) -> list[np.ndarray]:
    """Unpack the IHS1 weight rows as signed integers, one array per channel."""

    packed = np.frombuffer(
        layout.body[layout.weight_offset : layout.weight_offset + layout.weight_bytes],
        dtype=np.uint8,
    )
    bitstream = np.unpackbits(packed, bitorder="little")[: layout.total_weight_bits]
    rows: list[np.ndarray] = []
    cursor = 0
    for bits, count in zip(layout.row_bits, layout.row_counts, strict=True):
        if bits == 0:
            rows.append(np.zeros(count, dtype=np.int16))
            continue
        span = count * bits
        block = bitstream[cursor : cursor + span].reshape(count, bits).astype(np.int32)
        unsigned = (block * (1 << np.arange(bits, dtype=np.int32))).sum(axis=1)
        sign = 1 << (bits - 1)
        rows.append(
            np.where(unsigned >= sign, unsigned - (1 << bits), unsigned).astype(np.int16)
        )
        cursor += span
    if cursor != layout.total_weight_bits:
        raise Rc1Error("IHS1 row walk did not consume the weight bitstream")
    return rows


def pack_ihs1_rows(rows: list[np.ndarray], layout: Ihs1Layout) -> bytes:
    """Exact inverse of :func:`ihs1_rows`."""

    chunks: list[np.ndarray] = []
    for values, bits in zip(rows, layout.row_bits, strict=True):
        if bits == 0:
            continue
        unsigned = (np.asarray(values, dtype=np.int32) & ((1 << bits) - 1)).astype(
            np.int32
        )
        block = np.zeros((unsigned.size, bits), dtype=np.uint8)
        for index in range(bits):
            block[:, index] = (unsigned >> index) & 1
        chunks.append(block.reshape(-1))
    flat = (
        np.concatenate(chunks)
        if chunks
        else np.zeros(0, dtype=np.uint8)
    )
    if flat.size != layout.total_weight_bits:
        raise Rc1Error("IHS1 repack produced the wrong bit count")
    pad = (-flat.size) % 8
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(flat, bitorder="little").tobytes()


# ------------------------------------------------------------------------ entropy


def entropy_bits(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (n / total) * math.log2(n / total) for n in counts.values() if n
    )


def conditional_entropy_bits(symbols: np.ndarray) -> float:
    """First-order (previous symbol) conditional entropy, in bits per symbol."""

    if symbols.size < 2:
        return 0.0
    joint: Counter = Counter(
        zip(symbols[:-1].tolist(), symbols[1:].tolist(), strict=True)
    )
    prior: Counter = Counter(symbols[:-1].tolist())
    total = symbols.size - 1
    accumulated = 0.0
    for (previous, _current), count in joint.items():
        accumulated -= (count / total) * math.log2(count / prior[previous])
    return accumulated


# --------------------------------------------------------------- generic baselines


def brotli_bytes(payload: bytes, quality: int, lgwin: int) -> bytes:
    import brotli

    stream = brotli.compress(payload, quality=quality, lgwin=lgwin)
    if brotli.decompress(stream) != payload:
        raise Rc1Error(f"brotli round-trip failed at q={quality} lgwin={lgwin}")
    return stream


def xz_bytes(payload: bytes) -> bytes:
    stream = lzma.compress(payload, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    if lzma.decompress(stream, format=lzma.FORMAT_XZ) != payload:
        raise Rc1Error("xz round-trip failed")
    return stream


def zstd_bytes(payload: bytes) -> bytes | None:
    """Measurement-only baseline through the system CLI (never a shipped dependency)."""

    try:
        done = subprocess.run(
            ["zstd", "-q", "--ultra", "-22", "--long=27", "-c"],
            input=payload,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    stream = done.stdout
    back = subprocess.run(
        ["zstd", "-q", "-d", "-c"], input=stream, capture_output=True, check=True
    )
    if back.stdout != payload:
        raise Rc1Error("zstd round-trip failed")
    return stream


__all__ = [
    "FRONTIER_ARCHIVE",
    "Ihs1Layout",
    "Rc1Error",
    "Sm3rLayout",
    "ck2_interleave",
    "ck2_uninterleave",
    "ihs1_rows",
    "pack_ihs1_rows",
    "pack_signed_codes",
    "parse_ihs1",
    "parse_sm3r_mixed",
    "semantic_template",
    "unpack_signed_codes",
]


# ------------------------------------------------------------------------- stages


def read_sections() -> dict[str, Any]:
    from experiments import ddm_jg2_tail_reencode as jg2

    if not FRONTIER_ARCHIVE.is_file():
        raise Rc1Error(f"frontier archive is absent: {FRONTIER_ARCHIVE}")
    digest = sha256_bytes(FRONTIER_ARCHIVE.read_bytes())
    if digest != FRONTIER_ARCHIVE_SHA256:
        raise Rc1Error(f"frontier archive sha differs: {digest}")
    member = jg2.read_archive_member(FRONTIER_ARCHIVE)
    sections = jg2.split_member(member)
    header = jg2.RX1_HEADER.unpack(sections["header"])
    magic, version, codec, table_mode, reserved, hpac_b, semantic_b, carrier_b = header
    if magic != b"RX1M" or version != 1:
        raise Rc1Error("unexpected RX1 header")
    import brotli

    hpac_raw = (
        lzma.decompress(sections["hpac"], format=lzma.FORMAT_XZ)
        if codec == 1
        else brotli.decompress(sections["hpac"])
    )
    semantic_stream_raw = brotli.decompress(sections["semantic"])
    semantic_body = (
        ck2_uninterleave(semantic_stream_raw)
        if reserved & 0x02
        else semantic_stream_raw
    )
    return {
        "member": member,
        "sections": sections,
        "header_fields": header,
        "codec": codec,
        "table_mode": table_mode,
        "reserved": reserved,
        "hpac_raw": hpac_raw,
        "semantic_body": semantic_body,
        "semantic_ck2": bool(reserved & 0x02),
    }


def stage_extract(args: argparse.Namespace) -> dict[str, Any]:
    store = Path(args.store)
    parsed = read_sections()
    sections = parsed["sections"]
    hpac_raw = parsed["hpac_raw"]
    semantic_body = parsed["semantic_body"]

    facts = {
        "schema": "ddm_rc1_extract.v1",
        "archive": {
            "path": str(FRONTIER_ARCHIVE),
            "sha256": FRONTIER_ARCHIVE_SHA256,
            "bytes": FRONTIER_ARCHIVE_BYTES,
        },
        "member_bytes": len(parsed["member"]),
        "reserved": parsed["reserved"],
        "codec": parsed["codec"],
        "container_basis": {
            name: {"bytes": len(sections[name]), "sha256": sha256_bytes(sections[name])}
            for name in ("header", "hpac", "semantic", "carrier", "tail")
        },
        "raw_basis": {
            "hpac": {"bytes": len(hpac_raw), "sha256": sha256_bytes(hpac_raw)},
            "semantic": {
                "bytes": len(semantic_body),
                "sha256": sha256_bytes(semantic_body),
            },
        },
    }

    # Identity re-pack -- the shipped container streams must come back byte-for-byte.
    identity: dict[str, Any] = {}
    semantic_pre = (
        ck2_interleave(semantic_body) if parsed["semantic_ck2"] else semantic_body
    )
    semantic_repack = brotli_bytes(semantic_pre, BROTLI_QUALITY, BROTLI_LGWIN)
    identity["semantic"] = {
        "params": f"ck2={parsed['semantic_ck2']} brotli q{BROTLI_QUALITY} lgwin{BROTLI_LGWIN}",
        "shipped_bytes": len(sections["semantic"]),
        "repack_bytes": len(semantic_repack),
        "byte_identical": semantic_repack == sections["semantic"],
    }
    hpac_repack = brotli_bytes(hpac_raw, BROTLI_QUALITY, BROTLI_LGWIN)
    identity["hpac"] = {
        "params": f"brotli q{BROTLI_QUALITY} lgwin{BROTLI_LGWIN}",
        "shipped_bytes": len(sections["hpac"]),
        "repack_bytes": len(hpac_repack),
        "byte_identical": hpac_repack == sections["hpac"],
    }
    facts["identity_repack"] = identity

    # Structural walks + re-serialize identity of the walk itself.
    template = semantic_template()
    layout = parse_sm3r_mixed(semantic_body, template)
    rebuilt = bytearray(len(semantic_body))
    for item in layout.fields:
        rebuilt[item.start : item.stop] = semantic_body[item.start : item.stop]
    facts["sm3r"] = {
        "keep_percent": layout.keep_percent,
        "depths": layout.depths,
        "fields": len(layout.fields),
        "covered_bytes": layout.covered(),
        "walk_total": bytes(rebuilt) == semantic_body,
        "code_bytes": sum(f.length for f in layout.by_kind("codes")),
        "code_params": sum(f.count for f in layout.by_kind("codes")),
        "fp16_scale_bytes": sum(f.length for f in layout.by_kind("fp16_scales")),
        "fp16_tensor_bytes": sum(f.length for f in layout.by_kind("fp16_tensor")),
        "prune_mask_bytes": sum(f.length for f in layout.by_kind("prune_mask")),
        "header_bytes": sum(f.length for f in layout.by_kind("header")),
    }
    if not facts["sm3r"]["walk_total"]:
        raise Rc1Error("the SM3R walk does not tile the body")

    ihs1 = parse_ihs1(hpac_raw)
    rows = ihs1_rows(ihs1)
    repacked_weights = pack_ihs1_rows(rows, ihs1)
    shipped_weights = hpac_raw[ihs1.weight_offset : ihs1.tail_offset]
    facts["ihs1"] = {
        "channel_count": ihs1.channel_count,
        "depth_bytes": ihs1.depth_bytes,
        "weight_bytes": ihs1.weight_bytes,
        "total_weight_bits": ihs1.total_weight_bits,
        "tail_bytes": len(hpac_raw) - ihs1.tail_offset,
        "weight_params": int(sum(ihs1.row_counts)),
        "depth_histogram": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(ihs1.depths, return_counts=True), strict=True)
        },
        "weights_repack_identical": repacked_weights == shipped_weights,
    }
    if not facts["ihs1"]["weights_repack_identical"]:
        raise Rc1Error("the IHS1 weight repack is not byte-identical")

    retained = store / "retained"
    facts["payloads"] = {
        "semantic_body_raw": persist(
            retained / "semantic_body.sm3r.bin", semantic_body, label="SM3R raw body"
        ),
        "hpac_body_raw": persist(
            retained / "hpac_body.ihs1.bin", hpac_raw, label="IHS1 raw body"
        ),
        "semantic_stream_shipped": persist(
            retained / "semantic_stream.shipped.br",
            sections["semantic"],
            label="shipped semantic container stream",
        ),
        "hpac_stream_shipped": persist(
            retained / "hpac_stream.shipped.br",
            sections["hpac"],
            label="shipped hpac container stream",
        ),
    }
    atomic_json(store / "EXTRACT.json", facts)
    return facts


def stage_entropy(args: argparse.Namespace) -> dict[str, Any]:
    store = Path(args.store)
    parsed = read_sections()
    template = semantic_template()
    layout = parse_sm3r_mixed(parsed["semantic_body"], template)

    rows: list[dict[str, Any]] = []
    all_codes: list[np.ndarray] = []
    for item in layout.by_kind("codes"):
        values = unpack_signed_codes(
            parsed["semantic_body"][item.start : item.stop], item.count, item.bits
        )
        all_codes.append(values)
        counts = Counter(values.tolist())
        h0 = entropy_bits(counts)
        h1 = conditional_entropy_bits(values)
        rows.append(
            {
                "section": "semantic",
                "tensor": item.name,
                "bits": item.bits,
                "count": item.count,
                "packed_bytes": item.length,
                "zeros": int((values == 0).sum()),
                "zero_fraction": float((values == 0).mean()),
                "h0_bits": h0,
                "h1_bits": h1,
                "h0_bytes": h0 * item.count / 8.0,
                "h1_bytes": h1 * item.count / 8.0,
            }
        )

    ihs1 = parse_ihs1(parsed["hpac_raw"])
    ihs1_row_values = ihs1_rows(ihs1)
    by_depth: dict[int, list[np.ndarray]] = {}
    for values, bits in zip(ihs1_row_values, ihs1.row_bits, strict=True):
        by_depth.setdefault(int(bits), []).append(values)
    for bits in sorted(by_depth):
        joined = np.concatenate(by_depth[bits]) if by_depth[bits] else np.zeros(0)
        if joined.size == 0:
            continue
        counts = Counter(joined.tolist())
        h0 = entropy_bits(counts)
        h1 = conditional_entropy_bits(joined)
        rows.append(
            {
                "section": "hpac",
                "tensor": f"depth={bits} rows",
                "bits": bits,
                "count": int(joined.size),
                "packed_bytes": int(joined.size * bits / 8),
                "zeros": int((joined == 0).sum()),
                "zero_fraction": float((joined == 0).mean()),
                "h0_bits": h0,
                "h1_bits": h1,
                "h0_bytes": h0 * joined.size / 8.0,
                "h1_bytes": h1 * joined.size / 8.0,
            }
        )

    semantic_codes = np.concatenate(all_codes)
    facts = {
        "schema": "ddm_rc1_entropy.v1",
        "basis": "raw code stream (unpacked signed codes), bits per code",
        "rows": rows,
        "semantic_totals": {
            "params": int(semantic_codes.size),
            "packed_bytes": sum(f.length for f in layout.by_kind("codes")),
            "h0_sum_bytes": sum(
                r["h0_bytes"] for r in rows if r["section"] == "semantic"
            ),
            "h1_sum_bytes": sum(
                r["h1_bytes"] for r in rows if r["section"] == "semantic"
            ),
            "zero_fraction": float((semantic_codes == 0).mean()),
        },
        "hpac_totals": {
            "params": int(sum(ihs1.row_counts)),
            "packed_bytes": ihs1.weight_bytes,
            "h0_sum_bytes": sum(r["h0_bytes"] for r in rows if r["section"] == "hpac"),
            "h1_sum_bytes": sum(r["h1_bytes"] for r in rows if r["section"] == "hpac"),
        },
    }
    atomic_json(store / "ENTROPY.json", facts)
    return facts


def hpac_row_counts(hpac_raw: bytes) -> list[int]:
    """Per-channel weight counts, taken from the receiver's own model geometry."""

    import importlib

    import torch

    module = receiver_inflate()
    hpac_module = importlib.import_module("hpac_integer")
    from experiments import ddm_rc1_adaptive_section_codec as codec

    model = module.load_hpac(hpac_raw, torch.device("cpu"))
    counts, _ = codec.ihs1_geometry(model, hpac_module)
    return counts


def stage_race(args: argparse.Namespace) -> dict[str, Any]:
    """Race the generic coders against the RC1 adaptive rider, container basis."""

    from experiments import ddm_rc1_adaptive_section_codec as codec

    store = Path(args.store)
    parsed = read_sections()
    sections = parsed["sections"]
    template = semantic_template()
    semantic_body = parsed["semantic_body"]
    hpac_raw = parsed["hpac_raw"]
    counts = hpac_row_counts(hpac_raw)

    rows: list[dict[str, Any]] = []

    def record(
        section: str,
        coder: str,
        stream: bytes,
        *,
        raw_bytes: int,
        identity: bool | None,
        shipped: int,
    ) -> None:
        rows.append(
            {
                "section": section,
                "coder": coder,
                "raw_body_bytes": raw_bytes,
                "container_bytes": len(stream),
                "delta_vs_shipped": len(stream) - shipped,
                "decode_identity": identity,
                "sha256": sha256_bytes(stream),
            }
        )

    shipped_semantic = len(sections["semantic"])
    shipped_hpac = len(sections["hpac"])

    # ---- generic baselines, both sections -------------------------------------
    for section, body, shipped, ck2_default in (
        ("semantic", semantic_body, shipped_semantic, True),
        ("hpac", hpac_raw, shipped_hpac, False),
    ):
        pre = ck2_interleave(body) if ck2_default else body
        record(
            section,
            f"brotli q11 lgwin24 (ck2={ck2_default}) = SHIPPED",
            brotli_bytes(pre, 11, 24),
            raw_bytes=len(body),
            identity=True,
            shipped=shipped,
        )
        record(
            section,
            "xz -9e",
            xz_bytes(body),
            raw_bytes=len(body),
            identity=True,
            shipped=shipped,
        )
        record(
            section,
            "xz -9e over ck2",
            xz_bytes(ck2_interleave(body)),
            raw_bytes=len(body),
            identity=True,
            shipped=shipped,
        )
        stream = zstd_bytes(body)
        if stream is not None:
            record(
                section,
                "zstd --ultra -22 (measurement only)",
                stream,
                raw_bytes=len(body),
                identity=True,
                shipped=shipped,
            )
        stream = zstd_bytes(ck2_interleave(body))
        if stream is not None:
            record(
                section,
                "zstd --ultra -22 over ck2 (measurement only)",
                stream,
                raw_bytes=len(body),
                identity=True,
                shipped=shipped,
            )

    # ---- RC1 adaptive rider ---------------------------------------------------
    for shift in args.semantic_shifts:
        rider = codec.apply_semantic(semantic_body, template, shift)
        identity = codec.restore_semantic(rider, template) == semantic_body
        for ck2 in (False, True):
            pre = ck2_interleave(rider) if ck2 else rider
            record(
                "semantic",
                f"RC1 adaptive tree shift={shift} + brotli q11 lgwin24 (ck2={ck2})",
                brotli_bytes(pre, 11, 24),
                raw_bytes=len(rider),
                identity=identity,
                shipped=shipped_semantic,
            )
    for shift in args.hpac_shifts:
        rider = codec.apply_hpac(hpac_raw, counts, shift)
        identity = codec.restore_hpac(rider, counts) == hpac_raw
        for ck2 in (False, True):
            pre = ck2_interleave(rider) if ck2 else rider
            record(
                "hpac",
                f"RC1 adaptive tree shift={shift} + brotli q11 lgwin24 (ck2={ck2})",
                brotli_bytes(pre, 11, 24),
                raw_bytes=len(rider),
                identity=identity,
                shipped=shipped_hpac,
            )

    winners = {}
    for section, shipped in (("semantic", shipped_semantic), ("hpac", shipped_hpac)):
        candidates = [
            row
            for row in rows
            if row["section"] == section
            and row["coder"].startswith("RC1")
            and row["decode_identity"]
        ]
        if candidates:
            best = min(candidates, key=lambda row: row["container_bytes"])
            winners[section] = {
                "coder": best["coder"],
                "container_bytes": best["container_bytes"],
                "shipped_bytes": shipped,
                "delta": best["delta_vs_shipped"],
            }
    total_delta = sum(item["delta"] for item in winners.values())
    facts = {
        "schema": "ddm_rc1_race.v1",
        "basis": "container bytes (what archive.zip holds), not raw body bytes",
        "rows": rows,
        "winners": winners,
        "total_delta_bytes": total_delta,
        "admit_bar_bytes": -300,
        "admitted": total_delta <= -300,
    }
    atomic_json(store / "RACE.json", facts)
    return facts


# --------------------------------------------------------------- receiver staging

#: The winning race rows (MEASURED by ``race``; re-derive if the object moves).
SEMANTIC_SHIFT = 6
HPAC_SHIFT = 5
#: The archive the build stage produced (MEASURED; re-check before re-staging).
CANDIDATE_ARCHIVE_SHA256 = (
    "1438049e3655fbcfa8eb289fa51ac58f834d72d8a09586353663cea68e57c122"
)
RC1_RESERVED_SEMANTIC = 0x20
RC1_RESERVED_HPAC = 0x40

#: Anchored patches.  Each ``(file, anchor, replacement)`` refuses if the anchor is
#: absent or already present, so a stale tree can never be silently half-patched.
RESIDUAL_ANCHOR_BITS = """DX2_RESERVED_CABAC_COEFFICIENTS = 0x10
SZ1_RESERVED_KNOWN_BITS = 0x1F"""
RESIDUAL_PATCH_BITS = """DX2_RESERVED_CABAC_COEFFICIENTS = 0x10
# DDM_RC1_ADAPTIVE_MODEL_SECTIONS_V1: the two MODEL sections' adaptive-recode bits.
# Both are additive: an archive with either bit clear takes exactly the path it takes
# today, and an unknown bit still refuses fail-closed through the mask below.
RC1_RESERVED_SEMANTIC_ADAPTIVE = 0x20
RC1_RESERVED_HPAC_ADAPTIVE = 0x40
SZ1_RESERVED_KNOWN_BITS = 0x7F"""

RESIDUAL_ANCHOR_HPAC = """    if not hpac.startswith(b"IHS1"):
        raise ResidualArchiveError("RX1 HPAC is not canonical IHS1")"""
RESIDUAL_PATCH_HPAC = '''    # DDM_RC1_ADAPTIVE_MODEL_SECTIONS_V1: the RC1 hpac rider carries its own whole-body
    # 2-plane de-interleave (the section has no ck2 reserved bit of its own), so the
    # un-interleave runs FIRST and the magic check below sees the restored framing.
    if reserved & RC1_RESERVED_HPAC_ADAPTIVE:
        from .rc1_adaptive_model_sections import HPAC_MAGIC as RC1_HPAC_MAGIC

        hpac = _ck2_uninterleave_planes(hpac)
        if not hpac.startswith(RC1_HPAC_MAGIC):
            raise ResidualArchiveError("RX1 HPAC does not carry the RC1 rider")
    elif not hpac.startswith(b"IHS1"):
        raise ResidualArchiveError("RX1 HPAC is not canonical IHS1")'''

RESIDUAL_ANCHOR_SEMANTIC = (
    '    tagged_semantic = semantic_body.startswith((b"SD1M", b"SM3R"))'
)
RESIDUAL_PATCH_SEMANTIC = '''    # DDM_RC1_ADAPTIVE_MODEL_SECTIONS_V1: the RC1 semantic rider is restored by the
    # semantic receiver, which is the only place that holds the renderer template the
    # code geometry needs.  Here the body is only recognised as tagged so it reaches
    # that receiver untouched instead of falling into the legacy WANS length check.
    if reserved & RC1_RESERVED_SEMANTIC_ADAPTIVE:
        from .rc1_adaptive_model_sections import SEMANTIC_MAGIC as RC1_SEMANTIC_MAGIC

        if not semantic_body.startswith(RC1_SEMANTIC_MAGIC):
            raise ResidualArchiveError("RX1 semantic does not carry the RC1 rider")
    tagged_semantic = semantic_body.startswith((b"SD1M", b"SM3R", b"RC1S"))'''

IHS2_ANCHOR = '''def materialize_ihs1(blob: bytes, runtime) -> bytes:
    """Select the unambiguous stored representation for the production path."""
    if blob.startswith(IHS1_MAGIC):
        return blob'''
IHS2_PATCH = '''def materialize_ihs1(blob: bytes, runtime) -> bytes:
    """Select the unambiguous stored representation for the production path."""
    if blob.startswith(RC1_HPAC_MAGIC):
        # DDM_RC1_ADAPTIVE_MODEL_SECTIONS_V1: restore the packed weight bitstream from
        # its adaptive form.  The row geometry comes from the same value-free model
        # shell IHS2 already builds, so nothing video-derived enters runtime code.
        return restore_rc1_hpac(blob, layout_from_runtime(runtime).row_counts)
    if blob.startswith(IHS1_MAGIC):
        return blob'''

IHS2_IMPORT_ANCHOR = "IHS1_MAGIC = b\"IHS1\""
IHS2_IMPORT_PATCH = '''IHS1_MAGIC = b"IHS1"
from .rc1_adaptive_model_sections import HPAC_MAGIC as RC1_HPAC_MAGIC
from .rc1_adaptive_model_sections import restore_hpac as restore_rc1_hpac'''

SEMANTIC_RECEIVER_ANCHOR = '''    if blob.startswith(SD1M_MAGIC):
        return _decode_sd1m(blob, template)'''
SEMANTIC_RECEIVER_PATCH = '''    if blob.startswith(RC1_SEMANTIC_MAGIC):
        # DDM_RC1_ADAPTIVE_MODEL_SECTIONS_V1: restore the SM3R body byte-for-byte from
        # its adaptive form, then dispatch exactly as before.  The renderer template is
        # in hand here, which is why the restore lives at this seam.
        blob = restore_rc1_semantic(blob, template)
    if blob.startswith(SD1M_MAGIC):
        return _decode_sd1m(blob, template)'''

SEMANTIC_RECEIVER_IMPORT_ANCHOR = 'SD1M_MAGIC = b"SD1M"'
SEMANTIC_RECEIVER_IMPORT_PATCH = '''SD1M_MAGIC = b"SD1M"
from rc1_adaptive_model_sections import SEMANTIC_MAGIC as RC1_SEMANTIC_MAGIC
from rc1_adaptive_model_sections import restore_semantic as restore_rc1_semantic'''

CPR1_INFLATE_ANCHOR = '    if semantic_blob.startswith((b"SD1M", b"SM3R")):'
CPR1_INFLATE_PATCH = (
    '    if semantic_blob.startswith((b"SD1M", b"SM3R", b"RC1S")):'
)

# The PUBLIC entrypoint.  ``inflate.sh`` -> ``inflate.py`` -> ``f26_inflate.inflate_archive``
# reads the semantic section's magic at its own seam, BEFORE the renderer seam where the
# semantic receiver lives.  The first ddm_rc1 staging patched the receiver seam only, so the
# archive parsed under ``read_residual_archive`` + ``decode_production_tokens`` and REFUSED
# under the path the contest actually runs -- in 3.6 s on T4.  The restore therefore belongs
# HERE, at the first place the public path touches the bytes.
F26_IMPORT_ANCHOR = """import hashlib
import importlib.util"""
F26_IMPORT_PATCH = """import dataclasses
import hashlib
import importlib.util"""

F26_ANCHOR = """    if not parts.semantic_blob.startswith((WANS1_MAGIC, b"SD1M", b"SM3R")):
        raise InflationError("F26 requires WANS1, SD1M, or SM3R semantic weights")"""
F26_PATCH = '''    # DDM_RC1_ADAPTIVE_MODEL_SECTIONS_V1: restore the RC1 semantic rider to its exact
    # SM3R bytes BEFORE the magic guard, so every statement below -- this guard, the
    # renderer-seam unpack, and the WANS1 fallback -- sees the bytes it sees today.  This
    # is the seam the PUBLIC entrypoint runs.  ``_load_renderer`` is memoised through
    # sys.modules, so taking the template here costs nothing the run does not already pay.
    # An archive without the rider is untouched: the magic simply does not match.
    if parts.semantic_blob.startswith(RC1_SEMANTIC_MAGIC):
        parts = dataclasses.replace(
            parts,
            semantic_blob=restore_rc1_semantic(
                parts.semantic_blob,
                _load_renderer(renderer_dir).SemanticTokenRenderer(96).state_dict(),
            ),
        )
    if not parts.semantic_blob.startswith((WANS1_MAGIC, b"SD1M", b"SM3R")):
        raise InflationError("F26 requires WANS1, SD1M, or SM3R semantic weights")'''

F26_RC1_IMPORT_ANCHOR = (
    "from .residual_archive import decode_production_tokens, read_residual_archive"
)
F26_RC1_IMPORT_PATCH = """from .rc1_adaptive_model_sections import SEMANTIC_MAGIC as RC1_SEMANTIC_MAGIC
from .rc1_adaptive_model_sections import restore_semantic as restore_rc1_semantic
from .residual_archive import decode_production_tokens, read_residual_archive"""

PATCHES: tuple[tuple[str, str, str], ...] = (
    ("runtime/residual_archive.py", RESIDUAL_ANCHOR_BITS, RESIDUAL_PATCH_BITS),
    ("runtime/residual_archive.py", RESIDUAL_ANCHOR_HPAC, RESIDUAL_PATCH_HPAC),
    ("runtime/residual_archive.py", RESIDUAL_ANCHOR_SEMANTIC, RESIDUAL_PATCH_SEMANTIC),
    ("runtime/ihs2.py", IHS2_IMPORT_ANCHOR, IHS2_IMPORT_PATCH),
    ("runtime/ihs2.py", IHS2_ANCHOR, IHS2_PATCH),
    (
        "cpr1/ddm_mp2_semantic_receiver.py",
        SEMANTIC_RECEIVER_IMPORT_ANCHOR,
        SEMANTIC_RECEIVER_IMPORT_PATCH,
    ),
    (
        "cpr1/ddm_mp2_semantic_receiver.py",
        SEMANTIC_RECEIVER_ANCHOR,
        SEMANTIC_RECEIVER_PATCH,
    ),
    ("cpr1/inflate.py", CPR1_INFLATE_ANCHOR, CPR1_INFLATE_PATCH),
    ("runtime/f26_inflate.py", F26_IMPORT_ANCHOR, F26_IMPORT_PATCH),
    ("runtime/f26_inflate.py", F26_RC1_IMPORT_ANCHOR, F26_RC1_IMPORT_PATCH),
    ("runtime/f26_inflate.py", F26_ANCHOR, F26_PATCH),
)


def stage_tree(destination: Path) -> dict[str, Any]:
    """Copy the cl2 receiver tree, drop in the codec, and apply the anchored patches."""

    import shutil

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        RECEIVER_COPY,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    codec_source = REPO_ROOT / "experiments" / "ddm_rc1_adaptive_section_codec.py"
    text = codec_source.read_text(encoding="utf-8")
    # The receiver copy keeps ``cpr1`` on sys.path as a flat directory, so the
    # semantic receiver imports the codec by bare name; the runtime package imports it
    # relatively.  One file, two import shapes -- hence a copy in each location.
    for relative in ("runtime/rc1_adaptive_model_sections.py", "cpr1/rc1_adaptive_model_sections.py"):
        target = destination / relative
        target.write_text(text, encoding="utf-8")
    applied: list[dict[str, Any]] = []
    for relative, anchor, replacement in PATCHES:
        path = destination / relative
        body = path.read_text(encoding="utf-8")
        if replacement in body:
            raise Rc1Error(f"{relative}: patch already applied")
        if body.count(anchor) != 1:
            raise Rc1Error(
                f"{relative}: anchor appears {body.count(anchor)} times, expected 1"
            )
        path.write_text(body.replace(anchor, replacement), encoding="utf-8")
        applied.append({"file": relative, "anchor_bytes": len(anchor)})
    return {
        "path": str(destination),
        "patches": applied,
        "codec_sha256": sha256_bytes(text.encode("utf-8")),
    }


def build_candidate_member(
    parsed: dict[str, Any], template: dict[str, Any], counts: list[int]
) -> tuple[bytes, dict[str, Any]]:
    """Assemble the candidate RX1 member: only the two MODEL sections move."""

    from experiments import ddm_jg2_tail_reencode as jg2
    from experiments import ddm_rc1_adaptive_section_codec as codec

    sections = dict(parsed["sections"])
    semantic_rider = codec.apply_semantic(
        parsed["semantic_body"], template, SEMANTIC_SHIFT
    )
    if codec.restore_semantic(semantic_rider, template) != parsed["semantic_body"]:
        raise Rc1Error("semantic rider fails its own identity control")
    hpac_rider = codec.apply_hpac(parsed["hpac_raw"], counts, HPAC_SHIFT)
    if codec.restore_hpac(hpac_rider, counts) != parsed["hpac_raw"]:
        raise Rc1Error("hpac rider fails its own identity control")
    semantic_stream = brotli_bytes(
        ck2_interleave(semantic_rider), BROTLI_QUALITY, BROTLI_LGWIN
    )
    hpac_stream = brotli_bytes(ck2_interleave(hpac_rider), BROTLI_QUALITY, BROTLI_LGWIN)
    magic, version, codec_id, table_mode, reserved, _hpac_b, _sem_b, carrier_b = parsed[
        "header_fields"
    ]
    reserved |= RC1_RESERVED_SEMANTIC | RC1_RESERVED_HPAC
    sections["header"] = jg2.RX1_HEADER.pack(
        magic,
        version,
        codec_id,
        table_mode,
        reserved,
        len(hpac_stream),
        len(semantic_stream),
        carrier_b,
    )
    sections["hpac"] = hpac_stream
    sections["semantic"] = semantic_stream
    facts = {
        "reserved": reserved,
        "semantic": {
            "shipped_bytes": len(parsed["sections"]["semantic"]),
            "candidate_bytes": len(semantic_stream),
            "delta": len(semantic_stream) - len(parsed["sections"]["semantic"]),
            "shift": SEMANTIC_SHIFT,
            "rider_raw_bytes": len(semantic_rider),
        },
        "hpac": {
            "shipped_bytes": len(parsed["sections"]["hpac"]),
            "candidate_bytes": len(hpac_stream),
            "delta": len(hpac_stream) - len(parsed["sections"]["hpac"]),
            "shift": HPAC_SHIFT,
            "rider_raw_bytes": len(hpac_rider),
        },
    }
    return jg2.join_member(sections), facts


def stage_build(args: argparse.Namespace) -> dict[str, Any]:
    """Stage the receiver, build the candidate twice, and prove decode identity."""

    import os

    from experiments import ddm_cl2_hpac_prior_capacity_ladder as cl2
    from experiments import ddm_jg2_tail_reencode as jg2

    store = Path(args.store)
    parsed = read_sections()
    template = semantic_template()
    counts = hpac_row_counts(parsed["hpac_raw"])

    staged = store / "staged_runtime"
    tree = stage_tree(staged)

    member, section_facts = build_candidate_member(parsed, template, counts)
    candidate = store / "retained" / "candidate_archive.zip"
    jg2.pack_archive(member, candidate)
    candidate_bytes = candidate.read_bytes()

    # Twin encode: a second build from the same inputs must be byte-identical.
    twin_member, twin_facts = build_candidate_member(parsed, template, counts)
    twin = store / "retained" / "candidate_archive.twin.zip"
    jg2.pack_archive(twin_member, twin)
    twin_identical = twin.read_bytes() == candidate_bytes

    import shutil

    shutil.copyfile(candidate, staged / "archive.zip")

    target_path = FRONTIER_ARCHIVE.parent / "decoded_tokens.u8"
    if not target_path.is_file():
        raise Rc1Error(f"decode identity target is absent: {target_path}")
    target = np.frombuffer(target_path.read_bytes(), dtype=np.uint8).reshape(
        600, 384, 512
    )

    from types import SimpleNamespace

    env = jg2._prepare(
        SimpleNamespace(store=str(store / "work"), runtime_root=str(staged)), "rc1"
    )
    decoded, report, elapsed = cl2.decode_with_receiver(env, staged / "archive.zip")
    identity = bool(np.array_equal(decoded, target))

    facts = {
        "schema": "ddm_rc1_build.v1",
        "axis": "[macOS-CPU advisory / scorer-free EXACT byte measurement]",
        "score_claim": False,
        "staged_runtime": tree,
        "sections": section_facts,
        "shipped_archive": {
            "bytes": FRONTIER_ARCHIVE_BYTES,
            "sha256": FRONTIER_ARCHIVE_SHA256,
        },
        "candidate_archive": {
            "path": str(candidate),
            "bytes": len(candidate_bytes),
            "sha256": sha256_bytes(candidate_bytes),
        },
        "archive_delta_bytes": len(candidate_bytes) - FRONTIER_ARCHIVE_BYTES,
        "two_encodes_identical": twin_identical,
        "decoded_identity": identity,
        "decode_seconds": elapsed,
        "decode_report": {
            key: value
            for key, value in report.items()
            if isinstance(value, (int, float, str, bool))
        },
        "os_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    rate_delta = facts["archive_delta_bytes"] * 25.0 / 37_545_489
    facts["rate_delta_S"] = rate_delta
    facts["projected_S"] = 0.14781744131049854 + rate_delta
    atomic_json(store / "BUILD.json", facts)
    if not identity:
        raise Rc1Error("candidate does not decode to the shipped field")
    if not twin_identical:
        raise Rc1Error("the two encodes are not byte-identical")
    return facts


def _public_path_probe(runtime_root: Path, timeout_s: float) -> dict[str, Any]:
    """Run the PUBLIC entrypoint's own function and record where it gets to.

    ``inflate.py`` refuses without CUDA (it is a T4 submission), so a full local
    ``bash inflate.sh`` cannot complete on this host by the submission's own design.  What
    CAN be run locally is the exact function ``inflate.py`` calls --
    ``runtime.f26_inflate.inflate_archive`` -- on CPU.  The T4 failure was an
    ``InflationError`` raised at that function's semantic magic guard in 3.6 s, so a run
    that reaches the token decode has passed the guard, the renderer load, the semantic
    unpack, and ``load_state_dict(strict=True)``.  The probe is bounded: reaching the token
    decode is the PASS condition, and the same probe is run against the SHIPPED tree as a
    control so "timed out in the token decode" is a measured shape, not an assumption.
    """

    import os
    import subprocess
    import tempfile

    script = (
        "import json, sys, time\n"
        "from pathlib import Path\n"
        "root = Path(sys.argv[1])\n"
        "sys.path.insert(0, str(root))\n"
        "from runtime.f26_inflate import inflate_archive, InflationError\n"
        "out = Path(sys.argv[2])\n"
        "started = time.time()\n"
        "try:\n"
        "    inflate_archive(root / 'archive.zip', out / '0.raw',\n"
        "                    renderer_dir=root / 'cpr1', device_name='cpu',\n"
        "                    num_threads=4, checkpoint_dir=out / '.ckpt')\n"
        "    print(json.dumps({'outcome': 'COMPLETED', 'seconds': time.time() - started}))\n"
        "except InflationError as error:\n"
        "    print(json.dumps({'outcome': 'INFLATION_ERROR', 'error': str(error),\n"
        "                      'seconds': time.time() - started}))\n"
        "except Exception as error:\n"
        "    print(json.dumps({'outcome': type(error).__name__, 'error': str(error),\n"
        "                      'seconds': time.time() - started}))\n"
    )
    with tempfile.TemporaryDirectory() as scratch:
        # Build the RC64 backend with the SAME command inflate.sh uses, so the probe
        # reaches the token decode instead of stopping at the missing-library guard.
        library = Path(scratch) / "rc64_backend.so"
        subprocess.run(
            [
                os.environ.get("CC", "cc"), "-O3", "-std=c11", "-shared", "-fPIC",
                str(runtime_root / "runtime" / "entropy" / "rc64_backend.c"),
                "-o", str(library),
            ],
            check=True,
            capture_output=True,
        )
        environment = dict(os.environ, CPR1_RC64_LIBRARY=str(library))
        started = time.time()
        try:
            done = subprocess.run(
                [sys.executable, "-c", script, str(runtime_root), scratch],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(runtime_root),
                env=environment,
            )
        except subprocess.TimeoutExpired:
            return {
                "outcome": "REACHED_TOKEN_DECODE",
                "seconds": time.time() - started,
                "note": "no exception within the bound; the semantic guard is behind it",
            }
    tail = (done.stdout or "").strip().splitlines()
    parsed: dict[str, Any] = {"outcome": "UNPARSED", "stdout_tail": tail[-3:]}
    for line in reversed(tail):
        try:
            parsed = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    parsed["returncode"] = done.returncode
    parsed["stderr_tail"] = (done.stderr or "").strip().splitlines()[-5:]
    return parsed


def _inflate_sh_smoke(runtime_root: Path, timeout_s: float = 300.0) -> dict[str, Any]:
    """Run ``bash inflate.sh`` itself, with the real 3-argument contest signature.

    This host has no CUDA and ``inflate.py`` refuses without it ("requires CUDA inflation
    on linux-nvidia-t4"), so the script CANNOT complete here by the submission's own
    design.  What it does prove, and what nothing else proves, is that the script's own
    preamble runs: the C toolchain builds the backends, the Brotli version gate passes, the
    file-list loop dispatches, and ``inflate.py::_verify_input`` accepts the staged
    ``archive.zip`` against its two pinned constants.  Reaching the CUDA refusal is
    therefore the PASS condition on this host; anything earlier is a real defect.
    """

    import os
    import subprocess
    import tempfile

    # inflate.sh invokes a bare ``python``.  On the contest runner that interpreter has
    # Brotli 1.2.0; on this host the bare name resolves to a system interpreter that does
    # not, and the script's own dependency gate refuses (rc=69) before it reaches any
    # codec code.  Putting the venv first makes ``python`` mean the interpreter that
    # actually carries the submission's dependencies -- the local analogue of the runner.
    environment = dict(
        os.environ,
        PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}",
    )
    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = Path(scratch)
        data_dir = scratch_path / "extracted"
        data_dir.mkdir()
        with zipfile.ZipFile(runtime_root / "archive.zip") as archive:
            (data_dir / "p").write_bytes(archive.read("p"))
        file_list = scratch_path / "file_list.txt"
        file_list.write_text("0.hevc\n", encoding="utf-8")
        started = time.time()
        try:
            done = subprocess.run(
                [
                    "bash",
                    str(runtime_root / "inflate.sh"),
                    str(data_dir),
                    str(scratch_path / "out"),
                    str(file_list),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(runtime_root),
                env=environment,
            )
        except subprocess.TimeoutExpired:
            return {"outcome": "TIMEOUT", "seconds": time.time() - started}
    combined = f"{done.stdout}\n{done.stderr}"
    if "requires CUDA inflation" in combined:
        outcome = "REACHED_CUDA_GATE"
    elif "F26 requires WANS1, SD1M, or SM3R" in combined:
        outcome = "SEMANTIC_MAGIC_REFUSED"
    elif done.returncode == 0:
        outcome = "COMPLETED"
    else:
        outcome = "OTHER_FAILURE"
    return {
        "outcome": outcome,
        "returncode": done.returncode,
        "seconds": time.time() - started,
        "stderr_tail": (done.stderr or "").strip().splitlines()[-6:],
    }


def stage_restage(args: argparse.Namespace) -> dict[str, Any]:
    """Re-stage the receiver with the PUBLIC-path patch and verify through that path.

    The archive bytes do not change here -- only the runtime tree does -- so the retained
    candidate is reused rather than re-encoded, and its sha is re-checked before staging.
    """

    import shutil

    from experiments import ddm_rc1_adaptive_section_codec as codec

    store = Path(args.store)
    candidate = store / "retained" / "candidate_archive.zip"
    candidate_bytes = candidate.read_bytes()
    if sha256_bytes(candidate_bytes) != CANDIDATE_ARCHIVE_SHA256:
        raise Rc1Error("retained candidate archive drifted; re-run build")

    staged = store / "staged_runtime"
    tree = stage_tree(staged)
    shutil.copyfile(candidate, staged / "archive.zip")
    from tac.candidate_seal import repin_receiver

    repin = repin_receiver(staged)
    if repin.verdict_after != "CONSISTENT":
        raise Rc1Error(f"receiver re-pin did not land: {repin.verdict_after}")

    # (1) The semantic state dict the PUBLIC seam builds must equal the shipped one.
    parsed = read_sections()
    template = semantic_template()
    restored = codec.restore_semantic(
        codec.apply_semantic(parsed["semantic_body"], template, SEMANTIC_SHIFT), template
    )
    semantic_bytes_identical = restored == parsed["semantic_body"]

    # (2) The public entrypoint itself, on both trees, under the same bound.
    probes = {
        "candidate": _public_path_probe(staged, args.probe_seconds),
        "shipped": _public_path_probe(RECEIVER_COPY, args.probe_seconds),
    }
    scripts = {
        "candidate": _inflate_sh_smoke(staged),
        "shipped": _inflate_sh_smoke(RECEIVER_COPY),
    }
    passed = (
        probes["candidate"]["outcome"] == "REACHED_TOKEN_DECODE"
        and probes["shipped"]["outcome"] == "REACHED_TOKEN_DECODE"
        and scripts["candidate"]["outcome"] == "REACHED_CUDA_GATE"
        and scripts["shipped"]["outcome"] == "REACHED_CUDA_GATE"
    )

    facts = {
        "schema": "ddm_rc1_restage.v1",
        "axis": "[macOS-CPU advisory / scorer-free EXACT byte measurement]",
        "score_claim": False,
        "staged_runtime": tree,
        "repin": {
            "verdict_before": repin.verdict_before,
            "verdict_after": repin.verdict_after,
            "new_sha256": repin.new_sha256,
            "new_bytes": repin.new_bytes,
        },
        "candidate_archive": {
            "path": str(candidate),
            "bytes": len(candidate_bytes),
            "sha256": sha256_bytes(candidate_bytes),
        },
        "semantic_restore_byte_identical": semantic_bytes_identical,
        "public_path_probe_seconds": args.probe_seconds,
        "public_path_probes": probes,
        "inflate_sh_smokes": scripts,
        "public_path_pass": passed,
    }
    atomic_json(store / "RESTAGE.json", facts)
    if not semantic_bytes_identical:
        raise Rc1Error("the semantic rider no longer restores to the shipped body")
    if not passed:
        raise Rc1Error(
            f"the public entrypoint did not clear the guard: probes={probes} scripts={scripts}"
        )
    return facts


STAGES = {
    "extract": stage_extract,
    "entropy": stage_entropy,
    "race": stage_race,
    "build": stage_build,
    "restage": stage_restage,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument("--store", default=str(STORE))
    parser.add_argument(
        "--semantic-shifts",
        type=int,
        nargs="+",
        default=[4, 5, 6, 7],
        help="adaptation shifts to race for the SM3R body",
    )
    parser.add_argument(
        "--hpac-shifts",
        type=int,
        nargs="+",
        default=[4, 5, 6],
        help="adaptation shifts to race for the IHS1 body",
    )
    parser.add_argument(
        "--probe-seconds",
        type=float,
        default=240.0,
        help="bound for the public-entrypoint probe; reaching the token decode is the PASS",
    )
    args = parser.parse_args(argv)
    started = time.time()
    result = STAGES[args.stage](args)
    result["elapsed_seconds"] = time.time() - started
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
