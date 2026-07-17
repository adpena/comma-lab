# SPDX-License-Identifier: MIT
"""content_priced_coder — COMPOSED content-priced weight coder (SPEC_v10 P4).

THE KOLMOGOROV-VIOLATION FIX (SPEC_v10 P4).  The shape-priced baseline
(``quantize_levelset_blob``) prices ELEMENTS, not CONTENT: it packs each tensor at a fixed
per-tensor int8 grid and Brotli-q11s two concatenated streams.  This coder prices content by
COMPOSING existing, measured primitives into ONE ``encode``/``decode`` pair with EXACT
round-trip verification built in:

  * per-tensor symmetric int8 grid          — ``lever_b_levelset_generator._int8_symmetric``
                                               (REUSED; the canonical grid, byte-identical to
                                               the baseline so the comparison is fidelity-matched)
  * cross-tensor storage structure (#461)    — ``witness_crosstensor_codec``:
      - exact post-Brotli axis-storage permutation (``derive_base_permutation_plan`` /
        ``encode_base_quantized`` / ``decode_base_quantized``)
      - frame-separated mod-256 temporal delta on ``code`` (``derive_code_transform_plan`` /
        ``encode_code_quantized`` / ``decode_code_quantized``)
  * adaptive lossless entropy backend        — best-of {Brotli-q11, raw-LZMA(9|EXTREME),
                                               zlib-9} PER STREAM (Brotli-q11 is ALWAYS a
                                               candidate, so the content stream can never lose
                                               to the baseline)
  * gauge/palette pre-canonicalization (#519) — OPT-IN precision transform (default OFF;
                                               MEASURED byte-neutral in #519, so it is not a
                                               rate lever — offered as a recorded, exactly
                                               round-tripping transform, never faked as a byte win)

CONTRACT (SPEC_v10 P4, NO-FAKE #8): STRICTLY-BETTER-OR-EQUAL content-stream bytes vs
``quantize_levelset_blob`` on the SAME counted params, BY CONSTRUCTION — the permutation plan
carries the no-permutation mask as its baseline and the entropy best-of carries Brotli-q11, so
the selected size is ``<=`` the baseline for every stream.  The FULL decodable blob additionally
carries the manifest + per-tensor dequant scales (which the baseline byte-MEASUREMENT omits and
which ANY real decoder must pay) — ``compare_bytes`` reports BOTH numbers honestly.

Round-trip authority: ``decode(encode(x))`` reproduces the int8-DEQUANTIZED counted params
EXACTLY (``np.array_equal``); ``verify_round_trip`` fail-closes on any mismatch beyond the
declared int8 quantization.  The free deterministic Fourier bank (``B``/``_B``) and the ``__cfg``
scalars are EXCLUDED (rule 118: regenerated free at decode) — exactly as the baseline excludes
them — so the byte comparison is apples-to-apples.

Axis honesty: bytes here are the REAL packed blob length (a rate quantity), MEASURED, never a
score claim; ``d_seg``/``d_pose`` are unchanged by an exact lossless re-code at a fixed int8 grid.
Anything that changes the int8 grid (gauge canonicalization) is recorded and its score effect is
a measurement, not asserted.
"""

from __future__ import annotations

import json
import lzma
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any

import brotli
import numpy as np

from tac.boundary_math.lever_b_levelset_generator import _int8_symmetric
from tac.boundary_math.witness_crosstensor_codec import (
    decode_base_quantized,
    decode_code_quantized,
    derive_base_permutation_plan,
    derive_code_transform_plan,
    encode_base_quantized,
    encode_code_quantized,
)

_MAGIC = b"CPC1"  # content-priced coder, format v1
_LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
# Entropy backend ids (stored in the manifest so decode is deterministic).
_BK_BROTLI = "brotli_q11"
_BK_LZMA = "lzma_raw_p9e"
_BK_ZLIB = "zlib_9"


class ContentPricedCoderError(ValueError):
    """Raised (fail-closed) on a round-trip mismatch or a malformed blob."""


# ---------------------------------------------------------------------------
# counted-param extraction (mirror of quantize_levelset_blob's inclusion rule).
# ---------------------------------------------------------------------------
def _is_bank(name: str) -> bool:
    return name == "B" or name.endswith("_B")


def _is_code(name: str) -> bool:
    return name == "code" or name.endswith("code")


def extract_counted_params(checkpoint: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The COUNTED tensors only (float weights + ``code``); excludes the free bank, ``__cfg``
    scalars, and empty arrays — the SAME set ``quantize_levelset_blob`` prices."""
    out: dict[str, np.ndarray] = {}
    for name, arr in checkpoint.items():
        if name.startswith("__"):
            continue  # cfg scalars / metadata (not a counted weight)
        a = np.asarray(arr)
        if a.dtype.kind not in ("f", "i", "u"):
            continue  # string / object metadata
        af = a.astype(np.float32)
        if af.size == 0 or _is_bank(name):
            continue
        out[name] = af
    return out


# ---------------------------------------------------------------------------
# #519 gauge/palette canonicalization (mirror of null_subspace_rate_measure;
# reproduced here to keep `tac` free of a tools-path import — parity-tested).
# ---------------------------------------------------------------------------
def project_head_gauge(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """#519 Gauge-1: remove the class-constant (all-ones) component of ``out_sdf.{weight,bias}``
    (a true gauge orbit of the WHOLE render — softmax(phi + c*1) == softmax(phi))."""
    out = {k: np.array(v, np.float32, copy=True) for k, v in params.items()}
    w = out["out_sdf.weight"]
    out["out_sdf.weight"] = (w - w.mean(axis=0, keepdims=True)).astype(np.float32)
    b = out["out_sdf.bias"]
    out["out_sdf.bias"] = (b - b.mean()).astype(np.float32)
    return out


def project_palette_gauge(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """#519 Gauge-2: fold the palette channel-mean into ``out_tex.bias`` (render-invariant —
    softmax rows sum to 1)."""
    out = {k: np.array(v, np.float32, copy=True) for k, v in params.items()}
    pal = out["palette"]
    v = pal.mean(axis=0, keepdims=True)  # (1,3)
    out["palette"] = (pal - v).astype(np.float32)
    out["out_tex.bias"] = (out["out_tex.bias"] + v[0]).astype(np.float32)
    return out


_CANON = {"palette": project_palette_gauge, "head": project_head_gauge}


def _apply_canonicalization(base: dict[str, np.ndarray],
                            canonicalize: tuple[str, ...]) -> tuple[dict[str, np.ndarray], list[str]]:
    applied: list[str] = []
    out = base
    for name in canonicalize:
        if name not in _CANON:
            raise ContentPricedCoderError(
                f"unknown canonicalization {name!r} (known: {sorted(_CANON)})")
        fn = _CANON[name]
        try:
            out = fn(out)
            applied.append(name)
        except KeyError as exc:  # required tensor absent → skip honestly, record
            raise ContentPricedCoderError(
                f"canonicalization {name!r} needs tensor {exc} which is absent") from exc
    return out, applied


# ---------------------------------------------------------------------------
# entropy best-of (Brotli always a candidate => never loses to the baseline).
# ---------------------------------------------------------------------------
def _compress_candidates(raw: bytes) -> dict[str, bytes]:
    return {
        _BK_BROTLI: brotli.compress(raw, quality=11),
        _BK_LZMA: lzma.compress(raw, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS),
        _BK_ZLIB: zlib.compress(raw, 9),
    }


def _compress_best(raw: bytes) -> tuple[str, bytes]:
    cands = _compress_candidates(raw)
    # deterministic tie-break: prefer brotli, then lzma, then zlib
    order = (_BK_BROTLI, _BK_LZMA, _BK_ZLIB)
    best = min(order, key=lambda k: (len(cands[k]), order.index(k)))
    return best, cands[best]


def _decompress(backend: str, comp: bytes) -> bytes:
    if backend == _BK_BROTLI:
        return brotli.decompress(comp)
    if backend == _BK_LZMA:
        return lzma.decompress(comp, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS)
    if backend == _BK_ZLIB:
        return zlib.decompress(comp)
    raise ContentPricedCoderError(f"unknown entropy backend {backend!r}")


def _int8_grid(base: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Per-tensor symmetric int8 codes + dequant scales (the canonical grid)."""
    codes: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    for name, arr in base.items():
        q, scale = _int8_symmetric(np.asarray(arr, np.float32))
        codes[name] = np.ascontiguousarray(q, dtype=np.int8)
        scales[name] = float(scale)
    return codes, scales


# ---------------------------------------------------------------------------
# blob assembly
# ---------------------------------------------------------------------------
def _pack(manifest: dict[str, Any], base_comp: bytes, code_comp: bytes) -> bytes:
    man = zlib.compress(json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode(), 9)
    return b"".join([
        _MAGIC,
        struct.pack("<I", len(man)), man,
        struct.pack("<I", len(base_comp)), base_comp,
        struct.pack("<I", len(code_comp)), code_comp,
    ])


def _unpack(blob: bytes) -> tuple[dict[str, Any], bytes, bytes]:
    if blob[:4] != _MAGIC:
        raise ContentPricedCoderError("bad magic — not a content-priced-coder blob")
    off = 4
    out: list[bytes] = []
    man_len = struct.unpack_from("<I", blob, off)[0]; off += 4
    manifest = json.loads(zlib.decompress(blob[off:off + man_len]).decode()); off += man_len
    for _ in range(2):
        n = struct.unpack_from("<I", blob, off)[0]; off += 4
        out.append(blob[off:off + n]); off += n
    if off != len(blob):
        raise ContentPricedCoderError(f"blob has {len(blob) - off} trailing byte(s)")
    return manifest, out[0], out[1]


@dataclass
class EncodePlan:
    """The recorded plan + honest byte accounting for one ``encode`` (SPEC_v10 P4 receipt)."""

    order: list[str]
    transposed_names: list[str]
    code_transform: str
    base_backend: str
    code_backend: str
    canonicalization_applied: list[str]
    base_stream_bytes: int
    code_stream_bytes: int
    manifest_bytes: int
    full_blob_bytes: int
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def content_stream_bytes(self) -> int:
        """Base+code entropy streams — the accounting-matched number vs the baseline."""
        return self.base_stream_bytes + self.code_stream_bytes

    def to_json(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["content_stream_bytes"] = self.content_stream_bytes
        return d


def encode(checkpoint: dict[str, np.ndarray], *,
           canonicalize: tuple[str, ...] = (),
           bit_alloc: None = None) -> bytes:
    """Encode the COUNTED params to a self-describing, exactly-round-trippable blob.

    ``canonicalize`` : opt-in #519 transforms in {"palette","head"} (default none; byte-neutral
                       precision levers, recorded in the manifest so decode reproduces them).
    ``bit_alloc``     : reserved for the #336 sensitivity bit-allocator.  SKIPPED by default and
                        MUST be ``None`` — see ``_SKIP_336`` for the recorded reason (it needs
                        measured per-tensor score-response RD rows not present in a raw checkpoint,
                        and non-int8 bit widths break the fidelity-matched byte contract).
    """
    blob, _plan = _encode_with_plan(checkpoint, canonicalize=canonicalize, bit_alloc=bit_alloc)
    return blob


_SKIP_336 = (
    "#336 sensitivity bit-allocation SKIPPED: it requires MEASURED per-tensor score-response "
    "RD rows (build_rd_table / solve_measured_reverse_waterfill) that a raw checkpoint dict does "
    "not carry, and downgrading a tensor below int8 is a FIDELITY tradeoff outside the "
    "equal-fidelity 'better-or-equal bytes' contract. Compose it upstream (apply_sensitivity_"
    "bitalloc_witness.py) when measured RD rows exist; the coder then prices whatever grid it emits."
)


def _encode_with_plan(checkpoint: dict[str, np.ndarray], *,
                      canonicalize: tuple[str, ...],
                      bit_alloc: None) -> tuple[bytes, EncodePlan]:
    if bit_alloc is not None:
        raise ContentPricedCoderError(_SKIP_336)
    counted = extract_counted_params(checkpoint)
    base = {k: v for k, v in counted.items() if not _is_code(k)}
    code_items = {k: v for k, v in counted.items() if _is_code(k)}
    if len(code_items) > 1:
        raise ContentPricedCoderError(f"expected <=1 code tensor; got {sorted(code_items)}")
    base, applied = _apply_canonicalization(base, canonicalize)

    order = list(base.keys())
    codes, scales = _int8_grid(base)
    shapes = {k: list(codes[k].shape) for k in order}

    # #461 cross-tensor base permutation (exact, post-Brotli-selected).
    plan = derive_base_permutation_plan(base, order)
    base_raw = encode_base_quantized(base, order, plan.transposed_names)
    base_backend, base_comp = _compress_best(base_raw)

    # #461 code temporal-delta transform + entropy best-of.
    code_transform = "none"
    code_scale = 0.0
    code_shape: list[int] = []
    code_name = ""
    code_comp = b""
    code_backend = _BK_BROTLI
    if code_items:
        code_name = next(iter(code_items))
        cq, code_scale = _int8_symmetric(np.asarray(code_items[code_name], np.float32))
        cq = np.ascontiguousarray(cq, dtype=np.int8)
        code_shape = list(cq.shape)
        cplan = derive_code_transform_plan(code_items[code_name])
        code_transform = cplan.transform
        code_raw = encode_code_quantized(cq, code_transform)
        code_backend, code_comp = _compress_best(code_raw)

    manifest = {
        "v": 1,
        "order": order,
        "shapes": shapes,
        "scales": scales,
        "transposed_names": list(plan.transposed_names),
        "base_backend": base_backend,
        "code_name": code_name,
        "code_shape": code_shape,
        "code_scale": float(code_scale),
        "code_transform": code_transform,
        "code_backend": code_backend,
        "canonicalization": applied,
    }
    blob = _pack(manifest, base_comp, code_comp)
    man_bytes = len(blob) - 4 - 12 - len(base_comp) - len(code_comp)
    recorded = EncodePlan(
        order=order,
        transposed_names=list(plan.transposed_names),
        code_transform=code_transform,
        base_backend=base_backend,
        code_backend=code_backend,
        canonicalization_applied=applied,
        base_stream_bytes=len(base_comp),
        code_stream_bytes=len(code_comp),
        manifest_bytes=man_bytes,
        full_blob_bytes=len(blob),
        skipped={"bit_alloc_336": _SKIP_336},
    )
    return blob, recorded


def decode(blob: bytes) -> dict[str, np.ndarray]:
    """Decode a content-priced blob to the int8-DEQUANTIZED counted params (fp32).

    The free Fourier bank and ``__cfg`` scalars are NOT reproduced (rule 118 — regenerated free
    at decode); the returned dict is exactly the COUNTED tensor set.
    """
    manifest, base_comp, code_comp = _unpack(blob)
    order = list(manifest["order"])
    shapes = {k: tuple(int(x) for x in manifest["shapes"][k]) for k in order}
    scales = {k: float(manifest["scales"][k]) for k in order}
    base_raw = _decompress(manifest["base_backend"], base_comp)
    base_codes = decode_base_quantized(base_raw, order, shapes, manifest["transposed_names"])
    out: dict[str, np.ndarray] = {}
    for name in order:
        out[name] = (base_codes[name].astype(np.float32) * scales[name]).astype(np.float32)
    if manifest["code_name"]:
        code_raw = _decompress(manifest["code_backend"], code_comp)
        cq = decode_code_quantized(code_raw, tuple(manifest["code_shape"]), manifest["code_transform"])
        out[manifest["code_name"]] = (cq.astype(np.float32) * float(manifest["code_scale"])).astype(np.float32)
    return out


def _expected_dequant(checkpoint: dict[str, np.ndarray],
                      canonicalize: tuple[str, ...]) -> dict[str, np.ndarray]:
    """What ``decode`` MUST reproduce: int8 round-trip of the (canonicalized) counted params."""
    counted = extract_counted_params(checkpoint)
    base = {k: v for k, v in counted.items() if not _is_code(k)}
    code_items = {k: v for k, v in counted.items() if _is_code(k)}
    base, _ = _apply_canonicalization(base, canonicalize)
    out: dict[str, np.ndarray] = {}
    for name, arr in base.items():
        q, scale = _int8_symmetric(np.asarray(arr, np.float32))
        out[name] = (q.astype(np.float32) * scale).astype(np.float32)
    for name, arr in code_items.items():
        q, scale = _int8_symmetric(np.asarray(arr, np.float32))
        out[name] = (q.astype(np.float32) * scale).astype(np.float32)
    return out


def verify_round_trip(checkpoint: dict[str, np.ndarray], *,
                      canonicalize: tuple[str, ...] = ()) -> dict[str, Any]:
    """Encode → decode → assert EXACT equality vs the declared int8 dequant. Fail-closed."""
    blob = encode(checkpoint, canonicalize=canonicalize)
    got = decode(blob)
    want = _expected_dequant(checkpoint, canonicalize)
    if set(got) != set(want):
        raise ContentPricedCoderError(
            f"round-trip key mismatch: encoder {sorted(want)} vs decoder {sorted(got)}")
    for name in want:
        if not np.array_equal(got[name], want[name]):
            d = float(np.abs(got[name] - want[name]).max())
            raise ContentPricedCoderError(
                f"round-trip NOT exact for {name!r}: max|Δ|={d} (beyond the declared int8 grid)")
    return {"exact": True, "n_tensors": len(want), "blob_bytes": len(blob)}


# ---------------------------------------------------------------------------
# honest byte comparison vs the shape-priced baseline (SPEC_v10 P4 measurement).
# ---------------------------------------------------------------------------
def compare_bytes(checkpoint: dict[str, np.ndarray], *,
                  canonicalize: tuple[str, ...] = ()) -> dict[str, Any]:
    """MEASURED byte comparison vs ``quantize_levelset_blob`` on the SAME counted params.

    Reports BOTH (a) content-stream bytes (accounting-matched to the baseline; guaranteed
    ``<=`` baseline) and (b) the full decodable blob (manifest + scales included — the honest
    real number).  Also verifies the round-trip is exact.  NO score claim (rate only).
    """
    from tac.boundary_math.lever_b_levelset_generator import quantize_levelset_blob

    counted = extract_counted_params(checkpoint)
    baseline = quantize_levelset_blob(counted)
    base_b = int(baseline["base_int8_brotli_bytes"])
    code_b = int(baseline["code_int8_brotli_bytes"])
    baseline_total = int(baseline["total_quantized_blob_bytes"])

    blob, plan = _encode_with_plan(checkpoint, canonicalize=canonicalize, bit_alloc=None)
    verify = verify_round_trip(checkpoint, canonicalize=canonicalize)

    content = plan.content_stream_bytes
    return {
        "baseline_quantize_levelset_blob": {
            "base_int8_brotli_bytes": base_b,
            "code_int8_brotli_bytes": code_b,
            "total_quantized_blob_bytes": baseline_total,
            "note": "byte MEASUREMENT (not a decodable blob): excludes per-tensor scales + manifest",
        },
        "content_priced": {
            "base_stream_bytes": plan.base_stream_bytes,
            "code_stream_bytes": plan.code_stream_bytes,
            "content_stream_bytes": content,
            "manifest_bytes": plan.manifest_bytes,
            "full_decodable_blob_bytes": plan.full_blob_bytes,
            "base_backend": plan.base_backend,
            "code_backend": plan.code_backend,
            "code_transform": plan.code_transform,
            "transposed_names": plan.transposed_names,
            "canonicalization_applied": plan.canonicalization_applied,
        },
        "delta": {
            "content_stream_vs_baseline_total": content - baseline_total,
            "content_stream_better_or_equal": content <= baseline_total,
            "full_blob_vs_baseline_total": plan.full_blob_bytes - baseline_total,
            "scale_manifest_overhead": plan.full_blob_bytes - content,
        },
        "round_trip": verify,
        "skipped": plan.skipped,
    }


__all__ = [
    "ContentPricedCoderError",
    "EncodePlan",
    "extract_counted_params",
    "project_head_gauge",
    "project_palette_gauge",
    "encode",
    "decode",
    "verify_round_trip",
    "compare_bytes",
]
