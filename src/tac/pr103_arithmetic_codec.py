"""PR #103 (hnerv_lc_ac) arithmetic coding + adaptive lgwin codec port.

This is a 1:1 port of PR103's ``submissions/hnerv_lc_ac/inflate.py`` decoder
path PLUS the encoder side that PR103 only ships inside their compress-time
pipeline (PR103 does not publish a compressor — only an inflator).

Source-of-truth: ``experiments/results/public_pr_intake_full/
public_pr103_intake_20260505_auto/source/submissions/hnerv_lc_ac/inflate.py``
(222 LOC), pre-read in this session. PR body README documents the 5 tricks;
this module ports tricks #4 (arithmetic coding on densest payloads), #5
(adaptive ``lgwin`` per-stream), and #6 (9 AC streams merged into one
RangeEncoder).

What PR103's codec achieves vs PR101's split-Brotli decoder section:
    PR101 ``decoder_blob``:          162,164 bytes (split Brotli + byte-maps)
    PR103 ``decoder + AC + lat-hi``: ~161,800 bytes after the 9-stream merged AC

Per PR103 README:
    "switching the densest payloads to AC with q8 (uint8) histograms beats
    brotli's symbol-level entropy by ~290 B"

The score impact: -290 B × (25 / 37,545,489) ≈ -0.000193 rate component on
top of Op 1 (PR101's split-Brotli baseline).

Composition contract with Op 1 (:mod:`tac.pr101_split_brotli_codec`):

    Caller workflow (CPU-only, byte-faithful round-trip):

    1. ``op1_blob = encode_decoder_compact(state_dict)`` — PR101 split-Brotli
       encoder produces a multi-stream brotli blob.
    2. ``op2_blob = encode_decoder_ac(state_dict)`` — replace the 8 densest
       weight tensors + latent-hi with arithmetic coding; the remaining
       tensors stay as split-Brotli (with adaptive ``lgwin`` per-stream).
    3. Decode side: ``decode_decoder_ac(op2_blob)`` returns the same
       state_dict; bit-identical post-quantization.

    The AC codec ALWAYS produces a strict superset of the information that
    Op 1 carries — it writes its OWN brotli streams for the non-AC tensors
    (with adaptive lgwin search) plus the merged AC stream + per-tensor q8
    histograms. The two codecs are NOT wire-format compatible — Op 2 is a
    REPLACEMENT for Op 1's decoder section, not an ADDITION layered on top
    of it. Empirical: composing Op 1 → decode → Op 2 → encode produces a
    smaller blob than Op 1 alone iff PR103's claim reproduces on the caller's
    weights.

The Op 2 encoder also accepts pre-decoded weights produced by
:func:`tac.pr101_split_brotli_codec.decode_decoder_compact` — that is the
canonical "stack on Op 1 output" path used by
``experiments/build_pr103_repacked_archive.py``.

Strict-scorer-rule: this module loads NO scorer weights and has no MPS/CUDA
dependency. CPU-only, deterministic, byte-faithful round-trip.

Empirical regression risk (Contrarian gate):
    PR103's -290 B gain was measured on PR103's OWN fine-tuned weights. On
    different weight distributions (PR106, apogee_intN, our internal
    substrates), AC may regress vs brotli for some streams. Use
    :func:`validate_ac_savings` to measure per-tensor AC vs brotli on the
    caller's weights and log warnings on regressions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import brotli
import constriction
import numpy as np
import torch

from tac.pr101_split_brotli_codec import (
    DECODER_STORAGE_ORDER,
    DECODER_STREAM_ENDS,
    FIXED_STATE_SCHEMA,
    N_QUANT,
    Pr101SplitBrotliCodecError,
    _build_per_tensor_payload,
    _quantize_tensor,
    pack_brotli_stream,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants ported from PR103 inflate.py:51-62
# ---------------------------------------------------------------------------

AC_TENSOR_INDICES: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 21)
"""The 8 schema indices PR103 chose for arithmetic coding.

These are the densest payloads — the 7 largest weight tensors (stem + 6 conv
blocks) plus ``refine.0.bias`` (idx 21, only 9 elements but evidently AC was
slightly cheaper than brotli on PR103's checkpoint).

PR103 README: 'switching the densest payloads to AC with q8 (uint8)
histograms'. We treat this set as a literal source-of-truth port — do NOT
substitute "top-N by element count" without measuring; PR103 picked these
specific indices empirically.
"""

AC_HISTOGRAM_BITS = 8
"""Histogram precision: q8 means each AC symbol has a 256-bucket histogram.

Identical to PR103. The histograms themselves are uint8 (256 entries x 1
byte = 256 bytes per tensor), then concatenated and brotli'd as one short
stream (HIST_LEN ≈ 895 bytes on PR103's checkpoint)."""

MERGED_RANGE_ENCODER = True
"""PR103's 9-stream merged optimization: a single ``RangeEncoder`` is used for
all 8 weight AC streams PLUS the latent-hi byte stream. Per PR103 README:
"merging all 9 AC streams into one constriction RangeEncoder eliminates
per-stream rounding overhead". Empirically -50 to -100 bytes vs one encoder
per stream.

This is a constant rather than a kwarg because PR103's wire format hardcodes
this choice (the inflator uses one RangeDecoder for all 8 weight tensors AND
the latent-hi at decode time)."""


# Brotli adaptive-lgwin search bounds. Brotli accepts lgwin in [10, 24].
# Smaller lgwin = smaller window = less header overhead but worse compression
# for long inputs. PR103's "adaptive lgwin search" picks the best per stream.
ADAPTIVE_LGWIN_MIN = 10
ADAPTIVE_LGWIN_MAX = 24

# Brotli quality must match Op 1 (PR101 ships at 11).
DEFAULT_BROTLI_QUALITY = 11


# AC-side quant: PR103 stores int8 codes as ``(int8 + 128).astype(uint8)`` which
# is the "off" byte_map in PR101's terminology. Per AC symbol the alphabet is
# 0..255.
AC_SYMBOL_OFFSET = 128


# Latent-hi stream constants. PR103's latent-hi is the high-byte half of the
# uint16 zigzag-coded delta latents (600 pairs x 28 dims = 16,800 symbols).
# Histogram is uint16 (because the alphabet of a high-byte distribution is
# usually very sparse — often only a few unique values), brotli-compressed.
LATENT_HI_DTYPE = np.uint16


class Pr103ArithmeticCodecError(Pr101SplitBrotliCodecError):
    """Raised when a PR103 arithmetic-coded payload is invalid or input mismatched."""


# ---------------------------------------------------------------------------
# Range-coder primitive helpers (constriction wrappers)
# ---------------------------------------------------------------------------

def _make_categorical(weights: np.ndarray) -> object:
    """Build a constriction Categorical model from raw integer histogram weights.

    Verbatim port of PR103 inflate.py:103-107 ``make_categorical``. Weights
    must be > 0 (we apply the same 1e-10 floor PR103 uses to dodge log(0)).

    The returned model is suitable as the second arg to
    ``RangeEncoder.encode(symbol, model)`` and ``RangeDecoder.decode(model)``.
    """
    p = weights.astype(np.float64)
    p = np.maximum(p, 1e-10)
    p = p / p.sum()
    return constriction.stream.model.Categorical(p, perfect=False)


def _build_q8_histogram(symbols_u8: np.ndarray) -> np.ndarray:
    """Build a 256-bucket uint8-clamped histogram for a stream of u8 symbols.

    Returns an array of 256 uint8 weights. Each weight is clamped to [1, 255]
    so the stored histogram can be a 256-byte uint8 array (PR103's wire format).
    """
    if symbols_u8.dtype != np.uint8:
        raise Pr103ArithmeticCodecError(
            f"q8 histogram requires uint8 symbols, got dtype={symbols_u8.dtype}"
        )
    counts = np.bincount(symbols_u8, minlength=256).astype(np.float64)
    if counts.sum() == 0:
        # Degenerate: empty stream. Use uniform.
        return np.ones(256, dtype=np.uint8)
    # Renormalize counts to fit in uint8 [1, 255]; preserve relative
    # frequencies as best we can. PR103 stores the raw uint8 histogram at
    # encode time and uses it directly as the Categorical weights at decode.
    if counts.max() <= 255:
        # All counts already fit; only floor zero counts at 1.
        h = np.maximum(counts, 1.0).astype(np.uint8)
    else:
        # Scale so max → 255, floor at 1.
        scale = 255.0 / counts.max()
        scaled = np.round(counts * scale)
        scaled = np.where(counts > 0, np.maximum(scaled, 1.0), scaled)
        h = np.clip(scaled, 0, 255).astype(np.uint8)
        # Re-floor any nonzero buckets back to >= 1 (rounding may have zeroed them).
        h = np.where((counts > 0) & (h == 0), 1, h).astype(np.uint8)
    return h


def _build_latent_hi_histogram(symbols_u16: np.ndarray) -> np.ndarray:
    """Build a u16 histogram for the latent-hi stream.

    Latent-hi has a sparse alphabet (usually only a handful of distinct
    values), so PR103 stores it as a compact uint16 array. Length equals the
    max-symbol-value + 1, brotli'd. We mirror that exactly.
    """
    if symbols_u16.dtype != np.uint16 and symbols_u16.dtype != np.int32:
        raise Pr103ArithmeticCodecError(
            f"latent-hi histogram requires uint16/int32 symbols, "
            f"got dtype={symbols_u16.dtype}"
        )
    s = symbols_u16.astype(np.int64)
    if s.size == 0:
        # Empty stream: return a valid length-2 uniform histogram so any
        # subsequent constriction.Categorical(p) call won't fail. Round 1
        # review (Yousfi finding): single-bucket Categorical raises.
        return np.ones(2, dtype=np.uint16)
    max_sym = int(s.max())
    # Round 1 review (Yousfi finding): constriction.Categorical requires
    # at least 2 buckets — a single-symbol stream (e.g. all-zero latent-hi
    # in synthetic tests) hits this path. Pad to length 2 minimum.
    bucket_count = max(max_sym + 1, 2)
    counts = np.bincount(s.astype(np.int64), minlength=bucket_count).astype(np.int64)
    # Clamp counts to uint16 range (rare in practice — N_PAIRS*LATENT_DIM=16800
    # so max count fits comfortably in u16). Floor positive buckets at 1.
    counts = np.where(counts > 65535, 65535, counts)
    counts = np.where((counts == 0), 0, np.maximum(counts, 1))
    return counts.astype(np.uint16)


def pack_ac_stream(
    symbols: np.ndarray,
    histogram: np.ndarray,
) -> bytes:
    """Encode one stream of symbols via constriction RangeEncoder using the
    given histogram. Returns the encoder's compressed buffer as raw bytes.

    Wire format: the underlying RangeEncoder produces a uint32 array; we
    emit it as little-endian bytes via ``.tobytes()``. PR103 uses the same
    convention (``np.frombuffer(merged_ac, dtype=np.uint32)`` at decode).

    NB: this primitive encodes ONE stream. The merged-encoder optimization
    (PR103 trick #6) wraps multiple streams in a single encoder — see
    :func:`encode_decoder_ac` for the full path. ``pack_ac_stream`` is
    exposed primarily for unit-testing the per-stream path and for measuring
    AC-vs-brotli on individual tensors in :func:`validate_ac_savings`.
    """
    cat = _make_categorical(histogram)
    enc = constriction.stream.queue.RangeEncoder()
    for s in symbols:
        enc.encode(int(s), cat)
    buf = enc.get_compressed()
    return buf.tobytes()


def unpack_ac_stream(
    blob: bytes,
    histogram: np.ndarray,
    n_symbols: int,
) -> np.ndarray:
    """Decode one AC stream encoded by :func:`pack_ac_stream`.

    Mirrors :func:`pack_ac_stream` exactly. Returns an int32 array of length
    ``n_symbols``.
    """
    if len(blob) % 4 != 0:
        raise Pr103ArithmeticCodecError(
            f"unpack_ac_stream expects 4-byte-aligned blob, got len={len(blob)}"
        )
    buf = np.frombuffer(blob, dtype=np.uint32)
    cat = _make_categorical(histogram)
    dec = constriction.stream.queue.RangeDecoder(buf)
    out = np.zeros(n_symbols, dtype=np.int32)
    for i in range(n_symbols):
        out[i] = dec.decode(cat)
    return out


# ---------------------------------------------------------------------------
# Adaptive lgwin search (Trick #5)
# ---------------------------------------------------------------------------

def adaptive_lgwin_search(
    raw: bytes,
    *,
    quality: int = DEFAULT_BROTLI_QUALITY,
    lgwin_min: int = ADAPTIVE_LGWIN_MIN,
    lgwin_max: int = ADAPTIVE_LGWIN_MAX,
) -> tuple[int, bytes]:
    """For ``raw`` bytes, find the brotli ``lgwin`` (window size in [10, 24])
    that minimizes the compressed output. Returns (best_lgwin, best_compressed).

    Brotli's default ``lgwin`` is 22; the optimum is input-length-dependent
    — for short inputs (<<1MB) a smaller window saves header bytes without
    hurting compression. PR103 sweeps per-stream.

    Tradeoff: search is O((lgwin_max - lgwin_min + 1)) brotli calls per
    stream. At quality=11 each call costs ~50ms for a 50KB input → ~750ms
    per stream × 7 non-AC streams ≈ 5s total. Acceptable for offline
    encoding; a no-op at decode time.
    """
    if lgwin_min < ADAPTIVE_LGWIN_MIN or lgwin_max > ADAPTIVE_LGWIN_MAX:
        raise Pr103ArithmeticCodecError(
            f"lgwin out of range: requested [{lgwin_min}, {lgwin_max}] "
            f"vs supported [{ADAPTIVE_LGWIN_MIN}, {ADAPTIVE_LGWIN_MAX}]"
        )
    best_lgwin: int | None = None
    best_compressed: bytes | None = None
    for lgwin in range(lgwin_min, lgwin_max + 1):
        try:
            comp = brotli.compress(raw, quality=quality, lgwin=lgwin)
        except brotli.error:
            # Some lgwin values may fail on tiny inputs — skip.
            continue
        if best_compressed is None or len(comp) < len(best_compressed):
            best_compressed = comp
            best_lgwin = lgwin
    if best_lgwin is None or best_compressed is None:
        # Fallback: vanilla pack at default lgwin via pack_brotli_stream.
        best_compressed = pack_brotli_stream(raw, quality=quality)
        best_lgwin = 22  # brotli's default
    return best_lgwin, best_compressed


# ---------------------------------------------------------------------------
# Top-level encode (Trick #4 + #5 + #6 composed)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EncodedAcDecoderBlob:
    """Encoder output dataclass. Public so callers can audit per-stream sizes."""

    blob: bytes
    """Final concatenated decoder bytes (the "Op 2 blob")."""

    non_ac_brotli_streams: tuple[bytes, ...]
    """Per-stream brotli outputs for tensors NOT in AC_TENSOR_INDICES, after
    adaptive lgwin search. Length equals len(DECODER_STREAM_ENDS) minus any
    streams that contain AC tensors (those streams' AC-tensor portions move
    to the merged AC blob)."""

    merged_ac_blob: bytes
    """The single merged RangeEncoder output for the AC weight tensors that did
    NOT fall back to brotli + latent-hi.

    Empty if no AC tensors were found in the input (shouldn't happen on a
    standard FIXED_STATE_SCHEMA state_dict) OR if every AC tensor fell back."""

    histograms_blob: bytes
    """Brotli-compressed concatenation of 256-byte uint8 histograms for the AC
    tensors that did NOT fall back (length = (8 - len(ac_fallback_set)) x 256
    pre-brotli)."""

    latent_hi_hist_blob: bytes
    """Brotli-compressed uint16 latent-hi histogram (sparse alphabet)."""

    fp16_scales: bytes
    """Per-tensor fp16 scales, schema order (28 x 2 = 56 bytes)."""

    selected_lgwins: tuple[int, ...]
    """The lgwin chosen by adaptive_lgwin_search for each non-AC brotli stream
    (parallel to non_ac_brotli_streams)."""

    ac_fallback_set: tuple[int, ...] = ()
    """Sorted tuple of AC tensor indices that regressed vs brotli and fell back
    to brotli encoding. Empty when ``ac_auto_fallback=False`` or when no AC
    tensor regressed. Decoder must consult this set to route reconstruction
    correctly. Substrate-mismatch protection per CLAUDE.md
    (Contrarian gate)."""

    ac_fallback_blob: bytes = b""
    """Single brotli stream containing the concatenation of fallback tensors'
    'off' byte-mapped (``int8+128``) u8 payloads. Tensors appear in
    ``ac_fallback_set`` order (which matches their appearance in
    ``AC_TENSOR_INDICES``). Empty when ``ac_fallback_set`` is empty.
    """

    ac_fallback_lgwin: int = 22
    """The lgwin chosen by adaptive_lgwin_search for ``ac_fallback_blob``.
    Defaults to 22 (brotli's default) when the fallback blob is empty."""


def _ac_histograms_blob(per_tensor_hists: list[np.ndarray]) -> bytes:
    """Concatenate 8 × 256-byte uint8 histograms and brotli-compress.

    Matches PR103 inflate.py:141:
        ``hists = np.frombuffer(brotli.decompress(hists_b), dtype=np.uint8)
                  .reshape(len(AC_INDICES), 256)``
    """
    concat = b"".join(h.tobytes() for h in per_tensor_hists)
    return brotli.compress(concat, quality=DEFAULT_BROTLI_QUALITY)


def _encode_merged_ac(
    weight_streams_u8: list[np.ndarray],
    weight_histograms: list[np.ndarray],
    hi_symbols: np.ndarray | None = None,
    hi_histogram: np.ndarray | None = None,
) -> bytes:
    """Encode all weight streams + (optionally) latent-hi into ONE RangeEncoder.

    PR103 trick #6: a single encoder eliminates per-stream rounding overhead
    (the "tail" of each independent encoder's output rounds up to the next
    uint32 word; merging removes that overhead per stream).

    Order matters and is documented in PR103 inflate.py:123-136
    ``decode_merged_ac``: 8 weight streams in AC_TENSOR_INDICES order, then
    the latent-hi stream.
    """
    enc = constriction.stream.queue.RangeEncoder()
    for syms_u8, hist in zip(weight_streams_u8, weight_histograms, strict=True):
        cat = _make_categorical(hist)
        for s in syms_u8:
            enc.encode(int(s), cat)
    if hi_symbols is not None:
        if hi_histogram is None:
            raise Pr103ArithmeticCodecError(
                "encode_merged_ac: hi_symbols passed without hi_histogram"
            )
        hi_cat = _make_categorical(hi_histogram)
        for s in hi_symbols:
            enc.encode(int(s), hi_cat)
    buf = enc.get_compressed()
    return buf.tobytes()


def encode_decoder_ac(
    state_dict: dict[str, torch.Tensor],
    *,
    brotli_quality: int = DEFAULT_BROTLI_QUALITY,
    adaptive_lgwin: bool = True,
    latent_hi_symbols: np.ndarray | None = None,
    return_layout: bool = False,
    ac_auto_fallback: bool = True,
) -> bytes | EncodedAcDecoderBlob:
    """Encode a torch ``state_dict`` to a PR103-compatible decoder bytes blob.

    The output blob structure (mirrors PR103 inflate.py:110-120 ``parse_archive``,
    decoder section only — caller is responsible for splicing latents/sidecar
    on top):

        [scales:       SCA_LEN = 28 * 2 = 56 bytes (fp16 per tensor, schema order)]
        [br_blob:      variable                    (concat of N non-AC brotli streams)]
        [hists_b:      variable                    (brotli of (8-K) × 256-byte u8 histograms,
                                                    where K = len(ac_fallback_set))]
        [merged_ac:    variable                    (single RangeEncoder output for the
                                                    (8-K) AC tensors + latent-hi)]
        [hi_hist_b:    variable                    (brotli of u16 latent-hi histogram)]
        [ac_fb_blob:   variable                    (NEW — brotli of fallback tensors,
                                                    empty when ac_fallback_set is empty)]

    Wire-format compatibility note: PR103's full archive layout includes
    latent metadata + lo_b + wrp_b sections AFTER the AC + hi_hist blocks.
    This function emits ONLY the decoder section — the full-archive build
    is owned by ``experiments.build_pr103_repacked_archive`` (which is the
    canonical caller of this function). The latent-hi symbols are an
    optional pass-through input because (a) the latent stream is OWNED by
    the latent-section codec, not by us, and (b) at the decoder side
    PR103's parser invokes ``decode_merged_ac`` knowing both weights and
    latent-hi were merged into one encoder — so the encoder side must
    accept latent-hi as input to maintain that contract. When
    ``latent_hi_symbols=None`` we emit only weight AC (still wire-format
    compatible with a caller that has no latent stream).

    Per-tensor AC auto-fallback (Contrarian gate landed 2026-05-07):
    When ``ac_auto_fallback=True`` (default), each tensor in ``AC_TENSOR_INDICES``
    is measured against a brotli baseline via :func:`validate_ac_savings`.
    Tensors that REGRESS vs brotli (positive ``delta_bytes``) are diverted to
    a brotli stream (``ac_fb_blob``) instead of the merged AC encoder; the
    decoder consults ``ac_fallback_set`` (carried in op_state) to route
    reconstruction. Substrate-mismatch protection: AC's q8 histogram + entropy
    assumption fits PR103's fine-tuned weights but regresses badly on int6/int7
    quantized substrates (verified 2026-05-07: -11,498 B savings on int6
    substrate).

    Args:
        state_dict: HNeRVDecoder state dict keyed by FIXED_STATE_SCHEMA names.
        brotli_quality: Brotli compression level (PR103 ships at 11).
        adaptive_lgwin: Apply trick #5 (per-stream lgwin search) for non-AC
            brotli streams. Default True.
        latent_hi_symbols: Optional uint16/int32 array of latent-hi symbols
            (length = N_PAIRS * LATENT_DIM = 16800). When given, merged into
            the same RangeEncoder as the weight streams (PR103 trick #6).
        return_layout: When True, returns the EncodedAcDecoderBlob dataclass
            with per-section breakdown for debugging / measurement. Default
            False (returns just the bytes).
        ac_auto_fallback: When True (default), each AC tensor is measured
            against brotli and falls back to brotli if it regresses. Setting
            to False reproduces the strict pre-2026-05-07 PR103-faithful
            behavior (every AC_TENSOR_INDICES tensor goes through AC
            unconditionally). Substrate-mismatch protection per Contrarian
            gate; see ``ac_fallback_set`` in the returned layout.
    """
    # 1) Quantize every tensor; build fp16 scale table.
    schema_names = {name for name, _ in FIXED_STATE_SCHEMA}
    for name in schema_names:
        if name not in state_dict:
            raise Pr103ArithmeticCodecError(
                f"missing tensor {name!r} in state_dict"
            )
    quantized = [
        _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        for name, _shape in FIXED_STATE_SCHEMA
    ]
    fp16_scales = np.array([qt.scale for qt in quantized], dtype=np.float16).tobytes()

    # 2) Identify which tensors go to AC vs non-AC paths.
    ac_set = set(AC_TENSOR_INDICES)

    # 2b) Per-tensor AC auto-fallback: measure each AC tensor's AC bytes vs
    #     brotli bytes. Tensors that regress (positive delta) get diverted to
    #     a fallback brotli stream. This is the substrate-mismatch protection
    #     gate landed 2026-05-07: AC's q8 histogram model fits PR103's own
    #     fine-tuned weights but regresses on int6/int7-quantized substrates.
    ac_fallback_set: tuple[int, ...] = ()
    if ac_auto_fallback:
        per_tensor_audit: list[tuple[int, int]] = []  # (idx, delta_bytes)
        for idx in AC_TENSOR_INDICES:
            qt = quantized[idx]
            u8 = (qt.q_i8.astype(np.int16) + AC_SYMBOL_OFFSET).astype(np.uint8).reshape(-1)
            hist = _build_q8_histogram(u8)
            ac_size = len(pack_ac_stream(u8, hist))
            brotli_size = len(pack_brotli_stream(u8.tobytes(), quality=brotli_quality))
            per_tensor_audit.append((idx, ac_size - brotli_size))
        ac_fallback_set = tuple(
            sorted(idx for idx, delta in per_tensor_audit if delta > 0)
        )
        if ac_fallback_set:
            logger.info(
                "PR103 AC per-tensor fallback fired for tensor indices %s "
                "(saved %d total brotli-vs-AC bytes)",
                ac_fallback_set,
                sum(max(0, d) for _, d in per_tensor_audit),
            )

    fallback_set_for_loop = set(ac_fallback_set)

    # 3) Build merged AC: u8 symbols (off byte_map: int8 + 128) for each
    #    AC tensor that did NOT fall back, paired with a q8 histogram.
    weight_streams_u8: list[np.ndarray] = []
    weight_histograms: list[np.ndarray] = []
    ac_active_indices: list[int] = []  # AC_TENSOR_INDICES order, fallbacks dropped
    for idx in AC_TENSOR_INDICES:
        if idx in fallback_set_for_loop:
            continue
        qt = quantized[idx]
        # AC encoding uses the "off" byte_map (signed offset binary): see
        # PR103 inflate.py:147 ``(weight_arrays[k] - 128).astype(np.int8)``.
        u8 = (qt.q_i8.astype(np.int16) + AC_SYMBOL_OFFSET).astype(np.uint8).reshape(-1)
        hist = _build_q8_histogram(u8)
        weight_streams_u8.append(u8)
        weight_histograms.append(hist)
        ac_active_indices.append(idx)

    # 4) Build latent-hi histogram + merged AC stream.
    if latent_hi_symbols is not None:
        latent_hi_arr = np.asarray(latent_hi_symbols)
        if latent_hi_arr.dtype not in (np.uint16, np.int32, np.int64):
            latent_hi_arr = latent_hi_arr.astype(np.uint16)
        hi_hist = _build_latent_hi_histogram(latent_hi_arr.astype(np.uint16))
    else:
        latent_hi_arr = None
        hi_hist = np.zeros(0, dtype=np.uint16)

    if weight_streams_u8 or latent_hi_arr is not None:
        merged_ac = _encode_merged_ac(
            weight_streams_u8,
            weight_histograms,
            hi_symbols=latent_hi_arr,
            hi_histogram=hi_hist if latent_hi_arr is not None else None,
        )
    else:
        # All AC tensors fell back AND no latent-hi: emit empty merged stream.
        merged_ac = b""

    # 5) Build histograms blob ((8 - K) x 256 bytes brotli'd, K = fallbacks).
    histograms_blob = _ac_histograms_blob(weight_histograms) if weight_histograms else b""

    # 5b) Build AC-fallback brotli blob: concatenate u8 'off' byte-mapped
    #     payloads for fallback tensors in AC_TENSOR_INDICES order, then brotli.
    if ac_fallback_set:
        fallback_parts: list[bytes] = []
        # Iterate AC_TENSOR_INDICES order so the decoder can reconstruct
        # by walking the same order and slicing per-tensor counts.
        for idx in AC_TENSOR_INDICES:
            if idx not in fallback_set_for_loop:
                continue
            qt = quantized[idx]
            u8 = (qt.q_i8.astype(np.int16) + AC_SYMBOL_OFFSET).astype(np.uint8).reshape(-1)
            fallback_parts.append(u8.tobytes())
        fallback_raw = b"".join(fallback_parts)
        if adaptive_lgwin:
            ac_fallback_lgwin, ac_fallback_blob = adaptive_lgwin_search(
                fallback_raw, quality=brotli_quality
            )
        else:
            ac_fallback_blob = pack_brotli_stream(fallback_raw, quality=brotli_quality)
            ac_fallback_lgwin = 22
    else:
        ac_fallback_blob = b""
        ac_fallback_lgwin = 22

    # 6) Latent-hi histogram blob: brotli of u16 histogram raw bytes. If no
    #    latent-hi was supplied, emit a 0-length sentinel so the layout stays
    #    deterministic.
    if latent_hi_arr is not None:
        latent_hi_hist_blob = brotli.compress(
            hi_hist.tobytes(), quality=brotli_quality
        )
    else:
        latent_hi_hist_blob = b""

    # 7) Build non-AC brotli streams. The non-AC tensors keep PR101's
    #    split-stream layout (DECODER_STORAGE_ORDER + DECODER_STREAM_ENDS),
    #    BUT each stream has any AC tensors removed. We drop the AC tensor's
    #    payload bytes; what remains is the stream's non-AC tensors. A stream
    #    that becomes EMPTY after AC removal is skipped (no zero-byte brotli).
    storage_to_window: dict[int, int] = {}
    start = 0
    for w_idx, end in enumerate(DECODER_STREAM_ENDS):
        for so_pos in range(start, end):
            storage_to_window[DECODER_STORAGE_ORDER[so_pos]] = w_idx
        start = end

    non_ac_streams_raw: list[bytes] = []
    start = 0
    for end in DECODER_STREAM_ENDS:
        # For this stream window, gather payloads of the non-AC tensors only.
        # Each tensor's on-disk payload is mapped-bytes (NO scale — scale moved
        # to fp16_scales section per the Op 2 layout). Use 'twos' byte_map so
        # the AC and non-AC tensors share the same encoding convention as PR101
        # for non-AC tensors. Actually PR103 does NOT change byte_map for
        # non-AC tensors — it keeps PR101's defaults. We honor that.
        window_parts: list[bytes] = []
        for so_pos in range(start, end):
            tensor_idx = DECODER_STORAGE_ORDER[so_pos]
            if tensor_idx in ac_set:
                continue  # this tensor goes through AC, skip in brotli streams
            qt = quantized[tensor_idx]
            payload = _build_per_tensor_payload(qt, tensor_idx)
            # _build_per_tensor_payload appends fp16 scale; in Op 2 layout we
            # carry scales in the dedicated section, so strip the trailing 2
            # bytes here.
            payload = payload[:-2]
            window_parts.append(payload)
        if window_parts:
            non_ac_streams_raw.append(b"".join(window_parts))
        start = end

    # 8) Compress each non-empty raw stream — adaptive lgwin if requested.
    non_ac_streams_compressed: list[bytes] = []
    selected_lgwins: list[int] = []
    for raw in non_ac_streams_raw:
        if adaptive_lgwin:
            lgwin, comp = adaptive_lgwin_search(raw, quality=brotli_quality)
        else:
            comp = pack_brotli_stream(raw, quality=brotli_quality)
            lgwin = 22  # brotli default
        non_ac_streams_compressed.append(comp)
        selected_lgwins.append(lgwin)

    br_blob = b"".join(non_ac_streams_compressed)

    # 9) Concatenate the final blob. NEW section (ac_fallback_blob) is appended
    #    AFTER latent_hi_hist_blob so callers that don't care about fallback
    #    can ignore the trailing bytes safely (it is empty when no fallback
    #    fired).
    final = b"".join([
        fp16_scales,
        br_blob,
        histograms_blob,
        merged_ac,
        latent_hi_hist_blob,
        ac_fallback_blob,
    ])

    if return_layout:
        return EncodedAcDecoderBlob(
            blob=final,
            non_ac_brotli_streams=tuple(non_ac_streams_compressed),
            merged_ac_blob=merged_ac,
            histograms_blob=histograms_blob,
            latent_hi_hist_blob=latent_hi_hist_blob,
            fp16_scales=fp16_scales,
            selected_lgwins=tuple(selected_lgwins),
            ac_fallback_set=ac_fallback_set,
            ac_fallback_blob=ac_fallback_blob,
            ac_fallback_lgwin=ac_fallback_lgwin,
        )
    return final


# ---------------------------------------------------------------------------
# Top-level decode
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecodedAcDecoderBlob:
    """Decoder output dataclass. Returns state_dict + (optional) latent-hi."""

    state_dict: dict[str, torch.Tensor]
    latent_hi_symbols: np.ndarray | None


def decode_decoder_ac(
    blob: bytes,
    *,
    section_lengths: dict[str, int] | None = None,
    n_latent_hi_symbols: int = 0,
    ac_fallback_set: tuple[int, ...] = (),
) -> DecodedAcDecoderBlob:
    """Decode a PR103-format Op 2 blob to (state_dict, latent_hi_symbols).

    The PR103 wire format does NOT carry length prefixes — section lengths
    are HARDCODED in PR103's inflate.py (PR103 trick #1). For deterministic
    round-trip we require the caller to pass ``section_lengths`` or to have
    invoked :func:`encode_decoder_ac` with ``return_layout=True`` and used
    the resulting per-section sizes as input here.

    Args:
        blob: bytes produced by :func:`encode_decoder_ac` with the same
            section ordering:
            scales | br_blob | hists | merged_ac | hi_hist | ac_fallback.
        section_lengths: dict with keys ``br``, ``hists``, ``merged_ac``,
            ``hi_hist``. Optional ``ac_fallback`` key (defaults to 0). The
            scales section is fixed at 28*2=56 bytes; the others are variable.
        n_latent_hi_symbols: how many latent-hi symbols to drain from the
            merged RangeDecoder AFTER the AC weight streams. Set to 16800 for
            standard HNeRV (600 pairs × 28 dims), 0 if no latent-hi was
            embedded.
        ac_fallback_set: sorted tuple of AC tensor indices whose payloads were
            diverted to the brotli fallback section (per the per-tensor
            auto-fallback gate). Empty when no fallback fired. Decoder consults
            this to route reconstruction.

    Returns:
        DecodedAcDecoderBlob with state_dict + latent-hi symbols.

    Limitation: a fully self-describing wire format would store the section
    lengths in a small header. PR103 chose to hardcode them; we honor that
    by requiring the caller to pass them explicitly. The
    :class:`EncodedAcDecoderBlob` dataclass returned by encode-side carries
    all section sizes for round-trip.
    """
    if section_lengths is None:
        raise Pr103ArithmeticCodecError(
            "decode_decoder_ac requires explicit section_lengths "
            "(use encode_decoder_ac(return_layout=True) to capture sizes)"
        )
    required = {"br", "hists", "merged_ac", "hi_hist"}
    missing = required - section_lengths.keys()
    if missing:
        raise Pr103ArithmeticCodecError(
            f"section_lengths missing keys: {sorted(missing)}"
        )

    fallback_set = set(ac_fallback_set)
    if not fallback_set.issubset(set(AC_TENSOR_INDICES)):
        raise Pr103ArithmeticCodecError(
            f"ac_fallback_set {sorted(fallback_set)!r} contains indices "
            f"outside AC_TENSOR_INDICES {AC_TENSOR_INDICES!r}"
        )
    ac_active_indices: tuple[int, ...] = tuple(
        idx for idx in AC_TENSOR_INDICES if idx not in fallback_set
    )

    sca_len = len(FIXED_STATE_SCHEMA) * 2
    br_len = section_lengths["br"]
    hists_len = section_lengths["hists"]
    merged_ac_len = section_lengths["merged_ac"]
    hi_hist_len = section_lengths["hi_hist"]
    ac_fallback_len = int(section_lengths.get("ac_fallback", 0))
    expected = (
        sca_len + br_len + hists_len + merged_ac_len + hi_hist_len + ac_fallback_len
    )
    if expected != len(blob):
        raise Pr103ArithmeticCodecError(
            f"section_lengths sum ({expected}) != blob len ({len(blob)})"
        )

    o = 0
    sca = blob[o:o + sca_len]
    o += sca_len
    br_b = blob[o:o + br_len]
    o += br_len
    hists_b = blob[o:o + hists_len]
    o += hists_len
    merged_ac = blob[o:o + merged_ac_len]
    o += merged_ac_len
    hi_hist_b = blob[o:o + hi_hist_len]
    o += hi_hist_len
    ac_fallback_b = blob[o:o + ac_fallback_len]
    o += ac_fallback_len
    assert o == len(blob)

    fp16_scales = np.frombuffer(sca, dtype=np.float16)
    if len(fp16_scales) != len(FIXED_STATE_SCHEMA):
        raise Pr103ArithmeticCodecError(
            f"fp16_scales length mismatch: got {len(fp16_scales)} "
            f"expected {len(FIXED_STATE_SCHEMA)}"
        )

    # Decode the histograms blob. With per-tensor fallback the hists count is
    # len(ac_active_indices), not len(AC_TENSOR_INDICES).
    n_ac_active = len(ac_active_indices)
    if hists_len > 0:
        hists_raw = brotli.decompress(hists_b)
        if n_ac_active == 0:
            hists = np.zeros((0, 256), dtype=np.uint8)
        else:
            hists = np.frombuffer(hists_raw, dtype=np.uint8).reshape(
                n_ac_active, 256
            )
    else:
        hists = np.zeros((n_ac_active, 256), dtype=np.uint8)

    # Decode latent-hi histogram (if present).
    if hi_hist_len > 0:
        hi_hist_raw = brotli.decompress(hi_hist_b)
        hi_hist = np.frombuffer(hi_hist_raw, dtype=np.uint16)
    else:
        hi_hist = np.zeros(0, dtype=np.uint16)

    # Decode merged AC: ac_active_indices weight streams of known size,
    # then n_latent_hi_symbols.
    weight_arrays_u8_by_idx: dict[int, np.ndarray] = {}
    if merged_ac_len > 0:
        if merged_ac_len % 4 != 0:
            raise Pr103ArithmeticCodecError(
                f"merged_ac len not 4-aligned: {merged_ac_len}"
            )
        buf = np.frombuffer(merged_ac, dtype=np.uint32)
        dec = constriction.stream.queue.RangeDecoder(buf)
        for k, idx in enumerate(ac_active_indices):
            cat = _make_categorical(hists[k])
            count = int(np.prod(FIXED_STATE_SCHEMA[idx][1]))
            arr = np.zeros(count, dtype=np.int32)
            for i in range(count):
                arr[i] = dec.decode(cat)
            weight_arrays_u8_by_idx[idx] = arr.astype(np.int32)
        if n_latent_hi_symbols > 0:
            hi_cat = _make_categorical(hi_hist)
            hi_out = np.zeros(n_latent_hi_symbols, dtype=np.int32)
            for i in range(n_latent_hi_symbols):
                hi_out[i] = dec.decode(hi_cat)
        else:
            hi_out = None
    else:
        hi_out = None

    # Map AC arrays back to int8 (subtract +128 offset) and reshape.
    ac_arrays: dict[int, np.ndarray] = {}
    for idx, arr in weight_arrays_u8_by_idx.items():
        shape = FIXED_STATE_SCHEMA[idx][1]
        i8 = (arr - AC_SYMBOL_OFFSET).astype(np.int8).reshape(shape)
        ac_arrays[idx] = i8

    # Decode the AC-fallback brotli blob (if any) and slice per-tensor in
    # AC_TENSOR_INDICES order matching the encoder's fallback packing.
    if ac_fallback_len > 0:
        if not fallback_set:
            raise Pr103ArithmeticCodecError(
                "ac_fallback section non-empty but ac_fallback_set is empty "
                "(encoder/decoder out of sync)"
            )
        fallback_raw = brotli.decompress(ac_fallback_b)
        fb_pos = 0
        for idx in AC_TENSOR_INDICES:
            if idx not in fallback_set:
                continue
            shape = FIXED_STATE_SCHEMA[idx][1]
            n_el = int(np.prod(shape))
            chunk = fallback_raw[fb_pos:fb_pos + n_el]
            fb_pos += n_el
            if len(chunk) != n_el:
                raise Pr103ArithmeticCodecError(
                    f"ac_fallback truncated at idx={idx}: expected {n_el} got {len(chunk)}"
                )
            u8 = np.frombuffer(chunk, dtype=np.uint8)
            i8 = (u8.astype(np.int16) - AC_SYMBOL_OFFSET).astype(np.int8).reshape(shape)
            ac_arrays[idx] = i8
        if fb_pos != len(fallback_raw):
            raise Pr103ArithmeticCodecError(
                f"ac_fallback decompressed bytes had trailing data: "
                f"pos={fb_pos} len={len(fallback_raw)}"
            )
    elif fallback_set:
        raise Pr103ArithmeticCodecError(
            f"ac_fallback_set {sorted(fallback_set)} non-empty but "
            f"ac_fallback section length is zero"
        )

    # Decode non-AC brotli streams.
    n_streams = sum(
        1
        for w_idx, _ in enumerate(DECODER_STREAM_ENDS)
        if any(
            DECODER_STORAGE_ORDER[so_pos] not in set(AC_TENSOR_INDICES)
            for so_pos in range(
                0 if w_idx == 0 else DECODER_STREAM_ENDS[w_idx - 1],
                DECODER_STREAM_ENDS[w_idx],
            )
        )
    )
    if br_len > 0:
        # Streamed decompression: walk the bytes, peel off one stream at a
        # time. Mirrors PR101's decompress_brotli_streams.
        outputs: list[bytes] = []
        pos = 0
        data = br_b
        for _ in range(n_streams):
            decompressor = brotli.Decompressor()
            chunks: list[bytes] = []
            while pos < len(data) and not decompressor.is_finished():
                chunks.append(decompressor.process(data[pos:pos + 1]))
                pos += 1
            if not decompressor.is_finished():
                raise Pr103ArithmeticCodecError(
                    "truncated non-AC brotli payload"
                )
            outputs.append(b"".join(chunks))
        if pos != len(data):
            raise Pr103ArithmeticCodecError(
                "trailing non-AC brotli payload bytes"
            )
        non_ac_concat = b"".join(outputs)
    else:
        non_ac_concat = b""

    # Walk DECODER_STORAGE_ORDER and reassemble. AC tensors come from
    # ac_arrays; non-AC come from sequential bytes off non_ac_concat.
    sd: dict[str, torch.Tensor] = {}
    pos = 0
    for storage_idx in DECODER_STORAGE_ORDER:
        name, shape = FIXED_STATE_SCHEMA[storage_idx]
        if storage_idx in ac_arrays:
            i8 = ac_arrays[storage_idx]
        else:
            n_el = int(np.prod(shape))
            zz = np.frombuffer(non_ac_concat, dtype=np.uint8, count=n_el, offset=pos)
            pos += n_el
            from tac.pr101_split_brotli_codec import (
                CONV4_INVERSE_PERMS,
                CONV4_STORAGE_PERMS,
                DECODER_BYTE_MAPS,
                decode_mapped_u8,
            )
            decode_map = DECODER_BYTE_MAPS.get(storage_idx, "zig")
            i8 = decode_mapped_u8(zz, decode_map)
            if len(shape) == 4:
                storage_perm = CONV4_STORAGE_PERMS[storage_idx]
                inverse_perm = CONV4_INVERSE_PERMS[storage_idx]
                stored_shape = tuple(shape[i] for i in storage_perm)
                i8 = i8.reshape(stored_shape)
                i8 = np.transpose(i8, inverse_perm).copy()
            else:
                i8 = i8.reshape(shape)
        scale = float(fp16_scales[storage_idx])
        sd[name] = torch.from_numpy(i8.astype(np.float32)) * scale
    if pos != len(non_ac_concat):
        raise Pr103ArithmeticCodecError(
            f"non-AC concat had trailing bytes: pos={pos} len={len(non_ac_concat)}"
        )
    return DecodedAcDecoderBlob(state_dict=sd, latent_hi_symbols=hi_out)


# ---------------------------------------------------------------------------
# Contrarian gate: validate AC actually saves bytes per tensor (vs brotli)
# ---------------------------------------------------------------------------

def validate_ac_savings(
    state_dict: dict[str, torch.Tensor],
    *,
    brotli_quality: int = DEFAULT_BROTLI_QUALITY,
) -> dict[int, dict[str, int]]:
    """For each tensor in ``AC_TENSOR_INDICES``, measure the per-tensor AC
    bytes vs the per-tensor brotli bytes (PR101 baseline). Log warnings on
    regressions.

    Returns a dict keyed by schema-index containing:
        ac_bytes:           bytes after AC encoding (single-stream)
        brotli_bytes:       bytes after pack_brotli_stream(payload)
        delta_bytes:        ac_bytes - brotli_bytes (negative = AC wins)
        n_symbols:          number of u8 symbols in the AC stream

    PR103's -290 B claim should reproduce: most or all tensors should show
    delta_bytes < 0. If a tensor regresses (delta_bytes > 0), a WARNING is
    logged — operators can decide whether to skip AC for that tensor (which
    requires moving it back into a brotli stream).

    Note: this is a per-tensor comparison; it does NOT include the
    histogram-storage overhead (which is amortized across all 8 streams in
    the merged-AC blob). For a full empirical Op 2 vs Op 1 comparison, run
    ``experiments/build_pr103_repacked_archive.py`` end-to-end.
    """
    quantized = [
        _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        for name, _shape in FIXED_STATE_SCHEMA
    ]
    results: dict[int, dict[str, int]] = {}
    for idx in AC_TENSOR_INDICES:
        qt = quantized[idx]
        u8 = (qt.q_i8.astype(np.int16) + AC_SYMBOL_OFFSET).astype(np.uint8).reshape(-1)
        hist = _build_q8_histogram(u8)
        ac_bytes = pack_ac_stream(u8, hist)
        ac_size = len(ac_bytes)
        # Brotli baseline: same bytes, same byte_map (off — i.e. raw u8),
        # packed via pack_brotli_stream. Note this is NOT the per-tensor
        # split-Brotli context — it's an isolated brotli measurement, which
        # is the closest fair counterfactual for "is AC alone better than
        # brotli alone for this stream of bytes".
        brotli_size = len(pack_brotli_stream(u8.tobytes(), quality=brotli_quality))
        delta = ac_size - brotli_size
        results[idx] = {
            "ac_bytes": int(ac_size),
            "brotli_bytes": int(brotli_size),
            "delta_bytes": int(delta),
            "n_symbols": int(u8.size),
        }
        if delta > 0:
            logger.warning(
                "PR103 AC for tensor idx=%d (%s) REGRESSES vs brotli by "
                "%d bytes (ac=%d brotli=%d, n_symbols=%d) on this state_dict",
                idx,
                FIXED_STATE_SCHEMA[idx][0],
                delta,
                ac_size,
                brotli_size,
                u8.size,
            )
    return results


__all__ = [
    "AC_HISTOGRAM_BITS",
    "AC_SYMBOL_OFFSET",
    "AC_TENSOR_INDICES",
    "ADAPTIVE_LGWIN_MAX",
    "ADAPTIVE_LGWIN_MIN",
    "DEFAULT_BROTLI_QUALITY",
    "MERGED_RANGE_ENCODER",
    "DecodedAcDecoderBlob",
    "EncodedAcDecoderBlob",
    "Pr103ArithmeticCodecError",
    "adaptive_lgwin_search",
    "decode_decoder_ac",
    "encode_decoder_ac",
    "pack_ac_stream",
    "unpack_ac_stream",
    "validate_ac_savings",
]
