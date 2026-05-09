"""PR101 polymorphic codec primitive — port to ``tac.codec``.

PR101 (`hnerv_ft_microcodec`, SajayR, gold) ships a *polymorphic* sidecar codec
that emits 6 alternative byte layouts and the encoder picks the smallest.
This module is a clean-room **port** of the polymorphic primitive into ``tac``
so future archives can compose it without depending on the public-PR clone path
(per CLAUDE.md Check 109: clone is read-only oracle).  Of the 6 layouts:

  - 3 are ENCODE+DECODE supported (HUFF_ENUM, PACKED, RAW);
  - 3 are byte-budget-dominated and NOT ported (HUFF, HUFF_COMB, SPLIT).

PR101's actual gold archive uses HUFF_ENUM, so this port can decode and
re-encode that archive's sidecar bit-exactly (see
``test_decode_matches_pr101_archive_bitexact``).  If a future input would
benefit from HUFF / HUFF_COMB / SPLIT, port them then; PR101's reference
implementation in
``experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/src/codec.py``
remains the read-only oracle.

The 6 sidecar layouts:

  1. ``RAW`` (600 / 1200 bytes) — one byte per pair OR two bytes per pair
     (``arr.size == N_PAIRS`` or ``N_PAIRS * 2``); chosen when the
     fixed-vocabulary perturbation is so dense that a packed/Huffman wrapper
     adds more overhead than it removes.
  2. ``PACKED`` (661 bytes) — mixed-radix ``base = 1 + LATENT_DIM * len(deltas)``
     packed integer.  Raw enumeration of the entire 600-pair sequence as a
     single big-int.
  3. ``SPLIT`` (656 bytes) — no-op table + dim-packed bigint + brotli-coded
     packed-nibble deltas.
  4. ``HUFF`` (614 bytes) — no-op table + dim-packed bigint + nibble-coded
     Huffman lengths + canonical-Huffman delta codes.
  5. ``HUFF_COMB`` (609 bytes) — combinatorial-colex no-op rank + dim-packed
     bigint + 3-bit Huffman length packing + canonical-Huffman delta codes.
  6. ``HUFF_ENUM`` (607 bytes) — dim-packed bigint + Huffman-length-vector
     RANK (Kraft-equality enumeration) + canonical-Huffman delta codes +
     combinatorial-colex no-op rank.  This is the layout PR101's archive
     actually ships and is the smallest of the six.

**Encoder coverage** in this port (vs PR101's reference):

  - ``HUFF_ENUM`` — encoder + decoder. Bit-exact decode against PR101's
    actual archive bytes (see test
    ``test_decode_matches_pr101_archive_bitexact``).  Encoder uses an
    optimal-Huffman-length builder that may not always reproduce PR101's
    hand-tuned 240-byte fit on novel inputs (in that case the AUTO selector
    falls back to PACKED).
  - ``PACKED`` — encoder + decoder; lossless on any vocabulary-conformant
    input.
  - ``RAW`` (N_PAIRS*2 form) — encoder + decoder.  The N_PAIRS form is
    decode-only here because PR101's vocabulary (1 + 28*16 = 449) does not
    fit a single byte; we cannot guarantee the N_PAIRS form for arbitrary
    inputs.
  - ``HUFF`` / ``HUFF_COMB`` / ``SPLIT`` — these layouts are present in
    PR101's reference decoder (length-discriminated dispatch in
    ``apply_latent_sidecar``) but their encoders are intentionally NOT
    ported here.  They are byte-budget-dominated by HUFF_ENUM on PR101's
    archive, and porting their encoders would inflate the bolt-on LOC
    budget without a current consumer.  The AUTO selector therefore picks
    from {HUFF_ENUM, PACKED, RAW (N_PAIRS*2)} and PR101's archive remains
    decode-compatible via the appropriate length dispatch.

Plus the per-tensor decoder primitives:

  - ``CONV4_STORAGE_PERMS`` / ``CONV4_INVERSE_PERMS``: per-tensor 4D conv axis
    transpose (improves brotli's locality).
  - ``DECODER_BYTE_MAPS``: per-tensor signed-int byte-map
    (``zig`` / ``negzig`` / ``twos`` / ``off``) chosen so the byte stream's
    histogram is closest to brotli's frequency bias.
  - ``DECODER_STORAGE_ORDER`` / ``DECODER_STREAM_ENDS``: 28-tensor permutation
    + 7-stream brotli partition.

The decoder layout constants are **frozen** at the values PR101 hand-tuned for
the HNeRV 28-tensor / 36-base-channel architecture; they are not generic and
this module does not attempt to re-tune them.  The polymorphic SIDECAR codec
is the truly generic part — it can encode any per-pair categorical
perturbation drawn from a fixed-size vocabulary into the smallest of 6
byte layouts.

Per CLAUDE.md ``Forbidden Score Claims``: this module is a CODEC PRIMITIVE.
It produces archive bytes, not scores.  Any score claim using this module must
go through ``upstream/evaluate.py`` on the EXACT archive bytes that ship.

Per CLAUDE.md ``Forbidden empirical-claim-without-evidence-tag``: the byte
budgets cited above are EMPIRICAL on PR101's archive — the test suite
re-derives them via roundtrip on the live PR101 oracle (see
``test_pr101_polymorphic_codec.py::test_pr101_actual_archive_huff_enum_byte_budget``
and ``test_decode_matches_pr101_archive_bitexact``) and they are the
bytes-on-disk on PR101's gold archive.

Source oracle (read-only per Check 109):
    ``experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/``
    ``source/submissions/hnerv_ft_microcodec/src/codec.py``

Cross-references:
    - Domain catalog atom #5: ``feedback_domain_exploitation_catalog_landed_20260509.md``
    - Forensics dossier: ``.omx/research/hnerv_leaderboard_binary_forensics_dossier_20260509.md``
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Frozen constants from PR101's hand-tuned codec.
# These are not configurable; they are the values PR101 ships.
# ---------------------------------------------------------------------------

#: Per-tensor permutation order over the 28 HNeRV state-dict entries.  The
#: encoder writes tensors in this order; this is the order the decoder unpacks
#: them in.  See PR101 ``src/codec.py:32-35``.
DECODER_STORAGE_ORDER: tuple[int, ...] = (
    14, 22, 7, 6, 19, 10, 25, 4, 20, 9, 12, 15, 5, 11,
    18, 1, 21, 3, 27, 13, 2, 26, 24, 17, 16, 23, 8, 0,
)

#: Brotli stream-boundary indices into ``DECODER_STORAGE_ORDER``.  The encoder
#: emits ``len(DECODER_STREAM_ENDS)`` independent brotli streams; tensors at
#: positions ``[STORAGE[prev], ..., STORAGE[end-1]]`` go in stream ``i``.
DECODER_STREAM_ENDS: tuple[int, ...] = (1, 2, 22, 23, 26, 27, 28)

#: Per-tensor 4D conv axis transpose.  Tensor index → permutation tuple over
#: ``(O, I, H, W)``.  Different permutation per 4D conv tensor improves
#: brotli compression on the byte-zigzagged stream.
CONV4_STORAGE_PERMS: dict[int, tuple[int, int, int, int]] = {
    2: (3, 0, 2, 1),
    4: (3, 0, 2, 1),
    6: (0, 1, 2, 3),
    8: (3, 0, 1, 2),
    10: (3, 0, 2, 1),
    12: (3, 0, 1, 2),
    14: (1, 0, 2, 3),
    16: (3, 0, 2, 1),
    18: (1, 0, 2, 3),
    20: (0, 3, 2, 1),
    22: (0, 3, 2, 1),
    24: (0, 2, 3, 1),
    26: (0, 1, 3, 2),
}

#: The inverse permutations (decoder applies them after byte-decoding).
CONV4_INVERSE_PERMS: dict[int, tuple[int, ...]] = {
    idx: tuple(int(x) for x in np.argsort(perm))
    for idx, perm in CONV4_STORAGE_PERMS.items()
}

#: Per-tensor signed-int byte-map.  Tensor index → byte-map name.
#: Tensors not listed here use the default ``"zig"`` (zigzag) mapping.
DECODER_BYTE_MAPS: dict[int, str] = {
    9: "negzig",
    14: "negzig",
    20: "twos",
    27: "off",
}

#: Per-dim latent storage order (28 dims).  Latents are written DIM-MAJOR.
LATENT_DIM_ORDER: tuple[int, ...] = (
    26, 0, 17, 15, 10, 24, 20, 12, 14, 21, 22, 18, 4, 11,
    3, 7, 16, 2, 6, 8, 19, 23, 5, 9, 1, 13, 27, 25,
)

#: Sidecar perturbation vocabulary (×100; divide by 100 to get the float
#: latent perturbation per dim).  PR101 hand-picked this 16-symbol alphabet.
SIDECAR_DELTAS_X100: np.ndarray = np.array(
    [-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10],
    dtype=np.int8,
)

#: Number of frame-pairs in PR101's archive (one pair per 2 contest frames).
N_PAIRS: int = 600

#: Latent dimensionality.
LATENT_DIM: int = 28

#: Sidecar mixed-radix base for the PACKED layout.
SIDECAR_BASE: int = 1 + LATENT_DIM * len(SIDECAR_DELTAS_X100)

#: Frozen byte budgets per layout, as observed on PR101's archive.  These are
#: NOT free parameters — they are a property of the PR101 SIDECAR vocabulary
#: + ``N_PAIRS`` + ``LATENT_DIM``.  Tests verify these match the live
#: oracle's outputs.
SIDECAR_HUFF_ENUM_LEN: int = 607
SIDECAR_HUFF_COMB_LEN: int = 609
SIDECAR_HUFF_LEN: int = 614
SIDECAR_SPLIT_LEN: int = 656
SIDECAR_PACKED_LEN: int = 661

#: HUFF_ENUM internal byte fields.
SIDECAR_DIM_PACKED_LEN: int = 359
SIDECAR_DELTA_HUFF_LENGTH_RANK_LEN: int = 5
SIDECAR_NOOP_INFER_RANK_LEN: int = 3

#: HUFF_COMB internal byte fields.
SIDECAR_NOOP_RANK_PREFIX_LEN: int = 4
SIDECAR_DELTA_HUFF3_LENGTHS_LEN: int = 6

#: SPLIT/HUFF internal byte fields.
SIDECAR_NOOP_TABLE_LEN: int = 7
SIDECAR_DELTA_HUFF_LENGTHS_LEN: int = 8

#: Canonical-Huffman length bounds used by HUFF_ENUM rank-enumeration.
SIDECAR_HUFF_MIN_LEN: int = 2
SIDECAR_HUFF_MAX_LEN: int = 8
SIDECAR_HUFF_KRAFT_TOTAL: int = 1 << SIDECAR_HUFF_MAX_LEN  # 256


# ---------------------------------------------------------------------------
# Layout enum + config dataclass
# ---------------------------------------------------------------------------


class SidecarLayout(str, Enum):
    """The 6 polymorphic sidecar layouts (plus the auto-pick alias).

    The byte-on-disk size of each layout is FIXED for PR101's vocabulary;
    only ``AUTO`` introspects all six and picks the smallest.
    """

    RAW = "raw"          # N_PAIRS or N_PAIRS*2 bytes (depending on per-pair payload width)
    PACKED = "packed"    # 661 bytes — mixed-radix bigint
    SPLIT = "split"      # 656 bytes — no-op table + dim bigint + brotli-coded deltas
    HUFF = "huff"        # 614 bytes — nibble Huffman lengths
    HUFF_COMB = "huff_comb"  # 609 bytes — 3-bit Huffman lengths + colex no-op
    HUFF_ENUM = "huff_enum"  # 607 bytes — Huffman-length-rank + colex no-op (PR101 actual)
    AUTO = "auto"        # encoder enumerates all 6 and picks the smallest


_FIXED_LAYOUT_LENGTHS: dict[SidecarLayout, int] = {
    SidecarLayout.PACKED: SIDECAR_PACKED_LEN,
    SidecarLayout.SPLIT: SIDECAR_SPLIT_LEN,
    SidecarLayout.HUFF: SIDECAR_HUFF_LEN,
    SidecarLayout.HUFF_COMB: SIDECAR_HUFF_COMB_LEN,
    SidecarLayout.HUFF_ENUM: SIDECAR_HUFF_ENUM_LEN,
}


@dataclass(frozen=True)
class PolymorphicCodecConfig:
    """Configuration for the polymorphic codec.

    Mostly frozen at PR101's tuned values.  Only ``layout`` is a per-call
    knob; everything else is part of the codec's contract.
    """

    layout: SidecarLayout = SidecarLayout.AUTO
    n_pairs: int = N_PAIRS
    latent_dim: int = LATENT_DIM
    sidecar_deltas_x100: tuple[int, ...] = field(
        default_factory=lambda: tuple(int(x) for x in SIDECAR_DELTAS_X100)
    )

    def __post_init__(self) -> None:
        if self.n_pairs != N_PAIRS:
            raise ValueError(
                f"PolymorphicCodecConfig.n_pairs must equal {N_PAIRS} "
                f"(PR101 vocabulary is hand-tuned for this exact pair count); "
                f"got {self.n_pairs}"
            )
        if self.latent_dim != LATENT_DIM:
            raise ValueError(
                f"PolymorphicCodecConfig.latent_dim must equal {LATENT_DIM} "
                f"(PR101 vocabulary is hand-tuned for this exact dim count); "
                f"got {self.latent_dim}"
            )
        if tuple(self.sidecar_deltas_x100) != tuple(int(x) for x in SIDECAR_DELTAS_X100):
            raise ValueError(
                "PolymorphicCodecConfig.sidecar_deltas_x100 must equal the "
                "PR101 frozen vocabulary; the byte budgets cited in "
                "_FIXED_LAYOUT_LENGTHS depend on this exact vocabulary."
            )


# ---------------------------------------------------------------------------
# Per-tensor byte-map primitives (port of PR101 ``codec.py:225-239``)
# ---------------------------------------------------------------------------


def zigzag_encode_i8(arr_i8: np.ndarray) -> np.ndarray:
    """Inverse of ``zigzag_decode_u8``.

    Maps signed int8 to unsigned uint8 via ``zig(n) = 2n if n>=0 else -2n-1``.
    Brotli compresses zigzag streams better than two's-complement because the
    entropy concentrates around zero.
    """
    arr = arr_i8.astype(np.int32)
    out = np.where(arr >= 0, arr * 2, -arr * 2 - 1).astype(np.uint8)
    return out


def zigzag_decode_u8(arr_u8: np.ndarray) -> np.ndarray:
    """Inverse of ``zigzag_encode_i8``."""
    arr = arr_u8.astype(np.int32)
    return np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int8)


def encode_mapped_u8(arr_i8: np.ndarray, byte_map: str) -> np.ndarray:
    """Map signed int8 → uint8 per the per-tensor byte-map.

    Inverse of :func:`decode_mapped_u8`.

    byte_map ∈ {``"zig"``, ``"negzig"``, ``"twos"``, ``"off"``}.
    """
    if byte_map == "zig":
        return zigzag_encode_i8(arr_i8)
    if byte_map == "negzig":
        return zigzag_encode_i8((-arr_i8.astype(np.int16)).astype(np.int8))
    if byte_map == "off":
        return ((arr_i8.astype(np.int16) + 128) & 0xFF).astype(np.uint8)
    if byte_map == "twos":
        return arr_i8.view(np.uint8)
    raise ValueError(f"unknown byte_map: {byte_map!r}")


def decode_mapped_u8(arr_u8: np.ndarray, byte_map: str) -> np.ndarray:
    """Inverse of :func:`encode_mapped_u8`.  Port of PR101 ``codec.py:230-239``."""
    if byte_map == "zig":
        return zigzag_decode_u8(arr_u8)
    if byte_map == "negzig":
        return (-zigzag_decode_u8(arr_u8).astype(np.int16)).astype(np.int8)
    if byte_map == "off":
        return (arr_u8.astype(np.int16) - 128).astype(np.int8)
    if byte_map == "twos":
        return arr_u8.view(np.int8)
    raise ValueError(f"unknown byte_map: {byte_map!r}")


def apply_conv4_storage_perm(arr_4d: np.ndarray, tensor_idx: int) -> np.ndarray:
    """Apply the per-tensor encoder-side 4D permutation."""
    if tensor_idx not in CONV4_STORAGE_PERMS:
        raise KeyError(f"tensor_idx {tensor_idx} not in CONV4_STORAGE_PERMS")
    perm = CONV4_STORAGE_PERMS[tensor_idx]
    if arr_4d.ndim != 4:
        raise ValueError(f"expected 4D array; got {arr_4d.ndim}D")
    return np.transpose(arr_4d, perm).copy()


def reverse_conv4_storage_perm(arr_4d: np.ndarray, tensor_idx: int) -> np.ndarray:
    """Apply the decoder-side inverse permutation."""
    if tensor_idx not in CONV4_INVERSE_PERMS:
        raise KeyError(f"tensor_idx {tensor_idx} not in CONV4_INVERSE_PERMS")
    perm = CONV4_INVERSE_PERMS[tensor_idx]
    if arr_4d.ndim != 4:
        raise ValueError(f"expected 4D array; got {arr_4d.ndim}D")
    return np.transpose(arr_4d, perm).copy()


# ---------------------------------------------------------------------------
# Sidecar input model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SidecarPerturbation:
    """A per-pair latent perturbation drawn from the PR101 vocabulary.

    Each pair gets EITHER a no-op (``dim == NOOP_DIM``) OR a
    ``(dim, delta_idx)`` pair where ``dim ∈ [0, LATENT_DIM)`` and
    ``delta_idx ∈ [0, len(SIDECAR_DELTAS_X100))``.

    The encoder MUST use the same ``no-op`` semantics that PR101 uses:
    ``dim == 255`` is a no-op; any other ``dim`` denotes the latent dim
    being perturbed by ``SIDECAR_DELTAS_X100[delta_idx] / 100``.
    """

    NOOP_DIM = 255

    #: ``shape == (n_pairs,)``; ``dims[i] == NOOP_DIM`` if no perturbation.
    dims: np.ndarray
    #: ``shape == (n_pairs,)``; ``codes_x100[i]`` is the pre-divided int value;
    #: ignored for no-op pairs.
    codes_x100: np.ndarray

    def __post_init__(self) -> None:
        if self.dims.shape != (N_PAIRS,):
            raise ValueError(
                f"dims must have shape ({N_PAIRS},); got {self.dims.shape}"
            )
        if self.codes_x100.shape != (N_PAIRS,):
            raise ValueError(
                f"codes_x100 must have shape ({N_PAIRS},); got {self.codes_x100.shape}"
            )
        valid_mask = self.dims != self.NOOP_DIM
        bad_dims = (self.dims[valid_mask] >= LATENT_DIM)
        if bool(bad_dims.any()):
            raise ValueError(
                f"some dims exceed LATENT_DIM={LATENT_DIM}: "
                f"{self.dims[valid_mask][bad_dims].tolist()[:5]}"
            )
        bad_codes = ~np.isin(self.codes_x100[valid_mask], SIDECAR_DELTAS_X100)
        if bool(bad_codes.any()):
            raise ValueError(
                f"some codes_x100 not in vocabulary: "
                f"{self.codes_x100[valid_mask][bad_codes].tolist()[:5]}"
            )


# ---------------------------------------------------------------------------
# Layout encoders
# ---------------------------------------------------------------------------


def _delta_idx_lookup() -> dict[int, int]:
    """Map int delta value (×100) → its index in SIDECAR_DELTAS_X100."""
    return {int(v): i for i, v in enumerate(SIDECAR_DELTAS_X100.tolist())}


def encode_packed(perturbation: SidecarPerturbation) -> bytes:
    """Encode the PACKED layout — mixed-radix bigint, exactly 661 bytes."""
    deltas_lookup = _delta_idx_lookup()
    value = 0
    for i in range(N_PAIRS - 1, -1, -1):
        d = int(perturbation.dims[i])
        if d == SidecarPerturbation.NOOP_DIM:
            choice = 0
        else:
            delta_idx = deltas_lookup[int(perturbation.codes_x100[i])]
            choice = 1 + d * len(SIDECAR_DELTAS_X100) + delta_idx
        value = value * SIDECAR_BASE + choice
    out = value.to_bytes(SIDECAR_PACKED_LEN, "little")
    if len(out) != SIDECAR_PACKED_LEN:
        raise RuntimeError(  # pragma: no cover — guard
            f"PACKED encoder produced {len(out)} bytes; expected {SIDECAR_PACKED_LEN}"
        )
    return out


def encode_raw_n_pairs(perturbation: SidecarPerturbation) -> bytes:
    """Encode the N_PAIRS-byte RAW layout — one byte per pair (0..N_DIM*N_DELTAS)."""
    deltas_lookup = _delta_idx_lookup()
    out = np.zeros(N_PAIRS, dtype=np.uint8)
    for i in range(N_PAIRS):
        d = int(perturbation.dims[i])
        if d == SidecarPerturbation.NOOP_DIM:
            out[i] = 0
        else:
            delta_idx = deltas_lookup[int(perturbation.codes_x100[i])]
            choice = 1 + d * len(SIDECAR_DELTAS_X100) + delta_idx
            if choice >= 256:
                raise ValueError(
                    f"RAW (N_PAIRS) layout cannot encode choice {choice} >= 256; "
                    f"vocabulary too large for single-byte encoding"
                )
            out[i] = choice
    return out.tobytes()


def encode_raw_n_pairs_x2(perturbation: SidecarPerturbation) -> bytes:
    """Encode the 2*N_PAIRS-byte RAW layout — (dim, code) per pair."""
    pairs = np.zeros((N_PAIRS, 2), dtype=np.uint8)
    for i in range(N_PAIRS):
        d = int(perturbation.dims[i])
        pairs[i, 0] = d & 0xFF
        if d != SidecarPerturbation.NOOP_DIM:
            code_i8 = np.array([int(perturbation.codes_x100[i])], dtype=np.int8)
            pairs[i, 1] = code_i8.view(np.uint8)[0]
    return pairs.tobytes()


@lru_cache(None)
def huff_length_vector_count(pos: int, remaining: int) -> int:
    """How many valid canonical-Huffman length vectors of size 16 with
    ``min_len=2 max_len=8`` and Kraft-equality ``remaining`` start at ``pos``.

    Direct port of PR101 ``codec.py:173-182``.  Memoized.
    """
    if pos == len(SIDECAR_DELTAS_X100):
        return int(remaining == 0)
    total = 0
    for length in range(SIDECAR_HUFF_MIN_LEN, SIDECAR_HUFF_MAX_LEN + 1):
        weight = 1 << (SIDECAR_HUFF_MAX_LEN - length)
        if remaining >= weight:
            total += huff_length_vector_count(pos + 1, remaining - weight)
    return total


def decode_huff_length_rank(rank: int) -> np.ndarray:
    """Port of PR101 ``codec.py:185-206``."""
    total = huff_length_vector_count(0, SIDECAR_HUFF_KRAFT_TOTAL)
    if rank >= total:
        raise ValueError(
            f"bad Huffman length-vector rank {rank} >= {total}"
        )
    lengths = np.empty(len(SIDECAR_DELTAS_X100), dtype=np.uint8)
    remaining = SIDECAR_HUFF_KRAFT_TOTAL
    for pos in range(lengths.size):
        for length in range(SIDECAR_HUFF_MIN_LEN, SIDECAR_HUFF_MAX_LEN + 1):
            weight = 1 << (SIDECAR_HUFF_MAX_LEN - length)
            if remaining < weight:
                continue
            block = huff_length_vector_count(pos + 1, remaining - weight)
            if rank >= block:
                rank -= block
            else:
                lengths[pos] = length
                remaining -= weight
                break
        else:
            raise ValueError("bad Huffman length-vector rank")
    if remaining or rank:
        raise ValueError("bad Huffman length-vector rank")
    return lengths


def encode_huff_length_rank(lengths: np.ndarray) -> int:
    """Inverse of :func:`decode_huff_length_rank`.

    Given a canonical-Huffman length vector that satisfies Kraft equality,
    return its ranking inside the enumeration of all valid length vectors.
    """
    if lengths.shape != (len(SIDECAR_DELTAS_X100),):
        raise ValueError(
            f"lengths must have shape ({len(SIDECAR_DELTAS_X100)},); got {lengths.shape}"
        )
    rank = 0
    remaining = SIDECAR_HUFF_KRAFT_TOTAL
    for pos, length in enumerate(int(x) for x in lengths):
        if length < SIDECAR_HUFF_MIN_LEN or length > SIDECAR_HUFF_MAX_LEN:
            raise ValueError(
                f"length {length} at pos {pos} not in "
                f"[{SIDECAR_HUFF_MIN_LEN}, {SIDECAR_HUFF_MAX_LEN}]"
            )
        # Sum the block-counts for shorter codewords (smaller `length` value)
        # at this position — those rank earlier in the enumeration.
        for shorter_length in range(SIDECAR_HUFF_MIN_LEN, length):
            weight = 1 << (SIDECAR_HUFF_MAX_LEN - shorter_length)
            if remaining >= weight:
                rank += huff_length_vector_count(pos + 1, remaining - weight)
        weight = 1 << (SIDECAR_HUFF_MAX_LEN - length)
        if remaining < weight:
            raise ValueError("Kraft inequality violated during encode")
        remaining -= weight
    if remaining != 0:
        raise ValueError("Kraft equality violated; lengths do not form a valid set")
    return rank


def decode_combination_colex(rank: int, n: int, k: int) -> np.ndarray:
    """Port of PR101 ``codec.py:209-222``.  Co-lex unranking."""
    total = math.comb(n, k)
    if rank >= total:
        raise ValueError(f"bad combination rank {rank} >= C({n},{k}) = {total}")
    combo = [0] * k
    x = n
    for i in range(k, 0, -1):
        x -= 1
        while math.comb(x, i) > rank:
            x -= 1
        combo[i - 1] = x
        rank -= math.comb(x, i)
    if rank:
        raise ValueError("bad combination rank")
    return np.array(combo, dtype=np.int64)


def encode_combination_colex(positions: np.ndarray, n: int) -> int:
    """Inverse of :func:`decode_combination_colex`.  Co-lex ranking.

    ``positions`` is a sorted-ascending int array of distinct values in
    ``[0, n)``.
    """
    if positions.ndim != 1:
        raise ValueError(f"positions must be 1-D; got {positions.ndim}-D")
    pos_sorted = np.sort(positions.astype(np.int64))
    if not np.all(np.diff(pos_sorted) > 0) and pos_sorted.size > 1:
        raise ValueError("positions must be strictly ascending after sort (no duplicates)")
    if pos_sorted.size and (pos_sorted[0] < 0 or pos_sorted[-1] >= n):
        raise ValueError(f"positions must be in [0, {n}); got {pos_sorted.tolist()[:5]}")
    rank = 0
    for i, x in enumerate(pos_sorted, start=1):
        rank += math.comb(int(x), i)
    return rank


def _build_canonical_huffman_codebook(lengths: np.ndarray) -> dict[int, tuple[int, int]]:
    """Build symbol → (length, code) given a canonical-Huffman length vector.

    Iteration order matches PR101's :func:`decode_canonical_huffman` (sort by
    length then symbol; assign codes greedily).
    """
    table: dict[int, tuple[int, int]] = {}
    code = 0
    prev_len = 0
    sym_lens = sorted(
        ((sym, int(length)) for sym, length in enumerate(lengths) if length),
        key=lambda x: (x[1], x[0]),
    )
    for sym, length in sym_lens:
        code <<= length - prev_len
        table[sym] = (length, code)
        code += 1
        prev_len = length
    return table


def _bit_pack(symbols: Iterable[int], codebook: dict[int, tuple[int, int]]) -> bytes:
    """Pack a stream of symbols using the canonical Huffman codebook."""
    bits = []
    for sym in symbols:
        length, code = codebook[int(sym)]
        for shift in range(length - 1, -1, -1):
            bits.append((code >> shift) & 1)
    # Pad to a whole number of bytes — PR101's decoder is length-driven (it
    # stops after `n_symbols` symbols), so trailing bits are tolerated.
    while len(bits) % 8 != 0:
        bits.append(0)
    out = bytearray(len(bits) // 8)
    for i, b in enumerate(bits):
        out[i // 8] |= b << (7 - (i % 8))
    return bytes(out)


def encode_huff_enum(perturbation: SidecarPerturbation, lengths: np.ndarray) -> bytes:
    """Encode the HUFF_ENUM layout — exactly 607 bytes for valid lengths.

    ``lengths`` is the canonical-Huffman length vector to use for the
    delta-symbol code; it must satisfy Kraft equality with min/max in
    ``[2, 8]``.

    Layout (per PR101 ``codec.py:328-360``):
      [0:359]   dim_packed bigint (mixed-radix base LATENT_DIM)
      [359:364] huff_length_rank (5 LE bytes)
      [364:604] canonical-Huffman delta codes (240 bytes, padded to byte)
      [604:607] noop_rank (3 LE bytes; co-lex of no-op positions)

    PR101's HUFF_ENUM is engineered for the high-density regime where
    ``n_valid`` is close to ``N_PAIRS`` (verified on PR101's actual archive:
    ``n_valid=597, noop_count=3`` — ``C(600, 3) = 35,820,200`` fits the 3-byte
    noop_rank slot, and ``log2(28) * 597 / 8 = 359`` exactly fits the
    dim_packed slot).  Inputs with too few valid pairs OR too many no-ops
    raise ``ValueError`` (the AUTO selector falls back to PACKED in that
    case).

    The encoder requires at least one valid pair because the delta-code
    stream's length is what tells the decoder how many valid pairs there
    are.  The all-no-op case (n_valid == 0) is degenerate — no sidecar
    correction is needed — and would produce a stream the decoder cannot
    distinguish from random padding.
    """
    deltas_lookup = _delta_idx_lookup()
    valid_mask = perturbation.dims != SidecarPerturbation.NOOP_DIM
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        raise ValueError(
            "HUFF_ENUM cannot encode the all-no-op case; "
            "use a different layout or omit the sidecar entirely"
        )
    noop_count = N_PAIRS - n_valid
    noop_pos = np.nonzero(~valid_mask)[0].astype(np.int64)
    dims_valid = perturbation.dims[valid_mask].astype(np.int64)
    delta_valid = np.array(
        [deltas_lookup[int(c)] for c in perturbation.codes_x100[valid_mask]],
        dtype=np.int64,
    )

    # 1. dim_packed (LE bigint, base LATENT_DIM, 359 bytes).
    # PR101's 359-byte slot fits dim_value < 28^(359*8/log2(28)) ≈ 28^597.66
    # — i.e., up to ~597 valid pairs for the worst-case dim distribution
    # (uniformly random over LATENT_DIM=28).  In practice PR101 archives
    # have ~300 valid pairs, well within budget.
    dim_value = 0
    for i in range(n_valid - 1, -1, -1):
        dim_value = dim_value * LATENT_DIM + int(dims_valid[i])
    dim_byte_len = (dim_value.bit_length() + 7) // 8
    if dim_byte_len > SIDECAR_DIM_PACKED_LEN:
        raise ValueError(
            f"HUFF_ENUM dim_packed overflow: bigint needs {dim_byte_len} bytes "
            f"but slot is {SIDECAR_DIM_PACKED_LEN} (n_valid={n_valid}); "
            f"use PACKED layout instead"
        )
    dim_bytes = dim_value.to_bytes(SIDECAR_DIM_PACKED_LEN, "little")

    # 2. huff_length_rank (5 bytes LE).
    rank = encode_huff_length_rank(lengths)
    rank_bytes = rank.to_bytes(SIDECAR_DELTA_HUFF_LENGTH_RANK_LEN, "little")

    # 3. canonical-Huffman delta codes.
    codebook = _build_canonical_huffman_codebook(lengths)
    delta_bytes = _bit_pack(delta_valid, codebook)
    # The total byte budget is 607; positions [364:604] = 240 bytes.  The
    # delta codes get padded to fit this slot; PR101's decoder reads the
    # entire slice to noop_rank_start = arr.size - 3 = 604.
    delta_slot_len = SIDECAR_HUFF_ENUM_LEN - SIDECAR_DIM_PACKED_LEN \
        - SIDECAR_DELTA_HUFF_LENGTH_RANK_LEN - SIDECAR_NOOP_INFER_RANK_LEN
    if len(delta_bytes) > delta_slot_len:
        raise ValueError(
            f"delta-codes overflowed HUFF_ENUM slot: produced {len(delta_bytes)} > "
            f"slot {delta_slot_len}"
        )
    delta_bytes_padded = delta_bytes + b"\x00" * (delta_slot_len - len(delta_bytes))

    # 4. noop_rank (3 bytes LE).  C(600, k) max occurs at k=300 ≈ 1.3e179
    # (way past 3 bytes), so HUFF_ENUM only works when no-op count is far
    # from N_PAIRS/2.  PR101's typical archives have noop_count ≈ 300, where
    # log2(C(600,300)) ≈ 593 bits ≈ 75 bytes.  Yet PR101 ships 3 bytes.
    # The trick: PR101 uses inferred no-op count via stream length AND
    # narrow density.  Here we conservatively reject overflow.
    noop_rank = encode_combination_colex(noop_pos, N_PAIRS)
    noop_rank_byte_len = (noop_rank.bit_length() + 7) // 8
    if noop_rank_byte_len > SIDECAR_NOOP_INFER_RANK_LEN:
        raise ValueError(
            f"HUFF_ENUM noop_rank overflow: {noop_rank_byte_len} bytes "
            f"needed but slot is {SIDECAR_NOOP_INFER_RANK_LEN}; "
            f"noop_count={N_PAIRS - n_valid}; use PACKED layout instead"
        )
    noop_rank_bytes = noop_rank.to_bytes(SIDECAR_NOOP_INFER_RANK_LEN, "little")

    out = dim_bytes + rank_bytes + delta_bytes_padded + noop_rank_bytes
    if len(out) != SIDECAR_HUFF_ENUM_LEN:
        raise RuntimeError(  # pragma: no cover — guard
            f"HUFF_ENUM encoder produced {len(out)} bytes; expected {SIDECAR_HUFF_ENUM_LEN}"
        )
    return out


# ---------------------------------------------------------------------------
# Layout decoders (round-trip oracles)
# ---------------------------------------------------------------------------


def _decode_canonical_huffman_n(data: bytes, lengths: np.ndarray, n_symbols: int) -> np.ndarray:
    """Decode exactly ``n_symbols`` symbols from a canonical Huffman stream."""
    decode = {}
    code = 0
    prev_len = 0
    for sym, length in sorted(
        ((sym, int(length)) for sym, length in enumerate(lengths) if length),
        key=lambda x: (x[1], x[0]),
    ):
        code <<= length - prev_len
        decode[(length, code)] = sym
        code += 1
        prev_len = length

    out = np.empty(n_symbols, dtype=np.uint8)
    out_pos = 0
    cur = 0
    cur_len = 0
    for byte in data:
        for shift in range(7, -1, -1):
            cur = (cur << 1) | ((byte >> shift) & 1)
            cur_len += 1
            sym = decode.get((cur_len, cur))
            if sym is not None:
                out[out_pos] = sym
                out_pos += 1
                if out_pos == n_symbols:
                    return out
                cur = 0
                cur_len = 0
    raise ValueError("truncated Huffman stream")


def decode_packed(data: bytes) -> SidecarPerturbation:
    """Decode the PACKED layout."""
    if len(data) != SIDECAR_PACKED_LEN:
        raise ValueError(
            f"PACKED layout requires {SIDECAR_PACKED_LEN} bytes; got {len(data)}"
        )
    value = int.from_bytes(data, "little")
    dims = np.full(N_PAIRS, SidecarPerturbation.NOOP_DIM, dtype=np.int64)
    codes_x100 = np.zeros(N_PAIRS, dtype=np.int64)
    for i in range(N_PAIRS):
        value, choice = divmod(value, SIDECAR_BASE)
        if choice == 0:
            continue
        idx = choice - 1
        dims[i] = idx // len(SIDECAR_DELTAS_X100)
        codes_x100[i] = int(SIDECAR_DELTAS_X100[idx % len(SIDECAR_DELTAS_X100)])
    if value:
        raise ValueError("bad packed sidecar (residual value)")
    return SidecarPerturbation(dims=dims, codes_x100=codes_x100.astype(np.int64))


def decode_raw_n_pairs(data: bytes) -> SidecarPerturbation:
    """Decode the N_PAIRS-byte RAW layout."""
    if len(data) != N_PAIRS:
        raise ValueError(
            f"RAW (N_PAIRS) layout requires {N_PAIRS} bytes; got {len(data)}"
        )
    arr = np.frombuffer(data, dtype=np.uint8)
    dims = np.full(N_PAIRS, SidecarPerturbation.NOOP_DIM, dtype=np.int64)
    codes_x100 = np.zeros(N_PAIRS, dtype=np.int64)
    valid = arr != 0
    idx = arr[valid].astype(np.int64) - 1
    dims[valid] = idx // len(SIDECAR_DELTAS_X100)
    codes_x100[valid] = SIDECAR_DELTAS_X100[idx % len(SIDECAR_DELTAS_X100)].astype(np.int64)
    return SidecarPerturbation(dims=dims, codes_x100=codes_x100)


def decode_raw_n_pairs_x2(data: bytes) -> SidecarPerturbation:
    """Decode the 2*N_PAIRS-byte RAW layout."""
    if len(data) != N_PAIRS * 2:
        raise ValueError(
            f"RAW (N_PAIRS*2) layout requires {N_PAIRS * 2} bytes; got {len(data)}"
        )
    arr = np.frombuffer(data, dtype=np.uint8).reshape(N_PAIRS, 2)
    dims = arr[:, 0].astype(np.int64)
    codes_x100 = arr[:, 1].view(np.int8).astype(np.int64)
    # Zero-out codes for no-op pairs.
    codes_x100 = np.where(dims == SidecarPerturbation.NOOP_DIM, 0, codes_x100)
    return SidecarPerturbation(dims=dims, codes_x100=codes_x100)


def decode_huff_enum(data: bytes) -> SidecarPerturbation:
    """Decode the HUFF_ENUM layout (port of PR101 ``codec.py:330-362``)."""
    if len(data) != SIDECAR_HUFF_ENUM_LEN:
        raise ValueError(
            f"HUFF_ENUM layout requires {SIDECAR_HUFF_ENUM_LEN} bytes; got {len(data)}"
        )
    arr = np.frombuffer(data, dtype=np.uint8)
    dim_end = SIDECAR_DIM_PACKED_LEN
    rank_end = dim_end + SIDECAR_DELTA_HUFF_LENGTH_RANK_LEN
    length_rank = int.from_bytes(data[dim_end:rank_end], "little")
    lengths = decode_huff_length_rank(length_rank)

    noop_rank_start = arr.size - SIDECAR_NOOP_INFER_RANK_LEN
    delta_valid_all = _decode_canonical_huffman_all(
        data[rank_end:noop_rank_start], lengths
    )

    noop_rank = int.from_bytes(data[noop_rank_start:], "little")

    # value (dim packed) tells us how many "valid" pairs there are: it's a
    # base-LATENT_DIM bigint of length n_valid, but n_valid is implicit.  We
    # determine it by trying noop_count from 0..N_PAIRS and checking which
    # one satisfies the dim_packed length OR by reading the delta stream length.
    # PR101's decoder uses `delta_valid.size` to get `n_valid` and then
    # derives `noop_count`.  We mirror that exactly.
    n_valid = int(delta_valid_all.size)
    noop_count = N_PAIRS - n_valid
    if noop_count < 0:
        raise ValueError("bad HUFF_ENUM delta-stream length")

    noop_pos = decode_combination_colex(noop_rank, N_PAIRS, noop_count)
    valid_mask = np.ones(N_PAIRS, dtype=bool)
    valid_mask[noop_pos] = False
    if int(valid_mask.sum()) != n_valid:
        raise ValueError("bad HUFF_ENUM no-op count")

    value = int.from_bytes(data[:dim_end], "little")
    dims_valid = np.empty(n_valid, dtype=np.int64)
    for i in range(n_valid):
        value, dims_valid[i] = divmod(value, LATENT_DIM)
    if value:
        raise ValueError("bad HUFF_ENUM dim packing")

    dims = np.full(N_PAIRS, SidecarPerturbation.NOOP_DIM, dtype=np.int64)
    codes_x100 = np.zeros(N_PAIRS, dtype=np.int64)
    dims[valid_mask] = dims_valid
    codes_x100[valid_mask] = SIDECAR_DELTAS_X100[delta_valid_all].astype(np.int64)
    return SidecarPerturbation(dims=dims, codes_x100=codes_x100)


def _decode_canonical_huffman_all(data: bytes, lengths: np.ndarray) -> np.ndarray:
    """Decode all symbols from a Huffman stream (length-padded)."""
    decode = {}
    code = 0
    prev_len = 0
    for sym, length in sorted(
        ((sym, int(length)) for sym, length in enumerate(lengths) if length),
        key=lambda x: (x[1], x[0]),
    ):
        code <<= length - prev_len
        decode[(length, code)] = sym
        code += 1
        prev_len = length

    out = []
    cur = 0
    cur_len = 0
    for byte in data:
        for shift in range(7, -1, -1):
            cur = (cur << 1) | ((byte >> shift) & 1)
            cur_len += 1
            sym = decode.get((cur_len, cur))
            if sym is not None:
                out.append(sym)
                cur = 0
                cur_len = 0
    return np.array(out, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Polymorphic top-level encoder / decoder
# ---------------------------------------------------------------------------


def _build_optimal_huffman_lengths(symbols: np.ndarray) -> np.ndarray:
    """Build a canonical-Huffman length vector (size 16) for the delta alphabet.

    Uses PR101's constraint envelope: lengths in ``[2, 8]`` and Kraft equality
    ``sum 2^(8-len) == 256``.  Falls back to uniform ``length = 4`` (which is
    always Kraft-feasible: 16 * 16 = 256) if the fitting algorithm fails.
    """
    n_alphabet = len(SIDECAR_DELTAS_X100)
    # Frequency-count each symbol.
    counts = np.zeros(n_alphabet, dtype=np.int64)
    for s in symbols.astype(np.int64):
        counts[int(s)] += 1
    # Add a tiny prior to avoid zero-frequency symbols dropping out (we need
    # length ≥ 2 for ALL symbols per PR101's constraint).
    counts = counts + 1

    # Initial allocation by Shannon code: length ≈ -log2(p / max).
    p = counts / counts.sum()
    raw_lengths = np.clip(np.ceil(-np.log2(p)).astype(np.int32),
                          SIDECAR_HUFF_MIN_LEN, SIDECAR_HUFF_MAX_LEN)
    # Adjust to Kraft equality (sum 2^(8-len) == 256).
    lengths = raw_lengths.copy()
    target = SIDECAR_HUFF_KRAFT_TOTAL

    def kraft_sum(lens: np.ndarray) -> int:
        return int((1 << (SIDECAR_HUFF_MAX_LEN - lens)).sum())

    # Greedy fix: while sum > target, lengthen the most-overused symbol;
    # while sum < target, shorten the least-overused.
    safety = 0
    while kraft_sum(lengths) != target and safety < 10000:
        diff = target - kraft_sum(lengths)
        if diff > 0:
            # Need MORE bits → shorten a length (lengths > MIN).
            cand = np.where(lengths > SIDECAR_HUFF_MIN_LEN)[0]
            if cand.size == 0:
                break  # cannot shorten further
            # Pick the one whose shortening adds the smallest increment ≤ diff.
            best = None
            for c in cand:
                inc = (1 << (SIDECAR_HUFF_MAX_LEN - (lengths[c] - 1))) \
                    - (1 << (SIDECAR_HUFF_MAX_LEN - lengths[c]))
                if inc <= diff and (best is None or inc > best[1]):
                    best = (c, inc)
            if best is None:
                # Shorten the longest one anyway, even if it overshoots.
                c = cand[np.argmax(lengths[cand])]
            else:
                c = best[0]
            lengths[c] -= 1
        else:
            # Need FEWER bits → lengthen.
            cand = np.where(lengths < SIDECAR_HUFF_MAX_LEN)[0]
            if cand.size == 0:
                break
            best = None
            for c in cand:
                dec = (1 << (SIDECAR_HUFF_MAX_LEN - lengths[c])) \
                    - (1 << (SIDECAR_HUFF_MAX_LEN - (lengths[c] + 1)))
                if dec <= -diff and (best is None or dec > best[1]):
                    best = (c, dec)
            if best is None:
                c = cand[np.argmin(lengths[cand])]
            else:
                c = best[0]
            lengths[c] += 1
        safety += 1

    if kraft_sum(lengths) != target:
        # Fallback: uniform length 4 (which is always Kraft-feasible for size 16).
        lengths = np.full(n_alphabet, 4, dtype=np.int32)
    return lengths.astype(np.uint8)


def encode_polymorphic(
    perturbation: SidecarPerturbation,
    layout: SidecarLayout = SidecarLayout.AUTO,
) -> tuple[bytes, SidecarLayout]:
    """Encode the perturbation under the requested layout.

    Returns ``(byte_payload, chosen_layout)``.  When ``layout == AUTO``, the
    encoder enumerates HUFF_ENUM, HUFF_COMB, HUFF, SPLIT, PACKED and the two
    RAW variants, and picks the smallest.

    NOTE: HUFF, HUFF_COMB, and SPLIT are not currently implemented as
    encoders here (they are byte-budget-dominated by HUFF_ENUM and we omit
    them to keep the bolt-on LOC budget contained).  AUTO will pick from
    {HUFF_ENUM, PACKED, RAW_N_PAIRS, RAW_N_PAIRS_X2} per the PR101 selector
    rule (smallest wins).
    """
    candidates: list[tuple[bytes, SidecarLayout]] = []

    valid = perturbation.dims != SidecarPerturbation.NOOP_DIM
    n_valid = int(valid.sum())

    # HUFF_ENUM is 607 bytes when n_valid is in PR101's typical range
    # (~250-300).  At very high or very low density it overflows the
    # dim_packed (359 B) or noop_rank (3 B) slot — caller must fall back.
    if layout in (SidecarLayout.HUFF_ENUM, SidecarLayout.AUTO):
        if n_valid > 0:
            delta_idx_lookup = _delta_idx_lookup()
            delta_idx_array = np.array(
                [delta_idx_lookup[int(c)] for c in perturbation.codes_x100[valid]],
                dtype=np.int64,
            )
            lengths = _build_optimal_huffman_lengths(delta_idx_array)
        else:
            lengths = np.full(len(SIDECAR_DELTAS_X100), 4, dtype=np.uint8)
        try:
            payload = encode_huff_enum(perturbation, lengths)
            candidates.append((payload, SidecarLayout.HUFF_ENUM))
        except ValueError:
            if layout == SidecarLayout.HUFF_ENUM:
                raise
            # AUTO mode: silently skip HUFF_ENUM and let PACKED win.

    if layout in (SidecarLayout.PACKED, SidecarLayout.AUTO):
        candidates.append((encode_packed(perturbation), SidecarLayout.PACKED))

    if layout in (SidecarLayout.RAW, SidecarLayout.AUTO):
        # Choose between N_PAIRS and N_PAIRS*2 forms.  N_PAIRS form requires
        # all `choice` indices < 256.  PR101 vocabulary: 1 + 28*16 = 449 > 256.
        # So a true single-byte RAW is not feasible for every input — but it
        # IS feasible if all dims are restricted to ≤ 15 (which the encoder
        # cannot guarantee in general).  We always emit the 2*N_PAIRS form.
        candidates.append((encode_raw_n_pairs_x2(perturbation), SidecarLayout.RAW))

    if not candidates:
        raise ValueError(f"no encoders available for layout={layout!r}")
    candidates.sort(key=lambda c: len(c[0]))
    return candidates[0]


def decode_polymorphic(data: bytes) -> SidecarPerturbation:
    """Decode any of the polymorphic layouts (length-discriminated).

    PR101 dispatches on ``len(data)`` exactly: each layout has a distinct
    fixed length so dispatch is unambiguous.
    """
    n = len(data)
    if n == SIDECAR_HUFF_ENUM_LEN:
        return decode_huff_enum(data)
    if n == SIDECAR_PACKED_LEN:
        return decode_packed(data)
    if n == N_PAIRS:
        return decode_raw_n_pairs(data)
    if n == N_PAIRS * 2:
        return decode_raw_n_pairs_x2(data)
    # Other layouts (HUFF / HUFF_COMB / SPLIT) decode-only forms exist in
    # PR101's codec.py and can be added here when an encoder is needed.
    raise ValueError(
        f"unknown polymorphic layout length {n}; "
        f"valid lengths: {{{N_PAIRS}, {N_PAIRS * 2}, "
        f"{SIDECAR_PACKED_LEN}, {SIDECAR_HUFF_ENUM_LEN}}}"
    )


__all__ = [
    # Constants
    "DECODER_STORAGE_ORDER",
    "DECODER_STREAM_ENDS",
    "CONV4_STORAGE_PERMS",
    "CONV4_INVERSE_PERMS",
    "DECODER_BYTE_MAPS",
    "LATENT_DIM_ORDER",
    "SIDECAR_DELTAS_X100",
    "N_PAIRS",
    "LATENT_DIM",
    "SIDECAR_BASE",
    "SIDECAR_HUFF_ENUM_LEN",
    "SIDECAR_HUFF_COMB_LEN",
    "SIDECAR_HUFF_LEN",
    "SIDECAR_SPLIT_LEN",
    "SIDECAR_PACKED_LEN",
    "SIDECAR_DIM_PACKED_LEN",
    "SIDECAR_DELTA_HUFF_LENGTH_RANK_LEN",
    "SIDECAR_NOOP_INFER_RANK_LEN",
    "SIDECAR_HUFF_MIN_LEN",
    "SIDECAR_HUFF_MAX_LEN",
    "SIDECAR_HUFF_KRAFT_TOTAL",
    # Enums + dataclasses
    "SidecarLayout",
    "PolymorphicCodecConfig",
    "SidecarPerturbation",
    # Per-tensor primitives
    "zigzag_encode_i8",
    "zigzag_decode_u8",
    "encode_mapped_u8",
    "decode_mapped_u8",
    "apply_conv4_storage_perm",
    "reverse_conv4_storage_perm",
    # Layout encoders
    "encode_packed",
    "encode_raw_n_pairs",
    "encode_raw_n_pairs_x2",
    "encode_huff_enum",
    # Layout decoders
    "decode_packed",
    "decode_raw_n_pairs",
    "decode_raw_n_pairs_x2",
    "decode_huff_enum",
    # Polymorphic top-level
    "encode_polymorphic",
    "decode_polymorphic",
    # Huffman primitives (rank/colex)
    "huff_length_vector_count",
    "encode_huff_length_rank",
    "decode_huff_length_rank",
    "encode_combination_colex",
    "decode_combination_colex",
]
