# SPDX-License-Identifier: MIT
"""PR101 FEC7 selector entropy prototypes.

This module is a no-score packet-analysis surface for the PR101 FEC6
frame-selector archive.  It builds byte-closed alternative selector payloads
that decode to the same 600 selector codes as FEC6, then reports whether the
charged selector bytes can plausibly save a target number of bytes.

The prototypes intentionally stop at selector payload bytes.  They do not
alter inflate runtimes, dispatch jobs, or assert score movement.
"""

from __future__ import annotations

import math
import struct
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.packet_compiler.pr84_adaptive_mask import (
    AdaptiveContextSpec,
    decode_adaptive_context_stream,
    encode_adaptive_context_stream,
)
from tac.packet_compiler.pr103_arithmetic_coding import (
    MergedRangeStream,
    WeightTensorACSpec,
    decode_merged_range_stream,
    encode_merged_range_stream,
)
from tac.repo_io import sha256_bytes

FEC7_MAGIC = b"FEC7"
FEC7_VARIANT_GLOBAL_RANGE = 1
FEC7_VARIANT_SPLIT_NONE_RANGE = 2
FEC7_VARIANT_PAIRMOD_CONTEXT_RANGE = 3

DEFAULT_ALPHABET_SIZE = 16
DEFAULT_PAIRMOD_CONTEXTS = (2, 4, 8, 16, 25, 50, 100)


class PR101FEC7SelectorError(ValueError):
    """Raised when a FEC7 selector prototype is malformed."""


@dataclass(frozen=True)
class FEC7Candidate:
    """One byte-closed selector encoding candidate."""

    name: str
    payload: bytes
    decoded_codes: tuple[int, ...]
    description: str
    charged_model_bytes: int
    range_stream_bytes: int
    metadata_bytes: int
    zero_model_entropy_bytes: int | None = None
    notes: tuple[str, ...] = ()

    @property
    def payload_bytes(self) -> int:
        return len(self.payload)


def _validate_codes(
    codes: Sequence[int], *, alphabet_size: int = DEFAULT_ALPHABET_SIZE
) -> list[int]:
    out = [int(code) for code in codes]
    if not out:
        raise PR101FEC7SelectorError("selector code stream is empty")
    lo = min(out)
    hi = max(out)
    if lo < 0 or hi >= alphabet_size:
        raise PR101FEC7SelectorError(
            f"selector codes out of range [0, {alphabet_size}); min={lo} max={hi}"
        )
    if len(out) > 0xFFFF:
        raise PR101FEC7SelectorError(f"too many selector codes for u16 header: {len(out)}")
    return out


def _counts_u8(codes: Sequence[int], *, alphabet_size: int) -> np.ndarray:
    counts = np.bincount(np.asarray(codes, dtype=np.int64), minlength=alphabet_size)
    if int(counts.max(initial=0)) > 255:
        raise PR101FEC7SelectorError("u8 count table overflow; use a split model")
    return counts.astype(np.uint8)


def _counts_u16(codes: Sequence[int], *, alphabet_size: int) -> np.ndarray:
    counts = np.bincount(np.asarray(codes, dtype=np.int64), minlength=alphabet_size)
    if int(counts.max(initial=0)) > 0xFFFF:
        raise PR101FEC7SelectorError("u16 count table overflow")
    return counts.astype("<u2")


def _shannon_bits_from_counts(counts: Mapping[int, int] | Counter[int]) -> float:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return 0.0
    bits = 0.0
    for value in counts.values():
        count = int(value)
        if count <= 0:
            continue
        p = count / total
        bits -= count * math.log2(p)
    return bits


def ceil_bytes(bits: float) -> int:
    """Return ``ceil(bits / 8)`` for entropy accounting."""

    return math.ceil(float(bits) / 8.0)


def empirical_entropy_floor_bytes(codes: Sequence[int]) -> int:
    """Zero-overhead empirical entropy floor for one global selector model."""

    return ceil_bytes(_shannon_bits_from_counts(Counter(int(code) for code in codes)))


def pairmod_entropy_floor_bytes(
    codes: Sequence[int], *, context_mod: int
) -> int:
    """Zero-model lower bound for a deterministic pair-index modulo context."""

    if context_mod <= 0:
        raise PR101FEC7SelectorError("context_mod must be positive")
    buckets: dict[int, Counter[int]] = defaultdict(Counter)
    for index, code in enumerate(codes):
        buckets[index % context_mod][int(code)] += 1
    return ceil_bytes(sum(_shannon_bits_from_counts(counter) for counter in buckets.values()))


def _range_specs_for_global(
    codes: Sequence[int], *, alphabet_size: int
) -> tuple[np.ndarray, list[WeightTensorACSpec]]:
    arr = np.asarray(codes, dtype=np.int32)
    hist = _counts_u8(codes, alphabet_size=alphabet_size).astype(np.float64)
    spec = WeightTensorACSpec(
        name="pr101_fec7_selector_codes",
        shape=(len(arr),),
        histogram=hist,
        alphabet_size=alphabet_size,
    )
    return arr, [spec]


def encode_fec7_global_range(
    codes: Sequence[int], *, alphabet_size: int = DEFAULT_ALPHABET_SIZE
) -> bytes:
    """Encode all selector codes with one PR103-style range stream.

    Wire format:
    ``FEC7 | variant:u8 | n_pairs:u16 | alphabet:u8 | u8 histogram | range``.
    The histogram is charged archive data, so this is byte-closed.
    """

    codes = _validate_codes(codes, alphabet_size=alphabet_size)
    if alphabet_size > 255:
        raise PR101FEC7SelectorError("alphabet_size must fit in u8")
    arr, specs = _range_specs_for_global(codes, alphabet_size=alphabet_size)
    stream = encode_merged_range_stream([arr], specs)
    hist = _counts_u8(codes, alphabet_size=alphabet_size)
    return (
        FEC7_MAGIC
        + struct.pack("<BHB", FEC7_VARIANT_GLOBAL_RANGE, len(codes), alphabet_size)
        + hist.tobytes()
        + stream.payload
    )


def decode_fec7_global_range(payload: bytes) -> list[int]:
    if len(payload) < 8:
        raise PR101FEC7SelectorError("FEC7 global payload truncated")
    if payload[:4] != FEC7_MAGIC:
        raise PR101FEC7SelectorError(f"FEC7 magic mismatch: {payload[:4]!r}")
    variant, n_pairs, alphabet_size = struct.unpack_from("<BHB", payload, 4)
    if variant != FEC7_VARIANT_GLOBAL_RANGE:
        raise PR101FEC7SelectorError(f"not a global-range FEC7 payload: {variant}")
    model_start = 8
    model_end = model_start + alphabet_size
    if model_end > len(payload):
        raise PR101FEC7SelectorError("FEC7 global histogram truncated")
    hist = np.frombuffer(payload[model_start:model_end], dtype=np.uint8).astype(np.float64)
    stream_payload = payload[model_end:]
    spec = WeightTensorACSpec(
        name="pr101_fec7_selector_codes",
        shape=(n_pairs,),
        histogram=hist,
        alphabet_size=int(alphabet_size),
    )
    stream = MergedRangeStream(
        payload=stream_payload,
        tensor_symbol_counts=(int(n_pairs),),
        word_count=len(stream_payload) // 4,
    )
    return [int(x) for x in decode_merged_range_stream(stream, [spec])[0].tolist()]


def encode_fec7_split_none_range(
    codes: Sequence[int], *, alphabet_size: int = DEFAULT_ALPHABET_SIZE, none_code: int = 0
) -> bytes:
    """Encode a none/non-none mask plus non-none symbols in one range stream."""

    codes = _validate_codes(codes, alphabet_size=alphabet_size)
    if not (0 <= none_code < alphabet_size):
        raise PR101FEC7SelectorError("none_code outside alphabet")
    mask = np.asarray([0 if code == none_code else 1 for code in codes], dtype=np.int32)
    nonzero = np.asarray(
        [code - 1 if code > none_code else code for code in codes if code != none_code],
        dtype=np.int32,
    )
    mask_hist = _counts_u16(mask.tolist(), alphabet_size=2).astype(np.float64)
    nonzero_alphabet = alphabet_size - 1
    nz_hist_u8 = _counts_u8(nonzero.tolist(), alphabet_size=nonzero_alphabet)
    specs = [
        WeightTensorACSpec(
            name="pr101_fec7_none_mask",
            shape=(len(mask),),
            histogram=mask_hist,
            alphabet_size=2,
        ),
        WeightTensorACSpec(
            name="pr101_fec7_non_none_symbols",
            shape=(len(nonzero),),
            histogram=nz_hist_u8.astype(np.float64),
            alphabet_size=nonzero_alphabet,
        ),
    ]
    stream = encode_merged_range_stream([mask, nonzero], specs)
    return (
        FEC7_MAGIC
        + struct.pack(
            "<BHHBB",
            FEC7_VARIANT_SPLIT_NONE_RANGE,
            len(codes),
            len(nonzero),
            alphabet_size,
            none_code,
        )
        + _counts_u16(mask.tolist(), alphabet_size=2).tobytes()
        + nz_hist_u8.tobytes()
        + stream.payload
    )


def decode_fec7_split_none_range(payload: bytes) -> list[int]:
    if len(payload) < 11:
        raise PR101FEC7SelectorError("FEC7 split payload truncated")
    if payload[:4] != FEC7_MAGIC:
        raise PR101FEC7SelectorError(f"FEC7 magic mismatch: {payload[:4]!r}")
    variant, n_pairs, nonzero_count, alphabet_size, none_code = struct.unpack_from(
        "<BHHBB", payload, 4
    )
    if variant != FEC7_VARIANT_SPLIT_NONE_RANGE:
        raise PR101FEC7SelectorError(f"not a split-none FEC7 payload: {variant}")
    nonzero_alphabet = int(alphabet_size) - 1
    model_start = 11
    mask_model_end = model_start + 4
    nz_model_end = mask_model_end + nonzero_alphabet
    if nz_model_end > len(payload):
        raise PR101FEC7SelectorError("FEC7 split model truncated")
    mask_hist = np.frombuffer(payload[model_start:mask_model_end], dtype="<u2").astype(
        np.float64
    )
    nz_hist = np.frombuffer(payload[mask_model_end:nz_model_end], dtype=np.uint8).astype(
        np.float64
    )
    stream_payload = payload[nz_model_end:]
    specs = [
        WeightTensorACSpec(
            name="pr101_fec7_none_mask",
            shape=(int(n_pairs),),
            histogram=mask_hist,
            alphabet_size=2,
        ),
        WeightTensorACSpec(
            name="pr101_fec7_non_none_symbols",
            shape=(int(nonzero_count),),
            histogram=nz_hist,
            alphabet_size=nonzero_alphabet,
        ),
    ]
    stream = MergedRangeStream(
        payload=stream_payload,
        tensor_symbol_counts=(int(n_pairs), int(nonzero_count)),
        word_count=len(stream_payload) // 4,
    )
    mask, nonzero = decode_merged_range_stream(stream, specs)
    nz_iter = iter(int(x) for x in nonzero.tolist())
    out: list[int] = []
    for bit in mask.tolist():
        if int(bit) == 0:
            out.append(int(none_code))
        else:
            sym = next(nz_iter)
            out.append(sym + 1 if sym >= none_code else sym)
    return out


def _pairmod_context_ids(n_pairs: int, context_mod: int) -> np.ndarray:
    return (np.arange(n_pairs, dtype=np.int64) % int(context_mod)).astype(np.int64)


def _pairmod_cdf_table(
    codes: Sequence[int], *, context_mod: int, alphabet_size: int
) -> np.ndarray:
    table = np.zeros((context_mod, alphabet_size), dtype=np.uint8)
    for index, code in enumerate(codes):
        ctx = index % context_mod
        current = int(table[ctx, int(code)])
        if current < 255:
            table[ctx, int(code)] = current + 1
    return table


def encode_fec7_pairmod_context_range(
    codes: Sequence[int],
    *,
    context_mod: int,
    alphabet_size: int = DEFAULT_ALPHABET_SIZE,
) -> bytes:
    """Encode selector codes with a charged pair-index-modulo context table.

    This reuses the PR84 adaptive-context primitive with deterministic context
    ids ``pair_index % context_mod``.  The full context table is charged in the
    selector payload, so this is a compliance-safe upper bound for this family.
    """

    codes = _validate_codes(codes, alphabet_size=alphabet_size)
    if not (1 <= context_mod <= 255):
        raise PR101FEC7SelectorError("context_mod must fit in u8")
    if alphabet_size > 255:
        raise PR101FEC7SelectorError("alphabet_size must fit in u8")
    table = _pairmod_cdf_table(codes, context_mod=context_mod, alphabet_size=alphabet_size)
    context_ids = _pairmod_context_ids(len(codes), context_mod)
    payload = encode_adaptive_context_stream(
        np.asarray(codes, dtype=np.int64),
        context_ids,
        AdaptiveContextSpec(alphabet_size=alphabet_size, cdf_table=table.astype(np.float64)),
    )
    return (
        FEC7_MAGIC
        + struct.pack(
            "<BHBB",
            FEC7_VARIANT_PAIRMOD_CONTEXT_RANGE,
            len(codes),
            alphabet_size,
            context_mod,
        )
        + table.tobytes()
        + payload
    )


def decode_fec7_pairmod_context_range(payload: bytes) -> list[int]:
    if len(payload) < 9:
        raise PR101FEC7SelectorError("FEC7 pairmod payload truncated")
    if payload[:4] != FEC7_MAGIC:
        raise PR101FEC7SelectorError(f"FEC7 magic mismatch: {payload[:4]!r}")
    variant, n_pairs, alphabet_size, context_mod = struct.unpack_from("<BHBB", payload, 4)
    if variant != FEC7_VARIANT_PAIRMOD_CONTEXT_RANGE:
        raise PR101FEC7SelectorError(f"not a pairmod-context FEC7 payload: {variant}")
    model_start = 9
    model_end = model_start + int(alphabet_size) * int(context_mod)
    if model_end > len(payload):
        raise PR101FEC7SelectorError("FEC7 pairmod context table truncated")
    table = np.frombuffer(payload[model_start:model_end], dtype=np.uint8).reshape(
        int(context_mod), int(alphabet_size)
    )
    context_ids = _pairmod_context_ids(int(n_pairs), int(context_mod))
    out = decode_adaptive_context_stream(
        payload[model_end:],
        context_ids,
        AdaptiveContextSpec(
            alphabet_size=int(alphabet_size),
            cdf_table=table.astype(np.float64),
        ),
    )
    return [int(x) for x in out.tolist()]


def decode_fec7_selector_payload(payload: bytes) -> list[int]:
    """Decode any FEC7 prototype payload emitted by this module."""

    if len(payload) < 5 or payload[:4] != FEC7_MAGIC:
        raise PR101FEC7SelectorError("FEC7 selector payload magic mismatch")
    variant = int(payload[4])
    if variant == FEC7_VARIANT_GLOBAL_RANGE:
        return decode_fec7_global_range(payload)
    if variant == FEC7_VARIANT_SPLIT_NONE_RANGE:
        return decode_fec7_split_none_range(payload)
    if variant == FEC7_VARIANT_PAIRMOD_CONTEXT_RANGE:
        return decode_fec7_pairmod_context_range(payload)
    raise PR101FEC7SelectorError(f"unknown FEC7 selector variant {variant}")


def _candidate_from_payload(
    *,
    name: str,
    payload: bytes,
    source_codes: Sequence[int],
    decoded_codes: Sequence[int],
    description: str,
    charged_model_bytes: int,
    metadata_bytes: int,
    zero_model_entropy_bytes: int | None = None,
    notes: Sequence[str] = (),
) -> FEC7Candidate:
    decoded = tuple(int(code) for code in decoded_codes)
    expected = tuple(int(code) for code in source_codes)
    if decoded != expected:
        raise PR101FEC7SelectorError(f"{name} did not roundtrip selector codes")
    return FEC7Candidate(
        name=name,
        payload=payload,
        decoded_codes=decoded,
        description=description,
        charged_model_bytes=int(charged_model_bytes),
        range_stream_bytes=max(0, len(payload) - int(metadata_bytes) - int(charged_model_bytes)),
        metadata_bytes=int(metadata_bytes),
        zero_model_entropy_bytes=zero_model_entropy_bytes,
        notes=tuple(str(note) for note in notes),
    )


def build_fec7_candidates(
    codes: Sequence[int],
    *,
    alphabet_size: int = DEFAULT_ALPHABET_SIZE,
    pairmod_contexts: Sequence[int] = DEFAULT_PAIRMOD_CONTEXTS,
) -> list[FEC7Candidate]:
    """Return byte-closed FEC7 candidate payloads that roundtrip ``codes``."""

    codes = _validate_codes(codes, alphabet_size=alphabet_size)
    candidates: list[FEC7Candidate] = []

    global_payload = encode_fec7_global_range(codes, alphabet_size=alphabet_size)
    candidates.append(
        _candidate_from_payload(
            name="fec7_global_pr103_range_u8_hist",
            payload=global_payload,
            source_codes=codes,
            decoded_codes=decode_fec7_global_range(global_payload),
            description="one PR103 merged range stream over 16 selector symbols",
            charged_model_bytes=alphabet_size,
            metadata_bytes=8,
            zero_model_entropy_bytes=empirical_entropy_floor_bytes(codes),
        )
    )

    split_payload = encode_fec7_split_none_range(codes, alphabet_size=alphabet_size)
    candidates.append(
        _candidate_from_payload(
            name="fec7_split_none_pr103_range",
            payload=split_payload,
            source_codes=codes,
            decoded_codes=decode_fec7_split_none_range(split_payload),
            description="PR103 merged range stream over none mask plus non-none symbols",
            charged_model_bytes=4 + (alphabet_size - 1),
            metadata_bytes=12,
            zero_model_entropy_bytes=empirical_entropy_floor_bytes(codes),
        )
    )

    for context_mod in pairmod_contexts:
        payload = encode_fec7_pairmod_context_range(
            codes,
            context_mod=int(context_mod),
            alphabet_size=alphabet_size,
        )
        candidates.append(
            _candidate_from_payload(
                name=f"fec7_pairmod{int(context_mod)}_pr84_context_range",
                payload=payload,
                source_codes=codes,
                decoded_codes=decode_fec7_pairmod_context_range(payload),
                description=(
                    "PR84 adaptive-context range stream with charged "
                    f"pair_index_mod_{int(context_mod)} cdf table"
                ),
                charged_model_bytes=int(context_mod) * alphabet_size,
                metadata_bytes=9,
                zero_model_entropy_bytes=pairmod_entropy_floor_bytes(
                    codes, context_mod=int(context_mod)
                ),
                notes=(
                    "context model is charged; hardcoding it in runtime would be "
                    "source-embedded selector payload, not a byte-closed archive",
                ),
            )
        )

    return candidates


def candidate_record(
    candidate: FEC7Candidate, *, fec6_selector_payload_bytes: int, target_saving_bytes: int
) -> dict[str, Any]:
    saving = int(fec6_selector_payload_bytes) - candidate.payload_bytes
    return {
        "name": candidate.name,
        "payload_bytes": candidate.payload_bytes,
        "payload_sha256": sha256_bytes(candidate.payload),
        "saving_vs_fec6_selector_bytes": saving,
        "meets_target_saving": saving >= int(target_saving_bytes),
        "target_saving_bytes": int(target_saving_bytes),
        "charged_model_bytes": candidate.charged_model_bytes,
        "range_stream_bytes": candidate.range_stream_bytes,
        "metadata_bytes": candidate.metadata_bytes,
        "zero_model_entropy_bytes": candidate.zero_model_entropy_bytes,
        "description": candidate.description,
        "notes": list(candidate.notes),
    }


def profile_selector_encodings(
    codes: Sequence[int],
    *,
    fec6_selector_payload_bytes: int,
    target_saving_bytes: int = 79,
    alphabet_size: int = DEFAULT_ALPHABET_SIZE,
    pairmod_contexts: Sequence[int] = DEFAULT_PAIRMOD_CONTEXTS,
) -> dict[str, Any]:
    """Profile charged FEC7 selector candidates against the FEC6 byte target."""

    codes = _validate_codes(codes, alphabet_size=alphabet_size)
    candidates = build_fec7_candidates(
        codes,
        alphabet_size=alphabet_size,
        pairmod_contexts=pairmod_contexts,
    )
    records = [
        candidate_record(
            candidate,
            fec6_selector_payload_bytes=fec6_selector_payload_bytes,
            target_saving_bytes=target_saving_bytes,
        )
        for candidate in candidates
    ]
    records.sort(key=lambda row: (int(row["payload_bytes"]), str(row["name"])))
    best = records[0]
    global_floor = empirical_entropy_floor_bytes(codes)
    theoretical_pairmod = [
        {
            "context_mod": int(context_mod),
            "zero_model_entropy_bytes": pairmod_entropy_floor_bytes(
                codes, context_mod=int(context_mod)
            ),
            "saving_vs_fec6_if_model_free": int(fec6_selector_payload_bytes)
            - pairmod_entropy_floor_bytes(codes, context_mod=int(context_mod)),
            "byte_closed": False,
        }
        for context_mod in pairmod_contexts
    ]
    theoretical_pairmod.sort(key=lambda row: int(row["zero_model_entropy_bytes"]))
    blocker = (
        best["saving_vs_fec6_selector_bytes"] < int(target_saving_bytes)
        and (
            int(fec6_selector_payload_bytes) - global_floor < int(target_saving_bytes)
        )
    )
    return {
        "schema": "pr101_fec7_selector_entropy_profile.v1",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "selector_code_count": len(codes),
        "alphabet_size": int(alphabet_size),
        "fec6_selector_payload_bytes": int(fec6_selector_payload_bytes),
        "target_saving_bytes": int(target_saving_bytes),
        "selector_histogram": {
            str(code): int(count)
            for code, count in sorted(Counter(codes).items(), key=lambda item: int(item[0]))
        },
        "global_entropy_floor_bytes": global_floor,
        "global_entropy_floor_saving_vs_fec6_bytes": int(fec6_selector_payload_bytes)
        - global_floor,
        "charged_candidates": records,
        "best_charged_candidate": best,
        "theoretical_context_lower_bounds": theoretical_pairmod,
        "can_meet_target_with_charged_fec7_prototype": bool(
            best["saving_vs_fec6_selector_bytes"] >= int(target_saving_bytes)
        ),
        "explicit_blocker": {
            "blocked": bool(blocker),
            "reason": (
                "FEC6 selector bytes are already near the global entropy floor; "
                "tested byte-closed FEC7 range/adaptive prototypes charge their "
                "model bytes and do not approach the required saving."
            ),
            "reactivation_criteria": (
                "Reopen only with a selector model whose charged model plus range "
                "stream is at least target_saving_bytes smaller than FEC6, or with "
                "a compliance-reviewed runtime prior that is not source-embedded "
                "selector data."
            ),
        },
    }


__all__ = [
    "DEFAULT_ALPHABET_SIZE",
    "DEFAULT_PAIRMOD_CONTEXTS",
    "FEC7_MAGIC",
    "PR101FEC7SelectorError",
    "build_fec7_candidates",
    "candidate_record",
    "ceil_bytes",
    "decode_fec7_global_range",
    "decode_fec7_pairmod_context_range",
    "decode_fec7_selector_payload",
    "decode_fec7_split_none_range",
    "empirical_entropy_floor_bytes",
    "encode_fec7_global_range",
    "encode_fec7_pairmod_context_range",
    "encode_fec7_split_none_range",
    "pairmod_entropy_floor_bytes",
    "profile_selector_encodings",
]
