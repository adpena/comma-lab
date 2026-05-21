# SPDX-License-Identifier: MIT
"""ATW2 CDF-table dead-section probe helpers.

These helpers are research-only. They test the current ATW2 runtime fact that
``cdf_table_blob`` is parsed and copied into the model but is not consumed by
``reconstruct_from_wz_residual``. Any future range-decoder implementation that
uses the table should make these probes fail, forcing the lane to move from
byte-only cleanup to residual-correction or co-trained replacement accounting.
"""

from __future__ import annotations

import hashlib
import struct
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from tac.substrates.atw_codec_v2.archive import (
    ATW2_CDF_DEAD_SECTION_SENTINEL,
    ATW2_HEADER_FMT,
    ATW2_HEADER_SIZE,
    parse_atw2_archive_bytes,
)
from tac.substrates.atw_codec_v2.inflate import inflate_one_video

CANONICAL_RATE_DENOM_BYTES = 37_545_489
CANONICAL_RATE_MULTIPLIER = 25.0
DEFAULT_PROCEDURAL_ENVELOPE_BYTES = 32
CDF_SECTION_NAME = "cdf_table_blob"


@dataclass(frozen=True)
class Atw2CdfSectionAnalysis:
    """Static ATW2 cdf_table_blob section analysis."""

    cdf_offset: int
    cdf_bytes: int
    cdf_classes: int
    cdf_symbols: int
    dtype: str
    parser_visible: bool
    current_runtime_decode_visible: bool
    conservative_envelope_bytes: int
    conservative_bytes_saved: int
    conservative_delta_s_rate_only: float
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Atw2CdfDecodeInfluenceProof:
    """Inflate-output comparison after mutating only cdf_table_blob bytes."""

    analysis: Atw2CdfSectionAnalysis
    mutation_kind: str
    mutated_byte_count: int
    source_archive_sha256: str
    mutated_archive_sha256: str
    source_raw_sha256: str
    mutated_raw_sha256: str
    raw_equal: bool
    raw_byte_count: int
    max_abs_raw_byte_delta: int
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["analysis"] = self.analysis.to_dict()
        return payload


@dataclass(frozen=True)
class Atw2CdfCompactionProof:
    """Inflate-output comparison after replacing cdf_table_blob with sentinel."""

    analysis: Atw2CdfSectionAnalysis
    compact_cdf_bytes: int
    source_archive_sha256: str
    compact_archive_sha256: str
    source_archive_bytes: int
    compact_archive_bytes: int
    archive_bytes_saved: int
    delta_s_rate_only: float
    source_raw_sha256: str
    compact_raw_sha256: str
    raw_equal: bool
    raw_byte_count: int
    max_abs_raw_byte_delta: int
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["analysis"] = self.analysis.to_dict()
        return payload


def analyze_atw2_cdf_section(
    archive_bytes: bytes,
    *,
    envelope_bytes: int = DEFAULT_PROCEDURAL_ENVELOPE_BYTES,
) -> Atw2CdfSectionAnalysis:
    """Return the cdf_table_blob section geometry and rate-only estimate.

    The rate estimate is conservative envelope accounting. A pure schema shrink
    should recompute the byte delta from the actual candidate archive length.
    """

    sections = parse_atw2_archive_bytes(archive_bytes)
    cdf_offset, cdf_len = sections[CDF_SECTION_NAME]
    header = struct.unpack(ATW2_HEADER_FMT, archive_bytes[:ATW2_HEADER_SIZE])
    cdf_classes = int(header[6])
    cdf_symbols = int(header[7])
    expected_cdf_bytes = cdf_classes * cdf_symbols * 2
    compact_cdf_bytes = len(ATW2_CDF_DEAD_SECTION_SENTINEL)
    if cdf_len not in (expected_cdf_bytes, compact_cdf_bytes):
        raise ValueError(
            f"cdf_table_blob length {cdf_len} != cdf_classes*cdf_symbols*2 "
            f"({expected_cdf_bytes}) or compact sentinel bytes "
            f"({compact_cdf_bytes})"
        )
    bytes_saved = int(expected_cdf_bytes) - int(envelope_bytes)
    delta_s = -CANONICAL_RATE_MULTIPLIER * bytes_saved / CANONICAL_RATE_DENOM_BYTES
    return Atw2CdfSectionAnalysis(
        cdf_offset=int(cdf_offset),
        cdf_bytes=int(cdf_len),
        cdf_classes=cdf_classes,
        cdf_symbols=cdf_symbols,
        dtype="fp16",
        parser_visible=True,
        current_runtime_decode_visible=False,
        conservative_envelope_bytes=int(envelope_bytes),
        conservative_bytes_saved=bytes_saved,
        conservative_delta_s_rate_only=float(delta_s),
    )


def compose_atw2_archive_without_cdf_table(archive_bytes: bytes) -> bytes:
    """Replace the full cdf_table_blob with the compact dead-section sentinel."""

    analysis = analyze_atw2_cdf_section(archive_bytes)
    if analysis.cdf_bytes <= len(ATW2_CDF_DEAD_SECTION_SENTINEL):
        raise ValueError("archive does not contain a full cdf_table_blob to compact")
    header_values = list(struct.unpack(ATW2_HEADER_FMT, archive_bytes[:ATW2_HEADER_SIZE]))
    header_values[14] = len(ATW2_CDF_DEAD_SECTION_SENTINEL)
    compact_header = struct.pack(ATW2_HEADER_FMT, *header_values)
    cdf_start = analysis.cdf_offset
    cdf_end = cdf_start + analysis.cdf_bytes
    return (
        compact_header
        + archive_bytes[ATW2_HEADER_SIZE:cdf_start]
        + ATW2_CDF_DEAD_SECTION_SENTINEL
        + archive_bytes[cdf_end:]
    )


def mutate_atw2_cdf_table_bytes(
    archive_bytes: bytes,
    *,
    mutation_kind: Literal["xor_ff", "zero"] = "xor_ff",
) -> bytes:
    """Return archive bytes with only cdf_table_blob mutated."""

    analysis = analyze_atw2_cdf_section(archive_bytes)
    out = bytearray(archive_bytes)
    start = analysis.cdf_offset
    end = start + analysis.cdf_bytes
    if mutation_kind == "xor_ff":
        for i in range(start, end):
            out[i] ^= 0xFF
    elif mutation_kind == "zero":
        out[start:end] = b"\x00" * analysis.cdf_bytes
    else:
        raise ValueError(f"unsupported mutation_kind: {mutation_kind!r}")
    return bytes(out)


def _max_abs_byte_delta(a_bytes: bytes, b_bytes: bytes) -> int:
    limit = min(len(a_bytes), len(b_bytes))
    max_delta = 0
    if limit:
        max_delta = max(
            abs(int(a) - int(b)) for a, b in zip(a_bytes[:limit], b_bytes[:limit])
        )
    if len(a_bytes) != len(b_bytes):
        max_delta = max(max_delta, 255)
    return int(max_delta)


def prove_atw2_cdf_decode_influence(
    archive_bytes: bytes,
    *,
    mutation_kind: Literal["xor_ff", "zero"] = "xor_ff",
    device: str = "cpu",
) -> Atw2CdfDecodeInfluenceProof:
    """Inflate source and CDF-mutated archives, then compare raw outputs."""

    analysis = analyze_atw2_cdf_section(archive_bytes)
    mutated = mutate_atw2_cdf_table_bytes(archive_bytes, mutation_kind=mutation_kind)
    with tempfile.TemporaryDirectory(prefix="atw2-cdf-proof-") as tmp:
        root = Path(tmp)
        source_raw = root / "source.raw"
        mutated_raw = root / "mutated.raw"
        inflate_one_video(archive_bytes, source_raw, device=device)
        inflate_one_video(mutated, mutated_raw, device=device)
        source_raw_bytes = source_raw.read_bytes()
        mutated_raw_bytes = mutated_raw.read_bytes()

    return Atw2CdfDecodeInfluenceProof(
        analysis=analysis,
        mutation_kind=mutation_kind,
        mutated_byte_count=analysis.cdf_bytes,
        source_archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        mutated_archive_sha256=hashlib.sha256(mutated).hexdigest(),
        source_raw_sha256=hashlib.sha256(source_raw_bytes).hexdigest(),
        mutated_raw_sha256=hashlib.sha256(mutated_raw_bytes).hexdigest(),
        raw_equal=source_raw_bytes == mutated_raw_bytes,
        raw_byte_count=len(source_raw_bytes),
        max_abs_raw_byte_delta=_max_abs_byte_delta(source_raw_bytes, mutated_raw_bytes),
    )


def prove_atw2_cdf_compaction_parity(
    archive_bytes: bytes,
    *,
    device: str = "cpu",
) -> Atw2CdfCompactionProof:
    """Compact cdf_table_blob, inflate both archives, and compare raw outputs."""

    analysis = analyze_atw2_cdf_section(archive_bytes)
    compact = compose_atw2_archive_without_cdf_table(archive_bytes)
    with tempfile.TemporaryDirectory(prefix="atw2-cdf-compact-proof-") as tmp:
        root = Path(tmp)
        source_raw = root / "source.raw"
        compact_raw = root / "compact.raw"
        inflate_one_video(archive_bytes, source_raw, device=device)
        inflate_one_video(compact, compact_raw, device=device)
        source_raw_bytes = source_raw.read_bytes()
        compact_raw_bytes = compact_raw.read_bytes()

    bytes_saved = len(archive_bytes) - len(compact)
    delta_s = -CANONICAL_RATE_MULTIPLIER * bytes_saved / CANONICAL_RATE_DENOM_BYTES
    return Atw2CdfCompactionProof(
        analysis=analysis,
        compact_cdf_bytes=len(ATW2_CDF_DEAD_SECTION_SENTINEL),
        source_archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        compact_archive_sha256=hashlib.sha256(compact).hexdigest(),
        source_archive_bytes=len(archive_bytes),
        compact_archive_bytes=len(compact),
        archive_bytes_saved=bytes_saved,
        delta_s_rate_only=float(delta_s),
        source_raw_sha256=hashlib.sha256(source_raw_bytes).hexdigest(),
        compact_raw_sha256=hashlib.sha256(compact_raw_bytes).hexdigest(),
        raw_equal=source_raw_bytes == compact_raw_bytes,
        raw_byte_count=len(source_raw_bytes),
        max_abs_raw_byte_delta=_max_abs_byte_delta(source_raw_bytes, compact_raw_bytes),
    )
