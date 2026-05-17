# SPDX-License-Identifier: MIT
"""A1 Rule #6 byte-escape profiling.

This module profiles the exact A1 packet sections before proposing any
byte-only bolt-on. It is deliberately planning-only: it reads archive bytes,
reports section-conditioned coder evidence, and does not emit a candidate
archive or score claim.
"""
from __future__ import annotations

import lzma
import math
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.frontier_archive_layout import (
    A1_DECODER_SECTION_TOTAL,
    HNERV_PREFIX_LEN,
    PR101_INNER_MEMBER_NAME,
    PR101_LATENT_BLOB_LEN,
)
from tac.packet_compiler import RankedSidecarSchema
from tac.packet_compiler.pr101_sidecar_grammar import (
    _decode_combination_colex,
    _decode_huff_length_rank,
    _huff_length_vector_count,
)
from tac.repo_io import repo_relative, sha256_bytes, sha256_file

RATE_DENOMINATOR_BYTES = 37_545_489
N_PAIRS = 600
LATENT_DIM = 28
SIDECAR_HUFF_ENUM_LEN = 607
SIDECAR_HUFF_COMB_LEN = 609
SIDECAR_HUFF_LEN = 614
SIDECAR_SPLIT_LEN = 656
SIDECAR_PACKED_LEN = 661
SIDECAR_N_PAIRS_LEN = 600
SIDECAR_PAIR_BYTES_LEN = 1200
SIDECAR_DIM_PACKED_LEN = 359
SIDECAR_DELTA_HUFF_LENGTH_RANK_LEN = 5
SIDECAR_NOOP_INFER_RANK_LEN = 3
PR101_DELTAS = (-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10)
LATENT_LZMA_FILTERS = [
    {"id": lzma.FILTER_LZMA1, "dict_size": 4096, "lc": 3, "lp": 0, "pb": 0}
]
DEFAULT_LZMA_DICT_SIZES = (
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    12288,
    16384,
    24576,
    32768,
    49152,
    65536,
    98304,
    131072,
)
SUPPORTED_RUNTIME_SIDECAR_LENGTHS = (
    SIDECAR_N_PAIRS_LEN,
    SIDECAR_HUFF_ENUM_LEN,
    SIDECAR_HUFF_COMB_LEN,
    SIDECAR_HUFF_LEN,
    SIDECAR_SPLIT_LEN,
    SIDECAR_PACKED_LEN,
    SIDECAR_PAIR_BYTES_LEN,
)


@dataclass(frozen=True)
class A1ArchiveSections:
    """Logical byte sections consumed by the A1 inflate runtime."""

    archive_path: Path
    archive_bytes: int
    archive_sha256: str
    member_name: str
    member_bytes: int
    member_sha256: str
    section_total: int
    section_header: bytes
    decoder_blob: bytes
    latent_blob: bytes
    sidecar_blob: bytes


def read_a1_archive_sections(path: Path) -> A1ArchiveSections:
    """Read and validate the exact A1 single-member prefixed archive layout."""

    archive = Path(path)
    with zipfile.ZipFile(archive) as zf:
        members = [info for info in zf.infolist() if not info.is_dir()]
        if len(members) != 1:
            raise ValueError(f"expected one ZIP member in {archive}; found {len(members)}")
        member = members[0]
        payload = zf.read(member.filename)

    if member.filename != PR101_INNER_MEMBER_NAME:
        raise ValueError(f"expected A1 member {PR101_INNER_MEMBER_NAME!r}; got {member.filename!r}")
    if len(payload) < A1_DECODER_SECTION_TOTAL + PR101_LATENT_BLOB_LEN:
        raise ValueError("A1 payload too short for decoder section plus latent blob")

    section_total = int.from_bytes(payload[:HNERV_PREFIX_LEN], "little")
    if section_total != A1_DECODER_SECTION_TOTAL:
        raise ValueError(
            "bad A1 decoder section total: "
            f"got {section_total}, expected {A1_DECODER_SECTION_TOTAL}"
        )
    latent_start = section_total
    latent_end = latent_start + PR101_LATENT_BLOB_LEN
    sidecar_blob = payload[latent_end:]
    if not sidecar_blob:
        raise ValueError("A1 sidecar is empty; expected PR101-style correction sidecar")

    return A1ArchiveSections(
        archive_path=archive,
        archive_bytes=archive.stat().st_size,
        archive_sha256=sha256_file(archive),
        member_name=member.filename,
        member_bytes=len(payload),
        member_sha256=sha256_bytes(payload),
        section_total=section_total,
        section_header=payload[:HNERV_PREFIX_LEN],
        decoder_blob=payload[HNERV_PREFIX_LEN:section_total],
        latent_blob=payload[latent_start:latent_end],
        sidecar_blob=sidecar_blob,
    )


def _shannon_bits(counts: Mapping[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _ceil_bytes(bits: float) -> int:
    return max(0, math.ceil(bits / 8.0))


def _lzma_filter_row(filters: list[dict[str, int]]) -> dict[str, int]:
    row = dict(filters[0])
    row.pop("id", None)
    return {str(key): int(value) for key, value in row.items()}


def latent_lzma_sweep(
    latent_blob: bytes,
    *,
    dict_sizes: Iterable[int] = DEFAULT_LZMA_DICT_SIZES,
    max_lc: int = 4,
    max_lp: int = 4,
    max_pb: int = 4,
) -> dict[str, Any]:
    """Profile raw-LZMA settings for A1's already-consumed latent stream."""

    raw = lzma.decompress(latent_blob, format=lzma.FORMAT_RAW, filters=LATENT_LZMA_FILTERS)
    source_roundtrip = lzma.compress(raw, format=lzma.FORMAT_RAW, filters=LATENT_LZMA_FILTERS)
    rows: list[dict[str, Any]] = []
    invalid = 0
    for dict_size in dict_sizes:
        for lc in range(max_lc + 1):
            for lp in range(max_lp + 1):
                if lc + lp > 4:
                    continue
                for pb in range(max_pb + 1):
                    filters = [
                        {
                            "id": lzma.FILTER_LZMA1,
                            "dict_size": int(dict_size),
                            "lc": lc,
                            "lp": lp,
                            "pb": pb,
                        }
                    ]
                    try:
                        compressed = lzma.compress(
                            raw,
                            format=lzma.FORMAT_RAW,
                            filters=filters,
                        )
                    except lzma.LZMAError:
                        invalid += 1
                        continue
                    rows.append(
                        {
                            "bytes": len(compressed),
                            "delta_vs_source_bytes": len(compressed) - len(latent_blob),
                            "dict_size": int(dict_size),
                            "lc": lc,
                            "lp": lp,
                            "pb": pb,
                        }
                    )
    if not rows:
        raise ValueError("A1 latent raw-LZMA sweep produced no valid candidates")

    rows.sort(
        key=lambda row: (
            int(row["bytes"]),
            int(row["dict_size"]),
            int(row["lc"]),
            int(row["lp"]),
            int(row["pb"]),
        )
    )
    best = rows[0]
    return {
        "source_blob_bytes": len(latent_blob),
        "source_blob_sha256": sha256_bytes(latent_blob),
        "raw_latent_bytes": len(raw),
        "raw_latent_sha256": sha256_bytes(raw),
        "source_filter": _lzma_filter_row(LATENT_LZMA_FILTERS),
        "source_filter_roundtrip_bytes": len(source_roundtrip),
        "source_filter_roundtrip_sha256": sha256_bytes(source_roundtrip),
        "source_filter_roundtrip_exact": source_roundtrip == latent_blob,
        "valid_candidates": len(rows),
        "invalid_candidates": invalid,
        "best": best,
        "best_beats_source": int(best["bytes"]) < len(latent_blob),
        "top20": rows[:20],
    }


def decode_pr101_huff_enum_sidecar(sidecar: bytes) -> dict[str, Any]:
    """Decode A1's 607-byte PR101 Huffman-enum sidecar into diagnostics."""

    if len(sidecar) != SIDECAR_HUFF_ENUM_LEN:
        raise ValueError(
            f"expected {SIDECAR_HUFF_ENUM_LEN}-byte Huffman-enum sidecar, got {len(sidecar)}"
        )
    schema = RankedSidecarSchema(
        n_pairs=N_PAIRS,
        n_dims=LATENT_DIM,
        deltas=PR101_DELTAS,
        huff_min_len=2,
        huff_max_len=8,
    )
    dim_end = SIDECAR_DIM_PACKED_LEN
    rank_end = dim_end + SIDECAR_DELTA_HUFF_LENGTH_RANK_LEN
    noop_rank_start = len(sidecar) - SIDECAR_NOOP_INFER_RANK_LEN

    length_rank = int.from_bytes(sidecar[dim_end:rank_end], "little")
    lengths = _decode_huff_length_rank(length_rank, schema)

    decode: dict[tuple[int, int], int] = {}
    code = 0
    previous_length = 0
    for symbol, length in sorted(
        ((symbol, int(length)) for symbol, length in enumerate(lengths) if length),
        key=lambda item: (item[1], item[0]),
    ):
        code <<= length - previous_length
        decode[(length, code)] = symbol
        code += 1
        previous_length = length

    delta_valid: list[int] = []
    cur = 0
    cur_len = 0
    huff_payload = sidecar[rank_end:noop_rank_start]
    for byte in huff_payload:
        for shift in range(7, -1, -1):
            cur = (cur << 1) | ((byte >> shift) & 1)
            cur_len += 1
            symbol = decode.get((cur_len, cur))
            if symbol is not None:
                delta_valid.append(symbol)
                cur = 0
                cur_len = 0
    if cur_len:
        raise ValueError("A1 sidecar Huffman payload ended mid-symbol")

    n_valid = len(delta_valid)
    noop_count = N_PAIRS - n_valid
    noop_rank = int.from_bytes(sidecar[noop_rank_start:], "little")
    noop_pos = _decode_combination_colex(noop_rank, N_PAIRS, noop_count)

    dim_value = int.from_bytes(sidecar[:dim_end], "little")
    dims_valid = []
    for _ in range(n_valid):
        dim_value, dim = divmod(dim_value, LATENT_DIM)
        dims_valid.append(dim)
    if dim_value:
        raise ValueError("A1 sidecar dimension mixed-radix residue is nonzero")

    delta_counts = Counter(delta_valid)
    dim_counts = Counter(dims_valid)
    delta_entropy_bits_per_symbol = _shannon_bits(delta_counts)
    dim_entropy_bits_per_symbol = _shannon_bits(dim_counts)
    valid_mask_bits = math.log2(math.comb(N_PAIRS, noop_count))
    length_rank_bits = math.ceil(
        math.log2(
            _huff_length_vector_count(
                0,
                schema.kraft_total,
                n_symbols=len(PR101_DELTAS),
                huff_min_len=schema.huff_min_len,
                huff_max_len=schema.huff_max_len,
            )
        )
    )
    entropy_floor_bytes = (
        _ceil_bytes(delta_entropy_bits_per_symbol * n_valid)
        + _ceil_bytes(dim_entropy_bits_per_symbol * n_valid)
        + _ceil_bytes(valid_mask_bits)
        + _ceil_bytes(length_rank_bits)
    )
    choices = [
        1 + int(dim) * len(PR101_DELTAS) + int(delta_idx)
        for dim, delta_idx in zip(dims_valid, delta_valid, strict=True)
    ]
    n_pairs_valid = all(0 <= choice <= 255 for choice in choices)
    return {
        "current_sidecar_bytes": len(sidecar),
        "current_sidecar_sha256": sha256_bytes(sidecar),
        "format": "pr101_huff_enum_607",
        "n_valid": n_valid,
        "noop_count": noop_count,
        "noop_positions": [int(x) for x in noop_pos.tolist()],
        "component_bytes": {
            "dims_mixed_radix": SIDECAR_DIM_PACKED_LEN,
            "length_rank": SIDECAR_DELTA_HUFF_LENGTH_RANK_LEN,
            "huffman_delta_payload": len(huff_payload),
            "noop_rank": SIDECAR_NOOP_INFER_RANK_LEN,
        },
        "entropy_floor_bytes_estimate": entropy_floor_bytes,
        "gap_to_entropy_floor_bytes_estimate": len(sidecar) - entropy_floor_bytes,
        "delta_entropy_bits_per_symbol": delta_entropy_bits_per_symbol,
        "dim_entropy_bits_per_symbol": dim_entropy_bits_per_symbol,
        "delta_counts": {str(key): int(value) for key, value in sorted(delta_counts.items())},
        "top_dim_counts": [
            {"dim": int(dim), "count": int(count)}
            for dim, count in dim_counts.most_common(12)
        ],
        "choice_max_for_600_byte_runtime_format": max(choices) if choices else 0,
        "choice_values_fit_u8_runtime_format": n_pairs_valid,
        "runtime_supported_lengths_bytes": list(SUPPORTED_RUNTIME_SIDECAR_LENGTHS),
        "runtime_min_supported_length_bytes": min(SUPPORTED_RUNTIME_SIDECAR_LENGTHS),
        "runtime_min_supported_length_usable_for_current_semantics": (
            SIDECAR_N_PAIRS_LEN if n_pairs_valid else SIDECAR_HUFF_ENUM_LEN
        ),
        "runtime_min_supported_delta_vs_current_bytes": (
            (SIDECAR_N_PAIRS_LEN if n_pairs_valid else SIDECAR_HUFF_ENUM_LEN) - len(sidecar)
        ),
    }


def build_a1_byte_escape_profile(archive_path: Path, *, repo_root: Path) -> dict[str, Any]:
    """Build the deterministic A1 Rule #6 byte-escape profile."""

    sections = read_a1_archive_sections(archive_path)
    latent = latent_lzma_sweep(sections.latent_blob)
    sidecar = decode_pr101_huff_enum_sidecar(sections.sidecar_blob)
    byte_term_per_archive_byte = 25.0 / RATE_DENOMINATOR_BYTES
    best_latent_delta = int(latent["best"]["delta_vs_source_bytes"])
    best_sidecar_delta = int(sidecar["runtime_min_supported_delta_vs_current_bytes"])
    best_supported_delta = min(0, best_latent_delta) + min(0, best_sidecar_delta)

    return {
        "schema_version": 1,
        "tool": "tools/profile_a1_byte_escape_routes.py",
        "analysis_date": "2026-05-17",
        "lane_id": "a1_rule6_byte_escape_profile_20260517_codex",
        "source": {
            "archive_path": repo_relative(sections.archive_path, repo_root),
            "archive_bytes": sections.archive_bytes,
            "archive_sha256": sections.archive_sha256,
            "member_name": sections.member_name,
            "member_bytes": sections.member_bytes,
            "member_sha256": sections.member_sha256,
        },
        "sections": {
            "decoder_section_total_u32le": sections.section_total,
            "header_bytes": len(sections.section_header),
            "decoder_blob_bytes": len(sections.decoder_blob),
            "decoder_blob_sha256": sha256_bytes(sections.decoder_blob),
            "latent_blob_bytes": len(sections.latent_blob),
            "latent_blob_sha256": sha256_bytes(sections.latent_blob),
            "sidecar_blob_bytes": len(sections.sidecar_blob),
            "sidecar_blob_sha256": sha256_bytes(sections.sidecar_blob),
        },
        "latent_lzma": latent,
        "sidecar_huff_enum": sidecar,
        "byte_escape_summary": {
            "best_supported_delta_bytes_without_runtime_change": best_supported_delta,
            "rate_term_delta_if_component_distances_unchanged": (
                byte_term_per_archive_byte * best_supported_delta
            ),
            "classification": (
                "saturated_byte_only_current_runtime"
                if best_supported_delta == 0
                else "supported_byte_escape_available"
            ),
            "next_action": (
                "Do not retread generic arithmetic over A1 latent or sidecar bytes. "
                "Move Rule #6 to component-changing bolt-ons, per-section byte-consumption "
                "proofs, or a new runtime grammar before claiming score movement."
            ),
        },
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "contest_axis": None,
            "dispatch_blockers": [
                "planning_profile_only_no_candidate_archive",
                "no_component_distances_recomputed",
                "no_byte_consumption_mutation_proof_for_new_runtime",
            ],
        },
    }


def render_a1_byte_escape_markdown(profile: Mapping[str, Any]) -> str:
    """Render a compact ledger for human review."""

    source = profile["source"]
    sections = profile["sections"]
    latent = profile["latent_lzma"]
    sidecar = profile["sidecar_huff_enum"]
    summary = profile["byte_escape_summary"]
    lines = [
        "# A1 Rule #6 byte-escape profile",
        "",
        "This is a planning ledger, not a score claim. It profiles the current A1 "
        "archive sections before any Rule #6 byte-only bolt-on is treated as live "
        "frontier work.",
        "",
        "## Authority",
        "",
        "- score_claim: false",
        "- promotion_eligible: false",
        "- ready_for_exact_eval_dispatch: false",
        "- contest_axis: null",
        "",
        "## Source",
        "",
        f"- archive: `{source['archive_path']}`",
        f"- archive bytes: `{source['archive_bytes']}`",
        f"- archive sha256: `{source['archive_sha256']}`",
        f"- member `{source['member_name']}` bytes: `{source['member_bytes']}`",
        f"- member sha256: `{source['member_sha256']}`",
        "",
        "## Section Map",
        "",
        f"- header bytes: `{sections['header_bytes']}`",
        f"- decoder section total: `{sections['decoder_section_total_u32le']}`",
        f"- decoder blob bytes: `{sections['decoder_blob_bytes']}`",
        f"- latent blob bytes: `{sections['latent_blob_bytes']}`",
        f"- sidecar blob bytes: `{sections['sidecar_blob_bytes']}`",
        "",
        "## Latent Raw-LZMA Sweep",
        "",
        f"- source filter roundtrip exact: `{latent['source_filter_roundtrip_exact']}`",
        f"- raw latent bytes: `{latent['raw_latent_bytes']}`",
        f"- valid candidates: `{latent['valid_candidates']}`",
        f"- invalid candidates: `{latent['invalid_candidates']}`",
        f"- best bytes: `{latent['best']['bytes']}`",
        f"- best delta vs source bytes: `{latent['best']['delta_vs_source_bytes']}`",
        f"- best filter: `dict={latent['best']['dict_size']} lc={latent['best']['lc']} "
        f"lp={latent['best']['lp']} pb={latent['best']['pb']}`",
        "",
        "## Sidecar Runtime Formats",
        "",
        f"- current sidecar bytes: `{sidecar['current_sidecar_bytes']}`",
        f"- decoded valid corrections: `{sidecar['n_valid']}`",
        f"- no-op pairs: `{sidecar['noop_count']}`",
        f"- entropy floor estimate bytes: `{sidecar['entropy_floor_bytes_estimate']}`",
        f"- gap to entropy floor estimate bytes: `{sidecar['gap_to_entropy_floor_bytes_estimate']}`",
        f"- 600-byte runtime format fits current choices: "
        f"`{sidecar['choice_values_fit_u8_runtime_format']}`",
        f"- max encoded choice value: `{sidecar['choice_max_for_600_byte_runtime_format']}`",
        f"- usable minimum runtime-supported sidecar bytes: "
        f"`{sidecar['runtime_min_supported_length_usable_for_current_semantics']}`",
        f"- delta vs current sidecar bytes: "
        f"`{sidecar['runtime_min_supported_delta_vs_current_bytes']}`",
        "",
        "## Conclusion",
        "",
        f"- classification: `{summary['classification']}`",
        f"- best supported delta bytes without runtime change: "
        f"`{summary['best_supported_delta_bytes_without_runtime_change']}`",
        f"- rate-term delta if component distances unchanged: "
        f"`{summary['rate_term_delta_if_component_distances_unchanged']:.12g}`",
        "",
        summary["next_action"],
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "A1ArchiveSections",
    "build_a1_byte_escape_profile",
    "decode_pr101_huff_enum_sidecar",
    "latent_lzma_sweep",
    "read_a1_archive_sections",
    "render_a1_byte_escape_markdown",
]
