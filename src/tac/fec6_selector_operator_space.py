# SPDX-License-Identifier: MIT
"""Audit grammar-level FEC6 selector mutation operators.

The PR101/FEC6 CPU-frontier packet is close enough to the 0.192 threshold that
raw byte tinkering is tempting. This module keeps the search at the valid
grammar layer: decode the ``FP11`` wrapper, decode the ``FEC6`` fixed-Huffman
selector stream, and rank selector substitutions as packet operators while
leaving score and dispatch authority fail-closed.
"""

from __future__ import annotations

import json
import math
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tac.repo_io import repo_relative, sha256_bytes, sha256_file

OUTER_MAGIC = b"FP11"
SELECTOR_MAGIC = b"FEC6"
RATE_DENOMINATOR_BYTES = 37_545_489
CURRENT_FEC6_CPU_SCORE = 0.1920513168811056
SUB_0192_TARGET_SCORE = 0.192

FEC6_FIXED_K16_MODE_IDS: tuple[str, ...] = (
    "none",
    "frame0_blue_chroma_amp_1",
    "frame0_blue_chroma_amp_3",
    "frame0_luma_bias_+1",
    "frame0_luma_bias_-1",
    "frame0_luma_bias_-2",
    "frame0_luma_bias_-4",
    "frame0_rgb_bias_m2_p1_p1",
    "frame0_rgb_bias_m4_p2_p2",
    "frame0_rgb_bias_p0_m1_p1",
    "frame0_rgb_bias_p0_m2_p2",
    "frame0_rgb_bias_p0_p1_m1",
    "frame0_rgb_bias_p0_p2_m2",
    "frame0_rgb_bias_p2_m1_m1",
    "frame0_rgb_bias_p4_m2_m2",
    "frame0_roll_dx+0_dy+1",
)

FEC6_FIXED_K16_CODE_BITS: tuple[str, ...] = (
    "00",
    "1100",
    "01",
    "111010",
    "11010",
    "111011",
    "111100",
    "100",
    "111101",
    "11011",
    "1111110",
    "111110",
    "11111110",
    "101",
    "11100",
    "11111111",
)
FEC6_FIXED_K16_DECODE: dict[str, int] = {
    bits: code for code, bits in enumerate(FEC6_FIXED_K16_CODE_BITS)
}
FEC6_FIXED_K16_MODE_TO_CODE: dict[str, int] = {
    mode_id: code for code, mode_id in enumerate(FEC6_FIXED_K16_MODE_IDS)
}


class Fec6SelectorOperatorError(ValueError):
    """Raised when a selector operator audit cannot be built safely."""


@dataclass(frozen=True)
class Fec6SelectorArchive:
    archive_path: str
    archive_bytes: int
    archive_sha256: str
    member_name: str
    source_payload_bytes: int
    wrapper_overhead_bytes: int
    selector_payload_bytes: int
    selector_index_bytes: int
    selector_code_bits_total: int
    selector_payload_sha256: str
    n_pairs: int
    codes: tuple[int, ...]
    histogram: dict[str, int]

    def to_manifest(self) -> dict[str, object]:
        payload = asdict(self)
        payload["mode_histogram"] = {
            FEC6_FIXED_K16_MODE_IDS[int(code)]: count
            for code, count in self.histogram.items()
        }
        return payload


@dataclass(frozen=True)
class Fec6SelectorMutationRow:
    operator_id: str
    pair: int
    current_code: int
    current_mode_id: str
    candidate_code: int
    candidate_mode_id: str
    mutation_grain: str
    mutation_operator: str
    current_code_bits: int
    candidate_code_bits: int
    selector_code_bit_delta: int
    selector_index_byte_delta_if_single_mutation: int
    rate_delta_score_if_components_unchanged: float
    pair_component_delta_no_rate_proxy: float
    pair_posenet_delta_proxy: float
    pair_segnet_delta_proxy: float
    local_proxy_delta_with_rate: float
    evidence_axis: str
    ready_for_operator_probe: bool
    ready_for_provider_dispatch: bool
    ready_for_exact_eval_dispatch: bool
    score_claim: bool
    promotion_eligible: bool
    rank_or_kill_eligible: bool
    dispatch_attempted: bool
    blockers: tuple[str, ...]

    def to_manifest(self) -> dict[str, object]:
        return asdict(self)


def decode_fec6_fixed_huffman_codes(payload: bytes, *, n_pairs: int) -> tuple[tuple[int, ...], int]:
    """Decode the fixed-Huffman FEC6 selector payload body."""

    codes: list[int] = []
    prefix = ""
    bit_pos = 0
    max_bits = len(payload) * 8
    while len(codes) < n_pairs:
        if bit_pos >= max_bits:
            raise Fec6SelectorOperatorError("FEC6 selector bitstream truncated")
        bit = (payload[bit_pos // 8] >> (7 - (bit_pos % 8))) & 1
        bit_pos += 1
        prefix += "1" if bit else "0"
        code = FEC6_FIXED_K16_DECODE.get(prefix)
        if code is None:
            if len(prefix) > 8:
                raise Fec6SelectorOperatorError("FEC6 selector has invalid prefix code")
            continue
        codes.append(code)
        prefix = ""
    if prefix:
        raise Fec6SelectorOperatorError("FEC6 selector ended mid-symbol")
    for trailing in range(bit_pos, max_bits):
        if (payload[trailing // 8] >> (7 - (trailing % 8))) & 1:
            raise Fec6SelectorOperatorError("FEC6 selector has non-zero padding bits")
    return tuple(codes), bit_pos


def encode_fec6_fixed_huffman_codes(codes: Iterable[int]) -> tuple[bytes, int]:
    """Encode FEC6 selector codes with the fixed K16 Huffman table."""

    bit_string = "".join(FEC6_FIXED_K16_CODE_BITS[int(code)] for code in codes)
    bit_count = len(bit_string)
    if bit_count == 0:
        return b"", 0
    pad = (-bit_count) % 8
    bit_string += "0" * pad
    return (
        bytes(int(bit_string[pos: pos + 8], 2) for pos in range(0, len(bit_string), 8)),
        bit_count,
    )


def parse_fec6_selector_archive(path: Path, *, repo_root: Path | None = None) -> Fec6SelectorArchive:
    """Parse a single-member PR101 ``FP11`` archive carrying a FEC6 selector."""

    repo_root = repo_root or Path.cwd()
    member_name, wrapper = _single_member_payload(path)
    if len(wrapper) < 10 or wrapper[:4] != OUTER_MAGIC:
        raise Fec6SelectorOperatorError(f"expected FP11 wrapper in {path}")
    source_len = int.from_bytes(wrapper[4:8], "little")
    selector_len_offset = 8 + source_len
    if selector_len_offset + 2 > len(wrapper):
        raise Fec6SelectorOperatorError("FP11 wrapper truncated before selector length")
    selector_len = int.from_bytes(wrapper[selector_len_offset: selector_len_offset + 2], "little")
    selector_start = selector_len_offset + 2
    selector_end = selector_start + selector_len
    if selector_end != len(wrapper):
        raise Fec6SelectorOperatorError("FP11 wrapper selector length does not consume payload")
    selector_payload = wrapper[selector_start:selector_end]
    if len(selector_payload) < 6 or selector_payload[:4] != SELECTOR_MAGIC:
        raise Fec6SelectorOperatorError(
            f"expected FEC6 selector payload, got {selector_payload[:4]!r}"
        )
    n_pairs = int.from_bytes(selector_payload[4:6], "little")
    codes, used_bits = decode_fec6_fixed_huffman_codes(selector_payload[6:], n_pairs=n_pairs)
    encoded, reencoded_bits = encode_fec6_fixed_huffman_codes(codes)
    if reencoded_bits != used_bits or encoded != selector_payload[6:]:
        raise Fec6SelectorOperatorError("FEC6 selector failed fixed-Huffman roundtrip")
    counts = Counter(codes)
    return Fec6SelectorArchive(
        archive_path=repo_relative(path, repo_root),
        archive_bytes=path.stat().st_size,
        archive_sha256=sha256_file(path),
        member_name=member_name,
        source_payload_bytes=source_len,
        wrapper_overhead_bytes=10,
        selector_payload_bytes=len(selector_payload),
        selector_index_bytes=len(selector_payload) - 6,
        selector_code_bits_total=used_bits,
        selector_payload_sha256=sha256_bytes(selector_payload),
        n_pairs=n_pairs,
        codes=codes,
        histogram={str(code): int(count) for code, count in sorted(counts.items())},
    )


def build_fec6_selector_operator_space(
    *,
    fec6_archive: Path,
    pair_component_rows_paths: tuple[Path, ...] = (),
    repo_root: Path | None = None,
    current_cpu_score: float = CURRENT_FEC6_CPU_SCORE,
    target_cpu_score: float = SUB_0192_TARGET_SCORE,
    max_rows: int = 20,
) -> dict[str, Any]:
    """Build a fail-closed manifest of grammar-valid FEC6 selector operators."""

    repo_root = repo_root or Path.cwd()
    selector = parse_fec6_selector_archive(fec6_archive, repo_root=repo_root)
    pair_tables = _load_pair_component_tables(pair_component_rows_paths, repo_root=repo_root)
    mutation_rows = _build_mutation_rows(selector, pair_tables)
    rows_by_proxy = sorted(
        mutation_rows,
        key=lambda row: (
            row.local_proxy_delta_with_rate,
            row.selector_code_bit_delta,
            row.pair,
            row.candidate_mode_id,
        ),
    )
    rows_by_bit = sorted(
        mutation_rows,
        key=lambda row: (
            row.selector_code_bit_delta,
            row.local_proxy_delta_with_rate,
            row.pair,
            row.candidate_mode_id,
        ),
    )
    pareto = [
        row
        for row in rows_by_proxy
        if row.local_proxy_delta_with_rate < 0.0 and row.selector_code_bit_delta <= 0
    ][:max_rows]
    entropy_floor_bits = _shannon_bits(Counter(selector.codes)) * selector.n_pairs
    entropy_floor_bytes = _ceil_bytes(entropy_floor_bits)
    required_bytes = _required_rate_bytes_to_cross(current_cpu_score, target_cpu_score)

    return {
        "schema": "tac_fec6_selector_operator_space_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_operator_probe": False,
        "ready_for_provider_dispatch": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "raw_byte_gradient_valid": False,
        "raw_archive_byte_rows_emitted": 0,
        "axis_label": "planning_proxy_plus_current_contest_cpu_anchor",
        "source_archive": selector.to_manifest(),
        "score_threshold": {
            "current_cpu_score": float(current_cpu_score),
            "target_cpu_score": float(target_cpu_score),
            "required_rate_bytes_to_strictly_cross_target_if_components_unchanged": required_bytes,
            "rate_delta_per_byte": 25.0 / RATE_DENOMINATOR_BYTES,
        },
        "selector_entropy": {
            "selector_payload_bytes": selector.selector_payload_bytes,
            "selector_index_bytes": selector.selector_index_bytes,
            "selector_code_bits_total": selector.selector_code_bits_total,
            "zero_header_entropy_floor_bytes": entropy_floor_bytes,
            "gap_payload_to_zero_header_entropy_floor_bytes": selector.selector_payload_bytes
            - entropy_floor_bytes,
            "gap_index_to_zero_header_entropy_floor_bytes": selector.selector_index_bytes
            - entropy_floor_bytes,
        },
        "pair_component_rows": {
            "paths": [repo_relative(path, repo_root) for path in pair_component_rows_paths],
            "pair_count": len(pair_tables),
            "row_count": sum(len(modes) for modes in pair_tables.values()),
            "evidence_axis": "macOS-CPU or proxy pair rows; not score evidence",
        },
        "operator_row_count": len(mutation_rows),
        "top_proxy_improving_rows": [row.to_manifest() for row in rows_by_proxy[:max_rows]],
        "top_bit_saving_rows": [row.to_manifest() for row in rows_by_bit[:max_rows]],
        "proxy_and_nonpositive_bit_rows": [row.to_manifest() for row in pareto],
        "conclusion": {
            "same_runtime_byte_only_selector_polish_blocked": (
                selector.selector_payload_bytes - entropy_floor_bytes < required_bytes
            ),
            "same_runtime_selector_reason": (
                "selector entropy gap is below the strict byte gap to sub-0.192; "
                "component-moving selector mutations need packet materialization and exact axes"
            ),
            "next_packet_operator": (
                "materialize a grammar-level FEC6 selector substitution only after choosing "
                "a row with proxy-improving component signal and nonpositive bit delta, then "
                "prove byte consumption and run paired CPU/CUDA exact eval"
            ),
        },
        "dispatch_blockers": (
            "pair_component_rows_proxy_only",
            "packet_candidate_not_materialized",
            "inflate_success_proof_missing",
            "runtime_byte_consumption_noop_detector_missing",
            "paired_contest_cpu_cuda_exact_eval_missing",
        ),
    }


def _build_mutation_rows(
    selector: Fec6SelectorArchive,
    pair_tables: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> list[Fec6SelectorMutationRow]:
    rows: list[Fec6SelectorMutationRow] = []
    current_index_bytes = _ceil_bytes(selector.selector_code_bits_total)
    for pair, modes in sorted(pair_tables.items()):
        if pair < 0 or pair >= len(selector.codes):
            continue
        current_code = int(selector.codes[pair])
        current_mode = FEC6_FIXED_K16_MODE_IDS[current_code]
        current_row = modes.get(current_mode)
        if current_row is None:
            continue
        for candidate_mode, candidate_row in sorted(modes.items()):
            if candidate_mode == current_mode or candidate_mode not in FEC6_FIXED_K16_MODE_TO_CODE:
                continue
            candidate_code = FEC6_FIXED_K16_MODE_TO_CODE[candidate_mode]
            bit_delta = len(FEC6_FIXED_K16_CODE_BITS[candidate_code]) - len(
                FEC6_FIXED_K16_CODE_BITS[current_code]
            )
            index_byte_delta = _ceil_bytes(selector.selector_code_bits_total + bit_delta) - current_index_bytes
            rate_delta = 25.0 * index_byte_delta / RATE_DENOMINATOR_BYTES
            component_delta = float(candidate_row["component_score_no_rate"]) - float(
                current_row["component_score_no_rate"]
            )
            pose_delta = float(candidate_row["posenet_dist"]) - float(current_row["posenet_dist"])
            seg_delta = float(candidate_row["segnet_dist"]) - float(current_row["segnet_dist"])
            rows.append(
                Fec6SelectorMutationRow(
                    operator_id=f"fec6_selector_pair_{pair}_{current_mode}_to_{candidate_mode}",
                    pair=pair,
                    current_code=current_code,
                    current_mode_id=current_mode,
                    candidate_code=candidate_code,
                    candidate_mode_id=candidate_mode,
                    mutation_grain="grammar_aware_selector_symbol",
                    mutation_operator="fec6_fixed_huffman_single_pair_substitution",
                    current_code_bits=len(FEC6_FIXED_K16_CODE_BITS[current_code]),
                    candidate_code_bits=len(FEC6_FIXED_K16_CODE_BITS[candidate_code]),
                    selector_code_bit_delta=bit_delta,
                    selector_index_byte_delta_if_single_mutation=index_byte_delta,
                    rate_delta_score_if_components_unchanged=rate_delta,
                    pair_component_delta_no_rate_proxy=component_delta,
                    pair_posenet_delta_proxy=pose_delta,
                    pair_segnet_delta_proxy=seg_delta,
                    local_proxy_delta_with_rate=component_delta + rate_delta,
                    evidence_axis="pair-row proxy/advisory only",
                    ready_for_operator_probe=False,
                    ready_for_provider_dispatch=False,
                    ready_for_exact_eval_dispatch=False,
                    score_claim=False,
                    promotion_eligible=False,
                    rank_or_kill_eligible=False,
                    dispatch_attempted=False,
                    blockers=(
                        "pair_component_rows_proxy_only",
                        "packet_candidate_not_materialized",
                        "inflate_success_proof_missing",
                        "runtime_byte_consumption_noop_detector_missing",
                    ),
                )
            )
    return rows


def _single_member_payload(path: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise Fec6SelectorOperatorError(f"ZIP CRC failed for member {bad!r}")
        infos = zf.infolist()
        if len(infos) != 1:
            raise Fec6SelectorOperatorError(f"expected single-member archive, got {len(infos)}")
        info = infos[0]
        return info.filename, zf.read(info.filename)


def _load_pair_component_tables(
    paths: tuple[Path, ...],
    *,
    repo_root: Path,
) -> dict[int, dict[str, dict[str, Any]]]:
    tables: dict[int, dict[str, dict[str, Any]]] = {}
    for path in paths:
        if not path.exists():
            raise Fec6SelectorOperatorError(f"pair component rows path missing: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                pair = int(row["pair"])
                mode_id = str(row["mode_id"])
                if mode_id in tables.setdefault(pair, {}):
                    raise Fec6SelectorOperatorError(
                        f"duplicate pair/mode in {repo_relative(path, repo_root)}:{line_no}"
                    )
                tables[pair][mode_id] = row
    return tables


def _required_rate_bytes_to_cross(current_score: float, target_score: float) -> int:
    if current_score <= target_score:
        return 0
    return math.floor((current_score - target_score) * RATE_DENOMINATOR_BYTES / 25.0) + 1


def _shannon_bits(counts: Mapping[int, int] | Counter[int]) -> float:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        count = int(value)
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _ceil_bytes(bits: float | int) -> int:
    return math.ceil(float(bits) / 8.0)


__all__ = [
    "CURRENT_FEC6_CPU_SCORE",
    "FEC6_FIXED_K16_CODE_BITS",
    "FEC6_FIXED_K16_MODE_IDS",
    "Fec6SelectorOperatorError",
    "build_fec6_selector_operator_space",
    "decode_fec6_fixed_huffman_codes",
    "encode_fec6_fixed_huffman_codes",
    "parse_fec6_selector_archive",
]
