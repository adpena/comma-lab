#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile remaining PR101/FEC6 byte-only escape routes.

This is a planning/audit tool. It does not mutate archives, dispatch evals, or
claim score movement. Its purpose is to make the PR101/FEC6 near-0.192 basin
reproducible enough that future agents do not reopen saturated byte-only work
without a new component-changing hypothesis.
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import brotli

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.packet_compiler import RankedSidecarSchema  # noqa: E402
from tac.packet_compiler.pr101_sidecar_grammar import (  # noqa: E402
    _decode_combination_colex,
    _decode_huff_length_rank,
    _huff_length_vector_count,
)
from tac.repo_io import sha256_bytes, write_json  # noqa: E402

DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
)
DEFAULT_FEC6_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
)
DEFAULT_CPU_RESULT = (
    REPO_ROOT
    / "experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/modal_cpu_auth_eval_result.json"
)
DEFAULT_CUDA_RESULT = (
    REPO_ROOT
    / "experiments/results/modal_auth_eval/archive_6bae0201fb08/modal_cuda_auth_eval_result.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/pr101_fec6_byte_escape_profile_20260515_codex"
)

RATE_DENOMINATOR_BYTES = 37_545_489
TARGET_CPU_SCORE = 0.192

DECODER_BLOB_LEN = 162_164
LATENT_BLOB_LEN = 15_387
N_PAIRS = 600
LATENT_DIM = 28
LATENT_LZMA_FILTERS = [
    {"id": lzma.FILTER_LZMA1, "dict_size": 4096, "lc": 3, "lp": 0, "pb": 0}
]
PR101_DELTAS = (-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10)

FEC6_CODE_BITS = (
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


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _read_single_member_payload(path: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise ValueError(f"expected one archive member in {path}, found {len(infos)}")
        info = infos[0]
        return info.filename, zf.read(info.filename)


def _shannon_bits(counts: Mapping[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _ceil_bytes(bits: float) -> int:
    return math.ceil(bits / 8.0)


def _split_brotli_streams(data: bytes) -> list[tuple[bytes, bytes]]:
    rows: list[tuple[bytes, bytes]] = []
    pos = 0
    while pos < len(data):
        dec = brotli.Decompressor()
        chunks: list[bytes] = []
        start = pos
        while pos < len(data) and not dec.is_finished():
            chunks.append(dec.process(data[pos : pos + 1]))
            pos += 1
        if not dec.is_finished():
            raise ValueError("truncated PR101 decoder Brotli stream")
        rows.append((data[start:pos], b"".join(chunks)))
    return rows


def _best_brotli_recompress(raw_streams: Sequence[bytes]) -> dict[str, Any]:
    qualities = (9, 10, 11)
    lgwin_values = tuple(range(10, 25))
    quality_rows = []
    best_total: dict[str, Any] | None = None
    for quality in qualities:
        stream_rows = []
        total = 0
        for raw in raw_streams:
            best: tuple[int, int] | None = None
            for lgwin in lgwin_values:
                try:
                    compressed = brotli.compress(raw, quality=quality, lgwin=lgwin)
                except brotli.error:
                    continue
                candidate = (len(compressed), lgwin)
                if best is None or candidate[0] < best[0]:
                    best = candidate
            if best is None:
                raise ValueError("Brotli recompression grid produced no candidate")
            stream_rows.append({"bytes": best[0], "lgwin": best[1]})
            total += best[0]
        row = {"quality": quality, "total_bytes": total, "streams": stream_rows}
        quality_rows.append(row)
        if best_total is None or total < int(best_total["total_bytes"]):
            best_total = row
    return {"grid": quality_rows, "best": best_total}


def _latent_filter_sweep(raw_latent: bytes, source_bytes: int) -> dict[str, Any]:
    rows = []
    for dict_size in (4096, 8192, 16384, 32768, 65536, 131072):
        for lc in range(5):
            for lp in range(3):
                if lc + lp > 4:
                    continue
                for pb in range(5):
                    filters = [
                        {
                            "id": lzma.FILTER_LZMA1,
                            "dict_size": dict_size,
                            "lc": lc,
                            "lp": lp,
                            "pb": pb,
                        }
                    ]
                    try:
                        compressed = lzma.compress(
                            raw_latent,
                            format=lzma.FORMAT_RAW,
                            filters=filters,
                        )
                    except lzma.LZMAError:
                        continue
                    rows.append(
                        {
                            "bytes": len(compressed),
                            "delta_vs_source": len(compressed) - source_bytes,
                            "dict_size": dict_size,
                            "lc": lc,
                            "lp": lp,
                            "pb": pb,
                        }
                    )
    rows.sort(key=lambda row: (int(row["bytes"]), int(row["dict_size"]), int(row["lc"]), int(row["lp"]), int(row["pb"])))
    return {"top20": rows[:20], "best": rows[0]}


def _decode_pr101_sidecar(sidecar: bytes) -> dict[str, Any]:
    schema = RankedSidecarSchema(
        n_pairs=N_PAIRS,
        n_dims=LATENT_DIM,
        deltas=PR101_DELTAS,
        huff_min_len=2,
        huff_max_len=8,
    )
    dim_bytes = 359
    rank_bytes = 5
    noop_rank_bytes = 3
    if len(sidecar) != 607:
        raise ValueError(f"expected PR101 huff-enum sidecar length 607, got {len(sidecar)}")

    length_rank = int.from_bytes(sidecar[dim_bytes : dim_bytes + rank_bytes], "little")
    lengths = _decode_huff_length_rank(length_rank, schema)

    huff_payload = sidecar[dim_bytes + rank_bytes : -noop_rank_bytes]
    decode: dict[tuple[int, int], int] = {}
    code = 0
    prev_len = 0
    for sym, length in sorted(
        ((sym, int(length)) for sym, length in enumerate(lengths) if length),
        key=lambda item: (item[1], item[0]),
    ):
        code <<= length - prev_len
        decode[(length, code)] = sym
        code += 1
        prev_len = length

    delta_valid: list[int] = []
    cur = 0
    cur_len = 0
    for byte in huff_payload:
        for shift in range(7, -1, -1):
            cur = (cur << 1) | ((byte >> shift) & 1)
            cur_len += 1
            sym = decode.get((cur_len, cur))
            if sym is not None:
                delta_valid.append(sym)
                cur = 0
                cur_len = 0
    if cur_len:
        raise ValueError("PR101 huff-enum sidecar ended mid-symbol")

    n_valid = len(delta_valid)
    noop_count = N_PAIRS - n_valid
    noop_rank = int.from_bytes(sidecar[-noop_rank_bytes:], "little")
    noop_pos = _decode_combination_colex(noop_rank, N_PAIRS, noop_count)

    dim_value = int.from_bytes(sidecar[:dim_bytes], "little")
    dims_valid = []
    for _ in range(n_valid):
        dim_value, dim = divmod(dim_value, LATENT_DIM)
        dims_valid.append(dim)
    if dim_value:
        raise ValueError("PR101 sidecar dimension mixed-radix residue is nonzero")

    delta_counts = Counter(delta_valid)
    dim_counts = Counter(dims_valid)
    delta_h_bits_per_symbol = _shannon_bits(delta_counts)
    dim_h_bits_per_symbol = _shannon_bits(dim_counts)
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
        _ceil_bytes(delta_h_bits_per_symbol * n_valid)
        + _ceil_bytes(dim_h_bits_per_symbol * n_valid)
        + _ceil_bytes(valid_mask_bits)
        + _ceil_bytes(length_rank_bits)
    )
    return {
        "bytes": len(sidecar),
        "sha256": sha256_bytes(sidecar),
        "n_valid": n_valid,
        "noop_count": noop_count,
        "noop_positions": [int(x) for x in noop_pos.tolist()],
        "component_bytes": {
            "dims": dim_bytes,
            "length_rank": rank_bytes,
            "huffman_delta_payload": len(huff_payload),
            "noop_rank": noop_rank_bytes,
        },
        "entropy_floor_bytes_estimate": entropy_floor_bytes,
        "gap_to_entropy_floor_bytes_estimate": len(sidecar) - entropy_floor_bytes,
        "delta_entropy_bits_per_symbol": delta_h_bits_per_symbol,
        "dim_entropy_bits_per_symbol": dim_h_bits_per_symbol,
        "delta_counts": {str(key): int(value) for key, value in sorted(delta_counts.items())},
        "top_dim_counts": [
            {"dim": int(dim), "count": int(count)}
            for dim, count in dim_counts.most_common(12)
        ],
    }


def _decode_fec6_selector(wrapper_payload: bytes) -> dict[str, Any]:
    if wrapper_payload[:4] != b"FP11":
        raise ValueError("expected FP11 wrapper")
    source_len = int.from_bytes(wrapper_payload[4:8], "little")
    selector_len_offset = 8 + source_len
    selector_len = int.from_bytes(wrapper_payload[selector_len_offset : selector_len_offset + 2], "little")
    selector = wrapper_payload[selector_len_offset + 2 : selector_len_offset + 2 + selector_len]
    if len(selector) != selector_len:
        raise ValueError("truncated selector payload")
    if selector[:4] != b"FEC6":
        raise ValueError("expected FEC6 selector payload")
    n_pairs = int.from_bytes(selector[4:6], "little")
    if n_pairs != N_PAIRS:
        raise ValueError(f"expected {N_PAIRS} FEC6 selector pairs, got {n_pairs}")

    decode = {bits: code for code, bits in enumerate(FEC6_CODE_BITS)}
    codes: list[int] = []
    prefix = ""
    bit_pos = 0
    payload = selector[6:]
    while len(codes) < n_pairs:
        bit = (payload[bit_pos // 8] >> (7 - (bit_pos % 8))) & 1
        bit_pos += 1
        prefix += "1" if bit else "0"
        code = decode.get(prefix)
        if code is not None:
            codes.append(code)
            prefix = ""
        elif len(prefix) > 8:
            raise ValueError("invalid FEC6 selector prefix")
    counts = Counter(codes)
    entropy_bits_per_symbol = _shannon_bits(counts)
    code_bits = sum(len(FEC6_CODE_BITS[code]) for code in codes)
    entropy_floor_bytes = _ceil_bytes(entropy_bits_per_symbol * len(codes))
    return {
        "wrapper_bytes": 10,
        "source_payload_bytes_in_wrapper": source_len,
        "selector_payload_bytes": len(selector),
        "selector_index_bytes": len(selector) - 6,
        "selector_code_bits_total": code_bits,
        "entropy_floor_bytes": entropy_floor_bytes,
        "gap_to_entropy_floor_bytes": len(selector) - entropy_floor_bytes,
        "counts": {str(key): int(value) for key, value in sorted(counts.items())},
    }


def _load_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build_profile(args: argparse.Namespace) -> dict[str, Any]:
    source_archive = Path(args.source_archive)
    fec6_archive = Path(args.fec6_archive)
    source_member, source_payload = _read_single_member_payload(source_archive)
    fec6_member, fec6_payload = _read_single_member_payload(fec6_archive)
    if source_member != fec6_member:
        raise ValueError(f"source member {source_member!r} != FEC6 member {fec6_member!r}")

    decoder = source_payload[:DECODER_BLOB_LEN]
    latent = source_payload[DECODER_BLOB_LEN : DECODER_BLOB_LEN + LATENT_BLOB_LEN]
    sidecar = source_payload[DECODER_BLOB_LEN + LATENT_BLOB_LEN :]

    decoder_streams = _split_brotli_streams(decoder)
    decoder_recompress = _best_brotli_recompress([raw for _, raw in decoder_streams])
    raw_latent = lzma.decompress(latent, format=lzma.FORMAT_RAW, filters=LATENT_LZMA_FILTERS)
    latent_sweep = _latent_filter_sweep(raw_latent, len(latent))
    sidecar_profile = _decode_pr101_sidecar(sidecar)
    selector_profile = _decode_fec6_selector(fec6_payload)

    cpu_result = _load_json_or_empty(Path(args.cpu_result))
    cuda_result = _load_json_or_empty(Path(args.cuda_result))
    cpu_score = float(cpu_result.get("score_recomputed_from_components", 0.0) or 0.0)
    byte_gap_to_target = (
        int(math.floor((cpu_score - TARGET_CPU_SCORE) * RATE_DENOMINATOR_BYTES / 25.0) + 1)
        if cpu_score > TARGET_CPU_SCORE
        else 0
    )

    best_decoder_delta = int(decoder_recompress["best"]["total_bytes"]) - len(decoder)
    source_payload_lossless_realistic_delta = (
        min(best_decoder_delta, 0)
        + min(int(latent_sweep["best"]["delta_vs_source"]), 0)
        + min(int(sidecar_profile["gap_to_entropy_floor_bytes_estimate"]) * -1, 0)
    )
    wrapper_hardcode_saving = int(selector_profile["wrapper_bytes"])
    realistic_same_frame_saving_upper_bound = (
        max(-source_payload_lossless_realistic_delta, 0) + wrapper_hardcode_saving
    )

    return {
        "schema": "pr101_fec6_byte_escape_profile.v1",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "axis_notes": {
            "cpu": "[contest-CPU] exact eval is legitimate but not promotion-eligible",
            "cuda": "[contest-CUDA] paired eval is much worse; no CPU->CUDA conversion",
        },
        "inputs": {
            "source_archive": _rel(source_archive),
            "source_archive_bytes": source_archive.stat().st_size,
            "source_archive_sha256": sha256_bytes(source_archive.read_bytes()),
            "fec6_archive": _rel(fec6_archive),
            "fec6_archive_bytes": fec6_archive.stat().st_size,
            "fec6_archive_sha256": sha256_bytes(fec6_archive.read_bytes()),
            "member": source_member,
        },
        "exact_results": {
            "cpu_score_recomputed": cpu_score,
            "cpu_result": _rel(Path(args.cpu_result)),
            "cuda_score_recomputed": cuda_result.get("score_recomputed_from_components"),
            "cuda_result": _rel(Path(args.cuda_result)),
            "byte_gap_to_target_0_192_cpu": byte_gap_to_target,
        },
        "source_payload_sections": {
            "decoder": {
                "bytes": len(decoder),
                "sha256": sha256_bytes(decoder),
                "stream_count": len(decoder_streams),
                "compressed_stream_bytes": [len(comp) for comp, _ in decoder_streams],
                "raw_stream_bytes": [len(raw) for _, raw in decoder_streams],
                "best_recompress": decoder_recompress["best"],
                "best_recompress_delta_bytes": best_decoder_delta,
                "recompress_grid": decoder_recompress["grid"],
            },
            "latent": {
                "bytes": len(latent),
                "sha256": sha256_bytes(latent),
                "raw_bytes": len(raw_latent),
                "raw_sha256": sha256_bytes(raw_latent),
                "best_filter_sweep": latent_sweep["best"],
                "top20_filter_sweep": latent_sweep["top20"],
            },
            "sidecar": sidecar_profile,
        },
        "fec6_selector": selector_profile,
        "conclusion": {
            "selector_entropy_saturated": selector_profile["gap_to_entropy_floor_bytes"] < 8,
            "source_payload_lossless_realistic_saving_upper_bound_bytes": realistic_same_frame_saving_upper_bound,
            "cannot_reach_0_192_by_same_frame_bytes_only": (
                realistic_same_frame_saving_upper_bound < byte_gap_to_target
            ),
            "next_required_mechanism": (
                "component-changing CUDA-in-loop selector/waterfill or a new trained substrate; "
                "byte-only selector/source recoding is below the required gap"
            ),
        },
    }


def render_markdown(profile: Mapping[str, Any]) -> str:
    exact = profile["exact_results"]
    sections = profile["source_payload_sections"]
    selector = profile["fec6_selector"]
    conclusion = profile["conclusion"]
    lines = [
        "# PR101 FEC6 Byte Escape Profile",
        "",
        "- score_claim: `false`",
        "- dispatch_attempted: `false`",
        "- ready_for_exact_eval_dispatch: `false`",
        f"- FEC6 [contest-CPU] score: `{exact['cpu_score_recomputed']}`",
        f"- paired [contest-CUDA] score: `{exact['cuda_score_recomputed']}`",
        f"- byte gap to `<0.192` on [contest-CPU]: `{exact['byte_gap_to_target_0_192_cpu']}`",
        "",
        "## Byte-Only Findings",
        "",
        "| surface | current bytes | best/floor bytes | realistic saving | verdict |",
        "|---|---:|---:|---:|---|",
    ]
    decoder = sections["decoder"]
    latent = sections["latent"]
    sidecar = sections["sidecar"]
    lines.extend(
        [
            "| PR101 decoder Brotli streams | {cur} | {best} | {saving} | bounded recompress only |".format(
                cur=decoder["bytes"],
                best=decoder["best_recompress"]["total_bytes"],
                saving=max(0, -int(decoder["best_recompress_delta_bytes"])),
            ),
            "| PR101 latent raw-LZMA | {cur} | {best} | {saving} | filter sweep saturated |".format(
                cur=latent["bytes"],
                best=latent["best_filter_sweep"]["bytes"],
                saving=max(0, -int(latent["best_filter_sweep"]["delta_vs_source"])),
            ),
            "| PR101 latent sidecar | {cur} | {best} | {saving} | near entropy floor |".format(
                cur=sidecar["bytes"],
                best=sidecar["entropy_floor_bytes_estimate"],
                saving=sidecar["gap_to_entropy_floor_bytes_estimate"],
            ),
            "| FEC6 selector payload | {cur} | {best} | {saving} | selector entropy saturated |".format(
                cur=selector["selector_payload_bytes"],
                best=selector["entropy_floor_bytes"],
                saving=selector["gap_to_entropy_floor_bytes"],
            ),
            "| FP11 wrapper | 10 | 0 | 10 | hardcode-only, insufficient |",
            "",
            "## Conclusion",
            "",
            f"- same-frame realistic byte-saving upper bound: `{conclusion['source_payload_lossless_realistic_saving_upper_bound_bytes']}` bytes",
            f"- cannot reach `<0.192` by same-frame bytes only: `{str(conclusion['cannot_reach_0_192_by_same_frame_bytes_only']).lower()}`",
            f"- next required mechanism: {conclusion['next_required_mechanism']}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--fec6-archive", type=Path, default=DEFAULT_FEC6_ARCHIVE)
    parser.add_argument("--cpu-result", type=Path, default=DEFAULT_CPU_RESULT)
    parser.add_argument("--cuda-result", type=Path, default=DEFAULT_CUDA_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profile = build_profile(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "profile.json", profile)
    (output_dir / "profile.md").write_text(render_markdown(profile) + "\n")
    print(output_dir / "profile.json")
    print(output_dir / "profile.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
