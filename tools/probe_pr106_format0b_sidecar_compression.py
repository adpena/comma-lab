#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe PR106 format-0x0B sidecar compression headroom.

This is a local byte-custody probe, not a score claim. It inspects the active
PR106 PacketIR format-0x0B archive, decodes the 525-byte fixed PR101 sidecar,
and reports which savings are already byte-closed versus which require a new
runtime format before exact eval.
"""

from __future__ import annotations

import argparse
import bz2
import datetime as dt
import lzma
import math
import sys
import zlib
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import brotli  # type: ignore[import-not-found]
import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.packet_compiler.pr106_sidecar_packet import (  # noqa: E402
    PR106_PR101_FIXED_META_DIM_BYTES,
    PR106_PR101_FIXED_META_NOOP_COUNT,
    PR106_PR101_FIXED_META_NOOP_RANK_BYTE,
    PR106_PR101_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES,
    PR106_PR101_FIXED_META_RANK_BYTE,
    PR106_PR101_RANKED_SCHEMA,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED,
    decode_pr101_ranked_sidecar_payload_to_dim_delta,
    decode_pr106_sidecar_packet_dim_delta,
    parse_pr106_sidecar_packet,
    read_single_stored_member_archive,
)
from tac.repo_io import json_text, repo_relative, sha256_bytes  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402

CONTEST_ORIGINAL_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / CONTEST_ORIGINAL_BYTES
DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/"
    / "candidates/pr101_hdm9_hlm3_magicless_fixed_meta_noop_rank_elided_sidecar_format_0x0b.archive.zip"
)
DEFAULT_JSON_OUT = (
    REPO_ROOT / ".omx/research/pr106_format0b_sidecar_compression_probe_20260515_codex.json"
)
DEFAULT_MD_OUT = (
    REPO_ROOT / ".omx/research/pr106_format0b_sidecar_compression_probe_20260515_codex.md"
)


class ProbeError(ValueError):
    """Raised when the format-0x0B probe cannot classify an input."""


def byte_entropy(payload: bytes) -> float:
    """Return empirical byte entropy in bits per byte."""

    if not payload:
        return 0.0
    counts = Counter(payload)
    total = len(payload)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def symbol_entropy(values: Iterable[int]) -> float:
    """Return empirical symbol entropy in bits per symbol."""

    values = [int(value) for value in values]
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def value_distribution(values: Iterable[int]) -> list[dict[str, int | float]]:
    """Return a stable value/count/frequency table."""

    values = [int(value) for value in values]
    total = len(values)
    counts = Counter(values)
    return [
        {
            "value": value,
            "count": count,
            "frequency": count / total if total else 0.0,
        }
        for value, count in sorted(counts.items())
    ]


def fixed_meta_chunks(sidecar: bytes) -> tuple[bytes, bytes, bytes]:
    """Split a format-0x0B sidecar into dim / fixed length-rank / Huffman bytes."""

    expected = PR106_PR101_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES
    if len(sidecar) != expected:
        raise ProbeError(f"format-0x0B sidecar must be {expected} bytes; got {len(sidecar)}")
    dim = sidecar[:PR106_PR101_FIXED_META_DIM_BYTES]
    rank = PR106_PR101_FIXED_META_RANK_BYTE
    huff = sidecar[PR106_PR101_FIXED_META_DIM_BYTES:]
    return dim, rank, huff


def base_radix_dim_value(dims: np.ndarray) -> int:
    """Pack dims as the PR101 mixed-radix integer, without width padding."""

    value = 0
    for dim in reversed(np.asarray(dims, dtype=np.int64).tolist()):
        value = value * int(PR106_PR101_RANKED_SCHEMA.n_dims) + int(dim)
    return value


def exact_radix_dim_bytes(n_symbols: int, base: int) -> int:
    """Return minimal bytes needed for a base-radix sequence of fixed length."""

    if n_symbols <= 0:
        return 1
    max_value = pow(int(base), int(n_symbols)) - 1
    return max(1, (max_value.bit_length() + 7) // 8)


def verify_exact_radix_candidate(
    *,
    dims: np.ndarray,
    deltas: np.ndarray,
    current_rank: bytes,
    current_huff: bytes,
) -> dict[str, Any]:
    """Build and decode the tight base-radix sidecar body in memory."""

    dim_bytes = exact_radix_dim_bytes(
        int(PR106_PR101_RANKED_SCHEMA.n_pairs),
        int(PR106_PR101_RANKED_SCHEMA.n_dims),
    )
    dim_blob = base_radix_dim_value(dims).to_bytes(dim_bytes, "little")
    expanded_payload = (
        dim_blob + current_rank + current_huff + PR106_PR101_FIXED_META_NOOP_RANK_BYTE
    )
    framing_meta = (
        int(PR106_PR101_FIXED_META_NOOP_COUNT).to_bytes(2, "little")
        + int(dim_bytes).to_bytes(2, "little")
        + len(current_rank).to_bytes(1, "little")
        + len(PR106_PR101_FIXED_META_NOOP_RANK_BYTE).to_bytes(1, "little")
    )
    decoded_dims, decoded_deltas = decode_pr101_ranked_sidecar_payload_to_dim_delta(
        expanded_payload,
        framing_meta,
        schema=PR106_PR101_RANKED_SCHEMA,
    )
    lossless = bool(np.array_equal(decoded_dims, dims) and np.array_equal(decoded_deltas, deltas))
    if not lossless:
        raise ProbeError("exact-radix dim candidate failed sidecar decode equivalence")
    candidate_payload_bytes = dim_bytes + len(current_huff)
    current_payload_bytes = PR106_PR101_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES
    saved = current_payload_bytes - candidate_payload_bytes
    return {
        "name": "format0c_exact_base28_dim_width",
        "lossless_sidecar_equivalence": True,
        "runtime_supported_now": True,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "current_payload_bytes": current_payload_bytes,
        "candidate_payload_bytes": candidate_payload_bytes,
        "candidate_dim_bytes": dim_bytes,
        "candidate_charged_rank_bytes": 0,
        "candidate_fixed_runtime_rank_bytes": len(current_rank),
        "candidate_huffman_bytes": len(current_huff),
        "byte_savings_if_runtime_format_lands": saved,
        "rate_score_delta_if_components_equal": -saved * RATE_SCORE_PER_BYTE,
        "requires_new_runtime_format": False,
        "required_runtime_change": (
            "PacketIR format0C uses fixed dim_bytes="
            f"{dim_bytes} instead of {PR106_PR101_FIXED_META_DIM_BYTES}."
        ),
    }


def generic_compressor_rows(label: str, payload: bytes) -> list[dict[str, Any]]:
    """Run a small deterministic generic-compressor grid on one byte payload."""

    rows: list[dict[str, Any]] = []
    for quality in range(12):
        encoded = brotli.compress(payload, quality=quality)
        rows.append(
            {
                "source": label,
                "codec": "brotli",
                "params": {"quality": quality},
                "encoded_bytes": len(encoded),
                "byte_delta_vs_source": len(encoded) - len(payload),
                "encoded_sha256": sha256_bytes(encoded),
            }
        )
    for level in range(10):
        encoded = zlib.compress(payload, level)
        rows.append(
            {
                "source": label,
                "codec": "zlib",
                "params": {"level": level},
                "encoded_bytes": len(encoded),
                "byte_delta_vs_source": len(encoded) - len(payload),
                "encoded_sha256": sha256_bytes(encoded),
            }
        )
    for compresslevel in range(1, 10):
        encoded = bz2.compress(payload, compresslevel=compresslevel)
        rows.append(
            {
                "source": label,
                "codec": "bz2",
                "params": {"compresslevel": compresslevel},
                "encoded_bytes": len(encoded),
                "byte_delta_vs_source": len(encoded) - len(payload),
                "encoded_sha256": sha256_bytes(encoded),
            }
        )
    lzma_presets = range(10)
    for preset in lzma_presets:
        encoded = lzma.compress(payload, format=lzma.FORMAT_XZ, preset=preset)
        rows.append(
            {
                "source": label,
                "codec": "lzma_xz",
                "params": {"preset": preset},
                "encoded_bytes": len(encoded),
                "byte_delta_vs_source": len(encoded) - len(payload),
                "encoded_sha256": sha256_bytes(encoded),
            }
        )
    return sorted(rows, key=lambda row: (row["encoded_bytes"], row["codec"], str(row["params"])))


def best_rows_by_source(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the smallest generic-compressor row for each source label."""

    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        source = str(row["source"])
        if source not in best or int(row["encoded_bytes"]) < int(best[source]["encoded_bytes"]):
            best[source] = row
    return [best[source] for source in sorted(best)]


def build_probe(source_archive: Path) -> dict[str, Any]:
    """Build the format-0x0B sidecar compression probe manifest."""

    archive_bytes = source_archive.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    packet = parse_pr106_sidecar_packet(member.payload)
    expected_format = (
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED
    )
    if packet.format_id != expected_format:
        raise ProbeError(
            f"expected format_id=0x{expected_format:02X}; got 0x{packet.format_id:02X}"
        )

    dims, deltas = decode_pr106_sidecar_packet_dim_delta(packet)
    dim_blob, rank_blob, huff_blob = fixed_meta_chunks(packet.sidecar_payload)
    exact_radix = verify_exact_radix_candidate(
        dims=dims,
        deltas=deltas,
        current_rank=rank_blob,
        current_huff=huff_blob,
    )
    exact_payload = (
        base_radix_dim_value(dims).to_bytes(exact_radix["candidate_dim_bytes"], "little")
        + huff_blob
    )
    compressor_rows = [
        *generic_compressor_rows("current_525b_sidecar_payload", packet.sidecar_payload),
        *generic_compressor_rows("current_375b_dim_container", dim_blob),
        *generic_compressor_rows("exact_radix_511b_candidate_payload", exact_payload),
    ]
    current_payload_bytes = len(packet.sidecar_payload)
    byte_positive_generic = [
        row for row in compressor_rows if int(row["byte_delta_vs_source"]) < 0
    ]
    realized_runtime_supported_savings = exact_radix[
        "byte_savings_if_runtime_format_lands"
    ]
    blockers = [
        "planning_probe_only_no_archive_emitted",
        "requires_full_frame_parity_before_exact_eval",
        "requires_lane_dispatch_claim_before_exact_cuda",
        "generic_compressor_rows_do_not_have_runtime_decoder",
    ]
    verdict = (
        "small_exact_radix_runtime_format_candidate_found"
        if exact_radix["byte_savings_if_runtime_format_lands"] > 0
        else "format0b_sidecar_currently_saturated"
    )
    delta_counts = Counter(int(value) for value in deltas.tolist())
    lengths = [2 for _ in deltas.tolist()]
    return {
        "schema": "pr106_format0b_sidecar_compression_probe_v1",
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "tool": "tools/probe_pr106_format0b_sidecar_compression.py",
        "planning_only": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "source": {
            "archive_path": repo_relative(source_archive, REPO_ROOT),
            "archive_bytes": source_archive.stat().st_size,
            "archive_sha256": sha256_bytes(archive_bytes),
            "member_name": member.name,
            "member_bytes": len(member.payload),
            "member_sha256": sha256_bytes(member.payload),
            "format_id": "0x0B",
        },
        "sidecar": {
            "payload_bytes": current_payload_bytes,
            "payload_sha256": sha256_bytes(packet.sidecar_payload),
            "payload_entropy_bits_per_byte": byte_entropy(packet.sidecar_payload),
            "fixed_meta_noop_count": PR106_PR101_FIXED_META_NOOP_COUNT,
            "fixed_meta_dim_bytes": PR106_PR101_FIXED_META_DIM_BYTES,
            "charged_rank_bytes": 0,
            "fixed_runtime_rank_bytes": len(rank_blob),
            "huffman_bytes": len(huff_blob),
            "dim_container_sha256": sha256_bytes(dim_blob),
            "rank_byte_hex": rank_blob.hex(),
            "huffman_sha256": sha256_bytes(huff_blob),
        },
        "semantic_stats": {
            "pair_count": int(PR106_PR101_RANKED_SCHEMA.n_pairs),
            "dim_count": int(PR106_PR101_RANKED_SCHEMA.n_dims),
            "delta_vocab": [int(value) for value in PR106_PR101_RANKED_SCHEMA.deltas],
            "noop_count": int(np.count_nonzero(dims == PR106_PR101_RANKED_SCHEMA.no_op_sentinel)),
            "dim_entropy_bits_per_symbol": symbol_entropy(dims.tolist()),
            "delta_entropy_bits_per_symbol": symbol_entropy(deltas.tolist()),
            "ideal_dim_entropy_bytes_no_model_cost": math.ceil(
                len(dims) * symbol_entropy(dims.tolist()) / 8
            ),
            "ideal_delta_entropy_bytes_no_model_cost": math.ceil(
                len(deltas) * symbol_entropy(deltas.tolist()) / 8
            ),
            "dim_distribution": value_distribution(dims.tolist()),
            "delta_distribution": value_distribution(deltas.tolist()),
            "delta_code_length_assumption": {
                "current_uniform_length_bits": sorted(set(lengths)),
                "current_huffman_payload_bytes": len(huff_blob),
                "delta_counts": {str(key): int(value) for key, value in sorted(delta_counts.items())},
            },
        },
        "candidates": {
            "exact_radix_runtime_format": exact_radix,
            "generic_compressor_best_by_source": best_rows_by_source(compressor_rows),
            "generic_compressor_rate_positive_count": len(byte_positive_generic),
            "generic_compressor_best_overall": compressor_rows[0],
        },
        "decision": {
            "verdict": verdict,
            "realized_runtime_supported_byte_savings": realized_runtime_supported_savings,
            "best_unimplemented_byte_savings": 0,
            "best_unimplemented_rate_score_delta_if_components_equal": 0.0,
            "materiality": (
                "below_cpu_sub_0_192_gap_and_not_cuda_frontier_material"
                if exact_radix["byte_savings_if_runtime_format_lands"] < 78
                else "material_rate_delta_if_runtime_format_lands"
            ),
            "dispatch_blockers": blockers,
            "recommended_next": (
                "Emit a byte-closed format0C archive and run same-runtime parity before "
                "any exact-eval dispatch; prioritize decoder/latent stream transforms "
                "because the sidecar has only 14 byte-closed bytes of headroom."
            ),
        },
        "summary": {
            "current_sidecar_payload_bytes": current_payload_bytes,
            "exact_radix_candidate_payload_bytes": exact_radix["candidate_payload_bytes"],
            "identified_sidecar_savings_requires_runtime": 0,
            "format0c_runtime_supported_savings": realized_runtime_supported_savings,
            "realized_score_claim": False,
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a compact operator-facing report."""

    source = payload["source"]
    sidecar = payload["sidecar"]
    stats = payload["semantic_stats"]
    exact = payload["candidates"]["exact_radix_runtime_format"]
    decision = payload["decision"]
    best_generic = payload["candidates"]["generic_compressor_best_by_source"]
    lines = [
        "# PR106 Format 0x0B Sidecar Compression Probe",
        "",
        f"- score_claim: `{str(payload['score_claim']).lower()}`",
        f"- dispatch_attempted: `{str(payload['dispatch_attempted']).lower()}`",
        f"- ready_for_exact_eval_dispatch: `{str(payload['ready_for_exact_eval_dispatch']).lower()}`",
        f"- verdict: `{decision['verdict']}`",
        "",
        "## Source",
        "",
        f"- archive: `{source['archive_path']}`",
        f"- archive bytes: `{source['archive_bytes']}`",
        f"- archive sha256: `{source['archive_sha256']}`",
        f"- member: `{source['member_name']}`",
        "",
        "## Sidecar Headroom",
        "",
        f"- current sidecar payload: `{sidecar['payload_bytes']}` bytes",
        f"- current dim container: `{sidecar['fixed_meta_dim_bytes']}` bytes",
        f"- exact base-28 dim container: `{exact['candidate_dim_bytes']}` bytes",
        f"- format0C candidate payload: `{exact['candidate_payload_bytes']}` bytes",
        f"- runtime-supported format0C savings: `{decision['realized_runtime_supported_byte_savings']}` bytes",
        f"- rate-only score delta if components equal: `{exact['rate_score_delta_if_components_equal']:.12f}`",
        "",
        "## Semantic Stats",
        "",
        f"- dim entropy: `{stats['dim_entropy_bits_per_symbol']:.6f}` bits/symbol",
        f"- delta entropy: `{stats['delta_entropy_bits_per_symbol']:.6f}` bits/symbol",
        f"- ideal dim entropy bytes, no model cost: `{stats['ideal_dim_entropy_bytes_no_model_cost']}`",
        f"- ideal delta entropy bytes, no model cost: `{stats['ideal_delta_entropy_bytes_no_model_cost']}`",
        "",
        "## Generic Compressor Best Rows",
        "",
        "| source | codec | encoded bytes | delta vs source |",
        "|---|---|---:|---:|",
    ]
    for row in best_generic:
        lines.append(
            f"| `{row['source']}` | `{row['codec']}` | {row['encoded_bytes']} | {row['byte_delta_vs_source']} |"
        )
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- `{blocker}`" for blocker in decision["dispatch_blockers"])
    lines.extend(["", "## Recommendation", "", decision["recommended_next"], ""])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    try:
        payload = build_probe(args.source_archive)
    except (OSError, ProbeError, ValueError) as exc:
        print(f"FATAL: PR106 format-0x0B sidecar probe failed: {exc}", file=sys.stderr)
        return 2
    payload = attach_tool_run_manifest(
        payload,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=[args.source_archive],
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json_text(payload), encoding="utf-8")
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
