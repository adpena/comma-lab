#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an A5 per-channel q-bit schedule by exact discrete DP.

The schedule is CPU-prep only. It does not run inflate, does not invoke the
scorer, and does not claim score. It solves the separable latent-domain proxy:
choose one q-bit width per latent dimension, minimizing reconstructed latent
MSE under a total q-sum budget. This is a channel-aware alternative to A5's
older per-pair scalar schedules.
"""

from __future__ import annotations

import argparse
import json
import lzma
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.codec.frame_conditional_bit_budget import (  # noqa: E402
    FRAME_CONDITIONAL_Q_BITS_ENCODING_CHANNEL_RAW3,
    apply_frame_conditional_channel_q_bits,
    pack_frame_conditional_channel_latent_codes,
    pack_frame_conditional_channel_q_bits,
)
from tac.repo_io import json_text, repo_relative, sha256_bytes, sha256_file  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402

SCHEMA = "pr101_a5_channel_qbits_schedule.v1"
DEFAULT_SOURCE_ARCHIVE = Path(
    "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
)
PR101_DECODER_BLOB_LEN = 162_164
PR101_LATENT_BLOB_LEN = 15_387
PR101_N_PAIRS = 600
PR101_LATENT_DIM = 28
PR101_LATENT_LZMA_FILTERS = [
    {"id": lzma.FILTER_LZMA1, "dict_size": 4096, "lc": 3, "lp": 0, "pb": 0}
]


class A5ChannelQBitsScheduleError(ValueError):
    """Raised when a channel q-bit schedule cannot be built safely."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--candidate-id", default="a5_channel_qbits_dp")
    parser.add_argument(
        "--target-qsum",
        type=int,
        required=True,
        help=(
            "Total sum of the 28 per-channel q-bit widths. PR101 all-int8 is "
            "224; qsum=200 is roughly a 10.7 percent latent-wire bit cut."
        ),
    )
    return parser.parse_args(argv)


def build_schedule(
    *,
    source_archive: Path,
    target_qsum: int,
    candidate_id: str = "a5_channel_qbits_dp",
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    source_archive = _resolve(source_archive, repo_root)
    if not source_archive.is_file():
        raise A5ChannelQBitsScheduleError(f"source archive not found: {source_archive}")
    member_payload = _read_single_member(source_archive)
    latent_meta_blob, q_pair_first, scales = _extract_latents(member_payload)
    n_pairs, latent_dim = q_pair_first.shape
    target_qsum = _target_qsum_int(target_qsum, latent_dim=latent_dim)

    loss_table = _channel_loss_table(q_pair_first, scales)
    q_bits, proxy_loss = _solve_exact_dp(loss_table, target_qsum=target_qsum)
    sideinfo = pack_frame_conditional_channel_q_bits(q_bits)
    latent_wire_payload = pack_frame_conditional_channel_latent_codes(
        q_pair_first,
        q_bits,
    )
    truncated = apply_frame_conditional_channel_q_bits(q_pair_first, q_bits)
    all8_sideinfo = pack_frame_conditional_channel_q_bits(
        np.full(latent_dim, 8, dtype=np.uint8)
    )
    all8_latent_wire_payload = pack_frame_conditional_channel_latent_codes(
        q_pair_first,
        np.full(latent_dim, 8, dtype=np.uint8),
    )
    replacement_bytes = len(latent_meta_blob) + len(sideinfo) + len(latent_wire_payload)
    all8_replacement_bytes = (
        len(latent_meta_blob) + len(all8_sideinfo) + len(all8_latent_wire_payload)
    )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "candidate_id": candidate_id,
        "score_claim": False,
        "dispatch_attempted": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "[CPU-prep channel q-bit schedule; no eval]",
        "evidence_semantics": (
            "exact discrete dynamic program over per-channel q-bit widths; "
            "latent-domain proxy only, no score claim"
        ),
        "source_artifacts": {
            "source_archive": {
                "path": repo_relative(source_archive, repo_root),
                "bytes": source_archive.stat().st_size,
                "sha256": sha256_file(source_archive),
            }
        },
        "n_pairs": int(n_pairs),
        "latent_dim": int(latent_dim),
        "target_qsum": int(target_qsum),
        "all8_qsum": int(latent_dim * 8),
        "qsum_saved_vs_all8": int(latent_dim * 8 - target_qsum),
        "per_channel_q_bits": [int(value) for value in q_bits.tolist()],
        "q_bits_summary": _q_bits_summary(q_bits),
        "q_bits_sideinfo": {
            "encoding": FRAME_CONDITIONAL_Q_BITS_ENCODING_CHANNEL_RAW3,
            "bytes": len(sideinfo),
            "sha256": sha256_bytes(sideinfo),
        },
        "latent_wire_payload": {
            "bytes": len(latent_wire_payload),
            "sha256": sha256_bytes(latent_wire_payload),
            "all8_bytes": len(all8_latent_wire_payload),
            "saved_vs_all8_bytes": len(all8_latent_wire_payload)
            - len(latent_wire_payload),
        },
        "latent_replacement_bytes": {
            "original_lzma_blob_bytes": PR101_LATENT_BLOB_LEN,
            "channel_qbits_meta_plus_sideinfo_plus_wire_bytes": replacement_bytes,
            "channel_all8_meta_plus_sideinfo_plus_wire_bytes": all8_replacement_bytes,
            "estimated_delta_vs_original_lzma_blob_bytes": replacement_bytes
            - PR101_LATENT_BLOB_LEN,
        },
        "proxy_objective": {
            "name": "latent_scale_weighted_truncation_mse",
            "optimal_for_target_qsum": True,
            "target_qsum": int(target_qsum),
            "proxy_loss": float(proxy_loss),
            "proxy_loss_per_pair_channel_mean": float(proxy_loss / latent_dim),
            "source_q_codes_sha256": sha256_bytes(q_pair_first.tobytes()),
            "truncated_q_codes_sha256": sha256_bytes(truncated.tobytes()),
            "score_affecting_payload_changed": bool(
                not np.array_equal(q_pair_first, truncated)
            ),
        },
        "channel_table": _channel_table(loss_table, q_bits, scales),
        "dispatch_blockers": [
            "cpu_prep_schedule_only_no_runtime_packet",
            "latent_domain_proxy_not_score_domain",
            "missing_advisory_eval",
            "missing_exact_contest_cuda",
            "missing_exact_contest_cpu",
        ],
        "reactivation_criteria": [
            "Build a runtime packet with --q-bits-sideinfo-encoding channel_raw3.",
            "Run macOS CPU advisory only as a collapse screen after lane claim.",
            "Promote only after paired contest-CUDA and Linux x86_64 contest-CPU.",
        ],
    }
    payload["manifest_sha256_excluding_self"] = sha256_bytes(
        json_text(payload).encode("utf-8")
    )
    return payload


def _solve_exact_dp(loss_table: np.ndarray, *, target_qsum: int) -> tuple[np.ndarray, float]:
    latent_dim = loss_table.shape[0]
    inf = float("inf")
    dp = np.full((latent_dim + 1, target_qsum + 1), inf, dtype=np.float64)
    choice = np.zeros((latent_dim + 1, target_qsum + 1), dtype=np.uint8)
    dp[0, 0] = 0.0
    for dim in range(latent_dim):
        for used in range(target_qsum + 1):
            base = float(dp[dim, used])
            if not np.isfinite(base):
                continue
            for bits in range(1, 9):
                nxt = used + bits
                if nxt > target_qsum:
                    continue
                candidate = base + float(loss_table[dim, bits])
                if candidate < dp[dim + 1, nxt]:
                    dp[dim + 1, nxt] = candidate
                    choice[dim + 1, nxt] = bits
    if not np.isfinite(dp[latent_dim, target_qsum]):
        raise A5ChannelQBitsScheduleError(
            f"no channel q-bit solution for target_qsum={target_qsum}"
        )
    out = np.empty(latent_dim, dtype=np.uint8)
    remaining = target_qsum
    for dim in range(latent_dim, 0, -1):
        bits = int(choice[dim, remaining])
        if bits <= 0:
            raise A5ChannelQBitsScheduleError("DP backtrack failed")
        out[dim - 1] = bits
        remaining -= bits
    return out, float(dp[latent_dim, target_qsum])


def _channel_loss_table(q_pair_first: np.ndarray, scales: np.ndarray) -> np.ndarray:
    q = np.asarray(q_pair_first, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.float64)
    if q.ndim != 2:
        raise A5ChannelQBitsScheduleError(f"q_pair_first must be 2-D, got {q.shape}")
    if scales.shape != (q.shape[1],):
        raise A5ChannelQBitsScheduleError(
            f"scales shape {scales.shape} != latent_dim {q.shape[1]}"
        )
    table = np.full((q.shape[1], 9), np.inf, dtype=np.float64)
    for dim in range(q.shape[1]):
        source = q[:, dim].astype(np.uint16)
        for bits in range(1, 9):
            shift = 8 - bits
            truncated = ((source >> shift) << shift).astype(np.uint16)
            delta = (source.astype(np.float64) - truncated.astype(np.float64)) * scales[dim]
            table[dim, bits] = float(np.mean(delta * delta))
    return table


def _extract_latents(member_payload: bytes) -> tuple[bytes, np.ndarray, np.ndarray]:
    if len(member_payload) < PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN:
        raise A5ChannelQBitsScheduleError("PR101 member payload too short")
    latent_blob = member_payload[
        PR101_DECODER_BLOB_LEN : PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN
    ]
    raw = lzma.decompress(
        latent_blob,
        format=lzma.FORMAT_RAW,
        filters=PR101_LATENT_LZMA_FILTERS,
    )
    meta_len = PR101_LATENT_DIM * 4
    q_len = PR101_N_PAIRS * PR101_LATENT_DIM
    if len(raw) != meta_len + q_len:
        raise A5ChannelQBitsScheduleError(
            f"decoded latent payload length {len(raw)} != expected {meta_len + q_len}"
        )
    latent_meta_blob = raw[:meta_len]
    scales = np.frombuffer(
        latent_meta_blob[PR101_LATENT_DIM * 2 :],
        dtype=np.float16,
    ).astype(np.float64)
    stored = np.frombuffer(raw[meta_len:], dtype=np.uint8).reshape(
        PR101_LATENT_DIM,
        PR101_N_PAIRS,
    )
    q_ordered = stored.copy()
    q_ordered[:, 1:] = (
        np.cumsum(
            ((stored[:, 1:].astype(np.int16) - 128) & 255),
            axis=1,
            dtype=np.uint16,
        ).astype(np.uint8)
        + stored[:, :1]
    )
    return latent_meta_blob, q_ordered.T.copy(), scales


def _read_single_member(path: Path) -> bytes:
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            if names != ["x"]:
                raise A5ChannelQBitsScheduleError(
                    f"expected single ZIP member ['x'], got {names}"
                )
            return zf.read("x")
    except zipfile.BadZipFile as exc:
        raise A5ChannelQBitsScheduleError(f"bad ZIP archive: {path}") from exc


def _target_qsum_int(value: int, *, latent_dim: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise A5ChannelQBitsScheduleError("target_qsum must be an integer")
    lo = latent_dim
    hi = latent_dim * 8
    if value < lo or value > hi:
        raise A5ChannelQBitsScheduleError(
            f"target_qsum must be in [{lo}, {hi}], got {value}"
        )
    return value


def _q_bits_summary(q_bits: np.ndarray) -> dict[str, Any]:
    values, counts = np.unique(q_bits, return_counts=True)
    return {
        "min": int(q_bits.min()),
        "max": int(q_bits.max()),
        "mean": float(q_bits.mean()),
        "sha256": sha256_bytes(q_bits.tobytes()),
        "unique_counts": {
            str(int(value)): int(count)
            for value, count in zip(values, counts, strict=True)
        },
    }


def _channel_table(
    loss_table: np.ndarray, q_bits: np.ndarray, scales: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dim, bits in enumerate(q_bits):
        rows.append(
            {
                "channel": int(dim),
                "q_bits": int(bits),
                "scale": float(scales[dim]),
                "proxy_loss_at_selected_q": float(loss_table[dim, int(bits)]),
                "proxy_loss_at_q8": float(loss_table[dim, 8]),
            }
        )
    return rows


def _resolve(path: Path, repo_root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    try:
        payload = build_schedule(
            source_archive=args.source_archive,
            target_qsum=args.target_qsum,
            candidate_id=args.candidate_id,
            repo_root=REPO_ROOT,
        )
        payload = attach_tool_run_manifest(
            payload,
            tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
            argv=raw_argv,
            input_paths=[args.source_archive],
            repo_root=REPO_ROOT,
            output_path=args.json_out,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"FATAL: A5 channel q-bit schedule rejected: {exc}", file=sys.stderr)
        return 2
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json_text(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
