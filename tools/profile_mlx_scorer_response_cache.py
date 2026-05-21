#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile MLX scorer-response throughput on fixed scorer-input cache windows."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_scorer_response import build_mlx_scorer_response_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache-dir", required=True, type=Path)
    parser.add_argument("--candidate-cache-dir", required=True, type=Path)
    parser.add_argument("--archive-size-bytes", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument("--batch-pairs", default="1,2,4")
    parser.add_argument("--devices", default="cpu")
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--allow-gpu-research-signal",
        action="store_true",
        help=(
            "Permit gpu entries in --devices as local MLX prescreen/profiling "
            "signal only. GPU rows remain non-authoritative and require CPU "
            "transfer checks before they can affect exact-eval dispatch choices."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    profile = build_profile_payload(
        reference_cache_dir=args.reference_cache_dir,
        candidate_cache_dir=args.candidate_cache_dir,
        archive_size_bytes=args.archive_size_bytes,
        repo_root=args.repo_root,
        batch_pairs_values=parse_positive_int_csv(args.batch_pairs, flag_name="--batch-pairs"),
        device_values=parse_device_csv(args.devices),
        start_pair=args.start_pair,
        max_pairs=args.max_pairs,
        repeat=args.repeat,
        allow_gpu_research_signal=args.allow_gpu_research_signal,
    )
    write_profile_payload(profile, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(profile["rows"]),
                "best": profile["best"],
                "score_claim": profile["score_claim"],
            },
            sort_keys=True,
        )
    )
    return 0


def build_profile_payload(
    *,
    reference_cache_dir: str | Path,
    candidate_cache_dir: str | Path,
    archive_size_bytes: int,
    repo_root: str | Path,
    batch_pairs_values: list[int],
    device_values: list[str],
    start_pair: int,
    max_pairs: int,
    repeat: int = 1,
    allow_gpu_research_signal: bool = False,
) -> dict[str, Any]:
    if int(max_pairs) < 1:
        raise ValueError(f"max_pairs must be >= 1, got {max_pairs}")
    if int(start_pair) < 0:
        raise ValueError(f"start_pair must be >= 0, got {start_pair}")
    if int(repeat) < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    if "gpu" in device_values and not allow_gpu_research_signal:
        raise ValueError(
            "--devices gpu requires --allow-gpu-research-signal; MLX GPU scorer "
            "responses are profiling/prescreen signal only and are not "
            "CPU-transfer-stable by default"
        )

    rows: list[dict[str, Any]] = []
    for device in device_values:
        for batch_pairs in batch_pairs_values:
            for repeat_index in range(int(repeat)):
                started = time.time()
                payload = build_mlx_scorer_response_payload(
                    reference_cache_dir=reference_cache_dir,
                    candidate_cache_dir=candidate_cache_dir,
                    archive_size_bytes=archive_size_bytes,
                    repo_root=repo_root,
                    batch_pairs=batch_pairs,
                    device_type=device,
                    start_pair=start_pair,
                    max_pairs=max_pairs,
                    allow_gpu_research_signal=allow_gpu_research_signal,
                )
                wall_seconds = time.time() - started
                n_samples = int(payload["n_samples"])
                elapsed = float(payload["elapsed_seconds"])
                rows.append(
                    {
                        "device": device,
                        "batch_pairs": int(batch_pairs),
                        "repeat_index": repeat_index,
                        "n_samples": n_samples,
                        "start_pair": int(payload["start_pair"]),
                        "pair_window": payload["pair_window"],
                        "elapsed_seconds": elapsed,
                        "wall_seconds": wall_seconds,
                        "pairs_per_second": n_samples / elapsed if elapsed > 0 else 0.0,
                        "canonical_score": payload["canonical_score"],
                        "avg_posenet_dist": payload["avg_posenet_dist"],
                        "avg_segnet_dist": payload["avg_segnet_dist"],
                        "posenet_sha256": payload["components"]["posenet_sha256"],
                        "segnet_sha256": payload["components"]["segnet_sha256"],
                    }
                )

    best = max(rows, key=lambda row: row["pairs_per_second"]) if rows else None
    return {
        "schema_version": "mlx_scorer_response_profile.v1",
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
        "reference_cache_dir": str(reference_cache_dir),
        "candidate_cache_dir": str(candidate_cache_dir),
        "archive_size_bytes": int(archive_size_bytes),
        "start_pair": int(start_pair),
        "max_pairs": int(max_pairs),
        "batch_pairs_values": list(batch_pairs_values),
        "device_values": list(device_values),
        "repeat": int(repeat),
        "gpu_research_signal_allowed": bool(allow_gpu_research_signal),
        "rows": rows,
        "best": best,
        "authority_status": (
            "Profiler output is local MLX throughput/signal only; it is not an auth-eval score."
        ),
    }


def write_profile_payload(profile: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_positive_int_csv(value: str, *, flag_name: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed < 1:
            raise ValueError(f"{flag_name} values must be >= 1, got {parsed}")
        out.append(parsed)
    if not out:
        raise ValueError(f"{flag_name} must contain at least one integer")
    return out


def parse_device_csv(value: str) -> list[str]:
    out: list[str] = []
    for item in value.split(","):
        device = item.strip().lower()
        if not device:
            continue
        if device not in {"cpu", "gpu"}:
            raise ValueError(f"--devices values must be cpu or gpu, got {device!r}")
        out.append(device)
    if not out:
        raise ValueError("--devices must contain at least one device")
    return out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
