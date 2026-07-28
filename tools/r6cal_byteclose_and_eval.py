#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""R6CAL: byte-close a V10 production archive and score it with the real evaluator.

Inflates ``<submission_dir>/archive.zip`` through the production V10 receiver, then
runs the pinned ``upstream/evaluate.py`` over the full public sample set on the exact
archive bytes.  Emits a SHA-pinned receipt with the score recomputed from components.

No score claim: macOS-CPU is advisory per CLAUDE.md, never ``contest-CPU``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import zipfile
from collections.abc import Sequence
from math import sqrt
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = "r6cal_byteclose_eval_receipt.v1"
EVIDENCE_AXIS = "[macOS-CPU advisory - real evaluator, real bytes]"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive_inventory(archive_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path) as bundle:
        members = [
            {
                "name": info.filename,
                "compress_type": int(info.compress_type),
                "file_size": int(info.file_size),
                "compress_size": int(info.compress_size),
                "sha256": hashlib.sha256(bundle.read(info.filename)).hexdigest(),
            }
            for info in bundle.infolist()
        ]
    archive_bytes = archive_path.stat().st_size
    stored_bytes = sum(member["compress_size"] for member in members)
    return {
        "archive_bytes": archive_bytes,
        "archive_sha256": _sha256_file(archive_path),
        "container_overhead_bytes": archive_bytes - stored_bytes,
        "members": members,
    }


def _uncompressed_bytes(uncompressed_dir: Path) -> int:
    return sum(path.stat().st_size for path in uncompressed_dir.rglob("*") if path.is_file())


def _parse_report(text: str) -> dict[str, float | int]:
    patterns = {
        "posenet_distortion": r"Average PoseNet Distortion:\s*([0-9.eE+-]+)",
        "segnet_distortion": r"Average SegNet Distortion:\s*([0-9.eE+-]+)",
        "sample_count": r"Evaluation results over\s*(\d+)\s*samples",
        "compressed_size": r"Submission file size:\s*([0-9,]+)\s*bytes",
        "uncompressed_size": r"Original uncompressed size:\s*([0-9,]+)\s*bytes",
    }
    parsed: dict[str, float | int] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is None:
            raise SystemExit(f"evaluator report missing field {key!r}; raw output:\n{text}")
        raw = match.group(1).replace(",", "")
        parsed[key] = int(raw) if key in {"sample_count", "compressed_size", "uncompressed_size"} else float(raw)
    return parsed


def _score_components(segnet: float, posenet: float, compressed: int, uncompressed: int) -> dict[str, float]:
    seg_term = 100.0 * segnet
    pose_term = sqrt(10.0 * posenet)
    rate = compressed / uncompressed
    rate_term = 25.0 * rate
    return {
        "seg_term_100x_dseg": seg_term,
        "pose_term_sqrt_10x_dpose": pose_term,
        "rate": rate,
        "rate_term_25x_rate": rate_term,
        "score_recomputed_from_components": seg_term + pose_term + rate_term,
    }


def _inflate(submission_dir: Path, video_names_file: Path, repo_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(repo_root / "src"))
    from tac.witness_dsl.v10_production_receiver import inflate_archive

    started = time.time()
    result = inflate_archive(submission_dir, submission_dir / "inflated", video_names_file)
    elapsed = time.time() - started
    if not result.completed or result.raw_path is None or result.raw_sha256 is None:
        raise SystemExit("V10 production inflate stopped before final raw promotion")
    return {
        "completed": bool(result.completed),
        "raw_path": str(result.raw_path),
        "raw_bytes": int(result.raw_bytes),
        "raw_sha256": str(result.raw_sha256),
        "pair_stages_preserved": int(result.pair_stages_preserved),
        "numerator_values_verified": int(result.numerator_values_verified),
        "tree_sha256": str(result.tree_sha256),
        "wall_seconds": elapsed,
    }


def _evaluate(
    repo_root: Path, submission_dir: Path, uncompressed_dir: Path, video_names_file: Path,
    report_path: Path, device: str, batch_size: int, num_threads: int, python_exe: str,
) -> dict[str, Any]:
    upstream = repo_root / "upstream"
    argv = [
        python_exe, str(upstream / "evaluate.py"),
        "--submission-dir", str(submission_dir),
        "--uncompressed-dir", str(uncompressed_dir),
        "--video-names-file", str(video_names_file),
        "--report", str(report_path),
        "--device", device,
        "--batch-size", str(batch_size),
        "--num-threads", str(num_threads),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(upstream), str(repo_root / "src"), env.get("PYTHONPATH", "")])
    started = time.time()
    proc = subprocess.run(argv, cwd=upstream, env=env, capture_output=True, text=True, check=False)
    elapsed = time.time() - started
    combined = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        raise SystemExit(f"evaluator returned rc={proc.returncode}:\n{combined[-8000:]}")
    return {"argv": argv, "returncode": proc.returncode, "wall_seconds": elapsed, "stdout_tail": combined[-4000:]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True, help="holds archive.zip; inflated/ is written here")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--skip-inflate", action="store_true", help="reuse an existing inflated/ tree")
    parser.add_argument("--source-record", type=Path, action="append", default=[],
                        help="upstream provenance JSON to SHA-pin into the receipt. Repeatable.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = args.repo_root.resolve()
    submission_dir = args.submission_dir.resolve()
    uncompressed_dir = repo_root / "upstream" / "videos"
    video_names_file = repo_root / "upstream" / "public_test_video_names.txt"
    archive_path = submission_dir / "archive.zip"
    if not archive_path.exists():
        raise SystemExit(f"missing archive: {archive_path}")

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "host": {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0]},
        "archive": _archive_inventory(archive_path),
        "uncompressed_bytes_measured": _uncompressed_bytes(uncompressed_dir),
        "source_records": {
            str(path): {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in (p.resolve() for p in args.source_record) if path.is_file()
        },
    }
    receipt["git_head"] = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()

    if not args.skip_inflate:
        receipt["inflate"] = _inflate(submission_dir, video_names_file, repo_root)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True))

    report_path = submission_dir / "report.txt"
    receipt["evaluate"] = _evaluate(
        repo_root, submission_dir, uncompressed_dir, video_names_file, report_path,
        args.device, args.batch_size, args.num_threads, args.python_exe,
    )
    report_text = report_path.read_text() if report_path.exists() else receipt["evaluate"]["stdout_tail"]
    receipt["evaluator_report_text"] = report_text
    parsed = _parse_report(report_text)
    receipt["evaluator_parsed"] = parsed
    receipt["score"] = _score_components(
        float(parsed["segnet_distortion"]), float(parsed["posenet_distortion"]),
        int(parsed["compressed_size"]), int(parsed["uncompressed_size"]),
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps({"receipt": str(args.receipt), **receipt["score"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
