#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Evaluate an existing raw output through upstream/evaluate.py.

This avoids re-running inflate when a raw file has already been produced by an
official ``inflate.sh`` control. Results are diagnostic/advisory unless the
caller separately proves the full contest custody path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _advisory_cache_key(
    *,
    args: argparse.Namespace,
    raw_sha256: str,
    archive_sha256: str,
    upstream_dir: Path,
    video_names_file: Path,
) -> dict[str, Any]:
    evaluate_py = upstream_dir / "evaluate.py"
    return {
        "schema": "raw_advisory_eval_cache_key_v2",
        "raw_sha256": raw_sha256,
        "archive_sha256": archive_sha256,
        "axis": args.axis_label,
        "device": args.device,
        "batch_size": int(args.batch_size),
        "num_threads": int(args.num_threads),
        "evaluate_py_sha256": _sha256_file(evaluate_py),
        "video_names_file_sha256": _sha256_file(video_names_file),
        "video_names_file": str(video_names_file),
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }


def _parse_report(report_path: Path) -> dict[str, Any]:
    text = report_path.read_text(encoding="utf-8")
    out: dict[str, Any] = {"report_text": text}
    for line in text.splitlines():
        stripped = line.strip()
        if "Average PoseNet Distortion" in stripped:
            out["avg_posenet_dist"] = float(stripped.rsplit(":", 1)[1].strip())
        elif "Average SegNet Distortion" in stripped:
            out["avg_segnet_dist"] = float(stripped.rsplit(":", 1)[1].strip())
        elif "Submission file size" in stripped:
            out["archive_size_bytes"] = int(
                stripped.rsplit(":", 1)[1].strip().replace(",", "").split()[0]
            )
        elif "Original uncompressed size" in stripped:
            out["original_uncompressed_size"] = int(
                stripped.rsplit(":", 1)[1].strip().replace(",", "").split()[0]
            )
        elif "Compression Rate" in stripped:
            out["rate_unscaled"] = float(stripped.rsplit(":", 1)[1].strip())
        elif "Final score" in stripped:
            out["final_score_reported"] = float(stripped.rsplit("=", 1)[1].strip())
    required = ("avg_posenet_dist", "avg_segnet_dist", "rate_unscaled")
    missing = [key for key in required if key not in out]
    if missing:
        raise ValueError(f"failed to parse report {report_path}; missing {missing}")
    import math

    out["canonical_score"] = (
        100.0 * float(out["avg_segnet_dist"])
        + math.sqrt(10.0 * float(out["avg_posenet_dist"]))
        + 25.0 * float(out["rate_unscaled"])
    )
    return out


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    raw = args.raw.resolve()
    archive = args.archive.resolve()
    output_dir = args.output_dir.resolve()
    upstream_dir = args.upstream_dir.resolve()
    video_names_file = args.video_names_file.resolve()
    if not raw.is_file():
        raise FileNotFoundError(f"--raw not found: {raw}")
    if not archive.is_file():
        raise FileNotFoundError(f"--archive not found: {archive}")
    if not (upstream_dir / "evaluate.py").is_file():
        raise FileNotFoundError(f"--upstream-dir missing evaluate.py: {upstream_dir}")
    if not video_names_file.is_file():
        raise FileNotFoundError(f"--video-names-file not found: {video_names_file}")
    raw_sha256 = _sha256_file(raw)
    archive_sha256 = _sha256_file(archive)
    if args.axis_label.startswith("[contest-"):
        raise ValueError(
            "run_raw_advisory_eval.py evaluates an existing raw output and cannot emit [contest-*] axes"
        )
    cache_key = _advisory_cache_key(
        args=args,
        raw_sha256=raw_sha256,
        archive_sha256=archive_sha256,
        upstream_dir=upstream_dir,
        video_names_file=video_names_file,
    )

    cache_path = output_dir / "raw_advisory_eval.json"
    if args.reuse_cache and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("schema_version") == 1
            and cached.get("kind") == "raw_output_upstream_evaluate_advisory_v1"
            and cached.get("returncode") == 0
            and cached.get("cache_key") == cache_key
        ):
            cached["cache_hit"] = True
            return cached

    eval_root = output_dir / "submission_dir"
    inflated = eval_root / "inflated"
    inflated.mkdir(parents=True, exist_ok=True)
    archive_link = eval_root / "archive.zip"
    raw_link = inflated / Path(args.file_list_name).with_suffix(".raw").name
    for link, target in ((archive_link, archive), (raw_link, raw)):
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)

    report_path = output_dir / "report.txt"
    cmd = [
        sys.executable,
        str(upstream_dir / "evaluate.py"),
        "--submission-dir",
        str(eval_root),
        "--uncompressed-dir",
        str(upstream_dir / "videos"),
        "--video-names-file",
        str(video_names_file),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--num-threads",
        str(args.num_threads),
        "--report",
        str(report_path),
    ]
    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=args.timeout,
        check=False,
    )
    elapsed = time.monotonic() - start
    (output_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "raw_output_upstream_evaluate_advisory_v1",
        "cache_key": cache_key,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "axis": args.axis_label,
        "device": args.device,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "raw": {"path": str(raw), "bytes": raw.stat().st_size, "sha256": raw_sha256},
        "archive": {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": archive_sha256,
        },
        "work_dir": str(eval_root),
        "report": str(report_path),
        "stdout_log": str(output_dir / "stdout.log"),
        "stderr_log": str(output_dir / "stderr.log"),
        "blockers": [
            "raw_eval_advisory_not_full_archive_inflate_custody",
            "not_contest_cuda",
        ],
    }
    if proc.returncode == 0:
        payload.update(_parse_report(report_path))
    else:
        payload["blockers"].append("upstream_evaluate_failed")
    _write_json(output_dir / "raw_advisory_eval.json", payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument("--video-names-file", type=Path, default=REPO_ROOT / "upstream" / "public_test_video_names.txt")
    parser.add_argument("--file-list-name", default="0.mkv")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--axis-label", default="[macOS-CPU advisory]")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Return a matching raw_advisory_eval.json instead of rerunning upstream/evaluate.py.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    payload = run_eval(parse_args(argv))
    print(
        json.dumps(
            {
                "json": str(Path(payload["report"]).parent / "raw_advisory_eval.json"),
                "returncode": payload["returncode"],
                "canonical_score": payload.get("canonical_score"),
                "avg_posenet_dist": payload.get("avg_posenet_dist"),
                "avg_segnet_dist": payload.get("avg_segnet_dist"),
                "cache_hit": bool(payload.get("cache_hit")),
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return int(payload["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
