from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


BEST_RE = re.compile(r"best checkpoint -> epoch (?P<epoch>\d+) score=(?P<score>[0-9.]+) int8=(?P<int8_bytes>\d+) bytes")
SAVED_FP32_RE = re.compile(r"Saved fp32:\s+(?P<path>\S+)")
SAVED_INT8_RE = re.compile(r"Saved int8:\s+(?P<path>\S+)\s+\((?P<int8_bytes>\d+)\s+bytes\)")
SAVED_FINAL_META_RE = re.compile(r"Saved final meta:\s+(?P<path>\S+)")
PROXY_POSE_RE = re.compile(r"PoseNet distortion:\s*([0-9.]+)")
PROXY_SEG_RE = re.compile(r"SegNet distortion:\s*([0-9.]+)")
PROXY_RATE_RE = re.compile(r"Compression rate:\s*([0-9.]+)")
PROXY_SCORE_RE = re.compile(r"Final score:\s*([0-9.]+)")
FAILURE_RE = re.compile(r"(?P<error_type>[A-Za-z]+Error): (?P<message>[^\n]+)")


def _read_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _flatten_log_text(path: Path) -> str:
    text = path.read_text(errors="ignore")
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            rows = json.loads(stripped)
        except json.JSONDecodeError:
            return text
        if isinstance(rows, list):
            parts: list[str] = []
            for row in rows:
                if isinstance(row, dict):
                    parts.append(str(row.get("data", "")))
            return "".join(parts)
    return text


def extract_training_signals(log_path: Path) -> dict[str, object]:
    text = _flatten_log_text(log_path)
    signals: dict[str, object] = {}

    best = BEST_RE.search(text)
    if best:
        signals["best_checkpoint"] = {
            "epoch": int(best.group("epoch")),
            "score": float(best.group("score")),
            "int8_bytes": int(best.group("int8_bytes")),
        }

    saved: dict[str, object] = {}
    fp32 = SAVED_FP32_RE.search(text)
    if fp32:
        saved["fp32"] = fp32.group("path")
    int8 = SAVED_INT8_RE.search(text)
    if int8:
        saved["int8"] = int8.group("path")
        saved["int8_bytes"] = int(int8.group("int8_bytes"))
    final_meta = SAVED_FINAL_META_RE.search(text)
    if final_meta:
        saved["final_meta"] = final_meta.group("path")
    if saved:
        signals["saved"] = saved

    pose = PROXY_POSE_RE.search(text)
    seg = PROXY_SEG_RE.search(text)
    rate = PROXY_RATE_RE.search(text)
    score = PROXY_SCORE_RE.search(text)
    if pose and seg and rate and score:
        signals["proxy_result"] = {
            "pose_distortion": float(pose.group(1)),
            "seg_distortion": float(seg.group(1)),
            "current_workflow_rate": float(rate.group(1)),
            "current_workflow_score": float(score.group(1)),
        }

    failure = FAILURE_RE.search(text)
    if failure:
        signals["failure"] = {
            "error_type": failure.group("error_type"),
            "message": failure.group("message"),
        }

    return signals


def ingest_downloaded_outputs(
    *,
    manifest_path: Path,
    download_dir: Path,
    output_root: Path,
) -> dict[str, object]:
    manifest = _read_manifest(manifest_path)
    run_id = str(manifest["run_id"])
    evidence_dir = output_root / run_id
    evidence_dir.mkdir(parents=True, exist_ok=True)

    logs: list[dict[str, object]] = []
    latest_failure: dict[str, object] | None = None
    for path in sorted(download_dir.iterdir()):
        if path.is_file():
            dest = evidence_dir / path.name
            shutil.copy2(path, dest)
            if path.suffix == ".log":
                signals = extract_training_signals(dest)
                failure = signals.get("failure")
                if isinstance(failure, dict):
                    latest_failure = failure
                logs.append({
                    "file": path.name,
                    "signals": signals,
                })

    summary = {
        "run_id": run_id,
        "slug": manifest.get("slug"),
        "kernel_ref": manifest.get("kernel_ref"),
        "evidence_dir": str(evidence_dir),
        "logs": logs,
        "latest_failure": latest_failure,
    }
    (evidence_dir / "ingest_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def download_kernel_outputs(*, kernel_ref: str, download_dir: Path) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    command = ["uv", "run", "--with", "kaggle", "kaggle", "kernels", "output", kernel_ref, "-p", str(download_dir)]
    subprocess.run(command, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Kaggle kernel outputs into repo-local evidence.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Download outputs before ingesting")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = _read_manifest(args.manifest)
    kernel_ref = manifest.get("kernel_ref")
    if args.download:
        if not isinstance(kernel_ref, str) or not kernel_ref:
            raise ValueError(f"Missing kernel_ref in {args.manifest}")
        download_kernel_outputs(kernel_ref=kernel_ref, download_dir=args.download_dir)
    ingest_downloaded_outputs(
        manifest_path=args.manifest,
        download_dir=args.download_dir,
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
