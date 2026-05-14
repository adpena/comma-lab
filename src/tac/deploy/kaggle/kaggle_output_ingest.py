# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
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
DEFAULT_EXCLUDE_OUTPUT_PATTERNS = (
    r"(^|/)inflated/.*\.raw$",
    r"(^|/)__pycache__/",
    r"\.pyc$",
)


@dataclass(frozen=True)
class DownloadedOutput:
    file_name: str
    local_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class SkippedOutput:
    file_name: str
    reason: str


@dataclass(frozen=True)
class KernelOutputDownloadManifest:
    schema: str
    kernel_ref: str
    pages_seen: int
    files_seen: int
    files_matched: int
    files_downloaded: int
    files_skipped: int
    log_downloaded: bool
    downloaded: list[DownloadedOutput]
    skipped_by_reason: dict[str, int]
    skipped_sample: list[SkippedOutput]


def _read_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _ingestable_files(
    *,
    download_dir: Path,
    manifest_path: Path,
) -> list[Path]:
    """Return files belonging to the current download.

    Operators often reuse a ``latest`` directory. When a paginated download
    manifest exists, use it as the custody allowlist so stale files from an
    older harvest cannot silently enter the new evidence directory.
    """

    download_manifest_path = download_dir / "kaggle_output_download_manifest.json"
    if not download_manifest_path.is_file():
        return sorted(path for path in download_dir.rglob("*") if path.is_file())

    payload = _read_manifest(download_manifest_path)
    allowed = {"kaggle_output_download_manifest.json"}
    try:
        allowed.add(manifest_path.relative_to(download_dir).as_posix())
    except ValueError:
        pass

    downloaded = payload.get("downloaded", [])
    if isinstance(downloaded, list):
        for row in downloaded:
            if isinstance(row, dict) and isinstance(row.get("file_name"), str):
                allowed.add(row["file_name"])

    files: list[Path] = []
    for rel in sorted(allowed):
        candidate = download_dir / rel
        if candidate.is_file():
            files.append(candidate)
    return files


def _compile_patterns(patterns: Iterable[str] | None) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern) for pattern in patterns or ())


def _matches_any(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def should_download_output(
    file_name: str,
    *,
    include_patterns: Iterable[str] | None = None,
    exclude_patterns: Iterable[str] | None = DEFAULT_EXCLUDE_OUTPUT_PATTERNS,
) -> tuple[bool, str]:
    """Return whether a Kaggle output file should be harvested.

    The default policy keeps custody artifacts while dropping rebuildable large
    raw videos and interpreter cache files. Callers can supply include patterns
    to harvest a narrow run prefix without downloading the staged source tree.
    """

    include = _compile_patterns(include_patterns)
    exclude = _compile_patterns(exclude_patterns)
    if include and not _matches_any(include, file_name):
        return False, "not_matched_by_include_patterns"
    if exclude and _matches_any(exclude, file_name):
        return False, "matched_exclude_patterns"
    return True, "matched"


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
    latest_checkpoint: dict[str, object] | None = None
    latest_score_table: dict[str, object] | None = None
    for path in _ingestable_files(download_dir=download_dir, manifest_path=manifest_path):
        rel = path.relative_to(download_dir)
        dest = evidence_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        if path.suffix == ".log":
            signals = extract_training_signals(dest)
            failure = signals.get("failure")
            if isinstance(failure, dict):
                latest_failure = failure
            logs.append({
                "file": rel.as_posix(),
                "signals": signals,
            })
        if path.name.endswith("_best_meta.json"):
            payload = _read_manifest(dest)
            meta = payload.get("meta", {})
            latest_checkpoint = {
                "epoch": payload.get("epoch"),
                "scorer": payload.get("scorer"),
                "int8_size": payload.get("int8_size"),
                "variant": meta.get("variant"),
                "hidden": meta.get("hidden"),
                "meta_path": str(dest),
            }
        if path.name == "score_table_manifest.json":
            payload = _read_manifest(dest)
            manifest_schema = payload.get("manifest_schema")
            if manifest_schema in {
                "pr106_latent_score_table_manifest_v1",
                "pr106_yshift_score_table_manifest_v1",
            }:
                improvement_key = (
                    "strict_improvement_pair_count"
                    if manifest_schema == "pr106_latent_score_table_manifest_v1"
                    else "strict_improvement_frame_count"
                )
                latest_score_table = {
                    "manifest_schema": manifest_schema,
                    "manifest_path": str(dest),
                    "producer": payload.get("producer"),
                    "device": payload.get("device"),
                    "elapsed_seconds": payload.get("elapsed_seconds"),
                    "source_archive_sha256": payload.get("source_archive_sha256"),
                    "source_zero_bin_sha256": payload.get("source_zero_bin_sha256"),
                    "score_table_npy_path": payload.get("score_table_npy_path"),
                    "score_table_npy_bytes": payload.get("score_table_npy_bytes"),
                    "score_table_npy_sha256": payload.get("score_table_npy_sha256"),
                    "score_table_shape": payload.get("score_table_shape"),
                    "candidate_count": payload.get("candidate_count"),
                    "strict_improvement_count": payload.get(improvement_key),
                    improvement_key: payload.get(improvement_key),
                    "best_improvement_min": payload.get("best_improvement_min"),
                    "best_improvement_mean": payload.get("best_improvement_mean"),
                    "best_improvement_max": payload.get("best_improvement_max"),
                    "ready_for_builder": payload.get("ready_for_builder"),
                    "source_manifest_score_claim": payload.get("score_claim"),
                    "source_manifest_ready_for_exact_eval_dispatch": payload.get(
                        "ready_for_exact_eval_dispatch"
                    ),
                    "score_claim": False,
                    "ready_for_exact_eval_dispatch": False,
                    "promotion_eligible": False,
                    "proxy_authority_forced_false": True,
                }

    summary = {
        "run_id": run_id,
        "slug": manifest.get("slug"),
        "kernel_ref": manifest.get("kernel_ref"),
        "evidence_dir": str(evidence_dir),
        "logs": logs,
        "latest_failure": latest_failure,
        "latest_checkpoint": latest_checkpoint,
        "latest_score_table": latest_score_table,
    }
    (evidence_dir / "ingest_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _split_kernel_ref(kernel_ref: str) -> tuple[str, str]:
    if "/" not in kernel_ref:
        raise ValueError(f"kernel_ref must be <owner>/<slug>, got {kernel_ref!r}")
    owner_slug, kernel_slug = kernel_ref.split("/", 1)
    if not owner_slug or not kernel_slug:
        raise ValueError(f"kernel_ref must be <owner>/<slug>, got {kernel_ref!r}")
    return owner_slug, kernel_slug


def _write_output_file(
    *,
    file_name: str,
    content: bytes,
    download_dir: Path,
) -> DownloadedOutput:
    dest = download_dir / file_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return DownloadedOutput(
        file_name=file_name,
        local_path=str(dest),
        bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _download_output_items(
    *,
    kernel_slug: str,
    page_items: Iterable[object],
    log_text: str | None,
    download_dir: Path,
    fetch_bytes: Callable[[str], bytes],
    include_patterns: Iterable[str] | None,
    exclude_patterns: Iterable[str] | None,
) -> tuple[list[DownloadedOutput], list[SkippedOutput], bool, int, int]:
    downloaded: list[DownloadedOutput] = []
    skipped: list[SkippedOutput] = []
    files_seen = 0
    files_matched = 0
    for item in page_items:
        file_name = str(item.file_name)
        url = str(item.url)
        files_seen += 1
        keep, reason = should_download_output(
            file_name,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        if not keep:
            skipped.append(SkippedOutput(file_name=file_name, reason=reason))
            continue
        files_matched += 1
        downloaded.append(
            _write_output_file(
                file_name=file_name,
                content=fetch_bytes(url),
                download_dir=download_dir,
            )
        )

    log_downloaded = False
    if log_text is not None:
        record = _write_output_file(
            file_name=f"{kernel_slug}.log",
            content=log_text.encode("utf-8"),
            download_dir=download_dir,
        )
        if not any(existing.file_name == record.file_name for existing in downloaded):
            downloaded.append(record)
        log_downloaded = True

    return downloaded, skipped, log_downloaded, files_seen, files_matched


def _skipped_by_reason(skipped: Iterable[SkippedOutput]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in skipped:
        counts[row.reason] = counts.get(row.reason, 0) + 1
    return dict(sorted(counts.items()))


def download_kernel_outputs(
    *,
    kernel_ref: str,
    download_dir: Path,
    include_patterns: Iterable[str] | None = None,
    exclude_patterns: Iterable[str] | None = DEFAULT_EXCLUDE_OUTPUT_PATTERNS,
) -> dict[str, object]:
    """Download Kaggle kernel outputs through the paginated SDK API.

    `kaggle kernels output` currently prints a next-page token but does not
    follow it. This function follows every page, filters noisy files before
    downloading, and writes a compact custody manifest.
    """

    download_dir.mkdir(parents=True, exist_ok=True)
    owner_slug, kernel_slug = _split_kernel_ref(kernel_ref)

    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-untyped] # noqa: I001
    from kagglesdk.kernels.types.kernels_api_service import (  # type: ignore[import-untyped]
        ApiListKernelSessionOutputRequest,
    )
    import requests

    def fetch_bytes(url: str) -> bytes:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        return response.content

    api = KaggleApi()
    api.authenticate()
    pages_seen = 0
    files_seen = 0
    files_matched = 0
    downloaded: list[DownloadedOutput] = []
    skipped: list[SkippedOutput] = []
    log_downloaded = False
    token = None

    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner_slug
            request.kernel_slug = kernel_slug
            if token:
                request.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            page_downloaded, page_skipped, page_log_downloaded, page_seen, page_matched = _download_output_items(
                kernel_slug=kernel_slug,
                page_items=response.files or [],
                log_text=response.log if not log_downloaded else None,
                download_dir=download_dir,
                fetch_bytes=fetch_bytes,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )
            pages_seen += 1
            files_seen += page_seen
            files_matched += page_matched
            downloaded.extend(page_downloaded)
            skipped.extend(page_skipped)
            log_downloaded = log_downloaded or page_log_downloaded
            token = response.next_page_token
            if not token:
                break

    manifest = KernelOutputDownloadManifest(
        schema="kaggle_output_download_manifest_v1",
        kernel_ref=kernel_ref,
        pages_seen=pages_seen,
        files_seen=files_seen,
        files_matched=files_matched,
        files_downloaded=len(downloaded),
        files_skipped=len(skipped),
        log_downloaded=log_downloaded,
        downloaded=downloaded,
        skipped_by_reason=_skipped_by_reason(skipped),
        skipped_sample=skipped[:20],
    )
    payload = asdict(manifest)
    (download_dir / "kaggle_output_download_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Kaggle kernel outputs into repo-local evidence.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Download outputs before ingesting")
    parser.add_argument(
        "--include-output-pattern",
        action="append",
        default=[],
        help="Regex for output file paths to download; may be repeated.",
    )
    parser.add_argument(
        "--exclude-output-pattern",
        action="append",
        default=list(DEFAULT_EXCLUDE_OUTPUT_PATTERNS),
        help="Regex for output file paths to skip; may be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = _read_manifest(args.manifest)
    kernel_ref = manifest.get("kernel_ref")
    if args.download:
        if not isinstance(kernel_ref, str) or not kernel_ref:
            raise ValueError(f"Missing kernel_ref in {args.manifest}")
        download_kernel_outputs(
            kernel_ref=kernel_ref,
            download_dir=args.download_dir,
            include_patterns=args.include_output_pattern,
            exclude_patterns=args.exclude_output_pattern,
        )
    ingest_downloaded_outputs(
        manifest_path=args.manifest,
        download_dir=args.download_dir,
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
