from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .evaluate import _parse_report, _score, _upstream_env
from .paths import default_upstream_root, repo_root

FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


def _local_axis_tag(device: str) -> str:
    if device == "cpu":
        return "[macOS-CPU advisory]"
    return f"[local-{device.upper()} advisory]"


@dataclass(frozen=True)
class LocalSubmissionReplaySummary:
    schema: str
    submission_dir: str
    source_runtime_submission_dir: str
    archive_zip_path: str
    device: str
    returncode: int
    evaluation_passed: bool
    report_path: str
    stdout_path: str
    stderr_path: str
    wall_clock_seconds: float
    pose_distortion: float | None
    seg_distortion: float | None
    original_uncompressed_bytes: int | None
    archive_bytes: int | None
    rate: float | None
    local_score_estimate: float | None
    upstream_report_score_rounded: float | None
    inflated_dir: str
    inflated_dir_cleanup: str
    archive_extract_dir_cleanup: str
    scratch_cleanup_manifest_path: str
    blockers: list[str]
    axis_tag: str
    score_claim: bool
    score_claim_valid: bool
    promotion_eligible: bool
    promotable: bool
    rank_or_kill_eligible: bool
    ready_for_exact_eval_dispatch: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_records(root: Path) -> tuple[list[dict[str, object]], int]:
    if not root.exists():
        return [], 0
    records: list[dict[str, object]] = []
    total_bytes = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        size = path.stat().st_size
        total_bytes += size
        records.append(
            {
                "path": str(path),
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": int(size),
                "sha256": _sha256_file(path),
            }
        )
    return records, total_bytes


def _write_scratch_cleanup_manifest(
    *,
    output_path: Path,
    submission_dir: Path,
    source_runtime_submission_dir: Path,
    archive_zip_path: Path,
    upstream_root: Path,
    video_names_file: Path,
    device: str,
    command: list[str],
    returncode: int,
    evaluation_passed: bool,
    inflated_dir: Path,
    archive_extract_dir: Path,
    cleanup_reason: str,
) -> Path:
    inflated_records, inflated_bytes = _tree_records(inflated_dir)
    archive_records, archive_bytes = _tree_records(archive_extract_dir)
    tree_hash = hashlib.sha256()
    for section, records in (("inflated", inflated_records), ("archive_extract", archive_records)):
        for rec in records:
            tree_hash.update(section.encode("utf-8"))
            tree_hash.update(b"\0")
            tree_hash.update(str(rec["relative_path"]).encode("utf-8"))
            tree_hash.update(b"\0")
            tree_hash.update(str(rec["bytes"]).encode("ascii"))
            tree_hash.update(b"\0")
            tree_hash.update(str(rec["sha256"]).encode("ascii"))
            tree_hash.update(b"\0")
    manifest = {
        "schema": "local_submission_replay_scratch_cleanup_manifest.v1",
        "created_at_utc": _utc_now(),
        "submission_dir": str(submission_dir),
        "source_runtime_submission_dir": str(source_runtime_submission_dir),
        "archive_zip_path": str(archive_zip_path),
        "archive_zip_bytes": archive_zip_path.stat().st_size if archive_zip_path.exists() else None,
        "archive_zip_sha256": _sha256_file(archive_zip_path) if archive_zip_path.exists() else None,
        "upstream_root": str(upstream_root),
        "video_names_file": str(video_names_file),
        "device": device,
        "command": command,
        "returncode": int(returncode),
        "evaluation_passed": bool(evaluation_passed),
        "cleanup_reason": cleanup_reason,
        "rebuildable_from": {
            "archive_zip_path": str(archive_zip_path),
            "runtime_submission_dir": str(source_runtime_submission_dir),
            "upstream_root": str(upstream_root),
            "video_names_file": str(video_names_file),
            "device": device,
        },
        "deleted_after_manifest": True,
        "inflated_dir": str(inflated_dir),
        "inflated_file_count": len(inflated_records),
        "inflated_total_bytes": int(inflated_bytes),
        "archive_extract_dir": str(archive_extract_dir),
        "archive_extract_file_count": len(archive_records),
        "archive_extract_total_bytes": int(archive_bytes),
        "scratch_tree_sha256": tree_hash.hexdigest(),
        "files": {
            "inflated": inflated_records,
            "archive_extract": archive_records,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def stage_local_replay_submission(
    *,
    runtime_submission_dir: Path,
    archive_zip_path: Path | None,
    output_dir: Path,
    force: bool = False,
) -> Path:
    """Stage an arbitrary byte-closed submission packet for local replay."""

    runtime_submission_dir = runtime_submission_dir.resolve()
    output_dir = output_dir.resolve()
    if not runtime_submission_dir.is_dir():
        raise FileNotFoundError(f"runtime submission dir not found: {runtime_submission_dir}")
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"output dir exists; pass force to replace: {output_dir}")
        shutil.rmtree(output_dir)
    submission_dir = output_dir / "submission"
    ignore = shutil.ignore_patterns("inflated", "archive", "report.txt", "stdout.txt", "stderr.txt")
    shutil.copytree(runtime_submission_dir, submission_dir, ignore=ignore)

    source_archive = archive_zip_path.resolve() if archive_zip_path is not None else runtime_submission_dir / "archive.zip"
    if not source_archive.is_file():
        raise FileNotFoundError(f"archive.zip not found: {source_archive}")
    shutil.copy2(source_archive, submission_dir / "archive.zip")
    return submission_dir


def run_local_submission_replay(
    *,
    submission_dir: Path,
    source_runtime_submission_dir: Path,
    archive_zip_path: Path,
    device: str = "cpu",
    upstream_root: Path | None = None,
    video_names_file: Path | None = None,
    keep_inflated: bool = False,
) -> LocalSubmissionReplaySummary:
    """Run upstream evaluate.sh for an arbitrary staged submission and clean raw scratch."""

    root = repo_root()
    upstream_root = (upstream_root or default_upstream_root()).resolve()
    submission_dir = submission_dir.resolve()
    archive_zip_path = archive_zip_path.resolve()
    source_runtime_submission_dir = source_runtime_submission_dir.resolve()
    if video_names_file is None:
        video_names_file = upstream_root / "public_test_video_names.txt"
    else:
        video_names_file = video_names_file.resolve()

    archive_zip = submission_dir / "archive.zip"
    inflate_sh = submission_dir / "inflate.sh"
    if not archive_zip.is_file():
        raise FileNotFoundError(f"staged archive.zip missing: {archive_zip}")
    if not inflate_sh.is_file():
        raise FileNotFoundError(f"staged inflate.sh missing: {inflate_sh}")

    inflated_dir = submission_dir / "inflated"
    archive_extract_dir = submission_dir / "archive"
    if inflated_dir.exists():
        shutil.rmtree(inflated_dir)
    if archive_extract_dir.exists():
        shutil.rmtree(archive_extract_dir)

    stdout_path = submission_dir.parent / "local_replay_stdout.txt"
    stderr_path = submission_dir.parent / "local_replay_stderr.txt"
    evaluate_sh = upstream_root / "evaluate.sh"
    env = _upstream_env(upstream_root)
    env["PYTHON"] = sys.executable
    command = [
        "bash",
        str(evaluate_sh),
        "--submission-dir",
        str(submission_dir),
        "--video-names-file",
        str(video_names_file),
        "--device",
        device,
    ]
    start = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(
            command,
            cwd=root,
            env=env,
            text=True,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    wall = time.time() - start

    blockers: list[str] = []
    report_path = submission_dir / "report.txt"
    pose: float | None = None
    seg: float | None = None
    original_bytes: int | None = None
    archive_bytes: int | None = None
    rate: float | None = None
    local_score: float | None = None
    report_score: float | None = None
    passed = proc.returncode == 0 and report_path.is_file()
    if passed:
        parsed = _parse_report(report_path)
        pose = float(parsed["pose"])
        seg = float(parsed["seg"])
        original_bytes = int(parsed["original_bytes"])
        archive_bytes = int(parsed["submission_bytes"])
        rate = float(parsed["rate"])
        report_score = float(parsed["final_score"])
        local_score = _score(seg, pose, rate)
    else:
        blockers.append("local_replay_failed_or_report_missing")
        if proc.returncode != 0:
            blockers.append(f"local_replay_returncode:{proc.returncode}")

    if keep_inflated:
        inflated_cleanup = "retained_by_request"
        archive_cleanup = "retained_by_request"
        scratch_cleanup_manifest_path = None
    else:
        cleanup_reason = "deleted_after_success" if passed else "deleted_after_failed_replay"
        scratch_cleanup_manifest_path = _write_scratch_cleanup_manifest(
            output_path=submission_dir.parent / "scratch_cleanup_manifest.json",
            submission_dir=submission_dir,
            source_runtime_submission_dir=source_runtime_submission_dir,
            archive_zip_path=archive_zip_path,
            upstream_root=upstream_root,
            video_names_file=video_names_file,
            device=device,
            command=command,
            returncode=int(proc.returncode),
            evaluation_passed=passed,
            inflated_dir=inflated_dir,
            archive_extract_dir=archive_extract_dir,
            cleanup_reason=cleanup_reason,
        )
        shutil.rmtree(inflated_dir, ignore_errors=True)
        shutil.rmtree(archive_extract_dir, ignore_errors=True)
        inflated_cleanup = cleanup_reason
        archive_cleanup = cleanup_reason

    return LocalSubmissionReplaySummary(
        schema="local_submission_replay.v1",
        submission_dir=str(submission_dir),
        source_runtime_submission_dir=str(source_runtime_submission_dir),
        archive_zip_path=str(archive_zip_path),
        device=device,
        returncode=int(proc.returncode),
        evaluation_passed=passed,
        report_path=str(report_path),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        wall_clock_seconds=wall,
        pose_distortion=pose,
        seg_distortion=seg,
        original_uncompressed_bytes=original_bytes,
        archive_bytes=archive_bytes,
        rate=rate,
        local_score_estimate=local_score,
        upstream_report_score_rounded=report_score,
        inflated_dir=str(inflated_dir),
        inflated_dir_cleanup=inflated_cleanup,
        archive_extract_dir_cleanup=archive_cleanup,
        scratch_cleanup_manifest_path=str(scratch_cleanup_manifest_path) if scratch_cleanup_manifest_path else "",
        blockers=blockers,
        axis_tag=_local_axis_tag(device),
        **FALSE_AUTHORITY,
    )


__all__ = [
    "LocalSubmissionReplaySummary",
    "run_local_submission_replay",
    "stage_local_replay_submission",
]
