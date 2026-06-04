from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from .install import install_payload_bytes, install_payload_manifest, install_submission
from .lock import submission_lock
from .paths import default_upstream_root, repo_root
from .tracks.exact_current import create_minimal_archive

REPORT_PATTERNS = {
    "pose": re.compile(r"Average PoseNet Distortion:\s*([0-9.]+)"),
    "seg": re.compile(r"Average SegNet Distortion:\s*([0-9.]+)"),
    "submission_bytes": re.compile(r"Submission file size:\s*([0-9,]+) bytes"),
    "original_bytes": re.compile(r"Original uncompressed size:\s*([0-9,]+) bytes"),
    "rate": re.compile(r"Compression Rate:\s*([0-9.]+)"),
    "final_score": re.compile(r"Final score: .* =\s*([0-9.]+)"),
}


@dataclass
class EvaluationSummary:
    track: str
    device: str
    report_path: str
    copied_report_path: str | None
    current_workflow_archive_bytes: int
    pose_distortion: float
    seg_distortion: float
    original_uncompressed_bytes: int
    current_workflow_rate: float
    current_workflow_score: float
    rule_faithful_bundle_bytes: int | None
    rule_faithful_bundle_paths: list[str] | None
    rule_faithful_rate: float | None
    rule_faithful_score: float | None
    rule_faithful_status: str
    inflated_dir: str
    inflated_dir_retained: bool
    inflated_dir_cleanup: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


@dataclass
class ExternalEvaluationSummary:
    schema: str
    generated_at_utc: str
    axis_tag: str
    submission_dir: str
    upstream_root: str
    device: str
    command: list[str]
    returncode: int
    wall_seconds: float
    report_path: str
    report_copy_path: str | None
    stdout_path: str
    stderr_path: str
    python_environment: dict[str, Any]
    archive_zip: dict[str, Any]
    submission_manifest_before_eval: dict[str, Any]
    inflated_outputs_manifest: dict[str, Any]
    parsed_report: dict[str, float | int] | None
    score_claim: bool
    score_claim_valid: bool
    promotion_eligible: bool
    promotable: bool
    rank_or_kill_eligible: bool
    ready_for_exact_eval_dispatch: bool
    dispatch_attempted: bool
    gpu_launched: bool
    inflated_dir: str
    inflated_dir_retained: bool
    inflated_dir_cleanup: str
    blockers: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _upstream_env(upstream_root: Path) -> dict[str, str]:
    venv_bin = upstream_root / ".venv" / "bin"
    python_bin = venv_bin / "python"
    if not python_bin.exists():
        raise FileNotFoundError(
            f"Upstream virtualenv not found at {python_bin}. Run `uv sync --group cpu` in the upstream repo first."
        )

    env = os.environ.copy()
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    env["VIRTUAL_ENV"] = str(upstream_root / ".venv")
    env["COMMA_CHALLENGE_ROOT"] = str(upstream_root)
    return env


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=sys.stderr,
        stderr=sys.stderr,
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_manifest(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        rel = path.relative_to(root).as_posix()
        files.append(
            {
                "path": rel,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    aggregate = sha256()
    for row in files:
        aggregate.update(str(row["path"]).encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(row["bytes"]).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(row["sha256"]).encode("ascii"))
        aggregate.update(b"\0")
    return {
        "root": root.as_posix(),
        "file_count": len(files),
        "bytes": sum(int(row["bytes"]) for row in files),
        "tree_sha256": aggregate.hexdigest(),
        "files": files,
    }


def _file_manifest_excluding(root: Path, excluded_top_level: set[str]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        rel_path = path.relative_to(root)
        if rel_path.parts and rel_path.parts[0] in excluded_top_level:
            continue
        rel = rel_path.as_posix()
        files.append(
            {
                "path": rel,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    aggregate = sha256()
    for row in files:
        aggregate.update(str(row["path"]).encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(row["bytes"]).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(row["sha256"]).encode("ascii"))
        aggregate.update(b"\0")
    return {
        "root": root.as_posix(),
        "excluded_top_level": sorted(excluded_top_level),
        "file_count": len(files),
        "bytes": sum(int(row["bytes"]) for row in files),
        "tree_sha256": aggregate.hexdigest(),
        "files": files,
    }


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parse_report(report_path: Path) -> dict[str, float | int]:
    text = report_path.read_text()
    values: dict[str, float | int] = {}
    for key, pattern in REPORT_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            raise ValueError(f"Could not parse {key} from report: {report_path}")
        raw = match.group(1).replace(",", "")
        values[key] = int(raw) if key.endswith("bytes") else float(raw)
    return values


def _parse_report_if_present(report_path: Path, blockers: list[str]) -> dict[str, float | int] | None:
    if not report_path.is_file():
        blockers.append("upstream_report_missing")
        return None
    try:
        return _parse_report(report_path)
    except (OSError, ValueError) as exc:
        blockers.append(f"upstream_report_unparseable:{type(exc).__name__}")
        return None


def _score(seg_distortion: float, pose_distortion: float, rate: float) -> float:
    return 100.0 * seg_distortion + math.sqrt(10.0 * pose_distortion) + 25.0 * rate


def _rule_faithful_bundle_bytes(track: str, source_submission_dir: Path) -> int | None:
    if track == "exact_current":
        return None
    return install_payload_bytes(track, source_submission_dir)


def _rule_faithful_bundle_paths(track: str, source_submission_dir: Path) -> list[str] | None:
    if track == "exact_current":
        return None
    return [rel_path for rel_path, _ in install_payload_manifest(track, source_submission_dir)]


def evaluate_submission(
    name: str,
    *,
    device: str,
    upstream_root: Path | None = None,
    sync: bool = True,
    package: bool = False,
    report_copy: Path | None = None,
    keep_inflated: bool = False,
) -> EvaluationSummary:
    root = repo_root()
    upstream_root = upstream_root or default_upstream_root()
    source_submission_dir = root / "submissions" / name
    submission_dir = upstream_root / "submissions" / name

    with submission_lock(name, upstream_root):
        if package and not sync:
            raise ValueError("Packaging without sync is unsupported because the packaged artifact would not be the one under test.")

        if package:
            if name == "exact_current":
                create_minimal_archive(source_submission_dir / "archive.zip")
            elif name == "robust_current":
                package_env = os.environ.copy()
                package_env["COMMA_CHALLENGE_ROOT"] = str(upstream_root)
                _run(["bash", str(source_submission_dir / "compress.sh")], cwd=root, env=package_env)
            else:
                raise ValueError(f"Unsupported submission for packaging: {name}")

        if sync:
            install_submission(name, upstream_root=upstream_root, force=True)

        env = _upstream_env(upstream_root)
        evaluate_sh = upstream_root / "evaluate.sh"
        inflated_dir = submission_dir / "inflated"
        if inflated_dir.exists():
            shutil.rmtree(inflated_dir)
        inflated_dir_cleanup = "not_started"
        try:
            _run([
                "bash",
                str(evaluate_sh),
                "--submission-dir",
                str(submission_dir),
                "--device",
                device,
            ], cwd=root, env=env)
            inflated_dir_cleanup = "pending_success_cleanup"
        except BaseException:
            if keep_inflated:
                inflated_dir_cleanup = "retained_after_failed_evaluation_by_request"
            else:
                shutil.rmtree(inflated_dir, ignore_errors=True)
                inflated_dir_cleanup = "deleted_after_failed_evaluation"
            raise

        report_path = submission_dir / "report.txt"
        if not report_path.exists():
            raise FileNotFoundError(f"Expected report not found: {report_path}")

        copied_report_path: str | None = None
        if report_copy is not None:
            report_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(report_path, report_copy)
            copied_report_path = str(report_copy)

        parsed = _parse_report(report_path)
        pose = float(parsed["pose"])
        seg = float(parsed["seg"])
        archive_bytes = int(parsed["submission_bytes"])
        original_bytes = int(parsed["original_bytes"])
        current_rate = float(parsed["rate"])
        current_score = float(parsed["final_score"])

        rule_bytes = _rule_faithful_bundle_bytes(name, submission_dir)
        rule_paths = _rule_faithful_bundle_paths(name, submission_dir)
        if rule_bytes is None:
            rule_rate = None
            rule_score = None
            rule_status = "invalid_repo_side_dependency"
        else:
            rule_rate = rule_bytes / original_bytes
            rule_score = _score(seg, pose, rule_rate)
            rule_status = "estimated_from_scorer_distortions_plus_installed_runtime_payload"
        if keep_inflated:
            inflated_dir_cleanup = "retained_by_request"
        else:
            shutil.rmtree(inflated_dir, ignore_errors=True)
            inflated_dir_cleanup = "deleted_after_success"

        return EvaluationSummary(
            track=name,
            device=device,
            report_path=str(report_path),
            copied_report_path=copied_report_path,
            current_workflow_archive_bytes=archive_bytes,
            pose_distortion=pose,
            seg_distortion=seg,
            original_uncompressed_bytes=original_bytes,
            current_workflow_rate=current_rate,
            current_workflow_score=current_score,
            rule_faithful_bundle_bytes=rule_bytes,
            rule_faithful_bundle_paths=rule_paths,
            rule_faithful_rate=rule_rate,
            rule_faithful_score=rule_score,
            rule_faithful_status=rule_status,
            inflated_dir=str(inflated_dir),
            inflated_dir_retained=keep_inflated,
            inflated_dir_cleanup=inflated_dir_cleanup,
        )


def evaluate_external_submission_dir(
    *,
    submission_dir: Path,
    device: str,
    upstream_root: Path | None = None,
    artifact_dir: Path,
    keep_inflated: bool = False,
    min_free_bytes: int = 5 * 1024 * 1024 * 1024,
    require_upstream_venv: bool = True,
) -> ExternalEvaluationSummary:
    """Run upstream evaluate.sh on an already-materialized submission dir.

    This is the pipeline-safe path for candidate bundles that are not tracked
    under ``submissions/<name>``. It captures stdout/stderr, records archive and
    inflated-output hashes, and deletes success-only inflated raw output only
    after a rebuild certificate exists in the returned JSON.
    """

    root = repo_root()
    upstream_root = upstream_root or default_upstream_root()
    submission = submission_dir.expanduser().resolve(strict=False)
    artifact_dir = artifact_dir.expanduser().resolve(strict=False)
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device: {device}")
    if not submission.is_dir():
        raise FileNotFoundError(f"submission dir not found: {submission}")
    archive_zip = submission / "archive.zip"
    inflate_sh = submission / "inflate.sh"
    if not archive_zip.is_file():
        raise FileNotFoundError(f"archive.zip not found: {archive_zip}")
    if not inflate_sh.is_file():
        raise FileNotFoundError(f"inflate.sh not found: {inflate_sh}")
    inflated_dir = submission / "inflated"
    if inflated_dir.exists():
        raise FileExistsError(
            f"refusing to run with pre-existing inflated output; certify or remove first: {inflated_dir}"
        )
    free_bytes = shutil.disk_usage(submission).free
    if free_bytes < min_free_bytes:
        raise OSError(
            f"insufficient free space for upstream eval: free={free_bytes} required={min_free_bytes}"
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = submission / "report.txt"
    if report_path.exists():
        report_path.unlink()
    stdout_path = artifact_dir / "upstream_evaluate_stdout.txt"
    stderr_path = artifact_dir / "upstream_evaluate_stderr.txt"
    report_copy_path = artifact_dir / "report.txt"

    submission_manifest = _file_manifest_excluding(
        submission,
        excluded_top_level={"archive", "inflated"},
    )
    archive_row = {
        "path": archive_zip.as_posix(),
        "bytes": int(archive_zip.stat().st_size),
        "sha256": _sha256_file(archive_zip),
    }
    command = [
        "bash",
        str(upstream_root / "evaluate.sh"),
        "--submission-dir",
        str(submission),
        "--device",
        device,
    ]
    env = _upstream_env(upstream_root) if require_upstream_venv else os.environ.copy()
    env["COMMA_CHALLENGE_ROOT"] = str(upstream_root)
    python_environment = {
        "require_upstream_venv": require_upstream_venv,
        "python_executable": (
            str(upstream_root / ".venv" / "bin" / "python")
            if require_upstream_venv
            else sys.executable
        ),
        "comma_challenge_root": str(upstream_root),
    }

    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    wall_seconds = time.monotonic() - started
    _write_text(stdout_path, completed.stdout)
    _write_text(stderr_path, completed.stderr)

    blockers: list[str] = []
    if completed.returncode != 0:
        blockers.append(f"upstream_evaluate_returncode_nonzero:{completed.returncode}")

    parsed = _parse_report_if_present(report_path, blockers)
    copied_report: str | None = None
    if report_path.is_file():
        shutil.copy2(report_path, report_copy_path)
        copied_report = report_copy_path.as_posix()

    inflated_manifest = (
        _file_manifest(inflated_dir)
        if inflated_dir.is_dir()
        else {
            "root": inflated_dir.as_posix(),
            "file_count": 0,
            "bytes": 0,
            "tree_sha256": None,
            "files": [],
        }
    )
    if completed.returncode == 0 and int(inflated_manifest["file_count"]) <= 0:
        blockers.append("upstream_inflated_outputs_missing")

    if completed.returncode == 0:
        if keep_inflated:
            inflated_cleanup = "retained_by_request_after_success"
            inflated_retained = True
        else:
            shutil.rmtree(inflated_dir, ignore_errors=True)
            inflated_cleanup = "deleted_after_success_with_manifest_certificate"
            inflated_retained = False
    else:
        inflated_cleanup = (
            "retained_after_failed_evaluation_for_diagnosis"
            if inflated_dir.exists()
            else "no_inflated_output_after_failed_evaluation"
        )
        inflated_retained = inflated_dir.exists()

    return ExternalEvaluationSummary(
        schema="comma_lab.external_upstream_evaluation.v1",
        generated_at_utc=datetime.now(UTC).isoformat(),
        axis_tag=f"[upstream-{device}:false-authority]",
        submission_dir=submission.as_posix(),
        upstream_root=upstream_root.as_posix(),
        device=device,
        command=command,
        returncode=int(completed.returncode),
        wall_seconds=wall_seconds,
        report_path=report_path.as_posix(),
        report_copy_path=copied_report,
        stdout_path=stdout_path.as_posix(),
        stderr_path=stderr_path.as_posix(),
        python_environment=python_environment,
        archive_zip=archive_row,
        submission_manifest_before_eval=submission_manifest,
        inflated_outputs_manifest={
            **inflated_manifest,
            "certified_rebuildable": completed.returncode == 0,
            "rebuild_command": command,
            "source_archive_zip_sha256": archive_row["sha256"],
            "source_submission_manifest_tree_sha256": submission_manifest["tree_sha256"],
        },
        parsed_report=parsed,
        score_claim=False,
        score_claim_valid=False,
        promotion_eligible=False,
        promotable=False,
        rank_or_kill_eligible=False,
        ready_for_exact_eval_dispatch=False,
        dispatch_attempted=False,
        gpu_launched=device == "cuda",
        inflated_dir=inflated_dir.as_posix(),
        inflated_dir_retained=inflated_retained,
        inflated_dir_cleanup=inflated_cleanup,
        blockers=blockers,
    )
