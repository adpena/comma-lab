from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .install import install_payload_bytes, install_payload_manifest, install_submission
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

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


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
) -> EvaluationSummary:
    root = repo_root()
    upstream_root = upstream_root or default_upstream_root()
    source_submission_dir = root / "submissions" / name
    submission_dir = upstream_root / "submissions" / name

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
    _run([
        "bash",
        str(evaluate_sh),
        "--submission-dir",
        str(submission_dir),
        "--device",
        device,
    ], cwd=root, env=env)

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
    )
