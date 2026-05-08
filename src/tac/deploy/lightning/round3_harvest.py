"""Focused helpers for round-3 Lightning job harvest scripts.

These helpers are intentionally side-effect light: they derive candidate
artifact roots, normalize contest-auth-eval fields, and classify terminal
no-score artifacts without touching provider state.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path

from tac.deploy.lightning.batch_jobs import (
    lightning_sdk_artifact_path,
    lightning_sdk_persisted_studio_output_dir,
)

NO_SCORE_FAILURE_FILENAME = "lightning_no_score_failure_classification.json"
_LOG_TAIL_CHARS = 4096
_ARCHIVE_SHA_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_CONTEST_ARCHIVE_DENOMINATOR = 37_545_489
_SCORE_RECOMPUTE_ABS_TOL = 1e-7


def lightning_round3_remote_artifact_dirs(
    *,
    remote_pact: str,
    job_name: str,
) -> list[str]:
    """Return live-Studio and persisted SDK artifact dirs for a job.

    Lightning jobs write into the live Studio workspace while running, but
    terminal Batch Job artifacts are also exposed under
    ``/teamspace/jobs/<sdk-job-name>/artifacts``. Harvesters should try both
    so terminal artifacts are not lost when the live Studio path is empty or
    gone.
    """

    pact = str(remote_pact).rstrip("/")
    job = str(job_name).strip()
    if not pact:
        raise ValueError("remote_pact is required")
    if not job:
        raise ValueError("job_name is required")

    roots: list[str] = []

    def add(path: str | None) -> None:
        if not path:
            return
        normalized = str(path).rstrip("/") + "/"
        if normalized not in roots:
            roots.append(normalized)

    live_output = f"{pact}/experiments/results/lightning_batch/{job}"
    add(live_output)

    sdk_artifacts = lightning_sdk_artifact_path(job)
    add(
        lightning_sdk_persisted_studio_output_dir(
            sdk_artifact_path=sdk_artifacts,
            remote_output_dir=live_output,
        )
    )
    # Current round-3 jobs use the default Studio pact checkout, whose persisted
    # mirror is /teamspace/jobs/<job>/artifacts/pact/... . Keep this explicit
    # fallback even if callers passed an env-derived remote_pact.
    add(f"{sdk_artifacts}/pact/experiments/results/lightning_batch/{job}")
    return roots


def auth_eval_score(auth_eval: dict[str, object]) -> float | int | None:
    """Return the canonical contest score from a contest_auth_eval payload."""

    for key in ("score_recomputed_from_components", "canonical_score"):
        value = auth_eval.get(key)
        if _finite_number(value):
            return value
    return None


def auth_eval_archive_size(auth_eval: dict[str, object]) -> int | None:
    """Return charged archive bytes from a contest_auth_eval payload."""

    for key in ("archive_size_bytes", "archive_bytes"):
        value = auth_eval.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def sha256_file(path: Path) -> str:
    """Return SHA-256 for a local file."""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def contest_cuda_auth_eval_blockers(
    auth_eval: dict[str, object],
    *,
    expected_archive_bytes: int | None = None,
    expected_archive_sha256: str | None = None,
    expected_samples: int = 600,
    require_t4: bool = True,
) -> list[str]:
    """Return blockers that prevent a JSON from being score-grade CUDA evidence.

    This is intentionally stricter than "does the JSON contain a score". A
    harvester may emit ``[contest-CUDA]`` only when the archive bytes, archive
    SHA, recomputed component score, CUDA/T4 provenance, sample count, and
    runtime tree custody are all present and internally consistent.
    """

    blockers: list[str] = []
    archive_bytes_raw = auth_eval.get("archive_size_bytes")
    archive_bytes = (
        archive_bytes_raw
        if isinstance(archive_bytes_raw, int)
        and not isinstance(archive_bytes_raw, bool)
        and archive_bytes_raw > 0
        else None
    )
    score_raw = auth_eval.get("score_recomputed_from_components")
    score = score_raw if _finite_number(score_raw) else None
    pose = auth_eval.get("avg_posenet_dist")
    seg = auth_eval.get("avg_segnet_dist")
    rate = auth_eval.get("rate_unscaled")
    pose_contrib = auth_eval.get("score_pose_contribution")
    seg_contrib = auth_eval.get("score_seg_contribution")
    rate_contrib = auth_eval.get("score_rate_contribution")
    provenance = auth_eval.get("provenance")

    if not _finite_number(score):
        blockers.append("score_recomputed_from_components_missing_or_nonfinite")
    if auth_eval.get("canonical_score_source") != "score_recomputed_from_components":
        blockers.append("canonical_score_source_not_recomputed_from_components")
    if not _finite_number(pose):
        blockers.append("avg_posenet_dist_missing_or_nonfinite")
    if not _finite_number(seg):
        blockers.append("avg_segnet_dist_missing_or_nonfinite")
    if not _finite_number(rate):
        blockers.append("rate_unscaled_missing_or_nonfinite")
    if archive_bytes is None:
        blockers.append("archive_size_bytes_missing_or_invalid")
    if expected_archive_bytes is not None and archive_bytes != expected_archive_bytes:
        blockers.append("archive_size_bytes_mismatch_expected")

    if _finite_number(score) and _finite_number(pose) and _finite_number(seg) and archive_bytes is not None:
        has_contribs = (
            _finite_number(pose_contrib)
            and _finite_number(seg_contrib)
            and _finite_number(rate_contrib)
        )
        formula_seg = 100.0 * float(seg)
        formula_pose = math.sqrt(10.0 * float(pose))
        formula_rate = 25.0 * archive_bytes / _CONTEST_ARCHIVE_DENOMINATOR
        if has_contribs:
            if abs(float(seg_contrib) - formula_seg) > _SCORE_RECOMPUTE_ABS_TOL:
                blockers.append("score_seg_contribution_mismatch")
            if abs(float(pose_contrib) - formula_pose) > _SCORE_RECOMPUTE_ABS_TOL:
                blockers.append("score_pose_contribution_mismatch")
            if abs(float(rate_contrib) - formula_rate) > _SCORE_RECOMPUTE_ABS_TOL:
                blockers.append("score_rate_contribution_mismatch")
            recomputed = (
                float(seg_contrib)
                + float(pose_contrib)
                + float(rate_contrib)
            )
        else:
            recomputed = formula_seg + formula_pose + formula_rate
        if abs(float(score) - recomputed) > _SCORE_RECOMPUTE_ABS_TOL:
            blockers.append("score_recomputed_from_components_mismatch")

    if not isinstance(provenance, dict):
        blockers.append("provenance_missing_or_invalid")
        return blockers

    prov_archive_bytes = provenance.get("archive_size_bytes")
    if archive_bytes is not None and prov_archive_bytes != archive_bytes:
        blockers.append("provenance_archive_size_bytes_mismatch")

    archive_sha = provenance.get("archive_sha256")
    if not _valid_sha256(archive_sha):
        blockers.append("provenance_archive_sha256_missing_or_invalid")
    if (
        expected_archive_sha256
        and _valid_sha256(archive_sha)
        and str(archive_sha).lower() != expected_archive_sha256.lower()
    ):
        blockers.append("provenance_archive_sha256_mismatch_expected")

    if provenance.get("device") != "cuda":
        blockers.append("provenance_device_not_cuda")
    if provenance.get("cuda_available") is not True:
        blockers.append("provenance_cuda_available_not_true")
    cuda_device_count = provenance.get("cuda_device_count")
    if not isinstance(cuda_device_count, int) or isinstance(cuda_device_count, bool) or cuda_device_count < 1:
        blockers.append("provenance_cuda_device_count_missing_or_invalid")
    if require_t4 and provenance.get("gpu_t4_match") is not True:
        blockers.append("provenance_gpu_t4_match_not_true")

    n_samples = auth_eval.get("n_samples")
    if n_samples != expected_samples:
        blockers.append("n_samples_not_full_public_test")

    runtime_manifest = provenance.get("inflate_runtime_manifest")
    runtime_tree_sha256 = (
        runtime_manifest.get("runtime_tree_sha256")
        if isinstance(runtime_manifest, dict)
        else None
    )
    if not _valid_sha256(runtime_tree_sha256):
        blockers.append("runtime_tree_sha256_missing_or_invalid")
    return blockers


def require_contest_cuda_auth_eval(
    auth_eval: dict[str, object],
    *,
    expected_archive_bytes: int | None = None,
    expected_archive_sha256: str | None = None,
    expected_samples: int = 600,
    require_t4: bool = True,
) -> None:
    """Raise SystemExit unless ``auth_eval`` is strict score-grade CUDA evidence."""

    blockers = contest_cuda_auth_eval_blockers(
        auth_eval,
        expected_archive_bytes=expected_archive_bytes,
        expected_archive_sha256=expected_archive_sha256,
        expected_samples=expected_samples,
        require_t4=require_t4,
    )
    if blockers:
        raise SystemExit(
            "FATAL: refusing [contest-CUDA] evidence row; auth_eval custody blockers: "
            + ", ".join(blockers)
        )


def auth_eval_posenet_distortion(auth_eval: dict[str, object]) -> object:
    for key in ("avg_posenet_dist", "posenet_distortion", "pose_distortion"):
        value = auth_eval.get(key)
        if value is not None:
            return value
    return None


def auth_eval_segnet_distortion(auth_eval: dict[str, object]) -> object:
    for key in ("avg_segnet_dist", "segnet_distortion", "seg_distortion"):
        value = auth_eval.get(key)
        if value is not None:
            return value
    return None


def auth_eval_rate(auth_eval: dict[str, object]) -> object:
    for key in ("rate_unscaled", "rate"):
        value = auth_eval.get(key)
        if value is not None:
            return value
    return None


def classify_lightning_no_score_failure(artifact_dir: Path) -> dict[str, object]:
    """Classify terminal artifacts that do not contain a score JSON."""

    log_files = _collect_log_files(artifact_dir)
    log_text = "\n\n".join(_tail_log(path) for path in log_files)
    log_lower = log_text.lower()

    terminal_status = "failed_runtime_or_harness_no_auth_eval_json"
    failure_class = "runtime_or_harness_failure_before_score_json"
    failure_domain = "runtime"
    reason = (
        "contest_auth_eval.json is missing and harvested logs do not match a "
        "more precise round-3 dependency/runtime/train-contract signature"
    )

    if "modulenotfounderror" in log_lower or "no module named" in log_lower:
        missing_module = _missing_module_name(log_text)
        terminal_status = "failed_dependency_runtime_missing_module_before_score"
        failure_class = "dependency_runtime_missing_module_before_score"
        failure_domain = "dependency"
        if missing_module == "brotli":
            terminal_status = "failed_dependency_runtime_missing_brotli_before_score"
            failure_class = "dependency_runtime_missing_brotli_before_score"
        reason = (
            "harvested logs show a missing Python runtime dependency before "
            "contest_auth_eval.json could be written"
        )
    elif "q-faithful forward requires an explicit deployed pose tensor" in log_lower:
        terminal_status = "failed_train_contract_qfaithful_pose_missing_before_score"
        failure_class = "train_contract_qfaithful_pose_missing_before_score"
        failure_domain = "train_contract"
        reason = (
            "harvested logs show the Q-FAITHFUL training/eval contract rejected "
            "a missing deployed pose tensor before scoring"
        )
    elif (
        "contest_auth_eval rc=" in log_lower
        or "inflate failed" in log_lower
        or ("inflate" in log_lower and "returncode" in log_lower)
        or ("inflate.sh" in log_lower and "non-zero" in log_lower)
    ):
        terminal_status = "failed_runtime_inflate_or_auth_eval_before_score"
        failure_class = "runtime_inflate_or_auth_eval_failure_before_score"
        failure_domain = "runtime"
        reason = (
            "harvested logs show inflate/auth-eval runtime failure before "
            "contest_auth_eval.json could be written"
        )

    return {
        "schema_version": "lightning_round3_no_score_failure.v1",
        "classified_at_utc": dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "terminal_status": terminal_status,
        "failure_class": failure_class,
        "failure_domain": failure_domain,
        "reason": reason,
        "score_source": "none:missing_contest_auth_eval_json",
        "score_claim": False,
        "method_evidence": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "evidence_grade": "invalid",
        "artifact_dir": str(artifact_dir),
        "log_files": [str(path.relative_to(artifact_dir)) for path in log_files],
        "log_tail_snippet": log_text[-_LOG_TAIL_CHARS:],
        "recommended_action": (
            "Preserve the artifacts as a pre-score failure. Do not promote, "
            "rank, kill, or claim score without contest_auth_eval.json."
        ),
    }


def write_no_score_failure_classification(
    artifact_dir: Path,
    classification: dict[str, object],
) -> Path:
    path = artifact_dir / NO_SCORE_FAILURE_FILENAME
    path.write_text(json.dumps(classification, indent=2) + "\n", encoding="utf-8")
    return path


def _collect_log_files(artifact_dir: Path) -> list[Path]:
    names = ("auth_eval.log", "run.log", "train.log")
    seen: set[Path] = set()
    files: list[Path] = []
    for name in names:
        path = artifact_dir / name
        if path.is_file() and path not in seen:
            files.append(path)
            seen.add(path)
    for pattern in ("*.log", "**/*.log", "*.out", "**/*.out", "*.err", "**/*.err"):
        for path in sorted(artifact_dir.glob(pattern)):
            if path.is_file() and path not in seen:
                files.append(path)
                seen.add(path)
    return files


def _tail_log(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    rel = path.name
    return f"### {rel}\n{text[-_LOG_TAIL_CHARS:]}"


def _missing_module_name(log_text: str) -> str | None:
    marker = "No module named "
    for line in log_text.splitlines():
        if marker in line:
            raw = line.split(marker, 1)[1].strip().strip("'\"")
            return raw.split()[0].strip("'\".,:;").lower() or None
    return None


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _ARCHIVE_SHA_RE.fullmatch(value.strip()) is not None
