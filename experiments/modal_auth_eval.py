"""Run canonical CUDA contest auth eval on Modal T4.

This wrapper is intentionally thin: Modal only supplies the CUDA host. The
score path remains the repository's canonical evaluator:

    archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda

The wrapper fails closed. It does not retry on CPU, does not call
inflate_renderer.py directly, and does not mark results promotion-eligible.
Promotion/ranking still requires the normal adjudication step over the
harvested ``contest_auth_eval.json`` and the exact local archive bytes.

Usage:
    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run experiments/modal_auth_eval.py \
        --archive experiments/results/.../point_004_eps_p2.zip \
        --output-dir experiments/results/modal_auth_eval/point_004_eps_p2
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any

import modal

from tac.repo_io import json_text, read_json, sha256_file, write_json

APP_NAME = "comma-auth-eval"
REMOTE_REPO = Path("/workspace/pact")
REMOTE_OUT = Path("/tmp/modal_auth_eval")
REMOTE_WORK_ROOT = Path("/root/modal_auth_eval_work")
REQUIRED_SAMPLES = 600
DALI_DISABLE_NVML_VALUE = "1"
REMOTE_PYTHONPATH = f"{REMOTE_REPO / 'src'}:{REMOTE_REPO / 'upstream'}:{REMOTE_REPO}"
SKIPPED_RUNTIME_UPLOAD_FILENAMES = {".DS_Store"}
SENSITIVE_RUNTIME_UPLOAD_NAMES = {
    ".env",
    ".env.local",
    ".netrc",
    "authorized_keys",
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
SENSITIVE_RUNTIME_UPLOAD_SUBSTRINGS = (
    "apikey",
    "api_key",
    "credential",
    "private_key",
    "secret",
    "token",
)

app = modal.App(APP_NAME)


base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ca-certificates",
        "curl",
        "ffmpeg",
        "git",
        "libglib2.0-0",
        "libgl1",
        "unzip",
        "xz-utils",
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "safetensors",
        "einops",
        "segmentation-models-pytorch",
        "av",
        "click",
        "nvidia-dali-cuda120==1.52.0",
        "tqdm",
        "timm",
        "scipy",
        "numpy<2.0",
        "Pillow",
        "pydantic>=2.0",
        "brotli>=1.0",
        "constriction>=0.4,<0.5",
        "pyppmd>=1.3,<2.0",
        "cryptography>=41.0",
        extra_index_url="https://pypi.nvidia.com",
    )
    .run_commands(
        "curl -sL https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz -o /tmp/ffmpeg-master.tar.xz",
        "cd /opt && tar xf /tmp/ffmpeg-master.tar.xz",
        "ln -sf /opt/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ffmpeg-master",
        "ln -sf /opt/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ffmpeg-new",
        "/usr/local/bin/ffmpeg-master -hide_banner -h filter=scale 2>&1 | grep -q in_primaries || (echo FATAL: ffmpeg-master lacks in_primaries; exit 1)",
        "/usr/local/bin/ffmpeg-master -encoders 2>&1 | grep -qi svtav1 || (echo FATAL: ffmpeg-master lacks libsvtav1; exit 1)",
        "rm /tmp/ffmpeg-master.tar.xz",
    )
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "ln -sf /root/.local/bin/uv /usr/local/bin/uv",
    )
)


eval_image = (
    base_image
    .env(
        {
            "DALI_DISABLE_NVML": DALI_DISABLE_NVML_VALUE,
            "PYTHONPATH": REMOTE_PYTHONPATH,
        }
    )
    .add_local_dir("src", remote_path=str(REMOTE_REPO / "src"))
    .add_local_dir("upstream", remote_path=str(REMOTE_REPO / "upstream"))
    .add_local_dir(
        "submissions/robust_current",
        remote_path=str(REMOTE_REPO / "submissions/robust_current"),
    )
    .add_local_dir(
        "experiments/public_runtime_adapters",
        remote_path=str(REMOTE_REPO / "experiments/public_runtime_adapters"),
    )
    .add_local_dir(
        "experiments/results/public_pr95_intake_20260504_codex",
        remote_path=str(REMOTE_REPO / "experiments/results/public_pr95_intake_20260504_codex"),
    )
    .add_local_dir(
        "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source",
        remote_path=str(
            REMOTE_REPO
            / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source"
        ),
    )
    .add_local_file(
        "experiments/contest_auth_eval.py",
        remote_path=str(REMOTE_REPO / "experiments/contest_auth_eval.py"),
    )
    .add_local_file("pyproject.toml", remote_path=str(REMOTE_REPO / "pyproject.toml"))
    .add_local_file("uv.lock", remote_path=str(REMOTE_REPO / "uv.lock"))
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_sha256_path = sha256_file


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json_text(payload).encode("utf-8")


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return label or "archive"


def _runtime_upload_skip_reason(rel: str) -> str | None:
    path = Path(rel)
    if path.name in SKIPPED_RUNTIME_UPLOAD_FILENAMES:
        return "ignored host metadata"
    if path.suffix == ".pyc" or "__pycache__" in path.parts:
        return "ignored python bytecode cache"
    return None


def _validate_runtime_upload_file(path: Path, rel: str) -> None:
    rel_path = Path(rel)
    if path.is_symlink():
        raise ValueError(f"refusing symlink in uploaded runtime tree: {rel}")
    for part in rel_path.parts:
        if part.startswith("."):
            raise ValueError(f"refusing hidden file or directory in uploaded runtime tree: {rel}")
    lowered_parts = {part.lower() for part in rel_path.parts}
    if lowered_parts & SENSITIVE_RUNTIME_UPLOAD_NAMES:
        raise ValueError(f"refusing secret-looking file in uploaded runtime tree: {rel}")
    lowered_rel = rel.lower()
    if any(marker in lowered_rel for marker in SENSITIVE_RUNTIME_UPLOAD_SUBSTRINGS):
        raise ValueError(f"refusing secret-looking file in uploaded runtime tree: {rel}")


def _submission_dir_zip_bytes(submission_dir: Path) -> bytes:
    """Return a deterministic transport zip for an uploaded runtime tree."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(p for p in submission_dir.rglob("*") if p.is_file()):
            rel = path.relative_to(submission_dir).as_posix()
            if _runtime_upload_skip_reason(rel):
                continue
            _validate_runtime_upload_file(path, rel)
            info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return buffer.getvalue()


def _probe_cuda_environment() -> dict[str, Any]:
    import shutil
    import subprocess
    import sys
    import time

    preflight: dict[str, Any] = {
        "schema_version": 1,
        "tool": "experiments/modal_auth_eval.py",
        "app": APP_NAME,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "device_required": "cuda",
        "score_claim": False,
        "promotion_eligible": False,
    }

    try:
        import torch

        preflight["torch_version"] = torch.__version__
        preflight["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        preflight["torch_cuda_available"] = bool(torch.cuda.is_available())
        preflight["torch_cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            preflight["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
            preflight["torch_cuda_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as exc:  # pragma: no cover - remote diagnostic path
        preflight["torch_probe_error"] = repr(exc)
        preflight["torch_cuda_available"] = False

    try:
        import nvidia.dali as dali

        preflight["nvidia_dali_import_ok"] = True
        preflight["nvidia_dali_version"] = getattr(dali, "__version__", None)
    except Exception as exc:  # pragma: no cover - remote diagnostic path
        preflight["nvidia_dali_import_ok"] = False
        preflight["nvidia_dali_import_error"] = repr(exc)

    nvidia_smi = shutil.which("nvidia-smi")
    preflight["nvidia_smi_path"] = nvidia_smi
    if nvidia_smi:
        try:
            preflight["nvidia_smi_query"] = subprocess.check_output(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=15,
            ).strip()
        except Exception as exc:  # pragma: no cover - remote diagnostic path
            preflight["nvidia_smi_error"] = repr(exc)

    return preflight


def _validate_contest_result(
    payload: dict[str, Any],
    *,
    expected_archive_sha256: str,
    expected_archive_size_bytes: int,
) -> list[str]:
    errors: list[str] = []
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return ["contest_auth_eval.json missing provenance object"]

    if provenance.get("device") != "cuda":
        errors.append(f"provenance.device={provenance.get('device')!r}, expected 'cuda'")
    if provenance.get("cuda_available") is not True:
        errors.append("provenance.cuda_available is not true")
    if provenance.get("archive_sha256") != expected_archive_sha256:
        errors.append(
            "provenance.archive_sha256 mismatch: "
            f"{provenance.get('archive_sha256')!r} != {expected_archive_sha256!r}"
        )
    if payload.get("archive_size_bytes") != expected_archive_size_bytes:
        errors.append(
            "archive_size_bytes mismatch: "
            f"{payload.get('archive_size_bytes')!r} != {expected_archive_size_bytes!r}"
        )
    if payload.get("n_samples") != REQUIRED_SAMPLES:
        errors.append(f"n_samples={payload.get('n_samples')!r}, expected {REQUIRED_SAMPLES}")

    score = payload.get("score_recomputed_from_components")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)):
        errors.append("score_recomputed_from_components is missing or non-finite")

    return errors


def _collect_artifacts(out_dir: Path, work_dir: Path) -> dict[str, bytes]:
    artifacts: dict[str, bytes] = {}
    for path in (
        out_dir / "modal_cuda_preflight.json",
        out_dir / "modal_cuda_auth_eval_validation.json",
        out_dir / "contest_auth_eval.stdout.log",
        out_dir / "contest_auth_eval.stderr.log",
        work_dir / "contest_auth_eval.json",
        work_dir / "provenance.json",
        work_dir / "report.txt",
    ):
        if path.is_file():
            artifacts[path.name] = path.read_bytes()
    return artifacts


def _run_auth_eval_inner(
    *,
    archive_bytes: bytes,
    archive_sha256: str,
    archive_size_bytes: int,
    inflate_sh_rel: str,
    submission_dir_zip_bytes: bytes | None,
    submission_dir_zip_sha256: str | None,
    inflate_timeout: int,
    evaluate_timeout: int,
) -> dict[str, Any]:
    import os
    import shutil
    import subprocess
    import sys
    import time

    out_dir = REMOTE_OUT
    work_dir = REMOTE_WORK_ROOT / "eval_work"
    archive_path = out_dir / "archive.zip"
    uv_env = out_dir / "uv_project_env"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    if REMOTE_WORK_ROOT.exists():
        shutil.rmtree(REMOTE_WORK_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    os.environ["DALI_DISABLE_NVML"] = DALI_DISABLE_NVML_VALUE
    preflight = _probe_cuda_environment()
    preflight.update(
        {
            "archive_sha256": archive_sha256,
            "archive_size_bytes": archive_size_bytes,
            "inflate_sh_rel": inflate_sh_rel,
            "submission_dir_zip_sha256": submission_dir_zip_sha256,
            "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
        }
    )
    write_json(out_dir / "modal_cuda_preflight.json", preflight)

    if preflight.get("torch_cuda_available") is not True:
        validation = {
            "schema_version": 1,
            "passed": False,
            "returncode": 3,
            "error": "Modal runtime has no CUDA device; refusing CPU fallback",
            "score_claim": False,
            "promotion_eligible": False,
        }
        (out_dir / "modal_cuda_auth_eval_validation.json").write_bytes(_json_bytes(validation))
        return {
            **validation,
            "artifacts": _collect_artifacts(out_dir, work_dir),
        }

    if preflight.get("nvidia_dali_import_ok") is not True:
        validation = {
            "schema_version": 1,
            "passed": False,
            "returncode": 4,
            "error": "nvidia.dali import failed; refusing non-DALI CUDA eval",
            "score_claim": False,
            "promotion_eligible": False,
        }
        (out_dir / "modal_cuda_auth_eval_validation.json").write_bytes(_json_bytes(validation))
        return {
            **validation,
            "artifacts": _collect_artifacts(out_dir, work_dir),
        }

    archive_path.write_bytes(archive_bytes)
    observed_sha = _sha256_path(archive_path)
    observed_size = archive_path.stat().st_size
    if observed_sha != archive_sha256 or observed_size != archive_size_bytes:
        validation = {
            "schema_version": 1,
            "passed": False,
            "returncode": 5,
            "error": "uploaded archive custody mismatch",
            "expected_archive_sha256": archive_sha256,
            "observed_archive_sha256": observed_sha,
            "expected_archive_size_bytes": archive_size_bytes,
            "observed_archive_size_bytes": observed_size,
            "score_claim": False,
            "promotion_eligible": False,
        }
        (out_dir / "modal_cuda_auth_eval_validation.json").write_bytes(_json_bytes(validation))
        return {
            **validation,
            "artifacts": _collect_artifacts(out_dir, work_dir),
        }

    if submission_dir_zip_bytes is not None:
        if not submission_dir_zip_sha256:
            validation = {
                "schema_version": 1,
                "passed": False,
                "returncode": 8,
                "error": "submission_dir_zip_sha256 missing for uploaded runtime tree",
                "score_claim": False,
                "promotion_eligible": False,
            }
            (out_dir / "modal_cuda_auth_eval_validation.json").write_bytes(_json_bytes(validation))
            return {**validation, "artifacts": _collect_artifacts(out_dir, work_dir)}
        runtime_zip = out_dir / "submission_dir.zip"
        runtime_root = out_dir / "submission_dir"
        runtime_zip.write_bytes(submission_dir_zip_bytes)
        observed_runtime_sha = _sha256_path(runtime_zip)
        if observed_runtime_sha != submission_dir_zip_sha256:
            validation = {
                "schema_version": 1,
                "passed": False,
                "returncode": 9,
                "error": "uploaded submission_dir.zip custody mismatch",
                "expected_submission_dir_zip_sha256": submission_dir_zip_sha256,
                "observed_submission_dir_zip_sha256": observed_runtime_sha,
                "score_claim": False,
                "promotion_eligible": False,
            }
            (out_dir / "modal_cuda_auth_eval_validation.json").write_bytes(_json_bytes(validation))
            return {**validation, "artifacts": _collect_artifacts(out_dir, work_dir)}
        from tac.submission_archive import safe_extract_zip

        safe_extract_zip(runtime_zip, runtime_root)
        inflate_sh_path = (runtime_root / inflate_sh_rel).resolve()
        runtime_base = runtime_root.resolve()
        runtime_base_name = "uploaded submission_dir"
    else:
        inflate_sh_path = (REMOTE_REPO / inflate_sh_rel).resolve()
        runtime_base = REMOTE_REPO.resolve()
        runtime_base_name = "remote repo"
    if not str(inflate_sh_path).startswith(str(runtime_base) + "/") and inflate_sh_path != runtime_base:
        validation = {
            "schema_version": 1,
            "passed": False,
            "returncode": 6,
            "error": f"inflate_sh path escapes {runtime_base_name}: {inflate_sh_rel}",
            "score_claim": False,
            "promotion_eligible": False,
        }
        (out_dir / "modal_cuda_auth_eval_validation.json").write_bytes(_json_bytes(validation))
        return {**validation, "artifacts": _collect_artifacts(out_dir, work_dir)}
    if not inflate_sh_path.is_file():
        validation = {
            "schema_version": 1,
            "passed": False,
            "returncode": 7,
            "error": f"inflate_sh not found in Modal image: {inflate_sh_rel}",
            "score_claim": False,
            "promotion_eligible": False,
        }
        (out_dir / "modal_cuda_auth_eval_validation.json").write_bytes(_json_bytes(validation))
        return {**validation, "artifacts": _collect_artifacts(out_dir, work_dir)}

    cmd = [
        sys.executable,
        "-u",
        str(REMOTE_REPO / "experiments/contest_auth_eval.py"),
        "--archive",
        str(archive_path),
        "--inflate-sh",
        str(inflate_sh_path),
        "--upstream-dir",
        str(REMOTE_REPO / "upstream"),
        "--video-names-file",
        str(REMOTE_REPO / "upstream/public_test_video_names.txt"),
        "--device",
        "cuda",
        "--keep-work-dir",
        "--work-dir",
        str(work_dir),
        "--inflate-timeout",
        str(int(inflate_timeout)),
        "--evaluate-timeout",
        str(int(evaluate_timeout)),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": REMOTE_PYTHONPATH,
        "FFMPEG_BIN": "/usr/local/bin/ffmpeg-master",
        "UV_BIN": "/usr/local/bin/uv",
        "UV_LINK_MODE": "copy",
        "UV_PROJECT_ENVIRONMENT": str(uv_env),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        "DALI_DISABLE_NVML": DALI_DISABLE_NVML_VALUE,
    }

    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REMOTE_REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=int(inflate_timeout) + int(evaluate_timeout) + 600,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        (out_dir / "contest_auth_eval.stdout.log").write_text(stdout)
        (out_dir / "contest_auth_eval.stderr.log").write_text(stderr)
        validation = {
            "schema_version": 1,
            "passed": False,
            "returncode": 124,
            "modal_elapsed_seconds": elapsed,
            "command": cmd,
            "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
            "error": "outer Modal contest_auth_eval timeout expired",
            "score_claim": False,
            "promotion_eligible": False,
            "adjudication_required": True,
            "allowed_use": ["debug", "no_score_claim", "no_promotion"],
        }
        (out_dir / "modal_cuda_auth_eval_validation.json").write_bytes(_json_bytes(validation))
        return {
            **validation,
            "artifacts": _collect_artifacts(out_dir, work_dir),
        }
    elapsed = time.monotonic() - started
    (out_dir / "contest_auth_eval.stdout.log").write_text(proc.stdout)
    (out_dir / "contest_auth_eval.stderr.log").write_text(proc.stderr)
    print(proc.stdout[-4096:] if proc.stdout else "")
    if proc.stderr:
        print(proc.stderr[-2048:], file=sys.stderr)

    result_json = work_dir / "contest_auth_eval.json"
    payload: dict[str, Any] | None = None
    validation_errors: list[str] = []
    if result_json.is_file():
        try:
            payload = read_json(result_json)
        except json.JSONDecodeError as exc:
            validation_errors.append(f"contest_auth_eval.json is not valid JSON: {exc}")
    else:
        validation_errors.append("contest_auth_eval.json was not produced")

    if payload is not None:
        validation_errors.extend(
            _validate_contest_result(
                payload,
                expected_archive_sha256=archive_sha256,
                expected_archive_size_bytes=archive_size_bytes,
            )
        )

    passed = proc.returncode == 0 and not validation_errors
    validation = {
        "schema_version": 1,
        "passed": passed,
        "returncode": proc.returncode if passed else (proc.returncode or 10),
        "modal_elapsed_seconds": elapsed,
        "command": cmd,
        "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
        "expected_archive_sha256": archive_sha256,
        "expected_archive_size_bytes": archive_size_bytes,
        "inflate_sh_rel": inflate_sh_rel,
        "submission_dir_zip_sha256": submission_dir_zip_sha256,
        "validation_errors": validation_errors,
        "score_claim": bool(passed),
        "promotion_eligible": False,
        "adjudication_required": True,
        "allowed_use": (
            ["cuda_auth_eval_review", "adjudication_required"]
            if passed
            else ["debug", "no_score_claim", "no_promotion"]
        ),
    }
    if payload is not None:
        validation.update(
            {
                "final_score": payload.get("final_score"),
                "score_recomputed_from_components": payload.get(
                    "score_recomputed_from_components"
                ),
                "avg_posenet_dist": payload.get("avg_posenet_dist"),
                "avg_segnet_dist": payload.get("avg_segnet_dist"),
                "n_samples": payload.get("n_samples"),
                "archive_size_bytes": payload.get("archive_size_bytes"),
                "provenance_device": (payload.get("provenance") or {}).get("device")
                if isinstance(payload.get("provenance"), dict)
                else None,
                "gpu_model": (payload.get("provenance") or {}).get("gpu_model")
                if isinstance(payload.get("provenance"), dict)
                else None,
                "gpu_t4_match": (payload.get("provenance") or {}).get("gpu_t4_match")
                if isinstance(payload.get("provenance"), dict)
                else None,
            }
        )
    (out_dir / "modal_cuda_auth_eval_validation.json").write_bytes(_json_bytes(validation))

    return {
        **validation,
        "artifacts": _collect_artifacts(out_dir, work_dir),
    }


@app.function(
    image=eval_image,
    gpu="T4",
    timeout=4800,
)
def run_auth_eval(
    archive_bytes: bytes,
    archive_sha256: str,
    archive_size_bytes: int,
    inflate_sh_rel: str = "submissions/robust_current/inflate.sh",
    submission_dir_zip_bytes: bytes | None = None,
    submission_dir_zip_sha256: str | None = None,
    inflate_timeout: int = 1800,
    evaluate_timeout: int = 1800,
) -> dict[str, Any]:
    """Run the canonical CUDA auth eval on Modal T4."""

    return _run_auth_eval_inner(
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size_bytes,
        inflate_sh_rel=inflate_sh_rel,
        submission_dir_zip_bytes=submission_dir_zip_bytes,
        submission_dir_zip_sha256=submission_dir_zip_sha256,
        inflate_timeout=inflate_timeout,
        evaluate_timeout=evaluate_timeout,
    )


@app.function(
    image=eval_image,
    gpu="A100",
    timeout=4800,
)
def run_auth_eval_a100(
    archive_bytes: bytes,
    archive_sha256: str,
    archive_size_bytes: int,
    inflate_sh_rel: str = "submissions/robust_current/inflate.sh",
    submission_dir_zip_bytes: bytes | None = None,
    submission_dir_zip_sha256: str | None = None,
    inflate_timeout: int = 1800,
    evaluate_timeout: int = 1800,
) -> dict[str, Any]:
    """Run the canonical CUDA auth eval on Modal A100."""

    return _run_auth_eval_inner(
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size_bytes,
        inflate_sh_rel=inflate_sh_rel,
        submission_dir_zip_bytes=submission_dir_zip_bytes,
        submission_dir_zip_sha256=submission_dir_zip_sha256,
        inflate_timeout=inflate_timeout,
        evaluate_timeout=evaluate_timeout,
    )


@app.function(
    image=eval_image,
    gpu="H100",
    timeout=4800,
)
def run_auth_eval_h100(
    archive_bytes: bytes,
    archive_sha256: str,
    archive_size_bytes: int,
    inflate_sh_rel: str = "submissions/robust_current/inflate.sh",
    submission_dir_zip_bytes: bytes | None = None,
    submission_dir_zip_sha256: str | None = None,
    inflate_timeout: int = 1800,
    evaluate_timeout: int = 1800,
) -> dict[str, Any]:
    """Run the canonical CUDA auth eval on Modal H100."""

    return _run_auth_eval_inner(
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size_bytes,
        inflate_sh_rel=inflate_sh_rel,
        submission_dir_zip_bytes=submission_dir_zip_bytes,
        submission_dir_zip_sha256=submission_dir_zip_sha256,
        inflate_timeout=inflate_timeout,
        evaluate_timeout=evaluate_timeout,
    )


@app.local_entrypoint()
def main(
    archive: str = "/tmp/modal_submission/archive.zip",
    output_dir: str = "",
    inflate_sh: str = "submissions/robust_current/inflate.sh",
    submission_dir: str = "",
    gpu: str = "T4",
    inflate_timeout: int = 1800,
    evaluate_timeout: int = 1800,
) -> None:
    """Upload an archive and harvest Modal CUDA auth-eval artifacts."""

    archive_path = Path(archive).resolve()
    if not archive_path.is_file():
        raise SystemExit(f"FATAL: archive not found: {archive_path}")

    archive_bytes = archive_path.read_bytes()
    archive_sha256 = _sha256_bytes(archive_bytes)
    archive_size_bytes = len(archive_bytes)
    submission_dir_zip: bytes | None = None
    submission_dir_zip_sha256: str | None = None
    submission_dir_path = Path(submission_dir).resolve() if submission_dir else None
    inflate_sh_path = Path(inflate_sh)
    if submission_dir_path is not None:
        if not submission_dir_path.is_dir():
            raise SystemExit(f"FATAL: --submission-dir is not a directory: {submission_dir_path}")
        if inflate_sh_path.is_absolute():
            try:
                inflate_sh_rel = str(inflate_sh_path.resolve().relative_to(submission_dir_path))
            except ValueError as exc:
                raise SystemExit(
                    "FATAL: absolute --inflate-sh must be inside --submission-dir "
                    f"when uploading a runtime tree: {inflate_sh_path}"
                ) from exc
        else:
            inflate_sh_rel = str(inflate_sh_path)
        if ".." in Path(inflate_sh_rel).parts:
            raise SystemExit(f"FATAL: --inflate-sh must not contain parent traversal: {inflate_sh_rel}")
        if not (submission_dir_path / inflate_sh_rel).is_file():
            raise SystemExit(
                f"FATAL: --inflate-sh {inflate_sh_rel!r} not found under --submission-dir "
                f"{submission_dir_path}"
            )
        submission_dir_zip = _submission_dir_zip_bytes(submission_dir_path)
        submission_dir_zip_sha256 = _sha256_bytes(submission_dir_zip)
    else:
        if inflate_sh_path.is_absolute():
            try:
                inflate_sh_rel = str(inflate_sh_path.resolve().relative_to(Path.cwd().resolve()))
            except ValueError as exc:
                raise SystemExit(
                    f"FATAL: --inflate-sh must be relative to repo root or inside it: {inflate_sh_path}"
                ) from exc
        else:
            inflate_sh_rel = str(inflate_sh_path)
        if ".." in Path(inflate_sh_rel).parts:
            raise SystemExit(f"FATAL: --inflate-sh must not contain parent traversal: {inflate_sh_rel}")
    label = _safe_label(archive_path.stem)
    out_dir = (
        Path(output_dir).resolve()
        if output_dir
        else Path("experiments/results/modal_auth_eval") / f"{label}_{archive_sha256[:12]}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    local_summary = {
        "schema_version": 1,
        "tool": "experiments/modal_auth_eval.py",
        "app": APP_NAME,
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size_bytes,
        "inflate_sh": inflate_sh_rel,
        "submission_dir": str(submission_dir_path) if submission_dir_path else None,
        "submission_dir_zip_sha256": submission_dir_zip_sha256,
        "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
        "score_claim": False,
        "promotion_eligible": False,
        "adjudication_required": True,
    }
    write_json(out_dir / "modal_cuda_auth_eval_local_request.json", local_summary)

    print(
        f"Uploading {archive_size_bytes:,} bytes to Modal {gpu} for CUDA auth eval "
        f"(sha256={archive_sha256})..."
    )
    gpu_key = gpu.upper()
    if gpu_key == "T4":
        auth_eval_fn = run_auth_eval
    elif gpu_key in {"A100", "A100-40GB", "A100-80GB"}:
        auth_eval_fn = run_auth_eval_a100
    elif gpu_key in {"H100", "H100-80GB"}:
        auth_eval_fn = run_auth_eval_h100
    else:
        raise SystemExit(f"FATAL: unsupported --gpu {gpu!r}; use T4, A100, or H100")

    result = auth_eval_fn.remote(
        archive_bytes,
        archive_sha256,
        archive_size_bytes,
        inflate_sh_rel,
        submission_dir_zip,
        submission_dir_zip_sha256,
        int(inflate_timeout),
        int(evaluate_timeout),
    )

    artifacts = result.pop("artifacts", {})
    for name, data in sorted(artifacts.items()):
        (out_dir / name).write_bytes(data)

    result["local_output_dir"] = str(out_dir)
    result["archive_path"] = str(archive_path)
    result["archive_sha256"] = archive_sha256
    result["archive_size_bytes"] = archive_size_bytes
    result["inflate_sh"] = inflate_sh_rel
    write_json(out_dir / "modal_cuda_auth_eval_result.json", result)

    print("=" * 60)
    if result.get("passed"):
        print("MODAL CUDA AUTH EVAL PASSED PATH VALIDATION")
        print(f"  Score recomputed: {result.get('score_recomputed_from_components')}")
        print(f"  PoseNet dist:     {result.get('avg_posenet_dist')}")
        print(f"  SegNet dist:      {result.get('avg_segnet_dist')}")
        print(f"  Archive bytes:    {result.get('archive_size_bytes')}")
        print("  Promotion:        not eligible until adjudication gates pass")
    else:
        print("MODAL CUDA AUTH EVAL FAILED CLOSED")
        print(f"  Error:            {result.get('error')}")
        print(f"  Validation:       {result.get('validation_errors')}")
    print(f"  Artifacts:        {out_dir}")
    print("=" * 60)

    if not result.get("passed"):
        raise SystemExit(int(result.get("returncode") or 1))
