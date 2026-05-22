# SPDX-License-Identifier: MIT
"""Run canonical CUDA contest auth eval on Modal T4.

This wrapper is intentionally thin: Modal only supplies the CUDA host. The
score path remains the repository's canonical evaluator:

    archive.zip -> inflate.sh -> upstream/evaluate.py --device {cuda,cpu}

The wrapper fails closed. It does not retry on CPU, does not call
inflate_renderer.py directly, and does not mark results promotion-eligible.
Promotion/ranking still requires the normal adjudication step over the
harvested ``contest_auth_eval.json`` and the exact local archive bytes.

Usage:
    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run experiments/modal_auth_eval.py \
        --archive experiments/results/.../point_004_eps_p2.zip \
        --output-dir experiments/results/modal_auth_eval/point_004_eps_p2

Detached long runs must detach at BOTH layers: Modal CLI keeps the ephemeral
app alive, and the wrapper spawns the remote function.

    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach experiments/modal_auth_eval.py \
        --archive experiments/results/.../archive.zip \
        --output-dir experiments/results/modal_auth_eval/<run_id> \
        --detach --provider-detach-ack \
        --lane-id <lane> --instance-job-id <job>
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import modal

from tac.deploy.modal.auth_eval import (
    ClaimSpec,
    ModalArtifactWriteError,
    ModalAuthEvalPairingError,
    claim_modal_auth_eval_dispatch,
    fail_closed_remote_exception_result,
    function_call_id,
    materialize_modal_artifacts,
    modal_uploaded_submission_dir_runtime_manifest,
    prepare_modal_auth_eval_request,
    terminal_modal_auth_eval_claim,
    validate_modal_auth_eval_pairing,
    write_spawn_metadata,
)
from tac.deploy.modal.mount_ignore import ignore_generated_mount_path
from tac.repo_io import json_text, read_json, sha256_file, write_json

APP_NAME = "comma-auth-eval"
REMOTE_REPO = Path("/workspace/pact")
REMOTE_OUT = Path("/tmp/modal_auth_eval")
REMOTE_WORK_ROOT = Path("/root/modal_auth_eval_work")
AUTH_CACHE_VOLUME_NAME = "comma-auth-eval-cache-artifacts"
AUTH_CACHE_VOLUME_ROOT = Path("/modal_auth_cache")
REQUIRED_SAMPLES = 600
DALI_DISABLE_NVML_VALUE = "1"
REMOTE_PYTHONPATH = f"{REMOTE_REPO / 'src'}:{REMOTE_REPO / 'upstream'}:{REMOTE_REPO}"

app = modal.App(APP_NAME)
auth_cache_vol = modal.Volume.from_name(AUTH_CACHE_VOLUME_NAME, create_if_missing=True)


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
    # MODAL_MANUAL_MOUNT_OK:narrow auth-eval dispatcher; mounts public-PR intake clones + robust_current + contest_auth_eval; trainer-discovery N/A
    .add_local_dir(  # MODAL_MANUAL_MOUNT_OK:narrow auth-eval dispatcher
        "src",
        remote_path=str(REMOTE_REPO / "src"),
        ignore=ignore_generated_mount_path,
    )
    .add_local_dir(  # MODAL_MANUAL_MOUNT_OK:narrow auth-eval dispatcher
        "upstream",
        remote_path=str(REMOTE_REPO / "upstream"),
        ignore=ignore_generated_mount_path,
    )
    .add_local_dir(  # MODAL_MANUAL_MOUNT_OK:narrow auth-eval dispatcher; submissions subset
        "submissions/robust_current",
        remote_path=str(REMOTE_REPO / "submissions/robust_current"),
        ignore=ignore_generated_mount_path,
    )
    .add_local_dir(  # MODAL_MANUAL_MOUNT_OK:narrow auth-eval dispatcher; public-PR runtime adapters
        "experiments/public_runtime_adapters",
        remote_path=str(REMOTE_REPO / "experiments/public_runtime_adapters"),
        ignore=ignore_generated_mount_path,
    )
    .add_local_dir(  # MODAL_MANUAL_MOUNT_OK:narrow auth-eval dispatcher; public-PR95 intake clone
        "experiments/results/public_pr95_intake_20260504_codex",
        remote_path=str(REMOTE_REPO / "experiments/results/public_pr95_intake_20260504_codex"),
        ignore=ignore_generated_mount_path,
    )
    .add_local_dir(  # MODAL_MANUAL_MOUNT_OK:narrow auth-eval dispatcher; public-PR106 intake clone
        "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source",
        remote_path=str(
            REMOTE_REPO
            / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source"
        ),
        ignore=ignore_generated_mount_path,
    )
    .add_local_file(  # MODAL_MANUAL_MOUNT_OK:narrow auth-eval dispatcher; contest_auth_eval entry script
        "experiments/contest_auth_eval.py",
        remote_path=str(REMOTE_REPO / "experiments/contest_auth_eval.py"),
    )
    .add_local_file("pyproject.toml", remote_path=str(REMOTE_REPO / "pyproject.toml"))  # MODAL_MANUAL_MOUNT_OK:narrow auth-eval dispatcher
    .add_local_file("uv.lock", remote_path=str(REMOTE_REPO / "uv.lock"))  # MODAL_MANUAL_MOUNT_OK:narrow auth-eval dispatcher
)


_sha256_path = sha256_file


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json_text(payload).encode("utf-8")


def _expected_uploaded_runtime_tree_sha256(
    *,
    submission_dir_path: Path,
    inflate_sh_rel: str,
    remote_submission_dir: Path = REMOTE_OUT / "submission_dir",
) -> tuple[str, str]:
    from experiments.contest_auth_eval import _runtime_dependency_manifest

    local_inflate_sh = (submission_dir_path / inflate_sh_rel).resolve()
    local_manifest = _runtime_dependency_manifest(local_inflate_sh, Path("upstream"))
    projected = modal_uploaded_submission_dir_runtime_manifest(
        local_manifest,
        remote_submission_dir=str(remote_submission_dir),
    )
    return (
        str(projected.get("runtime_tree_sha256") or ""),
        str(projected.get("runtime_content_tree_sha256") or ""),
    )


def _validate_uploaded_runtime_tree_expectation(
    *,
    expected_runtime_tree_sha256: str,
    submission_dir_path: Path | None,
    inflate_sh_rel: str,
) -> None:
    if not expected_runtime_tree_sha256 or submission_dir_path is None:
        return
    remote_tree_sha256, content_tree_sha256 = _expected_uploaded_runtime_tree_sha256(
        submission_dir_path=submission_dir_path,
        inflate_sh_rel=inflate_sh_rel,
    )
    if expected_runtime_tree_sha256 != remote_tree_sha256:
        raise SystemExit(
            "FATAL: --expected-runtime-tree-sha256 does not match the Modal "
            "uploaded --submission-dir runtime tree. Modal extracts uploaded "
            f"runtimes under {REMOTE_OUT / 'submission_dir'}, so the expected "
            f"runtime_tree_sha256 is {remote_tree_sha256}; got "
            f"{expected_runtime_tree_sha256}. "
            f"runtime_content_tree_sha256={content_tree_sha256}"
        )


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
    expected_device: str = "cuda",
) -> list[str]:
    errors: list[str] = []
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return ["contest_auth_eval.json missing provenance object"]

    if provenance.get("device") != expected_device:
        errors.append(
            f"provenance.device={provenance.get('device')!r}, expected {expected_device!r}"
        )
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

    if expected_device == "cuda":
        grade = payload.get("evidence_grade")
        if grade != "contest-CUDA":
            errors.append(
                f"evidence_grade={grade!r}; expected 'contest-CUDA'"
            )
        if payload.get("score_axis") != "contest_cuda":
            errors.append(
                f"score_axis={payload.get('score_axis')!r}; expected 'contest_cuda'"
            )
        if payload.get("exact_cuda_eval_complete") is not True:
            errors.append("exact_cuda_eval_complete is not true")

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
        work_dir / "inflated_outputs_manifest.json",
        work_dir / "scorer_input_cache_hashes.json",
        work_dir / "provenance.json",
        work_dir / "report.txt",
        out_dir / "scorer_input_cache_tensor_volume_manifest.json",
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
    source_repo_commit: str,
    inflate_timeout: int,
    evaluate_timeout: int,
    scorer_device: str = "cuda",
    inflate_device_policy: str = "auto",
    inflate_env_overrides: tuple[str, ...] = (),
    expected_runtime_tree_sha256: str = "",
    scorer_input_cache_hashes: bool = False,
    scorer_input_cache_hash_batch_pairs: int = 8,
    scorer_input_cache_tensors: bool = False,
    scorer_input_cache_tensor_batch_pairs: int = 8,
    scorer_input_cache_tensor_large_pair_threshold: int = 64,
    allow_large_scorer_input_cache_tensor_export: bool = False,
    scorer_input_cache_tensor_volume_run_id: str = "",
) -> dict[str, Any]:
    import os
    import shutil
    import subprocess
    import sys
    import time

    if int(scorer_input_cache_hash_batch_pairs) < 1:
        return {
            "schema_version": 1,
            "passed": False,
            "returncode": 14,
            "error": "scorer_input_cache_hash_batch_pairs must be >= 1",
            "score_claim": False,
            "promotion_eligible": False,
        }
    if int(scorer_input_cache_tensor_batch_pairs) < 1:
        return {
            "schema_version": 1,
            "passed": False,
            "returncode": 15,
            "error": "scorer_input_cache_tensor_batch_pairs must be >= 1",
            "score_claim": False,
            "promotion_eligible": False,
        }
    if int(scorer_input_cache_tensor_large_pair_threshold) < 1:
        return {
            "schema_version": 1,
            "passed": False,
            "returncode": 16,
            "error": "scorer_input_cache_tensor_large_pair_threshold must be >= 1",
            "score_claim": False,
            "promotion_eligible": False,
        }
    scorer_device = str(scorer_device or "cuda").lower()
    inflate_device_policy = str(inflate_device_policy or "auto").lower()
    if scorer_device not in {"cuda", "cpu"}:
        return {
            "schema_version": 1,
            "passed": False,
            "returncode": 11,
            "error": f"invalid scorer_device {scorer_device!r}",
            "score_claim": False,
            "promotion_eligible": False,
        }
    if inflate_device_policy not in {"auto", "cpu", "cuda"}:
        return {
            "schema_version": 1,
            "passed": False,
            "returncode": 12,
            "error": f"invalid inflate_device_policy {inflate_device_policy!r}",
            "score_claim": False,
            "promotion_eligible": False,
        }
    if scorer_device == "cpu" and inflate_device_policy == "auto":
        return {
            "schema_version": 1,
            "passed": False,
            "returncode": 13,
            "error": (
                "Modal GPU-host CPU scorer diagnostics require explicit "
                "inflate_device_policy cpu or cuda; use modal_auth_eval_cpu.py "
                "for pure contest-CPU host eval"
            ),
            "score_claim": False,
            "promotion_eligible": False,
        }
    diagnostic_only = (
        bool(inflate_env_overrides)
        or inflate_device_policy != "auto"
        or scorer_device != "cuda"
    )

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
    canonical_path = (
        f"archive.zip -> inflate.sh -> upstream/evaluate.py --device {scorer_device}"
    )
    tensor_volume_run_id = _safe_tensor_volume_run_id(
        scorer_input_cache_tensor_volume_run_id,
        archive_sha256=archive_sha256,
        axis=f"modal_{scorer_device}",
    )
    tensor_volume_dir = AUTH_CACHE_VOLUME_ROOT / tensor_volume_run_id / "scorer_input_cache_tensors"

    os.environ["DALI_DISABLE_NVML"] = DALI_DISABLE_NVML_VALUE
    preflight = _probe_cuda_environment()
    preflight.update(
        {
            "archive_sha256": archive_sha256,
            "archive_size_bytes": archive_size_bytes,
            "inflate_sh_rel": inflate_sh_rel,
            "submission_dir_zip_sha256": submission_dir_zip_sha256,
            "source_repo_commit": source_repo_commit,
            "canonical_path": canonical_path,
            "scorer_device": scorer_device,
            "inflate_device_policy": inflate_device_policy,
            "inflate_env_overrides": list(inflate_env_overrides),
            "expected_runtime_tree_sha256": expected_runtime_tree_sha256,
            "scorer_input_cache_hashes_requested": bool(scorer_input_cache_hashes),
            "scorer_input_cache_hash_batch_pairs": int(scorer_input_cache_hash_batch_pairs),
            "scorer_input_cache_tensors_requested": bool(scorer_input_cache_tensors),
            "scorer_input_cache_tensor_batch_pairs": int(
                scorer_input_cache_tensor_batch_pairs
            ),
            "scorer_input_cache_tensor_large_pair_threshold": int(
                scorer_input_cache_tensor_large_pair_threshold
            ),
            "scorer_input_cache_tensor_volume_name": AUTH_CACHE_VOLUME_NAME,
            "scorer_input_cache_tensor_volume_run_id": tensor_volume_run_id,
            "scorer_input_cache_tensor_volume_path": str(tensor_volume_dir),
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
        scorer_device,
        "--keep-work-dir",
        "--work-dir",
        str(work_dir),
        "--inflate-timeout",
        str(int(inflate_timeout)),
        "--evaluate-timeout",
        str(int(evaluate_timeout)),
        "--inflate-device",
        inflate_device_policy,
    ]
    if expected_runtime_tree_sha256:
        cmd.extend(["--expected-runtime-tree-sha256", expected_runtime_tree_sha256])
    if scorer_input_cache_hashes:
        cmd.extend(
            [
                "--scorer-input-cache-hashes-out",
                str(work_dir / "scorer_input_cache_hashes.json"),
                "--scorer-input-cache-hash-batch-pairs",
                str(int(scorer_input_cache_hash_batch_pairs)),
            ]
        )
    if scorer_input_cache_tensors:
        cmd.extend(
            [
                "--scorer-input-cache-tensors-out-dir",
                str(tensor_volume_dir),
                "--scorer-input-cache-tensor-batch-pairs",
                str(int(scorer_input_cache_tensor_batch_pairs)),
                "--scorer-input-cache-tensor-large-pair-threshold",
                str(int(scorer_input_cache_tensor_large_pair_threshold)),
                "--allow-scorer-input-cache-artifact-output-outside-work-dir",
            ]
        )
        if allow_large_scorer_input_cache_tensor_export:
            cmd.append("--allow-large-scorer-input-cache-tensor-export")
    for item in inflate_env_overrides:
        cmd.extend(["--inflate-env", item])
    env = {
        **os.environ,
        "PYTHONPATH": REMOTE_PYTHONPATH,
        "FFMPEG_BIN": "/usr/local/bin/ffmpeg-master",
        "UV_BIN": "/usr/local/bin/uv",
        "UV_LINK_MODE": "copy",
        "UV_PROJECT_ENVIRONMENT": str(uv_env),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        "DALI_DISABLE_NVML": DALI_DISABLE_NVML_VALUE,
        "PACT_SOURCE_COMMIT": source_repo_commit,
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
            "canonical_path": canonical_path,
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
    tensor_volume_manifest = None
    if scorer_input_cache_tensors:
        tensor_volume_manifest = _record_tensor_volume_manifest(
            out_dir=out_dir,
            tensor_volume_dir=tensor_volume_dir,
            tensor_volume_run_id=tensor_volume_run_id,
        )
        if tensor_volume_manifest is None:
            validation_errors.append("scorer_input_cache_tensor_volume_manifest was not produced")
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
                expected_device=scorer_device,
            )
        )

    passed = proc.returncode == 0 and not validation_errors
    payload_score_claim = bool(
        passed
        and not diagnostic_only
        and isinstance(payload, dict)
        and (payload.get("score_claim") is True or payload.get("score_claim_valid") is True)
    )
    payload_promotion_eligible = bool(
        passed
        and not diagnostic_only
        and isinstance(payload, dict)
        and payload.get("promotion_eligible") is True
    )
    validation = {
        "schema_version": 1,
        "passed": passed,
        "returncode": proc.returncode if passed else (proc.returncode or 10),
        "modal_elapsed_seconds": elapsed,
        "command": cmd,
        "canonical_path": canonical_path,
        "expected_archive_sha256": archive_sha256,
        "expected_archive_size_bytes": archive_size_bytes,
        "inflate_sh_rel": inflate_sh_rel,
        "submission_dir_zip_sha256": submission_dir_zip_sha256,
        "expected_runtime_tree_sha256": expected_runtime_tree_sha256,
        "scorer_device": scorer_device,
        "inflate_device_policy": inflate_device_policy,
        "scorer_input_cache_hashes_requested": bool(scorer_input_cache_hashes),
        "scorer_input_cache_hash_batch_pairs": int(scorer_input_cache_hash_batch_pairs),
        "scorer_input_cache_tensors_requested": bool(scorer_input_cache_tensors),
        "scorer_input_cache_tensor_batch_pairs": int(scorer_input_cache_tensor_batch_pairs),
        "scorer_input_cache_tensor_large_pair_threshold": int(
            scorer_input_cache_tensor_large_pair_threshold
        ),
        "scorer_input_cache_tensor_volume_name": AUTH_CACHE_VOLUME_NAME,
        "scorer_input_cache_tensor_volume_run_id": tensor_volume_run_id,
        "scorer_input_cache_tensor_volume_path": str(tensor_volume_dir),
        "scorer_input_cache_tensor_volume_download_command": (
            f".venv/bin/modal volume get {AUTH_CACHE_VOLUME_NAME} "
            f"{tensor_volume_run_id}/ ./modal_{tensor_volume_run_id}/"
        ),
        "scorer_input_cache_tensor_volume_manifest": tensor_volume_manifest,
        "diagnostic_only": diagnostic_only,
        "validation_errors": validation_errors,
        "score_claim": payload_score_claim,
        "promotion_eligible": payload_promotion_eligible,
        "adjudication_required": True,
        "allowed_use": (
            ["cuda_auth_eval_review", "adjudication_required"]
            if payload_score_claim
            else ["diagnostic_debugging", "no_score_claim", "no_promotion"]
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
                "score_axis": payload.get("score_axis"),
                "evidence_grade": payload.get("evidence_grade"),
                "diagnostic_blockers": payload.get("diagnostic_blockers"),
            }
        )
    (out_dir / "modal_cuda_auth_eval_validation.json").write_bytes(_json_bytes(validation))

    return {
        **validation,
        "artifacts": _collect_artifacts(out_dir, work_dir),
    }


def _run_auth_eval_fail_closed(
    *,
    archive_bytes: bytes,
    archive_sha256: str,
    archive_size_bytes: int,
    inflate_sh_rel: str,
    submission_dir_zip_bytes: bytes | None,
    submission_dir_zip_sha256: str | None,
    source_repo_commit: str,
    inflate_timeout: int,
    evaluate_timeout: int,
    scorer_device: str = "cuda",
    inflate_device_policy: str = "auto",
    inflate_env_overrides: tuple[str, ...] = (),
    expected_runtime_tree_sha256: str = "",
    scorer_input_cache_hashes: bool = False,
    scorer_input_cache_hash_batch_pairs: int = 8,
    scorer_input_cache_tensors: bool = False,
    scorer_input_cache_tensor_batch_pairs: int = 8,
    scorer_input_cache_tensor_large_pair_threshold: int = 64,
    allow_large_scorer_input_cache_tensor_export: bool = False,
    scorer_input_cache_tensor_volume_run_id: str = "",
) -> dict[str, Any]:
    try:
        return _run_auth_eval_inner(
            archive_bytes=archive_bytes,
            archive_sha256=archive_sha256,
            archive_size_bytes=archive_size_bytes,
            inflate_sh_rel=inflate_sh_rel,
            submission_dir_zip_bytes=submission_dir_zip_bytes,
            submission_dir_zip_sha256=submission_dir_zip_sha256,
            source_repo_commit=source_repo_commit,
            inflate_timeout=inflate_timeout,
            evaluate_timeout=evaluate_timeout,
            scorer_device=scorer_device,
            inflate_device_policy=inflate_device_policy,
            inflate_env_overrides=inflate_env_overrides,
            expected_runtime_tree_sha256=expected_runtime_tree_sha256,
            scorer_input_cache_hashes=scorer_input_cache_hashes,
            scorer_input_cache_hash_batch_pairs=scorer_input_cache_hash_batch_pairs,
            scorer_input_cache_tensors=scorer_input_cache_tensors,
            scorer_input_cache_tensor_batch_pairs=scorer_input_cache_tensor_batch_pairs,
            scorer_input_cache_tensor_large_pair_threshold=(
                scorer_input_cache_tensor_large_pair_threshold
            ),
            allow_large_scorer_input_cache_tensor_export=(
                allow_large_scorer_input_cache_tensor_export
            ),
            scorer_input_cache_tensor_volume_run_id=scorer_input_cache_tensor_volume_run_id,
        )
    except Exception as exc:  # pragma: no cover - remote diagnostic path
        return fail_closed_remote_exception_result(
            out_dir=REMOTE_OUT,
            work_dir=REMOTE_WORK_ROOT / "eval_work",
            validation_path=REMOTE_OUT / "modal_cuda_auth_eval_validation.json",
            canonical_path=(
                f"archive.zip -> inflate.sh -> upstream/evaluate.py --device {scorer_device}"
            ),
            exc=exc,
            collect_artifacts=_collect_artifacts,
        )


@app.function(
    image=eval_image,
    gpu="T4",
    timeout=4800,
    volumes={str(AUTH_CACHE_VOLUME_ROOT): auth_cache_vol},
)
def run_auth_eval(
    archive_bytes: bytes,
    archive_sha256: str,
    archive_size_bytes: int,
    inflate_sh_rel: str = "submissions/robust_current/inflate.sh",
    submission_dir_zip_bytes: bytes | None = None,
    submission_dir_zip_sha256: str | None = None,
    source_repo_commit: str = "",
    inflate_timeout: int = 1800,
    evaluate_timeout: int = 1800,
    scorer_device: str = "cuda",
    inflate_device_policy: str = "auto",
    inflate_env_overrides: tuple[str, ...] = (),
    expected_runtime_tree_sha256: str = "",
    scorer_input_cache_hashes: bool = False,
    scorer_input_cache_hash_batch_pairs: int = 8,
    scorer_input_cache_tensors: bool = False,
    scorer_input_cache_tensor_batch_pairs: int = 8,
    scorer_input_cache_tensor_large_pair_threshold: int = 64,
    allow_large_scorer_input_cache_tensor_export: bool = False,
    scorer_input_cache_tensor_volume_run_id: str = "",
) -> dict[str, Any]:
    """Run the canonical CUDA auth eval on Modal T4."""

    return _run_auth_eval_fail_closed(
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size_bytes,
        inflate_sh_rel=inflate_sh_rel,
        submission_dir_zip_bytes=submission_dir_zip_bytes,
        submission_dir_zip_sha256=submission_dir_zip_sha256,
        source_repo_commit=source_repo_commit,
        inflate_timeout=inflate_timeout,
        evaluate_timeout=evaluate_timeout,
        scorer_device=scorer_device,
        inflate_device_policy=inflate_device_policy,
        inflate_env_overrides=inflate_env_overrides,
        expected_runtime_tree_sha256=expected_runtime_tree_sha256,
        scorer_input_cache_hashes=scorer_input_cache_hashes,
        scorer_input_cache_hash_batch_pairs=scorer_input_cache_hash_batch_pairs,
        scorer_input_cache_tensors=scorer_input_cache_tensors,
        scorer_input_cache_tensor_batch_pairs=scorer_input_cache_tensor_batch_pairs,
        scorer_input_cache_tensor_large_pair_threshold=scorer_input_cache_tensor_large_pair_threshold,
        allow_large_scorer_input_cache_tensor_export=allow_large_scorer_input_cache_tensor_export,
        scorer_input_cache_tensor_volume_run_id=scorer_input_cache_tensor_volume_run_id,
    )


@app.function(
    image=eval_image,
    gpu="A100",
    timeout=4800,
    volumes={str(AUTH_CACHE_VOLUME_ROOT): auth_cache_vol},
)
def run_auth_eval_a100(
    archive_bytes: bytes,
    archive_sha256: str,
    archive_size_bytes: int,
    inflate_sh_rel: str = "submissions/robust_current/inflate.sh",
    submission_dir_zip_bytes: bytes | None = None,
    submission_dir_zip_sha256: str | None = None,
    source_repo_commit: str = "",
    inflate_timeout: int = 1800,
    evaluate_timeout: int = 1800,
    scorer_device: str = "cuda",
    inflate_device_policy: str = "auto",
    inflate_env_overrides: tuple[str, ...] = (),
    expected_runtime_tree_sha256: str = "",
    scorer_input_cache_hashes: bool = False,
    scorer_input_cache_hash_batch_pairs: int = 8,
    scorer_input_cache_tensors: bool = False,
    scorer_input_cache_tensor_batch_pairs: int = 8,
    scorer_input_cache_tensor_large_pair_threshold: int = 64,
    allow_large_scorer_input_cache_tensor_export: bool = False,
    scorer_input_cache_tensor_volume_run_id: str = "",
) -> dict[str, Any]:
    """Run CUDA auth eval on Modal A100 as a diagnostic axis."""

    return _run_auth_eval_fail_closed(
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size_bytes,
        inflate_sh_rel=inflate_sh_rel,
        submission_dir_zip_bytes=submission_dir_zip_bytes,
        submission_dir_zip_sha256=submission_dir_zip_sha256,
        source_repo_commit=source_repo_commit,
        inflate_timeout=inflate_timeout,
        evaluate_timeout=evaluate_timeout,
        scorer_device=scorer_device,
        inflate_device_policy=inflate_device_policy,
        inflate_env_overrides=inflate_env_overrides,
        expected_runtime_tree_sha256=expected_runtime_tree_sha256,
        scorer_input_cache_hashes=scorer_input_cache_hashes,
        scorer_input_cache_hash_batch_pairs=scorer_input_cache_hash_batch_pairs,
        scorer_input_cache_tensors=scorer_input_cache_tensors,
        scorer_input_cache_tensor_batch_pairs=scorer_input_cache_tensor_batch_pairs,
        scorer_input_cache_tensor_large_pair_threshold=scorer_input_cache_tensor_large_pair_threshold,
        allow_large_scorer_input_cache_tensor_export=allow_large_scorer_input_cache_tensor_export,
        scorer_input_cache_tensor_volume_run_id=scorer_input_cache_tensor_volume_run_id,
    )


@app.function(
    image=eval_image,
    gpu="H100",
    timeout=4800,
    volumes={str(AUTH_CACHE_VOLUME_ROOT): auth_cache_vol},
)
def run_auth_eval_h100(
    archive_bytes: bytes,
    archive_sha256: str,
    archive_size_bytes: int,
    inflate_sh_rel: str = "submissions/robust_current/inflate.sh",
    submission_dir_zip_bytes: bytes | None = None,
    submission_dir_zip_sha256: str | None = None,
    source_repo_commit: str = "",
    inflate_timeout: int = 1800,
    evaluate_timeout: int = 1800,
    scorer_device: str = "cuda",
    inflate_device_policy: str = "auto",
    inflate_env_overrides: tuple[str, ...] = (),
    expected_runtime_tree_sha256: str = "",
    scorer_input_cache_hashes: bool = False,
    scorer_input_cache_hash_batch_pairs: int = 8,
    scorer_input_cache_tensors: bool = False,
    scorer_input_cache_tensor_batch_pairs: int = 8,
    scorer_input_cache_tensor_large_pair_threshold: int = 64,
    allow_large_scorer_input_cache_tensor_export: bool = False,
    scorer_input_cache_tensor_volume_run_id: str = "",
) -> dict[str, Any]:
    """Run CUDA auth eval on Modal H100 as a diagnostic axis."""

    return _run_auth_eval_fail_closed(
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size_bytes,
        inflate_sh_rel=inflate_sh_rel,
        submission_dir_zip_bytes=submission_dir_zip_bytes,
        submission_dir_zip_sha256=submission_dir_zip_sha256,
        source_repo_commit=source_repo_commit,
        inflate_timeout=inflate_timeout,
        evaluate_timeout=evaluate_timeout,
        scorer_device=scorer_device,
        inflate_device_policy=inflate_device_policy,
        inflate_env_overrides=inflate_env_overrides,
        expected_runtime_tree_sha256=expected_runtime_tree_sha256,
        scorer_input_cache_hashes=scorer_input_cache_hashes,
        scorer_input_cache_hash_batch_pairs=scorer_input_cache_hash_batch_pairs,
        scorer_input_cache_tensors=scorer_input_cache_tensors,
        scorer_input_cache_tensor_batch_pairs=scorer_input_cache_tensor_batch_pairs,
        scorer_input_cache_tensor_large_pair_threshold=scorer_input_cache_tensor_large_pair_threshold,
        allow_large_scorer_input_cache_tensor_export=allow_large_scorer_input_cache_tensor_export,
        scorer_input_cache_tensor_volume_run_id=scorer_input_cache_tensor_volume_run_id,
    )


@app.local_entrypoint()
def main(
    archive: str = "/tmp/modal_submission/archive.zip",
    output_dir: str = "",
    expected_archive_sha256: str = "",
    inflate_sh: str = "submissions/robust_current/inflate.sh",
    submission_dir: str = "",
    gpu: str = "T4",
    scorer_device: str = "cuda",
    inflate_timeout: int = 1800,
    evaluate_timeout: int = 1800,
    inflate_device: str = "auto",
    inflate_env: str = "",
    expected_runtime_tree_sha256: str = "",
    scorer_input_cache_hashes: bool = False,
    scorer_input_cache_hash_batch_pairs: int = 8,
    scorer_input_cache_tensors: bool = False,
    scorer_input_cache_tensor_batch_pairs: int = 8,
    scorer_input_cache_tensor_large_pair_threshold: int = 64,
    allow_large_scorer_input_cache_tensor_export: bool = False,
    scorer_input_cache_tensor_volume_run_id: str = "",
    detach: bool = False,
    provider_detach_ack: bool = False,
    lane_id: str = "",
    instance_job_id: str = "",
    claim_agent: str = "codex:modal_auth_eval",
    claim_notes: str = "",
    force_claim: bool = False,
    pair_group_id: str = "",
    single_axis_waiver_reason: str = "",
) -> None:
    """Upload an archive and harvest Modal CUDA auth-eval artifacts."""

    if int(scorer_input_cache_hash_batch_pairs) < 1:
        raise SystemExit("FATAL: --scorer-input-cache-hash-batch-pairs must be >= 1")
    if int(scorer_input_cache_tensor_batch_pairs) < 1:
        raise SystemExit("FATAL: --scorer-input-cache-tensor-batch-pairs must be >= 1")
    if int(scorer_input_cache_tensor_large_pair_threshold) < 1:
        raise SystemExit(
            "FATAL: --scorer-input-cache-tensor-large-pair-threshold must be >= 1"
        )
    if detach and not provider_detach_ack:
        raise SystemExit(
            "FATAL: wrapper --detach requires provider-level Modal CLI detach. "
            "Use `.venv/bin/modal run --detach experiments/modal_auth_eval.py ... "
            "--detach --provider-detach-ack ...`. Without CLI --detach the ephemeral "
            "Modal app may stop before the spawned function returns, producing a "
            "blank RemoteError and no score artifact."
        )

    prepared = prepare_modal_auth_eval_request(
        archive=archive,
        output_dir=output_dir,
        inflate_sh=inflate_sh,
        submission_dir=submission_dir,
        default_output_root=Path("experiments/results/modal_auth_eval"),
    )
    archive_path = prepared.archive_path
    archive_bytes = prepared.archive_bytes
    archive_sha256 = prepared.archive_sha256
    archive_size_bytes = prepared.archive_size_bytes
    expected_archive_sha256 = str(expected_archive_sha256 or "").strip().lower()
    if expected_archive_sha256 and expected_archive_sha256 != archive_sha256:
        raise SystemExit(
            "FATAL: --expected-archive-sha256 does not match selected archive: "
            f"expected={expected_archive_sha256} actual={archive_sha256}"
        )
    inflate_sh_rel = prepared.inflate_sh_rel
    submission_dir_path = prepared.submission_dir_path
    submission_dir_zip = prepared.submission_dir_zip
    submission_dir_zip_sha256 = prepared.submission_dir_zip_sha256
    out_dir = prepared.output_dir
    source_repo_commit = _local_git_commit()
    tensor_volume_run_id = _safe_tensor_volume_run_id(
        scorer_input_cache_tensor_volume_run_id or out_dir.name,
        archive_sha256=archive_sha256,
        axis="contest_cuda",
    )
    requested_axis = (
        "contest_cuda"
        if str(scorer_device or "cuda").lower() == "cuda"
        and str(inflate_device or "auto").lower() == "auto"
        and str(gpu or "T4").upper() == "T4"
        and not inflate_env
        else f"diagnostic_{str(scorer_device or 'cuda').lower()}"
    )
    try:
        pairing = validate_modal_auth_eval_pairing(
            axis=requested_axis,
            pair_group_id=pair_group_id,
            single_axis_waiver_reason=single_axis_waiver_reason,
        )
    except ModalAuthEvalPairingError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    _validate_uploaded_runtime_tree_expectation(
        expected_runtime_tree_sha256=expected_runtime_tree_sha256,
        submission_dir_path=submission_dir_path,
        inflate_sh_rel=inflate_sh_rel,
    )
    scorer_device_policy = str(scorer_device or "cuda").lower()
    if scorer_device_policy not in {"cuda", "cpu"}:
        raise SystemExit("FATAL: --scorer-device must be one of cuda, cpu")
    inflate_device_policy = str(inflate_device or "auto").lower()
    if inflate_device_policy not in {"auto", "cpu", "cuda"}:
        raise SystemExit("FATAL: --inflate-device must be one of auto, cpu, cuda")
    if scorer_device_policy == "cpu" and inflate_device_policy == "auto":
        raise SystemExit(
            "FATAL: Modal GPU-host CPU scorer diagnostics require an explicit "
            "--inflate-device cpu or --inflate-device cuda. Use "
            "experiments/modal_auth_eval_cpu.py for pure contest-CPU host eval."
        )
    inflate_env_overrides = (inflate_env,) if inflate_env else ()
    gpu_key = gpu.upper()
    non_t4_gpu_diagnostic = gpu_key not in {"T4"}
    diagnostic_only = (
        bool(inflate_env_overrides)
        or inflate_device_policy != "auto"
        or scorer_device_policy != "cuda"
        or non_t4_gpu_diagnostic
    )
    axis_label = f"diagnostic_{scorer_device_policy}" if diagnostic_only else "contest_cuda"

    local_summary = {
        "schema_version": 1,
        "tool": "experiments/modal_auth_eval.py",
        "app": APP_NAME,
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "expected_archive_sha256": expected_archive_sha256 or archive_sha256,
        "expected_archive_sha256_match": True,
        "archive_size_bytes": archive_size_bytes,
        "inflate_sh": inflate_sh_rel,
        "submission_dir": str(submission_dir_path) if submission_dir_path else None,
        "submission_dir_zip_sha256": submission_dir_zip_sha256,
        "source_repo_commit": source_repo_commit,
        "canonical_path": f"archive.zip -> inflate.sh -> upstream/evaluate.py --device {scorer_device_policy}",
        "modal_dispatch_mode": "detached_spawn" if detach else "blocking_remote",
        "scorer_device": scorer_device_policy,
        "inflate_device_policy": inflate_device_policy,
        "inflate_env_overrides": list(inflate_env_overrides),
        "expected_runtime_tree_sha256": expected_runtime_tree_sha256,
        "scorer_input_cache_hashes_requested": bool(scorer_input_cache_hashes),
        "scorer_input_cache_hash_batch_pairs": int(scorer_input_cache_hash_batch_pairs),
        "scorer_input_cache_tensors_requested": bool(scorer_input_cache_tensors),
        "scorer_input_cache_tensor_batch_pairs": int(scorer_input_cache_tensor_batch_pairs),
        "scorer_input_cache_tensor_large_pair_threshold": int(
            scorer_input_cache_tensor_large_pair_threshold
        ),
        "allow_large_scorer_input_cache_tensor_export": bool(
            allow_large_scorer_input_cache_tensor_export
        ),
        "scorer_input_cache_tensor_volume_name": AUTH_CACHE_VOLUME_NAME,
        "scorer_input_cache_tensor_volume_run_id": tensor_volume_run_id,
        "scorer_input_cache_tensor_volume_path": str(
            AUTH_CACHE_VOLUME_ROOT / tensor_volume_run_id / "scorer_input_cache_tensors"
        ),
        "scorer_input_cache_tensor_volume_download_command": (
            f".venv/bin/modal volume get {AUTH_CACHE_VOLUME_NAME} "
            f"{tensor_volume_run_id}/ ./modal_{tensor_volume_run_id}/"
        ),
        "diagnostic_only": diagnostic_only,
        "non_t4_gpu_diagnostic": non_t4_gpu_diagnostic,
        "score_claim": False,
        "promotion_eligible": False,
        "adjudication_required": True,
        **pairing,
    }
    write_json(out_dir / "modal_cuda_auth_eval_local_request.json", local_summary)

    claim_spec = ClaimSpec(
        lane_id=lane_id,
        instance_job_id=instance_job_id,
        agent=claim_agent,
        force=force_claim,
        notes=(
            claim_notes
            or (
                "Modal CUDA auth eval; exact archive path; "
                f"axis={axis_label}; archive_sha256={archive_sha256}; "
                f"pair_group_id={pairing.get('pair_group_id')}; "
                f"single_axis_waiver={pairing.get('single_axis_waiver_used')}"
            )
        ),
    )

    print(
        f"Uploading {archive_size_bytes:,} bytes to Modal {gpu} for CUDA auth eval "
        f"(sha256={archive_sha256})..."
    )
    if gpu_key == "T4":
        auth_eval_fn = run_auth_eval
    elif gpu_key in {"A100", "A100-40GB", "A100-80GB"}:
        auth_eval_fn = run_auth_eval_a100
    elif gpu_key in {"H100", "H100-80GB"}:
        auth_eval_fn = run_auth_eval_h100
    else:
        raise SystemExit(f"FATAL: unsupported --gpu {gpu!r}; use T4, A100, or H100")

    call_args = (
        archive_bytes,
        archive_sha256,
        archive_size_bytes,
        inflate_sh_rel,
        submission_dir_zip,
        submission_dir_zip_sha256,
        source_repo_commit,
        int(inflate_timeout),
        int(evaluate_timeout),
        scorer_device_policy,
        inflate_device_policy,
        inflate_env_overrides,
        expected_runtime_tree_sha256,
        bool(scorer_input_cache_hashes),
        int(scorer_input_cache_hash_batch_pairs),
        bool(scorer_input_cache_tensors),
        int(scorer_input_cache_tensor_batch_pairs),
        int(scorer_input_cache_tensor_large_pair_threshold),
        bool(allow_large_scorer_input_cache_tensor_export),
        tensor_volume_run_id,
    )
    claim_modal_auth_eval_dispatch(
        repo_root=Path.cwd(),
        spec=claim_spec,
        status="active_modal_auth_eval_spawning" if detach else "active_modal_auth_eval_running",
    )
    if detach:
        try:
            call = auth_eval_fn.spawn(*call_args)
        except Exception as exc:
            claim_modal_auth_eval_dispatch(
                repo_root=Path.cwd(),
                spec=ClaimSpec(
                    lane_id=lane_id,
                    instance_job_id=instance_job_id,
                    agent=claim_agent,
                    force=True,
                    notes=(
                        "Modal CUDA auth eval spawn raised after dispatch boundary; "
                        f"manual Modal reconciliation required; error={type(exc).__name__}"
                    ),
                ),
                status="ambiguous_modal_auth_eval_spawn_submission_recovery_required",
            )
            raise
        call_id = function_call_id(call)
        write_spawn_metadata(
            out_dir=out_dir,
            tool="experiments/modal_auth_eval.py",
            app=APP_NAME,
            axis=axis_label,
            call_id=call_id,
            local_request=local_summary,
            result_json_name="modal_cuda_auth_eval_result.json",
            extra={
                "gpu": gpu_key,
                "diagnostic_only": diagnostic_only,
                "non_t4_gpu_diagnostic": non_t4_gpu_diagnostic,
                "scorer_device": scorer_device_policy,
                "inflate_device_policy": inflate_device_policy,
                "inflate_env_overrides": list(inflate_env_overrides),
                "expected_runtime_tree_sha256": expected_runtime_tree_sha256,
                "scorer_input_cache_hashes_requested": bool(scorer_input_cache_hashes),
                "scorer_input_cache_hash_batch_pairs": int(scorer_input_cache_hash_batch_pairs),
                "scorer_input_cache_tensors_requested": bool(scorer_input_cache_tensors),
                "scorer_input_cache_tensor_volume_name": AUTH_CACHE_VOLUME_NAME,
                "scorer_input_cache_tensor_volume_run_id": tensor_volume_run_id,
                "scorer_input_cache_tensor_volume_download_command": (
                    f".venv/bin/modal volume get {AUTH_CACHE_VOLUME_NAME} "
                    f"{tensor_volume_run_id}/ ./modal_{tensor_volume_run_id}/"
                ),
                "expected_archive_sha256": expected_archive_sha256 or archive_sha256,
                "lane_id": lane_id,
                "instance_job_id": instance_job_id,
                "claim_agent": claim_agent,
                "claim_platform": "modal",
                **pairing,
            },
        )
        claim_modal_auth_eval_dispatch(
            repo_root=Path.cwd(),
            spec=ClaimSpec(
                lane_id=lane_id,
                instance_job_id=instance_job_id,
                agent=claim_agent,
                force=True,
                notes=(
                    "Modal CUDA auth eval detached spawn accepted; "
                    f"call_id={call_id}; output_dir={out_dir}"
                ),
            ),
            status="active_modal_auth_eval_spawned",
        )
        print("=" * 60)
        print(f"MODAL CUDA AUTH EVAL DISPATCHED DETACHED call_id={call_id}")
        print(f"  Artifacts: {out_dir}")
        print(
            "  Recover:   "
            f".venv/bin/python tools/recover_modal_auth_eval.py --output-dir {out_dir}"
        )
        print("=" * 60)
        return

    try:
        result = auth_eval_fn.remote(*call_args)
    except Exception as exc:
        terminal_modal_auth_eval_claim(
            repo_root=Path.cwd(),
            spec=claim_spec,
            status="failed_modal_auth_eval_exception",
            notes=f"Modal CUDA auth eval raised {type(exc).__name__}; no score claim",
        )
        raise

    artifacts = result.pop("artifacts", {})
    if isinstance(artifacts, dict):
        try:
            materialize_modal_artifacts(out_dir=out_dir, artifacts=artifacts)
        except ModalArtifactWriteError as exc:
            failure = {
                "schema_version": "modal_cuda_auth_eval_result_v1",
                "status": "invalid_artifacts",
                "artifact_write_errors": exc.errors,
                "score_claim": False,
                "promotion_eligible": False,
                "archive_sha256": archive_sha256,
                "archive_size_bytes": archive_size_bytes,
            }
            write_json(out_dir / "modal_cuda_auth_eval_result.json", failure)
            terminal_modal_auth_eval_claim(
                repo_root=Path.cwd(),
                spec=claim_spec,
                status="failed_modal_auth_eval_invalid_artifacts",
                notes=(
                    "Modal CUDA auth eval returned unsafe/malformed artifacts; "
                    f"archive_sha256={archive_sha256}; output_dir={out_dir}"
                ),
            )
            raise SystemExit(5) from exc

    result["local_output_dir"] = str(out_dir)
    result["archive_path"] = str(archive_path)
    result["archive_sha256"] = archive_sha256
    result["expected_archive_sha256"] = expected_archive_sha256 or archive_sha256
    result["expected_archive_sha256_match"] = True
    result["archive_size_bytes"] = archive_size_bytes
    result["inflate_sh"] = inflate_sh_rel
    result["source_repo_commit"] = source_repo_commit
    result["scorer_device"] = scorer_device_policy
    result["inflate_device_policy"] = inflate_device_policy
    result["inflate_env_overrides"] = list(inflate_env_overrides)
    result["expected_runtime_tree_sha256"] = expected_runtime_tree_sha256
    result["scorer_input_cache_hashes_requested"] = bool(scorer_input_cache_hashes)
    result["scorer_input_cache_hash_batch_pairs"] = int(scorer_input_cache_hash_batch_pairs)
    result["scorer_input_cache_tensors_requested"] = bool(scorer_input_cache_tensors)
    result["scorer_input_cache_tensor_batch_pairs"] = int(scorer_input_cache_tensor_batch_pairs)
    result["scorer_input_cache_tensor_large_pair_threshold"] = int(
        scorer_input_cache_tensor_large_pair_threshold
    )
    result["scorer_input_cache_tensor_volume_name"] = AUTH_CACHE_VOLUME_NAME
    result["scorer_input_cache_tensor_volume_run_id"] = tensor_volume_run_id
    result["scorer_input_cache_tensor_volume_download_command"] = (
        f".venv/bin/modal volume get {AUTH_CACHE_VOLUME_NAME} "
        f"{tensor_volume_run_id}/ ./modal_{tensor_volume_run_id}/"
    )
    result.update(pairing)
    if diagnostic_only:
        result["diagnostic_only"] = True
        result["non_t4_gpu_diagnostic"] = non_t4_gpu_diagnostic
        result["score_claim"] = False
        result["promotion_eligible"] = False
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

    terminal_notes = (
        "Modal CUDA auth eval "
        f"{'passed path validation' if result.get('passed') else 'failed closed'}; "
        f"score_axis={result.get('score_axis_from_payload') or result.get('score_axis')}; "
        f"hardware={scorer_device_policy}; "
        f"archive_sha256={archive_sha256}; "
        f"archive_bytes={archive_size_bytes}; "
        f"score={result.get('score_recomputed_from_components')}; "
        f"output_dir={out_dir}"
    )
    if not result.get("passed"):
        terminal_modal_auth_eval_claim(
            repo_root=Path.cwd(),
            spec=claim_spec,
            status="failed_modal_auth_eval_no_score_claim",
            notes=terminal_notes,
        )
        raise SystemExit(int(result.get("returncode") or 1))
    terminal_modal_auth_eval_claim(
        repo_root=Path.cwd(),
        spec=claim_spec,
        status="completed_modal_auth_eval_recovered",
        notes=terminal_notes,
    )


def _safe_tensor_volume_run_id(value: str, *, archive_sha256: str, axis: str) -> str:
    raw = str(value or "").strip() or f"{axis}_{archive_sha256[:16]}"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)
    safe = safe.strip("._-/")
    return safe or f"{axis}_{archive_sha256[:16]}"


def _record_tensor_volume_manifest(
    *,
    out_dir: Path,
    tensor_volume_dir: Path,
    tensor_volume_run_id: str,
) -> dict[str, Any] | None:
    manifest_path = tensor_volume_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    auth_cache_vol.commit()
    payload = read_json(manifest_path)
    record = {
        "schema_version": "modal_auth_eval_tensor_volume_manifest.v1",
        "volume_name": AUTH_CACHE_VOLUME_NAME,
        "volume_run_id": tensor_volume_run_id,
        "volume_path": str(tensor_volume_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_path(manifest_path),
        "tensor_payload_returned_via_modal_artifacts": False,
        "volume_download_command": (
            f".venv/bin/modal volume get {AUTH_CACHE_VOLUME_NAME} "
            f"{tensor_volume_run_id}/ ./modal_{tensor_volume_run_id}/"
        ),
        "payload": payload,
    }
    write_json(out_dir / "scorer_input_cache_tensor_volume_manifest.json", record)
    return record


def _local_git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
    except Exception as exc:  # pragma: no cover - local diagnostic fallback
        return f"<error obtaining local git commit: {exc!r}>"
