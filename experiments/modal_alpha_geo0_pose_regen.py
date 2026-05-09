"""Dispatch Alpha-Geo-0 pose regeneration plus canonical CUDA auth eval on Modal.

This is the non-Vast path for the Lane 12 Alpha-Geo-0 stale-pose isolation
experiment. It does not retrain NeRV. It keeps the exact measured
``masks.nrv`` payload, regenerates ``optimized_poses.bin`` against the decoded
candidate masks with ``experiments/optimize_poses.py``, rebuilds the
deterministic archive, then runs:

    archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda

Dispatch:

    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \\
      experiments/modal_alpha_geo0_pose_regen.py \\
      --label lane12_alpha_geo0_pose_regen_modal_t4_20260501

Recover:

    .venv/bin/python experiments/modal_alpha_geo0_pose_regen.py recover \\
      --label lane12_alpha_geo0_pose_regen_modal_t4_20260501
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import traceback
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_REPO = Path("/workspace/pact")
REMOTE_PYTHONPATH = f"{REMOTE_REPO / 'src'}:{REMOTE_REPO / 'upstream'}:{REMOTE_REPO}"
REMOTE_OUT_ROOT = Path("/tmp/modal_alpha_geo0_pose_regen")
RESULT_ROOT = REPO_ROOT / "experiments/results/modal_alpha_geo0_pose_regen"
APP_NAME = "comma-alpha-geo0-pose-regen"

DEFAULT_CANDIDATE_ARCHIVE = (
    REPO_ROOT / "experiments/results/lane_12_nerv_20260430_codex_jsonfix40/archive_lane_12_nerv.zip"
)
DEFAULT_BASELINE_ARCHIVE = (
    REPO_ROOT / "experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip"
)
DEFAULT_WARM_POSES = REPO_ROOT / "experiments/results/lane_g_v3_landed/iter_0/optimized_poses.pt"
DEFAULT_GT_POSE_TARGETS = REPO_ROOT / "experiments/results/lane_a_landed/gt_pose_targets.pt"

EXPECTED_BASELINE_SCORE = 1.043987524793892
EXPECTED_BASELINE_BYTES = 686635
REQUIRED_SAMPLES = 600
N_FRAMES = 1200
MASK_H = 384
MASK_W = 512

app = modal.App(APP_NAME)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ca-certificates",
        "curl",
        "ffmpeg",
        "git",
        "libgl1",
        "libglib2.0-0",
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

run_image = (
    base_image
    .env({"PYTHONPATH": REMOTE_PYTHONPATH})
    .add_local_dir("src", remote_path=str(REMOTE_REPO / "src"))
    .add_local_dir("upstream/models", remote_path=str(REMOTE_REPO / "upstream/models"))
    .add_local_dir("upstream/videos", remote_path=str(REMOTE_REPO / "upstream/videos"))
    .add_local_file("upstream/evaluate.py", remote_path=str(REMOTE_REPO / "upstream/evaluate.py"))
    .add_local_file("upstream/frame_utils.py", remote_path=str(REMOTE_REPO / "upstream/frame_utils.py"))
    .add_local_file("upstream/modules.py", remote_path=str(REMOTE_REPO / "upstream/modules.py"))
    .add_local_file(
        "upstream/public_test_video_names.txt",
        remote_path=str(REMOTE_REPO / "upstream/public_test_video_names.txt"),
    )
    .add_local_file("upstream/pyproject.toml", remote_path=str(REMOTE_REPO / "upstream/pyproject.toml"))
    .add_local_file("upstream/uv.lock", remote_path=str(REMOTE_REPO / "upstream/uv.lock"))
    .add_local_file("experiments/__init__.py", remote_path=str(REMOTE_REPO / "experiments/__init__.py"))
    .add_local_file(
        "experiments/diagnose_nerv_geometry.py",
        remote_path=str(REMOTE_REPO / "experiments/diagnose_nerv_geometry.py"),
    )
    .add_local_file(
        "experiments/optimize_poses.py",
        remote_path=str(REMOTE_REPO / "experiments/optimize_poses.py"),
    )
    .add_local_file(
        "experiments/contest_auth_eval.py",
        remote_path=str(REMOTE_REPO / "experiments/contest_auth_eval.py"),
    )
    .add_local_file(
        "scripts/adjudicate_contest_auth_eval.py",
        remote_path=str(REMOTE_REPO / "scripts/adjudicate_contest_auth_eval.py"),
    )
    .add_local_file("scripts/probe_nvdec.sh", remote_path=str(REMOTE_REPO / "scripts/probe_nvdec.sh"))
    .add_local_file(
        "submissions/robust_current/inflate.sh",
        remote_path=str(REMOTE_REPO / "submissions/robust_current/inflate.sh"),
    )
    .add_local_file(
        "submissions/robust_current/config.env",
        remote_path=str(REMOTE_REPO / "submissions/robust_current/config.env"),
    )
    .add_local_file(
        "submissions/robust_current/inflate_renderer.py",
        remote_path=str(REMOTE_REPO / "submissions/robust_current/inflate_renderer.py"),
    )
    .add_local_file("pyproject.toml", remote_path=str(REMOTE_REPO / "pyproject.toml"))
    .add_local_file("uv.lock", remote_path=str(REMOTE_REPO / "uv.lock"))
)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
    }


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return label or "alpha_geo0_pose_regen"


def _tail(value: str | bytes | None, limit: int = 4096) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-limit:]


def _mask_tensor_sha256(masks: Any) -> str:
    import torch

    if not isinstance(masks, torch.Tensor):
        raise TypeError(f"decoded masks must be a torch.Tensor, got {type(masks).__name__}")
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"decoded masks must have shape (T,H,W), got {tuple(masks.shape)}")
    normalized = masks.detach().cpu()
    if torch.is_floating_point(normalized):
        if not torch.equal(normalized, normalized.round()):
            raise ValueError("decoded masks floating tensor contains non-integer class IDs")
        normalized = normalized.round()
    normalized = normalized.to(torch.int64)
    if int(normalized.min().item()) < 0:
        raise ValueError("decoded masks contain negative class IDs")
    if int(normalized.max().item()) <= 255:
        normalized = normalized.to(torch.uint8)
    normalized = normalized.contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(normalized.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(normalized.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(normalized.numpy()))
    return digest.hexdigest()


def _load_mask_tensor(path: Path) -> Any:
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for key in ("masks", "mask_classes", "class_ids"):
            value = obj.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"{path} did not contain a mask tensor")


def _materialize_candidate_masks_from_cache(
    cache_dir: Path,
    *,
    candidate_archive_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Write a pure tensor .pt for optimize_poses.py from geometry cache output."""

    import torch

    matches: list[tuple[Path, Path, dict[str, Any]]] = []
    for meta_path in sorted(cache_dir.glob("*.json")):
        try:
            payload = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        fp = payload.get("fingerprint")
        if not isinstance(fp, dict):
            continue
        if fp.get("source_sha256") != candidate_archive_sha256:
            continue
        if fp.get("archive_member_resolved") != "masks.nrv":
            continue
        tensor_name = payload.get("tensor_file")
        if not isinstance(tensor_name, str):
            continue
        tensor_path = meta_path.with_name(tensor_name)
        if tensor_path.is_file():
            matches.append((meta_path, tensor_path, payload))
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one decoded candidate masks.nrv cache tensor, "
            f"found {len(matches)} in {cache_dir}"
        )

    meta_path, tensor_path, payload = matches[0]
    masks = _load_mask_tensor(tensor_path)
    decoded_sha = _mask_tensor_sha256(masks)
    expected_sha = payload.get("decoded_mask_sha256")
    if isinstance(expected_sha, str) and expected_sha != decoded_sha:
        raise RuntimeError(
            "decoded mask cache sha mismatch: "
            f"metadata={expected_sha} tensor={decoded_sha}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(masks.detach().cpu().contiguous(), output_path)
    reloaded = torch.load(output_path, map_location="cpu", weights_only=True)
    if not isinstance(reloaded, torch.Tensor):
        raise RuntimeError("materialized candidate masks file is not a tensor")

    return {
        "cache_metadata": _file_meta(meta_path),
        "cache_tensor": _file_meta(tensor_path),
        "materialized_masks_pt": _file_meta(output_path),
        "decoded_mask_sha256": decoded_sha,
        "decoded_mask_shape": [int(v) for v in masks.shape],
        "decoded_mask_dtype": str(masks.dtype),
        "materialized_format": "torch.Tensor",
    }


def _validated_zip_infos(path: Path) -> dict[str, zipfile.ZipInfo]:
    infos: dict[str, zipfile.ZipInfo] = {}
    with zipfile.ZipFile(path, "r") as zf:
        for info in zf.infolist():
            member_path = PurePosixPath(info.filename)
            if info.is_dir() or member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"unsafe archive member path: {info.filename!r}")
            if info.filename in infos:
                raise ValueError(f"duplicate archive member: {info.filename!r}")
            infos[info.filename] = info
    return infos


def _validate_archive_requirements(path: Path, *, required_members: set[str]) -> dict[str, Any]:
    infos = _validated_zip_infos(path)
    missing = sorted(required_members - set(infos))
    if missing:
        raise ValueError(f"{path} missing required member(s): {missing}")
    return {
        "archive": _file_meta(path),
        "members": {
            name: {
                "file_size": int(info.file_size),
                "compress_size": int(info.compress_size),
            }
            for name, info in sorted(infos.items())
        },
    }


def _safe_extract_members(zip_path: Path, members: set[str], dest: Path) -> dict[str, Any]:
    infos = _validated_zip_infos(zip_path)
    missing = sorted(members - set(infos))
    if missing:
        raise RuntimeError(f"{zip_path} missing required member(s): {missing}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in sorted(members):
            (dest / name).write_bytes(zf.read(name))
    return {
        "source_archive": _file_meta(zip_path),
        "extract_dir": str(dest),
        "members": {name: _file_meta(dest / name) for name in sorted(members)},
    }


def _build_alpha_geo0_archive(src_dir: Path, dst: Path) -> dict[str, Any]:
    from tac.submission_archive import detect_pose_manifest, validate_archive

    members = ("renderer.bin", "masks.nrv", "optimized_poses.bin")
    for name in members:
        path = src_dir / name
        if not path.is_file():
            raise RuntimeError(f"missing archive source member: {path}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zout:
        for name in members:
            data = (src_dir / name).read_bytes()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o644 & 0xFFFF) << 16
            zout.writestr(info, data, compresslevel=9)

    result = validate_archive(dst, manifest=detect_pose_manifest(dst), strict=True)
    if not result.valid:
        raise RuntimeError(f"Alpha-Geo-0 archive failed validation:\n{result.summary()}")
    return {
        "archive": _file_meta(dst),
        "member_order": list(members),
        "deterministic_zip": {
            "compression": "ZIP_DEFLATED",
            "compresslevel": 9,
            "date_time": [1980, 1, 1, 0, 0, 0],
            "permissions_octal": "0644",
        },
        "source_members": {name: _file_meta(src_dir / name) for name in members},
    }


def _run_logged(
    name: str,
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    import subprocess

    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=int(timeout),
            check=False,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = _tail(exc.stdout, limit=1_000_000)
        stderr = _tail(exc.stderr, limit=1_000_000)
        returncode = 124
        timed_out = True

    elapsed = time.monotonic() - started
    stdout_path.write_text(stdout or "")
    stderr_path.write_text(stderr or "")
    if stdout:
        print(f"[{name}] stdout tail:\n{stdout[-4096:]}")
    if stderr:
        print(f"[{name}] stderr tail:\n{stderr[-2048:]}", file=sys.stderr)
    return {
        "name": name,
        "command": cmd,
        "returncode": int(returncode),
        "timed_out": bool(timed_out),
        "elapsed_seconds": elapsed,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def _probe_cuda_environment(env: dict[str, str]) -> dict[str, Any]:
    import shutil
    import subprocess

    preflight: dict[str, Any] = {
        "schema_version": 1,
        "tool": "experiments/modal_alpha_geo0_pose_regen.py",
        "app": APP_NAME,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device_required": "cuda",
        "gpu_requested": "T4",
        "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
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
    except Exception as exc:  # pragma: no cover - remote diagnostics
        preflight["torch_probe_error"] = repr(exc)
        preflight["torch_cuda_available"] = False

    try:
        import nvidia.dali as dali

        preflight["nvidia_dali_import_ok"] = True
        preflight["nvidia_dali_version"] = getattr(dali, "__version__", None)
    except Exception as exc:  # pragma: no cover - remote diagnostics
        preflight["nvidia_dali_import_ok"] = False
        preflight["nvidia_dali_import_error"] = repr(exc)

    nvidia_smi = shutil.which("nvidia-smi")
    preflight["nvidia_smi_path"] = nvidia_smi
    if nvidia_smi:
        try:
            query = subprocess.check_output(
                [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=15,
            ).strip()
            preflight["nvidia_smi_query"] = query
            preflight["gpu_t4_match"] = "T4" in query
        except Exception as exc:  # pragma: no cover - remote diagnostics
            preflight["nvidia_smi_error"] = repr(exc)
            preflight["gpu_t4_match"] = False
    else:
        preflight["gpu_t4_match"] = False

    probe = REMOTE_REPO / "scripts/probe_nvdec.sh"
    if probe.is_file():
        try:
            proc = subprocess.run(
                ["bash", str(probe)],
                cwd=str(REMOTE_REPO),
                env={**env, "PYBIN": sys.executable},
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            preflight["nvdec_probe_returncode"] = int(proc.returncode)
            preflight["nvdec_probe_passed"] = proc.returncode == 0
            preflight["nvdec_probe_stdout_tail"] = _tail(proc.stdout)
            preflight["nvdec_probe_stderr_tail"] = _tail(proc.stderr)
        except Exception as exc:  # pragma: no cover - remote diagnostics
            preflight["nvdec_probe_returncode"] = 125
            preflight["nvdec_probe_passed"] = False
            preflight["nvdec_probe_error"] = repr(exc)
    else:
        preflight["nvdec_probe_passed"] = False
        preflight["nvdec_probe_error"] = f"missing probe script: {probe}"

    return preflight


def _preflight_errors(preflight: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if preflight.get("torch_cuda_available") is not True:
        errors.append("Modal runtime has no CUDA device; refusing CPU fallback")
    if preflight.get("nvidia_dali_import_ok") is not True:
        errors.append("nvidia.dali import failed; refusing non-DALI CUDA eval")
    if preflight.get("nvdec_probe_passed") is not True:
        errors.append("NVDEC/DALI video probe failed; upstream/evaluate.py CUDA path is unsafe")
    if preflight.get("gpu_t4_match") is not True:
        errors.append("Modal runtime did not report a T4 GPU")
    return errors


def _validate_contest_result(
    payload: dict[str, Any],
    *,
    expected_archive_sha256: str,
    expected_archive_size_bytes: int,
    require_t4: bool = True,
) -> list[str]:
    errors: list[str] = []
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return ["contest_auth_eval.json missing provenance object"]

    if provenance.get("device") != "cuda":
        errors.append(f"provenance.device={provenance.get('device')!r}, expected 'cuda'")
    if provenance.get("cuda_available") is not True:
        errors.append("provenance.cuda_available is not true")
    if require_t4 and provenance.get("gpu_t4_match") is not True:
        errors.append("provenance.gpu_t4_match is not true")
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


def _collect_artifacts(out_dir: Path) -> dict[str, bytes]:
    names = (
        "modal_alpha_geo0_preflight.json",
        "modal_alpha_geo0_validation.json",
        "modal_alpha_geo0_provenance.json",
        "modal_alpha_geo0_input_manifest.json",
        "modal_alpha_geo0_archive_manifest.json",
        "alpha_geo_0_geometry.json",
        "alpha_geo_0_primitive_contract.json",
        "candidate_masks_materialized.json",
        "diagnose_nerv_geometry.stdout.log",
        "diagnose_nerv_geometry.stderr.log",
        "optimize_poses.stdout.log",
        "optimize_poses.stderr.log",
        "contest_auth_eval.stdout.log",
        "contest_auth_eval.stderr.log",
        "adjudicate_contest_auth_eval.stdout.log",
        "adjudicate_contest_auth_eval.stderr.log",
        "archive_lane_12_alpha_geo0_pose_regen.zip",
        "adjudicated_contest_auth_eval.json",
        "pose_regen/optimized_poses.bin",
        "pose_regen/optimized_poses.meta",
        "eval_work/contest_auth_eval.json",
        "eval_work/provenance.json",
        "eval_work/report.txt",
    )
    artifacts: dict[str, bytes] = {}
    skipped: list[dict[str, Any]] = []
    for rel in names:
        path = out_dir / rel
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > 64 * 1024 * 1024:
            skipped.append({"path": rel, "bytes": size, "reason": "artifact_return_size_cap"})
            continue
        artifacts[rel] = path.read_bytes()
    if skipped:
        artifacts["modal_alpha_geo0_skipped_artifacts.json"] = _json_bytes({"skipped": skipped})
    return artifacts


def _finish(
    out_dir: Path,
    *,
    passed: bool,
    returncode: int,
    stage: str,
    validation_errors: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "tool": "experiments/modal_alpha_geo0_pose_regen.py",
        "app": APP_NAME,
        "passed": bool(passed),
        "returncode": int(returncode),
        "stage": stage,
        "validation_errors": validation_errors,
        "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
        "score_claim": bool(passed),
        "promotion_eligible": False,
        "allowed_use": ["cuda_auth_eval_review"] if passed else ["debug", "no_score_claim", "no_promotion"],
    }
    if extra:
        payload.update(extra)
    _write_json(out_dir / "modal_alpha_geo0_validation.json", payload)
    return {**payload, "artifacts": _collect_artifacts(out_dir)}


def _run_alpha_geo0_pose_regen_inner(
    *,
    candidate_archive_bytes: bytes,
    candidate_archive_sha256: str,
    candidate_archive_size_bytes: int,
    baseline_archive_bytes: bytes,
    baseline_archive_sha256: str,
    baseline_archive_size_bytes: int,
    warm_poses_bytes: bytes,
    warm_poses_sha256: str,
    warm_poses_size_bytes: int,
    gt_pose_targets_bytes: bytes,
    gt_pose_targets_sha256: str,
    gt_pose_targets_size_bytes: int,
    label: str,
    pose_steps: int,
    pose_batch_pairs: int,
    pose_lr: float,
    pose_seg_weight: float,
    pose_weight: float,
    inflate_timeout: int,
    evaluate_timeout: int,
    max_seconds: int,
) -> dict[str, Any]:
    import os
    import shutil

    label_safe = _safe_label(label)
    out_dir = REMOTE_OUT_ROOT / label_safe
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    candidate_archive = inputs_dir / "archive_lane_12_nerv.zip"
    baseline_archive = inputs_dir / "pfp16_baseline_archive.zip"
    warm_poses = inputs_dir / "warm_optimized_poses.pt"
    gt_pose_targets = inputs_dir / "gt_pose_targets.pt"
    candidate_archive.write_bytes(candidate_archive_bytes)
    baseline_archive.write_bytes(baseline_archive_bytes)
    warm_poses.write_bytes(warm_poses_bytes)
    gt_pose_targets.write_bytes(gt_pose_targets_bytes)

    stage = "input_custody"
    command_results: list[dict[str, Any]] = []
    env = {
        **os.environ,
        "PYTHONPATH": f"{REMOTE_REPO / 'src'}:{REMOTE_REPO / 'upstream'}:{REMOTE_REPO}",
        "TAC_UPSTREAM_DIR": str(REMOTE_REPO / "upstream"),
        "FFMPEG_BIN": "/usr/local/bin/ffmpeg-master",
        "UV_BIN": "/usr/local/bin/uv",
        "UV_LINK_MODE": "copy",
        "UV_PROJECT_ENVIRONMENT": str(out_dir / "uv_project_env"),
        "PYTHONHASHSEED": "1234",
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        "TAC_RESULTS_DIR": str(out_dir / "tac_results"),
    }

    try:
        observed_inputs = {
            "candidate_archive": _file_meta(candidate_archive),
            "baseline_archive": _file_meta(baseline_archive),
            "warm_poses": _file_meta(warm_poses),
            "gt_pose_targets": _file_meta(gt_pose_targets),
        }
        custody_errors = []
        expected = {
            "candidate_archive": (candidate_archive_sha256, candidate_archive_size_bytes),
            "baseline_archive": (baseline_archive_sha256, baseline_archive_size_bytes),
            "warm_poses": (warm_poses_sha256, warm_poses_size_bytes),
            "gt_pose_targets": (gt_pose_targets_sha256, gt_pose_targets_size_bytes),
        }
        for key, (sha, size) in expected.items():
            meta = observed_inputs[key]
            if meta["sha256"] != sha or meta["bytes"] != size:
                custody_errors.append(f"{key} custody mismatch")
        input_manifest = {
            "schema_version": 1,
            "label": label_safe,
            "inputs": observed_inputs,
            "candidate_archive_requirements": _validate_archive_requirements(
                candidate_archive,
                required_members={"renderer.bin", "masks.nrv"},
            ),
            "baseline_archive_requirements": _validate_archive_requirements(
                baseline_archive,
                required_members={"renderer.bin", "masks.mkv", "optimized_poses.bin"},
            ),
            "score_claim": False,
            "promotion_eligible": False,
        }
        _write_json(out_dir / "modal_alpha_geo0_input_manifest.json", input_manifest)
        if custody_errors:
            return _finish(
                out_dir,
                passed=False,
                returncode=5,
                stage=stage,
                validation_errors=custody_errors,
            )

        stage = "cuda_dali_nvdec_preflight"
        preflight = _probe_cuda_environment(env)
        _write_json(out_dir / "modal_alpha_geo0_preflight.json", preflight)
        errors = _preflight_errors(preflight)
        if errors:
            return _finish(out_dir, passed=False, returncode=6, stage=stage, validation_errors=errors)

        provenance = {
            "schema_version": 1,
            "lane": "lane_12_alpha_geo0_pose_regen_modal_t4",
            "label": label_safe,
            "status": "started",
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "purpose": "stale-pose isolation for exact Lane 12 masks.nrv; no NeRV retraining",
            "no_retraining": True,
            "candidate_masks_member": "masks.nrv",
            "canonical_score_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
            "inputs": observed_inputs,
            "preflight": preflight,
            "pose_regen": {
                "steps": int(pose_steps),
                "batch_pairs": int(pose_batch_pairs),
                "lr": float(pose_lr),
                "seg_weight": float(pose_seg_weight),
                "pose_weight": float(pose_weight),
                "eval_roundtrip": True,
                "posetto_noise_std": 0.5,
                "n_frames": N_FRAMES,
                "mask_height": MASK_H,
                "mask_width": MASK_W,
            },
            "score_claim": False,
            "promotion_eligible": False,
        }
        provenance_path = out_dir / "modal_alpha_geo0_provenance.json"
        _write_json(provenance_path, provenance)

        stage = "decode_candidate_masks"
        mask_cache_dir = out_dir / "mask_cache"
        alpha_geo_json = out_dir / "alpha_geo_0_geometry.json"
        alpha_contract_json = out_dir / "alpha_geo_0_primitive_contract.json"
        cmd = [
            sys.executable,
            "-u",
            str(REMOTE_REPO / "experiments/diagnose_nerv_geometry.py"),
            "--baseline",
            str(baseline_archive),
            "--baseline-member",
            "masks.mkv",
            "--candidate",
            str(candidate_archive),
            "--candidate-member",
            "masks.nrv",
            "--output-json",
            str(alpha_geo_json),
            "--primitive-contract-json",
            str(alpha_contract_json),
            "--mask-cache-dir",
            str(mask_cache_dir),
            "--threshold-preset",
            "promotion",
            "--num-frames",
            str(N_FRAMES),
            "--height",
            str(MASK_H),
            "--width",
            str(MASK_W),
        ]
        run = _run_logged(
            "diagnose_nerv_geometry",
            cmd,
            cwd=REMOTE_REPO,
            env=env,
            log_dir=out_dir,
            timeout=max(900, int(max_seconds)),
        )
        command_results.append(run)
        if run["returncode"] != 0:
            return _finish(
                out_dir,
                passed=False,
                returncode=run["returncode"],
                stage=stage,
                validation_errors=["diagnose_nerv_geometry failed"],
                extra={"commands": command_results},
            )

        candidate_masks_pt = out_dir / "candidate_masks.pt"
        mask_meta = _materialize_candidate_masks_from_cache(
            mask_cache_dir,
            candidate_archive_sha256=candidate_archive_sha256,
            output_path=candidate_masks_pt,
        )
        _write_json(out_dir / "candidate_masks_materialized.json", mask_meta)

        stage = "extract_candidate_payload"
        extract_dir = out_dir / "extracted_candidate"
        extract_meta = _safe_extract_members(candidate_archive, {"renderer.bin", "masks.nrv"}, extract_dir)

        stage = "optimize_poses"
        pose_dir = out_dir / "pose_regen"
        cmd = [
            sys.executable,
            "-u",
            str(REMOTE_REPO / "experiments/optimize_poses.py"),
            "--checkpoint",
            str(extract_dir / "renderer.bin"),
            "--masks",
            str(candidate_masks_pt),
            "--gt-poses-path",
            str(warm_poses),
            "--gt-pose-targets",
            str(gt_pose_targets),
            "--device",
            "cuda",
            "--n-frames",
            str(N_FRAMES),
            "--steps",
            str(int(pose_steps)),
            "--batch-pairs",
            str(int(pose_batch_pairs)),
            "--lr",
            str(float(pose_lr)),
            "--seg-weight",
            str(float(pose_seg_weight)),
            "--pose-weight",
            str(float(pose_weight)),
            "--eval-roundtrip",
            "--posetto-noise-std",
            "0.5",
            "--output-dir",
            str(pose_dir),
        ]
        run = _run_logged(
            "optimize_poses",
            cmd,
            cwd=REMOTE_REPO,
            env=env,
            log_dir=out_dir,
            timeout=max(900, int(max_seconds)),
        )
        command_results.append(run)
        if run["returncode"] != 0:
            return _finish(
                out_dir,
                passed=False,
                returncode=run["returncode"],
                stage=stage,
                validation_errors=["optimize_poses failed"],
                extra={"commands": command_results, "candidate_masks": mask_meta},
            )
        if not (pose_dir / "optimized_poses.bin").is_file():
            return _finish(
                out_dir,
                passed=False,
                returncode=7,
                stage=stage,
                validation_errors=["optimized_poses.bin was not produced"],
                extra={"commands": command_results, "candidate_masks": mask_meta},
            )

        stage = "deterministic_archive_rebuild"
        archive_src = out_dir / "archive_src"
        archive_src.mkdir(parents=True, exist_ok=True)
        shutil.copy2(extract_dir / "renderer.bin", archive_src / "renderer.bin")
        shutil.copy2(extract_dir / "masks.nrv", archive_src / "masks.nrv")
        shutil.copy2(pose_dir / "optimized_poses.bin", archive_src / "optimized_poses.bin")
        archive = out_dir / "archive_lane_12_alpha_geo0_pose_regen.zip"
        archive_manifest = _build_alpha_geo0_archive(archive_src, archive)
        archive_manifest.update(
            {
                "extract": extract_meta,
                "candidate_masks": mask_meta,
                "score_claim": False,
                "promotion_eligible": False,
            }
        )
        _write_json(out_dir / "modal_alpha_geo0_archive_manifest.json", archive_manifest)

        stage = "contest_auth_eval_cuda"
        eval_work = out_dir / "eval_work"
        cmd = [
            sys.executable,
            "-u",
            str(REMOTE_REPO / "experiments/contest_auth_eval.py"),
            "--archive",
            str(archive),
            "--inflate-sh",
            str(REMOTE_REPO / "submissions/robust_current/inflate.sh"),
            "--upstream-dir",
            str(REMOTE_REPO / "upstream"),
            "--video-names-file",
            str(REMOTE_REPO / "upstream/public_test_video_names.txt"),
            "--device",
            "cuda",
            "--keep-work-dir",
            "--work-dir",
            str(eval_work),
            "--inflate-timeout",
            str(int(inflate_timeout)),
            "--evaluate-timeout",
            str(int(evaluate_timeout)),
        ]
        run = _run_logged(
            "contest_auth_eval",
            cmd,
            cwd=REMOTE_REPO,
            env=env,
            log_dir=out_dir,
            timeout=int(inflate_timeout) + int(evaluate_timeout) + 600,
        )
        command_results.append(run)
        if run["returncode"] != 0:
            return _finish(
                out_dir,
                passed=False,
                returncode=run["returncode"],
                stage=stage,
                validation_errors=["contest_auth_eval failed"],
                extra={"commands": command_results, "archive": archive_manifest["archive"]},
            )

        contest_json = eval_work / "contest_auth_eval.json"
        if not contest_json.is_file():
            return _finish(
                out_dir,
                passed=False,
                returncode=8,
                stage=stage,
                validation_errors=["contest_auth_eval.json was not produced"],
                extra={"commands": command_results, "archive": archive_manifest["archive"]},
            )
        contest_payload = json.loads(contest_json.read_text())
        validation_errors = _validate_contest_result(
            contest_payload,
            expected_archive_sha256=archive_manifest["archive"]["sha256"],
            expected_archive_size_bytes=archive_manifest["archive"]["bytes"],
            require_t4=True,
        )
        if validation_errors:
            return _finish(
                out_dir,
                passed=False,
                returncode=9,
                stage=stage,
                validation_errors=validation_errors,
                extra={"commands": command_results, "archive": archive_manifest["archive"]},
            )

        stage = "adjudication"
        adjudicated_json = out_dir / "adjudicated_contest_auth_eval.json"
        cmd = [
            sys.executable,
            "-u",
            str(REMOTE_REPO / "scripts/adjudicate_contest_auth_eval.py"),
            "--contest-json",
            str(contest_json),
            "--provenance",
            str(provenance_path),
            "--archive",
            str(archive),
            "--result-copy",
            str(adjudicated_json),
            "--baseline-score",
            str(EXPECTED_BASELINE_SCORE),
            "--baseline-archive-bytes",
            str(EXPECTED_BASELINE_BYTES),
            "--predicted-band",
            "0.70",
            "1.30",
            "--regression-threshold",
            "1.30",
            "--delta-key",
            "score_delta_vs_pfp16_a_plus_plus",
            "--required-device",
            "cuda",
            "--required-samples",
            str(REQUIRED_SAMPLES),
            "--max-sane-score",
            "100.0",
            "--allow-component-gate-forensic-success",
        ]
        run = _run_logged(
            "adjudicate_contest_auth_eval",
            cmd,
            cwd=REMOTE_REPO,
            env=env,
            log_dir=out_dir,
            timeout=300,
        )
        command_results.append(run)
        if run["returncode"] != 0:
            return _finish(
                out_dir,
                passed=False,
                returncode=run["returncode"],
                stage=stage,
                validation_errors=["adjudicate_contest_auth_eval failed"],
                extra={"commands": command_results, "archive": archive_manifest["archive"]},
            )
        if not adjudicated_json.is_file():
            return _finish(
                out_dir,
                passed=False,
                returncode=10,
                stage=stage,
                validation_errors=["adjudicated result JSON was not produced"],
                extra={"commands": command_results, "archive": archive_manifest["archive"]},
            )

        adjudicated_payload = json.loads(adjudicated_json.read_text())
        final_extra = {
            "commands": command_results,
            "archive_path": str(archive),
            "archive_sha256": archive_manifest["archive"]["sha256"],
            "archive_size_bytes": archive_manifest["archive"]["bytes"],
            "score_recomputed_from_components": contest_payload.get("score_recomputed_from_components"),
            "final_score": contest_payload.get("final_score"),
            "avg_posenet_dist": contest_payload.get("avg_posenet_dist"),
            "avg_segnet_dist": contest_payload.get("avg_segnet_dist"),
            "n_samples": contest_payload.get("n_samples"),
            "adjudicated_result_json": str(adjudicated_json),
            "adjudication_lane_status": adjudicated_payload.get("lane_status"),
            "promotion_eligible": bool(adjudicated_payload.get("promotion_eligible") is True),
            "component_gate_triggered": bool(adjudicated_payload.get("component_gate_triggered") is True),
            "allowed_use": ["exact_cuda_score_review", "adjudicated"],
        }
        return _finish(
            out_dir,
            passed=True,
            returncode=0,
            stage="done",
            validation_errors=[],
            extra=final_extra,
        )
    except Exception as exc:  # pragma: no cover - remote diagnostics
        return _finish(
            out_dir,
            passed=False,
            returncode=99,
            stage=stage,
            validation_errors=[f"{type(exc).__name__}: {exc}"],
            extra={"traceback": traceback.format_exc(), "commands": command_results},
        )


@app.function(image=run_image, gpu="T4", timeout=6 * 3600)
def run_alpha_geo0_pose_regen_t4(
    candidate_archive_bytes: bytes,
    candidate_archive_sha256: str,
    candidate_archive_size_bytes: int,
    baseline_archive_bytes: bytes,
    baseline_archive_sha256: str,
    baseline_archive_size_bytes: int,
    warm_poses_bytes: bytes,
    warm_poses_sha256: str,
    warm_poses_size_bytes: int,
    gt_pose_targets_bytes: bytes,
    gt_pose_targets_sha256: str,
    gt_pose_targets_size_bytes: int,
    label: str,
    pose_steps: int = 500,
    pose_batch_pairs: int = 8,
    pose_lr: float = 0.01,
    pose_seg_weight: float = 100.0,
    pose_weight: float = 10.0,
    inflate_timeout: int = 1800,
    evaluate_timeout: int = 1800,
    max_seconds: int = 6 * 3600,
) -> dict[str, Any]:
    return _run_alpha_geo0_pose_regen_inner(
        candidate_archive_bytes=candidate_archive_bytes,
        candidate_archive_sha256=candidate_archive_sha256,
        candidate_archive_size_bytes=candidate_archive_size_bytes,
        baseline_archive_bytes=baseline_archive_bytes,
        baseline_archive_sha256=baseline_archive_sha256,
        baseline_archive_size_bytes=baseline_archive_size_bytes,
        warm_poses_bytes=warm_poses_bytes,
        warm_poses_sha256=warm_poses_sha256,
        warm_poses_size_bytes=warm_poses_size_bytes,
        gt_pose_targets_bytes=gt_pose_targets_bytes,
        gt_pose_targets_sha256=gt_pose_targets_sha256,
        gt_pose_targets_size_bytes=gt_pose_targets_size_bytes,
        label=label,
        pose_steps=pose_steps,
        pose_batch_pairs=pose_batch_pairs,
        pose_lr=pose_lr,
        pose_seg_weight=pose_seg_weight,
        pose_weight=pose_weight,
        inflate_timeout=inflate_timeout,
        evaluate_timeout=evaluate_timeout,
        max_seconds=max_seconds,
    )


def _read_input(path: Path, label: str) -> tuple[bytes, str, int]:
    if not path.is_file():
        raise SystemExit(f"FATAL: {label} not found: {path}")
    data = path.read_bytes()
    return data, _sha256_bytes(data), len(data)


def _write_dispatch_metadata(
    *,
    label: str,
    call_id: str,
    paths: dict[str, Path],
    metas: dict[str, dict[str, Any]],
    params: dict[str, Any],
) -> Path:
    out_dir = RESULT_ROOT / _safe_label(label)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "app": APP_NAME,
        "label": _safe_label(label),
        "call_id": call_id,
        "gpu": "T4",
        "paths": {key: str(path) for key, path in sorted(paths.items())},
        "inputs": metas,
        "params": params,
        "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
        "score_claim": False,
        "promotion_eligible": False,
        "recover_command": (
            ".venv/bin/python experiments/modal_alpha_geo0_pose_regen.py recover "
            f"--label {_safe_label(label)}"
        ),
        "dispatched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = out_dir / "modal_alpha_geo0_call.json"
    _write_json(path, payload)
    (out_dir / "modal_call_id.txt").write_text(call_id + "\n")
    return path


@app.local_entrypoint()
def main(
    candidate_archive: str = str(DEFAULT_CANDIDATE_ARCHIVE),
    baseline_archive: str = str(DEFAULT_BASELINE_ARCHIVE),
    warm_poses: str = str(DEFAULT_WARM_POSES),
    gt_pose_targets: str = str(DEFAULT_GT_POSE_TARGETS),
    label: str = "lane12_alpha_geo0_pose_regen_modal_t4_20260501",
    pose_steps: int = 500,
    pose_batch_pairs: int = 8,
    pose_lr: float = 0.01,
    pose_seg_weight: float = 100.0,
    pose_weight: float = 10.0,
    inflate_timeout: int = 1800,
    evaluate_timeout: int = 1800,
    timeout_hours: float = 6.0,
) -> None:
    candidate_path = Path(candidate_archive).resolve()
    baseline_path = Path(baseline_archive).resolve()
    warm_path = Path(warm_poses).resolve()
    targets_path = Path(gt_pose_targets).resolve()

    candidate_bytes, candidate_sha, candidate_size = _read_input(candidate_path, "candidate archive")
    baseline_bytes, baseline_sha, baseline_size = _read_input(baseline_path, "baseline archive")
    warm_bytes, warm_sha, warm_size = _read_input(warm_path, "warm poses")
    target_bytes, target_sha, target_size = _read_input(targets_path, "GT pose targets")

    if pose_steps <= 0 or pose_batch_pairs <= 0:
        raise SystemExit("FATAL: pose_steps and pose_batch_pairs must be positive")
    max_seconds = max(600, min(int(float(timeout_hours) * 3600), 6 * 3600))
    params = {
        "pose_steps": int(pose_steps),
        "pose_batch_pairs": int(pose_batch_pairs),
        "pose_lr": float(pose_lr),
        "pose_seg_weight": float(pose_seg_weight),
        "pose_weight": float(pose_weight),
        "inflate_timeout": int(inflate_timeout),
        "evaluate_timeout": int(evaluate_timeout),
        "max_seconds": int(max_seconds),
        "n_frames": N_FRAMES,
    }

    call = run_alpha_geo0_pose_regen_t4.spawn(
        candidate_bytes,
        candidate_sha,
        candidate_size,
        baseline_bytes,
        baseline_sha,
        baseline_size,
        warm_bytes,
        warm_sha,
        warm_size,
        target_bytes,
        target_sha,
        target_size,
        _safe_label(label),
        int(pose_steps),
        int(pose_batch_pairs),
        float(pose_lr),
        float(pose_seg_weight),
        float(pose_weight),
        int(inflate_timeout),
        int(evaluate_timeout),
        int(max_seconds),
    )
    metadata_path = _write_dispatch_metadata(
        label=label,
        call_id=call.object_id,
        paths={
            "candidate_archive": candidate_path,
            "baseline_archive": baseline_path,
            "warm_poses": warm_path,
            "gt_pose_targets": targets_path,
        },
        metas={
            "candidate_archive": {"sha256": candidate_sha, "bytes": candidate_size},
            "baseline_archive": {"sha256": baseline_sha, "bytes": baseline_size},
            "warm_poses": {"sha256": warm_sha, "bytes": warm_size},
            "gt_pose_targets": {"sha256": target_sha, "bytes": target_size},
        },
        params=params,
    )
    print(f"DISPATCHED Alpha-Geo-0 Modal T4 call_id={call.object_id}")
    print(f"metadata: {metadata_path}")
    print(f"recover: .venv/bin/python experiments/modal_alpha_geo0_pose_regen.py recover --label {_safe_label(label)}")


def recover(label: str) -> int:
    label_safe = _safe_label(label)
    out_dir = RESULT_ROOT / label_safe
    metadata_path = out_dir / "modal_alpha_geo0_call.json"
    if not metadata_path.is_file():
        print(f"FATAL: missing {metadata_path}", file=sys.stderr)
        return 2
    metadata = json.loads(metadata_path.read_text())
    call_id = metadata.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        print(f"FATAL: invalid call_id in {metadata_path}", file=sys.stderr)
        return 2

    fc = modal.FunctionCall.from_id(call_id)
    try:
        result = fc.get(timeout=0)
    except TimeoutError:
        print(json.dumps({"label": label_safe, "status": "pending", "call_id": call_id}, indent=2, sort_keys=True))
        return 1

    if not isinstance(result, dict):
        print(f"FATAL: Modal result for {call_id} was not a dict", file=sys.stderr)
        return 3

    artifacts = result.get("artifacts")
    saved = 0
    if isinstance(artifacts, dict):
        for name, data in sorted(artifacts.items()):
            rel = Path(str(name))
            if rel.is_absolute() or ".." in rel.parts:
                print(f"FATAL: unsafe artifact path returned by Modal: {name!r}", file=sys.stderr)
                return 4
            path = out_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            saved += 1

    summary = {key: value for key, value in result.items() if key != "artifacts"}
    summary["saved_artifacts"] = saved
    summary["result_dir"] = str(out_dir)
    _write_json(out_dir / "modal_alpha_geo0_result_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result.get("passed") else int(result.get("returncode") or 1)


def _recover_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Recover Modal Alpha-Geo-0 pose-regeneration artifacts.")
    parser.add_argument("command", choices=["recover"])
    parser.add_argument("--label", required=True)
    args = parser.parse_args(argv)
    return recover(args.label)


if __name__ == "__main__":
    raise SystemExit(_recover_cli(sys.argv[1:]))
