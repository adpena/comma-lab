#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run Alpha-Geo-0 stale-pose isolation with deterministic archive custody.

This experiment keeps the exact Lane 12 ``masks.nrv`` payload and renderer,
decodes the candidate masks, regenerates ``optimized_poses.bin`` against those
masks, rebuilds a deterministic archive, then runs canonical CUDA auth eval.

It performs no NeRV retraining. A bad score from this script is evidence about
the measured Lane 12 payload plus regenerated poses, not a family kill.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_SAMPLES = 600
N_FRAMES = 1200
MASK_H = 384
MASK_W = 512
EXPECTED_BASELINE_SCORE = 1.043987524793892
EXPECTED_BASELINE_BYTES = 686635


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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


def _tail(value: str | bytes | None, limit: int = 4096) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-limit:]


def _validated_zip_infos(path: Path) -> dict[str, zipfile.ZipInfo]:
    infos: dict[str, zipfile.ZipInfo] = {}
    with zipfile.ZipFile(path, "r") as zf:
        for info in zf.infolist():
            member_path = PurePosixPath(info.filename)
            if info.is_dir() or member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"unsafe archive member path: {info.filename!r}")
            if info.filename in infos:
                raise ValueError(f"duplicate archive member: {info.filename!r}")
            if info.filename.startswith(".") or "/." in info.filename or info.filename.startswith("__MACOSX/"):
                raise ValueError(f"hidden/resource-fork archive member: {info.filename!r}")
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


def _materialize_candidate_masks_from_cache(
    cache_dir: Path,
    *,
    candidate_archive_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
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
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        returncode = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = _tail(exc.stdout, limit=1_000_000)
        stderr = _tail(exc.stderr, limit=1_000_000)
        returncode = 124
        timed_out = True

    elapsed = time.monotonic() - started
    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)
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


def _validate_contest_result(
    payload: dict[str, Any],
    *,
    expected_archive_sha256: str,
    expected_archive_size_bytes: int,
    require_t4: bool,
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
        errors.append("provenance.archive_sha256 mismatch")
    if payload.get("archive_size_bytes") != expected_archive_size_bytes:
        errors.append("archive_size_bytes mismatch")
    if payload.get("n_samples") != REQUIRED_SAMPLES:
        errors.append(f"n_samples={payload.get('n_samples')!r}, expected {REQUIRED_SAMPLES}")
    score = payload.get("score_recomputed_from_components")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)):
        errors.append("score_recomputed_from_components is missing or non-finite")
    return errors


def _probe_nvdec(*, repo_root: Path, env: dict[str, str], log_dir: Path, timeout: int) -> dict[str, Any]:
    probe = repo_root / "scripts/probe_nvdec.sh"
    if not probe.is_file():
        return {"passed": False, "returncode": 127, "error": f"missing probe script: {probe}"}
    run = _run_logged(
        "probe_nvdec",
        ["bash", str(probe)],
        cwd=repo_root,
        env={**env, "PYBIN": sys.executable},
        log_dir=log_dir,
        timeout=timeout,
    )
    return {**run, "passed": run["returncode"] == 0}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-archive", required=True)
    parser.add_argument("--baseline-archive", required=True)
    parser.add_argument("--warm-poses", required=True)
    parser.add_argument("--gt-pose-targets", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--upstream-dir", default="upstream")
    parser.add_argument("--inflate-sh", default="submissions/robust_current/inflate.sh")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--pose-steps", type=int, default=500)
    parser.add_argument("--pose-batch-pairs", type=int, default=8)
    parser.add_argument("--pose-lr", type=float, default=0.01)
    parser.add_argument("--pose-seg-weight", type=float, default=100.0)
    parser.add_argument("--pose-weight", type=float, default=10.0)
    parser.add_argument("--posetto-noise-std", type=float, default=0.5)
    parser.add_argument("--inflate-timeout", type=int, default=1800)
    parser.add_argument("--evaluate-timeout", type=int, default=1800)
    parser.add_argument("--max-seconds", type=int, default=6 * 3600)
    parser.add_argument("--skip-nvdec-preflight", action="store_true")
    parser.add_argument("--allow-non-t4", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for rel in (
        "archive_src",
        "eval_work",
        "extracted_candidate",
        "mask_cache",
        "pose_regen",
        "tac_results",
        "alpha_geo0_archive_manifest.json",
        "alpha_geo0_input_manifest.json",
        "alpha_geo0_nvdec_preflight.json",
        "alpha_geo0_summary.json",
        "alpha_geo_0_geometry.json",
        "alpha_geo_0_primitive_contract.json",
        "archive.zip",
        "candidate_masks.pt",
        "candidate_masks_materialized.json",
        "contest_auth_eval.json",
        "contest_auth_eval.adjudicated.json",
        "eval_provenance.json",
        "report.txt",
        "adjudication_provenance.json",
        "adjudicate_contest_auth_eval.stderr.log",
        "adjudicate_contest_auth_eval.stdout.log",
        "contest_auth_eval.stderr.log",
        "contest_auth_eval.stdout.log",
        "diagnose_nerv_geometry.stderr.log",
        "diagnose_nerv_geometry.stdout.log",
        "optimize_poses.stderr.log",
        "optimize_poses.stdout.log",
        "probe_nvdec.stderr.log",
        "probe_nvdec.stdout.log",
    ):
        path = output_dir / rel
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    candidate_archive = Path(args.candidate_archive).resolve()
    baseline_archive = Path(args.baseline_archive).resolve()
    warm_poses = Path(args.warm_poses).resolve()
    gt_pose_targets = Path(args.gt_pose_targets).resolve()
    upstream_dir = Path(args.upstream_dir)
    if not upstream_dir.is_absolute():
        upstream_dir = repo_root / upstream_dir
    inflate_sh = Path(args.inflate_sh)
    if not inflate_sh.is_absolute():
        inflate_sh = repo_root / inflate_sh

    command_results: list[dict[str, Any]] = []
    stage = "input_custody"
    env = {
        **os.environ,
        "PYTHONPATH": f"{repo_root / 'src'}:{repo_root / 'upstream'}:{repo_root}",
        "TAC_UPSTREAM_DIR": str(upstream_dir),
        "UV_LINK_MODE": "copy",
        "UV_PROJECT_ENVIRONMENT": os.environ.get(
            "UV_PROJECT_ENVIRONMENT",
            str(output_dir / "uv_project_env"),
        ),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "1234"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        "TAC_RESULTS_DIR": str(output_dir / "tac_results"),
    }
    if "TAC_FFMPEG" not in env and shutil.which("ffmpeg"):
        env["TAC_FFMPEG"] = shutil.which("ffmpeg") or "ffmpeg"

    try:
        observed_inputs = {
            "candidate_archive": _file_meta(candidate_archive),
            "baseline_archive": _file_meta(baseline_archive),
            "warm_poses": _file_meta(warm_poses),
            "gt_pose_targets": _file_meta(gt_pose_targets),
        }
        input_manifest = {
            "schema_version": 1,
            "tool": "experiments/alpha_geo0_pose_regen.py",
            "started_at_utc": _utc_now(),
            "purpose": "Lane 12 stale-pose isolation; no NeRV retraining",
            "canonical_score_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
            "score_claim": False,
            "promotion_eligible": False,
            "inputs": observed_inputs,
            "candidate_archive_requirements": _validate_archive_requirements(
                candidate_archive,
                required_members={"renderer.bin", "masks.nrv"},
            ),
            "baseline_archive_requirements": _validate_archive_requirements(
                baseline_archive,
                required_members={"renderer.bin", "masks.mkv", "optimized_poses.bin"},
            ),
            "pose_regen": {
                "steps": int(args.pose_steps),
                "batch_pairs": int(args.pose_batch_pairs),
                "lr": float(args.pose_lr),
                "seg_weight": float(args.pose_seg_weight),
                "pose_weight": float(args.pose_weight),
                "eval_roundtrip": True,
                "posetto_noise_std": float(args.posetto_noise_std),
                "n_frames": N_FRAMES,
                "mask_height": MASK_H,
                "mask_width": MASK_W,
            },
        }
        _write_json(output_dir / "alpha_geo0_input_manifest.json", input_manifest)

        if not args.skip_nvdec_preflight:
            stage = "nvdec_preflight"
            preflight = _probe_nvdec(repo_root=repo_root, env=env, log_dir=output_dir, timeout=180)
            _write_json(output_dir / "alpha_geo0_nvdec_preflight.json", preflight)
            command_results.append(preflight)
            if not preflight.get("passed"):
                raise RuntimeError("NVDEC/DALI probe failed; refusing non-authoritative CUDA eval")

        stage = "decode_candidate_masks"
        mask_cache_dir = output_dir / "mask_cache"
        alpha_geo_json = output_dir / "alpha_geo_0_geometry.json"
        alpha_contract_json = output_dir / "alpha_geo_0_primitive_contract.json"
        run = _run_logged(
            "diagnose_nerv_geometry",
            [
                sys.executable,
                "-u",
                str(repo_root / "experiments/diagnose_nerv_geometry.py"),
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
            ],
            cwd=repo_root,
            env=env,
            log_dir=output_dir,
            timeout=max(900, int(args.max_seconds)),
        )
        command_results.append(run)
        if run["returncode"] != 0:
            raise RuntimeError("diagnose_nerv_geometry failed")

        candidate_masks_pt = output_dir / "candidate_masks.pt"
        mask_meta = _materialize_candidate_masks_from_cache(
            mask_cache_dir,
            candidate_archive_sha256=observed_inputs["candidate_archive"]["sha256"],
            output_path=candidate_masks_pt,
        )
        _write_json(output_dir / "candidate_masks_materialized.json", mask_meta)

        stage = "extract_candidate_payload"
        extract_dir = output_dir / "extracted_candidate"
        extract_meta = _safe_extract_members(candidate_archive, {"renderer.bin", "masks.nrv"}, extract_dir)

        stage = "optimize_poses"
        pose_dir = output_dir / "pose_regen"
        run = _run_logged(
            "optimize_poses",
            [
                sys.executable,
                "-u",
                str(repo_root / "experiments/optimize_poses.py"),
                "--checkpoint",
                str(extract_dir / "renderer.bin"),
                "--masks",
                str(candidate_masks_pt),
                "--gt-poses-path",
                str(warm_poses),
                "--gt-pose-targets",
                str(gt_pose_targets),
                "--device",
                args.device,
                "--n-frames",
                str(N_FRAMES),
                "--steps",
                str(int(args.pose_steps)),
                "--batch-pairs",
                str(int(args.pose_batch_pairs)),
                "--lr",
                str(float(args.pose_lr)),
                "--seg-weight",
                str(float(args.pose_seg_weight)),
                "--pose-weight",
                str(float(args.pose_weight)),
                "--eval-roundtrip",
                "--posetto-noise-std",
                str(float(args.posetto_noise_std)),
                "--output-dir",
                str(pose_dir),
            ],
            cwd=repo_root,
            env=env,
            log_dir=output_dir,
            timeout=max(900, int(args.max_seconds)),
        )
        command_results.append(run)
        if run["returncode"] != 0:
            raise RuntimeError("optimize_poses failed")
        if not (pose_dir / "optimized_poses.bin").is_file():
            raise RuntimeError("optimized_poses.bin was not produced")

        stage = "deterministic_archive_rebuild"
        archive_src = output_dir / "archive_src"
        archive_src.mkdir(parents=True, exist_ok=True)
        shutil.copy2(extract_dir / "renderer.bin", archive_src / "renderer.bin")
        shutil.copy2(extract_dir / "masks.nrv", archive_src / "masks.nrv")
        shutil.copy2(pose_dir / "optimized_poses.bin", archive_src / "optimized_poses.bin")
        archive_manifest = _build_alpha_geo0_archive(archive_src, output_dir / "archive.zip")
        archive_manifest.update(
            {
                "extract": extract_meta,
                "candidate_masks": mask_meta,
                "score_claim": False,
                "promotion_eligible": False,
            }
        )
        _write_json(output_dir / "alpha_geo0_archive_manifest.json", archive_manifest)

        stage = "contest_auth_eval_cuda"
        eval_work = output_dir / "eval_work"
        run = _run_logged(
            "contest_auth_eval",
            [
                sys.executable,
                "-u",
                str(repo_root / "experiments/contest_auth_eval.py"),
                "--archive",
                str(output_dir / "archive.zip"),
                "--inflate-sh",
                str(inflate_sh),
                "--upstream-dir",
                str(upstream_dir),
                "--video-names-file",
                str(upstream_dir / "public_test_video_names.txt"),
                "--device",
                args.device,
                "--keep-work-dir",
                "--work-dir",
                str(eval_work),
                "--inflate-timeout",
                str(int(args.inflate_timeout)),
                "--evaluate-timeout",
                str(int(args.evaluate_timeout)),
            ],
            cwd=repo_root,
            env=env,
            log_dir=output_dir,
            timeout=int(args.inflate_timeout) + int(args.evaluate_timeout) + 600,
        )
        command_results.append(run)
        if run["returncode"] != 0:
            raise RuntimeError("contest_auth_eval failed")

        contest_json = eval_work / "contest_auth_eval.json"
        if not contest_json.is_file():
            raise RuntimeError("contest_auth_eval.json was not produced")
        contest_payload = json.loads(contest_json.read_text())
        validation_errors = _validate_contest_result(
            contest_payload,
            expected_archive_sha256=archive_manifest["archive"]["sha256"],
            expected_archive_size_bytes=archive_manifest["archive"]["bytes"],
            require_t4=not args.allow_non_t4,
        )
        if validation_errors:
            raise RuntimeError("; ".join(validation_errors))
        shutil.copy2(contest_json, output_dir / "contest_auth_eval.json")
        shutil.copy2(eval_work / "provenance.json", output_dir / "eval_provenance.json")
        shutil.copy2(eval_work / "report.txt", output_dir / "report.txt")

        stage = "adjudication"
        run = _run_logged(
            "adjudicate_contest_auth_eval",
            [
                sys.executable,
                "-u",
                str(repo_root / "scripts/adjudicate_contest_auth_eval.py"),
                "--contest-json",
                str(output_dir / "contest_auth_eval.json"),
                "--provenance",
                str(output_dir / "adjudication_provenance.json"),
                "--archive",
                str(output_dir / "archive.zip"),
                "--result-copy",
                str(output_dir / "contest_auth_eval.adjudicated.json"),
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
            ],
            cwd=repo_root,
            env=env,
            log_dir=output_dir,
            timeout=300,
        )
        command_results.append(run)
        if run["returncode"] != 0:
            raise RuntimeError("adjudicate_contest_auth_eval failed")

        adjudicated_payload = json.loads((output_dir / "contest_auth_eval.adjudicated.json").read_text())
        summary = {
            "schema_version": 1,
            "tool": "experiments/alpha_geo0_pose_regen.py",
            "completed_at_utc": _utc_now(),
            "stage": "done",
            "passed": True,
            "score_claim": True,
            "promotion_eligible": bool(adjudicated_payload.get("promotion_eligible") is True),
            "archive_sha256": archive_manifest["archive"]["sha256"],
            "archive_size_bytes": archive_manifest["archive"]["bytes"],
            "score_recomputed_from_components": contest_payload.get("score_recomputed_from_components"),
            "final_score": contest_payload.get("final_score"),
            "avg_posenet_dist": contest_payload.get("avg_posenet_dist"),
            "avg_segnet_dist": contest_payload.get("avg_segnet_dist"),
            "n_samples": contest_payload.get("n_samples"),
            "adjudication_lane_status": adjudicated_payload.get("lane_status"),
            "component_gate_triggered": bool(adjudicated_payload.get("component_gate_triggered") is True),
            "commands": command_results,
            "inputs": observed_inputs,
        }
        _write_json(output_dir / "alpha_geo0_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "tool": "experiments/alpha_geo0_pose_regen.py",
            "completed_at_utc": _utc_now(),
            "stage": stage,
            "passed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "commands": command_results,
        }
        _write_json(output_dir / "alpha_geo0_summary.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 99


if __name__ == "__main__":
    raise SystemExit(main())
