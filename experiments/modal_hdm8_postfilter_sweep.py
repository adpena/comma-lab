# SPDX-License-Identifier: MIT
"""Run the HDM8 postfilter/selector screen on Modal T4 CUDA.

This is a CUDA-in-the-loop proxy-prefix scorer, not an exact-eval score claim.
It uploads one archive, runs ``tools/screen_hdm8_postfilter_sweep.py`` with
``--device cuda`` on a Modal T4, and writes per-mode/per-pair component arrays
for selector learning.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import modal

from experiments import modal_auth_eval as modal_auth_eval_module
from tac.deploy.claims import (
    DispatchClaimSpec,
    predicted_eta,
    record_dispatch_claim,
    terminal_dispatch_claim,
    utc_now,
)
from tac.deploy.modal.auth_eval import (
    ModalArtifactWriteError,
    function_call_id,
    materialize_modal_artifacts,
    safe_artifact_label,
    sha256_bytes,
)
from tac.repo_io import read_json, write_json
from tac.reproducibility import collect_source_transparency

APP_NAME = "comma-hdm8-postfilter-sweep"
AXIS = "modal-t4-cuda-proxy-prefix"
REMOTE_REPO = Path("/workspace/pact")
REMOTE_OUT = Path("/tmp/modal_hdm8_postfilter_sweep")
REMOTE_WORK_ROOT = Path("/root/modal_hdm8_postfilter_sweep_work")
REMOTE_PYTHONPATH = modal_auth_eval_module.REMOTE_PYTHONPATH
AUTH_EVAL_IMAGE = modal_auth_eval_module.eval_image
DALI_DISABLE_NVML_VALUE = modal_auth_eval_module.DALI_DISABLE_NVML_VALUE
DEFAULT_OUTPUT_ROOT = Path("experiments/results/modal_hdm8_postfilter_sweep")
DEFAULT_MODES = (
    "none",
    "unsharp:0.10",
    "unsharp:0.20",
    "unsharp:0.35",
    "adaptive:0.50",
    "adaptive:0.85",
    "soften:0.05",
)
RESULT_JSON_NAME = "modal_hdm8_postfilter_sweep_result.json"
SWEEP_JSON_NAME = "hdm8_postfilter_sweep.json"
SPAWN_JSON_NAME = "modal_hdm8_postfilter_sweep_spawn.json"
RECOVER_JSON_NAME = "modal_hdm8_postfilter_sweep_recover_summary.json"

app = modal.App(APP_NAME)

sweep_image = (
    AUTH_EVAL_IMAGE
    .add_local_file(  # MODAL_MANUAL_MOUNT_OK:narrow HDM8 postfilter scorer sweep
        "tools/screen_hdm8_postfilter_sweep.py",
        remote_path=str(REMOTE_REPO / "tools/screen_hdm8_postfilter_sweep.py"),
    )
    .add_local_dir(  # MODAL_MANUAL_MOUNT_OK:narrow HDM8 postfilter scorer sweep
        "submissions/pr106_latent_sidecar_r2_pr101_grammar",
        remote_path=str(
            REMOTE_REPO / "submissions/pr106_latent_sidecar_r2_pr101_grammar"
        ),
    )
)


def _local_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
    except Exception as exc:  # pragma: no cover - local diagnostic fallback
        return f"<error obtaining local git commit: {exc!r}>"


def _probe_cuda_environment() -> dict[str, Any]:
    import shutil

    preflight: dict[str, Any] = {
        "schema_version": 1,
        "tool": "experiments/modal_hdm8_postfilter_sweep.py",
        "app": APP_NAME,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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


def _coerce_mode_list(raw: Any, *, source: str) -> list[str]:
    if not isinstance(raw, list):
        raise SystemExit(f"FATAL: {source} must contain a JSON list of modes")
    modes: list[str] = []
    for item in raw:
        mode = item.get("mode") if isinstance(item, dict) else item
        if not isinstance(mode, str) or not mode.strip():
            raise SystemExit(f"FATAL: {source} contains a non-string mode: {item!r}")
        modes.append(mode.strip())
    return modes


def parse_modes(*, modes: str = "", modes_from_json: str = "") -> list[str]:
    """Parse Modal-friendly mode input.

    ``modes`` accepts either a JSON string list or a semicolon/newline-separated
    list. Semicolons avoid ambiguity with modes like ``even_rgb_bias:2,-1,-1``.
    """

    if modes_from_json:
        payload = read_json(Path(modes_from_json))
        if not isinstance(payload, dict):
            raise SystemExit("FATAL: --modes-from-json must point to a JSON object")
        return _validate_modes(_coerce_mode_list(payload.get("modes"), source=modes_from_json))

    text = str(modes or "").strip()
    if not text:
        return list(DEFAULT_MODES)
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"FATAL: --modes is not valid JSON: {exc}") from exc
        return _validate_modes(_coerce_mode_list(parsed, source="--modes"))
    parts = [part.strip() for part in text.replace("\n", ";").split(";")]
    return _validate_modes([part for part in parts if part])


def _validate_modes(modes: list[str]) -> list[str]:
    if not modes:
        raise SystemExit("FATAL: at least one --modes entry is required")
    if "none" not in modes:
        raise SystemExit("FATAL: modes must include 'none' so delta_vs_none is defined")
    duplicates = sorted({mode for mode in modes if modes.count(mode) > 1})
    if duplicates:
        raise SystemExit(f"FATAL: duplicate modes are not allowed: {duplicates}")
    return modes


def _validate_positive_int(name: str, value: int) -> int:
    if int(value) <= 0:
        raise SystemExit(f"FATAL: {name} must be positive")
    return int(value)


def _prepare_local_request(
    *,
    archive: str | Path,
    output_dir: str | Path,
    n_pairs: int,
    modes: list[str],
    include_per_pair: bool,
    decode_batch_pairs: int,
    score_batch_pairs: int,
    mode_batch_size: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    archive_path = Path(archive).resolve()
    if not archive_path.is_file():
        raise SystemExit(f"FATAL: archive not found: {archive_path}")
    archive_bytes = archive_path.read_bytes()
    archive_sha256 = sha256_bytes(archive_bytes)
    out_dir = (
        Path(output_dir).resolve()
        if str(output_dir or "")
        else (
            Path.cwd()
            / DEFAULT_OUTPUT_ROOT
            / f"{safe_artifact_label(archive_path.stem)}_{archive_sha256[:12]}"
        ).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "schema_version": 1,
        "tool": "experiments/modal_hdm8_postfilter_sweep.py",
        "app": APP_NAME,
        "axis": AXIS,
        "archive_path": str(archive_path),
        "archive_bytes_payload": archive_bytes,
        "archive_sha256": archive_sha256,
        "archive_size_bytes": len(archive_bytes),
        "output_dir": str(out_dir),
        "n_pairs": _validate_positive_int("--n-pairs", n_pairs),
        "modes": modes,
        "include_per_pair": bool(include_per_pair),
        "decode_batch_pairs": _validate_positive_int("--decode-batch-pairs", decode_batch_pairs),
        "score_batch_pairs": _validate_positive_int("--score-batch-pairs", score_batch_pairs),
        "mode_batch_size": _validate_positive_int("--mode-batch-size", mode_batch_size),
        "timeout_seconds": _validate_positive_int("--timeout-seconds", timeout_seconds),
        "source_repo_commit": _local_git_commit(),
        "score_claim": False,
        "promotion_eligible": False,
    }


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_sweep_payload(
    payload: dict[str, Any],
    *,
    expected_modes: list[str],
    expected_n_pairs: int,
    include_per_pair: bool,
    expected_archive_sha256: str,
    expected_archive_size_bytes: int,
) -> list[str]:
    errors: list[str] = []
    if payload.get("axis") != AXIS:
        errors.append(f"axis={payload.get('axis')!r}, expected {AXIS!r}")
    if payload.get("score_claim") is not False:
        errors.append("score_claim must be false")
    if payload.get("promotion_eligible") is not False:
        errors.append("promotion_eligible must be false")
    if payload.get("archive_sha256") != expected_archive_sha256:
        errors.append("archive_sha256 mismatch")
    if payload.get("archive_bytes") != expected_archive_size_bytes:
        errors.append("archive_bytes mismatch")
    if payload.get("n_pairs") != expected_n_pairs:
        errors.append(f"n_pairs={payload.get('n_pairs')!r}, expected {expected_n_pairs}")

    mode_rows = payload.get("modes")
    if not isinstance(mode_rows, list) or not mode_rows:
        errors.append("modes must be a non-empty list")
        return errors
    observed_modes = [str(row.get("mode")) for row in mode_rows if isinstance(row, dict)]
    if observed_modes != expected_modes:
        errors.append(f"mode order mismatch: {observed_modes!r} != {expected_modes!r}")

    for row in mode_rows:
        if not isinstance(row, dict):
            errors.append(f"mode row must be an object, got {type(row).__name__}")
            continue
        mode = row.get("mode")
        if not isinstance(mode, str) or not mode:
            errors.append("mode row missing mode string")
        for key in ("avg_posenet_dist", "avg_segnet_dist", "score_proxy", "delta_vs_none"):
            if not _finite_number(row.get(key)):
                errors.append(f"mode {mode!r} has non-finite {key}")
        if row.get("n_pairs") != expected_n_pairs:
            errors.append(f"mode {mode!r} n_pairs mismatch")
        if include_per_pair:
            for key in ("pair_posenet_dist", "pair_segnet_dist"):
                values = row.get(key)
                if not isinstance(values, list) or len(values) != expected_n_pairs:
                    errors.append(f"mode {mode!r} {key} length mismatch")
                    continue
                if any(not _finite_number(value) for value in values):
                    errors.append(f"mode {mode!r} {key} contains non-finite values")
    best = payload.get("best")
    if not isinstance(best, dict) or best.get("mode") not in expected_modes:
        errors.append("best must be a mode row with a requested mode")
    return errors


def _finalize_sweep_payload(
    payload: dict[str, Any],
    *,
    archive_sha256: str,
    archive_size_bytes: int,
    source_repo_commit: str,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    finalized = dict(payload)
    finalized.update(
        {
            "axis": AXIS,
            "score_claim": False,
            "promotion_eligible": False,
            "archive_sha256": archive_sha256,
            "archive_bytes": archive_size_bytes,
            "source_repo_commit": source_repo_commit,
            "tool": "experiments/modal_hdm8_postfilter_sweep.py",
            "screen_tool": "tools/screen_hdm8_postfilter_sweep.py",
            "app": APP_NAME,
            "gpu": "T4",
            "hardware_substrate": "linux_x86_64_t4",
            "scorer_device": "cuda",
            "canonical_path": "archive.zip -> tools/screen_hdm8_postfilter_sweep.py --device cuda",
            "modal_cuda_preflight": preflight,
        }
    )
    for row in finalized.get("modes", []):
        if isinstance(row, dict):
            row["score_claim"] = False
            row["promotion_eligible"] = False
    if isinstance(finalized.get("best"), dict):
        finalized["best"]["score_claim"] = False
        finalized["best"]["promotion_eligible"] = False
    return finalized


def _collect_artifacts(out_dir: Path) -> dict[str, bytes]:
    artifacts: dict[str, bytes] = {}
    for path in (
        out_dir / "modal_hdm8_postfilter_sweep_preflight.json",
        out_dir / "modal_hdm8_postfilter_sweep_validation.json",
        out_dir / "screen.stdout.log",
        out_dir / "screen.stderr.log",
        out_dir / SWEEP_JSON_NAME,
    ):
        if path.is_file():
            artifacts[path.name] = path.read_bytes()
    return artifacts


def _run_sweep_inner(
    *,
    archive_bytes: bytes,
    archive_sha256: str,
    archive_size_bytes: int,
    source_repo_commit: str,
    n_pairs: int,
    modes: tuple[str, ...],
    include_per_pair: bool,
    decode_batch_pairs: int,
    score_batch_pairs: int,
    mode_batch_size: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    import os
    import shutil

    out_dir = REMOTE_OUT
    work_dir = REMOTE_WORK_ROOT
    if out_dir.exists():
        shutil.rmtree(out_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    preflight = _probe_cuda_environment()
    preflight.update(
        {
            "tool": "experiments/modal_hdm8_postfilter_sweep.py",
            "axis": AXIS,
            "archive_sha256": archive_sha256,
            "archive_size_bytes": archive_size_bytes,
            "n_pairs_requested": n_pairs,
            "modes": list(modes),
            "include_per_pair": include_per_pair,
            "source_repo_commit": source_repo_commit,
        }
    )
    write_json(out_dir / "modal_hdm8_postfilter_sweep_preflight.json", preflight)
    if preflight.get("torch_cuda_available") is not True:
        validation = {
            "schema_version": 1,
            "passed": False,
            "returncode": 3,
            "error": "Modal runtime has no CUDA device; refusing CPU fallback",
            "score_claim": False,
            "promotion_eligible": False,
            "axis": AXIS,
        }
        write_json(out_dir / "modal_hdm8_postfilter_sweep_validation.json", validation)
        return {**validation, "artifacts": _collect_artifacts(out_dir)}
    if preflight.get("nvidia_dali_import_ok") is not True:
        validation = {
            "schema_version": 1,
            "passed": False,
            "returncode": 4,
            "error": "nvidia.dali import failed in Modal auth-eval image posture",
            "score_claim": False,
            "promotion_eligible": False,
            "axis": AXIS,
        }
        write_json(out_dir / "modal_hdm8_postfilter_sweep_validation.json", validation)
        return {**validation, "artifacts": _collect_artifacts(out_dir)}

    archive_path = work_dir / "archive.zip"
    archive_path.write_bytes(archive_bytes)
    observed_sha = sha256_bytes(archive_path.read_bytes())
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
            "axis": AXIS,
        }
        write_json(out_dir / "modal_hdm8_postfilter_sweep_validation.json", validation)
        return {**validation, "artifacts": _collect_artifacts(out_dir)}

    screen_json = out_dir / SWEEP_JSON_NAME
    cmd = [
        sys.executable,
        "-u",
        str(REMOTE_REPO / "tools/screen_hdm8_postfilter_sweep.py"),
        "--archive",
        str(archive_path),
        "--upstream-dir",
        str(REMOTE_REPO / "upstream"),
        "--output-json",
        str(screen_json),
        "--device",
        "cuda",
        "--n-pairs",
        str(int(n_pairs)),
        "--decode-batch-pairs",
        str(int(decode_batch_pairs)),
        "--score-batch-pairs",
        str(int(score_batch_pairs)),
        "--mode-batch-size",
        str(int(mode_batch_size)),
    ]
    if include_per_pair:
        cmd.append("--include-per-pair")
    for mode in modes:
        cmd.extend(["--mode", mode])

    env = {
        **os.environ,
        "PYTHONPATH": REMOTE_PYTHONPATH,
        "DALI_DISABLE_NVML": DALI_DISABLE_NVML_VALUE,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
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
            timeout=int(timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        (out_dir / "screen.stdout.log").write_text(stdout, encoding="utf-8")
        (out_dir / "screen.stderr.log").write_text(stderr, encoding="utf-8")
        validation = {
            "schema_version": 1,
            "passed": False,
            "returncode": 124,
            "modal_elapsed_seconds": time.monotonic() - started,
            "command": cmd,
            "error": "HDM8 postfilter sweep timed out",
            "score_claim": False,
            "promotion_eligible": False,
            "axis": AXIS,
        }
        write_json(out_dir / "modal_hdm8_postfilter_sweep_validation.json", validation)
        return {**validation, "artifacts": _collect_artifacts(out_dir)}

    (out_dir / "screen.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "screen.stderr.log").write_text(proc.stderr, encoding="utf-8")
    validation_errors: list[str] = []
    payload: dict[str, Any] | None = None
    if proc.returncode != 0:
        validation_errors.append(f"screen_hdm8_postfilter_sweep exited rc={proc.returncode}")
    if not screen_json.is_file():
        validation_errors.append("screen_hdm8_postfilter_sweep did not produce JSON")
    else:
        try:
            raw = read_json(screen_json)
            if not isinstance(raw, dict):
                validation_errors.append("sweep JSON must be an object")
            else:
                payload = _finalize_sweep_payload(
                    raw,
                    archive_sha256=archive_sha256,
                    archive_size_bytes=archive_size_bytes,
                    source_repo_commit=source_repo_commit,
                    preflight=preflight,
                )
                write_json(screen_json, payload)
        except Exception as exc:
            validation_errors.append(f"sweep JSON malformed: {type(exc).__name__}: {exc}")
    if payload is not None:
        validation_errors.extend(
            _validate_sweep_payload(
                payload,
                expected_modes=list(modes),
                expected_n_pairs=int(n_pairs),
                include_per_pair=include_per_pair,
                expected_archive_sha256=archive_sha256,
                expected_archive_size_bytes=archive_size_bytes,
            )
        )

    passed = proc.returncode == 0 and payload is not None and not validation_errors
    validation = {
        "schema_version": 1,
        "passed": passed,
        "returncode": proc.returncode if passed else (proc.returncode or 10),
        "modal_elapsed_seconds": time.monotonic() - started,
        "command": cmd,
        "validation_errors": validation_errors,
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "allowed_use": (
            ["cuda_proxy_selector_learning", "no_score_claim", "no_promotion"]
            if passed
            else ["debug", "no_score_claim", "no_promotion"]
        ),
    }
    write_json(out_dir / "modal_hdm8_postfilter_sweep_validation.json", validation)
    summary = {
        **validation,
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size_bytes,
        "n_pairs": payload.get("n_pairs") if payload else n_pairs,
        "modes": payload.get("modes") if payload else [],
        "best": payload.get("best") if payload else None,
        "artifacts": _collect_artifacts(out_dir),
    }
    return summary


def _run_sweep_fail_closed(**kwargs: Any) -> dict[str, Any]:
    try:
        return _run_sweep_inner(**kwargs)
    except Exception as exc:  # pragma: no cover - remote diagnostic path
        out_dir = REMOTE_OUT
        out_dir.mkdir(parents=True, exist_ok=True)
        validation = {
            "schema_version": 1,
            "passed": False,
            "returncode": 98,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "score_claim": False,
            "promotion_eligible": False,
            "axis": AXIS,
        }
        write_json(out_dir / "modal_hdm8_postfilter_sweep_validation.json", validation)
        return {**validation, "artifacts": _collect_artifacts(out_dir)}


@app.function(image=sweep_image, gpu="T4", timeout=7200)
def run_hdm8_postfilter_sweep_t4(
    archive_bytes: bytes,
    archive_sha256: str,
    archive_size_bytes: int,
    source_repo_commit: str,
    n_pairs: int,
    modes: tuple[str, ...],
    include_per_pair: bool,
    decode_batch_pairs: int,
    score_batch_pairs: int,
    mode_batch_size: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    return _run_sweep_fail_closed(
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size_bytes,
        source_repo_commit=source_repo_commit,
        n_pairs=n_pairs,
        modes=modes,
        include_per_pair=include_per_pair,
        decode_batch_pairs=decode_batch_pairs,
        score_batch_pairs=score_batch_pairs,
        mode_batch_size=mode_batch_size,
        timeout_seconds=timeout_seconds,
    )


def _write_spawn_metadata(
    *,
    out_dir: Path,
    call_id: str,
    local_request: dict[str, Any],
    lane_id: str,
    instance_job_id: str,
    claim_agent: str,
) -> Path:
    payload = {
        "schema_version": "modal_hdm8_postfilter_sweep_spawn_v1",
        "tool": "experiments/modal_hdm8_postfilter_sweep.py",
        "app": APP_NAME,
        "axis": AXIS,
        "call_id": call_id,
        "dispatched_at_utc": utc_now(),
        "result_json_name": RESULT_JSON_NAME,
        "score_claim": False,
        "promotion_eligible": False,
        "lane_id": lane_id,
        "instance_job_id": instance_job_id,
        "claim_agent": claim_agent,
        "claim_platform": "modal",
        "recover_command": (
            ".venv/bin/python experiments/modal_hdm8_postfilter_sweep.py recover "
            f"--output-dir {out_dir}"
        ),
        "local_request": {
            key: value
            for key, value in local_request.items()
            if key != "archive_bytes_payload"
        },
    }
    write_json(out_dir / SPAWN_JSON_NAME, payload)
    (out_dir / "modal_call_id.txt").write_text(call_id + "\n", encoding="utf-8")
    return out_dir / SPAWN_JSON_NAME


def _claim_spec(
    *,
    lane_id: str,
    instance_job_id: str,
    claim_agent: str,
    claim_notes: str,
    force_claim: bool,
) -> DispatchClaimSpec:
    return DispatchClaimSpec(
        lane_id=lane_id,
        instance_job_id=instance_job_id,
        agent=claim_agent,
        platform="modal",
        predicted_eta_utc=predicted_eta(3.0),
        force=force_claim,
        notes=claim_notes,
    )


@app.local_entrypoint()
def main(
    archive: str,
    output_dir: str = "",
    n_pairs: int = 24,
    modes: str = "",
    modes_from_json: str = "",
    include_per_pair: bool = False,
    decode_batch_pairs: int = 4,
    score_batch_pairs: int = 2,
    mode_batch_size: int = 4,
    timeout_seconds: int = 5400,
    detach: bool = False,
    provider_detach_ack: bool = False,
    lane_id: str = "",
    instance_job_id: str = "",
    claim_agent: str = "codex:modal_hdm8_postfilter_sweep",
    claim_notes: str = "",
    force_claim: bool = False,
    dry_run: bool = False,
) -> None:
    if detach and not provider_detach_ack:
        raise SystemExit(
            "FATAL: wrapper --detach requires provider-level Modal CLI detach. "
            "Use `.venv/bin/modal run --detach experiments/modal_hdm8_postfilter_sweep.py "
            "... --detach --provider-detach-ack ...`."
        )
    parsed_modes = parse_modes(modes=modes, modes_from_json=modes_from_json)
    request = _prepare_local_request(
        archive=archive,
        output_dir=output_dir,
        n_pairs=n_pairs,
        modes=parsed_modes,
        include_per_pair=include_per_pair,
        decode_batch_pairs=decode_batch_pairs,
        score_batch_pairs=score_batch_pairs,
        mode_batch_size=mode_batch_size,
        timeout_seconds=timeout_seconds,
    )
    out_dir = Path(str(request["output_dir"]))
    local_request = {
        key: value for key, value in request.items() if key != "archive_bytes_payload"
    }
    local_request.update(
        {
            "modal_dispatch_mode": (
                "dry_run" if dry_run else "detached_spawn" if detach else "blocking_remote"
            ),
            "dispatch_attempted": not dry_run,
            "score_claim": False,
            "promotion_eligible": False,
        }
    )
    local_request["source_transparency"] = collect_source_transparency(
        repo_root=REPO_ROOT,
        source_paths=[
            Path(__file__),
            REPO_ROOT / "tools/screen_hdm8_postfilter_sweep.py",
            Path(str(archive)).resolve(),
            *([Path(modes_from_json).resolve()] if modes_from_json else []),
        ],
        artifact_paths=[out_dir / "modal_hdm8_postfilter_sweep_local_request.json"],
        commands=[
            [
                ".venv/bin/modal",
                "run",
                "--detach",
                "experiments/modal_hdm8_postfilter_sweep.py",
                "--archive",
                str(archive),
                "--output-dir",
                str(output_dir),
                "--n-pairs",
                str(n_pairs),
                "--modes",
                modes,
            ]
        ],
    )
    write_json(out_dir / "modal_hdm8_postfilter_sweep_local_request.json", local_request)
    if dry_run:
        print(json.dumps({**local_request, "dispatch_attempted": False}, indent=2, sort_keys=True))
        return
    if not lane_id or not instance_job_id:
        raise SystemExit(
            "FATAL: Modal HDM8 postfilter sweep requires --lane-id and "
            "--instance-job-id before provider GPU spend"
        )

    spec = _claim_spec(
        lane_id=lane_id,
        instance_job_id=instance_job_id,
        claim_agent=claim_agent,
        claim_notes=(
            claim_notes
            or (
                "Modal T4 HDM8 postfilter CUDA proxy-prefix sweep; "
                f"axis={AXIS}; archive_sha256={request['archive_sha256']}; "
                f"n_pairs={request['n_pairs']}; modes={len(parsed_modes)}; "
                "score_claim=false"
            )
        ),
        force_claim=force_claim,
    )
    call_args = (
        request["archive_bytes_payload"],
        request["archive_sha256"],
        request["archive_size_bytes"],
        request["source_repo_commit"],
        request["n_pairs"],
        tuple(parsed_modes),
        bool(include_per_pair),
        request["decode_batch_pairs"],
        request["score_batch_pairs"],
        request["mode_batch_size"],
        request["timeout_seconds"],
    )
    record_dispatch_claim(
        repo_root=Path.cwd(),
        spec=spec,
        status=(
            "active_modal_hdm8_postfilter_sweep_spawning"
            if detach
            else "active_modal_hdm8_postfilter_sweep_running"
        ),
    )
    if detach:
        try:
            call = run_hdm8_postfilter_sweep_t4.spawn(*call_args)
        except Exception:
            record_dispatch_claim(
                repo_root=Path.cwd(),
                spec=DispatchClaimSpec(
                    lane_id=lane_id,
                    instance_job_id=instance_job_id,
                    agent=claim_agent,
                    platform="modal",
                    predicted_eta_utc=predicted_eta(3.0),
                    force=True,
                    notes=(
                        "Modal HDM8 postfilter sweep spawn raised after dispatch "
                        "boundary; manual reconciliation required"
                    ),
                ),
                status="ambiguous_modal_hdm8_postfilter_sweep_spawn_submission_recovery_required",
            )
            raise
        call_id = function_call_id(call)
        _write_spawn_metadata(
            out_dir=out_dir,
            call_id=call_id,
            local_request=request,
            lane_id=lane_id,
            instance_job_id=instance_job_id,
            claim_agent=claim_agent,
        )
        record_dispatch_claim(
            repo_root=Path.cwd(),
            spec=DispatchClaimSpec(
                lane_id=lane_id,
                instance_job_id=instance_job_id,
                agent=claim_agent,
                platform="modal",
                predicted_eta_utc=predicted_eta(3.0),
                force=True,
                notes=(
                    "Modal HDM8 postfilter sweep detached spawn accepted; "
                    f"call_id={call_id}; output_dir={out_dir}"
                ),
            ),
            status="active_modal_hdm8_postfilter_sweep_spawned",
        )
        print(f"MODAL HDM8 POSTFILTER SWEEP DISPATCHED DETACHED call_id={call_id}")
        print(f"artifacts: {out_dir}")
        print(
            "recover: .venv/bin/python experiments/modal_hdm8_postfilter_sweep.py "
            f"recover --output-dir {out_dir}"
        )
        return

    try:
        result = run_hdm8_postfilter_sweep_t4.remote(*call_args)
    except Exception:
        terminal_dispatch_claim(
            repo_root=Path.cwd(),
            spec=spec,
            status="failed_modal_hdm8_postfilter_sweep_exception",
            notes="Modal HDM8 postfilter sweep raised provider exception; no score claim",
        )
        raise
    _materialize_result(out_dir=out_dir, result=result)
    status = (
        "completed_modal_hdm8_postfilter_sweep_no_score_claim"
        if result.get("passed")
        else "failed_modal_hdm8_postfilter_sweep_no_score_claim"
    )
    terminal_dispatch_claim(
        repo_root=Path.cwd(),
        spec=spec,
        status=status,
        notes=(
            f"Modal HDM8 postfilter sweep finished; passed={result.get('passed')}; "
            f"axis={AXIS}; result_json={out_dir / RESULT_JSON_NAME}; "
            f"archive_sha256={request['archive_sha256']}; score_claim=false"
        ),
    )
    if not result.get("passed"):
        raise SystemExit(int(result.get("returncode") or 1))


def _materialize_result(*, out_dir: Path, result: dict[str, Any]) -> None:
    artifacts = result.get("artifacts")
    if isinstance(artifacts, dict):
        try:
            materialize_modal_artifacts(out_dir=out_dir, artifacts=artifacts)
        except ModalArtifactWriteError as exc:
            failure = {
                "schema_version": "modal_hdm8_postfilter_sweep_result_v1",
                "status": "invalid_artifacts",
                "artifact_write_errors": exc.errors,
                "score_claim": False,
                "promotion_eligible": False,
            }
            write_json(out_dir / RESULT_JSON_NAME, failure)
            raise SystemExit(5) from exc
    persisted = {key: value for key, value in result.items() if key != "artifacts"}
    persisted["local_output_dir"] = str(out_dir)
    persisted["score_claim"] = False
    persisted["promotion_eligible"] = False
    write_json(out_dir / RESULT_JSON_NAME, persisted)
    print(json.dumps(persisted, indent=2, sort_keys=True))


def _function_call_from_id(call_id: str) -> Any:
    functions = getattr(modal, "functions", None)
    function_call = getattr(functions, "FunctionCall", None) if functions else None
    if function_call is None:
        function_call = getattr(modal, "FunctionCall", None)
    if function_call is None:
        raise RuntimeError("Modal SDK has no FunctionCall interface")
    return function_call.from_id(call_id)


def _claim_note_fragment(value: Any, *, max_len: int = 240) -> str:
    """Return text safe for the markdown dispatch-claim table cell."""

    text = str(value)
    safe = "".join(" " if char == "|" or ord(char) < 0x20 else char for char in text)
    return " ".join(safe.split())[:max_len]


def _recovery_identity_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    request = metadata.get("local_request") if isinstance(metadata, dict) else None
    if not isinstance(request, dict):
        request = {}
    fields: dict[str, Any] = {
        "lane_id": metadata.get("lane_id"),
        "instance_job_id": metadata.get("instance_job_id"),
        "claim_agent": metadata.get("claim_agent"),
    }
    if request.get("archive_sha256") is not None:
        fields["archive_sha256"] = request.get("archive_sha256")
    if request.get("archive_size_bytes") is not None:
        fields["archive_size_bytes"] = request.get("archive_size_bytes")
    return fields


def _close_recovery_claim(
    *,
    metadata: dict[str, Any],
    status: str,
    notes: str,
    no_close_claim: bool,
) -> dict[str, Any]:
    """Best-effort terminal claim close with summary fields for custody."""

    lane_id = str(metadata.get("lane_id") or "")
    instance_job_id = str(metadata.get("instance_job_id") or "")
    claim_agent = str(
        metadata.get("claim_agent") or "codex:modal_hdm8_postfilter_sweep_recover"
    )
    if no_close_claim:
        return {
            "terminal_claim_closed": False,
            "terminal_claim_status": "skipped_no_close_claim",
            "terminal_claim_error": None,
        }
    if not lane_id or not instance_job_id:
        return {
            "terminal_claim_closed": False,
            "terminal_claim_status": "missing_claim_identity",
            "terminal_claim_error": "lane_id or instance_job_id missing from spawn metadata",
        }
    try:
        terminal_dispatch_claim(
            repo_root=Path.cwd(),
            spec=DispatchClaimSpec(
                lane_id=lane_id,
                instance_job_id=instance_job_id,
                agent=claim_agent,
                platform="modal",
                force=True,
            ),
            status=status,
            notes=_claim_note_fragment(notes, max_len=500),
        )
    except (Exception, SystemExit) as exc:
        return {
            "terminal_claim_closed": False,
            "terminal_claim_status": "terminal_claim_close_failed",
            "terminal_claim_error": _claim_note_fragment(repr(exc), max_len=500),
        }
    return {
        "terminal_claim_closed": True,
        "terminal_claim_status": status,
        "terminal_claim_error": None,
    }


def recover_detached(
    *,
    output_dir: Path,
    call_id: str = "",
    timeout_s: float = 0.0,
    no_close_claim: bool = False,
) -> dict[str, Any]:
    out_dir = output_dir.resolve()
    metadata_path = out_dir / SPAWN_JSON_NAME
    if not metadata_path.is_file():
        request_path = out_dir / "modal_hdm8_postfilter_sweep_local_request.json"
        if request_path.is_file():
            request = read_json(request_path)
            if isinstance(request, dict) and (
                request.get("dispatch_attempted") is False
                or request.get("modal_dispatch_mode") == "dry_run"
            ):
                summary = {
                    "schema_version": "modal_hdm8_postfilter_sweep_recover_summary_v1",
                    "status": "dry_run_no_remote_call",
                    "output_dir": str(out_dir),
                    "request_json": str(request_path),
                    "score_claim": False,
                    "promotion_eligible": False,
                    "recovered_at_utc": utc_now(),
                }
                write_json(out_dir / RECOVER_JSON_NAME, summary)
                return summary
        raise FileNotFoundError(metadata_path)
    metadata = read_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path} must contain a JSON object")
    resolved_call_id = call_id or str(metadata.get("call_id") or "").strip()
    if not resolved_call_id:
        raise ValueError(f"{metadata_path} has no call_id")
    try:
        result = _function_call_from_id(resolved_call_id).get(timeout=float(timeout_s))
    except TimeoutError:
        summary = {
            "schema_version": "modal_hdm8_postfilter_sweep_recover_summary_v1",
            "status": "pending",
            "call_id": resolved_call_id,
            "output_dir": str(out_dir),
            "score_claim": False,
            "promotion_eligible": False,
            "recovered_at_utc": utc_now(),
        }
        write_json(out_dir / RECOVER_JSON_NAME, summary)
        return summary
    except Exception as exc:
        message = str(exc)
        is_cancelled = "cancel" in message.lower()
        summary = {
            "schema_version": "modal_hdm8_postfilter_sweep_recover_summary_v1",
            "status": (
                "cancelled_provider_function_call"
                if is_cancelled
                else "failed_provider_exception"
            ),
            "call_id": resolved_call_id,
            "output_dir": str(out_dir),
            "provider_exception_type": type(exc).__name__,
            "provider_exception_message": message,
            "passed": False,
            "returncode": 1,
            "axis": AXIS,
            "score_claim": False,
            "promotion_eligible": False,
            "recovered_at_utc": utc_now(),
        }
        summary.update(_recovery_identity_fields(metadata))
        terminal_status = (
            "cancelled_modal_hdm8_postfilter_sweep_no_score_claim"
            if is_cancelled
            else "failed_modal_hdm8_postfilter_sweep_provider_exception_no_score_claim"
        )
        summary.update(
            _close_recovery_claim(
                metadata=metadata,
                status=terminal_status,
                notes=(
                    "Modal HDM8 postfilter sweep recovery observed provider "
                    f"exception type={type(exc).__name__}; "
                    f"message={message}; axis={AXIS}; score_claim=false"
                ),
                no_close_claim=no_close_claim,
            )
        )
        if (
            not no_close_claim
            and summary.get("terminal_claim_closed") is not True
        ):
            summary["returncode"] = 6
        write_json(out_dir / RECOVER_JSON_NAME, summary)
        return summary
    if not isinstance(result, dict):
        summary = {
            "schema_version": "modal_hdm8_postfilter_sweep_recover_summary_v1",
            "status": "invalid_result",
            "call_id": resolved_call_id,
            "output_dir": str(out_dir),
            "error": f"Modal result must be a dict, got {type(result).__name__}",
            "passed": False,
            "returncode": 5,
            "axis": AXIS,
            "score_claim": False,
            "promotion_eligible": False,
            "recovered_at_utc": utc_now(),
        }
        summary.update(_recovery_identity_fields(metadata))
        summary.update(
            _close_recovery_claim(
                metadata=metadata,
                status="failed_modal_hdm8_postfilter_sweep_invalid_result_no_score_claim",
                notes=(
                    "Modal HDM8 postfilter sweep recovery returned invalid result "
                    f"type={type(result).__name__}; axis={AXIS}; score_claim=false"
                ),
                no_close_claim=no_close_claim,
            )
        )
        if (
            not no_close_claim
            and summary.get("terminal_claim_closed") is not True
        ):
            summary["returncode"] = 6
        write_json(out_dir / RECOVER_JSON_NAME, summary)
        return summary
    _materialize_result(out_dir=out_dir, result=result)
    summary = {
        "schema_version": "modal_hdm8_postfilter_sweep_recover_summary_v1",
        "status": "recovered",
        "call_id": resolved_call_id,
        "output_dir": str(out_dir),
        "result_json": str(out_dir / RESULT_JSON_NAME),
        "passed": bool(result.get("passed")),
        "returncode": result.get("returncode"),
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "n_pairs": result.get("n_pairs"),
        "best": result.get("best"),
        "recovered_at_utc": utc_now(),
    }
    summary.update(_recovery_identity_fields(metadata))
    terminal_status = (
        "completed_modal_hdm8_postfilter_sweep_no_score_claim"
        if result.get("passed")
        else "failed_modal_hdm8_postfilter_sweep_no_score_claim"
    )
    summary.update(
        _close_recovery_claim(
            metadata=metadata,
            status=terminal_status,
            notes=(
                f"Modal HDM8 postfilter sweep recovered; passed={result.get('passed')}; "
                f"axis={AXIS}; result_json={out_dir / RESULT_JSON_NAME}; "
                "score_claim=false"
            ),
            no_close_claim=no_close_claim,
        )
    )
    if not no_close_claim and summary.get("terminal_claim_closed") is not True:
        summary["passed"] = False
        summary["returncode"] = 6
    write_json(out_dir / RECOVER_JSON_NAME, summary)
    return summary


def _recover_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Recover a detached Modal HDM8 postfilter sweep.")
    parser.add_argument("command", choices=["recover"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--call-id", default="")
    parser.add_argument("--timeout-s", type=float, default=0.0)
    parser.add_argument("--no-close-claim", action="store_true")
    args = parser.parse_args(argv)
    summary = recover_detached(
        output_dir=args.output_dir,
        call_id=args.call_id,
        timeout_s=args.timeout_s,
        no_close_claim=args.no_close_claim,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary.get("status") == "pending":
        return 4
    if summary.get("passed") is False:
        return int(summary.get("returncode") or 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(_recover_cli(sys.argv[1:]))
