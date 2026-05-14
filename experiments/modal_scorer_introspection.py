# SPDX-License-Identifier: MIT
"""Run scorer-activation introspection on Modal Linux x86_64 GPU (T4).

This wrapper carries the P5 CPU/CUDA xray diagnostic over to a GPU host so the
local CPU per-layer record can be paired against a Linux x86_64 CUDA record
under CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-
COMPLIANT HARDWARE".

The wrapper is DIAGNOSTIC ONLY. It runs ``experiments/dump_scorer_activations.py
--device cuda`` against the SAME shared RGB input tensor used by the local CPU
record, harvests ``{posenet,segnet}_record.pt`` and ``summary.json``, and writes
a Modal-side summary. It NEVER produces a score; it NEVER promotes; it NEVER
runs anything against an archive's inflate path.

Per CLAUDE.md non-negotiables this wrapper:
- Fails closed on CPU fallback (never accepts a CUDA->CPU silent fallback).
- Tags every output ``[diagnostic-not-score]`` with score_claim=False,
  promotion_eligible=False, rank_or_kill_eligible=False.
- Verifies the shared-input-tensor SHA-256 in custody.
- Never modifies upstream/.

Usage::

    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \\
        experiments/modal_scorer_introspection.py \\
        --shared-input-tensor experiments/results/.../shared_input.pt \\
        --output-dir experiments/results/cpu_cuda_xray_p5_cuda_capture_<UTC> \\
        --scorer both \\
        --detach --provider-detach-ack \\
        --lane-id lane_cpu_cuda_xray_p5_landing_segnet_cuda_capture \\
        --instance-job-id <job>
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.deploy.claims import (  # noqa: E402
    DispatchClaimSpec,
    predicted_eta,
    record_dispatch_claim,
    terminal_dispatch_claim,
)
from tac.deploy.modal.auth_eval import (  # noqa: E402
    function_call_id,
    safe_artifact_label,
)
from tac.repo_io import write_json  # noqa: E402

APP_NAME = "comma-scorer-introspection"
REMOTE_REPO = Path("/workspace/pact")
REMOTE_OUT = Path("/tmp/modal_scorer_introspection")
REMOTE_PYTHONPATH = f"{REMOTE_REPO / 'src'}:{REMOTE_REPO / 'upstream'}:{REMOTE_REPO}"

SHARED_INPUT_TENSOR_SCHEMA = "eval_loader_shared_input_tensor.v1"
NON_PROMOTABLE_FIELDS = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}

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
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "safetensors",
        "einops",
        "segmentation-models-pytorch",
        "av",
        "click",
        "tqdm",
        "timm",
        "scipy",
        "numpy<2.0",
        "Pillow",
    )
)

introspection_image = (
    base_image
    .env({"PYTHONPATH": REMOTE_PYTHONPATH})
    .add_local_dir("src", remote_path=str(REMOTE_REPO / "src"))  # MODAL_MANUAL_MOUNT_OK:narrow scorer-introspection dispatcher; targeted upstream files; trainer-discovery N/A
    .add_local_dir("upstream", remote_path=str(REMOTE_REPO / "upstream"))  # MODAL_MANUAL_MOUNT_OK:narrow scorer-introspection dispatcher; targeted upstream files; trainer-discovery N/A
    .add_local_file(  # MODAL_MANUAL_MOUNT_OK:narrow scorer-introspection dispatcher; targeted upstream files; trainer-discovery N/A
        "experiments/dump_scorer_activations.py",
        remote_path=str(REMOTE_REPO / "experiments/dump_scorer_activations.py"),
    )
)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _verify_shared_input_tensor_locally(path: Path) -> dict[str, Any]:
    """Verify the shared input tensor schema + custody before upload."""

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise SystemExit(f"FATAL: shared input tensor at {path} is not a dict")
    schema = payload.get("schema")
    if schema != SHARED_INPUT_TENSOR_SCHEMA:
        raise SystemExit(
            f"FATAL: shared input tensor schema={schema!r}, expected "
            f"{SHARED_INPUT_TENSOR_SCHEMA!r}"
        )
    for key, expected in NON_PROMOTABLE_FIELDS.items():
        if payload.get(key) is not expected:
            raise SystemExit(
                f"FATAL: shared input tensor {key}={payload.get(key)!r}, expected "
                f"{expected!r}"
            )
    tensor = payload.get("tensor")
    if not hasattr(tensor, "shape"):
        raise SystemExit("FATAL: shared input tensor missing tensor field")
    return {
        "tensor_sha256": _file_sha256(path),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
    }


@dataclass(frozen=True)
class IntrospectionRequest:
    """Local-side prepared request for the remote CUDA dump."""

    shared_input_tensor_path: Path
    shared_input_tensor_sha256: str
    shared_input_tensor_shape: list[int]
    shared_input_tensor_dtype: str
    scorer: str
    output_dir: Path
    capture_mode: str
    full_threshold_elements: int


def _prepare_request(
    *,
    shared_input_tensor: str,
    output_dir: str,
    scorer: str,
    capture_mode: str,
    full_threshold_elements: int,
    default_output_root: Path,
) -> IntrospectionRequest:
    path = Path(shared_input_tensor).resolve()
    if not path.is_file():
        raise SystemExit(f"FATAL: shared input tensor not found: {path}")
    if scorer not in {"both", "posenet", "segnet"}:
        raise SystemExit("FATAL: --scorer must be one of both, posenet, segnet")
    if capture_mode not in {"fingerprint", "full"}:
        raise SystemExit("FATAL: --capture-mode must be fingerprint or full")

    verified = _verify_shared_input_tensor_locally(path)

    if output_dir:
        out_path = Path(output_dir).resolve()
    else:
        label = safe_artifact_label(f"scorer_introspection_cuda_{int(time.time())}")
        out_path = (default_output_root / label).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    return IntrospectionRequest(
        shared_input_tensor_path=path,
        shared_input_tensor_sha256=verified["tensor_sha256"],
        shared_input_tensor_shape=verified["tensor_shape"],
        shared_input_tensor_dtype=verified["tensor_dtype"],
        scorer=scorer,
        output_dir=out_path,
        capture_mode=capture_mode,
        full_threshold_elements=full_threshold_elements,
    )


@app.function(
    image=introspection_image,
    cpu=4.0,
    memory=8192,
    timeout=1500,
)
def run_scorer_introspection_cpu(
    shared_input_tensor_bytes: bytes,
    shared_input_tensor_sha256: str,
    scorer: str,
    capture_mode: str,
    full_threshold_elements: int,
) -> dict[str, Any]:
    """Remote (Linux x86_64 CPU): run dump_scorer_activations.py on CPU."""

    actual_sha = hashlib.sha256(shared_input_tensor_bytes).hexdigest()
    if actual_sha != shared_input_tensor_sha256:
        return {
            "ok": False,
            "reason": f"shared input tensor SHA mismatch: {actual_sha} != "
                      f"{shared_input_tensor_sha256}",
            **NON_PROMOTABLE_FIELDS,
            "tag": "[diagnostic-not-score]",
        }

    import platform as _platform
    try:
        import torch
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"torch import failed: {exc!r}",
            **NON_PROMOTABLE_FIELDS,
            "tag": "[diagnostic-not-score]",
        }

    REMOTE_OUT.mkdir(parents=True, exist_ok=True)
    shared_input_path = REMOTE_OUT / "shared_input.pt"
    shared_input_path.write_bytes(shared_input_tensor_bytes)

    remote_output_dir = REMOTE_OUT / "output"
    remote_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(REMOTE_REPO / "experiments/dump_scorer_activations.py"),
        "--upstream-dir",
        str(REMOTE_REPO / "upstream"),
        "--device",
        "cpu",
        "--shared-input-tensor",
        str(shared_input_path),
        "--output-dir",
        str(remote_output_dir),
        "--capture-mode",
        capture_mode,
        "--full-threshold-elements",
        str(full_threshold_elements),
        "--scorer",
        scorer,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = REMOTE_PYTHONPATH
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REMOTE_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    elapsed = time.time() - started

    artifacts: dict[str, bytes] = {}
    for name in ("posenet_record.pt", "segnet_record.pt", "summary.json"):
        p = remote_output_dir / name
        if p.is_file():
            artifacts[name] = p.read_bytes()

    return {
        "ok": proc.returncode == 0 and "summary.json" in artifacts,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-32000:],
        "stderr": proc.stderr[-32000:],
        "elapsed_seconds": elapsed,
        "host_system": _platform.system(),
        "host_machine": _platform.machine(),
        "host_platform": _platform.platform(),
        "torch_version": torch.__version__,
        "artifacts": artifacts,
        "tag": "[diagnostic-not-score]",
        "dispatch_attempted": True,
        **NON_PROMOTABLE_FIELDS,
    }


@app.function(
    image=introspection_image,
    gpu="T4",
    timeout=1500,
)
def run_scorer_introspection_cuda(
    shared_input_tensor_bytes: bytes,
    shared_input_tensor_sha256: str,
    scorer: str,
    capture_mode: str,
    full_threshold_elements: int,
) -> dict[str, Any]:
    """Remote: run dump_scorer_activations.py on CUDA + return artifact bytes."""

    actual_sha = hashlib.sha256(shared_input_tensor_bytes).hexdigest()
    if actual_sha != shared_input_tensor_sha256:
        return {
            "ok": False,
            "reason": f"shared input tensor SHA mismatch: {actual_sha} != "
                      f"{shared_input_tensor_sha256}",
            **NON_PROMOTABLE_FIELDS,
            "tag": "[diagnostic-not-score]",
        }

    try:
        import torch
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"torch import failed: {exc!r}",
            **NON_PROMOTABLE_FIELDS,
            "tag": "[diagnostic-not-score]",
        }
    if not torch.cuda.is_available():
        return {
            "ok": False,
            "reason": "torch.cuda.is_available() is False inside remote container",
            **NON_PROMOTABLE_FIELDS,
            "tag": "[diagnostic-not-score]",
        }
    cuda_device_name = torch.cuda.get_device_name(0)
    cuda_capability = list(torch.cuda.get_device_capability(0))

    REMOTE_OUT.mkdir(parents=True, exist_ok=True)
    shared_input_path = REMOTE_OUT / "shared_input.pt"
    shared_input_path.write_bytes(shared_input_tensor_bytes)

    remote_output_dir = REMOTE_OUT / "output"
    remote_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(REMOTE_REPO / "experiments/dump_scorer_activations.py"),
        "--upstream-dir",
        str(REMOTE_REPO / "upstream"),
        "--device",
        "cuda",
        "--shared-input-tensor",
        str(shared_input_path),
        "--output-dir",
        str(remote_output_dir),
        "--capture-mode",
        capture_mode,
        "--full-threshold-elements",
        str(full_threshold_elements),
        "--scorer",
        scorer,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = REMOTE_PYTHONPATH
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REMOTE_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    elapsed = time.time() - started

    artifacts: dict[str, bytes] = {}
    for name in ("posenet_record.pt", "segnet_record.pt", "summary.json"):
        p = remote_output_dir / name
        if p.is_file():
            artifacts[name] = p.read_bytes()

    return {
        "ok": proc.returncode == 0 and "summary.json" in artifacts,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-32000:],
        "stderr": proc.stderr[-32000:],
        "elapsed_seconds": elapsed,
        "cuda_device_name": cuda_device_name,
        "cuda_capability": cuda_capability,
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "artifacts": artifacts,
        "tag": "[diagnostic-not-score]",
        "dispatch_attempted": True,
        **NON_PROMOTABLE_FIELDS,
    }


@app.local_entrypoint()
def main(
    shared_input_tensor: str = "",
    output_dir: str = "",
    scorer: str = "both",
    capture_mode: str = "fingerprint",
    full_threshold_elements: int = 1 << 20,
    remote_device: str = "cuda",
    detach: bool = False,
    provider_detach_ack: bool = False,
    lane_id: str = "",
    instance_job_id: str = "",
    claim_agent: str = "claude:modal_scorer_introspection",
    claim_notes: str = "",
    force_claim: bool = False,
) -> None:
    """Upload shared input tensor + harvest CUDA scorer-activation records."""

    if not shared_input_tensor:
        raise SystemExit("FATAL: --shared-input-tensor is required")
    if remote_device not in {"cuda", "cpu"}:
        raise SystemExit("FATAL: --remote-device must be one of cuda, cpu")
    if detach and not provider_detach_ack:
        raise SystemExit(
            "FATAL: wrapper --detach requires provider-level Modal CLI detach. "
            "Use `.venv/bin/modal run --detach experiments/modal_scorer_introspection.py "
            "... --detach --provider-detach-ack ...`."
        )

    prepared = _prepare_request(
        shared_input_tensor=shared_input_tensor,
        output_dir=output_dir,
        scorer=scorer,
        capture_mode=capture_mode,
        full_threshold_elements=full_threshold_elements,
        default_output_root=Path("experiments/results/modal_scorer_introspection"),
    )

    local_summary: dict[str, Any] = {
        "schema_version": 1,
        "tool": "experiments/modal_scorer_introspection.py",
        "app": APP_NAME,
        "shared_input_tensor_path": str(prepared.shared_input_tensor_path),
        "shared_input_tensor_sha256": prepared.shared_input_tensor_sha256,
        "shared_input_tensor_shape": prepared.shared_input_tensor_shape,
        "shared_input_tensor_dtype": prepared.shared_input_tensor_dtype,
        "scorer": prepared.scorer,
        "capture_mode": prepared.capture_mode,
        "full_threshold_elements": prepared.full_threshold_elements,
        "canonical_path": "experiments/dump_scorer_activations.py --device cuda (Modal T4)",
        "modal_dispatch_mode": "detached_spawn" if detach else "blocking_remote",
        "tag": "[diagnostic-not-score]",
        "dispatch_attempted": True,
        **NON_PROMOTABLE_FIELDS,
    }
    write_json(prepared.output_dir / "modal_scorer_introspection_request.json", local_summary)

    spec = DispatchClaimSpec(
        lane_id=lane_id,
        instance_job_id=instance_job_id,
        agent=claim_agent or "claude:modal_scorer_introspection",
        platform="modal",
        force=force_claim,
        notes=(
            claim_notes
            or (
                f"Modal CUDA scorer-introspection (scorer={scorer}; "
                f"shared_input_sha={prepared.shared_input_tensor_sha256[:12]})"
            )
        ),
    )
    if not spec.lane_id or not spec.instance_job_id:
        raise SystemExit(
            "FATAL: Modal scorer-introspection dispatch requires --lane-id and "
            "--instance-job-id before provider work starts"
        )
    record_dispatch_claim(
        repo_root=REPO_ROOT,
        spec=DispatchClaimSpec(
            lane_id=spec.lane_id,
            instance_job_id=spec.instance_job_id,
            agent=spec.agent,
            platform=spec.platform,
            predicted_eta_utc=predicted_eta(hours=1),
            force=spec.force,
            notes=spec.notes,
        ),
        status="active_modal_scorer_introspection_spawning",
        default_notes="Modal scorer-introspection dispatch; score_claim=false; diagnostic_only",
    )

    shared_input_tensor_bytes = prepared.shared_input_tensor_path.read_bytes()
    print(
        f"Uploading {len(shared_input_tensor_bytes):,} bytes shared input tensor to "
        f"Modal T4 for CUDA scorer introspection "
        f"(sha256={prepared.shared_input_tensor_sha256[:12]}...) ..."
    )

    remote_fn = (
        run_scorer_introspection_cuda
        if remote_device == "cuda"
        else run_scorer_introspection_cpu
    )

    if detach:
        call = remote_fn.spawn(
            shared_input_tensor_bytes,
            prepared.shared_input_tensor_sha256,
            prepared.scorer,
            prepared.capture_mode,
            prepared.full_threshold_elements,
        )
        call_id = function_call_id(call)
        spawn_meta = {
            "schema_version": 1,
            "tool": "experiments/modal_scorer_introspection.py",
            "modal_dispatch_mode": "detached_spawn",
            "call_id": call_id,
            "shared_input_tensor_sha256": prepared.shared_input_tensor_sha256,
            "scorer": prepared.scorer,
            "capture_mode": prepared.capture_mode,
            "tag": "[diagnostic-not-score]",
            **NON_PROMOTABLE_FIELDS,
        }
        write_json(prepared.output_dir / "modal_scorer_introspection_spawn.json", spawn_meta)
        record_dispatch_claim(
            repo_root=REPO_ROOT,
            spec=DispatchClaimSpec(
                lane_id=spec.lane_id,
                instance_job_id=spec.instance_job_id,
                agent=spec.agent,
                platform=spec.platform,
                predicted_eta_utc=predicted_eta(hours=1),
                force=True,
                notes=f"call_id={call_id}; output_dir={prepared.output_dir}",
            ),
            status="active_modal_scorer_introspection_spawned",
        )
        print(
            "============================================================\n"
            f"MODAL SCORER-INTROSPECTION DISPATCHED DETACHED call_id={call_id}\n"
            f"  Artifacts: {prepared.output_dir}\n"
            f"  Recover: .venv/bin/python -c \"import modal; r=modal.functions.FunctionCall.from_id('{call_id}').get(); print(r.get('ok'),'artifacts:',sorted(r.get('artifacts',{{}}).keys()))\"\n"
            "============================================================"
        )
        return

    print(f"Running blocking Modal scorer introspection (remote_device={remote_device}, no --detach)...")
    result = remote_fn.remote(
        shared_input_tensor_bytes,
        prepared.shared_input_tensor_sha256,
        prepared.scorer,
        prepared.capture_mode,
        prepared.full_threshold_elements,
    )
    _persist_result(prepared, result, spec)


def _persist_result(
    prepared: IntrospectionRequest,
    result: dict[str, Any],
    spec: DispatchClaimSpec,
) -> None:
    artifacts = result.pop("artifacts", {})
    for name, blob in artifacts.items():
        (prepared.output_dir / name).write_bytes(blob)
    result_meta = {
        "schema_version": 1,
        "tool": "experiments/modal_scorer_introspection.py",
        "tag": "[diagnostic-not-score]",
        **NON_PROMOTABLE_FIELDS,
        **{k: v for k, v in result.items() if not isinstance(v, (bytes, bytearray))},
        "artifacts_received": sorted(artifacts.keys()),
    }
    write_json(prepared.output_dir / "modal_scorer_introspection_result.json", result_meta)

    terminal_status = (
        "completed_modal_scorer_introspection_recovered"
        if result.get("ok")
        else "failed_modal_scorer_introspection"
    )
    terminal_dispatch_claim(
        repo_root=REPO_ROOT,
        spec=spec,
        status=terminal_status,
        notes=f"output_dir={prepared.output_dir}; artifacts={sorted(artifacts.keys())}",
    )

    print(
        "============================================================\n"
        f"MODAL SCORER-INTROSPECTION RESULT ok={result.get('ok')}\n"
        f"  Artifacts: {prepared.output_dir}\n"
        f"  Elapsed: {result.get('elapsed_seconds', 0):.1f}s\n"
        "============================================================"
    )
