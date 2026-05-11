"""D4 DALI loader-drift capture on Modal Linux x86_64 GPU (T4).

Closes the P5 4-cell device-axis matrix loader-drift cell. Per the
2026-05-11 drift permanent-fix memo
(`feedback_drift_permanent_fix_paired_anchors_landed_20260511.md`):

> A more surgical mechanism-specific fix, a follow-up DALI loader-drift
> capture on Linux x86_64 GPU (D4 in the parallel-Δ scope, operator-gated,
> ~$0.03 Modal) would close the loader-vs-preprocess attribution.

This wrapper dispatches a Modal T4 container with `nvidia-dali-cuda120` in
the image and runs:

    tools/probe_eval_loader_drift.py \
        --run-forward-cells \
        --save-shared-input-dir <output>/shared_inputs \
        --json-out <output>/probe_eval_loader_drift_report.json

DIAGNOSTIC ONLY. Every emitted artifact is tagged ``[diagnostic-not-score]``
and ``score_claim=False``. Per CLAUDE.md "Submission auth eval — BOTH CPU
AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE", this is a mechanism-attribution
diagnostic, never an authoritative score axis.

Usage (always --detach, the bash harness kills BG jobs at 3 min):

    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \\
        experiments/modal_loader_drift_capture.py \\
        --output-dir experiments/results/cpu_cuda_xray_p5_landing_loader_dali_capture_<utc> \\
        --lane-id lane_cpu_cuda_xray_p5_landing_loader_dali_capture \\
        --instance-job-id loader_dali_capture_<utc>

Recover (24h Modal call cache):

    .venv/bin/python tools/recover_modal_loader_drift.py --output-dir <output>
"""

from __future__ import annotations

import hashlib
import os
import platform as _platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tac.deploy.claims import (  # noqa: E402
    DispatchClaimSpec,
    record_dispatch_claim,
    terminal_dispatch_claim,
)
from tac.deploy.modal.auth_eval import (  # noqa: E402
    function_call_id,
    safe_artifact_label,
)
from tac.repo_io import write_json  # noqa: E402

APP_NAME = "comma-loader-drift-capture"
REMOTE_REPO = Path("/workspace/pact")
REMOTE_OUT = REMOTE_REPO / "_modal_loader_drift_capture"
REMOTE_PYTHONPATH = f"{REMOTE_REPO / 'src'}:{REMOTE_REPO / 'upstream'}:{REMOTE_REPO}"

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
    # DALI for CUDA 12.x — closes the D4 loader-drift cell.
    .pip_install(
        "nvidia-dali-cuda120",
        extra_index_url="https://developer.download.nvidia.com/compute/redist",
    )
)

drift_image = (
    base_image
    .env({"PYTHONPATH": REMOTE_PYTHONPATH})
    .add_local_dir("src", remote_path=str(REMOTE_REPO / "src"))
    .add_local_dir("upstream", remote_path=str(REMOTE_REPO / "upstream"))
    .add_local_file(
        "tools/probe_eval_loader_drift.py",
        remote_path=str(REMOTE_REPO / "tools/probe_eval_loader_drift.py"),
    )
)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


@app.function(
    image=drift_image,
    gpu="T4",
    timeout=1500,
)
def run_loader_drift_capture(
    *,
    video_limit: int,
    max_batches: int,
    batch_size: int,
    save_shared_input: bool,
) -> dict[str, Any]:
    """Remote: run probe_eval_loader_drift.py on Linux x86_64 T4 with DALI."""

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

    dali_available = False
    dali_version = None
    try:
        from nvidia.dali import __version__ as _dali_version  # noqa: F401

        dali_available = True
        dali_version = _dali_version
    except Exception as exc:
        dali_available = False
        dali_version = f"import_error: {exc!r}"

    REMOTE_OUT.mkdir(parents=True, exist_ok=True)
    remote_output_dir = REMOTE_OUT / "output"
    remote_output_dir.mkdir(parents=True, exist_ok=True)
    shared_input_dir = remote_output_dir / "shared_inputs"
    if save_shared_input:
        shared_input_dir.mkdir(parents=True, exist_ok=True)
    json_out = remote_output_dir / "probe_eval_loader_drift_report.json"

    cmd = [
        "python",
        str(REMOTE_REPO / "tools/probe_eval_loader_drift.py"),
        "--video-names-file",
        str(REMOTE_REPO / "upstream/public_test_video_names.txt"),
        "--data-dir",
        str(REMOTE_REPO / "upstream/videos"),
        "--video-limit",
        str(video_limit),
        "--max-batches",
        str(max_batches),
        "--batch-size",
        str(batch_size),
        "--json-out",
        str(json_out),
        "--run-forward-cells",
    ]
    if save_shared_input:
        cmd.extend(["--save-shared-input-dir", str(shared_input_dir)])

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
    if json_out.is_file():
        artifacts["probe_eval_loader_drift_report.json"] = json_out.read_bytes()
    if save_shared_input and shared_input_dir.is_dir():
        for p in sorted(shared_input_dir.iterdir()):
            if p.is_file():
                # Only inline small tensors (<25 MB); skip large ones.
                size = p.stat().st_size
                if size <= 25 * 1024 * 1024:
                    artifacts[f"shared_inputs/{p.name}"] = p.read_bytes()
                else:
                    artifacts[f"shared_inputs/{p.name}.size.txt"] = (
                        f"{size}\n".encode("utf-8")
                    )

    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-32000:],
        "stderr": proc.stderr[-32000:],
        "elapsed_seconds": elapsed,
        "host_system": _platform.system(),
        "host_machine": _platform.machine(),
        "host_platform": _platform.platform(),
        "cuda_device_name": cuda_device_name,
        "cuda_capability": cuda_capability,
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "dali_available": dali_available,
        "dali_version": dali_version,
        "artifacts": artifacts,
        "tag": "[diagnostic-not-score]",
        "dispatch_attempted": True,
        **NON_PROMOTABLE_FIELDS,
    }


@app.local_entrypoint()
def main(
    output_dir: str = "",
    video_limit: int = 1,
    max_batches: int = 1,
    batch_size: int = 4,
    save_shared_input: bool = True,
    detach: bool = True,
    provider_detach_ack: bool = True,
    lane_id: str = "lane_cpu_cuda_xray_p5_landing_loader_dali_capture",
    instance_job_id: str = "",
    claim_agent: str = "claude:modal_loader_drift_capture",
    claim_notes: str = "",
    force_claim: bool = False,
) -> None:
    if not output_dir:
        label = safe_artifact_label(f"loader_dali_capture_{int(time.time())}")
        output_path = (
            Path("experiments/results/cpu_cuda_xray_p5_landing_loader_dali_capture")
            / label
        ).resolve()
    else:
        output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if not instance_job_id:
        instance_job_id = f"loader_dali_capture_{int(time.time())}"

    local_summary: dict[str, Any] = {
        "tool": "experiments/modal_loader_drift_capture.py",
        "app": APP_NAME,
        "canonical_path": "tools/probe_eval_loader_drift.py --run-forward-cells (Modal T4 + nvidia-dali-cuda120)",
        "video_limit": video_limit,
        "max_batches": max_batches,
        "batch_size": batch_size,
        "save_shared_input": save_shared_input,
        "modal_dispatch_mode": "detached_spawn" if detach else "synchronous_remote",
        "schema_version": 1,
        "tag": "[diagnostic-not-score]",
        **NON_PROMOTABLE_FIELDS,
    }

    write_json(
        output_path / "modal_loader_drift_local_request.json", local_summary
    )

    claim_spec = DispatchClaimSpec(
        lane_id=lane_id,
        instance_job_id=instance_job_id,
        agent=claim_agent,
        platform="modal",
        force=force_claim,
        notes=(
            claim_notes
            or "D4 DALI loader-drift capture on Modal T4 (nvidia-dali-cuda120 in image)"
        ),
    )

    record_dispatch_claim(
        repo_root=Path.cwd(),
        spec=claim_spec,
        status="active_modal_loader_drift_spawning" if detach else "active_modal_loader_drift_running",
    )

    call_kwargs = {
        "video_limit": video_limit,
        "max_batches": max_batches,
        "batch_size": batch_size,
        "save_shared_input": save_shared_input,
    }

    if detach:
        try:
            call = run_loader_drift_capture.spawn(**call_kwargs)
        except Exception as exc:
            record_dispatch_claim(
                repo_root=Path.cwd(),
                spec=DispatchClaimSpec(
                    lane_id=lane_id,
                    instance_job_id=instance_job_id,
                    agent=claim_agent,
                    platform="modal",
                    force=True,
                    notes=(
                        "Modal loader-drift spawn raised after dispatch boundary; "
                        f"manual reconciliation required; error={type(exc).__name__}"
                    ),
                ),
                status="ambiguous_modal_loader_drift_spawn_submission_recovery_required",
            )
            raise
        call_id = function_call_id(call)
        write_json(
            output_path / "modal_loader_drift_spawn.json",
            {
                "tool": "experiments/modal_loader_drift_capture.py",
                "app": APP_NAME,
                "call_id": call_id,
                "lane_id": lane_id,
                "instance_job_id": instance_job_id,
                "claim_agent": claim_agent,
                "claim_platform": "modal",
                "local_request": local_summary,
                "result_json_name": "modal_loader_drift_result.json",
                "recover_command": (
                    f".venv/bin/python tools/recover_modal_loader_drift.py "
                    f"--output-dir {output_path}"
                ),
                "schema_version": "modal_loader_drift_spawn_v1",
                **NON_PROMOTABLE_FIELDS,
            },
        )
        record_dispatch_claim(
            repo_root=Path.cwd(),
            spec=DispatchClaimSpec(
                lane_id=lane_id,
                instance_job_id=instance_job_id,
                agent=claim_agent,
                platform="modal",
                force=True,
                notes=(
                    f"Modal loader-drift detached spawn accepted; "
                    f"call_id={call_id}; output_dir={output_path}"
                ),
            ),
            status="active_modal_loader_drift_spawned",
        )
        print("=" * 60)
        print(f"MODAL LOADER-DRIFT DISPATCHED DETACHED call_id={call_id}")
        print(f"  Artifacts: {output_path}")
        print("=" * 60)
        return

    try:
        result = run_loader_drift_capture.remote(**call_kwargs)
    except Exception as exc:
        terminal_dispatch_claim(
            repo_root=Path.cwd(),
            spec=claim_spec,
            status="failed_modal_loader_drift_exception",
            notes=f"Modal loader-drift raised {type(exc).__name__}; no score claim",
        )
        raise

    artifacts = result.pop("artifacts", {})
    for name, data in sorted(artifacts.items()):
        target = output_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    write_json(output_path / "modal_loader_drift_result.json", result)
    print("=" * 60)
    print(f"MODAL LOADER-DRIFT CAPTURE COMPLETE")
    print(f"  ok={result.get('ok')}")
    print(f"  Artifacts: {output_path}")
    print(f"  DALI version: {result.get('dali_version')}")
    print(f"  Elapsed: {result.get('elapsed_seconds')}s")
    print("=" * 60)
    terminal_dispatch_claim(
        repo_root=Path.cwd(),
        spec=claim_spec,
        status="completed_modal_loader_drift_recovered" if result.get("ok") else "failed_modal_loader_drift_nonzero_rc",
        notes=(
            f"Recovered via sync .remote(); ok={result.get('ok')}; "
            f"dali_version={result.get('dali_version')}; "
            f"elapsed={result.get('elapsed_seconds')}s"
        ),
    )
