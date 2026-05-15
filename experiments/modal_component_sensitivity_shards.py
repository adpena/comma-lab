# SPDX-License-Identifier: MIT
"""Dispatch CUDA direct-FD component-sensitivity shards on Modal.

This is a queue-diversity fallback for the May 3 contest push. It is
diagnostic only: Modal direct-renderer sensitivity is not canonical
``archive.zip -> inflate.sh -> upstream/evaluate.py`` score evidence.

Dispatch:

    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \\
      experiments/modal_component_sensitivity_shards.py \\
      --baseline-archive experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip \\
      --gpu A10G \\
      --label pfp16_direct_fd_modal_a10g_20260501 \\
      --shard-count 16 \\
      --shards 0-15

Recover:

    .venv/bin/python experiments/modal_component_sensitivity_shards.py recover \\
      --label pfp16_direct_fd_modal_a10g_20260501
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_PFP16_SHA256 = "0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f"
EXPECTED_PFP16_BYTES = 686635
RESULT_ROOT = REPO_ROOT / "experiments/results/modal_component_sensitivity"
APP_NAME = "comma-component-sensitivity"
REMOTE_PYTHONPATH = "/workspace/pact/src:/workspace/pact/upstream:/workspace/pact"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "unzip",
        "build-essential",
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "safetensors",
        "einops",
        "av",
        "tqdm",
        "scipy",
        "numpy<2.0",
        "Pillow",
        "pydantic>=2.0",
        "segmentation-models-pytorch",
        "timm",
    )
    .env({"PYTHONPATH": REMOTE_PYTHONPATH})
    .add_local_dir(
        "src", remote_path="/workspace/pact/src"
    )  # MODAL_MANUAL_MOUNT_OK:narrow sensitivity-shard dispatcher; targeted upstream files; trainer-discovery N/A
    .add_local_dir(
        "upstream", remote_path="/workspace/pact/upstream"
    )  # MODAL_MANUAL_MOUNT_OK:narrow sensitivity-shard dispatcher; targeted upstream files; trainer-discovery N/A
    .add_local_dir(
        "scripts", remote_path="/workspace/pact/scripts"
    )  # MODAL_MANUAL_MOUNT_OK:narrow sensitivity-shard dispatcher; targeted upstream files; trainer-discovery N/A
    .add_local_file(
        "experiments/__init__.py", remote_path="/workspace/pact/experiments/__init__.py"
    )  # MODAL_MANUAL_MOUNT_OK:narrow sensitivity-shard dispatcher; targeted upstream files; trainer-discovery N/A
    .add_local_file(  # MODAL_MANUAL_MOUNT_OK:narrow sensitivity-shard dispatcher; targeted upstream files; trainer-discovery N/A
        "experiments/profile_component_sensitivity.py",
        remote_path="/workspace/pact/experiments/profile_component_sensitivity.py",
    )
    .add_local_file(  # MODAL_MANUAL_MOUNT_OK:narrow sensitivity-shard dispatcher; targeted upstream files; trainer-discovery N/A
        "experiments/profile_hessian_per_weight.py",
        remote_path="/workspace/pact/experiments/profile_hessian_per_weight.py",
    )
    .add_local_file(  # MODAL_MANUAL_MOUNT_OK:narrow sensitivity-shard dispatcher; targeted upstream files; trainer-discovery N/A
        "experiments/convert_fisher_to_owv3_sensitivity_map.py",
        remote_path="/workspace/pact/experiments/convert_fisher_to_owv3_sensitivity_map.py",
    )
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_shards(spec: str, *, shard_count: int) -> list[int]:
    out: set[int] = set()
    for part in str(spec).split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            lo_s, hi_s = item.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if hi < lo:
                raise ValueError(f"invalid descending shard range: {item!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(item))
    shards = sorted(out)
    if not shards:
        raise ValueError("--shards selected no shard indices")
    bad = [idx for idx in shards if idx < 0 or idx >= shard_count]
    if bad:
        raise ValueError(f"shard index outside [0,{shard_count}): {bad}")
    return shards


def _artifact_files(root: Path) -> dict[str, bytes]:
    names = (
        "lightning_queue_metadata.json",
        "diagnostic_component_sensitivity_inputs.json",
        "diagnostic_component_sensitivity_run.json",
        "component_sensitivity_profile_summary.json",
        "sample_plan.json",
        "stability.json",
        "perturbation_basis_v1.json",
        "posenet_sensitivity_map.pt",
        "segnet_sensitivity_map.pt",
        "combined_sensitivity_map.pt",
        "posenet_holdout_sensitivity_map.pt",
        "segnet_holdout_sensitivity_map.pt",
        "combined_holdout_sensitivity_map.pt",
        "posenet_response_curve.json",
        "segnet_response_curve.json",
        "combined_response_curve.json",
        "diagnostic_component_sensitivity.log",
        "modal_component_sensitivity_preflight.json",
        "modal_component_sensitivity_validation.json",
        "lightning_supply_chain_scan.json",
    )
    artifacts: dict[str, bytes] = {}
    for name in names:
        path = root / name
        if path.is_file():
            artifacts[name] = path.read_bytes()
    return artifacts


def _write_local_modal_metadata(
    *,
    label: str,
    shard_count: int,
    shards: list[int],
    gpu: str,
    baseline_archive: Path,
    baseline_sha256: str,
    baseline_bytes: int,
    call_ids: dict[int, str],
) -> None:
    out_dir = RESULT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "modal_call_ids.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "app": APP_NAME,
                "label": label,
                "gpu": gpu,
                "shard_count": shard_count,
                "shards": shards,
                "baseline_archive": str(baseline_archive),
                "baseline_sha256": baseline_sha256,
                "baseline_bytes": baseline_bytes,
                "call_ids": {str(k): v for k, v in sorted(call_ids.items())},
                "score_claim": False,
                "promotion_eligible": False,
                "evidence_grade": "diagnostic_cuda_modal_direct_renderer_finite_difference",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _safe_extract_member(zf, member_name: str, dest: Path) -> None:

    info = zf.getinfo(member_name)
    if info.is_dir():
        raise ValueError(f"archive member is a directory: {member_name}")
    target = dest / Path(member_name).name
    if target.name != member_name:
        raise ValueError(f"unexpected archive member path: {member_name}")
    with zf.open(info, "r") as src, target.open("wb") as fh:
        fh.write(src.read())


def _run_component_sensitivity_inner(
    *,
    archive_bytes: bytes,
    archive_sha256: str,
    archive_size_bytes: int,
    label: str,
    shard_index: int,
    shard_count: int,
    max_seconds: int,
) -> dict:
    import os
    import subprocess
    import time
    import zipfile

    workspace = Path("/workspace/pact")
    out_dir = Path("/tmp/out")
    extracted = out_dir / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted.mkdir(parents=True, exist_ok=True)
    os.chdir(workspace)

    preflight = {
        "schema_version": 1,
        "tool": "experiments/modal_component_sensitivity_shards.py",
        "label": label,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "device_requested": "cuda",
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size_bytes,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "diagnostic_cuda_modal_direct_renderer_finite_difference",
    }
    try:
        import torch

        preflight["torch_cuda_available"] = bool(torch.cuda.is_available())
        preflight["torch_device_count"] = int(torch.cuda.device_count())
        preflight["torch_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:  # pragma: no cover - remote diagnostics
        preflight["torch_probe_error"] = repr(exc)
    (out_dir / "modal_component_sensitivity_preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n"
    )
    if preflight.get("torch_cuda_available") is not True:
        return {
            "returncode": 3,
            "label": label,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "artifacts": _artifact_files(out_dir),
            "error": "Modal runtime has no CUDA device",
        }
    if _sha256_bytes(archive_bytes) != archive_sha256:
        return {
            "returncode": 4,
            "label": label,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "artifacts": _artifact_files(out_dir),
            "error": "uploaded archive SHA mismatch",
        }
    if len(archive_bytes) != archive_size_bytes:
        return {
            "returncode": 5,
            "label": label,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "artifacts": _artifact_files(out_dir),
            "error": "uploaded archive byte-size mismatch",
        }

    archive_path = out_dir / "archive.zip"
    archive_path.write_bytes(archive_bytes)
    with zipfile.ZipFile(archive_path) as zf:
        names = set(zf.namelist())
        required = {"renderer.bin", "masks.mkv", "optimized_poses.bin"}
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(f"archive missing required members: {missing}")
        for name in sorted(required):
            _safe_extract_member(zf, name, extracted)

    metadata = {
        "schema_version": 1,
        "job_name": f"{label}_shard{shard_index:02d}_of{shard_count}",
        "role": "modal_diagnostic_component_sensitivity",
        "expected_baseline_archive_sha256": archive_sha256,
        "expected_baseline_archive_size_bytes": archive_size_bytes,
        "finite_difference_shard_index": shard_index,
        "finite_difference_shard_count": shard_count,
        "score_claim": False,
        "promotion_eligible": False,
        "score_source": "none:modal_diagnostic_component_sensitivity_non_promotable",
    }
    (out_dir / "lightning_queue_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (out_dir / "diagnostic_component_sensitivity_inputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "baseline_archive": {
                    "sha256": archive_sha256,
                    "bytes": archive_size_bytes,
                    "source": "modal_function_argument",
                },
                "role": "modal_diagnostic_component_sensitivity",
                "score_claim": False,
                "promotion_eligible": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    supply_scan = out_dir / "lightning_supply_chain_scan.json"
    subprocess.run(  # subprocess-no-check-OK: advisory supply-chain scan; non-zero rc must not abort the diagnostic run
        [
            sys.executable,
            "scripts/scan_lightning_supply_chain.py",
            "--json-out",
            str(supply_scan),
            "--quiet",
            "--strict",
        ],
        cwd=workspace,
        check=False,
    )

    argv = [
        sys.executable,
        "-u",
        "experiments/profile_component_sensitivity.py",
        "--checkpoint",
        str(extracted / "renderer.bin"),
        "--video",
        str(workspace / "upstream/videos/0.mkv"),
        "--masks-mkv",
        str(extracted / "masks.mkv"),
        "--poses",
        str(extracted / "optimized_poses.bin"),
        "--upstream",
        str(workspace / "upstream"),
        "--output-dir",
        str(out_dir),
        "--top-k-pairs",
        "64",
        "--pair-batch",
        "4",
        "--response-top-k",
        "8",
        "--response-epsilons=-0.002,-0.001,-0.0005,0.0,0.0005,0.001,0.002",
        "--split-seed",
        "20260430",
        "--holdout-fraction",
        "0.2",
        "--aggregate",
        "sum",
        "--device",
        "cuda",
        "--promotion-finite-difference",
        "--finite-difference-epsilon",
        "0.001",
        "--finite-difference-shard-index",
        str(shard_index),
        "--finite-difference-shard-count",
        str(shard_count),
        "--all-pairs",
    ]
    (out_dir / "diagnostic_component_sensitivity_run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool": "experiments/modal_component_sensitivity_shards.py",
                "profile_argv": argv,
                "score_claim": False,
                "promotion_eligible": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    t0 = time.monotonic()
    log_path = out_dir / "diagnostic_component_sensitivity.log"
    timed_out = False
    with log_path.open("w") as logf:
        try:
            proc = subprocess.run(
                argv,
                cwd=workspace,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=max_seconds,
                check=False,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            rc = 124
    elapsed = time.monotonic() - t0
    validation = {
        "schema_version": 1,
        "tool": "experiments/modal_component_sensitivity_shards.py",
        "returncode": rc,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed,
        "label": label,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "baseline_archive_sha256": archive_sha256,
        "baseline_archive_size_bytes": archive_size_bytes,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "diagnostic_cuda_modal_direct_renderer_finite_difference",
    }
    summary_path = out_dir / "component_sensitivity_profile_summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
            validation["summary_sensitivity_source"] = summary.get("sensitivity_source")
            validation["summary_device"] = summary.get("device")
            validation["certification_handoff_eligible"] = bool(summary.get("certification_handoff_eligible"))
        except json.JSONDecodeError:
            validation["summary_parse_error"] = True
    (out_dir / "modal_component_sensitivity_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    return {
        "returncode": rc,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed,
        "label": label,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "score_claim": False,
        "promotion_eligible": False,
        "artifacts": _artifact_files(out_dir),
    }


@app.function(image=image, gpu="A10G", timeout=6 * 3600)
def run_component_sensitivity_a10g(
    archive_bytes: bytes,
    archive_sha256: str,
    archive_size_bytes: int,
    label: str,
    shard_index: int,
    shard_count: int,
    max_seconds: int,
) -> dict:
    return _run_component_sensitivity_inner(
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size_bytes,
        label=label,
        shard_index=shard_index,
        shard_count=shard_count,
        max_seconds=max_seconds,
    )


@app.function(image=image, gpu="T4", timeout=6 * 3600)
def run_component_sensitivity_t4(
    archive_bytes: bytes,
    archive_sha256: str,
    archive_size_bytes: int,
    label: str,
    shard_index: int,
    shard_count: int,
    max_seconds: int,
) -> dict:
    return _run_component_sensitivity_inner(
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size_bytes,
        label=label,
        shard_index=shard_index,
        shard_count=shard_count,
        max_seconds=max_seconds,
    )


@app.local_entrypoint()
def main(
    baseline_archive: str,
    gpu: str = "A10G",
    label: str = "pfp16_direct_fd_modal_a10g_20260501",
    shard_count: int = 16,
    shards: str = "0-15",
    timeout_hours: float = 5.5,
):
    archive_path = Path(baseline_archive)
    if not archive_path.is_file():
        raise SystemExit(f"FATAL: baseline archive not found: {archive_path}")
    archive_bytes = archive_path.read_bytes()
    archive_sha = _sha256_bytes(archive_bytes)
    archive_size = len(archive_bytes)
    if archive_sha != EXPECTED_PFP16_SHA256 or archive_size != EXPECTED_PFP16_BYTES:
        raise SystemExit(
            "FATAL: this Modal launcher is currently fenced to the PFP16 A++ "
            f"baseline; got sha={archive_sha} bytes={archive_size}"
        )
    selected = _parse_shards(shards, shard_count=shard_count)
    gpu_norm = gpu.upper()
    if gpu_norm == "A10G":
        fn = run_component_sensitivity_a10g
    elif gpu_norm == "T4":
        fn = run_component_sensitivity_t4
    else:
        raise SystemExit("FATAL: --gpu must be A10G or T4")
    max_seconds = max(60, min(int(float(timeout_hours) * 3600), 6 * 3600))

    # Catalog #245 — register every shard's call_id in the canonical ledger
    # so the harvester can query unharvested shards by lane_id.
    try:
        from tac.deploy.modal.call_id_ledger import register_dispatched_call_id as _register_call_id
    except Exception:  # pragma: no cover
        _register_call_id = None  # type: ignore[assignment]

    call_ids: dict[int, str] = {}
    for shard_index in selected:
        call = fn.spawn(
            archive_bytes,
            archive_sha,
            archive_size,
            label,
            int(shard_index),
            int(shard_count),
            int(max_seconds),
        )
        call_ids[int(shard_index)] = call.object_id
        print(f"DISPATCHED shard={shard_index:02d}/{shard_count} gpu={gpu_norm} call_id={call.object_id}")
        if _register_call_id is not None:
            try:
                _register_call_id(
                    call_id=str(call.object_id),
                    lane_id=f"lane_component_sensitivity_{label}",
                    label=f"{label}_shard_{int(shard_index):02d}_of_{int(shard_count):02d}",
                    platform="modal",
                    gpu=gpu_norm,
                    max_seconds=int(max_seconds),
                    agent="claude",
                    recipe="modal_component_sensitivity_shards",
                )
            except Exception as exc:  # pragma: no cover — best-effort wire-in
                print(
                    f"  WARNING: ledger registration failed for shard {shard_index}: {exc}",
                    file=sys.stderr,
                )

    _write_local_modal_metadata(
        label=label,
        shard_count=shard_count,
        shards=selected,
        gpu=gpu_norm,
        baseline_archive=archive_path,
        baseline_sha256=archive_sha,
        baseline_bytes=archive_size,
        call_ids=call_ids,
    )
    print(f"call ids saved under {RESULT_ROOT / label}")


def recover(label: str) -> int:
    metadata_path = RESULT_ROOT / label / "modal_call_ids.json"
    if not metadata_path.is_file():
        print(f"FATAL: missing {metadata_path}", file=sys.stderr)
        return 2
    payload = json.loads(metadata_path.read_text())
    call_ids = payload.get("call_ids")
    if not isinstance(call_ids, dict):
        print(f"FATAL: invalid call id metadata at {metadata_path}", file=sys.stderr)
        return 2

    pending: list[int] = []
    failed: list[int] = []
    saved = 0
    for shard_s, call_id in sorted(call_ids.items(), key=lambda item: int(item[0])):
        shard_index = int(shard_s)
        fc = modal.FunctionCall.from_id(str(call_id))
        try:
            result = fc.get(timeout=0)
        except TimeoutError:
            pending.append(shard_index)
            continue
        if not isinstance(result, dict):
            failed.append(shard_index)
            continue
        shard_dir = RESULT_ROOT / label / f"shard_{shard_index:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        for name, data in result.get("artifacts", {}).items():
            path = shard_dir / str(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            saved += 1
        (shard_dir / "modal_result_summary.json").write_text(
            json.dumps(
                {key: value for key, value in result.items() if key != "artifacts"},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if result.get("returncode") not in (0, None):
            failed.append(shard_index)

    print(
        json.dumps(
            {
                "label": label,
                "saved_artifacts": saved,
                "pending_shards": pending,
                "failed_shards": failed,
                "result_dir": str(RESULT_ROOT / label),
                "score_claim": False,
                "promotion_eligible": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if failed else 0


def _recover_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("recover", choices=["recover"])
    parser.add_argument("--label", required=True)
    args = parser.parse_args(argv)
    return recover(args.label)


if __name__ == "__main__":
    raise SystemExit(_recover_cli(sys.argv[1:]))
