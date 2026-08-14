#!/usr/bin/env python3
"""Prepare, run, and receipt the retained MC36/F26 contest-CPU lift.

The sealed CUDA promotion stage is read-only.  This runner makes a new lifted
submission tree, applies only the audited device/thread changes, retains every
materialized payload, and can resume at the completed token-stage checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

SEALED_STAGE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/promotion_submission/runtime_stage"
)
WORK_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_f26p_runtime_cpu_lift_20260814")
LIFTED_STAGE = WORK_DIR / "lifted_submission_cpu"
INPUT_DIR = WORK_DIR / "input"
OUTPUT_DIR = WORK_DIR / "output"
RECEIPT_DIR = WORK_DIR / "receipts"
LOG_DIR = WORK_DIR / "logs"
FILE_LIST = WORK_DIR / "file_list.txt"

ARCHIVE_BYTES = 186_269
ARCHIVE_SHA256 = "f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de"
SEALED_F26_SHA256 = "a00eaf49534c3814b059a144494d0c2aeb3bcf060f4e15f716cd3bcca0175e89"
EXPECTED_T4_RAW_SHA256 = "a41ca69d2288d3edd8f009b03404ef070661297a8f962a067e663ff26f7c0e8b"
EXPECTED_RAW_BYTES = 600 * 2 * 874 * 1164 * 3
FRAME_BYTES = 874 * 1164 * 3
FRAME_COUNT = 600 * 2
THREAD_ENV = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "VECLIB_MAXIMUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
}


class CpuLiftError(RuntimeError):
    """Raised when a custody, runtime, or identity invariant is violated."""


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _tree_manifest(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        records.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return records


def _manifest_hash(records: list[dict[str, Any]]) -> str:
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verify_storage(required_bytes: int) -> dict[str, int]:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(WORK_DIR)
    if usage.free < required_bytes:
        raise CpuLiftError(f"Vertigo has {usage.free} free bytes; CPU lift requires {required_bytes}")
    return {"free_bytes": usage.free, "required_bytes": required_bytes}


def _replace_exact(path: Path, old: str, new: str, *, count: int = 1) -> None:
    source = path.read_text(encoding="utf-8")
    observed = source.count(old)
    if observed != count:
        raise CpuLiftError(f"expected {count} exact transform sites in {path}, observed {observed}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(source.replace(old, new), encoding="utf-8")
    os.replace(temporary, path)


def _copy_payload_from_archive(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["p"]:
            raise CpuLiftError("sealed archive must contain exactly member p")
        payload = archive.read("p")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def prepare(repo_root: Path) -> dict[str, Any]:
    """Create a fresh lifted copy without modifying the sealed runtime."""
    storage = _verify_storage(EXPECTED_RAW_BYTES + 2_000_000_000)
    sealed_archive = SEALED_STAGE / "archive.zip"
    sealed_f26 = SEALED_STAGE / "runtime" / "f26_inflate.py"
    lifted_source = repo_root / "experiments" / "ddm_f26p_f26_inflate_cpu.py"
    if _file_fact(sealed_archive)["sha256"] != ARCHIVE_SHA256:
        raise CpuLiftError("sealed MC36 archive SHA-256 differs from the charter pin")
    if sealed_archive.stat().st_size != ARCHIVE_BYTES:
        raise CpuLiftError("sealed MC36 archive byte count differs from the charter pin")
    if _sha256_file(sealed_f26) != SEALED_F26_SHA256:
        raise CpuLiftError("sealed F26 runtime source differs from the audited source")

    if LIFTED_STAGE.exists():
        existing = RECEIPT_DIR / "prepare.json"
        if not existing.is_file():
            raise CpuLiftError("lifted stage exists without a preparation receipt")
        receipt = json.loads(existing.read_text(encoding="utf-8"))
        current = _tree_manifest(LIFTED_STAGE)
        if _manifest_hash(current) != receipt["lifted_runtime"]["manifest_sha256"]:
            raise CpuLiftError("existing lifted stage differs from its preparation receipt")
        if _sha256_file(LIFTED_STAGE / "runtime" / "f26_inflate.py") != _sha256_file(lifted_source):
            raise CpuLiftError("existing lifted stage differs from the repo lift module")
        return receipt

    temporary_stage = WORK_DIR / f".lifted_submission_cpu.{os.getpid()}.tmp"
    if temporary_stage.exists():
        raise CpuLiftError(f"refusing to overwrite temporary stage {temporary_stage}")
    shutil.copytree(SEALED_STAGE, temporary_stage)
    shutil.copy2(lifted_source, temporary_stage / "runtime" / "f26_inflate.py")

    residual_path = temporary_stage / "runtime" / "residual_archive.py"
    _replace_exact(
        residual_path,
        "    configure_cuda_reproducibility()\n    base_hpac = materialize_ihs1(",
        '    if device.type == "cuda":\n        configure_cuda_reproducibility()\n    base_hpac = materialize_ihs1(',
    )
    entrypoint = temporary_stage / "inflate.py"
    _replace_exact(
        entrypoint,
        "import argparse\nimport hashlib\nimport json\nimport zipfile\n",
        "import argparse\nimport hashlib\nimport json\nimport os\nimport zipfile\n",
    )
    _replace_exact(
        entrypoint,
        "from runtime.f26_inflate import inflate_archive\n",
        'for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", '
        '"OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", '
        '"NUMEXPR_NUM_THREADS"):\n'
        '    os.environ[_name] = "4"\n\n'
        "from runtime.f26_inflate import inflate_archive\n",
    )
    _replace_exact(
        entrypoint,
        '        renderer_dir=here / "cpr1",\n    )\n',
        '        renderer_dir=here / "cpr1",\n'
        '        device_name="cpu",\n'
        "        num_threads=4,\n"
        '        checkpoint_dir=args.destination.parent / ".f26_cpu_checkpoints",\n'
        "    )\n",
    )

    os.replace(temporary_stage, LIFTED_STAGE)
    _copy_payload_from_archive(LIFTED_STAGE / "archive.zip", INPUT_DIR / "p")
    _atomic_text(FILE_LIST, "0.mkv\n")
    sealed_manifest = _tree_manifest(SEALED_STAGE)
    lifted_manifest = _tree_manifest(LIFTED_STAGE)
    receipt = {
        "schema": "ddm_f26p_prepare.v1",
        "complete": True,
        "storage_preflight": storage,
        "sealed_stage": str(SEALED_STAGE),
        "sealed_archive": _file_fact(sealed_archive),
        "sealed_f26_source": _file_fact(sealed_f26),
        "sealed_runtime": {
            "file_count": len(sealed_manifest),
            "manifest_sha256": _manifest_hash(sealed_manifest),
            "files": sealed_manifest,
            "expected_cuda_worker_tree_sha256": ("776849ba00fa0e942c84ec63643ef067324a021f139726afff80855cfb613db9"),
        },
        "lifted_runtime": {
            "root": str(LIFTED_STAGE),
            "file_count": len(lifted_manifest),
            "manifest_sha256": _manifest_hash(lifted_manifest),
            "files": lifted_manifest,
        },
        "input_payload": _file_fact(INPUT_DIR / "p"),
        "file_list": _file_fact(FILE_LIST),
        "transforms": [
            "lifted f26_inflate makes device explicit and requires CPU thread count 4",
            "residual token decoder calls CUDA reproducibility setup only on CUDA",
            "entrypoint pins CPU and OMP/MKL/BLAS thread environments to 4",
        ],
    }
    _atomic_json(RECEIPT_DIR / "prepare.json", receipt)
    return receipt


def _parse_report(log_path: Path) -> dict[str, Any]:
    for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("schema") == "ddm_f26p_inflate_report.v1":
            return value
    raise CpuLiftError("decode log contains no complete F26 inflation report")


def decode(repo_root: Path) -> dict[str, Any]:
    """Run the real n600 archive through the four-thread lifted decoder."""
    prepare(repo_root)
    raw_path = OUTPUT_DIR / "0.raw"
    receipt_path = RECEIPT_DIR / "decode.json"
    if receipt_path.is_file() and raw_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("complete")
            and raw_path.stat().st_size == EXPECTED_RAW_BYTES
            and _sha256_file(raw_path) == receipt["raw_output"]["sha256"]
        ):
            return receipt
        raise CpuLiftError("existing decode output differs from its completion receipt")
    if raw_path.exists():
        raise CpuLiftError("raw output exists without a valid completion receipt")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "decode.log"
    command = [
        str(LIFTED_STAGE / "inflate.sh"),
        str(INPUT_DIR),
        str(OUTPUT_DIR),
        str(FILE_LIST),
    ]
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    environment["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), environment.get("PATH", "")])
    started_wall = time.time()
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "thread_env": THREAD_ENV}) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return_code = process.wait()
    elapsed = time.perf_counter() - started
    if return_code:
        _atomic_json(
            RECEIPT_DIR / "decode_failed.json",
            {
                "schema": "ddm_f26p_decode_failure.v1",
                "complete": False,
                "return_code": return_code,
                "wall_seconds": elapsed,
                "log": _file_fact(log_path),
            },
        )
        raise CpuLiftError(f"lifted decoder exited {return_code}; payloads retained")
    if not raw_path.is_file() or raw_path.stat().st_size != EXPECTED_RAW_BYTES:
        raise CpuLiftError("completed decoder did not materialize the canonical raw byte count")
    report = _parse_report(log_path)
    if report["raw_sha256"] != _sha256_file(raw_path):
        raise CpuLiftError("decoder report raw SHA-256 differs from retained output")
    if elapsed < 900:
        budget_verdict = "LIKELY-IN-BUDGET"
    elif elapsed <= 1800:
        budget_verdict = "MARGINAL"
    else:
        budget_verdict = "OVER"
    receipt = {
        "schema": "ddm_f26p_decode_receipt.v1",
        "complete": True,
        "axis_label": "[M5-CPU 4-thread LOWER BOUND on contest wall]",
        "host": {
            "platform": sys.platform,
            "machine": os.uname().machine,
            "cpu_count_visible": os.cpu_count(),
        },
        "thread_env": THREAD_ENV,
        "command": command,
        "started_unix": started_wall,
        "wall_seconds": elapsed,
        "budget_verdict": budget_verdict,
        "inflate_report": report,
        "raw_output": _file_fact(raw_path),
        "log": _file_fact(log_path),
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def _frame_manifest(raw_path: Path) -> tuple[list[dict[str, Any]], str]:
    records = []
    aggregate = hashlib.sha256()
    with raw_path.open("rb") as stream:
        for frame_index in range(FRAME_COUNT):
            payload = stream.read(FRAME_BYTES)
            if len(payload) != FRAME_BYTES:
                raise CpuLiftError(f"raw output ended during frame {frame_index}")
            aggregate.update(payload)
            records.append(
                {
                    "frame_index": frame_index,
                    "pair_index": frame_index // 2,
                    "within_pair": frame_index % 2,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        if stream.read(1):
            raise CpuLiftError("raw output has trailing bytes after 1200 frames")
    return records, aggregate.hexdigest()


def _compare_raw(cpu_path: Path, t4_path: Path) -> dict[str, Any]:
    if _sha256_file(t4_path) != EXPECTED_T4_RAW_SHA256:
        raise CpuLiftError("provided T4 raw does not match the charter identity pin")
    if t4_path.stat().st_size != EXPECTED_RAW_BYTES:
        raise CpuLiftError("provided T4 raw has a non-canonical byte count")
    divergent_frames = 0
    max_abs_delta = 0
    changed_bytes = 0
    with cpu_path.open("rb") as cpu, t4_path.open("rb") as t4:
        for _ in range(FRAME_COUNT):
            cpu_frame = cpu.read(FRAME_BYTES)
            t4_frame = t4.read(FRAME_BYTES)
            if cpu_frame == t4_frame:
                continue
            divergent_frames += 1
            for cpu_byte, t4_byte in zip(cpu_frame, t4_frame, strict=True):
                delta = abs(cpu_byte - t4_byte)
                if delta:
                    changed_bytes += 1
                    max_abs_delta = max(max_abs_delta, delta)
    return {
        "status": "MATCH" if divergent_frames == 0 else "MISMATCH_QUANTIFIED",
        "divergent_frames": divergent_frames,
        "frame_denominator": FRAME_COUNT,
        "changed_bytes": changed_bytes,
        "byte_denominator": EXPECTED_RAW_BYTES,
        "max_abs_u8_delta": max_abs_delta,
        "t4_raw": _file_fact(t4_path),
    }


def analyze(repo_root: Path) -> dict[str, Any]:
    """Persist the F26 hot-path operation census and bounded reuse inventory."""
    prepare(repo_root)
    decode_receipt_path = RECEIPT_DIR / "decode.json"
    if not decode_receipt_path.is_file():
        raise CpuLiftError("runtime analysis requires a completed decode receipt")
    decode_receipt = json.loads(decode_receipt_path.read_text(encoding="utf-8"))

    import torch

    renderer_dir = LIFTED_STAGE / "cpr1"
    sys.path.insert(0, str(LIFTED_STAGE))
    sys.path.insert(0, str(renderer_dir))
    try:
        spec = importlib.util.spec_from_file_location("_ddm_f26p_analysis_renderer", renderer_dir / "inflate.py")
        if spec is None or spec.loader is None:
            raise CpuLiftError("cannot load the staged F26 renderer for analysis")
        renderer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(renderer)
        from runtime.hpac_inference import optimize_sparse_evaluator
        from runtime.ihs2 import materialize_ihs1
        from runtime.residual_archive import _sparse_class, read_residual_archive

        parts = read_residual_archive(LIFTED_STAGE / "archive.zip")
        model = renderer.load_hpac(materialize_ihs1(parts.hpac_blob, renderer), torch.device("cpu"))
        sparse = _sparse_class(renderer_dir)(model, renderer.EVAL_H, renderer.EVAL_W)
        optimize_sparse_evaluator(sparse)
    finally:
        sys.path.pop(0)
        sys.path.pop(0)

    cache = sparse._cpr1_sparse_cache
    frames = int(renderer.N)
    patches = int(sparse.patch_count)
    groups = len(sparse.plans)
    channels = int(cache.conv_a_weight.shape[0])
    conv_a_input_width = int(cache.conv_a_weight.shape[1])
    feature_planes = model.num_classes + 2
    if conv_a_input_width % feature_planes:
        raise CpuLiftError("conv-a input width does not factor into class+coordinate planes")
    active_offsets = conv_a_input_width // feature_planes
    summed_h = sum(int(plan.h_positions.shape[0]) for plan in sparse.plans)
    summed_b1 = sum(int(plan.b1_gather.shape[0]) for plan in sparse.plans)
    summed_targets = sum(int(plan.b2_gather.shape[0]) for plan in sparse.plans)
    depthwise = list(cache.depthwise_weights.values())
    b1_taps = int(depthwise[0].shape[2])
    b2_taps = int(depthwise[1].shape[2])
    conv_a_macs = frames * patches * summed_h * channels * conv_a_input_width
    b1_macs = frames * patches * summed_b1 * channels * b1_taps
    b2_macs = frames * patches * summed_targets * channels * b2_taps
    head_macs = frames * patches * summed_targets * channels * int(model.num_classes)
    gathered_class_terms = frames * patches * summed_h * active_offsets * channels

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))
    try:
        from experiments.contest_auth_eval import _runtime_dependency_manifest
        from tac.deploy.modal.auth_eval import (
            modal_uploaded_submission_dir_runtime_manifest,
        )
    finally:
        sys.path.pop(0)
        sys.path.pop(0)

    modal_local_manifest = _runtime_dependency_manifest(LIFTED_STAGE / "inflate.sh", repo_root / "upstream")
    modal_projected_manifest = modal_uploaded_submission_dir_runtime_manifest(
        modal_local_manifest,
        remote_submission_dir="/tmp/modal_auth_eval_cpu/submission_dir",
    )

    wall = float(decode_receipt["wall_seconds"])
    stages = decode_receipt["inflate_report"]["stage_seconds"]
    token_wall = float(stages["token_decode_or_checkpoint_load"])
    render_wall = float(stages["neural_render_and_resize"])
    selector_wall = float(stages["frame0_selector_and_io"])
    two_x_token_wall = wall - token_wall + token_wall / 2.0

    asset_specs = [
        (
            Path("/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/artifacts/tq1c/hpac.bin.xz"),
            "HPAC model-packing precedent; values are a different trained payload",
        ),
        (
            Path("/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/artifacts/tq1c/tokens.bin"),
            "exact HPAC encode/decode payload precedent; not the MC36 token stream",
        ),
        (
            repo_root / ".omx/research/ddm_hb2_20260808/hb2_receipts.jsonl",
            "deploy-bound and exact pack/decode gates for future HPAC retraining",
        ),
        (
            repo_root / "src/tac/substrates/_shared/mlx_score_aware/adapter.py",
            "local scorer-aware training adapter; no contest decoder role",
        ),
        (
            repo_root / "src/tac/substrates/_shared/mlx_score_aware/portability.py",
            "MLX-to-portable export contract for future trained carriers",
        ),
        (
            repo_root / "src/tac/local_acceleration/mlx_scorer_adapters.py",
            "local Seg/Pose screening adapter; advisory only",
        ),
        (
            repo_root / "src/tac/local_acceleration/metal_grouped_conv_backward.py",
            "local grouped-convolution gradient work; not F26 forward decode",
        ),
        (
            repo_root / "src/tac/local_acceleration/metal_segnet_conv.py",
            "local SegNet forward screening; not upstream or contest CPU",
        ),
        (
            repo_root / "src/tac/local_acceleration/metal_fused_r_operator.py",
            "local differentiable R-operator training/screening",
        ),
        (
            repo_root / "src/tac/local_acceleration/metal_sparse_adjoint.py",
            "local sparse scorer adjoint for candidate optimization",
        ),
        (
            repo_root / "src/tac/local_acceleration/metal_integer_r_adjoint.py",
            "local integer-R adjoint for candidate optimization",
        ),
    ]
    assets = []
    for path, reuse in asset_specs:
        if not path.is_file():
            assets.append({"path": str(path), "exists": False, "reuse": reuse})
            continue
        assets.append({**_file_fact(path), "exists": True, "reuse": reuse})

    result = {
        "schema": "ddm_f26p_runtime_analysis.v1",
        "complete": True,
        "axis_label": "[M5-CPU 4-thread LOWER BOUND on contest wall]",
        "score_claim": False,
        "measured_stage_seconds": {
            "subprocess_wall": wall,
            "token_hpac_and_rc64": token_wall,
            "neural_render_and_resize": render_wall,
            "frame0_selector_and_io": selector_wall,
            "token_fraction_of_subprocess_wall": token_wall / wall,
        },
        "hpac_operation_census": {
            "frames": frames,
            "groups_per_frame": groups,
            "group_calls": frames * groups,
            "patches_per_group_call": patches,
            "channels": channels,
            "active_conv_a_offsets": active_offsets,
            "conv_a_input_width": conv_a_input_width,
            "per_patch_summed_h_positions": summed_h,
            "per_patch_summed_b1_positions": summed_b1,
            "per_patch_summed_targets": summed_targets,
            "per_frame_output_symbols": patches * summed_targets,
            "selected_logits_dense_equivalent_macs": {
                "conv_a": conv_a_macs,
                "depthwise_b1": b1_macs,
                "depthwise_b2": b2_macs,
                "head": head_macs,
                "total": conv_a_macs + b1_macs + b2_macs + head_macs,
            },
            "direct_gathered_one_hot_class_terms": gathered_class_terms,
            "conv_a_dense_mac_to_gather_term_ratio": conv_a_macs / gathered_class_terms,
        },
        "derived_speed_bounds": {
            "ideal_remove_entire_token_stage_whole_wall_speedup": wall / (wall - token_wall),
            "target_two_x_token_stage_projected_local_wall_seconds": two_x_token_wall,
            "target_two_x_token_stage_projected_whole_wall_speedup": wall / two_x_token_wall,
            "ideal_remove_entire_render_stage_whole_wall_speedup": wall / (wall - render_wall),
            "scope": (
                "Amdahl projections from this one measured M5 run; not measured "
                "contest-CPU speedups and not a host-transfer model"
            ),
        },
        "structural_parallelism": {
            "archive_token_stream_count": 1,
            "within_frame_group_dependency": "causal current-token groups",
            "between_frame_dependency": "previous token frame enters HPAC context",
            "verdict": (
                "no independent existing streams to dispatch across four workers; "
                "parallel-stream work requires a new byte-closed wire/model"
            ),
        },
        "modal_runtime_custody": {
            "local_runtime_tree_sha256": modal_local_manifest.get("runtime_tree_sha256"),
            "runtime_files_sha256": modal_local_manifest.get("runtime_files_sha256"),
            "runtime_content_tree_sha256": modal_projected_manifest.get("runtime_content_tree_sha256"),
            "projected_remote_runtime_tree_sha256": modal_projected_manifest.get("runtime_tree_sha256"),
            "dispatch_expected_runtime_tree_argument": "auto",
            "reason": (
                "uploaded submission tree hashes are path-dependent; the wrapper "
                "validates the environment-free runtime FILES digest remotely"
            ),
        },
        "reuse_inventory": assets,
    }
    _atomic_json(RECEIPT_DIR / "runtime_analysis.json", result)
    return result


def finalize(repo_root: Path, t4_raw: Path | None) -> dict[str, Any]:
    """Write per-frame custody and the strongest locally supportable identity verdict."""
    decode_receipt = decode(repo_root)
    raw_path = OUTPUT_DIR / "0.raw"
    records, aggregate = _frame_manifest(raw_path)
    if aggregate != decode_receipt["raw_output"]["sha256"]:
        raise CpuLiftError("frame-manifest aggregate differs from raw-file SHA-256")
    manifest = {
        "schema": "ddm_f26p_frame_manifest.v1",
        "axis_label": "[M5-CPU 4-thread LOWER BOUND on contest wall]",
        "frame_order": "pair-major, frame-0 then frame-1, RGB uint8 HWC",
        "frame_count": FRAME_COUNT,
        "frame_bytes": FRAME_BYTES,
        "raw": _file_fact(raw_path),
        "frames": records,
    }
    _atomic_json(RECEIPT_DIR / "cpu_frame_manifest.json", manifest)
    if t4_raw is None:
        identity = {
            "status": ("MATCH" if aggregate == EXPECTED_T4_RAW_SHA256 else "MISMATCH_UNQUANTIFIED"),
            "cpu_raw_sha256": aggregate,
            "expected_t4_raw_sha256": EXPECTED_T4_RAW_SHA256,
            "comparison_scope": "aggregate SHA-256 only; retained T4 bytes not supplied",
            "divergent_frames": None,
            "frame_denominator": FRAME_COUNT,
            "max_abs_u8_delta": None,
        }
    else:
        identity = _compare_raw(raw_path, t4_raw.resolve())
        identity["cpu_raw_sha256"] = aggregate
        identity["expected_t4_raw_sha256"] = EXPECTED_T4_RAW_SHA256
    result = {
        "schema": "ddm_f26p_result.v1",
        "complete": True,
        "archive": _file_fact(LIFTED_STAGE / "archive.zip"),
        "lifted_runtime_manifest_sha256": json.loads((RECEIPT_DIR / "prepare.json").read_text(encoding="utf-8"))[
            "lifted_runtime"
        ]["manifest_sha256"],
        "decode": decode_receipt,
        "cpu_frame_manifest": _file_fact(RECEIPT_DIR / "cpu_frame_manifest.json"),
        "identity": identity,
        "runtime_analysis": _file_fact(RECEIPT_DIR / "runtime_analysis.json"),
    }
    _atomic_json(RECEIPT_DIR / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare", "decode", "analyze", "finalize", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument(
        "--t4-raw",
        type=Path,
        help="optional retained T4 raw whose SHA-256 is the charter pin",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.command == "prepare":
        result = prepare(repo_root)
    elif args.command == "decode":
        result = decode(repo_root)
    elif args.command == "analyze":
        result = analyze(repo_root)
    else:
        analyze(repo_root)
        result = finalize(repo_root, args.t4_raw)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
