#!/usr/bin/env python3
"""Retained fx5/rc2 scorer-payload producer for the JO1 prerequisite seal.

This worker performs only the producer stage: two exact receiver decodes for a
raw-byte determinism check, followed by one full n600 SegNet and PoseNet pass.
It reuses the proven JS1B scorer materializer so every per-batch scorer input,
full output, cursor, field, and Pose6 vector remains on the mounted Modal
auth-cache volume.  It never enters JO1 training or emits a score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments import ddm_jo1_joint_objective_design as design
from experiments import ddm_js1b_cuda_argmax_field_materializer_worker as retained

SCHEMA = "ddm_jo1_payload_materializer_result.v1"
REQUEST_SCHEMA = "ddm_jo1_payload_materializer_request.v1"
CHECKPOINT_SCHEMA = "ddm_jo1_payload_materializer_checkpoint.v1"
AXIS = "[contest-CUDA T4 frozen-SegNet/PoseNet component payloads, n600, batch=16] COMPONENT-ONLY"


class JO1MaterializerError(RuntimeError):
    """A producer, retention, storage, lineage, or resume invariant failed."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tree_record(root: Path) -> dict[str, Any]:
    rows = []
    digest = hashlib.sha256()
    for child in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = child.relative_to(root).as_posix()
        record = retained.file_record(child)
        rows.append({"relative_path": relative, **record})
        digest.update(relative.encode("utf-8"))
        digest.update(str(record["bytes"]).encode("ascii"))
        digest.update(str(record["sha256"]).encode("ascii"))
    return {
        "path": str(root),
        "file_count": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "sha256": digest.hexdigest(),
        "files": rows,
    }


def _require_upload(payload: bytes, record: dict[str, Any], label: str) -> None:
    if len(payload) != int(record["bytes"]):
        raise JO1MaterializerError(f"{label} upload byte count differs")
    if _sha256_bytes(payload) != str(record["sha256"]):
        raise JO1MaterializerError(f"{label} upload SHA-256 differs")


def stage_uploaded_inputs(
    *,
    run_root: Path,
    request: dict[str, Any],
    archive_bytes: bytes,
    runtime_zip_bytes: bytes,
) -> dict[str, Any]:
    """Persist uploaded bytes before any decode and make resume byte-exact."""
    run_root.mkdir(parents=True, exist_ok=True)
    if request.get("schema") != REQUEST_SCHEMA:
        raise JO1MaterializerError("materializer request schema differs")
    _require_upload(archive_bytes, request["uploads"]["archive.zip"], "archive")
    _require_upload(
        runtime_zip_bytes,
        request["uploads"]["submission_dir.zip"],
        "runtime bundle",
    )
    request_path = run_root / "inputs/REQUEST.json"
    request_payload = retained.canonical_json_bytes(request)
    if request_path.is_file() and request_path.read_bytes() != request_payload:
        raise JO1MaterializerError("retained request differs from resumed request")
    if not request_path.is_file():
        retained.atomic_bytes(request_path, request_payload)
    for filename, payload in (
        ("archive.zip", archive_bytes),
        ("submission_dir.zip", runtime_zip_bytes),
    ):
        path = run_root / "inputs" / filename
        expected = request["uploads"][filename]
        if path.is_file():
            retained.require_record(path, expected)
        else:
            retained.atomic_bytes(path, payload)
            retained.require_record(path, expected)
    return {
        "request": retained.file_record(request_path),
        "archive": retained.file_record(run_root / "inputs/archive.zip"),
        "runtime_bundle": retained.file_record(run_root / "inputs/submission_dir.zip"),
    }


def storage_preflight(run_root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(run_root)
    already_retained = retained.current_retained_bytes(run_root)
    remaining_payload = max(
        0, design.MATERIALIZER_EXPECTED_RETAINED_PAYLOAD_BYTES - already_retained
    )
    required_free = remaining_payload + design.MATERIALIZER_STORAGE_RESERVE_BYTES
    result = {
        "schema": "ddm_jo1_payload_materializer_storage_preflight.v1",
        "tier": str(run_root),
        "free_bytes": usage.free,
        "already_retained_bytes": already_retained,
        "expected_total_retained_payload_bytes": (
            design.MATERIALIZER_EXPECTED_RETAINED_PAYLOAD_BYTES
        ),
        "remaining_payload_bytes": remaining_payload,
        "reserve_bytes": design.MATERIALIZER_STORAGE_RESERVE_BYTES,
        "required_free_bytes": required_free,
        "training_requirement_bytes": design.TRAINING_MIN_AP_FREE_BYTES,
        "training_requirement_applied": False,
        "passed": usage.free >= required_free,
        "cleanup_policy": "certify-or-block; every receiver and scorer payload is retained",
    }
    retained.atomic_json(run_root / "STORAGE_PREFLIGHT.json", result)
    if not result["passed"]:
        raise JO1MaterializerError(
            f"materializer storage preflight failed: free={usage.free}, required={required_free}"
        )
    return result


def _validate_scorer_batches(
    scorer: dict[str, Any], payload_keys: tuple[str, ...]
) -> None:
    if int(scorer.get("batch_size", -1)) != retained.BATCH_SIZE:
        raise JO1MaterializerError("retained scorer batch size differs")
    receipt = scorer["retained_batch_receipts"]
    retained.require_record(Path(receipt["path"]), receipt)
    rows = [
        json.loads(line)
        for line in Path(receipt["path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cursor = 0
    for ordinal, row in enumerate(rows):
        pair_end = int(row.get("pair_end", -1))
        count = pair_end - cursor
        if (
            int(row.get("ordinal", -1)) != ordinal
            or int(row.get("pair_start", -1)) != cursor
            or count <= 0
            or count > retained.BATCH_SIZE
            or count > design.MATERIALIZER_MAX_CHUNK_PAIRS
        ):
            raise JO1MaterializerError("retained scorer batch cursor differs")
        for key in payload_keys:
            record = row[key]
            retained.require_record(Path(record["path"]), record)
        cursor = pair_end
    if cursor != retained.N_PAIRS:
        raise JO1MaterializerError(
            f"retained scorer batch census differs: {cursor}/{retained.N_PAIRS}"
        )


def _validate_final_payloads(result: dict[str, Any]) -> None:
    for record in (
        result["receivers"]["primary"]["raw"],
        result["receivers"]["repeat"]["raw"],
        result["scorers"]["segnet"]["argmax"],
        result["scorers"]["posenet"]["first6_vectors"],
        result["scorer_tuple"],
    ):
        retained.require_record(Path(record["path"]), record)
    _validate_scorer_batches(
        result["scorers"]["segnet"], ("source_payload", "seg_input", "logits")
    )
    _validate_scorer_batches(
        result["scorers"]["posenet"],
        ("source_payload", "pose_input", "pose_output_full"),
    )


def run(run_root: Path, resume_from: str) -> dict[str, Any]:
    """Run or resume the complete materializer, never a training stage."""
    import timm
    import torch
    import torchvision

    from tac.contest_compliance import compute_upstream_snapshot_sha256

    started = time.time()
    request_path = run_root / "inputs/REQUEST.json"
    if not request_path.is_file():
        raise JO1MaterializerError("retained request is absent")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("schema") != REQUEST_SCHEMA:
        raise JO1MaterializerError("retained request schema differs")
    if resume_from != request.get("resume_from"):
        raise JO1MaterializerError("resume token differs from the retained workload")
    if int(request.get("batch_pairs", -1)) != retained.BATCH_SIZE:
        raise JO1MaterializerError("materializer batch size differs from the retained scorer")
    if (
        int(request.get("chunk_pair_limit", -1))
        != design.MATERIALIZER_MAX_CHUNK_PAIRS
        or retained.BATCH_SIZE > design.MATERIALIZER_MAX_CHUNK_PAIRS
    ):
        raise JO1MaterializerError("materializer chunk bound differs or is exceeded")

    final_path = run_root / "FINAL_RESULT.json"
    if final_path.is_file():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        if result.get("execution_status") != "COMPLETE":
            raise JO1MaterializerError("retained final result is not complete")
        _validate_final_payloads(result)
        return result

    preflight = storage_preflight(run_root)
    for filename, record in request["uploads"].items():
        retained.require_record(run_root / f"inputs/{filename}", record)
    retained.checkpoint_once(
        run_root / "checkpoints/stage_00_inputs_and_storage.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "stage": "inputs_and_storage",
            "storage_preflight": preflight,
            "uploads": request["uploads"],
            "resume_from": resume_from,
            "complete": True,
        },
    )

    runtime_roots: dict[str, Path] = {}
    for name in ("primary", "repeat"):
        runtime_root = run_root / f"retained/runtime/{name}"
        retained.extract_zip_once(
            run_root / "inputs/submission_dir.zip",
            runtime_root,
            "RUNTIME_EXTRACTED.json",
        )
        if not (runtime_root / "inflate.sh").is_file():
            raise JO1MaterializerError(f"runtime {name} has no inflate.sh")
        runtime_roots[name] = runtime_root
    runtime_records = {name: _tree_record(root) for name, root in runtime_roots.items()}
    if runtime_records["primary"]["sha256"] != runtime_records["repeat"]["sha256"]:
        raise JO1MaterializerError("primary/repeat extracted runtime trees differ")
    retained.checkpoint_once(
        run_root / "checkpoints/stage_05_runtimes_extracted.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "stage": "runtimes_extracted",
            "runtime_records": runtime_records,
            "complete": True,
        },
    )

    archive_path = run_root / "inputs/archive.zip"
    archive_record = request["uploads"]["archive.zip"]
    receivers: dict[str, dict[str, Any]] = {}
    for ordinal, name in enumerate(("primary", "repeat"), start=10):
        receivers[name] = retained.decode_exact_receiver(
            name=f"{request['vehicle_id']}_{name}",
            archive_path=archive_path,
            archive_record=archive_record,
            runtime_root=runtime_roots[name],
            run_root=run_root,
        )
        retained.checkpoint_once(
            run_root / f"checkpoints/stage_{ordinal:02d}_{name}_decoded.json",
            {
                "schema": CHECKPOINT_SCHEMA,
                "stage": f"{name}_decoded",
                "receiver": receivers[name],
                "complete": True,
            },
        )
    raw_repeat = {
        "primary": receivers["primary"]["raw"],
        "repeat": receivers["repeat"]["raw"],
        "byte_identical": (
            receivers["primary"]["raw"]["bytes"] == receivers["repeat"]["raw"]["bytes"]
            and receivers["primary"]["raw"]["sha256"]
            == receivers["repeat"]["raw"]["sha256"]
        ),
    }
    if not raw_repeat["byte_identical"]:
        raise JO1MaterializerError("exact receiver deterministic repeat differs")
    retained.checkpoint_once(
        run_root / "checkpoints/stage_12_receiver_repeat.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "stage": "receiver_repeat",
            "repeat": raw_repeat,
            "complete": True,
        },
    )

    if not torch.cuda.is_available():
        raise JO1MaterializerError("materializer requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    if "T4" not in gpu_name.upper():
        raise JO1MaterializerError(f"materializer requires NVIDIA T4, got {gpu_name}")
    torch.manual_seed(retained.SEED)
    np.random.seed(retained.SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    raw_root = Path(receivers["primary"]["raw"]["path"]).parent
    segnet = retained.load_segnet(device)
    segnet_result = retained.score_argmax_field(
        source=request["vehicle_id"],
        raw_root=raw_root,
        raw_record=receivers["primary"]["raw"],
        scorer=segnet,
        device=device,
        run_root=run_root,
    )
    retained.checkpoint_once(
        run_root / "checkpoints/stage_20_segnet_scored.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "stage": "segnet_scored",
            "scorer": segnet_result,
            "complete": True,
        },
    )
    del segnet
    torch.cuda.empty_cache()

    posenet = retained.load_posenet(device)
    posenet_result = retained.score_pose_vectors(
        source=request["vehicle_id"],
        raw_root=raw_root,
        raw_record=receivers["primary"]["raw"],
        scorer=posenet,
        device=device,
        run_root=run_root,
    )
    retained.checkpoint_once(
        run_root / "checkpoints/stage_25_posenet_scored.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "stage": "posenet_scored",
            "scorer": posenet_result,
            "complete": True,
        },
    )

    upstream_snapshot_sha256 = compute_upstream_snapshot_sha256(
        retained.UPSTREAM,
        upstream_subdir=".",
        reject_executable_artifacts=True,
    )
    if not upstream_snapshot_sha256:
        raise JO1MaterializerError("remote upstream snapshot hash is absent")
    scorer_tuple_value = {
        "schema": "ddm_jo1_materialized_scorer_tuple.v1",
        "axis": AXIS,
        "vehicle_id": request["vehicle_id"],
        "pairs": retained.N_PAIRS,
        "batch_pairs": retained.BATCH_SIZE,
        "chunk_pair_limit": design.MATERIALIZER_MAX_CHUNK_PAIRS,
        "archive": retained.file_record(archive_path),
        "expected_modal_runtime_tree_sha256": request["expected_runtime_tree_sha256"],
        "runtime_bundle": retained.file_record(run_root / "inputs/submission_dir.zip"),
        "extracted_runtime_repeats": runtime_records,
        "source_targets_bound_by_seal": request["source_targets"],
        "base_argmax": segnet_result["argmax"],
        "base_pose6": posenet_result["first6_vectors"],
        "receiver_repeat": raw_repeat,
        "upstream_snapshot_sha256": upstream_snapshot_sha256,
        "scorer_weights": {
            "segnet": retained.file_record(retained.UPSTREAM / "models/segnet.safetensors"),
            "posenet": retained.file_record(retained.UPSTREAM / "models/posenet.safetensors"),
        },
        "score_claim": False,
        "promotion_eligible": False,
    }
    scorer_tuple_path = run_root / "SCORER_TUPLE.json"
    retained.atomic_json(scorer_tuple_path, scorer_tuple_value)
    scorer_tuple_record = retained.file_record(scorer_tuple_path)

    final = {
        "schema": SCHEMA,
        "execution_status": "COMPLETE",
        "status": "MATERIALIZED",
        "axis": AXIS,
        "selection_mode": "full population, no sampling, 600 non-overlapping pairs",
        "vehicle_id": request["vehicle_id"],
        "batch_pairs": retained.BATCH_SIZE,
        "chunk_pair_limit": design.MATERIALIZER_MAX_CHUNK_PAIRS,
        "seed": retained.SEED,
        "receivers": receivers,
        "deterministic_repeat": raw_repeat,
        "scorers": {"segnet": segnet_result, "posenet": posenet_result},
        "runtime_repeats": runtime_records,
        "scorer_tuple": scorer_tuple_record,
        "retention": {
            "volume_name": request["remote_volume_name"],
            "volume_run_root": str(run_root),
            "both_exact_receiver_raw_videos": True,
            "all_materialized_seg_inputs": True,
            "all_materialized_seg_logits": True,
            "base_argmax_field": segnet_result["argmax"],
            "all_materialized_pose_inputs": True,
            "all_materialized_pose_outputs": True,
            "base_pose6": posenet_result["first6_vectors"],
            "per_batch_cursors": {
                "segnet": segnet_result["retained_batch_receipts"],
                "posenet": posenet_result["retained_batch_receipts"],
            },
            "expected_payload_bytes_before_metadata": (
                design.MATERIALIZER_EXPECTED_RETAINED_PAYLOAD_BYTES
            ),
        },
        "provenance": {
            "source_git_head": request["source_git_head"],
            "workload_config_sha256": request["workload_config_sha256"],
            "dispatcher_source_sha256": request["dispatcher_source_sha256"],
            "worker_source_sha256": request["worker_source_sha256"],
            "upstream_snapshot_sha256": upstream_snapshot_sha256,
            "gpu_name": gpu_name,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "timm_version": timm.__version__,
            "numpy_version": np.__version__,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "torch_deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
        },
        "elapsed_seconds": time.time() - started,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "pointer_moved": False,
        "boundaries": {
            "measured": "exact receiver raw identity plus frozen SegNet/PoseNet component payloads",
            "not_measured": "JO1 training, candidate archive, complete score, or contest-CPU",
        },
    }
    _validate_final_payloads(final)
    retained.checkpoint_once(final_path, final)
    retained.checkpoint_once(run_root / "checkpoints/stage_30_final.json", final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resume-from", required=True)
    args = parser.parse_args(argv)
    result = run(args.run_root.resolve(), args.resume_from)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
