#!/usr/bin/env python3
"""Run resumable DDM MS4 metric producers under honest custody.

Only the Pose producer is identifiable from the presently landed inputs:
GT camera pairs identify pair IDs 0..599 directly.  The other three producers
require an explicit PF2 bucket-to-input/actuator assignment.  Until that edge
lands, this runner emits hashed blocker artifacts rather than duplicating one
measurement across 1,200 semantic keys.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_lambda_continuation_frontier import publish_immutable_json  # noqa: E402
from tac.optimization.ddm_metric_custody_bundle import (  # noqa: E402
    BUNDLE_SCHEMA,
    COMPONENT_SCHEMA,
    COMPOSITE_R_DATA_SCHEMA,
    DUAL_DATA_SCHEMA,
    EVIDENCE_AXIS,
    G3_REGISTRY_SCHEMA,
    HARD_PAIR_ORDER,
    PAIR_COUNT,
    PF2_ATLAS_SCHEMA,
    POINTER,
    POSE_DATA_SCHEMA,
    SCORER_BATCH_SIZE,
    SEG_DATA_SCHEMA,
    ArtifactCustody,
    ComponentId,
    CustodyStatus,
    artifact_custody,
    load_metric_custody_bundle,
)
from tac.optimization.ddm_metric_producers import (  # noqa: E402
    BLOCKER_SCHEMA,
    POSE_BLOCK_SCHEMA,
    POSE_TUBE_RADIUS,
    POSE_TUBE_SOURCE,
    PRODUCER_SCHEMA,
    audit_pf2_bucket_assignments,
    measurement_schedule,
    metric_tag,
    non_converged_pose_row,
    padded_batch32,
    pose_quadratic_row,
    validate_hard_pair_schedule,
)
from tac.optimization.ddm_min_description_contract import (  # noqa: E402
    LayerHome,
    StreamType,
)
from tac.optimization.ddm_pf2_bucket_assignment import (  # noqa: E402
    ASSIGNMENT_RECEIPT_SCHEMA,
    canonical_sha256,
    validate_assignment_table,
)
from tac.repo_io import sha256_file  # noqa: E402
from tac.scorer import load_default_scorers  # noqa: E402

RUN_ID: Final = "ddm_ms4_metric_producers_and_measurement_20260724T042005Z"
EXPECTED_PF2_SHA256: Final = "85084f7bd3a03dbd1b9f04fe6a9b84df4948a6caf64620beef42da8924345f73"
EXPECTED_G3_SHA256: Final = "0c9ce6d0ce2b2c0830400f096438355242527d40f682fc1b201f67d8d951a4e4"
EXPECTED_GT_SHA256: Final = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
EXPECTED_POSENET_SHA256: Final = "0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576"
EXPECTED_MODULES_SHA256: Final = "065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa"
MIN_FREE_BYTES: Final = 2 * 1024**3
STAGES: Final = tuple(HARD_PAIR_ORDER)

DEFAULT_PF2 = (
    REPO / ".omx/research/ddm_pf2_dimension_conditioned_two_type_20260724T020205Z/"
    "ddm_pf2_dimension_conditioned_two_type_receipt.json"
)
DEFAULT_G3 = REPO / ".omx/research/ddm_g3_score_atlas_n600_20260722T204000Z/hard_pair_registry.json"
DEFAULT_GT = Path("/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
DEFAULT_UPSTREAM = Path("/Users/adpena/Projects/pact/upstream")
DEFAULT_BULK = Path("/Volumes/VertigoDataTier/pact") / RUN_ID
DEFAULT_RECEIPTS = REPO / ".omx/research" / RUN_ID


class MS4ProducerError(RuntimeError):
    """The producer cannot preserve the delegated measurement contract."""


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise MS4ProducerError(f"expected JSON object: {path}")
    return value


def _verify(path: Path, expected: str, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    observed = sha256_file(resolved)
    if observed != expected:
        raise MS4ProducerError(f"{label} SHA-256 differs: {observed}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": observed}


def _load_assignment_receipt(
    path: Path,
    expected_sha256: str,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Load the MS5 table through both file and embedded content hashes."""

    receipt_custody = _verify(path, expected_sha256, "PF2 assignment receipt")
    receipt = _read_json(path)
    if receipt.get("schema") != ASSIGNMENT_RECEIPT_SCHEMA:
        raise MS4ProducerError("PF2 assignment receipt schema differs")
    payload = dict(receipt)
    claimed_receipt_sha = payload.pop("receipt_content_sha256", None)
    if claimed_receipt_sha != canonical_sha256(payload):
        raise MS4ProducerError("PF2 assignment receipt content SHA differs")
    table_binding = receipt.get("assignment_table")
    if not isinstance(table_binding, Mapping):
        raise MS4ProducerError("PF2 assignment receipt lacks its table binding")
    table_path = Path(str(table_binding.get("path")))
    if not table_path.is_absolute():
        table_path = REPO / table_path
    table_custody = _verify(
        table_path,
        str(table_binding.get("file_sha256")),
        "PF2 assignment table",
    )
    table = _read_json(table_path)
    try:
        validate_assignment_table(table, expected_pf2_sha256=EXPECTED_PF2_SHA256)
    except ValueError as exc:
        raise MS4ProducerError("PF2 assignment table failed strict validation") from exc
    if table.get("table_content_sha256") != table_binding.get("content_sha256"):
        raise MS4ProducerError("PF2 assignment table content binding differs")
    return table, {
        "receipt": receipt_custody,
        "table": table_custody,
    }


def _custody(path: Path, *, role: str, content_schema: str) -> ArtifactCustody:
    return artifact_custody(path, repository_root=REPO, role=role, content_schema=content_schema)


def _storage_preflight(path: Path) -> dict[str, Any]:
    resolved_parent = path.expanduser().resolve().parent
    if not str(resolved_parent).startswith(("/Volumes/VertigoDataTier/pact", "/Volumes/APDataStore/pact")):
        raise MS4ProducerError("bulk output must use the governed SSD waterfall")
    free = shutil.disk_usage(resolved_parent).free
    if free < MIN_FREE_BYTES:
        raise MS4ProducerError(f"storage preflight refused: free={free}, required={MIN_FREE_BYTES}")
    return {
        "status": "PASS",
        "tier": str(resolved_parent),
        "observed_free_bytes": free,
        "required_free_bytes": MIN_FREE_BYTES,
        "auto_cleanup": "NO_EPHEMERAL_BULK_CREATED; immutable JSON checkpoints preserved",
    }


def _attestation(required: bool) -> dict[str, Any]:
    """Require safe_run admission for the long full-n600 phase."""

    governed = os.environ.get("TAC_GOVERNED_ADMISSION") == "1"
    if not governed:
        if required:
            raise MS4ProducerError("full_n600 requires tools/safe_run.py attestation")
        return {"required": False, "present": False}
    return {
        "required": required,
        "present": True,
        "governed_marker": "TAC_GOVERNED_ADMISSION=1",
        "required_launcher": "tools/safe_run.py",
    }


def _torch_setup() -> Any:
    import torch

    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.use_deterministic_algorithms(True)
    return torch


def _pose_forward(posenet: Any, camera_pairs: np.ndarray) -> np.ndarray:
    torch = _torch_setup()
    camera = np.asarray(camera_pairs)
    if camera.dtype != np.uint8 or camera.shape != (SCORER_BATCH_SIZE, 2, 874, 1164, 3):
        raise MS4ProducerError(f"batch32 camera geometry differs: {camera.shape} {camera.dtype}")
    tensor = torch.from_numpy(np.array(camera, copy=True, order="C")).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        output = posenet(posenet.preprocess_input(tensor))
        pose = output["pose"] if isinstance(output, dict) else output
    result = pose[:, :6].cpu().numpy().astype(np.float64)
    if result.shape != (SCORER_BATCH_SIZE, 6) or not np.isfinite(result).all():
        raise MS4ProducerError("PoseNet batch32 result is malformed")
    return np.ascontiguousarray(result)


def _stage_groups(pair_ids: Sequence[int]) -> list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    groups = []
    for start in range(0, len(pair_ids), SCORER_BATCH_SIZE):
        primary = tuple(int(row) for row in pair_ids[start : start + SCORER_BATCH_SIZE])
        batch, padding = padded_batch32(primary)
        groups.append((primary, batch, padding))
    return groups


def run_pose_stage(
    *,
    stage: str,
    pair_ids: Sequence[int],
    checkpoints: Path,
    gt_f0: np.ndarray,
    gt_f1: np.ndarray,
    registered_pose: np.ndarray,
    posenet: Any,
    custody: Mapping[str, Any],
) -> list[Path]:
    """Run or resume one preregistered stage with immutable per-block saves."""

    stage_root = checkpoints / stage
    output: list[Path] = []
    for block_id, (primary, batch_ids, padding) in enumerate(_stage_groups(pair_ids)):
        checkpoint = stage_root / f"block_{block_id:03d}.json"
        if checkpoint.exists():
            value = _read_json(checkpoint)
            if (
                value.get("schema") != POSE_BLOCK_SCHEMA
                or value.get("stage") != stage
                or value.get("primary_pair_ids") != list(primary)
                or value.get("batch_pair_ids") != list(batch_ids)
                or value.get("input_custody") != custody
            ):
                raise MS4ProducerError(f"resume checkpoint differs: {checkpoint}")
            output.append(checkpoint)
            continue
        started = time.monotonic()
        rows: list[dict[str, Any]] = []
        status = "MEASURED"
        failure = None
        try:
            camera = np.stack(
                [
                    np.stack(
                        [
                            np.asarray(gt_f0[pair_id], dtype=np.uint8),
                            np.asarray(gt_f1[pair_id], dtype=np.uint8),
                        ],
                        axis=0,
                    )
                    for pair_id in batch_ids
                ],
                axis=0,
            )
            observed = _pose_forward(posenet, camera)
            by_id = {pair_id: observed[index] for index, pair_id in enumerate(batch_ids)}
            for pair_id in primary:
                registered = np.asarray(registered_pose[pair_id], dtype=np.float64)[:6]
                center = by_id[pair_id]
                rows.append(
                    pose_quadratic_row(
                        pair_id,
                        center,
                        observed_against_registered_center_max_abs=float(np.max(np.abs(center - registered))),
                    )
                )
        except Exception as exc:  # preserve an explicit scientific failure checkpoint
            status = "NON_CONVERGED_SCORER_FORWARD"
            failure = f"{type(exc).__name__}: {exc}"
            rows = [non_converged_pose_row(pair_id, "SCORER_FORWARD") for pair_id in primary]
        payload = {
            "schema": POSE_BLOCK_SCHEMA,
            "run_id": RUN_ID,
            "stage": stage,
            "block_id": block_id,
            "primary_pair_ids": list(primary),
            "batch_pair_ids": list(batch_ids),
            "padding_pair_ids": list(padding),
            "scorer_batch_size": SCORER_BATCH_SIZE,
            "threads": 4,
            "seed": 1234,
            "deterministic_algorithms": True,
            "input_custody": dict(custody),
            "status": status,
            "failure": failure,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rows": rows,
            "evidence_axis": EVIDENCE_AXIS,
            "score_claim": False,
            "research_only": True,
        }
        publish_immutable_json(checkpoint, payload)
        output.append(checkpoint)
    stage_receipt = {
        "schema": PRODUCER_SCHEMA,
        "run_id": RUN_ID,
        "component": ComponentId.POSE_METRIC.value,
        "stage": stage,
        "pair_ids": list(pair_ids),
        "block_checkpoints": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output
        ],
        "per_block_checkpointed_and_preserved": True,
        "resumable_from_disk": True,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "research_only": True,
    }
    publish_immutable_json(stage_root / "STAGE-COMPLETE.json", stage_receipt)
    return output


def _blocker_artifact(
    *,
    path: Path,
    component: ComponentId,
    audit: Any,
    custody: Mapping[str, Any],
) -> Path:
    payload = {
        "schema": BLOCKER_SCHEMA,
        "run_id": RUN_ID,
        "component": component.value,
        "status": "BLOCKED",
        "blocker": "PF2_BUCKET_INPUT_ASSIGNMENT_ABSENT",
        "verdict_scope": (
            "Current PF2 atlas only: semantic keys do not identify pair membership, "
            "receiver actuator, or perturbation direction."
        ),
        "pf2_bucket_count": audit.bucket_count,
        "assigned_bucket_count": audit.assigned_count,
        "unassigned_bucket_count": len(audit.unassigned_bucket_ids),
        "unassigned_bucket_ids_sha256": __import__("hashlib")
        .sha256(("\n".join(audit.unassigned_bucket_ids) + "\n").encode("utf-8"))
        .hexdigest(),
        "input_custody": dict(custody),
        "forbidden_inference": (
            "Do not duplicate one pair-level tensor across the 1,200 PF2 keys and "
            "do not interpret zero content_event_count as an identified zero metric."
        ),
        "next_measurement": (
            "Land a SHA-bound 1:1 PF2 bucket assignment containing pair_ids "
            "0..599, receiver_actuator_id, and direction_id for every bucket."
        ),
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "research_only": True,
        "main_landing_review_required": True,
    }
    publish_immutable_json(path, payload)
    return path


def _component_receipt(
    *,
    component: ComponentId,
    status: CustodyStatus,
    sample_count: int,
    lineage: Sequence[ArtifactCustody],
    data: ArtifactCustody,
    blockers: Sequence[str],
    next_measurement: str,
    tag: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": COMPONENT_SCHEMA,
        "component_id": component.value,
        "status": status.value,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "research_only": True,
        "sample_count": sample_count,
        "scorer_batch_size": SCORER_BATCH_SIZE,
        "input_lineage": [row.to_dict() for row in lineage],
        "data_artifact": data.to_dict(),
        "blockers": list(blockers),
        "next_measurement": next_measurement,
        "typed_stream_tags": [dict(tag)],
        "main_landing_review_required": True,
    }


def materialize_bundle(
    *,
    receipts: Path,
    bulk: Path,
    pf2_path: Path,
    g3_path: Path,
    audit: Any,
    measured_stage: str,
    input_custody: Mapping[str, Any],
) -> Path:
    """Materialize the strict bundle and prove it through the existing loader."""

    pf2_artifact = _custody(pf2_path, role="pf2_typed_atlas", content_schema=PF2_ATLAS_SCHEMA)
    g3_artifact = _custody(g3_path, role="g3_hard_pair_registry", content_schema=G3_REGISTRY_SCHEMA)
    receipts.mkdir(parents=True, exist_ok=True)
    lineage = [pf2_artifact, g3_artifact]
    components: dict[ComponentId, dict[str, Any]] = {}

    blocked_specs = {
        ComponentId.SEG_METRIC: (
            SEG_DATA_SCHEMA,
            StreamType.SKELETON,
            LayerHome.L4_SCORER_FEATURE,
            "Measure full n600 rank-4 margin-Fisher rows after PF2 bucket assignments land.",
        ),
        ComponentId.COMPOSITE_R_SECOND_ORDER: (
            COMPOSITE_R_DATA_SCHEMA,
            StreamType.CONNECTION,
            LayerHome.L4_SCORER_FEATURE,
            "Measure exact full-kernel Hessian/adjoint and paired realized secants after assignments land.",
        ),
        ComponentId.DUAL_METRIC_DIAGNOSTICS: (
            DUAL_DATA_SCHEMA,
            StreamType.RESIDUAL,
            LayerHome.L5_VERDICT,
            "Measure matched Fisher and LABELED_CONTROL_ONLY Euclidean vectors after assignments land.",
        ),
    }
    for component, (_, stream_type, layer, next_measurement) in blocked_specs.items():
        data_path = _blocker_artifact(
            path=receipts / f"{component.value.lower()}_blocker.json",
            component=component,
            audit=audit,
            custody=input_custody,
        )
        components[component] = _component_receipt(
            component=component,
            status=CustodyStatus.PARTIAL,
            sample_count=0,
            lineage=lineage,
            data=_custody(data_path, role=f"{component.value.lower()}_blocker", content_schema=BLOCKER_SCHEMA),
            blockers=["PF2_BUCKET_INPUT_ASSIGNMENT_ABSENT"],
            next_measurement=next_measurement,
            tag=metric_tag(stream_type, layer),
        )

    pose_blocks = sorted((bulk / "checkpoints" / "full_n600").glob("block_*.json"))
    pose_rows: list[Mapping[str, Any]] = []
    for path in pose_blocks:
        pose_rows.extend(_read_json(path).get("rows", []))
    pose_by_id = {int(row["pair_id"]): row for row in pose_rows if isinstance(row, Mapping)}
    pose_complete = measured_stage == "full_n600" and set(pose_by_id) == set(range(PAIR_COUNT))
    if pose_complete:
        pose_data_path = bulk / "pose_metric_n600_batch32.json"
        pose_data = {
            "schema": POSE_DATA_SCHEMA,
            "pf2_atlas_sha256": EXPECTED_PF2_SHA256,
            "g3_hard_pair_registry_sha256": EXPECTED_G3_SHA256,
            "measurement_schedule": measurement_schedule(),
            "pair_count": PAIR_COUNT,
            "scorer_batch_size": SCORER_BATCH_SIZE,
            "output_dimension": 6,
            "metric_surface": "EXACT_POSENET_OUTPUT_MSE_QUADRATIC",
            "quadratic_identity": "||I/sqrt(6) delta||_2^2 = mean(delta^2)",
            "tube_radius": POSE_TUBE_RADIUS,
            "tube_radius_source": POSE_TUBE_SOURCE,
            "rows": [pose_by_id[pair_id] for pair_id in range(PAIR_COUNT)],
            "evidence_axis": EVIDENCE_AXIS,
            "score_claim": False,
            "research_only": True,
        }
        publish_immutable_json(pose_data_path, pose_data)
        pose_status = CustodyStatus.COMPLETE
        pose_blockers: list[str] = []
        pose_next = "Consumer may use Pose output quadratic only after MAIN landing review."
        pose_sample_count = PAIR_COUNT
        pose_schema = POSE_DATA_SCHEMA
    else:
        stage_rows: list[Mapping[str, Any]] = []
        for path in sorted((bulk / "checkpoints" / measured_stage).glob("block_*.json")):
            stage_rows.extend(_read_json(path).get("rows", []))
        pose_data_path = receipts / "pose_metric_partial_measurement.json"
        pose_partial = {
            "schema": PRODUCER_SCHEMA,
            "run_id": RUN_ID,
            "component": ComponentId.POSE_METRIC.value,
            "measured_stage": measured_stage,
            "measured_pair_count": len(stage_rows),
            "rows": stage_rows,
            "input_custody": dict(input_custody),
            "blocker": "FULL_N600_BATCH32_POSE_NOT_YET_MEASURED",
            "evidence_axis": EVIDENCE_AXIS,
            "score_claim": False,
            "research_only": True,
        }
        publish_immutable_json(pose_data_path, pose_partial)
        pose_status = CustodyStatus.PARTIAL
        pose_blockers = ["FULL_N600_BATCH32_POSE_NOT_YET_MEASURED"]
        pose_next = "Resume governed stages through full_n600; retain all immutable block checkpoints."
        pose_sample_count = len(stage_rows)
        pose_schema = PRODUCER_SCHEMA
    components[ComponentId.POSE_METRIC] = _component_receipt(
        component=ComponentId.POSE_METRIC,
        status=pose_status,
        sample_count=pose_sample_count,
        lineage=lineage,
        data=_custody(pose_data_path, role="pose_metric_data", content_schema=pose_schema),
        blockers=pose_blockers,
        next_measurement=pose_next,
        tag=metric_tag(StreamType.FIBER, LayerHome.L5_VERDICT),
    )

    component_refs: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    for component in ComponentId:
        path = receipts / f"{component.value.lower()}_receipt.json"
        publish_immutable_json(path, components[component])
        component_refs[component.value] = _custody(
            path,
            role=f"{component.value.lower()}_component_receipt",
            content_schema=COMPONENT_SCHEMA,
        ).to_dict()
        blockers.extend(components[component]["blockers"])

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "bundle_id": RUN_ID,
        "status": CustodyStatus.PARTIAL.value,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "research_only": True,
        "pointer": POINTER,
        "pointer_moved": False,
        "pf2_atlas": pf2_artifact.to_dict(),
        "g3_hard_pair_registry": g3_artifact.to_dict(),
        "component_receipts": component_refs,
        "hard_pair_order": list(HARD_PAIR_ORDER),
        "consumers": [
            "ms2_typed_quotient_solve",
            "pf2r_metric_active_three_formulation",
            "rd1_dimension_duals",
        ],
        "blockers": list(dict.fromkeys(blockers)),
        "headline_admissibility": {
            "bundle_complete": False,
            "scorer_metric_active": False,
            "pose_tube_active": False,
            "score_claim": False,
        },
        "main_landing_review_required": True,
    }
    bundle_path = receipts / "BUNDLE-PARTIAL.json"
    publish_immutable_json(bundle_path, manifest)
    loaded = load_metric_custody_bundle(bundle_path, repository_root=REPO)
    if loaded.complete or loaded.headline_flags()["scorer_metric_active"]:
        raise MS4ProducerError("strict loader unexpectedly activated a partial bundle")
    return bundle_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--through-stage", choices=STAGES, default="top24")
    parser.add_argument("--pf2", type=Path, default=DEFAULT_PF2)
    parser.add_argument("--pf2-assignment-receipt", type=Path)
    parser.add_argument("--pf2-assignment-receipt-sha256")
    parser.add_argument("--g3", type=Path, default=DEFAULT_G3)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_GT)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--bulk-output", type=Path, default=DEFAULT_BULK)
    parser.add_argument("--receipt-output", type=Path, default=DEFAULT_RECEIPTS)
    args = parser.parse_args()
    if (args.pf2_assignment_receipt is None) != (args.pf2_assignment_receipt_sha256 is None):
        parser.error("--pf2-assignment-receipt and --pf2-assignment-receipt-sha256 must be supplied together")

    storage = _storage_preflight(args.bulk_output)
    args.bulk_output.mkdir(parents=True, exist_ok=True)
    args.receipt_output.mkdir(parents=True, exist_ok=True)
    attestation = _attestation(args.through_stage == "full_n600")
    source_custody = {
        "pf2": _verify(args.pf2, EXPECTED_PF2_SHA256, "PF2"),
        "g3": _verify(args.g3, EXPECTED_G3_SHA256, "G3"),
        "gt_cache": _verify(args.gt_cache, EXPECTED_GT_SHA256, "GT cache"),
        "posenet_weights": _verify(
            args.upstream / "models/posenet.safetensors",
            EXPECTED_POSENET_SHA256,
            "PoseNet weights",
        ),
        "upstream_modules": _verify(args.upstream / "modules.py", EXPECTED_MODULES_SHA256, "upstream modules"),
    }
    assignment_table = None
    if args.pf2_assignment_receipt is not None:
        assignment_table, assignment_custody = _load_assignment_receipt(
            args.pf2_assignment_receipt,
            args.pf2_assignment_receipt_sha256,
        )
        source_custody["pf2_assignment"] = assignment_custody
    execution_contract = {
        "schema": "ddm_ms4_execution_contract.v1",
        "run_id": RUN_ID,
        "storage": {
            "status": storage["status"],
            "tier": storage["tier"],
            "required_free_bytes": storage["required_free_bytes"],
            "auto_cleanup": storage["auto_cleanup"],
        },
        "governed_admission": attestation,
        "threads": 4,
        "seed": 1234,
        "score_claim": False,
        "research_only": True,
    }
    publish_immutable_json(args.bulk_output / f"execution_contract_{args.through_stage}.json", execution_contract)
    pf2 = _read_json(args.pf2)
    g3 = _read_json(args.g3)
    audit = audit_pf2_bucket_assignments(pf2, assignment_table)
    schedule = validate_hard_pair_schedule(g3)

    gt_f0 = open_stored_npy_memmap(args.gt_cache, "gt_f0")
    gt_f1 = open_stored_npy_memmap(args.gt_cache, "gt_f1")
    registered_pose = open_stored_npy_memmap(args.gt_cache, "gt_poses")
    posenet, _ = load_default_scorers(args.upstream, device="cpu")

    for stage in STAGES[: STAGES.index(args.through_stage) + 1]:
        run_pose_stage(
            stage=stage,
            pair_ids=schedule[stage],
            checkpoints=args.bulk_output / "checkpoints",
            gt_f0=gt_f0,
            gt_f1=gt_f1,
            registered_pose=registered_pose,
            posenet=posenet,
            custody=source_custody,
        )
    bundle = materialize_bundle(
        receipts=args.receipt_output / args.through_stage,
        bulk=args.bulk_output,
        pf2_path=args.pf2,
        g3_path=args.g3,
        audit=audit,
        measured_stage=args.through_stage,
        input_custody=source_custody,
    )
    print(bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
