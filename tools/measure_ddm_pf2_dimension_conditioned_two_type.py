#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure the DDM PF2 dimension-conditioned two-type formulations at n600."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_dimension_conditioned_two_type import (  # noqa: E402
    CLASS_NAMES,
    CLASS_STRATA,
    EVENT_SENTINEL,
    IDENTICAL_CONTENT_CODER_CONTROL,
    IDENTITY_EUCLIDEAN_CONTROL,
    REPRESENTATION_TYPES,
    TEMPORAL_CLASSES,
    VISIBILITY_CLASSES,
    moment_constrained_hood_projection,
    race_event_coders,
    resolve_formulation_metric_disposition,
    support_rgb_moments,
)
from tac.optimization.ddm_g3_score_atlas import reconstruct_v12_state  # noqa: E402
from tac.optimization.ddm_g4_spatial_stationarity import (  # noqa: E402
    boundary_mask,
    transition_codes,
)
from tac.optimization.direct_description_preuint8_channel import (  # noqa: E402
    receive_preuint8_q8_archive,
)
from tac.witness_dsl.bregman_dual_metric_guard import (  # noqa: E402
    canonical_bregman_dual_metric_binding,
    resolve_bregman_dual_metric_adoption,
)

SCHEMA = "ddm_pf2_dimension_conditioned_two_type_measurement.v1"
CONFIG_SCHEMA = "DDMPF2DimensionConditionedTwoTypeConfigV1"
LANE_ID = "lane_ddm_pf2_dimension_conditioned_two_type_20260724"
DELEGATION_KEY = (
    "codex_delegate:ddm_pf2_dimension_conditioned_two_type:20260724T020205Z"
)
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
POINTER = "0.1910828242 [contest-CPU]"
PARENT_ID = "statistics_hard_analytic_composed_frame1"
BASE_ID = "v19c_base"
POSE_CANDIDATE_ID = "pf2_pose_stat_moment_alpha_0p75_frame1"
RATE_PER_BYTE = 25.0 / 37_545_489.0


class PF2Error(RuntimeError):
    """Fail-closed PF2 measurement error."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_regular(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise PF2Error(f"regular non-symlink file required: {path}")
    return path.read_bytes()


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _bound(path: str, expected: str, label: str) -> bytes:
    source = _resolve(path)
    payload = _read_regular(source)
    actual = sha256_bytes(payload)
    if actual != expected:
        raise PF2Error(f"{label} sha256 differs: {actual} != {expected}")
    return payload


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_regular(path) != payload:
            raise PF2Error(f"immutable output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _publish_json(path: Path, value: Any) -> None:
    _publish(path, _canonical(value))


def _save_arrays(path: Path, *, cells: np.ndarray, pose6: np.ndarray) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if not np.array_equal(stored["cells"], cells) or not np.array_equal(
                stored["pose6"], pose6
            ):
                raise PF2Error(f"immutable array checkpoint differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary, cells=cells, pose6=pose6)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _import_tool(name: str, filename: str) -> Any:
    source = REPO_ROOT / "tools" / filename
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise PF2Error(f"import spec absent for {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PF2Config(BaseModel):
    """Strict local-only PF2 authority and source-custody contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["DDMPF2DimensionConditionedTwoTypeConfigV1"] = Field(
        CONFIG_SCHEMA, alias="schema", serialization_alias="schema"
    )
    run_id: str = Field(min_length=12)
    seed: Literal[210] = 210
    pair_count: Literal[600] = 600
    scorer_batch_size: Literal[16] = 16
    scorer_threads: Literal[4] = 4
    pose_projection_alpha: Literal[0.75] = 0.75
    mc1_config_path: str
    mc1_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mc1_receipt_path: str
    mc1_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pf1_receipt_path: str
    pf1_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    g4_receipt_path: str
    g4_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v12_receipt_path: str
    v12_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    xi_fiber_path: str
    xi_fiber_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    xi_fiber_bytes: Literal[44399] = 44_399
    identity_fiber_path: str
    identity_fiber_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity_fiber_bytes: Literal[42917] = 42_917
    xi_574_receipt_path: str
    xi_574_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_root: str
    pointer: Literal["0.1910828242 [contest-CPU]"] = POINTER
    execution_allowed: Literal[True] = True
    paid_dispatch_allowed: Literal[False] = False
    exact_eval_allowed: Literal[False] = False
    frontier_mutation_allowed: Literal[False] = False
    training_allowed: Literal[False] = False
    research_only: Literal[True] = True
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _ssd_and_gates(self) -> PF2Config:
        if not Path(self.checkpoint_root).is_absolute():
            raise ValueError("checkpoint_root must be absolute")
        if not self.checkpoint_root.startswith("/Volumes/VertigoDataTier/pact/"):
            raise ValueError("checkpoint_root must use the primary SSD tier")
        return self

    def stable_hash(self) -> str:
        return sha256_bytes(
            json.dumps(
                self.model_dump(mode="json", by_alias=True),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )


def _batch_paths(root: Path, start: int, stop: int) -> tuple[Path, Path]:
    stage = root / "02_pose_projection" / POSE_CANDIDATE_ID
    return (
        stage / f"batch_{start:04d}_{stop:04d}.json",
        stage / f"batch_{start:04d}_{stop:04d}.npz",
    )


def _menu_batch_paths(
    root: Path, candidate_id: str, start: int, stop: int
) -> tuple[Path, Path]:
    stage = root / "02_measurements" / candidate_id
    return (
        stage / f"batch_{start:04d}_{stop:04d}.json",
        stage / f"batch_{start:04d}_{stop:04d}.npz",
    )


def _official_pose_input_telemetry(
    posenet: Any, parent: np.ndarray, child: np.ndarray
) -> dict[str, Any]:
    import torch

    def _preprocess(camera: np.ndarray) -> Any:
        tensor = (
            torch.from_numpy(np.ascontiguousarray(camera))
            .permute(0, 1, 4, 2, 3)
            .contiguous()
            .float()
        )
        return posenet.preprocess_input(tensor).cpu()

    with torch.inference_mode():
        before = _preprocess(parent)
        after = _preprocess(child)
    delta = (after - before).abs()
    spatial = (0, 2, 3)
    return {
        "official_preprocess_shape": list(before.shape),
        "official_posenet_input_changed_coordinates": int(torch.count_nonzero(delta)),
        "official_posenet_input_l1": float(delta.sum(dtype=torch.float64)),
        "official_posenet_input_linf": float(delta.max()),
        "parent_channel_sum": [
            float(v) for v in before.sum(dim=spatial, dtype=torch.float64)
        ],
        "child_channel_sum": [
            float(v) for v in after.sum(dim=spatial, dtype=torch.float64)
        ],
        "parent_channel_sumsq": [
            float(v)
            for v in torch.square(before.to(dtype=torch.float64)).sum(dim=spatial)
        ],
        "child_channel_sumsq": [
            float(v)
            for v in torch.square(after.to(dtype=torch.float64)).sum(dim=spatial)
        ],
        "samples_per_channel": int(before.shape[0] * before.shape[2] * before.shape[3]),
    }


def _sum_vectors(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [
        math.fsum(float(row[key][index]) for row in rows)
        for index in range(len(rows[0][key]))
    ]


def _measure_pose_projection(
    *,
    config: PF2Config,
    menu1: Any,
    mc1: Any,
    menu_config: Any,
    receiver: Any,
    palette: np.ndarray,
    statistics_payload: bytes,
    labels: np.ndarray,
    poses: np.ndarray,
    segnet: Any,
    posenet: Any,
    supports: Any,
    support_counted_bytes: int,
    alpha_counted_bytes: int,
    root: Path,
    menu_root: Path,
    parent: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for start in range(0, config.pair_count, config.scorer_batch_size):
        stop = min(start + config.scorer_batch_size, config.pair_count)
        row_path, array_path = _batch_paths(root, start, stop)
        if row_path.exists() and array_path.exists():
            row = json.loads(_read_regular(row_path))
            if (
                row.get("typed_config_sha256") != config.stable_hash()
                or row.get("candidate_id") != POSE_CANDIDATE_ID
            ):
                raise PF2Error("pose projection resume identity differs")
            rows.append(row)
            continue
        ids = tuple(range(start, stop))
        base_camera = receiver.render_camera_pairs(ids)
        semantic, owned = menu1._semantic_cells(
            receiver, ids, base_camera, palette
        )
        winner = menu1._geometry_statistics_camera(
            base_camera=base_camera,
            semantic=semantic,
            owned=owned,
            palette=palette,
            statistics_payload=statistics_payload,
        )
        parent_row_path, parent_arrays_path = _menu_batch_paths(
            menu_root, PARENT_ID, start, stop
        )
        parent_row = json.loads(_read_regular(parent_row_path))
        if sha256_bytes(winner.tobytes()) != parent_row["camera_sha256"]:
            raise PF2Error("fresh MENU1 parent differs from preserved checkpoint")
        scorer_support = np.broadcast_to(
            supports.static, (stop - start, *supports.static.shape)
        )
        camera_support = mc1.expand_support_to_camera(
            scorer_support,
            batch_size=stop - start,
            camera_hw=menu1.CAMERA_HW,
        )
        camera = moment_constrained_hood_projection(
            base_camera=base_camera,
            winner_camera=winner,
            camera_support=camera_support,
            alpha=config.pose_projection_alpha,
        )
        cells, pose6 = menu1._forward(segnet, posenet, camera)
        if start == 0:
            replay_cells, replay_pose6 = menu1._forward(segnet, posenet, camera)
            if not np.array_equal(cells, replay_cells) or not np.array_equal(
                pose6, replay_pose6
            ):
                raise PF2Error("first pose-projection batch replay differs")
        with np.load(parent_arrays_path, allow_pickle=False) as stored:
            parent_cells = np.asarray(stored["cells"], dtype=np.uint8)
        target = np.asarray(labels[start:stop], dtype=np.uint8)
        target_pose = np.asarray(poses[start:stop], dtype=np.float64)
        transition = menu1.transition_counts(
            before=parent_cells, after=cells, target=target
        )
        per_class = mc1.class_transition_rows(
            before=parent_cells,
            after=cells,
            target=target,
            class_names=menu1.CLASS_NAMES,
        )
        official = _official_pose_input_telemetry(posenet, winner, camera)
        parent_moments = support_rgb_moments(winner, camera_support)
        child_moments = support_rgb_moments(camera, camera_support)
        changed = np.any(camera[:, 1] != winner[:, 1], axis=-1)
        row = {
            "schema": "ddm_pf2_pose_projection_batch.v1",
            "typed_config_sha256": config.stable_hash(),
            "candidate_id": POSE_CANDIDATE_ID,
            "pair_range": [start, stop],
            "errors": int(np.count_nonzero(cells != target)),
            "sites": int(cells.size),
            "pose_squared_error_sum": float(
                np.square(pose6 - target_pose).sum(dtype=np.float64)
            ),
            "pose_coordinates": int(pose6.size),
            "transition_from_parent": transition,
            "per_class": per_class,
            "camera_pixels_changed_vs_parent": int(np.count_nonzero(changed)),
            "frame0_byte_identical": bool(
                np.array_equal(camera[:, 0], base_camera[:, 0])
            ),
            "outside_support_byte_identical": bool(
                np.array_equal(camera[:, 1][~camera_support], winner[:, 1][~camera_support])
            ),
            "support_rgb_moment_max_abs_delta": float(
                np.max(np.abs(child_moments - parent_moments))
            ),
            **official,
            "camera_sha256": sha256_bytes(camera.tobytes()),
            "cells_sha256": sha256_bytes(cells.tobytes()),
            "pose6_sha256": sha256_bytes(pose6.tobytes()),
            "evidence_axis": EVIDENCE_AXIS,
            "research_only": True,
            "score_claim": False,
        }
        _save_arrays(array_path, cells=cells, pose6=pose6)
        _publish_json(row_path, row)
        rows.append(row)
        print(f"[PF2 pose] {start:04d}:{stop:04d}", flush=True)
    errors = sum(int(row["errors"]) for row in rows)
    sites = sum(int(row["sites"]) for row in rows)
    pose_sse = math.fsum(float(row["pose_squared_error_sum"]) for row in rows)
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in rows)
    d_pose = pose_sse / pose_coordinates
    projection_counted_bytes = support_counted_bytes + alpha_counted_bytes
    archive_bytes = int(parent["archive_bytes"]) + projection_counted_bytes
    parent_objective = menu1.advisory_objective(
        errors=round(float(parent["d_seg"]) * sites),
        sites=sites,
        d_pose=float(parent["d_pose"]),
        bytes_=int(parent["archive_bytes"]),
    )
    child_objective = menu1.advisory_objective(
        errors=errors,
        sites=sites,
        d_pose=d_pose,
        bytes_=archive_bytes,
    )
    per_class = mc1._sum_nested_class_rows(rows)
    parent_sum = _sum_vectors(rows, "parent_channel_sum")
    child_sum = _sum_vectors(rows, "child_channel_sum")
    parent_sumsq = _sum_vectors(rows, "parent_channel_sumsq")
    child_sumsq = _sum_vectors(rows, "child_channel_sumsq")
    samples = sum(int(row["samples_per_channel"]) for row in rows)
    parent_mean = np.asarray(parent_sum) / samples
    child_mean = np.asarray(child_sum) / samples
    parent_var = np.asarray(parent_sumsq) / samples - np.square(parent_mean)
    child_var = np.asarray(child_sumsq) / samples - np.square(child_mean)
    return {
        "formulation_id": "F2_POSE_STAT_PRESERVING_HOOD_PROJECTION",
        "candidate_id": POSE_CANDIDATE_ID,
        "parent_candidate_id": PARENT_ID,
        "projection": {
            "method": (
                "per-pair per-RGB-channel alpha move toward V19C base hood, "
                "renormalized to MENU1 winner support mean/std before uint8"
            ),
            "geometry": IDENTITY_EUCLIDEAN_CONTROL,
            "alpha": config.pose_projection_alpha,
            "support": "MC1 single-static majority hood support",
            "support_counted_bytes": support_counted_bytes,
            "alpha_counted_bytes": alpha_counted_bytes,
            "alpha_wire": "one uint8 q4 numerator; decoder divides by 4",
            "generic_projection_code": "FREE_rule118",
            "preselection_scope": (
                "fixed after one bounded first-batch design screen; all acceptance "
                "and claims use only the exact n600 row"
            ),
        },
        "archive_bytes": archive_bytes,
        "delta_counted_bytes_vs_parent": projection_counted_bytes,
        "byte_partition": {
            "COUNTED": projection_counted_bytes,
            "FREE": 0,
            "NULL": 0,
            "FREE_source": "generic moment-projection algorithm only",
        },
        "errors": errors,
        "sites": sites,
        "d_seg": errors / sites,
        "d_pose": d_pose,
        "advisory_objective": child_objective,
        "delta_advisory_objective_vs_parent": child_objective - parent_objective,
        "raw_control_improves_joint_objective": child_objective < parent_objective,
        "accepted": False,
        "metric_status": IDENTITY_EUCLIDEAN_CONTROL,
        "instance_scoped_naive_control": True,
        "verdict_eligible": False,
        "waterfill_eligible": False,
        "acceptance_law": (
            "identity/Euclidean controls cannot be accepted or routed; exact "
            "n600 joint Delta S is retained only as a control readback"
        ),
        "metric_active_rerun_blocker": {
            "status": "BLOCKED_MISSING_MEASURED_SCORER_GEOMETRY",
            "pose": (
                "per-pair PoseNet-6 Jacobian/Hessian or equivalent exact output "
                "quadratic for the hood basis is absent from the sealed inputs"
            ),
            "seg": (
                "bucket-complete rank-4 winner/rival margin-Fisher field for this "
                "camera perturbation is absent from the sealed inputs"
            ),
            "dual_metric": (
                "Euclidean-vs-Fisher cosine and relative-norm readback is absent"
            ),
            "reactivation": (
                "rerun formulation 2 from the measured pose quadratic/Hessian, "
                "rank-4 margin-Fisher coordinates, Bregman binding, and dual "
                "readback before any verdict or waterfill use"
            ),
        },
        "per_class": per_class,
        "frame0_byte_identical": all(row["frame0_byte_identical"] for row in rows),
        "outside_support_byte_identical": all(
            row["outside_support_byte_identical"] for row in rows
        ),
        "camera_pixels_changed_vs_parent": sum(
            int(row["camera_pixels_changed_vs_parent"]) for row in rows
        ),
        "support_rgb_moment_max_abs_delta": max(
            float(row["support_rgb_moment_max_abs_delta"]) for row in rows
        ),
        "official_preprocess": {
            "path": (
                "bilinear RGB resize to 512x384 -> rgb_to_yuv6 -> "
                "two-frame 12-channel FastViT input"
            ),
            "parent_channel_mean": parent_mean.tolist(),
            "child_channel_mean": child_mean.tolist(),
            "max_abs_channel_mean_delta": float(
                np.max(np.abs(child_mean - parent_mean))
            ),
            "parent_channel_std": np.sqrt(np.maximum(parent_var, 0.0)).tolist(),
            "child_channel_std": np.sqrt(np.maximum(child_var, 0.0)).tolist(),
            "max_abs_channel_std_delta": float(
                np.max(
                    np.abs(
                        np.sqrt(np.maximum(child_var, 0.0))
                        - np.sqrt(np.maximum(parent_var, 0.0))
                    )
                )
            ),
            "changed_coordinates": sum(
                int(row["official_posenet_input_changed_coordinates"])
                for row in rows
            ),
            "l1": math.fsum(
                float(row["official_posenet_input_l1"]) for row in rows
            ),
            "linf": max(float(row["official_posenet_input_linf"]) for row in rows),
        },
        "batch_count": len(rows),
        "all_batches_checkpointed_and_preserved": True,
        "batch_digest_chain_sha256": sha256_bytes(
            "".join(row["cells_sha256"] + row["pose6_sha256"] for row in rows).encode()
        ),
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
    }


def _artifact(path: Path, payload: bytes) -> dict[str, Any]:
    _publish(path, payload)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _rate_row(
    *,
    bucket_id: str,
    event_codes: np.ndarray,
    root: Path,
) -> dict[str, Any]:
    event_count = int(np.count_nonzero(event_codes != EVENT_SENTINEL))
    if event_count == 0:
        return {
            "bucket_id": bucket_id,
            "content_event_count": 0,
            "program_counted_bytes": 0,
            "flat_counted_bytes": 0,
            "delta_program_minus_flat_bytes": 0,
            "winner": "EMPTY",
            "identical_content_parseback": True,
            "train_decision": "NO_CONTENT",
            "coder_control_decision": "NO_CONTENT",
            "metric_status": IDENTICAL_CONTENT_CODER_CONTROL,
            "verdict_eligible": True,
            "waterfill_eligible": False,
        }
    race = race_event_coders(event_codes)
    artifact_root = root / "03_bucket_rate_races"
    program = _artifact(
        artifact_root / f"{bucket_id}.event_skeleton.counted",
        race.program_coded.payload,
    )
    flat = _artifact(
        artifact_root / f"{bucket_id}.flat_native.counted",
        race.flat_coded.payload,
    )
    delta = race.delta_program_minus_flat_bytes
    disposition = resolve_formulation_metric_disposition(
        IDENTICAL_CONTENT_CODER_CONTROL,
        identical_content_proven=True,
    )
    return {
        "bucket_id": bucket_id,
        "content_event_count": event_count,
        "program_raw_bytes": len(race.program_raw),
        "program_counted_bytes": program["bytes"],
        "program_codec": race.program_coded.codec,
        "program_artifact": program,
        "flat_raw_bytes": len(race.flat_raw),
        "flat_counted_bytes": flat["bytes"],
        "flat_codec": race.flat_coded.codec,
        "flat_artifact": flat,
        "delta_program_minus_flat_bytes": delta,
        "winner": "SKELETON_TOKENIZED" if delta < 0 else "FIBER_NATIVE_REAL",
        "identical_content_parseback": True,
        "coder_control_decision": (
            "SKELETON_TOKENIZED" if delta < 0 else "FIBER_NATIVE_REAL"
        ),
        "train_decision": "HOLD_METRIC_ACTIVE_TOLERANCE_PRICING_OWED",
        "metric_status": disposition.metric_status,
        "verdict_eligible": disposition.verdict_eligible,
        "waterfill_eligible": disposition.waterfill_eligible,
        "metric_scope": disposition.reason,
    }


def _dr2b_metric_reference(dr2b_receipt: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract settled margin/contest-unit rungs without inventing lambda."""

    rows: list[dict[str, Any]] = []
    source = dr2b_receipt["u1_lossy_tolerance_ladder"][
        "e2_semantic_boundary_samples"
    ]
    for row in source:
        fisher = row.get("fisher_margin")
        n600 = row.get("n600_rebase")
        if (
            row.get("measurement_status") != "MEASURED"
            or not isinstance(fisher, dict)
            or not isinstance(n600, dict)
        ):
            continue
        rows.append(
            {
                "probe_id": row["probe_id"],
                "metric": fisher["metric"],
                "top1_class": fisher["top1_class"],
                "top2_class": fisher["top2_class"],
                "margin": fisher["margin"],
                "head_normal_norm": fisher["head_normal_norm"],
                "flip_distance": fisher["flip_distance"],
                "delta_bytes": n600["delta_bytes"],
                "delta_d_seg": n600["delta_d_seg"],
                "delta_d_pose": n600["delta_d_pose"],
                "joint_delta_contest_units": n600["joint_delta"],
                "lambda": None,
                "lambda_status": (
                    "OWED_NO_CANONICAL_LAMBDA_IN_SETTLED_DR2B_ROW"
                ),
                "epistemic_status": row["epistemic_status"],
            }
        )
    if not rows:
        raise PF2Error("DR2b has no measured margin/contest-unit reference rows")
    return rows


def _measure_atlas_and_event_formulation(
    *,
    config: PF2Config,
    g4_receipt: dict[str, Any],
    v12_receipt: dict[str, Any],
    xi_fiber: bytes,
    identity_fiber: bytes,
    xi_574_receipt: dict[str, Any],
    dr2b_receipt: dict[str, Any],
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    recurrence_row = next(
        row
        for row in g4_receipt["outputs"]
        if row["path"].endswith("01_recurrence_arrays.npz")
    )
    recurrence_path = Path(recurrence_row["path"])
    if (
        recurrence_path.stat().st_size != int(recurrence_row["bytes"])
        or _sha256_file(recurrence_path) != recurrence_row["sha256"]
    ):
        raise PF2Error("G4 recurrence-array custody differs")
    with np.load(recurrence_path, allow_pickle=False) as stored:
        transition_counts = np.asarray(stored["transition_counts"], dtype=np.uint16)
    state = reconstruct_v12_state(REPO_ROOT, v12_receipt, n_pairs=600)
    target_path = Path(v12_receipt["target_custody"]["cache_path"])
    if target_path.stat().st_size != int(v12_receipt["target_custody"]["cache_bytes"]):
        raise PF2Error("V12 target cache byte custody differs")
    target = np.asarray(open_stored_npy_memmap(target_path, "lstars"), dtype=np.uint8)
    predicted = np.asarray(state.final_cells, dtype=np.uint8)
    if target.shape != (600, 384, 512) or predicted.shape != target.shape:
        raise PF2Error("G4/V12 n600 cell geometry differs")
    codes = transition_codes(predicted, target)
    flip = predicted != target
    events = np.where(flip, codes, EVENT_SENTINEL).astype(np.uint8)
    if int(np.count_nonzero(flip)) != int(
        g4_receipt["summary"]["concentration"]["total_flip_mass"]
    ):
        raise PF2Error("G4 event mass differs")
    rows_index = np.arange(384)[:, None]
    cols_index = np.arange(512)[None, :]
    static_image = np.zeros_like(flip)
    for pair_index in range(600):
        static_image[pair_index] = flip[pair_index] & (
            transition_counts[
                codes[pair_index], rows_index, cols_index
            ]
            >= 2
        )
    xi_proxy = np.zeros_like(flip)
    track_row = next(
        row
        for row in g4_receipt["outputs"]
        if row["path"].endswith("xi_proxy_tracks.jsonl")
    )
    track_path = Path(track_row["path"])
    if (
        track_path.stat().st_size != int(track_row["bytes"])
        or _sha256_file(track_path) != track_row["sha256"]
    ):
        raise PF2Error("G4 xi-track custody differs")
    for line in _read_regular(track_path).splitlines():
        track = json.loads(line)
        for event_id in track["event_ids"]:
            pair_index, pixel = divmod(int(event_id), 384 * 512)
            row, col = divmod(pixel, 512)
            xi_proxy[pair_index, row, col] = True
    xi_proxy &= flip & ~static_image
    transient = flip & ~static_image & ~xi_proxy
    temporal_masks = {
        "STATIC_IN_IMAGE": static_image,
        "STATIC_IN_XI_PROXY": xi_proxy,
        "TRANSIENT": transient,
    }
    boundary = np.stack([boundary_mask(row) for row in target], axis=0)
    stratum_masks = {"cell": ~boundary, "boundary": boundary}
    class_pairs = [
        (left, right)
        for left in range(len(CLASS_NAMES))
        for right in range(left + 1, len(CLASS_NAMES))
    ]
    pair_rate_rows: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    pair_mass_rows: list[dict[str, Any]] = []
    for left, right in class_pairs:
        pair_mask = (
            ((predicted == left) & (target == right))
            | ((predicted == right) & (target == left))
        )
        temporal_mass: dict[str, int] = {}
        for temporal_class in TEMPORAL_CLASSES:
            temporal_mass[temporal_class] = int(
                np.count_nonzero(pair_mask & temporal_masks[temporal_class])
            )
        pair_mass = int(np.count_nonzero(pair_mask))
        pair_mass_rows.append(
            {
                "pair_id": f"{CLASS_NAMES[left]}--{CLASS_NAMES[right]}",
                "class_ids": [left, right],
                "flip_mass": pair_mass,
                "flip_mass_fraction": pair_mass / int(np.count_nonzero(flip)),
                "temporal_mass": temporal_mass,
                "rank4_hyperplane": True,
                "sigma_cc_prime": "per-pair Gamma-limit authority",
                "lane_head_normal_anchor": (
                    "3.75-4.01 largest"
                    if 1 in (left, right)
                    else "not the Lane-normal anchor"
                ),
            }
        )
        for stratum in CLASS_STRATA:
            for temporal_class in TEMPORAL_CLASSES:
                membership = (
                    pair_mask
                    & stratum_masks[stratum]
                    & temporal_masks[temporal_class]
                )
                bucket_id = (
                    f"{CLASS_NAMES[left].lower()}_{CLASS_NAMES[right].lower()}"
                    f"__{stratum}__{temporal_class.lower()}"
                )
                bucket_events = np.where(
                    membership, events, EVENT_SENTINEL
                ).astype(np.uint8)
                pair_rate_rows[(left, right, stratum, temporal_class)] = _rate_row(
                    bucket_id=bucket_id,
                    event_codes=bucket_events,
                    root=root,
                )

    atlas_rows: list[dict[str, Any]] = []
    representation_contract = {
        "SKELETON": (
            "discrete class adjacency, event topology, separatrix membership; COUNTED"
        ),
        "CONNECTION": (
            "generic xi/se(3)/homography/static-BEV transport is FREE rule-118; "
            "video-derived parameters and exceptions are COUNTED"
        ),
        "FIBER": (
            "continuous coefficients are COUNTED as deltas in the transported frame, "
            "not independent symbolic slots"
        ),
        "GAUGE": (
            "receiver-canonical ker(A), within-cell, or pose-null slack is zero-byte; "
            "fiber/gauge membership is lambda- and DR2b-tolerance-dependent"
        ),
        "RESIDUAL": (
            "COUNTED exceptions against decoder-derived G4 context"
        ),
    }
    for left, right in class_pairs:
        pair_id = f"{CLASS_NAMES[left]}--{CLASS_NAMES[right]}"
        connection_applicability = (
            "SHARED_LANE_STREAM_EXACT_GLOBAL_RACE_NOT_UNIQUE_HOME"
            if 1 in (left, right)
            else "NO_MATCHED_CONNECTION_FIBER_SOURCE_FOR_THIS_PAIR"
        )
        for stratum in CLASS_STRATA:
            for visibility in VISIBILITY_CLASSES:
                for temporal_class in TEMPORAL_CLASSES:
                    measured_rate = pair_rate_rows[
                        (left, right, stratum, temporal_class)
                    ]
                    for representation_type in REPRESENTATION_TYPES:
                        owns_measured_skeleton = (
                            representation_type == "SKELETON"
                            and visibility == "seg-visible"
                        )
                        if owns_measured_skeleton:
                            rate = measured_rate
                            occupancy = "MEASURED_EXACT_G4_DISCRETE_SKELETON"
                        else:
                            rate = {
                                "bucket_id": (
                                    f"{measured_rate['bucket_id']}__"
                                    f"{visibility.replace('(','').replace(')','').replace('-','_')}"
                                    f"__{representation_type.lower()}"
                                ),
                                "content_event_count": 0,
                                "program_counted_bytes": 0,
                                "flat_counted_bytes": 0,
                                "delta_program_minus_flat_bytes": 0,
                                "winner": "NO_PF2_CONTENT_ASSIGNED",
                                "identical_content_parseback": None,
                                "train_decision": "MEASUREMENT_OWED",
                            }
                            occupancy = (
                                "NO_PF2_CONTENT_ASSIGNED_NOT_A_ZERO_UNIVERSE_CLAIM"
                            )
                        atlas_rows.append(
                            {
                                "schema": "ddm_pf2_typed_split_atlas_bucket.v2",
                                "class_pair": pair_id,
                                "class_ids": [left, right],
                                "class_stratum": stratum,
                                "visibility": visibility,
                                "g4_temporal_class": temporal_class,
                                "representation_type": representation_type,
                                "type_contract": representation_contract[
                                    representation_type
                                ],
                                "occupancy_status": occupancy,
                                "connection_conditioned_fiber_counted_bytes_shared": len(
                                    xi_fiber
                                ),
                                "independent_fiber_counted_bytes_shared": len(
                                    identity_fiber
                                ),
                                "connection_minus_independent_fiber_bytes_shared": len(
                                    xi_fiber
                                )
                                - len(identity_fiber),
                                "connection_column_status": connection_applicability,
                                "shared_bytes_must_not_be_summed_across_buckets": True,
                                "lambda_dependence": (
                                    "fiber<->gauge boundary follows the DR2b tolerance "
                                    "ladder at each lambda; no fixed universal split"
                                ),
                                "visibility_scope": (
                                    "discrete argmax skeleton occupancy is Seg-visible; "
                                    "receiver color realization may couple to Pose"
                                ),
                                **rate,
                                "measured_flip_mass_weight": (
                                    int(measured_rate["content_event_count"])
                                    / int(np.count_nonzero(flip))
                                    if owns_measured_skeleton
                                    else None
                                ),
                                "evidence_axis": EVIDENCE_AXIS,
                                "research_only": True,
                                "score_claim": False,
                            }
                        )
    measured_total = sum(
        int(row["content_event_count"])
        for row in atlas_rows
        if row["visibility"] == "seg-visible"
        and row["representation_type"] == "SKELETON"
    )
    if measured_total != int(np.count_nonzero(flip)):
        raise PF2Error("typed atlas event partition does not close")

    class_cell_mass = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        target_class = target == class_id
        mass = int(np.count_nonzero(flip & target_class))
        class_cell_mass.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "flip_mass": mass,
                "flip_mass_fraction": mass / int(np.count_nonzero(flip)),
                "scope": "exact G4/V12 target-class error mass",
            }
        )
    pair_mass_rows.sort(key=lambda row: (-row["flip_mass"], row["pair_id"]))
    for rank, row in enumerate(pair_mass_rows, start=1):
        row["heavy_tail_rank"] = rank

    from scipy import ndimage

    presence = [
        ndimage.maximum_filter(
            (target == class_id).astype(np.uint8), size=(1, 3, 3), mode="nearest"
        ).astype(bool)
        for class_id in range(len(CLASS_NAMES))
    ]
    presence_code = np.zeros(target.shape, dtype=np.uint8)
    for class_id, mask in enumerate(presence):
        presence_code |= mask.astype(np.uint8) << class_id
    triple_junction_rows = []
    for first in range(len(CLASS_NAMES)):
        for second in range(first + 1, len(CLASS_NAMES)):
            for third in range(second + 1, len(CLASS_NAMES)):
                code = (1 << first) | (1 << second) | (1 << third)
                junction = presence_code == code
                triple_junction_rows.append(
                    {
                        "junction_id": (
                            f"{CLASS_NAMES[first]}--{CLASS_NAMES[second]}--"
                            f"{CLASS_NAMES[third]}"
                        ),
                        "target_junction_cells": int(np.count_nonzero(junction)),
                        "flip_mass": int(np.count_nonzero(junction & flip)),
                        "scope": "exact 3x3 target-neighborhood triple-class incidence",
                    }
                )
    bit_counts = np.asarray([int(value).bit_count() for value in range(32)])
    higher_order = bit_counts[presence_code] >= 4

    global_race = race_event_coders(events)
    xi_rows = xi_574_receipt["measurements"]["n600"]["rows"]
    identity_row = next(
        row
        for row in xi_rows
        if row["arm"] == "XTDL1_identity_xi_context_control"
    )
    connection_row = next(
        row
        for row in xi_rows
        if row["arm"] == "XTDL1_planar3_from_composed_screw_predictor"
    )
    if (
        identity_row["semantic_lane_sha256"]
        != connection_row["semantic_lane_sha256"]
        or int(identity_row["description_wire_bytes_before_terminal"])
        != len(identity_fiber)
        or int(connection_row["description_wire_bytes_before_terminal"])
        != len(xi_fiber)
    ):
        raise PF2Error("settled #574 equal-semantic connection race differs")
    program_artifact = _artifact(
        root / "04_event_xi_formulation" / "event_skeleton.counted",
        global_race.program_coded.payload,
    )
    flat_artifact = _artifact(
        root / "04_event_xi_formulation" / "flat_event_native.counted",
        global_race.flat_coded.payload,
    )
    fiber_artifact = _artifact(
        root / "04_event_xi_formulation" / "xi_connection_conditioned_fiber.xtdl1",
        xi_fiber,
    )
    identity_artifact = _artifact(
        root / "04_event_xi_formulation" / "identity_independent_fiber.xtdl1",
        identity_fiber,
    )
    program_total = program_artifact["bytes"] + fiber_artifact["bytes"]
    flat_total = flat_artifact["bytes"] + identity_artifact["bytes"]
    event_formulation = {
        "formulation_id": "F3_EVENT_SKELETON_X_CONNECTION_CONDITIONED_XI_FIBER",
        "decisive_formulation": True,
        "content": (
            "exact n600 G4 predicted->target flip events plus #574's exact "
            "equal-semantic Lane connection race"
        ),
        "event_count": global_race.event_count,
        "program": {
            "skeleton_counted_bytes": program_artifact["bytes"],
            "fiber_counted_bytes": fiber_artifact["bytes"],
            "total_counted_bytes": program_total,
            "skeleton_codec": global_race.program_coded.codec,
            "skeleton_artifact": program_artifact,
            "fiber_artifact": fiber_artifact,
            "fiber_semantics": (
                "planar3 composed-screw connection-conditioned XTDL1"
            ),
        },
        "flat": {
            "event_native_counted_bytes": flat_artifact["bytes"],
            "fiber_counted_bytes": identity_artifact["bytes"],
            "total_counted_bytes": flat_total,
            "event_native_codec": global_race.flat_coded.codec,
            "event_native_artifact": flat_artifact,
            "fiber_artifact": identity_artifact,
            "fiber_semantics": "identity independent-slot XTDL1 control",
        },
        "event_skeleton_delta_bytes": global_race.delta_program_minus_flat_bytes,
        "connection_conditioning_delta_bytes": len(xi_fiber)
        - len(identity_fiber),
        "delta_program_minus_flat_bytes": program_total - flat_total,
        "identical_event_content_parseback": True,
        "equal_semantic_fiber_sha256": identity_row["semantic_lane_sha256"],
        "connection_operator": (
            "generic xi predictor FREE rule-118; parameters/presence/residual COUNTED"
        ),
        "fiber_policy": (
            "continuous residual coded in transported frame; never tokenized by skeleton"
        ),
        "accepted": program_total < flat_total,
        "metric_status": IDENTICAL_CONTENT_CODER_CONTROL,
        "verdict_eligible": True,
        "waterfill_eligible": False,
        "metric_scope": (
            "identical event parse-back and equal-semantic #574 fiber content "
            "cancel scorer distortion; this verdict prices exact bytes only"
        ),
        "verdict_scope": (
            "exact G4 event field plus #574 planar3 connection-conditioned Lane "
            "fiber against exact identity control; not receiver-realized RGB or "
            "independent physical-BEV custody"
        ),
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
    }
    atlas = {
        "schema": "ddm_pf2_dimension_conditioned_atlas.v1",
        "dimensions": {
            "class_cells": list(CLASS_NAMES),
            "class_pairs": [
                f"{CLASS_NAMES[left]}--{CLASS_NAMES[right]}"
                for left, right in class_pairs
            ],
            "class_stratum": list(CLASS_STRATA),
            "visibility": list(VISIBILITY_CLASSES),
            "g4_temporal_class": list(TEMPORAL_CLASSES),
            "representation_type": list(REPRESENTATION_TYPES),
        },
        "bucket_count": len(atlas_rows),
        "measured_seg_skeleton_event_total": measured_total,
        "rows": atlas_rows,
        "measured_mass_weighting": {
            "class_cells": class_cell_mass,
            "class_pair_boundaries_and_interiors": pair_mass_rows,
            "g3_heavy_tail_rank_source": (
                "exact G4 reconstructed event mass; route order is descending mass"
            ),
            "necessity_inversion_anchor": (
                "61% bytes -> edges, Road--Lane; settled operator directive anchor"
            ),
        },
        "metric_tolerance_pricing": {
            "status": "BLOCKED_BUCKET_COMPLETE_MARGIN_FISHER_LAMBDA_LADDER",
            "ladder_order": (
                "measured rank-4 margin-Fisher and scorer-Hessian geometry first; "
                "identity/Euclidean only as a labeled control"
            ),
            "settled_dr2b_reference_rows": _dr2b_metric_reference(dr2b_receipt),
            "per_bucket_lambda_rows_materialized": 0,
            "reason": (
                "the settled DR2b receipt supplies two instance rungs in contest "
                "units but no canonical lambda and no all-10-pair bucket field; "
                "PF2 therefore refuses to turn fixed-content coder winners into "
                "split-point training decisions"
            ),
            "reactivation": (
                "measure each occupied pair/stratum/temporal bucket in rank-4 "
                "winner-rival coordinates and record margin-Fisher tolerance, "
                "contest-unit Delta S, and fiber/gauge membership at every lambda"
            ),
        },
        "interactions": {
            "pairwise": (
                "all ten rank-4 head hyperplanes materialized; directed corrections "
                "are preserved inside exact event codes"
            ),
            "cross_class": [
                {
                    "anchor": "MC1 MENU1 pose win damaged Seg across classes",
                    "status": "MEASURED_REUSED",
                },
                {
                    "anchor": "v19b +0.0805 cross-correction synergy",
                    "status": "MEASURED_REUSED_NOT_REDERIVED",
                },
                {
                    "anchor": "SE-coupled batching and #535 by-class geometry",
                    "status": "REUSED_INTERACTION_LAW",
                },
            ],
            "triple_junctions": triple_junction_rows,
            "higher_order_target_neighborhood_cells": int(
                np.count_nonzero(higher_order)
            ),
        },
        "dynamics": {
            "worldsheet_evolution": "SKELETON plus CONNECTION",
            "island_birth_death": "SKELETON plus RESIDUAL",
            "temporal_flicker_anchor": (
                "66% degraded-lane x Lane-head-gain; applies to every class-pair row"
            ),
            "xi_advection": "CONNECTION with transported-frame FIBER deltas",
            "g4_partition": {
                name: int(np.count_nonzero(mask))
                for name, mask in temporal_masks.items()
            },
        },
        "cross_visibility_anchors": {
            "ker(A)-invisible": {
                "nullity_fraction": 0.806742315223,
                "status": "REUSED_SETTLED_GLOBAL_DIMENSION_NOT_ALLOCATED_TO_CLASS_BUCKETS",
            },
            "pose-visible": {
                "anchor": "PA1 frame0 affine; SegNet consumes last frame only",
                "status": "REUSED_ARCHITECTURAL_AND_MEASURED_ANCHOR_NO_G4_BUCKET_OCCUPANCY",
            },
            "both": {
                "anchor": (
                    "MC1 static hood reassert changed Seg and official PoseNet "
                    "input/output jointly"
                ),
                "status": "REUSED_MEASURED_ANCHOR_NO_CROSS_BASE_BUCKET_MERGE",
            },
        },
        "partition_rule": (
            "all 10 class pairs x 2 strata x 4 visibility classes x 3 temporal "
            "classes x 5 representation types are materialized. Only exact G4 "
            "discrete skeleton content is assigned; unassigned rows are measurement "
            "debt, not asserted zeros."
        ),
        "layer_scope": {
            "covered": "L1 scorer cells/edges, L2 trajectory connection, L3 typed coding",
            "excluded": "L4 scorer feature space",
            "excluded_owner": "rs1 Catalog #659; do not duplicate",
        },
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
    }
    return atlas, event_formulation


def run(config_path: Path, output_directory: Path) -> Path:
    config = PF2Config.model_validate_json(_read_regular(config_path))
    root = Path(config.checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    required = 2 << 30
    if shutil.disk_usage(root).free < required:
        raise PF2Error("SSD storage preflight failed")
    mc1_tool = _import_tool(
        "_ddm_pf2_mc1_parent", "measure_ddm_mc1_hood_static_reassert.py"
    )
    menu1 = _import_tool(
        "_ddm_pf2_menu1_parent", "measure_ddm_menu1_realized_flip_menu.py"
    )
    mc1_config_payload = _bound(
        config.mc1_config_path, config.mc1_config_sha256, "MC1 config"
    )
    mc1_config = mc1_tool.MC1Config.model_validate_json(mc1_config_payload)
    mc1_receipt = json.loads(
        _bound(config.mc1_receipt_path, config.mc1_receipt_sha256, "MC1 receipt")
    )
    pf1_receipt = json.loads(
        _bound(config.pf1_receipt_path, config.pf1_receipt_sha256, "PF1 receipt")
    )
    g4_receipt = json.loads(
        _bound(config.g4_receipt_path, config.g4_receipt_sha256, "G4 receipt")
    )
    v12_receipt = json.loads(
        _bound(config.v12_receipt_path, config.v12_receipt_sha256, "V12 receipt")
    )
    xi_fiber = _bound(config.xi_fiber_path, config.xi_fiber_sha256, "xi fiber")
    if len(xi_fiber) != config.xi_fiber_bytes:
        raise PF2Error("xi fiber bytes differ")
    identity_fiber = _bound(
        config.identity_fiber_path,
        config.identity_fiber_sha256,
        "identity fiber",
    )
    if len(identity_fiber) != config.identity_fiber_bytes:
        raise PF2Error("identity fiber bytes differ")
    xi_574_receipt = json.loads(
        _bound(
            config.xi_574_receipt_path,
            config.xi_574_receipt_sha256,
            "#574 receipt",
        )
    )
    if (
        mc1_receipt.get("verdict") != "MC1_MEASURED_INSTANCE_NOT_JOINT_POSITIVE"
        or pf1_receipt.get("falsifier", {}).get("verdict")
        != "POSITIVE_DISCRETE_SKELETON_RUNG_SURVIVES_FIBERS_OPEN"
        or g4_receipt.get("verdict")
        != "MEASURED_ADVISORY_SPATIAL_STATIONARITY_COMPLETE_XI_PROXY_SCOPED"
    ):
        raise PF2Error("parent authority verdict differs")
    menu_config, inputs = menu1._config_and_inputs(
        _resolve(mc1_config.menu1_config_path)
    )
    receiver = receive_preuint8_q8_archive(inputs["archive"])
    palette = menu1._palette(receiver)
    menu_root = Path(menu_config.checkpoint_root)
    statistics_payload = _read_regular(
        menu_root / "01_local_statistics_payload.bin"
    )
    base_cells = mc1_tool._load_all_base_cells(
        menu_root, batch_size=config.scorer_batch_size
    )
    supports = mc1_tool.derive_hood_supports(base_cells)
    static_payload = mc1_tool.encode_stored_support(supports.static)
    mc1_static = mc1_receipt["support_derivation"]["partition"]["static_stored"]
    if (
        len(static_payload) != int(mc1_static["COUNTED"])
        or sha256_bytes(static_payload) != mc1_static["sha256"]
    ):
        raise PF2Error("MC1 static support rederivation differs")
    alpha_numerator = round(config.pose_projection_alpha * 4)
    alpha_payload = bytes([alpha_numerator])
    if (
        len(alpha_payload) != 1
        or alpha_payload[0] / 4 != config.pose_projection_alpha
    ):
        raise PF2Error("pose projection alpha q4 parse-back differs")
    alpha_artifact = _artifact(
        root / "01_projection" / "alpha_q4.counted",
        alpha_payload,
    )
    labels = open_stored_npy_memmap(Path(menu_config.target_cache_path), "lstars")
    poses = open_stored_npy_memmap(Path(menu_config.target_cache_path), "gt_poses")
    segnet, posenet, scorer_custody = menu1._load_models(menu_config)
    parent = mc1_receipt["input_custody"]["menu1_parent"]
    pose_formulation = _measure_pose_projection(
        config=config,
        menu1=menu1,
        mc1=mc1_tool,
        menu_config=menu_config,
        receiver=receiver,
        palette=palette,
        statistics_payload=statistics_payload,
        labels=labels,
        poses=poses,
        segnet=segnet,
        posenet=posenet,
        supports=supports,
        support_counted_bytes=len(static_payload),
        alpha_counted_bytes=alpha_artifact["bytes"],
        root=root,
        menu_root=menu_root,
        parent=parent,
    )
    _publish_json(root / "stage_checkpoints" / "02_pose_complete.json", pose_formulation)
    atlas, event_formulation = _measure_atlas_and_event_formulation(
        config=config,
        g4_receipt=g4_receipt,
        v12_receipt=v12_receipt,
        xi_fiber=xi_fiber,
        identity_fiber=identity_fiber,
        xi_574_receipt=xi_574_receipt,
        dr2b_receipt=inputs["receipts"]["dr2b"],
        root=root,
    )
    _publish_json(root / "stage_checkpoints" / "03_atlas_complete.json", atlas)
    _publish_json(
        root / "stage_checkpoints" / "04_event_xi_complete.json", event_formulation
    )
    pf1_structural = pf1_receipt["composed_structural_rows"]["composed_typed"]
    if pf1_structural.get("semantic_parseback_exact") is not True:
        raise PF2Error("PF1 structural coder control lacks exact semantic parse-back")
    f1_metric = resolve_formulation_metric_disposition(
        IDENTICAL_CONTENT_CODER_CONTROL,
        identical_content_proven=True,
    )
    f2_metric = resolve_formulation_metric_disposition(
        IDENTITY_EUCLIDEAN_CONTROL,
        identical_content_proven=False,
    )
    f3_metric = resolve_formulation_metric_disposition(
        IDENTICAL_CONTENT_CODER_CONTROL,
        identical_content_proven=bool(
            event_formulation["identical_event_content_parseback"]
            and event_formulation["equal_semantic_fiber_sha256"]
        ),
    )
    formulations = [
        {
            "formulation_id": "F1_PF1_STRUCTURAL_DISCRETE_SKELETON",
            "metric": "delta_counted_bytes",
            "metric_status": f1_metric.metric_status,
            "verdict_eligible": f1_metric.verdict_eligible,
            "waterfill_eligible": f1_metric.waterfill_eligible,
            "delta": int(pf1_structural["delta_program_minus_flat_bytes"]),
            "accepted": (
                f1_metric.verdict_eligible
                and int(pf1_structural["delta_program_minus_flat_bytes"]) < 0
            ),
            "scope": "discrete skeleton plus opaque native-coded fibers",
        },
        {
            "formulation_id": pose_formulation["formulation_id"],
            "metric": "identity_control_delta_joint_advisory_S",
            "metric_status": f2_metric.metric_status,
            "verdict_eligible": f2_metric.verdict_eligible,
            "waterfill_eligible": f2_metric.waterfill_eligible,
            "delta": pose_formulation["delta_advisory_objective_vs_parent"],
            "accepted": False,
            "raw_control_improves": pose_formulation[
                "raw_control_improves_joint_objective"
            ],
            "scope": (
                "INSTANCE-scoped naive MENU1-parent static hood control at exact "
                "n600; metric-active Pose/Fisher/Bregman rerun owed"
            ),
        },
        {
            "formulation_id": event_formulation["formulation_id"],
            "metric": "delta_counted_bytes",
            "metric_status": f3_metric.metric_status,
            "verdict_eligible": f3_metric.verdict_eligible,
            "waterfill_eligible": f3_metric.waterfill_eligible,
            "delta": event_formulation["delta_program_minus_flat_bytes"],
            "accepted": (
                f3_metric.verdict_eligible
                and event_formulation["accepted"]
            ),
            "scope": event_formulation["verdict_scope"],
        },
    ]
    eligible = [row for row in formulations if row["verdict_eligible"]]
    accepted = [row for row in eligible if row["accepted"]]
    verdict = (
        "PF2_METRIC_ACTIVE_THREE_FORMULATION_ADJUDICATION_INCOMPLETE_"
        "F1_EXACT_CONTROL_SURVIVES_F3_NEGATIVE_F2_OWED"
    )
    route_rows = [
        {
            "route": "train-decision",
            "decision": row["train_decision"],
            "coder_control_decision": row["coder_control_decision"],
            "bucket_id": row["bucket_id"],
            "delta_program_minus_flat_bytes": row["delta_program_minus_flat_bytes"],
            "metric_status": row["metric_status"],
            "waterfill_eligible": row["waterfill_eligible"],
        }
        for row in atlas["rows"]
        if row["visibility"] == "seg-visible"
        and row["representation_type"] == "SKELETON"
        and row["content_event_count"] > 0
    ]
    route_rows.append(
        {
            "route": "c1/MyCar",
            "decision": "HOLD_METRIC_ACTIVE_RERUN_OWED",
            "candidate_id": POSE_CANDIDATE_ID,
            "identity_control_delta_advisory_objective": pose_formulation[
                "delta_advisory_objective_vs_parent"
            ],
            "delta_counted_bytes": pose_formulation["delta_counted_bytes_vs_parent"],
            "metric_status": f2_metric.metric_status,
            "verdict_eligible": f2_metric.verdict_eligible,
            "waterfill_eligible": f2_metric.waterfill_eligible,
        }
    )
    optimal_binding = resolve_bregman_dual_metric_adoption(
        [canonical_bregman_dual_metric_binding()]
    )
    binding_path = _resolve(optimal_binding.registry_entry.binding_artifact)
    metric_inventory = {
        "policy_bindings": {
            "optimal_metric": {
                "metric_id": optimal_binding.binding.metric_id,
                "binding_artifact": optimal_binding.registry_entry.binding_artifact,
                "binding_artifact_sha256": _sha256_file(binding_path),
                "fisher_natural_cotangent_geometry": (
                    optimal_binding.binding.fisher_natural_cotangent_geometry
                ),
                "fisher_natural_cotangent_solve": (
                    optimal_binding.binding.fisher_natural_cotangent_solve
                ),
                "status": "RESOLVED_NOT_APPLIED_TO_F2_CONTROL",
            }
        },
        "seg": {
            "required_geometry": (
                "rank-4 winner/rival head hyperplane coordinates plus measured "
                "margin-Fisher per bucket"
            ),
            "status": "PARTIAL_DR2B_INSTANCE_ROWS_ONLY_NOT_BUCKET_COMPLETE",
        },
        "pose": {
            "required_geometry": (
                "exact low-rank <=6 PoseNet output quadratic/Hessian through "
                "the shared receiver and R path"
            ),
            "status": "MISSING_FOR_F2_PROJECTION",
        },
        "second_order": {
            "required": (
                "exact composite-R adjoint #391, Hessian-preconditioned solve "
                "#423, SPD normal-coordinate momentum #552 where applicable"
            ),
            "status": "NOT_PRESENT_IN_SEALED_PF2_INPUTS",
        },
        "dual_metric_readback": {
            "required": "Euclidean-vs-Fisher cosine plus relative-norm readback",
            "status": "MISSING_FOR_F2_PROJECTION",
        },
        "ladder_order": (
            "metric-active scorer geometry first; identity/Euclidean retained "
            "only as a labeled control"
        ),
    }
    receipt = {
        "schema": SCHEMA,
        "lane_id": LANE_ID,
        "delegation_checkpoint_key": DELEGATION_KEY,
        "run_id": config.run_id,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.stable_hash(),
        "derivation": {
            "L0": (
                "upstream/evaluate.py:92 composes discrete Seg argmax disagreement "
                "with continuous Pose MSE and exact archive rate"
            ),
            "L1_seg": (
                "rank-4 Unet argmax cells define discrete skeleton membership; "
                "within-cell margins are continuous class-stratum fibers"
            ),
            "L1_pose": (
                "official two-frame resize/YUV6 FastViT path defines global-support "
                "continuous pose fibers; MC1 proves hood statistics collide with Pose"
            ),
            "L1_R": (
                "80.6742315223% real-linear resize nullity is an invisible zero-byte "
                "fiber class, not an archive-byte saving"
            ),
            "L2_temporal": (
                "600 pair evaluations are one trajectory: G4 supplies event skeleton "
                "classes; #574 supplies the CONNECTION-conditioned xi-flow fiber"
            ),
            "missing_layers_correction": (
                "two types are insufficient: PF2 carries SKELETON, CONNECTION, "
                "FIBER, GAUGE, and RESIDUAL; L4 feature space remains rs1/#659"
            ),
        },
        "typed_split_atlas": atlas,
        "pose_stat_projection": pose_formulation,
        "event_skeleton_x_xi_fiber": event_formulation,
        "metric_geometry_inventory": metric_inventory,
        "family_adjudication": {
            "total_formulation_count": len(formulations),
            "eligible_formulation_count": len(eligible),
            "ineligible_formulation_count": len(formulations) - len(eligible),
            "formulations": formulations,
            "accepted_formulation_count": len(accepted),
            "verdict": verdict,
            "law": (
                "identity/Euclidean rows are controls, never verdict-bearing; "
                "strict identical-content coder controls remain eligible because "
                "distortion cancels; a complete three-formulation family verdict "
                "requires F2's metric-active rerun"
            ),
        },
        "route_table": route_rows,
        "directive_consumption": [
            {
                "utc": "2026-07-24T02:07:59Z",
                "status": "CONSUMED",
                "effect": (
                    "replaced the insufficient skeleton/fiber split with five types; "
                    "added shared connection-conditioned versus identity fiber columns, "
                    "lambda-dependent fiber/gauge custody, and L4 rs1/#659 exclusion"
                ),
            },
            {
                "utc": "2026-07-24T02:08:43Z",
                "status": "CONSUMED",
                "effect": (
                    "made all ten class-pair edges, measured mass, interactions, triple "
                    "junctions, and dynamics the headline; demoted the MyCar hood to one "
                    "measured interaction row"
                ),
            },
            {
                "utc": "2026-07-24T02:27:12Z",
                "status": "CONSUMED",
                "effect": (
                    "reclassified F2 as an INSTANCE-scoped identity/Euclidean "
                    "control, blocked all per-bucket split routes without measured "
                    "margin-Fisher/lambda custody, resolved the #504 optimal-metric "
                    "binding, and retained only exact-content coder verdicts"
                ),
            },
            {
                "utc": "2026-07-24T02:28:21Z",
                "status": "CONSUMED",
                "effect": (
                    "made rank-4 scorer coordinates, exact second-order Pose/R "
                    "geometry, and metric-first ladder order explicit reactivation "
                    "requirements; identity remains the last control rung"
                ),
            },
        ],
        "guardrails": {
            "frame0_byte_identical": pose_formulation["frame0_byte_identical"],
            "v19c_total_error_count": 2_923_991,
            "v19c_residual_bucket_error_count": 2_265_811,
            "old_lineages_or_donor_spine_used": False,
            "fibers_opaque_to_skeleton_coder": True,
            "torch_threads": config.scorer_threads,
            "alpha_payload": alpha_artifact,
        },
        "input_custody": {
            "mc1_config": {
                "path": config.mc1_config_path,
                "sha256": config.mc1_config_sha256,
            },
            "mc1_receipt": {
                "path": config.mc1_receipt_path,
                "sha256": config.mc1_receipt_sha256,
            },
            "pf1_receipt": {
                "path": config.pf1_receipt_path,
                "sha256": config.pf1_receipt_sha256,
            },
            "g4_receipt": {
                "path": config.g4_receipt_path,
                "sha256": config.g4_receipt_sha256,
            },
            "v12_receipt": {
                "path": config.v12_receipt_path,
                "sha256": config.v12_receipt_sha256,
            },
            "xi_fiber": {
                "path": config.xi_fiber_path,
                "sha256": config.xi_fiber_sha256,
                "bytes": len(xi_fiber),
            },
            "identity_fiber": {
                "path": config.identity_fiber_path,
                "sha256": config.identity_fiber_sha256,
                "bytes": len(identity_fiber),
            },
            "xi_574_receipt": {
                "path": config.xi_574_receipt_path,
                "sha256": config.xi_574_receipt_sha256,
            },
            "dr2b_receipt": {
                "path": menu_config.dr2b_receipt_path,
                "sha256": menu_config.dr2b_receipt_sha256,
            },
        },
        "scorer_custody": scorer_custody,
        "storage_preflight": {
            "tier": "/Volumes/VertigoDataTier/pact",
            "checkpoint_root": str(root),
            "required_free_bytes": required,
            "status": "PASS",
            "cleanup_policy": (
                "preserve immutable n600 checkpoints and counted rate artifacts; "
                "no source or cache bytes moved or deleted"
            ),
        },
        "verdict": verdict,
        "verdict_scope": (
            "two exact-content local rate controls plus one identity-metric n600 "
            "hood control over settled V19C/MENU1, G4, PF1, #574, and DR2b "
            "custody; metric-active three-formulation adjudication remains "
            "incomplete; no contest axis, promotion, or family-global negative"
        ),
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": config.pointer,
        "pointer_moved": False,
        "paid_dispatch": False,
        "exact_eval": False,
        "training": False,
        "research_only": True,
        "main_landing_review_required": True,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / "ddm_pf2_dimension_conditioned_two_type_receipt.json"
    _publish_json(output, receipt)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = run(_resolve(args.config), _resolve(args.output_directory))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
