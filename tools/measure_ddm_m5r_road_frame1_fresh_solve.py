#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fresh receiver-closed Road frame-1 integer-lattice solve.

The g3 hard subset is a proposal/search surface only.  Every searched state is
compiled into real archive bytes and replayed through uint8, exact scorer
preprocess/R, and frozen scorers.  The selected endpoint is then replayed at
n600 in immutable batch-16 checkpoints before any full-video statement.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.canonical_equations.ddm_road_frame1_reach_curve_20260723 import (  # noqa: E402
    RoadReachPoint,
    certified_residual_interval,
    normalized_chord_knee,
    receiver_closed_reach_curve,
)
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    CLASS_ORDER,
    receive_carrier_compose_archive,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_joint_descent import (  # noqa: E402
    compile_parameterized_archive,
    lift_v15_archive,
    parameter_group_indices,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError  # noqa: E402
from tac.optimization.pure_priced_realized_objective import RealizedObjectiveState  # noqa: E402
from tac.through_r.resolution_chain import contest_faithful_R_numpy  # noqa: E402
from tools.measure_ddm_v16_coupled_joint_solve import _sha256_array, _torch_forward_full  # noqa: E402
from tools.measure_ddm_v19_pure_priced_objective import DDMV19PurePricedObjectiveConfigV1, _context  # noqa: E402

SCHEMA = "ddm_m5r_road_frame1_fresh_solve_receipt.v1"
AXIS = "[macOS-CPU frozen-scorer advisory]"
POINTER = "0.1910828242 [contest-CPU]"
RATE_DENOMINATOR = 37_545_489


class DDMM5RRoadFreshSolveConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["DDMM5RRoadFreshSolveConfigV1"] = Field(
        default="DDMM5RRoadFreshSolveConfigV1",
        alias="schema",
        serialization_alias="schema",
    )
    run_id: str
    seed: Literal[1234] = 1234
    source_v15_archive: str
    source_v15_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v19_config_path: str
    v19_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    g3_hard_pair_registry: str
    g3_hard_pair_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v18b_receipt_path: str
    v18b_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    m5_control_receipt_path: str
    m5_control_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    subset_name: Literal["top24"] = "top24"
    proxy_exact_singletons: int = Field(default=20, ge=8, le=32)
    restricted_master_width: int = Field(default=12, ge=4, le=20)
    maximum_combination_order: int = Field(default=4, ge=2, le=4)
    maximum_combination_replays: int = Field(default=24, ge=8, le=64)
    scorer_batch_size: Literal[16] = 16
    byte_box: Literal[200000] = 200000
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    def typed_hash(self) -> str:
        return hashlib.sha256(
            rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True))
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    family: str
    deltas: tuple[tuple[int, int], ...]

    def state(self, parameter_count: int) -> np.ndarray:
        result = np.zeros(parameter_count, dtype=np.float32)
        for index, value in self.deltas:
            result[index] += np.float32(value)
        return result


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bound(path_value: str, expected: str, label: str) -> bytes:
    path = Path(path_value)
    path = path if path.is_absolute() else REPO / path
    if not path.is_file() or path.is_symlink():
        raise DirectDescriptionError(f"{label} missing or unsafe: {path}")
    payload = path.read_bytes()
    actual = _sha(payload)
    if actual != expected:
        raise DirectDescriptionError(f"{label} SHA differs: {actual} != {expected}")
    return payload


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = rfc8785_canonicalize(dict(value))
    if path.exists():
        if path.read_bytes() != payload:
            raise DirectDescriptionError(f"immutable checkpoint differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise DirectDescriptionError(f"immutable artifact differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _candidate_pool(lift: Any, pair_ids: Sequence[int]) -> tuple[Candidate, ...]:
    pair_set = {int(value) for value in pair_ids}
    rows: list[Candidate] = []
    active_track_indexes = [
        index
        for index, track in enumerate(lift.g1.tracks)
        if any(lift.g1.knots[knot].pair_index in pair_set for knot in track.knot_indices)
    ]
    for track_index in active_track_indexes:
        for offset, axis in ((0, "x"), (1, "y")):
            parameter = 2 * track_index + offset
            for sign in (-1, 1):
                rows.append(
                    Candidate(
                        f"island_track{track_index}_{axis}_{sign:+d}",
                        "island_worldsheet",
                        ((parameter, sign),),
                    )
                )
    groups = parameter_group_indices(lift)
    for family in ("lane_program", "shared_template_dof"):
        for parameter in groups[family]:
            for sign in (-1, 1):
                rows.append(
                    Candidate(
                        f"{family}_{parameter}_{sign:+d}",
                        family,
                        ((parameter, sign),),
                    )
                )
    for offset, axis in ((0, "x"), (1, "y")):
        for sign in (-1, 1):
            deltas = tuple((2 * index + offset, sign) for index in active_track_indexes)
            rows.append(Candidate(f"coherent_worldsheet_{axis}_{sign:+d}", "coherent_worldsheet", deltas))
    if len({row.candidate_id for row in rows}) != len(rows):
        raise DirectDescriptionError("fresh-solve candidate IDs are not unique")
    return tuple(rows)


def _compile_state(lift: Any, candidates: Sequence[Candidate]) -> tuple[bytes, np.ndarray]:
    state = np.zeros(len(lift.parameter_names), dtype=np.float32)
    for candidate in candidates:
        state += candidate.state(len(lift.parameter_names))
    return compile_parameterized_archive(
        lift,
        state,
        include_lane_programs=bool(np.any(state[np.asarray(parameter_group_indices(lift)["lane_program"])])),
    )


def _measurement(
    *,
    archive: bytes,
    pair_ids: Sequence[int],
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    segnet: Any,
    posenet: Any,
    road_id: int,
    baseline_cells: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    camera = receive_carrier_compose_archive(archive, verify_member_effects=False).render_camera_pairs(pair_ids)
    _logits, cells, pose6 = _torch_forward_full(segnet, posenet, camera)
    selection = np.asarray(pair_ids, dtype=np.int64)
    labels = np.asarray(labels_all[selection])
    poses = np.asarray(poses_all[selection])
    errors = cells != labels
    baseline_errors = baseline_cells != labels
    road = labels == road_id
    d_seg = float(np.mean(errors))
    d_pose = float(np.mean(np.square(pose6 - poses), dtype=np.float64))
    objective = RealizedObjectiveState(d_seg, d_pose, len(archive)).objective
    return (
        {
            "archive_bytes": len(archive),
            "archive_sha256": _sha(archive),
            "d_seg": d_seg,
            "d_pose": d_pose,
            "joint_objective": objective,
            "errors": int(np.count_nonzero(errors)),
            "sites": int(errors.size),
            "road_control_errors": int(np.count_nonzero(baseline_errors & road)),
            "road_candidate_errors": int(np.count_nonzero(errors & road)),
            "road_helpful_flips": int(np.count_nonzero(baseline_errors & ~errors & road)),
            "road_harmful_flips": int(np.count_nonzero(~baseline_errors & errors & road)),
            "helpful_flips_all_classes": int(np.count_nonzero(baseline_errors & ~errors)),
            "harmful_flips_all_classes": int(np.count_nonzero(~baseline_errors & errors)),
            "cells_sha256": _sha256_array(cells),
            "evidence_axis": AXIS,
            "score_claim": False,
        },
        cells,
    )


def _screen_candidates(
    *,
    root: Path,
    lift: Any,
    pool: Sequence[Candidate],
    pair_ids: Sequence[int],
    baseline_camera: np.ndarray,
    baseline_cells: np.ndarray,
    labels: np.ndarray,
    road_id: int,
    config_hash: str,
) -> list[dict[str, Any]]:
    path = root / "stage_checkpoints" / "01_full_parameter_proxy_screen.json"
    if path.exists():
        payload = json.loads(path.read_bytes())
        if payload.get("typed_config_sha256") != config_hash:
            raise DirectDescriptionError("proxy-screen config identity differs")
        rows = list(payload["rows"])
        if len(rows) != len(pool):
            raise DirectDescriptionError("proxy-screen candidate count differs")
        for index, (candidate, row) in enumerate(zip(pool, rows, strict=True)):
            if row.get("candidate_id") != candidate.candidate_id:
                raise DirectDescriptionError("proxy-screen candidate order differs")
            _atomic_json(
                root / "stage_checkpoints" / "01_proxy_rows" / f"{index:03d}.json",
                {
                    "schema": "ddm_m5r_proxy_row.v1",
                    "typed_config_sha256": config_hash,
                    **row,
                },
            )
        return rows
    baseline_r = contest_faithful_R_numpy(baseline_camera[:, 1], ste_round=True)
    road_error = (labels == road_id) & (baseline_cells != labels)
    correct = baseline_cells == labels
    rows = []
    for index, candidate in enumerate(pool):
        row_path = root / "stage_checkpoints" / "01_proxy_rows" / f"{index:03d}.json"
        if row_path.exists():
            row = json.loads(row_path.read_bytes())
            if (
                row.get("typed_config_sha256") != config_hash
                or row.get("candidate_id") != candidate.candidate_id
            ):
                raise DirectDescriptionError("proxy-row checkpoint identity differs")
            row = {
                key: value
                for key, value in row.items()
                if key not in {"schema", "typed_config_sha256"}
            }
        else:
            try:
                archive, _state = _compile_state(lift, (candidate,))
                camera = receive_carrier_compose_archive(
                    archive, verify_member_effects=False
                ).render_camera_pairs(pair_ids)
                candidate_r = contest_faithful_R_numpy(camera[:, 1], ste_round=True)
                changed = np.max(np.abs(candidate_r - baseline_r), axis=-1)
                target_signal = float(changed[road_error].sum(dtype=np.float64))
                collateral_signal = float(changed[correct].sum(dtype=np.float64))
                proxy = target_signal / (1.0 + collateral_signal)
                status = "ELIGIBLE" if target_signal > 0.0 else "EXACT_R_NO_ROAD_ERROR_SUPPORT"
                row = {
                    "candidate_id": candidate.candidate_id,
                    "family": candidate.family,
                    "deltas": [list(value) for value in candidate.deltas],
                    "archive_bytes": len(archive),
                    "archive_sha256": _sha(archive),
                    "road_error_R_l1_signal": target_signal,
                    "baseline_correct_R_l1_signal": collateral_signal,
                    "proposal_proxy": proxy,
                    "status": status,
                    "proxy_only_not_acceptance": True,
                }
            except DirectDescriptionError as exc:
                row = {
                    "candidate_id": candidate.candidate_id,
                    "family": candidate.family,
                    "deltas": [list(value) for value in candidate.deltas],
                    "status": "COMPILER_REFUSAL",
                    "verdict_scope": f"INSTANCE:{candidate.candidate_id}",
                    "reason": str(exc),
                    "proxy_only_not_acceptance": True,
                }
            _atomic_json(
                row_path,
                {
                    "schema": "ddm_m5r_proxy_row.v1",
                    "typed_config_sha256": config_hash,
                    **row,
                },
            )
        rows.append(row)
    _atomic_json(
        path,
        {
            "schema": "ddm_m5r_full_parameter_proxy_screen.v1",
            "typed_config_sha256": config_hash,
            "candidate_count": len(pool),
            "rows": rows,
            "acceptance_authority": "NONE; exact frozen-scorer replay is separate",
        },
    )
    return rows


def _selected_for_exact(
    pool_by_id: Mapping[str, Candidate],
    rows: Sequence[Mapping[str, Any]],
    maximum: int,
) -> tuple[Candidate, ...]:
    eligible = sorted(
        (row for row in rows if row.get("status") == "ELIGIBLE"),
        key=lambda row: (-float(row["proposal_proxy"]), str(row["candidate_id"])),
    )
    selected = [str(row["candidate_id"]) for row in eligible[:maximum]]
    for family in sorted({row.family for row in pool_by_id.values()}):
        family_rows = [row for row in eligible if row["family"] == family]
        selected.extend(str(row["candidate_id"]) for row in family_rows[:2])
    selected.extend(
        row.candidate_id for row in pool_by_id.values() if row.family == "coherent_worldsheet"
    )
    unique = tuple(dict.fromkeys(selected))
    return tuple(pool_by_id[value] for value in unique[:32])


def _full_n600_replay(
    *,
    root: Path,
    config: DDMM5RRoadFreshSolveConfigV1,
    archive: bytes,
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    segnet: Any,
    posenet: Any,
    control: Mapping[str, Any],
    road_id: int,
) -> dict[str, Any]:
    class_names = tuple(CLASS_ORDER)
    totals = dict.fromkeys(class_names, 0)
    pose_sse = 0.0
    pose_coordinates = 0
    chain = hashlib.sha256()
    for start in range(0, 600, config.scorer_batch_size):
        stop = min(600, start + config.scorer_batch_size)
        path = root / "stage_checkpoints" / "04_selected_n600" / f"batch_{start:04d}_{stop:04d}.json"
        if path.exists():
            row = json.loads(path.read_bytes())
            if row.get("candidate_archive_sha256") != _sha(archive):
                raise DirectDescriptionError("n600 checkpoint archive identity differs")
        else:
            pair_ids = tuple(range(start, stop))
            camera = receive_carrier_compose_archive(
                archive, verify_member_effects=False
            ).render_camera_pairs(pair_ids)
            _logits, cells, pose6 = _torch_forward_full(segnet, posenet, camera)
            labels = np.asarray(labels_all[start:stop])
            poses = np.asarray(poses_all[start:stop])
            errors = cells != labels
            per_stratum = {
                name: int(np.count_nonzero(errors & (labels == class_id)))
                for class_id, name in enumerate(class_names)
            }
            row = {
                "schema": "ddm_m5r_n600_batch.v1",
                "typed_config_sha256": config.typed_hash(),
                "candidate_archive_sha256": _sha(archive),
                "source_pair_ids": list(pair_ids),
                "candidate_cells_sha256": _sha256_array(cells),
                "per_stratum_candidate_errors": per_stratum,
                "candidate_pose_squared_error_sum": float(
                    np.square(pose6 - poses).sum(dtype=np.float64)
                ),
                "pose_coordinates": int(pose6.size),
                "evidence_axis": AXIS,
                "score_claim": False,
            }
            _atomic_json(path, row)
        for name in class_names:
            totals[name] += int(row["per_stratum_candidate_errors"][name])
        pose_sse += float(row["candidate_pose_squared_error_sum"])
        pose_coordinates += int(row["pose_coordinates"])
        chain.update(rfc8785_canonicalize(row))
    total_errors = sum(totals.values())
    sites = 600 * 384 * 512
    d_seg = total_errors / sites
    d_pose = pose_sse / pose_coordinates
    objective = RealizedObjectiveState(d_seg, d_pose, len(archive)).objective
    control_strata = control["measurement"]["per_stratum"]
    transition = {
        name: {
            "control_errors": int(control_strata[name]["control_errors"]),
            "candidate_errors": totals[name],
            "net_errors_closed": int(control_strata[name]["control_errors"]) - totals[name],
        }
        for name in class_names
    }
    return {
        "pair_count": 600,
        "all_batches_checkpointed": True,
        "batch_chain_sha256": chain.hexdigest(),
        "archive_bytes": len(archive),
        "archive_sha256": _sha(archive),
        "d_seg": d_seg,
        "d_pose": d_pose,
        "joint_objective": objective,
        "control_joint_objective": RealizedObjectiveState(
            sum(int(value["control_errors"]) for value in control_strata.values()) / sites,
            float(control["measurement"]["aggregate"]["control_d_pose"]),
            int(control["measurement"]["control"]["archive_bytes"]),
        ).objective,
        "per_stratum": transition,
        "evidence_axis": AXIS,
        "score_claim": False,
    }


def run(config: DDMM5RRoadFreshSolveConfigV1, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config_hash = config.typed_hash()
    source_archive = _bound(config.source_v15_archive, config.source_v15_archive_sha256, "V15 archive")
    registry = json.loads(
        _bound(config.g3_hard_pair_registry, config.g3_hard_pair_registry_sha256, "g3 registry")
    )
    v18b = json.loads(_bound(config.v18b_receipt_path, config.v18b_receipt_sha256, "v18b receipt"))
    control = json.loads(
        _bound(config.m5_control_receipt_path, config.m5_control_receipt_sha256, "m5 control receipt")
    )
    v19_bytes = _bound(config.v19_config_path, config.v19_config_sha256, "v19 config")
    v19_config = DDMV19PurePricedObjectiveConfigV1.model_validate_json(v19_bytes)
    pair_ids = tuple(int(value) for value in registry[config.subset_name])
    correlation = float(registry["correlation_receipt"]["correlations"][config.subset_name]["pearson_r"])
    road_id = tuple(CLASS_ORDER).index("Road")
    if tuple(CLASS_ORDER) != tuple(control["class_order"]["names"]):
        raise DirectDescriptionError("self-detected scorer class order differs from m5 custody")
    lift = lift_v15_archive(source_archive)
    groups = parameter_group_indices(lift)
    ctx = _context(v19_config)
    if ctx["n600_archive"] != source_archive:
        raise DirectDescriptionError("v19 scorer context does not bind the selected V15 source archive")
    labels_all = ctx["labels_all"]
    poses_all = ctx["poses_all"]
    segnet, posenet = ctx["segnet"], ctx["posenet"]

    baseline_camera = receive_carrier_compose_archive(source_archive).render_camera_pairs(pair_ids)
    _baseline_logits, baseline_cells, baseline_pose = _torch_forward_full(segnet, posenet, baseline_camera)
    labels = np.asarray(labels_all[np.asarray(pair_ids, dtype=np.int64)])
    poses = np.asarray(poses_all[np.asarray(pair_ids, dtype=np.int64)])
    baseline_d_seg = float(np.mean(baseline_cells != labels))
    baseline_d_pose = float(np.mean(np.square(baseline_pose - poses), dtype=np.float64))
    baseline = {
        "state_id": "control",
        "archive_bytes": len(source_archive),
        "archive_sha256": _sha(source_archive),
        "d_seg": baseline_d_seg,
        "d_pose": baseline_d_pose,
        "joint_objective": RealizedObjectiveState(
            baseline_d_seg, baseline_d_pose, len(source_archive)
        ).objective,
        "road_control_errors": int(np.count_nonzero((baseline_cells != labels) & (labels == road_id))),
        "road_candidate_errors": int(np.count_nonzero((baseline_cells != labels) & (labels == road_id))),
        "road_helpful_flips": 0,
        "road_harmful_flips": 0,
        "evidence_axis": AXIS,
        "score_claim": False,
    }
    _atomic_json(
        root / "stage_checkpoints" / "00_control.json",
        {"schema": "ddm_m5r_subset_control.v1", "typed_config_sha256": config_hash, **baseline},
    )

    pool = _candidate_pool(lift, pair_ids)
    by_id = {row.candidate_id: row for row in pool}
    screen = _screen_candidates(
        root=root,
        lift=lift,
        pool=pool,
        pair_ids=pair_ids,
        baseline_camera=baseline_camera,
        baseline_cells=baseline_cells,
        labels=labels,
        road_id=road_id,
        config_hash=config_hash,
    )
    exact_candidates = _selected_for_exact(by_id, screen, config.proxy_exact_singletons)
    exact_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(exact_candidates):
        path = root / "stage_checkpoints" / "02_exact_singletons" / f"{index:03d}.json"
        if path.exists():
            row = json.loads(path.read_bytes())
            if (
                row.get("typed_config_sha256") != config_hash
                or row.get("candidate_id") != candidate.candidate_id
            ):
                raise DirectDescriptionError("exact-singleton checkpoint identity differs")
        else:
            try:
                archive, realized = _compile_state(lift, (candidate,))
                measurement, _cells = _measurement(
                    archive=archive,
                    pair_ids=pair_ids,
                    labels_all=labels_all,
                    poses_all=poses_all,
                    segnet=segnet,
                    posenet=posenet,
                    road_id=road_id,
                    baseline_cells=baseline_cells,
                )
                row = {
                    "schema": "ddm_m5r_exact_singleton.v1",
                    "typed_config_sha256": config_hash,
                    "candidate_id": candidate.candidate_id,
                    "family": candidate.family,
                    "deltas": [list(value) for value in candidate.deltas],
                    "status": "MEASURED",
                    "realized_nonzero": {
                        str(i): int(value) for i, value in enumerate(realized) if int(value) != 0
                    },
                    "measurement": measurement,
                    "joint_objective_delta": measurement["joint_objective"]
                    - baseline["joint_objective"],
                }
            except DirectDescriptionError as exc:
                row = {
                    "schema": "ddm_m5r_exact_singleton.v1",
                    "typed_config_sha256": config_hash,
                    "candidate_id": candidate.candidate_id,
                    "family": candidate.family,
                    "deltas": [list(value) for value in candidate.deltas],
                    "status": "COMPILER_REFUSAL",
                    "verdict_scope": f"INSTANCE:{candidate.candidate_id}",
                    "reason": str(exc),
                    "negative_family_claim": False,
                }
            _atomic_json(path, row)
        exact_rows.append(row)

    feasible_exact_rows = [
        row for row in exact_rows if row.get("status", "MEASURED") == "MEASURED"
    ]
    master = sorted(
        feasible_exact_rows,
        key=lambda row: (
            float(row["joint_objective_delta"]),
            int(row["measurement"]["road_candidate_errors"]),
            str(row["candidate_id"]),
        ),
    )[: config.restricted_master_width]
    master_candidates = [by_id[str(row["candidate_id"])] for row in master]
    delta_by_id = {str(row["candidate_id"]): float(row["joint_objective_delta"]) for row in master}
    proposals = []
    for order in range(2, config.maximum_combination_order + 1):
        for combo in itertools.combinations(master_candidates, order):
            proposals.append(
                (
                    sum(delta_by_id[row.candidate_id] for row in combo),
                    tuple(row.candidate_id for row in combo),
                    combo,
                )
            )
    proposals.sort(key=lambda value: (value[0], value[1]))
    combination_rows: list[dict[str, Any]] = []
    combination_refusals: list[dict[str, Any]] = []
    seen_states: set[bytes] = set()
    for proposal_rank, (_additive, ids, combo) in enumerate(proposals):
        if len(combination_rows) >= config.maximum_combination_replays:
            break
        state = sum(
            (row.state(len(lift.parameter_names)) for row in combo),
            start=np.zeros(len(lift.parameter_names), dtype=np.float32),
        )
        state_key = np.ascontiguousarray(state, dtype="<f4").tobytes()
        if state_key in seen_states:
            continue
        seen_states.add(state_key)
        refusal_path = (
            root / "stage_checkpoints" / "03_set_refusals" / f"{proposal_rank:05d}.json"
        )
        try:
            archive, realized = _compile_state(lift, combo)
        except DirectDescriptionError as exc:
            refusal = {
                "schema": "ddm_m5r_exact_set_refusal.v1",
                "typed_config_sha256": config_hash,
                "proposal_rank": proposal_rank,
                "candidate_ids": list(ids),
                "status": "COMPILER_REFUSAL",
                "verdict_scope": f"INSTANCE:RESTRICTED_MASTER_PROPOSAL_{proposal_rank}",
                "reason": str(exc),
                "negative_family_claim": False,
            }
            _atomic_json(refusal_path, refusal)
            combination_refusals.append(refusal)
            continue
        path = root / "stage_checkpoints" / "03_exact_sets" / f"{len(combination_rows):03d}.json"
        if path.exists():
            row = json.loads(path.read_bytes())
            if (
                row.get("typed_config_sha256") != config_hash
                or tuple(row.get("candidate_ids", ())) != ids
            ):
                raise DirectDescriptionError("exact-set checkpoint identity differs")
        else:
            measurement, _cells = _measurement(
                archive=archive,
                pair_ids=pair_ids,
                labels_all=labels_all,
                poses_all=poses_all,
                segnet=segnet,
                posenet=posenet,
                road_id=road_id,
                baseline_cells=baseline_cells,
            )
            row = {
                "schema": "ddm_m5r_exact_set_replay.v1",
                "typed_config_sha256": config_hash,
                "proposal_rank": proposal_rank,
                "candidate_ids": list(ids),
                "combination_order": len(ids),
                "realized_nonzero": {
                    str(i): int(value) for i, value in enumerate(realized) if int(value) != 0
                },
                "measurement": measurement,
                "joint_objective_delta": measurement["joint_objective"] - baseline["joint_objective"],
                "additive_delta_used_for_proposal_only": sum(delta_by_id[value] for value in ids),
                "acceptance_authority": "exact receiver/scorer replay",
            }
            _atomic_json(path, row)
        combination_rows.append(row)

    state_rows = [
        {
            "state_id": "control",
            "candidate_ids": [],
            "measurement": baseline,
            "joint_objective_delta": 0.0,
        },
        *[
            {
                "state_id": str(row["candidate_id"]),
                "candidate_ids": [str(row["candidate_id"])],
                "measurement": row["measurement"],
                "joint_objective_delta": row["joint_objective_delta"],
            }
            for row in exact_rows
            if row.get("status", "MEASURED") == "MEASURED"
        ],
        *[
            {
                "state_id": "+".join(row["candidate_ids"]),
                "candidate_ids": row["candidate_ids"],
                "measurement": row["measurement"],
                "joint_objective_delta": row["joint_objective_delta"],
            }
            for row in combination_rows
        ],
    ]
    eligible = [
        row
        for row in state_rows
        if int(row["measurement"]["archive_bytes"]) <= config.byte_box
        and float(row["joint_objective_delta"]) < 0.0
    ]
    winner = min(
        eligible or state_rows[:1],
        key=lambda row: (
            float(row["measurement"]["joint_objective"]),
            int(row["measurement"]["road_candidate_errors"]),
            str(row["state_id"]),
        ),
    )
    winner_candidates = tuple(by_id[value] for value in winner["candidate_ids"])
    selected_archive, selected_realized = _compile_state(lift, winner_candidates)
    _atomic_bytes(root / "selected.not_a_candidate.zip.receipt-bytes", selected_archive)

    full = _full_n600_replay(
        root=root,
        config=config,
        archive=selected_archive,
        labels_all=labels_all,
        poses_all=poses_all,
        segnet=segnet,
        posenet=posenet,
        control=control,
        road_id=road_id,
    )
    full_admitted = (
        len(selected_archive) <= config.byte_box
        and float(full["joint_objective"]) < float(full["control_joint_objective"])
    )
    road_full = full["per_stratum"]["Road"]
    measured_reach = max(0, int(road_full["net_errors_closed"])) if full_admitted else 0
    residual_interval = certified_residual_interval(
        control_errors=int(road_full["control_errors"]),
        measured_reachable_errors_closed=measured_reach,
        exhaustive_reachable_set=False,
    )
    subset_points = [
        RoadReachPoint(
            int(row["measurement"]["archive_bytes"]),
            int(row["measurement"]["road_control_errors"]),
            int(row["measurement"]["road_candidate_errors"]),
            float(row["measurement"]["joint_objective"]),
            str(row["state_id"]),
        )
        for row in state_rows
    ]
    curve = receiver_closed_reach_curve(subset_points)
    knee = normalized_chord_knee(curve)
    residual_bucket_names = ("Road", "Undrivable", "MyCar")
    residual_bucket_net = sum(
        int(full["per_stratum"][name]["net_errors_closed"]) for name in residual_bucket_names
    )
    scope_reduction = max(0, residual_bucket_net) if full_admitted else 0
    previous_scope = int(control["certification"]["catalog_366_true_scope_interval_errors"][1])
    road_fraction = int(road_full["net_errors_closed"]) / int(road_full["control_errors"])
    greedy_fraction = (
        int(control["measurement"]["per_stratum"]["Road"]["net_errors_closed"])
        / int(control["measurement"]["per_stratum"]["Road"]["control_errors"])
    )
    confound = full_admitted and road_fraction >= 2.0 * greedy_fraction
    v18_common_sha = str(v18b["common_master"]["archive_sha256"])
    v18_compatible = v18_common_sha == _sha(source_archive)
    if v18_compatible:
        raise DirectDescriptionError("unexpected v18b/V15 common-master identity needs explicit compiler review")
    receipt = {
        "schema": SCHEMA,
        "run_id": config.run_id,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config_hash,
        "class_order": {
            "names": list(CLASS_ORDER),
            "road_id": road_id,
            "self_detected_not_hardcoded": True,
        },
        "parameterization": {
            "historical_j2_named_count": 706,
            "historical_count_status": "SUPERSEDED_OVERSTATEMENT",
            "receiver_effective_count": len(lift.parameter_names),
            "receiver_effective_groups": {name: len(indexes) for name, indexes in groups.items()},
            "overcount_reason": (
                "aspect/rotation lift metadata and BEV/range seed fields lack current receiver wire coordinates"
            ),
            "fresh_proxy_screen_count": len(pool),
            "exact_singleton_replays": len(exact_rows),
            "exact_singleton_measurements": len(feasible_exact_rows),
            "exact_singleton_compiler_refusals": len(exact_rows) - len(feasible_exact_rows),
            "exact_non_greedy_set_replays": len(combination_rows),
            "exact_non_greedy_set_compiler_refusals": len(combination_refusals),
            "v18b_solve_generated_columns": {
                "source_receipt_sha256": config.v18b_receipt_sha256,
                "source_common_master_sha256": v18_common_sha,
                "target_v15_master_sha256": _sha(source_archive),
                "compatible": v18_compatible,
                "status": "REFUSED_CROSS_MASTER_COMPOSITION",
                "verdict_scope": (
                    "INSTANCE:V18B_POSTSOLVE_COMMON_MASTER_COLUMNS_X_V15_G1_LIFT; "
                    "column family remains open under a reviewed hybrid compiler"
                ),
            },
        },
        "subset": {
            "name": config.subset_name,
            "pair_ids": list(pair_ids),
            "pair_registry_sha256": config.g3_hard_pair_registry_sha256,
            "measured_subset_to_full_pearson_r": correlation,
            "correlation_source_proposals": int(
                registry["correlation_receipt"]["n_measured_proposals"]
            ),
            "rank_or_kill_from_subset_alone": False,
            "selected_state": winner,
        },
        "reach_curve": {
            "definition": "exact receiver/scorer states; byte-Pareto Road error envelope",
            "points": [
                {
                    "state_id": row.state_id,
                    "archive_bytes": row.archive_bytes,
                    "road_errors_closed": row.net_errors_closed,
                    "road_candidate_errors": row.candidate_errors,
                    "joint_objective": row.joint_objective,
                }
                for row in curve
            ],
            "knee_state_id": None if knee is None else knee.state_id,
            "knee_archive_bytes": None if knee is None else knee.archive_bytes,
            "knee_errors_closed": None if knee is None else knee.net_errors_closed,
        },
        "full_n600_selected_endpoint": {
            **full,
            "selected_state_id": winner["state_id"],
            "selected_candidate_ids": winner["candidate_ids"],
            "selected_realized_nonzero": {
                str(i): int(value)
                for i, value in enumerate(selected_realized)
                if int(value) != 0
            },
            "inside_c1_byte_box": len(selected_archive) <= config.byte_box,
            "strict_joint_objective_admitted": full_admitted,
            "road_solvable_fraction_net_at_box": road_fraction if full_admitted else 0.0,
            "road_certified_infeasible_residual_interval_errors": list(residual_interval),
            "residual_certification_exhaustive": False,
        },
        "catalog_366_true_scope_update": {
            "prior_interval_errors": [0, previous_scope],
            "updated_interval_errors": [0, previous_scope - scope_reduction],
            "measured_admitted_residual_bucket_net_errors_closed": scope_reduction,
            "numeric_certified_residual": None,
            "reason": (
                "one admitted receiver-closed state narrows the reachable upper scope but the "
                "finite coefficient set was proxy-screened and not exhaustively enumerated"
                if full_admitted
                else "selected subset state failed the full-n600 joint objective; no scope credit"
            ),
        },
        "greedy_instrument_comparison": {
            "v19b_greedy_road_net_errors_closed": int(
                control["measurement"]["per_stratum"]["Road"]["net_errors_closed"]
            ),
            "v19b_greedy_road_fraction": greedy_fraction,
            "fresh_solve_road_net_errors_closed": int(road_full["net_errors_closed"]),
            "fresh_solve_road_fraction": road_fraction,
            "fresh_minus_greedy_errors_closed": int(road_full["net_errors_closed"])
            - int(control["measurement"]["per_stratum"]["Road"]["net_errors_closed"]),
            "verdict": (
                "GREEDY-INSTRUMENT-CONFOUND-CONFIRMED"
                if confound
                else "GREEDY-INSTRUMENT-CONFOUND-NOT-CONFIRMED_BY_THIS_INSTANCE"
            ),
        },
        "certification": {
            "verdict": (
                "FRESH_RECEIVER_CLOSED_ROAD_SOLVE_ADMITTED_NONEXHAUSTIVE"
                if full_admitted
                else "SUBSET_PROPOSAL_NOT_ADMITTED_AT_FULL_N600"
            ),
            "verdict_scope": (
                "INSTANCE:V15_368_RECEIVER_EFFECTIVE_INTEGER_DOF_X_TOP24_PROXY_SCREEN_X_"
                "EXACT_RESTRICTED_MASTER_ORDER4_X_C1_200000_BYTE_BOX"
            ),
            "negative_family_claim": False,
            "score_claim": False,
            "evidence_axis": AXIS,
            "pointer": POINTER,
            "pointer_moved": False,
            "promotion_eligible": False,
            "main_landing_review_required": True,
        },
        "sources": {
            "v15_archive_sha256": config.source_v15_archive_sha256,
            "g3_registry_sha256": config.g3_hard_pair_registry_sha256,
            "v18b_receipt_sha256": config.v18b_receipt_sha256,
            "m5_control_receipt_sha256": config.m5_control_receipt_sha256,
            "v19_config_sha256": config.v19_config_sha256,
        },
        "resume": {
            "immutable_proxy_screen": True,
            "immutable_singletons": len(exact_rows),
            "immutable_set_replays": len(combination_rows),
            "immutable_n600_batches": 38,
        },
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "evidence_axis": AXIS,
        "pointer": POINTER,
        "pointer_moved": False,
        "main_landing_review_required": True,
    }
    path = root / "receipt.json"
    _atomic_json(path, receipt)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    config = DDMM5RRoadFreshSolveConfigV1.model_validate_json(args.config.read_bytes())
    print(run(config, args.output_directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
