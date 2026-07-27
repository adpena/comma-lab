#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize G90 V2 authority-separated, exact-all-coarse costates."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.materialize_taskspace_projected_population_costates as v1m
from tac.optimization.taskspace_exact_coarse_costates_v2 import (
    compute_batch_exact_coarse_costates_v2,
)
from tac.optimization.taskspace_projected_population_costates_v1 import (
    MAX_BATCH_PAIRS,
    PAIR_COUNT,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    PopulationScorePointV1,
    ProjectedOperandRowV1,
    exact_replay_projected_intervention,
    group_g72_batch_proposals,
    realize_and_project_g72_group,
)
from tac.witness_dsl.taskspace_g72_fresh_n600_g49_analytic_factor_compiler_v1 import (
    derive_v9_boundary_shearlet_stage_proposals,
)

CONFIG_SCHEMA: Final = "tac.taskspace_exact_coarse_costate_config.v2"
PREFLIGHT_SCHEMA: Final = "tac.taskspace_exact_coarse_costate_preflight.v2"
BATCH_SCHEMA: Final = "tac.taskspace_exact_coarse_costate_batch.v2"
STAGE_SCHEMA: Final = "tac.taskspace_exact_coarse_costate_stage.v2"
AGGREGATE_SCHEMA: Final = "tac.taskspace_exact_coarse_costate_aggregate.v2"
BLOCKER_SCHEMA: Final = "tac.taskspace_exact_coarse_costate_blocker.v2"
STAGE_PAIRS: Final = 120
STAGE_COUNT: Final = 5


class ExactCoarseCostateV2Error(RuntimeError):
    """A V2 input, authority, resume, or exact replay invariant failed."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _seal(body: dict[str, Any], *, field: str) -> dict[str, Any]:
    if field in body:
        raise ExactCoarseCostateV2Error(f"{field} already exists")
    return {**body, field: _sha256_bytes(_canonical_json_bytes(body))}


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    payload = _canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ExactCoarseCostateV2Error(f"immutable V2 checkpoint differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ExactCoarseCostateV2Error(f"{label} cannot be read") from exc
    if type(value) is not dict:
        raise ExactCoarseCostateV2Error(f"{label} is not one JSON object")
    return value


@dataclass(frozen=True, slots=True)
class ConfigV2:
    path: Path
    output_root: Path
    v1_config: dict[str, object]
    v1_terminal_receipt: dict[str, object]
    seed: int
    num_threads: int
    safety_reserve_bytes: int


def _identity(value: object, *, label: str) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value) != {"path", "bytes", "sha256"}
        or type(value["path"]) is not str
        or type(value["bytes"]) is not int
        or type(value["sha256"]) is not str
        or len(value["sha256"]) != 64
    ):
        raise ExactCoarseCostateV2Error(f"{label} identity differs")
    return dict(value)


def load_config(path: Path) -> ConfigV2:
    value = _load_json(path, label="G90 V2 config")
    required = {
        "schema",
        "output_root",
        "v1_config",
        "v1_terminal_receipt",
        "seed",
        "num_threads",
        "safety_reserve_bytes",
        "batch_pairs_maximum",
        "stage_pairs",
        "stage_count",
        "exact_replay_policy",
        "pareto_pruning_allowed",
        "dense_costates_persisted",
        "research_only",
    }
    if set(value) != required or value.get("schema") != CONFIG_SCHEMA:
        raise ExactCoarseCostateV2Error("G90 V2 config schema/key set differs")
    output_root = Path(str(value["output_root"])).resolve()
    if (
        not str(output_root).startswith("/Volumes/VertigoDataTier/pact/")
        or value["batch_pairs_maximum"] != MAX_BATCH_PAIRS
        or value["stage_pairs"] != STAGE_PAIRS
        or value["stage_count"] != STAGE_COUNT
        or value["exact_replay_policy"] != "ALL_DETERMINISTIC_PHYSICAL_GROUPS"
        or value["pareto_pruning_allowed"] is not False
        or value["dense_costates_persisted"] is not False
        or value["research_only"] is not True
        or type(value["seed"]) is not int
        or type(value["num_threads"]) is not int
        or type(value["safety_reserve_bytes"]) is not int
    ):
        raise ExactCoarseCostateV2Error("G90 V2 config contract differs")
    return ConfigV2(
        path=path.resolve(),
        output_root=output_root,
        v1_config=_identity(value["v1_config"], label="V1 config"),
        v1_terminal_receipt=_identity(
            value["v1_terminal_receipt"],
            label="V1 terminal receipt",
        ),
        seed=int(value["seed"]),
        num_threads=int(value["num_threads"]),
        safety_reserve_bytes=int(value["safety_reserve_bytes"]),
    )


def _verify_identity(value: dict[str, object], *, label: str) -> Path:
    path = Path(str(value["path"])).resolve()
    if not path.is_file() or path.stat().st_size != value["bytes"] or _sha256_file(path) != value["sha256"]:
        raise ExactCoarseCostateV2Error(f"{label} bytes differ")
    return path


def _source_identities() -> dict[str, dict[str, object]]:
    paths = (
        REPO_ROOT / "src/tac/optimization/taskspace_exact_coarse_costates_v2.py",
        REPO_ROOT / "tools/materialize_taskspace_exact_coarse_costates_v2.py",
    )
    return {
        str(path.relative_to(REPO_ROOT)): {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    }


def preflight(config: ConfigV2) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(config.output_root)
    if usage.free <= config.safety_reserve_bytes:
        raise ExactCoarseCostateV2Error("V2 SSD free bytes do not clear reserve")
    v1_config_path = _verify_identity(config.v1_config, label="V1 config")
    terminal_path = _verify_identity(
        config.v1_terminal_receipt,
        label="V1 terminal receipt",
    )
    terminal = _load_json(terminal_path, label="V1 terminal receipt")
    if (
        terminal.get("status") != "V1_TERMINAL_FAIL_CLOSED_SIX_BATCHES_STAGE0_INCOMPLETE"
        or terminal.get("terminal_v1", {}).get("sealed_batch_count") != 6
        or terminal.get("terminal_v1", {}).get("second_retry_allowed") is not False
    ):
        raise ExactCoarseCostateV2Error("V1 terminal classification differs")
    v1_config = v1m.load_config(v1_config_path)
    if v1_config.seed != config.seed or v1_config.num_threads != config.num_threads:
        raise ExactCoarseCostateV2Error("V1/V2 deterministic controls differ")
    # Reopen exact source/receiver custody through the already sealed V1 root.
    state = v1m._strict_preflight(v1_config)
    body = {
        "schema": PREFLIGHT_SCHEMA,
        "config": {
            "path": str(config.path),
            "bytes": config.path.stat().st_size,
            "sha256": _sha256_file(config.path),
        },
        "output_root": str(config.output_root),
        "source_artifacts": _source_identities(),
        "v1_config": config.v1_config,
        "v1_terminal_receipt": config.v1_terminal_receipt,
        "geometry": {
            "pair_count": PAIR_COUNT,
            "batch_pairs_maximum": MAX_BATCH_PAIRS,
            "stage_pairs": STAGE_PAIRS,
            "stage_count": STAGE_COUNT,
            "semantic_family_coordinates_per_batch": 8,
            "physical_group_coordinates_per_batch": "VARIABLE_GTE_8",
            "known_observed_maximum_physical_groups": 12,
        },
        "authority_contract": {
            "inference_cells_separate_from_differentiable_cells": True,
            "differentiable_tie_drift_annotated": True,
            "authority_cells_drive_exact_replay": True,
            "pose_fields_bound_to_exact_pair_state": True,
            "seg_fields_bound_to_exact_base_and_candidate_y1": True,
        },
        "replay_contract": {
            "policy": "ALL_DETERMINISTIC_PHYSICAL_GROUPS",
            "pareto_pruning_allowed": False,
            "local_admission_allowed": False,
            "actual_zip_rate_required_for_rate": True,
        },
        "resume": {
            "batch_checkpoint_policy": "immutable_atomic_every_batch",
            "stage_checkpoint_policy": "immutable_atomic_every_120_pairs",
            "prior_checkpoint_overwrite_allowed": False,
            "checkpoint_self_hash_sufficient_for_reuse": False,
            "skipped_batch_validation": ("FRESH_SOURCE_REDERIVATION_OF_ORDERED_PHYSICAL_GROUPS"),
        },
        "storage": {
            "free_bytes_observed": usage.free,
            "safety_reserve_bytes": config.safety_reserve_bytes,
            "dense_costates_persisted": False,
        },
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
        "research_only": True,
        "launch_review_required": True,
    }
    path = config.output_root / "00_preflight_receipt.json"
    if path.is_file():
        existing = _load_json(path, label="G90 V2 preflight resume")
        body["storage"]["free_bytes_observed"] = existing["storage"]["free_bytes_observed"]
    receipt = _seal(body, field="preflight_sha256")
    _atomic_write_json(path, receipt)
    return {"receipt": receipt, "state": state, "v1_config": v1_config}


def _next_incomplete_stage(root: Path) -> int | None:
    for stage_index in range(STAGE_COUNT):
        start = stage_index * STAGE_PAIRS
        path = root / f"20_stage_{stage_index:02d}_{start:04d}_{start + STAGE_PAIRS:04d}" / "stage_receipt.json"
        if not path.is_file():
            return stage_index
    return None


def _exact_replay_all(
    rows: tuple[ProjectedOperandRowV1, ...],
    *,
    expected_group_ids: tuple[str, ...],
    realized: dict[str, np.ndarray],
    target_cells: np.ndarray,
    costates: Any,
    posenet: Any,
    segnet: Any,
) -> tuple[ProjectedOperandRowV1, ...]:
    row_ids = tuple(row.operand_id for row in rows)
    if (
        len(rows) < 8
        or len(set(expected_group_ids)) != len(expected_group_ids)
        or row_ids != expected_group_ids
        or set(realized) != set(expected_group_ids)
    ):
        raise ExactCoarseCostateV2Error("V2 physical-group set differs from deterministic grouping")
    return tuple(
        exact_replay_projected_intervention(
            row,
            candidate_pairs_hwc=realized[row.operand_id],
            target_cells=target_cells,
            costates=costates,
            posenet=posenet,
            segnet=segnet,
            device="cpu",
        )
        for row in rows
    )


def _validate_batch_checkpoint(
    checkpoint: dict[str, Any],
    *,
    pair_start: int,
    pair_stop: int,
    expected_group_ids: tuple[str, ...],
) -> str:
    """Validate a sealed batch against freshly rederived physical groups."""

    expected_self = _sha256_bytes(
        _canonical_json_bytes({key: value for key, value in checkpoint.items() if key != "batch_checkpoint_sha256"})
    )
    projection_rows = checkpoint.get("projection_rows")
    basis_groups = checkpoint.get("actuator_basis_groups")
    replay_state_custody = checkpoint.get("exact_replay_state_custody")
    if (
        checkpoint.get("schema") != BATCH_SCHEMA
        or checkpoint.get("pair_range") != [pair_start, pair_stop]
        or checkpoint.get("batch_checkpoint_sha256") != expected_self
        or checkpoint.get("expected_physical_group_count") != len(expected_group_ids)
        or checkpoint.get("expected_physical_group_ids") != list(expected_group_ids)
        or checkpoint.get("projection_coordinate_count") != len(expected_group_ids)
        or checkpoint.get("exact_replay_policy") != "ALL_DETERMINISTIC_PHYSICAL_GROUPS"
        or checkpoint.get("all_deterministic_physical_groups_exact_replayed") is not True
        or type(projection_rows) is not list
        or tuple(row.get("operand_id") if type(row) is dict else None for row in projection_rows) != expected_group_ids
        or any(
            type(row) is not dict
            or row.get("exact_seg_score_delta") is None
            or row.get("exact_pose_score_delta") is None
            for row in projection_rows
        )
        or type(basis_groups) is not list
        or tuple(row.get("group_id") if type(row) is dict else None for row in basis_groups) != expected_group_ids
        or type(replay_state_custody) is not list
        or tuple(row.get("operand_id") if type(row) is dict else None for row in replay_state_custody)
        != expected_group_ids
        or checkpoint.get("pareto_pruning_performed") is not False
        or checkpoint.get("local_admission_performed") is not False
    ):
        raise ExactCoarseCostateV2Error("G90 V2 batch checkpoint differs from rederived physical groups")
    return expected_self


def _expected_group_ids_by_range(
    reopened: dict[str, Any],
) -> dict[tuple[int, int], tuple[str, ...]]:
    """Rederive every batch's physical groups from reopened source custody."""

    import torch

    state = reopened["state"]
    v1_config = reopened["v1_config"]
    posenet, segnet = v1m._load_models(v1_config)
    del posenet
    raw = np.memmap(
        Path(v1_config.g85_raw["path"]),
        mode="r",
        dtype=np.uint8,
        shape=(
            PAIR_COUNT,
            2,
            v1m.CAMERA_HEIGHT,
            v1m.CAMERA_WIDTH,
            3,
        ),
    )
    expected: dict[tuple[int, int], tuple[str, ...]] = {}
    try:
        for stage_index in range(STAGE_COUNT):
            stage_start = stage_index * STAGE_PAIRS
            stage_stop = stage_start + STAGE_PAIRS
            _, _, proposals, _ = _stage_inputs(
                state=state,
                stage_index=stage_index,
                segnet=segnet,
                raw=raw,
            )
            for pair_start in range(
                stage_start,
                stage_stop,
                MAX_BATCH_PAIRS,
            ):
                pair_stop = min(pair_start + MAX_BATCH_PAIRS, stage_stop)
                groups = group_g72_batch_proposals(
                    proposals,
                    pair_ids=tuple(range(pair_start, pair_stop)),
                )
                expected[(pair_start, pair_stop)] = tuple(group.group_id for group in groups)
    finally:
        del raw, segnet
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return expected


def _stage_inputs(
    *,
    state: dict[str, Any],
    stage_index: int,
    segnet: Any,
    raw: np.memmap,
) -> tuple[Any, dict[str, Any], tuple[Any, ...], np.ndarray]:
    stage_start = stage_index * STAGE_PAIRS
    stage_stop = stage_start + STAGE_PAIRS
    g78_stage = state["g78"].stages[stage_index]
    g87_stage_row = state["g87_receipt"]["stages"][stage_index]
    if (
        g78_stage.plan.pair_start != stage_start
        or g78_stage.plan.pair_stop_exclusive != stage_stop
        or g87_stage_row["pair_range"] != [stage_start, stage_stop]
    ):
        raise ExactCoarseCostateV2Error("V2 G78/G87 stage ranges differ")
    donor_proposals = v1m._stage_proposals(g87_stage_row)
    incumbent_atoms = tuple(state["incumbent_atoms"])
    incumbent_ids = tuple(
        sorted({atom.pair_index for atom in incumbent_atoms if stage_start <= atom.pair_index < stage_stop})
    )
    current_cells = np.ascontiguousarray(g78_stage.described_cells_u8.copy())
    if incumbent_ids:
        current_camera = np.ascontiguousarray(raw[list(incumbent_ids)])
        realized_current = state["g85_receiver"].render_camera_pair_batch(incumbent_ids)
        if not np.array_equal(realized_current, current_camera):
            raise ExactCoarseCostateV2Error("V2 exact G85 receiver differs from exact current raw")
        authority = v1m._seg_cells(segnet, current_camera)
        for local_index, pair_id in enumerate(incumbent_ids):
            current_cells[pair_id - stage_start] = authority[local_index]
        proposals = derive_v9_boundary_shearlet_stage_proposals(
            stage=g78_stage.plan,
            target_cells=g78_stage.target_cells_u8,
            target_margins=g78_stage.target_margins_f32,
            described_cells=current_cells,
            minimum_component_sites=v1m.COMPLETE_MINIMUM_COMPONENT_SITES,
            maximum_components_per_pair_role=(v1m.COMPLETE_MAXIMUM_COMPONENTS_PER_PAIR_ROLE),
        )
        incumbent_set = set(incumbent_ids)
        if tuple(row for row in proposals if row.atom.pair_index not in incumbent_set) != tuple(
            row for row in donor_proposals if row.atom.pair_index not in incumbent_set
        ):
            raise ExactCoarseCostateV2Error("V2 current-base proposal custody differs")
    else:
        proposals = donor_proposals
    return g78_stage, g87_stage_row, tuple(proposals), current_cells


def _run_stage(config: ConfigV2, reopened: dict[str, Any], stage_index: int) -> Path:
    import torch

    state = reopened["state"]
    v1_config = reopened["v1_config"]
    stage_start = stage_index * STAGE_PAIRS
    stage_stop = stage_start + STAGE_PAIRS
    posenet, segnet = v1m._load_models(v1_config)
    raw = np.memmap(
        Path(v1_config.g85_raw["path"]),
        mode="r",
        dtype=np.uint8,
        shape=(
            PAIR_COUNT,
            2,
            v1m.CAMERA_HEIGHT,
            v1m.CAMERA_WIDTH,
            3,
        ),
    )
    g78_stage, g87_stage_row, proposals, current_cells = _stage_inputs(
        state=state,
        stage_index=stage_index,
        segnet=segnet,
        raw=raw,
    )
    source = v1m._SourceCursor(
        source=Path(v1_config.source_video["path"]),
        batch_pairs=MAX_BATCH_PAIRS,
        seed=config.seed,
        num_threads=config.num_threads,
    )
    if stage_start:
        source.take(stage_start - 1, stage_start)
    labels = np.memmap(
        Path(v1_config.target_labels["path"]),
        mode="r",
        dtype=np.uint8,
        shape=(PAIR_COUNT, SCORER_HEIGHT, SCORER_WIDTH),
    )
    score_point = PopulationScorePointV1(
        global_mean_pose_dist=v1_config.global_mean_pose_dist,
        sample_count=PAIR_COUNT,
        archive_bytes=v1_config.g85_archive["bytes"],
        archive_sha256=v1_config.g85_archive["sha256"],
    )
    stage_dir = config.output_root / f"20_stage_{stage_index:02d}_{stage_start:04d}_{stage_stop:04d}"
    batch_bindings: list[dict[str, Any]] = []
    for pair_start in range(stage_start, stage_stop, MAX_BATCH_PAIRS):
        pair_stop = min(pair_start + MAX_BATCH_PAIRS, stage_stop)
        batch_path = stage_dir / "batches" / f"batch_{pair_start:04d}_{pair_stop:04d}.json"
        pair_ids = tuple(range(pair_start, pair_stop))
        groups = group_g72_batch_proposals(proposals, pair_ids=pair_ids)
        expected_group_ids = tuple(group.group_id for group in groups)
        if batch_path.is_file():
            checkpoint = _load_json(batch_path, label="G90 V2 batch resume")
            expected_self = _validate_batch_checkpoint(
                checkpoint,
                pair_start=pair_start,
                pair_stop=pair_stop,
                expected_group_ids=expected_group_ids,
            )
            source.take(pair_start, pair_stop)
            batch_bindings.append(
                {
                    "path": str(batch_path),
                    "bytes": batch_path.stat().st_size,
                    "sha256": _sha256_file(batch_path),
                    "batch_checkpoint_sha256": expected_self,
                    "pair_range": [pair_start, pair_stop],
                }
            )
            continue
        local_start = pair_start - stage_start
        local_stop = pair_stop - stage_start
        target = source.take(pair_start, pair_stop)
        base = np.ascontiguousarray(raw[pair_start:pair_stop])
        expected_target = np.ascontiguousarray(g78_stage.target_cells_u8[local_start:local_stop])
        expected_current = np.ascontiguousarray(current_cells[local_start:local_stop])
        if not np.array_equal(expected_target, labels[pair_start:pair_stop]):
            raise ExactCoarseCostateV2Error("V2 G78/G46 target custody differs")
        authority_current = v1m._seg_cells(segnet, base)
        authority_target = v1m._seg_cells(segnet, target)
        result = compute_batch_exact_coarse_costates_v2(
            candidate_pairs_hwc=base,
            target_pairs_hwc=target,
            expected_target_cells=expected_target,
            expected_current_cells=expected_current,
            authority_target_cells=authority_target,
            authority_current_cells=authority_current,
            pair_ids=pair_ids,
            posenet=posenet,
            segnet=segnet,
            device="cpu",
            score_point=score_point,
        )
        projected: list[ProjectedOperandRowV1] = []
        realized: dict[str, np.ndarray] = {}
        for group in groups:
            row, candidate = realize_and_project_g72_group(
                decoder=state["decoder"],
                group=group,
                base_camera_pairs=base,
                costates=result.costates,
                incumbent_atoms=tuple(state["incumbent_atoms"]),
                incumbent_frame_selector=state["incumbent_frame_selector"],
            )
            projected.append(row)
            realized[row.operand_id] = candidate
        replayed = _exact_replay_all(
            tuple(projected),
            expected_group_ids=expected_group_ids,
            realized=realized,
            target_cells=authority_target,
            costates=result.costates,
            posenet=posenet,
            segnet=segnet,
        )
        replayed_by_id = {row.operand_id: row for row in replayed}
        basis_groups = []
        replay_state_custody = []
        for group in groups:
            row = replayed_by_id[group.group_id]
            candidate = realized[group.group_id]
            basis_groups.append(
                {
                    "group_id": group.group_id,
                    "role": group.role,
                    "direction_rank": group.direction_rank,
                    "amplitude_scale": group.amplitude_scale,
                    "proposed_atoms_sha256": row.proposed_atoms_sha256,
                    "incumbent_atoms_sha256": row.incumbent_atoms_sha256,
                    "proposals": [proposal.to_dict() for proposal in group.proposals],
                    "proposal_fingerprints": [proposal.fingerprint for proposal in group.proposals],
                }
            )
            replay_state_custody.append(
                {
                    "operand_id": group.group_id,
                    "pose_conditioning_y0_sha256": _sha256_array(base[:, 0]),
                    "pose_conditioning_y1_sha256": _sha256_array(base[:, 1]),
                    "seg_base_y1_sha256": _sha256_array(base[:, 1]),
                    "seg_candidate_y1_sha256": _sha256_array(candidate[:, 1]),
                    "candidate_y0_preserved": bool(np.array_equal(candidate[:, 0], base[:, 0])),
                }
            )
        body = {
            "schema": BATCH_SCHEMA,
            "pair_range": [pair_start, pair_stop],
            "source_custody": {
                "candidate_camera_sha256": result.costates.candidate_sha256,
                "target_camera_sha256": result.costates.target_sha256,
                "target_cells_sha256": result.costates.target_cells_sha256,
                "current_cells_sha256": result.costates.described_cells_sha256,
            },
            "authority_drift": result.drift_dict(),
            "base_components": {
                "pair_pose_mse_f32": [float(value) for value in result.costates.base_pair_pose_mse],
                "seg_mismatch_count": result.costates.base_mismatch_count,
                "target_minus_current_gap_sum": result.costates.base_gap_sum,
            },
            "population_pose_pair_mse_vjp_scale": (score_point.pair_pose_mse_vjp_scale),
            "projection_coordinate_count": len(replayed),
            "expected_physical_group_count": len(groups),
            "expected_physical_group_ids": [group.group_id for group in groups],
            "projection_rows": [row.to_dict() for row in replayed],
            "actuator_basis_groups": basis_groups,
            "exact_replay_state_custody": replay_state_custody,
            "exact_replay_policy": "ALL_DETERMINISTIC_PHYSICAL_GROUPS",
            "all_deterministic_physical_groups_exact_replayed": (
                len(replayed) == len(groups)
                and tuple(row.operand_id for row in replayed) == tuple(group.group_id for group in groups)
                and all(
                    row.exact_seg_score_delta is not None and row.exact_pose_score_delta is not None for row in replayed
                )
            ),
            "pareto_pruning_performed": False,
            "local_admission_performed": False,
            "dense_costates_persisted": False,
            "actual_zip_delta_measured": False,
            "member_bytes_used_as_rate": False,
            "candidate_claim": False,
            "score_claim": False,
            "research_only": True,
            "encoder_only": True,
        }
        checkpoint = _seal(body, field="batch_checkpoint_sha256")
        _atomic_write_json(batch_path, checkpoint)
        batch_bindings.append(
            {
                "path": str(batch_path),
                "bytes": batch_path.stat().st_size,
                "sha256": _sha256_file(batch_path),
                "batch_checkpoint_sha256": checkpoint["batch_checkpoint_sha256"],
                "pair_range": [pair_start, pair_stop],
            }
        )
        del result, realized, target, base, authority_current, authority_target
        gc.collect()
        print(
            json.dumps(
                {
                    "status": "batch_complete",
                    "pair_range": [pair_start, pair_stop],
                    "exact_replay_count": len(replayed),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    base_pose_sum = 0.0
    base_seg_errors = 0
    current_drift_cells = 0
    target_drift_cells = 0
    projection_count = 0
    exact_replay_count = 0
    for binding in batch_bindings:
        checkpoint = _load_json(
            Path(binding["path"]),
            label="G90 V2 stage batch",
        )
        base_pose_sum += float(
            np.asarray(
                checkpoint["base_components"]["pair_pose_mse_f32"],
                dtype=np.float32,
            ).sum(dtype=np.float32)
        )
        base_seg_errors += int(checkpoint["base_components"]["seg_mismatch_count"])
        current_drift_cells += int(checkpoint["authority_drift"]["current"]["mismatch_cell_count"])
        target_drift_cells += int(checkpoint["authority_drift"]["target"]["mismatch_cell_count"])
        projection_count += int(checkpoint["projection_coordinate_count"])
        if checkpoint["all_deterministic_physical_groups_exact_replayed"] is not True:
            raise ExactCoarseCostateV2Error("V2 stage batch lost exact physical-group coverage")
        exact_replay_count += int(checkpoint["expected_physical_group_count"])
    stage_body = {
        "schema": STAGE_SCHEMA,
        "stage_index": stage_index,
        "pair_range": [stage_start, stage_stop],
        "batches": batch_bindings,
        "batch_count": len(batch_bindings),
        "projection_coordinate_count": projection_count,
        "exact_replay_count": exact_replay_count,
        "base_pose_squared_error_sum_f32": base_pose_sum,
        "base_segmentation_error_count": base_seg_errors,
        "differentiable_current_argmax_drift_cells": current_drift_cells,
        "differentiable_target_argmax_drift_cells": target_drift_cells,
        "exact_replay_policy": "ALL_DETERMINISTIC_PHYSICAL_GROUPS",
        "pareto_pruning_performed": False,
        "checkpoint_policy": "immutable_atomic_preserve_every_120_pair_stage",
        "dense_costates_persisted": False,
        "candidate_claim": False,
        "score_claim": False,
        "research_only": True,
        "encoder_only": True,
        "g78_stage_receipt_sha256": g78_stage.g78_stage_receipt_sha256,
        "g87_stage_checkpoint_sha256": g87_stage_row["checkpoint_sha256"],
    }
    receipt = _seal(stage_body, field="stage_receipt_sha256")
    path = stage_dir / "stage_receipt.json"
    _atomic_write_json(path, receipt)
    del posenet, segnet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path


def _write_aggregate_if_complete(config: ConfigV2) -> Path | None:
    stages = []
    pose_sum = 0.0
    seg_errors = 0
    current_drift_cells = 0
    target_drift_cells = 0
    projection_count = 0
    exact_replay_count = 0
    for stage_index in range(STAGE_COUNT):
        start = stage_index * STAGE_PAIRS
        path = (
            config.output_root
            / f"20_stage_{stage_index:02d}_{start:04d}_{start + STAGE_PAIRS:04d}"
            / "stage_receipt.json"
        )
        if not path.is_file():
            return None
        value = _load_json(path, label="G90 V2 stage")
        stages.append(
            {
                "stage_index": stage_index,
                "pair_range": value["pair_range"],
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "stage_receipt_sha256": value["stage_receipt_sha256"],
            }
        )
        pose_sum += float(value["base_pose_squared_error_sum_f32"])
        seg_errors += int(value["base_segmentation_error_count"])
        current_drift_cells += int(value["differentiable_current_argmax_drift_cells"])
        target_drift_cells += int(value["differentiable_target_argmax_drift_cells"])
        projection_count += int(value["projection_coordinate_count"])
        exact_replay_count += int(value["exact_replay_count"])
    v1_config = v1m.load_config(Path(str(config.v1_config["path"])))
    d_pose = pose_sum / PAIR_COUNT
    d_seg = seg_errors / (PAIR_COUNT * SCORER_HEIGHT * SCORER_WIDTH)
    if round(d_pose, 8) != round(v1_config.global_mean_pose_dist, 8) or round(d_seg, 8) != round(
        v1_config.global_mean_seg_dist, 8
    ):
        raise ExactCoarseCostateV2Error("V2 full-n600 authority base components do not reproduce G85")
    body = {
        "schema": AGGREGATE_SCHEMA,
        "pair_range": [0, PAIR_COUNT],
        "stages": stages,
        "projection_coordinate_count": projection_count,
        "exact_replay_count": exact_replay_count,
        "base_row": {
            "d_pose": d_pose,
            "d_seg": d_seg,
            "archive_bytes": v1_config.g85_archive["bytes"],
            "archive_sha256": v1_config.g85_archive["sha256"],
            "exact_g85_components_reproduced_to_reported_precision": True,
        },
        "authority_drift": {
            "differentiable_current_argmax_drift_cells": current_drift_cells,
            "differentiable_target_argmax_drift_cells": target_drift_cells,
            "inference_cells_remain_authoritative": True,
        },
        "exact_replay_policy": "ALL_DETERMINISTIC_PHYSICAL_GROUPS",
        "pareto_pruning_performed": False,
        "hierarchical_refinement_required_before_atom_selection": True,
        "rate_axis": "UNMEASURED_UNTIL_G94_COMPOSES_ACTUAL_ZIP",
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
        "research_only": True,
        "encoder_only": True,
    }
    receipt = _seal(body, field="aggregate_receipt_sha256")
    path = config.output_root / "aggregate_receipt.json"
    _atomic_write_json(path, receipt)
    return path


def _immutable_resume_frontier(
    root: Path,
    *,
    expected_group_ids_by_range: dict[
        tuple[int, int],
        tuple[str, ...],
    ],
) -> dict[str, Any]:
    sealed = []
    for stage_index in range(STAGE_COUNT):
        stage_start = stage_index * STAGE_PAIRS
        stage_stop = stage_start + STAGE_PAIRS
        stage_dir = root / f"20_stage_{stage_index:02d}_{stage_start:04d}_{stage_stop:04d}"
        for pair_start in range(stage_start, stage_stop, MAX_BATCH_PAIRS):
            pair_stop = min(pair_start + MAX_BATCH_PAIRS, stage_stop)
            path = stage_dir / "batches" / f"batch_{pair_start:04d}_{pair_stop:04d}.json"
            if not path.is_file():
                return {
                    "sealed_batch_count": len(sealed),
                    "sealed_batches": sealed,
                    "next_pair_range": [pair_start, pair_stop],
                }
            checkpoint = _load_json(path, label="G90 V2 resume-frontier batch")
            pair_range = (pair_start, pair_stop)
            if pair_range not in expected_group_ids_by_range:
                raise ExactCoarseCostateV2Error(f"G90 V2 resume-frontier lacks rederived physical groups: {path}")
            expected_self = _validate_batch_checkpoint(
                checkpoint,
                pair_start=pair_start,
                pair_stop=pair_stop,
                expected_group_ids=expected_group_ids_by_range[pair_range],
            )
            sealed.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                    "batch_checkpoint_sha256": expected_self,
                    "pair_range": [pair_start, pair_stop],
                }
            )
    return {
        "sealed_batch_count": len(sealed),
        "sealed_batches": sealed,
        "next_pair_range": None,
    }


def _write_blocker(config: ConfigV2, exc: BaseException) -> Path:
    try:
        reopened = preflight(config)
        expected_group_ids = _expected_group_ids_by_range(reopened)
        immutable_resume_frontier = _immutable_resume_frontier(
            config.output_root,
            expected_group_ids_by_range=expected_group_ids,
        )
        immutable_resume_frontier_validation_error = None
    except Exception as frontier_exc:
        immutable_resume_frontier = {
            "sealed_batch_count": None,
            "sealed_batches": None,
            "next_pair_range": None,
            "validation_completed": False,
        }
        immutable_resume_frontier_validation_error = {
            "exception_type": type(frontier_exc).__name__,
            "exception_message": str(frontier_exc),
        }
    body = {
        "schema": BLOCKER_SCHEMA,
        "config_path": str(config.path),
        "output_root": str(config.output_root),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "exception_context": getattr(exc, "context", {}),
        "next_stage": _next_incomplete_stage(config.output_root),
        "immutable_resume_frontier": immutable_resume_frontier,
        "immutable_resume_frontier_validation_error": (immutable_resume_frontier_validation_error),
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
        "research_only": True,
        "encoder_only": True,
    }
    receipt = _seal(body, field="blocker_receipt_sha256")
    path = config.output_root / f"blocker_receipt_{receipt['blocker_receipt_sha256'][:12]}.json"
    _atomic_write_json(path, receipt)
    return path


def run_next_stage(config: ConfigV2) -> dict[str, object]:
    v1m._configure_determinism(v1m.load_config(Path(str(config.v1_config["path"]))))
    reopened = preflight(config)
    stage_index = _next_incomplete_stage(config.output_root)
    if stage_index is None:
        aggregate = _write_aggregate_if_complete(config)
        return {"status": "already_complete", "aggregate_receipt": str(aggregate)}
    stage_path = _run_stage(config, reopened, stage_index)
    aggregate = _write_aggregate_if_complete(config)
    return {
        "status": "stage_complete",
        "stage_index": stage_index,
        "stage_receipt": str(stage_path),
        "aggregate_receipt": None if aggregate is None else str(aggregate),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--materialize-next-stage", action="store_true")
    mode.add_argument("--status", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.status:
        print(
            json.dumps(
                {
                    "status": ("complete" if _next_incomplete_stage(config.output_root) is None else "incomplete"),
                    "next_stage": _next_incomplete_stage(config.output_root),
                    "output_root": str(config.output_root),
                },
                sort_keys=True,
            )
        )
        return 0
    try:
        if args.preflight:
            reopened = preflight(config)
            print(
                json.dumps(
                    {
                        "status": "preflight_complete_launch_not_executed",
                        "preflight": str(config.output_root / "00_preflight_receipt.json"),
                        "preflight_sha256": reopened["receipt"]["preflight_sha256"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        result = run_next_stage(config)
    except Exception as exc:
        blocker = _write_blocker(config, exc)
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blocker_receipt": str(blocker),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
