#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run Probe B on one post-solve camera-resolution exact-R common master.

The runner is deliberately stage-resumable.  An invocation advances exactly
one pricing round, one bounded v12 rebaseline chunk, or one final n600 replay
stage.  Every scorer batch and accepted archive is immutable and preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for _path in (SRC, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_column_generation import (  # noqa: E402
    ExactReplay,
    PricedColumn,
    RestrictedMasterDuals,
    exact_replay_beam_select,
    price_columns,
    solve_restricted_master_lp,
)
from tac.optimization.ddm_v18_common_exact_r_master import (  # noqa: E402
    common_exact_r_byte_rows,
    compile_common_exact_r_master,
    receive_common_exact_r_master,
)
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    BoundaryShearletAtomV1,
    IslandShapeAtomV1,
    RowBandScorerTemplateV1,
    ScorerSolvedTemplateBankV1,
    parse_carrier_compose_archive,
    receive_carrier_compose_archive,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_coupled_margin import (  # noqa: E402
    receive_coupled_margin_archive,
)
from tac.optimization.direct_description_minimizer import (  # noqa: E402
    SOURCE_BYTES,
    DirectDescriptionError,
    _read_regular_file_once,
)
from tac.optimization.predictor_upgrade_xi_chart import LaneCoefficientDelta  # noqa: E402
from tools.measure_ddm_v14_realization_fidelity import _forward, _load_models  # noqa: E402
from tools.measure_ddm_v15_scorer_solved_templates import (  # noqa: E402
    DDMV15ScorerSolvedTemplateConfigV1,
)
from tools.probe_ddm_a1_column_generated_correction import (  # noqa: E402
    EVIDENCE_AXIS,
    FIXED_BUDGETS,
    FIXED_CODER_ENTRANTS,
    FIXED_FAMILIES,
    FIXED_SELECTORS,
    LANE_ID,
    POINTER,
    DDMA1ColumnGeneratedCorrectionConfigV1,
)
from tools.run_ddm_v9_carrier_compose import (  # noqa: E402
    _compile_obligation_candidates,
    _SearchCandidate,
)

SCHEMA: Final = "ddm_v18b_common_master_pricing_receipt.v1"
CHECKPOINT_SCHEMA: Final = "ddm_v18b_common_master_pricing_checkpoint.v1"
SOURCE_STAGE: Final = "00_common_master_source_closure.json"
SCREEN_START: Final = 448
SCREEN_COUNT: Final = 64
SCORER_BATCH: Final = 16
MAX_COLUMNS_PER_ROUND: Final = 64
BEAM_FINALISTS: Final = 6
LEGACY_SCORER_GRID_DSEG: Final = 0.034003668891
V12_CHECKPOINT_ROOT: Final = ".omx/research/ddm_v12_obligation_n600_20260722T161517Z/stage_checkpoints"
PERMUTATION_GAUGE_CODER_ENTRANT: Final = "permutation_gauge_canonical_address_order"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DirectDescriptionError("git SHA is unavailable for producer custody") from exc


def _producer_custody() -> list[dict[str, Any]]:
    paths = (
        Path(__file__),
        REPO_ROOT / "tools/probe_ddm_a1_column_generated_correction.py",
        REPO_ROOT / "tools/run_ddm_v9_carrier_compose.py",
        REPO_ROOT / "tools/measure_ddm_v14_realization_fidelity.py",
        REPO_ROOT / "src/tac/optimization/ddm_v18_common_exact_r_master.py",
        REPO_ROOT / "src/tac/optimization/ddm_column_generation.py",
        REPO_ROOT / "src/tac/optimization/direct_description_carrier_compose.py",
    )
    return [
        {
            "path": _portable(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(_read_regular_file_once(path)),
        }
        for path in paths
    ]


def _bound_bytes(path: Path, digest: str, name: str) -> bytes:
    payload = _read_regular_file_once(path)
    if _sha256(payload) != digest:
        raise DirectDescriptionError(f"{name} SHA-256 differs")
    return payload


def _bound_json(path: Path, digest: str, name: str) -> dict[str, Any]:
    try:
        value = json.loads(_bound_bytes(path, digest, name))
    except json.JSONDecodeError as exc:
        raise DirectDescriptionError(f"{name} is malformed JSON") from exc
    if not isinstance(value, dict):
        raise DirectDescriptionError(f"{name} must contain one JSON object")
    return value


def _publish_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_regular_file_once(path) != payload:
            raise DirectDescriptionError(f"immutable output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _publish_immutable(path, rfc8785_canonicalize(dict(value)))


def _receiver_output_noop_rejection(
    *,
    error: DirectDescriptionError,
    config_hash: str,
    candidate_index: int,
    spec: ColumnSpec,
    current_archive: bytes,
    accepted_count: int,
) -> dict[str, Any]:
    """Turn only an exact receiver-output no-op into a resumable rejection."""

    message = str(error)
    if "receiver-output no-op" not in message:
        raise error
    candidate = spec.row()
    return {
        "schema": CHECKPOINT_SCHEMA,
        "stage": "v12_common_exact_r_sequential_greedy",
        "typed_config_sha256": config_hash,
        "candidate_index": candidate_index,
        "candidate": candidate,
        "candidate_inventory_row_sha256": _sha256(rfc8785_canonicalize(candidate)),
        "exact_archive_before": {
            "bytes": len(current_archive),
            "sha256": _sha256(current_archive),
        },
        "exact_receiver_compile_refusal": message,
        "failure_evidence": {
            "classification": "automatic_exact_receiver_compile_refusal",
            "runner_git_sha": _git_sha(),
        },
        "admitted": False,
        "reason": "exact_receiver_parseback_refused_output_noop",
        "accepted_batch_updates": {},
        "accepted_count_after": accepted_count,
        "verdict_scope": (
            "this exact fixed-pool bundle is a receiver-output no-op; "
            "no broader family conclusion"
        ),
        "score_claim": False,
    }


def _permutation_gauge_coder_row(
    archive: bytes,
    selected_column_ids: Sequence[str],
) -> dict[str, Any]:
    """Report the ordering-gauge race on the byte-canonical typed atom stream."""

    selected_count = len(selected_column_ids)
    empty = selected_count == 0
    return {
        "status": (
            "MEASURED_TRIVIAL_EMPTY_SELECTED_PAYLOAD"
            if empty
            else "ALREADY_CANONICAL_BY_CONSTRUCTION_NO_DISTINCT_AS_IS_ENCODING"
        ),
        "measured": empty,
        "as_is_archive_bytes": len(archive),
        "canonical_archive_bytes": len(archive) if empty else None,
        "canonical_minus_as_is_bytes": 0 if empty else None,
        "archive_sha256": _sha256(archive),
        "selected_column_count": selected_count,
        "candidate_pool_is_counted_payload": False,
        "ordering_rule": (
            "lane, shearlet, and island atoms are sorted by typed address before encoding"
        ),
        "training_or_merging_claim": False,
        "admitted_to_equal_byte_table": empty,
    }


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read_regular_file_once(path))
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError(f"malformed JSON checkpoint: {path}") from exc
    if not isinstance(value, dict):
        raise DirectDescriptionError(f"checkpoint must contain one object: {path}")
    return value


def _require_checkpoint_identity(
    row: Mapping[str, Any],
    *,
    config_hash: str,
    stage: str,
) -> None:
    if row.get("schema") != CHECKPOINT_SCHEMA or row.get("stage") != stage:
        raise DirectDescriptionError(f"{stage} checkpoint schema/stage differs")
    if row.get("typed_config_sha256") != config_hash:
        raise DirectDescriptionError(f"{stage} checkpoint typed config differs")


@dataclass(frozen=True, slots=True)
class BatchScore:
    errors: int
    sites: int
    pose_sse: float
    pose_coordinates: int
    cells_sha256: str
    pose_sha256: str

    def row(self) -> dict[str, Any]:
        return {
            "errors": self.errors,
            "sites": self.sites,
            "pose_squared_error_sum": f"{self.pose_sse:.12f}",
            "pose_coordinates": self.pose_coordinates,
            "cells_sha256": self.cells_sha256,
            "pose_sha256": self.pose_sha256,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> BatchScore:
        return cls(
            int(row["errors"]),
            int(row["sites"]),
            float(row["pose_squared_error_sum"]),
            int(row["pose_coordinates"]),
            str(row["cells_sha256"]),
            str(row["pose_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    column_id: str
    family: str
    priority: float
    candidate: _SearchCandidate | None = None
    template_index: int | None = None
    template: RowBandScorerTemplateV1 | None = None

    def conflict_keys(self) -> tuple[str, ...]:
        if self.template_index is not None:
            return (f"template:{self.template_index}",)
        if self.candidate is None:
            raise DirectDescriptionError("column has neither correction nor template DOF")
        return tuple(sorted(repr(value) for value in self.candidate.conflict_keys()))

    def source_pair_ids(self) -> tuple[int, ...]:
        if self.candidate is None:
            return tuple(range(SCREEN_START, SCREEN_START + SCREEN_COUNT))
        return self.candidate.source_pair_ids

    def row(self) -> dict[str, Any]:
        if self.template_index is not None:
            origin = "v16_source_jacobian_template_endpoint_exactly_replayed"
        elif self.family == "g1_grammar_coordinate":
            origin = "v12_residual_ranked_g1_natural_lane_coordinate"
        elif self.family == "curvelet_boundary":
            origin = "v12_residual_ranked_governed_shearlet_coordinate"
        elif self.family == "realized_residual_vjp":
            origin = "v12_residual_ranked_island_coordinate"
        else:
            origin = "v12_fixed_pool_bundle_control"
        template_row = None
        if self.template is not None:
            template_row = {
                "role": self.template.role,
                "application": self.template.application,
                "scorer_row_start": self.template.scorer_row_start,
                "scorer_row_stop": self.template.scorer_row_stop,
                "patch_height": self.template.patch_height,
                "patch_width": self.template.patch_width,
                "rgb_u8_hex": self.template.rgb_u8.hex(),
            }
        return {
            "column_id": self.column_id,
            "family": self.family,
            "origin": origin,
            "priority": f"{self.priority:.12e}",
            "candidate": None if self.candidate is None else self.candidate.row(),
            "template_index": self.template_index,
            "template": template_row,
            "conflict_keys": list(self.conflict_keys()),
        }


def _candidate_from_row(row: Mapping[str, Any]) -> _SearchCandidate:
    return _SearchCandidate(
        str(row["candidate_id"]),
        str(row["mechanism"]),
        float(row["fisher_priority"]),
        tuple(int(value) for value in row["source_pair_ids"]),
        lane_symbols=tuple(LaneCoefficientDelta(**value) for value in row["lane_symbols"]),
        boundary_shearlets=tuple(BoundaryShearletAtomV1(**value) for value in row["boundary_shearlets"]),
        island_shapes=tuple(IslandShapeAtomV1(**value) for value in row["island_shapes"]),
    )


def _atomic_columns(
    bundle_rows: Sequence[Mapping[str, Any]],
    v16_bank: ScorerSolvedTemplateBankV1,
) -> tuple[ColumnSpec, ...]:
    result: list[ColumnSpec] = []
    for bundle_row in bundle_rows:
        bundle = _candidate_from_row(bundle_row["candidate"])
        atom_total = len(bundle.lane_symbols) + len(bundle.boundary_shearlets) + len(bundle.island_shapes)
        priority = bundle.fisher_priority / max(1, atom_total)
        for index, symbol in enumerate(bundle.lane_symbols):
            candidate = _SearchCandidate(
                f"{bundle.candidate_id}:lane:{index:02d}",
                bundle.mechanism,
                priority,
                (symbol.pair_index,),
                lane_symbols=(symbol,),
            )
            result.append(ColumnSpec(candidate.candidate_id, "g1_grammar_coordinate", priority, candidate))
        for index, atom in enumerate(bundle.boundary_shearlets):
            candidate = _SearchCandidate(
                f"{bundle.candidate_id}:shearlet:{index:02d}",
                bundle.mechanism,
                priority,
                (atom.pair_index,),
                boundary_shearlets=(atom,),
            )
            result.append(ColumnSpec(candidate.candidate_id, "curvelet_boundary", priority, candidate))
        for index, atom in enumerate(bundle.island_shapes):
            candidate = _SearchCandidate(
                f"{bundle.candidate_id}:island:{index:02d}",
                bundle.mechanism,
                priority,
                tuple(range(atom.pair_index, atom.pair_index + atom.lifetime)),
                island_shapes=(atom,),
            )
            result.append(ColumnSpec(candidate.candidate_id, "realized_residual_vjp", priority, candidate))
    maximum = max((row.priority for row in result), default=1.0)
    for index, template in enumerate(v16_bank.templates):
        result.append(
            ColumnSpec(
                f"v16_template_dof_{index:02d}",
                "realized_residual_vjp",
                maximum * (1.0 - index * 1.0e-9),
                template_index=index,
                template=template,
            )
        )
    return tuple(sorted(result, key=lambda row: (-row.priority, row.column_id)))


def _bundle_columns(bundle_rows: Sequence[Mapping[str, Any]]) -> tuple[ColumnSpec, ...]:
    return tuple(
        ColumnSpec(
            str(row["candidate"]["candidate_id"]),
            "v12_control",
            float(row["candidate"]["fisher_priority"]),
            _candidate_from_row(row["candidate"]),
        )
        for row in bundle_rows
    )


def _load_bundle_rows() -> tuple[dict[str, Any], ...]:
    root = REPO_ROOT / V12_CHECKPOINT_ROOT / "candidates"
    paths = sorted(root.glob("*.json"))
    rows = tuple(_json(path) for path in paths)
    if len(rows) != 353 or sum(int(row["atomic_obligation_count"]) for row in rows) != 4096:
        raise DirectDescriptionError("v12 fixed 4096-atom/353-bundle inventory custody differs")
    if [int(row["candidate_index"]) for row in rows] != list(range(len(rows))):
        raise DirectDescriptionError("v12 fixed-pool bundle order differs")
    candidates = [_candidate_from_row(row["candidate"]) for row in rows]
    inventory = _json(root.parent / "01_obligation_inventory.json")
    inventory_sha256 = inventory.get("candidate_inventory_sha256")
    if (
        not isinstance(inventory_sha256, str)
        or len(inventory_sha256) != 64
        or inventory.get("measured_bundle_count") != 353
        or inventory.get("atoms_in_measured_bundles") != 4096
        or {row.get("candidate_inventory_sha256") for row in rows} != {inventory_sha256}
    ):
        raise DirectDescriptionError("v12 fixed-pool inventory SHA-256 custody differs")
    if any(
        row.get("candidate_fingerprint") != candidate.fingerprint()
        for row, candidate in zip(rows, candidates, strict=True)
    ):
        raise DirectDescriptionError("v12 fixed-pool candidate fingerprint differs")
    return rows


def _source_state(
    config: DDMA1ColumnGeneratedCorrectionConfigV1,
) -> tuple[
    bytes,
    bytes,
    ScorerSolvedTemplateBankV1,
    ScorerSolvedTemplateBankV1,
    DDMV15ScorerSolvedTemplateConfigV1,
    dict[str, Any],
]:
    v12 = _bound_json(
        REPO_ROOT / config.source_v12_receipt,
        config.source_v12_receipt_sha256,
        "v12 receipt",
    )
    v12_base_row = next(row for row in v12["ladder"] if row["requested_added_budget_bytes"] == 0)
    v12_base = _bound_bytes(
        REPO_ROOT / v12_base_row["archive"]["path"],
        v12_base_row["archive"]["sha256"],
        "v12 zero-budget archive",
    )
    v12_max = _bound_bytes(
        REPO_ROOT / config.source_v12_archive,
        config.source_v12_archive_sha256,
        "v12 fixed-pool max-rung archive",
    )
    grammar = _bound_json(
        REPO_ROOT / config.grammar_receipt,
        config.grammar_receipt_sha256,
        "G1 grammar receipt",
    )
    v15 = _bound_json(
        REPO_ROOT / config.source_v15_receipt,
        config.source_v15_receipt_sha256,
        "v15 receipt",
    )
    v15_archive = _bound_bytes(
        REPO_ROOT / config.source_v15_archive,
        config.source_v15_archive_sha256,
        "v15 archive",
    )
    v15_receiver = receive_carrier_compose_archive(v15_archive)
    if v15_receiver.realization_profile is None or v15_receiver.scorer_solved_templates is None:
        raise DirectDescriptionError("v15 source lost realization/template payloads")
    v16 = _bound_json(
        REPO_ROOT / config.source_v16_receipt,
        config.source_v16_receipt_sha256,
        "v16 receipt",
    )
    n600_rows = [row for row in v16.get("receiver_realized_ladder", ()) if row.get("rung") == "n600"]
    if len(n600_rows) != 1:
        raise DirectDescriptionError("v16 receipt must bind exactly one n600 receiver rung")
    v16_archive_row = n600_rows[0].get("archive")
    if not isinstance(v16_archive_row, Mapping):
        raise DirectDescriptionError("v16 n600 receiver rung lost archive custody")
    v16_archive = _bound_bytes(
        REPO_ROOT / str(v16_archive_row["path"]),
        str(v16_archive_row["sha256"]),
        "v16 n600 archive",
    )
    v16_bank = receive_coupled_margin_archive(v16_archive).base.scorer_solved_templates
    if v16_bank is None or len(v16_bank.templates) != len(v15_receiver.scorer_solved_templates.templates):
        raise DirectDescriptionError("v16 template DOF bank differs in cardinality")
    common_base = compile_common_exact_r_master(
        v12_base,
        v15_receiver.realization_profile,
        v15_receiver.scorer_solved_templates,
    )
    if _sha256(v12_max) != config.source_v12_archive_sha256:
        raise DirectDescriptionError("v12 max-rung archive binding disappeared")
    if grammar.get("schema") != "ddm_g1_grammar_induction_compact_receipt.v1":
        raise DirectDescriptionError("G1 grammar receipt schema differs")
    pose_tube_dpose_radius = float(v12["typed_config"]["pose_tube_dpose_radius"])
    if not math.isfinite(pose_tube_dpose_radius) or pose_tube_dpose_radius < 0.0:
        raise DirectDescriptionError("v12 pose-tube radius is invalid")
    v15_cfg = DDMV15ScorerSolvedTemplateConfigV1.model_validate(v15["typed_config"])
    return (
        v12_base,
        common_base,
        v15_receiver.scorer_solved_templates,
        v16_bank,
        v15_cfg,
        {
            "v12_receipt_schema": v12.get("schema"),
            "v12_max_archive_sha256": _sha256(v12_max),
            "grammar_receipt_schema": grammar.get("schema"),
            "v16_receipt_schema": v16.get("schema"),
            "v16_n600_archive_sha256": _sha256(v16_archive),
            "v12_pose_tube_dpose_radius": pose_tube_dpose_radius,
        },
    )


def _compile_state(
    *,
    predictor_archive: bytes,
    base_bank: ScorerSolvedTemplateBankV1,
    specs: Sequence[ColumnSpec],
    profile: Any,
) -> bytes:
    corrections = [row.candidate for row in specs if row.candidate is not None]
    inner, _receiver = _compile_obligation_candidates(
        predictor_archive,
        [row for row in corrections if row is not None],
    )
    templates = list(base_bank.templates)
    for spec in specs:
        if spec.template_index is not None:
            if spec.template is None:
                raise DirectDescriptionError("template column has no replacement")
            templates[spec.template_index] = spec.template
    bank = ScorerSolvedTemplateBankV1(tuple(templates))
    return compile_common_exact_r_master(inner, profile, bank)


def _score_batch(
    *,
    receiver: Any,
    local_ids: tuple[int, ...],
    source_ids: np.ndarray,
    labels: np.ndarray,
    poses: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> BatchScore:
    camera = receiver.render_camera_pairs(local_ids)
    cells, pose6 = _forward(segnet, posenet, camera)
    target = np.asarray(labels[source_ids])
    target_pose = np.asarray(poses[source_ids])
    return BatchScore(
        errors=int(np.count_nonzero(cells != target)),
        sites=int(cells.size),
        pose_sse=float(np.square(pose6 - target_pose).sum(dtype=np.float64)),
        pose_coordinates=int(pose6.size),
        cells_sha256=_sha256(np.ascontiguousarray(cells).tobytes()),
        pose_sha256=_sha256(np.ascontiguousarray(pose6).tobytes()),
    )


def _batch_starts_for(
    specs: Sequence[ColumnSpec],
    *,
    pair_start: int,
    pair_count: int,
) -> tuple[int, ...]:
    starts: set[int] = set()
    for spec in specs:
        if spec.template_index is not None:
            return tuple(range(0, pair_count, SCORER_BATCH))
        for source_id in spec.source_pair_ids():
            if pair_start <= source_id < pair_start + pair_count:
                starts.add(((source_id - pair_start) // SCORER_BATCH) * SCORER_BATCH)
    return tuple(sorted(starts))


def _score_starts(
    *,
    archive: bytes,
    starts: Sequence[int],
    pair_start: int,
    pair_count: int,
    labels: np.ndarray,
    poses: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> dict[int, BatchScore]:
    receiver = receive_common_exact_r_master(archive)
    result = {}
    for start in starts:
        stop = min(pair_count, start + SCORER_BATCH)
        local_ids = tuple(range(start, stop))
        source_ids = pair_start + np.arange(start, stop, dtype=np.int64)
        result[start] = _score_batch(
            receiver=receiver,
            local_ids=local_ids,
            source_ids=source_ids,
            labels=labels,
            poses=poses,
            segnet=segnet,
            posenet=posenet,
        )
    return result


def _totals(rows: Mapping[int, BatchScore]) -> tuple[int, int, float, int]:
    return (
        sum(row.errors for row in rows.values()),
        sum(row.sites for row in rows.values()),
        sum(row.pose_sse for row in rows.values()),
        sum(row.pose_coordinates for row in rows.values()),
    )


def _measurement(archive: bytes, rows: Mapping[int, BatchScore]) -> dict[str, Any]:
    errors, sites, pose_sse, pose_coordinates = _totals(rows)
    dseg = errors / sites
    dpose = pose_sse / pose_coordinates
    objective = 100.0 * dseg + math.sqrt(10.0 * dpose) + 25.0 * len(archive) / SOURCE_BYTES
    return {
        "archive_bytes": len(archive),
        "archive_sha256": _sha256(archive),
        "errors": errors,
        "sites": sites,
        "d_seg": f"{dseg:.12f}",
        "pose_squared_error_sum": f"{pose_sse:.12f}",
        "pose_coordinates": pose_coordinates,
        "d_pose": f"{dpose:.12f}",
        "objective": f"{objective:.12f}",
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
    }


def _load_or_score_window(
    *,
    path: Path,
    archive: bytes,
    pair_start: int,
    pair_count: int,
    labels: np.ndarray,
    poses: np.ndarray,
    segnet: Any,
    posenet: Any,
    config_hash: str,
) -> dict[int, BatchScore]:
    if path.exists():
        row = _json(path)
        _require_checkpoint_identity(
            row,
            config_hash=config_hash,
            stage="exact_window_measurement",
        )
        if (
            row.get("archive_sha256") != _sha256(archive)
            or row.get("pair_start") != pair_start
            or row.get("pair_count") != pair_count
        ):
            raise DirectDescriptionError("window checkpoint custody differs")
        return {int(key): BatchScore.from_row(value) for key, value in row["batches"].items()}
    starts = tuple(range(0, pair_count, SCORER_BATCH))
    rows = _score_starts(
        archive=archive,
        starts=starts,
        pair_start=pair_start,
        pair_count=pair_count,
        labels=labels,
        poses=poses,
        segnet=segnet,
        posenet=posenet,
    )
    _write_json(
        path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "stage": "exact_window_measurement",
            "typed_config_sha256": config_hash,
            "archive_sha256": _sha256(archive),
            "pair_start": pair_start,
            "pair_count": pair_count,
            "batches": {str(key): value.row() for key, value in rows.items()},
            "measurement": _measurement(archive, rows),
            "score_claim": False,
        },
    )
    return rows


def _preserve_reused_window(
    *,
    path: Path,
    archive: bytes,
    pair_start: int,
    pair_count: int,
    rows: Mapping[int, BatchScore],
    config_hash: str,
    reused_from: str,
) -> None:
    _write_json(
        path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "stage": "exact_window_measurement",
            "typed_config_sha256": config_hash,
            "archive_sha256": _sha256(archive),
            "pair_start": pair_start,
            "pair_count": pair_count,
            "batches": {str(key): value.row() for key, value in rows.items()},
            "measurement": _measurement(archive, rows),
            "identical_archive_measurement_reused_from": reused_from,
            "score_claim": False,
        },
    )


def _replace_batches(
    current: Mapping[int, BatchScore],
    update: Mapping[int, BatchScore],
) -> dict[int, BatchScore]:
    result = dict(current)
    result.update(update)
    return result


def _objective_delta(
    before_archive: bytes,
    before: Mapping[int, BatchScore],
    after_archive: bytes,
    after: Mapping[int, BatchScore],
) -> float:
    return float(_measurement(after_archive, after)["objective"]) - float(
        _measurement(before_archive, before)["objective"]
    )


def _source_resume_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable source identity without live preflight telemetry."""

    identity = dict(payload)
    storage = dict(identity.get("storage_preflight", {}))
    storage.pop("free_bytes", None)
    identity["storage_preflight"] = storage
    return identity


def _source_checkpoint(
    *,
    root: Path,
    config: DDMA1ColumnGeneratedCorrectionConfigV1,
    common_base: bytes,
    bundle_rows: Sequence[Mapping[str, Any]],
    atomic: Sequence[ColumnSpec],
    source_custody: Mapping[str, Any],
) -> Path:
    path = root / "stage_checkpoints" / SOURCE_STAGE
    storage = shutil.disk_usage(root.parent if root.parent.exists() else REPO_ROOT)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "stage": "common_exact_r_source_closure",
        "typed_config_sha256": config.typed_config_hash(),
        "common_master": {
            "bytes": len(common_base),
            "sha256": _sha256(common_base),
            "receiver_custody": dict(receive_common_exact_r_master(common_base).custody),
            "byte_streams": common_exact_r_byte_rows(common_base),
        },
        "v12_fixed_pool": {
            "atomic_count": sum(int(row["atomic_obligation_count"]) for row in bundle_rows),
            "bundle_count": len(bundle_rows),
            "inventory_sha256": _sha256(rfc8785_canonicalize([row["candidate"] for row in bundle_rows])),
        },
        "screen_generated_vocabulary": {
            "column_count": len(atomic),
            "n64_column_count": sum(
                row.template_index is not None
                or all(SCREEN_START <= value < SCREEN_START + SCREEN_COUNT for value in row.source_pair_ids())
                for row in atomic
            ),
            "families": list(FIXED_FAMILIES),
            "selection_modes": list(FIXED_SELECTORS),
            "pricing_geometry": {
                "linear_objective_coefficient": ("receiver-realized singleton joint-objective delta"),
                "source_jacobian_use": (
                    "v16 template endpoints enter only as candidate coordinates; "
                    "no frozen-Jacobian predicted cost is consumed"
                ),
                "gate_refreshed_conditional_jacobian": ("NOT_APPLICABLE_EXACT_SINGLETON_REPLAY_IS_PRICING_AUTHORITY"),
                "operator_nested_jacobian_directive_consumed": "2026-07-23T04:10:16Z",
            },
        },
        "source_custody": dict(source_custody),
        "git_sha": _git_sha(),
        "producer_custody": _producer_custody(),
        "storage_preflight": {
            "free_bytes": storage.free,
            "memory_ceiling_gib": config.memory_ceiling_gib,
            "required_free_bytes": 8 * 1024**3,
            "pass": storage.free >= 8 * 1024**3,
            "large_artifacts_created": False,
        },
        "postsolve_only": True,
        "predict_productions_present": False,
        "score_claim": False,
    }
    if not payload["storage_preflight"]["pass"]:
        raise DirectDescriptionError("v18b storage preflight failed")
    if path.exists():
        existing = _json(path)
        _require_checkpoint_identity(
            existing,
            config_hash=config.typed_config_hash(),
            stage="common_exact_r_source_closure",
        )
        if rfc8785_canonicalize(_source_resume_identity(existing)) != rfc8785_canonicalize(
            _source_resume_identity(payload)
        ):
            raise DirectDescriptionError("common exact-R source checkpoint custody differs")
        return path
    _write_json(path, payload)
    return path


def _scorer_checkpoint(
    *,
    root: Path,
    config: DDMA1ColumnGeneratedCorrectionConfigV1,
    v15_config: DDMV15ScorerSolvedTemplateConfigV1,
    scorer_custody: Mapping[str, Any],
) -> Path:
    path = root / "stage_checkpoints" / "01_frozen_scorer_target_custody.json"
    _write_json(
        path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "stage": "frozen_scorer_target_custody",
            "typed_config_sha256": config.typed_config_hash(),
            "git_sha": _git_sha(),
            "producer_custody": _producer_custody(),
            "scorer_custody": dict(scorer_custody),
            "target_custody": {
                "path": v15_config.target_cache_path,
                "bytes": v15_config.target_cache_bytes,
                "sha256": v15_config.target_cache_sha256,
                "binding": "SHA-bound v15 receipt and typed config",
            },
            "score_claim": False,
        },
    )
    return path


def _round_candidates(
    atomic: Sequence[ColumnSpec],
    *,
    seen: set[str],
    occupied: set[str],
) -> tuple[ColumnSpec, ...]:
    eligible = [
        row
        for row in atomic
        if row.column_id not in seen
        and not occupied.intersection(row.conflict_keys())
        and (
            row.template_index is not None
            or all(SCREEN_START <= value < SCREEN_START + SCREEN_COUNT for value in row.source_pair_ids())
        )
    ]
    chosen: list[ColumnSpec] = []
    families: set[str] = set()
    for row in eligible:
        if row.family not in families:
            chosen.append(row)
            families.add(row.family)
    for row in eligible:
        if row not in chosen:
            chosen.append(row)
        if len(chosen) >= MAX_COLUMNS_PER_ROUND:
            break
    return tuple(chosen[:MAX_COLUMNS_PER_ROUND])


def _miqp_diagonal_proposal(
    columns: Sequence[PricedColumn],
    *,
    added_byte_budget: int,
) -> tuple[str, ...]:
    """Solve the preregistered conflict MIQP with measured diagonal Q only.

    The diagonal quadratic term is exactly the singleton objective coefficient
    on binary variables (x^2=x).  Unmeasured pair interactions stay zero and
    cannot decide acceptance; exact receiver replay does.
    """

    if not columns:
        return ()
    from scipy.optimize import Bounds, LinearConstraint, milp

    keys = sorted({key for row in columns for key in row.conflict_keys})
    matrix = [[float(row.real_coder_bytes) for row in columns]]
    upper = [float(added_byte_budget)]
    for key in keys:
        matrix.append([1.0 if key in row.conflict_keys else 0.0 for row in columns])
        upper.append(1.0)
    result = milp(
        c=np.asarray([row.linear_objective_delta for row in columns], dtype=np.float64),
        integrality=np.ones(len(columns), dtype=np.int8),
        bounds=Bounds(np.zeros(len(columns)), np.ones(len(columns))),
        constraints=LinearConstraint(
            np.asarray(matrix, dtype=np.float64),
            -np.inf,
            np.asarray(upper, dtype=np.float64),
        ),
        options={"time_limit": 60.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise DirectDescriptionError(f"conflict MIQP failed: {result.message}")
    return tuple(sorted(row.column_id for row, value in zip(columns, result.x, strict=True) if value > 0.5))


def _largest_prefix_not_exceeding(
    archive_sizes: Sequence[int],
    *,
    base_archive_bytes: int,
    realized_added_byte_cap: int,
) -> int:
    eligible = [
        index for index, size in enumerate(archive_sizes) if size - base_archive_bytes <= realized_added_byte_cap
    ]
    if not eligible:
        raise DirectDescriptionError("empty generated prefix exceeded the common-base byte cap")
    return max(eligible)


def _formulation_falsified(
    history: Sequence[Mapping[str, Any]],
    equal: Sequence[Mapping[str, Any]],
) -> tuple[bool, bool, bool]:
    three_clean_rounds = len(history) == 3 and all(
        row.get("complete") is True and row.get("exact_pricing") is True and row.get("negative_reduced_cost_count") == 0
        for row in history
    )
    any_win = any(row.get("beats_v12") is True for row in equal)
    eligible = three_clean_rounds and len(equal) == len(FIXED_BUDGETS)
    return eligible and not any_win, three_clean_rounds, any_win


def _exact_screen_replay(
    *,
    extra_ids: tuple[str, ...],
    current_specs: Sequence[ColumnSpec],
    by_id: Mapping[str, ColumnSpec],
    predictor_archive: bytes,
    base_bank: ScorerSolvedTemplateBankV1,
    profile: Any,
    current_batches: Mapping[int, BatchScore],
    labels: np.ndarray,
    poses: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> tuple[ExactReplay, dict[int, BatchScore], bytes]:
    extras = [by_id[value] for value in extra_ids]
    archive = _compile_state(
        predictor_archive=predictor_archive,
        base_bank=base_bank,
        specs=(*current_specs, *extras),
        profile=profile,
    )
    starts = _batch_starts_for(extras, pair_start=SCREEN_START, pair_count=SCREEN_COUNT)
    update = _score_starts(
        archive=archive,
        starts=starts,
        pair_start=SCREEN_START,
        pair_count=SCREEN_COUNT,
        labels=labels,
        poses=poses,
        segnet=segnet,
        posenet=posenet,
    )
    batches = _replace_batches(current_batches, update)
    measurement = _measurement(archive, batches)
    return (
        ExactReplay(
            extra_ids,
            len(archive),
            float(measurement["d_seg"]),
            float(measurement["d_pose"]),
            float(measurement["objective"]),
            _sha256(archive),
            True,
            True,
        ),
        batches,
        archive,
    )


def _run_pricing_round(
    *,
    round_index: int,
    root: Path,
    config: DDMA1ColumnGeneratedCorrectionConfigV1,
    atomic: Sequence[ColumnSpec],
    predictor_archive: bytes,
    base_bank: ScorerSolvedTemplateBankV1,
    profile: Any,
    labels: np.ndarray,
    poses: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> Path:
    path = root / "stage_checkpoints" / f"pricing_round_{round_index:02d}.json"
    if path.exists():
        existing = _json(path)
        _require_checkpoint_identity(
            existing,
            config_hash=config.typed_config_hash(),
            stage="complete_exact_pricing_round",
        )
        if existing.get("round") != round_index or existing.get("complete") is not True:
            raise DirectDescriptionError("pricing-round checkpoint identity differs")
        return path
    prior = [_json(root / "stage_checkpoints" / f"pricing_round_{index:02d}.json") for index in range(1, round_index)]
    for index, row in enumerate(prior, start=1):
        _require_checkpoint_identity(
            row,
            config_hash=config.typed_config_hash(),
            stage="complete_exact_pricing_round",
        )
        if row.get("round") != index or row.get("complete") is not True:
            raise DirectDescriptionError("prior pricing-round checkpoint identity differs")
    selected_ids = tuple(value for row in prior for value in row["accepted_new_column_ids"])
    seen = {value for row in prior for value in row["generated_column_ids"]}
    by_id = {row.column_id: row for row in atomic}
    selected_specs = [by_id[value] for value in selected_ids]
    occupied = {key for row in selected_specs for key in row.conflict_keys()}
    candidates = _round_candidates(atomic, seen=seen, occupied=occupied)
    current_archive = _compile_state(
        predictor_archive=predictor_archive,
        base_bank=base_bank,
        specs=selected_specs,
        profile=profile,
    )
    current_batches = _load_or_score_window(
        path=root / "stage_checkpoints" / f"pricing_round_{round_index:02d}_current_n64.json",
        archive=current_archive,
        pair_start=SCREEN_START,
        pair_count=SCREEN_COUNT,
        labels=labels,
        poses=poses,
        segnet=segnet,
        posenet=posenet,
        config_hash=config.typed_config_hash(),
    )
    candidate_root = root / "stage_checkpoints" / f"pricing_round_{round_index:02d}_columns"
    priced_columns: list[PricedColumn] = []
    for index, spec in enumerate(candidates):
        checkpoint = candidate_root / f"{index:03d}.json"
        candidate_archive = _compile_state(
            predictor_archive=predictor_archive,
            base_bank=base_bank,
            specs=(*selected_specs, spec),
            profile=profile,
        )
        if checkpoint.exists():
            row = _json(checkpoint)
            _require_checkpoint_identity(
                row,
                config_hash=config.typed_config_hash(),
                stage="exact_singleton_pricing",
            )
            if (
                row.get("round") != round_index
                or row.get("column", {}).get("column_id") != spec.column_id
                or row.get("current_archive_sha256") != _sha256(current_archive)
                or row.get("candidate_archive", {}).get("sha256") != _sha256(candidate_archive)
            ):
                raise DirectDescriptionError("singleton-pricing checkpoint custody differs")
        else:
            starts = _batch_starts_for((spec,), pair_start=SCREEN_START, pair_count=SCREEN_COUNT)
            update = _score_starts(
                archive=candidate_archive,
                starts=starts,
                pair_start=SCREEN_START,
                pair_count=SCREEN_COUNT,
                labels=labels,
                poses=poses,
                segnet=segnet,
                posenet=posenet,
            )
            candidate_batches = _replace_batches(current_batches, update)
            byte_delta = len(candidate_archive) - len(current_archive)
            if byte_delta <= 0:
                raise DirectDescriptionError("priced column has nonpositive exact coder-byte delta")
            row = {
                "schema": CHECKPOINT_SCHEMA,
                "stage": "exact_singleton_pricing",
                "typed_config_sha256": config.typed_config_hash(),
                "round": round_index,
                "column": spec.row(),
                "current_archive_sha256": _sha256(current_archive),
                "candidate_archive": {
                    "bytes": len(candidate_archive),
                    "sha256": _sha256(candidate_archive),
                },
                "exact_coder_byte_delta": byte_delta,
                "exact_objective_delta": f"{_objective_delta(current_archive, current_batches, candidate_archive, candidate_batches):.12e}",
                "affected_batch_starts": list(starts),
                "measurement": _measurement(candidate_archive, candidate_batches),
                "score_claim": False,
            }
            _write_json(checkpoint, row)
        priced_columns.append(
            PricedColumn(
                spec.column_id,
                spec.family,
                int(row["exact_coder_byte_delta"]),
                float(row["exact_objective_delta"]),
                spec.conflict_keys(),
            )
        )
    previous_columns = [
        PricedColumn(
            value["column_id"],
            value["family"],
            int(value["real_coder_bytes"]),
            float(value["linear_objective_delta"]),
            tuple(value["conflict_keys"]),
        )
        for row in prior
        for value in row["accepted_priced_columns"]
    ]
    duals = (
        solve_restricted_master_lp(previous_columns, added_byte_budget=max(FIXED_BUDGETS))
        if previous_columns
        else RestrictedMasterDuals(0.0, {}, 0.0, {})
    )
    reduced = price_columns(priced_columns, duals=duals)
    negative = [row for row in reduced if row.negative_reduced_cost]
    negative_columns = [next(value for value in priced_columns if value.column_id == row.column_id) for row in negative]
    remaining = max(FIXED_BUDGETS) - (
        len(current_archive)
        - len(
            _compile_state(
                predictor_archive=predictor_archive,
                base_bank=base_bank,
                specs=(),
                profile=profile,
            )
        )
    )
    miqp_ids = _miqp_diagonal_proposal(negative_columns, added_byte_budget=max(0, remaining))
    miqp_replay, miqp_batches, miqp_archive = _exact_screen_replay(
        extra_ids=miqp_ids,
        current_specs=selected_specs,
        by_id=by_id,
        predictor_archive=predictor_archive,
        base_bank=base_bank,
        profile=profile,
        current_batches=current_batches,
        labels=labels,
        poses=poses,
        segnet=segnet,
        posenet=posenet,
    )
    finalists = negative_columns[:BEAM_FINALISTS]
    beam_replay_cache: dict[tuple[str, ...], tuple[ExactReplay, dict[int, BatchScore], bytes]] = {}

    def replay(ids: tuple[str, ...]) -> ExactReplay:
        if ids not in beam_replay_cache:
            beam_replay_cache[ids] = _exact_screen_replay(
                extra_ids=ids,
                current_specs=selected_specs,
                by_id=by_id,
                predictor_archive=predictor_archive,
                base_bank=base_bank,
                profile=profile,
                current_batches=current_batches,
                labels=labels,
                poses=poses,
                segnet=segnet,
                posenet=posenet,
            )
        return beam_replay_cache[ids][0]

    beam_best, beam_rows = exact_replay_beam_select(
        finalists,
        base_archive_bytes=len(current_archive),
        added_byte_budget=max(0, remaining),
        replay=replay,
        beam_width=32,
    )
    current_measurement = _measurement(current_archive, current_batches)
    choices = [
        ("conflict_miqp", miqp_replay, miqp_batches, miqp_archive),
        (
            "beam_width_32",
            beam_best,
            beam_replay_cache[beam_best.column_ids][1],
            beam_replay_cache[beam_best.column_ids][2],
        ),
    ]
    selector, best, best_batches, best_archive = min(
        choices,
        key=lambda value: (value[1].objective, value[1].archive_bytes, value[0]),
    )
    accepted_ids = best.column_ids if best.objective < float(current_measurement["objective"]) else ()
    admitted = bool(accepted_ids)
    if not accepted_ids:
        best_batches = dict(current_batches)
        best_archive = current_archive
    accepted_path = root / f"pricing_round_{round_index:02d}_n64.not_a_candidate.zip.receipt-bytes"
    _publish_immutable(accepted_path, best_archive)
    accepted_by_id = {row.column_id: row for row in priced_columns}
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "stage": "complete_exact_pricing_round",
        "typed_config_sha256": config.typed_config_hash(),
        "round": round_index,
        "complete": True,
        "exact_pricing": True,
        "evidence_window": {"pair_start": SCREEN_START, "pair_count": SCREEN_COUNT},
        "generated_column_count": len(candidates),
        "generated_column_ids": [row.column_id for row in candidates],
        "negative_reduced_cost_count": len(negative),
        "reduced_cost_rows": [asdict(row) for row in reduced],
        "duals": {
            "byte_marginal": duals.byte_marginal,
            "conflict_marginals": dict(duals.conflict_marginals),
            "objective": duals.objective,
        },
        "miqp": {
            "formulation": "binary diagonal-Q conflict MIQP; pair interactions unmeasured and zero",
            "proposed_ids": list(miqp_ids),
            "exact_replay": asdict(miqp_replay),
        },
        "beam": {
            "width": 32,
            "finalist_count": len(finalists),
            "exact_replay_count": len(beam_rows),
            "best": asdict(beam_best),
        },
        "selected_global_mode": selector,
        "selected_global_mode_admitted": admitted,
        "accepted_new_column_ids": list(accepted_ids),
        "accepted_priced_columns": [
            {
                **asdict(accepted_by_id[value]),
                "conflict_keys": list(accepted_by_id[value].conflict_keys),
            }
            for value in accepted_ids
        ],
        "accepted_archive": {
            "path": _portable(accepted_path),
            "bytes": len(best_archive),
            "sha256": _sha256(best_archive),
        },
        "accepted_measurement": _measurement(best_archive, best_batches),
        "exact_replay_accepted_set": True,
        "score_claim": False,
    }
    _write_json(path, payload)
    return path


def _rebaseline_progress(
    root: Path,
    bundle_columns: Sequence[ColumnSpec],
    *,
    config_hash: str,
) -> tuple[list[dict[str, Any]], list[ColumnSpec]]:
    rows = []
    accepted = []
    for index, spec in enumerate(bundle_columns):
        path = root / "stage_checkpoints" / "v12_rebaseline_candidates" / f"{index:04d}.json"
        if not path.exists():
            break
        row = _json(path)
        _require_checkpoint_identity(
            row,
            config_hash=config_hash,
            stage="v12_common_exact_r_sequential_greedy",
        )
        if row.get("candidate_index") != index or row.get("candidate", {}).get("column_id") != spec.column_id:
            raise DirectDescriptionError("v12 rebaseline candidate checkpoint identity differs")
        rows.append(row)
        if row["admitted"]:
            accepted.append(spec)
    return rows, accepted


def _run_rebaseline_chunk(
    *,
    root: Path,
    config: DDMA1ColumnGeneratedCorrectionConfigV1,
    bundle_columns: Sequence[ColumnSpec],
    predictor_archive: bytes,
    base_bank: ScorerSolvedTemplateBankV1,
    profile: Any,
    labels: np.ndarray,
    poses: np.ndarray,
    segnet: Any,
    posenet: Any,
    max_work_items: int,
    pose_tube_dpose_radius: float,
) -> bool:
    rows, accepted = _rebaseline_progress(
        root,
        bundle_columns,
        config_hash=config.typed_config_hash(),
    )
    base_archive = _compile_state(
        predictor_archive=predictor_archive,
        base_bank=base_bank,
        specs=(),
        profile=profile,
    )
    baseline = _load_or_score_window(
        path=root / "stage_checkpoints" / "v12_rebaseline_base_n600.json",
        archive=base_archive,
        pair_start=0,
        pair_count=600,
        labels=labels,
        poses=poses,
        segnet=segnet,
        posenet=posenet,
        config_hash=config.typed_config_hash(),
    )
    current_batches = dict(baseline)
    for row in rows:
        if row["admitted"]:
            current_batches.update(
                {int(key): BatchScore.from_row(value) for key, value in row["accepted_batch_updates"].items()}
            )
    current_archive = _compile_state(
        predictor_archive=predictor_archive,
        base_bank=base_bank,
        specs=accepted,
        profile=profile,
    )
    occupied = {key for value in accepted for key in value.conflict_keys()}
    baseline_dpose = float(_measurement(base_archive, baseline)["d_pose"])
    stop = min(len(bundle_columns), len(rows) + max_work_items)
    for index in range(len(rows), stop):
        spec = bundle_columns[index]
        checkpoint = root / "stage_checkpoints" / "v12_rebaseline_candidates" / f"{index:04d}.json"
        conflict = bool(occupied.intersection(spec.conflict_keys()))
        if conflict:
            row = {
                "schema": CHECKPOINT_SCHEMA,
                "stage": "v12_common_exact_r_sequential_greedy",
                "typed_config_sha256": config.typed_config_hash(),
                "candidate_index": index,
                "candidate": spec.row(),
                "admitted": False,
                "reason": "address_conflict_with_earlier_common_master_admission",
                "accepted_batch_updates": {},
                "accepted_count_after": len(accepted),
                "score_claim": False,
            }
        else:
            try:
                candidate_archive = _compile_state(
                    predictor_archive=predictor_archive,
                    base_bank=base_bank,
                    specs=(*accepted, spec),
                    profile=profile,
                )
            except DirectDescriptionError as error:
                row = _receiver_output_noop_rejection(
                    error=error,
                    config_hash=config.typed_config_hash(),
                    candidate_index=index,
                    spec=spec,
                    current_archive=current_archive,
                    accepted_count=len(accepted),
                )
                _write_json(checkpoint, row)
                continue
            starts = _batch_starts_for((spec,), pair_start=0, pair_count=600)
            update = _score_starts(
                archive=candidate_archive,
                starts=starts,
                pair_start=0,
                pair_count=600,
                labels=labels,
                poses=poses,
                segnet=segnet,
                posenet=posenet,
            )
            candidate_batches = _replace_batches(current_batches, update)
            delta = _objective_delta(
                current_archive,
                current_batches,
                candidate_archive,
                candidate_batches,
            )
            candidate_measurement = _measurement(candidate_archive, candidate_batches)
            admitted = delta < 0.0 and float(candidate_measurement["d_pose"]) <= baseline_dpose + pose_tube_dpose_radius
            row = {
                "schema": CHECKPOINT_SCHEMA,
                "stage": "v12_common_exact_r_sequential_greedy",
                "typed_config_sha256": config.typed_config_hash(),
                "candidate_index": index,
                "candidate": spec.row(),
                "exact_archive_bytes_before": len(current_archive),
                "exact_archive_bytes_after": len(candidate_archive),
                "exact_marginal_bytes": len(candidate_archive) - len(current_archive),
                "exact_objective_delta": f"{delta:.12e}",
                "measurement_after": candidate_measurement,
                "affected_batch_starts": list(starts),
                "admitted": admitted,
                "reason": (
                    "measured_joint_objective_delta_negative_inside_pose_safety_tube"
                    if admitted
                    else "measured_joint_objective_nonnegative_or_pose_safety_tube_exceeded"
                ),
                "accepted_batch_updates": (
                    {str(key): value.row() for key, value in update.items()} if admitted else {}
                ),
                "accepted_count_after": len(accepted) + int(admitted),
                "score_claim": False,
            }
            if admitted:
                accepted.append(spec)
                occupied.update(spec.conflict_keys())
                current_archive = candidate_archive
                current_batches = candidate_batches
        _write_json(checkpoint, row)
    return stop == len(bundle_columns)


def _control_rows(
    *,
    root: Path,
    config: DDMA1ColumnGeneratedCorrectionConfigV1,
    bundle_columns: Sequence[ColumnSpec],
    predictor_archive: bytes,
    base_bank: ScorerSolvedTemplateBankV1,
    profile: Any,
    labels: np.ndarray,
    poses: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> list[dict[str, Any]]:
    result_path = root / "stage_checkpoints" / "v12_rebased_controls.json"
    if result_path.exists():
        existing = _json(result_path)
        _require_checkpoint_identity(
            existing,
            config_hash=config.typed_config_hash(),
            stage="v12_common_exact_r_rebased_controls",
        )
        return list(existing["rows"])
    progress, _accepted = _rebaseline_progress(
        root,
        bundle_columns,
        config_hash=config.typed_config_hash(),
    )
    admitted = [row for row in progress if row["admitted"]]
    base_archive = _compile_state(
        predictor_archive=predictor_archive,
        base_bank=base_bank,
        specs=(),
        profile=profile,
    )
    states = [
        (
            0,
            (),
            _measurement(
                base_archive,
                _load_or_score_window(
                    path=root / "stage_checkpoints" / "v12_rebaseline_base_n600.json",
                    archive=base_archive,
                    pair_start=0,
                    pair_count=600,
                    labels=labels,
                    poses=poses,
                    segnet=segnet,
                    posenet=posenet,
                    config_hash=config.typed_config_hash(),
                ),
            ),
        )
    ]
    accepted_ids: list[str] = []
    for row in admitted:
        accepted_ids.append(str(row["candidate"]["column_id"]))
        states.append(
            (
                int(row["measurement_after"]["archive_bytes"]) - len(base_archive),
                tuple(accepted_ids),
                row["measurement_after"],
            )
        )
    by_id = {row.column_id: row for row in bundle_columns}
    rows = []
    for budget in FIXED_BUDGETS:
        _added, ids, measurement = max(
            (value for value in states if value[0] <= budget),
            key=lambda value: (value[0], len(value[1])),
        )
        archive = _compile_state(
            predictor_archive=predictor_archive,
            base_bank=base_bank,
            specs=[by_id[value] for value in ids],
            profile=profile,
        )
        if int(measurement["archive_bytes"]) != len(archive) or measurement["archive_sha256"] != _sha256(archive):
            raise DirectDescriptionError("rebased control reconstruction differs from measured state")
        path = root / f"v12_rebased_add{budget}.not_a_candidate.zip.receipt-bytes"
        _publish_immutable(path, archive)
        rows.append(
            {
                "added_byte_budget": budget,
                "selected_bundle_count": len(ids),
                "selected_column_ids": list(ids),
                "realized_added_bytes": len(archive) - len(base_archive),
                "archive": {
                    "path": _portable(path),
                    "bytes": len(archive),
                    "sha256": _sha256(archive),
                },
                "d_seg": measurement["d_seg"],
                "legacy_scorer_grid_control_d_seg": f"{LEGACY_SCORER_GRID_DSEG:.12f}",
                "d_seg_delta_vs_legacy_scorer_grid": (f"{float(measurement['d_seg']) - LEGACY_SCORER_GRID_DSEG:.12f}"),
                "d_pose": measurement["d_pose"],
                "objective": measurement["objective"],
                "measurement_path": "camera_uint8_then_evaluator_R_then_frozen_scorers",
                "evidence_axis": EVIDENCE_AXIS,
                "score_claim": False,
            }
        )
    _write_json(
        result_path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "stage": "v12_common_exact_r_rebased_controls",
            "typed_config_sha256": config.typed_config_hash(),
            "rows": rows,
            "score_claim": False,
        },
    )
    return rows


def _n600_generated_rows(
    *,
    root: Path,
    config: DDMA1ColumnGeneratedCorrectionConfigV1,
    atomic: Sequence[ColumnSpec],
    predictor_archive: bytes,
    base_bank: ScorerSolvedTemplateBankV1,
    profile: Any,
    labels: np.ndarray,
    poses: np.ndarray,
    segnet: Any,
    posenet: Any,
    controls: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {row.column_id: row for row in atomic}
    rounds = [_json(root / "stage_checkpoints" / f"pricing_round_{index:02d}.json") for index in range(1, 4)]
    for index, row in enumerate(rounds, start=1):
        _require_checkpoint_identity(
            row,
            config_hash=config.typed_config_hash(),
            stage="complete_exact_pricing_round",
        )
        if row.get("round") != index or row.get("complete") is not True:
            raise DirectDescriptionError("pricing-round history identity differs")
    cumulative: list[str] = []
    replay_rows = []
    replay_cache: dict[
        tuple[str, ...],
        tuple[dict[str, Any], bytes, dict[int, BatchScore], Path],
    ] = {}
    for row in rounds:
        cumulative.extend(str(value) for value in row["accepted_new_column_ids"])
        ids = tuple(cumulative)
        stage = root / "stage_checkpoints" / f"pricing_round_{row['round']:02d}_accepted_n600.json"
        archive = _compile_state(
            predictor_archive=predictor_archive,
            base_bank=base_bank,
            specs=[by_id[value] for value in ids],
            profile=profile,
        )
        if ids in replay_cache and not stage.exists():
            _measurement_row, cached_archive, cached_batches, cached_path = replay_cache[ids]
            if cached_archive != archive:
                raise DirectDescriptionError("identical accepted-set archive reconstruction differs")
            _preserve_reused_window(
                path=stage,
                archive=archive,
                pair_start=0,
                pair_count=600,
                rows=cached_batches,
                config_hash=config.typed_config_hash(),
                reused_from=_portable(cached_path),
            )
        batches = _load_or_score_window(
            path=stage,
            archive=archive,
            pair_start=0,
            pair_count=600,
            labels=labels,
            poses=poses,
            segnet=segnet,
            posenet=posenet,
            config_hash=config.typed_config_hash(),
        )
        measurement = _measurement(archive, batches)
        archive_path = root / f"pricing_round_{int(row['round']):02d}_accepted_n600.not_a_candidate.zip.receipt-bytes"
        _publish_immutable(archive_path, archive)
        replay_rows.append(
            {
                "round": row["round"],
                "accepted_column_ids": list(ids),
                "measurement": measurement,
                "archive": {
                    "path": _portable(archive_path),
                    "bytes": len(archive),
                    "sha256": _sha256(archive),
                },
                "exact_n600_replay_complete": True,
            }
        )
        replay_cache[ids] = (measurement, archive, batches, stage)
    base_archive = _compile_state(
        predictor_archive=predictor_archive,
        base_bank=base_bank,
        specs=(),
        profile=profile,
    )
    prefix_archives = [
        _compile_state(
            predictor_archive=predictor_archive,
            base_bank=base_bank,
            specs=[by_id[value] for value in cumulative[:index]],
            profile=profile,
        )
        for index in range(len(cumulative) + 1)
    ]
    selector = str(rounds[-1]["selected_global_mode"])
    if selector not in {"beam_width_32", "conflict_miqp"}:
        raise DirectDescriptionError("global selector fell outside preregistered exact modes")
    equal = []
    for control in controls:
        budget = int(control["added_byte_budget"])
        control_realized_added_bytes = int(control["realized_added_bytes"])
        prefix_index = _largest_prefix_not_exceeding(
            [len(archive) for archive in prefix_archives],
            base_archive_bytes=len(base_archive),
            realized_added_byte_cap=control_realized_added_bytes,
        )
        ids = tuple(cumulative[:prefix_index])
        archive = prefix_archives[prefix_index]
        if ids not in replay_cache:
            batches = _load_or_score_window(
                path=root / "stage_checkpoints" / f"equal_byte_add{budget}_n600.json",
                archive=archive,
                pair_start=0,
                pair_count=600,
                labels=labels,
                poses=poses,
                segnet=segnet,
                posenet=posenet,
                config_hash=config.typed_config_hash(),
            )
            replay_cache[ids] = (
                _measurement(archive, batches),
                archive,
                batches,
                root / "stage_checkpoints" / f"equal_byte_add{budget}_n600.json",
            )
        generated, cached_archive, _batches, measurement_path = replay_cache[ids]
        if cached_archive != archive:
            raise DirectDescriptionError("equal-byte replay cache archive differs")
        archive_path = root / f"generated_equal_byte_add{budget}.not_a_candidate.zip.receipt-bytes"
        _publish_immutable(archive_path, archive)
        realized_added_bytes = int(generated["archive_bytes"]) - len(base_archive)
        equal.append(
            {
                "added_byte_budget": budget,
                "matching_rule": ("maximum exact generated prefix not exceeding the rebased v12 realized bytes"),
                "v12_control": dict(control),
                "generated_vocabulary": {
                    **generated,
                    "selected_column_ids": list(ids),
                    "realized_added_bytes": realized_added_bytes,
                    "archive": {
                        "path": _portable(archive_path),
                        "bytes": len(archive),
                        "sha256": _sha256(archive),
                    },
                    "measurement_checkpoint": _portable(measurement_path),
                },
                "exact_added_byte_gap_generated_minus_v12": (realized_added_bytes - control_realized_added_bytes),
                "global_selector": selector,
                "exact_replay_complete": True,
                "beats_v12": float(generated["objective"]) < float(control["objective"]),
                "coder_race": {
                    FIXED_CODER_ENTRANTS[0]: {
                        "status": "MEASURED_RECEIVER_CLOSED",
                        "measured": True,
                        "archive_bytes": int(generated["archive_bytes"]),
                    },
                    FIXED_CODER_ENTRANTS[1]: {
                        "status": "NOT_APPLICABLE_TYPED_ATOM_STREAM_NOT_DENSE_2_OF_4",
                        "measured": False,
                        "admitted_to_equal_byte_table": False,
                    },
                    FIXED_CODER_ENTRANTS[2]: {
                        "status": "QUEUED_NOT_MEASURED_NO_COMMON_RECEIVER_DECODER",
                        "measured": False,
                        "admitted_to_equal_byte_table": False,
                    },
                    PERMUTATION_GAUGE_CODER_ENTRANT: _permutation_gauge_coder_row(
                        archive,
                        ids,
                    ),
                },
            }
        )
    return replay_rows, equal


def _final_receipt(
    *,
    root: Path,
    config: DDMA1ColumnGeneratedCorrectionConfigV1,
    semantic_argv: Sequence[str],
    common_base: bytes,
    controls: Sequence[Mapping[str, Any]],
    round_replays: Sequence[Mapping[str, Any]],
    equal: Sequence[Mapping[str, Any]],
) -> Path:
    rounds = [_json(root / "stage_checkpoints" / f"pricing_round_{index:02d}.json") for index in range(1, 4)]
    history = [
        {
            "round": row["round"],
            "complete": row["complete"],
            "exact_pricing": row["exact_pricing"],
            "evidence_window": row["evidence_window"],
            "generated_column_count": row["generated_column_count"],
            "negative_reduced_cost_count": row["negative_reduced_cost_count"],
            "selected_global_mode": row["selected_global_mode"],
            "accepted_new_column_ids": row["accepted_new_column_ids"],
            "accepted_n600_replay": next(value for value in round_replays if value["round"] == row["round"]),
        }
        for row in rounds
    ]
    falsified, three_clean_rounds, any_win = _formulation_falsified(history, equal)
    if any_win:
        verdict = "DESCENT_FOUND_COMMON_EXACT_R"
        scope = "INSTANCE:POSTSOLVE_COMMON_MASTER_N64_PRICING_N600_ACCEPTED_REPLAY"
    elif falsified:
        verdict = "FALSIFIED_FORMULATION_THREE_CLEAN_PRICING_ROUNDS"
        scope = "FORMULATION:COLUMN_FAMILIES_AND_THREE_ROUND_GLOBAL_SELECTION"
    else:
        verdict = "FORMULATION_OPEN_NEGATIVE_REDUCED_COST_WITHOUT_EQUAL_BYTE_WIN"
        scope = "INSTANCE:POSTSOLVE_COMMON_MASTER_N64_SCREEN_AND_N600_ACCEPTED_REPLAY"
    path = root / "ddm_v18b_common_master_pricing_receipt.json"
    payload = {
        "schema": SCHEMA,
        "lane_id": LANE_ID,
        "run_id": config.run_id.replace("n64", "common_master_n600"),
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "semantic_argv": list(semantic_argv),
        "common_master": {
            "archive_bytes": len(common_base),
            "archive_sha256": _sha256(common_base),
            "postsolve_only": True,
            "predict_productions_present": False,
            "receiver_custody": dict(receive_common_exact_r_master(common_base).custody),
        },
        "implementation_custody": _json(root / "stage_checkpoints" / "01_frozen_scorer_target_custody.json"),
        "v12_rebased_control_rows": list(controls),
        "pricing_round_history": history,
        "accepted_set_n600_replays": list(round_replays),
        "equal_byte_rows": list(equal),
        "falsifier": {
            "condition": (
                "three complete exact pricing rounds have no negative reduced-cost column AND "
                "global exact joint-objective replay has no equal-byte v12 beat"
            ),
            "three_clean_rounds": three_clean_rounds,
            "any_equal_byte_joint_objective_win": any_win,
            "eligible": len(history) == 3 and len(equal) == len(FIXED_BUDGETS),
            "triggered": falsified,
        },
        "verdict": verdict,
        "verdict_scope": scope,
        "triality": {
            "dsl": "DDMA1ColumnGeneratedCorrectionConfigV1",
            "dag": ".omx/research/ddm_v18_column_generation_vocabulary_DAG_FEED_20260723.md",
            "equations": "tac.canonical_equations.ddm_v18_column_pricing_law_20260723",
        },
        "operator_directives_consumed": {
            "nested_jacobians_20260723T041016Z": (
                "No frozen-Jacobian predicted cost is used. V16 source-Jacobian endpoints "
                "are only candidate coordinates; exact singleton and selected-set replay "
                "through the current receiver are the pricing and admission authorities."
            ),
            "permutation_gauge_20260723T052721Z": (
                "The coder race reports canonical typed-address ordering versus as-is bytes. "
                "The candidate pool is not counted payload; an empty selected vocabulary has "
                "a measured zero-byte ordering-gauge delta."
            )
        },
        "resume": {
            "all_preserved": True,
            "pricing_rounds_preserved": 3,
            "v12_candidate_checkpoints_preserved": 353,
            "n600_accepted_replays_preserved": 3,
        },
        "pointer": POINTER,
        "pointer_moved": False,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "d_seg_claim": False,
        "d_pose_claim": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
        "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
    }
    _write_json(path, payload)
    return path


def run(
    config: DDMA1ColumnGeneratedCorrectionConfigV1,
    root: Path,
    semantic_argv: Sequence[str],
    *,
    max_work_items: int,
) -> Path:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / "ddm_v18b_common_master_pricing_receipt.json"
    if receipt.exists():
        value = _json(receipt)
        if value.get("typed_config_sha256") != config.typed_config_hash():
            raise DirectDescriptionError("completed v18b receipt typed config differs")
        print(json.dumps({"complete": True, "resumed": True, "receipt": str(receipt)}))
        return receipt
    (
        v12_base,
        common_base,
        base_bank,
        v16_bank,
        v15_cfg,
        source_custody,
    ) = _source_state(config)
    bundle_rows = _load_bundle_rows()
    atomic = _atomic_columns(bundle_rows, v16_bank)
    _source_checkpoint(
        root=root,
        config=config,
        common_base=common_base,
        bundle_rows=bundle_rows,
        atomic=atomic,
        source_custody=source_custody,
    )
    base_receiver = receive_common_exact_r_master(common_base)
    profile = base_receiver.base.realization_profile
    if profile is None:
        raise DirectDescriptionError("common-master profile disappeared")
    inner_members, _homes = parse_carrier_compose_archive(v12_base)
    predictor = inner_members["predictor.zip"]
    reconstructed_common_base = _compile_state(
        predictor_archive=predictor,
        base_bank=base_bank,
        specs=(),
        profile=profile,
    )
    if reconstructed_common_base != common_base:
        raise DirectDescriptionError("empty common-master reconstruction differs from bound source")
    target = Path(v15_cfg.target_cache_path)
    if target.stat().st_size != v15_cfg.target_cache_bytes:
        raise DirectDescriptionError("frozen n600 cache bytes differ")
    if _sha256_file(target) != v15_cfg.target_cache_sha256:
        raise DirectDescriptionError("frozen n600 cache SHA-256 differs")
    labels = open_stored_npy_memmap(target, "lstars")
    poses = open_stored_npy_memmap(target, "gt_poses")
    segnet, posenet, scorer_custody = _load_models(v15_cfg)
    _scorer_checkpoint(
        root=root,
        config=config,
        v15_config=v15_cfg,
        scorer_custody=scorer_custody,
    )
    for round_index in range(1, 4):
        path = root / "stage_checkpoints" / f"pricing_round_{round_index:02d}.json"
        if not path.exists():
            completed = _run_pricing_round(
                round_index=round_index,
                root=root,
                config=config,
                atomic=atomic,
                predictor_archive=predictor,
                base_bank=base_bank,
                profile=profile,
                labels=labels,
                poses=poses,
                segnet=segnet,
                posenet=posenet,
            )
            print(
                json.dumps({"complete": False, "stage": f"pricing_round_{round_index}", "checkpoint": str(completed)})
            )
            return completed
    bundle_columns = _bundle_columns(bundle_rows)
    rebaseline_complete = _run_rebaseline_chunk(
        root=root,
        config=config,
        bundle_columns=bundle_columns,
        predictor_archive=predictor,
        base_bank=base_bank,
        profile=profile,
        labels=labels,
        poses=poses,
        segnet=segnet,
        posenet=posenet,
        max_work_items=max_work_items,
        pose_tube_dpose_radius=float(source_custody["v12_pose_tube_dpose_radius"]),
    )
    if not rebaseline_complete:
        rows, _accepted = _rebaseline_progress(
            root,
            bundle_columns,
            config_hash=config.typed_config_hash(),
        )
        checkpoint = root / "stage_checkpoints" / "v12_rebaseline_candidates" / f"{len(rows) - 1:04d}.json"
        print(
            json.dumps(
                {"complete": False, "stage": "v12_rebaseline", "processed": len(rows), "total": len(bundle_columns)}
            )
        )
        return checkpoint
    controls = _control_rows(
        root=root,
        config=config,
        bundle_columns=bundle_columns,
        predictor_archive=predictor,
        base_bank=base_bank,
        profile=profile,
        labels=labels,
        poses=poses,
        segnet=segnet,
        posenet=posenet,
    )
    round_replays, equal = _n600_generated_rows(
        root=root,
        config=config,
        atomic=atomic,
        predictor_archive=predictor,
        base_bank=base_bank,
        profile=profile,
        labels=labels,
        poses=poses,
        segnet=segnet,
        posenet=posenet,
        controls=controls,
    )
    final = _final_receipt(
        root=root,
        config=config,
        semantic_argv=semantic_argv,
        common_base=common_base,
        controls=controls,
        round_replays=round_replays,
        equal=equal,
    )
    print(json.dumps({"complete": True, "resumed": False, "receipt": str(final)}))
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--max-work-items", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.max_work_items <= 64:
        raise DirectDescriptionError("max-work-items must be in [1,64]")
    config = DDMA1ColumnGeneratedCorrectionConfigV1.model_validate_json(_read_regular_file_once(args.config))
    semantic_argv = [
        "tools/run_ddm_v18b_common_master_pricing.py",
        "--config",
        _portable(args.config),
        "--output-directory",
        _portable(args.output_directory),
        "--max-work-items",
        str(args.max_work_items),
    ]
    run(config, args.output_directory, semantic_argv, max_work_items=args.max_work_items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
