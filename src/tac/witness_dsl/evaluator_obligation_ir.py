# SPDX-License-Identifier: MIT
"""Typed encode-side obligations for the coupled V9-to-V10 compiler seam.

This module does *not* map semantic class identifiers to RGB values.  The only
production compile operation accepts explicit uint8 scorer-plane candidates,
realizes them through :mod:`tac.optimization.v10_constructive_solver`, requires
a caller-supplied native-fp32 hard oracle, and parse-backs the two-plane bytes
through :mod:`tac.witness_dsl.v10_production_receiver`.

The obligation IR and oracle observations are encoder-side scientific state.
They are forbidden decoder payloads: the production packet contains only the
counted, source-derived preimage description and imports no scorer or target
table.  This seam deliberately makes no score or promotion claim.  Final
admission remains the exact coupled contest objective after archive decode.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

import tac.optimization.v10_constructive_solver as v10_solver
import tac.witness_dsl.v10_production_receiver as v10_receiver
from tac.codec.v10_predictor_residual import CODEC_ID as TWO_PLANE_CODEC_ID
from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    Factor2ExactVerification,
    verify_factor2_uint8_scorer_plane,
)
from tac.optimization.v10_constructive_solver import (
    EXPECTED_SOURCE_HASHES,
    HARD_ORACLE_SCHEMA,
    RECEIVER_ARITHMETIC,
    HardOracleDecision,
    LatticeAdmission,
    realize_factor2_and_require_hard_oracle,
)
from tac.witness_dsl.coupled_witness_state import (
    DECODER_PAYLOAD_POLICY,
    SOURCE_DERIVED_LINEAGE,
    CoupledWitnessStateError,
    FrozenSpaceIdentity,
    canonical_json_bytes,
    canonical_sha256,
    decode_canonical_json,
    sha256_bytes,
)
from tac.witness_dsl.v10_production_receiver import (
    DESCRIPTION_FRAME0_POLICY_ID,
    RECEIVER_CONTRACT_ID,
    TIE_POLICY_ID,
    build_packet,
    decode_y_plane_pair,
    parse_packet,
)

EVALUATOR_OBLIGATION_IR_SCHEMA = "tac.evaluator_obligation_ir.v1"
EVALUATOR_OBLIGATION_IR_ENVELOPE_SCHEMA = "tac.evaluator_obligation_ir.envelope.v1"
FRAME1_CELL_OBLIGATION_SCHEMA = "tac.frame1_cell_obligation.v1"
POSE_FIBRE_OBLIGATION_SCHEMA = "tac.conditional_frame0_pose_fibre_obligation.v1"
EXPLICIT_PREIMAGE_RESULT_SCHEMA = "tac.explicit_v10_preimage_compile_result.v1"
EXPLICIT_PREIMAGE_RESULT_ENVELOPE_SCHEMA = "tac.explicit_v10_preimage_compile_result.envelope.v1"
PAIR_PREIMAGE_RECEIPT_SCHEMA = "tac.explicit_v10_pair_preimage_receipt.v1"
HARD_ORACLE_EVIDENCE_RECEIPT_SCHEMA = "tac.evaluator_obligation_hard_oracle_evidence.v1"

CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
N_CLASSES = len(CLASS_NAMES)
CHANNELS = 3
POSE_DIMENSIONS = 6
V10_SOLVER_CONTRACT_ID = "v10-factor2-explicit-preimage-plus-native-hard-oracle.v1"
JOINT_OBJECTIVE_ID = "100*d_seg+sqrt(10*d_pose)+25*archive_bytes/37545489.v1"
FINAL_ADMISSION_POLICY = "exact-coupled-score-after-same-object-archive-decode-only"
IR_PAYLOAD_SCOPE = "encode-side-obligations-only-never-decoder-payload"
PREIMAGE_POLICY = "caller-supplied-explicit-uint8-never-class-to-rgb-synthesis"

_FROZEN_ORACLE_ARTIFACT_KEYS = {
    "upstream/frame_utils.py": "frame_utils_sha256",
    "upstream/models/posenet.safetensors": "posenet_weights_sha256",
    "upstream/models/segnet.safetensors": "segnet_weights_sha256",
    "upstream/modules.py": "modules_sha256",
}


class EvaluatorObligationIRError(ValueError):
    """Malformed obligation, absent evidence, or broken V10 custody."""


class CollateralOwner(StrEnum):
    """Logical owner charged for a frame-1 cell's realization collateral."""

    BULK_BOUNDARY = "bulk_boundary"
    LANE_CHART = "lane_chart"
    MOVABLE_MYCAR = "movable_mycar"
    CELL_VALUE_PREIMAGE = "cell_value_preimage"
    IRREDUCIBLE_QUOTIENT = "irreducible_quotient"


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise EvaluatorObligationIRError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise EvaluatorObligationIRError(
            f"{label} fields differ: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvaluatorObligationIRError(f"{label} must be lowercase SHA-256 hex")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise EvaluatorObligationIRError(f"{label} must be a positive exact integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise EvaluatorObligationIRError(f"{label} must be a nonnegative exact integer")
    return value


def _finite_float(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EvaluatorObligationIRError(f"{label} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        suffix = " nonnegative" if nonnegative else ""
        raise EvaluatorObligationIRError(f"{label} must be finite and{suffix}")
    return 0.0 if result == 0.0 else result


def _immutable_array(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def _array_bytes_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _array_identity(value: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(value)
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "byte_length": int(contiguous.nbytes),
        "bytes_sha256": _array_bytes_sha256(contiguous),
    }


def _array_identity_sha256(value: np.ndarray) -> str:
    return canonical_sha256(_array_identity(value))


def _module_source_sha256(module: Any, label: str) -> str:
    source_path = getattr(module, "__file__", None)
    if not isinstance(source_path, str) or not source_path:
        raise EvaluatorObligationIRError(f"{label} source path is unavailable")
    path = Path(source_path)
    if path.suffix == ".pyc" and path.with_suffix(".py").is_file():
        path = path.with_suffix(".py")
    if not path.is_file():
        raise EvaluatorObligationIRError(f"{label} source bytes are unavailable")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_frozen_oracle_custody(frozen_space: FrozenSpaceIdentity) -> None:
    artifacts = {item.artifact_id: item.sha256 for item in frozen_space.evaluator_artifacts}
    for artifact_id, source_key in _FROZEN_ORACLE_ARTIFACT_KEYS.items():
        if artifacts.get(artifact_id) != EXPECTED_SOURCE_HASHES[source_key]:
            raise EvaluatorObligationIRError(
                f"frozen-space evaluator artifact {artifact_id!r} differs from V10 hard-oracle custody"
            )


@dataclass(frozen=True, order=True)
class Frame1CellObligation:
    """One exact five-class winner cell and its winner-vs-class margins."""

    pair_id: int
    row: int
    col: int
    winner_class_id: int
    required_margin_by_class: tuple[float, float, float, float, float]
    collateral_owner: CollateralOwner

    def __post_init__(self) -> None:
        _nonnegative_int(self.pair_id, "frame1 cell pair_id")
        _nonnegative_int(self.row, "frame1 cell row")
        _nonnegative_int(self.col, "frame1 cell col")
        winner = _nonnegative_int(self.winner_class_id, "frame1 cell winner_class_id")
        if winner >= N_CLASSES:
            raise EvaluatorObligationIRError("frame1 cell winner_class_id is outside the five-class universe")
        margins = self.required_margin_by_class
        if type(margins) is not tuple or len(margins) != N_CLASSES:
            raise EvaluatorObligationIRError("frame1 cell margins must be an exact five-value tuple")
        normalized = tuple(
            _finite_float(value, f"frame1 cell margin[{class_id}]", nonnegative=True)
            for class_id, value in enumerate(margins)
        )
        if normalized[winner] != 0.0:
            raise EvaluatorObligationIRError("the winner-to-self margin must be exactly zero")
        if not isinstance(self.collateral_owner, CollateralOwner):
            raise EvaluatorObligationIRError("frame1 cell collateral_owner is invalid")
        object.__setattr__(self, "required_margin_by_class", normalized)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": FRAME1_CELL_OBLIGATION_SCHEMA,
            "pair_id": self.pair_id,
            "row": self.row,
            "col": self.col,
            "winner_class_id": self.winner_class_id,
            "winner_class_name": CLASS_NAMES[self.winner_class_id],
            "required_margin_by_class": list(self.required_margin_by_class),
            "collateral_owner": self.collateral_owner.value,
            "tie_policy_id": TIE_POLICY_ID,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Frame1CellObligation:
        _exact_keys(
            value,
            {
                "schema",
                "pair_id",
                "row",
                "col",
                "winner_class_id",
                "winner_class_name",
                "required_margin_by_class",
                "collateral_owner",
                "tie_policy_id",
            },
            "frame1 cell obligation",
        )
        if value["schema"] != FRAME1_CELL_OBLIGATION_SCHEMA or value["tie_policy_id"] != TIE_POLICY_ID:
            raise EvaluatorObligationIRError("frame1 cell schema/tie policy differs")
        winner = value["winner_class_id"]
        if type(winner) is not int or not 0 <= winner < N_CLASSES:
            raise EvaluatorObligationIRError("frame1 cell winner_class_id is invalid")
        if value["winner_class_name"] != CLASS_NAMES[winner]:
            raise EvaluatorObligationIRError("frame1 cell class name/id binding differs")
        margins = value["required_margin_by_class"]
        if not isinstance(margins, list):
            raise EvaluatorObligationIRError("frame1 cell margins must serialize as an array")
        try:
            owner = CollateralOwner(value["collateral_owner"])
        except (TypeError, ValueError) as exc:
            raise EvaluatorObligationIRError("frame1 cell collateral owner differs") from exc
        return cls(
            pair_id=value["pair_id"],
            row=value["row"],
            col=value["col"],
            winner_class_id=winner,
            required_margin_by_class=tuple(margins),  # type: ignore[arg-type]
            collateral_owner=owner,
        )


def frame1_pair_obligation_sha256(
    cells: Sequence[Frame1CellObligation],
    pair_id: int,
) -> str:
    """Hash the ordered frame-1 obligations for one exact pair address."""

    pair = _nonnegative_int(pair_id, "pair_id")
    selected = tuple(cell for cell in cells if isinstance(cell, Frame1CellObligation) and cell.pair_id == pair)
    if not selected:
        raise EvaluatorObligationIRError(f"pair {pair} has no frame1 cell obligations")
    if selected != tuple(sorted(selected, key=lambda item: (item.row, item.col))):
        raise EvaluatorObligationIRError(f"pair {pair} frame1 cell obligations are not canonically ordered")
    return canonical_sha256(
        {
            "schema": "tac.frame1_pair_obligations.v1",
            "pair_id": pair,
            "cells": [cell.as_dict() for cell in selected],
        }
    )


@dataclass(frozen=True, order=True)
class ConditionalFrame0PoseFibreObligation:
    """Pose target for frame 0, conditional on the exact frame-1 obligations."""

    pair_id: int
    target_pose6: tuple[float, float, float, float, float, float]
    conditioned_frame1_obligation_sha256: str

    def __post_init__(self) -> None:
        _nonnegative_int(self.pair_id, "pose fibre pair_id")
        if type(self.target_pose6) is not tuple or len(self.target_pose6) != POSE_DIMENSIONS:
            raise EvaluatorObligationIRError("pose fibre target must be an exact six-value tuple")
        object.__setattr__(
            self,
            "target_pose6",
            tuple(_finite_float(value, f"pose fibre target[{index}]") for index, value in enumerate(self.target_pose6)),
        )
        _sha256(self.conditioned_frame1_obligation_sha256, "conditioned frame1 obligation SHA-256")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": POSE_FIBRE_OBLIGATION_SCHEMA,
            "pair_id": self.pair_id,
            "target_pose6": list(self.target_pose6),
            "conditioned_frame1_obligation_sha256": self.conditioned_frame1_obligation_sha256,
            "conditioning_policy": "solve-frame0-only-after-frame1-obligation-is-fixed",
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ConditionalFrame0PoseFibreObligation:
        _exact_keys(
            value,
            {
                "schema",
                "pair_id",
                "target_pose6",
                "conditioned_frame1_obligation_sha256",
                "conditioning_policy",
            },
            "conditional frame0 pose fibre obligation",
        )
        if value["schema"] != POSE_FIBRE_OBLIGATION_SCHEMA:
            raise EvaluatorObligationIRError("pose fibre obligation schema differs")
        if value["conditioning_policy"] != "solve-frame0-only-after-frame1-obligation-is-fixed":
            raise EvaluatorObligationIRError("pose fibre conditioning policy differs")
        target = value["target_pose6"]
        if not isinstance(target, list):
            raise EvaluatorObligationIRError("pose fibre target must serialize as an array")
        return cls(
            pair_id=value["pair_id"],
            target_pose6=tuple(target),  # type: ignore[arg-type]
            conditioned_frame1_obligation_sha256=value["conditioned_frame1_obligation_sha256"],
        )


@dataclass(frozen=True)
class EvaluatorObligationIR:
    """Canonical encoder-side Seg/Pose obligations for one coupled state."""

    frozen_space: FrozenSpaceIdentity
    coupled_state_sha256: str
    predictor_state_sha256: str
    predictor_semantic_sha256: str
    camera_height: int
    camera_width: int
    frame1_cells: tuple[Frame1CellObligation, ...]
    conditional_frame0_pose_fibres: tuple[ConditionalFrame0PoseFibreObligation, ...]
    lineage: str = SOURCE_DERIVED_LINEAGE
    borrowed_candidate_bytes: int = 0
    payload_scope: str = IR_PAYLOAD_SCOPE
    decoder_payload_policy: str = DECODER_PAYLOAD_POLICY
    decoder_contains_scorer: bool = False
    decoder_contains_target_table: bool = False
    joint_objective_id: str = JOINT_OBJECTIVE_ID
    final_admission_policy: str = FINAL_ADMISSION_POLICY

    def __post_init__(self) -> None:
        if not isinstance(self.frozen_space, FrozenSpaceIdentity):
            raise EvaluatorObligationIRError("frozen_space must be a validated FrozenSpaceIdentity")
        _sha256(self.coupled_state_sha256, "coupled_state_sha256")
        _sha256(self.predictor_state_sha256, "predictor_state_sha256")
        _sha256(self.predictor_semantic_sha256, "predictor_semantic_sha256")
        _positive_int(self.camera_height, "camera_height")
        _positive_int(self.camera_width, "camera_width")
        if type(self.frame1_cells) is not tuple or not self.frame1_cells:
            raise EvaluatorObligationIRError("frame1_cells must be a nonempty exact tuple")
        if any(not isinstance(item, Frame1CellObligation) for item in self.frame1_cells):
            raise EvaluatorObligationIRError("frame1_cells contains an invalid obligation")
        cell_order = tuple(sorted(self.frame1_cells, key=lambda item: (item.pair_id, item.row, item.col)))
        if self.frame1_cells != cell_order:
            raise EvaluatorObligationIRError("frame1_cells must use canonical pair/row/col order")
        addresses = [(item.pair_id, item.row, item.col) for item in self.frame1_cells]
        if len(addresses) != len(set(addresses)):
            raise EvaluatorObligationIRError("frame1_cells contains duplicate pair/row/col addresses")
        pair_ids = set(range(self.frozen_space.pair_count))
        observed_pairs = {item.pair_id for item in self.frame1_cells}
        if observed_pairs != pair_ids:
            raise EvaluatorObligationIRError("frame1_cells must cover every canonical pair")
        for item in self.frame1_cells:
            if item.row >= self.frozen_space.scorer_height or item.col >= self.frozen_space.scorer_width:
                raise EvaluatorObligationIRError("frame1 cell address exceeds frozen scorer geometry")
        fibres = self.conditional_frame0_pose_fibres
        if type(fibres) is not tuple or len(fibres) != self.frozen_space.pair_count:
            raise EvaluatorObligationIRError("conditional pose fibres must contain exactly one row per pair")
        if any(not isinstance(item, ConditionalFrame0PoseFibreObligation) for item in fibres):
            raise EvaluatorObligationIRError("conditional pose fibres contain an invalid obligation")
        if tuple(item.pair_id for item in fibres) != tuple(range(self.frozen_space.pair_count)):
            raise EvaluatorObligationIRError("conditional pose fibres must use canonical contiguous pair order")
        for fibre in fibres:
            expected = frame1_pair_obligation_sha256(self.frame1_cells, fibre.pair_id)
            if fibre.conditioned_frame1_obligation_sha256 != expected:
                raise EvaluatorObligationIRError("pose fibre is not conditioned on its exact frame1 obligations")
        if self.lineage != SOURCE_DERIVED_LINEAGE or self.borrowed_candidate_bytes != 0:
            raise EvaluatorObligationIRError("obligation IR must have source-only original-work lineage")
        if self.payload_scope != IR_PAYLOAD_SCOPE or self.decoder_payload_policy != DECODER_PAYLOAD_POLICY:
            raise EvaluatorObligationIRError("obligation IR encoder/decoder payload policy differs")
        if type(self.decoder_contains_scorer) is not bool or self.decoder_contains_scorer:
            raise EvaluatorObligationIRError("a scorer in the decoder is forbidden")
        if type(self.decoder_contains_target_table) is not bool or self.decoder_contains_target_table:
            raise EvaluatorObligationIRError("a target table in the decoder is forbidden")
        if self.joint_objective_id != JOINT_OBJECTIVE_ID or self.final_admission_policy != FINAL_ADMISSION_POLICY:
            raise EvaluatorObligationIRError("independent component gates cannot replace the exact joint objective")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": EVALUATOR_OBLIGATION_IR_SCHEMA,
            "frozen_space": self.frozen_space.as_dict(),
            "coupled_state_sha256": self.coupled_state_sha256,
            "predictor_state_sha256": self.predictor_state_sha256,
            "predictor_semantic_sha256": self.predictor_semantic_sha256,
            "camera_height": self.camera_height,
            "camera_width": self.camera_width,
            "channels": CHANNELS,
            "class_names": list(CLASS_NAMES),
            "frame1_cells": [item.as_dict() for item in self.frame1_cells],
            "conditional_frame0_pose_fibres": [item.as_dict() for item in self.conditional_frame0_pose_fibres],
            "lineage": self.lineage,
            "borrowed_candidate_bytes": self.borrowed_candidate_bytes,
            "payload_scope": self.payload_scope,
            "decoder_payload_policy": self.decoder_payload_policy,
            "decoder_contains_scorer": self.decoder_contains_scorer,
            "decoder_contains_target_table": self.decoder_contains_target_table,
            "joint_objective_id": self.joint_objective_id,
            "final_admission_policy": self.final_admission_policy,
        }

    @property
    def ir_sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    def pair_obligation_sha256(self, pair_id: int) -> str:
        pair = _nonnegative_int(pair_id, "pair_id")
        if pair >= self.frozen_space.pair_count:
            raise EvaluatorObligationIRError("pair_id exceeds frozen pair count")
        fibre = self.conditional_frame0_pose_fibres[pair]
        return canonical_sha256(
            {
                "schema": "tac.coupled_pair_obligations.v1",
                "pair_id": pair,
                "frame1_obligation_sha256": frame1_pair_obligation_sha256(self.frame1_cells, pair),
                "conditional_frame0_pose_fibre": fibre.as_dict(),
            }
        )

    def to_bytes(self) -> bytes:
        body = self.as_dict()
        return canonical_json_bytes(
            {
                "schema": EVALUATOR_OBLIGATION_IR_ENVELOPE_SCHEMA,
                "body": body,
                "body_sha256": canonical_sha256(body),
            }
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvaluatorObligationIR:
        _exact_keys(
            value,
            {
                "schema",
                "frozen_space",
                "coupled_state_sha256",
                "predictor_state_sha256",
                "predictor_semantic_sha256",
                "camera_height",
                "camera_width",
                "channels",
                "class_names",
                "frame1_cells",
                "conditional_frame0_pose_fibres",
                "lineage",
                "borrowed_candidate_bytes",
                "payload_scope",
                "decoder_payload_policy",
                "decoder_contains_scorer",
                "decoder_contains_target_table",
                "joint_objective_id",
                "final_admission_policy",
            },
            "evaluator obligation IR",
        )
        if value["schema"] != EVALUATOR_OBLIGATION_IR_SCHEMA:
            raise EvaluatorObligationIRError("evaluator obligation IR schema differs")
        if value["channels"] != CHANNELS or value["class_names"] != list(CLASS_NAMES):
            raise EvaluatorObligationIRError("evaluator obligation class/channel geometry differs")
        cells = value["frame1_cells"]
        fibres = value["conditional_frame0_pose_fibres"]
        if not isinstance(cells, list) or not isinstance(fibres, list):
            raise EvaluatorObligationIRError("obligation rows must serialize as arrays")
        try:
            frozen_space = FrozenSpaceIdentity.from_dict(value["frozen_space"])
        except CoupledWitnessStateError as exc:
            raise EvaluatorObligationIRError("frozen-space identity is invalid") from exc
        return cls(
            frozen_space=frozen_space,
            coupled_state_sha256=value["coupled_state_sha256"],
            predictor_state_sha256=value["predictor_state_sha256"],
            predictor_semantic_sha256=value["predictor_semantic_sha256"],
            camera_height=value["camera_height"],
            camera_width=value["camera_width"],
            frame1_cells=tuple(Frame1CellObligation.from_dict(item) for item in cells),
            conditional_frame0_pose_fibres=tuple(
                ConditionalFrame0PoseFibreObligation.from_dict(item) for item in fibres
            ),
            lineage=value["lineage"],
            borrowed_candidate_bytes=value["borrowed_candidate_bytes"],
            payload_scope=value["payload_scope"],
            decoder_payload_policy=value["decoder_payload_policy"],
            decoder_contains_scorer=value["decoder_contains_scorer"],
            decoder_contains_target_table=value["decoder_contains_target_table"],
            joint_objective_id=value["joint_objective_id"],
            final_admission_policy=value["final_admission_policy"],
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> EvaluatorObligationIR:
        try:
            envelope = decode_canonical_json(payload)
        except CoupledWitnessStateError as exc:
            raise EvaluatorObligationIRError("obligation IR envelope is not canonical") from exc
        _exact_keys(envelope, {"schema", "body", "body_sha256"}, "obligation IR envelope")
        if envelope["schema"] != EVALUATOR_OBLIGATION_IR_ENVELOPE_SCHEMA:
            raise EvaluatorObligationIRError("obligation IR envelope schema differs")
        if canonical_sha256(envelope["body"]) != envelope["body_sha256"]:
            raise EvaluatorObligationIRError("obligation IR body hash differs")
        result = cls.from_dict(envelope["body"])
        if result.ir_sha256 != envelope["body_sha256"]:
            raise EvaluatorObligationIRError("obligation IR reconstructed identity differs")
        return result


@dataclass(frozen=True)
class PairCompileRequest:
    """Exact pair obligations and preimage identities presented to the oracle."""

    pair_id: int
    evaluator_obligation_ir_sha256: str
    pair_obligation_sha256: str
    scorer_y0_identity_sha256: str
    scorer_y1_identity_sha256: str
    frame1_cells: tuple[Frame1CellObligation, ...]
    conditional_frame0_pose_fibre: ConditionalFrame0PoseFibreObligation


@dataclass(frozen=True)
class PairHardOracleEvidence:
    """Fresh oracle observations returned while V10 holds immutable frames.

    ``frame1_cell_logits`` has one exact float32 five-logit row for every cell
    in ``PairCompileRequest.frame1_cells`` in canonical order.  ``pose6`` is
    the exact float32 official six-output observation for the coupled frames.
    The compile seam constructs and hashes its own receipt from these typed
    values; callers cannot substitute a digest for an observation payload.
    """

    pair_id: int
    evaluator_obligation_ir_sha256: str
    pair_obligation_sha256: str
    decision: HardOracleDecision
    frame1_cell_logits: np.ndarray
    pose6: np.ndarray

    def __post_init__(self) -> None:
        _nonnegative_int(self.pair_id, "hard-oracle pair_id")
        _sha256(self.evaluator_obligation_ir_sha256, "hard-oracle obligation IR SHA-256")
        _sha256(self.pair_obligation_sha256, "hard-oracle pair-obligation SHA-256")
        if not isinstance(self.decision, HardOracleDecision):
            raise EvaluatorObligationIRError("hard-oracle evidence requires a HardOracleDecision")
        logits = np.asarray(self.frame1_cell_logits)
        if logits.dtype != np.float32 or logits.ndim != 2 or logits.shape[1:] != (N_CLASSES,):
            raise EvaluatorObligationIRError("hard-oracle logits must be exact float32 [cells,5]")
        if logits.shape[0] == 0 or not np.all(np.isfinite(logits)):
            raise EvaluatorObligationIRError("hard-oracle logits must be nonempty and finite")
        pose = np.asarray(self.pose6)
        if pose.dtype != np.float32 or pose.shape != (POSE_DIMENSIONS,) or not np.all(np.isfinite(pose)):
            raise EvaluatorObligationIRError("hard-oracle pose must be exact finite float32 [6]")
        object.__setattr__(self, "frame1_cell_logits", _immutable_array(logits))
        object.__setattr__(self, "pose6", _immutable_array(pose))


HardOracle = Callable[[PairCompileRequest, tuple[np.ndarray, np.ndarray]], PairHardOracleEvidence]


def _proof_dict(proof: Factor2ExactVerification) -> dict[str, Any]:
    return {
        "scorer_values": proof.scorer_values,
        "owned_camera_values": proof.owned_camera_values,
        "unowned_camera_values": proof.unowned_camera_values,
        "numerator_equal_values": proof.numerator_equal_values,
        "canonical_equal_values": proof.canonical_equal_values,
        "denominator": proof.denominator,
        "numerator_exact": proof.numerator_exact,
        "certified_exact": proof.certified_exact,
    }


_PROOF_FIELDS = {
    "scorer_values",
    "owned_camera_values",
    "unowned_camera_values",
    "numerator_equal_values",
    "canonical_equal_values",
    "denominator",
    "numerator_exact",
    "certified_exact",
}


def _validated_proof_dict(proof: Mapping[str, Any]) -> Mapping[str, Any]:
    _exact_keys(proof, _PROOF_FIELDS, "factor2 proof")
    normalized: dict[str, Any] = {}
    for field in (
        "scorer_values",
        "owned_camera_values",
        "unowned_camera_values",
        "numerator_equal_values",
        "canonical_equal_values",
    ):
        normalized[field] = _nonnegative_int(proof[field], f"factor2 proof {field}")
    normalized["denominator"] = _positive_int(proof["denominator"], "factor2 proof denominator")
    for field in ("numerator_exact", "certified_exact"):
        if type(proof[field]) is not bool or not proof[field]:
            raise EvaluatorObligationIRError(f"factor2 proof {field} must be exactly true")
        normalized[field] = True
    if (
        normalized["numerator_equal_values"] != normalized["scorer_values"]
        or normalized["canonical_equal_values"]
        != normalized["owned_camera_values"] + normalized["unowned_camera_values"]
    ):
        raise EvaluatorObligationIRError("factor2 proof does not cover every scorer/camera value")
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class PairPreimageReceipt:
    pair_id: int
    pair_obligation_sha256: str
    scorer_y0_identity_sha256: str
    scorer_y1_identity_sha256: str
    camera_frame0_sha256: str
    camera_frame1_sha256: str
    factor2_proofs: tuple[Mapping[str, Any], Mapping[str, Any]]
    hard_oracle_decision_sha256: str
    hard_oracle_receipt_sha256: str
    hard_oracle_receipt_bytes: int
    observed_cell_logits_identity_sha256: str
    observed_pose6_identity_sha256: str
    observed_pose_mse: float
    oracle_d_seg: float
    oracle_d_pose: float

    def __post_init__(self) -> None:
        _nonnegative_int(self.pair_id, "pair receipt pair_id")
        for label, value in (
            ("pair obligation", self.pair_obligation_sha256),
            ("scorer Y0 identity", self.scorer_y0_identity_sha256),
            ("scorer Y1 identity", self.scorer_y1_identity_sha256),
            ("camera frame0", self.camera_frame0_sha256),
            ("camera frame1", self.camera_frame1_sha256),
            ("hard-oracle decision", self.hard_oracle_decision_sha256),
            ("hard-oracle receipt", self.hard_oracle_receipt_sha256),
            ("observed cell logits identity", self.observed_cell_logits_identity_sha256),
            ("observed pose identity", self.observed_pose6_identity_sha256),
        ):
            _sha256(value, f"{label} SHA-256")
        _positive_int(self.hard_oracle_receipt_bytes, "hard-oracle receipt bytes")
        if type(self.factor2_proofs) is not tuple or len(self.factor2_proofs) != 2:
            raise EvaluatorObligationIRError("pair receipt requires exactly two factor2 proofs")
        if any(not isinstance(proof, Mapping) for proof in self.factor2_proofs):
            raise EvaluatorObligationIRError("pair receipt factor2 proof is not an object")
        object.__setattr__(
            self,
            "factor2_proofs",
            tuple(_validated_proof_dict(proof) for proof in self.factor2_proofs),
        )
        object.__setattr__(
            self,
            "observed_pose_mse",
            _finite_float(self.observed_pose_mse, "observed pose MSE", nonnegative=True),
        )
        object.__setattr__(self, "oracle_d_seg", _finite_float(self.oracle_d_seg, "oracle d_seg", nonnegative=True))
        object.__setattr__(
            self,
            "oracle_d_pose",
            _finite_float(self.oracle_d_pose, "oracle d_pose", nonnegative=True),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PAIR_PREIMAGE_RECEIPT_SCHEMA,
            "pair_id": self.pair_id,
            "pair_obligation_sha256": self.pair_obligation_sha256,
            "scorer_y0_identity_sha256": self.scorer_y0_identity_sha256,
            "scorer_y1_identity_sha256": self.scorer_y1_identity_sha256,
            "camera_frame0_sha256": self.camera_frame0_sha256,
            "camera_frame1_sha256": self.camera_frame1_sha256,
            "factor2_proofs": [dict(item) for item in self.factor2_proofs],
            "hard_oracle_decision_sha256": self.hard_oracle_decision_sha256,
            "hard_oracle_receipt_sha256": self.hard_oracle_receipt_sha256,
            "hard_oracle_receipt_bytes": self.hard_oracle_receipt_bytes,
            "observed_cell_logits_identity_sha256": self.observed_cell_logits_identity_sha256,
            "observed_pose6_identity_sha256": self.observed_pose6_identity_sha256,
            "observed_pose_mse": self.observed_pose_mse,
            "oracle_d_seg": self.oracle_d_seg,
            "oracle_d_pose": self.oracle_d_pose,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PairPreimageReceipt:
        expected = {
            "schema",
            "pair_id",
            "pair_obligation_sha256",
            "scorer_y0_identity_sha256",
            "scorer_y1_identity_sha256",
            "camera_frame0_sha256",
            "camera_frame1_sha256",
            "factor2_proofs",
            "hard_oracle_decision_sha256",
            "hard_oracle_receipt_sha256",
            "hard_oracle_receipt_bytes",
            "observed_cell_logits_identity_sha256",
            "observed_pose6_identity_sha256",
            "observed_pose_mse",
            "oracle_d_seg",
            "oracle_d_pose",
        }
        _exact_keys(value, expected, "pair preimage receipt")
        if value["schema"] != PAIR_PREIMAGE_RECEIPT_SCHEMA:
            raise EvaluatorObligationIRError("pair preimage receipt schema differs")
        proofs = value["factor2_proofs"]
        if not isinstance(proofs, list) or len(proofs) != 2 or any(not isinstance(item, Mapping) for item in proofs):
            raise EvaluatorObligationIRError("pair preimage proofs must serialize as two objects")
        return cls(
            pair_id=value["pair_id"],
            pair_obligation_sha256=value["pair_obligation_sha256"],
            scorer_y0_identity_sha256=value["scorer_y0_identity_sha256"],
            scorer_y1_identity_sha256=value["scorer_y1_identity_sha256"],
            camera_frame0_sha256=value["camera_frame0_sha256"],
            camera_frame1_sha256=value["camera_frame1_sha256"],
            factor2_proofs=(dict(proofs[0]), dict(proofs[1])),
            hard_oracle_decision_sha256=value["hard_oracle_decision_sha256"],
            hard_oracle_receipt_sha256=value["hard_oracle_receipt_sha256"],
            hard_oracle_receipt_bytes=value["hard_oracle_receipt_bytes"],
            observed_cell_logits_identity_sha256=value["observed_cell_logits_identity_sha256"],
            observed_pose6_identity_sha256=value["observed_pose6_identity_sha256"],
            observed_pose_mse=value["observed_pose_mse"],
            oracle_d_seg=value["oracle_d_seg"],
            oracle_d_pose=value["oracle_d_pose"],
        )


@dataclass(frozen=True)
class ExplicitV10PreimageCompileResult:
    """Hash-only receipt for real V10 solver and receiver consumption."""

    evaluator_obligation_ir_sha256: str
    frozen_space_identity_sha256: str
    coupled_state_sha256: str
    predictor_state_sha256: str
    predictor_semantic_sha256: str
    pair_count: int
    camera_height: int
    camera_width: int
    scorer_height: int
    scorer_width: int
    scorer_y0_identity_sha256: str
    scorer_y1_identity_sha256: str
    pair_receipts: tuple[PairPreimageReceipt, ...]
    solver_contract_id: str
    solver_source_sha256: str
    receiver_contract_id: str
    receiver_source_sha256: str
    receiver_packet_sha256: str
    receiver_packet_bytes: int
    y_codec_id: str
    frame0_policy_id: str
    joint_objective_id: str = JOINT_OBJECTIVE_ID
    final_admission_policy: str = FINAL_ADMISSION_POLICY
    preimage_policy: str = PREIMAGE_POLICY
    decoder_contains_scorer: bool = False
    decoder_contains_target_table: bool = False
    score_claim: bool = False
    promotion_eligible: bool = False
    archive_receipt_owed: bool = True

    def __post_init__(self) -> None:
        for label, value in (
            ("obligation IR", self.evaluator_obligation_ir_sha256),
            ("frozen space", self.frozen_space_identity_sha256),
            ("coupled state", self.coupled_state_sha256),
            ("predictor state", self.predictor_state_sha256),
            ("predictor semantic", self.predictor_semantic_sha256),
            ("scorer Y0 identity", self.scorer_y0_identity_sha256),
            ("scorer Y1 identity", self.scorer_y1_identity_sha256),
            ("solver source", self.solver_source_sha256),
            ("receiver source", self.receiver_source_sha256),
            ("receiver packet", self.receiver_packet_sha256),
        ):
            _sha256(value, f"{label} SHA-256")
        pair_count = _positive_int(self.pair_count, "result pair_count")
        for label, value in (
            ("camera_height", self.camera_height),
            ("camera_width", self.camera_width),
            ("scorer_height", self.scorer_height),
            ("scorer_width", self.scorer_width),
            ("receiver_packet_bytes", self.receiver_packet_bytes),
        ):
            _positive_int(value, label)
        if (
            type(self.pair_receipts) is not tuple
            or len(self.pair_receipts) != pair_count
            or any(not isinstance(item, PairPreimageReceipt) for item in self.pair_receipts)
        ):
            raise EvaluatorObligationIRError("result requires exactly one pair receipt per pair")
        if tuple(item.pair_id for item in self.pair_receipts) != tuple(range(pair_count)):
            raise EvaluatorObligationIRError("pair receipts must use canonical contiguous order")
        if self.solver_contract_id != V10_SOLVER_CONTRACT_ID:
            raise EvaluatorObligationIRError("V10 solver contract differs")
        if self.receiver_contract_id != RECEIVER_CONTRACT_ID:
            raise EvaluatorObligationIRError("V10 receiver contract differs")
        if self.y_codec_id != TWO_PLANE_CODEC_ID or self.frame0_policy_id != DESCRIPTION_FRAME0_POLICY_ID:
            raise EvaluatorObligationIRError("two-plane receiver codec/policy differs")
        if (
            self.joint_objective_id != JOINT_OBJECTIVE_ID
            or self.final_admission_policy != FINAL_ADMISSION_POLICY
            or self.preimage_policy != PREIMAGE_POLICY
        ):
            raise EvaluatorObligationIRError("compile result replaced the exact joint admission policy")
        false_fields = {
            "decoder_contains_scorer": self.decoder_contains_scorer,
            "decoder_contains_target_table": self.decoder_contains_target_table,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
        }
        if any(type(value) is not bool or value for value in false_fields.values()):
            raise EvaluatorObligationIRError("compile result contains a forbidden authority/payload claim")
        if type(self.archive_receipt_owed) is not bool or not self.archive_receipt_owed:
            raise EvaluatorObligationIRError("compile result cannot erase the downstream archive receipt debt")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": EXPLICIT_PREIMAGE_RESULT_SCHEMA,
            "evaluator_obligation_ir_sha256": self.evaluator_obligation_ir_sha256,
            "frozen_space_identity_sha256": self.frozen_space_identity_sha256,
            "coupled_state_sha256": self.coupled_state_sha256,
            "predictor_state_sha256": self.predictor_state_sha256,
            "predictor_semantic_sha256": self.predictor_semantic_sha256,
            "geometry": {
                "pair_count": self.pair_count,
                "camera_height": self.camera_height,
                "camera_width": self.camera_width,
                "scorer_height": self.scorer_height,
                "scorer_width": self.scorer_width,
                "channels": CHANNELS,
            },
            "scorer_y0_identity_sha256": self.scorer_y0_identity_sha256,
            "scorer_y1_identity_sha256": self.scorer_y1_identity_sha256,
            "pair_receipts": [item.as_dict() for item in self.pair_receipts],
            "solver_contract_id": self.solver_contract_id,
            "solver_source_sha256": self.solver_source_sha256,
            "receiver_contract_id": self.receiver_contract_id,
            "receiver_source_sha256": self.receiver_source_sha256,
            "receiver_packet_sha256": self.receiver_packet_sha256,
            "receiver_packet_bytes": self.receiver_packet_bytes,
            "y_codec_id": self.y_codec_id,
            "frame0_policy_id": self.frame0_policy_id,
            "joint_objective_id": self.joint_objective_id,
            "final_admission_policy": self.final_admission_policy,
            "preimage_policy": self.preimage_policy,
            "decoder_contains_scorer": self.decoder_contains_scorer,
            "decoder_contains_target_table": self.decoder_contains_target_table,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "archive_receipt_owed": self.archive_receipt_owed,
        }

    @property
    def result_sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    def to_bytes(self) -> bytes:
        body = self.as_dict()
        return canonical_json_bytes(
            {
                "schema": EXPLICIT_PREIMAGE_RESULT_ENVELOPE_SCHEMA,
                "body": body,
                "body_sha256": canonical_sha256(body),
            }
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExplicitV10PreimageCompileResult:
        expected = {
            "schema",
            "evaluator_obligation_ir_sha256",
            "frozen_space_identity_sha256",
            "coupled_state_sha256",
            "predictor_state_sha256",
            "predictor_semantic_sha256",
            "geometry",
            "scorer_y0_identity_sha256",
            "scorer_y1_identity_sha256",
            "pair_receipts",
            "solver_contract_id",
            "solver_source_sha256",
            "receiver_contract_id",
            "receiver_source_sha256",
            "receiver_packet_sha256",
            "receiver_packet_bytes",
            "y_codec_id",
            "frame0_policy_id",
            "joint_objective_id",
            "final_admission_policy",
            "preimage_policy",
            "decoder_contains_scorer",
            "decoder_contains_target_table",
            "score_claim",
            "promotion_eligible",
            "archive_receipt_owed",
        }
        _exact_keys(value, expected, "explicit V10 preimage result")
        if value["schema"] != EXPLICIT_PREIMAGE_RESULT_SCHEMA:
            raise EvaluatorObligationIRError("explicit V10 preimage result schema differs")
        geometry = value["geometry"]
        _exact_keys(
            geometry,
            {"pair_count", "camera_height", "camera_width", "scorer_height", "scorer_width", "channels"},
            "explicit V10 preimage geometry",
        )
        if geometry["channels"] != CHANNELS:
            raise EvaluatorObligationIRError("explicit V10 preimage channel geometry differs")
        rows = value["pair_receipts"]
        if not isinstance(rows, list):
            raise EvaluatorObligationIRError("pair receipts must serialize as an array")
        return cls(
            evaluator_obligation_ir_sha256=value["evaluator_obligation_ir_sha256"],
            frozen_space_identity_sha256=value["frozen_space_identity_sha256"],
            coupled_state_sha256=value["coupled_state_sha256"],
            predictor_state_sha256=value["predictor_state_sha256"],
            predictor_semantic_sha256=value["predictor_semantic_sha256"],
            pair_count=geometry["pair_count"],
            camera_height=geometry["camera_height"],
            camera_width=geometry["camera_width"],
            scorer_height=geometry["scorer_height"],
            scorer_width=geometry["scorer_width"],
            scorer_y0_identity_sha256=value["scorer_y0_identity_sha256"],
            scorer_y1_identity_sha256=value["scorer_y1_identity_sha256"],
            pair_receipts=tuple(PairPreimageReceipt.from_dict(item) for item in rows),
            solver_contract_id=value["solver_contract_id"],
            solver_source_sha256=value["solver_source_sha256"],
            receiver_contract_id=value["receiver_contract_id"],
            receiver_source_sha256=value["receiver_source_sha256"],
            receiver_packet_sha256=value["receiver_packet_sha256"],
            receiver_packet_bytes=value["receiver_packet_bytes"],
            y_codec_id=value["y_codec_id"],
            frame0_policy_id=value["frame0_policy_id"],
            joint_objective_id=value["joint_objective_id"],
            final_admission_policy=value["final_admission_policy"],
            preimage_policy=value["preimage_policy"],
            decoder_contains_scorer=value["decoder_contains_scorer"],
            decoder_contains_target_table=value["decoder_contains_target_table"],
            score_claim=value["score_claim"],
            promotion_eligible=value["promotion_eligible"],
            archive_receipt_owed=value["archive_receipt_owed"],
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> ExplicitV10PreimageCompileResult:
        try:
            envelope = decode_canonical_json(payload)
        except CoupledWitnessStateError as exc:
            raise EvaluatorObligationIRError("preimage result envelope is not canonical") from exc
        _exact_keys(envelope, {"schema", "body", "body_sha256"}, "preimage result envelope")
        if envelope["schema"] != EXPLICIT_PREIMAGE_RESULT_ENVELOPE_SCHEMA:
            raise EvaluatorObligationIRError("preimage result envelope schema differs")
        if canonical_sha256(envelope["body"]) != envelope["body_sha256"]:
            raise EvaluatorObligationIRError("preimage result body hash differs")
        result = cls.from_dict(envelope["body"])
        if result.result_sha256 != envelope["body_sha256"]:
            raise EvaluatorObligationIRError("preimage result reconstructed identity differs")
        return result

    def verify_preimages(self, scorer_y0: np.ndarray, scorer_y1: np.ndarray) -> None:
        y0, y1 = _validated_explicit_preimages(self, scorer_y0, scorer_y1)
        if _array_identity_sha256(y0) != self.scorer_y0_identity_sha256:
            raise EvaluatorObligationIRError("explicit scorer Y0 bytes differ from compile result")
        if _array_identity_sha256(y1) != self.scorer_y1_identity_sha256:
            raise EvaluatorObligationIRError("explicit scorer Y1 bytes differ from compile result")


def _validated_explicit_preimages(
    geometry: EvaluatorObligationIR | ExplicitV10PreimageCompileResult,
    scorer_y0: np.ndarray,
    scorer_y1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(scorer_y0, np.ndarray) or not isinstance(scorer_y1, np.ndarray):
        raise EvaluatorObligationIRError("explicit scorer Y0/Y1 preimages must be NumPy arrays")
    pair_count = (
        geometry.frozen_space.pair_count if isinstance(geometry, EvaluatorObligationIR) else geometry.pair_count
    )
    scorer_height = (
        geometry.frozen_space.scorer_height if isinstance(geometry, EvaluatorObligationIR) else geometry.scorer_height
    )
    scorer_width = (
        geometry.frozen_space.scorer_width if isinstance(geometry, EvaluatorObligationIR) else geometry.scorer_width
    )
    expected = (pair_count, scorer_height, scorer_width, CHANNELS)
    for label, value in (("Y0", scorer_y0), ("Y1", scorer_y1)):
        if value.dtype != np.uint8 or value.shape != expected:
            raise EvaluatorObligationIRError(f"explicit scorer {label} must be exact uint8 geometry {expected}")
    return _immutable_array(scorer_y0), _immutable_array(scorer_y1)


def _hard_decision_dict(decision: HardOracleDecision) -> dict[str, Any]:
    return {
        "schema": decision.schema,
        "admitted": decision.admitted,
        "receiver_arithmetic": decision.receiver_arithmetic,
        "realized_frame_sha256s": list(decision.realized_frame_sha256s),
        "d_seg": decision.d_seg,
        "d_pose": decision.d_pose,
        "source_hashes": dict(decision.source_hashes),
    }


def _validate_observation(
    request: PairCompileRequest,
    evidence: PairHardOracleEvidence,
) -> float:
    if evidence.pair_id != request.pair_id:
        raise EvaluatorObligationIRError("hard-oracle evidence pair_id differs")
    if evidence.evaluator_obligation_ir_sha256 != request.evaluator_obligation_ir_sha256:
        raise EvaluatorObligationIRError("hard-oracle evidence obligation IR identity differs")
    if evidence.pair_obligation_sha256 != request.pair_obligation_sha256:
        raise EvaluatorObligationIRError("hard-oracle evidence pair-obligation identity differs")
    if evidence.frame1_cell_logits.shape != (len(request.frame1_cells), N_CLASSES):
        raise EvaluatorObligationIRError("hard-oracle cell-logit count differs from pair obligations")
    for cell, logits in zip(request.frame1_cells, evidence.frame1_cell_logits, strict=True):
        observed_winner = int(np.argmax(logits))
        if observed_winner != cell.winner_class_id:
            raise EvaluatorObligationIRError("hard oracle does not realize a required frame1 winner cell")
        winner_logit = np.float32(logits[cell.winner_class_id])
        for class_id, required_margin in enumerate(cell.required_margin_by_class):
            observed_margin = float(np.float32(winner_logit - np.float32(logits[class_id])))
            if observed_margin < required_margin:
                raise EvaluatorObligationIRError("hard oracle does not realize a required frame1 class margin")
    target = np.asarray(request.conditional_frame0_pose_fibre.target_pose6, dtype=np.float32)
    residual = evidence.pose6 - target
    pose_mse = float(np.mean(residual * residual, dtype=np.float32))
    if not math.isclose(evidence.decision.d_pose, pose_mse, rel_tol=2e-6, abs_tol=2e-8):
        raise EvaluatorObligationIRError("hard-oracle d_pose differs from its observed conditional Pose fibre")
    return pose_mse


def compile_explicit_v10_preimages(
    obligation_ir: EvaluatorObligationIR,
    *,
    scorer_y0: np.ndarray,
    scorer_y1: np.ndarray,
    operator: DisjointResizeOperator,
    hard_oracle: HardOracle | None,
) -> ExplicitV10PreimageCompileResult:
    """Seal caller-supplied Y0/Y1 through the real V10 solver and receiver.

    There is intentionally no semantic-class-to-RGB implementation here.  A
    caller must supply both exact uint8 planes and a fresh hard-oracle callback.
    Each pair is realized and integer-verified by the existing V10 solver; the
    coupled two-plane packet is then decoded by the existing production
    receiver and compared byte-for-byte.  Missing or inconsistent evidence is
    refused.  This result is not an archive or score receipt.
    """

    if not isinstance(obligation_ir, EvaluatorObligationIR):
        raise EvaluatorObligationIRError("compile requires an EvaluatorObligationIR")
    _validate_frozen_oracle_custody(obligation_ir.frozen_space)
    if not isinstance(operator, DisjointResizeOperator):
        raise EvaluatorObligationIRError("compile requires the real DisjointResizeOperator")
    if not callable(hard_oracle):
        raise EvaluatorObligationIRError("compile requires a caller-supplied real hard oracle")
    if (
        operator.camera_h != obligation_ir.camera_height
        or operator.camera_w != obligation_ir.camera_width
        or operator.scorer_h != obligation_ir.frozen_space.scorer_height
        or operator.scorer_w != obligation_ir.frozen_space.scorer_width
    ):
        raise EvaluatorObligationIRError("V10 resize operator geometry differs from the obligation IR")

    y0, y1 = _validated_explicit_preimages(obligation_ir, scorer_y0, scorer_y1)
    pair_receipts: list[PairPreimageReceipt] = []
    for pair_id in range(obligation_ir.frozen_space.pair_count):
        pair_cells = tuple(cell for cell in obligation_ir.frame1_cells if cell.pair_id == pair_id)
        request = PairCompileRequest(
            pair_id=pair_id,
            evaluator_obligation_ir_sha256=obligation_ir.ir_sha256,
            pair_obligation_sha256=obligation_ir.pair_obligation_sha256(pair_id),
            scorer_y0_identity_sha256=_array_identity_sha256(y0[pair_id]),
            scorer_y1_identity_sha256=_array_identity_sha256(y1[pair_id]),
            frame1_cells=pair_cells,
            conditional_frame0_pose_fibre=obligation_ir.conditional_frame0_pose_fibres[pair_id],
        )
        evidence_holder: list[PairHardOracleEvidence] = []
        pose_mse_holder: list[float] = []

        def checked_oracle(
            frames: tuple[np.ndarray, np.ndarray],
            pair_request: PairCompileRequest = request,
            evidence_sink: list[PairHardOracleEvidence] = evidence_holder,
            pose_mse_sink: list[float] = pose_mse_holder,
        ) -> HardOracleDecision:
            evidence = hard_oracle(pair_request, frames)
            if not isinstance(evidence, PairHardOracleEvidence):
                raise EvaluatorObligationIRError("hard oracle returned no typed observation evidence")
            pose_mse = _validate_observation(pair_request, evidence)
            evidence_sink.append(evidence)
            pose_mse_sink.append(pose_mse)
            return evidence.decision

        scorer_pair = np.stack((y0[pair_id], y1[pair_id]), axis=0)
        admission: LatticeAdmission = realize_factor2_and_require_hard_oracle(
            operator,
            scorer_pair,
            checked_oracle,
        )
        if len(evidence_holder) != 1 or len(pose_mse_holder) != 1:
            raise EvaluatorObligationIRError("V10 hard oracle did not emit exactly one evidence row")
        evidence = evidence_holder[0]
        if evidence.decision is not admission.hard_oracle:
            raise EvaluatorObligationIRError("V10 admission does not retain the exact hard-oracle decision object")
        if (
            admission.hard_oracle.schema != HARD_ORACLE_SCHEMA
            or admission.hard_oracle.receiver_arithmetic != RECEIVER_ARITHMETIC
            or not admission.hard_oracle.admitted
            or dict(admission.hard_oracle.source_hashes) != EXPECTED_SOURCE_HASHES
        ):
            raise EvaluatorObligationIRError("V10 admission lacks native hard-oracle source custody")
        reverified = tuple(
            verify_factor2_uint8_scorer_plane(operator, frame, plane)
            for frame, plane in zip(admission.camera_frames, scorer_pair, strict=True)
        )
        if any(not proof.certified_exact or not proof.numerator_exact for proof in reverified):
            raise EvaluatorObligationIRError("V10 admission failed independent factor2 re-verification")
        if tuple(_proof_dict(item) for item in reverified) != tuple(_proof_dict(item) for item in admission.proofs):
            raise EvaluatorObligationIRError("V10 admission proof differs from independent re-verification")
        frame_hashes = tuple(_array_bytes_sha256(frame) for frame in admission.camera_frames)
        if frame_hashes != admission.hard_oracle.realized_frame_sha256s:
            raise EvaluatorObligationIRError("V10 hard oracle does not bind the admitted camera frames")
        oracle_evidence_bytes = canonical_json_bytes(
            {
                "schema": HARD_ORACLE_EVIDENCE_RECEIPT_SCHEMA,
                "pair_id": pair_id,
                "evaluator_obligation_ir_sha256": request.evaluator_obligation_ir_sha256,
                "pair_obligation_sha256": request.pair_obligation_sha256,
                "realized_frame_sha256s": list(frame_hashes),
                "frame1_cell_logits": _array_identity(evidence.frame1_cell_logits),
                "pose6": _array_identity(evidence.pose6),
                "hard_oracle_decision": _hard_decision_dict(admission.hard_oracle),
            }
        )
        pair_receipts.append(
            PairPreimageReceipt(
                pair_id=pair_id,
                pair_obligation_sha256=request.pair_obligation_sha256,
                scorer_y0_identity_sha256=request.scorer_y0_identity_sha256,
                scorer_y1_identity_sha256=request.scorer_y1_identity_sha256,
                camera_frame0_sha256=frame_hashes[0],
                camera_frame1_sha256=frame_hashes[1],
                factor2_proofs=(_proof_dict(reverified[0]), _proof_dict(reverified[1])),
                hard_oracle_decision_sha256=canonical_sha256(_hard_decision_dict(admission.hard_oracle)),
                hard_oracle_receipt_sha256=sha256_bytes(oracle_evidence_bytes),
                hard_oracle_receipt_bytes=len(oracle_evidence_bytes),
                observed_cell_logits_identity_sha256=_array_identity_sha256(evidence.frame1_cell_logits),
                observed_pose6_identity_sha256=_array_identity_sha256(evidence.pose6),
                observed_pose_mse=pose_mse_holder[0],
                oracle_d_seg=admission.hard_oracle.d_seg,
                oracle_d_pose=admission.hard_oracle.d_pose,
            )
        )

    packet_bytes = build_packet(
        y1,
        frame0_y_planes=y0,
        camera_height=obligation_ir.camera_height,
        camera_width=obligation_ir.camera_width,
        y_codec_id=TWO_PLANE_CODEC_ID,
        predictor_pair_ids=tuple(range(obligation_ir.frozen_space.pair_count)),
    )
    parsed = parse_packet(packet_bytes)
    decoded = decode_y_plane_pair(parsed)
    if not np.array_equal(decoded.frame0, y0) or not np.array_equal(decoded.frame1, y1):
        raise EvaluatorObligationIRError("production V10 receiver parse-back changed explicit Y0/Y1 bytes")
    if parsed.header["receiver_contract_id"] != RECEIVER_CONTRACT_ID:
        raise EvaluatorObligationIRError("production V10 receiver contract identity differs")
    if parsed.header["frame0_policy_id"] != DESCRIPTION_FRAME0_POLICY_ID:
        raise EvaluatorObligationIRError("production V10 receiver did not retain an explicit Y0 description")
    if parsed.header["launch_ready"] is not False or parsed.header["score_claim"] is not False:
        raise EvaluatorObligationIRError("production V10 packet made an unauthorized launch/score claim")

    return ExplicitV10PreimageCompileResult(
        evaluator_obligation_ir_sha256=obligation_ir.ir_sha256,
        frozen_space_identity_sha256=obligation_ir.frozen_space.identity_sha256,
        coupled_state_sha256=obligation_ir.coupled_state_sha256,
        predictor_state_sha256=obligation_ir.predictor_state_sha256,
        predictor_semantic_sha256=obligation_ir.predictor_semantic_sha256,
        pair_count=obligation_ir.frozen_space.pair_count,
        camera_height=obligation_ir.camera_height,
        camera_width=obligation_ir.camera_width,
        scorer_height=obligation_ir.frozen_space.scorer_height,
        scorer_width=obligation_ir.frozen_space.scorer_width,
        scorer_y0_identity_sha256=_array_identity_sha256(y0),
        scorer_y1_identity_sha256=_array_identity_sha256(y1),
        pair_receipts=tuple(pair_receipts),
        solver_contract_id=V10_SOLVER_CONTRACT_ID,
        solver_source_sha256=_module_source_sha256(v10_solver, "V10 solver"),
        receiver_contract_id=RECEIVER_CONTRACT_ID,
        receiver_source_sha256=_module_source_sha256(v10_receiver, "V10 receiver"),
        receiver_packet_sha256=parsed.packet_sha256,
        receiver_packet_bytes=len(packet_bytes),
        y_codec_id=TWO_PLANE_CODEC_ID,
        frame0_policy_id=DESCRIPTION_FRAME0_POLICY_ID,
    )


__all__ = [
    "CHANNELS",
    "CLASS_NAMES",
    "FINAL_ADMISSION_POLICY",
    "IR_PAYLOAD_SCOPE",
    "JOINT_OBJECTIVE_ID",
    "N_CLASSES",
    "PREIMAGE_POLICY",
    "V10_SOLVER_CONTRACT_ID",
    "CollateralOwner",
    "ConditionalFrame0PoseFibreObligation",
    "EvaluatorObligationIR",
    "EvaluatorObligationIRError",
    "ExplicitV10PreimageCompileResult",
    "Frame1CellObligation",
    "HardOracle",
    "PairCompileRequest",
    "PairHardOracleEvidence",
    "PairPreimageReceipt",
    "compile_explicit_v10_preimages",
    "frame1_pair_obligation_sha256",
]
