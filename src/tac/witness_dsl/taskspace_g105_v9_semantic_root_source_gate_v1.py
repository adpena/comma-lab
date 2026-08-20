"""Historical G105 inventory gate, superseded by the physical G109/G111 seam.

This module is deliberately a custody gate, not a second task-space IR.  The
only semantic population/evidence types it constructs are imported from
``taskspace_selected_solution_compiler``.  It reads the real V9 checkpoint
arrays and refuses to call historical or subset state a fresh G105 source.

This module remains readable for historical receipts, but it is no longer
promotion authority: it predates the physical G109 target projection, cold
``fresh_producer`` lineage, generated-Y1 conditional ownership, and exact G105
packet compiler.  New work must use
``taskspace_g105_exact_v9_semantic_root_adapter_v1`` plus the G112 total-state
partition.  The permanent superseded blocker below prevents the old FreSh/G94
contract from being mistaken for a current green gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import numpy as np

from tac.witness_dsl.taskspace_selected_solution_compiler import (
    G17EncoderOnlyTeacherOracleEvidenceV1,
    G17PairPopulationV1,
)

SCHEMA_V1: Final = "tac.taskspace_g105_v9_semantic_root_source_gate.v1"
LINEAGE_SCHEMA_V1: Final = "tac.taskspace_g105_v9_semantic_root_lineage.v1"
EXPECTED_PAIR_COUNT: Final = 600
EXPECTED_LATENT_ROWS: Final = 2 * EXPECTED_PAIR_COUNT
TEMPORAL_CODE_TRANSFORM: Final = "best_of_raw_i16le_and_delta_rice_v1"
CANONICAL_SUCCESSOR: Final = (
    "tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1"
)
Y1_LATENT_PROJECTION: Final = "root_latent[p]=code[2*p+1]"

_SOURCE_AUTHORITY_NAMES: Final = (
    "g46_preflight_receipt",
    "g46_teacher_receipt",
    "g46_target_labels",
    "source_video",
    "segnet_weights",
)
_IMPLEMENTATION_AUTHORITY_NAMES: Final = (
    "trainer",
    "levelset_generator",
    "byte_close_exporter",
    "cross_tensor_codec",
    "v10_receiver",
    "selected_solution_compiler",
    "public_layered_inflate",
    "upstream_evaluator",
)
_REQUIRED_SHARED_TENSORS: Final = (
    "in_proj.weight",
    "in_proj.bias",
    "film.weight",
    "film.bias",
    "out_sdf.weight",
    "out_sdf.bias",
    "out_tex.weight",
    "out_tex.bias",
    "palette",
)
_REQUIRED_TRAINER_MARKERS: Final = (
    b"--resume-from",
    b"--stage-checkpoints",
    b"--ckpt-every",
    b"assert_governed_admission",
    b"levelset_resume_state.npz",
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_HEX = re.compile(r"[0-9a-f]{40,64}\Z")
_HIDDEN_KEY = re.compile(r"hidden\.(\d+)\.(weight|bias)\Z")


class G105SourceGateError(ValueError):
    """Raised when source custody is malformed rather than merely absent."""


class G105BlockerCodeV1(StrEnum):
    """Exact fail-closed outcomes emitted by the source gate."""

    FRESH_G46_SOURCE_BOUND_CHECKPOINT_OWED = "G105_FRESH_G46_SOURCE_BOUND_V9_V10_CHECKPOINT_OWED"
    FRESH_LINEAGE_RECEIPT_OWED = "G105_FRESH_LINEAGE_RECEIPT_OWED"
    CHECKPOINT_NOT_FULL_N600 = "G105_CHECKPOINT_NOT_FULL_N600"
    CHECKPOINT_LEARNED_SURFACE_INCOMPLETE = "G105_CHECKPOINT_LEARNED_SURFACE_INCOMPLETE"
    CHECKPOINT_PROVENANCE_UNKNOWN = "G105_CHECKPOINT_PROVENANCE_UNKNOWN"
    CHECKPOINT_NOT_FRESH_INIT = "G105_CHECKPOINT_NOT_FRESH_INIT"
    CHECKPOINT_NOT_STAGE_RESUMABLE = "G105_CHECKPOINT_NOT_STAGE_RESUMABLE"
    CHECKPOINT_SOURCE_AUTHORITY_DRIFT = "G105_CHECKPOINT_SOURCE_AUTHORITY_DRIFT"
    SUPERSEDED_BY_PHYSICAL_G109_G111_G112 = (
        "G105_SUPERSEDED_BY_PHYSICAL_G109_G111_G112"
    )


@dataclass(frozen=True, slots=True)
class ArtifactRefV1:
    """An exact artifact coordinate used only for custody verification."""

    path: Path
    bytes: int
    sha256: str

    @classmethod
    def parse(cls, value: object, *, label: str) -> ArtifactRefV1:
        row = _mapping(value, label=label)
        _exact_keys(row, {"path", "bytes", "sha256"}, label=label)
        path = Path(_string(row["path"], label=f"{label}.path"))
        byte_count = _integer(row["bytes"], label=f"{label}.bytes", minimum=1)
        digest = _digest(row["sha256"], label=f"{label}.sha256")
        return cls(path=path, bytes=byte_count, sha256=digest)

    def reopen(self, *, label: str) -> bytes:
        try:
            mode = self.path.lstat().st_mode
        except OSError as exc:
            raise G105SourceGateError(f"{label} cannot be stated: {self.path}") from exc
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise G105SourceGateError(f"{label} must be a non-symlink regular file: {self.path}")
        try:
            payload = self.path.read_bytes()
        except OSError as exc:
            raise G105SourceGateError(f"{label} cannot be reopened: {self.path}") from exc
        if len(payload) != self.bytes:
            raise G105SourceGateError(f"{label} byte drift: expected {self.bytes}, observed {len(payload)}")
        observed = hashlib.sha256(payload).hexdigest()
        if observed != self.sha256:
            raise G105SourceGateError(f"{label} SHA-256 drift: expected {self.sha256}, observed {observed}")
        return payload


@dataclass(frozen=True, slots=True)
class TensorInventoryRowV1:
    name: str
    shape: tuple[int, ...]
    dtype: str
    content_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True, slots=True)
class V9CheckpointInventoryV1:
    """Content-derived inventory of the learned V9 generator state."""

    artifact: ArtifactRefV1
    pair_count: int
    latent_rows: int
    modulation_dim: int
    hidden_layer_ids: tuple[int, ...]
    tensors: tuple[TensorInventoryRowV1, ...]
    git_sha: str | None
    git_dirty: bool | None
    upstream_snapshot_sha256: str | None
    epoch: int | None
    fresh_init: bool
    y1_latent_projection_sha256: str | None
    chroma_enabled: bool
    pose_weight: float | None

    @property
    def full_n600(self) -> bool:
        return self.pair_count == EXPECTED_PAIR_COUNT and self.latent_rows == EXPECTED_LATENT_ROWS

    @property
    def learned_surface_complete(self) -> bool:
        names = {row.name for row in self.tensors}
        return (
            set(_REQUIRED_SHARED_TENSORS).issubset(names)
            and "code" in names
            and bool(self.hidden_layer_ids)
            and all(
                f"hidden.{index}.{suffix}" in names for index in self.hidden_layer_ids for suffix in ("weight", "bias")
            )
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.artifact.path),
            "bytes": self.artifact.bytes,
            "sha256": self.artifact.sha256,
            "pair_count": self.pair_count,
            "latent_rows": self.latent_rows,
            "modulation_dim": self.modulation_dim,
            "hidden_layer_ids": list(self.hidden_layer_ids),
            "full_n600": self.full_n600,
            "learned_surface_complete": self.learned_surface_complete,
            "chroma_enabled": self.chroma_enabled,
            "pose_weight": self.pose_weight,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "upstream_snapshot_sha256": self.upstream_snapshot_sha256,
            "epoch": self.epoch,
            "fresh_init": self.fresh_init,
            "semantic_root_counted_y1_rows": (EXPECTED_PAIR_COUNT if self.full_n600 else max(0, self.pair_count)),
            "semantic_root_y1_projection": Y1_LATENT_PROJECTION,
            "semantic_root_y1_projection_sha256": self.y1_latent_projection_sha256,
            "frame0_latents_candidate_bytes_allowed": False,
            "temporal_code_transform": TEMPORAL_CODE_TRANSFORM,
            "tensors": [row.as_dict() for row in self.tensors],
        }


@dataclass(frozen=True, slots=True)
class V9ResumeInventoryV1:
    """Physical full-state continuation artifact paired with one deploy EMA."""

    artifact: ArtifactRefV1
    epoch: int | None
    stage: str | None
    has_optimizer: bool
    has_live_state: bool
    has_ema_state: bool
    fresh_init: bool
    git_sha: str | None
    git_dirty: bool | None
    upstream_snapshot_sha256: str | None

    @property
    def stage_resumable(self) -> bool:
        return (
            self.epoch is not None
            and self.epoch > 0
            and self.stage not in (None, "", "unknown", "epoch_position_only")
            and self.has_optimizer
            and self.has_live_state
            and self.has_ema_state
            and self.fresh_init
        )


@dataclass(frozen=True, slots=True)
class G105SourceGateReceiptV1:
    status: str
    source_authority_sha256: str
    pair_population_binding_sha256: str
    teacher_evidence_identity_sha256: str
    candidate_inventory: V9CheckpointInventoryV1 | None
    historical_inventories: tuple[V9CheckpointInventoryV1, ...]
    blockers: tuple[G105BlockerCodeV1, ...]
    next_physical_producer_edge: str

    @property
    def ready_for_semantic_root_adapter(self) -> bool:
        return not self.blockers and self.candidate_inventory is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA_V1,
            "status": self.status,
            "research_only": True,
            "score_claim": False,
            "candidate_claim": False,
            "pointer_mutation_allowed": False,
            "authority_status": "SUPERSEDED_HISTORICAL_INVENTORY_ONLY",
            "canonical_successor": CANONICAL_SUCCESSOR,
            "ready_for_semantic_root_adapter": self.ready_for_semantic_root_adapter,
            "selected_solution_authority": ("src/tac/witness_dsl/taskspace_selected_solution_compiler.py"),
            "source_authority_sha256": self.source_authority_sha256,
            "pair_population_binding_sha256": self.pair_population_binding_sha256,
            "teacher_evidence_identity_sha256": self.teacher_evidence_identity_sha256,
            "learned_root_contract": {
                "population_shared_state": [
                    *_REQUIRED_SHARED_TENSORS,
                    "hidden.<contiguous_index>.weight",
                    "hidden.<contiguous_index>.bias",
                ],
                "trainer_latents": "code[2*pair+frame]",
                "expected_trainer_latent_rows": EXPECTED_LATENT_ROWS,
                "counted_semantic_root_y1_latents": Y1_LATENT_PROJECTION,
                "expected_counted_semantic_root_y1_rows": EXPECTED_PAIR_COUNT,
                "frame0_latents": "encoder-only compute overhead; discard before candidate packing",
                "exclusive_y0_owner": "G110 typed conditional Y0|final-G105-Y1",
                "historical_full_state_temporal_code_transform": TEMPORAL_CODE_TRANSFORM,
                "topology_factor_is_not_a_substitute_for_learned_rgb_state": True,
            },
            "candidate_inventory": (None if self.candidate_inventory is None else self.candidate_inventory.as_dict()),
            "historical_inventories": [item.as_dict() for item in self.historical_inventories],
            "historical_payload_reuse_allowed": False,
            "blockers": [item.value for item in self.blockers],
            "next_physical_producer_edge": self.next_physical_producer_edge,
        }

    def to_json_bytes(self) -> bytes:
        return _canonical_json_bytes(self.as_dict())


def canonical_g105_pair_population() -> G17PairPopulationV1:
    """Return the selected-compiler-owned exact public population mapping."""

    coordinates = tuple(range(EXPECTED_PAIR_COUNT))
    return G17PairPopulationV1(
        global_pair_ids=coordinates,
        source_pair_ids=coordinates,
        v9_pair_coordinates=coordinates,
        pbr_pair_coordinates=coordinates,
        obligation_ir_coordinates=coordinates,
        v10_local_coordinates=coordinates,
    )


def inspect_v9_checkpoint(artifact: ArtifactRefV1) -> V9CheckpointInventoryV1:
    """Reopen and content-inventory one real NPZ; no filename inference is used."""

    artifact.reopen(label="V9 checkpoint")
    try:
        with np.load(artifact.path, allow_pickle=False) as opened:
            arrays = {name: np.asarray(opened[name]) for name in opened.files}
    except (OSError, ValueError) as exc:
        raise G105SourceGateError(f"V9 checkpoint is not a safe NPZ: {artifact.path}") from exc

    tensor_rows: list[TensorInventoryRowV1] = []
    for name, array in sorted(arrays.items()):
        if name.startswith("__"):
            continue
        if array.dtype.kind not in "biuf":
            raise G105SourceGateError(f"V9 tensor {name!r} has unsupported dtype {array.dtype}")
        contiguous = np.ascontiguousarray(array)
        if array.dtype.kind == "f" and not bool(np.isfinite(contiguous).all()):
            raise G105SourceGateError(f"V9 tensor {name!r} contains non-finite values")
        tensor_rows.append(
            TensorInventoryRowV1(
                name=name,
                shape=tuple(int(value) for value in contiguous.shape),
                dtype=contiguous.dtype.str,
                content_sha256=hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest(),
            )
        )

    if "code" not in arrays or arrays["code"].ndim != 2:
        latent_rows = 0
        modulation_dim = 0
        pair_count = 0
    else:
        latent_rows, modulation_dim = (int(value) for value in arrays["code"].shape)
        pair_count = latent_rows // 2 if latent_rows % 2 == 0 else -1
    y1_projection_sha256 = None
    if pair_count > 0:
        y1_code = np.ascontiguousarray(arrays["code"][1::2])
        y1_projection_sha256 = hashlib.sha256(memoryview(y1_code).cast("B")).hexdigest()

    hidden_sides: dict[int, set[str]] = {}
    for name in arrays:
        match = _HIDDEN_KEY.fullmatch(name)
        if match is not None:
            hidden_sides.setdefault(int(match.group(1)), set()).add(match.group(2))
    hidden_ids = tuple(sorted(hidden_sides))
    if hidden_ids and hidden_ids != tuple(range(hidden_ids[-1] + 1)):
        raise G105SourceGateError("V9 hidden-layer indices are not contiguous from zero")

    return V9CheckpointInventoryV1(
        artifact=artifact,
        pair_count=pair_count,
        latent_rows=latent_rows,
        modulation_dim=modulation_dim,
        hidden_layer_ids=hidden_ids,
        tensors=tuple(tensor_rows),
        git_sha=_optional_scalar_string(arrays.get("__cfg_git_sha")),
        git_dirty=_optional_scalar_bool(arrays.get("__cfg_git_dirty")),
        upstream_snapshot_sha256=_optional_scalar_string(arrays.get("__cfg_upstream_snapshot_sha256")),
        epoch=_optional_scalar_int(arrays.get("__epoch")),
        fresh_init=_optional_scalar_int(arrays.get("__cfg_fresh_init")) == 1,
        y1_latent_projection_sha256=y1_projection_sha256,
        chroma_enabled=_optional_scalar_int(arrays.get("__cfg_chroma")) == 1,
        pose_weight=_optional_scalar_float(arrays.get("__cfg_w_pose")),
    )


def inspect_v9_resume_checkpoint(artifact: ArtifactRefV1) -> V9ResumeInventoryV1:
    """Reopen the exact full-state sidecar paired with a deploy checkpoint."""

    artifact.reopen(label="V9 resume checkpoint")
    try:
        with np.load(artifact.path, allow_pickle=False) as opened:
            names = tuple(opened.files)
            scalars = {
                name: np.asarray(opened[name])
                for name in (
                    "__resume_epoch",
                    "__resume_stage",
                    "__resume_has_opt",
                    "__cfg_fresh_init",
                    "__cfg_git_sha",
                    "__cfg_git_dirty",
                    "__cfg_upstream_snapshot_sha256",
                )
                if name in opened
            }
    except (OSError, ValueError) as exc:
        raise G105SourceGateError(f"V9 resume checkpoint is not a safe NPZ: {artifact.path}") from exc
    return V9ResumeInventoryV1(
        artifact=artifact,
        epoch=_optional_scalar_int(scalars.get("__resume_epoch")),
        stage=_optional_scalar_string(scalars.get("__resume_stage")),
        has_optimizer=(
            _optional_scalar_int(scalars.get("__resume_has_opt")) == 1
            and any(name.startswith("optP__") for name in names)
        ),
        has_live_state=any(name.startswith("P__") for name in names),
        has_ema_state=any(name.startswith("EMA__") for name in names),
        fresh_init=_optional_scalar_int(scalars.get("__cfg_fresh_init")) == 1,
        git_sha=_optional_scalar_string(scalars.get("__cfg_git_sha")),
        git_dirty=_optional_scalar_bool(scalars.get("__cfg_git_dirty")),
        upstream_snapshot_sha256=_optional_scalar_string(scalars.get("__cfg_upstream_snapshot_sha256")),
    )


def candidate_physical_blockers(
    candidate: V9CheckpointInventoryV1,
    resume: V9ResumeInventoryV1,
    *,
    expected_git_sha: str,
    expected_upstream_sha256: str,
) -> tuple[G105BlockerCodeV1, ...]:
    """Derive readiness from checkpoint bytes, never lineage assertions."""

    blockers: list[G105BlockerCodeV1] = [
        G105BlockerCodeV1.SUPERSEDED_BY_PHYSICAL_G109_G111_G112
    ]
    if not candidate.full_n600:
        blockers.append(G105BlockerCodeV1.CHECKPOINT_NOT_FULL_N600)
    if not candidate.learned_surface_complete:
        blockers.append(G105BlockerCodeV1.CHECKPOINT_LEARNED_SURFACE_INCOMPLETE)
    if (
        candidate.git_sha != expected_git_sha
        or candidate.git_dirty is not False
        or candidate.upstream_snapshot_sha256 != expected_upstream_sha256
    ):
        blockers.append(G105BlockerCodeV1.CHECKPOINT_SOURCE_AUTHORITY_DRIFT)
    if candidate.epoch is None or candidate.epoch <= 0:
        blockers.append(G105BlockerCodeV1.CHECKPOINT_NOT_STAGE_RESUMABLE)
    if not candidate.fresh_init:
        blockers.append(G105BlockerCodeV1.CHECKPOINT_NOT_FRESH_INIT)
    if not resume.stage_resumable or resume.epoch != candidate.epoch:
        blockers.append(G105BlockerCodeV1.CHECKPOINT_NOT_STAGE_RESUMABLE)
    if (
        resume.git_sha != expected_git_sha
        or resume.git_dirty is not False
        or resume.upstream_snapshot_sha256 != expected_upstream_sha256
    ):
        blockers.append(G105BlockerCodeV1.CHECKPOINT_SOURCE_AUTHORITY_DRIFT)
    return tuple(dict.fromkeys(blockers))


def audit_g105_source_premise(config_path: Path) -> G105SourceGateReceiptV1:
    """Verify exact G46/source/scorer custody and classify candidate availability."""

    config_bytes = config_path.read_bytes()
    config = _json_mapping(config_bytes, label="G105 source-gate config")
    _exact_keys(
        config,
        {
            "schema",
            "research_only",
            "score_claim",
            "candidate_claim",
            "pointer_mutation_allowed",
            "source_authority",
            "implementation_authority",
            "expected_provenance",
            "candidate",
            "historical_mechanism_checkpoints",
            "next_physical_producer_edge",
        },
        label="G105 source-gate config",
    )
    if config["schema"] != SCHEMA_V1:
        raise G105SourceGateError("G105 source-gate schema mismatch")
    for name, expected in (
        ("research_only", True),
        ("score_claim", False),
        ("candidate_claim", False),
        ("pointer_mutation_allowed", False),
    ):
        if config[name] is not expected:
            raise G105SourceGateError(f"G105 source-gate {name} must be {expected!r}")

    source_rows = _named_artifacts(
        config["source_authority"],
        expected_names=_SOURCE_AUTHORITY_NAMES,
        label="source_authority",
    )
    implementation_rows = _named_artifacts(
        config["implementation_authority"],
        expected_names=_IMPLEMENTATION_AUTHORITY_NAMES,
        label="implementation_authority",
    )
    expected_provenance = _mapping(config["expected_provenance"], label="expected_provenance")
    _exact_keys(
        expected_provenance,
        {"git_sha", "git_dirty", "upstream_snapshot_sha256"},
        label="expected_provenance",
    )
    expected_git_sha = _git_digest(expected_provenance["git_sha"], label="expected_provenance.git_sha")
    expected_upstream_sha256 = _digest(
        expected_provenance["upstream_snapshot_sha256"],
        label="expected_provenance.upstream_snapshot_sha256",
    )
    if expected_provenance["git_dirty"] is not False:
        raise G105SourceGateError("expected_provenance.git_dirty must be false")
    source_payloads = {name: ref.reopen(label=f"source_authority.{name}") for name, ref in source_rows.items()}
    implementation_payloads = {
        name: ref.reopen(label=f"implementation_authority.{name}") for name, ref in implementation_rows.items()
    }
    _verify_g46_authority(source_rows=source_rows, source_payloads=source_payloads)
    for marker in _REQUIRED_TRAINER_MARKERS:
        if marker not in implementation_payloads["trainer"]:
            raise G105SourceGateError(f"trainer lacks mandatory resumability/admission marker {marker!r}")

    source_authority_sha256 = hashlib.sha256(
        b"G105-SOURCE-AUTHORITY-V1\0"
        + b"".join(
            name.encode("ascii") + b"\0" + bytes.fromhex(source_rows[name].sha256) for name in _SOURCE_AUTHORITY_NAMES
        )
        + b"".join(
            name.encode("ascii") + b"\0" + bytes.fromhex(implementation_rows[name].sha256)
            for name in _IMPLEMENTATION_AUTHORITY_NAMES
        )
    ).hexdigest()
    population = canonical_g105_pair_population()
    teacher_evidence = G17EncoderOnlyTeacherOracleEvidenceV1(exact_bytes=source_payloads["g46_teacher_receipt"])

    historical_values = _sequence(
        config["historical_mechanism_checkpoints"],
        label="historical_mechanism_checkpoints",
    )
    historical = tuple(
        inspect_v9_checkpoint(ArtifactRefV1.parse(value, label=f"historical[{index}]"))
        for index, value in enumerate(historical_values)
    )

    candidate_inventory: V9CheckpointInventoryV1 | None = None
    blockers: list[G105BlockerCodeV1] = []
    candidate = config["candidate"]
    if candidate is None:
        blockers.append(G105BlockerCodeV1.FRESH_G46_SOURCE_BOUND_CHECKPOINT_OWED)
    else:
        candidate_row = _mapping(candidate, label="candidate")
        _exact_keys(
            candidate_row,
            {"checkpoint", "resume_checkpoint", "lineage_receipt"},
            label="candidate",
        )
        candidate_ref = ArtifactRefV1.parse(candidate_row["checkpoint"], label="candidate.checkpoint")
        candidate_inventory = inspect_v9_checkpoint(candidate_ref)
        resume_ref = ArtifactRefV1.parse(candidate_row["resume_checkpoint"], label="candidate.resume_checkpoint")
        resume_inventory = inspect_v9_resume_checkpoint(resume_ref)
        blockers.extend(
            candidate_physical_blockers(
                candidate_inventory,
                resume_inventory,
                expected_git_sha=expected_git_sha,
                expected_upstream_sha256=expected_upstream_sha256,
            )
        )
        lineage_ref = ArtifactRefV1.parse(candidate_row["lineage_receipt"], label="candidate.lineage_receipt")
        lineage_payload = lineage_ref.reopen(label="candidate.lineage_receipt")
        blockers.extend(
            _verify_lineage_receipt(
                lineage_payload,
                candidate=candidate_inventory,
                resume=resume_inventory,
                source_authority_sha256=source_authority_sha256,
                implementation_rows=implementation_rows,
                expected_git_sha=expected_git_sha,
                expected_upstream_sha256=expected_upstream_sha256,
            )
        )

    unique_blockers = tuple(dict.fromkeys(blockers))
    status = "SUPERSEDED_HISTORICAL_INVENTORY_ONLY"
    return G105SourceGateReceiptV1(
        status=status,
        source_authority_sha256=source_authority_sha256,
        pair_population_binding_sha256=population.binding_sha256,
        teacher_evidence_identity_sha256=teacher_evidence.identity_sha256,
        candidate_inventory=candidate_inventory,
        historical_inventories=historical,
        blockers=unique_blockers,
        next_physical_producer_edge=_string(
            config["next_physical_producer_edge"],
            label="next_physical_producer_edge",
        ),
    )


def _verify_g46_authority(
    *,
    source_rows: Mapping[str, ArtifactRefV1],
    source_payloads: Mapping[str, bytes],
) -> None:
    preflight = _json_mapping(source_payloads["g46_preflight_receipt"], label="G46 preflight")
    teacher = _json_mapping(source_payloads["g46_teacher_receipt"], label="G46 teacher receipt")
    required_teacher_values = {
        "schema": "tac.taskspace_fresh_teacher_materialization.v1",
        "encoder_only": True,
        "candidate_payload_allowed": False,
        "target_labels_encoder_only": True,
        "target_labels_serialized_in_candidate": False,
        "full_public_population_proven": True,
        "pair_count": EXPECTED_PAIR_COUNT,
        "frame_count": 2 * EXPECTED_PAIR_COUNT,
        "batch_size": 16,
        "scorer_pair_batch_size": 16,
    }
    for key, expected in required_teacher_values.items():
        if teacher.get(key) != expected:
            raise G105SourceGateError(
                f"G46 teacher receipt {key} mismatch: expected {expected!r}, got {teacher.get(key)!r}"
            )
    if preflight.get("source_video", {}).get("sha256") != source_rows["source_video"].sha256:
        raise G105SourceGateError("G46 preflight/source-video custody drift")
    if preflight.get("segnet_weights", {}).get("sha256") != source_rows["segnet_weights"].sha256:
        raise G105SourceGateError("G46 preflight/SegNet custody drift")
    target = teacher.get("target_labels")
    if not isinstance(target, dict):
        raise G105SourceGateError("G46 teacher receipt lacks target_labels mapping")
    if (
        target.get("sha256") != source_rows["g46_target_labels"].sha256
        or target.get("bytes") != source_rows["g46_target_labels"].bytes
        or target.get("shape") != [EXPECTED_PAIR_COUNT, 384, 512]
        or target.get("dtype") != "uint8"
    ):
        raise G105SourceGateError("G46 target-label aggregate custody/shape drift")


def _verify_lineage_receipt(
    payload: bytes,
    *,
    candidate: V9CheckpointInventoryV1,
    resume: V9ResumeInventoryV1,
    source_authority_sha256: str,
    implementation_rows: Mapping[str, ArtifactRefV1],
    expected_git_sha: str,
    expected_upstream_sha256: str,
) -> list[G105BlockerCodeV1]:
    row = _json_mapping(payload, label="G105 lineage receipt")
    required = {
        "schema",
        "research_only",
        "score_claim",
        "candidate_checkpoint_sha256",
        "resume_checkpoint_sha256",
        "source_authority_sha256",
        "trainer_sha256",
        "byte_close_exporter_sha256",
        "selected_solution_compiler_sha256",
        "git_sha",
        "git_dirty",
        "upstream_snapshot_sha256",
        "fresh_init",
        "pair_count",
        "latent_rows",
        "counted_y1_latent_rows",
        "y1_latent_projection_sha256",
        "checkpoint_epoch",
        "resume_stage",
        "seed",
        "stage_checkpoint_preserved",
        "resume_state_preserved",
        "governed_launch",
        "launch_argv",
    }
    _exact_keys(row, required, label="G105 lineage receipt")
    if row["schema"] != LINEAGE_SCHEMA_V1 or row["research_only"] is not True:
        raise G105SourceGateError("G105 lineage receipt schema/research axis mismatch")
    if row["score_claim"] is not False:
        raise G105SourceGateError("G105 lineage receipt cannot make a score claim")

    blockers: list[G105BlockerCodeV1] = []
    if row["candidate_checkpoint_sha256"] != candidate.artifact.sha256:
        blockers.append(G105BlockerCodeV1.CHECKPOINT_SOURCE_AUTHORITY_DRIFT)
    if row["resume_checkpoint_sha256"] != resume.artifact.sha256:
        blockers.append(G105BlockerCodeV1.CHECKPOINT_SOURCE_AUTHORITY_DRIFT)
    expected_authority = {
        "source_authority_sha256": source_authority_sha256,
        "trainer_sha256": implementation_rows["trainer"].sha256,
        "byte_close_exporter_sha256": implementation_rows["byte_close_exporter"].sha256,
        "selected_solution_compiler_sha256": implementation_rows["selected_solution_compiler"].sha256,
        "git_sha": expected_git_sha,
        "upstream_snapshot_sha256": expected_upstream_sha256,
    }
    if any(row[key] != expected for key, expected in expected_authority.items()) or row["git_dirty"] is not False:
        blockers.append(G105BlockerCodeV1.CHECKPOINT_SOURCE_AUTHORITY_DRIFT)
    if (
        row["fresh_init"] is not True
        or row["pair_count"] != EXPECTED_PAIR_COUNT
        or row["latent_rows"] != EXPECTED_LATENT_ROWS
        or row["counted_y1_latent_rows"] != EXPECTED_PAIR_COUNT
        or row["y1_latent_projection_sha256"] != candidate.y1_latent_projection_sha256
    ):
        blockers.append(G105BlockerCodeV1.CHECKPOINT_NOT_FRESH_INIT)
    if (
        row["stage_checkpoint_preserved"] is not True
        or row["resume_state_preserved"] is not True
        or row["governed_launch"] is not True
        or row["checkpoint_epoch"] != candidate.epoch
        or row["resume_stage"] != resume.stage
    ):
        blockers.append(G105BlockerCodeV1.CHECKPOINT_NOT_STAGE_RESUMABLE)
    _integer(row["seed"], label="G105 lineage seed", minimum=0)
    argv = _sequence(row["launch_argv"], label="G105 lineage launch_argv")
    if not argv or any(type(value) is not str or not value for value in argv):
        raise G105SourceGateError("G105 lineage launch_argv must contain exact strings")
    argv_strings = tuple(argv)
    required_flags = ("--fresh-init", "--stage-checkpoints", "--num-pairs", "--ckpt-every")
    if any(flag not in argv_strings for flag in required_flags):
        blockers.append(G105BlockerCodeV1.CHECKPOINT_NOT_STAGE_RESUMABLE)
    else:
        if _argv_int(argv_strings, "--num-pairs") != EXPECTED_PAIR_COUNT:
            blockers.append(G105BlockerCodeV1.CHECKPOINT_NOT_FULL_N600)
        if _argv_int(argv_strings, "--ckpt-every") <= 0:
            blockers.append(G105BlockerCodeV1.CHECKPOINT_NOT_STAGE_RESUMABLE)
    return blockers


def _argv_int(argv: Sequence[str], flag: str) -> int:
    try:
        index = argv.index(flag)
        return int(argv[index + 1])
    except (ValueError, IndexError) as exc:
        raise G105SourceGateError(f"G105 launch argv lacks an integer after {flag}") from exc


def _named_artifacts(
    value: object,
    *,
    expected_names: Sequence[str],
    label: str,
) -> dict[str, ArtifactRefV1]:
    row = _mapping(value, label=label)
    _exact_keys(row, set(expected_names), label=label)
    return {name: ArtifactRefV1.parse(row[name], label=f"{label}.{name}") for name in expected_names}


def _json_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G105SourceGateError(f"{label} is not strict JSON") from exc
    return dict(_mapping(parsed, label=label))


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise G105SourceGateError(f"{label} must be a string-keyed mapping")
    return value


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise G105SourceGateError(f"{label} must be a JSON list")
    return value


def _exact_keys(row: Mapping[str, object], expected: set[str], *, label: str) -> None:
    observed = set(row)
    if observed != expected:
        raise G105SourceGateError(
            f"{label} keys mismatch: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _string(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise G105SourceGateError(f"{label} must be a nonempty exact string")
    return value


def _integer(value: object, *, label: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise G105SourceGateError(f"{label} must be an integer >= {minimum}")
    return value


def _digest(value: object, *, label: str) -> str:
    text = _string(value, label=label)
    if _HEX64.fullmatch(text) is None:
        raise G105SourceGateError(f"{label} must be a lowercase SHA-256")
    return text


def _git_digest(value: object, *, label: str) -> str:
    text = _string(value, label=label)
    if _GIT_HEX.fullmatch(text) is None:
        raise G105SourceGateError(f"{label} must be a lowercase 40-64 digit git identity")
    return text


def _optional_scalar_string(array: np.ndarray | None) -> str | None:
    if array is None or array.ndim != 0:
        return None
    value = array.item()
    return value if type(value) is str else None


def _optional_scalar_int(array: np.ndarray | None) -> int | None:
    if array is None or array.ndim != 0:
        return None
    value = array.item()
    return int(value) if type(value) in (int, np.int64, np.int32) else None


def _optional_scalar_bool(array: np.ndarray | None) -> bool | None:
    value = _optional_scalar_int(array)
    return bool(value) if value in (0, 1) else None


def _optional_scalar_float(array: np.ndarray | None) -> float | None:
    if array is None or array.ndim != 0:
        return None
    value = array.item()
    if not isinstance(value, (float, int, np.floating, np.integer)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = audit_g105_source_premise(args.config)
    print(receipt.to_json_bytes().decode("ascii"), end="")
    return 0 if receipt.ready_for_semantic_root_adapter else 2


if __name__ == "__main__":
    raise SystemExit(main())
