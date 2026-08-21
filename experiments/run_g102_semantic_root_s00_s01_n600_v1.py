#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed S00→S01 runner for the future G102 SemanticRootY1V1.

This runner does not implement the missing semantic-root packet.  S00 records
custody and storage readiness.  S01 can execute only after a separate,
source-custodied own-lineage compiler/receiver and public codec section satisfy
the capability interface below.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Any, Final

from comma_lab.storage_tiers import (
    ExperimentStoragePlan,
    StorageTierSpec,
    default_storage_tiers,
    plan_experiment_storage,
    require_selected_storage,
)
from tac import score_geometry
from tac.process_group_kill import run_in_process_group
from tac.witness_control.taskspace_fresh_teacher_materializer_v1 import (
    load_compile_ready_materialization_receipt,
)
from tac.witness_dsl.dynamic_frontier_target import (
    DynamicFrontierTargetSnapshot,
    load_dynamic_frontier_target,
    verify_dynamic_frontier_target_snapshot,
)
from tac.witness_dsl.taskspace_selected_solution_compiler import (
    G17PlacementClassV1,
    G17WholeObjectStateV1,
    build_g17_whole_object_state_receipt,
    parse_g17_whole_object_state_receipt,
)

CONFIG_SCHEMA: Final = "tac.g102_semantic_root_s00_s01_n600_runner_config.v2"
S00_SCHEMA: Final = "tac.g102_semantic_root_s00_custody_checkpoint.v2"
S01_STAGE_SCHEMA: Final = "tac.g102_semantic_root_s01_stage_checkpoint.v2"
SELECTION_SCHEMA: Final = "tac.g102_semantic_root_s01_coupled_selection.v2"
PUBLIC_INFLATE_RECEIPT_SCHEMA: Final = "tac.g102_public_inflate_authority.v1"
EVALUATOR_PROCESS_SCHEMA: Final = "tac.g102_evaluator_process_receipt.v1"
PUBLIC_SCRATCH_CLEANUP_SCHEMA: Final = "tac.g102_public_scratch_cleanup_certificate.v1"
PUBLIC_SCRATCH_CLEANUP_COMPLETE_SCHEMA: Final = "tac.g102_public_scratch_cleanup_completion.v1"
SOURCE_LINEAGE_SCHEMA: Final = "tac.semantic_root_y1.source_lineage_manifest.v1"
CAPABILITY_INTERFACE_ID: Final = "tac.semantic_root_y1.compiler_receiver.v2"
SOURCE_LINEAGE_G17_OWNER_ID: Final = "semantic_root_source_lineage_manifest_v1"
PAIR_COUNT: Final = 600
STAGE_COUNT: Final = 5
STAGE_PAIR_SPAN: Final = 120
EVALUATOR_BATCH_SIZE: Final = 16
CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
OUTPUT_FRAME_COUNT: Final = 1200
EXPECTED_RAW_BYTES: Final = OUTPUT_FRAME_COUNT * CAMERA_HEIGHT * CAMERA_WIDTH * 3
MIN_REQUESTED_BYTES: Final = 2 * EXPECTED_RAW_BYTES
EVALUATOR_ENTRYPOINT: Final = "upstream/evaluate.py"
VIDEO_NAMES_ENTRYPOINT: Final = "upstream/public_test_video_names.txt"
PUBLIC_INFLATE_TIMEOUT_SECONDS: Final = 1800
EVALUATOR_TIMEOUT_SECONDS: Final = 1800
PUBLIC_CODEC_BLOCKER: Final = "G102_PUBLIC_SEMANTIC_ROOT_CODEC_SECTION_OWED"
COMPILER_BLOCKER: Final = "G102_FRESH_OWN_LINEAGE_SEMANTIC_ROOT_COMPILER_RECEIVER_OWED"
RGB_CLOSURE_BLOCKER: Final = "G102_SCORER_NATIVE_RGB_CHROMA_PARALLAX_POST_R_CLOSURE_OWED"
FORBIDDEN_LINEAGE_TOKENS: Final = (
    "v15",
    "c1",
    "g85",
    "g57_raster",
    "g57",
    "pr86",
    "pr130",
    "ms1",
    "ms2r_plane",
    "ms2r",
)
REQUIRED_CAPABILITY_KEYS: Final = frozenset(
    {
        "interface_id",
        "producer_identity",
        "own_lineage",
        "p_free",
        "full_population_n600",
        "label_topology_is_one_factor",
        "label_mask_palette_only",
        "scorer_native_rgb_appearance",
        "chroma_gauge",
        "parallax_gauge",
        "irreducible_rgb_quotient_seam",
        "exact_post_r_seg_closure",
        "exact_post_r_pose_closure",
        "teacher_quarantined",
        "scorer_free_receiver",
        "public_codec_section_sha256",
    }
)
REQUIRED_MODULE_CALLS: Final = (
    "semantic_root_y1_v1_capability",
    "compile_semantic_root_y1_v1_stage",
    "parse_semantic_root_y1_v1",
    "build_semantic_root_y1_v1_public_archive",
    "parse_semantic_root_y1_v1_public_archive",
    "semantic_root_y1_v1_source_lineage_manifest",
    "semantic_root_y1_v1_g17_whole_object_state",
)
LINEAGE_RECORD_FIELDS: Final = frozenset(
    {
        "path",
        "bytes",
        "sha256",
        "role",
        "candidate_dependency",
        "packaged_in_archive",
        "video_derived",
    }
)
CORE_LINEAGE_ROLES: Final = frozenset(
    {
        "OWN_LINEAGE_COMPILER_SOURCE",
        "PUBLIC_RUNTIME_SOURCE",
        "FRESH_SOURCE_VIDEO_ENCODER_INPUT",
        "FROZEN_SEGNET_ENCODER_ONLY",
        "G46_BATCH16_AUDIT_ENCODER_ONLY",
        "G46_PRIMARY_RECEIPT_ENCODER_ONLY",
        "G46_TARGET_LABELS_ENCODER_ONLY",
    }
)
ALLOWED_LINEAGE_ROLES: Final = CORE_LINEAGE_ROLES | {
    "OWN_LINEAGE_COMPILER_DEPENDENCY",
    "GENERIC_RUNTIME_DEPENDENCY",
    "HISTORICAL_PLANNING_EVIDENCE_ONLY",
}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{2,127}\Z")


class G102RunnerError(RuntimeError):
    """A custody, capability, resume, archive, or evidence invariant failed."""


class G102State(StrEnum):
    S00_CUSTODY = "S00_CUSTODY"
    S01_ROOT_PROGRAM = "S01_ROOT_PROGRAM"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise G102RunnerError("value is not finite canonical ASCII JSON") from exc


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise G102RunnerError(f"{label} must be canonical lowercase SHA-256")
    return value


def _sealed(value: dict[str, Any], *, seal_key: str) -> bytes:
    if seal_key in value:
        raise G102RunnerError("receipt already contains its self seal")
    body = dict(value)
    body[seal_key] = _sha256(_canonical_json(value))
    return _canonical_json(body)


def _parse_sealed(path: Path, *, schema: str, seal_key: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G102RunnerError(f"checkpoint cannot be reopened: {path}") from exc
    if type(value) is not dict or value.get("schema") != schema or _canonical_json(value) != payload:
        raise G102RunnerError("checkpoint schema or canonical bytes differ")
    seal = value.pop(seal_key, None)
    if seal != _sha256(_canonical_json(value)):
        raise G102RunnerError("checkpoint self seal differs")
    value[seal_key] = seal
    return value


def _immutable_atomic_write(path: Path, payload: bytes) -> None:
    """Create immutable content atomically; exact existing bytes are resumable."""

    if path.exists():
        if path.read_bytes() != payload:
            raise G102RunnerError(f"immutable artifact differs on resume: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            if path.read_bytes() != payload:
                raise G102RunnerError(f"concurrent immutable artifact differs: {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_custody_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _regular_file_identity(path: Path, *, display_path: str, role: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise G102RunnerError(f"{role} must be a regular non-symlink file: {display_path}")
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "role": role,
        "candidate_dependency": role != "PUBLIC_RUNTIME_SOURCE",
        "packaged_in_archive": False,
        "video_derived": role
        in {
            "FRESH_SOURCE_VIDEO_ENCODER_INPUT",
            "G46_BATCH16_AUDIT_ENCODER_ONLY",
            "G46_PRIMARY_RECEIPT_ENCODER_ONLY",
            "G46_TARGET_LABELS_ENCODER_ONLY",
        },
    }


def _public_runtime_records(repo_root: Path, relative_root: str) -> tuple[dict[str, Any], ...]:
    root = repo_root / relative_root
    if root.is_symlink() or not root.is_dir():
        raise G102RunnerError("public codec section must be a regular directory")
    if not (root / "inflate.sh").is_file() or (root / "inflate.sh").is_symlink():
        raise G102RunnerError("public codec section lacks actual inflate.sh")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise G102RunnerError("public codec runtime contains a symlink")
        if path.is_dir():
            continue
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            raise G102RunnerError("public codec runtime contains volatile bytecode")
        display = str(Path(relative_root) / relative)
        records.append(
            _regular_file_identity(
                path,
                display_path=display,
                role="PUBLIC_RUNTIME_SOURCE",
            )
        )
    if not records:
        raise G102RunnerError("public codec runtime tree is empty")
    return tuple(records)


def _public_runtime_sha256(records: tuple[dict[str, Any], ...]) -> str:
    return _sha256(_canonical_json(list(records)))


def _verify_self_seal(value: dict[str, Any], *, seal_key: str, label: str) -> str:
    seal = value.get(seal_key)
    if _require_sha256(seal, label=f"{label}.{seal_key}") != _sha256(
        _canonical_json({key: row for key, row in value.items() if key != seal_key})
    ):
        raise G102RunnerError(f"{label} self seal differs")
    return seal


def _load_g46_custody(
    repo_root: Path,
    config: G102RunnerConfigV1,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    audit_path = repo_root / config.g46_batch_geometry_audit_path
    if (
        audit_path.is_symlink()
        or not audit_path.is_file()
        or _sha256_file(audit_path) != config.g46_batch_geometry_audit_sha256
    ):
        raise G102RunnerError("G46 batch-geometry audit custody differs")
    try:
        audit_bytes = audit_path.read_bytes()
        audit = json.loads(audit_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G102RunnerError("G46 batch-geometry audit cannot be decoded") from exc
    if (
        type(audit) is not dict
        or audit_bytes not in {_canonical_json(audit), _canonical_json(audit) + b"\n"}
        or audit.get("verdict") != "PRIMARY_MATCHES_FROZEN_UPSTREAM_DEFAULT_BATCH_GEOMETRY"
        or audit.get("research_only") is not True
        or audit.get("score_claim") is not False
    ):
        raise G102RunnerError("G46 audit is not the canonical batch-16 authority")
    audit_seal = _verify_self_seal(audit, seal_key="audit_sha256", label="G46 audit")
    primary = audit.get("primary")
    if type(primary) is not dict or primary.get("batch_size") != EVALUATOR_BATCH_SIZE:
        raise G102RunnerError("G46 primary batch geometry differs")
    receipt_file = primary.get("receipt_file")
    if type(receipt_file) is not dict:
        raise G102RunnerError("G46 primary receipt binding is absent")
    receipt_path = Path(str(receipt_file.get("path")))
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.stat().st_size != receipt_file.get("bytes")
        or _sha256_file(receipt_path) != receipt_file.get("sha256")
    ):
        raise G102RunnerError("G46 primary receipt file custody differs")
    try:
        receipt = load_compile_ready_materialization_receipt(receipt_path)
    except Exception as exc:
        raise G102RunnerError("G46 primary receipt does not strict-reopen") from exc
    if receipt.get("receipt_sha256") != primary.get("receipt_sha256"):
        raise G102RunnerError("G46 primary receipt self identity differs")
    required_rows = (
        (
            receipt["source_video"],
            "FRESH_SOURCE_VIDEO_ENCODER_INPUT",
        ),
        (
            receipt["segnet_weights"],
            "FROZEN_SEGNET_ENCODER_ONLY",
        ),
        (
            receipt["target_labels"],
            "G46_TARGET_LABELS_ENCODER_ONLY",
        ),
    )
    records = [
        _regular_file_identity(
            audit_path,
            display_path=config.g46_batch_geometry_audit_path,
            role="G46_BATCH16_AUDIT_ENCODER_ONLY",
        ),
        _regular_file_identity(
            receipt_path,
            display_path=str(receipt_path),
            role="G46_PRIMARY_RECEIPT_ENCODER_ONLY",
        ),
    ]
    for row, role in required_rows:
        path = Path(str(row["path"]))
        record = _regular_file_identity(path, display_path=str(path), role=role)
        if record["bytes"] != row["bytes"] or record["sha256"] != row["sha256"]:
            raise G102RunnerError(f"{role} differs from strict G46 receipt")
        records.append(record)
    custody = {
        "audit_file_sha256": config.g46_batch_geometry_audit_sha256,
        "audit_seal_sha256": audit_seal,
        "primary_receipt_file_sha256": receipt_file["sha256"],
        "primary_receipt_seal_sha256": receipt["receipt_sha256"],
        "target_labels_sha256": receipt["target_labels"]["sha256"],
        "source_video_sha256": receipt["source_video"]["sha256"],
        "segnet_weights_sha256": receipt["segnet_weights"]["sha256"],
    }
    return custody, tuple(records)


@dataclass(frozen=True, slots=True)
class G102RunnerConfigV1:
    run_id: str
    seed: int
    requested_bytes: int
    reserve_free_gib: float
    compiler_source_path: str
    compiler_source_sha256: str
    public_codec_section_path: str
    public_codec_section_sha256: str
    g46_batch_geometry_audit_path: str
    g46_batch_geometry_audit_sha256: str
    evaluator_source_sha256: str
    upstream_snapshot_sha256: str
    eval_device: str
    schema: str = CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CONFIG_SCHEMA or _ID.fullmatch(self.run_id) is None:
            raise G102RunnerError("config schema or run_id differs")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise G102RunnerError("seed must be an exact nonnegative int64")
        if type(self.requested_bytes) is not int or self.requested_bytes < MIN_REQUESTED_BYTES:
            raise G102RunnerError("requested_bytes is below the double-inflate full-n600 bound")
        if (
            type(self.reserve_free_gib) not in (int, float)
            or not math.isfinite(float(self.reserve_free_gib))
            or float(self.reserve_free_gib) < 0
        ):
            raise G102RunnerError("reserve_free_gib must be finite and nonnegative")
        for name in (
            "compiler_source_sha256",
            "public_codec_section_sha256",
            "g46_batch_geometry_audit_sha256",
            "evaluator_source_sha256",
            "upstream_snapshot_sha256",
        ):
            _require_sha256(getattr(self, name), label=f"config.{name}")
        for name in (
            "compiler_source_path",
            "public_codec_section_path",
            "g46_batch_geometry_audit_path",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value or Path(value).is_absolute():
                raise G102RunnerError(f"config.{name} must be a nonempty repo-relative path")
        if self.eval_device not in {"cpu", "cuda"}:
            raise G102RunnerError("eval_device must be explicit cpu or cuda; MPS/auto are forbidden")

    @property
    def sha256(self) -> str:
        return _sha256(_canonical_json(asdict(self)))


def load_config(path: Path) -> G102RunnerConfigV1:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise G102RunnerError("G102 config cannot be read") from exc
    if type(value) is not dict or set(value) != {
        "schema",
        "run_id",
        "seed",
        "requested_bytes",
        "reserve_free_gib",
        "compiler_source_path",
        "compiler_source_sha256",
        "public_codec_section_path",
        "public_codec_section_sha256",
        "g46_batch_geometry_audit_path",
        "g46_batch_geometry_audit_sha256",
        "evaluator_source_sha256",
        "upstream_snapshot_sha256",
        "eval_device",
    }:
        raise G102RunnerError("G102 config key set differs")
    return G102RunnerConfigV1(**value)


@dataclass(frozen=True, slots=True)
class ExactCompleteArchiveRowV1:
    stage_index: int
    archive_path: str
    archive_sha256: str
    archive_bytes: int
    decoded_raw_sha256: str
    d_seg: float
    d_pose: float
    score: float
    sample_count: int
    evaluator_batch_size: int
    evaluator_source_sha256: str
    report_sha256: str
    proxy: bool = False
    research_only: bool = True
    candidate_claim: bool = False
    score_claim: bool = False

    def __post_init__(self) -> None:
        if type(self.stage_index) is not int or not 0 <= self.stage_index < STAGE_COUNT:
            raise G102RunnerError("row stage index differs")
        for name in (
            "archive_sha256",
            "decoded_raw_sha256",
            "evaluator_source_sha256",
            "report_sha256",
        ):
            _require_sha256(getattr(self, name), label=f"row.{name}")
        if type(self.archive_bytes) is not int or self.archive_bytes <= 0:
            raise G102RunnerError("row archive bytes must be positive")
        for name in ("d_seg", "d_pose", "score"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < 0:
                raise G102RunnerError(f"row.{name} must be finite and nonnegative")
        recomposed = score_geometry.contest_score(
            float(self.d_seg),
            float(self.d_pose),
            self.archive_bytes,
        )
        if abs(recomposed - float(self.score)) > 1e-12:
            raise G102RunnerError("row score differs from coupled component recomposition")
        if (
            self.sample_count != PAIR_COUNT
            or self.evaluator_batch_size != EVALUATOR_BATCH_SIZE
            or self.proxy is not False
            or self.research_only is not True
            or self.candidate_claim is not False
            or self.score_claim is not False
        ):
            raise G102RunnerError("row is partial, proxy, or carries forbidden authority")


def select_coupled_complete_row(
    rows: tuple[ExactCompleteArchiveRowV1, ...],
    target: DynamicFrontierTargetSnapshot,
) -> dict[str, Any]:
    """Select only by the complete nonlinear score against a live target."""

    verify_dynamic_frontier_target_snapshot(target)
    if not rows or any(type(row) is not ExactCompleteArchiveRowV1 for row in rows):
        raise G102RunnerError("selection requires exact complete archive rows")
    best = min(rows, key=lambda row: (row.score, row.archive_bytes, row.archive_sha256))
    return {
        "schema": SELECTION_SCHEMA,
        "selected_stage_index": best.stage_index,
        "selected_archive_sha256": best.archive_sha256,
        "selected_score": best.score,
        "dynamic_target_score": target.target_score,
        "strictly_below_dynamic_target": best.score < target.target_score,
        "pointer_sha256": target.pointer_sha256,
        "selection_rule": "min_complete_coupled_score_then_bytes_then_sha256",
        "independent_component_thresholds_used": False,
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_mutated": False,
    }


class G102SemanticRootS00S01RunnerV1:
    def __init__(self, *, repo_root: Path, config: G102RunnerConfigV1) -> None:
        self.repo_root = repo_root.resolve()
        self.config = config

    def _workload_subdir(self) -> str:
        return f"experiments/results/g102_semantic_root_s00_s01/{self.config.run_id}"

    def _lineage_policy_sha256(self, g46_custody: dict[str, Any]) -> str:
        return _sha256(
            _canonical_json(
                {
                    "schema": SOURCE_LINEAGE_SCHEMA,
                    "config_sha256": self.config.sha256,
                    "compiler_source": {
                        "path": self.config.compiler_source_path,
                        "sha256": self.config.compiler_source_sha256,
                    },
                    "public_runtime": {
                        "path": self.config.public_codec_section_path,
                        "tree_sha256": self.config.public_codec_section_sha256,
                    },
                    "g46": g46_custody,
                    "forbidden_historical_candidate_tokens": list(FORBIDDEN_LINEAGE_TOKENS),
                }
            )
        )

    def prepare_s00(
        self,
        *,
        tiers: tuple[StorageTierSpec, ...] | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        evaluator = self.repo_root / EVALUATOR_ENTRYPOINT
        if not evaluator.is_file() or _sha256_file(evaluator) != self.config.evaluator_source_sha256:
            raise G102RunnerError("S00 evaluator source custody differs")
        g46_custody, _ = _load_g46_custody(self.repo_root, self.config)
        lineage_policy_sha256 = self._lineage_policy_sha256(g46_custody)
        target = load_dynamic_frontier_target(repo_root=self.repo_root)
        effective_tiers = tiers or default_storage_tiers(
            repo_root=self.repo_root,
            reserve_free_gb=float(self.config.reserve_free_gib),
            allow_local_disk=False,
        )
        plan = plan_experiment_storage(
            effective_tiers,
            workload_subdir=self._workload_subdir(),
            requested_bytes=self.config.requested_bytes,
            min_free_bytes=0,
            create=True,
        )
        run_root = require_selected_storage(plan)
        checkpoint_path = run_root / "S00_CUSTODY" / "checkpoint.json"
        if checkpoint_path.is_file():
            resumed = _parse_sealed(
                checkpoint_path,
                schema=S00_SCHEMA,
                seal_key="checkpoint_sha256",
            )
            if resumed["config_sha256"] != self.config.sha256:
                raise G102RunnerError("S00 checkpoint belongs to another config")
            return run_root, resumed
        blockers = self._s01_presence_blockers()
        receipt = {
            "schema": S00_SCHEMA,
            "state": G102State.S00_CUSTODY,
            "config_sha256": self.config.sha256,
            "seed": self.config.seed,
            "pair_count": PAIR_COUNT,
            "stage_count": STAGE_COUNT,
            "stage_pair_span": STAGE_PAIR_SPAN,
            "evaluator_batch_size": EVALUATOR_BATCH_SIZE,
            "source_lineage_policy_sha256": lineage_policy_sha256,
            "g46_custody": g46_custody,
            "upstream_snapshot_sha256": self.config.upstream_snapshot_sha256,
            "evaluator_source_sha256": self.config.evaluator_source_sha256,
            "storage_plan": _stable_storage_plan(plan),
            "dynamic_target": {
                "pointer_path": target.pointer_path,
                "pointer_sha256": target.pointer_sha256,
                "target_score": target.target_score,
                "selection_rule": target.selection_rule,
            },
            "s01_blockers": blockers,
            "s01_ready": not blockers,
            "historical_payload_reused": False,
            "heavy_run_launched": False,
            "proxy_rows_allowed": False,
            "research_only": True,
            "candidate_claim": False,
            "score_claim": False,
            "pointer_mutated": False,
        }
        payload = _sealed(receipt, seal_key="checkpoint_sha256")
        _immutable_atomic_write(checkpoint_path, payload)
        return run_root, _parse_sealed(
            checkpoint_path,
            schema=S00_SCHEMA,
            seal_key="checkpoint_sha256",
        )

    def _s01_presence_blockers(self) -> list[str]:
        blockers: list[str] = []
        compiler = self.repo_root / self.config.compiler_source_path
        if not compiler.is_file() or _sha256_file(compiler) != self.config.compiler_source_sha256:
            blockers.append(COMPILER_BLOCKER)
        try:
            runtime_sha256 = _public_runtime_sha256(
                _public_runtime_records(
                    self.repo_root,
                    self.config.public_codec_section_path,
                )
            )
        except G102RunnerError:
            runtime_sha256 = None
        if runtime_sha256 != self.config.public_codec_section_sha256:
            blockers.append(PUBLIC_CODEC_BLOCKER)
        return blockers

    def _load_s01_module(
        self,
    ) -> tuple[
        ModuleType,
        dict[str, Any],
        Path,
        tuple[dict[str, Any], ...],
        dict[str, Any],
        str,
    ]:
        blockers = self._s01_presence_blockers()
        if blockers:
            raise G102RunnerError("S01 refused: " + ",".join(blockers))
        source = self.repo_root / self.config.compiler_source_path
        runtime_root = self.repo_root / self.config.public_codec_section_path
        spec = importlib.util.spec_from_file_location(
            f"_g102_semantic_root_{self.config.compiler_source_sha256[:12]}",
            source,
        )
        if spec is None or spec.loader is None:
            raise G102RunnerError("S01 compiler source cannot be imported")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _require_module_interface(module)
        capability = module.semantic_root_y1_v1_capability()
        _validate_capability(capability, expected_codec_sha=self.config.public_codec_section_sha256)
        runtime_records = _public_runtime_records(
            self.repo_root,
            self.config.public_codec_section_path,
        )
        if _public_runtime_sha256(runtime_records) != self.config.public_codec_section_sha256:
            raise G102RunnerError("public runtime tree changed after presence gate")
        g46_custody, g46_records = _load_g46_custody(self.repo_root, self.config)
        compiler_record = _regular_file_identity(
            source,
            display_path=self.config.compiler_source_path,
            role="OWN_LINEAGE_COMPILER_SOURCE",
        )
        core_records = tuple(
            sorted(
                (compiler_record, *runtime_records, *g46_records),
                key=lambda row: (row["role"], row["path"], row["sha256"]),
            )
        )
        return (
            module,
            capability,
            runtime_root,
            core_records,
            g46_custody,
            self._lineage_policy_sha256(g46_custody),
        )

    def run_s01(self, run_root: Path) -> dict[str, Any]:
        """Execute five cumulative 120-pair stages only after all hard gates."""

        s00 = _parse_sealed(
            run_root / "S00_CUSTODY" / "checkpoint.json",
            schema=S00_SCHEMA,
            seal_key="checkpoint_sha256",
        )
        if s00["config_sha256"] != self.config.sha256:
            raise G102RunnerError("S01 run root belongs to another config")
        (
            module,
            capability,
            runtime_root,
            core_lineage_records,
            g46_custody,
            lineage_policy_sha256,
        ) = self._load_s01_module()
        if s00["source_lineage_policy_sha256"] != lineage_policy_sha256:
            raise G102RunnerError("S01 lineage policy differs from S00 custody")
        source_lineage_context = {
            "schema": SOURCE_LINEAGE_SCHEMA,
            "lineage_policy_sha256": lineage_policy_sha256,
            "g46_custody": g46_custody,
            "required_records": list(core_lineage_records),
        }
        previous_packet: bytes | None = None
        previous_checkpoint_sha: str | None = None
        rows: list[ExactCompleteArchiveRowV1] = []
        for stage_index in range(STAGE_COUNT):
            pair_ids = tuple(
                range(
                    stage_index * STAGE_PAIR_SPAN,
                    (stage_index + 1) * STAGE_PAIR_SPAN,
                )
            )
            stage_dir = run_root / G102State.S01_ROOT_PROGRAM / f"stage_{stage_index:02d}"
            checkpoint_path = stage_dir / "checkpoint.json"
            if checkpoint_path.is_file():
                checkpoint = _parse_sealed(
                    checkpoint_path,
                    schema=S01_STAGE_SCHEMA,
                    seal_key="checkpoint_sha256",
                )
                previous_packet = module.parse_semantic_root_y1_v1_public_archive(
                    (stage_dir / "archive.zip").read_bytes()
                )
                _validate_resumed_stage(
                    checkpoint,
                    config=self.config,
                    repo_root=self.repo_root,
                    packet=previous_packet,
                    required_lineage_records=core_lineage_records,
                    lineage_policy_sha256=lineage_policy_sha256,
                    stage_index=stage_index,
                    pair_ids=pair_ids,
                    previous_checkpoint_sha=previous_checkpoint_sha,
                    stage_dir=stage_dir,
                )
                self._cleanup_public_inflate_scratch(
                    stage_dir=stage_dir,
                    archive_path=stage_dir / "archive.zip",
                    runtime_root=runtime_root,
                    reason="RESUMED_STAGE_CHECKPOINT_ALREADY_DURABLE",
                )
                row = ExactCompleteArchiveRowV1(**checkpoint["exact_complete_archive_row"])
            else:
                packet = module.compile_semantic_root_y1_v1_stage(
                    previous_packet,
                    pair_ids,
                    self.config.seed,
                    self.config.sha256,
                    source_lineage_context,
                )
                if type(packet) is not bytes or not packet:
                    raise G102RunnerError("compiler did not emit nonempty exact packet bytes")
                parsed = module.parse_semantic_root_y1_v1(packet)
                if not hasattr(parsed, "to_bytes") or parsed.to_bytes() != packet:
                    raise G102RunnerError("SemanticRootY1V1 packet parse/re-emit identity differs")
                lineage_manifest_bytes = module.semantic_root_y1_v1_source_lineage_manifest(
                    packet,
                    self.config.sha256,
                    source_lineage_context,
                )
                lineage_manifest = _validate_source_lineage_manifest(
                    lineage_manifest_bytes,
                    packet=packet,
                    config_sha256=self.config.sha256,
                    lineage_policy_sha256=lineage_policy_sha256,
                    required_records=core_lineage_records,
                    repo_root=self.repo_root,
                )
                lineage_path = stage_dir / "source_lineage_manifest.json"
                _immutable_atomic_write(lineage_path, lineage_manifest_bytes)
                archive = module.build_semantic_root_y1_v1_public_archive(packet)
                if type(archive) is not bytes or not archive:
                    raise G102RunnerError("public codec did not emit complete archive bytes")
                if module.parse_semantic_root_y1_v1_public_archive(archive) != packet:
                    raise G102RunnerError("public archive parse-back changed packet bytes")
                archive_path = stage_dir / "archive.zip"
                _immutable_atomic_write(archive_path, archive)
                row = self._double_inflate_and_evaluate(
                    archive_path=archive_path,
                    runtime_root=runtime_root,
                    stage_index=stage_index,
                    stage_dir=stage_dir,
                )
                g17_receipt = self._bind_g17_authority(
                    module=module,
                    packet=packet,
                    archive=archive,
                    row=row,
                    report=(stage_dir / "evaluate_report.txt").read_bytes(),
                    lineage_manifest=lineage_manifest_bytes,
                )
                g17_receipt_path = stage_dir / "g17_whole_object_receipt.json"
                _immutable_atomic_write(g17_receipt_path, g17_receipt)
                checkpoint_body = {
                    "schema": S01_STAGE_SCHEMA,
                    "state": G102State.S01_ROOT_PROGRAM,
                    "config_sha256": self.config.sha256,
                    "stage_index": stage_index,
                    "pair_ids": list(pair_ids),
                    "previous_checkpoint_sha256": previous_checkpoint_sha,
                    "packet_sha256": _sha256(packet),
                    "archive_file": {
                        "path": "archive.zip",
                        "bytes": len(archive),
                        "sha256": _sha256(archive),
                    },
                    "capability_evidence": capability,
                    "source_lineage_manifest": {
                        "path": lineage_path.name,
                        "file_sha256": _sha256(lineage_manifest_bytes),
                        "manifest_sha256": lineage_manifest["manifest_sha256"],
                        "lineage_policy_sha256": lineage_policy_sha256,
                        "packet_sha256": _sha256(packet),
                    },
                    "public_inflate_authority": {
                        "receipt_path": "public_inflate_authority.json",
                        "receipt_sha256": _sha256_file(stage_dir / "public_inflate_authority.json"),
                        "entrypoint": "inflate.sh",
                        "timeout_seconds": PUBLIC_INFLATE_TIMEOUT_SECONDS,
                    },
                    "evaluator_process_evidence": {
                        "receipt_path": "evaluator_process.json",
                        "receipt_sha256": _sha256_file(stage_dir / "evaluator_process.json"),
                        "timeout_seconds": EVALUATOR_TIMEOUT_SECONDS,
                        "batch_size": EVALUATOR_BATCH_SIZE,
                    },
                    "g17_selected_solution_authority": {
                        "canonical_module": ("tac.witness_dsl.taskspace_selected_solution_compiler"),
                        "receipt_path": g17_receipt_path.name,
                        "receipt_sha256": _sha256(g17_receipt),
                    },
                    "exact_complete_archive_row": asdict(row),
                    "parse_back_equal": True,
                    "double_decode_equal": True,
                    "population_pair_count": PAIR_COUNT,
                    "evaluator_batch_size": EVALUATOR_BATCH_SIZE,
                    "research_only": True,
                    "candidate_claim": False,
                    "score_claim": False,
                    "pointer_mutated": False,
                }
                _immutable_atomic_write(
                    checkpoint_path,
                    _sealed(checkpoint_body, seal_key="checkpoint_sha256"),
                )
                checkpoint = _parse_sealed(
                    checkpoint_path,
                    schema=S01_STAGE_SCHEMA,
                    seal_key="checkpoint_sha256",
                )
                self._cleanup_public_inflate_scratch(
                    stage_dir=stage_dir,
                    archive_path=archive_path,
                    runtime_root=runtime_root,
                    reason="STAGE_CHECKPOINT_AND_SCORER_REPORT_DURABLE",
                )
                previous_packet = packet
            previous_checkpoint_sha = checkpoint["checkpoint_sha256"]
            rows.append(row)
        target = load_dynamic_frontier_target(repo_root=self.repo_root)
        selection = select_coupled_complete_row(tuple(rows), target)
        selection_path = run_root / G102State.S01_ROOT_PROGRAM / "selection.json"
        _immutable_atomic_write(selection_path, _sealed(selection, seal_key="selection_sha256"))
        return selection

    def _bind_g17_authority(
        self,
        *,
        module: ModuleType,
        packet: bytes,
        archive: bytes,
        row: ExactCompleteArchiveRowV1,
        report: bytes,
        lineage_manifest: bytes,
    ) -> bytes:
        """Delegate all logical ownership/lifecycle proof to canonical G17."""

        state = module.semantic_root_y1_v1_g17_whole_object_state(
            packet,
            archive,
            report,
            lineage_manifest,
        )
        if type(state) is not G17WholeObjectStateV1:
            raise G102RunnerError("SemanticRootY1V1 did not return the exact canonical G17 whole-object state")
        try:
            receipt = build_g17_whole_object_state_receipt(state)
            exact = receipt.to_receipt_bytes()
            reopened = parse_g17_whole_object_state_receipt(exact)
        except Exception as exc:
            raise G102RunnerError("canonical G17 selected-solution authority refused SemanticRootY1V1") from exc
        if (
            reopened.to_receipt_bytes() != exact
            or reopened.archive_bytes != archive
            or reopened.population.global_pair_ids != tuple(range(PAIR_COUNT))
            or reopened.population.source_pair_ids != tuple(range(PAIR_COUNT))
            or reopened.sample_count != PAIR_COUNT
            or reopened.aggregate_d_seg != row.d_seg
            or reopened.aggregate_d_pose != row.d_pose
            or reopened.archive_nbytes != row.archive_bytes
            or reopened.total_score != row.score
        ):
            raise G102RunnerError("SemanticRootY1V1 differs from canonical G17 n600 archive/score authority")
        lineage_records = [
            record
            for record in reopened.placement_manifest.records
            if record.logical_owner.owner_id == SOURCE_LINEAGE_G17_OWNER_ID
        ]
        if not lineage_records or any(
            record.logical_owner.value.exact_bytes != lineage_manifest
            or record.placement_class is not G17PlacementClassV1.ENCODER_ONLY_EVIDENCE
            or record.packaged_inside_archive
            for record in lineage_records
        ):
            raise G102RunnerError(
                "canonical G17 receipt does not bind source-lineage manifest as encoder-only evidence"
            )
        return exact

    def _double_inflate_and_evaluate(
        self,
        *,
        archive_path: Path,
        runtime_root: Path,
        stage_index: int,
        stage_dir: Path,
    ) -> ExactCompleteArchiveRowV1:
        stage_dir.mkdir(parents=True, exist_ok=True)
        raw_a, _raw_b = self._public_double_inflate(
            archive_path=archive_path,
            runtime_root=runtime_root,
            stage_dir=stage_dir,
        )
        return self._evaluate_public_raw(
            archive_path=archive_path,
            raw_path=raw_a,
            stage_index=stage_index,
            stage_dir=stage_dir,
        )

    def _public_double_inflate(
        self,
        *,
        archive_path: Path,
        runtime_root: Path,
        stage_dir: Path,
    ) -> tuple[Path, Path]:
        receipt_path = stage_dir / "public_inflate_authority.json"
        work_root = stage_dir / ".public_inflate_work"
        if receipt_path.is_file():
            receipt = _parse_sealed(
                receipt_path,
                schema=PUBLIC_INFLATE_RECEIPT_SCHEMA,
                seal_key="receipt_sha256",
            )
            if (
                receipt["authority_complete"] is not True
                or receipt["archive_sha256"] != _sha256_file(archive_path)
                or receipt["runtime_tree_sha256"] != self.config.public_codec_section_sha256
                or receipt["timeout_seconds"] != PUBLIC_INFLATE_TIMEOUT_SECONDS
                or len(receipt["runs"]) != 2
            ):
                raise G102RunnerError("public inflate authority receipt differs")
            raw_paths = tuple(work_root / row["output_relative_path"] for row in receipt["runs"])
            if any(
                not path.is_file()
                or path.stat().st_size != EXPECTED_RAW_BYTES
                or _sha256_file(path) != receipt["runs"][index]["output_sha256"]
                for index, path in enumerate(raw_paths)
            ):
                raise G102RunnerError("public inflate resumable raw custody differs")
            return raw_paths
        if work_root.exists():
            if work_root.is_symlink() or work_root.parent != stage_dir:
                raise G102RunnerError("public inflate scratch target is unsafe")
            self._cleanup_public_inflate_scratch(
                stage_dir=stage_dir,
                archive_path=archive_path,
                runtime_root=runtime_root,
                reason="INCOMPLETE_PRE_RECEIPT_SCRATCH_REBUILD",
            )
        work_root.mkdir(parents=True)
        video_names = self.repo_root / VIDEO_NAMES_ENTRYPOINT
        runtime_records = _public_runtime_records(
            self.repo_root,
            self.config.public_codec_section_path,
        )
        runs: list[dict[str, Any]] = []
        for label in ("clean_root_a", "clean_root_b"):
            run = self._run_public_inflate_case(
                archive_path=archive_path,
                runtime_root=runtime_root,
                runtime_records=runtime_records,
                video_names=video_names,
                case_root=work_root / label,
                work_root=work_root,
            )
            runs.append(run)
            if run["returncode"] != 0 or run["timed_out"] is not False:
                break
        authority_complete = (
            len(runs) == 2
            and all(row["returncode"] == 0 and row["timed_out"] is False for row in runs)
            and all(row["output_bytes"] == EXPECTED_RAW_BYTES for row in runs)
            and all(len(row["output_records"]) == 1 and row["output_records"][0]["path"] == "0.raw" for row in runs)
            and runs[0]["output_sha256"] == runs[1]["output_sha256"]
        )
        receipt_body = {
            "schema": PUBLIC_INFLATE_RECEIPT_SCHEMA,
            "archive_sha256": _sha256_file(archive_path),
            "archive_bytes": archive_path.stat().st_size,
            "runtime_tree_sha256": _public_runtime_sha256(runtime_records),
            "runtime_inflate_sh_sha256": _sha256_file(runtime_root / "inflate.sh"),
            "video_names_sha256": _sha256_file(video_names),
            "timeout_seconds": PUBLIC_INFLATE_TIMEOUT_SECONDS,
            "runs": runs,
            "authority_complete": authority_complete,
            "double_decode_equal": authority_complete,
            "private_module_inflater_used_as_authority": False,
            "research_only": True,
            "candidate_claim": False,
        }
        _immutable_atomic_write(
            receipt_path,
            _sealed(receipt_body, seal_key="receipt_sha256"),
        )
        if not authority_complete:
            raise G102RunnerError("actual public inflate.sh did not close twice from clean extracted roots")
        return tuple(work_root / row["output_relative_path"] for row in runs)

    def _run_public_inflate_case(
        self,
        *,
        archive_path: Path,
        runtime_root: Path,
        runtime_records: tuple[dict[str, Any], ...],
        video_names: Path,
        case_root: Path,
        work_root: Path,
    ) -> dict[str, Any]:
        archive_root = case_root / "archive"
        runtime_copy = case_root / "runtime"
        output_root = case_root / "output"
        names_copy = case_root / "public_test_video_names.txt"
        archive_root.mkdir(parents=True)
        output_root.mkdir()
        member_records = _safe_extract_archive(archive_path, archive_root)
        shutil.copytree(runtime_root, runtime_copy)
        copied_records = _public_runtime_records(case_root, "runtime")
        normalized_copied = tuple(
            {
                **row,
                "path": str(Path(self.config.public_codec_section_path) / Path(row["path"]).relative_to("runtime")),
            }
            for row in copied_records
        )
        if normalized_copied != runtime_records:
            raise G102RunnerError("clean-root public runtime copy differs")
        shutil.copyfile(video_names, names_copy)
        guard_bin, guard_sha256 = _write_python_import_guard(
            case_root=case_root,
            forbidden_repo_root=self.repo_root,
            allowed_runtime_root=runtime_copy,
        )
        command = [
            "bash",
            str(runtime_copy / "inflate.sh"),
            str(archive_root),
            str(output_root),
            str(names_copy),
        ]
        environment = {
            "PATH": f"{guard_bin}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "PYTHON": str(guard_bin / "python"),
            "PYTHON_BIN": str(guard_bin / "python"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "",
        }
        started = time.monotonic()
        timed_out = False
        try:
            completed = run_in_process_group(
                command,
                cwd=case_root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=PUBLIC_INFLATE_TIMEOUT_SECONDS,
            )
            returncode: int | None = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = None
            stdout = _timeout_text(exc.stdout)
            stderr = _timeout_text(exc.stderr)
        elapsed = time.monotonic() - started
        expected_output = output_root / "0.raw"
        output_records = []
        for path in sorted(output_root.rglob("*")):
            if path.is_symlink():
                raise G102RunnerError("public inflate output contains a symlink")
            if path.is_file():
                output_records.append(
                    {
                        "path": str(path.relative_to(output_root)),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
        output_bytes = expected_output.stat().st_size if expected_output.is_file() else None
        output_sha256 = _sha256_file(expected_output) if expected_output.is_file() else None
        return {
            "argv": command,
            "cwd": str(case_root),
            "elapsed_seconds": elapsed,
            "returncode": returncode,
            "timed_out": timed_out,
            "stdout": stdout,
            "stderr": stderr,
            "environment": environment,
            "python_import_guard_sha256": guard_sha256,
            "external_repo_imports_forbidden": True,
            "archive_members": member_records,
            "output_records": output_records,
            "output_relative_path": str(expected_output.relative_to(work_root)),
            "output_bytes": output_bytes,
            "output_sha256": output_sha256,
        }

    def _evaluate_public_raw(
        self,
        *,
        archive_path: Path,
        raw_path: Path,
        stage_index: int,
        stage_dir: Path,
    ) -> ExactCompleteArchiveRowV1:
        process_path = stage_dir / "evaluator_process.json"
        report_durable = stage_dir / "evaluate_report.txt"
        if process_path.is_file():
            process = _parse_sealed(
                process_path,
                schema=EVALUATOR_PROCESS_SCHEMA,
                seal_key="receipt_sha256",
            )
            if (
                process["authority_complete"] is not True
                or process["raw_sha256"] != _sha256_file(raw_path)
                or not report_durable.is_file()
                or _sha256_file(report_durable) != process["report_sha256"]
            ):
                raise G102RunnerError("evaluator process resume custody differs")
            return self._row_from_report(
                archive_path=archive_path,
                raw_path=raw_path,
                stage_index=stage_index,
                report=report_durable.read_bytes(),
            )
        with tempfile.TemporaryDirectory(prefix=".g102_evaluate.", dir=stage_dir) as raw_tmp:
            scratch = Path(raw_tmp)
            submission = scratch / "submission"
            inflated = submission / "inflated"
            inflated.mkdir(parents=True)
            os.link(archive_path, submission / "archive.zip")
            os.link(raw_path, inflated / "0.raw")
            report_path = scratch / "report.txt"
            command = [
                sys.executable,
                str(self.repo_root / EVALUATOR_ENTRYPOINT),
                "--batch-size",
                str(EVALUATOR_BATCH_SIZE),
                "--submission-dir",
                str(submission),
                "--uncompressed-dir",
                str(self.repo_root / "upstream/videos"),
                "--seed",
                str(self.config.seed),
                "--device",
                self.config.eval_device,
                "--report",
                str(report_path),
                "--video-names-file",
                str(self.repo_root / VIDEO_NAMES_ENTRYPOINT),
            ]
            started = time.monotonic()
            timed_out = False
            try:
                completed = run_in_process_group(
                    command,
                    cwd=self.repo_root / "upstream",
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=EVALUATOR_TIMEOUT_SECONDS,
                )
                returncode: int | None = completed.returncode
                stdout = completed.stdout
                stderr = completed.stderr
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                returncode = None
                stdout = _timeout_text(exc.stdout)
                stderr = _timeout_text(exc.stderr)
            elapsed = time.monotonic() - started
            report = report_path.read_bytes() if report_path.is_file() else None
            process_body = {
                "schema": EVALUATOR_PROCESS_SCHEMA,
                "argv": command,
                "cwd": str(self.repo_root / "upstream"),
                "elapsed_seconds": elapsed,
                "timeout_seconds": EVALUATOR_TIMEOUT_SECONDS,
                "returncode": returncode,
                "timed_out": timed_out,
                "stdout": stdout,
                "stderr": stderr,
                "raw_sha256": _sha256_file(raw_path),
                "report_sha256": _sha256(report) if report is not None else None,
                "authority_complete": returncode == 0 and timed_out is False and report is not None,
                "research_only": True,
                "candidate_claim": False,
            }
            _immutable_atomic_write(
                process_path,
                _sealed(process_body, seal_key="receipt_sha256"),
            )
            if returncode != 0 or timed_out or report is None:
                raise G102RunnerError(f"exact batch-16 upstream evaluator failed rc={returncode}")
            _immutable_atomic_write(report_durable, report)
            return self._row_from_report(
                archive_path=archive_path,
                raw_path=raw_path,
                stage_index=stage_index,
                report=report,
            )

    def _row_from_report(
        self,
        *,
        archive_path: Path,
        raw_path: Path,
        stage_index: int,
        report: bytes,
    ) -> ExactCompleteArchiveRowV1:
        d_pose, d_seg, sample_count = _parse_evaluator_report(report.decode("utf-8"))
        archive_bytes = archive_path.stat().st_size
        return ExactCompleteArchiveRowV1(
            stage_index=stage_index,
            archive_path=str(archive_path),
            archive_sha256=_sha256_file(archive_path),
            archive_bytes=archive_bytes,
            decoded_raw_sha256=_sha256_file(raw_path),
            d_seg=d_seg,
            d_pose=d_pose,
            score=score_geometry.contest_score(d_seg, d_pose, archive_bytes),
            sample_count=sample_count,
            evaluator_batch_size=EVALUATOR_BATCH_SIZE,
            evaluator_source_sha256=self.config.evaluator_source_sha256,
            report_sha256=_sha256(report),
        )

    def _cleanup_public_inflate_scratch(
        self,
        *,
        stage_dir: Path,
        archive_path: Path,
        runtime_root: Path,
        reason: str,
    ) -> None:
        work_root = stage_dir / ".public_inflate_work"
        if work_root.is_symlink() or work_root.parent != stage_dir:
            raise G102RunnerError("public inflate scratch cleanup target is unsafe")

        def current_tree_records() -> list[dict[str, Any]]:
            records = []
            if not work_root.exists():
                return records
            for path in sorted(work_root.rglob("*")):
                if path.is_symlink():
                    raise G102RunnerError("public inflate scratch contains a symlink")
                if path.is_file():
                    records.append(
                        {
                            "path": str(path.relative_to(work_root)),
                            "bytes": path.stat().st_size,
                            "sha256": _sha256_file(path),
                        }
                    )
            return records

        tree_records = current_tree_records()
        archive_sha256 = _sha256_file(archive_path)
        current_map = {row["path"]: row for row in tree_records}
        candidate_paths = [
            stage_dir / "public_inflate_cleanup_certificate.json",
            *sorted(stage_dir.glob("public_inflate_incomplete_cleanup_certificate_*.json")),
        ]
        certificate_path: Path | None = None
        certificate_payload: bytes | None = None
        for candidate in candidate_paths:
            if not candidate.is_file():
                continue
            prior = _parse_sealed(
                candidate,
                schema=PUBLIC_SCRATCH_CLEANUP_SCHEMA,
                seal_key="certificate_sha256",
            )
            rebuild = prior["rebuild_inputs"]
            certified_map = {row["path"]: row for row in prior["tree_records"]}
            if (
                prior["original_path"] == str(work_root)
                and rebuild["archive_sha256"] == archive_sha256
                and rebuild["public_runtime_tree_sha256"] == self.config.public_codec_section_sha256
                and rebuild["config_sha256"] == self.config.sha256
                and all(certified_map.get(path) == row for path, row in current_map.items())
            ):
                certificate_path = candidate
                certificate_payload = candidate.read_bytes()
                break
        if certificate_path is not None:
            if work_root.exists():
                shutil.rmtree(work_root)
            self._write_cleanup_completion(
                certificate_path=certificate_path,
                certificate_payload=certificate_payload,
                work_root=work_root,
            )
            return
        if not work_root.exists():
            return

        total_bytes = sum(row["bytes"] for row in tree_records)
        public_receipt = stage_dir / "public_inflate_authority.json"
        stage_checkpoint = stage_dir / "checkpoint.json"
        evaluator_receipt = stage_dir / "evaluator_process.json"
        success = public_receipt.is_file() and stage_checkpoint.is_file()
        if success:
            reason = "CERTIFIED_SUCCESS_SCRATCH_AFTER_DURABLE_STAGE_CHECKPOINT"
            certificate_path = stage_dir / "public_inflate_cleanup_certificate.json"
        else:
            reason = "CERTIFIED_INCOMPLETE_PRE_RECEIPT_REBUILDABLE_SCRATCH"
            certificate_path = None
        certificate = {
            "schema": PUBLIC_SCRATCH_CLEANUP_SCHEMA,
            "reason": reason,
            "original_path": str(work_root),
            "tree_records": tree_records,
            "tree_sha256": _sha256(_canonical_json(tree_records)),
            "total_bytes": total_bytes,
            "rebuild_inputs": {
                "archive_path": str(archive_path),
                "archive_bytes": archive_path.stat().st_size,
                "archive_sha256": archive_sha256,
                "public_runtime_path": str(runtime_root),
                "public_runtime_tree_sha256": self.config.public_codec_section_sha256,
                "config_sha256": self.config.sha256,
                "seed": self.config.seed,
                "public_inflate_timeout_seconds": PUBLIC_INFLATE_TIMEOUT_SECONDS,
            },
            "durable_evidence": {
                "public_inflate_receipt_sha256": (_sha256_file(public_receipt) if public_receipt.is_file() else None),
                "evaluator_process_receipt_sha256": (
                    _sha256_file(evaluator_receipt) if evaluator_receipt.is_file() else None
                ),
                "stage_checkpoint_sha256": (_sha256_file(stage_checkpoint) if stage_checkpoint.is_file() else None),
            },
            "rebuild_command_contract": (
                "extract exact archive into two clean roots; copy exact public runtime; "
                "invoke inflate.sh archive output public_test_video_names.txt"
            ),
            "scratch_rebuildable": True,
            "signal_loss_allowed": False,
            "delete_after_certificate": True,
        }
        payload = _sealed(certificate, seal_key="certificate_sha256")
        if certificate_path is None:
            existing = sorted(stage_dir.glob("public_inflate_incomplete_cleanup_certificate_*.json"))
            certificate_path = next(
                (path for path in existing if path.read_bytes() == payload),
                None,
            )
            if certificate_path is None:
                certificate_path = stage_dir / f"public_inflate_incomplete_cleanup_certificate_{len(existing):03d}.json"
        if certificate_path.exists():
            if certificate_path.read_bytes() != payload:
                raise G102RunnerError("public scratch cleanup certificate differs")
        else:
            _immutable_atomic_write(certificate_path, payload)
        shutil.rmtree(work_root)
        self._write_cleanup_completion(
            certificate_path=certificate_path,
            certificate_payload=payload,
            work_root=work_root,
        )

    @staticmethod
    def _write_cleanup_completion(
        *,
        certificate_path: Path,
        certificate_payload: bytes | None,
        work_root: Path,
    ) -> None:
        if certificate_payload is None or work_root.exists():
            raise G102RunnerError("public scratch cleanup did not reach deletion completion")
        completion_path = certificate_path.with_name(f"{certificate_path.stem}_complete.json")
        completion = {
            "schema": PUBLIC_SCRATCH_CLEANUP_COMPLETE_SCHEMA,
            "certificate_path": certificate_path.name,
            "certificate_file_sha256": _sha256(certificate_payload),
            "deleted_path": str(work_root),
            "deletion_complete": True,
        }
        _immutable_atomic_write(
            completion_path,
            _sealed(completion, seal_key="completion_sha256"),
        )


def _timeout_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if type(value) is bytes:
        return value.decode("utf-8", errors="replace")
    return value


def _write_python_import_guard(
    *,
    case_root: Path,
    forbidden_repo_root: Path,
    allowed_runtime_root: Path,
) -> tuple[Path, str]:
    guard_bin = case_root / "guard_bin"
    guard_bin.mkdir()
    source = f"""#!{sys.executable}
import importlib.abc
import importlib.machinery
import pathlib
import runpy
import sys

FORBIDDEN = pathlib.Path({str(forbidden_repo_root.resolve())!r})
ALLOWED = pathlib.Path({str(allowed_runtime_root.resolve())!r})

class _NoRepoImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        origin = None if spec is None else spec.origin
        if origin not in (None, "built-in", "frozen"):
            resolved = pathlib.Path(origin).resolve()
            if resolved.is_relative_to(FORBIDDEN) and not resolved.is_relative_to(ALLOWED):
                raise ImportError("public runtime attempted external repository import: " + str(resolved))
        return spec

sys.meta_path.insert(0, _NoRepoImport())
args = sys.argv[1:]
while args and args[0] in ("-B", "-u"):
    args = args[1:]
if not args:
    raise SystemExit("guarded Python requires a script, -m module, or -c code")
if args[0] == "-m":
    sys.argv = args[1:]
    runpy.run_module(args[1], run_name="__main__", alter_sys=True)
elif args[0] == "-c":
    sys.argv = ["-c", *args[2:]]
    exec(compile(args[1], "<string>", "exec"), {{"__name__": "__main__"}})
else:
    sys.argv = args
    runpy.run_path(args[0], run_name="__main__")
"""
    payload = source.encode("utf-8")
    for name in ("python", "python3"):
        path = guard_bin / name
        path.write_bytes(payload)
        path.chmod(0o755)
    return guard_bin, _sha256(payload)


def _safe_extract_archive(archive_path: Path, destination: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if not members:
                raise G102RunnerError("public archive contains no members")
            for info in members:
                path = Path(info.filename)
                if (
                    info.filename in seen
                    or path.is_absolute()
                    or not path.parts
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or "\\" in info.filename
                    or info.flag_bits & 0x1
                    or ((info.external_attr >> 16) & 0o170000) == 0o120000
                ):
                    raise G102RunnerError("public archive contains an unsafe member")
                seen.add(info.filename)
                target = destination.joinpath(*path.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                records.append(
                    {
                        "path": info.filename,
                        "bytes": target.stat().st_size,
                        "sha256": _sha256_file(target),
                    }
                )
    except (OSError, zipfile.BadZipFile) as exc:
        raise G102RunnerError("public archive cannot be safely extracted") from exc
    if not records:
        raise G102RunnerError("public archive contains no regular members")
    return records


def _stable_storage_plan(plan: ExperimentStoragePlan) -> dict[str, Any]:
    payload = plan.to_dict()
    payload.pop("generated_at_utc", None)
    for row in payload["tiers"]:
        row.pop("free_bytes", None)
        row.pop("total_bytes", None)
        row.pop("usable_bytes", None)
    return payload


def _validate_capability(value: object, *, expected_codec_sha: str) -> None:
    if type(value) is not dict or frozenset(value) != REQUIRED_CAPABILITY_KEYS:
        raise G102RunnerError("SemanticRootY1V1 capability key set differs")
    if (
        value["interface_id"] != CAPABILITY_INTERFACE_ID
        or value["producer_identity"] != "fresh_own_lineage_semantic_root_y1_v1"
        or value["own_lineage"] is not True
        or value["p_free"] is not True
        or value["full_population_n600"] is not True
        or value["label_topology_is_one_factor"] is not True
        or value["label_mask_palette_only"] is not False
        or value["scorer_native_rgb_appearance"] is not True
        or value["chroma_gauge"] is not True
        or value["parallax_gauge"] is not True
        or value["irreducible_rgb_quotient_seam"] is not True
        or value["exact_post_r_seg_closure"] is not True
        or value["exact_post_r_pose_closure"] is not True
        or value["teacher_quarantined"] is not True
        or value["scorer_free_receiver"] is not True
        or value["public_codec_section_sha256"] != expected_codec_sha
    ):
        raise G102RunnerError(RGB_CLOSURE_BLOCKER)


def _require_module_interface(module: object) -> None:
    missing = [name for name in REQUIRED_MODULE_CALLS if not callable(getattr(module, name, None))]
    if missing:
        raise G102RunnerError(f"S01 compiler/receiver interface is incomplete: {missing}")


def _validate_source_lineage_manifest(
    payload: object,
    *,
    packet: bytes,
    config_sha256: str,
    lineage_policy_sha256: str,
    required_records: tuple[dict[str, Any], ...],
    repo_root: Path,
) -> dict[str, Any]:
    if type(payload) is not bytes:
        raise G102RunnerError("source-lineage manifest callable did not return bytes")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise G102RunnerError("source-lineage manifest is not ASCII JSON") from exc
    expected_keys = {
        "schema",
        "producer_identity",
        "packet_sha256",
        "config_sha256",
        "lineage_policy_sha256",
        "dependency_closure_sha256",
        "records",
        "manifest_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or _canonical_json(value) != payload
        or value["schema"] != SOURCE_LINEAGE_SCHEMA
        or value["producer_identity"] != "fresh_own_lineage_semantic_root_y1_v1"
        or value["packet_sha256"] != _sha256(packet)
        or value["config_sha256"] != config_sha256
        or value["lineage_policy_sha256"] != lineage_policy_sha256
    ):
        raise G102RunnerError("source-lineage manifest identity fields differ")
    records = value["records"]
    if type(records) is not list or not records:
        raise G102RunnerError("source-lineage manifest has no exact records")
    if records != sorted(
        records,
        key=lambda row: (
            row.get("role", ""),
            row.get("path", ""),
            row.get("sha256", ""),
        ),
    ):
        raise G102RunnerError("source-lineage records are not canonical ordered")
    seen: set[tuple[str, str]] = set()
    for row in records:
        if type(row) is not dict or frozenset(row) != LINEAGE_RECORD_FIELDS:
            raise G102RunnerError("source-lineage record fields differ")
        path_value = row["path"]
        role = row["role"]
        if (
            type(path_value) is not str
            or not path_value
            or role not in ALLOWED_LINEAGE_ROLES
            or type(row["bytes"]) is not int
            or row["bytes"] <= 0
            or type(row["candidate_dependency"]) is not bool
            or type(row["packaged_in_archive"]) is not bool
            or type(row["video_derived"]) is not bool
        ):
            raise G102RunnerError("source-lineage record types differ")
        _require_sha256(row["sha256"], label="source-lineage record SHA")
        identity = (role, path_value)
        if identity in seen:
            raise G102RunnerError("source-lineage record aliases a role/path")
        seen.add(identity)
        path = _resolve_custody_path(repo_root, path_value)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row["bytes"]
            or _sha256_file(path) != row["sha256"]
        ):
            raise G102RunnerError("source-lineage record file custody differs")
        lowered = f"{role}/{path_value}".lower()
        if row["candidate_dependency"] and any(token in lowered for token in FORBIDDEN_LINEAGE_TOKENS):
            raise G102RunnerError("forbidden historical dependency entered candidate lineage")
        if row["packaged_in_archive"]:
            raise G102RunnerError("source-lineage evidence or runtime was falsely packaged as payload")
    required_by_identity = {(row["role"], row["path"]): row for row in required_records}
    observed_by_identity = {(row["role"], row["path"]): row for row in records}
    if any(observed_by_identity.get(identity) != required for identity, required in required_by_identity.items()):
        raise G102RunnerError("source-lineage manifest omits or changes compiler/runtime/G46 custody")
    if {row["role"] for row in records} & CORE_LINEAGE_ROLES != CORE_LINEAGE_ROLES:
        raise G102RunnerError("source-lineage manifest lacks a required role")
    closure_sha256 = _sha256(_canonical_json(records))
    if value["dependency_closure_sha256"] != closure_sha256:
        raise G102RunnerError("source-lineage dependency closure seal differs")
    if value["manifest_sha256"] != _sha256(
        _canonical_json({key: row for key, row in value.items() if key != "manifest_sha256"})
    ):
        raise G102RunnerError("source-lineage manifest self seal differs")
    return value


def _validate_resumed_stage(
    checkpoint: dict[str, Any],
    *,
    config: G102RunnerConfigV1,
    repo_root: Path,
    packet: bytes,
    required_lineage_records: tuple[dict[str, Any], ...],
    lineage_policy_sha256: str,
    stage_index: int,
    pair_ids: tuple[int, ...],
    previous_checkpoint_sha: str | None,
    stage_dir: Path,
) -> None:
    required_checkpoint_fields = {
        "config_sha256",
        "stage_index",
        "pair_ids",
        "previous_checkpoint_sha256",
        "population_pair_count",
        "evaluator_batch_size",
        "parse_back_equal",
        "double_decode_equal",
        "candidate_claim",
        "score_claim",
        "archive_file",
        "source_lineage_manifest",
        "public_inflate_authority",
        "evaluator_process_evidence",
        "g17_selected_solution_authority",
        "exact_complete_archive_row",
    }
    if not required_checkpoint_fields.issubset(checkpoint):
        raise G102RunnerError("S01 checkpoint resume custody differs")
    if (
        checkpoint["config_sha256"] != config.sha256
        or checkpoint["stage_index"] != stage_index
        or checkpoint["pair_ids"] != list(pair_ids)
        or checkpoint["previous_checkpoint_sha256"] != previous_checkpoint_sha
        or checkpoint["population_pair_count"] != PAIR_COUNT
        or checkpoint["evaluator_batch_size"] != EVALUATOR_BATCH_SIZE
        or checkpoint["parse_back_equal"] is not True
        or checkpoint["double_decode_equal"] is not True
        or checkpoint["candidate_claim"] is not False
        or checkpoint["score_claim"] is not False
    ):
        raise G102RunnerError("S01 checkpoint resume custody differs")
    archive = checkpoint["archive_file"]
    path = stage_dir / archive["path"]
    if not path.is_file() or path.stat().st_size != archive["bytes"] or _sha256_file(path) != archive["sha256"]:
        raise G102RunnerError("S01 checkpoint archive custody differs")
    lineage = checkpoint["source_lineage_manifest"]
    lineage_path = stage_dir / lineage["path"]
    if (
        not lineage_path.is_file()
        or _sha256_file(lineage_path) != lineage["file_sha256"]
        or lineage["packet_sha256"] != _sha256(packet)
        or lineage["lineage_policy_sha256"] != lineage_policy_sha256
    ):
        raise G102RunnerError("S01 source-lineage manifest custody differs")
    parsed_lineage = _validate_source_lineage_manifest(
        lineage_path.read_bytes(),
        packet=packet,
        config_sha256=config.sha256,
        lineage_policy_sha256=lineage_policy_sha256,
        required_records=required_lineage_records,
        repo_root=repo_root,
    )
    public = checkpoint["public_inflate_authority"]
    public_path = stage_dir / public["receipt_path"]
    if (
        public["entrypoint"] != "inflate.sh"
        or public["timeout_seconds"] != PUBLIC_INFLATE_TIMEOUT_SECONDS
        or not public_path.is_file()
        or _sha256_file(public_path) != public["receipt_sha256"]
    ):
        raise G102RunnerError("S01 public inflate authority custody differs")
    public_receipt = _parse_sealed(
        public_path,
        schema=PUBLIC_INFLATE_RECEIPT_SCHEMA,
        seal_key="receipt_sha256",
    )
    if (
        public_receipt["authority_complete"] is not True
        or public_receipt["double_decode_equal"] is not True
        or public_receipt["private_module_inflater_used_as_authority"] is not False
        or public_receipt["archive_sha256"] != archive["sha256"]
        or public_receipt["runtime_tree_sha256"] != config.public_codec_section_sha256
        or len(public_receipt["runs"]) != 2
        or any(row["returncode"] != 0 or row["timed_out"] is not False for row in public_receipt["runs"])
    ):
        raise G102RunnerError("S01 public inflate authority evidence differs")
    evaluator = checkpoint["evaluator_process_evidence"]
    evaluator_path = stage_dir / evaluator["receipt_path"]
    if (
        evaluator["timeout_seconds"] != EVALUATOR_TIMEOUT_SECONDS
        or evaluator["batch_size"] != EVALUATOR_BATCH_SIZE
        or not evaluator_path.is_file()
        or _sha256_file(evaluator_path) != evaluator["receipt_sha256"]
    ):
        raise G102RunnerError("S01 evaluator process evidence custody differs")
    evaluator_receipt = _parse_sealed(
        evaluator_path,
        schema=EVALUATOR_PROCESS_SCHEMA,
        seal_key="receipt_sha256",
    )
    authority = checkpoint["g17_selected_solution_authority"]
    receipt_path = stage_dir / authority["receipt_path"]
    if (
        authority["canonical_module"] != "tac.witness_dsl.taskspace_selected_solution_compiler"
        or not receipt_path.is_file()
        or _sha256_file(receipt_path) != authority["receipt_sha256"]
    ):
        raise G102RunnerError("S01 canonical G17 authority receipt custody differs")
    try:
        receipt = parse_g17_whole_object_state_receipt(receipt_path.read_bytes())
    except Exception as exc:
        raise G102RunnerError("S01 canonical G17 authority receipt no longer strict-reopens") from exc
    row = ExactCompleteArchiveRowV1(**checkpoint["exact_complete_archive_row"])
    if (
        evaluator_receipt["authority_complete"] is not True
        or evaluator_receipt["returncode"] != 0
        or evaluator_receipt["timed_out"] is not False
        or evaluator_receipt["report_sha256"] != row.report_sha256
        or evaluator_receipt["raw_sha256"] != row.decoded_raw_sha256
    ):
        raise G102RunnerError("S01 evaluator process authority differs on resume")
    if (
        receipt.archive_bytes != path.read_bytes()
        or receipt.population.global_pair_ids != tuple(range(PAIR_COUNT))
        or receipt.population.source_pair_ids != tuple(range(PAIR_COUNT))
        or receipt.sample_count != PAIR_COUNT
        or receipt.aggregate_d_seg != row.d_seg
        or receipt.aggregate_d_pose != row.d_pose
        or receipt.archive_nbytes != row.archive_bytes
        or receipt.total_score != row.score
    ):
        raise G102RunnerError("S01 canonical G17 authority differs on resume")
    lineage_records = [
        record
        for record in receipt.placement_manifest.records
        if record.logical_owner.owner_id == SOURCE_LINEAGE_G17_OWNER_ID
    ]
    if (
        not lineage_records
        or any(
            record.logical_owner.value.exact_bytes != lineage_path.read_bytes()
            or record.placement_class is not G17PlacementClassV1.ENCODER_ONLY_EVIDENCE
            or record.packaged_inside_archive
            for record in lineage_records
        )
        or parsed_lineage["manifest_sha256"] != lineage["manifest_sha256"]
    ):
        raise G102RunnerError("S01 G17/source-lineage receipt seam differs")


def _parse_evaluator_report(text: str) -> tuple[float, float, int]:
    patterns = {
        "samples": r"Evaluation results over ([0-9]+) samples",
        "pose": r"Average PoseNet Distortion: ([0-9.eE+-]+)",
        "seg": r"Average SegNet Distortion: ([0-9.eE+-]+)",
    }
    values: dict[str, str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is None:
            raise G102RunnerError(f"evaluator report lacks {key}")
        values[key] = match.group(1)
    return float(values["pose"]), float(values["seg"]), int(values["samples"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-s00", action="store_true")
    mode.add_argument("--run-s01", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Exact S00 run root; mandatory for S01 and status reopening.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    runner = G102SemanticRootS00S01RunnerV1(
        repo_root=repo_root,
        config=load_config(args.config),
    )
    if args.prepare_s00:
        run_root, receipt = runner.prepare_s00()
        print(
            json.dumps(
                {
                    "run_root": str(run_root),
                    "state": receipt["state"],
                    "s01_ready": receipt["s01_ready"],
                    "s01_blockers": receipt["s01_blockers"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.resume_from is None:
        raise G102RunnerError("--resume-from is required for --run-s01/--status")
    if args.status:
        checkpoint = _parse_sealed(
            args.resume_from / "S00_CUSTODY" / "checkpoint.json",
            schema=S00_SCHEMA,
            seal_key="checkpoint_sha256",
        )
        print(json.dumps(checkpoint, sort_keys=True))
        return 0
    print(json.dumps(runner.run_s01(args.resume_from), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
