# SPDX-License-Identifier: MIT
"""Strict G49 selected-preimage program to G52 operand-provider bridge.

Only the G49 program decoder may produce ``Y0/Y1``.  The injected auxiliary
provider contributes chronology, current target labels, and advisory poses;
its direct source-resize planes are fingerprinted and deliberately discarded.

This bridge is not by itself ``PROGRAM_RESIDUAL_LAYERED``.  That claim requires
an independently reopened counted ZIP containing both the exact fresh semantic
archive and the exact selected-preimage packet.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

from tac.optimization.uint8_lattice_feasibility import (
    realize_factor2_uint8_scorer_plane,
)
from tac.witness_control.taskspace_fresh_scorer_plane_materializer_v1 import (
    AGGREGATE_SCHEMA as FRESH_SCORER_PLANE_AGGREGATE_SCHEMA,
)
from tac.witness_control.taskspace_fresh_scorer_plane_materializer_v1 import (
    FreshScorerPlaneMaterializationError,
    FreshScorerPlaneOperandLoaderV1,
)
from tac.witness_control.taskspace_fresh_teacher_materializer_v1 import (
    FreshTeacherMaterializationError,
    verify_sealed_payload,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    SelectedPreimageFactorRoleV1,
    TaskspaceSelectedPreimageDecoderV1,
    TaskspaceSelectedPreimageProgramError,
    TaskspaceSelectedPreimageProgramV1,
    _parse_learned_payload,
    decode_selected_preimage_pair,
    encode_selected_preimage_program,
    iter_selected_preimage_segment,
    parse_selected_preimage_program,
)

SCHEMA: Final = "tac.taskspace_selected_preimage_operand_adapter.v1"
ADMISSION_SCHEMA: Final = "tac.taskspace_selected_preimage_pre_encode_admission.v1"
OUTER_PROOF_SCHEMA: Final = "tac.taskspace_program_residual_outer_archive_proof.v1"
TERMINAL_STAGE_CHAIN_SCHEMA: Final = "tac.taskspace_selected_preimage_terminal_stage_chain.v1"
PRODUCTION_PRE_ENCODE_EVIDENCE_SCHEMA: Final = "tac.taskspace_program_residual_production_pre_encode_evidence.v1"
FIXTURE_AUXILIARY_AGGREGATE_SCHEMA: Final = "tac.taskspace_selected_preimage_operand_adapter_fixture_auxiliary.v1"
PRODUCTION_PAIR_COUNT: Final = 600
PRODUCTION_PAIRS_PER_STAGE: Final = 120
PRODUCTION_STAGE_COUNT: Final = 5
SCORER_SHAPE: Final = (384, 512, 3)
POSE_AUTHORITY: Final = "SEALED_SOURCE_CACHE_ADVISORY_ONLY"
PROGRAM_RESIDUAL_MODE: Final = "PROGRAM_RESIDUAL_LAYERED"
BLOCKED_REPRESENTATION_STATUS: Final = "PROGRAM_RESIDUAL_LAYERED_BLOCKED_OUTER_EMBEDDING_OWED"
NEXT_PRECLOSURE_GATE: Final = (
    "full-n600 coupled d_seg/d_pose plus exact counted archive rate, then "
    "public inflate parse-back/double-decode custody"
)
_FORBIDDEN_ARCHIVE_MEMBER_TOKENS: Final = (
    "c1",
    "target",
    "label",
    "pose",
    "scorer",
    "historical",
)
_FORBIDDEN_PATH_TOKENS: Final = (
    "historical",
    "/c1_",
    "/v15_",
    "prepared_plane",
    "precomputed_plane",
)


class SelectedPreimageOperandAdapterError(RuntimeError):
    """A program, decoder, auxiliary-custody, or outer-charge check failed."""


@runtime_checkable
class AuxiliaryFreshOperandProviderV1(Protocol):
    """G51-shaped provider used only for chronology, labels, and poses."""

    def iter_stages(self, *, max_pairs: int = PRODUCTION_PAIRS_PER_STAGE) -> Iterator[object]:
        """Yield immutable chronological auxiliary stages."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _lexical_absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _stable_regular_file_identity_and_bytes(
    path: str | os.PathLike[str],
    *,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    target = _lexical_absolute(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(
            os,
            "O_NOFOLLOW",
            0,
        )
    )
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise SelectedPreimageOperandAdapterError(f"{label} cannot be reopened as a no-follow regular file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1:
            raise SelectedPreimageOperandAdapterError(f"{label} is not a nonempty regular file")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            block = os.read(descriptor, min(remaining, 8 * 1024 * 1024))
            if not block:
                raise SelectedPreimageOperandAdapterError(f"{label} truncated during reopen")
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise SelectedPreimageOperandAdapterError(f"{label} grew during reopen")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    named = target.stat(follow_symlinks=False)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    named_identity = (
        named.st_dev,
        named.st_ino,
        named.st_size,
        named.st_mtime_ns,
        named.st_ctime_ns,
    )
    if not stat.S_ISREG(named.st_mode) or before_identity != after_identity or after_identity != named_identity:
        raise SelectedPreimageOperandAdapterError(f"{label} identity changed during reopen")
    payload = b"".join(chunks)
    return (
        {
            "path": str(target),
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        },
        payload,
    )


def _stable_regular_file_bytes(
    path: str | os.PathLike[str],
    *,
    label: str,
) -> bytes:
    return _stable_regular_file_identity_and_bytes(
        path,
        label=label,
    )[1]


def _regular_file_identity(
    path: str | os.PathLike[str],
    *,
    label: str,
) -> dict[str, Any]:
    return _stable_regular_file_identity_and_bytes(
        path,
        label=label,
    )[0]


def _publish_write_once(
    path: str | os.PathLike[str],
    payload: bytes,
    *,
    label: str,
) -> Mapping[str, Any]:
    destination = _lexical_absolute(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if _stable_regular_file_bytes(destination, label=label) != payload:
            raise SelectedPreimageOperandAdapterError(f"preserved {label} differs")
    else:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination, follow_symlinks=False)
            except FileExistsError as exc:
                if _stable_regular_file_bytes(destination, label=label) != payload:
                    raise SelectedPreimageOperandAdapterError(f"raced {label} differs") from exc
        finally:
            temporary.unlink(missing_ok=True)
    return _regular_file_identity(destination, label=label)


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SelectedPreimageOperandAdapterError(f"{label} must be a lowercase SHA-256")
    return value


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        if name not in value:
            raise SelectedPreimageOperandAdapterError(f"auxiliary stage lacks {name}")
        return value[name]
    try:
        return getattr(value, name)
    except AttributeError as exc:
        raise SelectedPreimageOperandAdapterError(f"auxiliary stage lacks {name}") from exc


def _readonly_contiguous(value: np.ndarray) -> np.ndarray:
    output = np.ascontiguousarray(value)
    output.flags.writeable = False
    return output


@dataclass(frozen=True, slots=True)
class _AuxiliaryReceiptStageBindingV1:
    pair_range: tuple[int, int]
    direct_source_y0_sha256: str
    direct_source_y1_sha256: str
    gt_poses_f32_sha256: str
    target_labels_sha256: str | None = None


def _load_exact_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            _stable_regular_file_bytes(
                path,
                label=label,
            )
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectedPreimageOperandAdapterError(f"{label} is not exact JSON") from exc
    if type(value) is not dict:
        raise SelectedPreimageOperandAdapterError(f"{label} must be an object")
    return value


def _fixture_auxiliary_binding(
    receipt: Mapping[str, Any],
) -> tuple[int, int, int, str, str, tuple[_AuxiliaryReceiptStageBindingV1, ...]]:
    expected_keys = {
        "aggregate_receipt_sha256",
        "gt_poses_f32_sha256",
        "historical_payload_reused",
        "pair_count",
        "pose_authority",
        "schema",
        "scorer_batch_size",
        "stage_count",
        "stage_pairs",
        "stages",
        "target_labels_sha256",
    }
    if set(receipt) != expected_keys:
        raise SelectedPreimageOperandAdapterError("fixture auxiliary receipt fields differ")
    body = dict(receipt)
    observed_seal = _require_sha256(
        body.pop("aggregate_receipt_sha256"),
        "fixture aggregate_receipt_sha256",
    )
    if _sha256_bytes(_canonical_json(body)) != observed_seal:
        raise SelectedPreimageOperandAdapterError("fixture auxiliary receipt self-seal differs")
    pair_count = receipt["pair_count"]
    stage_pairs = receipt["stage_pairs"]
    stage_count = receipt["stage_count"]
    if (
        type(pair_count) is not int
        or type(stage_pairs) is not int
        or type(stage_count) is not int
        or pair_count < 1
        or stage_pairs < 1
        or pair_count != stage_pairs * stage_count
        or receipt["scorer_batch_size"] != 16
        or receipt["pose_authority"] != POSE_AUTHORITY
        or receipt["historical_payload_reused"] is not False
    ):
        raise SelectedPreimageOperandAdapterError("fixture auxiliary receipt geometry/truth differs")
    target_sha256 = _require_sha256(
        receipt["target_labels_sha256"],
        "fixture target_labels_sha256",
    )
    poses_sha256 = _require_sha256(
        receipt["gt_poses_f32_sha256"],
        "fixture gt_poses_f32_sha256",
    )
    raw_stages = receipt["stages"]
    if type(raw_stages) is not list or len(raw_stages) != stage_count:
        raise SelectedPreimageOperandAdapterError("fixture auxiliary stages differ")
    rows: list[_AuxiliaryReceiptStageBindingV1] = []
    for stage_index, raw in enumerate(raw_stages):
        if type(raw) is not dict or set(raw) != {
            "direct_source_y0_sha256",
            "direct_source_y1_sha256",
            "gt_poses_f32_sha256",
            "pair_range",
            "target_labels_sha256",
        }:
            raise SelectedPreimageOperandAdapterError("fixture auxiliary stage binding differs")
        pair_range = (stage_index * stage_pairs, (stage_index + 1) * stage_pairs)
        if raw["pair_range"] != list(pair_range):
            raise SelectedPreimageOperandAdapterError("fixture auxiliary stage range differs")
        rows.append(
            _AuxiliaryReceiptStageBindingV1(
                pair_range=pair_range,
                direct_source_y0_sha256=_require_sha256(
                    raw["direct_source_y0_sha256"],
                    f"fixture stages[{stage_index}].direct_source_y0_sha256",
                ),
                direct_source_y1_sha256=_require_sha256(
                    raw["direct_source_y1_sha256"],
                    f"fixture stages[{stage_index}].direct_source_y1_sha256",
                ),
                gt_poses_f32_sha256=_require_sha256(
                    raw["gt_poses_f32_sha256"],
                    f"fixture stages[{stage_index}].gt_poses_f32_sha256",
                ),
                target_labels_sha256=_require_sha256(
                    raw["target_labels_sha256"],
                    f"fixture stages[{stage_index}].target_labels_sha256",
                ),
            )
        )
    return pair_count, stage_pairs, stage_count, target_sha256, poses_sha256, tuple(rows)


def _production_auxiliary_binding(
    path: Path,
    expected_sha256: str,
    receipt: Mapping[str, Any],
) -> tuple[int, int, int, str, tuple[_AuxiliaryReceiptStageBindingV1, ...]]:
    try:
        FreshScorerPlaneOperandLoaderV1.open(
            path,
            expected_sha256=expected_sha256,
        )
    except FreshScorerPlaneMaterializationError as exc:
        raise SelectedPreimageOperandAdapterError(
            "production auxiliary aggregate failed recursive G51 closure"
        ) from exc
    pair_count = receipt.get("pair_count")
    stage_pairs = receipt.get("stage_pairs")
    raw_stages = receipt.get("stages")
    teacher = receipt.get("fresh_teacher_receipt")
    target = receipt.get("target_labels")
    if (
        receipt.get("schema") != FRESH_SCORER_PLANE_AGGREGATE_SCHEMA
        or pair_count != PRODUCTION_PAIR_COUNT
        or stage_pairs != PRODUCTION_PAIRS_PER_STAGE
        or type(raw_stages) is not list
        or len(raw_stages) != PRODUCTION_STAGE_COUNT
        or type(teacher) is not dict
        or teacher.get("scorer_pair_batch_size") != 16
        or type(target) is not dict
        or target.get("shape") != [PRODUCTION_PAIR_COUNT, *SCORER_SHAPE[:2]]
        or target.get("dtype") != "uint8"
        or receipt.get("pose_authority") != POSE_AUTHORITY
    ):
        raise SelectedPreimageOperandAdapterError(
            "production auxiliary aggregate is not n600/five-stage/batch16 custody"
        )
    target_sha256 = _require_sha256(
        target.get("sha256"),
        "production target_labels.sha256",
    )
    rows: list[_AuxiliaryReceiptStageBindingV1] = []
    for stage_index, raw in enumerate(raw_stages):
        if type(raw) is not dict:
            raise SelectedPreimageOperandAdapterError("production auxiliary stage row differs")
        manifest_path = Path(str(raw.get("path", "")))
        manifest = _load_exact_json(manifest_path, f"production stage {stage_index} manifest")
        files = manifest.get("files")
        pair_range = (
            stage_index * PRODUCTION_PAIRS_PER_STAGE,
            (stage_index + 1) * PRODUCTION_PAIRS_PER_STAGE,
        )
        if (
            manifest.get("pair_range") != list(pair_range)
            or manifest.get("pose_authority") != POSE_AUTHORITY
            or type(files) is not dict
        ):
            raise SelectedPreimageOperandAdapterError(f"production auxiliary stage {stage_index} custody differs")
        try:
            y0 = files["y0_u8"]
            y1 = files["y1_u8"]
            poses = files["gt_poses_f32"]
        except KeyError as exc:
            raise SelectedPreimageOperandAdapterError(f"production auxiliary stage {stage_index} files differ") from exc
        rows.append(
            _AuxiliaryReceiptStageBindingV1(
                pair_range=pair_range,
                direct_source_y0_sha256=_require_sha256(
                    y0.get("sha256"),
                    f"production stages[{stage_index}].y0_u8.sha256",
                ),
                direct_source_y1_sha256=_require_sha256(
                    y1.get("sha256"),
                    f"production stages[{stage_index}].y1_u8.sha256",
                ),
                gt_poses_f32_sha256=_require_sha256(
                    poses.get("sha256"),
                    f"production stages[{stage_index}].gt_poses_f32.sha256",
                ),
            )
        )
    return pair_count, stage_pairs, len(rows), target_sha256, tuple(rows)


@dataclass(frozen=True, slots=True)
class AuxiliaryOperandCustodyV1:
    """Recursively reopened G51 custody; caller-supplied target hashes are forbidden."""

    aggregate_receipt_path: str
    aggregate_receipt_sha256: str
    pose_authority: str = POSE_AUTHORITY
    historical_payload_reused: bool = False
    target_labels_embedded: bool = False
    advisory_poses_embedded: bool = False
    receipt_schema: str = field(init=False)
    pair_count: int = field(init=False)
    stage_pairs: int = field(init=False)
    stage_count: int = field(init=False)
    scorer_batch_size: int = field(init=False)
    target_labels_sha256: str = field(init=False)
    gt_poses_f32_sha256: str | None = field(init=False)
    aggregate_receipt_self_seal_sha256: str = field(init=False)
    stage_bindings: tuple[_AuxiliaryReceiptStageBindingV1, ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        path = Path(self.aggregate_receipt_path)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise SelectedPreimageOperandAdapterError("auxiliary aggregate receipt must be an absolute regular file")
        normalized = path.as_posix().lower()
        if any(token in normalized for token in _FORBIDDEN_PATH_TOKENS):
            raise SelectedPreimageOperandAdapterError("auxiliary aggregate receipt names a forbidden historical input")
        _require_sha256(self.aggregate_receipt_sha256, "aggregate_receipt_sha256")
        if (
            _regular_file_identity(
                path,
                label="auxiliary aggregate receipt",
            )["sha256"]
            != self.aggregate_receipt_sha256
        ):
            raise SelectedPreimageOperandAdapterError("auxiliary aggregate receipt SHA-256 mismatch")
        if (
            self.pose_authority != POSE_AUTHORITY
            or self.historical_payload_reused is not False
            or self.target_labels_embedded is not False
            or self.advisory_poses_embedded is not False
        ):
            raise SelectedPreimageOperandAdapterError(
                "auxiliary custody cannot embed targets/poses or claim pose authority"
            )
        receipt = _load_exact_json(path, "auxiliary aggregate receipt")
        schema = receipt.get("schema")
        if schema == FIXTURE_AUXILIARY_AGGREGATE_SCHEMA:
            pair_count, stage_pairs, stage_count, target_sha256, poses_sha256, rows = _fixture_auxiliary_binding(
                receipt
            )
        elif schema == FRESH_SCORER_PLANE_AGGREGATE_SCHEMA:
            pair_count, stage_pairs, stage_count, target_sha256, rows = _production_auxiliary_binding(
                path,
                self.aggregate_receipt_sha256,
                receipt,
            )
            poses_sha256 = None
        else:
            raise SelectedPreimageOperandAdapterError("auxiliary aggregate schema differs")
        object.__setattr__(self, "receipt_schema", str(schema))
        object.__setattr__(self, "pair_count", pair_count)
        object.__setattr__(self, "stage_pairs", stage_pairs)
        object.__setattr__(self, "stage_count", stage_count)
        object.__setattr__(self, "scorer_batch_size", 16)
        object.__setattr__(self, "target_labels_sha256", target_sha256)
        object.__setattr__(self, "gt_poses_f32_sha256", poses_sha256)
        object.__setattr__(
            self,
            "aggregate_receipt_self_seal_sha256",
            _require_sha256(
                receipt.get("aggregate_receipt_sha256"),
                "aggregate receipt self-seal",
            ),
        )
        object.__setattr__(self, "stage_bindings", rows)


@dataclass(frozen=True, slots=True)
class _AuxiliaryStageIdentityV1:
    pair_range: tuple[int, int]
    pair_ids_sha256: str
    direct_source_y0_sha256: str
    direct_source_y1_sha256: str
    target_labels_sha256: str
    gt_poses_f32_sha256: str


@dataclass(frozen=True, slots=True)
class ReopenableRegularFileIdentityV1:
    """Stable regular-file identity for production lifecycle reopening."""

    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not Path(self.path).is_absolute():
            raise SelectedPreimageOperandAdapterError("production custody paths must be absolute")
        if type(self.bytes) is not int or self.bytes < 1:
            raise SelectedPreimageOperandAdapterError("production custody file bytes must be positive")
        _require_sha256(self.sha256, "production custody file sha256")
        if _regular_file_identity(
            self.path,
            label="production custody file",
        ) != asdict(self):
            raise SelectedPreimageOperandAdapterError("production custody regular-file identity differs")

    @classmethod
    def from_path(
        cls,
        path: str | os.PathLike[str],
    ) -> ReopenableRegularFileIdentityV1:
        return cls(**_regular_file_identity(path, label="production custody file"))


@dataclass(frozen=True, slots=True)
class LearnedDecoderSourceCustodyV1:
    """Exact generic source backing one counted learned quotient contract."""

    section_id: str
    decoder_contract_id: str
    implementation_source: ReopenableRegularFileIdentityV1

    def __post_init__(self) -> None:
        if (
            type(self.section_id) is not str
            or not self.section_id
            or type(self.decoder_contract_id) is not str
            or not self.decoder_contract_id
        ):
            raise SelectedPreimageOperandAdapterError("learned decoder source custody IDs must be nonempty")


@dataclass(frozen=True, slots=True)
class SelectedPreimageProductionCustodyV1:
    """Reopenable files absent from the compact G49 wire identity."""

    semantic_archive: ReopenableRegularFileIdentityV1
    semantic_compile_receipt: ReopenableRegularFileIdentityV1
    target_custody_receipt: ReopenableRegularFileIdentityV1
    target_custody_receipt_seal_sha256: str
    compiler_source: ReopenableRegularFileIdentityV1
    generic_v10_source: ReopenableRegularFileIdentityV1
    decoder_callable_source_sha256: str
    learned_decoder_sources: tuple[LearnedDecoderSourceCustodyV1, ...] = ()

    def verify_against(
        self,
        program: TaskspaceSelectedPreimageProgramV1,
        decoder: TaskspaceSelectedPreimageDecoderV1,
    ) -> None:
        for file_identity in (
            self.semantic_archive,
            self.semantic_compile_receipt,
            self.target_custody_receipt,
            self.compiler_source,
            self.generic_v10_source,
            *(row.implementation_source for row in self.learned_decoder_sources),
        ):
            ReopenableRegularFileIdentityV1(**asdict(file_identity))
        semantic = program.semantic_program_identity
        try:
            target_receipt = json.loads(
                _stable_regular_file_bytes(
                    self.target_custody_receipt.path,
                    label="target custody receipt",
                )
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SelectedPreimageOperandAdapterError("target custody receipt is not JSON") from exc
        target_labels = target_receipt.get("target_labels") if type(target_receipt) is dict else None
        if type(target_receipt) is not dict:
            raise SelectedPreimageOperandAdapterError("target custody receipt must be an object")
        try:
            verify_sealed_payload(
                target_receipt,
                hash_field="receipt_sha256",
            )
        except FreshTeacherMaterializationError as exc:
            raise SelectedPreimageOperandAdapterError("target custody receipt self-seal differs") from exc
        learned_factors = {
            factor.section_id: (
                factor,
                _parse_learned_payload(factor.payload, factor=factor),
            )
            for factor in program.factors
            if factor.role is SelectedPreimageFactorRoleV1.LEARNED_IRREDUCIBLE_QUOTIENT
        }
        learned_sources = {row.section_id: row for row in self.learned_decoder_sources}
        if len(learned_sources) != len(self.learned_decoder_sources) or set(learned_sources) != set(learned_factors):
            raise SelectedPreimageOperandAdapterError("learned decoder sources do not cover exact learned factors")
        for section_id, (factor, learned) in learned_factors.items():
            row = learned_sources[section_id]
            try:
                contract_id = decoder.learned_quotient_decoder_contract_id(factor)
                source_sha256 = decoder.learned_quotient_decoder_implementation_source_sha256(factor)
            except TaskspaceSelectedPreimageProgramError as exc:
                raise SelectedPreimageOperandAdapterError(
                    f"learned decoder source cannot reopen for {section_id}"
                ) from exc
            if (
                row.decoder_contract_id != learned.decoder_contract_id
                or contract_id != learned.decoder_contract_id
                or row.implementation_source.sha256 != learned.decoder_implementation_source_sha256
                or source_sha256 != learned.decoder_implementation_source_sha256
            ):
                raise SelectedPreimageOperandAdapterError(f"learned decoder source custody differs for {section_id}")
        if (
            self.semantic_archive.bytes != semantic.compiled_semantic_archive_bytes
            or self.semantic_archive.sha256 != semantic.compiled_semantic_archive_sha256
            or self.semantic_compile_receipt.sha256 != semantic.fresh_compile_receipt_sha256
            or _require_sha256(
                self.target_custody_receipt_seal_sha256,
                "target_custody_receipt_seal_sha256",
            )
            != program.target_custody_identity.target_custody_receipt_sha256
            or self.compiler_source.sha256 != semantic.compiler_source_sha256
            or _require_sha256(
                self.decoder_callable_source_sha256,
                "decoder_callable_source_sha256",
            )
            != program.decoder_identity.implementation_source_sha256
            or self.generic_v10_source.sha256 != self.decoder_callable_source_sha256
            or Path(self.generic_v10_source.path)
            != Path(inspect.getsourcefile(realize_factor2_uint8_scorer_plane) or "").resolve()
            or type(target_receipt) is not dict
            or target_receipt.get("schema") != "tac.taskspace_fresh_teacher_materialization.v1"
            or target_receipt.get("receipt_sha256") != self.target_custody_receipt_seal_sha256
            or target_receipt.get("pair_count") != PRODUCTION_PAIR_COUNT
            or target_receipt.get("scorer_pair_batch_size") != 16
            or target_receipt.get("batch_geometry_matches_upstream_default") is not True
            or target_receipt.get("encoder_only") is not True
            or target_receipt.get("candidate_payload_allowed") is not False
            or type(target_labels) is not dict
            or target_labels.get("sha256") != program.target_custody_identity.target_bank_sha256
            or target_labels.get("shape") != [PRODUCTION_PAIR_COUNT, *SCORER_SHAPE[:2]]
            or target_labels.get("dtype") != "uint8"
        ):
            raise SelectedPreimageOperandAdapterError("production reopenable custody differs from G49 identities")


@dataclass(frozen=True, slots=True)
class SelectedPreimagePreEncodeAdmissionV1:
    """Recurrent machine-checkable identity for one stage before encoding."""

    schema: str
    status: str
    stage_index: int
    stage_count: int
    pair_range: tuple[int, int]
    representation_source: str
    program_packet_sha256: str
    program_packet_bytes: int
    semantic_archive_sha256: str
    semantic_archive_bytes: int
    fresh_semantic_compile_receipt_sha256: str
    target_custody_receipt_sha256: str
    target_bank_sha256: str
    decoder_id: str
    decoder_implementation_source_sha256: str
    factor_section_ids: tuple[str, ...]
    behavior_changing_factor_section_ids: tuple[str, ...]
    factor_payload_bytes_inside_packet: int
    scorer_y0_sha256: str
    scorer_y1_sha256: str
    discarded_direct_source_y0_sha256: str
    discarded_direct_source_y1_sha256: str
    target_labels_sha256: str
    gt_poses_f32_sha256: str
    auxiliary_planes_forwarded: bool
    g49_decode_custody_verified: bool
    prior_stage_chain_sha256: str
    stage_chain_sha256: str
    representation_status: str
    next_preclosure_gate: str

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SelectedPreimageFreshOperandStageV1:
    """G52-compatible stage whose planes came only from the G49 decoder."""

    pair_range: tuple[int, int]
    pair_ids: np.ndarray
    y0_u8: np.ndarray
    y1_u8: np.ndarray
    target_labels_u8: np.ndarray
    gt_poses_f32: np.ndarray
    pre_encode_admission: SelectedPreimagePreEncodeAdmissionV1
    pose_authority: str = POSE_AUTHORITY


@dataclass(frozen=True, slots=True)
class ProgramResidualOuterArchiveProofV1:
    """Physical counted-member proof; still not public/evaluator closure."""

    schema: str
    status: str
    representation_mode: str
    archive_path: str
    archive_bytes: int
    archive_sha256: str
    outer_member_names: tuple[str, str]
    outer_members_partition_exactly: bool
    semantic_member_name: str
    semantic_member_compressed_bytes: int
    semantic_archive_bytes: int
    semantic_archive_sha256: str
    program_member_name: str
    program_member_compressed_bytes: int
    program_packet_bytes: int
    program_packet_sha256: str
    factor_payload_bytes_inside_packet: int
    packet_byte_homes_partition_exactly: bool
    learned_decoder_contracts: tuple[tuple[str, str, str], ...]
    generic_learned_decoder_source_bound: bool
    target_labels_embedded: bool
    advisory_poses_embedded: bool
    precontainer_counted_source_bytes: int
    next_preclosure_gate: str

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def _validate_auxiliary_stage(
    stage: object,
    *,
    expected_start: int,
    expected_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pair_range = _field(stage, "pair_range")
    if pair_range not in ((expected_start, expected_stop), [expected_start, expected_stop]):
        raise SelectedPreimageOperandAdapterError("auxiliary stages are not the exact chronological lattice")
    pair_ids = np.asarray(_field(stage, "pair_ids"))
    count = expected_stop - expected_start
    if (
        pair_ids.shape != (count,)
        or not np.issubdtype(pair_ids.dtype, np.integer)
        or not np.array_equal(
            pair_ids.astype(np.int64, copy=False),
            np.arange(expected_start, expected_stop, dtype=np.int64),
        )
    ):
        raise SelectedPreimageOperandAdapterError("auxiliary pair IDs are not exact chronological IDs")
    y0 = np.asarray(_field(stage, "y0_u8"))
    y1 = np.asarray(_field(stage, "y1_u8"))
    labels = np.asarray(_field(stage, "target_labels_u8"))
    poses = np.asarray(_field(stage, "gt_poses_f32"))
    expected_plane_shape = (count, *SCORER_SHAPE)
    if (
        y0.dtype != np.uint8
        or y1.dtype != np.uint8
        or y0.shape != expected_plane_shape
        or y1.shape != expected_plane_shape
    ):
        raise SelectedPreimageOperandAdapterError("auxiliary direct-source planes have invalid scorer geometry")
    if labels.dtype != np.uint8 or labels.shape != expected_plane_shape[:3]:
        raise SelectedPreimageOperandAdapterError("auxiliary target labels have invalid geometry or dtype")
    if poses.dtype != np.float32 or poses.shape != (count, 6):
        raise SelectedPreimageOperandAdapterError("auxiliary advisory poses have invalid shape or dtype")
    if _field(stage, "pose_authority") != POSE_AUTHORITY:
        raise SelectedPreimageOperandAdapterError("auxiliary pose authority drifted")
    return tuple(np.ascontiguousarray(value) for value in (pair_ids, y0, y1, labels, poses))  # type: ignore[return-value]


class TaskspaceSelectedPreimageFreshOperandAdapterV1:
    """Production G49-to-G52 bridge with recurrent adversarial admission."""

    def __init__(
        self,
        *,
        program: TaskspaceSelectedPreimageProgramV1,
        decoder: TaskspaceSelectedPreimageDecoderV1,
        auxiliary_provider: AuxiliaryFreshOperandProviderV1,
        auxiliary_custody: AuxiliaryOperandCustodyV1,
        production_custody: SelectedPreimageProductionCustodyV1 | None = None,
        pairs_per_stage: int = PRODUCTION_PAIRS_PER_STAGE,
        test_only_small_fixture: bool = False,
    ) -> None:
        if type(program) is not TaskspaceSelectedPreimageProgramV1:
            raise SelectedPreimageOperandAdapterError("adapter requires exact TaskspaceSelectedPreimageProgramV1")
        if not isinstance(decoder, TaskspaceSelectedPreimageDecoderV1):
            raise SelectedPreimageOperandAdapterError("adapter decoder does not implement the G49 protocol")
        if not isinstance(auxiliary_provider, AuxiliaryFreshOperandProviderV1):
            raise SelectedPreimageOperandAdapterError("auxiliary provider lacks iter_stages(max_pairs=...)")
        if type(auxiliary_custody) is not AuxiliaryOperandCustodyV1:
            raise SelectedPreimageOperandAdapterError("adapter requires exact AuxiliaryOperandCustodyV1")
        if production_custody is not None and type(production_custody) is not SelectedPreimageProductionCustodyV1:
            raise SelectedPreimageOperandAdapterError("production custody must use the exact G58 type")
        if type(pairs_per_stage) is not int or pairs_per_stage < 1:
            raise SelectedPreimageOperandAdapterError("pairs_per_stage must be positive")
        pair_count = program.compile_config.pair_count
        if program.compile_config.source_pair_start != 0 or pair_count % pairs_per_stage:
            raise SelectedPreimageOperandAdapterError("adapter requires a zero-based exactly divisible pair lattice")
        if not test_only_small_fixture and (
            pair_count,
            pairs_per_stage,
            pair_count // pairs_per_stage,
        ) != (
            PRODUCTION_PAIR_COUNT,
            PRODUCTION_PAIRS_PER_STAGE,
            PRODUCTION_STAGE_COUNT,
        ):
            raise SelectedPreimageOperandAdapterError("production adapter requires n600 in five 120-pair stages")
        if (
            auxiliary_custody.pair_count != pair_count
            or auxiliary_custody.stage_pairs != pairs_per_stage
            or auxiliary_custody.stage_count != pair_count // pairs_per_stage
            or auxiliary_custody.scorer_batch_size != 16
        ):
            raise SelectedPreimageOperandAdapterError(
                "auxiliary aggregate lattice/batch differs from the selected-preimage program"
            )
        if test_only_small_fixture and auxiliary_custody.receipt_schema != FIXTURE_AUXILIARY_AGGREGATE_SCHEMA:
            raise SelectedPreimageOperandAdapterError(
                "small fixtures require the explicit G58 fixture auxiliary schema"
            )
        if not test_only_small_fixture and auxiliary_custody.receipt_schema != FRESH_SCORER_PLANE_AGGREGATE_SCHEMA:
            raise SelectedPreimageOperandAdapterError(
                "production requires the recursively reopened G51 aggregate schema"
            )
        if test_only_small_fixture and production_custody is not None:
            raise SelectedPreimageOperandAdapterError("small fixtures cannot impersonate production reopenable custody")
        if not test_only_small_fixture:
            if production_custody is None:
                raise SelectedPreimageOperandAdapterError(
                    "production requires reopenable semantic/compiler/decoder custody"
                )
            production_custody.verify_against(program, decoder)
        if auxiliary_custody.target_labels_sha256 != program.target_custody_identity.target_bank_sha256:
            raise SelectedPreimageOperandAdapterError("auxiliary labels do not match the G49 batch16 target bank")
        packet = encode_selected_preimage_program(program)
        parsed = parse_selected_preimage_program(
            packet,
            maximum_packet_bytes=len(packet),
        )
        if parsed != program or parsed.packet_bytes != packet:
            raise SelectedPreimageOperandAdapterError("selected-preimage packet parse-back changed the program")
        if (
            decoder.decoder_id != program.decoder_identity.decoder_id
            or decoder.implementation_source_sha256 != program.decoder_identity.implementation_source_sha256
        ):
            raise SelectedPreimageOperandAdapterError("decoder identity differs from the program")
        behavior_probe_pairs: dict[str, int] = {}
        for factor in program.factors:
            addressed_pair = next(
                (source_pair_id for source_pair_id in range(pair_count) if factor.addresses(source_pair_id)),
                None,
            )
            if addressed_pair is None:
                raise SelectedPreimageOperandAdapterError(f"factor {factor.section_id} addresses no pair")
            try:
                # G49 verifies every factor addressed at this pair is non-inert.
                decode_selected_preimage_pair(
                    program,
                    addressed_pair,
                    decoder,
                )
            except TaskspaceSelectedPreimageProgramError as exc:
                raise SelectedPreimageOperandAdapterError(
                    f"counted-factor behavior probe failed for {factor.section_id}"
                ) from exc
            behavior_probe_pairs[factor.section_id] = addressed_pair
        if not behavior_probe_pairs:
            raise SelectedPreimageOperandAdapterError("program has no addressed behavior-changing counted factor")

        self.program = program
        self.decoder = decoder
        self.auxiliary_provider = auxiliary_provider
        self.auxiliary_custody = auxiliary_custody
        self.production_custody = production_custody
        self.pairs_per_stage = pairs_per_stage
        self.stage_count = pair_count // pairs_per_stage
        self.test_only_small_fixture = test_only_small_fixture
        self._packet = packet
        self._behavior_probe_pairs_by_section = behavior_probe_pairs
        self._behavior_changing_factor_section_ids = tuple(factor.section_id for factor in program.factors)
        self._auxiliary_stage_identities = self._prevalidate_auxiliary()
        self._last_complete_stage_chain_sha256: str | None = None
        self._complete_stage_admissions: tuple[SelectedPreimagePreEncodeAdmissionV1, ...] | None = None

    def _prevalidate_auxiliary(self) -> tuple[_AuxiliaryStageIdentityV1, ...]:
        rows: list[_AuxiliaryStageIdentityV1] = []
        labels_digest = hashlib.sha256()
        poses_digest = hashlib.sha256()
        try:
            stages = self.auxiliary_provider.iter_stages(max_pairs=self.pairs_per_stage)
        except TypeError as exc:
            raise SelectedPreimageOperandAdapterError("auxiliary provider API drift") from exc
        for stage_index, stage in enumerate(stages):
            if stage_index >= self.stage_count:
                raise SelectedPreimageOperandAdapterError("auxiliary provider yielded extra stages")
            start = stage_index * self.pairs_per_stage
            stop = start + self.pairs_per_stage
            pair_ids, y0, y1, labels, poses = _validate_auxiliary_stage(
                stage,
                expected_start=start,
                expected_stop=stop,
            )
            labels_digest.update(memoryview(labels).cast("B"))
            poses_digest.update(memoryview(poses).cast("B"))
            observed = _AuxiliaryStageIdentityV1(
                pair_range=(start, stop),
                pair_ids_sha256=_sha256_array(pair_ids),
                direct_source_y0_sha256=_sha256_array(y0),
                direct_source_y1_sha256=_sha256_array(y1),
                target_labels_sha256=_sha256_array(labels),
                gt_poses_f32_sha256=_sha256_array(poses),
            )
            receipt_binding = self.auxiliary_custody.stage_bindings[stage_index]
            if (
                observed.pair_range != receipt_binding.pair_range
                or observed.direct_source_y0_sha256 != receipt_binding.direct_source_y0_sha256
                or observed.direct_source_y1_sha256 != receipt_binding.direct_source_y1_sha256
                or observed.gt_poses_f32_sha256 != receipt_binding.gt_poses_f32_sha256
                or (
                    receipt_binding.target_labels_sha256 is not None
                    and observed.target_labels_sha256 != receipt_binding.target_labels_sha256
                )
            ):
                raise SelectedPreimageOperandAdapterError(
                    "auxiliary provider stage differs from recursively bound aggregate custody"
                )
            rows.append(observed)
        if len(rows) != self.stage_count:
            raise SelectedPreimageOperandAdapterError("auxiliary provider did not cover the exact stage lattice")
        if labels_digest.hexdigest() != self.auxiliary_custody.target_labels_sha256:
            raise SelectedPreimageOperandAdapterError("full-population auxiliary label hash mismatch")
        if (
            self.auxiliary_custody.gt_poses_f32_sha256 is not None
            and poses_digest.hexdigest() != self.auxiliary_custody.gt_poses_f32_sha256
        ):
            raise SelectedPreimageOperandAdapterError("full-population advisory pose hash mismatch")
        return tuple(rows)

    @property
    def pre_encode_identity_receipt(self) -> Mapping[str, Any]:
        semantic = self.program.semantic_program_identity
        target = self.program.target_custody_identity
        homes = self.program.byte_homes()
        factor_payload_bytes = sum(home.byte_length for home in homes[1:])
        return {
            "schema": SCHEMA,
            "status": "PRE_ENCODE_SELECTED_PREIMAGE_IDENTITY_CLOSED",
            "research_only": True,
            "score_claim": False,
            "pair_count": self.program.compile_config.pair_count,
            "pairs_per_stage": self.pairs_per_stage,
            "stage_count": self.stage_count,
            "test_only_small_fixture": self.test_only_small_fixture,
            "representation_source": ("G49_TASKSPACE_SELECTED_PREIMAGE_PROGRAM_DECODE_ONLY"),
            "provider_kind": "G49_SELECTED_PREIMAGE_PROGRAM",
            "source_plane_definition": "G49_DECODE_SELECTED_PREIMAGE_PAIR",
            "program_packet": {
                "bytes": len(self._packet),
                "sha256": _sha256_bytes(self._packet),
            },
            "semantic_program": {
                "bytes": semantic.compiled_semantic_archive_bytes,
                "sha256": semantic.compiled_semantic_archive_sha256,
                "fresh_compile_receipt_sha256": (semantic.fresh_compile_receipt_sha256),
                "compile_proof_dependency_sha256": (semantic.compile_proof_dependency_sha256),
                "compiler_source_sha256": semantic.compiler_source_sha256,
                "embedded_here": False,
            },
            "target_custody": {
                "receipt_sha256": target.target_custody_receipt_sha256,
                "bank_sha256": target.target_bank_sha256,
                "payload_embedded": False,
            },
            "decoder": {
                "id": self.decoder.decoder_id,
                "implementation_source_sha256": (self.decoder.implementation_source_sha256),
            },
            "factors": {
                "section_ids": [factor.section_id for factor in self.program.factors],
                "behavior_probe_pairs_by_section": (self._behavior_probe_pairs_by_section),
                "behavior_changing_section_ids": list(self._behavior_changing_factor_section_ids),
                "payload_bytes_inside_packet": factor_payload_bytes,
                "byte_homes_partition_packet": (sum(home.byte_length for home in homes) == len(self._packet)),
            },
            "auxiliary_custody": asdict(self.auxiliary_custody),
            "production_custody": (None if self.production_custody is None else asdict(self.production_custody)),
            "auxiliary_planes_forwarded": False,
            "g49_decode_custody_verified": True,
            "precontainer_counted_source_bytes": (len(self._packet) + semantic.compiled_semantic_archive_bytes),
            "representation_status": BLOCKED_REPRESENTATION_STATUS,
            "next_preclosure_gate": NEXT_PRECLOSURE_GATE,
        }

    @property
    def pre_encode_identity_sha256(self) -> str:
        return _sha256_bytes(_canonical_json(self.pre_encode_identity_receipt))

    def publish_pre_encode_identity_receipt(
        self,
        path: str | os.PathLike[str],
    ) -> Mapping[str, Any]:
        """Atomically publish the exact provider receipt consumed by G52."""

        payload = _canonical_json(self.pre_encode_identity_receipt)
        return _publish_write_once(
            path,
            payload,
            label="pre-encode identity receipt",
        )

    @property
    def last_complete_stage_chain_sha256(self) -> str | None:
        return self._last_complete_stage_chain_sha256

    def publish_terminal_stage_chain_receipt(
        self,
        path: str | os.PathLike[str],
        *,
        campaign_receipt: ReopenableRegularFileIdentityV1,
        pre_encode_identity_receipt: ReopenableRegularFileIdentityV1,
    ) -> Mapping[str, Any]:
        """Publish the sealed full-lattice chain only after all stages completed."""

        if (
            self._last_complete_stage_chain_sha256 is None
            or self._complete_stage_admissions is None
            or len(self._complete_stage_admissions) != self.stage_count
        ):
            raise SelectedPreimageOperandAdapterError(
                "terminal stage-chain receipt is unavailable before full iteration"
            )
        ReopenableRegularFileIdentityV1(**asdict(campaign_receipt))
        ReopenableRegularFileIdentityV1(**asdict(pre_encode_identity_receipt))
        if pre_encode_identity_receipt.sha256 != self.pre_encode_identity_sha256 or _stable_regular_file_bytes(
            pre_encode_identity_receipt.path,
            label="published pre-encode identity receipt",
        ) != _canonical_json(self.pre_encode_identity_receipt):
            raise SelectedPreimageOperandAdapterError("published pre-encode identity differs before terminal sealing")
        aggregate_identity = _regular_file_identity(
            self.auxiliary_custody.aggregate_receipt_path,
            label="auxiliary aggregate receipt",
        )
        if aggregate_identity["sha256"] != self.auxiliary_custody.aggregate_receipt_sha256:
            raise SelectedPreimageOperandAdapterError("auxiliary aggregate drifted before terminal sealing")
        body = {
            "schema": TERMINAL_STAGE_CHAIN_SCHEMA,
            "status": "FULL_STAGE_LATTICE_DECODED_AND_CHAIN_CLOSED",
            "campaign_receipt": asdict(campaign_receipt),
            "pre_encode_identity_receipt": asdict(pre_encode_identity_receipt),
            "auxiliary_aggregate_receipt": {
                **aggregate_identity,
                "self_seal_sha256": (self.auxiliary_custody.aggregate_receipt_self_seal_sha256),
            },
            "pair_count": self.program.compile_config.pair_count,
            "pairs_per_stage": self.pairs_per_stage,
            "stage_count": self.stage_count,
            "scorer_batch_size": self.auxiliary_custody.scorer_batch_size,
            "program_packet_sha256": self.program.packet_sha256,
            "stages": [admission.to_mapping() for admission in self._complete_stage_admissions],
            "terminal_stage_chain_sha256": (self._last_complete_stage_chain_sha256),
            "representation_status": BLOCKED_REPRESENTATION_STATUS,
        }
        receipt = {
            **body,
            "receipt_sha256": _sha256_bytes(_canonical_json(body)),
        }
        payload = _canonical_json(receipt)
        published = dict(
            _publish_write_once(
                path,
                payload,
                label="terminal stage-chain receipt",
            )
        )
        published["receipt_sha256"] = receipt["receipt_sha256"]
        return published

    def program_residual_pre_encode_gate_evidence(
        self,
        outer_proof: ProgramResidualOuterArchiveProofV1,
    ) -> Mapping[str, Any]:
        """Compose the exact evidence consumed by the recurrent PRE_ENCODE gate.

        The semantic counted/reopened booleans exist only on this join with a
        physical outer proof; the provider identity receipt alone keeps them
        false by construction.
        """

        if type(outer_proof) is not ProgramResidualOuterArchiveProofV1:
            raise SelectedPreimageOperandAdapterError("PRE_ENCODE evidence requires exact physical outer proof")
        semantic = self.program.semantic_program_identity
        if (
            outer_proof.representation_mode != PROGRAM_RESIDUAL_MODE
            or outer_proof.semantic_archive_bytes != semantic.compiled_semantic_archive_bytes
            or outer_proof.semantic_archive_sha256 != semantic.compiled_semantic_archive_sha256
            or outer_proof.program_packet_bytes != len(self._packet)
            or outer_proof.program_packet_sha256 != self.program.packet_sha256
            or outer_proof.packet_byte_homes_partition_exactly is not True
            or outer_proof.generic_learned_decoder_source_bound is not True
            or outer_proof.target_labels_embedded is not False
            or outer_proof.advisory_poses_embedded is not False
        ):
            raise SelectedPreimageOperandAdapterError("physical outer proof differs from adapter identities")
        return {
            "actual_representation": PROGRAM_RESIDUAL_MODE,
            "pair_count": self.program.compile_config.pair_count,
            "scorer_batch_size": (self.program.target_custody_identity.scorer_batch_size),
            "provider_kind": "G49_SELECTED_PREIMAGE_PROGRAM",
            "source_plane_definition": "G49_DECODE_SELECTED_PREIMAGE_PAIR",
            "semantic_archive_bytes": outer_proof.semantic_archive_bytes,
            "semantic_archive_sha256": outer_proof.semantic_archive_sha256,
            "semantic_archive_counted": True,
            "semantic_archive_reopened": True,
            "program_packet_bytes": outer_proof.program_packet_bytes,
            "program_packet_sha256": outer_proof.program_packet_sha256,
            "factor_count": len(self.program.factors),
            "behavior_changing_factor_count": len(self._behavior_changing_factor_section_ids),
            "target_payload_embedded": False,
            "historical_payload_reused": False,
        }

    def iter_stages(
        self,
        *,
        max_pairs: int = PRODUCTION_PAIRS_PER_STAGE,
    ) -> Iterator[SelectedPreimageFreshOperandStageV1]:
        if type(max_pairs) is not int or max_pairs < self.pairs_per_stage:
            raise SelectedPreimageOperandAdapterError("max_pairs is smaller than the immutable stage size")
        prior_chain = "0" * 64
        observed_behavior_sections: set[str] = set()
        observed_stage_count = 0
        complete_admissions: list[SelectedPreimagePreEncodeAdmissionV1] = []
        stages = self.auxiliary_provider.iter_stages(max_pairs=self.pairs_per_stage)
        for stage_index, auxiliary_stage in enumerate(stages):
            if stage_index >= self.stage_count:
                raise SelectedPreimageOperandAdapterError("auxiliary provider yielded extra stages on recurrent reopen")
            start = stage_index * self.pairs_per_stage
            stop = start + self.pairs_per_stage
            pair_ids, direct_y0, direct_y1, labels, poses = _validate_auxiliary_stage(
                auxiliary_stage,
                expected_start=start,
                expected_stop=stop,
            )
            expected_auxiliary = self._auxiliary_stage_identities[stage_index]
            observed_auxiliary = _AuxiliaryStageIdentityV1(
                pair_range=(start, stop),
                pair_ids_sha256=_sha256_array(pair_ids),
                direct_source_y0_sha256=_sha256_array(direct_y0),
                direct_source_y1_sha256=_sha256_array(direct_y1),
                target_labels_sha256=_sha256_array(labels),
                gt_poses_f32_sha256=_sha256_array(poses),
            )
            if observed_auxiliary != expected_auxiliary:
                raise SelectedPreimageOperandAdapterError("auxiliary stage changed after pre-encoding admission")

            y0_buffer = np.empty(
                (self.pairs_per_stage, *SCORER_SHAPE),
                dtype=np.uint8,
            )
            y1_buffer = np.empty_like(y0_buffer)
            decoded_count = 0
            decoded_pairs = iter_selected_preimage_segment(
                self.program,
                self.decoder,
                segment_index=stage_index,
                pairs_per_segment=self.pairs_per_stage,
            )
            for local_offset, pair in enumerate(decoded_pairs):
                if local_offset >= self.pairs_per_stage:
                    raise SelectedPreimageOperandAdapterError("G49 segment returned extra pairs")
                pair_id = start + local_offset
                if (
                    pair.pair_index != pair_id
                    or pair.source_pair_id != pair_id
                    or pair.segment_index != stage_index
                    or pair.segment_count != self.stage_count
                    or pair.program_packet_sha256 != self.program.packet_sha256
                    or pair.target_custody_receipt_sha256
                    != self.program.target_custody_identity.target_custody_receipt_sha256
                    or pair.target_bank_sha256 != self.program.target_custody_identity.target_bank_sha256
                ):
                    raise SelectedPreimageOperandAdapterError("G49 decoded pair identity drifted")
                y0_buffer[local_offset] = pair.scorer_y0
                y1_buffer[local_offset] = pair.scorer_y1
                decoded_count += 1
                for factor in self.program.factors:
                    if factor.addresses(pair_id):
                        observed_behavior_sections.add(factor.section_id)
            if decoded_count != self.pairs_per_stage:
                raise SelectedPreimageOperandAdapterError("G49 segment did not return the exact immutable stage")

            y0 = _readonly_contiguous(y0_buffer)
            y1 = _readonly_contiguous(y1_buffer)
            if any(
                np.shares_memory(decoded, auxiliary) for decoded in (y0, y1) for auxiliary in (direct_y0, direct_y1)
            ):
                raise SelectedPreimageOperandAdapterError("G49 decoded planes alias auxiliary direct-source storage")
            factor_ids = tuple(factor.section_id for factor in self.program.factors)
            behavior_ids = tuple(
                factor.section_id
                for factor in self.program.factors
                if factor.section_id in observed_behavior_sections
                and any(factor.addresses(pair_id) for pair_id in range(start, stop))
            )
            homes = self.program.byte_homes()
            semantic = self.program.semantic_program_identity
            target = self.program.target_custody_identity
            admission_values: dict[str, Any] = {
                "schema": ADMISSION_SCHEMA,
                "status": "PRE_ENCODE_SELECTED_PREIMAGE_STAGE_ADMITTED",
                "stage_index": stage_index,
                "stage_count": self.stage_count,
                "pair_range": (start, stop),
                "representation_source": ("G49_TASKSPACE_SELECTED_PREIMAGE_PROGRAM_DECODE_ONLY"),
                "program_packet_sha256": self.program.packet_sha256,
                "program_packet_bytes": len(self._packet),
                "semantic_archive_sha256": semantic.compiled_semantic_archive_sha256,
                "semantic_archive_bytes": semantic.compiled_semantic_archive_bytes,
                "fresh_semantic_compile_receipt_sha256": (semantic.fresh_compile_receipt_sha256),
                "target_custody_receipt_sha256": (target.target_custody_receipt_sha256),
                "target_bank_sha256": target.target_bank_sha256,
                "decoder_id": self.decoder.decoder_id,
                "decoder_implementation_source_sha256": (self.decoder.implementation_source_sha256),
                "factor_section_ids": factor_ids,
                "behavior_changing_factor_section_ids": behavior_ids,
                "factor_payload_bytes_inside_packet": sum(home.byte_length for home in homes[1:]),
                "scorer_y0_sha256": _sha256_array(y0),
                "scorer_y1_sha256": _sha256_array(y1),
                "discarded_direct_source_y0_sha256": (expected_auxiliary.direct_source_y0_sha256),
                "discarded_direct_source_y1_sha256": (expected_auxiliary.direct_source_y1_sha256),
                "target_labels_sha256": expected_auxiliary.target_labels_sha256,
                "gt_poses_f32_sha256": expected_auxiliary.gt_poses_f32_sha256,
                "auxiliary_planes_forwarded": False,
                "g49_decode_custody_verified": True,
                "prior_stage_chain_sha256": prior_chain,
                "representation_status": BLOCKED_REPRESENTATION_STATUS,
                "next_preclosure_gate": NEXT_PRECLOSURE_GATE,
            }
            stage_chain = _sha256_bytes(bytes.fromhex(prior_chain) + _canonical_json(admission_values))
            admission = SelectedPreimagePreEncodeAdmissionV1(
                **admission_values,
                stage_chain_sha256=stage_chain,
            )
            complete_admissions.append(admission)
            yield SelectedPreimageFreshOperandStageV1(
                pair_range=(start, stop),
                pair_ids=_readonly_contiguous(pair_ids),
                y0_u8=y0,
                y1_u8=y1,
                target_labels_u8=_readonly_contiguous(labels),
                gt_poses_f32=_readonly_contiguous(poses),
                pre_encode_admission=admission,
            )
            prior_chain = stage_chain
            observed_stage_count += 1
        if observed_stage_count != self.stage_count:
            raise SelectedPreimageOperandAdapterError("auxiliary provider did not recurrently cover all stages")
        if observed_behavior_sections != set(self._behavior_changing_factor_section_ids):
            raise SelectedPreimageOperandAdapterError("not every counted behavior-changing factor was observed")
        self._last_complete_stage_chain_sha256 = prior_chain
        self._complete_stage_admissions = tuple(complete_admissions)


def validate_pre_encode_stage(
    stage: SelectedPreimageFreshOperandStageV1,
    *,
    expected_prior_stage_chain_sha256: str,
) -> str:
    """Reopen one stage identity before handing it to any encoder."""

    if type(stage) is not SelectedPreimageFreshOperandStageV1:
        raise SelectedPreimageOperandAdapterError("pre-encode validation requires exact adapter stage")
    admission = stage.pre_encode_admission
    if (
        type(admission.pair_range) is not tuple
        or len(admission.pair_range) != 2
        or any(type(value) is not int for value in admission.pair_range)
    ):
        raise SelectedPreimageOperandAdapterError("pre-encode stage pair range is malformed")
    pair_start, pair_stop = admission.pair_range
    pair_count = pair_stop - pair_start
    expected_pair_ids = np.arange(pair_start, pair_stop, dtype=np.int64)
    expected_plane_shape = (pair_count, *SCORER_SHAPE)
    if (
        admission.schema != ADMISSION_SCHEMA
        or admission.status != "PRE_ENCODE_SELECTED_PREIMAGE_STAGE_ADMITTED"
        or admission.representation_source != "G49_TASKSPACE_SELECTED_PREIMAGE_PROGRAM_DECODE_ONLY"
        or admission.stage_count < 1
        or admission.stage_index < 0
        or admission.stage_index >= admission.stage_count
        or stage.pair_range != admission.pair_range
        or pair_start < 0
        or pair_stop <= pair_start
        or stage.pair_ids.shape != expected_pair_ids.shape
        or not np.issubdtype(stage.pair_ids.dtype, np.integer)
        or not np.array_equal(
            stage.pair_ids.astype(np.int64, copy=False),
            expected_pair_ids,
        )
        or stage.y0_u8.dtype != np.uint8
        or stage.y1_u8.dtype != np.uint8
        or stage.y0_u8.shape != expected_plane_shape
        or stage.y1_u8.shape != expected_plane_shape
        or stage.target_labels_u8.dtype != np.uint8
        or stage.target_labels_u8.shape != expected_plane_shape[:3]
        or stage.gt_poses_f32.dtype != np.float32
        or stage.gt_poses_f32.shape != (pair_count, 6)
        or stage.pose_authority != POSE_AUTHORITY
        or admission.prior_stage_chain_sha256
        != _require_sha256(
            expected_prior_stage_chain_sha256,
            "expected_prior_stage_chain_sha256",
        )
        or admission.scorer_y0_sha256 != _sha256_array(stage.y0_u8)
        or admission.scorer_y1_sha256 != _sha256_array(stage.y1_u8)
        or admission.target_labels_sha256 != _sha256_array(stage.target_labels_u8)
        or admission.gt_poses_f32_sha256 != _sha256_array(stage.gt_poses_f32)
        or admission.auxiliary_planes_forwarded is not False
        or admission.g49_decode_custody_verified is not True
        or not admission.factor_section_ids
        or not set(admission.behavior_changing_factor_section_ids).issubset(admission.factor_section_ids)
        or admission.factor_payload_bytes_inside_packet < 1
        or admission.factor_payload_bytes_inside_packet >= admission.program_packet_bytes
        or admission.semantic_archive_bytes < 1
        or admission.representation_status != BLOCKED_REPRESENTATION_STATUS
        or admission.next_preclosure_gate != NEXT_PRECLOSURE_GATE
    ):
        raise SelectedPreimageOperandAdapterError("pre-encode selected-preimage stage admission failed")
    for field_name in (
        "program_packet_sha256",
        "semantic_archive_sha256",
        "fresh_semantic_compile_receipt_sha256",
        "target_custody_receipt_sha256",
        "target_bank_sha256",
        "decoder_implementation_source_sha256",
        "scorer_y0_sha256",
        "scorer_y1_sha256",
        "discarded_direct_source_y0_sha256",
        "discarded_direct_source_y1_sha256",
        "target_labels_sha256",
        "gt_poses_f32_sha256",
    ):
        _require_sha256(getattr(admission, field_name), field_name)
    chain_body = asdict(admission)
    chain_body.pop("stage_chain_sha256")
    expected_chain = _sha256_bytes(bytes.fromhex(admission.prior_stage_chain_sha256) + _canonical_json(chain_body))
    if admission.stage_chain_sha256 != expected_chain:
        raise SelectedPreimageOperandAdapterError("pre-encode stage identity chain mismatch")
    return expected_chain


def _safe_member_name(value: str, label: str) -> str:
    if type(value) is not str or not value:
        raise SelectedPreimageOperandAdapterError(f"{label} must be nonempty")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.endswith("/"):
        raise SelectedPreimageOperandAdapterError(f"{label} is unsafe")
    return value


def _member_name_has_forbidden_payload_role(value: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", value.lower()))
    return any(token in tokens for token in _FORBIDDEN_ARCHIVE_MEMBER_TOKENS)


def reopen_program_residual_outer_archive(
    *,
    archive_path: str | os.PathLike[str],
    semantic_archive_bytes: bytes,
    semantic_member_name: str,
    program_member_name: str,
    program: TaskspaceSelectedPreimageProgramV1,
    decoder: TaskspaceSelectedPreimageDecoderV1,
    forbidden_payload_sha256s: tuple[str, ...] = (),
) -> ProgramResidualOuterArchiveProofV1:
    """Reopen physical counted members before allowing the representation name."""

    if type(program) is not TaskspaceSelectedPreimageProgramV1:
        raise SelectedPreimageOperandAdapterError("outer proof requires exact selected-preimage program")
    if not isinstance(decoder, TaskspaceSelectedPreimageDecoderV1):
        raise SelectedPreimageOperandAdapterError("outer proof decoder does not implement G49")
    if type(semantic_archive_bytes) is not bytes or not semantic_archive_bytes:
        raise SelectedPreimageOperandAdapterError("outer proof requires exact semantic archive bytes")
    semantic_name = _safe_member_name(
        semantic_member_name,
        "semantic_member_name",
    )
    program_name = _safe_member_name(program_member_name, "program_member_name")
    if semantic_name == program_name:
        raise SelectedPreimageOperandAdapterError("semantic and program members must be distinct")
    forbidden_hashes = {
        _require_sha256(value, f"forbidden_payload_sha256s[{index}]")
        for index, value in enumerate(forbidden_payload_sha256s)
    }
    archive_identity, archive_payload = _stable_regular_file_identity_and_bytes(
        archive_path,
        label="counted outer archive",
    )
    path = Path(archive_identity["path"])
    try:
        with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise SelectedPreimageOperandAdapterError("outer archive repeats member names")
            for info in infos:
                name = _safe_member_name(info.filename, "outer archive member")
                if _member_name_has_forbidden_payload_role(name):
                    raise SelectedPreimageOperandAdapterError(
                        "outer archive names a forbidden target/pose/scorer payload"
                    )
                mode = info.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if info.is_dir() or info.flag_bits & 0x1 or file_type not in (0, stat.S_IFREG):
                    raise SelectedPreimageOperandAdapterError(
                        "outer archive contains a non-regular or encrypted member"
                    )
                payload = archive.read(info)
                if _sha256_bytes(payload) in forbidden_hashes:
                    raise SelectedPreimageOperandAdapterError("outer archive embeds a forbidden target/pose payload")
            if len(names) != 2 or set(names) != {semantic_name, program_name}:
                raise SelectedPreimageOperandAdapterError("outer archive contains an unexpected untyped member")
            semantic_info = archive.getinfo(semantic_name)
            program_info = archive.getinfo(program_name)
            embedded_semantic = archive.read(semantic_info)
            embedded_packet = archive.read(program_info)
    except (KeyError, zipfile.BadZipFile) as exc:
        raise SelectedPreimageOperandAdapterError("outer archive cannot reopen both counted members") from exc

    semantic_identity = program.semantic_program_identity
    if (
        embedded_semantic != semantic_archive_bytes
        or len(embedded_semantic) != semantic_identity.compiled_semantic_archive_bytes
        or _sha256_bytes(embedded_semantic) != semantic_identity.compiled_semantic_archive_sha256
    ):
        raise SelectedPreimageOperandAdapterError("outer semantic member differs from the fresh compile identity")
    packet = encode_selected_preimage_program(program)
    if embedded_packet != packet or _sha256_bytes(embedded_packet) != program.packet_sha256:
        raise SelectedPreimageOperandAdapterError("outer program member differs from the counted G49 packet")
    parsed = parse_selected_preimage_program(
        embedded_packet,
        maximum_packet_bytes=len(embedded_packet),
    )
    if parsed != program or parsed.packet_bytes != embedded_packet:
        raise SelectedPreimageOperandAdapterError("outer program member failed exact parse-back")
    homes = parsed.byte_homes()
    if (
        not homes
        or homes[0].offset != 0
        or any(previous.offset + previous.byte_length != current.offset for previous, current in pairwise(homes))
        or homes[-1].offset + homes[-1].byte_length != len(embedded_packet)
    ):
        raise SelectedPreimageOperandAdapterError("program byte homes do not partition the counted packet")

    learned_contracts: list[tuple[str, str, str]] = []
    learned_factors = [
        factor for factor in program.factors if factor.role is SelectedPreimageFactorRoleV1.LEARNED_IRREDUCIBLE_QUOTIENT
    ]
    for factor in learned_factors:
        addressed_pair = next(
            (
                local_pair_id
                for local_pair_id in range(program.compile_config.pair_count)
                if factor.addresses(program.compile_config.source_pair_start + local_pair_id)
            ),
            None,
        )
        if addressed_pair is None:
            raise SelectedPreimageOperandAdapterError("learned factor addresses no pair")
        try:
            decode_selected_preimage_pair(program, addressed_pair, decoder)
            contract_id = decoder.learned_quotient_decoder_contract_id(factor)
            source_sha256 = decoder.learned_quotient_decoder_implementation_source_sha256(factor)
        except TaskspaceSelectedPreimageProgramError as exc:
            raise SelectedPreimageOperandAdapterError("learned factor decoder contract/source did not reopen") from exc
        _require_sha256(source_sha256, "learned decoder source")
        learned_contracts.append((factor.section_id, contract_id, source_sha256))

    return ProgramResidualOuterArchiveProofV1(
        schema=OUTER_PROOF_SCHEMA,
        status="COUNTED_OUTER_MEMBERS_REOPENED_PUBLIC_CLOSURE_OWED",
        representation_mode=PROGRAM_RESIDUAL_MODE,
        archive_path=str(path),
        archive_bytes=archive_identity["bytes"],
        archive_sha256=archive_identity["sha256"],
        outer_member_names=(semantic_name, program_name),
        outer_members_partition_exactly=True,
        semantic_member_name=semantic_name,
        semantic_member_compressed_bytes=semantic_info.compress_size,
        semantic_archive_bytes=len(embedded_semantic),
        semantic_archive_sha256=_sha256_bytes(embedded_semantic),
        program_member_name=program_name,
        program_member_compressed_bytes=program_info.compress_size,
        program_packet_bytes=len(embedded_packet),
        program_packet_sha256=_sha256_bytes(embedded_packet),
        factor_payload_bytes_inside_packet=sum(home.byte_length for home in homes[1:]),
        packet_byte_homes_partition_exactly=True,
        learned_decoder_contracts=tuple(learned_contracts),
        generic_learned_decoder_source_bound=True,
        target_labels_embedded=False,
        advisory_poses_embedded=False,
        precontainer_counted_source_bytes=(len(embedded_semantic) + len(embedded_packet)),
        next_preclosure_gate=NEXT_PRECLOSURE_GATE,
    )


def publish_program_residual_outer_archive_proof(
    proof: ProgramResidualOuterArchiveProofV1,
    path: str | os.PathLike[str],
) -> Mapping[str, Any]:
    """Publish one self-sealed physical outer proof without overwriting."""

    if type(proof) is not ProgramResidualOuterArchiveProofV1:
        raise SelectedPreimageOperandAdapterError("outer proof publication requires exact G58 proof")
    body = proof.to_mapping()
    receipt = {
        **body,
        "receipt_sha256": _sha256_bytes(_canonical_json(body)),
    }
    return _publish_write_once(
        path,
        _canonical_json(receipt),
        label="program-residual outer proof",
    )


def _load_canonical_receipt(
    path: str | os.PathLike[str],
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record, payload = _stable_regular_file_identity_and_bytes(
        path,
        label=label,
    )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectedPreimageOperandAdapterError(f"{label} is not JSON") from exc
    if type(value) is not dict or _canonical_json(value) != payload:
        raise SelectedPreimageOperandAdapterError(f"{label} is not a canonical object")
    return record, value


def _reopen_file_row(
    value: object,
    *,
    label: str,
) -> ReopenableRegularFileIdentityV1:
    if type(value) is not dict or set(value) != {"path", "bytes", "sha256"}:
        raise SelectedPreimageOperandAdapterError(f"{label} is not an exact file identity")
    try:
        return ReopenableRegularFileIdentityV1(**value)
    except TypeError as exc:
        raise SelectedPreimageOperandAdapterError(f"{label} file identity fields differ") from exc


def _reopen_file_row_and_payload(
    value: object,
    *,
    label: str,
) -> tuple[ReopenableRegularFileIdentityV1, bytes]:
    if type(value) is not dict or set(value) != {"path", "bytes", "sha256"}:
        raise SelectedPreimageOperandAdapterError(f"{label} is not an exact file identity")
    record, payload = _stable_regular_file_identity_and_bytes(
        value["path"],
        label=label,
    )
    if record != value:
        raise SelectedPreimageOperandAdapterError(f"{label} exact file identity differs")
    return ReopenableRegularFileIdentityV1(**record), payload


def _admission_from_mapping(
    value: object,
) -> SelectedPreimagePreEncodeAdmissionV1:
    if type(value) is not dict or set(value) != {
        field_info.name for field_info in SelectedPreimagePreEncodeAdmissionV1.__dataclass_fields__.values()
    }:
        raise SelectedPreimageOperandAdapterError("terminal stage admission fields differ")
    converted = dict(value)
    for field_name in (
        "pair_range",
        "factor_section_ids",
        "behavior_changing_factor_section_ids",
    ):
        raw = converted[field_name]
        if type(raw) is not list:
            raise SelectedPreimageOperandAdapterError(f"terminal admission {field_name} is not canonical JSON")
        converted[field_name] = tuple(raw)
    try:
        return SelectedPreimagePreEncodeAdmissionV1(**converted)
    except TypeError as exc:
        raise SelectedPreimageOperandAdapterError("terminal stage admission cannot reopen") from exc


def reopen_program_residual_production_pre_encode_evidence(
    *,
    identity_path: str | os.PathLike[str],
    terminal_stage_chain_path: str | os.PathLike[str],
    outer_proof_path: str | os.PathLike[str],
) -> Mapping[str, Any]:
    """Strict G58 production verifier consumed by G59 PRE_ENCODE.

    This is the only production interpretation of the three G58 lifecycle
    artifacts. It admits no aliases and does not treat an older gate receipt as
    authority.
    """

    identity_record, identity = _load_canonical_receipt(
        identity_path,
        label="G58 pre-encode identity",
    )
    if (
        identity.get("schema") != SCHEMA
        or identity.get("status") != "PRE_ENCODE_SELECTED_PREIMAGE_IDENTITY_CLOSED"
        or identity.get("pair_count") != PRODUCTION_PAIR_COUNT
        or identity.get("pairs_per_stage") != PRODUCTION_PAIRS_PER_STAGE
        or identity.get("stage_count") != PRODUCTION_STAGE_COUNT
        or identity.get("test_only_small_fixture") is not False
        or identity.get("representation_source") != "G49_TASKSPACE_SELECTED_PREIMAGE_PROGRAM_DECODE_ONLY"
        or identity.get("provider_kind") != "G49_SELECTED_PREIMAGE_PROGRAM"
        or identity.get("source_plane_definition") != "G49_DECODE_SELECTED_PREIMAGE_PAIR"
        or identity.get("auxiliary_planes_forwarded") is not False
        or identity.get("g49_decode_custody_verified") is not True
        or identity.get("representation_status") != BLOCKED_REPRESENTATION_STATUS
    ):
        raise SelectedPreimageOperandAdapterError("G58 production identity fields differ")
    semantic = identity.get("semantic_program")
    target = identity.get("target_custody")
    decoder = identity.get("decoder")
    factors = identity.get("factors")
    production = identity.get("production_custody")
    auxiliary = identity.get("auxiliary_custody")
    if not all(
        type(value) is dict
        for value in (
            semantic,
            target,
            decoder,
            factors,
            production,
            auxiliary,
        )
    ):
        raise SelectedPreimageOperandAdapterError("G58 production identity nested custody is absent")
    expected_production_fields = {
        "semantic_archive",
        "semantic_compile_receipt",
        "target_custody_receipt",
        "target_custody_receipt_seal_sha256",
        "compiler_source",
        "generic_v10_source",
        "decoder_callable_source_sha256",
        "learned_decoder_sources",
    }
    if set(production) != expected_production_fields:
        raise SelectedPreimageOperandAdapterError("G58 production custody fields differ")
    semantic_archive, semantic_archive_payload = _reopen_file_row_and_payload(
        production["semantic_archive"],
        label="semantic archive",
    )
    semantic_compile_receipt, _semantic_compile_payload = _reopen_file_row_and_payload(
        production["semantic_compile_receipt"],
        label="semantic compile receipt",
    )
    target_custody_receipt, target_custody_payload = _reopen_file_row_and_payload(
        production["target_custody_receipt"],
        label="target custody receipt",
    )
    target_custody_receipt_seal_sha256 = _require_sha256(
        production["target_custody_receipt_seal_sha256"],
        "target_custody_receipt_seal_sha256",
    )
    compiler_source, _compiler_source_payload = _reopen_file_row_and_payload(
        production["compiler_source"],
        label="semantic compiler source",
    )
    generic_v10_source, _generic_v10_payload = _reopen_file_row_and_payload(
        production["generic_v10_source"],
        label="generic V10 source",
    )
    decoder_callable_sha256 = _require_sha256(
        production["decoder_callable_source_sha256"],
        "decoder_callable_source_sha256",
    )
    raw_learned_sources = production["learned_decoder_sources"]
    if type(raw_learned_sources) is not list:
        raise SelectedPreimageOperandAdapterError("learned decoder sources must be a canonical list")
    learned_source_rows: dict[
        str,
        tuple[str, ReopenableRegularFileIdentityV1],
    ] = {}
    for index, raw in enumerate(raw_learned_sources):
        if type(raw) is not dict or set(raw) != {
            "section_id",
            "decoder_contract_id",
            "implementation_source",
        }:
            raise SelectedPreimageOperandAdapterError(f"learned decoder source {index} fields differ")
        section_id = raw["section_id"]
        contract_id = raw["decoder_contract_id"]
        if (
            type(section_id) is not str
            or not section_id
            or section_id in learned_source_rows
            or type(contract_id) is not str
            or not contract_id
        ):
            raise SelectedPreimageOperandAdapterError(f"learned decoder source {index} IDs differ")
        source, _source_payload = _reopen_file_row_and_payload(
            raw["implementation_source"],
            label=f"learned decoder source {section_id}",
        )
        learned_source_rows[section_id] = (contract_id, source)
    if (
        semantic.get("bytes") != semantic_archive.bytes
        or semantic.get("sha256") != semantic_archive.sha256
        or semantic.get("fresh_compile_receipt_sha256") != semantic_compile_receipt.sha256
        or semantic.get("embedded_here") is not False
        or target.get("receipt_sha256") != target_custody_receipt_seal_sha256
        or target.get("payload_embedded") is not False
        or decoder.get("implementation_source_sha256") != decoder_callable_sha256
        or generic_v10_source.sha256 != decoder_callable_sha256
        or compiler_source.sha256 != semantic.get("compiler_source_sha256")
    ):
        raise SelectedPreimageOperandAdapterError("G58 reopenable production custody differs from compact identity")
    try:
        target_receipt = json.loads(target_custody_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectedPreimageOperandAdapterError("target custody receipt is not JSON") from exc
    target_labels = target_receipt.get("target_labels") if type(target_receipt) is dict else None
    if type(target_receipt) is not dict:
        raise SelectedPreimageOperandAdapterError("target custody receipt must be an object")
    try:
        verify_sealed_payload(
            target_receipt,
            hash_field="receipt_sha256",
        )
    except FreshTeacherMaterializationError as exc:
        raise SelectedPreimageOperandAdapterError("target custody receipt self-seal differs") from exc
    if (
        type(target_receipt) is not dict
        or target_receipt.get("schema") != "tac.taskspace_fresh_teacher_materialization.v1"
        or target_receipt.get("receipt_sha256") != target_custody_receipt_seal_sha256
        or target_receipt.get("pair_count") != PRODUCTION_PAIR_COUNT
        or target_receipt.get("scorer_pair_batch_size") != 16
        or target_receipt.get("batch_geometry_matches_upstream_default") is not True
        or target_receipt.get("encoder_only") is not True
        or target_receipt.get("candidate_payload_allowed") is not False
        or type(target_labels) is not dict
        or target_labels.get("sha256") != target.get("bank_sha256")
        or target_labels.get("shape") != [PRODUCTION_PAIR_COUNT, *SCORER_SHAPE[:2]]
        or target_labels.get("dtype") != "uint8"
    ):
        raise SelectedPreimageOperandAdapterError("target custody receipt is not sealed batch16 n600 custody")
    # The compact semantic row historically omitted compiler_source_sha256;
    # the compile receipt verifier established it before construction, while
    # production_custody now makes the source itself reopenable.

    aggregate = AuxiliaryOperandCustodyV1(
        aggregate_receipt_path=auxiliary.get("aggregate_receipt_path"),
        aggregate_receipt_sha256=auxiliary.get("aggregate_receipt_sha256"),
    )
    if (
        aggregate.receipt_schema != FRESH_SCORER_PLANE_AGGREGATE_SCHEMA
        or aggregate.pair_count != PRODUCTION_PAIR_COUNT
        or aggregate.stage_pairs != PRODUCTION_PAIRS_PER_STAGE
        or aggregate.stage_count != PRODUCTION_STAGE_COUNT
        or aggregate.scorer_batch_size != 16
        or aggregate.target_labels_sha256 != target.get("bank_sha256")
    ):
        raise SelectedPreimageOperandAdapterError("G51 auxiliary aggregate differs from target custody")

    terminal_record, terminal = _load_canonical_receipt(
        terminal_stage_chain_path,
        label="G58 terminal stage chain",
    )
    terminal_body = dict(terminal)
    terminal_seal = terminal_body.pop("receipt_sha256", None)
    if (
        terminal.get("schema") != TERMINAL_STAGE_CHAIN_SCHEMA
        or terminal.get("status") != "FULL_STAGE_LATTICE_DECODED_AND_CHAIN_CLOSED"
        or terminal_seal != _sha256_bytes(_canonical_json(terminal_body))
        or terminal.get("pair_count") != PRODUCTION_PAIR_COUNT
        or terminal.get("pairs_per_stage") != PRODUCTION_PAIRS_PER_STAGE
        or terminal.get("stage_count") != PRODUCTION_STAGE_COUNT
        or terminal.get("scorer_batch_size") != 16
        or terminal.get("program_packet_sha256") != identity.get("program_packet", {}).get("sha256")
        or terminal.get("representation_status") != BLOCKED_REPRESENTATION_STATUS
        or terminal.get("pre_encode_identity_receipt") != identity_record
    ):
        raise SelectedPreimageOperandAdapterError("G58 terminal stage-chain binding differs")
    aggregate_row = terminal.get("auxiliary_aggregate_receipt")
    expected_aggregate_row = {
        **_regular_file_identity(
            aggregate.aggregate_receipt_path,
            label="G51 aggregate receipt",
        ),
        "self_seal_sha256": (aggregate.aggregate_receipt_self_seal_sha256),
    }
    if aggregate_row != expected_aggregate_row:
        raise SelectedPreimageOperandAdapterError("terminal chain does not bind the exact G51 aggregate")
    _reopen_file_row(
        terminal.get("campaign_receipt"),
        label="G59 campaign receipt",
    )
    raw_stages = terminal.get("stages")
    if type(raw_stages) is not list or len(raw_stages) != PRODUCTION_STAGE_COUNT:
        raise SelectedPreimageOperandAdapterError("terminal chain does not contain five stages")
    prior_chain = "0" * 64
    observed_behavior_sections: set[str] = set()
    identity_factor_ids = tuple(factors.get("section_ids", ()))
    if not identity_factor_ids or factors.get("behavior_changing_section_ids") != list(identity_factor_ids):
        raise SelectedPreimageOperandAdapterError("identity does not bind all behavior-changing factor IDs")
    for stage_index, raw_stage in enumerate(raw_stages):
        admission = _admission_from_mapping(raw_stage)
        start = stage_index * PRODUCTION_PAIRS_PER_STAGE
        stop = start + PRODUCTION_PAIRS_PER_STAGE
        if (
            admission.stage_index != stage_index
            or admission.stage_count != PRODUCTION_STAGE_COUNT
            or admission.pair_range != (start, stop)
            or admission.prior_stage_chain_sha256 != prior_chain
            or admission.program_packet_sha256 != identity["program_packet"]["sha256"]
            or admission.program_packet_bytes != identity["program_packet"]["bytes"]
            or admission.semantic_archive_sha256 != semantic["sha256"]
            or admission.semantic_archive_bytes != semantic["bytes"]
            or admission.fresh_semantic_compile_receipt_sha256 != semantic["fresh_compile_receipt_sha256"]
            or admission.target_custody_receipt_sha256 != target["receipt_sha256"]
            or admission.target_bank_sha256 != target["bank_sha256"]
            or admission.decoder_id != decoder["id"]
            or admission.decoder_implementation_source_sha256 != decoder["implementation_source_sha256"]
            or admission.factor_section_ids != identity_factor_ids
            or admission.auxiliary_planes_forwarded is not False
            or admission.g49_decode_custody_verified is not True
        ):
            raise SelectedPreimageOperandAdapterError(f"terminal stage {stage_index} differs from identity")
        chain_body = asdict(admission)
        chain_body.pop("stage_chain_sha256")
        expected_chain = _sha256_bytes(bytes.fromhex(prior_chain) + _canonical_json(chain_body))
        if admission.stage_chain_sha256 != expected_chain:
            raise SelectedPreimageOperandAdapterError(f"terminal stage {stage_index} chain differs")
        observed_behavior_sections.update(admission.behavior_changing_factor_section_ids)
        prior_chain = expected_chain
    if prior_chain != terminal.get("terminal_stage_chain_sha256") or observed_behavior_sections != set(
        identity_factor_ids
    ):
        raise SelectedPreimageOperandAdapterError("terminal chain does not close all factor behavior")

    outer_record, outer = _load_canonical_receipt(
        outer_proof_path,
        label="G58 outer proof",
    )
    outer_body = dict(outer)
    outer_seal = outer_body.pop("receipt_sha256", None)
    if (
        outer.get("schema") != OUTER_PROOF_SCHEMA
        or outer.get("status") != "COUNTED_OUTER_MEMBERS_REOPENED_PUBLIC_CLOSURE_OWED"
        or outer.get("representation_mode") != PROGRAM_RESIDUAL_MODE
        or outer_seal != _sha256_bytes(_canonical_json(outer_body))
        or outer.get("outer_member_names")
        != [
            outer.get("semantic_member_name"),
            outer.get("program_member_name"),
        ]
        or outer.get("outer_members_partition_exactly") is not True
        or outer.get("packet_byte_homes_partition_exactly") is not True
        or outer.get("generic_learned_decoder_source_bound") is not True
        or outer.get("target_labels_embedded") is not False
        or outer.get("advisory_poses_embedded") is not False
    ):
        raise SelectedPreimageOperandAdapterError("G58 physical outer proof fields differ")
    archive_record, archive_payload = _stable_regular_file_identity_and_bytes(
        outer.get("archive_path"),
        label="G58 counted outer archive",
    )
    if archive_record != {
        "path": outer.get("archive_path"),
        "bytes": outer.get("archive_bytes"),
        "sha256": outer.get("archive_sha256"),
    }:
        raise SelectedPreimageOperandAdapterError("G58 counted outer archive identity differs")
    try:
        with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (
                len(names) != 2
                or len(names) != len(set(names))
                or set(names)
                != {
                    outer["semantic_member_name"],
                    outer["program_member_name"],
                }
            ):
                raise SelectedPreimageOperandAdapterError("G58 counted outer archive has an unexpected untyped member")
            for info in infos:
                name = _safe_member_name(info.filename, "G58 counted outer archive member")
                mode = info.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if (
                    name not in set(outer["outer_member_names"])
                    or info.is_dir()
                    or info.flag_bits & 0x1
                    or file_type not in (0, stat.S_IFREG)
                ):
                    raise SelectedPreimageOperandAdapterError("G58 counted outer archive member custody differs")
            semantic_payload = archive.read(outer["semantic_member_name"])
            packet_payload = archive.read(outer["program_member_name"])
    except (KeyError, zipfile.BadZipFile) as exc:
        raise SelectedPreimageOperandAdapterError("G58 counted outer archive members cannot reopen") from exc
    if (
        semantic_payload != semantic_archive_payload
        or len(semantic_payload) != semantic["bytes"]
        or _sha256_bytes(semantic_payload) != semantic["sha256"]
        or len(packet_payload) != identity["program_packet"]["bytes"]
        or _sha256_bytes(packet_payload) != identity["program_packet"]["sha256"]
    ):
        raise SelectedPreimageOperandAdapterError("outer counted members differ from production identity")
    parsed = parse_selected_preimage_program(
        packet_payload,
        maximum_packet_bytes=len(packet_payload),
    )
    homes = parsed.byte_homes()
    parsed_factor_ids = tuple(factor.section_id for factor in parsed.factors)
    learned_factor_contracts: dict[str, tuple[str, str]] = {}
    for factor in parsed.factors:
        if factor.role is SelectedPreimageFactorRoleV1.LEARNED_IRREDUCIBLE_QUOTIENT:
            learned = _parse_learned_payload(
                factor.payload,
                factor=factor,
            )
            learned_factor_contracts[factor.section_id] = (
                learned.decoder_contract_id,
                learned.decoder_implementation_source_sha256,
            )
    raw_contracts = outer.get("learned_decoder_contracts")
    if type(raw_contracts) is not list:
        raise SelectedPreimageOperandAdapterError("outer learned-decoder contracts are not canonical JSON")
    if any(
        type(row) is not list
        or len(row) != 3
        or any(type(value) is not str or not value for value in row)
        or _require_sha256(
            row[2],
            "outer learned decoder source",
        )
        != row[2]
        for row in raw_contracts
    ):
        raise SelectedPreimageOperandAdapterError("outer learned-decoder contract row differs")
    outer_contracts = {row[0]: (row[1], row[2]) for row in raw_contracts}
    source_contracts = {
        section_id: (contract_id, source.sha256)
        for section_id, (
            contract_id,
            source,
        ) in learned_source_rows.items()
    }
    if (
        not homes
        or homes[0].offset != 0
        or any(previous.offset + previous.byte_length != current.offset for previous, current in pairwise(homes))
        or homes[-1].offset + homes[-1].byte_length != len(packet_payload)
        or parsed_factor_ids != identity_factor_ids
        or parsed.semantic_program_identity.compiled_semantic_archive_sha256 != semantic["sha256"]
        or parsed.target_custody_identity.target_custody_receipt_sha256 != target["receipt_sha256"]
        or parsed.target_custody_identity.target_bank_sha256 != target["bank_sha256"]
        or parsed.decoder_identity.implementation_source_sha256 != decoder_callable_sha256
        or len(outer_contracts) != len(raw_contracts)
        or outer_contracts != learned_factor_contracts
        or source_contracts != learned_factor_contracts
        or sum(home.byte_length for home in homes[1:]) != outer.get("factor_payload_bytes_inside_packet")
    ):
        raise SelectedPreimageOperandAdapterError("outer packet parse-back differs from production custody")
    return {
        "schema": PRODUCTION_PRE_ENCODE_EVIDENCE_SCHEMA,
        "status": "ADMIT",
        "representation_mode": PROGRAM_RESIDUAL_MODE,
        "identity_receipt": identity_record,
        "terminal_stage_chain_receipt": terminal_record,
        "outer_proof_receipt": outer_record,
        "campaign_receipt": terminal["campaign_receipt"],
        "auxiliary_aggregate_receipt": aggregate_row,
        "semantic_archive": asdict(semantic_archive),
        "semantic_compile_receipt": asdict(semantic_compile_receipt),
        "target_custody_receipt": asdict(target_custody_receipt),
        "generic_v10_source": asdict(generic_v10_source),
        "outer_archive": archive_record,
        "pair_count": PRODUCTION_PAIR_COUNT,
        "pairs_per_stage": PRODUCTION_PAIRS_PER_STAGE,
        "stage_count": PRODUCTION_STAGE_COUNT,
        "scorer_batch_size": 16,
        "program_packet_sha256": parsed.packet_sha256,
        "factor_section_ids": list(parsed_factor_ids),
        "behavior_changing_factor_section_ids": sorted(observed_behavior_sections),
        "learned_factor_section_ids": sorted(learned_factor_contracts),
        "analytic_only": not learned_factor_contracts,
        "target_payload_embedded": False,
        "historical_payload_reused": False,
    }


__all__ = [
    "ADMISSION_SCHEMA",
    "BLOCKED_REPRESENTATION_STATUS",
    "FIXTURE_AUXILIARY_AGGREGATE_SCHEMA",
    "NEXT_PRECLOSURE_GATE",
    "OUTER_PROOF_SCHEMA",
    "PRODUCTION_PRE_ENCODE_EVIDENCE_SCHEMA",
    "PROGRAM_RESIDUAL_MODE",
    "SCHEMA",
    "TERMINAL_STAGE_CHAIN_SCHEMA",
    "AuxiliaryFreshOperandProviderV1",
    "AuxiliaryOperandCustodyV1",
    "LearnedDecoderSourceCustodyV1",
    "ProgramResidualOuterArchiveProofV1",
    "ReopenableRegularFileIdentityV1",
    "SelectedPreimageFreshOperandStageV1",
    "SelectedPreimageOperandAdapterError",
    "SelectedPreimagePreEncodeAdmissionV1",
    "SelectedPreimageProductionCustodyV1",
    "TaskspaceSelectedPreimageFreshOperandAdapterV1",
    "publish_program_residual_outer_archive_proof",
    "reopen_program_residual_outer_archive",
    "reopen_program_residual_production_pre_encode_evidence",
    "validate_pre_encode_stage",
]
