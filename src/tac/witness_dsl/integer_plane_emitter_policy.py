"""Typed, default-OFF contract for the C2 integer-plane emitter.

This module deliberately owns no trainer flag.  It describes the build-only C2
vehicle, resolves the measured U4 head constants, and provides a standalone
checkpoint envelope for the future trainer integration.  The eventual live
controller hooks are reserved under ``__ipe_`` but are not registered here.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from tac.witness_dsl.lawref import (
    LADDER_MEASURED_ANCHOR,
    InputRef,
    LawRef,
    lawref_to_declaration,
    resolve,
)

POLICY_SCHEMA = "integer_plane_emitter_policy.v1"
POLICY_NAME = "IntegerPlaneEmitter"
LANE_ID = "lane_c2_integer_plane_emitter_build_20260719"
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)
PLANE_COUNT = 2
CHANNELS = 3
RESIDUAL_WIDTH = 4
U4_SOURCE_ARTIFACT = ".omx/research/v10_power_diagram_frame195_diagnostic_20260718.json"
U4_SOURCE_SHA256 = "65d97194c6298a5502d0fcc792ee2fe3bf05599c69f1130d64c270dec5ec36ee"
FROZEN_SEGNET_SHA256 = "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"
MEASURED_U4_SINGULAR_VALUES = (
    3.1283763256,
    2.1542713873,
    2.0247078699,
    1.7962638357,
)
FUTURE_RESUME_HOOK_PREFIX = "__ipe_"
CHECKPOINT_SCHEMA = "integer_plane_emitter_stage_checkpoint.v1"
POLICY_CONTRACT_RECEIPT_KEY = "c2_integer_plane_emitter_policy_contract"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")


class IntegerPlaneEmitterPolicyError(ValueError):
    """Raised when the sealed build-only contract is violated."""


class IntegerPlaneEmitterCheckpointError(ValueError):
    """Raised when checkpoint custody, canonicality, or state is invalid."""


def _stat_identity(value: os.stat_result) -> tuple[int, int]:
    return (value.st_dev, value.st_ino)


def _fsync_directory(directory: Path) -> None:
    directory_fd = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _verify_published_checkpoint_target(
    target: Path,
    *,
    expected_stat: os.stat_result,
    payload: bytes,
) -> None:
    open_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        target_fd = os.open(target, open_flags)
    except OSError as exc:
        raise IntegerPlaneEmitterCheckpointError(
            f"published checkpoint target cannot be opened safely: {target}"
        ) from exc
    try:
        opened_stat = os.fstat(target_fd)
        if not stat.S_ISREG(opened_stat.st_mode) or _stat_identity(
            opened_stat
        ) != _stat_identity(expected_stat):
            raise IntegerPlaneEmitterCheckpointError(
                f"published checkpoint target has unexpected identity: {target}"
            )
        with os.fdopen(target_fd, "rb") as target_handle:
            target_fd = -1
            recovered = target_handle.read()
            after_read_stat = os.fstat(target_handle.fileno())
    finally:
        if target_fd >= 0:
            os.close(target_fd)
    if recovered != payload:
        raise IntegerPlaneEmitterCheckpointError(
            f"published checkpoint target bytes differ from requested bytes: {target}"
        )
    if (
        not stat.S_ISREG(after_read_stat.st_mode)
        or _stat_identity(after_read_stat) != _stat_identity(expected_stat)
        or after_read_stat.st_size != len(payload)
        or after_read_stat.st_size != opened_stat.st_size
        or after_read_stat.st_mtime_ns != opened_stat.st_mtime_ns
    ):
        raise IntegerPlaneEmitterCheckpointError(
            f"published checkpoint target changed during verification: {target}"
        )
    try:
        final_lstat = target.lstat()
    except OSError as exc:
        raise IntegerPlaneEmitterCheckpointError(
            f"published checkpoint target disappeared during verification: {target}"
        ) from exc
    if (
        not stat.S_ISREG(final_lstat.st_mode)
        or _stat_identity(final_lstat) != _stat_identity(after_read_stat)
        or final_lstat.st_size != after_read_stat.st_size
        or final_lstat.st_mtime_ns != after_read_stat.st_mtime_ns
    ):
        raise IntegerPlaneEmitterCheckpointError(
            f"published checkpoint target changed before final verification: {target}"
        )


def _remove_failed_checkpoint_target(target: Path, directory: Path) -> None:
    cleanup_error: OSError | None = None
    try:
        os.unlink(target)
    except FileNotFoundError:
        pass
    except OSError as exc:
        cleanup_error = exc
    try:
        _fsync_directory(directory)
    except OSError as exc:
        if cleanup_error is None:
            cleanup_error = exc
    if cleanup_error is not None:
        raise IntegerPlaneEmitterCheckpointError(
            f"failed to remove an unverified checkpoint target: {target}"
        ) from cleanup_error


def _verify_or_remove_checkpoint_target(
    target: Path,
    directory: Path,
    *,
    expected_stat: os.stat_result,
    payload: bytes,
) -> None:
    try:
        _verify_published_checkpoint_target(
            target,
            expected_stat=expected_stat,
            payload=payload,
        )
    except (IntegerPlaneEmitterCheckpointError, OSError) as exc:
        try:
            _remove_failed_checkpoint_target(target, directory)
        except IntegerPlaneEmitterCheckpointError as cleanup_exc:
            raise cleanup_exc from exc
        if isinstance(exc, IntegerPlaneEmitterCheckpointError):
            raise
        raise IntegerPlaneEmitterCheckpointError(
            f"published checkpoint target verification failed: {target}"
        ) from exc


def _publish_checkpoint_link(
    source: Path,
    target: Path,
    directory: Path,
    *,
    expected_stat: os.stat_result,
    payload: bytes,
) -> Path:
    try:
        os.link(source, target, follow_symlinks=False)
    except FileExistsError as exc:
        raise IntegerPlaneEmitterCheckpointError(
            f"checkpoint path appeared during publication; overwrite refused: {target}"
        ) from exc
    except OSError as exc:
        raise IntegerPlaneEmitterCheckpointError(
            f"checkpoint no-clobber publication failed: {source} -> {target}"
        ) from exc

    _verify_or_remove_checkpoint_target(
        target,
        directory,
        expected_stat=expected_stat,
        payload=payload,
    )

    # Persist the verified hard link before removing the already-fsynced source
    # name.  A crash can therefore leave two names, never zero names.
    try:
        _fsync_directory(directory)
    except OSError as exc:
        _verify_or_remove_checkpoint_target(
            target,
            directory,
            expected_stat=expected_stat,
            payload=payload,
        )
        raise IntegerPlaneEmitterCheckpointError(
            f"published checkpoint directory could not be synced: {directory}"
        ) from exc
    _verify_or_remove_checkpoint_target(
        target,
        directory,
        expected_stat=expected_stat,
        payload=payload,
    )
    try:
        source_lstat = source.lstat()
    except OSError as exc:
        _verify_or_remove_checkpoint_target(
            target,
            directory,
            expected_stat=expected_stat,
            payload=payload,
        )
        raise IntegerPlaneEmitterCheckpointError(
            "published checkpoint is verified, but source cleanup was refused because "
            f"the source disappeared: {source}"
        ) from exc
    if (
        not stat.S_ISREG(source_lstat.st_mode)
        or _stat_identity(source_lstat) != _stat_identity(expected_stat)
        or source_lstat.st_size != expected_stat.st_size
        or source_lstat.st_mtime_ns != expected_stat.st_mtime_ns
    ):
        _verify_or_remove_checkpoint_target(
            target,
            directory,
            expected_stat=expected_stat,
            payload=payload,
        )
        raise IntegerPlaneEmitterCheckpointError(
            "published checkpoint is verified, but source cleanup was refused because "
            f"the source identity changed: {source}"
        )
    # Recheck after the durability barrier and immediately before destructive
    # source cleanup. This closes substitutions during the first verification
    # or directory fsync; no local protocol can control replacements made by an
    # uncooperative process after this function returns.
    _verify_or_remove_checkpoint_target(
        target,
        directory,
        expected_stat=expected_stat,
        payload=payload,
    )
    try:
        os.unlink(source)
    except OSError as exc:
        _verify_or_remove_checkpoint_target(
            target,
            directory,
            expected_stat=expected_stat,
            payload=payload,
        )
        raise IntegerPlaneEmitterCheckpointError(
            f"published checkpoint source could not be removed safely: {source}"
        ) from exc
    try:
        _fsync_directory(directory)
    except OSError as exc:
        _verify_or_remove_checkpoint_target(
            target,
            directory,
            expected_stat=expected_stat,
            payload=payload,
        )
        raise IntegerPlaneEmitterCheckpointError(
            f"checkpoint source removal could not be synced: {directory}"
        ) from exc
    _verify_or_remove_checkpoint_target(
        target,
        directory,
        expected_stat=expected_stat,
        payload=payload,
    )
    return target


class BasisMode(StrEnum):
    """The fixed-capacity basis A/B; it does not change emitter capacity."""

    RAW_CENTERED = "raw_centered"
    SIGN_FIXED_U4_PAIR_MARGIN = "sign_fixed_u4_pair_margin"


class STEMode(StrEnum):
    """The only C2 receiver-compatible straight-through mode."""

    SATURATION_AWARE_UINT8 = "saturation_aware_uint8"


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise IntegerPlaneEmitterCheckpointError("checkpoint state is not canonical JSON") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise IntegerPlaneEmitterCheckpointError(f"{field} must be a lowercase SHA-256")
    if value == "0" * 64:
        raise IntegerPlaneEmitterCheckpointError(f"{field} must not be an all-zero placeholder")


def _require_nonempty_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise IntegerPlaneEmitterCheckpointError(f"{field} must be a nonempty mapping")
    return value


def _require_f32_tensor_payload(
    value: Any,
    *,
    field: str,
    expected_tail: tuple[int, ...],
    variable_leading_dim: bool,
) -> tuple[int, ...]:
    tensor = _require_nonempty_mapping(value, field)
    if set(tensor) != {"dtype", "shape", "data"}:
        raise IntegerPlaneEmitterCheckpointError(
            f"{field} must contain exactly dtype, shape, and data"
        )
    if tensor["dtype"] != "float32":
        raise IntegerPlaneEmitterCheckpointError(f"{field}.dtype must be float32")
    shape = tensor["shape"]
    if not isinstance(shape, list) or not shape:
        raise IntegerPlaneEmitterCheckpointError(f"{field}.shape must be a nonempty list")
    if any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape):
        raise IntegerPlaneEmitterCheckpointError(f"{field}.shape dimensions must be positive integers")
    shape_tuple = tuple(shape)
    if variable_leading_dim:
        if len(shape_tuple) != len(expected_tail) + 1 or shape_tuple[1:] != expected_tail:
            raise IntegerPlaneEmitterCheckpointError(
                f"{field}.shape must be [N,{','.join(map(str, expected_tail))}]"
            )
    elif shape_tuple != expected_tail:
        raise IntegerPlaneEmitterCheckpointError(
            f"{field}.shape must be {list(expected_tail)}"
        )
    data = tensor["data"]
    if not isinstance(data, list):
        raise IntegerPlaneEmitterCheckpointError(f"{field}.data must be a flat list")
    expected_count = math.prod(shape_tuple)
    if len(data) != expected_count:
        raise IntegerPlaneEmitterCheckpointError(
            f"{field}.data length {len(data)} does not match shape product {expected_count}"
        )
    for item in data:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise IntegerPlaneEmitterCheckpointError(
                f"{field}.data must contain only finite numeric values"
            )
    return shape_tuple


def _validate_residual_state(value: Any, field: str, residual_width: int) -> int:
    state = _require_nonempty_mapping(value, field)
    if set(state) != {"pair_plane_codes", "shared_rgb_head"}:
        raise IntegerPlaneEmitterCheckpointError(
            f"{field} must contain exactly pair_plane_codes and shared_rgb_head"
        )
    code_shape = _require_f32_tensor_payload(
        state["pair_plane_codes"],
        field=f"{field}.pair_plane_codes",
        expected_tail=(PLANE_COUNT, residual_width),
        variable_leading_dim=True,
    )
    _require_f32_tensor_payload(
        state["shared_rgb_head"],
        field=f"{field}.shared_rgb_head",
        expected_tail=(residual_width, CHANNELS),
        variable_leading_dim=False,
    )
    return code_shape[0]


def _validate_policy_contract(contract: Any) -> Mapping[str, Any]:
    doc = _require_nonempty_mapping(contract, "policy_contract")
    required = {
        "schema",
        "name",
        "basis",
        "ste",
        "residual_width",
        "capacity_signature",
        "capacity_locked",
        "basis_verdict_state",
        "pair_parallel_expansion",
        "cross_pair_autoregression",
        "score_affecting_enabled",
        "trainer_activation",
        "launch",
        "paid_dispatch",
        "score_claim",
        "promotion",
        "pointer_mutation",
        "policy_sha256",
        "lawref_declaration_sha256",
    }
    missing = sorted(required - set(doc))
    if missing:
        raise IntegerPlaneEmitterCheckpointError(
            f"policy_contract missing required fields: {missing}"
        )
    _require_sha256(doc["policy_sha256"], "policy_contract.policy_sha256")
    _require_sha256(doc["capacity_signature"], "policy_contract.capacity_signature")
    try:
        basis = BasisMode(doc["basis"])
    except (TypeError, ValueError) as exc:
        raise IntegerPlaneEmitterCheckpointError("policy_contract basis is unknown") from exc
    expected = IntegerPlaneEmitterPolicy(basis=basis).compile_contract()
    drifted = sorted(
        field
        for field in set(doc) | set(expected)
        if doc.get(field) != expected.get(field)
    )
    if drifted:
        raise IntegerPlaneEmitterCheckpointError(
            "policy_contract differs from the sealed compiled C2 policy: " + ", ".join(drifted)
        )
    _canonical_json(dict(doc))
    return doc


def _u4_lawrefs() -> dict[str, LawRef]:
    """Return the four independent, SHA-pinned measured head-spectrum refs."""

    refs: dict[str, LawRef] = {}
    for index in range(4):
        name = f"sigma_{index + 1}"
        refs[name] = LawRef(
            equation_id="dsl_custodied_scalar_identity_v1",
            inputs={
                "value": InputRef.anchor(
                    U4_SOURCE_ARTIFACT,
                    f"frozen_target/singular_values/{index}",
                    "MEASURED frozen SegNet centered-head singular value; "
                    f"head_sha256={FROZEN_SEGNET_SHA256}; index={index}",
                    expected_sha256=U4_SOURCE_SHA256,
                    config_tags={"frozen_segnet_sha256": FROZEN_SEGNET_SHA256},
                )
            },
            ladder_class=LADDER_MEASURED_ANCHOR,
        )
    return refs


def u4_lawrefs() -> Mapping[str, LawRef]:
    """Public construction surface; callers receive a fresh immutable-value map."""

    return _u4_lawrefs()


@dataclass(frozen=True, slots=True)
class IntegerPlaneEmitterPolicy:
    """Build-only C2 policy with a typed fixed-capacity basis choice.

    ``basis`` may select either arm, while every capacity-bearing field remains
    the same.  All authority fields are sealed false and expansion is explicitly
    pair-independent so a receiver may use deterministic pooling or CUDA.
    """

    basis: BasisMode = BasisMode.RAW_CENTERED
    ste: STEMode = STEMode.SATURATION_AWARE_UINT8
    residual_width: int = RESIDUAL_WIDTH
    camera_hw: tuple[int, int] = CAMERA_HW
    scorer_hw: tuple[int, int] = SCORER_HW
    plane_count: int = PLANE_COUNT
    channels: int = CHANNELS
    capacity_locked: bool = True
    basis_verdict_state: str = "UNRESOLVED_BUILD_ONLY"
    pair_parallel_expansion: bool = True
    cross_pair_autoregression: bool = False
    score_affecting_enabled: bool = False
    research_only: bool = True
    trainer_activation: bool = False
    launch: bool = False
    paid_dispatch: bool = False
    score_claim: bool = False
    promotion: bool = False
    pointer_mutation: bool = False
    schema: str = POLICY_SCHEMA
    name: str = POLICY_NAME
    lane_id: str = LANE_ID

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.basis, BasisMode):
            raise IntegerPlaneEmitterPolicyError("basis must be a BasisMode")
        if self.ste is not STEMode.SATURATION_AWARE_UINT8:
            raise IntegerPlaneEmitterPolicyError("STE mode is sealed to saturation_aware_uint8")
        if not isinstance(self.residual_width, int) or isinstance(self.residual_width, bool):
            raise IntegerPlaneEmitterPolicyError("residual_width must be an integer")
        if self.residual_width != RESIDUAL_WIDTH:
            raise IntegerPlaneEmitterPolicyError(
                "residual_width is capacity-locked until the basis verdict resolves"
            )
        expected = {
            "camera_hw": CAMERA_HW,
            "scorer_hw": SCORER_HW,
            "plane_count": PLANE_COUNT,
            "channels": CHANNELS,
            "capacity_locked": True,
            "basis_verdict_state": "UNRESOLVED_BUILD_ONLY",
            "pair_parallel_expansion": True,
            "cross_pair_autoregression": False,
            "score_affecting_enabled": False,
            "research_only": True,
            "trainer_activation": False,
            "launch": False,
            "paid_dispatch": False,
            "score_claim": False,
            "promotion": False,
            "pointer_mutation": False,
            "schema": POLICY_SCHEMA,
            "name": POLICY_NAME,
            "lane_id": LANE_ID,
        }
        changed = [key for key, value in expected.items() if getattr(self, key) != value]
        if changed:
            raise IntegerPlaneEmitterPolicyError(
                "integer-plane policy sealed field changed: " + ", ".join(changed)
            )

    def capacity_signature(self) -> str:
        """Hash capacity only; intentionally exclude the A/B basis identity."""

        payload = {
            "schema": "integer_plane_emitter_capacity.v1",
            "geometry": [self.plane_count, *self.scorer_hw, self.channels],
            "residual_width": self.residual_width,
            "pair_plane_codes": "independent_pair_plane_codes",
            "shared_residual_head": True,
        }
        return _sha256(_canonical_json(payload))

    def compile_contract(self, **requested_authority: bool) -> dict[str, Any]:
        """Resolve custody and return a JSON-safe, deterministic-hash contract."""

        self.validate()
        authority_names = {
            "score_affecting_enabled",
            "trainer_activation",
            "launch",
            "paid_dispatch",
            "score_claim",
            "promotion",
            "pointer_mutation",
        }
        unknown = sorted(set(requested_authority) - authority_names)
        if unknown:
            raise IntegerPlaneEmitterPolicyError(f"unknown authority request: {unknown}")
        attempted = sorted(name for name, value in requested_authority.items() if value)
        if attempted:
            raise IntegerPlaneEmitterPolicyError(
                "integer-plane build policy cannot authorize " + ", ".join(attempted)
            )

        refs = _u4_lawrefs()
        declarations = {
            name: lawref_to_declaration(ref) for name, ref in sorted(refs.items())
        }
        declaration_hashes = {
            name: _sha256(_canonical_json(declaration))
            for name, declaration in declarations.items()
        }
        resolution: dict[str, Any] = {}
        values: list[float] = []
        target_tags = {"frozen_segnet_sha256": FROZEN_SEGNET_SHA256}
        for index, (name, ref) in enumerate(sorted(refs.items())):
            record = resolve(ref, target_config_tags=target_tags).to_dict()
            value = float(record["value"])
            if not math.isfinite(value) or round(value, 10) != MEASURED_U4_SINGULAR_VALUES[index]:
                raise IntegerPlaneEmitterPolicyError(
                    f"{name} no longer matches the measured ten-decimal U4 custody value"
                )
            record.pop("resolved_at", None)
            for input_record in record.get("inputs", []):
                if isinstance(input_record, dict) and input_record.get("kind") == "anchor":
                    input_record["source"] = U4_SOURCE_ARTIFACT
            resolution[name] = record
            values.append(value)

        hash_basis = {
            "schema": self.schema,
            "name": self.name,
            "lane_id": self.lane_id,
            "basis": self.basis.value,
            "ste": self.ste.value,
            "residual_width": self.residual_width,
            "camera_hw": list(self.camera_hw),
            "scorer_hw": list(self.scorer_hw),
            "plane_count": self.plane_count,
            "channels": self.channels,
            "capacity_signature": self.capacity_signature(),
            "capacity_locked": self.capacity_locked,
            "basis_verdict_state": self.basis_verdict_state,
            "pair_parallel_expansion": self.pair_parallel_expansion,
            "cross_pair_autoregression": self.cross_pair_autoregression,
            "score_affecting_enabled": self.score_affecting_enabled,
            "research_only": self.research_only,
            "trainer_activation": self.trainer_activation,
            "launch": self.launch,
            "paid_dispatch": self.paid_dispatch,
            "score_claim": self.score_claim,
            "promotion": self.promotion,
            "pointer_mutation": self.pointer_mutation,
            "lawref_declaration_sha256": declaration_hashes,
            "resume_hook_status": "FUTURE_ONLY_NOT_REGISTERED",
            "future_resume_hook_prefix": FUTURE_RESUME_HOOK_PREFIX,
        }
        return {
            **hash_basis,
            "policy_sha256": _sha256(_canonical_json(hash_basis)),
            "u4_singular_values": values,
            "u4_measured_ten_decimal_values": list(MEASURED_U4_SINGULAR_VALUES),
            "lawref_declarations": declarations,
            "lawref_resolution": resolution,
        }

    def compile(self, **requested_authority: bool) -> dict[str, Any]:
        return self.compile_contract(**requested_authority)


_CHECKPOINT_BODY_FIELDS = (
    "schema",
    "schema_sha256",
    "policy_contract",
    "config_sha256",
    "stage_name",
    "stage_index",
    "epoch",
    "global_step",
    "next_pair",
    "basis_id",
    "ste_id",
    "fixed_capacity_signature",
    "live_residual_parameters",
    "ema_shadow",
    "optimizer_state",
    "rng_state",
    "topology_state_sha256",
    "discrete_state_sha256",
    "event_state_sha256",
    "dual_state_sha256",
)
CHECKPOINT_SCHEMA_SHA256 = _sha256(_canonical_json(list(_CHECKPOINT_BODY_FIELDS)))


@dataclass(frozen=True, slots=True)
class IntegerPlaneEmitterStageCheckpoint:
    """Canonical, integrity-sealed C2 stage state, independent of ResumeRegistry."""

    policy_contract: Mapping[str, Any]
    config_sha256: str
    stage_name: str
    stage_index: int
    epoch: int
    global_step: int
    next_pair: int
    basis_id: str
    ste_id: str
    fixed_capacity_signature: str
    live_residual_parameters: Mapping[str, Any]
    ema_shadow: Mapping[str, Any]
    optimizer_state: Mapping[str, Any]
    rng_state: Mapping[str, Any]
    topology_state_sha256: str
    discrete_state_sha256: str
    event_state_sha256: str
    dual_state_sha256: str
    schema: str = CHECKPOINT_SCHEMA
    schema_sha256: str = CHECKPOINT_SCHEMA_SHA256

    def __post_init__(self) -> None:
        if self.schema != CHECKPOINT_SCHEMA or self.schema_sha256 != CHECKPOINT_SCHEMA_SHA256:
            raise IntegerPlaneEmitterCheckpointError("checkpoint schema custody mismatch")
        for field in (
            "config_sha256",
            "fixed_capacity_signature",
            "topology_state_sha256",
            "discrete_state_sha256",
            "event_state_sha256",
            "dual_state_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        policy_contract = _validate_policy_contract(self.policy_contract)
        if self.config_sha256 != policy_contract["policy_sha256"]:
            raise IntegerPlaneEmitterCheckpointError(
                "config_sha256 must equal policy_contract.policy_sha256"
            )
        if self.fixed_capacity_signature != policy_contract["capacity_signature"]:
            raise IntegerPlaneEmitterCheckpointError(
                "fixed_capacity_signature must equal policy_contract.capacity_signature"
            )
        if not isinstance(self.stage_name, str) or _SAFE_COMPONENT_RE.fullmatch(self.stage_name) is None:
            raise IntegerPlaneEmitterCheckpointError("stage_name is not filename-safe")
        for field in ("stage_index", "epoch", "global_step", "next_pair"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise IntegerPlaneEmitterCheckpointError(f"{field} must be a nonnegative integer")
        if self.basis_id not in {mode.value for mode in BasisMode}:
            raise IntegerPlaneEmitterCheckpointError("basis_id is unknown")
        if self.basis_id != policy_contract["basis"]:
            raise IntegerPlaneEmitterCheckpointError("basis_id differs from policy_contract basis")
        if self.ste_id != STEMode.SATURATION_AWARE_UINT8.value:
            raise IntegerPlaneEmitterCheckpointError("ste_id is not the sealed C2 STE")
        if self.ste_id != policy_contract["ste"]:
            raise IntegerPlaneEmitterCheckpointError("ste_id differs from policy_contract STE")
        residual_width = int(policy_contract["residual_width"])
        live_pair_count = _validate_residual_state(
            self.live_residual_parameters, "live_residual_parameters", residual_width
        )
        ema_pair_count = _validate_residual_state(
            self.ema_shadow, "ema_shadow", residual_width
        )
        if live_pair_count != ema_pair_count:
            raise IntegerPlaneEmitterCheckpointError(
                "live_residual_parameters and ema_shadow pair counts differ"
            )
        if self.next_pair > live_pair_count:
            raise IntegerPlaneEmitterCheckpointError(
                "next_pair exceeds checkpoint residual pair count"
            )
        optimizer_state = _require_nonempty_mapping(self.optimizer_state, "optimizer_state")
        optimizer_step = optimizer_state.get("step")
        if (
            not isinstance(optimizer_step, int)
            or isinstance(optimizer_step, bool)
            or optimizer_step < 0
        ):
            raise IntegerPlaneEmitterCheckpointError(
                "optimizer_state.step must be a nonnegative integer"
            )
        _require_nonempty_mapping(self.rng_state, "rng_state")
        _canonical_json(self._body())
        for field in (
            "policy_contract",
            "live_residual_parameters",
            "ema_shadow",
            "optimizer_state",
            "rng_state",
        ):
            copied = json.loads(_canonical_json(getattr(self, field)).decode("ascii"))
            object.__setattr__(self, field, copied)

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_sha256": self.schema_sha256,
            "policy_contract": dict(self.policy_contract),
            "config_sha256": self.config_sha256,
            "stage_name": self.stage_name,
            "stage_index": self.stage_index,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "next_pair": self.next_pair,
            "basis_id": self.basis_id,
            "ste_id": self.ste_id,
            "fixed_capacity_signature": self.fixed_capacity_signature,
            "live_residual_parameters": dict(self.live_residual_parameters),
            "ema_shadow": dict(self.ema_shadow),
            "optimizer_state": dict(self.optimizer_state),
            "rng_state": dict(self.rng_state),
            "topology_state_sha256": self.topology_state_sha256,
            "discrete_state_sha256": self.discrete_state_sha256,
            "event_state_sha256": self.event_state_sha256,
            "dual_state_sha256": self.dual_state_sha256,
        }

    def to_bytes(self) -> bytes:
        body = self._body()
        body_bytes = _canonical_json(body)
        return _canonical_json({"body": body, "body_sha256": _sha256(body_bytes)})

    @classmethod
    def from_bytes(cls, payload: bytes) -> IntegerPlaneEmitterStageCheckpoint:
        if not isinstance(payload, bytes) or not payload:
            raise IntegerPlaneEmitterCheckpointError("checkpoint input must be nonempty bytes")
        try:
            doc = json.loads(payload.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IntegerPlaneEmitterCheckpointError("checkpoint is not canonical JSON") from exc
        if not isinstance(doc, dict) or set(doc) != {"body", "body_sha256"}:
            raise IntegerPlaneEmitterCheckpointError("checkpoint envelope fields mismatch")
        if _canonical_json(doc) != payload:
            raise IntegerPlaneEmitterCheckpointError("checkpoint encoding is noncanonical")
        body = doc["body"]
        if not isinstance(body, dict) or set(body) != set(_CHECKPOINT_BODY_FIELDS):
            raise IntegerPlaneEmitterCheckpointError("checkpoint body fields mismatch")
        _require_sha256(doc["body_sha256"], "body_sha256")
        if _sha256(_canonical_json(body)) != doc["body_sha256"]:
            raise IntegerPlaneEmitterCheckpointError("checkpoint body hash mismatch")
        return cls(**body)

    def filename(self, run_id: str) -> str:
        if not isinstance(run_id, str) or _SAFE_COMPONENT_RE.fullmatch(run_id) is None:
            raise IntegerPlaneEmitterCheckpointError("run_id is not filename-safe")
        return (
            f"{run_id}__ipe_stage{self.stage_index:03d}_{self.stage_name}"
            f"_ep{self.epoch:06d}_step{self.global_step:012d}.json"
        )

    def write_new(self, directory: str | Path, run_id: str) -> Path:
        """Atomically no-clobber publish a distinctly named checkpoint."""

        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        target = root / self.filename(run_id)
        lock_path = root / ".integer_plane_emitter_checkpoint.lock"
        payload = self.to_bytes()
        with lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                target.lstat()
            except FileNotFoundError:
                pass
            else:
                raise IntegerPlaneEmitterCheckpointError(
                    f"checkpoint path already exists; overwrite refused: {target}"
                )
            stale = sorted(root.glob(f".{target.name}.tmp.*"))
            if len(stale) > 1:
                raise IntegerPlaneEmitterCheckpointError(
                    f"multiple stale checkpoint temporaries require review: {stale}"
                )
            if stale:
                prior = stale[0]
                try:
                    prior_lstat = prior.lstat()
                except OSError as exc:
                    raise IntegerPlaneEmitterCheckpointError(
                        f"stale checkpoint temporary cannot be inspected: {prior}"
                    ) from exc
                if stat.S_ISLNK(prior_lstat.st_mode) or not stat.S_ISREG(
                    prior_lstat.st_mode
                ):
                    raise IntegerPlaneEmitterCheckpointError(
                        f"stale checkpoint temporary must be a non-symlink regular file: {prior}"
                    )
                open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
                    os, "O_NOFOLLOW", 0
                )
                try:
                    prior_fd = os.open(prior, open_flags)
                except OSError as exc:
                    raise IntegerPlaneEmitterCheckpointError(
                        f"stale checkpoint temporary cannot be opened safely: {prior}"
                    ) from exc
                try:
                    opened_stat = os.fstat(prior_fd)
                    if not stat.S_ISREG(opened_stat.st_mode) or (
                        opened_stat.st_dev,
                        opened_stat.st_ino,
                    ) != (prior_lstat.st_dev, prior_lstat.st_ino):
                        raise IntegerPlaneEmitterCheckpointError(
                            f"stale checkpoint temporary changed during safe open: {prior}"
                        )
                    with os.fdopen(prior_fd, "rb") as prior_handle:
                        prior_fd = -1
                        recovered = prior_handle.read()
                        if recovered != payload:
                            raise IntegerPlaneEmitterCheckpointError(
                                "stale checkpoint temporary differs from requested bytes: "
                                f"{prior}"
                            )
                        # Recovery promotes bytes written by a prior process. Sync
                        # that exact open regular file before publishing its second
                        # same-directory hard link.
                        os.fsync(prior_handle.fileno())
                finally:
                    if prior_fd >= 0:
                        os.close(prior_fd)
                try:
                    final_lstat = prior.lstat()
                except OSError as exc:
                    raise IntegerPlaneEmitterCheckpointError(
                        f"stale checkpoint temporary disappeared before recovery: {prior}"
                    ) from exc
                if (
                    not stat.S_ISREG(final_lstat.st_mode)
                    or (final_lstat.st_dev, final_lstat.st_ino)
                    != (opened_stat.st_dev, opened_stat.st_ino)
                    or final_lstat.st_size != opened_stat.st_size
                    or final_lstat.st_mtime_ns != opened_stat.st_mtime_ns
                ):
                    raise IntegerPlaneEmitterCheckpointError(
                        f"stale checkpoint temporary changed before recovery: {prior}"
                    )
                return _publish_checkpoint_link(
                    prior,
                    target,
                    root,
                    expected_stat=opened_stat,
                    payload=payload,
                )
            tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.tmp.", dir=root)
            tmp = Path(tmp_name)
            with os.fdopen(tmp_fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                opened_stat = os.fstat(handle.fileno())
                if not stat.S_ISREG(opened_stat.st_mode):
                    raise IntegerPlaneEmitterCheckpointError(
                        f"fresh checkpoint temporary is not a regular file: {tmp}"
                    )
            return _publish_checkpoint_link(
                tmp,
                target,
                root,
                expected_stat=opened_stat,
                payload=payload,
            )


__all__ = [
    "CAMERA_HW",
    "CHANNELS",
    "CHECKPOINT_SCHEMA",
    "CHECKPOINT_SCHEMA_SHA256",
    "FROZEN_SEGNET_SHA256",
    "FUTURE_RESUME_HOOK_PREFIX",
    "LANE_ID",
    "MEASURED_U4_SINGULAR_VALUES",
    "PLANE_COUNT",
    "POLICY_CONTRACT_RECEIPT_KEY",
    "POLICY_NAME",
    "POLICY_SCHEMA",
    "RESIDUAL_WIDTH",
    "SCORER_HW",
    "U4_SOURCE_ARTIFACT",
    "U4_SOURCE_SHA256",
    "BasisMode",
    "IntegerPlaneEmitterCheckpointError",
    "IntegerPlaneEmitterPolicy",
    "IntegerPlaneEmitterPolicyError",
    "IntegerPlaneEmitterStageCheckpoint",
    "STEMode",
    "u4_lawrefs",
]
