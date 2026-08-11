# SPDX-License-Identifier: MIT
"""Typed, scorer-free pre-stage apparatus for HR1 realization engineering.

This module implements only the six SAFE-TO-PREPARE obligations named by
``ddm_hr1_realization_engineering_20260811.md``.  It deliberately contains no
renderer, SegNet, PoseNet, optimizer, archive-builder, or launcher import.

The four arm factories compile typed manifests with an empty argv and an
explicit ``execution_allowed=False`` result until real consumers and terminal
bindings land.  These are executable schema/refusal surfaces, not fake
training programs.  The memory compiler reports a shape-derived storage lower
bound and REFUSES without a fresh, exact-config real memory receipt; it never
promotes that lower bound into a peak-RSS claim.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "hr1_prestage.v1"
MEMORY_RECEIPT_MAX_AGE_SECONDS = 24 * 60 * 60


class Hr1PrestageError(ValueError):
    """A typed pre-stage contract is malformed or a custody check failed."""


class Hr1Arm(StrEnum):
    FROZEN_DECODE = "frozen_decode"
    FULL_RENDERER_FINETUNE = "full_renderer_finetune"
    LOW_RANK_ADAPTER = "low_rank_adapter"
    JOINT_TOKEN_RENDERER = "joint_token_renderer"


class BindingState(StrEnum):
    BOUND = "bound"
    UNRESOLVED_TERMINAL = "unresolved_terminal"


class SameParentObjectKind(StrEnum):
    FIT = "fit"
    MAP = "map"
    SELECTOR = "selector"
    CORRECTION = "correction"


class MemoryDisposition(StrEnum):
    REFUSE = "REFUSE"
    PASS = "PASS"


class TensorDType(StrEnum):
    UINT8 = "uint8"
    INT12_PACKED_LOWER_BOUND = "int12_packed_lower_bound"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT64 = "int64"

    @property
    def bits_per_value(self) -> int:
        return {
            TensorDType.UINT8: 8,
            TensorDType.INT12_PACKED_LOWER_BOUND: 12,
            TensorDType.FLOAT16: 16,
            TensorDType.FLOAT32: 32,
            TensorDType.FLOAT64: 64,
            TensorDType.INT64: 64,
        }[self]


def _canonical_json(payload: Mapping[str, Any] | Sequence[Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Hr1PrestageError(f"{label} must be canonical lowercase SHA-256")
    return value


def stream_sha256(path: Path, *, chunk_bytes: int = 1024 * 1024) -> tuple[str, int]:
    """Hash one file with bounded memory and return ``(sha256, bytes)``."""
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise Hr1PrestageError("chunk_bytes must be a positive exact integer")
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise Hr1PrestageError(f"binding target must be a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


@dataclass(frozen=True, slots=True)
class Hr1RoundTripSpec:
    ordering: str = "camera_uint8"
    lift_kernel: str = "bicubic"
    camera_hw: tuple[int, int] = (874, 1164)
    scorer_hw: tuple[int, int] = (384, 512)
    ste_backward: str = "saturation_aware_identity"

    def __post_init__(self) -> None:
        if self.ordering != "camera_uint8":
            raise Hr1PrestageError("HR1 learned-arm round trip must place uint8 at camera resolution")
        if self.lift_kernel not in {"bicubic", "bilinear"}:
            raise Hr1PrestageError("HR1 lift kernel must be typed as bicubic or bilinear")
        for label, shape in (("camera_hw", self.camera_hw), ("scorer_hw", self.scorer_hw)):
            if len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape):
                raise Hr1PrestageError(f"{label} must contain two positive exact integers")


@dataclass(frozen=True, slots=True)
class Hr1ArmProgram:
    arm: Hr1Arm
    roundtrip: Hr1RoundTripSpec
    trainable_state_roles: tuple[str, ...]
    event_graph: tuple[str, ...]
    counted_payload_roles: tuple[str, ...]
    required_consumer_roles: tuple[str, ...]
    initialization_invariant: str
    execution_allowed: bool = False
    consumer_bindings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.execution_allowed:
            raise Hr1PrestageError("pre-stage arm schemas cannot authorize execution")
        for label, values in (
            ("event_graph", self.event_graph),
            ("counted_payload_roles", self.counted_payload_roles),
            ("required_consumer_roles", self.required_consumer_roles),
        ):
            if not values or any(not value.strip() for value in values):
                raise Hr1PrestageError(f"{label} must be a non-empty typed sequence")
        if self.consumer_bindings:
            raise Hr1PrestageError("pre-stage factories must not fabricate consumer bindings")
        if self.arm is Hr1Arm.FROZEN_DECODE and self.trainable_state_roles:
            raise Hr1PrestageError("frozen decode cannot carry trainable state")
        if self.arm is not Hr1Arm.FROZEN_DECODE and not self.trainable_state_roles:
            raise Hr1PrestageError(f"{self.arm.value} must declare its distinct trainable state")

    def compile(self) -> CompiledHr1Program:
        typed_config = {
            "schema_version": SCHEMA_VERSION,
            "arm": self.arm.value,
            "roundtrip": asdict(self.roundtrip),
            "trainable_state_roles": list(self.trainable_state_roles),
            "event_graph": list(self.event_graph),
            "counted_payload_roles": list(self.counted_payload_roles),
            "required_consumer_roles": list(self.required_consumer_roles),
            "initialization_invariant": self.initialization_invariant,
            "consumer_bindings": [],
            "execution_allowed": False,
        }
        config_sha256 = _sha256_bytes(_canonical_json(typed_config))
        return CompiledHr1Program(
            arm=self.arm,
            typed_config_json=_canonical_json(typed_config).decode("ascii"),
            typed_config_sha256=config_sha256,
            argv=(),
            consumer_bound=False,
            execution_allowed=False,
            refusal_reasons=(
                "TERMINAL_BASE_BINDINGS_REQUIRED",
                "REAL_CONSUMERS_REQUIRED",
                "FRESH_REAL_CONFIG_MEMORY_PROBE_REQUIRED",
            ),
        )


@dataclass(frozen=True, slots=True)
class CompiledHr1Program:
    arm: Hr1Arm
    typed_config_json: str
    typed_config_sha256: str
    argv: tuple[str, ...]
    consumer_bound: bool
    execution_allowed: bool
    refusal_reasons: tuple[str, ...]

    @property
    def typed_config(self) -> dict[str, Any]:
        """Return a fresh decoded copy so callers cannot mutate hash-bound state."""
        decoded = json.loads(self.typed_config_json)
        if type(decoded) is not dict:  # pragma: no cover - constructor owns the JSON
            raise Hr1PrestageError("compiled typed config must decode to an object")
        return decoded

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "typed_config": self.typed_config,
            "typed_config_sha256": self.typed_config_sha256,
            "argv": list(self.argv),
            "consumer_bound": self.consumer_bound,
            "execution_allowed": self.execution_allowed,
            "refusal_reasons": list(self.refusal_reasons),
        }


def make_frozen_decode_program() -> Hr1ArmProgram:
    return Hr1ArmProgram(
        arm=Hr1Arm.FROZEN_DECODE,
        roundtrip=Hr1RoundTripSpec(),
        trainable_state_roles=(),
        event_graph=("receiver_bound", "v14_ladder_complete", "rate_hardening", "n600_admitted"),
        counted_payload_roles=("hard_tokens", "probability_object", "coded_stream", "complete_archive"),
        required_consumer_roles=("terminal_receiver", "hpac_encoder", "archive_compiler"),
        initialization_invariant="no_optimizer_no_ema_no_trainable_state",
    )


def make_full_renderer_finetune_program() -> Hr1ArmProgram:
    return Hr1ArmProgram(
        arm=Hr1Arm.FULL_RENDERER_FINETUNE,
        roundtrip=Hr1RoundTripSpec(),
        trainable_state_roles=("complete_semantic_renderer",),
        event_graph=("receiver_bound", "ce_cell_entry", "margin_robustness", "rate_hardening", "pose_terminal", "n600_admitted"),
        counted_payload_roles=("hard_tokens", "complete_renderer_blob", "probability_object", "coded_stream", "complete_archive"),
        required_consumer_roles=("terminal_renderer", "source_lineage_adamw", "frozen_scorers", "archive_compiler"),
        initialization_invariant="terminal_renderer_state_exact_before_first_step",
    )


def make_low_rank_adapter_program() -> Hr1ArmProgram:
    return Hr1ArmProgram(
        arm=Hr1Arm.LOW_RANK_ADAPTER,
        roundtrip=Hr1RoundTripSpec(),
        trainable_state_roles=("film_projection_low_rank_factors", "rgb_head_low_rank_factors"),
        event_graph=("receiver_bound", "rank_birth", "ce_cell_entry", "margin_robustness", "rate_hardening", "pose_terminal", "n600_admitted"),
        counted_payload_roles=("hard_tokens", "adapter_factors", "adapter_scales", "adapter_schema", "probability_object", "coded_stream", "complete_archive"),
        required_consumer_roles=("terminal_renderer_adapter_hooks", "source_lineage_adamw", "frozen_scorers", "archive_compiler"),
        initialization_invariant="rank_factors_make_first_hard_forward_byte_identical_to_frozen_decode",
    )


def make_joint_token_renderer_program() -> Hr1ArmProgram:
    return Hr1ArmProgram(
        arm=Hr1Arm.JOINT_TOKEN_RENDERER,
        roundtrip=Hr1RoundTripSpec(),
        trainable_state_roles=("complete_semantic_renderer", "pair_chunk_local_token_proposals"),
        event_graph=("receiver_bound", "ce_cell_entry", "margin_robustness", "hard_token_reencode", "rate_hardening", "pose_terminal", "n600_admitted"),
        counted_payload_roles=("hard_tokens", "complete_renderer_blob", "token_proposal_state", "probability_object", "coded_stream", "complete_archive"),
        required_consumer_roles=("terminal_renderer", "categorical_token_descent", "hpac_exact_nll", "frozen_scorers", "archive_compiler"),
        initialization_invariant="hard_tokens_start_at_c1_and_renderer_starts_at_terminal_state",
    )


def make_four_arm_race_programs() -> tuple[Hr1ArmProgram, ...]:
    programs = (
        make_frozen_decode_program(),
        make_full_renderer_finetune_program(),
        make_low_rank_adapter_program(),
        make_joint_token_renderer_program(),
    )
    if len({program.arm for program in programs}) != 4:
        raise Hr1PrestageError("four-arm factory must emit every arm exactly once")
    fingerprints = {
        (program.trainable_state_roles, program.event_graph, program.counted_payload_roles)
        for program in programs
    }
    if len(fingerprints) != 4:
        raise Hr1PrestageError("four-arm factory contains enum padding rather than distinct programs")
    return programs


@dataclass(frozen=True, slots=True)
class PayloadRecord:
    role: str
    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.role.strip() or not self.path.strip():
            raise Hr1PrestageError("payload role and path are required")
        if type(self.bytes) is not int or self.bytes < 0:
            raise Hr1PrestageError("payload bytes must be a non-negative exact integer")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise Hr1PrestageError("payload sha256 must be canonical lowercase hex")


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    arm: Hr1Arm
    event: str
    step: int
    config_sha256: str
    root_seed: int
    live_state_roles: tuple[str, ...]
    ema_state_roles: tuple[str, ...]
    optimizer_state_roles: tuple[str, ...]
    rng_state_roles: tuple[str, ...]
    guard_state_roles: tuple[str, ...]
    payloads: tuple[PayloadRecord, ...]

    def __post_init__(self) -> None:
        if not self.event.strip() or type(self.step) is not int or self.step < 0:
            raise Hr1PrestageError("checkpoint event/step is invalid")
        _require_sha256(self.config_sha256, label="checkpoint config_sha256")
        required_groups = {
            "live_state_roles": self.live_state_roles,
            "rng_state_roles": self.rng_state_roles,
            "guard_state_roles": self.guard_state_roles,
        }
        for label, roles in required_groups.items():
            if not roles or any(not role.strip() for role in roles):
                raise Hr1PrestageError(f"checkpoint {label} must be non-empty")
        if self.arm is Hr1Arm.FROZEN_DECODE:
            if self.ema_state_roles or self.optimizer_state_roles:
                raise Hr1PrestageError("frozen decode checkpoint cannot fabricate EMA/optimizer state")
        elif not self.ema_state_roles or not self.optimizer_state_roles:
            raise Hr1PrestageError("trained-arm checkpoint must carry EMA and optimizer state")
        if not self.payloads:
            raise Hr1PrestageError("checkpoint must bind at least one retained payload")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "hr1_checkpoint_manifest.v1",
            "arm": self.arm.value,
            "event": self.event,
            "step": self.step,
            "config_sha256": self.config_sha256,
            "root_seed": self.root_seed,
            "live_state_roles": list(self.live_state_roles),
            "ema_state_roles": list(self.ema_state_roles),
            "optimizer_state_roles": list(self.optimizer_state_roles),
            "rng_state_roles": list(self.rng_state_roles),
            "guard_state_roles": list(self.guard_state_roles),
            "payloads": [asdict(payload) for payload in self.payloads],
        }


@dataclass(frozen=True, slots=True)
class ResumeManifest:
    checkpoint_path: str
    checkpoint_sha256: str
    resume_event: str
    next_step: int
    max_recovery_loss_steps: int

    def __post_init__(self) -> None:
        if not self.checkpoint_path.strip():
            raise Hr1PrestageError("resume checkpoint custody is incomplete")
        _require_sha256(self.checkpoint_sha256, label="resume checkpoint_sha256")
        if not self.resume_event.strip() or type(self.next_step) is not int or self.next_step < 0:
            raise Hr1PrestageError("resume event/next_step is invalid")
        if type(self.max_recovery_loss_steps) is not int or self.max_recovery_loss_steps < 0:
            raise Hr1PrestageError("max recovery loss must be a non-negative exact integer")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> PayloadRecord:
    """Durably write canonical JSON via same-directory tmp + fsync + replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise Hr1PrestageError(f"atomic-write temporary already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest, size = stream_sha256(path)
    return PayloadRecord(role="atomic_json", path=str(path), bytes=size, sha256=digest)


@dataclass(frozen=True, slots=True)
class FileBindingRequest:
    role: str
    path: Path
    expected_bytes: int | None = None
    expected_sha256: str | None = None
    public_intake_read_only: bool = False


@dataclass(frozen=True, slots=True)
class ObjectBinding:
    role: str
    state: BindingState
    path: str | None
    bytes: int | None
    sha256: str | None
    access: str
    resolution_trigger: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "state": self.state.value,
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "access": self.access,
            "resolution_trigger": self.resolution_trigger,
        }


@dataclass(frozen=True, slots=True)
class SameParentFreshnessReceipt:
    """Scorer-free proof that one cached object consumes its producing parent."""

    object_kind: SameParentObjectKind
    producer_parent_sha: str
    consumer_parent_sha: str
    freshness_ok: bool = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.object_kind, SameParentObjectKind):
            try:
                object.__setattr__(self, "object_kind", SameParentObjectKind(self.object_kind))
            except (TypeError, ValueError) as exc:
                raise Hr1PrestageError(
                    "object_kind must be one of fit, map, selector, correction"
                ) from exc
        _require_sha256(self.producer_parent_sha, label="producer_parent_sha")
        _require_sha256(self.consumer_parent_sha, label="consumer_parent_sha")
        expected = self.producer_parent_sha == self.consumer_parent_sha
        object.__setattr__(self, "freshness_ok", expected)
        if not expected:
            raise Hr1PrestageError(
                f"REFUSE stale {self.object_kind.value}: producer and consumer parents differ"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "same_parent_freshness.v1",
            "object_kind": self.object_kind.value,
            "producer_parent_sha": self.producer_parent_sha,
            "consumer_parent_sha": self.consumer_parent_sha,
            "freshness_ok": self.freshness_ok,
        }


def assert_same_parent_freshness(
    *,
    object_kind: SameParentObjectKind | str,
    producer_parent_sha: str,
    consumer_parent_sha: str,
) -> SameParentFreshnessReceipt:
    """Return a typed receipt or refuse absent, malformed, or stale custody."""

    return SameParentFreshnessReceipt(
        object_kind=object_kind,  # type: ignore[arg-type]
        producer_parent_sha=producer_parent_sha,
        consumer_parent_sha=consumer_parent_sha,
    )


def bind_existing_file(request: FileBindingRequest) -> ObjectBinding:
    digest, size = stream_sha256(request.path)
    if request.expected_bytes is not None and size != request.expected_bytes:
        raise Hr1PrestageError(
            f"{request.role} byte mismatch: expected {request.expected_bytes}, observed {size}"
        )
    if request.expected_sha256 is not None and digest != request.expected_sha256:
        raise Hr1PrestageError(
            f"{request.role} SHA-256 mismatch: expected {request.expected_sha256}, observed {digest}"
        )
    return ObjectBinding(
        role=request.role,
        state=BindingState.BOUND,
        path=str(request.path.resolve()),
        bytes=size,
        sha256=digest,
        access="read_only" if request.public_intake_read_only else "bound_read",
    )


def unresolved_terminal_binding(role: str, *, resolution_trigger: str) -> ObjectBinding:
    if not role.strip() or not resolution_trigger.strip():
        raise Hr1PrestageError("unresolved terminal binding requires role and resolution trigger")
    return ObjectBinding(
        role=role,
        state=BindingState.UNRESOLVED_TERMINAL,
        path=None,
        bytes=None,
        sha256=None,
        access="none_until_resolved",
        resolution_trigger=resolution_trigger,
    )


def build_hr1_binding_manifest(repo_root: Path) -> dict[str, Any]:
    """Bind HY1 payloads and HPAC sources; leave terminal objects typed-unresolved."""
    repo_root = Path(repo_root).resolve()
    hy1_root = Path("/Volumes/VertigoDataTier/pact/ddm_hy1_capstone_hybrid_20260811/retained")
    public_root = repo_root / "experiments/results/public_pr130_intake_20260725_fable/source/submissions/semantic-pose-HPAC_CPR1"
    requests = (
        FileBindingRequest(
            "hy1_memo",
            repo_root / ".omx/research/ddm_hy1_capstone_hybrid_20260811.md",
        ),
        FileBindingRequest(
            "c1_solved_tokens",
            hy1_root / "c1_solved_tokens_n600.u8",
            expected_bytes=117_964_800,
            expected_sha256="2b0bdfc38a131ab1ebc3a2c2153a79b1ba23be0037adda66d01ab56f29f4fed5",
        ),
        FileBindingRequest(
            "f26_hpac_rc64_stream",
            hy1_root / "c1_solved_tokens_n600.f26_hpac.rc64",
            expected_bytes=114_717,
            expected_sha256="9def0a4ba849757d473ba2a23cb0fd5370f2566355e5a5cfd398f847349636e8",
        ),
        FileBindingRequest(
            "hy1_hpac_probe_source",
            repo_root / "experiments/ddm_hy1_capstone_hybrid_probe.py",
        ),
        FileBindingRequest(
            "pr130_hpac_integer_source",
            public_root / "hpac_integer.py",
            public_intake_read_only=True,
        ),
        FileBindingRequest(
            "pr130_public_receiver_source",
            public_root / "inflate.py",
            public_intake_read_only=True,
        ),
        FileBindingRequest(
            "lifted_semantic_renderer_source",
            repo_root / "src/tac/pr130_lift/lifted/semantic_renderer_oracle.py",
        ),
        FileBindingRequest(
            "lifted_semantic_training_source",
            repo_root / "src/tac/pr130_lift/lifted/train_semantic_full.py",
        ),
        FileBindingRequest(
            "hb2_deploy_bounds_patch",
            repo_root / ".omx/research/ddm_hb2_20260808/0001-Fix-HPAC-self-compress-deploy-bounds.patch",
        ),
    )
    bindings = [bind_existing_file(request) for request in requests]
    terminal_trigger = "ps135 terminal safe-run receipt lands and MAIN reseals exact content hashes"
    for role in (
        "terminal_archive",
        "terminal_renderer",
        "terminal_carrier",
        "terminal_coefficients",
        "terminal_probability_object",
        "terminal_convergence_receipt",
        "terminal_sensitivity_map",
    ):
        bindings.append(unresolved_terminal_binding(role, resolution_trigger=terminal_trigger))
    payload = {
        "schema_version": "hr1_content_binding_manifest.v1",
        "execution_allowed": False,
        "bindings": [binding.to_dict() for binding in bindings],
    }
    payload["manifest_sha256"] = _sha256_bytes(_canonical_json(payload))
    return payload


@dataclass(frozen=True, slots=True)
class TensorShapeSpec:
    role: str
    shape: tuple[int, ...]
    dtype: TensorDType
    persistence: str

    def __post_init__(self) -> None:
        if not self.role.strip() or not self.persistence.strip():
            raise Hr1PrestageError("tensor shape role/persistence is required")
        if not self.shape or any(type(dim) is not int or dim <= 0 for dim in self.shape):
            raise Hr1PrestageError("tensor shape dimensions must be positive exact integers")

    @property
    def storage_lower_bound_bytes(self) -> int:
        values = 1
        for dim in self.shape:
            values *= dim
        bits = values * self.dtype.bits_per_value
        return (bits + 7) // 8


@dataclass(frozen=True, slots=True)
class MemoryConfiguration:
    arm: Hr1Arm
    pair_chunk: int
    verdict_batch: int
    tensors: tuple[TensorShapeSpec, ...]
    unresolved_shape_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.pair_chunk) is not int or not 1 <= self.pair_chunk <= 120:
            raise Hr1PrestageError("pair_chunk must be an exact integer in [1,120]")
        if type(self.verdict_batch) is not int or not 1 <= self.verdict_batch <= 120:
            raise Hr1PrestageError("verdict_batch must be an exact integer in [1,120]")
        if not self.tensors:
            raise Hr1PrestageError("memory configuration must declare at least one tensor shape")

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "pair_chunk": self.pair_chunk,
            "verdict_batch": self.verdict_batch,
            "tensors": [
                {
                    "role": tensor.role,
                    "shape": list(tensor.shape),
                    "dtype": tensor.dtype.value,
                    "persistence": tensor.persistence,
                    "storage_lower_bound_bytes": tensor.storage_lower_bound_bytes,
                }
                for tensor in self.tensors
            ],
            "unresolved_shape_roles": list(self.unresolved_shape_roles),
        }

    @property
    def config_sha256(self) -> str:
        return _sha256_bytes(_canonical_json(self.to_dict()))


@dataclass(frozen=True, slots=True)
class MemoryProbeReceipt:
    config_sha256: str
    measured_peak_bytes: int
    measured_at_utc: str
    measurement_kind: str
    command: str
    receipt_path: str
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class MemoryCompileDecision:
    arm: Hr1Arm
    disposition: MemoryDisposition
    config_sha256: str
    tensor_storage_lower_bound_bytes: int
    measured_peak_bytes: int | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "disposition": self.disposition.value,
            "config_sha256": self.config_sha256,
            "tensor_storage_lower_bound_bytes": self.tensor_storage_lower_bound_bytes,
            "measured_peak_bytes": self.measured_peak_bytes,
            "reasons": list(self.reasons),
        }


def make_shape_only_memory_configuration(arm: Hr1Arm) -> MemoryConfiguration:
    common = (
        TensorShapeSpec("work_rgb_pair_chunk", (1, 2, 3, 384, 512), TensorDType.FLOAT32, "ephemeral"),
        TensorShapeSpec("camera_rgb_pair_chunk", (1, 2, 3, 874, 1164), TensorDType.FLOAT32, "ephemeral"),
        TensorShapeSpec("retained_camera_uint8_pair", (1, 2, 874, 1164, 3), TensorDType.UINT8, "retained"),
    )
    if arm is Hr1Arm.FROZEN_DECODE:
        extra = (TensorShapeSpec("hard_tokens_n600", (600, 384, 512), TensorDType.UINT8, "retained"),)
        unresolved = ("terminal_renderer_state", "terminal_probability_object")
    elif arm is Hr1Arm.FULL_RENDERER_FINETUNE:
        extra = (TensorShapeSpec("hard_tokens_n600", (600, 384, 512), TensorDType.UINT8, "retained"),)
        unresolved = ("terminal_renderer_parameters", "optimizer_moments", "ema_shadow")
    elif arm is Hr1Arm.LOW_RANK_ADAPTER:
        extra = (TensorShapeSpec("hard_tokens_n600", (600, 384, 512), TensorDType.UINT8, "retained"),)
        unresolved = ("film_adapter_factors_by_rank", "rgb_head_adapter_factors_by_rank", "optimizer_moments", "ema_shadow")
    elif arm is Hr1Arm.JOINT_TOKEN_RENDERER:
        extra = (
            TensorShapeSpec("hard_tokens_n600", (600, 384, 512), TensorDType.UINT8, "retained"),
            TensorShapeSpec("pair_local_token_logits", (1, 384, 512, 5), TensorDType.FLOAT32, "ephemeral"),
        )
        unresolved = ("terminal_renderer_parameters", "sparse_token_proposal_state", "optimizer_moments", "ema_shadow")
    else:  # pragma: no cover - StrEnum exhaustiveness guard
        raise Hr1PrestageError(f"unknown HR1 arm: {arm}")
    return MemoryConfiguration(
        arm=arm,
        pair_chunk=1,
        verdict_batch=32,
        tensors=common + extra,
        unresolved_shape_roles=unresolved,
    )


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Hr1PrestageError(f"invalid UTC timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise Hr1PrestageError("memory receipt timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def compile_memory_configuration(
    config: MemoryConfiguration,
    receipt: MemoryProbeReceipt | None = None,
    *,
    now_utc: datetime | None = None,
) -> MemoryCompileDecision:
    lower_bound = sum(tensor.storage_lower_bound_bytes for tensor in config.tensors)
    reasons: list[str] = []
    measured_peak: int | None = None
    if config.unresolved_shape_roles:
        reasons.append("UNRESOLVED_EXACT_SHAPES:" + ",".join(config.unresolved_shape_roles))
    if receipt is None:
        reasons.append("FRESH_REAL_CONFIG_MEMORY_PROBE_REQUIRED")
    else:
        if receipt.config_sha256 != config.config_sha256:
            reasons.append("MEMORY_PROBE_CONFIG_HASH_MISMATCH")
        if receipt.measurement_kind != "real_config":
            reasons.append("MEMORY_PROBE_NOT_REAL_CONFIG")
        if type(receipt.measured_peak_bytes) is not int or receipt.measured_peak_bytes <= 0:
            reasons.append("MEMORY_PROBE_PEAK_INVALID")
        else:
            measured_peak = receipt.measured_peak_bytes
        if not receipt.command.strip():
            reasons.append("MEMORY_PROBE_COMMAND_MISSING")
        try:
            receipt_time = _parse_utc(receipt.measured_at_utc)
            now = (now_utc or datetime.now(UTC)).astimezone(UTC)
            age = (now - receipt_time).total_seconds()
            if age < 0 or age > MEMORY_RECEIPT_MAX_AGE_SECONDS:
                reasons.append("MEMORY_PROBE_STALE")
        except Hr1PrestageError:
            reasons.append("MEMORY_PROBE_TIMESTAMP_INVALID")
        receipt_path = Path(receipt.receipt_path)
        try:
            digest, _ = stream_sha256(receipt_path)
            if digest != receipt.receipt_sha256:
                reasons.append("MEMORY_PROBE_RECEIPT_HASH_MISMATCH")
        except Hr1PrestageError:
            reasons.append("MEMORY_PROBE_RECEIPT_UNAVAILABLE")
    return MemoryCompileDecision(
        arm=config.arm,
        disposition=MemoryDisposition.REFUSE if reasons else MemoryDisposition.PASS,
        config_sha256=config.config_sha256,
        tensor_storage_lower_bound_bytes=lower_bound,
        measured_peak_bytes=measured_peak,
        reasons=tuple(reasons),
    )


def payload_manifest_for_tree(
    root: Path,
    *,
    exclude_names: Iterable[str] = (),
) -> dict[str, Any]:
    """Hash every retained regular file under ``root`` except explicit manifest names."""
    root = Path(root).resolve()
    excluded = set(exclude_names)
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        digest, size = stream_sha256(path)
        records.append(
            {
                "relative_path": str(path.relative_to(root)),
                "bytes": size,
                "sha256": digest,
            }
        )
    payload = {
        "schema_version": "hr1_retained_payload_tree.v1",
        "root": str(root),
        "records": records,
        "total_bytes": sum(record["bytes"] for record in records),
    }
    payload["records_sha256"] = _sha256_bytes(_canonical_json(records))
    return payload


__all__ = [
    "BindingState",
    "CheckpointManifest",
    "CompiledHr1Program",
    "FileBindingRequest",
    "Hr1Arm",
    "Hr1ArmProgram",
    "Hr1PrestageError",
    "Hr1RoundTripSpec",
    "MemoryCompileDecision",
    "MemoryConfiguration",
    "MemoryDisposition",
    "MemoryProbeReceipt",
    "ObjectBinding",
    "PayloadRecord",
    "ResumeManifest",
    "SameParentFreshnessReceipt",
    "SameParentObjectKind",
    "TensorDType",
    "TensorShapeSpec",
    "assert_same_parent_freshness",
    "atomic_write_json",
    "bind_existing_file",
    "build_hr1_binding_manifest",
    "compile_memory_configuration",
    "make_four_arm_race_programs",
    "make_frozen_decode_program",
    "make_full_renderer_finetune_program",
    "make_joint_token_renderer_program",
    "make_low_rank_adapter_program",
    "make_shape_only_memory_configuration",
    "payload_manifest_for_tree",
    "stream_sha256",
    "unresolved_terminal_binding",
]
