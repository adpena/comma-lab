"""Planning-only bridge from CodecOp-style objects to Joint-ADMM streams.

The canonical ``CodecOp`` contract is encode/decode over a tensor state_dict.
The Joint-ADMM coordinator consumes ``StreamProximalCodec`` objects that expose
one proximal byte/score operating point. This module connects those surfaces
without pretending a CodecOp search trial is a contest-ready archive:

* encode/decode is run once and cached;
* every input tensor must full-decode with the same shape;
* tensor contracts and blob custody are recorded deterministically;
* all emitted rows remain score_claim=false, dispatchable=false, and
  ready_for_exact_eval_dispatch=false until a byte-closed archive and exact
  CUDA auth eval exist.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from tac.joint_admm_coordinator import ProximalStepResult, StreamProximalCodec

PLANNING_ROW_SCHEMA = "codec_op_admm_adapter_planning_row_v1"
EVIDENCE_GRADE = "planning"
EVIDENCE_SEMANTICS = "cpu_codec_op_admm_bridge_planning_only"
TARGET_MODES = ("contest_exact_eval_planning",)
DEPLOYMENT_TARGET = "desktop_research"
DEFAULT_DISPATCH_BLOCKERS = (
    "codec_op_admm_adapter_is_planning_only",
    "no_byte_closed_archive_manifest",
    "no_archive_substitution_performed",
    "no_score_affecting_payload_change_proof",
    "missing_exact_cuda_auth_eval",
)
DEFAULT_MATERIALIZED_PAYLOAD_CONTRACT = "raw_codecop_encode_blob"


class CodecOpADMMAdapterError(ValueError):
    """Raised when a CodecOp cannot be adapted fail-closed."""


@dataclass(frozen=True)
class DecodeValidation:
    """Full-decode/shape validation report for a CodecOp trial."""

    expected_tensor_count: int
    matched_tensor_count: int
    missing_tensor_keys: list[str] = field(default_factory=list)
    non_tensor_decoded_keys: list[str] = field(default_factory=list)
    shape_mismatch_tensor_keys: list[str] = field(default_factory=list)
    dtype_mismatch_tensor_keys: list[str] = field(default_factory=list)
    decoded_tensor_keys: list[str] = field(default_factory=list)
    matched_tensor_keys: list[str] = field(default_factory=list)
    extra_decoded_tensor_keys: list[str] = field(default_factory=list)
    reconstruction_rms: float | None = None

    @property
    def full_decode_ok(self) -> bool:
        return (
            self.expected_tensor_count > 0
            and self.matched_tensor_count == self.expected_tensor_count
            and not self.missing_tensor_keys
            and not self.non_tensor_decoded_keys
            and not self.shape_mismatch_tensor_keys
            and not self.dtype_mismatch_tensor_keys
        )

    @property
    def coverage_status(self) -> str:
        return "full" if self.full_decode_ok else "failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_tensor_count": self.expected_tensor_count,
            "matched_tensor_count": self.matched_tensor_count,
            "missing_tensor_keys": list(self.missing_tensor_keys),
            "non_tensor_decoded_keys": list(self.non_tensor_decoded_keys),
            "shape_mismatch_tensor_keys": list(self.shape_mismatch_tensor_keys),
            "dtype_mismatch_tensor_keys": list(self.dtype_mismatch_tensor_keys),
            "decoded_tensor_keys": list(self.decoded_tensor_keys),
            "matched_tensor_keys": list(self.matched_tensor_keys),
            "extra_decoded_tensor_keys": list(self.extra_decoded_tensor_keys),
            "decode_coverage_status": self.coverage_status,
            "reconstruction_rms": self.reconstruction_rms,
        }


@dataclass(frozen=True)
class MaterializedPayloadCustody:
    """Custody metadata for a real payload file tied to the CodecOp blob."""

    path: str
    bytes: int
    sha256: str
    contract: str


@dataclass(frozen=True)
class CodecOpADMMSnapshot:
    """Cached encode/decode custody for one CodecOp operating point."""

    stream_name: str
    op_module: str
    op_class: str
    op_name: str
    op_params: dict[str, Any]
    source_label: str | None
    context_keys: list[str]
    state_dict_tensor_bytes: int
    tensor_contract: list[dict[str, Any]]
    tensor_contract_sha256: str
    bytes_in: int
    bytes_out: int
    blob_sha256: str
    op_state_sha256: str
    op_state_keys: list[str]
    decode_validation: DecodeValidation
    score_delta: float
    marginal: float
    materialized_payload: MaterializedPayloadCustody | None = None
    validate_passed: bool | None = None
    validate_findings: list[str] = field(default_factory=list)

    def to_planning_row(self) -> dict[str, Any]:
        blockers = list(DEFAULT_DISPATCH_BLOCKERS)
        if not self.decode_validation.full_decode_ok:
            blockers.append("codec_op_full_decode_shape_validation_failed")
        row = {
            "schema": PLANNING_ROW_SCHEMA,
            "family": "joint_admm_codec_op_adapter",
            "family_group": f"joint_admm_codec_op_adapter:{self.op_class}",
            "stream_name": self.stream_name,
            "cathedral_op": f"{self.op_module}.{self.op_class}",
            "op_module": self.op_module,
            "op_class": self.op_class,
            "op_name": self.op_name,
            "op_params": _jsonable(self.op_params),
            "source_label": self.source_label,
            "context_keys": list(self.context_keys),
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "byte_delta": self.bytes_out - self.bytes_in,
            "state_dict_tensor_bytes": self.state_dict_tensor_bytes,
            "tensor_contract": list(self.tensor_contract),
            "tensor_contract_sha256": self.tensor_contract_sha256,
            "blob_sha256": self.blob_sha256,
            "op_state_sha256": self.op_state_sha256,
            "op_state_keys": list(self.op_state_keys),
            "score_delta": self.score_delta,
            "marginal": self.marginal,
            "decode_coverage_required": True,
            "decode_coverage_status": self.decode_validation.coverage_status,
            "decode_validation": self.decode_validation.to_dict(),
            "codec_op_validate_passed": self.validate_passed,
            "codec_op_validate_findings": list(self.validate_findings),
            "evidence_grade": EVIDENCE_GRADE,
            "evidence_semantics": EVIDENCE_SEMANTICS,
            "target_modes": list(TARGET_MODES),
            "deployment_target": DEPLOYMENT_TARGET,
            "score_claim": False,
            "dispatchable": False,
            "ready_for_exact_eval_dispatch": False,
            "field_selection_ready_for_exact_eval_dispatch": False,
            "promotion_eligible": False,
            "dispatch_attempted": False,
            "score_affecting_payload_changed": False,
            "charged_bits_changed": False,
            "exact_cuda_auth_eval": False,
            "cuda_auth_eval_artifact": None,
            "archive_path": None,
            "archive_sha256": None,
            "archive_bytes": None,
            "dispatch_blockers": blockers,
            "planning_objectives": {
                "bytes_out": self.bytes_out,
                "reconstruction_rms": self.decode_validation.reconstruction_rms,
                "matched_tensor_count": self.decode_validation.matched_tensor_count,
                "expected_tensor_count": self.decode_validation.expected_tensor_count,
                "decode_coverage_status": self.decode_validation.coverage_status,
            },
            "notes": (
                "CodecOp was adapted into a Joint-ADMM StreamProximalCodec "
                "operating point for planning only; no archive substitution, "
                "score claim, or dispatch readiness is implied."
            ),
        }
        if self.materialized_payload is not None:
            row.update(
                {
                    "materialized_payload_path": self.materialized_payload.path,
                    "materialized_payload_bytes": self.materialized_payload.bytes,
                    "materialized_payload_sha256": self.materialized_payload.sha256,
                    "materialized_payload_contract": self.materialized_payload.contract,
                }
            )
        return row


class CodecOpADMMAdapter:
    """Expose one CodecOp encode/decode operating point as a proximal stream.

    ``score_delta`` and ``marginal`` are caller-supplied cached planning
    quantities. The adapter never loads scorers and never marks a row
    dispatchable; ADMM can consume the point for allocation planning only.
    """

    def __init__(
        self,
        op: Any,
        state_dict: Mapping[str, torch.Tensor],
        *,
        score_delta: float,
        marginal: float,
        context: Mapping[str, Any] | None = None,
        op_params: Mapping[str, Any] | None = None,
        stream_name: str | None = None,
        source_label: str | None = None,
        op_module: str | None = None,
        op_class: str | None = None,
        require_validate: bool = True,
        materialized_payload_path: str | Path | None = None,
        materialized_payload_contract: str | None = None,
    ) -> None:
        self._op = _require_codec_op_like(op)
        self._state_dict = _normalise_state_dict(state_dict)
        self._context = dict(context or {})
        self._op_params = dict(op_params or {})
        self._score_delta = _finite_float(score_delta, "score_delta")
        self._marginal = _nonnegative_finite_float(marginal, "marginal")
        self._op_module = op_module or type(op).__module__
        self._op_class = op_class or type(op).__qualname__
        self._op_name = str(getattr(op, "name", self._op_class))
        self._stream_name = stream_name or f"codec_op:{self._op_name}"
        self._source_label = source_label
        self._require_validate = bool(require_validate)
        self._materialized_payload_path = (
            Path(materialized_payload_path)
            if materialized_payload_path is not None
            else None
        )
        self._materialized_payload_contract = _normalise_materialized_payload_contract(
            materialized_payload_contract,
            has_path=self._materialized_payload_path is not None,
        )
        self._snapshot: CodecOpADMMSnapshot | None = None

    @property
    def name(self) -> str:
        return self._stream_name

    def proximal_step(self, target_bytes: float, dual: float) -> ProximalStepResult:
        target = _nonnegative_finite_float(target_bytes, "target_bytes")
        dual_value = _finite_float(dual, "dual")
        snapshot = self._evaluate_once()
        row = snapshot.to_planning_row()
        row["admm_query"] = {
            "target_bytes": target,
            "dual": dual_value,
            "fixed_codec_op_bytes_out": snapshot.bytes_out,
            "target_bytes_exceeded": snapshot.bytes_out > target,
        }
        if snapshot.bytes_out > target:
            row["dispatch_blockers"] = [
                *row["dispatch_blockers"],
                "admm_target_bytes_exceeded_by_fixed_codec_op",
            ]
        return ProximalStepResult(
            encoded_bytes=int(snapshot.bytes_out),
            score_delta=float(snapshot.score_delta),
            marginal=float(snapshot.marginal),
            state=row,
        )

    def to_planning_row(self) -> dict[str, Any]:
        """Return a deterministic, non-dispatchable planning row."""
        return self._evaluate_once().to_planning_row()

    def _evaluate_once(self) -> CodecOpADMMSnapshot:
        if self._snapshot is not None:
            return self._snapshot

        validate_passed: bool | None = None
        validate_findings: list[str] = []
        validator = getattr(self._op, "validate", None)
        if self._require_validate and callable(validator):
            report = validator(self._state_dict, context=self._context)
            validate_passed = bool(getattr(report, "passed", False))
            raw_findings = getattr(report, "findings", [])
            validate_findings = [str(item) for item in raw_findings]
            if not validate_passed:
                raise CodecOpADMMAdapterError(
                    f"CodecOp validate() failed for {self._op_name}: {validate_findings}"
                )

        result = self._op.encode(self._state_dict, context=self._context)
        blob = _require_bytes(getattr(result, "blob", None), "CodecOp encode().blob")
        bytes_out = int(getattr(result, "bytes_out", -1))
        if bytes_out < 0:
            raise CodecOpADMMAdapterError(f"CodecOp returned negative bytes_out={bytes_out}")
        if bytes_out != len(blob):
            raise CodecOpADMMAdapterError(
                f"CodecOp bytes_out={bytes_out} does not match actual blob length {len(blob)}"
            )
        bytes_in = int(getattr(result, "bytes_in", _state_dict_tensor_bytes(self._state_dict)))
        if bytes_in < 0:
            raise CodecOpADMMAdapterError(f"CodecOp returned negative bytes_in={bytes_in}")
        op_state = getattr(result, "op_state", {})
        if op_state is None:
            op_state = {}
        if not isinstance(op_state, Mapping):
            raise CodecOpADMMAdapterError(
                f"CodecOp op_state must be a Mapping or None; got {type(op_state).__name__}"
            )
        decoded = self._op.decode(blob, op_state=dict(op_state), context=self._context)
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        if not isinstance(decoded, Mapping):
            raise CodecOpADMMAdapterError(
                f"CodecOp decode returned {type(decoded).__name__}, expected Mapping"
            )

        validation = validate_full_decode_shapes(self._state_dict, decoded)
        if not validation.full_decode_ok:
            raise CodecOpADMMAdapterError(
                "CodecOpADMMAdapter requires full decode/shape validation; "
                f"status={validation.to_dict()}"
            )

        tensor_contract = build_codec_op_tensor_contract(self._state_dict)
        materialized_payload = _materialized_payload_custody(
            self._materialized_payload_path,
            expected_blob=blob,
            payload_contract=self._materialized_payload_contract,
        )
        snapshot = CodecOpADMMSnapshot(
            stream_name=self._stream_name,
            op_module=self._op_module,
            op_class=self._op_class,
            op_name=self._op_name,
            op_params=_jsonable(self._op_params),
            source_label=self._source_label,
            context_keys=sorted(str(k) for k in self._context),
            state_dict_tensor_bytes=_state_dict_tensor_bytes(self._state_dict),
            tensor_contract=tensor_contract,
            tensor_contract_sha256=_sha256_json(tensor_contract),
            bytes_in=bytes_in,
            bytes_out=bytes_out,
            blob_sha256=_sha256_bytes(blob),
            op_state_sha256=_sha256_json(op_state),
            op_state_keys=sorted(str(k) for k in op_state),
            decode_validation=validation,
            score_delta=self._score_delta,
            marginal=self._marginal,
            materialized_payload=materialized_payload,
            validate_passed=validate_passed,
            validate_findings=validate_findings,
        )
        self._snapshot = snapshot
        return snapshot


def adapt_codec_op_class(
    op_cls: type[Any],
    state_dict: Mapping[str, torch.Tensor],
    *,
    score_delta: float,
    marginal: float,
    op_params: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    stream_name: str | None = None,
    source_label: str | None = None,
    require_validate: bool = True,
    materialized_payload_path: str | Path | None = None,
    materialized_payload_contract: str | None = None,
) -> CodecOpADMMAdapter:
    """Instantiate ``op_cls`` and return a ``StreamProximalCodec`` adapter."""
    params = dict(op_params or {})
    op = op_cls(**params)
    return CodecOpADMMAdapter(
        op,
        state_dict,
        score_delta=score_delta,
        marginal=marginal,
        context=context,
        op_params=params,
        stream_name=stream_name,
        source_label=source_label,
        op_module=op_cls.__module__,
        op_class=op_cls.__qualname__,
        require_validate=require_validate,
        materialized_payload_path=materialized_payload_path,
        materialized_payload_contract=materialized_payload_contract,
    )


def codec_op_to_admm_planning_row(
    op: Any,
    state_dict: Mapping[str, torch.Tensor],
    *,
    score_delta: float,
    marginal: float,
    context: Mapping[str, Any] | None = None,
    op_params: Mapping[str, Any] | None = None,
    stream_name: str | None = None,
    source_label: str | None = None,
    require_validate: bool = True,
    materialized_payload_path: str | Path | None = None,
    materialized_payload_contract: str | None = None,
) -> dict[str, Any]:
    """Evaluate a CodecOp-like instance and emit a fail-closed planning row."""
    return CodecOpADMMAdapter(
        op,
        state_dict,
        score_delta=score_delta,
        marginal=marginal,
        context=context,
        op_params=op_params,
        stream_name=stream_name,
        source_label=source_label,
        require_validate=require_validate,
        materialized_payload_path=materialized_payload_path,
        materialized_payload_contract=materialized_payload_contract,
    ).to_planning_row()


def build_codec_op_tensor_contract(
    state_dict: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    """Build a deterministic tensor contract for CodecOp custody rows."""
    normalised = _normalise_state_dict(state_dict)
    contract: list[dict[str, Any]] = []
    for key, tensor in normalised.items():
        contract.append(
            {
                "key": key,
                "shape": [int(dim) for dim in tensor.shape],
                "dtype": str(tensor.dtype),
                "numel": int(tensor.numel()),
                "element_size": int(tensor.element_size()),
                "nbytes": int(tensor.numel() * tensor.element_size()),
                "sha256": _sha256_tensor(tensor),
            }
        )
    return contract


def validate_full_decode_shapes(
    expected_state_dict: Mapping[str, torch.Tensor],
    decoded: Mapping[Any, Any],
) -> DecodeValidation:
    """Require every input tensor key to decode as a tensor with the same shape."""
    expected = _normalise_state_dict(expected_state_dict)
    expected_keys = sorted(expected)
    decoded_tensor_keys = sorted(
        str(k) for k, value in decoded.items() if isinstance(k, str) and isinstance(value, torch.Tensor)
    )
    expected_key_set = set(expected_keys)
    missing: list[str] = []
    non_tensor: list[str] = []
    shape_mismatch: list[str] = []
    dtype_mismatch: list[str] = []
    matched: list[str] = []
    mse_values: list[float] = []

    for key in expected_keys:
        original = expected[key]
        if key not in decoded:
            missing.append(key)
            continue
        recon = decoded[key]
        if not isinstance(recon, torch.Tensor):
            non_tensor.append(key)
            continue
        recon_cpu = recon.detach().cpu().contiguous()
        if tuple(int(dim) for dim in recon_cpu.shape) != tuple(int(dim) for dim in original.shape):
            shape_mismatch.append(key)
            continue
        if recon_cpu.dtype != original.dtype:
            dtype_mismatch.append(key)
        matched.append(key)
        if original.numel() == 0:
            mse_values.append(0.0)
            continue
        if original.is_complex() or recon_cpu.is_complex():
            diff_abs = (recon_cpu.to(torch.complex64) - original.to(torch.complex64)).abs()
            mse_values.append(float((diff_abs * diff_abs).mean().item()))
        else:
            diff = recon_cpu.to(torch.float64) - original.to(torch.float64)
            mse_values.append(float((diff * diff).mean().item()))

    rms = math.sqrt(sum(mse_values) / len(mse_values)) if mse_values else None
    return DecodeValidation(
        expected_tensor_count=len(expected_keys),
        matched_tensor_count=len(matched),
        missing_tensor_keys=missing,
        non_tensor_decoded_keys=non_tensor,
        shape_mismatch_tensor_keys=shape_mismatch,
        dtype_mismatch_tensor_keys=dtype_mismatch,
        decoded_tensor_keys=decoded_tensor_keys,
        matched_tensor_keys=matched,
        extra_decoded_tensor_keys=[key for key in decoded_tensor_keys if key not in expected_key_set],
        reconstruction_rms=rms,
    )


def _require_codec_op_like(op: Any) -> Any:
    for attr in ("encode", "decode"):
        if not callable(getattr(op, attr, None)):
            raise CodecOpADMMAdapterError(
                f"CodecOp-like object must expose callable {attr}(); got {op!r}"
            )
    return op


def _normalise_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise CodecOpADMMAdapterError("state_dict must be a non-empty Mapping[str, torch.Tensor]")
    out: dict[str, torch.Tensor] = {}
    for key in sorted(state_dict):
        if not isinstance(key, str):
            raise CodecOpADMMAdapterError(f"state_dict key {key!r} is not a str")
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise CodecOpADMMAdapterError(
                f"state_dict[{key!r}] is {type(tensor).__name__}, expected torch.Tensor"
            )
        cpu = tensor.detach().cpu().clone().contiguous()
        if cpu.dtype.is_floating_point and not torch.isfinite(cpu).all():
            raise CodecOpADMMAdapterError(f"state_dict[{key!r}] contains non-finite values")
        if cpu.is_complex() and not torch.isfinite(torch.view_as_real(cpu)).all():
            raise CodecOpADMMAdapterError(f"state_dict[{key!r}] contains non-finite complex values")
        out[key] = cpu
    return out


def _state_dict_tensor_bytes(state_dict: Mapping[str, torch.Tensor]) -> int:
    return sum(int(t.numel() * t.element_size()) for t in state_dict.values())


def _require_bytes(value: Any, label: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    raise CodecOpADMMAdapterError(f"{label} must be bytes; got {type(value).__name__}")


def _normalise_materialized_payload_contract(
    payload_contract: str | None,
    *,
    has_path: bool,
) -> str | None:
    if payload_contract is None:
        return DEFAULT_MATERIALIZED_PAYLOAD_CONTRACT if has_path else None
    normalised = payload_contract.strip()
    if not normalised:
        return DEFAULT_MATERIALIZED_PAYLOAD_CONTRACT if has_path else None
    if not has_path:
        raise CodecOpADMMAdapterError(
            "materialized_payload_contract requires materialized_payload_path"
        )
    return normalised


def _materialized_payload_custody(
    path: Path | None,
    *,
    expected_blob: bytes,
    payload_contract: str | None,
) -> MaterializedPayloadCustody | None:
    if path is None:
        return None
    if not path.is_file():
        raise CodecOpADMMAdapterError(
            f"materialized_payload_path must be an existing file: {path}"
        )
    payload = path.read_bytes()
    if len(payload) != len(expected_blob):
        raise CodecOpADMMAdapterError(
            "materialized_payload_path bytes do not match CodecOp blob: "
            f"path_bytes={len(payload)} blob_bytes={len(expected_blob)}"
        )
    payload_sha256 = _sha256_bytes(payload)
    blob_sha256 = _sha256_bytes(expected_blob)
    if payload_sha256 != blob_sha256:
        raise CodecOpADMMAdapterError(
            "materialized_payload_path sha256 does not match CodecOp blob: "
            f"path_sha256={payload_sha256} blob_sha256={blob_sha256}"
        )
    return MaterializedPayloadCustody(
        path=path.as_posix(),
        bytes=len(payload),
        sha256=payload_sha256,
        contract=payload_contract or DEFAULT_MATERIALIZED_PAYLOAD_CONTRACT,
    )


def _finite_float(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise CodecOpADMMAdapterError(f"{label} must be finite; got {value!r}")
    return number


def _nonnegative_finite_float(value: float, label: str) -> float:
    number = _finite_float(value, label)
    if number < 0.0:
        raise CodecOpADMMAdapterError(f"{label} must be >= 0; got {value!r}")
    return number


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            _jsonable(value),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _sha256_tensor(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    try:
        payload = cpu.numpy().tobytes(order="C")
    except TypeError:
        payload = cpu.view(torch.uint8).numpy().tobytes(order="C")
    return _sha256_bytes(payload)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CodecOpADMMAdapterError(f"non-finite float is not JSON custody-safe: {value!r}")
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for raw_key in sorted(value, key=lambda item: str(item)):
            key = str(raw_key)
            if key in out:
                raise CodecOpADMMAdapterError(
                    f"mapping has duplicate JSON key after stringification: {key!r}"
                )
            out[key] = _jsonable(value[raw_key])
        return out
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except Exception as exc:
            raise CodecOpADMMAdapterError(
                f"value of type {type(value).__name__} is not JSON custody-safe"
            ) from exc
    raise CodecOpADMMAdapterError(
        f"value of type {type(value).__name__} is not JSON custody-safe"
    )


_assert_protocol: StreamProximalCodec
_assert_protocol = CodecOpADMMAdapter(
    op=type(
        "_ProtocolNoopOp",
        (),
        {
            "name": "protocol_noop",
            "encode": lambda self, state_dict, *, context: type(
                "_R",
                (),
                {"blob": b"x", "bytes_in": 4, "bytes_out": 1, "op_state": {}},
            )(),
            "decode": lambda self, blob, *, op_state, context: {"x": torch.zeros(1)},
        },
    )(),
    state_dict={"x": torch.zeros(1)},
    score_delta=0.0,
    marginal=0.0,
)


__all__ = [
    "CodecOpADMMAdapter",
    "CodecOpADMMAdapterError",
    "CodecOpADMMSnapshot",
    "DecodeValidation",
    "MaterializedPayloadCustody",
    "adapt_codec_op_class",
    "build_codec_op_tensor_contract",
    "codec_op_to_admm_planning_row",
    "validate_full_decode_shapes",
]
