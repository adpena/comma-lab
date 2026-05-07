# ROUNDTRIP_NOT_REQUIRED: orchestrator composes child codecs (each with its own roundtrip test); does not implement encode/decode itself
"""PARADIGM-γ — Joint Codec Stack Pipeline (JCSP) orchestrator.

Composes the canonical codec stack per Grand Council #294:

    representation → prediction → quantization → hyperprior → arithmetic → pack

This module sits at the **hyperprior + arithmetic** layer of the canonical
order. Its job is to wire the Joint-ADMM 4-stream coordinator, the Ballé
hyperprior codec, and the arithmetic terminal into a single bytes-out
pipeline that takes a multi-stream qint corpus (renderer FP4 indices,
SegNet logits qints, pose qints, residual qints) and produces a unified
JCSv1 payload.

Magic byte ``JCSv1`` (Joint Codec Stack Pipeline v1):

::

    magic              : 4 bytes  = b"JCSP"
    version            : 2 bytes  uint16 = 1
    n_streams          : 1 byte   (number of streams composed)
    For each stream:
        name_len       : 1 byte
        name           : <name_len> bytes UTF-8
        codec_kind     : 1 byte  (0=arithmetic_static_AQv1,
                                   1=balle_hyperprior_BHv1,
                                   2=raw_passthrough)
        admm_bytes_target : 4 bytes uint32  (ADMM-projected target)
        actual_bytes      : 4 bytes uint32  (post-codec actual)
        score_delta_milli : 4 bytes  int32  (score-delta × 1e6)
        marginal_milli    : 4 bytes  int32  (dScore/dByte × 1e6)
        payload_len       : 4 bytes  uint32
        payload           : <payload_len> bytes
    kkt_residual_milli : 4 bytes uint32  (waterline KKT residual × 1e6)
    iters              : 4 bytes uint32  (ADMM iterations)
    converged          : 1 byte  (0 or 1)

The unified container lets the pack-time bit-level optimizer work on a
single byte stream while preserving per-stream provenance for the
inflate-side dispatch.

CLAUDE.md compliance
--------------------
* COMPRESS-time only. The orchestrator runs at compress time; inflate-time
  reads each stream's per-codec wire format (BHv1/AQv1/raw).
* Strict-scorer-rule: NO scorer load. The score-delta and marginal values
  carried in the JCSP wire format are CACHED by the caller (frontier-sampler
  or sensitivity-map) — the orchestrator just packs them.
* No silent defaults: every public function arg is required-keyword.
* Encoder verifies decoder roundtrip per-stream on the way out. A
  malformed JCSP container cannot ship.
* Tagged claims: every empirical assertion in tests/measurements carries a
  ``[empirical:reports/paradigm_gamma_*.json]`` tag.

Architecture
------------
::

    ┌───────────────────────────────────────────────────────────────┐
    │ JointCodecStackOrchestrator                                   │
    │                                                               │
    │   streams_in (list[StreamSource])                             │
    │       │                                                       │
    │       ▼                                                       │
    │   [Joint-ADMM coordinator]  ← byte budget B, score surfaces   │
    │       │   per-stream (target_bytes, dual)                     │
    │       ▼                                                       │
    │   [Per-stream codec dispatch]                                 │
    │       ├─ arithmetic_static_AQv1  (homoscedastic, small alpha) │
    │       ├─ balle_hyperprior_BHv1   (heteroscedastic, ≥30KB)     │
    │       │     └─ arithmetic terminal under Ballé σ-prior        │
    │       └─ raw_passthrough         (already optimal e.g. AV1)   │
    │       │                                                       │
    │       ▼                                                       │
    │   [Pack into JCSP container]                                  │
    │       │                                                       │
    │       ▼                                                       │
    │   bytes (JCSP container; ZIP-deterministic at archive layer)  │
    └───────────────────────────────────────────────────────────────┘

References
----------
* Grand Council Paradigm Shift §6 — `:omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md`
* Boyd 2011 — ADMM operational. `:tac/joint_admm_coordinator.py`.
* Ballé 2018 — Variational image compression with scale hyperprior.
  `:tac/balle_hyperprior_codec.py`.
* memory `project_codec_stacking_composition_canonical_orders_20260429.md`.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tac.arithmetic_qint_codec import (
    decode_qints_arithmetic,
    encode_qints_arithmetic,
)
from tac.balle_hyperprior_codec import (
    BalleHyperpriorCodec,
    decode_qints_balle,
    encode_qints_balle_auto,
)
from tac.joint_admm_coordinator import (
    AdmmResult,
    JointADMMConfig,
    ProximalStepResult,
    run_admm,
)
from tac.optimization.research_basis import research_basis_ids_for_family
from tac.submission_archive import (
    DETERMINISTIC_ZIP_DATE_TIME,
    DETERMINISTIC_ZIP_FILE_MODE,
    validate_archive_member_name,
    validate_zip_member_infos,
    write_deterministic_zip_member,
)

_JCSP_MAGIC: bytes = b"JCSP"
_JCSP_VERSION: int = 1
_JCSP_LOCAL_SKELETON_MAGIC: bytes = b"JCSK"
_JCSP_LOCAL_SKELETON_VERSION: int = 1
JCSP_STREAM_METADATA_SCHEMA: str = "jcsp_tensor_stream_specs_manifest_v1"
JCSP_ARCHIVE_MEMBER_NAME: str = "jcsp.bin"
JCSP_ARCHIVE_MEMBER_CONTRACT_SCHEMA: str = "jcsp_archive_member_runtime_contract_v1"
JCSP_LOCAL_SKELETON_SCHEMA: str = "jcsp_local_archive_member_skeleton_v1"
JCSP_LOCAL_ARCHIVE_MEMBER_CONTRACT_SCHEMA: str = (
    "jcsp_local_archive_member_skeleton_contract_v1"
)
JCSP_STREAM_ARCHIVE_BYTE_RECONCILIATION_SCHEMA: str = (
    "jcsp_stream_archive_byte_reconciliation_v1"
)
JCSP_MODEL_STREAM_ARCHIVE_READINESS_SCHEMA: str = (
    "jcsp_model_stream_archive_readiness_v1"
)
JCSP_SUBMISSION_RUNTIME_CONSUMPTION_SCHEMA: str = (
    "jcsp_submission_runtime_consumption_contract_v1"
)
JCSP_SUBMISSION_RUNTIME_BRIDGE_PROBE_SCHEMA: str = (
    "jcsp_submission_runtime_bridge_probe_v1"
)
JCSP_REQUIRED_SUBMISSION_RUNTIME: str = "submissions/robust_current"
JCSP_SUBMISSION_RUNTIME_BRIDGE_PATH: str = (
    "submissions/robust_current/jcsp_runtime_bridge.py"
)
JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER: str = (
    "submissions_robust_current_jcsp_bin_consumption_missing"
)
JCSP_LOCAL_SKELETON_RUNTIME_BLOCKER: str = (
    "jcsp_local_skeleton_not_submission_runtime_container"
)
JCSP_SUBMISSION_RUNTIME_OUTPUT_CONTRACT_SCHEMA: str = (
    "jcsp_submission_runtime_output_contract_v1"
)
JCSP_SUBMISSION_RUNTIME_OUTPUT_PARITY_BLOCKER: str = (
    "jcsp_runtime_raw_output_parity_missing"
)
JCSP_RUNTIME_RAW_OUTPUT_PARITY_CONTRACT_SCHEMA: str = (
    "jcsp_runtime_raw_output_parity_contract_v1"
)
JCSP_RUNTIME_RAW_OUTPUT_PARITY_PROOF_SCHEMA: str = (
    "jcsp_runtime_raw_output_parity_proof_v1"
)
JCSP_RUNTIME_RAW_OUTPUT_CONSUMER_READINESS_SCHEMA: str = (
    "jcsp_runtime_raw_output_consumer_readiness_v1"
)
JCSP_RUNTIME_AQ_RAWVIDEO_ADAPTER_ID: str = (
    "jcsp_arithmetic_uint8_rawvideo_adapter_v1"
)
JCSP_RUNTIME_AQ_RAWVIDEO_STREAM_KIND: str = (
    "arithmetic_static_uint8_rawvideo_v1"
)
JCSP_RUNTIME_REAL_AQ_PREFLIGHT_ADAPTER_ID: str = (
    "jcsp_real_aq_rawvideo_runtime_preflight_adapter_v1"
)

# Codec kind flags
KIND_ARITHMETIC_STATIC: int = 0
KIND_BALLE_HYPERPRIOR: int = 1
KIND_RAW_PASSTHROUGH: int = 2

_ARITHMETIC_PAYLOAD_MAGICS: tuple[bytes, ...] = (b"AQv1", b"AQc1")
_BALLE_PAYLOAD_MAGICS: tuple[bytes, ...] = (b"BHv1",)
_PAYLOAD_MAGICS_BY_CODEC_KIND: dict[int, tuple[bytes, ...]] = {
    KIND_ARITHMETIC_STATIC: _ARITHMETIC_PAYLOAD_MAGICS,
    KIND_BALLE_HYPERPRIOR: _BALLE_PAYLOAD_MAGICS,
}


# ────────────────────────────────────────────────────────────────────────────
# Compress-time model stream decomposition
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JCSPTensorStreamSpec:
    """Compress-time annotation for one model tensor before JCSP encoding.

    This is a dispatch prerequisite, not a score claim and not an encoder. It
    makes the per-tensor stream contract explicit so later γ-JCSP work can turn
    selected tensors into ``StreamSource`` objects only after quantization,
    cached sensitivity/score-marginal evidence, and decode validation exist.
    """

    name: str
    stream_id: str
    decomposition_index: int
    codec_kind: int
    tensor_shape: tuple[int, ...]
    tensor_dtype: str
    num_elements: int
    raw_bytes: int
    byte_estimate: int
    byte_estimate_source: str
    bytes_charged: int
    bytes_charged_source: str
    score_per_byte_marginal: float
    score_marginal_source: str
    score_marginal_evidence: str
    scorer_term_targeted: str
    research_basis_ids: tuple[str, ...] = ()
    constraint_tags: tuple[str, ...] = ()
    dispatch_blockers: tuple[str, ...] = ()
    fail_closed_criteria: tuple[str, ...] = ()


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reject_duplicate_json_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _tensor_shape_tuple(tensor: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in getattr(tensor, "shape", ()))


def _tensor_numel(tensor: Any, shape: tuple[int, ...]) -> int:
    if hasattr(tensor, "numel"):
        return int(tensor.numel())
    n = 1
    for dim in shape:
        n *= int(dim)
    return int(n)


def _tensor_element_size(tensor: Any, dtype_name: str) -> int:
    if hasattr(tensor, "element_size"):
        return int(tensor.element_size())
    if "64" in dtype_name:
        return 8
    if "32" in dtype_name:
        return 4
    if "16" in dtype_name or "bfloat" in dtype_name:
        return 2
    if "bool" in dtype_name or "uint8" in dtype_name or "int8" in dtype_name:
        return 1
    return 4


def _is_floating_dtype(dtype_name: str) -> bool:
    return "float" in dtype_name or "bfloat" in dtype_name


def _validate_jcsp_stream_name(name: str, *, context: str) -> str:
    encoded = name.encode("utf-8")
    if not (1 <= len(encoded) <= 255):
        raise ValueError(
            f"{context} stream name must be 1..255 UTF-8 bytes, got "
            f"{len(encoded)} for {name!r}"
        )
    if "\x00" in name:
        raise ValueError(f"{context} stream name must not contain NUL bytes")
    return name


def _jcsp_stream_name_sort_key(name: str) -> tuple[bytes, str]:
    return (name.encode("utf-8"), name)


def _iter_model_tensors(model: Any, *, include_buffers: bool) -> list[tuple[str, Any]]:
    """Return deterministic ``(name, tensor)`` pairs from a module or mapping."""
    if isinstance(model, Mapping):
        return [(str(name), tensor) for name, tensor in model.items()]
    if hasattr(model, "state_dict"):
        state = model.state_dict()
        return [(str(name), tensor) for name, tensor in state.items()]
    if hasattr(model, "named_parameters"):
        items: list[tuple[str, Any]] = [
            (str(name), tensor) for name, tensor in model.named_parameters()
        ]
        if include_buffers and hasattr(model, "named_buffers"):
            items.extend((str(name), tensor) for name, tensor in model.named_buffers())
        return items
    raise TypeError(
        "model_to_jcsp_streams expects a torch-like module, a state_dict-like "
        "mapping, or an object exposing named_parameters()."
    )


def _sorted_unique_model_tensors(
    model: Any,
    *,
    include_buffers: bool,
) -> list[tuple[str, Any]]:
    items = [
        (
            _validate_jcsp_stream_name(
                name,
                context="model_to_jcsp_streams",
            ),
            tensor,
        )
        for name, tensor in _iter_model_tensors(
            model,
            include_buffers=include_buffers,
        )
    ]
    counts: dict[str, int] = {}
    for name, _tensor in items:
        counts[name] = counts.get(name, 0) + 1
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            "model_to_jcsp_streams requires unique tensor stream names; "
            f"duplicates: {', '.join(duplicates)}"
        )
    return sorted(items, key=lambda item: _jcsp_stream_name_sort_key(item[0]))


def _infer_model_tensor_codec_kind(
    *,
    dtype_name: str,
    raw_bytes: int,
    ndim: int,
    large_tensor_balle_threshold_bytes: int,
) -> int:
    if not _is_floating_dtype(dtype_name):
        return KIND_ARITHMETIC_STATIC
    if raw_bytes >= large_tensor_balle_threshold_bytes and ndim >= 2:
        return KIND_BALLE_HYPERPRIOR
    return KIND_ARITHMETIC_STATIC


def _estimate_model_tensor_stream_bytes(
    *,
    dtype_name: str,
    num_elements: int,
    raw_bytes: int,
) -> tuple[int, str]:
    # Conservative compress-time estimate: floating tensors are assumed to be
    # at least fp16-quantizable, while integer/bool metadata stays raw unless a
    # later exact qint stream proves a smaller charged wire format.
    if (
        _is_floating_dtype(dtype_name)
        and "16" not in dtype_name
        and "bfloat" not in dtype_name
    ):
        return max(0, int(num_elements) * 2), "fp16_quantization_floor_estimate"
    return int(raw_bytes), "raw_tensor_bytes_estimate"


def _match_name_set(name: str, candidates: Iterable[str] | None) -> bool:
    if candidates is None:
        return False
    for candidate in candidates:
        text = str(candidate)
        if name == text or (
            name.startswith(text.rstrip("*")) and text.endswith("*")
        ):
            return True
    return False


def _matched_wet_stream_names(
    *,
    tensor_names: set[str],
    wet_streams: Iterable[str] | None,
) -> set[str]:
    if wet_streams is None:
        return set()
    patterns = tuple(str(candidate) for candidate in wet_streams)
    matched = {
        name
        for name in tensor_names
        if _match_name_set(name, patterns)
    }
    unmatched = [
        pattern
        for pattern in patterns
        if not any(_match_name_set(name, (pattern,)) for name in tensor_names)
    ]
    if unmatched:
        raise ValueError(
            "model_to_jcsp_streams got wet_streams patterns that matched no "
            "streams: " + ", ".join(sorted(unmatched))
        )
    return matched


def _require_finite(value: float, *, field: str, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field} for stream {name!r} must be finite")
    return out


def _score_annotation_for_stream(
    *,
    name: str,
    score_marginals: Mapping[str, Any] | None,
    default_score_per_byte_marginal: float,
) -> tuple[float, str, str, str, tuple[str, ...]]:
    raw = score_marginals.get(name) if score_marginals is not None else None
    if raw is None:
        return (
            _require_finite(
                default_score_per_byte_marginal,
                field="default_score_per_byte_marginal",
                name=name,
            ),
            "missing_score_marginal",
            "prediction",
            "joint",
            ("missing_score_marginal",),
        )
    if isinstance(raw, (int, float)):
        return (
            _require_finite(raw, field="score_per_byte_marginal", name=name),
            "caller_exact_name",
            "empirical",
            "joint",
            (),
        )
    if isinstance(raw, Mapping):
        value = raw.get("score_per_byte_marginal", raw.get("marginal", None))
        if value is None:
            raise ValueError(
                f"score_marginals[{name!r}] must define "
                "'score_per_byte_marginal' or 'marginal'"
            )
        marginal = _require_finite(
            value,
            field="score_per_byte_marginal",
            name=name,
        )
        tags_raw = raw.get("constraint_tags", ())
        return (
            marginal,
            str(raw.get("source", "caller_exact_name")),
            str(raw.get("evidence_grade", "empirical")),
            str(raw.get("scorer_term_targeted", "joint")),
            tuple(str(tag) for tag in tags_raw),
        )
    raise TypeError(
        f"score_marginals[{name!r}] must be a float or mapping, got "
        f"{type(raw).__name__}"
    )


def _stream_identity_sha256(
    *,
    name: str,
    dtype_name: str,
    shape: tuple[int, ...],
    num_elements: int,
    raw_bytes: int,
) -> str:
    payload = {
        "name": name,
        "tensor_dtype": dtype_name,
        "tensor_shape": list(shape),
        "num_elements": int(num_elements),
        "raw_bytes": int(raw_bytes),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def model_to_jcsp_streams(
    model: Any,
    *,
    score_marginals: Mapping[str, Any] | None = None,
    codec_overrides: Mapping[str, int] | None = None,
    wet_streams: Iterable[str] | None = None,
    include_buffers: bool = True,
    default_score_per_byte_marginal: float = 0.0,
    large_tensor_balle_threshold_bytes: int = 30 * 1024,
) -> list[JCSPTensorStreamSpec]:
    """Decompose a model/state_dict into deterministic JCSP tensor streams.

    The helper intentionally does not quantize, encode, load scorers, or infer
    score evidence. Missing score marginals stay explicit blockers. This mirrors
    the additive-distortion discipline used by γ-JCSP: per-stream costs may be
    summed only after each stream has a local cached marginal and a validated
    charged-byte contract.
    """
    if large_tensor_balle_threshold_bytes <= 0:
        raise ValueError("large_tensor_balle_threshold_bytes must be > 0")
    out: list[JCSPTensorStreamSpec] = []
    tensor_items = _sorted_unique_model_tensors(
        model,
        include_buffers=include_buffers,
    )
    tensor_names = {name for name, _tensor in tensor_items}
    score_marginals_by_name: dict[str, Any] | None = None
    if score_marginals is not None:
        score_marginals_by_name = {
            str(name): value for name, value in score_marginals.items()
        }
        unknown_marginals = sorted(
            name for name in score_marginals_by_name if name not in tensor_names
        )
        if unknown_marginals:
            raise ValueError(
                "model_to_jcsp_streams got score_marginals for unknown streams: "
                + ", ".join(unknown_marginals)
            )
    codec_overrides_by_name: dict[str, int] | None = None
    if codec_overrides is not None:
        raw_codec_overrides_by_name = {
            str(name): value for name, value in codec_overrides.items()
        }
        unknown_overrides = sorted(
            name for name in raw_codec_overrides_by_name if name not in tensor_names
        )
        if unknown_overrides:
            raise ValueError(
                "model_to_jcsp_streams got codec_overrides for unknown streams: "
                + ", ".join(unknown_overrides)
            )
        codec_overrides_by_name = {}
        for name, value in raw_codec_overrides_by_name.items():
            try:
                codec_overrides_by_name[name] = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid codec override for {name!r}: {value!r}"
                ) from exc
    wet_stream_names = _matched_wet_stream_names(
        tensor_names=tensor_names,
        wet_streams=wet_streams,
    )
    if wet_stream_names:
        wet_without_raw_override = sorted(
            name
            for name in wet_stream_names
            if (
                codec_overrides_by_name is None
                or name not in codec_overrides_by_name
                or codec_overrides_by_name[name] != KIND_RAW_PASSTHROUGH
            )
        )
        if wet_without_raw_override:
            raise ValueError(
                "wet streams require explicit RAW_PASSTHROUGH codec overrides: "
                + ", ".join(wet_without_raw_override)
            )
    for stream_index, (name, tensor) in enumerate(tensor_items):
        shape = _tensor_shape_tuple(tensor)
        num_elements = _tensor_numel(tensor, shape)
        dtype_name = str(getattr(tensor, "dtype", type(tensor).__name__))
        elem_size = _tensor_element_size(tensor, dtype_name)
        raw_bytes = int(num_elements) * int(elem_size)
        inferred_kind = _infer_model_tensor_codec_kind(
            dtype_name=dtype_name,
            raw_bytes=raw_bytes,
            ndim=len(shape),
            large_tensor_balle_threshold_bytes=large_tensor_balle_threshold_bytes,
        )
        codec_kind = (
            codec_overrides_by_name[name]
            if codec_overrides_by_name is not None and name in codec_overrides_by_name
            else inferred_kind
        )
        if codec_kind not in (
            KIND_ARITHMETIC_STATIC,
            KIND_BALLE_HYPERPRIOR,
            KIND_RAW_PASSTHROUGH,
        ):
            raise ValueError(f"invalid codec override for {name!r}: {codec_kind}")
        byte_estimate, byte_estimate_source = _estimate_model_tensor_stream_bytes(
            dtype_name=dtype_name,
            num_elements=num_elements,
            raw_bytes=raw_bytes,
        )
        if codec_kind == KIND_RAW_PASSTHROUGH:
            byte_estimate = int(raw_bytes)
            byte_estimate_source = "raw_passthrough_tensor_bytes"
        (
            marginal,
            marginal_source,
            marginal_evidence,
            scorer_term,
            score_tags,
        ) = _score_annotation_for_stream(
            name=name,
            score_marginals=score_marginals_by_name,
            default_score_per_byte_marginal=default_score_per_byte_marginal,
        )
        constraint_tags = [
            "additive_score_marginal_required",
            "no_scorer_load_at_stream_decomposition",
            "score_affecting_sidecars_forbidden",
            "single_member_jcsp_archive_required",
            *score_tags,
        ]
        dispatch_blockers = [
            "qint_or_exact_wire_stream_missing",
            "decode_validation_missing",
            "byte_closed_archive_member_missing",
        ]
        if "missing_score_marginal" in constraint_tags:
            dispatch_blockers.append("score_marginal_artifact_missing")
        if name in wet_stream_names:
            constraint_tags = [
                tag for tag in constraint_tags if tag != "missing_score_marginal"
            ]
            dispatch_blockers = [
                blocker
                for blocker in dispatch_blockers
                if blocker != "score_marginal_artifact_missing"
            ]
            constraint_tags.append("wet_stream_do_not_perturb")
            constraint_tags.append("wet_stream_raw_passthrough_required")
            marginal = 0.0
            marginal_source = "wet_stream"
            marginal_evidence = "derivation"
        fail_closed = [
            "refuse_dispatch_if_score_marginal_missing",
            "refuse_dispatch_if_decode_validation_missing",
            "refuse_dispatch_if_charged_bytes_unverified",
            "refuse_dispatch_if_archive_member_not_byte_closed",
            "refuse_dispatch_if_sidecar_dependency_detected",
        ]
        research_basis_ids = tuple(
            dict.fromkeys(
                [
                    *research_basis_ids_for_family("gamma"),
                    *(
                        research_basis_ids_for_family("foveation")
                        if scorer_term in {"seg", "pose", "joint"}
                        else []
                    ),
                ]
            )
        )
        out.append(
            JCSPTensorStreamSpec(
                name=name,
                stream_id=_stream_identity_sha256(
                    name=name,
                    dtype_name=dtype_name,
                    shape=shape,
                    num_elements=num_elements,
                    raw_bytes=raw_bytes,
                ),
                decomposition_index=stream_index,
                codec_kind=codec_kind,
                tensor_shape=shape,
                tensor_dtype=dtype_name,
                num_elements=num_elements,
                raw_bytes=raw_bytes,
                byte_estimate=byte_estimate,
                byte_estimate_source=byte_estimate_source,
                bytes_charged=byte_estimate,
                bytes_charged_source="planning_estimate_not_archive_closed",
                score_per_byte_marginal=float(marginal),
                score_marginal_source=marginal_source,
                score_marginal_evidence=marginal_evidence,
                scorer_term_targeted=scorer_term,
                research_basis_ids=research_basis_ids,
                constraint_tags=tuple(dict.fromkeys(constraint_tags)),
                dispatch_blockers=tuple(dict.fromkeys(dispatch_blockers)),
                fail_closed_criteria=tuple(fail_closed),
            )
        )
    return out


def _stream_spec_record(stream: JCSPTensorStreamSpec) -> dict[str, Any]:
    marginal = _require_finite(
        stream.score_per_byte_marginal,
        field="score_per_byte_marginal",
        name=stream.name,
    )
    return {
        "name": stream.name,
        "stream_id": stream.stream_id,
        "decomposition_index": int(stream.decomposition_index),
        "codec_kind": int(stream.codec_kind),
        "tensor_shape": list(stream.tensor_shape),
        "tensor_dtype": stream.tensor_dtype,
        "num_elements": int(stream.num_elements),
        "raw_bytes": int(stream.raw_bytes),
        "byte_estimate": int(stream.byte_estimate),
        "byte_estimate_source": stream.byte_estimate_source,
        "bytes_charged": int(stream.bytes_charged),
        "bytes_charged_source": stream.bytes_charged_source,
        "score_per_byte_marginal": float(marginal),
        "score_marginal_source": stream.score_marginal_source,
        "score_marginal_evidence": stream.score_marginal_evidence,
        "scorer_term_targeted": stream.scorer_term_targeted,
        "research_basis_ids": list(stream.research_basis_ids),
        "constraint_tags": list(stream.constraint_tags),
        "dispatch_blockers": list(stream.dispatch_blockers),
        "fail_closed_criteria": list(stream.fail_closed_criteria),
    }


def jcsp_stream_specs_manifest(
    streams: Iterable[JCSPTensorStreamSpec],
) -> dict[str, Any]:
    """Return a deterministic JSON-ready manifest for JCSP stream metadata.

    The manifest is a compress-time planning artifact only: it records the
    byte/marginal contract that must be closed before archive construction, but
    it does not encode payloads, load scorers, dispatch jobs, or claim scores.
    """

    records = [
        _stream_spec_record(stream)
        for stream in sorted(
            streams,
            key=lambda item: (int(item.decomposition_index), item.name),
        )
    ]
    names = [record["name"] for record in records]
    duplicate_names = sorted(
        name for name in set(names) if names.count(name) > 1
    )
    if duplicate_names:
        raise ValueError(
            "jcsp_stream_specs_manifest requires unique stream names; "
            f"duplicates: {', '.join(duplicate_names)}"
        )
    stream_ids = [record["stream_id"] for record in records]
    duplicate_ids = sorted(
        stream_id for stream_id in set(stream_ids) if stream_ids.count(stream_id) > 1
    )
    if duplicate_ids:
        raise ValueError(
            "jcsp_stream_specs_manifest requires unique stream ids; "
            f"duplicates: {', '.join(duplicate_ids)}"
        )
    indices = [int(record["decomposition_index"]) for record in records]
    expected_indices = list(range(len(records)))
    if indices != expected_indices:
        raise ValueError(
            "jcsp_stream_specs_manifest requires contiguous decomposition_index "
            f"values {expected_indices}, got {indices}"
        )
    payload: dict[str, Any] = {
        "schema": JCSP_STREAM_METADATA_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "stream_count": len(records),
        "streams": records,
        "determinism_contract": {
            "json_encoding": "sort_keys=True,separators=(',', ':'),allow_nan=False",
            "stream_order": "decomposition_index_assigned_after_utf8_name_byte_sort",
            "stream_id": (
                "sha256(name,tensor_dtype,tensor_shape,num_elements,raw_bytes)"
            ),
        },
        "sidecar_policy": {
            "score_affecting_sidecars_allowed": False,
            "required_archive_members": [JCSP_ARCHIVE_MEMBER_NAME],
            "archive_member_contract_schema": JCSP_ARCHIVE_MEMBER_CONTRACT_SCHEMA,
        },
        "promotion_blockers": [
            "qint_or_exact_wire_stream_missing",
            "decode_validation_missing",
            "byte_closed_archive_member_missing",
            "stream_bytes_charged_reconciliation_missing",
            "runtime_loader_parity_missing",
            JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
            "exact_cuda_auth_eval_missing",
        ],
    }
    payload["manifest_sha256"] = _canonical_json_sha256(payload)
    return payload


# ────────────────────────────────────────────────────────────────────────────
# Stream specification
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class StreamSource:
    """Input specification for one codec stream.

    Fields
    ------
    name : str
        Short identifier (e.g., "renderer_qint", "segnet_logit_qint",
        "pose_qint", "entropy_residual"). Length must be 1..255 chars.
    qints : np.ndarray
        1-D integer array. Required.
    num_symbols : int
        Alphabet size after offset. Required.
    offset : int
        Offset added to ``qints`` to get symbol indices in [0, num_symbols).
        Required.
    codec_kind : int
        ``KIND_ARITHMETIC_STATIC`` / ``KIND_BALLE_HYPERPRIOR`` /
        ``KIND_RAW_PASSTHROUGH``. Required.
    balle_codec : BalleHyperpriorCodec or None
        Required iff ``codec_kind == KIND_BALLE_HYPERPRIOR``.
    raw_passthrough_bytes : bytes or None
        Required iff ``codec_kind == KIND_RAW_PASSTHROUGH``. The pre-encoded
        bytes that should be carried through unchanged (e.g., AV1 payload).
    score_per_byte_marginal : float
        Cached dScore/dByte estimate from frontier-sampler. Used by the ADMM
        coordinator to project byte allocations. Required.
    """

    name: str
    qints: np.ndarray
    num_symbols: int
    offset: int
    codec_kind: int
    balle_codec: BalleHyperpriorCodec | None = None
    raw_passthrough_bytes: bytes | None = None
    score_per_byte_marginal: float = 0.0

    def __post_init__(self) -> None:
        self.name = _validate_jcsp_stream_name(
            self.name,
            context="StreamSource",
        )
        if self.codec_kind not in (
            KIND_ARITHMETIC_STATIC,
            KIND_BALLE_HYPERPRIOR,
            KIND_RAW_PASSTHROUGH,
        ):
            raise ValueError(
                f"StreamSource.codec_kind must be one of "
                f"({KIND_ARITHMETIC_STATIC}, {KIND_BALLE_HYPERPRIOR}, "
                f"{KIND_RAW_PASSTHROUGH}); got {self.codec_kind}"
            )
        if self.codec_kind == KIND_BALLE_HYPERPRIOR and self.balle_codec is None:
            raise ValueError(
                f"StreamSource(name={self.name!r}): codec_kind=BALLE_HYPERPRIOR "
                f"requires non-None balle_codec"
            )
        if (
            self.codec_kind == KIND_RAW_PASSTHROUGH
            and self.raw_passthrough_bytes is None
        ):
            raise ValueError(
                f"StreamSource(name={self.name!r}): codec_kind=RAW_PASSTHROUGH "
                f"requires non-None raw_passthrough_bytes"
            )


# ────────────────────────────────────────────────────────────────────────────
# Per-stream proximal-codec wrapper for ADMM
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class _CodecProximal:
    """Wraps a StreamSource as a StreamProximalCodec for the ADMM coordinator.

    The proximal_step "encodes" the stream at a target byte allocation and
    returns the actual bytes plus the cached score-marginal. For Lane 20-class
    arithmetic codecs the byte count is largely INDEPENDENT of the dual
    (the codec already runs at its R-D operating point, given its alphabet
    distribution). For the Ballé hyperprior, byte count varies with the
    block_size / hyperprior config, but those are fixed per-codec instance.

    So the proximal subproblem reduces to: "what is the actual byte cost
    of encoding stream s with codec_kind, and what is its cached marginal
    dScore/dByte?" The dual lambda is unused here because the codec is
    parameter-free at this level — the COMPLETED byte cost IS its operating
    point. Future Phase 3 work: parametrize the codec on the dual (e.g.,
    Ballé block_size sweep, arithmetic chunk-K sweep) so ADMM can move
    along the within-codec R-D curve too.

    For RAW_PASSTHROUGH, the bytes are pre-fixed (e.g., AV1-encoded mask
    payload that has its own internal codec); the proximal_step returns the
    fixed cost.
    """

    src: StreamSource
    _cached_bytes: int = field(init=False, default=-1)
    _cached_payload: bytes = field(init=False, default=b"")
    # CRITICAL fix (audit finding 4 / data corruption): _cached_kind tracks the
    # codec that was actually USED. When ``codec_kind == KIND_BALLE_HYPERPRIOR``
    # but ``encode_qints_balle_auto`` returns ``static_wins``, the payload IS an
    # arithmetic-static blob, not BHv1. The orchestrator must record the actual
    # kind in the JCSP container so the inflate-side dispatcher routes the
    # payload to ``decode_qints_arithmetic``, not ``decode_qints_balle``.
    _cached_kind: int = field(init=False, default=-1)

    @property
    def name(self) -> str:
        return self.src.name

    def proximal_step(
        self, target_bytes: float, dual: float
    ) -> ProximalStepResult:
        if self._cached_bytes < 0:
            payload, byte_count = self._encode_once()
            self._cached_payload = payload
            self._cached_bytes = byte_count
        # For homogeneous codecs (no within-codec parameter), score and
        # marginal are stream-fixed. We carry the cached marginal forward.
        return ProximalStepResult(
            encoded_bytes=int(self._cached_bytes),
            score_delta=float(
                self.src.score_per_byte_marginal * self._cached_bytes
            ),
            marginal=float(self.src.score_per_byte_marginal),
            state=("payload", id(self._cached_payload)),
        )

    def _encode_once(self) -> tuple[bytes, int]:
        s = self.src
        if s.codec_kind == KIND_ARITHMETIC_STATIC:
            blob = encode_qints_arithmetic(
                qints=s.qints, num_symbols=s.num_symbols, offset=s.offset
            )
            self._cached_kind = KIND_ARITHMETIC_STATIC
            return blob, len(blob)
        if s.codec_kind == KIND_BALLE_HYPERPRIOR:
            assert s.balle_codec is not None
            # Try BHv1 (lite + full); fall back to static if neither beats it.
            static_baseline = encode_qints_arithmetic(
                qints=s.qints, num_symbols=s.num_symbols, offset=s.offset
            )
            blob, mode_name, _stats = encode_qints_balle_auto(
                qints=s.qints,
                num_symbols=s.num_symbols,
                offset=s.offset,
                num_chunks_lite=4,
                full_codec=s.balle_codec,
                static_baseline_bytes=len(static_baseline),
            )
            if mode_name == "static_wins":
                # Auto says static beats BHv1 — ship the static blob. CRITICAL:
                # the inflate dispatcher must see KIND_ARITHMETIC_STATIC so it
                # routes the AQv1 payload to decode_qints_arithmetic, not
                # decode_qints_balle. (Audit finding CRITICAL 4 / data corrupt.)
                self._cached_kind = KIND_ARITHMETIC_STATIC
                return static_baseline, len(static_baseline)
            self._cached_kind = KIND_BALLE_HYPERPRIOR
            return blob, len(blob)
        if s.codec_kind == KIND_RAW_PASSTHROUGH:
            assert s.raw_passthrough_bytes is not None
            self._cached_kind = KIND_RAW_PASSTHROUGH
            return s.raw_passthrough_bytes, len(s.raw_passthrough_bytes)
        raise AssertionError(
            f"_CodecProximal: unhandled codec_kind {s.codec_kind}"
        )

    @property
    def cached_payload(self) -> bytes:
        if self._cached_bytes < 0:
            self.proximal_step(target_bytes=0.0, dual=0.0)
        return self._cached_payload

    @property
    def actual_codec_kind(self) -> int:
        """The codec kind actually used (may differ from src.codec_kind when
        encode_qints_balle_auto returns static_wins). The JCSP container must
        record THIS value so the inflate-side dispatcher routes correctly.
        (Audit finding CRITICAL 4: data-corruption fix.)
        """
        if self._cached_bytes < 0:
            self.proximal_step(target_bytes=0.0, dual=0.0)
        return self._cached_kind


# ────────────────────────────────────────────────────────────────────────────
# Stack result
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class StackStreamResult:
    """Per-stream result emitted by the orchestrator."""

    name: str
    codec_kind: int
    admm_bytes_target: float
    actual_bytes: int
    score_delta: float
    marginal: float
    payload: bytes


@dataclass
class JointCodecStackResult:
    """Top-level orchestrator result."""

    converged: bool
    iters: int
    total_bytes: int
    waterline_kkt_residual: float
    streams: list[StackStreamResult]
    container_bytes: bytes
    admm_history_len: int = 0


# ────────────────────────────────────────────────────────────────────────────
# JCSP wire format
# ────────────────────────────────────────────────────────────────────────────


def _payload_magic(payload: bytes) -> bytes:
    if len(payload) < 4:
        raise ValueError(
            f"JCSP stream payload is too small for codec magic: {len(payload)} bytes"
        )
    return payload[:4]


def _require_payload_magic_matches_codec_kind(
    *,
    codec_kind: int,
    payload: bytes,
    context: str,
) -> bytes:
    """Fail closed when runtime dispatch kind and payload wire magic disagree."""

    if codec_kind == KIND_RAW_PASSTHROUGH:
        if len(payload) <= 0:
            raise ValueError(f"{context}: raw passthrough payload is empty")
        return payload[:4]
    allowed_magics = _PAYLOAD_MAGICS_BY_CODEC_KIND.get(int(codec_kind))
    if allowed_magics is None:
        raise ValueError(f"{context}: invalid codec_kind {codec_kind}")
    magic = _payload_magic(payload)
    if magic not in allowed_magics:
        allowed = ", ".join(repr(item) for item in allowed_magics)
        raise ValueError(
            f"{context}: payload magic {magic!r} is incompatible with "
            f"codec_kind {codec_kind}; expected one of {allowed}"
        )
    return magic


def _pack_jcsp_container(
    *,
    streams: list[StackStreamResult],
    waterline_kkt_residual: float,
    iters: int,
    converged: bool,
) -> bytes:
    """Pack per-stream results into the JCSP container bytes."""
    if len(streams) > 255:
        raise ValueError(
            f"_pack_jcsp_container: max 255 streams, got {len(streams)}"
        )
    out = io.BytesIO()
    out.write(_JCSP_MAGIC)
    out.write(struct.pack("<H", _JCSP_VERSION))
    out.write(struct.pack("<B", len(streams)))
    for s in streams:
        name_b = s.name.encode("utf-8")
        if len(name_b) > 255:
            raise ValueError(
                f"_pack_jcsp_container: stream name {s.name!r} too long"
            )
        if s.codec_kind not in (
            KIND_ARITHMETIC_STATIC,
            KIND_BALLE_HYPERPRIOR,
            KIND_RAW_PASSTHROUGH,
        ):
            raise ValueError(
                f"_pack_jcsp_container: invalid codec_kind {s.codec_kind} "
                f"for stream {s.name!r}"
            )
        if int(s.actual_bytes) != len(s.payload):
            raise ValueError(
                f"_pack_jcsp_container: actual_bytes={s.actual_bytes} does "
                f"not match payload_len={len(s.payload)} for stream {s.name!r}"
            )
        _require_payload_magic_matches_codec_kind(
            codec_kind=s.codec_kind,
            payload=s.payload,
            context=f"_pack_jcsp_container stream {s.name!r}",
        )
        out.write(struct.pack("<B", len(name_b)))
        out.write(name_b)
        out.write(struct.pack("<B", int(s.codec_kind)))
        out.write(struct.pack("<I", round(max(0, s.admm_bytes_target))))
        out.write(struct.pack("<I", int(s.actual_bytes)))
        out.write(struct.pack("<i", round(s.score_delta * 1e6)))
        out.write(struct.pack("<i", round(s.marginal * 1e6)))
        out.write(struct.pack("<I", len(s.payload)))
        out.write(s.payload)
    out.write(struct.pack("<I", max(0, round(waterline_kkt_residual * 1e6))))
    out.write(struct.pack("<I", int(iters)))
    out.write(struct.pack("<B", int(bool(converged))))
    return out.getvalue()


def _require_jcsp_bytes(blob: bytes, cursor: int, n_bytes: int, context: str) -> None:
    if n_bytes < 0 or cursor < 0 or cursor + n_bytes > len(blob):
        raise ValueError(
            f"unpack_jcsp_container: truncated {context} at offset {cursor}; "
            f"need {n_bytes} bytes, blob len={len(blob)}"
        )


def unpack_jcsp_container(blob: bytes) -> dict:
    """Inverse of ``_pack_jcsp_container``: unpack the JCSP container.

    Returns a dict with the parsed metadata + raw per-stream payloads. The
    caller is responsible for dispatching each payload to its appropriate
    decoder (decode_qints_arithmetic, decode_qints_balle, or raw bytes).

    Used by the inflate-side and by the test harness.
    """
    if blob is None or len(blob) < 4 + 2 + 1:
        raise ValueError(
            f"unpack_jcsp_container: blob too small (len={len(blob) if blob else 0})"
        )
    if blob[:4] != _JCSP_MAGIC:
        raise ValueError(
            f"unpack_jcsp_container: bad magic {blob[:4]!r}, expected {_JCSP_MAGIC!r}"
        )
    cursor = 4
    (version,) = struct.unpack_from("<H", blob, cursor)
    cursor += 2
    if version != _JCSP_VERSION:
        raise ValueError(
            f"unpack_jcsp_container: unsupported version {version}"
        )
    (n_streams,) = struct.unpack_from("<B", blob, cursor)
    cursor += 1
    streams: list[dict] = []
    for _ in range(n_streams):
        _require_jcsp_bytes(blob, cursor, 1, "stream name length")
        (name_len,) = struct.unpack_from("<B", blob, cursor)
        cursor += 1
        _require_jcsp_bytes(blob, cursor, name_len, "stream name")
        name = blob[cursor : cursor + name_len].decode("utf-8")
        cursor += name_len
        _require_jcsp_bytes(blob, cursor, 1, f"stream {name!r} codec kind")
        (codec_kind,) = struct.unpack_from("<B", blob, cursor)
        cursor += 1
        if codec_kind not in (
            KIND_ARITHMETIC_STATIC,
            KIND_BALLE_HYPERPRIOR,
            KIND_RAW_PASSTHROUGH,
        ):
            raise ValueError(
                f"unpack_jcsp_container: invalid codec_kind {codec_kind} "
                f"for stream {name!r}"
            )
        _require_jcsp_bytes(blob, cursor, 4, f"stream {name!r} ADMM target")
        (admm_target,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        _require_jcsp_bytes(blob, cursor, 4, f"stream {name!r} actual bytes")
        (actual_bytes,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        _require_jcsp_bytes(blob, cursor, 4, f"stream {name!r} score delta")
        (score_milli,) = struct.unpack_from("<i", blob, cursor)
        cursor += 4
        _require_jcsp_bytes(blob, cursor, 4, f"stream {name!r} marginal")
        (margin_milli,) = struct.unpack_from("<i", blob, cursor)
        cursor += 4
        _require_jcsp_bytes(blob, cursor, 4, f"stream {name!r} payload length")
        (payload_len,) = struct.unpack_from("<I", blob, cursor)
        cursor += 4
        if actual_bytes != payload_len:
            raise ValueError(
                f"unpack_jcsp_container: actual_bytes={actual_bytes} does "
                f"not match payload_len={payload_len} for stream {name!r}"
            )
        _require_jcsp_bytes(blob, cursor, payload_len, f"stream {name!r} payload")
        payload = blob[cursor : cursor + payload_len]
        payload_magic = _require_payload_magic_matches_codec_kind(
            codec_kind=codec_kind,
            payload=payload,
            context=f"unpack_jcsp_container stream {name!r}",
        )
        cursor += payload_len
        streams.append(
            {
                "name": name,
                "codec_kind": codec_kind,
                "admm_bytes_target": admm_target,
                "actual_bytes": actual_bytes,
                "score_delta": score_milli / 1e6,
                "marginal": margin_milli / 1e6,
                "payload": payload,
                "payload_magic": payload_magic,
            }
        )
    _require_jcsp_bytes(blob, cursor, 4, "KKT residual")
    (kkt_milli,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    _require_jcsp_bytes(blob, cursor, 4, "ADMM iteration count")
    (iters,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4
    _require_jcsp_bytes(blob, cursor, 1, "converged flag")
    (converged,) = struct.unpack_from("<B", blob, cursor)
    cursor += 1
    if cursor != len(blob):
        raise ValueError(
            f"unpack_jcsp_container: trailing bytes after JCSP payload "
            f"(cursor={cursor}, len={len(blob)})"
        )
    return {
        "version": version,
        "streams": streams,
        "waterline_kkt_residual": kkt_milli / 1e6,
        "iters": iters,
        "converged": bool(converged),
    }


def validate_jcsp_container_runtime_parity(blob: bytes) -> dict[str, Any]:
    """Validate JCSP metadata that an inflate-side runtime loader would trust.

    This is a byte-structure/readiness check, not a score claim. It verifies
    the container can be unpacked fail-closed and that each stream's recorded
    ``codec_kind`` matches the payload wire magic used for runtime dispatch.
    """

    parsed = unpack_jcsp_container(blob)
    stream_records: list[dict[str, Any]] = []
    for stream in parsed["streams"]:
        magic = _require_payload_magic_matches_codec_kind(
            codec_kind=int(stream["codec_kind"]),
            payload=stream["payload"],
            context=f"validate_jcsp_container_runtime_parity stream {stream['name']!r}",
        )
        stream_records.append(
            {
                "name": stream["name"],
                "codec_kind": int(stream["codec_kind"]),
                "payload_magic": magic.decode("ascii", errors="replace"),
                "actual_bytes": int(stream["actual_bytes"]),
                "runtime_dispatch_checked": True,
            }
        )
    return {
        "schema": "jcsp_runtime_loader_parity_v1",
        "score_claim": False,
        "ready_for_runtime_loader": True,
        "stream_count": len(stream_records),
        "streams": stream_records,
        "waterline_kkt_residual": parsed["waterline_kkt_residual"],
        "iters": parsed["iters"],
        "converged": parsed["converged"],
    }


# ────────────────────────────────────────────────────────────────────────────
# JCSP archive-member runtime contract
# ────────────────────────────────────────────────────────────────────────────


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _require_bytes(blob: bytes | bytearray | memoryview, *, context: str) -> bytes:
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError(f"{context} must be bytes-like, got {type(blob).__name__}")
    out = bytes(blob)
    if not out:
        raise ValueError(f"{context} must be non-empty")
    return out


def _validate_jcsp_member_compression(compress_type: int) -> int:
    out = int(compress_type)
    if out not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise ValueError(
            "JCSP archive members must use ZIP_STORED or ZIP_DEFLATED; "
            f"got compress_type={compress_type}"
        )
    return out


def _decode_zip_member_name(raw: bytes, flag_bits: int) -> str:
    encoding = "utf-8" if flag_bits & 0x800 else "cp437"
    return raw.decode(encoding, errors="strict")


def _local_header_name_from_archive_bytes(
    archive_bytes: bytes,
    info: zipfile.ZipInfo,
) -> str:
    offset = int(info.header_offset)
    if offset < 0 or offset + 30 > len(archive_bytes):
        raise ValueError(f"invalid ZIP local header offset for {info.filename!r}")
    fixed = archive_bytes[offset : offset + 30]
    if fixed[:4] != b"PK\x03\x04":
        raise ValueError(f"invalid ZIP local header for {info.filename!r}")
    flag_bits = struct.unpack_from("<H", fixed, 6)[0]
    name_len, extra_len = struct.unpack_from("<HH", fixed, 26)
    name_start = offset + 30
    extra_start = name_start + int(name_len)
    extra_end = extra_start + int(extra_len)
    if extra_end > len(archive_bytes):
        raise ValueError(f"truncated ZIP local header for {info.filename!r}")
    local_raw = archive_bytes[name_start:extra_start]
    return _decode_zip_member_name(local_raw, flag_bits)


def _require_jcsp_member_metadata_deterministic(info: zipfile.ZipInfo) -> None:
    if tuple(info.date_time) != DETERMINISTIC_ZIP_DATE_TIME:
        raise ValueError(
            f"JCSP member {info.filename!r} has non-deterministic timestamp "
            f"{info.date_time!r}; expected {DETERMINISTIC_ZIP_DATE_TIME!r}"
        )
    if info.extra:
        raise ValueError(f"JCSP member {info.filename!r} has ZIP extra fields")
    if info.comment:
        raise ValueError(f"JCSP member {info.filename!r} has ZIP comment")
    mode = (int(info.external_attr) >> 16) & 0o777
    if mode != DETERMINISTIC_ZIP_FILE_MODE:
        raise ValueError(
            f"JCSP member {info.filename!r} has file mode {mode:o}; "
            f"expected {DETERMINISTIC_ZIP_FILE_MODE:o}"
        )
    if int(info.create_system) != 3:
        raise ValueError(
            f"JCSP member {info.filename!r} has create_system={info.create_system}; "
            "expected 3 for deterministic Unix-style metadata"
        )


def build_jcsp_archive_member(
    *,
    container_bytes: bytes,
    member_name: str = JCSP_ARCHIVE_MEMBER_NAME,
    compress_type: int = zipfile.ZIP_STORED,
) -> bytes:
    """Return deterministic ``archive.zip`` bytes with one JCSP member.

    The returned archive is a byte-closed fixture/candidate surface: the JCSP
    payload is inside the ZIP member, member metadata is deterministic, and the
    member is immediately passed through the runtime-loader contract below.
    This is still not score evidence and does not imply dispatch readiness.
    """

    member = validate_archive_member_name(member_name)
    payload = _require_bytes(container_bytes, context="container_bytes")
    validate_jcsp_container_runtime_parity(payload)
    compression = _validate_jcsp_member_compression(compress_type)
    compresslevel = None if compression == zipfile.ZIP_STORED else 9

    out = io.BytesIO()
    with zipfile.ZipFile(
        out,
        "w",
        compression=compression,
        allowZip64=False,
    ) as zf:
        write_deterministic_zip_member(
            zf,
            member,
            payload,
            compress_type=compression,
            compresslevel=compresslevel,
        )
    archive_bytes = out.getvalue()
    load_jcsp_archive_member_for_runtime(
        archive_bytes=archive_bytes,
        member_name=member,
        require_single_member=True,
    )
    return archive_bytes


def build_jcsp_noop_archive_fixture(
    *,
    member_name: str = JCSP_ARCHIVE_MEMBER_NAME,
    compress_type: int = zipfile.ZIP_STORED,
) -> bytes:
    """Build a deterministic scoreless archive fixture with a zero-stream JCSP.

    This gives the runtime contract a byte-closed archive member to consume
    before any real joint-stack payload is dispatchable. The fixture carries no
    qint streams, performs no scorer work, and cannot support a score claim.
    """

    container = _pack_jcsp_container(
        streams=[],
        waterline_kkt_residual=0.0,
        iters=0,
        converged=True,
    )
    return build_jcsp_archive_member(
        container_bytes=container,
        member_name=member_name,
        compress_type=compress_type,
    )


def pack_jcsp_local_skeleton_container(
    *,
    manifest: Mapping[str, Any],
) -> bytes:
    """Pack a local-only JCSP skeleton manifest into deterministic bytes.

    ``JCSK`` is intentionally distinct from the runtime ``JCSP`` magic. The
    resulting member can be byte-closed and hashed, but it must fail closed for
    exact-eval dispatch until a submission runtime consumes the real JCSP
    container and strict preflight proves that path.
    """

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    payload["schema"] = str(payload.get("schema", JCSP_LOCAL_SKELETON_SCHEMA))
    if payload["schema"] != JCSP_LOCAL_SKELETON_SCHEMA:
        raise ValueError(
            "JCSP local skeleton manifest has wrong schema "
            f"{payload['schema']!r}; expected {JCSP_LOCAL_SKELETON_SCHEMA!r}"
        )
    payload["container_magic"] = _JCSP_LOCAL_SKELETON_MAGIC.decode("ascii")
    payload["container_version"] = _JCSP_LOCAL_SKELETON_VERSION
    for required_false_field in (
        "score_claim",
        "dispatch_attempted",
        "ready_for_runtime_loader",
        "ready_for_submission_runtime_consumption",
        "ready_for_exact_eval_dispatch",
    ):
        if payload.get(required_false_field, False) is not False:
            raise ValueError(
                "JCSP local skeleton manifest field "
                f"{required_false_field!r} must be False"
            )
    streams = payload.get("streams", [])
    if not isinstance(streams, list):
        raise ValueError("JCSP local skeleton manifest streams must be a list")
    if int(payload.get("stream_count", len(streams))) != len(streams):
        raise ValueError(
            "JCSP local skeleton manifest stream_count does not match streams"
        )
    payload["manifest_sha256"] = _canonical_json_sha256(payload)
    body = _canonical_json_bytes(payload)

    out = io.BytesIO()
    out.write(_JCSP_LOCAL_SKELETON_MAGIC)
    out.write(struct.pack("<H", _JCSP_LOCAL_SKELETON_VERSION))
    out.write(struct.pack("<I", len(body)))
    out.write(body)
    return out.getvalue()


def unpack_jcsp_local_skeleton_container(blob: bytes) -> dict[str, Any]:
    """Unpack and verify a local-only ``JCSK`` skeleton container."""

    payload = _require_bytes(blob, context="local JCSP skeleton container")
    fixed_header_len = 4 + 2 + 4
    if len(payload) < fixed_header_len:
        raise ValueError(
            "unpack_jcsp_local_skeleton_container: blob too small "
            f"(len={len(payload)})"
        )
    if payload[:4] != _JCSP_LOCAL_SKELETON_MAGIC:
        raise ValueError(
            "unpack_jcsp_local_skeleton_container: bad magic "
            f"{payload[:4]!r}, expected {_JCSP_LOCAL_SKELETON_MAGIC!r}"
        )
    (version,) = struct.unpack_from("<H", payload, 4)
    if version != _JCSP_LOCAL_SKELETON_VERSION:
        raise ValueError(
            "unpack_jcsp_local_skeleton_container: unsupported version "
            f"{version}"
        )
    (body_len,) = struct.unpack_from("<I", payload, 6)
    body_start = fixed_header_len
    body_end = body_start + int(body_len)
    if body_end != len(payload):
        raise ValueError(
            "unpack_jcsp_local_skeleton_container: body length mismatch "
            f"declared={body_len} actual={len(payload) - body_start}"
        )
    try:
        manifest = json.loads(
            payload[body_start:body_end].decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_object_pairs,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(
            "unpack_jcsp_local_skeleton_container: manifest is not "
            "deterministic UTF-8 JSON"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise ValueError(
            "unpack_jcsp_local_skeleton_container: manifest must be a mapping"
        )
    if manifest.get("schema") != JCSP_LOCAL_SKELETON_SCHEMA:
        raise ValueError(
            "unpack_jcsp_local_skeleton_container: wrong schema "
            f"{manifest.get('schema')!r}"
        )
    if manifest.get("container_magic") != _JCSP_LOCAL_SKELETON_MAGIC.decode("ascii"):
        raise ValueError(
            "unpack_jcsp_local_skeleton_container: container_magic mismatch"
        )
    if int(manifest.get("container_version", -1)) != _JCSP_LOCAL_SKELETON_VERSION:
        raise ValueError(
            "unpack_jcsp_local_skeleton_container: container_version mismatch"
        )
    for required_false_field in (
        "score_claim",
        "dispatch_attempted",
        "ready_for_runtime_loader",
        "ready_for_submission_runtime_consumption",
        "ready_for_exact_eval_dispatch",
    ):
        if manifest.get(required_false_field) is not False:
            raise ValueError(
                "unpack_jcsp_local_skeleton_container: "
                f"{required_false_field} must be False"
            )
    streams = manifest.get("streams")
    if not isinstance(streams, list):
        raise ValueError(
            "unpack_jcsp_local_skeleton_container: streams must be a list"
        )
    if int(manifest.get("stream_count", -1)) != len(streams):
        raise ValueError(
            "unpack_jcsp_local_skeleton_container: stream_count mismatch"
        )
    recorded_sha = manifest.get("manifest_sha256")
    if not isinstance(recorded_sha, str) or len(recorded_sha) != 64:
        raise ValueError(
            "unpack_jcsp_local_skeleton_container: missing manifest_sha256"
        )
    without_sha = dict(manifest)
    without_sha.pop("manifest_sha256", None)
    actual_sha = _canonical_json_sha256(without_sha)
    if actual_sha != recorded_sha:
        raise ValueError(
            "unpack_jcsp_local_skeleton_container: manifest_sha256 mismatch"
        )
    return dict(manifest)


def _collect_local_skeleton_blockers(
    manifest: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    for blocker_field in ("promotion_blockers", "dispatch_blockers"):
        raw = manifest.get(blocker_field, ())
        if isinstance(raw, list):
            blockers.extend(str(item) for item in raw)
    streams = manifest.get("streams", ())
    if isinstance(streams, list):
        for stream in streams:
            if not isinstance(stream, Mapping):
                continue
            raw = stream.get("dispatch_blockers", ())
            if isinstance(raw, list):
                blockers.extend(str(item) for item in raw)
    blockers.extend(
        [
            JCSP_LOCAL_SKELETON_RUNTIME_BLOCKER,
            JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
            "strict_preflight_proof_missing",
            "exact_cuda_auth_eval_missing",
        ]
    )
    return list(dict.fromkeys(blockers))


def build_jcsp_local_skeleton_archive_member(
    *,
    manifest: Mapping[str, Any],
    member_name: str = JCSP_ARCHIVE_MEMBER_NAME,
    compress_type: int = zipfile.ZIP_STORED,
) -> tuple[bytes, dict[str, Any]]:
    """Build a deterministic one-member ZIP carrying a local ``JCSK`` payload."""

    member = validate_archive_member_name(member_name)
    container_bytes = pack_jcsp_local_skeleton_container(manifest=manifest)
    compression = _validate_jcsp_member_compression(compress_type)
    compresslevel = None if compression == zipfile.ZIP_STORED else 9

    out = io.BytesIO()
    with zipfile.ZipFile(
        out,
        "w",
        compression=compression,
        allowZip64=False,
    ) as zf:
        write_deterministic_zip_member(
            zf,
            member,
            container_bytes,
            compress_type=compression,
            compresslevel=compresslevel,
        )
    archive_bytes = out.getvalue()
    contract = load_jcsp_local_skeleton_archive_member(
        archive_bytes=archive_bytes,
        member_name=member,
        require_single_member=True,
    )
    return archive_bytes, contract


def _jcsp_runtime_raw_output_parity_contract() -> dict[str, Any]:
    """Return the fail-closed raw-output parity proof contract.

    The archive/runtime builder does not know the contest names file yet, so it
    records the required proof schema and acceptance conditions. The
    stdlib-only submission bridge fills in expected output paths during inflate.
    """

    expected_raw_outputs: list[str] = []
    return {
        "schema": JCSP_RUNTIME_RAW_OUTPUT_PARITY_CONTRACT_SCHEMA,
        "required_proof_schema": JCSP_RUNTIME_RAW_OUTPUT_PARITY_PROOF_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "expected_raw_outputs_known": False,
        "expected_raw_output_count": 0,
        "expected_raw_outputs_sha256": _sha256_bytes(
            _canonical_json_bytes({"expected_raw_outputs": expected_raw_outputs})
        ),
        "required_candidate_output_source": "jcsp_runtime_bridge_emitted_rawvideo",
        "required_reference_output_source": (
            "contest_reference_runtime_or_byte_custody_baseline"
        ),
        "preexisting_raw_outputs_are_not_parity_proof": True,
        "existing_raw_output_count_at_probe": 0,
        "required_per_output_fields": [
            "path",
            "candidate_exists",
            "candidate_bytes",
            "candidate_sha256",
            "reference_exists",
            "reference_bytes",
            "reference_sha256",
            "byte_exact_match",
        ],
        "acceptance_conditions": [
            "jcsp_stream_consumer_decodes_jcsp_streams",
            "bridge_emits_exactly_expected_raw_outputs",
            "candidate_outputs_are_from_current_bridge_run",
            "reference_outputs_are_from_contest_runtime_or_custody_baseline",
            "all_candidate_sha256_values_match_reference_sha256_values",
            "parity_proof_manifest_uses_required_schema",
        ],
        "ready_for_output_parity": False,
        "ready_for_submission_runtime_consumption": False,
        "dispatch_blocker": JCSP_SUBMISSION_RUNTIME_OUTPUT_PARITY_BLOCKER,
    }


def _jcsp_runtime_raw_output_consumer_contract() -> dict[str, Any]:
    """Return the fail-closed narrow runtime consumer readiness contract."""

    return {
        "schema": JCSP_RUNTIME_RAW_OUTPUT_CONSUMER_READINESS_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "runtime_bridge_modes": [
            "probe",
            "plan-real-raw-outputs",
            "emit-real-raw-outputs",
        ],
        "planning_bridge_mode": "plan-real-raw-outputs",
        "emission_bridge_mode": "emit-real-raw-outputs",
        "runtime_preflight_adapter_id": JCSP_RUNTIME_REAL_AQ_PREFLIGHT_ADAPTER_ID,
        "decoder_adapter_id": JCSP_RUNTIME_AQ_RAWVIDEO_ADAPTER_ID,
        "decoder_stream_kind": JCSP_RUNTIME_AQ_RAWVIDEO_STREAM_KIND,
        "accepted_codec_kind": KIND_ARITHMETIC_STATIC,
        "accepted_payload_magics": ["AQv1", "AQc1"],
        "stream_name_contract": (
            "stream name must exactly equal expected .raw relative path"
        ),
        "stream_payload_contract": (
            "AQ num_symbols=256, offset=0, decoded nonempty bytes are RGB24 "
            "rawvideo with length divisible by 3"
        ),
        "expected_raw_outputs_known": False,
        "no_scorer_at_inflate": True,
        "score_affecting_sidecars_allowed": False,
        "candidate_output_source": "jcsp_runtime_bridge_emitted_rawvideo",
        "would_emit_contest_raw_outputs": False,
        "raw_output_files_written": False,
        "ready_for_raw_output_emission": False,
        "ready_for_output_parity": False,
        "ready_for_submission_runtime_consumption": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            "jcsp_expected_raw_outputs_unknown",
            "jcsp_runtime_preflight_adapter_not_run",
            JCSP_SUBMISSION_RUNTIME_OUTPUT_PARITY_BLOCKER,
            "exact_cuda_auth_eval_missing",
        ],
    }


def _jcsp_submission_runtime_consumption_contract(
    *,
    member_name: str,
) -> dict[str, Any]:
    """Return the fail-closed runtime consumption gate for JCSP archives."""

    raw_output_parity_contract = _jcsp_runtime_raw_output_parity_contract()
    raw_output_consumer_contract = _jcsp_runtime_raw_output_consumer_contract()
    output_contract = {
        "schema": JCSP_SUBMISSION_RUNTIME_OUTPUT_CONTRACT_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "required_when_jcsp_member_present": True,
        "contest_output_format": "one uint8 RGB rawvideo .raw per names_file entry",
        "bridge_has_fail_closed_raw_output_consumer": True,
        "raw_output_consumer_contract": raw_output_consumer_contract,
        "bridge_emits_contest_raw_outputs": False,
        "raw_output_emission_attempted": False,
        "raw_output_parity_contract": raw_output_parity_contract,
        "required_raw_output_parity_proof_schema": (
            JCSP_RUNTIME_RAW_OUTPUT_PARITY_PROOF_SCHEMA
        ),
        "output_parity_checked": False,
        "output_parity_artifact": None,
        "ready_for_output_parity": False,
        "ready_for_submission_runtime_consumption": False,
        "dispatch_blocker": JCSP_SUBMISSION_RUNTIME_OUTPUT_PARITY_BLOCKER,
        "dispatch_blockers": [
            "jcsp_expected_raw_outputs_unknown",
            "jcsp_runtime_preflight_adapter_not_run",
            JCSP_SUBMISSION_RUNTIME_OUTPUT_PARITY_BLOCKER,
            "jcsp_raw_output_emission_missing",
        ],
    }
    return {
        "schema": JCSP_SUBMISSION_RUNTIME_CONSUMPTION_SCHEMA,
        "score_claim": False,
        "required_submission_runtime": JCSP_REQUIRED_SUBMISSION_RUNTIME,
        "runtime_bridge_path": JCSP_SUBMISSION_RUNTIME_BRIDGE_PATH,
        "runtime_bridge_probe_schema": JCSP_SUBMISSION_RUNTIME_BRIDGE_PROBE_SCHEMA,
        "required_member_name": member_name,
        "detects_required_member": True,
        "consumes_required_member": False,
        "dispatch_blocker": JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
        "dispatch_blockers": [
            JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
            *output_contract["dispatch_blockers"],
        ],
        "contest_output_contract": output_contract,
        "evidence": (
            "submissions/robust_current probes jcsp.bin during inflate and "
            "has a fail-closed narrow AQ/rawvideo preflight consumer for "
            "real JCSP bytes, but archive-level readiness does not know the "
            "contest names file, output parity artifact, or exact CUDA eval"
        ),
    }


def load_jcsp_local_skeleton_archive_member(
    *,
    archive_bytes: bytes,
    member_name: str = JCSP_ARCHIVE_MEMBER_NAME,
    require_single_member: bool = False,
) -> dict[str, Any]:
    """Validate a byte-closed local JCSP skeleton archive member.

    This loader proves deterministic local byte closure only. It deliberately
    returns ``ready_for_runtime_loader=False`` because ``JCSK`` is not consumed
    by ``submissions/robust_current`` and is not a substitute for the runtime
    ``JCSP`` container.
    """

    archive_blob = _require_bytes(archive_bytes, context="archive_bytes")
    member = validate_archive_member_name(member_name)

    try:
        with zipfile.ZipFile(io.BytesIO(archive_blob), "r") as zf:
            infos = zf.infolist()
            names = validate_zip_member_infos(infos)
            if require_single_member and names != [member]:
                raise ValueError(
                    "JCSP local skeleton archive must contain exactly one "
                    f"member {member!r}; got {names!r}"
                )
            if member not in names:
                raise ValueError(
                    f"JCSP local skeleton member {member!r} not found; "
                    f"got {names!r}"
                )

            local_header_names: list[dict[str, Any]] = []
            for info_row in infos:
                local_name = _local_header_name_from_archive_bytes(
                    archive_blob,
                    info_row,
                )
                if local_name != info_row.filename:
                    raise ValueError(
                        "ZIP central/local name mismatch: "
                        f"central={info_row.filename!r} local={local_name!r}"
                    )
                local_header_names.append(
                    {
                        "central": info_row.filename,
                        "local": local_name,
                    }
                )

            bad_crc = zf.testzip()
            if bad_crc is not None:
                raise ValueError(
                    f"JCSP local skeleton archive CRC check failed for {bad_crc!r}"
                )

            info = zf.getinfo(member)
            _validate_jcsp_member_compression(info.compress_type)
            _require_jcsp_member_metadata_deterministic(info)
            payload = zf.read(info)
    except zipfile.BadZipFile as exc:
        raise ValueError(
            "JCSP local skeleton loader got invalid ZIP bytes"
        ) from exc

    skeleton_manifest = unpack_jcsp_local_skeleton_container(payload)
    runtime_consumption = _jcsp_submission_runtime_consumption_contract(
        member_name=member,
    )
    dispatch_blockers = _collect_local_skeleton_blockers(skeleton_manifest)
    return {
        "schema": JCSP_LOCAL_ARCHIVE_MEMBER_CONTRACT_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_local_skeleton_loader": True,
        "ready_for_runtime_loader": False,
        "ready_for_submission_runtime_consumption": False,
        "ready_for_exact_eval_dispatch": False,
        "member_name": member,
        "archive_bytes": len(archive_blob),
        "archive_sha256": _sha256_bytes(archive_blob),
        "archive_members": names,
        "member_bytes": len(payload),
        "member_sha256": _sha256_bytes(payload),
        "member_compress_type": int(info.compress_type),
        "member_compress_size": int(info.compress_size),
        "member_crc32": f"{int(info.CRC):08x}",
        "member_file_mode": DETERMINISTIC_ZIP_FILE_MODE,
        "member_date_time": list(info.date_time),
        "local_header_names": local_header_names,
        "container_magic": _JCSP_LOCAL_SKELETON_MAGIC.decode("ascii"),
        "container_version": _JCSP_LOCAL_SKELETON_VERSION,
        "skeleton_manifest": skeleton_manifest,
        "runtime_consumption_contract": runtime_consumption,
        "byte_closed_archive_member": True,
        "single_member_no_sidecars": names == [member],
        "dispatch_blockers": dispatch_blockers,
    }


def load_jcsp_archive_member_for_runtime(
    *,
    archive_bytes: bytes,
    member_name: str = JCSP_ARCHIVE_MEMBER_NAME,
    require_single_member: bool = False,
) -> dict[str, Any]:
    """Validate and load a JCSP archive member as inflate-side code would.

    The contract is deliberately stricter than a bare ``ZipFile.read``: it
    rejects duplicate/unsafe names, central/local header name divergence,
    missing members, non-deterministic JCSP member metadata, CRC failures, and
    JCSP payload/kind mismatches before returning loader-ready metadata.
    """

    archive_blob = _require_bytes(archive_bytes, context="archive_bytes")
    member = validate_archive_member_name(member_name)

    try:
        with zipfile.ZipFile(io.BytesIO(archive_blob), "r") as zf:
            infos = zf.infolist()
            names = validate_zip_member_infos(infos)
            if require_single_member and names != [member]:
                raise ValueError(
                    "JCSP archive fixture must contain exactly one member "
                    f"{member!r}; got {names!r}"
                )
            if member not in names:
                raise ValueError(
                    f"JCSP archive member {member!r} not found; got {names!r}"
                )

            local_header_names: list[dict[str, Any]] = []
            for info in infos:
                local_name = _local_header_name_from_archive_bytes(
                    archive_blob,
                    info,
                )
                if local_name != info.filename:
                    raise ValueError(
                        "ZIP central/local name mismatch: "
                        f"central={info.filename!r} local={local_name!r}"
                    )
                local_header_names.append(
                    {
                        "central": info.filename,
                        "local": local_name,
                    }
                )

            bad_crc = zf.testzip()
            if bad_crc is not None:
                raise ValueError(f"JCSP archive CRC check failed for {bad_crc!r}")

            info = zf.getinfo(member)
            _validate_jcsp_member_compression(info.compress_type)
            _require_jcsp_member_metadata_deterministic(info)
            payload = zf.read(info)
    except zipfile.BadZipFile as exc:
        raise ValueError("JCSP archive member loader got invalid ZIP bytes") from exc

    parity = validate_jcsp_container_runtime_parity(payload)
    runtime_consumption = _jcsp_submission_runtime_consumption_contract(
        member_name=member,
    )
    return {
        "schema": JCSP_ARCHIVE_MEMBER_CONTRACT_SCHEMA,
        "score_claim": False,
        "ready_for_runtime_loader": True,
        "ready_for_submission_runtime_consumption": False,
        "ready_for_exact_eval_dispatch": False,
        "member_name": member,
        "archive_bytes": len(archive_blob),
        "archive_sha256": _sha256_bytes(archive_blob),
        "archive_members": names,
        "member_bytes": len(payload),
        "member_sha256": _sha256_bytes(payload),
        "member_compress_type": int(info.compress_type),
        "member_compress_size": int(info.compress_size),
        "member_crc32": f"{int(info.CRC):08x}",
        "member_file_mode": DETERMINISTIC_ZIP_FILE_MODE,
        "member_date_time": list(info.date_time),
        "local_header_names": local_header_names,
        "jcsp_runtime_parity": parity,
        "runtime_consumption_contract": runtime_consumption,
        "noop_fixture": parity["stream_count"] == 0,
        "dispatch_blockers": [
            *runtime_consumption["dispatch_blockers"],
            "not_integrated_into_submission_inflate_path",
            "no_lane_dispatch_claim",
            "exact_cuda_auth_eval_missing",
        ],
    }


def _codec_kind_reconciles_model_plan(
    *,
    planned: int,
    actual: int,
) -> tuple[bool, str]:
    if actual == planned:
        return True, "planned_codec_kind_used"
    if planned == KIND_BALLE_HYPERPRIOR and actual == KIND_ARITHMETIC_STATIC:
        return True, "balle_static_wins_actual_arithmetic"
    return False, "codec_kind_mismatch"


def _jcsp_stream_archive_byte_reconciliation(
    *,
    stream_records: list[dict[str, Any]],
    runtime_streams: list[dict[str, Any]],
    archive_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconcile planned per-stream bytes against the charged JCSP member."""

    per_stream: list[dict[str, Any]] = []
    intended_stream_payload_bytes = 0
    actual_stream_payload_bytes = 0
    for record, runtime in zip(stream_records, runtime_streams, strict=True):
        planned_bytes = int(record["bytes_charged"])
        actual_bytes = int(runtime["actual_bytes"])
        intended_stream_payload_bytes += planned_bytes
        actual_stream_payload_bytes += actual_bytes
        per_stream.append(
            {
                "name": record["name"],
                "stream_id": record["stream_id"],
                "bytes_charged": planned_bytes,
                "bytes_charged_source": record["bytes_charged_source"],
                "archive_actual_bytes": actual_bytes,
                "archive_actual_bytes_source": "jcsp_container_member_payload",
                "delta_bytes": actual_bytes - planned_bytes,
                "bytes_charged_reconciled": actual_bytes == planned_bytes,
            }
        )

    actual_member_payload_bytes = int(archive_contract["member_bytes"])
    actual_charged_member_compressed_bytes = int(
        archive_contract["member_compress_size"]
    )
    actual_charged_archive_bytes = int(archive_contract["archive_bytes"])
    member_container_overhead_bytes = (
        actual_member_payload_bytes - actual_stream_payload_bytes
    )
    if member_container_overhead_bytes < 0:
        raise ValueError(
            "JCSP archive member reports fewer bytes than its contained stream "
            "payloads"
        )
    zip_wrapper_overhead_bytes = (
        actual_charged_archive_bytes - actual_charged_member_compressed_bytes
    )
    if zip_wrapper_overhead_bytes < 0:
        raise ValueError(
            "JCSP archive reports fewer charged bytes than the compressed member"
        )

    payload: dict[str, Any] = {
        "schema": JCSP_STREAM_ARCHIVE_BYTE_RECONCILIATION_SCHEMA,
        "score_claim": False,
        "stream_count": len(per_stream),
        "stream_payload_bytes_reconciled": all(
            row["bytes_charged_reconciled"] for row in per_stream
        ),
        "intended_stream_payload_bytes": intended_stream_payload_bytes,
        "actual_stream_payload_bytes": actual_stream_payload_bytes,
        "actual_member_payload_bytes": actual_member_payload_bytes,
        "actual_charged_member_compressed_bytes": (
            actual_charged_member_compressed_bytes
        ),
        "actual_charged_archive_bytes": actual_charged_archive_bytes,
        "member_container_overhead_bytes": member_container_overhead_bytes,
        "zip_wrapper_overhead_bytes": zip_wrapper_overhead_bytes,
        "member_compress_type": int(archive_contract["member_compress_type"]),
        "member_name": archive_contract["member_name"],
        "validated_archive_members": list(archive_contract["archive_members"]),
        "single_member_no_sidecars": (
            list(archive_contract["archive_members"])
            == [archive_contract["member_name"]]
        ),
        "per_stream": per_stream,
        "dispatch_blocker_if_unreconciled": (
            "stream_bytes_charged_reconciliation_missing"
        ),
    }
    payload["reconciliation_sha256"] = _canonical_json_sha256(payload)
    return payload


def jcsp_model_stream_archive_readiness(
    *,
    streams: Iterable[JCSPTensorStreamSpec],
    archive_bytes: bytes,
    member_name: str = JCSP_ARCHIVE_MEMBER_NAME,
) -> dict[str, Any]:
    """Tie model stream metadata to a byte-closed JCSP archive member.

    This is still a scoreless planning/readiness artifact. It requires the ZIP
    archive to contain exactly one deterministic JCSP member, rejects sidecar
    members through the runtime archive contract, and then verifies that the
    container stream order/names reconcile with the deterministic
    ``model_to_jcsp_streams`` manifest before reporting per-stream actual
    payload bytes.
    """

    stream_manifest = jcsp_stream_specs_manifest(streams)
    archive_contract = load_jcsp_archive_member_for_runtime(
        archive_bytes=archive_bytes,
        member_name=member_name,
        require_single_member=True,
    )

    stream_records = list(stream_manifest["streams"])
    runtime_streams = list(archive_contract["jcsp_runtime_parity"]["streams"])
    manifest_names = [str(record["name"]) for record in stream_records]
    runtime_names = [str(record["name"]) for record in runtime_streams]
    if runtime_names != manifest_names:
        raise ValueError(
            "JCSP model stream archive readiness requires archive stream order "
            "to match the deterministic model stream manifest; "
            f"manifest={manifest_names!r} archive={runtime_names!r}"
        )

    closed_by_archive = {
        "qint_or_exact_wire_stream_missing",
        "byte_closed_archive_member_missing",
        "charged_archive_member_missing",
        "runtime_loader_parity_missing",
    }
    byte_reconciliation = _jcsp_stream_archive_byte_reconciliation(
        stream_records=stream_records,
        runtime_streams=runtime_streams,
        archive_contract=archive_contract,
    )
    dispatch_blockers: list[str] = [
        str(blocker)
        for blocker in archive_contract.get("dispatch_blockers", ())
    ]
    if not byte_reconciliation["stream_payload_bytes_reconciled"]:
        dispatch_blockers.append("stream_bytes_charged_reconciliation_missing")
    stream_readiness: list[dict[str, Any]] = []
    for record, runtime in zip(stream_records, runtime_streams, strict=True):
        planned_kind = int(record["codec_kind"])
        actual_kind = int(runtime["codec_kind"])
        codec_reconciled, codec_note = _codec_kind_reconciles_model_plan(
            planned=planned_kind,
            actual=actual_kind,
        )
        if not codec_reconciled:
            raise ValueError(
                "JCSP archive stream codec kind does not match model stream "
                f"plan for {record['name']!r}: planned={planned_kind} "
                f"actual={actual_kind}"
            )

        actual_bytes = int(runtime["actual_bytes"])
        planned_bytes = int(record["bytes_charged"])
        stream_blockers = [
            str(blocker)
            for blocker in record["dispatch_blockers"]
            if str(blocker) not in closed_by_archive
        ]
        bytes_reconciled = actual_bytes == planned_bytes
        if not bytes_reconciled:
            stream_blockers.append("stream_bytes_charged_reconciliation_missing")

        dispatch_blockers.extend(stream_blockers)
        stream_readiness.append(
            {
                "name": record["name"],
                "stream_id": record["stream_id"],
                "planned_codec_kind": planned_kind,
                "archive_codec_kind": actual_kind,
                "codec_kind_reconciliation": codec_note,
                "raw_bytes": int(record["raw_bytes"]),
                "byte_estimate": int(record["byte_estimate"]),
                "byte_estimate_source": record["byte_estimate_source"],
                "bytes_charged": planned_bytes,
                "bytes_charged_source": record["bytes_charged_source"],
                "archive_actual_bytes": actual_bytes,
                "archive_actual_bytes_source": "jcsp_container_member_payload",
                "bytes_charged_reconciled": bytes_reconciled,
                "score_per_byte_marginal": float(
                    record["score_per_byte_marginal"]
                ),
                "score_marginal_source": record["score_marginal_source"],
                "score_marginal_evidence": record["score_marginal_evidence"],
                "scorer_term_targeted": record["scorer_term_targeted"],
                "dispatch_blockers": list(dict.fromkeys(stream_blockers)),
            }
        )

    payload: dict[str, Any] = {
        "schema": JCSP_MODEL_STREAM_ARCHIVE_READINESS_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_runtime_loader": True,
        "ready_for_submission_runtime_consumption": False,
        "ready_for_exact_eval_dispatch": False,
        "stream_manifest_sha256": stream_manifest["manifest_sha256"],
        "archive_contract_schema": archive_contract["schema"],
        "archive_sha256": archive_contract["archive_sha256"],
        "archive_bytes": int(archive_contract["archive_bytes"]),
        "member_name": archive_contract["member_name"],
        "member_sha256": archive_contract["member_sha256"],
        "member_bytes": int(archive_contract["member_bytes"]),
        "byte_closed_archive_member": True,
        "single_member_no_sidecars": (
            archive_contract["archive_members"] == [member_name]
        ),
        "runtime_stream_order_matches_manifest": True,
        "stream_archive_byte_reconciliation": byte_reconciliation,
        "runtime_consumption_contract": archive_contract[
            "runtime_consumption_contract"
        ],
        "stream_count": len(stream_readiness),
        "streams": stream_readiness,
        "sidecar_policy": {
            "score_affecting_sidecars_allowed": False,
            "validated_archive_members": archive_contract["archive_members"],
        },
        "dispatch_blockers": list(dict.fromkeys(dispatch_blockers)),
    }
    payload["readiness_sha256"] = _canonical_json_sha256(payload)
    return payload


# ────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────────


def run_joint_codec_stack(
    *,
    streams: list[StreamSource],
    byte_budget: float,
    admm_max_iters: int = 50,
    admm_rho_init: float = 1.0,
    admm_kkt_tol: float = 5e-2,
    verbose: bool = False,
) -> JointCodecStackResult:
    """Compose the canonical codec stack on a list of streams.

    Args:
        streams: list of ``StreamSource``. Required keyword.
        byte_budget: total JCSP container byte budget, including stream
            payloads and JCSP metadata. Required keyword.
        admm_max_iters: ADMM coordinator iteration cap.
        admm_rho_init: ADMM initial penalty.
        admm_kkt_tol: KKT waterline tolerance.
        verbose: print ADMM trace.

    Returns:
        ``JointCodecStackResult`` with per-stream provenance + bundled JCSP
        container bytes.

    Strict-scorer-rule: this function does not load any neural-net scorer.
    Score-cost surfaces come from ``StreamSource.score_per_byte_marginal``,
    which the caller pre-computes via the frontier-sampler.
    """
    if not streams:
        raise ValueError("run_joint_codec_stack: at least one stream required")
    if byte_budget <= 0:
        raise ValueError(
            f"run_joint_codec_stack: byte_budget must be > 0, got {byte_budget}"
        )

    proximals: list[_CodecProximal] = [_CodecProximal(src=s) for s in streams]

    cfg = JointADMMConfig(
        rho_init=admm_rho_init,
        max_iters=max(2, admm_max_iters),
        byte_budget=byte_budget,
        kkt_waterline_tol=admm_kkt_tol,
        verbose=verbose,
    )

    admm_result: AdmmResult = run_admm(streams=proximals, cfg=cfg)

    stream_results: list[StackStreamResult] = []
    for prox, target in zip(
        proximals,
        admm_result.final_bytes_per_stream,
        strict=True,
    ):
        # Roundtrip-verify each stream so a malformed bytes-out cannot ship.
        payload = prox.cached_payload
        _verify_stream_roundtrip(prox.src, payload)
        stream_results.append(
            StackStreamResult(
                name=prox.src.name,
                codec_kind=prox.actual_codec_kind,
                admm_bytes_target=float(target),
                actual_bytes=len(payload),
                score_delta=float(
                    prox.src.score_per_byte_marginal * len(payload)
                ),
                marginal=float(prox.src.score_per_byte_marginal),
                payload=payload,
            )
        )

    container = _pack_jcsp_container(
        streams=stream_results,
        waterline_kkt_residual=admm_result.waterline_kkt_residual,
        iters=admm_result.iters,
        converged=admm_result.converged,
    )
    total_bytes = len(container)
    if total_bytes > byte_budget:
        raise ValueError(
            f"run_joint_codec_stack: JCSP container bytes {total_bytes} exceed "
            f"byte_budget {byte_budget} after metadata overhead"
        )
    return JointCodecStackResult(
        converged=admm_result.converged,
        iters=admm_result.iters,
        total_bytes=total_bytes,
        waterline_kkt_residual=admm_result.waterline_kkt_residual,
        streams=stream_results,
        container_bytes=container,
        admm_history_len=len(admm_result.history),
    )


def _verify_stream_roundtrip(src: StreamSource, payload: bytes) -> None:
    """Roundtrip check: decode the payload and assert array_equal with src.qints.

    Skipped for RAW_PASSTHROUGH (caller is responsible for the externally-
    encoded format's roundtrip integrity, e.g., AV1).
    """
    if src.codec_kind == KIND_ARITHMETIC_STATIC:
        decoded = decode_qints_arithmetic(payload, expected_dtype=src.qints.dtype)
        if not np.array_equal(decoded, src.qints):
            raise AssertionError(
                f"_verify_stream_roundtrip: arithmetic-static roundtrip "
                f"failed for stream {src.name!r}"
            )
        return
    if src.codec_kind == KIND_BALLE_HYPERPRIOR:
        # Auto-select may have produced an AQv1 payload (static-wins). Detect
        # by magic byte and dispatch.
        if payload[:4] in _ARITHMETIC_PAYLOAD_MAGICS:
            decoded = decode_qints_arithmetic(
                payload, expected_dtype=src.qints.dtype
            )
        elif payload[:4] in _BALLE_PAYLOAD_MAGICS:
            decoded = decode_qints_balle(
                blob=payload, expected_dtype=src.qints.dtype
            )
        else:
            raise AssertionError(
                f"_verify_stream_roundtrip: unknown payload magic "
                f"{payload[:4]!r} for BHv1-class stream {src.name!r}"
            )
        if not np.array_equal(decoded, src.qints):
            raise AssertionError(
                f"_verify_stream_roundtrip: BHv1/AQv1 roundtrip failed for "
                f"stream {src.name!r}"
            )
        return
    if src.codec_kind == KIND_RAW_PASSTHROUGH:
        # Externally-encoded payload (e.g., AV1) — roundtrip is the caller's
        # responsibility. Verify only that the carried bytes match the
        # passthrough bytes (sanity check that the orchestrator did not
        # corrupt them).
        assert src.raw_passthrough_bytes is not None
        if payload != src.raw_passthrough_bytes:
            raise AssertionError(
                f"_verify_stream_roundtrip: RAW_PASSTHROUGH bytes mutated "
                f"for stream {src.name!r}"
            )
        return
    raise AssertionError(
        f"_verify_stream_roundtrip: unhandled codec_kind {src.codec_kind}"
    )


# ────────────────────────────────────────────────────────────────────────────
# Sequential-allocation baseline (for empirical comparison against ADMM)
# ────────────────────────────────────────────────────────────────────────────


def run_sequential_codec_stack(
    *, streams: list[StreamSource]
) -> JointCodecStackResult:
    """Encode each stream with its codec, no joint allocation.

    The "naive" baseline: every stream runs its own codec independently with
    no global byte budget. This is what every standalone codec lane in the
    repo does today. The empirical comparison ADMM-vs-sequential is the
    paradigm-γ delta we report.

    Returns:
        ``JointCodecStackResult`` with no ADMM convergence (single-pass).
    """
    if not streams:
        raise ValueError(
            "run_sequential_codec_stack: at least one stream required"
        )
    proximals = [_CodecProximal(src=s) for s in streams]
    stream_results: list[StackStreamResult] = []
    for prox in proximals:
        payload = prox.cached_payload
        _verify_stream_roundtrip(prox.src, payload)
        stream_results.append(
            StackStreamResult(
                name=prox.src.name,
                codec_kind=prox.actual_codec_kind,
                admm_bytes_target=float(len(payload)),  # no ADMM target
                actual_bytes=len(payload),
                score_delta=float(
                    prox.src.score_per_byte_marginal * len(payload)
                ),
                marginal=float(prox.src.score_per_byte_marginal),
                payload=payload,
            )
        )
    container = _pack_jcsp_container(
        streams=stream_results,
        waterline_kkt_residual=0.0,  # sequential is not on the KKT manifold
        iters=0,
        converged=False,
    )
    total_bytes = len(container)
    return JointCodecStackResult(
        converged=False,
        iters=0,
        total_bytes=total_bytes,
        waterline_kkt_residual=0.0,
        streams=stream_results,
        container_bytes=container,
    )


__all__ = [
    "JCSP_ARCHIVE_MEMBER_CONTRACT_SCHEMA",
    "JCSP_ARCHIVE_MEMBER_NAME",
    "JCSP_LOCAL_ARCHIVE_MEMBER_CONTRACT_SCHEMA",
    "JCSP_LOCAL_SKELETON_RUNTIME_BLOCKER",
    "JCSP_LOCAL_SKELETON_SCHEMA",
    "JCSP_MODEL_STREAM_ARCHIVE_READINESS_SCHEMA",
    "JCSP_RUNTIME_AQ_RAWVIDEO_ADAPTER_ID",
    "JCSP_RUNTIME_AQ_RAWVIDEO_STREAM_KIND",
    "JCSP_RUNTIME_RAW_OUTPUT_CONSUMER_READINESS_SCHEMA",
    "JCSP_RUNTIME_RAW_OUTPUT_PARITY_CONTRACT_SCHEMA",
    "JCSP_RUNTIME_RAW_OUTPUT_PARITY_PROOF_SCHEMA",
    "JCSP_RUNTIME_REAL_AQ_PREFLIGHT_ADAPTER_ID",
    "JCSP_STREAM_ARCHIVE_BYTE_RECONCILIATION_SCHEMA",
    "JCSP_STREAM_METADATA_SCHEMA",
    "JCSP_SUBMISSION_RUNTIME_BRIDGE_PROBE_SCHEMA",
    "JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER",
    "JCSP_SUBMISSION_RUNTIME_CONSUMPTION_SCHEMA",
    "JCSP_SUBMISSION_RUNTIME_OUTPUT_CONTRACT_SCHEMA",
    "JCSP_SUBMISSION_RUNTIME_OUTPUT_PARITY_BLOCKER",
    "KIND_ARITHMETIC_STATIC",
    "KIND_BALLE_HYPERPRIOR",
    "KIND_RAW_PASSTHROUGH",
    "JCSPTensorStreamSpec",
    "JointCodecStackResult",
    "StackStreamResult",
    "StreamSource",
    "build_jcsp_archive_member",
    "build_jcsp_local_skeleton_archive_member",
    "build_jcsp_noop_archive_fixture",
    "jcsp_model_stream_archive_readiness",
    "jcsp_stream_specs_manifest",
    "load_jcsp_archive_member_for_runtime",
    "load_jcsp_local_skeleton_archive_member",
    "model_to_jcsp_streams",
    "pack_jcsp_local_skeleton_container",
    "run_joint_codec_stack",
    "run_sequential_codec_stack",
    "unpack_jcsp_container",
    "unpack_jcsp_local_skeleton_container",
    "validate_jcsp_container_runtime_parity",
]
