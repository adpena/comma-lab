"""Bridge ``JCSPTensorStreamSpec`` planning specs to runnable ``StreamSource``.

``model_to_jcsp_streams`` already decomposes a model into per-tensor planning
specs (codec_kind / shape / dtype / byte estimates). The ADMM coordinator and
sequential codec stack consume ``StreamSource`` objects, which require
pre-quantized integer symbols (``qints``, ``num_symbols``, ``offset``).

This module bridges those two stages with a deterministic symmetric
round-to-nearest int8 quantizer, plus a model-walking convenience helper.

Exposed surface:

* ``quantize_tensor_symmetric(tensor, *, num_levels=15)`` — returns
  ``(qints, num_symbols, offset, scale)``.
* ``tensor_to_stream_source(tensor, *, name, codec_kind, ...)`` — single-
  tensor wrapper that builds a validated ``StreamSource``.
* ``model_to_stream_sources(model, ...)`` — walks ``model_to_jcsp_streams``
  + per-tensor quantization, returning ``(streams, specs)`` aligned by index.

The helper makes NO score claims, never loads scorers, and never invents
``raw_passthrough_bytes`` for wet streams — those must be supplied by the
caller. RAW_PASSTHROUGH streams without supplied bytes raise immediately
(no silent-no-op trap).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tac.joint_codec_stack_orchestrator import (
    JCSP_ARCHIVE_MEMBER_NAME,
    JCSP_LOCAL_SKELETON_RUNTIME_BLOCKER,
    JCSP_LOCAL_SKELETON_SCHEMA,
    JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
    KIND_ARITHMETIC_STATIC,
    KIND_BALLE_HYPERPRIOR,
    KIND_RAW_PASSTHROUGH,
    JCSPTensorStreamSpec,
    StreamSource,
    build_jcsp_archive_member,
    build_jcsp_local_skeleton_archive_member,
    jcsp_model_stream_archive_readiness,
    jcsp_stream_specs_manifest,
    load_jcsp_archive_member_for_runtime,
    model_to_jcsp_streams,
    run_sequential_codec_stack,
)

if TYPE_CHECKING:
    from tac.balle_hyperprior_codec import BalleHyperpriorCodec


JCSP_STREAM_SOURCE_DRY_RUN_SCHEMA = "jcsp_stream_source_dry_run_v1"
JCSP_STREAM_SOURCE_LOCAL_ARCHIVE_MEMBER_SCHEMA = JCSP_LOCAL_SKELETON_SCHEMA
JCSP_STREAM_SOURCE_ARCHIVE_MEMBER_SCHEMA = (
    "jcsp_stream_source_archive_member_contract_v1"
)


def _coerce_to_float32_numpy(tensor: Any) -> np.ndarray:
    """Coerce a torch tensor / numpy array / array-like to a flat fp32 array.

    Torch tensors are detached and moved to CPU first. No device default is
    applied (per CLAUDE.md forbidden-default-to-mps rule); the caller owns
    the source-device decision.
    """
    if (
        hasattr(tensor, "detach")
        and hasattr(tensor, "cpu")
        and hasattr(tensor, "numpy")
    ):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    return np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)


def quantize_tensor_symmetric(
    tensor: Any,
    *,
    num_levels: int = 15,
) -> tuple[np.ndarray, int, int, float]:
    """Symmetric round-to-nearest quantization to int8 qints.

    Args:
        tensor: torch tensor, numpy array, or array-like.
        num_levels: alphabet size; must be odd and >= 3. Default 15 maps
            to qints in ``[-7, 7]`` with offset 7.

    Returns:
        ``(qints, num_symbols, offset, scale)`` where:

        * ``qints`` is a flat ``int8`` array in ``[-k, k]`` with
          ``k = (num_levels - 1) // 2``.
        * ``num_symbols == num_levels``.
        * ``offset == k`` so symbol indices ``qints + offset`` lie in
          ``[0, num_symbols)``.
        * ``scale`` is the per-tensor scale such that
          ``tensor ≈ qints * scale``.

    Raises:
        ValueError: if ``num_levels`` is not an odd integer >= 3, or if the
            tensor is empty.
    """
    if num_levels < 3 or num_levels % 2 == 0:
        raise ValueError(
            f"quantize_tensor_symmetric: num_levels must be odd and >=3 "
            f"(got {num_levels})"
        )
    arr = _coerce_to_float32_numpy(tensor)
    if arr.size == 0:
        raise ValueError("quantize_tensor_symmetric: empty tensor")
    if not np.all(np.isfinite(arr)):
        raise ValueError("quantize_tensor_symmetric: tensor has nan/inf")
    k = (num_levels - 1) // 2
    abs_max = float(np.max(np.abs(arr)))
    scale = (abs_max / k) if abs_max > 0.0 else 1.0
    qints = np.clip(np.round(arr / scale), -k, k).astype(np.int8)
    return qints, int(num_levels), int(k), float(scale)


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _reject_duplicate_json_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _normalize_score_marginals_mapping(
    raw: Any,
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} must be a mapping of stream name to marginal")
    out: dict[str, Any] = {}
    for name_raw, value in raw.items():
        if not isinstance(name_raw, str) or not name_raw:
            raise ValueError(f"{context} has a non-string or empty stream name")
        name = str(name_raw)
        if isinstance(value, bool):
            raise ValueError(f"{context}[{name!r}] must not be a bool")
        if isinstance(value, (int, float)):
            out[name] = float(value)
        elif isinstance(value, Mapping):
            out[name] = dict(value)
        else:
            raise ValueError(
                f"{context}[{name!r}] must be a float or mapping, got "
                f"{type(value).__name__}"
            )
    return out


def _score_marginals_from_stream_rows(raw_streams: Any) -> dict[str, Any]:
    if not isinstance(raw_streams, list):
        raise ValueError("score marginals streams field must be a list")
    out: dict[str, Any] = {}
    for index, row in enumerate(raw_streams):
        if not isinstance(row, Mapping):
            raise ValueError(f"score marginals streams[{index}] must be a mapping")
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"score marginals streams[{index}] has invalid name {name!r}"
            )
        if name in out:
            raise ValueError(f"duplicate stream marginal row for {name!r}")
        if "score_per_byte_marginal" not in row:
            raise ValueError(
                f"score marginals streams[{index}] for {name!r} is missing "
                "score_per_byte_marginal"
            )
        tags_raw = row.get("constraint_tags", ())
        if tags_raw is None:
            tags: list[str] = []
        elif isinstance(tags_raw, list):
            tags = [str(tag) for tag in tags_raw]
        else:
            raise ValueError(
                f"score marginals streams[{index}] constraint_tags must be a list"
            )
        out[name] = {
            "score_per_byte_marginal": row["score_per_byte_marginal"],
            "source": row.get("score_marginal_source", "jcsp_stream_manifest"),
            "evidence_grade": row.get("score_marginal_evidence", "empirical"),
            "scorer_term_targeted": row.get("scorer_term_targeted", "joint"),
            "constraint_tags": tags,
        }
    return _normalize_score_marginals_mapping(out, context="score marginals")


def load_jcsp_score_marginals(path: str | Path) -> dict[str, Any]:
    """Load a deterministic JSON JCSP score-marginals artifact.

    Supported shapes:
    * ``{"tensor.name": 1.0e-6, ...}``
    * ``{"score_marginals": {"tensor.name": 1.0e-6, ...}}``
    * a JCSP stream manifest with ``streams[*].score_per_byte_marginal`` rows

    The loader intentionally does not use pickle / ``torch.load``. A score
    marginal is small structured metadata, and JSON with duplicate-key
    rejection is the safest deterministic contract for a dry run.
    """
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"JCSP score marginals artifact does not exist: {artifact_path}"
        )
    raw_bytes = artifact_path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "JCSP score marginals artifact must be UTF-8 JSON, not a binary "
            f"payload: {artifact_path}"
        ) from exc
    try:
        data = json.loads(text, object_pairs_hook=_reject_duplicate_json_object_pairs)
    except ValueError as exc:
        raise ValueError(
            f"JCSP score marginals artifact is not valid deterministic JSON: "
            f"{artifact_path}: {exc}"
        ) from exc
    if not isinstance(data, Mapping):
        raise ValueError("JCSP score marginals artifact top level must be a mapping")
    if "score_marginals" in data:
        return _normalize_score_marginals_mapping(
            data["score_marginals"],
            context="score_marginals",
        )
    if "tensor_score_marginals" in data:
        return _normalize_score_marginals_mapping(
            data["tensor_score_marginals"],
            context="tensor_score_marginals",
        )
    if "streams" in data:
        return _score_marginals_from_stream_rows(data["streams"])
    return _normalize_score_marginals_mapping(data, context="score_marginals")


def tensor_to_stream_source(
    tensor: Any,
    *,
    name: str,
    codec_kind: int,
    score_per_byte_marginal: float,
    num_levels: int = 15,
    balle_codec: BalleHyperpriorCodec | None = None,
    raw_passthrough_bytes: bytes | None = None,
) -> StreamSource:
    """Quantize a tensor and wrap it as a ``StreamSource``.

    * ``KIND_ARITHMETIC_STATIC`` and ``KIND_BALLE_HYPERPRIOR``: the tensor
      is quantized symmetrically; the resulting qints/num_symbols/offset
      flow into the ``StreamSource``.
    * ``KIND_RAW_PASSTHROUGH``: ``tensor`` is ignored;
      ``raw_passthrough_bytes`` is required.

    Raises:
        ValueError: on invalid ``codec_kind``, non-finite marginal, missing
            ``raw_passthrough_bytes`` for RAW, or missing ``balle_codec``
            for BALLE.
    """
    if codec_kind not in (
        KIND_ARITHMETIC_STATIC,
        KIND_BALLE_HYPERPRIOR,
        KIND_RAW_PASSTHROUGH,
    ):
        raise ValueError(
            f"tensor_to_stream_source: invalid codec_kind {codec_kind}"
        )
    if not np.isfinite(score_per_byte_marginal):
        raise ValueError(
            f"tensor_to_stream_source: score_per_byte_marginal must be "
            f"finite (got {score_per_byte_marginal})"
        )
    if codec_kind == KIND_RAW_PASSTHROUGH:
        if raw_passthrough_bytes is None:
            raise ValueError(
                "tensor_to_stream_source: KIND_RAW_PASSTHROUGH requires "
                "raw_passthrough_bytes"
            )
        return StreamSource(
            name=name,
            qints=np.zeros(0, dtype=np.int8),
            num_symbols=2,
            offset=0,
            codec_kind=codec_kind,
            raw_passthrough_bytes=bytes(raw_passthrough_bytes),
            score_per_byte_marginal=float(score_per_byte_marginal),
        )
    if codec_kind == KIND_BALLE_HYPERPRIOR and balle_codec is None:
        raise ValueError(
            "tensor_to_stream_source: KIND_BALLE_HYPERPRIOR requires "
            "balle_codec"
        )
    qints, num_symbols, offset, _scale = quantize_tensor_symmetric(
        tensor, num_levels=num_levels
    )
    return StreamSource(
        name=name,
        qints=qints,
        num_symbols=num_symbols,
        offset=offset,
        codec_kind=codec_kind,
        balle_codec=balle_codec,
        score_per_byte_marginal=float(score_per_byte_marginal),
    )


def model_to_stream_sources(
    model: Any,
    *,
    score_marginals: Mapping[str, float],
    num_levels: int = 15,
    codec_overrides: Mapping[str, int] | None = None,
    balle_codecs: Mapping[str, BalleHyperpriorCodec] | None = None,
    raw_passthrough_bytes_by_name: Mapping[str, bytes] | None = None,
    wet_streams: Iterable[str] | None = None,
    include_buffers: bool = True,
) -> tuple[list[StreamSource], list[JCSPTensorStreamSpec]]:
    """Decompose a model into ``list[StreamSource]`` + parallel planning specs.

    Walks ``model_to_jcsp_streams`` to get the planning specs, then quantizes
    each tensor (or routes pre-encoded payloads for RAW_PASSTHROUGH) into a
    ``StreamSource``. Returns ``(streams, specs)`` where ``streams[i]``
    corresponds to ``specs[i]``.

    Caller is responsible for supplying pre-encoded bytes for every wet /
    RAW_PASSTHROUGH stream — the helper does NOT invent payloads.

    Raises:
        ValueError: if a stream name from the planning specs is missing
            from the model state_dict, if a RAW stream has no entry in
            ``raw_passthrough_bytes_by_name``, or if a BALLE stream has no
            entry in ``balle_codecs``.
    """
    specs = model_to_jcsp_streams(
        model,
        score_marginals=score_marginals,
        codec_overrides=codec_overrides,
        wet_streams=wet_streams,
        include_buffers=include_buffers,
    )
    raw_payloads = dict(raw_passthrough_bytes_by_name or {})
    balle_by_name = dict(balle_codecs or {})
    score_by_name = {
        spec.name: float(spec.score_per_byte_marginal) for spec in specs
    }

    state_dict_iter = model.state_dict().items() if hasattr(model, "state_dict") else dict(model).items()
    name_to_tensor: dict[str, Any] = {str(k): v for k, v in state_dict_iter}

    streams: list[StreamSource] = []
    for spec in specs:
        name = spec.name
        codec_kind = int(spec.codec_kind)
        marginal = score_by_name.get(name, 0.0)
        if codec_kind == KIND_RAW_PASSTHROUGH:
            payload = raw_payloads.get(name)
            if payload is None:
                raise ValueError(
                    f"model_to_stream_sources: RAW_PASSTHROUGH stream "
                    f"{name!r} requires pre-encoded bytes in "
                    f"raw_passthrough_bytes_by_name"
                )
            streams.append(
                StreamSource(
                    name=name,
                    qints=np.zeros(0, dtype=np.int8),
                    num_symbols=2,
                    offset=0,
                    codec_kind=codec_kind,
                    raw_passthrough_bytes=bytes(payload),
                    score_per_byte_marginal=marginal,
                )
            )
            continue
        tensor = name_to_tensor.get(name)
        if tensor is None:
            raise ValueError(
                f"model_to_stream_sources: stream {name!r} from planning "
                f"specs not present in model state_dict"
            )
        balle_codec = (
            balle_by_name.get(name)
            if codec_kind == KIND_BALLE_HYPERPRIOR
            else None
        )
        if codec_kind == KIND_BALLE_HYPERPRIOR and balle_codec is None:
            raise ValueError(
                f"model_to_stream_sources: BALLE stream {name!r} requires "
                f"balle_codecs[{name!r}]"
            )
        streams.append(
            tensor_to_stream_source(
                tensor,
                name=name,
                codec_kind=codec_kind,
                score_per_byte_marginal=marginal,
                num_levels=num_levels,
                balle_codec=balle_codec,
            )
        )
    return streams, specs


def _normalize_raw_payloads(
    raw_passthrough_bytes_by_name: Mapping[str, bytes] | None,
) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for name_raw, payload in (raw_passthrough_bytes_by_name or {}).items():
        name = str(name_raw)
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValueError(
                f"raw passthrough payload for {name!r} must be bytes-like"
            )
        out[name] = bytes(payload)
    return out


def _state_dict_by_name(model: Any) -> dict[str, Any]:
    state_dict_iter = (
        model.state_dict().items()
        if hasattr(model, "state_dict")
        else dict(model).items()
    )
    return {str(k): v for k, v in state_dict_iter}


def jcsp_stream_source_dry_run_metadata(
    model: Any,
    *,
    score_marginals_path: str | Path,
    num_levels: int = 15,
    codec_overrides: Mapping[str, int] | None = None,
    raw_passthrough_bytes_by_name: Mapping[str, bytes] | None = None,
    wet_streams: Iterable[str] | None = None,
    include_buffers: bool = True,
) -> dict[str, Any]:
    """Build deterministic metadata for future ``StreamSource`` objects.

    This is a dry-run/runtime-integration helper only. It loads cached
    score-marginal metadata, decomposes the model into JCSP tensor specs, and
    records the fields needed to instantiate ``StreamSource`` objects later.
    It does not encode streams, build a JCSP container, write archive bytes,
    load scorers, dispatch jobs, or claim a score.
    """
    artifact_path = Path(score_marginals_path)
    score_marginals = load_jcsp_score_marginals(artifact_path)
    specs = model_to_jcsp_streams(
        model,
        score_marginals=score_marginals,
        codec_overrides=codec_overrides,
        wet_streams=wet_streams,
        include_buffers=include_buffers,
    )
    spec_manifest = jcsp_stream_specs_manifest(specs)
    name_to_tensor = _state_dict_by_name(model)
    raw_payloads = _normalize_raw_payloads(raw_passthrough_bytes_by_name)

    records: list[dict[str, Any]] = []
    for spec in specs:
        raw_payload = raw_payloads.get(spec.name)
        if spec.codec_kind == KIND_RAW_PASSTHROUGH:
            if raw_payload is None:
                raise ValueError(
                    f"JCSP dry run: RAW_PASSTHROUGH stream {spec.name!r} "
                    "requires raw_passthrough_bytes_by_name"
                )
            qint_count = 0
            num_symbols = 2
            offset = 0
            qint_min = None
            qint_max = None
            quantization_scale = None
            raw_payload_bytes = len(raw_payload)
            raw_payload_sha256 = hashlib.sha256(raw_payload).hexdigest()
        else:
            if spec.name not in name_to_tensor:
                raise ValueError(
                    f"JCSP dry run: stream {spec.name!r} from planning specs "
                    "is missing from model state_dict"
                )
            qints, num_symbols, offset, scale = quantize_tensor_symmetric(
                name_to_tensor[spec.name],
                num_levels=num_levels,
            )
            qint_count = int(qints.size)
            qint_min = int(qints.min())
            qint_max = int(qints.max())
            quantization_scale = float(scale)
            raw_payload_bytes = 0
            raw_payload_sha256 = None
        dispatch_blockers = list(
            dict.fromkeys(
                [
                    *spec.dispatch_blockers,
                    "dry_run_only_no_archive_bytes_written",
                    "pipeline_dispatch_loop_not_wired",
                ]
            )
        )
        if spec.codec_kind == KIND_BALLE_HYPERPRIOR:
            dispatch_blockers.append("balle_codec_not_instantiated_in_dry_run")
        records.append(
            {
                "name": spec.name,
                "stream_id": spec.stream_id,
                "decomposition_index": int(spec.decomposition_index),
                "codec_kind": int(spec.codec_kind),
                "qint_count": qint_count,
                "qint_dtype": "int8",
                "qint_min": qint_min,
                "qint_max": qint_max,
                "num_symbols": int(num_symbols),
                "offset": int(offset),
                "quantization": "symmetric_round_to_nearest",
                "quantization_scale": quantization_scale,
                "raw_passthrough_bytes": raw_payload_bytes,
                "raw_passthrough_sha256": raw_payload_sha256,
                "score_per_byte_marginal": float(spec.score_per_byte_marginal),
                "score_marginal_source": spec.score_marginal_source,
                "score_marginal_evidence": spec.score_marginal_evidence,
                "scorer_term_targeted": spec.scorer_term_targeted,
                "stream_source_metadata_only": True,
                "dispatch_blockers": dispatch_blockers,
            }
        )

    raw_artifact = artifact_path.read_bytes()
    payload: dict[str, Any] = {
        "schema": JCSP_STREAM_SOURCE_DRY_RUN_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "archive_bytes_written": False,
        "container_bytes_built": False,
        "stream_source_objects_built": False,
        "stream_source_metadata_only": True,
        "score_marginals_artifact": {
            "path": str(artifact_path),
            "bytes": len(raw_artifact),
            "sha256": hashlib.sha256(raw_artifact).hexdigest(),
            "format": "deterministic_json",
            "marginal_count": len(score_marginals),
        },
        "stream_count": len(records),
        "streams": records,
        "spec_manifest_sha256": spec_manifest["manifest_sha256"],
        "promotion_blockers": [
            "dry_run_only_no_archive_bytes_written",
            "pipeline_dispatch_loop_not_wired",
            "runtime_loader_parity_missing",
            "exact_cuda_auth_eval_missing",
        ],
    }
    payload["manifest_sha256"] = _canonical_json_sha256(payload)
    return payload


def _preview_payload_record(
    payload: bytes,
    *,
    preview_bytes_per_stream: int,
    source_bytes_kind: str,
) -> dict[str, Any]:
    if preview_bytes_per_stream < 0:
        raise ValueError("preview_bytes_per_stream must be >= 0")
    preview = payload[:preview_bytes_per_stream]
    return {
        "source_bytes_kind": source_bytes_kind,
        "source_bytes": len(payload),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "preview_bytes": len(preview),
        "preview_sha256": hashlib.sha256(preview).hexdigest(),
        "preview_hex": preview.hex(),
        "preview_truncated": len(preview) < len(payload),
    }


def jcsp_stream_source_local_archive_member(
    model: Any,
    *,
    score_marginals_path: str | Path,
    num_levels: int = 15,
    preview_bytes_per_stream: int = 64,
    codec_overrides: Mapping[str, int] | None = None,
    raw_passthrough_bytes_by_name: Mapping[str, bytes] | None = None,
    wet_streams: Iterable[str] | None = None,
    include_buffers: bool = True,
    member_name: str = JCSP_ARCHIVE_MEMBER_NAME,
) -> tuple[bytes, dict[str, Any]]:
    """Build a byte-closed local JCSP skeleton archive member.

    The payload is a deterministic ``JCSK`` skeleton, not a runtime ``JCSP``
    container. It records score-marginal custody, stream-source metadata, and
    qint/raw prefix previews with byte counts and SHA-256s so pipeline runs can
    prove local archive-member closure before the real runtime consumer exists.
    It never claims a score and remains non-dispatchable by construction.
    """

    artifact_path = Path(score_marginals_path)
    score_marginals = load_jcsp_score_marginals(artifact_path)
    specs = model_to_jcsp_streams(
        model,
        score_marginals=score_marginals,
        codec_overrides=codec_overrides,
        wet_streams=wet_streams,
        include_buffers=include_buffers,
    )
    spec_manifest = jcsp_stream_specs_manifest(specs)
    name_to_tensor = _state_dict_by_name(model)
    raw_payloads = _normalize_raw_payloads(raw_passthrough_bytes_by_name)

    records: list[dict[str, Any]] = []
    total_source_bytes = 0
    total_preview_bytes = 0
    for spec in specs:
        stream_blockers = [
            *spec.dispatch_blockers,
            "local_skeleton_preview_only_not_runtime_payload",
            "full_codec_payload_not_encoded",
            JCSP_LOCAL_SKELETON_RUNTIME_BLOCKER,
            "strict_preflight_proof_missing",
            "exact_cuda_auth_eval_missing",
        ]
        if spec.codec_kind == KIND_BALLE_HYPERPRIOR:
            stream_blockers.append(
                "balle_codec_not_instantiated_for_local_skeleton"
            )

        if spec.codec_kind == KIND_RAW_PASSTHROUGH:
            raw_payload = raw_payloads.get(spec.name)
            if raw_payload is None:
                raise ValueError(
                    f"JCSP local skeleton: RAW_PASSTHROUGH stream "
                    f"{spec.name!r} requires raw_passthrough_bytes_by_name"
                )
            qint_count = 0
            qint_min = None
            qint_max = None
            quantization_scale = None
            preview_record = _preview_payload_record(
                raw_payload,
                preview_bytes_per_stream=preview_bytes_per_stream,
                source_bytes_kind="raw_passthrough_bytes_prefix",
            )
        else:
            if spec.name not in name_to_tensor:
                raise ValueError(
                    f"JCSP local skeleton: stream {spec.name!r} from planning "
                    "specs is missing from model state_dict"
                )
            qints, _num_symbols, _offset, scale = quantize_tensor_symmetric(
                name_to_tensor[spec.name],
                num_levels=num_levels,
            )
            qint_count = int(qints.size)
            qint_min = int(qints.min())
            qint_max = int(qints.max())
            quantization_scale = float(scale)
            preview_record = _preview_payload_record(
                qints.tobytes(),
                preview_bytes_per_stream=preview_bytes_per_stream,
                source_bytes_kind="quantized_qint_int8_prefix_not_codec_payload",
            )

        total_source_bytes += int(preview_record["source_bytes"])
        total_preview_bytes += int(preview_record["preview_bytes"])
        records.append(
            {
                "name": spec.name,
                "stream_id": spec.stream_id,
                "decomposition_index": int(spec.decomposition_index),
                "planned_codec_kind": int(spec.codec_kind),
                "tensor_shape": list(spec.tensor_shape),
                "tensor_dtype": spec.tensor_dtype,
                "num_elements": int(spec.num_elements),
                "raw_bytes": int(spec.raw_bytes),
                "byte_estimate": int(spec.byte_estimate),
                "byte_estimate_source": spec.byte_estimate_source,
                "bytes_charged": int(spec.bytes_charged),
                "bytes_charged_source": spec.bytes_charged_source,
                "qint_count": qint_count,
                "qint_dtype": "int8",
                "qint_min": qint_min,
                "qint_max": qint_max,
                "quantization": "symmetric_round_to_nearest",
                "quantization_scale": quantization_scale,
                "preview": preview_record,
                "score_per_byte_marginal": float(spec.score_per_byte_marginal),
                "score_marginal_source": spec.score_marginal_source,
                "score_marginal_evidence": spec.score_marginal_evidence,
                "scorer_term_targeted": spec.scorer_term_targeted,
                "local_preview_only": True,
                "runtime_payload_encoded": False,
                "dispatch_blockers": list(dict.fromkeys(stream_blockers)),
            }
        )

    raw_artifact = artifact_path.read_bytes()
    skeleton_manifest: dict[str, Any] = {
        "schema": JCSP_STREAM_SOURCE_LOCAL_ARCHIVE_MEMBER_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_runtime_loader": False,
        "ready_for_submission_runtime_consumption": False,
        "ready_for_exact_eval_dispatch": False,
        "archive_member_payload_kind": "jcsp_stream_source_local_skeleton",
        "archive_member_name": member_name,
        "container_bytes_built": True,
        "archive_bytes_written": True,
        "stream_source_objects_built": False,
        "runtime_payloads_encoded": False,
        "quantized_stream_previews_built": True,
        "score_marginals_artifact": {
            "path": str(artifact_path),
            "bytes": len(raw_artifact),
            "sha256": hashlib.sha256(raw_artifact).hexdigest(),
            "format": "deterministic_json",
            "marginal_count": len(score_marginals),
        },
        "stream_count": len(records),
        "streams": records,
        "stream_spec_manifest_sha256": spec_manifest["manifest_sha256"],
        "byte_manifest": {
            "source_stream_bytes": total_source_bytes,
            "encoded_preview_bytes": total_preview_bytes,
            "preview_bytes_per_stream": int(preview_bytes_per_stream),
            "member_bytes_known_after_zip_write": True,
        },
        "promotion_blockers": [
            "local_skeleton_preview_only_not_runtime_payload",
            "full_codec_payload_not_encoded",
            "runtime_loader_parity_missing",
            JCSP_LOCAL_SKELETON_RUNTIME_BLOCKER,
            JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
            "strict_preflight_proof_missing",
            "exact_cuda_auth_eval_missing",
        ],
    }
    return build_jcsp_local_skeleton_archive_member(
        manifest=skeleton_manifest,
        member_name=member_name,
    )


def _unique_strings(values: Iterable[object]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def jcsp_stream_source_archive_member(
    model: Any,
    *,
    score_marginals_path: str | Path,
    num_levels: int = 15,
    codec_overrides: Mapping[str, int] | None = None,
    balle_codecs: Mapping[str, BalleHyperpriorCodec] | None = None,
    raw_passthrough_bytes_by_name: Mapping[str, bytes] | None = None,
    wet_streams: Iterable[str] | None = None,
    include_buffers: bool = True,
    member_name: str = JCSP_ARCHIVE_MEMBER_NAME,
    archive_path_for_manifest: str | Path | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Build a byte-closed one-member ZIP carrying a real ``JCSP`` container.

    This is the runtime-loader-parity closure step after the local ``JCSK``
    preview. It quantizes/builds concrete ``StreamSource`` payloads, packs a
    real ``JCSP`` member, validates the archive through the runtime contract,
    and records per-stream charged bytes from the actual member payload. It
    still does not make a score claim or unlock dispatch: the submission
    runtime detects ``jcsp.bin`` but refuses consumption until a real stream
    consumer emits contest outputs.
    """

    artifact_path = Path(score_marginals_path)
    score_marginals = load_jcsp_score_marginals(artifact_path)
    streams, specs = model_to_stream_sources(
        model,
        score_marginals=score_marginals,
        num_levels=num_levels,
        codec_overrides=codec_overrides,
        balle_codecs=balle_codecs,
        raw_passthrough_bytes_by_name=raw_passthrough_bytes_by_name,
        wet_streams=wet_streams,
        include_buffers=include_buffers,
    )
    result = run_sequential_codec_stack(streams=streams)
    archive_bytes = build_jcsp_archive_member(
        container_bytes=result.container_bytes,
        member_name=member_name,
    )
    archive_contract = load_jcsp_archive_member_for_runtime(
        archive_bytes=archive_bytes,
        member_name=member_name,
        require_single_member=True,
    )
    archive_closed_specs = [
        replace(
            spec,
            bytes_charged=int(stream.actual_bytes),
            bytes_charged_source="jcsp_container_member_payload",
        )
        for spec, stream in zip(specs, result.streams, strict=True)
    ]
    readiness = jcsp_model_stream_archive_readiness(
        streams=archive_closed_specs,
        archive_bytes=archive_bytes,
        member_name=member_name,
    )
    raw_artifact = artifact_path.read_bytes()
    dispatch_blockers = _unique_strings(
        [
            *readiness["dispatch_blockers"],
            "strict_candidate_preflight_proof_missing",
            "no_lane_dispatch_claim",
            "exact_cuda_auth_eval_missing",
        ]
    )
    archive_path_text = (
        str(Path(archive_path_for_manifest))
        if archive_path_for_manifest is not None
        else ""
    )
    payload: dict[str, Any] = {
        "schema": JCSP_STREAM_SOURCE_ARCHIVE_MEMBER_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_runtime_loader": True,
        "ready_for_submission_runtime_consumption": False,
        "ready_for_exact_eval_dispatch": False,
        "archive_member_payload_kind": "jcsp_runtime_container",
        "archive_member_name": member_name,
        "container_bytes_built": True,
        "archive_bytes_written": True,
        "stream_source_objects_built": True,
        "runtime_payloads_encoded": True,
        "stream_count": len(streams),
        "score_marginals_artifact": {
            "path": str(artifact_path),
            "bytes": len(raw_artifact),
            "sha256": hashlib.sha256(raw_artifact).hexdigest(),
            "format": "deterministic_json",
            "marginal_count": len(score_marginals),
        },
        "stream_manifest_sha256": readiness["stream_manifest_sha256"],
        "archive_path": archive_path_text,
        "candidate_archive_path": archive_path_text,
        "archive_sha256": readiness["archive_sha256"],
        "archive_bytes": int(readiness["archive_bytes"]),
        "candidate_archive_sha256": readiness["archive_sha256"],
        "candidate_archive_bytes": int(readiness["archive_bytes"]),
        "member_name": readiness["member_name"],
        "member_sha256": readiness["member_sha256"],
        "member_bytes": int(readiness["member_bytes"]),
        "byte_closed_archive_member": True,
        "single_member_no_sidecars": readiness["single_member_no_sidecars"],
        "runtime_loader_parity": archive_contract["jcsp_runtime_parity"],
        "runtime_consumption_contract": readiness["runtime_consumption_contract"],
        "stream_archive_byte_reconciliation": readiness[
            "stream_archive_byte_reconciliation"
        ],
        "jcsp_model_stream_archive_readiness": readiness,
        "runtime": {
            "ready_for_runtime_loader": True,
            "ready_for_submission_runtime_consumption": False,
            "runtime_tree_sha256": None,
            "blockers": [
                JCSP_SUBMISSION_RUNTIME_CONSUMPTION_BLOCKER,
                "submission_runtime_stream_consumer_missing",
                "runtime_tree_sha256_missing_until_submission_consumer_lands",
            ],
        },
        "dispatch_blockers": dispatch_blockers,
    }
    payload["manifest_sha256"] = _canonical_json_sha256(payload)
    return archive_bytes, payload


__all__ = [
    "JCSP_STREAM_SOURCE_ARCHIVE_MEMBER_SCHEMA",
    "JCSP_STREAM_SOURCE_DRY_RUN_SCHEMA",
    "JCSP_STREAM_SOURCE_LOCAL_ARCHIVE_MEMBER_SCHEMA",
    "jcsp_stream_source_archive_member",
    "jcsp_stream_source_dry_run_metadata",
    "jcsp_stream_source_local_archive_member",
    "load_jcsp_score_marginals",
    "model_to_stream_sources",
    "quantize_tensor_symmetric",
    "tensor_to_stream_source",
]
