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

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from tac.joint_codec_stack_orchestrator import (
    KIND_ARITHMETIC_STATIC,
    KIND_BALLE_HYPERPRIOR,
    KIND_RAW_PASSTHROUGH,
    JCSPTensorStreamSpec,
    StreamSource,
    model_to_jcsp_streams,
)

if TYPE_CHECKING:
    from tac.balle_hyperprior_codec import BalleHyperpriorCodec


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
    score_by_name = {str(k): float(v) for k, v in score_marginals.items()}

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


__all__ = [
    "model_to_stream_sources",
    "quantize_tensor_symmetric",
    "tensor_to_stream_source",
]
