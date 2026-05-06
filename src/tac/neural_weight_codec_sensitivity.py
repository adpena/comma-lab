"""Lane J-NWCS: Sensitivity-aware Neural Weight Compression.

Composition of three existing techniques:

  * Lane J-NWC (``tac.neural_weight_codec``)
        Base VQ-VAE-style codec for blocks of weight elements.

  * Lane Ω-V2 (``tac.learnable_bit_quant``)
        Hessian / gradient-magnitude-based per-parameter sensitivity.

  * Lane W (``tac.learnable_pair_weights``)
        Hard-pair gradient signal: which (frame_t, frame_{t+1}) pairs
        are PoseNet-critical.

Why a NEW lane (vs hand-stacking the three at deploy time)?  Because the
codebook design changes when sensitivity is known up-front:

    * Per-block sensitivity (Hessian magnitude × hard-pair gradient norm)
      is computed against the Lane G v3 anchor renderer.
    * Block sensitivities are bucketed by quantile (default 4 buckets).
    * Each bucket gets its own VQ codebook size: high-sensitivity blocks
      get K=256 (8 bits/code), low-sensitivity blocks get K=4 (2 bits/code).
    * Total bytes/block is amortized across buckets so the average bits
      per weight stays inside the lane's rate budget.

This is **strictly Pareto-dominant** over uniform-K NWC: spending more
bits on the blocks that drive PoseNet/SegNet error and fewer on the
blocks that don't is exactly the geometry the scorer rewards.

Compose-with rules (per ``docs/stacking_architecture.md``):
    Slot:        renderer-encoder
    Stacks-with: any renderer-replacement output (Lane G v3, J-JBL, J-IMP)
    Exclusive-with: Lane J-NWC (J-NWCS supersedes), Lane F-V5 FP8
    Composes-with: Lane Ω-V2 per-weight bits (Ω-V2 first → J-NWCS quantizes
                   within the resulting bit budget)
    Required signal-sidecar: Lane W hard_pair_weights.npy
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.neural_weight_codec import (
    WeightCodec,
    WeightCodecConfig,
    tensor_to_blocks,
)


__all__ = [
    "NWCS_RENDERER_MAGIC",
    "NWCSRendererContainer",
    "NWCSRendererTensorEntry",
    "SensitivityAwareCodecConfig",
    "SensitivityAwareWeightCodec",
    "compute_per_block_sensitivity",
    "encode_with_variable_codebook",
    "decode_with_per_block_codebook",
    "export_nwcs_renderer_container",
    "is_nwcs_renderer_container",
    "load_nwcs_renderer_container",
]


# ── Renderer container format ─────────────────────────────────────────────

NWCS_RENDERER_MAGIC = b"NWCS1\0\0\0"
_NWCS_RENDERER_VERSION = 1
# Signed int64 length fields make corrupt negative lengths reject explicitly.
_NWCS_RENDERER_LEN = struct.Struct("<q")
_NWCS_MAX_HEADER_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class NWCSRendererTensorEntry:
    """One encoded tensor inside an NWCS renderer container."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    original_dtype: str
    block_metadata: dict[str, Any]
    blob: bytes

    @classmethod
    def from_tensor_blob(
        cls,
        name: str,
        tensor: torch.Tensor,
        blob: bytes | bytearray | memoryview,
        *,
        block_size: int,
        codebook_sizes: list[int] | tuple[int, ...] | None = None,
        dtype: str | None = None,
        original_dtype: str | None = None,
        block_metadata: dict[str, Any] | None = None,
    ) -> "NWCSRendererTensorEntry":
        """Build metadata for an already encoded per-tensor NWCS blob.

        This helper does not encode the tensor. It records the original tensor
        geometry next to the caller-supplied encoded blob.
        """
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        numel = int(tensor.numel())
        n_blocks = numel // int(block_size)
        tail_elements = numel - n_blocks * int(block_size)
        dtype_name = dtype or _dtype_to_name(tensor.dtype)
        original_dtype_name = original_dtype or dtype_name
        meta: dict[str, Any] = {
            "block_size": int(block_size),
            "num_blocks": int(n_blocks),
            "tail_elements": int(tail_elements),
        }
        if codebook_sizes is not None:
            meta["codebook_sizes"] = [int(k) for k in codebook_sizes]
        if block_metadata:
            meta.update(block_metadata)
        return cls(
            name=name,
            shape=tuple(int(s) for s in tensor.shape),
            dtype=dtype_name,
            original_dtype=original_dtype_name,
            block_metadata=meta,
            blob=bytes(blob),
        )


@dataclass(frozen=True)
class NWCSRendererContainer:
    """Decoded NWCS renderer container payload."""

    header: dict[str, Any]
    codec_checkpoint_blob: bytes
    tensors: tuple[NWCSRendererTensorEntry, ...]

    def tensor_blobs(self) -> dict[str, bytes]:
        """Return encoded tensor blobs keyed by tensor name."""
        return {entry.name: entry.blob for entry in self.tensors}


def export_nwcs_renderer_container(
    tensors: list[NWCSRendererTensorEntry | dict[str, Any]]
    | tuple[NWCSRendererTensorEntry | dict[str, Any], ...],
    *,
    codec_checkpoint_blob: bytes | bytearray | memoryview | str | Path = b"",
    output_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> bytes:
    """Serialize encoded per-tensor NWCS blobs into a renderer container.

    The returned bytes are deterministic for the same logical input: tensor
    entries are emitted in lexicographic name order and the JSON header uses
    sorted keys with compact separators.
    """
    entries = [_coerce_nwcs_tensor_entry(item) for item in tensors]
    names = [entry.name for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError("duplicate tensor name in NWCS renderer container")
    entries = sorted(entries, key=lambda entry: entry.name)

    codec_blob = _coerce_blob_or_path(codec_checkpoint_blob, "codec checkpoint")
    container_metadata = dict(metadata or {})
    _assert_json_serializable(container_metadata, "container metadata")
    header_tensors: list[dict[str, Any]] = []
    for entry in entries:
        _validate_nwcs_tensor_entry(entry)
        header_tensors.append(
            {
                "name": entry.name,
                "shape": list(entry.shape),
                "dtype": entry.dtype,
                "original_dtype": entry.original_dtype,
                "block_metadata": entry.block_metadata,
                "blob_len": len(entry.blob),
            }
        )

    header = {
        "magic": "NWCS1",
        "version": _NWCS_RENDERER_VERSION,
        "format": "nwcs_renderer_container",
        "codec_checkpoint_len": len(codec_blob),
        "tensor_count": len(entries),
        "metadata": container_metadata,
        "tensors": header_tensors,
    }
    header_json = json.dumps(
        header, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(header_json) > _NWCS_MAX_HEADER_BYTES:
        raise ValueError(f"NWCS renderer header too large: {len(header_json)} bytes")

    buf = bytearray()
    buf.extend(NWCS_RENDERER_MAGIC)
    buf.extend(_NWCS_RENDERER_LEN.pack(len(header_json)))
    buf.extend(header_json)
    buf.extend(_NWCS_RENDERER_LEN.pack(len(codec_blob)))
    buf.extend(codec_blob)
    for entry in entries:
        buf.extend(_NWCS_RENDERER_LEN.pack(len(entry.blob)))
        buf.extend(entry.blob)

    raw = bytes(buf)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return raw


def load_nwcs_renderer_container(
    path_or_bytes: str | Path | bytes | bytearray | memoryview,
) -> NWCSRendererContainer:
    """Load and validate an NWCS renderer container.

    Raises ``ValueError`` for bad magic, malformed JSON, duplicate tensor
    names, truncated fields, negative or oversized lengths, metadata/blob
    length disagreement, and trailing bytes.
    """
    data = _coerce_blob_or_path(path_or_bytes, "NWCS renderer container")
    if not data.startswith(NWCS_RENDERER_MAGIC):
        raise ValueError(
            f"NWCS renderer container bad magic: got "
            f"{data[:len(NWCS_RENDERER_MAGIC)]!r}, expected {NWCS_RENDERER_MAGIC!r}"
        )

    offset = len(NWCS_RENDERER_MAGIC)
    header_len, offset = _read_nwcs_len(data, offset, "header")
    header_bytes, offset = _take_nwcs_field(
        data,
        offset,
        header_len,
        "header JSON",
        max_len=_NWCS_MAX_HEADER_BYTES,
    )
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("NWCS renderer container invalid JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError("NWCS renderer header must be a JSON object")
    if header.get("magic") != "NWCS1":
        raise ValueError(f"NWCS renderer header magic mismatch: {header.get('magic')!r}")
    if header.get("version") != _NWCS_RENDERER_VERSION:
        raise ValueError(
            f"NWCS renderer unsupported version {header.get('version')!r}"
        )

    tensor_meta = header.get("tensors")
    if not isinstance(tensor_meta, list):
        raise ValueError("NWCS renderer header tensors must be a list")
    tensor_count = _validate_json_int(
        header.get("tensor_count", len(tensor_meta)),
        "tensor_count",
        minimum=0,
    )
    if tensor_count != len(tensor_meta):
        raise ValueError(
            f"NWCS renderer metadata/blob count mismatch: "
            f"tensor_count={tensor_count}, tensors={len(tensor_meta)}"
        )

    expected_codec_len = _validate_json_int(
        header.get("codec_checkpoint_len", 0),
        "codec_checkpoint_len",
        minimum=0,
    )
    codec_len, offset = _read_nwcs_len(data, offset, "codec checkpoint")
    if codec_len != expected_codec_len:
        raise ValueError(
            f"NWCS renderer codec length mismatch: header={expected_codec_len}, "
            f"stream={codec_len}"
        )
    codec_blob, offset = _take_nwcs_field(
        data, offset, codec_len, "codec checkpoint blob"
    )

    entries: list[NWCSRendererTensorEntry] = []
    seen_names: set[str] = set()
    for index, meta in enumerate(tensor_meta):
        if not isinstance(meta, dict):
            raise ValueError(f"NWCS renderer tensor[{index}] metadata must be an object")
        name = _validate_tensor_name(meta.get("name"), f"tensor[{index}].name")
        if name in seen_names:
            raise ValueError(f"NWCS renderer duplicate tensor name: {name!r}")
        seen_names.add(name)
        shape = _validate_shape(meta.get("shape"), f"tensor[{index}].shape")
        dtype = _validate_dtype_name(meta.get("dtype"), f"tensor[{index}].dtype")
        original_dtype = _validate_dtype_name(
            meta.get("original_dtype", dtype), f"tensor[{index}].original_dtype"
        )
        block_metadata = meta.get("block_metadata")
        if not isinstance(block_metadata, dict):
            raise ValueError(
                f"NWCS renderer tensor[{index}].block_metadata must be an object"
            )
        expected_blob_len = _validate_json_int(
            meta.get("blob_len"), f"tensor[{index}].blob_len", minimum=0
        )
        blob_len, offset = _read_nwcs_len(data, offset, f"tensor[{index}] blob")
        if blob_len != expected_blob_len:
            raise ValueError(
                f"NWCS renderer metadata/blob length mismatch for {name!r}: "
                f"header={expected_blob_len}, stream={blob_len}"
            )
        blob, offset = _take_nwcs_field(data, offset, blob_len, f"tensor[{index}] blob")
        entries.append(
            NWCSRendererTensorEntry(
                name=name,
                shape=shape,
                dtype=dtype,
                original_dtype=original_dtype,
                block_metadata=dict(block_metadata),
                blob=blob,
            )
        )

    if offset != len(data):
        raise ValueError(
            f"NWCS renderer metadata/blob count mismatch: {len(data) - offset} "
            "trailing bytes"
        )
    return NWCSRendererContainer(
        header=header,
        codec_checkpoint_blob=codec_blob,
        tensors=tuple(entries),
    )


def is_nwcs_renderer_container(
    path_or_bytes: str | Path | bytes | bytearray | memoryview,
) -> bool:
    """Return True when the path or bytes start with the NWCS container magic."""
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        return bytes(path_or_bytes).startswith(NWCS_RENDERER_MAGIC)
    try:
        with Path(path_or_bytes).open("rb") as f:
            return f.read(len(NWCS_RENDERER_MAGIC)) == NWCS_RENDERER_MAGIC
    except OSError:
        return False


def _dtype_to_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _coerce_blob_or_path(
    value: bytes | bytearray | memoryview | str | Path,
    label: str,
) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, (str, Path)):
        return Path(value).read_bytes()
    raise TypeError(f"{label} must be bytes-like or a path")


def _coerce_nwcs_tensor_entry(
    item: NWCSRendererTensorEntry | dict[str, Any],
) -> NWCSRendererTensorEntry:
    if isinstance(item, NWCSRendererTensorEntry):
        return NWCSRendererTensorEntry(
            name=_validate_tensor_name(item.name, "tensor.name"),
            shape=_validate_shape(item.shape, "tensor.shape"),
            dtype=_validate_dtype_name(item.dtype, "tensor.dtype"),
            original_dtype=_validate_dtype_name(
                item.original_dtype, "tensor.original_dtype"
            ),
            block_metadata=dict(item.block_metadata),
            blob=bytes(item.blob),
        )
    if not isinstance(item, dict):
        raise TypeError("NWCS tensor entry must be a NWCSRendererTensorEntry or dict")
    blob_value = item.get("blob", item.get("encoded_blob"))
    if blob_value is None:
        raise ValueError("NWCS tensor entry missing blob")
    dtype = item.get("dtype", item.get("original_dtype", "float32"))
    original_dtype = item.get("original_dtype", dtype)
    block_metadata = item.get("block_metadata", {})
    if not isinstance(block_metadata, dict):
        raise ValueError("NWCS tensor entry block_metadata must be a dict")
    return NWCSRendererTensorEntry(
        name=_validate_tensor_name(item.get("name"), "tensor.name"),
        shape=_validate_shape(item.get("shape"), "tensor.shape"),
        dtype=_validate_dtype_name(dtype, "tensor.dtype"),
        original_dtype=_validate_dtype_name(original_dtype, "tensor.original_dtype"),
        block_metadata=dict(block_metadata),
        blob=bytes(blob_value),
    )


def _validate_nwcs_tensor_entry(entry: NWCSRendererTensorEntry) -> None:
    _validate_tensor_name(entry.name, "tensor.name")
    _validate_shape(entry.shape, "tensor.shape")
    _validate_dtype_name(entry.dtype, "tensor.dtype")
    _validate_dtype_name(entry.original_dtype, "tensor.original_dtype")
    if not isinstance(entry.block_metadata, dict):
        raise ValueError("NWCS tensor block_metadata must be a dict")
    _assert_json_serializable(entry.block_metadata, "tensor block_metadata")
    if not isinstance(entry.blob, (bytes, bytearray, memoryview)):
        raise TypeError("NWCS tensor blob must be bytes-like")


def _read_nwcs_len(data: bytes, offset: int, label: str) -> tuple[int, int]:
    end = offset + _NWCS_RENDERER_LEN.size
    if end > len(data):
        raise ValueError(f"NWCS renderer truncated {label} length")
    value = _NWCS_RENDERER_LEN.unpack_from(data, offset)[0]
    if value < 0:
        raise ValueError(f"NWCS renderer negative {label} length: {value}")
    return value, end


def _take_nwcs_field(
    data: bytes,
    offset: int,
    length: int,
    label: str,
    *,
    max_len: int | None = None,
) -> tuple[bytes, int]:
    if length < 0:
        raise ValueError(f"NWCS renderer negative {label} length: {length}")
    if max_len is not None and length > max_len:
        raise ValueError(f"NWCS renderer oversized {label}: {length} bytes")
    end = offset + length
    if end > len(data):
        remaining = len(data) - offset
        raise ValueError(
            f"NWCS renderer truncated/oversized {label}: "
            f"declared={length}, remaining={remaining}"
        )
    return data[offset:end], end


def _validate_json_int(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"NWCS renderer {label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"NWCS renderer negative {label}: {value}")
    return int(value)


def _validate_tensor_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"NWCS renderer {label} must be a non-empty string")
    if "\0" in value:
        raise ValueError(f"NWCS renderer {label} contains NUL")
    return value


def _validate_shape(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"NWCS renderer {label} must be a list or tuple")
    shape: list[int] = []
    for index, dim in enumerate(value):
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ValueError(f"NWCS renderer {label}[{index}] must be an integer")
        if dim < 0:
            raise ValueError(f"NWCS renderer {label}[{index}] is negative")
        shape.append(int(dim))
    return tuple(shape)


def _validate_dtype_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"NWCS renderer {label} must be a non-empty string")
    if "\0" in value:
        raise ValueError(f"NWCS renderer {label} contains NUL")
    return value


def _assert_json_serializable(value: Any, label: str) -> None:
    try:
        json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"NWCS renderer {label} must be JSON serializable") from exc


# ── Config ────────────────────────────────────────────────────────────────


@dataclass
class SensitivityAwareCodecConfig:
    """Static config for a sensitivity-aware NWC codec.

    Default codebook ladder is [4, 16, 64, 256], keyed by sensitivity
    quantile (Q1..Q4). Operators may override; sizes must all be ≤ 256
    so codes still fit in uint8.

    block_size matches the underlying base codec.
    """

    block_size: int = 16
    latent_dim: int = 16
    hidden: int = 64
    codebook_sizes: list[int] = field(default_factory=lambda: [4, 16, 64, 256])
    importance_weight: float = 2.0
    """Weight applied to high-sensitivity blocks during VQ-VAE training.

    When training the codec on a corpus, a block whose sensitivity falls
    in the top quartile contributes ``importance_weight`` × MSE to the
    reconstruction loss. Default 2.0 is a conservative choice; ablations
    in [0.5, 5.0] are reasonable.
    """

    def __post_init__(self) -> None:
        if not self.codebook_sizes:
            raise ValueError("codebook_sizes must be non-empty")
        for k in self.codebook_sizes:
            if k <= 0 or k > 256:
                raise ValueError(
                    f"codebook size {k} not in (0, 256] — must fit in uint8"
                )
        if self.block_size <= 0 or self.latent_dim <= 0:
            raise ValueError("block_size and latent_dim must be positive")
        if self.importance_weight < 0:
            raise ValueError("importance_weight must be ≥ 0")


# ── Sensitivity computation ───────────────────────────────────────────────


def compute_per_block_sensitivity(
    model: nn.Module,
    hard_pairs: torch.Tensor,
    gt_pairs: torch.Tensor,
    scorer: nn.Module,
    *,
    block_size: int = 16,
    hessian_proxy: str = "grad_squared",
) -> dict[str, torch.Tensor]:
    """Compute a per-block sensitivity score for every floating-point
    parameter of ``model``.

    The score is the elementwise product of two signals:

        Hessian magnitude   ≈ ⟨∂L/∂w⟩²    (gradient-squared diagonal proxy)
        hard-pair magnitude = |∂L_pair/∂w|  averaged across pairs

    Sensitivity is then aggregated from per-element to per-block via
    ``mean(|score|)``. Returns a dict keyed by parameter NAME mapping to
    a 1-D tensor of length ``ceil(numel / block_size)``.

    Args:
        model:           nn.Module to score (Lane G v3 renderer typically).
        hard_pairs:      (N, ...) tensor of input pairs; the "hard" subset
                         identified by Lane W.
        gt_pairs:        (N, ...) ground-truth target tensor for hard_pairs.
        scorer:          callable producing a scalar loss per pair when
                         called as ``scorer(model_output, target)``. Lane W's
                         standard PoseNet/SegNet loss works.
        block_size:      partition size for sensitivity aggregation.
        hessian_proxy:   currently only ``grad_squared`` (diagonal Fisher)
                         is supported. Future: full Hessian-vector product
                         via finite-diff if memory permits.

    Returns:
        Dict[str, Tensor]: per-parameter per-block sensitivity tensors.
        Each entry is float32, on CPU, shape ``(n_blocks,)``.

    Notes:
        * Uses ``torch.autograd.grad`` so model.parameters() grads are not
          left dirty afterwards.
        * Skips bias-shaped tensors (1-D, < 2048 elements) to match the
          base codec's corpus-builder filter.
        * The hard_pairs / gt_pairs need not be on any specific device —
          they are moved to ``next(model.parameters()).device``.
    """
    if hessian_proxy != "grad_squared":
        raise NotImplementedError(
            f"hessian_proxy={hessian_proxy!r} not supported (only 'grad_squared')"
        )
    device = next(model.parameters()).device
    hard = hard_pairs.to(device)
    gt = gt_pairs.to(device)

    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    grad_sums: dict[str, torch.Tensor] = {n: torch.zeros_like(p) for n, p in params}
    grad_sq_sums: dict[str, torch.Tensor] = {n: torch.zeros_like(p) for n, p in params}

    n_pairs = hard.shape[0]
    if n_pairs == 0:
        raise ValueError("hard_pairs has zero entries; need at least 1 pair")

    for i in range(n_pairs):
        x_i = hard[i : i + 1]
        y_i = gt[i : i + 1]
        out = model(x_i)
        loss = scorer(out, y_i)
        if loss.dim() != 0:
            loss = loss.mean()
        grads = torch.autograd.grad(
            loss,
            [p for _, p in params],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        for (name, _), g in zip(params, grads):
            if g is None:
                continue
            grad_sums[name] = grad_sums[name] + g.detach().abs()
            grad_sq_sums[name] = grad_sq_sums[name] + g.detach() ** 2

    out: dict[str, torch.Tensor] = {}
    for name, p in params:
        if not torch.is_floating_point(p):
            continue
        if p.dim() == 1 and p.numel() < 2048:
            # bias filter — same heuristic as base codec corpus builder
            continue
        if p.numel() < block_size:
            continue
        # mean over pairs
        gmag = (grad_sums[name] / float(n_pairs)).reshape(-1)
        ghess = (grad_sq_sums[name] / float(n_pairs)).reshape(-1)
        # Fisher diagonal proxy: E[g^2].  The gradient magnitude accumulator is
        # retained for diagnostics, but multiplying by it would produce |g|^3.
        per_elem = ghess
        # aggregate to blocks (drop tail; same as tensor_to_blocks)
        n_blocks = per_elem.numel() // block_size
        if n_blocks == 0:
            continue
        per_block = (
            per_elem[: n_blocks * block_size]
            .reshape(n_blocks, block_size)
            .mean(dim=1)
        )
        out[name] = per_block.detach().cpu().float()
    return out


def _bucket_by_quantile(
    sensitivity: torch.Tensor, n_buckets: int
) -> torch.Tensor:
    """Return a long tensor of bucket indices in [0, n_buckets-1] for each
    block, using quantile-edges of the sensitivity distribution.

    Bucket 0 = lowest sensitivity, n_buckets-1 = highest.
    """
    sensitivity = _validate_block_sensitivities(
        sensitivity,
        name="sensitivity",
        allow_empty=True,
    )
    if sensitivity.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    if n_buckets <= 1:
        return torch.zeros_like(sensitivity, dtype=torch.long)
    qs = torch.linspace(0.0, 1.0, n_buckets + 1)[1:-1]
    edges = torch.quantile(sensitivity.float(), qs)
    bucket = torch.bucketize(sensitivity.float(), edges)
    return bucket.long().clamp(max=n_buckets - 1)


def _validate_block_sensitivities(
    sensitivities: torch.Tensor,
    *,
    name: str,
    expected_len: int | None = None,
    allow_empty: bool = False,
) -> torch.Tensor:
    if not torch.is_tensor(sensitivities):
        raise TypeError(f"{name} must be a torch.Tensor")
    out = sensitivities.detach().cpu().float().reshape(-1)
    if expected_len is not None and out.numel() != expected_len:
        raise ValueError(
            f"{name} length {out.numel()} != expected length {expected_len}"
        )
    if out.numel() == 0 and not allow_empty:
        raise ValueError(f"{name} must be non-empty")
    if not torch.isfinite(out).all():
        raise ValueError(f"{name} contains NaN/Inf values")
    if (out < 0).any():
        raise ValueError(f"{name} must be non-negative")
    return out


# ── Codec module ──────────────────────────────────────────────────────────


class SensitivityAwareWeightCodec(WeightCodec):
    """Extension of WeightCodec with per-bucket codebooks of varying size.

    The codec maintains ``len(codebook_sizes)`` separate codebooks, each
    of width ``latent_dim`` and height equal to its bucket's K. At
    encode-time, each block is routed to one bucket (by sensitivity
    quantile), quantized against that bucket's codebook, and serialized
    with a small per-block header byte indicating which bucket was used.

    Design notes:
        * The encoder/decoder MLPs are SHARED across buckets — only the
          codebooks differ. This keeps codec parameter count comparable
          to the base codec (encoder + decoder + sum(K) × latent_dim).
        * For the default ladder [4, 16, 64, 256] @ latent_dim=16, the
          codebooks total 340 × 16 = 5440 floats = 21.7 KB. Shared MLPs
          are unchanged from the base codec.
        * The per-block bucket header is 1 byte (uint8); for the default
          4-bucket config only 2 bits are used. This is the small
          overhead that ``test_byte_size_breakdown_per_codebook_size``
          asserts is < 5% of total.
    """

    def __init__(self, config: SensitivityAwareCodecConfig | None = None):
        # initialize the base WeightCodec with a placeholder codebook_size
        # (we ignore self.codebook below; the per-bucket codebooks are stored
        # in ``self.bucket_codebooks``).
        cfg = config or SensitivityAwareCodecConfig()
        base_cfg = WeightCodecConfig(
            block_size=cfg.block_size,
            codebook_size=max(cfg.codebook_sizes),
            latent_dim=cfg.latent_dim,
            hidden=cfg.hidden,
        )
        super().__init__(base_cfg)
        self.sens_config = cfg

        # Replace the single codebook with a per-bucket ParameterList.
        del self.codebook  # remove inherited single codebook
        self.bucket_codebooks = nn.ParameterList(
            [
                nn.Parameter(torch.randn(K, cfg.latent_dim) * 0.1)
                for K in cfg.codebook_sizes
            ]
        )
        # Re-expose self.codebook as a property (largest bucket) so that
        # WeightCodec methods which read self.codebook still work for
        # untested paths. Tests should always use the bucket-aware API.

    # ── codebook helpers ─────────────────────────────────────────────

    def _quantize_to_bucket(
        self, z_e: torch.Tensor, bucket_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Snap encoder output to the codebook of bucket ``bucket_idx``.

        Returns (z_q, indices) where indices are uint8-compatible long.
        """
        cb = self.bucket_codebooks[bucket_idx]
        z_sq = (z_e ** 2).sum(dim=-1, keepdim=True)
        c_sq = (cb ** 2).sum(dim=-1).unsqueeze(0)
        cross = z_e @ cb.T
        distances = z_sq + c_sq - 2.0 * cross
        indices = distances.argmin(dim=-1)
        z_q = cb.index_select(0, indices)
        return z_q, indices

    # ── train_with_sensitivity ───────────────────────────────────────

    def train_with_sensitivity(
        self,
        corpus: torch.Tensor,
        sensitivities: torch.Tensor,
        *,
        importance_weight: float | None = None,
        num_steps: int = 1000,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
        log_interval: int = 100,
        seed: int = 1234,
    ) -> tuple["SensitivityAwareWeightCodec", list[float]]:
        """Train the codec with importance-weighted reconstruction loss.

        High-sensitivity blocks (top quartile by default) get an extra
        ``importance_weight`` × MSE penalty, biasing the codec toward
        better reconstruction on the blocks that matter for the scorer.
        """
        if corpus.dim() != 2 or corpus.shape[1] != self.config.block_size:
            raise ValueError(
                f"corpus must be (N, block_size={self.config.block_size}), "
                f"got {tuple(corpus.shape)}"
            )
        if sensitivities.dim() != 1 or sensitivities.numel() != corpus.shape[0]:
            raise ValueError(
                f"sensitivities must be 1-D length N={corpus.shape[0]}, "
                f"got shape {tuple(sensitivities.shape)}"
            )
        sensitivities = _validate_block_sensitivities(
            sensitivities,
            name="sensitivities",
            expected_len=int(corpus.shape[0]),
        )
        iw = float(importance_weight) if importance_weight is not None else self.sens_config.importance_weight
        device_t = torch.device(device)
        self.to(device_t)
        corpus = corpus.to(device_t)
        sensitivities = sensitivities.to(device_t)

        n_buckets = len(self.sens_config.codebook_sizes)
        buckets = _bucket_by_quantile(sensitivities, n_buckets).to(device_t)
        # high-sensitivity = top bucket
        high_mask = (buckets == n_buckets - 1).float()

        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))

        losses: list[float] = []
        n = corpus.shape[0]
        for step in range(num_steps):
            idx = torch.randint(0, n, (batch_size,), generator=g)
            x = corpus[idx]
            b = buckets[idx]
            hi = high_mask[idx]

            # forward through encoder
            z_e = self.encoder(x)
            # quantize per-element by bucket
            z_q = torch.zeros_like(z_e)
            for k in range(n_buckets):
                mask = (b == k)
                if not mask.any():
                    continue
                zk_q, _ = self._quantize_to_bucket(z_e[mask], k)
                z_q[mask] = zk_q
            # VQ-VAE STE
            z_q_st = z_e + (z_q - z_e).detach()
            recon = self.decoder(z_q_st)

            # importance-weighted reconstruction loss (per-block scalar)
            per_block = (recon - x).pow(2).mean(dim=1)
            weights = 1.0 + iw * hi  # baseline 1.0, +iw on high-sens blocks
            recon_loss = (per_block * weights).sum() / weights.sum()
            commit = (
                F.mse_loss(z_e, z_q.detach())
                + self.commitment_beta * F.mse_loss(z_q, z_e.detach())
            )
            loss = recon_loss + commit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            if step % log_interval == 0 or step == num_steps - 1:
                print(
                    f"[nwcs-train] step={step:5d} "
                    f"recon={recon_loss.detach().item():.6f} "
                    f"commit={commit.detach().item():.6f} "
                    f"hi_frac={hi.mean().item():.3f} "
                    f"total={loss.item():.6f}"
                )
        return self, losses


# ── Variable-codebook serialization ───────────────────────────────────────

# NWCS1 binary layout (single tensor):
#
#   [4 B] uint32 ndim
#   [4 B × ndim] uint32 shape
#   [4 B] uint32 n_blocks
#   [1 B] uint8 n_buckets
#   [2 B × n_buckets] uint16 bucket sizes (codebook K's; K may be up to 256)
#   [n_blocks × 1 B] uint8 bucket id per block
#   [n_blocks × 2 B] float16 per-block scale
#   [n_blocks × 1 B] uint8 codebook index per block (within its bucket; K=256 stored as 0)
#   [tail × 2 B] float16 leftover-tail elements
#
# Note: a codebook size of K=256 stores indices 0..255 into a uint8 byte
# (the max value uint8 can represent), but the *count* 256 itself does
# not fit in a uint8, hence the 2-byte uint16 used for the bucket-size
# header field.


def encode_with_variable_codebook(
    codec: SensitivityAwareWeightCodec,
    weights: torch.Tensor,
    sensitivities: torch.Tensor,
) -> bytes:
    """Encode a single weight tensor under variable codebook sizes.

    Args:
        codec:        a trained SensitivityAwareWeightCodec.
        weights:      the tensor to encode (any floating dtype).
        sensitivities: 1-D tensor of length ``ceil(numel/block_size)``
                      giving per-block sensitivity. (Tail blocks excluded.)

    Returns:
        bytes blob in NWCS1 layout above.
    """
    if not torch.is_floating_point(weights):
        raise TypeError(f"encode expects floating tensor, got {weights.dtype}")
    device = next(codec.parameters()).device
    Bs = codec.config.block_size
    flat = weights.detach().to(device).float().reshape(-1)
    N = flat.numel()
    n_blocks = N // Bs
    tail_n = N - n_blocks * Bs

    sensitivities = _validate_block_sensitivities(
        sensitivities,
        name="sensitivities",
        expected_len=int(n_blocks),
        allow_empty=(n_blocks == 0),
    )

    n_buckets = len(codec.sens_config.codebook_sizes)
    buckets = _bucket_by_quantile(sensitivities, n_buckets).to(device)

    if n_blocks == 0:
        scales = torch.zeros(0, dtype=torch.float32, device=device)
        codes = torch.zeros(0, dtype=torch.long, device=device)
        bucket_ids = torch.zeros(0, dtype=torch.long, device=device)
    else:
        blocks = flat[: n_blocks * Bs].reshape(n_blocks, Bs)
        scales = blocks.abs().amax(dim=1).clamp(min=1e-8)
        blocks_norm = blocks / scales.unsqueeze(1)
        with torch.no_grad():
            z_e = codec.encoder(blocks_norm)
            codes = torch.zeros(n_blocks, dtype=torch.long, device=device)
            for k in range(n_buckets):
                mask = (buckets == k)
                if not mask.any():
                    continue
                _, code_k = codec._quantize_to_bucket(z_e[mask], k)
                codes[mask] = code_k
        bucket_ids = buckets

    tail = flat[n_blocks * Bs :] if tail_n > 0 else torch.zeros(0, device=device)

    buf = bytearray()
    shape = list(weights.shape)
    buf.extend(struct.pack("<I", len(shape)))
    for s in shape:
        buf.extend(struct.pack("<I", int(s)))
    buf.extend(struct.pack("<I", int(n_blocks)))
    buf.extend(struct.pack("<B", int(n_buckets)))
    for K in codec.sens_config.codebook_sizes:
        # uint16 — 256 fits even though uint8 cannot hold the count.
        buf.extend(struct.pack("<H", int(K)))
    buf.extend(bucket_ids.cpu().to(torch.uint8).numpy().tobytes())
    buf.extend(scales.cpu().to(torch.float16).numpy().tobytes())
    buf.extend(codes.cpu().to(torch.uint8).numpy().tobytes())
    buf.extend(tail.cpu().to(torch.float16).numpy().tobytes())
    return bytes(buf)


def decode_with_per_block_codebook(
    codec: SensitivityAwareWeightCodec, blob: bytes
) -> torch.Tensor:
    """Inverse of ``encode_with_variable_codebook``.

    Returns a CPU float32 tensor with the original shape.
    """
    import numpy as np

    device = next(codec.parameters()).device
    Bs = codec.config.block_size
    offset = 0
    blob_len = len(blob)

    def _read(fmt: str, label: str) -> tuple[int, ...]:
        nonlocal offset
        size = struct.calcsize(fmt)
        end = offset + size
        if end > blob_len:
            raise ValueError(f"NWCS1.decode truncated {label}")
        values = struct.unpack_from(fmt, blob, offset)
        offset = end
        return values

    def _take(length: int, label: str) -> bytes:
        nonlocal offset
        if length < 0:
            raise ValueError(f"NWCS1.decode negative {label} length")
        end = offset + length
        if end > blob_len:
            raise ValueError(
                f"NWCS1.decode truncated {label}: declared={length}, "
                f"remaining={blob_len - offset}"
            )
        out = blob[offset:end]
        offset = end
        return out

    ndim = _read("<I", "ndim")[0]
    if ndim == 0 or ndim > 8:
        raise ValueError(f"NWCS1.decode: implausible ndim={ndim}")
    shape = []
    for _ in range(ndim):
        shape.append(_read("<I", "shape")[0])
    n_blocks = _read("<I", "n_blocks")[0]

    n_buckets = _read("<B", "n_buckets")[0]
    bucket_sizes = []
    for _ in range(n_buckets):
        bucket_sizes.append(_read("<H", "bucket size")[0])
    if list(bucket_sizes) != list(codec.sens_config.codebook_sizes):
        raise ValueError(
            f"NWCS1.decode: codec bucket sizes mismatch "
            f"(blob {bucket_sizes} vs codec {codec.sens_config.codebook_sizes})"
        )

    bucket_ids_buf = _take(n_blocks, "bucket ids")
    scales_buf = _take(n_blocks * 2, "scales")
    codes_buf = _take(n_blocks, "codes")

    numel = 1
    for s in shape:
        numel *= int(s)
    tail_n = numel - n_blocks * Bs
    if tail_n < 0:
        raise ValueError(
            f"NWCS1.decode negative tail (n_blocks={n_blocks}, Bs={Bs}, numel={numel})"
        )
    tail_buf = _take(tail_n * 2, "tail")
    if offset != blob_len:
        raise ValueError(f"NWCS1.decode trailing bytes: {blob_len - offset}")

    if n_blocks > 0:
        bucket_ids = torch.from_numpy(
            np.frombuffer(bucket_ids_buf, dtype=np.uint8).copy()
        ).to(device=device, dtype=torch.long)
        if bucket_ids.numel() != n_blocks:
            raise ValueError(
                f"NWCS1.decode bucket id count mismatch: "
                f"{bucket_ids.numel()} != {n_blocks}"
            )
        if int(bucket_ids.max().item()) >= n_buckets:
            raise ValueError("NWCS1.decode bucket id outside bucket table")
        scales = torch.from_numpy(
            np.frombuffer(scales_buf, dtype=np.float16).copy()
        ).to(device=device, dtype=torch.float32)
        if scales.numel() != n_blocks:
            raise ValueError(
                f"NWCS1.decode scale count mismatch: {scales.numel()} != {n_blocks}"
            )
        codes = torch.from_numpy(
            np.frombuffer(codes_buf, dtype=np.uint8).copy()
        ).to(device=device, dtype=torch.long)
        if codes.numel() != n_blocks:
            raise ValueError(
                f"NWCS1.decode code count mismatch: {codes.numel()} != {n_blocks}"
            )
        with torch.no_grad():
            z_q = torch.zeros(n_blocks, codec.config.latent_dim, device=device)
            for k in range(n_buckets):
                mask = (bucket_ids == k)
                if not mask.any():
                    continue
                if int(codes[mask].max().item()) >= codec.sens_config.codebook_sizes[k]:
                    raise ValueError(f"NWCS1.decode code index outside bucket {k}")
                z_q[mask] = codec.bucket_codebooks[k].index_select(0, codes[mask])
            recon_norm = codec.decoder(z_q)
            recon_blocks = recon_norm * scales.unsqueeze(1)
        flat_recon = recon_blocks.reshape(-1)
    else:
        flat_recon = torch.zeros(0, device=device, dtype=torch.float32)

    if tail_n > 0:
        tail = torch.from_numpy(
            np.frombuffer(tail_buf, dtype=np.float16).copy()
        ).to(device=device, dtype=torch.float32)
    else:
        tail = torch.zeros(0, device=device, dtype=torch.float32)

    full = torch.cat([flat_recon, tail], dim=0)
    if full.numel() != numel:
        raise ValueError(
            f"NWCS1.decode size mismatch: got {full.numel()}, expected {numel}"
        )
    return full.reshape(*shape).cpu()
