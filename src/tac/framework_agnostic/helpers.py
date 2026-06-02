# SPDX-License-Identifier: MIT
"""Canonical helpers for framework-agnostic bridge contracts.

Per CLAUDE.md "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL" 8th standing
directive: ``MLX state_dict → npz → ZIP-member → numpy inflate primitives``
is the canonical bridge contract. THIS module exposes the canonical bridge
helpers so substrate trainers can route through them without re-implementing
the npz bridge per substrate (which would re-introduce the duplicate-
implementation anti-pattern this whole package extincts).

Per CLAUDE.md "Deterministic packet compiler" + Catalog #146: the npz
serialization is byte-deterministic across backends — substrate trainers
can fork the framework choice per Catalog #205 sister discipline (MLX-LOCAL
for $0 development; PyTorch CUDA for contest-resolution) while preserving
byte-identical inflate output.

Per CLAUDE.md "Bit-level deconstruction and entropy discipline": the bridge
preserves per-tensor metadata (shape + dtype + quantization scale) so
downstream consumers (autopilot ranker, sensitivity map, bit-allocator)
inherit canonical Provenance per Catalog #323.

Cross-references:
  * Catalog #205 — sister at inflate-time device-selection surface
  * Catalog #146 — contest-compliant inflate runtime contract
  * Catalog #323 — canonical Provenance umbrella
  * Catalog #287 — placeholder-rationale rejection sister discipline
  * Catalog #371 — orphan-auto-trigger-stub sister discipline (this module
    has zero stubs; every helper has a working numpy reference path)
"""
from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.framework_agnostic.backend import (
    _AVAILABILITY_CHECK,
    Backend,
    BackendUnavailableError,
)

NPZ_BRIDGE_MANIFEST_SCHEMA = "framework_agnostic_npz_bridge_manifest.v1"


def assert_no_framework_mismatch(tensor: Any, expected_backend: Backend) -> None:
    """Fail-closed check that ``tensor`` is from the expected backend.

    Per CLAUDE.md "Forbidden score claims" + Catalog #1 sister: silently
    routing a torch.Tensor through an MLX-pinned code path would produce
    surprising behavior. This canonical helper raises with a clear error
    message so the caller can route through the canonical bridge instead.

    Args:
        tensor: FrameworkAgnosticTensor.
        expected_backend: The Backend the caller expects.

    Raises:
        TypeError: If the tensor is NOT from the expected backend.
    """
    actual = _detect_tensor_backend(tensor)
    if actual is None:
        # Unknown tensor type — let downstream routing decide.
        return
    if actual is not expected_backend:
        raise TypeError(
            f"assert_no_framework_mismatch: expected Backend.{expected_backend.name} "
            f"tensor, got Backend.{actual.name} tensor (type={type(tensor).__name__}). "
            f"Route through tac.framework_agnostic.helpers.coerce_to_backend "
            f"or the canonical bridge helpers to convert."
        )


def _detect_tensor_backend(tensor: Any) -> Backend | None:
    """Best-effort detection of which backend a tensor is from.

    Returns None if the tensor's backend cannot be classified (e.g., a
    Python list / scalar).
    """
    module_name = type(tensor).__module__
    if module_name.startswith("torch"):
        return Backend.PYTORCH
    if module_name.startswith("mlx") or module_name.startswith("mx"):
        return Backend.MLX
    if module_name.startswith("numpy"):
        return Backend.NUMPY
    if module_name.startswith("tinygrad"):
        return Backend.TINYGRAD
    return None


# -----------------------------------------------------------------------------
# Canonical bridge helpers: state_dict → npz
# -----------------------------------------------------------------------------


def _as_canonical_numpy_state_dict(
    state_dict: Mapping[str, Any],
    *,
    require_finite: bool = True,
) -> dict[str, Any]:
    """Return a sorted, numeric, C-contiguous NumPy state dict."""

    import numpy as np

    if not state_dict:
        raise ValueError("state_dict must not be empty")
    out: dict[str, Any] = {}
    for raw_name in sorted(state_dict.keys(), key=str):
        name = str(raw_name)
        if not name:
            raise ValueError("state_dict contains an empty tensor name")
        arr = np.asarray(state_dict[raw_name])
        if arr.dtype == object:
            raise TypeError(f"{name}: object dtype is not portable")
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"{name}: non-numeric dtype {arr.dtype} is not portable")
        arr = np.ascontiguousarray(arr)
        if require_finite and not bool(np.isfinite(arr).all()):
            raise ValueError(f"{name}: non-finite values are not portable")
        out[name] = arr
    return out


def numpy_state_dict_to_npz_bridge(
    numpy_state_dict: Mapping[str, Any],
    *,
    require_finite: bool = True,
) -> bytes:
    """Canonical NumPy state_dict → compressed NPZ bridge.

    This is the backend-neutral middle of the MLX/PyTorch/tinygrad export
    contract. Inputs are sorted by tensor name, validated as numeric and
    C-contiguous, then serialized through NumPy's compressed NPZ container.
    """

    import numpy as np

    numpy_dict = _as_canonical_numpy_state_dict(
        numpy_state_dict,
        require_finite=require_finite,
    )
    buf = io.BytesIO()
    np.savez_compressed(buf, **numpy_dict)
    return buf.getvalue()


def build_npz_bridge_manifest(
    npz_bytes: bytes,
    *,
    source_backend: str,
    bridge_kind: str = "state_dict_to_npz",
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a manifest for a canonical NPZ bridge artifact."""

    import numpy as np

    arrays = npz_to_numpy_primitives(npz_bytes)
    per_tensor: dict[str, dict[str, Any]] = {}
    total_uncompressed_bytes = 0
    finite = True
    for name in sorted(arrays):
        arr = np.ascontiguousarray(arrays[name])
        arr_finite = (
            bool(np.isfinite(arr).all())
            if np.issubdtype(arr.dtype, np.number)
            else False
        )
        finite = finite and arr_finite
        nbytes = int(arr.nbytes)
        total_uncompressed_bytes += nbytes
        per_tensor[name] = {
            "shape": [int(v) for v in arr.shape],
            "dtype": str(arr.dtype),
            "nbytes": nbytes,
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            "finite": arr_finite,
        }
    blockers: list[str] = []
    if not finite:
        blockers.append("nonfinite_tensor_values")
    return {
        "schema": NPZ_BRIDGE_MANIFEST_SCHEMA,
        "bridge_kind": str(bridge_kind),
        "source_backend": str(source_backend),
        "artifact_path": (
            Path(artifact_path).as_posix() if artifact_path is not None else None
        ),
        "artifact_bytes": len(npz_bytes),
        "artifact_sha256": hashlib.sha256(npz_bytes).hexdigest(),
        "tensor_count": len(arrays),
        "tensor_names_sorted": sorted(arrays),
        "total_uncompressed_tensor_bytes": int(total_uncompressed_bytes),
        "all_tensors_finite": bool(finite),
        "per_tensor": per_tensor,
        "consumption_recommended": not blockers,
        "blockers": blockers,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
    }


def write_npz_bridge_artifact(
    state_dict: Mapping[str, Any],
    npz_path: str | Path,
    *,
    source_backend: str,
    bridge_kind: str = "state_dict_to_npz",
    manifest_path: str | Path | None = None,
    require_finite: bool = True,
) -> dict[str, Any]:
    """Write a canonical NPZ bridge artifact and adjacent manifest."""

    import json

    npz_out = Path(npz_path)
    npz_out.parent.mkdir(parents=True, exist_ok=True)
    npz_bytes = numpy_state_dict_to_npz_bridge(
        state_dict,
        require_finite=require_finite,
    )
    npz_out.write_bytes(npz_bytes)
    manifest = build_npz_bridge_manifest(
        npz_bytes,
        source_backend=source_backend,
        bridge_kind=bridge_kind,
        artifact_path=npz_out,
    )
    manifest_out = (
        Path(manifest_path)
        if manifest_path is not None
        else npz_out.with_suffix(npz_out.suffix + ".manifest.json")
    )
    manifest["manifest_path"] = manifest_out.as_posix()
    manifest_out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def mlx_state_dict_to_npz_bridge(mlx_state_dict: Mapping[str, Any]) -> bytes:
    """Canonical MLX state_dict → npz bridge per 8th standing directive.

    Per CLAUDE.md "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL" 8th
    standing directive: ``MLX state_dict → npz → ZIP-member → numpy inflate``
    is the canonical bridge contract. This helper produces the canonical
    npz bytes from an MLX state_dict so the downstream ZIP archive builder
    + numpy inflate runtime can consume without MLX dependency.

    Args:
        mlx_state_dict: Mapping of param name → mx.array.

    Returns:
        npz bytes (canonical numpy.savez_compressed format).

    Raises:
        BackendUnavailableError: If MLX not installed (the input MUST be
            MLX arrays per the bridge contract).
    """
    if not _AVAILABILITY_CHECK[Backend.MLX]():
        raise BackendUnavailableError(
            "mlx_state_dict_to_npz_bridge requires MLX installed; "
            "install via `uv pip install mlx` (Darwin ARM64 only)"
        )
    # Convert every MLX array to numpy via np.asarray (MLX supports __array__).
    import numpy as np

    numpy_dict = {k: np.asarray(v) for k, v in mlx_state_dict.items()}
    return numpy_state_dict_to_npz_bridge(numpy_dict)


def pytorch_state_dict_to_npz_bridge(pytorch_state_dict: Mapping[str, Any]) -> bytes:
    """Canonical PyTorch state_dict → npz bridge.

    Sister of :func:`mlx_state_dict_to_npz_bridge` for contest-resolution
    paths per Catalog #205. Routes the PyTorch tensors through the canonical
    numpy oracle so the downstream ZIP archive builder + inflate runtime
    remains numpy-portable per HNeRV parity L4.

    Args:
        pytorch_state_dict: Mapping of param name → torch.Tensor.

    Returns:
        npz bytes (canonical numpy.savez_compressed format).

    Raises:
        BackendUnavailableError: If PyTorch not installed.
    """
    if not _AVAILABILITY_CHECK[Backend.PYTORCH]():
        raise BackendUnavailableError(
            "pytorch_state_dict_to_npz_bridge requires torch installed; "
            "install via `uv pip install torch`"
        )
    import numpy as np
    import torch
    numpy_dict = {}
    for k, v in pytorch_state_dict.items():
        if isinstance(v, torch.Tensor):
            numpy_dict[k] = v.detach().cpu().numpy()
        else:
            numpy_dict[k] = np.asarray(v)
    return numpy_state_dict_to_npz_bridge(numpy_dict)


def tinygrad_state_dict_to_npz_bridge(tinygrad_state_dict: Mapping[str, Any]) -> bytes:
    """Canonical tinygrad state_dict → npz bridge.

    Deferred import; tinygrad is OPTIONAL per Catalog #287. Sister of
    :func:`mlx_state_dict_to_npz_bridge` for the tinygrad backend.

    Args:
        tinygrad_state_dict: Mapping of param name → tinygrad.Tensor.

    Returns:
        npz bytes (canonical numpy.savez_compressed format).

    Raises:
        BackendUnavailableError: If tinygrad not installed.
    """
    if not _AVAILABILITY_CHECK[Backend.TINYGRAD]():
        raise BackendUnavailableError(
            "tinygrad_state_dict_to_npz_bridge requires tinygrad installed; "
            "install via `uv pip install tinygrad` (optional)"
        )
    import numpy as np
    from tinygrad import Tensor
    numpy_dict = {}
    for k, v in tinygrad_state_dict.items():
        if isinstance(v, Tensor):
            numpy_dict[k] = v.numpy()
        else:
            numpy_dict[k] = np.asarray(v)
    return numpy_state_dict_to_npz_bridge(numpy_dict)


def npz_to_numpy_primitives(npz_bytes: bytes) -> dict[str, Any]:
    """Inverse of the *_state_dict_to_npz_bridge helpers.

    Canonical inflate-side consumer per the bridge contract: any of the
    *_to_npz_bridge helpers above produce bytes consumed by THIS helper to
    yield a dict[str, numpy.ndarray] ready for canonical numpy-portable
    inflate primitives per HNeRV parity L4.

    Per CLAUDE.md "Deterministic packet compiler" + Catalog #146: the npz
    round-trip is byte-deterministic so substrate trainers can fork the
    framework choice (MLX vs PyTorch vs tinygrad) while preserving
    byte-identical inflate output.

    Args:
        npz_bytes: Output of *_state_dict_to_npz_bridge helpers.

    Returns:
        Dict[str, numpy.ndarray].
    """
    import numpy as np
    buf = io.BytesIO(npz_bytes)
    with np.load(buf, allow_pickle=False) as data:
        # NpzFile is lazy; materialize to dict.
        return {k: data[k] for k in data.files}


def detect_available_backends_dict() -> dict[Backend, bool]:
    """Return mapping of every backend → availability status.

    Sister of
    :func:`tac.framework_agnostic.backend.detect_available_backends` but
    returns a dict mapping for cathedral consumers + autopilot ranker
    consumption (per Catalog #335 sister discipline; the consumer's
    consume_candidate hook can use this to annotate candidates with backend
    availability).
    """
    return {b: _AVAILABILITY_CHECK[b]() for b in (Backend.MLX, Backend.PYTORCH, Backend.NUMPY, Backend.TINYGRAD)}


# -----------------------------------------------------------------------------
# Canonical bridge helpers: MLX HWIO Conv2d weight -> PyTorch OIHW Conv2d weight
# -----------------------------------------------------------------------------


def convert_mlx_state_dict_to_pytorch_oihw(
    mlx_state_dict_numpy: Mapping[str, Any],
    *,
    skip_buffer_name_predicate: Any = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Canonical MLX HWIO -> PyTorch OIHW Conv2d weight transpose bridge.

    Empirically extracted 2026-05-30 from 5-of-6 MLX -> PyTorch export tools
    (``tools/export_pact_nerv_{ia3,selector_v2,selector_v3,selector_v4}_mlx_to_pytorch_state_dict.py``
    + ``tools/export_z6_v2_cargo_cult_unwind_mlx_to_pytorch_state_dict.py``)
    where the identical Conv2d weight transpose pattern was duplicated
    verbatim. The 6th tool (``tools/export_pact_nerv_vq_mlx_to_pytorch_state_dict.py``)
    is a PRINCIPLED FORK per Catalog #290 because it must skip
    ``quantizer.*`` VQ buffer names; the optional
    ``skip_buffer_name_predicate`` callback preserves that distinction
    without forcing the VQ tool to fork the canonical helper.

    Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" Catalog #290
    falling-rule: this extraction is OBVIOUS-FIT for 5 tools (identical
    transpose semantics) while VQ remains substrate-distinguished via the
    predicate. Per CLAUDE.md NO FAKE IMPLEMENTATIONS Slot EEE Class 5:
    this helper performs SUBSTANTIVE work (transpose + ascontiguousarray +
    fp32 cast + torch.from_numpy + per-tensor sha256 sidecar) — NOT a
    cargo-culted thin wrapper.

    Args:
        mlx_state_dict_numpy: ``{key: np.ndarray}`` from
            :func:`tac.substrates._shared.numpy_portable_inflate.unpack_state_dict_numpy`
            (canonical MLX numpy-portable state_dict consumer).
        skip_buffer_name_predicate: Optional callable ``str -> bool``; when
            it returns ``True`` for a given tensor name, the tensor's
            Conv2d HWIO -> OIHW transpose is SKIPPED (preserves substrate-
            distinguishing buffer layouts; canonical example: VQ
            ``quantizer.*`` buffers per VQ-VAE §3.2). Default ``None``
            applies the transpose to every ``.weight`` tensor with
            ``ndim == 4``.

    Returns:
        ``(pytorch_sd, per_tensor)`` where:
          * ``pytorch_sd``: ``{name: torch.Tensor}`` mapping ready for
            ``torch.nn.Module.load_state_dict(strict=True)``; Conv2d
            weights in canonical OIHW layout (out_channels, in_channels,
            kH, kW); all tensors cast to ``np.float32`` for canonical
            contest-faithful storage.
          * ``per_tensor``: ``{name: {shape_mlx, shape_pytorch, dtype,
            sha256, layout}}`` sidecar dict for the export manifest
            (canonical sha256 per Catalog #323 Provenance; layout token
            ``"mlx_hwio_to_pytorch_oihw"`` vs ``"preserved"`` vs
            ``"skipped_by_predicate"``).

    Raises:
        BackendUnavailableError: If PyTorch + numpy not installed.
    """
    if not _AVAILABILITY_CHECK[Backend.PYTORCH]():
        raise BackendUnavailableError(
            "convert_mlx_state_dict_to_pytorch_oihw requires torch installed; "
            "install via `uv pip install torch`"
        )
    import hashlib

    import numpy as np
    import torch

    pytorch_sd: dict[str, Any] = {}
    per_tensor: dict[str, dict[str, Any]] = {}
    for name, arr in mlx_state_dict_numpy.items():
        out_arr = arr
        layout_note = "preserved"
        predicate_skip = (
            skip_buffer_name_predicate is not None
            and bool(skip_buffer_name_predicate(name))
        )
        if predicate_skip:
            # Predicate explicitly skips transpose for substrate-
            # distinguishing buffer (e.g. VQ ``quantizer.*`` per Catalog
            # #290 PRINCIPLED FORK).
            layout_note = "skipped_by_predicate"
        elif name.endswith(".weight") and arr.ndim == 4:
            # MLX Conv2d weight is (out_channels, kH, kW, in_channels);
            # PyTorch wants (out_channels, in_channels, kH, kW).
            out_arr = np.transpose(arr, (0, 3, 1, 2))
            layout_note = "mlx_hwio_to_pytorch_oihw"
        out_arr = np.ascontiguousarray(out_arr).astype(np.float32)
        pytorch_sd[name] = torch.from_numpy(out_arr.copy())
        per_tensor[name] = {
            "shape_mlx": list(arr.shape),
            "shape_pytorch": list(out_arr.shape),
            "dtype": str(out_arr.dtype),
            "sha256": hashlib.sha256(out_arr.tobytes()).hexdigest()[:16],
            "layout": layout_note,
        }
    return pytorch_sd, per_tensor


__all__ = [
    "NPZ_BRIDGE_MANIFEST_SCHEMA",
    "assert_no_framework_mismatch",
    "build_npz_bridge_manifest",
    "convert_mlx_state_dict_to_pytorch_oihw",
    "detect_available_backends_dict",
    "mlx_state_dict_to_npz_bridge",
    "npz_to_numpy_primitives",
    "numpy_state_dict_to_npz_bridge",
    "pytorch_state_dict_to_npz_bridge",
    "tinygrad_state_dict_to_npz_bridge",
    "write_npz_bridge_artifact",
]
