# SPDX-License-Identifier: MIT
"""Tinygrad-portable inflate primitive bridge — 3rd sister surface per MLX-FIRST 8th standing directive.

Per operator NON-NEGOTIABLE cascade item 6 of 2026-05-28 7-item cascade
(``feedback_cathedral_autopilot_is_the_canonical_meta_orchestrator_proceed_with_all_7_cascade_20260528.md``):
"TINYGRAD-PORTABLE INFLATE PRIMITIVE BRIDGE — 3rd sister surface per
MLX-FIRST 8th standing directive". The 8th standing directive
(``feedback_mlx_first_numpy_portable_individually_fractally_optimized_standing_directive_20260526.md``):
**Inflate.py L4 ≤200 LOC budget gets 3 framework choices not 2** —
training-time framework is MLX-first OR PyTorch-first OR tinygrad; inflate-time
runtime is ALWAYS numpy-portable per HNeRV parity L4 (no MLX / no torch / no
tinygrad dep at inflate time).

Sister of ``src/tac/local_acceleration/pr95_hnerv_mlx.py`` (PR95 HNeRV MLX
bridge) at the **tinygrad framework alternative surface**. Different framework,
same canonical bridge contract: ``state_dict → npz → ZIP-member → numpy
inflate primitives``.

This module is deliberately narrow (≤250 LOC per HNeRV parity L4 substrate-
engineering exception per L7): it provides the canonical ZIP-member packaging
+ per-tensor-metadata-preservation + decorator surface that future substrate
trainers choosing tinygrad-on-Apple-Silicon-Metal / tinygrad-on-Linux-CUDA /
tinygrad-on-WebGPU as their training-time framework consume without per-
substrate rediscovery of the canonical bridge contract.

Per CLAUDE.md "tac stays clean" non-negotiable: this module lives under
``tac.local_acceleration`` (sister of MLX bridge) because the framework-
selection surface is local-acceleration-specific; the framework-agnostic
PRIMITIVES live in ``tac.framework_agnostic`` and this module CONSUMES them.

Cross-references:
  * CLAUDE.md "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL" 8th standing
    directive (parent canonical contract)
  * CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L4
    (≤200 LOC + ≤2 ext deps inflate budget)
  * CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable
    (per-substrate fork-vs-canonical decision per Catalog #290)
  * CLAUDE.md "Bit-level deconstruction and entropy discipline" (npz round-
    trip byte-deterministic per Catalog #146)
  * ``src/tac/local_acceleration/pr95_hnerv_mlx.py`` — MLX bridge reference
  * ``src/tac/framework_agnostic/helpers.py::tinygrad_state_dict_to_npz_bridge``
    — canonical bridge primitive THIS module wraps
  * ``src/tac/framework_agnostic/backend.py::Backend.TINYGRAD`` — canonical
    Backend taxonomy sister surface
  * ``src/tac/substrates/_shared/inflate_runtime.py::select_inflate_device``
    — Catalog #205 canonical inflate-time device-selection sister
  * Catalog #146 / #205 / #220 / #270 / #272 / #287 / #290 / #294 / #295 /
    #303 / #305 / #309 / #310 / #323 / #335 / #341 / #344 / #356 / #357
  * ``.omx/research/tinygrad_portable_inflate_primitive_bridge_design_20260529.md``
    (canonical design memo)
"""

from __future__ import annotations

import functools
import io
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np

from tac.framework_agnostic.backend import (
    Backend,
    BackendUnavailableError,
    _AVAILABILITY_CHECK,
)
from tac.framework_agnostic.decorators import inflate_runtime_helper
from tac.framework_agnostic.helpers import (
    npz_to_numpy_primitives,
    tinygrad_state_dict_to_npz_bridge,
)

# Canonical constants per sister `pr95_hnerv_mlx.py` pattern.
LANE_ID = "lane_slot_j_cascade_item_6_tinygrad_portable_inflate_primitive_bridge_20260529"
TINYGRAD_BRIDGE_SCHEMA = "tinygrad_portable_inflate_primitive_bridge.v1"
BRIDGE_MANIFEST_SCHEMA = "tinygrad_bridge_manifest.v1"

# Canonical ZIP-member name within the bridge archive. Sister of HNeRV parity
# L3 monolithic 4-section archive grammar single-file convention.
DEFAULT_ZIP_MEMBER_NAME = "tinygrad_weights.npz"

# Canonical evidence grade for tinygrad-trained signals. Sister of
# `tac.local_acceleration.EVIDENCE_GRADE_MLX` per Catalog #287/#323.
EVIDENCE_GRADE_TINYGRAD = "tinygrad-research-signal"
EVIDENCE_TAG_TINYGRAD = "[tinygrad research-signal]"

F = TypeVar("F", bound=Callable[..., Any])


def is_tinygrad_available() -> bool:
    """Return True iff tinygrad is importable in the current environment.

    Sister of ``_is_tinygrad_available`` in
    ``tac.framework_agnostic.backend`` exposed as a public API on this
    bridge module for convenient caller-side gating.
    """
    return _AVAILABILITY_CHECK[Backend.TINYGRAD]()


@dataclass(frozen=True)
class TinygradBridgeManifest:
    """Canonical per-export bridge manifest with per-tensor metadata.

    Sister of ``AxisDecomposition`` (Catalog #356) +
    ``DeliverabilityProof`` (Catalog #319) frozen-dataclass canonical
    pattern. Carries non-promotable canonical Provenance per Catalog
    #287/#323 (tinygrad outputs are training-time research signal; paired
    contest-axis empirical anchor required for promotion per Catalog
    #246).
    """

    schema_version: str
    tensor_count: int
    total_uncompressed_bytes: int
    compressed_bytes: int
    per_tensor_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    per_tensor_dtypes: dict[str, str] = field(default_factory=dict)
    zip_member_name: str = DEFAULT_ZIP_MEMBER_NAME
    canonical_provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fail-closed invariants per canonical frozen-dataclass discipline."""
        if not isinstance(self.tensor_count, int) or self.tensor_count < 0:
            raise ValueError(
                f"tensor_count must be a non-negative int; got {self.tensor_count!r}"
            )
        if not isinstance(self.total_uncompressed_bytes, int) or self.total_uncompressed_bytes < 0:
            raise ValueError(
                f"total_uncompressed_bytes must be a non-negative int; got {self.total_uncompressed_bytes!r}"
            )
        if not isinstance(self.compressed_bytes, int) or self.compressed_bytes < 0:
            raise ValueError(
                f"compressed_bytes must be a non-negative int; got {self.compressed_bytes!r}"
            )
        if self.tensor_count != len(self.per_tensor_shapes):
            raise ValueError(
                f"tensor_count={self.tensor_count} does not match per_tensor_shapes "
                f"count={len(self.per_tensor_shapes)}"
            )
        if set(self.per_tensor_shapes.keys()) != set(self.per_tensor_dtypes.keys()):
            raise ValueError(
                "per_tensor_shapes + per_tensor_dtypes must have identical keys"
            )

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable representation per canonical posterior consumer surface."""
        return {
            "schema_version": self.schema_version,
            "tensor_count": self.tensor_count,
            "total_uncompressed_bytes": self.total_uncompressed_bytes,
            "compressed_bytes": self.compressed_bytes,
            "per_tensor_shapes": {k: list(v) for k, v in self.per_tensor_shapes.items()},
            "per_tensor_dtypes": dict(self.per_tensor_dtypes),
            "zip_member_name": self.zip_member_name,
            "canonical_provenance": dict(self.canonical_provenance),
        }


def tinygrad_state_dict_to_zip_member_bytes(
    state_dict: Mapping[str, Any],
    *,
    archive_name: str = DEFAULT_ZIP_MEMBER_NAME,
) -> tuple[bytes, TinygradBridgeManifest]:
    """Canonical tinygrad state_dict → ZIP-archive-bytes bridge.

    Wraps :func:`tac.framework_agnostic.helpers.tinygrad_state_dict_to_npz_bridge`
    + packages the npz bytes as a single ZIP-member with the canonical
    DEFLATE compression for byte-stable archives. The output is a
    self-contained ZIP archive whose member name is ``archive_name`` (a
    monolithic single-file archive sister of HNeRV parity L3 convention).

    Args:
        state_dict: Mapping of param name → tinygrad.Tensor (or convertible).
        archive_name: ZIP-member name; default ``tinygrad_weights.npz``.

    Returns:
        Tuple ``(zip_bytes, manifest)`` where:
          * ``zip_bytes`` is a self-contained ZIP archive bytes; consume
            via :func:`load_tinygrad_trained_weights_for_numpy_inflate`.
          * ``manifest`` is a :class:`TinygradBridgeManifest` carrying
            per-tensor metadata + non-promotable canonical Provenance.

    Raises:
        BackendUnavailableError: If tinygrad is not installed.
    """
    npz_bytes = tinygrad_state_dict_to_npz_bridge(state_dict)
    # Per-tensor metadata extraction via the canonical numpy oracle.
    primitives = npz_to_numpy_primitives(npz_bytes)
    per_tensor_shapes = {k: tuple(v.shape) for k, v in primitives.items()}
    per_tensor_dtypes = {k: str(v.dtype) for k, v in primitives.items()}
    total_uncompressed = sum(int(v.nbytes) for v in primitives.values())
    # Deterministic ZIP packaging: ZIP_DEFLATED with canonical compresslevel.
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zip_info = zipfile.ZipInfo(filename=archive_name, date_time=(1980, 1, 1, 0, 0, 0))
        zip_info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zip_info, npz_bytes)
    zip_bytes = zip_buf.getvalue()
    manifest = TinygradBridgeManifest(
        schema_version=BRIDGE_MANIFEST_SCHEMA,
        tensor_count=len(primitives),
        total_uncompressed_bytes=total_uncompressed,
        compressed_bytes=len(zip_bytes),
        per_tensor_shapes=per_tensor_shapes,
        per_tensor_dtypes=per_tensor_dtypes,
        zip_member_name=archive_name,
        canonical_provenance={
            "evidence_grade": EVIDENCE_GRADE_TINYGRAD,
            "axis_tag": "[predicted]",
            "score_claim": False,
            "promotable": False,
            "lane_id": LANE_ID,
            "schema_version": TINYGRAD_BRIDGE_SCHEMA,
        },
    )
    return zip_bytes, manifest


@inflate_runtime_helper
def load_tinygrad_trained_weights_for_numpy_inflate(
    zip_bytes: bytes,
    *,
    member_name: str = DEFAULT_ZIP_MEMBER_NAME,
    backend: Backend | None = None,  # pinned to NUMPY by decorator
) -> dict[str, np.ndarray]:
    """Inflate-side canonical numpy consumer.

    Reads the ZIP archive bytes produced by
    :func:`tinygrad_state_dict_to_zip_member_bytes`, extracts the
    canonical npz member, and returns a ``dict[str, numpy.ndarray]`` ready
    for numpy-portable inflate primitives per HNeRV parity L4 (≤200 LOC +
    ≤2 deps + CUDA-or-CPU agnostic).

    Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"
    L4 + Catalog #295 PYTHONPATH self-containment: this function uses ONLY
    ``numpy`` + ``zipfile`` (Python stdlib) — **zero tinygrad dependency at
    inflate time**. The bridge contract guarantees byte-determinism via
    the canonical numpy oracle.

    Args:
        zip_bytes: ZIP archive bytes from
            :func:`tinygrad_state_dict_to_zip_member_bytes`.
        member_name: ZIP-member name; default ``tinygrad_weights.npz``.
        backend: Pinned to ``Backend.NUMPY`` by ``@inflate_runtime_helper``;
            caller override silently ignored per canonical decorator
            contract.

    Returns:
        Dict mapping param name → numpy.ndarray.
    """
    del backend  # pinned to NUMPY by decorator per bridge contract
    with zipfile.ZipFile(io.BytesIO(zip_bytes), mode="r") as zf:
        npz_bytes = zf.read(member_name)
    return npz_to_numpy_primitives(npz_bytes)


def build_tinygrad_bridge_manifest(
    state_dict: Mapping[str, Any],
    *,
    archive_name: str = DEFAULT_ZIP_MEMBER_NAME,
    extra_provenance: Mapping[str, Any] | None = None,
) -> TinygradBridgeManifest:
    """Operator-facing manifest builder; convenience wrapper for audit consumers.

    Builds the bridge manifest WITHOUT emitting the ZIP bytes (cheaper than
    :func:`tinygrad_state_dict_to_zip_member_bytes` for operators only needing
    per-tensor metadata + canonical Provenance for audit / cathedral consumer
    routing per Catalog #335).

    Args:
        state_dict: Mapping of param name → tinygrad.Tensor.
        archive_name: ZIP-member name; default ``tinygrad_weights.npz``.
        extra_provenance: Optional extra Provenance fields to merge per
            Catalog #323 canonical Provenance umbrella (e.g. ``commit_sha``,
            ``call_id``, ``utc_emit``).

    Returns:
        :class:`TinygradBridgeManifest` with per-tensor metadata + canonical
        Provenance.
    """
    _, manifest = tinygrad_state_dict_to_zip_member_bytes(state_dict, archive_name=archive_name)
    if extra_provenance:
        merged = dict(manifest.canonical_provenance)
        merged.update(extra_provenance)
        # Build a fresh frozen manifest with the merged provenance.
        return TinygradBridgeManifest(
            schema_version=manifest.schema_version,
            tensor_count=manifest.tensor_count,
            total_uncompressed_bytes=manifest.total_uncompressed_bytes,
            compressed_bytes=manifest.compressed_bytes,
            per_tensor_shapes=manifest.per_tensor_shapes,
            per_tensor_dtypes=manifest.per_tensor_dtypes,
            zip_member_name=manifest.zip_member_name,
            canonical_provenance=merged,
        )
    return manifest


def tinygrad_with_numpy_inflate_bridge(fn: F) -> F:
    """Decorator: resolve tinygrad-at-training + numpy-at-inflate routing.

    Sister of :func:`tac.framework_agnostic.decorators.mlx_first_with_numpy_fallback`
    for the tinygrad backend. The decorated function MUST accept a ``backend``
    keyword argument. If tinygrad is unavailable, falls back to numpy
    (deterministic + canonical bridge contract per Catalog #146).

    Per CLAUDE.md "Forbidden device-selection defaults (the MPS-fallback
    trap)" + Catalog #1 sister: the fallback is to NUMPY (deterministic;
    canonical bridge contract), NOT to MPS (silent + non-promotable). This
    decorator is therefore safe-by-construction.

    Returns:
        Decorated function that resolves to Backend.TINYGRAD or Backend.NUMPY
        at call-time.
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Caller's explicit override beats decorator priority.
        if "backend" in kwargs and kwargs["backend"] is not None:
            return fn(*args, **kwargs)
        if is_tinygrad_available():
            backend = Backend.TINYGRAD
        else:
            backend = Backend.NUMPY
        return fn(*args, backend=backend, **kwargs)

    return wrapper  # type: ignore[return-value]


__all__ = [
    "BRIDGE_MANIFEST_SCHEMA",
    "DEFAULT_ZIP_MEMBER_NAME",
    "EVIDENCE_GRADE_TINYGRAD",
    "EVIDENCE_TAG_TINYGRAD",
    "LANE_ID",
    "TINYGRAD_BRIDGE_SCHEMA",
    "TinygradBridgeManifest",
    "build_tinygrad_bridge_manifest",
    "is_tinygrad_available",
    "load_tinygrad_trained_weights_for_numpy_inflate",
    "tinygrad_state_dict_to_zip_member_bytes",
    "tinygrad_with_numpy_inflate_bridge",
]
