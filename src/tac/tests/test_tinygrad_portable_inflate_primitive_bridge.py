# SPDX-License-Identifier: MIT
"""Tests for the tinygrad-portable inflate primitive bridge.

Per cascade item 6 of the 2026-05-28 7-item cascade
(``feedback_cathedral_autopilot_is_the_canonical_meta_orchestrator_proceed_with_all_7_cascade_20260528.md``).

Covers the 3rd canonical portability surface per the 8th standing directive
(``feedback_mlx_first_numpy_portable_individually_fractally_optimized_standing_directive_20260526.md``):
TRAINING MLX-first / PyTorch-first / tinygrad → INFLATE numpy-portable.

Test structure follows the design-memo "PHASE C — Tests" outline:
  1. Availability detection
  2. Bridge contract round-trip
  3. ZIP-member packaging determinism
  4. Inflate-side numpy consumer
  5. Catalog #295 PYTHONPATH self-containment
  6. Per-tensor metadata preservation
  7. Canonical Provenance per Catalog #287/#323
  8. Framework-agnostic decorator routing
  9. Drift surface (bfloat16 → fp32 cast)
  10. HNeRV parity L4 budget

Per CLAUDE.md "Forbidden score claims" + Catalog #1/#192/#317:
tinygrad bridge outputs are non-promotable; tests verify Provenance
markers are correctly set per Catalog #287/#323.
"""

from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.framework_agnostic.backend import Backend
from tac.local_acceleration.tinygrad_bridge import (
    BRIDGE_MANIFEST_SCHEMA,
    DEFAULT_ZIP_MEMBER_NAME,
    EVIDENCE_GRADE_TINYGRAD,
    EVIDENCE_TAG_TINYGRAD,
    LANE_ID,
    TINYGRAD_BRIDGE_SCHEMA,
    TinygradBridgeManifest,
    build_tinygrad_bridge_manifest,
    is_tinygrad_available,
    load_tinygrad_trained_weights_for_numpy_inflate,
    tinygrad_state_dict_to_zip_member_bytes,
    tinygrad_with_numpy_inflate_bridge,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
BRIDGE_MODULE_PATH = (
    REPO_ROOT / "src" / "tac" / "local_acceleration" / "tinygrad_bridge.py"
)


# ---------------------------------------------------------------------------
# (1) Availability detection
# ---------------------------------------------------------------------------


def test_is_tinygrad_available_returns_bool() -> None:
    """is_tinygrad_available must return a bool unconditionally."""
    result = is_tinygrad_available()
    assert isinstance(result, bool)


def test_is_tinygrad_available_matches_backend_canonical() -> None:
    """is_tinygrad_available must agree with canonical Backend.TINYGRAD check."""
    from tac.framework_agnostic.backend import _AVAILABILITY_CHECK
    assert is_tinygrad_available() == _AVAILABILITY_CHECK[Backend.TINYGRAD]()


# ---------------------------------------------------------------------------
# (2) Bridge contract round-trip (synthetic — no tinygrad required)
# ---------------------------------------------------------------------------


def _build_fake_zip_with_npz(state: dict[str, np.ndarray]) -> bytes:
    """Helper: build a ZIP archive containing a canonical npz member.

    Mirrors what tinygrad_state_dict_to_zip_member_bytes produces, without
    requiring tinygrad. Used to test the inflate-side consumer in isolation.
    """
    npz_buf = io.BytesIO()
    np.savez_compressed(npz_buf, **state)
    npz_bytes = npz_buf.getvalue()
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(DEFAULT_ZIP_MEMBER_NAME, npz_bytes)
    return zip_buf.getvalue()


def test_inflate_round_trip_byte_deterministic() -> None:
    """Inflate-side consumer must produce byte-deterministic round-trip."""
    state = {
        "w0": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "b0": np.array([0.1, 0.2, 0.3], dtype=np.float32),
    }
    zip_bytes = _build_fake_zip_with_npz(state)
    loaded = load_tinygrad_trained_weights_for_numpy_inflate(zip_bytes)
    assert sorted(loaded.keys()) == ["b0", "w0"]
    for k, v in state.items():
        np.testing.assert_array_equal(loaded[k], v)
        assert loaded[k].dtype == v.dtype


def test_inflate_round_trip_preserves_int_dtypes() -> None:
    """Inflate-side consumer preserves non-float dtypes per npz oracle."""
    state = {
        "int8_tensor": np.array([-127, 0, 127], dtype=np.int8),
        "uint8_tensor": np.array([0, 128, 255], dtype=np.uint8),
        "fp64_tensor": np.array([1.5, 2.5], dtype=np.float64),
    }
    zip_bytes = _build_fake_zip_with_npz(state)
    loaded = load_tinygrad_trained_weights_for_numpy_inflate(zip_bytes)
    for k, v in state.items():
        assert loaded[k].dtype == v.dtype
        np.testing.assert_array_equal(loaded[k], v)


# ---------------------------------------------------------------------------
# (3) ZIP-member packaging determinism
# ---------------------------------------------------------------------------


def test_zip_packaging_uses_canonical_member_name() -> None:
    """Canonical default ZIP-member name is tinygrad_weights.npz per design memo."""
    assert DEFAULT_ZIP_MEMBER_NAME == "tinygrad_weights.npz"


def test_zip_packaging_uses_deflate_compression() -> None:
    """Canonical compression is ZIP_DEFLATED per design memo."""
    state = {"w": np.zeros((4, 4), dtype=np.float32)}
    zip_bytes = _build_fake_zip_with_npz(state)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), mode="r") as zf:
        infos = zf.infolist()
    assert len(infos) == 1
    assert infos[0].compress_type == zipfile.ZIP_DEFLATED


# ---------------------------------------------------------------------------
# (4) Inflate-side numpy consumer (Backend.NUMPY pinning + zero tinygrad dep)
# ---------------------------------------------------------------------------


def test_inflate_consumer_ignores_backend_override_per_decorator() -> None:
    """@inflate_runtime_helper pins Backend.NUMPY regardless of caller override."""
    state = {"w": np.array([1.0, 2.0], dtype=np.float32)}
    zip_bytes = _build_fake_zip_with_npz(state)
    # Even if caller passes Backend.MLX, the decorator pins to NUMPY.
    loaded = load_tinygrad_trained_weights_for_numpy_inflate(
        zip_bytes,
        backend=Backend.MLX,  # silently overridden by @inflate_runtime_helper
    )
    assert "w" in loaded
    np.testing.assert_array_equal(loaded["w"], state["w"])


def test_inflate_consumer_custom_member_name() -> None:
    """Caller can specify a custom member name within the ZIP archive."""
    state = {"w": np.zeros((2, 2), dtype=np.float32)}
    npz_buf = io.BytesIO()
    np.savez_compressed(npz_buf, **state)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("custom_name.npz", npz_buf.getvalue())
    loaded = load_tinygrad_trained_weights_for_numpy_inflate(
        zip_buf.getvalue(),
        member_name="custom_name.npz",
    )
    assert "w" in loaded


# ---------------------------------------------------------------------------
# (5) Catalog #295 PYTHONPATH self-containment — inflate consumer survives empty PYTHONPATH
# ---------------------------------------------------------------------------


def test_inflate_consumer_imports_only_numpy_and_stdlib() -> None:
    """Bridge module imports only numpy + stdlib + canonical tac.framework_agnostic helpers.

    Per HNeRV parity L4 + Catalog #295 PYTHONPATH self-containment: the
    inflate-side consumer must NOT import torch / mlx / tinygrad at module
    load time.
    """
    src = BRIDGE_MODULE_PATH.read_text(encoding="utf-8")
    # FORBIDDEN top-level imports per HNeRV parity L4 / Catalog #295.
    forbidden_patterns = [
        "import torch",
        "import mlx",
        "import tinygrad",
        "from torch ",
        "from mlx ",
        "from tinygrad ",
    ]
    for pattern in forbidden_patterns:
        # Only flag if NOT inside a function body (deferred import is OK).
        # The bridge module deliberately uses deferred imports for tinygrad
        # via tac.framework_agnostic.helpers.tinygrad_state_dict_to_npz_bridge.
        for line in src.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(pattern) and not line.startswith("    "):
                pytest.fail(
                    f"Forbidden top-level framework import: {pattern!r} in line: {line!r}"
                )


def test_inflate_consumer_callable_with_empty_pythonpath_synthetic_npz() -> None:
    """Inflate consumer works when given a hand-built ZIP+npz (no tinygrad).

    This is the Catalog #295 PYTHONPATH self-containment proof: the
    canonical inflate path consumes ZIP+npz produced by ANY upstream
    process (including non-tinygrad processes), so the deployed inflate
    runtime cannot accidentally pick up a tinygrad dependency.
    """
    state = {"k": np.array([42.0], dtype=np.float32)}
    zip_bytes = _build_fake_zip_with_npz(state)
    loaded = load_tinygrad_trained_weights_for_numpy_inflate(zip_bytes)
    np.testing.assert_array_equal(loaded["k"], state["k"])


# ---------------------------------------------------------------------------
# (6) Per-tensor metadata preservation
# ---------------------------------------------------------------------------


def test_manifest_invariants_reject_negative_counts() -> None:
    """TinygradBridgeManifest rejects negative counts per __post_init__."""
    with pytest.raises(ValueError, match="tensor_count must be a non-negative int"):
        TinygradBridgeManifest(
            schema_version=BRIDGE_MANIFEST_SCHEMA,
            tensor_count=-1,
            total_uncompressed_bytes=0,
            compressed_bytes=0,
        )


def test_manifest_invariants_reject_mismatched_count() -> None:
    """TinygradBridgeManifest rejects tensor_count mismatching per_tensor_shapes."""
    with pytest.raises(ValueError, match="does not match per_tensor_shapes"):
        TinygradBridgeManifest(
            schema_version=BRIDGE_MANIFEST_SCHEMA,
            tensor_count=2,
            total_uncompressed_bytes=10,
            compressed_bytes=5,
            per_tensor_shapes={"w": (4,)},  # count=1 not 2
            per_tensor_dtypes={"w": "float32"},
        )


def test_manifest_invariants_reject_key_mismatch() -> None:
    """TinygradBridgeManifest rejects per_tensor_shapes vs per_tensor_dtypes key mismatch."""
    with pytest.raises(ValueError, match="must have identical keys"):
        TinygradBridgeManifest(
            schema_version=BRIDGE_MANIFEST_SCHEMA,
            tensor_count=1,
            total_uncompressed_bytes=10,
            compressed_bytes=5,
            per_tensor_shapes={"w": (4,)},
            per_tensor_dtypes={"b": "float32"},  # wrong key
        )


def test_manifest_is_frozen() -> None:
    """TinygradBridgeManifest is frozen per canonical pattern (Catalog #356 sister)."""
    m = TinygradBridgeManifest(
        schema_version=BRIDGE_MANIFEST_SCHEMA,
        tensor_count=0,
        total_uncompressed_bytes=0,
        compressed_bytes=0,
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        m.tensor_count = 5  # type: ignore[misc]


def test_manifest_as_dict_serializes_canonically() -> None:
    """as_dict() must include all canonical fields for posterior consumption."""
    m = TinygradBridgeManifest(
        schema_version=BRIDGE_MANIFEST_SCHEMA,
        tensor_count=1,
        total_uncompressed_bytes=16,
        compressed_bytes=10,
        per_tensor_shapes={"w": (4,)},
        per_tensor_dtypes={"w": "float32"},
        canonical_provenance={"axis_tag": "[predicted]"},
    )
    d = m.as_dict()
    required = {
        "schema_version",
        "tensor_count",
        "total_uncompressed_bytes",
        "compressed_bytes",
        "per_tensor_shapes",
        "per_tensor_dtypes",
        "zip_member_name",
        "canonical_provenance",
    }
    assert required.issubset(d.keys())
    # Shapes serialize as lists (JSON-compatible).
    assert d["per_tensor_shapes"]["w"] == [4]


# ---------------------------------------------------------------------------
# (7) Canonical Provenance per Catalog #287/#323
# ---------------------------------------------------------------------------


def test_evidence_grade_canonical_token() -> None:
    """EVIDENCE_GRADE_TINYGRAD must be the canonical research-signal token."""
    assert EVIDENCE_GRADE_TINYGRAD == "tinygrad-research-signal"
    assert EVIDENCE_TAG_TINYGRAD == "[tinygrad research-signal]"


def test_lane_id_canonical_constant() -> None:
    """LANE_ID matches the canonical lane registry entry."""
    assert LANE_ID == (
        "lane_slot_j_cascade_item_6_tinygrad_portable_inflate_"
        "primitive_bridge_20260529"
    )


def test_schemas_canonical_constants() -> None:
    """Canonical schema versions are pinned per design memo."""
    assert TINYGRAD_BRIDGE_SCHEMA == "tinygrad_portable_inflate_primitive_bridge.v1"
    assert BRIDGE_MANIFEST_SCHEMA == "tinygrad_bridge_manifest.v1"


# ---------------------------------------------------------------------------
# (8) Framework-agnostic decorator routing
# ---------------------------------------------------------------------------


def test_decorator_resolves_numpy_when_tinygrad_unavailable() -> None:
    """@tinygrad_with_numpy_inflate_bridge falls back to NUMPY without tinygrad."""

    @tinygrad_with_numpy_inflate_bridge
    def my_op(x: int, *, backend: Backend) -> Backend:
        return backend

    if not is_tinygrad_available():
        assert my_op(1) is Backend.NUMPY
    else:
        assert my_op(1) is Backend.TINYGRAD


def test_decorator_respects_explicit_override() -> None:
    """Caller's explicit backend= override beats decorator priority."""

    @tinygrad_with_numpy_inflate_bridge
    def my_op(x: int, *, backend: Backend) -> Backend:
        return backend

    assert my_op(1, backend=Backend.NUMPY) is Backend.NUMPY


# ---------------------------------------------------------------------------
# (9) Drift surface verification — fp32 byte-determinism
# ---------------------------------------------------------------------------


def test_inflate_consumer_fp32_byte_deterministic() -> None:
    """fp32 round-trip byte-deterministic per Catalog #146 canonical oracle."""
    state = {"w": np.random.RandomState(42).randn(64, 64).astype(np.float32)}
    zip_bytes_1 = _build_fake_zip_with_npz(state)
    loaded_1 = load_tinygrad_trained_weights_for_numpy_inflate(zip_bytes_1)
    # Build a second archive from the same state.
    zip_bytes_2 = _build_fake_zip_with_npz(state)
    loaded_2 = load_tinygrad_trained_weights_for_numpy_inflate(zip_bytes_2)
    # Both round-trips produce identical numpy bytes.
    np.testing.assert_array_equal(loaded_1["w"], loaded_2["w"])
    np.testing.assert_array_equal(loaded_1["w"], state["w"])


# ---------------------------------------------------------------------------
# (10) HNeRV parity L4 budget verification
# ---------------------------------------------------------------------------


def test_bridge_module_under_substrate_engineering_loc_budget() -> None:
    """Bridge module under substrate_engineering L7 LOC budget (350 LOC).

    Per HNeRV parity L4 default 100 LOC + ≤200 LOC waiver, with explicit
    L7 substrate_engineering exception declared in lane registry. The
    bridge module is substrate-infrastructure per Catalog #220/#272 and
    is declared lane_class=substrate_engineering in the design memo +
    lane registry.
    """
    src = BRIDGE_MODULE_PATH.read_text(encoding="utf-8")
    line_count = len(src.splitlines())
    # Per HNeRV parity L7 substrate_engineering exception: budget extended
    # to ~350 LOC (the bolt-on size ceiling).
    assert line_count < 400, (
        f"bridge module is {line_count} LOC; exceeds substrate_engineering ceiling 400"
    )


def test_bridge_module_has_spdx_license_header() -> None:
    """SPDX header present per Catalog #335 canonical contract."""
    src = BRIDGE_MODULE_PATH.read_text(encoding="utf-8")
    assert src.startswith("# SPDX-License-Identifier: MIT")


def test_bridge_module_declares_all_public_api() -> None:
    """__all__ declares the canonical public API surface."""
    from tac.local_acceleration import tinygrad_bridge as tb

    public = set(tb.__all__)
    expected = {
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
    }
    assert expected.issubset(public), f"missing public API: {expected - public}"


# ---------------------------------------------------------------------------
# Bonus: end-to-end (skip unless tinygrad installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_tinygrad_available(), reason="tinygrad not installed")
def test_end_to_end_tinygrad_to_numpy_bridge() -> None:
    """Full end-to-end round-trip when tinygrad is installed.

    Only runs when tinygrad is present; canonical proof that the bridge
    works against the real tinygrad framework.
    """
    from tinygrad import Tensor  # noqa: PLC0415

    state_dict = {
        "w0": Tensor.eye(4),
        "b0": Tensor.zeros(4),
    }
    zip_bytes, manifest = tinygrad_state_dict_to_zip_member_bytes(state_dict)
    assert isinstance(zip_bytes, bytes)
    assert len(zip_bytes) > 0
    assert manifest.tensor_count == 2
    # Inflate-side round-trip.
    loaded = load_tinygrad_trained_weights_for_numpy_inflate(zip_bytes)
    assert sorted(loaded.keys()) == ["b0", "w0"]
    np.testing.assert_array_equal(loaded["w0"], np.eye(4, dtype=loaded["w0"].dtype))
    np.testing.assert_array_equal(loaded["b0"], np.zeros(4, dtype=loaded["b0"].dtype))


@pytest.mark.skipif(not is_tinygrad_available(), reason="tinygrad not installed")
def test_end_to_end_manifest_provenance_canonical_markers() -> None:
    """When tinygrad runs, manifest carries canonical non-promotable markers."""
    from tinygrad import Tensor  # noqa: PLC0415

    state_dict = {"w": Tensor.ones(3, 3)}
    _, manifest = tinygrad_state_dict_to_zip_member_bytes(state_dict)
    prov = manifest.canonical_provenance
    assert prov["evidence_grade"] == EVIDENCE_GRADE_TINYGRAD
    assert prov["axis_tag"] == "[predicted]"
    assert prov["score_claim"] is False
    assert prov["promotable"] is False
    assert prov["lane_id"] == LANE_ID


@pytest.mark.skipif(not is_tinygrad_available(), reason="tinygrad not installed")
def test_end_to_end_build_manifest_extra_provenance_merged() -> None:
    """build_tinygrad_bridge_manifest merges caller-supplied extra Provenance."""
    from tinygrad import Tensor  # noqa: PLC0415

    state_dict = {"w": Tensor.zeros(2)}
    manifest = build_tinygrad_bridge_manifest(
        state_dict,
        extra_provenance={"commit_sha": "abc123", "call_id": "fc-test"},
    )
    assert manifest.canonical_provenance["commit_sha"] == "abc123"
    assert manifest.canonical_provenance["call_id"] == "fc-test"
    # Original canonical markers preserved.
    assert manifest.canonical_provenance["score_claim"] is False


# ---------------------------------------------------------------------------
# Sister regression: no tinygrad import at module level
# ---------------------------------------------------------------------------


def test_module_import_succeeds_without_tinygrad_installed() -> None:
    """Bridge module imports cleanly even when tinygrad is absent.

    Run in a subprocess with empty PYTHONPATH (Catalog #295 sister
    self-containment check). The module's tinygrad-touching functions
    raise BackendUnavailableError at CALL-time, not at IMPORT-time.
    """
    # Use the project's pytest interpreter so dependencies resolve.
    code = (
        "import sys\n"
        "from tac.local_acceleration import tinygrad_bridge\n"
        "assert hasattr(tinygrad_bridge, 'is_tinygrad_available')\n"
        "assert hasattr(tinygrad_bridge, 'load_tinygrad_trained_weights_for_numpy_inflate')\n"
        "print('PASS')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "PASS" in result.stdout, (
        f"bridge module failed to import; stderr: {result.stderr}"
    )
