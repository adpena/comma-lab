# SPDX-License-Identifier: MIT
"""Tests for the WAVE-3-AUTO-TRIGGER-RUNTIME-WIRE-IN runtime hook.

Per closure of the orphan-signal gap from commit ``7b9d5e280``
(PER-BYTE-METHODOLOGY-FOLLOWUP) which landed the structural consumer
package at ``src/tac/cathedral_consumers/auto_trigger_similarity_after_master_gradient_anchor_consumer/``
but explicitly deferred the runtime hook into
:func:`tac.master_gradient.append_anchor_locked`.

Sister of:
- :func:`tac.canonical_frontier_pointer.auto_refresh_canonical_frontier_after_dispatch_outcome`
  (Catalog #343 sister hook pattern at the Modal call_id ledger surface)
- ``src/tac/tests/test_modal_call_id_ledger.py`` (canonical 4-layer ledger
  sister tests; same post-canonical-event hook pattern)

Coverage:

* Runtime hook fires the opt-in consumer's ``update_from_anchor`` after
  the fcntl-locked append succeeds.
* Idempotency: re-running the hook with the same anchor row produces the
  same per-consumer side-effect (the canonical consumer is observability-
  only; this asserts no state corruption).
* Exception swallowing: a raising consumer does NOT block the ledger
  write OR sister consumers.
* Non-matching consumers (``CONSUMES_MASTER_GRADIENT_ANCHORS = False``)
  are skipped — the gate is opt-in not opt-out.
* The canonical auto_trigger consumer specifically opt-ins + receives.
* Live-repo regression: existing master_gradient append + read paths
  still pass (the hook is additive + fail-quiet).
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

try:
    import numpy as np
except ImportError:  # pragma: no cover — minimal env
    np = None

from tac.master_gradient import (
    MasterGradient,
    OperatingPoint,
    append_anchor_locked,
    load_anchors_lenient,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _build_synthetic_anchor(tmp_path: Path) -> MasterGradient:
    """Construct a minimal MasterGradient suitable for ledger append tests.

    Mirrors the canonical pattern in
    ``src/tac/tests/test_extract_master_gradient.py::test_append_anchor_locked_roundtrip``.
    """
    if np is None:
        pytest.skip("numpy unavailable in minimal env")
    sidecar = tmp_path / "grad.npy"
    np.save(sidecar, np.zeros((10, 3), dtype=np.float32))
    op = OperatingPoint(d_seg=0.05, d_pose=1e-5, rate=0.119, score=0.193)
    return MasterGradient(
        archive_sha256="a" * 64,
        operating_point=op,
        gradient_array_path=str(sidecar),
        n_bytes=10,
        measurement_method="autograd_per_parameter_projected_fec6_int8_fp16_jacobian",
        measurement_axis="[contest-CPU]",
        measurement_hardware="linux_x86_64_modal_cpu",
        measurement_call_id=None,
        measurement_utc="2026-05-17T12:00:00+00:00",
        score_axis_dominance={
            "schema": "fixture_dominance_v1",
            "pose_axis_dominant_byte_count": 2,
        },
    )


def _install_mock_consumer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    consumes: bool,
    side_effect: Exception | None = None,
    received: list | None = None,
) -> types.ModuleType:
    """Construct + register a synthetic consumer module the discovery loop returns.

    Patches
    :func:`tools.cathedral_autopilot_autonomous_loop.discover_compliant_consumer_modules`
    to return the synthetic module so the test is hermetic (does NOT depend
    on the live ``src/tac/cathedral_consumers/`` set).
    """
    mod = types.ModuleType(name)
    mod.CONSUMES_MASTER_GRADIENT_ANCHORS = consumes
    mod.CONSUMER_NAME = name

    received_list = received if received is not None else []

    def _update_from_anchor(anchor):
        if side_effect is not None:
            raise side_effect
        received_list.append(anchor)

    mod.update_from_anchor = _update_from_anchor
    mod._received = received_list  # accessor for tests

    sys.modules[name] = mod
    return mod


def _patch_discovery(
    monkeypatch: pytest.MonkeyPatch, modules: list
) -> None:
    """Patch the discovery helper to return the supplied module list."""
    import tools.cathedral_autopilot_autonomous_loop as loop_module

    monkeypatch.setattr(
        loop_module,
        "discover_compliant_consumer_modules",
        lambda *args, **kwargs: modules,
    )


# ---------------------------------------------------------------------------
# Catalog #335 / opt-in marker discovery
# ---------------------------------------------------------------------------


def test_canonical_auto_trigger_consumer_declares_opt_in():
    """The canonical auto_trigger consumer must declare CONSUMES_MASTER_GRADIENT_ANCHORS=True."""
    mod = importlib.import_module(
        "tac.cathedral_consumers."
        "auto_trigger_similarity_after_master_gradient_anchor_consumer"
    )
    assert getattr(mod, "CONSUMES_MASTER_GRADIENT_ANCHORS", False) is True


def test_canonical_auto_trigger_consumer_update_from_anchor_callable():
    """The canonical consumer must expose update_from_anchor."""
    mod = importlib.import_module(
        "tac.cathedral_consumers."
        "auto_trigger_similarity_after_master_gradient_anchor_consumer"
    )
    assert callable(mod.update_from_anchor)


# ---------------------------------------------------------------------------
# Runtime hook fires after append
# ---------------------------------------------------------------------------


def test_runtime_hook_fires_for_opt_in_consumer(tmp_path, monkeypatch):
    received: list = []
    _install_mock_consumer(
        monkeypatch,
        name="tests.synthetic_opt_in_consumer_a",
        consumes=True,
        received=received,
    )
    _patch_discovery(
        monkeypatch, [sys.modules["tests.synthetic_opt_in_consumer_a"]]
    )
    grad = _build_synthetic_anchor(tmp_path)
    append_anchor_locked(
        grad, path=tmp_path / "anchors.jsonl", lock_path=tmp_path / ".lock"
    )
    # Hook fired exactly once, anchor row is a dict with canonical fields.
    assert len(received) == 1
    anchor = received[0]
    assert isinstance(anchor, dict)
    assert anchor["archive_sha256"] == "a" * 64
    assert anchor["measurement_axis"] == "[contest-CPU]"
    assert anchor["schema_version"] == "master_gradient_anchor_v1"


def test_non_opt_in_consumer_skipped(tmp_path, monkeypatch):
    received: list = []
    _install_mock_consumer(
        monkeypatch,
        name="tests.synthetic_non_opt_in_consumer",
        consumes=False,
        received=received,
    )
    _patch_discovery(
        monkeypatch, [sys.modules["tests.synthetic_non_opt_in_consumer"]]
    )
    grad = _build_synthetic_anchor(tmp_path)
    append_anchor_locked(
        grad, path=tmp_path / "anchors.jsonl", lock_path=tmp_path / ".lock"
    )
    assert len(received) == 0


def test_consumer_without_marker_attr_skipped(tmp_path, monkeypatch):
    """Consumer that omits CONSUMES_MASTER_GRADIENT_ANCHORS treated as opt-out."""
    mod = types.ModuleType("tests.synthetic_no_marker_consumer")
    received: list = []
    mod.update_from_anchor = lambda anchor: received.append(anchor)
    sys.modules["tests.synthetic_no_marker_consumer"] = mod
    # Note: no CONSUMES_MASTER_GRADIENT_ANCHORS attribute at all.
    _patch_discovery(monkeypatch, [mod])
    grad = _build_synthetic_anchor(tmp_path)
    append_anchor_locked(
        grad, path=tmp_path / "anchors.jsonl", lock_path=tmp_path / ".lock"
    )
    assert len(received) == 0


# ---------------------------------------------------------------------------
# Exception swallowing — ledger write must succeed even when consumer raises
# ---------------------------------------------------------------------------


def test_raising_consumer_does_not_block_ledger(tmp_path, monkeypatch, caplog):
    _install_mock_consumer(
        monkeypatch,
        name="tests.synthetic_raising_consumer",
        consumes=True,
        side_effect=RuntimeError("synthetic consumer failure"),
    )
    _patch_discovery(
        monkeypatch, [sys.modules["tests.synthetic_raising_consumer"]]
    )
    grad = _build_synthetic_anchor(tmp_path)
    ledger = tmp_path / "anchors.jsonl"
    # Must NOT raise.
    append_anchor_locked(grad, path=ledger, lock_path=tmp_path / ".lock")
    # Ledger row landed.
    rows = load_anchors_lenient(ledger)
    assert len(rows) == 1
    assert rows[0]["archive_sha256"] == "a" * 64


def test_raising_consumer_does_not_block_sister_consumer(tmp_path, monkeypatch):
    """A raising consumer must NOT prevent sister consumers from receiving."""
    sister_received: list = []
    _install_mock_consumer(
        monkeypatch,
        name="tests.synthetic_raising_consumer_a",
        consumes=True,
        side_effect=ValueError("boom"),
    )
    _install_mock_consumer(
        monkeypatch,
        name="tests.synthetic_sister_consumer_b",
        consumes=True,
        received=sister_received,
    )
    _patch_discovery(
        monkeypatch,
        [
            sys.modules["tests.synthetic_raising_consumer_a"],
            sys.modules["tests.synthetic_sister_consumer_b"],
        ],
    )
    grad = _build_synthetic_anchor(tmp_path)
    append_anchor_locked(
        grad, path=tmp_path / "anchors.jsonl", lock_path=tmp_path / ".lock"
    )
    assert len(sister_received) == 1


def test_discovery_failure_does_not_block_ledger(tmp_path, monkeypatch):
    """A failing discovery helper must NOT block the canonical write."""
    import tools.cathedral_autopilot_autonomous_loop as loop_module

    def _raise_on_discover(*args, **kwargs):
        raise RuntimeError("synthetic discovery failure")

    monkeypatch.setattr(
        loop_module,
        "discover_compliant_consumer_modules",
        _raise_on_discover,
    )
    grad = _build_synthetic_anchor(tmp_path)
    ledger = tmp_path / "anchors.jsonl"
    append_anchor_locked(grad, path=ledger, lock_path=tmp_path / ".lock")
    rows = load_anchors_lenient(ledger)
    assert len(rows) == 1


def test_consumer_with_non_callable_hook_skipped(tmp_path, monkeypatch):
    """If update_from_anchor is present but not callable, consumer is skipped."""
    mod = types.ModuleType("tests.synthetic_non_callable_hook_consumer")
    mod.CONSUMES_MASTER_GRADIENT_ANCHORS = True
    mod.update_from_anchor = "not_a_callable"  # type: ignore[assignment]
    sys.modules["tests.synthetic_non_callable_hook_consumer"] = mod
    _patch_discovery(monkeypatch, [mod])
    grad = _build_synthetic_anchor(tmp_path)
    ledger = tmp_path / "anchors.jsonl"
    # Must NOT raise.
    append_anchor_locked(grad, path=ledger, lock_path=tmp_path / ".lock")
    rows = load_anchors_lenient(ledger)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# Idempotency — re-running the hook produces the same observable state
# ---------------------------------------------------------------------------


def test_hook_idempotent_on_repeated_append(tmp_path, monkeypatch):
    received: list = []
    _install_mock_consumer(
        monkeypatch,
        name="tests.synthetic_idempotent_consumer",
        consumes=True,
        received=received,
    )
    _patch_discovery(
        monkeypatch, [sys.modules["tests.synthetic_idempotent_consumer"]]
    )
    grad = _build_synthetic_anchor(tmp_path)
    ledger = tmp_path / "anchors.jsonl"
    lock = tmp_path / ".lock"
    append_anchor_locked(grad, path=ledger, lock_path=lock)
    append_anchor_locked(grad, path=ledger, lock_path=lock)
    append_anchor_locked(grad, path=ledger, lock_path=lock)
    # Hook fired once per append (the canonical contract; consumer is
    # observability-only so receives every event).
    assert len(received) == 3
    # All three are dicts with the same canonical archive sha.
    for entry in received:
        assert entry["archive_sha256"] == "a" * 64
    # Ledger has 3 append-only rows.
    rows = load_anchors_lenient(ledger)
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# Multi-consumer fan-out + opt-in filtering
# ---------------------------------------------------------------------------


def test_multi_consumer_fan_out_only_opt_in_receive(tmp_path, monkeypatch):
    opt_in_received: list = []
    other_opt_in_received: list = []
    skipped_received: list = []

    _install_mock_consumer(
        monkeypatch,
        name="tests.synthetic_multi_opt_in_a",
        consumes=True,
        received=opt_in_received,
    )
    _install_mock_consumer(
        monkeypatch,
        name="tests.synthetic_multi_opt_in_b",
        consumes=True,
        received=other_opt_in_received,
    )
    _install_mock_consumer(
        monkeypatch,
        name="tests.synthetic_multi_skipped_c",
        consumes=False,
        received=skipped_received,
    )
    _patch_discovery(
        monkeypatch,
        [
            sys.modules["tests.synthetic_multi_opt_in_a"],
            sys.modules["tests.synthetic_multi_opt_in_b"],
            sys.modules["tests.synthetic_multi_skipped_c"],
        ],
    )
    grad = _build_synthetic_anchor(tmp_path)
    append_anchor_locked(
        grad, path=tmp_path / "anchors.jsonl", lock_path=tmp_path / ".lock"
    )
    assert len(opt_in_received) == 1
    assert len(other_opt_in_received) == 1
    assert len(skipped_received) == 0


# ---------------------------------------------------------------------------
# Live-repo regression — existing append round-trip still works
# ---------------------------------------------------------------------------


def test_existing_append_round_trip_unchanged(tmp_path, monkeypatch):
    """Sister regression: hook addition must not break the canonical round-trip.

    Patches discovery to empty so we exercise the canonical ledger path
    without any consumer fan-out (mirrors environments where the discovery
    helper is unavailable / no consumers are opt-in).
    """
    _patch_discovery(monkeypatch, [])
    grad = _build_synthetic_anchor(tmp_path)
    ledger = tmp_path / "anchors.jsonl"
    append_anchor_locked(grad, path=ledger, lock_path=tmp_path / ".lock")
    rows = load_anchors_lenient(ledger)
    assert len(rows) == 1
    row = rows[0]
    assert row["archive_sha256"] == "a" * 64
    assert row["measurement_axis"] == "[contest-CPU]"
    assert row["measurement_hardware"] == "linux_x86_64_modal_cpu"
    assert row["schema_version"] == "master_gradient_anchor_v1"


def test_canonical_consumer_received_via_live_discovery(tmp_path, monkeypatch):
    """End-to-end via canonical (non-patched) discovery: auto_trigger consumer receives.

    Uses live discovery (no patching) to verify the canonical auto_trigger
    consumer at
    ``tac.cathedral_consumers.auto_trigger_similarity_after_master_gradient_anchor_consumer``
    actually receives the anchor when its module is on disk.

    The canonical consumer's ``update_from_anchor`` is observability-only
    (no observable side-effect), so we verify by patching the consumer
    module's hook to record receipt.
    """
    canonical = importlib.import_module(
        "tac.cathedral_consumers."
        "auto_trigger_similarity_after_master_gradient_anchor_consumer"
    )
    received: list = []
    original_hook = canonical.update_from_anchor

    def _spy(anchor):
        received.append(anchor)
        return original_hook(anchor)

    monkeypatch.setattr(canonical, "update_from_anchor", _spy)
    grad = _build_synthetic_anchor(tmp_path)
    append_anchor_locked(
        grad, path=tmp_path / "anchors.jsonl", lock_path=tmp_path / ".lock"
    )
    # The canonical consumer SHOULD have received the anchor row.
    assert len(received) >= 1, (
        "canonical auto_trigger consumer did NOT receive the anchor; "
        "runtime hook wire-in is broken"
    )
    assert any(r.get("archive_sha256") == "a" * 64 for r in received)
