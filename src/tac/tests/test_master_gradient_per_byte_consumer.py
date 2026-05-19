# SPDX-License-Identifier: MIT
"""Tests for tac.master_gradient_per_byte_consumer canonical helper.

Per PER-BYTE-SENSITIVITY-WIRE-INS subagent landing 2026-05-19. Tests the
canonical reader helper that the cathedral consumer
``per_byte_sensitivity_consumer`` uses to extract per-byte sensitivity
payloads from ``.omx/state/master_gradient_anchors.jsonl``.

Sister of:
- ``src/tac/tests/test_per_byte_sensitivity_consumer.py`` (the cathedral
  consumer that consumes payloads from this helper)
- ``src/tac/tests/test_cathedral_consumer_contract.py`` (canonical contract)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.master_gradient_per_byte_consumer import (
    MasterGradientPerByteCorruptError,
    PerByteSensitivityPayload,
    load_per_byte_sensitivity_for_archive,
    payload_provenance,
    summarize_payload,
    top_k_sensitive_byte_indices,
)


# ─────────────────────────────────────────────────────────────────────────
# PerByteSensitivityPayload contract tests
# ─────────────────────────────────────────────────────────────────────────


def _make_payload(**overrides) -> PerByteSensitivityPayload:
    """Helper: synthesize a valid payload with optional field overrides."""
    defaults = dict(
        archive_sha256="a" * 64,
        gradient_array_path="/tmp/example.npy",
        n_bytes=10,
        measurement_axis="[predicted]",
        measurement_hardware="darwin_arm64_advisory",
        measurement_method="autograd_per_parameter",
        measurement_utc="2026-05-19T00:00:00Z",
        top_k_sensitivity_indices=(0, 1, 2),
        aggregate_l1_importance_sum=1.5,
        n_bytes_above_zero=8,
    )
    defaults.update(overrides)
    return PerByteSensitivityPayload(**defaults)


def test_payload_frozen_invariant() -> None:
    """Frozen dataclass refuses post-construction mutation."""
    payload = _make_payload()
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        payload.n_bytes = 99  # type: ignore[misc]


def test_payload_rejects_empty_archive_sha() -> None:
    with pytest.raises(ValueError, match="archive_sha256"):
        _make_payload(archive_sha256="")


def test_payload_rejects_empty_gradient_array_path() -> None:
    with pytest.raises(ValueError, match="gradient_array_path"):
        _make_payload(gradient_array_path="")


def test_payload_rejects_non_positive_n_bytes() -> None:
    with pytest.raises(ValueError, match="n_bytes must be positive"):
        _make_payload(n_bytes=0)


def test_payload_rejects_top_k_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        _make_payload(n_bytes=5, top_k_sensitivity_indices=(0, 1, 99))


def test_payload_rejects_non_tuple_top_k() -> None:
    with pytest.raises(TypeError, match="must be a tuple"):
        _make_payload(top_k_sensitivity_indices=[0, 1, 2])  # type: ignore[arg-type]


def test_payload_rejects_negative_aggregate_sum() -> None:
    with pytest.raises(ValueError, match="aggregate_l1_importance_sum"):
        _make_payload(aggregate_l1_importance_sum=-0.1)


def test_payload_rejects_out_of_range_n_above_zero() -> None:
    with pytest.raises(ValueError, match="n_bytes_above_zero"):
        _make_payload(n_bytes=10, n_bytes_above_zero=99)


def test_payload_accepts_empty_top_k() -> None:
    """Empty top-K is valid (array load failed; payload still constructible)."""
    p = _make_payload(top_k_sensitivity_indices=())
    assert p.top_k_sensitivity_indices == ()


# ─────────────────────────────────────────────────────────────────────────
# load_per_byte_sensitivity_for_archive tests (synthetic ledger fixtures)
# ─────────────────────────────────────────────────────────────────────────


def _write_synthetic_anchor(
    tmp_path: Path,
    *,
    archive_sha: str,
    n_bytes: int,
    measurement_axis: str = "[macOS-CPU advisory]",
    measurement_hardware: str = "darwin_arm64_advisory",
    measurement_utc: str = "2026-05-19T01:00:00Z",
    grad_shape_override: tuple[int, ...] | None = None,
) -> Path:
    """Write a synthetic master_gradient ledger + .npy array to tmp_path."""
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    npy_path = tmp_path / f"grad_{archive_sha[:8]}.npy"
    shape = grad_shape_override or (n_bytes, 3)
    arr = np.random.RandomState(42).randn(*shape).astype(np.float32)
    np.save(npy_path, arr)
    row = {
        "archive_sha256": archive_sha,
        "gradient_array_path": str(npy_path),
        "gradient_tensor_kind": "aggregate_per_byte_v1",
        "n_bytes": n_bytes,
        "operating_point": {
            "d_seg": 0.001,
            "d_pose": 0.002,
            "rate": 0.005,
            "score": 0.34,
        },
        "measurement_axis": measurement_axis,
        "measurement_hardware": measurement_hardware,
        "measurement_method": "autograd_per_parameter_test",
        "measurement_utc": measurement_utc,
        "written_at_utc": measurement_utc,
        "n_pairs_total": 600,
        "n_pairs_used": 8,
        "pareto_facets": [],
        "schema_version": "master_gradient_anchor_v1",
    }
    ledger.write_text(json.dumps(row) + "\n")
    return ledger


def test_load_returns_none_for_empty_sha() -> None:
    payload = load_per_byte_sensitivity_for_archive("")
    assert payload is None


def test_load_returns_none_for_missing_archive(tmp_path: Path) -> None:
    ledger = tmp_path / "empty.jsonl"
    ledger.write_text("")
    payload = load_per_byte_sensitivity_for_archive("a" * 64, path=ledger)
    assert payload is None


def test_load_happy_path_synthetic(tmp_path: Path) -> None:
    archive_sha = "b" * 64
    ledger = _write_synthetic_anchor(tmp_path, archive_sha=archive_sha, n_bytes=50)
    payload = load_per_byte_sensitivity_for_archive(
        archive_sha, path=ledger, top_k=10
    )
    assert payload is not None
    assert payload.archive_sha256 == archive_sha
    assert payload.n_bytes == 50
    assert payload.measurement_axis == "[macOS-CPU advisory]"
    assert len(payload.top_k_sensitivity_indices) == 10
    # Top-K indices must be unique
    assert len(set(payload.top_k_sensitivity_indices)) == 10
    # All indices in valid range
    for idx in payload.top_k_sensitivity_indices:
        assert 0 <= idx < 50
    assert payload.aggregate_l1_importance_sum > 0
    assert payload.n_bytes_above_zero > 0


def test_load_axis_filter_excludes_non_matching(tmp_path: Path) -> None:
    """axis= parameter filters by measurement_axis."""
    archive_sha = "c" * 64
    ledger = _write_synthetic_anchor(
        tmp_path,
        archive_sha=archive_sha,
        n_bytes=20,
        measurement_axis="[macOS-CPU advisory]",
    )
    # Request a different axis → no match.
    payload = load_per_byte_sensitivity_for_archive(
        archive_sha, path=ledger, axis="[contest-CUDA]"
    )
    assert payload is None
    # Request matching axis → match.
    payload = load_per_byte_sensitivity_for_archive(
        archive_sha, path=ledger, axis="[macOS-CPU advisory]"
    )
    assert payload is not None


def test_load_picks_most_recent_by_utc(tmp_path: Path) -> None:
    """When multiple anchors exist, the most-recent (by measurement_utc) wins."""
    archive_sha = "d" * 64
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    npy_old = tmp_path / "grad_old.npy"
    npy_new = tmp_path / "grad_new.npy"
    arr_old = np.zeros((10, 3), dtype=np.float32)
    arr_new = np.ones((10, 3), dtype=np.float32)
    np.save(npy_old, arr_old)
    np.save(npy_new, arr_new)
    lines = []
    for utc, npy_path in [
        ("2026-05-01T00:00:00Z", npy_old),
        ("2026-05-19T00:00:00Z", npy_new),
    ]:
        row = {
            "archive_sha256": archive_sha,
            "gradient_array_path": str(npy_path),
            "gradient_tensor_kind": "aggregate_per_byte_v1",
            "n_bytes": 10,
            "operating_point": {
                "d_seg": 0.001,
                "d_pose": 0.002,
                "rate": 0.005,
                "score": 0.34,
            },
            "measurement_axis": "[predicted]",
            "measurement_hardware": "darwin_arm64_advisory",
            "measurement_method": "autograd_per_parameter_test",
            "measurement_utc": utc,
            "written_at_utc": utc,
            "schema_version": "master_gradient_anchor_v1",
        }
        lines.append(json.dumps(row))
    ledger.write_text("\n".join(lines) + "\n")

    payload = load_per_byte_sensitivity_for_archive(archive_sha, path=ledger)
    assert payload is not None
    # The new anchor's array is all ones → aggregate L1 = 10 * 3 = 30
    # The old anchor's array is all zeros → aggregate L1 = 0
    # If we picked the most-recent, aggregate > 0
    assert payload.aggregate_l1_importance_sum > 0
    assert payload.gradient_array_path == str(npy_new)


def test_load_returns_payload_when_array_missing(tmp_path: Path) -> None:
    """Best-effort: when .npy is missing, payload still returns with empty stats."""
    archive_sha = "e" * 64
    ledger = _write_synthetic_anchor(tmp_path, archive_sha=archive_sha, n_bytes=20)
    # Delete the .npy file
    for npy in tmp_path.glob("*.npy"):
        npy.unlink()
    payload = load_per_byte_sensitivity_for_archive(archive_sha, path=ledger)
    assert payload is not None
    assert payload.aggregate_l1_importance_sum == 0.0
    assert payload.n_bytes_above_zero == 0
    assert payload.top_k_sensitivity_indices == ()


def test_load_strict_raises_on_shape_mismatch(tmp_path: Path) -> None:
    """Strict mode raises MasterGradientPerByteCorruptError on shape mismatch."""
    archive_sha = "f" * 64
    # n_bytes declared as 50 but array is shape (10, 3) → mismatch
    ledger = _write_synthetic_anchor(
        tmp_path,
        archive_sha=archive_sha,
        n_bytes=50,
        grad_shape_override=(10, 3),
    )
    with pytest.raises(MasterGradientPerByteCorruptError, match="shape"):
        load_per_byte_sensitivity_for_archive(
            archive_sha, path=ledger, strict=True
        )


def test_load_skips_non_per_byte_anchors(tmp_path: Path) -> None:
    """Anchors with gradient_tensor_kind != aggregate_per_byte_v1 are skipped."""
    archive_sha = "1" * 64
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    npy_path = tmp_path / "grad_perpair.npy"
    arr = np.zeros((10, 5, 3), dtype=np.float32)
    np.save(npy_path, arr)
    row = {
        "archive_sha256": archive_sha,
        "gradient_array_path": str(npy_path),
        "gradient_tensor_kind": "per_pair_per_byte_v1",  # NOT aggregate
        "n_bytes": 10,
        "n_pairs": 5,
        "operating_point": {
            "d_seg": 0.001,
            "d_pose": 0.002,
            "rate": 0.005,
            "score": 0.34,
        },
        "measurement_axis": "[predicted]",
        "measurement_hardware": "darwin_arm64_advisory",
        "measurement_method": "autograd_per_parameter_test",
        "measurement_utc": "2026-05-19T00:00:00Z",
        "schema_version": "master_gradient_anchor_v1",
    }
    ledger.write_text(json.dumps(row) + "\n")
    payload = load_per_byte_sensitivity_for_archive(archive_sha, path=ledger)
    assert payload is None  # filtered out


# ─────────────────────────────────────────────────────────────────────────
# top_k_sensitive_byte_indices tests
# ─────────────────────────────────────────────────────────────────────────


def test_top_k_returns_empty_for_zero_k(tmp_path: Path) -> None:
    payload = _make_payload()
    assert top_k_sensitive_byte_indices(payload, k=0) == []


def test_top_k_returns_empty_when_array_missing(tmp_path: Path) -> None:
    payload = _make_payload(gradient_array_path="/nonexistent/path.npy")
    assert top_k_sensitive_byte_indices(payload, k=10) == []


def test_top_k_descending_order(tmp_path: Path) -> None:
    """Top-K indices are returned in descending importance order."""
    archive_sha = "2" * 64
    npy_path = tmp_path / "grad_ordered.npy"
    # Build an array where byte i has importance i (so byte 9 is most important)
    n_bytes = 10
    arr = np.zeros((n_bytes, 3), dtype=np.float32)
    for i in range(n_bytes):
        arr[i, 0] = float(i)  # importance = abs(i) (seg axis only)
    np.save(npy_path, arr)
    payload = PerByteSensitivityPayload(
        archive_sha256=archive_sha,
        gradient_array_path=str(npy_path),
        n_bytes=n_bytes,
        measurement_axis="[predicted]",
        measurement_hardware="test",
        measurement_method="test",
        measurement_utc="2026-05-19T00:00:00Z",
    )
    top_5 = top_k_sensitive_byte_indices(payload, k=5)
    assert top_5 == [9, 8, 7, 6, 5]


def test_top_k_bounded_by_n_bytes(tmp_path: Path) -> None:
    """top_k > n_bytes returns at most n_bytes indices."""
    archive_sha = "3" * 64
    npy_path = tmp_path / "grad_small.npy"
    arr = np.ones((5, 3), dtype=np.float32)
    np.save(npy_path, arr)
    payload = PerByteSensitivityPayload(
        archive_sha256=archive_sha,
        gradient_array_path=str(npy_path),
        n_bytes=5,
        measurement_axis="[predicted]",
        measurement_hardware="test",
        measurement_method="test",
        measurement_utc="2026-05-19T00:00:00Z",
    )
    top_100 = top_k_sensitive_byte_indices(payload, k=100)
    assert len(top_100) == 5


def test_top_k_raises_on_shape_mismatch(tmp_path: Path) -> None:
    """Shape mismatch raises MasterGradientPerByteCorruptError."""
    archive_sha = "4" * 64
    npy_path = tmp_path / "grad_mismatch.npy"
    arr = np.zeros((3, 3), dtype=np.float32)
    np.save(npy_path, arr)
    payload = PerByteSensitivityPayload(
        archive_sha256=archive_sha,
        gradient_array_path=str(npy_path),
        n_bytes=10,  # declared 10 but array is (3, 3)
        measurement_axis="[predicted]",
        measurement_hardware="test",
        measurement_method="test",
        measurement_utc="2026-05-19T00:00:00Z",
    )
    with pytest.raises(MasterGradientPerByteCorruptError):
        top_k_sensitive_byte_indices(payload, k=5)


# ─────────────────────────────────────────────────────────────────────────
# payload_provenance + summarize_payload tests
# ─────────────────────────────────────────────────────────────────────────


def test_payload_provenance_is_predicted_grade() -> None:
    """Per Catalog #287/#323: per-byte payload provenance is PREDICTED_FROM_MODEL."""
    from tac.provenance.contract import ProvenanceEvidenceGrade, ProvenanceKind

    payload = _make_payload(archive_sha256="abc" * 21 + "d")  # 64 chars
    prov = payload_provenance(payload)
    assert prov.artifact_kind == ProvenanceKind.PREDICTED_FROM_MODEL
    assert prov.evidence_grade == ProvenanceEvidenceGrade.PREDICTED
    assert prov.promotion_eligible is False
    assert prov.score_claim_valid is False


def test_payload_provenance_propagates_hardware() -> None:
    payload = _make_payload(measurement_hardware="darwin_arm64_advisory")
    prov = payload_provenance(payload)
    assert prov.hardware_substrate == "darwin_arm64_advisory"


def test_summarize_payload_returns_dict() -> None:
    payload = _make_payload(
        archive_sha256="abc123" + "0" * 58,
        n_bytes=100,
        n_bytes_above_zero=80,
        aggregate_l1_importance_sum=1.234,
        top_k_sensitivity_indices=tuple(range(20)),
    )
    summary = summarize_payload(payload)
    assert isinstance(summary, dict)
    assert summary["archive_sha256_prefix"] == "abc123000000"
    assert summary["n_bytes"] == 100
    assert summary["n_bytes_above_zero"] == 80
    assert summary["sparsity_pct"] == pytest.approx(20.0)
    assert summary["top_k_count"] == 20
    assert summary["aggregate_l1_importance_sum"] == 1.234
    assert summary["measurement_axis"] == "[predicted]"


def test_summarize_payload_handles_zero_n_bytes_edge() -> None:
    """Pathological n_bytes=0 cannot construct payload (rejected at __post_init__)."""
    with pytest.raises(ValueError):
        _make_payload(n_bytes=0)


# ─────────────────────────────────────────────────────────────────────────
# Live-repo regression guard
# ─────────────────────────────────────────────────────────────────────────


def test_live_repo_loads_known_fec6_frontier_anchor() -> None:
    """Regression guard: the known FEC6 frontier archive has a per-byte anchor.

    Per `feedback_master_gradient_consumer_cathedral_wire_in_landed_20260519.md`
    + the live `.omx/state/master_gradient_anchors.jsonl`, archive
    ``6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`` carries
    an aggregate_per_byte_v1 anchor extracted on macOS advisory hardware.
    This regression test verifies the canonical loader can read it.
    """
    fec6_frontier_sha = (
        "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
    )
    # Load using default path (live ledger). May be None on a fresh clone
    # without the ledger; treat that as a soft skip.
    repo_root = Path(__file__).parent.parent.parent.parent
    ledger_path = repo_root / ".omx" / "state" / "master_gradient_anchors.jsonl"
    if not ledger_path.exists():
        pytest.skip("live master_gradient_anchors.jsonl missing on this checkout")
    payload = load_per_byte_sensitivity_for_archive(
        fec6_frontier_sha, path=ledger_path, top_k=10
    )
    if payload is None:
        pytest.skip(
            "FEC6 frontier per-byte anchor not in this ledger snapshot"
        )
    # If the anchor exists, validate the canonical structure.
    assert payload.archive_sha256 == fec6_frontier_sha
    assert payload.n_bytes > 0
    # macOS advisory hardware was the original measurement substrate.
    assert "advisory" in payload.measurement_hardware.lower() or \
           "darwin" in payload.measurement_hardware.lower() or \
           "macos" in payload.measurement_hardware.lower()
