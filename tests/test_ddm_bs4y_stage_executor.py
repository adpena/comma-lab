"""Focused controls for the BS4Y stages 1-4 executor.

These cover the pure logic that guards the measurement: the in-compile
compensation binding (the qs4 stale-compensation cure), the retained-JSON
non-finite guard, and the storage waterfall's fail-closed direction.
"""

from __future__ import annotations

import json

import pytest

from experiments import ddm_bs4y_stage_executor as bs4y

FIELD = {"bytes": 117_964_800, "sha256": "a" * 64}
MASTER = {"path": "/tmp/master.npy", "bytes": 3_052_008, "sha256": "b" * 64}
ARCHIVE_SHA = "c" * 64


def _binding(fingerprint: str) -> dict:
    return {
        "schema": "ddm_bs4y_compensation_object_binding.v1",
        "pair": 26,
        "semantic_field": FIELD,
        "master_camera": MASTER,
        "carrier_archive_sha256": ARCHIVE_SHA,
        "fingerprint_sha256": fingerprint,
    }


def test_fingerprint_is_object_bound_and_path_independent() -> None:
    first = bs4y.compensation_object_fingerprint(
        pair=26, semantic_field=FIELD, master_camera=MASTER, carrier_archive_sha256=ARCHIVE_SHA
    )
    moved = dict(MASTER, path="/elsewhere/master.npy")
    assert first == bs4y.compensation_object_fingerprint(
        pair=26, semantic_field=FIELD, master_camera=moved, carrier_archive_sha256=ARCHIVE_SHA
    )
    # A changed frame-1 object must change the fingerprint.
    changed = dict(MASTER, sha256="d" * 64)
    assert first != bs4y.compensation_object_fingerprint(
        pair=26, semantic_field=FIELD, master_camera=changed, carrier_archive_sha256=ARCHIVE_SHA
    )
    # A different carrier object must change it too; no cross-object transfer.
    assert first != bs4y.compensation_object_fingerprint(
        pair=26, semantic_field=FIELD, master_camera=MASTER, carrier_archive_sha256="e" * 64
    )


def test_compensation_assert_refuses_a_missing_binding() -> None:
    with pytest.raises(bs4y.BS4YError, match="carries no compensation-object binding"):
        bs4y.assert_compensation_matches_compile_object(
            {"pair": 26, "solve": {}},
            semantic_field=FIELD,
            carrier_archive_sha256=ARCHIVE_SHA,
        )


def test_compensation_assert_refuses_a_stale_solve(tmp_path) -> None:
    master_path = tmp_path / "master.npy"
    master_path.write_bytes(b"born-small master")
    record = bs4y.file_fact(master_path)
    fingerprint = bs4y.compensation_object_fingerprint(
        pair=26, semantic_field=FIELD, master_camera=record, carrier_archive_sha256=ARCHIVE_SHA
    )
    binding = _binding(fingerprint)
    binding["master_camera"] = record
    # A solve carrying another object's compensation is a compile error.
    row = {"pair": 26, "compensation_object": binding, "solve": {
        "compensation_object_fingerprint_sha256": "f" * 64}}
    with pytest.raises(bs4y.BS4YError, match="stale for the compile object"):
        bs4y.assert_compensation_matches_compile_object(
            row, semantic_field=FIELD, carrier_archive_sha256=ARCHIVE_SHA
        )
    # The matching fresh solve passes.
    row["solve"]["compensation_object_fingerprint_sha256"] = fingerprint
    verdict = bs4y.assert_compensation_matches_compile_object(
        row, semantic_field=FIELD, carrier_archive_sha256=ARCHIVE_SHA
    )
    assert verdict["mode"] == "EXACT_OBJECT_BOUND_FRESH_SOLVE"
    assert verdict["passed"] is True


def test_non_finite_diagnostics_stay_json_serialisable() -> None:
    assert bs4y.finite_or_none(float("inf")) is None
    assert bs4y.finite_or_none(float("nan")) is None
    assert bs4y.finite_or_none(2.5) == 2.5
    json.dumps({"condition": bs4y.finite_or_none(float("inf"))}, allow_nan=False)


def test_storage_guard_fails_closed_on_an_impossible_request() -> None:
    with pytest.raises(bs4y.BS4YError, match="storage waterfall refuses"):
        bs4y.require_free_bytes(1 << 62, "impossible")


def test_candidate_price_matches_the_sealed_selected_object_gate() -> None:
    from experiments import ddm_bs4x_selected_storage_preflight as gate

    assert bs4y.BYTES_PER_CANDIDATE == gate.BYTES_PER_CANDIDATE == 9_156_024
