# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

import pytest

from tac.derived_upstream_refresh import (
    BASE_SHA_FORMULA_INVARIANT,
    BASE_SHA_PRESENT,
    DISPOSITION_CURRENT,
    DISPOSITION_EXACT_INVARIANT,
    DISPOSITION_QUEUED_HEAVY_REFRESH,
    DISPOSITION_REFRESHED_SCORER_FREE,
    ROUTE_ALREADY_CURRENT,
    ROUTE_EXACT_INVARIANT,
    ROUTE_FULL_RECOMPUTE,
    ROUTE_SCORER_FREE_DERIVATION,
    RefreshRegistryError,
    RefreshRegistryRow,
    load_refresh_registry_jsonl,
    registry_denominators,
    require_fresh_for_consumption,
    write_refresh_registry_jsonl,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _row(**updates: object) -> RefreshRegistryRow:
    params = {
        "quantity_id": "demo_quantity",
        "description": "demo",
        "base_identity_kind": "archive_zip_sha256",
        "computed_at_base_sha256": SHA_A,
        "base_sha_status": BASE_SHA_PRESENT,
        "base_age": "CURRENT",
        "current_base_sha256": SHA_A,
        "consumers": ("demo.consumer",),
        "validity_radius_status": "KNOWN",
        "validity_radius_derive_route": "exact same-base check",
        "refresh_route": ROUTE_ALREADY_CURRENT,
        "trigger": "base sha mismatch",
        "owner": "uf1",
        "disposition": DISPOSITION_CURRENT,
        "evidence_paths": ("receipts/demo.json",),
        "score_claim": False,
        "promotion_eligible": False,
    }
    params.update(updates)
    return RefreshRegistryRow(**params)  # type: ignore[arg-type]


def test_current_row_passes_declared_consumer() -> None:
    require_fresh_for_consumption(
        _row(), current_base_sha256=SHA_A, consumer="demo.consumer"
    )


def test_stale_row_refuses_at_consumption() -> None:
    with pytest.raises(RefreshRegistryError, match=r"stale for demo\.consumer"):
        require_fresh_for_consumption(
            _row(current_base_sha256=SHA_B), current_base_sha256=SHA_B, consumer="demo.consumer"
        )


def test_queued_row_refuses_even_when_sha_matches() -> None:
    queued = _row(
        refresh_route=ROUTE_FULL_RECOMPUTE,
        disposition=DISPOSITION_QUEUED_HEAVY_REFRESH,
        trigger="scorer slot unavailable",
    )
    with pytest.raises(RefreshRegistryError, match="QUEUED_HEAVY_REFRESH"):
        require_fresh_for_consumption(
            queued, current_base_sha256=SHA_A, consumer="demo.consumer"
        )


def test_undeclared_consumer_refuses() -> None:
    with pytest.raises(RefreshRegistryError, match="not declared"):
        require_fresh_for_consumption(_row(), current_base_sha256=SHA_A, consumer="other")


def test_formula_invariant_row_passes_without_base_equality() -> None:
    row = _row(
        computed_at_base_sha256=SHA_A,
        base_sha_status=BASE_SHA_FORMULA_INVARIANT,
        current_base_sha256=SHA_B,
        refresh_route=ROUTE_EXACT_INVARIANT,
        disposition=DISPOSITION_EXACT_INVARIANT,
        validity_radius_status="NOT_APPLICABLE",
        validity_radius_derive_route="linear formula",
    )
    require_fresh_for_consumption(row, current_base_sha256=SHA_B, consumer="demo.consumer")


def test_schema_rejects_missing_trigger_or_fake_unknown_sha() -> None:
    with pytest.raises(RefreshRegistryError, match="trigger"):
        _row(trigger="")
    with pytest.raises(RefreshRegistryError, match="must not carry a fake sha"):
        _row(base_sha_status="UNKNOWN_IN_UF1_SCOPE", computed_at_base_sha256=SHA_A)


def test_registry_round_trip_and_denominators(tmp_path) -> None:
    rows = (
        _row(),
        _row(
            quantity_id="m66",
            refresh_route=ROUTE_SCORER_FREE_DERIVATION,
            disposition=DISPOSITION_REFRESHED_SCORER_FREE,
        ),
        _row(
            quantity_id="w",
            base_sha_status=BASE_SHA_FORMULA_INVARIANT,
            refresh_route=ROUTE_EXACT_INVARIANT,
            disposition=DISPOSITION_EXACT_INVARIANT,
            validity_radius_status="NOT_APPLICABLE",
        ),
    )
    path = tmp_path / "registry.jsonl"
    write_refresh_registry_jsonl(rows, path)

    loaded = load_refresh_registry_jsonl(path)
    assert [row.quantity_id for row in loaded] == ["demo_quantity", "m66", "w"]
    assert registry_denominators(loaded) == {
        "quantities_found": 3,
        "with_consumers": 3,
        "with_triggers": 3,
        "with_known_validity_radius": 2,
        "scorer_free_refreshed": 1,
        "exact_invariants": 1,
        "heavy_refreshes_queued": 0,
        "fiber_input_blockers_queued": 0,
        "current_rows": 1,
    }
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["schema"]
