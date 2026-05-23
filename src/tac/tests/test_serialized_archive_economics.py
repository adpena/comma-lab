# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.optimization.serialized_archive_economics import (
    CANDIDATE_ARCHIVE_LARGER_BLOCKER,
    MISSING_ARCHIVE_BYTES_BLOCKER,
    MODELED_SAVINGS_WITHOUT_REALIZED_BLOCKER,
    SERIALIZED_ARCHIVE_DELTA_SCHEMA,
    SERIALIZED_SAVINGS_NOT_POSITIVE_BLOCKER,
    archive_record_bytes,
    build_serialized_archive_delta_contract,
    serialized_archive_delta_blockers,
)

FALSE_AUTHORITY_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "promotable",
    "ready_for_exact_eval_dispatch",
    "dispatch_attempted",
    "gpu_launched",
)


def test_archive_record_bytes_accepts_common_shapes_and_rejects_nonpositive() -> None:
    assert archive_record_bytes(7) == 7
    assert archive_record_bytes("8") == 8
    assert archive_record_bytes({"bytes": "9"}) == 9
    assert archive_record_bytes({"archive_bytes": 10}) == 10
    assert archive_record_bytes({"size_bytes": 11.0}) == 11
    assert archive_record_bytes({"archive_size_bytes": 12}) == 12

    assert archive_record_bytes(True) is None
    assert archive_record_bytes(0) is None
    assert archive_record_bytes(-1) is None
    assert archive_record_bytes({"bytes": 1.5}) is None
    assert archive_record_bytes({"bytes": "0"}) is None
    assert archive_record_bytes({"bytes": "not-bytes"}) is None


def test_serialized_archive_delta_contract_separates_modeled_from_realized() -> None:
    contract = build_serialized_archive_delta_contract(
        modeled_saved_bytes=120,
        require_realized_saving=True,
    )

    assert contract["schema"] == SERIALIZED_ARCHIVE_DELTA_SCHEMA
    assert contract["status"] == "missing_archive_bytes"
    assert contract["savings_realized"] is False
    assert MISSING_ARCHIVE_BYTES_BLOCKER in contract["blockers"]
    assert MODELED_SAVINGS_WITHOUT_REALIZED_BLOCKER in contract["blockers"]
    assert SERIALIZED_SAVINGS_NOT_POSITIVE_BLOCKER in contract["blockers"]
    for field in FALSE_AUTHORITY_FIELDS:
        assert contract[field] is False


def test_serialized_archive_delta_contract_accepts_real_saving_only() -> None:
    contract = build_serialized_archive_delta_contract(
        source_archive={"bytes": 1000, "sha256": "a" * 64},
        candidate_archive={"bytes": 920, "sha256": "b" * 64},
        modeled_saved_bytes=120,
        require_realized_saving=True,
    )

    assert contract["status"] == "realized_saving"
    assert contract["archive_delta_bytes"] == -80
    assert contract["realized_saved_bytes"] == 80
    assert contract["savings_realized"] is True
    assert serialized_archive_delta_blockers(contract) == []

    zero_delta = build_serialized_archive_delta_contract(
        source_archive_bytes=100,
        candidate_archive_bytes=100,
        require_realized_saving=True,
    )
    assert zero_delta["status"] == "zero_delta"
    assert SERIALIZED_SAVINGS_NOT_POSITIVE_BLOCKER in zero_delta["blockers"]


def test_serialized_archive_delta_contract_blocks_larger_candidate() -> None:
    contract = build_serialized_archive_delta_contract(
        source_archive_bytes=1000,
        candidate_archive_bytes=1001,
        modeled_saved_bytes=120,
        require_realized_saving=True,
    )

    assert contract["status"] == "realized_cost"
    assert contract["archive_delta_bytes"] == 1
    assert CANDIDATE_ARCHIVE_LARGER_BLOCKER in contract["blockers"]
    assert MODELED_SAVINGS_WITHOUT_REALIZED_BLOCKER in contract["blockers"]


def test_serialized_archive_delta_contract_allows_rate_only_control() -> None:
    contract = build_serialized_archive_delta_contract(
        source_archive_bytes=100,
        candidate_archive_bytes=101,
        rate_only_control=True,
    )

    assert contract["status"] == "rate_only_control"
    assert contract["blockers"] == []
    assert contract["savings_realized"] is False
