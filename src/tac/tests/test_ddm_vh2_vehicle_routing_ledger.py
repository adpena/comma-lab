# SPDX-License-Identifier: MIT
"""Typed vehicle-harvest routing tests for ddm_vh2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.probe_outcomes_ledger import (
    EVIDENCE_CLASS_MEASURED,
    ROUTING_STATUS_DEAD_ON_THIS_BASE,
    ROUTING_STATUS_ROUTED_QUEUED,
    ProbeOutcomesLedgerCorruptError,
    coverage,
    partition_vehicle_corpus,
    register_vehicle_routing_findings,
)


def _queued_finding(**overrides: object) -> dict[str, object]:
    finding: dict[str, object] = {
        "route_id": "ddm_vh2_test_route_v1",
        "lineage": "v",
        "source_doc": ".omx/research/ddm_v19_probe.md",
        "corpus_artifact": ".omx/research/ddm_v19_probe.md",
        "sha": "a" * 64,
        "finding": "The receiver-closed candidate needs a current-base replay.",
        "evidence_class": EVIDENCE_CLASS_MEASURED,
        "vehicle_scope": "PR130 CPR1 archive bytes only",
        "status": ROUTING_STATUS_ROUTED_QUEUED,
        "owner": "ddm_rc2",
        "consumer": ".omx/state/probe_outcomes.jsonl",
        "fire_order": "After rc2 closes the #996 coder-axis scope review.",
        "source_harvest": "ddm_vh2",
    }
    finding.update(overrides)
    return finding


def test_batch_append_is_idempotent_and_preserves_existing_text(tmp_path: Path) -> None:
    ledger = tmp_path / "probe_outcomes.jsonl"
    lock = tmp_path / "probe_outcomes.jsonl.lock"
    historical = '{"z": 1, "event_type": "historical"}\n'
    ledger.write_text(historical, encoding="utf-8")

    first = register_vehicle_routing_findings([_queued_finding()], path=ledger, lock_path=lock)
    assert len(first.appended) == 1
    assert first.already_present == ()
    assert ledger.read_text(encoding="utf-8").startswith(historical)

    second = register_vehicle_routing_findings([_queued_finding()], path=ledger, lock_path=lock)
    assert second.appended == ()
    assert len(second.already_present) == 1
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.parametrize("key", ["owner", "consumer"])
def test_unowned_is_forbidden(tmp_path: Path, key: str) -> None:
    ledger = tmp_path / "probe_outcomes.jsonl"
    with pytest.raises(ValueError, match="UNOWNED is forbidden"):
        register_vehicle_routing_findings(
            [_queued_finding(**{key: "UNOWNED"})],
            path=ledger,
            lock_path=tmp_path / "lock",
        )
    assert not ledger.exists()


def test_dead_status_requires_named_blocker_and_fire_condition(tmp_path: Path) -> None:
    finding = _queued_finding(
        status=ROUTING_STATUS_DEAD_ON_THIS_BASE,
        fire_order=None,
        blocker=None,
        fire_condition=None,
    )
    with pytest.raises(ValueError, match="requires a named blocker"):
        register_vehicle_routing_findings(
            [finding],
            path=tmp_path / "probe_outcomes.jsonl",
            lock_path=tmp_path / "lock",
        )


def test_conflicting_route_id_requires_append_only_supersession(tmp_path: Path) -> None:
    ledger = tmp_path / "probe_outcomes.jsonl"
    lock = tmp_path / "lock"
    register_vehicle_routing_findings([_queued_finding()], path=ledger, lock_path=lock)
    before = ledger.read_bytes()
    with pytest.raises(ValueError, match="superseding route"):
        register_vehicle_routing_findings([_queued_finding(finding="Changed finding.")], path=ledger, lock_path=lock)
    assert ledger.read_bytes() == before


def test_duplicate_existing_route_ids_refuse_append(tmp_path: Path) -> None:
    ledger = tmp_path / "probe_outcomes.jsonl"
    lock = tmp_path / "lock"
    register_vehicle_routing_findings([_queued_finding()], path=ledger, lock_path=lock)
    duplicated = ledger.read_bytes() * 2
    ledger.write_bytes(duplicated)
    with pytest.raises(ValueError, match="occurs more than once"):
        register_vehicle_routing_findings(
            [_queued_finding(route_id="ddm_vh2_another_route_v1")],
            path=ledger,
            lock_path=lock,
        )
    assert ledger.read_bytes() == duplicated


def test_corrupt_ledger_refuses_without_write(tmp_path: Path) -> None:
    ledger = tmp_path / "probe_outcomes.jsonl"
    ledger.write_text("not-json\n", encoding="utf-8")
    before = ledger.read_bytes()
    with pytest.raises(ProbeOutcomesLedgerCorruptError):
        register_vehicle_routing_findings([_queued_finding()], path=ledger, lock_path=tmp_path / "lock")
    assert ledger.read_bytes() == before


def test_partition_and_coverage_use_explicit_top_level_denominator(
    tmp_path: Path,
) -> None:
    research = tmp_path / "research"
    research.mkdir()
    (research / "ddm_v19_probe.md").write_text("v", encoding="utf-8")
    nested = research / "ddm_ms2_bundle"
    nested.mkdir()
    (nested / "payload.bin").write_bytes(b"payload")
    (research / "ddm_entropy_probe.md").write_text("entropy", encoding="utf-8")
    (research / "not_ddm.md").write_text("ignored", encoding="utf-8")

    artifacts = partition_vehicle_corpus(research_root=research)
    assert [(artifact.lineage, artifact.kind) for artifact in artifacts] == [
        ("entropy", "file"),
        ("ms", "directory"),
        ("v", "file"),
    ]

    ledger = tmp_path / "probe_outcomes.jsonl"
    register_vehicle_routing_findings(
        [
            _queued_finding(),
            _queued_finding(
                route_id="ddm_vh2_test_entropy_v1",
                lineage="entropy",
                source_doc=".omx/research/ddm_entropy_probe.md",
                corpus_artifact=".omx/research/ddm_entropy_probe.md",
            ),
        ],
        path=ledger,
        lock_path=tmp_path / "lock",
    )
    report = coverage(research_root=research, path=ledger)
    assert report["totals"] == {
        "artifacts": 3,
        "harvested": 2,
        "routed": 2,
        "un_harvested": 1,
    }
    assert report["lineages"]["ms"]["un_harvested"] == 1

    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert all(row["score_claim"] is False for row in rows)
