from __future__ import annotations

from pathlib import Path

import pytest

from experiments import ddm_js6_event_proposal_acceptance as js6


def _row(
    ordinal: int,
    *,
    family: str = "boundary",
    pose_pass: bool = False,
    robust: int = 0,
    coded_bytes: int = 21,
) -> dict[str, object]:
    bpf = js6.bytes_per_robust_flip(coded_bytes, robust)
    return {
        "ordinal": ordinal,
        "proposal_id": f"p{ordinal:04d}",
        "family": family,
        "pose_gate_pass": pose_pass,
        "bare_admission": pose_pass and robust < 0,
        "projected_n600_robust_delta_flips": robust,
        "bytes_per_projected_robust_flip": bpf,
    }


def test_projected_pose_delta_uses_sealed_n600_denominator() -> None:
    assert js6.projected_pose_delta(0.03, 0.01, 18) == pytest.approx(0.0006)
    with pytest.raises(ValueError):
        js6.projected_pose_delta(float("nan"), 0.01, 18)


def test_bytes_per_robust_flip_has_typed_zero_denominator() -> None:
    assert js6.bytes_per_robust_flip(21, -42) == 0.5
    assert js6.bytes_per_robust_flip(21, 0) is None
    assert js6.bytes_per_robust_flip(21, 42) is None
    with pytest.raises(ValueError):
        js6.bytes_per_robust_flip(-1, -1)


def test_first_useful_bare_admission_stops_f1_and_reports_economics() -> None:
    rows = [_row(0, robust=19), _row(1, pose_pass=True, robust=-21)]
    summary = js6.summarize(rows)
    assert summary["stop_reason"] == "FIRST_USEFUL_NONZERO_BARE_ADMISSION"
    assert summary["pose_accepted"] == 1
    assert summary["bare_admissions"] == 1
    assert summary["falsifiers"]["F1"] == {
        "eligible": False,
        "fired": False,
        "scope": "FAMILY at the EC1 representation-level event-coordinate endpoint only",
        "criterion": "all 200 measured, no useful bare admission, and pose acceptance below 5%",
    }
    assert summary["economics"]["selected_bare_bytes_per_projected_robust_flip"] == 1.0


def test_f1_fires_only_after_all_200_and_below_five_percent() -> None:
    rows = [_row(index, pose_pass=index < 9) for index in range(200)]
    summary = js6.summarize(rows)
    assert summary["stop_reason"] == "ALL_200_MEASURED"
    assert summary["pose_acceptance_rate"] == pytest.approx(0.045)
    assert summary["falsifiers"]["F1"]["eligible"] is True
    assert summary["falsifiers"]["F1"]["fired"] is True

    exactly_five_percent = [_row(index, pose_pass=index < 10) for index in range(200)]
    assert js6.summarize(exactly_five_percent)["falsifiers"]["F1"]["fired"] is False


def test_per_family_denominators_are_explicit() -> None:
    rows = [
        _row(0, family="boundary", pose_pass=True, robust=0),
        _row(1, family="boundary"),
        _row(2, family="lane"),
    ]
    summary = js6.summarize(rows)
    by_family = {row["family"]: row for row in summary["per_family"]}
    assert by_family["boundary"]["measured"] == 2
    assert by_family["boundary"]["pose_acceptance_rate"] == 0.5
    assert by_family["lane"]["measured"] == 1
    assert by_family["island"]["measured"] == 0
    assert by_family["island"]["pose_acceptance_rate"] is None


def test_queue_rows_keep_typed_fire_orders(tmp_path: Path) -> None:
    partial = js6.summarize([_row(0)])
    assert js6.queue_rows(partial, tmp_path)[0]["disposition"] == "QUEUED-WITH-A-FIRE-ORDER"

    fired = js6.summarize([_row(index) for index in range(200)])
    queue = js6.queue_rows(fired, tmp_path)
    assert queue[0]["owner"] == "MAIN SE1 executor/harvester"
    assert queue[0]["consumer_store"] == "/Volumes/APDataStore/pact/ddm_se1_20260812"


def test_parser_has_no_device_or_payload_regeneration_flag() -> None:
    parser = js6.parser()
    destinations = {action.dest for action in parser._actions}
    assert "device" not in destinations
    assert "regenerate" not in destinations
    args = parser.parse_args([])
    assert args.output == js6.DEFAULT_OUTPUT
    assert args.proposal_store == js6.DEFAULT_PROPOSAL_ROOT
