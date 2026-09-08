"""Report-contract guards for the ddm_cl3 RC1 capacity ladder closer."""

from __future__ import annotations

import json
from types import SimpleNamespace

from experiments import ddm_cl3_rc1_rung_price as cl3


def _result(rung: str, rate_lambda: float, seed: int, hpac: int, stream: int) -> dict[str, object]:
    joint = hpac + stream
    return {
        "rung": rung,
        "rate_lambda": rate_lambda,
        "seed": seed,
        "hpac_container": {"bytes": hpac},
        "stream": {"bytes": stream},
        "joint_bytes": joint,
        "joint_delta_vs_live_125762": joint - 125_762,
        "candidate_archive": {"bytes": 174_786 + joint - 125_762},
        "two_encodes_identical": True,
        "decoded_identity": True,
        "only_model_and_stream_moved": True,
    }


def test_row_admissibility_requires_all_three_independent_legs() -> None:
    row = _result("lambda_1p0_s18", 1.0, 20260718, 12_416, 113_483)
    assert cl3.row_admissibility(row)["admissible"] is True
    row["only_model_and_stream_moved"] = False
    admission = cl3.row_admissibility(row)
    assert admission["admissible"] is False
    assert admission["legs"]["section_census_only_model_and_stream_moved"] is False


def test_report_carries_every_rung_and_refuses_false_current_pointer_comparison(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(cl3, "STORE", tmp_path)
    for row in (
        _result("lambda_2p0", 2.0, 20260716, 11_886, 114_100),
        _result("lambda_1p0_s17", 1.0, 20260717, 12_375, 113_416),
        _result("lambda_1p0_s18", 1.0, 20260718, 12_416, 113_483),
    ):
        root = tmp_path / "rungs" / str(row["rung"])
        root.mkdir(parents=True)
        (root / "RC1_RUNG_RESULT.json").write_text(json.dumps(row), encoding="utf-8")

    monkeypatch.setattr(
        cl3,
        "source_verified_ladder_control",
        lambda: {
            "rate_lambda": 1.0,
            "seed": 20260716,
            "hpac_container_bytes": 12_343,
            "stream_bytes": 113_419,
            "joint_bytes": 125_762,
            "joint_delta_vs_ladder_control": 0,
            "candidate_archive_bytes": 174_786,
            "status": "MEASURED_CONTROL",
            "legs": {
                "twin_encode_byte_identical": True,
                "receiver_copy_decode_identity": True,
                "section_census_only_model_and_stream_moved": True,
            },
            "admissible": True,
        },
    )
    monkeypatch.setattr(
        cl3,
        "source_verified_current_pointer",
        lambda: {
            "score": 0.13900437796841966,
            "archive_bytes": 181_645,
            "archive_sha256": "06c44dc",
            "source_archive": "candidate_pass3/candidate_runtime/archive.zip",
            "hpac_bytes": 12_343,
            "stream_bytes": 120_225,
            "joint_bytes": 132_568,
            "verified_at_source": True,
        },
    )

    report = cl3.stage_report(SimpleNamespace(out=tmp_path / "report.json"))
    assert set(report["rows"]) == {
        "live_pointer_control_lambda_1p0",
        "lambda_2p0",
        "lambda_4p0",
        "lambda_1p0_s17",
        "lambda_1p0_s18",
    }
    assert report["rows"]["lambda_1p0_s18"]["admissible"] is True
    assert report["rows"]["lambda_4p0"]["status"] == "NOT_RUN_BECAUSE_FALSIFIED"
    assert report["best_rung"] == "live_pointer_control_lambda_1p0"
    assert report["best_beats_ladder_control"] is False
    assert report["best_beats_live_pointer"] is False
    assert report["current_pointer"]["archive_bytes"] == 181_645
