from __future__ import annotations

from pathlib import Path

from tac.deploy.modal.harvest_summary import (
    modal_training_summary_entry,
    normalise_modal_training_result_summary,
    partial_modal_training_result_summary,
)


def test_already_harvested_summary_preserves_result_signal() -> None:
    row = modal_training_summary_entry(
        label="substrate_sane_hnerv_modal_a100_dispatch",
        status="already_harvested",
        call_id="fc-test",
        harvested={
            "rc": 1,
            "elapsed_seconds": 72.0,
            "timed_out": False,
            "n_artifacts": 51,
            "crash_kind": "RC_1",
            "cost_band_anchor": {"ignored": "nested duplicate"},
        },
        cost_anchor={"appended": True, "score_claim": False},
        terminal_claim={"appended": True, "status": "failed_modal_training_rc_1"},
        terminal_evidence={"appended": True, "already_covered": False},
    )

    assert row == {
        "label": "substrate_sane_hnerv_modal_a100_dispatch",
        "status": "already_harvested",
        "call_id": "fc-test",
        "rc": 1,
        "elapsed_seconds": 72.0,
        "timed_out": False,
        "n_artifacts": 51,
        "crash_kind": "RC_1",
        "cost_band_anchor": {"appended": True, "score_claim": False},
        "terminal_claim": {
            "appended": True,
            "status": "failed_modal_training_rc_1",
        },
        "terminal_evidence": {"appended": True, "already_covered": False},
    }


def test_normalise_legacy_root_harvest_summary_preserves_failure_signal(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "harvested_artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "remote.log").write_text("fatal\n", encoding="utf-8")
    source_summary = tmp_path / "harvest_summary.json"
    loaded = {
        "returncode": 21,
        "score_claim": False,
        "promotion_eligible": False,
    }

    row = normalise_modal_training_result_summary(
        loaded,
        artifacts_dir=artifacts_dir,
        source_summary=source_summary,
    )

    assert row["rc"] == 21
    assert row["crash_kind"] == "RC_21"
    assert row["n_artifacts"] == 1
    assert row["source_summary"] == str(source_summary)
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False


def test_partial_harvest_summary_is_non_score_claim(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "harvested_artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "train.json").write_text("{}", encoding="utf-8")

    row = partial_modal_training_result_summary(artifacts_dir=artifacts_dir)

    assert row["crash_kind"] == "HARVESTED_PARTIAL"
    assert row["n_artifacts"] == 1
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["rank_or_kill_eligible"] is False


def test_partial_harvest_summary_handles_missing_artifacts_dir(tmp_path: Path) -> None:
    row = partial_modal_training_result_summary(
        artifacts_dir=tmp_path / "missing_harvested_artifacts",
    )

    assert row["crash_kind"] == "HARVESTED_PARTIAL"
    assert row["n_artifacts"] == 0
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["rank_or_kill_eligible"] is False
