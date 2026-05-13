from __future__ import annotations

from tac.deploy.modal.harvest_summary import modal_training_summary_entry


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
    }
