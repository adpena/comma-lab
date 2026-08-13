from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from experiments import ddm_vd1_batch_event_validator_worker as worker
from experiments import ddm_vd1_modal_batch_event_validator as vd1
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def test_k_arithmetic_fits_full_200_with_conservative_reserve() -> None:
    row = vd1.k_arithmetic()
    assert row["charged_seconds_per_event"] == pytest.approx(1.21175)
    assert row["k_max_with_reserve"] == 913
    assert row["projected_target_seconds_with_reserve"] == pytest.approx(935.916)
    assert row["full_200_fits"] is True
    assert row["epistemic_status"].startswith("DERIVED_FROM_MEASURED")


def test_runtime_bundle_is_deterministic_and_excludes_generated_residue(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime_source"
    (runtime / "runtime").mkdir(parents=True)
    (runtime / "cpr1").mkdir()
    (runtime / "runtime/__pycache__").mkdir()
    (runtime / "inflate.sh").write_text("#!/bin/sh\n")
    (runtime / "runtime/f26_inflate.py").write_text("pass\n")
    (runtime / "runtime/entropy").mkdir()
    (runtime / "runtime/entropy/rc64_backend.c").write_text("/* retained */\n")
    (runtime / "cpr1/inflate.py").write_text("pass\n")
    (runtime / "runtime/__pycache__/bad.pyc").write_bytes(b"bad")
    (runtime / "._bad").write_bytes(b"bad")
    (runtime / "archive.zip").write_bytes(b"separate input")
    first, first_manifest = vd1.build_runtime_bundle(runtime)
    second, second_manifest = vd1.build_runtime_bundle(runtime)
    assert first == second
    assert first_manifest["bundle_sha256"] == second_manifest["bundle_sha256"]
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        names = set(archive.namelist())
    assert "runtime/f26_inflate.py" in names
    assert "cpr1/inflate.py" in names
    assert "archive.zip" not in names
    assert not any("__pycache__" in name or name.startswith("._") for name in names)


def test_full_event_bundle_keeps_all_200_payloads_and_uses_census_mode() -> None:
    payload, manifest = vd1.build_event_bundle(
        vd1.DEFAULT_EVENT_STORE,
        vd1.DEFAULT_JO1_ANALYSIS,
        k=200,
    )
    assert manifest["selected_events"] == 200
    assert manifest["selection_mode"] == "full_200_census"
    assert len({row["proposal_id"] for row in manifest["events"]}) == 200
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert len([name for name in archive.namelist() if name.endswith(".ec1p")]) == 200


def test_top_k_fallback_uses_jo1_plus3b_rate_order() -> None:
    _payload, manifest = vd1.build_event_bundle(
        vd1.DEFAULT_EVENT_STORE,
        vd1.DEFAULT_JO1_ANALYSIS,
        k=6,
    )
    analysis = json.loads(vd1.DEFAULT_JO1_ANALYSIS.read_text())
    expected = sorted(
        analysis["features"],
        key=lambda row: (
            float("inf")
            if row["bytes_per_projected_robust_flip"] is None
            else float(row["bytes_per_projected_robust_flip"]),
            int(row["projected_robust_delta_flips"]),
            int(row["ordinal"]),
        ),
    )[:6]
    assert manifest["selection_mode"] == "jo1_plus3B_rate_order"
    assert [row["proposal_id"] for row in manifest["events"]] == [
        row["proposal_id"] for row in expected
    ]


def test_pose_budget_crosswalk_is_pair_and_global_consistent() -> None:
    assert pytest.approx(1.3e-7 / 44) == worker.POSE_PER_EVENT_GLOBAL_BUDGET
    assert pytest.approx((1.3e-7 / 44) * 600) == worker.POSE_PER_EVENT_PAIR_BUDGET


def test_selection_projection_is_explicitly_not_composition_authority() -> None:
    rows = [
        {
            "proposal_id": "gain",
            "ordinal": 0,
            "net_flip_gain_base_minus_candidate": 500,
            "delta_d_pose_global_n600": -1e-9,
            "downstream_selection_eligible": True,
        },
        {
            "proposal_id": "pose_spend",
            "ordinal": 1,
            "net_flip_gain_base_minus_candidate": 1000,
            "delta_d_pose_global_n600": 2e-7,
            "downstream_selection_eligible": False,
        },
    ]
    result = worker.selection_projection(rows)
    assert result["selection_only_not_composed"] is True
    assert result["singleton_interactions_unmeasured"] is True
    assert result["selected_ids_under_additive_pose_budget"] == ["gain"]


def test_dispatch_source_pins_locked_env_single_flight_and_periodic_volume_commit() -> None:
    source = Path(vd1.__file__).read_text()
    assert "UPSTREAM_LOCKED_VENV" in source
    assert "assert_modal_single_flight" in source
    assert "run_validator.spawn" in source
    assert "retained_volume.commit()" in source
    assert "--resume-from" in source
    assert "score_claim\": False" in source


def test_worker_retains_n600_batches_and_per_event_payloads() -> None:
    source = Path(worker.__file__).read_text()
    assert "gt/n600_batches" in source
    assert "candidate_tokens.uint8.npy" in source
    assert "candidate_master.uint8.npy" in source
    assert "candidate_pair.uint8.npy" in source
    assert "delta_d_pose_pair" in source
    assert "delta_flips_candidate_minus_base" in source
    assert "stage_40_events_" in source


def test_vd1_files_pass_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=Path.cwd(),
        strict=False,
        roots=(
            "experiments/ddm_vd1_modal_batch_event_validator.py",
            "experiments/ddm_vd1_batch_event_validator_worker.py",
            "experiments/tests/test_ddm_vd1_modal_batch_event_validator.py",
        ),
    )
    assert findings == []
