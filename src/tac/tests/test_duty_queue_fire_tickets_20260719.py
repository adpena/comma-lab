from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from tac.witness_dsl.duty_queue_fire_tickets_20260719 import materialize_duty_queue_fire_tickets

EXPECTED_ORDER = [
    "01_dseg_aware_taper",
    "02_horizon_weighted_margin",
    "03_step_native_activation",
    "04_curvelet_matched_bytes_p0_497",
]

EXPECTED_CURVELET_HASHES = {
    "control": {
        "dsl_compile_hash": "be96e7498b2f63d208187231d1f36c9b31a96dad1fd009b48dde9f147e35826a",
        "typed_config_hash": "36ee86bb385cffbdbd34d763676227d93c566ebc876e2017f2c1d9c63e630e6a",
        "resolved_argv_hash": "421a2855f20de332a184b85ac844124613743dae1d218852579da5fa8ea055d5",
    },
    "treatment": {
        "dsl_compile_hash": "7ed4982087f723495ac8f8f2e41f6ac655655dab87749dd18bd294a24ce709a2",
        "typed_config_hash": "3de19c82df8ceb9b052b17d9ce063696d2fcc9d57114930c82d8c5d4ac978b26",
        "resolved_argv_hash": "4697f58c55a51ace52473ed02ac2d9230212ef5b814ae0cf27d41e6393d6e450",
    },
}

EXPECTED_MAIN_SOURCE_AUTHORITY_PATHS = {
    ".omx/research/p0_resume_warmup_geometry_build_20260717.md",
    "experiments/train_levelset_witness_realized_through_R_mlx.py",
    "reports/latest.md",
    "src/tac/canonical_equations/adam_v_variance_warmup_20260717.py",
    "src/tac/canonical_equations/cgauge_parametrization_optima_20260711.py",
    "src/tac/witness_control/ncde_trajectory.py",
    "src/tac/witness_dsl/curriculum_dsl.py",
    "src/tac/witness_dsl/lever_registry.py",
    "src/tac/witness_dsl/spec_v9_cgauge.py",
    "tools/fire_curvelet_matched_bytes_ab_p0_497.py",
    "tools/launch_witness_run.py",
    "tools/operator_authorize.py",
    "tools/safe_run.py",
    "tools/spawn_durable_daemon.py",
    "tools/witness_launch_readiness_gate.py",
}


def _source_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _main_repo_with_checkpoint(tmp_path: Path) -> Path:
    checkpoint = (
        tmp_path
        / "main"
        / "experiments"
        / "results"
        / "banks"
        / "v9c2_defensive_bank_20260718"
        / "levelset_witness_ema_BEST.npz"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"custody-only-test-checkpoint")
    return tmp_path / "main"


def test_materializer_writes_four_ordered_fail_closed_packages(tmp_path: Path) -> None:
    out_dir = tmp_path / "tickets"
    main_repo = _main_repo_with_checkpoint(tmp_path)

    summary = materialize_duty_queue_fire_tickets(
        out_dir,
        main_repo=main_repo,
        main_source_repo=_source_root(),
        repo_root=_source_root(),
        created_utc="2026-07-19T00:00:00Z",
    )

    assert summary["ticket_order"] == EXPECTED_ORDER
    assert summary["all_status"] == "BLOCKED"
    assert summary["wrappers_executed"] == 4
    assert summary["trainer_governed_launches_executed"] == 0
    source_custody_path = out_dir / "main_source_custody.json"
    source_custody = json.loads(source_custody_path.read_text())
    assert source_custody["status"] == "VERIFIED_SNAPSHOT"
    assert source_custody["created_utc_snapshot_label"] == "2026-07-19T00:00:00Z"
    assert source_custody["main_descends_from_source_head"] is True
    assert source_custody["mismatches"] == []
    assert all(row["match"] for row in source_custody["authority_input_sha256_comparisons"])
    assert all(
        row["source_sha256"]
        == row["source_head_blob"]["sha256"]
        == row["main_sha256"]
        == row["main_head_blob"]["sha256"]
        for row in source_custody["authority_input_sha256_comparisons"]
    )
    assert all(
        row["source_head_blob"]["resolved"] and row["main_head_blob"]["resolved"]
        for row in source_custody["authority_input_sha256_comparisons"]
    )
    assert {
        row["path"] for row in source_custody["authority_input_sha256_comparisons"]
    } == EXPECTED_MAIN_SOURCE_AUTHORITY_PATHS
    assert summary["main_source_custody"]["sha256"] == hashlib.sha256(source_custody_path.read_bytes()).hexdigest()
    root_manifest = json.loads((out_dir / "summary_manifest.json").read_text())
    custody_manifest_row = next(row for row in root_manifest["files"] if row["path"] == "main_source_custody.json")
    assert custody_manifest_row["sha256"] == summary["main_source_custody"]["sha256"]
    for ticket_id in EXPECTED_ORDER:
        ticket = out_dir / ticket_id
        assert ticket.is_dir()
        assert (ticket / "launch.sh").is_file()
        assert (ticket / "compiled_config.json").is_file()
        assert (ticket / "provenance.json").is_file()
        assert (ticket / "dry_start_receipt.json").is_file()
        assert (ticket / "verdict_card.md").is_file()
        assert (ticket / "confound_self_audit.md").is_file()
        assert (ticket / "confound_self_audit.json").is_file()
        assert (ticket / "artifact_manifest.json").is_file()
        config = json.loads((ticket / "compiled_config.json").read_text())
        receipt = json.loads((ticket / "dry_start_receipt.json").read_text())
        launch = (ticket / "launch.sh").read_text()
        assert config["status"] == "BLOCKED"
        assert config["compile_state"] == "BLOCKED_BEFORE_FULL_COMPILE"
        assert config["full_dsl_compile_hash"] is None
        assert config["argv_authority"] == "NONE"
        assert receipt["verdict"] == "GREEN_STATIC_REFUSAL"
        assert receipt["scope"] == "materializer_integrity"
        assert receipt["launch_ready"] is False
        assert "train_levelset_witness" not in launch
        assert "subprocess" not in launch
        assert "exit 6" in launch
        assert receipt["returncode"] == 6
        assert receipt["stdout"] != "REFUSE: non-authorizing ticket package"
        provenance = json.loads((ticket / "provenance.json").read_text())
        assert provenance["main_source_custody"] == {
            "schema": "main_source_custody.v1",
            "status": "VERIFIED_SNAPSHOT",
            "path": "../main_source_custody.json",
            "sha256": summary["main_source_custody"]["sha256"],
            "created_utc_snapshot_label": "2026-07-19T00:00:00Z",
            "source_worktree_head": source_custody["source_worktree_head"]["sha"],
            "main_head": source_custody["main_head"]["sha"],
        }
        assert config["common_readiness"]["geometry_518"]["margin_step_cap"] == "WIRED_DEFAULT_OFF_MEASURED_CAP_ABSENT"
        assert config["common_readiness"]["verdict_thresholds"]["FIRED-PAYS"] == "UNSEALED/BLOCKED"
        custody = config["common_readiness"]["checkpoint_custody"]
        assert custody["path"].endswith("banks/v9c2_defensive_bank_20260718/levelset_witness_ema_BEST.npz")
        assert custody["expected_epoch"] == 725
        assert custody["expected_bytes"] == 460448
    for ticket_id in EXPECTED_ORDER[:3]:
        assert (out_dir / ticket_id / "typed_treatment_delta.json").is_file()
        assert (out_dir / ticket_id / "witness_program.json").is_file()
    assert (out_dir / EXPECTED_ORDER[3] / "audited_config.json").is_file()
    curvelet = json.loads((out_dir / EXPECTED_ORDER[3] / "audited_config.json").read_text())
    assert curvelet["source_sha256_matches_expected"] is True
    assert curvelet["pure_compile_evidence"]["classification"] == (
        "MEASURED_AT_CURRENT_HEAD_PURE_PRODUCTION_EQUIVALENT"
    )
    for arm, expected in EXPECTED_CURVELET_HASHES.items():
        row = curvelet["pure_compile_evidence"]["arms"][arm]
        assert {key: row[key] for key in expected} == expected
        assert row["schedule_rc6"] == 0
        assert row["schedule_violation_count"] == 0


def test_treatment_delta_hashes_are_canonical_and_ticket_wrapper_refuses(tmp_path: Path) -> None:
    out_dir = tmp_path / "tickets"
    main_repo = _main_repo_with_checkpoint(tmp_path)
    materialize_duty_queue_fire_tickets(
        out_dir, main_repo=main_repo, main_source_repo=_source_root(), repo_root=_source_root()
    )

    for ticket_id in EXPECTED_ORDER[:3]:
        config = json.loads((out_dir / ticket_id / "compiled_config.json").read_text())
        treatment = config["treatment"]
        canonical = json.dumps(
            treatment["treatment_delta"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        assert treatment["canonical_treatment_delta_hash"] == hashlib.sha256(canonical).hexdigest()
        assert treatment["trainer_surface"]["override_flags_accepted"]
        assert all(treatment["trainer_surface"]["override_flags_accepted"].values())
        result = subprocess.run([str(out_dir / ticket_id / "launch.sh")], capture_output=True, text=True)
        assert result.returncode == 6
        assert "REFUSE" in result.stdout

    taper = json.loads((out_dir / EXPECTED_ORDER[0] / "typed_treatment_delta.json").read_text())
    assert taper["kind"] == "WitnessProgram.SignedLeverContrast"
    assert taper["operation"] == "REMOVE_COMPLETE_CONTROL_LEVER"
    assert taper["declared_lever_role"] == "CONTROL"
    assert taper["control_config"] == "v9_cgauge_ideal_mod19"
    assert taper["treatment_config"] == "v9_cgauge_432_taper_off"
    assert set(taper["signed_argv_diff"]) == set(taper["lever_declaration"]["overrides"])
    assert all(values[1] == "<ABSENT>" for values in taper["signed_argv_diff"].values())
    assert "overrides" not in taper

    horizon = json.loads((out_dir / EXPECTED_ORDER[1] / "typed_treatment_delta.json").read_text())
    assert horizon["operation"] == "ADD_TREATMENT_LEVER"
    assert horizon["declared_lever_role"] == "TREATMENT"
    assert all(values[0] == "<ABSENT>" for values in horizon["signed_argv_diff"].values())

    step = json.loads((out_dir / EXPECTED_ORDER[2] / "typed_treatment_delta.json").read_text())
    assert step["operation"] == "REPLACE_TREATMENT_LEVER_ENDPOINT"
    assert step["declared_lever_role"] == "TREATMENT"
    assert set(step["signed_argv_diff"]) == {"--hosc-beta-end"}


def test_output_is_stable_when_timestamp_is_omitted(tmp_path: Path) -> None:
    main_repo = _main_repo_with_checkpoint(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    materialize_duty_queue_fire_tickets(
        first, main_repo=main_repo, main_source_repo=_source_root(), repo_root=_source_root()
    )
    materialize_duty_queue_fire_tickets(
        second, main_repo=main_repo, main_source_repo=_source_root(), repo_root=_source_root()
    )

    for ticket_id in EXPECTED_ORDER:
        for relative in (
            "compiled_config.json",
            "provenance.json",
            "dry_start_receipt.json",
            "confound_self_audit.json",
            "verdict_card.md",
            "launch.sh",
            "artifact_manifest.json",
        ):
            assert (first / ticket_id / relative).read_bytes() == (second / ticket_id / relative).read_bytes()
    for relative in ("main_source_custody.json", "summary.json", "summary_manifest.json"):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_main_source_custody_fails_closed_for_non_git_repo(tmp_path: Path) -> None:
    out_dir = tmp_path / "tickets"
    main_repo = _main_repo_with_checkpoint(tmp_path)
    non_git_main_source = tmp_path / "non_git_main_source"
    non_git_main_source.mkdir()

    summary = materialize_duty_queue_fire_tickets(
        out_dir,
        main_repo=main_repo,
        main_source_repo=non_git_main_source,
        repo_root=_source_root(),
        created_utc="2026-07-19T00:00:01Z",
    )

    custody = json.loads((out_dir / "main_source_custody.json").read_text())
    mismatch_codes = {mismatch["code"] for mismatch in custody["mismatches"]}
    assert custody["status"] == "BLOCKED"
    assert custody["main_head"]["resolved"] is False
    assert custody["main_descends_from_source_head"] is None
    assert custody["merge_base_is_ancestor"]["result"] == "BLOCKED_UNRESOLVED_HEAD"
    assert "MAIN_HEAD_UNRESOLVED" in mismatch_codes
    assert "MAIN_AUTHORITY_INPUT_MISSING" in mismatch_codes
    assert summary["main_source_custody"]["status"] == "BLOCKED"
    for ticket_id in EXPECTED_ORDER:
        provenance = json.loads((out_dir / ticket_id / "provenance.json").read_text())
        assert provenance["main_source_custody"]["status"] == "BLOCKED"
