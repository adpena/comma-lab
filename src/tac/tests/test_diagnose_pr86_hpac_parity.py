from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO / "experiments" / "diagnose_pr86_hpac_parity.py"
PR86_DIR = REPO / "experiments/results/public_pr86_intake_20260504_codex"


def _load_script():
    spec = importlib.util.spec_from_file_location("diagnose_pr86_hpac_parity_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_report_classifies_pr86_as_fail_closed_decode_blocker() -> None:
    script = _load_script()

    report = script.build_report()

    assert report["score_claim"] is False
    assert report["dispatch_performed"] is False
    assert report["gpu_or_remote_work"] is False
    assert report["status"] == "blocked"
    assert report["archive_custody"]["size_bytes"] == 207579
    assert (
        report["archive_custody"]["sha256"]
        == "e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef"
    )
    assert report["archive_custody"]["member_contract_status"] == "passed"
    assert report["artifact_identity_consistency"]["status"] == "passed"

    hpac = report["hpac_replay_parity"]
    assert hpac["blocker_class"] == "hpac_entropy_decode_contract_mismatch"
    stages = {row["stage"]: row for row in hpac["stage_statuses"]}
    assert stages["submitted_tokens_decode"]["status"] == "failed_closed"
    assert stages["submitted_tokens_decode"]["error_type"] == "AssertionError"
    assert stages["submitted_tokens_decode"]["failed_at"] == {
        "frame": 0,
        "group": 10,
        "symbol_in_group": 191,
    }
    assert stages["decode_then_reencode_byte_parity"]["status"] == "blocked_not_reached"
    assert hpac["token_semantics"]["submitted_archive_token_encoding"] == "raw_tokens"


def test_archive_custody_fails_closed_on_member_byte_mismatch(tmp_path: Path) -> None:
    script = _load_script()
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
        for name in script.REQUIRED_MEMBERS:
            zf.writestr(name, b"x")

    custody = script.archive_custody(archive)

    assert custody["exists"] is True
    assert custody["identity_matches_expected"] is False
    assert custody["member_contract_status"] == "blocked"
    assert custody["byte_mismatches"]["tokens.bin"] == {
        "expected": 113900,
        "actual": 1,
    }


def test_archive_custody_fails_closed_on_unsafe_or_duplicate_members(tmp_path: Path) -> None:
    script = _load_script()
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
        for name, size in script.EXPECTED_MEMBER_BYTES.items():
            zf.writestr(name, b"x" * size)
        zf.writestr("../escape", b"bad")
        zf.writestr("tokens.bin", b"duplicate")

    custody = script.archive_custody(archive)

    assert custody["member_contract_status"] == "blocked"
    assert custody["duplicate_member_names"] == ["tokens.bin"]
    assert custody["unsafe_member_names"] == ["../escape"]
    assert custody["unexpected_members"] == ["../escape"]


def test_artifact_identity_consistency_detects_stale_artifact() -> None:
    script = _load_script()
    custody = {
        "sha256": "archive-sha",
        "size_bytes": 10,
    }
    artifacts = {
        "good.json": {"archive": {"sha256": "archive-sha", "size_bytes": 10}},
        "stale.json": {"archive": {"sha256": "other", "size_bytes": 10}},
    }

    report = script.artifact_identity_consistency(custody, artifacts)

    assert report["status"] == "blocked"
    rows = {row["artifact"]: row for row in report["artifacts"]}
    assert rows["good.json"]["matches_local_archive"] is True
    assert rows["stale.json"]["matches_local_archive"] is False


def test_cli_writes_optional_json_to_requested_path(tmp_path: Path) -> None:
    script = _load_script()
    out = tmp_path / "report.json"

    assert script.main(["--json-out", str(out)]) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["tool"] == "experiments/diagnose_pr86_hpac_parity.py"
    assert payload["status"] == "blocked"
    assert payload["public_design_claim"]["public_claimed_score"] == "0.27"


def test_main_prints_without_repo_output_by_default(capsys) -> None:
    script = _load_script()

    assert script.main([]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["source_artifacts"] == sorted(
        [
            "experiments/results/public_pr86_intake_20260504_codex/pr86_hpac_full_decode_reencode_gate_20260504_codex.json",
            "experiments/results/public_pr86_intake_20260504_codex/pr86_hpac_pr85_qma9_parity_probe.json",
            "experiments/results/public_pr86_intake_20260504_codex/pr86_hpac_token_anatomy_forensics.json",
        ]
    ) + ["experiments/results/public_pr86_intake_20260504_codex/pr86_view.json"]
