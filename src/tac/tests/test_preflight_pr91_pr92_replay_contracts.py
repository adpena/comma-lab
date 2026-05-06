from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments/preflight_pr91_pr92_replay_contracts.py"
PR92_LOG_DIR = (
    REPO_ROOT
    / "experiments/results/lightning_batch/exact_eval_pr85_stbm1br_pr92_rmb1_t4_20260504T082220Z"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("preflight_pr91_pr92_replay_contracts", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_pr91_probability_report(path: Path) -> None:
    payload = {
        "score_claim": False,
        "dispatch_allowed": False,
        "hpm1_static_contract": {"status": "passed"},
        "variant_results": [
            {
                "variant": "source_float64_perfect_false",
                "status": "failed_closed",
                "failure_reason": "hpac_entropy_decode_contract_mismatch",
            },
            {
                "variant": "source_float32_perfect_false",
                "status": "failed_closed",
                "failure_reason": "hpac_entropy_decode_contract_mismatch",
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.skipif(
    not (PR92_LOG_DIR / "auth_eval.log").is_file()
    or not (PR92_LOG_DIR / "adjudication.log").is_file(),
    reason="PR92 exact-eval logs not present",
)
def test_pr92_contract_recovers_a_plus_plus_from_exact_logs(tmp_path: Path) -> None:
    mod = _load_module()

    report = mod.validate_pr92_rmb1_contract(
        manifest_path=tmp_path / "missing_manifest.json",
        exact_json_path=tmp_path / "missing_exact.json",
        log_dir=PR92_LOG_DIR,
    )

    assert report["status"] == "passed_t4_exact_pr92_rmb1_stack_validated"
    assert report["evidence_grade"] == "A++"
    assert report["dispatch_allowed"] is False
    assert report["source"]["mode"] == "recovered_from_logs"
    assert report["exact_eval"]["score"] == pytest.approx(0.2535063602939779)
    assert report["exact_eval"]["archive_bytes"] == 229480
    assert report["failed_checks"] == []


def test_pr91_contract_blocks_when_no_probability_variant_passes(tmp_path: Path) -> None:
    mod = _load_module()
    archive = tmp_path / "pr91_archive.zip"
    archive.write_bytes(b"synthetic archive placeholder")
    pr91_report = tmp_path / "pr91_probability.json"
    _write_pr91_probability_report(pr91_report)

    report = mod.validate_pr91_hpm1_contract(
        archive=archive,
        probability_report=pr91_report,
        rerun=False,
    )

    assert report["status"] == "blocked_hpm1_probability_range_contract_mismatch"
    assert report["score_claim"] is False
    assert report["dispatch_allowed"] is False
    assert report["classification"]["bug_class"] == "hpm1_probability_range_contract_mismatch"
    assert report["classification"]["passed_variants"] == []
    assert report["failed_checks"] == []


@pytest.mark.skipif(
    not (PR92_LOG_DIR / "auth_eval.log").is_file()
    or not (PR92_LOG_DIR / "adjudication.log").is_file(),
    reason="PR92 exact-eval logs not present",
)
def test_combined_preflight_report_pr92_validated_pr91_fail_closed(tmp_path: Path) -> None:
    mod = _load_module()
    archive = tmp_path / "pr91_archive.zip"
    archive.write_bytes(b"synthetic archive placeholder")
    pr91_report = tmp_path / "pr91_probability.json"
    _write_pr91_probability_report(pr91_report)
    args = argparse.Namespace(
        pr91_archive=archive,
        pr91_probability_report=pr91_report,
        rerun_pr91_prefix=False,
        pr92_manifest=tmp_path / "missing_manifest.json",
        pr92_exact_json=tmp_path / "missing_exact.json",
        pr92_log_dir=PR92_LOG_DIR,
    )

    report = mod.build_report(args)

    assert report["status"] == "passed_pr92_a_plus_plus_pr91_fail_closed"
    assert report["dispatch_performed"] is False
    assert report["remote_jobs_dispatched"] is False
    assert report["pr91_hpm1"]["status"] == "blocked_hpm1_probability_range_contract_mismatch"
    assert report["pr92_rmb1_stack"]["status"] == "passed_t4_exact_pr92_rmb1_stack_validated"
    assert "Recover byte-exact HPM1" in report["next_actions"][0]


@pytest.mark.skipif(
    not (PR92_LOG_DIR / "auth_eval.log").is_file()
    or not (PR92_LOG_DIR / "adjudication.log").is_file(),
    reason="PR92 exact-eval logs not present",
)
def test_preflight_cli_writes_json_and_ledger(tmp_path: Path) -> None:
    mod = _load_module()
    archive = tmp_path / "pr91_archive.zip"
    archive.write_bytes(b"synthetic archive placeholder")
    pr91_report = tmp_path / "pr91_probability.json"
    _write_pr91_probability_report(pr91_report)
    output_json = tmp_path / "out" / "preflight.json"
    ledger_md = tmp_path / "out" / "preflight.md"

    rc = mod.main(
        [
            "--pr91-archive",
            str(archive),
            "--pr91-probability-report",
            str(pr91_report),
            "--pr92-manifest",
            str(tmp_path / "missing_manifest.json"),
            "--pr92-exact-json",
            str(tmp_path / "missing_exact.json"),
            "--pr92-log-dir",
            str(PR92_LOG_DIR),
            "--output-json",
            str(output_json),
            "--ledger-md",
            str(ledger_md),
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "passed_pr92_a_plus_plus_pr91_fail_closed"
    text = ledger_md.read_text(encoding="utf-8")
    assert "PR91 HPM1" in text
    assert "PR92 RMB1 Stack" in text
