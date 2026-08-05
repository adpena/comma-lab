# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.check_no_silent_cap_defaults as silent_caps
from tac.optimization.trajectory_stopping import (
    TrajectoryStoppingError,
    build_cap_stop_receipt,
)
from tac.preflight import PreflightError, check_no_silent_cap_defaults
from tools.check_no_silent_cap_defaults import main, scan_cap_defaults


def test_scanner_reports_the_denominator_and_silent_site(tmp_path: Path) -> None:
    p = tmp_path / "silent.py"
    p.write_text(
        "import argparse\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--steps', type=int, default=25)\n",
        encoding="utf-8",
    )

    out = scan_cap_defaults([p])

    assert out["files_scanned"] == 1
    assert out["cap_default_sites"] == 1
    assert out["silent_cap_default_sites"] == 1
    assert out["sites"][0]["flag"] == "--steps"
    assert out["sites"][0]["status"] == "silent_cap_default"


def test_stop_reason_marker_clears_the_site(tmp_path: Path) -> None:
    p = tmp_path / "reported.py"
    p.write_text(
        "import argparse\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--max-iterations', type=int, default=50)\n"
        "def emit():\n"
        "    return {'stop_reason': 'iteration_cap_best_at_cap'}\n",
        encoding="utf-8",
    )

    out = scan_cap_defaults([p])

    assert out["cap_default_sites"] == 1
    assert out["silent_cap_default_sites"] == 0
    assert out["sites"][0]["status"] == "reports_stop_reason"


def test_cap_stop_receipt_payload_is_the_canonical_small_shape() -> None:
    receipt = build_cap_stop_receipt(
        stop_reason="cap_bound",
        steps_run=25,
        cap=25,
        still_descending=True,
    )

    assert receipt.to_payload() == {
        "stop_reason": "cap_bound",
        "steps_run": 25,
        "cap": 25,
        "still_descending": True,
    }
    with pytest.raises(TrajectoryStoppingError):
        build_cap_stop_receipt(
            stop_reason="converged",
            steps_run=3,
            cap=25,
            still_descending=True,
        )


def test_cap_stop_receipt_marker_clears_trajectory_consumer(tmp_path: Path) -> None:
    p = tmp_path / "consumer.py"
    p.write_text(
        "import argparse\n"
        "from tac.optimization.trajectory_stopping import build_cap_stop_receipt\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--max-steps', type=int, default=25)\n"
        "def emit():\n"
        "    return build_cap_stop_receipt(\n"
        "        stop_reason='cap_bound', steps_run=25, cap=25, still_descending=True\n"
        "    ).to_payload()\n",
        encoding="utf-8",
    )

    out = scan_cap_defaults([p])

    assert out["cap_default_sites"] == 1
    assert out["silent_cap_default_sites"] == 0
    assert out["sites"][0]["status"] == "reports_stop_reason"


def test_experiments_results_is_not_part_of_live_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(silent_caps, "REPO", tmp_path)
    live = tmp_path / "experiments" / "live.py"
    archived = tmp_path / "experiments" / "results" / "bundle.py"
    live.parent.mkdir(parents=True)
    archived.parent.mkdir(parents=True)
    live.write_text(
        "import argparse\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--steps', type=int, default=25)\n",
        encoding="utf-8",
    )
    archived.write_text(
        "import argparse\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--steps', type=int, default=999)\n",
        encoding="utf-8",
    )

    out = scan_cap_defaults([tmp_path / "experiments"])

    assert out["files_scanned"] == 1
    assert out["sites"][0]["path"] == "experiments/live.py"


def test_preflight_wrapper_warns_and_strict_raises(tmp_path: Path) -> None:
    p = tmp_path / "tools" / "known_censored.py"
    p.parent.mkdir(parents=True)
    p.write_text(
        "import argparse\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--max-iters', type=int, default=3)\n",
        encoding="utf-8",
    )

    violations = check_no_silent_cap_defaults(
        repo_root=tmp_path, strict=False, verbose=False
    )

    assert len(violations) == 1
    assert "known_censored.py" in violations[0]
    with pytest.raises(PreflightError):
        check_no_silent_cap_defaults(repo_root=tmp_path, strict=True, verbose=False)


def test_baseline_refuses_only_new_silent_caps(tmp_path: Path, capsys) -> None:
    old = tmp_path / "old.py"
    old.write_text(
        "import argparse\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--steps', type=int, default=25)\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(scan_cap_defaults([old])), encoding="utf-8")

    assert main([str(old), "--baseline", str(baseline)]) == 0

    new = tmp_path / "new.py"
    new.write_text(
        "import argparse\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--max-iters', type=int, default=3)\n",
        encoding="utf-8",
    )

    assert main([str(old), str(new), "--baseline", str(baseline)]) == 1
    assert "NEW" in capsys.readouterr().out
