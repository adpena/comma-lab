# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

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
