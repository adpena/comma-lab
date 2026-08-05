# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.check_subset_default_scope_fields as subset_scope
from tac.preflight import PreflightError
from tools.check_subset_default_scope_fields import (
    main,
    scan_subset_default_scope_fields,
)


def test_positive_control_silent_verdict_emitter_is_flagged(tmp_path: Path) -> None:
    p = tmp_path / "silent.py"
    p.write_text(
        "import argparse, json\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--pairs', type=int, default=32)\n"
        "def emit(path):\n"
        "    verdict = {'delta_s': -0.122, 'd_pose': 0.1}\n"
        "    path.write_text(json.dumps(verdict))\n",
        encoding="utf-8",
    )

    out = scan_subset_default_scope_fields([p])

    assert out["files_scanned"] == 1
    assert out["subset_default_sites"] == 1
    assert out["silent_verdict_subset_default"] == 1
    assert out["sites"][0]["status"] == "silent_verdict_subset_default"


def test_na3_style_scoped_receipt_is_not_flagged(tmp_path: Path) -> None:
    p = tmp_path / "scoped.py"
    p.write_text(
        "import argparse, json\n"
        "from tac.subset_selection import MODE_STRATIFIED, select\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--verdict-pairs', type=int, default=120)\n"
        "def emit(path, governing):\n"
        "    sel = select(\n"
        "        120, 600, mode=MODE_STRATIFIED, seed=20260805,\n"
        "        governing=governing, governing_name='d_pose_shipped_f16'\n"
        "    )\n"
        "    receipt = {\n"
        "        'verdict': 'MATCHED',\n"
        "        'n': sel.n,\n"
        "        'population': sel.population,\n"
        "        'selection_mode': sel.mode,\n"
        "        'axis_bias_caveat': 'pose axis prefix bias checked by governing_ratio',\n"
        "        'selection': sel.provenance(),\n"
        "    }\n"
        "    path.write_text(json.dumps(receipt))\n",
        encoding="utf-8",
    )

    out = scan_subset_default_scope_fields([p])

    assert out["subset_default_sites"] == 1
    assert out["scope_reported"] == 1
    assert out["silent_verdict_subset_default"] == 0


def test_dormant_subset_default_is_inventoried_not_flagged(tmp_path: Path) -> None:
    p = tmp_path / "dormant.py"
    p.write_text(
        "import argparse\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--pairs', type=int, default=16)\n",
        encoding="utf-8",
    )

    out = scan_subset_default_scope_fields([p])

    assert out["subset_default_sites"] == 1
    assert out["dormant_no_verdict"] == 1
    assert out["silent_verdict_subset_default"] == 0


def test_full_population_num_pairs_default_is_not_a_subset_site(tmp_path: Path) -> None:
    p = tmp_path / "n600.py"
    p.write_text(
        "import argparse, json\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--num-pairs', type=int, default=600)\n"
        "def emit(args, path):\n"
        "    rows = []\n"
        "    for i in range(args.num_pairs):\n"
        "        rows.append({'i': i, 'd_seg': 0.0})\n"
        "    path.write_text(json.dumps({'verdict': 'n600', 'rows': rows}))\n",
        encoding="utf-8",
    )

    out = scan_subset_default_scope_fields([p])

    assert out["subset_default_sites"] == 0


def test_experiments_results_is_excluded_from_live_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subset_scope, "REPO", tmp_path)
    live = tmp_path / "experiments" / "live.py"
    archived = tmp_path / "experiments" / "results" / "bundle.py"
    live.parent.mkdir(parents=True)
    archived.parent.mkdir(parents=True)
    live.write_text(
        "import argparse\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--pairs', type=int, default=16)\n",
        encoding="utf-8",
    )
    archived.write_text(
        "import argparse\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--pairs', type=int, default=16)\n",
        encoding="utf-8",
    )

    out = scan_subset_default_scope_fields([tmp_path / "experiments"])

    assert out["files_scanned"] == 1
    assert out["sites"][0]["path"] == "experiments/live.py"


def test_preflight_wrapper_warns_and_strict_raises(tmp_path: Path) -> None:
    from tac.preflight import check_subset_default_scope_fields

    p = tmp_path / "tools" / "known_silent.py"
    p.parent.mkdir(parents=True)
    p.write_text(
        "import argparse, json\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--pairs', type=int, default=32)\n"
        "def emit(path):\n"
        "    path.write_text(json.dumps({'verdict': 'WIN', 'score': 0.1}))\n",
        encoding="utf-8",
    )

    violations = check_subset_default_scope_fields(
        repo_root=tmp_path, strict=False, verbose=False
    )

    assert len(violations) == 1
    assert "known_silent.py" in violations[0]
    with pytest.raises(PreflightError):
        check_subset_default_scope_fields(repo_root=tmp_path, strict=True, verbose=False)


def test_baseline_refuses_only_new_silent_sites(tmp_path: Path, capsys) -> None:
    old = tmp_path / "old.py"
    old.write_text(
        "import argparse, json\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--pairs', type=int, default=32)\n"
        "def emit(path):\n"
        "    path.write_text(json.dumps({'verdict': 'WIN', 'score': 0.1}))\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(scan_subset_default_scope_fields([old])), encoding="utf-8")

    assert main([str(old), "--baseline", str(baseline)]) == 0

    new = tmp_path / "new.py"
    new.write_text(old.read_text(encoding="utf-8"), encoding="utf-8")

    assert main([str(old), str(new), "--baseline", str(baseline)]) == 1
    assert "NEW" in capsys.readouterr().out
