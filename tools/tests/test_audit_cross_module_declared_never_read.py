from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.audit_cross_module_declared_never_read import (
    POSITIVE_CONTROL,
    audit_repository,
    controls_failure_text,
    run_canonical_controls,
)


def _write(root: Path, rel: str, body: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def _keys(report: dict) -> set[str]:
    return {hit["key"] for hit in report["hits"]}


def _audit_fixture(root: Path, **kwargs: object) -> dict:
    return audit_repository(root, scope="broad", **kwargs)


def test_canonical_control_gate_passes_on_live_repo() -> None:
    report = audit_repository(Path(__file__).resolve().parents[2])
    controls = report["controls"]
    assert controls["passed"], controls_failure_text(controls)
    assert POSITIVE_CONTROL in _keys(report)
    for item in controls["negatives"]:
        assert item["key"] not in _keys(report)
    assert report["denominator"]["files_total"] > 100
    assert report["denominator"]["chains_traced"] >= 8


def test_synthetic_two_module_positive(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/tac/a.py",
        "class Bad:\n"
        "    def __init__(self, *, lever=None, used=None):\n"
        "        self.lever = lever\n"
        "        self.used = used\n"
        "    def run(self):\n"
        "        return self.used\n",
    )
    _write(
        tmp_path,
        "tools/driver.py",
        "from tac.a import Bad\n"
        "def build():\n"
        "    return Bad(used=1)\n",
    )
    report = _audit_fixture(tmp_path)
    assert "tac.a.Bad.lever" in _keys(report)
    assert "tac.a.Bad.used" not in _keys(report)
    hit = next(hit for hit in report["hits"] if hit["key"] == "tac.a.Bad.lever")
    assert hit["external_constructor_calls"][0]["bindings"]["lever"] == "defaulted"


def test_consumed_negative_and_same_file_only_exclusion(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/tac/a.py",
        "class LocalOnly:\n"
        "    def __init__(self, *, x=None):\n"
        "        self.x = x\n"
        "def build():\n"
        "    return LocalOnly()\n"
        "class Consumed:\n"
        "    def __init__(self, *, x=None):\n"
        "        self.x = x\n"
        "    def read(self):\n"
        "        return self.x\n",
    )
    _write(tmp_path, "tools/driver.py", "from tac.a import Consumed\nc = Consumed(x=1)\n")
    report = _audit_fixture(tmp_path)
    assert "tac.a.LocalOnly.x" not in _keys(report)
    assert "tac.a.Consumed.x" not in _keys(report)


def test_reexport_import_chain_is_traced(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/tac/base.py",
        "class Bad:\n"
        "    def __init__(self, *, x=None):\n"
        "        self.x = x\n",
    )
    _write(tmp_path, "src/tac/reexport.py", "from tac.base import Bad\n")
    _write(tmp_path, "tools/driver.py", "from tac.reexport import Bad\nobj = Bad()\n")
    report = _audit_fixture(tmp_path)
    hit = next(hit for hit in report["hits"] if hit["key"] == "tac.base.Bad.x")
    assert hit["shortest_import_hops"] == 2
    assert hit["external_constructor_calls"][0]["import_hops"] == [
        "tac.reexport",
        "tac.base",
    ]


def test_unrelated_same_name_attribute_does_not_consume(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/tac/a.py",
        "class Bad:\n"
        "    def __init__(self, *, x=None):\n"
        "        self.x = x\n"
        "class Other:\n"
        "    def __init__(self):\n"
        "        self.x = 1\n"
        "    def read(self):\n"
        "        return self.x\n",
    )
    _write(tmp_path, "tools/driver.py", "from tac.a import Bad\nbad = Bad()\n")
    assert "tac.a.Bad.x" in _keys(_audit_fixture(tmp_path))


def test_test_only_read_is_classified_but_not_consuming(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/tac/a.py",
        "class Bad:\n"
        "    def __init__(self, *, x=None):\n"
        "        self.x = x\n",
    )
    _write(tmp_path, "tools/driver.py", "from tac.a import Bad\nobj = Bad()\n")
    _write(
        tmp_path,
        "src/tac/tests/test_a.py",
        "from tac.a import Bad\n"
        "def test_read():\n"
        "    obj = Bad(x=1)\n"
        "    assert obj.x == 1\n",
    )
    report = _audit_fixture(tmp_path, include_tests=True)
    hit = next(hit for hit in report["hits"] if hit["key"] == "tac.a.Bad.x")
    assert hit["test_only_reads"]
    assert hit["production_reads"] == []


def test_argparse_consumed_and_unconsumed_behavior(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tools/cli.py",
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--used-flag')\n"
        "p.add_argument('--dead-flag', dest='dead')\n"
        "args = p.parse_args([])\n"
        "print(args.used_flag)\n",
    )
    report = _audit_fixture(tmp_path)
    keys = _keys(report)
    assert "tools.cli:argparse:used_flag" not in keys
    assert "tools.cli:argparse:dead" in keys
    hit = next(hit for hit in report["hits"] if hit["key"] == "tools.cli:argparse:dead")
    assert hit["declaration"]["line"] == 4


def test_results_directory_exclusion_and_parse_error_accounting(tmp_path: Path) -> None:
    _write(tmp_path, "experiments/results/bad.py", "this is not python !\n")
    _write(tmp_path, "src/tac/bad_parse.py", "def nope(:\n")
    _write(tmp_path, "tools/clean.py", "VALUE = 1\n")
    report = _audit_fixture(tmp_path)
    denom = report["denominator"]
    assert denom["excluded_results_files"] == 1
    assert denom["parse_error_count"] == 1
    assert denom["parse_errors"][0]["path"] == "src/tac/bad_parse.py"


def test_deterministic_rank_and_output(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/tac/a.py",
        "class Bad:\n"
        "    def __init__(self, *, x=None):\n"
        "        self.x = x\n",
    )
    _write(tmp_path, "tools/driver.py", "from tac.a import Bad\nobj = Bad()\n")
    a = _audit_fixture(tmp_path)
    b = _audit_fixture(tmp_path)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    hit = next(hit for hit in a["hits"] if hit["key"] == "tac.a.Bad.x")
    assert hit["blast_radius_rank"] == 16


def test_control_gate_fails_before_cli_sweep_output_when_positive_missing(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "tools/clean.py", "VALUE = 1\n")
    report = _audit_fixture(tmp_path)
    controls = run_canonical_controls(report)
    assert not controls["passed"]
    proc = subprocess.run(
        [
            sys.executable,
            "tools/audit_cross_module_declared_never_read.py",
            "--repo-root",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 2
    assert "CONTROL FAILURE" in proc.stderr
    assert "cross_module_declared_never_read.v1:" not in proc.stdout
