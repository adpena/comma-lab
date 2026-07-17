"""Tests for the 2026-07-17 bug-class sweep: CLASS-1 (raw-vm safety basis) + CLASS-2 (observer
flag exclusion) gates and the tools/argv_role helper."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tac.confound_gates import (  # noqa: E402
    check_no_raw_virtual_memory_safety_basis,
    check_process_guard_excludes_observer_flag_values,
)

argv_role = importlib.import_module("tools.argv_role")


# ─────────────────────────── argv_role ───────────────────────────
def test_strip_observer_flag_values_space_form():
    cl = ["python", "sup.py", "--training-sig", "train_levelset_witness", "--port", "8790"]
    out = argv_role.strip_observer_flag_values(cl)
    assert "train_levelset_witness" not in " ".join(out)
    assert "--port" in out


def test_strip_observer_flag_values_equals_form():
    cl = ["python", "x.py", "--training-sig=train_witness"]
    assert "train_witness" not in " ".join(argv_role.strip_observer_flag_values(cl))


def test_observer_not_classified_as_launch():
    obs = ["python", "tools/dashboard_supervisor.py", "--training-sig", "train_levelset_witness"]
    assert argv_role.is_observer_stripped_launch(obs) is False


def test_real_trainer_argv_still_a_launch():
    raw = ["python", "experiments/train_levelset_witness_realized_through_R_mlx.py", "--x", "1"]
    assert argv_role.is_observer_stripped_launch(raw) is True


def test_empty_cmdline_safe():
    assert argv_role.strip_observer_flag_values(None) == []
    assert argv_role.is_observer_stripped_launch([]) is False


# ─────────────────────────── CLASS-1 gate ───────────────────────────
def test_class1_gate_clean_on_repo():
    assert check_no_raw_virtual_memory_safety_basis(strict=False, verbose=False) == []


def test_class1_gate_catches_raw_available(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "bad.py").write_text(
        "import psutil\n"
        "def guard():\n"
        "    if psutil.virtual_memory().available < 1: raise SystemExit(7)\n"
    )
    v = check_no_raw_virtual_memory_safety_basis(repo_root=tmp_path, strict=False, verbose=False)
    assert any("bad.py" in x and "available" in x for x in v)


def test_class1_gate_respects_waiver(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "tele.py").write_text(
        "import psutil\n"
        "def show():\n"
        "    return psutil.virtual_memory().available  # RAW_VM_BASIS_OK:telemetry display only\n"
    )
    assert check_no_raw_virtual_memory_safety_basis(repo_root=tmp_path, strict=False, verbose=False) == []


def test_class1_gate_ignores_total(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "denom.py").write_text(
        "import psutil\n"
        "def total_gib():\n"
        "    return psutil.virtual_memory().total / 2**30\n"
    )
    assert check_no_raw_virtual_memory_safety_basis(repo_root=tmp_path, strict=False, verbose=False) == []


# ─────────────────────────── CLASS-2 gate ───────────────────────────
def test_class2_gate_clean_on_repo():
    assert check_process_guard_excludes_observer_flag_values(strict=False, verbose=False) == []


def test_class2_gate_catches_unexcluded_token_guard(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "badguard.py").write_text(
        "import psutil\n"
        "def refuse_dup(out_dir):\n"
        "    for p in psutil.process_iter(['cmdline']):\n"
        "        cl = ' '.join(p.info.get('cmdline') or ())\n"
        "        if 'train_levelset_witness' in cl:\n"
        "            raise SystemExit(12)\n"
    )
    v = check_process_guard_excludes_observer_flag_values(repo_root=tmp_path, strict=False, verbose=False)
    assert any("badguard.py" in x for x in v)


def test_class2_gate_respects_argv_role_exclusion(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "goodguard.py").write_text(
        "import psutil\n"
        "from tools.argv_role import strip_observer_flag_values\n"
        "def refuse_dup(out_dir):\n"
        "    for p in psutil.process_iter(['cmdline']):\n"
        "        scan = ' '.join(strip_observer_flag_values(p.info.get('cmdline') or ()))\n"
        "        if 'train_levelset_witness' in scan:\n"
        "            raise SystemExit(12)\n"
    )
    assert check_process_guard_excludes_observer_flag_values(repo_root=tmp_path, strict=False, verbose=False) == []


def test_class2_gate_respects_waiver(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "telemetry_guard.py").write_text(
        "import psutil\n"
        "def report(out_dir):  # OBSERVER_ROLE_OK: telemetry status, not a refuse decision\n"
        "    for p in psutil.process_iter(['cmdline']):\n"
        "        cl = ' '.join(p.info.get('cmdline') or ())\n"
        "        if 'train_levelset_witness' in cl:\n"
        "            sys.exit(1)\n"
    )
    assert check_process_guard_excludes_observer_flag_values(repo_root=tmp_path, strict=False, verbose=False) == []
