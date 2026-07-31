# SPDX-License-Identifier: MIT
"""ddm_gh1 #830 — the raw-``virtual_memory`` safety-basis gate: ownership, scope, denominator.

The gate was OWNERLESS and RED: its docstring declared live-count 0 while it MEASURED 6. All six
were re-derived and closed in the fix landing, and the gate now (a) DECLARES its scan denominator
so "0 violations" cannot be confused with "0 scanned", (b) covers ``experiments/*.py`` and
``scripts/*.py``, which were silently out of scope, and (c) runs STRICT.

Every test here carries a POSITIVE CONTROL: a fixture the gate MUST still flag. A future narrowing
that guts the detector fails loudly instead of printing a clean OK over an empty scan.
"""
from __future__ import annotations

import pytest

from tac.confound_gates import (
    _python_source_files,
    check_no_raw_virtual_memory_safety_basis,
)

GUARD = (
    "import psutil\n"
    "def guard():\n"
    "    if psutil.virtual_memory().available < 10:\n"
    "        raise RuntimeError('refuse')\n"
)


def _seed(root, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(GUARD)


def test_live_count_is_zero_on_the_real_repo():
    assert check_no_raw_virtual_memory_safety_basis(strict=False, verbose=False) == []


def test_strict_does_not_raise_on_the_real_repo():
    """The gate is in ``_CONFOUND_STRICT``; a nonzero live count would refuse every launch."""
    assert check_no_raw_virtual_memory_safety_basis(strict=True, verbose=False) == []


def test_gate_is_registered_strict_in_preflight():
    import inspect

    from tac import preflight

    source = inspect.getsource(preflight.preflight_all)
    marker = '"check_no_raw_virtual_memory_safety_basis",'
    assert marker in source, "gate is not in the _CONFOUND_STRICT set (ownership regression)"


@pytest.mark.parametrize(
    "relative",
    [
        "tools/guard.py",
        "src/tac/optimization/guard.py",
        "experiments/guard.py",  # #830: was silently OUT of scope
        "scripts/guard.py",      # #830: was silently OUT of scope
    ],
)
def test_positive_control_every_declared_scope_leg_is_actually_scanned(tmp_path, relative):
    """POSITIVE CONTROL. Each declared scope leg must really flag a planted violation."""
    _seed(tmp_path, relative)
    violations = check_no_raw_virtual_memory_safety_basis(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any(relative in v for v in violations), (
        f"{relative} is inside the DECLARED scope but the gate did not scan it"
    )


def test_waiver_still_suppresses():
    """The designed escape hatch must keep working (telemetry-only / last-resort fallback)."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        path = root / "tools" / "telemetry.py"
        path.parent.mkdir(parents=True)
        path.write_text(
            "import psutil\n"
            "def show():\n"
            "    return psutil.virtual_memory().available  # RAW_VM_BASIS_OK: display only\n"
        )
        assert check_no_raw_virtual_memory_safety_basis(
            repo_root=root, strict=False, verbose=False
        ) == []


def test_vendored_and_result_trees_stay_out_of_the_denominator(tmp_path):
    """The scope is a DECLARATION, not an accident: experiments/ + scripts/ are TOP-LEVEL only,
    so vendored venvs and frozen run bundles cannot swamp the denominator."""
    _seed(tmp_path, "experiments/results/run_x/source_bundle/tools/vendored.py")
    _seed(tmp_path, "experiments/manim/.venv/lib/python3.12/site-packages/numpy/x.py")
    _seed(tmp_path, "experiments/real_guard.py")
    considered = {p.relative_to(tmp_path).as_posix() for p in _python_source_files(tmp_path)}
    assert "experiments/real_guard.py" in considered
    assert not any(part in c for c in considered for part in ("results/", ".venv/"))


def test_tests_directories_remain_excluded(tmp_path):
    _seed(tmp_path, "tools/tests/guard.py")
    _seed(tmp_path, "src/tac/tests/guard.py")
    assert check_no_raw_virtual_memory_safety_basis(
        repo_root=tmp_path, strict=False, verbose=False
    ) == []


def test_canonical_providers_are_excluded_by_construction(tmp_path):
    """``mem_basis`` / ``system_memory_governor`` ARE the reclaimable-aware basis; they must read
    raw psutil without a waiver, and must never appear in the denominator."""
    _seed(tmp_path, "tools/mem_basis.py")
    _seed(tmp_path, "tools/system_memory_governor.py")
    considered = {p.relative_to(tmp_path).as_posix() for p in _python_source_files(tmp_path)}
    assert considered == set()
    assert check_no_raw_virtual_memory_safety_basis(
        repo_root=tmp_path, strict=False, verbose=False
    ) == []


def test_the_three_repaired_guards_route_through_the_canonical_basis():
    """Wire-in proof for the 3 GENUINE sites (the other 3 are telemetry-only waivers)."""
    import pathlib

    for rel in (
        "tools/remeasure_ddm_e4_ws1_packet.py",
        "tools/run_ddm_j12_receiver_coordinate_custody.py",
        "tools/run_ddm_ms2r_r3_366box_typed_fisher_g4_waterfill.py",
    ):
        text = pathlib.Path(rel).read_text()
        assert "conservative_free_gib" in text, f"{rel} does not use the canonical basis"
        assert "default=0.0" in text, f"{rel} must fail CLOSED when memory is unmeasurable"
