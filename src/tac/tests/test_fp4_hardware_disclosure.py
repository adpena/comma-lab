"""Regression tests for Check 40: FP4 hardware-disclosure markers.

Catches the Lane F bug class — production FakeQuantFP4 instantiation in code
paths without disclosing that 4090 (CC 8.9) only supports SIMULATED FP4
(NVFP4 hardware needs Blackwell CC 10.0).

Reference: project_cosmos_deep_dive_addendum_20260428.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    check_fp4_production_paths_disclose_hardware,
    _scan_for_fp4_production_paths,
)


def _setup_fake_repo(root: Path) -> None:
    (root / "src" / "tac").mkdir(parents=True, exist_ok=True)
    (root / "experiments").mkdir(parents=True, exist_ok=True)


def test_strict_passes_on_real_codebase() -> None:
    """Live repo must pass — every production FP4 path discloses hardware."""
    violations = check_fp4_production_paths_disclose_hardware(
        strict=False, verbose=False,
    )
    assert violations == [], (
        f"check_fp4_production_paths_disclose_hardware found "
        f"{len(violations)} violation(s). Add disclosure marker or "
        f"exempt with WHY comment. Violations:\n"
        + "\n".join(f"  • {v}" for v in violations)
    )


def test_detects_undisclosed_fakequant_call(tmp_path: Path) -> None:
    """Production .py with FakeQuantFP4(...) call but no disclosure → flag."""
    _setup_fake_repo(tmp_path)
    bad = tmp_path / "experiments" / "do_fp4.py"
    bad.write_text(textwrap.dedent('''\
        from tac.quantization import FakeQuantFP4
        def quantize(x):
            return FakeQuantFP4(bits=4)(x)
    '''))
    violations = check_fp4_production_paths_disclose_hardware(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "do_fp4.py" in violations[0]
    assert "Blackwell" in violations[0] or "CC" in violations[0]


def test_passes_with_runtime_banner_marker(tmp_path: Path) -> None:
    """[SIMULATED-FP4] string literal counts as disclosure."""
    _setup_fake_repo(tmp_path)
    good = tmp_path / "experiments" / "do_fp4.py"
    good.write_text(textwrap.dedent('''\
        from tac.quantization import FakeQuantFP4
        print("[SIMULATED-FP4] hardware capability < 10.0 — FP4 simulated")
        def quantize(x):
            return FakeQuantFP4(bits=4)(x)
    '''))
    assert check_fp4_production_paths_disclose_hardware(
        repo_root=tmp_path, strict=False, verbose=False,
    ) == []


def test_passes_with_hardware_disclosed_comment(tmp_path: Path) -> None:
    """`# FP4_HARDWARE_DISCLOSED:` comment counts as disclosure."""
    _setup_fake_repo(tmp_path)
    good = tmp_path / "experiments" / "do_fp4.py"
    good.write_text(textwrap.dedent('''\
        from tac.quantization import FakeQuantFP4
        def quantize(x):
            # FP4_HARDWARE_DISCLOSED: 4090 simulates FP4 in FP32
            return FakeQuantFP4(bits=4)(x)
    '''))
    assert check_fp4_production_paths_disclose_hardware(
        repo_root=tmp_path, strict=False, verbose=False,
    ) == []


def test_passes_with_assert_helper_call(tmp_path: Path) -> None:
    """Calling assert_quantization_hardware_supported counts as disclosure."""
    _setup_fake_repo(tmp_path)
    good = tmp_path / "experiments" / "do_fp4.py"
    good.write_text(textwrap.dedent('''\
        from tac.quantization import FakeQuantFP4, assert_quantization_hardware_supported
        def quantize(x, device):
            assert_quantization_hardware_supported("fp4", device, allow_simulation=True)
            return FakeQuantFP4(bits=4)(x)
    '''))
    assert check_fp4_production_paths_disclose_hardware(
        repo_root=tmp_path, strict=False, verbose=False,
    ) == []


def test_test_files_exempt(tmp_path: Path) -> None:
    """Files under tests/ should NOT be flagged — testing FP4 simulation
    is fine; the test IS the simulation context."""
    _setup_fake_repo(tmp_path)
    (tmp_path / "src" / "tac" / "tests").mkdir(parents=True, exist_ok=True)
    test_file = tmp_path / "src" / "tac" / "tests" / "test_fp4.py"
    test_file.write_text(textwrap.dedent('''\
        from tac.quantization import FakeQuantFP4
        def test_fp4_quantization():
            assert FakeQuantFP4(bits=4) is not None
    '''))
    assert check_fp4_production_paths_disclose_hardware(
        repo_root=tmp_path, strict=False, verbose=False,
    ) == []


def test_lowercase_function_call_detected(tmp_path: Path) -> None:
    """fake_quant_fp4(...) (lowercase apply form) should also be detected."""
    _setup_fake_repo(tmp_path)
    bad = tmp_path / "experiments" / "do_fp4.py"
    bad.write_text(textwrap.dedent('''\
        from tac.fp4_quantize import fake_quant_fp4
        def quantize(x):
            return fake_quant_fp4(x)
    '''))
    violations = check_fp4_production_paths_disclose_hardware(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1
    assert "do_fp4.py" in violations[0]


def test_import_only_no_call_not_flagged(tmp_path: Path) -> None:
    """Importing FakeQuantFP4 without calling it (e.g., re-exports) → no flag."""
    _setup_fake_repo(tmp_path)
    good = tmp_path / "src" / "tac" / "reexport.py"
    good.write_text(textwrap.dedent('''\
        from tac.quantization import FakeQuantFP4  # re-export
        __all__ = ["FakeQuantFP4"]
    '''))
    assert check_fp4_production_paths_disclose_hardware(
        repo_root=tmp_path, strict=False, verbose=False,
    ) == []


def test_strict_raises_metabugviolation(tmp_path: Path) -> None:
    _setup_fake_repo(tmp_path)
    bad = tmp_path / "experiments" / "do_fp4.py"
    bad.write_text(textwrap.dedent('''\
        from tac.quantization import FakeQuantFP4
        FakeQuantFP4(bits=4)
    '''))
    with pytest.raises(MetaBugViolation, match="FP4 PRODUCTION PATHS"):
        check_fp4_production_paths_disclose_hardware(
            repo_root=tmp_path, strict=True, verbose=False,
        )
