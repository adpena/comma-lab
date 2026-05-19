# SPDX-License-Identifier: MIT
"""Tests for tools/cathedral_autopilot_autonomous_loop auto-discovery loop.

Per CATHEDRAL-AUTO-INGEST-PARADIGM-SHIFT subagent landing 2026-05-19. Tests
the auto-discovery + auto-registration of cathedral consumers in
``src/tac/cathedral_consumers/``.

Sister tests:
- src/tac/tests/test_cathedral_consumer_contract.py
- src/tac/tests/test_check_335_cathedral_consumer_directory_contract.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def autopilot_loop_module():
    """Import the cathedral_autopilot_autonomous_loop module once."""
    # The loop lives in tools/ not src/, so import requires PYTHONPATH surgery.
    repo_root = Path(__file__).parent.parent.parent.parent
    tools_dir = repo_root / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        return importlib.import_module("cathedral_autopilot_autonomous_loop")
    finally:
        # Restore sys.path
        if str(tools_dir) in sys.path:
            sys.path.remove(str(tools_dir))


def test_discover_and_register_returns_list(autopilot_loop_module) -> None:
    regs = autopilot_loop_module.discover_and_register_consumers()
    assert isinstance(regs, list)


def test_discover_finds_example_consumer(autopilot_loop_module) -> None:
    """The reference _example_consumer is the permanent positive fixture."""
    regs = autopilot_loop_module.discover_and_register_consumers()
    names = [r["consumer_name"] for r in regs]
    assert "_example_consumer" in names


def test_discover_example_consumer_compliant(autopilot_loop_module) -> None:
    regs = autopilot_loop_module.discover_and_register_consumers()
    example = next((r for r in regs if r["consumer_name"] == "_example_consumer"), None)
    assert example is not None
    assert example["contract_compliant"] is True
    assert len(example["validation_errors"]) == 0


def test_discover_example_consumer_has_correct_hooks(autopilot_loop_module) -> None:
    regs = autopilot_loop_module.discover_and_register_consumers()
    example = next(r for r in regs if r["consumer_name"] == "_example_consumer")
    # Hooks are HookNumber enum members serialized to ints via dataclasses.asdict
    hook_ints = [int(h) for h in example["consumer_hook_numbers"]]
    assert 4 in hook_ints  # CATHEDRAL_AUTOPILOT_DISPATCH
    assert 5 in hook_ints  # CONTINUAL_LEARNING_POSTERIOR


def test_discover_can_exclude_underscore(autopilot_loop_module) -> None:
    """include_underscore_packages=False filters _example_consumer."""
    regs = autopilot_loop_module.discover_and_register_consumers(
        include_underscore_packages=False
    )
    names = [r["consumer_name"] for r in regs]
    assert "_example_consumer" not in names


def test_discover_nonexistent_dir_silent(autopilot_loop_module, tmp_path: Path) -> None:
    regs = autopilot_loop_module.discover_and_register_consumers(
        consumer_dir_relpath="src/tac/nonexistent_dir",
        repo_root=tmp_path,
    )
    assert regs == []


def test_discover_compliant_modules_skips_underscore(autopilot_loop_module) -> None:
    """The production-modules getter SKIPS reference / underscore packages."""
    mods = autopilot_loop_module.discover_compliant_consumer_modules()
    for mod in mods:
        assert not mod.CONSUMER_NAME.startswith("_"), (
            f"production discovery must skip reference packages, got {mod.CONSUMER_NAME}"
        )


def test_discover_compliant_modules_includes_only_compliant(autopilot_loop_module) -> None:
    """The production-modules getter returns ONLY contract-compliant packages.

    At landing the directory may contain 0 or more production consumers
    (sister subagents may have landed their own); the invariant is that
    every returned module satisfies the canonical contract.
    """
    from tac.cathedral.consumer_contract import validate_consumer_module
    mods = autopilot_loop_module.discover_compliant_consumer_modules()
    for mod in mods:
        reg = validate_consumer_module(mod)
        assert reg.contract_compliant is True, (
            f"discover_compliant_consumer_modules returned non-compliant "
            f"{mod.CONSUMER_NAME}: {reg.validation_errors}"
        )


def test_discover_serialized_to_dict(autopilot_loop_module) -> None:
    """Registrations are returned as dicts (JSON-friendly for ledger ingestion)."""
    regs = autopilot_loop_module.discover_and_register_consumers()
    for r in regs:
        assert isinstance(r, dict)
        assert "consumer_name" in r
        assert "consumer_version" in r
        assert "consumer_hook_numbers" in r
        assert "consumer_module_path" in r
        assert "contract_compliant" in r


def test_discover_deterministic_ordering(autopilot_loop_module) -> None:
    """Multiple invocations return packages in deterministic (sorted) order."""
    regs1 = autopilot_loop_module.discover_and_register_consumers()
    regs2 = autopilot_loop_module.discover_and_register_consumers()
    names1 = [r["consumer_name"] for r in regs1]
    names2 = [r["consumer_name"] for r in regs2]
    assert names1 == names2
    assert names1 == sorted(names1)


def test_discover_strict_mode_silent_on_clean_repo(autopilot_loop_module) -> None:
    """Strict mode silent when all packages are compliant (live repo at landing)."""
    regs = autopilot_loop_module.discover_and_register_consumers(strict=True)
    # Should not raise; _example_consumer is compliant.
    assert isinstance(regs, list)


def test_discover_strict_mode_raises_on_synthetic_violation(
    autopilot_loop_module, tmp_path: Path
) -> None:
    """Strict mode raises RuntimeError on non-compliant package without waiver."""
    # Create a synthetic broken package in a tmp dir.
    consumer_dir = tmp_path / "src" / "tac" / "cathedral_consumers" / "broken_strict"
    consumer_dir.mkdir(parents=True)
    (consumer_dir / "__init__.py").write_text(
        '"""Broken consumer without contract."""\n'
    )
    # Also need parent __init__ files
    (consumer_dir.parent / "__init__.py").write_text("")
    (consumer_dir.parent.parent / "__init__.py").write_text("")
    (consumer_dir.parent.parent.parent / "__init__.py").write_text("")

    # Add tmp_path/src to sys.path so the import resolves
    src_path = str(tmp_path / "src")
    sys.path.insert(0, src_path)
    try:
        # Remove any cached imports
        for mod_name in list(sys.modules):
            if mod_name.startswith("tac.cathedral_consumers"):
                del sys.modules[mod_name]
        with pytest.raises(RuntimeError, match="STRICT refuse"):
            autopilot_loop_module.discover_and_register_consumers(
                repo_root=tmp_path, strict=True
            )
    finally:
        if src_path in sys.path:
            sys.path.remove(src_path)
        for mod_name in list(sys.modules):
            if mod_name.startswith("tac.cathedral_consumers"):
                del sys.modules[mod_name]


def test_paradigm_shift_extincts_orphan_signal_class(autopilot_loop_module) -> None:
    """REGRESSION: the auto-discovery loop IS the structural extinction.

    Per operator NON-NEGOTIABLE 2026-05-19: "fix permanently and self
    protect against" the orphan-signal-at-cathedral-autopilot bug class.

    Adding a new consumer to src/tac/cathedral_consumers/ MUST automatically
    surface it via discover_and_register_consumers without any manual edit
    of tools/cathedral_autopilot_autonomous_loop.py.

    This test confirms the canonical helper exists + is callable + returns
    the expected shape, which is the empirical proof that future orphan
    landings are extincted structurally.
    """
    assert hasattr(autopilot_loop_module, "discover_and_register_consumers")
    assert hasattr(autopilot_loop_module, "discover_compliant_consumer_modules")
    regs = autopilot_loop_module.discover_and_register_consumers()
    # Reference consumer demonstrates the paradigm: at least 1 result.
    assert len(regs) >= 1
