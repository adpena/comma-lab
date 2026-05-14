# SPDX-License-Identifier: MIT
"""Tests for codex round-5 catalog #141 — cross-module state-helper calls must thread paths.

Bug class (codex round-5 HIGH 3, 2026-05-09): ``azure_dispatch.py``
defined its own ``ACTIVE_VMS_PATH`` constant AND imported
``register_active_vm_record`` / ``unregister_active_vm_by_name`` /
``load_active_vms`` from ``tac.deploy.azure.active_vms_state`` (which has
its OWN ``ACTIVE_VMS_PATH``). The calls did NOT thread
``path=ACTIVE_VMS_PATH`` so tests that monkeypatched
``azd.ACTIVE_VMS_PATH`` silently observed the helper's canonical path.

The fix:

1. ``azure_dispatch.py`` adds ``ACTIVE_VMS_LOCK = ACTIVE_VMS_PATH.with_suffix(...)``
   sibling constant.
2. ``_load_active_vms`` calls ``load_active_vms(path=ACTIVE_VMS_PATH)``.
3. ``register_active_vm`` calls
   ``register_active_vm_record(record, path=ACTIVE_VMS_PATH, lock_path=ACTIVE_VMS_LOCK)``.
4. ``unregister_active_vm`` calls
   ``unregister_active_vm_by_name(vm_name, path=ACTIVE_VMS_PATH, lock_path=ACTIVE_VMS_LOCK)``.
5. STRICT preflight gate #141 refuses any future cross-module helper call
   that fails to thread `path=` when both modules define a canonical
   tracker constant.

Memory: feedback_codex_round5_findings_fix_with_self_protection_landed_20260509.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_state_helper_paths_explicit,
)


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "tools").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    return tmp_path


# ── Live-repo sanity ────────────────────────────────────────────────────


def test_141_live_repo_clean():
    """Live-repo: catalog #141 lands at 0 violations after fix."""
    v = check_state_helper_paths_explicit(strict=False, verbose=False)
    assert v == [], (
        f"Catalog #141 landed with {len(v)} violations:\n"
        + "\n".join(v[:3])
    )


# ── Catch the bug class ────────────────────────────────────────────────


def test_141_catches_unthreaded_helper_call(tmp_path):
    """A module that defines its own ACTIVE_VMS_PATH and calls
    register_active_vm_record() WITHOUT path= MUST be flagged."""
    root = _make_repo(tmp_path)
    target = root / "src" / "tac" / "fake_dispatch.py"
    target.write_text(
        "from pathlib import Path\n"
        "ACTIVE_VMS_PATH = Path('.omx/state/azure_active_vms.json')\n"
        "ACTIVE_VMS_LOCK = ACTIVE_VMS_PATH.with_suffix('.lock')\n"
        "from tac.deploy.azure.active_vms_state import register_active_vm_record\n"
        "\n"
        "def register_vm(record):\n"
        "    register_active_vm_record(record)\n"
    )
    v = check_state_helper_paths_explicit(
        repo_root=root, strict=False, verbose=False
    )
    assert any("Check 141" in x and "register_active_vm_record" in x for x in v), (
        f"expected Check 141 hit on register_active_vm_record; got: {v}"
    )


def test_141_accepts_threaded_helper_call(tmp_path):
    """A module that threads `path=ACTIVE_VMS_PATH` is OK."""
    root = _make_repo(tmp_path)
    target = root / "src" / "tac" / "fake_dispatch.py"
    target.write_text(
        "from pathlib import Path\n"
        "ACTIVE_VMS_PATH = Path('.omx/state/azure_active_vms.json')\n"
        "ACTIVE_VMS_LOCK = ACTIVE_VMS_PATH.with_suffix('.lock')\n"
        "from tac.deploy.azure.active_vms_state import register_active_vm_record\n"
        "\n"
        "def register_vm(record):\n"
        "    register_active_vm_record(record, path=ACTIVE_VMS_PATH, lock_path=ACTIVE_VMS_LOCK)\n"
    )
    v = check_state_helper_paths_explicit(
        repo_root=root, strict=False, verbose=False
    )
    assert v == [], f"expected 0 violations; got: {v}"


def test_141_respects_same_line_waiver(tmp_path):
    """Same-line `# STATE_HELPER_PATH_OK:...` waives the gate."""
    root = _make_repo(tmp_path)
    target = root / "src" / "tac" / "fake_dispatch.py"
    target.write_text(
        "from pathlib import Path\n"
        "ACTIVE_VMS_PATH = Path('.omx/state/azure_active_vms.json')\n"
        "from tac.deploy.azure.active_vms_state import register_active_vm_record\n"
        "\n"
        "def register_vm(record):\n"
        "    register_active_vm_record(record)  # STATE_HELPER_PATH_OK:deliberate-canonical-path\n"
    )
    v = check_state_helper_paths_explicit(
        repo_root=root, strict=False, verbose=False
    )
    assert v == [], f"expected 0 violations after waiver; got: {v}"


def test_141_strict_mode_raises(tmp_path):
    """strict=True raises PreflightError when violations are present."""
    root = _make_repo(tmp_path)
    target = root / "src" / "tac" / "fake_dispatch.py"
    target.write_text(
        "from pathlib import Path\n"
        "ACTIVE_VMS_PATH = Path('.omx/state/azure_active_vms.json')\n"
        "from tac.deploy.azure.active_vms_state import register_active_vm_record\n"
        "\n"
        "def register_vm(record):\n"
        "    register_active_vm_record(record)\n"
    )
    with pytest.raises(PreflightError, match="Check 141|state-helper-paths"):
        check_state_helper_paths_explicit(
            repo_root=root, strict=True, verbose=False
        )


def test_141_no_tracker_constant_out_of_scope(tmp_path):
    """Modules that don't define a tracker constant are out-of-scope —
    they intentionally use the helper's canonical path."""
    root = _make_repo(tmp_path)
    target = root / "src" / "tac" / "fake_caller.py"
    target.write_text(
        "from tac.deploy.azure.active_vms_state import register_active_vm_record\n"
        "\n"
        "def register_vm(record):\n"
        "    register_active_vm_record(record)\n"
    )
    v = check_state_helper_paths_explicit(
        repo_root=root, strict=False, verbose=False
    )
    assert v == [], (
        f"modules without tracker constants should be out of scope; got: {v}"
    )


def test_141_lightning_active_jobs_caller_also_caught(tmp_path):
    """Sister bug class: lightning active_jobs_state callers also need path threading."""
    root = _make_repo(tmp_path)
    target = root / "src" / "tac" / "fake_lightning.py"
    target.write_text(
        "from pathlib import Path\n"
        "ACTIVE_JOBS_PATH = Path('.omx/state/lightning_active_jobs.json')\n"
        "from tac.deploy.lightning.active_jobs_state import register_job\n"
        "\n"
        "def register(record):\n"
        "    register_job(record)\n"
    )
    v = check_state_helper_paths_explicit(
        repo_root=root, strict=False, verbose=False
    )
    assert any(
        "Check 141" in x and "register_job" in x for x in v
    ), f"expected Check 141 hit on register_job; got: {v}"


def test_141_test_files_excluded(tmp_path):
    """Test files (test_*.py / under tests/) must not be scanned."""
    root = _make_repo(tmp_path)
    tests_dir = root / "src" / "tac" / "tests"
    tests_dir.mkdir(parents=True)
    target = tests_dir / "test_fake.py"
    target.write_text(
        "from pathlib import Path\n"
        "ACTIVE_VMS_PATH = Path('.omx/state/azure_active_vms.json')\n"
        "from tac.deploy.azure.active_vms_state import register_active_vm_record\n"
        "\n"
        "def test_register():\n"
        "    register_active_vm_record({})\n"
    )
    v = check_state_helper_paths_explicit(
        repo_root=root, strict=False, verbose=False
    )
    assert v == [], f"test files should be excluded; got: {v}"


def test_141_public_pr_intake_excluded(tmp_path):
    """Public-PR intake clones must be out-of-scope."""
    root = _make_repo(tmp_path)
    intake = (
        root
        / "experiments"
        / "results"
        / "public_pr107_intake_20260508_codex"
        / "source"
    )
    intake.mkdir(parents=True)
    target = intake / "fake_caller.py"
    target.write_text(
        "from pathlib import Path\n"
        "ACTIVE_VMS_PATH = Path('.omx/state/azure_active_vms.json')\n"
        "from tac.deploy.azure.active_vms_state import register_active_vm_record\n"
        "\n"
        "def register(record):\n"
        "    register_active_vm_record(record)\n"
    )
    v = check_state_helper_paths_explicit(
        repo_root=root, strict=False, verbose=False
    )
    assert v == [], f"public PR intake should be excluded; got: {v}"


# ── Behavioural test of the actual azure_dispatch fix ──────────────────


def test_azure_dispatch_threads_path_to_register(tmp_path, monkeypatch):
    """register_active_vm honors monkeypatched ACTIVE_VMS_PATH."""
    from tac.deploy.azure import azure_dispatch as azd

    custom_path = tmp_path / "custom_active_vms.json"
    custom_lock = tmp_path / "custom_active_vms.lock"
    monkeypatch.setattr(azd, "ACTIVE_VMS_PATH", custom_path)
    monkeypatch.setattr(azd, "ACTIVE_VMS_LOCK", custom_lock)

    handle = azd.AzureVMHandle(
        vm_name="test-vm",
        resource_group="test-rg",
        region="eastus",
        public_ip="10.0.0.1",
        username="azureuser",
    )
    azd.register_active_vm(handle, "test-label")

    assert custom_path.exists(), (
        f"register_active_vm should have written to custom_path; "
        f"got: {list(tmp_path.iterdir())}"
    )


def test_azure_dispatch_threads_path_to_unregister(tmp_path, monkeypatch):
    """unregister_active_vm honors monkeypatched ACTIVE_VMS_PATH."""
    from tac.deploy.azure import azure_dispatch as azd

    custom_path = tmp_path / "custom_active_vms.json"
    custom_lock = tmp_path / "custom_active_vms.lock"
    monkeypatch.setattr(azd, "ACTIVE_VMS_PATH", custom_path)
    monkeypatch.setattr(azd, "ACTIVE_VMS_LOCK", custom_lock)

    custom_path.write_text(
        '[{"vm_name": "test-vm", "label": "test"}, {"vm_name": "other-vm", "label": "keep"}]'
    )

    azd.unregister_active_vm("test-vm")

    import json
    rows = json.loads(custom_path.read_text())
    assert {r["vm_name"] for r in rows} == {"other-vm"}, (
        f"only test-vm should be removed; got: {rows}"
    )


def test_azure_dispatch_threads_path_to_load(tmp_path, monkeypatch):
    """_load_active_vms honors monkeypatched ACTIVE_VMS_PATH."""
    from tac.deploy.azure import azure_dispatch as azd

    custom_path = tmp_path / "custom_active_vms.json"
    custom_lock = tmp_path / "custom_active_vms.lock"
    monkeypatch.setattr(azd, "ACTIVE_VMS_PATH", custom_path)
    monkeypatch.setattr(azd, "ACTIVE_VMS_LOCK", custom_lock)

    custom_path.write_text('[{"vm_name": "test-vm"}]')

    rows = azd._load_active_vms()
    assert rows == [{"vm_name": "test-vm"}]
