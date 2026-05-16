# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "launch_lane_azure.py"


def load_module():
    spec = importlib.util.spec_from_file_location("launch_lane_azure", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_no_dry_run_refuses_scaffold_contract_before_azure_login(monkeypatch, capsys):
    mod = load_module()

    def fail_if_called():
        raise AssertionError("Azure login check must not run for a refused scaffold contract")

    monkeypatch.setattr(mod.azd, "ensure_azure_logged_in", fail_if_called)

    rc = mod.main(
        [
            "--lane-script",
            "scripts/launch_lane_azure.py",
            "--label",
            "unit_test_lane",
            "--no-dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "Azure --no-dry-run refused by provider contract" in captured.err
    assert "execution_flag=None" in captured.err
    assert "exact_cuda_eval_supported=False" in captured.err
    assert "status=scaffold" in captured.err
