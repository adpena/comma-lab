# SPDX-License-Identifier: MIT
"""Tests for tac.deploy.azure.azure_dispatch — Azure dispatch wiring.

Strategy: the actual ``az`` CLI is mocked via ``subprocess.run`` patching;
no test ever hits real Azure infrastructure. We verify:
1. Pricing table is sane (no zero/negative rates, all SKUs in ALLOWED set)
2. AzureVMSpec name sanitization (Azure 1-64 alnum+hyphen rule)
3. ensure_azure_logged_in raises the right errors when az is missing
   or no account is active
4. provision_spot_vm dry-run does not call az
5. SSH command construction is well-formed
6. Active VM tracker round-trip (register + unregister)
7. estimate_cost rounds correctly for known SKUs
8. remaining_budget_usd respects the $200 cap
9. run_lane builds Pattern A nohup detach wrapper correctly
"""
from __future__ import annotations

import json
import subprocess

import pytest

from tac.deploy.azure import azure_dispatch as azd


# ── 1. Pricing table sanity ──────────────────────────────────────────────────


def test_pricing_table_is_sane():
    """Every SKU has positive on-demand and spot rates, with spot < on-demand."""
    assert azd.AZURE_GPU_PRICING, "Pricing table empty"
    for sku, info in azd.AZURE_GPU_PRICING.items():
        assert info["on_demand_usd_per_hour"] > 0, f"{sku} non-positive on-demand"
        assert info["spot_usd_per_hour"] > 0, f"{sku} non-positive spot"
        assert info["spot_usd_per_hour"] < info["on_demand_usd_per_hour"], (
            f"{sku} spot ({info['spot_usd_per_hour']}) "
            f">= on-demand ({info['on_demand_usd_per_hour']})"
        )
        assert info["vram_gb"] >= 16
        assert info["gpu"] in {"V100", "A100", "H100"}


def test_default_gpu_is_in_pricing_table():
    assert azd.DEFAULT_GPU_TYPE in azd.AZURE_GPU_PRICING


def test_hard_cap_matches_claude_md():
    """CLAUDE.md says $200 free credits — assert constant agrees."""
    assert azd.AZURE_HARD_CAP_USD == 200.00


# ── 2. AzureVMSpec name sanitization ─────────────────────────────────────────


def test_vm_name_sanitizes_special_chars():
    spec = azd.AzureVMSpec(label="lane_pd_v2/test 1")
    # Azure allows only alphanumeric + hyphen
    assert all(c.isalnum() or c == "-" for c in spec.vm_name), spec.vm_name
    assert len(spec.vm_name) <= 64


def test_vm_name_truncates_long_labels():
    long_label = "a" * 200
    spec = azd.AzureVMSpec(label=long_label)
    assert len(spec.vm_name) <= 64


def test_vm_name_falls_back_when_label_strips_to_empty():
    spec = azd.AzureVMSpec(label="!!!___///")
    assert spec.vm_name, "VM name must be non-empty"
    assert spec.vm_name == "pact-vm"


def test_vm_spec_estimated_cost_per_hour_uses_spot_when_spot():
    spec = azd.AzureVMSpec(label="x", gpu_type="Standard_NC6s_v3", spot=True)
    expected = azd.AZURE_GPU_PRICING["Standard_NC6s_v3"]["spot_usd_per_hour"]
    assert spec.estimated_cost_per_hour == expected


def test_vm_spec_estimated_cost_per_hour_uses_on_demand_when_not_spot():
    spec = azd.AzureVMSpec(label="x", gpu_type="Standard_NC6s_v3", spot=False)
    expected = azd.AZURE_GPU_PRICING["Standard_NC6s_v3"]["on_demand_usd_per_hour"]
    assert spec.estimated_cost_per_hour == expected


# ── 3. Pre-flight error conditions ───────────────────────────────────────────


def test_ensure_logged_in_raises_when_az_missing(monkeypatch):
    """When `which az` returns rc=1, AzureCLIMissing fires (not silent)."""
    def fake_run(cmd, **kwargs):
        # `which az` returns nothing
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
    monkeypatch.setattr(azd.subprocess, "run", fake_run)
    with pytest.raises(azd.AzureCLIMissing):
        azd.ensure_azure_logged_in()


def test_ensure_logged_in_raises_when_no_active_subscription(monkeypatch):
    """When `az account show` returns rc=1, AzureNotLoggedIn fires."""
    calls = {"n": 0}
    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        if "which" in cmd and "az" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="/usr/bin/az\n", stderr="")
        # az account show
        return subprocess.CompletedProcess(
            cmd, 1, stdout="",
            stderr="Please run 'az login' to setup account.")
    monkeypatch.setattr(azd.subprocess, "run", fake_run)
    with pytest.raises(azd.AzureNotLoggedIn):
        azd.ensure_azure_logged_in()


def test_ensure_logged_in_returns_account_when_active(monkeypatch):
    """Happy path: returns parsed account dict."""
    def fake_run(cmd, **kwargs):
        if "which" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="/usr/bin/az\n", stderr="")
        return subprocess.CompletedProcess(
            cmd, 0,
            stdout=json.dumps({"id": "sub-id", "name": "MySub"}),
            stderr="",
        )
    monkeypatch.setattr(azd.subprocess, "run", fake_run)
    account = azd.ensure_azure_logged_in()
    assert account["id"] == "sub-id"
    assert account["name"] == "MySub"


# ── 4. Dry-run path doesn't side-effect ──────────────────────────────────────


def test_provision_dry_run_does_not_call_az(monkeypatch, capsys):
    # Mock ensure_azure_logged_in to bypass auth check
    monkeypatch.setattr(azd, "ensure_azure_logged_in", lambda: {"id": "x", "name": "x"})
    az_calls = []

    def fake_run_az(args, **kwargs):
        az_calls.append(args)
        return 0, "{}", ""
    monkeypatch.setattr(azd, "_run_az", fake_run_az)

    spec = azd.AzureVMSpec(label="dry-test")
    handle = azd.provision_spot_vm(spec, dry_run=True)
    assert handle.public_ip == "0.0.0.0"
    assert handle.vm_name == "dry-test"
    # Zero side-effecting az calls in dry-run
    assert az_calls == []
    captured = capsys.readouterr()
    assert "[dry_run]" in captured.out
    assert "az vm create" in captured.out


def test_provision_dry_run_includes_spot_flags(monkeypatch, capsys):
    monkeypatch.setattr(azd, "ensure_azure_logged_in", lambda: {"id": "x", "name": "x"})
    monkeypatch.setattr(azd, "_run_az", lambda *a, **kw: (0, "{}", ""))

    spec = azd.AzureVMSpec(label="spot-test", spot=True, spot_max_price_usd=0.50)
    azd.provision_spot_vm(spec, dry_run=True)
    captured = capsys.readouterr()
    assert "--priority" in captured.out
    assert "Spot" in captured.out
    assert "0.5000" in captured.out  # spot max price formatted to 4 decimals


def test_provision_unknown_gpu_raises(monkeypatch):
    monkeypatch.setattr(azd, "ensure_azure_logged_in", lambda: {"id": "x", "name": "x"})
    spec = azd.AzureVMSpec(label="x", gpu_type="Standard_NOT_A_REAL_SKU")
    with pytest.raises(ValueError):
        azd.provision_spot_vm(spec, dry_run=True)


# ── 5. SSH command construction ──────────────────────────────────────────────


def test_ssh_in_uses_strict_host_key_no(monkeypatch):
    captured_args: list = []

    def fake_run(args, **kwargs):
        captured_args.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="hello\n", stderr="")

    monkeypatch.setattr(azd.subprocess, "run", fake_run)
    handle = azd.AzureVMHandle(
        vm_name="x", resource_group="rg", region="eastus",
        public_ip="1.2.3.4", username="azureuser",
    )
    rc, out, err = azd.ssh_in(handle, "echo hello")
    assert rc == 0
    assert out == "hello\n"
    assert "ssh" in captured_args[0]
    assert "StrictHostKeyChecking=no" in " ".join(captured_args[0])
    assert "azureuser@1.2.3.4" in captured_args[0]


# ── 6. Active VM tracker round-trip ──────────────────────────────────────────


def test_active_vm_tracker_round_trip(tmp_path, monkeypatch):
    """register_active_vm + unregister_active_vm round-trip via JSON file."""
    fake_path = tmp_path / "azure_active_vms.json"
    monkeypatch.setattr(azd, "ACTIVE_VMS_PATH", fake_path)

    handle = azd.AzureVMHandle(
        vm_name="vm-a", resource_group="rg-a", region="eastus",
        public_ip="1.2.3.4", username="azureuser",
        gpu_type="Standard_NC6s_v3", spot=True,
        estimated_cost_per_hour=0.50,
    )
    azd.register_active_vm(handle, label="lane-test")
    assert fake_path.exists()
    rows = json.loads(fake_path.read_text())
    assert len(rows) == 1
    assert rows[0]["vm_name"] == "vm-a"
    assert rows[0]["label"] == "lane-test"
    assert rows[0]["gpu_type"] == "Standard_NC6s_v3"
    assert "provisioned_at" in rows[0]

    # Register a second VM
    h2 = azd.AzureVMHandle(
        vm_name="vm-b", resource_group="rg-b", region="eastus",
        public_ip="5.6.7.8", username="azureuser",
    )
    azd.register_active_vm(h2, label="lane-test-2")
    rows = json.loads(fake_path.read_text())
    assert {r["vm_name"] for r in rows} == {"vm-a", "vm-b"}

    # Unregister vm-a; vm-b should remain
    azd.unregister_active_vm("vm-a")
    rows = json.loads(fake_path.read_text())
    assert len(rows) == 1
    assert rows[0]["vm_name"] == "vm-b"


def test_load_active_vms_returns_empty_when_missing(tmp_path, monkeypatch):
    fake_path = tmp_path / "missing.json"
    monkeypatch.setattr(azd, "ACTIVE_VMS_PATH", fake_path)
    assert azd._load_active_vms() == []


# ── 7. Cost estimation ───────────────────────────────────────────────────────


def test_estimate_cost_v100_spot_one_hour():
    # V100 spot ≈ $0.50/hr
    cost = azd.estimate_cost("Standard_NC6s_v3", hours=1.0, spot=True)
    assert cost == pytest.approx(0.50, rel=1e-6)


def test_estimate_cost_a100_on_demand_two_hours():
    cost = azd.estimate_cost("Standard_NC24ads_A100_v4", hours=2.0, spot=False)
    assert cost == pytest.approx(7.34, rel=1e-6)  # 3.67 * 2


def test_estimate_cost_unknown_sku_raises():
    with pytest.raises(ValueError):
        azd.estimate_cost("Standard_FAKE", hours=1.0)


# ── 8. Remaining budget against $200 cap ─────────────────────────────────────


def test_remaining_budget_full():
    assert azd.remaining_budget_usd(0.0) == 200.00


def test_remaining_budget_partial():
    assert azd.remaining_budget_usd(50.0) == 150.00


def test_remaining_budget_overspent_floors_to_zero():
    assert azd.remaining_budget_usd(250.0) == 0.0


# ── 9. run_lane builds Pattern A nohup detach wrapper ────────────────────────


def test_run_lane_emits_pattern_a_detach_wrapper(monkeypatch):
    """The remote command must use nohup + bash -c + disown + tee log."""
    captured_commands: list[str] = []

    def fake_ssh_in(handle, command, *, timeout=120.0):
        captured_commands.append(command)
        return 0, "PID=12345\n", ""

    monkeypatch.setattr(azd, "ssh_in", fake_ssh_in)
    handle = azd.AzureVMHandle(
        vm_name="x", resource_group="rg", region="eastus",
        public_ip="1.2.3.4", username="azureuser",
    )
    rc, log_path = azd.run_lane(
        handle,
        lane_script_path="/home/azureuser/pact/scripts/remote_lane_X.sh",
        env_vars={"PYTHONPATH": "src:upstream:."},
        log_path="/tmp/lane_run.log",
    )
    assert rc == 0
    assert log_path == "/tmp/lane_run.log"
    cmd = captured_commands[0]
    # Pattern A markers
    assert "nohup" in cmd
    assert "disown" in cmd
    assert "/dev/null" in cmd
    assert "/tmp/lane_run.log" in cmd
    assert "export PYTHONPATH" in cmd


def test_run_lane_returns_nonzero_when_ssh_fails(monkeypatch):
    def fake_ssh_in(handle, command, *, timeout=120.0):
        return 255, "", "Connection refused"
    monkeypatch.setattr(azd, "ssh_in", fake_ssh_in)
    handle = azd.AzureVMHandle(
        vm_name="x", resource_group="rg", region="eastus",
        public_ip="1.2.3.4", username="azureuser",
    )
    rc, msg = azd.run_lane(handle, "/some/path.sh")
    assert rc == 255
    assert "ssh launch failed" in msg
