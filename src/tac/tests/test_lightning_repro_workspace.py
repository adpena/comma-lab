"""Tests for deterministic Lightning Studio workspace preparation."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "lightning_repro_workspace.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "lightning_repro_workspace_under_test",
        str(SCRIPT),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_files_excludes_result_trees_unless_explicit_artifact(tmp_path: Path) -> None:
    mod = _load_module()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "pkg.py").write_text("x = 1\n")
    (tmp_path / "experiments" / "results" / "stale").mkdir(parents=True)
    stale = tmp_path / "experiments" / "results" / "stale" / "contest_auth_eval.json"
    stale.write_text('{"score": 999}\n')
    (tmp_path / "experiments" / "tool.py").write_text("print('tool')\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='tmp'\n")

    source_only = mod.collect_files(
        ["pyproject.toml", "src", "experiments"],
        [],
        repo_root=tmp_path,
    )
    source_paths = {item["path"] for item in source_only}
    assert "experiments/tool.py" in source_paths
    assert "experiments/results/stale/contest_auth_eval.json" not in source_paths

    with_artifact = mod.collect_files(
        ["pyproject.toml", "src", "experiments"],
        ["experiments/results/stale/contest_auth_eval.json"],
        repo_root=tmp_path,
    )
    by_path = {item["path"]: item for item in with_artifact}
    assert by_path["experiments/results/stale/contest_auth_eval.json"]["role"] == "artifact"


def test_build_manifest_filters_excluded_dirty_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_module()

    def fake_capture(cmd: list[str], *, cwd: Path = mod.REPO_ROOT) -> str | None:
        if cmd[:3] == ["git", "status", "--short"]:
            return "\n".join(
                [
                    " M src/tac/live.py",
                    "?? .recovery_quarantine_20260505T004735Z/private.py",
                    "?? .omx/state/provider.json",
                    "?? experiments/results/run/archive.zip",
                ]
            )
        if cmd == ["git", "rev-parse", "HEAD"]:
            return "abc123"
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return "main"
        raise AssertionError(f"unexpected capture command: {cmd}")

    monkeypatch.setattr(mod, "_capture", fake_capture)
    args = mod.parse_args(
        [
            "--remote",
            "lightning-pact",
            "--run-id",
            "unit",
            "--source",
            "src",
            "--dry-run",
        ]
    )
    manifest = mod.build_manifest(args, [{"path": "src/tac/live.py", "role": "source", "bytes": 1, "sha256": "a"}])

    assert manifest["git"]["status_short"] == [" M src/tac/live.py"]
    assert manifest["git"]["status_short_excluded_private"] == [
        "?? .recovery_quarantine_20260505T004735Z/private.py",
        "?? .omx/state/provider.json",
        "?? experiments/results/run/archive.zip",
    ]


def test_collect_files_excludes_generated_binary_source_payloads(tmp_path: Path) -> None:
    mod = _load_module()
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "tool.py").write_text("print('tool')\n")
    (tmp_path / "experiments" / "raft_flow.pt").write_bytes(b"large generated tensor")
    (tmp_path / "submissions" / "robust_current" / "eval_runs").mkdir(parents=True)
    (tmp_path / "submissions" / "robust_current" / "eval_runs" / "0.raw").write_bytes(b"raw")
    (tmp_path / "submissions" / "robust_current" / "archive.zip").write_bytes(b"zip")

    source_only = mod.collect_files(
        ["experiments", "submissions"],
        [],
        repo_root=tmp_path,
    )
    source_paths = {item["path"] for item in source_only}
    assert "experiments/tool.py" in source_paths
    assert "experiments/raft_flow.pt" not in source_paths
    assert "submissions/robust_current/eval_runs/0.raw" not in source_paths
    assert "submissions/robust_current/archive.zip" not in source_paths

    with_artifacts = mod.collect_files(
        ["experiments", "submissions"],
        ["experiments/raft_flow.pt", "submissions/robust_current/archive.zip"],
        repo_root=tmp_path,
    )
    by_path = {item["path"]: item for item in with_artifacts}
    assert by_path["experiments/raft_flow.pt"]["role"] == "artifact"
    assert by_path["submissions/robust_current/archive.zip"]["role"] == "artifact"


def test_collect_files_excludes_hidden_and_macos_source_paths(tmp_path: Path) -> None:
    mod = _load_module()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "pkg.py").write_text("x = 1\n")
    (tmp_path / "src" / "._pkg.py").write_text("fork\n")
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text("name: ci\n")
    (tmp_path / "__MACOSX").mkdir()
    (tmp_path / "__MACOSX" / "payload").write_text("fork\n")
    (tmp_path / ".DS_Store").write_bytes(b"finder")

    source_only = mod.collect_files(
        ["."],
        [],
        repo_root=tmp_path,
    )
    source_paths = {item["path"] for item in source_only}

    assert "src/pkg.py" in source_paths
    assert "src/._pkg.py" not in source_paths
    assert ".github/workflows/ci.yml" not in source_paths
    assert "__MACOSX/payload" not in source_paths
    assert ".DS_Store" not in source_paths


def test_parse_args_defaults_to_oss_repro_source_set() -> None:
    mod = _load_module()
    args = mod.parse_args([
        "--remote",
        "s_example@ssh.lightning.ai",
        "--dry-run",
    ])
    assert "pyproject.toml" in args.source
    assert "uv.lock" in args.source
    assert "experiments" in args.source
    assert "submissions" not in args.source
    assert "submissions/robust_current/inflate.sh" in args.source
    assert "submissions/robust_current/unpack_renderer_payload.py" in args.source
    assert args.requirements_mode == "uv-sync"


def test_parse_args_default_remote_requires_env_or_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    for name in ("LIGHTNING_SSH_TARGET", "LIGHTNING_REMOTE", "REMOTE", "LIGHTNING_USER"):
        monkeypatch.delenv(name, raising=False)

    args = mod.parse_args(["--dry-run"])

    assert args.remote == ""


def test_parse_args_default_remote_honors_lightning_env(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    monkeypatch.setenv("LIGHTNING_SSH_TARGET", "s_example@ssh.lightning.ai")

    args = mod.parse_args(["--dry-run"])

    assert args.remote == "s_example@ssh.lightning.ai"


def test_parse_args_supports_deterministic_runtime_verification() -> None:
    mod = _load_module()
    args = mod.parse_args([
        "--remote",
        "s_example@ssh.lightning.ai",
        "--requirements-mode",
        "verify-only",
        "--python-bin",
        "/opt/conda/bin/python",
        "--require-cuda",
        "--dry-run",
    ])
    assert args.requirements_mode == "verify-only"
    assert args.python_bin == "/opt/conda/bin/python"
    assert args.require_cuda is True


def test_parse_args_supports_ssh_preflight_diagnostics() -> None:
    mod = _load_module()
    args = mod.parse_args([
        "--remote",
        "lightning-pact",
        "--ssh-check-only",
        "--ssh-diagnostics-out",
        ".omx/state/ssh.json",
        "--ssh-connect-timeout",
        "7",
    ])
    assert args.ssh_check_only is True
    assert args.ssh_diagnostics_out == ".omx/state/ssh.json"
    assert args.ssh_connect_timeout == 7


def test_local_uv_lock_preflight_accepts_current_lock(tmp_path: Path) -> None:
    mod = _load_module()

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert cmd == ["uv", "lock", "--check"]
        assert kwargs["cwd"] == tmp_path
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(cmd, 0, "lock ok\n", "")

    diagnostic = mod.ensure_local_uv_lock_current(cwd=tmp_path, runner=fake_run)

    assert diagnostic["status"] == "ok"
    assert diagnostic["command"] == ["uv", "lock", "--check"]


def test_local_uv_lock_preflight_fails_before_remote_staging(tmp_path: Path) -> None:
    mod = _load_module()

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert cmd == ["uv", "lock", "--check"]
        return subprocess.CompletedProcess(
            cmd,
            1,
            "Resolved 152 packages\n",
            "The lockfile at `uv.lock` needs to be updated\n",
        )

    with pytest.raises(mod.LightningLocalUvLockError) as excinfo:
        mod.ensure_local_uv_lock_current(cwd=tmp_path, runner=fake_run)

    assert "local uv.lock is stale" in str(excinfo.value)
    assert excinfo.value.diagnostic["status"] == "fail"
    assert excinfo.value.diagnostic["returncode"] == 1


def test_staging_transfer_commands_reuse_noninteractive_ssh_policy(tmp_path: Path) -> None:
    mod = _load_module()
    file_list = tmp_path / "files.txt"
    rsync_cmd = mod._rsync_command(
        file_list,
        "lightning-pact",
        "/remote/pact",
        connect_timeout=7,
    )
    ssh_cmd = mod._ssh_command("ssh", "lightning-pact", "true", connect_timeout=7)
    scp_cmd = mod._scp_command("scp", tmp_path / "manifest.json", "lightning-pact:/remote/manifest.json", connect_timeout=7)

    assert "BatchMode=yes" in ssh_cmd
    assert "PasswordAuthentication=no" in ssh_cmd
    assert "KbdInteractiveAuthentication=no" in ssh_cmd
    assert "ServerAliveInterval=15" in ssh_cmd
    assert "ServerAliveCountMax=4" in ssh_cmd
    assert "ConnectionAttempts=3" in ssh_cmd
    assert "ConnectTimeout=7" in ssh_cmd
    assert "BatchMode=yes" in scp_cmd
    assert "ConnectTimeout=7" in scp_cmd
    assert "-e" in rsync_cmd
    transport = rsync_cmd[rsync_cmd.index("-e") + 1]
    assert "BatchMode=yes" in transport
    assert "PasswordAuthentication=no" in transport
    assert "KbdInteractiveAuthentication=no" in transport
    assert "ServerAliveInterval=15" in transport
    assert "ServerAliveCountMax=4" in transport
    assert "ConnectionAttempts=3" in transport
    assert "ConnectTimeout=7" in transport


def test_remote_runtime_probe_command_can_require_cuda() -> None:
    mod = _load_module()
    command = mod._remote_runtime_probe_command(
        "/teamspace/studios/this_studio/pact",
        python_bin=".venv/bin/python",
        require_cuda=True,
    )

    assert "cd /teamspace/studios/this_studio/pact" in command
    assert "PYBIN=.venv/bin/python" in command
    assert "LIGHTNING_RUNTIME_DIAGNOSTIC_JSON=" in command
    assert "torch.cuda.is_available()" in command
    assert "FATAL: --require-cuda requested" in command


def test_lightning_remote_runtime_preflight_rejects_cpu_only_studio() -> None:
    mod = _load_module()

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "ssh" and "lightning_remote_runtime_preflight" in cmd[-1]:
            payload = {
                "schema_version": 1,
                "tool": "lightning_remote_runtime_preflight",
                "python": "/remote/pact/.venv/bin/python",
                "torch_version": "2.11.0+cu130",
                "torch_cuda_version": "13.0",
                "torch_cuda_available": False,
                "torch_cuda_device_count": 0,
                "torch_cuda_device_names": [],
                "nvidia_smi": None,
            }
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout=mod.RUNTIME_DIAGNOSTIC_MARKER + json.dumps(payload) + "\n",
                stderr="FATAL: --require-cuda requested but torch.cuda.is_available() is not true\n",
            )
        raise AssertionError(f"unexpected command: {cmd}")

    with pytest.raises(mod.LightningRuntimePreflightError) as excinfo:
        mod.ensure_lightning_remote_runtime_ready(
            "lightning-pact",
            remote_pact="/remote/pact",
            python_bin=".venv/bin/python",
            require_cuda=True,
            runner=fake_run,
        )

    message = str(excinfo.value)
    assert "not valid for interactive CUDA score work" in message
    assert "cuda_available=False" in message
    assert excinfo.value.diagnostic["status"] == "fail"
    assert excinfo.value.diagnostic["runtime_policy_violations"] == [
        "torch.cuda.is_available() is not true on the reachable Lightning Studio runtime"
    ]


def test_lightning_ssh_diagnostic_records_public_key_metadata(tmp_path: Path) -> None:
    mod = _load_module()
    key = tmp_path / "lightning_rsa"
    pub = tmp_path / "lightning_rsa.pub"
    key.write_text("private placeholder\n")
    pub.write_text("public placeholder\n")

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["ssh", "-G"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "host lightning-pact\n"
                    "hostname ssh.lightning.ai\n"
                    "user s_example\n"
                    f"identityfile {key}\n"
                    "identitiesonly yes\n"
                    "stricthostkeychecking accept-new\n"
                ),
                stderr="",
            )
        if cmd[:2] == ["ssh-keygen", "-lf"]:
            assert cmd[2] == str(pub)
            return subprocess.CompletedProcess(cmd, 0, stdout="2048 SHA256:testfp test (RSA)\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    diagnostic = mod.lightning_ssh_diagnostic(
        "lightning-pact",
        connect=False,
        runner=fake_run,
    )

    assert diagnostic["status"] == "not_probed"
    assert diagnostic["resolved"]["user"] == "s_example"
    assert diagnostic["identity_files"][0]["expanded_identity_file"] == str(key.resolve())
    assert diagnostic["identity_files"][0]["public_key_file"] == str(pub.resolve())
    assert diagnostic["identity_files"][0]["public_key_fingerprint"] == "2048 SHA256:testfp test (RSA)"


def test_lightning_ssh_diagnostic_rejects_bare_provider_host() -> None:
    mod = _load_module()

    with pytest.raises(ValueError, match=r"bare ssh\.lightning\.ai"):
        mod.lightning_ssh_diagnostic("ssh.lightning.ai", connect=False)


def test_lightning_ssh_preflight_failure_points_to_public_key(tmp_path: Path) -> None:
    mod = _load_module()
    key = tmp_path / "lightning_rsa"
    pub = tmp_path / "lightning_rsa.pub"
    key.write_text("private placeholder\n")
    pub.write_text("public placeholder\n")

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["ssh", "-G"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "host lightning-pact\n"
                    "hostname ssh.lightning.ai\n"
                    "user s_example\n"
                    f"identityfile {key}\n"
                    "identitiesonly yes\n"
                    "stricthostkeychecking accept-new\n"
                ),
                stderr="",
            )
        if cmd[:2] == ["ssh-keygen", "-lf"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="2048 SHA256:testfp test (RSA)\n", stderr="")
        if cmd[0] == "ssh" and cmd[-1] == "true":
            return subprocess.CompletedProcess(
                cmd,
                255,
                stdout="",
                stderr="s_example@ssh.lightning.ai: Permission denied (publickey).\n",
            )
        raise AssertionError(f"unexpected command: {cmd}")

    with pytest.raises(mod.LightningSshPreflightError) as excinfo:
        mod.ensure_lightning_ssh_ready("lightning-pact", connect_timeout=7, runner=fake_run)

    message = str(excinfo.value)
    assert "Permission denied (publickey)" in message
    assert str(key.resolve()) in message
    assert str(pub.resolve()) in message
    assert "Add the selected *.pub key" in message
    assert "bare lightning CLI" in message
    assert excinfo.value.diagnostic["auth_probe"]["returncode"] == 255


def test_lightning_ssh_preflight_rejects_disabled_host_key_checking(tmp_path: Path) -> None:
    mod = _load_module()
    key = tmp_path / "lightning_rsa"
    pub = tmp_path / "lightning_rsa.pub"
    key.write_text("private placeholder\n")
    pub.write_text("public placeholder\n")

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["ssh", "-G"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "host lightning-pact\n"
                    "hostname ssh.lightning.ai\n"
                    "user s_example\n"
                    f"identityfile {key}\n"
                    "identitiesonly yes\n"
                    "stricthostkeychecking false\n"
                ),
                stderr="",
            )
        if cmd[:2] == ["ssh-keygen", "-lf"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="2048 SHA256:testfp test (RSA)\n", stderr="")
        if cmd[0] == "ssh" and cmd[-1] == "true":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    with pytest.raises(mod.LightningSshPreflightError) as excinfo:
        mod.ensure_lightning_ssh_ready("lightning-pact", runner=fake_run)

    assert "StrictHostKeyChecking is disabled" in str(excinfo.value)
    assert excinfo.value.diagnostic["status"] == "fail"


def test_no_install_remote_command_falls_back_when_venv_missing() -> None:
    mod = _load_module()
    command = mod._remote_python_command(
        "/remote/pact",
        "/remote/pact/.omx/state/manifest.json",
        "/remote/pact/.omx/state/env.json",
        requirements_mode="no-install",
        python_bin=None,
        require_cuda=False,
    )
    assert "if [ -x .venv/bin/python ]; then" in command
    assert "command -v python3 || command -v python" in command
    assert "'python_bin_requested': None" in command
    assert "'python_bin_requested': null" not in command
    assert "FATAL_RUNTIME_SECURITY_JSON" in command
    assert "2.6.2" in command
    assert "2.6.3" in command


def test_verify_only_remote_command_uses_requested_python_without_uv_sync() -> None:
    mod = _load_module()
    command = mod._remote_python_command(
        "/remote/pact",
        "/remote/pact/.omx/state/manifest.json",
        "/remote/pact/.omx/state/env.json",
        requirements_mode="verify-only",
        python_bin="/opt/conda/bin/python",
        require_cuda=True,
    )
    assert "PYBIN=/opt/conda/bin/python" in command
    assert 'test -x "$PYBIN"' in command
    assert "'python_bin_requested': '/opt/conda/bin/python'" in command
    assert "uv sync --locked --extra runtime" not in command
    assert "FATAL: --require-cuda requested" in command
    assert "lightning/_runtime" in command


def test_uv_sync_remote_command_forces_copy_link_mode_for_lightning_filesystems() -> None:
    mod = _load_module()
    command = mod._remote_python_command(
        "/remote/pact",
        "/remote/pact/.omx/state/manifest.json",
        "/remote/pact/.omx/state/env.json",
        requirements_mode="uv-sync",
        python_bin=None,
        require_cuda=False,
    )
    ensure_idx = command.index("scripts/ensure_remote_uv.sh --symlink-system")
    sync_idx = command.index('"$UV_BIN" sync --locked --extra runtime')
    assert ensure_idx < sync_idx
    assert "UV_LINK_MODE=${UV_LINK_MODE:-copy}" in command
    assert "FATAL: uv is required for --requirements-mode uv-sync" not in command
    assert "PYBIN=.venv/bin/python" in command
