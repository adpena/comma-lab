#!/usr/bin/env python3
"""Bootstrap the exact DALI wheel set from direct hash-pinned wheels.

This is the non-Lightning companion to the Lightning exact-eval bootstrap.
It deliberately avoids package indexes so Vast/Modal/debug runners cannot
silently drift to a different DALI build while producing promotion evidence.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.deploy.lightning.batch_jobs import (  # noqa: E402
    DALI_BOOTSTRAP_VERSION,
    DALI_BOOTSTRAP_WHEELS,
)


def _version_or_none(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _probe() -> dict[str, Any]:
    try:
        import nvidia.dali  # type: ignore[import-not-found]
        import nvidia.dali.fn as dali_fn  # type: ignore[import-not-found]
    except Exception as exc:
        return {
            "ok": False,
            "error": repr(exc),
            "installed_distributions": {
                "nvidia-dali-cuda120": _version_or_none("nvidia-dali-cuda120"),
                "nvidia-dali-cuda130": _version_or_none("nvidia-dali-cuda130"),
            },
        }
    return {
        "ok": True,
        "dali_version": getattr(nvidia.dali, "__version__", None),
        "nvidia_dali_fn_module": getattr(dali_fn, "__name__", None),
        "installed_distributions": {
            "nvidia-dali-cuda120": _version_or_none("nvidia-dali-cuda120"),
            "nvidia-dali-cuda130": _version_or_none("nvidia-dali-cuda130"),
        },
    }


def _violations(probe: dict[str, Any], *, selected_package: str, unexpected_package: str) -> list[str]:
    if not probe.get("ok"):
        return [f"import failed: {probe.get('error')}"]
    installed = probe.get("installed_distributions") or {}
    violations: list[str] = []
    if probe.get("dali_version") != DALI_BOOTSTRAP_VERSION:
        violations.append(
            f"nvidia.dali.__version__={probe.get('dali_version')!r}, "
            f"expected {DALI_BOOTSTRAP_VERSION!r}"
        )
    if installed.get(selected_package) != DALI_BOOTSTRAP_VERSION:
        violations.append(
            f"{selected_package}={installed.get(selected_package)!r}, "
            f"expected {DALI_BOOTSTRAP_VERSION!r}"
        )
    if installed.get(unexpected_package) is not None:
        violations.append(
            f"unexpected CUDA-family DALI package installed: "
            f"{unexpected_package}={installed.get(unexpected_package)!r}"
        )
    if probe.get("nvidia_dali_fn_module") != "nvidia.dali.fn":
        violations.append(f"nvidia_dali_fn_module={probe.get('nvidia_dali_fn_module')!r}")
    return violations


def _write_requirements(path: pathlib.Path, wheels: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{wheel['url']} --hash=sha256:{wheel['sha256']}" for wheel in wheels]
    path.write_text("\n".join(lines) + "\n")


def _probe_python_pip() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", "import pip"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _ensure_uv(payload: dict[str, Any], *, timeout: int) -> str | None:
    uv = shutil.which("uv")
    if uv:
        payload["installer_bootstrap_action"] = "uv_already_available"
        payload["installer_uv_path"] = uv
        return uv

    ensure_uv = REPO_ROOT / "scripts" / "ensure_remote_uv.sh"
    if not ensure_uv.is_file():
        payload["ensure_remote_uv_available"] = False
        return None

    cmd = ["bash", str(ensure_uv), "--symlink-system"]
    payload["ensure_remote_uv_available"] = True
    payload["ensure_remote_uv_command"] = cmd
    install = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    payload["ensure_remote_uv_returncode"] = install.returncode
    payload["ensure_remote_uv_stdout_tail"] = install.stdout[-2000:]
    payload["ensure_remote_uv_stderr_tail"] = install.stderr[-4000:]
    if install.returncode != 0:
        payload["ensure_remote_uv_failed"] = True
        return None

    stdout_paths = [line.strip() for line in install.stdout.splitlines() if line.strip()]
    uv_path = stdout_paths[-1] if stdout_paths else shutil.which("uv")
    if not uv_path or not os.access(uv_path, os.X_OK):
        payload["ensure_remote_uv_failed"] = True
        payload["ensure_remote_uv_error"] = f"uv path not executable after bootstrap: {uv_path!r}"
        return None
    payload["installer_bootstrap_action"] = "uv_bootstrapped_by_ensure_remote_uv"
    payload["installer_uv_path"] = uv_path
    return uv_path


def _ensure_pip(payload: dict[str, Any], *, timeout: int) -> None:
    probe = _probe_python_pip()
    payload["pip_probe_returncode"] = probe.returncode
    payload["pip_probe_stdout_tail"] = probe.stdout[-1000:]
    payload["pip_probe_stderr_tail"] = probe.stderr[-1000:]
    if probe.returncode == 0:
        payload["installer_bootstrap_action"] = "pip_already_available"
        return

    cmd = [sys.executable, "-m", "ensurepip", "--upgrade"]
    payload["guarded_ensurepip_command"] = cmd
    ensurepip = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    payload["guarded_ensurepip_returncode"] = ensurepip.returncode
    payload["guarded_ensurepip_stdout_tail"] = ensurepip.stdout[-2000:]
    payload["guarded_ensurepip_stderr_tail"] = ensurepip.stderr[-4000:]
    if ensurepip.returncode != 0:
        payload["installer_bootstrap_action"] = "guarded_ensurepip_failed"
        raise SystemExit(f"FATAL: guarded ensurepip failed with returncode={ensurepip.returncode}")
    payload["installer_bootstrap_action"] = "pip_bootstrapped_by_guarded_ensurepip"


def _install_command(requirements_path: pathlib.Path, payload: dict[str, Any], *, timeout: int) -> list[str]:
    uv = _ensure_uv(payload, timeout=timeout)
    if uv:
        return [
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--require-hashes",
            "--no-deps",
            "--only-binary",
            ":all:",
            "--strict",
            "-r",
            str(requirements_path),
        ]
    _ensure_pip(payload, timeout=timeout)
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--require-hashes",
        "--no-deps",
        "--only-binary",
        ":all:",
        "-r",
        str(requirements_path),
    ]


def bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise SystemExit(f"FATAL: torch import failed while selecting DALI package: {exc!r}") from exc

    cuda_version = getattr(torch.version, "cuda", None)
    cuda_major_s = str(cuda_version or "").split(".", 1)[0]
    if cuda_major_s not in {"12", "13"}:
        raise SystemExit(f"FATAL: unsupported torch CUDA version for DALI bootstrap: {cuda_version!r}")
    cuda_family = "cu130" if cuda_major_s == "13" else "cu120"
    selected_package = "nvidia-dali-cuda130" if cuda_family == "cu130" else "nvidia-dali-cuda120"
    unexpected_package = "nvidia-dali-cuda120" if cuda_family == "cu130" else "nvidia-dali-cuda130"
    py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    if py_tag not in DALI_BOOTSTRAP_WHEELS:
        raise SystemExit(f"FATAL: no hash-pinned DALI dependency wheels registered for {py_tag}")

    selected_wheels: list[dict[str, str]] = (
        DALI_BOOTSTRAP_WHEELS["common"]
        + DALI_BOOTSTRAP_WHEELS[py_tag]
        + DALI_BOOTSTRAP_WHEELS[cuda_family]
    )
    requirements_path = pathlib.Path(args.requirements_out)
    _write_requirements(requirements_path, selected_wheels)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "tool": "scripts/bootstrap_dali_hash_pinned.py",
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.executable,
        "required_dali_version": DALI_BOOTSTRAP_VERSION,
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": cuda_version,
        "cuda_family": cuda_family,
        "selected_package": selected_package,
        "selected_requirement": f"{selected_package}=={DALI_BOOTSTRAP_VERSION}",
        "unexpected_package": unexpected_package,
        "requirements_path": str(requirements_path),
        "selected_wheels": selected_wheels,
        "installed": False,
        "installer_bootstrap_action": None,
    }

    initial = _probe()
    payload["initial_probe"] = initial
    initial_violations = _violations(
        initial,
        selected_package=selected_package,
        unexpected_package=unexpected_package,
    )
    payload["initial_probe_violations"] = initial_violations
    if not initial_violations:
        payload["bootstrap_action"] = "already_exact"
        payload["installer_bootstrap_action"] = "not_needed_already_exact_dali"
    elif initial.get("ok") and not args.force:
        payload["bootstrap_action"] = "fail_wrong_preinstalled_dali"
        payload["installer_bootstrap_action"] = "blocked_wrong_preinstalled_dali"
        _write_payload(pathlib.Path(args.json_out), payload)
        raise SystemExit(
            "FATAL: preinstalled DALI is not the exact expected package/version: "
            + "; ".join(initial_violations)
        )
    else:
        payload["bootstrap_action"] = "install_hash_pinned_wheels"
        try:
            cmd = _install_command(requirements_path, payload, timeout=args.timeout)
        except SystemExit:
            _write_payload(pathlib.Path(args.json_out), payload)
            raise
        payload["install_command"] = cmd
        install = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=args.timeout)
        payload["install_returncode"] = install.returncode
        payload["install_stdout_tail"] = install.stdout[-4000:]
        payload["install_stderr_tail"] = install.stderr[-4000:]
        if install.returncode != 0:
            _write_payload(pathlib.Path(args.json_out), payload)
            raise SystemExit(f"FATAL: hash-pinned DALI install failed with returncode={install.returncode}")
        payload["installed"] = True

    final = _probe()
    payload["final_probe"] = final
    final_violations = _violations(
        final,
        selected_package=selected_package,
        unexpected_package=unexpected_package,
    )
    payload["final_probe_violations"] = final_violations
    _write_payload(pathlib.Path(args.json_out), payload)
    if final_violations:
        raise SystemExit("FATAL: DALI exact preflight failed: " + "; ".join(final_violations))
    return payload


def _write_payload(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, help="Path to write bootstrap provenance JSON.")
    parser.add_argument(
        "--requirements-out",
        default=None,
        help="Path to write direct-wheel hash requirements. Defaults beside --json-out.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow reinstalling after a wrong DALI package imports; default is fail-closed.",
    )
    parser.add_argument("--timeout", type=int, default=900, help="Install timeout in seconds.")
    args = parser.parse_args(argv)
    if args.requirements_out is None:
        args.requirements_out = str(pathlib.Path(args.json_out).with_suffix(".requirements.txt"))
    payload = bootstrap(args)
    print("DALI_HASH_PINNED_BOOTSTRAP_OK")
    print(json.dumps({
        "selected_package": payload["selected_package"],
        "required_dali_version": payload["required_dali_version"],
        "bootstrap_action": payload["bootstrap_action"],
        "json_out": args.json_out,
        "requirements_out": args.requirements_out,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
